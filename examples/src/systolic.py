# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""4x4 systolic array in Psyre with WS and OS execution modes.

Implements a 4x4 systolic array (Kung & Leiserson, 1979; Kung, 1982) processing
bf16 inputs and multiplying and accumulating in IEEE-754 fp32 (`fp32_mul` /
`fp32_add`) across pipelined 4x4 matrix products. Control uses hierarchical
mode automata (Colaço, Pagano & Pouzet, EMSOFT '05), distributed across the
processing elements with a minimal global orchestrator:
- Global phase orchestrator (`systolic_control`, `mode_ctrl`) sequencing global
  weight-loading, WS compute, and OS mode switching (`GlobalControl`).
- Per-PE local control automaton (`pe_ctrl` inside `systolic_pe`) cycling
  through `PEState` (`IDLE`, `WS_ACTIVE`, `OS_INIT`, `OS_ACC1..3`) with a
  single state register that also serves as the 1-bit East/South wavefront
  token (`state != PEState.IDLE`), keeping each PE to 4 registers total
  (`row_reg`, `col_reg`, `acc_reg`, and `pe_ctrl/pns`).

Supports non-accumulating direct multiplication storage into the accumulator for
initialization as well as a global single-cycle synchronous reset.
"""

from collections.abc import Callable, Sequence
import enum
import struct
from typing import Any, NamedTuple

# pylint: disable=g-importing-member,g-multiple-import
from psyre.automaton import (
    automaton,
    state,
    until,
)
from psyre.common import (
    buffer,
    counter,
    fby,
    ifthenelse,
    merge,
    psyre_comb,
    psyre_node,
    reset,
    validate_non_negative_int,
    when,
)
from psyre.static_stream import EMPTY_WORD, StaticWord, repeat
from psyre.stream import Stream
from psyre.trace import present_values

# pylint: enable=g-importing-member,g-multiple-import

ARRAY_SIZE = 4
DRAIN_CYCLES = 2 * (ARRAY_SIZE - 1)

Matrix = Sequence[Sequence[float | int]]
StreamRow = tuple[Stream, ...]
StreamGrid = tuple[tuple[Stream, ...], ...]


class SystolicMode(enum.Enum):
  """Top-level operating modes of the 4x4 systolic array."""

  WEIGHT_STATIONARY = enum.auto()
  OUTPUT_STATIONARY = enum.auto()


class _GlobalPhase(enum.Enum):
  """Internal states of the global phase orchestrator (`mode_ctrl`)."""

  WS_LOAD = enum.auto()
  WS_COMPUTE = enum.auto()
  OUTPUT_STATIONARY = enum.auto()


class PEState(enum.Enum):
  """Local control states of each systolic Processing Element (PE)."""

  IDLE = enum.auto()
  WS_ACTIVE = enum.auto()
  OS_INIT = enum.auto()
  OS_ACC1 = enum.auto()
  OS_ACC2 = enum.auto()
  OS_ACC3 = enum.auto()


class GlobalControl(NamedTuple):
  """Global orchestration streams broadcast by `systolic_control`.

  Attributes:
    mode: Active top-level `SystolicMode` stream.
    load_weights: Boolean stream asserting during the 4-cycle WS weight load.
    datapath_reset: Single-cycle boolean reset stream asserting on global phase
      entry (`WS_LOAD`, `WS_COMPUTE`, `OUTPUT_STATIONARY`).
  """

  mode: Stream
  load_weights: Stream
  datapath_reset: Stream


class PEOutputs(NamedTuple):
  """Output streams of a single systolic Processing Element (PE).

  Attributes:
    row_out: Registered bf16 horizontal stream passed East to `(i, j + 1)`.
    col_out: Registered fp32 vertical stream passed South to `(i + 1, j)`
      (carrying shifted weights/inputs or fp32 partial sums).
    ps_out: Current-cycle fp32 MAC output stream.
    acc: Current-cycle fp32 merged weight/accumulator stream (holding the
      stationary bf16 weight in fp32 during weight-stationary mode and the
      running fp32 accumulator during output-stationary mode).
    state: Local `PEState` stream produced by the PE's control automaton.
    valid: Local boolean stream indicating when this PE holds a valid output.
    wave_out: Registered 1-bit wavefront token (`state != PEState.IDLE`) passed
      East and South.
  """

  row_out: Stream
  col_out: Stream
  ps_out: Stream
  acc: Stream
  state: Stream
  valid: Stream
  wave_out: Stream


class SystolicOutputs(NamedTuple):
  """Complete output bundle of the 4x4 systolic array.

  Attributes:
    mode: Active top-level `SystolicMode` stream from `systolic_control`.
    load_weights: Global boolean stream asserting during WS weight loading.
    pe_states: 4x4 grid of local `PEState` streams from each PE automaton.
    pe_valid: 4x4 grid of local boolean valid streams from each PE automaton.
    col_valid: 4-tuple of boolean streams indicating when column `j` emits a
      completed matrix product element at the South boundary.
    output_valid: Boolean stream indicating when all 4 deskewed column outputs
      carry a completed matrix row `(C[i, 0], C[i, 1], C[i, 2], C[i, 3])`.
    col_outputs: 4-tuple of raw fp32 column output streams at the South boundary
      before deskewing.
    deskewed_outputs: 4-tuple of deskewed fp32 column streams gated by
      `output_valid` (`Absent` when invalid).
    accumulators: 4x4 grid of current-cycle fp32 PE merged weight/accumulator
      streams (holding stationary weights in weight-stationary mode and
      accumulated outputs in output-stationary mode).
  """

  mode: Stream
  load_weights: Stream
  pe_states: StreamGrid
  pe_valid: StreamGrid
  col_valid: StreamRow
  output_valid: Stream
  col_outputs: StreamRow
  deskewed_outputs: StreamRow
  accumulators: StreamGrid


def _quantize_recursive(
    value: Any,
    scalar_fn: Callable[[float], float],
    label: str,
) -> Any:
  """Applies scalar rounding `scalar_fn` recursively across nested sequences."""
  if isinstance(value, (str, bytes, bytearray)):
    raise TypeError(
        f'{label} expects a number or sequence of numbers, not '
        f'{type(value).__name__}.'
    )
  if isinstance(value, Sequence):
    return [_quantize_recursive(item, scalar_fn, label) for item in value]
  if isinstance(value, bool):
    raise TypeError(f'{label} expects a numeric value, not bool ({value!r}).')
  return scalar_fn(float(value))


def _round_fp32_scalar(val: float) -> float:
  return struct.unpack('<f', struct.pack('<f', val))[0]


def _round_bf16_scalar(val: float) -> float:
  bits = struct.unpack('<I', struct.pack('<f', val))[0]
  if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0:
    bf16_bits = (bits | 0x00400000) & 0xFFFF0000
  else:
    rounding_bias = 0x00007FFF + ((bits >> 16) & 1)
    bf16_bits = (bits + rounding_bias) & 0xFFFF0000
  return struct.unpack('<f', struct.pack('<I', bf16_bits))[0]


def to_fp32(value: Any) -> Any:
  """Rounds a numeric value or nested sequence of values to IEEE-754 fp32."""
  return _quantize_recursive(value, _round_fp32_scalar, 'to_fp32')


def to_bf16(value: Any) -> Any:
  """Rounds a numeric value or nested sequence of values to IEEE-754 bf16."""
  return _quantize_recursive(value, _round_bf16_scalar, 'to_bf16')


@psyre_comb(pure=True)
def fp32_mul(lhs: float, rhs: float) -> float:
  """Multiplies two floats with IEEE-754 fp32 rounding."""
  return struct.unpack('<f', struct.pack('<f', float(lhs) * float(rhs)))[0]


@psyre_comb(pure=True)
def fp32_add(lhs: float, rhs: float) -> float:
  """Adds two floats with IEEE-754 fp32 rounding."""
  return struct.unpack('<f', struct.pack('<f', float(lhs) + float(rhs)))[0]


def _build_grid(fn: Callable[[int, int], Stream]) -> StreamGrid:
  """Builds an ARRAY_SIZE x ARRAY_SIZE tuple grid by calling fn(i, j)."""
  return tuple(
      tuple(fn(i, j) for j in range(ARRAY_SIZE)) for i in range(ARRAY_SIZE)
  )


def _validate_grid_shape(grid: Sequence[Sequence[Any]], label: str) -> None:
  """Verifies `grid` is an `ARRAY_SIZE x ARRAY_SIZE` rectangular grid."""
  if len(grid) != ARRAY_SIZE:
    raise ValueError(f'{label} must have {ARRAY_SIZE} rows, got {len(grid)}.')
  for i, row in enumerate(grid):
    if len(row) != ARRAY_SIZE:
      raise ValueError(
          f'{label} row {i} must have {ARRAY_SIZE} columns, got {len(row)}.'
      )


def _validate_matrix(
    matrix: Matrix, label: str = 'matrix'
) -> list[list[float]]:
  """Validates a 4x4 matrix and quantizes its entries to bf16."""
  _validate_grid_shape(matrix, label)
  return to_bf16(matrix)


def reference_matmul_bf16_fp32(
    mat_a: Matrix, mat_b: Matrix
) -> list[list[float]]:
  """Computes exact reference 4x4 matrix product with bf16 inputs and fp32 MAC.

  Args:
    mat_a: Left-hand 4x4 input matrix.
    mat_b: Right-hand 4x4 input matrix.

  Returns:
    4x4 fp32 product matrix where `k = 0` initializes the accumulator directly
    with `bf16(A[i, 0]) * bf16(B[0, j])` and `k = 1..3` accumulate in fp32.
  """
  a_bf16 = _validate_matrix(mat_a, 'mat_a')
  b_bf16 = _validate_matrix(mat_b, 'mat_b')
  result: list[list[float]] = []
  for i in range(ARRAY_SIZE):
    row_out: list[float] = []
    for j in range(ARRAY_SIZE):
      acc = to_fp32(a_bf16[i][0] * b_bf16[0][j])
      for k in range(1, ARRAY_SIZE):
        prod = to_fp32(a_bf16[i][k] * b_bf16[k][j])
        acc = to_fp32(acc + prod)
      row_out.append(acc)
    result.append(row_out)
  return result


def delay_padded(
    input_stream: Stream,
    k: int,
    pad: Stream | None = None,
    *,
    tick: Stream | None = None,
    name: str = 'delay_padded',
) -> Stream:
  """Delays `input_stream` by `k` local ticks, emitting `pad` for the first `k`.

  Uses a bounded ring `buffer` gated by the resettable warmup schedule so the
  output is dense and well-clocked in both `SYNC` and `KAHN` semantics.

  Args:
    input_stream: Stream to delay.
    k: Non-negative integer tick delay.
    pad: Optional padding stream emitted during local ticks `0 .. k - 1`
      (defaults to `Stream(0.0)` when `None`).
    tick: Optional shared local tick counter stream (saturated at `>= k`).
      When `None`, instantiates a dedicated `counter(max_val=k)`.
    name: Debug name prefix for created nodes.

  Returns:
    A dense `Stream` equal to `pad` for `k` ticks and `input_stream` shifted by
    `k` ticks thereafter.

  Raises:
    ValueError: If `k < 0`.
  """
  validate_non_negative_int(k, 'Delay k')
  if k == 0:
    return input_stream
  pad_s = Stream(0.0) if pad is None else pad
  cnt = counter(max_val=k, name=f'{name}_cnt') if tick is None else tick
  warmup = cnt < k
  delayed = buffer(
      input_stream,
      Stream(True),
      ~warmup,
      size=k + 2,
      name=f'{name}_delay',
  )
  return merge(
      warmup,
      pad_s,
      delayed,
      name=name,
  )


def skew_streams(
    streams: Sequence[Stream],
    *,
    pad: Stream | None = None,
    tick: Stream | None = None,
    prefix: str = 'skew',
) -> StreamRow:
  """Skews 4 row or column streams by `0, 1, 2, 3` local ticks.

  All delayed lanes share a single saturated tick counter compared against
  different constant streams (`1`, `2`, `3`).

  Args:
    streams: Sequence of `ARRAY_SIZE` streams to skew.
    pad: Optional padding stream emitted during the initial skew ticks of each
      stream (defaults to `Stream(0.0)` when `None`).
    tick: Optional shared local tick counter stream (saturated at
      `>= ARRAY_SIZE - 1`). When `None`, instantiates one shared counter.
    prefix: Debug name prefix for the skewing nodes.

  Returns:
    4-tuple of skewed `Stream`s where index `idx` is delayed by `idx` ticks.

  Raises:
    ValueError: If `len(streams) != ARRAY_SIZE`.
  """
  if len(streams) != ARRAY_SIZE:
    raise ValueError(
        f'skew_streams expects {ARRAY_SIZE} streams, got {len(streams)}.'
    )
  cnt = (
      counter(max_val=ARRAY_SIZE - 1, name=f'{prefix}_cnt')
      if tick is None
      else tick
  )
  return tuple(
      delay_padded(
          streams[idx], idx, pad=pad, tick=cnt, name=f'{prefix}_{idx}'
      )
      for idx in range(ARRAY_SIZE)
  )


def deskew_streams(
    col_streams: Sequence[Stream],
    *,
    valid: Stream | None = None,
    tick: Stream | None = None,
    prefix: str = 'deskew',
) -> StreamRow:
  """Deskews 4 column streams by `3, 2, 1, 0` local ticks.

  Args:
    col_streams: Sequence of `ARRAY_SIZE` column output streams.
    valid: Optional boolean `Stream` gating the deskewed outputs via `when`
      (defaults to `Stream(True)` when `None`).
    tick: Optional shared local tick counter stream (saturated at
      `>= ARRAY_SIZE - 1`). When `None`, instantiates one shared counter.
    prefix: Debug name prefix for the deskewing nodes.

  Returns:
    4-tuple of deskewed `Stream`s aligned to the rightmost column (`j = 3`).

  Raises:
    ValueError: If `len(col_streams) != ARRAY_SIZE`.
  """
  if len(col_streams) != ARRAY_SIZE:
    raise ValueError(
        f'deskew_streams expects {ARRAY_SIZE} streams, got {len(col_streams)}.'
    )
  valid_s = Stream(True) if valid is None else valid
  cnt = (
      counter(max_val=ARRAY_SIZE - 1, name=f'{prefix}_cnt')
      if tick is None
      else tick
  )
  deskewed: list[Stream] = []
  for j in range(ARRAY_SIZE):
    shift = ARRAY_SIZE - 1 - j
    aligned = delay_padded(
        col_streams[j], shift, tick=cnt, name=f'{prefix}_{j}'
    )
    aligned = when(valid_s, aligned, name=f'{prefix}_valid_{j}')
    deskewed.append(aligned)
  return tuple(deskewed)


def systolic_control(
    *,
    initial_mode: SystolicMode = SystolicMode.WEIGHT_STATIONARY,
    switch_mode: Stream | None = None,
    reload_weights: Stream | None = None,
) -> GlobalControl:
  """Minimal global orchestrator sequencing WS load/compute and OS mode.

  Args:
    initial_mode: Initial top-level `SystolicMode`.
    switch_mode: Optional boolean `Stream` toggling between `WEIGHT_STATIONARY`
      and `OUTPUT_STATIONARY` at tick end (defaults to `Stream(False)`).
    reload_weights: Optional boolean `Stream` returning WS execution from
      compute back to the 4-cycle weight-load phase (defaults to
      `Stream(False)`).

  Returns:
    `GlobalControl(mode, load_weights, datapath_reset)` broadcast to all PEs.
  """
  switch_s = Stream(False) if switch_mode is None else switch_mode
  reload_s = Stream(False) if reload_weights is None else reload_weights
  init_phase = (
      _GlobalPhase.WS_LOAD
      if initial_mode == SystolicMode.WEIGHT_STATIONARY
      else _GlobalPhase.OUTPUT_STATIONARY
  )
  with automaton(init=init_phase, name='mode_ctrl') as top:
    with state(_GlobalPhase.WS_LOAD):
      load_done = Stream.once(ARRAY_SIZE - 1, name='load_done')
      until(switch_s, _GlobalPhase.OUTPUT_STATIONARY)
      until(load_done, _GlobalPhase.WS_COMPUTE)
      top.mode = SystolicMode.WEIGHT_STATIONARY
      top.load_weights = True
      top.datapath_reset = Stream.once(0, name='load_entry')
    with state(_GlobalPhase.WS_COMPUTE):
      until(switch_s, _GlobalPhase.OUTPUT_STATIONARY)
      until(reload_s, _GlobalPhase.WS_LOAD)
      top.mode = SystolicMode.WEIGHT_STATIONARY
      top.load_weights = False
      top.datapath_reset = Stream.once(0, name='comp_entry')
    with state(_GlobalPhase.OUTPUT_STATIONARY):
      until(switch_s, _GlobalPhase.WS_LOAD)
      top.mode = SystolicMode.OUTPUT_STATIONARY
      top.load_weights = False
      top.datapath_reset = Stream.once(0, name='os_entry')
  return GlobalControl(
      mode=top.mode,
      load_weights=top.load_weights,
      datapath_reset=top.datapath_reset,
  )


@psyre_node('systolic_pe')
def systolic_pe(
    row_in: Stream,
    col_in: Stream,
    *,
    mode: Stream | None = None,
    load_weights: Stream | None = None,
    datapath_reset: Stream | None = None,
    accumulate: Stream | None = None,
    wave_west: Stream | None = None,
    wave_north: Stream | None = None,
    is_top_row: bool = False,
) -> PEOutputs:
  """Instantiates one Processing Element (PE) with a 1-register local FSM.

  Each PE owns 4 registers total: `row_reg` (bf16 East), `col_reg` (fp32 South),
  `acc_reg` (merged stationary weight / fp32 accumulator), and the single state
  register of its local control automaton `pe_ctrl` (which also emits the
  registered 1-bit wavefront token `wave_out = state != PEState.IDLE`).

  Args:
    row_in: Horizontal West bf16 input stream.
    col_in: Vertical North fp32 input stream (carrying weights or partial sums
      in WS mode, or B-matrix inputs in OS mode).
    mode: Global `SystolicMode` stream from `systolic_control`.
    load_weights: Global boolean stream asserting during WS weight loading.
    datapath_reset: Global single-cycle phase-entry reset clearing `pe_ctrl`,
      `row_reg`, `col_reg`, and (in OS mode) `acc_reg`.
    accumulate: Boolean override stream; when `False`, forces direct storage of
      `row * col` into the accumulator instead of accumulating.
    wave_west: Incoming 1-bit wavefront token from West neighbor (or `True` on
      the West boundary).
    wave_north: Incoming 1-bit wavefront token from North neighbor (or `True` on
      the North boundary).
    is_top_row: Whether this PE sits in row 0 (no incoming partial sum from
      North in WS mode).

  Returns:
    `PEOutputs` carrying `row_out`, `col_out`, `ps_out`, `acc`, `state`,
    `valid`, and `wave_out`.
  """
  mode_s = Stream(SystolicMode.WEIGHT_STATIONARY) if mode is None else mode
  load_s = Stream(False) if load_weights is None else load_weights
  dp_rst_s = Stream(False) if datapath_reset is None else datapath_reset
  acc_s = Stream(True) if accumulate is None else accumulate
  ww_s = Stream(True) if wave_west is None else wave_west
  wn_s = Stream(True) if wave_north is None else wave_north

  is_ws = mode_s == SystolicMode.WEIGHT_STATIONARY
  is_os = ~is_ws
  is_ws_compute = is_ws & ~load_s
  wave_in = ww_s & wn_s
  ws_active = wave_in & is_ws_compute
  os_start = wave_in & is_os

  # 1-register local FSM (`reset=False` avoids emitting a `pnr` reset bit):
  # On the arrival tick (`IDLE` + `os_start`), this PE directly stores the `k=0`
  # product (no accumulation) and enters `OS_ACC1` on the *next* tick, skipping
  # `OS_INIT` until wrapping around from `OS_ACC3` for back-to-back tiles.
  with reset(dp_rst_s):
    with automaton(init=PEState.IDLE, name='pe_ctrl') as ctrl:
      with state(PEState.IDLE) as s:
        until(ws_active, PEState.WS_ACTIVE, reset=False)
        until(os_start, PEState.OS_ACC1, reset=False)
        ctrl.state = s
        ctrl.os_acc = False
      with state(PEState.WS_ACTIVE) as s:
        ctrl.state = s
        ctrl.os_acc = False
      with state(PEState.OS_INIT) as s:
        until(True, PEState.OS_ACC1, reset=False)
        ctrl.state = s
        ctrl.os_acc = False
      with state(PEState.OS_ACC1) as s:
        until(True, PEState.OS_ACC2, reset=False)
        ctrl.state = s
        ctrl.os_acc = True
      with state(PEState.OS_ACC2) as s:
        until(True, PEState.OS_ACC3, reset=False)
        ctrl.state = s
        ctrl.os_acc = True
      with state(PEState.OS_ACC3) as s:
        until(True, PEState.OS_INIT, reset=False)
        ctrl.state = s
        ctrl.os_acc = True

  # Reuses `pe_ctrl` state register as the registered 1-bit wavefront token
  # propagated East/South one cycle after this PE becomes active.
  wave_out = ctrl.state != PEState.IDLE
  can_accumulate = ifthenelse(is_ws, Stream(not is_top_row), ctrl.os_acc)
  pe_accumulate = can_accumulate & acc_s
  pe_valid = ifthenelse(is_ws, ws_active, ctrl.state == PEState.OS_ACC3)

  # Reset `acc_reg` only in OS mode so stationary weights shifted during
  # `WS_LOAD` survive the `WS_LOAD -> WS_COMPUTE` `datapath_reset` pulse.
  with reset(dp_rst_s & is_os):
    acc_prev = fby(0.0, lambda: acc_curr, name='acc_reg')

  with reset(dp_rst_s):
    row_out = fby(0.0, row_in, name='row_reg')
    mul_rhs = ifthenelse(is_ws, acc_prev, col_in, name='rhs_mux')
    # True IEEE-754 fp32 MAC rounding matching `reference_matmul_bf16_fp32`.
    prod = fp32_mul(row_in, mul_rhs)
    acc_base = ifthenelse(is_ws, col_in, acc_prev, name='base_mux')
    acc_sum = fp32_add(acc_base, prod)
    mac_out = ifthenelse(pe_accumulate, acc_sum, prod, name='acc_mux')
    ws_acc = ifthenelse(load_s, col_in, acc_prev, name='ws_acc_mux')
    ws_col = ifthenelse(load_s, col_in, mac_out, name='ws_col_mux')
    acc_curr = ifthenelse(is_ws, ws_acc, mac_out, name='mode_acc_mux')
    col_next = ifthenelse(is_ws, ws_col, col_in, name='mode_col_mux')
    col_out = fby(0.0, col_next, name='col_reg')

  return PEOutputs(
      row_out=row_out,
      col_out=col_out,
      ps_out=mac_out,
      acc=acc_curr,
      state=ctrl.state,
      valid=pe_valid,
      wave_out=wave_out,
  )


def systolic_array(
    row_inputs: Sequence[Stream],
    col_inputs: Sequence[Stream],
    *,
    initial_mode: SystolicMode = SystolicMode.WEIGHT_STATIONARY,
    switch_mode: Stream | None = None,
    reload_weights: Stream | None = None,
    accumulate: Stream | None = None,
    reset_stream: Stream | None = None,
    pe_fn: Callable[..., PEOutputs] | None = None,
) -> SystolicOutputs:
  """Builds the 4x4 systolic array with distributed per-PE control.

  Args:
    row_inputs: 4 West boundary bf16 input streams (skewed for active compute).
    col_inputs: 4 North boundary fp32 input streams (unskewed weight columns
      during WS weight loading, or skewed B-matrix column streams during
      `SystolicMode.OUTPUT_STATIONARY`).
    initial_mode: Initial top-level `SystolicMode`.
    switch_mode: Boolean `Stream` toggling top-level mode at tick end.
    reload_weights: Boolean `Stream` returning WS execution to weight loading.
    accumulate: Boolean `Stream` override; when `False`, forces direct storage
      of multiplication products into PE accumulators.
    reset_stream: Boolean `Stream` for global single-cycle synchronous reset of
      the global orchestrator and all 4x4 PEs.
    pe_fn: Optional PE constructor overriding `systolic_pe`.

  Returns:
    `SystolicOutputs` containing global mode/load_weights, per-PE states and
    valids, boundary valids, raw column outputs, deskewed outputs, and merged
    PE accumulators.

  Raises:
    ValueError: If `row_inputs` or `col_inputs` does not have length 4.
  """
  if len(row_inputs) != ARRAY_SIZE:
    raise ValueError(
        f'row_inputs must have length {ARRAY_SIZE}, got {len(row_inputs)}.'
    )
  if len(col_inputs) != ARRAY_SIZE:
    raise ValueError(
        f'col_inputs must have length {ARRAY_SIZE}, got {len(col_inputs)}.'
    )

  build_pe = systolic_pe if pe_fn is None else pe_fn
  acc_s = Stream(True) if accumulate is None else accumulate
  rst_s = Stream(False) if reset_stream is None else reset_stream
  with reset(rst_s):
    ctrl = systolic_control(
        initial_mode=initial_mode,
        switch_mode=switch_mode,
        reload_weights=reload_weights,
    )
    is_ws = ctrl.mode == SystolicMode.WEIGHT_STATIONARY

    pe_grid: list[list[PEOutputs]] = []
    for i in range(ARRAY_SIZE):
      pe_row: list[PEOutputs] = []
      for j in range(ARRAY_SIZE):
        row_in = row_inputs[i] if j == 0 else pe_row[j - 1].row_out
        col_in = col_inputs[j] if i == 0 else pe_grid[i - 1][j].col_out
        wave_west = Stream(True) if j == 0 else pe_row[j - 1].wave_out
        wave_north = Stream(True) if i == 0 else pe_grid[i - 1][j].wave_out
        pe_out = build_pe(
            row_in,
            col_in,
            mode=ctrl.mode,
            load_weights=ctrl.load_weights,
            datapath_reset=ctrl.datapath_reset,
            accumulate=acc_s,
            wave_west=wave_west,
            wave_north=wave_north,
            is_top_row=(i == 0),
            name=f'pe_{i}_{j}',
        )
        pe_row.append(pe_out)
      pe_grid.append(pe_row)

    accumulators = _build_grid(lambda i, j: pe_grid[i][j].acc)
    pe_states = _build_grid(lambda i, j: pe_grid[i][j].state)
    pe_valid = _build_grid(lambda i, j: pe_grid[i][j].valid)

    col_outputs: list[Stream] = []
    col_valid: list[Stream] = []
    for j in range(ARRAY_SIZE):
      ws_col_out = pe_grid[ARRAY_SIZE - 1][j].ps_out
      ws_col_v = pe_valid[ARRAY_SIZE - 1][j]
      os_col_out = Stream(0.0)
      os_col_v = pe_valid[0][j]
      for i in range(ARRAY_SIZE):
        os_col_out = ifthenelse(
            pe_valid[i][j],
            accumulators[i][j],
            os_col_out,
            name=f'os_readout_{i}_{j}',
        )
        if i > 0:
          os_col_v = os_col_v | pe_valid[i][j]
      col_outputs.append(
          ifthenelse(is_ws, ws_col_out, os_col_out, name=f'col_out_{j}')
      )
      col_valid.append(
          ifthenelse(is_ws, ws_col_v, os_col_v, name=f'col_valid_{j}')
      )

    output_valid = col_valid[ARRAY_SIZE - 1]
    deskewed = deskew_streams(
        col_outputs, valid=output_valid, prefix='deskew_out'
    )

  return SystolicOutputs(
      mode=ctrl.mode,
      load_weights=ctrl.load_weights,
      pe_states=pe_states,
      pe_valid=pe_valid,
      col_valid=tuple(col_valid),
      output_valid=output_valid,
      col_outputs=tuple(col_outputs),
      deskewed_outputs=deskewed,
      accumulators=accumulators,
  )


def build_ws_schedule(
    epochs: Sequence[tuple[Matrix, Sequence[Matrix]]],
    *,
    drain_cycles: int = DRAIN_CYCLES,
) -> tuple[list[StaticWord], list[StaticWord], StaticWord, int]:
  """Builds unskewed row/col `StaticWord`s and reload schedule for WS mode.

  Each epoch `(weight_matrix, activation_matrices)` consists of:
  1. `ARRAY_SIZE` (4) weight-loading cycles where `col_seqs[j]` shifts in
     `[W[3, j], W[2, j], W[1, j], W[0, j]]` from North to South while
     `row_seqs[i]` pads with `0.0`.
  2. `4 * len(activation_matrices) + drain_cycles` compute cycles where
     `row_seqs[k]` streams `A^{(m)}[0..3, k]` back-to-back for each matrix
     `m` followed by `drain_cycles` of `0.0`. If another epoch follows,
     `reload_seq` asserts `True` on the final drain cycle to transition the
     global orchestrator back to weight loading.

  Zero-padding and `False` stretches are encoded symbolically via `repeat(...)`
  so memory scales only with the matrix payloads, not `drain_cycles`.

  Args:
    epochs: Non-empty sequence of `(weight_matrix, activation_matrices)` pairs.
    drain_cycles: Pipeline drain cycles appended at the end of each epoch's
      compute phase (defaults to `DRAIN_CYCLES = 6`).

  Returns:
    Tuple `(row_seqs, col_seqs, reload_seq, horizon)` where `row_seqs` and
    `col_seqs` each contain 4 `StaticWord`s of length `horizon`, `reload_seq`
    is a boolean `StaticWord` of length `horizon`, and `horizon` is the total
    tick count.

  Raises:
    ValueError: If `epochs` is empty or any epoch has no activation matrices.
  """
  if not epochs:
    raise ValueError('epochs must be non-empty.')
  validate_non_negative_int(drain_cycles, 'drain_cycles')
  if len(epochs) > 1 and drain_cycles < 1:
    raise ValueError(
        'drain_cycles must be >= 1 when scheduling multiple WS epochs so the '
        'reload_weights pulse can trigger between epochs.'
    )
  row_seqs: list[StaticWord] = [EMPTY_WORD] * ARRAY_SIZE
  col_seqs: list[StaticWord] = [EMPTY_WORD] * ARRAY_SIZE
  reload_seq: StaticWord = EMPTY_WORD
  load_zeros = repeat(0.0, ARRAY_SIZE)
  drain_zeros = repeat(0.0, drain_cycles)

  for epoch_idx, (w_mat, a_mats) in enumerate(epochs):
    if not a_mats:
      raise ValueError(f'Epoch {epoch_idx} must have at least one A matrix.')
    w_bf16 = _validate_matrix(w_mat, f'epochs[{epoch_idx}].weight')
    a_bf16_list = [
        _validate_matrix(a_mat, f'epochs[{epoch_idx}].a[{m}]')
        for m, a_mat in enumerate(a_mats)
    ]

    # 1. Weight load phase (ARRAY_SIZE cycles, bottom row first).
    for i in range(ARRAY_SIZE):
      row_seqs[i] += load_zeros
    for j in range(ARRAY_SIZE):
      col_seqs[j] += [
          w_bf16[ARRAY_SIZE - 1 - step][j] for step in range(ARRAY_SIZE)
      ]

    # 2. Streaming compute phase (pipelined 4x4 matrices + drain_cycles).
    compute_cycles = ARRAY_SIZE * len(a_bf16_list)
    for k in range(ARRAY_SIZE):
      col_vals: list[float] = []
      for a_bf16 in a_bf16_list:
        for m_row in range(ARRAY_SIZE):
          col_vals.append(a_bf16[m_row][k])
      row_seqs[k] += col_vals
      row_seqs[k] += drain_zeros
    idle_cols = repeat(0.0, compute_cycles + drain_cycles)
    for j in range(ARRAY_SIZE):
      col_seqs[j] += idle_cols

    epoch_cycles = ARRAY_SIZE + compute_cycles + drain_cycles
    has_next_epoch = epoch_idx < len(epochs) - 1
    if has_next_epoch:
      reload_seq += repeat(False, epoch_cycles - 1) + True
    else:
      reload_seq += repeat(False, epoch_cycles)

  return row_seqs, col_seqs, reload_seq, reload_seq.length


def build_os_schedule(
    matrix_pairs: Sequence[tuple[Matrix, Matrix]],
    *,
    drain_cycles: int = DRAIN_CYCLES,
) -> tuple[list[StaticWord], list[StaticWord], int]:
  """Builds unskewed row and column `StaticWord`s for pipelined OS matmuls.

  Streams each 4x4 pair `(A^{(m)}, B^{(m)})` back-to-back every 4 cycles with
  zero bubbles between pairs, followed by `drain_cycles` of `0.0` (encoded via
  `repeat(0.0, drain_cycles)`) so all deskewed outputs complete.

  Args:
    matrix_pairs: Non-empty sequence of `(mat_a, mat_b)` 4x4 matrix pairs.
    drain_cycles: Pipeline drain cycles appended after the final pair.

  Returns:
    Tuple `(row_seqs, col_seqs, horizon)` of 4 row `StaticWord`s, 4 column
    `StaticWord`s, and total tick count `horizon`.

  Raises:
    ValueError: If `matrix_pairs` is empty.
  """
  if not matrix_pairs:
    raise ValueError('matrix_pairs must be non-empty.')
  validate_non_negative_int(drain_cycles, 'drain_cycles')
  raw_rows: list[list[float]] = [[] for _ in range(ARRAY_SIZE)]
  raw_cols: list[list[float]] = [[] for _ in range(ARRAY_SIZE)]

  for pair_idx, (mat_a, mat_b) in enumerate(matrix_pairs):
    a_bf16 = _validate_matrix(mat_a, f'matrix_pairs[{pair_idx}].a')
    b_bf16 = _validate_matrix(mat_b, f'matrix_pairs[{pair_idx}].b')
    for k in range(ARRAY_SIZE):
      for i in range(ARRAY_SIZE):
        raw_rows[i].append(a_bf16[i][k])
      for j in range(ARRAY_SIZE):
        raw_cols[j].append(b_bf16[k][j])

  drain_zeros = repeat(0.0, drain_cycles)
  row_seqs = [StaticWord(lane) + drain_zeros for lane in raw_rows]
  col_seqs = [StaticWord(lane) + drain_zeros for lane in raw_cols]
  return row_seqs, col_seqs, row_seqs[0].length


def extract_matrices(
    deskewed_traces: Sequence[Sequence[Any]],
) -> list[list[list[float]]]:
  """Extracts completed 4x4 fp32 matrices from 4 deskewed column traces.

  Args:
    deskewed_traces: Sequence of 4 post-run traces from
      `SystolicOutputs.deskewed_outputs`.

  Returns:
    List of 4x4 fp32 matrices `[C^{(0)}, C^{(1)}, ...]` in emission order.

  Raises:
    ValueError: If `deskewed_traces` does not have 4 columns, if columns have
      mismatched counts of present values, or if the present count is not a
      multiple of `ARRAY_SIZE`.
  """
  if len(deskewed_traces) != ARRAY_SIZE:
    raise ValueError(
        f'Expected {ARRAY_SIZE} column traces, got {len(deskewed_traces)}.'
    )
  cols = [
      [float(val) for val in present_values(trace)] for trace in deskewed_traces
  ]
  num_rows = len(cols[0])
  if any(len(col) != num_rows for col in cols):
    lengths = [len(col) for col in cols]
    raise ValueError(f'Mismatched present column lengths: {lengths}.')
  if num_rows % ARRAY_SIZE != 0:
    raise ValueError(
        f'Present row count {num_rows} is not a multiple of {ARRAY_SIZE}.'
    )
  num_matrices = num_rows // ARRAY_SIZE
  matrices: list[list[list[float]]] = []
  for m in range(num_matrices):
    mat: list[list[float]] = []
    for i in range(ARRAY_SIZE):
      row_idx = m * ARRAY_SIZE + i
      mat.append([cols[j][row_idx] for j in range(ARRAY_SIZE)])
    matrices.append(mat)
  return matrices


def extract_os_pe_matrices(
    acc_traces: Sequence[Sequence[Sequence[Any]]],
    pe_valid_traces: Sequence[Sequence[Sequence[Any]]],
    *,
    num_matrices: int | None = None,
) -> list[list[list[float]]]:
  """Extracts completed 4x4 fp32 matrices directly from OS PE accumulators.

  Samples each PE `(i, j)`'s accumulator trace on the ticks where
  `pe_valid_traces[i][j]` is `True`. Because the systolic wavefront reaches
  `PE (3, 3)` last (`d = 6`), top-left PEs (`d < 3`) may complete an extra
  drain tile while bottom-right PEs finish the final matrix; the number of
  fully completed 4x4 matrices defaults to the count completed by `PE (3, 3)`.

  Args:
    acc_traces: 4x4 grid of post-run PE accumulator traces.
    pe_valid_traces: 4x4 grid of post-run boolean `pe_valid` traces.
    num_matrices: Optional explicit number of matrices to extract. Defaults to
      the number of valid tiles completed by the final wavefront PE `(3, 3)`.

  Returns:
    List of 4x4 fp32 matrices `[C^{(0)}, C^{(1)}, ...]`.

  Raises:
    ValueError: If the grid is not `4x4`, `num_matrices` is negative, or any PE
      has fewer completed tiles than required.
  """
  _validate_grid_shape(acc_traces, 'acc_traces')
  _validate_grid_shape(pe_valid_traces, 'pe_valid_traces')
  if num_matrices is not None:
    validate_non_negative_int(num_matrices, 'num_matrices')
  per_pe_samples: list[list[list[float]]] = []
  for i in range(ARRAY_SIZE):
    row_samples: list[list[float]] = []
    for j in range(ARRAY_SIZE):
      samples = [
          float(acc_val)
          for acc_val, is_val in zip(
              acc_traces[i][j], pe_valid_traces[i][j], strict=True
          )
          if is_val is True  # pylint: disable=g-bool-id-comparison
      ]
      row_samples.append(samples)
    per_pe_samples.append(row_samples)

  total = (
      len(per_pe_samples[ARRAY_SIZE - 1][ARRAY_SIZE - 1])
      if num_matrices is None
      else num_matrices
  )
  for i in range(ARRAY_SIZE):
    for j in range(ARRAY_SIZE):
      if len(per_pe_samples[i][j]) < total:
        raise ValueError(
            f'PE ({i}, {j}) recorded {len(per_pe_samples[i][j])} valid '
            f'outputs, expected at least {total}.'
        )

  matrices: list[list[list[float]]] = []
  for m in range(total):
    mat: list[list[float]] = []
    for i in range(ARRAY_SIZE):
      mat.append([per_pe_samples[i][j][m] for j in range(ARRAY_SIZE)])
    matrices.append(mat)
  return matrices
