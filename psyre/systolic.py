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
"""4x4 systolic array in Psyre with weight-stationary and output-stationary modes.

Implements a 4x4 systolic array processing bf16 inputs and multiplying and
accumulating in fp32 across pipelined 4x4 matrix products. Control is
centralized as a two-level hierarchical automaton:
- Top-level mode automaton (`SystolicMode.WEIGHT_STATIONARY` vs.
  `SystolicMode.OUTPUT_STATIONARY`).
- Lower-level systolic control automata (`WeightStationaryState` with `LOAD`
  and `COMPUTE`, and `OutputStationaryState` with `INIT` and `ACCUMULATE`).

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
from psyre.common import validate_non_negative_int
from psyre.psyre import (
    Scheduler,
    Stream,
    counter,
    delay,
    fby,
    ifthenelse,
    merge,
    once_stream,
    present_values,
    psyre_node,
    reset,
    when,
    whennot,
)

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


class WeightStationaryState(enum.Enum):
  """Lower-level control states for weight-stationary execution."""

  LOAD = enum.auto()
  COMPUTE = enum.auto()


class OutputStationaryState(enum.Enum):
  """Lower-level control states for output-stationary execution."""

  INIT = enum.auto()
  ACCUMULATE = enum.auto()


class SystolicControl(NamedTuple):
  """Centralized control stream bundle produced by the hierarchical automaton.

  Attributes:
    mode: Active top-level `SystolicMode` stream.
    substate: Active lower-level state stream (`WeightStationaryState` or
      `OutputStationaryState`).
    load_weights: Boolean stream asserting when PEs shift and latch weights.
    datapath_reset: Boolean stream asserting on mode or phase entry to clear
      transient inter-PE pipeline registers.
    pe_accumulate: 4x4 grid of boolean streams; when `True`, PE `(i, j)`
      accumulates `acc_base + a * b` in fp32, and when `False`, stores `a * b`
      directly into the accumulator without accumulating.
    pe_valid: 4x4 grid of boolean streams indicating when PE `(i, j)` holds a
      valid multiplication/accumulation result.
    col_valid: 4-tuple of boolean streams indicating when column `j` emits a
      completed matrix product element at the South boundary.
    output_valid: Boolean stream indicating when all 4 deskewed column outputs
      carry a completed matrix row `(C[i, 0], C[i, 1], C[i, 2], C[i, 3])`.
  """

  mode: Stream
  substate: Stream
  load_weights: Stream
  datapath_reset: Stream
  pe_accumulate: StreamGrid
  pe_valid: StreamGrid
  col_valid: StreamRow
  output_valid: Stream


class PEOutputs(NamedTuple):
  """Output streams of a single systolic Processing Element (PE).

  Attributes:
    a_out: Registered bf16 horizontal stream passed East to `(i, j + 1)`.
    b_out: Registered fp32 vertical stream passed South to `(i + 1, j)`
      (carrying shifted weights/inputs or fp32 partial sums).
    ps_out: Current-cycle fp32 MAC output stream.
    acc: Current-cycle fp32 merged weight/accumulator stream (holding the
      stationary bf16 weight in fp32 during weight-stationary mode and the
      running fp32 accumulator during output-stationary mode).
  """

  a_out: Stream
  b_out: Stream
  ps_out: Stream
  acc: Stream


class SystolicOutputs(NamedTuple):
  """Complete output bundle of the 4x4 systolic array.

  Attributes:
    control: `SystolicControl` streams from the centralized hierarchical
      controller.
    col_outputs: 4-tuple of raw fp32 column output streams at the South boundary
      before deskewing.
    deskewed_outputs: 4-tuple of deskewed fp32 column streams gated by
      `control.output_valid` (`Absent` when invalid).
    accumulators: 4x4 grid of current-cycle fp32 PE merged weight/accumulator
      streams (holding stationary weights in weight-stationary mode and
      accumulated outputs in output-stationary mode).
  """

  control: SystolicControl
  col_outputs: StreamRow
  deskewed_outputs: StreamRow
  accumulators: StreamGrid


def to_fp32(value: Any) -> Any:
  """Rounds a numeric value or nested sequence of values to IEEE-754 fp32."""
  if isinstance(value, Sequence):
    return [to_fp32(item) for item in value]
  return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def to_bf16(value: Any) -> Any:
  """Rounds a numeric value or nested sequence of values to IEEE-754 bf16."""
  if isinstance(value, Sequence):
    return [to_bf16(item) for item in value]
  bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
  if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0:
    bf16_bits = (bits | 0x00400000) & 0xFFFF0000
  else:
    rounding_bias = 0x00007FFF + ((bits >> 16) & 1)
    bf16_bits = (bits + rounding_bias) & 0xFFFF0000
  return struct.unpack('<f', struct.pack('<I', bf16_bits))[0]


def _build_grid(fn: Callable[[int, int], Stream]) -> StreamGrid:
  """Builds an ARRAY_SIZE x ARRAY_SIZE tuple grid by calling fn(i, j)."""
  return tuple(
      tuple(fn(i, j) for j in range(ARRAY_SIZE)) for i in range(ARRAY_SIZE)
  )


def _validate_matrix(
    matrix: Matrix, label: str = 'matrix'
) -> list[list[float]]:
  """Validates a 4x4 matrix and quantizes its entries to bf16."""
  if len(matrix) != ARRAY_SIZE:
    raise ValueError(f'{label} must have {ARRAY_SIZE} rows, got {len(matrix)}.')
  for i, row in enumerate(matrix):
    if len(row) != ARRAY_SIZE:
      raise ValueError(
          f'{label} row {i} must have {ARRAY_SIZE} columns, got {len(row)}.'
      )
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
    pad: Stream = Stream(0.0),
    *,
    name: str = 'delay_padded',
) -> Stream:
  """Delays `input_stream` by `k` local ticks, emitting `pad` for the first `k`.

  Uses the synchronous `delay` primitive and replaces its initial `k` `Absent`
  ticks with `pad` so downstream hardware registers receive dense inputs.

  Args:
    input_stream: Stream to delay.
    k: Non-negative integer tick delay.
    pad: Stream emitted during local ticks `0 .. k - 1`.
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
  delayed = delay(input_stream, k, name=f'{name}_delay')
  warmup = counter(max_val=k, name=f'{name}_cnt') < k
  return merge(
      warmup,
      pad,
      whennot(warmup, delayed, name=f'{name}_after'),
      name=name,
  )


def skew_streams(
    streams: Sequence[Stream],
    *,
    pad: Stream = Stream(0.0),
    prefix: str = 'skew',
) -> StreamRow:
  """Skews 4 row or column streams by `0, 1, 2, 3` local ticks using `delay`.

  Args:
    streams: Sequence of `ARRAY_SIZE` streams to skew.
    pad: Padding stream emitted during the initial skew ticks of each stream.
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
  return tuple(
      delay_padded(streams[idx], idx, pad=pad, name=f'{prefix}_{idx}')
      for idx in range(ARRAY_SIZE)
  )


def deskew_streams(
    col_streams: Sequence[Stream],
    *,
    valid: Stream = Stream(True),
    prefix: str = 'deskew',
) -> StreamRow:
  """Deskews 4 column streams by `3, 2, 1, 0` local ticks using `delay`.

  Args:
    col_streams: Sequence of `ARRAY_SIZE` column output streams.
    valid: Boolean `Stream` gating the deskewed outputs via `when`.
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
  deskewed: list[Stream] = []
  for j in range(ARRAY_SIZE):
    shift = ARRAY_SIZE - 1 - j
    if shift == 0:
      aligned = col_streams[j]
    else:
      aligned = delay(col_streams[j], shift, name=f'{prefix}_{j}')
    aligned = when(valid, aligned, name=f'{prefix}_valid_{j}')
    deskewed.append(aligned)
  return tuple(deskewed)


def systolic_control(
    *,
    initial_mode: SystolicMode = SystolicMode.WEIGHT_STATIONARY,
    switch_mode: Stream = Stream(False),
    reload_weights: Stream = Stream(False),
    accumulate: Stream = Stream(True),
) -> SystolicControl:
  """Centralized hierarchical automaton generating 4x4 systolic control signals.

  The top-level automaton (`mode_ctrl`) selects between
  `SystolicMode.WEIGHT_STATIONARY` and `SystolicMode.OUTPUT_STATIONARY`.
  - Under `WEIGHT_STATIONARY`, the lower-level automaton (`ws_ctrl`) starts in
    `WeightStationaryState.LOAD` for `ARRAY_SIZE` (4) cycles to shift weights
    down the columns, then transitions to `WeightStationaryState.COMPUTE` to
    stream pipelined 4x4 activation matrices (and transitions back to `LOAD`
    whenever `reload_weights` is true). Row 0 sets `pe_accumulate = False` so
    the top PEs store `a * w` directly into their accumulators without
    accumulating, while rows 1..3 accumulate partial sums flowing South.
  - Under `OUTPUT_STATIONARY`, the lower-level automaton (`os_ctrl`) alternates
    between `OutputStationaryState.INIT` (1 cycle, setting `accumulate = False`
    so the product `a * b` is stored directly into the accumulator) and
    `OutputStationaryState.ACCUMULATE` (3 cycles, accumulating `acc + a * b` in
    fp32 and asserting completion on the 4th cycle of each tile). These phase
    signals are delayed across anti-diagonals `d = i + j` to match the systolic
    wavefront.

  Args:
    initial_mode: Initial top-level `SystolicMode`.
    switch_mode: Boolean `Stream` triggering a transition between
      `WEIGHT_STATIONARY` and `OUTPUT_STATIONARY` at the end of a tick.
    reload_weights: Boolean `Stream` triggering a transition from
      `WeightStationaryState.COMPUTE` back to `WeightStationaryState.LOAD`.
    accumulate: Boolean `Stream` override; when `False`, forces PEs to store the
      multiplication result directly into the accumulator instead of
      accumulating.

  Returns:
    A `SystolicControl` bundle of synchronized control streams.
  """
  Scheduler.require_sync_semantics('systolic_control')
  max_diag = 2 * (ARRAY_SIZE - 1)

  with automaton(init=initial_mode, name='mode_ctrl') as top:
    with state(SystolicMode.WEIGHT_STATIONARY) as ws_mode:
      until(switch_mode, SystolicMode.OUTPUT_STATIONARY)
      with automaton(init=WeightStationaryState.LOAD, name='ws_ctrl') as ws:
        with state(WeightStationaryState.LOAD) as load_state:
          load_done = once_stream(ARRAY_SIZE - 1, name='load_done')
          until(load_done, WeightStationaryState.COMPUTE)
          ws.substate = load_state
          ws.load_weights = True
          ws.datapath_reset = once_stream(0, name='load_entry')
          ws.row_acc = False
          ws.diag_valid = tuple(Stream(False) for _ in range(max_diag + 1))
        with state(WeightStationaryState.COMPUTE) as comp_state:
          until(reload_weights, WeightStationaryState.LOAD)
          comp_tick = counter(max_val=max_diag, name='comp_tick')
          ws.substate = comp_state
          ws.load_weights = False
          ws.datapath_reset = once_stream(0, name='comp_entry')
          ws.row_acc = accumulate
          ws.diag_valid = tuple(comp_tick >= d for d in range(max_diag + 1))

      ws_col_valid = tuple(
          ws.diag_valid[ARRAY_SIZE - 1 + j] for j in range(ARRAY_SIZE)
      )
      top.mode = ws_mode
      top.substate = ws.substate
      top.load_weights = ws.load_weights
      top.datapath_reset = ws.datapath_reset
      top.row0_acc = tuple(Stream(False) for _ in range(ARRAY_SIZE))
      top.rows_diag_acc = tuple(ws.row_acc for _ in range(max_diag))
      top.diag_valid = ws.diag_valid
      top.col_valid = ws_col_valid
      top.output_valid = ws_col_valid[ARRAY_SIZE - 1]

    with state(SystolicMode.OUTPUT_STATIONARY) as os_mode:
      until(switch_mode, SystolicMode.WEIGHT_STATIONARY)
      os_entry = once_stream(0, name='os_entry')
      with automaton(
          init=OutputStationaryState.INIT, name='os_ctrl'
      ) as os_fsm:
        with state(OutputStationaryState.INIT) as init_state:
          until(True, OutputStationaryState.ACCUMULATE)
          os_fsm.substate = init_state
          os_fsm.base_accumulate = False
          os_fsm.base_done = False
        with state(OutputStationaryState.ACCUMULATE) as acc_state:
          tile_last = once_stream(ARRAY_SIZE - 2, name='tile_last')
          until(tile_last, OutputStationaryState.INIT)
          os_fsm.substate = acc_state
          os_fsm.base_accumulate = True
          os_fsm.base_done = tile_last

      diag_acc: list[Stream] = [os_fsm.base_accumulate & accumulate]
      diag_done: list[Stream] = [os_fsm.base_done]
      for d in range(1, max_diag + 1):
        diag_acc.append(fby(False, diag_acc[d - 1], name=f'os_acc_d{d}'))
        diag_done.append(fby(False, diag_done[d - 1], name=f'os_done_d{d}'))

      os_col_valid: list[Stream] = []
      for j in range(ARRAY_SIZE):
        col_v = diag_done[j]
        for i in range(1, ARRAY_SIZE):
          col_v = col_v | diag_done[i + j]
        os_col_valid.append(col_v)

      top.mode = os_mode
      top.substate = os_fsm.substate
      top.load_weights = Stream(False)
      top.datapath_reset = os_entry
      top.row0_acc = tuple(diag_acc[j] for j in range(ARRAY_SIZE))
      top.rows_diag_acc = tuple(diag_acc[d] for d in range(1, max_diag + 1))
      top.diag_valid = tuple(diag_done)
      top.col_valid = tuple(os_col_valid)
      top.output_valid = os_col_valid[ARRAY_SIZE - 1]

  pe_accumulate = _build_grid(
      lambda i, j: (top.row0_acc[j] if i == 0 else top.rows_diag_acc[i + j - 1])
  )
  pe_valid = _build_grid(lambda i, j: top.diag_valid[i + j])

  return SystolicControl(
      mode=top.mode,
      substate=top.substate,
      load_weights=top.load_weights,
      datapath_reset=top.datapath_reset,
      pe_accumulate=pe_accumulate,
      pe_valid=pe_valid,
      col_valid=top.col_valid,
      output_valid=top.output_valid,
  )


@psyre_node('systolic_pe')
def systolic_pe(
    a_in: Stream,
    b_in: Stream,
    *,
    is_weight_stationary: Stream,
    load_weights: Stream,
    accumulate: Stream,
    datapath_reset: Stream = Stream(False),
) -> PEOutputs:
  """Instantiates one Processing Element (PE) of the systolic array.

  Uses a single merged fp32 register (`acc_reg`) for both stationary weights
  (in weight-stationary mode, representing bf16 weights without loss of
  precision) and stationary output accumulation (in output-stationary mode), and
  routes both vertical inputs and partial sums through the fp32 `b_in` stream:
  - In weight-stationary mode (`is_weight_stationary=True`), `acc_reg` latches
    `b_in` when `load_weights` is `True` and holds the weight when
    `load_weights` is `False`. During compute, `a_in` multiplies the stationary
    weight in `acc_reg`, combines with the incoming fp32 partial sum on `b_in`
    when `accumulate` is `True` (or bypasses addition when `accumulate` is
    `False`), and shifts the partial sum South through `b_reg`.
  - In output-stationary mode (`is_weight_stationary=False`), `a_in` multiplies
    `b_in` in fp32 and updates `acc_reg` with `acc_prev + prod` when
    `accumulate` is `True` or `prod` directly when `accumulate` is `False`.

  Args:
    a_in: Horizontal West bf16 input stream.
    b_in: Vertical North fp32 input stream (carrying weights or partial sums in
      WS mode, or B-matrix inputs in OS mode).
    is_weight_stationary: Boolean stream (`True` in WS mode, `False` in OS).
    load_weights: Boolean stream (`True` when latching weights in WS mode).
    accumulate: Boolean stream (`True` to accumulate, `False` to store product
      directly into the accumulator).
    datapath_reset: Boolean reset stream clearing pipeline registers (`a_out`,
      `b_out`, and `acc_prev` in OS mode) on mode/phase transitions.

  Returns:
    `PEOutputs` carrying `a_out`, `b_out`, `ps_out`, and `acc`.
  """
  with reset(datapath_reset & ~is_weight_stationary):
    acc_prev = fby(0.0, lambda: acc_curr, name='acc_reg')

  with reset(datapath_reset):
    a_out = fby(0.0, a_in, name='a_reg')
    mul_rhs = ifthenelse(is_weight_stationary, acc_prev, b_in, name='rhs_mux')
    prod = a_in * mul_rhs
    acc_base = ifthenelse(is_weight_stationary, b_in, acc_prev, name='base_mux')
    acc_sum = acc_base + prod
    mac_out = ifthenelse(accumulate, acc_sum, prod, name='acc_mux')
    ws_acc = ifthenelse(load_weights, b_in, acc_prev, name='ws_acc_mux')
    ws_b = ifthenelse(load_weights, b_in, mac_out, name='ws_b_mux')
    acc_curr = ifthenelse(
        is_weight_stationary, ws_acc, mac_out, name='mode_acc_mux'
    )
    b_next = ifthenelse(is_weight_stationary, ws_b, b_in, name='mode_b_mux')
    b_out = fby(0.0, b_next, name='b_reg')

  return PEOutputs(
      a_out=a_out,
      b_out=b_out,
      ps_out=mac_out,
      acc=acc_curr,
  )


def systolic_array(
    row_inputs: Sequence[Stream],
    col_inputs: Sequence[Stream],
    *,
    initial_mode: SystolicMode = SystolicMode.WEIGHT_STATIONARY,
    switch_mode: Stream = Stream(False),
    reload_weights: Stream = Stream(False),
    accumulate: Stream = Stream(True),
    reset_stream: Stream = Stream(False),
) -> SystolicOutputs:
  """Builds the 4x4 systolic array with centralized hierarchical control.

  Args:
    row_inputs: 4 West boundary bf16 input streams (skewed for active compute).
    col_inputs: 4 North boundary fp32 input streams (unskewed weight columns
      during `WeightStationaryState.LOAD`, or skewed B-matrix column streams
      during `SystolicMode.OUTPUT_STATIONARY`).
    initial_mode: Initial top-level `SystolicMode`.
    switch_mode: Boolean `Stream` toggling top-level mode at tick end.
    reload_weights: Boolean `Stream` returning WS control to `LOAD`.
    accumulate: Boolean `Stream` override; when `False`, forces direct storage
      of multiplication products into PE accumulators.
    reset_stream: Boolean `Stream` for global single-cycle synchronous reset of
      the entire hierarchical controller and all 4x4 PEs.

  Returns:
    `SystolicOutputs` containing control, raw column outputs, deskewed outputs,
    and the 4x4 merged weight/accumulator grid.

  Raises:
    ValueError: If `row_inputs` or `col_inputs` does not have length 4.
  """
  Scheduler.require_sync_semantics('systolic_array')
  if len(row_inputs) != ARRAY_SIZE:
    raise ValueError(
        f'row_inputs must have length {ARRAY_SIZE}, got {len(row_inputs)}.'
    )
  if len(col_inputs) != ARRAY_SIZE:
    raise ValueError(
        f'col_inputs must have length {ARRAY_SIZE}, got {len(col_inputs)}.'
    )

  with reset(reset_stream):
    ctrl = systolic_control(
        initial_mode=initial_mode,
        switch_mode=switch_mode,
        reload_weights=reload_weights,
        accumulate=accumulate,
    )
    is_ws = ctrl.mode == SystolicMode.WEIGHT_STATIONARY

    pe_grid: list[list[PEOutputs]] = []
    for i in range(ARRAY_SIZE):
      pe_row: list[PEOutputs] = []
      for j in range(ARRAY_SIZE):
        a_in = row_inputs[i] if j == 0 else pe_row[j - 1].a_out
        b_in = col_inputs[j] if i == 0 else pe_grid[i - 1][j].b_out
        pe_out = systolic_pe(
            a_in,
            b_in,
            is_weight_stationary=is_ws,
            load_weights=ctrl.load_weights,
            accumulate=ctrl.pe_accumulate[i][j],
            datapath_reset=ctrl.datapath_reset,
            name=f'pe_{i}_{j}',
        )
        pe_row.append(pe_out)
      pe_grid.append(pe_row)

    accumulators = _build_grid(lambda i, j: pe_grid[i][j].acc)

    col_outputs: list[Stream] = []
    for j in range(ARRAY_SIZE):
      ws_col_out = pe_grid[ARRAY_SIZE - 1][j].ps_out
      os_col_out = Stream(0.0)
      for i in range(ARRAY_SIZE):
        os_col_out = ifthenelse(
            ctrl.pe_valid[i][j],
            accumulators[i][j],
            os_col_out,
            name=f'os_readout_{i}_{j}',
        )
      col_outputs.append(
          ifthenelse(is_ws, ws_col_out, os_col_out, name=f'col_out_{j}')
      )

    with reset(ctrl.datapath_reset):
      deskewed = deskew_streams(
          col_outputs, valid=ctrl.output_valid, prefix='deskew_out'
      )

  return SystolicOutputs(
      control=ctrl,
      col_outputs=tuple(col_outputs),
      deskewed_outputs=deskewed,
      accumulators=accumulators,
  )


def build_ws_schedule(
    epochs: Sequence[tuple[Matrix, Sequence[Matrix]]],
    *,
    drain_cycles: int = DRAIN_CYCLES,
) -> tuple[list[list[float]], list[list[float]], list[bool], int]:
  """Builds unskewed row/col sequences and reload_weights schedule for WS mode.

  Each epoch `(weight_matrix, activation_matrices)` consists of:
  1. `ARRAY_SIZE` (4) weight-loading cycles where `col_seqs[j]` shifts in
     `[W[3, j], W[2, j], W[1, j], W[0, j]]` from North to South while
     `row_seqs[i]` pads with `0.0`.
  2. `4 * len(activation_matrices) + drain_cycles` compute cycles where
     `row_seqs[k]` streams `A^{(m)}[0..3, k]` back-to-back for each matrix
     `m` followed by `drain_cycles` of `0.0`. If another epoch follows,
     `reload_seq` asserts `True` on the final drain cycle to transition the
     lower-level automaton back to `WeightStationaryState.LOAD`.

  Args:
    epochs: Non-empty sequence of `(weight_matrix, activation_matrices)` pairs.
    drain_cycles: Pipeline drain cycles appended at the end of each epoch's
      compute phase (defaults to `DRAIN_CYCLES = 6`).

  Returns:
    Tuple `(row_seqs, col_seqs, reload_seq, horizon)` where `row_seqs` and
    `col_seqs` each contain 4 lists of floats of length `horizon`, `reload_seq`
    is a boolean list of length `horizon`, and `horizon` is the total tick
    count.

  Raises:
    ValueError: If `epochs` is empty or any epoch has no activation matrices.
  """
  if not epochs:
    raise ValueError('epochs must be non-empty.')
  row_seqs: list[list[float]] = [[] for _ in range(ARRAY_SIZE)]
  col_seqs: list[list[float]] = [[] for _ in range(ARRAY_SIZE)]
  reload_seq: list[bool] = []

  for epoch_idx, (w_mat, a_mats) in enumerate(epochs):
    if not a_mats:
      raise ValueError(f'Epoch {epoch_idx} must have at least one A matrix.')
    w_bf16 = _validate_matrix(w_mat, f'epochs[{epoch_idx}].weight')
    a_bf16_list = [
        _validate_matrix(a_mat, f'epochs[{epoch_idx}].a[{m}]')
        for m, a_mat in enumerate(a_mats)
    ]

    # 1. Weight load phase (ARRAY_SIZE cycles, bottom row first).
    for step in range(ARRAY_SIZE):
      src_row = ARRAY_SIZE - 1 - step
      for i in range(ARRAY_SIZE):
        row_seqs[i].append(0.0)
      for j in range(ARRAY_SIZE):
        col_seqs[j].append(w_bf16[src_row][j])
      reload_seq.append(False)

    # 2. Streaming compute phase (pipelined 4x4 matrices + drain_cycles).
    for a_bf16 in a_bf16_list:
      for m_row in range(ARRAY_SIZE):
        for k in range(ARRAY_SIZE):
          row_seqs[k].append(a_bf16[m_row][k])
        for j in range(ARRAY_SIZE):
          col_seqs[j].append(0.0)
        reload_seq.append(False)

    for d_step in range(drain_cycles):
      for i in range(ARRAY_SIZE):
        row_seqs[i].append(0.0)
      for j in range(ARRAY_SIZE):
        col_seqs[j].append(0.0)
      is_last_drain = d_step == drain_cycles - 1
      has_next_epoch = epoch_idx < len(epochs) - 1
      reload_seq.append(is_last_drain and has_next_epoch)

  return row_seqs, col_seqs, reload_seq, len(reload_seq)


def build_os_schedule(
    matrix_pairs: Sequence[tuple[Matrix, Matrix]],
    *,
    drain_cycles: int = DRAIN_CYCLES,
) -> tuple[list[list[float]], list[list[float]], int]:
  """Builds unskewed row and column sequences for pipelined OS matrix products.

  Streams each 4x4 pair `(A^{(m)}, B^{(m)})` back-to-back every 4 cycles with
  zero bubbles between pairs, followed by `drain_cycles` of `0.0` so all
  deskewed outputs complete.

  Args:
    matrix_pairs: Non-empty sequence of `(mat_a, mat_b)` 4x4 matrix pairs.
    drain_cycles: Pipeline drain cycles appended after the final pair.

  Returns:
    Tuple `(row_seqs, col_seqs, horizon)` of 4 row sequences, 4 column
    sequences, and total tick count `horizon`.

  Raises:
    ValueError: If `matrix_pairs` is empty.
  """
  if not matrix_pairs:
    raise ValueError('matrix_pairs must be non-empty.')
  row_seqs: list[list[float]] = [[] for _ in range(ARRAY_SIZE)]
  col_seqs: list[list[float]] = [[] for _ in range(ARRAY_SIZE)]

  for pair_idx, (mat_a, mat_b) in enumerate(matrix_pairs):
    a_bf16 = _validate_matrix(mat_a, f'matrix_pairs[{pair_idx}].a')
    b_bf16 = _validate_matrix(mat_b, f'matrix_pairs[{pair_idx}].b')
    for k in range(ARRAY_SIZE):
      for i in range(ARRAY_SIZE):
        row_seqs[i].append(a_bf16[i][k])
      for j in range(ARRAY_SIZE):
        col_seqs[j].append(b_bf16[k][j])

  for _ in range(drain_cycles):
    for i in range(ARRAY_SIZE):
      row_seqs[i].append(0.0)
    for j in range(ARRAY_SIZE):
      col_seqs[j].append(0.0)

  return row_seqs, col_seqs, len(row_seqs[0])


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
    ValueError: If any PE has fewer completed tiles than required.
  """
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
