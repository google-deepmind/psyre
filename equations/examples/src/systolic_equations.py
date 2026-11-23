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
"""4x4 systolic array in Psyre via `psyre_equations` (experimental, WIP).

Note: Everything under `psyre/equations/` is experimental and a work in
progress; APIs and internals may change without notice. Replicates
`examples/src/systolic.py` without explicit `lambda:` thunks, relying on
`with psyre_equations` and automatic stream naming (see Kung & Leiserson, 1979;
Kung, 1982; Colaço, Pagano & Pouzet, EMSOFT '05).
"""

from collections.abc import Sequence

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
    psyre_node,
    reset,
    validate_non_negative_int,
    when,
)
from equations import psyre_equations
from examples.src.systolic import (
    ARRAY_SIZE,
    DRAIN_CYCLES,
    GlobalControl,
    Matrix,
    PEOutputs,
    PEState,
    StreamGrid,
    StreamRow,
    SystolicMode,
    SystolicOutputs,
    _GlobalPhase,
    _build_grid,
    build_os_schedule,
    build_ws_schedule,
    extract_matrices,
    extract_os_pe_matrices,
    fp32_add,
    fp32_mul,
    reference_matmul_bf16_fp32,
    to_bf16,
    to_fp32,
)
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import

# Re-export shared constants, types, schedules, and trace extractors.
__all__ = (
    'ARRAY_SIZE',
    'DRAIN_CYCLES',
    'GlobalControl',
    'Matrix',
    'PEOutputs',
    'PEState',
    'StreamGrid',
    'StreamRow',
    'SystolicMode',
    'SystolicOutputs',
    'build_os_schedule',
    'build_ws_schedule',
    'delay_padded',
    'deskew_streams',
    'extract_matrices',
    'extract_os_pe_matrices',
    'fp32_add',
    'fp32_mul',
    'reference_matmul_bf16_fp32',
    'skew_streams',
    'systolic_array',
    'systolic_control',
    'systolic_pe',
    'to_bf16',
    'to_fp32',
)


def delay_padded(
    input_stream: Stream,
    k: int,
    pad: Stream | None = None,
    *,
    tick: Stream | None = None,
    name: str = 'delay_padded',
) -> Stream:
  """Delays `input_stream` by `k` local ticks, emitting `pad` for the first `k`."""
  validate_non_negative_int(k, 'Delay k')
  if k == 0:
    return input_stream
  pad_s = Stream(0.0) if pad is None else pad
  with psyre_equations() as eq:

    class _:
      cnt = counter(max_val=k, name=f'{name}_cnt') if tick is None else tick
      warmup = cnt < k
      delayed = buffer(
          input_stream,
          Stream(True),
          ~warmup,
          size=k + 2,
          name=f'{name}_delay',
      )
      __output__ = merge(
          warmup,
          pad_s,
          delayed,
          name=name,
      )

  return eq.output


def skew_streams(
    streams: Sequence[Stream],
    *,
    pad: Stream | None = None,
    tick: Stream | None = None,
    prefix: str = 'skew',
) -> StreamRow:
  """Skews 4 row or column streams by `0, 1, 2, 3` local ticks."""
  if len(streams) != ARRAY_SIZE:
    raise ValueError(
        f'skew_streams expects {ARRAY_SIZE} streams, got {len(streams)}.'
    )
  with psyre_equations() as eq:

    class _:
      """Skews each lane by its index using a shared counter."""
      cnt = (
          counter(max_val=ARRAY_SIZE - 1, name=f'{prefix}_cnt')
          if tick is None
          else tick
      )
      out_lanes = []
      for idx in range(ARRAY_SIZE):
        out_lanes.append(
            delay_padded(
                streams[idx], idx, pad=pad, tick=cnt, name=f'{prefix}_{idx}'
            )
        )
      __output__ = tuple(out_lanes)

  return eq.output


def deskew_streams(
    col_streams: Sequence[Stream],
    *,
    valid: Stream | None = None,
    tick: Stream | None = None,
    prefix: str = 'deskew',
) -> StreamRow:
  """Deskews 4 column streams by `3, 2, 1, 0` local ticks."""
  if len(col_streams) != ARRAY_SIZE:
    raise ValueError(
        f'deskew_streams expects {ARRAY_SIZE} streams, got {len(col_streams)}.'
    )
  valid_s = Stream(True) if valid is None else valid
  with psyre_equations() as eq:

    class _:
      """Deskews each column stream and gates with `valid`."""
      cnt = (
          counter(max_val=ARRAY_SIZE - 1, name=f'{prefix}_cnt')
          if tick is None
          else tick
      )
      deskewed = []
      for j in range(ARRAY_SIZE):
        shift = ARRAY_SIZE - 1 - j
        aligned = delay_padded(
            col_streams[j], shift, tick=cnt, name=f'{prefix}_{j}'
        )
        aligned = when(valid_s, aligned, name=f'{prefix}_valid_{j}')
        deskewed.append(aligned)
      __output__ = tuple(deskewed)

  return eq.output


def systolic_control(
    *,
    initial_mode: SystolicMode = SystolicMode.WEIGHT_STATIONARY,
    switch_mode: Stream | None = None,
    reload_weights: Stream | None = None,
) -> GlobalControl:
  """Minimal global orchestrator sequencing WS load/compute and OS mode."""
  switch_s = Stream(False) if switch_mode is None else switch_mode
  reload_s = Stream(False) if reload_weights is None else reload_weights
  init_phase = (
      _GlobalPhase.WS_LOAD
      if initial_mode == SystolicMode.WEIGHT_STATIONARY
      else _GlobalPhase.OUTPUT_STATIONARY
  )
  with psyre_equations() as eq:

    class _:
      """Global phase orchestrator automaton equations."""
      with automaton(init=init_phase, name='mode_ctrl') as top:
        with state(_GlobalPhase.WS_LOAD):
          load_done = Stream.once(ARRAY_SIZE - 1)
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
      __output__ = GlobalControl(
          mode=top.mode,
          load_weights=top.load_weights,
          datapath_reset=top.datapath_reset,
      )

  return eq.output


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
  """Instantiates one Processing Element (PE) with a 1-register local FSM."""
  mode_s = Stream(SystolicMode.WEIGHT_STATIONARY) if mode is None else mode
  load_s = Stream(False) if load_weights is None else load_weights
  dp_rst_s = Stream(False) if datapath_reset is None else datapath_reset
  acc_s = Stream(True) if accumulate is None else accumulate
  ww_s = Stream(True) if wave_west is None else wave_west
  wn_s = Stream(True) if wave_north is None else wave_north
  with psyre_equations() as eq:

    class _:
      """PE local FSM and MAC datapath equations."""
      is_ws = mode_s == SystolicMode.WEIGHT_STATIONARY
      is_os = ~is_ws
      is_ws_compute = is_ws & ~load_s
      wave_in = ww_s & wn_s
      ws_active = wave_in & is_ws_compute
      os_start = wave_in & is_os

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

      wave_out = ctrl.state != PEState.IDLE
      can_accumulate = ifthenelse(is_ws, Stream(not is_top_row), ctrl.os_acc)
      pe_accumulate = can_accumulate & acc_s
      pe_valid = ifthenelse(is_ws, ws_active, ctrl.state == PEState.OS_ACC3)

      with reset(dp_rst_s & is_os):
        # pylint: disable-next=undefined-variable,used-before-assignment
        acc_reg = fby(0.0, acc_curr)  # pyrefly: ignore[unknown-name]

      with reset(dp_rst_s):
        row_reg = fby(0.0, row_in)
        rhs_mux = ifthenelse(is_ws, acc_reg, col_in)
        # True IEEE-754 fp32 MAC rounding matching `reference_matmul_bf16_fp32`.
        prod = fp32_mul(row_in, rhs_mux)
        base_mux = ifthenelse(is_ws, col_in, acc_reg)
        acc_sum = fp32_add(base_mux, prod)
        acc_mux = ifthenelse(pe_accumulate, acc_sum, prod)
        ws_acc_mux = ifthenelse(load_s, col_in, acc_reg)
        ws_col_mux = ifthenelse(load_s, col_in, acc_mux)
        acc_curr = ifthenelse(is_ws, ws_acc_mux, acc_mux, name='mode_acc_mux')
        mode_col_mux = ifthenelse(is_ws, ws_col_mux, col_in)
        col_reg = fby(0.0, mode_col_mux)

      __output__ = PEOutputs(
          row_out=row_reg,
          col_out=col_reg,
          ps_out=acc_mux,
          acc=acc_curr,
          state=ctrl.state,
          valid=pe_valid,
          wave_out=wave_out,
      )

  return eq.output


def systolic_array(
    row_inputs: Sequence[Stream],
    col_inputs: Sequence[Stream],
    *,
    initial_mode: SystolicMode = SystolicMode.WEIGHT_STATIONARY,
    switch_mode: Stream | None = None,
    reload_weights: Stream | None = None,
    accumulate: Stream | None = None,
    reset_stream: Stream | None = None,
) -> SystolicOutputs:
  """Builds the 4x4 systolic array with distributed per-PE control."""
  if len(row_inputs) != ARRAY_SIZE:
    raise ValueError(
        f'row_inputs must have length {ARRAY_SIZE}, got {len(row_inputs)}.'
    )
  if len(col_inputs) != ARRAY_SIZE:
    raise ValueError(
        f'col_inputs must have length {ARRAY_SIZE}, got {len(col_inputs)}.'
    )
  acc_s = Stream(True) if accumulate is None else accumulate
  rst_s = Stream(False) if reset_stream is None else reset_stream

  with psyre_equations() as eq:

    class _:
      """4x4 systolic array grid and readout equations."""
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
            pe_out = systolic_pe(
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

        accumulators = _build_grid(lambda i, j, g=pe_grid: g[i][j].acc)
        pe_states = _build_grid(lambda i, j, g=pe_grid: g[i][j].state)
        pe_valid = _build_grid(lambda i, j, g=pe_grid: g[i][j].valid)

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

      __output__ = SystolicOutputs(
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

  return eq.output
