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
"""4x4 systolic array in Psyre via `psyre_equations` (experimental).

Note: Everything under `psyre/equations/` is experimental; APIs and internals
may change without notice. Replicates `examples/src/systolic.py` without
explicit `lambda:` thunks in `systolic_pe`, relying on `with psyre_equations`
and automatic stream naming while reusing the thunk-free array helpers from
`examples/src/systolic.py` (see Kung & Leiserson, 1979; Kung, 1982; Colaço,
Pagano & Pouzet, EMSOFT '05).
"""

from collections.abc import Sequence

# pylint: disable=g-import-not-at-top,g-importing-member,g-multiple-import
from psyre.automaton import (
    automaton,
    state,
    until,
)
from psyre.common import (
    fby,
    ifthenelse,
    psyre_node,
    reset,
)
from psyre.equations import psyre_equations
from psyre.stream import Stream

try:
  from examples.src import systolic as _systolic
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
      build_os_schedule,
      build_ws_schedule,
      delay_padded,
      deskew_streams,
      extract_matrices,
      extract_os_pe_matrices,
      fp32_add,
      fp32_mul,
      reference_matmul_bf16_fp32,
      skew_streams,
      systolic_control,
      to_bf16,
      to_fp32,
  )
except ModuleNotFoundError as _err:
  if _err.name and (
      _err.name == 'examples' or _err.name.startswith('examples.')
  ):
    raise ImportError(
        'psyre.equations.examples requires the repository `examples` package '
        'on sys.path (e.g. an editable install or source checkout).'
    ) from _err
  raise

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import

# Re-export shared constants, types, schedules, helpers, and trace extractors.
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
  """Builds the 4x4 systolic array using the `psyre_equations` `systolic_pe`."""
  return _systolic.systolic_array(
      row_inputs,
      col_inputs,
      initial_mode=initial_mode,
      switch_mode=switch_mode,
      reload_weights=reload_weights,
      accumulate=accumulate,
      reset_stream=reset_stream,
      pe_fn=systolic_pe,
  )
