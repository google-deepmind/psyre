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
"""Unit tests for the 4x4 systolic array in Psyre."""

import logging
import math

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    Semantics,
    counter,
    ifthenelse,
    logger,
    reset,
    when,
)
from examples.src import systolic
from psyre.scheduler import Scheduler
from psyre.static_stream import repeat
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


class SystolicTest(absltest.TestCase):

  MOD = systolic

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def _assert_matrices_equal(
      self,
      actual_matrices: list[list[list[float]]],
      expected_matrices: list[list[list[float]]],
  ) -> None:
    self.assertLen(actual_matrices, len(expected_matrices))
    for m_idx, (actual, expected) in enumerate(
        zip(actual_matrices, expected_matrices, strict=True)
    ):
      self.assertEqual(
          actual,
          expected,
          msg=f'Matrix {m_idx} mismatch:\nactual={actual}\nexpected={expected}',
      )

  def test_bf16_quantization_and_fp32_accumulation_precision(self):
    mod = self.MOD
    # 1.0 + 2^-7 (0.0078125) is exact in bf16 (7 fraction bits).
    self.assertEqual(mod.to_bf16(1.0 + 2**-7), 1.0 + 2**-7)
    # 1.0 + 2^-8 (0.00390625) lies halfway between 1.0 (even LSB) and
    # 1.0 + 2^-7 (odd LSB); round-to-nearest-even rounds to 1.0.
    self.assertEqual(mod.to_bf16(1.0 + 2**-8), 1.0)
    # 1.0 + 2^-7 + 2^-8 lies halfway between 1.0 + 2^-7 (odd LSB) and
    # 1.0 + 2^-6 (even LSB); round-to-nearest-even rounds up to 1.0 + 2^-6.
    self.assertEqual(mod.to_bf16(1.0 + 2**-7 + 2**-8), 1.0 + 2**-6)
    self.assertTrue(math.isnan(mod.to_bf16(float('nan'))))

    # Array conversion to bf16 before providing inputs to the graph.
    self.assertEqual(
        mod.to_bf16([1.0 + 2**-7, 1.0 + 2**-8, 1.0 + 2**-7 + 2**-8]),
        [1.0 + 2**-7, 1.0, 1.0 + 2**-6],
    )

  def test_skew_and_deskew_streams_roundtrip(self):
    mod = self.MOD

    def program():
      raw_cols = [
          Stream(mod.to_bf16([10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0])),
          Stream(mod.to_bf16([11.0, 21.0, 31.0, 41.0, 0.0, 0.0, 0.0])),
          Stream(mod.to_bf16([12.0, 22.0, 32.0, 42.0, 0.0, 0.0, 0.0])),
          Stream(mod.to_bf16([13.0, 23.0, 33.0, 43.0, 0.0, 0.0, 0.0])),
      ]
      tick = counter(max_val=mod.ARRAY_SIZE - 1, name='skew_tick')
      skewed = mod.skew_streams(
          raw_cols, pad=Stream(-1.0), tick=tick, prefix='test_skew'
      )
      valid = tick >= (mod.ARRAY_SIZE - 1)
      deskewed = mod.deskew_streams(
          skewed, valid=valid, tick=tick, prefix='test_deskew'
      )
      return skewed, deskewed

    skewed_tr, deskewed_tr = Scheduler.run(
        program, horizon=7, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertLen([n for n in Scheduler.graph.nodes if n.op == 'counter'], 1)
    # Column j is delayed by j ticks with -1.0 padding.
    self.assertEqual(skewed_tr[0], [10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0])
    self.assertEqual(skewed_tr[1], [-1.0, 11.0, 21.0, 31.0, 41.0, 0.0, 0.0])
    self.assertEqual(skewed_tr[2], [-1.0, -1.0, 12.0, 22.0, 32.0, 42.0, 0.0])
    self.assertEqual(skewed_tr[3], [-1.0, -1.0, -1.0, 13.0, 23.0, 33.0, 43.0])

    # Deskewing shifts column j by (3 - j) ticks so all align at tick 3.
    mats = mod.extract_matrices(deskewed_tr)
    self.assertEqual(
        mats,
        [[
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
            [40.0, 41.0, 42.0, 43.0],
        ]],
    )

  def test_pe_direct_store_vs_accumulate_and_datapath_reset(self):
    mod = self.MOD
    pe_state = mod.PEState
    mode_enum = mod.SystolicMode

    def program():
      # 6 ticks in OS mode (wave_west=True, wave_north=True):
      # tick 0: IDLE    -> direct store 2.0 * 3.0 = 6.0
      # tick 1: OS_ACC1 -> accumulate 6.0 + 4.0 * 5.0 = 26.0
      # tick 2: OS_ACC2, accumulate=False -> direct store 1.5 * 2.0 = 3.0
      # tick 3: OS_ACC3 -> accumulate 3.0 + 2.0 * 0.5 = 4.0 (valid=True)
      # tick 4: datapath_reset=True -> resets to IDLE, clears row_out/col_out,
      #         direct store 3.0 * 3.0 = 9.0
      # tick 5: OS_ACC1 -> accumulate 9.0 + 1.0 * 2.0 = 11.0
      row_in = Stream(mod.to_bf16([2.0, 4.0, 1.5, 2.0, 3.0, 1.0]))
      col_in = Stream(mod.to_fp32([3.0, 5.0, 2.0, 0.5, 3.0, 2.0]))
      acc_flag = Stream([False, True, False, True, True, True])
      dp_reset = Stream([False, False, False, False, True, False])
      pe = mod.systolic_pe(
          row_in,
          col_in,
          mode=Stream(mode_enum.OUTPUT_STATIONARY),
          accumulate=acc_flag,
          datapath_reset=dp_reset,
          name='unit_pe',
      )
      return pe.acc, pe.row_out, pe.col_out, pe.state, pe.valid

    acc_tr, row_out_tr, col_out_tr, state_tr, valid_tr = Scheduler.run(
        program, horizon=6, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(acc_tr, [6.0, 26.0, 3.0, 4.0, 9.0, 11.0])
    # On tick 4, datapath_reset clears row_out and col_out to 0.0.
    self.assertEqual(row_out_tr, [0.0, 2.0, 4.0, 1.5, 0.0, 3.0])
    self.assertEqual(col_out_tr, [0.0, 3.0, 5.0, 2.0, 0.0, 3.0])
    self.assertEqual(
        state_tr,
        [
            pe_state.IDLE,
            pe_state.OS_ACC1,
            pe_state.OS_ACC2,
            pe_state.OS_ACC3,
            pe_state.IDLE,
            pe_state.OS_ACC1,
        ],
    )
    self.assertEqual(valid_tr, [False, False, False, True, False, False])
    fby_names = [n.name for n in Scheduler.graph.nodes if n.op == 'fby']
    self.assertCountEqual(
        fby_names,
        [
            'unit_pe/acc_reg',
            'unit_pe/row_reg',
            'unit_pe/col_reg',
            'unit_pe/pe_ctrl/pns',
        ],
    )
    self.assertEmpty(
        [n for n in Scheduler.graph.nodes if n.op == 'static_stream']
    )
    self.assertEmpty([
        n
        for n in Scheduler.graph.nodes
        if n.name.startswith('unit_pe/pe_ctrl/')
        and n.op in {'__and__', '__or__'}
    ])

  def test_distributed_control_prefixes_and_transitions(self):
    mod = self.MOD
    pe_state = mod.PEState
    mode_enum = mod.SystolicMode

    def program():
      # Global orchestrator starts in WEIGHT_STATIONARY, switches to
      # OUTPUT_STATIONARY after tick 5:
      # ticks 0..3: load_weights=True, PEState.IDLE
      # tick 4:     load_weights=False, datapath_reset=True, PEState.IDLE
      # tick 5:     PEState.WS_ACTIVE, switch_mode=True
      # tick 6:     mode=OS, datapath_reset=True, PEState.IDLE
      # ticks 7..9: PEState.OS_ACC1 -> OS_ACC2 -> OS_ACC3
      ctrl = mod.systolic_control(
          initial_mode=mode_enum.WEIGHT_STATIONARY,
          switch_mode=when(Stream(True), Stream.once(5)),
      )
      pe = mod.systolic_pe(
          Stream(0.0),
          Stream(0.0),
          mode=ctrl.mode,
          load_weights=ctrl.load_weights,
          datapath_reset=ctrl.datapath_reset,
          name='pe_0_0',
      )
      return (
          ctrl.mode,
          ctrl.load_weights,
          ctrl.datapath_reset,
          pe.state,
          pe.valid,
      )

    mode_tr, load_tr, reset_tr, state_tr, valid_tr = Scheduler.run(
        program, horizon=10, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(
        mode_tr,
        [mode_enum.WEIGHT_STATIONARY] * 6
        + [mode_enum.OUTPUT_STATIONARY] * 4,
    )
    self.assertEqual(
        load_tr,
        [True, True, True, True, False, False, False, False, False, False],
    )
    self.assertEqual(
        reset_tr,
        [True, False, False, False, True, False, True, False, False, False],
    )
    self.assertEqual(
        state_tr,
        [
            pe_state.IDLE,
            pe_state.IDLE,
            pe_state.IDLE,
            pe_state.IDLE,
            pe_state.IDLE,
            pe_state.WS_ACTIVE,
            pe_state.IDLE,
            pe_state.OS_ACC1,
            pe_state.OS_ACC2,
            pe_state.OS_ACC3,
        ],
    )
    self.assertEqual(
        valid_tr,
        [False, False, False, False, True, True, False, False, False, True],
    )

    # Verify global orchestrator ('mode_ctrl') and per-PE ('pe_0_0/pe_ctrl')
    # automaton prefixes exist cleanly separated in the graph.
    node_names = [node.name for node in Scheduler.graph.nodes]
    self.assertTrue(
        any(name.startswith('mode_ctrl/') for name in node_names),
        msg=f'Expected mode_ctrl/ prefix in {node_names}',
    )
    self.assertTrue(
        any(name.startswith('pe_0_0/pe_ctrl/') for name in node_names),
        msg=f'Expected pe_0_0/pe_ctrl/ prefix in {node_names}',
    )

  def test_weight_stationary_pipelined_products_and_weight_reload(self):
    mod = self.MOD
    pe_state = mod.PEState
    mode_enum = mod.SystolicMode
    weights_epoch0 = mod.to_bf16([
        [1.0, 2.0, -1.0, 0.5],
        [0.5, -1.5, 2.0, 1.0],
        [2.0, 0.25, 1.0, -0.5],
        [-1.0, 1.0, 0.5, 2.0],
    ])
    a0 = mod.to_bf16([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, -2.0, 1.0],
        [-1.0, 0.0, 2.5, -0.5],
        [2.0, -1.0, 0.5, 1.5],
    ])
    a1 = mod.to_bf16([
        [1.0 + 2**-7, 2**-5, -1.25, 0.75],
        [2.5, -0.5, 1.125, 3.0],
        [0.25, 1.75, -2.25, 0.5],
        [-1.5, 0.5, 2.0, -0.25],
    ])
    weights_epoch1 = mod.to_bf16([
        [0.5, -0.5, 1.5, 2.0],
        [1.0, 2.0, -1.0, 0.25],
        [-1.5, 0.75, 0.5, 1.0],
        [2.0, 1.0, -0.25, -1.0],
    ])
    a2 = mod.to_bf16([
        [2.0, -1.0, 0.5, 3.0],
        [1.25, 2.5, -1.5, 0.5],
        [-0.5, 1.0, 2.0, -2.0],
        [3.0, 0.25, -1.0, 1.5],
    ])

    epochs = [
        (weights_epoch0, [a0, a1]),
        (weights_epoch1, [a2]),
    ]
    row_seqs, col_seqs, reload_seq, horizon = mod.build_ws_schedule(epochs)

    def program():
      # Delay/skew row activations only during COMPUTE; reset skew delays when
      # entering a new phase. We skew row streams continuously because row_seqs
      # pads with 0.0 during the 4 LOAD cycles and 6 DRAIN cycles.
      row_streams = [Stream(seq) for seq in row_seqs]
      skewed_rows = mod.skew_streams(row_streams, prefix='ws_row')
      col_streams = [Stream(seq) for seq in col_seqs]
      out = mod.systolic_array(
          skewed_rows,
          col_streams,
          initial_mode=mode_enum.WEIGHT_STATIONARY,
          reload_weights=when(Stream(True), reload_seq),
      )
      return (
          out.deskewed_outputs,
          out.accumulators,
          out.load_weights,
          out.pe_states[0][0],
      )

    expected_matrices = [
        mod.reference_matmul_bf16_fp32(a0, weights_epoch0),
        mod.reference_matmul_bf16_fp32(a1, weights_epoch0),
        mod.reference_matmul_bf16_fp32(a2, weights_epoch1),
    ]
    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        deskewed_tr, acc_tr, load_tr, state_tr = Scheduler.run(
            program, horizon=horizon, semantics=sem
        )
        self.assertFalse(Scheduler.deadlocked)

        # Verify merged weight/accumulator registers after epoch 0 load (tick 3)
        # and epoch 1 load.
        loaded_w0 = [[acc_tr[i][j][3] for j in range(4)] for i in range(4)]
        loaded_w1 = [[acc_tr[i][j][-1] for j in range(4)] for i in range(4)]
        self.assertEqual(loaded_w0, weights_epoch0)
        self.assertEqual(loaded_w1, weights_epoch1)

        # Verify global load_weights pulses and local PE (0, 0) transitions:
        # LOAD (4) -> COMPUTE (14) -> LOAD (4) -> COMPUTE (10).
        self.assertEqual(load_tr[:5], [True, True, True, True, False])
        self.assertEqual(load_tr[18:23], [True, True, True, True, False])
        self.assertEqual(
            state_tr[:6], [pe_state.IDLE] * 5 + [pe_state.WS_ACTIVE]
        )
        self.assertEqual(
            state_tr[18:24], [pe_state.IDLE] * 5 + [pe_state.WS_ACTIVE]
        )

        actual_matrices = mod.extract_matrices(deskewed_tr)
        self._assert_matrices_equal(actual_matrices, expected_matrices)

  def test_output_stationary_pipelined_products(self):
    mod = self.MOD
    mode_enum = mod.SystolicMode
    a0 = mod.to_bf16([
        [1.0, 2.0, -1.0, 0.5],
        [3.0, -0.5, 2.0, 1.5],
        [-2.0, 1.0, 0.25, -1.0],
        [0.5, 2.5, -1.5, 2.0],
    ])
    b0 = mod.to_bf16([
        [2.0, -1.0, 0.5, 1.0],
        [0.5, 1.5, -2.0, 0.25],
        [-1.0, 2.0, 1.0, -0.5],
        [1.5, -0.5, 0.75, 2.0],
    ])
    a1 = mod.to_bf16([
        [1.0, 2**-5, 2**-5, 2**-5],
        [1.25, -2.25, 0.75, -0.5],
        [0.5, 1.5, -1.25, 2.5],
        [-0.75, 0.25, 1.0, -1.5],
    ])
    b1 = mod.to_bf16([
        [1.0, -0.5, 2.0, 0.25],
        [2**-5, 1.25, -0.75, 1.5],
        [2**-5, -1.5, 0.5, -2.0],
        [2**-5, 0.75, 1.25, 0.5],
    ])
    a2 = mod.to_bf16([
        [2.0, 1.0, 0.0, -1.0],
        [-1.0, 0.5, 2.0, 1.0],
        [0.25, -0.5, 1.5, 0.75],
        [1.5, 2.0, -1.0, 0.5],
    ])
    b2 = mod.to_bf16([
        [0.5, 1.0, -1.5, 2.0],
        [1.0, -2.0, 0.5, -0.5],
        [-0.5, 0.25, 1.0, 1.5],
        [2.0, 1.5, -0.25, 0.5],
    ])

    pairs = [(a0, b0), (a1, b1), (a2, b2)]
    row_seqs, col_seqs, horizon = mod.build_os_schedule(pairs)

    def program():
      row_streams = [Stream(seq) for seq in row_seqs]
      col_streams = [Stream(seq) for seq in col_seqs]
      os_tick = counter(max_val=mod.ARRAY_SIZE - 1, name='os_skew_cnt')
      skewed_rows = mod.skew_streams(
          row_streams, tick=os_tick, prefix='os_row'
      )
      skewed_cols = mod.skew_streams(
          col_streams, tick=os_tick, prefix='os_col'
      )
      out = mod.systolic_array(
          skewed_rows,
          skewed_cols,
          initial_mode=mode_enum.OUTPUT_STATIONARY,
      )
      return out.deskewed_outputs, out.accumulators, out.pe_valid

    expected_matrices = [
        mod.reference_matmul_bf16_fp32(a0, b0),
        mod.reference_matmul_bf16_fp32(a1, b1),
        mod.reference_matmul_bf16_fp32(a2, b2),
    ]
    # Verify fp32 accumulation preserves sub-bf16 increments (1.0 + 3 * 2^-10).
    self.assertEqual(
        expected_matrices[1][0][0], mod.to_fp32(1.0 + 3.0 * (2**-10))
    )
    self.assertNotEqual(
        expected_matrices[1][0][0], mod.to_bf16(expected_matrices[1][0][0])
    )
    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        deskewed_tr, acc_tr, pe_valid_tr = Scheduler.run(
            program, horizon=horizon, semantics=sem
        )
        self.assertFalse(Scheduler.deadlocked)
        deskewed_matrices = mod.extract_matrices(deskewed_tr)
        pe_matrices = mod.extract_os_pe_matrices(acc_tr, pe_valid_tr)
        self._assert_matrices_equal(deskewed_matrices, expected_matrices)
        self._assert_matrices_equal(pe_matrices, expected_matrices)

  def test_mode_switching_between_ws_and_os_in_single_run(self):
    mod = self.MOD
    mode_enum = mod.SystolicMode
    w_ws = mod.to_bf16([
        [1.0, 0.5, -1.0, 2.0],
        [2.0, -1.0, 0.5, 1.0],
        [-0.5, 1.5, 2.0, -1.0],
        [1.0, 2.0, -0.5, 0.5],
    ])
    a_ws = mod.to_bf16([
        [2.0, 1.0, -1.0, 0.5],
        [0.5, -2.0, 1.5, 1.0],
        [1.0, 0.5, 2.0, -1.5],
        [-1.0, 1.5, 0.5, 2.0],
    ])
    a_os = mod.to_bf16([
        [1.5, -0.5, 2.0, 1.0],
        [0.5, 2.0, -1.0, 0.5],
        [-1.0, 1.0, 0.5, 2.0],
        [2.0, -1.5, 1.0, -0.5],
    ])
    b_os = mod.to_bf16([
        [1.0, 2.0, -0.5, 1.5],
        [-1.0, 0.5, 2.0, -1.0],
        [0.5, -1.0, 1.5, 0.5],
        [2.0, 1.0, -1.0, 2.0],
    ])

    ws_rows, ws_cols, _, ws_horizon = mod.build_ws_schedule([(w_ws, [a_ws])])
    os_rows, os_cols, os_horizon = mod.build_os_schedule([(a_os, b_os)])
    total_horizon = ws_horizon + os_horizon

    def program():
      global_tick = counter(name='global_tick')
      is_os_phase = global_tick >= ws_horizon
      ws_pad = repeat(0.0, os_horizon)
      os_pad = repeat(0.0, ws_horizon)
      ws_row_streams = [Stream(row_w + ws_pad) for row_w in ws_rows]
      os_row_streams = [Stream(os_pad + row_o) for row_o in os_rows]
      os_col_streams = [Stream(os_pad + col_o) for col_o in os_cols]
      with reset(global_tick == ws_horizon):
        skew_tick = counter(
            max_val=mod.ARRAY_SIZE - 1, name='mix_skew_cnt'
        )
        skewed_ws_rows = mod.skew_streams(
            ws_row_streams,
            tick=skew_tick,
            prefix='mix_ws_row',
        )
        skewed_os_rows = mod.skew_streams(
            os_row_streams,
            tick=skew_tick,
            prefix='mix_os_row',
        )
        skewed_os_cols = mod.skew_streams(
            os_col_streams,
            tick=skew_tick,
            prefix='mix_os_col',
        )

      row_inputs = [
          ifthenelse(is_os_phase, skewed_os_rows[i], skewed_ws_rows[i])
          for i in range(4)
      ]
      col_inputs = [
          ifthenelse(
              is_os_phase,
              skewed_os_cols[j],
              Stream(ws_cols[j] + ws_pad),
          )
          for j in range(4)
      ]
      out = mod.systolic_array(
          row_inputs,
          col_inputs,
          initial_mode=mode_enum.WEIGHT_STATIONARY,
          switch_mode=when(Stream(True), Stream.once(ws_horizon - 1)),
      )
      return out.deskewed_outputs, out.mode

    expected_matrices = [
        mod.reference_matmul_bf16_fp32(a_ws, w_ws),
        mod.reference_matmul_bf16_fp32(a_os, b_os),
    ]
    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        deskewed_tr, mode_tr = Scheduler.run(
            program, horizon=total_horizon, semantics=sem
        )
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(
            mode_tr,
            [mode_enum.WEIGHT_STATIONARY] * ws_horizon
            + [mode_enum.OUTPUT_STATIONARY] * os_horizon,
        )
        actual_matrices = mod.extract_matrices(deskewed_tr)
        self._assert_matrices_equal(actual_matrices, expected_matrices)

  def test_non_accumulate_override_stores_product_directly(self):
    mod = self.MOD
    mode_enum = mod.SystolicMode
    # When accumulate=False is forced across the array in OS mode, each PE
    # overwrites its accumulator with a * b on every cycle instead of adding.
    # Therefore, at the end of a 4-cycle tile (k = 3), PE (i, j) holds only the
    # final product bf16(A[i, 3]) * bf16(B[3, j]) rather than the sum over k.
    mat_a = mod.to_bf16([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ])
    mat_b = mod.to_bf16([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0],
        [0.5, 1.5, -0.5, 2.0],
    ])
    row_seqs, col_seqs, horizon = mod.build_os_schedule([(mat_a, mat_b)])

    def program():
      row_streams = [Stream(seq) for seq in row_seqs]
      col_streams = [Stream(seq) for seq in col_seqs]
      ovr_tick = counter(max_val=mod.ARRAY_SIZE - 1, name='ovr_skew_cnt')
      skewed_rows = mod.skew_streams(
          row_streams, tick=ovr_tick, prefix='ovr_row'
      )
      skewed_cols = mod.skew_streams(
          col_streams, tick=ovr_tick, prefix='ovr_col'
      )
      out = mod.systolic_array(
          skewed_rows,
          skewed_cols,
          initial_mode=mode_enum.OUTPUT_STATIONARY,
          accumulate=Stream(False),
      )
      return out.deskewed_outputs

    expected_last_outer_product = [
        [mod.to_fp32(mat_a[i][3] * mat_b[3][j]) for j in range(4)]
        for i in range(4)
    ]
    deskewed_tr = Scheduler.run(
        program, horizon=horizon, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    actual_matrices = mod.extract_matrices(deskewed_tr)
    self._assert_matrices_equal(actual_matrices, [expected_last_outer_product])

  def test_global_single_cycle_reset_aborts_and_restarts_cleanly(self):
    mod = self.MOD
    pe_state = mod.PEState
    mode_enum = mod.SystolicMode
    w_good = mod.to_bf16([
        [1.0, -1.0, 2.0, 0.5],
        [0.5, 2.0, -0.5, 1.0],
        [2.0, 0.5, 1.0, -1.5],
        [-1.0, 1.5, 0.5, 2.0],
    ])
    a_good = mod.to_bf16([
        [2.0, 0.5, -1.0, 1.5],
        [-1.0, 2.0, 0.5, -0.5],
        [1.5, -0.5, 2.0, 1.0],
        [0.5, 1.0, -1.5, 2.0],
    ])
    good_rows, good_cols, good_reload, good_horizon = mod.build_ws_schedule(
        [(w_good, [a_good])]
    )

    # Run 6 ticks of garbage/corrupted activity, assert global single-cycle
    # reset at tick 6, and run the clean schedule starting at tick 6.
    abort_ticks = 6
    total_horizon = abort_ticks + good_horizon
    full_rows = [
        repeat(mod.to_bf16(99.0), abort_ticks) + good_rows[i]
        for i in range(mod.ARRAY_SIZE)
    ]
    full_cols = [
        repeat(mod.to_bf16(77.0), abort_ticks) + good_cols[j]
        for j in range(mod.ARRAY_SIZE)
    ]
    full_reload = repeat(False, abort_ticks) + good_reload

    def program():
      rst = Stream.once(abort_ticks)
      raw_rows = [Stream(seq) for seq in full_rows]
      with reset(rst):
        skewed_rows = mod.skew_streams(raw_rows, prefix='rst_row')
      col_streams = [Stream(seq) for seq in full_cols]
      out = mod.systolic_array(
          skewed_rows,
          col_streams,
          initial_mode=mode_enum.WEIGHT_STATIONARY,
          reload_weights=when(Stream(True), full_reload),
          reset_stream=rst,
      )
      return (
          out.deskewed_outputs,
          out.load_weights,
          out.pe_states[0][0],
          out.accumulators[0][0],
          out.accumulators[1][0],
      )

    expected_matrices = [mod.reference_matmul_bf16_fp32(a_good, w_good)]
    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        deskewed_tr, load_tr, state00_tr, acc00_tr, acc10_tr = Scheduler.run(
            program, horizon=total_horizon, semantics=sem
        )
        self.assertFalse(Scheduler.deadlocked)

        # At tick 6 (when reset_stream=True), the array restarts in WS load:
        # PE (0, 0) latches w_good[3][0] while PE (1, 0)'s acc resets to 0.
        self.assertTrue(load_tr[abort_ticks])
        self.assertEqual(state00_tr[abort_ticks], pe_state.IDLE)
        self.assertEqual(acc00_tr[abort_ticks], w_good[3][0])
        self.assertEqual(acc10_tr[abort_ticks], 0.0)

        # All valid outputs produced after reset match the clean matrix product.
        actual_matrices = mod.extract_matrices(deskewed_tr)
        self._assert_matrices_equal(actual_matrices, expected_matrices)

  def test_validation_errors(self):
    mod = self.MOD
    with self.assertRaisesRegex(ValueError, 'must have 4 rows'):
      mod.reference_matmul_bf16_fp32([[1.0] * 4] * 3, [[1.0] * 4] * 4)
    with self.assertRaisesRegex(ValueError, 'must have 4 columns'):
      mod.reference_matmul_bf16_fp32([[1.0] * 3] * 4, [[1.0] * 4] * 4)
    with self.assertRaisesRegex(ValueError, 'non-negative integer'):
      mod.delay_padded(Stream([1.0, 2.0]), -1)
    with self.assertRaisesRegex(ValueError, 'expects 4 streams'):
      mod.skew_streams([Stream(1.0), Stream(2.0)])
    with self.assertRaisesRegex(ValueError, 'expects 4 streams'):
      mod.deskew_streams([Stream(1.0), Stream(2.0)])
    with self.assertRaisesRegex(ValueError, 'epochs must be non-empty'):
      mod.build_ws_schedule([])
    with self.assertRaisesRegex(ValueError, 'matrix_pairs must be non-empty'):
      mod.build_os_schedule([])
    with self.assertRaisesRegex(ValueError, 'non-negative integer'):
      mod.build_ws_schedule(
          [([[0.0] * 4] * 4, [[[0.0] * 4] * 4])], drain_cycles=-1
      )

    # Large drain_cycles must stay compact via symbolic StaticWord repetition.
    mat = [[1.0] * 4] * 4
    rows, cols, reload_w, horizon = mod.build_ws_schedule(
        [(mat, [mat])], drain_cycles=10**6
    )
    self.assertEqual(horizon, 4 + 4 + 10**6)
    self.assertLen(rows[0], horizon)
    self.assertLen(cols[0], horizon)
    self.assertLen(reload_w, horizon)
    self.assertEqual(rows[0][horizon - 1], 0.0)
    self.assertFalse(reload_w[horizon - 1])

    # Multi-epoch WS schedule requires >= 1 drain cycle for the reload pulse.
    with self.assertRaisesRegex(ValueError, 'drain_cycles must be >= 1'):
      mod.build_ws_schedule([(mat, [mat]), (mat, [mat])], drain_cycles=0)

    # Reject str/bytes/bool in to_fp32 / to_bf16 instead of recursing forever.
    for bad_val in ('a', b'a', True):
      with self.assertRaises(TypeError):
        mod.to_fp32(bad_val)
      with self.assertRaises(TypeError):
        mod.to_bf16(bad_val)

    # Validate 4x4 grid shape and non-negative num_matrices in PE extractor.
    ok_grid = [[[0.0] for _ in range(4)] for _ in range(4)]
    with self.assertRaisesRegex(ValueError, 'must have 4 rows'):
      mod.extract_os_pe_matrices(ok_grid[:3], ok_grid)
    with self.assertRaisesRegex(ValueError, 'must have 4 columns'):
      mod.extract_os_pe_matrices([[] for _ in range(4)], ok_grid)
    with self.assertRaisesRegex(ValueError, 'non-negative integer'):
      mod.extract_os_pe_matrices(ok_grid, ok_grid, num_matrices=-1)

  def test_true_fp32_mac_rounding_matches_reference(self):
    mod = self.MOD
    mode_enum = mod.SystolicMode
    # 1.0 + 2^-25 rounds to 1.0 twice in fp32, whereas fp64 yields 1.0 + 2^-24.
    a_mat = [[1.0, 2**-12, 2**-12, 0.0]] + [[0.0] * 4] * 3
    b_mat = [
        [1.0, 0.0, 0.0, 0.0],
        [2**-13, 0.0, 0.0, 0.0],
        [2**-13, 0.0, 0.0, 0.0],
        [0.0] * 4,
    ]
    ref = mod.reference_matmul_bf16_fp32(a_mat, b_mat)
    self.assertEqual(ref[0][0], 1.0)

    os_rows, os_cols, os_h = mod.build_os_schedule([(a_mat, b_mat)])
    def os_prog():
      r = mod.skew_streams([Stream(os_rows[i]) for i in range(4)])
      c = mod.skew_streams([Stream(os_cols[j]) for j in range(4)])
      return mod.systolic_array(
          r, c, initial_mode=mode_enum.OUTPUT_STATIONARY
      ).deskewed_outputs
    os_got = mod.extract_matrices(Scheduler.run(os_prog, horizon=os_h))
    self._assert_matrices_equal(os_got, [ref])

    ws_rows, ws_cols, _, ws_h = mod.build_ws_schedule([(b_mat, [a_mat])])
    def ws_prog():
      r = mod.skew_streams([Stream(ws_rows[i]) for i in range(4)])
      c = [Stream(ws_cols[j]) for j in range(4)]
      return mod.systolic_array(
          r, c, initial_mode=mode_enum.WEIGHT_STATIONARY
      ).deskewed_outputs
    ws_got = mod.extract_matrices(Scheduler.run(ws_prog, horizon=ws_h))
    self._assert_matrices_equal(ws_got, [ref])


if __name__ == '__main__':
  absltest.main()
