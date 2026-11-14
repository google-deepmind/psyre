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
from psyre import systolic
from psyre.psyre import (
    Scheduler,
    Semantics,
    Stream,
    counter,
    ifthenelse,
    logger,
    reset,
)
from psyre.systolic import (
    OutputStationaryState,
    SystolicMode,
    WeightStationaryState,
)

# pylint: enable=g-importing-member,g-multiple-import


class SystolicTest(absltest.TestCase):

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
    # 1.0 + 2^-7 (0.0078125) is exact in bf16 (7 fraction bits).
    self.assertEqual(systolic.to_bf16(1.0 + 2**-7), 1.0 + 2**-7)
    # 1.0 + 2^-8 (0.00390625) lies halfway between 1.0 (even LSB) and
    # 1.0 + 2^-7 (odd LSB); round-to-nearest-even rounds to 1.0.
    self.assertEqual(systolic.to_bf16(1.0 + 2**-8), 1.0)
    # 1.0 + 2^-7 + 2^-8 lies halfway between 1.0 + 2^-7 (odd LSB) and
    # 1.0 + 2^-6 (even LSB); round-to-nearest-even rounds up to 1.0 + 2^-6.
    self.assertEqual(systolic.to_bf16(1.0 + 2**-7 + 2**-8), 1.0 + 2**-6)
    self.assertTrue(math.isnan(systolic.to_bf16(float('nan'))))

    # Array conversion to bf16 before providing inputs to the graph.
    self.assertEqual(
        systolic.to_bf16([1.0 + 2**-7, 1.0 + 2**-8, 1.0 + 2**-7 + 2**-8]),
        [1.0 + 2**-7, 1.0, 1.0 + 2**-6],
    )

    # Verify that multiplication and accumulation happen in fp32 rather than
    # bf16: 1.0 * 1.0 = 1.0, and 2^-5 * 2^-5 = 2^-10 (exact in bf16 inputs,
    # product is 2^-10). Adding 1.0 + 2^-10 in fp32 preserves 2^-10, whereas
    # rounding the accumulator to bf16 would lose 2^-10 and return 1.0.
    mat_a = systolic.to_bf16([
        [1.0, 2**-5, 2**-5, 2**-5],
        [0.5, -0.25, 1.5, 2.0],
        [1.25, 0.75, -0.5, 0.125],
        [3.0, -1.0, 0.5, 2**-4],
    ])
    mat_b = systolic.to_bf16([
        [1.0, 0.5, -1.0, 2.0],
        [2**-5, 1.0, 0.25, -0.5],
        [2**-5, -0.5, 1.5, 0.75],
        [2**-5, 0.25, 0.125, 1.25],
    ])
    ref = systolic.reference_matmul_bf16_fp32(mat_a, mat_b)
    expected_00 = systolic.to_fp32(1.0 + 3.0 * (2**-10))
    self.assertEqual(ref[0][0], expected_00)
    self.assertNotEqual(ref[0][0], systolic.to_bf16(ref[0][0]))

  def test_skew_and_deskew_streams_roundtrip(self):
    def program():
      raw_cols = [
          Stream(systolic.to_bf16([10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0])),
          Stream(systolic.to_bf16([11.0, 21.0, 31.0, 41.0, 0.0, 0.0, 0.0])),
          Stream(systolic.to_bf16([12.0, 22.0, 32.0, 42.0, 0.0, 0.0, 0.0])),
          Stream(systolic.to_bf16([13.0, 23.0, 33.0, 43.0, 0.0, 0.0, 0.0])),
      ]
      skewed = systolic.skew_streams(
          raw_cols, pad=Stream(-1.0), prefix='test_skew'
      )
      valid = counter() >= 3
      deskewed = systolic.deskew_streams(
          skewed, valid=valid, prefix='test_deskew'
      )
      return skewed, deskewed

    skewed_tr, deskewed_tr = Scheduler.run(
        program, horizon=7, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    # Column j is delayed by j ticks with -1.0 padding.
    self.assertEqual(skewed_tr[0], [10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0])
    self.assertEqual(skewed_tr[1], [-1.0, 11.0, 21.0, 31.0, 41.0, 0.0, 0.0])
    self.assertEqual(skewed_tr[2], [-1.0, -1.0, 12.0, 22.0, 32.0, 42.0, 0.0])
    self.assertEqual(skewed_tr[3], [-1.0, -1.0, -1.0, 13.0, 23.0, 33.0, 43.0])

    # Deskewing shifts column j by (3 - j) ticks so all columns align at tick 3.
    mats = systolic.extract_matrices(deskewed_tr)
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
    def program():
      # 6 ticks in OS mode (is_weight_stationary=False):
      # tick 0: accumulate=False -> direct store 2.0 * 3.0 = 6.0
      # tick 1: accumulate=True  -> accumulate 6.0 + 4.0 * 5.0 = 26.0
      # tick 2: accumulate=False -> direct store 1.5 * 2.0 = 3.0 (overwrites 26)
      # tick 3: accumulate=True  -> accumulate 3.0 + 2.0 * 0.5 = 4.0
      # tick 4: datapath_reset=True, accumulate=True -> acc_prev resets to 0.0,
      #         so 0.0 + 3.0 * 3.0 = 9.0
      # tick 5: accumulate=True  -> accumulate 9.0 + 1.0 * 2.0 = 11.0
      a_in = Stream(systolic.to_bf16([2.0, 4.0, 1.5, 2.0, 3.0, 1.0]))
      b_in = Stream(systolic.to_fp32([3.0, 5.0, 2.0, 0.5, 3.0, 2.0]))
      acc_flag = Stream([False, True, False, True, True, True])
      dp_reset = Stream([False, False, False, False, True, False])
      pe = systolic.systolic_pe(
          a_in,
          b_in,
          is_weight_stationary=Stream(False),
          load_weights=Stream(False),
          accumulate=acc_flag,
          datapath_reset=dp_reset,
          name='unit_pe',
      )
      return pe.acc, pe.a_out, pe.b_out

    acc_tr, a_out_tr, b_out_tr = Scheduler.run(
        program, horizon=6, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(acc_tr, [6.0, 26.0, 3.0, 4.0, 9.0, 11.0])
    # On tick 4, datapath_reset clears a_out and b_out to 0.0.
    self.assertEqual(a_out_tr, [0.0, 2.0, 4.0, 1.5, 0.0, 3.0])
    self.assertEqual(b_out_tr, [0.0, 3.0, 5.0, 2.0, 0.0, 3.0])
    node_names = [node.name for node in Scheduler.graph.nodes]
    self.assertIn('unit_pe/acc_reg', node_names)
    self.assertIn('unit_pe/a_reg', node_names)
    self.assertIn('unit_pe/b_reg', node_names)

  def test_hierarchical_automaton_node_prefixes_and_transitions(self):
    def program():
      # Start in WEIGHT_STATIONARY:
      # ticks 0..3: LOAD (4 ticks)
      # ticks 4..5: COMPUTE (2 ticks), switch_mode=True at tick 5
      # ticks 6..9: OUTPUT_STATIONARY (INIT on tick 6, ACCUMULATE on ticks 7..9)
      switch_seq = [
          False,
          False,
          False,
          False,
          False,
          True,
          False,
          False,
          False,
          False,
      ]
      ctrl = systolic.systolic_control(
          initial_mode=SystolicMode.WEIGHT_STATIONARY,
          switch_mode=Stream(switch_seq),
      )
      return ctrl.mode, ctrl.substate, ctrl.load_weights, ctrl.datapath_reset

    mode_tr, substate_tr, load_tr, reset_tr = Scheduler.run(
        program, horizon=10, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(
        mode_tr,
        [SystolicMode.WEIGHT_STATIONARY] * 6
        + [SystolicMode.OUTPUT_STATIONARY] * 4,
    )
    self.assertEqual(
        substate_tr,
        [
            WeightStationaryState.LOAD,
            WeightStationaryState.LOAD,
            WeightStationaryState.LOAD,
            WeightStationaryState.LOAD,
            WeightStationaryState.COMPUTE,
            WeightStationaryState.COMPUTE,
            OutputStationaryState.INIT,
            OutputStationaryState.ACCUMULATE,
            OutputStationaryState.ACCUMULATE,
            OutputStationaryState.ACCUMULATE,
        ],
    )
    self.assertEqual(
        load_tr,
        [True, True, True, True, False, False, False, False, False, False],
    )
    self.assertEqual(
        reset_tr,
        [True, False, False, False, True, False, True, False, False, False],
    )

    # Verify that the two-level hierarchical automaton prefixes node names with
    # both the outer ('mode_ctrl') and inner ('ws_ctrl' / 'os_ctrl') names.
    node_names = [node.name for node in Scheduler.graph.nodes]
    self.assertTrue(
        any(name.startswith('mode_ctrl/ws_ctrl/') for name in node_names),
        msg=f'Expected mode_ctrl/ws_ctrl/ prefix in {node_names}',
    )
    self.assertTrue(
        any(name.startswith('mode_ctrl/os_ctrl/') for name in node_names),
        msg=f'Expected mode_ctrl/os_ctrl/ prefix in {node_names}',
    )

  def test_weight_stationary_pipelined_products_and_weight_reload(self):
    weights_epoch0 = systolic.to_bf16([
        [1.0, 2.0, -1.0, 0.5],
        [0.5, -1.5, 2.0, 1.0],
        [2.0, 0.25, 1.0, -0.5],
        [-1.0, 1.0, 0.5, 2.0],
    ])
    a0 = systolic.to_bf16([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, -2.0, 1.0],
        [-1.0, 0.0, 2.5, -0.5],
        [2.0, -1.0, 0.5, 1.5],
    ])
    a1 = systolic.to_bf16([
        [1.0 + 2**-7, 2**-5, -1.25, 0.75],
        [2.5, -0.5, 1.125, 3.0],
        [0.25, 1.75, -2.25, 0.5],
        [-1.5, 0.5, 2.0, -0.25],
    ])
    weights_epoch1 = systolic.to_bf16([
        [0.5, -0.5, 1.5, 2.0],
        [1.0, 2.0, -1.0, 0.25],
        [-1.5, 0.75, 0.5, 1.0],
        [2.0, 1.0, -0.25, -1.0],
    ])
    a2 = systolic.to_bf16([
        [2.0, -1.0, 0.5, 3.0],
        [1.25, 2.5, -1.5, 0.5],
        [-0.5, 1.0, 2.0, -2.0],
        [3.0, 0.25, -1.0, 1.5],
    ])

    epochs = [
        (weights_epoch0, [a0, a1]),
        (weights_epoch1, [a2]),
    ]
    row_seqs, col_seqs, reload_seq, horizon = systolic.build_ws_schedule(epochs)

    def program():
      # Delay/skew row activations only during COMPUTE; reset skew delays when
      # entering a new phase. We skew row streams continuously because row_seqs
      # pads with 0.0 during the 4 LOAD cycles and 6 DRAIN cycles.
      row_streams = [Stream(seq) for seq in row_seqs]
      skewed_rows = systolic.skew_streams(row_streams, prefix='ws_row')
      col_streams = [Stream(seq) for seq in col_seqs]
      out = systolic.systolic_array(
          skewed_rows,
          col_streams,
          initial_mode=SystolicMode.WEIGHT_STATIONARY,
          reload_weights=Stream(reload_seq),
      )
      return out.deskewed_outputs, out.accumulators, out.control.substate

    deskewed_tr, acc_tr, substate_tr = Scheduler.run(
        program, horizon=horizon, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)

    # Verify merged weight/accumulator registers after epoch 0 load (tick 3) and
    # epoch 1 load.
    loaded_w0 = [[acc_tr[i][j][3] for j in range(4)] for i in range(4)]
    loaded_w1 = [[acc_tr[i][j][-1] for j in range(4)] for i in range(4)]
    self.assertEqual(loaded_w0, weights_epoch0)
    self.assertEqual(loaded_w1, weights_epoch1)

    # Verify substate transitions:
    # LOAD (4) -> COMPUTE (14) -> LOAD (4) -> COMPUTE (10).
    self.assertEqual(
        substate_tr[:5],
        [
            WeightStationaryState.LOAD,
            WeightStationaryState.LOAD,
            WeightStationaryState.LOAD,
            WeightStationaryState.LOAD,
            WeightStationaryState.COMPUTE,
        ],
    )
    self.assertEqual(
        substate_tr[18:23],
        [
            WeightStationaryState.LOAD,
            WeightStationaryState.LOAD,
            WeightStationaryState.LOAD,
            WeightStationaryState.LOAD,
            WeightStationaryState.COMPUTE,
        ],
    )

    actual_matrices = systolic.extract_matrices(deskewed_tr)
    expected_matrices = [
        systolic.reference_matmul_bf16_fp32(a0, weights_epoch0),
        systolic.reference_matmul_bf16_fp32(a1, weights_epoch0),
        systolic.reference_matmul_bf16_fp32(a2, weights_epoch1),
    ]
    self._assert_matrices_equal(actual_matrices, expected_matrices)

  def test_output_stationary_pipelined_products(self):
    a0 = systolic.to_bf16([
        [1.0, 2.0, -1.0, 0.5],
        [3.0, -0.5, 2.0, 1.5],
        [-2.0, 1.0, 0.25, -1.0],
        [0.5, 2.5, -1.5, 2.0],
    ])
    b0 = systolic.to_bf16([
        [2.0, -1.0, 0.5, 1.0],
        [0.5, 1.5, -2.0, 0.25],
        [-1.0, 2.0, 1.0, -0.5],
        [1.5, -0.5, 0.75, 2.0],
    ])
    a1 = systolic.to_bf16([
        [1.0, 2**-5, 2**-5, 2**-5],
        [1.25, -2.25, 0.75, -0.5],
        [0.5, 1.5, -1.25, 2.5],
        [-0.75, 0.25, 1.0, -1.5],
    ])
    b1 = systolic.to_bf16([
        [1.0, -0.5, 2.0, 0.25],
        [2**-5, 1.25, -0.75, 1.5],
        [2**-5, -1.5, 0.5, -2.0],
        [2**-5, 0.75, 1.25, 0.5],
    ])
    a2 = systolic.to_bf16([
        [2.0, 1.0, 0.0, -1.0],
        [-1.0, 0.5, 2.0, 1.0],
        [0.25, -0.5, 1.5, 0.75],
        [1.5, 2.0, -1.0, 0.5],
    ])
    b2 = systolic.to_bf16([
        [0.5, 1.0, -1.5, 2.0],
        [1.0, -2.0, 0.5, -0.5],
        [-0.5, 0.25, 1.0, 1.5],
        [2.0, 1.5, -0.25, 0.5],
    ])

    pairs = [(a0, b0), (a1, b1), (a2, b2)]
    row_seqs, col_seqs, horizon = systolic.build_os_schedule(pairs)

    def program():
      row_streams = [Stream(seq) for seq in row_seqs]
      col_streams = [Stream(seq) for seq in col_seqs]
      skewed_rows = systolic.skew_streams(row_streams, prefix='os_row')
      skewed_cols = systolic.skew_streams(col_streams, prefix='os_col')
      out = systolic.systolic_array(
          skewed_rows,
          skewed_cols,
          initial_mode=SystolicMode.OUTPUT_STATIONARY,
      )
      return out.deskewed_outputs, out.accumulators, out.control.pe_valid

    deskewed_tr, acc_tr, pe_valid_tr = Scheduler.run(
        program, horizon=horizon, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)

    expected_matrices = [
        systolic.reference_matmul_bf16_fp32(a0, b0),
        systolic.reference_matmul_bf16_fp32(a1, b1),
        systolic.reference_matmul_bf16_fp32(a2, b2),
    ]
    deskewed_matrices = systolic.extract_matrices(deskewed_tr)
    pe_matrices = systolic.extract_os_pe_matrices(acc_tr, pe_valid_tr)
    self._assert_matrices_equal(deskewed_matrices, expected_matrices)
    self._assert_matrices_equal(pe_matrices, expected_matrices)

  def test_mode_switching_between_ws_and_os_in_single_run(self):
    w_ws = systolic.to_bf16([
        [1.0, 0.5, -1.0, 2.0],
        [2.0, -1.0, 0.5, 1.0],
        [-0.5, 1.5, 2.0, -1.0],
        [1.0, 2.0, -0.5, 0.5],
    ])
    a_ws = systolic.to_bf16([
        [2.0, 1.0, -1.0, 0.5],
        [0.5, -2.0, 1.5, 1.0],
        [1.0, 0.5, 2.0, -1.5],
        [-1.0, 1.5, 0.5, 2.0],
    ])
    a_os = systolic.to_bf16([
        [1.5, -0.5, 2.0, 1.0],
        [0.5, 2.0, -1.0, 0.5],
        [-1.0, 1.0, 0.5, 2.0],
        [2.0, -1.5, 1.0, -0.5],
    ])
    b_os = systolic.to_bf16([
        [1.0, 2.0, -0.5, 1.5],
        [-1.0, 0.5, 2.0, -1.0],
        [0.5, -1.0, 1.5, 0.5],
        [2.0, 1.0, -1.0, 2.0],
    ])

    ws_rows, ws_cols, _, ws_horizon = systolic.build_ws_schedule(
        [(w_ws, [a_ws])]
    )
    os_rows, os_cols, os_horizon = systolic.build_os_schedule([(a_os, b_os)])
    total_horizon = ws_horizon + os_horizon
    switch_seq = [t == ws_horizon - 1 for t in range(total_horizon)]

    def program():
      is_os_phase = counter() >= ws_horizon
      with reset(counter() == ws_horizon):
        skewed_ws_rows = systolic.skew_streams(
            [Stream(row_w + [0.0] * os_horizon) for row_w in ws_rows],
            prefix='mix_ws_row',
        )
        skewed_os_rows = systolic.skew_streams(
            [Stream([0.0] * ws_horizon + row_o) for row_o in os_rows],
            prefix='mix_os_row',
        )
        skewed_os_cols = systolic.skew_streams(
            [Stream([0.0] * ws_horizon + col_o) for col_o in os_cols],
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
              Stream(ws_cols[j] + [0.0] * os_horizon),
          )
          for j in range(4)
      ]
      out = systolic.systolic_array(
          row_inputs,
          col_inputs,
          initial_mode=SystolicMode.WEIGHT_STATIONARY,
          switch_mode=Stream(switch_seq),
      )
      return out.deskewed_outputs, out.control.mode

    deskewed_tr, mode_tr = Scheduler.run(
        program, horizon=total_horizon, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(
        mode_tr,
        [SystolicMode.WEIGHT_STATIONARY] * ws_horizon
        + [SystolicMode.OUTPUT_STATIONARY] * os_horizon,
    )
    actual_matrices = systolic.extract_matrices(deskewed_tr)
    expected_matrices = [
        systolic.reference_matmul_bf16_fp32(a_ws, w_ws),
        systolic.reference_matmul_bf16_fp32(a_os, b_os),
    ]
    self._assert_matrices_equal(actual_matrices, expected_matrices)

  def test_non_accumulate_override_stores_product_directly(self):
    # When accumulate=False is forced across the array in OS mode, each PE
    # overwrites its accumulator with a * b on every cycle instead of adding.
    # Therefore, at the end of a 4-cycle tile (k = 3), PE (i, j) holds only the
    # final product bf16(A[i, 3]) * bf16(B[3, j]) rather than the sum over k.
    mat_a = systolic.to_bf16([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ])
    mat_b = systolic.to_bf16([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0],
        [0.5, 1.5, -0.5, 2.0],
    ])
    row_seqs, col_seqs, horizon = systolic.build_os_schedule([(mat_a, mat_b)])

    def program():
      row_streams = [Stream(seq) for seq in row_seqs]
      col_streams = [Stream(seq) for seq in col_seqs]
      skewed_rows = systolic.skew_streams(row_streams, prefix='ovr_row')
      skewed_cols = systolic.skew_streams(col_streams, prefix='ovr_col')
      out = systolic.systolic_array(
          skewed_rows,
          skewed_cols,
          initial_mode=SystolicMode.OUTPUT_STATIONARY,
          accumulate=Stream(False),
      )
      return out.deskewed_outputs

    deskewed_tr = Scheduler.run(
        program, horizon=horizon, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    actual_matrices = systolic.extract_matrices(deskewed_tr)
    expected_last_outer_product = [
        [systolic.to_fp32(mat_a[i][3] * mat_b[3][j]) for j in range(4)]
        for i in range(4)
    ]
    self._assert_matrices_equal(actual_matrices, [expected_last_outer_product])

  def test_global_single_cycle_reset_aborts_and_restarts_cleanly(self):
    w_good = systolic.to_bf16([
        [1.0, -1.0, 2.0, 0.5],
        [0.5, 2.0, -0.5, 1.0],
        [2.0, 0.5, 1.0, -1.5],
        [-1.0, 1.5, 0.5, 2.0],
    ])
    a_good = systolic.to_bf16([
        [2.0, 0.5, -1.0, 1.5],
        [-1.0, 2.0, 0.5, -0.5],
        [1.5, -0.5, 2.0, 1.0],
        [0.5, 1.0, -1.5, 2.0],
    ])
    good_rows, good_cols, good_reload, good_horizon = (
        systolic.build_ws_schedule([(w_good, [a_good])])
    )

    # Run 6 ticks of garbage/corrupted activity, assert global single-cycle
    # reset at tick 6, and run the clean schedule starting at tick 6.
    abort_ticks = 6
    total_horizon = abort_ticks + good_horizon
    reset_seq = [t == abort_ticks for t in range(total_horizon)]
    full_rows = [
        systolic.to_bf16([99.0] * abort_ticks) + good_rows[i]
        for i in range(systolic.ARRAY_SIZE)
    ]
    full_cols = [
        systolic.to_bf16([77.0] * abort_ticks) + good_cols[j]
        for j in range(systolic.ARRAY_SIZE)
    ]
    full_reload = [False] * abort_ticks + good_reload

    def program():
      rst = Stream(reset_seq)
      with reset(rst):
        skewed_rows = systolic.skew_streams(
            [Stream(seq) for seq in full_rows], prefix='rst_row'
        )
      col_streams = [Stream(seq) for seq in full_cols]
      out = systolic.systolic_array(
          skewed_rows,
          col_streams,
          initial_mode=SystolicMode.WEIGHT_STATIONARY,
          reload_weights=Stream(full_reload),
          reset_stream=rst,
      )
      return (
          out.deskewed_outputs,
          out.control.substate,
          out.accumulators[0][0],
          out.accumulators[1][0],
      )

    deskewed_tr, substate_tr, acc00_tr, acc10_tr = Scheduler.run(
        program, horizon=total_horizon, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)

    # At tick 6 (when reset_stream=True), the controller immediately restarts in
    # WeightStationaryState.LOAD: PE (0, 0) latches the first clean weight
    # w_good[3][0] while PE (1, 0)'s accumulator resets to 0.0.
    self.assertEqual(substate_tr[abort_ticks], WeightStationaryState.LOAD)
    self.assertEqual(acc00_tr[abort_ticks], w_good[3][0])
    self.assertEqual(acc10_tr[abort_ticks], 0.0)

    # All valid outputs produced after the reset match the clean matrix product.
    post_reset_deskewed = [col[abort_ticks:] for col in deskewed_tr]
    actual_matrices = systolic.extract_matrices(post_reset_deskewed)
    expected_matrices = [systolic.reference_matmul_bf16_fp32(a_good, w_good)]
    self._assert_matrices_equal(actual_matrices, expected_matrices)

  def test_validation_errors(self):
    with self.assertRaisesRegex(ValueError, 'must have 4 rows'):
      systolic.reference_matmul_bf16_fp32([[1.0] * 4] * 3, [[1.0] * 4] * 4)
    with self.assertRaisesRegex(ValueError, 'must have 4 columns'):
      systolic.reference_matmul_bf16_fp32([[1.0] * 3] * 4, [[1.0] * 4] * 4)
    with self.assertRaisesRegex(ValueError, 'non-negative integer'):
      systolic.delay_padded(Stream([1.0, 2.0]), -1)
    with self.assertRaisesRegex(ValueError, 'expects 4 streams'):
      systolic.skew_streams([Stream(1.0), Stream(2.0)])
    with self.assertRaisesRegex(ValueError, 'expects 4 streams'):
      systolic.deskew_streams([Stream(1.0), Stream(2.0)])
    with self.assertRaisesRegex(ValueError, 'epochs must be non-empty'):
      systolic.build_ws_schedule([])
    with self.assertRaisesRegex(ValueError, 'matrix_pairs must be non-empty'):
      systolic.build_os_schedule([])


if __name__ == '__main__':
  absltest.main()
