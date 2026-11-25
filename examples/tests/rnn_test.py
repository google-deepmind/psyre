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
"""Unit tests for the Psyre neural network model."""

from collections.abc import Sequence
import logging
import math
import random
from typing import Any

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    Absent,
    Semantics,
    logger,
    when,
)
from examples.src import rnn
from psyre.match import (
    case,
    match,
)
from psyre.scheduler import Scheduler
from psyre.stream import Stream
from tests import test_utils
from psyre.trace import (
    is_absent,
    is_present,
    present_values,
)

# pylint: enable=g-importing-member,g-multiple-import

_LEARNING_STEPS = 20
_RunOutput = tuple[list[Any], Any, list[Any] | None, list[Any] | None]


def _build_sample_dataset() -> list[tuple[list[float], float]]:
  return [
      ([0.5, 0.2, 0.4, 0.1, 0.3], 2.0),
      ([0.2, 0.6, 0.1, 0.5, 0.4], 2.1),
      ([0.4, 0.3, 0.5, 0.2, 0.6], 2.2),
      ([0.3, 0.4, 0.2, 0.6, 0.3], 2.05),
  ]


def reference_rnn_perceptron(
    x_seq: Sequence[float],
    params: Any,
    num_cells: int | None = None,
) -> float:
  """Computes exact mathematical RNN + ReLU perceptron prediction."""
  dim = len(x_seq)
  cells = dim if num_cells is None else num_cells
  h_grid: list[list[float]] = []
  for k in range(cells):
    row: list[float] = []
    for t in range(dim):
      h_prev = 0.0 if k == 0 or t == 0 else h_grid[k - 1][t - 1]
      row.append(
          math.tanh(
              params.rnn.wxh * x_seq[t]
              + params.rnn.whh * h_prev
              + params.rnn.bh
          )
      )
    h_grid.append(row)
  y_rnn = [h_grid[min(t, cells - 1)][t] for t in range(dim)]
  dot_acc = params.per.bh[0] + math.sumprod(y_rnn, params.per.wxh)
  return max(0.0, dot_acc)


def _reference_eval_loss(
    x_seq: Sequence[float],
    y_target: float,
    params: Any,
    num_cells: int | None = None,
) -> float:
  """Computes exact mathematical RNN + ReLU perceptron squared loss."""
  y_pred = reference_rnn_perceptron(x_seq, params, num_cells=num_cells)
  return (y_pred - y_target) ** 2


class ModelTest(absltest.TestCase):

  MOD: Any = rnn

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def _ticks(self, num_passes: int) -> int:
    """Converts a pass count into a tick horizon."""
    return num_passes * self.MOD.DIMENSION

  def _pass_params(self, traces: Any, pass_idx: int) -> Any:
    """Extracts and validates the ModelParams slice for pass pass_idx."""
    dim = self.MOD.DIMENSION
    start = self._ticks(pass_idx)
    return self.MOD.extract_params(
        traces.map(lambda seq: seq[start : start + dim])
    )

  def test_simple_model(self):
    mod = self.MOD
    self.assertEqual(mod.DIMENSION, 5)
    self.assertEqual(mod.DEFAULT_MAX_STEPS, 3)
    default_hp = mod.HyperParams()
    self.assertEqual(default_hp.lr, mod.DEFAULT_LEARNING_RATE)
    self.assertEqual(default_hp.max_steps, mod.DEFAULT_MAX_STEPS)
    self.assertEqual(default_hp.holdout_index, 0)
    with self.assertRaisesRegex(ValueError, 'max_steps must be >= 1'):
      mod.HyperParams(max_steps=0)
    for semantics in (Semantics.KAHN, Semantics.SYNC):
      with self.subTest(semantics=semantics):
        y_pred_fwd, loss_fwd, y_pred, loss_bwd, param_traces = Scheduler.run(
            mod.simple_model, horizon=mod.DIMENSION, semantics=semantics
        )
        mod.extract_params(param_traces)

        self.assertEqual(y_pred_fwd, y_pred)
        self.assertEqual(loss_fwd, loss_bwd)
        param_traces.map(lambda trace: self.assertLen(trace, mod.DIMENSION))
        node_names = [node.name for node in Scheduler.graph.nodes]
        for expected_pfx in (
            'rnn_layer/',
            'per_layer/',
            'grad_rnn_layer/',
            'grad_per_layer/',
        ):
          self.assertTrue(
              any(name.startswith(expected_pfx) for name in node_names),
              msg=f'Expected {expected_pfx} prefix in {node_names}',
          )

  def test_hierarchical_training_random_init_leave_one_out_and_inference(self):
    mod = self.MOD
    samples = _build_sample_dataset()

    extracted = mod.dataset_streams(samples)
    self.assertLen(extracted, len(samples))
    for item in extracted:
      self.assertIsInstance(item.x, Stream)
      self.assertIsInstance(item.y, Stream)

    # Run 2 extra passes past the 20-step bound to verify bounded updates.
    train_passes = 1 + _LEARNING_STEPS + 2
    train_horizon = self._ticks(train_passes)
    infer_passes = 1 + len(samples) + 1
    infer_horizon = self._ticks(infer_passes)

    def run_train_and_eval(
        ds: Sequence[tuple[Sequence[float], float]],
    ) -> tuple[_RunOutput, _RunOutput, Any]:
      train_out: _RunOutput = Scheduler.run(
          mod.hierarchical_model,
          dataset=ds,
          initial_mode=mod.Mode.TRAINING,
          train_init_state=mod.TrainingState.INIT_RANDOM,
          hyperparams=mod.HyperParams(
              lr=0.01,
              max_steps=_LEARNING_STEPS,
              holdout_index=3,
          ),
          horizon=train_horizon,
      )
      trained = mod.normalize_params(mod.extract_params(train_out[1]))
      infer_out: _RunOutput = Scheduler.run(
          mod.hierarchical_model,
          dataset=ds,
          params=trained,
          initial_mode=mod.Mode.INFERENCE,
          horizon=infer_horizon,
      )
      return train_out, infer_out, trained

    saved_rng = random.getstate()
    try:
      random.seed(7)
      train_out, infer_out, saved_params = run_train_and_eval(samples)
    finally:
      random.setstate(saved_rng)
    train_modes, train_traces, train_losses, train_eval_losses = train_out
    infer_modes, infer_traces, infer_train_losses, eval_losses = infer_out

    self.assertEqual(train_modes, [mod.Mode.TRAINING] * train_horizon)
    self.assertIsNone(train_eval_losses)
    assert train_losses is not None
    assert train_traces is not None

    # Bounded to exactly 20 BPTT learning steps in Mode.TRAINING.
    present_train_losses = present_values(train_losses)
    self.assertLen(present_train_losses, _LEARNING_STEPS)
    expected_train_ticks = {
        self._ticks(step_idx + 2) - 1 for step_idx in range(_LEARNING_STEPS)
    }
    for tick in range(train_horizon):
      self.assertEqual(
          is_present(train_losses, tick), tick in expected_train_ticks
      )
    for step_loss in present_train_losses:
      self.assertIsInstance(step_loss, float)

    initial_avg = sum(float(v) for v in present_train_losses[:3]) / 3.0
    final_avg = sum(float(v) for v in present_train_losses[-3:]) / 3.0
    self.assertLess(final_avg, initial_avg)

    # Parameters stay constant past step 20.
    step_20_slice = self._pass_params(train_traces, _LEARNING_STEPS)
    step_21_slice = self._pass_params(train_traces, _LEARNING_STEPS + 1)
    step_22_slice = self._pass_params(train_traces, _LEARNING_STEPS + 2)
    for pass_idx in range(train_passes):
      self._pass_params(train_traces, pass_idx)
    self.assertEqual(step_20_slice, step_21_slice)
    self.assertEqual(step_20_slice, step_22_slice)
    self.assertEqual(saved_params, step_20_slice)
    saved_params.rnn.map(lambda val: self.assertIsInstance(val, float))
    saved_params.per.map(lambda vec: self.assertLen(vec, mod.DIMENSION))

    # InferenceState.EVAL evaluates loss on every input stream in the dataset.
    self.assertEqual(infer_modes, [mod.Mode.INFERENCE] * infer_horizon)
    self.assertIsNone(infer_traces)
    self.assertIsNone(infer_train_losses)
    assert eval_losses is not None
    expected_eval_ticks = {
        self._ticks(idx + 2) - 1 for idx in range(len(samples))
    }
    for tick in range(infer_horizon):
      self.assertEqual(
          is_present(eval_losses, tick), tick in expected_eval_ticks
      )
    recorded_eval_losses = [float(val) for val in present_values(eval_losses)]
    self.assertLen(recorded_eval_losses, len(samples))
    expected_eval_losses = [
        _reference_eval_loss(x_seq, y_target, saved_params)
        for x_seq, y_target in samples
    ]
    for actual, expected in zip(
        recorded_eval_losses, expected_eval_losses, strict=True
    ):
      self.assertAlmostEqual(actual, expected, places=9)

    # Leave-one-out check: perturbing only the held-out sample (index 3) leaves
    # all 20 training steps and trained parameters unchanged.
    perturbed_samples = list(samples)
    perturbed_samples[3] = (perturbed_samples[3][0], 10.0)

    saved_rng = random.getstate()
    try:
      random.seed(7)
      alt_train_out, alt_infer_out, alt_params = run_train_and_eval(
          perturbed_samples
      )
    finally:
      random.setstate(saved_rng)
    alt_train_losses = present_values(alt_train_out[2])
    self.assertEqual(alt_train_losses, present_train_losses)
    self.assertEqual(alt_params, saved_params)
    alt_eval_losses = [float(val) for val in present_values(alt_infer_out[3])]
    self.assertEqual(alt_eval_losses[:3], recorded_eval_losses[:3])
    self.assertNotAlmostEqual(alt_eval_losses[3], recorded_eval_losses[3])

  def test_hierarchical_mode_switch_and_load_init_state(self):
    mod = self.MOD
    dim = mod.DIMENSION
    samples = _build_sample_dataset()

    initial_params = mod.ModelParams(
        rnn=mod.RnnParams(wxh=0.25, whh=0.15, bh=0.1),
        per=mod.PerceptronParams(
            wxh=[0.2] * dim,
            bh=[0.3] + [0.0] * (dim - 1),
        ),
    )

    infer_passes = 1 + len(samples)
    train_passes = 1 + mod.DEFAULT_MAX_STEPS + 2
    infer_ticks = self._ticks(infer_passes)
    train_ticks = self._ticks(train_passes)
    total_horizon = infer_ticks + train_ticks

    def main() -> tuple[tuple[Stream, Any, Stream, Stream], Stream]:
      switch = when(Stream(True), Stream.once(infer_ticks - 1))
      out = mod.hierarchical_model(
          dataset=samples,
          params=initial_params,
          initial_mode=mod.Mode.INFERENCE,
          switch_mode=switch,
          train_init_state=mod.TrainingState.INIT_LOAD,
          hyperparams=mod.HyperParams(lr=0.01, holdout_index=0),
      )
      return out, switch

    (modes, param_traces, train_losses, eval_losses), switch_trace = (
        Scheduler.run(main, horizon=total_horizon)
    )

    expected_switch = [tick == infer_ticks - 1 for tick in range(total_horizon)]
    self.assertEqual(switch_trace, expected_switch)

    eval_done_ticks = {self._ticks(idx + 2) - 1 for idx in range(len(samples))}
    train_done_ticks = {
        infer_ticks + self._ticks(idx + 2) - 1
        for idx in range(mod.DEFAULT_MAX_STEPS)
    }
    for tick in range(total_horizon):
      self.assertEqual(is_present(eval_losses, tick), tick in eval_done_ticks)
      self.assertEqual(is_present(train_losses, tick), tick in train_done_ticks)
      self.assertEqual(
          is_absent(param_traces.rnn.wxh, tick), tick < infer_ticks
      )

    expected_modes = [mod.Mode.INFERENCE] * infer_ticks + [
        mod.Mode.TRAINING
    ] * train_ticks
    self.assertEqual(modes, expected_modes)

    # First phase (Mode.INFERENCE) evaluates initial_params on all 4 inputs.
    pre_eval_losses = [float(val) for val in present_values(eval_losses)]
    self.assertLen(pre_eval_losses, len(samples))
    for actual, (x_seq, y_target) in zip(pre_eval_losses, samples, strict=True):
      self.assertAlmostEqual(
          actual,
          _reference_eval_loss(x_seq, y_target, initial_params),
          places=9,
      )

    # Second phase (Mode.TRAINING with INIT_LOAD) loads initial_params and runs
    # DEFAULT_MAX_STEPS bounded BPTT steps.
    self.assertEqual(
        self._pass_params(param_traces, infer_passes), initial_params
    )

    present_train_losses = present_values(train_losses)
    self.assertLen(present_train_losses, mod.DEFAULT_MAX_STEPS)

    trained_params = mod.extract_params(param_traces)
    self.assertEqual(
        self._pass_params(param_traces, infer_passes + mod.DEFAULT_MAX_STEPS),
        trained_params,
    )
    self.assertEqual(
        self._pass_params(
            param_traces, infer_passes + mod.DEFAULT_MAX_STEPS + 1
        ),
        trained_params,
    )
    self.assertNotEqual(trained_params, initial_params)
    pre_train_subset = [
        loss_val for idx, loss_val in enumerate(pre_eval_losses) if idx != 0
    ]
    post_train_subset = [
        _reference_eval_loss(x_seq, y_target, trained_params)
        for idx, (x_seq, y_target) in enumerate(samples)
        if idx != 0
    ]
    self.assertLess(sum(post_train_subset), sum(pre_train_subset))

    # Verify Absent-elision equivalence under KAHN for the full hierarchical
    # automaton (mode switch, nested training/inference automata, and BPTT).
    (k_modes, k_params, k_train_losses, k_eval_losses), k_switch = (
        Scheduler.run(main, horizon=total_horizon, semantics=Semantics.KAHN)
    )
    self.assertFalse(Scheduler.deadlocked)
    for k_tr in (k_modes, k_switch, k_train_losses, k_eval_losses):
      test_utils.assert_no_sentinels(self, k_tr)
    self.assertEqual(k_modes, modes)
    self.assertEqual(k_switch, switch_trace)
    self.assertEqual(k_train_losses, present_train_losses)
    self.assertEqual(k_eval_losses, pre_eval_losses)
    k_params.map(
        lambda k_tr, s_tr: test_utils.assert_absent_bottom_elision(
            self, s_tr, k_tr
        ),
        param_traces,
    )

  def test_normalize_params_requires_shared_rnn_scalars(self):
    mod = self.MOD
    bad_params = mod.ModelParams(
        rnn=mod.RnnParams(wxh=[0.1] * mod.DIMENSION, whh=0.2, bh=0.05),
        per=mod.PerceptronParams(wxh=0.25, bh=0.1),
    )
    with self.assertRaisesRegex(ValueError, 'Shared parameter'):
      mod.normalize_params(bad_params)

  def test_inference_machine_requires_params_under_match_scope(self):
    mod = self.MOD
    samples = _build_sample_dataset()

    def run_matched_inference_without_params():
      with match(True) as m:
        with case(True):
          m.out = mod.inference_machine(
              dataset=samples,
              params=None,
          )
        with case(False):
          m.out = 0
      return m.out

    with self.assertRaisesRegex(
        ValueError, 'params is required for inference_machine'
    ):
      Scheduler.run(
          run_matched_inference_without_params,
          horizon=mod.DIMENSION,
      )
    with self.assertRaisesRegex(ValueError, 'dataset must be non-empty'):
      Scheduler.run(
          lambda: mod.inference_machine(
              dataset=[],
              params=mod.ModelParams(
                  rnn=mod.RnnParams(wxh=0.1, whh=0.1, bh=0.0),
                  per=mod.PerceptronParams(wxh=[0.1] * mod.DIMENSION, bh=0.0),
              ),
          ),
          horizon=mod.DIMENSION,
      )

  def test_relu(self):
    mod = self.MOD

    def run_relu():
      scalar_neg = mod.relu(-2.5)
      scalar_pos = mod.relu(1.75)
      self.assertIsInstance(scalar_neg, Stream)
      self.assertIsInstance(scalar_pos, Stream)
      cond = Stream([True, False, True, True])
      return (
          scalar_neg,
          scalar_pos,
          mod.relu(Stream([-2.5, 0.0, 1.75, -0.5])),
          mod.relu(when(cond, Stream([-2.5, 1.0, 1.75, 0.0]))),
      )

    scalar_neg, scalar_pos, y, sparse = Scheduler.run(run_relu, horizon=4)
    self.assertEqual(scalar_neg, [0.0] * 4)
    self.assertEqual(scalar_pos, [1.75] * 4)
    self.assertEqual(y, [0.0, 0.0, 1.75, 0.0])
    self.assertEqual(sparse, [0.0, Absent, 1.75, 0.0])

  def test_grad_rnn_layer_backpropagates_from_y(self):
    mod = self.MOD
    dim = mod.DIMENSION
    x_seq = [0.6, -0.4, 0.3, -0.2, 0.5]
    dy_seq = [0.1, -0.25, 0.4, 0.15, -0.3]
    wxh, whh, bh = 0.35, -0.45, 0.12

    y_expected: list[float] = []
    h_prev_seq: list[float] = []
    h_prev = 0.0
    for x_val in x_seq:
      h_prev_seq.append(h_prev)
      h_prev = math.tanh(wxh * x_val + whh * h_prev + bh)
      y_expected.append(h_prev)

    dz_expected = [0.0] * dim
    dh_next = 0.0
    for idx in range(dim - 1, -1, -1):
      dz_val = (dy_seq[idx] + dh_next) * (1.0 - y_expected[idx] ** 2)
      dz_expected[idx] = dz_val
      dh_next = dz_val * whh
    dx_expected = [dz_val * wxh for dz_val in dz_expected]
    dwxh_expected = [
        dz_val * x_val for dz_val, x_val in zip(dz_expected, x_seq, strict=True)
    ]
    dwhh_expected = [
        dz_val * hp_val
        for dz_val, hp_val in zip(dz_expected, h_prev_seq, strict=True)
    ]

    def run_grad_rnn():
      rnn_p = mod.RnnParams(wxh=Stream(wxh), whh=Stream(whh), bh=Stream(bh))
      y, dx, drnn = mod.grad_rnn_layer(Stream(x_seq), rnn_p, lambda: dy_delayed)
      dy_delayed = Stream(dy_seq)
      return y, dx, drnn

    for semantics in (Semantics.KAHN, Semantics.SYNC):
      with self.subTest(semantics=semantics):
        y_trace, dx_trace, drnn_traces = Scheduler.run(
            run_grad_rnn, horizon=dim, semantics=semantics
        )
        assert Scheduler.graph is not None
        self.assertEmpty(
            Scheduler.graph.consumers(Scheduler.graph.output[0])
        )
        self.assertSequenceAlmostEqual(y_trace, y_expected, places=12)
        self.assertSequenceAlmostEqual(dx_trace, dx_expected, places=12)
        self.assertSequenceAlmostEqual(
            drnn_traces.wxh, dwxh_expected, places=12
        )
        self.assertSequenceAlmostEqual(
            drnn_traces.whh, dwhh_expected, places=12
        )
        self.assertSequenceAlmostEqual(drnn_traces.bh, dz_expected, places=12)

  def test_variable_num_cells_and_perceptron_gradients_match_finite_diff(self):
    mod = self.MOD
    dim = mod.DIMENSION
    x_seq = [0.6, -0.4, 0.3, -0.2, 0.5]
    y_target = 1.75
    params = mod.ModelParams(
        rnn=mod.RnnParams(wxh=0.35, whh=-0.45, bh=0.12),
        per=mod.PerceptronParams(
            wxh=[0.4, -0.3, 0.5, 0.25, 0.6],
            bh=[0.8, 0.0, 0.0, 0.0, 0.0],
        ),
    )

    def make_grad_circuit(nc: int):
      def circuit():
        p_streams = mod.ModelParams(
            rnn=params.rnn.map(Stream),
            per=params.per.map(Stream),
        )
        x_s = Stream(x_seq)
        y_fwd = mod.forward(x_s, p_streams, num_cells=nc)
        y_gm, loss_gm, _ = mod.grad_model(
            x_s, y_target, p_streams, num_cells=nc
        )
        dloss = Stream([0.0] * (dim - 1) + [1.0])
        y_bwd, dx, grads = mod.grad_rnn_perceptron(
            x_s,
            p_streams,
            lambda: 2.0 * (y_bwd - y_target) * dloss,
            num_cells=nc,
        )
        return y_fwd, y_bwd, y_gm, loss_gm, (y_bwd - y_target) ** 2, dx, grads

      return circuit

    eps = 1e-6
    for num_cells in (3, 4, 5):
      with self.subTest(num_cells=num_cells):
        y_fwd, y_bwd, y_gm, loss_gm, loss_trace, dx_trace, grad_traces = (
            Scheduler.run(make_grad_circuit(num_cells), horizon=dim)
        )
        self.assertSequenceAlmostEqual(y_fwd, y_bwd, places=12)
        self.assertSequenceAlmostEqual(y_fwd, y_gm, places=12)
        self.assertSequenceAlmostEqual(loss_trace, loss_gm, places=12)
        self.assertAlmostEqual(
            loss_trace[-1],
            _reference_eval_loss(x_seq, y_target, params, num_cells),
            places=12,
        )

        # Check RNN shared scalar parameter gradients (sum over pass).
        for field in ('wxh', 'whh', 'bh'):
          base_val = getattr(params.rnn, field)
          p_plus = params._replace(
              rnn=params.rnn._replace(**{field: base_val + eps})
          )
          p_minus = params._replace(
              rnn=params.rnn._replace(**{field: base_val - eps})
          )
          fd = (
              _reference_eval_loss(x_seq, y_target, p_plus, num_cells)
              - _reference_eval_loss(x_seq, y_target, p_minus, num_cells)
          ) / (2.0 * eps)
          self.assertAlmostEqual(
              sum(getattr(grad_traces.rnn, field)), fd, places=6
          )

        # Check perceptron vector weight gradients & bias gradient.
        for idx in range(dim):
          w_plus = list(params.per.wxh)
          w_plus[idx] += eps
          w_minus = list(params.per.wxh)
          w_minus[idx] -= eps
          fd_w = (
              _reference_eval_loss(
                  x_seq,
                  y_target,
                  params._replace(per=params.per._replace(wxh=w_plus)),
                  num_cells,
              )
              - _reference_eval_loss(
                  x_seq,
                  y_target,
                  params._replace(per=params.per._replace(wxh=w_minus)),
                  num_cells,
              )
          ) / (2.0 * eps)
          self.assertAlmostEqual(grad_traces.per.wxh[idx], fd_w, places=6)

        bh_plus = [params.per.bh[0] + eps] + list(params.per.bh[1:])
        bh_minus = [params.per.bh[0] - eps] + list(params.per.bh[1:])
        fd_bh = (
            _reference_eval_loss(
                x_seq,
                y_target,
                params._replace(per=params.per._replace(bh=bh_plus)),
                num_cells,
            )
            - _reference_eval_loss(
                x_seq,
                y_target,
                params._replace(per=params.per._replace(bh=bh_minus)),
                num_cells,
            )
        ) / (2.0 * eps)
        self.assertAlmostEqual(grad_traces.per.bh[0], fd_bh, places=6)
        self.assertEqual(grad_traces.per.bh[1:], [0.0] * (dim - 1))

        # Check input gradient dx[0..DIMENSION-1].
        for idx in range(dim):
          x_plus = list(x_seq)
          x_plus[idx] += eps
          x_minus = list(x_seq)
          x_minus[idx] -= eps
          fd_x = (
              _reference_eval_loss(x_plus, y_target, params, num_cells)
              - _reference_eval_loss(x_minus, y_target, params, num_cells)
          ) / (2.0 * eps)
          self.assertAlmostEqual(dx_trace[idx], fd_x, places=6)

  def test_normalize_and_extract_params_dimension_and_type_guards(self):
    mod = self.MOD
    p3 = mod.ModelParams(
        rnn=mod.RnnParams(wxh=0.2, whh=-0.1, bh=0.05),
        per=mod.PerceptronParams(wxh=[0.1, 0.2, 0.3], bh=0.5),
    )
    norm = mod.normalize_params(p3, dimension=3)
    self.assertEqual(norm.per.wxh, [0.1, 0.2, 0.3])
    self.assertEqual(norm.per.bh, [0.5, 0.5, 0.5])

    traces = mod.ModelParams(
        rnn=mod.RnnParams(wxh=[0.2] * 3, whh=[-0.1] * 3, bh=[0.05] * 3),
        per=mod.PerceptronParams(wxh=[0.1, 0.2, 0.3], bh=[0.5, 0.5, 0.5]),
    )
    self.assertEqual(mod.extract_params(traces, dimension=3), norm)

    bad_bool = p3._replace(rnn=mod.RnnParams(wxh=True, whh=0.0, bh=0.0))
    with self.assertRaisesRegex(TypeError, 'not bool'):
      mod.normalize_params(bad_bool, dimension=3)
    bad_str = p3._replace(per=mod.PerceptronParams(wxh='abc', bh=0.0))
    with self.assertRaisesRegex(TypeError, 'not str'):
      mod.normalize_params(bad_str, dimension=3)
    with self.assertRaisesRegex(ValueError, 'could not convert|invalid'):
      mod.dataset_streams([([0.0] * mod.DIMENSION, 'bad')])


if __name__ == '__main__':
  absltest.main()
