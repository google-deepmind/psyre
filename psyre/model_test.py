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
from psyre.match import (
    case,
    match,
)
from psyre.model import (
    DEFAULT_MAX_STEPS,
    DIMENSION,
    Mode,
    ModelParams,
    ParamTraces,
    ParamValues,
    PerceptronParams,
    RnnParams,
    TrainingState,
    dataset_streams,
    extract_params,
    grad_rnn_layer,
    hierarchical_model,
    inference_machine,
    normalize_params,
    relu,
    simple_model,
)
from psyre.psyre import (
    Absent,
    Scheduler,
    Semantics,
    Stream,
    counter,
    is_absent,
    is_present,
    logger,
    once_stream,
    present_values,
    when,
)

# pylint: enable=g-importing-member,g-multiple-import

_LEARNING_STEPS = 20
_RunOutput = tuple[
    list[Any], ParamTraces | None, list[Any] | None, list[Any] | None
]


def _ticks(num_passes: int) -> int:
  """Converts a pass count into a tick horizon."""
  return num_passes * DIMENSION


def _pass_params(traces: ParamTraces, pass_idx: int) -> ParamValues:
  """Extracts and validates the ModelParams slice for pass pass_idx."""
  start = _ticks(pass_idx)
  return extract_params(traces.map(lambda seq: seq[start : start + DIMENSION]))


def _build_sample_dataset() -> list[tuple[list[float], float]]:
  return [
      ([0.5, 0.2, 0.4, 0.1, 0.3], 2.0),
      ([0.2, 0.6, 0.1, 0.5, 0.4], 2.1),
      ([0.4, 0.3, 0.5, 0.2, 0.6], 2.2),
      ([0.3, 0.4, 0.2, 0.6, 0.3], 2.05),
  ]


def _reference_eval_loss(
    x_seq: Sequence[float],
    y_target: float,
    params: ParamValues,
) -> float:
  """Computes exact mathematical RNN + ReLU perceptron squared loss."""
  h_prev = 0.0
  dot_acc = params.per.bh[0]
  for x_val, w_val in zip(x_seq, params.per.wxh, strict=True):
    h_cur = math.tanh(
        params.rnn.wxh * x_val + params.rnn.whh * h_prev + params.rnn.bh
    )
    dot_acc += h_cur * w_val
    h_prev = h_cur
  y_pred = max(0.0, dot_acc)
  return (y_pred - y_target) ** 2


class ModelTest(absltest.TestCase):

  def test_simple_model(self):
    logger.setLevel(logging.WARNING)
    self.assertEqual(DIMENSION, 5)
    self.assertEqual(DEFAULT_MAX_STEPS, 3)
    for semantics in (Semantics.KAHN, Semantics.SYNC):
      with self.subTest(semantics=semantics):
        y_pred_fwd, loss_fwd, y_pred, loss_bwd, param_traces = Scheduler.run(
            simple_model, horizon=DIMENSION, semantics=semantics
        )
        updated = extract_params(param_traces)
        print('Forward y_pred:', y_pred_fwd[-1])
        print('Forward loss:', loss_fwd[-1])
        print('Grad y_pred:', y_pred[-1])
        print('Grad loss:', loss_bwd[-1])
        print('Updated params:', updated)

        self.assertEqual(y_pred_fwd, y_pred)
        self.assertEqual(loss_fwd, loss_bwd)
        param_traces.map(lambda trace: self.assertLen(trace, DIMENSION))
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
    logger.setLevel(logging.WARNING)
    samples = _build_sample_dataset()

    extracted = dataset_streams(samples)
    self.assertLen(extracted, len(samples))
    for item in extracted:
      self.assertIsInstance(item.x, Stream)
      self.assertIsInstance(item.y, Stream)

    # Run 2 extra passes past the 20-step bound to verify bounded updates.
    train_passes = 1 + _LEARNING_STEPS + 2
    train_horizon = _ticks(train_passes)
    infer_passes = 1 + len(samples) + 1
    infer_horizon = _ticks(infer_passes)

    def run_train_and_eval(
        ds: Sequence[tuple[Sequence[float], float]],
    ) -> tuple[_RunOutput, _RunOutput, ParamValues]:
      train_out: _RunOutput = Scheduler.run(
          hierarchical_model,
          dataset=ds,
          initial_mode=Mode.TRAINING,
          train_init_state=TrainingState.INIT_RANDOM,
          max_steps=_LEARNING_STEPS,
          holdout_index=3,
          lr=0.01,
          horizon=train_horizon,
          semantics=Semantics.SYNC,
      )
      trained = normalize_params(extract_params(train_out[1]))
      infer_out: _RunOutput = Scheduler.run(
          hierarchical_model,
          dataset=ds,
          params=trained,
          initial_mode=Mode.INFERENCE,
          horizon=infer_horizon,
          semantics=Semantics.SYNC,
      )
      return train_out, infer_out, trained

    random.seed(7)
    train_out, infer_out, saved_params = run_train_and_eval(samples)
    train_modes, train_traces, train_losses, train_eval_losses = train_out
    infer_modes, infer_traces, infer_train_losses, eval_losses = infer_out

    self.assertEqual(train_modes, [Mode.TRAINING] * train_horizon)
    self.assertIsNone(train_eval_losses)
    assert train_losses is not None
    assert train_traces is not None

    # Bounded to exactly 20 BPTT learning steps in Mode.TRAINING.
    present_train_losses = present_values(train_losses)
    self.assertLen(present_train_losses, _LEARNING_STEPS)
    expected_train_ticks = {
        _ticks(step_idx + 2) - 1 for step_idx in range(_LEARNING_STEPS)
    }
    for tick in range(train_horizon):
      self.assertEqual(
          is_present(train_losses, tick), tick in expected_train_ticks
      )
    for step_idx, step_loss in enumerate(present_train_losses):
      self.assertIsInstance(step_loss, float)
      expected_tick = _ticks(step_idx + 2) - 1
      self.assertEqual(train_losses[expected_tick], step_loss)

    initial_avg = sum(float(v) for v in present_train_losses[:3]) / 3.0
    final_avg = sum(float(v) for v in present_train_losses[-3:]) / 3.0
    self.assertLess(final_avg, initial_avg)

    # Parameters stay constant past step 20.
    step_20_slice = _pass_params(train_traces, _LEARNING_STEPS)
    step_21_slice = _pass_params(train_traces, _LEARNING_STEPS + 1)
    step_22_slice = _pass_params(train_traces, _LEARNING_STEPS + 2)
    for pass_idx in range(train_passes):
      _pass_params(train_traces, pass_idx)
    self.assertEqual(step_20_slice, step_21_slice)
    self.assertEqual(step_20_slice, step_22_slice)
    self.assertEqual(saved_params, step_20_slice)
    saved_params.rnn.map(lambda val: self.assertIsInstance(val, float))
    saved_params.per.map(lambda vec: self.assertLen(vec, DIMENSION))

    # InferenceState.EVAL evaluates loss on every input stream in the dataset.
    self.assertEqual(infer_modes, [Mode.INFERENCE] * infer_horizon)
    self.assertIsNone(infer_traces)
    self.assertIsNone(infer_train_losses)
    assert eval_losses is not None
    expected_eval_ticks = {_ticks(idx + 2) - 1 for idx in range(len(samples))}
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

    random.seed(7)
    alt_train_out, alt_infer_out, alt_params = run_train_and_eval(
        perturbed_samples
    )
    alt_train_losses = present_values(alt_train_out[2])
    self.assertEqual(alt_train_losses, present_train_losses)
    self.assertEqual(alt_params, saved_params)
    alt_eval_losses = [float(val) for val in present_values(alt_infer_out[3])]
    self.assertEqual(alt_eval_losses[:3], recorded_eval_losses[:3])
    self.assertNotAlmostEqual(alt_eval_losses[3], recorded_eval_losses[3])

  def test_hierarchical_mode_switch_and_load_init_state(self):
    logger.setLevel(logging.WARNING)
    samples = _build_sample_dataset()

    initial_params = ModelParams(
        rnn=RnnParams(wxh=0.25, whh=0.15, bh=0.1),
        per=PerceptronParams(
            wxh=[0.2] * DIMENSION,
            bh=[0.3] + [0.0] * (DIMENSION - 1),
        ),
    )

    infer_passes = 1 + len(samples)
    train_passes = 1 + DEFAULT_MAX_STEPS + 2
    infer_ticks = _ticks(infer_passes)
    train_ticks = _ticks(train_passes)
    total_horizon = infer_ticks + train_ticks

    def main() -> tuple[tuple[Stream, Any, Stream, Stream], Stream]:
      pass_end = (counter() % DIMENSION) == (DIMENSION - 1)
      with match(pass_end) as switch_m:
        with case(True):
          switch_m.out = once_stream(len(samples))
        with case(False):
          switch_m.out = False
      switch = switch_m.out
      out = hierarchical_model(
          dataset=samples,
          params=initial_params,
          initial_mode=Mode.INFERENCE,
          switch_mode=switch,
          train_init_state=TrainingState.INIT_LOAD,
          holdout_index=0,
          lr=0.01,
      )
      return out, switch

    (modes, param_traces, train_losses, eval_losses), switch_trace = (
        Scheduler.run(main, horizon=total_horizon, semantics=Semantics.SYNC)
    )

    expected_switch = [tick == infer_ticks - 1 for tick in range(total_horizon)]
    self.assertEqual(switch_trace, expected_switch)

    eval_done_ticks = {_ticks(idx + 2) - 1 for idx in range(len(samples))}
    train_done_ticks = {
        infer_ticks + _ticks(idx + 2) - 1 for idx in range(DEFAULT_MAX_STEPS)
    }
    for tick in range(total_horizon):
      self.assertEqual(is_present(eval_losses, tick), tick in eval_done_ticks)
      self.assertEqual(is_present(train_losses, tick), tick in train_done_ticks)
      self.assertEqual(
          is_absent(param_traces.rnn.wxh, tick), tick < infer_ticks
      )

    expected_modes = [Mode.INFERENCE] * infer_ticks + [
        Mode.TRAINING
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
    self.assertEqual(_pass_params(param_traces, infer_passes), initial_params)

    present_train_losses = present_values(train_losses)
    self.assertLen(present_train_losses, DEFAULT_MAX_STEPS)

    trained_params = extract_params(param_traces)
    self.assertEqual(
        _pass_params(param_traces, infer_passes + DEFAULT_MAX_STEPS),
        trained_params,
    )
    self.assertEqual(
        _pass_params(param_traces, infer_passes + DEFAULT_MAX_STEPS + 1),
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

  def test_normalize_params_requires_shared_rnn_scalars(self):
    bad_params = ModelParams(
        rnn=RnnParams(wxh=[0.1] * DIMENSION, whh=0.2, bh=0.05),
        per=PerceptronParams(wxh=0.25, bh=0.1),
    )
    with self.assertRaisesRegex(ValueError, 'Shared parameter'):
      normalize_params(bad_params)

  def test_inference_machine_requires_params_under_match_scope(self):
    logger.setLevel(logging.WARNING)
    samples = _build_sample_dataset()

    def run_matched_inference_without_params():
      with match(True) as m:
        with case(True):
          m.out = inference_machine(
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
          horizon=DIMENSION,
          semantics=Semantics.SYNC,
      )

  def test_relu(self):
    logger.setLevel(logging.WARNING)

    def run_relu():
      scalar_neg = relu(-2.5)
      scalar_pos = relu(1.75)
      self.assertIsInstance(scalar_neg, Stream)
      self.assertIsInstance(scalar_pos, Stream)
      cond = Stream([True, False, True, True])
      return (
          scalar_neg,
          scalar_pos,
          relu(Stream([-2.5, 0.0, 1.75, -0.5])),
          relu(when(cond, Stream([-2.5, 1.0, 1.75, 0.0]))),
      )

    scalar_neg, scalar_pos, y, sparse = Scheduler.run(
        run_relu, horizon=4, semantics=Semantics.SYNC
    )
    self.assertEqual(scalar_neg, [0.0] * 4)
    self.assertEqual(scalar_pos, [1.75] * 4)
    self.assertEqual(y, [0.0, 0.0, 1.75, 0.0])
    self.assertEqual(sparse, [0.0, Absent, 1.75, 0.0])

  def test_grad_rnn_layer_backpropagates_from_y(self):
    logger.setLevel(logging.WARNING)
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

    dz_expected = [0.0] * DIMENSION
    dh_next = 0.0
    for idx in range(DIMENSION - 1, -1, -1):
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
      rnn = RnnParams(wxh=Stream(wxh), whh=Stream(whh), bh=Stream(bh))
      y, dx, drnn = grad_rnn_layer(Stream(x_seq), rnn, lambda: dy_delayed)
      dy_delayed = Stream(dy_seq)
      return y, dx, drnn

    for semantics in (Semantics.KAHN, Semantics.SYNC):
      with self.subTest(semantics=semantics):
        y_trace, dx_trace, drnn_traces = Scheduler.run(
            run_grad_rnn, horizon=DIMENSION, semantics=semantics
        )
        assert Scheduler.graph is not None
        self.assertEmpty(
            Scheduler.graph.consumers(Scheduler.graph.output[0].istruct)
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


if __name__ == '__main__':
  absltest.main()
