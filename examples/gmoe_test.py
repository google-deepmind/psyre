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
"""Unit tests for sparsely-gated Mixture of Experts (GMoE) and GMoE-RNN."""

from collections.abc import Sequence
import logging
import math

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import check_causality
from psyre.builder import build_graph
from psyre.clock import check_clocks
from psyre.common import (
    Absent,
    counter,
    logger,
)
from examples import gmoe
from examples import rnn
from psyre.scheduler import Scheduler
from psyre.stream import Stream
from psyre.trace import (
    is_absent,
    is_present,
    present_values,
)

# pylint: enable=g-importing-member,g-multiple-import


def _ref_top2_softmax(
    s0: float, s1: float, s2: float
) -> tuple[float, float, float]:
  """Exact Python reference for `_sparse_top2_softmax`."""
  drop2 = s2 <= s0 and s2 <= s1
  drop1 = (not drop2) and (s1 <= s0)
  drop0 = (not drop2) and (not drop1)
  active = (not drop0, not drop1, not drop2)
  scores = (s0, s1, s2)
  s_max = max(s for s, a in zip(scores, active, strict=True) if a)
  exps = tuple(
      math.exp(s - s_max) if a else 0.0
      for s, a in zip(scores, active, strict=True)
  )
  z = sum(exps)
  return (exps[0] / z, exps[1] / z, exps[2] / z)


def _ref_scalar_gmoe_loss(
    x: float, y: float, params: gmoe.ScalarGmoeParamValues
) -> tuple[float, float, tuple[float, float, float]]:
  """Exact Python reference for scalar GMoE output, loss, and gate weights."""
  s = tuple(params.gate.w[k] * x + params.gate.b[k] for k in range(3))
  g = _ref_top2_softmax(*s)
  out = 0.0
  for k in range(3):
    if g[k] != 0.0:
      ek = params.experts[k]
      out += g[k] * (ek.w1 * x + ek.w2 * x * x + ek.b)
  return out, (out - y) ** 2, g


def _ref_rnn_perceptron(
    x_seq: Sequence[float],
    params: rnn.ParamValues,
    num_cells: int,
) -> float:
  """Exact Python reference for variable-depth RNN + ReLU perceptron."""
  dim = len(x_seq)
  h_grid: list[list[float]] = []
  for k in range(num_cells):
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
  y_rnn = [h_grid[min(t, num_cells - 1)][t] for t in range(dim)]
  z = params.per.bh[0] + sum(
      h_val * w_val for h_val, w_val in zip(y_rnn, params.per.wxh, strict=True)
  )
  return max(0.0, z)


def _ref_chained_gmoe_loss(
    x_seq: Sequence[float],
    y_target: float,
    params: gmoe.ChainedGmoeParamValues,
    expert_cells: tuple[int, int, int] = gmoe.DEFAULT_EXPERT_CELLS,
) -> tuple[float, float, tuple[float, float, float]]:
  """Exact Python reference for chained GMoE with RNN+perceptron experts."""
  scores = tuple(
      params.gate.b[k]
      + sum(
          xv * wv for xv, wv in zip(x_seq, params.gate.w[k], strict=True)
      )
      for k in range(3)
  )
  g = _ref_top2_softmax(*scores)
  out = 0.0
  for k in range(3):
    if g[k] != 0.0:
      out += g[k] * _ref_rnn_perceptron(
          x_seq, params.experts[k], expert_cells[k]
      )
  return out, (out - y_target) ** 2, g


def _initial_scalar_params() -> gmoe.ScalarGmoeParamValues:
  """Returns deterministic scalar GMoE parameters routing all 3 expert pairs."""
  return gmoe.GmoeParams(
      gate=gmoe.GateParams(w=(-1.2, 0.1, 1.3), b=(0.5, -0.5, 0.5)),
      experts=(
          gmoe.ScalarExpertParams(w1=1.5, w2=-0.2, b=0.8),
          gmoe.ScalarExpertParams(w1=-0.7, w2=0.15, b=2.5),
          gmoe.ScalarExpertParams(w1=0.3, w2=0.45, b=-0.4),
      ),
  )


def _initial_chained_params() -> gmoe.ChainedGmoeParamValues:
  """Returns deterministic chained GMoE-RNN parameters for 3 experts."""
  return gmoe.GmoeParams(
      gate=gmoe.GateParams(
          w=(
              [1.2, -0.4, 0.3, -0.2, 0.1],
              [-0.3, 1.1, -0.5, 0.4, -0.2],
              [-0.8, -0.5, 1.3, 0.3, 0.6],
          ),
          b=(0.15, -0.05, 0.1),
      ),
      experts=(
          rnn.ModelParams(
              rnn=rnn.RnnParams(wxh=0.35, whh=-0.25, bh=0.1),
              per=rnn.PerceptronParams(
                  wxh=[0.3, 0.4, -0.2, 0.5, 0.35],
                  bh=[0.6, 0.0, 0.0, 0.0, 0.0],
              ),
          ),
          rnn.ModelParams(
              rnn=rnn.RnnParams(wxh=-0.3, whh=0.4, bh=0.15),
              per=rnn.PerceptronParams(
                  wxh=[0.45, -0.25, 0.35, 0.2, 0.5],
                  bh=[0.7, 0.0, 0.0, 0.0, 0.0],
              ),
          ),
          rnn.ModelParams(
              rnn=rnn.RnnParams(wxh=0.25, whh=0.3, bh=-0.1),
              per=rnn.PerceptronParams(
                  wxh=[0.2, 0.35, 0.45, -0.15, 0.4],
                  bh=[0.8, 0.0, 0.0, 0.0, 0.0],
              ),
          ),
      ),
  )


def _chained_slice_params(
    traces: gmoe.ChainedGmoeParamTraces, pass_idx: int
) -> gmoe.ChainedGmoeParamValues:
  """Extracts chained GMoE-RNN parameter values for `pass_idx`."""
  start = pass_idx * rnn.DIMENSION
  end = start + rnn.DIMENSION
  return gmoe.extract_chained_params(traces.map(lambda seq: seq[start:end]))


class GmoeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def test_fixed_moe_demo_matches_paper_chronogram_and_clocks(self):
    graph = build_graph(gmoe.mixture_of_experts_demo)
    check_causality(graph)
    clocks = check_clocks(graph)
    traces = Scheduler.run_graph(graph, horizon=8, clocks=clocks)

    self.assertEqual(str(clocks.stream_clock(graph.output['o1'])), '. on c1')
    self.assertEqual(str(clocks.stream_clock(graph.output['o2'])), '. on c2')
    self.assertEqual(str(clocks.stream_clock(graph.output['o3'])), '. on c3')

    self.assertEqual(
        traces['o1'], [3.0, Absent, 7.0, Absent, 11.0, Absent, 5.0, Absent]
    )
    self.assertEqual(
        traces['o2'], [4.0, 3.0, Absent, Absent, Absent, -1.0, 3.0, Absent]
    )
    self.assertEqual(
        traces['o3'], [Absent, 2.0, 4.5, 8.0, Absent, 18.0, Absent, 8.0]
    )
    self.assertSequenceAlmostEqual(
        traces['o'], [3.4, 2.7, 5.75, 8.0, 11.0, 2.8, 3.8, 8.0], places=12
    )
    self.assertEqual(
        traces['e1_recurrent'], [1.0, 0.0, 4.0, 0.0, 9.0, 0.0, 11.0, 0.0]
    )

  def test_scalar_gmoe_dynamic_dispatch_and_clocks(self):
    params = _initial_scalar_params()
    # Inputs designed to drop experts 2, 0, and 1 respectively across ticks.
    xs = [-2.0, 0.0, 2.0]
    ys = [1.0, 2.5, 4.0]

    def circuit():
      p_s = params.map(Stream)
      x_s = Stream(xs)
      y_s = Stream(ys)
      fwd = gmoe.gmoe_forward(x_s, p_s)
      bwd = gmoe.grad_gmoe(x_s, y_s, p_s)
      return fwd, bwd

    graph = build_graph(circuit)
    check_causality(graph)
    clocks = check_clocks(graph)
    fwd_out, bwd_out = Scheduler.run_graph(
        graph, horizon=len(xs), clocks=clocks
    )

    self.assertSequenceAlmostEqual(fwd_out.o, bwd_out.o, places=12)
    # Confirm every expert is skipped on at least one cycle and is Absent there.
    skipped_experts: set[int] = set()
    for t in range(len(xs)):
      active_count = sum(1 for k in range(3) if bwd_out.conds[k][t])
      self.assertEqual(active_count, 2)
      for k in range(3):
        if bwd_out.conds[k][t]:
          self.assertTrue(is_present(bwd_out.expert_outputs[k], t))
          self.assertTrue(is_present(fwd_out.expert_outputs[k], t))
        else:
          skipped_experts.add(k)
          self.assertTrue(is_absent(bwd_out.expert_outputs[k], t))
          self.assertTrue(is_absent(fwd_out.expert_outputs[k], t))
          # Unselected expert MUST have exact zero parameter gradients.
          self.assertEqual(bwd_out.grads.experts[k].w1[t], 0.0)
          self.assertEqual(bwd_out.grads.experts[k].w2[t], 0.0)
          self.assertEqual(bwd_out.grads.experts[k].b[t], 0.0)
    self.assertEqual(skipped_experts, {0, 1, 2})

  def test_scalar_gmoe_reverse_mode_gradients_match_finite_differences(self):
    params = _initial_scalar_params()
    # Exercises all 3 active pairs: {0,1} (drop 2), {0,2} (drop 1), {1,2}
    # (drop 0).
    cases = [(-2.0, 1.0), (0.5, 2.2), (2.0, 4.5)]
    xs = [c[0] for c in cases]
    ys = [c[1] for c in cases]

    def circuit():
      return gmoe.grad_gmoe(Stream(xs), Stream(ys), params.map(Stream))

    out = Scheduler.run(circuit, horizon=len(cases))
    eps = 1e-6
    dropped_seen: set[int] = set()

    for t, (xv, yv) in enumerate(cases):
      ref_o, ref_l, ref_g = _ref_scalar_gmoe_loss(xv, yv, params)
      self.assertAlmostEqual(out.o[t], ref_o, places=12)
      self.assertAlmostEqual(out.loss[t], ref_l, places=12)
      for k in range(3):
        self.assertAlmostEqual(out.gates[k][t], ref_g[k], places=12)
        if ref_g[k] == 0.0:
          dropped_seen.add(k)

      # Finite-difference check on input dx.
      fd_x = (
          _ref_scalar_gmoe_loss(xv + eps, yv, params)[1]
          - _ref_scalar_gmoe_loss(xv - eps, yv, params)[1]
      ) / (2.0 * eps)
      self.assertAlmostEqual(out.dx[t], fd_x, places=6)

      # Finite-difference check on all 6 gate parameters.
      for k in range(3):
        w_plus = list(params.gate.w)
        w_plus[k] += eps
        w_minus = list(params.gate.w)
        w_minus[k] -= eps
        p_plus = params._replace(gate=params.gate._replace(w=tuple(w_plus)))
        p_minus = params._replace(gate=params.gate._replace(w=tuple(w_minus)))
        fd_gw = (
            _ref_scalar_gmoe_loss(xv, yv, p_plus)[1]
            - _ref_scalar_gmoe_loss(xv, yv, p_minus)[1]
        ) / (2.0 * eps)
        self.assertAlmostEqual(out.grads.gate.w[k][t], fd_gw, places=6)

        b_plus = list(params.gate.b)
        b_plus[k] += eps
        b_minus = list(params.gate.b)
        b_minus[k] -= eps
        p_plus = params._replace(gate=params.gate._replace(b=tuple(b_plus)))
        p_minus = params._replace(gate=params.gate._replace(b=tuple(b_minus)))
        fd_gb = (
            _ref_scalar_gmoe_loss(xv, yv, p_plus)[1]
            - _ref_scalar_gmoe_loss(xv, yv, p_minus)[1]
        ) / (2.0 * eps)
        self.assertAlmostEqual(out.grads.gate.b[k][t], fd_gb, places=6)

      # Finite-difference check on all 9 expert parameters.
      for k in range(3):
        for field in ('w1', 'w2', 'b'):
          base_val = getattr(params.experts[k], field)
          exps_plus = list(params.experts)
          exps_plus[k] = params.experts[k]._replace(**{field: base_val + eps})
          exps_minus = list(params.experts)
          exps_minus[k] = params.experts[k]._replace(**{field: base_val - eps})
          fd_e = (
              _ref_scalar_gmoe_loss(
                  xv, yv, params._replace(experts=tuple(exps_plus))
              )[1]
              - _ref_scalar_gmoe_loss(
                  xv, yv, params._replace(experts=tuple(exps_minus))
              )[1]
          ) / (2.0 * eps)
          self.assertAlmostEqual(
              getattr(out.grads.experts[k], field)[t], fd_e, places=6
          )

    # Confirm all 3 experts were dropped at least once (dynamic dispatch).
    self.assertEqual(dropped_seen, {0, 1, 2})

  def test_scalar_gmoe_training_converges_and_freezes_inactive_experts(self):
    initial = _initial_scalar_params()
    # Multi-regime dataset targeting different functions across x ranges.
    dataset = [
        (-2.0, -3.0),
        (-1.0, -0.8),
        (0.5, 2.2),
        (1.5, 3.6),
        (2.0, 5.2),
    ]
    max_steps = 40
    horizon = max_steps + 3

    out = Scheduler.run(
        gmoe.train_gmoe,
        dataset,
        initial,
        max_steps=max_steps,
        lr=0.01,
        horizon=horizon,
    )
    self.assertFalse(Scheduler.deadlocked)

    losses = [float(v) for v in present_values(out.loss)]
    self.assertLen(losses, max_steps)

    # Verify dynamic dispatch: unselected expert outputs are Absent and their
    # parameters remain bit-for-bit unchanged on every skipped step!
    skipped_steps = [0, 0, 0]
    for t in range(max_steps):
      prev_p = (
          initial if t == 0 else gmoe.extract_scalar_params(out.params, t - 1)
      )
      next_p = gmoe.extract_scalar_params(out.params, t)
      for k in range(3):
        if out.conds[k][t]:
          self.assertTrue(is_present(out.expert_outputs[k], t))
        else:
          skipped_steps[k] += 1
          self.assertTrue(is_absent(out.expert_outputs[k], t))
          self.assertEqual(next_p.experts[k], prev_p.experts[k])
    for k in range(3):
      self.assertGreater(skipped_steps[k], 0)

    # Parameters stay strictly constant past max_steps.
    trained = gmoe.extract_scalar_params(out.params, max_steps - 1)
    for t in range(max_steps, horizon):
      self.assertTrue(is_absent(out.loss, t))
      self.assertEqual(gmoe.extract_scalar_params(out.params, t), trained)

    # Dataset evaluation loss improves substantially after training.
    init_eval = sum(
        _ref_scalar_gmoe_loss(xv, yv, initial)[1] for xv, yv in dataset
    )
    trained_eval = sum(
        _ref_scalar_gmoe_loss(xv, yv, trained)[1] for xv, yv in dataset
    )
    self.assertLess(trained_eval, init_eval * 0.5)

  def test_chained_gmoe_rnn_dynamic_dispatch_and_clocks(self):
    params = _initial_chained_params()
    # 3 distinct sequences designed to drop experts 2, 0, and 1 respectively.
    sequences = [
        ([1.0, 0.2, -0.8, 0.1, -0.3], 1.8),
        ([-0.9, 0.8, 0.9, 0.5, 0.6], 1.4),
        ([0.8, -0.9, 0.9, -0.2, 0.4], 1.6),
    ]
    dim = rnn.DIMENSION
    horizon = len(sequences) * dim

    def circuit():
      tick = counter()
      pass_idx = tick // dim
      in_pass = tick % dim
      pass_reset = in_pass == 0
      samples = rnn.dataset_streams(sequences, in_pass=in_pass)
      sample = gmoe._select_sequence_sample(pass_idx, samples)
      p_s = gmoe.chained_param_streams(in_pass, params)
      fwd = gmoe.chained_gmoe_forward(
          sample.x,
          p_s,
          expert_cells=gmoe.DEFAULT_EXPERT_CELLS,
          in_pass=in_pass,
          pass_reset=pass_reset,
      )
      bwd = gmoe.grad_chained_gmoe(
          sample.x,
          sample.y,
          p_s,
          expert_cells=gmoe.DEFAULT_EXPERT_CELLS,
          in_pass=in_pass,
          pass_reset=pass_reset,
      )
      return fwd, bwd

    graph = build_graph(circuit)
    check_causality(graph)
    clocks = check_clocks(graph)
    fwd_out, bwd_out = Scheduler.run_graph(
        graph, horizon=horizon, clocks=clocks
    )
    self.assertFalse(Scheduler.deadlocked)

    # Forward outputs match between pure forward and reverse-mode forward pass.
    self.assertSequenceAlmostEqual(fwd_out.o, bwd_out.o, places=12)

    # Verify each pass dynamically dispatches 2 experts (all 5 ticks present)
    # and skips 1 expert (all 5 ticks Absent), exercising all 3 drop choices.
    dropped_passes: set[int] = set()
    for p_idx, (x_seq, y_target) in enumerate(sequences):
      start = p_idx * dim
      end = start + dim
      ref_o, ref_l, ref_g = _ref_chained_gmoe_loss(
          x_seq, y_target, params, gmoe.DEFAULT_EXPERT_CELLS
      )
      self.assertAlmostEqual(bwd_out.o[end - 1], ref_o, places=12)
      self.assertAlmostEqual(bwd_out.loss[end - 1], ref_l, places=12)
      for k in range(3):
        pass_conds = bwd_out.conds[k][start:end]
        if ref_g[k] == 0.0:
          dropped_passes.add(k)
          self.assertEqual(pass_conds, [False] * dim)
          self.assertEqual(bwd_out.expert_outputs[k][start:end], [Absent] * dim)
          self.assertEqual(fwd_out.expert_outputs[k][start:end], [Absent] * dim)
        else:
          self.assertEqual(pass_conds, [True] * dim)
          for t in range(start, end):
            self.assertTrue(is_present(bwd_out.expert_outputs[k], t))
            self.assertTrue(is_present(fwd_out.expert_outputs[k], t))
    self.assertEqual(dropped_passes, {0, 1, 2})

  def test_chained_gmoe_rnn_reverse_mode_gradients_match_finite_differences(
      self,
  ):
    params = _initial_chained_params()
    expert_cells = gmoe.DEFAULT_EXPERT_CELLS  # (3, 4, 5)
    dim = rnn.DIMENSION
    # 3 passes dropping experts 2, 0, and 1 respectively.
    sequences = [
        ((1.0, 0.2, -0.8, 0.1, -0.3), 1.8),
        ((-0.9, 0.8, 0.9, 0.5, 0.6), 1.4),
        ((0.8, -0.9, 0.9, -0.2, 0.4), 1.6),
    ]
    eps = 1e-6

    def make_one_pass(xs: Sequence[float], yt: float):
      def circuit():
        in_pass = counter() % dim
        pass_reset = in_pass == 0
        p_s = gmoe.chained_param_streams(in_pass, params)
        return gmoe.grad_chained_gmoe(
            Stream(list(xs)),
            yt,
            p_s,
            expert_cells=expert_cells,
            in_pass=in_pass,
            pass_reset=pass_reset,
        )

      return circuit

    for x_seq, y_target in sequences:
      with self.subTest(x_seq=x_seq):
        out = Scheduler.run(make_one_pass(x_seq, y_target), horizon=dim)
        self.assertFalse(Scheduler.deadlocked)
        ref_o, ref_l, ref_g = _ref_chained_gmoe_loss(
            x_seq, y_target, params, expert_cells
        )
        self.assertAlmostEqual(out.o[-1], ref_o, places=12)
        self.assertAlmostEqual(out.loss[-1], ref_l, places=12)

        # 1. Input gradients dx[0..4].
        for i in range(dim):
          x_p = list(x_seq)
          x_p[i] += eps
          x_m = list(x_seq)
          x_m[i] -= eps
          fd_x = (
              _ref_chained_gmoe_loss(x_p, y_target, params, expert_cells)[1]
              - _ref_chained_gmoe_loss(x_m, y_target, params, expert_cells)[1]
          ) / (2.0 * eps)
          self.assertAlmostEqual(out.dx[i], fd_x, places=6)

        # 2. All 15 gate weights + 3 gate biases.
        for k in range(3):
          for i in range(dim):
            w_p = [list(v) for v in params.gate.w]
            w_p[k][i] += eps
            w_m = [list(v) for v in params.gate.w]
            w_m[k][i] -= eps
            p_p = params._replace(gate=params.gate._replace(w=tuple(w_p)))
            p_m = params._replace(gate=params.gate._replace(w=tuple(w_m)))
            fd_w = (
                _ref_chained_gmoe_loss(x_seq, y_target, p_p, expert_cells)[1]
                - _ref_chained_gmoe_loss(x_seq, y_target, p_m, expert_cells)[1]
            ) / (2.0 * eps)
            self.assertAlmostEqual(out.grads.gate.w[k][i], fd_w, places=6)

          b_p = list(params.gate.b)
          b_p[k] += eps
          b_m = list(params.gate.b)
          b_m[k] -= eps
          p_p = params._replace(gate=params.gate._replace(b=tuple(b_p)))
          p_m = params._replace(gate=params.gate._replace(b=tuple(b_m)))
          fd_b = (
              _ref_chained_gmoe_loss(x_seq, y_target, p_p, expert_cells)[1]
              - _ref_chained_gmoe_loss(x_seq, y_target, p_m, expert_cells)[1]
          ) / (2.0 * eps)
          for t in range(dim):
            self.assertAlmostEqual(out.grads.gate.b[k][t], fd_b, places=6)

        # 3. All 3 experts' RNN scalars (wxh, whh, bh) and perceptron params.
        for k in range(3):
          exp_k = params.experts[k]
          grad_k = out.grads.experts[k]
          if ref_g[k] == 0.0:
            # Inactive expert MUST have exact zero gradients on every leaf.
            self.assertEqual(grad_k.rnn.wxh, [0.0] * dim)
            self.assertEqual(grad_k.rnn.whh, [0.0] * dim)
            self.assertEqual(grad_k.rnn.bh, [0.0] * dim)
            self.assertEqual(grad_k.per.wxh, [0.0] * dim)
            self.assertEqual(grad_k.per.bh, [0.0] * dim)
            continue

          for field in ('wxh', 'whh', 'bh'):
            base_val = getattr(exp_k.rnn, field)
            exps_p = list(params.experts)
            exps_p[k] = exp_k._replace(
                rnn=exp_k.rnn._replace(**{field: base_val + eps})
            )
            exps_m = list(params.experts)
            exps_m[k] = exp_k._replace(
                rnn=exp_k.rnn._replace(**{field: base_val - eps})
            )
            fd_r = (
                _ref_chained_gmoe_loss(
                    x_seq,
                    y_target,
                    params._replace(experts=tuple(exps_p)),
                    expert_cells,
                )[1]
                - _ref_chained_gmoe_loss(
                    x_seq,
                    y_target,
                    params._replace(experts=tuple(exps_m)),
                    expert_cells,
                )[1]
            ) / (2.0 * eps)
            self.assertAlmostEqual(
                sum(getattr(grad_k.rnn, field)), fd_r, places=6
            )

          for i in range(dim):
            pw_p = list(exp_k.per.wxh)
            pw_p[i] += eps
            pw_m = list(exp_k.per.wxh)
            pw_m[i] -= eps
            exps_p = list(params.experts)
            exps_p[k] = exp_k._replace(per=exp_k.per._replace(wxh=pw_p))
            exps_m = list(params.experts)
            exps_m[k] = exp_k._replace(per=exp_k.per._replace(wxh=pw_m))
            fd_pw = (
                _ref_chained_gmoe_loss(
                    x_seq,
                    y_target,
                    params._replace(experts=tuple(exps_p)),
                    expert_cells,
                )[1]
                - _ref_chained_gmoe_loss(
                    x_seq,
                    y_target,
                    params._replace(experts=tuple(exps_m)),
                    expert_cells,
                )[1]
            ) / (2.0 * eps)
            self.assertAlmostEqual(grad_k.per.wxh[i], fd_pw, places=6)

          pb_p = [exp_k.per.bh[0] + eps] + list(exp_k.per.bh[1:])
          pb_m = [exp_k.per.bh[0] - eps] + list(exp_k.per.bh[1:])
          exps_p = list(params.experts)
          exps_p[k] = exp_k._replace(per=exp_k.per._replace(bh=pb_p))
          exps_m = list(params.experts)
          exps_m[k] = exp_k._replace(per=exp_k.per._replace(bh=pb_m))
          fd_pb = (
              _ref_chained_gmoe_loss(
                  x_seq,
                  y_target,
                  params._replace(experts=tuple(exps_p)),
                  expert_cells,
              )[1]
              - _ref_chained_gmoe_loss(
                  x_seq,
                  y_target,
                  params._replace(experts=tuple(exps_m)),
                  expert_cells,
              )[1]
          ) / (2.0 * eps)
          self.assertAlmostEqual(grad_k.per.bh[0], fd_pb, places=6)
          self.assertEqual(grad_k.per.bh[1:], [0.0] * (dim - 1))

  def test_chained_gmoe_rnn_training_converges_and_freezes_inactive_experts(
      self,
  ):
    initial = _initial_chained_params()
    expert_cells = gmoe.DEFAULT_EXPERT_CELLS  # (3, 4, 5)
    dim = rnn.DIMENSION
    # Alternating sequences activating {0,1}, {1,2}, {0,2} cyclically —
    # every expert transitions active -> inactive -> active repeatedly.
    dataset = [
        ([1.0, 0.2, -0.8, 0.1, -0.3], 2.0),
        ([-0.9, 0.8, 0.9, 0.5, 0.6], 1.5),
        ([0.8, -0.9, 0.9, -0.2, 0.4], 1.8),
    ]
    max_steps = 12
    # 1 INIT_LOAD pass + max_steps BPTT passes + 2 frozen tail passes.
    total_passes = 1 + max_steps + 2
    horizon = total_passes * dim

    out = Scheduler.run(
        gmoe.train_chained_gmoe,
        dataset,
        initial,
        expert_cells=expert_cells,
        max_steps=max_steps,
        lr=0.02,
        horizon=horizon,
    )
    self.assertFalse(Scheduler.deadlocked)

    # Verify exact INIT_LOAD pass.
    self.assertEqual(_chained_slice_params(out.params, 0), initial)

    # Verify exact training loss emission ticks.
    train_losses = [float(v) for v in present_values(out.loss)]
    self.assertLen(train_losses, max_steps)
    expected_ticks = {(step + 2) * dim - 1 for step in range(max_steps)}
    for t in range(horizon):
      self.assertEqual(is_present(out.loss, t), t in expected_ticks)

    # Verify dynamic dispatch & frozen inactive experts on every training pass.
    skipped_counts = [0, 0, 0]
    for step in range(max_steps):
      pass_idx = step + 1
      start = pass_idx * dim
      end = start + dim
      prev_p = _chained_slice_params(out.params, pass_idx - 1)
      next_p = _chained_slice_params(out.params, pass_idx)
      for k in range(3):
        cond_slice = out.conds[k][start:end]
        if all(cond_slice):
          for t in range(start, end):
            self.assertTrue(is_present(out.expert_outputs[k], t))
        else:
          self.assertEqual(cond_slice, [False] * dim)
          skipped_counts[k] += 1
          self.assertEqual(out.expert_outputs[k][start:end], [Absent] * dim)
          self.assertEqual(next_p.experts[k], prev_p.experts[k])
    for k in range(3):
      self.assertGreater(skipped_counts[k], 0)

    # Verify parameters freeze completely past max_steps.
    trained = _chained_slice_params(out.params, max_steps)
    self.assertEqual(_chained_slice_params(out.params, max_steps + 1), trained)
    self.assertEqual(_chained_slice_params(out.params, max_steps + 2), trained)
    self.assertEqual(gmoe.extract_chained_params(out.params), trained)

    # Evaluate trained model with `eval_chained_gmoe` and compare against
    # exact Python reference and initial evaluation loss.
    eval_horizon = len(dataset) * dim
    eval_trace = Scheduler.run(
        gmoe.eval_chained_gmoe,
        dataset,
        trained,
        expert_cells=expert_cells,
        horizon=eval_horizon,
    )
    eval_losses = [float(v) for v in present_values(eval_trace)]
    self.assertLen(eval_losses, len(dataset))
    ref_trained_losses = [
        _ref_chained_gmoe_loss(x_seq, y_target, trained, expert_cells)[1]
        for x_seq, y_target in dataset
    ]
    for actual, expected in zip(eval_losses, ref_trained_losses, strict=True):
      self.assertAlmostEqual(actual, expected, places=9)

    ref_init_losses = [
        _ref_chained_gmoe_loss(x_seq, y_target, initial, expert_cells)[1]
        for x_seq, y_target in dataset
    ]
    self.assertLess(sum(eval_losses), sum(ref_init_losses))


if __name__ == '__main__':
  absltest.main()
