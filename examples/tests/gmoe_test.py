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
from typing import Any

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import check_causality
from psyre.builder import build_graph
from psyre.clock import check_clocks
from psyre.common import (
    Absent,
    counter,
    logger,
    switch,
)
from examples.src import gmoe
from examples.src import rnn
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
  z = params.per.bh[0] + math.sumprod(y_rnn, params.per.wxh)
  return max(0.0, z)


def _ref_chained_gmoe_loss(
    x_seq: Sequence[float],
    y_target: float,
    params: gmoe.ChainedGmoeParamValues,
    expert_cells: tuple[int, int, int] = gmoe.DEFAULT_EXPERT_CELLS,
) -> tuple[float, float, tuple[float, float, float]]:
  """Exact Python reference for chained GMoE with RNN+perceptron experts."""
  scores = tuple(
      params.gate.b[k] + math.sumprod(x_seq, params.gate.w[k])
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


def _initial_scalar_params(gmoe_mod: Any = gmoe) -> Any:
  """Returns deterministic scalar GMoE parameters routing all 3 expert pairs."""
  return gmoe_mod.GmoeParams(
      gate=gmoe_mod.GateParams(w=(-1.2, 0.1, 1.3), b=(0.5, -0.5, 0.5)),
      experts=(
          gmoe_mod.ScalarExpertParams(w1=1.5, w2=-0.2, b=0.8),
          gmoe_mod.ScalarExpertParams(w1=-0.7, w2=0.15, b=2.5),
          gmoe_mod.ScalarExpertParams(w1=0.3, w2=0.45, b=-0.4),
      ),
  )


def _initial_chained_params(
    gmoe_mod: Any = gmoe, rnn_mod: Any = rnn
) -> Any:
  """Returns deterministic chained GMoE-RNN parameters for 3 experts."""
  return gmoe_mod.GmoeParams(
      gate=gmoe_mod.GateParams(
          w=(
              [1.2, -0.4, 0.3, -0.2, 0.1],
              [-0.3, 1.1, -0.5, 0.4, -0.2],
              [-0.8, -0.5, 1.3, 0.3, 0.6],
          ),
          b=(0.15, -0.05, 0.1),
      ),
      experts=(
          rnn_mod.ModelParams(
              rnn=rnn_mod.RnnParams(wxh=0.35, whh=-0.25, bh=0.1),
              per=rnn_mod.PerceptronParams(
                  wxh=[0.3, 0.4, -0.2, 0.5, 0.35],
                  bh=[0.6, 0.0, 0.0, 0.0, 0.0],
              ),
          ),
          rnn_mod.ModelParams(
              rnn=rnn_mod.RnnParams(wxh=-0.3, whh=0.4, bh=0.15),
              per=rnn_mod.PerceptronParams(
                  wxh=[0.45, -0.25, 0.35, 0.2, 0.5],
                  bh=[0.7, 0.0, 0.0, 0.0, 0.0],
              ),
          ),
          rnn_mod.ModelParams(
              rnn=rnn_mod.RnnParams(wxh=0.25, whh=0.3, bh=-0.1),
              per=rnn_mod.PerceptronParams(
                  wxh=[0.2, 0.35, 0.45, -0.15, 0.4],
                  bh=[0.8, 0.0, 0.0, 0.0, 0.0],
              ),
          ),
      ),
  )


def _chained_slice_params(
    traces: Any, pass_idx: int, gmoe_mod: Any = gmoe, rnn_mod: Any = rnn
) -> Any:
  """Extracts chained GMoE-RNN parameter values for `pass_idx`."""
  start = pass_idx * rnn_mod.DIMENSION
  end = start + rnn_mod.DIMENSION
  return gmoe_mod.extract_chained_params(traces.map(lambda seq: seq[start:end]))


class GmoeTest(absltest.TestCase):

  GMOE: Any = gmoe
  RNN: Any = rnn

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def test_fixed_moe_demo_matches_paper_chronogram_and_clocks(self):
    gmoe_mod = self.GMOE
    graph = build_graph(gmoe_mod.mixture_of_experts_demo)
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

  def test_scalar_gmoe_dispatch_clocks_and_reverse_mode_gradients(self):
    gmoe_mod = self.GMOE
    params = _initial_scalar_params(gmoe_mod)
    # Exercises all 3 active pairs: {0,1} (drop 2), {0,2} (drop 1), {1,2}
    # (drop 0), avoiding exact gate-score ties for finite differences.
    cases = [(-2.0, 1.0), (0.5, 2.2), (2.0, 4.5)]
    xs = [c[0] for c in cases]
    ys = [c[1] for c in cases]

    def circuit():
      p_s = params.map(Stream)
      x_s = Stream(xs)
      y_s = Stream(ys)
      return gmoe_mod.gmoe_forward(x_s, p_s), gmoe_mod.grad_gmoe(x_s, y_s, p_s)

    fwd_out, out = Scheduler.run(circuit, horizon=len(cases))
    self.assertSequenceAlmostEqual(fwd_out.o, out.o, places=12)
    eps = 1e-6
    dropped_seen: set[int] = set()

    for t, (xv, yv) in enumerate(cases):
      ref_o, ref_l, ref_g = _ref_scalar_gmoe_loss(xv, yv, params)
      self.assertAlmostEqual(out.o[t], ref_o, places=12)
      self.assertAlmostEqual(out.loss[t], ref_l, places=12)
      self.assertEqual(sum(1 for k in range(3) if out.conds[k][t]), 2)
      for k in range(3):
        self.assertAlmostEqual(out.gates[k][t], ref_g[k], places=12)
        if out.conds[k][t]:
          self.assertTrue(is_present(out.expert_outputs[k], t))
          self.assertTrue(is_present(fwd_out.expert_outputs[k], t))
        else:
          dropped_seen.add(k)
          self.assertEqual(ref_g[k], 0.0)
          self.assertTrue(is_absent(out.expert_outputs[k], t))
          self.assertTrue(is_absent(fwd_out.expert_outputs[k], t))
          # Unselected expert MUST have exact zero parameter gradients.
          for field in ('w1', 'w2', 'b'):
            self.assertEqual(getattr(out.grads.experts[k], field)[t], 0.0)

      def _scalar_fd(p_plus, p_minus, x_p=xv, x_m=xv, target_y=yv):
        return (
            _ref_scalar_gmoe_loss(x_p, target_y, p_plus)[1]
            - _ref_scalar_gmoe_loss(x_m, target_y, p_minus)[1]
        ) / (2.0 * eps)

      self.assertAlmostEqual(
          out.dx[t],
          _scalar_fd(params, params, xv + eps, xv - eps),
          places=6,
      )
      for k in range(3):
        for field in ('w', 'b'):
          base = list(getattr(params.gate, field))
          v_p, v_m = list(base), list(base)
          v_p[k] += eps
          v_m[k] -= eps
          p_p = params._replace(
              gate=params.gate._replace(**{field: tuple(v_p)})
          )
          p_m = params._replace(
              gate=params.gate._replace(**{field: tuple(v_m)})
          )
          self.assertAlmostEqual(
              getattr(out.grads.gate, field)[k][t],
              _scalar_fd(p_p, p_m),
              places=6,
          )
        for field in ('w1', 'w2', 'b'):
          base_val = getattr(params.experts[k], field)
          exps_p = list(params.experts)
          exps_p[k] = params.experts[k]._replace(**{field: base_val + eps})
          exps_m = list(params.experts)
          exps_m[k] = params.experts[k]._replace(**{field: base_val - eps})
          self.assertAlmostEqual(
              getattr(out.grads.experts[k], field)[t],
              _scalar_fd(
                  params._replace(experts=tuple(exps_p)),
                  params._replace(experts=tuple(exps_m)),
              ),
              places=6,
          )

    # Confirm all 3 experts were skipped on at least one cycle.
    self.assertEqual(dropped_seen, {0, 1, 2})

  def test_scalar_gmoe_training_converges_and_freezes_inactive_experts(self):
    gmoe_mod = self.GMOE
    initial = _initial_scalar_params(gmoe_mod)
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
        gmoe_mod.train_gmoe,
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
          initial
          if t == 0
          else gmoe_mod.extract_scalar_params(out.params, t - 1)
      )
      next_p = gmoe_mod.extract_scalar_params(out.params, t)
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
    trained = gmoe_mod.extract_scalar_params(out.params, max_steps - 1)
    for t in range(max_steps, horizon):
      self.assertTrue(is_absent(out.loss, t))
      self.assertEqual(gmoe_mod.extract_scalar_params(out.params, t), trained)

    # Dataset evaluation loss improves substantially after training.
    init_eval = sum(
        _ref_scalar_gmoe_loss(xv, yv, initial)[1] for xv, yv in dataset
    )
    trained_eval = sum(
        _ref_scalar_gmoe_loss(xv, yv, trained)[1] for xv, yv in dataset
    )
    self.assertLess(trained_eval, init_eval * 0.5)

  def test_chained_gmoe_rnn_dispatch_clocks_and_reverse_mode_gradients(self):
    gmoe_mod = self.GMOE
    rnn_mod = self.RNN
    params = _initial_chained_params(gmoe_mod, rnn_mod)
    expert_cells = gmoe_mod.DEFAULT_EXPERT_CELLS  # (3, 4, 5)
    # 3 distinct sequences designed to drop experts 2, 0, and 1 respectively.
    sequences = [
        ([1.0, 0.2, -0.8, 0.1, -0.3], 1.8),
        ([-0.9, 0.8, 0.9, 0.5, 0.6], 1.4),
        ([0.8, -0.9, 0.9, -0.2, 0.4], 1.6),
    ]
    dim = rnn_mod.DIMENSION
    horizon = len(sequences) * dim

    def circuit():
      tick = counter()
      pass_idx = tick // dim
      in_pass = tick % dim
      pass_reset = in_pass == 0
      samples = rnn_mod.dataset_streams(sequences, in_pass=in_pass)
      sample = rnn_mod.LabeledStream(
          x=switch(
              pass_idx,
              {i: s.x for i, s in enumerate(samples)},
              default=0.0,
          ),
          y=switch(
              pass_idx,
              {i: s.y for i, s in enumerate(samples)},
              default=0.0,
          ),
      )
      p_s = gmoe_mod.chained_param_streams(in_pass, params)
      fwd = gmoe_mod.chained_gmoe_forward(
          sample.x,
          p_s,
          expert_cells=expert_cells,
          in_pass=in_pass,
          pass_reset=pass_reset,
      )
      bwd = gmoe_mod.grad_chained_gmoe(
          sample.x,
          sample.y,
          p_s,
          expert_cells=expert_cells,
          in_pass=in_pass,
          pass_reset=pass_reset,
      )
      return fwd, bwd

    fwd_out, bwd_out = Scheduler.run(circuit, horizon=horizon)
    self.assertFalse(Scheduler.deadlocked)
    self.assertSequenceAlmostEqual(fwd_out.o, bwd_out.o, places=12)

    eps = 1e-6
    dropped_passes: set[int] = set()
    for p_idx, (x_seq, y_target) in enumerate(sequences):
      start = p_idx * dim
      end = start + dim
      with self.subTest(pass_idx=p_idx, x_seq=x_seq):
        ref_o, ref_l, ref_g = _ref_chained_gmoe_loss(
            x_seq, y_target, params, expert_cells
        )
        self.assertAlmostEqual(bwd_out.o[end - 1], ref_o, places=12)
        self.assertAlmostEqual(bwd_out.loss[end - 1], ref_l, places=12)

        def _chain_fd(p_plus, p_minus, xs_p=x_seq, xs_m=x_seq, yt=y_target):
          return (
              _ref_chained_gmoe_loss(xs_p, yt, p_plus, expert_cells)[1]
              - _ref_chained_gmoe_loss(xs_m, yt, p_minus, expert_cells)[1]
          ) / (2.0 * eps)

        # 1. Input gradients dx[0..4].
        for i in range(dim):
          x_p, x_m = list(x_seq), list(x_seq)
          x_p[i] += eps
          x_m[i] -= eps
          self.assertAlmostEqual(
              bwd_out.dx[start + i],
              _chain_fd(params, params, x_p, x_m),
              places=6,
          )

        # 2. All 15 gate weights + 3 gate biases.
        for k in range(3):
          for i in range(dim):
            w_p = [list(v) for v in params.gate.w]
            w_m = [list(v) for v in params.gate.w]
            w_p[k][i] += eps
            w_m[k][i] -= eps
            fd_w = _chain_fd(
                params._replace(gate=params.gate._replace(w=tuple(w_p))),
                params._replace(gate=params.gate._replace(w=tuple(w_m))),
            )
            self.assertAlmostEqual(
                bwd_out.grads.gate.w[k][start + i], fd_w, places=6
            )

          b_p, b_m = list(params.gate.b), list(params.gate.b)
          b_p[k] += eps
          b_m[k] -= eps
          fd_b = _chain_fd(
              params._replace(gate=params.gate._replace(b=tuple(b_p))),
              params._replace(gate=params.gate._replace(b=tuple(b_m))),
          )
          for t in range(start, end):
            self.assertAlmostEqual(bwd_out.grads.gate.b[k][t], fd_b, places=6)

        # 3. Dynamic dispatch presence & all 3 experts' RNN / perceptron grads.
        for k in range(3):
          exp_k = params.experts[k]
          grad_k = bwd_out.grads.experts[k]
          pass_conds = bwd_out.conds[k][start:end]
          if ref_g[k] == 0.0:
            dropped_passes.add(k)
            self.assertEqual(pass_conds, [False] * dim)
            self.assertEqual(
                bwd_out.expert_outputs[k][start:end], [Absent] * dim
            )
            self.assertEqual(
                fwd_out.expert_outputs[k][start:end], [Absent] * dim
            )
            self.assertEqual(grad_k.rnn.wxh[start:end], [0.0] * dim)
            self.assertEqual(grad_k.rnn.whh[start:end], [0.0] * dim)
            self.assertEqual(grad_k.rnn.bh[start:end], [0.0] * dim)
            self.assertEqual(grad_k.per.wxh[start:end], [0.0] * dim)
            self.assertEqual(grad_k.per.bh[start:end], [0.0] * dim)
            continue

          self.assertEqual(pass_conds, [True] * dim)
          for t in range(start, end):
            self.assertTrue(is_present(bwd_out.expert_outputs[k], t))
            self.assertTrue(is_present(fwd_out.expert_outputs[k], t))

          for field in ('wxh', 'whh', 'bh'):
            base_val = getattr(exp_k.rnn, field)
            exps_p, exps_m = list(params.experts), list(params.experts)
            exps_p[k] = exp_k._replace(
                rnn=exp_k.rnn._replace(**{field: base_val + eps})
            )
            exps_m[k] = exp_k._replace(
                rnn=exp_k.rnn._replace(**{field: base_val - eps})
            )
            fd_r = _chain_fd(
                params._replace(experts=tuple(exps_p)),
                params._replace(experts=tuple(exps_m)),
            )
            self.assertAlmostEqual(
                sum(getattr(grad_k.rnn, field)[start:end]), fd_r, places=6
            )

          for i in range(dim):
            pw_p, pw_m = list(exp_k.per.wxh), list(exp_k.per.wxh)
            pw_p[i] += eps
            pw_m[i] -= eps
            exps_p, exps_m = list(params.experts), list(params.experts)
            exps_p[k] = exp_k._replace(per=exp_k.per._replace(wxh=pw_p))
            exps_m[k] = exp_k._replace(per=exp_k.per._replace(wxh=pw_m))
            fd_pw = _chain_fd(
                params._replace(experts=tuple(exps_p)),
                params._replace(experts=tuple(exps_m)),
            )
            self.assertAlmostEqual(grad_k.per.wxh[start + i], fd_pw, places=6)

          pb_p = [exp_k.per.bh[0] + eps] + list(exp_k.per.bh[1:])
          pb_m = [exp_k.per.bh[0] - eps] + list(exp_k.per.bh[1:])
          exps_p, exps_m = list(params.experts), list(params.experts)
          exps_p[k] = exp_k._replace(per=exp_k.per._replace(bh=pb_p))
          exps_m[k] = exp_k._replace(per=exp_k.per._replace(bh=pb_m))
          fd_pb = _chain_fd(
              params._replace(experts=tuple(exps_p)),
              params._replace(experts=tuple(exps_m)),
          )
          self.assertAlmostEqual(grad_k.per.bh[start], fd_pb, places=6)
          self.assertEqual(grad_k.per.bh[start + 1 : end], [0.0] * (dim - 1))
    self.assertEqual(dropped_passes, {0, 1, 2})

  def test_chained_gmoe_rnn_training_converges_and_freezes_inactive_experts(
      self,
  ):
    gmoe_mod = self.GMOE
    rnn_mod = self.RNN
    initial = _initial_chained_params(gmoe_mod, rnn_mod)
    expert_cells = gmoe_mod.DEFAULT_EXPERT_CELLS  # (3, 4, 5)
    dim = rnn_mod.DIMENSION
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
        gmoe_mod.train_chained_gmoe,
        dataset,
        initial,
        expert_cells=expert_cells,
        max_steps=max_steps,
        lr=0.02,
        horizon=horizon,
    )
    self.assertFalse(Scheduler.deadlocked)

    # Verify exact INIT_LOAD pass.
    self.assertEqual(
        _chained_slice_params(out.params, 0, gmoe_mod, rnn_mod), initial
    )

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
      prev_p = _chained_slice_params(
          out.params, pass_idx - 1, gmoe_mod, rnn_mod
      )
      next_p = _chained_slice_params(out.params, pass_idx, gmoe_mod, rnn_mod)
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
    trained = _chained_slice_params(out.params, max_steps, gmoe_mod, rnn_mod)
    self.assertEqual(
        _chained_slice_params(out.params, max_steps + 1, gmoe_mod, rnn_mod),
        trained,
    )
    self.assertEqual(
        _chained_slice_params(out.params, max_steps + 2, gmoe_mod, rnn_mod),
        trained,
    )
    self.assertEqual(gmoe_mod.extract_chained_params(out.params), trained)

    # Evaluate trained model with `eval_chained_gmoe` and compare against
    # exact Python reference and initial evaluation loss.
    eval_horizon = len(dataset) * dim
    eval_trace = Scheduler.run(
        gmoe_mod.eval_chained_gmoe,
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

  def test_chained_params_normalize_and_extract_non_default_dimension(self):
    gmoe_mod = self.GMOE
    rnn_mod = self.RNN
    expert = rnn_mod.ModelParams(
        rnn=rnn_mod.RnnParams(wxh=0.25, whh=-0.1, bh=0.05),
        per=rnn_mod.PerceptronParams(wxh=[0.1, 0.2, 0.3], bh=0.4),
    )
    params = gmoe_mod.GmoeParams(
        gate=gmoe_mod.GateParams(
            w=([1.0, 2.0, 3.0], 0.5, [-1.0, 0.0, 1.0]),
            b=(0.1, 0.2, 0.3),
        ),
        experts=(expert, expert, expert),
    )
    norm = gmoe_mod.normalize_chained_params(params, dimension=3)
    self.assertEqual(norm.gate.w[1], [0.5, 0.5, 0.5])
    self.assertEqual(norm.experts[0].per.bh, [0.4, 0.4, 0.4])

    exp_trace = rnn_mod.ModelParams(
        rnn=rnn_mod.RnnParams(wxh=[0.25] * 3, whh=[-0.1] * 3, bh=[0.05] * 3),
        per=rnn_mod.PerceptronParams(wxh=[0.1, 0.2, 0.3], bh=[0.4] * 3),
    )
    traces = gmoe_mod.GmoeParams(
        gate=gmoe_mod.GateParams(
            w=([1.0, 2.0, 3.0], [0.5] * 3, [-1.0, 0.0, 1.0]),
            b=([0.1] * 3, [0.2] * 3, [0.3] * 3),
        ),
        experts=(exp_trace, exp_trace, exp_trace),
    )
    self.assertEqual(
        gmoe_mod.extract_chained_params(traces, dimension=3), norm
    )

    bad_bool = params._replace(
        gate=gmoe_mod.GateParams(w=params.gate.w, b=(True, 0.0, 0.0))
    )
    with self.assertRaisesRegex(TypeError, 'not bool'):
      gmoe_mod.normalize_chained_params(bad_bool, dimension=3)


if __name__ == '__main__':
  absltest.main()
