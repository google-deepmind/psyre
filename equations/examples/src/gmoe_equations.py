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
"""Sparsely-gated MoE with AD via `psyre_equations` (experimental, WIP).

Note: Everything under `psyre/equations/` is experimental and a work in
progress; APIs and internals may change without notice. Replicates
`examples/src/gmoe.py` without explicit `lambda:` thunks, relying on
`with psyre_equations` and automatic stream naming (see Shazeer et al.,
ICLR '17, arXiv:1701.06538; Potop-Butucaru, Cohen, Plotkin & Pompougnac, 2023,
arXiv:2311.16977).
"""

from collections.abc import Callable, Sequence
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.automaton import (
    automaton,
    state,
    until,
)
from psyre.common import (
    fby,
    ifthenelse,
    is_namedtuple,
    merge,
    prefix_sum,
    psyre_node,
    push_prefix,
    require_sync_semantics,
    reset,
    when,
)
from psyre.equations import psyre_equations
from equations.examples.src import rnn_equations
from examples.src.gmoe import (
    ChainedGmoeParamStreams,
    ChainedGmoeParamTraces,
    ChainedGmoeParamValues,
    DEFAULT_CHAINED_LR,
    DEFAULT_CHAINED_MAX_STEPS,
    DEFAULT_EXPERT_CELLS,
    DEFAULT_SCALAR_LR,
    DEFAULT_SCALAR_MAX_STEPS,
    GateParams,
    GmoeForwardOutput,
    GmoeGradOutput,
    GmoeParams,
    GmoeTrainingOutput,
    GmoeTrainingState,
    NUM_EXPERTS,
    ScalarExpertParams,
    ScalarGmoeParamStreams,
    ScalarGmoeParamTraces,
    ScalarGmoeParamValues,
    _grad_sparse_top2_softmax,
    _make_subclock_shift_back,
    _pass_coords,
    _periodic_chained_param_streams,
    _triple,
    _validate_expert_cells,
    _validate_training_args,
    chained_param_streams,
    demo_expert1,
    demo_expert2,
    demo_expert3,
    extract_chained_params,
    extract_scalar_params,
    grad_gating_network,
    grad_scalar_expert,
    normalize_chained_params,
    scalar_expert,
)
from examples.src.rnn import _to_param_float
from psyre.match import (
    case,
    last,
    match,
)
from psyre.math import exp
from psyre.static_stream import repeat
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import

__all__ = (
    'ChainedGmoeParamStreams',
    'ChainedGmoeParamTraces',
    'ChainedGmoeParamValues',
    'DEFAULT_CHAINED_LR',
    'DEFAULT_CHAINED_MAX_STEPS',
    'DEFAULT_EXPERT_CELLS',
    'DEFAULT_SCALAR_LR',
    'DEFAULT_SCALAR_MAX_STEPS',
    'GateParams',
    'GmoeForwardOutput',
    'GmoeGradOutput',
    'GmoeParams',
    'GmoeTrainingOutput',
    'GmoeTrainingState',
    'NUM_EXPERTS',
    'ScalarExpertParams',
    'ScalarGmoeParamStreams',
    'ScalarGmoeParamTraces',
    'ScalarGmoeParamValues',
    'apply_chained_gmoe_grads',
    'chained_gmoe_forward',
    'chained_param_streams',
    'demo_expert1',
    'demo_expert2',
    'demo_expert3',
    'eval_chained_gmoe',
    'extract_chained_params',
    'extract_scalar_params',
    'fixed_gating_network',
    'gating_network',
    'gmoe_forward',
    'grad_chained_gmoe',
    'grad_gating_network',
    'grad_gmoe',
    'grad_pass_gating_network',
    'grad_scalar_expert',
    'mixture_of_experts_demo',
    'normalize_chained_params',
    'pass_gating_network',
    'scalar_expert',
    'train_chained_gmoe',
    'train_gmoe',
)


def _delay_tree(
    template: Any,
    tree: Any,
    delay_fn: Callable[[Any, Any], Stream],
) -> Any:
  """Recursively delays every leaf of a NamedTuple/tuple parameter tree."""
  if isinstance(template, tuple):
    items = tuple(
        _delay_tree(template[i], tree[i], delay_fn)
        for i in range(len(template))
    )
    return type(template)(*items) if is_namedtuple(template) else items
  return delay_fn(template, tree)


def _make_last_delay(dimension: int) -> Callable[[Any, Any], Stream]:
  """Returns a pass-length `last(..., init=0.0, n=dimension)` delay helper."""

  def _delay(_: Any, src: Any) -> Stream:
    return last(src, init=0.0, n=dimension)

  return _delay


def _merge_expert_outputs(
    conds: tuple[Stream, Stream, Stream],
    expert_outputs: tuple[Stream, Stream, Stream],
) -> tuple[Stream, Stream, Stream]:
  """Merges subclock expert outputs back to base clock with 0.0 fill."""
  return _triple(lambda k: merge(conds[k], expert_outputs[k], 0.0))


def _sample_expert_outputs(
    conds: tuple[Stream, Stream, Stream],
    merged_outputs: tuple[Stream, Stream, Stream],
) -> tuple[Stream, Stream, Stream]:
  """Re-samples merged base-clock expert outputs onto subclocks `c1..c3`."""
  return _triple(
      lambda k: when(conds[k], merged_outputs[k], name=f'o{k + 1}')
  )


def _dispatch_conds(
    gates: tuple[Stream, Stream, Stream],
) -> tuple[Stream, Stream, Stream]:
  """Computes auto-named boolean dispatch clocks `c1`, `c2`, `c3`."""
  with psyre_equations() as eq:

    class _:
      c1 = gates[0] != 0.0
      c2 = gates[1] != 0.0
      c3 = gates[2] != 0.0
      __output__ = (c1, c2, c3)

  return eq.output


def _combine_experts(
    gates: tuple[Stream, Stream, Stream],
    conds: tuple[Stream, Stream, Stream],
    expert_outputs: tuple[Stream, Stream, Stream],
) -> tuple[Stream, tuple[Stream, Stream, Stream]]:
  """Merges subclock expert outputs back to base clock and forms gated sum."""
  with psyre_equations() as eq:

    class _:
      m1 = merge(conds[0], expert_outputs[0], 0.0)
      m2 = merge(conds[1], expert_outputs[1], 0.0)
      m3 = merge(conds[2], expert_outputs[2], 0.0)
      o = Stream.sum(gates[0] * m1, gates[1] * m2, gates[2] * m3)
      __output__ = (o, (m1, m2, m3))

  return eq.output


def _lift_expert_grads[ExpT](
    conds: tuple[Stream, Stream, Stream],
    dx_gate: Stream,
    dx_subs: tuple[Stream, Stream, Stream],
    dp_subs: tuple[ExpT, ExpT, ExpT],
) -> tuple[Stream, tuple[ExpT, ExpT, ExpT]]:
  """Lifts subclock expert input and parameter adjoints to base clock."""

  def lift_dp(k: int) -> ExpT:
    c_k = conds[k]
    return dp_subs[k].map(lambda g_sub: merge(c_k, g_sub, 0.0))

  with psyre_equations() as eq:

    class _:
      dx_e1 = merge(conds[0], dx_subs[0], 0.0)
      dx_e2 = merge(conds[1], dx_subs[1], 0.0)
      dx_e3 = merge(conds[2], dx_subs[2], 0.0)
      dx = Stream.sum(dx_gate, dx_e1, dx_e2, dx_e3)
      __output__ = (dx, _triple(lift_dp))

  return eq.output


# ---------------------------------------------------------------------------
# 1. Fixed-gate Mixture of Experts demo
# ---------------------------------------------------------------------------


@psyre_node('gating_network')
def fixed_gating_network(x: Stream) -> tuple[Stream, Stream, Stream]:
  """Fixed 8-cycle sparse gating schedule from arXiv:2311.16977 Fig. 12."""
  _ = x
  with psyre_equations() as eq:

    class _:
      g1 = Stream([0.6, 0.0, 0.5, 0.0, 1.0, 0.0, 0.4, 0.0])
      g2 = Stream([0.4, 0.7, 0.0, 0.0, 0.0, 0.8, 0.6, 0.0])
      g3 = Stream([0.0, 0.3, 0.5, 1.0, 0.0, 0.2, 0.0, 1.0])
      __output__ = (g1, g2, g3)

  return eq.output


def mixture_of_experts_demo() -> dict[str, Stream]:
  """Sparsely-gated Mixture of Experts with 3 experts (arXiv:2311.16977)."""
  with psyre_equations() as eq:

    class _:
      """Fixed-gate 3-expert MoE demo equations."""
      x = Stream([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0])
      gates = fixed_gating_network(x)
      conds = _dispatch_conds(gates)

      x_c1 = when(conds[0], x, name='x_c1')
      o1 = demo_expert1(x_c1)
      o2 = demo_expert2(when(conds[1], x, name='x_c2'))
      o3 = demo_expert3(when(conds[2], x, name='x_c3'))
      o, _ = _combine_experts(gates, conds, (o1, o2, o3))

      with match(conds[0]) as rec_e1:
        with case(True):
          rec_e1.out = prefix_sum(x_c1, start=0.0, name='e1_state')
        with case(False):
          rec_e1.out = 0.0

      __output__ = {
          'x': x,
          'g1': gates[0],
          'g2': gates[1],
          'g3': gates[2],
          'o1': o1,
          'o2': o2,
          'o3': o3,
          'o': o,
          'e1_recurrent': rec_e1.out,
      }

  return eq.output


# ---------------------------------------------------------------------------
# 2. Trainable Sparsely-Gated MoE (Scalar Experts) with Reverse-Mode AD
# ---------------------------------------------------------------------------


def _sparse_top2_softmax(
    s0: Stream, s1: Stream, s2: Stream
) -> tuple[Stream, Stream, Stream]:
  """Top-2 sparse softmax over 3 logits; drops the minimum logit to 0.0."""
  with psyre_equations() as eq:

    class _:
      """Top-2 sparse softmax equations."""
      drop2 = (s2 <= s0) & (s2 <= s1)
      drop1 = (~drop2) & (s1 <= s0)
      drop0 = (~drop2) & (~drop1)
      s_max = ifthenelse(
          drop2,
          s0.max(s1),
          ifthenelse(drop1, s0.max(s2), s1.max(s2)),
      )
      e0 = ifthenelse(~drop0, exp(s0 - s_max), 0.0)
      e1 = ifthenelse(~drop1, exp(s1 - s_max), 0.0)
      e2 = ifthenelse(~drop2, exp(s2 - s_max), 0.0)
      denom = Stream.sum(e0, e1, e2)
      g1 = e0 / denom
      g2 = e1 / denom
      g3 = e2 / denom
      __output__ = (g1, g2, g3)

  return eq.output


@psyre_node('gating_network')
def gating_network(
    x: Stream, gate: GateParams[Stream]
) -> tuple[Stream, Stream, Stream]:
  """Top-2 sparse softmax gating network over scalar input stream `x`."""
  return _sparse_top2_softmax(*_triple(lambda k: gate.w[k] * x + gate.b[k]))


def _scalar_forward(
    x: Stream, params: ScalarGmoeParamStreams
) -> tuple[
    GmoeForwardOutput,
    tuple[Stream, Stream, Stream],
    tuple[
        ScalarExpertParams[Stream],
        ScalarExpertParams[Stream],
        ScalarExpertParams[Stream],
    ],
    tuple[Stream, Stream, Stream],
]:
  """Shared forward pass for `gmoe_forward` and `grad_gmoe`."""
  with psyre_equations() as eq:

    class _:
      """Scalar GMoE forward dispatch equations."""
      gates = gating_network(x, params.gate)
      conds = _dispatch_conds(gates)
      x_subs_list = []
      p_subs_list = []
      o_subs_list = []
      for k in range(NUM_EXPERTS):
        c_k = conds[k]
        x_k = when(c_k, x, name=f'x_c{k + 1}')
        p_k = rnn_equations._sample_params(c_k, params.experts[k])  # pylint: disable=protected-access
        o_k = scalar_expert(x_k, p_k, name=f'expert{k + 1}')
        o_k.name = f'o{k + 1}'
        x_subs_list.append(x_k)
        p_subs_list.append(p_k)
        o_subs_list.append(o_k)
      x_subs = tuple(x_subs_list)
      p_subs = tuple(p_subs_list)
      o_subs = tuple(o_subs_list)
      o, merged_os = _combine_experts(gates, conds, o_subs)
      fwd = GmoeForwardOutput(
          o=o, gates=gates, conds=conds, expert_outputs=o_subs
      )
      __output__ = (fwd, x_subs, p_subs, merged_os)

  return eq.output


def gmoe_forward(
    x: Stream, params: ScalarGmoeParamStreams
) -> GmoeForwardOutput:
  """Sparsely-gated MoE forward pass dynamically dispatching 3 scalar experts."""
  require_sync_semantics('gmoe_forward')
  return _scalar_forward(x, params)[0]


def grad_gmoe(
    x: Stream,
    y_true: Stream | float,
    params: ScalarGmoeParamStreams,
) -> GmoeGradOutput[ScalarExpertParams[Stream]]:
  """Forward and reverse-mode AD of sparsely-gated MoE with scalar experts."""
  require_sync_semantics('grad_gmoe')
  with psyre_equations() as eq:

    class _:
      """Scalar GMoE reverse-mode AD equations."""
      fwd, x_subs, p_subs, merged_os = _scalar_forward(x, params)
      loss, do = rnn_equations.grad_loss(fwd.o, y_true, 1.0)
      dg = (do * merged_os[0], do * merged_os[1], do * merged_os[2])
      dx_gate, dgate = grad_gating_network(x, params.gate, fwd.gates, dg)
      dx_subs_list = []
      dp_subs_list = []
      for k in range(NUM_EXPERTS):
        do_k = when(fwd.conds[k], do * fwd.gates[k], name=f'do{k + 1}')
        dx_k, dp_k = grad_scalar_expert(
            x_subs[k], p_subs[k], do_k, name=f'grad_expert{k + 1}'
        )
        dx_subs_list.append(dx_k)
        dp_subs_list.append(dp_k)
      dx, dexperts = _lift_expert_grads(
          fwd.conds, dx_gate, tuple(dx_subs_list), tuple(dp_subs_list)
      )
      __output__ = GmoeGradOutput(
          o=fwd.o,
          loss=loss,
          dx=dx,
          grads=GmoeParams(gate=dgate, experts=dexperts),
          gates=fwd.gates,
          conds=fwd.conds,
          expert_outputs=fwd.expert_outputs,
      )

  return eq.output


def train_gmoe(
    dataset: Sequence[tuple[float, float]],
    initial_params: ScalarGmoeParamValues,
    *,
    max_steps: int = DEFAULT_SCALAR_MAX_STEPS,
    lr: float = DEFAULT_SCALAR_LR,
) -> GmoeTrainingOutput[ScalarExpertParams[Stream]]:
  """Trains the sparsely-gated scalar MoE with dynamic dispatch and SGD."""
  require_sync_semantics('train_gmoe')
  _validate_training_args(dataset, max_steps)

  with psyre_equations() as eq:

    class _:
      """Scalar GMoE SGD training loop equations."""
      x = Stream.periodic([], [float(item[0]) for item in dataset])
      y = Stream.periodic([], [float(item[1]) for item in dataset])
      active_step = Stream.periodic(repeat(True, max_steps), [False])

      prev_params = _delay_tree(
          initial_params,
          # pylint: disable-next=undefined-variable,used-before-assignment
          next_params,  # pyrefly: ignore[unknown-name]
          lambda init_val, src: fby(
              _to_param_float(init_val, 'initial_params'), src
          ),
      )
      grad_out = grad_gmoe(x, y, prev_params)
      updated = prev_params.map(
          lambda p, g: p - lr * g, grad_out.grads
      )
      next_params = rnn_equations._select_params(  # pylint: disable=protected-access
          active_step, updated, prev_params
      )
      train_loss = when(active_step, grad_out.loss)
      __output__ = GmoeTrainingOutput(
          params=next_params,
          loss=train_loss,
          gates=grad_out.gates,
          conds=grad_out.conds,
          expert_outputs=grad_out.expert_outputs,
      )

  return eq.output


# ---------------------------------------------------------------------------
# 3. Chained GMoE with RNN + Perceptron Experts (variable unroll depths)
# ---------------------------------------------------------------------------


@psyre_node('pass_gating_network')
def pass_gating_network(
    x: Stream,
    gate: GateParams[Stream],
    *,
    dimension: int = rnn_equations.DIMENSION,
    pass_end: Stream | None = None,
) -> tuple[Stream, Stream, Stream]:
  """Pass-constant top-2 sparse softmax gating over sequence `x[0..D-1]`."""
  scores = _triple(
      lambda k: (
          rnn_equations.sum_stream(
              x * gate.w[k], dimension=dimension, end=pass_end
          )
          + gate.b[k]
      )
  )
  return _sparse_top2_softmax(*scores)


@psyre_node('grad_pass_gating_network')
def grad_pass_gating_network(
    x: Stream,
    gate: GateParams[Stream],
    gates: tuple[Stream, Stream, Stream],
    dg: tuple[Stream, Stream, Stream],
    *,
    dimension: int = rnn_equations.DIMENSION,
    pass_end: Stream | None = None,
) -> tuple[Stream, GateParams[Stream]]:
  """Reverse-mode backward pass of `pass_gating_network`."""
  ds = _grad_sparse_top2_softmax(gates, dg)
  ds_total = _triple(
      lambda k: rnn_equations.sum_stream(
          ds[k], dimension=dimension, end=pass_end
      )
  )
  dx_gate = Stream.sum(*(ds_total[k] * gate.w[k] for k in range(NUM_EXPERTS)))
  dgate = GateParams(
      w=_triple(lambda k: ds_total[k] * x),
      b=ds_total,
  )
  return dx_gate, dgate


def chained_gmoe_forward(
    x: Stream,
    params: ChainedGmoeParamStreams,
    *,
    expert_cells: Sequence[int] = DEFAULT_EXPERT_CELLS,
    dimension: int = rnn_equations.DIMENSION,
    in_pass: Stream | None = None,
    pass_reset: Stream | None = None,
) -> GmoeForwardOutput:
  """Forward pass chaining sparse MoE with 3 RNN+perceptron experts."""
  require_sync_semantics('chained_gmoe_forward')
  cells = _validate_expert_cells(expert_cells, dimension)
  with psyre_equations() as eq:

    class _:
      """Chained GMoE-RNN forward pass equations."""
      pos, rst, pass_final = _pass_coords(dimension, in_pass, pass_reset)
      with reset(rst):
        gates = pass_gating_network(
            x, params.gate, dimension=dimension, pass_end=pass_final
        )
      conds = _dispatch_conds(gates)
      o_subs_list = []
      for k in range(NUM_EXPERTS):
        c_k = conds[k]
        x_k = when(c_k, x, name=f'x_e{k + 1}')
        pos_k = when(c_k, pos, name=f'pos_e{k + 1}')
        rst_k = when(c_k, rst, name=f'rst_e{k + 1}')
        p_k = rnn_equations._sample_params(c_k, params.experts[k])  # pylint: disable=protected-access
        with push_prefix(f'expert{k + 1}'):
          with reset(rst_k):
            o_k = rnn_equations.forward(
                x_k, p_k, num_cells=cells[k], in_pass=pos_k
            )
        o_k.name = f'o{k + 1}'
        o_subs_list.append(o_k)
      o_subs = tuple(o_subs_list)
      o, _ = _combine_experts(gates, conds, o_subs)
      __output__ = GmoeForwardOutput(
          o=o, gates=gates, conds=conds, expert_outputs=o_subs
      )

  return eq.output


def grad_chained_gmoe(
    x: Stream,
    y_true: Stream | float,
    params: ChainedGmoeParamStreams,
    *,
    expert_cells: Sequence[int] = DEFAULT_EXPERT_CELLS,
    dimension: int = rnn_equations.DIMENSION,
    in_pass: Stream | None = None,
    pass_reset: Stream | None = None,
) -> GmoeGradOutput[rnn_equations.ParamStreams]:
  """Forward + reverse-mode AD of GMoE chaining 3 RNN+perceptron experts."""
  require_sync_semantics('grad_chained_gmoe')
  cells = _validate_expert_cells(expert_cells, dimension)

  with psyre_equations() as eq:

    class _:
      """Chained GMoE-RNN forward and reverse-mode AD equations."""
      pos, rst, pass_final = _pass_coords(dimension, in_pass, pass_reset)
      make_shift_back = _make_subclock_shift_back(rst)
      with reset(rst):
        gates = pass_gating_network(
            x, params.gate, dimension=dimension, pass_end=pass_final
        )
      conds = _dispatch_conds(gates)

      o_subs_list = []
      dx_subs_list = []
      dp_subs_list = []
      for k in range(NUM_EXPERTS):
        c_k = conds[k]
        x_k = when(c_k, x, name=f'x_e{k + 1}')
        pos_k = when(c_k, pos, name=f'pos_e{k + 1}')
        rst_k = when(c_k, rst, name=f'rst_e{k + 1}')
        p_k = rnn_equations._sample_params(c_k, params.experts[k])  # pylint: disable=protected-access
        with push_prefix(f'expert{k + 1}'):
          with reset(rst_k):
            o_k, dx_k, dp_k = rnn_equations.grad_rnn_perceptron(
                x_k,
                p_k,
                # pylint: disable-next=undefined-variable,used-before-assignment
                do_subs[k],  # pyrefly: ignore[unknown-name]
                num_cells=cells[k],
                dimension=dimension,
                in_pass=pos_k,
                shift_back=make_shift_back(c_k),
            )
        o_k.name = f'o{k + 1}'
        o_subs_list.append(o_k)
        dx_subs_list.append(dx_k)
        dp_subs_list.append(dp_k)

      o_subs = tuple(o_subs_list)
      dx_subs = tuple(dx_subs_list)
      dp_subs = tuple(dp_subs_list)
      o, merged_os = _combine_experts(gates, conds, o_subs)

      loss, do = rnn_equations.grad_loss(
          o, y_true, ifthenelse(pass_final, 1.0, 0.0)
      )
      dg = (do * merged_os[0], do * merged_os[1], do * merged_os[2])
      with reset(rst):
        dx_gate, dgate = grad_pass_gating_network(
            x, params.gate, gates, dg, dimension=dimension, pass_end=pass_final
        )
      do_subs = (
          when(conds[0], do * gates[0], name='do1'),
          when(conds[1], do * gates[1], name='do2'),
          when(conds[2], do * gates[2], name='do3'),
      )
      dx, dexperts = _lift_expert_grads(conds, dx_gate, dx_subs, dp_subs)
      __output__ = GmoeGradOutput(
          o=o,
          loss=loss,
          dx=dx,
          grads=GmoeParams(gate=dgate, experts=dexperts),
          gates=gates,
          conds=conds,
          expert_outputs=o_subs,
      )

  return eq.output


def apply_chained_gmoe_grads(
    params: ChainedGmoeParamStreams,
    grads: ChainedGmoeParamStreams,
    lr: float,
    *,
    dimension: int = rnn_equations.DIMENSION,
    pass_end: Stream | None = None,
) -> ChainedGmoeParamStreams:
  """Applies one gradient descent step to gate and 3 RNN+perceptron experts."""
  return GmoeParams(
      gate=params.gate.map(lambda p, g: p - lr * g, grads.gate),
      experts=_triple(
          lambda k: rnn_equations.apply_grads(
              params.experts[k],
              grads.experts[k],
              lr,
              dimension=dimension,
              end=pass_end,
          )
      ),
  )


def eval_chained_gmoe(
    dataset: Sequence[tuple[Sequence[float], float]],
    params: ChainedGmoeParamValues,
    *,
    expert_cells: Sequence[int] = DEFAULT_EXPERT_CELLS,
    dimension: int = rnn_equations.DIMENSION,
) -> Stream:
  """Evaluates squared loss of chained GMoE-RNN across every sample in dataset."""
  require_sync_semantics('eval_chained_gmoe')
  if not dataset:
    raise ValueError('dataset must be non-empty.')
  x_word, y_word = rnn_equations._dataset_words(dataset, dimension=dimension)  # pylint: disable=protected-access
  with psyre_equations() as eq:

    class _:
      """Chained GMoE-RNN dataset evaluation equations."""
      in_pass = Stream.periodic([], list(range(dimension)))
      pass_reset = Stream.periodic(
          [], rnn_equations._pass_reset_word(dimension)  # pylint: disable=protected-access
      )
      p_streams = _periodic_chained_param_streams(params, dimension=dimension)
      fwd = chained_gmoe_forward(
          Stream.periodic(x_word, [0.0]),
          p_streams,
          expert_cells=expert_cells,
          dimension=dimension,
          in_pass=in_pass,
          pass_reset=pass_reset,
      )
      emit = Stream.periodic(
          repeat(rnn_equations._pass_end_word(dimension), len(dataset)), [False]  # pylint: disable=protected-access
      )
      eval_loss = when(
          emit, rnn_equations.loss(fwd.o, Stream.periodic(y_word, [0.0]))
      )
      __output__ = eval_loss

  return eq.output


def train_chained_gmoe(
    dataset: Sequence[tuple[Sequence[float], float]],
    initial_params: ChainedGmoeParamValues,
    *,
    expert_cells: Sequence[int] = DEFAULT_EXPERT_CELLS,
    max_steps: int = DEFAULT_CHAINED_MAX_STEPS,
    lr: float = DEFAULT_CHAINED_LR,
    dimension: int = rnn_equations.DIMENSION,
) -> GmoeTrainingOutput[rnn_equations.ParamStreams]:
  """Multi-pass BPTT training automaton for the chained GMoE-RNN model."""
  require_sync_semantics('train_chained_gmoe')
  _validate_training_args(dataset, max_steps)
  _validate_expert_cells(expert_cells, dimension)
  norm_init = normalize_chained_params(initial_params, dimension=dimension)
  x_word, y_word = rnn_equations._dataset_words(dataset, dimension=dimension)  # pylint: disable=protected-access
  delay_pass = _make_last_delay(dimension)

  with psyre_equations() as eq:

    class _:
      """Chained GMoE-RNN multi-pass BPTT training automaton equations."""
      with automaton(
          init=GmoeTrainingState.INIT_LOAD, name='gmoe_train'
      ) as train:
        prev = _delay_tree(norm_init, train.params, delay_pass)
        with state(GmoeTrainingState.INIT_LOAD):
          until(Stream.once(dimension - 1), GmoeTrainingState.GRAD_DESCENT)
          train.params = _periodic_chained_param_streams(
              norm_init, dimension=dimension
          )
          train.loss = 0.0
          train.emit_loss = False
          train.gates = (0.0, 0.0, 0.0)
          train.conds = (False, False, False)
          train.merged_outputs = (0.0, 0.0, 0.0)
        with state(GmoeTrainingState.GRAD_DESCENT):
          in_pass = Stream.periodic([], list(range(dimension)))
          pass_reset = Stream.periodic(
              [], rnn_equations._pass_reset_word(dimension)  # pylint: disable=protected-access
          )
          pass_final = Stream.periodic(
              [], rnn_equations._pass_end_word(dimension)  # pylint: disable=protected-access
          )
          sample_x = Stream.periodic([], x_word)
          sample_y = Stream.periodic([], y_word)
          active_step = Stream.periodic(
              repeat(True, max_steps * dimension), [False]
          )

          grad_out = grad_chained_gmoe(
              sample_x,
              sample_y,
              prev,
              expert_cells=expert_cells,
              dimension=dimension,
              in_pass=in_pass,
              pass_reset=pass_reset,
          )
          with reset(pass_reset):
            updated = apply_chained_gmoe_grads(
                prev,
                grad_out.grads,
                lr,
                dimension=dimension,
                pass_end=pass_final,
            )
          train.params = rnn_equations._select_params(  # pylint: disable=protected-access
              active_step, updated, prev
          )
          train.loss = grad_out.loss
          train.emit_loss = Stream.periodic(
              repeat(rnn_equations._pass_end_word(dimension), max_steps),  # pylint: disable=protected-access
              [False],
          )
          train.gates = grad_out.gates
          train.conds = grad_out.conds
          train.merged_outputs = _merge_expert_outputs(
              grad_out.conds, grad_out.expert_outputs
          )

      train_loss = when(train.emit_loss, train.loss)
      __output__ = GmoeTrainingOutput(
          params=train.params,
          loss=train_loss,
          gates=train.gates,
          conds=train.conds,
          expert_outputs=_sample_expert_outputs(
              train.conds, train.merged_outputs
          ),
      )

  return eq.output
