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
"""Sparsely-gated Mixture of Experts (GMoE) with reverse-mode AD and training.

Includes:
1. The fixed-gate Mixture of Experts demo from arXiv:2311.16977 (Fig. 12 &
   §4.4), moved out of `notebooks/simple_colab.ipynb`.
2. A trainable sparsely-gated MoE with 3 dynamically dispatched scalar experts
   (`c_k = (g_k != 0.0)`), exact reverse-mode differentiation across
   `when`/`merge` subclocks, and backpropagation training.
3. A chained GMoE whose 3 dynamically dispatched experts each run the
   `examples/rnn.py` RNN + ReLU perceptron model with distinct unrolling depths
   (default `(3, 4, 5)` cells), complete with pass-level sparse gating,
   reverse-mode differentiation, and multi-pass BPTT training.
"""

from collections.abc import Callable, Sequence
import enum
from typing import Any, Generic, NamedTuple, TypeVar

# pylint: disable=g-importing-member,g-multiple-import
from psyre.automaton import (
    automaton,
    state,
    until,
)
from psyre.common import (
    CapturedEnv,
    counter,
    fby,
    ifthenelse,
    merge,
    post,
    prefix_stack,
    prefix_sum,
    psyre_node,
    push_prefix,
    require_sync_semantics,
    reset,
    switch,
    when,
)
from examples import rnn
from psyre.match import (
    case,
    last,
    match,
)
from psyre.math import exp
from psyre.stream import Stream
from psyre.trace import present_values

# pylint: enable=g-importing-member,g-multiple-import

NUM_EXPERTS = 3
DEFAULT_EXPERT_CELLS: tuple[int, int, int] = (3, 4, 5)
DEFAULT_SCALAR_LR = 0.02
DEFAULT_SCALAR_MAX_STEPS = 20
DEFAULT_CHAINED_LR = 0.01
DEFAULT_CHAINED_MAX_STEPS = 12

_T = TypeVar('_T')
_U = TypeVar('_U')
_GateT = TypeVar('_GateT')
_ExpT = TypeVar('_ExpT')


def _triple(fn: Callable[[int], _T]) -> tuple[_T, _T, _T]:
  """Evaluates `fn(k)` for expert indices `0, 1, 2` as a typed 3-tuple."""
  return (fn(0), fn(1), fn(2))


def _delay_tree(
    template: Any,
    get_tree: Callable[[], Any],
    delay_fn: Callable[[Any, Callable[[], Stream]], Stream],
) -> Any:
  """Recursively delays every leaf of a NamedTuple/tuple parameter tree."""
  if isinstance(template, tuple):

    def child(idx: int) -> Any:
      return _delay_tree(template[idx], lambda: get_tree()[idx], delay_fn)

    items = tuple(child(i) for i in range(len(template)))
    cls: Any = type(template)
    return items if cls is tuple else cls(*items)
  return delay_fn(template, get_tree)


def _vector_stream(in_pass: Stream, values: Sequence[float]) -> Stream:
  """Selects `values[in_pass]` at each local coordinate of a pass."""
  return switch(
      in_pass,
      {coord: float(val) for coord, val in enumerate(values)},
      default=0.0,
  )


def _validate_training_args(dataset: Sequence[Any], max_steps: int) -> None:
  """Validates non-empty dataset and positive training step count."""
  if not dataset:
    raise ValueError('dataset must be non-empty.')
  if max_steps < 1:
    raise ValueError(f'max_steps must be >= 1, got {max_steps}.')


def _dispatch_conds(
    gates: tuple[Stream, Stream, Stream],
) -> tuple[Stream, Stream, Stream]:
  """Computes named boolean dispatch clocks `c_k = (g_k != 0.0)`."""

  def make_cond(k: int) -> Stream:
    c_k = gates[k] != 0.0
    c_k.name = f'c{k + 1}'
    return c_k

  return _triple(make_cond)


def _combine_experts(
    gates: tuple[Stream, Stream, Stream],
    conds: tuple[Stream, Stream, Stream],
    expert_outputs: tuple[Stream, Stream, Stream],
) -> tuple[Stream, tuple[Stream, Stream, Stream]]:
  """Merges subclock expert outputs back to base clock and forms gated sum."""
  merged_os = _triple(
      lambda k: merge(conds[k], expert_outputs[k], 0.0, name=f'm{k + 1}')
  )
  o = Stream.sum(*(gates[k] * merged_os[k] for k in range(NUM_EXPERTS)))
  o.name = 'o'
  return o, merged_os


def _lift_expert_grads(
    conds: tuple[Stream, Stream, Stream],
    dx_gate: Stream,
    dx_subs: tuple[Stream, Stream, Stream],
    dp_subs: tuple[_ExpT, _ExpT, _ExpT],
) -> tuple[Stream, tuple[_ExpT, _ExpT, _ExpT]]:
  """Lifts subclock expert input and parameter adjoints to base clock."""

  def lift_dp(k: int) -> _ExpT:
    c_k = conds[k]
    return dp_subs[k].map(lambda g_sub: merge(c_k, g_sub, 0.0))

  dx = Stream.sum(
      dx_gate,
      *(
          merge(conds[k], dx_subs[k], 0.0, name=f'dx_e{k + 1}')
          for k in range(NUM_EXPERTS)
      ),
  )
  dx.name = 'dx'
  return dx, _triple(lift_dp)


def _pass_coords(
    dimension: int,
    in_pass: Stream | None = None,
    pass_reset: Stream | None = None,
) -> tuple[Stream, Stream, Stream]:
  """Returns `(pos, rst, pass_final)` coordinate streams for a pass."""
  pos = counter() % dimension if in_pass is None else in_pass
  rst = (pos == 0) if pass_reset is None else pass_reset
  return pos, rst, pos == (dimension - 1)


class GmoeTrainingState(enum.Enum):
  """States of the multi-pass chained GMoE training automaton."""

  INIT_LOAD = enum.auto()
  GRAD_DESCENT = enum.auto()


class GateParams(NamedTuple, Generic[_T]):
  """Parameters `(w, b)` for a 3-expert linear sparse gating network."""

  w: tuple[_T, _T, _T]
  b: tuple[_T, _T, _T]

  def map(
      self, fn: Callable[..., _U], *others: 'GateParams[Any]'
  ) -> 'GateParams[_U]':
    """Applies fn leaf-wise across self and any parallel GateParams."""
    return GateParams(
        w=_triple(lambda k: fn(self.w[k], *(o.w[k] for o in others))),
        b=_triple(lambda k: fn(self.b[k], *(o.b[k] for o in others))),
    )


class ScalarExpertParams(NamedTuple, Generic[_T]):
  """Parameters `(w1, w2, b)` for quadratic/affine expert `w1*x + w2*x^2 + b`."""

  w1: _T
  w2: _T
  b: _T

  def map(
      self, fn: Callable[..., _U], *others: 'ScalarExpertParams[Any]'
  ) -> 'ScalarExpertParams[_U]':
    """Applies fn field-wise across self and parallel ScalarExpertParams."""
    return ScalarExpertParams(*(fn(*col) for col in zip(self, *others)))


class GmoeParams(NamedTuple, Generic[_GateT, _ExpT]):
  """Composite parameters bundling gate parameters and 3 expert parameter sets."""

  gate: GateParams[_GateT]
  experts: tuple[_ExpT, _ExpT, _ExpT]

  def map(
      self, fn: Callable[..., Any], *others: 'GmoeParams[Any, Any]'
  ) -> 'GmoeParams[Any, Any]':
    """Applies fn leaf-wise across gate and all 3 expert parameter trees."""
    return GmoeParams(
        gate=self.gate.map(fn, *(o.gate for o in others)),
        experts=_triple(
            lambda k: self.experts[k].map(fn, *(o.experts[k] for o in others))
        ),
    )


ScalarGmoeParamValues = GmoeParams[float, ScalarExpertParams[float]]
ScalarGmoeParamStreams = GmoeParams[Stream, ScalarExpertParams[Stream]]
ScalarGmoeParamTraces = GmoeParams[list[Any], ScalarExpertParams[list[Any]]]

ChainedGmoeParamValues = GmoeParams[Any, rnn.ParamValues]
ChainedGmoeParamStreams = GmoeParams[Stream, rnn.ParamStreams]
ChainedGmoeParamTraces = GmoeParams[list[Any], rnn.ParamTraces]


class GmoeForwardOutput(NamedTuple):
  """Output streams of a sparsely-gated MoE forward pass."""

  o: Stream
  gates: tuple[Stream, Stream, Stream]
  conds: tuple[Stream, Stream, Stream]
  expert_outputs: tuple[Stream, Stream, Stream]


class GmoeGradOutput(NamedTuple, Generic[_ExpT]):
  """Output streams of a sparsely-gated MoE forward + reverse-mode pass."""

  o: Stream
  loss: Stream
  dx: Stream
  grads: GmoeParams[Stream, _ExpT]
  gates: tuple[Stream, Stream, Stream]
  conds: tuple[Stream, Stream, Stream]
  expert_outputs: tuple[Stream, Stream, Stream]


class GmoeTrainingOutput(NamedTuple, Generic[_ExpT]):
  """Output streams of a sparsely-gated MoE training circuit."""

  params: GmoeParams[Stream, _ExpT]
  loss: Stream
  gates: tuple[Stream, Stream, Stream]
  conds: tuple[Stream, Stream, Stream]
  expert_outputs: tuple[Stream, Stream, Stream]


# ---------------------------------------------------------------------------
# 1. Fixed-gate Mixture of Experts demo (moved from notebooks/simple_colab)
# ---------------------------------------------------------------------------


@psyre_node('gating_network')
def fixed_gating_network(x: Stream) -> tuple[Stream, Stream, Stream]:
  """Fixed 8-cycle sparse gating schedule from arXiv:2311.16977 Fig. 12."""
  g1 = Stream([0.6, 0.0, 0.5, 0.0, 1.0, 0.0, 0.4, 0.0], name='g1')
  g2 = Stream([0.4, 0.7, 0.0, 0.0, 0.0, 0.8, 0.6, 0.0], name='g2')
  g3 = Stream([0.0, 0.3, 0.5, 1.0, 0.0, 0.2, 0.0, 1.0], name='g3')
  _ = x
  return g1, g2, g3


@psyre_node('expert1')
def demo_expert1(x_sub: Stream) -> Stream:
  """Fixed affine expert 1 (`2x + 1`) on subclock `. on c1`."""
  return 2.0 * x_sub + 1.0


@psyre_node('expert2')
def demo_expert2(x_sub: Stream) -> Stream:
  """Fixed affine expert 2 (`-x + 5`) on subclock `. on c2`."""
  return -1.0 * x_sub + 5.0


@psyre_node('expert3')
def demo_expert3(x_sub: Stream) -> Stream:
  """Fixed quadratic expert 3 (`0.5 x^2`) on subclock `. on c3`."""
  return 0.5 * x_sub * x_sub


def mixture_of_experts_demo() -> dict[str, Stream]:
  """Sparsely-gated Mixture of Experts with 3 experts (arXiv:2311.16977)."""
  x = Stream([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0], name='x')
  gates = fixed_gating_network(x)
  conds = _dispatch_conds(gates)

  o1 = demo_expert1(when(conds[0], x, name='x_c1'))
  o1.name = 'o1'
  o2 = demo_expert2(when(conds[1], x, name='x_c2'))
  o2.name = 'o2'
  o3 = demo_expert3(when(conds[2], x, name='x_c3'))
  o3.name = 'o3'
  o, _ = _combine_experts(gates, conds, (o1, o2, o3))

  # Time-modular stateful expert (§4.4): state advances only when `c1` is True.
  with match(conds[0]) as rec_e1:
    with case(True):
      rec_e1.out = prefix_sum(x, start=0.0, name='e1_state')
    with case(False):
      rec_e1.out = 0.0

  return {
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


# ---------------------------------------------------------------------------
# 2. Trainable Sparsely-Gated MoE (Scalar Experts) with Reverse-Mode AD
# ---------------------------------------------------------------------------


def _sparse_top2_softmax(
    s0: Stream, s1: Stream, s2: Stream
) -> tuple[Stream, Stream, Stream]:
  """Top-2 sparse softmax over 3 logits; drops the minimum logit to 0.0."""
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
  g0, g1, g2 = e0 / denom, e1 / denom, e2 / denom
  g0.name, g1.name, g2.name = 'g1', 'g2', 'g3'
  return g0, g1, g2


def _grad_sparse_top2_softmax(
    gates: tuple[Stream, Stream, Stream],
    dg: tuple[Stream, Stream, Stream],
) -> tuple[Stream, Stream, Stream]:
  """Reverse-mode VJP of `_sparse_top2_softmax` (`ds_k = g_k * (dg_k - dot)`)."""
  g0, g1, g2 = gates
  dg0, dg1, dg2 = dg
  dot = Stream.sum(g0 * dg0, g1 * dg1, g2 * dg2)
  return (g0 * (dg0 - dot), g1 * (dg1 - dot), g2 * (dg2 - dot))


@psyre_node('gating_network')
def gating_network(
    x: Stream, gate: GateParams[Stream]
) -> tuple[Stream, Stream, Stream]:
  """Top-2 sparse softmax gating network over scalar input stream `x`."""
  return _sparse_top2_softmax(*_triple(lambda k: gate.w[k] * x + gate.b[k]))


@psyre_node('grad_gating_network')
def grad_gating_network(
    x: Stream,
    gate: GateParams[Stream],
    gates: tuple[Stream, Stream, Stream],
    dg: tuple[Stream, Stream, Stream],
) -> tuple[Stream, GateParams[Stream]]:
  """Reverse-mode backward pass of `gating_network`."""
  ds = _grad_sparse_top2_softmax(gates, dg)
  dx_gate = Stream.sum(*(ds[k] * gate.w[k] for k in range(NUM_EXPERTS)))
  dgate = GateParams(w=_triple(lambda k: ds[k] * x), b=ds)
  return dx_gate, dgate


@psyre_node('scalar_expert')
def scalar_expert(x_sub: Stream, params: ScalarExpertParams[Stream]) -> Stream:
  """Quadratic/affine expert `w1 * x + w2 * x^2 + b` on subclock `. on c_k`."""
  return Stream.sum(params.w1 * x_sub, params.w2 * (x_sub * x_sub), params.b)


@psyre_node('grad_scalar_expert')
def grad_scalar_expert(
    x_sub: Stream,
    params: ScalarExpertParams[Stream],
    do_sub: Stream,
) -> tuple[Stream, ScalarExpertParams[Stream]]:
  """Reverse-mode backward pass of `scalar_expert` on subclock `. on c_k`."""
  dx_sub = (params.w1 + 2.0 * params.w2 * x_sub) * do_sub
  dparams = ScalarExpertParams(
      w1=x_sub * do_sub,
      w2=(x_sub * x_sub) * do_sub,
      b=do_sub,
  )
  return dx_sub, dparams


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
  gates = gating_network(x, params.gate)
  conds = _dispatch_conds(gates)

  def run_expert(k: int) -> tuple[Stream, ScalarExpertParams[Stream], Stream]:
    c_k = conds[k]
    x_k = when(c_k, x, name=f'x_c{k + 1}')
    p_k = params.experts[k].map(lambda p: when(c_k, p))
    o_k = scalar_expert(x_k, p_k, name=f'expert{k + 1}')
    o_k.name = f'o{k + 1}'
    return x_k, p_k, o_k

  exp_triples = _triple(run_expert)
  x_subs = _triple(lambda k: exp_triples[k][0])
  p_subs = _triple(lambda k: exp_triples[k][1])
  o_subs = _triple(lambda k: exp_triples[k][2])
  o, merged_os = _combine_experts(gates, conds, o_subs)
  fwd = GmoeForwardOutput(o=o, gates=gates, conds=conds, expert_outputs=o_subs)
  return fwd, x_subs, p_subs, merged_os


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
  """Forward and reverse-mode AD of sparsely-gated MoE with scalar experts.

  Each expert's forward and backward passes execute strictly on its active
  subclock `. on c_k` (`c_k = (g_k != 0.0)`). The reverse adjoint of
  `merge(c_k, o_k, 0.0)` is `when(c_k, do * g_k)` and the reverse adjoint of
  `when(c_k, ...)` is `merge(c_k, d_sub, 0.0)`, guaranteeing zero parameter
  gradients for unselected experts without executing them.

  Args:
    x: Base-clock scalar input stream.
    y_true: Base-clock target stream or scalar.
    params: Base-clock gate and scalar expert parameter streams.

  Returns:
    `GmoeGradOutput` containing prediction, loss, input gradient, parameter
    gradients, gating weights, dispatch conditions, and subclock expert outputs.
  """
  require_sync_semantics('grad_gmoe')
  fwd, x_subs, p_subs, merged_os = _scalar_forward(x, params)
  loss_val, do = rnn.grad_loss(fwd.o, y_true, 1.0)
  loss_val.name = 'loss'

  dg = _triple(lambda k: do * merged_os[k])
  dx_gate, dgate = grad_gating_network(x, params.gate, fwd.gates, dg)

  def run_backward(k: int) -> tuple[Stream, ScalarExpertParams[Stream]]:
    do_k = when(fwd.conds[k], do * fwd.gates[k], name=f'do{k + 1}')
    return grad_scalar_expert(
        x_subs[k], p_subs[k], do_k, name=f'grad_expert{k + 1}'
    )

  bwd = _triple(run_backward)
  dx, dexperts = _lift_expert_grads(
      fwd.conds,
      dx_gate,
      _triple(lambda k: bwd[k][0]),
      _triple(lambda k: bwd[k][1]),
  )
  return GmoeGradOutput(
      o=fwd.o,
      loss=loss_val,
      dx=dx,
      grads=GmoeParams(gate=dgate, experts=dexperts),
      gates=fwd.gates,
      conds=fwd.conds,
      expert_outputs=fwd.expert_outputs,
  )


def extract_scalar_params(
    traces: ScalarGmoeParamTraces, step: int = -1
) -> ScalarGmoeParamValues:
  """Extracts scalar GMoE parameter values at `step` from recorded traces."""
  return traces.map(lambda seq: float(seq[step]))


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

  step = counter()
  idx = step % len(dataset)
  x = switch(
      idx, {i: float(item[0]) for i, item in enumerate(dataset)}, default=0.0
  )
  y = switch(
      idx, {i: float(item[1]) for i, item in enumerate(dataset)}, default=0.0
  )
  x.name, y.name = 'x', 'y'
  active_step = step < max_steps

  next_params: ScalarGmoeParamStreams
  prev_params = _delay_tree(
      initial_params,
      lambda: next_params,
      lambda init_val, src: fby(float(init_val), src),
  )
  grad_out = grad_gmoe(x, y, prev_params)
  updated = prev_params.map(lambda p, g: p - lr * g, grad_out.grads)
  next_params = updated.map(
      lambda upd, old: ifthenelse(active_step, upd, old), prev_params
  )
  return GmoeTrainingOutput(
      params=next_params,
      loss=when(active_step, grad_out.loss, name='train_loss'),
      gates=grad_out.gates,
      conds=grad_out.conds,
      expert_outputs=grad_out.expert_outputs,
  )


# ---------------------------------------------------------------------------
# 3. Chained GMoE with RNN + Perceptron Experts (variable unroll depths)
# ---------------------------------------------------------------------------


def _validate_expert_cells(
    expert_cells: Sequence[int], dimension: int
) -> tuple[int, int, int]:
  """Validates `expert_cells` has 3 entries in `[1, dimension]`."""
  if len(expert_cells) != NUM_EXPERTS:
    raise ValueError(
        f'expert_cells must have {NUM_EXPERTS} entries, got {expert_cells!r}.'
    )
  for k, cells in enumerate(expert_cells):
    if cells < 1 or cells > dimension:
      raise ValueError(
          f'expert_cells[{k}]={cells} must be in [1, {dimension}].'
      )
  return (expert_cells[0], expert_cells[1], expert_cells[2])


def _normalize_gate_vector(
    raw_value: Any, dimension: int, label: str
) -> list[float]:
  """Normalizes a gate weight vector to length `dimension`."""
  if isinstance(raw_value, (int, float)):
    return [float(raw_value)] * dimension
  vals = [float(v) for v in raw_value]
  if len(vals) == 1:
    return vals * dimension
  if len(vals) != dimension:
    raise ValueError(
        f'Gate parameter {label!r} has length {len(vals)}, expected 1 or '
        f'{dimension}.'
    )
  return vals


def normalize_chained_params(
    params: ChainedGmoeParamValues,
    *,
    dimension: int = rnn.DIMENSION,
) -> ChainedGmoeParamValues:
  """Validates and normalizes chained GMoE-RNN parameters."""
  gate = GateParams(
      w=_triple(
          lambda k: _normalize_gate_vector(
              params.gate.w[k], dimension, f'gate.w[{k}]'
          )
      ),
      b=_triple(lambda k: float(params.gate.b[k])),
  )
  experts = _triple(lambda k: rnn.normalize_params(params.experts[k]))
  return GmoeParams(gate=gate, experts=experts)


def chained_param_streams(
    in_pass: Stream,
    params: ChainedGmoeParamValues,
    *,
    dimension: int = rnn.DIMENSION,
) -> ChainedGmoeParamStreams:
  """Converts `ChainedGmoeParamValues` into pass-indexed parameter streams."""
  norm = normalize_chained_params(params, dimension=dimension)
  gate = GateParams(
      w=_triple(lambda k: _vector_stream(in_pass, norm.gate.w[k])),
      b=_triple(lambda k: Stream(norm.gate.b[k])),
  )
  experts = _triple(
      lambda k: rnn.ModelParams(
          rnn=norm.experts[k].rnn.map(Stream),
          per=norm.experts[k].per.map(
              lambda vec: _vector_stream(in_pass, vec)
          ),
      )
  )
  return GmoeParams(gate=gate, experts=experts)


def extract_chained_params(
    traces: ChainedGmoeParamTraces,
    *,
    dimension: int = rnn.DIMENSION,
) -> ChainedGmoeParamValues:
  """Extracts the trailing full pass of chained GMoE-RNN parameters."""

  def extract_vec(seq: Sequence[Any], label: str) -> list[float]:
    vals = [float(v) for v in present_values(seq)]
    if len(vals) < dimension:
      raise ValueError(
          f'Trace {label!r} has {len(vals)} present values, expected at least '
          f'{dimension}.'
      )
    return vals[-dimension:]

  def extract_scalar(seq: Sequence[Any], label: str) -> float:
    win = extract_vec(seq, label)
    if any(v != win[0] for v in win):
      raise ValueError(
          f'Shared parameter {label!r} is not constant across pass: {win!r}.'
      )
    return win[-1]

  gate = GateParams(
      w=_triple(lambda k: extract_vec(traces.gate.w[k], f'gate.w[{k}]')),
      b=_triple(lambda k: extract_scalar(traces.gate.b[k], f'gate.b[{k}]')),
  )
  experts = _triple(lambda k: rnn.extract_params(traces.experts[k]))
  return GmoeParams(gate=gate, experts=experts)


@psyre_node('pass_gating_network')
def pass_gating_network(
    x: Stream,
    gate: GateParams[Stream],
    *,
    dimension: int = rnn.DIMENSION,
    pass_end: Stream | None = None,
) -> tuple[Stream, Stream, Stream]:
  """Pass-constant top-2 sparse softmax gating over sequence `x[0..D-1]`."""
  scores = _triple(
      lambda k: (
          rnn.sum_stream(x * gate.w[k], dimension=dimension, end=pass_end)
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
    dimension: int = rnn.DIMENSION,
    pass_end: Stream | None = None,
) -> tuple[Stream, GateParams[Stream]]:
  """Reverse-mode backward pass of `pass_gating_network`."""
  ds = _grad_sparse_top2_softmax(gates, dg)
  ds_total = _triple(
      lambda k: rnn.sum_stream(ds[k], dimension=dimension, end=pass_end)
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
    dimension: int = rnn.DIMENSION,
    in_pass: Stream | None = None,
    pass_reset: Stream | None = None,
) -> GmoeForwardOutput:
  """Forward pass chaining sparse MoE with 3 RNN+perceptron experts.

  Each expert runs `rnn.forward` with its own unrolling depth
  `expert_cells[k]` (default `(3, 4, 5)`) strictly on subclock `. on c_k`
  under its own downsampled pass reset `when(c_k, pass_reset)`.

  Args:
    x: Base-clock sequence input stream (one pass = `dimension` ticks).
    params: Base-clock gate and 3 RNN+perceptron expert parameter streams.
    expert_cells: Sequence of 3 RNN unrolling depths, each in `[1, dimension]`.
    dimension: Number of time steps per pass.
    in_pass: Optional pass coordinate stream in `0..dimension-1`.
    pass_reset: Optional boolean pass-start reset stream.

  Returns:
    `GmoeForwardOutput` containing prediction, gating weights, dispatch
    conditions, and subclock expert outputs.
  """
  require_sync_semantics('chained_gmoe_forward')
  cells = _validate_expert_cells(expert_cells, dimension)
  pos, rst, pass_final = _pass_coords(dimension, in_pass, pass_reset)

  with reset(rst):
    gates = pass_gating_network(
        x, params.gate, dimension=dimension, pass_end=pass_final
    )
  conds = _dispatch_conds(gates)

  def run_expert(k: int) -> Stream:
    c_k = conds[k]
    x_k = when(c_k, x, name=f'x_e{k + 1}')
    pos_k = when(c_k, pos, name=f'pos_e{k + 1}')
    rst_k = when(c_k, rst, name=f'rst_e{k + 1}')
    p_k = params.experts[k].map(lambda s: when(c_k, s))
    with push_prefix(f'expert{k + 1}'):
      with reset(rst_k):
        o_k = rnn.forward(x_k, p_k, num_cells=cells[k], in_pass=pos_k)
    o_k.name = f'o{k + 1}'
    return o_k

  o_subs = _triple(run_expert)
  o, _ = _combine_experts(gates, conds, o_subs)
  return GmoeForwardOutput(o=o, gates=gates, conds=conds, expert_outputs=o_subs)


def grad_chained_gmoe(
    x: Stream,
    y_true: Stream | float,
    params: ChainedGmoeParamStreams,
    *,
    expert_cells: Sequence[int] = DEFAULT_EXPERT_CELLS,
    dimension: int = rnn.DIMENSION,
    in_pass: Stream | None = None,
    pass_reset: Stream | None = None,
) -> GmoeGradOutput[rnn.ParamStreams]:
  """Forward + reverse-mode AD of GMoE chaining 3 RNN+perceptron experts.

  Carefully preserves both clock soundness and deadlock-free bidirectional
  reactivity across dynamically skipped passes:
  - Pass-constant sparse gating (`c_k = (g_k != 0.0)`) ensures every active
    expert executes all `dimension` ticks of a pass while inactive experts
    remain completely `Absent` across all ticks of the pass.
  - Each expert's forward and backward `rnn.grad_rnn_perceptron` runs strictly
    on `. on c_k` under its downsampled pass reset `when(c_k, pass_reset)`.
  - Lookahead `post` inside `grad_rnn_layer` / `grad_per_layer` is lifted to
    the enclosing base clock via `merge(c_k, u_sub, 0.0)` and re-sampled back
    onto `. on c_k`, preventing `post` from skipping across inactive passes and
    deadlocking multi-pass training loops.

  Args:
    x: Base-clock sequence input stream (one pass = `dimension` ticks).
    y_true: Base-clock target stream or scalar evaluated at `pass_final`.
    params: Base-clock gate and 3 RNN+perceptron expert parameter streams.
    expert_cells: Sequence of 3 RNN unrolling depths, each in `[1, dimension]`.
    dimension: Number of time steps per pass.
    in_pass: Optional pass coordinate stream in `0..dimension-1`.
    pass_reset: Optional boolean pass-start reset stream.

  Returns:
    `GmoeGradOutput` containing prediction, loss, input gradient, parameter
    gradients, gating weights, dispatch conditions, and subclock expert outputs.
  """
  require_sync_semantics('grad_chained_gmoe')
  cells = _validate_expert_cells(expert_cells, dimension)
  pos, rst, pass_final = _pass_coords(dimension, in_pass, pass_reset)
  base_env = CapturedEnv.capture()

  with reset(rst):
    gates = pass_gating_network(
        x, params.gate, dimension=dimension, pass_end=pass_final
    )
  conds = _dispatch_conds(gates)

  def make_shift_back(cond: Stream) -> Callable[[Stream], Stream]:
    def shift_back(u_sub: Stream) -> Stream:
      cur_pfxs = prefix_stack.get()
      with CapturedEnv(
          arms=base_env.arms,
          prefixes=cur_pfxs,
          resets=(*base_env.resets, rst),
      ).restore():
        return when(cond, post(merge(cond, u_sub, 0.0)))

    return shift_back

  do_subs: tuple[Stream, Stream, Stream]

  def run_expert(k: int) -> tuple[Stream, Stream, rnn.ParamStreams]:
    c_k = conds[k]
    x_k = when(c_k, x, name=f'x_e{k + 1}')
    pos_k = when(c_k, pos, name=f'pos_e{k + 1}')
    rst_k = when(c_k, rst, name=f'rst_e{k + 1}')
    p_k = params.experts[k].map(lambda s: when(c_k, s))
    with push_prefix(f'expert{k + 1}'):
      with reset(rst_k):
        o_k, dx_k, dp_k = rnn.grad_rnn_perceptron(
            x_k,
            p_k,
            lambda: do_subs[k],
            num_cells=cells[k],
            dimension=dimension,
            in_pass=pos_k,
            shift_back=make_shift_back(c_k),
        )
    o_k.name = f'o{k + 1}'
    return o_k, dx_k, dp_k

  exp_results = _triple(run_expert)
  o_subs = _triple(lambda k: exp_results[k][0])
  dx_subs = _triple(lambda k: exp_results[k][1])
  dp_subs = _triple(lambda k: exp_results[k][2])
  o, merged_os = _combine_experts(gates, conds, o_subs)

  loss_val, do = rnn.grad_loss(o, y_true, ifthenelse(pass_final, 1.0, 0.0))
  loss_val.name = 'loss'

  dg = _triple(lambda k: do * merged_os[k])
  with reset(rst):
    dx_gate, dgate = grad_pass_gating_network(
        x, params.gate, gates, dg, dimension=dimension, pass_end=pass_final
    )
  do_subs = _triple(
      lambda k: when(conds[k], do * gates[k], name=f'do{k + 1}')
  )
  dx, dexperts = _lift_expert_grads(conds, dx_gate, dx_subs, dp_subs)
  return GmoeGradOutput(
      o=o,
      loss=loss_val,
      dx=dx,
      grads=GmoeParams(gate=dgate, experts=dexperts),
      gates=gates,
      conds=conds,
      expert_outputs=o_subs,
  )


def apply_chained_gmoe_grads(
    params: ChainedGmoeParamStreams,
    grads: ChainedGmoeParamStreams,
    lr: float,
    *,
    dimension: int = rnn.DIMENSION,
    pass_end: Stream | None = None,
) -> ChainedGmoeParamStreams:
  """Applies one gradient descent step to gate and 3 RNN+perceptron experts."""
  return GmoeParams(
      gate=params.gate.map(lambda p, g: p - lr * g, grads.gate),
      experts=_triple(
          lambda k: rnn.apply_grads(
              params.experts[k],
              grads.experts[k],
              lr,
              dimension=dimension,
              end=pass_end,
          )
      ),
  )


def _select_sequence_sample(
    index_stream: Stream,
    samples: Sequence[rnn.LabeledStream],
) -> rnn.LabeledStream:
  """Multiplexes sequence LabeledStreams by `index_stream`."""
  return rnn.LabeledStream(
      x=switch(
          index_stream,
          {i: sample.x for i, sample in enumerate(samples)},
          default=0.0,
      ),
      y=switch(
          index_stream,
          {i: sample.y for i, sample in enumerate(samples)},
          default=0.0,
      ),
  )


def eval_chained_gmoe(
    dataset: Sequence[tuple[Sequence[float], float]],
    params: ChainedGmoeParamValues,
    *,
    expert_cells: Sequence[int] = DEFAULT_EXPERT_CELLS,
    dimension: int = rnn.DIMENSION,
) -> Stream:
  """Evaluates squared loss of chained GMoE-RNN across every sample in dataset."""
  require_sync_semantics('eval_chained_gmoe')
  if not dataset:
    raise ValueError('dataset must be non-empty.')
  tick = counter()
  eval_idx = tick // dimension
  in_pass, pass_reset, pass_final = _pass_coords(dimension, tick % dimension)
  samples = rnn.dataset_streams(dataset, in_pass=in_pass)
  sample = _select_sequence_sample(eval_idx, samples)
  p_streams = chained_param_streams(in_pass, params, dimension=dimension)
  fwd = chained_gmoe_forward(
      sample.x,
      p_streams,
      expert_cells=expert_cells,
      dimension=dimension,
      in_pass=in_pass,
      pass_reset=pass_reset,
  )
  emit = pass_final & (eval_idx < len(samples))
  return when(emit, rnn.loss(fwd.o, sample.y), name='eval_loss')


def train_chained_gmoe(
    dataset: Sequence[tuple[Sequence[float], float]],
    initial_params: ChainedGmoeParamValues,
    *,
    expert_cells: Sequence[int] = DEFAULT_EXPERT_CELLS,
    max_steps: int = DEFAULT_CHAINED_MAX_STEPS,
    lr: float = DEFAULT_CHAINED_LR,
    dimension: int = rnn.DIMENSION,
) -> GmoeTrainingOutput[rnn.ParamStreams]:
  """Multi-pass BPTT training automaton for the chained GMoE-RNN model."""
  require_sync_semantics('train_chained_gmoe')
  _validate_training_args(dataset, max_steps)
  _validate_expert_cells(expert_cells, dimension)
  norm_init = normalize_chained_params(initial_params, dimension=dimension)

  with automaton(
      init=GmoeTrainingState.INIT_LOAD, name='gmoe_train'
  ) as train:
    prev = _delay_tree(
        norm_init,
        lambda: train.params,
        lambda _, src: last(src, init=0.0, n=dimension),
    )
    with state(GmoeTrainingState.INIT_LOAD):
      init_pos = counter()
      until(init_pos == (dimension - 1), GmoeTrainingState.GRAD_DESCENT)
      train.params = chained_param_streams(
          init_pos, norm_init, dimension=dimension
      )
      train.loss = 0.0
      train.emit_loss = False
      train.gates = (0.0, 0.0, 0.0)
      train.conds = (False, False, False)
      train.merged_outputs = (0.0, 0.0, 0.0)
    with state(GmoeTrainingState.GRAD_DESCENT):
      tick = counter()
      step = tick // dimension
      in_pass, pass_reset, pass_final = _pass_coords(
          dimension, tick % dimension
      )
      samples = rnn.dataset_streams(dataset, in_pass=in_pass)
      sample = _select_sequence_sample(step % len(samples), samples)
      active_step = step < max_steps

      grad_out = grad_chained_gmoe(
          sample.x,
          sample.y,
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
      train.params = updated.map(
          lambda upd, old: ifthenelse(active_step, upd, old), prev
      )
      train.loss = grad_out.loss
      train.emit_loss = pass_final & active_step
      train.gates = grad_out.gates
      train.conds = grad_out.conds
      train.merged_outputs = _triple(
          lambda k: merge(grad_out.conds[k], grad_out.expert_outputs[k], 0.0)
      )

  return GmoeTrainingOutput(
      params=train.params,
      loss=when(train.emit_loss, train.loss, name='train_loss'),
      gates=train.gates,
      conds=train.conds,
      expert_outputs=_triple(
          lambda k: when(
              train.conds[k], train.merged_outputs[k], name=f'o{k + 1}'
          )
      ),
  )
