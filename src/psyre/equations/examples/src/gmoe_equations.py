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
"""Sparsely-gated MoE with AD via `psyre_equations` (experimental).

Note: Everything under `psyre/equations/` is experimental; APIs and internals
may change without notice. Replicates `examples/src/gmoe.py` without explicit
`lambda:` thunks in the forward-referencing training and chained-AD loops,
while reusing thunk-free forward/gating helpers from `examples/src/gmoe.py`
(see Shazeer et al., ICLR '17, arXiv:1701.06538; Potop-Butucaru, Cohen,
Plotkin & Pompougnac, 2023, arXiv:2311.16977).
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
    merge,
    push_prefix,
    require_sync_semantics,
    reset,
    when,
)
from psyre.equations import psyre_equations
from psyre.equations.examples.src import rnn_equations
from psyre.match import last
from psyre.static_stream import repeat
from psyre.stream import Stream

try:
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
      apply_chained_gmoe_grads,
      chained_gmoe_forward,
      chained_param_streams,
      combine_experts,
      demo_expert1,
      demo_expert2,
      demo_expert3,
      dispatch_conds,
      eval_chained_gmoe,
      extract_chained_params,
      extract_scalar_params,
      fixed_gating_network,
      gating_network,
      gmoe_forward,
      grad_gating_network,
      grad_gmoe,
      grad_pass_gating_network,
      grad_scalar_expert,
      lift_expert_grads,
      make_subclock_shift_back,
      mixture_of_experts_demo,
      normalize_chained_params,
      pass_coords,
      pass_gating_network,
      periodic_chained_param_streams,
      scalar_expert,
      triple,
      validate_expert_cells,
      validate_training_args,
  )
  from examples.src.rnn import (
      dataset_words,
      delay_tree,
      pass_end_word,
      pass_reset_word,
      sample_params,
      select_params,
      to_param_float,
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


def train_gmoe(
    dataset: Sequence[tuple[float, float]],
    initial_params: ScalarGmoeParamValues,
    *,
    max_steps: int = DEFAULT_SCALAR_MAX_STEPS,
    lr: float = DEFAULT_SCALAR_LR,
) -> GmoeTrainingOutput[ScalarExpertParams[Stream]]:
  """Trains the sparsely-gated scalar MoE with dynamic dispatch and SGD."""
  require_sync_semantics('train_gmoe')
  validate_training_args(dataset, max_steps)

  with psyre_equations() as eq:

    class _:
      """Scalar GMoE SGD training loop equations."""
      x = Stream.periodic([], [float(item[0]) for item in dataset])
      y = Stream.periodic([], [float(item[1]) for item in dataset])
      active_step = Stream.periodic(repeat(True, max_steps), [False])

      prev_params = delay_tree(
          initial_params,
          # pylint: disable-next=undefined-variable,used-before-assignment
          next_params,  # pyrefly: ignore[unknown-name]
          lambda init_val, src: fby(
              to_param_float(init_val, 'initial_params'), src
          ),
      )
      grad_out = grad_gmoe(x, y, prev_params)
      updated = prev_params.map(
          lambda p, g: p - lr * g, grad_out.grads
      )
      next_params = select_params(active_step, updated, prev_params)
      train_loss = when(active_step, grad_out.loss)
      __output__ = GmoeTrainingOutput(
          params=next_params,
          loss=train_loss,
          gates=grad_out.gates,
          conds=grad_out.conds,
          expert_outputs=grad_out.expert_outputs,
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
  cells = validate_expert_cells(expert_cells, dimension)

  with psyre_equations() as eq:

    class _:
      """Chained GMoE-RNN forward and reverse-mode AD equations."""
      pos, rst, pass_final = pass_coords(dimension, in_pass, pass_reset)
      make_shift_back = make_subclock_shift_back(rst)
      with reset(rst):
        gates = pass_gating_network(
            x, params.gate, dimension=dimension, pass_end=pass_final
        )
      conds = dispatch_conds(gates)

      _o_subs_list = []
      _dx_subs_list = []
      _dp_subs_list = []
      for _k in range(NUM_EXPERTS):
        _c_k = conds[_k]
        _x_k = when(_c_k, x, name=f'x_e{_k + 1}')
        _pos_k = when(_c_k, pos, name=f'pos_e{_k + 1}')
        _rst_k = when(_c_k, rst, name=f'rst_e{_k + 1}')
        _p_k = sample_params(_c_k, params.experts[_k])
        with push_prefix(f'expert{_k + 1}'):
          with reset(_rst_k):
            _o_k, _dx_k, _dp_k = rnn_equations.grad_rnn_perceptron(
                _x_k,
                _p_k,
                # pylint: disable-next=undefined-variable,used-before-assignment
                do_subs[_k],  # pyrefly: ignore[unknown-name]
                num_cells=cells[_k],
                dimension=dimension,
                in_pass=_pos_k,
                shift_back=make_shift_back(_c_k),
            )
        _o_k.name = f'o{_k + 1}'
        _o_subs_list.append(_o_k)
        _dx_subs_list.append(_dx_k)
        _dp_subs_list.append(_dp_k)

      o_subs = tuple(_o_subs_list)
      dx_subs = tuple(_dx_subs_list)
      dp_subs = tuple(_dp_subs_list)
      o, merged_os = combine_experts(gates, conds, o_subs)

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
      dx, dexperts = lift_expert_grads(conds, dx_gate, dx_subs, dp_subs)
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
  validate_training_args(dataset, max_steps)
  validate_expert_cells(expert_cells, dimension)
  norm_init = normalize_chained_params(initial_params, dimension=dimension)
  x_word, y_word = dataset_words(dataset, dimension=dimension)

  with psyre_equations() as eq:

    class _:
      """Chained GMoE-RNN multi-pass BPTT training automaton equations."""
      with automaton(
          init=GmoeTrainingState.INIT_LOAD, name='gmoe_train'
      ) as train:
        prev = delay_tree(
            norm_init,
            train.params,
            lambda _, src: last(src, init=0.0, n=dimension),
        )
        with state(GmoeTrainingState.INIT_LOAD):
          until(Stream.once(dimension - 1), GmoeTrainingState.GRAD_DESCENT)
          train.params = periodic_chained_param_streams(
              norm_init, dimension=dimension
          )
          train.loss = 0.0
          train.emit_loss = False
          train.gates = (0.0, 0.0, 0.0)
          train.conds = (False, False, False)
          train.merged_outputs = (0.0, 0.0, 0.0)
        with state(GmoeTrainingState.GRAD_DESCENT):
          in_pass = Stream.periodic([], list(range(dimension)))
          pass_reset = Stream.periodic([], pass_reset_word(dimension))
          pass_final = Stream.periodic([], pass_end_word(dimension))
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
          train.params = select_params(active_step, updated, prev)
          train.loss = grad_out.loss
          train.emit_loss = Stream.periodic(
              repeat(pass_end_word(dimension), max_steps), [False]
          )
          train.gates = grad_out.gates
          train.conds = grad_out.conds
          train.merged_outputs = triple(
              lambda k, go=grad_out: merge(
                  go.conds[k], go.expert_outputs[k], 0.0
              )
          )

      train_loss = when(train.emit_loss, train.loss)
      __output__ = GmoeTrainingOutput(
          params=train.params,
          loss=train_loss,
          gates=train.gates,
          conds=train.conds,
          expert_outputs=triple(
              lambda k, tr=train: when(
                  tr.conds[k], tr.merged_outputs[k], name=f'o{k + 1}'
              )
          ),
      )

  return eq.output
