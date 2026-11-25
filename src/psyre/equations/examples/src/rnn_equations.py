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
"""Psyre RNN, perceptron, and loss model via `psyre_equations` (experimental).

Note: Everything under `psyre/equations/` is experimental; APIs and internals
may change without notice. Replicates `examples/src/rnn.py` without explicit
`lambda:` thunks in the forward-referencing BPTT and training automata, while
reusing thunk-free layers and helpers from `examples/src/rnn.py` (see
Potop-Butucaru, Cohen, Plotkin & Pompougnac, 2023, arXiv:2311.16977; Colaço,
Pagano & Pouzet, EMSOFT '05).
"""

from collections.abc import Callable, Sequence
from typing import Any

# pylint: disable=g-import-not-at-top,g-importing-member,g-multiple-import
from psyre.automaton import (
    automaton,
    state,
    until,
)
from psyre.common import (
    counter,
    ifthenelse,
    post,
    reset,
)
from psyre.equations import psyre_equations
from psyre.static_stream import repeat
from psyre.stream import Stream

try:
  from examples.src import rnn as _rnn
  from examples.src.rnn import (
      DEFAULT_LEARNING_RATE,
      DEFAULT_MAX_STEPS,
      DIMENSION,
      HierarchicalOutput,
      HyperParams,
      InferenceState,
      LabeledStream,
      Mode,
      ModelParams,
      ParamStreams,
      ParamTraces,
      ParamValues,
      PerceptronParams,
      RnnParams,
      SimpleModelOutput,
      TrainingOutput,
      TrainingState,
      apply_grads,
      backfill,
      biased_dot_product,
      dataset_streams,
      dataset_words,
      delay_pass,
      delay_tree,
      extract_params,
      forward,
      grad_loss,
      grad_per_layer,
      grad_rnn_layer,
      inference_machine,
      leave_one_out_samples,
      loss,
      normalize_params,
      pass_end_word,
      pass_final,
      pass_reset_word,
      per_layer,
      periodic_param_streams,
      random_param_streams,
      relu,
      rnn_cell,
      rnn_layer,
      select_params,
      sum_stream,
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
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_MAX_STEPS',
    'DIMENSION',
    'HierarchicalOutput',
    'HyperParams',
    'InferenceState',
    'LabeledStream',
    'Mode',
    'ModelParams',
    'ParamStreams',
    'ParamTraces',
    'ParamValues',
    'PerceptronParams',
    'RnnParams',
    'SimpleModelOutput',
    'TrainingOutput',
    'TrainingState',
    'apply_grads',
    'backfill',
    'biased_dot_product',
    'dataset_streams',
    'extract_params',
    'forward',
    'grad_loss',
    'grad_model',
    'grad_per_layer',
    'grad_rnn_layer',
    'grad_rnn_perceptron',
    'hierarchical_model',
    'inference_machine',
    'loss',
    'normalize_params',
    'per_layer',
    'relu',
    'rnn_cell',
    'rnn_layer',
    'simple_model',
    'sum_stream',
    'training_machine',
)

_PARAM_TEMPLATE = ModelParams.build(lambda: 0.0)


def grad_rnn_perceptron(
    x,
    params: ParamStreams,
    dy,
    *,
    num_cells: int = DIMENSION,
    dimension: int = DIMENSION,
    in_pass: Stream | None = None,
    shift_back: Callable[[Stream], Stream] = post,
) -> tuple[Stream, Any, ParamStreams]:
  """Computes forward and reverse-mode backward pass of RNN + perceptron."""
  with psyre_equations() as eq:

    class _:
      """Coupled forward/backward RNN + perceptron equations."""
      pos = counter() if in_pass is None else in_pass
      final = pos == (dimension - 1)
      h, dx, drnn = grad_rnn_layer(
          x,
          params.rnn,
          # pylint: disable-next=undefined-variable,used-before-assignment
          dh,  # pyrefly: ignore[unknown-name]
          num_cells=num_cells,
          dimension=dimension,
          in_pass=pos,
          shift_back=shift_back,
      )
      y_pred, dh, dper = grad_per_layer(
          h,
          params.per,
          dy,
          dimension=dimension,
          final=final,
          shift_back=shift_back,
      )
      __output__ = (y_pred, dx, ModelParams(rnn=drnn, per=dper))

  return eq.output


def grad_model(
    x,
    y_true,
    params: ParamStreams,
    *,
    num_cells: int = DIMENSION,
    dimension: int = DIMENSION,
) -> tuple[Stream, Stream, ParamStreams]:
  """Computes predictions, loss, and parameter gradients over one pass."""
  with psyre_equations() as eq:

    class _:
      dloss = ifthenelse(pass_final(dimension), 1.0, 0.0)
      y_pred, _, grads = grad_rnn_perceptron(
          x,
          params,
          # pylint: disable-next=undefined-variable,used-before-assignment
          dy_pred,  # pyrefly: ignore[unknown-name]
          num_cells=num_cells,
          dimension=dimension,
      )
      loss_bwd, dy_pred = grad_loss(y_pred, y_true, dloss)
      __output__ = (y_pred, loss_bwd, grads)

  return eq.output


def simple_model(
    *,
    hyperparams: HyperParams = HyperParams(lr=0.1),
) -> SimpleModelOutput:
  """Runs the forward pass and one iteration of gradient descent."""
  return _rnn.simple_model(
      hyperparams=hyperparams,
      grad_model_fn=grad_model,
  )


def _training_streams(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
    *,
    init_state: TrainingState = TrainingState.INIT_RANDOM,
    hyperparams: HyperParams = HyperParams(),
) -> tuple[ParamStreams, Stream, Stream]:
  """Returns `(params, loss, emit_loss)` streams from the training automaton."""
  if init_state == TrainingState.GRAD_DESCENT:
    raise ValueError(
        'init_state must be TrainingState.INIT_RANDOM or '
        'TrainingState.INIT_LOAD.'
    )
  if init_state == TrainingState.INIT_LOAD and params is None:
    raise ValueError('params is required for TrainingState.INIT_LOAD.')
  train_dataset = leave_one_out_samples(dataset, hyperparams.holdout_index)
  x_word, y_word = dataset_words(train_dataset)

  with psyre_equations() as eq:

    class _:
      """Nested BPTT training automaton equations."""
      with automaton(init=init_state.name, name='train') as train:
        prev = delay_tree(
            _PARAM_TEMPLATE, train.params, lambda _, src: delay_pass(src)
        )
        with state(init_state.name):
          until(pass_final(), TrainingState.GRAD_DESCENT.name)
          if init_state == TrainingState.INIT_RANDOM:
            train.params = random_param_streams()
          else:
            assert params is not None
            train.params = periodic_param_streams(params)
          train.loss = 0.0
          train.emit_loss = False
        with state(TrainingState.GRAD_DESCENT.name):
          pass_reset = Stream.periodic([], pass_reset_word())
          sample_x = Stream.periodic([], x_word)
          sample_y = Stream.periodic([], y_word)
          active_step = Stream.periodic(
              repeat(True, hyperparams.max_steps * DIMENSION), [False]
          )
          with reset(pass_reset):
            _, train.loss, grads = grad_model(sample_x, sample_y, prev)
            updated = apply_grads(prev, grads, hyperparams.lr)
          train.emit_loss = Stream.periodic(
              repeat(pass_end_word(), hyperparams.max_steps), [False]
          )
          train.params = select_params(active_step, updated, prev)
      __output__ = (train.params, train.loss, train.emit_loss)

  return eq.output


def training_machine(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
    *,
    init_state: TrainingState = TrainingState.INIT_RANDOM,
    hyperparams: HyperParams = HyperParams(),
) -> TrainingOutput:
  """Nested automaton state machine for training mode (init and BPTT)."""
  return _rnn.training_machine(
      dataset,
      params,
      init_state=init_state,
      hyperparams=hyperparams,
      training_streams_fn=_training_streams,
  )


def hierarchical_model(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
    *,
    initial_mode: Mode = Mode.TRAINING,
    switch_mode: Stream | None = None,
    train_init_state: TrainingState = TrainingState.INIT_RANDOM,
    hyperparams: HyperParams = HyperParams(),
) -> HierarchicalOutput:
  """High-level automaton machine orchestrating training and inference modes."""
  return _rnn.hierarchical_model(
      dataset,
      params,
      initial_mode=initial_mode,
      switch_mode=switch_mode,
      train_init_state=train_init_state,
      hyperparams=hyperparams,
      training_streams_fn=_training_streams,
  )
