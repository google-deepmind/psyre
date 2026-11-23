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

Note: Everything under `psyre/equations/` is experimental and a work in
progress; APIs and internals may change without notice. Replicates
`examples/src/rnn.py` without explicit `lambda:` thunks, relying on
`with psyre_equations` and automatic stream naming (see Potop-Butucaru, Cohen,
Plotkin & Pompougnac, 2023, arXiv:2311.16977; Colaço, Pagano & Pouzet,
EMSOFT '05).
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
    counter,
    fby,
    ifthenelse,
    merge,
    morph,
    post,
    prefix_sum,
    psyre_node,
    reset,
    when,
    whennot,
)
from psyre.equations import psyre_equations
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
    _cell_mask,
    _dataset_words,
    _delay_pass,
    _leave_one_out_samples,
    _pass_end_word,
    _pass_final,
    _pass_reset_word,
    _periodic_param_streams,
    _random_param_streams,
    _zero_param_streams,
    biased_dot_product,
    dataset_streams,
    extract_params,
    grad_loss,
    loss,
    normalize_params,
    per_layer,
    relu,
)
from psyre.math import tanh
from psyre.static_stream import repeat
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import

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


def _select_params(
    cond: Stream, then_params: Any, else_params: Any
) -> Any:
  """Selects leaf-wise between two parameter trees under `cond`."""
  return then_params.map(
      lambda upd, old: ifthenelse(cond, upd, old), else_params
  )


def _sample_params(cond: Stream, params: Any) -> Any:
  """Downsamples every leaf stream of `params` onto subclock `cond`."""
  return params.map(lambda p: when(cond, p))


@psyre_node('backfill')
def backfill(
    stream: Stream,
    end: Stream,
    *,
    max_span: int,
    shift_back: Callable[[Stream], Stream] = post,
) -> Stream:
  """Broadcasts values at `end` ticks backward up to `max_span` steps."""
  with psyre_equations() as eq:

    class _:
      out = stream
      for _ in range(max_span):
        out = merge(end, when(end, stream), whennot(end, shift_back(out)))
      __output__ = out

  return eq.output


def sum_stream(
    x,
    *,
    dimension: int = DIMENSION,
    end: Stream | None = None,
    shift_back: Callable[[Stream], Stream] = post,
):
  """Broadcasts the sum of one pass of x as a constant stream."""
  if dimension < 1:
    raise ValueError(f'dimension must be >= 1, got {dimension}.')
  pass_end = _pass_final(dimension) if end is None else end
  return backfill(
      prefix_sum(x),
      pass_end,
      max_span=dimension - 1,
      shift_back=shift_back,
  )


def rnn_cell(x, rnn: RnnParams[Any], h) -> tuple[Stream, Stream]:
  """Computes one RNN cell step, returning (h_next, h_prev)."""
  with psyre_equations() as eq:

    class _:
      h_prev = Stream(0.0) if h is None else fby(0.0, h, name='rnn_fby')
      h_next = tanh(Stream.sum(rnn.wxh * x, rnn.whh * h_prev, rnn.bh))
      __output__ = (h_next, h_prev)

  return eq.output


@psyre_node('rnn_layer')
def rnn_layer(
    x,
    rnn: RnnParams[Any],
    *,
    num_cells: int = DIMENSION,
    in_pass: Stream | None = None,
) -> Stream:
  """Computes the forward pass of a 1-layer RNN unrolled to `num_cells`."""
  if num_cells < 1:
    raise ValueError(f'num_cells must be >= 1, got {num_cells}.')
  with psyre_equations() as eq:

    class _:
      """Forward equations for the unrolled RNN layer."""
      pos = counter() if in_pass is None else in_pass
      h: Any = None
      y: Any = 0.0
      for k in range(num_cells):
        h, _ = rnn_cell(x, rnn, h)
        now = _cell_mask(pos, k, num_cells)
        y = h if k == 0 and num_cells == 1 else ifthenelse(now, h, y)
      __output__ = y

  return eq.output


@psyre_node('grad_rnn_layer')
def grad_rnn_layer(
    x,
    rnn: RnnParams[Any],
    dy,
    *,
    num_cells: int = DIMENSION,
    dimension: int = DIMENSION,
    in_pass: Stream | None = None,
    shift_back: Callable[[Stream], Stream] = post,
) -> tuple[Stream, Any, RnnParams[Any]]:
  """Computes the forward and reverse-mode backward pass of the RNN layer."""
  if num_cells < 1:
    raise ValueError(f'num_cells must be >= 1, got {num_cells}.')
  if num_cells > dimension:
    raise ValueError(
        f'num_cells ({num_cells}) cannot exceed dimension ({dimension}).'
    )
  with psyre_equations() as eq:

    class _:
      """Forward and reverse-mode BPTT equations for the RNN layer."""
      pos = counter() if in_pass is None else in_pass
      final = pos == (dimension - 1)
      cells: list[tuple[Stream, Stream, Stream]] = []
      h: Any = None
      y: Any = 0.0
      for k in range(num_cells):
        h, h_prev = rnn_cell(x, rnn, h)
        now = _cell_mask(pos, k, num_cells)
        y = h if k == 0 and num_cells == 1 else ifthenelse(now, h, y)
        cells.append((now, h, h_prev))

      dh_next: Any = 0.0
      dz_terms: list[Stream] = []
      dwhh_terms: list[Stream] = []
      for idx, (now, h, h_prev) in enumerate(reversed(cells)):
        dh_y = ifthenelse(now, dy, 0.0)
        dh = dh_y if idx == 0 else dh_y + dh_next
        dz_k = dh * (1.0 - h**2)
        dz_terms.append(dz_k)
        dwhh_terms.append(dz_k * h_prev)
        if idx < num_cells - 1:
          dh_next = merge(
              final, 0.0, whennot(final, shift_back(dz_k * rnn.whh))
          )
      if len(dz_terms) == 1:
        dz, dwhh = dz_terms[0], dwhh_terms[0]
      elif len(dz_terms) == 2:
        dz = dz_terms[0] + dz_terms[1]
        dwhh = dwhh_terms[0] + dwhh_terms[1]
      else:
        dz = Stream.sum(*dz_terms)
        dwhh = Stream.sum(*dwhh_terms)
      __output__ = (y, dz * rnn.wxh, RnnParams(wxh=dz * x, whh=dwhh, bh=dz))

  return eq.output


@psyre_node('grad_per_layer')
def grad_per_layer(
    x,
    per: PerceptronParams[Any],
    dy,
    *,
    dimension: int = DIMENSION,
    final: Stream | None = None,
    shift_back: Callable[[Stream], Stream] = post,
) -> tuple[Stream, Any, PerceptronParams[Any]]:
  """Computes the forward and backward pass of the perceptron layer."""
  with psyre_equations() as eq:

    class _:
      z = biased_dot_product(x, per.wxh, per.bh)
      y = relu(z)
      dz = ifthenelse(z > 0.0, dy, 0.0)
      ps = prefix_sum(dz)
      end = _pass_final(dimension) if final is None else final
      total = backfill(ps, end, max_span=dimension - 1, shift_back=shift_back)
      dp = total - fby(0.0, ps)
      __output__ = (
          y,
          dp * per.wxh,
          PerceptronParams(wxh=dp * x, bh=morph(dp, 0.0)),
      )

  return eq.output


def forward(
    x,
    params: ParamStreams,
    *,
    num_cells: int = DIMENSION,
    in_pass: Stream | None = None,
) -> Stream:
  """Computes the 2-layer RNN + perceptron forward prediction stream."""
  with psyre_equations() as eq:

    class _:
      y_pred = per_layer(
          rnn_layer(x, params.rnn, num_cells=num_cells, in_pass=in_pass),
          params.per,
      )
      __output__ = y_pred

  return eq.output


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
      dloss = ifthenelse(_pass_final(dimension), 1.0, 0.0)
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


def apply_grads(
    params: ParamStreams,
    grads: ParamStreams,
    lr: float,
    *,
    dimension: int = DIMENSION,
    end: Stream | None = None,
) -> ParamStreams:
  """Applies one gradient descent step to shared RNN and perceptron params."""
  return ModelParams(
      rnn=params.rnn.map(
          lambda p, g: p - lr * sum_stream(g, dimension=dimension, end=end),
          grads.rnn,
      ),
      per=params.per.map(lambda p, g: p - lr * g, grads.per),
  )


def simple_model(
    *,
    hyperparams: HyperParams = HyperParams(lr=0.1),
) -> SimpleModelOutput:
  """Runs the forward pass and one iteration of gradient descent."""
  with psyre_equations() as eq:

    class _:
      x = Stream.periodic([], [1.0, 2.0, -1.0, 0.0, 1.0])
      y_true = 42.0
      params = _random_param_streams()

      y_pred_fwd = forward(x, params)
      loss_fwd = loss(y_pred_fwd, y_true)
      y_pred, loss_bwd, grads = grad_model(x, y_true, params)
      next_params = apply_grads(params, grads, hyperparams.lr)
      __output__ = SimpleModelOutput(
          y_pred_fwd=y_pred_fwd,
          loss_fwd=loss_fwd,
          y_pred=y_pred,
          loss_bwd=loss_bwd,
          params=next_params,
      )

  return eq.output


def _delay_params(params: Any) -> ParamStreams:
  """Delays all parameter streams in `params` by one pass via `last()`."""
  return ModelParams(
      rnn=RnnParams(
          wxh=_delay_pass(params.rnn.wxh),
          whh=_delay_pass(params.rnn.whh),
          bh=_delay_pass(params.rnn.bh),
      ),
      per=PerceptronParams(
          wxh=_delay_pass(params.per.wxh),
          bh=_delay_pass(params.per.bh),
      ),
  )


def _inference_streams(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
) -> tuple[Stream, Stream]:
  """Returns `(loss, emit_loss)` streams from the nested inference automaton."""
  if params is None:
    raise ValueError('params is required for inference_machine.')
  x_word, y_word = _dataset_words(dataset)

  with psyre_equations() as eq:

    class _:
      """Nested inference automaton equations."""
      with automaton(name='infer') as infer:
        with state(InferenceState.INIT):
          until(_pass_final(), InferenceState.EVAL)
          infer.loss = 0.0
          infer.emit_loss = False
        with state(InferenceState.EVAL):
          pass_reset = Stream.periodic([], _pass_reset_word())
          sample_x = Stream.periodic(x_word, [0.0])
          sample_y = Stream.periodic(y_word, [0.0])
          eval_params = _periodic_param_streams(params)
          with reset(pass_reset):
            infer.loss = loss(forward(sample_x, eval_params), sample_y)
          infer.emit_loss = Stream.periodic(
              repeat(_pass_end_word(), len(dataset)), [False]
          )
      __output__ = (infer.loss, infer.emit_loss)

  return eq.output


def inference_machine(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
) -> Stream:
  """Nested automaton state machine for inference mode (init and eval)."""
  eval_loss, emit_loss = _inference_streams(dataset, params)
  return when(emit_loss, eval_loss, name='eval_loss')


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
  train_dataset = _leave_one_out_samples(dataset, hyperparams.holdout_index)
  x_word, y_word = _dataset_words(train_dataset)

  with psyre_equations() as eq:

    class _:
      """Nested BPTT training automaton equations."""
      with automaton(init=init_state.name, name='train') as train:
        prev = _delay_params(train.params)
        with state(init_state.name):
          until(_pass_final(), TrainingState.GRAD_DESCENT.name)
          if init_state == TrainingState.INIT_RANDOM:
            train.params = _random_param_streams()
          else:
            assert params is not None
            train.params = _periodic_param_streams(params)
          train.loss = 0.0
          train.emit_loss = False
        with state(TrainingState.GRAD_DESCENT.name):
          pass_reset = Stream.periodic([], _pass_reset_word())
          sample_x = Stream.periodic([], x_word)
          sample_y = Stream.periodic([], y_word)
          active_step = Stream.periodic(
              repeat(True, hyperparams.max_steps * DIMENSION), [False]
          )
          with reset(pass_reset):
            _, train.loss, grads = grad_model(sample_x, sample_y, prev)
            updated = apply_grads(prev, grads, hyperparams.lr)
          train.emit_loss = Stream.periodic(
              repeat(_pass_end_word(), hyperparams.max_steps), [False]
          )
          train.params = _select_params(active_step, updated, prev)
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
  train_params, loss_bwd, emit_loss = _training_streams(
      dataset,
      params,
      init_state=init_state,
      hyperparams=hyperparams,
  )
  return TrainingOutput(
      params=train_params,
      train_loss=when(emit_loss, loss_bwd, name='train_loss'),
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
  if switch_mode is None:
    if initial_mode == Mode.TRAINING:
      train_out = training_machine(
          dataset=dataset,
          params=params,
          init_state=train_init_state,
          hyperparams=hyperparams,
      )
      return HierarchicalOutput(
          mode=Stream(Mode.TRAINING),
          params=train_out.params,
          train_loss=train_out.train_loss,
          eval_loss=None,
      )
    if params is None:
      raise ValueError('params is required when Mode.INFERENCE can be active.')
    return HierarchicalOutput(
        mode=Stream(Mode.INFERENCE),
        params=None,
        train_loss=None,
        eval_loss=inference_machine(dataset=dataset, params=params),
    )

  if params is None:
    raise ValueError('params is required when Mode.INFERENCE can be active.')

  with psyre_equations() as eq:

    class _:
      """Top-level mode-switching automaton equations."""
      with automaton(init=initial_mode, name='top') as top:
        with state(Mode.TRAINING) as mode:
          until(switch_mode, Mode.INFERENCE)
          top.mode = mode
          top.params, top.loss, top.emit_loss = _training_streams(
              dataset=dataset,
              params=params,
              init_state=train_init_state,
              hyperparams=hyperparams,
          )
        with state(Mode.INFERENCE) as mode:
          until(switch_mode, Mode.TRAINING)
          top.mode = mode
          top.params = _zero_param_streams()
          top.loss, top.emit_loss = _inference_streams(
              dataset=dataset,
              params=params,
          )

      is_training = top.mode == Mode.TRAINING
      train_loss = when(is_training & top.emit_loss, top.loss)
      eval_loss = when(~is_training & top.emit_loss, top.loss)
      __output__ = HierarchicalOutput(
          mode=top.mode,
          params=_sample_params(is_training, top.params),
          train_loss=train_loss,
          eval_loss=eval_loss,
      )

  return eq.output
