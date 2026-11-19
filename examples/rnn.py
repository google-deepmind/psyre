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
"""Psyre recurrent neural network, perceptron layer, and loss model."""

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
    counter,
    current,
    fby,
    ifthenelse,
    merge,
    morph,
    post,
    prefix_sum,
    psyre_node,
    require_sync_semantics,
    reset,
    switch,
    when,
    whennot,
)
from psyre.match import last
from psyre.math import tanh
from psyre.stream import Stream
from psyre.trace import present_values

# pylint: enable=g-importing-member,g-multiple-import

DIMENSION = 5
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MAX_STEPS = 3

_T = TypeVar('_T')
_U = TypeVar('_U')
_RnnT = TypeVar('_RnnT')
_PerT = TypeVar('_PerT')


class Mode(enum.Enum):
  """High-level mutually exclusive operating modes of the model machine."""

  INFERENCE = enum.auto()
  TRAINING = enum.auto()


class InferenceState(enum.Enum):
  """States of the nested inference-mode state machine."""

  INIT = enum.auto()
  EVAL = enum.auto()


class TrainingState(enum.Enum):
  """States of the nested training-mode state machine."""

  INIT_RANDOM = enum.auto()
  INIT_LOAD = enum.auto()
  GRAD_DESCENT = enum.auto()


class RnnParams(NamedTuple, Generic[_T]):
  """Parameters (wxh, whh, bh) of the 1-layer recurrent neural network."""

  wxh: _T
  whh: _T
  bh: _T

  @classmethod
  def build(cls, fn: Callable[[], _U]) -> 'RnnParams[_U]':
    """Constructs RnnParams by evaluating fn() for each field."""
    return cls(wxh=fn(), whh=fn(), bh=fn())

  def map(
      self, fn: Callable[..., _U], *others: 'RnnParams[Any]'
  ) -> 'RnnParams[_U]':
    """Applies fn field-wise across self and any parallel RnnParams."""
    return RnnParams(*(fn(*col) for col in zip(self, *others)))


class PerceptronParams(NamedTuple, Generic[_T]):
  """Parameters (wxh, bh) of the perceptron readout layer."""

  wxh: _T
  bh: _T

  @classmethod
  def build(cls, fn: Callable[[], _U]) -> 'PerceptronParams[_U]':
    """Constructs PerceptronParams by evaluating fn() per field."""
    return cls(wxh=fn(), bh=fn())

  def map(
      self, fn: Callable[..., _U], *others: 'PerceptronParams[Any]'
  ) -> 'PerceptronParams[_U]':
    """Applies fn field-wise across self and any parallel PerceptronParams."""
    return PerceptronParams(*(fn(*col) for col in zip(self, *others)))


class ModelParams(NamedTuple, Generic[_RnnT, _PerT]):
  """Composite parameters bundling the RNN and perceptron readout layers."""

  rnn: RnnParams[_RnnT]
  per: PerceptronParams[_PerT]

  @classmethod
  def build(
      cls,
      rnn_fn: Callable[[], _RnnT],
      per_fn: Callable[[], _PerT] | None = None,
  ) -> 'ModelParams[_RnnT, _PerT]':
    """Constructs ModelParams by evaluating rnn_fn() and per_fn() per field."""
    actual_per_fn: Any = rnn_fn if per_fn is None else per_fn
    return cls(
        rnn=RnnParams.build(rnn_fn),
        per=PerceptronParams.build(actual_per_fn),
    )

  def map(
      self, fn: Callable[..., Any], *others: 'ModelParams[Any, Any]'
  ) -> 'ModelParams[Any, Any]':
    """Applies fn leaf-wise across self and any parallel ModelParams."""
    return ModelParams(
        rnn=self.rnn.map(fn, *(other.rnn for other in others)),
        per=self.per.map(fn, *(other.per for other in others)),
    )


ParamValues = ModelParams[float, list[float]]
ParamStreams = ModelParams[Stream, Stream]
ParamTraces = ModelParams[list[Any], list[Any]]

_PARAM_LABELS = ModelParams(
    rnn=RnnParams(wxh='rnn.wxh', whh='rnn.whh', bh='rnn.bh'),
    per=PerceptronParams(wxh='per.wxh', bh='per.bh'),
)


class LabeledStream(NamedTuple):
  """Single dataset example represented as input and target Streams."""

  x: Stream
  y: Stream


class SimpleModelOutput(NamedTuple):
  """Output streams of `simple_model`."""

  y_pred_fwd: Stream
  loss_fwd: Stream
  y_pred: Stream
  loss_bwd: Stream
  params: ParamStreams


class TrainingOutput(NamedTuple):
  """Output streams of `training_machine`."""

  params: ParamStreams
  train_loss: Stream


class HierarchicalOutput(NamedTuple):
  """Output streams of `hierarchical_model`."""

  mode: Stream
  params: ParamStreams | None
  train_loss: Stream | None
  eval_loss: Stream | None


def _pass_final(dimension: int = DIMENSION) -> Stream:
  """Emits True on the final coordinate (dimension - 1) of a pass."""
  return Stream.once(dimension - 1)


def _pass_indices(
    dimension: int = DIMENSION,
) -> tuple[Stream, Stream, Stream, Stream]:
  """Returns (pass_index, in_pass, pass_reset, pass_final) index streams."""
  tick = counter()
  in_pass = tick % dimension
  return tick // dimension, in_pass, in_pass == 0, in_pass == (dimension - 1)


def relu(v) -> Stream:
  """Pointwise rectified linear unit over a Stream or scalar."""
  return Stream(v).max(0.0)


def biased_dot_product(x, w, b):
  """Computes the running dot product of x and w starting from bias b."""
  return prefix_sum(x * w, start=b)


@psyre_node('backfill')
def backfill(
    stream: Stream,
    end: Stream,
    *,
    max_span: int,
    shift_back: Callable[[Stream], Stream] | None = None,
) -> Stream:
  """Broadcasts values at `end` ticks backward up to `max_span` steps."""
  shift = post if shift_back is None else shift_back
  out = stream
  for _ in range(max_span):
    out = merge(end, when(end, stream), whennot(end, shift(out)))
  return out


def sum_stream(
    x,
    *,
    dimension: int = DIMENSION,
    end: Stream | None = None,
    shift_back: Callable[[Stream], Stream] | None = None,
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
  h_prev = Stream(0.0) if h is None else fby(0.0, h, name='rnn_fby')
  return tanh(Stream.sum(rnn.wxh * x, rnn.whh * h_prev, rnn.bh)), h_prev


def _cell_mask(in_pass: Stream, cell_idx: int, num_cells: int) -> Stream:
  """Selects ticks owned by unrolled RNN cell `cell_idx`."""
  return in_pass == cell_idx if cell_idx < num_cells - 1 else (
      in_pass >= cell_idx
  )


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
  pos = counter() if in_pass is None else in_pass
  h: Any = None
  y: Any = 0.0
  for k in range(num_cells):
    h, _ = rnn_cell(x, rnn, h)
    now = _cell_mask(pos, k, num_cells)
    y = h if k == 0 and num_cells == 1 else ifthenelse(now, h, y)
  return y


@psyre_node('grad_rnn_layer')
def grad_rnn_layer(
    x,
    rnn: RnnParams[Any],
    dy,
    *,
    num_cells: int = DIMENSION,
    dimension: int = DIMENSION,
    in_pass: Stream | None = None,
    shift_back: Callable[[Stream], Stream] | None = None,
) -> tuple[Stream, Any, RnnParams[Any]]:
  """Computes the forward and reverse-mode backward pass of the RNN layer."""
  if num_cells < 1:
    raise ValueError(f'num_cells must be >= 1, got {num_cells}.')
  if num_cells > dimension:
    raise ValueError(
        f'num_cells ({num_cells}) cannot exceed dimension ({dimension}).'
    )
  pos = counter() if in_pass is None else in_pass
  final = pos == (dimension - 1)
  shift = post if shift_back is None else shift_back
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
      dh_next = merge(final, 0.0, whennot(final, shift(dz_k * rnn.whh)))
  if len(dz_terms) == 1:
    dz, dwhh = dz_terms[0], dwhh_terms[0]
  elif len(dz_terms) == 2:
    dz = dz_terms[0] + dz_terms[1]
    dwhh = dwhh_terms[0] + dwhh_terms[1]
  else:
    dz = Stream.sum(*dz_terms)
    dwhh = Stream.sum(*dwhh_terms)
  return y, dz * rnn.wxh, RnnParams(wxh=dz * x, whh=dwhh, bh=dz)


@psyre_node('per_layer')
def per_layer(x, per: PerceptronParams[Any]) -> Stream:
  """Computes a dot-product perceptron layer with ReLU activation."""
  return relu(biased_dot_product(x, per.wxh, per.bh))


@psyre_node('grad_per_layer')
def grad_per_layer(
    x,
    per: PerceptronParams[Any],
    dy,
    *,
    dimension: int = DIMENSION,
    final: Stream | None = None,
    shift_back: Callable[[Stream], Stream] | None = None,
) -> tuple[Stream, Any, PerceptronParams[Any]]:
  """Computes the forward and backward pass of the perceptron layer."""
  z = biased_dot_product(x, per.wxh, per.bh)
  y = relu(z)
  dz = ifthenelse(z > 0.0, dy, 0.0)
  ps = prefix_sum(dz)
  end = _pass_final(dimension) if final is None else final
  total = backfill(ps, end, max_span=dimension - 1, shift_back=shift_back)
  dp = total - fby(0.0, ps)
  return y, dp * per.wxh, PerceptronParams(wxh=dp * x, bh=morph(dp, 0.0))


def loss(y_pred, y_true):
  """Computes pointwise squared error between y_pred and y_true."""
  return (y_pred - y_true) ** 2


def grad_loss(y_pred, y_true, dloss):
  """Computes pointwise squared error and its gradient with respect to y_pred."""
  err = y_pred - y_true
  return err**2, 2.0 * err * dloss


def forward(
    x,
    params: ParamStreams,
    *,
    num_cells: int = DIMENSION,
    in_pass: Stream | None = None,
) -> Stream:
  """Computes the 2-layer RNN + perceptron forward prediction stream."""
  y_pred = per_layer(
      rnn_layer(x, params.rnn, num_cells=num_cells, in_pass=in_pass),
      params.per,
  )
  y_pred.name = 'y_pred'
  return y_pred


def grad_rnn_perceptron(
    x,
    params: ParamStreams,
    dy,
    *,
    num_cells: int = DIMENSION,
    dimension: int = DIMENSION,
    in_pass: Stream | None = None,
    shift_back: Callable[[Stream], Stream] | None = None,
) -> tuple[Stream, Any, ParamStreams]:
  """Computes forward and reverse-mode backward pass of RNN + perceptron."""
  pos = counter() if in_pass is None else in_pass
  final = pos == (dimension - 1)
  h, dx, drnn = grad_rnn_layer(
      x,
      params.rnn,
      lambda: dh,
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
  return y_pred, dx, ModelParams(rnn=drnn, per=dper)


def grad_model(
    x,
    y_true,
    params: ParamStreams,
    *,
    num_cells: int = DIMENSION,
    dimension: int = DIMENSION,
) -> tuple[Stream, Stream, ParamStreams]:
  """Computes predictions, loss, and parameter gradients over one pass."""
  dloss = ifthenelse(_pass_final(dimension), 1.0, 0.0)
  y_pred, _, grads = grad_rnn_perceptron(
      x,
      params,
      lambda: dy_pred,
      num_cells=num_cells,
      dimension=dimension,
  )
  loss_bwd, dy_pred = grad_loss(y_pred, y_true, dloss)
  return y_pred, loss_bwd, grads


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


def _random_param_streams() -> ParamStreams:
  """Generates initial random parameter streams with shared RNN scalars."""
  first = Stream.once(0)
  return ModelParams.build(
      lambda: current(first, Stream.random(), default=0.0),
      Stream.random,
  )


def simple_model() -> SimpleModelOutput:
  """Runs the forward pass and one iteration of gradient descent."""
  lr = 0.1
  x = Stream.periodic([], [1.0, 2.0, -1.0, 0.0, 1.0])
  y_true = 42.0
  params = _random_param_streams()

  y_pred_fwd = forward(x, params)
  y_pred_fwd.name = 'y_pred_fwd'
  loss_fwd = loss(y_pred_fwd, y_true)
  loss_fwd.name = 'loss_fwd'
  y_pred, loss_bwd, grads = grad_model(x, y_true, params)
  y_pred.name = 'y_pred'
  loss_bwd.name = 'loss_bwd'
  next_params = apply_grads(params, grads, lr)
  return SimpleModelOutput(
      y_pred_fwd=y_pred_fwd,
      loss_fwd=loss_fwd,
      y_pred=y_pred,
      loss_bwd=loss_bwd,
      params=next_params,
  )


def _normalize_param_scalar(raw_value: Any, field_name: str) -> float:
  """Normalizes a shared scalar parameter entry (float or length-1 sequence)."""
  if isinstance(raw_value, (int, float)):
    return float(raw_value)
  values = [float(item) for item in raw_value]
  if len(values) != 1:
    raise ValueError(
        f'Shared parameter {field_name!r} has length {len(values)}, expected '
        'a scalar or length-1 sequence.'
    )
  return values[0]


def _normalize_param_vector(raw_value: Any, field_name: str) -> list[float]:
  """Normalizes a scalar or sequence parameter entry to a pass-length vector."""
  if isinstance(raw_value, (int, float)):
    return [float(raw_value)] * DIMENSION
  values = [float(item) for item in raw_value]
  if len(values) == 1:
    return values * DIMENSION
  if len(values) != DIMENSION:
    raise ValueError(
        f'Parameter {field_name!r} has length {len(values)}, expected 1 or '
        f'{DIMENSION}.'
    )
  return values


def normalize_params(params: ModelParams[Any, Any]) -> ParamValues:
  """Validates and normalizes model parameters into `ParamValues`."""
  return ModelParams(
      rnn=params.rnn.map(_normalize_param_scalar, _PARAM_LABELS.rnn),
      per=params.per.map(_normalize_param_vector, _PARAM_LABELS.per),
  )


def _extract_present_window(values: Sequence[Any], label: str) -> list[float]:
  """Returns the trailing pass of present float values from a trace."""
  extracted = [float(val) for val in present_values(values)]
  if len(extracted) < DIMENSION:
    raise ValueError(
        f'Trace {label!r} has {len(extracted)} present values, expected '
        f'at least {DIMENSION}.'
    )
  return extracted[-DIMENSION:]


def _extract_shared_scalar(values: Sequence[Any], label: str) -> float:
  """Extracts and validates a pass-constant scalar parameter from a trace."""
  window = _extract_present_window(values, label)
  if any(val != window[0] for val in window):
    raise ValueError(
        f'Shared parameter {label!r} is not constant across pass: {window!r}.'
    )
  return window[-1]


def extract_params(traces: ParamTraces) -> ParamValues:
  """Extracts the latest full pass of ModelParams from recorded traces."""
  return ModelParams(
      rnn=traces.rnn.map(_extract_shared_scalar, _PARAM_LABELS.rnn),
      per=traces.per.map(_extract_present_window, _PARAM_LABELS.per),
  )


def _vector_stream(in_pass: Stream, values: Sequence[float]) -> Stream:
  """Selects values[in_pass] at each local coordinate of a pass."""
  return switch(
      in_pass,
      {coord: float(val) for coord, val in enumerate(values)},
      default=0.0,
  )


def _parse_pass_stream(
    raw_values: Sequence[Any],
    in_pass: Stream | None,
    label: str,
) -> Stream:
  """Validates a pass-length sequence and converts it to a Psyre Stream."""
  values = [float(item) for item in raw_values]
  if len(values) != DIMENSION:
    raise ValueError(f'{label} has length {len(values)}, expected {DIMENSION}.')
  if in_pass is None:
    return Stream.periodic([], values)
  return _vector_stream(in_pass, values)


def dataset_streams(
    samples: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    *,
    in_pass: Stream | None = None,
) -> list[LabeledStream]:
  """Converts in-memory `(x_seq, y_target)` examples into `LabeledStream`s."""
  streams: list[LabeledStream] = []
  for idx, (x_seq, raw_y) in enumerate(samples):
    x_stream = _parse_pass_stream(x_seq, in_pass, f'Dataset item {idx}')
    if isinstance(raw_y, Sequence):
      y_stream = _parse_pass_stream(raw_y, in_pass, f'Dataset target {idx}')
    else:
      y_stream = Stream(float(raw_y))
    streams.append(LabeledStream(x=x_stream, y=y_stream))
  return streams


def _delay_pass(source: Any, init: Any = 0.0) -> Stream:
  """Delays defined values of source by one pass at automaton scope."""
  return last(source, init=init, n=DIMENSION)


def _delay_params(get_params: Callable[[], ParamStreams]) -> ParamStreams:
  """Delays all parameter streams in get_params() by one pass via last()."""
  return ModelParams(
      rnn=RnnParams(
          wxh=_delay_pass(lambda: get_params().rnn.wxh),
          whh=_delay_pass(lambda: get_params().rnn.whh),
          bh=_delay_pass(lambda: get_params().rnn.bh),
      ),
      per=PerceptronParams(
          wxh=_delay_pass(lambda: get_params().per.wxh),
          bh=_delay_pass(lambda: get_params().per.bh),
      ),
  )


def _zero_param_streams() -> ParamStreams:
  """Returns constant zero ParamStreams for unreachable states."""
  return ModelParams.build(lambda: Stream(0.0))


def _param_streams(
    in_pass: Stream, params: ModelParams[Any, Any]
) -> ParamStreams:
  """Streams the shared RNN scalars and perceptron vectors from `params`."""
  normalized = normalize_params(params)
  return ModelParams(
      rnn=normalized.rnn.map(Stream),
      per=normalized.per.map(lambda vec: _vector_stream(in_pass, vec)),
  )


def _leave_one_out_samples(
    samples: Sequence[_T],
    holdout_index: int,
) -> list[_T]:
  """Excludes holdout_index from samples under the leave-one-out policy."""
  if len(samples) < 2:
    raise ValueError(
        'Leave-one-out training requires at least 2 dataset streams.'
    )
  if not 0 <= holdout_index < len(samples):
    raise ValueError(
        f'holdout_index {holdout_index} out of range for '
        f'{len(samples)} dataset streams.'
    )
  return [sample for idx, sample in enumerate(samples) if idx != holdout_index]


def _select_sample(
    index_stream: Stream,
    samples: Sequence[LabeledStream],
) -> LabeledStream:
  """Multiplexes a sequence of LabeledStreams by index_stream."""
  return LabeledStream(
      x=switch(
          index_stream,
          {idx: sample.x for idx, sample in enumerate(samples)},
          default=0.0,
      ),
      y=switch(
          index_stream,
          {idx: sample.y for idx, sample in enumerate(samples)},
          default=0.0,
      ),
  )


def _inference_streams(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
) -> tuple[Stream, Stream]:
  """Returns `(loss, emit_loss)` streams from the nested inference automaton."""
  require_sync_semantics('inference_machine')
  if params is None:
    raise ValueError('params is required for inference_machine.')

  with automaton(name='infer') as infer:
    with state(InferenceState.INIT):
      until(_pass_final(), InferenceState.EVAL)
      infer.loss = 0.0
      infer.emit_loss = False
    with state(InferenceState.EVAL):
      eval_idx, in_pass, pass_reset, pass_final = _pass_indices()
      samples = dataset_streams(dataset, in_pass=in_pass)
      sample = _select_sample(eval_idx, samples)
      eval_params = _param_streams(in_pass, params)
      with reset(pass_reset):
        infer.loss = loss(forward(sample.x, eval_params), sample.y)
      infer.emit_loss = pass_final & (eval_idx < len(samples))

  return infer.loss, infer.emit_loss


def inference_machine(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
) -> Stream:
  """Nested automaton state machine for inference mode (init and eval).

  Args:
    dataset: Sequence of `(x_seq, y_target)` examples.
    params: Initial model parameters evaluated during inference.

  Returns:
    Evaluation loss stream emitting the loss at the end of every input.
  """
  eval_loss, emit_loss = _inference_streams(dataset, params)
  return when(emit_loss, eval_loss, name='eval_loss')


def _training_streams(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
    *,
    init_state: TrainingState = TrainingState.INIT_RANDOM,
    max_steps: int = DEFAULT_MAX_STEPS,
    holdout_index: int = 0,
    lr: float = DEFAULT_LEARNING_RATE,
) -> tuple[ParamStreams, Stream, Stream]:
  """Returns `(params, loss, emit_loss)` streams from the training automaton."""
  require_sync_semantics('training_machine')
  if max_steps < 1:
    raise ValueError(f'max_steps must be >= 1, got {max_steps}.')
  if init_state == TrainingState.GRAD_DESCENT:
    raise ValueError(
        'init_state must be TrainingState.INIT_RANDOM or '
        'TrainingState.INIT_LOAD.'
    )
  if init_state == TrainingState.INIT_LOAD and params is None:
    raise ValueError('params is required for TrainingState.INIT_LOAD.')
  train_dataset = _leave_one_out_samples(dataset, holdout_index)

  with automaton(init=init_state.name, name='train') as train:
    prev = _delay_params(lambda: train.params)
    with state(init_state.name):
      until(_pass_final(), TrainingState.GRAD_DESCENT.name)
      if init_state == TrainingState.INIT_RANDOM:
        train.params = _random_param_streams()
      else:
        assert params is not None
        train.params = _param_streams(counter(), params)
      train.loss = 0.0
      train.emit_loss = False
    with state(TrainingState.GRAD_DESCENT.name):
      step, in_pass, pass_reset, pass_final = _pass_indices()
      train_samples = dataset_streams(train_dataset, in_pass=in_pass)
      sample = _select_sample(step % len(train_samples), train_samples)
      active_step = step < max_steps
      with reset(pass_reset):
        _, train.loss, grads = grad_model(sample.x, sample.y, prev)
        updated = apply_grads(prev, grads, lr)
      train.emit_loss = pass_final & active_step
      train.params = updated.map(
          lambda upd, old: ifthenelse(active_step, upd, old), prev
      )

  return train.params, train.loss, train.emit_loss


def training_machine(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
    *,
    init_state: TrainingState = TrainingState.INIT_RANDOM,
    max_steps: int = DEFAULT_MAX_STEPS,
    holdout_index: int = 0,
    lr: float = DEFAULT_LEARNING_RATE,
) -> TrainingOutput:
  """Nested automaton state machine for training mode (init and BPTT).

  Args:
    dataset: Sequence of `(x_seq, y_target)` examples.
    params: Initial model parameters loaded in `TrainingState.INIT_LOAD`.
    init_state: Initial state (`TrainingState.INIT_RANDOM` or
      `TrainingState.INIT_LOAD`).
    max_steps: Upper bound on the number of BPTT learning steps.
    holdout_index: Index of the dataset stream excluded from BPTT updates under
      the leave-one-out policy.
    lr: Gradient descent learning rate.

  Returns:
    `TrainingOutput(params, train_loss)`.
  """
  train_params, loss_bwd, emit_loss = _training_streams(
      dataset,
      params,
      init_state=init_state,
      max_steps=max_steps,
      holdout_index=holdout_index,
      lr=lr,
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
    max_steps: int = DEFAULT_MAX_STEPS,
    holdout_index: int = 0,
    lr: float = DEFAULT_LEARNING_RATE,
) -> HierarchicalOutput:
  """High-level automaton machine orchestrating training and inference modes.

  Args:
    dataset: Sequence of `(x_seq, y_target)` examples.
    params: Initial model parameters used in `Mode.INFERENCE` and
      `TrainingState.INIT_LOAD`.
    initial_mode: Initial high-level mode (`Mode.TRAINING` or `Mode.INFERENCE`).
    switch_mode: Optional boolean `Stream` guard toggling between
      `Mode.TRAINING` and `Mode.INFERENCE` at the end of a tick. If `None`, no
      mode switch occurs.
    train_init_state: Initial nested training state (`TrainingState.INIT_RANDOM`
      or `TrainingState.INIT_LOAD`).
    max_steps: Bounded number of BPTT learning steps in `Mode.TRAINING`.
    holdout_index: Dataset index held out during leave-one-out BPTT training.
    lr: Gradient descent learning rate.

  Returns:
    `HierarchicalOutput(mode, params, train_loss, eval_loss)`.
  """
  require_sync_semantics('hierarchical_model')
  if switch_mode is None:
    if initial_mode == Mode.TRAINING:
      train_out = training_machine(
          dataset=dataset,
          params=params,
          init_state=train_init_state,
          max_steps=max_steps,
          holdout_index=holdout_index,
          lr=lr,
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

  with automaton(init=initial_mode, name='top') as top:
    with state(Mode.TRAINING) as mode:
      until(switch_mode, Mode.INFERENCE)
      top.mode = mode
      top.params, top.loss, top.emit_loss = _training_streams(
          dataset=dataset,
          params=params,
          init_state=train_init_state,
          max_steps=max_steps,
          holdout_index=holdout_index,
          lr=lr,
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
  return HierarchicalOutput(
      mode=top.mode,
      params=top.params.map(lambda p: when(is_training, p)),
      train_loss=when(
          is_training & top.emit_loss, top.loss, name='train_loss'
      ),
      eval_loss=when(~is_training & top.emit_loss, top.loss, name='eval_loss'),
  )
