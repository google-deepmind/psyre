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
"""Psyre recurrent neural network, perceptron layer, and loss model.

Implements an unrolled RNN + ReLU perceptron with bidirectional reverse-mode
automatic differentiation (via `post` and `backfill`; see Potop-Butucaru,
Cohen, Plotkin & Pompougnac, 2023, arXiv:2311.16977) and hierarchical
training/inference mode automata (Colaço, Pagano & Pouzet, EMSOFT '05).
"""

from collections.abc import Callable, Sequence
import dataclasses
import enum
from typing import Any, NamedTuple

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
    is_namedtuple,
    merge,
    morph,
    post,
    prefix_sum,
    psyre_node,
    reset,
    switch,
    when,
    whennot,
)
from psyre.match import last
from psyre.math import tanh
from psyre.static_stream import (
    StaticWord,
    concat,
    repeat,
)
from psyre.stream import Stream
from psyre.trace import present_values

# pylint: enable=g-importing-member,g-multiple-import

DIMENSION = 5
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MAX_STEPS = 3


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


@dataclasses.dataclass(frozen=True)
class HyperParams:
  """Training hyperparameters for the RNN + perceptron model."""

  lr: float = DEFAULT_LEARNING_RATE
  max_steps: int = DEFAULT_MAX_STEPS
  holdout_index: int = 0

  def __post_init__(self) -> None:
    if self.max_steps < 1:
      raise ValueError(f'max_steps must be >= 1, got {self.max_steps}.')


def map_fields(tree: Any, fn: Callable[..., Any], *others: Any) -> Any:
  """Applies `fn` field-wise across a NamedTuple `tree` and parallel trees."""
  return type(tree)(*(fn(*col) for col in zip(tree, *others)))


def select_params(cond: Stream, then_params: Any, else_params: Any) -> Any:
  """Selects leaf-wise between two parameter trees under boolean `cond`."""
  return then_params.map(
      lambda upd, old: ifthenelse(cond, upd, old), else_params
  )


def sample_params(cond: Stream, params: Any) -> Any:
  """Downsamples every leaf stream of `params` onto subclock `cond`."""
  return params.map(lambda p: when(cond, p))


def delay_tree(
    template: Any,
    source: Any,
    delay_fn: Callable[[Any, Any], Stream],
    *,
    by_getter: bool = False,
) -> Any:
  """Recursively delays every leaf of a NamedTuple/tuple parameter tree."""
  if isinstance(template, tuple):
    if by_getter:

      def _child(idx: int) -> Any:
        return delay_tree(
            template[idx],
            lambda: source()[idx],
            delay_fn,
            by_getter=True,
        )

      items = tuple(_child(i) for i in range(len(template)))
    else:
      items = tuple(
          delay_tree(template[i], source[i], delay_fn)
          for i in range(len(template))
      )
    return type(template)(*items) if is_namedtuple(template) else items
  return delay_fn(template, source)


class RnnParams[T](NamedTuple):
  """Parameters (wxh, whh, bh) of the 1-layer recurrent neural network."""

  wxh: T
  whh: T
  bh: T

  @classmethod
  def build[U](cls, fn: Callable[[], U]) -> 'RnnParams[U]':
    """Constructs RnnParams by evaluating fn() for each field."""
    return cls(wxh=fn(), whh=fn(), bh=fn())

  def map[U](
      self, fn: Callable[..., U], *others: 'RnnParams[Any]'
  ) -> 'RnnParams[U]':
    """Applies fn field-wise across self and any parallel RnnParams."""
    return map_fields(self, fn, *others)


class PerceptronParams[T](NamedTuple):
  """Parameters (wxh, bh) of the perceptron readout layer."""

  wxh: T
  bh: T

  @classmethod
  def build[U](cls, fn: Callable[[], U]) -> 'PerceptronParams[U]':
    """Constructs PerceptronParams by evaluating fn() per field."""
    return cls(wxh=fn(), bh=fn())

  def map[U](
      self, fn: Callable[..., U], *others: 'PerceptronParams[Any]'
  ) -> 'PerceptronParams[U]':
    """Applies fn field-wise across self and any parallel PerceptronParams."""
    return map_fields(self, fn, *others)


class ModelParams[RnnT, PerT](NamedTuple):
  """Composite parameters bundling the RNN and perceptron readout layers."""

  rnn: RnnParams[RnnT]
  per: PerceptronParams[PerT]

  @classmethod
  def build(
      cls,
      rnn_fn: Callable[[], RnnT],
      per_fn: Callable[[], PerT] | None = None,
  ) -> 'ModelParams[RnnT, PerT]':
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


def pass_final(dimension: int = DIMENSION) -> Stream:
  """Emits True on the final coordinate (dimension - 1) of a pass."""
  return Stream.once(dimension - 1)


def pass_reset_word(dimension: int = DIMENSION) -> StaticWord:
  """Returns the periodic `dimension`-cycle pass-start reset pattern."""
  return StaticWord(True) + repeat(False, dimension - 1)


def pass_end_word(dimension: int = DIMENSION) -> StaticWord:
  """Returns the periodic `dimension`-cycle pass-end pulse pattern."""
  return repeat(False, dimension - 1) + True


def relu(v) -> Stream:
  """Pointwise rectified linear unit over a Stream, LazyThunk, or scalar."""
  return Stream.max(v, 0.0)


def biased_dot_product(x, w, b):
  """Computes the running dot product of x and w starting from bias b."""
  return prefix_sum(x * w, start=b)


@psyre_node('backfill')
def backfill(
    stream: Stream,
    end: Stream,
    *,
    max_span: int,
    shift_back: Callable[[Stream], Stream] = post,
) -> Stream:
  """Broadcasts values at `end` ticks backward up to `max_span` steps."""
  # Unrolling the reverse-time recurrence `out = i if end else post(out)` into
  # `max_span` distinct pipeline stages breaks the single-node causality cycle
  # so each upstream stage can look ahead one tick at a time.
  out = stream
  for _ in range(max_span):
    out = merge(end, when(end, stream), whennot(end, shift_back(out)))
  return out


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
  pass_end = pass_final(dimension) if end is None else end
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


def _unroll_rnn_cells(
    x,
    rnn: RnnParams[Any],
    pos: Stream,
    num_cells: int,
) -> tuple[Stream, list[tuple[Stream, Stream, Stream]]]:
  """Unrolls `num_cells` forward RNN stages and collects `(now, h, h_prev)`."""
  cells: list[tuple[Stream, Stream, Stream]] = []
  h: Any = None
  y: Any = 0.0
  for k in range(num_cells):
    h, h_prev = rnn_cell(x, rnn, h)
    now = _cell_mask(pos, k, num_cells)
    y = h if k == 0 and num_cells == 1 else ifthenelse(now, h, y)
    cells.append((now, h, h_prev))
  return y, cells


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
  y, _ = _unroll_rnn_cells(x, rnn, pos, num_cells)
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
    shift_back: Callable[[Stream], Stream] = post,
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
  y, cells = _unroll_rnn_cells(x, rnn, pos, num_cells)

  # Reverse-mode BPTT sweeps unrolled cells back-to-front; `shift_back` moves
  # each `whh` adjoint one tick earlier in time (zeroed at pass end `final`).
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
      dh_next = merge(final, 0.0, whennot(final, shift_back(dz_k * rnn.whh)))
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
    shift_back: Callable[[Stream], Stream] = post,
) -> tuple[Stream, Any, PerceptronParams[Any]]:
  """Computes the forward and backward pass of the perceptron layer."""
  z = biased_dot_product(x, per.wxh, per.bh)
  y = relu(z)
  dz = ifthenelse(z > 0.0, dy, 0.0)
  # Adjoint of `prefix_sum` is the suffix sum `dp[t] = sum_{j>=t} dz[j]`,
  # obtained as `total - prefix_sum_prev`. Only bias `bh[0]` seeds `prefix_sum`,
  # so `morph(dp, 0.0)` emits `dp[0]` at tick 0 and `0.0` afterward.
  ps = prefix_sum(dz)
  end = pass_final(dimension) if final is None else final
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
    shift_back: Callable[[Stream], Stream] = post,
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
  dloss = ifthenelse(pass_final(dimension), 1.0, 0.0)
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


def random_param_streams() -> ParamStreams:
  """Generates initial random parameter streams with shared RNN scalars."""
  first = Stream.once(0)
  return ModelParams.build(
      lambda: current(first, Stream.random(), default=0.0),
      Stream.random,
  )


def simple_model(
    *,
    hyperparams: HyperParams = HyperParams(lr=0.1),
    grad_model_fn: Callable[..., tuple[Stream, Stream, ParamStreams]] | None = (
        None
    ),
) -> SimpleModelOutput:
  """Runs the forward pass and one iteration of gradient descent."""
  x = Stream.periodic([], [1.0, 2.0, -1.0, 0.0, 1.0])
  y_true = 42.0
  params = random_param_streams()
  build_grad = grad_model if grad_model_fn is None else grad_model_fn

  y_pred_fwd = forward(x, params)
  y_pred_fwd.name = 'y_pred_fwd'
  loss_fwd = loss(y_pred_fwd, y_true)
  loss_fwd.name = 'loss_fwd'
  y_pred, loss_bwd, grads = build_grad(x, y_true, params)
  y_pred.name = 'y_pred'
  loss_bwd.name = 'loss_bwd'
  next_params = apply_grads(params, grads, hyperparams.lr)
  return SimpleModelOutput(
      y_pred_fwd=y_pred_fwd,
      loss_fwd=loss_fwd,
      y_pred=y_pred,
      loss_bwd=loss_bwd,
      params=next_params,
  )


def to_param_float(raw_value: Any, field_name: str) -> float:
  """Converts a non-bool numeric parameter value to float."""
  if isinstance(raw_value, bool):
    raise TypeError(
        f'Parameter {field_name!r} must be numeric, not bool ({raw_value!r}).'
    )
  return float(raw_value)


def _validate_dimension(dimension: int) -> None:
  """Validates a pass dimension is a positive integer."""
  if isinstance(dimension, bool) or not isinstance(dimension, int):
    raise TypeError(
        f'dimension must be an int >= 1, got {type(dimension).__name__}.'
    )
  if dimension < 1:
    raise ValueError(f'dimension must be >= 1, got {dimension}.')


def _normalize_param_values(
    raw_value: Any,
    field_name: str,
    *,
    label: str,
    expected: str,
) -> list[float]:
  """Validates a raw parameter entry and returns normalized float value(s)."""
  if isinstance(raw_value, bool):
    raise TypeError(f'{label} {field_name!r} must be numeric, not bool.')
  if isinstance(raw_value, (int, float)):
    return [float(raw_value)]
  if isinstance(raw_value, (str, bytes, bytearray)):
    raise TypeError(
        f'{label} {field_name!r} must be {expected}, not '
        f'{type(raw_value).__name__}.'
    )
  return [to_param_float(item, field_name) for item in raw_value]


def normalize_param_scalar(raw_value: Any, field_name: str) -> float:
  """Normalizes a shared scalar parameter entry (float or length-1 sequence)."""
  values = _normalize_param_values(
      raw_value, field_name, label='Shared parameter', expected='numeric'
  )
  if len(values) != 1:
    raise ValueError(
        f'Shared parameter {field_name!r} has length {len(values)}, expected '
        'a scalar or length-1 sequence.'
    )
  return values[0]


def normalize_param_vector(
    raw_value: Any,
    field_name: str,
    *,
    dimension: int = DIMENSION,
) -> list[float]:
  """Normalizes a scalar or sequence parameter entry to a pass-length vector."""
  _validate_dimension(dimension)
  values = _normalize_param_values(
      raw_value,
      field_name,
      label='Parameter',
      expected='numeric or a sequence of numbers',
  )
  if len(values) == 1:
    return values * dimension
  if len(values) != dimension:
    raise ValueError(
        f'Parameter {field_name!r} has length {len(values)}, expected 1 or '
        f'{dimension}.'
    )
  return values


def normalize_params(
    params: ModelParams[Any, Any],
    *,
    dimension: int = DIMENSION,
) -> ParamValues:
  """Validates and normalizes model parameters into `ParamValues`."""
  _validate_dimension(dimension)
  return ModelParams(
      rnn=params.rnn.map(normalize_param_scalar, _PARAM_LABELS.rnn),
      per=params.per.map(
          lambda val, lbl: normalize_param_vector(
              val, lbl, dimension=dimension
          ),
          _PARAM_LABELS.per,
      ),
  )


def extract_present_window(
    values: Sequence[Any],
    label: str,
    *,
    dimension: int = DIMENSION,
) -> list[float]:
  """Returns the trailing pass of present float values from a trace."""
  _validate_dimension(dimension)
  extracted = [float(val) for val in present_values(values)]
  if len(extracted) < dimension:
    raise ValueError(
        f'Trace {label!r} has {len(extracted)} present values, expected '
        f'at least {dimension}.'
    )
  return extracted[-dimension:]


def extract_shared_scalar(
    values: Sequence[Any],
    label: str,
    *,
    dimension: int = DIMENSION,
) -> float:
  """Extracts and validates a pass-constant scalar parameter from a trace."""
  window = extract_present_window(values, label, dimension=dimension)
  if any(val != window[0] for val in window):
    raise ValueError(
        f'Shared parameter {label!r} is not constant across pass: {window!r}.'
    )
  return window[-1]


def extract_params(
    traces: ParamTraces,
    *,
    dimension: int = DIMENSION,
) -> ParamValues:
  """Extracts the latest full pass of ModelParams from recorded traces."""
  _validate_dimension(dimension)
  return ModelParams(
      rnn=traces.rnn.map(
          lambda seq, lbl: extract_shared_scalar(seq, lbl, dimension=dimension),
          _PARAM_LABELS.rnn,
      ),
      per=traces.per.map(
          lambda seq, lbl: extract_present_window(
              seq, lbl, dimension=dimension
          ),
          _PARAM_LABELS.per,
      ),
  )


def vector_stream(in_pass: Stream, values: Sequence[float]) -> Stream:
  """Selects values[in_pass] at each local coordinate of a pass."""
  return switch(
      in_pass,
      {coord: float(val) for coord, val in enumerate(values)},
      default=0.0,
  )


def _parse_pass_values(
    raw_values: Sequence[Any],
    label: str,
    *,
    dimension: int = DIMENSION,
) -> list[float]:
  """Validates a pass-length sequence of floats and returns it as a list."""
  if isinstance(raw_values, (str, bytes, bytearray)):
    raise TypeError(
        f'{label} must be a numeric sequence, not {type(raw_values).__name__}.'
    )
  values = [to_param_float(item, label) for item in raw_values]
  if len(values) != dimension:
    raise ValueError(f'{label} has length {len(values)}, expected {dimension}.')
  return values


def _parse_pass_stream(
    raw_values: Sequence[Any],
    in_pass: Stream | None,
    label: str,
) -> Stream:
  """Validates a pass-length sequence and converts it to a Psyre Stream."""
  values = _parse_pass_values(raw_values, label)
  if in_pass is None:
    return Stream.periodic([], values)
  return vector_stream(in_pass, values)


def dataset_words(
    samples: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    *,
    dimension: int = DIMENSION,
) -> tuple[StaticWord, StaticWord]:
  """Encodes `(x_seq, y_target)` examples as concatenated `StaticWord`s."""
  x_parts: list[StaticWord] = []
  y_parts: list[StaticWord] = []
  for idx, (x_seq, raw_y) in enumerate(samples):
    x_parts.append(
        StaticWord(
            _parse_pass_values(
                x_seq, f'Dataset item {idx}', dimension=dimension
            )
        )
    )
    if isinstance(raw_y, Sequence) and not isinstance(
        raw_y, (str, bytes, bytearray)
    ):
      y_parts.append(
          StaticWord(
              _parse_pass_values(
                  raw_y, f'Dataset target {idx}', dimension=dimension
              )
          )
      )
    else:
      y_parts.append(
          repeat(to_param_float(raw_y, f'Dataset target {idx}'), dimension)
      )
  return concat(*x_parts), concat(*y_parts)


def dataset_streams(
    samples: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    *,
    in_pass: Stream | None = None,
) -> list[LabeledStream]:
  """Converts in-memory `(x_seq, y_target)` examples into `LabeledStream`s."""
  streams: list[LabeledStream] = []
  for idx, (x_seq, raw_y) in enumerate(samples):
    x_stream = _parse_pass_stream(x_seq, in_pass, f'Dataset item {idx}')
    if isinstance(raw_y, Sequence) and not isinstance(
        raw_y, (str, bytes, bytearray)
    ):
      y_stream = _parse_pass_stream(raw_y, in_pass, f'Dataset target {idx}')
    else:
      y_stream = Stream(to_param_float(raw_y, f'Dataset target {idx}'))
    streams.append(LabeledStream(x=x_stream, y=y_stream))
  return streams


def delay_pass(source: Any, init: Any = 0.0) -> Stream:
  """Delays defined values of source by one pass at automaton scope."""
  return last(source, init=init, n=DIMENSION)


def zero_param_streams() -> ParamStreams:
  """Returns placeholder zero `ParamStreams` for `Mode.INFERENCE`."""
  return ModelParams.build(lambda: Stream(0.0))


def periodic_param_streams(params: ModelParams[Any, Any]) -> ParamStreams:
  """Streams shared RNN scalars and periodic perceptron vectors."""
  normalized = normalize_params(params)
  return ModelParams(
      rnn=normalized.rnn.map(Stream),
      per=normalized.per.map(lambda vec: Stream.periodic([], vec)),
  )


def leave_one_out_samples[T](
    samples: Sequence[T],
    holdout_index: int,
) -> list[T]:
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


def _inference_streams(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
) -> tuple[Stream, Stream]:
  """Returns `(loss, emit_loss)` streams from the nested inference automaton."""
  if params is None:
    raise ValueError('params is required for inference_machine.')
  if not dataset:
    raise ValueError('dataset must be non-empty.')
  x_word, y_word = dataset_words(dataset)

  with automaton(name='infer') as infer:
    with state(InferenceState.INIT):
      until(pass_final(), InferenceState.EVAL)
      infer.loss = 0.0
      infer.emit_loss = False
    with state(InferenceState.EVAL):
      pass_reset = Stream.periodic([], pass_reset_word())
      sample_x = Stream.periodic(x_word, [0.0])
      sample_y = Stream.periodic(y_word, [0.0])
      eval_params = periodic_param_streams(params)
      with reset(pass_reset):
        infer.loss = loss(forward(sample_x, eval_params), sample_y)
      infer.emit_loss = Stream.periodic(
          repeat(pass_end_word(), len(dataset)), [False]
      )

  return infer.loss, infer.emit_loss


def inference_machine(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
) -> Stream:
  """Nested automaton state machine for inference mode (init and eval).

  Args:
    dataset: Non-empty sequence of `(x_seq, y_target)` examples.
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

  with automaton(init=init_state.name, name='train') as train:
    prev = delay_tree(
        _PARAM_LABELS,
        lambda: train.params,
        lambda _, src: delay_pass(src),
        by_getter=True,
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

  return train.params, train.loss, train.emit_loss


def training_machine(
    dataset: Sequence[tuple[Sequence[float], float | Sequence[float]]],
    params: ModelParams[Any, Any] | None = None,
    *,
    init_state: TrainingState = TrainingState.INIT_RANDOM,
    hyperparams: HyperParams = HyperParams(),
    training_streams_fn: Callable[..., tuple[ParamStreams, Stream, Stream]]
    | None = None,
) -> TrainingOutput:
  """Nested automaton state machine for training mode (init and BPTT).

  Args:
    dataset: Sequence of `(x_seq, y_target)` examples.
    params: Initial model parameters loaded in `TrainingState.INIT_LOAD`.
    init_state: Initial state (`TrainingState.INIT_RANDOM` or
      `TrainingState.INIT_LOAD`).
    hyperparams: Training hyperparameters (`lr`, `max_steps`, `holdout_index`).
    training_streams_fn: Optional builder overriding `_training_streams`.

  Returns:
    `TrainingOutput(params, train_loss)`.
  """
  build_train = (
      _training_streams if training_streams_fn is None else training_streams_fn
  )
  train_params, loss_bwd, emit_loss = build_train(
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
    training_streams_fn: Callable[..., tuple[ParamStreams, Stream, Stream]]
    | None = None,
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
    hyperparams: Training hyperparameters (`lr`, `max_steps`, `holdout_index`).
    training_streams_fn: Optional builder overriding `_training_streams`.

  Returns:
    `HierarchicalOutput(mode, params, train_loss, eval_loss)`.
  """
  build_train = (
      _training_streams if training_streams_fn is None else training_streams_fn
  )
  if switch_mode is None:
    if initial_mode == Mode.TRAINING:
      train_out = training_machine(
          dataset=dataset,
          params=params,
          init_state=train_init_state,
          hyperparams=hyperparams,
          training_streams_fn=build_train,
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
      top.params, top.loss, top.emit_loss = build_train(
          dataset=dataset,
          params=params,
          init_state=train_init_state,
          hyperparams=hyperparams,
      )
    with state(Mode.INFERENCE) as mode:
      until(switch_mode, Mode.TRAINING)
      top.mode = mode
      top.params = zero_param_streams()
      top.loss, top.emit_loss = _inference_streams(
          dataset=dataset,
          params=params,
      )

  is_training = top.mode == Mode.TRAINING
  return HierarchicalOutput(
      mode=top.mode,
      params=sample_params(is_training, top.params),
      train_loss=when(
          is_training & top.emit_loss, top.loss, name='train_loss'
      ),
      eval_loss=when(~is_training & top.emit_loss, top.loss, name='eval_loss'),
  )
