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
"""Shared JAX MNIST CNN model, stream training loop, and IDX data loader.

Implements a fixed `float32` 4-layer CNN (`Conv+ReLU -> MaxPool -> Flatten ->
Dense+ReLU -> Dense+Softmax`) using `jax.lax`, `psyre.jax.value_and_grad`, a
single `fby` parameter feedback stream carrying `CnnParams`, and
`Stream.periodic` mini-batch cycling. See `cnn_mnist_eager.py` and
`cnn_mnist_jit.py` for the eager and XLA JIT-compiled specializations.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import gzip
import math
import os
from typing import Any, NamedTuple

# pylint: disable=g-import-not-at-top
try:
  import jax
  from jax import lax
  import jax.numpy as jnp
  import numpy as np
except ImportError as err:  # pragma: no cover
  raise ImportError(
      'psyre.jax.examples.cnn_mnist requires JAX. Install it with: '
      'pip install "psyre[jax]"'
  ) from err

# pylint: disable=g-importing-member
from psyre import jax as srjax
from psyre.builder import build_graph
from psyre.common import fby
from psyre.graph import Graph
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member


class CnnParams(NamedTuple):
  """Immutable `float32` CNN parameter arrays."""

  conv_w: jax.Array
  conv_b: jax.Array
  fc1_w: jax.Array
  fc1_b: jax.Array
  fc2_w: jax.Array
  fc2_b: jax.Array


@dataclasses.dataclass
class Cnn:
  """Base JAX MNIST CNN holding hyperparameters and trained state."""

  num_filters: int = 32
  filter_width: int = 2
  conv_stride: int = 1
  pool_width: int = 3
  pool_stride: int = 3
  hidden_dim: int = 100
  num_classes: int = 10
  learning_rate: float = 0.001
  num_epochs: int = 3
  mini_batches: int = 500
  seed: int = 0

  params: CnnParams | None = dataclasses.field(
      default=None, init=False, repr=False
  )
  params_history: list[CnnParams] = dataclasses.field(
      default_factory=list, init=False, repr=False
  )
  cce_history: list[float] = dataclasses.field(
      default_factory=list, init=False, repr=False
  )
  ce_history: list[float] = dataclasses.field(
      default_factory=list, init=False, repr=False
  )
  test_cce: float | None = dataclasses.field(default=None, init=False)
  test_ce: float | None = dataclasses.field(default=None, init=False)

  def _require_params(self) -> CnnParams:
    if self.params is None:
      raise RuntimeError('Parameters uninitialized; call train() first.')
    return self.params

  def init_params(self, input_shape: tuple[int, int, int]) -> CnnParams:
    """Initializes and stores He-normal `float32` parameters."""
    ih, iw, c_in = input_shape
    ch = (ih - self.filter_width) // self.conv_stride + 1
    cw = (iw - self.filter_width) // self.conv_stride + 1
    ph = (ch - self.pool_width) // self.pool_stride + 1
    pw = (cw - self.pool_width) // self.pool_stride + 1
    if ch < 1 or cw < 1 or ph < 1 or pw < 1:
      raise ValueError(f'Input shape {input_shape} is too small for {self}.')
    flat_dim = ph * pw * self.num_filters
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(self.seed), 3)
    fw = self.filter_width
    self.params = CnnParams(
        conv_w=jax.random.normal(k1, (fw, fw, c_in, self.num_filters))
        * math.sqrt(2.0 / (fw * fw * c_in)),
        conv_b=jnp.zeros((1, 1, 1, self.num_filters)),
        fc1_w=jax.random.normal(k2, (self.hidden_dim, flat_dim))
        * math.sqrt(2.0 / flat_dim),
        fc1_b=jnp.zeros((self.hidden_dim,)),
        fc2_w=jax.random.normal(k3, (self.num_classes, self.hidden_dim))
        * math.sqrt(2.0 / self.hidden_dim),
        fc2_b=jnp.zeros((self.num_classes,)),
    )
    return self.params

  def forward_logits(self, params: CnnParams, x: jax.Array) -> jax.Array:
    """Pure instantaneous JAX forward pass returning class logits."""
    h = (
        lax.conv_general_dilated(
            x,
            params.conv_w,
            window_strides=(self.conv_stride, self.conv_stride),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        + params.conv_b
    )
    h = jax.nn.relu(h)
    # Static host scalar keeps `reduce_window` differentiable under `jax.jit`.
    h = lax.reduce_window(
        h,
        np.asarray(-np.inf, dtype=h.dtype),
        lax.max,
        window_dimensions=(1, self.pool_width, self.pool_width, 1),
        window_strides=(1, self.pool_stride, self.pool_stride, 1),
        padding='VALID',
    )
    h = h.reshape((h.shape[0], -1))
    h = jax.nn.relu(h @ params.fc1_w.T + params.fc1_b)
    return h @ params.fc2_w.T + params.fc2_b

  def loss_and_error(
      self, params: CnnParams, x: jax.Array, y: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Returns `(mean_cross_entropy, classification_error)` for `(x, y)`."""
    logits = self.forward_logits(params, jnp.asarray(x))
    y_arr = jnp.asarray(y)
    cce = -jnp.sum(y_arr * jax.nn.log_softmax(logits, axis=1)) / logits.shape[0]
    ce = jnp.mean(jnp.argmax(logits, axis=1) != jnp.argmax(y_arr, axis=1))
    return cce, ce

  def _probs(self, params: CnnParams, x: jax.Array) -> jax.Array:
    return jax.nn.softmax(self.forward_logits(params, jnp.asarray(x)))

  def _sgd_step(self, params: CnnParams, grads: CnnParams) -> CnnParams:
    return jax.tree_util.tree_map(
        lambda p, g: p - self.learning_rate * g, params, grads
    )

  def predict(self, x: Any) -> Any:
    """Returns softmax probabilities (implemented by eager/JIT subclasses)."""
    raise NotImplementedError

  def evaluate(self, x: Any, y: Any) -> tuple[float, float]:
    """Evaluates `(cce, ce)` (implemented by eager/JIT subclasses)."""
    raise NotImplementedError

  def training_streams(
      self, x_batch: Stream, y_batch: Stream
  ) -> tuple[Stream, Stream, Stream, Stream]:
    """Builds `(params, next_params, cce, ce)` streams with `fby` feedback."""
    step = srjax.value_and_grad(
        self.loss_and_error, has_aux=True, name='cnn_value_and_grad'
    )
    sgd = srjax.lift(self._sgd_step, pure=True, name='sgd')
    params = fby(self._require_params(), lambda: next_params, name='params')
    (cce, ce), grads = step(params, x_batch, y_batch)
    next_params = sgd(params, grads)
    return params, next_params, cce, ce

  def build_graph(
      self, x_batches: Sequence[Any], y_batches: Sequence[Any]
  ) -> Graph:
    """Builds the Stage-1 training `Graph` with periodic mini-batches."""
    xs = [jnp.asarray(b) for b in x_batches]
    ys = [jnp.asarray(b) for b in y_batches]
    if not xs or len(xs) != len(ys):
      raise ValueError('x_batches and y_batches must be non-empty and equal.')
    if self.params is None:
      self.init_params(tuple(xs[0].shape[1:]))
    return build_graph(
        lambda: self.training_streams(
            Stream.periodic([], xs, name='x_batch'),
            Stream.periodic([], ys, name='y_batch'),
        )
    )

  def train(
      self,
      train_set: tuple[Any, Any],
      test_set: tuple[Any, Any] | None = None,
  ) -> None:
    """Trains via Psyre streams and stores parameters and histories on `self`."""
    if self.num_epochs < 1:
      raise ValueError(f'num_epochs must be >= 1, got {self.num_epochs}.')
    xs, ys = make_minibatches(train_set, self.mini_batches)
    graph = self.build_graph(xs, ys)
    params_hist, next_params_hist, cce, ce = Scheduler.run_graph(
        graph, horizon=self.num_epochs * len(xs)
    )
    self.params_history = list(params_hist)
    self.params = next_params_hist[-1]
    self.cce_history = [float(v) for v in cce]
    self.ce_history = [float(v) for v in ce]
    if test_set is not None:
      self.test_cce, self.test_ce = self.evaluate(*test_set)

  def train_mnist(self, data_dir: str | os.PathLike[str]) -> None:
    """Loads local MNIST IDX archives and runs `self.train(...)`."""
    self.train(*load_mnist_idx(data_dir))


def make_minibatches(
    dataset: tuple[Any, Any], num_batches: int
) -> tuple[list[jax.Array], list[jax.Array]]:
  """Splits `(x, y)` into non-empty contiguous JAX mini-batches."""
  if num_batches < 1:
    raise ValueError(f'num_batches must be >= 1, got {num_batches}.')
  x_arr, y_arr = jnp.asarray(dataset[0]), jnp.asarray(dataset[1])
  if x_arr.shape[0] == 0 or x_arr.shape[0] != y_arr.shape[0]:
    raise ValueError('x and y must be non-empty with equal sample counts.')
  xs = [b for b in jnp.array_split(x_arr, num_batches) if b.shape[0] > 0]
  ys = [b for b in jnp.array_split(y_arr, num_batches) if b.shape[0] > 0]
  return xs, ys


def _read_idx(path: str, n_dims: int) -> tuple[tuple[int, ...], bytes]:
  with gzip.open(path, 'rb') as f:
    header = f.read(4 * (1 + n_dims))
    dims = tuple(
        int.from_bytes(header[4 * i : 4 * (i + 1)], 'big')
        for i in range(1, 1 + n_dims)
    )
    return dims, f.read(math.prod(dims))


def load_mnist_idx(
    data_dir: str | os.PathLike[str],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
  """Loads local MNIST `.gz` IDX archives as `(train_set, test_set)`."""
  root = os.fspath(data_dir)

  def _images(name: str) -> jax.Array:
    dims, buf = _read_idx(os.path.join(root, name), 3)
    return (jnp.frombuffer(buf, dtype=jnp.uint8) / 255.0).reshape((*dims, 1))

  def _labels(name: str) -> jax.Array:
    _, buf = _read_idx(os.path.join(root, name), 1)
    return jnp.frombuffer(buf, dtype=jnp.uint8)

  tr_x = _images('train-images-idx3-ubyte.gz')
  te_x = _images('t10k-images-idx3-ubyte.gz')
  tr_y = _labels('train-labels-idx1-ubyte.gz')
  te_y = _labels('t10k-labels-idx1-ubyte.gz')
  eye = jnp.eye(max(int(jnp.max(tr_y)), int(jnp.max(te_y))) + 1)
  return (tr_x, eye[tr_y]), (te_x, eye[te_y])
