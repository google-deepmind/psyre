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
"""CNN for MNIST trained via Psyre + NumPy streams.

Inspired from the Andrew Ng's convolutional neural network code:
https://www.coursera.org/learn/convolutional-neural-networks/programming/qO8ng/convolutional-model-step-by-step
in pure functional form, implementing a fixed `float32` 4-layer CNN
(`Conv+ReLU -> MaxPool -> Flatten -> Dense+ReLU -> Dense+Softmax`):

* `Cnn` holds hyperparameters, trained parameters, and training histories so
  stream equations and calls stay minimal.
* Stride-group convolution and max-pooling partition NHWC inputs into
  non-overlapping `(n_rows, n_cols)` window groups.
* A single `fby` feedback stream carries `CnnParams` across ticks, and
  `Stream.periodic` cycles mini-batches across epochs.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import gzip
import math
import os
from typing import NamedTuple

# pylint: disable=g-import-not-at-top
try:
  import numpy as np
except ImportError as err:  # pragma: no cover
  raise ImportError(
      'psyre.numpy.examples.cnn_mnist requires NumPy. Install it with: '
      'pip install "psyre[numpy]"'
  ) from err

# pylint: disable=g-importing-member,g-multiple-import
from psyre import numpy as pnp
from psyre.builder import build_graph
from psyre.common import fby
from psyre.graph import Graph
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import

_SgCache = tuple[tuple[int, ...], list[list[np.ndarray]], np.ndarray]


class CnnParams(NamedTuple):
  """Immutable `float32` CNN parameter arrays."""

  conv_w: np.ndarray
  conv_b: np.ndarray
  fc1_w: np.ndarray
  fc1_b: np.ndarray
  fc2_w: np.ndarray
  fc2_b: np.ndarray


# ---------------------------------------------------------------------------
# Pure functional stride-group NumPy kernels
# ---------------------------------------------------------------------------


def _stride_group_geometry(
    ih: int, iw: int, fw: int, stride: int
) -> tuple[int, int, int, int, int, int]:
  z_h = (ih - fw) // stride + 1
  z_w = (iw - fw) // stride + 1
  g_step = math.ceil(fw / stride)
  pix_step = g_step * stride
  return z_h, z_w, g_step, pix_step, min(g_step, z_h), min(g_step, z_w)


def _extract_group_blocks(
    x: np.ndarray,
    top: int,
    left: int,
    gh: int,
    gw: int,
    fw: int,
    pix_step: int,
) -> np.ndarray:
  """Extracts non-overlapping `(m, gh, gw, fw, fw, c)` windows for one group."""
  m, ih, iw, c = x.shape
  bottom, right = top + gh * pix_step, left + gw * pix_step
  region = x[:, top : min(bottom, ih), left : min(right, iw), :]
  pad_h, pad_w = max(0, bottom - ih), max(0, right - iw)
  if pad_h or pad_w:
    region = np.pad(region, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
  blocks = region.reshape(m, gh, pix_step, gw, pix_step, c)
  if pix_step != fw:
    blocks = blocks[:, :, :fw, :, :fw, :]
  return blocks.transpose(0, 1, 3, 2, 4, 5)


def _scatter_group_blocks(
    target: np.ndarray,
    d_blocks: np.ndarray,
    top: int,
    left: int,
    fw: int,
    pix_step: int,
) -> None:
  """Accumulates `(m, gh, gw, fw, fw, c)` gradients into `target` in place."""
  m, ih, iw, c = target.shape
  gh, gw = d_blocks.shape[1], d_blocks.shape[2]
  if pix_step != fw:
    gap = pix_step - fw
    d_blocks = np.pad(
        d_blocks, ((0, 0), (0, 0), (0, 0), (0, gap), (0, gap), (0, 0))
    )
  d_region = d_blocks.transpose(0, 1, 3, 2, 4, 5).reshape(
      m, gh * pix_step, gw * pix_step, c
  )
  h_avail = min(gh * pix_step, ih - top)
  w_avail = min(gw * pix_step, iw - left)
  target[:, top : top + h_avail, left : left + w_avail, :] += d_region[
      :, :h_avail, :w_avail, :
  ]


def _split_stride_groups(
    x: np.ndarray, fw: int, stride: int
) -> tuple[int, int, int, list[list[np.ndarray]]]:
  """Extracts all `(n_rows, n_cols)` non-overlapping stride-group blocks."""
  _, ih, iw, _ = x.shape
  z_h, z_w, g_step, pix_step, n_rows, n_cols = _stride_group_geometry(
      ih, iw, fw, stride
  )
  groups = [
      [
          _extract_group_blocks(
              x,
              t * stride,
              i * stride,
              (z_h - 1 - t) // g_step + 1,
              (z_w - 1 - i) // g_step + 1,
              fw,
              pix_step,
          )
          for i in range(n_cols)
      ]
      for t in range(n_rows)
  ]
  return z_h, z_w, g_step, groups


def conv_forward_sg(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    *,
    stride: int = 1,
    padding: int = 0,
) -> tuple[np.ndarray, _SgCache]:
  """2D NHWC convolution + ReLU forward pass."""
  x_arr = np.asarray(x)
  x_pad = (
      np.pad(x_arr, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
      if padding > 0
      else x_arr
  )
  z_h, z_w, g_step, groups = _split_stride_groups(x_pad, w.shape[0], stride)
  z = np.empty(
      (x_pad.shape[0], z_h, z_w, w.shape[3]),
      dtype=np.result_type(x_pad.dtype, w.dtype),
  )
  for t, row in enumerate(groups):
    for i, blocks in enumerate(row):
      z[:, t::g_step, i::g_step, :] = np.tensordot(
          blocks, w, axes=([3, 4, 5], [0, 1, 2])
      )
  z = z + b
  return np.maximum(0.0, z), (x_pad.shape, groups, z)


def conv_backward_sg(
    d_a: np.ndarray,
    w: np.ndarray,
    cache: _SgCache,
    *,
    stride: int = 1,
    padding: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """2D NHWC convolution + ReLU backward pass."""
  input_shape, groups, z = cache
  d_z = d_a * (z > 0.0)
  fw = w.shape[0]
  _, _, g_step, pix_step, _, _ = _stride_group_geometry(
      input_shape[1], input_shape[2], fw, stride
  )
  d_x_pad = np.zeros(input_shape, dtype=d_z.dtype)
  d_w = np.zeros_like(w, dtype=d_z.dtype)
  d_b = np.sum(d_z, axis=(0, 1, 2), keepdims=True)
  for t, row in enumerate(groups):
    for i, blocks in enumerate(row):
      dz_g = d_z[:, t::g_step, i::g_step, :]
      d_w += np.tensordot(blocks, dz_g, axes=([0, 1, 2], [0, 1, 2]))
      d_blocks = np.tensordot(dz_g, w, axes=([3], [3]))
      _scatter_group_blocks(
          d_x_pad, d_blocks, t * stride, i * stride, fw, pix_step
      )
  d_x = (
      d_x_pad[:, padding:-padding, padding:-padding, :]
      if padding > 0
      else d_x_pad
  )
  return d_x, d_w, d_b


def max_pool_forward_sg(
    x: np.ndarray, *, filter_width: int, stride: int = 1
) -> tuple[np.ndarray, _SgCache]:
  """2D NHWC max-pooling forward pass."""
  x_arr = np.asarray(x)
  z_h, z_w, g_step, groups = _split_stride_groups(x_arr, filter_width, stride)
  z = np.empty((x_arr.shape[0], z_h, z_w, x_arr.shape[3]), dtype=x_arr.dtype)
  for t, row in enumerate(groups):
    for i, blocks in enumerate(row):
      z[:, t::g_step, i::g_step, :] = np.max(blocks, axis=(3, 4))
  return z, (x_arr.shape, groups, z)


def max_pool_backward_sg(
    d_z: np.ndarray,
    cache: _SgCache,
    *,
    filter_width: int,
    stride: int = 1,
) -> np.ndarray:
  """2D NHWC max-pooling backward pass (splitting ties evenly)."""
  input_shape, groups, z = cache
  _, _, g_step, pix_step, _, _ = _stride_group_geometry(
      input_shape[1], input_shape[2], filter_width, stride
  )
  d_x = np.zeros(input_shape, dtype=d_z.dtype)
  for t, row in enumerate(groups):
    for i, blocks in enumerate(row):
      dz_g = d_z[:, t::g_step, i::g_step, :]
      z_g = z[:, t::g_step, i::g_step, :]
      mask = (blocks == z_g[:, :, :, None, None, :]).astype(d_z.dtype)
      mask /= np.maximum(np.sum(mask, axis=(3, 4), keepdims=True), 1.0)
      _scatter_group_blocks(
          d_x,
          dz_g[:, :, :, None, None, :] * mask,
          t * stride,
          i * stride,
          filter_width,
          pix_step,
      )
  return d_x


def _flatten_fwd(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
  return x.reshape((x.shape[0], -1)), x.shape


def _fc_relu_fwd(
    x: np.ndarray, w: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  z = x @ w.T + b
  return np.maximum(0.0, z), z


def _fc_softmax_fwd(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
  z = x @ w.T + b
  exps = np.exp(z - np.max(z, axis=1, keepdims=True))
  return exps / np.sum(exps, axis=1, keepdims=True)


def _fc_bwd(
    d_z: np.ndarray, x: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  return d_z @ w, d_z.T @ x, np.sum(d_z, axis=0)


def _fc_relu_bwd(
    d_a: np.ndarray, x: np.ndarray, z: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  return _fc_bwd(d_a * (z > 0.0), x, w)


def _loss_and_grad(
    probs: np.ndarray, y: np.ndarray
) -> tuple[float, float, np.ndarray]:
  y_arr = np.asarray(y, dtype=probs.dtype)
  clipped = np.clip(probs, 1e-15, 1.0)
  cce = float(-np.sum(y_arr * np.log(clipped)) / probs.shape[0])
  ce = float(np.mean(np.argmax(probs, axis=1) != np.argmax(y_arr, axis=1)))
  return cce, ce, (probs - y_arr) / probs.shape[0]


# ---------------------------------------------------------------------------
# Encapsulated CNN model & stream training graph
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Cnn:
  """Stride-group NumPy MNIST CNN holding hyperparameters and trained state."""

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
    """Initializes and stores He-normal `float32` NumPy parameters."""
    ih, iw, c_in = input_shape
    ch = (ih - self.filter_width) // self.conv_stride + 1
    cw = (iw - self.filter_width) // self.conv_stride + 1
    ph = (ch - self.pool_width) // self.pool_stride + 1
    pw = (cw - self.pool_width) // self.pool_stride + 1
    if ch < 1 or cw < 1 or ph < 1 or pw < 1:
      raise ValueError(f'Input shape {input_shape} is too small for {self}.')
    flat_dim = ph * pw * self.num_filters
    rng = np.random.default_rng(self.seed)
    fw = self.filter_width
    self.params = CnnParams(
        conv_w=rng.normal(
            0.0,
            math.sqrt(2.0 / (fw * fw * c_in)),
            size=(fw, fw, c_in, self.num_filters),
        ).astype(np.float32),
        conv_b=np.zeros((1, 1, 1, self.num_filters), dtype=np.float32),
        fc1_w=rng.normal(
            0.0, math.sqrt(2.0 / flat_dim), size=(self.hidden_dim, flat_dim)
        ).astype(np.float32),
        fc1_b=np.zeros((self.hidden_dim,), dtype=np.float32),
        fc2_w=rng.normal(
            0.0,
            math.sqrt(2.0 / self.hidden_dim),
            size=(self.num_classes, self.hidden_dim),
        ).astype(np.float32),
        fc2_b=np.zeros((self.num_classes,), dtype=np.float32),
    )
    return self.params

  def _conv_fwd(
      self, x: np.ndarray, w: np.ndarray, b: np.ndarray
  ) -> tuple[np.ndarray, _SgCache]:
    return conv_forward_sg(x, w, b, stride=self.conv_stride)

  def _conv_bwd(
      self, d_a: np.ndarray, w: np.ndarray, cache: _SgCache
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return conv_backward_sg(d_a, w, cache, stride=self.conv_stride)

  def _pool_fwd(self, x: np.ndarray) -> tuple[np.ndarray, _SgCache]:
    return max_pool_forward_sg(
        x, filter_width=self.pool_width, stride=self.pool_stride
    )

  def _pool_bwd(self, d_z: np.ndarray, cache: _SgCache) -> np.ndarray:
    return max_pool_backward_sg(
        d_z, cache, filter_width=self.pool_width, stride=self.pool_stride
    )

  def predict(self, x: np.ndarray) -> np.ndarray:
    """Runs an eager forward pass using `self.params`."""
    p = self._require_params()
    a, _ = self._conv_fwd(x, p.conv_w, p.conv_b)
    a, _ = self._pool_fwd(a)
    h, _ = _fc_relu_fwd(a.reshape((a.shape[0], -1)), p.fc1_w, p.fc1_b)
    return _fc_softmax_fwd(h, p.fc2_w, p.fc2_b)

  def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Evaluates `(cce, ce)` on `(x, y)` using `self.params`."""
    cce, ce, _ = _loss_and_grad(self.predict(x), y)
    return cce, ce

  def _sgd_step(self, params: CnnParams, *grads: np.ndarray) -> CnnParams:
    lr = np.float32(self.learning_rate)
    return CnnParams(*(p - lr * g for p, g in zip(params, grads, strict=True)))

  def training_streams(
      self, x_batch: Stream, y_batch: Stream
  ) -> tuple[Stream, Stream, Stream, Stream]:
    """Builds `(params, next_params, cce, ce)` streams with `fby` feedback."""
    params = fby(self._require_params(), lambda: next_params, name='params')
    conv_w, conv_b, fc1_w, fc1_b, fc2_w, fc2_b = pnp.lift(
        tuple, pure=True, nout=6
    )(params)

    a1, c1 = pnp.lift(self._conv_fwd, pure=True, nout=2)(
        x_batch, conv_w, conv_b
    )
    a2, c2 = pnp.lift(self._pool_fwd, pure=True, nout=2)(a1)
    flat, shape = pnp.lift(_flatten_fwd, pure=True, nout=2)(a2)
    h1, z1 = pnp.lift(_fc_relu_fwd, pure=True, nout=2)(flat, fc1_w, fc1_b)
    probs = pnp.lift(_fc_softmax_fwd, pure=True)(h1, fc2_w, fc2_b)

    cce, ce, dz2 = pnp.lift(_loss_and_grad, pure=True, nout=3)(probs, y_batch)

    dh1, dw_fc2, db_fc2 = pnp.lift(_fc_bwd, pure=True, nout=3)(dz2, h1, fc2_w)
    dflat, dw_fc1, db_fc1 = pnp.lift(_fc_relu_bwd, pure=True, nout=3)(
        dh1, flat, z1, fc1_w
    )
    da2 = pnp.lift(np.reshape, pure=True)(dflat, shape)
    da1 = pnp.lift(self._pool_bwd, pure=True)(da2, c2)
    _, dw_conv, db_conv = pnp.lift(self._conv_bwd, pure=True, nout=3)(
        da1, conv_w, c1
    )

    next_params = pnp.lift(self._sgd_step, pure=True)(
        params, dw_conv, db_conv, dw_fc1, db_fc1, dw_fc2, db_fc2
    )
    return params, next_params, cce, ce

  def build_graph(
      self, x_batches: Sequence[np.ndarray], y_batches: Sequence[np.ndarray]
  ) -> Graph:
    """Builds a finalized Psyre `Graph` with periodic mini-batch streams."""
    xs = [np.asarray(b) for b in x_batches]
    ys = [np.asarray(b) for b in y_batches]
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
      train_set: tuple[np.ndarray, np.ndarray],
      test_set: tuple[np.ndarray, np.ndarray] | None = None,
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
    dataset: tuple[np.ndarray, np.ndarray], num_batches: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Splits `(x, y)` into non-empty contiguous NumPy mini-batches."""
  if num_batches < 1:
    raise ValueError(f'num_batches must be >= 1, got {num_batches}.')
  x_arr, y_arr = np.asarray(dataset[0]), np.asarray(dataset[1])
  if len(x_arr) == 0 or len(x_arr) != len(y_arr):
    raise ValueError('x and y must be non-empty with equal sample counts.')
  xs = [b for b in np.array_split(x_arr, num_batches) if len(b) > 0]
  ys = [b for b in np.array_split(y_arr, num_batches) if len(b) > 0]
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
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
  """Loads local MNIST `.gz` IDX archives as `(train_set, test_set)`."""
  root = os.fspath(data_dir)

  def _images(name: str) -> np.ndarray:
    dims, buf = _read_idx(os.path.join(root, name), 3)
    imgs = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    return (imgs / np.float32(255.0)).reshape((*dims, 1))

  def _labels(name: str) -> np.ndarray:
    _, buf = _read_idx(os.path.join(root, name), 1)
    return np.frombuffer(buf, dtype=np.uint8)

  tr_x = _images('train-images-idx3-ubyte.gz')
  te_x = _images('t10k-images-idx3-ubyte.gz')
  tr_y = _labels('train-labels-idx1-ubyte.gz')
  te_y = _labels('t10k-labels-idx1-ubyte.gz')
  n_cls = max(int(np.max(tr_y)), int(np.max(te_y))) + 1
  eye = np.eye(n_cls, dtype=np.float32)
  return (tr_x, eye[tr_y]), (te_x, eye[te_y])
