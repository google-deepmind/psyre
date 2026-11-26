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
"""Illustrative tests for the stride-group NumPy CNN stream example."""

import gzip
import os
import tempfile

from absl.testing import absltest

# pylint: disable=g-import-not-at-top,g-importing-member,g-multiple-import
try:
  import numpy as np
except ImportError:  # pragma: no cover
  np = None

if np is not None:
  from psyre.numpy.examples import cnn_mnist
else:  # pragma: no cover
  cnn_mnist = None

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import


def _finite_diff(fn, arr: np.ndarray, eps: float = 1e-5) -> np.ndarray:
  grad = np.zeros_like(arr, dtype=np.float64)
  flat = arr.astype(np.float64).copy()
  it = np.nditer(flat, flags=['multi_index'])
  while not it.finished:
    idx = it.multi_index
    orig = flat[idx]
    flat[idx] = orig + eps
    fp = float(fn(flat))
    flat[idx] = orig - eps
    fm = float(fn(flat))
    flat[idx] = orig
    grad[idx] = (fp - fm) / (2.0 * eps)
    it.iternext()
  return grad


def _write_idx_gz(path: str, magic: int, dims: tuple[int, ...], data: bytes):
  with gzip.open(path, 'wb') as f:
    f.write(magic.to_bytes(4, 'big'))
    for d in dims:
      f.write(d.to_bytes(4, 'big'))
    f.write(data)


def _write_quadrant_mnist(root: str, n_train: int = 36, n_test: int = 12):
  """Writes small separable 8x8 4-class IDX `.gz` archives for offline tests."""
  rng = np.random.default_rng(3)
  quads = ((0, 0), (0, 4), (4, 0), (4, 4))

  def _make(n: int) -> tuple[np.ndarray, np.ndarray]:
    labels = (np.arange(n, dtype=np.uint8) % 4).copy()
    rng.shuffle(labels)
    imgs = rng.integers(0, 20, size=(n, 8, 8), dtype=np.uint8)
    for i, cls in enumerate(labels):
      r, c = quads[int(cls)]
      imgs[i, r : r + 4, c : c + 4] = 240
    return imgs, labels

  tr_x, tr_y = _make(n_train)
  te_x, te_y = _make(n_test)
  _write_idx_gz(
      os.path.join(root, 'train-images-idx3-ubyte.gz'),
      2051,
      tr_x.shape,
      tr_x.tobytes(),
  )
  _write_idx_gz(
      os.path.join(root, 't10k-images-idx3-ubyte.gz'),
      2051,
      te_x.shape,
      te_x.tobytes(),
  )
  _write_idx_gz(
      os.path.join(root, 'train-labels-idx1-ubyte.gz'),
      2049,
      tr_y.shape,
      tr_y.tobytes(),
  )
  _write_idx_gz(
      os.path.join(root, 't10k-labels-idx1-ubyte.gz'),
      2049,
      te_y.shape,
      te_y.tobytes(),
  )


@absltest.skipIf(np is None, 'numpy is not installed')
class NumpyCnnMnistTest(absltest.TestCase):
  """Tests stride-group NumPy kernels and end-to-end MNIST stream training."""

  def test_stride_group_conv_and_pool_gradients(self):
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=(2, 6, 6, 2))
    w = rng.normal(0.0, 0.5, size=(3, 3, 2, 3))
    b = rng.normal(0.0, 0.1, size=(1, 1, 1, 3))
    a, c_cache = cnn_mnist.conv_forward_sg(x, w, b, stride=2, padding=1)
    up_c = rng.normal(0.0, 1.0, size=a.shape)
    dx, dw, db = cnn_mnist.conv_backward_sg(
        up_c, w, c_cache, stride=2, padding=1
    )
    np.testing.assert_allclose(
        dx,
        _finite_diff(
            lambda v: np.sum(
                cnn_mnist.conv_forward_sg(v, w, b, stride=2, padding=1)[0]
                * up_c
            ),
            x,
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        dw,
        _finite_diff(
            lambda v: np.sum(
                cnn_mnist.conv_forward_sg(x, v, b, stride=2, padding=1)[0]
                * up_c
            ),
            w,
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        db,
        _finite_diff(
            lambda v: np.sum(
                cnn_mnist.conv_forward_sg(x, w, v, stride=2, padding=1)[0]
                * up_c
            ),
            b,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    z, p_cache = cnn_mnist.max_pool_forward_sg(x, filter_width=2, stride=2)
    up_p = rng.normal(0.0, 1.0, size=z.shape)
    dx_p = cnn_mnist.max_pool_backward_sg(
        up_p, p_cache, filter_width=2, stride=2
    )
    np.testing.assert_allclose(
        dx_p,
        _finite_diff(
            lambda v: np.sum(
                cnn_mnist.max_pool_forward_sg(v, filter_width=2, stride=2)[0]
                * up_p
            ),
            x,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

  def test_train_and_infer_mnist(self):
    with tempfile.TemporaryDirectory() as tmp:
      _write_quadrant_mnist(tmp)
      cnn = cnn_mnist.Cnn(
          num_filters=4,
          filter_width=2,
          conv_stride=1,
          pool_width=2,
          pool_stride=2,
          hidden_dim=16,
          num_classes=4,
          learning_rate=0.15,
          num_epochs=15,
          mini_batches=3,
          seed=5,
      )
      cnn.train_mnist(tmp)
      self.assertLen(cnn.cce_history, 45)
      self.assertLess(cnn.cce_history[-1], cnn.cce_history[0] * 0.25)
      self.assertLess(cnn.test_ce, 0.15)
      _, (te_x, te_y) = cnn_mnist.load_mnist_idx(tmp)
      self.assertEqual(cnn.predict(te_x).shape, (12, 4))
      cce, ce = cnn.evaluate(te_x, te_y)
      self.assertAlmostEqual(cce, cnn.test_cce)
      self.assertAlmostEqual(ce, cnn.test_ce)

  def test_minibatch_streams_cycle_across_epochs(self):
    x = np.stack([
        np.full((8, 8, 1), 0.1, dtype=np.float32),
        np.full((8, 8, 1), 0.5, dtype=np.float32),
        np.full((8, 8, 1), 0.9, dtype=np.float32),
    ])
    y = np.eye(4, dtype=np.float32)[:3]
    cnn = cnn_mnist.Cnn(
        num_filters=4,
        filter_width=2,
        pool_width=2,
        pool_stride=2,
        hidden_dim=16,
        num_classes=4,
        learning_rate=0.0,
        num_epochs=2,
        mini_batches=3,
    )
    cnn.train((x, y))
    self.assertNotAlmostEqual(cnn.cce_history[0], cnn.cce_history[2], places=4)
    np.testing.assert_allclose(cnn.cce_history[:3], cnn.cce_history[3:])


if __name__ == '__main__':
  absltest.main()
