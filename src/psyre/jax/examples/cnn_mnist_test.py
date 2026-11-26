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
"""Illustrative tests for `cnn_mnist_eager` and `cnn_mnist_jit`."""

import ast
import gzip
import inspect
import os
import tempfile

from absl.testing import absltest

# pylint: disable=g-import-not-at-top,g-importing-member,g-multiple-import
try:
  import jax
  import jax.numpy as jnp
  import numpy as np
except ImportError:  # pragma: no cover
  jax = None
  jnp = None
  np = None

if jax is not None:
  from psyre.jax.examples import cnn_mnist_eager
  from psyre.jax.examples import cnn_mnist_jit
else:  # pragma: no cover
  cnn_mnist_eager = None
  cnn_mnist_jit = None

from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import


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


def _small_kwargs(**overrides):
  base = dict(
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
  base.update(overrides)
  return base


@absltest.skipIf(jax is None, 'jax is not installed')
class JaxCnnMnistTest(absltest.TestCase):
  """Tests standalone eager and JIT JAX CNN examples on MNIST IDX archives."""

  def test_independent_from_numpy_cnn_example(self):
    for mod in (cnn_mnist_eager, cnn_mnist_jit):
      for node in ast.walk(ast.parse(inspect.getsource(mod))):
        if isinstance(node, ast.Import):
          for alias in node.names:
            self.assertFalse(alias.name.startswith('psyre.numpy'))
        elif isinstance(node, ast.ImportFrom) and node.module:
          self.assertFalse(node.module.startswith('psyre.numpy'))

  def test_eager_and_jit_train_and_infer_mnist(self):
    with tempfile.TemporaryDirectory() as tmp:
      _write_quadrant_mnist(tmp)
      eager = cnn_mnist_eager.Cnn(**_small_kwargs())
      jitted = cnn_mnist_jit.Cnn(**_small_kwargs())
      eager.train_mnist(tmp)
      jitted.train_mnist(tmp)

      self.assertLen(jitted.islands, 1)
      self.assertLen(jitted.cce_history, 45)
      self.assertLess(jitted.cce_history[-1], jitted.cce_history[0] * 0.25)
      self.assertLess(jitted.test_ce, 0.15)
      np.testing.assert_allclose(
          jitted.cce_history, eager.cce_history, rtol=1e-4, atol=1e-5
      )
      for p_e, p_j in zip(eager.params, jitted.params, strict=True):
        np.testing.assert_allclose(p_j, p_e, rtol=1e-4, atol=1e-5)

      _, (te_x, _) = cnn_mnist_eager.load_mnist_idx(tmp)
      p_eager = eager.predict(te_x)
      p_jit = jitted.predict(te_x)
      np.testing.assert_allclose(p_jit, p_eager, rtol=1e-5, atol=1e-6)
      stream_p = Scheduler.run(
          lambda: jitted.predict(Stream([te_x])), horizon=1
      )
      np.testing.assert_allclose(stream_p[0], p_eager, rtol=1e-5, atol=1e-6)

  def test_minibatch_streams_cycle_across_epochs(self):
    x = jnp.stack([
        jnp.full((8, 8, 1), 0.1),
        jnp.full((8, 8, 1), 0.5),
        jnp.full((8, 8, 1), 0.9),
    ])
    y = jnp.eye(4)[:3]
    for cls in (cnn_mnist_eager.Cnn, cnn_mnist_jit.Cnn):
      cnn = cls(**_small_kwargs(learning_rate=0.0, num_epochs=2))
      cnn.train((x, y))
      self.assertNotAlmostEqual(
          cnn.cce_history[0], cnn.cce_history[2], places=4
      )
      np.testing.assert_allclose(cnn.cce_history[:3], cnn.cce_history[3:])


if __name__ == '__main__':
  absltest.main()
