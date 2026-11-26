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
"""JIT-compiled JAX MNIST CNN specialization using `psyre.jax.fuse_jit_islands`.

Builds on the shared model and stream loop in `cnn_mnist.py`:

* `build_graph` fuses the per-tick combinational step (`value_and_grad` +
  projections + SGD) into one `jax.jit` super-node bounded by the `fby`
  parameter feedback stream.
* `predict` and `evaluate` run through `psyre.jax.jit`.
* See `cnn_mnist_eager.py` for the uncompiled eager per-tick counterpart.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  import jax.numpy as jnp
except ImportError as err:  # pragma: no cover
  raise ImportError(
      'psyre.jax.examples.cnn_mnist_jit requires JAX. Install it with: '
      'pip install "psyre[jax]"'
  ) from err

# pylint: disable=g-importing-member
from psyre import jax as srjax
from psyre.graph import Graph
from psyre.jax.examples import cnn_mnist

# pylint: enable=g-import-not-at-top,g-importing-member

CnnParams = cnn_mnist.CnnParams
load_mnist_idx = cnn_mnist.load_mnist_idx
make_minibatches = cnn_mnist.make_minibatches

__all__ = ('Cnn', 'CnnParams', 'load_mnist_idx', 'make_minibatches')


@dataclasses.dataclass
class Cnn(cnn_mnist.Cnn):
  """JIT-compiled JAX MNIST CNN holding hyperparameters and trained state."""

  islands: tuple[srjax.JitIslandInfo, ...] = dataclasses.field(
      default=(), init=False, repr=False
  )

  def predict(self, x: Any) -> Any:
    """Returns XLA-compiled softmax probabilities (array or `Stream`)."""
    return srjax.jit(self._probs, name='cnn_predict')(self._require_params(), x)

  def evaluate(self, x: Any, y: Any) -> tuple[float, float]:
    """Evaluates `(cce, ce)` through `jax.jit` using `self.params`."""
    cce, ce = srjax.jit(self.loss_and_error, name='cnn_eval')(
        self._require_params(), jnp.asarray(x), jnp.asarray(y)
    )
    return float(cce), float(ce)

  def build_graph(
      self, x_batches: Sequence[Any], y_batches: Sequence[Any]
  ) -> Graph:
    """Builds the training `Graph` and fuses the step into a `jax.jit` node."""
    fused, self.islands = srjax.fuse_jit_islands(
        super().build_graph(x_batches, y_batches)
    )
    return fused
