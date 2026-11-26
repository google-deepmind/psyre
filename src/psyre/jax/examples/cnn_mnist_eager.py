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
"""Eager-mode JAX MNIST CNN specialization (no XLA JIT compilation).

Builds on the shared model and stream loop in `cnn_mnist.py`, running each tick
and inference/evaluation call eagerly without `jax.jit` (see `cnn_mnist_jit.py`
for the fused XLA JIT counterpart).
"""

from __future__ import annotations

import dataclasses
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  import jax.numpy as jnp
except ImportError as err:  # pragma: no cover
  raise ImportError(
      'psyre.jax.examples.cnn_mnist_eager requires JAX. Install it with: '
      'pip install "psyre[jax]"'
  ) from err

from psyre import jax as srjax
from psyre.jax.examples import cnn_mnist

# pylint: enable=g-import-not-at-top

CnnParams = cnn_mnist.CnnParams
load_mnist_idx = cnn_mnist.load_mnist_idx
make_minibatches = cnn_mnist.make_minibatches

__all__ = ('Cnn', 'CnnParams', 'load_mnist_idx', 'make_minibatches')


@dataclasses.dataclass
class Cnn(cnn_mnist.Cnn):
  """Eager-mode JAX MNIST CNN holding hyperparameters and trained state."""

  def predict(self, x: Any) -> Any:
    """Returns softmax probabilities eagerly or on a Psyre `Stream`."""
    return srjax.lift(self._probs, pure=True, name='cnn_predict')(
        self._require_params(), x
    )

  def evaluate(self, x: Any, y: Any) -> tuple[float, float]:
    """Eagerly evaluates `(cce, ce)` using `self.params`."""
    cce, ce = self.loss_and_error(
        self._require_params(), jnp.asarray(x), jnp.asarray(y)
    )
    return float(cce), float(ce)
