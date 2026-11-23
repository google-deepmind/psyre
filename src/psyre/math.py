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
"""Scalar pointwise mathematical stream primitives lifted via `@psyre_comb`.

These wrap Python `math.*` functions for scalar numeric streams; for dense
NumPy or JAX array streams, use `np.exp`/`np.tanh` (auto-lifted via NEP-13) or
`@psyre_comb`-wrapped `jnp.*` operations instead.
"""

import math
import random

from psyre.common import (  # pylint: disable=g-importing-member,g-multiple-import
    mark_builtin_scalar_op,
    psyre_comb,
)


@psyre_comb(pure=True)
def exp(v):
  """Pointwise exponential stream function."""
  return math.exp(v)


@psyre_comb
def random_stream():
  """Returns a random float in [0.0, 1.0) at each stream step."""
  return random.random()


@psyre_comb(pure=True)
def tanh(v):
  """Pointwise hyperbolic tangent stream function."""
  return math.tanh(v)


mark_builtin_scalar_op(exp)
mark_builtin_scalar_op(random_stream)
mark_builtin_scalar_op(tanh)
