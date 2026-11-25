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
"""Optional JAX stream-lifting module (`psyre.jax`).

Install JAX support (including `psyre.jax.lax`) via ``pip install
"psyre[jax]"``. Exposes lifted `jax.lax` (`lax`), `jax.numpy` (`jnp` / `numpy`),
`jax.nn` (`nn`), `jax.random` (`random`), `jax.scipy` (`scipy`), explicit
backend conversion adapters (`to_jax`, `to_numpy`), PyTree bridging
(`register_jax_pytree`), lifting-time transformations (`vmap`, `stack_streams`,
`unstack_stream`, `batch_streams`, `rewrite_dtype`), and `jax.jit` fusion /
opt-in `Jaxpr` tracing (`jit`, `jit_graph`, `fuse_jit_islands`, `expand_jaxpr`).

Core `psyre` remains pure stdlib with no JAX or NumPy dependency.
"""

from __future__ import annotations

# pylint: disable=g-importing-member,g-multiple-import
from psyre.jax import lax
from psyre.jax.jit import (
    ImpureJaxprError,
    JitIslandInfo,
    expand_jaxpr,
    fuse_jit_islands,
    jit,
    jit_graph,
)
from psyre.jax.ops import (
    get_primitive,
    is_jax_step_func,
    lift,
    nn,
    numpy,
    random,
    register_jax_pytree,
    scipy,
    to_jax,
    to_numpy,
)
from psyre.jax.transforms import (
    BatchError,
    batch_streams,
    rewrite_dtype,
    stack_streams,
    unstack_stream,
    vmap,
)
from psyre.payload import LiftedNamespace

# pylint: enable=g-importing-member,g-multiple-import

# Alias matching canonical `import jax.numpy as jnp` convention.
jnp = numpy

__all__ = (
    'BatchError',
    'ImpureJaxprError',
    'JitIslandInfo',
    'LiftedNamespace',
    'batch_streams',
    'expand_jaxpr',
    'fuse_jit_islands',
    'get_primitive',
    'is_jax_step_func',
    'jit',
    'jit_graph',
    'jnp',
    'lax',
    'lift',
    'nn',
    'numpy',
    'random',
    'register_jax_pytree',
    'rewrite_dtype',
    'scipy',
    'stack_streams',
    'to_jax',
    'to_numpy',
    'unstack_stream',
    'vmap',
)
