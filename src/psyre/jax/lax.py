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
"""Optional JAX `lax` stream-lifting module (`psyre.jax.lax`).

Lifts pure `jax.lax` and `jax.lax.linalg` primitives onto Psyre `Stream`s with
canonical `Primitive` metadata annotations (`get_primitive(node)`). Marks
higher-order intra-tick `lax` control-flow callables (`scan`, `cond`,
`while_loop`, `fori_loop`, `switch`, `reduce`, `reduce_window`,
`associative_scan`, `map`, `select_and_scatter`) as static operator parameters
under Psyre's strict first-order stream discipline.
"""

from __future__ import annotations

from collections.abc import Mapping
import types
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  from jax import lax as _jax_lax
except ImportError as err:  # pragma: no cover
  raise ImportError(
      'psyre.jax.lax requires JAX. Install it with: pip install "psyre[jax]"'
  ) from err

# pylint: disable=g-importing-member,g-multiple-import
from psyre.jax.ops import (
    PRIMITIVE_TYPE,
    get_primitive,
    lift,
    make_jax_namespace,
)
from psyre.payload import NoutSpec

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import

# Static callable parameter specs for higher-order intra-tick `jax.lax` ops.
# Because Psyre streams are strictly first-order, these callable arguments are
# unambiguously static operator parameters (not dynamic stream values).
_HIGHER_ORDER_SPECS: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] = {
    'associative_scan': ((0,), ('fn',)),
    'cond': ((1, 2), ('true_fun', 'false_fun')),
    'fori_loop': ((2,), ('body_fun',)),
    'map': ((0,), ('f',)),
    'reduce': ((2,), ('computation',)),
    'reduce_window': ((2,), ('computation',)),
    'scan': ((0,), ('f',)),
    'select_and_scatter': ((1, 7), ('select', 'scatter')),
    'switch': ((1,), ('branches',)),
    'while_loop': ((0, 1), ('cond_fun', 'body_fun')),
}


def _lax_qr_nout(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
  del args
  return 3 if bool(kwargs.get('pivoting', False)) else 2


def _lax_svd_nout(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
  del args
  return 3 if bool(kwargs.get('compute_uv', True)) else 1


def _lax_eig_nout(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
  del args
  left = bool(kwargs.get('compute_left_eigenvectors', True))
  right = bool(kwargs.get('compute_right_eigenvectors', True))
  return 1 + int(left) + int(right)


def _lax_schur_nout(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
  del args
  return 2 if bool(kwargs.get('compute_schur_vectors', True)) else 1


_LAX_MULTI_OUTPUT: dict[str, NoutSpec] = {
    'approx_max_k': 2,
    'approx_min_k': 2,
    'rng_bit_generator': 2,
    'scan': 2,
    'sort_key_val': 2,
    'top_k': 2,
}

_LAX_LINALG_MULTI_OUTPUT: dict[str, NoutSpec] = {
    'eig': _lax_eig_nout,
    'eigh': 2,
    'hessenberg': 2,
    'lu': 3,
    'qr': _lax_qr_nout,
    'schur': _lax_schur_nout,
    'svd': _lax_svd_nout,
    'tridiagonal': 4,
}

# Explicit mappings for `jax.lax` ops whose `*_p` primitive name differs from
# `f'{api_name}_p'`. All other 1-to-1 `foo` / `foo_p` pairs are discovered from
# the module's `__dict__` without touching deprecated `__getattr__` entries.
_LAX_PRIM_OVERRIDES: dict[str, str] = {
    'approx_max_k': 'approx_top_k_p',
    'approx_min_k': 'approx_top_k_p',
    'bitwise_and': 'and_p',
    'bitwise_not': 'not_p',
    'bitwise_or': 'or_p',
    'bitwise_xor': 'xor_p',
    'sort_key_val': 'sort_p',
    'while_loop': 'while_p',
}


def _discover_primitives(
    source_module: Any,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Discovers public `api_name -> Primitive` pairs exported on `source_module`."""
  mod_dict = getattr(source_module, '__dict__', {})
  resolved: dict[str, Any] = {}
  for attr_name, val in mod_dict.items():
    if (
        attr_name.endswith('_p')
        and not attr_name.startswith('_')
        and len(attr_name) > 2
        and isinstance(val, PRIMITIVE_TYPE)
    ):
      api_name = attr_name[:-2]
      if api_name in mod_dict:
        resolved[api_name] = val
  if overrides:
    for api_name, prim_name in overrides.items():
      prim = mod_dict.get(prim_name)
      if prim is None:
        prim = getattr(source_module, prim_name, None)
      if isinstance(prim, PRIMITIVE_TYPE):
        resolved[api_name] = prim
  return resolved


_LAX_PRIMITIVES = _discover_primitives(_jax_lax, _LAX_PRIM_OVERRIDES)
_LAX_LINALG_PRIMITIVES = _discover_primitives(_jax_lax.linalg)

linalg = make_jax_namespace(
    _jax_lax.linalg,
    'psyre.jax.lax.linalg',
    primitives=_LAX_LINALG_PRIMITIVES,
    multi_output=_LAX_LINALG_MULTI_OUTPUT,
)

_ROOT = make_jax_namespace(
    _jax_lax,
    'psyre.jax.lax',
    primitives=_LAX_PRIMITIVES,
    multi_output=_LAX_MULTI_OUTPUT,
    higher_order=_HIGHER_ORDER_SPECS,
    submodules={'linalg': linalg},
)

PRIMITIVES: types.MappingProxyType[str, Any] = types.MappingProxyType({
    **_LAX_PRIMITIVES,
    **{f'linalg.{k}': v for k, v in _LAX_LINALG_PRIMITIVES.items()},
})


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
  """Resolves lifted `jax.lax` operations lazily (PEP 562)."""
  return getattr(_ROOT, name)


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  """Returns discoverable attribute names on `psyre.jax.lax`."""
  return sorted(set(__all__) | set(dir(_ROOT)))


__all__ = (
    'PRIMITIVES',
    'get_primitive',
    'lift',
    'linalg',
)
