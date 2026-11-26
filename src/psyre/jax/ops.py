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
"""Shared stream-lifting machinery for optional `psyre.jax` (`psyre.jax.lax`)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import functools
import operator
import types
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  import jax
  from jax import lax as _jax_lax
  import jax.numpy as jnp
  import numpy as np
except ImportError as err:  # pragma: no cover
  raise ImportError(
      'psyre.jax requires JAX. Install it with: pip install "psyre[jax]"'
  ) from err

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    emit_comb_node,
    make_comb_step_func,
    unwrap_inst_func,
    validate_positive_int,
)
from psyre.payload import (
    ARRAY_ELEMWISE_NOUT,
    ARRAY_LINALG_NOUT,
    LiftedNamespace,
    NoutSpec,
    is_callable_target,
    is_from_package,
    normalize_static_argnames,
    normalize_static_argnums,
    partition_comb_args,
    register_pytree_node,
    reject_out_kwarg,
    resolve_nout,
    strip_null_out,
)

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import

BACKEND_ATTR = '_psyre_jax_backend'
PRIMITIVE_ATTR = '_psyre_jax_primitive'
# Canonical `Primitive` class obtained via `jax.lax.add_p` without relying on
# the restricted `jax.extend` namespace.
PRIMITIVE_TYPE: type[Any] = type(_jax_lax.add_p)

_COERCION_NAMES = ('array', 'asarray', 'asanyarray')
_REJECTED_IMPURE_NAMES = frozenset({
    'breakpoint',
    'callback',
    'infeed',
    'infeed_p',
    'io_callback',
    'outfeed',
    'outfeed_p',
    'print',
})


def _collect_ids(
    sources: Iterable[tuple[Any, tuple[str, ...]]],
) -> frozenset[int]:
  ids: list[int] = []
  for mod, names in sources:
    if mod is None:
      continue
    for attr_name in names:
      val = getattr(mod, attr_name, None)
      if val is not None:
        ids.append(id(val))
  return frozenset(ids)


_JNP_COERCION_IDS = _collect_ids(((jnp, _COERCION_NAMES),))
_JAX_IMPURE_IDS = _collect_ids((
    (getattr(jax, 'debug', None), ('breakpoint', 'callback', 'print')),
    (getattr(jax, 'experimental', None), ('io_callback',)),
    (_jax_lax, ('infeed', 'infeed_p', 'outfeed', 'outfeed_p')),
))


def read_wrapper_nout_spec(fn: Any) -> NoutSpec | None:
  """Returns the attached static `nout` int or resolver on `fn`, if any."""
  nout = getattr(fn, '_psyre_nout', None)
  if nout is not None:
    return nout
  return getattr(fn, '_psyre_nout_resolver', None)


def attach_wrapper_meta(
    wrapper: Any,
    *,
    inst_func: Callable[..., Any],
    step_func: Callable[..., Any] | None = None,
    pure: bool = True,
    primitive: Any = None,
    static_argnums: frozenset[int] = frozenset(),
    static_argnames: frozenset[str] = frozenset(),
    nout_spec: NoutSpec | None = None,
) -> Any:
  """Attaches canonical Psyre lifting metadata onto a JAX wrapper callable."""
  if step_func is not None:
    setattr(wrapper, '_psyre_step_func', step_func)
  setattr(wrapper, '_psyre_inst_func', inst_func)
  setattr(wrapper, BACKEND_ATTR, True)
  setattr(wrapper, '_psyre_pure', bool(pure))
  if primitive is not None:
    setattr(wrapper, PRIMITIVE_ATTR, primitive)
  if static_argnums:
    setattr(wrapper, '_psyre_static_argnums', static_argnums)
  if static_argnames:
    setattr(wrapper, '_psyre_static_argnames', static_argnames)
  if isinstance(nout_spec, int):
    setattr(wrapper, '_psyre_nout', nout_spec)
  elif nout_spec is not None:
    setattr(wrapper, '_psyre_nout_resolver', nout_spec)
  return wrapper


def get_primitive(obj: Any) -> Any:
  """Returns the attached JAX `Primitive` metadata on `obj`, or `None`."""
  for holder in (
      obj,
      getattr(obj, 'step_func', None),
      getattr(obj, '_psyre_step_func', None),
  ):
    if holder is not None:
      prim = getattr(holder, PRIMITIVE_ATTR, None)
      if prim is not None:
        return prim
  return None


def is_jax_step_func(obj: Any) -> bool:
  """Returns True when `obj` is a `Node`, wrapper, or `step_func` from JAX."""
  return any(
      bool(holder is not None and getattr(holder, BACKEND_ATTR, False))
      for holder in (
          obj,
          getattr(obj, 'step_func', None),
          getattr(obj, '_psyre_step_func', None),
      )
  )


def make_jax_step_func(
    fn: Callable[..., Any],
    *,
    name: str = '',
    pure: bool = True,
    primitive: Any = None,
) -> Callable[..., Any]:
  """Builds a JAX-tagged combinational `step_func` closure."""
  label = name or getattr(fn, '__name__', 'jax_op')
  step_func = make_comb_step_func(fn, name=label, pure=pure)
  setattr(step_func, BACKEND_ATTR, True)
  if primitive is not None:
    setattr(step_func, PRIMITIVE_ATTR, primitive)
  return step_func


@functools.lru_cache(maxsize=128)
def make_proj_step_func(
    index: int, *, jax_tagged: bool = True
) -> Callable[..., Any]:
  """Returns a cached tuple-projection `step_func` for multi-output nodes."""
  def _proj(bundle: Any) -> Any:
    return operator.getitem(bundle, index)

  label = f'jax_out_{index}' if jax_tagged else f'island_out_{index}'
  _proj.__name__ = label
  if jax_tagged:
    return make_jax_step_func(_proj, name=label, pure=True)
  # Non-JAX-tagged projection keeps multi-exit `fuse_jit_islands` idempotent.
  return make_comb_step_func(_proj, name=label, pure=True)


def emit_jax_comb_node(
    step_func: Callable[..., Any],
    dyn_inputs: tuple[Any, ...],
    node_kwargs: Mapping[str, Any],
    *,
    name: str,
    nout: int = 1,
) -> Any:
  """Emits a JAX combinational node, unpacking `nout > 1` via JAX projections."""
  raw_out = emit_comb_node(
      step_func, dyn_inputs, node_kwargs, name=name, nout=1
  )
  if nout <= 1:
    return raw_out
  return tuple(
      emit_comb_node(
          make_proj_step_func(i, jax_tagged=True),
          (raw_out,),
          {},
          name=f'{name}[{i}]',
      )
      for i in range(nout)
  )


def lift(
    fn: Callable[..., Any],
    *,
    pure: bool = True,
    nout: int | None = None,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    static_literals: bool = True,
    primitive: Any = None,
    name: str = '',
) -> Callable[..., Any]:
  """Lifts a pure JAX/Python callable onto Psyre `Stream`s.

  When invoked with *no* dynamic `Stream` or Stage-1 thunk arguments, executes
  `fn` eagerly and returns the raw JAX value/array so constants stay
  clock-polymorphic across `match`/`automaton` scopes.

  Args:
    fn: Pure JAX/lax callable or `@psyre_comb` wrapper.
    pure: Purity flag for `optimize_graph` CSE (default `True`).
    nout: Optional positive output arity unpacking a tuple output stream into
      an `nout`-tuple of projected `Stream`s.
    static_argnums: Zero-based positional indices kept verbatim as static args.
    static_argnames: Keyword argument names kept verbatim as static args.
    static_literals: When `True` (default), non-stream positional literals stay
      verbatim inside `CombCallPlan` instead of turning into static `Stream`s.
    primitive: Optional JAX `Primitive` associated with `fn`.
    name: Optional Stage-1 node name override.

  Returns:
    A lifted callable accepting plain values, `Stream`s, PyTrees of `Stream`s,
    or Stage-1 thunks.

  Raises:
    TypeError: If `fn` is not callable or `primitive` is invalid.
    ValueError: If `nout` is not a positive integer.
  """
  if nout is not None:
    validate_positive_int(nout, 'psyre.jax.lift nout')
  return lift_with_nout_spec(
      fn,
      pure=pure,
      nout_spec=nout,
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      static_literals=static_literals,
      primitive=primitive,
      name=name,
  )


def lift_with_nout_spec(
    fn: Callable[..., Any],
    *,
    pure: bool = True,
    nout_spec: NoutSpec | None = None,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    static_literals: bool = True,
    primitive: Any = None,
    name: str = '',
) -> Callable[..., Any]:
  """Internal `lift` variant accepting an int or argument-aware `NoutSpec`."""
  if not is_callable_target(fn):
    raise TypeError(f'psyre.jax.lift expects a callable, got {type(fn)}.')
  if primitive is not None and not isinstance(primitive, PRIMITIVE_TYPE):
    raise TypeError(
        f'primitive must be a JAX Primitive or None, got '
        f'{type(primitive).__name__}.'
    )
  raw_fn = unwrap_inst_func(fn)
  if raw_fn is not fn:
    if primitive is None:
      primitive = get_primitive(fn)
    if nout_spec is None:
      nout_spec = read_wrapper_nout_spec(fn)
  argnums = normalize_static_argnums(static_argnums)
  argnames = normalize_static_argnames(static_argnames)
  if raw_fn is not fn:
    argnums = argnums | getattr(fn, '_psyre_static_argnums', frozenset())
    argnames = argnames | getattr(fn, '_psyre_static_argnames', frozenset())
  op_name = name or getattr(raw_fn, '__name__', 'jax_op')
  is_coercion = id(raw_fn) in _JNP_COERCION_IDS
  is_jax_fn = is_from_package(raw_fn, 'jax')
  is_impure_op = id(raw_fn) in _JAX_IMPURE_IDS or (
      is_jax_fn and getattr(raw_fn, '__name__', '') in _REJECTED_IMPURE_NAMES
  )
  step_func = make_jax_step_func(
      raw_fn, name=op_name, pure=bool(pure), primitive=primitive
  )

  @functools.wraps(raw_fn)
  def wrapper(*args: Any, **kwargs: Any) -> Any:
    if is_impure_op:
      raise TypeError(
          f'Impure/IO JAX operation {op_name!r} is not supported on pure '
          'Psyre streams.'
      )
    if is_jax_fn:
      reject_out_kwarg(op_name, kwargs)
      clean_kwargs = strip_null_out(kwargs)
    else:
      clean_kwargs = dict(kwargs)
    dyn_inputs, node_kwargs = partition_comb_args(
        args,
        clean_kwargs,
        static_argnums=argnums,
        static_argnames=argnames,
        static_literals=static_literals,
    )
    if not dyn_inputs:
      return raw_fn(*args, **clean_kwargs)
    if is_coercion:
      raise TypeError(
          f'Cannot convert a Psyre Stream via {op_name}(); use '
          'psyre.jax.to_jax(...) or psyre.jax.stack_streams(...) instead.'
      )
    out_arity = resolve_nout(nout_spec, args, clean_kwargs)
    return emit_jax_comb_node(
        step_func, dyn_inputs, node_kwargs, name=op_name, nout=out_arity
    )

  return attach_wrapper_meta(
      wrapper,
      inst_func=raw_fn,
      step_func=step_func,
      pure=bool(pure),
      primitive=primitive,
      static_argnums=argnums,
      static_argnames=argnames,
      nout_spec=nout_spec,
  )


def make_jax_namespace(
    module: types.ModuleType,
    label: str = '',
    *,
    primitives: Mapping[str, Any] | None = None,
    multi_output: Mapping[str, NoutSpec] | None = None,
    higher_order: Mapping[
        str, tuple[tuple[int, ...], tuple[str, ...]]
    ] | None = None,
    submodules: Mapping[str, LiftedNamespace] | None = None,
    rejected: frozenset[str] = _REJECTED_IMPURE_NAMES,
) -> LiftedNamespace:
  """Constructs a lazy `LiftedNamespace` lifting a JAX module/submodule."""
  prim_map = dict(primitives or {})
  nout_map = dict(multi_output or {})
  ho_map = dict(higher_order or {})

  def _wrap(name: str, attr: Any) -> Any:
    if callable(attr) and not isinstance(attr, type):
      argnums, argnames = ho_map.get(name, ((), ()))
      return lift_with_nout_spec(
          attr,
          pure=True,
          nout_spec=nout_map.get(name),
          static_argnums=argnums,
          static_argnames=argnames,
          primitive=prim_map.get(name),
          name=name,
      )
    return attr

  return LiftedNamespace(
      module,
      label,
      wrap_attr=_wrap,
      submodules=submodules,
      rejected=rejected,
      reject_message=(
          'Impure/IO JAX operation {label}.{name} is not supported on pure '
          'Psyre streams.'
      ),
  )


def _make_converter(
    asarray_fn: Callable[..., Any],
    default_label: str,
    step_factory: Callable[..., Any],
    doc: str,
) -> Callable[..., Any]:
  """Builds an explicit host/device boundary conversion helper."""

  @functools.lru_cache(maxsize=64)
  def _cached_step(dtype: Any) -> Callable[..., Any]:
    def _convert(val: Any) -> Any:
      return asarray_fn(val, dtype=dtype)

    _convert.__name__ = default_label
    return step_factory(_convert, name=default_label, pure=True)

  def _convert_stream(
      value: Any, *, dtype: Any = None, name: str = ''
  ) -> Any:
    dyn_inputs, node_kwargs = partition_comb_args(
        (value,), {}, static_literals=True
    )
    if not dyn_inputs:
      return asarray_fn(value, dtype=dtype)
    label = name or default_label
    return emit_comb_node(
        _cached_step(dtype), dyn_inputs, node_kwargs, name=label
    )

  _convert_stream.__name__ = default_label
  _convert_stream.__doc__ = doc
  return _convert_stream


to_jax = _make_converter(
    jnp.asarray,
    'to_jax',
    make_jax_step_func,
    'Explicitly converts a `Stream` (or eager value) to `jax.Array`.',
)

# Intentionally emits plain (non-JAX-tagged) `make_comb_step_func` so host
# NumPy outputs terminate `jax.jit` compilation islands cleanly.
to_numpy = _make_converter(
    np.asarray,
    'to_numpy',
    make_comb_step_func,
    'Explicitly converts a `Stream` (or eager `jax.Array`) to host '
    '`np.ndarray`.',
)


def register_jax_pytree(node_type: type[Any]) -> type[Any]:
  """Bridges a JAX-registered PyTree class into Psyre's `CombCallPlan` registry.

  Args:
    node_type: Custom container class already registered in `jax.tree_util`.

  Returns:
    `node_type` unchanged (suitable for decorator stacking).

  Raises:
    TypeError: If `node_type` is not a class.
  """
  if not isinstance(node_type, type):
    raise TypeError(
        f'register_jax_pytree expects a class, got {type(node_type).__name__}.'
    )
  if node_type not in (tuple, list, dict):
    def _flatten(val: Any) -> tuple[tuple[Any, ...], Any]:
      children, treedef = jax.tree_util.tree_flatten(
          val, is_leaf=lambda x: x is not val
      )
      if len(children) == 1 and children[0] is val:
        raise TypeError(
            f'Type {type(val).__name__!r} is not registered as a JAX PyTree '
            'node; register it with jax.tree_util before calling '
            'register_jax_pytree().'
        )
      return tuple(children), treedef

    def _unflatten(treedef: Any, children: Iterable[Any]) -> Any:
      return jax.tree_util.tree_unflatten(treedef, children)

    register_pytree_node(node_type, _flatten, _unflatten)
  return node_type


_jnp_linalg = make_jax_namespace(
    jnp.linalg, 'psyre.jax.numpy.linalg', multi_output=ARRAY_LINALG_NOUT
)
_jnp_fft = make_jax_namespace(jnp.fft, 'psyre.jax.numpy.fft')

numpy = make_jax_namespace(
    jnp,
    'psyre.jax.numpy',
    multi_output=ARRAY_ELEMWISE_NOUT,
    submodules={'linalg': _jnp_linalg, 'fft': _jnp_fft},
)
nn = make_jax_namespace(jax.nn, 'psyre.jax.nn')
random = make_jax_namespace(jax.random, 'psyre.jax.random')
scipy = (
    make_jax_namespace(jax.scipy, 'psyre.jax.scipy')
    if hasattr(jax, 'scipy')
    else None
)
