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
"""First-order payload, array/scalar inspection, PyTree, and call-plan helpers.

Dependency-free leaf module providing object-, scalar-, array-, and
container-level machinery used across core `psyre` without importing `numpy` or
`jax` (array inspection reads only `.shape` and `.dtype` metadata so JAX device
buffers are never transferred to the host):

* Scalar/array predicates and NumPy NEP-13/NEP-18 non-mutability/purity guards
  (`is_array_object`, `is_numpy_bool`, `is_numpy_scalar`, `is_nan_scalar`,
  `is_builtin_scalar_value`, `reject_numpy_inplace_ufunc`,
  `reject_numpy_mutating_func`, `is_pure_numpy_func`).
* `reject_callable`: enforces strict first-order stream payloads (rejecting any
  callable directly or inside nested containers, with cycle protection).
* `register_pytree_node` / `unregister_pytree_node`: extensible PyTree registry
  matching the `(flatten_fn, unflatten_fn)` contract of `jax.tree_util`.
* `CombCallPlan` and `partition_comb_args`: splits nested dynamic `Stream` /
  thunk leaves from static arguments and reconstructs `(args, kwargs)` per tick.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import enum
import functools
import inspect
import itertools
import math
import types
from typing import Any

STREAM_MARKER_ATTR = '_psyre_is_stream'
SENTINEL_MARKER_ATTR = '_psyre_is_sentinel'
LAZY_THUNK_MARKER_ATTR = '_psyre_is_lazy_thunk'

BUILTIN_SCALAR_TYPES = frozenset({
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
})

_ENUM_STANDARD_EQ = (object.__eq__, int.__eq__, str.__eq__, enum.Enum.__eq__)
_ENUM_STANDARD_NE = (object.__ne__, int.__ne__, str.__ne__, enum.Enum.__ne__)
_ENUM_STANDARD_LT = (object.__lt__, int.__lt__, str.__lt__)
_ENUM_STANDARD_LE = (object.__le__, int.__le__, str.__le__)
_ENUM_STANDARD_GT = (object.__gt__, int.__gt__, str.__gt__)
_ENUM_STANDARD_GE = (object.__ge__, int.__ge__, str.__ge__)

NUMPY_UFUNC_METHODS = frozenset({
    '__call__',
    'reduce',
    'accumulate',
    'reduceat',
    'outer',
})

MUTATING_NUMPY_FUNCS = frozenset({
    'copyto',
    'fill_diagonal',
    'place',
    'put',
    'put_along_axis',
    'putmask',
})

_IMPURE_NUMPY_FUNC_NAMES = frozenset({
    'empty',
    'empty_like',
    'fromfile',
    'genfromtxt',
    'load',
    'loadtxt',
    'memmap',
    'save',
    'savetxt',
    'savez',
    'savez_compressed',
    'tofile',
})

_NON_SYNC_FUNC_FLAGS = (
    inspect.CO_VARARGS
    | inspect.CO_VARKEYWORDS
    | inspect.CO_GENERATOR
    | inspect.CO_COROUTINE
    | inspect.CO_ITERABLE_COROUTINE
    | inspect.CO_ASYNC_GENERATOR
)

CALL_PLAN_KWARG = '_psyre_call_plan'

PyTreeFlattenFn = Callable[[Any], tuple[Sequence[Any], Any]]
PyTreeUnflattenFn = Callable[[Any, Sequence[Any]], Any]

_pytree_registry: dict[
    type[Any], tuple[PyTreeFlattenFn, PyTreeUnflattenFn]
] = {}


def is_stream_obj(value: Any) -> bool:
  """Returns True when `value` is a Psyre `Stream` (avoids circular import)."""
  return bool(getattr(type(value), STREAM_MARKER_ATTR, False))


def is_sentinel_obj(value: Any) -> bool:
  """Returns True when `value` is an internal sentinel (`Absent`/`Bottom`)."""
  return bool(getattr(type(value), SENTINEL_MARKER_ATTR, False))


def is_lazy_thunk_obj(value: Any) -> bool:
  """Returns True if `value` is a `LazyThunk` proxy (no circular import)."""
  return bool(getattr(type(value), LAZY_THUNK_MARKER_ATTR, False))


def is_callable_target(value: Any) -> bool:
  """Returns True when `value` is callable and not a deferred `LazyThunk`."""
  return callable(value) and not is_lazy_thunk_obj(value)


def zero_sign_key(value: Any) -> Any:
  """Returns a hashable signed-zero discriminator for float/complex scalars."""
  if isinstance(value, float):
    return math.copysign(1.0, value) if value == 0.0 else 0.0
  if isinstance(value, complex):
    return (
        math.copysign(1.0, value.real) if value.real == 0.0 else 0.0,
        math.copysign(1.0, value.imag) if value.imag == 0.0 else 0.0,
    )
  if is_numpy_scalar(value):
    py_val = value.item()
    if isinstance(py_val, (float, complex)):
      return zero_sign_key(py_val)
  return 0.0


def exact_value_key(value: Any) -> Any | None:
  """Returns a hashable exact key separating types and signed zeros, or None."""
  if (
      is_nan_scalar(value)
      or is_array_object(value)
      or is_stream_obj(value)
      or is_lazy_thunk_obj(value)
  ):
    return None
  cls = type(value)
  if isinstance(value, tuple):
    parts = tuple(exact_value_key(item) for item in value)
    if any(p is None for p in parts):
      return None
    return (cls, parts)
  if isinstance(value, frozenset):
    items = [exact_value_key(item) for item in value]
    if any(p is None for p in items):
      return None
    try:
      return (cls, frozenset(items))
    except (TypeError, ValueError, RuntimeError):
      return None
  key = (cls, value, zero_sign_key(value))
  try:
    hash(key)
  except (TypeError, ValueError, RuntimeError):
    return None
  return key


def ufunc_nout(ufunc: Any, method: str) -> int:
  """Returns the output tuple arity for a NumPy ufunc method dispatch."""
  return getattr(ufunc, 'nout', 1) if method in ('__call__', 'outer') else 1


def is_numpy_bool(value: Any) -> bool:
  """Returns True for NumPy scalar booleans (`np.bool_`) without importing."""
  cls = type(value)
  return cls.__module__ == 'numpy' and cls.__name__ in {'bool_', 'bool'}


def is_numpy_bool_array_scalar(value: Any) -> bool:
  """Returns True for 0-D boolean `np.ndarray` without importing `numpy`."""
  return (
      is_array_object(value)
      and not value.shape
      and type(value).__module__ == 'numpy'
      and getattr(value.dtype, 'kind', None) == 'b'
  )


def is_numpy_scalar(value: Any) -> bool:
  """Returns True for NumPy scalar values (`np.generic`) without importing."""
  cls = type(value)
  if cls in BUILTIN_SCALAR_TYPES:
    return False
  return (
      cls.__module__ == 'numpy'
      and cls.__name__ != 'ndarray'
      and hasattr(cls, 'item')
      and getattr(value, 'ndim', None) == 0
  )


def is_array_object(value: Any) -> bool:
  """Returns True for dense array/tracer objects (e.g. `ndarray`, `jax.Array`).

  Reads only `.shape` and `.dtype` metadata so JAX device buffers are never
  transferred to host.

  Args:
    value: Candidate Python object to inspect.

  Returns:
    `True` iff `value` exposes a tuple `.shape` and non-`None` `.dtype`.
  """
  cls = type(value)
  if (
      cls in BUILTIN_SCALAR_TYPES
      or getattr(cls, STREAM_MARKER_ATTR, False)
      or getattr(cls, SENTINEL_MARKER_ATTR, False)
      or getattr(cls, LAZY_THUNK_MARKER_ATTR, False)
      or is_numpy_scalar(value)
      or isinstance(
          value,
          (enum.Enum, tuple, list, set, frozenset, Mapping),
      )
  ):
    return False
  has_cls_attr = (
      getattr(cls, 'shape', None) is not None
      or getattr(cls, 'dtype', None) is not None
  )
  if not has_cls_attr and 'shape' not in getattr(value, '__dict__', ()):
    return False
  try:
    shape = getattr(value, 'shape', None)
    dtype = getattr(value, 'dtype', None)
  except (AttributeError, TypeError, ValueError, RuntimeError):
    return False
  return isinstance(shape, tuple) and dtype is not None


def is_nan_scalar(value: Any) -> bool:
  """Returns True for NaN `float`, `complex`, or NumPy scalar values."""
  if is_numpy_scalar(value):
    value = value.item()
  if isinstance(value, float):
    return math.isnan(value)
  if isinstance(value, complex):
    return math.isnan(value.real) or math.isnan(value.imag)
  return False


def is_int_ge(value: Any, minimum: int) -> bool:
  """Returns True when `value` is a non-bool `int` with `value >= minimum`."""
  return (
      not isinstance(value, bool)
      and isinstance(value, int)
      and value >= minimum
  )


def validate_non_negative_int(k: int, op_name: str = 'value') -> int:
  """Validates that `k` is a non-negative integer (rejecting booleans)."""
  if not is_int_ge(k, 0):
    raise ValueError(f'{op_name} must be a non-negative integer, got {k!r}.')
  return k


def validate_positive_int(n: int, label: str = 'value') -> int:
  """Validates that `n` is an integer >= 1 (rejecting booleans)."""
  if not is_int_ge(n, 1):
    raise ValueError(f'{label} must be an integer >= 1, got {n!r}.')
  return n


def contains_stream_or_thunk(
    value: Any,
    seen: set[int] | None = None,
    *,
    include_zero_arg_fn: bool = False,
) -> bool:
  """Returns True when `value` is or encloses a `Stream` or thunk leaf."""
  if is_lazy_thunk_obj(value) or is_stream_obj(value):
    return True
  if include_zero_arg_fn and is_zero_arg_function(value):
    return True
  if not isinstance(value, (tuple, list, set, frozenset, Mapping)):
    return False
  active = set() if seen is None else seen
  obj_id = id(value)
  if obj_id in active:
    return False
  active.add(obj_id)
  try:
    items = (
        itertools.chain(value.keys(), value.values())
        if isinstance(value, Mapping)
        else value
    )
    return any(
        contains_stream_or_thunk(
            item, active, include_zero_arg_fn=include_zero_arg_fn
        )
        for item in items
    )
  finally:
    active.discard(obj_id)


def _is_non_scalar_array(value: Any) -> bool:
  """Returns True for arrays whose `==` should not be invoked as scalars."""
  # Preserves 0-D `np.ndarray` comparisons (which return `np.bool_`) while
  # skipping non-0-D arrays and non-NumPy array objects (such as `jax.Array`
  # or duck arrays) to avoid unnecessary elementwise/device evaluation.
  if not is_array_object(value):
    return False
  return bool(value.shape) or type(value).__module__ != 'numpy'


def safe_scalar_eq(av: Any, bv: Any) -> bool:
  """Evaluates `av == bv` safely without `Stream`/array/exception pitfalls."""
  if _is_non_scalar_array(av) or _is_non_scalar_array(bv):
    return False
  if contains_stream_or_thunk(av) or contains_stream_or_thunk(bv):
    return False
  try:
    eq = av == bv
  except (TypeError, ValueError, RuntimeError):
    return False
  if isinstance(eq, bool):
    return eq
  if is_numpy_bool(eq):
    return bool(eq)
  return False


def is_builtin_scalar_value(value: Any) -> bool:
  """Returns True if `value` is a built-in scalar returning `bool` on cmp."""
  cls = type(value)
  if cls in BUILTIN_SCALAR_TYPES:
    return True
  if isinstance(value, enum.Enum):
    return (
        cls.__eq__ in _ENUM_STANDARD_EQ
        and cls.__ne__ in _ENUM_STANDARD_NE
        and cls.__lt__ in _ENUM_STANDARD_LT
        and cls.__le__ in _ENUM_STANDARD_LE
        and cls.__gt__ in _ENUM_STANDARD_GT
        and cls.__ge__ in _ENUM_STANDARD_GE
    )
  return False


def _has_non_null_out(out: Any) -> bool:
  """Returns True when a NumPy `out=` argument specifies a destination array."""
  if out is None:
    return False
  if isinstance(out, tuple):
    return any(item is not None for item in out)
  return True


def reject_out_kwarg(
    fn_name: str, kwargs: Mapping[str, Any], *, kind: str = 'function'
) -> None:
  """Rejects a non-`None` in-place `out=` destination keyword argument."""
  if 'out' in kwargs and _has_non_null_out(kwargs['out']):
    raise TypeError(
        f"In-place 'out=' argument is not supported on Psyre Streams "
        f'({kind} {fn_name!r}).'
    )


def strip_null_out(kwargs: Mapping[str, Any]) -> dict[str, Any]:
  """Returns a copy of `kwargs` with any validated null `out` key removed."""
  if 'out' in kwargs:
    return {k: v for k, v in kwargs.items() if k != 'out'}
  return dict(kwargs)


def reject_numpy_inplace_ufunc(
    ufunc: Any, method: str, kwargs: Mapping[str, Any]
) -> None:
  """Rejects mutating NumPy ufunc calls (`at` or non-`None` `out=`)."""
  uname = getattr(ufunc, '__name__', repr(ufunc))
  if method == 'at':
    raise TypeError(
        f'In-place ufunc method {uname}.at is not supported on Psyre '
        'Streams; streams carry immutable values.'
    )
  reject_out_kwarg(uname, kwargs, kind='ufunc')


def reject_numpy_mutating_func(
    func: Any, kwargs: Mapping[str, Any]
) -> None:
  """Rejects in-place / mutating NumPy array function calls."""
  fname = getattr(func, '__name__', repr(func))
  if fname in MUTATING_NUMPY_FUNCS:
    raise TypeError(
        f'In-place NumPy function {fname!r} is not supported on Psyre '
        'Streams; streams carry immutable values.'
    )
  reject_out_kwarg(fname, kwargs, kind='function')


def is_from_package(obj: Any, package: str) -> bool:
  """Returns True when `obj.__module__` is `package` or a submodule of it."""
  mod = getattr(obj, '__module__', '') or ''
  return mod == package or mod.startswith(f'{package}.')


def is_pure_numpy_func(func: Any) -> bool:
  """Returns True if `func` is a deterministic, non-I/O NumPy function."""
  mod = getattr(func, '__module__', '') or ''
  fname = getattr(func, '__name__', '')
  if mod.startswith('numpy.random') or fname in _IMPURE_NUMPY_FUNC_NAMES:
    return False
  return is_from_package(func, 'numpy')


def is_namedtuple(value: Any) -> bool:
  """Returns True when `value` is a `NamedTuple` instance."""
  return isinstance(value, tuple) and hasattr(type(value), '_fields')


def is_zero_arg_function(val: Any) -> bool:
  """Returns True when `val` is a strictly zero-argument Python function."""
  if not isinstance(val, types.FunctionType):
    return False
  code = val.__code__
  return (
      code.co_argcount == 0
      and code.co_kwonlyargcount == 0
      and not (code.co_flags & _NON_SYNC_FUNC_FLAGS)
  )


def reject_callable(
    value: Any,
    *,
    label: str = 'Stream value',
    reject_streams: bool = False,
) -> None:
  """Raises TypeError if `value` is or encloses a callable (or `Stream`).

  Psyre streams are strictly first-order: functions, methods, classes/types,
  and callable objects are rejected as stream payloads (directly or nested
  inside `tuple`, `list`, `set`, `frozenset`, or `Mapping` containers). When
  `reject_streams=True`, nested `Stream` instances are also rejected in the
  same traversal (preserving `LazyThunk`/callable error precedence).

  Args:
    value: Candidate payload value to check.
    label: Subject label prefix used in `TypeError` messages.
    reject_streams: When True, also reject any `Stream` object encountered.

  Raises:
    TypeError: If `value` or any nested container item is a `LazyThunk`,
      callable, or (when `reject_streams=True`) a `Stream`.
  """
  if is_sentinel_obj(value) or type(value) in BUILTIN_SCALAR_TYPES:
    return
  active: set[int] | None = None
  has_stream = False

  def check(v: Any) -> None:
    nonlocal active, has_stream
    if is_sentinel_obj(v):
      return
    cls = type(v)
    if cls in BUILTIN_SCALAR_TYPES:
      return
    if is_lazy_thunk_obj(v):
      raise TypeError(
          'Cannot wrap an unresolved LazyThunk in Stream(...); pass it '
          'directly to Psyre combinators or resolve it first.'
      )
    if reject_streams and is_stream_obj(v):
      has_stream = True
      return
    if callable(v):
      raise TypeError(
          f'{label} cannot contain a callable ({cls.__name__}); Psyre streams '
          'are first-order.'
      )
    if not isinstance(v, (tuple, list, set, frozenset, Mapping)):
      return
    if active is None:
      active = set()
    obj_id = id(v)
    if obj_id in active:
      return
    active.add(obj_id)
    try:
      items = (
          itertools.chain(v.keys(), v.values())
          if isinstance(v, Mapping)
          else v
      )
      for item in items:
        check(item)
    finally:
      active.discard(obj_id)

  check(value)
  if has_stream:
    raise TypeError(
        f'{label} cannot contain a Stream instance; Psyre streams are '
        'first-order.'
    )


def register_pytree_node(
    node_type: type[Any],
    flatten_fn: PyTreeFlattenFn,
    unflatten_fn: PyTreeUnflattenFn,
) -> None:
  """Registers a custom container type for `@psyre_comb` PyTree flattening.

  Matches the `(flatten_fn, unflatten_fn)` contract of
  `jax.tree_util.register_pytree_node` so JAX/Flax/Equinox or user containers
  can plug into Stage-1 argument flattening without adding a hard `jax`
  dependency to core `psyre`.

  Args:
    node_type: Concrete Python class to treat as a PyTree internal node.
    flatten_fn: Callable mapping `node -> (children, aux_data)`.
    unflatten_fn: Callable mapping `(aux_data, children) -> node`.

  Raises:
    TypeError: If `node_type` is not a `type` or handlers are not callable.
  """
  if not isinstance(node_type, type):
    raise TypeError(
        f'node_type must be a class/type, got {type(node_type).__name__}.'
    )
  if not is_callable_target(flatten_fn) or not is_callable_target(unflatten_fn):
    raise TypeError('flatten_fn and unflatten_fn must be callable.')
  _pytree_registry[node_type] = (flatten_fn, unflatten_fn)


def unregister_pytree_node(node_type: type[Any]) -> None:
  """Removes a previously registered custom PyTree node type, if present."""
  _pytree_registry.pop(node_type, None)


def lookup_pytree_node(
    node_type: type[Any],
) -> tuple[PyTreeFlattenFn, PyTreeUnflattenFn] | None:
  """Returns the registered `(flatten_fn, unflatten_fn)` pair or `None`."""
  return _pytree_registry.get(node_type)


class PlanNode:
  """Base class for frozen template nodes inside a `CombCallPlan`."""

  __slots__ = ()

  def materialize(self, values: Sequence[Any]) -> Any:
    """Reconstructs the runtime Python value using pulled tick `values`."""
    raise NotImplementedError


@dataclasses.dataclass(frozen=True, slots=True)
class DynamicSlot(PlanNode):
  """Placeholder inside a `CombCallPlan` referencing `node.inputs[index]`."""

  index: int

  def materialize(self, values: Sequence[Any]) -> Any:
    return values[self.index]


def _materialize_item(item: Any, values: Sequence[Any]) -> Any:
  return item.materialize(values) if isinstance(item, PlanNode) else item


@dataclasses.dataclass(frozen=True, slots=True)
class SeqTemplate(PlanNode):
  """Template reconstructing a `tuple`, `list`, or `NamedTuple` container."""

  seq_type: type[Any]
  items: tuple[Any, ...]
  named: bool = False

  def materialize(self, values: Sequence[Any]) -> Any:
    elems = (_materialize_item(x, values) for x in self.items)
    if self.named:
      return self.seq_type(*elems)
    return list(elems) if self.seq_type is list else tuple(elems)


@dataclasses.dataclass(frozen=True, slots=True)
class DictTemplate(PlanNode):
  """Template reconstructing a `dict` or `Mapping` with dynamic values."""

  mapping_type: type[Any]
  items: tuple[tuple[Any, Any], ...]

  def materialize(self, values: Sequence[Any]) -> Any:
    if self.mapping_type is types.MappingProxyType:
      return types.MappingProxyType(
          {k: _materialize_item(v, values) for k, v in self.items}
      )
    if self.mapping_type is collections.OrderedDict:
      return collections.OrderedDict(
          (k, _materialize_item(v, values)) for k, v in self.items
      )
    return {k: _materialize_item(v, values) for k, v in self.items}


@dataclasses.dataclass(frozen=True, slots=True)
class SliceTemplate(PlanNode):
  """Template reconstructing a `slice` with dynamic start/stop/step."""

  start: Any
  stop: Any
  step: Any

  def materialize(self, values: Sequence[Any]) -> slice:
    return slice(
        _materialize_item(self.start, values),
        _materialize_item(self.stop, values),
        _materialize_item(self.step, values),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CustomNodeTemplate(PlanNode):
  """Template reconstructing a registered custom PyTree node."""

  node_type: type[Any]
  unflatten_fn: PyTreeUnflattenFn
  aux_data: Any
  children: tuple[Any, ...]

  def materialize(self, values: Sequence[Any]) -> Any:
    return self.unflatten_fn(
        self.aux_data,
        tuple(_materialize_item(c, values) for c in self.children),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CombCallPlan:
  """Frozen call template rebuilding `(args, kwargs)` from flat stream inputs.

  Attributes:
    args_template: Tuple of static values and `PlanNode` slots for positional
      arguments.
    kwargs_template: Tuple of `(name, value_or_plan_node)` pairs for keyword
      arguments.
  """

  args_template: tuple[Any, ...]
  kwargs_template: tuple[tuple[str, Any], ...]

  def materialize(
      self, values: Sequence[Any]
  ) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Materializes runtime `(args, kwargs)` from per-tick stream `values`."""
    call_args = tuple(
        _materialize_item(item, values) for item in self.args_template
    )
    call_kwargs = {
        k: _materialize_item(v, values) for k, v in self.kwargs_template
    }
    return call_args, call_kwargs


def materialize_comb_call(
    kwargs: Mapping[str, Any], values: Sequence[Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Rebuilds `(call_args, call_kwargs)` from `node.kwargs` and tick values."""
  plan = kwargs.get(CALL_PLAN_KWARG)
  if plan is not None:
    return plan.materialize(values)
  return tuple(values), dict(kwargs)


def is_dynamic_leaf(val: Any) -> bool:
  """Returns True when `val` is a `Stream` or Stage-1 deferred thunk."""
  return (
      is_stream_obj(val)
      or is_lazy_thunk_obj(val)
      or is_zero_arg_function(val)
  )


def _build_template(
    val: Any, dyn_inputs: list[Any], active: set[int] | None = None
) -> Any:
  """Replaces nested dynamic leaves in `val` with `DynamicSlot` markers."""
  # Single bottom-up pass tracking `len(dyn_inputs)`: subtrees that add no
  # dynamic leaves stay verbatim (no `PlanNode` allocated), and custom PyTree
  # `flatten_fn` callbacks run at most once per node.
  if is_dynamic_leaf(val):
    idx = len(dyn_inputs)
    dyn_inputs.append(val)
    return DynamicSlot(idx)
  cls = type(val)
  if cls in BUILTIN_SCALAR_TYPES:
    return val
  if isinstance(val, slice):
    start_len = len(dyn_inputs)
    start = _build_template(val.start, dyn_inputs, active)
    stop = _build_template(val.stop, dyn_inputs, active)
    step = _build_template(val.step, dyn_inputs, active)
    if len(dyn_inputs) == start_len:
      return val
    return SliceTemplate(start=start, stop=stop, step=step)
  reg = _pytree_registry.get(cls)
  if reg is None and not isinstance(val, (tuple, list, Mapping)):
    return val
  seen = set() if active is None else active
  obj_id = id(val)
  if obj_id in seen:
    if contains_stream_or_thunk(val, include_zero_arg_fn=True):
      raise ValueError(
          'Cyclic container containing dynamic Stream/thunk arguments is not '
          f'supported ({cls.__name__}).'
      )
    return val
  seen.add(obj_id)
  try:
    start_len = len(dyn_inputs)
    if reg is not None:
      flatten_fn, unflatten_fn = reg
      children, aux_data = flatten_fn(val)
      tmpl_children = tuple(
          _build_template(c, dyn_inputs, seen) for c in children
      )
      if len(dyn_inputs) == start_len:
        return val
      return CustomNodeTemplate(
          node_type=cls,
          unflatten_fn=unflatten_fn,
          aux_data=aux_data,
          children=tmpl_children,
      )
    if isinstance(val, (tuple, list)):
      items = tuple(_build_template(x, dyn_inputs, seen) for x in val)
      if len(dyn_inputs) == start_len:
        return val
      named = isinstance(val, tuple) and is_namedtuple(val)
      return SeqTemplate(
          cls if named else (list if isinstance(val, list) else tuple),
          items,
          named=named,
      )
    pairs = tuple(
        (k, _build_template(v, dyn_inputs, seen)) for k, v in val.items()
    )
    if len(dyn_inputs) == start_len:
      return val
    return DictTemplate(cls, pairs)
  finally:
    seen.discard(obj_id)


def partition_comb_args(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    static_argnums: frozenset[int] = frozenset(),
    static_argnames: frozenset[str] = frozenset(),
    static_literals: bool = False,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Partitions `(args, kwargs)` into flat stream `inputs` and `node.kwargs`.

  Preserves the zero-overhead fast path (no `CombCallPlan`) when all positional
  arguments are flat dynamic slots and all keyword arguments are static.

  Args:
    args: Positional arguments passed to the `@psyre_comb` wrapper.
    kwargs: Keyword arguments passed to the `@psyre_comb` wrapper.
    static_argnums: Zero-based positional indices kept verbatim as static args.
    static_argnames: Keyword argument names kept verbatim as static args.
    static_literals: When True, positional args containing no `Stream` or
      Stage-1 thunk remain static literals inside `CombCallPlan` instead of
      being promoted via `resolve_stream`.

  Returns:
    `(raw_inputs, node_kwargs)` tuple ready for `GraphBuilder.add_loop_node`.
  """
  if CALL_PLAN_KWARG in kwargs:
    raise ValueError(
        f'Keyword argument {CALL_PLAN_KWARG!r} is reserved for internal '
        'Psyre call plans.'
    )
  dyn_inputs: list[Any] = []
  args_template: list[Any] = []
  # Fast path stays `needs_plan == False` when every positional arg maps 1:1
  # to a flat `node.inputs[i]` slot and all kwargs are pure static scalars; on
  # the slow path (`needs_plan == True`), all args/kwargs bundle into a single
  # `CombCallPlan` stored under `node.kwargs[CALL_PLAN_KWARG]`.
  needs_plan = False
  for i, arg in enumerate(args):
    if i in static_argnums:
      args_template.append(arg)
      needs_plan = True
    elif is_dynamic_leaf(arg):
      slot = len(dyn_inputs)
      dyn_inputs.append(arg)
      args_template.append(DynamicSlot(slot))
    else:
      start_len = len(dyn_inputs)
      tmpl = _build_template(arg, dyn_inputs)
      if len(dyn_inputs) > start_len:
        args_template.append(tmpl)
        needs_plan = True
      elif static_literals:
        args_template.append(arg)
        needs_plan = True
      else:
        slot = len(dyn_inputs)
        dyn_inputs.append(arg)
        args_template.append(DynamicSlot(slot))

  kw_pairs: list[tuple[str, Any]] = []
  for k, val in kwargs.items():
    if k in static_argnames:
      kw_pairs.append((k, val))
    else:
      start_len = len(dyn_inputs)
      tmpl = _build_template(val, dyn_inputs)
      if len(dyn_inputs) > start_len:
        needs_plan = True
      kw_pairs.append((k, tmpl))

  if not needs_plan:
    return tuple(dyn_inputs), dict(kwargs)
  plan = CombCallPlan(
      args_template=tuple(args_template),
      kwargs_template=tuple(kw_pairs),
  )
  return tuple(dyn_inputs), {CALL_PLAN_KWARG: plan}


def normalize_static_argnums(
    static_argnums: int | Iterable[int],
) -> frozenset[int]:
  """Normalizes `static_argnums` to a validated `frozenset[int]`."""
  if isinstance(static_argnums, int) and not isinstance(static_argnums, bool):
    validate_non_negative_int(static_argnums, 'static_argnums')
    return frozenset({static_argnums})
  if isinstance(
      static_argnums, (bool, str, bytes, bytearray)
  ) or not isinstance(static_argnums, Iterable):
    raise TypeError(
        'static_argnums must be a non-negative int or iterable of ints, got '
        f'{type(static_argnums).__name__}.'
    )
  return frozenset(
      validate_non_negative_int(idx, 'static_argnums entry')
      for idx in static_argnums
  )


def normalize_static_argnames(
    static_argnames: str | Iterable[str],
) -> frozenset[str]:
  """Normalizes `static_argnames` to a validated `frozenset[str]`."""
  if isinstance(static_argnames, str):
    return frozenset({static_argnames})
  if not isinstance(static_argnames, Iterable):
    raise TypeError(
        'static_argnames must be a str or iterable of strs, got '
        f'{type(static_argnames).__name__}.'
    )
  names: list[str] = []
  for name in static_argnames:
    if not isinstance(name, str):
      raise TypeError(
          'static_argnames entries must be str, got '
          f'{type(name).__name__}.'
      )
    names.append(name)
  return frozenset(names)


# Shared pure-stdlib lifting helpers (zero numpy/jax dependency) used by
# optional array backends (`psyre.numpy`, `psyre.jax`, `psyre.jax.lax`).

NoutSpec = int | Callable[[tuple[Any, ...], Mapping[str, Any]], int]


def _numpy_qr_nout(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
  mode = args[1] if len(args) > 1 else kwargs.get('mode', 'reduced')
  if isinstance(mode, str) and mode in (
      'reduced',
      'complete',
      'raw',
      'full',
      'f',
  ):
    return 2
  return 1


def _numpy_svd_nout(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
  raw = args[2] if len(args) > 2 else kwargs.get('compute_uv', True)
  if is_dynamic_leaf(raw):
    return 1
  return 3 if bool(raw) else 1


ARRAY_ELEMWISE_NOUT: types.MappingProxyType[str, NoutSpec] = (
    types.MappingProxyType({
        'divmod': 2,
        'frexp': 2,
        'modf': 2,
    })
)

ARRAY_LINALG_NOUT: types.MappingProxyType[str, NoutSpec] = (
    types.MappingProxyType({
        'eig': 2,
        'eigh': 2,
        'lstsq': 4,
        'qr': _numpy_qr_nout,
        'slogdet': 2,
        'svd': _numpy_svd_nout,
    })
)


def resolve_nout(
    spec: NoutSpec | None,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> int:
  """Evaluates a static `int` or argument-aware output-arity `NoutSpec`."""
  if spec is None:
    return 1
  return spec if isinstance(spec, int) else spec(args, kwargs)


def is_int_axis(value: Any) -> bool:
  """Returns True when `value` is an `int` axis (rejecting `bool`)."""
  return isinstance(value, int) and not isinstance(value, bool)


def validate_axis(axis: Any, label: str = 'axis') -> int:
  """Validates that `axis` is a plain `int` (not `bool`)."""
  if not is_int_axis(axis):
    raise TypeError(f'{label} must be an int, got {type(axis).__name__}.')
  return axis


def validate_optional_axis(axis: Any, label: str = 'axis') -> int | None:
  """Validates that `axis` is an `int` (not `bool`) or `None`."""
  if axis is None:
    return None
  if not is_int_axis(axis):
    raise TypeError(
        f'{label} entries must be int or None, got {type(axis).__name__}.'
    )
  return axis


def normalize_arg_axes(
    axes_spec: Any,
    num_args: int,
    label: str = 'in_axes',
    *,
    error_cls: type[Exception] = ValueError,
) -> tuple[Any, ...]:
  """Normalizes an `in_axes` spec to one entry per positional argument."""
  if axes_spec is None or is_int_axis(axes_spec):
    return (axes_spec,) * num_args
  if isinstance(axes_spec, (tuple, list)):
    if len(axes_spec) != num_args:
      raise error_cls(
          f'{label} length {len(axes_spec)} does not match {num_args} '
          'positional arguments.'
      )
    return tuple(axes_spec)
  raise TypeError(
      f'{label} must be int, None, or a tuple/list matching positional args, '
      f'got {type(axes_spec).__name__}.'
  )


def map_tree_with_axes(
    value: Any,
    axis_spec: Any,
    leaf_fn: Callable[[Any, Any], Any],
    *,
    label: str = 'axes',
    is_leaf: Callable[[Any], bool] | None = None,
    error_cls: type[Exception] = ValueError,
) -> Any:
  """Walks a nested container / registered PyTree in lockstep with `axis_spec`.

  Scalar `axis_spec` (`None` or `int`) broadcasts across all children. Supports
  `tuple`, `namedtuple`, `list`, `Mapping`, and custom classes registered via
  `register_pytree_node`.

  Args:
    value: Nested input or output tree.
    axis_spec: Matching axis specification tree, or scalar `int | None`.
    leaf_fn: Callback invoked as `leaf_fn(leaf_value, leaf_axis)`.
    label: Label used in structure-mismatch error messages.
    is_leaf: Optional predicate treating matching objects as leaves before
      container descent (defaults to `is_dynamic_leaf`).
    error_cls: Exception class raised on container structure mismatches.

  Returns:
    Reconstructed tree with leaves transformed by `leaf_fn`.
  """
  leaf_pred = is_dynamic_leaf if is_leaf is None else is_leaf
  if value is None or leaf_pred(value):
    return leaf_fn(value, axis_spec)
  cls = type(value)
  reg = _pytree_registry.get(cls)
  if reg is not None:
    flatten_fn, unflatten_fn = reg
    children, aux = flatten_fn(value)
    if axis_spec is None or is_int_axis(axis_spec):
      sub_axes = (axis_spec,) * len(children)
    elif isinstance(axis_spec, cls) and type(axis_spec) is cls:  # pylint: disable=unidiomatic-typecheck
      spec_children, spec_aux = flatten_fn(axis_spec)
      if spec_aux != aux or len(spec_children) != len(children):
        raise error_cls(
            f'{label} {cls.__name__} structure does not match value.'
        )
      sub_axes = spec_children
    else:
      raise error_cls(
          f'{label} must broadcast (int/None) or match {cls.__name__}.'
      )
    mapped = tuple(
        map_tree_with_axes(
            c, s, leaf_fn, label=label, is_leaf=leaf_pred, error_cls=error_cls
        )
        for c, s in zip(children, sub_axes)
    )
    return unflatten_fn(aux, mapped)
  if isinstance(value, tuple):
    if axis_spec is None or is_int_axis(axis_spec):
      sub_axes = (axis_spec,) * len(value)
    elif isinstance(axis_spec, tuple) and len(axis_spec) == len(value):
      sub_axes = axis_spec
    else:
      raise error_cls(f'{label} tuple structure does not match argument tuple.')
    items = tuple(
        map_tree_with_axes(
            v, s, leaf_fn, label=label, is_leaf=leaf_pred, error_cls=error_cls
        )
        for v, s in zip(value, sub_axes)
    )
    return cls(*items) if is_namedtuple(value) else items
  if isinstance(value, list):
    if axis_spec is None or is_int_axis(axis_spec):
      sub_axes = (axis_spec,) * len(value)
    elif isinstance(axis_spec, list) and len(axis_spec) == len(value):
      sub_axes = axis_spec
    else:
      raise error_cls(f'{label} list structure does not match argument list.')
    return [
        map_tree_with_axes(
            v, s, leaf_fn, label=label, is_leaf=leaf_pred, error_cls=error_cls
        )
        for v, s in zip(value, sub_axes)
    ]
  if isinstance(value, Mapping):
    if axis_spec is None or is_int_axis(axis_spec):
      return {
          k: map_tree_with_axes(
              v,
              axis_spec,
              leaf_fn,
              label=label,
              is_leaf=leaf_pred,
              error_cls=error_cls,
          )
          for k, v in value.items()
      }
    if isinstance(axis_spec, Mapping) and set(axis_spec) == set(value):
      return {
          k: map_tree_with_axes(
              v,
              axis_spec[k],
              leaf_fn,
              label=label,
              is_leaf=leaf_pred,
              error_cls=error_cls,
          )
          for k, v in value.items()
      }
    raise error_cls(f'{label} dict keys do not match argument dict keys.')
  return leaf_fn(value, axis_spec)


def stack_output_trees(
    items: Sequence[Any],
    leaf_stack_fn: Callable[[list[Any], str], Any],
    *,
    name: str = '',
) -> Any:
  """Stacks matching container trees across a non-empty sequence of lanes."""
  first = items[0]
  if first is None:
    if not all(it is None for it in items):
      raise ValueError(
          'Batched lane outputs have mismatched structure (None vs non-None).'
      )
    return None
  cls = type(first)
  reg = _pytree_registry.get(cls)
  if reg is not None:
    flatten_fn, unflatten_fn = reg
    if not all(
        type(it) is cls for it in items  # pylint: disable=unidiomatic-typecheck
    ):
      raise ValueError(
          'Batched lane outputs have mismatched PyTree node types: expected '
          f'{cls.__name__}.'
      )
    flat_items = [flatten_fn(it) for it in items]
    aux = flat_items[0][1]
    arity = len(flat_items[0][0])
    if any(a != aux or len(ch) != arity for ch, a in flat_items):
      raise ValueError(
          'Batched lane outputs have mismatched PyTree structure for '
          f'{cls.__name__}.'
      )
    children = [ch for ch, _ in flat_items]
    transposed = tuple(
        stack_output_trees(
            [ch[i] for ch in children], leaf_stack_fn, name=f'{name}_{i}'
        )
        for i in range(arity)
    )
    return unflatten_fn(aux, transposed)
  if isinstance(first, tuple):
    if is_namedtuple(first):
      if any(
          type(it) is not cls  # pylint: disable=unidiomatic-typecheck
          for it in items
      ):
        raise ValueError(
            'Batched lane outputs have mismatched NamedTuple types: expected '
            f'{cls.__name__}.'
        )
    elif any(
        not isinstance(it, tuple) or len(it) != len(first) for it in items
    ):
      raise ValueError(
          'Batched lane outputs have mismatched tuple lengths or types.'
      )
    transposed = tuple(
        stack_output_trees(
            [it[i] for it in items], leaf_stack_fn, name=f'{name}_{i}'
        )
        for i in range(len(first))
    )
    return cls(*transposed) if is_namedtuple(first) else transposed
  if isinstance(first, list):
    if any(not isinstance(it, list) or len(it) != len(first) for it in items):
      raise ValueError(
          'Batched lane outputs have mismatched list lengths or types.'
      )
    return [
        stack_output_trees(
            [it[i] for it in items], leaf_stack_fn, name=f'{name}_{i}'
        )
        for i in range(len(first))
    ]
  if isinstance(first, Mapping):
    keys = set(first.keys())
    if any(
        not isinstance(it, Mapping) or set(it.keys()) != keys for it in items
    ):
      raise ValueError(
          'Batched lane outputs have mismatched mapping keys or types.'
      )
    return {
        k: stack_output_trees(
            [it[k] for it in items], leaf_stack_fn, name=f'{name}_{k}'
        )
        for k in first
    }
  if any(it is None for it in items):
    raise ValueError(
        'Batched lane outputs have mismatched structure (None vs non-None).'
    )
  return leaf_stack_fn(list(items), name)


class LiftedNamespace:
  """Lazy attribute proxy lifting a Python module onto Psyre `Stream`s."""

  def __init__(
      self,
      module: types.ModuleType,
      label: str = '',
      *,
      wrap_attr: Callable[[str, Any], Any],
      submodules: Mapping[str, Any] | None = None,
      rejected: frozenset[str] = frozenset(),
      reject_message: str = (
          'Unsupported operation {label}.{name} on pure Psyre streams.'
      ),
  ):
    self._module = module
    self._label = label or module.__name__
    self._wrap_attr = wrap_attr
    self._submodules = dict(submodules or {})
    self._rejected = rejected
    self._reject_message = reject_message
    self._cache: dict[str, Any] = {}

  def __getattr__(self, name: str) -> Any:
    if name.startswith('__') and name.endswith('__'):
      raise AttributeError(name)
    if name in self._cache:
      return self._cache[name]
    if name in self._submodules:
      sub = self._submodules[name]
      self._cache[name] = sub
      return sub
    try:
      attr = getattr(self._module, name)
    except AttributeError as err:
      raise AttributeError(
          f'module {self._label!r} has no attribute {name!r}'
      ) from err
    if name in self._rejected:
      msg = self._reject_message.format(label=self._label, name=name)

      def _reject(*_a: Any, **_kw: Any) -> Any:
        raise TypeError(msg)

      _reject.__name__ = name
      self._cache[name] = _reject
      return _reject
    if isinstance(attr, types.ModuleType):
      wrapped: Any = LiftedNamespace(
          attr,
          f'{self._label}.{name}',
          wrap_attr=self._wrap_attr,
          rejected=self._rejected,
          reject_message=self._reject_message,
      )
    else:
      wrapped = self._wrap_attr(name, attr)
    self._cache[name] = wrapped
    return wrapped

  def __dir__(self) -> list[str]:
    return sorted(set(dir(self._module)) | set(self._submodules))

  def __repr__(self) -> str:
    return f'<LiftedNamespace {self._label!r}>'


def make_stack_streams(
    stack_fn: Callable[..., Any],
    make_step_fn: Callable[..., Any],
    emit_fn: Callable[..., Any],
    *,
    doc: str = '',
) -> Callable[..., Any]:
  """Builds a backend-parameterized `stack_streams(*streams, axis=0)`."""

  @functools.lru_cache(maxsize=64)
  def _cached_step(axis: int) -> Callable[..., Any]:
    def _do_stack(*vals: Any) -> Any:
      return stack_fn(vals, axis=axis)

    _do_stack.__name__ = 'stack_streams'
    return make_step_fn(_do_stack, name='stack_streams', pure=True)

  def stack_streams(*streams: Any, axis: int = 0, name: str = '') -> Any:
    if not streams:
      raise ValueError('stack_streams() requires at least one stream.')
    validate_axis(axis, 'axis')
    label = name or 'stack_streams'
    dyn_inputs, node_kwargs = partition_comb_args(
        streams, {}, static_literals=True
    )
    return emit_fn(_cached_step(axis), dyn_inputs, node_kwargs, name=label)

  if doc:
    stack_streams.__doc__ = doc
  return stack_streams


def make_unstack_stream(
    asarray_fn: Callable[[Any], Any],
    moveaxis_fn: Callable[[Any, int, int], Any],
    make_step_fn: Callable[..., Any],
    emit_fn: Callable[..., Any],
    *,
    doc: str = '',
) -> Callable[..., tuple[Any, ...]]:
  """Builds a backend-parameterized `unstack_stream(stream, size, axis=0)`."""

  @functools.lru_cache(maxsize=256)
  def _cached_slice_step(axis: int, size: int, idx: int) -> Callable[..., Any]:
    def _slice_axis(arr: Any) -> Any:
      arr_v = asarray_fn(arr)
      moved = moveaxis_fn(arr_v, axis, 0)
      if moved.shape[0] != size:
        raise ValueError(
            f'unstack_stream expected axis {axis} size {size}, got '
            f'shape {arr_v.shape}.'
        )
      return moved[idx]

    _slice_axis.__name__ = 'unstack_stream'
    return make_step_fn(_slice_axis, name='unstack_stream', pure=True)

  def unstack_stream(
      stream: Any, size: int, *, axis: int = 0, name: str = ''
  ) -> tuple[Any, ...]:
    validate_positive_int(size, 'unstack_stream size')
    validate_axis(axis, 'axis')
    base_name = name or 'unstack'
    dyn_inputs, node_kwargs = partition_comb_args(
        (stream,), {}, static_literals=True
    )
    out: list[Any] = []
    for i in range(size):
      label = f'{base_name}_{i}'
      step_fn = _cached_slice_step(axis, size, i)
      out.append(emit_fn(step_fn, dyn_inputs, node_kwargs, name=label))
    return tuple(out)

  if doc:
    unstack_stream.__doc__ = doc
  return unstack_stream


def make_batch_streams(
    unstack_fn: Callable[..., tuple[Any, ...]],
    stack_fn: Callable[..., Any],
    require_builder_fn: Callable[[], Any],
    push_prefix_fn: Callable[[str], Any],
    *,
    error_cls: type[Exception] = ValueError,
    doc: str = '',
) -> Callable[..., Any]:
  """Builds a backend-parameterized structural Stage-1 `batch_streams`."""

  def batch_streams(
      fn: Callable[..., Any],
      *streams: Any,
      size: int,
      in_axes: Any = 0,
      out_axis: int | None = 0,
      name: str = '',
  ) -> Any:
    if not is_callable_target(fn):
      raise TypeError(
          f'batch_streams expects a callable, got {type(fn).__name__}.'
      )
    validate_positive_int(size, 'batch_streams size')
    validate_optional_axis(out_axis, 'out_axis')
    require_builder_fn()
    axes = normalize_arg_axes(
        in_axes, len(streams), 'batch_streams in_axes', error_cls=error_cls
    )
    base = name or getattr(fn, '__name__', 'batch')
    per_arg_lanes: list[tuple[Any, ...]] = []
    for pos, (arg, ax) in enumerate(zip(streams, axes)):
      if ax is None:
        per_arg_lanes.append((arg,) * size)
      else:
        validate_axis(ax, 'batch_streams in_axes entry')
        per_arg_lanes.append(
            unstack_fn(arg, size, axis=ax, name=f'{base}_in{pos}')
        )
    lane_outputs: list[Any] = []
    for lane in range(size):
      lane_args = tuple(lanes[lane] for lanes in per_arg_lanes)
      with push_prefix_fn(f'{base}[{lane}]'):
        lane_outputs.append(fn(*lane_args))

    def _stack_leaf(vals: list[Any], leaf_name: str) -> Any:
      if out_axis is None or not any(is_dynamic_leaf(v) for v in vals):
        return tuple(vals)
      return stack_fn(*vals, axis=out_axis, name=leaf_name)

    return stack_output_trees(lane_outputs, _stack_leaf, name=f'{base}_out')

  if doc:
    batch_streams.__doc__ = doc
  return batch_streams
