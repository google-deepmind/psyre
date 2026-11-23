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

from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import enum
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
    'empty_like',
    'save',
    'savetxt',
    'savez',
    'savez_compressed',
    'tofile',
})

_VAR_ARG_FLAGS = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

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


def is_numpy_bool(value: Any) -> bool:
  """Returns True for NumPy scalar booleans (`np.bool_`) without importing."""
  cls = type(value)
  return cls.__module__ == 'numpy' and cls.__name__ in {'bool_', 'bool'}


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
    return cls.__eq__ in _ENUM_STANDARD_EQ and cls.__ne__ in _ENUM_STANDARD_NE
  return False


def _has_non_null_out(out: Any) -> bool:
  """Returns True when a NumPy `out=` argument specifies a destination array."""
  if out is None:
    return False
  if isinstance(out, tuple):
    return any(item is not None for item in out)
  return True


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
  if 'out' in kwargs and _has_non_null_out(kwargs['out']):
    raise TypeError(
        f"In-place 'out=' argument is not supported on Psyre Streams "
        f'(ufunc {uname!r}).'
    )


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
  if 'out' in kwargs and _has_non_null_out(kwargs['out']):
    raise TypeError(
        f"In-place 'out=' argument is not supported on Psyre Streams "
        f'(function {fname!r}).'
    )


def is_pure_numpy_func(func: Any) -> bool:
  """Returns True if `func` is a deterministic, non-I/O NumPy function."""
  mod = getattr(func, '__module__', '') or ''
  fname = getattr(func, '__name__', '')
  if mod.startswith('numpy.random') or fname in _IMPURE_NUMPY_FUNC_NAMES:
    return False
  return mod == 'numpy' or mod.startswith('numpy.')


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
      and not (code.co_flags & _VAR_ARG_FLAGS)
  )


def reject_callable(value: Any, *, label: str = 'Stream value') -> None:
  """Raises TypeError if `value` is or encloses a callable object.

  Psyre streams are strictly first-order: functions, methods, classes/types,
  and callable objects are rejected as stream payloads (directly or nested
  inside `tuple`, `list`, `set`, `frozenset`, or `Mapping` containers).

  Args:
    value: Candidate payload value to check.
    label: Subject label prefix used in `TypeError` messages.

  Raises:
    TypeError: If `value` or any nested container item is callable.
  """
  if is_sentinel_obj(value) or type(value) in BUILTIN_SCALAR_TYPES:
    return
  active: set[int] | None = None

  def check(v: Any) -> None:
    nonlocal active
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
  if not callable(flatten_fn) or not callable(unflatten_fn):
    raise TypeError('flatten_fn and unflatten_fn must be callable.')
  _pytree_registry[node_type] = (flatten_fn, unflatten_fn)


def unregister_pytree_node(node_type: type[Any]) -> None:
  """Removes a previously registered custom PyTree node type, if present."""
  _pytree_registry.pop(node_type, None)


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
    if self.mapping_type is dict:
      return {k: _materialize_item(v, values) for k, v in self.items}
    if self.mapping_type is types.MappingProxyType:
      return types.MappingProxyType(
          {k: _materialize_item(v, values) for k, v in self.items}
      )
    return self.mapping_type(
        (k, _materialize_item(v, values)) for k, v in self.items
    )


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


def _is_dynamic_leaf(val: Any) -> bool:
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
  if _is_dynamic_leaf(val):
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
    elif _is_dynamic_leaf(arg):
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
