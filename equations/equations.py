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
"""Declarative lambda-free Psyre equations (experimental, work in progress).

Note: Everything under `psyre/equations/` is highly experimental; APIs,
semantics, and internals may change or be removed without notice. Inspired by
Lustre/Lucid Synchrone recursive node equations (Halbwachs et al., 1991; Caspi
& Pouzet, 1996; Pouzet, 2006).

Provides a context manager (`psyre_equations`) and metaclass (`Equations`)
allowing Psyre stream equations to refer to variables and
`match()`/`automaton()` outputs before they are defined, without explicit
`lambda:` wrappers or `metaclass=Equations` boilerplate:

```python
with psyre_equations() as prog:
  class prefix_sum(data=Stream([1, 2, 3, 4]), start=0):
    acc = fby(start, tmp)
    tmp = acc + data
    __output__ = tmp

traces = Scheduler.run_graph(prog.graph, horizon=4)
```

Inside a `with psyre_equations()` block, any direct base-less `class`
statement without an explicit `metaclass=` automatically uses `Equations` via
a frame-scoped `builtins.__build_class__` hook. `psyre_equations()` works both
standalone (creating, finalizing, and optionally optimizing a `Graph`,
defaulting to `Semantics.SYNC`) and nested inside an active Stage-1
`GraphBuilder` (such as inside `Scheduler.run`, `build_graph`, or a
`@psyre_node` function).
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Iterable, Iterator, Mapping
import contextlib
import contextvars
import dataclasses
import dis
import operator
import sys
import threading
import types
from typing import Any, override

# pylint: disable=g-importing-member,g-multiple-import
from psyre.builder import GraphBuilder
from psyre.common import (
    BlockHandle,
    CapturedEnv,
    LazyThunk,
    Semantics,
    active_builder,
    force_thunk,
    format_tag,
    is_building,
    is_namedtuple,
    is_thunk,
    prefix_stack,
    push_prefix,
)
from psyre.common_payload import (
    NUMPY_UFUNC_METHODS,
    reject_numpy_inplace_ufunc,
    reject_numpy_mutating_func,
)
from psyre.graph import Graph
from psyre.optimization import optimize_graph
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import

_UNRESOLVED = object()
_MAX_THUNK_DESC_LEN = 120


def _format_desc_operand(value: Any) -> str:
  """Formats an operand concisely for `Thunk._desc` diagnostics."""
  if isinstance(value, Thunk):
    return value._desc  # pylint: disable=protected-access
  return format_tag(value, 40)


def _is_dunder(name: str) -> bool:
  return len(name) > 4 and name.startswith('__') and name.endswith('__')


class CircularEquationError(RecursionError):
  """Raised when unresolved equation Thunks form a non-delayed cycle."""


class _DeferredError(LookupError):
  """Internal error when a Thunk reads a not-yet-bound name or handle.

  `scope` distinguishes an unbound local forward reference (set to the owning
  `EquationScope`, converted to `NameError` at class finalization) from an
  unfinalized outer `BlockHandle` read (`scope=None`, kept deferred across an
  inner class exit so an enclosing block can finish populating it).
  """

  def __init__(self, message: str, *, scope: EquationScope | None = None):
    super().__init__(message)
    self.scope = scope


def resolve_value(val: Any) -> Any:
  """Forces a `LazyThunk` or zero-arg function thunk, if present."""
  if is_thunk(val):
    return force_thunk(val)
  return val


def _force_in_scope(thunk: Thunk, owner_scope: EquationScope) -> Any:
  """Forces thunk, keeping outer-scope or unfinalized-handle refs deferred."""
  try:
    return thunk.__psyre_force__()
  except _DeferredError as err:
    if err.scope is owner_scope:
      raise NameError(str(err)) from err
    return thunk


def _resolve_tree(
    value: Any,
    owner_scope: EquationScope | None = None,
    active: set[int] | None = None,
    *,
    via_thunk: bool = False,
) -> Any:
  """Recursively forces `LazyThunk`s inside a PyTree of args or outputs."""
  # When `owner_scope` is provided (`Equations.__new__` finalization), keep
  # outer-scope/unfinalized-handle `Thunk`s deferred, unwrap
  # `_LazyBlockHandle`, and raise `CircularEquationError` on cyclic containers
  # reached via a thunk. When `owner_scope` is `None` (deferred-op argument
  # flattening), also descend into `slice` objects and force eagerly.
  if isinstance(value, _LazyBlockHandle):
    return value._handle  # pylint: disable=protected-access
  if owner_scope is not None:
    if isinstance(value, LazyThunk):
      resolved = (
          _force_in_scope(value, owner_scope)
          if isinstance(value, Thunk)
          else force_thunk(value)
      )
      if resolved is value:
        return value
      return _resolve_tree(resolved, owner_scope, active, via_thunk=True)
  else:
    if isinstance(value, LazyThunk):
      return force_thunk(value)
    if isinstance(value, slice):
      return slice(
          _resolve_tree(value.start, None, active),
          _resolve_tree(value.stop, None, active),
          _resolve_tree(value.step, None, active),
      )
  if isinstance(value, BlockHandle) or not isinstance(
      value, (tuple, list, Mapping)
  ):
    return value
  seen = set() if active is None else active
  obj_id = id(value)
  if obj_id in seen:
    if owner_scope is not None and via_thunk:
      raise CircularEquationError(
          'Instantaneous circular container reference while resolving '
          f'{type(value).__name__}.'
      )
    return value
  seen.add(obj_id)
  try:
    if isinstance(value, tuple):
      items = tuple(
          _resolve_tree(item, owner_scope, seen, via_thunk=via_thunk)
          for item in value
      )
      return type(value)(*items) if is_namedtuple(value) else items
    if isinstance(value, list):
      return [
          _resolve_tree(item, owner_scope, seen, via_thunk=via_thunk)
          for item in value
      ]
    resolved_dict = {
        k: _resolve_tree(v, owner_scope, seen, via_thunk=via_thunk)
        for k, v in value.items()
    }
    if isinstance(value, types.MappingProxyType):
      return types.MappingProxyType(resolved_dict)
    return resolved_dict
  finally:
    seen.discard(obj_id)


class Thunk(LazyThunk):
  """Memoized, cycle-checked, Stage-1-env-capturing deferred proxy."""

  __slots__ = ('_fn', '_desc', '_env', '_cache', '_resolving')
  __hash__ = None  # type: ignore[assignment]
  __array_priority__ = 1001

  def __init__(
      self,
      fn: Callable[[], Any],
      *,
      desc: str = '<thunk>',
      env: CapturedEnv | None = None,
  ):
    self._fn = fn
    self._desc = (
        desc
        if len(desc) <= _MAX_THUNK_DESC_LEN
        else f'{desc[:_MAX_THUNK_DESC_LEN - 3]}...'
    )
    self._env = CapturedEnv.capture() if env is None else env
    self._cache: Any = _UNRESOLVED
    self._resolving = False

  @override
  def __psyre_force__(self) -> Any:
    """Evaluates and memoizes the deferred expression in its captured env."""
    if self._cache is not _UNRESOLVED:
      return self._cache
    if self._resolving:
      raise CircularEquationError(
          f'Instantaneous circular reference while resolving {self._desc}.'
      )
    self._resolving = True
    try:
      with self._env.restore() if is_building() else contextlib.nullcontext():
        val = self._fn()
        # `force_thunk` rejects nested `LazyThunk` results, so a single force
        # step suffices (chained `Thunk` aliases resolve recursively inside).
        if isinstance(val, LazyThunk):
          val = force_thunk(val)
        if callable(val):
          raise TypeError(
              f'Higher-order forward reference {self._desc!r} resolved to a '
              f'callable ({type(val).__name__}); define callables before use.'
          )
      self._cache = val
      return val
    finally:
      self._resolving = False

  def __repr__(self) -> str:
    if self._cache is not _UNRESOLVED:
      return f'Thunk({self._desc} => {format_tag(self._cache)})'
    return f'Thunk({self._desc})'

  def __bool__(self) -> bool:
    raise TypeError(
        'Truth value of an unresolved Thunk is ambiguous; use Psyre '
        'combinators (e.g. ifthenelse, merge, when, Stream.max, or Stream.min).'
    )

  def __iter__(self) -> Iterator[Any]:  # pylint: disable=non-iterator-returned
    raise TypeError(
        f'Cannot iterate or unpack unresolved Thunk {self._desc!r}; index '
        'elements explicitly or define the tuple before unpacking.'
    )

  def __getattr__(self, name: str) -> Thunk:
    if _is_dunder(name):
      raise AttributeError(name)
    return Thunk(
        lambda: getattr(self.__psyre_force__(), name),
        desc=f'{self._desc}.{name}',
    )

  def __getitem__(self, key: Any) -> Thunk:
    return Thunk(
        lambda: self.__psyre_force__()[
            _resolve_tree(resolve_value(key))
        ],
        desc=f'{self._desc}[{_format_desc_operand(key)}]',
    )

  def __setitem__(self, key: Any, value: Any) -> None:
    raise TypeError(
        f'Cannot mutate unresolved forward reference {self._desc!r} via '
        f'[{key!r}] = ...'
    )

  def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
    del dtype, copy
    raise TypeError(
        f'Cannot convert unresolved Thunk {self._desc!r} to a NumPy array; '
        'use lifted NumPy operations inside equations instead.'
    )

  def __array_ufunc__(
      self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
  ) -> Any:
    """Defers NumPy ufunc dispatch (NEP-13) until equation resolution."""
    reject_numpy_inplace_ufunc(ufunc, method, kwargs)
    if method not in NUMPY_UFUNC_METHODS:
      return NotImplemented
    uname = getattr(ufunc, '__name__', 'ufunc')
    label = uname if method == '__call__' else f'{uname}.{method}'
    target = ufunc if method == '__call__' else getattr(ufunc, method)
    shared = _defer_call(target, label, inputs, kwargs)
    nout = getattr(ufunc, 'nout', 1) if method == '__call__' else 1
    if nout > 1:
      return tuple(shared[idx] for idx in range(nout))
    return shared

  def __array_function__(
      self,
      func: Any,
      types_arg: Any,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
  ) -> Thunk:
    """Defers NumPy array-function dispatch (NEP-18) until resolution."""
    del types_arg
    reject_numpy_mutating_func(func, kwargs)
    fname = getattr(func, '__name__', 'numpy_func')
    return _defer_call(func, fname, args, kwargs)

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    raise TypeError(
        f'Cannot call unresolved Thunk {self._desc!r}: higher-order forward '
        'references to callables are not supported; define callables before '
        'use.'
    )


def _defer_call(
    target: Callable[..., Any],
    label: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Thunk:
  """Builds a `Thunk` deferring `target(*args, **kwargs)` (dropping `out`)."""
  clean_kwargs = {k: v for k, v in kwargs.items() if k != 'out'}
  arg_desc = ', '.join(_format_desc_operand(arg) for arg in args)

  def _apply() -> Any:
    return target(
        *(_resolve_tree(arg) for arg in args),
        **{k: _resolve_tree(v) for k, v in clean_kwargs.items()},
    )

  return Thunk(_apply, desc=f'{label}({arg_desc})')


def _make_binop(
    op: Callable[[Any, Any], Any], *, reflected: bool = False
) -> Callable[[Thunk, Any], Thunk]:
  """Builds a deferred binary or reflected binary operator for `Thunk`."""

  def method(self: Thunk, other: Any) -> Thunk:
    desc = self._desc  # pylint: disable=protected-access
    other_desc = _format_desc_operand(other)
    if reflected:
      return Thunk(
          lambda: op(resolve_value(other), self.__psyre_force__()),
          desc=f'({other_desc} {op.__name__} {desc})',
      )
    return Thunk(
        lambda: op(self.__psyre_force__(), resolve_value(other)),
        desc=f'({desc} {op.__name__} {other_desc})',
    )

  return method


def _make_unaryop(op: Callable[[Any], Any]) -> Callable[[Thunk], Thunk]:
  """Builds a deferred unary operator for `Thunk`."""

  def method(self: Thunk) -> Thunk:
    desc = self._desc  # pylint: disable=protected-access
    return Thunk(
        lambda: op(self.__psyre_force__()),
        desc=f'({op.__name__} {desc})',
    )

  return method


for _op_name in (
    'add',
    'sub',
    'mul',
    'matmul',
    'truediv',
    'floordiv',
    'mod',
    'pow',
    'lshift',
    'rshift',
    'and',
    'or',
    'xor',
):
  _op_fn = getattr(
      operator, f'{_op_name}_' if _op_name in ('and', 'or') else _op_name
  )
  setattr(Thunk, f'__{_op_name}__', _make_binop(_op_fn))
  setattr(Thunk, f'__r{_op_name}__', _make_binop(_op_fn, reflected=True))
for _op_name in ('lt', 'le', 'eq', 'ne', 'gt', 'ge'):
  setattr(Thunk, f'__{_op_name}__', _make_binop(getattr(operator, _op_name)))
for _op_name in ('pos', 'neg', 'invert', 'abs'):
  setattr(Thunk, f'__{_op_name}__', _make_unaryop(getattr(operator, _op_name)))


class _LazyBlockHandle(Mapping[str, Any]):
  """Proxies a `BlockHandle` inside `EquationScope` for lazy forward reads."""

  __slots__ = ('_handle',)

  def __init__(self, handle: BlockHandle):
    object.__setattr__(self, '_handle', handle)

  def __getattr__(self, name: str) -> Any:
    if _is_dunder(name):
      raise AttributeError(name)
    handle = self._handle
    ctx = handle._ctx  # pylint: disable=protected-access
    if ctx.finalized:
      return getattr(handle, name)

    def _read_finalized(h: BlockHandle = handle, attr: str = name) -> Any:
      if not h._ctx.finalized:  # pylint: disable=protected-access
        raise _DeferredError(
            f'Output {attr!r} is not finalized until its block exits.'
        )
      return getattr(h, attr)

    return Thunk(_read_finalized, desc=f'<{ctx.kind}>.{name}')

  def __setattr__(self, name: str, value: Any) -> None:
    setattr(self._handle, name, value)

  def __getitem__(self, name: str) -> Any:
    return self._handle[name]

  def __iter__(self) -> Iterator[str]:
    return iter(self._handle)

  def __len__(self) -> int:
    return len(self._handle)


class EquationScope(dict):
  """Class namespace intercepting forward-referenced equations as `Thunk`s."""

  def __init__(
      self,
      initial: Mapping[str, Any] | None = None,
      *,
      auto_name: bool = True,
  ):
    super().__init__()
    self._auto_name = auto_name
    self._assigned_names: frozenset[str] | None = None
    self._forward_refs: dict[str, Thunk] = {}
    if initial:
      for key, val in initial.items():
        if auto_name and isinstance(val, Stream) and not val.name:
          self.name_stream(key, val)
        super().__setitem__(key, val)

  def _ensure_assigned_names(self) -> frozenset[str]:
    """Returns non-dunder names stored in the enclosing class body bytecode."""
    if self._assigned_names is None:
      # Scanning bytecode once lets `EquationScope.__missing__` intercept only
      # real forward-referenced equation bindings (`x = ...` later in the class
      # body) so genuine typos/undefined globals still raise `NameError`.
      # Frame 0: this method, Frame 1: __missing__, Frame 2: class body.
      frame = sys._getframe(2)  # pylint: disable=protected-access
      self._assigned_names = frozenset(
          instr.argval
          for instr in dis.get_instructions(frame.f_code)
          if instr.opname == 'STORE_NAME' and not _is_dunder(instr.argval)
      )
    return self._assigned_names

  def name_stream(self, key: str, stream: Stream) -> None:
    """Applies automatic Stage-1 stream/node naming for an equation binding."""
    if not key or key.startswith('_'):
      return
    builder = GraphBuilder.current()
    if builder is None:
      return
    cur_prefixes = prefix_stack.get()
    prefix = '/'.join(p for p, _ in cur_prefixes)
    qualified = f'{prefix}/{key}' if prefix else key
    prod = builder.graph.producer(stream)
    if prod is not None:
      leaf = prod.name.rsplit('/', 1)[-1]
      if prod.prefixes == cur_prefixes and (
          not stream.name or leaf == prod.op
      ):
        stream.name = prod.name = qualified
        return
    elif stream.name == qualified:
      return
    builder.add_stream(stream)
    builder.graph.add_stream_name(stream, qualified)

  def __missing__(self, key: str) -> Any:
    if _is_dunder(key) or key not in self._ensure_assigned_names():
      raise KeyError(key)
    thunk = self._forward_refs.get(key)
    if thunk is None:

      def _lookup(k: str = key) -> Any:
        if k not in self:
          raise _DeferredError(
              f'Forward-referenced equation {k!r} was never bound in scope.',
              scope=self,
          )
        val = dict.__getitem__(self, k)
        if isinstance(val, _LazyBlockHandle):
          return val._handle  # pylint: disable=protected-access
        return val

      thunk = Thunk(_lookup, desc=key)
      self._forward_refs[key] = thunk
    return thunk

  def __setitem__(self, key: str, value: Any) -> None:
    if not _is_dunder(key):
      if key in self._forward_refs:
        if key in self:
          raise RuntimeError(
              f'Forward-referenced equation {key!r} cannot be reassigned.'
          )
        if callable(value) and not isinstance(value, LazyThunk):
          raise TypeError(
              f'Forward-referenced equation {key!r} cannot bind a callable '
              f'({type(value).__name__}); define callables before use.'
          )
      if isinstance(value, BlockHandle):
        value = _LazyBlockHandle(value)
      elif self._auto_name and isinstance(value, Stream):
        self.name_stream(key, value)
    super().__setitem__(key, value)


@dataclasses.dataclass
class PsyreEquations:
  """Session handle yielded by `with psyre_equations(...) as prog:`."""

  output: Any = None
  graph: Graph | None = None
  last: type[Any] | None = None
  _owner_frame: types.FrameType | None = None


_active_equations_session: contextvars.ContextVar[PsyreEquations | None] = (
    contextvars.ContextVar('psyre_active_equations_session', default=None)
)
_build_class_lock = threading.Lock()
_build_class_refcount = 0
_orig_build_class: Any = None


def _make_build_class_wrapper(orig: Any) -> Any:
  """Returns a `builtins.__build_class__` wrapper closing over `orig`."""
  # Closing over `orig` (rather than reading `_orig_build_class` globally) lets
  # any thread that fetched `builtins.__build_class__` right before an exiting
  # `psyre_equations()` restores it keep constructing plain classes safely.

  def _equations_build_class(
      func: Any, name: str, /, *bases: Any, **kwds: Any
  ) -> Any:
    """Injects `metaclass=Equations` for direct classes in `psyre_equations()`."""
    session = _active_equations_session.get()
    if (
        session is not None
        and sys._getframe(1) is session._owner_frame  # pylint: disable=protected-access
        and not bases
        and 'metaclass' not in kwds
    ):
      kwds['metaclass'] = Equations
    return orig(func, name, *bases, **kwds)

  return _equations_build_class


@contextlib.contextmanager
def _build_class_hook() -> Iterator[None]:
  """Installs the ref-counted `builtins.__build_class__` wrapper."""
  global _build_class_refcount, _orig_build_class
  with _build_class_lock:
    if _build_class_refcount == 0:
      _orig_build_class = builtins.__build_class__
      builtins.__build_class__ = _make_build_class_wrapper(_orig_build_class)
    _build_class_refcount += 1
  try:
    yield
  finally:
    with _build_class_lock:
      _build_class_refcount -= 1
      if _build_class_refcount == 0:
        builtins.__build_class__ = _orig_build_class
        _orig_build_class = None


class Equations(type):
  """Metaclass turning a class body into a lambda-free Psyre equation block."""

  @classmethod
  def __prepare__(
      mcs,
      name: str,
      bases: tuple[type[Any], ...],
      /,
      *,
      auto_name: bool = True,
      **inputs: Any,
  ) -> EquationScope:
    del name, bases
    return EquationScope(inputs, auto_name=auto_name)

  def __new__(
      mcs,
      name: str,
      bases: tuple[type[Any], ...],
      namespace: EquationScope,
      /,
      *,
      auto_name: bool = True,
      **inputs: Any,
  ) -> Any:
    del inputs
    unbound = [k for k in namespace._forward_refs if k not in namespace]  # pylint: disable=protected-access
    if unbound:
      raise NameError(
          f'Unbound forward reference(s) in {name}: '
          + ', '.join(repr(k) for k in unbound)
      )

    def _resolve_binding(key: str, val: Thunk) -> None:
      resolved = _force_in_scope(val, namespace)
      if resolved is not val:
        resolved = _resolve_tree(resolved, namespace)
        if auto_name and isinstance(resolved, Stream):
          with val._env.restore():  # pylint: disable=protected-access
            namespace.name_stream(key, resolved)
        dict.__setitem__(namespace, key, resolved)

    # Resolve compound Thunk bindings before leaf forward-ref aliases (e.g.
    # `alias = target`) so concrete producer expressions claim automatic Stage-1
    # names first and alias bindings record extra stream aliases afterward.
    leaf_ids = {id(t) for t in namespace._forward_refs.values()}  # pylint: disable=protected-access
    leaf_bindings: list[tuple[str, Thunk]] = []
    container_bindings: list[tuple[str, Any]] = []
    for key in list(namespace.keys()):
      if _is_dunder(key):
        continue
      val = dict.__getitem__(namespace, key)
      if isinstance(val, _LazyBlockHandle):
        dict.__setitem__(namespace, key, val._handle)  # pylint: disable=protected-access
      elif isinstance(val, Thunk):
        if id(val) in leaf_ids:
          leaf_bindings.append((key, val))
        else:
          _resolve_binding(key, val)
      elif isinstance(val, (tuple, list, Mapping)) and not isinstance(
          val, BlockHandle
      ):
        container_bindings.append((key, val))
    for key, val in leaf_bindings:
      _resolve_binding(key, val)
    for key, val in container_bindings:
      dict.__setitem__(namespace, key, _resolve_tree(val, namespace))

    has_output = '__output__' in namespace
    if has_output:
      dict.__setitem__(
          namespace,
          '__output__',
          _resolve_tree(dict.__getitem__(namespace, '__output__'), namespace),
      )

    cls = super().__new__(mcs, name, bases, dict(namespace))
    session = _active_equations_session.get()
    if session is not None:
      session.last = cls
      if has_output:
        session.output = getattr(cls, '__output__')
    return cls

  def __init__(
      cls,
      name: str,
      bases: tuple[type[Any], ...],
      namespace: EquationScope,
      /,
      **kwds: Any,
  ):
    del kwds
    super().__init__(name, bases, namespace)


@contextlib.contextmanager
def psyre_equations(
    *,
    semantics: Semantics | None = None,
    optimize: bool | None = None,
    extra_roots: Iterable[Any] | None = None,
    name: str = '',
) -> Iterator[PsyreEquations]:
  """Context manager hosting lambda-free `Equations` blocks in Stage 1.

  Inside `with psyre_equations() as eq:`, any direct base-less `class`
  statement without an explicit `metaclass=` automatically receives
  `metaclass=Equations`. When entered outside an active `GraphBuilder`, builds,
  finalizes, and optionally optimizes a `Graph` (defaulting to `Semantics.SYNC`)
  and exposes it on `eq.graph`. When nested inside an active `GraphBuilder`,
  reuses it and exposes `__output__` on `eq.output`.

  Args:
    semantics: Execution semantics (`Semantics.SYNC` or `Semantics.KAHN`) for a
      standalone `Graph` (defaults to `Semantics.SYNC`). When nested inside an
      active `GraphBuilder`, passing a mismatched `semantics` raises
      `ValueError`.
    optimize: When standalone, whether to run `optimize_graph` (defaults to
      `True`). Cannot be specified when nested inside an active `GraphBuilder`.
    extra_roots: Optional extra roots retained by `optimize_graph` when
      standalone. Cannot be specified when nested inside an active
      `GraphBuilder`.
    name: Optional hierarchical prefix pushed for the duration of the block.

  Yields:
    A `PsyreEquations` session handle exposing `output`, `graph`, and `last`.

  Raises:
    TypeError: If `name` is not a `str`, `semantics` is not a `Semantics`, or
      `optimize` is not a `bool`.
    ValueError: If nested inside an active `GraphBuilder` with conflicting
      `semantics`, or with explicit `optimize`/`extra_roots`.
  """
  if not isinstance(name, str):
    raise TypeError(
        f'psyre_equations(name=...) must be a str, got {type(name).__name__}.'
    )
  if semantics is not None and not isinstance(semantics, Semantics):
    raise TypeError(
        'psyre_equations(semantics=...) must be a Semantics value, got '
        f'{type(semantics).__name__}.'
    )
  if optimize is not None and not isinstance(optimize, bool):
    raise TypeError(
        'psyre_equations(optimize=...) must be a bool, got '
        f'{type(optimize).__name__}.'
    )
  # Walk past `contextlib` frames so `ExitStack.enter_context()` finds caller.
  owner_frame: types.FrameType | None = sys._getframe(1)  # pylint: disable=protected-access
  while (
      owner_frame is not None
      and owner_frame.f_globals.get('__name__') == 'contextlib'
  ):
    owner_frame = owner_frame.f_back
  existing = active_builder.get()
  owns_builder = existing is None
  if not owns_builder:
    if semantics is not None and semantics != existing.semantics:
      raise ValueError(
          f'Nested psyre_equations(semantics={semantics}) conflicts with '
          f'active GraphBuilder semantics ({existing.semantics}).'
      )
    if optimize is not None:
      raise ValueError(
          'optimize cannot be set on a nested psyre_equations() block; '
          'configure optimization on the enclosing GraphBuilder.'
      )
    if extra_roots is not None:
      raise ValueError(
          'extra_roots cannot be set on a nested psyre_equations() block; '
          'pass extra_roots to the enclosing GraphBuilder/optimize call.'
      )
  eff_semantics = Semantics.SYNC if semantics is None else semantics
  eff_optimize = True if optimize is None else optimize
  builder = GraphBuilder(semantics=eff_semantics) if owns_builder else existing
  session = PsyreEquations(
      graph=None if owns_builder else builder.graph,
      _owner_frame=owner_frame,
  )
  graph: Graph | None = None
  with contextlib.ExitStack() as stack:
    if owns_builder:
      builder_tok = active_builder.set(builder)
      stack.callback(active_builder.reset, builder_tok)
      stack.enter_context(CapturedEnv().restore())
    session_tok = _active_equations_session.set(session)
    stack.callback(_active_equations_session.reset, session_tok)
    stack.enter_context(_build_class_hook())
    try:
      with push_prefix(name) if name else contextlib.nullcontext():
        yield session
      if owns_builder:
        builder.bind_output(session.output)
        graph = builder.finalize()
    finally:
      session._owner_frame = None  # pylint: disable=protected-access
  if owns_builder and graph is not None:
    if eff_optimize:
      optimize_graph(graph, extra_roots=extra_roots)
    session.graph = graph
