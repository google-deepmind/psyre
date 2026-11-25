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
"""Stream-level NumPy lifting helpers, proxies, and eager batch mapping."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import functools
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  import numpy as np
except ImportError as err:  # pragma: no cover
  raise ImportError(
      'psyre.numpy requires NumPy. Install it with: '
      'pip install "psyre[numpy]"'
  ) from err

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    LazyThunk,
    emit_comb_node,
    is_building,
    make_comb_step_func,
    push_prefix,
    require_builder,
    unwrap_inst_func,
    validate_positive_int,
)
from psyre.payload import (
    ARRAY_LINALG_NOUT,
    LiftedNamespace,
    is_callable_target,
    is_from_package,
    is_pure_numpy_func,
    is_stream_obj,
    is_zero_arg_function,
    make_batch_streams,
    make_stack_streams,
    make_unstack_stream,
    normalize_arg_axes,
    normalize_static_argnames,
    normalize_static_argnums,
    partition_comb_args,
    reject_numpy_inplace_ufunc,
    reject_numpy_mutating_func,
    resolve_nout,
    stack_output_trees,
    strip_null_out,
    ufunc_nout,
    validate_axis,
    validate_optional_axis,
)

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import

# NumPy array-creation routines that reject wrapping a bare `Stream` / thunk
# directly (mirroring `Stream.__array__`). Matched by object identity so a user
# function named `array` lifted via `lift(array)` is unaffected.
_ARRAY_COERCION_NAMES = (
    'array',
    'asanyarray',
    'asarray',
    'asarray_chkfinite',
    'ascontiguousarray',
    'asfortranarray',
    'asmatrix',
)


def _collect_coercion_ids() -> frozenset[int]:
  ids: list[int] = []
  for attr_name in _ARRAY_COERCION_NAMES:
    val = getattr(np, attr_name, None)
    if val is not None:
      ids.append(id(val))
  return frozenset(ids)


_ARRAY_COERCION_IDS = _collect_coercion_ids()

# Legacy `numpy.random` helpers managing RNG state or returning generators
# rather than per-tick random samples.
_NON_STREAM_RANDOM_FUNCS = frozenset({
    'default_rng',
    'get_bit_generator',
    'get_state',
    'seed',
    'set_bit_generator',
    'set_state',
    'spawn',
})
_MUTATING_RANDOM_FUNCS = frozenset({'shuffle'})
_RNG_OWNER_TYPES = (np.random.RandomState, np.random.Generator)


def _is_rng_owner(fn: Any) -> bool:
  return isinstance(getattr(fn, '__self__', None), _RNG_OWNER_TYPES)


def _is_random_sampler(fn: Any) -> bool:
  """Returns True for `numpy.random` sampling functions (not RNG managers)."""
  fname = getattr(fn, '__name__', '')
  if fname in _NON_STREAM_RANDOM_FUNCS or fname in _MUTATING_RANDOM_FUNCS:
    return False
  if _is_rng_owner(fn):
    return True
  mod = getattr(fn, '__module__', '') or ''
  return mod == 'numpy.random' or mod.startswith('numpy.random.')


def _is_pure_default(fn: Any) -> bool:
  """Returns whether `fn` defaults to `pure=True` for Stage-1 CSE."""
  if isinstance(fn, np.ufunc):
    return True
  if _is_rng_owner(fn):
    return False
  return is_pure_numpy_func(fn)


def _has_stream_or_ufunc_thunk(inputs: Iterable[Any]) -> bool:
  return any(
      is_stream_obj(x)
      or (
          isinstance(x, LazyThunk)
          and getattr(type(x), '__array_ufunc__', None) is not None
      )
      for x in inputs
  )


def _contains_zero_arg_thunk(dyn_inputs: Iterable[Any]) -> bool:
  return any(is_zero_arg_function(x) for x in dyn_inputs)


class LiftedUfunc:
  """Stream-lifted proxy around a NumPy `np.ufunc` (rejects `.at` and `out=`).

  Delegates plain NumPy/Python arguments eagerly to `ufunc` (zero-stream
  polymorphic fast path), delegates top-level `Stream` / `LazyThunk` inputs via
  NEP-13, and handles nested/zero-arg thunk inputs via `CombCallPlan`.
  """

  def __init__(self, ufunc: np.ufunc):
    self._ufunc = ufunc
    self._psyre_inst_func = ufunc
    self._psyre_pure = True
    self.__name__ = ufunc.__name__
    self.__doc__ = getattr(ufunc, '__doc__', None)

  def _invoke(
      self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
  ) -> Any:
    """Dispatches a ufunc method on plain values, `Stream`s, or thunks."""
    reject_numpy_inplace_ufunc(self._ufunc, method, kwargs)
    clean_kwargs = strip_null_out(kwargs)
    dyn_inputs, node_kwargs = partition_comb_args(
        args, clean_kwargs, static_literals=True
    )
    target = (
        self._ufunc if method == '__call__' else getattr(self._ufunc, method)
    )
    if not dyn_inputs:
      return target(*args, **clean_kwargs)
    # Delegate straight through NumPy NEP-13 when all dynamic leaves are
    # top-level `Stream`/`LazyThunk` (no bare `lambda:` thunks).
    top_vals = (
        (*args, clean_kwargs['where']) if 'where' in clean_kwargs else args
    )
    if (
        not _contains_zero_arg_thunk(dyn_inputs)
        and _has_stream_or_ufunc_thunk(top_vals)
        and sum(
            1 for a in top_vals if is_stream_obj(a) or isinstance(a, LazyThunk)
        )
        == len(dyn_inputs)
        and all(
            is_stream_obj(x)
            or (
                isinstance(x, LazyThunk)
                and getattr(type(x), '__array_ufunc__', None) is not None
            )
            for x in dyn_inputs
        )
    ):
      return target(*args, **clean_kwargs)
    op_name = (
        self._ufunc.__name__
        if method == '__call__'
        else f'{self._ufunc.__name__}.{method}'
    )
    step_func = make_comb_step_func(target, name=op_name, pure=True)
    nout = ufunc_nout(self._ufunc, method)
    return emit_comb_node(
        step_func, dyn_inputs, node_kwargs, name=op_name, nout=nout
    )

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self._invoke('__call__', args, kwargs)

  def reduce(self, *args: Any, **kwargs: Any) -> Any:
    return self._invoke('reduce', args, kwargs)

  def accumulate(self, *args: Any, **kwargs: Any) -> Any:
    return self._invoke('accumulate', args, kwargs)

  def reduceat(self, *args: Any, **kwargs: Any) -> Any:
    return self._invoke('reduceat', args, kwargs)

  def outer(self, *args: Any, **kwargs: Any) -> Any:
    return self._invoke('outer', args, kwargs)

  def at(self, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    reject_numpy_inplace_ufunc(self._ufunc, 'at', {})

  def __getattr__(self, name: str) -> Any:
    return getattr(self._ufunc, name)

  def __repr__(self) -> str:
    return f'<LiftedUfunc {self._ufunc.__name__!r}>'


def _infer_nout(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    explicit_nout: int | None,
) -> int:
  """Infers output tuple arity for a NumPy ufunc or `numpy.linalg` routine."""
  if explicit_nout is not None:
    return explicit_nout
  if isinstance(fn, np.ufunc):
    return fn.nout
  mod = getattr(fn, '__module__', '') or ''
  if mod.startswith('numpy.linalg'):
    spec = ARRAY_LINALG_NOUT.get(getattr(fn, '__name__', ''))
    return resolve_nout(spec, args, kwargs)
  return 1


def _can_delegate_nep18(
    fn: Callable[..., Any],
    dyn_inputs: tuple[Any, ...],
    explicit_pure: bool | None,
    nout: int,
) -> bool:
  """Returns True when delegating to `fn` invokes NEP-18 cleanly and matches."""
  if explicit_pure is not None or nout > 1:
    return False
  if _contains_zero_arg_thunk(dyn_inputs):
    return False
  if getattr(fn, '_implementation', None) is None and not hasattr(
      fn, '__wrapped__'
  ):
    return False
  return all(
      is_stream_obj(x)
      or (
          isinstance(x, LazyThunk)
          and getattr(type(x), '__array_function__', None) is not None
      )
      for x in dyn_inputs
  )


def lift(
    fn: Callable[..., Any],
    *,
    pure: bool | None = None,
    nout: int | None = None,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    static_literals: bool = True,
    name: str = '',
) -> Any:
  """Lifts a NumPy ufunc or callable onto Psyre `Stream`s.

  When invoked with *no* dynamic `Stream` or Stage-1 thunk arguments, pure
  functions execute eagerly and return a plain NumPy value/array so constants
  stay clock-polymorphic inside `match`/`automaton` arms. Impure `numpy.random`
  functions invoked during Stage 1 with no dynamic inputs emit a 0-input stream
  node producing a fresh sample on every tick.

  In-place `out=` destination arguments and known mutating NumPy functions
  (`copyto`, `fill_diagonal`, `place`, `put`, `put_along_axis`, `putmask`,
  `random.shuffle`) are rejected with `TypeError`.

  Args:
    fn: NumPy `ufunc`, module function, or user callable to lift.
    pure: Optional purity override for `optimize_graph` CSE. Defaults to `True`
      for deterministic, non-I/O NumPy functions and `False` for `empty`,
      `empty_like`, `numpy.random.*`, I/O ops, and arbitrary non-NumPy
      callables.
    nout: Optional positive output arity for multi-output unpacking into a
      tuple of `Stream`s (inferred automatically for `np.ufunc` and common
      `numpy.linalg` routines such as `qr`, `svd`, `eigh`, `eig`, `slogdet`).
    static_argnums: Zero-based positional indices kept verbatim as static args.
    static_argnames: Keyword argument names kept verbatim as static args.
    static_literals: When `True` (default), non-stream positional literals stay
      verbatim in `CombCallPlan` instead of becoming periodic sequences.
    name: Optional Stage-1 node name override.

  Returns:
    A lifted callable (or `LiftedUfunc` for bare NumPy ufuncs with default
    options).

  Raises:
    TypeError: If `fn` is not callable.
    ValueError: If `nout` is not a positive integer.
  """
  if not is_callable_target(fn):
    raise TypeError(f'psyre.numpy.lift expects a callable, got {type(fn)}.')
  if nout is not None:
    validate_positive_int(nout, 'psyre.numpy.lift nout')
  raw_fn = unwrap_inst_func(fn)
  argnums = normalize_static_argnums(static_argnums)
  argnames = normalize_static_argnames(static_argnames)
  if (
      isinstance(raw_fn, np.ufunc)
      and pure is None
      and nout is None
      and not argnums
      and not argnames
      and static_literals
      and not name
  ):
    return LiftedUfunc(raw_fn)

  is_pure = _is_pure_default(raw_fn) if pure is None else bool(pure)
  op_name = name or getattr(raw_fn, '__name__', 'numpy_op')
  is_coercion = id(raw_fn) in _ARRAY_COERCION_IDS
  is_numpy_fn = (
      is_from_package(raw_fn, 'numpy')
      or isinstance(raw_fn, np.ufunc)
      or _is_rng_owner(raw_fn)
  )
  is_random_mutator = getattr(
      raw_fn, '__name__', ''
  ) in _MUTATING_RANDOM_FUNCS and (
      _is_rng_owner(raw_fn)
      or (getattr(raw_fn, '__module__', '') or '').startswith('numpy.random')
  )
  is_random_gen = _is_random_sampler(raw_fn)
  step_func = make_comb_step_func(raw_fn, name=op_name, pure=is_pure)

  @functools.wraps(raw_fn)
  def wrapper(*args: Any, **kwargs: Any) -> Any:
    if is_random_mutator:
      raise TypeError(
          f'In-place NumPy random function {op_name!r} is not supported on '
          'Psyre Streams; use permutation() instead.'
      )
    if is_numpy_fn:
      reject_numpy_mutating_func(raw_fn, kwargs)
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
      if is_random_gen and is_building():
        return emit_comb_node(
            step_func, (), node_kwargs, name=op_name, nout=nout or 1
        )
      return raw_fn(*args, **clean_kwargs)
    if is_coercion:
      raise TypeError(
          f'Cannot convert a Psyre Stream via {op_name}(); use '
          'psyre.numpy.stack_streams(...) to combine streams of arrays.'
      )
    out_arity = _infer_nout(raw_fn, args, clean_kwargs, nout)
    if (
        not argnums
        and not argnames
        and static_literals
        and not name
        and _can_delegate_nep18(raw_fn, dyn_inputs, pure, out_arity)
    ):
      try:
        return raw_fn(*args, **clean_kwargs)
      except TypeError as exc:
        msg = str(exc)
        if not (
            'Cannot convert a Psyre Stream' in msg
            or 'Cannot convert unresolved Thunk' in msg
            or 'no implementation found for' in msg
        ):
          raise
    return emit_comb_node(
        step_func, dyn_inputs, node_kwargs, name=op_name, nout=out_arity
    )

  setattr(wrapper, '_psyre_step_func', step_func)
  setattr(wrapper, '_psyre_inst_func', raw_fn)
  setattr(wrapper, '_psyre_pure', is_pure)
  if argnums:
    setattr(wrapper, '_psyre_static_argnums', argnums)
  if argnames:
    setattr(wrapper, '_psyre_static_argnames', argnames)
  if nout is not None:
    setattr(wrapper, '_psyre_nout', nout)
  return wrapper


def _wrap_numpy_attr(name: str, attr: Any) -> Any:
  del name
  if isinstance(attr, np.ufunc):
    return LiftedUfunc(attr)
  if callable(attr) and not isinstance(attr, type):
    return lift(attr)
  return attr


linalg = LiftedNamespace(
    np.linalg, 'psyre.numpy.linalg', wrap_attr=_wrap_numpy_attr
)
fft = LiftedNamespace(np.fft, 'psyre.numpy.fft', wrap_attr=_wrap_numpy_attr)
random = LiftedNamespace(
    np.random, 'psyre.numpy.random', wrap_attr=_wrap_numpy_attr
)
_ROOT = LiftedNamespace(
    np,
    'psyre.numpy',
    wrap_attr=_wrap_numpy_attr,
    submodules={'linalg': linalg, 'fft': fft, 'random': random},
)


def get_attr(name: str) -> Any:
  """Resolves an attribute lazily on the lifted `numpy` namespace."""
  return getattr(_ROOT, name)


def dir_names() -> list[str]:
  """Returns discoverable attribute names on `psyre.numpy`."""
  return dir(_ROOT)


stack_streams = make_stack_streams(
    np.stack,
    make_comb_step_func,
    emit_comb_node,
    doc=(
        'Stacks one or more array-valued `Stream`s along `axis` via '
        '`np.stack`.'
    ),
)

unstack_stream = make_unstack_stream(
    np.asarray,
    np.moveaxis,
    make_comb_step_func,
    emit_comb_node,
    doc='Unstacks an array-valued `Stream` into a `size`-tuple of `Stream`s.',
)

batch_streams = make_batch_streams(
    unstack_stream,
    stack_streams,
    require_builder,
    push_prefix,
    doc=(
        'Structural Stage-1 lane expansion instantiating `size` independent '
        'copies of `fn` on unstacked `np.ndarray` `Stream` lanes.'
    ),
)


def vmap(
    fn: Callable[..., Any],
    in_axes: int | None | tuple[Any, ...] | list[Any] = 0,
    out_axes: int = 0,
    *,
    pure: bool = True,
    nout: int | None = None,
    name: str = '',
) -> Callable[..., Any]:
  """Eager NumPy pointwise vectorization over batched array `Stream`s.

  Maps a pure per-example Python/NumPy function `fn` across a leading (or
  specified) batch axis at each Stage-2 tick without requiring JAX. Supports
  tuple/list/dict outputs stacked along `out_axes`.

  Args:
    fn: Per-example callable (or lifted `@psyre_comb` wrapper).
    in_axes: Batch axis (`int` or `None` to broadcast) for all positional
      arguments, or a per-argument tuple/list matching the call arity.
    out_axes: Destination batch axis for each stacked output array (`int`).
    pure: Purity flag for `optimize_graph` CSE (defaults to `True`).
    nout: Optional positive output arity unpacking a tuple output stream into
      an `nout`-tuple of `Stream`s.
    name: Optional Stage-1 node name override.

  Returns:
    A lifted callable returning a batched `Stream` (or eager `np.ndarray`/tree
    when invoked without any `Stream`/thunk arguments).
  """
  if not is_callable_target(fn):
    raise TypeError(f'vmap expects a callable, got {type(fn).__name__}.')
  validate_axis(out_axes, 'out_axes')
  if nout is not None:
    validate_positive_int(nout, 'psyre.numpy.vmap nout')
  raw_fn = unwrap_inst_func(fn)
  fn_name = name or f'vmap({getattr(raw_fn, "__name__", "fn")})'

  def _vmapped(*vals: Any, **kwargs: Any) -> Any:
    axes = tuple(
        validate_optional_axis(ax, 'in_axes')
        for ax in normalize_arg_axes(in_axes, len(vals), 'vmap in_axes')
    )
    batch_size: int | None = None
    moved_vals: list[Any] = []
    for pos, (val, ax) in enumerate(zip(vals, axes)):
      if ax is None:
        moved_vals.append(val)
        continue
      arr = np.asarray(val)
      if arr.ndim == 0:
        raise ValueError(
            f'vmap argument {pos} cannot be mapped along axis {ax} '
            '(scalar has 0 dimensions).'
        )
      moved = np.moveaxis(arr, ax, 0)
      cur_size = moved.shape[0]
      if batch_size is None:
        batch_size = cur_size
      elif cur_size != batch_size:
        raise ValueError(
            f'vmap got inconsistent batch sizes: {batch_size} vs {cur_size} '
            f'at argument {pos}.'
        )
      moved_vals.append(moved)
    if batch_size is None:
      raise ValueError('vmap requires at least one mapped input axis.')
    if batch_size == 0:
      raise ValueError('vmap does not support empty batch size 0.')
    results = [
        raw_fn(
            *(
                mv if ax is None else mv[i]
                for mv, ax in zip(moved_vals, axes)
            ),
            **kwargs,
        )
        for i in range(batch_size)
    ]
    return stack_output_trees(
        results, lambda items, _: np.stack(items, axis=out_axes)
    )

  _vmapped.__name__ = fn_name
  step_func = make_comb_step_func(_vmapped, name=fn_name, pure=pure)

  def wrapper(*args: Any, **kwargs: Any) -> Any:
    axes = tuple(
        validate_optional_axis(ax, 'in_axes')
        for ax in normalize_arg_axes(in_axes, len(args), 'vmap in_axes')
    )
    if all(ax is None for ax in axes):
      raise ValueError('vmap requires at least one mapped input axis.')
    dyn_inputs, node_kwargs = partition_comb_args(
        args, kwargs, static_literals=True
    )
    if not dyn_inputs:
      return _vmapped(*args, **kwargs)
    return emit_comb_node(
        step_func, dyn_inputs, node_kwargs, name=fn_name, nout=nout or 1
    )

  setattr(wrapper, '_psyre_step_func', step_func)
  setattr(wrapper, '_psyre_inst_func', _vmapped)
  if pure:
    setattr(wrapper, '_psyre_pure', True)
  return wrapper
