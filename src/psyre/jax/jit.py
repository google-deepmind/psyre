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
"""`jax.jit` lifting, convex island fusion, and opt-in `Jaxpr` expansion.

Implements both tracing modes behind `psyre.jax.jit`:

1. **Direction A (default, `trace_jaxpr=False`)**:
   - Single-callable lazy compilation via `jax.jit`: **zero** Stage-1 tracing
     and no shape/dtype hints required. Inactive `Absent` ticks short-circuit
     before entering XLA.
   - Whole-graph island fusion (`fuse_jit_islands` / `jit_graph`): contracts
     maximal same-clock **convex** combinational DAGs of pure JAX/lifted-`lax`
     nodes into single `jax.jit`-compiled `@psyre_comb(pure=True)` super-nodes
     while keeping `fby`/`pre`/clocks/`when`/`merge`/`ifthenelse`/host-object
     boundaries intact and guaranteeing zero cycles.

2. **Direction B (opt-in, `trace_jaxpr=True`)**:
   - Strictly gated behind explicit **`jax.ShapeDtypeStruct`** hints passed via
     `in_avals` at Stage 1 (no speculative tracing or deferred post-tick
     capture).
   - Uses `jax.make_jaxpr` to trace pure functions (rejecting effectful
     `Jaxpr`s with `ImpureJaxprError`), and expands each `JaxprEqn` via
     `expand_jaxpr` into a fine-grained `@psyre_comb(pure=True)` `Node` carrying
     its canonical JAX `Primitive` tag inside a `push_prefix(name)` scope.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterable, Mapping
import contextlib
import dataclasses
import functools
import math
import operator
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  import jax
except ImportError as err:  # pragma: no cover
  raise ImportError(
      'psyre.jax requires JAX. Install it with: pip install "psyre[jax]"'
  ) from err

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import causal_node_order, check_causality
from psyre.clock import Clock, check_clocks
from psyre.common import (
    active_builder,
    callable_name,
    emit_comb_node,
    is_pure_step_func,
    make_comb_step_func,
    push_prefix,
    require_builder,
    unwrap_inst_func,
    validate_positive_int,
)
from psyre.graph import Graph, Node, iter_output_leaves, map_output_leaves
from psyre.jax.ops import (
    attach_wrapper_meta,
    is_jax_step_func,
    make_jax_step_func,
    read_wrapper_nout_spec,
)
from psyre.payload import (
    CALL_PLAN_KWARG,
    is_callable_target,
    is_dynamic_leaf,
    is_nan_scalar,
    is_numpy_scalar,
    materialize_comb_call,
    normalize_static_argnames,
    normalize_static_argnums,
    partition_comb_args,
    resolve_nout,
    zero_sign_key,
)
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import

_NON_FUSIBLE_PRIMITIVES = frozenset({
    'buffer',
    'counter',
    'current',
    'fby',
    'ifthenelse',
    'merge',
    'morph',
    'post',
    'pre',
    'prefix_sum',
    'scan',
    'switch',
    'when',
    'whennot',
})

_SUBJAXPR_JIT_NAMES = frozenset({'pjit', 'jit', 'xla_call'})


class ImpureJaxprError(ValueError):
  """Raised when `expand_jaxpr` / `jit(..., trace_jaxpr=True)` sees effects."""


@dataclasses.dataclass(frozen=True)
class JitIslandInfo:
  """Metadata summarizing one fused same-clock `jax.jit` combinational island.

  Attributes:
    name: Name of the emitted fused super-node in the returned `Graph`.
    clock: Synchronous `Clock` shared by all nodes in the island.
    node_names: Ordered causal tuple of original node names fused into the
      island.
    num_inputs: Number of distinct external input `Stream` ports.
    num_outputs: Number of distinct exit `Stream` ports read outside the island.
  """

  name: str
  clock: Clock
  node_names: tuple[str, ...]
  num_inputs: int
  num_outputs: int

  @property
  def num_nodes(self) -> int:
    """Number of original primitive nodes contracted into this island."""
    return len(self.node_names)


def _is_literal_atom(atom: Any) -> bool:
  """Returns True when a `Jaxpr` equation atom is a `Literal` constant."""
  return hasattr(atom, 'val')


def _is_drop_var(var: Any) -> bool:
  """Returns True when a `Jaxpr` output binder is a discarded `DropVar`."""
  return type(var).__name__ == 'DropVar'


def _unwrap_jaxpr(closed_or_jaxpr: Any) -> tuple[Any, tuple[Any, ...]]:
  """Normalizes `ClosedJaxpr` / unified `Jaxpr` to `(jaxpr, consts)`."""
  inner = getattr(closed_or_jaxpr, 'jaxpr', None)
  if inner is not None and inner is not closed_or_jaxpr:
    consts = tuple(getattr(closed_or_jaxpr, 'consts', ()))
    return inner, consts
  consts = tuple(getattr(closed_or_jaxpr, 'consts', ()))
  return closed_or_jaxpr, consts


def _raise_if_effects(obj: Any, *, label: str, detail: str = '') -> None:
  effects = getattr(obj, 'effects', frozenset())
  if effects:
    names = ', '.join(sorted(repr(e) for e in effects))
    suffix = (
        f'primitive {detail!r} carries effects'
        if detail
        else 'non-empty effects'
    )
    raise ImpureJaxprError(
        f'Cannot expand impure Jaxpr {label!r}: {suffix} {{{names}}}.'
    )


def _check_pure_jaxpr(jaxpr_obj: Any, *, label: str) -> None:
  """Raises `ImpureJaxprError` when a `Jaxpr` (or nested eqn) has effects."""
  _raise_if_effects(jaxpr_obj, label=label)
  raw_jaxpr, _ = _unwrap_jaxpr(jaxpr_obj)
  _raise_if_effects(raw_jaxpr, label=label)
  for eqn in raw_jaxpr.eqns:
    _raise_if_effects(eqn, label=label, detail=eqn.primitive.name)
    for param_val in eqn.params.values():
      items = (
          param_val if isinstance(param_val, (tuple, list)) else (param_val,)
      )
      for item in items:
        if hasattr(item, 'eqns') or hasattr(item, 'jaxpr'):
          _check_pure_jaxpr(item, label=f'{label}/{eqn.primitive.name}')


_EQN_STEP_CACHE: collections.OrderedDict[Any, Callable[..., Any]] = (
    collections.OrderedDict()
)
_EQN_STEP_CACHE_MAX = 2048
_NON_TRACED_STREAM_OPS = frozenset({'max', 'min'})


def _freeze_param(val: Any) -> Any:
  """Converts a `JaxprEqn` param value into a hashable, dtype-safe key."""
  if is_numpy_scalar(val):
    if is_nan_scalar(val):
      raise TypeError('NaN parameter not cacheable')
    return (
        type(val),
        getattr(val, 'dtype', None),
        val.item(),
        zero_sign_key(val),
    )
  if isinstance(val, (bool, int)):
    return (type(val), getattr(val, 'dtype', None), int(val))
  if isinstance(val, float):
    if math.isnan(val):
      raise TypeError('NaN parameter not cacheable')
    return (type(val), float(val), zero_sign_key(val))
  if isinstance(val, (str, bytes, type(None))):
    return val
  if isinstance(val, tuple):
    return tuple(_freeze_param(x) for x in val)
  if isinstance(val, list):
    return ('__list__', tuple(_freeze_param(x) for x in val))
  if isinstance(val, Mapping):
    return (
        '__map__',
        tuple(sorted((str(k), _freeze_param(v)) for k, v in val.items())),
    )
  hash(val)
  return val


def _make_eqn_evaluator(
    prim: Any, raw_params: Mapping[str, Any], eqn_ctx: Any
) -> Callable[..., Any]:
  """Builds a per-tick callable evaluating a single `JaxprEqn`."""
  multi = bool(prim.multiple_results)

  def _eval_eqn(*in_vals: Any, **override_params: Any) -> Any:
    if override_params:
      params = dict(raw_params)
      for k, v in override_params.items():
        if params.get(k) is None:
          params[k] = v
    else:
      params = raw_params
    if hasattr(prim, 'get_bind_params'):
      bound = prim.get_bind_params(params)
      if isinstance(bound, tuple) and len(bound) == 2:
        subfuns, bind_params = bound
      else:
        subfuns, bind_params = (), bound
    else:
      subfuns, bind_params = (), params
    ctx_mgr = (
        getattr(eqn_ctx, 'manager', None) if eqn_ctx is not None else None
    )
    with contextlib.nullcontext() if ctx_mgr is None else ctx_mgr:
      ans = prim.bind(*subfuns, *in_vals, **bind_params)
    if multi:
      return tuple(ans)
    return ans

  _eval_eqn.__name__ = prim.name
  return _eval_eqn


def _eqn_step_func(eqn: Any) -> Callable[..., Any]:
  """Returns a cached `step_func` for `eqn` so `optimize_graph` CSE works."""
  prim = eqn.primitive
  raw_params = dict(eqn.params)
  eqn_ctx = getattr(eqn, 'ctx', None)
  try:
    frozen = tuple(
        sorted((k, _freeze_param(v)) for k, v in raw_params.items())
    )
    cache_key = (prim, frozen, eqn_ctx)
    cached = _EQN_STEP_CACHE.get(cache_key)
    if cached is not None:
      _EQN_STEP_CACHE.move_to_end(cache_key)
      return cached
  except TypeError:
    cache_key = None
  step_fn = make_jax_step_func(
      _make_eqn_evaluator(prim, raw_params, eqn_ctx),
      name=prim.name,
      pure=True,
      primitive=prim,
  )
  if cache_key is not None:
    if len(_EQN_STEP_CACHE) >= _EQN_STEP_CACHE_MAX:
      _EQN_STEP_CACHE.popitem(last=False)
    _EQN_STEP_CACHE[cache_key] = step_fn
  return step_fn


def _as_dynamic_stream(val: Any) -> Stream:
  return val if is_dynamic_leaf(val) else Stream.constant(val)


@functools.lru_cache(maxsize=128)
def _proj_step_func(
    index: int, *, jax_tagged: bool = True
) -> Callable[..., Any]:
  """Returns a cached tuple-projection `step_func` for multi-output nodes."""
  def _proj(bundle: Any) -> Any:
    return operator.getitem(bundle, index)

  label = f'island_out_{index}'
  _proj.__name__ = label
  if jax_tagged:
    return make_jax_step_func(_proj, name=label, pure=True)
  # Non-JAX-tagged projection keeps multi-exit `fuse_jit_islands` idempotent.
  return make_comb_step_func(_proj, name=label, pure=True)


def _literal_cache_key(atom: Any) -> Any:
  """Returns a dtype- and signed-zero-safe cache key, or `None` if uncached."""
  val = atom.val
  aval_dt = getattr(getattr(atom, 'aval', None), 'dtype', None)
  val_dt = getattr(val, 'dtype', None)
  if is_numpy_scalar(val) and not is_nan_scalar(val):
    return (type(val), val_dt, aval_dt, val.item(), zero_sign_key(val))
  if isinstance(val, (bool, int)):
    return (type(val), val_dt, aval_dt, int(val))
  if isinstance(val, float) and not math.isnan(val):
    return (type(val), val_dt, aval_dt, float(val), zero_sign_key(val))
  return None


def _expand_eqns(
    closed_or_jaxpr: Any,
    in_streams: tuple[Any, ...],
    *,
    expand_subjaxprs: bool,
    scope_name: str,
) -> list[Stream]:
  """Expands a checked pure `Jaxpr` into `Stream` nodes on active builder."""
  raw_jaxpr, consts = _unwrap_jaxpr(closed_or_jaxpr)
  constvars = tuple(getattr(raw_jaxpr, 'constvars', ()))
  invars = tuple(raw_jaxpr.invars)
  if len(consts) != len(constvars):
    raise ValueError(
        f'Jaxpr {scope_name!r} has {len(constvars)} constvars but '
        f'{len(consts)} const values.'
    )
  if len(in_streams) != len(invars):
    raise ValueError(
        f'Jaxpr {scope_name!r} expects {len(invars)} input leaves, got '
        f'{len(in_streams)}.'
    )

  env: dict[Any, Any] = {}
  const_env: dict[Any, Any] = {}
  const_cache: dict[Any, Stream] = {}
  for cvar, cval in zip(constvars, consts):
    env[cvar] = Stream.constant(cval)
    const_env[cvar] = cval
  for ivar, istream in zip(invars, in_streams):
    env[ivar] = _as_dynamic_stream(istream)

  def _read_atom(atom: Any) -> Any:
    if _is_literal_atom(atom):
      key = _literal_cache_key(atom)
      if key is None:
        return Stream.constant(atom.val)
      cached = const_cache.get(key)
      if cached is None:
        cached = Stream.constant(atom.val)
        const_cache[key] = cached
      return cached
    return env[atom]

  for eqn in raw_jaxpr.eqns:
    prim = eqn.primitive
    if (
        'axis_name' not in eqn.params
        and prim.name != 'axis_index'
        and all(_is_literal_atom(a) or a in const_env for a in eqn.invars)
    ):
      const_vals = tuple(
          a.val if _is_literal_atom(a) else const_env[a] for a in eqn.invars
      )
      eval_fn = _make_eqn_evaluator(
          prim, eqn.params, getattr(eqn, 'ctx', None)
      )
      folded = eval_fn(*const_vals)
      if prim.multiple_results:
        for ovar, c_val in zip(eqn.outvars, folded):
          if not _is_drop_var(ovar):
            const_env[ovar] = c_val
            env[ovar] = Stream.constant(c_val)
      else:
        ovar = eqn.outvars[0]
        if not _is_drop_var(ovar):
          const_env[ovar] = folded
          env[ovar] = Stream.constant(folded)
      continue

    in_args = tuple(_read_atom(atom) for atom in eqn.invars)
    sub_jaxpr = (
        eqn.params.get('jaxpr') or eqn.params.get('call_jaxpr')
        if prim.name in _SUBJAXPR_JIT_NAMES
        else None
    )
    if expand_subjaxprs and sub_jaxpr is not None:
      sub_name = str(eqn.params.get('name', prim.name))
      with push_prefix(sub_name):
        sub_outs = _expand_eqns(
            sub_jaxpr,
            in_args,
            expand_subjaxprs=True,
            scope_name=f'{scope_name}/{sub_name}',
        )
      for ovar, ostream in zip(eqn.outvars, sub_outs):
        if not _is_drop_var(ovar):
          env[ovar] = ostream
      continue

    step_fn = _eqn_step_func(eqn)
    raw_out = emit_comb_node(step_fn, in_args, {}, name=prim.name, nout=1)
    if prim.multiple_results:
      for out_idx, ovar in enumerate(eqn.outvars):
        if not _is_drop_var(ovar):
          env[ovar] = emit_comb_node(
              _proj_step_func(out_idx, jax_tagged=True),
              (raw_out,),
              {},
              name=f'{prim.name}[{out_idx}]',
          )
    else:
      ovar = eqn.outvars[0]
      if not _is_drop_var(ovar):
        env[ovar] = raw_out

  return [_read_atom(atom) for atom in raw_jaxpr.outvars]


def expand_jaxpr(
    closed_jaxpr: Any,
    *args: Any,
    out_tree: Any = None,
    expand_subjaxprs: bool = False,
    name: str = 'jaxpr',
) -> Any:
  """Expands a traced pure `Jaxpr` into fine-grained Psyre `Node`s.

  Emits one `@psyre_comb(pure=True)` `Node` per `JaxprEqn` inside a
  `push_prefix(name)` scope, attaching each node's canonical JAX `Primitive`
  (`psyre.jax.get_primitive(node)`) and `_psyre_jax_backend = True` flag.

  Args:
    closed_jaxpr: A pure `ClosedJaxpr` / `Jaxpr` produced by `jax.make_jaxpr`.
    *args: Input `Stream`s (or PyTrees of `Stream`s/constants matching the flat
      `invars` of `closed_jaxpr`).
    out_tree: Optional `jax.tree_util.PyTreeDef` reconstructing the structured
      output container from flat `outvars` streams. When omitted, returns a
      single `Stream` for 1-output `Jaxpr`s or a `tuple` of `Stream`s
      otherwise.
    expand_subjaxprs: When `True`, recursively flattens nested `jit`/`pjit`
      sub-`Jaxpr` equations inside child prefix scopes.
    name: Hierarchy prefix label for the expanded `Node` scope.

  Returns:
    Output `Stream` or PyTree of `Stream`s matching `closed_jaxpr.outvars`.

  Raises:
    ImpureJaxprError: If `closed_jaxpr` (or a nested sub-`Jaxpr`) contains any
      non-empty side effect set.
    RuntimeError: If called outside Stage 1.
    ValueError: If the number of flattened input leaves disagrees with
      `jaxpr.invars`.
  """
  require_builder()
  _check_pure_jaxpr(closed_jaxpr, label=name)
  flat_inputs = tuple(
      jax.tree_util.tree_leaves(args, is_leaf=is_dynamic_leaf)
  )
  with push_prefix(name):
    out_streams = _expand_eqns(
        closed_jaxpr,
        flat_inputs,
        expand_subjaxprs=expand_subjaxprs,
        scope_name=name,
    )
  if out_tree is not None:
    return jax.tree_util.tree_unflatten(out_tree, out_streams)
  if len(out_streams) == 1:
    return out_streams[0]
  return tuple(out_streams)


def _validate_shape_dtype_tree(
    tree: Any, label: str, *, allow_empty: bool = False
) -> None:
  """Requires every leaf in `tree` to be an explicit `jax.ShapeDtypeStruct`."""
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves and not allow_empty:
    raise ValueError(
        f'{label} must contain at least one jax.ShapeDtypeStruct leaf.'
    )
  for leaf in leaves:
    if not isinstance(leaf, jax.ShapeDtypeStruct):
      raise TypeError(
          f'{label} leaves must be explicit jax.ShapeDtypeStruct hints when '
          f'trace_jaxpr=True; got {type(leaf).__name__}.'
      )


def _is_jax_pytree_node(val: Any) -> bool:
  """Returns True when `val` is registered as an internal `jax.tree_util` node."""
  children, _ = jax.tree_util.tree_flatten(
      val, is_leaf=lambda x: x is not val
  )
  return not (len(children) == 1 and children[0] is val)


def _normalize_in_avals(
    in_avals: Any,
    args: tuple[Any, ...],
    static_argnums: frozenset[int],
) -> tuple[Any, ...]:
  """Aligns `in_avals` with the non-static positional arguments of `fn`."""
  dyn_positions = [i for i in range(len(args)) if i not in static_argnums]
  num_dyn = len(dyn_positions)
  if isinstance(in_avals, jax.ShapeDtypeStruct):
    if num_dyn != 1:
      raise ValueError(
          f'trace_jaxpr=True expected {num_dyn} in_avals entries matching '
          'non-static positional arguments, got a single ShapeDtypeStruct.'
      )
    aval_args: tuple[Any, ...] = (in_avals,)
  elif isinstance(in_avals, (tuple, list)):
    if num_dyn == 1 and isinstance(args[dyn_positions[0]], (tuple, list)):
      # Treat `in_avals` as the single container arg unless already wrapped as a
      # 1-element tuple/list around it.
      if len(in_avals) == 1 and isinstance(in_avals[0], (tuple, list)):
        aval_args = tuple(in_avals)
      elif (
          len(in_avals) != 1
          or jax.tree_util.tree_structure(in_avals)
          == jax.tree_util.tree_structure(
              args[dyn_positions[0]], is_leaf=is_dynamic_leaf
          )
      ):
        aval_args = (in_avals,)
      else:
        aval_args = tuple(in_avals)
    elif len(in_avals) == num_dyn:
      aval_args = tuple(in_avals)
    else:
      raise ValueError(
          f'trace_jaxpr=True expected {num_dyn} in_avals entries matching '
          f'non-static positional arguments, got {len(in_avals)}.'
      )
  elif num_dyn == 1 and (
      isinstance(in_avals, Mapping) or _is_jax_pytree_node(in_avals)
  ):
    aval_args = (in_avals,)
  else:
    raise TypeError(
        'in_avals must be a jax.ShapeDtypeStruct or tuple/list of '
        f'ShapeDtypeStruct PyTrees, got {type(in_avals).__name__}.'
    )

  for pos, aval_tree in zip(dyn_positions, aval_args):
    _validate_shape_dtype_tree(aval_tree, f'in_avals[{pos}]', allow_empty=True)
    arg_tree = args[pos]
    aval_struct = jax.tree_util.tree_structure(aval_tree)
    arg_leaves, arg_struct = jax.tree_util.tree_flatten(
        arg_tree, is_leaf=is_dynamic_leaf
    )
    del arg_leaves
    if aval_struct != arg_struct:
      raise ValueError(
          f'in_avals[{pos}] PyTree structure {aval_struct} does not match '
          f'argument structure {arg_struct}.'
      )
  return aval_args


def jit(
    fn: Callable[..., Any],
    *,
    trace_jaxpr: bool = False,
    in_avals: Any = None,
    expand_subjaxprs: bool = False,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    nout: int | None = None,
    name: str = '',
    **jit_kwargs: Any,
) -> Callable[..., Any]:
  """Lifts a pure JAX function behind `jax.jit`, with optional `Jaxpr` tracing.

  Two execution modes:

  - **`trace_jaxpr=False` (default — lazy first-tick compilation)**:
    Performs **no** Stage-1 `Jaxpr` tracing and requires **no** shape/dtype
    annotations. Emits a single pure `@psyre_comb(pure=True)` node wrapping
    `jax.jit(unwrap_inst_func(fn), **jit_kwargs)`. Inactive (`Absent`) ticks
    short-circuit before calling XLA.
  - **`trace_jaxpr=True` (opt-in Stage-1 `Jaxpr` expansion)**:
    **Requires** explicit `jax.ShapeDtypeStruct` hints in `in_avals` for every
    non-static positional input. Traces `fn` with `jax.make_jaxpr`, rejects any
    effectful `Jaxpr` (`ImpureJaxprError`), and expands each `JaxprEqn` into a
    fine-grained `@psyre_comb(pure=True)` `Node` inside a `push_prefix(name)`
    scope carrying canonical `Primitive` metadata. The expanded subgraph works
    with `optimize_graph`, `check_clocks`, `vmap`, visualization, and can be
    re-fused back into a single `jax.jit` super-node via `fuse_jit_islands`.

  Args:
    fn: Pure JAX callable (or already lifted `@psyre_comb` wrapper).
    trace_jaxpr: When `True`, trace and expand the `Jaxpr` into Stage-1 `Node`s
      using `in_avals`.
    in_avals: Required when `trace_jaxpr=True`: a `jax.ShapeDtypeStruct` (for a
      single non-static input) or a sequence of PyTrees whose leaves are all
      `jax.ShapeDtypeStruct` matching the non-static positional arguments.
    expand_subjaxprs: When `True` with `trace_jaxpr=True`, recursively expands
      nested `jax.jit`/`pjit` calls into child prefix scopes.
    static_argnums: Zero-based positional indices kept static (forwarded to
      `jax.jit` / `partition_comb_args`).
    static_argnames: Keyword argument names kept static.
    nout: Optional positive output arity unpacking a tuple output stream into
      an `nout`-tuple of projected `Stream`s (lazy `trace_jaxpr=False` mode).
    name: Optional Stage-1 node / prefix scope label.
    **jit_kwargs: Additional keyword arguments forwarded to `jax.jit` (e.g.,
      `donate_argnums`, `in_shardings`, `out_shardings`, `backend`).

  Returns:
    A lifted callable producing `Stream`s (or PyTrees of `Stream`s).

  Raises:
    TypeError: If `fn` is not callable or `in_avals` contains non-
      `ShapeDtypeStruct` leaves.
    ValueError: If `trace_jaxpr=True` without `in_avals`, `trace_jaxpr=False`
      with `in_avals` set, or `nout` is not a positive integer.
    ImpureJaxprError: If `trace_jaxpr=True` traces an effectful `Jaxpr`.
  """
  if not is_callable_target(fn):
    raise TypeError(f'psyre.jax.jit expects a callable, got {type(fn)}.')
  if nout is not None:
    validate_positive_int(nout, 'psyre.jax.jit nout')
  if trace_jaxpr and in_avals is None:
    raise ValueError(
        'psyre.jax.jit(..., trace_jaxpr=True) requires explicit in_avals of '
        'jax.ShapeDtypeStruct hints at Stage 1.'
    )
  if not trace_jaxpr and in_avals is not None:
    raise ValueError(
        'in_avals is only accepted when trace_jaxpr=True.'
    )

  raw_fn = unwrap_inst_func(fn)
  argnums = normalize_static_argnums(static_argnums) | getattr(
      fn, '_psyre_static_argnums', frozenset()
  )
  argnames = normalize_static_argnames(static_argnames) | getattr(
      fn, '_psyre_static_argnames', frozenset()
  )
  nout_spec = nout if nout is not None else read_wrapper_nout_spec(fn)
  scope_name = name or callable_name(raw_fn, 'jit_fn')

  jit_fn_kwargs = dict(jit_kwargs)
  if argnums:
    jit_fn_kwargs['static_argnums'] = tuple(sorted(argnums))
  if argnames:
    jit_fn_kwargs['static_argnames'] = tuple(sorted(argnames))
  jitted_fn = jax.jit(raw_fn, **jit_fn_kwargs)

  if not trace_jaxpr:
    label = f'jit({scope_name})'
    step_func = make_jax_step_func(jitted_fn, name=label, pure=True)

    @functools.wraps(raw_fn)
    def lazy_wrapper(*args: Any, **kwargs: Any) -> Any:
      dyn_inputs, node_kwargs = partition_comb_args(
          args,
          kwargs,
          static_argnums=argnums,
          static_argnames=argnames,
          static_literals=True,
      )
      if not dyn_inputs:
        return jitted_fn(*args, **kwargs)
      return emit_comb_node(
          step_func,
          dyn_inputs,
          node_kwargs,
          name=label,
          nout=resolve_nout(nout_spec, args, kwargs),
      )

    return attach_wrapper_meta(
        lazy_wrapper,
        inst_func=jitted_fn,
        step_func=step_func,
        pure=True,
        static_argnums=argnums,
        static_argnames=argnames,
        nout_spec=nout_spec,
    )

  # Eagerly verify `in_avals` contains only `ShapeDtypeStruct` at wrap time so
  # obvious hint errors surface immediately even before the wrapper is called.
  if isinstance(in_avals, jax.ShapeDtypeStruct):
    pass
  elif isinstance(in_avals, (tuple, list, Mapping)):
    entries = in_avals.values() if isinstance(in_avals, Mapping) else in_avals
    for entry in entries:
      _validate_shape_dtype_tree(entry, 'in_avals', allow_empty=True)
    if not jax.tree_util.tree_leaves(in_avals):
      raise ValueError(
          'in_avals must contain at least one jax.ShapeDtypeStruct leaf.'
      )
  elif _is_jax_pytree_node(in_avals):
    _validate_shape_dtype_tree(in_avals, 'in_avals')
  else:
    raise TypeError(
        'in_avals must be a jax.ShapeDtypeStruct or container of '
        f'ShapeDtypeStruct leaves, got {type(in_avals).__name__}.'
    )

  @functools.wraps(raw_fn)
  def traced_wrapper(*args: Any, **kwargs: Any) -> Any:
    if active_builder.get() is None:
      return jitted_fn(*args, **kwargs)
    if any(
        is_dynamic_leaf(v)
        for v in jax.tree_util.tree_leaves(kwargs, is_leaf=is_dynamic_leaf)
    ):
      raise TypeError(
          'trace_jaxpr=True requires dynamic Stream arguments to be passed '
          'positionally; keyword arguments must be static.'
      )
    dyn_positions = [i for i in range(len(args)) if i not in argnums]
    dyn_args = tuple(args[i] for i in dyn_positions)
    if not any(
        is_dynamic_leaf(v)
        for v in jax.tree_util.tree_leaves(dyn_args, is_leaf=is_dynamic_leaf)
    ):
      return jitted_fn(*args, **kwargs)
    aval_args = _normalize_in_avals(in_avals, args, argnums)
    static_pos_vals = {i: args[i] for i in argnums if i < len(args)}

    def _trace_target(*dyn_vals: Any) -> Any:
      full_args: list[Any] = []
      dyn_it = iter(dyn_vals)
      for i in range(len(args)):
        if i in static_pos_vals:
          full_args.append(static_pos_vals[i])
        else:
          full_args.append(next(dyn_it))
      return raw_fn(*full_args, **kwargs) if kwargs else raw_fn(*full_args)

    closed_jaxpr, out_shape_tree = jax.make_jaxpr(
        _trace_target, return_shape=True
    )(*aval_args)
    out_treedef = jax.tree_util.tree_structure(out_shape_tree)
    return expand_jaxpr(
        closed_jaxpr,
        *dyn_args,
        out_tree=out_treedef,
        expand_subjaxprs=expand_subjaxprs,
        name=scope_name,
    )

  return attach_wrapper_meta(
      traced_wrapper,
      inst_func=jitted_fn,
      pure=True,
      static_argnums=argnums,
      static_argnames=argnames,
      nout_spec=nout_spec,
  )


def _is_island_eligible(node: Node, include_stream_ops: bool) -> bool:
  """Returns True when `node` may participate in a `jax.jit` fusion island."""
  if node.is_stateful:
    return False
  if node.is_primitive(*_NON_FUSIBLE_PRIMITIVES):
    return False
  if not is_pure_step_func(node.step_func):
    return False
  if getattr(node.step_func, '_psyre_inst_func', None) is None:
    return False
  if is_jax_step_func(node):
    return True
  if not include_stream_ops:
    return False
  if node.op == '__getitem__':
    # Only static-index / slice projections (`len(node.inputs) == 1`) trace
    # safely inside `jax.jit` (e.g. `top_k(x, k)[0]`, constant `x[1:3]`).
    return (
        node.is_stream_op('__getitem__')
        and len(node.inputs) == 1
        and CALL_PLAN_KWARG in node.kwargs
    )
  if node.op in _NON_TRACED_STREAM_OPS:
    return False
  return node.is_stream_op()


def _find_convex_islands(
    graph: Graph,
    topo_order: tuple[Node, ...],
    clocks: Any,
    *,
    include_stream_ops: bool,
    min_nodes: int,
) -> list[list[Node]]:
  """Partitions same-clock eligible nodes into maximal convex DAG clusters."""
  n_nodes = len(graph.nodes)
  eligible = [
      _is_island_eligible(node, include_stream_ops) for node in graph.nodes
  ]
  node_clock = [clocks.node_clock(node) for node in graph.nodes]
  node_arms = [node.arms for node in graph.nodes]

  # Quotient graph adjacency in terms of cluster representative IDs.
  rep: list[int] = list(range(n_nodes))
  members: dict[int, set[int]] = {i: {i} for i in range(n_nodes)}
  q_succ: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
  q_pred: dict[int, set[int]] = {i: set() for i in range(n_nodes)}

  for node in graph.nodes:
    for stream in node.instantaneous_input_streams():
      prod = graph.producer(stream)
      if prod is not None and prod.id != node.id:
        q_succ[prod.id].add(node.id)
        q_pred[node.id].add(prod.id)

  def _find(nid: int) -> int:
    while rep[nid] != nid:
      rep[nid] = rep[rep[nid]]
      nid = rep[nid]
    return nid

  def _can_merge(ca: int, cb: int) -> bool:
    # Merging `ca -> cb` is acyclic/convex iff there is NO alternative path
    # `ca -> X ->+ cb` bypassing direct edge `ca -> cb`.
    if ca == cb:
      return False
    seeds = [nxt for nxt in q_succ[ca] if nxt != cb]
    if not seeds:
      return True
    visited: set[int] = set(seeds)
    queue: collections.deque[int] = collections.deque(seeds)
    while queue:
      cur = queue.popleft()
      if cur == cb:
        return False
      for nxt in q_succ[cur]:
        if nxt == cb:
          return False
        if nxt not in visited:
          visited.add(nxt)
          queue.append(nxt)
    return True

  def _contract(ca: int, cb: int) -> int:
    # Merge `cb` into `ca` in the quotient DAG (preserving edges without self-
    # loops).
    members[ca].update(members.pop(cb))
    for mid in members[ca]:
      rep[mid] = ca
    for p in list(q_pred.pop(cb)):
      q_succ[p].discard(cb)
      if p != ca:
        q_succ[p].add(ca)
        q_pred[ca].add(p)
    for s in list(q_succ.pop(cb)):
      q_pred[s].discard(cb)
      if s != ca:
        q_pred[s].add(ca)
        q_succ[ca].add(s)
    q_succ[ca].discard(ca)
    q_pred[ca].discard(ca)
    return ca

  # First contract direct same-clock connected edges `u -> v` until fixpoint.
  changed = True
  while changed:
    changed = False
    for node in topo_order:
      if not eligible[node.id]:
        continue
      for stream in node.instantaneous_input_streams():
        prod = graph.producer(stream)
        if prod is None or not eligible[prod.id]:
          continue
        if (
            node_clock[prod.id] != node_clock[node.id]
            or node_arms[prod.id] != node_arms[node.id]
        ):
          continue
        ca, cb = _find(prod.id), _find(node.id)
        if ca == cb:
          continue
        if _can_merge(ca, cb):
          _contract(ca, cb)
          changed = True

  # Return each cluster ordered causally (`topo_order`) keeping only islands
  # with >= `min_nodes` nodes AND at least one explicit JAX-backend node.
  topo_pos = {node.id: pos for pos, node in enumerate(topo_order)}
  islands: list[list[Node]] = []
  seen_reps: set[int] = set()
  for node in topo_order:
    if not eligible[node.id]:
      continue
    r = _find(node.id)
    if r in seen_reps:
      continue
    seen_reps.add(r)
    cluster = sorted(
        (graph.nodes[mid] for mid in members[r]),
        key=lambda nd: topo_pos[nd.id],
    )
    if len(cluster) >= min_nodes and any(
        is_jax_step_func(nd) for nd in cluster
    ):
      islands.append(cluster)
  return islands


def _build_island_evaluator(
    cluster: list[Node],
    ext_inputs: tuple[Stream, ...],
    exit_streams: tuple[Stream, ...],
    label: str,
    jit_kwargs: Mapping[str, Any],
) -> Callable[..., Any]:
  """Builds a single `jax.jit`-compiled combinational `step_func` for island."""
  steps: list[
      tuple[int, Callable[..., Any], tuple[int, ...], Mapping[str, Any]]
  ] = []
  for node in cluster:
    inst_func = getattr(node.step_func, '_psyre_inst_func')
    in_ids = tuple(inp.id for inp in node.inputs)
    steps.append((node.output.id, inst_func, in_ids, dict(node.kwargs)))

  ext_ids = tuple(s.id for s in ext_inputs)
  exit_ids = tuple(s.id for s in exit_streams)
  single_exit = len(exit_ids) == 1

  def _eval_island(*ext_vals: Any) -> Any:
    env: dict[int, Any] = dict(zip(ext_ids, ext_vals))
    for out_id, inst_fn, in_ids, node_kwargs in steps:
      vals = tuple(env[sid] for sid in in_ids)
      c_args, c_kw = materialize_comb_call(node_kwargs, vals)
      env[out_id] = inst_fn(*c_args, **c_kw) if c_kw else inst_fn(*c_args)
    if single_exit:
      return env[exit_ids[0]]
    return tuple(env[sid] for sid in exit_ids)

  _eval_island.__name__ = label
  compiled = jax.jit(_eval_island, **jit_kwargs)
  return make_jax_step_func(compiled, name=label, pure=True)


def _collect_extra_root_ids(
    graph: Graph, extra_roots: Iterable[Any] | None
) -> set[int]:
  """Resolves caller-provided `extra_roots` to `Stream` IDs in `graph`."""
  if extra_roots is None:
    return set()
  return {
      s.id
      for s in graph.collect_root_streams(extra_roots, include_output=False)
  }


def _common_prefixes(cluster: list[Node]) -> tuple[tuple[str, int], ...]:
  """Returns the longest shared hierarchy prefix across nodes in `cluster`."""
  common = cluster[0].prefixes
  for nd in cluster[1:]:
    limit = min(len(common), len(nd.prefixes))
    keep = 0
    while keep < limit and common[keep] == nd.prefixes[keep]:
      keep += 1
    common = common[:keep]
    if not common:
      break
  return common


def fuse_jit_islands(
    graph: Graph,
    *,
    min_nodes: int = 2,
    include_stream_ops: bool = True,
    extra_roots: Iterable[Any] | None = None,
    jit_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Graph, tuple[JitIslandInfo, ...]]:
  """Fuses maximal same-clock convex JAX islands into single `jax.jit` nodes.

  Returns a **new** finalized `Graph` leaving the original `graph` untouched,
  along with a `tuple` of `JitIslandInfo` descriptors documenting each fused
  combinational island.

  Boundaries that terminate an island:
  - Stateful temporal primitives (`fby`, `pre`, `morph`, `current`, `buffer`,
    `counter`, `prefix_sum`, `scan`, `static_stream`, user `@psyre_sequ`)
  - Clock-changing / scalar mux primitives (`when`, `whennot`, `merge`,
    `switch`, `ifthenelse`, `post`)
  - Changes in inferred synchronous `Clock` or `ArmScope`
  - Impure (`pure=False`) and non-JAX host nodes (`to_numpy`, `psyre.numpy`
    calls, arbitrary Python callbacks)
  - Non-convex merges that would create an instantaneous cycle through an
    external node

  Args:
    graph: Finalized Stage-1 `Graph` to fuse.
    min_nodes: Minimum number of nodes required to form a fused island
      (integer >= 1, default `2`).
    include_stream_ops: When `True` (default), includes adjacent pure `Stream`
      arithmetic/projection ops connected to a JAX node inside the same island.
    extra_roots: Optional extra root `Stream`s / `Node`s / stream names whose
      outputs must stay observable after fusion.
    jit_kwargs: Optional dict of keyword args forwarded to `jax.jit` for each
      synthesized island evaluator.

  Returns:
    `(fused_graph, islands)` pair.
  """
  graph.require_finalized()
  validate_positive_int(min_nodes, 'fuse_jit_islands min_nodes')
  topo_order = causal_node_order(graph)
  clocks = check_clocks(graph)
  extra_root_ids = _collect_extra_root_ids(graph, extra_roots)
  arm_cond_ids: set[int] = set()
  for nd in graph.nodes:
    arm_cond_ids.update(graph.resolve_stream(sc.cond).id for sc in nd.arms)
  output_ids = (
      {s.id for s in iter_output_leaves(graph.output)}
      | extra_root_ids
      | arm_cond_ids
  )
  jit_opts = dict(jit_kwargs or {})

  islands = _find_convex_islands(
      graph,
      topo_order,
      clocks,
      include_stream_ops=include_stream_ops,
      min_nodes=min_nodes,
  )

  island_specs: list[
      tuple[list[Node], tuple[Stream, ...], tuple[Stream, ...], str, Clock]
  ] = []
  active_node_to_island: dict[int, int] = {}
  for cluster in islands:
    member_ids = {nd.id for nd in cluster}
    ext_in_map: dict[int, Stream] = {}
    for nd in cluster:
      for inp in nd.inputs:
        prod = graph.producer(inp)
        if prod is None or prod.id not in member_ids:
          ext_in_map.setdefault(inp.id, inp)
    exits: list[Stream] = []
    exit_seen: set[int] = set()
    for nd in cluster:
      s = nd.output
      if s.id in exit_seen:
        continue
      needed = s.id in output_ids or any(
          consumer.id not in member_ids for consumer in graph.consumers(s)
      )
      if needed:
        exit_seen.add(s.id)
        exits.append(s)
    if not exits:
      continue
    cur_slot = len(island_specs)
    clk = clocks.node_clock(cluster[0])
    label = f'jit_island_{cur_slot}'
    island_specs.append(
        (cluster, tuple(ext_in_map.values()), tuple(exits), label, clk)
    )
    for nd in cluster:
      active_node_to_island[nd.id] = cur_slot

  # Reconstruct a fresh non-destructive Graph keeping every observable Stream.
  hidden_internal_ids: set[int] = set()
  for cluster, _, exits, _, _ in island_specs:
    exit_id_set = {e.id for e in exits}
    for nd in cluster:
      if nd.output.id not in exit_id_set:
        hidden_internal_ids.add(nd.output.id)
  new_graph, stream_map = graph.clone_streams(exclude_ids=hidden_internal_ids)

  emitted_islands: set[int] = set()
  info_by_slot: dict[int, JitIslandInfo] = {}

  for node in graph.nodes:
    slot = active_node_to_island.get(node.id)
    if slot is None:
      new_graph.copy_node_into(node, stream_map)
      continue

    if slot in emitted_islands:
      continue
    emitted_islands.add(slot)
    cluster, ext_inputs, exits, label, clk = island_specs[slot]
    remapped_arms = new_graph.remap_arms(cluster[0].arms, stream_map)
    island_prefixes = _common_prefixes(cluster)
    prefix_str = '/'.join(seg for seg, _ in island_prefixes)
    scoped_name = f'{prefix_str}/{label}' if prefix_str else label
    step_func = _build_island_evaluator(
        cluster, ext_inputs, exits, label, jit_opts
    )
    mapped_inputs = tuple(stream_map.get(s.id, s) for s in ext_inputs)
    single_exit = len(exits) == 1
    super_out = (
        stream_map[exits[0].id]
        if single_exit
        else Stream(name=f'{label}_bundle')
    )
    super_nd = new_graph.add_node(
        op=label,
        name=scoped_name,
        output=super_out,
        step_func=step_func,
        kwargs={},
        is_stateful=False,
        arms=remapped_arms,
        prefixes=island_prefixes,
    )
    new_graph.rewire_node(super_nd, inputs=mapped_inputs, resets=())
    if not single_exit:
      for out_idx, exit_s in enumerate(exits):
        proj_nd = new_graph.add_node(
            op=f'island_out_{out_idx}',
            name=f'{scoped_name}[{out_idx}]',
            output=stream_map[exit_s.id],
            step_func=_proj_step_func(out_idx, jax_tagged=False),
            kwargs={},
            is_stateful=False,
            arms=remapped_arms,
            prefixes=island_prefixes,
        )
        new_graph.rewire_node(proj_nd, inputs=(super_out,), resets=())
    info_by_slot[slot] = JitIslandInfo(
        name=scoped_name,
        clock=clk,
        node_names=tuple(nd.name or nd.op for nd in cluster),
        num_inputs=len(ext_inputs),
        num_outputs=len(exits),
    )

  infos = tuple(info_by_slot[slot] for slot in range(len(island_specs)))

  if graph.output is not None:
    new_graph.set_output(
        map_output_leaves(lambda s: stream_map.get(s.id, s), graph.output)
    )
  new_graph.mark_finalized()
  # Verify our quotient contraction strictly preserved acyclicity.
  check_causality(new_graph)
  return new_graph, infos


def jit_graph(
    graph: Graph,
    *,
    min_nodes: int = 2,
    include_stream_ops: bool = True,
    extra_roots: Iterable[Any] | None = None,
    jit_kwargs: Mapping[str, Any] | None = None,
) -> Graph:
  """Convenience wrapper returning the fused `Graph` from `fuse_jit_islands`."""
  fused, _ = fuse_jit_islands(
      graph,
      min_nodes=min_nodes,
      include_stream_ops=include_stream_ops,
      extra_roots=extra_roots,
      jit_kwargs=jit_kwargs,
  )
  return fused
