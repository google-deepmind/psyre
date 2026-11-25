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
"""Lifting-time JAX transformations (`vmap`, `batch_streams`, `rewrite_dtype`).

Implements Stage-1 transformations over lifted `psyre.jax` / `psyre.jax.lax`
callables and composite Psyre `Graph` subprograms **without** requiring
`Jaxpr` tracing or concrete static shapes up front.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import functools
import operator
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  import jax
  from jax import lax as _jax_lax
  import jax.numpy as jnp
except ImportError as err:  # pragma: no cover
  raise ImportError(
      'psyre.jax requires JAX. Install it with: pip install "psyre[jax]"'
  ) from err

# pylint: disable=g-importing-member,g-multiple-import
from psyre.builder import GraphBuilder
from psyre.clock import Clock, ClockError, check_clocks
from psyre.common import (
    ArmScope,
    CapturedEnv,
    MergeSelector,
    Semantics,
    active_builder,
    arm_stack,
    callable_name,
    emit_comb_node,
    emit_when,
    fby,
    is_pure_step_func,
    prefix_stack,
    push_prefix,
    remap_arm_scopes,
    require_builder,
    reset_stack,
    unwrap_inst_func,
    validate_positive_int,
)
from psyre.graph import Graph, Node, map_output_leaves
from psyre.jax.ops import (
    attach_wrapper_meta,
    get_primitive,
    is_jax_step_func,
    make_jax_step_func,
    read_wrapper_nout_spec,
)
from psyre.payload import (
    CALL_PLAN_KWARG,
    CombCallPlan,
    is_array_object,
    is_callable_target,
    is_dynamic_leaf,
    is_int_axis,
    is_namedtuple,
    is_numpy_scalar,
    lookup_pytree_node,
    make_batch_streams,
    make_stack_streams,
    make_unstack_stream,
    map_tree_with_axes,
    materialize_comb_call,
    normalize_arg_axes,
    resolve_nout,
    validate_optional_axis,
)
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import

_PREFERRED_DTYPE_OPS = frozenset({
    'dot_general',
    'conv_general_dilated',
    'matmul',
})
_PREFERRED_DTYPE_PRIMS = frozenset({
    'dot_general',
    'conv_general_dilated',
})
_COLLECTIVE_OPS = frozenset({
    'all_gather',
    'all_to_all',
    'axis_index',
    'axis_size',
    'pgather',
    'pmax',
    'pmean',
    'pmin',
    'ppermute',
    'pshuffle',
    'psum',
    'psum_scatter',
    'pswapaxes',
})


class BatchError(ValueError):
  """Raised when a Stage-1 graph cannot be batched by `psyre.jax.vmap`."""


stack_streams = make_stack_streams(
    jnp.stack,
    make_jax_step_func,
    emit_comb_node,
    doc='Stacks one or more array `Stream`s along `axis` via `jnp.stack`.',
)

unstack_stream = make_unstack_stream(
    jnp.asarray,
    jnp.moveaxis,
    make_jax_step_func,
    emit_comb_node,
    doc='Unstacks an array `Stream` into a `size`-tuple of `Stream`s.',
)

batch_streams = make_batch_streams(
    unstack_stream,
    stack_streams,
    require_builder,
    push_prefix,
    error_cls=BatchError,
    doc=(
        'Structural Stage-1 lane expansion instantiating `size` independent '
        'copies of `fn` on unstacked `jax.Array` `Stream` lanes. Supports '
        'arbitrary heterogeneous per-lane clocks, automata, and state machines.'
    ),
)


def _jax_vmap(
    fn: Callable[..., Any],
    in_axes: Any,
    out_axes: Any = 0,
    *,
    axis_name: Any = None,
    axis_size: int | None = None,
    spmd_axis_name: Any = None,
) -> Callable[..., Any]:
  """Calls `jax.vmap` forwarding only non-`None` optional keyword flags."""
  vm_kwargs: dict[str, Any] = {'in_axes': in_axes, 'out_axes': out_axes}
  if axis_name is not None:
    vm_kwargs['axis_name'] = axis_name
  if axis_size is not None:
    vm_kwargs['axis_size'] = axis_size
  if spmd_axis_name is not None:
    vm_kwargs['spmd_axis_name'] = spmd_axis_name
  return jax.vmap(fn, **vm_kwargs)


def _has_dynamic_leaf(val: Any) -> bool:
  """Returns True when `val` encloses a `Stream`, thunk, or PyTree of them."""
  if is_dynamic_leaf(val):
    return True
  if val is None or isinstance(val, (bool, int, float, complex, str, bytes)):
    return False
  reg = lookup_pytree_node(type(val))
  if reg is not None:
    children, _ = reg[0](val)
    return any(_has_dynamic_leaf(c) for c in children)
  if isinstance(val, (tuple, list)):
    return any(_has_dynamic_leaf(x) for x in val)
  if isinstance(val, Mapping):
    return any(_has_dynamic_leaf(v) for v in val.values())
  return False


@dataclasses.dataclass(frozen=True, slots=True)
class _VmapSlot:
  index: int


def _materialize_vmap_template(tmpl: Any, vals: tuple[Any, ...]) -> Any:
  """Substitutes `_VmapSlot` markers in `tmpl` with per-lane input values."""
  if isinstance(tmpl, _VmapSlot):
    return vals[tmpl.index]
  if tmpl is None or isinstance(tmpl, (bool, int, float, complex, str, bytes)):
    return tmpl
  reg = lookup_pytree_node(type(tmpl))
  if reg is not None:
    flatten_fn, unflatten_fn = reg
    children, aux_data = flatten_fn(tmpl)
    return unflatten_fn(
        aux_data,
        tuple(_materialize_vmap_template(c, vals) for c in children),
    )
  if isinstance(tmpl, tuple):
    items = tuple(_materialize_vmap_template(x, vals) for x in tmpl)
    return type(tmpl)(*items) if is_namedtuple(tmpl) else items
  if isinstance(tmpl, list):
    return [_materialize_vmap_template(x, vals) for x in tmpl]
  if isinstance(tmpl, Mapping):
    return {k: _materialize_vmap_template(v, vals) for k, v in tmpl.items()}
  return tmpl


def _make_moveaxis_step(src: int, dst: int) -> Callable[..., Any]:
  def _move(x: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: jnp.moveaxis(jnp.asarray(leaf), src, dst), x
    )

  label = f'moveaxis_{src}_to_{dst}'
  _move.__name__ = label
  return make_jax_step_func(_move, name=label, pure=True)


def _broadcast_tree(val: Any, b_size: int, out_axis: int) -> Any:
  def _leaf(x: Any) -> jax.Array:
    arr = jnp.asarray(x)
    rep = jnp.broadcast_to(arr, (b_size,) + arr.shape)
    return rep if out_axis == 0 else jnp.moveaxis(rep, 0, out_axis)

  return jax.tree_util.tree_map(_leaf, val)


def _make_broadcast_step(
    ref_axis: int, axis_size: int | None, out_axis: int = 0
) -> Callable[..., Any]:
  """Builds a nested-`vmap`-safe broadcast helper step function."""
  if axis_size is not None:

    def _bcast_static(val: Any) -> Any:
      return _broadcast_tree(val, axis_size, out_axis)

    _bcast_static.__name__ = 'vmap_broadcast'
    return make_jax_step_func(_bcast_static, name='vmap_broadcast', pure=True)

  def _bcast_ref(val: Any, ref: Any) -> Any:
    ref_leaves = jax.tree_util.tree_leaves(ref)
    if not ref_leaves:
      raise BatchError('Cannot infer batch size from empty reference PyTree.')
    b_size = jnp.asarray(ref_leaves[0]).shape[ref_axis]
    return _broadcast_tree(val, b_size, out_axis)

  _bcast_ref.__name__ = 'vmap_broadcast'
  return make_jax_step_func(_bcast_ref, name='vmap_broadcast', pure=True)


def _vmap_single_comb(
    fn: Callable[..., Any],
    *,
    in_axes: Any,
    out_axes: Any,
    axis_name: Any,
    axis_size: int | None,
    spmd_axis_name: Any,
    name: str,
) -> Callable[..., Any]:
  """Commutes `vmap` directly with a lifted `@psyre_comb` callable."""
  raw_fn = unwrap_inst_func(fn)
  static_argnums: frozenset[int] = getattr(
      fn, '_psyre_static_argnums', frozenset()
  )
  static_argnames: frozenset[str] = getattr(
      fn, '_psyre_static_argnames', frozenset()
  )
  is_pure = bool(getattr(fn, '_psyre_pure', False))
  nout_spec = read_wrapper_nout_spec(fn)
  label = name or f'vmap({callable_name(raw_fn, "fn")})'

  def _wrap_vm(inner: Callable[..., Any], mapped_axes: Any) -> Any:
    return _jax_vmap(
        inner,
        mapped_axes,
        out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
    )

  def _eager_vmap(*a: Any, **kw: Any) -> Any:
    a_axes = normalize_arg_axes(
        in_axes, len(a), 'vmap in_axes', error_cls=BatchError
    )
    dyn_pos = [i for i in range(len(a)) if i not in static_argnums]
    if len(dyn_pos) == len(a):
      def _run(*pos: Any) -> Any:
        return raw_fn(*pos, **kw) if kw else raw_fn(*pos)

      return _wrap_vm(_run, a_axes)(*a)
    dyn_axes_e = tuple(a_axes[i] for i in dyn_pos)
    dyn_vals_e = tuple(a[i] for i in dyn_pos)

    def _run_static(*vals: Any) -> Any:
      it = iter(vals)
      full = tuple(
          a[i] if i in static_argnums else next(it) for i in range(len(a))
      )
      return raw_fn(*full, **kw) if kw else raw_fn(*full)

    return _wrap_vm(_run_static, dyn_axes_e)(*dyn_vals_e)

  _eager_vmap.__name__ = label

  def wrapper(*args: Any, **kwargs: Any) -> Any:
    arg_axes = normalize_arg_axes(
        in_axes, len(args), 'vmap in_axes', error_cls=BatchError
    )
    has_dyn = any(
        _has_dynamic_leaf(a)
        for i, a in enumerate(args)
        if i not in static_argnums
    ) or any(
        _has_dynamic_leaf(v)
        for k, v in kwargs.items()
        if k not in static_argnames
    )
    if not has_dyn:
      return _eager_vmap(*args, **kwargs)

    dyn_streams: list[Any] = []
    dyn_axes: list[int | None] = []

    def _bind_slot(leaf: Any, ax: Any) -> Any:
      if leaf is None:
        return None
      if is_dynamic_leaf(leaf):
        slot_ax = validate_optional_axis(ax, 'in_axes')
        idx = len(dyn_streams)
        dyn_streams.append(leaf)
        dyn_axes.append(slot_ax)
        return _VmapSlot(idx)
      if is_int_axis(ax):
        idx = len(dyn_streams)
        dyn_streams.append(Stream.constant(leaf))
        dyn_axes.append(ax)
        return _VmapSlot(idx)
      return leaf

    def _map_arg(val: Any, ax: Any, *, is_static: bool) -> Any:
      if is_static:
        return val
      return map_tree_with_axes(
          val, ax, _bind_slot, label='in_axes', error_cls=BatchError
      )

    pos_templates = tuple(
        _map_arg(a, ax, is_static=i in static_argnums)
        for i, (a, ax) in enumerate(zip(args, arg_axes))
    )
    kw_templates = tuple(
        (k, _map_arg(v, None, is_static=k in static_argnames))
        for k, v in kwargs.items()
    )
    if all(ax is None for ax in dyn_axes) and axis_size is None:
      raise BatchError(
          'vmap requires at least one mapped input axis or explicit axis_size.'
      )

    def _inner(*flat_vals: Any) -> Any:
      call_args = tuple(
          _materialize_vmap_template(t, flat_vals) for t in pos_templates
      )
      call_kwargs = {
          k: _materialize_vmap_template(v, flat_vals) for k, v in kw_templates
      }
      return raw_fn(*call_args, **call_kwargs)

    vmapped = _wrap_vm(_inner, tuple(dyn_axes))
    vmapped.__name__ = label
    step_func = make_jax_step_func(vmapped, name=label, pure=is_pure)
    out_arity = resolve_nout(nout_spec, args, kwargs)
    return emit_comb_node(
        step_func, tuple(dyn_streams), {}, name=label, nout=out_arity
    )

  return attach_wrapper_meta(
      wrapper,
      inst_func=_eager_vmap,
      pure=is_pure,
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      nout_spec=nout_spec,
  )


def _merge_batched_axes(axes: Sequence[int | None]) -> int | None:
  """Combines batch axes across value-preserving ports (`None` joins `int`)."""
  present = {a for a in axes if a is not None}
  if not present:
    return None
  if len(present) == 1 and all(a is not None for a in axes):
    return next(iter(present))
  return 0


def _validate_batched_switch_lookup(
    lookup: Any, num_arms: int, node: Node
) -> tuple[str, list[int], bool]:
  """Validates `switch` lookup for vectorized lowering under batched `sel`."""
  if not isinstance(lookup, MergeSelector):
    raise BatchError(
        f'vmap cannot vectorize switch node {node.name!r}: unsupported lookup.'
    )
  if lookup.is_boolean:
    if num_arms != 2:
      raise BatchError(
          f'vmap boolean switch {node.name!r} expected 2 arms, got {num_arms}.'
      )
    return 'bool', [1, 0], False
  int_arms: dict[int, int] = {}
  for (is_bool_tag, val), arm_idx in lookup.arms.items():
    if is_bool_tag or not isinstance(val, int) or val < 0:
      raise BatchError(
          f'vmap cannot batch switch selector on {node.name!r} with '
          f'non-integer tag {val!r}; dense non-negative integer tags '
          '0..k-1 are required.'
      )
    int_arms[val] = arm_idx
  if not int_arms:
    raise BatchError(f'vmap switch {node.name!r} has empty arm table.')
  max_tag = max(int_arms)
  order: list[int] = []
  for tag in range(max_tag + 1):
    if tag in int_arms:
      order.append(int_arms[tag])
    elif lookup.default_index is not None:
      order.append(lookup.default_index)
    else:
      raise BatchError(
          f'vmap cannot batch switch {node.name!r}: missing tag {tag} in '
          f'dense range 0..{max_tag} with no default branch.'
      )
  has_default = lookup.default_index is not None
  if has_default:
    order.append(lookup.default_index)
  return 'int', order, has_default


def _infer_batch_axes(
    sub_graph: Graph,
    ph_axes: Mapping[int, int | None],
    *,
    axis_name: Any = None,
) -> dict[int, int | None]:
  """Runs monotone fixed-point `batch_axis` inference on `sub_graph`."""
  batch_axis: dict[int, int | None] = dict(ph_axes)

  def _ax(stream: Stream) -> int | None:
    return batch_axis.get(stream.id)

  def _require_unbatched(stream: Stream, role: str, node: Node) -> None:
    if _ax(stream) is not None:
      raise BatchError(
          f'vmap cannot batch {role} stream on node {node.name!r} '
          f'(op={node.op!r}): clocks and control predicates must be uniform '
          'across the batch. Use batch_streams() for heterogeneous per-lane '
          'clocks.'
      )

  changed = True
  while changed:
    changed = False
    for node in sub_graph.nodes:
      for r in node.resets:
        _require_unbatched(r, 'reset', node)
      for scope in node.arms:
        _require_unbatched(scope.cond, 'arm condition', node)
      out_id = node.output.id
      prev = batch_axis.get(out_id)
      if node.is_static_stream():
        nxt = None
      elif node.is_primitive('when', 'whennot'):
        _require_unbatched(node.inputs[0], node.op + ' selector', node)
        nxt = _ax(node.inputs[1])
      elif node.is_primitive('merge'):
        _require_unbatched(node.inputs[0], 'merge selector', node)
        nxt = _merge_batched_axes(tuple(_ax(a) for a in node.inputs[1:]))
      elif node.is_primitive('buffer'):
        _require_unbatched(node.inputs[1], 'buffer push predicate', node)
        _require_unbatched(node.inputs[2], 'buffer pull predicate', node)
        nxt = _ax(node.inputs[0])
      elif node.is_primitive('fby', 'morph'):
        nxt = _merge_batched_axes((_ax(node.inputs[0]), _ax(node.inputs[1])))
      elif node.is_primitive('pre', 'post'):
        nxt = _ax(node.inputs[0])
      elif node.is_primitive('current'):
        _require_unbatched(node.inputs[0], 'current hold predicate', node)
        nxt = _merge_batched_axes((_ax(node.inputs[1]), _ax(node.inputs[2])))
      elif node.is_primitive('counter'):
        for port in node.inputs:
          _require_unbatched(port, 'counter', node)
        nxt = None
      elif node.is_primitive('prefix_sum'):
        nxt = _merge_batched_axes(tuple(_ax(p) for p in node.inputs))
      elif node.is_primitive('scan'):
        nxt = 0 if any(_ax(p) is not None for p in node.inputs) else None
      elif node.is_primitive('ifthenelse'):
        if _ax(node.inputs[0]) is not None:
          nxt = 0
        else:
          nxt = _merge_batched_axes(
              (_ax(node.inputs[1]), _ax(node.inputs[2]))
          )
      elif node.is_primitive('switch'):
        if _ax(node.inputs[0]) is not None:
          _validate_batched_switch_lookup(
              node.kwargs.get('lookup'), len(node.inputs) - 1, node
          )
          nxt = 0
        else:
          nxt = _merge_batched_axes(tuple(_ax(a) for a in node.inputs[1:]))
      elif node.is_stateful:
        if any(_ax(inp) is not None for inp in node.inputs):
          raise BatchError(
              f'vmap does not support batched stateful operator '
              f'{node.name!r} (op={node.op!r}); use batch_streams() instead.'
          )
        nxt = None
      else:
        inst = getattr(node.step_func, '_psyre_inst_func', None)
        has_batched = any(_ax(inp) is not None for inp in node.inputs)
        prim = get_primitive(node)
        is_collective = (
            axis_name is not None
            and inst is not None
            and (
                node.op in _COLLECTIVE_OPS
                or getattr(prim, 'name', '') in _COLLECTIVE_OPS
            )
        )
        if has_batched and inst is None:
          raise BatchError(
              f'vmap cannot batch primitive node {node.name!r} '
              f'(op={node.op!r}).'
          )
        nxt = 0 if (has_batched or is_collective) else None
      if nxt != prev:
        batch_axis[out_id] = nxt
        changed = True
  return batch_axis


def _sample_ref_to_clock(
    ref_stream: Any,
    node_clk: Clock | None,
    arm_scopes: tuple[ArmScope, ...],
    mapped: Mapping[int, Stream],
    outer_arms: tuple[ArmScope, ...],
    outer_prefixes: tuple[tuple[str, int], ...],
    cache: dict[tuple[Any, ...], Stream],
) -> Any:
  """Samples a base-clock reference stream down to a node's activation clock."""
  ref_key = ref_stream.id if isinstance(ref_stream, Stream) else id(ref_stream)
  if node_clk is not None and node_clk.steps:
    steps = [
        (mapped.get(st.selector.id, st.selector), st.to_when_selector())
        for st in node_clk.steps
    ]
    key: list[Any] = [ref_key]
  elif arm_scopes:
    steps = [
        (mapped.get(sc.cond.id, sc.cond), sc.selector) for sc in arm_scopes
    ]
    key = ['arms', ref_key]
  else:
    return ref_stream

  cur = ref_stream
  for sel, ws in steps:
    sel_key = sel.id if isinstance(sel, Stream) else id(sel)
    key.append((sel_key, ws))
    k_tuple = tuple(key)
    cached = cache.get(k_tuple)
    if cached is None:
      with CapturedEnv(arms=outer_arms, prefixes=outer_prefixes).restore():
        cached = emit_when(sel, cur, ws)
      cache[k_tuple] = cached
    cur = cached
  return cur


def _remap_node_comb_step(
    node: Node,
    in_axes_tuple: tuple[int | None, ...],
    *,
    axis_name: Any,
    axis_size: int | None,
    spmd_axis_name: Any,
) -> Callable[..., Any]:
  """Wraps a combinational `Node`'s `inst_func` in `jax.vmap`."""
  inst_func = getattr(node.step_func, '_psyre_inst_func')
  node_kwargs = node.kwargs

  def _per_lane(*vals: Any) -> Any:
    call_args, call_kwargs = materialize_comb_call(node_kwargs, vals)
    return inst_func(*call_args, **call_kwargs) if call_kwargs else inst_func(
        *call_args
    )

  vmapped = _jax_vmap(
      _per_lane,
      in_axes_tuple,
      0,
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
  )
  label = f'vmap({node.op})'
  vmapped.__name__ = label
  return make_jax_step_func(
      vmapped, name=label, pure=is_pure_step_func(node.step_func)
  )


def _where_tree(cond: Any, t_val: Any, f_val: Any) -> Any:
  cond_b = jnp.asarray(cond, dtype=jnp.bool_)
  return jax.tree_util.tree_map(
      lambda t, f: jnp.where(cond_b, t, f), t_val, f_val
  )


def _make_batched_select_step(
    c_ax: int | None, t_ax: int | None, f_ax: int | None
) -> Callable[..., Any]:
  """Builds a `jax.vmap(jnp.where)` step for batched `ifthenelse`."""
  sel_fn = jax.vmap(_where_tree, in_axes=(c_ax, t_ax, f_ax), out_axes=0)
  sel_fn.__name__ = 'vmap(ifthenelse)'
  return make_jax_step_func(sel_fn, name='vmap(ifthenelse)', pure=True)


def _select_n_leaf(idx: jax.Array, *leaves: Any) -> jax.Array:
  dt = jnp.result_type(*leaves)
  casted = tuple(jnp.asarray(x, dtype=dt) for x in leaves)
  return _jax_lax.select_n(idx, *jnp.broadcast_arrays(*casted))


def _make_batched_switch_step(
    kind: str,
    sel_ax: int | None,
    arm_axes: tuple[int | None, ...],
    *,
    has_default: bool = False,
) -> Callable[..., Any]:
  """Builds a vectorized `where` / `select_n` step for batched `switch`."""
  if kind == 'bool':

    def _sw_bool(sel_v: Any, f_v: Any, t_v: Any) -> Any:
      return _where_tree(sel_v, t_v, f_v)

    v_sw = jax.vmap(
        _sw_bool, in_axes=(sel_ax, arm_axes[0], arm_axes[1]), out_axes=0
    )
  else:

    def _sw_int(sel_v: Any, *arm_vals: Any) -> Any:
      sel_arr = jnp.asarray(sel_v)
      if not jnp.issubdtype(sel_arr.dtype, jnp.integer):
        raise BatchError(
            f'Batched integer switch selector must have integer dtype, got '
            f'{sel_arr.dtype}.'
        )
      if has_default:
        default_pos = len(arm_vals) - 1
        in_range = (sel_arr >= 0) & (sel_arr < default_pos)
        idx = jnp.where(in_range, sel_arr, default_pos).astype(jnp.int32)
      else:
        idx = jnp.clip(sel_arr, 0, len(arm_vals) - 1).astype(jnp.int32)
      return jax.tree_util.tree_map(
          lambda *xs: _select_n_leaf(idx, *xs), *arm_vals
      )

    v_sw = jax.vmap(_sw_int, in_axes=(sel_ax, *arm_axes), out_axes=0)
  v_sw.__name__ = 'vmap(switch)'
  return make_jax_step_func(v_sw, name='vmap(switch)', pure=True)


def _replay_batched_subgraph(
    sub_graph: Graph,
    outer_builder: GraphBuilder,
    ph_to_outer: Mapping[int, Stream],
    ph_axes: Mapping[int, int | None],
    batch_axis: Mapping[int, int | None],
    *,
    axis_name: Any,
    axis_size: int | None,
    spmd_axis_name: Any,
    scope_name: str,
    out_axes: Any,
) -> Any:
  """Replays a batched `sub_graph` into the caller's `outer_builder`."""
  try:
    clocks = check_clocks(sub_graph)
  except ClockError:
    if outer_builder.semantics != Semantics.KAHN:
      raise
    clocks = None

  def _is_base_outer(stream: Any) -> bool:
    if not isinstance(stream, Stream):
      return False
    prod = outer_builder.graph.producer(stream)
    return prod is None or (
        not prod.arms and not prod.is_primitive('when', 'whennot')
    )

  mapped_phs = [
      (ph_to_outer[pid], ax) for pid, ax in ph_axes.items() if ax is not None
  ]
  root_ref_pair = next(
      ((s, ax) for s, ax in mapped_phs if _is_base_outer(s)),
      mapped_phs[0] if mapped_phs else None,
  )
  outer_arms = arm_stack.get()
  outer_resets = reset_stack.get()
  outer_prefixes = prefix_stack.get()
  scope_pfx = outer_prefixes + ((scope_name, len(outer_resets)),)
  mapped: dict[int, Stream] = dict(ph_to_outer)
  arm_cache: dict[int, ArmScope] = {}
  ref_cache: dict[tuple[Any, ...], Stream] = {}

  # Pre-allocate every replayed node's output `Stream` so cyclic `fby`/`pre`
  # back-edges link directly without re-entering thunks.
  for node in sub_graph.nodes:
    mapped[node.output.id] = Stream(name=node.output.name or node.op)

  def _resolve_in(s: Stream) -> Stream:
    return mapped.get(s.id, s)

  def _remap_arms(scopes: tuple[ArmScope, ...]) -> tuple[ArmScope, ...]:
    return remap_arm_scopes(
        scopes, _resolve_in, arm_cache, reset_depth_offset=len(outer_resets)
    )

  def _port_arms(
      orig_s: Stream, fallback_arms: tuple[ArmScope, ...]
  ) -> tuple[ArmScope, ...]:
    prod = sub_graph.producer(orig_s)
    if prod is not None and prod.arms:
      return outer_arms + _remap_arms(prod.arms)
    return fallback_arms

  def _align_port(
      orig_s: Stream,
      target_ax: int,
      target_clk: Clock | None,
      env_arms: tuple[ArmScope, ...],
      fallback_scopes: tuple[ArmScope, ...] = (),
  ) -> Stream:
    cur_s = _resolve_in(orig_s)
    cur_ax = batch_axis.get(orig_s.id)
    if cur_ax == target_ax:
      return cur_s
    if cur_ax is None:
      if axis_size is not None:
        step = _make_broadcast_step(0, axis_size, out_axis=target_ax)
        with CapturedEnv(arms=env_arms, prefixes=scope_pfx).restore():
          bcast = emit_comb_node(step, (cur_s,), {}, name='vmap_broadcast')
        if (
            target_clk is not None
            and target_clk.steps
            and len(env_arms) == len(outer_arms)
            and orig_s.static
        ):
          return _sample_ref_to_clock(
              bcast, target_clk, (), mapped, outer_arms, scope_pfx, ref_cache
          )
        return bcast
      if root_ref_pair is None:
        raise BatchError(
            'vmap needs at least one mapped input or axis_size to broadcast.'
        )
      base_ref, ref_ax = root_ref_pair
      sampled_ref = _sample_ref_to_clock(
          base_ref,
          target_clk,
          fallback_scopes,
          mapped,
          outer_arms,
          scope_pfx,
          ref_cache,
      )
      step = _make_broadcast_step(ref_ax, None, out_axis=target_ax)
      with CapturedEnv(arms=env_arms, prefixes=scope_pfx).restore():
        return emit_comb_node(
            step, (cur_s, sampled_ref), {}, name='vmap_broadcast'
        )
    move_step = _make_moveaxis_step(cur_ax, target_ax)
    with CapturedEnv(arms=env_arms, prefixes=scope_pfx).restore():
      return emit_comb_node(
          move_step, (cur_s,), {}, name=f'moveaxis_{cur_ax}_to_{target_ax}'
      )

  replayed_merges: list[tuple[Node, tuple[Any, ...], tuple[ArmScope, ...]]] = []

  def _node_env(node: Node) -> CapturedEnv:
    extra_pfx = tuple(
        (seg, len(outer_resets) + depth) for seg, depth in node.prefixes
    )
    return CapturedEnv(
        arms=outer_arms + _remap_arms(node.arms),
        prefixes=scope_pfx + extra_pfx,
        resets=outer_resets + tuple(_resolve_in(r) for r in node.resets),
    )

  def _emit_into_node(
      node: Node,
      in_streams: tuple[Any, ...],
      out_s: Stream,
      *,
      step_func: Callable[..., Any] | None = None,
      node_kwargs: dict[str, Any] | None = None,
      node_name: str | None = None,
  ) -> None:
    kw = dict(node.kwargs) if node_kwargs is None else node_kwargs
    env = _node_env(node)
    with env.restore():
      outer_builder.add_loop_node(
          node.step_func if step_func is None else step_func,
          *in_streams,
          o_stream=out_s,
          name=node.name or node.op if node_name is None else node_name,
          **kw,
      )
      if node.is_primitive('merge') and len(in_streams) >= 2:
        replayed_merges.append(
            (outer_builder.graph.nodes[-1], in_streams[1:], env.arms)
        )

  with push_prefix(scope_name):
    for node in sub_graph.nodes:
      out_s = mapped[node.output.id]
      out_ax = batch_axis.get(node.output.id)
      if out_ax is None or node.is_primitive(
          'when', 'whennot', 'pre', 'post', 'buffer'
      ):
        _emit_into_node(node, tuple(_resolve_in(p) for p in node.inputs), out_s)
        continue

      node_clk = clocks.node_clock(node) if clocks is not None else None
      in_axes_list = tuple(batch_axis.get(inp.id) for inp in node.inputs)
      node_full_arms = outer_arms + _remap_arms(node.arms)
      with _node_env(node).restore():
        if node.is_primitive('fby', 'morph'):
          aligned = tuple(
              _align_port(p, out_ax, node_clk, node_full_arms, node.arms)
              for p in node.inputs
          )
          _emit_into_node(node, aligned, out_s)
        elif node.is_primitive('prefix_sum'):
          data_s = _align_port(
              node.inputs[0], out_ax, node_clk, node_full_arms, node.arms
          )
          init_s = _align_port(
              node.inputs[1], out_ax, node_clk, node_full_arms, node.arms
          )
          prev_s = fby(init_s, out_s)
          add_fn = _jax_vmap(
              operator.add,
              (out_ax, out_ax),
              out_ax,
              axis_name=axis_name,
              axis_size=axis_size,
              spmd_axis_name=spmd_axis_name,
          )
          add_fn.__name__ = 'vmap(prefix_sum)'
          add_step = make_jax_step_func(
              add_fn, name='vmap(prefix_sum)', pure=True
          )
          _emit_into_node(
              node,
              (prev_s, data_s),
              out_s,
              step_func=add_step,
              node_kwargs={},
          )
        elif node.is_primitive('current'):
          x_clk = (
              clocks.input_clock(node, 1) if clocks is not None else node_clk
          )
          x_arms = _port_arms(node.inputs[1], node_full_arms)
          _emit_into_node(
              node,
              (
                  _resolve_in(node.inputs[0]),
                  _align_port(node.inputs[1], out_ax, x_clk, x_arms, node.arms),
                  _align_port(
                      node.inputs[2], out_ax, node_clk, node_full_arms,
                      node.arms,
                  ),
              ),
              out_s,
          )
        elif node.is_primitive('merge'):
          arm_ports = tuple(
              _align_port(
                  arm,
                  out_ax,
                  (
                      clocks.input_clock(node, 1 + i)
                      if clocks is not None
                      else node_clk
                  ),
                  _port_arms(arm, node_full_arms),
                  node.arms,
              )
              for i, arm in enumerate(node.inputs[1:])
          )
          _emit_into_node(
              node, (_resolve_in(node.inputs[0]), *arm_ports), out_s
          )
        elif node.is_primitive('scan'):
          aligned_ports = tuple(
              _align_port(p, out_ax, node_clk, node_full_arms, node.arms)
              for p in node.inputs
          )
          new_kw = dict(node.kwargs)
          plan: CombCallPlan | None = new_kw.get(CALL_PLAN_KWARG)
          if plan is not None:
            new_plan_kw: list[tuple[str, Any]] = []
            for k, v in plan.kwargs_template:
              if k == 'bin_op':
                v = _jax_vmap(
                    v,
                    (out_ax, out_ax),
                    out_ax,
                    axis_name=axis_name,
                    axis_size=axis_size,
                    spmd_axis_name=spmd_axis_name,
                )
              new_plan_kw.append((k, v))
            new_kw[CALL_PLAN_KWARG] = CombCallPlan(
                args_template=plan.args_template,
                kwargs_template=tuple(new_plan_kw),
            )
          else:
            new_kw['bin_op'] = _jax_vmap(
                new_kw['bin_op'],
                (out_ax, out_ax),
                out_ax,
                axis_name=axis_name,
                axis_size=axis_size,
                spmd_axis_name=spmd_axis_name,
            )
          _emit_into_node(node, aligned_ports, out_s, node_kwargs=new_kw)
        elif node.is_primitive('ifthenelse'):
          c_ax, t_ax, f_ax = in_axes_list
          if c_ax is None:
            _emit_into_node(
                node,
                (
                    _resolve_in(node.inputs[0]),
                    _align_port(
                        node.inputs[1], out_ax, node_clk, node_full_arms,
                        node.arms,
                    ),
                    _align_port(
                        node.inputs[2], out_ax, node_clk, node_full_arms,
                        node.arms,
                    ),
                ),
                out_s,
            )
          else:
            _emit_into_node(
                node,
                tuple(_resolve_in(p) for p in node.inputs),
                out_s,
                step_func=_make_batched_select_step(c_ax, t_ax, f_ax),
                node_kwargs={},
                node_name=f'{node.name or "ifthenelse"}_select',
            )
        elif node.is_primitive('switch'):
          sel_ax = in_axes_list[0]
          if sel_ax is None:
            arm_ports = tuple(
                _align_port(arm, out_ax, node_clk, node_full_arms, node.arms)
                for arm in node.inputs[1:]
            )
            _emit_into_node(
                node, (_resolve_in(node.inputs[0]), *arm_ports), out_s
            )
          else:
            kind, order, has_default = _validate_batched_switch_lookup(
                node.kwargs.get('lookup'), len(node.inputs) - 1, node
            )
            arm_axes = tuple(
                batch_axis.get(node.inputs[1 + i].id) for i in order
            )
            ordered_ports = tuple(
                _resolve_in(node.inputs[1 + i]) for i in order
            )
            _emit_into_node(
                node,
                (_resolve_in(node.inputs[0]), *ordered_ports),
                out_s,
                step_func=_make_batched_switch_step(
                    kind, sel_ax, arm_axes, has_default=has_default
                ),
                node_kwargs={},
                node_name=f'{node.name or "switch"}_select_n',
            )
        else:
          step = _remap_node_comb_step(
              node,
              in_axes_list,
              axis_name=axis_name,
              axis_size=axis_size,
              spmd_axis_name=spmd_axis_name,
          )
          _emit_into_node(
              node,
              tuple(_resolve_in(p) for p in node.inputs),
              out_s,
              step_func=step,
              node_kwargs={},
          )

  # Populate `merge_arms` once all replayed nodes are registered so recursive
  # `automaton`/`match` arm confinement checks pass on `outer_builder.finalize`.
  for emitted_merge, arm_streams, fallback_arms in replayed_merges:
    m_arms: list[tuple[ArmScope, ...]] = []
    for arm_stream in arm_streams:
      prod = (
          outer_builder.graph.producer(arm_stream)
          if isinstance(arm_stream, Stream)
          else None
      )
      m_arms.append(
          prod.arms if prod is not None and prod.arms else fallback_arms
      )
    emitted_merge.merge_arms = tuple(m_arms)

  def _map_out_leaf(val: Any, spec: Any) -> Any:
    if not isinstance(val, Stream):
      return val
    target_ax = validate_optional_axis(spec, 'out_axes')
    cur_ax = batch_axis.get(val.id)
    replayed = _resolve_in(val)
    if target_ax is None:
      if cur_ax is not None:
        raise BatchError(
            f'vmap out_axes=None requires unbatched output, got batched '
            f'stream {val.name!r} (axis={cur_ax}).'
        )
      return replayed
    if cur_ax == target_ax:
      return replayed
    out_clk = clocks.stream_clock(val) if clocks is not None else None
    if out_clk is not None and not out_clk.steps:
      out_clk = None
    with push_prefix(scope_name):
      return _align_port(val, target_ax, out_clk, outer_arms)

  return map_tree_with_axes(
      sub_graph.output,
      out_axes,
      _map_out_leaf,
      label='out_axes',
      is_leaf=lambda v: isinstance(v, Stream),
      error_cls=BatchError,
  )


def vmap(
    fn: Callable[..., Any],
    in_axes: Any = 0,
    out_axes: Any = 0,
    *,
    axis_name: Any = None,
    axis_size: int | None = None,
    spmd_axis_name: Any = None,
    name: str = '',
) -> Callable[..., Any]:
  """Vectorizes a lifted JAX operator or composite Stage-1 Psyre subgraph.

  Supports both single lifted `@psyre_comb` / `psyre.jax.lax` / `jax.numpy`
  ops (commuting directly with `@psyre_comb` via `jax.vmap`) and composite
  Stage-1 Psyre subgraphs containing `fby`, `pre`, `morph`, `current`, `post`,
  `prefix_sum`, `scan`, `ifthenelse`, `switch`, `when`, `merge`, `match`,
  `automaton`, and recursive feedback loops.

  Under Lustre synchronous semantics, clock/control selectors (`when`/`whennot`/
  `merge`/`buffer` predicates, `current` hold condition, and `reset` signals)
  must be unbatched (`axis=None`); attempting to batch one raises `BatchError`.
  Use `batch_streams()` for heterogeneous per-lane clocks or automata.

  Args:
    fn: Lifted combinational operator or composite Stage-1 callable.
    in_axes: Integer axis, `None` (broadcast), or PyTree matching positional
      inputs.
    out_axes: Integer axis, `None` (unbatched output), or PyTree matching
      returned outputs.
    axis_name: Optional `jax.vmap` collective axis name.
    axis_size: Optional explicit batch size.
    spmd_axis_name: Optional `jax.vmap` `spmd_axis_name`.
    name: Optional Stage-1 prefix scope / node name.

  Returns:
    Vectorized callable taking/returning batched `Stream`s (or plain JAX values
    outside Stage 1).
  """
  if not is_callable_target(fn):
    raise TypeError(f'psyre.jax.vmap expects a callable, got {type(fn)}.')
  if axis_size is not None:
    validate_positive_int(axis_size, 'vmap axis_size')

  if getattr(fn, '_psyre_inst_func', None) is not None:
    return _vmap_single_comb(
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
        name=name,
    )

  scope_name = name or f'vmap({callable_name(fn, "fn")})'

  def wrapper(*args: Any, **kwargs: Any) -> Any:
    outer_builder = active_builder.get()
    if outer_builder is None:
      inner = fn if not kwargs else functools.partial(fn, **kwargs)
      return _jax_vmap(
          inner,
          in_axes,
          out_axes,
          axis_name=axis_name,
          axis_size=axis_size,
          spmd_axis_name=spmd_axis_name,
      )(*args)

    arg_axes = normalize_arg_axes(
        in_axes, len(args), 'vmap in_axes', error_cls=BatchError
    )
    if any(_has_dynamic_leaf(v) for v in kwargs.values()):
      raise BatchError(
          'Composite psyre.jax.vmap does not accept dynamic Stream keyword '
          'arguments; pass dynamic streams positionally.'
      )

    sub_builder = GraphBuilder(semantics=outer_builder.semantics)
    ph_to_outer: dict[int, Stream] = {}
    ph_axes: dict[int, int | None] = {}

    def _bind_placeholder(leaf: Any, ax: Any) -> Any:
      if leaf is None:
        return None
      if is_dynamic_leaf(leaf):
        slot_ax = validate_optional_axis(ax, 'in_axes')
        ph = Stream(name=f'_vmap_in_{len(ph_to_outer)}')
        ph_to_outer[ph.id] = leaf
        ph_axes[ph.id] = slot_ax
        return ph
      if is_int_axis(ax):
        ph = Stream(name=f'_vmap_lit_{len(ph_to_outer)}')
        ph_to_outer[ph.id] = Stream.constant(jnp.asarray(leaf))
        ph_axes[ph.id] = ax
        return ph
      return leaf

    token = active_builder.set(sub_builder)
    try:
      with CapturedEnv().restore():
        sub_args = tuple(
            map_tree_with_axes(
                a, ax, _bind_placeholder, label='in_axes', error_cls=BatchError
            )
            for a, ax in zip(args, arg_axes)
        )
        if all(ax is None for ax in ph_axes.values()) and axis_size is None:
          raise BatchError(
              'vmap requires at least one mapped input axis or explicit '
              'axis_size.'
          )
        sub_out = fn(*sub_args, **kwargs)
        sub_builder.bind_output(sub_out)
        sub_graph = sub_builder.finalize()
    finally:
      active_builder.reset(token)

    batch_axis = _infer_batch_axes(sub_graph, ph_axes, axis_name=axis_name)
    return _replay_batched_subgraph(
        sub_graph,
        outer_builder,
        ph_to_outer,
        ph_axes,
        batch_axis,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
        scope_name=scope_name,
        out_axes=out_axes,
    )

  return wrapper


def _matches_dtype(val: Any, source_dtype: Any) -> bool:
  """Returns True when `val` should be cast to `target_dtype`."""
  if not is_array_object(val) and not is_numpy_scalar(val):
    return source_dtype is None and isinstance(val, float)
  try:
    dt = jnp.result_type(val)
  except TypeError:
    return False
  if source_dtype is not None:
    return dt == jnp.dtype(source_dtype)
  return jnp.issubdtype(dt, jnp.floating)


def _cast_tree(val: Any, target_dtype: Any, source_dtype: Any) -> Any:
  return jax.tree_util.tree_map(
      lambda leaf: (
          jnp.asarray(leaf, dtype=target_dtype)
          if _matches_dtype(leaf, source_dtype)
          else leaf
      ),
      val,
  )


def rewrite_dtype(
    graph: Graph,
    *,
    target_dtype: Any,
    source_dtype: Any = None,
    ops: Iterable[str] | None = None,
    primitives: Iterable[Any] | None = None,
    preferred_element_type: Any = None,
    include_stream_ops: bool = False,
) -> Graph:
  """Returns a cloned `Graph` casting inputs of matching JAX/lax nodes.

  Useful for mixed-precision lowering (for example, casting `float32` operands
  to `bfloat16` on `dot_general` / `conv_general_dilated` / `__matmul__` nodes
  while optionally accumulating in `preferred_element_type=jnp.float32`).

  The original `graph` is left untouched; the returned `Graph` carries freshly
  wrapped `step_func`s on matching nodes with all clocks, resets, feedback
  edges, and outputs preserved.

  Args:
    graph: Finalized Stage-1 `Graph`.
    target_dtype: Destination JAX dtype (e.g., `jnp.bfloat16`, `jnp.float16`).
    source_dtype: Optional source dtype filter (defaults to any floating
      dtype).
    ops: Optional iterable of `node.op` names to rewrite (e.g., `('dot_general',
      '__matmul__')`).
    primitives: Optional iterable of JAX `Primitive` objects (or their `.name`
      strings) to rewrite.
    preferred_element_type: Optional accumulation dtype forwarded via
      `preferred_element_type=` to `dot_general` / `conv_general_dilated` /
      `matmul` when not already pinned.
    include_stream_ops: When `True`, also matches plain `Stream` binary/unary
      operator nodes (`__add__`, `__matmul__`, etc.).

  Returns:
    A newly finalized `Graph` with cloned `Stream` channels and cast wrappers.
  """
  graph.require_finalized()
  target_dt = jnp.dtype(target_dtype)
  src_dt = None if source_dtype is None else jnp.dtype(source_dtype)
  pref_dt = (
      None
      if preferred_element_type is None
      else jnp.dtype(preferred_element_type)
  )
  op_filter = None if ops is None else frozenset(ops)
  prim_filter_objs: set[Any] = set()
  prim_filter_names: set[str] = set()
  if primitives is not None:
    for p in primitives:
      if isinstance(p, str):
        prim_filter_names.add(p)
      else:
        prim_filter_objs.add(p)

  def _should_rewrite(node: Node) -> bool:
    if node.is_stateful:
      return False
    if getattr(node.step_func, '_psyre_inst_func', None) is None:
      return False
    backend_match = is_jax_step_func(node) or (
        include_stream_ops and node.is_stream_op()
    )
    if op_filter is None and primitives is None:
      return backend_match
    if op_filter is not None and node.op in op_filter and backend_match:
      return True
    prim = get_primitive(node)
    if primitives is not None and prim is not None:
      return (
          prim in prim_filter_objs
          or getattr(prim, 'name', '') in prim_filter_names
      )
    return False

  def _wrap_cast_step(node: Node) -> Callable[..., Any]:
    raw_fn = getattr(node.step_func, '_psyre_inst_func')
    node_kwargs = node.kwargs
    prim = get_primitive(node)
    prim_name = getattr(prim, 'name', '')
    use_pref = pref_dt is not None and (
        node.op in _PREFERRED_DTYPE_OPS or prim_name in _PREFERRED_DTYPE_PRIMS
    )

    def _cast_inst(*vals: Any) -> Any:
      c_args, c_kw = materialize_comb_call(node_kwargs, vals)
      c_args = tuple(_cast_tree(a, target_dt, src_dt) for a in c_args)
      if c_kw:
        c_kw = {
            k: (
                v
                if k in ('preferred_element_type', 'dtype')
                else _cast_tree(v, target_dt, src_dt)
            )
            for k, v in c_kw.items()
        }
      if use_pref and c_kw.get('preferred_element_type') is None:
        # Respect an already-supplied positional `preferred_element_type`
        # (`dot_general` arg 4 / `conv_general_dilated` arg 7).
        pos_pinned = (
            (prim_name == 'dot_general' or node.op == 'dot_general')
            and len(c_args) >= 5
            and c_args[4] is not None
        ) or (
            (
                prim_name == 'conv_general_dilated'
                or node.op == 'conv_general_dilated'
            )
            and len(c_args) >= 8
            and c_args[7] is not None
        )
        if not pos_pinned:
          c_kw = dict(c_kw)
          c_kw['preferred_element_type'] = pref_dt
      return raw_fn(*c_args, **c_kw) if c_kw else raw_fn(*c_args)

    _cast_inst.__name__ = callable_name(raw_fn, 'cast_op')
    return make_jax_step_func(
        _cast_inst,
        name=node.op,
        pure=is_pure_step_func(node.step_func),
        primitive=prim,
    )

  new_graph, stream_map = graph.clone_streams()
  for node in graph.nodes:
    if _should_rewrite(node):
      new_graph.copy_node_into(
          node, stream_map, step_func=_wrap_cast_step(node), kwargs={}
      )
    else:
      new_graph.copy_node_into(node, stream_map)

  if graph.output is not None:
    new_graph.set_output(
        map_output_leaves(lambda s: stream_map.get(s.id, s), graph.output)
    )
  new_graph.mark_finalized()
  return new_graph
