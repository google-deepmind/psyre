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
"""Whole-program dataflow Graph optimizations and dead-code elimination (DCE).

Applies clock-preserving algebraic simplifications (`when`/`whennot`, `merge`/
`switch`, `ifthenelse`, boolean identities), pure combinational CSE, and
backward reachability DCE to a fixed point, preserving Lustre/Lucid Synchrone
clock polymorphism and synchronous/Kahn equivalence (Halbwachs et al., *Proc.
IEEE*, 1991; Caspi & Pouzet, ICFP 1996; Colaço & Pouzet, EMSOFT 2003).
"""

import collections
from collections.abc import Iterable
import dataclasses
import math
from typing import Any

from psyre import analysis
from psyre import clock
from psyre import common
from psyre import common_payload
from psyre import graph as graph_lib
from psyre import static_stream as static_stream_lib
from psyre import stream as stream_lib

Graph = graph_lib.Graph
Node = graph_lib.Node
Stream = stream_lib.Stream
MergeSelector = common.MergeSelector
TagKey = common.TagKey
WhenSelector = common.WhenSelector

_SIMPLIFY_BOOL_OPS = frozenset({
    '__and__', '__rand__', '__or__', '__ror__', '__invert__',
})
_BOOL_STREAM_OPS = (
    common.BOOL_PRODUCER_OPS | common.BOOL_BINARY_OPS | {'__invert__'}
)
_MAX_TAG_DOMAIN_SIZE = 256


def _zero_sign(value: Any) -> Any:
  """Returns a hashable signed-zero discriminator for floats/complex scalars."""
  if isinstance(value, float):
    return math.copysign(1.0, value) if value == 0.0 else 0.0
  if isinstance(value, complex):
    return (
        math.copysign(1.0, value.real) if value.real == 0.0 else 0.0,
        math.copysign(1.0, value.imag) if value.imag == 0.0 else 0.0,
    )
  if common_payload.is_numpy_scalar(value):
    py_val = value.item()
    if isinstance(py_val, (float, complex)):
      return _zero_sign(py_val)
  return 0.0


def _exact_value_key(value: Any) -> Any | None:
  """Returns a hashable exact key separating types and signed zeros, or None."""
  if common_payload.is_nan_scalar(value):
    return None
  if common_payload.is_array_object(value):
    return None
  cls = type(value)
  if isinstance(value, tuple):
    parts = tuple(_exact_value_key(item) for item in value)
    if any(p is None for p in parts):
      return None
    return (cls, parts)
  if isinstance(value, frozenset):
    items = [_exact_value_key(item) for item in value]
    if any(p is None for p in items):
      return None
    try:
      return (cls, frozenset(items))
    except (TypeError, ValueError, RuntimeError):
      return None
  key = (cls, value, _zero_sign(value))
  try:
    hash(key)
  except (TypeError, ValueError, RuntimeError):
    return None
  return key


def _same_kwarg_value(a: Any, b: Any) -> bool:
  """Returns True iff two static node kwarg values are safely equivalent."""
  if a is b:
    return not common_payload.is_nan_scalar(a)
  if type(a) is not type(b):
    return False
  if callable(a) and not isinstance(
      a, (common.WhenSelector, common.MergeSelector)
  ):
    # Distinct functions/types (`a is not b`) never equate via `__eq__`.
    # `WhenSelector` and `MergeSelector` are frozen value dataclasses whose
    # `__call__` should still compare structurally via `same_const_value`.
    return False
  if isinstance(a, (common_payload.CombCallPlan, common_payload.PlanNode)):
    return all(
        _same_kwarg_value(getattr(a, field.name), getattr(b, field.name))
        for field in dataclasses.fields(a)
    )
  if isinstance(a, slice):
    return (
        _same_kwarg_value(a.start, b.start)
        and _same_kwarg_value(a.stop, b.stop)
        and _same_kwarg_value(a.step, b.step)
    )
  if isinstance(a, tuple):
    return len(a) == len(b) and all(
        _same_kwarg_value(x, y) for x, y in zip(a, b)
    )
  if isinstance(a, frozenset):
    if len(a) != len(b):
      return False
    # Compare elements by exact type/zero-sign keys so `frozenset({1})` never
    # matches `frozenset({True})` and `{0.0}` never matches `{-0.0}`.
    key_a = _exact_value_key(a)
    key_b = _exact_value_key(b)
    if key_a is None or key_b is None:
      return False
    try:
      return bool(key_a == key_b)
    except (TypeError, ValueError, RuntimeError):
      return False
  if common_payload.is_array_object(a) or isinstance(a, (list, dict, set)):
    return False
  return static_stream_lib.same_const_value(a, b)


def _same_kwargs(
    ka: dict[str, Any], kb: dict[str, Any]
) -> bool:
  """Returns True iff two `node.kwargs` mappings are safely equivalent."""
  if not ka and not kb:
    return True
  if ka.keys() != kb.keys():
    return False
  return all(_same_kwarg_value(ka[k], kb[k]) for k in ka)


def _tag_value_key(value: Any) -> Any | None:
  """Returns a hashable exact key separating types and signed zeros, or None."""
  key = _exact_value_key(value)
  if key is None:
    return None
  try:
    hash(common.tag_key(value))
  except (TypeError, ValueError, RuntimeError):
    return None
  return key


def _normalize_extra_roots(
    extra_roots: Iterable[Any] | None,
) -> list[Any] | None:
  """Validates `extra_roots` and materializes it into a list (or None)."""
  if extra_roots is None:
    return None
  if isinstance(extra_roots, str):
    raise TypeError(
        'extra_roots must be an iterable of streams/nodes/names, not a bare '
        'str.'
    )
  return list(extra_roots)


def _collect_root_streams(
    graph: Graph,
    extra_roots: Iterable[Any] | None = None,
) -> list[Stream]:
  """Collects unique root Streams from graph.output and optional extra_roots."""
  roots: list[Stream] = list(graph.output_streams())
  seen: set[int] = {s.id for s in roots}

  def add_stream(stream: Stream) -> None:
    if stream.id not in seen:
      seen.add(stream.id)
      roots.append(stream)

  extra_list = _normalize_extra_roots(extra_roots)
  if extra_list is not None:
    named_roots: list[str] = []
    for item in extra_list:
      if isinstance(item, str):
        named_roots.append(item)
      elif isinstance(item, Node):
        graph.validate_node(item)
        add_stream(item.output)
      elif isinstance(item, Stream):
        add_stream(graph.resolve_stream(item))
      else:
        raise TypeError(
            'extra_roots elements must be Stream, Node, or '
            f'str, got {type(item).__name__}.'
        )
    if named_roots:
      for stream in graph.resolve_named_streams(named_roots).values():
        add_stream(stream)

  return roots


def find_live_nodes(
    graph: Graph,
    *,
    extra_roots: Iterable[Any] | None = None,
) -> tuple[Node, ...]:
  """Returns all Nodes transitively feeding graph.output (or extra_roots).

  Traverses backward from `graph.output` (and any `extra_roots`) across data
  `inputs`, synchronous/asynchronous `resets`, and enclosing `ArmScope`
  condition streams.

  Args:
    graph: Finalized Psyre `Graph` to analyze.
    extra_roots: Optional iterable of additional root `Stream`s, `Node`s, or
      stream names to treat as live outputs.

  Returns:
    Tuple of live `Node`s in their original instantiation order.

  Raises:
    RuntimeError: If `graph` has not been finalized.
  """
  graph.require_finalized()
  root_streams = _collect_root_streams(graph, extra_roots=extra_roots)
  visited_ids: set[int] = {s.id for s in root_streams}
  queue: collections.deque[Stream] = collections.deque(root_streams)
  live_ids: set[int] = set()

  def enqueue(stream: Stream) -> None:
    if stream.id not in visited_ids:
      visited_ids.add(stream.id)
      queue.append(stream)

  while queue:
    stream = queue.popleft()
    prod = graph.producer(stream)
    if prod is None or prod.id in live_ids:
      continue
    live_ids.add(prod.id)
    for in_stream in prod.input_streams():
      enqueue(in_stream)
    for scope in prod.arms:
      if isinstance(scope.cond, Stream):
        enqueue(graph.resolve_stream(scope.cond))

  return tuple(node for node in graph.nodes if node.id in live_ids)


def find_dead_nodes(
    graph: Graph,
    *,
    extra_roots: Iterable[Any] | None = None,
) -> tuple[Node, ...]:
  """Returns all Nodes in `graph` that do not transitively feed any output.

  Args:
    graph: Finalized Psyre `Graph` to analyze.
    extra_roots: Optional iterable of additional root streams, nodes, or names.

  Returns:
    Tuple of dead `Node`s in their original instantiation order.
  """
  live_ids = {
      node.id for node in find_live_nodes(graph, extra_roots=extra_roots)
  }
  return tuple(node for node in graph.nodes if node.id not in live_ids)


def eliminate_dead_code(
    graph: Graph,
    *,
    extra_roots: Iterable[Any] | None = None,
) -> Graph:
  """Eliminates dead Nodes not transitively reachable from `graph.output`.

  Mutates `graph` in place via `graph.retain_nodes(...)`, renumbering remaining
  live `Node.id`s contiguously (`0 .. len(graph.nodes) - 1`) and pruning dead
  producer/consumer mappings and unreferenced dead output streams.

  Args:
    graph: Finalized Psyre `Graph` to prune.
    extra_roots: Optional iterable of additional root streams, nodes, or names
      to keep alive alongside `graph.output`.

  Returns:
    The pruned `graph`.
  """
  live_nodes = find_live_nodes(graph, extra_roots=extra_roots)
  graph.retain_nodes(live_nodes)
  return graph


def _clock_compatible(
    clocks: clock.ClockAnalysis, node: Node, replacement: Stream
) -> bool:
  """Returns True when `replacement` preserves `node.output`'s clock.

  Unpinned static streams are clock-polymorphic (`∀ 'a. 'a`) and have no
  `Absent` ticks or finite-horizon `KAHN` token-count cap of their own, so they
  may replace a `Node` only when `node.output` executes on `BASE_CLOCK` (`.`).
  Dynamic streams and base-clock-pinned anchors must match `node.output`'s exact
  inferred clock.

  Args:
    clocks: Inferred `ClockAnalysis` for the finalized `Graph`.
    node: Candidate `Node` whose output stream would be replaced.
    replacement: Replacement `Stream` candidate.

  Returns:
    `True` if `replacement` matches `node.output`'s clock.
  """
  out_clk = clocks.stream_clock(node.output)
  if clocks.graph.is_polymorphic_static(replacement):
    return out_clk.is_base
  return clocks.stream_clock(replacement) == out_clk


def _may_truncate_eof(graph: Graph) -> frozenset[int]:
  """Returns stream IDs transitively reachable from any `post` node."""
  queue: collections.deque[Stream] = collections.deque(
      node.output for node in graph.nodes if node.is_primitive('post')
  )
  if not queue:
    return frozenset()
  seen: set[int] = {s.id for s in queue}
  while queue:
    stream = queue.popleft()
    for consumer in graph.consumers(stream):
      if consumer.output.id not in seen:
        seen.add(consumer.output.id)
        queue.append(consumer.output)
  return frozenset(seen)


def _preserves_eof(
    node: Node, replacement: Stream, eof_ids: frozenset[int]
) -> bool:
  """Returns True when folding `node` to `replacement` keeps `post` EOF."""
  if not eof_ids:
    return True
  # Double negation `~(~x) -> x` stays in `x`'s `post` EOF cone; disallow
  # dropping a `post`-reset constant source `~c -> Stream(not c)`.
  if node.op == '__invert__':
    return node.inputs[0].id not in eof_ids or replacement.id in eof_ids
  # `merge` eagerly pulls only selector `inputs[0]`; arms are lazy and guarded.
  if node.op == 'merge':
    return node.inputs[0].id not in eof_ids or node.inputs[0] is replacement
  return not any(s is not replacement and s.id in eof_ids for s in node.inputs)


def _unwrap_matching_when(
    graph: Graph,
    clocks: clock.ClockAnalysis,
    mux_node: Node,
    arm_h: Stream,
    sel_s: Stream,
    expected_pred: tuple[bool, frozenset[TagKey], bool],
) -> Stream | None:
  """Returns payload Stream if `arm_h` is `when`/`whennot(sel_s, payload)`."""
  prod = graph.producer(arm_h)
  if (
      prod is None
      or not prod.is_primitive('when', 'whennot')
      or len(prod.inputs) < 2
      or prod.inputs[0] is not sel_s
      or clocks.node_clock(prod) != clocks.node_clock(mux_node)
  ):
    return None
  actual_pred = graph.normalize_when_selector(
      sel_s, prod.kwargs.get('selector'), op=prod.op
  )
  return prod.inputs[1] if actual_pred == expected_pred else None


def _is_const_value(graph: Graph, stream: Stream, expected: Any) -> bool:
  """Returns True when `stream` is a 1-periodic constant matching `expected`."""
  const_s = graph.one_periodic_stream(stream)
  return const_s is not None and static_stream_lib.same_const_value(
      const_s.const_value, expected
  )


def _selector_tag_domain(
    graph: Graph, sel_s: Stream
) -> tuple[Any, ...] | None:
  """Returns a finite upper bound of values `sel_s` can emit, or None.

  Uses `Graph.static_source` and `Node.value_forward_args` to propagate exact
  values (distinguishing Python types and signed float zeros) to a fixed point,
  and checks every nested non-default `merge`/`switch` selector against its own
  arms so folding never masks a `KeyError`.

  Args:
    graph: Finalized Stage-1 `Graph` being optimized.
    sel_s: Selector `Stream` whose finite value domain is computed.

  Returns:
    Tuple of reachable discrete values, or `None` if unbounded.
  """
  root = graph.resolve_stream(sel_s)
  queue: collections.deque[Stream] = collections.deque([root])
  seen: set[int] = {root.id}
  seed_vals: dict[int, dict[Any, Any]] = {}
  forward_deps: dict[int, list[int]] = {}
  mux_checks: list[tuple[int, MergeSelector]] = []

  def enqueue(stream: Stream) -> int:
    resolved = graph.resolve_stream(stream)
    if resolved.id not in seen:
      seen.add(resolved.id)
      queue.append(resolved)
    return resolved.id

  def add_values(
      bucket: dict[Any, Any], values: Iterable[Any]
  ) -> bool:
    for val in values:
      key = _tag_value_key(val)
      if key is None:
        return False
      bucket[key] = val
      if len(bucket) > _MAX_TAG_DOMAIN_SIZE:
        return False
    return True

  while queue:
    curr = queue.popleft()
    bucket = seed_vals.setdefault(curr.id, {})
    deps = forward_deps.setdefault(curr.id, [])
    static_src = graph.static_source(curr)
    if static_src is not None and static_src.static_sequence is not None:
      if not add_values(bucket, static_src.static_sequence.iter_atoms()):
        return None
      continue
    prod = graph.producer(curr)
    if prod is None:
      return None
    if (
        prod.op in _BOOL_STREAM_OPS
        and prod.is_stream_op()
        and graph.is_bool_stream(curr)
    ):
      add_values(bucket, (True, False))
    elif prod.is_primitive('pre') and prod.inputs:
      if not add_values(bucket, (None,)):
        return None
      deps.append(enqueue(prod.inputs[0]))
    elif (fwd := prod.value_forward_args()) is not None:
      if not fwd:
        return None
      for src in fwd:
        deps.append(enqueue(src))
      if prod.op in {'merge', 'switch'}:
        lk = prod.kwargs.get('lookup')
        if not isinstance(lk, MergeSelector):
          return None
        # Only non-default muxes can raise `KeyError` on missing tags; record
        # them to verify exhaustiveness (default muxes fold like `ifthenelse`).
        if not lk.is_boolean and lk.default_index is None:
          mux_checks.append((enqueue(prod.inputs[0]), lk))
    else:
      return None

  changed = True
  while changed:
    changed = False
    for sid, deps in forward_deps.items():
      cur = seed_vals[sid]
      for dep_id in deps:
        for key, val in seed_vals[dep_id].items():
          if key not in cur:
            cur[key] = val
            if len(cur) > _MAX_TAG_DOMAIN_SIZE:
              return None
            changed = True

  for mux_sel_id, lk in mux_checks:
    sel_vals = seed_vals[mux_sel_id]
    if not sel_vals:
      return None
    try:
      if any(common.tag_key(v) not in lk.arms for v in sel_vals.values()):
        return None
    except (TypeError, ValueError, RuntimeError):
      return None

  root_vals = seed_vals[root.id]
  return tuple(root_vals.values()) if root_vals else None


def _const_bool_value(graph: Graph, stream: Stream) -> bool | None:
  """Returns the constant `bool` emitted by `stream` when present, or `None`."""
  has_val, val = graph.const_value(stream)
  return val if has_val and isinstance(val, bool) else None


def _simplify_bool_op(
    graph: Graph, clocks: clock.ClockAnalysis, node: Node
) -> Stream | None:
  """Returns a replacement Stream for a degenerate boolean `&`/`|`/`~` op."""
  if node.op == '__invert__':
    if len(node.inputs) != 1:
      return None
    a_h = node.inputs[0]
    if not graph.is_bool_stream(a_h):
      return None
    const_s = graph.one_periodic_stream(a_h)
    if const_s is not None and isinstance(const_s.const_value, bool):
      if clocks.stream_clock(node.output).is_base:
        return Stream(not const_s.const_value)
      return None
    prod = graph.producer(a_h)
    if (
        prod is not None
        and prod.is_stream_op('__invert__')
        and len(prod.inputs) == 1
        and graph.is_bool_stream(prod.inputs[0])
    ):
      return prod.inputs[0]
    return None
  if len(node.inputs) != 2:
    return None
  a_h, b_h = node.inputs[0], node.inputs[1]
  if not (graph.is_bool_stream(a_h) and graph.is_bool_stream(b_h)):
    return None
  a_val = _const_bool_value(graph, a_h)
  b_val = _const_bool_value(graph, b_h)
  is_and = node.op in {'__and__', '__rand__'}
  identity, absorb = (True, False) if is_and else (False, True)
  for target in (identity, absorb):
    if a_val is target and b_val is target:
      return a_h if _clock_compatible(clocks, node, a_h) else b_h
    if a_val is target:
      return b_h if target is identity else a_h
    if b_val is target:
      return a_h if target is identity else b_h
  if stream_lib.same_stream_or_const(a_h, b_h):
    return a_h
  return None


def _simplify_ifthenelse(graph: Graph, node: Node) -> Stream | None:
  """Returns a candidate replacement Stream for a degenerate `ifthenelse`."""
  if len(node.inputs) != 3:
    return None
  c_h, t_h, f_h = node.inputs
  has_c, c_val = graph.const_value(c_h)
  if has_c:
    truth = common.when_selector_matches_const(None, c_val)
    if truth is not None:
      return t_h if truth else f_h
  if stream_lib.same_stream_or_const(t_h, f_h):
    return t_h
  if (
      graph.is_bool_stream(c_h)
      and _is_const_value(graph, t_h, True)
      and _is_const_value(graph, f_h, False)
  ):
    return c_h
  return None


def _simplify_when(graph: Graph, node: Node) -> tuple[Stream | None, bool]:
  """Normalizes boolean `WhenSelector` and folds trivial `when`/`whennot`."""
  if len(node.inputs) < 2:
    return None, False
  sel_h, data_h = node.inputs[0], node.inputs[1]
  selector = node.kwargs.get('selector')
  rewired = False
  norm = graph.normalize_when_selector(sel_h, selector, op=node.op)
  if (
      node.op == 'when'
      and isinstance(selector, WhenSelector)
      and not selector.is_boolean
      and norm[0]
      and len(norm[1]) == 1
  ):
    selector = common.selector_from_norm(norm)
    if selector is None:
      node.kwargs.pop('selector', None)
    else:
      node.kwargs['selector'] = selector
    rewired = True
  if graph.is_always_active_when(sel_h, selector, op=node.op, norm=norm):
    return data_h, rewired
  return None, rewired


def _simplify_mux(
    graph: Graph,
    clocks: clock.ClockAnalysis,
    node: Node,
    eof_ids: frozenset[int],
) -> tuple[Stream | None, bool]:
  """Folds constant/identity/uniform muxes and normalizes boolean muxes.

  Under uniform static clock polymorphism, each arm of `merge` maintains an
  independent cursor over any un-sampled multi-element static `Stream`. Only
  `switch` arms (pulled in lockstep every tick), 1-periodic constant arms
  (value-invariant under downsampling), tautological single-arm predicates, and
  explicit `when`/`whennot(sel_s, payload)` arms on the same `node_clock` may
  be unwrapped.

  Args:
    graph: Finalized `Graph` being simplified.
    clocks: Inferred `ClockAnalysis` for `graph`.
    node: Candidate `merge` or `switch` `Node`.
    eof_ids: Stream IDs transitively reachable from any `post` node.

  Returns:
    `(replacement, rewired)` where `replacement` is a candidate replacement
    `Stream` (or `None`), and `rewired` is `True` when `node` was rewritten to
    a normalized boolean mux in place.
  """
  lookup = node.kwargs.get('lookup')
  if not isinstance(lookup, MergeSelector) or len(node.inputs) < 2:
    return None, False
  sel_s = node.inputs[0]
  arm_streams = node.inputs[1:]

  def unwrap_arm(arm_idx: int) -> Stream | None:
    arm_h = arm_streams[arm_idx]
    if node.op == 'switch':
      return arm_h
    const_arm = graph.one_periodic_stream(arm_h)
    if const_arm is not None:
      return const_arm
    pred = graph.normalize_when_selector(sel_s, lookup.arm_selector(arm_idx))
    if common.is_tautological_selector(pred):
      return arm_h
    return _unwrap_matching_when(graph, clocks, node, arm_h, sel_s, pred)

  # 0. Constant selector: pick the active arm (keeping KeyError intact).
  has_sel_val, sel_val = graph.const_value(sel_s)
  if has_sel_val:
    try:
      sel_idx = lookup(sel_val)
    except (KeyError, TypeError, ValueError):
      sel_idx = None
    if sel_idx is not None:
      cand = arm_streams[sel_idx]
      if _clock_compatible(clocks, node, cand):
        return cand, False

  sel_domain: tuple[Any, ...] | None = None
  if lookup.is_boolean or lookup.default_index is not None:
    is_exhaustive = True
  else:
    sel_domain = _selector_tag_domain(graph, sel_s)
    try:
      is_exhaustive = sel_domain is not None and all(
          common.tag_key(val) in lookup.arms for val in sel_domain
      )
    except (TypeError, ValueError, RuntimeError):
      is_exhaustive = False

  def _is_identity_arm(arm_idx: int, expected_val: Any) -> bool:
    src = unwrap_arm(arm_idx)
    return src is not None and (
        src is sel_s or _is_const_value(graph, src, expected_val)
    )

  # 1. Identity mux: every arm reproduces `sel_s` (or its exact-type tag).
  if is_exhaustive:
    identity_match = False
    if (
        lookup.is_boolean
        and len(arm_streams) == 2
        and graph.is_bool_stream(sel_s)
    ):
      identity_match = _is_identity_arm(0, True) and _is_identity_arm(1, False)
    elif (
        not lookup.is_boolean
        and lookup.default_index is None
        and lookup.arms
        and sel_domain is not None
    ):
      identity_match = all(
          _is_identity_arm(lookup.arms[common.tag_key(v)], v)
          for v in sel_domain
      ) and all(
          _is_identity_arm(idx, val) for (_, val), idx in lookup.arms.items()
      )
    if identity_match and _clock_compatible(clocks, node, sel_s):
      return sel_s, False

    # 2. Uniform arms: every arm yields the exact same stream or constant.
    unwrapped = [unwrap_arm(i) for i in range(len(arm_streams))]
    first = unwrapped[0]
    if (
        first is not None
        and all(
            s is not None and stream_lib.same_stream_or_const(s, first)
            for s in unwrapped[1:]
        )
        and _clock_compatible(clocks, node, first)
    ):
      return first, False

  # 3. Normalize exhaustive tag-mapped boolean merge/switch to is_boolean=True.
  if not lookup.is_boolean and graph.is_bool_stream(sel_s):
    t_idx = lookup.arms.get((True, True), lookup.default_index)
    f_idx = lookup.arms.get((True, False), lookup.default_index)
    if (
        t_idx is not None
        and f_idx is not None
        and (
            t_idx != f_idx
            or node.op == 'switch'
            or (
                graph.is_polymorphic_static(arm_streams[t_idx])
                and arm_streams[t_idx].is_one_periodic
            )
        )
        and not (
            eof_ids
            and node.op == 'switch'
            and any(
                s.id in eof_ids
                for idx, s in enumerate(arm_streams)
                if idx not in {t_idx, f_idx}
            )
        )
    ):
      node.kwargs['lookup'] = MergeSelector(is_boolean=True)
      graph.rewire_node(
          node,
          inputs=(node.inputs[0], arm_streams[t_idx], arm_streams[f_idx]),
          resets=node.resets,
      )
      replacement, _ = _simplify_mux(graph, clocks, node, eof_ids)
      return replacement, True
  return None, False


_CseBucketKey = tuple[str, tuple[str, ...], int | None]


def _cse_bucket(node: Node) -> _CseBucketKey | None:
  """Returns a lightweight identity bucket for pure CSE, or `None`."""
  if node.is_stateful or not (
      node.is_primitive('when', 'whennot', 'merge', 'switch')
      or common.is_pure_step_func(node.step_func)
  ):
    return None
  if (
      node.is_primitive('when', 'whennot', 'merge', 'switch')
      and len(node.inputs) < 2
  ):
    return None
  op_group = (
      'when' if node.is_primitive('when', 'whennot') else node.op
  )
  pfx_key = tuple(p for p, _ in node.prefixes)
  if node.is_primitive('merge', 'switch'):
    sel_s = node.inputs[0]
    key_stream_id = None if sel_s.is_one_periodic else sel_s.id
  else:
    key_stream_id = next(
        (h.id for h in node.inputs if not h.is_one_periodic),
        None,
    )
  return (op_group, pfx_key, key_stream_id)


def _same_pure_node(graph: Graph, a: Node, b: Node) -> bool:
  """Returns True when two pure nodes in the same bucket are equivalent."""
  if a.is_primitive('when', 'whennot') and b.is_primitive('when', 'whennot'):
    return (
        stream_lib.same_stream_or_const(a.inputs[0], b.inputs[0])
        and stream_lib.same_stream_or_const(a.inputs[1], b.inputs[1])
        and graph.normalize_when_selector(
            a.inputs[0], a.kwargs.get('selector'), op=a.op
        )
        == graph.normalize_when_selector(
            b.inputs[0], b.kwargs.get('selector'), op=b.op
        )
    )
  if a.op != b.op or a.step_func is not b.step_func:
    return False
  if a.is_primitive('merge', 'switch'):
    la, lb = a.kwargs.get('lookup'), b.kwargs.get('lookup')
    if (
        not isinstance(la, MergeSelector)
        or not isinstance(lb, MergeSelector)
        or la.is_boolean != lb.is_boolean
        or not stream_lib.same_stream_or_const(a.inputs[0], b.inputs[0])
    ):
      return False
    if la.is_boolean:
      return (
          len(a.inputs) == 3
          and len(b.inputs) == 3
          and all(
              stream_lib.same_stream_or_const(ha, hb)
              for ha, hb in zip(a.inputs[1:], b.inputs[1:])
          )
      )
    if la.arms.keys() != lb.arms.keys() or (
        (la.default_index is None) != (lb.default_index is None)
    ):
      return False
    if la.default_index is not None and not stream_lib.same_stream_or_const(
        a.inputs[1 + la.default_index], b.inputs[1 + lb.default_index]
    ):
      return False
    return all(
        stream_lib.same_stream_or_const(
            a.inputs[1 + idx], b.inputs[1 + lb.arms[tag_k]]
        )
        for tag_k, idx in la.arms.items()
    )
  return (
      len(a.inputs) == len(b.inputs)
      and _same_kwargs(a.kwargs, b.kwargs)
      and all(
          stream_lib.same_stream_or_const(ha, hb)
          for ha, hb in zip(a.inputs, b.inputs)
      )
  )


def _simplify_once(
    graph: Graph,
    clocks: clock.ClockAnalysis,
    protected_nodes: frozenset[Node],
) -> bool:
  """Runs one pass of algebraic simplification and combinational CSE."""
  changed = False
  cse_buckets: dict[_CseBucketKey, list[Node]] = {}
  eof_ids = _may_truncate_eof(graph)
  order = analysis.causal_node_order(graph)
  for node in order:
    if node in protected_nodes:
      bucket_key = _cse_bucket(node)
      if bucket_key is not None:
        cse_buckets.setdefault(bucket_key, []).append(node)
      continue

    replacement: Stream | None = None
    if not node.is_stateful:
      # Guard every fold with builtin identity (`is_primitive` / `is_step_of`)
      # so user `@psyre_comb(pure=True)` ops colliding with builtin names are
      # never mis-folded.
      if node.is_primitive('ifthenelse'):
        replacement = _simplify_ifthenelse(graph, node)
      elif node.is_primitive('when', 'whennot'):
        replacement, rewired = _simplify_when(graph, node)
        if rewired:
          changed = True
      elif node.is_primitive('merge', 'switch'):
        replacement, rewired = _simplify_mux(graph, clocks, node, eof_ids)
        if rewired:
          changed = True
      elif node.op in _SIMPLIFY_BOOL_OPS and node.is_stream_op():
        replacement = _simplify_bool_op(graph, clocks, node)

    if (
        replacement is not None
        and replacement is not node.output
        and _clock_compatible(clocks, node, replacement)
        and _preserves_eof(node, replacement, eof_ids)
    ):
      graph.replace_stream(node.output, replacement)
      changed = True
      continue

    bucket_key = _cse_bucket(node)
    if bucket_key is not None:
      bucket = cse_buckets.setdefault(bucket_key, [])
      node_clk = clocks.node_clock(node)
      existing = next(
          (
              cand
              for cand in bucket
              if _same_pure_node(graph, cand, node)
              and clocks.node_clock(cand) == node_clk
              and _clock_compatible(clocks, node, cand.output)
          ),
          None,
      )
      if existing is not None:
        graph.replace_stream(node.output, existing.output)
        changed = True
      else:
        bucket.append(node)
  return changed


def optimize_graph(
    graph: Graph,
    *,
    extra_roots: Iterable[Any] | None = None,
) -> Graph:
  """Optimizes a finalized Graph in place (algebraic folding, CSE, and DCE).

  Iterates sound algebraic simplification (constant/identity `ifthenelse`,
  constant/identity/uniform `merge`/`switch`, always-active `when`/`whennot`,
  boolean identities/double negation, and boolean mux/when selector
  normalization), pure combinational CSE, and dead-code elimination to a fixed
  point. Respects uniform clock polymorphism (`∀ 'a. 'a`) on all unpinned
  static `Stream`s: polymorphic statics replace `Node`s only on `BASE_CLOCK`
  (`.`), un-sampled multi-element `merge` arms are never collapsed to a single
  cursor, CSE merges nodes only with identical execution and output clocks, and
  folds never drop eagerly-pulled `post`-derived streams or unhandled static
  mux tags. Leaves graphs containing instantaneous causality cycles or static
  clock errors untouched so diagnostics remain exact.

  Args:
    graph: Finalized Psyre `Graph` to optimize.
    extra_roots: Optional iterable of additional root streams, nodes, or names
      to keep alive. Explicit `Node` objects in `extra_roots` are not replaced.

  Returns:
    The optimized `graph`.
  """
  graph.require_finalized()
  extra_list = _normalize_extra_roots(extra_roots)
  protected_nodes = frozenset(
      item for item in (extra_list or ()) if isinstance(item, Node)
  )
  eliminate_dead_code(graph, extra_roots=extra_list)
  requested_names = [
      item for item in (extra_list or ()) if isinstance(item, str)
  ]
  named_roots: dict[str, tuple[Stream, tuple[str, ...]]] = {}
  if requested_names:
    for req, stream in graph.resolve_named_streams(requested_names).items():
      named_roots[req] = (stream, graph.stream_names(stream))
  if analysis.find_instantaneous_cycle(graph) is not None:
    return graph
  while True:
    try:
      clocks = clock.check_clocks(graph)
    except clock.ClockError:
      return graph
    if not _simplify_once(graph, clocks, protected_nodes):
      return graph
    # Re-attach explicit `extra_roots` names whose stream matched a default
    # `prod.op` (skipped by `replace_stream`'s `is_explicit_stream_name` filter)
    # so `resolve_named_streams` still resolves them across DCE passes.
    for req_name, (orig_stream, orig_names) in named_roots.items():
      target = graph.resolve_stream(orig_stream)
      if target is not orig_stream:
        for s_name in orig_names:
          if s_name == req_name or (
              '/' not in req_name and s_name.rsplit('/', 1)[-1] == req_name
          ):
            graph.add_stream_name(target, s_name)
    eliminate_dead_code(graph, extra_roots=extra_list)
