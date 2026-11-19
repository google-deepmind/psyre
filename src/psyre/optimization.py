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
"""Whole-program dataflow Graph optimizations and dead-code elimination (DCE)."""

import collections
from collections.abc import Iterable
from typing import Any

from psyre import analysis
from psyre import clock
from psyre import common
from psyre import graph as graph_lib
from psyre import istructure
from psyre import stream as stream_lib

Graph = graph_lib.Graph
Node = graph_lib.Node
IStruct = istructure.IStruct
Stream = stream_lib.Stream
MergeSelector = common.MergeSelector
TagKey = common.TagKey
WhenSelector = common.WhenSelector

_CSE_PURE_OPS = frozenset({
    'when',
    'whennot',
    'merge',
    'switch',
    'ifthenelse',
    '__add__',
    '__radd__',
    '__sub__',
    '__rsub__',
    '__mul__',
    '__rmul__',
    '__matmul__',
    '__rmatmul__',
    '__truediv__',
    '__rtruediv__',
    '__floordiv__',
    '__rfloordiv__',
    '__mod__',
    '__rmod__',
    '__pow__',
    '__rpow__',
    '__lshift__',
    '__rlshift__',
    '__rshift__',
    '__rrshift__',
    '__and__',
    '__rand__',
    '__or__',
    '__ror__',
    '__xor__',
    '__rxor__',
    '__pos__',
    '__neg__',
    '__invert__',
    '__lt__',
    '__le__',
    '__eq__',
    '__ne__',
    '__gt__',
    '__ge__',
    'max',
    'min',
    'sum',
    'exp',
    'tanh',
})


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
) -> list[IStruct]:
  """Collects unique root IStructs from graph.output and optional extra_roots."""
  roots: list[IStruct] = list(graph.output_streams())
  seen: set[IStruct] = set(roots)

  def add_stream(istruct: IStruct) -> None:
    if istruct not in seen:
      seen.add(istruct)
      roots.append(istruct)

  extra_list = _normalize_extra_roots(extra_roots)
  if extra_list is not None:
    named_roots: list[str] = []
    for item in extra_list:
      if isinstance(item, str):
        named_roots.append(item)
      elif isinstance(item, Node):
        graph.validate_node(item)
        add_stream(item.output_stream)
      elif isinstance(item, (IStruct, Stream)):
        add_stream(graph.resolve_stream(item))
      else:
        raise TypeError(
            'extra_roots elements must be IStruct, Stream, Node, or '
            f'str, got {type(item).__name__}.'
        )
    if named_roots:
      for istruct in graph.resolve_named_streams(named_roots).values():
        add_stream(istruct)

  return roots


def find_live_nodes(
    graph: Graph,
    *,
    extra_roots: Iterable[Any] | None = None,
) -> tuple[Node, ...]:
  """Returns all Nodes transitively feeding graph.output (or extra_roots).

  Traverses backward from `graph.output` (and any `extra_roots`) across data
  `inputs` and synchronous/asynchronous `resets`.

  Args:
    graph: Finalized Psyre `Graph` to analyze.
    extra_roots: Optional iterable of additional root `IStruct`s, `Stream`s,
      `Node`s, or stream names to treat as live outputs.

  Returns:
    Tuple of live `Node`s in their original instantiation order.

  Raises:
    RuntimeError: If `graph` has not been finalized.
  """
  graph.require_finalized()
  root_streams = _collect_root_streams(graph, extra_roots=extra_roots)
  visited_streams: set[IStruct] = set(root_streams)
  queue: collections.deque[IStruct] = collections.deque(root_streams)
  live_ids: set[int] = set()

  while queue:
    istruct = queue.popleft()
    prod = graph.producer(istruct)
    if prod is None or prod.id in live_ids:
      continue
    live_ids.add(prod.id)
    for in_stream in prod.input_streams():
      if in_stream not in visited_streams:
        visited_streams.add(in_stream)
        queue.append(in_stream)

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


dce = eliminate_dead_code


def _clock_compatible(
    clocks: clock.ClockAnalysis, node: Node, replacement: IStruct
) -> bool:
  """Returns True when replacement preserves node's output clock."""
  out_clk = clocks.stream_clock(node.output_stream)
  if istructure.is_one_periodic(replacement):
    return out_clk.is_base
  return clocks.stream_clock(replacement) == out_clk


def _normalize_when_predicate(
    graph: Graph,
    sel_s: IStruct,
    op: str,
    selector: WhenSelector | None,
) -> tuple[bool, frozenset[TagKey], bool]:
  """Returns canonical (is_boolean, values, negated) for a when/whennot."""
  return common.normalize_when_selector(
      selector, is_bool=graph.is_bool_stream(sel_s), op=op
  )


def _unwrap_matching_when(
    graph: Graph,
    arm_s: IStruct,
    sel_s: IStruct,
    expected_pred: tuple[bool, frozenset[TagKey], bool],
) -> IStruct | None:
  """Returns payload IStruct if arm_s matches when/whennot(sel_s, payload)."""
  prod = graph.producer(arm_s)
  if (
      prod is None
      or prod.op not in {'when', 'whennot'}
      or len(prod.inputs) < 2
      or prod.inputs[0].istruct is not sel_s
  ):
    return None
  actual_pred = _normalize_when_predicate(
      graph, sel_s, prod.op, prod.kwargs.get('selector')
  )
  return prod.inputs[1].istruct if actual_pred == expected_pred else None


def _is_const_tag(istruct: IStruct, expected: TagKey) -> bool:
  """Returns True when istruct is a 1-periodic constant matching expected."""
  return (
      istructure.is_one_periodic(istruct)
      and common.tag_key(istruct.period[0]) == expected
  )


def _const_bool_value(graph: Graph, istruct: IStruct) -> bool | None:
  """Returns the constant bool emitted by istruct when present, or None."""
  seen: set[IStruct] = set()
  curr = istruct
  while curr not in seen:
    seen.add(curr)
    if istructure.is_one_periodic(curr):
      val = curr.period[0]
      return val if isinstance(val, bool) else None
    prod = graph.producer(curr)
    if (
        prod is not None
        and prod.op in {'when', 'whennot'}
        and len(prod.inputs) >= 2
    ):
      curr = prod.inputs[1].istruct
      continue
    return None
  return None


def _simplify_bool_op(graph: Graph, node: Node) -> IStruct | None:
  """Returns a replacement IStruct for a degenerate boolean &/|/~ op."""
  if node.op == '__invert__':
    if len(node.inputs) != 1:
      return None
    a_s = node.inputs[0].istruct
    if not graph.is_bool_stream(a_s):
      return None
    if istructure.is_one_periodic(a_s):
      return IStruct(not bool(a_s.period[0]))
    prod = graph.producer(a_s)
    if (
        prod is not None
        and prod.op == '__invert__'
        and len(prod.inputs) == 1
        and graph.is_bool_stream(prod.inputs[0].istruct)
    ):
      return prod.inputs[0].istruct
    return None
  if len(node.inputs) != 2:
    return None
  a_s, b_s = node.inputs[0].istruct, node.inputs[1].istruct
  if not (graph.is_bool_stream(a_s) and graph.is_bool_stream(b_s)):
    return None
  a_val = _const_bool_value(graph, a_s)
  b_val = _const_bool_value(graph, b_s)
  is_and = node.op in {'__and__', '__rand__'}
  identity, absorb = (True, False) if is_and else (False, True)
  if a_val is identity:
    return b_s
  if b_val is identity:
    return a_s
  if a_val is absorb:
    return a_s
  if b_val is absorb:
    return b_s
  if common.same_stream_or_const(a_s, b_s):
    return a_s
  return None


def _simplify_ifthenelse(graph: Graph, node: Node) -> IStruct | None:
  """Returns a candidate replacement IStruct for a degenerate ifthenelse."""
  if len(node.inputs) != 3:
    return None
  c_s, t_s, f_s = (h.istruct for h in node.inputs)
  c_bool = _const_bool_value(graph, c_s)
  if c_bool is not None:
    return t_s if c_bool else f_s
  if istructure.is_one_periodic(c_s):
    return t_s if bool(c_s.period[0]) else f_s
  if common.same_stream_or_const(t_s, f_s):
    return t_s
  if (
      graph.is_bool_stream(c_s)
      and _is_const_tag(t_s, (True, True))
      and _is_const_tag(f_s, (True, False))
  ):
    return c_s
  return None


def _simplify_when(graph: Graph, node: Node) -> IStruct | None:
  """Normalizes boolean WhenSelector and folds trivial when/whennot nodes."""
  if len(node.inputs) < 2:
    return None
  sel_s, data_s = node.inputs[0].istruct, node.inputs[1].istruct
  selector = node.kwargs.get('selector')
  norm = _normalize_when_predicate(graph, sel_s, node.op, selector)
  is_bool, vals, _ = norm
  if (
      node.op == 'when'
      and isinstance(selector, WhenSelector)
      and not selector.is_boolean
      and is_bool
  ):
    if vals == frozenset({(True, True)}):
      node.kwargs.pop('selector', None)
      selector = None
    elif vals == frozenset({(True, False)}):
      selector = WhenSelector(is_boolean=True, negated=True)
      node.kwargs['selector'] = selector
  if node.op == 'when' and common.is_tautological_selector(norm):
    return data_s
  if istructure.is_one_periodic(sel_s):
    const_val = sel_s.period[0]
    passes = (
        not bool(const_val)
        if node.op == 'whennot'
        else (bool(const_val) if selector is None else selector(const_val))
    )
    if passes:
      return data_s
  return None


def _simplify_mux(
    graph: Graph, clocks: clock.ClockAnalysis, node: Node
) -> IStruct | None:
  """Folds identity/uniform merge/switch nodes and normalizes boolean muxes."""
  lookup = node.kwargs.get('lookup')
  if not isinstance(lookup, MergeSelector) or len(node.inputs) < 2:
    return None
  sel_s = node.inputs[0].istruct
  arm_streams = [h.istruct for h in node.inputs[1:]]

  def unwrap_arm(
      arm_idx: int, *, require_match: bool = False
  ) -> IStruct | None:
    arm_s = arm_streams[arm_idx]
    if node.op == 'switch' or istructure.is_one_periodic(arm_s):
      return arm_s
    pred = _normalize_when_predicate(
        graph, sel_s, 'when', lookup.arm_selector(arm_idx)
    )
    if common.is_tautological_selector(pred):
      return arm_s
    payload = _unwrap_matching_when(graph, arm_s, sel_s, pred)
    if require_match:
      return payload
    return arm_s if payload is None else payload

  # 1. Identity mux: every arm reproduces sel_s (or its own constant tag).
  tag_pairs: Iterable[tuple[TagKey, int]] = ()
  if (
      lookup.is_boolean
      and len(arm_streams) == 2
      and graph.is_bool_stream(sel_s)
  ):
    tag_pairs = (((True, True), 0), ((True, False), 1))
  elif not lookup.is_boolean and lookup.default_index is None and lookup.arms:
    tag_pairs = lookup.arms.items()
  if (
      tag_pairs
      and all(
          (src := unwrap_arm(arm_idx)) is sel_s
          or (src is not None and _is_const_tag(src, tag_k))
          for tag_k, arm_idx in tag_pairs
      )
      and _clock_compatible(clocks, node, sel_s)
  ):
    return sel_s

  # 2. Uniform arms: every arm yields the exact same stream or constant.
  unwrapped = [
      unwrap_arm(i, require_match=True) for i in range(len(arm_streams))
  ]
  first = unwrapped[0]
  if (
      first is not None
      and all(
          s is not None and common.same_stream_or_const(s, first)
          for s in unwrapped[1:]
      )
      and _clock_compatible(clocks, node, first)
  ):
    return first

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
            or istructure.is_one_periodic(arm_streams[t_idx])
        )
    ):
      node.kwargs['lookup'] = MergeSelector(is_boolean=True)
      graph.rewire_node(
          node,
          inputs=(
              node.inputs[0],
              Stream(arm_streams[t_idx], name=node.inputs[1 + t_idx].name),
              Stream(arm_streams[f_idx], name=node.inputs[1 + f_idx].name),
          ),
          resets=node.resets,
      )
      return _simplify_mux(graph, clocks, node)
  return None


_CseBucketKey = tuple[str, tuple[str, ...], IStruct | None]


def _cse_bucket(node: Node) -> _CseBucketKey | None:
  """Returns a lightweight identity bucket for pure CSE, or None."""
  if node.is_stateful or node.op not in _CSE_PURE_OPS:
    return None
  if node.op in {'when', 'whennot', 'merge', 'switch'} and len(node.inputs) < 2:
    return None
  op_group = 'when' if node.op in {'when', 'whennot'} else node.op
  pfx_key = tuple(p for p, _ in node.prefixes)
  if node.op in {'merge', 'switch'}:
    sel_s = node.inputs[0].istruct
    key_stream = None if istructure.is_one_periodic(sel_s) else sel_s
  else:
    key_stream = next(
        (h.istruct for h in node.inputs if not istructure.is_one_periodic(h)),
        None,
    )
  return (op_group, pfx_key, key_stream)


def _same_pure_node(graph: Graph, a: Node, b: Node) -> bool:
  """Returns True when two pure nodes in the same bucket are equivalent."""
  if a.op in {'when', 'whennot'} and b.op in {'when', 'whennot'}:
    return (
        common.same_stream_or_const(a.inputs[0], b.inputs[0])
        and common.same_stream_or_const(a.inputs[1], b.inputs[1])
        and _normalize_when_predicate(
            graph, a.inputs[0].istruct, a.op, a.kwargs.get('selector')
        )
        == _normalize_when_predicate(
            graph, b.inputs[0].istruct, b.op, b.kwargs.get('selector')
        )
    )
  if a.op != b.op:
    return False
  if a.op in {'merge', 'switch'}:
    la, lb = a.kwargs.get('lookup'), b.kwargs.get('lookup')
    if (
        not isinstance(la, MergeSelector)
        or not isinstance(lb, MergeSelector)
        or la.is_boolean != lb.is_boolean
        or not common.same_stream_or_const(a.inputs[0], b.inputs[0])
    ):
      return False
    if la.is_boolean:
      return (
          len(a.inputs) == 3
          and len(b.inputs) == 3
          and all(
              common.same_stream_or_const(ha, hb)
              for ha, hb in zip(a.inputs[1:], b.inputs[1:])
          )
      )
    if la.arms.keys() != lb.arms.keys() or (
        (la.default_index is None) != (lb.default_index is None)
    ):
      return False
    if la.default_index is not None and not common.same_stream_or_const(
        a.inputs[1 + la.default_index], b.inputs[1 + lb.default_index]
    ):
      return False
    return all(
        common.same_stream_or_const(
            a.inputs[1 + idx], b.inputs[1 + lb.arms[tag_k]]
        )
        for tag_k, idx in la.arms.items()
    )
  return (
      len(a.inputs) == len(b.inputs)
      and not a.kwargs
      and not b.kwargs
      and all(
          common.same_stream_or_const(ha, hb)
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
  order = analysis.causal_node_order(graph)
  for node in order:
    if node in protected_nodes:
      bucket_key = _cse_bucket(node)
      if bucket_key is not None:
        cse_buckets.setdefault(bucket_key, []).append(node)
      continue

    replacement: IStruct | None = None
    if node.op == 'ifthenelse':
      replacement = _simplify_ifthenelse(graph, node)
    elif node.op in {'when', 'whennot'}:
      replacement = _simplify_when(graph, node)
    elif node.op in {'merge', 'switch'}:
      replacement = _simplify_mux(graph, clocks, node)
    elif node.op in {'__and__', '__rand__', '__or__', '__ror__', '__invert__'}:
      replacement = _simplify_bool_op(graph, node)

    if (
        replacement is not None
        and replacement is not node.output_stream
        and _clock_compatible(clocks, node, replacement)
    ):
      graph.replace_stream(node.output_stream, replacement)
      changed = True
      continue

    bucket_key = _cse_bucket(node)
    if bucket_key is not None:
      bucket = cse_buckets.setdefault(bucket_key, [])
      existing = next(
          (cand for cand in bucket if _same_pure_node(graph, cand, node)),
          None,
      )
      if existing is not None and existing is not node:
        graph.replace_stream(node.output_stream, existing.output_stream)
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

  Iterates algebraic simplification (constant/identity `ifthenelse`,
  identity/uniform `merge`/`switch`, boolean mux/when normalization), pure
  combinational CSE, and dead-code elimination to a fixed point. Leaves graphs
  containing instantaneous causality cycles or static clock errors untouched so
  diagnostics remain exact.

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
  if analysis.find_instantaneous_cycle(graph) is not None:
    return graph
  while True:
    try:
      clocks = clock.infer_clocks(graph)
    except clock.ClockError:
      return graph
    if not _simplify_once(graph, clocks, protected_nodes):
      return graph
    eliminate_dead_code(graph, extra_roots=extra_list)
