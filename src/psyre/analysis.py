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
"""Whole-program instantaneous causality analysis for Psyre dataflow Graphs."""

import collections
from collections.abc import Sequence

# pylint: disable=g-importing-member,g-multiple-import
from psyre.graph import (
    Graph,
    Node,
)

# pylint: enable=g-importing-member,g-multiple-import


class CausalityError(RuntimeError):
  """Raised when a Psyre Graph contains a causality-breaking instantaneous loop."""

  cycle: tuple[Node, ...]

  def __init__(self, cycle: Sequence[Node]):
    """Initializes CausalityError with the elementary instantaneous cycle."""
    if not cycle:
      raise ValueError('CausalityError requires a non-empty node cycle.')
    self.cycle = tuple(cycle)
    chain = ' -> '.join(str(node) for node in (*self.cycle, self.cycle[0]))
    super().__init__(f'Instantaneous causality cycle detected: {chain}')


def _instantaneous_graph(
    graph: Graph,
) -> tuple[tuple[Node, ...], list[list[Node]], list[list[Node]], list[int]]:
  """Builds instantaneous predecessor, successor, and in-degree tables."""
  graph.require_finalized()
  nodes = graph.nodes
  num_nodes = len(nodes)
  preds: list[list[Node]] = [[] for _ in range(num_nodes)]
  succs: list[list[Node]] = [[] for _ in range(num_nodes)]
  in_deg = [0] * num_nodes
  for node in nodes:
    for istruct in node.instantaneous_input_streams():
      prod = graph.producer(istruct)
      if prod is not None:
        preds[node.id].append(prod)
        succs[prod.id].append(node)
    in_deg[node.id] = len(preds[node.id])
  return nodes, preds, succs, in_deg


def _peel_causal_prefix(
    graph: Graph,
) -> tuple[
    tuple[Node, ...], list[Node], list[list[Node]], list[list[Node]], list[int]
]:
  """Runs Kahn's algorithm on instantaneous edges."""
  nodes, preds, succs, in_deg = _instantaneous_graph(graph)
  queue: collections.deque[Node] = collections.deque(
      node for node in nodes if in_deg[node.id] == 0
  )
  order: list[Node] = []
  while queue:
    curr = queue.popleft()
    order.append(curr)
    for nxt in succs[curr.id]:
      in_deg[nxt.id] -= 1
      if in_deg[nxt.id] == 0:
        queue.append(nxt)
  return nodes, order, preds, succs, in_deg


def _extract_shortest_cycle(
    nodes: Sequence[Node],
    preds: Sequence[Sequence[Node]],
    succs: Sequence[Sequence[Node]],
    in_deg: Sequence[int],
) -> tuple[Node, ...]:
  """Extracts a canonical shortest elementary instantaneous cycle."""
  remaining_ids = {node.id for node in nodes if in_deg[node.id] > 0}
  curr_id = min(remaining_ids)
  visited: set[int] = set()
  while curr_id not in visited:
    visited.add(curr_id)
    curr_id = next(u.id for u in preds[curr_id] if u.id in remaining_ids)
  seed_id = curr_id

  parent: dict[int, int] = {}
  bfs_queue: collections.deque[int] = collections.deque([seed_id])
  cycle_ids: list[int] = []
  while bfs_queue and not cycle_ids:
    u_id = bfs_queue.popleft()
    for nxt in succs[u_id]:
      v_id = nxt.id
      if v_id not in remaining_ids:
        continue
      if v_id == seed_id:
        path = [u_id]
        while path[-1] != seed_id:
          path.append(parent[path[-1]])
        path.reverse()
        cycle_ids = path
        break
      if v_id not in parent:
        parent[v_id] = u_id
        bfs_queue.append(v_id)

  min_pos = min(range(len(cycle_ids)), key=cycle_ids.__getitem__)
  rotated = cycle_ids[min_pos:] + cycle_ids[:min_pos]
  return tuple(nodes[node_id] for node_id in rotated)


def find_instantaneous_cycle(graph: Graph) -> tuple[Node, ...] | None:
  """Finds a canonical shortest instantaneous dependency cycle if present.

  Args:
    graph: Finalized Psyre `Graph` to inspect.

  Returns:
    A non-empty tuple of `Node`s forming an elementary instantaneous cycle
    rotated to start at the minimum node ID, or `None` if `graph` is causal.

  Raises:
    RuntimeError: If `graph` has not been finalized.
  """
  nodes, order, preds, succs, in_deg = _peel_causal_prefix(graph)
  if len(order) == len(nodes):
    return None
  return _extract_shortest_cycle(nodes, preds, succs, in_deg)


def check_causality(graph: Graph) -> None:
  """Validates that graph contains no instantaneous dependency cycles.

  Args:
    graph: Finalized Psyre `Graph` to validate.

  Raises:
    RuntimeError: If `graph` has not been finalized.
    CausalityError: If an instantaneous dependency cycle exists in `graph`.
  """
  cycle = find_instantaneous_cycle(graph)
  if cycle is not None:
    raise CausalityError(cycle)


def topological_sort(graph: Graph) -> tuple[Node, ...]:
  """Returns nodes of graph in causal topological order.

  Args:
    graph: Finalized Psyre `Graph` to sort.

  Returns:
    All `Node`s of `graph` ordered so every instantaneous predecessor precedes
    its consumers.

  Raises:
    RuntimeError: If `graph` has not been finalized.
    CausalityError: If an instantaneous dependency cycle exists in `graph`.
  """
  nodes, order, preds, succs, in_deg = _peel_causal_prefix(graph)
  if len(order) < len(nodes):
    raise CausalityError(_extract_shortest_cycle(nodes, preds, succs, in_deg))
  return tuple(order)


def causal_node_order(graph: Graph) -> tuple[Node, ...]:
  """Returns the causal prefix of graph followed by any cyclic nodes by ID."""
  nodes, order, _, _, _ = _peel_causal_prefix(graph)
  if len(order) == len(nodes):
    return tuple(order)
  visited = {node.id for node in order}
  order.extend(node for node in nodes if node.id not in visited)
  return tuple(order)
