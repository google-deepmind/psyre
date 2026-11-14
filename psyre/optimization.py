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

from psyre import graph as graph_lib
from psyre import istructure

Graph = graph_lib.Graph
Node = graph_lib.Node
IStruct = istructure.IStruct
IStructHandle = istructure.IStructHandle


def _collect_root_streams(
    graph: Graph,
    extra_roots: Iterable[Any] | None = None,
) -> list[IStruct]:
  """Collects unique root IStructs from graph.output and optional extra_roots."""
  seen: set[IStruct] = set()
  roots: list[IStruct] = []

  def add_stream(istruct: IStruct) -> None:
    if istruct not in seen:
      seen.add(istruct)
      roots.append(istruct)

  for istruct in graph.output_streams():
    add_stream(istruct)

  if extra_roots is not None:
    if isinstance(extra_roots, str):
      raise TypeError(
          'extra_roots must be an iterable of streams/nodes/names, not a bare '
          'str.'
      )
    named_roots: list[str] = []
    for item in extra_roots:
      if isinstance(item, str):
        named_roots.append(item)
      elif isinstance(item, Node):
        graph.validate_node(item)
        add_stream(item.output_stream)
      elif isinstance(item, (IStruct, IStructHandle)):
        add_stream(graph_lib.as_istruct(item))
      else:
        raise TypeError(
            'extra_roots elements must be IStruct, IStructHandle, Node, or '
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
    extra_roots: Optional iterable of additional root `IStruct`s,
      `IStructHandle`s, `Node`s, or stream names to treat as live outputs.

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
