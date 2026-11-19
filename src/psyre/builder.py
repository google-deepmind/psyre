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
"""Stage-1 Psyre dataflow Graph builder, independent of Stage-2 Scheduler."""

from collections.abc import Callable, Iterable, Iterator, Sequence
import contextlib
import dataclasses
import functools
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre import kahn_primitives as _kahn_primitives  # pylint: disable=unused-import
from psyre import sync_primitives as _sync_primitives  # pylint: disable=unused-import
from psyre.common import (
    ArmScope,
    CapturedEnv,
    PrepareFn,
    STATEFUL_PRIMITIVE_OPS,
    Semantics,
    active_builder,
    arm_stack,
    callable_name,
    is_stateful_step_func,
    prefix_stack,
    reset_stack,
)
from psyre.graph import Graph, Node
from psyre.istructure import IStruct
from psyre.optimization import optimize_graph
from psyre.stream import Stream, resolve_handle

# pylint: enable=g-importing-member,g-multiple-import


@dataclasses.dataclass(frozen=True)
class _PendingNode:
  """Unresolved Stage-1 input and reset thunks for a primitive Node."""

  raw_inputs: tuple[Any, ...]
  raw_resets: tuple[Any, ...]


class GraphBuilder:
  """Stage-1 builder constructing and finalizing a Psyre dataflow Graph."""

  on_finalize: Callable[['GraphBuilder'], None] | None = None

  def __init__(self, semantics: Semantics | None = None):
    """Initializes a fresh Stage-1 GraphBuilder."""
    self._graph = Graph(semantics=semantics)
    self._pending: dict[int, _PendingNode] = {}
    self._next_fsm_id = 0
    self.scoped_lasts: list[tuple[Stream, tuple[ArmScope, ...]]] = []

  @classmethod
  def current(cls) -> 'GraphBuilder | None':
    """Returns the currently active Stage-1 GraphBuilder, or None."""
    return active_builder.get()

  @classmethod
  def is_building(cls) -> bool:
    """Returns True while a Stage-1 Graph build is active."""
    return active_builder.get() is not None

  @property
  def graph(self) -> Graph:
    """The Graph being constructed by this builder."""
    return self._graph

  @property
  def semantics(self) -> Semantics | None:
    """Execution semantics configured on the Graph under construction."""
    return self._graph.semantics

  def next_fsm_id(self) -> int:
    """Allocates and returns the next zero-based automaton ID."""
    fsm_id = self._next_fsm_id
    self._next_fsm_id += 1
    return fsm_id

  def add_stream(self, stream: IStruct | Stream) -> IStruct:
    """Registers stream in the underlying Graph and returns its IStruct."""
    return self._graph.add_stream(stream)

  def add_loop_node(
      self,
      step_func: Callable[..., Any],
      *args: Any,
      _prepare: PrepareFn | None = None,
      **kwargs: Any,
  ) -> Stream:
    """Registers a primitive Node in this build and returns its Stream."""
    if active_builder.get() is not self:
      raise RuntimeError('Psyre nodes can only be instantiated during Stage 1.')
    op = callable_name(step_func)
    name = kwargs.pop('name', op)
    o_stream = kwargs.pop('o_stream', None)
    if _prepare is not None:
      args, kwargs = _prepare(*args, **kwargs)
    if o_stream is None:
      o_stream = Stream(name='' if name == '<lambda>' else name)
    is_stateful = op in STATEFUL_PRIMITIVE_OPS or is_stateful_step_func(
        step_func
    )
    arms = arm_stack.get()
    prefixes = prefix_stack.get()
    if prefixes and name:
      prefix = '/'.join(p for p, _ in prefixes)
      if not name.startswith(f'{prefix}/'):
        raw_name = name
        name = f'{prefix}/{raw_name}'
        if o_stream.name == raw_name:
          o_stream.name = name
    node = self._graph.add_node(
        op=op,
        name=name,
        output=o_stream,
        step_func=functools.partial(step_func, o_stream),
        kwargs=kwargs,
        is_stateful=is_stateful,
        arms=arms,
        prefixes=prefixes,
    )
    self._pending[node.id] = _PendingNode(
        raw_inputs=args,
        raw_resets=reset_stack.get() if is_stateful else (),
    )
    return o_stream

  def _node_args(self, node: Node) -> Sequence[Any]:
    if node.is_resolved:
      return node.inputs
    pending = self._pending.get(node.id)
    return pending.raw_inputs if pending is not None else ()

  def is_bool_stream(self, stream: IStruct | Stream) -> bool:
    """Returns True when stream is statically guaranteed to emit booleans."""
    return self._graph.is_bool_stream(stream, args_fn=self._node_args)

  @contextlib.contextmanager
  def _enter_thunk(
      self,
      node: Node,
      raw_resets: tuple[Any, ...],
      reset_index: int | None = None,
  ) -> Iterator[None]:
    """Restores a node's (or reset entry's) Stage-1 scope stacks."""
    resets = raw_resets if reset_index is None else raw_resets[:reset_index]
    arms = (
        node.arms
        if reset_index is None
        else tuple(a for a in node.arms if a.reset_depth <= reset_index)
    )
    pfxs = (
        node.prefixes
        if reset_index is None
        else tuple(p for p in node.prefixes if p[1] <= reset_index)
    )
    with CapturedEnv(arms=arms, prefixes=pfxs, resets=resets).restore():
      yield

  def bind_output(self, output: Any) -> None:
    """Validates, normalizes, and registers streams in the Stage-1 output."""
    self._graph.set_output(output)

  def _resolve_pending_nodes(
      self, start_idx: int, resolve_fn: Callable[[Any], Stream]
  ) -> int:
    """Resolves all pending nodes from start_idx onward and returns next idx."""
    idx = start_idx
    while True:
      nodes = self._graph.nodes
      if idx >= len(nodes):
        return idx
      for node in nodes[idx:]:
        idx += 1
        if not node.is_resolved:
          pending = self._pending.pop(
              node.id, _PendingNode(raw_inputs=(), raw_resets=())
          )
          with self._enter_thunk(node, pending.raw_resets):
            inputs = tuple(resolve_fn(s) for s in pending.raw_inputs)
          if node.is_stateful and pending.raw_resets:
            resolved_resets: list[Stream] = []
            for r_idx, raw_r in enumerate(pending.raw_resets):
              with self._enter_thunk(node, pending.raw_resets, r_idx):
                resolved_resets.append(resolve_fn(raw_r))
            resets = tuple(resolved_resets)
          else:
            resets = ()
          self._graph.rewire_node(node, inputs=inputs, resets=resets)

  def finalize(
      self, resolve_fn: Callable[[Any], Stream] = resolve_handle
  ) -> Graph:
    """Resolves deferred thunks, runs finalize hooks, and locks the Graph."""
    idx = self._resolve_pending_nodes(0, resolve_fn)
    if GraphBuilder.on_finalize is not None:
      GraphBuilder.on_finalize(self)
      self._resolve_pending_nodes(idx, resolve_fn)
    self._graph.mark_finalized()
    return self._graph


def build_graph(
    main_func: Callable[..., Any],
    *args: Any,
    semantics: Semantics = Semantics.SYNC,
    optimize: bool = True,
    extra_roots: Iterable[Any] | None = None,
    **kwargs: Any,
) -> Graph:
  """Executes Stage 1 to construct, finalize, and optionally optimize a Graph.

  Args:
    main_func: Stage-1 Python callable instantiating Psyre streams and nodes.
    *args: Positional arguments forwarded to `main_func`.
    semantics: Execution semantics (`Semantics.SYNC` or `Semantics.KAHN`).
    optimize: When `True` (default), runs `optimize_graph` after finalization.
    extra_roots: Optional iterable of additional root streams, nodes, or names
      to retain during optimization / dead-code elimination.
    **kwargs: Keyword arguments forwarded to `main_func`.

  Returns:
    The finalized (and optionally optimized) Stage-1 `Graph`.
  """
  builder = GraphBuilder(semantics=semantics)
  token = active_builder.set(builder)
  try:
    with CapturedEnv().restore():
      output = main_func(*args, **kwargs)
      builder.bind_output(output)
      graph = builder.finalize(resolve_handle)
  finally:
    active_builder.reset(token)
  if optimize:
    optimize_graph(graph, extra_roots=extra_roots)
  return graph
