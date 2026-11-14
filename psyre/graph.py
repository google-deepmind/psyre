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
"""Explicit whole-program Psyre dataflow graph and trace capture."""

from collections.abc import Callable, Iterable, Iterator, Mapping
import contextlib
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    ArmScope,
    CapturedEnv,
    Semantics,
    arm_stack,
    prefix_stack,
)
from psyre.istructure import (
    IStruct,
    IStructHandle,
    as_istruct,
)

# pylint: enable=g-importing-member,g-multiple-import


def _unique_streams(handles: Iterable[IStructHandle]) -> tuple[IStruct, ...]:
  """Returns unique IStructs referenced by handles in first-seen order."""
  seen: set[IStruct] = set()
  ordered: list[IStruct] = []
  for handle in handles:
    istruct = handle.istruct
    if istruct not in seen:
      seen.add(istruct)
      ordered.append(istruct)
  return tuple(ordered)


def _map_output_leaves(
    fn: Callable[[IStruct | IStructHandle], Any], value: Any
) -> Any:
  """Recursively applies fn to stream leaves in a Psyre output PyTree."""
  if value is None:
    return None
  if isinstance(value, (IStruct, IStructHandle)):
    return fn(value)
  if isinstance(value, tuple):
    cls: Any = type(value)
    mapped = (_map_output_leaves(fn, item) for item in value)
    return tuple(mapped) if cls is tuple else cls(*mapped)
  if isinstance(value, list):
    return [_map_output_leaves(fn, item) for item in value]
  if isinstance(value, Mapping):
    return {k: _map_output_leaves(fn, v) for k, v in value.items()}
  raise TypeError(
      'Psyre output must be None, a Stream/IStructHandle, or a '
      f'tuple/list/dict of them, got {type(value).__name__}.'
  )


def _iter_output_leaves(value: Any) -> Iterator[IStruct | IStructHandle]:
  """Yields all stream leaves in an already validated Psyre output PyTree."""
  if value is None:
    return
  if isinstance(value, (IStruct, IStructHandle)):
    yield value
  elif isinstance(value, (tuple, list)):
    for item in value:
      yield from _iter_output_leaves(item)
  elif isinstance(value, Mapping):
    for item in value.values():
      yield from _iter_output_leaves(item)


class Node:
  """Single primitive operator instance in a Psyre dataflow Graph."""

  id: int
  op: str
  name: str
  output: IStructHandle
  step_func: Callable[..., Any]
  kwargs: dict[str, Any]
  inputs: tuple[IStructHandle, ...]
  resets: tuple[IStructHandle, ...]
  is_stateful: bool
  arms: tuple[ArmScope, ...]
  in_arm: bool
  prefixes: tuple[tuple[str, int], ...]
  merge_arms: tuple[tuple[ArmScope, ...], ...] | None

  def __init__(
      self,
      *,
      node_id: int,
      op: str,
      name: str,
      output: IStructHandle,
      step_func: Callable[..., Any],
      raw_inputs: tuple[Any, ...],
      raw_resets: tuple[Any, ...] = (),
      kwargs: dict[str, Any],
      is_stateful: bool = False,
  ):
    """Initializes an unresolved Stage-1 primitive Node."""
    self.id = node_id
    self.op = op
    self.name = name
    self.output = output
    self.step_func = step_func
    self.kwargs = dict(kwargs)
    self.is_stateful = is_stateful
    self.inputs = ()
    self.resets = ()
    self.arms = arm_stack.get()
    self.in_arm = bool(self.arms)
    self.prefixes = prefix_stack.get()
    if self.prefixes and self.name:
      prefix = '/'.join(p for p, _ in self.prefixes)
      if not self.name.startswith(f'{prefix}/'):
        raw_name = self.name
        self.name = f'{prefix}/{raw_name}'
        if self.output.name == raw_name:
          self.output.name = self.name
    self.merge_arms = None
    self._raw_inputs = raw_inputs
    self._raw_resets = raw_resets
    self._resolved = False

  @contextlib.contextmanager
  def enter_thunk(self, reset_index: int | None = None) -> Iterator[None]:
    """Restores this node's (or reset entry's) Stage-1 scope stacks."""
    resets = (
        self._raw_resets
        if reset_index is None
        else self._raw_resets[:reset_index]
    )
    arms = (
        self.arms
        if reset_index is None
        else tuple(a for a in self.arms if a.reset_depth <= reset_index)
    )
    pfxs = (
        self.prefixes
        if reset_index is None
        else tuple(p for p in self.prefixes if p[1] <= reset_index)
    )
    with CapturedEnv(arms=arms, prefixes=pfxs, resets=resets).restore():
      yield

  @property
  def output_stream(self) -> IStruct:
    """Underlying IStruct written by this node."""
    return self.output.istruct

  @property
  def is_resolved(self) -> bool:
    """True once all deferred input and reset thunks are resolved."""
    return self._resolved

  @property
  def raw_inputs(self) -> tuple[Any, ...]:
    """Unresolved Stage-1 positional input specifications."""
    return self._raw_inputs

  @property
  def raw_resets(self) -> tuple[Any, ...]:
    """Unresolved Stage-1 enclosing reset stream specifications."""
    return self._raw_resets

  def resolve(
      self,
      *,
      inputs: tuple[IStructHandle, ...],
      resets: tuple[IStructHandle, ...] = (),
  ) -> None:
    """Binds resolved stream handles for Stage-2 execution and analysis."""
    self.inputs = inputs
    self.resets = resets if self.is_stateful else ()
    self._resolved = True

  def _require_resolved(self) -> None:
    if not self._resolved:
      raise RuntimeError('Node stream handles have not been resolved yet.')

  def input_streams(self) -> tuple[IStruct, ...]:
    """Returns all unique IStructs read by this node across ports and resets."""
    self._require_resolved()
    return _unique_streams((*self.inputs, *self.resets))

  def instantaneous_input_streams(self) -> tuple[IStruct, ...]:
    """Returns unique IStructs needed instantaneously at the current step.

    For `fby(i_stream, j_stream)`, only the initial value stream `i_stream`
    (`inputs[0]`) and enclosing `resets` are needed before the node emits its
    current output; the next-state stream `j_stream` (`inputs[1]`) is buffered
    for the following step and breaks instantaneous cycles. Similarly,
    `pre(i_stream)` emits its held state (`None` on start) before reading
    `i_stream`, so only its enclosing `resets` are instantaneous. All other
    primitives (including `buffer` for now) treat every input and reset stream
    as instantaneous.
    """
    self._require_resolved()
    if self.op == 'pre':
      inst_inputs = ()
    elif self.op == 'fby':
      inst_inputs = self.inputs[:1]
    else:
      inst_inputs = self.inputs
    return _unique_streams((*inst_inputs, *self.resets))

  def delayed_input_streams(self) -> tuple[IStruct, ...]:
    """Returns unique IStructs connected to unit-delayed input ports."""
    self._require_resolved()
    if self.op == 'pre':
      return _unique_streams(self.inputs[:1])
    if self.op == 'fby':
      return _unique_streams(self.inputs[1:])
    return ()

  def __str__(self) -> str:
    return f'{self.name or self.op}#{self.id}'

  def __repr__(self) -> str:
    return f'Node(id={self.id}, op={self.op!r}, name={self.name!r})'


class Graph:
  """Whole-program Psyre graph of primitive nodes and connecting streams."""

  semantics: Semantics | None
  scoped_lasts: list[tuple[IStructHandle, tuple[ArmScope, ...]]]
  on_finalize: Callable[['Graph'], None] | None = None

  def __init__(self, semantics: Semantics | None = None):
    """Initializes an empty Stage-1 Graph."""
    self.semantics = semantics
    self.scoped_lasts = []
    self._output: Any = None
    self._nodes: list[Node] = []
    self._streams: list[IStruct] = []
    self._stream_set: set[IStruct] = set()
    self._handles: list[IStructHandle] = []
    self._handle_ids: set[int] = set()
    self._producer: dict[IStruct, Node] = {}
    self._consumers: dict[IStruct, list[Node]] = {}
    self._next_fsm_id = 0
    self._finalized = False
    self._finalizing = False

  def next_fsm_id(self) -> int:
    """Allocates and returns the next zero-based automaton ID in this Graph."""
    fsm_id = self._next_fsm_id
    self._next_fsm_id += 1
    return fsm_id

  @property
  def output(self) -> Any:
    """Structured Stage-1 output PyTree returned by main_func."""
    return self._output

  @property
  def nodes(self) -> tuple[Node, ...]:
    """All primitive nodes in instantiation order."""
    return tuple(self._nodes)

  @property
  def streams(self) -> tuple[IStruct, ...]:
    """All registered static and dynamic IStructs in registration order."""
    return tuple(self._streams)

  @property
  def handles(self) -> tuple[IStructHandle, ...]:
    """All registered IStructHandles in registration order."""
    return tuple(self._handles)

  @property
  def is_finalized(self) -> bool:
    """True once Stage-1 thunk resolution and edge wiring have completed."""
    return self._finalized

  def require_finalized(self) -> None:
    """Raises RuntimeError unless finalize() has completed on this Graph."""
    if not self._finalized:
      raise RuntimeError('Graph has not been finalized; call finalize() first.')

  def validate_node(self, node: Node) -> None:
    """Verifies that this Graph is finalized and owns node."""
    self.require_finalized()
    if (
        node.id < 0
        or node.id >= len(self._nodes)
        or self._nodes[node.id] is not node
    ):
      raise ValueError(f'Node {node!r} does not belong to this Graph.')

  def bind_output(self, output: Any) -> None:
    """Validates, normalizes, and registers streams in the Stage-1 output."""
    if self._finalized:
      raise RuntimeError('Cannot bind output on an already finalized Graph.')

    def register(leaf: IStruct | IStructHandle) -> IStructHandle:
      self.add_stream(leaf)
      return leaf if isinstance(leaf, IStructHandle) else IStructHandle(leaf)

    self._output = _map_output_leaves(register, output)

  def resolve_named_streams(self, names: Iterable[str]) -> dict[str, IStruct]:
    """Resolves explicit stream names to unique IStructs in this Graph."""
    self.require_finalized()
    if isinstance(names, str):
      raise TypeError(
          'streams must be an iterable of stream names, not a bare str.'
      )
    by_name: dict[str, list[IStruct]] = {}
    by_leaf: dict[str, list[IStruct]] = {}
    for handle in self._handles:
      if handle.name:
        bucket = by_name.setdefault(handle.name, [])
        if handle.istruct not in bucket:
          bucket.append(handle.istruct)
        leaf = handle.name.rsplit('/', 1)[-1]
        leaf_bucket = by_leaf.setdefault(leaf, [])
        if handle.istruct not in leaf_bucket:
          leaf_bucket.append(handle.istruct)
    resolved: dict[str, IStruct] = {}
    for name in names:
      if not isinstance(name, str) or not name:
        raise ValueError(f'Stream name must be a non-empty str, got {name!r}.')
      matches = by_name.get(name, ())
      if not matches and '/' not in name:
        matches = by_leaf.get(name, ())
      if not matches:
        raise KeyError(f'Unknown stream name {name!r} in Graph.')
      if len(matches) > 1:
        raise ValueError(
            f'Ambiguous stream name {name!r} matches {len(matches)} streams '
            'in Graph.'
        )
      resolved[name] = matches[0]
    return resolved

  def prepare_capture(
      self, streams: Iterable[str] | None = None
  ) -> dict[str, IStruct] | None:
    """Resets runtime state and enables capture only on selected streams.

    Args:
      streams: Optional iterable of stream names to capture instead of the
        default `output` PyTree. When `None`, captures leaves of `self.output`.

    Returns:
      A mapping from stream name to `IStruct` when `streams` is provided, or
      `None` when capturing the `output` PyTree.
    """
    named = None if streams is None else self.resolve_named_streams(streams)
    self.reset_runtime_state()
    if named is not None:
      for istruct in named.values():
        istruct.capture = True
    else:
      for leaf in _iter_output_leaves(self._output):
        as_istruct(leaf).capture = True
    return named

  def extract_traces(
      self, horizon: int, named: dict[str, IStruct] | None = None
  ) -> Any:
    """Materializes captured value traces for named streams or graph output.

    Args:
      horizon: Maximum number of steps to materialize per stream trace.
      named: Optional mapping returned by `prepare_capture`. When `None`,
        materializes traces in the shape of `self.output`.

    Returns:
      Either a `dict[str, list[Any]]` when `named` is provided, or a PyTree
      matching `self.output` whose stream leaves are replaced by trace lists.
    """
    self.require_finalized()
    if named is not None:
      return {name: istruct.trace(horizon) for name, istruct in named.items()}
    return _map_output_leaves(
        lambda h: as_istruct(h).trace(horizon), self._output
    )

  def reset_runtime_state(self) -> None:
    """Clears dynamic stream buffers and resets handle cursors for Stage 2."""
    self.require_finalized()
    for istruct in self._streams:
      istruct.capture = False
      if not istruct.static:
        istruct.seq.clear()
        istruct.keep.clear()
    for node in self._nodes:
      node.output.index = 0
      for handle in node.inputs:
        handle.index = 0
      for r_handle in node.resets:
        r_handle.index = 0

  def add_stream(self, stream: IStruct | IStructHandle) -> IStruct:
    """Registers stream in the graph if absent and returns its IStruct."""
    if isinstance(stream, IStructHandle) and id(stream) not in self._handle_ids:
      self._handle_ids.add(id(stream))
      self._handles.append(stream)
    istruct = as_istruct(stream)
    if istruct not in self._stream_set:
      self._stream_set.add(istruct)
      self._streams.append(istruct)
      self._consumers.setdefault(istruct, [])
    return istruct

  def add_node(
      self,
      *,
      op: str,
      name: str,
      output: IStructHandle,
      step_func: Callable[..., Any],
      raw_inputs: tuple[Any, ...],
      raw_resets: tuple[Any, ...] = (),
      kwargs: dict[str, Any],
      is_stateful: bool = False,
  ) -> Node:
    """Appends a new primitive Node and registers its output stream."""
    if self._finalized and not self._finalizing:
      raise RuntimeError('Cannot add nodes to an already finalized Graph.')
    out_istruct = self.add_stream(output)
    if out_istruct in self._producer:
      raise ValueError(
          f'Stream {out_istruct} already has producer'
          f' {self._producer[out_istruct]}.'
      )
    node = Node(
        node_id=len(self._nodes),
        op=op,
        name=name,
        output=output,
        step_func=step_func,
        raw_inputs=raw_inputs,
        raw_resets=raw_resets,
        kwargs=kwargs,
        is_stateful=is_stateful,
    )
    self._nodes.append(node)
    self._producer[out_istruct] = node
    return node

  def rewire_node(
      self,
      node: Node,
      *,
      inputs: tuple[IStructHandle, ...],
      resets: tuple[IStructHandle, ...] = (),
  ) -> None:
    """Updates resolved input and reset handles of node and refreshes consumers."""
    for istruct in node.input_streams():
      cons = self._consumers.get(istruct)
      if cons is not None and node in cons:
        cons.remove(node)
    node.resolve(inputs=inputs, resets=resets)
    for handle in (*node.inputs, *node.resets):
      self.add_stream(handle)
    for istruct in node.input_streams():
      self._consumers[istruct].append(node)

  def finalize(self, resolve_fn: Callable[[Any], IStructHandle]) -> None:
    """Resolves all deferred stream thunks, wiring streams to consumer nodes."""

    def resolve_pending_nodes(start_idx: int) -> int:
      idx = start_idx
      while idx < len(self._nodes):
        node = self._nodes[idx]
        if not node.is_resolved:
          with node.enter_thunk():
            inputs = tuple(resolve_fn(s) for s in node.raw_inputs)
          if node.is_stateful and node.raw_resets:
            resolved_resets: list[IStructHandle] = []
            for r_idx, raw_r in enumerate(node.raw_resets):
              with node.enter_thunk(r_idx):
                resolved_resets.append(resolve_fn(raw_r))
            resets = tuple(resolved_resets)
          else:
            resets = ()
          node.resolve(inputs=inputs, resets=resets)
          for handle in (*inputs, *resets):
            self.add_stream(handle)
          for istruct in node.input_streams():
            self._consumers[istruct].append(node)
        idx += 1
      return idx

    idx = resolve_pending_nodes(0)
    self._finalized = True
    if Graph.on_finalize is not None:
      self._finalizing = True
      try:
        Graph.on_finalize(self)
        resolve_pending_nodes(idx)
      finally:
        self._finalizing = False

  def output_streams(self) -> tuple[IStruct, ...]:
    """Returns unique IStruct leaves in self.output in first-seen order."""
    seen: set[IStruct] = set()
    ordered: list[IStruct] = []
    for leaf in _iter_output_leaves(self._output):
      istruct = as_istruct(leaf)
      if istruct not in seen:
        seen.add(istruct)
        ordered.append(istruct)
    return tuple(ordered)

  def retain_nodes(self, live_nodes: Iterable[Node]) -> int:
    """Retains only live_nodes in instantiation order and renumbers node IDs.

    Args:
      live_nodes: Iterable of `Node`s belonging to this finalized `Graph` to
        keep.

    Returns:
      The number of dead nodes removed from this `Graph`.
    """
    self.require_finalized()
    keep_ids: set[int] = set()
    for node in live_nodes:
      self.validate_node(node)
      keep_ids.add(node.id)
    removed_count = len(self._nodes) - len(keep_ids)
    if removed_count == 0:
      return 0
    retained = [node for node in self._nodes if node.id in keep_ids]
    dead_outputs = {
        node.output_stream for node in self._nodes if node.id not in keep_ids
    }
    for out_s in dead_outputs:
      self._producer.pop(out_s, None)
      self._consumers.pop(out_s, None)
    for istruct, cons_list in list(self._consumers.items()):
      self._consumers[istruct] = [c for c in cons_list if c.id in keep_ids]
    self._streams = [s for s in self._streams if s not in dead_outputs]
    self._stream_set.difference_update(dead_outputs)
    self._handles = [h for h in self._handles if h.istruct not in dead_outputs]
    self._handle_ids = {id(h) for h in self._handles}
    for new_id, node in enumerate(retained):
      node.id = new_id
    self._nodes = retained
    return removed_count

  def producer(self, stream: IStruct | IStructHandle) -> Node | None:
    """Returns the Node producing stream, or None for static/input streams."""
    return self._producer.get(as_istruct(stream))

  def consumers(self, stream: IStruct | IStructHandle) -> tuple[Node, ...]:
    """Returns all Nodes reading stream in instantiation order."""
    self.require_finalized()
    return tuple(self._consumers.get(as_istruct(stream), ()))

  def predecessors(
      self, node: Node, *, instantaneous_only: bool = False
  ) -> tuple[Node, ...]:
    """Returns producer Nodes feeding node (all or instantaneous-only)."""
    self.validate_node(node)
    streams = (
        node.instantaneous_input_streams()
        if instantaneous_only
        else node.input_streams()
    )
    preds: list[Node] = []
    for istruct in streams:
      prod = self._producer.get(istruct)
      if prod is not None:
        preds.append(prod)
    return tuple(preds)

  def successors(
      self, node: Node, *, instantaneous_only: bool = False
  ) -> tuple[Node, ...]:
    """Returns consumer Nodes reading node's output stream."""
    self.validate_node(node)
    out = node.output_stream
    consumers = self._consumers.get(out, ())
    if not instantaneous_only:
      return tuple(consumers)
    return tuple(c for c in consumers if out in c.instantaneous_input_streams())
