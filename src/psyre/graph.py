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
"""Explicit whole-program Psyre dataflow graph IR."""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    ArmScope,
    BOOL_BINARY_OPS,
    BOOL_PRODUCER_OPS,
    Semantics,
)
from psyre.istructure import (
    IStruct,
    as_istruct,
)
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


def _unique_streams(handles: Iterable[Stream]) -> tuple[IStruct, ...]:
  """Returns unique IStructs referenced by handles in first-seen order."""
  return tuple(dict.fromkeys(handle.istruct for handle in handles))


def map_output_leaves(
    fn: Callable[[IStruct | Stream], Any], value: Any
) -> Any:
  """Recursively applies fn to stream leaves in a Psyre output PyTree."""
  if value is None:
    return None
  if isinstance(value, (IStruct, Stream)):
    return fn(value)
  if isinstance(value, tuple):
    cls: Any = type(value)
    mapped = (map_output_leaves(fn, item) for item in value)
    return tuple(mapped) if cls is tuple else cls(*mapped)
  if isinstance(value, list):
    return [map_output_leaves(fn, item) for item in value]
  if isinstance(value, Mapping):
    return {k: map_output_leaves(fn, v) for k, v in value.items()}
  raise TypeError(
      'Psyre output must be None, a Stream, or a '
      f'tuple/list/dict of them, got {type(value).__name__}.'
  )


def iter_output_leaves(value: Any) -> Iterator[IStruct | Stream]:
  """Yields all stream leaves in an already validated Psyre output PyTree."""
  if value is None:
    return
  if isinstance(value, (IStruct, Stream)):
    yield value
  elif isinstance(value, (tuple, list)):
    for item in value:
      yield from iter_output_leaves(item)
  elif isinstance(value, Mapping):
    for item in value.values():
      yield from iter_output_leaves(item)


class Node:
  """Single primitive operator instance in a Psyre dataflow Graph."""

  id: int
  op: str
  name: str
  output: Stream
  step_func: Callable[..., Any]
  kwargs: dict[str, Any]
  inputs: tuple[Stream, ...]
  resets: tuple[Stream, ...]
  is_stateful: bool
  arms: tuple[ArmScope, ...]
  prefixes: tuple[tuple[str, int], ...]
  merge_arms: tuple[tuple[ArmScope, ...], ...] | None

  def __init__(
      self,
      *,
      node_id: int,
      op: str,
      name: str,
      output: Stream,
      step_func: Callable[..., Any],
      kwargs: dict[str, Any],
      inputs: tuple[Stream, ...] = (),
      resets: tuple[Stream, ...] = (),
      is_stateful: bool = False,
      arms: tuple[ArmScope, ...] = (),
      prefixes: tuple[tuple[str, int], ...] = (),
      resolved: bool = False,
  ):
    """Initializes a primitive Node."""
    self.id = node_id
    self.op = op
    self.name = name
    self.output = output
    self.step_func = step_func
    self.kwargs = dict(kwargs)
    self.is_stateful = is_stateful
    self.inputs = inputs if resolved else ()
    self.resets = (resets if is_stateful else ()) if resolved else ()
    self.arms = arms
    self.prefixes = prefixes
    self.merge_arms = None
    self._resolved = resolved

  @property
  def output_stream(self) -> IStruct:
    """Underlying IStruct written by this node."""
    return self.output.istruct

  @property
  def is_resolved(self) -> bool:
    """True once all input and reset stream handles are bound."""
    return self._resolved

  def resolve(
      self,
      *,
      inputs: tuple[Stream, ...],
      resets: tuple[Stream, ...] = (),
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

  def __init__(self, semantics: Semantics | None = None):
    """Initializes an empty Psyre Graph."""
    self.semantics = semantics
    self._output: Any = None
    self._nodes: list[Node] = []
    self._streams: list[IStruct] = []
    self._stream_set: set[IStruct] = set()
    self._handles: list[Stream] = []
    self._handle_ids: set[int] = set()
    self._producer: dict[IStruct, Node] = {}
    self._consumers: dict[IStruct, list[Node]] = {}
    self._stream_aliases: dict[IStruct, IStruct] = {}
    self._finalized = False

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
  def handles(self) -> tuple[Stream, ...]:
    """All registered Stream handles in registration order."""
    return tuple(self._handles)

  @property
  def is_finalized(self) -> bool:
    """True once Stage-1 thunk resolution and edge wiring have completed."""
    return self._finalized

  def require_finalized(self) -> None:
    """Raises RuntimeError unless mark_finalized() has completed."""
    if not self._finalized:
      raise RuntimeError('Graph has not been finalized; call finalize() first.')

  def mark_finalized(self) -> None:
    """Marks this Graph finalized after verifying all nodes are resolved."""
    if any(not node.is_resolved for node in self._nodes):
      raise RuntimeError('Cannot finalize a Graph with unresolved nodes.')
    self._finalized = True

  def validate_node(self, node: Node) -> None:
    """Verifies that this Graph is finalized and owns node."""
    self.require_finalized()
    if (
        node.id < 0
        or node.id >= len(self._nodes)
        or self._nodes[node.id] is not node
    ):
      raise ValueError(f'Node {node!r} does not belong to this Graph.')

  def set_output(self, output: Any) -> None:
    """Validates, normalizes, and registers streams in the Graph output."""
    if self._finalized:
      raise RuntimeError('Cannot set output on an already finalized Graph.')

    def register(leaf: IStruct | Stream) -> Stream:
      handle = leaf if isinstance(leaf, Stream) else Stream(leaf)
      self.add_stream(handle)
      return handle

    self._output = map_output_leaves(register, output)

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
      if handle.name and handle.istruct not in self._stream_aliases:
        for index, key in (
            (by_name, handle.name),
            (by_leaf, handle.name.rsplit('/', 1)[-1]),
        ):
          bucket = index.setdefault(key, [])
          if handle.istruct not in bucket:
            bucket.append(handle.istruct)
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

  def add_stream(self, stream: IStruct | Stream) -> IStruct:
    """Registers stream in the graph if absent and returns its IStruct."""
    if isinstance(stream, Stream) and id(stream) not in self._handle_ids:
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
      output: Stream,
      step_func: Callable[..., Any],
      kwargs: dict[str, Any],
      is_stateful: bool = False,
      arms: tuple[ArmScope, ...] = (),
      prefixes: tuple[tuple[str, int], ...] = (),
  ) -> Node:
    """Appends a new primitive Node and registers its output stream."""
    if self._finalized:
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
        kwargs=kwargs,
        is_stateful=is_stateful,
        arms=arms,
        prefixes=prefixes,
    )
    self._nodes.append(node)
    self._producer[out_istruct] = node
    return node

  def rewire_node(
      self,
      node: Node,
      *,
      inputs: tuple[Stream, ...],
      resets: tuple[Stream, ...] = (),
  ) -> None:
    """Updates resolved input and reset handles of node and refreshes edges."""
    if node.is_resolved:
      for istruct in node.input_streams():
        cons = self._consumers.get(istruct)
        if cons is not None and node in cons:
          cons.remove(node)
    node.resolve(inputs=inputs, resets=resets)
    for handle in (*node.inputs, *node.resets):
      self.add_stream(handle)
    for istruct in node.input_streams():
      self._consumers[istruct].append(node)

  def resolve_stream(self, stream: IStruct | Stream) -> IStruct:
    """Follows replacement aliases and returns the active IStruct."""
    curr = as_istruct(stream)
    seen: set[IStruct] = set()
    while curr in self._stream_aliases and curr not in seen:
      seen.add(curr)
      curr = self._stream_aliases[curr]
    return curr

  def replace_stream(
      self,
      old_stream: IStruct | Stream,
      new_stream: IStruct | Stream,
  ) -> None:
    """Rewires all consumers and outputs from old_stream to new_stream."""
    self.require_finalized()
    old_s = as_istruct(old_stream)
    new_s = self.resolve_stream(new_stream)
    if old_s is new_s:
      return
    self.add_stream(new_s)

    def remap(
        handles: tuple[Stream, ...],
    ) -> tuple[Stream, ...]:
      return tuple(
          Stream(new_s, name=h.name) if h.istruct is old_s else h
          for h in handles
      )

    for consumer in list(self._consumers.get(old_s, ())):
      self.rewire_node(
          consumer,
          inputs=remap(consumer.inputs),
          resets=remap(consumer.resets),
      )
    if self._output is not None:

      def rewrite_leaf(leaf: IStruct | Stream) -> Stream:
        if as_istruct(leaf) is old_s:
          name = leaf.name if isinstance(leaf, Stream) else ''
          handle = Stream(new_s, name=name)
          self.add_stream(handle)
          return handle
        return leaf if isinstance(leaf, Stream) else Stream(leaf)

      self._output = map_output_leaves(rewrite_leaf, self._output)
    old_prod = self._producer.get(old_s)
    existing_names = {
        h.name for h in self._handles if h.istruct is new_s and h.name
    }
    for handle in list(self._handles):
      if (
          handle.istruct is old_s
          and handle.name
          and handle.name not in existing_names
      ):
        leaf = handle.name.rsplit('/', 1)[-1]
        if old_prod is None or leaf != old_prod.op:
          self.add_stream(Stream(new_s, name=handle.name))
          existing_names.add(handle.name)
    self._stream_aliases[old_s] = new_s
    for src, dst in list(self._stream_aliases.items()):
      if dst is old_s:
        self._stream_aliases[src] = new_s

  def is_bool_stream(
      self,
      stream: IStruct | Stream,
      seen: set[IStruct] | None = None,
      *,
      args_fn: Callable[[Node], Sequence[Any]] | None = None,
  ) -> bool:
    """Returns True when stream is statically guaranteed to emit booleans."""
    istruct = self.resolve_stream(stream)
    active_seen = set() if seen is None else seen
    if istruct in active_seen:
      return True
    active_seen.add(istruct)
    prod = self._producer.get(istruct)
    if prod is None:
      return istruct.static and all(
          isinstance(v, bool) for v in (*istruct.prefix, *istruct.period)
      )
    if prod.op in BOOL_PRODUCER_OPS:
      return True

    def arg_is_bool(arg: Any) -> bool:
      if isinstance(arg, bool):
        return True
      if isinstance(arg, (IStruct, Stream)):
        return self.is_bool_stream(arg, active_seen, args_fn=args_fn)
      return False

    args: Sequence[Any] = prod.inputs if args_fn is None else args_fn(prod)
    if prod.op == '__invert__':
      check_args = args[:1]
    elif prod.op in BOOL_BINARY_OPS or prod.op in {'fby', 'morph'}:
      check_args = args[:2] if len(args) >= 2 else ()
    elif prod.op == 'ifthenelse':
      check_args = args[1:3] if len(args) >= 3 else ()
    elif prod.op in {'merge', 'switch'}:
      check_args = args[1:]
    elif prod.op in {'when', 'whennot'}:
      check_args = args[1:2]
    else:
      return False
    return bool(check_args) and all(arg_is_bool(arg) for arg in check_args)

  def output_streams(self) -> tuple[IStruct, ...]:
    """Returns unique IStruct leaves in self.output in first-seen order."""
    leaves = iter_output_leaves(self._output)
    return tuple(dict.fromkeys(as_istruct(leaf) for leaf in leaves))

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

  def producer(self, stream: IStruct | Stream) -> Node | None:
    """Returns the Node producing stream, or None for static/input streams."""
    return self._producer.get(as_istruct(stream))

  def consumers(self, stream: IStruct | Stream) -> tuple[Node, ...]:
    """Returns all Nodes reading stream in instantiation order."""
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
    return tuple(
        prod for s in streams if (prod := self._producer.get(s)) is not None
    )

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
