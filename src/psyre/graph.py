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
"""Explicit whole-program Psyre dataflow graph IR.

Represents reactive stream equations as a single-assignment whole-program
graph decoupled from Stage-2 execution, supporting both synchronous
Lustre-style execution (Caspi et al., POPL 1987; Halbwachs et al., *Proc.
IEEE*, 1991) and asynchronous Kahn Process Network execution (Kahn, IFIP 1974;
Kahn & MacQueen, IFIP 1977).
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    ArmScope,
    BOOL_BINARY_OPS,
    BOOL_PRODUCER_OPS,
    Semantics,
    TagKey,
    WhenSelector,
    is_builtin_primitive_step,
    is_builtin_scalar_step_func,
    is_namedtuple,
    is_step_of,
    is_tautological_selector,
    normalize_when_selector,
    when_selector_matches_const,
)
from psyre.common_payload import is_builtin_scalar_value
from psyre.static_stream import is_static_stream_step
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


_SCALAR_STREAM_OPS = (
    frozenset({
        '__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
        '__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__',
        '__mod__', '__rmod__', '__pow__', '__rpow__',
        '__lshift__', '__rlshift__', '__rshift__', '__rrshift__',
        '__pos__', '__neg__', '__invert__', '__abs__',
        'max', 'min', 'sum',
    })
    | BOOL_PRODUCER_OPS
    | BOOL_BINARY_OPS
)


# Slice into `node.inputs` carrying values that flow directly to `node.output`
# (`is_bool_stream`, `_is_builtin_scalar_stream`, and optimization tag-domain
# propagation share this table).
_VALUE_FORWARD_PORTS: dict[str, tuple[slice, int]] = {
    'when': (slice(1, 2), 2),
    'whennot': (slice(1, 2), 2),
    'post': (slice(0, 1), 1),
    'fby': (slice(0, 2), 2),
    'morph': (slice(0, 2), 2),
    'current': (slice(1, 3), 3),
    'ifthenelse': (slice(1, 3), 3),
    'merge': (slice(1, None), 2),
    'switch': (slice(1, None), 2),
}


def _unique_streams(streams: Iterable[Stream]) -> tuple[Stream, ...]:
  """Returns unique Streams in first-seen order."""
  return tuple({s.id: s for s in streams}.values())


def map_output_leaves(
    fn: Callable[[Stream], Any], value: Any
) -> Any:
  """Recursively applies fn to Stream leaves in a Psyre output PyTree."""
  if value is None:
    return None
  if isinstance(value, Stream):
    return fn(value)
  if isinstance(value, tuple):
    mapped = (map_output_leaves(fn, item) for item in value)
    if is_namedtuple(value):
      return type(value)(*mapped)
    return tuple(mapped)
  if isinstance(value, list):
    return [map_output_leaves(fn, item) for item in value]
  if isinstance(value, Mapping):
    return {k: map_output_leaves(fn, v) for k, v in value.items()}
  raise TypeError(
      'Psyre output must be None, a Stream, or a '
      f'tuple/list/dict of them, got {type(value).__name__}.'
  )


def iter_output_leaves(value: Any) -> Iterator[Stream]:
  """Yields all Stream leaves in an already validated Psyre output PyTree."""
  if value is None:
    return
  if isinstance(value, Stream):
    yield value
  elif isinstance(value, (tuple, list)):
    for item in value:
      yield from iter_output_leaves(item)
  elif isinstance(value, Mapping):
    for item in value.values():
      yield from iter_output_leaves(item)


class Node:
  """Single primitive operator instance in a Psyre dataflow Graph.

  Attributes:
    id: Zero-based contiguous index in `Graph.nodes`.
    op: Primitive or operator name (e.g., `'fby'`, `'merge'`, `'__add__'`).
    name: Hierarchical node label prefixed by enclosing `@psyre_node` /
      `automaton()` scopes.
    output: Dynamic `Stream` populated by this node.
    step_func: Underlying Stage-2 coroutine or async-generator factory.
    kwargs: Static keyword metadata bound at Stage 1 (e.g., `lookup`,
      `selector`, `n`, or `_psyre_call_plan`).
    inputs: Resolved positional input `Stream` ports (empty until `resolve()`).
    resets: Enclosing `with reset(r):` `Stream` ports for stateful nodes,
      ordered outermost-to-innermost.
    is_stateful: True for async-generator operators carrying cross-tick state.
    arms: Enclosing `match()`/`automaton()` `ArmScope` stack active at creation;
      each scope's `reset_depth` records `len(reset_stack)` at entry so reset
      port `r_idx` sees scopes with `reset_depth <= r_idx`.
    prefixes: Enclosing hierarchy stack of `(segment_name, reset_depth)` pairs.
    merge_arms: Transient per-arm `ArmScope` stack attached to `merge` nodes by
      `match.merge_arm_outputs` during Stage 1 and cleared to `None` once
      `match._lower_arms` finishes.
  """

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
      is_stateful: bool = False,
      arms: tuple[ArmScope, ...] = (),
      prefixes: tuple[tuple[str, int], ...] = (),
  ):
    """Initializes an unresolved primitive Node."""
    self.id = node_id
    self.op = op
    self.name = name
    self.output = output
    self.step_func = step_func
    self.kwargs = dict(kwargs)
    self.is_stateful = is_stateful
    self.inputs = ()
    self.resets = ()
    self.arms = arms
    self.prefixes = prefixes
    self.merge_arms = None
    self._resolved = False

  @property
  def is_resolved(self) -> bool:
    """True once all input and reset streams are bound."""
    return self._resolved

  def resolve(
      self,
      *,
      inputs: tuple[Stream, ...],
      resets: tuple[Stream, ...] = (),
  ) -> None:
    """Binds resolved streams for Stage-2 execution and analysis."""
    self.inputs = inputs
    self.resets = resets if self.is_stateful else ()
    self._resolved = True

  def _require_resolved(self) -> None:
    if not self._resolved:
      raise RuntimeError('Node stream inputs have not been resolved yet.')

  def input_streams(self) -> tuple[Stream, ...]:
    """Returns all unique Streams read by this node across ports and resets."""
    self._require_resolved()
    return _unique_streams((*self.inputs, *self.resets))

  def rate_inputs(self) -> tuple[Stream, ...]:
    """Returns input streams whose clock determines this node's execution clock."""
    self._require_resolved()
    if self.is_primitive('merge'):
      return self.inputs[:1]
    if self.is_primitive('buffer'):
      return self.inputs[1:3]
    if self.is_primitive('current') and len(self.inputs) >= 3:
      return (self.inputs[0], self.inputs[2])
    return self.inputs

  def is_primitive(self, *ops: str) -> bool:
    """Returns True when this node is a registered builtin primitive in `ops`."""
    return self.op in ops and is_builtin_primitive_step(self.op, self.step_func)

  def is_stream_op(self, *ops: str) -> bool:
    """Returns True for a plain (kwarg-free) `Stream` operator in `ops`."""
    return (
        (not ops or self.op in ops)
        and not self.kwargs
        and is_step_of(Stream, self.op, self.step_func)
    )

  def is_static_stream(self) -> bool:
    """Returns True when this node is a builtin `static_stream` emitter."""
    return self.is_stateful and is_static_stream_step(self.step_func)

  def inherits_reset_clock(self) -> bool:
    """Returns True for stateful nodes whose fallback clock follows resets."""
    # Restricted to `pre`, `fby`, and `static_stream`: allows manual subclock
    # resets (`with reset(when(c, r)): s = Stream.periodic(...)`) and synthetic
    # reset-latch registers (`lower_resets`) to inherit the reset scope clock,
    # while keeping `counter`/`morph`/`prefix_sum`/`scan` rejecting bare
    # subclock resets outside a matching `match`/`automaton` arm.
    return self.is_stateful and (
        self.is_primitive('pre', 'fby') or self.is_static_stream()
    )

  def arms_at_reset(self, reset_index: int) -> tuple[ArmScope, ...]:
    """Returns `self.arms` entries opened at or before `resets[reset_index]`."""
    return tuple(a for a in self.arms if a.reset_depth <= reset_index)

  def prefixes_at_reset(
      self, reset_index: int
  ) -> tuple[tuple[str, int], ...]:
    """Returns `self.prefixes` entries active at `resets[reset_index]`."""
    return tuple(p for p in self.prefixes if p[1] <= reset_index)

  def value_forward_args(
      self, args: Sequence[Any] | None = None
  ) -> Sequence[Any] | None:
    """Returns input args flowing directly to `output`, or `None` if opaque."""
    spec = _VALUE_FORWARD_PORTS.get(self.op)
    if spec is None or not is_builtin_primitive_step(self.op, self.step_func):
      return None
    port_slice, min_len = spec
    seq = self.inputs if args is None else args
    return seq[port_slice] if len(seq) >= min_len else ()

  def instantaneous_input_streams(self) -> tuple[Stream, ...]:
    """Returns unique Streams needed instantaneously at the current step.

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
    if self.is_stateful and self.is_primitive('pre'):
      inst_inputs = ()
    elif self.is_stateful and self.is_primitive('fby'):
      inst_inputs = self.inputs[:1]
    else:
      inst_inputs = self.inputs
    return _unique_streams((*inst_inputs, *self.resets))

  def delayed_input_streams(self) -> tuple[Stream, ...]:
    """Returns unique Streams connected to unit-delayed input ports."""
    self._require_resolved()
    if self.is_stateful and self.is_primitive('pre'):
      return self.inputs[:1]
    if self.is_stateful and self.is_primitive('fby'):
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
    self._streams: dict[int, Stream] = {}
    self._extra_names: dict[int, list[str]] = {}
    self._producer: dict[int, Node] = {}
    self._consumers: dict[int, list[Node]] = {}
    self._stream_aliases: dict[int, Stream] = {}
    self._base_clock_streams: set[int] = set()
    self._lowered_semantics: set[Semantics] = set()
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
  def streams(self) -> tuple[Stream, ...]:
    """All registered static and dynamic Streams in registration order."""
    return tuple(self._streams.values())

  @property
  def is_finalized(self) -> bool:
    """True once Stage-1 thunk resolution and edge wiring have completed."""
    return self._finalized

  def has_stream(self, stream: Stream) -> bool:
    """Returns True if `stream` is registered in this Graph."""
    return stream.id in self._streams

  def pin_base_clock(self, stream: Stream) -> None:
    """Marks a synthetic static stream (e.g., `base_false`) as base-clocked."""
    self._base_clock_streams.add(stream.id)
    self._lowered_semantics.clear()

  def is_base_clock_pinned(self, stream: Stream) -> bool:
    """Returns True if `stream` is pinned to the base clock `.`."""
    return stream.id in self._base_clock_streams

  def mark_resets_lowered(self, semantics: Semantics) -> None:
    """Records that `lower_resets()` already aligned this Graph for semantics."""
    self._lowered_semantics.add(semantics)

  def is_resets_lowered_for(self, semantics: Semantics) -> bool:
    """Returns True if this Graph has already passed `lower_resets()`."""
    return semantics in self._lowered_semantics

  def is_polymorphic_static(self, stream: Stream) -> bool:
    """Returns True for unpinned producer-less static streams (`∀ 'a. 'a`)."""
    return (
        stream.static
        and stream.id not in self._producer
        and stream.id not in self._base_clock_streams
    )

  def add_stream_name(self, stream: Stream, name: str) -> None:
    """Associates an additional lookup alias `name` with `stream`."""
    if not name or name == stream.name:
      return
    bucket = self._extra_names.setdefault(stream.id, [])
    if name not in bucket:
      bucket.append(name)

  def stream_names(self, stream: Stream) -> tuple[str, ...]:
    """Returns all primary and alias names associated with `stream`."""
    names: list[str] = []
    if stream.name:
      names.append(stream.name)
    for alias in self._extra_names.get(stream.id, ()):
      if alias and alias not in names:
        names.append(alias)
    return tuple(names)

  def is_explicit_stream_name(self, stream: Stream, name: str) -> bool:
    """Returns True if `name` is user-assigned rather than a default op name."""
    prod = self._producer.get(stream.id)
    return prod is None or name.rsplit('/', 1)[-1] != prod.op

  def explicit_stream_names(self) -> dict[int, str]:
    """Maps stream IDs to explicit user-assigned names (ignoring op names)."""
    names: dict[int, str] = {}
    for stream in self._streams.values():
      for s_name in self.stream_names(stream):
        if self.is_explicit_stream_name(stream, s_name):
          names[stream.id] = s_name
          break
    return names

  def static_source(self, stream: Stream) -> Stream | None:
    """Returns the backing static `Stream` for static or `static_stream` output."""
    s = self.resolve_stream(stream)
    if s.static:
      return s
    prod = self._producer.get(s.id)
    if prod is not None and prod.is_static_stream():
      src = prod.kwargs.get('source')
      if isinstance(src, Stream) and src.static:
        return src
    return None

  def one_periodic_stream(self, stream: Stream) -> Stream | None:
    """Returns `stream`'s underlying 1-periodic static `Stream`, or `None`."""
    # Does not unwrap `when`/`whennot` gates (unlike `const_value()`), keeping
    # clock and EOF-sensitive checks intact.
    src = self.static_source(stream)
    return src if src is not None and src.is_one_periodic else None

  def const_bool(self, stream: Stream) -> bool | None:
    """Returns `bool(v)` for a direct 1-periodic static `Stream`, or `None`."""
    const_s = self.one_periodic_stream(stream)
    if const_s is None:
      return None
    return when_selector_matches_const(None, const_s.const_value)

  def unwrap_when(
      self,
      stream: Stream,
      *,
      can_unwrap: Callable[[Node], bool] | None = None,
  ) -> Stream:
    """Walks backward through `when`/`whennot` payload chains (cycle-safe)."""
    curr = self.resolve_stream(stream)
    seen: set[int] = set()
    while curr.id not in seen:
      seen.add(curr.id)
      prod = self._producer.get(curr.id)
      if (
          prod is not None
          and prod.is_primitive('when', 'whennot')
          and len(prod.inputs) >= 2
          and (can_unwrap is None or can_unwrap(prod))
      ):
        curr = self.resolve_stream(prod.inputs[1])
        continue
      break
    return curr

  def const_value(self, stream: Stream) -> tuple[bool, Any]:
    """Returns `(True, val)` if `stream` is 1-periodic under `when` gates."""
    # Walks backward through `when`/`whennot` chains via `unwrap_when()`.
    const_s = self.one_periodic_stream(self.unwrap_when(stream))
    if const_s is None:
      return False, None
    return True, const_s.const_value

  def normalize_when_selector(
      self,
      selector_stream: Stream,
      selector: WhenSelector | None = None,
      *,
      op: str = 'when',
  ) -> tuple[bool, frozenset[TagKey], bool]:
    """Returns canonical `(is_boolean, values, negated)` for a when/whennot."""
    return normalize_when_selector(
        selector, is_bool=self.is_bool_stream(selector_stream), op=op
    )

  def is_always_active_when(
      self,
      selector_stream: Stream,
      selector: WhenSelector | None = None,
      *,
      op: str = 'when',
      norm: tuple[bool, frozenset[TagKey], bool] | None = None,
  ) -> bool:
    """Returns True when `when`/`whennot` statically selects every parent tick."""
    if norm is None:
      norm = self.normalize_when_selector(selector_stream, selector, op=op)
    if is_tautological_selector(norm):
      return True
    const_s = self.one_periodic_stream(selector_stream)
    if const_s is None:
      return False
    return bool(
        when_selector_matches_const(selector, const_s.const_value, op=op)
    )

  def require_finalized(self) -> None:
    """Raises RuntimeError unless mark_finalized() has completed."""
    if not self._finalized:
      raise RuntimeError('Graph has not been finalized; call finalize() first.')

  def mark_finalized(self) -> None:
    """Marks this Graph finalized after verifying all nodes are resolved."""
    if any(not node.is_resolved for node in self._nodes):
      raise RuntimeError('Cannot finalize a Graph with unresolved nodes.')
    self._finalized = True

  def clone_unfinalized(self) -> 'Graph':
    """Returns an unfinalized shallow copy sharing existing `Stream`s."""
    self.require_finalized()
    cloned = Graph(semantics=self.semantics)
    for stream in self._streams.values():
      cloned.add_stream(stream)
    for sid, aliases in self._extra_names.items():
      cloned._extra_names[sid] = list(aliases)  # pylint: disable=protected-access
    for node in self._nodes:
      node_copy = cloned.add_node(
          op=node.op,
          name=node.name,
          output=node.output,
          step_func=node.step_func,
          kwargs=dict(node.kwargs),
          is_stateful=node.is_stateful,
          arms=node.arms,
          prefixes=node.prefixes,
      )
      cloned.rewire_node(
          node_copy,
          inputs=node.inputs,
          resets=node.resets,
      )
    cloned._stream_aliases.update(self._stream_aliases)  # pylint: disable=protected-access
    cloned._base_clock_streams.update(self._base_clock_streams)  # pylint: disable=protected-access
    if self._output is not None:
      cloned.set_output(self._output)
    return cloned

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

    self._output = map_output_leaves(self.add_stream, output)

  def resolve_named_streams(self, names: Iterable[str]) -> dict[str, Stream]:
    """Resolves explicit stream names to unique Streams in this Graph."""
    self.require_finalized()
    if isinstance(names, str):
      raise TypeError(
          'streams must be an iterable of stream names, not a bare str.'
      )
    by_name: dict[str, dict[int, Stream]] = {}
    by_leaf: dict[str, dict[int, Stream]] = {}
    for stream in self._streams.values():
      if stream.id in self._stream_aliases:
        continue
      for s_name in self.stream_names(stream):
        for index, key in (
            (by_name, s_name),
            (by_leaf, s_name.rsplit('/', 1)[-1]),
        ):
          bucket = index.setdefault(key, {})
          bucket.setdefault(stream.id, stream)
    resolved: dict[str, Stream] = {}
    for name in names:
      if not isinstance(name, str) or not name:
        raise ValueError(f'Stream name must be a non-empty str, got {name!r}.')
      matches = tuple(by_name.get(name, {}).values())
      if not matches and '/' not in name:
        matches = tuple(by_leaf.get(name, {}).values())
      if not matches:
        raise KeyError(f'Unknown stream name {name!r} in Graph.')
      if len(matches) > 1:
        raise ValueError(
            f'Ambiguous stream name {name!r} matches {len(matches)} streams '
            'in Graph.'
        )
      resolved[name] = matches[0]
    return resolved

  def add_stream(self, stream: Stream) -> Stream:
    """Registers a Stream in the graph if absent and returns it."""
    if not isinstance(stream, Stream):
      raise TypeError(f'Expected Stream, got {type(stream).__name__}.')
    if stream.id not in self._streams:
      self._streams[stream.id] = stream
      self._consumers.setdefault(stream.id, [])
    return stream

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
    self.add_stream(output)
    if output.id in self._producer:
      raise ValueError(
          f'Stream {output} already has producer {self._producer[output.id]}.'
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
    self._producer[output.id] = node
    self._lowered_semantics.clear()
    return node

  def rewire_node(
      self,
      node: Node,
      *,
      inputs: tuple[Stream, ...],
      resets: tuple[Stream, ...] = (),
  ) -> None:
    """Updates resolved input and reset streams of node and refreshes edges."""
    if node.is_resolved:
      for stream in node.input_streams():
        cons = self._consumers.get(stream.id)
        if cons is not None and node in cons:
          cons.remove(node)
    node.resolve(inputs=inputs, resets=resets)
    for stream in (*node.inputs, *node.resets):
      self.add_stream(stream)
    for stream in node.input_streams():
      self._consumers[stream.id].append(node)
    self._lowered_semantics.clear()

  def resolve_stream(self, stream: Stream) -> Stream:
    """Follows replacement aliases with path compression and returns target."""
    if not isinstance(stream, Stream):
      raise TypeError(f'Expected Stream, got {type(stream).__name__}.')
    curr = stream
    aliases = self._stream_aliases
    nxt = aliases.get(curr.id)
    if nxt is None:
      return curr
    if nxt.id not in aliases:
      return nxt
    seen: set[int] = {curr.id}
    path: list[int] = [curr.id]
    curr = nxt
    while (nxt := aliases.get(curr.id)) is not None and curr.id not in seen:
      seen.add(curr.id)
      path.append(curr.id)
      curr = nxt
    if len(path) > 1 and curr.id not in seen:
      for src_id in path[:-1]:
        aliases[src_id] = curr
    return curr

  def resolve_stream_id(self, stream_id: int) -> Stream | None:
    """Resolves a live or replaced stream ID to its active Stream, if any."""
    if isinstance(stream_id, bool) or not isinstance(stream_id, int):
      return None
    curr = self._stream_aliases.get(stream_id)
    if curr is None:
      return self._streams.get(stream_id)
    return self.resolve_stream(curr)

  def replace_stream(
      self,
      old_stream: Stream,
      new_stream: Stream,
  ) -> None:
    """Rewires all consumers and outputs from old_stream to new_stream."""
    self.require_finalized()
    if not isinstance(old_stream, Stream) or not isinstance(new_stream, Stream):
      raise TypeError('replace_stream expects Stream arguments.')
    if old_stream.id in self._stream_aliases:
      prev = self.resolve_stream(old_stream)
      raise ValueError(
          f'Stream {old_stream} has already been replaced by {prev}; '
          'replace the active alias target instead.'
      )
    new_s = self.resolve_stream(new_stream)
    if old_stream is new_s:
      return
    self.add_stream(new_s)

    def remap(streams: tuple[Stream, ...]) -> tuple[Stream, ...]:
      return tuple(new_s if s is old_stream else s for s in streams)

    for consumer in list(self._consumers.get(old_stream.id, ())):
      self.rewire_node(
          consumer,
          inputs=remap(consumer.inputs),
          resets=remap(consumer.resets),
      )
    if self._output is not None:
      self._output = map_output_leaves(
          lambda leaf: new_s if leaf is old_stream else leaf, self._output
      )
    for s_name in self.stream_names(old_stream):
      if self.is_explicit_stream_name(old_stream, s_name):
        self.add_stream_name(new_s, s_name)
    self._stream_aliases[old_stream.id] = new_s
    self._lowered_semantics.clear()

  def _is_builtin_scalar_arg(
      self,
      arg: Any,
      seen: set[int],
      args_fn: Callable[[Node], Sequence[Any]] | None,
  ) -> bool:
    """Returns True if `arg` is a built-in scalar literal or scalar `Stream`."""
    if isinstance(arg, Stream):
      return self._is_builtin_scalar_stream(arg, seen, args_fn=args_fn)
    return is_builtin_scalar_value(arg)

  def _is_builtin_scalar_stream(
      self,
      stream: Stream,
      seen: set[int],
      *,
      args_fn: Callable[[Node], Sequence[Any]] | None = None,
  ) -> bool:
    """Returns True if `stream` only emits built-in Python scalars."""
    s = self.resolve_stream(stream)
    # Coinductive (greatest-fixed-point) cycle guard: sound with a shared
    # non-backtracking `seen` set only because all recursive checks below
    # combine strictly via short-circuiting `all(...)`.
    if s.id in seen:
      return True
    seen.add(s.id)
    src = self.static_source(s)
    if src is not None and src.static_sequence is not None:
      return all(
          is_builtin_scalar_value(v) for v in src.static_sequence.iter_atoms()
      )
    prod = self._producer.get(s.id)
    if prod is None:
      return False
    if is_builtin_scalar_step_func(prod.step_func):
      return True
    args: Sequence[Any] = prod.inputs if args_fn is None else args_fn(prod)
    if prod.is_primitive('counter', 'prefix_sum', 'pre'):
      check_args: Sequence[Any] | None = args
    elif prod.op in _SCALAR_STREAM_OPS and prod.is_stream_op():
      check_args = args
    else:
      check_args = prod.value_forward_args(args)
    if not check_args:
      return False
    return all(
        self._is_builtin_scalar_arg(arg, seen, args_fn)
        for arg in check_args
    )

  def is_bool_stream(
      self,
      stream: Stream,
      seen: set[int] | None = None,
      *,
      args_fn: Callable[[Node], Sequence[Any]] | None = None,
  ) -> bool:
    """Returns True when stream is statically guaranteed to emit Python `bool`s.

    Conservatively disables the unconditional scalar-bool assumption on rich
    comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`): comparisons are
    only treated as boolean streams when both operands are statically proven to
    carry built-in Python scalar types (`bool`, `int`, `float`, `complex`,
    `str`, `bytes`, `bytearray`, `None`, or standard `enum.Enum`). Array
    comparisons (`np.ndarray`, `jax.Array`) and custom objects overloading rich
    comparisons return `False`.

    Args:
      stream: Target `Stream` to classify.
      seen: Optional cycle-detection set of already-visited stream IDs.
      args_fn: Optional callback providing unresolved pending node inputs
        during Stage-1 construction.

    Returns:
      `True` if every emitted non-`Absent` value of `stream` is guaranteed to
      be a Python `bool`.
    """
    s = self.resolve_stream(stream)
    active_seen = set() if seen is None else seen
    # Coinductive (greatest-fixed-point) cycle guard; sound with a shared
    # non-backtracking `active_seen` set only because all recursive checks
    # combine strictly via short-circuiting `all(...)`.
    if s.id in active_seen:
      return True
    active_seen.add(s.id)
    src = self.static_source(s)
    if src is not None and src.static_sequence is not None:
      return src.static_sequence.is_bool()
    prod = self._producer.get(s.id)
    if prod is None:
      return False
    args: Sequence[Any] = prod.inputs if args_fn is None else args_fn(prod)
    if prod.op in BOOL_PRODUCER_OPS and prod.is_stream_op():
      return len(args) == 2 and all(
          self._is_builtin_scalar_arg(arg, set(), args_fn) for arg in args
      )
    if prod.is_stream_op('__invert__'):
      check_args: Sequence[Any] | None = args[:1]
    elif prod.op in BOOL_BINARY_OPS and prod.is_stream_op():
      check_args = args[:2] if len(args) >= 2 else ()
    else:
      check_args = prod.value_forward_args(args)
    if not check_args:
      return False
    return all(
        isinstance(arg, bool)
        or (
            isinstance(arg, Stream)
            and self.is_bool_stream(arg, active_seen, args_fn=args_fn)
        )
        for arg in check_args
    )

  def output_streams(self) -> tuple[Stream, ...]:
    """Returns unique Stream leaves in self.output in first-seen order."""
    return _unique_streams(iter_output_leaves(self._output))

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
    dead_output_ids = {
        node.output.id for node in self._nodes if node.id not in keep_ids
    }
    for out_id in dead_output_ids:
      self._producer.pop(out_id, None)
      self._consumers.pop(out_id, None)
      self._streams.pop(out_id, None)
      self._extra_names.pop(out_id, None)
      self._base_clock_streams.discard(out_id)
    stale = [
        src_id
        for src_id, dst in list(self._stream_aliases.items())
        if self.resolve_stream(dst).id in dead_output_ids
    ]
    for src_id in stale:
      self._stream_aliases.pop(src_id, None)
    for sid, cons_list in list(self._consumers.items()):
      self._consumers[sid] = [c for c in cons_list if c.id in keep_ids]
    for new_id, node in enumerate(retained):
      node.id = new_id
    self._nodes = retained
    self._lowered_semantics.clear()
    return removed_count

  def producer(self, stream: Stream) -> Node | None:
    """Returns the Node producing stream, or None for static/input streams."""
    if not isinstance(stream, Stream):
      raise TypeError(f'Expected Stream, got {type(stream).__name__}.')
    return self._producer.get(stream.id)

  def consumers(self, stream: Stream) -> tuple[Node, ...]:
    """Returns all Nodes reading stream in instantiation order."""
    if not isinstance(stream, Stream):
      raise TypeError(f'Expected Stream, got {type(stream).__name__}.')
    return tuple(self._consumers.get(stream.id, ()))

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
        prod for s in streams if (prod := self._producer.get(s.id)) is not None
    )

  def successors(
      self, node: Node, *, instantaneous_only: bool = False
  ) -> tuple[Node, ...]:
    """Returns consumer Nodes reading node's output stream."""
    self.validate_node(node)
    out_id = node.output.id
    consumers = self._consumers.get(out_id, ())
    if not instantaneous_only:
      return tuple(consumers)
    return tuple(
        c
        for c in consumers
        if any(s.id == out_id for s in c.instantaneous_input_streams())
    )
