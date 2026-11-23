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
"""Stage-1 Psyre dataflow Graph builder, independent of Stage-2 Scheduler.

Implements two-stage elaboration (`GraphBuilder` / `build_graph`) and
reset/clock alignment lowering (`lower_resets`), instantiating clock-polymorphic
static streams (`Colaço & Pouzet`, EMSOFT 2003), latching ancestor modular
resets across subclocks (`Hamon & Pouzet`, PPDP 2000; mechanized compilation in
`Bourke, Brun & Pouzet`, POPL 2020), and lifting subclocked `current()` inputs
under Kahn Process Network semantics (`Halbwachs et al.`, *Proc. IEEE* 1991;
`Caspi & Pouzet`, ICFP 1996; `Potop-Butucaru, Cohen, Plotkin & Pompougnac`,
2023, arXiv:2311.16977).
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
import contextlib
import dataclasses
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre import kahn_primitives as _kahn_primitives  # pylint: disable=unused-import
from psyre import sync_primitives as _sync_primitives  # pylint: disable=unused-import
from psyre.clock import (
    BASE_CLOCK,
    Clock,
    ClockAnalysis,
    ClockStep,
)
from psyre.common import (
    ArmScope,
    CapturedEnv,
    MergeSelector,
    PrepareFn,
    Semantics,
    WhenSelector,
    active_builder,
    active_scheduler,
    arm_stack,
    callable_name,
    emit_when,
    fby,
    ifthenelse,
    is_stateful_step_func,
    merge,
    prefix_stack,
    push_reset,
    reset_stack,
    switch,
    when_selector_matches_const,
)
from psyre.common_payload import is_array_object
from psyre.graph import Graph, Node
from psyre.optimization import optimize_graph
from psyre.static_stream import static_stream
from psyre.stream import Stream, resolve_stream

# pylint: enable=g-importing-member,g-multiple-import


def _dedup_mux_arms(
    node: Node, inputs: tuple[Stream, ...]
) -> tuple[Stream, ...]:
  """Deduplicates shared resolved arm `Stream`s on non-boolean `merge`/`switch`."""
  # Dict-mapped `merge(s, {0: x, 1: x})` shares one input port for tags routed
  # to the same resolved `Stream` (`s in {0, 1}`). Skip when `merge_arms` is
  # set so `match()`/`automaton()` keep distinct per-arm `StreamPort` cursors
  # on polymorphic static `Stream`s.
  if (
      node.merge_arms is not None
      or len(inputs) < 2
      or not node.is_primitive('merge', 'switch')
  ):
    return inputs
  lookup = node.kwargs.get('lookup')
  if not isinstance(lookup, MergeSelector) or lookup.is_boolean:
    return inputs
  unique_arms: dict[int, tuple[int, Stream]] = {}
  old_to_new: list[int] = []
  for arm in inputs[1:]:
    entry = unique_arms.get(arm.id)
    if entry is None:
      entry = (len(unique_arms), arm)
      unique_arms[arm.id] = entry
    old_to_new.append(entry[0])
  if len(unique_arms) == len(inputs) - 1:
    return inputs
  node.kwargs['lookup'] = MergeSelector(
      is_boolean=False,
      arms={k: old_to_new[idx] for k, idx in lookup.arms.items()},
      default_index=(
          None
          if lookup.default_index is None
          else old_to_new[lookup.default_index]
      ),
  )
  return (inputs[0], *(arm for _, arm in unique_arms.values()))


class GraphBuilder:
  """Stage-1 builder constructing and finalizing a Psyre dataflow Graph."""

  on_finalize: Callable[['GraphBuilder'], None] | None = None

  def __init__(
      self,
      semantics: Semantics | None = None,
      *,
      graph: Graph | None = None,
  ):
    """Initializes a fresh Stage-1 GraphBuilder."""
    self._graph = Graph(semantics=semantics) if graph is None else graph
    self._pending: dict[int, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
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

  def add_stream(self, stream: Stream) -> Stream:
    """Registers a Stream in the underlying Graph and returns it."""
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
    is_stateful = is_stateful_step_func(step_func)
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
        step_func=step_func,
        kwargs=kwargs,
        is_stateful=is_stateful,
        arms=arms,
        prefixes=prefixes,
    )
    self._pending[node.id] = (args, reset_stack.get())
    return o_stream

  def add_static_stream_node(
      self, output: Stream, source: Stream, *, name: str = ''
  ) -> Stream:
    """Registers a resettable `static_stream` node populating `output`."""
    raw_name = name or 'static_stream'
    output.name = raw_name
    return self.add_loop_node(
        static_stream, o_stream=output, source=source, name=raw_name
    )

  def _node_args(self, node: Node) -> Sequence[Any]:
    if node.is_resolved:
      return node.inputs
    pending = self._pending.get(node.id)
    return pending[0] if pending is not None else ()

  def is_bool_stream(self, stream: Stream) -> bool:
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
    arms = node.arms if reset_index is None else node.arms_at_reset(reset_index)
    pfxs = (
        node.prefixes
        if reset_index is None
        else node.prefixes_at_reset(reset_index)
    )
    with CapturedEnv(arms=arms, prefixes=pfxs, resets=resets).restore():
      yield

  def bind_output(self, output: Any) -> None:
    """Validates, normalizes, and registers streams in the Stage-1 output."""
    self._graph.set_output(output)

  def resolve_pending_nodes(
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
          raw_inputs, raw_resets = self._pending.pop(node.id, ((), ()))
          with self._enter_thunk(node, raw_resets):
            inputs = _dedup_mux_arms(
                node, tuple(resolve_fn(s) for s in raw_inputs)
            )
          if node.is_stateful and raw_resets:
            resolved_resets: list[Stream] = []
            for r_idx, raw_r in enumerate(raw_resets):
              with self._enter_thunk(node, raw_resets, r_idx):
                resolved_resets.append(resolve_fn(raw_r))
            resets = tuple(resolved_resets)
          else:
            resets = ()
          self._graph.rewire_node(node, inputs=inputs, resets=resets)

  def finalize(
      self, resolve_fn: Callable[[Any], Stream] = resolve_stream
  ) -> Graph:
    """Resolves deferred thunks, runs finalize hooks, and locks the Graph."""
    idx = self.resolve_pending_nodes(0, resolve_fn)
    if GraphBuilder.on_finalize is not None:
      GraphBuilder.on_finalize(self)
      self.resolve_pending_nodes(idx, resolve_fn)
    self._graph.mark_finalized()
    return self._graph


def _needs_static_input_lowering(
    node: Node, idx: int, src: Stream, semantics: Semantics
) -> bool:
  """Returns True when polymorphic static port `(node, idx)` needs lowering."""
  return (
      (idx == 0 and node.is_primitive('merge', 'when', 'whennot', 'post'))
      or (idx in (1, 2) and node.is_primitive('buffer'))
      or (semantics == Semantics.SYNC and not src.is_one_periodic)
  )


def requires_clock_analysis(graph: Graph) -> bool:
  """Conservative check for subclocked KAHN graphs needing `lower_resets()`.

  Used by `Scheduler` under `Semantics.KAHN` when `clocks=None`
  (`clock_calculus=False`) to reject graphs whose execution depends on
  `lower_resets()` — subclocked stateful resets (which may need ancestor
  latching or a `static_stream` activation carrier), subclocked `current()`
  data inputs, and subclocked polymorphic static inputs that require
  `static_stream` instantiation. Graphs without those dependencies can run
  under `Semantics.KAHN` without static clock analysis (e.g. co-active
  subclocks such as `. on c1 on c2` vs `. on c2 on c1`, or deliberately
  clock-mismatched streams illustrating token drift). Under `Semantics.SYNC`,
  static clock analysis (`check_clocks`) and `lower_resets()` are mandatory.

  Args:
    graph: Finalized Psyre `Graph`.

  Returns:
    True when executing `graph` under `Semantics.KAHN` without `lower_resets()`
    may skip required reset latching, `current()` lifting, or `static_stream`
    instantiation.
  """
  if graph.is_resets_lowered_for(Semantics.KAHN):
    return False
  has_subclocks = any(
      n.is_primitive('when', 'whennot', 'buffer') or n.arms
      for n in graph.nodes
  )
  if not has_subclocks:
    return False

  subclocked_streams: set[int] = set()
  subclocked_nodes: set[int] = set()
  changed = True
  while changed:
    changed = False
    for node in graph.nodes:
      if node.id not in subclocked_nodes:
        in_sub = bool(node.arms) or any(
            s.id in subclocked_streams
            for s in (*node.rate_inputs(), *node.resets)
        )
        if in_sub:
          subclocked_nodes.add(node.id)
          subclocked_streams.add(node.output.id)
          changed = True
      if node.output.id not in subclocked_streams:
        if node.is_primitive('when', 'whennot') and len(node.inputs) >= 2:
          if not graph.is_always_active_when(
              node.inputs[0], node.kwargs.get('selector'), op=node.op
          ):
            subclocked_streams.add(node.output.id)
            changed = True
        elif node.is_primitive('buffer') and len(node.inputs) >= 3:
          if not graph.const_bool(node.inputs[2]):
            subclocked_streams.add(node.output.id)
            changed = True

  if any(
      n.is_primitive('current')
      and len(n.inputs) >= 2
      and n.inputs[1].id in subclocked_streams
      for n in graph.nodes
  ):
    return True

  for node in graph.nodes:
    if node.id not in subclocked_nodes:
      continue
    if any(
        graph.is_polymorphic_static(src)
        and _needs_static_input_lowering(node, idx, src, Semantics.KAHN)
        for idx, src in enumerate(node.inputs)
    ):
      return True
    if (
        not node.is_stateful
        and node.inputs
        and all(graph.is_polymorphic_static(src) for src in node.inputs)
    ):
      return True
    if (
        node.is_stateful
        and node.resets
        and (
            any(not r.is_one_periodic for r in node.resets)
            or all(graph.is_polymorphic_static(r) for r in node.resets)
        )
    ):
      return True
  return False


def lower_resets(graph: Graph, clocks: ClockAnalysis) -> Graph:
  """Lowers ancestor resets and subclocked static ports onto clocks.

  Every unpinned static `Stream` is clock-polymorphic (`∀ 'a. 'a`) and
  instantiated independently per `(stream, target_clock)` consumer port:

  * Any reset stream whose scope clock (`clocks.reset_clock(node, r_idx)`) is
    a strict prefix of `clocks.node_clock(node)` is lowered across the
    intervening `ClockStep`s: constant `False`/`True` resets become activation
    `when` chains, and non-constant resets are latched via `fby`/`ifthenelse`/
    `switch`/`when`/`whennot`.
  * Under `Semantics.SYNC`, every subclocked polymorphic static `Stream` input
    or reset on `target_clk != BASE_CLOCK` is lowered to a gated
    `static_stream` emitter producing its dense `prefix`/`period` values on
    active ticks and `Absent` on inactive ticks (avoiding wrappers for
    1-periodic constants unless a `merge`/`when`/`whennot`/`post` port 0, a
    `buffer` predicate port 1 or 2, a stateless node with only static inputs, or
    a stateful node with only static resets needs an explicit `Absent` clock
    carrier).
  * Under `Semantics.KAHN`, `static_stream` caps port 0 of subclocked
    `merge`/`when`/`whennot`/`post` nodes, predicate ports 1 and 2 of subclocked
    `buffer` nodes, stateless nodes whose inputs are all polymorphic static, and
    stateful nodes whose resets are all polymorphic static, and any `current`
    node whose dynamic data input port 1 lives on a strict subclock of
    `clocks.node_clock(node)` is lowered via nested `merge` + presence-gated
    control back to `node_clock`, ensuring finite-horizon token counts match
    `SYNC` under `Absent`/`Bottom` elision (`present_values`).

  Leaves `graph` unmodified, returning it directly when no lowering is needed.
  Calling `lower_resets(lowered, check_clocks(lowered))` on the result is a
  no-op returning `lowered`.

  Args:
    graph: Finalized Psyre `Graph` to lower.
    clocks: `ClockAnalysis` inferred for `graph`.

  Returns:
    Either `graph` unchanged when all resets and subclocked static ports are
    already aligned, or a newly finalized `Graph` sharing existing dynamic
    `Stream`s with lowered resets and `static_stream` emitters.
  """
  graph.require_finalized()
  if not isinstance(clocks, ClockAnalysis):
    raise TypeError(
        'clocks must be a ClockAnalysis instance, got '
        f'{type(clocks).__name__}.'
    )
  if clocks.graph is not graph:
    raise ValueError('ClockAnalysis belongs to a different Graph.')
  semantics = graph.semantics or Semantics.SYNC
  if graph.is_resets_lowered_for(semantics):
    return graph

  def static_input_ports(node: Node) -> frozenset[int]:
    node_clk = clocks.node_clock(node)
    ports: set[int] = set()
    for idx, src in enumerate(node.inputs):
      if not graph.is_polymorphic_static(src):
        continue
      if clocks.input_clock(node, idx) == BASE_CLOCK:
        continue
      if _needs_static_input_lowering(node, idx, src, semantics):
        ports.add(idx)
    if (
        not ports
        and not node.is_stateful
        and node_clk != BASE_CLOCK
        and node.inputs
        and all(graph.is_polymorphic_static(src) for src in node.inputs)
    ):
      ports.add(0)
    return frozenset(ports)

  def static_reset_ports(node: Node) -> frozenset[int]:
    if not node.is_stateful or not node.resets:
      return frozenset()
    node_clk = clocks.node_clock(node)
    if node_clk == BASE_CLOCK:
      return frozenset()
    ports: set[int] = set()
    if semantics == Semantics.SYNC:
      ports = {
          r_idx
          for r_idx, r_s in enumerate(node.resets)
          if graph.is_polymorphic_static(r_s)
          and not r_s.is_one_periodic
          and clocks.reset_clock(node, r_idx) == node_clk
      }
    if not ports and all(
        graph.is_polymorphic_static(r_s)
        and clocks.reset_clock(node, r_idx) == node_clk
        for r_idx, r_s in enumerate(node.resets)
    ):
      ports.add(0)
    return frozenset(ports)

  node_static_inputs = {
      n.id: p for n in graph.nodes if (p := static_input_ports(n))
  }
  node_static_resets = {
      n.id: p for n in graph.nodes if (p := static_reset_ports(n))
  }
  needs_reset_lowering = any(
      node.is_stateful
      and any(
          clocks.reset_clock(node, r_idx) != clocks.node_clock(node)
          for r_idx in range(len(node.resets))
      )
      for node in graph.nodes
  )
  subclock_currents = {
      n.id
      for n in graph.nodes
      if semantics == Semantics.KAHN
      and n.is_primitive('current')
      and len(n.inputs) >= 2
      and not graph.is_polymorphic_static(n.inputs[1])
      and clocks.stream_clock(n.inputs[1]) != clocks.node_clock(n)
  }
  if not (
      needs_reset_lowering
      or node_static_inputs
      or node_static_resets
      or subclock_currents
  ):
    graph.mark_resets_lowered(semantics)
    return graph

  lowered = graph.clone_unfinalized()
  builder = GraphBuilder(semantics=semantics, graph=lowered)
  start_idx = len(lowered.nodes)
  derived: dict[tuple[bool, int, Clock], Stream] = {}
  base_bool_consts: dict[bool, Stream] = {}
  static_map: dict[tuple[int, Clock], Stream] = {}
  unmapped_synth_scopes: list[tuple[ArmScope, int, Clock]] = []

  def clock_arms(clk: Clock) -> tuple[ArmScope, ...]:
    scopes: list[ArmScope] = []
    curr_clk = BASE_CLOCK
    for step in clk.steps:
      orig_sel = step.selector
      cond = static_map.get((orig_sel.id, curr_clk), orig_sel)
      scope = ArmScope(cond=cond, selector=step.to_when_selector())
      if cond is orig_sel and lowered.is_polymorphic_static(orig_sel):
        unmapped_synth_scopes.append((scope, orig_sel.id, curr_clk))
      scopes.append(scope)
      curr_clk = curr_clk.on(step)
    return tuple(scopes)

  def _step_lookup(ws: WhenSelector) -> MergeSelector:
    """Two-arm lookup mapping active tags to index 0, inactive to index 1."""
    tag_idx = 1 if ws.negated else 0
    return MergeSelector(
        is_boolean=False,
        arms={key: tag_idx for key in ws.values},
        default_index=1 - tag_idx,
    )

  def _mux_step(
      sel: Stream, ws: WhenSelector, on_active: Any, on_inactive: Any
  ) -> Stream:
    """Merges an active subclock arm with an inactive fallback for `ws`."""
    if ws.is_boolean:
      if ws.negated:
        return merge(sel, on_inactive, on_active)
      return merge(sel, on_active, on_inactive)
    return merge(sel, on_active, on_inactive, lookup=_step_lookup(ws))

  def step_predicate(sel: Stream, step: ClockStep) -> Stream:
    selector = step.to_when_selector()
    if sel.is_one_periodic:
      const_match = when_selector_matches_const(selector, sel.const_value)
      if const_match is not None:
        return Stream(const_match)
    if not selector.is_boolean and not selector.values:
      return Stream(selector.negated)
    if selector.is_boolean:
      return ifthenelse(sel, False, True) if selector.negated else sel
    return switch(sel, True, False, lookup=_step_lookup(selector))

  lifted_currents: dict[tuple[int, Clock], tuple[Stream, Stream]] = {}

  def lift_subclock_stream(
      src: Stream, target_clk: Clock, src_clk: Clock
  ) -> tuple[Stream, Stream]:
    """Lifts `src` from `src_clk` to ancestor `target_clk` under KAHN."""
    # Walks intervening `ClockStep`s bottom-up (`src_clk` -> `target_clk`),
    # nesting complementary `merge` nodes returning `(lifted, present)` where
    # `present` is truthy iff every intervening step activates and `lifted`
    # holds `src` (with a `None` placeholder otherwise). Under finite-horizon
    # `post` EOF on an inactive subclock tick, `KAHN` gates `b` by `present`
    # (skipping the dead `lifted` stream and holding), whereas `SYNC` pulls
    # expecting `Absent` and observes `Bottom`.
    key = (src.id, target_clk)
    cached = lifted_currents.get(key)
    if cached is not None:
      return cached
    steps = src_clk.steps[len(target_clk) :]
    # Empty non-negated tagged step is never active: short-circuit cleanly.
    if any(
        not (ws := step.to_when_selector()).is_boolean
        and not ws.values
        and not ws.negated
        for step in steps
    ):
      result = (
          Stream(None),
          Stream(False)
          if target_clk == BASE_CLOCK
          else activation_const(target_clk, False),
      )
      lifted_currents[key] = result
      return result
    prefixes: list[Clock] = [target_clk]
    for step in steps[:-1]:
      prefixes.append(prefixes[-1].on(step))
    lifted = src
    present: Stream | None = None
    for parent_clk, step in reversed(list(zip(prefixes, steps))):
      ws = step.to_when_selector()
      sel = align(step.selector, parent_clk, latch=False)
      with CapturedEnv(arms=clock_arms(parent_clk)).restore():
        lifted = _mux_step(sel, ws, lifted, None)
        present = (
            step_predicate(sel, step)
            if present is None
            else _mux_step(sel, ws, present, False)
        )
    assert present is not None
    result = (lifted, present)
    lifted_currents[key] = result
    return result

  def activation_const(clk: Clock, value: bool = False) -> Stream:
    base = base_bool_consts.get(value)
    if base is None:
      base = Stream(value)
      lowered.pin_base_clock(base)
      base_bool_consts[value] = base
    return align(base, clk, latch=False, src_clk=BASE_CLOCK)

  def get_static(src: Stream, target_clk: Clock) -> Stream:
    if target_clk == BASE_CLOCK:
      return src
    key = (src.id, target_clk)
    cached = static_map.get(key)
    if cached is not None:
      return cached
    act_stream = activation_const(target_clk, False)
    aligned_s = Stream(name=src.name)
    s_node = lowered.add_node(
        op='static_stream',
        name=f'static_stream({src.name})' if src.name else 'static_stream',
        output=aligned_s,
        step_func=static_stream,
        kwargs={'source': src},
        is_stateful=True,
        arms=clock_arms(target_clk),
    )
    lowered.rewire_node(s_node, inputs=(), resets=(act_stream,))
    static_map[key] = aligned_s
    return aligned_s

  def make_latch(
      curr: Stream, sel: Stream, step: ClockStep, act_rst: Stream | None
  ) -> Stream:
    # Compositional 1-step reset latch: holds an ancestor reset firing on an
    # inactive tick until the very next tick where `step` is active:
    #   pending = fby(False, if active then False else held)
    #   held    = curr or pending
    active = step_predicate(sel, step)
    with (
        push_reset(act_rst)
        if act_rst is not None
        else contextlib.nullcontext()
    ):
      pending = fby(
          False,
          lambda: next_pending,  # pylint: disable=undefined-variable
      )
    held = ifthenelse(curr, True, pending)
    next_pending = ifthenelse(active, False, held)
    return emit_when(sel, held, step.to_when_selector())

  def align(
      src: Stream,
      target_clk: Clock,
      *,
      latch: bool,
      src_clk: Clock | None = None,
  ) -> Stream:
    if lowered.is_polymorphic_static(src):
      if not latch:
        if target_clk != BASE_CLOCK:
          return get_static(src, target_clk)
        return src
      start_clk = BASE_CLOCK if src_clk is None else src_clk
      if src.is_one_periodic:
        if is_array_object(src.const_value):
          raise TypeError(
              'Reset stream constant must be a scalar truth value, got '
              f'{type(src.const_value).__name__}.'
          )
        return activation_const(target_clk, bool(src.const_value))
      curr = get_static(src, start_clk) if start_clk != BASE_CLOCK else src
      curr_clk = start_clk
    else:
      curr = src
      curr_clk = clocks.stream_clock(src) if src_clk is None else src_clk
    if curr_clk == target_clk:
      return curr
    assert curr_clk.is_prefix_of(target_clk), (curr_clk, target_clk)
    for step in target_clk.steps[len(curr_clk) :]:
      next_clk = curr_clk.on(step)
      key = (latch, curr.id, next_clk)
      out = derived.get(key)
      if out is None:
        sel = align(step.selector, curr_clk, latch=False)
        act_rst = (
            activation_const(curr_clk, False)
            if latch and curr_clk != BASE_CLOCK
            else None
        )
        with CapturedEnv(arms=clock_arms(curr_clk)).restore():
          if latch:
            out = make_latch(curr, sel, step, act_rst)
          else:
            out = emit_when(sel, curr, step.to_when_selector())
        derived[key] = out
      curr = out
      curr_clk = next_clk
    return curr

  token = active_builder.set(builder)
  try:
    with CapturedEnv().restore():
      orig_cloned = list(lowered.nodes)
      for orig_node, cloned_node in zip(graph.nodes, orig_cloned):
        in_ports = node_static_inputs.get(orig_node.id)
        if in_ports:
          new_inputs = tuple(
              get_static(src, clocks.input_clock(orig_node, idx))
              if idx in in_ports
              else src
              for idx, src in enumerate(cloned_node.inputs)
          )
        else:
          new_inputs = cloned_node.inputs

        inputs_changed = bool(in_ports)
        if orig_node.id in subclock_currents:
          node_clk = clocks.node_clock(orig_node)
          src_i = new_inputs[1]
          i_clk = clocks.stream_clock(orig_node.inputs[1])
          lifted_i, present = lift_subclock_stream(src_i, node_clk, i_clk)
          b_port = new_inputs[0]
          with CapturedEnv(arms=clock_arms(node_clk)).restore():
            b_const = lowered.const_bool(b_port)
            if b_const or b_port is present:
              eff_b = present
            elif b_const is not None:
              eff_b = b_port
            else:
              eff_b = ifthenelse(b_port, present, False)
          new_inputs = (eff_b, lifted_i, *new_inputs[2:])
          inputs_changed = True

        new_resets = cloned_node.resets
        resets_changed = False
        if orig_node.is_stateful and orig_node.resets:
          node_clk = clocks.node_clock(orig_node)
          rst_ports = node_static_resets.get(orig_node.id, frozenset())
          aligned_resets: list[Stream] = []
          for r_idx, r_stream in enumerate(orig_node.resets):
            r_clk = clocks.reset_clock(orig_node, r_idx)
            if r_clk != node_clk:
              aligned = align(r_stream, node_clk, latch=True, src_clk=r_clk)
            elif r_idx in rst_ports:
              aligned = get_static(r_stream, node_clk)
            else:
              aligned = r_stream
            aligned_resets.append(aligned)
            if aligned is not r_stream:
              resets_changed = True
          new_resets = tuple(aligned_resets)

        if inputs_changed or resets_changed:
          lowered.rewire_node(
              cloned_node, inputs=new_inputs, resets=new_resets
          )
      if static_map:
        for orig_node, cloned_node in zip(graph.nodes, orig_cloned):
          if not orig_node.arms:
            continue
          prefix_clks = clocks.arm_prefix_clocks(orig_node)
          updated_arms: list[ArmScope] = []
          arms_changed = False
          for scope, prefix_clk in zip(orig_node.arms, prefix_clks):
            cond_s = graph.resolve_stream(scope.cond)
            mapped = static_map.get((cond_s.id, prefix_clk))
            if mapped is not None:
              updated_arms.append(dataclasses.replace(scope, cond=mapped))
              arms_changed = True
            else:
              updated_arms.append(scope)
          if arms_changed:
            cloned_node.arms = tuple(updated_arms)
        for scope, sel_id, prefix_clk in unmapped_synth_scopes:
          mapped = static_map.get((sel_id, prefix_clk))
          if mapped is not None:
            scope.cond = mapped
      builder.resolve_pending_nodes(start_idx, resolve_stream)
      lowered.mark_finalized()
      lowered.mark_resets_lowered(semantics)
  finally:
    active_builder.reset(token)
  return lowered


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

  Raises:
    RuntimeError: If called re-entrantly inside an active Stage-1 graph build
      or Stage-2 `Scheduler` execution.
  """
  if active_builder.get() is not None:
    raise RuntimeError(
        'Re-entrant build_graph() inside an active Stage-1 graph build is not '
        'allowed.'
    )
  curr_sched = active_scheduler.get()
  if curr_sched is not None and curr_sched.is_active_run():
    raise RuntimeError(
        'build_graph() cannot be called inside an active Stage-2 execution.'
    )
  builder = GraphBuilder(semantics=semantics)
  token = active_builder.set(builder)
  try:
    with CapturedEnv().restore():
      output = main_func(*args, **kwargs)
      builder.bind_output(output)
      graph = builder.finalize()
  finally:
    active_builder.reset(token)
  if optimize:
    optimize_graph(graph, extra_roots=extra_roots)
  return graph
