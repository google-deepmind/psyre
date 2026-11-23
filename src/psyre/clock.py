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
"""Synchronous Lustre-style clock representation, inference, and checking.

Every stream in a finalized Psyre `Graph` is assigned a synchronous clock:

* **Base clock (`.` / `BASE_CLOCK`)**: ticks once on every global activation.
* **Subclock (`<root> on <pred>`)**: ticks strictly on the subset of parent
  ticks where the selector stream satisfies `<pred>` (boolean `on c` /
  `on not c`, or tagged `on (s == v)` / `on (s in {...})` / negated forms).
* **Polymorphic clock (`'a`, `'b`, ...)**: a universally quantified clock
  variable (`∀ 'a. 'a`) attached to **all** unpinned static `Stream`s — both
  1-periodic constants such as `Stream(5)` and finite/periodic sequences such
  as `Stream([10, 20, 30])` or `Stream.periodic(prefix, period)`.

Polymorphism means a static `Stream` is **not** bound to a single fixed clock
(and `match`/`automaton` lowering never inserts `when(...)` downsampling nodes
around it). Instead, every consumer port (and every `with reset(r):` scope)
instantiates the static stream independently on its own local clock:

* A single static stream `s = Stream([10, 20, 30])` may be consumed both on
  the base clock `.` (e.g. `s + 1`) and on subclocks (e.g. inside `match(c)`
  or `merge(c, s, s)`). Each distinct `(stream, target_clock)` instantiation
  consumes `s`'s dense values `s[0], s[1], s[2], ...` starting from index `0`
  on *its own* active ticks.
* Under `Semantics.SYNC`, `builder.lower_resets` realizes every subclocked
  polymorphic static stream instantiation with a gated `static_stream` node
  that emits `s[0], s[1], ...` on active ticks and `Absent` on inactive ticks.
* Under `Semantics.KAHN`, each consumer port's `StreamPort` cursor advances
  only on that consumer's active ticks (so no `Absent` insertion is needed),
  and `builder.lower_resets` inserts `static_stream` emitters where required
  to bound finite-horizon token counts (subclocked `merge`/`when`/`whennot`/
  `post` port 0, all-static stateless nodes, and all-static stateful resets).
* When a non-1-periodic static `Stream` is instantiated *inside* an active
  `with reset(r):` or `automaton` `state(...)` scope, Stage 1 emits it directly
  as a resettable stateful `static_stream` node (monomorphic to that scope's
  clock) whose index resets to `0` whenever an enclosing reset fires.
* For a static reset stream `r` in `with reset(r):`, its instantiation clock is
  determined by the enclosing `match`/`automaton` arms visible where `reset(r)`
  was entered (`reset_depth <= reset_index`). If `with reset(r):` wraps outside
  a `match`/`automaton`, `r` runs on that outer clock and is latched across the
  intervening `ClockStep`s down to the inner stateful node's clock.
* By contrast, *dynamic* streams (produced by a `Node` or unbound dynamic
  `Stream()`) and internal base-clock-pinned anchors (`Graph.pin_base_clock`)
  have concrete monomorphic clocks and *must* match their consumers' clocks
  (via explicit `when`/`whennot` or `match`/`automaton` lowering). To feed a
  global base-clock timed waveform into a `match`/`automaton` so inactive arms
  skip ticks (as in testbenches like ABRO or cruise control), anchor it with
  `when(Stream(True), seq)`: `match`/`automaton` inserts the per-arm `when`
  gates against that base-clock dynamic node before `optimize_graph` eliminates
  the outer `when(Stream(True), ...)` wrapper.

References:
  * Halbwachs et al., *The synchronous data flow programming language LUSTRE*,
    Proc. IEEE 79(9), 1991 (`when`/`current` hierarchical clocks).
  * Caspi & Pouzet, *Synchronous Kahn networks*, ICFP 1996.
  * Colaço & Pouzet, *Clocks as first class abstract types*, EMSOFT 2003
    (polymorphic clock calculus `∀ 'a. 'a` and `on` sampling).
  * Gaudelier, Cohen & Potop-Butucaru, *Relaxed activation analysis of dataflow
    networks — A clock calculus for machine learning and real-time scheduling*,
    FDL 2026 (arXiv:2607.21797).
"""

import collections
from collections.abc import Iterator, Mapping, Sequence
import dataclasses

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import (
    causal_node_order,
)
from psyre.common import (
    ArmScope,
    MergeSelector,
    TagKey,
    WhenSelector,
    format_tag,
    is_tautological_selector,
    tag_sort_key,
)
from psyre.graph import (
    Graph,
    Node,
)
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


@dataclasses.dataclass(frozen=True)
class ClockStep:
  """Single subclock sampling refinement in synchronous `on` notation."""

  selector: Stream = dataclasses.field(compare=False)
  values: frozenset[TagKey]
  negated: bool = False
  is_boolean: bool = False
  selector_label: str = dataclasses.field(default='', compare=False)
  selector_id: int = dataclasses.field(init=False, repr=False)

  def __post_init__(self) -> None:
    object.__setattr__(self, 'selector_id', self.selector.id)

  def __str__(self) -> str:
    sel = self.selector_label or f'S_{self.selector.id}'
    if self.is_boolean and not self.negated and len(self.values) == 1:
      ((_, truth),) = tuple(self.values)
      return f'on {sel}' if truth else f'on not {sel}'
    formatted = [
        format_tag(val) for _, val in sorted(self.values, key=tag_sort_key)
    ]
    if len(formatted) == 1:
      op = '!=' if self.negated else '=='
      return f'on ({sel} {op} {formatted[0]})'
    vals = ', '.join(formatted)
    op = 'not in' if self.negated else 'in'
    return f'on ({sel} {op} {{{vals}}})'

  def to_when_selector(self) -> WhenSelector:
    """Converts this `ClockStep` into the equivalent `WhenSelector`."""
    if self.is_boolean:
      if self.values == frozenset({(True, True)}):
        return WhenSelector(is_boolean=True, negated=self.negated)
      if self.values == frozenset({(True, False)}):
        return WhenSelector(is_boolean=True, negated=not self.negated)
    return WhenSelector(
        is_boolean=False,
        values=self.values,
        negated=self.negated,
    )


@dataclasses.dataclass(frozen=True)
class Clock:
  """Synchronous stream or node clock (`.`, `'a`, or `<root> on ...`).

  A `Clock` is either:
  * Concrete (`var is None`): rooted at the global base clock `.`
    (`BASE_CLOCK`) followed by zero or more `ClockStep` sampling refinements.
  * Polymorphic (`var is not None`, e.g. `'a`, `'b`): the universally
    quantified clock assigned to unpinned static `Stream`s. Each consumer port
    instantiates a polymorphic static stream at the port's own concrete clock.
  """

  steps: tuple[ClockStep, ...] = ()
  var: str | None = None

  def __post_init__(self) -> None:
    if self.var is not None:
      if not self.var or self.var == "'":
        raise ValueError(f'Invalid polymorphic clock variable {self.var!r}.')
      if not self.var.startswith("'"):
        object.__setattr__(self, 'var', f"'{self.var}")

  @classmethod
  def polymorphic(cls, var: int | str = 0) -> 'Clock':
    """Creates a polymorphic clock variable (e.g. `'a`, `'b`)."""
    if isinstance(var, int):
      if var < 0:
        raise ValueError(f'Clock variable index must be >= 0, got {var}.')
      letter = chr(ord('a') + (var % 26))
      cycle = var // 26
      name = f"'{letter}" if cycle == 0 else f"'{letter}{cycle}"
    else:
      name = var
    return cls(steps=(), var=name)

  @property
  def is_base(self) -> bool:
    """True when this clock is the root base clock `.`."""
    return self.var is None and not self.steps

  @property
  def is_polymorphic(self) -> bool:
    """True when rooted at a polymorphic clock variable such as `'a`."""
    return self.var is not None

  def on(self, step: ClockStep) -> 'Clock':
    """Returns a subclock refined by appending `step`."""
    return Clock(steps=self.steps + (step,), var=self.var)

  def is_prefix_of(self, target: 'Clock') -> bool:
    """Returns True when this clock is an ancestor or equal clock of target."""
    return (
        self.var == target.var
        and len(self.steps) <= len(target.steps)
        and target.steps[: len(self.steps)] == self.steps
    )

  def __len__(self) -> int:
    return len(self.steps)

  def __iter__(self) -> Iterator[ClockStep]:
    return iter(self.steps)

  def __bool__(self) -> bool:
    # Keep BASE_CLOCK truthy even though len(BASE_CLOCK) == 0.
    return True

  def __str__(self) -> str:
    root = '.' if self.var is None else self.var
    if not self.steps:
      return root
    return f"{root} {' '.join(str(step) for step in self.steps)}"


BASE_CLOCK = Clock()


class ClockError(RuntimeError):  # pylint: disable=g-bad-exception-name
  """Raised when a Psyre Graph violates static synchronous clock rules."""

  node: Node | None

  def __init__(self, message: str, *, node: Node | None = None):
    """Initializes ClockError with a diagnostic message and optional Node."""
    self.node = node
    prefix = f'Clock error at {node}: ' if node is not None else 'Clock error: '
    super().__init__(f'{prefix}{message}')


class ClockAnalysis:
  """Inferred node, stream, port, and reset clocks for a finalized `Graph`.

  Unpinned static `Stream`s appear as distinct polymorphic clock variables
  (`'a`, `'b`, ...) in `stream_clock(stream)`. Concrete instantiation clocks
  for each consumer port and reset scope are available via `input_clock` and
  `reset_clock`.
  """

  def __init__(
      self,
      graph: Graph,
      node_clocks: Mapping[int, Clock],
      stream_clocks: Mapping[int, Clock],
      *,
      port_clocks: Mapping[tuple[int, int], Clock] | None = None,
      reset_clocks: Mapping[tuple[int, int], Clock] | None = None,
      arm_prefix_clocks: Mapping[int, tuple[Clock, ...]] | None = None,
  ):
    """Initializes ClockAnalysis from inferred clock tables."""
    self._graph = graph
    self._node_clocks = dict(node_clocks)
    self._stream_clocks = dict(stream_clocks)
    self._port_clocks = {} if port_clocks is None else dict(port_clocks)
    self._reset_clocks = {} if reset_clocks is None else dict(reset_clocks)
    self._arm_prefix_clocks = (
        {} if arm_prefix_clocks is None else dict(arm_prefix_clocks)
    )

  @property
  def graph(self) -> Graph:
    """Finalized Graph analyzed by this ClockAnalysis."""
    return self._graph

  def stream_clock(self, stream: Stream) -> Clock:
    """Returns the inferred `Clock` of `stream`.

    Unpinned static streams return polymorphic clocks (`'a`, `'b`, ...).
    Dynamic streams and base-clock-pinned synthetic anchors return concrete
    clocks rooted at `BASE_CLOCK` (`.`).

    Args:
      stream: `Stream` belonging to `self.graph` (aliases are resolved).

    Returns:
      The inferred `Clock` for `stream`.

    Raises:
      ValueError: If `stream` does not belong to `self.graph`.
    """
    self._graph.require_finalized()
    resolved = self._graph.resolve_stream(stream)
    if resolved.id not in self._stream_clocks:
      raise ValueError(f'Stream {resolved!r} does not belong to this Graph.')
    return self._stream_clocks[resolved.id]

  def node_clock(self, node: Node) -> Clock:
    """Returns `node`'s execution `Clock`."""
    self._graph.validate_node(node)
    return self._node_clocks[node.id]

  def input_clock(self, node: Node, port_index: int) -> Clock:
    """Returns the expected execution `Clock` at `node.inputs[port_index]`.

    For same-clock inputs this equals `node_clock(node)`. For `merge` arms,
    `buffer`'s input port (`0`), and `current`'s subclocked data input port
    (`1`), this returns the concrete clock on which that port is consumed (and
    on which a polymorphic static input is instantiated).

    Args:
      node: `Node` belonging to `self.graph`.
      port_index: Zero-based index into `node.inputs`.

    Returns:
      The concrete `Clock` expected at input `port_index`.

    Raises:
      IndexError: If `port_index` is outside `[0, len(node.inputs))`.
    """
    self._graph.validate_node(node)
    if port_index < 0 or port_index >= len(node.inputs):
      raise IndexError(
          f'Input port index {port_index} out of range for {node} '
          f'({len(node.inputs)} inputs).'
      )
    return self._port_clocks.get(
        (node.id, port_index), self._node_clocks[node.id]
    )

  def reset_clock(self, node: Node, reset_index: int) -> Clock:
    """Returns the scope `Clock` of `node.resets[reset_index]`.

    For a polymorphic static reset stream, this is the clock implied by the
    enclosing `match`/`automaton` arm scopes visible when that `reset(...)`
    context was opened (`scope.reset_depth <= reset_index`). For a dynamic or
    pinned reset stream, this equals its `stream_clock`.

    Args:
      node: `Node` belonging to `self.graph`.
      reset_index: Zero-based index into `node.resets`.

    Returns:
      The concrete `Clock` of the reset scope at `reset_index`.

    Raises:
      IndexError: If `reset_index` is outside `[0, len(node.resets))`.
    """
    self._graph.validate_node(node)
    if reset_index < 0 or reset_index >= len(node.resets):
      raise IndexError(
          f'Reset index {reset_index} out of range for {node} '
          f'({len(node.resets)} resets).'
      )
    if (node.id, reset_index) in self._reset_clocks:
      return self._reset_clocks[(node.id, reset_index)]
    return self.stream_clock(node.resets[reset_index])

  def arm_prefix_clocks(self, node: Node) -> tuple[Clock, ...]:
    """Returns the selector evaluation clock for each `ArmScope` on `node`."""
    self._graph.validate_node(node)
    return self._arm_prefix_clocks.get(node.id, ())

  def node_activity_clocks(self, node: Node) -> tuple[Clock, ...]:
    """Returns prefix Clocks from base/root up to node's execution Clock."""
    self._graph.validate_node(node)
    node_clk = self._node_clocks[node.id]
    return tuple(
        Clock(steps=node_clk.steps[:i], var=node_clk.var)
        for i in range(len(node_clk.steps) + 1)
    )


class _ClockAnalyzer:
  """Infers and validates Lustre-style syntactic stream clocks."""

  def __init__(self, graph: Graph):
    self._graph = graph
    self._ext_clocks: dict[int, Clock] = {}
    self._node_clocks: dict[int, Clock] = {}
    self._stream_clocks: dict[int, Clock] = {}
    self._port_clocks: dict[tuple[int, int], Clock] = {}
    self._reset_clocks: dict[tuple[int, int], Clock] = {}
    self._arm_prefix_clocks: dict[int, tuple[Clock, ...]] = {}
    self._selector_labels: dict[int, str] = self._compute_selector_labels()

  def _compute_selector_labels(self) -> dict[int, str]:
    """Computes human-readable selector labels from explicit stream names."""
    explicit_by_id = self._graph.explicit_stream_names()
    name_counts = collections.Counter(explicit_by_id.values())
    labels: dict[int, str] = {}
    for stream in self._graph.streams:
      name = explicit_by_id.get(stream.id)
      if name is None:
        labels[stream.id] = f'S_{stream.id}'
      elif name_counts[name] > 1:
        labels[stream.id] = f'{name}_{stream.id}'
      else:
        labels[stream.id] = name
    return labels

  def _label(self, stream: Stream) -> str:
    """Returns the disambiguated human-readable label for stream."""
    return self._selector_labels.get(stream.id, f'S_{stream.id}')

  def run(self) -> ClockAnalysis:
    """Computes node and stream clocks across the finalized Graph."""
    order = causal_node_order(self._graph)
    pinned_ext = {
        s.id: BASE_CLOCK
        for s in self._graph.streams
        if self._graph.producer(s) is None
        and self._graph.is_base_clock_pinned(s)
    }
    # Iterate to fixed point so delayed `fby`/`pre` feedback cycles propagate
    # clocks along every acyclic chain plus one settling pass for indicator
    # merges/`~` selectors (`len(order) + 3` upper bound). Reset `_ext_clocks`
    # from `pinned_ext` each sweep so stale speculative bindings cannot persist.
    last_changed: Node | None = None
    for _ in range(len(order) + 3):
      prev_ext = dict(self._ext_clocks)
      self._ext_clocks = dict(pinned_ext)
      changed = False
      for node in order:
        node_clk, out_clk = self._infer_node_clocks(node)
        if (
            self._node_clocks.get(node.id) != node_clk
            or self._stream_clocks.get(node.output.id) != out_clk
        ):
          self._node_clocks[node.id] = node_clk
          self._stream_clocks[node.output.id] = out_clk
          last_changed = node
          changed = True
      if not changed and self._ext_clocks == prev_ext:
        break
    else:
      raise ClockError(
          'cyclic clock dependency did not converge.',
          node=last_changed,
      )

    poly_idx = 0
    for stream in self._graph.streams:
      if self._graph.producer(stream) is None:
        if self._graph.is_polymorphic_static(stream):
          self._stream_clocks[stream.id] = Clock.polymorphic(poly_idx)
          poly_idx += 1
        elif stream.id in self._ext_clocks:
          self._stream_clocks[stream.id] = self._ext_clocks[stream.id]
        else:
          self._stream_clocks[stream.id] = BASE_CLOCK

    for node in order:
      self._validate_node_clocks(node)
    return ClockAnalysis(
        self._graph,
        self._node_clocks,
        self._stream_clocks,
        port_clocks=self._port_clocks,
        reset_clocks=self._reset_clocks,
        arm_prefix_clocks=self._arm_prefix_clocks,
    )

  def _append_norm_step(
      self,
      base: Clock,
      selector: Stream,
      norm: tuple[bool, frozenset[TagKey], bool],
      *,
      label: str | None = None,
  ) -> Clock:
    """Appends a canonical ClockStep to base (or returns base if trivial)."""
    if is_tautological_selector(norm):
      return base
    is_bool, vals, neg = norm
    return base.on(
        ClockStep(
            selector,
            vals,
            negated=neg,
            is_boolean=is_bool,
            selector_label=self._label(selector) if label is None else label,
        )
    )

  def _bool_step(
      self,
      base: Clock,
      stream: Stream,
      truth: bool,
      *,
      label: str | None = None,
  ) -> Clock:
    """Appends a boolean `on` / `on not` ClockStep onto `base`."""
    return self._append_norm_step(
        base, stream, (True, frozenset({(True, truth)}), False), label=label
    )

  def _unwrap_sampled_selector(self, stream: Stream, base: Clock) -> Stream:
    """Unwraps `when`/`whennot` nodes sampling a parent selector onto base."""

    def can_unwrap(prod: Node) -> bool:
      curr_clk = self._stream_clocks.get(prod.output.id)
      payload = prod.inputs[1]
      return (
          curr_clk is not None
          and curr_clk.is_prefix_of(base)
          and (
              not self._graph.is_polymorphic_static(payload)
              or payload.is_one_periodic
          )
      )

    return self._graph.unwrap_when(stream, can_unwrap=can_unwrap)

  def _can_unwrap_poly_selector(
      self, selector: Stream, prod_clk: Clock | None, base: Clock
  ) -> bool:
    """Returns True when substituting `selector` on `base` preserves rate."""
    # A non-1-periodic polymorphic static stream on a strict ancestor clock
    # `prod_clk < base` advances on `prod_clk`'s ticks, *not* on `base`'s
    # ticks; collapsing across `prod` onto `selector` would lose the
    # intermediate sampling rate.
    return (
        prod_clk == base
        or not self._graph.is_polymorphic_static(selector)
        or selector.is_one_periodic
    )

  def _indicator_merge_step(
      self, base: Clock, prod: Node, expected: bool, seen: set[int]
  ) -> Clock | None:
    """Collapses a boolean-constant indicator `merge` into one `ClockStep`."""
    # Matches Lustre/Heptagon `match` outputs whose arms are all 1-periodic
    # truthy/falsy constants (e.g., `merge(s, {A: True, B: False})`), folding
    # `on flag` back to the equivalent predicate directly on `s`.
    lookup = prod.kwargs.get('lookup')
    if not isinstance(lookup, MergeSelector):
      return None
    arm_bools: list[bool] = []
    for arm_stream in prod.inputs[1:]:
      val = self._graph.const_bool(arm_stream)
      if val is None:
        return None
      arm_bools.append(val)
    prod_clk = self._node_clocks.get(prod.id)
    sel_stream = self._unwrap_sampled_selector(prod.inputs[0], base)
    if lookup.is_boolean:
      if arm_bools[0] == arm_bools[1]:
        if arm_bools[0] == expected:
          return base
        return self._bool_step(base, prod.output, expected)
      if not self._can_unwrap_poly_selector(sel_stream, prod_clk, base):
        return None
      return self._refine_clock(
          base, sel_stream, arm_bools[0] == expected, seen=seen
      )
    default_idx = lookup.default_index
    negated = (
        default_idx is not None and arm_bools[default_idx] == expected
    )
    tags = frozenset(
        key
        for key, idx in lookup.arms.items()
        if (arm_bools[idx] == expected) != negated
    )
    if not negated and not tags:
      return self._bool_step(base, prod.output, expected)
    norm = self._graph.normalize_when_selector(
        sel_stream,
        WhenSelector(is_boolean=False, values=tags, negated=negated),
    )
    if is_tautological_selector(norm):
      return base
    if not self._can_unwrap_poly_selector(sel_stream, prod_clk, base):
      return None
    return self._append_norm_step(base, sel_stream, norm)

  def _refine_clock(
      self,
      base: Clock,
      cond: Stream,
      truth: bool,
      *,
      seen: set[int] | None = None,
  ) -> Clock:
    """Refines base by sampling when cond evaluates to truth."""
    active_seen = set() if seen is None else seen
    curr = cond
    expected = truth
    while curr.id not in active_seen:
      active_seen.add(curr.id)
      unwrapped = self._unwrap_sampled_selector(curr, base)
      if unwrapped is not curr:
        curr = unwrapped
        if curr.id in active_seen:
          break
        active_seen.add(curr.id)
      const_bool = self._graph.const_bool(curr)
      if const_bool is not None:
        if const_bool == expected:
          return base
        label = self._label(curr)
        if label == f'S_{curr.id}':
          label = repr(const_bool)
        return self._bool_step(base, curr, expected, label=label)
      prod = self._graph.producer(curr)
      if prod is None:
        break
      prod_clk = self._node_clocks.get(prod.id)
      if (
          prod.is_stream_op('__invert__')
          and len(prod.inputs) == 1
          and prod_clk is not None
          and prod_clk.is_prefix_of(base)
          and self._graph.is_bool_stream(prod.inputs[0])
          and self._can_unwrap_poly_selector(prod.inputs[0], prod_clk, base)
      ):
        curr = prod.inputs[0]
        expected = not expected
        continue
      if (
          prod.is_primitive('merge')
          and prod_clk is not None
          and prod_clk.is_prefix_of(base)
      ):
        indicator = self._indicator_merge_step(
            base, prod, expected, active_seen
        )
        if indicator is not None:
          return indicator
      break
    return self._bool_step(base, curr, expected)

  def _refine_by_selector(
      self, base: Clock, cond_s: Stream, selector: WhenSelector | None
  ) -> Clock:
    """Refines base using a WhenSelector (or plain boolean when if None)."""
    if selector is None or selector.is_boolean:
      truth = True if selector is None else not selector.negated
      return self._refine_clock(base, cond_s, truth)
    sel_s = self._unwrap_sampled_selector(cond_s, base)
    norm = self._graph.normalize_when_selector(sel_s, selector)
    is_bool, vals, _ = norm
    if is_bool and len(vals) == 1:
      ((_, truth),) = tuple(vals)
      return self._refine_clock(base, cond_s, truth)
    if self._graph.is_always_active_when(sel_s, selector, norm=norm):
      return base
    return self._append_norm_step(base, sel_s, norm)

  def _trace_arms_clock(
      self, arms: Sequence[ArmScope]
  ) -> tuple[Clock, tuple[Clock, ...]]:
    """Computes final subclock and per-arm selector evaluation clocks."""
    clk = BASE_CLOCK
    prefixes: list[Clock] = []
    for scope in arms:
      cond_s = self._graph.resolve_stream(scope.cond)
      if not self._graph.is_polymorphic_static(cond_s):
        cond_clk = self._bind_or_lookup_stream_clock(cond_s, BASE_CLOCK)
        if clk.is_prefix_of(cond_clk):
          clk = cond_clk
      prefixes.append(clk)
      clk = self._refine_by_selector(clk, cond_s, scope.selector)
    return clk, tuple(prefixes)

  def _arms_clock(self, arms: Sequence[ArmScope]) -> Clock:
    """Computes the nested subclock implied by Stage-1 ArmScopes."""
    return self._trace_arms_clock(arms)[0]

  def _bind_or_lookup_stream_clock(
      self, stream: Stream, default_ext_clk: Clock = BASE_CLOCK
  ) -> Clock:
    """Returns concrete stream's clock (dynamic/external/base-pinned)."""
    if self._graph.producer(stream) is None:
      return self._ext_clocks.setdefault(stream.id, default_ext_clk)
    return self._stream_clocks.get(stream.id, BASE_CLOCK)

  def _default_node_clock(self, node: Node) -> Clock:
    """Computes fallback clock when inputs are all polymorphic or absent."""
    # Zero-input stateless nodes (e.g. `random_stream()`) have no input carrying
    # `Absent`, so they stay on `BASE_CLOCK` and are sampled by `when` later.
    arms_clk = (
        self._arms_clock(node.arms)
        if node.arms and (node.is_stateful or node.inputs)
        else None
    )
    if node.inherits_reset_clock() and node.resets:
      reset_clks = [
          self._bind_or_lookup_stream_clock(r, BASE_CLOCK)
          for r in node.resets
          if not self._graph.is_polymorphic_static(r)
      ]
      if arms_clk is not None:
        reset_clks = [c for c in reset_clks if arms_clk.is_prefix_of(c)]
      if reset_clks:
        return max(reset_clks, key=len)
    if arms_clk is not None:
      return arms_clk
    return BASE_CLOCK

  def _unify_inputs(
      self,
      streams: Sequence[Stream],
      node: Node,
      *,
      validate: bool = False,
  ) -> Clock:
    """Unifies concrete clocks of streams and returns the agreed Clock."""
    agreed: Clock | None = None
    unbound_ext: list[Stream] = []
    for stream in streams:
      if self._graph.is_polymorphic_static(stream):
        continue
      if self._graph.producer(stream) is not None:
        if stream.id not in self._stream_clocks:
          continue
        eff_clk = self._stream_clocks[stream.id]
      elif stream.id in self._ext_clocks:
        eff_clk = self._ext_clocks[stream.id]
      else:
        unbound_ext.append(stream)
        continue
      if agreed is None:
        agreed = eff_clk
      elif agreed != eff_clk:
        if validate:
          if (
              not node.arms
              or agreed.is_prefix_of(eff_clk)
              or eff_clk.is_prefix_of(agreed)
          ):
            suffix = ''
          else:
            suffix = ' (incompatible with enclosing activity clock)'
          raise ClockError(
              f'incompatible input clocks {agreed} vs {eff_clk}{suffix}.',
              node=node,
          )
        elif agreed.is_prefix_of(eff_clk):
          agreed = eff_clk
    if agreed is None:
      agreed = self._default_node_clock(node)
    for stream in unbound_ext:
      self._ext_clocks[stream.id] = agreed
    return agreed

  def _subclock_ports(
      self, node: Node, node_clk: Clock
  ) -> list[tuple[int, Stream, Clock, str]]:
    """Returns `(port_idx, stream, expected_clk, desc)` for subclocked ports."""
    if node.is_primitive('merge'):
      s_stream = node.inputs[0]
      lookup: MergeSelector = node.kwargs['lookup']
      return [
          (
              1 + idx,
              arm_stream,
              self._refine_by_selector(
                  node_clk, s_stream, lookup.arm_selector(idx)
              ),
              f'merge arm {idx}',
          )
          for idx, arm_stream in enumerate(node.inputs[1:])
      ]
    if node.is_primitive('buffer'):
      return [(
          0,
          node.inputs[0],
          self._refine_clock(node_clk, node.inputs[1], True),
          'buffer input',
      )]
    return []

  def _infer_node_clocks(self, node: Node) -> tuple[Clock, Clock]:
    """Returns `(node_execution_clock, output_stream_clock)` for `node`."""
    node_clk = self._unify_inputs(node.rate_inputs(), node)
    if node.is_primitive('when', 'whennot'):
      sel = (
          WhenSelector(is_boolean=True, negated=True)
          if node.op == 'whennot'
          else node.kwargs.get('selector')
      )
      return node_clk, self._refine_by_selector(node_clk, node.inputs[0], sel)
    for _, port_stream, expected_clk, _ in self._subclock_ports(node, node_clk):
      if not self._graph.is_polymorphic_static(port_stream):
        self._bind_or_lookup_stream_clock(port_stream, expected_clk)
    if node.is_primitive('current') and len(node.inputs) >= 2:
      i_stream = node.inputs[1]
      if not self._graph.is_polymorphic_static(i_stream):
        self._bind_or_lookup_stream_clock(i_stream, node_clk)
    if node.is_primitive('buffer'):
      return node_clk, self._refine_clock(node_clk, node.inputs[2], True)
    return node_clk, node_clk

  def _validate_node_clocks(self, node: Node) -> None:
    """Validates all instantaneous, subclock, delayed, and reset clocks."""
    if node.arms:
      self._arm_prefix_clocks[node.id] = self._trace_arms_clock(node.arms)[1]
    self._unify_inputs(node.rate_inputs(), node, validate=True)
    node_clk = self._node_clocks[node.id]
    for port_idx, port_stream, expected_clk, desc in self._subclock_ports(
        node, node_clk
    ):
      if expected_clk != node_clk:
        self._port_clocks[(node.id, port_idx)] = expected_clk
      if not self._graph.is_polymorphic_static(port_stream):
        eff_clk = self._bind_or_lookup_stream_clock(port_stream, expected_clk)
        if eff_clk != expected_clk:
          raise ClockError(
              f'{desc} {self._label(port_stream)} has clock '
              f'{eff_clk}, expected {expected_clk}.',
              node=node,
          )
    if node.is_primitive('current') and len(node.inputs) >= 2:
      i_stream = node.inputs[1]
      if not self._graph.is_polymorphic_static(i_stream):
        i_clk = self._bind_or_lookup_stream_clock(i_stream, node_clk)
        if not node_clk.is_prefix_of(i_clk):
          raise ClockError(
              f'current input {self._label(i_stream)} has clock {i_clk}, '
              f'expected {node_clk} or a subclock of it.',
              node=node,
          )
        if i_clk != node_clk:
          self._port_clocks[(node.id, 1)] = i_clk
    self._check_node_resets(node)

  def _check_node_resets(self, node: Node) -> None:
    """Checks reset stream clocks and activation scopes."""
    node_clk = self._node_clocks[node.id]
    for r_idx, r_stream in enumerate(node.resets):
      if self._graph.is_polymorphic_static(r_stream):
        r_arms = node.arms_at_reset(r_idx)
        r_clk = self._arms_clock(r_arms) if r_arms else BASE_CLOCK
      else:
        r_clk = self._bind_or_lookup_stream_clock(r_stream)
      if not r_clk.is_prefix_of(node_clk):
        raise ClockError(
            f'reset stream {self._label(r_stream)} on clock {r_clk} is '
            f'incompatible with reset scope clock {node_clk}.',
            node=node,
        )
      self._reset_clocks[(node.id, r_idx)] = r_clk
    if node.is_stateful and node_clk != BASE_CLOCK and not node.resets:
      raise ClockError(
          f'stateful node executes on subclock {node_clk} outside a matching '
          f'activation scope (expected {BASE_CLOCK}).',
          node=node,
      )


def check_clocks(graph: Graph) -> ClockAnalysis:
  """Infers and validates synchronous clocks across graph.

  Static `Stream`s without a producer (and not base-clock-pinned) receive
  polymorphic clocks (`'a`, `'b`, ...), regardless of whether they are
  1-periodic constants or multi-element periodic sequences. Each consumer port
  instantiates them at the port's own inferred clock (`input_clock` /
  `reset_clock`).

  Args:
    graph: Finalized Psyre `Graph` to analyze.

  Returns:
    A `ClockAnalysis` containing inferred node, stream, port, reset, and
    activity clocks.

  Raises:
    RuntimeError: If `graph` has not been finalized.
    ClockError: If any node, port, activity, or reset violates clock rules.
  """
  graph.require_finalized()
  return _ClockAnalyzer(graph).run()
