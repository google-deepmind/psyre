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


class _ClockVar:
  """Mutable union-find clock variable (`α`) for Colaço-Pouzet unification."""

  __slots__ = ('link',)

  def __init__(self):
    self.link: '_ClockTerm | None' = None


class _BaseTerm:
  """Singleton representing the global base clock (`.`)."""

  __slots__ = ()


_BASE_TERM = _BaseTerm()
_TRUE_SELECTOR = WhenSelector(is_boolean=True, negated=False)


class _OnTerm:
  """Lazy subclock refinement `base on (cond, selector)`."""

  __slots__ = ('base', 'cond', 'selector', 'owner_id')

  def __init__(
      self,
      base: '_ClockTerm',
      cond: Stream,
      selector: WhenSelector | None,
      owner_id: int | None = None,
  ):
    self.base = base
    self.cond = cond
    self.selector = selector
    self.owner_id = owner_id


_ClockTerm = _BaseTerm | _ClockVar | _OnTerm
_RootTerm = _BaseTerm | _ClockVar
_Decomposed = tuple[_RootTerm, tuple[ClockStep, ...], tuple[_ClockTerm, ...]]


def _dec_is_prefix(a: _Decomposed, b: _Decomposed) -> bool:
  """Returns True when decomposed clock term `a` is a prefix of `b`."""
  r1, s1, _ = a
  r2, s2, _ = b
  if r1 is _BASE_TERM and not s1:
    # Every concrete clock is rooted at `.`, so `.` prefixes every clock term.
    return True
  return r1 is r2 and len(s1) <= len(s2) and s2[: len(s1)] == s1


def _dec_is_equal(a: _Decomposed, b: _Decomposed) -> bool:
  """Returns True when decomposed clock terms `a` and `b` are equal."""
  return a[0] is b[0] and a[1] == b[1]


class _ClockAnalyzer:
  """Infers and validates Lustre-style clocks via Colaço-Pouzet unification."""

  def __init__(self, graph: Graph):
    self._graph = graph
    self._all_vars: list[_ClockVar] = []
    self._node_terms: dict[int, _ClockTerm] = {}
    self._stream_terms: dict[int, _ClockTerm] = {}
    self._node_clocks: dict[int, Clock] = {}
    self._stream_clocks: dict[int, Clock] = {}
    self._port_clocks: dict[tuple[int, int], Clock] = {}
    self._reset_clocks: dict[tuple[int, int], Clock] = {}
    self._arm_prefix_clocks: dict[int, tuple[Clock, ...]] = {}
    self._selector_labels: dict[int, str] = self._compute_selector_labels()

  def _fresh_var(self) -> _ClockVar:
    var = _ClockVar()
    self._all_vars.append(var)
    return var

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

  def _find(self, term: _ClockTerm) -> _ClockTerm:
    """Follows union-find links with path compression."""
    if isinstance(term, _ClockVar) and term.link is not None:
      root = self._find(term.link)
      term.link = root
      return root
    return term

  def _resolve_step(
      self,
      base_dec: _Decomposed,
      cond_s: Stream,
      selector: WhenSelector | None,
      owner_id: int | None,
      resolving: frozenset[int],
  ) -> ClockStep | None:
    """Canonicalizes `(cond_s, selector)` on `base_dec`, or None if trivial."""
    inner_resolving = (
        resolving | {owner_id} if owner_id is not None else resolving
    )

    def can_unwrap_when(prod: Node) -> bool:
      if prod.id in inner_resolving:
        return False
      curr_term = self._stream_terms.get(prod.output.id)
      if curr_term is None:
        return False
      payload = prod.inputs[1]
      if (
          self._graph.is_polymorphic_static(payload)
          and not payload.is_one_periodic
      ):
        return False
      curr_dec = self._decompose(curr_term, inner_resolving | {prod.id})
      return _dec_is_prefix(curr_dec, base_dec)

    def can_unwrap_node(prod: Node) -> bool:
      if prod.id in inner_resolving:
        return False
      prod_term = self._node_terms.get(prod.id)
      if prod_term is None:
        return False
      prod_dec = self._decompose(prod_term, inner_resolving | {prod.id})
      return _dec_is_prefix(prod_dec, base_dec)

    def can_unwrap_poly(sel_s: Stream, prod: Node) -> bool:
      # A non-1-periodic polymorphic static stream on a strict ancestor clock
      # `prod_clk < base` advances on `prod_clk`'s ticks, *not* on `base`'s
      # ticks; collapsing across `prod` onto `sel_s` would lose that rate.
      if (
          not self._graph.is_polymorphic_static(sel_s)
          or sel_s.is_one_periodic
      ):
        return True
      if prod.id in inner_resolving:
        return False
      prod_term = self._node_terms.get(prod.id)
      if prod_term is None:
        return False
      prod_dec = self._decompose(prod_term, inner_resolving | {prod.id})
      return _dec_is_equal(prod_dec, base_dec)

    resolved = self._graph.resolve_clock_selector_step(
        cond_s,
        selector,
        can_unwrap_when=can_unwrap_when,
        can_unwrap_node=can_unwrap_node,
        can_unwrap_poly=can_unwrap_poly,
    )
    if resolved is None:
      return None
    sel_stream, norm, is_const_false = resolved
    if is_tautological_selector(norm):
      return None
    label: str | None = None
    if is_const_false:
      cur_label = self._label(sel_stream)
      if cur_label == f'S_{sel_stream.id}':
        label = repr(self._graph.const_bool(sel_stream))
    is_bool, vals, neg = norm
    return ClockStep(
        sel_stream,
        vals,
        negated=neg,
        is_boolean=is_bool,
        selector_label=self._label(sel_stream) if label is None else label,
    )

  def _decompose(
      self,
      term: _ClockTerm,
      resolving: frozenset[int] = frozenset(),
  ) -> _Decomposed:
    """Reduces `term` lazily to `(root, steps, prefix_terms)`."""
    t = self._find(term)
    if isinstance(t, (_BaseTerm, _ClockVar)):
      return (t, (), (t,))
    base_dec = self._decompose(t.base, resolving)
    step = self._resolve_step(
        base_dec, t.cond, t.selector, t.owner_id, resolving
    )
    if step is None:
      return base_dec
    root, steps, prefixes = base_dec
    return (root, steps + (step,), prefixes + (t,))

  def _materialize(self, term: _ClockTerm) -> Clock:
    """Converts a grounded `_ClockTerm` into a concrete `Clock`."""
    _, steps, _ = self._decompose(term)
    return Clock(steps=steps)

  def _raise_mismatch(
      self,
      s1: tuple[ClockStep, ...],
      s2: tuple[ClockStep, ...],
      node: Node,
      mismatch: tuple[str, Stream] | None,
  ) -> None:
    """Raises a descriptive `ClockError` for incompatible clock step chains."""
    c1, c2 = Clock(steps=s1), Clock(steps=s2)
    if mismatch is not None:
      desc, stream = mismatch
      raise ClockError(
          f'{desc} {self._label(stream)} has clock {c1}, expected {c2}.',
          node=node,
      )
    if not node.arms or c1.is_prefix_of(c2) or c2.is_prefix_of(c1):
      suffix = ''
    else:
      suffix = ' (incompatible with enclosing activity clock)'
    raise ClockError(
        f'incompatible input clocks {c1} vs {c2}{suffix}.',
        node=node,
    )

  def _unify(
      self,
      t1: _ClockTerm,
      t2: _ClockTerm,
      *,
      node: Node,
      mismatch: tuple[str, Stream] | None = None,
      is_output: bool = False,
  ) -> None:
    """Unifies two `_ClockTerm`s with lazy step reduction and occurs check."""
    r1, s1, p1 = self._decompose(t1)
    r2, s2, p2 = self._decompose(t2)
    m, n = len(s1), len(s2)
    while m > 0 and n > 0:
      if s1[m - 1] != s2[n - 1]:
        self._raise_mismatch(s1, s2, node, mismatch)
      m -= 1
      n -= 1
    if m == 0 and n == 0:
      if r1 is r2:
        return
      if isinstance(r1, _ClockVar):
        r1.link = r2
      elif isinstance(r2, _ClockVar):
        r2.link = r1
      return
    root, target = (r1, p2[n]) if m == 0 else (r2, p1[m])
    if r1 is r2:
      if is_output or node.is_primitive('fby', 'pre'):
        raise ClockError(
            'cyclic clock dependency did not converge.', node=node
        )
      self._raise_mismatch(s1, s2, node, mismatch)
    if isinstance(root, _ClockVar):
      root.link = target
      return
    self._raise_mismatch(s1, s2, node, mismatch)

  def _trace_arms_terms(
      self, arms: Sequence[ArmScope]
  ) -> tuple[_ClockTerm, tuple[_ClockTerm, ...]]:
    """Computes symbolic subclock and per-arm selector evaluation terms."""
    term: _ClockTerm = _BASE_TERM
    prefixes: list[_ClockTerm] = []
    for scope in arms:
      cond_s = self._graph.resolve_stream(scope.cond)
      if not self._graph.is_polymorphic_static(cond_s):
        cond_term = self._stream_terms[cond_s.id]
        if _dec_is_prefix(self._decompose(term), self._decompose(cond_term)):
          term = cond_term
      prefixes.append(term)
      term = _OnTerm(term, cond_s, scope.selector)
    return term, tuple(prefixes)

  def _has_arm_scope(self, node: Node) -> bool:
    """True when `node`'s default clock inherits enclosing `ArmScope`s."""
    # Zero-input stateless nodes (e.g. `random_stream()`) have no input carrying
    # `Absent`, so they stay on `BASE_CLOCK` and are sampled by `when` later.
    return bool(node.arms and (node.is_stateful or node.inputs))

  def _default_node_term(self, node: Node) -> _ClockTerm:
    """Computes default scope clock term from enclosing `arms` / `resets`."""
    arms_term = (
        self._trace_arms_terms(node.arms)[0]
        if self._has_arm_scope(node)
        else None
    )
    if node.inherits_reset_clock() and node.resets:
      reset_items: list[tuple[_ClockTerm, _Decomposed]] = []
      arms_dec = (
          self._decompose(arms_term) if arms_term is not None else None
      )
      for r in node.resets:
        if self._graph.is_polymorphic_static(r):
          continue
        r_term = self._stream_terms[r.id]
        r_dec = self._decompose(r_term)
        if arms_dec is None or _dec_is_prefix(arms_dec, r_dec):
          reset_items.append((r_term, r_dec))
      if reset_items:
        return max(
            reset_items,
            key=lambda item: (
                len(item[1][1]),
                isinstance(item[1][0], _ClockVar),
            ),
        )[0]
    if arms_term is not None:
      return arms_term
    return _BASE_TERM

  def _has_free_scope_var(self, node: Node) -> bool:
    """True if `node`'s scope default depends on an unresolved `_ClockVar`."""
    streams: list[Stream] = []
    if node.inherits_reset_clock():
      streams.extend(node.resets)
    if len(node.arms) > 1:
      streams.extend(
          self._graph.resolve_stream(scope.cond) for scope in node.arms[1:]
      )
    return any(
        not self._graph.is_polymorphic_static(s)
        and isinstance(self._decompose(self._stream_terms[s.id])[0], _ClockVar)
        for s in streams
    )

  def _subclock_port_terms(
      self, node: Node, node_term: _ClockTerm
  ) -> list[tuple[int, Stream, _ClockTerm, str]]:
    """Returns `(port_idx, stream, expected_term, desc)` for subclock ports."""
    if node.is_primitive('merge'):
      s_stream = node.inputs[0]
      lookup: MergeSelector = node.kwargs['lookup']
      return [
          (
              1 + idx,
              arm_stream,
              _OnTerm(node_term, s_stream, lookup.arm_selector(idx)),
              f'merge arm {idx}',
          )
          for idx, arm_stream in enumerate(node.inputs[1:])
      ]
    if node.is_primitive('buffer'):
      return [(
          0,
          node.inputs[0],
          _OnTerm(node_term, node.inputs[1], _TRUE_SELECTOR),
          'buffer input',
      )]
    return []

  def run(self) -> ClockAnalysis:
    """Computes node and stream clocks across the finalized Graph."""
    order = causal_node_order(self._graph)
    causal_pos = {node.id: idx for idx, node in enumerate(order)}

    # Phase 0 (Elaboration): polymorphic static streams have no shared var
    # (each use instantiates a fresh variable), base-pinned streams get
    # `_BASE_TERM`, unbound dynamic `Stream()`s get a shared `_ClockVar`, and
    # every node gets a `_ClockVar` (plus an output var for clock-changing ops).
    for stream in self._graph.streams:
      if self._graph.producer(stream) is None:
        if self._graph.is_polymorphic_static(stream):
          continue
        self._stream_terms[stream.id] = (
            _BASE_TERM
            if self._graph.is_base_clock_pinned(stream)
            else self._fresh_var()
        )
    for node in order:
      node_var = self._fresh_var()
      self._node_terms[node.id] = node_var
      self._stream_terms[node.output.id] = (
          self._fresh_var()
          if node.is_primitive('when', 'whennot', 'buffer')
          else node_var
      )

    # Phase 1a: unify `rate_inputs()`, clock-changing outputs, and un-scoped
    # input-free nodes (anchored to `_BASE_TERM`).
    scoped_defaults: list[Node] = []
    for node in order:
      node_term = self._node_terms[node.id]
      dyn_ins = [
          s
          for s in node.rate_inputs()
          if not self._graph.is_polymorphic_static(s)
      ]
      for s in dyn_ins:
        self._unify(node_term, self._stream_terms[s.id], node=node)
      if not dyn_ins:
        has_dyn_resets = node.inherits_reset_clock() and any(
            not self._graph.is_polymorphic_static(r) for r in node.resets
        )
        if self._has_arm_scope(node) or has_dyn_resets:
          scoped_defaults.append(node)
        else:
          self._unify(node_term, _BASE_TERM, node=node)
      if node.is_primitive('when', 'whennot'):
        sel = (
            WhenSelector(is_boolean=True, negated=True)
            if node.op == 'whennot'
            else node.kwargs.get('selector')
        )
        self._unify(
            self._stream_terms[node.output.id],
            _OnTerm(node_term, node.inputs[0], sel, owner_id=node.id),
            node=node,
            is_output=True,
        )
      elif node.is_primitive('buffer'):
        self._unify(
            self._stream_terms[node.output.id],
            _OnTerm(
                node_term, node.inputs[2], _TRUE_SELECTOR, owner_id=node.id
            ),
            node=node,
            is_output=True,
        )

    # Phase 1b: anchor input-free scoped nodes whose scope terms are already
    # determined (ordered by arm nesting depth, then causal order).
    scoped_defaults.sort(key=lambda n: (len(n.arms), causal_pos[n.id]))
    deferred_scoped: list[Node] = []
    for node in scoped_defaults:
      if self._has_free_scope_var(node):
        deferred_scoped.append(node)
      else:
        self._unify(
            self._node_terms[node.id],
            self._default_node_term(node),
            node=node,
        )

    # Phase 1c: unify subclocked input ports (`merge` arms, `buffer` input 0).
    for node in order:
      node_term = self._node_terms[node.id]
      for _, port_stream, expected_term, desc in self._subclock_port_terms(
          node, node_term
      ):
        if not self._graph.is_polymorphic_static(port_stream):
          self._unify(
              self._stream_terms[port_stream.id],
              expected_term,
              node=node,
              mismatch=(desc, port_stream),
          )

    # Phase 1d: anchor input-free scoped nodes whose dynamic resets/nested arms
    # depended on constraints resolved in Phase 1c.
    for node in deferred_scoped:
      self._unify(
          self._node_terms[node.id],
          self._default_node_term(node),
          node=node,
      )

    # Phase 2 (Grounding): default any still-free clock variables to their
    # enclosing scope default, deepest `current` consumer, or `_BASE_TERM`.
    scoped_nodes = [
        n
        for n in order
        if self._has_arm_scope(n) or (n.inherits_reset_clock() and n.resets)
    ]
    scoped_nodes.sort(key=lambda n: (len(n.arms), causal_pos[n.id]))
    for node in scoped_nodes:
      if isinstance(self._find(self._node_terms[node.id]), _ClockVar):
        default_term = self._default_node_term(node)
        default_root, default_steps, _ = self._decompose(default_term)
        if default_root is not _BASE_TERM or default_steps:
          self._unify(self._node_terms[node.id], default_term, node=node)
    current_nodes = [
        n
        for n in order
        if n.is_primitive('current')
        and len(n.inputs) >= 2
        and not self._graph.is_polymorphic_static(n.inputs[1])
    ]
    current_nodes.sort(
        key=lambda n: len(self._decompose(self._node_terms[n.id])[1]),
        reverse=True,
    )
    for node in current_nodes:
      node_term = self._node_terms[node.id]
      i_root = self._decompose(self._stream_terms[node.inputs[1].id])[0]
      n_root = self._decompose(node_term)[0]
      if isinstance(i_root, _ClockVar) and i_root is not n_root:
        self._unify(i_root, node_term, node=node)
    for var in self._all_vars:
      root = self._find(var)
      if isinstance(root, _ClockVar):
        root.link = _BASE_TERM

    # Phase 3 (Materialization & Validation): convert grounded terms into
    # concrete `Clock`s and validate prefix/reset/scope constraints.
    poly_idx = 0
    for stream in self._graph.streams:
      if self._graph.producer(stream) is None:
        if self._graph.is_polymorphic_static(stream):
          self._stream_clocks[stream.id] = Clock.polymorphic(poly_idx)
          poly_idx += 1
        else:
          self._stream_clocks[stream.id] = self._materialize(
              self._stream_terms[stream.id]
          )
    for node in order:
      self._node_clocks[node.id] = self._materialize(self._node_terms[node.id])
      self._stream_clocks[node.output.id] = self._materialize(
          self._stream_terms[node.output.id]
      )
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

  def _validate_node_clocks(self, node: Node) -> None:
    """Validates all instantaneous, subclock, current, and reset clocks."""
    if node.arms:
      _, prefix_terms = self._trace_arms_terms(node.arms)
      self._arm_prefix_clocks[node.id] = tuple(
          self._materialize(t) for t in prefix_terms
      )
    node_clk = self._node_clocks[node.id]
    for stream in node.rate_inputs():
      if self._graph.is_polymorphic_static(stream):
        continue
      eff_clk = self._stream_clocks[stream.id]
      if eff_clk != node_clk:
        self._raise_mismatch(node_clk.steps, eff_clk.steps, node, None)
    for port_idx, port_stream, expected_term, desc in self._subclock_port_terms(
        node, self._node_terms[node.id]
    ):
      expected_clk = self._materialize(expected_term)
      if expected_clk != node_clk:
        self._port_clocks[(node.id, port_idx)] = expected_clk
      if not self._graph.is_polymorphic_static(port_stream):
        eff_clk = self._stream_clocks[port_stream.id]
        if eff_clk != expected_clk:
          self._raise_mismatch(
              eff_clk.steps, expected_clk.steps, node, (desc, port_stream)
          )
    if node.is_primitive('current') and len(node.inputs) >= 2:
      i_stream = node.inputs[1]
      if not self._graph.is_polymorphic_static(i_stream):
        i_clk = self._stream_clocks[i_stream.id]
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
        r_clk = (
            self._materialize(self._trace_arms_terms(r_arms)[0])
            if r_arms
            else BASE_CLOCK
        )
      else:
        r_clk = self._stream_clocks[r_stream.id]
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
