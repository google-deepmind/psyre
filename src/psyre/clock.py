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
"""Synchronous Lustre-style clock representation, inference, and checking."""

import collections
from collections.abc import Callable, Iterator, Mapping, Sequence
import dataclasses
from typing import NoReturn

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import (
    causal_node_order,
)
from psyre.common import (
    ArmScope,
    MergeSelector,
    Semantics,
    TagKey,
    WhenSelector,
    format_tag,
    is_tautological_selector,
    normalize_when_selector,
    tag_sort_key,
)
from psyre.graph import (
    Graph,
    Node,
)
from psyre.istructure import (
    IStruct,
    as_istruct,
    is_one_periodic,
)
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


def _clock_var_name(index: int) -> str:
  """Returns the quoted polymorphic clock variable name for a 0-based index."""
  if index < 0:
    raise ValueError(f'Clock variable index must be >= 0, got {index}.')
  letter = chr(ord('a') + (index % 26))
  cycle = index // 26
  return f"'{letter}" if cycle == 0 else f"'{letter}{cycle}"


@dataclasses.dataclass(frozen=True)
class ClockStep:
  """Single subclock sampling refinement in synchronous `on` notation."""

  selector: IStruct
  values: frozenset[TagKey]
  negated: bool = False
  is_boolean: bool = dataclasses.field(default=False, compare=False)
  selector_label: str = dataclasses.field(default='', compare=False)

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


@dataclasses.dataclass(frozen=True)
class Clock:
  """Synchronous stream or node clock (`.`, `'a`, or `<root> on ...`)."""

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
    name = _clock_var_name(var) if isinstance(var, int) else var
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


def _static_constant_bool(istruct: IStruct) -> bool | None:
  """Returns bool(v) for a constant stream, or None otherwise."""
  if not is_one_periodic(istruct):
    return None
  return bool(istruct.period[0])


def _validate_node_resets(
    node: Node,
    node_clk: Clock,
    lookup_clk: Callable[[IStruct], Clock],
    *,
    forbid_cross_activity: bool,
    label_fn: Callable[[IStruct], str] | None = None,
) -> None:
  """Validates reset stream clocks and activation scopes for `node`."""
  has_valid_reset = False
  for r_handle in node.resets:
    r_istruct = r_handle.istruct
    if is_one_periodic(r_istruct):
      continue
    r_clk = lookup_clk(r_istruct)
    if not r_clk.is_prefix_of(node_clk):
      r_desc = (
          f'reset stream {label_fn(r_istruct)}'
          if label_fn is not None
          else 'reset stream'
      )
      raise ClockError(
          f'{r_desc} on clock {r_clk} is incompatible with reset scope '
          f'clock {node_clk}.',
          node=node,
      )
    if r_clk != node_clk and node.is_stateful and forbid_cross_activity:
      raise ClockError(
          f'stateful node has cross-activity reset on clock {r_clk} across '
          f'node clock {node_clk}.',
          node=node,
      )
    has_valid_reset = True
  if node.is_stateful and node_clk != BASE_CLOCK and not has_valid_reset:
    raise ClockError(
        f'stateful node executes on subclock {node_clk} outside a matching '
        f'activation scope (expected {BASE_CLOCK}).',
        node=node,
    )


class ClockAnalysis:
  """Inferred node and stream clocks for a finalized Psyre Graph."""

  def __init__(
      self,
      graph: Graph,
      node_clocks: Mapping[int, Clock],
      stream_clocks: Mapping[IStruct, Clock],
  ):
    """Initializes ClockAnalysis from inferred clock tables."""
    self._graph = graph
    self._node_clocks = dict(node_clocks)
    self._stream_clocks = dict(stream_clocks)

  @property
  def graph(self) -> Graph:
    """Finalized Graph analyzed by this ClockAnalysis."""
    return self._graph

  def stream_clock(self, stream: IStruct | Stream) -> Clock:
    """Returns the inferred Clock of stream."""
    self._graph.require_finalized()
    istruct = self._graph.resolve_stream(stream)
    if istruct not in self._stream_clocks:
      raise ValueError(f'Stream {istruct!r} does not belong to this Graph.')
    return self._stream_clocks[istruct]

  def node_clock(self, node: Node) -> Clock:
    """Returns node's execution Clock."""
    self._graph.validate_node(node)
    return self._node_clocks[node.id]

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
    self._ext_clocks: dict[IStruct, Clock] = {}
    self._node_clocks: dict[int, Clock] = {}
    self._stream_clocks: dict[IStruct, Clock] = {}
    self._selector_labels: dict[IStruct, str] = self._compute_selector_labels()

  def _compute_selector_labels(self) -> dict[IStruct, str]:
    """Computes human-readable selector labels from explicit handle names."""
    explicit_by_istruct: dict[IStruct, str] = {}
    for handle in self._graph.handles:
      if not handle.name or handle.istruct in explicit_by_istruct:
        continue
      prod = self._graph.producer(handle.istruct)
      leaf_name = handle.name.rsplit('/', 1)[-1]
      if prod is None or leaf_name != prod.op:
        explicit_by_istruct[handle.istruct] = handle.name
    name_counts = collections.Counter(explicit_by_istruct.values())
    labels: dict[IStruct, str] = {}
    for istruct in self._graph.streams:
      name = explicit_by_istruct.get(istruct)
      if name is None:
        labels[istruct] = f'S_{istruct.id}'
      elif name_counts[name] > 1:
        labels[istruct] = f'{name}_{istruct.id}'
      else:
        labels[istruct] = name
    return labels

  def _label(self, istruct: IStruct) -> str:
    """Returns the disambiguated human-readable label for istruct."""
    return self._selector_labels.get(istruct, f'S_{istruct.id}')

  def run(self) -> ClockAnalysis:
    """Computes node and stream clocks across the finalized Graph."""
    order = causal_node_order(self._graph)
    passes = 2 if any(n.op in {'fby', 'pre'} for n in order) else 1
    for _ in range(passes):
      for node in order:
        node_clk, out_clk = self._infer_node_clocks(node)
        self._node_clocks[node.id] = node_clk
        self._stream_clocks[node.output_stream] = out_clk
        p_h = node.kwargs.get('p_stream')
        if isinstance(p_h, (IStruct, Stream)):
          self._stream_clocks[as_istruct(p_h)] = node_clk

    poly_idx = 0
    for istruct in self._graph.streams:
      if self._graph.producer(istruct) is None:
        if is_one_periodic(istruct):
          self._stream_clocks[istruct] = Clock.polymorphic(poly_idx)
          poly_idx += 1
        else:
          self._stream_clocks[istruct] = self._ext_clocks.get(
              istruct, BASE_CLOCK
          )

    for node in order:
      if node.op in {'fby', 'pre'}:
        self._check_delayed_port(node)
      self._check_node_resets(node)
    return ClockAnalysis(self._graph, self._node_clocks, self._stream_clocks)

  def _fail(self, node: Node, message: str) -> NoReturn:
    raise ClockError(message, node=node)

  def _bool_step(
      self,
      base: Clock,
      stream: IStruct,
      truth: bool,
      *,
      label: str | None = None,
  ) -> Clock:
    """Appends a boolean `on` / `on not` ClockStep onto `base`."""
    return base.on(
        ClockStep(
            stream,
            frozenset({(True, truth)}),
            negated=False,
            is_boolean=True,
            selector_label=self._label(stream) if label is None else label,
        )
    )

  def _step_from_norm(
      self,
      base: Clock,
      selector: IStruct,
      norm: tuple[bool, frozenset[TagKey], bool],
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
            selector_label=self._label(selector),
        )
    )

  def _append_step(
      self,
      base: Clock,
      selector: IStruct,
      values: frozenset[TagKey],
      *,
      negated: bool = False,
  ) -> Clock:
    """Appends a normalized ClockStep to base (or returns base if trivial)."""
    norm = normalize_when_selector(
        WhenSelector(is_boolean=False, values=values, negated=negated),
        is_bool=self._graph.is_bool_stream(selector),
    )
    return self._step_from_norm(base, selector, norm)

  def _unwrap_sampled_selector(self, istruct: IStruct, base: Clock) -> IStruct:
    """Unwraps `when`/`whennot` nodes sampling a parent selector onto base."""
    curr = istruct
    seen: set[IStruct] = set()
    while curr not in seen:
      seen.add(curr)
      prod = self._graph.producer(curr)
      if (
          prod is not None
          and prod.op in {'when', 'whennot'}
          and self._stream_clocks.get(curr) == base
      ):
        curr = prod.inputs[1].istruct
        continue
      break
    return curr

  def _indicator_merge_step(
      self, base: Clock, prod: Node, expected: bool
  ) -> Clock | None:
    """Collapses a boolean-constant indicator merge into a single ClockStep."""
    lookup = prod.kwargs.get('lookup')
    if not isinstance(lookup, MergeSelector):
      return None
    arm_bools: list[bool] = []
    for arm_handle in prod.inputs[1:]:
      val = _static_constant_bool(arm_handle.istruct)
      if val is None:
        return None
      arm_bools.append(val)
    sel_istruct = self._unwrap_sampled_selector(prod.inputs[0].istruct, base)
    if lookup.is_boolean:
      if arm_bools[0] == arm_bools[1]:
        if arm_bools[0] == expected:
          return base
        return self._bool_step(base, prod.output_stream, expected)
      return self._refine_clock(base, sel_istruct, arm_bools[0] == expected)
    default_idx = lookup.default_index
    default_matches = (
        default_idx is not None and arm_bools[default_idx] == expected
    )
    if default_matches:
      excluded = frozenset(
          key for key, idx in lookup.arms.items() if arm_bools[idx] != expected
      )
      return self._append_step(base, sel_istruct, excluded, negated=True)
    matching = frozenset(
        key for key, idx in lookup.arms.items() if arm_bools[idx] == expected
    )
    if not matching:
      return self._bool_step(base, prod.output_stream, expected)
    return self._append_step(base, sel_istruct, matching, negated=False)

  def _refine_clock(self, base: Clock, cond: IStruct, truth: bool) -> Clock:
    """Refines base by sampling when cond evaluates to truth."""
    seen: set[IStruct] = set()
    curr = cond
    expected = truth
    while curr not in seen:
      seen.add(curr)
      curr = self._unwrap_sampled_selector(curr, base)
      const_bool = _static_constant_bool(curr)
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
      if prod.op == '__invert__':
        inner = prod.inputs[0].istruct
        if self._graph.is_bool_stream(inner):
          curr = inner
          expected = not expected
          continue
      prod_clk = self._node_clocks.get(prod.id)
      if (
          prod.op == 'merge'
          and prod_clk is not None
          and prod_clk.is_prefix_of(base)
      ):
        indicator = self._indicator_merge_step(base, prod, expected)
        if indicator is not None:
          return indicator
      break
    return self._append_step(
        base, curr, frozenset({(True, expected)}), negated=False
    )

  def _refine_by_selector(
      self, base: Clock, cond_s: IStruct, selector: WhenSelector | None
  ) -> Clock:
    """Refines base using a WhenSelector (or plain boolean when if None)."""
    if selector is None or selector.is_boolean:
      truth = True if selector is None else not selector.negated
      return self._refine_clock(base, cond_s, truth)
    sel_s = self._unwrap_sampled_selector(cond_s, base)
    norm = normalize_when_selector(
        selector, is_bool=self._graph.is_bool_stream(sel_s)
    )
    is_bool, vals, _ = norm
    if is_bool and len(vals) == 1:
      ((_, truth),) = tuple(vals)
      return self._refine_clock(base, cond_s, truth)
    return self._step_from_norm(base, sel_s, norm)

  def _arms_clock(self, arms: Sequence[ArmScope]) -> Clock:
    """Computes the nested subclock implied by Stage-1 ArmScopes."""
    clk = BASE_CLOCK
    for scope in arms:
      cond_s = self._graph.resolve_stream(scope.cond)
      cond_clk = self._bind_or_lookup_stream_clock(cond_s, BASE_CLOCK)
      if clk.is_prefix_of(cond_clk):
        clk = cond_clk
      clk = self._refine_by_selector(clk, cond_s, scope.selector)
    return clk

  def _bind_or_lookup_stream_clock(
      self, istruct: IStruct, default_ext_clk: Clock = BASE_CLOCK
  ) -> Clock:
    """Returns istruct's clock, binding unconstrained static streams."""
    if self._graph.producer(istruct) is None:
      return self._ext_clocks.setdefault(istruct, default_ext_clk)
    return self._stream_clocks.get(istruct, BASE_CLOCK)

  def _unify_inputs(
      self,
      handles: Sequence[Stream],
      node: Node,
  ) -> Clock:
    """Unifies effective clocks of handles and returns the agreed Clock."""
    agreed: Clock | None = None
    for handle in handles:
      istruct = handle.istruct
      if is_one_periodic(istruct):
        continue
      if (
          self._graph.producer(istruct) is not None
          and istruct not in self._stream_clocks
      ):
        continue
      eff_clk = self._bind_or_lookup_stream_clock(istruct, BASE_CLOCK)
      if agreed is None:
        agreed = eff_clk
      elif agreed != eff_clk:
        if agreed.is_prefix_of(eff_clk) or eff_clk.is_prefix_of(agreed):
          suffix = ''
        else:
          suffix = ' (incompatible with enclosing activity clock)'
        self._fail(
            node, f'incompatible input clocks {agreed} vs {eff_clk}{suffix}.'
        )
    if agreed is not None:
      return agreed
    if node.is_stateful:
      arms_clk = self._arms_clock(node.arms) if node.arms else None
      if node.op in {'pre', 'fby'} and node.resets:
        reset_clks = [
            self._bind_or_lookup_stream_clock(r.istruct, BASE_CLOCK)
            for r in node.resets
            if not is_one_periodic(r)
        ]
        if arms_clk is not None:
          reset_clks = [c for c in reset_clks if arms_clk.is_prefix_of(c)]
        if reset_clks:
          return max(reset_clks, key=len)
      if arms_clk is not None:
        return arms_clk
    return BASE_CLOCK

  def _check_exact_port(
      self,
      handle: Stream,
      expected_clk: Clock,
      node: Node,
      port_desc: str,
  ) -> None:
    """Checks a subclock port (merge arm or buffer input)."""
    istruct = handle.istruct
    if is_one_periodic(istruct):
      return
    if (
        self._graph.producer(istruct) is None
        and istruct not in self._ext_clocks
    ):
      self._ext_clocks[istruct] = expected_clk
    eff_clk = self._bind_or_lookup_stream_clock(istruct, BASE_CLOCK)
    if eff_clk != expected_clk:
      self._fail(
          node,
          f'{port_desc} {self._label(istruct)} has clock '
          f'{eff_clk}, expected {expected_clk}.',
      )

  def _infer_node_clocks(self, node: Node) -> tuple[Clock, Clock]:
    """Returns (node_execution_clock, output_stream_clock) for node."""
    if node.op in {'when', 'whennot'}:
      node_clk = self._unify_inputs(node.inputs, node)
      sel = (
          WhenSelector(is_boolean=True, negated=True)
          if node.op == 'whennot'
          else node.kwargs.get('selector')
      )
      return node_clk, self._refine_by_selector(
          node_clk, node.inputs[0].istruct, sel
      )

    if node.op == 'merge':
      s_handle = node.inputs[0]
      node_clk = self._unify_inputs((s_handle,), node)
      lookup: MergeSelector = node.kwargs['lookup']
      for idx, arm_handle in enumerate(node.inputs[1:]):
        expected_arm = self._refine_by_selector(
            node_clk, s_handle.istruct, lookup.arm_selector(idx)
        )
        self._check_exact_port(
            arm_handle, expected_arm, node, f'merge arm {idx}'
        )
      return node_clk, node_clk

    if node.op == 'buffer':
      i_handle, ip_handle, op_handle = node.inputs[:3]
      node_clk = self._unify_inputs((ip_handle, op_handle), node)
      expected_in = self._refine_clock(node_clk, ip_handle.istruct, True)
      self._check_exact_port(i_handle, expected_in, node, 'buffer input')
      return node_clk, self._refine_clock(node_clk, op_handle.istruct, True)

    node_clk = self._unify_inputs(node.inputs, node)
    return node_clk, node_clk

  def _check_delayed_port(self, node: Node) -> None:
    """Checks that fby/pre's delayed next-state stream matches node's clock."""
    node_clk = self._node_clocks[node.id]
    delayed_port = 0 if node.op == 'pre' else 1
    delayed_istruct = node.inputs[delayed_port].istruct
    if is_one_periodic(delayed_istruct):
      return
    eff_j_clk = self._bind_or_lookup_stream_clock(delayed_istruct, BASE_CLOCK)
    if eff_j_clk != node_clk:
      self._fail(
          node,
          f'delayed {node.op} input '
          f'{self._label(delayed_istruct)} has clock '
          f'{eff_j_clk}, expected {node_clk}.',
      )

  def _check_node_resets(self, node: Node) -> None:
    """Checks reset stream clocks and activation scopes."""
    _validate_node_resets(
        node,
        self._node_clocks[node.id],
        self._bind_or_lookup_stream_clock,
        forbid_cross_activity=(self._graph.semantics == Semantics.KAHN),
        label_fn=self._label,
    )


def infer_clocks(graph: Graph) -> ClockAnalysis:
  """Infers and validates synchronous clocks across graph.

  Args:
    graph: Finalized Psyre `Graph` to analyze.

  Returns:
    A `ClockAnalysis` containing inferred node, stream, and activity clocks.

  Raises:
    RuntimeError: If `graph` has not been finalized.
    ClockError: If any node, port, activity, or reset violates clock rules.
  """
  graph.require_finalized()
  return _ClockAnalyzer(graph).run()


check_clocks = infer_clocks


def check_kahn_resets(
    graph: Graph, clocks: ClockAnalysis | None = None
) -> ClockAnalysis:
  """Forbids cross-activity or supersampled stateful resets under KAHN.

  Args:
    graph: Finalized Psyre `Graph` scheduled under `Semantics.KAHN`.
    clocks: Optional precomputed `ClockAnalysis` for `graph`. When `None`,
      clocks are inferred via `infer_clocks(graph)`.

  Returns:
    The `ClockAnalysis` used to validate resets and schedule Stage-2 nodes.

  Raises:
    RuntimeError: If `graph` has not been finalized.
    ClockError: If any stateful node has a cross-activity or supersampled reset.
  """
  graph.require_finalized()
  resolved_clocks = infer_clocks(graph) if clocks is None else clocks
  for node in graph.nodes:
    if node.is_stateful:
      _validate_node_resets(
          node,
          resolved_clocks.node_clock(node),
          resolved_clocks.stream_clock,
          forbid_cross_activity=True,
      )
  return resolved_clocks
