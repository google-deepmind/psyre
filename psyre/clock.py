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
from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from typing import NoReturn

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import (
    causal_node_order,
)
from psyre.common import (
    BOOL_BINARY_OPS,
    BOOL_PRODUCER_OPS,
    MergeSelector,
    Semantics,
    TagKey,
    format_tag,
)
from psyre.graph import (
    Graph,
    Node,
)
from psyre.istructure import (
    IStruct,
    IStructHandle,
    as_istruct,
    is_one_periodic,
)

# pylint: enable=g-importing-member,g-multiple-import


def _tag_sort_key(item: TagKey) -> tuple[str, str]:
  """Provides a deterministic sort key across heterogeneous discrete tags."""
  is_bool, val = item
  return ('0' if is_bool else '1', format_tag(val))


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
        format_tag(val) for _, val in sorted(self.values, key=_tag_sort_key)
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

  def stream_clock(self, stream: IStruct | IStructHandle) -> Clock:
    """Returns the inferred Clock of stream."""
    self._graph.require_finalized()
    istruct = as_istruct(stream)
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
        if isinstance(p_h, IStructHandle):
          self._stream_clocks[p_h.istruct] = node_clk

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

  def _is_bool_stream(
      self, istruct: IStruct, seen: set[IStruct] | None = None
  ) -> bool:
    """Returns True when istruct is statically guaranteed to emit booleans."""
    active_seen = set() if seen is None else seen
    if istruct in active_seen:
      return True
    active_seen.add(istruct)
    prod = self._graph.producer(istruct)
    if prod is None:
      return istruct.static and all(
          isinstance(v, bool) for v in (*istruct.prefix, *istruct.period)
      )
    if prod.op in BOOL_PRODUCER_OPS:
      return True
    if prod.op == '__invert__':
      return self._is_bool_stream(prod.inputs[0].istruct, active_seen)
    if prod.op in BOOL_BINARY_OPS:
      return self._is_bool_stream(
          prod.inputs[0].istruct, active_seen
      ) and self._is_bool_stream(prod.inputs[1].istruct, active_seen)
    if prod.op == 'ifthenelse':
      return self._is_bool_stream(
          prod.inputs[1].istruct, active_seen
      ) and self._is_bool_stream(prod.inputs[2].istruct, active_seen)
    if prod.op == 'merge':
      return all(
          self._is_bool_stream(h.istruct, active_seen) for h in prod.inputs[1:]
      )
    if prod.op in {'fby', 'morph'}:
      return self._is_bool_stream(
          prod.inputs[0].istruct, active_seen
      ) and self._is_bool_stream(prod.inputs[1].istruct, active_seen)
    if prod.op == 'when':
      return self._is_bool_stream(prod.inputs[1].istruct, active_seen)
    return False

  def _append_step(
      self,
      base: Clock,
      selector: IStruct,
      values: frozenset[TagKey],
      *,
      negated: bool = False,
  ) -> Clock:
    """Appends a normalized ClockStep to base (or returns base if trivial)."""
    is_bool = self._is_bool_stream(selector)
    if is_bool:
      bool_domain: frozenset[TagKey] = frozenset({(True, True), (True, False)})
      relevant = values & bool_domain
      values = (bool_domain - relevant) if negated else relevant
      negated = False
      if values == bool_domain:
        return base
    elif not values and negated:
      return base
    return base.on(
        ClockStep(
            selector,
            values,
            negated=negated,
            is_boolean=is_bool,
            selector_label=self._label(selector),
        )
    )

  def _unwrap_sampled_selector(self, istruct: IStruct, base: Clock) -> IStruct:
    """Unwraps `when` nodes that only sample a parent selector onto base."""
    curr = istruct
    seen: set[IStruct] = set()
    while curr not in seen:
      seen.add(curr)
      prod = self._graph.producer(curr)
      if (
          prod is not None
          and prod.op == 'when'
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
        return base.on(
            ClockStep(
                prod.output_stream,
                frozenset({(True, expected)}),
                negated=False,
                is_boolean=True,
                selector_label=self._label(prod.output_stream),
            )
        )
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
      return base.on(
          ClockStep(
              prod.output_stream,
              frozenset({(True, expected)}),
              negated=False,
              is_boolean=True,
              selector_label=self._label(prod.output_stream),
          )
      )
    return self._append_step(base, sel_istruct, matching, negated=False)

  def _refine_clock(self, base: Clock, cond: IStruct, truth: bool) -> Clock:
    """Refines base by sampling when cond evaluates to truth."""
    seen: set[IStruct] = set()
    curr = self._unwrap_sampled_selector(cond, base)
    expected = truth
    while curr not in seen:
      seen.add(curr)
      curr = self._unwrap_sampled_selector(curr, base)
      const_bool = _static_constant_bool(curr)
      if const_bool is not None:
        if const_bool == expected:
          return base
        label = self._selector_labels.get(curr, f'S_{curr.id}')
        if label == f'S_{curr.id}':
          label = repr(const_bool)
        return base.on(
            ClockStep(
                curr,
                frozenset({(True, expected)}),
                negated=False,
                is_boolean=True,
                selector_label=label,
            )
        )
      prod = self._graph.producer(curr)
      if prod is None:
        break
      if prod.op == '__invert__':
        inner = prod.inputs[0].istruct
        if self._is_bool_stream(inner):
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

  def _merge_arm_clock(
      self,
      node_clk: Clock,
      selector: IStruct,
      lookup: MergeSelector,
      arm_index: int,
  ) -> Clock:
    """Computes the required subclock for arm_index of a merge node."""
    selector = self._unwrap_sampled_selector(selector, node_clk)
    if lookup.is_boolean:
      return self._refine_clock(node_clk, selector, arm_index == 0)
    if arm_index == lookup.default_index:
      excluded = frozenset(
          key for key, idx in lookup.arms.items() if idx != arm_index
      )
      return self._append_step(node_clk, selector, excluded, negated=True)
    matching = frozenset(
        key for key, idx in lookup.arms.items() if idx == arm_index
    )
    return self._append_step(node_clk, selector, matching, negated=False)

  def _bind_or_lookup_stream_clock(
      self, istruct: IStruct, default_ext_clk: Clock = BASE_CLOCK
  ) -> Clock:
    """Returns istruct's clock, binding unconstrained static streams."""
    if self._graph.producer(istruct) is None:
      return self._ext_clocks.setdefault(istruct, default_ext_clk)
    return self._stream_clocks.get(istruct, BASE_CLOCK)

  def _unify_inputs(
      self,
      handles: Sequence[IStructHandle],
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
          self._fail(
              node,
              f'incompatible input clocks {agreed} vs {eff_clk}.',
          )
        else:
          self._fail(
              node,
              f'incompatible input clocks {agreed} vs {eff_clk} '
              '(incompatible with enclosing activity clock).',
          )
    if agreed is not None:
      return agreed
    if node.op in {'pre', 'fby'} and node.resets:
      reset_clks = [
          self._bind_or_lookup_stream_clock(r.istruct, BASE_CLOCK)
          for r in node.resets
          if not is_one_periodic(r)
      ]
      if reset_clks:
        return max(reset_clks, key=len)
    return BASE_CLOCK

  def _check_exact_port(
      self,
      handle: IStructHandle,
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
      out_clk = self._refine_clock(
          node_clk, node.inputs[0].istruct, node.op == 'when'
      )
      return node_clk, out_clk

    if node.op == 'merge':
      s_handle = node.inputs[0]
      node_clk = self._unify_inputs((s_handle,), node)
      lookup: MergeSelector = node.kwargs['lookup']
      for idx, arm_handle in enumerate(node.inputs[1:]):
        expected_arm = self._merge_arm_clock(
            node_clk, s_handle.istruct, lookup, idx
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
    node_clk = self._node_clocks[node.id]
    is_kahn = self._graph.semantics == Semantics.KAHN
    has_valid_reset = False
    for r_handle in node.resets:
      r_istruct = r_handle.istruct
      if is_one_periodic(r_istruct):
        continue
      r_clk = self._bind_or_lookup_stream_clock(r_istruct, BASE_CLOCK)
      if not r_clk.is_prefix_of(node_clk):
        self._fail(
            node,
            f'reset stream {self._label(r_istruct)} on clock {r_clk} is '
            f'incompatible with reset scope clock {node_clk}.',
        )
      if r_clk != node_clk and node.is_stateful and is_kahn:
        self._fail(
            node,
            f'stateful node has cross-activity reset on clock {r_clk} across '
            f'node clock {node_clk}.',
        )
      has_valid_reset = True
    if node.is_stateful and node_clk != BASE_CLOCK and not has_valid_reset:
      self._fail(
          node,
          f'stateful node executes on subclock {node_clk} outside a matching '
          f'activation scope (expected {BASE_CLOCK}).',
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


def check_clocks(graph: Graph) -> ClockAnalysis:
  """Infers and validates synchronous clocks across graph.

  Args:
    graph: Finalized Psyre `Graph` to validate.

  Returns:
    The validated `ClockAnalysis` for `graph`.

  Raises:
    RuntimeError: If `graph` has not been finalized.
    ClockError: If any node, port, activity, or reset violates clock rules.
  """
  return infer_clocks(graph)


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
    if not node.is_stateful:
      continue
    node_clk = resolved_clocks.node_clock(node)
    has_scope_reset = False
    for r_handle in node.resets:
      if is_one_periodic(r_handle):
        continue
      r_clk = resolved_clocks.stream_clock(r_handle)
      if not r_clk.is_prefix_of(node_clk):
        raise ClockError(
            f'reset stream on clock {r_clk} is incompatible with node clock '
            f'{node_clk}.',
            node=node,
        )
      if r_clk != node_clk:
        raise ClockError(
            f'stateful node has cross-activity reset on clock {r_clk} across '
            f'node clock {node_clk}.',
            node=node,
        )
      has_scope_reset = True
    if node_clk != BASE_CLOCK and not has_scope_reset:
      raise ClockError(
          f'stateful node executes on subclock {node_clk} outside a matching '
          f'activation scope (expected {BASE_CLOCK}).',
          node=node,
      )
  return resolved_clocks
