# `psyre.clock`

Synchronous Lustre / Colaço-Pouzet constraint-based clock calculus: clock term
representation, union-find unification, lazy selector canonicalization, and
whole-graph clock validation.

## Overview & Role

`psyre.clock` infers and validates hierarchical synchronous subclocks (`BASE_CLOCK`
`.` refined by `on <predicate>` steps) across every node, stream, subclocked
input port, reset port, and `match`/`automaton` arm prefix in a finalized
[`Graph`](graph.md).

Static clock analysis is **mandatory under `Semantics.SYNC`** (enforced by
[`Scheduler`](scheduler.md)) and also drives [`optimize_graph`](optimization.md)
(clock-compatible folding and CSE), [`builder.lower_resets`](builder.md#lower_resets)
(lowering ancestor resets, polymorphic static ports, and `KAHN` subclocked
`current()` inputs), [`psyre.visualize`](visualize.md) (subclock cluster boxes),
and [`psyre.jax.jit`](jax_jit.md) (same-clock island fusion).

## Design & Key Invariants

### Uniform Static Clock Polymorphism (`∀ 'a. 'a`)

Every unpinned, producer-less static [`Stream`](stream.md)—whether a 1-periodic
constant (`Stream(0)`) or a multi-element periodic sequence
(`Stream([1, 2, 3])`, `Stream.once(k)`)—is **clock-polymorphic**
(`Graph.is_polymorphic_static`):

- Each consumer port and reset scope instantiates a polymorphic static stream at
  its own concrete clock (`ClockAnalysis.input_clock` / `reset_clock`) without
  forcing sibling consumers onto the same clock.
- In the resulting `ClockAnalysis`, each polymorphic static stream is displayed
  with a fresh polymorphic clock variable (`'a`, `'b`, …, `'z`, `'a1`, …).
- By contrast, streams pinned via `Graph.pin_base_clock()` are anchored to
  `BASE_CLOCK` (`.`), and each unbound external dynamic `Stream()` receives a
  single shared **monomorphic** clock variable so all its consumers must agree
  on its clock.
- To anchor a static waveform explicitly to the base clock (so inactive subclock
  ticks skip elements rather than pausing the static cursor), wrap it in
  `when(Stream(True), seq)`.

### Colaço-Pouzet Constraint-Based Unification

Earlier fixed-point sweep solvers suffered from declaration-order sensitivity
across delayed `fby`/`pre` edges and fixed sweep bounds on cyclic clock
dependencies. `psyre.clock` implements Colaço-Pouzet constraint-based
unification:

- **Internal term algebra**:
  - `_BaseTerm` (`BASE_CLOCK`, `.`),
  - `_ClockVar` (mutable union-find variable with path compression), and
  - `_OnTerm(base, cond, selector, owner_id)` (lazy subclock refinement).
- **Lazy selector canonicalization**: `_OnTerm` defers selector normalization
  until decomposition so clocks inferred later justify unwrapping. Resolution
  delegates to [`Graph.resolve_clock_selector_step`](graph.md#coinductive-boolean-classification--shared-selector-unwrapping)
  with strict cycle guards and rate guards:
  - Never unwraps through a non-1-periodic polymorphic static payload inside a
    `when` (whose rate belongs to the `when` node).
  - Never collapses a non-1-periodic polymorphic static selector across an
    indicator `merge` or `~` unless the producer's clock is **strictly equal**
    to the refinement base clock (not merely a prefix).
  - Drops always-active / tautological steps cleanly.
- **Right-to-left unification & occurs check**: Unifying two clock terms strips
  matching `ClockStep`s from right to left (raising `ClockError` on step
  mismatch), binds a remaining `_ClockVar` root to the opposing prefix, and
  catches cyclic clock dependencies (`α = α on ...`, such as instantaneous
  self-sampling `x = when(lambda: x, 0)`) directly via an **occurs check**
  (`ClockError`).

### Four-Phase Inference, Grounding & Validation Pipeline

`_ClockAnalyzer.run()` traverses nodes in `causal_node_order(graph)`:

1. **Phase 0 (Elaboration)**: Assigns `_BASE_TERM` to pinned streams, a shared
   `_ClockVar` to each unbound dynamic `Stream()`, a `node_var` to each node,
   and a distinct output `_ClockVar` to clock-changing ops (`when`, `whennot`,
   `buffer`).
2. **Phase 1 (Constraint Generation)**:
   - Unifies `node_term` with all non-polymorphic `rate_inputs()`.
   - Defers input-free nodes with arm scopes (`node.arms` on stateful or
     input-bearing nodes; zero-input stateless nodes such as `random_stream()`
     stay on `BASE_CLOCK`) or reset-clock inheritance (`pre`, `fby`,
     `static_stream`) so they anchor to their scope clocks rather than premature
     `BASE_CLOCK`.
   - Emits output `_OnTerm` constraints for `when`, `whennot`, and `buffer`, and
     subclock port constraints for `merge` arms (`1 + idx`) and `buffer` input
     port `0`.
3. **Phase 2 (Grounding)**: Grounds any remaining free `_ClockVar`s in priority
   order: enclosing scope defaults (`arms` / `inherits_reset_clock` resets),
   deepest `current` consumers (for unbound `i_stream`), and finally
   `BASE_CLOCK`.
4. **Phase 3 (Materialization & Validation)**:
   - Assigns polymorphic names (`'a`, `'b`, …) to polymorphic static streams and
     materializes all concrete `Clock`s.
   - Verifies all `rate_inputs()` match `node_clk` and all `merge`/`buffer`
     subclock ports match their expected subclocks, recording per-port clocks
     for [`builder.lower_resets`](builder.md#lower_resets).
   - **Prefix/ancestor `current` contract**: Validates
     `node_clk.is_prefix_of(i_clk)` for dynamic `current(b, i, default)` inputs
     (allowing same-clock or arbitrarily nested subclocks while rejecting faster
     or sibling clocks with `ClockError`).
   - **Reset prefix & subclocked-state contract**: Validates
     `r_clk.is_prefix_of(node_clk)` for every reset port, and rejects any
     subclocked stateful node (`node_clk != BASE_CLOCK`) with no reset carrier
     (`ClockError`).

## Public API Reference

All symbols below are re-exported from top-level `psyre`.

### `BASE_CLOCK`

```python
BASE_CLOCK: Clock = Clock()
```

Concrete root base clock (formatted as `'.'`). Note that `bool(Clock)` is
always `True` even though `len(BASE_CLOCK) == 0`.

### `ClockStep`

```python
@dataclasses.dataclass(frozen=True)
class ClockStep:
    selector: Stream            # compare=False
    values: frozenset[TagKey]
    negated: bool = False
    is_boolean: bool = False
    selector_label: str = ''    # compare=False
    selector_id: int            # init=False, repr=False (used for eq/hash)
```

Represents one `on <predicate>` clock refinement step. Equality and hashing key
on `(selector_id, values, negated, is_boolean)` because `Stream.__hash__` is
disabled.

- `to_when_selector() -> WhenSelector`: Converts the step back into an
  executable [`WhenSelector`](common.md#thunks-tags--validation) for `lower_resets`.
- `__str__() -> str`: Formats as `on c`, `on not c`, `on (s == v)`,
  `on (s != v)`, `on (s in {...})`, or `on (s not in {...})`.

### `Clock`

```python
@dataclasses.dataclass(frozen=True)
class Clock:
    steps: tuple[ClockStep, ...] = ()
    var: str | None = None
```

Immutable synchronous clock (`.`, `'a`, or `<root> on <step1> on ...`).

- `Clock.polymorphic(var: int | str = 0) -> Clock`: Builds a polymorphic clock
  variable (`'a`..`'z`, `'a1`, …; auto-prefixes `'` on strings; rejects
  negative ints or empty names with `ValueError`).
- Properties: `is_base -> bool`, `is_polymorphic -> bool`.
- Methods: `on(step: ClockStep) -> Clock`, `is_prefix_of(target: Clock) -> bool`,
  `__len__() -> int`, `__iter__() -> Iterator[ClockStep]`, `__bool__() -> True`.

### `ClockError`

```python
class ClockError(RuntimeError):
    node: Node | None

    def __init__(self, message: str, *, node: Node | None = None) -> None
```

Raised when a finalized `Graph` violates synchronous clock unification, occurs
check, `current` prefix, or reset rules.

### `ClockAnalysis`

Immutable query table returned by `check_clocks(graph)`:

- Property: `graph -> Graph`.
- `stream_clock(stream: Stream) -> Clock`: Resolves stream aliases; raises
  `ValueError` if `stream` does not belong to `graph`.
- `node_clock(node: Node) -> Clock`: Validates node ownership.
- `input_clock(node: Node, port_index: int) -> Clock`: Returns the expected
  clock of `node.inputs[port_index]` (differing from `node_clock` on `merge`
  arms, `buffer` port `0`, and strictly subclocked `current` port `1`).
- `reset_clock(node: Node, reset_index: int) -> Clock`: Returns the clock of
  `node.resets[reset_index]`.
- `arm_prefix_clocks(node: Node) -> tuple[Clock, ...]`: Selector evaluation
  clock for each enclosing `ArmScope` on `node`.
- `node_activity_clocks(node: Node) -> tuple[Clock, ...]`: All prefix clocks
  from root down to `node_clock(node)`.

### `check_clocks`

```python
def check_clocks(graph: Graph) -> ClockAnalysis
```

Infers and validates synchronous clocks across `graph`. Raises `RuntimeError` if
`graph` is not finalized and `ClockError` on any clock violation.

## See Also

- [`psyre.graph`](graph.md) — `resolve_clock_selector_step`, `is_polymorphic_static`, `pin_base_clock`
- [`psyre.builder`](builder.md#lower_resets) — `lower_resets(graph, clocks)`
- [`psyre.optimization`](optimization.md) — Clock-compatible folding and CSE
