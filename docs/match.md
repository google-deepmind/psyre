# `psyre.match`

Clocked combinational pattern matching (`match`, `case`, `default`), shared
inter-arm memory (`last`), arm-confinement verification, and multi-pass arm
lowering to primitive `when`/`merge` equations.

## Overview & Role

`psyre.match` provides structured multi-way subclocking so users can write
conditional reactive branches without manually threading `when`, `whennot`, and
`merge` across every signal:

```python
from psyre import Stream, match as pmatch

with pmatch.match(mode) as m:
  with pmatch.case('A'):
    m.y = x + 1
  with pmatch.default():
    m.y = x * 2
out = m.y
```

In addition to user-facing pattern matching, `psyre.match` supplies the shared
arm-validation, `last()`, and output-merging machinery reused by
[`psyre.automaton`](automaton.md), and registers the Stage-1 finalization hook
`GraphBuilder.on_finalize` that verifies arm confinement and lowers all
`ArmScope`s into primitive `when`/`whennot`/`merge` nodes.

## Design & Key Invariants

### Memoized Selector & Exhaustiveness

- **Deferred selector memoization**: `match(selector)` wraps `selector` in a
  memoized thunk evaluated inside the enclosing `CapturedEnv`. If a raw
  non-1-periodic list literal is passed as `selector` inside `with reset(r):`,
  it emits a single resettable `static_stream` node shared across every arm's
  `when` gate and the closing `merge`.
- **Discrete tag validation (`validate_discrete_tag`)**: Rejects `Stream`,
  thunks, `Absent`/`Bottom`, callables, `NaN`, and unhashable objects. Duplicate
  or overlapping tags across `case(*tags)` arms raise `ValueError`.
- **Exhaustiveness checking (`validate_exhaustiveness`)**: When no `default()`
  wildcard arm is present and all arm tags belong to a single `Enum` subclass or
  are all `bool`/NumPy `bool_`, missing members raise `ValueError` at Stage 1.
- **Structured arm outputs (`finalize_named_outputs`)**: Every arm must assign
  the exact same set of attribute names on the yielded `BlockHandle`. Matching
  nested `tuple` / `NamedTuple` structures merge recursively leaf-by-leaf;
  mixed scalar/tuple shapes raise `ValueError`, and raw `Stream`/thunk leaves
  inside non-tuple `list`/`set`/`frozenset`/`Mapping` outputs raise `TypeError`.

### Inter-Arm Memory (`last`) & Arm Confinement

- `last(source, init=None, n=1)` emits `fby(init, source, n=n)` in the
  **enclosing** block's clock and reset environment (`ctx.enclosing_env`),
  allowing arms or automaton states to read the previous enclosing-clock value
  of a shared merged output.
- **Confinement checking (`_check_arm_confinement`)**: Runs inside
  `GraphBuilder.on_finalize` before arm lowering. Every stream produced inside
  an arm (`node.arms` non-empty) and every `last()` created inside an arm
  (`builder.scoped_lasts`) is confined to that arm scope: leaking it into
  `graph.output`, consuming it from a node outside that arm prefix, or using it
  in an outer reset raises `RuntimeError`.

### Multi-Pass Arm Lowering (`_lower_arms`)

During `GraphBuilder.on_finalize`, `_lower_arms(graph)` compiles lexical
`ArmScope` stacks into explicit `when`/`whennot` sampling nodes:

- **Pass 0 (Fixed-point `rate_inputs()` propagation)**: Iterates over initial
  nodes (up to `len + 2` passes) propagating symbolic clock-step tuples
  canonicalized via [`Graph.resolve_clock_selector_step`](graph.md#coinductive-boolean-classification--shared-selector-unwrapping)
  across `node.rate_inputs()` (including delayed `fby`/`pre` edges and
  `inherits_reset_clock()` resets). This ensures forward-thunk subclocked state
  nodes constructed before their producers already know their exact subclock
  steps before Pass 1 begins.
- **Pass 1 (Instantaneous DFS lowering)**: Visits instantaneous inputs before
  consumers, inserting memoized chained `when`/`whennot` nodes (`emit_when`)
  whenever an outer concrete stream enters a deeper arm scope (skipping
  clock-polymorphic static streams!). Places synthetic `when` nodes inside the
  narrowest shared visualization prefix, rewires `merge` arm ports against
  `node.merge_arms`, and appends a base-clock-pinned `Stream(False)` carrier
  reset downsampled onto `node.arms` for any arm-scoped stateful node lacking a
  full-depth activation reset.
- **Pass 2 (Delayed back-edges)**: Lowers delayed `pre` port `0` and `fby` port
  `1` of arm-scoped nodes once Pass 1 has recorded all producer steps.
- **Pass 3 (Nested `ArmScope` alignment)**: For nodes nested inside 2+ arms,
  replaces inner dynamic `ArmScope.cond` references with their lowered `when`
  streams so `node.arms` stays aligned with inner `merge` selectors even after
  `optimize_graph` folds outer `when(Stream(True), seq)`.
- Clears transient `node.merge_arms = None` on every node upon completion.

## Public API Reference

The `match` submodule is re-exported at top level (`from psyre import match`).

### Context Managers

#### `match`

```python
@contextlib.contextmanager
def match(selector: Any) -> Iterator[BlockHandle]
```

Opens a clocked combinational pattern-matching block keyed on `selector`,
yielding a [`BlockHandle`](common.md#stage-1-contexts-scopes--control-records)
used to bind named outputs inside each `case()`/`default()` arm and read merged
output `Stream`s after the `with` block exits. Requires at least one arm
(`ValueError`).

#### `case`

```python
@contextlib.contextmanager
def case(*tags: Any) -> Iterator[Stream]
```

Declares an arm inside `with match(...)` matching any tag in `tags` (passing no
tags creates a trailing wildcard arm). Yields the downsampled selector `Stream`
on the arm's subclock. Rejects duplicate tags, tags after a wildcard, or use
outside `match()`.

#### `default`

```python
@contextlib.contextmanager
def default() -> Iterator[Stream]
```

Trailing wildcard arm shorthand equivalent to `case()`.

### Shared Inter-Arm & Lowering Helpers

- `last(source: Any, init: Any = None, n: int = 1, *, name: str = 'last') -> Stream`:
  Returns the `n`-th previous enclosing-clock value of `source` (raises
  `RuntimeError` outside `match()`/`automaton()`, `ValueError`/`TypeError` if
  `n < 1`).
- `validate_discrete_tag(tag: Any, caller: str) -> None`
- `validate_exhaustiveness(arms: Sequence[MatchArm], *, construct: str) -> None`
- `make_arm_selector(arm: MatchArm, excluded_for_wildcard: tuple[Any, ...] = ()) -> WhenSelector`
- `bind_thunk_env(val: Any, env: CapturedEnv) -> Callable[[], Stream]`
- `merge_arm_outputs(selector_stream, arms, arm_values, name, arm_scopes, arm_envs) -> Stream`
- `finalize_named_outputs(ctx: BlockContext, selector_stream: Any, arms: Sequence[MatchArm]) -> None`

## See Also

- [`psyre.automaton`](automaton.md) — Hierarchical mode automata built on `psyre.match`
- [`psyre.clock`](clock.md) — Subclock inference over lowered `when`/`merge` graphs
- [`psyre.builder`](builder.md) — Stage-1 `GraphBuilder.on_finalize` lifecycle
