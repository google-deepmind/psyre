# `psyre.automaton`

Hierarchical synchronous mode automata (`automaton`, `state`, `until`, `unless`)
with strong and weak guarded transitions, reset vs. history re-entry, and
inter-state `last()` memory.

## Overview & Role

`psyre.automaton` implements Lucid Synchrone / SCADE-style hierarchical mode
automata (Maraninchi & Rémond, 2003; Colaço, Pagano & Pouzet, 2005) as pure
Stage-1 lowering onto core Psyre primitives (`fby`, `switch`, `merge`, `when`,
and `reset`). Because automata lower completely to primitive dataflow equations
and [`builder.lower_resets`](builder.md#lower_resets) compiles modular resets
for both semantics, automata execute identically (up to `Absent`/`Bottom`
elision) under both `Semantics.SYNC` and `Semantics.KAHN`.

```python
from psyre import automaton as pauto, match as pmatch

with pauto.automaton(init='IDLE', name='ctrl') as a:
  with pauto.state('IDLE', pauto.unless(go, 'RUN', reset=True)):
    a.out = 0
  with pauto.state('RUN'):
    a.out = pmatch.last(lambda: a.out, 0) + 1
    pauto.until(done, 'IDLE', reset=False)
result = a.out
```

## Design & Key Invariants

### Two-Phase Strong/Weak Transition Lowering

Every `automaton(init=..., name=...)` block pushes a hierarchy prefix
(`name` or `fsm_<id>`) and compiles to the standard Colaço-Pagano-Pouzet
two-phase Moore/Mealy encoding:

1. **State & reset registers on the enclosing clock**:
   - `pns = fby(init_state, lambda: ctx.ns_stream, name='pns')` (previous
     next-state register; defaults `init_state` to the first declared `state()`
     tag if `init` is omitted).
   - `pnr = fby(False, lambda: ctx.nr_stream, name='pnr')` (previous weak-reset
     register; omitted and folded to constant `False` when no `until()` clause
     can reset).
2. **Strong transitions (`unless`, evaluated *before* the current state step)**:
   - Strong `unless(cond, target, reset=...)` clauses are passed as arguments to
     `state(tag, unless(...), ...)` and evaluated on the **enclosing clock** via
     same-clock `switch(pns, ..., lookup=MergeSelector(is_boolean=False, ...))`
     to produce the active-tick state `s_stream` (`state_s`) and reset
     `r_stream` (`state_r`).
   - Using an explicit non-boolean `MergeSelector` keyed by `tag_key(tag)`
     ensures `0`/`False` and `1`/`True` state tags never collide.
   - When no `unless()` clauses exist, `s_stream = pns` and `r_stream = pnr`
     with zero extra `switch` nodes.
3. **State bodies & scoped reset**:
   - Each `with state(tag, ...):` pushes both `ArmScope(selector=...)` (clocked
     on `s_stream == tag`) and `push_reset(lambda: ctx.r_stream)`, so stateful
     nodes and non-1-periodic static streams inside the state reset cleanly on
     `reset=True` transitions and resume retained state on `reset=False`
     (history) transitions.
   - Shared inter-state memory uses [`match.last(...)`](match.md#shared-inter-arm--lowering-helpers) on the
     enclosing clock.
4. **Weak transitions (`until`, evaluated *after* the current state step)**:
   - Registered inside a `state()` body via `until(cond, target, reset=...)`,
     folded inside the state's arm and reset scope, and merged on `s_stream` via
     [`match.merge_arm_outputs`](match.md#shared-inter-arm--lowering-helpers) to
     produce `ns_stream` (`state_ns`) and `nr_stream` (`state_nr`).
   - When no `until()` clauses exist, `ns_stream = s_stream` directly (skipping
     `state_ns` / `state_nr` nodes).
5. **Named output merging**: Finalized via `match.finalize_named_outputs(ctx,
   s_stream, arms)`, enforcing identical output names and matching tuple shapes
   across all states.

### Constant Folding, Priority & Scope Preservation

- **Prioritized evaluation (`_fold_transitions`)**: Clauses are evaluated in
  declaration order; the first truthy condition wins. Constant-`False` guards
  are elided, constant-`True` guards short-circuit remaining clauses (while
  still eagerly validating unreachable targets and reset values), and identical
  `then`/`else` branches or boolean-reset conditions (`ifthenelse(c, True,
  False) -> c`, `ifthenelse(c, False, True) -> ~c`) fold immediately.
- **Captured environments on unconditional transitions**: Every `_Transition`
  captures its `CapturedEnv`. Even when a guard is constant `True`, target and
  reset thunks (and non-1-periodic literal lists) are bound to `trans.env` so
  inner `with reset(...)` and `@psyre_node` prefix scopes are preserved.
- **Validation guards**:
  - Rejects non-`str` `name`, duplicate state tags, invalid/non-matching target
    or `init` tags, missing `Enum`/`bool` states (`validate_exhaustiveness`),
    array-valued transition conditions or resets (`TypeError`), `unless()`
    called inside a `state()` body, unconsumed `unless()` clauses, or `unless()`
    clauses reused across states or different automata (`RuntimeError`).

## Public API Reference

The `automaton` submodule is re-exported at top level (`from psyre import
automaton`).

### `automaton`

```python
@contextlib.contextmanager
def automaton(init: Any = UNSET, name: str = '') -> Iterator[BlockHandle]
```

Opens a stateful hierarchical automaton block (pushing prefix `name` or
`fsm_<id>`) and yields a `BlockHandle` for binding per-state outputs and reading
merged output streams after the block exits. Requires at least one `state()`
(`ValueError`).

### `state`

```python
@contextlib.contextmanager
def state(tag: Any, *unless_clauses: _Transition) -> Iterator[Stream]
```

Declares a discrete state inside `with automaton(...)` with optional prioritized
strong `unless(...)` transition clauses evaluated before entering the state
body. Yields the downsampled active-state `Stream`.

### `unless`

```python
def unless(cond: Any, target: Any, *, reset: Any = True) -> _Transition
```

Constructs a strong transition clause to be passed directly to
`state(tag, unless(...), ...)`. When `cond` is truthy on entry to `tag`,
immediately switches to `target` for the current tick, resetting `target`'s
local state if `reset` is truthy (or resuming history if `reset=False`).

### `until`

```python
def until(cond: Any, target: Any, *, reset: Any = True) -> None
```

Registers a weak transition inside the active `state()` body. Evaluated during
the current tick's state execution; if `cond` is truthy, transitions to `target`
on the *next* enclosing tick (resetting or resuming according to `reset`).

## See Also

- [`psyre.match`](match.md) — Shared `last()`, exhaustiveness checks, arm confinement, and lowering
- [`psyre.builder`](builder.md#lower_resets) — Compilation of modular resets across `SYNC` and `KAHN`
- [`psyre.visualize`](visualize.md) — Collapsible automaton cluster visualization
