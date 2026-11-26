# `psyre.builder`

Stage-1 graph elaboration (`GraphBuilder`, `build_graph`), deferred thunk
resolution with scope preservation, and Stage-2 pre-execution reset/subclock
lowering (`lower_resets`, `requires_clock_analysis`).

## Overview & Role

`psyre.builder` owns the boundary between user Python code (Stage 1) and the
pure [`Graph`](graph.md) IR consumed by static analyses and [`Scheduler`](scheduler.md)
(Stage 2). Importing `psyre.builder` also loads `psyre.sync_primitives` and
`psyre.kahn_primitives` so both semantics registries are populated before any
build starts.

Its two primary responsibilities are:

1. **Stage-1 elaboration (`GraphBuilder`, `build_graph`)**: Runs a user function
   under an active `active_builder` `ContextVar`, records unresolved nodes along
   with their lexical `arm_stack`, `prefix_stack`, and `reset_stack`, forces
   deferred `lambda:` / `LazyThunk` references inside their captured scopes,
   invokes `GraphBuilder.on_finalize` (arm confinement and lowering from
   [`psyre.match`](match.md)), finalizes the `Graph`, and optionally runs
   [`optimize_graph`](optimization.md).
2. **Reset & subclock lowering (`lower_resets`)**: Non-destructively compiles
   ancestor modular resets, subclocked polymorphic static ports, and (under
   `KAHN`) subclocked `current()` inputs into standard Psyre primitives so
   Stage-2 [`Scheduler`](scheduler.md) execution remains completely clock-free.

## Design & Key Invariants

### Deferred Thunk Resolution & `CapturedEnv` Preservation

- `GraphBuilder.add_loop_node()` captures the active `arm_stack`, `prefix_stack`,
  and `reset_stack` unconditionally (even on stateless nodes!) so deferred
  thunks forced during `resolve_pending_nodes()` evaluate inside the exact
  enclosing `with reset(r):`, `with match(...):`, and `@psyre_node` scopes where
  the node was constructed.
- `resolve_pending_nodes()` loops dynamically until `idx >= len(graph.nodes)`
  because forcing a thunk may append new nodes. For stateful resets, each
  reset `raw_r` at `r_idx` is resolved inside only the scopes strictly enclosing
  that reset (`resets[:r_idx]`, `arms_at_reset(r_idx)`, `prefixes_at_reset(r_idx)`).
- On user non-boolean `merge`/`switch` nodes (`node.merge_arms is None`),
  `_dedup_mux_arms` deduplicates shared resolved arm streams after thunk forcing
  so `{0: lambda: w, 1: lambda: w}` shares a single `StreamPort` (skipped when
  `merge_arms is not None` so `match`/`automaton` preserve distinct per-arm
  cursors on polymorphic statics).
- **Re-entrancy guard**: `build_graph()` refuses to run inside either an active
  Stage-1 build (`active_builder` set) or an active Stage-2 `Scheduler` run
  (`RuntimeError`), and isolates user execution inside a clean empty
  `CapturedEnv().restore()`.

### Non-Destructive, Idempotent `lower_resets(graph, clocks)`

`lower_resets` validates that `graph` is finalized and matches `clocks.graph`
(`TypeError`/`ValueError`). If no lowering is required (or
`graph.is_resets_lowered_for(semantics)` is already `True`), it marks the graph
lowered and returns `graph` unchanged. Otherwise, it shallow-clones via
`graph.clone_unfinalized()` (preserving dynamic `Stream` identities) and applies
four composable lowerings:

1. **Polymorphic static input lowering**:
   - Under both `SYNC` and `KAHN`: lowers port `0` of subclocked `merge`,
     `when`, `whennot`, and `post`, predicate ports `1` and `2` of subclocked
     `buffer`, and port `0` of any subclocked stateless node whose inputs are
     all polymorphic static.
   - Under `SYNC` additionally: lowers any non-1-periodic polymorphic static
     input on `input_clock != BASE_CLOCK`.
   - Each `(src.id, target_clk)` pair is memoized into a stateful
     `static_stream` node driven by a base-clock-pinned `activation_const`
     chain (`when` down `target_clk`).
2. **Polymorphic static reset lowering**: Lowers non-1-periodic polymorphic
   static resets on subclocks (`SYNC`) and all-polymorphic-static reset tuples
   on subclocked stateful nodes (`SYNC` and `KAHN`).
3. **Ancestor reset latching (`r_clk != node_clk`)**:
   - 1-periodic polymorphic static constants (rejecting array objects with
     `TypeError`) align directly via `activation_const(target_clk, bool(val))`.
   - General ancestor resets walk each intervening `ClockStep` from `r_clk` to
     `node_clk` (memoized per `(latch, curr.id, next_clk)`), building a
     compositional 1-step reset latch (`pending = fby(False, lambda: next_pending)`,
     `held = ifthenelse(curr, True, pending)`,
     `next_pending = ifthenelse(active, False, held)`, downsampled by
     `emit_when`) so an ancestor reset pulse arriving on an inactive tick is
     held until the subclock's next active tick.
4. **KAHN subclocked `current()` lifting**: Under `Semantics.KAHN` only, when a
   `current(b, i, default)` node's dynamic `i` lies on a strict subclock
   `i_clk != node_clk`, walks intervening `ClockStep`s bottom-up nesting
   complementary `merge` nodes (`lifted_i`, `present`) and gates `b` by
   `present` so the shared `current` kernel executes on a single dense clock
   matching `SYNC` up to `Absent`/`Bottom` elision.

Finally, any `ArmScope.cond` referencing a lowered polymorphic static stream is
patched to the synthesized `static_stream` output, newly added nodes are
resolved, and the cloned `Graph` is finalized and marked lowered.

### Conservative KAHN Lowering Gate (`requires_clock_analysis`)

When running under `Semantics.KAHN` with `clock_calculus=False` (or
`clocks=None`), `Scheduler` calls `requires_clock_analysis(graph)` to detect
whether subclocked `current()` inputs, subclocked static ports, all-static
subclocked nodes, or subclocked stateful resets require `check_clocks` +
`lower_resets()`. If so, execution raises `ClockError` rather than silently
producing wrong token counts.

## Public API Reference

`GraphBuilder`, `build_graph`, and `lower_resets` are re-exported from top-level
`psyre`.

### `GraphBuilder`

```python
class GraphBuilder:
    on_finalize: Callable[[GraphBuilder], None] | None = None
    scoped_lasts: list[tuple[Stream, tuple[ArmScope, ...]]]

    def __init__(
        self,
        semantics: Semantics | None = None,
        *,
        graph: Graph | None = None,
    ) -> None
```

- Classmethods: `current() -> GraphBuilder | None`, `is_building() -> bool`.
- Properties: `graph -> Graph`, `semantics -> Semantics | None`.
- Methods:
  - `next_fsm_id() -> int`: Allocates sequential IDs for anonymous `automaton()`
    prefixes (`fsm_0`, `fsm_1`, …).
  - `add_stream(stream: Stream) -> Stream`
  - `add_loop_node(step_func, *args, _prepare=None, **kwargs) -> Stream`:
    Validates active builder, runs `_prepare`, attaches qualified prefix name,
    records pending inputs/resets, and returns the output `Stream`.
  - `add_static_stream_node(output: Stream, source: Stream, *, name: str = '') -> Stream`:
    Registers a resettable `static_stream` node backed by detached static
    `source`.
  - `is_bool_stream(stream: Stream) -> bool`: Pre-resolution boolean stream
    check inspecting pending unresolved node inputs.
  - `bind_output(output: Any) -> None`
  - `resolve_pending_nodes(start_idx: int, resolve_fn: Callable[[Any], Stream]) -> int`
  - `finalize(resolve_fn: Callable[[Any], Stream] = resolve_stream) -> Graph`

### `build_graph`

```python
def build_graph(
    main_func: Callable[..., Any],
    *args: Any,
    semantics: Semantics = Semantics.SYNC,
    optimize: bool = True,
    extra_roots: Iterable[Any] | None = None,
    **kwargs: Any,
) -> Graph
```

Executes Stage 1 to construct, finalize, and optionally optimize a `Graph` from
`main_func(*args, **kwargs)`. Defaults to `Semantics.SYNC` and `optimize=True`.

### `lower_resets`

```python
def lower_resets(graph: Graph, clocks: ClockAnalysis) -> Graph
```

Non-destructively compiles ancestor resets, subclocked polymorphic statics, and
(under `KAHN`) subclocked `current()` inputs into primitive nodes. Idempotent.

### `requires_clock_analysis`

```python
def requires_clock_analysis(graph: Graph) -> bool
```

Conservative check indicating whether a `Semantics.KAHN` graph requires static
clock analysis and `lower_resets()` before Stage-2 execution.

## See Also

- [`psyre.graph`](graph.md) — Pure `Graph` and `Node` IR
- [`psyre.match`](match.md) — `GraphBuilder.on_finalize` arm confinement & lowering
- [`psyre.scheduler`](scheduler.md) — Stage-2 execution calling `lower_resets`
