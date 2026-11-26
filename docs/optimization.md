# `psyre.optimization`

Fixed-point Stage-1 whole-program optimizer performing clock-preserving and
EOF-preserving algebraic/boolean folding, mux simplification, cache-free pure
combinational CSE, and backward-reachability dead-code elimination.

## Overview & Role

`psyre.optimization` shrinks and simplifies finalized [`Graph`](graph.md) IRs in
place (invoked automatically by [`build_graph(..., optimize=True)`](builder.md)
and directly via `optimize_graph` / `eliminate_dead_code`). Optimization is
strictly **semantics-preserving** across both `Semantics.SYNC` and
`Semantics.KAHN`: it preserves exact clocks, `Absent` carriers, runtime
`KeyError` behavior on non-exhaustive selectors, finite-horizon `post` EOF
truncation (`Bottom` padding), user-requested stream names, and diagnostics on
invalid graphs.

## Design & Key Invariants

### Fixed-Point Pipeline & Invalid-Graph Preservation

`optimize_graph(graph, extra_roots=...)` runs:

1. Initial `eliminate_dead_code(graph, extra_roots=...)`.
2. Snapshots `protected_nodes` (any explicit `Node` in `extra_roots` is never
   folded away, though later equivalent nodes may still CSE onto it) and
   `named_roots` (preserving requested `str` aliases even when they match a
   default `prod.op` leaf name).
3. **Diagnostic safety**: Immediately returns `graph` unchanged if
   [`find_instantaneous_cycle(graph)`](analysis.md) is non-`None` or
   [`check_clocks(graph)`](clock.md) raises `ClockError`, ensuring causality and
   clock errors report against the unmodified graph.
4. Iterates `check_clocks` → single-pass simplification + CSE → named-root alias
   re-attachment → `eliminate_dead_code` until a fixed point is reached.

### Backward-Reachability Dead-Code Elimination

`find_live_nodes` performs a backward BFS from
`graph.collect_root_streams(extra_roots)` across both `node.input_streams()`
(data inputs + resets) **and** `node.arms` condition streams. Retaining
`ArmScope.cond` ensures that probing an arm-local all-static stream via
`extra_roots` keeps its enclosing clock selectors alive. `eliminate_dead_code`
compacts the graph in place via `graph.retain_nodes()`, preserving instantiation
order.

### Soundness Guards: Clock Compatibility, `post` EOF & Tag Domains

Three static invariants guard every rewrite:

- **Clock compatibility (`_clock_compatible`)**: Replacing a node with a
  concrete stream requires exact clock equality (`clocks.stream_clock(repl) ==
  out_clk`). Replacing a node with an unpinned polymorphic static stream is
  permitted **only on `BASE_CLOCK`** (`.`), because polymorphic statics carry no
  `Absent` markers (`SYNC`) or finite-horizon subclock token bounds (`KAHN`).
- **`post` early-EOF (`Bottom`) preservation**: Computes `eof_ids`, the forward
  transitive cone of all `post` nodes (zero overhead when no `post` exists).
  Simplifications never drop an eagerly pulled input in `eof_ids` unless the
  replacement preserves that truncation (note: `merge` only pulls selector
  `inputs[0]` eagerly, while `switch` and binary ops pull all inputs eagerly).
- **Finite-domain tag analysis (`_selector_tag_domain`)**: Abstractly computes a
  finite upper bound (capped at 256 distinct values) on discrete tags a selector
  can emit (using [`payload.exact_value_key`](payload.md) to keep `bool`/`int`/
  `float` and `-0.0`/`+0.0` distinct) across static streams, boolean ops, `pre`,
  and `Node.value_forward_args()`. Folding a non-boolean, default-less `merge`/
  `switch` requires provable exhaustiveness across all nested muxes so runtime
  `KeyError` exceptions are never masked.

### Algebraic, Boolean & Mux Simplifications

All rewrites apply exclusively to stateless nodes verified with
`node.is_primitive(...)` or `node.is_stream_op(...)` (never matching user
functions by string name):

- **`ifthenelse`**: Constant selector → selected branch; identical branches
  (`same_stream_or_const(t, f)`) → `t`; `ifthenelse(bool_c, True, False)` → `c`.
- **`when` / `whennot`**: Normalizes single-boolean-tag `WhenSelector`s on
  boolean selectors; folds always-active selectors (`is_always_active_when`,
  such as `when(Stream(True), x)` or `when(Stream('A'), x, values='A')`) to
  `data_h`.
- **`merge` / `switch`**:
  - Constant 1-periodic selector → handled arm.
  - Exhaustive identity mux (every unwrapped arm yields `sel` or its exact-type
    tag constant) → `sel`.
  - Exhaustive uniform mux (all unwrapped arms pairwise `same_stream_or_const`)
    → first arm. Never unwraps bare multi-element polymorphic static `merge`
    arms (which advance independent per-arm cursors).
  - Non-boolean `merge`/`switch` over a boolean selector (`is_bool_stream`) →
    normalized to `MergeSelector(is_boolean=True)` (guarding single-arm
    duplication and `switch` `post`-EOF arms).
- **Boolean ops (`&`, `|`, `~`)**: Gated on `is_bool_stream` for all operands;
  folds constant `~c` (on `BASE_CLOCK`), double negation `~(~x) -> x`,
  identity (`True` for `&`, `False` for `|`), absorption (`False` for `&`,
  `True` for `|`) symmetrically in operand order, and idempotence (`x & x -> x`).

### Cache-Free Pure Combinational CSE

Pure stateless nodes—builtin `when`, `whennot`, `merge`, `switch` and any node
where `common.is_pure_step_func(node.step_func)` is `True` (including built-in
`Stream` ops, `psyre.math.{exp,tanh}`, `@psyre_comb(pure=True)`, and pure
NumPy/JAX lifted ops)—are deduplicated when:

1. They share the same hierarchy prefix names (`node.prefixes`), so CSE never
   merges across distinct `@psyre_node` / `automaton` scopes.
2. They share the same operator identity via
   [`common.same_step_func(a.step_func, b.step_func)`](common.md#node-decorators--cache-free-structural-cse-same_step_func)
   (with `when`/`whennot` unified under `normalize_when_selector` and
   `merge`/`switch` compared structurally over `MergeSelector` tag-to-arm
   bindings).
3. All inputs match via `same_stream_or_const` and static kwargs /
   `CombCallPlan`s match strictly (rejecting `NaN`, distinct callables, and
   dense array objects).
4. Clocks match strictly (`node_clock(cand) == node_clock(node)` and
   `_clock_compatible`).

## Public API Reference

All four functions are re-exported from top-level `psyre`.

### `find_live_nodes`

```python
def find_live_nodes(
    graph: Graph,
    *,
    extra_roots: Iterable[Any] | None = None,
) -> tuple[Node, ...]
```

Returns all `Node`s in instantiation order transitively feeding `graph.output`
or `extra_roots` (accepting `Stream`, `Node`, or stream-name `str` entries;
rejecting bare `str` with `TypeError`).

### `find_dead_nodes`

```python
def find_dead_nodes(
    graph: Graph,
    *,
    extra_roots: Iterable[Any] | None = None,
) -> tuple[Node, ...]
```

Returns all `Node`s in `graph` not included in `find_live_nodes(...)`.

### `eliminate_dead_code`

```python
def eliminate_dead_code(
    graph: Graph,
    *,
    extra_roots: Iterable[Any] | None = None,
) -> Graph
```

Prunes dead `Node`s from `graph` in place via `graph.retain_nodes()` and returns
`graph`.

### `optimize_graph`

```python
def optimize_graph(
    graph: Graph,
    *,
    extra_roots: Iterable[Any] | None = None,
) -> Graph
```

Runs fixed-point algebraic/boolean/mux folding, pure combinational CSE, and DCE
in place on a finalized `Graph` and returns `graph`.

## See Also

- [`psyre.clock`](clock.md) — Clock checks gating every rewrite
- [`psyre.common`](common.md#node-decorators--cache-free-structural-cse-same_step_func) — `same_step_func` and `is_pure_step_func`
- [`psyre.builder`](builder.md) — `build_graph(..., optimize=True)`
