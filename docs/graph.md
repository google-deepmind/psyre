# `psyre.graph`

Pure, `ContextVar`-free Stage-1 dataflow intermediate representation (`Node`
and `Graph`), single-assignment connectivity, union-find stream alias
compression, and shared IR query/rewriting utilities.

## Overview & Role

`psyre.graph` is Psyre's standalone whole-program IR decoupling Stage-1
construction ([`psyre.builder`](builder.md)) from static analyses
([`psyre.analysis`](analysis.md), [`psyre.clock`](clock.md)), optimization
([`psyre.optimization`](optimization.md)), visualization
([`psyre.visualize`](visualize.md)), transformations
([`psyre.jax.transforms`](jax_transforms.md), [`psyre.jax.jit`](jax_jit.md)),
and Stage-2 execution ([`psyre.scheduler`](scheduler.md)).

It holds **zero global `ContextVar` state**: every `Graph` is a self-contained,
inspectable dataflow DAG (with explicit delayed feedback edges) that can be
cloned, analyzed, optimized, visualized, and executed repeatedly.

## Design & Key Invariants

### Single-Assignment & Two-Phase Lifecycle

- **Single assignment**: Every dynamic `Stream` has at most one producer `Node`
  (`_producer[stream.id]`). Static `Stream`s cannot be node outputs
  (`TypeError`).
- **Contiguous node IDs**: `Node.id` runs contiguously `0 .. len(graph.nodes) - 1`;
  `retain_nodes()` renumbers surviving nodes contiguously in instantiation
  order.
- **Unresolved → resolved → finalized**:
  1. During Stage-1 elaboration, `Node`s start unresolved (`_resolved=False`,
     `inputs=()`, `resets=()`). Querying `input_streams()`, `rate_inputs()`,
     `instantaneous_input_streams()`, or `delayed_input_streams()` before
     resolution raises `RuntimeError`.
  2. `resolve()` / `rewire_node()` binds resolved `Stream` inputs and resets
     (forcing `resets=()` on stateless nodes) and maintains consumer adjacency.
  3. `Graph.mark_finalized()` verifies every node is resolved (`RuntimeError`
     otherwise). Mutating `add_node()`/`set_output()` is forbidden after
     finalization; analysis, cloning, and rewriting APIs require
     `require_finalized()`.

### Union-Find Stream Alias Compression

When [`optimize_graph`](optimization.md) folds or deduplicates a node via
`Graph.replace_stream(old_stream, new_stream)`:

- All consumer `inputs` and `resets` are rewired, `graph.output` PyTree leaves
  are remapped, explicit user stream names (`is_explicit_stream_name`) are
  copied to `new_stream`, and `_stream_aliases[old_stream.id] = new_s` is
  recorded in `O(1)` time (replacing an already-aliased `old_stream` raises
  `ValueError`).
- `resolve_stream(stream)` follows aliases with 0-hop and 1-hop fast paths,
  cycle protection, and **union-find path compression**, keeping repeated
  resolutions amortized `O(1)`.
- `retain_nodes()` prunes dead nodes and transitively cleans dangling alias
  targets.
- Every structural mutation (`pin_base_clock`, `add_node`, `rewire_node`,
  `replace_stream`, `retain_nodes`) clears `_lowered_semantics` so
  [`builder.lower_resets`](builder.md#lower_resets) idempotence checks never
  serve stale state.

### Builtin Identity & Port Classification

All semantic queries use `Node.is_primitive(*ops)` (verifying both `op` and
registered builtin `step_func` identity) rather than raw `node.op` string
matching, preventing collisions with user `@psyre_comb`/`@psyre_sequ` functions
named `merge`, `when`, `fby`, `ifthenelse`, etc.:

- **Instantaneous vs. delayed edges**: `pre` treats `inputs[0]` as delayed
  (`instantaneous=()`, `delayed=(inputs[0],)`); `fby` treats `inputs[0]` (init)
  as instantaneous and `inputs[1:]` as delayed; all other primitives (including
  `buffer`) and all `resets` are instantaneous.
- **Rate-determining inputs (`rate_inputs`)**: `merge` → `(inputs[0],)`
  (selector only); `buffer` → `inputs[1:3]` (`pull_p`, `push_p`); `current` →
  `(inputs[0], inputs[2])` (`b`, `default`, excluding subclocked `i`); all
  others → `inputs`.
- **Value-forwarding ports (`value_forward_args`)**: Single source of truth for
  `when`/`whennot`, `post`, `fby`/`morph`, `current`/`ifthenelse`, and
  `merge`/`switch` shared by `is_bool_stream`, scalar inference, and
  `optimization._selector_tag_domain`.
- **Reset-clock inheritance (`inherits_reset_clock`)**: `True` only for stateful
  `pre`, `fby`, and `static_stream`, enabling manual subclocked resets and
  synthetic latches while keeping bare subclock resets on `counter`/`morph`/
  `prefix_sum`/`scan` checked against matching arms.

### Coinductive Boolean Classification & Shared Selector Unwrapping

- **`is_bool_stream(stream, seen=None, *, args_fn=None)`**: Greatest-fixed-point
  cycle-guarded traversal (`seen` is sound because all recursive checks combine
  with `all(...)`). Rich comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) are
  classified as boolean **only** when both operands are proven built-in Python
  scalars (preventing unsound folds like `(arr > 0) & False -> False` on NumPy
  or JAX arrays).
- **`resolve_clock_selector_step(...)`**: Shared canonicalizer used by both
  [`psyre.clock`](clock.md) and [`psyre.match`](match.md). Iteratively unwraps
  `when`/`whennot` chains, folds 1-periodic boolean constants, flips expectations
  across boolean `~`, and unwraps constant-boolean indicator `merge` nodes under
  caller-supplied rate-prefix guards (`can_unwrap_when`, `can_unwrap_node`,
  `can_unwrap_poly`).

### Cloning Utilities

- `clone_unfinalized()`: Shallow-copies nodes and metadata while sharing
  `Stream` instances (used by `lower_resets` to preserve stream identity).
- `clone_streams(exclude_ids=...)`: Allocates fresh detached `Stream` clones
  for non-excluded dynamic streams while sharing static streams and preserving
  aliases and base-clock pins; paired with `remap_arms()` (preserving `ArmScope`
  object identity) and `copy_node_into()` for non-destructive whole-graph
  rewrites (`rewrite_dtype`, `fuse_jit_islands`).

## Public API Reference

### Output PyTree Helpers

- `map_output_leaves(fn: Callable[[Stream], Any], value: Any) -> Any`: Applies
  `fn` across `None`, `Stream`, `tuple` (preserving `NamedTuple` types), `list`,
  and `Mapping` outputs (`TypeError` on unsupported containers/values).
- `iter_output_leaves(value: Any) -> Iterator[Stream]`: Yields all `Stream`
  leaves of a validated output PyTree.

### `Node`

Re-exported from top-level `psyre`.

- **Attributes**: `id: int`, `op: str`, `name: str`, `output: Stream`,
  `step_func: Callable`, `kwargs: dict[str, Any]`, `inputs: tuple[Stream, ...]`,
  `resets: tuple[Stream, ...]`, `is_stateful: bool`,
  `arms: tuple[ArmScope, ...]`, `prefixes: tuple[tuple[str, int], ...]`,
  `merge_arms: tuple[tuple[ArmScope, ...], ...] | None`.
- **Properties & methods**: `is_resolved`, `resolve(*, inputs, resets=())`,
  `input_streams()`, `rate_inputs()`, `is_primitive(*ops)`,
  `is_stream_op(*ops)`, `is_static_stream()`, `inherits_reset_clock()`,
  `arms_at_reset(reset_index)`, `prefixes_at_reset(reset_index)`,
  `value_forward_args(args=None)`, `instantaneous_input_streams()`,
  `delayed_input_streams()`.

### `Graph`

Re-exported from top-level `psyre`.

- **Attribute & properties**: `semantics: Semantics | None`, `output`,
  `nodes: tuple[Node, ...]`, `streams: tuple[Stream, ...]`, `is_finalized`.
- **Construction & mutation**: `add_stream(stream)`, `add_node(...)`,
  `rewire_node(node, *, inputs, resets=())`, `set_output(output)`,
  `mark_finalized()`, `pin_base_clock(stream)`,
  `mark_resets_lowered(semantics)`, `add_stream_name(stream, name)`,
  `replace_stream(old_stream, new_stream)`, `retain_nodes(live_nodes) -> int`.
- **Cloning**: `clone_unfinalized()`, `clone_streams(*, exclude_ids=frozenset())`,
  `remap_arms(scopes, stream_map)`, `copy_node_into(node, stream_map, *, step_func=None, kwargs=None)`.
- **Queries & lookups**: `require_finalized()`, `validate_node(node)`,
  `has_stream(stream)`, `is_base_clock_pinned(stream)`,
  `is_resets_lowered_for(semantics)`, `is_polymorphic_static(stream)`,
  `stream_names(stream)`, `is_explicit_stream_name(stream, name)`,
  `explicit_stream_names()`, `resolve_stream(stream)`,
  `resolve_stream_id(stream_id)`, `resolve_named_streams(names)`,
  `collect_root_streams(extra_roots=None, *, include_output=True)`,
  `output_streams()`, `producer(stream)`, `consumers(stream)`,
  `predecessors(node, *, instantaneous_only=False)`,
  `successors(node, *, instantaneous_only=False)`.
- **Static & selector analysis**: `static_source(stream)`,
  `one_periodic_stream(stream)`, `const_bool(stream)`,
  `unwrap_when(stream, *, can_unwrap=None)`, `const_value(stream)`,
  `normalize_when_selector(...)`, `is_always_active_when(...)`,
  `resolve_clock_selector_step(...)`, `is_bool_stream(...)`.

## See Also

- [`psyre.builder`](builder.md) — Stage-1 `GraphBuilder` producing finalized `Graph`s
- [`psyre.analysis`](analysis.md), [`psyre.clock`](clock.md), [`psyre.optimization`](optimization.md) — Static passes over `Graph`
