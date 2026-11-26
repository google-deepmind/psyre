# `psyre.jax.transforms` *(Optional)*

Lifting-time Stage-1 JAX transformations—without requiring `Jaxpr` tracing or
upfront static shapes—covering single-op and composite-subgraph `vmap`,
instantaneous `grad` / `value_and_grad`, JAX-tagged stream batching, and
whole-graph mixed-precision `rewrite_dtype`.

## Overview & Role

`psyre.jax.transforms` brings JAX-style functional transformations to Psyre
programs while preserving Psyre's two-stage architecture, dynamic shapes,
subclocks, resets, and automata. Every public symbol is re-exported from
[`psyre.jax`](jax.md).

## Design & Key Invariants

### Lifting-Time `vmap` (Dual-Path Vectorization)

`vmap(fn, in_axes=0, out_axes=0, ...)` operates in two modes depending on
whether `fn` is a single lifted combinational operator (`_psyre_inst_func`
present) or an arbitrary composite Psyre function:

#### Path A — Single Lifted Combinational Op

Commutes `jax.vmap` directly with `@psyre_comb`, `psyre.jax.*`, `lax.*`,
`grad`, `jit`, or a nested single-op `vmap`:

- Preserves `_psyre_static_argnums`, `_psyre_static_argnames`, `_psyre_pure`,
  and `nout_spec`.
- Zero dynamic leaves execute eagerly via `jax.vmap` (keeping `static_argnums`
  out of `jax.vmap`).
- Supports nested PyTree `in_axes` and `out_axes` via
  `payload.map_tree_with_axes`, promoting integer-axis static literals to
  `Stream.constant` and emitting a single pure JAX-tagged node.

#### Path B — Composite Stage-1 Psyre Subgraph

Vectorizes arbitrary stateful, cyclic, and clocked Psyre subgraphs (`fby`,
`pre`, `morph`, `current`, `post`, `prefix_sum`, `scan`, `match`, `automaton`,
`reset`, `psyre.equations`, nested `vmap(vmap(...))`, and `axis_name`
collectives) **without** tracing through JAX:

1. **Isolated capture**: Elaborates `fn` into a temporary `GraphBuilder`
   (under an empty `CapturedEnv`) with placeholder streams for mapped/unmapped
   leaves, rejecting dynamic streams inside `kwargs` (`BatchError`).
2. **Monotone fixed-point `batch_axis` inference**: Propagates a finite lattice
   per stream (`None` unbatched → `k` mapped axis → `0` canonicalized axis on
   mixed/distinct merges), guaranteeing convergence across cyclic `fby`/`pre`
   loops.
   - **Clock/control uniformity contract** (raises `BatchError`, directing
     callers to [`batch_streams`](#stack_streams-unstack_stream--batch_streams)
     for heterogeneous lanes): all `node.resets`, `node.arms[*].cond`, `when` /
     `whennot` / `merge` selectors, `buffer` push/pull predicates, `current`
     hold predicates, all `counter` ports, and any non-primitive stateful node
     (e.g. `@psyre_sequ`) with batched inputs must remain unbatched (`None`).
   - **Elementwise dataflow control**: Batched `ifthenelse` lowers to
     `jax.vmap(jnp.where)` over PyTrees; batched boolean `switch` lowers to
     `jax.vmap(jnp.where)`; batched dense non-negative integer `switch`
     (`0..max_tag`, or with `default_index`) lowers to `jax.vmap(jax.lax.select_n)`
     (non-dense/string/negative tags raise `BatchError`).
   - **Collectives**: When `axis_name` is set, the 13 `jax.lax` collective ops
     (`all_gather`, `all_to_all`, `axis_index`, `axis_size`, `pgather`, `pmax`,
     `pmean`, `pmin`, `ppermute`, `pshuffle`, `psum`, `psum_scatter`,
     `pswapaxes`) force output axis `0` even with unbatched inputs.
3. **Subgraph replay**: Runs `check_clocks(sub_graph)` (allowing unclocked
   `KAHN` graphs), pre-allocates replayed output streams so cyclic `fby`/`pre`
   edges connect directly, remaps `ArmScope`s and reset/prefix depths into the
   enclosing builder, inserts clock-aligned `vmap_broadcast` / `moveaxis`
   adapters where axes differ, rewrites batched `prefix_sum` into
   `fby` + `jax.vmap(operator.add)`, wraps `scan`'s `bin_op` in `jax.vmap`,
   fixes `merge_arms` for arm confinement, and aligns outputs to `out_axes`
   (rejecting `out_axes=None` on a batched stream with `BatchError`).

### Instantaneous Reverse-Mode AD (`grad` & `value_and_grad`)

`grad` and `value_and_grad` lift `jax.grad` and `jax.value_and_grad` onto
**instantaneous pure JAX functions** (single-tick combinational differentiation,
not stateful/temporal BPTT across Psyre `fby` cycles):

- **Validation**: Rejects non-callables, `LazyThunk`, impure or stateful Psyre
  wrappers, host `numpy` functions, and invalid `argnums` (e.g., `bool` or
  `str`) with `TypeError`. Unwraps `@psyre_comb` / `psyre.jax` / `lax` / `jit` /
  single-op `vmap` / nested `grad` via `unwrap_inst_func`.
- **Stage-1 isolation**: Clears `active_builder` temporarily while `jax.grad`
  traces `raw_fn`, ensuring `fun` cannot mutate an enclosing graph and fails
  fast (`RuntimeError`) if it attempts to invoke stateful/clocked Psyre
  primitives.
- **Zero-stream eager path & PyTree stream unpacking**:
  - Zero dynamic inputs evaluate `jax.grad` / `jax.value_and_grad` eagerly.
  - With dynamic inputs, flattens the differentiated argument(s) via
    `jax.tree_util.tree_flatten(..., is_leaf=is_dynamic_leaf)` and unpacks the
    resulting gradients into a matching PyTree of `Stream`s (while keeping a
    single `Stream` carrying a runtime dict/PyTree payload intact as one leaf,
    and keeping `aux` under `has_aux=True` as a single `Stream`).
  - Caches JAX-tagged pure step functions per `(treedef, n_leaves,
    target_is_leaf)` so `optimize_graph` CSE deduplicates repeated gradient
    nodes.
  - Custom PyTree targets must be registered with both `jax.tree_util` and
    [`register_jax_pytree`](jax.md#pytree-bridging-register_jax_pytree).
- **Composability**: Composes cleanly with higher-order `grad(grad(f))`,
  `vmap(grad(f))`, `jit(grad(f))`, `optimize_graph`, and `fuse_jit_islands`.

### Non-Destructive Mixed-Precision Casting (`rewrite_dtype`)

`rewrite_dtype` clones a finalized `Graph` via `graph.clone_streams()` (leaving
the original untouched) and wraps matching stateless combinational nodes to cast
floating inputs to `target_dtype`:

- Matches pure/impure JAX-tagged nodes (and optionally built-in `Stream` ops
  when `include_stream_ops=True`), filtered by `ops` (node op names) and/or
  `primitives` (JAX `Primitive` objects or `.name` strings).
- Casts all floating arrays/scalars and Python `float`s when `source_dtype is
  None`, or only exact `jnp.dtype(source_dtype)` matches when set (skipping
  `'dtype'` and `'preferred_element_type'` kwargs).
- Optionally injects `preferred_element_type` on `dot_general`,
  `conv_general_dilated`, and `matmul` nodes when not already specified.
- Preserves `pure`, `_psyre_jax_backend`, and `_psyre_jax_primitive` metadata.

## Public API Reference

All symbols below are re-exported from `psyre.jax`.

### `BatchError`

```python
class BatchError(ValueError)
```

Raised when `psyre.jax.vmap` encounters an unbatchable control/clock/stateful
construct or invalid `in_axes`/`out_axes`, or when `psyre.jax.batch_streams`
receives mismatched `in_axes`.

### `stack_streams`, `unstack_stream`, & `batch_streams`

```python
def stack_streams(*streams: Any, axis: int = 0, name: str = '') -> Stream
def unstack_stream(stream: Any, size: int, *, axis: int = 0, name: str = '') -> tuple[Stream, ...]
def batch_streams(
    fn: Callable[..., Any],
    *streams: Any,
    size: int,
    in_axes: int | None | Sequence[int | None] = 0,
    out_axis: int | None = 0,
    name: str = '',
) -> Any
```

JAX-tagged equivalents of [`psyre.numpy`](numpy.md#stack_streams--unstack_stream)'s
batching helpers. Because emitted nodes carry `_psyre_jax_backend = True`, they
fuse directly into [`fuse_jit_islands`](jax_jit.md). `batch_streams` expands
heterogeneous per-lane subgraphs structurally at Stage 1 and raises `BatchError`
on invalid axes.

### `vmap`

```python
def vmap(
    fn: Callable[..., Any],
    in_axes: Any = 0,
    out_axes: Any = 0,
    *,
    axis_name: Any = None,
    axis_size: int | None = None,
    spmd_axis_name: Any = None,
    name: str = '',
) -> Callable[..., Any]
```

Lifting-time vectorization over a single lifted JAX operator or a composite
stateful/clocked Stage-1 Psyre subgraph.

### `grad` & `value_and_grad`

```python
def grad(
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    *,
    name: str = '',
) -> Callable[..., Any]

def value_and_grad(
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    *,
    name: str = '',
) -> Callable[..., Any]
```

Lifts `jax.grad` / `jax.value_and_grad` onto instantaneous pure JAX callables
and Psyre `Stream`s, returning gradient PyTrees of `Stream`s (or
`(grad_tree, aux)` / `(val, grad_tree)` / `((val, aux), grad_tree)`).

### `rewrite_dtype`

```python
def rewrite_dtype(
    graph: Graph,
    *,
    target_dtype: Any,
    source_dtype: Any = None,
    ops: Iterable[str] | None = None,
    primitives: Iterable[Any] | None = None,
    preferred_element_type: Any = None,
    include_stream_ops: bool = False,
) -> Graph
```

Returns a cloned, finalized `Graph` casting floating inputs of matching JAX/lax
nodes to `target_dtype` (and optionally setting `preferred_element_type` on
contraction ops).

## See Also

- [`psyre.jax`](jax.md) — `register_jax_pytree`, `get_primitive`, `is_jax_step_func`
- [`psyre.jax.lax`](jax_lax.md) — Canonical `Primitive` metadata for `rewrite_dtype`
- [`psyre.jax.jit`](jax_jit.md) — Fusing transformed/grad/vmapped graphs into `jax.jit` islands
