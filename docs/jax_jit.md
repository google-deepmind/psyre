# `psyre.jax.jit` *(Optional)*

XLA compilation and graph fusion for JAX-backed Psyre programs: lazy `jax.jit`
wrapping, maximal same-clock convex island fusion, and opt-in Stage-1 `Jaxpr`
expansion gated on explicit `jax.ShapeDtypeStruct` hints.

## Overview & Role

`psyre.jax.jit` (all public symbols re-exported from [`psyre.jax`](jax.md))
supports two complementary compilation workflows—letting users stay completely
shape-polymorphic by default while opting into fine-grained `Jaxpr` IR lowering
when static shapes are known:

1. **Direction A (`trace_jaxpr=False`, default)**: Lazy `jit(fn)` wrapping (zero
   Stage-1 tracing, no shape annotations) paired with whole-graph
   `fuse_jit_islands(graph)` / `jit_graph(graph)` contracting maximal same-clock
   convex DAGs of pure JAX nodes into fused `jax.jit` super-nodes.
2. **Direction B (`trace_jaxpr=True`, opt-in)**: Traces a pure JAX function at
   Stage 1 via `jax.make_jaxpr` using explicit `jax.ShapeDtypeStruct` hints
   (`in_avals`) and expands each `JaxprEqn` via `expand_jaxpr` into fine-grained
   Psyre `Node`s carrying canonical JAX `Primitive` metadata.

## Design & Key Invariants

### Direction A: Lazy `jit` & Convex Same-Clock Island Fusion

- **Lazy `jit(fn, trace_jaxpr=False, **jit_kwargs)`**:
  - Rejects `in_avals is not None` (`ValueError`).
  - Unwraps `raw_fn = unwrap_inst_func(fn)`, inherits static arg specs and
    `nout_spec`, wraps `jax.jit(raw_fn, **jit_kwargs)`, and emits a single pure
    JAX-tagged node on dynamic calls (or executes eagerly on zero dynamic
    inputs).
  - Under `Semantics.SYNC`, inactive `Absent` ticks short-circuit in
    `psyre.common._comb` before ever invoking XLA.
- **Whole-graph island fusion (`fuse_jit_islands`, `jit_graph`)**:
  - **Eligibility**: Pure, stateless JAX-tagged nodes (`is_jax_step_func`,
    including JAX-tagged `jax_out_{i}` tuple projections emitted for
    multi-output JAX ops such as `top_k`, `svd`, `jit(..., nout=...)`, `vmap`,
    and `value_and_grad`), plus pure kwarg-free `Stream` arithmetic/comparison
    operators when `include_stream_ops=True` (excluding Python `max`/`min`,
    which fail tracing on 0-D JAX arrays, and `Stream.__getitem__`, keeping
    arbitrary user dict/metadata indexing outside XLA islands). Stateful nodes, reactive primitives
    (`fby`, `pre`, `post`, `when`, `whennot`, `merge`, `switch`, `ifthenelse`,
    `buffer`, `current`, `morph`, `counter`, `prefix_sum`, `scan`), impure
    nodes, and host `to_numpy` / NumPy nodes are never fused and act as clean
    island boundaries.
  - **Convexity-preserving contraction**: Builds the instantaneous dependency
    DAG (excluding delayed `fby`/`pre` back-edges) and iteratively merges
    connected eligible clusters sharing **both** identical `node_clock` and
    identical `node.arms` only when a BFS on the quotient DAG proves no external
    path bypasses the direct edge (`ca -> X ->+ cb`). This guarantees fused
    islands never introduce instantaneous causality cycles across host/control
    diamonds.
  - **Filtering & idempotency**: Retains only clusters with
    `len(cluster) >= min_nodes` containing at least one JAX-tagged node (so
    pure non-JAX `Stream` subgraphs are untouched) and at least one observable
    exit (`graph.output`, `extra_roots`, `ArmScope.cond`, or external consumer).
    Multi-exit islands emit a `{label}_bundle` super-node followed by non-JAX-tagged
    `island_out_{i}` projection nodes, keeping `fuse_jit_islands` idempotent.
  - **Non-destructive reconstruction**: Clones via `graph.clone_streams`, labels
    each super-node with the longest shared hierarchy prefix across its member
    nodes, preserves `check_clocks`, and verifies `check_causality(new_graph)`.

### Direction B: Opt-In Stage-1 `Jaxpr` Expansion (`trace_jaxpr=True`)

- **Strict `ShapeDtypeStruct` gating**:
  - `jit(fn, trace_jaxpr=True, in_avals=...)` requires explicit `in_avals` whose
    leaves are all `jax.ShapeDtypeStruct` instances (`TypeError` on concrete
    arrays/scalars; `ValueError` if missing or empty).
  - Enforces exact PyTree structure equality between `in_avals` and dynamic
    arguments at call time (`ValueError` on mismatch) and rejects dynamic
    streams inside `kwargs` (`TypeError`).
  - Outside Stage 1 or with zero dynamic inputs, delegates eagerly to
    `jax.jit(raw_fn)`.
  - *Note*: The returned wrapper sets `_psyre_inst_func = jitted_fn`, so passing
    it directly to `pjax.vmap(...)` commutes `jax.vmap` with `jitted_fn`; wrap
    in a `lambda` to run composite Stage-1 `vmap` over the expanded `JaxprEqn`
    nodes.
- **Recursive purity verification (`ImpureJaxprError`)**:
  - Before expanding, recursively inspects `.effects` on the `ClosedJaxpr`,
    unwrapped `Jaxpr`, every `JaxprEqn`, and any nested sub-`Jaxpr` inside
    `eqn.params`, raising `ImpureJaxprError` if any side effect (such as
    `jax.debug.print`) is present.
- **Fine-grained `JaxprEqn` lowering (`expand_jaxpr`)**:
  - Emits nodes inside `push_prefix(name)`, attaching each equation's canonical
    JAX `Primitive` (`_psyre_jax_primitive`) and caching evaluators in a bounded
    2,048-entry LRU `OrderedDict` (with dtype- and signed-zero-safe parameter
    freezing) so [`optimize_graph`](optimization.md) CSE merges identical
    expanded primitives.
  - **Stage-1 constant folding**: Pure non-collective equations whose inputs are
    all `Literal`s or closed-over `constvars` evaluate eagerly into
    `Stream.constant`s at Stage 1.
  - **Multi-result, `DropVar` & `expand_subjaxprs`**: Multi-result primitives
    (e.g. `top_k`) emit one bundle node plus JAX-tagged projections only for
    non-`DropVar` outputs (which round-trip cleanly through `fuse_jit_islands`).
    Setting `expand_subjaxprs=True` recursively flattens nested
    `jit`/`pjit`/`xla_call` sub-`Jaxpr`s under child prefix scopes.
  - Composes cleanly with `optimize_graph`, `rewrite_dtype` (forwarding
    `preferred_element_type` overrides), and `fuse_jit_islands`.

## Public API Reference

All symbols below are re-exported from `psyre.jax`.

### `ImpureJaxprError`

```python
class ImpureJaxprError(ValueError)
```

Raised by `expand_jaxpr` or `jit(..., trace_jaxpr=True)` when a traced `Jaxpr`
contains non-empty effects.

### `JitIslandInfo`

```python
@dataclasses.dataclass(frozen=True)
class JitIslandInfo:
    name: str
    clock: Clock
    node_names: tuple[str, ...]
    num_inputs: int
    num_outputs: int

    @property
    def num_nodes(self) -> int: ...
```

Summary metadata for one fused same-clock `jax.jit` super-node returned by
`fuse_jit_islands`.

### `jit`

```python
def jit(
    fn: Callable[..., Any],
    *,
    trace_jaxpr: bool = False,
    in_avals: Any = None,
    expand_subjaxprs: bool = False,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    nout: int | None = None,
    name: str = '',
    **jit_kwargs: Any,
) -> Callable[..., Any]
```

Lifts a pure JAX function behind `jax.jit` (Direction A when `trace_jaxpr=False`,
or Direction B Stage-1 `Jaxpr` expansion when `trace_jaxpr=True` with explicit
`ShapeDtypeStruct` `in_avals`).

### `expand_jaxpr`

```python
def expand_jaxpr(
    closed_jaxpr: Any,
    *args: Any,
    out_tree: Any = None,
    expand_subjaxprs: bool = False,
    name: str = 'jaxpr',
) -> Any
```

Expands a traced pure `ClosedJaxpr` or `Jaxpr` into fine-grained JAX-tagged
Psyre `Node`s on the active `GraphBuilder`. Raises `RuntimeError` outside Stage
1, `ImpureJaxprError` on effects, and `ValueError` on `constvars`/`invars` arity
mismatches.

### `fuse_jit_islands` & `jit_graph`

```python
def fuse_jit_islands(
    graph: Graph,
    *,
    min_nodes: int = 2,
    include_stream_ops: bool = True,
    extra_roots: Iterable[Any] | None = None,
    jit_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Graph, tuple[JitIslandInfo, ...]]

def jit_graph(
    graph: Graph,
    *,
    min_nodes: int = 2,
    include_stream_ops: bool = True,
    extra_roots: Iterable[Any] | None = None,
    jit_kwargs: Mapping[str, Any] | None = None,
) -> Graph
```

Non-destructively contracts maximal same-clock convex JAX islands in `graph`
into single `jax.jit` super-nodes. `fuse_jit_islands` returns `(fused_graph,
islands)`; `jit_graph` returns only `fused_graph`.

## See Also

- [`psyre.jax`](jax.md) — `to_jax` / `to_numpy` island boundaries and `is_jax_step_func`
- [`psyre.jax.transforms`](jax_transforms.md) — `vmap`, `grad`, and `rewrite_dtype`
- [`psyre.clock`](clock.md) and [`psyre.analysis`](analysis.md) — Clocks and causality preserved across fusion
