# `psyre.jax` & `psyre.jax.ops` *(Optional)*

Optional JAX stream-lifting package (`pip install "psyre[jax]"`) providing
lifted JAX namespaces (`jnp`, `lax`, `nn`, `random`, `scipy`), JAX backend and
`Primitive` metadata tagging, explicit host/device converters, PyTree bridging,
and unified exports for transformations and XLA JIT compilation.

## Overview & Role

`psyre.jax` brings JAX arrays, pure operations, lifting-time transformations,
and XLA compilation to Psyre without adding a hard `jax` or `numpy` dependency
to core `psyre`. Importing `psyre.jax` without JAX installed raises an
actionable `ImportError` pointing to `pip install "psyre[jax]"`.

The package is organized into four focused implementation modules re-exported
from `psyre.jax`:

- **`psyre.jax.ops`** *(documented on this page)*: Core JAX lifting engine
  (`lift`, `make_jax_namespace`), backend/primitive metadata (`get_primitive`,
  `is_jax_step_func`), explicit boundary adapters (`to_jax`, `to_numpy`),
  `register_jax_pytree`, and lifted namespaces `numpy` (`jnp`), `nn`, `random`,
  and `scipy`.
- **[`psyre.jax.lax`](jax_lax.md)**: Lifted `jax.lax` and `jax.lax.linalg` with
  auto-discovered canonical JAX `Primitive` metadata, higher-order intra-tick
  control flow, and multi-output unpacking.
- **[`psyre.jax.transforms`](jax_transforms.md)**: Lifting-time transformations
  without Stage-1 `Jaxpr` tracing (`vmap` over single ops and composite stateful
  Psyre subgraphs, `grad`, `value_and_grad`, `stack_streams`, `unstack_stream`,
  `batch_streams`, and `rewrite_dtype`).
- **[`psyre.jax.jit`](jax_jit.md)**: Lazy `jax.jit`, maximal same-clock convex
  island fusion (`fuse_jit_islands`, `jit_graph`), and opt-in Stage-1 `Jaxpr`
  expansion (`trace_jaxpr=True`, `expand_jaxpr`).

## Design & Key Invariants (`psyre.jax.ops`)

### Backend & `Primitive` Metadata Tagging

- `PRIMITIVE_TYPE = type(jax.lax.add_p)` discovers JAX's `Primitive` type via
  public `jax.lax` without depending on internal `jax.extend` APIs.
- `make_jax_step_func` stamps `_psyre_jax_backend = True` and optional
  `_psyre_jax_primitive` onto combinational `step_func` closures.
- Because [`common.same_step_func`](common.md#node-decorators--cache-free-structural-cse-same_step_func)
  compares all `__dict__` metadata tags strictly by identity, `optimize_graph`
  CSE never merges JAX-tagged nodes with host NumPy/Python nodes or nodes with
  different `Primitive` tags.
- `get_primitive(obj)` and `is_jax_step_func(obj)` work uniformly on `Node`s,
  lifted wrappers, and `step_func` closures.

### Pure-by-Default Lifting & Metadata Inheritance

- `lift` defaults to `pure=True` (all supported JAX ops, including explicit-key
  `jax.random`, are functionally pure).
- Zero dynamic inputs execute eagerly and return plain JAX arrays/scalars,
  preserving clock polymorphism for constants (`pjax.jnp.zeros`, `plax.iota`)
  inside `match` and `automaton` arms.
- Re-lifting an already-lifted wrapper inherits its `primitive`, `nout_spec`,
  and unions `_psyre_static_argnums` / `_psyre_static_argnames`.
- Unlike `psyre.numpy`, `psyre.jax` always emits via `partition_comb_args` and
  `emit_comb_node` (no NEP-13/NEP-18 delegation).

### Impure Op, `out=`, and Coercion Guards

- Impure/IO APIs (`breakpoint`, `callback`, `infeed`, `infeed_p`, `io_callback`,
  `outfeed`, `outfeed_p`, `print`, plus `jax.debug`/`jax.experimental` impure
  targets matched by object identity) are rejected unconditionally with
  `TypeError`.
- Non-null `out=` kwargs on JAX functions are rejected (`TypeError`), while
  null `out=None` is stripped.
- `jnp.array`, `jnp.asarray`, and `jnp.asanyarray` run eagerly on non-streams
  and raise `TypeError` when given dynamic `Stream`s, directing callers to
  `to_jax(...)` or `stack_streams(...)`.

### Host/Device Boundary Adapters (`to_jax` vs. `to_numpy`)

Both adapters run eagerly on non-streams and LRU-cache per-`dtype` step
functions (`maxsize=64`) for CSE, with one deliberate tagging difference:

- **`to_jax`** uses `make_jax_step_func` (JAX-tagged), so it fuses cleanly into
  downstream [`jax.jit`](jax_jit.md) islands.
- **`to_numpy`** uses plain `make_comb_step_func` (intentionally **not**
  JAX-tagged), cleanly terminating [`fuse_jit_islands`](jax_jit.md#direction-a-lazy-jit--convex-same-clock-island-fusion)
  at host NumPy boundaries.

### PyTree Bridging (`register_jax_pytree`)

`register_jax_pytree(node_type)` bridges any class registered with
`jax.tree_util` into [`psyre.payload.register_pytree_node`](payload.md#pytree-registration--call-plans)
so custom dataclasses and containers pass seamlessly through `@psyre_comb`,
`CombCallPlan`, `psyre.jax.vmap`, and `psyre.jax.jit`. Built-in `(tuple, list,
dict)` are no-ops. If an unregistered class is flattened at call time (where
`jax.tree_util.tree_flatten` treats it as an opaque 1-leaf self-reference), the
bridge raises `TypeError` instead of recursing infinitely.

## Public API Reference

### `psyre.jax` Package Exports (`__all__`)

- **From `psyre.jax.ops`**: `get_primitive`, `is_jax_step_func`, `lift`,
  `numpy` (and alias `jnp`), `nn`, `random`, `scipy`, `register_jax_pytree`,
  `to_jax`, `to_numpy`.
- **From `psyre.payload`**: `LiftedNamespace`.
- **From [`psyre.jax.lax`](jax_lax.md)**: `lax` submodule.
- **From [`psyre.jax.transforms`](jax_transforms.md)**: `BatchError`, `vmap`,
  `grad`, `value_and_grad`, `stack_streams`, `unstack_stream`, `batch_streams`,
  `rewrite_dtype`.
- **From [`psyre.jax.jit`](jax_jit.md)**: `ImpureJaxprError`, `JitIslandInfo`,
  `jit`, `expand_jaxpr`, `fuse_jit_islands`, `jit_graph`.

### `psyre.jax.ops` Symbols

- **Constants**: `BACKEND_ATTR = '_psyre_jax_backend'`,
  `PRIMITIVE_ATTR = '_psyre_jax_primitive'`, `PRIMITIVE_TYPE: type[Any]`.
- **Namespaces**:
  - `numpy` / `jnp`: `LiftedNamespace` over `jax.numpy` with `ARRAY_ELEMWISE_NOUT`
    multi-output unpacking (`divmod`, `frexp`, `modf`) and submodules `linalg`
    (`ARRAY_LINALG_NOUT`) and `fft`.
  - `nn`: `LiftedNamespace` over `jax.nn`.
  - `random`: `LiftedNamespace` over `jax.random`.
  - `scipy`: `LiftedNamespace` over `jax.scipy` (or `None` if unavailable).
- **Functions**:
  - `get_primitive(obj: Any) -> Any`: Returns attached JAX `Primitive` or `None`.
  - `is_jax_step_func(obj: Any) -> bool`: True if `obj` (a `Node`, wrapper, or
    `step_func`) is JAX-tagged.
  - `lift(fn, *, pure=True, nout=None, static_argnums=(), static_argnames=(), static_literals=True, primitive=None, name='') -> Callable[..., Any]`:
    Lifts a pure JAX/Python callable onto Psyre `Stream`s.
  - `to_jax(value: Any, *, dtype: Any = None, name: str = '') -> Any`:
    Converts a `Stream` (or eager value) to a JAX array via `jnp.asarray`.
  - `to_numpy(value: Any, *, dtype: Any = None, name: str = '') -> Any`:
    Converts a `Stream` (or eager value) to a host NumPy array via `np.asarray`.
  - `register_jax_pytree(node_type: type[Any]) -> type[Any]`: Decorator-friendly
    bridge registering a `jax.tree_util` PyTree class with `psyre.payload`.
  - Shared internal/extension helpers on `psyre.jax.ops`:
    `lift_with_nout_spec`, `make_jax_step_func`, `make_jax_namespace`,
    `attach_wrapper_meta`, `read_wrapper_nout_spec`.

## See Also

- [`psyre.jax.lax`](jax_lax.md) — Lifted `jax.lax` & `jax.lax.linalg` primitives
- [`psyre.jax.transforms`](jax_transforms.md) — `vmap`, `grad`, `value_and_grad`, `batch_streams`, `rewrite_dtype`
- [`psyre.jax.jit`](jax_jit.md) — Lazy `jit`, convex island fusion, and `Jaxpr` expansion
- [`psyre.numpy`](numpy.md) — Standalone NumPy lifting
- `psyre.jax.examples.cnn_mnist_eager` and `psyre.jax.examples.cnn_mnist_jit` (sharing `psyre.jax.examples.cnn_mnist`) — Eager and XLA JIT-compiled MNIST CNN training examples using `value_and_grad`
