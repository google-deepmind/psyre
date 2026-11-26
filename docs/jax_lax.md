# `psyre.jax.lax` *(Optional)*

Lifted `jax.lax` and `jax.lax.linalg` primitives with canonical JAX `Primitive`
metadata, first-order intra-tick higher-order control flow, and multi-output
tuple unpacking.

## Overview & Role

`psyre.jax.lax` (accessible as `from psyre.jax import lax as plax` or
`pjax.lax`) provides a PEP 562 lazy lifted proxy over `jax.lax` and
`jax.lax.linalg`, built on [`psyre.jax.ops.make_jax_namespace`](jax.md).

Every lifted operator carries its canonical JAX `Primitive` metadata
(`get_primitive(node)`), enabling compiler passes such as
[`rewrite_dtype(primitives=...)`](jax_transforms.md#rewrite_dtype) and
collective-operation detection in [`psyre.jax.vmap`](jax_transforms.md#vmap) to
match operators cleanly by JAX `Primitive` identity rather than fragile string
names.

## Design & Key Invariants

### Safe Auto-Discovery of JAX `Primitive` Metadata

- **No deprecated attribute triggers**: `_discover_primitives` inspects
  `jax.lax.__dict__` and `jax.lax.linalg.__dict__` directly (never calling
  `getattr` across `dir()`), avoiding deprecation warnings on legacy JAX
  aliases.
- For every public `*_p` entry whose value is an instance of `PRIMITIVE_TYPE`
  and whose stripped name `api_name = attr[:-2]` exists in `__dict__`,
  `api_name` is mapped automatically to that `Primitive`.
- Eight explicit overrides cover non-uniform JAX naming:
  - `approx_max_k`, `approx_min_k` → `approx_top_k_p`
  - `bitwise_and` → `and_p`, `bitwise_not` → `not_p`, `bitwise_or` → `or_p`,
    `bitwise_xor` → `xor_p`
  - `sort_key_val` → `sort_p`
  - `while_loop` → `while_p`
- All discovered mappings are exposed in the read-only `PRIMITIVES` dictionary
  (prefixing `jax.lax.linalg` entries with `'linalg.'`).

### Intra-Tick Higher-Order Control Flow Under First-Order Discipline

Psyre streams are strictly first-order (callables cannot flow on streams). To
support intra-tick `jax.lax` combinators whose arguments include Python
callables, `psyre.jax.lax` automatically marks those callable parameters as
static (`static_argnums` and `static_argnames`) so `CombCallPlan` keeps them
verbatim while lifting data arguments as streams:

| Operator | Static positional indices | Static keyword names |
|---|---|---|
| `associative_scan` | `(0,)` | `('fn',)` |
| `cond` | `(1, 2)` | `('true_fun', 'false_fun')` |
| `fori_loop` | `(2,)` | `('body_fun',)` |
| `map` | `(0,)` | `('f',)` |
| `reduce`, `reduce_window` | `(2,)` | `('computation',)` |
| `scan` | `(0,)` | `('f',)` |
| `select_and_scatter` | `(1, 7)` | `('select', 'scatter')` |
| `switch` | `(1,)` | `('branches',)` |
| `while_loop` | `(0, 1)` | `('cond_fun', 'body_fun')` |

### Multi-Output Tuple Unpacking

Multi-output `lax` and `lax.linalg` routines automatically unpack their return
tuples into individual projected `Stream`s (so unused outputs are eliminated by
DCE):

- **`jax.lax`**: `approx_max_k` (2), `approx_min_k` (2), `rng_bit_generator`
  (2), `scan` (2), `sort_key_val` (2), `top_k` (2).
- **`jax.lax.linalg`**: `eigh` (2), `hessenberg` (2), `lu` (3),
  `tridiagonal` (4), plus keyword-aware resolvers:
  - `qr`: `3` when `pivoting=True`, else `2`;
  - `svd`: `3` when `compute_uv=True` (default), else `1`;
  - `eig`: `1 + int(bool(compute_left_eigenvectors)) + int(bool(compute_right_eigenvectors))` (both default `True` → `3`);
  - `schur`: `2` when `compute_schur_vectors=True` (default), else `1`.

### Impure Op Rejection

Inherits `make_jax_namespace`'s impure blocklist, rejecting `infeed`,
`infeed_p`, `outfeed`, and `outfeed_p` with `TypeError`.

## Public API Reference

All symbols below are exported in `psyre.jax.lax.__all__` (plus any lazily
resolved `jax.lax` attribute via `__getattr__` / `__dir__`).

- `PRIMITIVES: types.MappingProxyType[str, Any]`: Read-only mapping from
  `lax` op names (and `'linalg.<op>'` keys) to canonical JAX `Primitive`
  objects.
- `linalg: LiftedNamespace`: Lifted `jax.lax.linalg` namespace with attached
  `Primitive` metadata and multi-output unpacking.
- `get_primitive` *(re-exported from `psyre.jax.ops`)*: Retrieves attached JAX
  `Primitive` metadata from a `Node`, lifted wrapper, or `step_func`.
- `lift` *(re-exported from `psyre.jax.ops`)*: Lifts custom pure callables with
  optional explicit `primitive=...` metadata.

## See Also

- [`psyre.jax`](jax.md) — Core JAX lifting engine (`psyre.jax.ops`)
- [`psyre.jax.transforms`](jax_transforms.md) — `vmap` collective detection and `rewrite_dtype(primitives=...)`
- [`psyre.jax.jit`](jax_jit.md) — `Jaxpr` expansion attaching canonical `Primitive` metadata
