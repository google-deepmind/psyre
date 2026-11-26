# `psyre.numpy` *(Optional)*

Standalone optional NumPy stream-lifting package (`pip install "psyre[numpy]"`)
exposing a lazy drop-in `numpy` proxy, multi-output ufunc/linalg unpacking,
stream stacking/batching, and eager per-tick pointwise `vmap`.

## Overview & Role

`psyre.numpy` (implemented in `psyre.numpy.numpy` and re-exported from
`psyre.numpy`) lets you write NumPy computations directly over Psyre
[`Stream`](stream.md)s without adding a hard `numpy` dependency to core `psyre`:

```python
from psyre import Scheduler, Stream
from psyre import numpy as pnp

def prog():
  x = Stream([pnp.array([1.0, 2.0]), pnp.array([3.0, 4.0])])
  return pnp.sin(x) + pnp.sum(x, axis=-1, keepdims=True)

traces = Scheduler.run(prog, horizon=2)
```

Importing `psyre.numpy` when NumPy is not installed raises an actionable
`ImportError` directing to `pip install "psyre[numpy]"`. All pure-Python lifting
and batching plumbing (`LiftedNamespace`, `partition_comb_args`, mutation/purity
guards, `make_stack_streams`, `make_unstack_stream`, `make_batch_streams`) lives
in [`psyre.payload`](payload.md).

## Design & Key Invariants

### Lazy Proxy & Eager Zero-Stream Constants vs. `numpy.random`

- `psyre.numpy` implements module-level PEP 562 `__getattr__` and `__dir__`
  backed by a [`LiftedNamespace`](payload.md#shared-backend-lifting--batching-helpers)
  over `numpy`, alongside eager submodule proxies `linalg`, `fft`, and `random`:
  - `np.ufunc` attributes wrap in `LiftedUfunc`;
  - non-type callables wrap via `lift(attr)`;
  - submodules wrap recursively in `LiftedNamespace`;
  - classes, dtypes, and constants (e.g. `pnp.float32`, `pnp.pi`, `pnp.newaxis`)
    pass through verbatim.
- **Zero-stream eager execution**: When `partition_comb_args(...,
  static_literals=True)` finds no dynamic `Stream` or thunk inputs, pure
  functions execute **eagerly** and return plain NumPy arrays/scalars—keeping
  constants such as `pnp.zeros((2, 2))` clock-polymorphic across `match` /
  `automaton` arms.
- **Impure `numpy.random` samplers**: Bound methods of `np.random.Generator` /
  `np.random.RandomState` and `numpy.random.*` sampling functions (excluding
  setup helpers `default_rng`, `seed`, `get_state`, `set_state`,
  `get_bit_generator`, `set_bit_generator`, `spawn`) called during Stage 1 with
  zero dynamic inputs emit a 0-input impure (`pure=False`) stream node yielding
  a fresh draw on every tick.

### Purity Defaults, Mutation Guards & Coercion Rejection

- **Purity (`pure=None` default)**: `True` for `np.ufunc` and deterministic
  non-I/O `numpy.*` functions (`payload.is_pure_numpy_func`); `False` for
  `numpy.random` samplers, `empty`/`empty_like`, NumPy file I/O functions, and
  arbitrary non-NumPy user callables. Explicit `pure=True|False` overrides.
- **In-place mutation rejection**: `LiftedUfunc` rejects `.at` and non-null
  `out=` (`TypeError`). `lift` rejects `random.shuffle` / `rng.shuffle`
  (suggesting `permutation()`), `MUTATING_NUMPY_FUNCS` (`copyto`,
  `fill_diagonal`, `place`, `put`, `put_along_axis`, `putmask`), and non-null
  `out=` (`TypeError`), while stripping validated null `out=None` / `out=(None, ...)`.
- **Stream coercion rejection**: `pnp.array`, `pnp.asarray`, `pnp.asanyarray`,
  `pnp.asarray_chkfinite`, `pnp.ascontiguousarray`, `pnp.asfortranarray`, and
  `pnp.asmatrix` (matched by object identity) run eagerly on non-streams and
  raise `TypeError` when given dynamic `Stream`s, directing users to
  `stack_streams(...)`.

### NEP-13 / NEP-18 Delegation & Multi-Output Unpacking

- **CSE parity with bare `np.<op>`**: When top-level `Stream` or
  `__array_ufunc__`/`__array_function__`-capable `LazyThunk` arguments are
  passed without custom `lift` overrides, `LiftedUfunc` and `lift` delegate
  through NumPy's NEP-13 / NEP-18 hooks (falling back cleanly to
  `emit_comb_node` on nested containers, `lambda:` thunks, or non-NEP-18
  functions). Because [`common.same_step_func`](common.md#node-decorators--cache-free-structural-cse-same_step_func)
  matches step functions structurally by underlying callable (including bound C
  methods such as `np.add.reduce`), `pnp.sin(x)` and `np.sin(x)` deduplicate
  under `optimize_graph` CSE.
- **Automatic multi-output unpacking**: Multi-output ufuncs (`divmod`, `modf`,
  `frexp`, including `.outer`) and `numpy.linalg` routines (`eig`, `eigh`,
  `lstsq`, `qr`, `slogdet`, `svd`) automatically unpack into tuples of projected
  `Stream`s whose unused outputs are pruned by DCE.

### Eager Pointwise `vmap` vs. Structural `batch_streams`

- **`vmap`**: Runs per-tick Python-loop vectorization inside a single
  combinational node (no JAX dependency, no composite Psyre subgraph rewriting).
  Stacks homogeneous output PyTrees (`tuple`, `NamedTuple`, `list`, `Mapping`,
  registered PyTrees) via `stack_output_trees`.
- **`batch_streams`**: Performs Stage-1 structural lane expansion—unstacks
  mapped stream inputs into `size` scalar lanes, instantiates `fn` per lane
  inside `push_prefix(f'{name}[{lane}]')`, and stacks matching output trees.
  Supports arbitrary stateful nodes, automata, and heterogeneous per-lane
  clocks.

## Public API Reference

All symbols below are exported in `psyre.numpy.__all__` (plus any lazily
resolved `numpy` attribute via `__getattr__` / `__dir__`).

### Classes & Submodules

- `LiftedNamespace` *(re-exported from `psyre.payload`)*: Lazy caching module
  proxy.
- `LiftedUfunc(ufunc: np.ufunc)`: Stream-lifted proxy exposing `__call__`,
  `reduce`, `accumulate`, `reduceat`, `outer`, forwarding ufunc metadata
  (`.nin`, `.nout`, `.identity`), and rejecting `.at` and non-null `out=`.
- `linalg`, `fft`, `random`: `LiftedNamespace` proxies over `np.linalg`,
  `np.fft`, and `np.random`.

### Functions

#### `lift`

```python
def lift(
    fn: Callable[..., Any],
    *,
    pure: bool | None = None,
    nout: int | None = None,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    static_literals: bool = True,
    name: str = '',
) -> Any
```

Lifts a NumPy ufunc or callable onto Psyre `Stream`s. Returns `LiftedUfunc(fn)`
for a bare `np.ufunc` with default options; otherwise returns a wrapped callable
with inferred or explicit `pure` and `nout`.

#### `stack_streams` & `unstack_stream`

```python
def stack_streams(*streams: Any, axis: int = 0, name: str = '') -> Stream
def unstack_stream(stream: Any, size: int, *, axis: int = 0, name: str = '') -> tuple[Stream, ...]
```

Emits pure `np.stack` / axis-slice nodes (requiring an active Stage-1 builder;
step functions are LRU-cached by axis/size/index for CSE). `unstack_stream`
validates `shape[axis] == size` at Stage 2.

#### `batch_streams`

```python
def batch_streams(
    fn: Callable[..., Any],
    *streams: Any,
    size: int,
    in_axes: int | None | Sequence[int | None] = 0,
    out_axis: int | None = 0,
    name: str = '',
) -> Any
```

Structural Stage-1 lane expansion unstacking mapped inputs into `size` lanes,
evaluating `fn` per lane under `push_prefix`, and re-stacking outputs along
`out_axis` (returning a plain `tuple` of lane results if `out_axis is None` or
no lane output is dynamic).

#### `vmap`

```python
def vmap(
    fn: Callable[..., Any],
    in_axes: int | None | tuple[Any, ...] | list[Any] = 0,
    out_axes: int = 0,
    *,
    pure: bool = True,
    nout: int | None = None,
    name: str = '',
) -> Callable[..., Any]
```

Eager NumPy per-tick pointwise vectorization over batched array `Stream`s.
Validates non-zero consistent batch sizes across mapped inputs (`ValueError` on
0-D scalars, empty batch size `0`, mismatched sizes, or all-`None` `in_axes`).
`out_axes` must be a single `int`. Runs eagerly on zero-stream calls.

*(Internal module-hook helpers in `psyre.numpy.numpy`: `get_attr(name)` and
`dir_names()`.)*

## See Also

- [`psyre.payload`](payload.md) — Shared zero-dependency lifting & PyTree foundation
- [`psyre.stream`](stream.md#numpy-nep-13--nep-18-auto-lifting) — Built-in `Stream` NEP-13/NEP-18 hooks
- [`psyre.jax`](jax.md) — Optional JAX & `lax` lifting, subgraph `vmap`, `grad`, and `jit`
- `psyre.numpy.examples.cnn_mnist` — Stride-group MNIST CNN training example using `fby` parameter streams
