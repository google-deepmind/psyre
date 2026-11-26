# `psyre.payload`

Dependency-free leaf module providing first-order payload enforcement,
metadata-only scalar/array inspection, NumPy immutability and purity guards,
PyTree registration and call plans, and shared optional-backend lifting
factories.

## Overview & Role

`psyre.payload` sits at the bottom of Psyre's dependency graph: it imports
**only Python's standard library** (zero imports of `psyre`, `numpy`, or `jax`).
Separating payload inspection, PyTree flattening, and lifting infrastructure
into this leaf module achieves two goals:

1. **Zero-dependency core**: Core `psyre` remains pure Python stdlib while
   interoperating seamlessly with plain Python scalars, NumPy arrays
   (NEP-13/NEP-18), and JAX arrays without importing either package.
2. **Shared backend foundation**: Optional array subpackages
   ([`psyre.numpy`](numpy.md), [`psyre.jax`](jax.md),
   [`psyre.jax.lax`](jax_lax.md)) reuse the same `LiftedNamespace`, multi-output
   arity tables, axis-mapping utilities, and structural stream-batching
   factories without duplicating pure-Python plumbing.

## Design & Key Invariants

### Zero-Import & Zero-Transfer Inspection

- **Marker-attribute type checks**: `is_stream_obj`, `is_sentinel_obj`, and
  `is_lazy_thunk_obj` inspect class attributes (`_psyre_is_stream`,
  `_psyre_is_sentinel`, `_psyre_is_lazy_thunk`) on `type(value)` rather than
  importing `Stream` or calling `getattr` on instances, avoiding both circular
  imports and custom `__getattr__` side effects.
- **Metadata-only array checks**: `is_array_object` fast-rejects built-in
  scalars, streams, sentinels, thunks, NumPy scalars, enums, and standard
  containers, then inspects only `.shape` (must be a `tuple`) and `.dtype`
  (non-`None`). It never calls `repr()`, `bool()`, or `__array__()` and never
  triggers JAX device-to-host synchronization.
- **NumPy scalar detection**: `is_numpy_scalar`, `is_numpy_bool`, and
  `is_numpy_bool_array_scalar` inspect `type(value).__module__`, `ndim`,
  `dtype.kind`, and `.item` without importing `numpy`.

### Strict First-Order Payload Enforcement (`reject_callable`)

Psyre enforces a strict first-order stream discipline: streams carry values,
never callables or nested `Stream`s. `reject_callable` runs a cycle-safe DFS
(`O(|visited|)`) across `tuple`, `list`, `set`, `frozenset`, and `Mapping`
(both keys and values), fast-pathing sentinels and `BUILTIN_SCALAR_TYPES`.

- **Error precedence**: `LazyThunk` proxies and user callables raise `TypeError`
  immediately. When `reject_streams=True`, encountering a nested `Stream`
  continues scanning the container so an unresolved `LazyThunk` or callable
  elsewhere in the structure still reports first before raising the nested
  `Stream` error.

### Exact Value Keys & Signed-Zero Safety

- `zero_sign_key(value)` uses `math.copysign(1.0, ...)` to separate `+0.0` from
  `-0.0` on `float`, `complex` (2-tuple for real and imaginary parts), and
  NumPy float/complex scalars.
- `exact_value_key(value)` returns a hashable key `(cls, value, sign_key)` that
  separates exact Python types (`True` vs `1` vs `1.0`) and signed zeros,
  recursing on `tuple` and `frozenset`. Returns `None` for `NaN`, dense arrays,
  streams, thunks, and unhashable/mutable containers. Shared by
  [`static_stream.same_const_value`](static_stream.md) and
  [`optimization.py`](optimization.md).
- `safe_scalar_eq(av, bv)` evaluates `av == bv` safely: skips non-0-D arrays,
  non-NumPy arrays (such as `jax.Array`), and containers enclosing streams or
  thunks; permits 0-D `np.ndarray` comparisons; catches `(TypeError, ValueError,
  RuntimeError)`; and returns `True` only when equality yields a `bool` or
  `np.bool_`.

### NumPy Immutability & Purity Guards

- `reject_numpy_inplace_ufunc` rejects ufunc `.at` and any non-null `out=`
  (`TypeError`).
- `reject_numpy_mutating_func` rejects in-place functions in
  `MUTATING_NUMPY_FUNCS` (`copyto`, `fill_diagonal`, `place`, `put`,
  `put_along_axis`, `putmask`) and non-null `out=`.
- `strip_null_out` strips validated null `out=None` or tuples of `None`
  (e.g. `out=(None, None)` from multi-output ufuncs) before building nodes.
- `is_pure_numpy_func` returns `True` only for deterministic, non-I/O,
  non-`numpy.random` NumPy functions (excluding `empty`, `empty_like`, file I/O
  loaders/savers, and `memmap`) so CSE never merges impure NumPy calls.

### PyTree Registry & `CombCallPlan`

[`@psyre_comb`](common.md#runtime-execution--decorators) and [`@psyre_sequ`](common.md#runtime-execution--decorators)
accept nested containers (`tuple`, `NamedTuple`, `list`, `Mapping`, `slice`) and
custom registered PyTrees mixing `Stream` leaves with static data:

- **Zero-overhead fast path**: When `partition_comb_args` sees flat positional
  arguments, no matching `static_argnums`, no literals under
  `static_literals=True`, and no dynamic leaves inside `kwargs`, it skips
  `CombCallPlan` allocation entirely and returns `(tuple(dyn_inputs),
  dict(kwargs))`.
- **Single-pass bottom-up template construction**: Otherwise, `partition_comb_args`
  builds a frozen `CombCallPlan` stored under `CALL_PLAN_KWARG`
  (`'_psyre_call_plan'`). Subtrees containing zero dynamic leaves remain
  verbatim without allocating `PlanNode` wrappers, and custom PyTree
  `flatten_fn` callbacks run at most once per node. Mapping keys remain static
  while values are templated. Cyclic containers enclosing a dynamic leaf raise
  `ValueError` (purely static cyclic containers pass through verbatim).
- **Stage-2 materialization**: `CombCallPlan.materialize(values)` and
  `materialize_comb_call(kwargs, values)` reconstruct `(args, kwargs)` on each
  active tick from the flat stream tick values.

## Public API Reference

### Constants & Type Aliases

- `STREAM_MARKER_ATTR = '_psyre_is_stream'`
- `SENTINEL_MARKER_ATTR = '_psyre_is_sentinel'`
- `LAZY_THUNK_MARKER_ATTR = '_psyre_is_lazy_thunk'`
- `CALL_PLAN_KWARG = '_psyre_call_plan'`
- `BUILTIN_SCALAR_TYPES`: `frozenset({type(None), bool, int, float, complex, str, bytes, bytearray})`
- `NUMPY_UFUNC_METHODS`: `frozenset({'__call__', 'reduce', 'accumulate', 'reduceat', 'outer'})`
- `MUTATING_NUMPY_FUNCS`: `frozenset({'copyto', 'fill_diagonal', 'place', 'put', 'put_along_axis', 'putmask'})`
- `ARRAY_ELEMWISE_NOUT`: `MappingProxyType({'divmod': 2, 'frexp': 2, 'modf': 2})`
- `ARRAY_LINALG_NOUT`: `MappingProxyType` mapping `eig`, `eigh`, `lstsq`, `qr`,
  `slogdet`, and `svd` to static or argument-aware output arities (`NoutSpec`).
- `PyTreeFlattenFn = Callable[[Any], tuple[Sequence[Any], Any]]`
- `PyTreeUnflattenFn = Callable[[Any, Sequence[Any]], Any]`
- `NoutSpec = int | Callable[[tuple[Any, ...], Mapping[str, Any]], int]`

### Predicates, Keys & Validators

- `is_stream_obj(value: Any) -> bool`
- `is_sentinel_obj(value: Any) -> bool`
- `is_lazy_thunk_obj(value: Any) -> bool`
- `is_callable_target(value: Any) -> bool`: True when callable and not a
  `LazyThunk`.
- `is_dynamic_leaf(val: Any) -> bool`: True when `val` is a `Stream`,
  `LazyThunk`, or strictly zero-argument Python function.
- `is_zero_arg_function(val: Any) -> bool`: True for plain 0-arg `FunctionType`
  (rejecting `*args`, `**kwargs`, generators, coroutines, async generators).
- `is_namedtuple(value: Any) -> bool`: Distinguishes `NamedTuple` subclasses
  from arbitrary `tuple` subclasses.
- `is_from_package(obj: Any, package: str) -> bool`
- `is_numpy_bool(value: Any) -> bool`, `is_numpy_bool_array_scalar(value: Any) -> bool`,
  `is_numpy_scalar(value: Any) -> bool`, `is_array_object(value: Any) -> bool`,
  `is_nan_scalar(value: Any) -> bool`, `is_builtin_scalar_value(value: Any) -> bool`
- `contains_stream_or_thunk(value: Any, seen: set[int] | None = None, *, include_zero_arg_fn: bool = False) -> bool`
- `zero_sign_key(value: Any) -> Any`, `exact_value_key(value: Any) -> Any | None`,
  `safe_scalar_eq(av: Any, bv: Any) -> bool`
- `is_int_ge(value: Any, minimum: int) -> bool`,
  `validate_non_negative_int(k: int, op_name: str = 'value') -> int`,
  `validate_positive_int(n: int, label: str = 'value') -> int`
- `is_int_axis(value: Any) -> bool`,
  `validate_axis(axis: Any, label: str = 'axis') -> int`,
  `validate_optional_axis(axis: Any, label: str = 'axis') -> int | None`
- `reject_callable(value: Any, *, label: str = 'Stream value', reject_streams: bool = False) -> None`
- `ufunc_nout(ufunc: Any, method: str) -> int`,
  `reject_out_kwarg(fn_name: str, kwargs: Mapping[str, Any], *, kind: str = 'function') -> None`,
  `strip_null_out(kwargs: Mapping[str, Any]) -> dict[str, Any]`,
  `reject_numpy_inplace_ufunc(ufunc: Any, method: str, kwargs: Mapping[str, Any]) -> None`,
  `reject_numpy_mutating_func(func: Any, kwargs: Mapping[str, Any]) -> None`,
  `is_pure_numpy_func(func: Any) -> bool`

### PyTree Registration & Call Plans

- `register_pytree_node(node_type: type[Any], flatten_fn: PyTreeFlattenFn, unflatten_fn: PyTreeUnflattenFn) -> None`
  *(re-exported from top-level `psyre`)*: Registers a custom container class for
  `@psyre_comb` / `@psyre_sequ` PyTree traversal (matching `jax.tree_util`'s
  contract).
- `unregister_pytree_node(node_type: type[Any]) -> None` *(re-exported from
  top-level `psyre`)*: Removes a registered custom PyTree node type if present.
- `lookup_pytree_node(node_type: type[Any]) -> tuple[PyTreeFlattenFn, PyTreeUnflattenFn] | None`
- Template classes (frozen slotted dataclasses implementing
  `.materialize(values)`): `PlanNode`, `DynamicSlot(index)`,
  `SeqTemplate(seq_type, items, named=False)`,
  `DictTemplate(mapping_type, items)`, `SliceTemplate(start, stop, step)`,
  `CustomNodeTemplate(node_type, unflatten_fn, aux_data, children)`, and
  `CombCallPlan(args_template, kwargs_template)`.
- `materialize_comb_call(kwargs: Mapping[str, Any], values: Sequence[Any]) -> tuple[tuple[Any, ...], dict[str, Any]]`
- `partition_comb_args(args, kwargs, *, static_argnums=frozenset(), static_argnames=frozenset(), static_literals=False) -> tuple[tuple[Any, ...], dict[str, Any]]`
- `normalize_static_argnums(static_argnums: int | Iterable[int]) -> frozenset[int]`
- `normalize_static_argnames(static_argnames: str | Iterable[str]) -> frozenset[str]`

### Shared Backend Lifting & Batching Helpers

- `resolve_nout(spec: NoutSpec | None, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int`
- `normalize_arg_axes(axes_spec: Any, num_args: int, label: str = 'in_axes', *, error_cls: type[Exception] = ValueError) -> tuple[Any, ...]`
- `map_tree_with_axes(value, axis_spec, leaf_fn, *, label='axes', is_leaf=None, error_cls=ValueError) -> Any`:
  Walks nested containers and registered PyTrees in lockstep with `axis_spec`,
  broadcasting scalar `int | None` specs down subtrees.
- `stack_output_trees(items: Sequence[Any], leaf_stack_fn: Callable[[list[Any], str], Any], *, name: str = '') -> Any`:
  Stacks matching container trees across lanes, raising `ValueError` on type,
  length, key, or `None` mismatches.
- `LiftedNamespace(module, label='', *, wrap_attr, submodules=None, rejected=frozenset(), reject_message=...)`:
  Lazy caching attribute proxy lifting a Python module onto Psyre `Stream`s.
- `make_stack_streams(...)`, `make_unstack_stream(...)`,
  `make_batch_streams(...)`: Backend-parameterized factories producing
  `stack_streams`, `unstack_stream`, and structural Stage-1 `batch_streams` for
  `psyre.numpy` and `psyre.jax`.

## See Also

- [`psyre.common`](common.md) — `@psyre_comb`, `@psyre_sequ`, and `CombCallPlan` execution
- [`psyre.numpy`](numpy.md) and [`psyre.jax`](jax.md) — Optional backends built on `psyre.payload`
