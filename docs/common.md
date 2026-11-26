# `psyre.common`

Shared runtime infrastructure bridging Stage-1 graph elaboration and Stage-2
execution: sentinels, semantics enum, tag/selector algebra, Stage-1 scope
stacks, node decorators, `RingBuffer`, structural step equivalence, and the 14
core primitive dispatchers.

## Overview & Role

`psyre.common` is the shared contract layer sitting directly above
[`psyre.payload`](payload.md). Both Stage-1 modules (`stream`, `graph`,
`builder`, `match`, `automaton`, `clock`, `optimization`) and Stage-2 modules
(`sync_primitives`, `kahn_primitives`, `scheduler`) depend on `psyre.common`,
while `psyre.common` avoids circular imports back into `stream`, `graph`, or
`scheduler` by communicating through `ContextVar` slots (`active_builder`,
`active_scheduler`) and internal protocols.

## Design & Key Invariants

### Sentinels: `Absent`, `Bottom`, `None`, `UNSET`, and `UNREADY`

Psyre separates five distinct concepts so ordinary Python `None` remains a
first-class payload value on every stream:

| Value | Role & Contract |
|---|---|
| `Absent` | Synchronous clock-absence marker emitted on inactive subclock ticks under `Semantics.SYNC` (never emitted under `Semantics.KAHN`). Forbidden in static streams, `RingBuffer`, and as a return/yield value from `@psyre_comb`/`@psyre_sequ` (`TypeError`). |
| `Bottom` | Undefined/uninitialized slot marker (empty `RingBuffer` slots, uninitialized internal primitive state, and post-horizon EOF trace padding). Never written to `Stream.seq` (`Stream.push(Bottom)` raises `TypeError`; observing `Bottom` on `pull()` raises `RuntimeError`). |
| `None` | Ordinary valid stream payload value (e.g., `pre` initial values, `current` default, valid buffered element). |
| `UNSET` | Internal marker distinguishing omitted optional arguments (such as `default=UNSET` in `merge`/`switch`) from explicit `default=None`. |
| `UNREADY` | Returned synchronously by `StreamPort.try_pull_ready()` when a dynamic slot has not been pushed yet. |

`Absent` and `Bottom` are singleton instances of `_Sentinel` (`__bool__ ->
False`, `__eq__ -> False` even for `x == x`, `__hash__ = None`). Always test
them with identity (`is Absent` / `is Bottom`).

### Strict Lossless `RingBuffer`

`RingBuffer(size)` backs the `buffer` primitive in both `SYNC` and `KAHN`:

- Initializes `slots = [Bottom] * size` with `pull_idx = 0`, `push_idx = 0`.
- Using `Bottom` for empty/drained slots allows `None` to be buffered safely and
  distinguishes full (`slots[pull_idx] is not Bottom`) from empty
  (`slots[push_idx] is Bottom`) in `O(1)` time without a length counter.
- `pull(val)` rejects `Absent`/`Bottom` (`TypeError`) and raises `BufferError`
  on overflow without mutating indices or slots.
- `push()` raises `BufferError` on underflow without advancing `push_idx`, and
  resets each popped slot back to `Bottom`.

### Type-Disambiguated Tag & Selector Algebra

Python equates `True == 1` and `False == 0`, which would corrupt discrete tag
dispatch when `bool` and `int` tags mix. `psyre.common` prevents collisions via:

- `tag_key(tag) -> (is_bool, normalized_tag)`: normalizes NumPy `np.bool_` and
  0-D boolean `np.ndarray` to Python `bool`, separating `(True, True)` from
  `(False, 1)`.
- `WhenSelector` (frozen dataclass): inspectable tag predicate (`is_boolean`,
  `values: frozenset[TagKey]`, `negated`) driving `when`, `ClockStep`, and
  optimizer simplifications. `normalize_when_selector` folds boolean-domain
  tagged selectors against `BOOL_DOMAIN = {(True, True), (True, False)}`.
- `MergeSelector` (frozen dataclass): inspectable branch-index lookup
  (`is_boolean`, frozen `MappingProxyType` `arms`, optional `default_index`).
  Raises `KeyError` on unhandled discrete tags when `default_index is None`, and
  derives each arm's `WhenSelector` via `arm_selector(arm_index)`.

### Stage-1 Scope Stacks & `BlockHandle`

Stage-1 constructs track lexical nesting via `ContextVar` tuples:

- `reset_stack`: enclosing `with reset(r):` condition streams.
- `prefix_stack`: enclosing `@psyre_node(name)` / `automaton(name=...)` segments
  stored as `(segment_name, reset_depth)` pairs so inner resets know their
  enclosing hierarchy.
- `arm_stack`: enclosing `ArmScope` tokens (compared by object identity,
  `eq=False`, storing `cond`, `selector`, `reset_depth`, and `prefixes`).
- `CapturedEnv.capture()` / `.restore()` snapshots and restores all three stacks
  around deferred Stage-1 thunks and `match`/`automaton` arm outputs.
- `active_block` + `BlockHandle(Mapping)`: collects named arm outputs during
  `match`/`automaton` construction (rejecting reads before finalization and
  reserved `Mapping` attribute names) and exposes read-only merged `Stream`
  outputs once finalized.

### Single-Step `LazyThunk` & `force_thunk`

Forward references use either a strictly zero-argument Python function
(`lambda: ...`) or a [`LazyThunk`](#thunks-tags--validation) subclass (such as
[`psyre.equations.Thunk`](equations.md#thunk)). `force_thunk(val)` performs one
forcing step and rejects nested thunks or returned callables with `TypeError`,
enforcing first-order streams and preventing double-thunk escape bugs.

### Node Decorators & Cache-Free Structural CSE (`same_step_func`)

- `@psyre_loop(step_func, *, prepare=None, stateful=False)`: registers a
  low-level coroutine or async-generator primitive. Explicit `stateful=True` is
  used for factories like `fby`/`pre` that return an async generator without
  being `async def` generator functions themselves.
- `@psyre_comb(..., pure=False, nout=None, static_argnums=(), static_argnames=(), static_literals=False)`:
  lifts a stateless Python function into a Stage-1 node. Defaults to
  `pure=False` (safe for impure functions like `random_stream`); pure operators
  opt into CSE with `pure=True`. Arities 0, 1, and 2 on flat inputs take an
  unrolled inline fast path using `StreamPort.try_pull_ready()`; nested PyTrees
  or static arguments compile into a [`CombCallPlan`](payload.md#pytree-registry--combcallplan).
  Always pulls all inputs before checking `Absent` so port cursors and EOF
  barriers remain aligned under `SYNC`.
- `@psyre_sequ(...)`: lifts a Python generator function into a resettable
  stateful async-generator node. Instantiates and primes (`next(gen)`) the
  generator on `start`, feeds subsequent ticks via `gen.send(...)`, skips
  `Absent` ticks without advancing or resetting the generator, closes previous
  generators cleanly on reset and task exit (`gen.close()`), and rejects dynamic
  leaves inside `kwargs` (`TypeError`, since `send()` only carries positional
  values).
- **Step-function identity (`same_step_func`)**: Rather than matching nodes by
  string `node.op` names (which fails when user functions collide with builtin
  names) or caching closure factories with `lru_cache`, `same_step_func(a, b)`
  compares `make_comb_step_func` closures structurally: identical code object,
  equivalent `_psyre_inst_func` (including bound Python methods and bound C
  methods like `np.add.reduce` compared by `__self__` identity without invoking
  user `__eq__`), and pointer-identical `__dict__` metadata tags (`_psyre_pure`,
  `_psyre_returns_scalar`, `_psyre_jax_backend`, `_psyre_jax_primitive`).
- **Logging hygiene**: `logger = logging.getLogger('psyre')` sets
  `propagate = False` and attaches a handler without mutating the root logger.

## Public API Reference

### Sentinels, Enum, Logger & Registries

- `Absent`, `Bottom`: Singleton `_Sentinel` markers *(re-exported from `psyre`)*.
- `UNSET`, `UNREADY`: Internal sentinel objects.
- `Semantics`: `Enum` with members `Semantics.KAHN` and `Semantics.SYNC`
  *(re-exported from `psyre`)*.
- `logger`: Module logger `'psyre'` *(re-exported from `psyre`)*.
- `BOOL_PRODUCER_OPS`, `BOOL_BINARY_OPS`, `BOOL_DOMAIN`, `TagKey`, `PrepareFn`,
  `primitives_registry`.
- `register_primitives(semantics: Semantics, module: Any) -> None`
  *(re-exported from `psyre`)*.

### Thunks, Tags & Validation

- `class LazyThunk(abc.ABC)` *(re-exported from `psyre`)*: Abstract base class
  for non-lambda Stage-1 deferred proxies requiring
  `@abc.abstractmethod def __psyre_force__(self) -> Any`.
- `is_thunk(val: Any) -> bool`, `force_thunk(val: Any) -> Any` *(re-exported
  from `psyre`)*.
- `reject_sentinel(value: Any, *, label: str = 'Stream value') -> None`.
- `tag_key(tag: Any) -> TagKey`, `matches_tag(selector_value: Any, target_tag: Any) -> bool`,
  `format_tag(value: Any, max_len: int | None = None) -> str`,
  `tag_sort_key(item: TagKey) -> tuple[str, str]`.
- `callable_name(fn, default='') -> str`, `unwrap_inst_func(fn) -> Callable`,
  `prepare_iterated_n(n: int, op_name: str = 'iterated op') -> int`.
- `class WhenSelector(is_boolean=True, values=frozenset(), negated=False)` with
  `__call__(tag) -> bool`; helpers `normalize_when_selector`,
  `selector_from_norm`, `is_tautological_selector`,
  `when_selector_matches_const`.
- `class MergeSelector(is_boolean, arms=..., default_index=None)` with
  `__call__(tag) -> int` and `arm_selector(arm_index) -> WhenSelector`.
- `prepare_*` argument normalizers: `prepare_when`, `prepare_whennot`,
  `prepare_merge`, `prepare_switch`, `prepare_fby`, `prepare_pre`,
  `prepare_morph`, `prepare_post`, `prepare_buffer`, `prepare_current`,
  `prepare_counter`, `prepare_prefix_sum`, `prepare_scan`.

### Stage-1 Contexts, Scopes & Control Records

- `ContextVar`s: `active_builder`, `active_scheduler`, `reset_stack`,
  `prefix_stack`, `arm_stack`, `active_block`.
- Scope managers: `push_reset(r_stream)` (aliased as `reset`, *re-exported from
  `psyre`*), `push_prefix(prefix)`, `push_arm(arm)`,
  `activate_block_context(ctx)`, `require_open_arm_context(expected_kind, caller)`.
- Data classes & helpers: `ArmScope`, `remap_arm_scopes`, `CapturedEnv`
  (`.capture()`, `.restore()`), `MatchArm`, `ArmRecord`, `BlockContext`,
  `BlockHandle(Mapping)`.
- Phase/semantics guards: `is_building() -> bool`, `require_builder()`,
  `require_sync_semantics(construct: str = '') -> None` *(re-exported from
  `psyre`)*.

### Runtime Execution & Decorators

- `class RingBuffer(size: int = 1)` with `pull(val)` and `push() -> Any`.
- `async def pull(stream: StreamPort) -> Any`,
  `async def pull_all(*args: StreamPort) -> tuple[Any, ...]`.
- Decorators *(re-exported from `psyre`)*:
  - `psyre_loop(step_func=None, *, prepare=None, stateful=False)`
  - `psyre_comb(func=None, *, prepare=None, pure=False, nout=None, static_argnums=(), static_argnames=(), static_literals=False)`
  - `psyre_sequ(inst_func=None, *, prepare=None, nout=None, static_argnums=(), static_argnames=(), static_literals=False)`
  - `psyre_node(name: str)`
- Step-function helpers & predicates: `make_comb_step_func`, `same_step_func`,
  `emit_comb_node`, `is_pure_step_func` *(re-exported)*,
  `is_builtin_scalar_step_func`, `mark_builtin_scalar_op`,
  `is_stateful_step_func`, `is_step_of`, `is_builtin_primitive_step`,
  `make_primitive`, `emit_when`.

### Primitive Dispatchers (Re-Exported from `psyre`)

All 14 built-in stream combinators dispatch dynamically to the active
`GraphBuilder`'s registered `Semantics` implementation (`sync_primitives` or
`kahn_primitives`):

`buffer`, `counter`, `current`, `fby`, `ifthenelse`, `merge`, `morph`, `post`,
`pre`, `prefix_sum`, `scan`, `switch`, `when`, `whennot`.

## See Also

- [`psyre.payload`](payload.md) — First-order payload validation and `CombCallPlan`
- [`psyre.sync_primitives`](sync_primitives.md) and [`psyre.kahn_primitives`](kahn_primitives.md) — Stage-2 kernel implementations
- [`psyre.builder`](builder.md) — Stage-1 `GraphBuilder` and `lower_resets`
