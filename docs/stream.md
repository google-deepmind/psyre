# `psyre.stream`

Unified single-assignment `Stream` channel and Stage-1 DSL handle, built-in
pointwise operators and NumPy NEP-13/NEP-18 auto-lifting, Stage-2 asynchronous
storage with sliding-window eviction, and lightweight `StreamPort` cursors.

## Overview & Role

`Stream` is Psyre's single user-facing abstraction for both Stage-1 dataflow
edges and Stage-2 single-assignment channels. Earlier prototypes split backing
channels (`IStruct`) from handle wrappers (`IStructHandle`); folding both into
`Stream` eliminates redundant unwrapping across every compiler pass while
isolating per-consumer read cursors into lightweight Stage-2 `StreamPort`
objects created only at execution time.

> **Reference & Immutability Contract:** Streams store payloads by reference
> without deep-copying or enforcing runtime immutability. Treat values carried
> on streams as immutable; mutating an object carried on a stream produces
> unspecified behavior.

## Design & Key Invariants

### Dynamic vs. Static Streams & Reset Scoping

- **Dynamic stream (`Stream()`, `static=False`)**: Produced by at most one
  Stage-1 `Node` (single assignment) or supplied as an external channel.
- **Static stream (`Stream(val)`, `Stream.periodic(...)`, `Stream.once(k)`,
  `Stream.constant(val)`)**: Backed by an immutable
  [`StaticSequence`](static_stream.md#staticsequence). Unpinned static streams
  are **uniformly clock-polymorphic (`∀ 'a. 'a`)**—both 1-periodic constants and
  multi-element periodic sequences instantiate at each consumer's clock.
- **No variadic `Stream(a, b)`**: `Stream` accepts `(value=UNSET, *, name='',
  period=UNSET)`. Multi-element sequences must be passed explicitly as a `list`,
  `StaticWord`, or `StaticSequence`.
- **Identity-preserving re-wrapping**: `Stream(existing_stream)` (with
  `period=UNSET`) returns `existing_stream` via `__new__`, recording any
  distinct `name` as an alias in the active `Graph`.
- **Lexical reset lowering**: When a **non-1-periodic** static `Stream` is
  freshly instantiated inside an active `with reset(r):` or `automaton`
  `state(...)` scope (`reset_stack` non-empty), Stage 1 automatically converts
  it into a dynamic stream driven by a stateful, resettable `static_stream` node
  backed by a detached static source. 1-periodic constants remain node-free and
  reset-invariant, and aliasing an existing outer stream via `Stream(outer)`
  inside `reset(...)` does not reset it.

### Pointwise Operators, Indexing & Safety Guards

- `Stream` disables `__hash__` (`__hash__ = None` because `__eq__` builds a
  comparison node; internal compiler tables key by `stream.id: int`) and
  supports runtime generic typing (`Stream[T]`) via `types.GenericAlias`.
- `__bool__`, `__iter__`, `__setitem__`, and `__array__` all raise `TypeError`,
  preventing accidental Python truthiness tests, iteration, in-place mutation,
  or `np.asarray(stream)` boxing into an object array.
- `__getitem__` is `@psyre_comb(pure=True, static_literals=True)`, supporting
  both static and dynamic stream indices/slices.
- Arithmetic (`+`, `-`, `*`, `@`, `/`, `//`, `%`, `**`, `<<`, `>>` and reflected
  variants), bitwise (`&`, `|`, `^` and reflected), comparisons (`<`, `<=`,
  `==`, `!=`, `>`, `>=`), unary (`+`, `-`, `abs`, `~`), and `max`, `min`, `sum`
  are all `@psyre_comb(pure=True)` operators:
  - `__invert__` (`~x`) evaluates `not x` on Python `bool` payloads (avoiding
    Python's `~True == -2` integer quirk) and bitwise `~x` otherwise.
  - `Stream.sum(self, *rest)` folds left with `operator.add` starting from
    `self` (no `int(0)` seed, preserving array dtypes and custom objects).

### NumPy NEP-13 & NEP-18 Auto-Lifting

Without importing `numpy`, `Stream` implements `__array_priority__ = 1000`,
`__array_ufunc__` (NEP-13), and `__array_function__` (NEP-18):

- Rejects mutating APIs (`.at`, `MUTATING_NUMPY_FUNCS`, and non-null `out=`)
  with `TypeError`.
- Defers (`NotImplemented`) when another operand is a `LazyThunk` implementing
  the same protocol (so [`psyre.equations.Thunk`](equations.md#thunk) can defer
  forward references cleanly).
- Unpacks multi-output ufuncs (`ufunc_nout(ufunc, method) > 1`, e.g., `divmod`,
  `modf`, `frexp`) into tuples of projected `Stream`s.
- Marks ufuncs `pure=True` and array functions `pure=is_pure_numpy_func(func)`;
  CSE deduplicates matching lifted nodes via
  [`common.same_step_func`](common.md#node-decorators--cache-free-structural-cse-same_step_func)
  without closure caches.

### Stage-2 Storage, Waiters & Sliding-Window Eviction

During Stage-2 execution, dynamic `Stream`s hold logical slots
`[_seq_base + _head, _seq_base + len(seq))` in `self.seq` (`pushed_count ==
_seq_base + len(seq)`):

- **Private per-waiter futures & `O(1)` wakeup**: Blocked `pull(index)` calls
  register `(future, task)` entries in `_waiters[index]`. When `push(val, idx)`
  writes slot `idx`, it resolves each pending future and immediately calls
  `Scheduler.unblock_task(task, self)` before the next event-loop turn, keeping
  deadlock detection `O(1)` and preventing sibling consumer cancellations from
  interfering.
- **Validation on `push`/`pull`**: `push` rejects static streams
  (`RuntimeError`), `Bottom` (`TypeError`), callables/thunks/nested streams
  (`TypeError`), and out-of-order writes `index != pushed_count`
  (`RuntimeError`). `pull` rejects negative indices (`ValueError`), evicted
  slots (`RuntimeError`), and `Bottom` (`RuntimeError`).
- **Sliding-window slot eviction (`configure_eviction`)**: For unprobed
  intermediate dynamic streams, [`Scheduler`](scheduler.md) registers all active
  readers so consumed slots are cleared and reclaimed in amortized `O(1)` time
  (compacting `del seq[:rel]` with hysteresis once `rel >= 64` and
  `rel * 2 >= len(seq)`):
  - *Zero readers or already-skipped single reader*: `push()` drops values
    immediately without appending to `seq`.
  - *Single monotonic `StreamPort` reader*: `StreamPort.try_pull_ready()`,
    `pull()`, and `skip()` evict inline on every read.
  - *Multi-reader or `post` readers*: `post` registers `(in_port, out_port)`
    tuples whose watermark `min(in_port.index, out_port.index)` protects reset
    rewinds (`i_stream.index = o_stream.index`); `push()` runs strided sweeps
    every `min(max(8, 2 * num_readers), 64)` ticks.
- **Cursor discipline (`StreamPort`)**:
  - `skip()` advances `index` unconditionally (used for discarded active ticks).
  - `skip_absent()` advances `index` **only on dynamic streams**, leaving static
    stream cursors untouched across `Absent` ticks so clock-polymorphic static
    streams advance strictly on active ticks.

## Public API Reference

### `Stream`

Re-exported from top-level `psyre`.

```python
class Stream:
    id: int
    name: str
    static: bool
    static_sequence: StaticSequence | None
    seq: list[Any]

    def __init__(
        self,
        value: Any = UNSET,
        *,
        name: str = '',
        period: Any = UNSET,
    ) -> None
```

- **Class constructors**:
  - `Stream.constant(value: Any, *, name: str = '') -> Stream`: Wraps any
    non-`Stream` first-order value verbatim as a 1-periodic constant without
    unpacking `list` or `StaticWord` (`TypeError` on `Stream`).
  - `Stream.periodic(prefix: Any, period: Any, name: str = '') -> Stream`:
    Constructs an eventually periodic static stream `prefix + period^ω`.
  - `Stream.once(k: int, *, name: str = '') -> Stream`: `O(1)` symbolic boolean
    impulse stream (`True` at active tick `k >= 0`, `False` elsewhere; becomes a
    resettable `static_stream` node inside `with reset(...)`).
  - `Stream.random(**kwargs: Any) -> Stream`: Delegates to
    [`psyre.math.random_stream`](math.md#random_stream).
- **Properties**: `prefix`, `period`, `pushed_count`, `is_one_periodic`,
  `const_value`.
- **Stage-2 & trace methods**:
  - `static_value_at(index: int) -> Any`
  - `trace(horizon: int, *, pad_bottom: bool = False) -> list[Any]`: Returns
    materialized values (`StaticSequence.trace(horizon)` for static streams).
    With `pad_bottom=True`, guards against Stage-1/active Stage-2 execution
    (`RuntimeError`) and `horizon < len(seq)` (`ValueError`), padding trailing
    missing ticks with `Bottom` without mutating `seq`. Raises `RuntimeError` if
    dynamic slots were evicted.
  - `configure_eviction(readers: list[Any]) -> None`, `evict_ready() -> None`,
    `remove_reader(entry: Any) -> None`, `reset() -> None`.
  - `async def pull(index: int) -> Any`, `push(value: Any, index: int) -> None`.
- **Operators & combinators**: `__getitem__`, arithmetic/bitwise/comparison/unary
  dunders, `max(stream, *rest)`, `min(stream, *rest)`, `sum(*rest)`,
  `__array_ufunc__`, `__array_function__`.

### `StreamPort`

```python
class StreamPort:
    stream: Stream
    index: int
```

Lightweight slotted (`__slots__ = ('stream', 'index')`) Stage-2 per-port
read/write cursor:

- `try_pull_ready() -> Any`: Synchronous non-blocking fast path returning the
  value at `index` (evicting immediately on single-reader streams) or `UNREADY`.
- `async def pull() -> Any`: Asynchronous pull advancing `index`.
- `push(value: Any) -> None`: Pushes `value` at `index` and advances `index`.
- `skip() -> None`: Advances `index` unconditionally (discarded active tick).
- `skip_absent() -> None`: Advances `index` only for dynamic streams.

### Functions

- `same_stream_or_const(a: Stream, b: Stream) -> bool`: Returns `True` if
  `a is b` or both are 1-periodic static streams with equal constants
  (`same_const_value`).
- `resolve_stream(stream_or_thunk: Any) -> Stream` *(re-exported from `psyre`)*:
  Forces a single `LazyThunk` or zero-arg function step via `force_thunk`
  (requiring the result to be a `Stream`, else `TypeError`), or wraps a literal
  via `Stream(...)`.

## See Also

- [`psyre.static_stream`](static_stream.md) — `StaticWord`, `StaticSequence`, and `static_stream`
- [`psyre.scheduler`](scheduler.md) — Stage-2 execution and eviction wiring
- [`psyre.trace`](trace.md) — Post-run trace inspection predicates
