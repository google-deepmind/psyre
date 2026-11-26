# `psyre.static_stream`

Symbolic repetition-concatenation finite-word algebra (`StaticWord`),
eventually periodic infinite sequences (`StaticSequence`), exact constant
equivalence, and the resettable Stage-2 `static_stream` kernel.

## Overview & Role

Static streams in Psyre represent infinite, eventually periodic sequences of the
form `prefix + period^ω`. Naively expanding repeated patterns (such as
`Stream.once(10**9)` or long padded testbench schedules) into Python lists would
consume `O(k)` time and memory. `psyre.static_stream` backs every static
[`Stream`](stream.md) with a compact symbolic grammar that keeps construction,
indexing, slicing, and iteration efficient regardless of repetition counts.

This module also centralizes:

- `same_const_value`: exact type-, value-, and signed-zero-preserving constant
  equivalence shared across static normalization, `Stream`, and
  [`psyre.optimization`](optimization.md).
- `static_stream`: the resettable Stage-2 async-generator coroutine emitted when
  non-1-periodic static streams appear inside `with reset(...)` scopes or
  subclocked [`lower_resets`](builder.md#lower_resets) lowerings.

## Design & Key Invariants

### Symbolic Word Grammar (`StaticWord`)

Finite words follow the immutable, slotted, unhashable (`__hash__ = None`)
algebraic data type `S ::= w | S * k | S1 + S2` (`k >= 0` non-bool `int`), with
total word length capped at `MAX_WORD_LENGTH = 2**31 - 1` (`ValueError` on
overflow):

- **`LiteralWord(items)`**: stores an immutable `tuple` of validated atoms
  (`EMPTY_WORD = LiteralWord(())`).
- **`RepeatWord(inner, count_n)`**: always normalized to `count_n >= 2` and
  non-empty `inner`. Repeating by `0` yields `EMPTY_WORD`, repeating by `1`
  returns `inner`, and nested repeats collapse multiplicatively
  (`(S * a) * b -> S * (a * b)`).
- **`ConcatWord(parts, offsets)`**: always flattened to `>= 2` non-empty child
  words (never nested `ConcatWord`s), caching cumulative `offsets` for
  `O(log |parts|)` binary-search indexing via `bisect_right`.

#### Normalization & Boundary Fusion (`concat`)

`concat(*parts)` flattens nested concatenations and fuses adjacent parts:

- Merges adjacent `LiteralWord + LiteralWord`.
- Peels leading/trailing literal atoms matching an adjacent single-atom
  `RepeatWord` via `same_const_value` (e.g. `repeat(False, 5) + [False]` folds to
  `repeat(False, 6)`).
- Merges adjacent `RepeatWord(S1, c1) + RepeatWord(S2, c2)` when structurally
  equal (`_struct_eq(S1, S2)`).

#### Complexity & DAG Safety

- `StaticWord` intentionally does **not** inherit from
  `collections.abc.Sequence` to avoid accidental `O(length)` fallback scans.
- `value_at(index)` runs in `O(depth * log(branching))` without expanding
  repeats.
- `slice_word(start, stop)` and `word[start:stop]` split `RepeatWord`s
  symbolically via `divmod` (`head + repeat(inner, full) + tail`). Only unit
  step (`step is None` or `1`) is supported (`ValueError` otherwise).
- `iter_atoms()`, `uniform_value()`, `StaticSequence.is_bool()`, and structural
  equality (`__eq__` via memoized `_struct_eq`) all run in **`O(|DAG|)`** time,
  preventing exponential blowup on shared word DAGs.
- **Structural equality trade-off**: `StaticWord.__eq__` compares normalized
  DAG structure (after boundary fusion) rather than running full word-equivalence
  recompression, keeping equality simple and fast `O(|DAG|)` while accepting
  that differently factored multi-element repeats may compare unequal.
- `+`, `*`, and reflected `*` return `NotImplemented` when the other operand is
  a `Stream` or `LazyThunk` so `Stream` / `Thunk` operators take precedence.
- All atoms are validated with `reject_sentinel` and
  `reject_callable(..., reject_streams=True)` (`TypeError`). Passing a `list` to
  `StaticWord(val)` unpacks its elements; pass `StaticWord.atom(val)` to keep a
  `list` as a single element.

### Exact Constant Equivalence (`same_const_value`)

`same_const_value(av, bv)` defines when two static values may be coalesced by
normalization or compiler folding:

- Rejects `NaN` (always `False`, even if `av is bv`).
- Requires exact Python type equality (`type(av) is type(bv)`), distinguishing
  `True` vs `1` vs `1.0`.
- Compares `tuple` recursively, `frozenset` via [`payload.exact_value_key`](payload.md#exact-value-keys--signed-zero-safety),
  and 0-D non-object `np.ndarray` via `.item()` after verifying matching shape
  and dtype.
- Compares **mutable containers** (`list`, `set`, `Mapping`, `bytearray`,
  `deque`) and **non-0-D, non-NumPy (e.g. `jax.Array`), or object-dtype arrays**
  strictly by **object identity (`is`)**, preventing the compiler from aliasing
  distinct mutable references or triggering device transfers.
- Compares all other scalars via `safe_scalar_eq` and `zero_sign_key`
  (separating `-0.0` from `+0.0` on floats and complex numbers).

### `StaticSequence` Canonicalization

`StaticSequence(prefix, period)` requires `period.length > 0` (`ValueError`) and
canonicalizes three invariants on construction:

1. Unwraps outer `RepeatWord` wrappers on `period` (`(S * k)^ω = S^ω`).
2. Collapses any uniform multi-element `period` to a length-1 `LiteralWord`.
3. When `period.length == 1`, trims all trailing matching atoms from `prefix`
   via DAG-memoized `_trim_trailing` so `[v, v] + [v]^ω` normalizes to
   1-periodic `EMPTY_WORD + [v]^ω`.

`StaticSequence.once(k)` builds `concat(repeat(False, k), True)` followed by
`LiteralWord((False,))` in **`O(1)`** time and space, even for `k = 10**9`.
`StaticSequence.from_parts(prefix, period)` validates `period` before `prefix`
so an unresolved `LazyThunk` in `period` reports before any `Stream` prefix
error.

### Resettable `static_stream` Stage-2 Kernel

`static_stream(o_stream, *, source)` is an async generator tagged with
`_psyre_static_stream = True` (recognized by `is_static_stream_step`). On
`start is Absent`, it pushes `Absent` without advancing its cursor; on truthy
`start` (first active tick or reset), it rewinds to `source.static_sequence.iter_values()`,
achieving amortized `O(1)` per tick.

## Public API Reference

### Constants

- `MAX_WORD_LENGTH = (1 << 31) - 1`: Maximum permitted `StaticWord` length.
- `EMPTY_WORD: StaticWord`: Canonical empty `LiteralWord(())` *(re-exported from
  `psyre`)*.

### Classes

#### `StaticWord`, `LiteralWord`, `RepeatWord`, `ConcatWord`

`StaticWord` is re-exported from top-level `psyre`.

- `StaticWord(value=UNSET) -> StaticWord`: Coerces omitted → `EMPTY_WORD`,
  `StaticWord` → unchanged, `list` → unpacked `LiteralWord`, any other value
  (including `tuple`) → length-1 `LiteralWord`.
- `StaticWord.atom(value, *, label='Static word') -> StaticWord`: Wraps any
  validated value (including a `list` or `StaticWord`) as a single length-1
  atom.
- Properties / methods: `.length`, `__len__()`, `value_at(index)`, `__iter__()`,
  `slice_word(start, stop)`, `__getitem__(index_or_slice)`, `iter_atoms()`,
  `uniform_value() -> tuple[bool, Any]`, `__contains__(value)`,
  `format_word(*, item_fmt=..., max_items=None, parenthesize_concat=False)`,
  `+`, `*`, reflected `*`, `==`.
- Variant attributes: `LiteralWord.items`, `RepeatWord.inner` / `.count_n`,
  `ConcatWord.parts` / `.offsets`.

#### `StaticSequence`

Re-exported from top-level `psyre`.

- `StaticSequence(prefix: StaticWord, period: StaticWord)`
- Class constructors:
  - `from_parts(prefix, period) -> StaticSequence`
  - `from_value(value) -> StaticSequence`: Stutters the last element of a
    non-empty `list` or `StaticWord`, or creates a 1-periodic scalar constant.
  - `constant(value) -> StaticSequence`: Wraps any validated first-order value
    verbatim as a 1-periodic constant (never unpacking `list` or `StaticWord`).
  - `once(k: int) -> StaticSequence`: `O(1)` symbolic impulse at tick `k >= 0`.
- Properties & methods:
  - `prefix: StaticWord`, `period: StaticWord`
  - `is_one_periodic: bool`, `const_value: Any` (`RuntimeError` if not
    1-periodic)
  - `same_constant(other: StaticSequence) -> bool`
  - `value_at(index: int) -> Any` (`ValueError` if `index < 0`)
  - `iter_values() -> Iterator[Any]`, `trace(horizon: int) -> list[Any]`
  - `iter_atoms() -> Iterator[Any]`, `is_bool() -> bool`
  - `format_compact(*, max_tag_len=10, max_items=4) -> str`, `==`.

### Functions

- `same_const_value(av: Any, bv: Any) -> bool`
- `repeat(value: Any, count: int, *, label: str = 'Static word') -> StaticWord`
  *(re-exported from `psyre`)*
- `concat(*parts: Any, label: str = 'Static word') -> StaticWord`
  *(re-exported from `psyre`)*
- `is_static_stream_step(step_func: Any) -> bool`
- `async def static_stream(o_stream: Any, *, source: Any) -> AsyncGenerator[None, Any]`

## See Also

- [`psyre.stream`](stream.md) — `Stream` wrapper, `Stream.periodic`, `Stream.once`, `Stream.constant`
- [`psyre.clock`](clock.md) — Uniform clock polymorphism (`∀ 'a. 'a`) of static streams
- [`psyre.builder`](builder.md) — `lower_resets` lowering of subclocked static ports
