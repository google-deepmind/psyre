# `psyre.trace`

Post-run inspection utilities for querying presence, absence, and post-horizon
bottom padding on materialized trace sequences.

## Overview & Role

After Stage-2 execution completes, [`Scheduler.run`](scheduler.md#scheduler) and
[`Scheduler.run_graph`](scheduler.md#scheduler) return Python lists (or nested
containers of lists) representing per-tick stream traces. Under
`Semantics.SYNC`, a trace entry can be:

- a **present payload value** (including `None`, `False`, `0`, or `''`),
- **`Absent`** (an inactive subclock tick), or
- **`Bottom`** (a trailing slot past early finite-horizon EOF caused by `post`
  lookahead).

`psyre.trace` provides identity-safe inspection helpers that never confuse
`None` or falsy payloads with `Absent` or `Bottom`, and prevent accidental use
during Stage-1 elaboration or active Stage-2 execution. All four functions are
re-exported from top-level `psyre`.

## Design & Key Invariants

- **Strict post-run phase guard**: Every helper raises `RuntimeError` if called
  while `active_builder` is set (Stage 1) or while `active_scheduler.is_active_run()`
  is `True` (Stage 2). Trace inspection is strictly a post-run operation.
- **Strict sequence & index validation**:
  - Rejects `str`, `bytes`, `bytearray` (despite being Python `Sequence`s) and
    non-`Sequence` objects such as live `Stream` instances with `TypeError`.
  - Rejects `bool` indices with `TypeError` and normalizes integer indices via
    `operator.index(index)` (supporting negative indices and integer-like types
    such as `np.int64`, while rejecting `slice` and `float` with `TypeError` and
    out-of-range indices with `IndexError`).
- **Identity comparisons**: Uses `is` / `is not` against `Absent` and `Bottom`
  so custom payload `__eq__` methods (such as NumPy/JAX arrays) are never
  invoked and `None` / `False` / `0` are treated as present values.

## Public API Reference

All functions below are re-exported from top-level `psyre`.

### `is_present`

```python
def is_present(trace: Sequence[Any], index: int) -> bool
```

Returns `True` in `O(1)` time when `trace[index]` is neither `Absent` nor
`Bottom` (ordinary `None` payloads return `True`).

### `is_absent`

```python
def is_absent(trace: Sequence[Any], index: int) -> bool
```

Returns `True` in `O(1)` time when `trace[index] is Absent`.

### `is_bottom`

```python
def is_bottom(trace: Sequence[Any], index: int) -> bool
```

Returns `True` in `O(1)` time when `trace[index] is Bottom`.

### `present_values`

```python
def present_values(trace: Sequence[Any]) -> list[Any]
```

Returns a new list containing all elements of `trace` that are neither `Absent`
nor `Bottom` (`O(len(trace))`). Useful for comparing `Semantics.SYNC` and
`Semantics.KAHN` traces under `Absent`/`Bottom` elision.

## See Also

- [`psyre.common`](common.md#sentinels-absent-bottom-none-unset-and-unready) — `Absent` and `Bottom` sentinel semantics
- [`psyre.scheduler`](scheduler.md) — Stage-2 execution and trace extraction
- [`psyre.stream`](stream.md) — `Stream.trace(horizon, *, pad_bottom=False)`
