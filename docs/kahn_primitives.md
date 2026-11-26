# `psyre.kahn_primitives`

Asynchronous Kahn Process Network (`Semantics.KAHN`) Stage-2 primitive kernels
operating on dense, `Absent`-free token streams.

## Overview & Role

`psyre.kahn_primitives` provides the `Semantics.KAHN` implementations of Psyre's
14 reactive primitives and registers itself at import time via
`register_primitives(Semantics.KAHN, sys.modules[__name__])`.

Under `Semantics.KAHN`, streams carry only present tokens—rate-changing
primitives (`when`, `whennot`, `merge`, `buffer`) alter the number of emitted
tokens rather than writing `Absent` placeholders. When a well-clocked graph is
lowered by [`builder.lower_resets`](builder.md#lower_resets) (handling ancestor
modular resets, subclocked polymorphic static ports, and subclocked `current()`
inputs), `Semantics.KAHN` and `Semantics.SYNC` produce identical traces after
eliding `Absent` and `Bottom` sentinels (`present_values`).

## Design & Key Invariants

### Shared Kernels via Direct Re-Export

12 of the 16 exported symbols are **re-exported directly from
[`psyre.sync_primitives`](sync_primitives.md)** with zero duplication:

`counter`, `current`, `fby`, `fby_step`, `fby_step_1`, `ifthenelse`, `morph`,
`post`, `pre`, `prefix_sum`, `scan`, and `switch`.

- **Why sharing is sound and fast**: Under `Semantics.KAHN`, no primitive ever
  emits `Absent`, so every `is Absent` guard in the shared `sync_primitives`
  kernels is an untaken pointer-identity check (`O(1)`), and discarded active
  ticks use `.skip()` uniformly across both semantics so finite-horizon `post()`
  lengths match.
- **How `current` works on subclocks under `KAHN`**: Because `KAHN` streams have
  no `Absent` tokens, [`builder.lower_resets`](builder.md#lower_resets) lifts
  any strictly subclocked `current(b, i, default)` data input `i` up to `b`'s
  clock via nested complementary `merge` nodes plus a presence gate *before*
  Stage-2 execution, so the shared `sync_primitives.current` kernel sees a
  same-clock dense input.

### The 4 KPN-Specific Primitives

Only the 4 rate-changing primitives require bespoke `KAHN` implementations
because they change token counts instead of emitting `Absent`:

- **`when`**: Pulls `cond`; if matched, pulls and pushes `i_stream`; otherwise
  calls `i_stream.skip()` and emits **nothing**.
- **`whennot`**: Pulls `cond`; if `not cond`, pulls and pushes `i_stream`;
  otherwise calls `i_stream.skip()` and emits **nothing**.
- **`merge`**: Pulls selector `tag`, then pulls and pushes **only**
  `arm_streams[lookup(tag)]`. Unselected arm streams are never pulled or skipped
  because they carry no `Absent` placeholders.
- **`buffer`**: Backed by [`RingBuffer(size)`](common.md#strict-lossless-ringbuffer)
  (reset on truthy `start`). Pulls `(pull_p, push_p)` and enforces strict
  pull-before-push (`BufferError` on overflow/underflow): if `pull_p` is truthy,
  pulls `i_stream` into the ring buffer (does *not* skip `i_stream` when falsy);
  if `push_p` is truthy, emits `buf.push()` (emits **nothing** when falsy).

## Public API Reference

All 16 symbols in `psyre.kahn_primitives.__all__`:

- **KPN-specific implementations**:
  - `when(o_stream, c_stream, i_stream, *, selector=None)`
  - `whennot(o_stream, c_stream, i_stream)`
  - `merge(o_stream, s_stream, *arm_streams, lookup)`
  - `buffer(o_stream, i_stream, i_p_stream, o_p_stream, *, size=1)`
- **Re-exported from [`psyre.sync_primitives`](sync_primitives.md)**:
  - `counter`, `current`, `fby`, `fby_step`, `fby_step_1`, `ifthenelse`,
    `morph`, `post`, `pre`, `prefix_sum`, `scan`, `switch`.

## See Also

- [`psyre.sync_primitives`](sync_primitives.md) — Synchronous `Absent`-propagating kernels
- [`psyre.builder`](builder.md#lower_resets) — `lower_resets` and `requires_clock_analysis` for `KAHN`
- [`psyre.scheduler`](scheduler.md) — Stage-2 KPN coroutine scheduling and EOF detection
