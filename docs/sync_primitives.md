# `psyre.sync_primitives`

Synchronous (`Semantics.SYNC`) Stage-2 primitive kernels implementing
length-preserving, `Absent`-propagating Lustre / Lucid Synchrone and bounded
lookahead stream combinators.

## Overview & Role

`psyre.sync_primitives` implements all 14 core reactive primitives under
`Semantics.SYNC` and registers itself into `psyre.common.primitives_registry`
at import time via `register_primitives(Semantics.SYNC, sys.modules[__name__])`.
User code normally invokes primitives through top-level `psyre` (which
dispatches dynamically to the active `GraphBuilder`'s semantics); importing
`psyre.sync_primitives` directly is mainly needed by runtime internals or custom
extensions.

Under `Semantics.SYNC`, every stream carries one slot per base-clock tick up to
the execution horizon: active ticks carry present payloads and inactive subclock
ticks carry [`Absent`](common.md#sentinels-absent-bottom-none-unset-and-unready).
Static clock analysis ([`psyre.clock.check_clocks`](clock.md)) and
[`builder.lower_resets`](builder.md#lower_resets) are mandatory under
`Semantics.SYNC`, allowing every kernel here to assume all input ports and
resets satisfy their static clock invariants without ad-hoc unclocked fallback
branches.

## Design & Key Invariants

### Inactive-Tick & Reset Protocol

- Stateful async generators receive `start = yield` from [`Scheduler`](scheduler.md)
  on every base tick (`True`/`False` on active ticks; `Absent` when an
  enclosing reset or activation clock is inactive).
- On `start is Absent`, every stateful kernel pushes `Absent`, advances dynamic
  input ports via `StreamPort.skip_absent()` (leaving clock-polymorphic static
  stream cursors untouched), and preserves internal state unchanged.

### `O(1)` Multi-Stage Delays & Feedback Causality (`fby`, `pre`)

- `fby` and `pre` are decorated with `@psyre_loop(..., stateful=True)` and
  dispatch to a specialized single-stage `fby_step_1` when `n == 1` and a
  general sliding-window `fby_step` when `n >= 2`.
- **`fby_step_1`**: Pushes its held state *before* awaiting `pull(j_stream)`,
  breaking instantaneous feedback cycles.
- **`fby_step` (`n >= 2`)**: Replaces naive `O(n)`/tick stage shifting with an
  **amortized `O(1)`/tick** interlock queue (`j_ticks: deque[bool]`,
  `active_lag`, `init_held`, `state`, and reset fill counter `rem`). It defers
  the first `n - 1` pulls of `j_stream` so multi-stage feedback stays causal,
  and queues `False` entries on `Absent` ticks while `active_lag > 0` so
  `skip_absent()` never overtakes buffered active slots. Its observable trace
  matches `n` chained `fby_step_1` stages identically across `Absent` ticks,
  `None` payloads, and repeated resets.

### Sample-and-Hold Latch (`current`) Prefix Contract

- Static clock checking guarantees `clk(b_stream) == clk(default_stream) ==
  clk(o_stream)` and `clk(b_stream).is_prefix_of(clk(i_stream))`, allowing
  `i_stream` to lie on `b_stream`'s clock or any strictly slower subclock of it.
- Initializes `memory = Bottom`. On `start`, loads `default_stream` into
  `memory` (skipping `default_stream` on subsequent active ticks). When `b_val`
  is truthy, pulls `i_val` and updates `memory = i_val` only when
  `i_val is not Absent` (preserving the held value on slower subclock ticks).
  When `b_val` is falsy, pushes `memory` and skips `i_stream`.

### Strict Bounded FIFO (`buffer`)

- Backed by [`RingBuffer(size)`](common.md#strict-lossless-ringbuffer) (reset to
  a fresh `RingBuffer(size)` on truthy `start`).
- Always executes **`pull` before `push`** on each active tick:
  - When `pull_p` is truthy, writes `await pull(i_stream)` into the ring buffer;
    when falsy, calls `i_stream.skip_absent()` (matching `i_stream`'s subclock
    `clk on i_p_stream`).
  - When `push_p` is truthy, emits `buf.push()`; when falsy, emits `Absent`.
- Raises `BufferError` on overflow or underflow (including when buffering `None`
  payloads).

### Lookahead (`post`) & Reset Rewind

- On `start is Absent`: pushes `Absent` without moving `i_stream`'s lookahead
  cursor.
- On truthy `start` (initial tick or reset): rewinds
  `i_stream.index = o_stream.index` (protected against sliding-window eviction
  by `Scheduler`'s `(in_port, o_port)` watermark), then skips `n` non-`Absent`
  values from `i_stream`.
- On every active tick, pulls forward until the next non-`Absent` element and
  emits it.

### Rate-Changing & Multiplexing Cursor Discipline

- **`when` / `whennot`**: On `cond is Absent`, calls `i_stream.skip_absent()`
  and emits `Absent`; when `cond` fails the predicate, calls `i_stream.skip()`
  and emits `Absent`; when `cond` matches, pulls and emits `i_stream`.
- **`merge`**: Pulls selector `s_stream`; if `Absent`, emits `Absent` and calls
  `skip_absent()` on all arms; otherwise pulls *only* `arm_streams[lookup(tag)]`
  and calls `skip_absent()` on all unselected arms.
- **`switch`**: Pulls selector and *all* arms via `pull_all`; emits `Absent` if
  any input is `Absent`, else emits the selected arm.
- **`ifthenelse`**: `@psyre_comb(pure=True)` evaluating strictly scalar
  `t if c else f` (never array-valued `c`).

### Sequential Accumulators (`counter`, `prefix_sum`, `scan`)

- Implemented via `@psyre_sequ` using out-of-place addition (`+`, never `+=`) so
  mutable payload references are never mutated in place.
- `counter(start=0, max_val=None, tick=True)` accepts a polymorphic boolean
  `tick` enable (holding when falsy, incrementing when truthy) and tracks
  dynamic `max_val` cleanly across `None` ticks.
- `scan(data, op=operator.add)` unwraps `@psyre_comb` operators via
  `unwrap_inst_func` and passes the raw callable as static kwarg `bin_op`.

## Public API Reference

- **Combinational (`@psyre_comb(pure=True)`)**:
  - `ifthenelse(c, t, f)`: Strictly scalar pointwise conditional.
- **Stateless loops (`@psyre_loop`)**:
  - `when(o_stream, c_stream, i_stream, *, selector=None)`
  - `whennot(o_stream, c_stream, i_stream)`
  - `merge(o_stream, s_stream, *arm_streams, lookup)`
  - `switch(o_stream, s_stream, *arm_streams, lookup)`
- **Stateful loops (`@psyre_loop`)**:
  - `buffer(o_stream, i_stream, i_p_stream, o_p_stream, *, size=1)`
  - `fby(o_stream, i_stream, j_stream, *, n=1)` and `pre(o_stream, i_stream, *, n=1)`
  - `morph(o_stream, i_stream, j_stream, *, n=1)`: Emits `i_stream` for `n` active
    ticks after start/reset (skipping `j_stream`), then `j_stream` (skipping
    `i_stream`).
  - `current(o_stream, b_stream, i_stream, default_stream)`
  - `post(o_stream, i_stream, *, n=1)`
- **Delay step coroutines**:
  - `fby_step_1(o_stream, i_stream, j_stream) -> AsyncGenerator[None, Any]`
  - `fby_step(o_stream, i_stream, j_stream, *, n=1) -> AsyncGenerator[None, Any]`
- **Sequential generators (`@psyre_sequ`)**:
  - `counter(init_val, tick_val=True, max_val=UNSET)` (Stage-1 signature:
    `counter(start=0, max_val=None, tick=True)`)
  - `prefix_sum(data_val, init_val)` (Stage-1 signature:
    `prefix_sum(data, start=0)`)
  - `scan(data_val, *, bin_op)` (Stage-1 signature:
    `scan(data, op=operator.add)`)

## See Also

- [`psyre.kahn_primitives`](kahn_primitives.md) — Asynchronous KPN counterparts
- [`psyre.common`](common.md) — Primitive dispatchers, `RingBuffer`, and `prepare_*` validators
- [`psyre.clock`](clock.md) — Static synchronous clock rules enforced on every primitive
