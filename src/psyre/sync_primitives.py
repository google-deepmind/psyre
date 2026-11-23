# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synchronous (Lustre-style) length-preserving Psyre node primitives.

Implements base operators from Lustre (`when`, `current`, `pre`, `fby`,
`morph`; Halbwachs et al., *Proc. IEEE* 1991), Lucid Synchrone / modular reset
(`merge`, tagged `when`, reset hooks; Caspi & Pouzet, *ICFP* 1996; Hamon &
Pouzet, *PPDP* 2000), and bounded reverse-time lookahead (`post`, `buffer`;
Potop-Butucaru, Cohen, Plotkin & Pompougnac, 2023, arXiv:2311.16977).
"""

import collections
from collections.abc import AsyncGenerator
import sys
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    Absent,
    Bottom,
    RingBuffer,
    Semantics,
    UNSET,
    prepare_buffer,
    prepare_counter,
    prepare_current,
    prepare_fby,
    prepare_merge,
    prepare_morph,
    prepare_post,
    prepare_pre,
    prepare_prefix_sum,
    prepare_scan,
    prepare_switch,
    prepare_when,
    psyre_comb,
    psyre_loop,
    psyre_sequ,
    pull,
    pull_all,
    register_primitives,
)
from psyre.stream import StreamPort

# pylint: enable=g-importing-member,g-multiple-import


@psyre_comb(pure=True)
def ifthenelse(c, t, f):
  """Length-preserving strictly scalar multiplexer (`t if c else f`).

  Evaluates `c` as a Python scalar truth value. For elementwise selection over
  NumPy or JAX boolean arrays, use `np.where`, `jnp.where`, or `jax.lax.select`
  instead.
  """
  return t if c else f


@psyre_loop(prepare=prepare_when)
async def when(o_stream, c_stream, i_stream, *, selector=None):
  """Downsamples an input stream, emitting Absent when condition fails."""
  cond = await pull(c_stream)
  if cond is Absent:
    o_stream.push(Absent)
    i_stream.skip_absent()
  elif bool(cond) if selector is None else selector(cond):
    o_stream.push(await pull(i_stream))
  else:
    o_stream.push(Absent)
    i_stream.skip()


@psyre_loop
async def whennot(o_stream, c_stream, i_stream):
  """Downsamples an input stream, emitting Absent when condition is true."""
  cond = await pull(c_stream)
  if cond is Absent:
    o_stream.push(Absent)
    i_stream.skip_absent()
  elif not cond:
    o_stream.push(await pull(i_stream))
  else:
    o_stream.push(Absent)
    i_stream.skip()


@psyre_loop(prepare=prepare_merge)
async def merge(o_stream, s_stream, *arm_streams, lookup):
  """Merges synchronous streams according to a selector stream."""
  tag = await pull(s_stream)
  sel = None if tag is Absent else lookup(tag)
  o_stream.push(
      Absent if sel is None else await pull(arm_streams[sel])
  )
  for idx, arm_stream in enumerate(arm_streams):
    if idx != sel:
      arm_stream.skip_absent()


@psyre_loop(prepare=prepare_switch)
async def switch(o_stream, s_stream, *arm_streams, lookup):
  """Selects among same-clock synchronous streams using a selector stream."""
  vals = await pull_all(s_stream, *arm_streams)
  o_stream.push(
      Absent
      if any(val is Absent for val in vals)
      else vals[1 + lookup(vals[0])]
  )


@psyre_loop(prepare=prepare_buffer)
async def buffer(o_stream, i_stream, i_p_stream, o_p_stream, *, size=1):
  """Length-preserving bounded FIFO buffer emitting `Absent` when not pushing.

  Executes `pull` before `push` on each active tick, backed by `RingBuffer`
  with empty slots initialized/cleared to `Bottom`. Raises `BufferError` on
  overflow (pull into an occupied slot) or underflow (push from an empty slot).
  """
  buf = RingBuffer(size)
  while True:
    start = yield
    if start is Absent:
      i_stream.skip_absent()
      i_p_stream.skip_absent()
      o_p_stream.skip_absent()
      o_stream.push(Absent)
      continue
    pull_p, push_p = await pull_all(i_p_stream, o_p_stream)
    if start:
      buf = RingBuffer(size)
    if pull_p:
      buf.pull(await pull(i_stream))
    else:
      i_stream.skip_absent()
    o_stream.push(buf.push() if push_p else Absent)


async def fby_step_1(
    o_stream: StreamPort,
    i_stream: StreamPort | None,
    j_stream: StreamPort,
) -> AsyncGenerator[None, Any]:
  """Delays `j_stream` by 1 active tick, initialized from `i_stream`.

  When the start/reset signal is `Absent`, skips `i_stream` (if present),
  pushes `Absent` to `o_stream`, and skips `j_stream` without updating state.
  Otherwise, loads `i_stream` (or `None` for `pre`) on start/reset, pushes the
  held state to `o_stream`, and updates state from `j_stream`.

  Args:
    o_stream: Output stream receiving the delayed values (or `Absent`).
    i_stream: Initial value stream sampled on start/reset, or `None` for `pre`.
    j_stream: Input stream delayed by 1 active tick.

  Yields:
    None at each tick while receiving the start/reset flag (`bool` or `Absent`).
  """
  state: Any = Bottom
  while True:
    start = yield
    if start is Absent:
      if i_stream is not None:
        i_stream.skip_absent()
      o_stream.push(Absent)
      j_stream.skip_absent()
      continue
    if start:
      state = None if i_stream is None else await pull(i_stream)
    elif i_stream is not None:
      i_stream.skip()
    o_stream.push(state)
    state = await pull(j_stream)


async def fby_step(
    o_stream: StreamPort,
    i_stream: StreamPort | None,
    j_stream: StreamPort,
    *,
    n: int = 1,
) -> AsyncGenerator[None, Any]:
  """Delays `j_stream` by `n` active ticks, initialized from `i_stream`.

  Equivalent to `n` chained synchronous 1-stage `fby`/`pre` delays sharing the
  same `i_stream` (or `None` for `pre`) and reset signal, preserving `Absent`
  ticks without corrupting stage state and deferring the first `n - 1` pulls of
  `j_stream`.

  Args:
    o_stream: Output stream receiving the delayed values (or `Absent`).
    i_stream: Initial value stream sampled on start/reset, or `None` for `pre`.
    j_stream: Input stream delayed by `n` active ticks.
    n: Number of delay stages (`n >= 1`).

  Yields:
    None at each tick while receiving the start/reset flag (`bool` or `Absent`).
  """
  # O(1)-per-tick equivalent of `n` chained `fby_step_1` stages:
  # - `j_ticks`: interlock queue (True = active tick, False = deferred `Absent`
  #   tick) so `j_stream.skip_absent()` stays behind the `n - 1` active-tick
  #   lag and never overtakes buffered active slots.
  # - `active_lag`: count of `True` entries currently queued in `j_ticks`.
  # - `init_held`/`rem`: active reset fill counter emitting `init_held` for `n`
  #   active ticks after start or any reset before releasing `state`.
  # - `state`: tail register emitted once `rem == 0`.
  j_ticks: collections.deque[bool] = collections.deque()
  active_lag = 0
  init_held: Any = Bottom
  state: Any = Bottom
  rem = 0

  while True:
    start = yield
    if start is Absent:
      if i_stream is not None:
        i_stream.skip_absent()
      o_stream.push(Absent)
      if active_lag:
        # Defer the `Absent` skip until buffered active ticks drain past `j`.
        j_ticks.append(False)
      else:
        j_stream.skip_absent()
      continue
    if start:
      init_held = None if i_stream is None else await pull(i_stream)
      rem = n
    elif i_stream is not None:
      i_stream.skip()
    if rem > 0:
      rem -= 1
      o_stream.push(init_held)
    else:
      o_stream.push(state)
    j_ticks.append(True)
    active_lag += 1
    if active_lag == n:
      # Lag of `n - 1` active ticks is full: advance `state` by one `j` tick
      # and drain any contiguous deferred `Absent` ticks immediately behind it.
      j_ticks.popleft()
      active_lag -= 1
      state = await pull(j_stream)
      while j_ticks and not j_ticks[0]:
        j_ticks.popleft()
        j_stream.skip_absent()


@psyre_loop(prepare=prepare_fby, stateful=True)
def fby(o_stream, i_stream, j_stream, *, n: int = 1):
  """Followed-by delay node equivalent to n chained `fby` stages."""
  if n == 1:
    return fby_step_1(o_stream, i_stream, j_stream)
  return fby_step(o_stream, i_stream, j_stream, n=n)


@psyre_loop(prepare=prepare_pre, stateful=True)
def pre(o_stream, i_stream, *, n: int = 1):
  """Stateful delay node equivalent to n chained `pre` stages."""
  if n == 1:
    return fby_step_1(o_stream, None, i_stream)
  return fby_step(o_stream, None, i_stream, n=n)


@psyre_loop(prepare=prepare_morph)
async def morph(
    o_stream: StreamPort,
    i_stream: StreamPort,
    j_stream: StreamPort,
    *,
    n: int = 1,
) -> AsyncGenerator[None, Any]:
  """Emits `i_stream` for `n` active ticks after start/reset, then `j_stream`."""
  rem = n
  while True:
    start = yield
    if start is Absent:
      o_stream.push(Absent)
      i_stream.skip_absent()
      j_stream.skip_absent()
      continue
    if start:
      rem = n
    if rem > 0:
      rem -= 1
      o_stream.push(await pull(i_stream))
      j_stream.skip()
    else:
      o_stream.push(await pull(j_stream))
      i_stream.skip()


@psyre_loop(prepare=prepare_current)
async def current(o_stream, b_stream, i_stream, default_stream):
  """Sample-and-hold latch projecting `i_stream` onto `b_stream`'s clock.

  Enforced statically by mandatory `psyre.clock.check_clocks`:
  `clk(b_stream) == clk(default_stream) == clk(o_stream)` and
  `clk(b_stream).is_prefix_of(clk(i_stream))`.
  """
  memory: Any = Bottom
  while True:
    start = yield
    if start is Absent:
      b_stream.skip_absent()
      i_stream.skip_absent()
      default_stream.skip_absent()
      o_stream.push(Absent)
      continue
    b_val = await pull(b_stream)
    if start:
      memory = await pull(default_stream)
    else:
      default_stream.skip()
    if b_val:
      i_val = await pull(i_stream)
      if i_val is not Absent:
        memory = i_val
      o_stream.push(memory)
    else:
      o_stream.push(memory)
      i_stream.skip()


@psyre_sequ(prepare=prepare_counter)
def counter(init_val, tick_val=True, max_val=UNSET):
  """Resettable integer counter enabled by `tick_val` from `init_val`."""
  has_max = max_val is not UNSET
  cur = init_val
  while True:
    if has_max and max_val is not None:
      cur = min(cur, max_val)
    out = cur
    if tick_val:
      cur = cur + 1
    sent = yield out
    if has_max:
      _, tick_val, max_val = sent
    else:
      _, tick_val = sent


@psyre_sequ(prepare=prepare_prefix_sum)
def prefix_sum(data_val, init_val):
  """Resettable running sum of an input stream starting from init_val."""
  acc = init_val
  while True:
    acc = acc + data_val
    data_val, _ = yield acc


@psyre_sequ(prepare=prepare_scan, static_argnames=('bin_op',))
def scan(data_val, *, bin_op):
  """Applies binary stream operator bin_op cumulatively across data_val."""
  acc = data_val
  while True:
    data_val = yield acc
    acc = bin_op(acc, data_val)


@psyre_loop(prepare=prepare_post)
async def post(o_stream, i_stream, *, n: int = 1):
  """Post node equivalent to n chained `post` stages."""
  while True:
    start = yield
    if start is Absent:
      # Inactive tick: emit `Absent` without advancing `i_stream` (its ahead
      # cursor already stays parked at or ahead of the next active index).
      o_stream.push(Absent)
      continue
    if start:
      # Rewind lookahead cursor to the output index on start/reset and drop
      # `n` active ticks (`Scheduler` tracks `(in_port, o_port)` watermarks so
      # slot `o_stream.index` is never evicted before reset completes).
      i_stream.index = o_stream.index
      for _ in range(n):
        while (await pull(i_stream)) is Absent:
          pass
    while (i_val := await pull(i_stream)) is Absent:
      pass
    o_stream.push(i_val)


register_primitives(Semantics.SYNC, sys.modules[__name__])
