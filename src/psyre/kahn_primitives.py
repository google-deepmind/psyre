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
"""Asynchronous (Kahn-style) Psyre node primitives."""

from collections.abc import AsyncGenerator
import random
import sys
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    RingBuffer,
    Semantics,
    prepare_counter,
    prepare_current,
    prepare_delay,
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
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


@psyre_comb
def random_stream():
  """Returns a random float in [0.0, 1.0) at each stream step."""
  return random.random()


@psyre_sequ
def once_stream(k):
  """Returns a resettable stream with a single True value at position k."""
  idx = 0
  while True:
    k = yield idx == k
    idx += 1


@psyre_comb
def ifthenelse(c, t, f):
  """Length-preserving multiplexer."""
  return t if c else f


@psyre_loop(prepare=prepare_when)
async def when(o_stream, c_stream, i_stream, *, selector=None):
  """Downsamples an input stream when the condition stream matches."""
  cond, val = await pull_all(c_stream, i_stream)
  if bool(cond) if selector is None else selector(cond):
    o_stream.push(val)


@psyre_loop
async def whennot(o_stream, c_stream, i_stream):
  """Downsamples an input stream when the condition stream is false."""
  cond, val = await pull_all(c_stream, i_stream)
  if not cond:
    o_stream.push(val)


@psyre_loop(prepare=prepare_merge)
async def merge(o_stream, s_stream, *arm_streams, lookup):
  """Merges complementary streams according to a selector stream."""
  tag = await pull(s_stream)
  o_stream.push(await pull(arm_streams[lookup(tag)]))


@psyre_loop(prepare=prepare_switch)
async def switch(o_stream, s_stream, *arm_streams, lookup):
  """Selects among same-clock streams according to a selector stream."""
  vals = await pull_all(s_stream, *arm_streams)
  o_stream.push(vals[1 + lookup(vals[0])])


@psyre_loop
async def buffer(o_stream, i_stream, i_p_stream, o_p_stream, size=1):
  """Buffer node."""
  buf = RingBuffer(size)
  while True:
    if (yield):
      buf = RingBuffer(size)
    pull_p, push_p = await pull_all(i_p_stream, o_p_stream)
    if pull_p:
      buf.pull(await pull(i_stream))
    if push_p:
      o_stream.push(buf.push())


@psyre_loop(prepare=prepare_delay)
async def delay(o_stream, i_stream, **_kwargs):
  """Identity delay operator in KAHN semantics."""
  while True:
    yield
    o_stream.push(await pull(i_stream))


async def fby_step_1(
    o_stream: Stream,
    i_stream: Stream | None,
    j_stream: Stream,
) -> AsyncGenerator[None, Any]:
  """Delays `j_stream` by 1 tick, initialized from `i_stream` on reset.

  On each tick, pulls `i_stream` (or uses `None` when `i_stream is None`, i.e.,
  `pre`), pushes that value to `o_stream` on start/reset or the previous tick's
  `j_stream` value otherwise, and then pulls the next value from `j_stream`.

  Args:
    o_stream: Output stream receiving the delayed values.
    i_stream: Initial value stream sampled on start/reset, or `None` for `pre`.
    j_stream: Input stream delayed by 1 tick.

  Yields:
    None at each tick while receiving the boolean start/reset flag via `.asend`.
  """
  state: Any = None
  while True:
    start = yield
    init_val = None if i_stream is None else await pull(i_stream)
    if start:
      state = init_val
    o_stream.push(state)
    state = await pull(j_stream)


async def fby_step(
    o_stream: Stream,
    i_stream: Stream | None,
    j_stream: Stream,
    *,
    n: int = 1,
) -> AsyncGenerator[None, Any]:
  """Delays `j_stream` by `n` ticks, initialized from `i_stream` on reset.

  Pulls `i_stream` (or uses `None` for `pre`) every tick and pushes the value
  captured at start or the latest reset to `o_stream` for `n` ticks, then emits
  `j_stream` delayed by `n` ticks (`j[t - n]`). The first `n - 1` warmup ticks
  skip pulling `j_stream` so its read cursor stays `n - 1` ticks behind
  `o_stream` (giving an `n`-tick delay together with the 1-slot `state`).

  Args:
    o_stream: Output stream receiving the delayed values.
    i_stream: Initial value stream sampled on start/reset, or `None` for `pre`.
    j_stream: Input stream delayed by `n` ticks.
    n: Number of delay stages (`n >= 1`).

  Yields:
    None at each tick while receiving the boolean start/reset flag via `.asend`.
  """
  init_held: Any = None
  state: Any = None
  rem = 0
  # Warmup: emit the first n - 1 ticks without pulling j_stream so j_stream's
  # cursor lags o_stream by n - 1 ticks (n ticks total including `state`).
  for _ in range(n - 1):
    start = yield
    init_val = None if i_stream is None else await pull(i_stream)
    if start:
      init_held = init_val
      rem = n
    rem -= 1
    o_stream.push(init_held)
  while True:
    start = yield
    init_val = None if i_stream is None else await pull(i_stream)
    if start:
      init_held = init_val
      rem = n
    if rem > 0:
      rem -= 1
      o_stream.push(init_held)
    else:
      o_stream.push(state)
    state = await pull(j_stream)


async def morph_step(
    o_stream: Stream,
    i_stream: Stream,
    j_stream: Stream,
    *,
    n: int = 1,
) -> AsyncGenerator[None, Any]:
  """Emits `i_stream` for `n` ticks after start/reset, then `j_stream`.

  On each tick, pulls one value from both `i_stream` and `j_stream` (pulling
  the active input first before pushing to `o_stream`) and pushes the value
  from `i_stream` for the first `n` ticks following start or any reset, then
  from `j_stream` on subsequent ticks.

  Args:
    o_stream: Output stream receiving the selected values.
    i_stream: Prefix stream emitted for `n` ticks after start or reset.
    j_stream: Suffix stream emitted on subsequent ticks.
    n: Number of ticks to emit from `i_stream` after each start or reset.

  Yields:
    None at each tick while receiving the boolean start/reset flag via `.asend`.
  """
  rem = n
  while True:
    if (yield):
      rem = n
    if rem > 0:
      rem -= 1
      o_stream.push(await pull(i_stream))
      await pull(j_stream)
    else:
      o_stream.push(await pull(j_stream))
      await pull(i_stream)


@psyre_loop(prepare=prepare_fby)
def fby(o_stream, i_stream, j_stream, *, n: int = 1):
  """Followed-by delay node equivalent to n chained `fby` stages."""
  if n == 1:
    return fby_step_1(o_stream, i_stream, j_stream)
  return fby_step(o_stream, i_stream, j_stream, n=n)


@psyre_loop(prepare=prepare_pre)
def pre(o_stream, i_stream, *, n: int = 1):
  """Stateful delay node equivalent to n chained `pre` stages."""
  if n == 1:
    return fby_step_1(o_stream, None, i_stream)
  return fby_step(o_stream, None, i_stream, n=n)


@psyre_loop(prepare=prepare_morph)
def morph(o_stream, i_stream, j_stream, *, n: int = 1):
  """Initialization operator emitting the first n values of i_stream then j_stream."""
  return morph_step(o_stream, i_stream, j_stream, n=n)


@psyre_loop(prepare=prepare_current)
async def current(o_stream, b_stream, i_stream, default_stream):
  """Holds the most recent value of stream i_stream downsampled by b_stream."""
  memory = None
  while True:
    start = yield
    cond, init = await pull_all(b_stream, default_stream)
    if start:
      memory = init
    if cond:
      memory = await pull(i_stream)
      o_stream.push(memory)
    else:
      o_stream.push(memory)
      await pull(i_stream)


@psyre_sequ(prepare=prepare_counter)
def counter(init_val, max_val=None):
  """Resettable incrementing integer stream starting from init_val."""
  cur = init_val
  while True:
    if max_val is not None:
      cur = min(cur, max_val)
    sent = yield cur
    if max_val is not None:
      _, max_val = sent
    cur += 1


@psyre_sequ(prepare=prepare_prefix_sum)
def prefix_sum(data_val, init_val):
  """Resettable running sum of an input stream starting from init_val."""
  acc = init_val
  while True:
    acc += data_val
    data_val, _ = yield acc


@psyre_sequ(prepare=prepare_scan)
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
    if (yield):
      i_stream.index = o_stream.index
      for _ in range(n):
        await pull(i_stream)
    o_stream.push(await pull(i_stream))


register_primitives(Semantics.KAHN, sys.modules[__name__])
