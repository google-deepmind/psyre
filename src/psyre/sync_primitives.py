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
"""Synchronous (Lustre-style) length-preserving Psyre node primitives."""

from collections.abc import AsyncGenerator
import random
import sys
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    Absent,
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
    if k is Absent:
      out = Absent
    else:
      out = idx == k
      idx += 1
    k = yield out


@psyre_comb
def ifthenelse(c, t, f):
  """Length-preserving multiplexer."""
  return t if c else f


@psyre_loop(prepare=prepare_when)
async def when(o_stream, c_stream, i_stream, *, selector=None):
  """Downsamples an input stream, emitting Absent when condition fails."""
  cond = await pull(c_stream)
  active = cond is not Absent and (
      bool(cond) if selector is None else selector(cond)
  )
  if active:
    o_stream.push(await pull(i_stream))
  else:
    o_stream.push(Absent)
    await pull(i_stream)


@psyre_loop
async def whennot(o_stream, c_stream, i_stream):
  """Downsamples an input stream, emitting Absent when condition is true."""
  cond = await pull(c_stream)
  if cond is not Absent and not cond:
    o_stream.push(await pull(i_stream))
  else:
    o_stream.push(Absent)
    await pull(i_stream)


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
      await pull(arm_stream)


@psyre_loop(prepare=prepare_switch)
async def switch(o_stream, s_stream, *arm_streams, lookup):
  """Selects among same-clock synchronous streams using a selector stream."""
  vals = await pull_all(s_stream, *arm_streams)
  o_stream.push(Absent if Absent in vals else vals[1 + lookup(vals[0])])


@psyre_loop
async def buffer(o_stream, i_stream, i_p_stream, o_p_stream, size=1):
  """Length-preserving buffer node emitting Absent when not pushing."""
  buf = RingBuffer(size)
  while True:
    start = yield
    i_val, pull_p, push_p = await pull_all(
        i_stream, i_p_stream, o_p_stream
    )
    if start is Absent:
      o_stream.push(Absent)
      continue
    if start:
      buf = RingBuffer(size)
    if pull_p:
      buf.pull(i_val)
    o_stream.push(buf.push() if push_p else Absent)


@psyre_loop(prepare=prepare_delay)
async def delay(o_stream, i_stream, *, p_stream, size: int):
  """Delays i_stream by k local ticks or along a static periodic schedule."""
  p_handle = Stream(p_stream)
  buf = RingBuffer(size)
  while True:
    start = yield
    if start is Absent:
      await pull(i_stream)
      o_stream.push(Absent)
      continue
    if start:
      p_handle.index = 0
      buf = RingBuffer(size)
    i_val = await pull(i_stream)
    if i_val is not Absent:
      buf.pull(i_val)
    o_stream.push(buf.push() if await pull(p_handle) else Absent)


async def fby_step_1(
    o_stream: Stream,
    i_stream: Stream | None,
    j_stream: Stream,
) -> AsyncGenerator[None, Any]:
  """Delays `j_stream` by 1 active tick, initialized from `i_stream`.

  When the start/reset signal is `Absent`, pulls `i_stream` (if present),
  pushes `Absent` to `o_stream`, and pulls `j_stream` without updating state.
  Otherwise, loads a non-`Absent` `i_stream` value (or `None` for `pre`) on
  start/reset, pushes the held state to `o_stream`, and updates state from any
  non-`Absent` value pulled from `j_stream`.

  Args:
    o_stream: Output stream receiving the delayed values (or `Absent`).
    i_stream: Initial value stream sampled on start/reset, or `None` for `pre`.
    j_stream: Input stream delayed by 1 active tick.

  Yields:
    None at each tick while receiving the start/reset flag (`bool` or `Absent`).
  """
  state: Any = Absent
  while True:
    start = yield
    init_val = None if i_stream is None else await pull(i_stream)
    if start is Absent:
      o_stream.push(Absent)
      await pull(j_stream)
      continue
    if start and init_val is not Absent:
      state = init_val
    o_stream.push(state)
    in_val = await pull(j_stream)
    if in_val is not Absent:
      state = in_val


async def fby_step(
    o_stream: Stream,
    i_stream: Stream | None,
    j_stream: Stream,
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
  stages: list[Any] = [Absent] * n
  pending: list[tuple[Any, Any]] = []

  def emit(m: int, s_m: Any, i_m: Any) -> Any:
    if s_m is Absent:
      return Absent
    if s_m and i_m is not Absent:
      stages[m] = i_m
    return stages[m]

  def update(m: int, s_m: Any, val: Any) -> None:
    if s_m is not Absent and val is not Absent:
      stages[m] = val

  while True:
    start = yield
    init_val = None if i_stream is None else await pull(i_stream)
    if start is Absent and not pending:
      o_stream.push(Absent)
      await pull(j_stream)
      continue
    pending.append((start, init_val))
    if len(pending) < n:
      prev: Any = Absent
      for m in range(len(pending) - 1, n):
        out = emit(m, start, init_val)
        update(m, start, prev)
        prev = out
      o_stream.push(prev)
    else:
      o_stream.push(emit(n - 1, start, init_val))
      in_val = await pull(j_stream)
      s_0, _ = pending.pop(0)
      update(0, s_0, in_val)
      for m, (s_m, i_m) in enumerate(pending):
        update(m + 1, s_m, emit(m, s_m, i_m))


async def morph_step(
    o_stream: Stream,
    i_stream: Stream,
    j_stream: Stream,
    *,
    n: int = 1,
) -> AsyncGenerator[None, Any]:
  """Emits `i_stream` for `n` active ticks after start/reset, then `j_stream`.

  On each tick, pulls one value from both `i_stream` and `j_stream`. When the
  received reset/start signal is `Absent`, pushes `Absent` to `o_stream`
  without advancing the `n`-tick counter. Otherwise, pushes `i_stream` (pulled
  first) for the first `n` active ticks after start or any reset, and
  `j_stream` (pulled first) thereafter.

  Args:
    o_stream: Output stream receiving the selected values (or `Absent`).
    i_stream: Prefix stream emitted for `n` active ticks after start or reset.
    j_stream: Suffix stream emitted on subsequent active ticks.
    n: Number of active ticks to emit from `i_stream` after start or reset.

  Yields:
    None at each tick while receiving the start/reset flag (`bool` or `Absent`).
  """
  rem = n
  while True:
    start = yield
    if start is Absent:
      o_stream.push(Absent)
      await pull_all(i_stream, j_stream)
      continue
    if start:
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
  memory = Absent
  while True:
    start = yield
    b_val, init = await pull_all(b_stream, default_stream)
    if start is Absent:
      o_stream.push(Absent)
      await pull(i_stream)
      continue
    if start and init is not Absent:
      memory = init
    if b_val is Absent:
      o_stream.push(Absent)
      await pull(i_stream)
    elif b_val:
      i_val = await pull(i_stream)
      if i_val is not Absent:
        memory = i_val
      o_stream.push(i_val)
    else:
      o_stream.push(memory)
      await pull(i_stream)


@psyre_sequ(prepare=prepare_counter)
def counter(init_val, max_val=None):
  """Resettable incrementing integer stream starting from init_val."""
  cur = init_val
  while True:
    if cur is Absent or max_val is Absent:
      out = Absent
    else:
      if max_val is not None:
        cur = min(cur, max_val)
      out = cur
      cur += 1
    sent = yield out
    if max_val is not None:
      _, max_val = sent


@psyre_sequ(prepare=prepare_prefix_sum)
def prefix_sum(data_val, init_val):
  """Resettable running sum of an input stream starting from init_val."""
  acc = init_val
  while True:
    if acc is Absent or data_val is Absent:
      out = Absent
    else:
      acc += data_val
      out = acc
    data_val, _ = yield out


@psyre_sequ(prepare=prepare_scan)
def scan(data_val, *, bin_op):
  """Applies binary stream operator bin_op cumulatively across data_val."""
  acc = data_val
  data_val = yield acc
  while True:
    if acc is Absent or data_val is Absent:
      out = Absent
    else:
      acc = bin_op(acc, data_val)
      out = acc
    data_val = yield out


@psyre_loop(prepare=prepare_post)
async def post(o_stream, i_stream, *, n: int = 1):
  """Post node equivalent to n chained `post` stages."""
  need_drop = True
  while True:
    start = yield
    if start is Absent:
      o_stream.push(Absent)
      continue
    if start:
      i_stream.index = o_stream.index
      need_drop = True
    if need_drop:
      if (await pull(i_stream)) is Absent:
        o_stream.push(Absent)
        continue
      for _ in range(n - 1):
        while (await pull(i_stream)) is Absent:
          pass
      need_drop = False
    while (i_val := await pull(i_stream)) is Absent:
      pass
    o_stream.push(i_val)


register_primitives(Semantics.SYNC, sys.modules[__name__])
