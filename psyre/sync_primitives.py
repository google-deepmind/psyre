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

import random
import sys

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    Absent,
    RingBuffer,
    Semantics,
    fby_step,
    morph_step,
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
)
from psyre.psyre import (
    Scheduler,
    psyre_comb,
    psyre_loop,
    psyre_sequ,
    register_primitives,
)

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


@psyre_loop
async def when(o_stream, c_stream, i_stream):
  """Downsamples an input stream, emitting Absent when condition is false."""
  if await Scheduler.pull(c_stream):
    o_stream.push(await Scheduler.pull(i_stream))
  else:
    o_stream.push(Absent)
    await Scheduler.pull(i_stream)


@psyre_loop
async def whennot(o_stream, c_stream, i_stream):
  """Downsamples an input stream, emitting Absent when condition is true."""
  cond = await Scheduler.pull(c_stream)
  if cond is not Absent and not cond:
    o_stream.push(await Scheduler.pull(i_stream))
  else:
    o_stream.push(Absent)
    await Scheduler.pull(i_stream)


@psyre_loop(prepare=prepare_merge)
async def merge(o_stream, s_stream, *arm_streams, lookup):
  """Merges synchronous streams according to a selector stream."""
  tag = await Scheduler.pull(s_stream)
  sel = None if tag is Absent else lookup(tag)
  o_stream.push(
      Absent if sel is None else await Scheduler.pull(arm_streams[sel])
  )
  for idx, arm_stream in enumerate(arm_streams):
    if idx != sel:
      await Scheduler.pull(arm_stream)


@psyre_loop
async def buffer(o_stream, i_stream, i_p_stream, o_p_stream, size=1):
  """Length-preserving buffer node emitting Absent when not pushing."""
  buf = RingBuffer(size)
  while True:
    start = yield
    i_val, pull_p, push_p = await Scheduler.pull_all(
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
  buf = RingBuffer(size)
  while True:
    start = yield
    if start is Absent:
      await Scheduler.pull(i_stream)
      o_stream.push(Absent)
      continue
    if start:
      p_stream.index = 0
      buf = RingBuffer(size)
    i_val = await Scheduler.pull(i_stream)
    if i_val is not Absent:
      buf.pull(i_val)
    o_stream.push(buf.push() if await Scheduler.pull(p_stream) else Absent)


@psyre_loop(prepare=prepare_fby)
def fby(o_stream, i_stream, j_stream, *, n: int = 1):
  """Followed-by delay node equivalent to n chained `fby` stages."""
  return fby_step(o_stream, i_stream, j_stream, n=n)


@psyre_loop(prepare=prepare_pre)
def pre(o_stream, i_stream, *, n: int = 1):
  """Stateful delay node equivalent to n chained `pre` stages."""
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
    b_val, init = await Scheduler.pull_all(b_stream, default_stream)
    if start is Absent:
      o_stream.push(Absent)
      await Scheduler.pull(i_stream)
      continue
    if start and init is not Absent:
      memory = init
    if b_val is Absent:
      o_stream.push(Absent)
      await Scheduler.pull(i_stream)
    elif b_val:
      i_val = await Scheduler.pull(i_stream)
      if i_val is not Absent:
        memory = i_val
      o_stream.push(i_val)
    else:
      o_stream.push(memory)
      await Scheduler.pull(i_stream)


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
    if start:
      i_stream.index = o_stream.index
      need_drop = True
    if need_drop:
      if (await Scheduler.pull(i_stream)) is Absent:
        o_stream.push(Absent)
        continue
      for _ in range(n - 1):
        while (await Scheduler.pull(i_stream)) is Absent:
          pass
      need_drop = False
    elif start is Absent:
      o_stream.push(Absent)
      continue
    while (i_val := await Scheduler.pull(i_stream)) is Absent:
      pass
    o_stream.push(i_val)


register_primitives(Semantics.SYNC, sys.modules[__name__])
