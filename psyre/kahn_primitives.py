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

import random
import sys

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
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
    k = yield idx == k
    idx += 1


@psyre_comb
def ifthenelse(c, t, f):
  """Length-preserving multiplexer."""
  return t if c else f


@psyre_loop
async def when(o_stream, c_stream, i_stream):
  """Downsamples an input stream when the condition stream is true."""
  cond, val = await Scheduler.pull_all(c_stream, i_stream)
  if cond:
    o_stream.push(val)


@psyre_loop
async def whennot(o_stream, c_stream, i_stream):
  """Downsamples an input stream when the condition stream is false."""
  cond, val = await Scheduler.pull_all(c_stream, i_stream)
  if not cond:
    o_stream.push(val)


@psyre_loop(prepare=prepare_merge)
async def merge(o_stream, s_stream, *arm_streams, lookup):
  """Merges complementary streams according to a selector stream."""
  tag = await Scheduler.pull(s_stream)
  o_stream.push(await Scheduler.pull(arm_streams[lookup(tag)]))


@psyre_loop
async def buffer(o_stream, i_stream, i_p_stream, o_p_stream, size=1):
  """Buffer node."""
  buf = RingBuffer(size)
  while True:
    if (yield):
      buf = RingBuffer(size)
    pull_p, push_p = await Scheduler.pull_all(i_p_stream, o_p_stream)
    if pull_p:
      buf.pull(await Scheduler.pull(i_stream))
    if push_p:
      o_stream.push(buf.push())


@psyre_loop(prepare=prepare_delay)
async def delay(o_stream, i_stream, **_kwargs):
  """Identity delay operator in KAHN semantics."""
  while True:
    yield
    o_stream.push(await Scheduler.pull(i_stream))


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
  memory = None
  while True:
    start = yield
    cond, init = await Scheduler.pull_all(b_stream, default_stream)
    if start:
      memory = init
    if cond:
      memory = await Scheduler.pull(i_stream)
      o_stream.push(memory)
    else:
      o_stream.push(memory)
      await Scheduler.pull(i_stream)


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
        await Scheduler.pull(i_stream)
    o_stream.push(await Scheduler.pull(i_stream))


register_primitives(Semantics.KAHN, sys.modules[__name__])
