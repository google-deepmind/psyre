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
"""Asynchronous (Kahn-style) Psyre node primitives.

Implements deterministic asynchronous Kahn Process Network (KPN) node kernels
without `Absent` tokens (Kahn, *IFIP* 1974; Kahn & MacQueen, *IFIP* 1977). On
well-clocked lowered graphs (`psyre.clock` + `builder.lower_resets`), outputs
agree with `psyre.sync_primitives` after eliding `Absent`/`Bottom` sentinels.
"""

import sys

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    RingBuffer,
    Semantics,
    prepare_buffer,
    prepare_merge,
    prepare_when,
    psyre_loop,
    pull,
    pull_all,
    register_primitives,
)
# Re-export primitives shared with synchronous semantics: `KAHN` streams never
# emit `Absent`, so every `is Absent` check in these `sync_primitives` kernels
# is a never-taken pointer-identity compare. Under `KAHN`, `lower_resets` lifts
# any strictly slower `current` data input via nested `merge` + presence gating
# before Stage-2 execution.
from psyre.sync_primitives import (
    counter,
    current,
    fby,
    fby_step,
    fby_step_1,
    ifthenelse,
    morph,
    post,
    pre,
    prefix_sum,
    scan,
    switch,
)

# pylint: enable=g-importing-member,g-multiple-import

__all__ = (
    'buffer',
    'counter',
    'current',
    'fby',
    'fby_step',
    'fby_step_1',
    'ifthenelse',
    'merge',
    'morph',
    'post',
    'pre',
    'prefix_sum',
    'scan',
    'switch',
    'when',
    'whennot',
)


@psyre_loop(prepare=prepare_when)
async def when(o_stream, c_stream, i_stream, *, selector=None):
  """Downsamples an input stream when the condition stream matches."""
  cond = await pull(c_stream)
  if bool(cond) if selector is None else selector(cond):
    o_stream.push(await pull(i_stream))
  else:
    i_stream.skip()


@psyre_loop
async def whennot(o_stream, c_stream, i_stream):
  """Downsamples an input stream when the condition stream is false."""
  cond = await pull(c_stream)
  if not cond:
    o_stream.push(await pull(i_stream))
  else:
    i_stream.skip()


@psyre_loop(prepare=prepare_merge)
async def merge(o_stream, s_stream, *arm_streams, lookup):
  """Merges complementary streams according to a selector stream."""
  tag = await pull(s_stream)
  o_stream.push(await pull(arm_streams[lookup(tag)]))


@psyre_loop(prepare=prepare_buffer)
async def buffer(o_stream, i_stream, i_p_stream, o_p_stream, *, size=1):
  """Bounded lossless FIFO buffer (`pull` before `push`, checks `Bottom`)."""
  buf = RingBuffer(size)
  while True:
    if (yield):
      buf = RingBuffer(size)
    pull_p, push_p = await pull_all(i_p_stream, o_p_stream)
    if pull_p:
      buf.pull(await pull(i_stream))
    if push_p:
      o_stream.push(buf.push())


register_primitives(Semantics.KAHN, sys.modules[__name__])
