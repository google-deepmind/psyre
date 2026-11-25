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
"""Unit tests for synchronous (Lustre) Psyre stream primitives and execution."""

import asyncio
from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import io
import logging
import math
import operator
import types
from typing import Any

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre import common
from psyre import scheduler
from psyre import sync_primitives
from psyre.clock import ClockError, check_clocks
from psyre.common import (
    Absent,
    Bottom,
    Semantics,
    buffer,
    counter,
    current,
    fby,
    ifthenelse,
    logger,
    merge,
    morph,
    post,
    pre,
    prefix_sum,
    psyre_comb,
    psyre_loop,
    psyre_sequ,
    pull,
    reset,
    scan,
    switch,
    when,
    whennot,
)
from psyre.match import (
    case,
    match,
)
from psyre.math import (
    exp,
    tanh,
)
from psyre.scheduler import Scheduler
from psyre.stream import Stream, StreamPort
from tests import test_utils
from psyre.trace import (
    is_absent,
    is_bottom,
    is_present,
    present_values,
)

# pylint: enable=g-importing-member,g-multiple-import


class SyncTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def test_operators_absent_propagation(self):
    def inner():
      c = Stream([True, False, True, False])
      s = when(c, [2, 0, 4, 0])
      b = when(c, [True, False, False, True])
      rhs = when(c, [3, 5, 1, 1])
      return {
          'add': s + rhs,
          'rsub': 10 - s,
          'neg': -s,
          'invert_bool': ~b,
          'eq': s == 4,
          'rlt': 3 < s,
      }

    res = Scheduler.run(inner, horizon=4, semantics=Semantics.SYNC)
    self.assertEqual(res['add'], [5, Absent, 5, Absent])
    self.assertEqual(res['rsub'], [8, Absent, 6, Absent])
    self.assertEqual(res['neg'], [-2, Absent, -4, Absent])
    self.assertEqual(res['invert_bool'], [False, Absent, True, Absent])
    self.assertEqual(res['eq'], [False, Absent, True, Absent])
    self.assertEqual(res['rlt'], [False, Absent, True, Absent])

  def test_ifthenelse_absent(self):
    def inner():
      c = Stream([True, False, True, False])
      cond_absent = when(c, [True, False, False, False])
      t_branch = when(c, [1, 2, 3, 4])
      f_branch = when(c, [10, 20, 30, 40])
      return (
          ifthenelse(cond_absent, t_branch, f_branch),
          ifthenelse(cond_absent, 1, 30),
      )

    ite_absent, ite_const = Scheduler.run(
        inner, horizon=4, semantics=Semantics.SYNC
    )
    self.assertEqual(ite_absent, [1, Absent, 30, Absent])
    self.assertEqual(ite_const, [1, Absent, 30, Absent])

  def test_scan_absent(self):
    def inner():
      cond1 = Stream([True, False, True, False])
      cond2 = Stream([False, True, False, True])
      with match(cond1):
        with case(True):
          scan([1, 3], op=operator.add, name='sparse_tail')
        with case(False):
          pass
      with match(cond2):
        with case(True):
          scan([1, 3], op=operator.add, name='with_absent')
          scan([3, 5], op=Stream.max, name='max_absent')
        with case(False):
          pass

    res = Scheduler.run(
        inner,
        horizon=4,
        semantics=Semantics.SYNC,
        streams=('sparse_tail', 'with_absent', 'max_absent'),
    )
    self.assertEqual(res['sparse_tail'], [1, Absent, 4, Absent])
    self.assertEqual(res['with_absent'], [Absent, 1, Absent, 4])
    self.assertEqual(res['max_absent'], [Absent, 3, Absent, 5])

  def test_when_whennot_and_merge(self):
    def inner():
      cond = Stream([True, False, False, True, False])
      inp = [10, 20, 30, 40, 50]
      true_branch = when(cond, [1, 0, 0, 2, 0])
      false_branch = whennot(cond, [0, 10, 20, 0, 30])
      c_outer = Stream([True, False, True, True, True])
      sel = when(c_outer, ['a', 'x', 'b', 'a', 'z'])
      typed_sel = [True, 1, False, 0, None]
      return {
          'when': when(cond, inp),
          'whennot': whennot(cond, inp),
          'whennot_absent_cond': whennot(
              when(c_outer, [True, False, False, True, False]), 7
          ),
          'merge': merge(cond, true_branch, false_branch),
          'multi_merge': merge(
              sel,
              {
                  'a': [1, 2],
                  'b': [10],
              },
              default=[-1],
          ),
          'bool_none_merge': merge(
              typed_sel,
              {
                  True: [10],
                  False: [30],
                  None: [50],
              },
              default=[-1, -2],
          ),
          'int_merge': merge(
              typed_sel,
              {
                  1: [20],
                  0: [40],
              },
              default=[-1, -2, -3],
          ),
      }

    res = Scheduler.run(inner, horizon=5, semantics=Semantics.SYNC)
    self.assertEqual(res['when'], [10, Absent, Absent, 40, Absent])
    self.assertEqual(res['whennot'], [Absent, 20, 30, Absent, 50])
    self.assertEqual(res['whennot_absent_cond'], [Absent, Absent, 7, Absent, 7])
    self.assertEqual(res['merge'], [1, 10, 20, 2, 30])
    self.assertEqual(res['multi_merge'], [1, Absent, 10, 2, -1])
    self.assertEqual(res['bool_none_merge'], [10, -1, 30, -2, 50])
    self.assertEqual(res['int_merge'], [-1, 20, -2, 40, -3])

  def test_fby_absent(self):
    def inner():
      c = Stream([True, False, True, True])
      with match(c):
        with case(True):
          fby([10, 30, 40], [1, 3, 4], name='f_out')
          fby([10, 30, 40], [1, 3, 4], 2, name='f2_out')
          pre([10, 30, 40], 2, name='p2_out')
        with case(False):
          pass

    res = Scheduler.run(
        inner,
        horizon=4,
        semantics=Semantics.SYNC,
        streams=('f_out', 'f2_out', 'p2_out'),
    )
    self.assertEqual(res['f_out'], [10, Absent, 1, 3])
    self.assertEqual(res['f2_out'], [10, Absent, 10, 1])
    self.assertEqual(res['p2_out'], [None, Absent, None, 10])

  def test_buffer(self):
    def inner():
      inp = [10, 20, 30, 40]
      inp_p = [True, True, False, False, True, False, True]
      out_p = [False, False, True, True, False, True, True]
      return buffer(inp, inp_p, out_p, size=2)

    size_two = Scheduler.run(inner, horizon=7, semantics=Semantics.SYNC)
    self.assertEqual(
        size_two, [Absent, Absent, 10, 20, Absent, 30, 40]
    )

    # Valid `None` stream values flow cleanly through RingBuffer (distinct from
    # empty `Bottom` slots).
    def none_payload():
      inp = [None, 20, None, 40]
      inp_p = [True, True, False, False, True, False, True]
      out_p = [False, False, True, True, False, True, True]
      return buffer(inp, inp_p, out_p, size=2)

    self.assertEqual(
        Scheduler.run(none_payload, horizon=7, semantics=Semantics.SYNC),
        [Absent, Absent, None, 20, Absent, None, 40],
    )

    # Overflow and underflow fail fast with `BufferError` (full matrix including
    # `None` slots and resets is verified across SYNC/KAHN in pervasives_test).
    for prog, horizon, pattern in (
        (
            lambda: buffer([10, 20], [True, True], [False, False], size=1),
            2,
            'RingBuffer overflow',
        ),
        (
            lambda: buffer([10], [False], [True], size=1),
            1,
            'RingBuffer underflow',
        ),
    ):
      with self.subTest(pattern=pattern):
        test_utils.assert_buffer_error(
            self, prog, horizon, pattern, semantics=Semantics.SYNC
        )

  def test_once_stream_absent(self):
    def inner():
      gate = [True, False, True, True, False]
      with match(gate) as gate_m:
        with case(True):
          o1_act = Stream.once(1, name='o1_act')
          gate_m.out = o1_act
        with case(False):
          gate_m.out = False
      return gate_m.out

    merged = Scheduler.run(inner, horizon=5, semantics=Semantics.SYNC)
    # Without an enclosing reset, `Stream.once(1)` is a clock-polymorphic static
    # stream: its direct trace is the unclocked dense sequence, while `merged`
    # advances it only on `gate == True` ticks.
    o1_act = Scheduler.run(
        inner, horizon=5, semantics=Semantics.SYNC, streams=('o1_act',)
    )['o1_act']
    self.assertEqual(o1_act, [False, True, False, False, False])
    self.assertEqual(merged, [False, False, True, False, False])

    # Under an enclosing `reset`, `Stream.once(1)` emits a stateful
    # `static_stream` node whose SYNC trace carries `Absent` on inactive ticks.
    def inner_resettable():
      gate = Stream([True, False, True, True, False, True, True])
      rst = Stream([False, False, False, True, False, False, False])
      with reset(rst):
        with match(gate) as gate_m:
          with case(True):
            o1_rst = Stream.once(1, name='o1_rst')
            gate_m.out = o1_rst
          with case(False):
            gate_m.out = False
      return gate_m.out

    o1_rst = Scheduler.run(
        inner_resettable, horizon=7, semantics=Semantics.SYNC,
        streams=('o1_rst',),
    )['o1_rst']
    self.assertEqual(
        o1_rst, [False, Absent, True, False, Absent, True, False]
    )
    self.assertEqual(
        Scheduler.run(inner_resettable, horizon=7, semantics=Semantics.SYNC),
        [False, False, True, False, False, True, False],
    )

  def test_post(self):
    def inner():
      inp = [10, 20, 30, 40]
      gate = [False, True, True, False]
      gate2 = [False, True, True, True]
      with match(gate):
        with case(True):
          post([10, 20], name='p_abs')
        with case(False):
          pass
      with match(gate2):
        with case(True):
          post([10, 20, 30], 2, name='pn2_abs')
        with case(False):
          pass
      return (
          post(inp),
          post(post(inp)),
          post(inp, 2),
      )

    p, pp, pn2 = Scheduler.run(inner, horizon=4, semantics=Semantics.SYNC)
    abs_res = Scheduler.run(
        inner,
        horizon=4,
        semantics=Semantics.SYNC,
        streams=('p_abs', 'pn2_abs'),
    )
    self.assertEqual(p, [20, 30, 40, 40])
    self.assertEqual(pp, [30, 40, 40, Bottom])
    self.assertEqual(pn2, [30, 40, 40, 40])
    self.assertEqual(abs_res['p_abs'], [Absent, 20, Bottom, Bottom])
    self.assertEqual(abs_res['pn2_abs'], [Absent, 30, Bottom, Bottom])

  def test_morph_absent(self):
    def inner():
      cond = Stream([False, True, False, True, True])
      with match(cond) as morph_m:
        with case(True):
          m_act = morph([10, 30, 50], [2, 4, 6], name='m_act')
          morph([10, 30, 50], [2, 4, 6], 2, name='m2_act')
          morph_m.out = m_act
        with case(False):
          morph_m.out = 0
      return morph_m.out

    m_merged = Scheduler.run(inner, horizon=5, semantics=Semantics.SYNC)
    act_res = Scheduler.run(
        inner,
        horizon=5,
        semantics=Semantics.SYNC,
        streams=('m_act', 'm2_act'),
    )
    self.assertEqual(act_res['m_act'], [Absent, 10, Absent, 4, 6])
    self.assertEqual(act_res['m2_act'], [Absent, 10, Absent, 30, 6])
    self.assertEqual(m_merged, [0, 10, 0, 4, 6])

  def test_reset_sync_and_supersampled(self):
    def inner():
      inp = Stream([10, 20, 30, 40, 50])
      init = Stream([100, 200, 300, 400, 500])
      delayed = Stream([1, 2, 3, 4, 5])
      cond = Stream([True, False, True, False, True])
      rst = Stream([False, False, True, False, False])
      super_reset = Stream([False, True, False, False, False])

      with reset(rst):
        post(inp, name='post_r')
        when(cond, fby(init, delayed), name='when_fby_r')

      # Supersampled reset triggers at step 1 while cond is False, resetting
      # nested stateful nodes before the next True tick of cond at step 2.
      with reset(super_reset):
        when(cond, counter(start=1), name='super_when_counter')
        with match(cond):
          with case(True):
            fby(init, delayed, name='super_when_fby')
            post(inp, name='super_post_when')
            scan(inp, op=operator.add, name='super_scan_when')
          with case(False):
            pass

    res = Scheduler.run(
        inner,
        horizon=5,
        semantics=Semantics.SYNC,
        streams=(
            'post_r',
            'when_fby_r',
            'super_when_fby',
            'super_when_counter',
            'super_post_when',
            'super_scan_when',
        ),
    )
    self.assertEqual(res['post_r'], [20, 30, 40, 50, 50])
    self.assertEqual(res['when_fby_r'], [100, Absent, 300, Absent, 4])
    self.assertEqual(res['super_when_fby'], [100, Absent, 200, Absent, 2])
    self.assertEqual(res['super_when_counter'], [1, Absent, 2, Absent, 4])
    self.assertEqual(
        res['super_post_when'], [20, Absent, 30, Absent, Bottom]
    )
    self.assertEqual(res['super_scan_when'], [10, Absent, 20, Absent, 50])

  def test_nested_match_and_reset(self):
    def inner():
      outer_gate = Stream([True, False, True, True, False, True])
      # Both `inner_gate` and `inner_reset` are dense static sequences aligned
      # to active ticks of `outer_gate` (indices 0..3 -> base ticks 0, 2, 3, 5).
      inner_gate = Stream([True, False, True, True])
      outer_reset = Stream([False, True, False, False, False, False])
      inner_reset = Stream([False, True, False, False])

      with reset(outer_reset):
        with match(outer_gate):
          with case(True):
            counter(start=10, name='gated')
            with reset(inner_reset):
              with match(inner_gate):
                with case(True):
                  counter(start=100, name='nested')
                with case(False):
                  pass
          with case(False):
            pass

    res = Scheduler.run(
        inner,
        horizon=6,
        semantics=Semantics.SYNC,
        streams=('gated', 'nested'),
    )
    self.assertEqual(res['gated'], [10, Absent, 10, 11, Absent, 12])
    self.assertEqual(res['nested'], [100, Absent, Absent, 100, Absent, 101])

  def test_post_run_presence_helpers(self):
    self.assertFalse(hasattr(common, 'present'))
    self.assertFalse(hasattr(scheduler, 'present'))
    self.assertFalse(hasattr(sync_primitives, 'present'))

    def inner():
      cond = Stream([False, True, True, False, True, True])
      vals = Stream([99, False, 0, 99, None, []])
      return when(cond, vals)

    trace = Scheduler.run(
        inner, horizon=6, semantics=Semantics.SYNC, clock_calculus=True
    )
    self.assertEqual(trace, [Absent, False, 0, Absent, None, []])
    expected_present = [False, True, True, False, True, True]
    for idx, expected in enumerate(expected_present):
      self.assertEqual(is_present(trace, idx), expected)
      self.assertEqual(is_absent(trace, idx), not expected)
      self.assertFalse(is_bottom(trace, idx))
    self.assertTrue(is_present(trace, -1))
    self.assertTrue(is_absent(trace, -3))
    self.assertEqual(present_values(trace), [False, 0, None, []])
    self.assertEqual(present_values(tuple(trace)), [False, 0, None, []])

    bottom_trace = [1, Absent, Bottom]
    self.assertTrue(is_present(bottom_trace, 0))
    self.assertFalse(is_present(bottom_trace, 1))
    self.assertFalse(is_present(bottom_trace, 2))
    self.assertTrue(is_absent(bottom_trace, 1))
    self.assertFalse(is_absent(bottom_trace, 2))
    self.assertFalse(is_bottom(bottom_trace, 0))
    self.assertFalse(is_bottom(bottom_trace, 1))
    self.assertTrue(is_bottom(bottom_trace, 2))
    self.assertEqual(present_values(bottom_trace), [1])

    with self.assertRaises(IndexError):
      is_present(trace, 6)
    with self.assertRaises(IndexError):
      is_absent(trace, 6)
    with self.assertRaises(IndexError):
      is_bottom(trace, 6)

    bad_traces: tuple[Any, ...] = (Stream([1, 2]), 'abc', b'abc', 123)
    for bad_trace in bad_traces:
      with self.assertRaisesRegex(TypeError, 'post-run trace Sequence'):
        is_present(bad_trace, 0)
      with self.assertRaisesRegex(TypeError, 'post-run trace Sequence'):
        is_absent(bad_trace, 0)
      with self.assertRaisesRegex(TypeError, 'post-run trace Sequence'):
        is_bottom(bad_trace, 0)
      with self.assertRaisesRegex(TypeError, 'post-run trace Sequence'):
        present_values(bad_trace)

    for fn in (
        lambda: is_present([1], 0),
        lambda: is_absent([1], 0),
        lambda: is_bottom([1], 0),
        lambda: present_values([1]),
    ):
      with self.assertRaisesRegex(RuntimeError, 'outside a Scheduler run'):
        Scheduler.run(fn, horizon=1, semantics=Semantics.SYNC)

    @psyre_comb
    def illegal_stage_two_probe(val):
      is_present([val], 0)
      return val

    def failing_with_blocked_consumer():
      bad = illegal_stage_two_probe(Stream([1]))
      return bad + 1

    with self.assertRaises(ExceptionGroup) as stage_two_err:
      Scheduler.run(
          failing_with_blocked_consumer,
          horizon=2,
          semantics=Semantics.SYNC,
      )
    self.assertFalse(Scheduler.deadlocked)
    self.assertIsInstance(stage_two_err.exception.exceptions[0], RuntimeError)
    self.assertRegex(
        str(stage_two_err.exception.exceptions[0]), 'outside a Scheduler run'
    )

  def test_max_min_sync_and_absent(self):
    def inner():
      cond = Stream([True, False, True, True])
      sparse = when(cond, [-3, 0, 4, -1])
      other = when(cond, [2, 5, -2, -5])
      scalar_min = Stream.min(2, -5)
      self.assertIsInstance(scalar_min, Stream)
      with match([True, False, True, False]):
        with case(True):
          gated_max = Stream([2, -2]).max(0)
          gated_max.name = 'gated_max'
        with case(False):
          pass
      return scalar_min, sparse.max(0), sparse.min(other)

    scalar_min, max_sparse, min_sparse = Scheduler.run(
        inner, horizon=4, semantics=Semantics.SYNC
    )
    gated_max = Scheduler.run(
        inner, horizon=4, semantics=Semantics.SYNC, streams=('gated_max',)
    )['gated_max']
    self.assertEqual(scalar_min, [-5, -5, -5, -5])
    self.assertEqual(max_sparse, [0, Absent, 4, 0])
    self.assertEqual(min_sparse, [-3, Absent, -2, -5])
    self.assertEqual(gated_max, [2, Absent, 0, Absent])

  def test_deadlock_and_local_progress(self):
    def instantaneous_cycle_with_progress():
      a = ifthenelse(True, lambda: b, 0)
      b = a + 1
      healthy = prefix_sum(counter(start=1))
      return a, b, healthy

    a_out, b_out, healthy_out = Scheduler.run(
        instantaneous_cycle_with_progress,
        horizon=5,
        semantics=Semantics.SYNC,
        causality_analysis=False,
    )
    self.assertTrue(Scheduler.deadlocked)
    self.assertEqual(a_out, [])
    self.assertEqual(b_out, [])
    self.assertEqual(healthy_out, [1, 3, 6, 10, 15])

    def delayed_cycle():
      trigger = Stream([False, False, True, False, False])
      x = merge(
          trigger,
          when(trigger, lambda: x + 1),  # pylint: disable=undefined-variable
          whennot(trigger, counter(start=10)),
      )
      return x

    x_out = Scheduler.run(
        delayed_cycle,
        horizon=5,
        semantics=Semantics.SYNC,
        causality_analysis=False,
    )
    self.assertTrue(Scheduler.deadlocked)
    self.assertEqual(x_out, [10, 11])

  def test_math_primitives_and_stream_sum(self):
    def prog():
      c = Stream([True, False, True, True])
      a = Stream([0.0, 1.0, -0.5, 2.0])
      b = Stream([1.0, 2.0, 3.0, 4.0])
      d = Stream([10.0, 20.0, 30.0, 40.0])
      return (
          exp(a),
          tanh(a),
          Stream.sum(a, b, d),
          a.sum(b, d, 0.5),
          Stream.sum(when(c, a), when(c, b), when(c, d)),
      )

    exp_out, tanh_out, sum3, sum4, sum_sub = Scheduler.run(prog, horizon=4)
    xs = [0.0, 1.0, -0.5, 2.0]
    self.assertSequenceAlmostEqual(
        exp_out, [math.exp(v) for v in xs], places=12
    )
    self.assertSequenceAlmostEqual(
        tanh_out, [math.tanh(v) for v in xs], places=12
    )
    self.assertSequenceAlmostEqual(sum3, [11.0, 23.0, 32.5, 46.0], places=12)
    self.assertSequenceAlmostEqual(sum4, [11.5, 23.5, 33.0, 46.5], places=12)
    self.assertEqual(sum_sub, [11.0, Absent, 32.5, 46.0])

  def test_bottom_singleton_isolation_and_post_padding(self):
    self.assertFalse(bool(Bottom))
    self.assertNotEqual(Bottom, Bottom)
    self.assertNotEqual(Bottom, Absent)
    self.assertNotEqual(Bottom, None)
    self.assertEqual(repr(Bottom), 'Bottom')
    self.assertEqual(str(Bottom), 'Bottom')

    self.assertFalse(hasattr(Stream(), 'pad'))
    self.assertFalse(hasattr(Stream(), 'pad_to'))
    self.assertFalse(hasattr(Stream(), 'keep'))
    self.assertFalse(hasattr(Stream(), 'capture'))

    static_s = Stream([1, 2])
    self.assertEqual(static_s.trace(4, pad_bottom=True), [1, 2, 2, 2])

    async def check_stream_guards() -> None:
      dyn = Stream()
      with self.assertRaisesRegex(TypeError, 'Bottom cannot be pushed'):
        dyn.push(Bottom, 0)
      with self.assertRaisesRegex(RuntimeError, 'Out-of-order push'):
        dyn.push(1, 1)
      dyn.push(10, 0)
      with self.assertRaisesRegex(ValueError, 'pushed count 1'):
        dyn.trace(0, pad_bottom=True)
      # Bottom padding happens only on the returned trace; seq stays unmutated.
      padded = dyn.trace(3, pad_bottom=True)
      self.assertEqual(padded[0], 10)
      self.assertTrue(is_bottom(padded, 1))
      self.assertTrue(is_bottom(padded, 2))
      self.assertEqual(dyn.trace(3), [10])
      self.assertEqual(await dyn.pull(0), 10)

      # Per-pull waiter isolation: cancelling one waiter never cancels or blocks
      # a sibling waiter on the same slot (even if push races before unwinding).
      shared = Stream()
      with self.assertRaisesRegex(ValueError, 'non-negative'):
        await shared.pull(-1)
      w1 = asyncio.create_task(shared.pull(0))
      w2 = asyncio.create_task(shared.pull(0))
      await asyncio.sleep(0)
      w1.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await w1
      shared.push(42, 0)
      self.assertEqual(await w2, 42)
      # Synchronous push right after cancel (before cancelled task unwinds).
      r1 = asyncio.create_task(shared.pull(1))
      r2 = asyncio.create_task(shared.pull(1))
      await asyncio.sleep(0)
      r1.cancel()
      shared.push(99, 1)
      with self.assertRaises(asyncio.CancelledError):
        await r1
      self.assertEqual(await r2, 99)

      # Defense-in-depth: pull() rejects a manually injected Bottom value.
      injected = Stream()
      injected.seq.append(Bottom)
      with self.assertRaisesRegex(RuntimeError, 'must never observe Bottom'):
        await injected.pull(0)

    asyncio.run(check_stream_guards())

    # Primitives must never push Bottom during Stage 2.
    @psyre_comb
    def emit_bottom(val):
      del val
      return Bottom

    with self.assertRaises(ExceptionGroup) as eg_push:
      Scheduler.run(
          lambda: emit_bottom(Stream([1, 2])),
          horizon=2,
          semantics=Semantics.SYNC,
      )
    self.assertTrue(
        any(
            isinstance(e, TypeError) and 'Bottom cannot be pushed' in str(e)
            for e in eg_push.exception.exceptions
        )
    )

    observed: list[Any] = []

    @psyre_comb
    def record_and_inc(val):
      observed.append(val)
      return val + 1

    def prog():
      cnt = counter(start=10)
      p1 = post(cnt)
      p2 = post(cnt, 2)
      c_tail = Stream([False, False, True, True])
      c_head = Stream([True, True, False, False])
      c_once = Stream([True, False, False, False])
      with reset(when(c_once, False)):
        f_once = fby(when(c_once, 99), post(when(c_once, cnt)))
      return {
          'p1': p1,
          'inc_p1': record_and_inc(p1),
          'whennot_skip': whennot(c_tail, p2),
          'when_skip': when(c_head, p2),
          'merge_skip': merge(c_tail, when(c_tail, cnt), whennot(c_tail, p2)),
          'fby_p1': fby(0, p1),
          'fby_init_p2': fby(p2, cnt),
          'morph_p2': morph(p2, cnt, 2),
          'current_skip': current(c_head, p2, default=-1),
          'buffer_skip_in': buffer(f_once, c_once, c_once, size=4),
      }

    res = Scheduler.run(prog, horizon=4, semantics=Semantics.SYNC)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(observed, [11, 12, 13])
    self.assertEqual(res['p1'], [11, 12, 13, Bottom])
    self.assertEqual(res['inc_p1'], [12, 13, 14, Bottom])
    self.assertEqual(res['whennot_skip'], [12, 13, Absent, Absent])
    self.assertEqual(res['when_skip'], [12, 13, Absent, Absent])
    self.assertEqual(res['merge_skip'], [12, 13, 12, 13])
    self.assertEqual(res['fby_p1'], [0, 11, 12, 13])
    self.assertEqual(res['fby_init_p2'], [12, 10, 11, 12])
    self.assertEqual(res['morph_p2'], [12, 13, 12, 13])
    self.assertEqual(res['current_skip'], [12, 13, 13, 13])
    self.assertEqual(res['buffer_skip_in'], [99, Absent, Absent, Absent])

    # Re-running the same finalized Graph must produce identical traces.
    observed.clear()
    res_again = Scheduler.run_graph(
        Scheduler.graph, horizon=4, clocks=check_clocks(Scheduler.graph)
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(observed, [11, 12, 13])
    self.assertEqual(res_again['p1'][:3], [11, 12, 13])
    self.assertTrue(is_bottom(res_again['p1'], 3))
    self.assertEqual(res_again['buffer_skip_in'], [99, Absent, Absent, Absent])

  def test_inactive_tick_skipping_against_eof_stopped_streams(self):
    # Multi-stage fby/pre must skip discarded j_stream ticks on s_0 is Absent,
    # matching chained 1-stage delays on fby-wrapped and direct EOF streams.
    for label, cond_vals, horizon, post_n, wrap_fby, want_fby, want_pre in (
        (
            'wrapped_post3',
            (True, False, True, True),
            4,
            3,
            True,
            [99, Absent, 99, 0],
            [None, Absent, None, 0],
        ),
        (
            'post2_inactive_middle',
            (True, False, True),
            3,
            2,
            False,
            [99, Absent, 99],
            [None, Absent, None],
        ),
        (
            'post1_trailing_inactive',
            (True, False, False),
            3,
            1,
            False,
            [99, Absent, Absent],
            [None, Absent, Absent],
        ),
        (
            'post2_active_before_inactive',
            (True, True, False, True),
            4,
            2,
            False,
            [99, 99, Absent, 2],
            [None, None, Absent, 2],
        ),
    ):
      with self.subTest(case=label):
        def eof_prog(c_vals, n_look, do_wrap):
          c = Stream(list(c_vals))
          with match(c):
            with case(True):
              x = counter(start=0)
              src = post(x, n_look)
              if do_wrap:
                src = fby(0, src)
              fby(99, src, 2, name='fby2')
              fby(99, fby(99, src), name='fby_chain')
              pre(src, 2, name='pre2')
              pre(pre(src), name='pre_chain')
            with case(False):
              pass

        res = Scheduler.run(
            eof_prog,
            cond_vals,
            post_n,
            wrap_fby,
            horizon=horizon,
            semantics=Semantics.SYNC,
            streams=('fby2', 'fby_chain', 'pre2', 'pre_chain'),
        )
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(res['fby_chain'], want_fby)
        self.assertEqual(res['fby2'], want_fby)
        self.assertEqual(res['pre_chain'], want_pre)
        self.assertEqual(res['pre2'], want_pre)

    # 3-stage delay with leading/interleaved Absent ticks and a modular reset.
    def multistage_n3_with_reset():
      c = Stream([False, True, False, True, True, False, True])
      r = when(c, Stream([False, False, False, True, False, False, False]))
      init = when(c, Stream([0, 100, 0, 200, 300, 0, 400]))
      with match(c):
        with case(True):
          p = post(counter(start=10), 2)
          fby(init, p, 3, name='fby3_plain')
          fby(init, fby(init, fby(init, p)), name='fby3_plain_chain')
          with reset(r):
            fby(init, p, 3, name='fby3')
            fby(init, fby(init, fby(init, p)), name='fby3_chain')
            pre(p, 3, name='pre3')
            pre(pre(pre(p)), name='pre3_chain')
        with case(False):
          pass

    res_n3 = Scheduler.run(
        multistage_n3_with_reset,
        horizon=7,
        semantics=Semantics.SYNC,
        streams=(
            'fby3_plain',
            'fby3_plain_chain',
            'fby3',
            'fby3_chain',
            'pre3',
            'pre3_chain',
        ),
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(
        res_n3['fby3_plain_chain'],
        [Absent, 100, Absent, 100, 100, Absent, 12],
    )
    self.assertEqual(res_n3['fby3_plain'], res_n3['fby3_plain_chain'])
    self.assertEqual(
        res_n3['fby3_chain'],
        [Absent, 100, Absent, 200, 200, Absent, 200],
    )
    self.assertEqual(res_n3['fby3'], res_n3['fby3_chain'])
    self.assertEqual(
        res_n3['pre3_chain'],
        [Absent, None, Absent, None, None, Absent, None],
    )
    self.assertEqual(res_n3['pre3'], res_n3['pre3_chain'])

    # buffer, morph, current, and @psyre_sequ (prefix_sum/counter) must
    # skip all inputs when start is Absent rather than blocking on EOF-stopped
    # streams.
    def inactive_subclock_skips():
      c = Stream([True, True, False])
      with match(c):
        with case(True):
          x = counter(start=0)
          p = post(x)
          q = fby(True, p >= 0)
          q_int = fby(10, p)
          buffer(when(q, x), q, q, name='buf_sub')
          morph(x, q_int, 1, name='morph_sub')
          current(q, q_int, default=-1, name='curr_sub')
          prefix_sum(q_int, start=0, name='psum_sub')
          counter(start=0, max_val=q_int, name='cnt_sub')
        with case(False):
          pass

    res_sub = Scheduler.run(
        inactive_subclock_skips,
        horizon=3,
        semantics=Semantics.SYNC,
        streams=(
            'buf_sub',
            'morph_sub',
            'curr_sub',
            'psum_sub',
            'cnt_sub',
        ),
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(res_sub['buf_sub'], [0, 1, Absent])
    self.assertEqual(res_sub['morph_sub'], [0, 1, Absent])
    self.assertEqual(res_sub['curr_sub'], [10, 1, Absent])
    self.assertEqual(res_sub['psum_sub'], [10, 11, Absent])
    self.assertEqual(res_sub['cnt_sub'], [0, 1, Absent])

  def test_buffer_clock_mismatch_rejected_under_sync(self):
    def absent_on_pull():
      data_clk = Stream([True, False, True, False])
      data = when(data_clk, [10, 20, 30, 40])
      pull_p = Stream([True, True, True, False])
      push_p = Stream([False, False, True, True])
      return buffer(data, pull_p, push_p, size=2)

    with self.assertRaisesRegex(ClockError, 'buffer input'):
      Scheduler.run(absent_on_pull, horizon=4, semantics=Semantics.SYNC)
    bad_g = scheduler.build_graph(absent_on_pull, semantics=Semantics.SYNC)
    with self.assertRaisesRegex(ClockError, 'buffer input'):
      Scheduler.run_graph(bad_g, horizon=4)
    with self.assertRaisesRegex(ValueError, 'mandatory under Semantics.SYNC'):
      Scheduler.run(
          absent_on_pull,
          horizon=4,
          semantics=Semantics.SYNC,
          clock_calculus=False,
      )

  def test_absent_checks_use_identity_not_equality(self):
    @dataclasses.dataclass(frozen=True)
    class _AmbiguousEq:
      value: int

      def __eq__(self, other: object) -> bool:
        del other
        raise ValueError('truth value of array-like eq is ambiguous')

    @psyre_comb
    def identity(x: Any) -> Any:
      return x

    eq_all = test_utils.EqAll('sentinel')
    amb = _AmbiguousEq(7)

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        for payload in (eq_all, amb):
          c_out = Scheduler.run(
              lambda p=payload: identity(Stream(p)),
              horizon=2,
              semantics=sem,
          )
          self.assertLen(c_out, 2)
          self.assertIs(c_out[0], payload)
          self.assertIs(c_out[1], payload)

          sw_out = Scheduler.run(
              lambda p=payload: switch(Stream(True), Stream(p), Stream(0)),
              horizon=2,
              semantics=sem,
              optimize=False,
          )
          self.assertIs(sw_out[0], payload)
          self.assertIs(sw_out[1], payload)

          # Reset values must not be mistaken for Absent in Scheduler._run_node.
          def reset_with_payload(p=payload):
            with reset(Stream([p, False], period=[False])):
              return counter(0)

          self.assertEqual(
              Scheduler.run(reset_with_payload, horizon=3, semantics=sem),
              [0, 1, 2],
          )

  def test_ring_buffer_bottom_slots_and_strict_bounds(self):
    with self.assertRaisesRegex(ValueError, 'size must be >= 1'):
      common.RingBuffer(0)
    with self.assertRaisesRegex(ValueError, 'size must be >= 1'):
      common.RingBuffer(True)

    buf = common.RingBuffer(2)
    self.assertTrue(all(slot is Bottom for slot in buf.slots))
    # Empty buffer underflows without moving push_idx or leaking Bottom.
    with self.assertRaisesRegex(BufferError, 'RingBuffer underflow'):
      buf.push()
    self.assertEqual(buf.push_idx, 0)

    # Sentinels are rejected on pull without mutating slots or pull_idx.
    for sentinel, name in ((Absent, 'Absent'), (Bottom, 'Bottom')):
      with self.assertRaisesRegex(TypeError, f'cannot contain {name}'):
        buf.pull(sentinel)
    self.assertEqual(buf.pull_idx, 0)
    self.assertTrue(all(slot is Bottom for slot in buf.slots))

    # Valid None payload occupies slot 0 cleanly (distinct from Bottom).
    buf.pull(None)
    self.assertIsNone(buf.slots[0])
    self.assertIs(buf.slots[1], Bottom)
    buf.pull(42)
    # Full buffer overflows on slot 0 holding None (leaves state intact).
    with self.assertRaisesRegex(BufferError, 'RingBuffer overflow'):
      buf.pull(99)
    self.assertEqual(buf.pull_idx, 0)
    self.assertIsNone(buf.slots[0])

    # Drain wraps around and clears drained slots back to Bottom.
    self.assertIsNone(buf.push())
    self.assertIs(buf.slots[0], Bottom)
    buf.pull(77)
    self.assertEqual(buf.push(), 42)
    self.assertIs(buf.slots[1], Bottom)
    self.assertEqual(buf.push(), 77)
    self.assertTrue(all(slot is Bottom for slot in buf.slots))
    with self.assertRaisesRegex(BufferError, 'RingBuffer underflow'):
      buf.push()

    # Identity-based sentinel check tolerates objects with custom `__eq__`.
    tok = test_utils.EqAll()
    eq_buf = common.RingBuffer(1)
    eq_buf.pull(tok)
    with self.assertRaisesRegex(BufferError, 'RingBuffer overflow'):
      eq_buf.pull(tok)
    self.assertIs(eq_buf.push(), tok)

  def test_none_as_stream_value_and_mux_default(self):
    def none_prog():
      sel = Stream(['A', 'B', None, 'C', 'A', None])
      cond = Stream([False, True, False, True, True, False])
      vals = Stream([None, 1, None, 2, None, 3])
      return {
          'sw_default_none': switch(sel, {'A': 10, None: 20}, default=None),
          'm_default_none': merge(
              sel,
              {'A': when(sel, 10, values='A')},
              default=None,
          ),
          'when_none': when(sel, vals, values=None),
          'fby_none': fby(None, vals),
          'cur_none': current(cond, vals, default=None),
          'const_none': Stream(None),
      }

    for opt in (False, True):
      for sem in (Semantics.SYNC, Semantics.KAHN):
        with self.subTest(optimize=opt, semantics=sem):
          res = Scheduler.run(
              none_prog, horizon=6, semantics=sem, optimize=opt
          )
          self.assertEqual(
              res['sw_default_none'], [10, None, 20, None, 10, 20]
          )
          self.assertEqual(
              res['m_default_none'], [10, None, None, None, 10, None]
          )
          expected_when = (
              [Absent, Absent, None, Absent, Absent, 3]
              if sem == Semantics.SYNC
              else [None, 3]
          )
          self.assertEqual(res['when_none'], expected_when)
          self.assertEqual(res['fby_none'], [None, None, 1, None, 2, None])
          self.assertEqual(res['cur_none'], [None, 1, 1, 2, None, None])
          self.assertEqual(res['const_none'], [None] * 6)

    self.assertTrue(is_present([None, Absent, Bottom], 0))
    self.assertFalse(is_absent([None, Absent, Bottom], 0))
    self.assertFalse(is_bottom([None, Absent, Bottom], 0))
    self.assertEqual(present_values([None, Absent, 1, Bottom]), [None, 1])

    # Explicit `default=None` must still be rejected alongside bool mux/lookup.
    for fn_op in (merge, switch):
      with self.assertRaisesRegex(ValueError, 'does not accept default='):
        Scheduler.run(
            lambda op=fn_op: op(Stream(True), 1, 2, default=None), horizon=1
        )
      lk = common.MergeSelector(is_boolean=True)
      with self.assertRaisesRegex(ValueError, 'either lookup or default'):
        Scheduler.run(
            lambda op=fn_op, lk=lk: op(
                Stream(True), 1, 2, lookup=lk, default=None
            ),
            horizon=1,
        )

  def test_bottom_guards_on_trace_and_selectors(self):
    # `pad_bottom=True` must not be observable during Stage 1 or Stage 2 tasks.
    dyn = Stream()

    def stage1_trace_probe():
      with self.assertRaisesRegex(RuntimeError, 'after a Scheduler run'):
        dyn.trace(2, pad_bottom=True)
      return Stream(1)

    Scheduler.run(stage1_trace_probe, horizon=2, semantics=Semantics.SYNC)

    @psyre_comb
    def stage2_trace_probe(x):
      with self.assertRaisesRegex(RuntimeError, 'after a Scheduler run'):
        dyn.trace(2, pad_bottom=True)
      with self.assertRaisesRegex(RuntimeError, 'after a Scheduler run'):
        Scheduler.current().extract_traces(pad_bottom=True)
      return x

    self.assertEqual(
        Scheduler.run(
            lambda: stage2_trace_probe(Stream([1, 2])),
            horizon=2,
            semantics=Semantics.SYNC,
        ),
        [1, 2],
    )

    # `prepare_when` and `_prepare_mux` reject `Absent`/`Bottom` tag keys.
    class _RawMapping(Mapping[Any, Any]):

      def __init__(self, pairs: tuple[tuple[Any, Any], ...]):
        self._pairs = pairs

      def __getitem__(self, key: Any) -> Any:
        for k, v in self._pairs:
          if k is key:
            return v
        raise KeyError(key)

      def __iter__(self) -> Iterator[Any]:
        return (k for k, _ in self._pairs)

      def __len__(self) -> int:
        return len(self._pairs)

    for sentinel, name in ((Absent, 'Absent'), (Bottom, 'Bottom')):
      with self.assertRaisesRegex(TypeError, f'cannot contain {name}'):
        Scheduler.run(
            lambda s=sentinel: when(Stream(1), 1, values=s), horizon=1
        )
      with self.assertRaisesRegex(TypeError, f'cannot contain {name}'):
        Scheduler.run(
            lambda s=sentinel: switch(
                Stream(1), _RawMapping(((s, 1),))
            ),
            horizon=1,
        )

  def test_fby_step_o1_matches_chained_fby_and_pre(self):
    def run_pair(
        n: int,
        start_seq: list[Any],
        init_seq: list[Any],
        j_seq: list[Any],
        *,
        is_pre: bool = False,
    ) -> tuple[list[Any], list[Any]]:
      async def run_one() -> tuple[list[Any], list[Any]]:
        h = len(start_seq)
        sched = Scheduler(semantics=Semantics.SYNC, horizon=h)
        tok = common.active_scheduler.set(sched)
        try:
          # Direct O(1) multi-stage fby_step.
          i_dir = Stream()
          j_dir = Stream()
          for k, (iv, jv) in enumerate(zip(init_seq, j_seq)):
            i_dir.push(iv, k)
            j_dir.push(jv, k)
          o_dir = Stream()
          gen_dir = sync_primitives.fby_step(
              StreamPort(o_dir),
              None if is_pre else StreamPort(i_dir),
              StreamPort(j_dir),
              n=n,
          )
          await anext(gen_dir)
          for st in start_seq:
            await gen_dir.asend(st)
          await gen_dir.aclose()

          # Reference chain of `n` synchronous 1-stage `fby_step_1` delays.
          stage_out = j_dir
          for _ in range(n):
            next_out = Stream()
            ref_gen = sync_primitives.fby_step_1(
                StreamPort(next_out),
                None if is_pre else StreamPort(i_dir),
                StreamPort(stage_out),
            )
            await anext(ref_gen)
            for st in start_seq:
              await ref_gen.asend(st)
            await ref_gen.aclose()
            stage_out = next_out
          return list(o_dir.seq), list(stage_out.seq)
        finally:
          common.active_scheduler.reset(tok)

      return asyncio.run(run_one())

    start_seq = [
        True,
        Absent,
        False,
        False,
        True,
        Absent,
        True,
        False,
        False,
        False,
        Absent,
        False,
        True,
        False,
        False,
        False,
    ]
    init_seq = [
        10,
        Absent,
        99,
        99,
        Absent,
        Absent,
        None,
        99,
        99,
        99,
        Absent,
        99,
        'hi',
        99,
        99,
        99,
    ]
    j_seq = [
        1,
        Absent,
        Absent,
        2,
        3,
        Absent,
        None,
        4,
        Absent,
        5,
        Absent,
        6,
        7,
        8,
        9,
        10,
    ]
    for n in (1, 2, 3, 5, 8):
      for is_pre in (False, True):
        with self.subTest(n=n, is_pre=is_pre):
          got, ref = run_pair(n, start_seq, init_seq, j_seq, is_pre=is_pre)
          self.assertEqual(len(got), len(ref))
          for g_v, r_v in zip(got, ref):
            if g_v is Absent or r_v is Absent:
              self.assertIs(g_v, r_v)
            else:
              self.assertEqual(g_v, r_v)

    # High-depth fby/pre inside Scheduler.run (with reset mid-run).
    def deep_fby():
      rst = Stream.once(100)
      with reset(rst):
        c = counter(0)
        return fby(7, c, n=64), pre(c, n=64)

    fb_tr, pr_tr = Scheduler.run(
        deep_fby, horizon=170, semantics=Semantics.SYNC
    )
    self.assertEqual(fb_tr[:64], [7] * 64)
    self.assertEqual(fb_tr[64:100], list(range(36)))
    self.assertEqual(fb_tr[100:164], [7] * 64)
    self.assertEqual(fb_tr[164:], list(range(6)))
    self.assertEqual(pr_tr[:64], [None] * 64)
    self.assertEqual(pr_tr[64:100], list(range(36)))
    self.assertEqual(pr_tr[100:164], [None] * 64)
    self.assertEqual(pr_tr[164:], list(range(6)))

  def test_sliding_window_eviction_and_logging(self):
    # 1. Single-reader feedback loop & 0-reader unused dynamic node stay bounded
    # while unprobed intermediate streams evict consumed slots.
    observed_mid_lens: list[int] = []
    observed_dead_lens: list[int] = []
    tracked: dict[str, Stream] = {}

    @psyre_comb
    def probe_step(x):
      if 'mid' in tracked:
        observed_mid_lens.append(len(tracked['mid'].seq))
      if 'dead' in tracked:
        observed_dead_lens.append(len(tracked['dead'].seq))
      return x + 1

    def evict_prog():
      dead = counter(0) + 100
      tracked['dead'] = dead
      acc = fby(0, lambda: nxt)
      mid = acc + 1
      tracked['mid'] = mid
      nxt = probe_step(mid)
      return nxt

    out = Scheduler.run(
        evict_prog, horizon=250, semantics=Semantics.SYNC, optimize=False
    )
    self.assertEqual(out[0], 2)
    self.assertEqual(out[-1], 500)
    self.assertTrue(all(length <= 2 for length in observed_mid_lens))
    self.assertTrue(all(length == 0 for length in observed_dead_lens))

    # 2. Multi-reader fanout (including `x + x` sharing the same stream twice).
    def fanout_prog():
      x = counter(1) * 3
      return (x + x) + fby(0, x)

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(fanout_semantics=sem):
        res = Scheduler.run(fanout_prog, horizon=60, semantics=sem)
        expected = [
            6 * (k + 1) + (0 if k == 0 else 3 * k) for k in range(60)
        ]
        self.assertEqual(res, expected)

    # 3. `post(n)` rewind across repeated resets (including subclocked `match`).
    def post_reset_prog():
      rst = Stream.periodic([False], [False, False, False, True])
      with reset(rst):
        x = counter(0)
        p1 = post(x, n=1)
        p2 = post(x, n=2)
      return p1, p2

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(post_reset_semantics=sem):
        p1, p2 = Scheduler.run(
            post_reset_prog, horizon=12, semantics=sem, optimize=False
        )
        # `rst` fires at ticks 4 and 8, so `x` is [0,1,2,3, 0,1,2,3, 0,1,2,3].
        # Rewind checks ensure `post(x, n=1/2)` safely reads back across resets
        # without losing lookahead slots to sliding-window eviction.
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(p1[:8], [1, 2, 3, 0, 1, 2, 3, 0])
        self.assertEqual(p2[:8], [2, 3, 0, 1, 2, 3, 0, 1])

    def subclock_post_reset_prog():
      gate = Stream.periodic([], [True, False, True])
      rst = Stream.periodic([False, False, False], [False, True, False, False])
      with reset(rst):
        with match(gate) as m:
          with case(True):
            sub_x = counter(10)
            m.out = post(sub_x, n=1)
          with case(False):
            m.out = -1
      return m.out

    sub_sync = Scheduler.run(
        subclock_post_reset_prog, horizon=15, semantics=Semantics.SYNC
    )
    self.assertFalse(Scheduler.deadlocked)
    # `rst` fires at ticks 4 (latched onto tick 5), 8, and 12; `sub_x` ticks on
    # `gate=True` (0, 2, 3, 5, 6, 8, 9, 11, 12, 14) with values
    # [10, 11, 12, 10, 11, 10, 11, 12, 10, 11], so `post(sub_x, n=1)` yields:
    self.assertEqual(
        sub_sync,
        [11, -1, 12, 10, -1, 11, 10, -1, 11, 12, -1, 10, 11, -1, Bottom],
    )

    # 4. Custom `@psyre_loop` rewinding `port.index` back to 0 must NOT have
    # its input stream evicted.
    @psyre_loop
    async def replay_first(o_port, i_port):
      val = await pull(i_port)
      o_port.push(val)
      if i_port.index >= 4:
        i_port.index = 0

    def loop_prog():
      src = counter(10)
      return replay_first(src)

    self.assertEqual(
        Scheduler.run(loop_prog, horizon=10, semantics=Semantics.SYNC),
        [10, 11, 12, 13, 10, 11, 12, 13, 10, 11],
    )

    # 5. Direct evicted stream checks: `trace()` and `pull()` raise cleanly.
    ev = Stream()
    ev_port = StreamPort(ev)
    ev.configure_eviction([ev_port])
    ev.push(10, 0)
    ev.push(20, 1)
    self.assertEqual(ev_port.try_pull_ready(), 10)
    with self.assertRaisesRegex(RuntimeError, 'evicted intermediate stream'):
      ev.trace(2)
    with self.assertRaisesRegex(RuntimeError, 'already evicted'):
      asyncio.run(ev.pull(0))
    ev_port.index = 0
    with self.assertRaisesRegex(RuntimeError, 'already evicted'):
      ev_port.try_pull_ready()

    # 6. `skip_absent` ahead of push drops skipped slots without stalling GC.
    sa_stream = Stream()
    sa_port = StreamPort(sa_stream)
    sa_stream.configure_eviction([sa_port])
    sa_port.skip_absent()
    sa_port.skip_absent()
    sa_stream.push(Absent, 0)
    sa_stream.push(Absent, 1)
    self.assertEqual(sa_stream.seq, [])
    self.assertEqual(sa_stream.pushed_count, 2)
    sa_stream.push(42, 2)
    self.assertEqual(sa_port.try_pull_ready(), 42)
    self.assertEqual(sa_stream.seq, [])

    # 7. Port-matched `unblock_task(task, stream)` preserves unrelated blocks.
    async def check_unblock_matching() -> None:
      sched = Scheduler(semantics=Semantics.SYNC, horizon=1)
      task = asyncio.current_task()
      assert task is not None
      s_a = Stream()
      s_b = Stream()
      p_b = StreamPort(s_b)
      sched._blocked[task] = p_b  # pylint: disable=protected-access
      sched.unblock_task(task, s_a)
      self.assertIn(task, sched._blocked)  # pylint: disable=protected-access
      sched.unblock_task(task, s_b)
      self.assertNotIn(task, sched._blocked)  # pylint: disable=protected-access

    asyncio.run(check_unblock_matching())

    # 8. INFO-level trace dump was removed from Scheduler; DEBUG still logs.
    records: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):

      def emit(self, record: logging.LogRecord) -> None:
        records.append(record)

    handler = _CaptureHandler()
    logger.addHandler(handler)
    try:
      logger.setLevel(logging.INFO)
      Scheduler.run(lambda: counter(0) + 1, horizon=3, semantics=Semantics.SYNC)
      self.assertEqual(
          [r for r in records if r.levelno == logging.INFO], []
      )
      records.clear()
      logger.setLevel(logging.DEBUG)
      Scheduler.run(lambda: counter(0) + 1, horizon=2, semantics=Semantics.SYNC)
      self.assertTrue(
          any(r.levelno == logging.DEBUG for r in records),
          'Expected DEBUG logs when logger level is DEBUG.',
      )
    finally:
      logger.removeHandler(handler)
      logger.setLevel(logging.WARNING)

  def test_current_subclock_and_runtime_presence_check(self):
    # 1. Subclocked `current` holds `memory` instead of leaking `Absent` when
    #    `b_stream` is truthy and `i_stream` is `Absent`.
    def subclock_holds_memory():
      nat = counter(0, name='nat')
      is_odd = (nat % 2) == 1
      odd = when(is_odd, nat, name='odd')
      none_sub = when(
          Stream([False, True, False, True, False, True]),
          Stream([9, None, 9, 42, 9, None]),
      )
      return {
          'cur_odd': current(is_odd, odd, default=0),
          'cur_true_odd': current(True, odd, default=-1),
          'cur_even_odd': current(~is_odd, odd, default=99),
          'cur_none_payload': current(True, none_sub, default=7),
          'cur_none_default': current(True, odd, default=None),
      }

    res = Scheduler.run(
        subclock_holds_memory, horizon=6, semantics=Semantics.SYNC
    )
    self.assertEqual(res['cur_odd'], [0, 1, 1, 3, 3, 5])
    self.assertEqual(res['cur_true_odd'], [-1, 1, 1, 3, 3, 5])
    self.assertEqual(res['cur_even_odd'], [99, 99, 99, 99, 99, 99])
    self.assertEqual(res['cur_none_payload'], [7, None, None, 42, 42, None])
    self.assertEqual(res['cur_none_default'], [None, 1, 1, 3, 3, 5])

    # 2. Subclocked `current` with initial `Absent` tick initializes on tick 1.
    def subclocked_init_after_absent():
      c = Stream([False, True, True, True])
      b = when(c, Stream([False, False, True, False]))
      d = when(c, Stream([99, 88, 77, 66]))
      sub_i = when(c, Stream([10, 20, 30, 40]))
      with reset(when(c, False)):
        return current(b, sub_i, default=d)

    self.assertEqual(
        Scheduler.run(
            subclocked_init_after_absent,
            horizon=4,
            semantics=Semantics.SYNC,
        ),
        [Absent, 88, 30, 30],
    )

    # 3. Faster dynamic `i_stream` than `b_stream` is rejected statically by
    #    mandatory `SYNC` clock analysis (even via `run_graph` after
    #    `mark_resets_lowered`).
    def faster_dynamic_b_absent():
      c = Stream([False, True, False, True])
      b = when(c, True, name='slow_b')
      x = counter(0, name='fast_i')
      with reset(when(c, False)):
        return current(b, x, default=-1)

    with self.assertRaisesRegex(ClockError, 'current input fast_i'):
      Scheduler.run(
          faster_dynamic_b_absent, horizon=4, semantics=Semantics.SYNC
      )
    reset_g = scheduler.build_graph(
        faster_dynamic_b_absent, semantics=Semantics.SYNC
    )
    reset_g.mark_resets_lowered(Semantics.SYNC)
    with self.assertRaisesRegex(ClockError, 'current input fast_i'):
      Scheduler.run_graph(reset_g, horizon=4)

    # 4. Polymorphic static `i_stream` inside subclocked `current`.
    def static_input_with_absent_b():
      c = Stream([False, True, False, True])
      with reset(when(c, False)):
        return current(when(c, True), Stream(42), default=-1)

    self.assertEqual(
        Scheduler.run(
            static_input_with_absent_b,
            horizon=4,
            semantics=Semantics.SYNC,
        ),
        [Absent, 42, Absent, 42],
    )

    # 5. Early-EOF `post` producer when `b_stream` is `Absent` yields `Absent`.
    def eof_post_when_b_absent():
      c = Stream([True, True, False, False])
      cnt = counter(10)
      b = when(c, Stream([True, False, False, False]))
      with reset(when(c, False)):
        p = post(when(c, cnt))
        return current(b, p, default=-1)

    self.assertEqual(
        Scheduler.run(
            eof_post_when_b_absent, horizon=4, semantics=Semantics.SYNC
        ),
        [11, 11, Absent, Absent],
    )

  def test_counter_tick_enable_and_initial_absent_recovery(self):
    def prog():
      en = Stream([False, True, False, True, True, False, True])
      return {
          'default_tick': counter(start=0),
          'gated': counter(start=0, tick=en),
          'stall': counter(start=5, tick=False),
          'clamped': counter(start=0, max_val=2, tick=en),
          'positional_max': counter(10, 12),
      }

    res = Scheduler.run(prog, horizon=7, semantics=Semantics.SYNC)
    self.assertEqual(res['default_tick'], [0, 1, 2, 3, 4, 5, 6])
    self.assertEqual(res['gated'], [0, 0, 1, 1, 2, 3, 3])
    self.assertEqual(res['stall'], [5, 5, 5, 5, 5, 5, 5])
    self.assertEqual(res['clamped'], [0, 0, 1, 1, 2, 2, 2])
    self.assertEqual(res['positional_max'], [10, 11, 12, 12, 12, 12, 12])

    # Subclocked `counter` initializes cleanly on its first active tick (1).
    def subclocked_absent_start():
      c = Stream([False, True, False, True, True])
      init_s = when(c, Stream([99, 10, 99, 20, 30]))
      en_s = when(c, Stream([False, True, False, False, True]))
      with reset(when(c, False)):
        return counter(start=init_s, tick=en_s)

    self.assertEqual(
        Scheduler.run(
            subclocked_absent_start,
            horizon=5,
            semantics=Semantics.SYNC,
        ),
        [Absent, 10, Absent, 11, 11],
    )

  def test_runtime_hardening_and_fail_fast_contracts(self):
    # Subclocked `prefix_sum` and `scan` initialize cleanly after an `Absent`
    # tick 0 via `lower_resets()`.
    def subclocked_psum():
      c = Stream([False, True, False, True, True])
      d = when(c, [99, 2, 99, 3, 5])
      init = when(c, [99, 10, 99, 99, 99])
      with reset(when(c, False)):
        return prefix_sum(d, start=init)

    self.assertEqual(
        Scheduler.run(
            subclocked_psum,
            horizon=5,
            semantics=Semantics.SYNC,
        ),
        [Absent, 12, Absent, 15, 20],
    )

    def subclocked_scan():
      c = Stream([False, True, False, True, True])
      d = when(c, [99, 2, 99, 3, 5])
      with reset(when(c, False)):
        return scan(d, op=operator.add)

    self.assertEqual(
        Scheduler.run(
            subclocked_scan,
            horizon=5,
            semantics=Semantics.SYNC,
        ),
        [Absent, 2, Absent, 5, 10],
    )

    # Outer reset latched across an inactive subclock tick resets `_sequ`.
    def latched_sequ_reset():
      c = Stream([True, False, True, True])
      r = Stream([False, True, False, False])
      d = when(c, Stream([1, 99, 3, 4]))
      with reset(r):
        return prefix_sum(d, start=10)

    self.assertEqual(
        Scheduler.run(
            latched_sequ_reset, horizon=4, semantics=Semantics.SYNC
        ),
        [11, Absent, 13, 17],
    )

    # User `@psyre_comb` / `@psyre_sequ` callbacks cannot emit `Absent`.
    @psyre_comb
    def bad_comb(x):
      del x
      return Absent

    @psyre_sequ
    def bad_sequ(x):
      del x
      while True:
        yield Absent

    for sem in (Semantics.SYNC, Semantics.KAHN):
      for bad_prog in (
          lambda: bad_comb(Stream(1)),
          lambda: bad_sequ(Stream(1)),
      ):
        with self.subTest(bad_sentinel=bad_prog, semantics=sem):
          with self.assertRaises(ExceptionGroup) as eg:
            Scheduler.run(bad_prog, horizon=2, semantics=sem)
          self.assertTrue(
              any(
                  isinstance(e, TypeError) and 'cannot contain Absent' in str(e)
                  for e in eg.exception.exceptions
              ),
              eg.exception.exceptions,
          )

    # Trace index predicates reject slices, booleans, floats, and strings.
    class _Idx:

      def __index__(self) -> int:
        return 2

    tr = [Absent, Bottom, 42]
    self.assertTrue(is_absent(tr, 0))
    self.assertTrue(is_bottom(tr, 1))
    self.assertTrue(is_present(tr, -1))
    self.assertTrue(is_present(tr, _Idx()))
    for bad_idx in (slice(0, 2), True, False, 1.5, '0'):
      for pred in (is_present, is_absent, is_bottom):
        with self.assertRaises(TypeError):
          pred(tr, bad_idx)  # type: ignore[arg-type]

    # Deadlock cancellation must not dump raw tracebacks to stderr.
    stderr_buf = io.StringIO()
    with contextlib.redirect_stderr(stderr_buf):
      Scheduler.run(
          lambda: when(False, counter()) + 1,
          horizon=4,
          semantics=Semantics.KAHN,
      )
    self.assertTrue(Scheduler.deadlocked)
    self.assertEqual(stderr_buf.getvalue(), '')

    # Exported primitive dispatchers expose non-empty docstrings.
    for op_name in (
        'buffer',
        'counter',
        'current',
        'fby',
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
    ):
      self.assertTrue(getattr(common, op_name).__doc__, op_name)

    # MergeSelector.arms is immutable via MappingProxyType.
    ms = common.MergeSelector(is_boolean=False, arms={(False, 1): 0})
    self.assertIsInstance(ms.arms, types.MappingProxyType)
    with self.assertRaises(TypeError):
      ms.arms[(False, 2)] = 1  # type: ignore[index]

    # Stage-1 validation for scan op, empty mux mapping, decorators, run_graph.
    existing = Stream([1, 2])
    with self.assertRaisesRegex(TypeError, 'callable'):
      common.prepare_scan(existing, op=42)  # type: ignore[arg-type]
    with self.assertRaisesRegex(ValueError, 'at least one branch'):
      common.prepare_merge(existing, {})
    with self.assertRaisesRegex(ValueError, 'at least one branch'):
      common.prepare_switch(existing, {})
    for dec in (psyre_loop, psyre_comb, common.psyre_sequ):
      with self.assertRaisesRegex(TypeError, 'callable'):
        dec(42)  # type: ignore[arg-type]
    with self.assertRaisesRegex(ValueError, r'psyre_comb nout must be'):
      psyre_comb(nout=0)

    g = scheduler.build_graph(lambda: counter(0, name='c'))
    with self.assertRaisesRegex(TypeError, 'bare str'):
      Scheduler.run_graph(g, streams='c')
    with self.assertRaises(ValueError):
      Scheduler.run_graph(g, horizon=-1)
    for bad_h in (-1, True, False):
      with self.assertRaises(ValueError):
        existing.trace(bad_h)

    # pull()/pull_all() outside an active Scheduler fail fast with RuntimeError.
    async def _pull_outside_scheduler() -> None:
      tok = common.active_scheduler.set(None)
      try:
        port = StreamPort(Stream())
        with self.assertRaisesRegex(RuntimeError, 'active Scheduler'):
          await pull(port)
        with self.assertRaisesRegex(RuntimeError, 'active Scheduler'):
          await common.pull_all(port)
      finally:
        common.active_scheduler.reset(tok)

    asyncio.run(_pull_outside_scheduler())
    self.assertFalse(hasattr(Scheduler, 'pull_all'))


if __name__ == '__main__':
  absltest.main()
