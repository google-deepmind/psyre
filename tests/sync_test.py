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

import logging
import math
import operator
from typing import Any

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre import common
from psyre import scheduler
from psyre import sync_primitives
from psyre.common import (
    Absent,
    Semantics,
    buffer,
    counter,
    delay,
    fby,
    ifthenelse,
    logger,
    merge,
    morph,
    post,
    pre,
    prefix_sum,
    psyre_comb,
    reset,
    scan,
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
from psyre.stream import Stream
from psyre.trace import (
    is_absent,
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
          scan([1, 0, 3, 0], op=operator.add, name='sparse_tail')
        with case(False):
          pass
      with match(cond2):
        with case(True):
          scan([0, 1, 0, 3], op=operator.add, name='with_absent')
          scan([0, 3, 0, 5], op=Stream.max, name='max_absent')
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
      sel = when(
          Stream([True, False, True, True, True]), ['a', 'x', 'b', 'a', 'z']
      )
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
                  'a': [1, 0, 0, 2, 0],
                  'b': [0, 0, 10, 0, 0],
              },
              default=[0, 0, 0, 0, -1],
          ),
          'bool_none_merge': merge(
              typed_sel,
              {
                  True: [10, 0, 0, 0, 0],
                  False: [0, 0, 30, 0, 0],
                  None: [0, 0, 0, 0, 50],
              },
              default=[0, -1, 0, -2, 0],
          ),
          'int_merge': merge(
              typed_sel,
              {
                  1: [0, 20, 0, 0, 0],
                  0: [0, 0, 0, 40, 0],
              },
              default=[-1, 0, -2, 0, -3],
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
          fby([10, 20, 30, 40], [1, 0, 3, 4], name='f_out')
          fby([10, 20, 30, 40], [1, 0, 3, 4], 2, name='f2_out')
          pre([10, 0, 30, 40], 2, name='p2_out')
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
      inp = [10, 20, 0, 0, 30, 0, 40]
      inp_p = [True, True, False, False, True, False, True]
      out_p = [False, False, True, True, False, True, True]
      return buffer(inp, inp_p, out_p), buffer(inp, inp_p, out_p, size=2)

    size_one, size_two = Scheduler.run(
        inner, horizon=7, semantics=Semantics.SYNC
    )
    self.assertEqual(size_one, [Absent, Absent, 20, None, Absent, 30, 40])
    self.assertEqual(size_two, [Absent, Absent, 10, 20, Absent, 30, 40])

  def test_delay(self):
    def inner():
      inp = [10, 20, 30, 40, 50, 60]
      gate = [True, False, True, True, False, True]
      sparse_gate = [True, False, True, False, True, False]
      rst = [False, False, False, True, False, False]
      sched = Stream.periodic([False, False], [True, False])
      sched_alt = Stream.periodic([False], [True, False])
      delay(inp, 0, name='d0')
      delay(inp, name='d1')
      delay(inp, 2, name='d2')
      with reset(rst):
        delay(inp, 1, name='d_rst')
        delay(inp, period=sched_alt, name='dp_rst')
      with match(sparse_gate):
        with case(True):
          delay([10, 0, 20, 0, 30, 0], period=sched, name='dp')
        with case(False):
          pass
      with match(gate):
        with case(True):
          delay(inp, 1, name='gate_dk')
          delay(inp, period=sched_alt, name='gate_dp')
        with case(False):
          pass

    res = Scheduler.run(
        inner,
        horizon=6,
        semantics=Semantics.SYNC,
        streams=(
            'd0',
            'd1',
            'd2',
            'dp',
            'd_rst',
            'dp_rst',
            'gate_dk',
            'gate_dp',
        ),
    )
    self.assertEqual(res['d0'], [10, 20, 30, 40, 50, 60])
    self.assertEqual(res['d1'], [Absent, 10, 20, 30, 40, 50])
    self.assertEqual(res['d2'], [Absent, Absent, 10, 20, 30, 40])
    self.assertEqual(res['dp'], [Absent, Absent, Absent, Absent, 10, Absent])
    self.assertEqual(res['d_rst'], [Absent, 10, 20, Absent, 40, 50])
    self.assertEqual(res['dp_rst'], [Absent, 10, Absent, Absent, 40, Absent])
    self.assertEqual(res['gate_dk'], [Absent, Absent, 10, 30, Absent, 40])
    self.assertEqual(res['gate_dp'], [Absent, Absent, 10, Absent, Absent, 30])

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
    o1_act = Scheduler.run(
        inner, horizon=5, semantics=Semantics.SYNC, streams=('o1_act',)
    )['o1_act']
    self.assertEqual(o1_act, [False, Absent, True, False, Absent])
    self.assertEqual(merged, [False, False, True, False, False])

  def test_post(self):
    def inner():
      inp = [10, 20, 30, 40]
      gate = [False, True, True, False]
      gate2 = [False, True, True, True]
      with match(gate):
        with case(True):
          post([0, 10, 20, 0], name='p_abs')
        with case(False):
          pass
      with match(gate2):
        with case(True):
          post([0, 10, 20, 30], 2, name='pn2_abs')
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
    self.assertEqual(pp, [30, 40, 40, Absent])
    self.assertEqual(pn2, [30, 40, 40, 40])
    self.assertEqual(abs_res['p_abs'], [Absent, 20, Absent, Absent])
    self.assertEqual(abs_res['pn2_abs'], [Absent, 30, Absent, Absent])

  def test_morph_absent(self):
    def inner():
      cond = Stream([False, True, False, True, True])
      with match(cond) as morph_m:
        with case(True):
          m_act = morph([0, 10, 0, 30, 50], [0, 2, 0, 4, 6], name='m_act')
          morph([0, 10, 0, 30, 50], [0, 2, 0, 4, 6], 2, name='m2_act')
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
      inp = [10, 20, 30, 40, 50]
      init = [100, 200, 300, 400, 500]
      delayed = [1, 2, 3, 4, 5]
      cond = Stream([True, False, True, False, True])
      rst = [False, False, True, False, False]
      super_reset = [False, True, False, False, False]

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
    self.assertEqual(res['super_when_fby'], [100, Absent, 300, Absent, 3])
    self.assertEqual(res['super_when_counter'], [1, Absent, 2, Absent, 4])
    self.assertEqual(
        res['super_post_when'], [30, Absent, 50, Absent, Absent]
    )
    self.assertEqual(res['super_scan_when'], [10, Absent, 30, Absent, 80])

  def test_nested_match_and_reset(self):
    def inner():
      outer_gate = [True, False, True, True, False, True]
      inner_gate = [True, True, False, True, True, True]
      outer_reset = [False, True, False, False, False, False]
      inner_reset = [False, False, True, False, False, False]

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
    self.assertTrue(is_present(trace, -1))
    self.assertTrue(is_absent(trace, -3))
    self.assertEqual(present_values(trace), [False, 0, None, []])
    self.assertEqual(present_values(tuple(trace)), [False, 0, None, []])

    with self.assertRaises(IndexError):
      is_present(trace, 6)
    with self.assertRaises(IndexError):
      is_absent(trace, 6)

    bad_traces: tuple[Any, ...] = (Stream([1, 2]), 'abc', b'abc', 123)
    for bad_trace in bad_traces:
      with self.assertRaisesRegex(TypeError, 'post-run trace Sequence'):
        is_present(bad_trace, 0)
      with self.assertRaisesRegex(TypeError, 'post-run trace Sequence'):
        is_absent(bad_trace, 0)
      with self.assertRaisesRegex(TypeError, 'post-run trace Sequence'):
        present_values(bad_trace)

    for fn in (
        lambda: is_present([1], 0),
        lambda: is_absent([1], 0),
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
          gated_max = Stream([2, 5, -2, -5]).max(0)
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


if __name__ == '__main__':
  absltest.main()
