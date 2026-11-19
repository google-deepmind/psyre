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
"""Unit tests for asynchronous (Kahn) Psyre stream primitives and execution."""

import enum
import logging
import operator
from typing import Any

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.automaton import (
    automaton,
    state,
    unless,
    until,
)
from psyre.common import (
    Semantics,
    buffer,
    counter,
    current,
    delay,
    fby,
    ifthenelse,
    logger,
    merge,
    morph,
    post,
    pre,
    prefix_sum,
    reset,
    scan,
    when,
    whennot,
)
from psyre.match import (
    case,
    default,
    last,
    match,
)
from psyre.scheduler import Scheduler
from psyre.stream import Stream, resolve_handle
from psyre.trace import present_values

# pylint: enable=g-importing-member,g-multiple-import


class _Mode(enum.Enum):
  FAST = enum.auto()
  SLOW = enum.auto()


class _Vec:
  """Minimal object supporting the @ operator for testing Stream matmul."""

  value: int

  def __init__(self, value: int):
    self.value = value

  def __matmul__(self, other: '_Vec') -> int:
    if not isinstance(other, _Vec):
      return NotImplemented
    return self.value * other.value


class KahnTest(absltest.TestCase):

  def test_experiments(self):
    logger.setLevel(logging.INFO)

    def inner():
      nat = counter()
      prefix = prefix_sum(nat)
      odd_p = prefix & 1 == 1
      even_p = ~odd_p
      scaled = prefix * 2
      odd = whennot(even_p, prefix)
      even = when(even_p, scaled)
      merged = merge(odd_p, odd, even)
      b1 = Stream.periodic([False], [False, True])
      b2 = fby(True, b1)
      b3 = morph(False, pre(b2))
      cur = current(b3, merged)
      return ifthenelse(cur, cur, 0)

    self.assertEqual(
        Scheduler.run(inner, horizon=6, semantics=Semantics.KAHN),
        [0, 1, 1, 1, 20, 20],
    )

  def test_operators(self):
    logger.setLevel(logging.WARNING)

    def inner():
      s = Stream([2, 3, 4])
      v = Stream([_Vec(2), _Vec(3), _Vec(4)])
      b = Stream([True, False, True])

      return {
          # Arithmetic and matrix forward operators
          'add': s + 5,
          'sub': s - 1,
          'mul': s * 3,
          'matmul': v @ _Vec(5),
          'truediv': s / 2,
          'floordiv': s // 3,
          'mod': s % 3,
          'pow': s**2,
          # Arithmetic and matrix reflected operators
          'radd': 5 + s,
          'rsub': 10 - s,
          'rmul': 3 * s,
          'rmatmul': _Vec(5) @ v,
          'rtruediv': 12 / s,
          'rfloordiv': 10 // s,
          'rmod': 10 % s,
          'rpow': 2**s,
          # Bitwise and shift forward and reflected operators
          'lshift': s << 1,
          'rlshift': 1 << s,
          'rshift': s >> 1,
          'rrshift': 16 >> s,
          'and': s & 2,
          'rand': 2 & s,
          'or': s | 1,
          'ror': 1 | s,
          'xor': s ^ 3,
          'rxor': 3 ^ s,
          # Unary operators
          'pos': +s,
          'neg': -s,
          'invert_int': ~s,
          'invert_bool': ~b,
          # Comparison operators (forward and reflected via Python fallback)
          'lt': s < 3,
          'le': s <= 3,
          'eq': s == 3,
          'ne': s != 3,
          'gt': s > 3,
          'ge': s >= 3,
          'rlt': 3 < s,
          'rle': 3 <= s,
          'rgt': 3 > s,
          'rge': 3 >= s,
      }

    actual = Scheduler.run(inner, horizon=3, semantics=Semantics.KAHN)
    expected = {
        'add': [7, 8, 9],
        'sub': [1, 2, 3],
        'mul': [6, 9, 12],
        'matmul': [10, 15, 20],
        'truediv': [1.0, 1.5, 2.0],
        'floordiv': [0, 1, 1],
        'mod': [2, 0, 1],
        'pow': [4, 9, 16],
        'radd': [7, 8, 9],
        'rsub': [8, 7, 6],
        'rmul': [6, 9, 12],
        'rmatmul': [10, 15, 20],
        'rtruediv': [6.0, 4.0, 3.0],
        'rfloordiv': [5, 3, 2],
        'rmod': [0, 1, 2],
        'rpow': [4, 8, 16],
        'lshift': [4, 6, 8],
        'rlshift': [4, 8, 16],
        'rshift': [1, 1, 2],
        'rrshift': [4, 2, 1],
        'and': [2, 2, 0],
        'rand': [2, 2, 0],
        'or': [3, 3, 5],
        'ror': [3, 3, 5],
        'xor': [1, 0, 7],
        'rxor': [1, 0, 7],
        'pos': [2, 3, 4],
        'neg': [-2, -3, -4],
        'invert_int': [-3, -4, -5],
        'invert_bool': [False, True, False],
        'lt': [True, False, False],
        'le': [True, True, False],
        'eq': [False, True, False],
        'ne': [True, False, True],
        'gt': [False, False, True],
        'ge': [False, True, True],
        'rlt': [False, False, True],
        'rle': [False, True, True],
        'rgt': [True, False, False],
        'rge': [True, True, False],
    }
    for name, expected_values in expected.items():
      with self.subTest(operator=name):
        self.assertEqual(actual[name], expected_values)

  def test_random_stream(self):
    logger.setLevel(logging.WARNING)
    r1, r2 = Scheduler.run(
        lambda: (Stream.random(), Stream.random()),
        horizon=5,
        semantics=Semantics.KAHN,
    )
    self.assertLen(r1, 5)
    self.assertLen(r2, 5)
    for v in r1 + r2:
      self.assertGreaterEqual(v, 0.0)
      self.assertLess(v, 1.0)
    self.assertGreater(len(set(r1)), 1)
    self.assertNotEqual(r1, r2)

  def test_ifthenelse(self):
    logger.setLevel(logging.WARNING)

    def inner():
      cond = [True, False, True, False]
      true_branch = [1, 2, 3, 4]
      false_branch = [10, 20, 30, 40]
      return ifthenelse(cond, true_branch, false_branch)

    self.assertEqual(
        Scheduler.run(inner, horizon=4, semantics=Semantics.KAHN),
        [1, 20, 3, 40],
    )

  def test_scan(self):
    logger.setLevel(logging.WARNING)

    def inner():
      return (
          scan([1, 2, 3, 4], op=operator.add),
          scan([2.0, 0.5, 3.0, -2.0], op=operator.mul),
          scan([3, 1, 5, 2], op=Stream.max),
          scan([3, 1, 5, -2], op=Stream.min),
      )

    int_add, float_mul, max_s, min_s = Scheduler.run(
        inner, horizon=4, semantics=Semantics.KAHN
    )
    self.assertEqual(int_add, [1, 3, 6, 10])
    self.assertEqual(float_mul, [2.0, 1.0, 3.0, -6.0])
    self.assertEqual(max_s, [3, 3, 5, 5])
    self.assertEqual(min_s, [3, 1, 1, -2])

  def test_when_and_whennot(self):
    logger.setLevel(logging.WARNING)

    def inner():
      cond = [True, False, True, False, True]
      inp = [10, 20, 30, 40, 50]
      return when(cond, inp), whennot(cond, inp)

    w_out, wn_out = Scheduler.run(inner, horizon=5, semantics=Semantics.KAHN)
    self.assertEqual(w_out, [10, 30, 50])
    self.assertEqual(wn_out, [20, 40])

  def test_merge(self):
    logger.setLevel(logging.WARNING)

    def inner():
      cond = Stream([True, False, False, True, False])
      true_branch = [1, 2]
      false_branch = [10, 20, 30]
      inp = [100, 200, 300, 400, 500]
      sel = ['a', 'b', 'c', 'a', 'z']
      typed_sel = [True, 1, False, 0, None]
      return {
          'merge': merge(cond, true_branch, false_branch),
          'merge_ignore_default': merge(
              cond, true_branch, false_branch, default=[-1]
          ),
          'roundtrip': merge(cond, when(cond, inp), whennot(cond, inp)),
          'multi_merge': merge(
              sel,
              {'a': [1, 2], 'b': [10], 'c': [100]},
              default=[-1],
          ),
          'bool_none_merge': merge(
              typed_sel,
              {True: [10], False: [30], None: [50]},
              default=[-1, -2],
          ),
          'int_merge': merge(
              typed_sel,
              {1: [20], 0: [40]},
              default=[-1, -2, -3],
          ),
      }

    res = Scheduler.run(inner, horizon=5, semantics=Semantics.KAHN)
    self.assertEqual(res['merge'], [1, 10, 20, 2, 30])
    self.assertEqual(res['merge_ignore_default'], [1, 10, 20, 2, 30])
    self.assertEqual(res['roundtrip'], [100, 200, 300, 400, 500])
    self.assertEqual(res['multi_merge'], [1, 10, 100, 2, -1])
    self.assertEqual(res['bool_none_merge'], [10, -1, 30, -2, 50])
    self.assertEqual(res['int_merge'], [-1, 20, -2, 40, -3])

  def test_fby(self):
    logger.setLevel(logging.WARNING)

    def inner():
      init = [10, 20, 30, 40, 50]
      delayed = [1, 2, 3, 4, 5]
      rst = [False, False, False, True, False]
      with reset(rst):
        fby_rst = fby(init, delayed)
        fby_n_rst = fby(init, delayed, 2)
      return (
          fby(init, delayed),
          fby_rst,
          fby(init, delayed, 2),
          fby_n_rst,
      )

    fby_out, fby_rst, fby_n_out, fby_n_rst = Scheduler.run(
        inner, horizon=5, semantics=Semantics.KAHN
    )
    self.assertEqual(fby_out, [10, 1, 2, 3, 4])
    self.assertEqual(fby_rst, [10, 1, 2, 40, 4])
    self.assertEqual(fby_n_out, [10, 10, 1, 2, 3])
    self.assertEqual(fby_n_rst, [10, 10, 1, 40, 40])

    def post_feedback():
      y = fby(
          1, lambda: post(y + 1, 2), 3
      )  # pylint: disable=undefined-variable
      return y

    self.assertEqual(
        Scheduler.run(post_feedback, horizon=8, semantics=Semantics.KAHN),
        [1, 1, 1, 2, 3, 4, 5, 6],
    )
    self.assertFalse(Scheduler.deadlocked)

  def test_buffer(self):
    logger.setLevel(logging.WARNING)

    def inner():
      inp = [10, 20, 30, 40]
      inp_p = [True, True, False, False, True, False, True]
      out_p = [False, False, True, True, False, True, True]
      return buffer(inp, inp_p, out_p), buffer(inp, inp_p, out_p, size=2)

    size_one, size_two = Scheduler.run(
        inner, horizon=7, semantics=Semantics.KAHN
    )
    self.assertEqual(size_one, [20, None, 30, 40])
    self.assertEqual(size_two, [10, 20, 30, 40])

  def test_delay(self):
    logger.setLevel(logging.WARNING)

    def inner():
      inp = [10, 20, 30, 40]
      sched = Stream.periodic([False, False], [True, False])
      return delay(inp), delay(inp, 2), delay(inp, period=sched)

    d1, d2, dp = Scheduler.run(inner, horizon=4, semantics=Semantics.KAHN)
    self.assertEqual(d1, [10, 20, 30, 40])
    self.assertEqual(d2, [10, 20, 30, 40])
    self.assertEqual(dp, [10, 20, 30, 40])

  def test_once_stream(self):
    logger.setLevel(logging.WARNING)

    def inner():
      rst = [False, False, True, False, False]
      with reset(rst):
        once_1_rst = Stream.once(1)
      return Stream.once(0), Stream.once(2), Stream.once(4), once_1_rst

    o0, o2, o4, o1_rst = Scheduler.run(
        inner, horizon=5, semantics=Semantics.KAHN
    )
    self.assertEqual(o0, [True, False, False, False, False])
    self.assertEqual(o2, [False, False, True, False, False])
    self.assertEqual(o4, [False, False, False, False, True])
    self.assertEqual(o1_rst, [False, True, False, True, False])

  def test_periodic_stream(self):
    logger.setLevel(logging.WARNING)

    def inner():
      return Stream.periodic([1, 2], [10, 20]), Stream.periodic([], [7, 8, 9])

    with_prefix, empty_prefix = Scheduler.run(
        inner, horizon=6, semantics=Semantics.KAHN
    )
    self.assertEqual(with_prefix, [1, 2, 10, 20, 10, 20])
    self.assertEqual(empty_prefix, [7, 8, 9, 7, 8, 9])

  def test_pre(self):
    logger.setLevel(logging.WARNING)
    self.assertEqual(
        Scheduler.run(
            lambda: (pre([10, 20, 30, 40]), pre([10, 20, 30, 40], 2)),
            horizon=4,
            semantics=Semantics.KAHN,
        ),
        ([None, 10, 20, 30], [None, None, 10, 20]),
    )

  def test_post(self):
    logger.setLevel(logging.WARNING)

    def inner():
      inp = [10, 20, 30, 40]
      return (
          post(inp),
          post(pre(inp)),
          post(inp, 2),
          post(pre(inp, 2), 2),
      )

    p, p_pre, pn, pn_pre = Scheduler.run(
        inner, horizon=4, semantics=Semantics.KAHN
    )
    self.assertEqual(p, [20, 30, 40, 40])
    self.assertEqual(p_pre, [10, 20, 30])
    self.assertEqual(pn, [30, 40, 40, 40])
    self.assertEqual(pn_pre, [10, 20])

  def test_morph(self):
    logger.setLevel(logging.WARNING)

    def inner():
      first = [10, 20, 30, 40, 50]
      rest = [1, 2, 3, 4, 5]
      rst = [False, False, False, True, False]
      with reset(rst):
        m_rst = morph(first, rest)
        m2_rst = morph(first, rest, 2)
      return morph(first, rest), m_rst, morph(first, rest, 2), m2_rst

    m, m_rst, m2, m2_rst = Scheduler.run(
        inner, horizon=5, semantics=Semantics.KAHN
    )
    self.assertEqual(m, [10, 2, 3, 4, 5])
    self.assertEqual(m_rst, [10, 2, 3, 40, 5])
    self.assertEqual(m2, [10, 20, 3, 4, 5])
    self.assertEqual(m2_rst, [10, 20, 3, 40, 50])

  def test_current(self):
    logger.setLevel(logging.WARNING)

    def inner():
      cond = [False, True, False, False, True, False]
      inp = [10, 20, 30, 40, 50, 60]
      return current(cond, inp), current(cond, inp, default=-1)

    default_d, custom_d = Scheduler.run(
        inner, horizon=6, semantics=Semantics.KAHN
    )
    self.assertEqual(default_d, [None, 20, 20, 20, 50, 50])
    self.assertEqual(custom_d, [-1, 20, 20, 20, 50, 50])

  def test_counter(self):
    logger.setLevel(logging.WARNING)

    def inner():
      rst = [False, False, True, False, False]
      rst2 = [False, False, False, False, True]
      with reset(rst):
        with_rst = counter(start=10)
        with reset(rst2):
          nested_rst = counter(start=10)
      return counter(), counter(start=10), with_rst, nested_rst

    def_s, cust_s, with_rst, nested_rst = Scheduler.run(
        inner, horizon=5, semantics=Semantics.KAHN
    )
    self.assertEqual(def_s, [0, 1, 2, 3, 4])
    self.assertEqual(cust_s, [10, 11, 12, 13, 14])
    self.assertEqual(with_rst, [10, 11, 10, 11, 12])
    self.assertEqual(nested_rst, [10, 11, 10, 11, 10])

  def test_prefix_sum(self):
    logger.setLevel(logging.WARNING)

    def inner():
      inp = [1, 2, 3, 4, 5]
      rst = [False, False, True, False, False]
      with reset(rst):
        with_rst = prefix_sum(inp, start=10)
      return prefix_sum(inp), prefix_sum(inp, start=10), with_rst

    def_s, cust_s, with_rst = Scheduler.run(
        inner, horizon=5, semantics=Semantics.KAHN
    )
    self.assertEqual(def_s, [1, 3, 6, 10, 15])
    self.assertEqual(cust_s, [11, 13, 16, 20, 25])
    self.assertEqual(with_rst, [11, 13, 13, 17, 22])

  def test_reset(self):
    logger.setLevel(logging.WARNING)

    def inner():
      inp = [10, 20, 30, 40, 50]
      init = [100, 200, 300, 400, 500]
      delayed = [1, 2, 3, 4, 5]
      cond = [True, False, True, False, True]
      rst = [False, False, True, False, False]
      rst2 = [False, False, False, False, True]

      with reset(rst):
        scan_r = scan(inp, op=operator.add)
        scan_max_r = scan([3, 5, 1, 4, 2], op=Stream.max)
        fby_r = fby(init, delayed)
        post_r = post(inp)
        morph_r = morph(init, delayed)
        counter_r = counter(start=10)
        psum_r = prefix_sum(inp, start=100)
        when_fby_r = when(cond, fby(init, delayed))
        with reset(rst2):
          nested_counter_r = counter(start=10)

      return {
          'scan': scan_r,
          'scan_max': scan_max_r,
          'fby': fby_r,
          'post': post_r,
          'morph': morph_r,
          'counter': counter_r,
          'prefix_sum': psum_r,
          'when_fby': when_fby_r,
          'nested_counter': nested_counter_r,
      }

    res = Scheduler.run(inner, horizon=5, semantics=Semantics.KAHN)
    self.assertEqual(res['scan'], [10, 30, 30, 70, 120])
    self.assertEqual(res['scan_max'], [3, 5, 1, 4, 4])
    self.assertEqual(res['fby'], [100, 1, 300, 3, 4])
    self.assertEqual(res['post'], [20, 30, 40, 50, 50])
    self.assertEqual(res['morph'], [100, 2, 300, 4, 5])
    self.assertEqual(res['counter'], [10, 11, 10, 11, 12])
    self.assertEqual(res['prefix_sum'], [110, 130, 130, 170, 220])
    self.assertEqual(res['when_fby'], [100, 300, 4])
    self.assertEqual(res['nested_counter'], [10, 11, 10, 11, 10])

  def test_nested_match_subclocks_and_resets(self):
    logger.setLevel(logging.WARNING)

    def inner():
      outer_gate = Stream([True, False, True, True, False, True])
      inner_gate = Stream([True, True, False, True, True, True])
      outer_reset = Stream([False, False, True, False, False, False])
      inner_reset = Stream([False, False, False, True, False, False])
      downsampled = when(outer_gate, [1, 2, 3, 4, 5, 6])

      with match(outer_gate):
        with case(True):
          when_in_case = downsampled + [100, 200, 300, 400, 500, 600]
          when_in_case.name = 'when_in_case'
          with reset(outer_reset):
            counter(start=10, name='gated_counter')
          with match(inner_gate):
            with case(True):
              with reset(inner_reset):
                counter(start=100, name='nested_counter')
            with case(False):
              pass
        with case(False):
          pass

    res = Scheduler.run(
        inner,
        horizon=6,
        semantics=Semantics.KAHN,
        streams=('when_in_case', 'gated_counter', 'nested_counter'),
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(res['when_in_case'], [101, 303, 404, 606])
    self.assertEqual(res['gated_counter'], [10, 10, 11, 12])
    self.assertEqual(res['nested_counter'], [100, 100, 101])

  def test_match(self):
    logger.setLevel(logging.WARNING)

    def inner():
      mode = Stream(
          [_Mode.FAST, _Mode.SLOW, _Mode.FAST, _Mode.SLOW, _Mode.FAST]
      )
      x = Stream([10, 20, 30, 40, 50])
      dyn_x = counter(start=1) * 10
      rst = Stream([False, False, False, False, True])

      with match(mode) as enum_m:
        with case(_Mode.FAST):
          with reset(rst):
            enum_m.out = counter(start=0) + 2 * x
          enum_m.passthrough = dyn_x
          enum_m.seq_or_const = [100, 200, 300, 400, 500]
          enum_m.delayed = lambda: late_x
        with case(_Mode.SLOW):
          enum_m.out = counter(start=100) + x
          enum_m.passthrough = -1
          enum_m.seq_or_const = 7
          enum_m.delayed = 0

      late_x = Stream([1, 2, 3, 4, 5])

      # Guarded partial operation (avoids division by zero) + never-taken arm.
      denom = Stream([2, 0, 5, 0, 10])
      with match(denom) as div_m:
        with case(0):
          div_m.q_sampled = -1
          div_m.q_outer = -1
        with case(-999):
          div_m.q_sampled = counter(start=500)
          div_m.q_outer = 999
        with default() as nz:
          div_m.q_sampled = 100 // nz
          div_m.q_outer = 100 // denom

      # Nested match where inner selector carries invalid tags when outer arm
      # is inactive.
      outer_sel = Stream(['LEFT', 'RIGHT', 'LEFT', 'RIGHT', 'LEFT'])
      inner_sel = Stream([1, 999, 2, 999, 1])
      with match(outer_sel) as nested_m:
        with case('LEFT'):
          with match(inner_sel) as inner_m:
            with case(1):
              inner_m.val = scan(x)
            with case(2):
              inner_m.val = x + 2
          nested_m.out = inner_m.val
        with case():
          nested_m.out = -x

      return enum_m, div_m, nested_m.out

    enum_res, div_res, nested_out = Scheduler.run(
        inner, horizon=5, semantics=Semantics.KAHN
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(enum_res['out'], [20, 120, 61, 141, 100])
    self.assertEqual(enum_res['passthrough'], [10, -1, 30, -1, 50])
    self.assertEqual(enum_res['seq_or_const'], [100, 7, 300, 7, 500])
    self.assertEqual(enum_res['delayed'], [1, 0, 3, 0, 5])
    self.assertEqual(div_res['q_sampled'], [50, -1, 20, -1, 10])
    self.assertEqual(div_res['q_outer'], [50, -1, 20, -1, 10])
    self.assertEqual(nested_out, [10, -20, 32, -40, 60])

  def test_match_last(self):
    logger.setLevel(logging.WARNING)

    def accum(cmd: Stream) -> Stream:
      with match(cmd) as m:
        prev = last(lambda: m.val, 10)
        with case('INC'):
          m.val = prev + 5
        with case('DEC'):
          m.val = prev - 2
        with default():
          m.val = prev
      return m.val

    def inner():
      x = Stream([10, 20, 30, 40, 50, 60])
      sel_ab = Stream(['A', 'A', 'B', 'B', 'A', 'B'])
      with match(sel_ab) as obs_m:
        with case('A'):
          obs_m.p = pre(x)
          obs_m.f = fby(0, x)
          obs_m.l = last(x, 0)
          obs_m.l2 = last(x, 0, 2)
        with case('B'):
          obs_m.p = -1
          obs_m.f = -1
          obs_m.l = -1
          obs_m.l2 = -1

      cmd = Stream(['INC', 'INC', 'HOLD', 'DEC', 'INC', 'DEC'])
      with match(cmd) as inline_m:
        with case('INC'):
          inline_m.val = last(lambda: inline_m.val, 10) + 5
        with case('DEC'):
          inline_m.val = last(lambda: inline_m.val, 10) - 2
        with case('HOLD'):
          inline_m.val = last(lambda: inline_m.val, 10)

      outer_sel = Stream(['ON', 'OFF', 'ON', 'OFF', 'ON', 'ON'])
      sub_cmd = Stream(['INC', 'INC', 'INC', 'DEC', 'DEC', 'INC'])
      with match(outer_sel) as hier_m:
        with case('ON'):
          hier_m.out = accum(sub_cmd)
        with case('OFF'):
          hier_m.out = -1

      return {
          'obs': obs_m,
          'scoped_accum': accum(cmd),
          'inline_accum': inline_m.val,
          'hier_accum': hier_m.out,
      }

    res = Scheduler.run(inner, horizon=6, semantics=Semantics.KAHN)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(res['obs']['p'], [None, 10, -1, -1, 20, -1])
    self.assertEqual(res['obs']['f'], [0, 10, -1, -1, 20, -1])
    self.assertEqual(res['obs']['l'], [0, 10, -1, -1, 40, -1])
    self.assertEqual(res['obs']['l2'], [0, 0, -1, -1, 30, -1])
    self.assertEqual(res['scoped_accum'], [15, 20, 20, 18, 23, 21])
    self.assertEqual(res['inline_accum'], [15, 20, 20, 18, 23, 21])
    self.assertEqual(res['hier_accum'], [15, -1, 20, -1, 18, 23])

  def test_automaton_unsupported(self):
    logger.setLevel(logging.WARNING)

    def automaton_fn():
      with automaton() as fsm:
        with state('A'):
          fsm.out = 1

    def state_fn():
      with state('A'):
        pass

    def unless_fn():
      unless(Stream(True), 'A')

    def until_fn():
      until(Stream(True), 'A')

    for fn in (
        automaton_fn,
        state_fn,
        unless_fn,
        until_fn,
    ):
      with self.assertRaisesRegex(
          RuntimeError, 'only supported under SYNC semantics'
      ):
        Scheduler.run(fn, horizon=2, semantics=Semantics.KAHN)

  def test_deadlock_and_local_starvation(self):
    logger.setLevel(logging.WARNING)

    def causal_starvation():
      starved = when(False, counter()) + 1
      short = post(counter(start=10)) + 1
      healthy = prefix_sum(counter(start=1))
      return starved, short, healthy

    starved, short, healthy = Scheduler.run(
        causal_starvation, horizon=8, semantics=Semantics.KAHN
    )
    self.assertTrue(Scheduler.deadlocked)
    self.assertEqual(starved, [])
    self.assertEqual(short, [12, 13, 14, 15, 16, 17, 18])
    self.assertEqual(healthy, [1, 3, 6, 10, 15, 21, 28, 36])

    def local_starvation_with_progress():
      starved_s = when(False, counter()) + 1
      cyc = ifthenelse(True, lambda: cyc, 0)  # pylint: disable=undefined-variable
      short_s = post(counter(start=10)) + 1
      healthy_s = prefix_sum(counter(start=1))
      return starved_s, cyc, short_s, healthy_s

    starved_s, cyc_s, short_s, healthy_s = Scheduler.run(
        local_starvation_with_progress,
        horizon=8,
        semantics=Semantics.KAHN,
        causality_analysis=False,
    )
    self.assertTrue(Scheduler.deadlocked)
    self.assertEqual(starved_s, [])
    self.assertEqual(cyc_s, [])
    self.assertEqual(short_s, [12, 13, 14, 15, 16, 17, 18])
    self.assertEqual(healthy_s, [1, 3, 6, 10, 15, 21, 28, 36])

  def test_max_min_and_stream_bool(self):
    logger.setLevel(logging.WARNING)
    s = Stream([-1.0, 2.0])
    self.assertIsInstance(s, Stream)
    s_any: Any = s
    with self.assertRaisesRegex(TypeError, 'Truth value of a Stream'):
      bool(s_any)

    with self.assertRaises(ExceptionGroup) as single_arg_ctx:
      Scheduler.run(s_any.max, horizon=1, semantics=Semantics.KAHN)
    self.assertFalse(Scheduler.deadlocked)
    self.assertIsInstance(single_arg_ctx.exception.exceptions[0], TypeError)

    def inner():
      a = Stream([-3, 0, 5, -1])
      b = Stream([2, -2, 1, -4])
      a_any: Any = a
      with self.assertRaisesRegex(TypeError, 'Truth value of a Stream'):
        max(0.0, a_any)
      scalar_max = Stream.max(2, 7)
      self.assertIsInstance(scalar_max, Stream)
      clamped = fby(0, lambda: (clamped + 2).min(5))  # pylint: disable=undefined-variable
      return {
          'scalar_max': scalar_max,
          'max_l': Stream.max(0, a),
          'max_r': a.max(0),
          'min_l': Stream.min(0, a),
          'min_r': a.min(0),
          'max_multi': a.max(b, 1),
          'min_multi': a.min(b, 0),
          'clamped': clamped,
      }

    res = Scheduler.run(inner, horizon=4, semantics=Semantics.KAHN)
    self.assertEqual(res['scalar_max'], [7, 7, 7, 7])
    self.assertEqual(res['max_l'], [0, 0, 5, 0])
    self.assertEqual(res['max_r'], [0, 0, 5, 0])
    self.assertEqual(res['min_l'], [-3, 0, 0, -1])
    self.assertEqual(res['min_r'], [-3, 0, 0, -1])
    self.assertEqual(res['max_multi'], [2, 1, 5, 1])
    self.assertEqual(res['min_multi'], [-3, -2, 0, -4])
    self.assertEqual(res['clamped'], [0, 2, 4, 5])

  def test_resolve_handle(self):
    logger.setLevel(logging.WARNING)
    s = Stream([10, 20])
    h_direct = resolve_handle(s)
    h_thunk = resolve_handle(lambda: s)
    h_nested = resolve_handle(lambda: lambda: s)
    h_static = resolve_handle([7, 8])
    self.assertIs(h_direct.istruct, s.istruct)
    self.assertIs(h_thunk.istruct, s.istruct)
    self.assertIs(h_nested.istruct, s.istruct)
    self.assertTrue(h_static.istruct.static)

  def test_absent_elision_equivalence_with_sync(self):
    logger.setLevel(logging.WARNING)
    horizon = 8

    def program():
      xs = Stream([10, 20, 30, 40, 50, 60, 70, 80])
      ys = Stream([1, 2, 3, 4, 5, 6, 7, 8])
      cond = Stream([True, False, True, True, False, True, False, False])
      not_cond = ~cond
      base_rst = Stream([False, False, True, False, False, False, True, False])

      # Base-clock primitives and same-clock modular reset:
      fby_base = fby(99, xs)
      pre_base = pre(xs)
      post_base = post(xs)
      post_pre_base = post(pre(xs))
      morph_base = morph(xs * 10, ys)
      current_base = current(cond, xs, default=-1)
      scan_base = scan(ys)
      with reset(base_rst):
        fby_rst = fby(xs * 10, ys)
        post_rst = post(xs)
        morph_rst = morph(xs * 10, ys)
        scan_rst = scan(xs)
        counter_rst = counter(start=5)
        psum_rst = prefix_sum(ys, start=100)
        once_rst = Stream.once(1)

      # Subclock sampling + merge (gated by match/case when computing on
      # subclocks so all Kahn nodes complete the base horizon without EOF
      # starvation):
      w_true = when(cond, (xs + 1) * 2)
      w_false = whennot(cond, -ys)
      when_neg_equiv = when(not_cond, -ys)
      merged_bool = merge(cond, w_true, w_false)
      with match(cond) as morph_m:
        with case(True):
          morph_sub = morph(w_true, ys, name='morph_sub')
          morph_m.val = morph_sub
        with case(False):
          morph_m.val = 0

      # Buffer (lossless, overwriting, under base reset, and inside match/case):
      ip = Stream([True, True, False, False, True, False, True, False])
      op = Stream([False, True, True, False, False, True, True, False])
      buf2 = buffer(when(ip, xs), ip, op, size=2)
      buf1 = buffer(when(ip, xs), ip, op, size=1)
      with reset(base_rst):
        buf_rst = buffer(when(ip, xs), ip, op, size=2)
      with match(op) as buf_m:
        with case(True):
          buf_active = buf2 + ys + scan(ys)
          buf_active.name = 'buf_active'
          buf_m.val = buf_active
        with case(False):
          buf_m.val = 0

      # when + nested match/case + arm-local reset:
      inner_gate = Stream([True, True, False, True, True, True, False, True])
      inner_rst = Stream(
          [False, False, False, False, False, True, False, False]
      )
      with match(cond) as outer_sub_m:
        with case(True):
          sub_sum = w_true + ys + scan(xs)
          sub_sum.name = 'sub_sum'
          post(w_true, name='post_sub')
          with reset(base_rst):
            counter(start=10, name='sub_cnt')
            post(ys, name='post_sub_rst')
          with match(inner_gate) as inner_sub_m:
            with case(True):
              w_inner = when(True, w_true) + counter(start=1)
              w_inner.name = 'w_inner'
              with reset(inner_rst):
                nested_scan = scan(ys, name='nested_scan')
              inner_sub_m.val = w_inner + nested_scan
            with case(False):
              inner_sub_m.val = 0
          outer_sub_m.val = inner_sub_m.val
        with case(False):
          outer_sub_m.val = 0

      # match / case / default / multi-tag / bool-selector / last:
      tags = Stream(['A', 'B', 'C', 'A', 'Z', 'B', 'A', 'Z'])
      with match(tags) as tag_m:
        prev = last(lambda: tag_m.acc, 0)
        with case('A', 'B'):
          with reset(base_rst):
            tag_m.acc = prev + counter(start=1) + ys
        with case('C'):
          tag_m.acc = prev + 100
        with default():
          tag_m.acc = prev - 1

      with match(cond) as bool_m:
        with case(True):
          bool_m.val = w_true + counter(start=0)
        with default():
          bool_m.val = w_false - 1

      with reset(base_rst):
        with match(tags) as rst_last_m:
          prev_r = last(lambda: rst_last_m.total, 50)
          with case('A'):
            rst_last_m.total = prev_r + xs
          with default():
            rst_last_m.total = prev_r - ys

      return {
          'fby_base': fby_base,
          'pre_base': pre_base,
          'post_base': post_base,
          'post_pre_base': post_pre_base,
          'morph_base': morph_base,
          'current_base': current_base,
          'scan_base': scan_base,
          'fby_rst': fby_rst,
          'post_rst': post_rst,
          'morph_rst': morph_rst,
          'scan_rst': scan_rst,
          'counter_rst': counter_rst,
          'psum_rst': psum_rst,
          'once_rst': once_rst,
          'when_sub': w_true,
          'whennot_sub': w_false,
          'when_neg_equiv': when_neg_equiv,
          'merged_bool': merged_bool,
          'morph_match': morph_m.val,
          'buf2': buf2,
          'buf1': buf1,
          'buf_rst': buf_rst,
          'buf_match': buf_m.val,
          'outer_sub_match': outer_sub_m.val,
          'tag_match': tag_m.acc,
          'bool_match': bool_m.val,
          'rst_last_match': rst_last_m.total,
      }

    sync_traces = Scheduler.run(
        program, horizon=horizon, semantics=Semantics.SYNC, clock_calculus=True
    )
    self.assertFalse(Scheduler.deadlocked)

    kahn_traces = Scheduler.run(
        program, horizon=horizon, semantics=Semantics.KAHN, clock_calculus=True
    )
    self.assertFalse(Scheduler.deadlocked)

    for key, sync_vals in sync_traces.items():
      with self.subTest(output=key):
        self.assertEqual(present_values(sync_vals), kahn_traces[key])

    sub_names = (
        'morph_sub',
        'buf_active',
        'sub_sum',
        'post_sub',
        'sub_cnt',
        'post_sub_rst',
        'w_inner',
        'nested_scan',
    )
    sync_sub_traces = Scheduler.run(
        program,
        horizon=horizon,
        semantics=Semantics.SYNC,
        clock_calculus=True,
        streams=sub_names,
    )
    self.assertFalse(Scheduler.deadlocked)
    kahn_sub_traces = Scheduler.run(
        program,
        horizon=horizon,
        semantics=Semantics.KAHN,
        clock_calculus=True,
        streams=sub_names,
    )
    self.assertFalse(Scheduler.deadlocked)
    for key, sync_vals in sync_sub_traces.items():
      with self.subTest(subclock_output=key):
        self.assertEqual(present_values(sync_vals), kahn_sub_traces[key])

    # Ungated Absent-strict combinational pipelines directly on a `when`
    # subclock also preserve Absent-elision across all emitted tokens within
    # the horizon:
    def ungated_subclock_pipeline():
      cond = Stream([True, False, True, True, False, True])
      xs = Stream([10, 20, 30, 40, 50, 60])
      return (when(cond, xs) + 1) * 2

    sync_w_raw = Scheduler.run(
        ungated_subclock_pipeline,
        horizon=6,
        semantics=Semantics.SYNC,
        clock_calculus=True,
    )
    sync_w = present_values(sync_w_raw)

    kahn_w = Scheduler.run(
        ungated_subclock_pipeline,
        horizon=6,
        semantics=Semantics.KAHN,
        clock_calculus=True,
    )
    self.assertEqual(kahn_w, sync_w)

    # Horizon-shortened `post` pipelines (where downstream consumers in KAHN
    # block at the horizon boundary after `post` drops initial token(s)) also
    # satisfy Absent-elision across all emitted values:
    def horizon_shortened_post_pipelines():
      cond = Stream([True, False, True, True, False, True])
      xs = Stream([10, 20, 30, 40, 50, 60])
      ys = Stream([1, 2, 3, 4, 5, 6])
      w_true = when(cond, xs)
      base_final = counter() == 5
      post_plus = post(xs) + 1
      post_chain = post(post(xs))
      post_bounded = merge(base_final, 0, whennot(base_final, post(xs)))
      with match(cond) as sub_m:
        with case(True):
          sub_final = counter() == 3
          sub_post_plus = post(w_true) + ys
          sub_post_plus.name = 'sub_post_plus'
          post(post(w_true), name='sub_post_chain')
          sub_post_bounded = merge(
              sub_final,
              0,
              whennot(sub_final, post(w_true)),
              name='sub_post_bounded',
          )
          sub_m.val = sub_post_bounded
        with case(False):
          sub_m.val = 0
      return {
          'post_plus': post_plus,
          'post_chain': post_chain,
          'post_bounded': post_bounded,
          'sub_match_merged': sub_m.val,
      }

    sync_post_traces = Scheduler.run(
        horizon_shortened_post_pipelines,
        horizon=6,
        semantics=Semantics.SYNC,
        clock_calculus=True,
    )
    kahn_post_traces = Scheduler.run(
        horizon_shortened_post_pipelines,
        horizon=6,
        semantics=Semantics.KAHN,
        clock_calculus=True,
    )
    for key, sync_vals in sync_post_traces.items():
      with self.subTest(post_pipeline=key):
        self.assertEqual(present_values(sync_vals), kahn_post_traces[key])

    sub_post_names = ('sub_post_plus', 'sub_post_chain', 'sub_post_bounded')
    sync_sub_post = Scheduler.run(
        horizon_shortened_post_pipelines,
        horizon=6,
        semantics=Semantics.SYNC,
        clock_calculus=True,
        streams=sub_post_names,
    )
    kahn_sub_post = Scheduler.run(
        horizon_shortened_post_pipelines,
        horizon=6,
        semantics=Semantics.KAHN,
        clock_calculus=True,
        streams=sub_post_names,
    )
    for key, sync_vals in sync_sub_post.items():
      with self.subTest(sub_post_pipeline=key):
        self.assertEqual(present_values(sync_vals), kahn_sub_post[key])


if __name__ == '__main__':
  absltest.main()
