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

import io
import logging
import operator
from typing import Any
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.automaton import (
    automaton,
    state,
    unless,
    until,
)
from psyre.common import (
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
    reset,
    scan,
    switch,
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
from psyre.stream import Stream
from tests import test_utils

# pylint: enable=g-importing-member,g-multiple-import

_Mode = test_utils.Mode


class _Vec:
  """Minimal object supporting the @ operator for testing Stream matmul."""

  def __init__(self, value: int):
    self.value = value

  def __matmul__(self, other: '_Vec') -> int:
    if not isinstance(other, _Vec):
      return NotImplemented
    return self.value * other.value


class KahnTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def _assert_absent_bottom_elision(
      self, sync_vals: list[Any], kahn_vals: list[Any]
  ) -> None:
    """Asserts unpadded KAHN trace has no Absent/Bottom and matches SYNC."""
    test_utils.assert_absent_bottom_elision(self, sync_vals, kahn_vals)

  def _run_and_assert_elision(
      self,
      prog: Any,
      *,
      horizon: int,
      streams: tuple[str, ...] | None = None,
      label: str = 'output',
  ) -> tuple[Any, Any]:
    """Runs `prog` in SYNC and KAHN and checks Absent/Bottom-elision match."""
    sync_res = Scheduler.run(
        prog,
        horizon=horizon,
        semantics=Semantics.SYNC,
        clock_calculus=True,
        streams=streams,
    )
    self.assertFalse(Scheduler.deadlocked)
    kahn_res = Scheduler.run(
        prog,
        horizon=horizon,
        semantics=Semantics.KAHN,
        clock_calculus=True,
        streams=streams,
    )
    self.assertFalse(Scheduler.deadlocked)
    if isinstance(sync_res, dict):
      for key, sync_vals in sync_res.items():
        with self.subTest(**{label: key}):
          self._assert_absent_bottom_elision(sync_vals, kahn_res[key])
    else:
      self._assert_absent_bottom_elision(sync_res, kahn_res)
    return sync_res, kahn_res

  def test_experiments(self):
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

    logger.setLevel(logging.INFO)
    try:
      self.assertEqual(
          Scheduler.run(inner, horizon=6, semantics=Semantics.KAHN),
          [0, 1, 1, 1, 20, 20],
      )
    finally:
      logger.setLevel(logging.WARNING)

  def test_operators(self):
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

  def test_ifthenelse(self):
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
    def inner():
      cond = [True, False, True, False, True]
      inp = [10, 20, 30, 40, 50]
      return when(cond, inp), whennot(cond, inp)

    w_out, wn_out = Scheduler.run(inner, horizon=5, semantics=Semantics.KAHN)
    self.assertEqual(w_out, [10, 30, 50])
    self.assertEqual(wn_out, [20, 40])

  def test_merge(self):
    def inner():
      cond = Stream([True, False, False, True, False])
      true_branch = [1, 2]
      false_branch = [10, 20, 30]
      inp = [100, 200, 300, 400, 500]
      sel = ['a', 'b', 'c', 'a', 'z']
      typed_sel = [True, 1, False, 0, None]
      return {
          'merge': merge(cond, true_branch, false_branch),
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
    self.assertEqual(res['roundtrip'], [100, 200, 300, 400, 500])
    self.assertEqual(res['multi_merge'], [1, 10, 100, 2, -1])
    self.assertEqual(res['bool_none_merge'], [10, -1, 30, -2, 50])
    self.assertEqual(res['int_merge'], [-1, 20, -2, 40, -3])

    for op_fn in (merge, switch):
      with self.assertRaisesRegex(ValueError, 'does not accept default='):
        Scheduler.run(
            lambda fn=op_fn: fn(
                Stream([True, False]), [1, 2], [3, 4], default=[-1]
            ),
            horizon=2,
            semantics=Semantics.KAHN,
        )

  def test_fby(self):
    def inner():
      init = Stream([10, 20, 30, 40, 50])
      delayed = Stream([1, 2, 3, 4, 5])
      rst = Stream([False, False, False, True, False])
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
    def inner():
      inp = [10, 20, 30, 40]
      inp_p = [True, True, False, False, True, False, True]
      out_p = [False, False, True, True, False, True, True]
      return buffer(inp, inp_p, out_p, size=2)

    size_two = Scheduler.run(inner, horizon=7, semantics=Semantics.KAHN)
    self.assertEqual(size_two, [10, 20, 30, 40])

    # Valid `None` stream values flow cleanly through `buffer` in KAHN.
    def none_payload():
      inp = [None, 20, None, 40]
      inp_p = [True, True, False, False, True, False, True]
      out_p = [False, False, True, True, False, True, True]
      return buffer(inp, inp_p, out_p, size=2)

    self.assertEqual(
        Scheduler.run(none_payload, horizon=7, semantics=Semantics.KAHN),
        [None, 20, None, 40],
    )

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
            self, prog, horizon, pattern, semantics=Semantics.KAHN
        )

  def test_once_stream(self):
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
    def inner():
      return Stream.periodic([1, 2], [10, 20]), Stream.periodic([], [7, 8, 9])

    with_prefix, empty_prefix = Scheduler.run(
        inner, horizon=6, semantics=Semantics.KAHN
    )
    self.assertEqual(with_prefix, [1, 2, 10, 20, 10, 20])
    self.assertEqual(empty_prefix, [7, 8, 9, 7, 8, 9])

  def test_pre(self):
    self.assertEqual(
        Scheduler.run(
            lambda: (pre([10, 20, 30, 40]), pre([10, 20, 30, 40], 2)),
            horizon=4,
            semantics=Semantics.KAHN,
        ),
        ([None, 10, 20, 30], [None, None, 10, 20]),
    )

  def test_post(self):
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
    def inner():
      first = Stream([10, 20, 30, 40, 50])
      rest = Stream([1, 2, 3, 4, 5])
      rst = Stream([False, False, False, True, False])
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
    def inner():
      rst = Stream([False, False, True, False, False])
      rst2 = Stream([False, False, False, False, True])
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
    def inner():
      inp = Stream([1, 2, 3, 4, 5])
      rst = Stream([False, False, True, False, False])
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
    def inner():
      inp = Stream([10, 20, 30, 40, 50])
      init = Stream([100, 200, 300, 400, 500])
      delayed = Stream([1, 2, 3, 4, 5])
      cond = Stream([True, False, True, False, True])
      rst = Stream([False, False, True, False, False])
      max_inp = Stream([3, 5, 1, 4, 2])

      with reset(rst):
        scan_r = scan(inp, op=operator.add)
        scan_max_r = scan(max_inp, op=Stream.max)
        fby_r = fby(init, delayed)
        post_r = post(inp)
        morph_r = morph(init, delayed)
        when_fby_r = when(cond, fby(init, delayed))

      return {
          'scan': scan_r,
          'scan_max': scan_max_r,
          'fby': fby_r,
          'post': post_r,
          'morph': morph_r,
          'when_fby': when_fby_r,
      }

    res = Scheduler.run(inner, horizon=5, semantics=Semantics.KAHN)
    self.assertEqual(res['scan'], [10, 30, 30, 70, 120])
    self.assertEqual(res['scan_max'], [3, 5, 1, 4, 4])
    self.assertEqual(res['fby'], [100, 1, 300, 3, 4])
    self.assertEqual(res['post'], [20, 30, 40, 50, 50])
    self.assertEqual(res['morph'], [100, 2, 300, 4, 5])
    self.assertEqual(res['when_fby'], [100, 300, 4])

  def test_nested_match_subclocks_and_resets(self):
    def inner():
      outer_gate = Stream([True, False, True, True, False, True])
      inner_gate = Stream([True, False, True, True])
      outer_reset = Stream([False, True, False, False])
      inner_reset = Stream([False, True, False])
      downsampled = when(outer_gate, [1, 2, 3, 4, 5, 6])

      with match(outer_gate):
        with case(True):
          when_in_case = downsampled + [100, 200, 300, 400]
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
    self.assertEqual(res['when_in_case'], [101, 203, 304, 406])
    self.assertEqual(res['gated_counter'], [10, 10, 11, 12])
    self.assertEqual(res['nested_counter'], [100, 100, 101])

  def test_match(self):
    def inner():
      mode = Stream(
          [_Mode.FAST, _Mode.SLOW, _Mode.FAST, _Mode.SLOW, _Mode.FAST]
      )
      x = Stream([10, 20, 30, 40, 50])
      dyn_x = counter(start=1) * 10
      rst = Stream([False, False, True])

      with match(mode) as enum_m:
        with case(_Mode.FAST):
          with reset(rst):
            enum_m.out = counter(start=0) + 2 * x
          enum_m.passthrough = dyn_x
          enum_m.seq_or_const = [100, 200, 300]
          enum_m.delayed = lambda: late_x
        with case(_Mode.SLOW):
          enum_m.out = counter(start=100) + x
          enum_m.passthrough = -1
          enum_m.seq_or_const = 7
          enum_m.delayed = 0

      late_x = Stream([1, 2, 3, 4, 5])

      # Guarded partial operation (avoids division by zero) + never-taken arm.
      # `denom` is made a dynamic base-clock stream so `match` downsamples
      # `denom` (skipping zeros) in the default arm.
      denom = when(Stream(True), [2, 0, 5, 0, 10])
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

      # Nested match where a dynamic base-clock inner selector carries invalid
      # tags when the outer arm is inactive.
      outer_sel = Stream(['LEFT', 'RIGHT', 'LEFT', 'RIGHT', 'LEFT'])
      inner_sel = when(Stream(True), [1, 999, 2, 999, 1])
      with match(outer_sel) as nested_m:
        with case('LEFT'):
          with match(inner_sel) as inner_m:
            with case(1):
              inner_m.val = scan(dyn_x)
            with case(2):
              inner_m.val = dyn_x + 2
          nested_m.out = inner_m.val
        with case():
          nested_m.out = -dyn_x

      return enum_m, div_m, nested_m.out

    enum_res, div_res, nested_out = Scheduler.run(
        inner, horizon=5, semantics=Semantics.KAHN
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(enum_res['out'], [20, 110, 41, 121, 60])
    self.assertEqual(enum_res['passthrough'], [10, -1, 30, -1, 50])
    self.assertEqual(enum_res['seq_or_const'], [100, 7, 200, 7, 300])
    self.assertEqual(enum_res['delayed'], [1, 0, 2, 0, 3])
    self.assertEqual(div_res['q_sampled'], [50, -1, 20, -1, 10])
    self.assertEqual(div_res['q_outer'], [50, -1, 20, -1, 10])
    self.assertEqual(nested_out, [10, -20, 32, -40, 60])

  def test_match_last(self):
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
      sub_cmd = Stream(['INC', 'INC', 'DEC', 'INC'])
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

  def test_automaton_and_cross_activity_reset(self):
    def program():
      # 1. Instantaneous and internally-delayed cross-activity resets:
      c = Stream([True, False, True, False, True, True])
      r = Stream([False, True, False, False, False, True])
      with reset(r):
        with match(c) as sup_m:
          with case(True):
            sup_m.cnt = counter(start=10)
          with case(False):
            sup_m.cnt = 0

      # 2. Static-False gate under cross-activity reset terminates cleanly:
      with reset(r):
        with match(Stream(False)) as dead_m:
          with case(True):
            dead_m.out = counter(start=1)
          with case(False):
            dead_m.out = -1

      # 3. Resetting subclock stateful nodes outside `when` or `match`
      #    (directly, over `buffer`, and with an extra `when`-guarded reset):
      x = Stream([10, 20, 30, 40, 50, 60])
      r_d = Stream([False, False, True, False, False, False])
      with reset(r):
        coarse_fby = fby(99, when(c, x))
        coarse_scan = scan(when(c, x))
        with reset(when(c, False)):
          manual_fby = fby(99, when(c, x))
      with reset(r_d):
        buf_fby = fby(99, buffer(x, True, morph(False, True, n=1), size=3))

      # 4. Hierarchical automaton with outer reset=True and inner reset=False:
      outer_flip = Stream([False, False, True, True, False, False])
      inner_flip = Stream([True, False, False, False, True, False])
      with automaton(init='ON') as outer:
        with state('ON'):
          with automaton(init='X') as inner_fsm:
            with state('X'):
              until(inner_flip, 'Y', reset=False)
              inner_fsm.y = counter(start=0)
            with state('Y', unless(r, 'X', reset=True)):
              until(inner_flip, 'X', reset=False)
              inner_fsm.y = counter(start=10)
          until(outer_flip, 'OFF', reset=True)
          outer.out = inner_fsm.y
        with state('OFF'):
          until(outer_flip, 'ON', reset=True)
          outer.out = -1

      return {
          'sup_cnt': sup_m.cnt,
          'dead_gate': dead_m.out,
          'coarse_fby': coarse_fby,
          'coarse_scan': coarse_scan,
          'manual_fby': manual_fby,
          'buf_fby': buf_fby,
          'hier_fsm': outer.out,
      }

    _, kahn_res = self._run_and_assert_elision(program, horizon=6)
    self.assertEqual(kahn_res['sup_cnt'], [10, 0, 10, 0, 11, 10])
    self.assertEqual(kahn_res['dead_gate'], [-1, -1, -1, -1, -1, -1])
    self.assertEqual(kahn_res['coarse_fby'], [99, 99, 30, 99])
    self.assertEqual(kahn_res['coarse_scan'], [10, 30, 80, 60])
    self.assertEqual(kahn_res['manual_fby'], [99, 99, 30, 99])
    self.assertEqual(kahn_res['buf_fby'], [99, 99, 30, 40])

    # Base-clock KAHN programs with resets still run when clock_calculus=False:
    def base_clock_reset():
      r = Stream([False, False, True, False, False, False])
      with reset(r):
        return counter(start=1)

    base_cnt = Scheduler.run(
        base_clock_reset,
        horizon=6,
        semantics=Semantics.KAHN,
        clock_calculus=False,
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(base_cnt, [1, 2, 1, 2, 3, 4])

  def test_deadlock_and_local_starvation(self):
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

  def test_absent_elision_equivalence_with_sync(self):
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

      # Buffer (size=2, size=1, under base reset, and inside match/case):
      ip = Stream([True, True, False, False, True, False, True, False])
      op = Stream([False, True, True, False, False, True, True, False])
      ip1 = Stream([True, False, True, True, False, True, False, False])
      op1 = Stream([False, True, True, False, True, True, False, False])
      ip_rst = Stream([True, True, True, False, True, False, True, False])
      op_rst = Stream([False, True, False, True, False, True, True, False])
      buf2 = buffer(when(ip, xs), ip, op, size=2)
      buf1 = buffer(when(ip1, xs), ip1, op1, size=1)
      with reset(base_rst):
        buf_rst = buffer(when(ip_rst, xs), ip_rst, op_rst, size=2)
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

      # Cross-activity / supersampled resets (both delayed on inactive ticks
      # and instantaneous on active ticks, single and nested match + reset):
      super_rst = Stream(
          [False, True, False, False, True, True, False, False]
      )
      with reset(super_rst):
        with match(cond) as sup_m:
          with case(True):
            sup_fby = fby(xs * 10, ys, name='sup_fby')
            sup_pre = pre(xs, name='sup_pre')
            sup_cnt = counter(start=10, name='sup_cnt')
            sup_scan = scan(xs, name='sup_scan')
            sup_psum = prefix_sum(ys, start=100, name='sup_psum')
            sup_morph = morph(xs * 10, ys, name='sup_morph')
            sup_cur = current(inner_gate, xs, default=-1, name='sup_cur')
            sup_once = Stream.once(1, name='sup_once')
            post(xs, name='sup_post')
            sup_buf = buffer(
                when(inner_gate, xs), inner_gate, inner_gate, size=2
            )
            with reset(inner_rst):
              with match(inner_gate) as sup_nested_m:
                with case(True):
                  sup_nested_cnt = counter(start=50, name='sup_nested_cnt')
                  sup_nested_m.val = sup_nested_cnt + ys
                with case(False):
                  sup_nested_m.val = 0
            sup_m.val = (
                sup_fby
                + sup_cnt
                + sup_scan
                + sup_psum
                + sup_morph
                + sup_cur
                + merge(inner_gate, sup_buf, 0)
                + sup_nested_m.val
            )
            sup_m.once = sup_once
            sup_m.pre = sup_pre
          with case(False):
            sup_m.val = 0
            sup_m.once = False
            sup_m.pre = -1

      # Flat and hierarchical automata (weak until, strong unless, reset=True
      # and reset=False, last(), and enclosing outer reset):
      abort = Stream([False, False, False, True, False, False, False, False])
      wake = Stream([False, False, False, False, True, False, False, False])
      with automaton(init='RUN') as flat_fsm:
        prev_v = last(lambda: flat_fsm.v, init=0)
        with state('RUN', unless(abort, 'IDLE')):
          c_run = counter(start=1, name='flat_run_cnt')
          until(c_run >= 2, 'HOLD', reset=False)
          flat_fsm.v = prev_v + c_run
        with state('HOLD'):
          until(True, 'RUN', reset=False)
          flat_fsm.v = prev_v + 100
        with state('IDLE', unless(wake, 'RUN', reset=True)):
          flat_fsm.v = counter(start=500, name='flat_idle_cnt')

      outer_flip = Stream(
          [False, False, True, True, False, False, True, False]
      )
      inner_flip = Stream(
          [True, False, False, False, False, True, False, False]
      )
      with reset(super_rst):
        with automaton(init='ON') as hier_fsm:
          with state('ON'):
            with automaton(init='X') as inner_fsm:
              with state('X', unless(abort, 'Y')):
                inner_x = counter(start=1, name='hier_inner_x')
                until(inner_flip, 'Y', reset=False)
                inner_fsm.z = last(lambda: inner_fsm.z, 10) + inner_x
              with state('Y'):
                inner_y = counter(start=20, name='hier_inner_y')
                until(inner_flip, 'X', reset=False)
                inner_fsm.z = last(lambda: inner_fsm.z, 10) + inner_y
            until(outer_flip, 'OFF', reset=True)
            hier_fsm.out = inner_fsm.z
          with state('OFF'):
            until(outer_flip, 'ON', reset=True)
            hier_fsm.out = -1

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
          'sup_match_val': sup_m.val,
          'sup_match_once': sup_m.once,
          'sup_match_pre': sup_m.pre,
          'flat_fsm': flat_fsm.v,
          'hier_fsm': hier_fsm.out,
      }

    self._run_and_assert_elision(program, horizon=horizon)

    sub_names = (
        'morph_sub',
        'buf_active',
        'sub_sum',
        'post_sub',
        'sub_cnt',
        'post_sub_rst',
        'w_inner',
        'nested_scan',
        'sup_fby',
        'sup_pre',
        'sup_cnt',
        'sup_scan',
        'sup_psum',
        'sup_morph',
        'sup_cur',
        'sup_once',
        'sup_post',
        'sup_nested_cnt',
        'flat_run_cnt',
        'flat_idle_cnt',
        'hier_inner_x',
        'hier_inner_y',
    )
    self._run_and_assert_elision(
        program,
        horizon=horizon,
        streams=sub_names,
        label='subclock_output',
    )

    # Ungated Absent-strict combinational pipelines directly on a `when`
    # subclock also preserve Absent-elision across all emitted tokens within
    # the horizon:
    def ungated_subclock_pipeline():
      cond = Stream([True, False, True, True, False, True])
      xs = Stream([10, 20, 30, 40, 50, 60])
      return (when(cond, xs) + 1) * 2

    self._run_and_assert_elision(ungated_subclock_pipeline, horizon=6)

    # Horizon-shortened `post` pipelines (where downstream consumers in KAHN
    # block at the horizon boundary after `post` drops initial token(s)) also
    # satisfy Absent/Bottom-elision across all emitted values:
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

    sync_post_traces, _ = self._run_and_assert_elision(
        horizon_shortened_post_pipelines, horizon=6, label='post_pipeline'
    )
    self.assertIs(sync_post_traces['post_chain'][-1], Bottom)

    sub_post_names = ('sub_post_plus', 'sub_post_chain', 'sub_post_bounded')
    sync_sub_post, _ = self._run_and_assert_elision(
        horizon_shortened_post_pipelines,
        horizon=6,
        streams=sub_post_names,
        label='sub_post_pipeline',
    )
    self.assertIs(sync_sub_post['sub_post_plus'][-1], Bottom)
    self.assertIs(sync_sub_post['sub_post_chain'][-1], Bottom)

  def test_kahn_shared_stopped_stream_with_post_and_non_post_consumers(self):
    # Cancelling a post() or arm-scoped waiter on a shared stopped stream must
    # not cancel sibling waiters or hang deadlock detection.
    def shared_post_and_comb(lookahead: int):
      p = post(counter(start=10), lookahead)
      c1 = post(p)
      c2 = p + 1
      c3 = c2 + 1
      return p, c1, c2, c3

    p_tr, c1_tr, c2_tr, c3_tr = Scheduler.run(
        shared_post_and_comb, 1, horizon=4, semantics=Semantics.KAHN
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(p_tr, [11, 12, 13])
    self.assertEqual(c1_tr, [12, 13])
    self.assertEqual(c2_tr, [12, 13, 14])
    self.assertEqual(c3_tr, [13, 14, 15])

    ep, ec1, ec2, ec3 = Scheduler.run(
        shared_post_and_comb, 4, horizon=4, semantics=Semantics.KAHN
    )
    self.assertTrue(Scheduler.deadlocked)
    self.assertEqual((ep, ec1, ec2, ec3), ([], [], [], []))

    def shared_when_arm_and_outer_consumer():
      sel = Stream([True, False, False, False])
      short = when(sel, counter(start=1))
      with match(sel) as m:
        with case(True):
          m.arm = short + 10
        with case(False):
          m.arm = 0
      outer = short + 1
      tail = outer + 1
      return m.arm, outer, tail

    arm_tr, outer_tr, tail_tr = Scheduler.run(
        shared_when_arm_and_outer_consumer,
        horizon=4,
        semantics=Semantics.KAHN,
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(arm_tr, [11, 0, 0, 0])
    self.assertEqual(outer_tr, [2])
    self.assertEqual(tail_tr, [3])

  def test_kahn_task_exception_does_not_deadlock_or_leak_buffers(self):
    @psyre_comb
    def boom_tick0(val):
      raise ValueError(f'boom at {val}')

    @psyre_comb
    def boom_tick2(val):
      if val >= 2:
        raise ValueError(f'later boom at {val}')
      return val + 1

    for sem in (Semantics.KAHN, Semantics.SYNC):
      for label, fn, match_msg in (
          ('tick0', boom_tick0, 'boom at 0'),
          ('tick2', boom_tick2, 'later boom at 2'),
      ):
        with self.subTest(semantics=sem, case=label):
          handles: dict[str, Stream] = {}

          def cyclic_fail(step_fn, out_handles):
            a = fby(0, lambda: b, name='a')  # pylint: disable=undefined-variable
            b = step_fn(a)
            out_handles['a'] = a
            out_handles['b'] = b
            return a, b

          stderr = io.StringIO()
          with mock.patch('sys.stderr', stderr):
            with self.assertRaises(ExceptionGroup) as eg:
              Scheduler.run(
                  cyclic_fail,
                  fn,
                  handles,
                  horizon=5,
                  semantics=sem,
              )
          self.assertFalse(Scheduler.deadlocked)
          self.assertIsInstance(eg.exception.exceptions[0], ValueError)
          self.assertIn(match_msg, str(eg.exception.exceptions[0]))
          self.assertEqual(stderr.getvalue(), '')
          self.assertEmpty(handles['a'].seq)
          self.assertEmpty(handles['b'].seq)

  def test_deferred_mux_dedup_and_post_skip_parity(self):
    def mux_thunk_dedup():
      t = Stream([0, 1, 2, 0, 1, 2])
      x = counter(10)
      w = when(t, x, values=(0, 1))
      w2 = when(t, counter(100), values=2)
      return merge(t, {0: lambda: w, 1: lambda: w, 2: w2})

    self.assertEqual(
        Scheduler.run(mux_thunk_dedup, horizon=6, semantics=Semantics.SYNC),
        [10, 11, 102, 13, 14, 105],
    )
    for cc in (True, False):
      with self.subTest(case='mux_thunk', semantics=Semantics.KAHN, cc=cc):
        self.assertEqual(
            Scheduler.run(
                mux_thunk_dedup,
                horizon=6,
                semantics=Semantics.KAHN,
                clock_calculus=cc,
            ),
            [10, 11, 102, 13, 14, 105],
        )

    def post_discard_prog():
      src_hi = post(counter(10), 4)
      src_def = post(counter(100), 4)
      src_cur = post(counter(20), 4)
      return {
          'fby1': fby(src_hi, counter(0)),
          'fby2': fby(src_hi, counter(0), 2),
          'morph': morph(src_hi, counter(0)),
          'current': current(
              Stream([False, True, False, False, False, False]),
              src_cur,
              default=src_def,
          ),
      }

    kahn_res = Scheduler.run(
        post_discard_prog, horizon=6, semantics=Semantics.KAHN
    )
    sync_res = Scheduler.run(
        post_discard_prog, horizon=6, semantics=Semantics.SYNC
    )
    self.assertEqual(kahn_res, sync_res)
    self.assertEqual(kahn_res['fby1'], [14, 0, 1, 2, 3, 4])
    self.assertEqual(kahn_res['fby2'], [14, 14, 0, 1, 2, 3])
    self.assertEqual(kahn_res['morph'], [14, 1, 2, 3, 4, 5])
    self.assertEqual(kahn_res['current'], [104, 25, 25, 25, 25, 25])

  def test_none_as_automaton_state_and_match_case(self):
    def prog():
      flip = Stream([False, True, False, True])
      with automaton(init=None) as fsm:
        with state(None):
          until(flip, 'ON')
          fsm.out = None
        with state('ON'):
          until(flip, None)
          fsm.out = 1
      with match(fsm.out) as m:
        with case(None):
          m.tag = 'none'
        with case(1):
          m.tag = None
      return fsm.out, m.tag

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        out, tag = Scheduler.run(prog, horizon=4, semantics=sem)
        self.assertEqual(out, [None, None, 1, 1])
        self.assertEqual(tag, ['none', 'none', None, None])

  def test_subclock_current_and_counter_tick_sync_kahn_parity(self):
    def prog():
      nat = counter(0, name='nat')
      is_odd = (nat % 2) == 1
      is_even = ~is_odd
      c_dyn = (nat % 3) == 0
      d_stat = Stream([False, True, True, False])
      ind_tag = Stream(['A', 'B', 'A', 'C', 'A', 'B', 'C', 'A'])
      ind_bool = merge(
          ind_tag, {'A': True, 'B': False, 'C': False}
      )
      int_sel = Stream([0, 2, 0, 5, 0, 0, 3, 1])

      odd = when(is_odd, nat)
      even = whennot(is_odd, nat * 10)
      nested_dyn = when(when(is_odd, c_dyn), odd)
      nested_stat = when(d_stat, odd)
      nested_neg = whennot(d_stat, odd)
      tagged_a = when(ind_tag, nat + 100, values='A')
      tagged_not_a = when(ind_tag, nat + 200, values='A', negated=True)
      empty_sub = when(ind_tag, nat, values=())
      ind_sub = when(ind_bool, nat + 300)
      int_sub = when(int_sel, nat)
      none_sub = when(
          is_odd, Stream([9, None, 9, 42, 9, None, 9, 77])
      )
      buf_sub = buffer(odd, is_odd, is_odd, size=4)
      rst = Stream([False, False, False, False, True, False, False, False])

      with reset(rst):
        rst_cur = current(True, odd, default=500)
        rst_cnt = counter(
            start=10,
            max_val=12,
            tick=Stream([False, True, False, True, True, False, True, True]),
        )

      with match(is_odd) as m:
        with case(True):
          inner_sub = when(d_stat, nat)
          inner_empty = when(ind_tag, nat, values=())
          m.cur = current(True, inner_sub, default=-5)
          m.cur_empty = current(True, inner_empty, default=-9)
          m.cnt = counter(start=0, tick=d_stat)
        with case(False):
          m.cur = -1
          m.cur_empty = -1
          m.cnt = -1

      match_rst = Stream([
          False, True, False, False, False, False, False, False
      ])
      with reset(match_rst):
        with match(is_odd) as m_rst:
          with case(True):
            m_rst.out = current(True, when(d_stat, nat), default=777)
          with case(False):
            m_rst.out = 0

      return {
          'odd_self': current(is_odd, odd, default=0),
          'odd_shared': current(True, odd, default=-1),
          'odd_disjoint': current(is_even, odd, default=99),
          'odd_false': current(False, odd, default=88),
          'odd_gated': current(c_dyn, odd, default=-2),
          'even_neg': current(True, even, default=-3),
          'nested_dyn': current(True, nested_dyn, default=-4),
          'nested_stat': current(True, nested_stat, default=-5),
          'nested_neg': current(True, nested_neg, default=-6),
          'tagged_a': current(True, tagged_a, default=-7),
          'tagged_not_a': current(True, tagged_not_a, default=-8),
          'empty_sub': current(True, empty_sub, default=-9),
          'ind_sub': current(True, ind_sub, default=-10),
          'int_sub': current(True, int_sub, default=-11),
          'none_sub': current(True, none_sub, default=123),
          'none_def': current(is_odd, odd, default=None),
          'buf_sub': current(True, buf_sub, default=-12),
          'rst_cur': rst_cur,
          'rst_cnt': rst_cnt,
          'match_cur': m.cur,
          'match_cur_empty': m.cur_empty,
          'match_cnt': m.cnt,
          'match_rst_cur': m_rst.out,
      }

    sync_res = Scheduler.run(prog, horizon=8, semantics=Semantics.SYNC)
    kahn_res = Scheduler.run(prog, horizon=8, semantics=Semantics.KAHN)
    for key in sync_res:
      with self.subTest(stream=key):
        self._assert_absent_bottom_elision(sync_res[key], kahn_res[key])
    self.assertEqual(kahn_res['odd_self'], [0, 1, 1, 3, 3, 5, 5, 7])
    self.assertEqual(kahn_res['odd_shared'], [-1, 1, 1, 3, 3, 5, 5, 7])
    self.assertEqual(kahn_res['odd_disjoint'], [99] * 8)
    self.assertEqual(kahn_res['odd_false'], [88] * 8)
    self.assertEqual(kahn_res['odd_gated'], [-2, -2, -2, 3, 3, 3, 3, 3])
    self.assertEqual(kahn_res['empty_sub'], [-9] * 8)
    self.assertEqual(
        kahn_res['none_sub'], [123, None, None, 42, 42, None, None, 77]
    )
    self.assertEqual(kahn_res['none_def'], [None, 1, 1, 3, 3, 5, 5, 7])
    self.assertEqual(kahn_res['rst_cur'], [500, 1, 1, 3, 500, 5, 5, 7])

    # Early-EOF subclocked `post` into `current` matches under elision.
    def eof_post_subclock():
      c = Stream([True, True, False, True])
      cnt = counter(10)
      with reset(when(c, False)):
        sub_p = post(when(c, cnt))
      b = Stream([True, False, True, False])
      return current(b, sub_p, default=-1)

    kahn_eof = Scheduler.run(
        eof_post_subclock, horizon=4, semantics=Semantics.KAHN
    )
    sync_eof = Scheduler.run(
        eof_post_subclock, horizon=4, semantics=Semantics.SYNC
    )
    self._assert_absent_bottom_elision(sync_eof, kahn_eof)


if __name__ == '__main__':
  absltest.main()
