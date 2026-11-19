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
"""Unit tests for Psyre match, case, default, and last."""

import enum
import logging
from typing import Any, NamedTuple

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.automaton import (
    state,
    until,
)
from psyre.builder import build_graph
from psyre.common import (
    Absent,
    Semantics,
    counter,
    logger,
    post,
    switch,
    when,
)
from psyre.match import (
    case,
    default,
    last,
    match,
)
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


class _Mode(enum.Enum):
  FAST = enum.auto()
  SLOW = enum.auto()


class _TriMode(enum.Enum):
  IDLE = enum.auto()
  RUN = enum.auto()
  STOP = enum.auto()


class _Pair(NamedTuple):
  """Product of Streams testing nested NamedTuple lowering in match()."""

  a: Any
  b: Any


class _NamedPair(NamedTuple):
  """NamedTuple of Streams testing static NamedTuple lowering in match()."""

  left: Any
  right: Any


class MatchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def _assert_psyre_error(
      self,
      main_fn,
      expected_exc_type: type[Exception],
      expected_regex: str,
      semantics: Semantics = Semantics.SYNC,
  ):
    with self.assertRaisesRegex(expected_exc_type, expected_regex):
      Scheduler.run(main_fn, horizon=1, semantics=semantics)

  def test_standalone_match_singletons_and_delayed_selector(self):
    def inner():
      sel = Stream([True, False, None, True, False])
      with match(sel) as sing_m:
        with case(True):
          sing_m.out = counter(start=1)
        with case(False):
          sing_m.out = counter(start=10)
        with case(None):
          sing_m.out = -1

      with match(lambda: delayed_sel) as del_m:
        with case(True) as s:
          del_m.out = s
        with case(False) as s:
          del_m.out = s

      delayed_sel = Stream([True, False, True, False, True])
      return sing_m.out, del_m.out

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        singleton_out, delayed_out = Scheduler.run(
            inner, horizon=5, semantics=sem
        )
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(singleton_out, [1, 10, -1, 2, 11])
        self.assertEqual(delayed_out, [True, False, True, False, True])

  def test_match_multi_tag_wildcard_and_named_outputs(self):
    def inner():
      cmd = Stream(['A', 'B', 'C', 'Z', 'A', 'Q'])

      with match(cmd) as m:
        with case('A', 'B') as c:
          m.out_0 = counter(start=1)
          m.out_1 = 10
          m.out_2 = c
        with case('C'):
          m.out_0 = 0
          m.out_1 = 20
          m.out_2 = 'seen_C'
        with default() as c:
          m.out_0 = counter(start=100)
          m.out_1 = -1
          m.out_2 = c

      with match(cmd) as empty_case_m:
        with case('A'):
          empty_case_m.tag = 'is_A'
        with case() as other:
          empty_case_m.tag = other

      return m, empty_case_m.tag

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        m_out, empty_case_tag = Scheduler.run(inner, horizon=6, semantics=sem)
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(
            m_out,
            {
                'out_0': [1, 2, 0, 100, 3, 101],
                'out_1': [10, 10, 20, -1, 10, -1],
                'out_2': ['A', 'B', 'seen_C', 'Z', 'A', 'Q'],
            },
        )
        self.assertEqual(empty_case_tag, ['is_A', 'B', 'C', 'Z', 'is_A', 'Q'])

  def test_match_structured_dataclass_namedtuple_and_tuple_outputs(self):
    """Checks static field-wise merging of structured stream bundles."""

    def dense_prog():
      sel = Stream([_Mode.FAST, _Mode.SLOW, _Mode.FAST, _Mode.SLOW])
      with match(sel) as structured:
        with case(_Mode.FAST):
          structured.pair = _Pair(Stream(1), Stream(2))
          structured.tup = (Stream(10), None)
          structured.named = _NamedPair(Stream(100), Stream(200))
        with case(_Mode.SLOW):
          structured.pair = _Pair(counter(start=5), Stream(8))
          structured.tup = (Stream(30), Stream(40))
          structured.named = _NamedPair(Stream(300), Stream(400))

      self.assertIsInstance(structured.pair, _Pair)
      self.assertIsInstance(structured.named, _NamedPair)
      return structured.pair, structured.tup, structured.named

    for sem in (Semantics.KAHN, Semantics.SYNC):
      with self.subTest(semantics=sem):
        pair, tup, named = Scheduler.run(dense_prog, horizon=4, semantics=sem)
        self.assertEqual(pair, _Pair([1, 5, 1, 6], [2, 8, 2, 8]))
        self.assertEqual(tup, ([10, 30, 10, 30], [None, 40, None, 40]))
        self.assertEqual(
            named, _NamedPair([100, 300, 100, 300], [200, 400] * 2)
        )

    def sparse_sync_prog():
      active = Stream([True, True, False, True])
      sel = Stream([_Mode.FAST, _Mode.SLOW, _Mode.FAST, _Mode.SLOW])
      with match(active):
        with case(True):
          with match(sel) as structured:
            with case(_Mode.FAST):
              structured.pair = _Pair(Stream(1), Stream(2))
              structured.tup = (Stream(10), Stream(20))
              structured.named = _NamedPair(Stream(100), Stream(200))
            with case(_Mode.SLOW):
              structured.pair = _Pair(counter(start=5), Stream(8))
              structured.tup = (Stream(30), Stream(40))
              structured.named = _NamedPair(Stream(300), Stream(400))
          self.assertIsInstance(structured.pair, _Pair)
          self.assertIsInstance(structured.named, _NamedPair)
        with case(False):
          pass

    res_s = Scheduler.run(
        sparse_sync_prog,
        horizon=4,
        semantics=Semantics.SYNC,
        streams=(
            'pair_a',
            'pair_b',
            'tup_0',
            'tup_1',
            'named_left',
            'named_right',
        ),
    )
    self.assertEqual(res_s['pair_a'], [1, 5, Absent, 6])
    self.assertEqual(res_s['pair_b'], [2, 8, Absent, 8])
    self.assertEqual(res_s['tup_0'], [10, 30, Absent, 30])
    self.assertEqual(res_s['tup_1'], [20, 40, Absent, 40])
    self.assertEqual(res_s['named_left'], [100, 300, Absent, 300])
    self.assertEqual(res_s['named_right'], [200, 400, Absent, 400])

  def test_sync_absent_selector_does_not_tick_wildcard(self):
    def inner():
      active = Stream([True, False, True, False])
      sub_cmd = Stream(['A', 'A', 'Z', 'Z'])

      with match(active):
        with case(True):
          with match(sub_cmd) as m:
            with case('A'):
              m.y = counter(start=1)
            with default():
              m.y = counter(start=100)
        with case(False):
          pass

    self.assertEqual(
        Scheduler.run(
            inner, horizon=4, semantics=Semantics.SYNC, streams=('y',)
        )['y'],
        [1, Absent, 100, Absent],
    )

  def test_last_in_absent_match_arm(self):
    def inner():
      active = Stream([True, True, False, True, False, True])
      sel = Stream(['INC', 'INC', 'INC', 'DEC', 'DEC', 'INC'])

      with match(active):
        with case(True):
          with match(sel) as m:
            prev = last(lambda: m.val, 0)
            with case('INC'):
              m.val = prev + 5
            with default():
              m.val = prev - 2
        with case(False):
          pass

    self.assertEqual(
        Scheduler.run(
            inner, horizon=6, semantics=Semantics.SYNC, streams=('val',)
        )['val'],
        [5, 10, Absent, 8, Absent, 13],
    )

  def test_match_validation_and_context_errors(self):
    def missing_enum():
      with match(Stream(_TriMode.IDLE)) as m:
        with case(_TriMode.IDLE):
          m.out = 0
        with case(_TriMode.RUN):
          m.out = 1

    self._assert_psyre_error(missing_enum, ValueError, 'Non-exhaustive match')

    def duplicate_case():
      with match(Stream('A')) as m:
        with case('A'):
          m.out = 1
        with case('A'):
          m.out = 2

    self._assert_psyre_error(duplicate_case, ValueError, 'Duplicate')

    def case_after_default():
      with match(Stream('A')) as m:
        with default():
          m.out = 0
        with case('A'):
          m.out = 1

    self._assert_psyre_error(case_after_default, ValueError, 'Unreachable')

    def state_in_match():
      with match(Stream('UP')) as m:
        with state('UP'):
          m.out = 1

    self._assert_psyre_error(
        state_in_match,
        RuntimeError,
        'state\\(\\) can only be used directly inside an automaton\\(\\) block',
    )

    def until_in_match():
      with match(Stream('UP')) as m:
        with case('UP'):
          until(Stream(True), 'DOWN')
          m.out = 1
        with default():
          m.out = 0

    self._assert_psyre_error(
        until_in_match,
        RuntimeError,
        'only be called directly inside a state\\(\\) block',
    )

    def inconsistent_outputs():
      with match(Stream(True)) as m:
        with case(True):
          m.a = 1
          m.b = 2
        with case(False):
          m.a = 3

    self._assert_psyre_error(
        inconsistent_outputs, ValueError, 'Inconsistent match\\(\\) outputs'
    )

    def assign_outside_arm():
      with match(Stream(True)) as m:
        m.out = 1

    self._assert_psyre_error(
        assign_outside_arm, RuntimeError, 'must be assigned inside a case\\(\\)'
    )

    def last_outside_scope():
      last(Stream([1, 2]), 0)

    self._assert_psyre_error(
        last_outside_scope,
        RuntimeError,
        'only be used inside match\\(\\) or automaton\\(\\)',
    )

  def test_match_arm_local_last_sampling_elision_and_confinement(self):
    def inline_match():
      sel = Stream([True, False, True, False])
      with match(sel) as m:
        with case(True):
          m.v = last(lambda: m.v, 10) + 2
        with case(False):
          m.v = last(lambda: m.v, 10) - 1
      return m.v

    def scoped_match():
      sel = Stream([True, False, True, False])
      with match(sel) as m:
        prev = last(lambda: m.v, 10)
        with case(True):
          m.v = prev + 2
        with case(False):
          m.v = prev - 1
      return m.v

    for sem in (Semantics.KAHN, Semantics.SYNC):
      with self.subTest(semantics=sem):
        inline_g = build_graph(inline_match, semantics=sem)
        scoped_g = build_graph(scoped_match, semantics=sem)
        self.assertEqual(
            sum(1 for n in inline_g.nodes if n.op == 'when'),
            sum(1 for n in scoped_g.nodes if n.op == 'when'),
        )
        self.assertEqual(
            Scheduler.run_graph(inline_g, horizon=4),
            Scheduler.run_graph(scoped_g, horizon=4),
        )

    def return_from_case():
      with match(Stream('A')):
        with default():
          return last(Stream([1, 2]), 0)

    for sem in (Semantics.KAHN, Semantics.SYNC):
      self._assert_psyre_error(
          return_from_case,
          RuntimeError,
          'Arm-local last\\(\\) cannot escape.*program output',
          semantics=sem,
      )

    def escape_match_output():
      x = Stream([10, 20])
      with match(Stream([True, False])) as m:
        with case(True):
          lv = last(x, 0)
          m.out = lv
        with case(False):
          m.out = 0
      return lv

    for sem in (Semantics.KAHN, Semantics.SYNC):
      self._assert_psyre_error(
          escape_match_output,
          RuntimeError,
          'Arm-local last\\(\\) cannot escape.*program output',
          semantics=sem,
      )

    def escape_stream_return():
      with match(Stream([True, False])):
        with case(True):
          x = counter(start=1)
        with case(False):
          pass
      return x

    def escape_stream_outer_consumer():
      with match(Stream([True, False])):
        with case(True):
          x = counter(start=1)
        with case(False):
          pass
      return x + 1

    def escape_stream_sibling_arm():
      with match(Stream([True, False])) as m:
        with case(True):
          x = counter(start=1)
          m.out = x
        with case(False):
          m.out = x + 1
      return m.out

    for sem in (Semantics.KAHN, Semantics.SYNC):
      self._assert_psyre_error(
          escape_stream_return,
          RuntimeError,
          'Arm-local stream cannot escape.*program output',
          semantics=sem,
      )
      for fn in (escape_stream_outer_consumer, escape_stream_sibling_arm):
        self._assert_psyre_error(
            fn,
            RuntimeError,
            'Arm-local stream cannot escape.*consumed by',
            semantics=sem,
        )

  def test_multi_valued_when_switch_and_reduced_match_lowering(self):
    def direct_prims():
      tags = Stream(['A', 'B', 'C', 'A'], name='tags')
      mixed = Stream([1, True, 0, False], name='mixed')
      vals = Stream([10, 20, 30, 40], name='vals')
      flag = Stream([True, False, True, False], name='flag')
      only_a = when(tags, vals, values='A')
      a_or_b = when(tags, vals, values=('A', 'B'))
      not_a = when(tags, vals, values='A', negated=True)
      only_int_one = when(mixed, vals, values=1)
      only_bool_true = when(mixed, vals, values=True)
      sw_bool = switch(flag, vals, vals * 10)
      sw_tag = switch(tags, {'A': vals, 'B': vals + 1}, default=vals + 2)
      return (
          only_a,
          a_or_b,
          not_a,
          only_int_one,
          only_bool_true,
          sw_bool,
          sw_tag,
      )

    sync_res = Scheduler.run(direct_prims, horizon=4, semantics=Semantics.SYNC)
    self.assertEqual(
        sync_res,
        (
            [10, Absent, Absent, 40],
            [10, 20, Absent, 40],
            [Absent, 20, 30, Absent],
            [10, Absent, Absent, Absent],
            [Absent, 20, Absent, Absent],
            [10, 200, 30, 400],
            [10, 21, 32, 40],
        ),
    )
    kahn_res = Scheduler.run(direct_prims, horizon=4, semantics=Semantics.KAHN)
    self.assertEqual(
        kahn_res,
        (
            [10, 40],
            [10, 20, 40],
            [20, 30],
            [10],
            [20],
            [10, 200, 30, 400],
            [10, 21, 32, 40],
        ),
    )

    # Negated multi-valued when must emit Absent when selector is Absent:
    def absent_negated_when():
      act = Stream([True, False, True, False])
      sel = Stream(['A', 'B', 'C', 'B'])
      x = Stream([1, 2, 3, 4])
      return when(when(act, sel), when(act, x), values='A', negated=True)

    self.assertEqual(
        Scheduler.run(absent_negated_when, horizon=4, semantics=Semantics.SYNC),
        [Absent, Absent, 3, Absent],
    )

    # Boolean match lowers cleanly to 5 nodes (when, whennot, +, -, bool merge):
    def bool_match_prog():
      c = Stream([True, False, True, False], name='c')
      x = Stream([1, 2, 3, 4], name='x')
      with match(c) as m:
        with case(True):
          m.y = x + 1
        with case(False):
          m.y = x - 1
      return m.y

    g_bool = build_graph(bool_match_prog, semantics=Semantics.SYNC)
    self.assertLen(g_bool.nodes, 5)
    merge_nodes = [n for n in g_bool.nodes if n.op == 'merge']
    self.assertLen(merge_nodes, 1)
    self.assertTrue(merge_nodes[0].kwargs['lookup'].is_boolean)
    self.assertEqual(Scheduler.run_graph(g_bool, horizon=4), [2, 1, 4, 3])

    # Multi-way match emits zero indicator merges; post(const) stays Absent
    # on inactive tick 0:
    def multi_match_post_const():
      tag = Stream(['B', 'A', 'A', 'B'], name='tag')
      with match(tag) as m:
        with case('A'):
          m.y = post(7)
        with case('B'):
          m.y = 0
      return m.y

    g_multi = build_graph(
        multi_match_post_const, semantics=Semantics.SYNC
    )
    self.assertLen([n for n in g_multi.nodes if n.op == 'merge'], 1)
    self.assertEqual(Scheduler.run_graph(g_multi, horizon=4), [0, 7, 7, 0])


if __name__ == '__main__':
  absltest.main()
