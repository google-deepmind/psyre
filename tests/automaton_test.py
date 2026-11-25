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
"""Unit tests for Psyre automaton, state, until, and unless."""

import enum
import logging
from typing import Any

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.automaton import (
    automaton,
    state,
    unless,
    until,
)
from psyre.builder import build_graph
from psyre.clock import check_clocks
from psyre.common import (
    Absent,
    Bottom,
    Semantics,
    counter,
    fby,
    ifthenelse,
    logger,
    pre,
    psyre_node,
    reset,
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
from tests import test_utils

# pylint: enable=g-importing-member,g-multiple-import

_Mode = test_utils.Mode
_TriMode = test_utils.TriMode


class AutomatonTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def test_automaton_until_continue_vs_reset(self):
    def up_down(toggle: Stream, reset_on_flip: Stream) -> Stream:
      with automaton() as fsm:
        with state('UP'):
          until(toggle, 'DOWN', reset=reset_on_flip)
          fsm.y = counter(start=0)
        with state('DOWN'):
          until(toggle, 'UP', reset=reset_on_flip)
          fsm.y = counter(start=100)

      return fsm.y

    def inner():
      toggle = Stream([False, True, False, True, False, False])
      return up_down(toggle, Stream(False)), up_down(toggle, Stream(True))

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        cont_out, rst_out = Scheduler.run(inner, horizon=6, semantics=sem)
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(cont_out, [0, 1, 100, 101, 2, 3])
        self.assertEqual(rst_out, [0, 1, 100, 101, 0, 1])

  def test_automaton_rejects_absent_guards(self):
    bad_cases = [
        (Absent, True, 'cannot contain Absent'),
        (True, Absent, 'cannot contain Absent'),
        ([Absent, True], True, 'cannot contain Absent'),
        (Bottom, True, 'cannot contain Bottom'),
        (True, Bottom, 'cannot contain Bottom'),
        ([Bottom, True], True, 'cannot contain Bottom'),
    ]
    for bad_cond, bad_reset, regex in bad_cases:

      def bad_until(cond=bad_cond, rst=bad_reset):
        with automaton() as fsm:
          with state('UP'):
            until(cond, 'DOWN', reset=rst)
            fsm.y = 0
          with state('DOWN'):
            fsm.y = 1
        return fsm.y

      def bad_unless(cond=bad_cond, rst=bad_reset):
        with automaton() as fsm:
          with state('RUN', unless(cond, 'IDLE', reset=rst)):
            fsm.y = 0
          with state('IDLE'):
            fsm.y = 1
        return fsm.y

      with self.assertRaisesRegex(TypeError, regex):
        Scheduler.run(bad_until, horizon=2, semantics=Semantics.SYNC)
      with self.assertRaisesRegex(TypeError, regex):
        Scheduler.run(bad_unless, horizon=2, semantics=Semantics.SYNC)

  def test_automaton_unless_and_until_transitions(self):
    def controller(done: Stream, abort: Stream, wake: Stream) -> Stream:
      with automaton() as fsm:
        with state('RUN', unless(abort, 'IDLE')):
          c = counter(start=1)
          until(done, 'IDLE')
          fsm.y = c
        with state('IDLE', unless(wake, 'RUN')):
          fsm.y = counter(start=100)

      return fsm.y

    def strong_history(done: Stream, wake: Stream) -> Stream:
      with automaton() as fsm:
        with state('RUN'):
          c = counter(start=1)
          until(done, 'IDLE')
          fsm.y = c
        with state('IDLE', unless(wake, 'RUN', reset=False)):
          fsm.y = counter(start=100)

      return fsm.y

    def inner():
      done = Stream([False, True, False, False, False, False])
      abort = Stream([False, False, False, False, True, False])
      wake = Stream([False, False, False, True, False, False])
      return controller(done, abort, wake), strong_history(done, wake)

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        resetting, history = Scheduler.run(inner, horizon=6, semantics=sem)
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(resetting, [1, 2, 100, 1, 100, 101])
        self.assertEqual(history, [1, 2, 100, 3, 4, 5])

  def test_automaton_self_transition_and_init_selection(self):
    def strong_restart(restart: Stream) -> Stream:
      with automaton() as fsm:
        with state('RUN', unless(restart, 'RUN')):
          fsm.y = counter(start=0)

      return fsm.y

    def weak_restart(restart: Stream) -> Stream:
      with automaton() as fsm:
        with state('RUN'):
          until(restart, 'RUN')
          fsm.y = counter(start=0)

      return fsm.y

    def start_mode(flip: Stream, init_mode: _Mode | None = None) -> Stream:
      auto_cm = automaton() if init_mode is None else automaton(init=init_mode)
      with auto_cm as fsm:
        with state(_Mode.SLOW):
          until(flip, _Mode.FAST)
          fsm.y = counter(start=10)
        with state(_Mode.FAST):
          until(flip, _Mode.SLOW)
          fsm.y = counter(start=1)

      return fsm.y

    def inner():
      restart = Stream([False, False, True, False])
      flip = Stream([False, True, False, False])
      return (
          strong_restart(restart),
          weak_restart(restart),
          start_mode(flip),
          start_mode(flip, init_mode=_Mode.FAST),
      )

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        s_rst, w_rst, slow_def, fast_exp = Scheduler.run(
            inner, horizon=4, semantics=sem
        )
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(s_rst, [0, 1, 0, 1])
        self.assertEqual(w_rst, [0, 1, 2, 0])
        self.assertEqual(slow_def, [10, 11, 1, 2])
        self.assertEqual(fast_exp, [1, 2, 10, 11])

  def test_automaton_priority_and_body_local_guard(self):
    def prio_auto(
        hi_unless: Stream,
        lo_unless: Stream,
        hi_until: Stream,
        lo_until: Stream,
    ) -> Stream:
      with automaton() as fsm:
        with state(
            _TriMode.IDLE,
            unless(hi_unless, _TriMode.STOP),
            unless(lo_unless, _TriMode.RUN),
        ):
          until(hi_until, _TriMode.RUN)
          until(lo_until, _TriMode.STOP)
          fsm.y = 0
        with state(_TriMode.RUN):
          c = counter(start=1)
          until(c >= 2, _TriMode.IDLE)
          fsm.y = c
        with state(_TriMode.STOP):
          until(True, _TriMode.IDLE)
          fsm.y = -1

      return fsm.y

    def inner():
      hi_unless = Stream([False, False, False, False, True, False])
      lo_unless = Stream([False, False, False, False, True, False])
      hi_until = Stream([True, False, False, False, False, False])
      lo_until = Stream([True, False, False, False, False, False])
      return prio_auto(hi_unless, lo_unless, hi_until, lo_until)

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        self.assertEqual(
            Scheduler.run(inner, horizon=6, semantics=sem),
            [0, 1, 2, 0, -1, 0],
        )
        self.assertFalse(Scheduler.deadlocked)

  def test_automaton_shared_fby_variable(self):
    seq = [False, True, False, False, True]

    def make_prog(dynamic: bool):
      def prog():
        # `dynamic=False`: clock-polymorphic static `toggle` advances on each
        # state's own ticks (`UP` gets [F, T, F], `DOWN` gets [F, T]).
        toggle = when(Stream(True), seq) if dynamic else Stream(seq)
        with automaton() as fsm:
          v = fby(10, lambda: fsm.v_next)
          with state('UP'):
            until(toggle, 'DOWN')
            fsm.v_next = v + 2
          with state('DOWN'):
            until(toggle, 'UP')
            fsm.v_next = v - 1
        return fsm.v_next

      return prog

    for sem in (Semantics.SYNC, Semantics.KAHN):
      for dynamic, expected in (
          (True, [12, 14, 13, 12, 11]),
          (False, [12, 14, 13, 12, 14]),
      ):
        with self.subTest(semantics=sem, dynamic=dynamic):
          self.assertEqual(
              Scheduler.run(make_prog(dynamic), horizon=5, semantics=sem),
              expected,
          )
          self.assertFalse(Scheduler.deadlocked)

  def test_last_defined_vs_fby_pre_observed(self):
    def compare_external(
        x: Stream, flip: Stream, reset_flag: Stream
    ) -> tuple[Stream, Stream, Stream, Stream]:
      with automaton() as fsm:
        with state('A'):
          until(flip, 'B', reset=reset_flag)
          fsm.p_out = pre(x)
          fsm.f_out = fby(0, x)
          fsm.ln_out = last(x)
          fsm.lz_out = last(x, 0)
        with state('B'):
          until(flip, 'A', reset=reset_flag)
          fsm.p_out = -1
          fsm.f_out = -1
          fsm.ln_out = -1
          fsm.lz_out = -1

      return fsm.p_out, fsm.f_out, fsm.ln_out, fsm.lz_out

    def inner():
      x = Stream([10, 20, 30, 40, 50])
      flip = Stream([False, True, False, True, False])
      pre_c, fby_c, last_none_c, last_zero_c = compare_external(
          x, flip, Stream(False)
      )
      _, fby_r, _, last_zero_r = compare_external(x, flip, Stream(True))

      return {
          'pre_continue': pre_c,
          'fby_continue': fby_c,
          'last_none_continue': last_none_c,
          'last_zero_continue': last_zero_c,
          'fby_reset': fby_r,
          'last_zero_reset': last_zero_r,
      }

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        res = Scheduler.run(inner, horizon=5, semantics=sem)
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(res['pre_continue'], [None, 10, -1, -1, 20])
        self.assertEqual(res['fby_continue'], [0, 10, -1, -1, 20])
        self.assertEqual(res['last_none_continue'], [None, 10, -1, -1, 40])
        self.assertEqual(res['last_zero_continue'], [0, 10, -1, -1, 40])
        self.assertEqual(res['fby_reset'], [0, 10, -1, -1, 0])
        self.assertEqual(res['last_zero_reset'], [0, 10, -1, -1, 40])

  def test_automaton_last_shared_variable_and_strong_guard(self):
    def inner():
      toggle = when(Stream(True), [False, True, False, False, True, False])
      inc = Stream([False, True, True, True, False, False])
      dec = Stream([False, False, False, False, True, False])

      with automaton() as inline_fsm:
        with state('UP'):
          until(toggle, 'DOWN')
          inline_fsm.v = last(lambda: inline_fsm.v, 10) + 2
        with state('DOWN'):
          until(toggle, 'UP')
          inline_fsm.v = last(lambda: inline_fsm.v, 10) - 1

      with automaton() as scoped_fsm:
        prev_v = last(lambda: scoped_fsm.v, init=10)
        with state('UP'):
          until(toggle, 'DOWN')
          scoped_fsm.v = prev_v + 2
        with state('DOWN'):
          until(toggle, 'UP')
          scoped_fsm.v = prev_v - 1

      with automaton() as inline_o:
        with state('IDLE', unless(inc, 'INCR'), unless(dec, 'DECR')):
          inline_o.o = last(lambda: inline_o.o, 0)
        with state(
            'INCR', unless(~inc | (last(lambda: inline_o.o, 0) >= 2), 'IDLE')
        ):
          inline_o.o = last(lambda: inline_o.o, 0) + 1
        with state('DECR', unless(~dec, 'IDLE')):
          inline_o.o = last(lambda: inline_o.o, 0) - 1

      with automaton() as preamble_o:
        prev_o = last(lambda: preamble_o.o, init=0)
        with state('IDLE', unless(inc, 'INCR'), unless(dec, 'DECR')):
          preamble_o.o = prev_o
        with state('INCR', unless(~inc | (prev_o >= 2), 'IDLE')):
          preamble_o.o = prev_o + 1
        with state('DECR', unless(~dec, 'IDLE')):
          preamble_o.o = prev_o - 1

      return inline_fsm.v, scoped_fsm.v, inline_o.o, preamble_o.o

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        inline_acc, scoped_acc, bounded_adj, preamble_adj = Scheduler.run(
            inner, horizon=6, semantics=sem
        )
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(inline_acc[:5], [12, 14, 13, 12, 11])
        self.assertEqual(scoped_acc[:5], [12, 14, 13, 12, 11])
        self.assertEqual(bounded_adj, [0, 1, 2, 2, 1, 1])
        self.assertEqual(preamble_adj, [0, 1, 2, 2, 1, 1])

  def test_last_hierarchical_automaton(self):
    def inner_accum(sub_flip: Stream) -> Stream:
      with automaton() as fsm:
        with state('ADD'):
          until(sub_flip, 'SUB')
          fsm.z = last(lambda: fsm.z, 10) + 2
        with state('SUB'):
          until(sub_flip, 'ADD')
          fsm.z = last(lambda: fsm.z, 10) - 1

      return fsm.z

    def outer_ctrl(
        outer_flip: Stream,
        reset_on_flip: Stream,
        sub_flip: Stream,
    ) -> Stream:
      with automaton() as fsm:
        with state('ON'):
          until(outer_flip, 'OFF', reset=reset_on_flip)
          fsm.y = inner_accum(sub_flip)
        with state('OFF'):
          until(outer_flip, 'ON', reset=reset_on_flip)
          fsm.y = -1

      return fsm.y

    def inner():
      sub_flip = when(Stream(True), [True, False, False, False, True, False])
      outer_flip = when(Stream(True), [False, True, True, False, False, False])
      ext_reset = Stream([False, False, False, True, False, False])

      with reset(ext_reset):
        ext_rst = inner_accum(sub_flip)
      return (
          outer_ctrl(outer_flip, Stream(False), sub_flip),
          outer_ctrl(outer_flip, Stream(True), sub_flip),
          ext_rst,
      )

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        deep_hist, hier_rst, ext_rst = Scheduler.run(
            inner, horizon=6, semantics=sem
        )
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(deep_hist, [12, 11, -1, 10, 9, 11])
        self.assertEqual(hier_rst, [12, 11, -1, 12, 14, 13])
        self.assertEqual(ext_rst, [12, 11, 10, 12, 14, 13])

  def test_hierarchical_automaton_and_modular_reset(self):
    def inner_auto(sub_toggle: Stream) -> Stream:
      with automaton() as fsm:
        with state('X'):
          until(sub_toggle, 'Y', reset=False)
          fsm.y = counter(start=0)
        with state('Y'):
          until(sub_toggle, 'X', reset=False)
          fsm.y = counter(start=10)

      return fsm.y

    def outer_auto(
        outer_toggle: Stream,
        reset_on_flip: Stream,
        sub_toggle: Stream,
    ) -> Stream:
      with automaton() as fsm:
        with state('A'):
          until(outer_toggle, 'B', reset=reset_on_flip)
          fsm.y = inner_auto(sub_toggle)
        with state('B'):
          until(outer_toggle, 'A', reset=reset_on_flip)
          fsm.y = -1

      return fsm.y

    def inline_hier(outer_flip: Stream, inner_preempt: Stream) -> Stream:
      with automaton() as outer:
        with state('ON'):
          with automaton() as inner_fsm:
            with state('S0', unless(inner_preempt, 'S1')):
              inner_fsm.z = counter(start=1)
            with state('S1'):
              inner_fsm.z = counter(start=50)

          until(outer_flip, 'OFF')
          outer.y = inner_fsm.z
        with state('OFF'):
          until(outer_flip, 'ON')
          outer.y = 0

      return outer.y

    def inner():
      sub_toggle = when(
          Stream(True), [True, False, False, False, False, True, False]
      )
      outer_toggle = when(
          Stream(True), [False, False, True, True, False, False, False]
      )
      ext_reset = Stream([False, False, False, True, False, False, False])

      with reset(ext_reset):
        ext_rst = inner_auto(sub_toggle)
      return (
          outer_auto(outer_toggle, Stream(False), sub_toggle),
          outer_auto(outer_toggle, Stream(True), sub_toggle),
          ext_rst,
          inline_hier(outer_toggle, sub_toggle),
      )

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        deep_hist, hier_rst, ext_rst, inl_hier = Scheduler.run(
            inner, horizon=7, semantics=sem
        )
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(deep_hist, [0, 10, 11, -1, 12, 13, 1])
        self.assertEqual(hier_rst, [0, 10, 11, -1, 0, 1, 10])
        self.assertEqual(ext_rst, [0, 10, 11, 0, 1, 2, 10])
        self.assertEqual(inl_hier, [50, 51, 52, 0, 1, 50, 51])

  def _assert_psyre_error(
      self,
      main_fn,
      expected_exc_type: type[Exception],
      expected_regex: str,
      semantics: Semantics = Semantics.SYNC,
  ):
    test_utils.assert_psyre_error(
        self, main_fn, expected_exc_type, expected_regex, semantics=semantics
    )

  def test_automaton_validation_and_context_errors(self):
    def missing_bool():
      with automaton() as fsm:
        with state(True):
          fsm.out = 1

    self._assert_psyre_error(
        missing_bool, ValueError, 'Non-exhaustive automaton'
    )

    def duplicate_state():
      with automaton() as fsm:
        with state('UP'):
          fsm.out = 1
        with state('UP'):
          fsm.out = 2

    self._assert_psyre_error(duplicate_state, ValueError, 'Duplicate state')

    def wildcard_in_automaton():
      with automaton() as fsm:
        with state('UP'):
          fsm.out = 1
        with default():
          fsm.out = 0

    self._assert_psyre_error(
        wildcard_in_automaton,
        RuntimeError,
        'default\\(\\) can only be used directly inside a match\\(\\) block',
    )

    def case_in_automaton():
      with automaton() as fsm:
        with case('UP'):
          fsm.out = 1

    self._assert_psyre_error(
        case_in_automaton,
        RuntimeError,
        'case\\(\\) can only be used directly inside a match\\(\\) block',
    )

    def bad_init():
      with automaton(init='UNKNOWN') as fsm:
        with state('UP'):
          fsm.out = 1

    self._assert_psyre_error(bad_init, ValueError, 'Initial state')

    def unless_inside_state_body():
      with automaton() as fsm:
        with state('RUN'):
          unless(Stream(True), 'IDLE')
          fsm.out = 1
        with state('IDLE'):
          fsm.out = 0

    self._assert_psyre_error(
        unless_inside_state_body,
        RuntimeError,
        'cannot be called inside a state\\(\\) body',
    )

    def unconsumed_unless():
      with automaton() as fsm:
        unless(Stream(True), 'IDLE')
        with state('RUN'):
          fsm.out = 1
        with state('IDLE'):
          fsm.out = 0

    self._assert_psyre_error(
        unconsumed_unless,
        RuntimeError,
        'must be passed directly to state',
    )

    def eager_forward_read():
      with automaton() as fsm:
        _ = fsm.out
        with state('UP'):
          fsm.out = 1

    self._assert_psyre_error(
        eager_forward_read, AttributeError, 'wrap forward references in lambda'
    )

  def test_automaton_arm_local_last_sampling_elision_and_confinement(self):
    def inline_auto():
      flip = Stream([False, True, False, True])
      with automaton() as fsm:
        with state('A'):
          until(flip, 'B')
          fsm.v = last(lambda: fsm.v, 10) + 2
        with state('B'):
          until(flip, 'A')
          fsm.v = last(lambda: fsm.v, 10) - 1
      return fsm.v

    def scoped_auto():
      flip = Stream([False, True, False, True])
      with automaton() as fsm:
        prev = last(lambda: fsm.v, 10)
        with state('A'):
          until(flip, 'B')
          fsm.v = prev + 2
        with state('B'):
          until(flip, 'A')
          fsm.v = prev - 1
      return fsm.v

    inline_auto_g = build_graph(inline_auto, semantics=Semantics.SYNC)
    scoped_auto_g = build_graph(scoped_auto, semantics=Semantics.SYNC)
    self.assertEqual(
        sum(1 for n in inline_auto_g.nodes if n.op == 'when'),
        sum(1 for n in scoped_auto_g.nodes if n.op == 'when'),
    )

    def nested_arm_last():
      x = Stream([10, 20, 30, 40])
      flag = Stream([True, False, True, True])
      with automaton() as fsm:
        with state('A'):
          lv = last(x, 0)
          until(lv >= 20, 'B')
          with match(flag) as inner_m:
            with case(True):
              inner_m.y = lv + 1
            with case(False):
              inner_m.y = lv - 1
          fsm.out = inner_m.y
        with state('B'):
          fsm.out = -1
      return fsm.out

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        self.assertEqual(
            Scheduler.run(nested_arm_last, horizon=4, semantics=sem),
            [1, 9, 21, -1],
        )
        self.assertFalse(Scheduler.deadlocked)

    def escape_state_return_after_assign():
      x = Stream([10, 20])
      with automaton() as fsm:
        with state('A'):
          lv = last(x, 0)
          fsm.out = 1
          return lv

    self._assert_psyre_error(
        escape_state_return_after_assign,
        RuntimeError,
        'Arm-local last\\(\\) cannot escape.*program output',
    )

    def escape_outer_expr():
      x = Stream([10, 20])
      with automaton() as fsm:
        with state('A'):
          until(Stream(True), 'B')
          lv = last(x, 0)
          fsm.out = lv
        with state('B'):
          fsm.out = 0
      return lv + 1

    self._assert_psyre_error(
        escape_outer_expr,
        RuntimeError,
        'Arm-local last\\(\\) cannot escape.*consumed by',
    )

    def escape_sibling_state():
      x = Stream([10, 20])
      with automaton() as fsm:
        with state('A'):
          until(Stream(True), 'B')
          lv = last(x, 0)
          fsm.out = lv
        with state('B'):
          fsm.out = lv + 1
      return fsm.out

    self._assert_psyre_error(
        escape_sibling_state,
        RuntimeError,
        'Arm-local last\\(\\) cannot escape.*consumed by',
    )

    def escape_chained_arm_last():
      x = Stream([10, 20])
      with automaton() as fsm:
        with state('A'):
          lv = last(x, 0)
          fsm.out = last(lv, 0)
      return fsm.out

    self._assert_psyre_error(
        escape_chained_arm_last,
        RuntimeError,
        'Arm-local last\\(\\) cannot escape.*consumed by',
    )

    def escape_state_stream_return():
      with automaton():
        with state('A'):
          x = counter(start=1)
      return x

    def escape_state_stream_sibling():
      with automaton() as fsm:
        with state('A'):
          until(Stream(True), 'B')
          x = counter(start=1)
          fsm.out = x
        with state('B'):
          fsm.out = x + 1
      return fsm.out

    self._assert_psyre_error(
        escape_state_stream_return,
        RuntimeError,
        'Arm-local stream cannot escape.*program output',
    )
    self._assert_psyre_error(
        escape_state_stream_sibling,
        RuntimeError,
        'Arm-local stream cannot escape.*consumed by',
    )

  def test_automaton_name_prefixing_and_inheritance(self):
    def default_and_named():
      x = Stream([1, 2, 3], name='x')
      with automaton() as a0:
        with state('A', unless(x >= 3, 'B')):
          until(x >= 2, 'B')
          a0.v = fby(0, lambda: a0.v + 1, name='cnt')
        with state('B'):
          a0.v = 10
      with automaton(init='IDLE', name='ctrl') as a1:
        with state('IDLE'):
          until(a0.v >= 10, 'ACTIVE')
          a1.out = a0.v
        with state('ACTIVE'):
          with automaton(name='sub') as sub_fsm:
            with state('S0'):
              sub_fsm.y = fby(1, lambda: sub_fsm.y + 1, name='inner_cnt')
          a1.out = sub_fsm.y
      return a1.out

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        g = build_graph(default_and_named, semantics=sem)
        node_names = {n.name for n in g.nodes}
        # Default fsm_0 prefix on the first unnamed automaton:
        self.assertIn('fsm_0/state_s', node_names)
        self.assertIn('fsm_0/pns', node_names)
        self.assertIn('fsm_0/cnt', node_names)
        # Explicit 'ctrl' prefix on the second automaton:
        self.assertIn('ctrl/state_ns', node_names)
        self.assertIn('ctrl/pns', node_names)
        self.assertNotIn('ctrl/state', node_names)
        # Inherited 'ctrl/sub' prefix on the nested automaton:
        self.assertNotIn('ctrl/sub/state_ns', node_names)
        self.assertIn('ctrl/sub/pns', node_names)
        self.assertIn('ctrl/sub/inner_cnt', node_names)
        self.assertNotIn('ctrl/sub/state', node_names)
        self.assertEqual(
            Scheduler.run_graph(g, horizon=4, clocks=check_clocks(g)),
            [0, 1, 10, 1],
        )
        self.assertFalse(Scheduler.deadlocked)

    g_unopt = build_graph(
        default_and_named, semantics=Semantics.SYNC, optimize=False
    )
    unopt_names = {n.name for n in g_unopt.nodes}
    self.assertIn('ctrl/state', unopt_names)
    self.assertIn('ctrl/sub/state', unopt_names)

    def invalid_name():
      bad_name: Any = 123
      with automaton(name=bad_name):
        with state('A'):
          pass

    self._assert_psyre_error(
        invalid_name, TypeError, 'automaton\\(name=\\.\\.\\.\\) must be a str'
    )

  def test_simplified_automaton_lowering_node_counts_and_switch(self):
    def cruise_control_prog():
      on = Stream([False, True, False, False, False], name='on')
      off = Stream([False, False, False, False, True], name='off')
      brake = Stream([False, False, True, False, False], name='brake')
      resume = Stream([False, False, False, True, False], name='resume')
      with automaton(init='OFF', name='cc') as fsm:
        with state('OFF', unless(on, 'ON')):
          fsm.active = False
        with state('ON', unless(off, 'OFF'), unless(brake, 'STANDBY')):
          fsm.active = True
        with state('STANDBY', unless(off, 'OFF'), unless(resume, 'ON')):
          fsm.active = False
      return fsm.active

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(cruise_control_semantics=sem):
        g_cc = build_graph(cruise_control_prog, semantics=sem)
        cc_ops = [n.op for n in g_cc.nodes]
        self.assertLen(g_cc.nodes, 8)
        self.assertEqual(cc_ops.count('switch'), 1)
        self.assertEqual(cc_ops.count('when'), 0)
        self.assertEqual(cc_ops.count('fby'), 1)
        self.assertEqual(
            Scheduler.run_graph(g_cc, horizon=5),
            [False, True, False, True, False],
        )
        self.assertFalse(Scheduler.deadlocked)

    def abro_prog():
      a = Stream([False, True, False, False, True, False], name='a')
      b = Stream([False, False, True, False, True, False], name='b')
      r = Stream([False, False, False, True, False, False], name='r')
      with automaton(init='AWAIT_AB', name='abro') as fsm:
        with state('AWAIT_AB', unless(r, 'AWAIT_AB')):
          seen_a = fby(False, lambda: seen_a | a, name='seen_a')
          seen_b = fby(False, lambda: seen_b | b, name='seen_b')
          done = (seen_a | a) & (seen_b | b)
          until(done, 'DONE', reset=False)
          fsm.o = done
        with state('DONE', unless(r, 'AWAIT_AB')):
          fsm.o = False
      return fsm.o

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(abro_semantics=sem):
        g_abro_raw = build_graph(abro_prog, semantics=sem, optimize=False)
        g_abro = build_graph(abro_prog, semantics=sem)
        # Uniform static clock polymorphism avoids `when` wrappers on `a`, `b`;
        # static folding collapses `abro/state_r` -> `r` and CSE shares
        # `seen_a | a` and `seen_b | b` (12 nodes total vs 15 unoptimized).
        self.assertLen(g_abro.nodes, 12)
        self.assertNotIn('abro/pnr', {n.name for n in g_abro.nodes})
        self.assertNotIn('abro/state_r', {n.name for n in g_abro.nodes})
        self.assertNotIn('abro/state_nr', {n.name for n in g_abro.nodes})
        expected = [False, False, True, False, True, False]
        self.assertEqual(
            Scheduler.run_graph(
                g_abro_raw, horizon=6, clocks=check_clocks(g_abro_raw)
            ),
            expected,
        )
        self.assertEqual(
            Scheduler.run_graph(g_abro, horizon=6, clocks=check_clocks(g_abro)),
            expected,
        )
        self.assertFalse(Scheduler.deadlocked)

    # Unreachable transition targets after static True must still be validated:
    def invalid_after_true():
      with automaton() as fsm:
        with state('A'):
          until(True, 'B')
          until(False, 'NO_SUCH_STATE')
          fsm.y = 0
        with state('B'):
          fsm.y = 1
      return fsm.y

    self._assert_psyre_error(
        invalid_after_true,
        ValueError,
        'Transition target.*does not match any state',
    )

  def test_same_arm_handle_read_after_assign_requires_lambda(self):
    def eager_const_last():
      with automaton() as fsm:
        with state('A'):
          fsm.v = 10
          until(last(fsm.v, 0) >= 10, 'B')
        with state('B'):
          fsm.v = 20
      return fsm.v

    self._assert_psyre_error(
        eager_const_last, AttributeError, 'wrap forward references in lambda'
    )

    def deferred_const_last():
      with automaton() as fsm:
        with state('A'):
          fsm.v = 10
          until(last(lambda: fsm.v, 0) >= 10, 'B')
        with state('B'):
          fsm.v = 20
          until(last(lambda: fsm.v, 0) >= 20, 'A')
      return fsm.v

    self.assertEqual(
        Scheduler.run(deferred_const_last, horizon=6, semantics=Semantics.SYNC),
        [10, 10, 20, 20, 10, 20],
    )

  def test_automaton_deferred_thunks_preserve_reset_scopes(self):
    def deferred_state_output():
      t = counter(0)
      with automaton() as a:
        with state('A'):
          a.o = lambda: counter(10)
          until(t == 1, 'B', reset=True)
        with state('B'):
          a.o = 0
          until(t == 2, 'A', reset=True)
      return a.o

    def deferred_until_cond():
      with automaton() as a:
        with state('A'):
          a.o = counter(10)
          until(lambda: counter(0) == 1, 'B', reset=True)
        with state('B'):
          a.o = 0
          until(True, 'A', reset=True)
      return a.o

    def deferred_unless_under_outer_reset():
      with reset(Stream.once(2)):
        with automaton() as a:
          with state('A', unless(lambda: counter(0) == 2, 'B')):
            a.o = 1
          with state('B'):
            a.o = 2
      return a.o

    cases = (
        (deferred_state_output, 6, [10, 11, 0, 10, 11, 12]),
        (deferred_until_cond, 7, [10, 11, 0, 10, 11, 0, 10]),
        (deferred_unless_under_outer_reset, 6, [1, 1, 1, 1, 2, 2]),
    )
    for sem in (Semantics.SYNC, Semantics.KAHN):
      for opt in (False, True):
        for prog, horizon, expected in cases:
          with self.subTest(prog=prog.__name__, semantics=sem, optimize=opt):
            self.assertEqual(
                Scheduler.run(
                    prog, horizon=horizon, semantics=sem, optimize=opt
                ),
                expected,
            )

  def test_automaton_rejects_double_thunks_at_stage_one(self):
    for label, is_strong, cond_spec, target_spec, reset_spec in (
        ('until_cond', False, lambda: (lambda: counter(0) == 1), 'B', None),
        ('until_target', False, True, lambda: (lambda: Stream('B')), None),
        ('until_reset', False, True, 'B', lambda: (lambda: Stream(True))),
        ('unless_cond', True, lambda: (lambda: counter(0) == 1), 'B', None),
        ('unless_target', True, True, lambda: (lambda: Stream('B')), None),
        ('unless_reset', True, counter, 'B', lambda: (lambda: Stream(True))),
    ):
      def make_bad_prog(strong, cond_v, target_v, reset_v):
        def bad_prog():
          cond_arg = (counter(0) == 1) if cond_v is counter else cond_v
          kw = {} if reset_v is None else {'reset': reset_v}
          with automaton() as a:
            if strong:
              with state('A', unless(cond_arg, target_v, **kw)):
                a.o = 1
            else:
              with state('A'):
                a.o = 1
                until(cond_arg, target_v, **kw)
            with state('B'):
              a.o = 2
          return a.o

        return bad_prog

      with self.subTest(case=label):
        with self.assertRaisesRegex(TypeError, 'Nested Stage-1 thunks'):
          build_graph(
              make_bad_prog(is_strong, cond_spec, target_spec, reset_spec)
          )

  def test_unconditional_transitions_preserve_inner_reset_and_prefix(self):
    def sole_uncond_until_target_and_reset():
      r_inner = Stream([False, True, False, False, False, False])

      @psyre_node('weak_sub')
      def emit_until():
        with reset(r_inner):
          until(
              True,
              lambda: ifthenelse(counter(0, name='wt') == 1, 'B', 'A'),
              reset=lambda: counter(0, name='wr') == 1,
          )

      with automaton(name='fsm_w') as a:
        with state('A'):
          a.o = counter(10)
          emit_until()
        with state('B'):
          a.o = counter(100)
      return a.o

    def sole_uncond_until_self_reset():
      r_inner = Stream([False, True, False, False, False, False])
      with automaton() as a:
        with state('A'):
          a.o = counter(10)
          with reset(r_inner):
            until(True, 'A', reset=lambda: counter(0) == 1)
      return a.o

    def guarded_before_uncond_until(guard_rst: list[bool], unc_rst: list[bool]):
      def prog():
        r_g = Stream(guard_rst)
        r_u = Stream(unc_rst)
        guard = counter(0) == 99

        @psyre_node('u_sub')
        def emit_uncond():
          with reset(r_u):
            until(
                True,
                lambda: ifthenelse(counter(0, name='ut') == 1, 'B', 'A'),
                reset=lambda: counter(0, name='ur') == 1,
            )

        with automaton(name='fsm_gu') as a:
          with state('A'):
            a.o = counter(10)
            with reset(r_g):
              until(guard, 'B', reset=False)
            emit_uncond()
          with state('B'):
            a.o = 99
        return a.o

      return prog

    def sole_uncond_unless_reset_and_prefix():
      r_inner = Stream([False, True, False, False, False])

      @psyre_node('strong_sub')
      def make_strong():
        with reset(r_inner):
          return unless(
              True,
              lambda: ifthenelse(counter(0, name='st') == 1, 'B', 'A'),
              reset=lambda: counter(0, name='sr') == 1,
          )

      with automaton(name='fsm_s') as a:
        with state('A', make_strong()):
          a.o = counter(10)
        with state('B'):
          a.o = counter(100)
      return a.o

    def sole_uncond_unless_self_reset():
      r_inner = Stream([False, True, False, False, False])
      with automaton() as a:
        with reset(r_inner):
          clause = unless(True, 'A', reset=lambda: counter(0) == 1)
        with state('A', clause):
          a.o = counter(10)
      return a.o

    def guarded_before_uncond_unless(
        guard_rst: list[bool], unc_rst: list[bool]
    ):
      def prog():
        r_g = Stream(guard_rst)
        r_u = Stream(unc_rst)

        @psyre_node('us_sub')
        def make_uncond():
          with reset(r_u):
            return unless(
                True,
                lambda: ifthenelse(counter(0, name='ust') == 1, 'B', 'A'),
                reset=lambda: counter(0, name='usr') == 1,
            )

        with automaton(name='fsm_gs') as a:
          with reset(r_g):
            c_g = unless(counter(0) == 99, 'B', reset=False)
          with state('A', c_g, make_uncond()):
            a.o = counter(10)
          with state('B'):
            a.o = 99
        return a.o

      return prog

    # Verify @psyre_node prefix captured on deferred unconditional transitions.
    prefix_checks = (
        (sole_uncond_until_target_and_reset, 'fsm_w', 'weak_sub', ('wt', 'wr')),
        (
            sole_uncond_unless_reset_and_prefix,
            'fsm_s',
            'strong_sub',
            ('st', 'sr'),
        ),
        (
            guarded_before_uncond_until([False, False], [False, True, False]),
            'fsm_gu',
            'u_sub',
            ('ut', 'ur'),
        ),
        (
            guarded_before_uncond_unless([False, False], [False, True, False]),
            'fsm_gs',
            'us_sub',
            ('ust', 'usr'),
        ),
    )
    for prog, fsm_pfx, sub_pfx, node_names in prefix_checks:
      g = build_graph(prog, optimize=False)
      for node_name in node_names:
        stream = g.resolve_named_streams([f'{fsm_pfx}/{sub_pfx}/{node_name}'])[
            f'{fsm_pfx}/{sub_pfx}/{node_name}'
        ]
        node = g.producer(stream)
        self.assertIsNotNone(node)
        self.assertEqual([p for p, _ in node.prefixes], [fsm_pfx, sub_pfx])

    no_rst = [False] * 4
    pulse1 = [False, True, False, False]
    run_cases = (
        ('until_target', sole_uncond_until_target_and_reset, 5, [
            10, 11, 12, 100, 101
        ]),
        (
            'until_self',
            sole_uncond_until_self_reset,
            6,
            [10, 11, 12, 10, 11, 10],
        ),
        (
            'guarded_until_u',
            guarded_before_uncond_until(no_rst, pulse1),
            4,
            [10, 11, 12, 99],
        ),
        (
            'guarded_until_g',
            guarded_before_uncond_until(pulse1, no_rst),
            4,
            [10, 11, 99, 99],
        ),
        ('unless_target', sole_uncond_unless_reset_and_prefix, 4, [
            10, 11, 100, 101
        ]),
        (
            'unless_self',
            sole_uncond_unless_self_reset,
            5,
            [10, 11, 10, 11, 12],
        ),
        (
            'guarded_unless_u',
            guarded_before_uncond_unless(no_rst, pulse1),
            4,
            [10, 11, 99, 99],
        ),
        (
            'guarded_unless_g',
            guarded_before_uncond_unless(pulse1, no_rst),
            4,
            [10, 99, 99, 99],
        ),
    )
    for sem in (Semantics.SYNC, Semantics.KAHN):
      for opt in (False, True):
        for label, prog, horizon, expected in run_cases:
          with self.subTest(case=label, semantics=sem, optimize=opt):
            self.assertEqual(
                Scheduler.run(
                    prog, horizon=horizon, semantics=sem, optimize=opt
                ),
                expected,
            )

  def test_strong_unless_disambiguates_bool_and_int_state_tags(self):
    def bool_and_int_states():
      t = counter(0)
      with automaton(init=0) as a:
        with state(0, unless(t == 1, False)):
          a.o = 'ZERO'
        with state(False, unless(t == 2, 1)):
          a.o = 'FALSE'
        with state(1, unless(t == 3, True)):
          a.o = 'ONE'
        with state(True, unless(t == 4, 0)):
          a.o = 'TRUE'
      return a.o

    for sem in (Semantics.SYNC, Semantics.KAHN):
      for opt in (False, True):
        with self.subTest(semantics=sem, optimize=opt):
          self.assertEqual(
              Scheduler.run(
                  bool_and_int_states, horizon=6, semantics=sem, optimize=opt
              ),
              ['ZERO', 'FALSE', 'ONE', 'TRUE', 'ZERO', 'ZERO'],
          )

  def test_unconditional_literal_reset_preserves_enclosing_reset(self):
    def prog():
      r_inner = Stream([False, True, False, False])
      with automaton() as a:
        with state('A'):
          with reset(r_inner):
            until(True, 'A', reset=[False, True, False])
          a.y = counter(10)
      return a.y

    for sem in (Semantics.SYNC, Semantics.KAHN):
      for opt in (False, True):
        with self.subTest(semantics=sem, optimize=opt):
          self.assertEqual(
              Scheduler.run(prog, horizon=4, semantics=sem, optimize=opt),
              [10, 11, 12, 10],
          )

  def test_intenum_exhaustiveness_nan_tag_and_array_guard(self):
    class _IntMode(enum.IntEnum):
      A = 0
      B = 1

    def non_exhaustive_int_init():
      with automaton(init=0) as a:
        with state(_IntMode.A):
          a.y = 1
      return a.y

    with self.assertRaisesRegex(ValueError, 'Non-exhaustive'):
      build_graph(non_exhaustive_int_init)

    def nan_state():
      with automaton() as a:
        with state(float('nan')):
          a.y = 1
      return a.y

    with self.assertRaisesRegex(ValueError, 'NaN'):
      build_graph(nan_state)

    def array_cond():
      with automaton() as a:
        with state('A'):
          until(test_utils.FakeArray(), 'A')
          a.y = 1
      return a.y

    def array_reset():
      with automaton() as a:
        with state('A'):
          until(False, 'A', reset=test_utils.FakeArray())
          a.y = 1
      return a.y

    for bad_prog in (array_cond, array_reset):
      with self.assertRaisesRegex(TypeError, 'scalar truth value'):
        build_graph(bad_prog)


if __name__ == '__main__':
  absltest.main()
