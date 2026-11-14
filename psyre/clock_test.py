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
"""Unit tests for the Psyre synchronous clock representation and calculus."""

import enum
import logging

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import (
    check_causality,
)
from psyre.automaton import (
    automaton,
    state,
    unless,
    until,
)
from psyre.clock import (
    BASE_CLOCK,
    Clock,
    ClockError,
    check_clocks,
    infer_clocks,
)
from psyre.graph import Graph
from psyre.istructure import is_one_periodic
from psyre.match import (
    case,
    default,
    last,
    match,
)
from psyre.psyre import (
    Absent,
    Scheduler,
    Semantics,
    Stream,
    buffer,
    counter,
    delay,
    fby,
    logger,
    merge,
    periodic_stream,
    post,
    pre,
    prefix_sum,
    reset,
    scan,
    when,
    whennot,
)

# pylint: enable=g-importing-member,g-multiple-import


class _TagMode(enum.Enum):
  """Discrete mode enum testing Clock pretty-printing."""

  FAST = 1
  SLOW = 2


class ClockTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def test_clock_representation_and_validation_guards(self):
    self.assertIsInstance(BASE_CLOCK, Clock)
    self.assertTrue(BASE_CLOCK.is_base)
    self.assertFalse(BASE_CLOCK.is_polymorphic)
    self.assertTrue(bool(BASE_CLOCK))
    self.assertEqual(str(BASE_CLOCK), '.')
    self.assertEqual(str(Clock.polymorphic(0)), "'a")
    self.assertEqual(str(Clock.polymorphic(1)), "'b")
    self.assertEqual(str(Clock.polymorphic(26)), "'a1")
    self.assertEqual(str(Clock.polymorphic('k')), "'k")
    self.assertEqual(str(Clock.polymorphic("'m")), "'m")

    with self.assertRaisesRegex(ValueError, 'Invalid polymorphic clock'):
      Clock(var='')
    with self.assertRaisesRegex(ValueError, 'Clock variable index'):
      Clock.polymorphic(-1)
    with self.assertRaisesRegex(RuntimeError, 'not been finalized'):
      check_clocks(Graph())
    with self.assertRaisesRegex(RuntimeError, 'not been finalized'):
      infer_clocks(Graph())

  def test_stage_one_clock_inference_and_validation(self):
    holders: dict[str, Stream] = {}

    def well_clocked():
      x = Stream([10, 20, 30, 40], name='x')
      c_const = Stream(7, name='c_const')
      c_const2 = Stream(3, name='c_const2')
      odd_p = (x & 1) == 1
      odd_p.name = 'odd_p'
      even_p = ~odd_p
      odd = whennot(even_p, x)
      even = when(even_p, x * 2)
      merged = merge(odd_p, odd, even)
      ip = Stream([True, False, True, False], name='ip')
      op = Stream([False, True, False, True], name='op')
      buf = buffer(when(ip, x), ip, op, size=2)
      with match(op) as buf_m:
        with case(True):
          consumed_buf = buf + c_const
          sub_pre = pre(consumed_buf)
          sub_pre2 = pre(consumed_buf, 2)
          sub_fby = fby(c_const, consumed_buf)
          sub_fby2 = fby(c_const, consumed_buf, 2)
          sub_post = post(consumed_buf)
          sub_post2 = post(consumed_buf, 2)
          buf_m.out = consumed_buf + sub_fby + sub_fby2 + sub_post + sub_post2
          buf_m.prev = sub_pre
          buf_m.prev2 = sub_pre2
        with case(False):
          buf_m.out = 0
          buf_m.prev = 0
          buf_m.prev2 = 0
      with match(odd_p) as m:
        prev = last(lambda: m.out, 0, 2)
        with case(True):
          m.out = x + prev + c_const2
        with case(False):
          m.out = x * 2 + prev
      cmd = Stream(['ADD', 'A', 'B', 'NOP'], name='cmd')
      with match(cmd) as cmd_m:
        with case('ADD'):
          add_arm = x + 1
          cmd_m.out = add_arm
        with case('A', 'B'):
          ab_arm = x * 2
          cmd_m.out = ab_arm
        with default():
          def_arm = x - 1
          cmd_m.out = def_arm
      mode = Stream(
          [_TagMode.FAST, _TagMode.SLOW, _TagMode.FAST, _TagMode.SLOW],
          name='mode',
      )
      with match(mode) as mode_m:
        with case(_TagMode.FAST):
          fast_arm = x + 10
          mode_m.out = fast_arm
        with default():
          slow_arm = x - 10
          mode_m.out = slow_arm
      holders.update(
          x=x,
          c_const=c_const,
          c_const2=c_const2,
          odd=odd,
          even=even,
          merged=merged,
          buf=buf,
          consumed_buf=consumed_buf,
          sub_pre=sub_pre,
          sub_pre2=sub_pre2,
          sub_fby=sub_fby,
          sub_fby2=sub_fby2,
          sub_post=sub_post,
          sub_post2=sub_post2,
          match_out=m.out,
          add_arm=add_arm,
          ab_arm=ab_arm,
          def_arm=def_arm,
          fast_arm=fast_arm,
          slow_arm=slow_arm,
      )
      return (
          merged,
          buf_m.out,
          buf_m.prev,
          buf_m.prev2,
          m.out,
          cmd_m.out,
          mode_m.out,
      )

    for sem in (Semantics.KAHN, Semantics.SYNC):
      graph = Scheduler.build_graph(well_clocked, semantics=sem)
      check_causality(graph)
      clocks = check_clocks(graph)
      self.assertIs(clocks.graph, graph)
      with self.assertRaisesRegex(ValueError, 'does not belong to this Graph'):
        clocks.stream_clock(Stream([1, 2]))
      self.assertTrue(is_one_periodic(holders['c_const']))
      c_clk = clocks.stream_clock(holders['c_const'])
      c2_clk = clocks.stream_clock(holders['c_const2'])
      self.assertIsNotNone(c_clk)
      self.assertIsNotNone(c2_clk)
      self.assertTrue(c_clk.is_polymorphic)
      self.assertEqual(str(c_clk), "'a")
      self.assertEqual(str(c2_clk), "'b")
      self.assertEqual(clocks.stream_clock(holders['x']), BASE_CLOCK)
      self.assertEqual(str(clocks.stream_clock(holders['x'])), '.')
      self.assertEqual(clocks.stream_clock(holders['merged']), BASE_CLOCK)
      self.assertEqual(clocks.stream_clock(holders['match_out']), BASE_CLOCK)
      odd_clk = clocks.stream_clock(holders['odd'])
      even_clk = clocks.stream_clock(holders['even'])
      buf_clk = clocks.stream_clock(holders['buf'])
      consumed_clk = clocks.stream_clock(holders['consumed_buf'])
      self.assertIsNotNone(odd_clk)
      self.assertLen(odd_clk, 1)
      self.assertNotEqual(odd_clk, even_clk)
      self.assertEqual(str(odd_clk), '. on odd_p')
      self.assertEqual(str(even_clk), '. on not odd_p')
      poly_sub = c_clk.on(odd_clk.steps[0])
      self.assertEqual(str(poly_sub), "'a on odd_p")
      self.assertTrue(c_clk.is_prefix_of(poly_sub))
      self.assertFalse(BASE_CLOCK.is_prefix_of(poly_sub))
      self.assertIsNotNone(buf_clk)
      self.assertEqual(buf_clk, consumed_clk)
      self.assertEqual(str(buf_clk), '. on op')
      self.assertEqual(str(consumed_clk), '. on op')
      for delay_key in (
          'sub_pre',
          'sub_pre2',
          'sub_fby',
          'sub_fby2',
          'sub_post',
          'sub_post2',
      ):
        self.assertEqual(clocks.stream_clock(holders[delay_key]), buf_clk)
        delay_prod = graph.producer(holders[delay_key])
        self.assertIsNotNone(delay_prod)
        self.assertEqual(clocks.node_clock(delay_prod), buf_clk)
        for in_h in delay_prod.inputs:
          self.assertEqual(clocks.stream_clock(in_h), buf_clk)
      self.assertEqual(
          str(clocks.stream_clock(holders['add_arm'])), ". on (cmd == 'ADD')"
      )
      self.assertEqual(
          str(clocks.stream_clock(holders['ab_arm'])),
          ". on (cmd in {'A', 'B'})",
      )
      self.assertEqual(
          str(clocks.stream_clock(holders['def_arm'])),
          ". on (cmd not in {'A', 'ADD', 'B'})",
      )
      self.assertEqual(
          str(clocks.stream_clock(holders['fast_arm'])),
          '. on (mode == _TagMode.FAST)',
      )
      self.assertEqual(
          str(clocks.stream_clock(holders['slow_arm'])),
          '. on (mode != _TagMode.FAST)',
      )
      buf_prod = graph.producer(holders['buf'])
      self.assertIsNotNone(buf_prod)
      self.assertEqual(clocks.node_clock(buf_prod), BASE_CLOCK)
      consumed_prod = graph.producer(holders['consumed_buf'])
      self.assertIsNotNone(consumed_prod)
      self.assertEqual(
          clocks.node_activity_clocks(consumed_prod), (BASE_CLOCK, buf_clk)
      )

    def disjoint_subclocks():
      c = Stream([True, False, True, False])
      x = Stream([1, 2, 3, 4])
      return when(c, x) + whennot(c, x)

    def subclock_plus_base():
      c = Stream([True, False, True, False])
      return when(c, [1, 2, 3, 4]) + counter()

    def unguarded_merge_arm():
      c = Stream([True, False, True, False])
      return merge(c, counter(), whennot(c, [1, 2, 3, 4]))

    def unguarded_buffer_input():
      ip = Stream([True, False, True, False])
      op = Stream([False, True, False, True])
      return buffer(counter(), ip, op)

    def subclock_into_base_fby():
      c = Stream([True, False, True, False])
      return fby(0, when(c, [1, 2, 3, 4]))

    def subclock_into_base_pre():
      c = Stream([True, False, True, False])
      return pre(when(c, [1, 2, 3, 4]))

    def stateful_without_match():
      c = Stream([True, False, True, False])
      return scan(when(c, [1, 2, 3, 4]))

    def subclock_match_gate():
      c = Stream([True, False, True, False])
      with match(when(c, [True, True, False, True])) as m:
        with case(True):
          m.out = counter()
        with case(False):
          m.out = 0
      return m.out

    def subclock_reset_gate():
      c = Stream([True, False, True, False])
      with reset(when(c, [False, True, False, False])):
        return counter()

    def mismatched_when_in_match():
      c1 = Stream([True, False, True, False])
      c2 = Stream([False, True, False, True])
      w = when(c2, [1, 2, 3, 4])
      with match(c1) as m:
        with case(True):
          m.out = w + 1
        with case(False):
          m.out = 0
      return m.out

    def redundant_when_in_match():
      c = Stream([True, False, True, False])
      x = Stream([1, 2, 3, 4])
      y = Stream([10, 20, 30, 40])
      with match(c) as m:
        with case(True):
          m.out = when(c, x) + y
        with case(False):
          m.out = 0
      return m.out

    def conflicting_external_stream():
      c = Stream([True, False, True, False])
      shared = Stream([10, 20, 30, 40])
      return shared + 1, merge(c, shared, whennot(c, [1, 2, 3, 4]))

    def cross_activity_reset():
      c = Stream([True, False, True, False])
      r = Stream([False, True, False, False])
      with reset(r):
        with match(c) as m:
          with case(True):
            m.out = counter()
          with case(False):
            m.out = 0
      return m.out

    bad_programs = (
        disjoint_subclocks,
        subclock_plus_base,
        unguarded_merge_arm,
        unguarded_buffer_input,
        subclock_into_base_fby,
        subclock_into_base_pre,
        stateful_without_match,
        subclock_match_gate,
        subclock_reset_gate,
        mismatched_when_in_match,
        redundant_when_in_match,
        conflicting_external_stream,
    )
    for prog in bad_programs:
      with self.subTest(program=prog.__name__):
        sync_g = Scheduler.build_graph(prog, semantics=Semantics.SYNC)
        with self.assertRaisesRegex(ClockError, 'Clock error'):
          check_clocks(sync_g)
        with self.assertRaisesRegex(ClockError, 'Clock error'):
          Scheduler.run(prog, horizon=4, semantics=Semantics.KAHN)

    # Supersampled outer reset around match is valid in SYNC, but cross-activity
    # stateful reset is forbidden in KAHN (including clock_calculus=False):
    self.assertEqual(
        Scheduler.run(
            cross_activity_reset, horizon=4, semantics=Semantics.SYNC
        ),
        [0, 0, 0, 0],
    )
    with self.assertRaisesRegex(ClockError, 'cross-activity reset'):
      Scheduler.run(cross_activity_reset, horizon=4, semantics=Semantics.KAHN)
    with self.assertRaisesRegex(ClockError, 'cross-activity reset'):
      Scheduler.run(
          cross_activity_reset,
          horizon=4,
          semantics=Semantics.KAHN,
          clock_calculus=False,
      )
    kahn_g = Scheduler.build_graph(
        cross_activity_reset, semantics=Semantics.KAHN
    )
    with self.assertRaisesRegex(ClockError, 'cross-activity reset'):
      check_clocks(kahn_g)
    with self.assertRaisesRegex(ClockError, 'cross-activity reset'):
      Scheduler.run_graph(kahn_g, horizon=4)

  def test_automaton_clock_inference_and_validation(self):
    holders: dict[str, Stream] = {}

    def well_clocked_automaton():
      inp = Stream([1, 2, 3, 4, 5], name='inp')
      tag_sel = Stream([True, False, True, False, True], name='tag_sel')
      with automaton() as fsm:
        prev = last(lambda: fsm.out, init=0, name='prev')
        with state('RUN', unless(prev >= 6, 'DONE')):
          cnt = counter(start=1, name='run_cnt')
          with match(tag_sel) as bonus:
            with case(True):
              bonus_true = inp + 0
              bonus.v = bonus_true
            with case(False):
              bonus.v = 0
          run_val = cnt + bonus.v
          until(lambda: fsm.out > 3, 'HOLD')
          fsm.out = run_val
          holders['bonus_true'] = bonus_true
          holders['run_val'] = run_val
        with state('HOLD'):
          hold_val = prev + 1
          until(True, 'RUN', reset=False)
          fsm.out = hold_val
          holders['hold_val'] = hold_val
        with state('DONE'):
          fsm.out = 99
      holders['fsm_out'] = fsm.out
      return fsm.out

    graph = Scheduler.build_graph(
        well_clocked_automaton, semantics=Semantics.SYNC
    )
    check_causality(graph)
    clocks = check_clocks(graph)
    self.assertEqual(clocks.stream_clock(holders['fsm_out']), BASE_CLOCK)
    self.assertEqual(str(clocks.stream_clock(holders['fsm_out'])), '.')
    run_clk = clocks.stream_clock(holders['run_val'])
    hold_clk = clocks.stream_clock(holders['hold_val'])
    bonus_true_clk = clocks.stream_clock(holders['bonus_true'])
    self.assertIsNotNone(run_clk)
    self.assertIsNotNone(hold_clk)
    self.assertIsNotNone(bonus_true_clk)
    self.assertLen(run_clk, 1)
    self.assertLen(hold_clk, 1)
    self.assertLen(bonus_true_clk, 2)
    self.assertNotEqual(run_clk, hold_clk)
    self.assertEqual(str(run_clk), ". on (fsm_0/state_s == 'RUN')")
    self.assertEqual(str(hold_clk), ". on (fsm_0/state_s == 'HOLD')")
    self.assertEqual(
        str(bonus_true_clk), ". on (fsm_0/state_s == 'RUN') on tag_sel"
    )
    self.assertEqual(
        Scheduler.run_graph(graph, horizon=5),
        [2, 2, 6, 7, 99],
    )

    def subclock_state_output():
      sub = Stream([True, False, True, False])
      x = Stream([10, 20, 30, 40])
      with automaton() as fsm:
        with state('A'):
          until(False, 'B')
          fsm.out = when(sub, x)
        with state('B'):
          fsm.out = 0
      return fsm.out

    sub_out_g = Scheduler.build_graph(
        subclock_state_output, semantics=Semantics.SYNC
    )
    with self.assertRaisesRegex(ClockError, 'merge arm 0'):
      check_clocks(sub_out_g)

    def subclock_until_guard():
      sub = Stream([True, False, True, False])
      with automaton() as fsm:
        with state('A'):
          until(when(sub, True), 'B')
          fsm.out = 1
        with state('B'):
          fsm.out = 2
      return fsm.out

    until_g = Scheduler.build_graph(
        subclock_until_guard, semantics=Semantics.SYNC
    )
    with self.assertRaisesRegex(ClockError, 'incompatible input clocks'):
      check_clocks(until_g)

    def subclock_unless_guard():
      sub = Stream([True, False, True, False])
      with automaton() as fsm:
        with state('A', unless(when(sub, True), 'B')):
          fsm.out = 1
        with state('B'):
          fsm.out = 2
      return fsm.out

    unless_g = Scheduler.build_graph(
        subclock_unless_guard, semantics=Semantics.SYNC
    )
    with self.assertRaisesRegex(
        ClockError, 'incompatible with enclosing activity clock'
    ):
      check_clocks(unless_g)

    def subclock_state_reset():
      sub = Stream([True, False, True, False])
      with automaton() as fsm:
        with state('A'):
          with reset(when(sub, True)):
            fsm.out = counter()
        with state('B'):
          fsm.out = 0
      return fsm.out

    sub_rst_g = Scheduler.build_graph(
        subclock_state_reset, semantics=Semantics.SYNC
    )
    with self.assertRaisesRegex(
        ClockError, 'incompatible with reset scope clock'
    ):
      check_clocks(sub_rst_g)

    def hierarchical_modular_reset():
      outer_flip = Stream([False, True, True, False])
      inner_flip = Stream([True, False, False, False])
      with automaton() as outer:
        with state('ON'):
          with automaton() as inner:
            with state('X'):
              until(inner_flip, 'Y', reset=False)
              inner.v = counter(start=1)
            with state('Y'):
              until(inner_flip, 'X', reset=False)
              inner.v = counter(start=10)
          until(outer_flip, 'OFF')
          outer.out = inner.v
        with state('OFF'):
          until(outer_flip, 'ON')
          outer.out = -1
      return outer.out

    hier_g = Scheduler.build_graph(
        hierarchical_modular_reset, semantics=Semantics.SYNC
    )
    check_causality(hier_g)
    hier_clocks = check_clocks(hier_g)
    self.assertEqual(hier_clocks.stream_clock(hier_g.output), BASE_CLOCK)
    self.assertEqual(
        infer_clocks(hier_g).stream_clock(hier_g.output), BASE_CLOCK
    )
    self.assertEqual(Scheduler.run_graph(hier_g, horizon=4), [1, 10, -1, 1])

  def test_supersampling_and_subclock_nodes(self):
    for bad_fn in (
        lambda: Stream(Absent),
        lambda: Stream([10, Absent]),
        lambda: periodic_stream([Absent], [10]),
        lambda: periodic_stream([10], [Absent]),
    ):
      with self.assertRaisesRegex(TypeError, 'cannot contain Absent'):
        bad_fn()

    holders: dict[str, Stream] = {}

    def supersampled_program():
      tick_0 = (counter(name='cnt') % 3) == 0
      tick_0.name = 'tick_0'

      # External input clocked on subclock `. on tick_0` and supersampled onto
      # the faster base clock `.` via `buffer`:
      ext_sub = Stream([10, 0, 0, 20, 0, 0], name='ext_sub')
      sup_buf = buffer(ext_sub, tick_0, True, size=2, name='sup_buf')

      # Downsampled stream `. on tick_0` feeding combinational nodes outside
      # `match` and stateful nodes inside `match(tick_0)`:
      dense_x = Stream([1, 2, 3, 4, 5, 6], name='dense_x')
      sub_w = when(tick_0, dense_x, name='sub_w')
      sub_comb = sub_w * 2 + 1
      with match(tick_0) as m:
        with case(True):
          fby_held = fby(0, dense_x * 10, name='fby_held')
          pre_out = pre(dense_x * 10, name='pre_out')
          delay_out = delay(dense_x * 10, 1, name='delay_out')
          scan_out = scan(dense_x * 10, name='scan_out')
          psum_out = prefix_sum(dense_x * 10, start=100, name='psum_out')
          m.out = sub_comb
        with case(False):
          m.out = 0
      sup_match = m.out

      holders.update(
          tick_0=tick_0,
          ext_sub=ext_sub,
          sup_buf=sup_buf,
          sub_w=sub_w,
          sub_comb=sub_comb,
          sup_match=sup_match,
          fby_held=fby_held,
          pre_out=pre_out,
          delay_out=delay_out,
          scan_out=scan_out,
          psum_out=psum_out,
      )
      return (
          sup_buf,
          sup_match,
          sub_comb,
      )

    sub_names = ('fby_held', 'pre_out', 'delay_out', 'scan_out', 'psum_out')
    graph = Scheduler.build_graph(
        supersampled_program,
        semantics=Semantics.SYNC,
        extra_roots=sub_names,
    )
    check_causality(graph)
    clocks = check_clocks(graph)

    self.assertEqual(
        str(clocks.stream_clock(holders['ext_sub'])), '. on tick_0'
    )
    self.assertEqual(str(clocks.stream_clock(holders['sub_w'])), '. on tick_0')
    for key in (
        'sub_comb',
        'fby_held',
        'pre_out',
        'delay_out',
        'scan_out',
        'psum_out',
    ):
      with self.subTest(subclock_stream=key):
        self.assertEqual(str(clocks.stream_clock(holders[key])), '. on tick_0')
        prod = graph.producer(holders[key])
        self.assertIsNotNone(prod)
        self.assertEqual(str(clocks.node_clock(prod)), '. on tick_0')
        self.assertEqual(
            clocks.node_activity_clocks(prod),
            (BASE_CLOCK, clocks.node_clock(prod)),
        )

    for key in ('tick_0', 'sup_buf', 'sup_match'):
      with self.subTest(stream=key):
        self.assertEqual(clocks.stream_clock(holders[key]), BASE_CLOCK)
        prod = graph.producer(holders[key])
        self.assertIsNotNone(prod)
        self.assertEqual(clocks.node_clock(prod), BASE_CLOCK)

    sup_buf_tr, sup_match_tr, sub_comb_tr = Scheduler.run_graph(
        graph, horizon=6
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(sup_buf_tr, [10, None, None, 20, None, None])
    self.assertEqual(sup_match_tr, [3, 0, 0, 9, 0, 0])
    self.assertEqual(sub_comb_tr, [3, Absent, Absent, 9, Absent, Absent])
    sub_traces = Scheduler.run_graph(graph, horizon=6, streams=sub_names)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(
        sub_traces['fby_held'], [0, Absent, Absent, 10, Absent, Absent]
    )
    self.assertEqual(
        sub_traces['pre_out'], [None, Absent, Absent, 10, Absent, Absent]
    )
    self.assertEqual(
        sub_traces['delay_out'], [Absent, Absent, Absent, 10, Absent, Absent]
    )
    self.assertEqual(
        sub_traces['scan_out'], [10, Absent, Absent, 50, Absent, Absent]
    )
    self.assertEqual(
        sub_traces['psum_out'], [110, Absent, Absent, 150, Absent, Absent]
    )

    # Also verify KAHN supersampling via `buffer` where the slow external input
    # only provides 2 tokens across a 6-tick base-clock horizon:
    kahn_holders: dict[str, Stream] = {}

    def kahn_supersample():
      t0 = (counter(name='k_cnt') % 3) == 0
      t0.name = 't0'
      slow_in = Stream([10, 20], name='slow_in')
      k_buf = buffer(slow_in, t0, True, size=2, name='k_buf')
      kahn_holders.update(slow_in=slow_in, k_buf=k_buf)
      return k_buf

    kahn_g = Scheduler.build_graph(kahn_supersample, semantics=Semantics.KAHN)
    kahn_clocks = check_clocks(kahn_g)
    self.assertEqual(
        str(kahn_clocks.stream_clock(kahn_holders['slow_in'])), '. on t0'
    )
    self.assertEqual(
        kahn_clocks.stream_clock(kahn_holders['k_buf']), BASE_CLOCK
    )
    kahn_tr = Scheduler.run_graph(kahn_g, horizon=6)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(kahn_tr, [10, None, None, 20, None, None])


if __name__ == '__main__':
  absltest.main()
