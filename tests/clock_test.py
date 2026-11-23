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
from psyre.builder import build_graph, lower_resets
from psyre.clock import (
    BASE_CLOCK,
    Clock,
    ClockError,
    ClockStep,
    check_clocks,
)
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
from psyre.graph import Graph
from psyre.match import (
    case,
    default,
    last,
    match,
)
from psyre.scheduler import Scheduler
from psyre.stream import Stream
from psyre.trace import present_values

# pylint: enable=g-importing-member,g-multiple-import


class _TagMode(enum.Enum):
  """Discrete mode enum testing Clock pretty-printing."""

  FAST = 1
  SLOW = 2


class ClockTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def _assert_no_sentinels(self, trace):
    """Asserts an unpadded trace contains neither Absent nor Bottom."""
    self.assertTrue(
        all(v is not Absent and v is not Bottom for v in trace),
        f'Unexpected Absent/Bottom sentinel in trace: {trace!r}',
    )

  def _assert_sync_kahn_elision(self, sync_tr, kahn_tr):
    """Asserts SYNC and KAHN agree after Absent/Bottom elision."""
    self._assert_no_sentinels(kahn_tr)
    self.assertEqual(present_values(sync_tr), kahn_tr)

  def _assert_lower_resets_idempotent(self, g, clocks=None):
    """Checks `lower_resets` causality, clock soundness, and idempotency."""
    low = lower_resets(g, check_clocks(g) if clocks is None else clocks)
    check_causality(low)
    low_clks = check_clocks(low)
    self.assertIs(lower_resets(low, low_clks), low)
    return low, low_clks

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

    # ClockStep equality & hashing distinguish distinct Stream selectors with
    # identical tags via selector_id, and separate truthy vs tag-match steps.
    s1 = Stream([True, False], name='s1')
    s2 = Stream([True, False], name='s2')
    tags = frozenset({(True, True)})
    step_s1_a = ClockStep(s1, tags, is_boolean=True, selector_label='a')
    step_s1_b = ClockStep(s1, tags, is_boolean=True, selector_label='b')
    step_s2 = ClockStep(s2, tags, is_boolean=True, selector_label='a')
    step_s1_tag = ClockStep(s1, tags, is_boolean=False, selector_label='a')
    self.assertEqual(step_s1_a, step_s1_b)
    self.assertEqual(hash(step_s1_a), hash(step_s1_b))
    self.assertNotEqual(step_s1_a, step_s2)
    self.assertNotEqual(step_s1_a, step_s1_tag)
    self.assertLen({step_s1_a, step_s1_b, step_s2, step_s1_tag}, 3)

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
      graph = build_graph(well_clocked, semantics=sem)
      check_causality(graph)
      clocks = check_clocks(graph)
      self.assertIs(clocks.graph, graph)
      with self.assertRaisesRegex(ValueError, 'does not belong to this Graph'):
        clocks.stream_clock(Stream([1, 2]))
      self.assertTrue(holders['c_const'].is_one_periodic)
      c_clk = clocks.stream_clock(holders['c_const'])
      c2_clk = clocks.stream_clock(holders['c_const2'])
      self.assertIsNotNone(c_clk)
      self.assertIsNotNone(c2_clk)
      self.assertTrue(c_clk.is_polymorphic)
      self.assertTrue(c2_clk.is_polymorphic)
      self.assertNotEqual(c_clk, c2_clk)
      # All unpinned static streams (both 1-periodic and multi-element) are
      # clock-polymorphic (`'a`, `'b`, ...).
      self.assertTrue(clocks.stream_clock(holders['x']).is_polymorphic)
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
      self.assertEqual(str(poly_sub), f'{c_clk} on odd_p')
      self.assertTrue(c_clk.is_prefix_of(poly_sub))
      self.assertFalse(BASE_CLOCK.is_prefix_of(poly_sub))
      self.assertIsNotNone(buf_clk)
      self.assertEqual(buf_clk, consumed_clk)
      self.assertEqual(str(buf_clk), '. on op')
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
          if in_h.static and not graph.is_base_clock_pinned(in_h):
            self.assertTrue(clocks.stream_clock(in_h).is_polymorphic)
          else:
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
      shared = Stream()
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

    def coarse_reset_without_match():
      c = Stream([True, False, True, False])
      r = Stream([False, True, False, False])
      x = Stream([1, 2, 3, 4])
      with reset(r):
        return fby(0, when(c, x)), scan(when(c, x))

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
        sync_g = build_graph(prog, semantics=Semantics.SYNC)
        with self.assertRaisesRegex(ClockError, 'Clock error'):
          check_clocks(sync_g)
        with self.assertRaisesRegex(ClockError, 'Clock error'):
          Scheduler.run(prog, horizon=4, semantics=Semantics.KAHN)

    # Supersampled outer reset around match or outside match/when is valid in
    # both SYNC and KAHN when clock analysis is provided, and fails explicitly
    # in KAHN when clock analysis is omitted.
    self.assertEqual(
        Scheduler.run(
            cross_activity_reset, horizon=4, semantics=Semantics.SYNC
        ),
        [0, 0, 0, 0],
    )
    kahn_g = build_graph(
        cross_activity_reset, semantics=Semantics.KAHN
    )
    kahn_clocks = check_clocks(kahn_g)
    self.assertEqual(
        Scheduler.run_graph(kahn_g, horizon=4, clocks=kahn_clocks),
        [0, 0, 0, 0],
    )
    with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
      Scheduler.run_graph(kahn_g, horizon=4)
    with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
      Scheduler.run(
          cross_activity_reset,
          horizon=4,
          semantics=Semantics.KAHN,
          clock_calculus=False,
      )

    # Resetting a subclock stateful node outside `when` or `match`:
    self.assertEqual(
        Scheduler.run(
            coarse_reset_without_match, horizon=4, semantics=Semantics.SYNC
        ),
        ([0, Absent, 0, Absent], [1, Absent, 3, Absent]),
    )
    self.assertEqual(
        Scheduler.run(
            coarse_reset_without_match, horizon=4, semantics=Semantics.KAHN
        ),
        ([0, 0], [1, 3]),
    )
    # Direct Stage-1 lower_resets produces a causal, well-clocked graph whose
    # resets all match their consumer node clocks (idempotent) without mutating
    # the original graph:
    coarse_g = build_graph(
        coarse_reset_without_match, semantics=Semantics.KAHN
    )
    orig_len = len(coarse_g.nodes)
    coarse_clocks = check_clocks(coarse_g)
    lowered_coarse = lower_resets(coarse_g, coarse_clocks)
    self.assertLen(coarse_g.nodes, orig_len)
    self.assertIsNot(lowered_coarse, coarse_g)
    check_causality(lowered_coarse)
    lowered_clocks = check_clocks(lowered_coarse)
    self.assertIs(lower_resets(lowered_coarse, lowered_clocks), lowered_coarse)
    self.assertEqual(
        Scheduler.run_graph(
            lowered_coarse, horizon=4, clocks=lowered_clocks
        ),
        ([0, 0], [1, 3]),
    )
    with self.assertRaisesRegex(ValueError, 'mandatory under Semantics.SYNC'):
      Scheduler.run(
          coarse_reset_without_match,
          horizon=4,
          semantics=Semantics.SYNC,
          clock_calculus=False,
      )
    with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
      Scheduler.run(
          coarse_reset_without_match,
          horizon=4,
          semantics=Semantics.KAHN,
          clock_calculus=False,
      )

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

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        graph = build_graph(well_clocked_automaton, semantics=sem)
        check_causality(graph)
        clocks = check_clocks(graph)
        self.assertEqual(clocks.stream_clock(holders['fsm_out']), BASE_CLOCK)
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
            Scheduler.run_graph(graph, horizon=5, clocks=clocks),
            [2, 2, 5, 6, 99],
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

    sub_out_g = build_graph(
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

    until_g = build_graph(
        subclock_until_guard, semantics=Semantics.SYNC
    )
    with self.assertRaisesRegex(ClockError, 'merge arm 0'):
      check_clocks(until_g)

    def subclock_unless_guard():
      sub = Stream([True, False, True, False])
      with automaton() as fsm:
        with state('A', unless(when(sub, True), 'B')):
          fsm.out = 1
        with state('B'):
          fsm.out = 2
      return fsm.out

    unless_g = build_graph(
        subclock_unless_guard, semantics=Semantics.SYNC
    )
    with self.assertRaisesRegex(
        ClockError, 'outside a matching activation scope'
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

    sub_rst_g = build_graph(
        subclock_state_reset, semantics=Semantics.SYNC
    )
    with self.assertRaisesRegex(
        ClockError, 'incompatible with reset scope clock'
    ):
      check_clocks(sub_rst_g)

    def hierarchical_modular_reset():
      outer_flip = when(Stream(True), [False, True, True, False])
      inner_flip = when(Stream(True), [True, False, False, False])
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

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(hierarchical_semantics=sem):
        hier_g = build_graph(hierarchical_modular_reset, semantics=sem)
        check_causality(hier_g)
        hier_clocks = check_clocks(hier_g)
        self.assertEqual(hier_clocks.stream_clock(hier_g.output), BASE_CLOCK)
        self.assertEqual(
            Scheduler.run_graph(hier_g, horizon=4, clocks=hier_clocks),
            [1, 10, -1, 1],
        )
        if sem == Semantics.SYNC:
          self.assertEqual(
              Scheduler.run_graph(hier_g, horizon=4),
              [1, 10, -1, 1],
          )
        else:
          with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
            Scheduler.run_graph(hier_g, horizon=4)

  def test_supersampling_and_subclock_nodes(self):
    holders: dict[str, Stream] = {}

    def supersampled_program():
      tick_0 = (counter(name='cnt') % 3) == 0
      tick_0.name = 'tick_0'

      # External input clocked on subclock `. on tick_0`, buffered onto
      # `. on tick_1` (shifted 1 tick, avoiding underflow on empty ticks):
      tick_1 = (counter(name='cnt_out') % 3) == 1
      tick_1.name = 'tick_1'
      ext_sub = Stream([10, 20], name='ext_sub')
      sup_buf = buffer(ext_sub, tick_0, tick_1, size=2, name='sup_buf')

      # Downsampled stream `. on tick_0` feeding combinational nodes outside
      # `match` and stateful nodes inside `match(tick_0)`:
      dense_x = Stream([1, 2, 3, 4, 5, 6], name='dense_x')
      sub_w = when(tick_0, dense_x, name='sub_w')
      sub_comb = sub_w * 2 + 1
      with match(tick_0) as m:
        with case(True):
          fby_held = fby(0, dense_x * 10, name='fby_held')
          pre_out = pre(dense_x * 10, name='pre_out')
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
          scan_out=scan_out,
          psum_out=psum_out,
      )
      return (
          sup_buf,
          sup_match,
          sub_comb,
      )

    sub_names = ('fby_held', 'pre_out', 'scan_out', 'psum_out')
    graph = build_graph(
        supersampled_program,
        semantics=Semantics.SYNC,
        extra_roots=sub_names,
    )
    check_causality(graph)
    clocks = check_clocks(graph)

    self.assertTrue(clocks.stream_clock(holders['ext_sub']).is_polymorphic)
    buf_node = graph.producer(holders['sup_buf'])
    assert buf_node is not None
    self.assertEqual(str(clocks.input_clock(buf_node, 0)), '. on tick_0')
    self.assertEqual(str(clocks.stream_clock(holders['sub_w'])), '. on tick_0')
    for key in (
        'sub_comb',
        'fby_held',
        'pre_out',
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

    for key in ('tick_0', 'sup_match'):
      with self.subTest(stream=key):
        self.assertEqual(clocks.stream_clock(holders[key]), BASE_CLOCK)
        prod = graph.producer(holders[key])
        self.assertIsNotNone(prod)
        self.assertEqual(clocks.node_clock(prod), BASE_CLOCK)
    self.assertEqual(
        str(clocks.stream_clock(holders['sup_buf'])), '. on tick_1'
    )
    self.assertEqual(clocks.node_clock(buf_node), BASE_CLOCK)

    sup_buf_tr, sup_match_tr, sub_comb_tr = Scheduler.run_graph(
        graph, horizon=6, clocks=clocks
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(
        sup_buf_tr, [Absent, 10, Absent, Absent, 20, Absent]
    )
    self.assertEqual(sup_match_tr, [3, 0, 0, 9, 0, 0])
    self.assertEqual(sub_comb_tr, [3, Absent, Absent, 9, Absent, Absent])
    sub_traces = Scheduler.run_graph(
        graph, horizon=6, streams=(*sub_names, 'ext_sub'), clocks=clocks
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(
        sub_traces['ext_sub'], [10, 20, 20, 20, 20, 20]
    )
    self.assertEqual(
        sub_traces['fby_held'], [0, Absent, Absent, 10, Absent, Absent]
    )
    self.assertEqual(
        sub_traces['pre_out'], [None, Absent, Absent, 10, Absent, Absent]
    )
    self.assertEqual(
        sub_traces['scan_out'], [10, Absent, Absent, 30, Absent, Absent]
    )
    self.assertEqual(
        sub_traces['psum_out'], [110, Absent, Absent, 130, Absent, Absent]
    )

    # Check static clock typing of supersampling `buffer(..., tick_0, True)` to
    # `BASE_CLOCK` and confirm executing the unmatched `push_p=True` raises
    # `BufferError` (underflow) in both SYNC and KAHN.
    def kahn_buffered():
      t0 = (counter(name='k_cnt') % 3) == 0
      t1 = (counter(name='k_cnt_out') % 3) == 1
      return buffer(Stream([10, 20], name='slow_in'), t0, t1, size=2)

    kahn_g = build_graph(kahn_buffered, semantics=Semantics.KAHN)
    kahn_tr = Scheduler.run_graph(kahn_g, horizon=6)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(kahn_tr, [10, 20])

    def unmatched_supersample():
      t0 = (counter(name='u_cnt') % 3) == 0
      return buffer(Stream([10, 20], name='slow_in'), t0, True, size=2)

    for sem in (Semantics.SYNC, Semantics.KAHN):
      ug = build_graph(unmatched_supersample, semantics=sem)
      u_clks = check_clocks(ug)
      self.assertEqual(u_clks.stream_clock(ug.output), BASE_CLOCK)
      with self.assertRaises(ExceptionGroup) as eg:
        Scheduler.run_graph(ug, horizon=6, clocks=u_clks)
      self.assertTrue(
          any(
              isinstance(err, BufferError) and 'underflow' in str(err)
              for err in eg.exception.exceptions
          )
      )

  def test_multi_valued_when_switch_and_aliased_stream_clocks(self):
    holders: dict[str, Stream] = {}

    def prog():
      sel = Stream(['A', 'B', 'C', 'A'], name='sel')
      x = Stream([1, 2, 3, 4], name='x')
      c_dyn = counter(0, name='c_dyn') == 0
      v_dyn = counter(10, name='v_dyn')
      w_pos = when(sel, x, values=('A', 'B'))
      w_neg = when(sel, x, values='A', negated=True)
      sw = switch(sel, {'A': x, 'B': x + 1}, default=x + 2)
      id_sw = switch(sel, {'A': 'A', 'B': 'B', 'C': 'C'})
      unif_sw = switch(c_dyn, v_dyn, v_dyn)
      holders.update(
          w_pos=w_pos, w_neg=w_neg, sw=sw, id_sw=id_sw, unif_sw=unif_sw
      )
      return w_pos, w_neg, sw, id_sw, unif_sw

    g = build_graph(prog, semantics=Semantics.SYNC)
    clocks = check_clocks(g)
    self.assertEqual(
        str(clocks.stream_clock(holders['w_pos'])), ". on (sel in {'A', 'B'})"
    )
    self.assertEqual(
        str(clocks.stream_clock(holders['w_neg'])), ". on (sel != 'A')"
    )
    self.assertEqual(clocks.stream_clock(holders['sw']), BASE_CLOCK)
    # id_sw was folded to polymorphic static `sel` and stream_clock resolves
    # the alias to `sel`'s polymorphic clock; unif_sw folded to dynamic `v_dyn`.
    self.assertTrue(clocks.stream_clock(holders['id_sw']).is_polymorphic)
    self.assertEqual(
        clocks.stream_clock(holders['id_sw']),
        clocks.stream_clock(g.resolve_stream(holders['id_sw'])),
    )
    self.assertEqual(g.resolve_stream(holders['id_sw']).name, 'sel')
    self.assertEqual(clocks.stream_clock(holders['unif_sw']), BASE_CLOCK)
    self.assertEqual(g.resolve_stream(holders['unif_sw']).name, 'v_dyn')

    def bad_switch():
      c = Stream([True, False, True, False], name='c')
      x = counter(start=1, name='x')
      return switch(c, when(c, x), x)

    bad_g = build_graph(bad_switch, semantics=Semantics.SYNC)
    with self.assertRaisesRegex(ClockError, 'incompatible input clocks'):
      check_clocks(bad_g)

  def test_subclock_reset_fby_inside_match_arm(self):
    holders: dict[str, Stream] = {}

    def prog():
      arm_c = Stream([True, True, False, True, True, True], name='arm_c')
      sub_c = Stream([True, False, True, True, False], name='sub_c')
      rst = Stream([True, False, False, True, False], name='rst')
      x = Stream([1, 2, 4, 5, 6], name='x')
      with match(arm_c) as m:
        with case(True):
          x_sub = when(sub_c, x)
          rst_sub = when(sub_c, rst)
          with reset(rst_sub):
            acc: Stream
            acc = fby(0, lambda: acc) + x_sub
          holders['acc'] = acc
          m.out = merge(sub_c, acc, 0)
        with case(False):
          m.out = -1
      return m.out

    g = build_graph(prog, semantics=Semantics.SYNC)
    clocks = check_clocks(g)
    self.assertEqual(
        str(clocks.stream_clock(holders['acc'])), '. on arm_c on sub_c'
    )
    self.assertEqual(
        Scheduler.run_graph(g, horizon=6, clocks=clocks),
        [1, 0, -1, 5, 5, 0],
    )

  def test_lower_resets_handles_none_and_float_outer_resets(self):
    def prog():
      c = Stream([True, False, True, True, True, True])
      r_raw = Stream([False, True, False, False, False, False])
      r_pre = pre(r_raw)  # [None, False, True, False, False, False]
      b = Stream([False, True, False, False, False, False])
      r_cur = current(b, [False, True, False, False, False, False])
      r_float = Stream([0.0, 1.5, 0.0, 0.0, 0.0, 0.0])
      with reset(r_pre):
        with match(c) as m_pre:
          with case(True):
            m_pre.out = counter(start=10)
          with case(False):
            m_pre.out = -1
      with reset(r_cur):
        with match(c) as m_cur:
          with case(True):
            m_cur.out = counter(start=1)
          with case(False):
            m_cur.out = 0
      f_vals = Stream([1, 2, 3, 4, 5, 6])
      with reset(r_float):
        f_sub = fby(99, when(c, f_vals))
      with match(Stream([True, True, True, True, True, True])) as outer:
        with case(True):
          prev_r = last(r_raw)
          with reset(prev_r):
            with match(c) as m_last:
              with case(True):
                m_last.out = counter(start=5)
              with case(False):
                m_last.out = -2
          outer.out = m_last.out
        with case(False):
          outer.out = 0
      return m_pre.out, m_cur.out, f_sub, outer.out

    sync_res = Scheduler.run(prog, horizon=6, semantics=Semantics.SYNC)
    kahn_res = Scheduler.run(prog, horizon=6, semantics=Semantics.KAHN)
    self.assertEqual(sync_res[0], [10, -1, 10, 11, 12, 13])
    self.assertEqual(sync_res[1], [1, 0, 1, 1, 1, 1])
    self.assertEqual(sync_res[2], [99, Absent, 99, 3, 4, 5])
    self.assertEqual(sync_res[3], [5, -2, 5, 6, 7, 8])
    for s_tr, k_tr in zip(sync_res, kahn_res):
      self._assert_sync_kahn_elision(s_tr, k_tr)

  def test_lower_resets_clock_step_edge_cases(self):
    class _Flag:

      def __init__(self, flag: bool):
        self._flag = flag

      def __bool__(self) -> bool:
        return self._flag

    @psyre_comb
    def to_flag(v: int) -> _Flag:
      return _Flag(bool(v))

    # 1. Non-boolean-proved truthy selectors (%, int stream, pre(bool), custom
    #    __bool__) preserve truthiness semantics across lower_resets, distinct
    #    from exact tag-matching when(..., values=[True]).
    def truthy_selectors():
      x = Stream([10, 20, 30, 40, 50, 60])
      r = Stream([False, True, False, False, False, False])
      mod_sel = counter() % 2  # [0, 1, 0, 1, 0, 1]
      int_sel = Stream([1, 0, 1, 0, 1, 0])
      b_pre = pre(Stream([True, False, True, False, True, False]))
      flag_sel = to_flag(int_sel)
      mixed = Stream([1, 0, True, False, 1, True])
      with reset(r):
        f_mod = fby(0, when(mod_sel, x))
        f_int = fby(0, when(int_sel, x))
        f_wn_pre = fby(0, whennot(b_pre, x))
        f_flag = fby(0, when(flag_sel, x))
        f_tag_true = fby(0, when(mixed, x, values=[True]))
      return f_mod, f_int, f_wn_pre, f_flag, f_tag_true

    for sem in (Semantics.SYNC, Semantics.KAHN):
      g = build_graph(truthy_selectors, semantics=sem)
      clks = check_clocks(g)
      lowered = lower_resets(g, clks)
      check_causality(lowered)
      l_clks = check_clocks(lowered)
      self.assertIs(lower_resets(lowered, l_clks), lowered)
    sync_t = Scheduler.run(
        truthy_selectors, horizon=6, semantics=Semantics.SYNC
    )
    kahn_t = Scheduler.run(
        truthy_selectors, horizon=6, semantics=Semantics.KAHN
    )
    # mod_sel=[0,1,0,1,0,1], reset at t=1 (active!): ticks 1,3,5 -> [0, 20, 40]
    self.assertEqual(kahn_t[0], [0, 20, 40])
    # int_sel=[1,0,1,0,1,0], reset at t=1 (inactive -> held to t=2): [0, 0, 30]
    self.assertEqual(kahn_t[1], [0, 0, 30])
    # b_pre=[None,True,False,True,False,True]; whennot active at t=0,2,4:
    # reset at t=1 (inactive -> latched to t=2!): [0, 0, 30]
    self.assertEqual(kahn_t[2], [0, 0, 30])
    self.assertEqual(kahn_t[3], [0, 0, 30])
    # mixed=[1,0,True,False,1,True]; exact values=[True] active at t=2,5:
    self.assertEqual(kahn_t[4], [0, 30])
    for s_tr, k_tr in zip(sync_t, kahn_t):
      self._assert_sync_kahn_elision(s_tr, k_tr)

    # Truthy when(s, x) and tag-exact when(s, x, values=[True]) have distinct
    # clocks when s is not statically proved boolean.
    def truthy_vs_tag_conflict():
      s = Stream([1, 0, True, False])
      x = Stream([10, 20, 30, 40])
      return when(s, x) + when(s, x, values=[True])

    with self.assertRaisesRegex(ClockError, 'incompatible input clocks'):
      check_clocks(
          build_graph(truthy_vs_tag_conflict, semantics=Semantics.SYNC)
      )

    # 2. Empty boolean subclocks: values=[] and unreachable default() after
    #    case(True) and case(False) on a bool stream.
    def empty_bool_subclocks():
      b = Stream([True, False, True, False])
      r = Stream([False, True, False, False])
      x = Stream([10, 20, 30, 40])
      with reset(r):
        w_empty = fby(0, when(b, x, values=[]))
        with match(b) as m:
          with case(True):
            m.out = 1
          with case(False):
            m.out = 2
          with default():
            m.out = counter(start=99)
      return w_empty, m.out

    for sem in (Semantics.SYNC, Semantics.KAHN):
      for opt in (False, True):
        g = build_graph(empty_bool_subclocks, semantics=sem, optimize=opt)
        clks = check_clocks(g)
        lowered = lower_resets(g, clks)
        check_causality(lowered)
        l_clks = check_clocks(lowered)
        res = Scheduler.run_graph(lowered, horizon=4, clocks=l_clks)
        if sem == Semantics.SYNC:
          self.assertEqual(res[0], [Absent, Absent, Absent, Absent])
        else:
          self.assertEqual(res[0], [])
        self.assertEqual(res[1], [1, 2, 1, 2])

    # 3. bool/int tag mixes (case(0, False) and default() after case(1)/True).
    def bool_int_tag_mix():
      s = Stream([0, False, 0, False, 1, True])
      r = Stream([False, True, False, False, False, False])
      s2 = Stream([2, 1, True, 2, 2, 1])
      with reset(r):
        with match(s) as m1:
          with case(0, False):
            m1.out = counter(start=10)
          with default():
            m1.out = -1
        with match(s2) as m2:
          with case(1):
            m2.out = -1
          with case(True):
            m2.out = -2
          with default():
            m2.out = counter(start=100)
      return m1.out, m2.out

    sync_m1, sync_m2 = Scheduler.run(
        bool_int_tag_mix, horizon=6, semantics=Semantics.SYNC
    )
    kahn_m1, kahn_m2 = Scheduler.run(
        bool_int_tag_mix, horizon=6, semantics=Semantics.KAHN
    )
    # m1: t=0(0)->10, t=1(False,r=True)->10, t=2(0)->11, t=3(False)->12
    self.assertEqual(sync_m1, [10, 10, 11, 12, -1, -1])
    self.assertEqual(kahn_m1, sync_m1)
    # m2: t=0(2)->100, t=1(1,r=True latched), t=2(True,still latched),
    # t=3(2)->100 (reset consumed!), t=4(2)->101, t=5(1)->-1
    self.assertEqual(sync_m2, [100, -1, -2, 100, 101, -1])
    self.assertEqual(kahn_m2, sync_m2)

    # 4. Nested 1-periodic constant selectors under an outer subclock.
    def nested_const_selector():
      c_outer = Stream([True, False, True, False])
      r = Stream([False, True, False, False])
      with reset(r):
        with match(c_outer) as m:
          with case(True):
            with match(Stream(True)) as inner_b:
              with case(True):
                inner_b.out = 1
              with case(False):
                inner_b.out = counter(start=50)
            with match(Stream(0)) as inner_i:
              with case(1):
                inner_i.out = counter(start=70)
              with default():
                inner_i.out = 2
            m.out = inner_b.out + inner_i.out
          with case(False):
            m.out = 0
      return m.out

    for sem in (Semantics.SYNC, Semantics.KAHN):
      for opt in (False, True):
        g = build_graph(nested_const_selector, semantics=sem, optimize=opt)
        lowered, l_clks = self._assert_lower_resets_idempotent(g)
        out = Scheduler.run_graph(lowered, horizon=4, clocks=l_clks)
        self.assertFalse(Scheduler.deadlocked)
        self.assertEqual(out, [3, 0, 3, 0])

  def test_lower_resets_direct_kahn_empty_nested_subclock_no_deadlock(self):
    def nested_never_active():
      c1 = Stream(False)
      c2 = Stream([True, False, True, False])
      c3 = Stream(True)
      r = Stream.once(1)
      with reset(r):
        with match(c1) as m1:
          with case(True):
            with match(c2) as m2:
              with case(True):
                with match(c3) as m3:
                  with case(True):
                    m3.out = counter(start=1)
                  with case(False):
                    m3.out = 0
                m2.out = m3.out
              with case(False):
                m2.out = 0
            m1.out = m2.out
          with case(False):
            m1.out = -1
      return m1.out

    g = build_graph(nested_never_active, semantics=Semantics.KAHN)
    clks = check_clocks(g)
    self.assertEqual(
        Scheduler.run_graph(g, horizon=4, clocks=clks), [-1, -1, -1, -1]
    )
    self.assertFalse(Scheduler.deadlocked)

    lowered = lower_resets(g, clks)
    l_clks = check_clocks(lowered)
    self.assertEqual(
        Scheduler.run_graph(lowered, horizon=4, clocks=l_clks),
        [-1, -1, -1, -1],
    )
    self.assertFalse(Scheduler.deadlocked)

  def test_fixed_point_clock_inference_delayed_merge_buffer_and_chains(self):
    def merge_delayed_arms():
      c = Stream([True, False, True, False, True])
      x = Stream([10, 20, 30, 40, 50])
      w = when(c, x)
      with reset(False):
        p = pre(w)
        f = fby(0, w)
      return (
          merge(c, p, whennot(c, x)),
          merge(c, f, whennot(c, x)),
      )

    def buffer_of_pre():
      c = Stream([True, False, True, False, True])
      o = Stream([False, True, False, True, False])
      x = Stream([10, 20, 30, 40, 50])
      with reset(False):
        p = pre(when(c, x))
      return buffer(p, c, o, size=2)

    def multi_hop_forward_fby():
      k = Stream.periodic([], [True, False])
      x = when(k, counter(1))
      with reset(False):
        a = fby(0, lambda: b)  # pylint: disable=undefined-variable
        b = fby(0, lambda: c_hop)  # pylint: disable=undefined-variable
        c_hop = fby(0, x)
        d = fby(0, lambda: c_hop + 1)
      return a, b, c_hop, d

    def delayed_plus_same_subclock():
      c = Stream([True, False, True, False])
      x = Stream([1, 2, 3, 4])
      w = when(c, x)
      with reset(False):
        delayed = fby(0, w)
      return delayed + w

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(case='merge_delayed', semantics=sem):
        p_out, f_out = Scheduler.run(
            merge_delayed_arms, horizon=5, semantics=sem
        )
        self.assertEqual(p_out, [None, 20, 10, 40, 30])
        self.assertEqual(f_out, [0, 20, 10, 40, 30])
      with self.subTest(case='buffer_of_pre', semantics=sem):
        g_buf = build_graph(buffer_of_pre, semantics=sem)
        check_clocks(g_buf)
      with self.subTest(case='multi_hop', semantics=sem):
        ra, rb, rc, rd = Scheduler.run(
            multi_hop_forward_fby, horizon=6, semantics=sem
        )
        if sem == Semantics.KAHN:
          for tr in (ra, rb, rc, rd):
            self._assert_no_sentinels(tr)
        self.assertEqual(present_values(rc), [0, 1, 3])
        self.assertEqual(present_values(rb), [0, 0, 1])
        self.assertEqual(present_values(ra), [0, 0, 0])
        self.assertEqual(present_values(rd), [0, 1, 2])
      with self.subTest(case='delayed_plus', semantics=sem):
        dp = Scheduler.run(delayed_plus_same_subclock, horizon=4, semantics=sem)
        if sem == Semantics.KAHN:
          self._assert_no_sentinels(dp)
        self.assertEqual(present_values(dp), [1, 4])

    def merge_delayed_selector_external_arms():
      outer = Stream([True, False, True, False])
      inner = when(outer, Stream([True, False, False, True]))
      with reset(False):
        sel = fby(True, inner)
      t_arm = Stream([10, 20])
      f_arm = Stream([30, 40])
      return merge(sel, t_arm, f_arm)

    check_clocks(
        build_graph(merge_delayed_selector_external_arms, optimize=False)
    )

    # Cyclic (and sampled-wrapped) indicator merges must terminate without
    # `RecursionError` during clock checking.
    def cyclic_indicator_merge():
      m = merge(lambda: m, True, False)  # pylint: disable=undefined-variable
      return when(m, counter(0))

    def cyclic_wrapped_indicator():
      c = Stream(True)
      m = merge(when(c, lambda: m), True, False)  # pylint: disable=undefined-variable
      return when(m, c)

    check_clocks(build_graph(cyclic_indicator_merge, optimize=False))
    try:
      check_clocks(build_graph(cyclic_wrapped_indicator, optimize=False))
    except ClockError:
      pass

  def test_cse_preserves_distinct_one_periodic_when_selector_clocks(self):
    def two_const_matches():
      x = counter(0)
      with match('A') as m1:
        with case('A'):
          m1.o = x + 1
        with default():
          m1.o = 0
      with match('A') as m2:
        with case('A'):
          m2.o = x + 2
        with default():
          m2.o = 0
      return m1.o, m2.o

    def two_false_whens():
      x = counter(0)
      y = counter(10)
      c1 = Stream(False)
      c2 = Stream(False)
      return when(c1, x), when(c2, x) + when(c2, y)

    def two_never_tagged_whens():
      x = counter(0)
      y = counter(10)
      t1 = Stream('A')
      t2 = Stream('A')
      return (
          when(t1, x, values='B'),
          when(t2, x, values='B') + when(t2, y, values='B'),
      )

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(case='const_matches', semantics=sem):
        g_const = build_graph(two_const_matches, semantics=sem, optimize=True)
        # Constant tagged match('A') collapses `when` and `merge` completely.
        self.assertEmpty(
            [n for n in g_const.nodes if n.op in {'when', 'merge'}]
        )
        o1, o2 = Scheduler.run(
            two_const_matches, horizon=3, semantics=sem, optimize=True
        )
        self.assertEqual(o1, [1, 2, 3])
        self.assertEqual(o2, [2, 3, 4])
    g_false = build_graph(two_false_whens, optimize=True)
    check_clocks(g_false)
    g_tagged = build_graph(two_never_tagged_whens, optimize=True)
    check_clocks(g_tagged)

    # Instantaneous self-sampling cycle must fail clock convergence cleanly.
    def self_sampling_when():
      x = when(lambda: x, 0)  # pylint: disable=undefined-variable
      return x

    g_self = build_graph(self_sampling_when, optimize=False)
    with self.assertRaisesRegex(ClockError, 'did not converge'):
      check_clocks(g_self)

  def test_static_streams_aligned_to_consumer_clock_and_scope(self):
    # 1. Static periodic streams are uniformly clock-polymorphic (`∀ 'a. 'a`):
    #    each distinct `(stream, clock)` port gets an independent dense copy.
    def merge_and_buffer_prog():
      c = Stream([True, False, True, False, True, False, True], name='c')
      t_arm = Stream.periodic([10], [20, 30], name='t_arm')
      f_arm = Stream.periodic([100], [200, 300], name='f_arm')
      buf_in = Stream.periodic([1], [2, 3], name='buf_in')
      m = merge(c, t_arm, f_arm)
      # Sharing the same static stream across BOTH merge arms gives each arm
      # its own dense cursor starting at index 0.
      m_shared = merge(c, t_arm, t_arm)
      b = buffer(buf_in, c, c, size=1)
      # Consuming `t_arm` simultaneously on BASE_CLOCK (`.`) and on `. on c`
      # instantiates it independently at each clock.
      return m, m_shared, b, t_arm + 1

    s_g = build_graph(merge_and_buffer_prog, semantics=Semantics.SYNC)
    s_clks = check_clocks(s_g)
    named_s_map = s_g.resolve_named_streams(('c', 't_arm', 'f_arm', 'buf_in'))
    for nm in ('c', 't_arm', 'f_arm', 'buf_in'):
      self.assertTrue(s_clks.stream_clock(named_s_map[nm]).is_polymorphic)
    m_node = s_g.producer(s_g.output[0])
    assert m_node is not None
    self.assertEqual(str(s_clks.input_clock(m_node, 1)), '. on c')
    self.assertEqual(str(s_clks.input_clock(m_node, 2)), '. on not c')
    b_node = s_g.producer(s_g.output[2])
    assert b_node is not None
    self.assertEqual(str(s_clks.input_clock(b_node, 0)), '. on c')
    lowered = lower_resets(s_g, s_clks)
    l_clks = check_clocks(lowered)
    self.assertIs(lower_resets(lowered, l_clks), lowered)
    sm, sm_shared, sb, st_plus = Scheduler.run_graph(
        s_g, horizon=7, clocks=s_clks
    )
    km, km_shared, kb, kt_plus = Scheduler.run(
        merge_and_buffer_prog, horizon=7, semantics=Semantics.KAHN
    )
    self.assertEqual(sm, [10, 100, 20, 200, 30, 300, 20])
    self.assertEqual(km, sm)
    self.assertEqual(sm_shared, [10, 10, 20, 20, 30, 30, 20])
    self.assertEqual(km_shared, sm_shared)
    self.assertEqual(sb, [1, Absent, 2, Absent, 3, Absent, 2])
    self.assertEqual(kb, [1, 2, 3, 2])
    self.assertEqual(st_plus, [11, 21, 31, 21, 31, 21, 31])
    self.assertEqual(kt_plus, st_plus)
    # Direct traces of polymorphic static streams return their unclocked dense
    # sequence `prefix + period*ω`.
    named_s = Scheduler.run_graph(
        s_g, horizon=5, streams=('t_arm', 'f_arm', 'buf_in'), clocks=s_clks
    )
    self.assertEqual(named_s['t_arm'], [10, 20, 30, 20, 30])
    self.assertEqual(named_s['f_arm'], [100, 200, 300, 200, 300])
    self.assertEqual(named_s['buf_in'], [1, 2, 3, 2, 3])

    # 2. Subclock binary ops (either operand order) instantiate static stream
    #    at their consumer's clock — including sharing ONE static stream across
    #    opposite subclocks (`when(c, ...)` and `whennot(c, ...)`).
    def subclock_add():
      c = Stream([True, False, True, False, True])
      w = when(c, Stream([1, 2, 3, 4, 5]))
      wn = whennot(c, Stream([1, 2, 3, 4, 5]))
      s_rhs = Stream([10, 20, 30], name='s_rhs')
      s_lhs = Stream.periodic([100], [200, 300], name='s_lhs')
      return w + s_rhs, s_lhs + w, wn + s_rhs

    sr, sl, swn = Scheduler.run(
        subclock_add, horizon=5, semantics=Semantics.SYNC
    )
    kr, kl, kwn = Scheduler.run(
        subclock_add, horizon=5, semantics=Semantics.KAHN
    )
    self.assertEqual(sr, [11, Absent, 23, Absent, 35])
    self.assertEqual(sl, [101, Absent, 203, Absent, 305])
    self.assertEqual(swn, [Absent, 12, Absent, 24, Absent])
    self._assert_sync_kahn_elision(sr, kr)
    self._assert_sync_kahn_elision(sl, kl)
    self._assert_sync_kahn_elision(swn, kwn)

    # Dynamic unbound external `Stream()` shared across conflicting subclocks
    # still fails (monomorphic!).
    def shared_dyn_conflict():
      c = Stream([True, False, True, False])
      shared = Stream()
      return when(c, counter(0)) + shared, whennot(c, counter(0)) + shared

    with self.assertRaisesRegex(ClockError, 'incompatible input clocks'):
      check_clocks(build_graph(shared_dyn_conflict, semantics=Semantics.SYNC))

    # 3. Nested match() reusing static selectors, data, and intermediate reset,
    #    contrasting pre-built polymorphic static `shared_s` vs anchored wire.
    shared_s = Stream([1, 2, 3, 4, 5, 6], name='shared_s')

    def nested_match_scopes():
      c1 = Stream([True, False, True, False, True, True], name='c1')
      c1_opt = ifthenelse(Stream(True), c1, Stream(False))
      outer_dyn = when(Stream(True), [1, 2, 3, 4, 5, 6], name='outer_dyn')
      with match(c1_opt) as m1:
        with case(True):
          c2 = Stream([True, False, True, False], name='c2')
          r_mid = Stream([False, True, False, False], name='r_mid')
          with reset(r_mid):
            with match(c2) as m2:
              with case(True):
                leaf = Stream([10, 20], name='leaf')
                m2.val = leaf
                m2.cnt = counter(start=50)
              with case(False):
                m2.val = -5
                m2.cnt = -1
          m1.val = m2.val
          m1.cnt = m2.cnt
          m1.from_dyn = outer_dyn
          m1.from_poly = shared_s
        with case(False):
          m1.val = 0
          m1.cnt = 0
          m1.from_dyn = -outer_dyn
          m1.from_poly = -shared_s
      return m1.val, m1.cnt, m1.from_dyn, m1.from_poly

    for opt in (False, True):
      with self.subTest(case='nested_match', optimize=opt):
        sv, sc, sd, sp = Scheduler.run(
            nested_match_scopes,
            horizon=6,
            semantics=Semantics.SYNC,
            optimize=opt,
        )
        kv, kc, kd, kp = Scheduler.run(
            nested_match_scopes,
            horizon=6,
            semantics=Semantics.KAHN,
            optimize=opt,
        )
        # `leaf = Stream([10, 20])` was constructed inside `with reset(r_mid):`,
        # so the latched `r_mid` at t=2 restarts both `leaf` and `m2.cnt` at
        # t=4 (`10` and `50`, respectively).
        self.assertEqual(sv, [10, 0, -5, 0, 10, -5])
        self.assertEqual(kv, sv)
        self.assertEqual(sc, [50, 0, -1, 0, 50, -1])
        self.assertEqual(kc, sc)
        # `outer_dyn` is a concrete BASE_CLOCK stream, downsampled by both arms.
        self.assertEqual(sd, [1, -2, 3, -4, 5, 6])
        self.assertEqual(kd, sd)
        # `shared_s` is polymorphic: each arm consumes [1, 2, ...] from index 0.
        self.assertEqual(sp, [1, -1, 2, -2, 3, 4])
        self.assertEqual(kp, sp)

    # 4. Nested match(c) reusing the EXACT SAME static Stream `c` as both outer
    #    and inner selector (each scope consumes `c` independently!).
    def nested_same_static_selector():
      c = Stream([True, False, True, False, True, True], name='c')
      with match(c) as m1:
        with case(True):
          with match(c) as m2:
            with case(True):
              m2.out = counter(start=1)
            with case(False):
              m2.out = -1
          m1.out = m2.out
        with case(False):
          m1.out = 0
      return m1.out

    for sem in (Semantics.SYNC, Semantics.KAHN):
      g_same = build_graph(nested_same_static_selector, semantics=sem)
      c_same = check_clocks(g_same)
      self._assert_lower_resets_idempotent(g_same, c_same)
      # Outer c active ticks: t=0, 2, 4, 5 -> inner c sees [T, F, T, F].
      self.assertEqual(
          Scheduler.run_graph(g_same, horizon=6, clocks=c_same),
          [1, 0, -1, 0, 2, -1],
      )

    # 5. Constant `reset(False)` and `reset(True)` inside/outside match() behave
    #    uniformly with general static resets, and `lower_resets` is idempotent.
    def const_reset_subclock():
      c = Stream([True, False, True, False, True])
      with reset(False):
        r_f = fby(0, when(c, [1, 0, 2, 0, 3]))
      with reset(True):
        r_t = fby(99, when(c, [1, 0, 2, 0, 3]))
      return r_f, r_t

    for sem in (Semantics.SYNC, Semantics.KAHN):
      g_cr = build_graph(const_reset_subclock, semantics=sem)
      cl_cr = check_clocks(g_cr)
      self._assert_lower_resets_idempotent(g_cr, cl_cr)
      rf, rt = Scheduler.run_graph(g_cr, horizon=5, clocks=cl_cr)
      if sem == Semantics.SYNC:
        self.assertEqual(rf, [0, Absent, 1, Absent, 2])
        self.assertEqual(rt, [99, Absent, 99, Absent, 99])
      else:
        self.assertEqual(rf, [0, 1, 2])
        self.assertEqual(rt, [99, 99, 99])

    # 6. Automaton arm-local static outputs & weak until() static conditions
    #    reset to index 0 when re-entered with `reset=True`, and resume from
    #    their held cursor when re-entered with `reset=False` (history).
    def make_automaton_statics(reset_on_exit: bool):
      def prog():
        with automaton(init='A') as fsm:
          with state('A'):
            until([False, True, False], 'B', reset=reset_on_exit)
            fsm.out = Stream([10, 20, 30])
          with state('B'):
            until([False, True], 'A', reset=reset_on_exit)
            fsm.out = Stream([100, 200, 300])
        return fsm.out

      return prog

    for sem in (Semantics.SYNC, Semantics.KAHN):
      for reset_on_exit, expected in (
          (True, [10, 20, 100, 200, 10, 20]),
          (False, [10, 20, 100, 200, 30, 30]),
      ):
        self.assertEqual(
            Scheduler.run(
                make_automaton_statics(reset_on_exit),
                horizon=6,
                semantics=sem,
            ),
            expected,
        )

    # 7. Never-active arm expression over static stream emits `Absent` in SYNC
    #    and `[]` in KAHN (both 1-periodic and multi-element `Stream`).
    def dead_arm_static():
      with match(Stream(False)) as m:
        with case(True):
          dead_expr = Stream([10, 20]) + 1
          dead_expr.name = 'dead_expr'
          const_expr = Stream(5) + Stream(7)
          const_expr.name = 'const_expr'
          m.out = dead_expr
        with case(False):
          m.out = 0
      return m.out

    for opt in (False, True):
      res_dead = Scheduler.run(
          dead_arm_static,
          horizon=3,
          semantics=Semantics.SYNC,
          streams=('dead_expr', 'const_expr'),
          optimize=opt,
      )
      self.assertEqual(res_dead['dead_expr'], [Absent, Absent, Absent])
      self.assertEqual(res_dead['const_expr'], [Absent, Absent, Absent])
      res_dead_k = Scheduler.run(
          dead_arm_static,
          horizon=3,
          semantics=Semantics.KAHN,
          streams=('dead_expr', 'const_expr'),
          optimize=opt,
      )
      self.assertFalse(Scheduler.deadlocked)
      self._assert_sync_kahn_elision(
          res_dead['dead_expr'], res_dead_k['dead_expr']
      )
      self._assert_sync_kahn_elision(
          res_dead['const_expr'], res_dead_k['const_expr']
      )

    # 8b. Subclocked all-static stateless ops, static-selector `when`/`whennot`/
    #     `merge`, and `post(when(...))` are finite-horizon bounded in BOTH SYNC
    #     and KAHN, including `Bottom`-padded tails and `lower_resets`
    #     idempotency.
    def subclock_all_static_bounds():
      c = Stream([True, False, True, False], name='c_sub')
      xs = Stream([10, 20, 30, 40], name='xs_sub')
      ys = Stream([1, 2, 3, 4], name='ys_sub')
      with match(c) as m:
        with case(True):
          s_add = xs + ys
          s_add.name = 's_add'
          s_ite = ifthenelse(c, xs, ys, name='s_ite')
          s_w = when(c, xs, name='s_w')
          whennot(c, xs, name='s_wn')
          post(s_w, name='s_pw')
          post(xs, name='s_px')
          merge(c, xs, ys, name='s_ms')
          m.top = s_add + s_ite
        with case(False):
          m.top = -1
      return m.top

    bound_names = ('s_add', 's_ite', 's_w', 's_wn', 's_pw', 's_px', 's_ms')
    for opt in (False, True):
      for sem in (Semantics.SYNC, Semantics.KAHN):
        bg = build_graph(
            subclock_all_static_bounds,
            semantics=sem,
            optimize=opt,
            extra_roots=bound_names,
        )
        b_clks = check_clocks(bg)
        b_low = lower_resets(bg, b_clks)
        check_causality(b_low)
        self.assertIs(lower_resets(b_low, check_clocks(b_low)), b_low)
      bs = Scheduler.run(
          subclock_all_static_bounds,
          horizon=4,
          semantics=Semantics.SYNC,
          streams=bound_names,
          optimize=opt,
      )
      self.assertFalse(Scheduler.deadlocked)
      bk = Scheduler.run(
          subclock_all_static_bounds,
          horizon=4,
          semantics=Semantics.KAHN,
          streams=bound_names,
          optimize=opt,
      )
      self.assertFalse(Scheduler.deadlocked)
      self.assertEqual(bs['s_add'], [11, Absent, 22, Absent])
      self.assertEqual(bs['s_w'], [10, Absent, Absent, Absent])
      self.assertEqual(bs['s_wn'], [Absent, Absent, 20, Absent])
      self.assertEqual(bs['s_pw'], [Bottom, Bottom, Bottom, Bottom])
      self.assertEqual(bs['s_px'], [20, Absent, Bottom, Bottom])
      self.assertEqual(bs['s_ms'], [10, Absent, 1, Absent])
      for key in bound_names:
        self._assert_sync_kahn_elision(bs[key], bk[key])

    # 9. Zero-input stateless node (`Stream.random()`) inside `match` gets
    #    sampled onto the arm subclock and stays `Absent` on inactive ticks.
    def random_in_match():
      c = Stream([True, False, True, False])
      with match(c) as m:
        with case(True):
          r_val = Stream.random() + 0.0
          r_val.name = 'r_val'
          m.out = r_val
        with case(False):
          m.out = -1.0
      return m.out

    r_tr = Scheduler.run(
        random_in_match,
        horizon=4,
        semantics=Semantics.SYNC,
        streams=('r_val',),
    )['r_val']
    self.assertIsNot(r_tr[0], Absent)
    self.assertIs(r_tr[1], Absent)
    self.assertIsNot(r_tr[2], Absent)
    self.assertIs(r_tr[3], Absent)

    # 10. Subclocked polymorphic statics in `merge` and `buffer` preserve dense
    #     static elements under both `SYNC` and `KAHN` clock lowering.
    def poly_merge_buffer():
      c = Stream([True, False, True, False, True])
      return (
          merge(c, [10, 20, 30], [100, 200]),
          buffer([1, 2, 3], c, c, size=1),
      )

    um, ub = Scheduler.run(
        poly_merge_buffer,
        horizon=5,
        semantics=Semantics.SYNC,
    )
    self.assertEqual(um, [10, 100, 20, 200, 30])
    self.assertEqual(ub, [1, Absent, 2, Absent, 3])
    km, kb = Scheduler.run(
        poly_merge_buffer,
        horizon=5,
        semantics=Semantics.KAHN,
        clock_calculus=False,
    )
    self.assertEqual(km, [10, 100, 20, 200, 30])
    self.assertEqual(kb, [1, 2, 3])

  def test_lower_resets_idempotent_on_subclocked_buffer_predicates(self):
    def shared_op_merge():
      c = Stream([True, False, True, True, False, True])
      op = Stream([True, False, True, True])
      with match(c) as m:
        with case(True):
          b = buffer(Stream([1, 2, 3, 4]), Stream(True), op, size=4)
          m.out = merge(op, b, 0)
        with case(False):
          m.out = -1
      return m.out

    def shared_ip_when():
      c = Stream([True, False, True, True, False, True])
      ip = Stream([True, False, True, True])
      op = Stream([False, True, True, True])
      x = counter(10)
      with match(c) as m:
        with case(True):
          w = when(ip, x)
          b = buffer(w, ip, op, size=4)
          m.out = merge(op, b, 0)
        with case(False):
          m.out = -1
      return m.out

    def shared_false_pred_buffer():
      c = Stream([True, False, True, False])
      f_pred = Stream(False)
      with match(c) as m:
        with case(True):
          b = buffer(when(f_pred, Stream([1, 2])), f_pred, f_pred, size=2)
          m.out = merge(f_pred, b, 7)
        with case(False):
          m.out = -1
      return m.out

    for prog in (shared_op_merge, shared_ip_when, shared_false_pred_buffer):
      sync_g = build_graph(prog, semantics=Semantics.SYNC)
      kahn_g = build_graph(prog, semantics=Semantics.KAHN)
      sync_res = Scheduler.run_graph(
          sync_g, horizon=6, clocks=check_clocks(sync_g)
      )
      kahn_res = Scheduler.run_graph(
          kahn_g, horizon=6, clocks=check_clocks(kahn_g)
      )
      self._assert_sync_kahn_elision(sync_res, kahn_res)
      for sem, g in ((Semantics.SYNC, sync_g), (Semantics.KAHN, kahn_g)):
        with self.subTest(prog=prog.__name__, semantics=sem):
          self._assert_lower_resets_idempotent(g)

  def test_run_graph_without_clocks_rejects_unlowered_subclocked_statics(self):
    def sub_add_static():
      c = Stream([True, False, True, False])
      return when(c, [1, 2, 3, 4]) + Stream([10, 20, 30])

    def match_all_static():
      c = Stream([True, False, True, False])
      xs = Stream([10, 20, 30])
      ys = Stream([1, 2, 3])
      with match(c) as m:
        with case(True):
          m.out = xs + ys
        with case(False):
          m.out = 0
      return m.out

    def dead_arm_static():
      with match(Stream(False)) as m:
        with case(True):
          m.out = Stream([10, 20]) + 1
        with case(False):
          m.out = 0
      return m.out

    def automaton_static_until():
      toggle = Stream([False, True, False, False, True, False])
      with automaton() as fsm:
        with state('UP'):
          until(toggle, 'DOWN')
          fsm.v = 1
        with state('DOWN'):
          until(toggle, 'UP')
          fsm.v = 2
      return fsm.v

    with self.assertRaisesRegex(ValueError, 'mandatory under Semantics.SYNC'):
      Scheduler.run(
          sub_add_static,
          horizon=4,
          semantics=Semantics.SYNC,
          clock_calculus=False,
      )
    sync_sub_g = build_graph(sub_add_static, semantics=Semantics.SYNC)
    # Under SYNC, `run_graph(..., clocks=None)` automatically checks and lowers
    # clocks on well-clocked graphs.
    self.assertEqual(
        Scheduler.run_graph(sync_sub_g, horizon=4),
        [11, Absent, 23, Absent],
    )
    self.assertEqual(
        Scheduler.run_graph(
            sync_sub_g, horizon=4, clocks=check_clocks(sync_sub_g)
        ),
        [11, Absent, 23, Absent],
    )
    other_g = build_graph(lambda: counter(0), semantics=Semantics.SYNC)
    with self.assertRaisesRegex(ValueError, 'different Graph'):
      Scheduler.run_graph(sync_sub_g, horizon=4, clocks=check_clocks(other_g))
    with self.assertRaisesRegex(ValueError, 'different Graph'):
      lower_resets(sync_sub_g, check_clocks(other_g))
    with self.assertRaisesRegex(TypeError, 'ClockAnalysis'):
      lower_resets(sync_sub_g, 'not-clocks')  # type: ignore[arg-type]

    for prog in (match_all_static, automaton_static_until):
      with self.subTest(prog=prog.__name__):
        sync_g = build_graph(prog, semantics=Semantics.SYNC)
        with self.assertRaisesRegex(
            ValueError, 'mandatory under Semantics.SYNC'
        ):
          Scheduler.run(
              prog, horizon=4, semantics=Semantics.SYNC, clock_calculus=False
          )
        self.assertEqual(
            Scheduler.run_graph(sync_g, horizon=4),
            Scheduler.run(prog, horizon=4, semantics=Semantics.SYNC),
        )
        kahn_g = build_graph(prog, semantics=Semantics.KAHN)
        with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
          Scheduler.run_graph(kahn_g, horizon=4)
        with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
          Scheduler.run(
              prog, horizon=4, semantics=Semantics.KAHN, clock_calculus=False
          )

    # `dead_arm_static` retains its unlowered dead arm when optimize=False
    # (triggering the KAHN guard), while optimize=True folds it away.
    raw_dead_sync = build_graph(
        dead_arm_static, semantics=Semantics.SYNC, optimize=False
    )
    self.assertEqual(
        Scheduler.run_graph(raw_dead_sync, horizon=4), [0, 0, 0, 0]
    )
    raw_dead_kahn = build_graph(
        dead_arm_static, semantics=Semantics.KAHN, optimize=False
    )
    with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
      Scheduler.run_graph(raw_dead_kahn, horizon=4)
    with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
      Scheduler.run(
          dead_arm_static,
          horizon=4,
          semantics=Semantics.KAHN,
          optimize=False,
          clock_calculus=False,
      )
    for sem in (Semantics.SYNC, Semantics.KAHN):
      opt_dead = build_graph(dead_arm_static, semantics=sem, optimize=True)
      self.assertEmpty(opt_dead.nodes)
      self.assertEqual(
          Scheduler.run_graph(opt_dead, horizon=4), [0, 0, 0, 0]
      )

    # Clock-mismatched `nat + odd` is rejected under SYNC (both `run` and
    # `run_graph`), while unclocked KAHN runs exhibiting FIFO token drift.
    def relaxed_nat_plus_odd():
      nat = counter(0)
      odd = when((nat % 2) == 1, nat)
      return nat + odd

    bad_sync_g = build_graph(relaxed_nat_plus_odd, semantics=Semantics.SYNC)
    with self.assertRaisesRegex(ClockError, 'incompatible input clocks'):
      Scheduler.run_graph(bad_sync_g, horizon=4)
    self.assertEqual(
        Scheduler.run(
            relaxed_nat_plus_odd,
            horizon=4,
            semantics=Semantics.KAHN,
            clock_calculus=False,
        ),
        [1, 4],
    )

    # Already-lowered `lower_resets(g, clk)` runs cleanly without `clocks=`.
    def sub_reset_prog():
      c = Stream([True, False, True, False])
      with reset(Stream.once(1)):
        return fby(0, when(c, Stream([1, 2, 3, 4])))

    for sem in (Semantics.SYNC, Semantics.KAHN):
      raw_g = build_graph(sub_reset_prog, semantics=sem)
      lowered, _ = self._assert_lower_resets_idempotent(raw_g)
      expected = [0, Absent, 0, Absent] if sem == Semantics.SYNC else [0, 0]
      self.assertEqual(Scheduler.run_graph(lowered, horizon=4), expected)

    # Base-clock reset next to an unrelated `when()` runs in unclocked KAHN.
    def unrelated_when_and_base_reset():
      c = Stream([True, False, True, False])
      with reset(Stream.once(2)):
        base_cnt = counter(1)
      return base_cnt, when(c, counter(10))

    cnt_tr, w_tr = Scheduler.run(
        unrelated_when_and_base_reset,
        horizon=4,
        semantics=Semantics.KAHN,
        clock_calculus=False,
    )
    self.assertEqual(cnt_tr, [1, 2, 1, 2])
    self.assertEqual(w_tr, [10, 12])

  def test_fby_feedback_clock_inference_independent_of_operand_order(self):
    def make_prog(lhs_first: bool):
      def prog():
        c = Stream([True, False, True, True, False])
        x = Stream([1, 9, 2, 3, 9])
        w = when(c, x)
        with reset(False):
          acc = fby(
              0,
              lambda: (acc + w) if lhs_first else (w + acc),  # pylint: disable=undefined-variable
          )
        return acc

      return prog

    for sem in (Semantics.SYNC, Semantics.KAHN):
      for lhs_first in (True, False):
        with self.subTest(semantics=sem, lhs_first=lhs_first):
          res = Scheduler.run(make_prog(lhs_first), horizon=5, semantics=sem)
          expected = [0, Absent, 1, 3, Absent] if sem == Semantics.SYNC else [
              0, 1, 3
          ]
          self.assertEqual(res, expected)

    # Multi-hop subclocked feedback with delayed operand on LHS.
    def multihop_lhs():
      c = Stream([True, False, True, True])
      w = when(c, Stream([10, 99, 20, 30]))
      with reset(False):
        acc = fby(0, lambda: nxt)  # pylint: disable=undefined-variable
        scaled = acc + 1
        nxt = scaled + w
      return acc

    self.assertEqual(
        Scheduler.run(multihop_lhs, horizon=4, semantics=Semantics.SYNC),
        [0, Absent, 11, 32],
    )

    # True mismatch (`init` on base clock vs delayed on `. on c`) still fails.
    def invalid_base_init_fby():
      c = Stream([True, False, True, False])
      w = when(c, counter(1))
      with reset(False):
        acc = fby(counter(0), lambda: acc + w)  # pylint: disable=undefined-variable
      return acc

    with self.assertRaisesRegex(ClockError, 'incompatible input clocks'):
      check_clocks(build_graph(invalid_base_init_fby, semantics=Semantics.SYNC))

  def test_current_prefix_clock_contract_and_counter_tick_clocks(self):
    def valid_subclock_currents():
      nat = counter(0, name='nat')
      is_odd = (nat % 2) == 1
      is_odd.name = 'is_odd'
      odd = when(is_odd, nat, name='odd')
      gate = Stream([True, False, True, True, False, True], name='gate')
      deep = when(gate, odd, name='deep')
      tag = Stream(['A', 'B', 'A', 'C', 'A', 'B'], name='tag')
      tagged = when(tag, nat, values='A', name='tagged')
      neg_tagged = when(tag, nat, values='A', negated=True, name='neg_tag')
      empty_sub = when(tag, nat, values=(), name='empty_sub')
      buf = buffer(odd, is_odd, is_odd, size=4, name='buf')
      cur_same = current(is_odd, nat, default=0, name='cur_same')
      cur_sub = current(is_odd, odd, default=0, name='cur_sub')
      cur_true = current(True, odd, default=-1, name='cur_true')
      cur_deep = current(True, deep, default=-1, name='cur_deep')
      cur_tag = current(True, tagged, default=-1, name='cur_tag')
      cur_neg = current(True, neg_tagged, default=-1, name='cur_neg')
      cur_empty = current(True, empty_sub, default=-1, name='cur_empty')
      cur_buf = current(True, buf, default=-1, name='cur_buf')
      with match(is_odd) as m:
        with case(True):
          m.inner_cur = current(
              True, when(gate, nat, name='inner_gate_nat'), default=-9
          )
        with case(False):
          m.inner_cur = -1
      return {
          'cur_same': cur_same,
          'cur_sub': cur_sub,
          'cur_true': cur_true,
          'cur_deep': cur_deep,
          'cur_tag': cur_tag,
          'cur_neg': cur_neg,
          'cur_empty': cur_empty,
          'cur_buf': cur_buf,
          'inner_cur': m.inner_cur,
      }

    sync_g = build_graph(valid_subclock_currents, semantics=Semantics.SYNC)
    clks = check_clocks(sync_g)
    for key, s in sync_g.output.items():
      self.assertEqual(clks.stream_clock(s), BASE_CLOCK, msg=key)
    by_name = {n.name: n for n in sync_g.nodes}
    self.assertEqual(
        str(clks.input_clock(by_name['cur_same'], 1)), '.'
    )
    self.assertEqual(
        str(clks.input_clock(by_name['cur_sub'], 1)), '. on is_odd'
    )
    self.assertEqual(
        str(clks.input_clock(by_name['cur_deep'], 1)),
        '. on is_odd on gate',
    )
    self.assertEqual(
        str(clks.input_clock(by_name['cur_tag'], 1)),
        ". on (tag == 'A')",
    )
    self.assertEqual(
        str(clks.input_clock(by_name['cur_neg'], 1)),
        ". on (tag != 'A')",
    )
    self.assertEqual(
        str(clks.input_clock(by_name['cur_buf'], 1)), '. on is_odd'
    )
    # `rate_inputs()` of `current` excludes data input port 1 (`i_stream`).
    cur_sub_node = by_name['cur_sub']
    self.assertEqual(
        cur_sub_node.rate_inputs(),
        (cur_sub_node.inputs[0], cur_sub_node.inputs[2]),
    )

    # KAHN lowering of subclocked `current` is idempotent and clock-clean.
    kahn_g = build_graph(valid_subclock_currents, semantics=Semantics.KAHN)
    with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
      Scheduler.run_graph(kahn_g, horizon=6)
    with self.assertRaisesRegex(ClockError, 'requires clock analysis'):
      Scheduler.run(
          valid_subclock_currents,
          horizon=6,
          semantics=Semantics.KAHN,
          clock_calculus=False,
      )
    lowered, _ = self._assert_lower_resets_idempotent(kahn_g)
    self.assertEqual(
        Scheduler.run_graph(lowered, horizon=6),
        Scheduler.run_graph(sync_g, horizon=6, clocks=clks),
    )

    # Negative clock-contract cases must fail static check_clocks.
    def faster_input_into_subclock_current():
      c = Stream([True, False, True, False], name='c')
      b = when(c, True, name='b')
      fast = counter(0, name='fast')
      with reset(when(c, False)):
        return current(b, fast, default=-1)

    def sibling_subclock_current():
      a = Stream([True, False, True, False], name='a')
      b = Stream([False, True, False, True], name='b')
      x = counter(0, name='x')
      sub_b = when(b, x, name='sub_b')
      with reset(when(a, False)):
        return current(when(a, True), sub_b, default=-1)

    def mismatched_default_clock():
      cnt = counter(0, name='cnt')
      b = (cnt % 2) == 0
      return current(b, cnt, default=when(b, 0))

    def conflicting_external_into_current():
      c = Stream([True, False, True, False], name='c')
      ext = Stream(name='ext')
      with reset(when(c, False)):
        cur = current(when(c, True), ext, default=-1)
      return cur, ext + counter(0)

    def mismatched_counter_tick_clock():
      c = Stream([True, False, True, False], name='c')
      return counter(start=0, tick=when(c, True))

    for bad_fn, regex in (
        (faster_input_into_subclock_current, 'current input.*expected'),
        (sibling_subclock_current, 'current input.*expected'),
        (mismatched_default_clock, 'incompatible input clocks'),
        (conflicting_external_into_current, 'Clock error'),
        (mismatched_counter_tick_clock, 'stateful node executes on subclock'),
    ):
      with self.subTest(bad=bad_fn.__name__):
        g = build_graph(bad_fn, semantics=Semantics.SYNC)
        with self.assertRaisesRegex(ClockError, regex):
          check_clocks(g)

    def counter_tick_vs_start_mismatch():
      c = Stream([True, False, True, False], name='c')
      start_base = counter(0, name='start_base')
      with reset(when(c, False)):
        return counter(start=start_base, tick=when(c, True))

    with self.assertRaisesRegex(ClockError, 'incompatible input clocks'):
      check_clocks(
          build_graph(counter_tick_vs_start_mismatch, semantics=Semantics.SYNC)
      )

  def test_non_const_poly_static_not_unwrapped_across_strict_prefix(self):
    def merge_current_prog():
      outer = when(Stream(True), Stream([True, True, False, True, True, True]))
      s_static = Stream.periodic([], [True, False])
      flag = merge(s_static, Stream(True), Stream(False))
      with match(outer) as m:
        with case(True):
          m.y = current(Stream(True), when(flag, counter(10)), default=-1)
        with case(False):
          m.y = -99
      return m.y

    def nested_reset_prog(use_invert: bool):
      def prog():
        outer = when(
            Stream(True), Stream([True, False, True, True, False, True])
        )
        s_static = Stream.periodic([], [True, False])
        gate = ~s_static if use_invert else merge(s_static, True, False)
        r = when(
            Stream(True), Stream([False, True, False, False, False, False])
        )
        with reset(r):
          with match(outer) as m1:
            with case(True):
              with match(gate) as m2:
                with case(True):
                  m2.v = counter(0)
                with case(False):
                  m2.v = 99
              m1.y = m2.v
            with case(False):
              m1.y = -1
        return m1.y

      return prog

    for opt in (False, True):
      for sem in (Semantics.SYNC, Semantics.KAHN):
        with self.subTest(case='merge_current', opt=opt, sem=sem):
          self.assertEqual(
              Scheduler.run(
                  merge_current_prog, horizon=6, semantics=sem, optimize=opt
              ),
              [10, 10, -99, 10, 13, 13],
          )
        with self.subTest(case='merge_reset', opt=opt, sem=sem):
          self.assertEqual(
              Scheduler.run(
                  nested_reset_prog(False),
                  horizon=6,
                  semantics=sem,
                  optimize=opt,
              ),
              [0, -1, 0, 99, -1, 99],
          )
        with self.subTest(case='invert_reset', opt=opt, sem=sem):
          self.assertEqual(
              Scheduler.run(
                  nested_reset_prog(True),
                  horizon=6,
                  semantics=sem,
                  optimize=opt,
              ),
              [99, -1, 99, 0, -1, 1],
          )

    def alias_merge_prog():
      outer = when(Stream(True), Stream([True, False, True, True, False, True]))
      s_static = Stream.periodic([], [True, False])
      flag = merge(s_static, Stream(True), Stream(False))
      return when(when(outer, flag), when(outer, counter(0))) + when(
          s_static, when(outer, counter(10))
      )

    def alias_invert_prog():
      outer = when(Stream(True), Stream([True, False, True, True, False, True]))
      s_static = Stream.periodic([], [True, False])
      return when(when(outer, ~s_static), when(outer, counter(0))) + whennot(
          s_static, when(outer, counter(10))
      )

    for bad_fn in (alias_merge_prog, alias_invert_prog):
      for opt in (False, True):
        with self.subTest(bad=bad_fn.__name__, opt=opt):
          with self.assertRaisesRegex(ClockError, 'incompatible input clocks'):
            check_clocks(
                build_graph(bad_fn, semantics=Semantics.SYNC, optimize=opt)
            )

    def tautological_tagged_merge():
      outer = when(Stream(True), Stream([True, False, True, False]))
      s_static = Stream.periodic([], [True, False])
      flag = merge(s_static, {True: Stream(True), False: Stream(True)})
      return when(when(outer, flag), when(outer, counter(0))) + when(
          outer, counter(10)
      )

    self.assertEqual(
        Scheduler.run(tautological_tagged_merge, horizon=4, optimize=False),
        [10, Absent, 14, Absent],
    )

  def test_lower_resets_patches_synthetic_arm_scopes(self):
    def prog():
      outer = when(Stream(True), Stream([True, False, True, True, False, True]))
      s_mid = Stream.periodic([], [True, False])
      cur = current(
          Stream(True),
          lambda: sub,  # pylint: disable=undefined-variable
          default=-1,
      )
      mid = when(s_mid, when(outer, counter(10)))
      sub = when(mid == 10, mid)
      return cur

    gk = build_graph(prog, semantics=Semantics.KAHN, optimize=False)
    gl = lower_resets(gk, check_clocks(gk))
    for node in gl.nodes:
      for idx, scope in enumerate(node.arms):
        if idx > 0:
          self.assertFalse(gl.is_polymorphic_static(scope.cond))
    self.assertIsNotNone(check_clocks(gl))

  def test_disjoint_clocks_message_and_array_reset_guard(self):
    def plain_mismatch():
      c1 = when(Stream(True), Stream([True, False]))
      c2 = when(Stream(True), Stream([False, True]))
      return when(c1, counter(0)) + when(c2, counter(1))

    g = build_graph(plain_mismatch, semantics=Semantics.SYNC, optimize=False)
    with self.assertRaises(ClockError) as ctx:
      check_clocks(g)
    self.assertNotIn('enclosing activity clock', str(ctx.exception))

    class _FakeArray:
      shape = (1,)
      dtype = 'bool'

    def array_const_reset():
      c = Stream.periodic([], [True, False])
      with reset(Stream.constant(_FakeArray())):
        with match(c) as m:
          with case(True):
            m.y = counter(0)
          with case(False):
            m.y = -1
      return m.y

    ag = build_graph(array_const_reset, optimize=False)
    with self.assertRaisesRegex(TypeError, 'scalar truth value'):
      lower_resets(ag, check_clocks(ag))


if __name__ == '__main__':
  absltest.main()
