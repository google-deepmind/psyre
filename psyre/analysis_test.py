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
"""Unit tests for Psyre instantaneous causality analysis."""

import logging

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import (
    CausalityError,
    check_causality,
    find_instantaneous_cycle,
    topological_sort,
)
from psyre.automaton import (
    automaton,
    state,
    unless,
    until,
)
from psyre.graph import Graph
from psyre.match import (
    case,
    last,
    match,
)
from psyre.psyre import (
    Scheduler,
    Semantics,
    Stream,
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
)

# pylint: enable=g-importing-member,g-multiple-import


def _assert_valid_topological_order(
    testcase: absltest.TestCase, graph: Graph
) -> None:
  """Verifies that topological_sort(graph) respects instantaneous edges."""
  order = topological_sort(graph)
  testcase.assertLen(order, len(graph.nodes))
  testcase.assertCountEqual(order, graph.nodes)
  rank = {node: idx for idx, node in enumerate(order)}
  for node in graph.nodes:
    for pred in graph.predecessors(node, instantaneous_only=True):
      testcase.assertLess(rank[pred], rank[node])


def _assert_valid_instantaneous_cycle(
    testcase: absltest.TestCase,
    graph: Graph,
    cycle: tuple[object, ...],
) -> None:
  """Verifies that cycle is a non-empty elementary instantaneous loop."""
  testcase.assertNotEmpty(cycle)
  testcase.assertLen(set(cycle), len(cycle))
  nodes = list(graph.nodes)
  typed_cycle = [nodes[n.id] for n in cycle if hasattr(n, 'id')]
  testcase.assertLen(typed_cycle, len(cycle))
  testcase.assertEqual(typed_cycle[0].id, min(node.id for node in typed_cycle))
  for idx, src in enumerate(typed_cycle):
    dst = typed_cycle[(idx + 1) % len(typed_cycle)]
    testcase.assertIn(dst, graph.successors(src, instantaneous_only=True))


class AnalysisTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def test_causal_feedback_idioms_and_whole_program_flattening(self):
    @psyre_comb
    def add_two(v):
      return v + 2

    def pair_box(x, y):
      # In modular causality without inlining, calling pair_box(inp, lambda: a)
      # where (a, b) = pair_box(...) creates an apparent box-level cycle.
      # Whole-program Stage-1 graph construction flattens pair_box into two
      # independent nodes: inp -> a -> b.
      return x + 1, add_two(y)

    def prog():
      inp = Stream([1, 2, 3, 4])
      _ = (
          pre(inp),
          prefix_sum(inp),
          morph(10, inp),
          current(Stream([True, False, True, False]), inp, default=0),
          scan(inp),
      )
      cnt = counter(start=1)
      rec_pre = ifthenelse(fby(True, False), 10, pre(lambda: rec_pre))  # pylint: disable=undefined-variable
      a, b = pair_box(inp, lambda: a)  # pylint: disable=undefined-variable

      with automaton() as fsm:
        prev_out = last(lambda: fsm.out, init=0)
        with state('RUN', unless(prev_out >= 6, 'DONE')):
          local = cnt + 1
          until(local > 2, 'HOLD')
          fsm.out = local
        with state('HOLD'):
          until(True, 'RUN', reset=False)
          fsm.out = prev_out
        with state('DONE'):
          fsm.out = 99

      return a, b, fsm.out, rec_pre

    graph = Scheduler.build_graph(prog, semantics=Semantics.SYNC)
    self.assertIsNone(find_instantaneous_cycle(graph))
    check_causality(graph)
    _assert_valid_topological_order(self, graph)

    a, b, fsm, rec_pre = Scheduler.run_graph(graph, horizon=4)
    self.assertEqual(a, [2, 3, 4, 5])
    self.assertEqual(b, [4, 5, 6, 7])
    self.assertEqual(fsm, [2, 3, 3, 5])
    self.assertEqual(rec_pre, [10, 10, 10, 10])

  def test_direct_and_multi_node_combinational_cycles(self):
    def self_loop():
      cyc = ifthenelse(True, lambda: cyc, 0)  # pylint: disable=undefined-variable
      return cyc

    self_graph = Scheduler.build_graph(self_loop, semantics=Semantics.SYNC)
    self_cycle = find_instantaneous_cycle(self_graph)
    self.assertIsNotNone(self_cycle)
    self.assertEqual(self_cycle, self_graph.nodes)
    with self.assertRaisesRegex(
        CausalityError,
        r'Instantaneous causality cycle detected: ifthenelse#0 -> ifthenelse#0',
    ) as self_err:
      check_causality(self_graph)
    self.assertEqual(self_err.exception.cycle, self_graph.nodes)

    def two_node_cycle():
      a = ifthenelse(True, lambda: b, 0)  # pylint: disable=undefined-variable
      b = a + 1
      return a

    two_graph = Scheduler.build_graph(two_node_cycle, semantics=Semantics.SYNC)
    with self.assertRaisesRegex(
        CausalityError,
        r'ifthenelse#0 -> __add__#1 -> ifthenelse#0',
    ) as two_err:
      topological_sort(two_graph)
    self.assertEqual(two_err.exception.cycle, two_graph.nodes)

    def merge_arm_cycle():
      cond = Stream([False, False, True])
      x = merge(cond, lambda: x + 1, 0)  # pylint: disable=undefined-variable
      return x

    merge_graph = Scheduler.build_graph(
        merge_arm_cycle, semantics=Semantics.SYNC
    )
    with self.assertRaises(CausalityError) as merge_err:
      check_causality(merge_graph)
    _assert_valid_instantaneous_cycle(
        self, merge_graph, merge_err.exception.cycle
    )

  def test_fby_initial_and_control_stream_cycles(self):
    def fby_init_cycle():
      x = fby(lambda: x + 1, 0)  # pylint: disable=undefined-variable
      return x

    init_graph = Scheduler.build_graph(fby_init_cycle, semantics=Semantics.SYNC)
    with self.assertRaisesRegex(CausalityError, r'fby#0 -> __add__#1 -> fby#0'):
      check_causality(init_graph)

    def fby_both_ports_self_cycle():
      y = fby(lambda: y, lambda: y)  # pylint: disable=undefined-variable
      return y

    both_graph = Scheduler.build_graph(
        fby_both_ports_self_cycle, semantics=Semantics.SYNC
    )
    with self.assertRaisesRegex(CausalityError, r'fby#0 -> fby#0'):
      check_causality(both_graph)

    def fby_reset_cycle():
      with reset(lambda: y > 0):  # pylint: disable=undefined-variable
        y = fby(0, 1, name='y')
      return y

    rst_graph = Scheduler.build_graph(fby_reset_cycle, semantics=Semantics.SYNC)
    with self.assertRaises(CausalityError) as rst_err:
      check_causality(rst_graph)
    _assert_valid_instantaneous_cycle(self, rst_graph, rst_err.exception.cycle)

    def fby_match_activity_cycle():
      with match(lambda: m.out > 0) as m:
        with case(True):
          y = fby(0, 1, name='y')
          m.out = y
        with case(False):
          m.out = 0
      return m.out

    act_graph = Scheduler.build_graph(
        fby_match_activity_cycle, semantics=Semantics.SYNC
    )
    with self.assertRaises(CausalityError) as act_err:
      check_causality(act_graph)
    _assert_valid_instantaneous_cycle(self, act_graph, act_err.exception.cycle)

  def test_automaton_buffer_and_post_instantaneous_cycles(self):
    def stateless_match_selector_cycle():
      with match(lambda: m.out) as m:
        with case(True):
          m.out = False
        with case(False):
          m.out = True

      return m.out

    for sem in (Semantics.SYNC, Semantics.KAHN):
      match_graph = Scheduler.build_graph(
          stateless_match_selector_cycle, semantics=sem
      )
      with self.assertRaises(CausalityError) as match_err:
        check_causality(match_graph)
      _assert_valid_instantaneous_cycle(
          self, match_graph, match_err.exception.cycle
      )

    def strong_unless_output_cycle():
      with automaton() as fsm:
        with state(False, unless(lambda: fsm.out, True)):
          fsm.out = True
        with state(True):
          fsm.out = False

      return fsm.out

    unless_graph = Scheduler.build_graph(
        strong_unless_output_cycle, semantics=Semantics.SYNC
    )
    with self.assertRaises(CausalityError) as unless_err:
      check_causality(unless_graph)
    _assert_valid_instantaneous_cycle(
        self, unless_graph, unless_err.exception.cycle
    )

    def intra_state_combinational_cycle():
      with automaton() as fsm:
        with state('A'):
          v = ifthenelse(True, lambda: fsm.out, 0)
          fsm.out = v + 1
      return fsm.out

    intra_g = Scheduler.build_graph(
        intra_state_combinational_cycle, semantics=Semantics.SYNC
    )
    with self.assertRaises(CausalityError) as intra_err:
      check_causality(intra_g)
    _assert_valid_instantaneous_cycle(self, intra_g, intra_err.exception.cycle)

    def buffer_cycle():
      b = buffer(lambda: b, True, True)  # pylint: disable=undefined-variable
      return b

    buf_graph = Scheduler.build_graph(buffer_cycle, semantics=Semantics.SYNC)
    with self.assertRaisesRegex(CausalityError, r'buffer#0 -> buffer#0'):
      check_causality(buf_graph)

    def post_cycle():
      p = post(lambda: p)  # pylint: disable=undefined-variable
      return p

    post_graph = Scheduler.build_graph(post_cycle, semantics=Semantics.SYNC)
    with self.assertRaisesRegex(CausalityError, r'post#0 -> post#0'):
      check_causality(post_graph)

  def test_overlapping_cycles_and_check_causality_controls(self):
    with self.assertRaisesRegex(ValueError, 'non-empty'):
      _ = CausalityError(())
    with self.assertRaisesRegex(RuntimeError, 'not been finalized'):
      check_causality(Graph())

    def mixed_network():
      src = counter(start=1)
      delayed = fby(0, lambda: mid)  # pylint: disable=undefined-variable
      a = src + delayed + (lambda: mid)  # pylint: disable=undefined-variable
      mid = a + 1
      return mid * 2

    graph = Scheduler.build_graph(mixed_network, semantics=Semantics.SYNC)
    with self.assertRaises(CausalityError) as err:
      check_causality(graph)
    _assert_valid_instantaneous_cycle(self, graph, err.exception.cycle)
    cycle_ops = [n.op for n in err.exception.cycle]
    self.assertNotIn('fby', cycle_ops)

    def cyclic_sync_or_kahn():
      a = ifthenelse(True, lambda: b, 0)  # pylint: disable=undefined-variable
      b = a + 1
      return a

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.assertRaises(CausalityError):
        Scheduler.run(cyclic_sync_or_kahn, horizon=3, semantics=sem)

      a_trace = Scheduler.run(
          cyclic_sync_or_kahn,
          horizon=3,
          semantics=sem,
          causality_analysis=False,
      )
      self.assertTrue(Scheduler.deadlocked)
      self.assertEqual(a_trace, [])


if __name__ == '__main__':
  absltest.main()
