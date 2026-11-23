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
from psyre.builder import build_graph
from psyre.common import (
    Semantics,
    buffer,
    counter,
    fby,
    ifthenelse,
    logger,
    merge,
    post,
    pre,
    psyre_comb,
    reset,
)
from psyre.graph import Graph
from psyre.match import (
    case,
    last,
    match,
)
from psyre.scheduler import Scheduler
from psyre.stream import Stream

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

    graph = build_graph(prog, semantics=Semantics.SYNC)
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

    self_graph = build_graph(self_loop, semantics=Semantics.SYNC)
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

    two_graph = build_graph(two_node_cycle, semantics=Semantics.SYNC)
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

    self._assert_cycle(merge_arm_cycle)

  def _assert_cycle(
      self,
      prog,
      *,
      regex: str | None = None,
      semantics: Semantics = Semantics.SYNC,
  ):
    graph = build_graph(prog, semantics=semantics)
    if regex is not None:
      with self.assertRaisesRegex(CausalityError, regex) as err:
        check_causality(graph)
    else:
      with self.assertRaises(CausalityError) as err:
        check_causality(graph)
    _assert_valid_instantaneous_cycle(self, graph, err.exception.cycle)
    return err.exception.cycle

  def test_fby_initial_and_control_stream_cycles(self):
    def fby_init_cycle():
      x = fby(lambda: x + 1, 0)  # pylint: disable=undefined-variable
      return x

    def fby_both_ports_self_cycle():
      y = fby(lambda: y, lambda: y)  # pylint: disable=undefined-variable
      return y

    def fby_reset_cycle():
      with reset(lambda: y > 0):  # pylint: disable=undefined-variable
        y = fby(0, 1, name='y')
      return y

    def fby_match_activity_cycle():
      with match(lambda: m.out > 0) as m:
        with case(True):
          y = fby(0, 1, name='y')
          m.out = y
        with case(False):
          m.out = 0
      return m.out

    self._assert_cycle(fby_init_cycle, regex=r'fby#0 -> __add__#1 -> fby#0')
    self._assert_cycle(fby_both_ports_self_cycle, regex=r'fby#0 -> fby#0')
    self._assert_cycle(fby_reset_cycle)
    self._assert_cycle(fby_match_activity_cycle)

  def test_automaton_buffer_and_post_instantaneous_cycles(self):
    def stateless_match_selector_cycle():
      with match(lambda: m.out) as m:
        with case(True):
          m.out = False
        with case(False):
          m.out = True
      return m.out

    for sem in (Semantics.SYNC, Semantics.KAHN):
      self._assert_cycle(stateless_match_selector_cycle, semantics=sem)

    def strong_unless_output_cycle():
      with automaton() as fsm:
        with state(False, unless(lambda: fsm.out, True)):
          fsm.out = True
        with state(True):
          fsm.out = False
      return fsm.out

    def intra_state_combinational_cycle():
      with automaton() as fsm:
        with state('A'):
          v = ifthenelse(True, lambda: fsm.out, 0)
          fsm.out = v + 1
      return fsm.out

    def buffer_cycle():
      b = buffer(lambda: b, True, True)  # pylint: disable=undefined-variable
      return b

    def post_cycle():
      p = post(lambda: p)  # pylint: disable=undefined-variable
      return p

    self._assert_cycle(strong_unless_output_cycle)
    self._assert_cycle(intra_state_combinational_cycle)
    self._assert_cycle(buffer_cycle, regex=r'buffer#0 -> buffer#0')
    self._assert_cycle(post_cycle, regex=r'post#0 -> post#0')

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

    cycle_ops = [n.op for n in self._assert_cycle(mixed_network)]
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
