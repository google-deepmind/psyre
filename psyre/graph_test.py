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
"""Unit tests for Psyre Stage-1 Graph construction and trace capture."""

import logging
from typing import Any, NamedTuple

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre import optimization
from psyre.graph import (
    Graph,
    as_istruct,
)
from psyre.match import (
    case,
    match,
)
from psyre.psyre import (
    Absent,
    Scheduler,
    Semantics,
    Stream,
    counter,
    fby,
    logger,
    merge,
    periodic_stream,
    psyre_comb,
    psyre_node,
    psyre_sequ,
    reset,
    resolve_handle,
    when,
    whennot,
)

# pylint: enable=g-importing-member,g-multiple-import


class _Bundle(NamedTuple):
  """Container testing nested NamedTuple PyTree output capture."""

  left: Any
  right: Any


class _NamedPair(NamedTuple):
  """Container testing NamedTuple PyTree output capture."""

  first: Any
  second: Any


class GraphTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def test_stage_one_graph_construction_and_wiring(self):
    holders: dict[str, Stream] = {}

    def prog():
      x = Stream([10, 20, 30], name='x')
      raw = Stream([7, 8, 9], name='raw')
      acc = fby(0, lambda: nxt, name='acc')  # pylint: disable=undefined-variable
      nxt = acc + x
      holders['x'] = x
      holders['raw'] = raw
      holders['acc'] = acc
      holders['nxt'] = nxt
      return nxt

    graph = Scheduler.build_graph(prog, semantics=Semantics.SYNC)
    self.assertIs(Scheduler.graph, graph)
    self.assertTrue(graph.is_finalized)
    self.assertEqual([n.op for n in graph.nodes], ['fby', '__add__'])
    self.assertEqual([n.name for n in graph.nodes], ['acc', '__add__'])
    self.assertEqual([n.id for n in graph.nodes], [0, 1])

    fby_node, add_node = graph.nodes
    self.assertTrue(fby_node.is_resolved)
    self.assertTrue(add_node.is_resolved)
    self.assertIs(graph.producer(holders['acc']), fby_node)
    self.assertIs(graph.producer(holders['nxt']), add_node)
    self.assertIsNone(graph.producer(holders['x']))
    self.assertIsNone(graph.producer(holders['raw']))

    self.assertEqual(graph.consumers(holders['x']), (add_node,))
    self.assertEqual(graph.consumers(holders['acc']), (add_node,))
    self.assertEqual(graph.consumers(holders['nxt']), (fby_node,))
    self.assertEmpty(graph.consumers(holders['raw']))
    self.assertIn(holders['raw'].istruct, graph.streams)

    # Full graph edges vs instantaneous-only edges:
    self.assertEqual(graph.predecessors(fby_node), (add_node,))
    self.assertEmpty(graph.predecessors(fby_node, instantaneous_only=True))
    self.assertEqual(
        fby_node.delayed_input_streams(), (holders['nxt'].istruct,)
    )
    self.assertEmpty(add_node.delayed_input_streams())
    self.assertEqual(graph.predecessors(add_node), (fby_node,))
    self.assertEqual(
        graph.predecessors(add_node, instantaneous_only=True), (fby_node,)
    )
    self.assertEqual(graph.successors(add_node), (fby_node,))
    self.assertEmpty(graph.successors(add_node, instantaneous_only=True))
    self.assertEqual(
        graph.successors(fby_node, instantaneous_only=True), (add_node,)
    )

    nxt_trace = Scheduler.run(prog, horizon=3, semantics=Semantics.SYNC)
    self.assertEqual(nxt_trace, [10, 30, 60])
    self.assertLen(Scheduler.graph.nodes, 2)
    self.assertEqual(
        Scheduler.run_graph(graph, horizon=3, streams=('acc', 'raw')),
        {'acc': [0, 10, 30], 'raw': [7, 8, 9]},
    )

  def test_deferred_scope_thunk_resolution_in_match_and_reset(self):
    def prog():
      with match(lambda: gate > 0):  # pylint: disable=undefined-variable
        with case(True):
          with reset(lambda: rst):  # pylint: disable=undefined-variable
            y = fby(0, lambda: y + 1, name='y')  # pylint: disable=undefined-variable
            doubled = y * 2
            doubled.name = 'doubled'
        with case(False):
          pass
      gate = Stream([1, 0, 1, 1, 1])
      rst = Stream([False, False, False, True, False])

    graph = Scheduler.build_graph(
        prog, semantics=Semantics.SYNC, extra_roots=('y', 'doubled')
    )

    gt_nodes = [n for n in graph.nodes if n.op == '__gt__']
    self.assertLen(gt_nodes, 1)
    gt_node = gt_nodes[0]
    self.assertEmpty(gt_node.resets)

    # y, doubled, and thunk-instantiated (y + 1) all run in case(True), with
    # enclosing inputs/resets lowered to when(act, ...) and fby inheriting the
    # gated reset.
    inner_ops = {'fby', '__mul__', '__add__'}
    inner_nodes = [n for n in graph.nodes if n.op in inner_ops]
    self.assertLen(inner_nodes, 3)
    fby_nodes = [n for n in inner_nodes if n.op == 'fby']
    self.assertLen(fby_nodes, 1)
    fby_node = fby_nodes[0]
    self.assertLen(fby_node.resets, 1)
    rst_prod = graph.producer(fby_node.resets[0].istruct)
    self.assertIsNotNone(rst_prod)
    self.assertEqual(rst_prod.op, 'when')
    act_prod = graph.producer(rst_prod.inputs[0].istruct)
    self.assertIsNotNone(act_prod)
    self.assertEqual(act_prod.op, 'merge')
    self.assertIs(act_prod.inputs[0].istruct, gt_node.output_stream)
    for node in inner_nodes:
      if node.op != 'fby':
        self.assertEmpty(node.resets)

    sync_res = Scheduler.run(
        prog, horizon=5, semantics=Semantics.SYNC, streams=('y', 'doubled')
    )
    self.assertEqual(sync_res['y'], [0, Absent, 1, 0, 1])
    self.assertEqual(sync_res['doubled'], [0, Absent, 2, 0, 2])

    kahn_res = Scheduler.run(
        prog, horizon=5, semantics=Semantics.KAHN, streams=('y', 'doubled')
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(kahn_res['y'], [0, 1, 0, 1])
    self.assertEqual(kahn_res['doubled'], [0, 2, 0, 2])

  def test_run_graph_stage_two_kahn_and_sync(self):
    def prog():
      mask = Stream([True, False, True, False])
      vals = Stream([10, 20, 30, 40])
      sampled = when(mask, vals, name='sampled')
      complement = whennot(mask, -vals, name='complement')
      merged = merge(mask, sampled, complement, name='merged')
      cnt = counter(start=1, name='cnt')
      return sampled, merged, cnt

    kahn_graph = Scheduler.build_graph(prog, semantics=Semantics.KAHN)
    sync_graph = Scheduler.build_graph(prog, semantics=Semantics.SYNC)
    self.assertEqual(kahn_graph.semantics, Semantics.KAHN)
    self.assertEqual(sync_graph.semantics, Semantics.SYNC)

    # Default graph=None runs the most recently built Scheduler.graph:
    sampled_default, _, _ = Scheduler.run_graph(horizon=3)
    self.assertIs(Scheduler.graph, sync_graph)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(sampled_default, [10, Absent, 30])

    # Stage 2 runs kahn_graph and sync_graph explicitly and can re-run a Graph:
    kahn_4 = Scheduler.run_graph(kahn_graph, horizon=4)
    self.assertIs(Scheduler.graph, kahn_graph)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(kahn_4, ([10, 30], [10, -20, 30, -40], [1, 2, 3, 4]))

    sync_4 = Scheduler.run_graph(sync_graph, horizon=4)
    self.assertIs(Scheduler.graph, sync_graph)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(
        sync_4, ([10, Absent, 30, Absent], [10, -20, 30, -40], [1, 2, 3, 4])
    )

    kahn_2 = Scheduler.run_graph(kahn_graph, horizon=2)
    self.assertEqual(kahn_2, ([10], [10, -20], [1, 2]))
    # Earlier returned traces remain untouched across subsequent Graph runs:
    self.assertEqual(kahn_4[0], [10, 30])

    probed = Scheduler.run_graph(
        kahn_graph, horizon=4, streams=('complement', 'sampled')
    )
    self.assertEqual(probed, {'complement': [-20, -40], 'sampled': [10, 30]})

  def test_graph_and_scheduler_validation_guards(self):
    self.assertFalse(hasattr(Scheduler, 'active'))
    sched_instance = Scheduler()
    private_bool_attrs = {
        name
        for name, value in vars(sched_instance).items()
        if name.startswith('_')
        and not name.startswith('__')
        and isinstance(value, bool)
    }
    self.assertEqual(private_bool_attrs, {'_building', '_deadlocked'})

    bad_stream: Any = 123
    with self.assertRaisesRegex(TypeError, 'Expected IStruct or IStructHandle'):
      as_istruct(bad_stream)

    @psyre_comb
    def inc(x):
      return x + 1

    with self.assertRaisesRegex(RuntimeError, 'Stage 1'):
      inc(1)

    @psyre_sequ
    def step_acc(x):
      acc = x
      while True:
        x = yield acc
        acc += x

    with self.assertRaisesRegex(RuntimeError, 'Stage 1'):
      step_acc(1)

    @psyre_node('sub_block')
    def sub_block(x):
      return x + 1

    with self.assertRaisesRegex(RuntimeError, 'Stage 1'):
      sub_block(1)
    with self.assertRaisesRegex(TypeError, 'non-empty str'):
      psyre_node('')(lambda x: x)

    raw_graph = Graph()
    out = Stream()

    async def dummy_step(**_kwargs):
      return None

    node = raw_graph.add_node(
        op='dummy',
        name='dummy',
        output=out,
        step_func=dummy_step,
        raw_inputs=(),
        raw_resets=(),
        kwargs={},
    )
    with self.assertRaisesRegex(RuntimeError, 'not been resolved'):
      node.input_streams()
    with self.assertRaisesRegex(RuntimeError, 'not been finalized'):
      Scheduler.run_graph(raw_graph, horizon=1)
    with self.assertRaisesRegex(ValueError, 'already has producer'):
      raw_graph.add_node(
          op='dup',
          name='dup',
          output=out,
          step_func=dummy_step,
          raw_inputs=(),
          raw_resets=(),
          kwargs={},
      )

    raw_graph.finalize(resolve_handle)
    with self.assertRaisesRegex(RuntimeError, 'already finalized'):
      raw_graph.add_node(
          op='late',
          name='late',
          output=Stream(),
          step_func=dummy_step,
          raw_inputs=(),
          raw_resets=(),
          kwargs={},
      )

    other_graph = Graph()
    other_graph.finalize(resolve_handle)
    with self.assertRaisesRegex(ValueError, 'does not belong'):
      other_graph.predecessors(node)

  def test_functional_output_pytree_streams_filter_and_selective_keep(self):
    sched = Scheduler()
    self.assertFalse(sched.deadlocked)
    self.assertIsInstance(sched.graph, Graph)
    with self.assertRaisesRegex(TypeError, 'Psyre output'):
      Scheduler.run(lambda: 42, horizon=2)

    self.assertIsNone(
        Scheduler.run(lambda: (counter(start=1), None)[1], horizon=2)
    )

    handles: dict[str, Stream] = {}

    def rich_prog():
      mid = counter(start=1, name='mid') * 10
      tagged = mid + 1
      tagged.name = 'tagged'
      stat = periodic_stream([5], [6, 7], name='stat')
      with match(Stream([True, False, True, False])) as m:
        with case(True):
          m.a = tagged
          m.b = 1
        with case(False):
          m.a = 0
          m.b = tagged
      handles['mid'] = mid
      handles['tagged'] = tagged
      return {
          'bundle': _Bundle(left=tagged, right=_NamedPair(stat, [mid + 2])),
          'block': m,
      }

    graph = Scheduler.build_graph(rich_prog, semantics=Semantics.KAHN)
    out = Scheduler.run_graph(graph, horizon=4)
    self.assertEqual(
        out,
        {
            'bundle': _Bundle(
                left=[11, 21, 31, 41],
                right=_NamedPair([5, 6, 7, 6], [[12, 22, 32, 42]]),
            ),
            'block': {'a': [11, 0, 31, 0], 'b': [1, 21, 1, 41]},
        },
    )
    # Uncaptured intermediate stream `mid` does not buffer values in keep:
    self.assertEmpty(handles['mid'].istruct.keep)
    self.assertEqual(handles['tagged'].istruct.keep, [11, 21, 31, 41])

    # Explicit streams= overrides output PyTree and captures only named streams:
    named_out = Scheduler.run_graph(
        graph, horizon=3, streams=('tagged', 'stat')
    )
    self.assertEqual(named_out, {'tagged': [11, 21, 31], 'stat': [5, 6, 7]})
    self.assertEqual(
        Scheduler.run(rich_prog, horizon=2, streams=['tagged']),
        {'tagged': [11, 21]},
    )

    streams_str: Any = 'tagged'
    with self.assertRaisesRegex(TypeError, 'not a bare str'):
      Scheduler.run_graph(graph, horizon=2, streams=streams_str)
    with self.assertRaisesRegex(ValueError, 'non-empty str'):
      Scheduler.run_graph(graph, horizon=2, streams=('',))
    with self.assertRaisesRegex(KeyError, 'Unknown stream name'):
      Scheduler.run_graph(graph, horizon=2, streams=('nonexistent',))

    def duplicate_names():
      return counter(start=0), counter(start=10)

    dup_graph = Scheduler.build_graph(duplicate_names, semantics=Semantics.KAHN)
    with self.assertRaisesRegex(ValueError, 'Ambiguous stream name'):
      Scheduler.run_graph(dup_graph, horizon=2, streams=('counter',))

  def test_dead_code_elimination_and_extra_roots(self):
    holders: dict[str, Stream] = {}

    def prog_with_dead_branches():
      x = Stream([1, 2, 3, 4], name='x')
      live_cnt = fby(0, lambda: live_cnt + x, name='live_cnt')  # pylint: disable=undefined-variable
      dead_scalar = x * 10
      dead_scalar.name = 'dead_scalar'
      dead_loop = fby(99, lambda: dead_loop + dead_scalar)  # pylint: disable=undefined-variable
      dead_loop.name = 'dead_loop'
      holders['dead_scalar'] = dead_scalar
      holders['dead_loop'] = dead_loop
      return live_cnt

    # By default (optimize=True), build_graph eliminates dead_scalar and
    # dead_loop (even though dead_loop has a self-referential feedback cycle).
    opt_graph = Scheduler.build_graph(
        prog_with_dead_branches, semantics=Semantics.SYNC
    )
    self.assertEmpty(optimization.find_dead_nodes(opt_graph))
    self.assertEqual([n.name for n in opt_graph.nodes], ['live_cnt', '__add__'])
    self.assertEqual([n.id for n in opt_graph.nodes], [0, 1])
    self.assertIsNone(opt_graph.producer(holders['dead_scalar']))
    self.assertIsNone(opt_graph.producer(holders['dead_loop']))
    self.assertEqual(Scheduler.run_graph(opt_graph, horizon=4), [0, 1, 3, 6])

    # With optimize=False, dead nodes are retained until dce() is called.
    raw_graph = Scheduler.build_graph(
        prog_with_dead_branches, semantics=Semantics.SYNC, optimize=False
    )
    dead_names = [n.name for n in optimization.find_dead_nodes(raw_graph)]
    self.assertEqual(dead_names, ['__mul__', 'fby', '__add__'])

    # Keeping 'dead_scalar' via extra_roots preserves dead_scalar while still
    # pruning dead_loop.
    live_with_extra = optimization.find_live_nodes(
        raw_graph, extra_roots=('dead_scalar',)
    )
    self.assertEqual(
        [n.name for n in live_with_extra],
        ['live_cnt', '__mul__', '__add__'],
    )
    dead_scalar_node = raw_graph.producer(holders['dead_scalar'])
    self.assertIsNotNone(dead_scalar_node)
    self.assertEqual(
        optimization.find_live_nodes(raw_graph, extra_roots=[dead_scalar_node]),
        live_with_extra,
    )
    self.assertEqual(
        optimization.find_live_nodes(
            raw_graph, extra_roots=[holders['dead_scalar']]
        ),
        live_with_extra,
    )

    bad_roots_str: Any = 'dead_scalar'
    with self.assertRaisesRegex(TypeError, 'not a bare str'):
      optimization.find_live_nodes(raw_graph, extra_roots=bad_roots_str)
    with self.assertRaisesRegex(TypeError, 'extra_roots elements'):
      optimization.find_live_nodes(raw_graph, extra_roots=[123])

    optimization.dce(raw_graph)
    self.assertEmpty(optimization.find_dead_nodes(raw_graph))
    self.assertEqual([n.name for n in raw_graph.nodes], ['live_cnt', '__add__'])

    # Scheduler.run(..., streams=('dead_scalar',)) automatically keeps
    # 'dead_scalar' as an extra root during DCE so it can be probed.
    probed = Scheduler.run(
        prog_with_dead_branches,
        horizon=4,
        semantics=Semantics.SYNC,
        streams=('dead_scalar',),
    )
    self.assertEqual(probed, {'dead_scalar': [10, 20, 30, 40]})


if __name__ == '__main__':
  absltest.main()
