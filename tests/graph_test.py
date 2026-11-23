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

import asyncio
import enum
import gc
import logging
import math
from typing import Any, NamedTuple
import weakref

from absl.testing import absltest

import psyre
# pylint: disable=g-importing-member,g-multiple-import
from psyre import clock
from psyre import kahn_primitives
from psyre import math as psyre_math
from psyre import optimization
from psyre import static_stream as static_stream_lib
from psyre import sync_primitives
from psyre.builder import GraphBuilder, build_graph, lower_resets
from psyre.clock import check_clocks
from psyre.common import (
    Absent,
    Bottom,
    LazyThunk,
    Semantics,
    active_scheduler,
    buffer,
    counter,
    fby,
    force_thunk,
    ifthenelse,
    is_building,
    is_thunk,
    logger,
    merge,
    post,
    prefix_sum,
    psyre_comb,
    psyre_loop,
    psyre_node,
    psyre_sequ,
    pull,
    reset,
    switch,
    unwrap_inst_func,
    when,
    whennot,
)
from psyre.graph import Graph
from psyre.match import (
    case,
    match,
)
from psyre.scheduler import Scheduler
from psyre.static_stream import (
    ConcatWord,
    EMPTY_WORD,
    LiteralWord,
    RepeatWord,
    StaticSequence,
    StaticWord,
    concat,
    repeat,
)
from psyre.stream import (
    Stream,
    StreamPort,
    resolve_stream,
    same_stream_or_const,
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


def _norm_trace(trace: list[Any]) -> list[Any]:
  """Normalizes `Absent`/`Bottom` sentinels for structural list equality."""
  return [
      '<A>' if v is Absent else ('<B>' if v is Bottom else v) for v in trace
  ]


class GraphTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def _assert_opt_parity(self, prog_fn, *, horizon: int = 4) -> None:
    """Asserts optimize=True matches optimize=False under SYNC and KAHN."""
    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(prog=prog_fn.__name__, semantics=sem):
        raw_res = Scheduler.run(
            prog_fn, horizon=horizon, semantics=sem, optimize=False
        )
        opt_res = Scheduler.run(
            prog_fn, horizon=horizon, semantics=sem, optimize=True
        )
        self.assertEqual(
            tuple(_norm_trace(tr) for tr in opt_res),
            tuple(_norm_trace(tr) for tr in raw_res),
        )

  def _assert_mux_key_error(
      self,
      prog_fn,
      *,
      horizon: int = 2,
      optimize: bool = True,
  ) -> None:
    """Asserts `prog_fn` fails with `ExceptionGroup` wrapping mux `KeyError`."""
    with self.assertRaises(ExceptionGroup) as ctx:
      Scheduler.run(
          prog_fn,
          horizon=horizon,
          semantics=Semantics.SYNC,
          optimize=optimize,
      )
    self.assertTrue(
        any(
            isinstance(err, KeyError)
            and 'Unhandled merge selector' in str(err)
            for err in ctx.exception.exceptions
        )
    )

  def _assert_sync_kahn(self, prog_fn, expected: Any, *, horizon: int) -> None:
    """Asserts `prog_fn` produces `expected` under both SYNC and KAHN."""
    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(prog=prog_fn.__name__, semantics=sem):
        self.assertEqual(
            Scheduler.run(prog_fn, horizon=horizon, semantics=sem), expected
        )

  def test_stage_one_graph_construction_and_wiring(self):
    holders: dict[str, Stream] = {}

    def prog():
      self.assertTrue(GraphBuilder.is_building())
      self.assertTrue(is_building())
      self.assertIsNotNone(GraphBuilder.current())
      x = Stream([10, 20, 30], name='x')
      raw = Stream([7, 8, 9], name='raw')
      acc = fby(0, lambda: nxt, name='acc')  # pylint: disable=undefined-variable
      nxt = acc + x
      holders['x'] = x
      holders['raw'] = raw
      holders['acc'] = acc
      holders['nxt'] = nxt
      return nxt

    self.assertFalse(GraphBuilder.is_building())
    self.assertFalse(is_building())
    self.assertIsNone(GraphBuilder.current())
    graph = build_graph(prog, semantics=Semantics.SYNC)
    self.assertFalse(GraphBuilder.is_building())
    self.assertFalse(is_building())
    self.assertIsNone(GraphBuilder.current())
    self.assertIsNot(Scheduler.graph, graph)
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
    self.assertIn(holders['raw'].id, {s.id for s in graph.streams})

    # Full graph edges vs instantaneous-only edges:
    self.assertEqual(graph.predecessors(fby_node), (add_node,))
    self.assertEmpty(graph.predecessors(fby_node, instantaneous_only=True))
    delayed = fby_node.delayed_input_streams()
    self.assertLen(delayed, 1)
    self.assertIs(delayed[0], holders['nxt'])
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

    graph = build_graph(
        prog, semantics=Semantics.SYNC, extra_roots=('y', 'doubled')
    )

    gt_nodes = [n for n in graph.nodes if n.op == '__gt__']
    self.assertLen(gt_nodes, 1)
    gt_node = gt_nodes[0]
    self.assertEmpty(gt_node.resets)

    # y, doubled, and thunk-instantiated (y + 1) all run in case(True). Static
    # polymorphic `rst` stays unwrapped; match lowering appends the activation
    # reset `when(act, base_false)` to `fby_node.resets`.
    inner_ops = {'fby', '__mul__', '__add__'}
    inner_nodes = [n for n in graph.nodes if n.op in inner_ops]
    self.assertLen(inner_nodes, 3)
    fby_nodes = [n for n in inner_nodes if n.op == 'fby']
    self.assertLen(fby_nodes, 1)
    fby_node = fby_nodes[0]
    self.assertLen(fby_node.resets, 2)
    self.assertTrue(fby_node.resets[0].static)
    act_prod = graph.producer(fby_node.resets[1])
    self.assertIsNotNone(act_prod)
    self.assertEqual(act_prod.op, 'when')
    self.assertIs(act_prod.inputs[0], gt_node.output)
    for node in inner_nodes:
      if node.op != 'fby':
        self.assertEmpty(node.resets)

    # Under uniform static clock polymorphism, `rst` inside `case(True)` tracks
    # active ticks of `gate > 0` (t=0, 2, 3, 4), so `rst[3]=True` fires at t=4.
    sync_res = Scheduler.run(
        prog, horizon=5, semantics=Semantics.SYNC, streams=('y', 'doubled')
    )
    self.assertEqual(sync_res['y'], [0, Absent, 1, 2, 0])
    self.assertEqual(sync_res['doubled'], [0, Absent, 2, 4, 0])

    kahn_res = Scheduler.run(
        prog, horizon=5, semantics=Semantics.KAHN, streams=('y', 'doubled')
    )
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(kahn_res['y'], [0, 1, 2, 0])
    self.assertEqual(kahn_res['doubled'], [0, 2, 4, 0])

  def test_run_graph_stage_two_kahn_and_sync(self):
    def prog():
      mask = Stream([True, False, True, False])
      vals = Stream([10, 20, 30, 40])
      sampled = when(mask, vals, name='sampled')
      complement = whennot(mask, -vals, name='complement')
      merged = merge(mask, sampled, complement, name='merged')
      cnt = counter(start=1, name='cnt')
      return sampled, merged, cnt

    kahn_graph = build_graph(prog, semantics=Semantics.KAHN)
    sync_graph = build_graph(prog, semantics=Semantics.SYNC)
    self.assertEqual(kahn_graph.semantics, Semantics.KAHN)
    self.assertEqual(sync_graph.semantics, Semantics.SYNC)

    # Stage 2 runs sync_graph, then graph=None re-runs Scheduler.graph:
    sampled_sync_3, _, _ = Scheduler.run_graph(sync_graph, horizon=3)
    self.assertIs(Scheduler.graph, sync_graph)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEqual(sampled_sync_3, [10, Absent, 30])
    sampled_default, _, _ = Scheduler.run_graph(horizon=3)
    self.assertIs(Scheduler.graph, sync_graph)
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
    self.assertEqual(private_bool_attrs, {'_deadlocked', '_failed', '_running'})

    bad_stream: Any = 123
    with self.assertRaisesRegex(TypeError, 'Expected Stream'):
      Graph().add_stream(bad_stream)
    with self.assertRaisesRegex(TypeError, 'Expected Stream'):
      Graph().producer(bad_stream)
    with self.assertRaisesRegex(TypeError, 'Psyre output'):
      Graph().set_output(bad_stream)
    finalized_g = build_graph(lambda: Stream([1, 2]))
    with self.assertRaisesRegex(TypeError, 'replace_stream expects Stream'):
      finalized_g.replace_stream(bad_stream, Stream([3, 4]))
    with self.assertRaisesRegex(TypeError, 'replace_stream expects Stream'):
      finalized_g.replace_stream(finalized_g.output, bad_stream)

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
        kwargs={},
    )
    with self.assertRaisesRegex(RuntimeError, 'not been resolved'):
      node.input_streams()
    with self.assertRaisesRegex(RuntimeError, 'unresolved nodes'):
      raw_graph.mark_finalized()
    with self.assertRaisesRegex(RuntimeError, 'not been finalized'):
      Scheduler.run_graph(raw_graph, horizon=1)
    with self.assertRaisesRegex(ValueError, 'already has producer'):
      raw_graph.add_node(
          op='dup',
          name='dup',
          output=out,
          step_func=dummy_step,
          kwargs={},
      )

    raw_graph.rewire_node(node, inputs=(), resets=())
    raw_graph.mark_finalized()
    with self.assertRaisesRegex(RuntimeError, 'already finalized'):
      raw_graph.add_node(
          op='late',
          name='late',
          output=Stream(),
          step_func=dummy_step,
          kwargs={},
      )

    other_graph = Graph()
    other_graph.mark_finalized()
    with self.assertRaisesRegex(ValueError, 'does not belong'):
      other_graph.predecessors(node)

  def test_functional_output_pytree_streams_filter_and_selective_keep(self):
    sched = Scheduler()
    self.assertEqual(sched.semantics, Semantics.SYNC)
    self.assertFalse(sched.deadlocked)
    self.assertIsInstance(sched.graph, Graph)
    self.assertEqual(sched.graph.semantics, Semantics.SYNC)
    default_graph = build_graph(lambda: counter(start=1))
    self.assertEqual(default_graph.semantics, Semantics.SYNC)
    self.assertEqual(Scheduler.run_graph(default_graph, horizon=3), [1, 2, 3])
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
      stat = Stream.periodic([5], [6, 7], name='stat')
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

    graph = build_graph(rich_prog, semantics=Semantics.KAHN)
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
    # Runtime buffers are released once run_graph() finishes extracting traces:
    self.assertEmpty(handles['mid'].seq)
    self.assertEmpty(handles['tagged'].seq)

    # Buffers are also cleared if Stage 2 raises mid-run.
    @psyre_comb
    def fail_at_two(v):
      if v >= 2:
        raise ValueError('mid-run error')
      return v

    fail_handles: dict[str, Stream] = {}

    def failing_prog():
      src = counter(start=0)
      out_s = fail_at_two(src)
      fail_handles['src'] = src
      fail_handles['out'] = out_s
      return out_s

    fail_graph = build_graph(failing_prog, semantics=Semantics.KAHN)
    with self.assertRaises(ExceptionGroup):
      Scheduler.run_graph(fail_graph, horizon=4)
    self.assertFalse(Scheduler.deadlocked)
    self.assertEmpty(fail_handles['src'].seq)
    self.assertEmpty(fail_handles['out'].seq)
    self.assertEqual(Scheduler.run_graph(fail_graph, horizon=2), [0, 1])

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

    dup_graph = build_graph(duplicate_names, semantics=Semantics.KAHN)
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
    opt_graph = build_graph(
        prog_with_dead_branches, semantics=Semantics.SYNC
    )
    self.assertEmpty(optimization.find_dead_nodes(opt_graph))
    self.assertEqual([n.name for n in opt_graph.nodes], ['live_cnt', '__add__'])
    self.assertEqual([n.id for n in opt_graph.nodes], [0, 1])
    self.assertIsNone(opt_graph.producer(holders['dead_scalar']))
    self.assertIsNone(opt_graph.producer(holders['dead_loop']))
    self.assertEqual(Scheduler.run_graph(opt_graph, horizon=4), [0, 1, 3, 6])

    # With optimize=False, dead nodes are retained until dce() is called.
    raw_graph = build_graph(
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

    optimization.eliminate_dead_code(raw_graph)
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

  def test_graph_optimization_simplification_and_cse(self):
    @psyre_node('box_a')
    def box_a(u, v):
      return u + v

    @psyre_node('box_b')
    def box_b(u, v):
      return u + v

    def prog():
      t = counter(0, name='t')
      c = (t % 2) == 0
      x = counter(10, name='x')
      y = counter(1, name='y')
      s = ifthenelse(c, 'A', 'B')

      # Dynamic degenerate ifthenelse / mux / boolean folding:
      ite_same = ifthenelse(c, x, x, name='ite_same')
      unif_sw = switch(s, {'A': x, 'B': x}, name='unif_sw')
      id_m = merge(c, when(c, c), whennot(c, c), name='id_m')
      unif_m = merge(c, when(c, y), whennot(c, y), name='unif_m')
      bool_idemp = (c & c) | (c | c)
      bool_dneg = ~~c

      # Pure combinational CSE within same scope, preserved across prefixes:
      dup1 = x + y
      dup2 = x + y
      w1 = when(c, x)
      w2 = when(c, x)
      scoped = box_a(x, y) + box_b(x, y)
      return (
          ite_same,
          unif_sw,
          id_m,
          unif_m,
          bool_idemp,
          bool_dneg,
          dup1 + dup2,
          merge(c, w1 + w2, whennot(c, y)),
          scoped,
      )

    raw_g = build_graph(prog, semantics=Semantics.SYNC, optimize=False)
    opt_g = build_graph(prog, semantics=Semantics.SYNC, optimize=True)
    self.assertLess(len(opt_g.nodes), len(raw_g.nodes))
    # Uniform ifthenelse/switch fold away; DCE prunes the dead selector `s`:
    self.assertEmpty([n for n in opt_g.nodes if n.op == 'ifthenelse'])
    self.assertEmpty([n for n in opt_g.nodes if n.op == 'switch'])
    self.assertEmpty(
        [n for n in opt_g.nodes if n.op in {'__and__', '__or__', '__invert__'}]
    )
    # Only the non-degenerate merge remains, and duplicate when(c, x) is CSE'd:
    self.assertLen([n for n in opt_g.nodes if n.op == 'merge'], 1)
    self.assertLen([n for n in opt_g.nodes if n.op == 'when'], 1)
    # Scoped @psyre_node ops stay distinct across box_a vs box_b:
    opt_names = [n.name for n in opt_g.nodes]
    self.assertIn('box_a/__add__', opt_names)
    self.assertIn('box_b/__add__', opt_names)
    # Results match unoptimized execution exactly:
    self.assertEqual(
        Scheduler.run_graph(opt_g, horizon=4),
        Scheduler.run_graph(raw_g, horizon=4),
    )
    # Replaced named streams remain probeable via streams=(...):
    probed = Scheduler.run_graph(
        opt_g, horizon=4, streams=('ite_same', 'unif_sw', 'id_m', 'unif_m')
    )
    self.assertEqual(
        probed,
        {
            'ite_same': [10, 11, 12, 13],
            'unif_sw': [10, 11, 12, 13],
            'id_m': [True, False, True, False],
            'unif_m': [1, 2, 3, 4],
        },
    )

    # Polymorphic static folding, algebraic identities, boolean ops, and CSE:
    def static_prog():
      c = Stream([True, False, True, False], name='c')
      s = Stream(['A', 'B', 'A', 'B'], name='s')
      x = Stream([10, 20, 30, 40], name='x')
      y = Stream([1, 2, 3, 4], name='y')

      # Degenerate ifthenelse folding (subclock constants must not lose Absent):
      ite_t = ifthenelse(True, x, y, name='ite_t')
      ite_f = ifthenelse(False, x, y, name='ite_f')
      ite_same = ifthenelse(c, x, x, name='ite_same')
      ite_bool = ifthenelse(c, True, False, name='ite_bool')
      ite_sub_const = ifthenelse(when(c, c), 0, 0, name='ite_sub_const')

      # Constant, identity & uniform mux folding and tagged when folding:
      w_triv = when(Stream(True), x, name='w_triv')
      w_tag_triv = when(Stream('A'), y, values='A', name='w_tag_triv')
      const_sw = switch(Stream('A'), {'A': x, 'B': y}, name='const_sw')
      const_m = merge(Stream(False), y, x, name='const_m')
      id_sw = switch(s, {'A': 'A', 'B': 'B'}, name='id_sw')
      id_m_const = merge(s, {'A': 'A', 'B': 'B'}, name='id_m_const')
      unif_sw = switch(s, {'A': x, 'B': x}, name='unif_sw')
      unif_m_const = merge(c, 5, 5, name='unif_m_const')
      id_m_when = merge(c, when(c, c), whennot(c, c), name='id_m_when')
      unif_m_when = merge(c, when(c, y), whennot(c, y), name='unif_m_when')

      # Degenerate boolean ops (&, |, ~):
      bool_and_t = c & True
      bool_and_f = c & False
      bool_or_f = c | False
      bool_or_t = c | True
      bool_idemp = (c & c) | (c | c)
      bool_dneg = ~~c
      bool_inv_c = ~Stream(False)

      # Pure combinational CSE with static inputs across same vs diff scopes:
      dup1 = x + y
      dup2 = x + y
      w1 = when(c, x)
      w2 = when(c, x)
      scoped = box_a(x, Stream(1)) + box_b(x, Stream(1))
      return (
          ite_t,
          ite_f,
          ite_same,
          ite_bool,
          ite_sub_const,
          w_triv,
          w_tag_triv,
          const_sw,
          const_m,
          id_sw,
          id_m_const,
          unif_sw,
          unif_m_const,
          id_m_when,
          unif_m_when,
          bool_and_t,
          bool_and_f,
          bool_or_f,
          bool_or_t,
          bool_idemp,
          bool_dneg,
          bool_inv_c,
          dup1 + dup2,
          merge(c, w1 + w2, whennot(c, y)),
          scoped,
      )

    raw_sg = build_graph(static_prog, semantics=Semantics.SYNC, optimize=False)
    opt_sg = build_graph(static_prog, semantics=Semantics.SYNC, optimize=True)
    self.assertLess(len(opt_sg.nodes), len(raw_sg.nodes))
    # Only the subclock constant ifthenelse remains; base-clock ones fold away:
    self.assertLen([n for n in opt_sg.nodes if n.op == 'ifthenelse'], 1)
    self.assertEmpty([n for n in opt_sg.nodes if n.op == 'switch'])
    self.assertEmpty(
        [n for n in opt_sg.nodes if n.op in {'__and__', '__or__', '__invert__'}]
    )
    self.assertLen([n for n in opt_sg.nodes if n.op == 'merge'], 1)
    # when(c, c) feeds ite_sub_const and when(c, x) is CSE'd (2 total):
    self.assertLen([n for n in opt_sg.nodes if n.op == 'when'], 2)
    self._assert_opt_parity(static_prog, horizon=4)
    self.assertEqual(
        Scheduler.run_graph(
            opt_sg,
            horizon=4,
            clocks=check_clocks(opt_sg),
            streams=(
                'ite_t',
                'w_triv',
                'w_tag_triv',
                'const_sw',
                'const_m',
                'id_sw',
                'id_m_const',
                'unif_sw',
                'unif_m_const',
                'id_m_when',
                'unif_m_when',
            ),
        ),
        {
            'ite_t': [10, 20, 30, 40],
            'w_triv': [10, 20, 30, 40],
            'w_tag_triv': [1, 2, 3, 4],
            'const_sw': [10, 20, 30, 40],
            'const_m': [10, 20, 30, 40],
            'id_sw': ['A', 'B', 'A', 'B'],
            'id_m_const': ['A', 'B', 'A', 'B'],
            'unif_sw': [10, 20, 30, 40],
            'unif_m_const': [5, 5, 5, 5],
            'id_m_when': [True, False, True, False],
            'unif_m_when': [1, 2, 3, 4],
        },
    )

    # Outer `when(Stream(True), seq)` folds away after optimization.
    anchor_g = build_graph(
        lambda: when(Stream(True), Stream([10, 20])), semantics=Semantics.SYNC
    )
    self.assertEmpty(anchor_g.nodes)

  def test_static_optimization_polymorphic_edge_cases(self):
    # 1. Un-sampled polymorphic multi-element static merge/match arms MUST NOT
    #    collapse because each arm consumes its own independent dense cursor.
    def static_merge_self():
      sel = Stream([True, False, True, False], name='sel')
      return merge(sel, sel, sel)

    def static_match_self():
      sel = Stream([True, False, True, False], name='sel')
      with match(sel) as m:
        with case(True):
          m.y = sel
        with case(False):
          m.y = sel
      return m.y

    def static_merge_false():
      sel = Stream([True, False, True, False], name='sel')
      return merge(sel, sel, False)

    def static_tagged_match_self():
      tag = Stream(['A', 'B', 'A', 'B'], name='tag')
      with match(tag) as m:
        with case('A'):
          m.y = tag
        with case('B'):
          m.y = tag
      return m.y

    def static_merge_same_seq():
      c = Stream([True, False, True, False], name='c')
      seq = Stream([10, 20, 30, 40], name='seq')
      return merge(c, seq, seq)

    # 2. Sound folds matching unoptimized execution in SYNC and KAHN:
    def static_switch_self_and_false():
      sel = Stream([True, False, True, False], name='sel')
      return switch(sel, sel, sel), switch(sel, sel, False)

    def static_merge_dedup_tautological():
      c = Stream([True, False, True, False], name='c')
      seq = Stream([10, 20, 30, 40], name='seq')
      # _prepare_mux shares one arm port for {True, False}; tautological.
      return merge(c, {True: seq, False: seq})

    def subclock_static_preservation():
      c = Stream([True, False, True, False], name='c')
      seq = Stream([10, 20, 30, 40], name='seq')
      b = Stream([True, False, False, True], name='b')
      with match(c) as m:
        with case(True):
          # Subclocked nodes must NOT drop Absent/EOF onto raw polymorphic seq.
          m.inv_c = ~Stream(False)
          m.inv_d = ~~b
          m.m_c = merge(b, True, False)
          m.m_s = merge(b, 5, 5)
          m.w_triv = when(Stream(True), seq)
          m.add_l = seq + 1
          m.add_r = seq + 1  # CSE-eligible (same subclock `. on c`)
        with case(False):
          m.inv_c = False
          m.inv_d = False
          m.m_c = False
          m.m_s = 0
          m.w_triv = 0
          m.add_l = seq + 1  # Distinct subclock `. on not c` (no CSE w/ above!)
          m.add_r = 0
      return (
          m.inv_c,
          m.inv_d,
          m.m_c,
          m.m_s,
          m.w_triv,
          m.add_l,
          m.add_r,
          seq + 1,  # Base-clock `.` (distinct from both subclocks!)
      )

    def bool_absorb_identity_symmetry():
      c = Stream([True, False, True, False], name='c')
      wf = when(c, False)
      wt = when(c, True)
      return (
          wf & Stream(False),
          Stream(False) & wf,
          wt & Stream(True),
          Stream(True) & wt,
          wt | Stream(True),
          Stream(True) | wt,
          wf | Stream(False),
          Stream(False) | wf,
      )

    # 3. Nested anchored dynamic `when(Stream(True), raw)` selectors (2 & 3
    #    levels) with all-static arm bodies must keep `node.arms` aligned with
    #    the inner `merge` selector after `when(Stream(True), ...)` folds.
    def nested_anchored_2lvl():
      outer = Stream([True, False, True, False], name='outer')
      anchored = when(
          Stream(True), [True, True, False, True], name='anchored'
      )
      with match(outer) as mo:
        with case(True):
          with match(anchored) as mi:
            with case(True):
              mi.y = Stream([1, 2, 3, 4]) + 10
            with case(False):
              mi.y = 0
          mo.y = mi.y
        with case(False):
          mo.y = 99
      return mo.y

    def nested_anchored_3lvl():
      s0 = Stream([True, True, False, True, True, False], name='s0')
      s1 = when(
          Stream(True), [True, False, True, True, False, True], name='s1'
      )
      s2 = when(
          Stream(True), [True, True, True, False, True, True], name='s2'
      )
      with match(s0) as m0:
        with case(True):
          with match(s1) as m1:
            with case(True):
              with match(s2) as m2:
                with case(True):
                  m2.y = Stream([10, 20, 30]) + 1
                with case(False):
                  m2.y = -2
              m1.y = m2.y
            with case(False):
              m1.y = -1
          m0.y = m1.y
        with case(False):
          m0.y = -9
      return m0.y

    cases = (
        (static_merge_self, 4, [True, True, False, False]),
        (static_match_self, 4, [True, True, False, False]),
        (static_merge_false, 4, [True, False, False, False]),
        (static_tagged_match_self, 4, ['A', 'A', 'B', 'B']),
        (static_merge_same_seq, 4, [10, 10, 20, 20]),
        (
            static_switch_self_and_false,
            4,
            ([True, False, True, False], [True, False, True, False]),
        ),
        (static_merge_dedup_tautological, 4, [10, 20, 30, 40]),
        (nested_anchored_2lvl, 4, [11, 99, 0, 99]),
        (nested_anchored_3lvl, 6, [11, -1, -9, -2, -1, -9]),
    )
    for prog_fn, horizon, expected in cases:
      for sem in (Semantics.SYNC, Semantics.KAHN):
        for opt in (False, True):
          with self.subTest(prog=prog_fn.__name__, semantics=sem, optimize=opt):
            g = build_graph(prog_fn, semantics=sem, optimize=opt)
            clks = check_clocks(g)
            low = lower_resets(g, clks)
            self.assertIs(lower_resets(low, check_clocks(low)), low)
            self.assertEqual(
                Scheduler.run_graph(g, horizon=horizon, clocks=clks),
                expected,
            )

    for prog_fn in (
        subclock_static_preservation,
        bool_absorb_identity_symmetry,
    ):
      self._assert_opt_parity(prog_fn, horizon=4)

    # Operand-order symmetry inside bool_absorb_identity_symmetry: all 8 fold.
    g_sym = build_graph(
        bool_absorb_identity_symmetry, semantics=Semantics.SYNC, optimize=True
    )
    self.assertEmpty(
        [n for n in g_sym.nodes if n.op in {'__and__', '__or__'}]
    )

    # Same-subclock CSE keeps `add_l` / `add_r` shared (3 distinct `__add__`
    # nodes: base `.`, `. on c`, and `. on not c`).
    g_sub = build_graph(
        subclock_static_preservation, semantics=Semantics.SYNC, optimize=True
    )
    self.assertLen([n for n in g_sub.nodes if n.op == '__add__'], 3)

    # 4. Probing a dead arm-local all-static node via `streams=` preserves its
    #    enclosing `ArmScope.cond` producer through DCE.
    def dead_local_static_in_match():
      c = counter(0) % 2 == 0
      with match(c) as m:
        with case(True):
          dead_local = Stream([10, 20, 30]) + 5
          dead_local.name = 'dead_local'
          m.out = 1
        with case(False):
          m.out = 0
      return m.out

    sync_dead = Scheduler.run(
        dead_local_static_in_match,
        horizon=4,
        semantics=Semantics.SYNC,
        streams=('dead_local',),
        optimize=True,
    )['dead_local']
    self.assertEqual(_norm_trace(sync_dead), [15, '<A>', 25, '<A>'])
    self.assertEqual(
        Scheduler.run(
            dead_local_static_in_match,
            horizon=4,
            semantics=Semantics.KAHN,
            streams=('dead_local',),
            optimize=True,
        ),
        {'dead_local': [15, 25]},
    )

    # 5. Unhandled static tags in tagged `merge`/`switch` preserve `KeyError`.
    def bad_switch_id():
      return switch(Stream(['A', 'C']), {'A': 'A', 'B': 'B'})

    def bad_merge_unif():
      return merge(Stream(['A', 'C']), {'A': 5, 'B': 5})

    def bad_switch_const():
      return switch(Stream('C'), {'A': 1, 'B': 2})

    for bad_fn in (bad_switch_id, bad_merge_unif, bad_switch_const):
      with self.subTest(bad_tag=bad_fn.__name__):
        self._assert_mux_key_error(bad_fn, horizon=2)

    # 6. `post(...)` early-EOF (`Bottom` in SYNC / shorter length in KAHN) is
    #    preserved identically between optimize=False and optimize=True.
    def post_eof_prog():
      t = counter(0)
      b = (t % 2) == 0
      pb = post(b, 1)
      pt = post(t, 1)
      ps = post(ifthenelse(b, 'A', 'B'), 1)
      c_true = Stream(True)
      return (
          pb & False,
          pb | True,
          pb & True,
          ~~pb,
          when(Stream(True), pt),
          when(pb, t, values=(True, False)),
          ifthenelse(pb, t, t),
          ifthenelse(True, t, pt),
          switch(ps, {'A': t, 'B': t}),
          switch(b, {True: t, False: t + 1, 'extra': pt}),
          merge(pb, when(pb, t), whennot(pb, t)),
          merge(c_true, t, whennot(c_true, pt)),
      )

    self._assert_opt_parity(post_eof_prog, horizon=4)

    # 7. Unhashable / ambiguous-`__bool__` / ambiguous-`__eq__` 1-periodic
    #    static values never crash optimization or clock checking.
    class _Ambiguous:

      def __bool__(self):
        raise ValueError('truth value is ambiguous')

      def __eq__(self, other):
        raise ValueError('equality is ambiguous')

    amb = _Ambiguous()

    def exotic_static_prog():
      unhash = Stream([[1, 2]])
      return (
          ifthenelse(Stream(amb), 1, 2),
          when(Stream(amb), 1),
          switch(Stream(['A', 'A']), {'A': unhash}),
          merge(Stream([True, False]), [1], [1]),
      )

    opt_exotic = build_graph(
        exotic_static_prog, semantics=Semantics.SYNC, optimize=True
    )
    check_clocks(opt_exotic)

  def test_thunk_detection_excludes_varargs_kwonly_and_decorated_callables(
      self,
  ):
    def zero_arg():
      return 42

    def pos_arg(x):
      return x

    def default_arg(x=1):
      return x

    def varargs_fn(*args):
      return args

    def kwargs_fn(**kwargs):
      return kwargs

    def varargs_and_kwargs(*args, **kwargs):
      return (args, kwargs)

    def kwonly_req(*, x):
      return x

    def kwonly_opt(*, x=1):
      return x

    @psyre_comb
    def comb_fn(x):
      return x + 1

    @psyre_sequ
    def sequ_fn(x):
      while True:
        x = yield x

    @psyre_loop
    async def loop_fn(o_stream):
      o_stream.push(1)

    @psyre_node('box')
    def node_fn(x):
      return x

    # Strict zero-argument functions are thunks.
    self.assertTrue(is_thunk(zero_arg))
    self.assertTrue(is_thunk(lambda: 7))
    self.assertEqual(force_thunk(zero_arg), 42)
    self.assertEqual(force_thunk(lambda: 7), 7)

    # Functions with *args, **kwargs, keyword-only args, positional args, or
    # Psyre decorators/primitives must never be misclassified as thunks.
    non_thunks = (
        pos_arg,
        default_arg,
        varargs_fn,
        kwargs_fn,
        varargs_and_kwargs,
        kwonly_req,
        kwonly_opt,
        comb_fn,
        sequ_fn,
        loop_fn,
        node_fn,
        fby,
        when,
        counter,
    )
    for fn in non_thunks:
      with self.subTest(fn=getattr(fn, '__name__', repr(fn))):
        self.assertFalse(is_thunk(fn))
        with self.assertRaises(TypeError):
          force_thunk(fn)
        # Strict first-order enforcement rejects non-thunk callables without
        # invoking them as thunks.
        with self.assertRaisesRegex(TypeError, 'first-order'):
          resolve_stream(fn)

    # First-order check rejects callables in Stream([...]) without calling them.
    called = False

    def spy_varargs(*args):
      nonlocal called
      called = True
      return args

    with self.assertRaisesRegex(TypeError, 'first-order'):
      resolve_stream(spy_varargs)
    self.assertFalse(called)
    with self.assertRaisesRegex(TypeError, 'first-order'):
      Stream([varargs_fn, comb_fn])

  def test_stream_rewrap_aliases_unhashable_and_graph_id_lookup(self):
    # Stream disables hashing because __eq__ builds a dataflow comparison node.
    with self.assertRaisesRegex(TypeError, 'unhashable'):
      hash(Stream())

    # Outside Stage 1: re-wrapping sets an unnamed stream's name once and drops
    # any secondary name rather than overwriting the primary name.
    unnamed = Stream([1, 2, 3])
    self.assertIs(Stream(unnamed, name='first'), unnamed)
    self.assertEqual(unnamed.name, 'first')
    self.assertIs(Stream(unnamed, name='second'), unnamed)
    self.assertEqual(unnamed.name, 'first')

    # Inside Stage 1: re-wrapping records a Graph alias without mutating .name.
    holders: dict[str, Stream] = {}

    def aliased_prog():
      src = Stream([10, 20, 30], name='primary')
      alias_ref = Stream(src, name='alias_a')
      self.assertIs(alias_ref, src)
      self.assertEqual(src.name, 'primary')
      old = src + 0
      holders.update(src=src, old=old)
      return src

    g = build_graph(aliased_prog, optimize=False)
    src = holders['src']
    old = holders['old']
    outside = Stream([99])
    self.assertTrue(g.has_stream(src))
    self.assertFalse(g.has_stream(outside))
    self.assertEqual(g.stream_names(src), ('primary', 'alias_a'))
    g.add_stream_name(src, 'alias_b')
    g.add_stream_name(src, 'alias_a')  # duplicate ignored
    g.add_stream_name(src, 'primary')  # matches primary name, ignored
    g.add_stream_name(src, '')  # empty ignored
    self.assertEqual(g.stream_names(src), ('primary', 'alias_a', 'alias_b'))
    self.assertEqual(
        Scheduler.run_graph(g, horizon=3, streams=('alias_a', 'alias_b')),
        {'alias_a': [10, 20, 30], 'alias_b': [10, 20, 30]},
    )

    # resolve_stream_id resolves live, replaced, dangling-alias, and bool IDs.
    self.assertIs(g.resolve_stream_id(src.id), src)
    self.assertIsNone(g.resolve_stream_id(outside.id))
    self.assertIsNone(g.resolve_stream_id(False))
    self.assertIsNone(g.resolve_stream_id(True))
    g.replace_stream(old, src)
    g.retain_nodes(())
    self.assertFalse(g.has_stream(old))
    self.assertIs(g.resolve_stream_id(old.id), src)

    # Dangling alias targets pointing to a pruned dynamic node are dropped.
    d_holder: dict[str, Stream] = {}

    def dangling_prog():
      x = counter(0, name='x')
      d_holder['x'] = x
      d_holder['y'] = x + 1
      d_holder['z'] = counter(5, name='z')
      return d_holder['z']

    dg = build_graph(dangling_prog, optimize=False)
    dg.replace_stream(d_holder['x'], d_holder['y'])
    self.assertIs(dg.resolve_stream_id(d_holder['x'].id), d_holder['y'])
    dg.retain_nodes([n for n in dg.nodes if n.output is d_holder['z']])
    self.assertIsNone(dg.resolve_stream_id(d_holder['x'].id))
    self.assertIsNone(dg.resolve_stream_id(d_holder['y'].id))
    self.assertIs(dg.resolve_stream_id(d_holder['z'].id), d_holder['z'])

  def test_stateless_node_thunk_preserves_enclosing_reset(self):
    def prog_if():
      with reset(Stream.once(2)):
        return ifthenelse(True, lambda: counter(0), 0)

    def prog_add():
      with reset(Stream.once(2)):
        return Stream(10) + (lambda: counter(0))

    def prog_when():
      c = Stream([True, True, False, True, True])
      with reset(Stream.once(3)):
        return when(c, lambda: counter(0))

    def prog_deferred_reset():
      with reset(lambda: rst):  # pylint: disable=undefined-variable
        out = ifthenelse(True, lambda: counter(0), 0)
      rst = Stream.once(2)
      return out

    g_if = build_graph(prog_if, semantics=Semantics.SYNC, optimize=False)
    if_nodes = [n for n in g_if.nodes if n.op == 'ifthenelse']
    cnt_nodes = [n for n in g_if.nodes if n.op == 'counter']
    self.assertLen(if_nodes, 1)
    self.assertLen(cnt_nodes, 1)
    self.assertEmpty(if_nodes[0].resets)
    self.assertLen(cnt_nodes[0].resets, 1)

    for sem in (Semantics.SYNC, Semantics.KAHN):
      for opt in (False, True):
        for fn, expected in (
            (prog_if, [0, 1, 0, 1, 2]),
            (prog_add, [10, 11, 10, 11, 12]),
            (prog_deferred_reset, [0, 1, 0, 1, 2]),
        ):
          with self.subTest(fn=fn.__name__, semantics=sem, optimize=opt):
            self.assertEqual(
                Scheduler.run(fn, horizon=5, semantics=sem, optimize=opt),
                expected,
            )
    self.assertEqual(
        Scheduler.run(prog_when, horizon=5, semantics=Semantics.SYNC),
        [0, 1, Absent, 0, 1],
    )
    self.assertEqual(
        Scheduler.run(prog_when, horizon=5, semantics=Semantics.KAHN),
        [0, 1, 0, 1],
    )

  def test_thunk_must_resolve_to_non_callable_stream(self):
    class _ConstThunk(LazyThunk):

      def __init__(self, value: Any):
        self._value = value

      def __psyre_force__(self) -> Any:
        return self._value

    s = Stream([1, 2])
    self.assertIs(resolve_stream(s), s)
    self.assertIs(resolve_stream(lambda: s), s)
    self.assertIs(resolve_stream(_ConstThunk(s)), s)
    wrapped = resolve_stream([7, 8])
    self.assertTrue(wrapped.static)
    self.assertEqual(list(wrapped.prefix), [7])
    self.assertEqual(list(wrapped.period), [8])

    def helper(x):
      return x

    # force_thunk rejects callables and nested thunks.
    for bad in (
        lambda: lambda: s,
        lambda: helper,
        lambda: counter,
        lambda: _ConstThunk(s),
        _ConstThunk(lambda: s),
        _ConstThunk(helper),
        _ConstThunk(_ConstThunk(s)),
    ):
      with self.subTest(case='force_thunk', bad=repr(bad)):
        with self.assertRaisesRegex(TypeError, 'non-callable'):
          force_thunk(bad)

    def make_zero_arg(value: Any) -> Any:
      return lambda: value

    # resolve_stream rejects any thunk resolving to a non-Stream value.
    for bad_val in (42, False, 'A', None, [1, 2], (s,)):
      with self.subTest(case='non_stream_lambda', bad_val=repr(bad_val)):
        with self.assertRaisesRegex(TypeError, 'Stream instance'):
          resolve_stream(make_zero_arg(bad_val))
      with self.subTest(case='non_stream_lazy_thunk', bad_val=repr(bad_val)):
        with self.assertRaisesRegex(TypeError, 'Stream instance'):
          resolve_stream(_ConstThunk(bad_val))

    for bad_prog in (
        lambda: fby(0, lambda: 1),
        lambda: ifthenelse(True, lambda: False, 0),
        lambda: fby(0, lambda: counter),
        lambda: fby(0, lambda: lambda: s),
    ):
      with self.subTest(case='build_graph', prog=repr(bad_prog)):
        with self.assertRaisesRegex(
            TypeError, 'Stream instance|non-callable'
        ):
          build_graph(bad_prog, optimize=False)

    def bad_reset():
      with reset(lambda: False):
        return counter(0)

    def bad_nested_reset():
      rst = Stream([False, True])
      with reset(lambda: lambda: rst):
        return counter(0)

    def bad_match_selector():
      with match(lambda: 'A') as m:
        with case('A'):
          m.o = 1
      return m.o

    for bad_scope in (bad_reset, bad_match_selector):
      with self.subTest(scope=bad_scope.__name__):
        with self.assertRaisesRegex(TypeError, 'Stream instance'):
          build_graph(bad_scope, optimize=False)
    with self.assertRaisesRegex(TypeError, 'Nested Stage-1 thunks'):
      build_graph(bad_nested_reset, optimize=False)

  def test_static_stream_copies_caller_lists_and_prefix_sum_is_out_of_place(
      self,
  ):
    # Caller list mutations must not leak into static Stream prefix/period.
    prefix_list = [1, 2]
    period_list = [3, 4]
    isolated = Stream(prefix_list, period=period_list)
    prefix_list.append(99)
    period_list[0] = 99
    self.assertEqual(isolated.trace(6), [1, 2, 3, 4, 3, 4])

    # Stream accepts arbitrary non-sentinel values without immutability checks,
    # and prefix_sum uses out-of-place `+` rather than mutating via `+=`.
    def list_prefix():
      rst = Stream([False, False, True, False])
      items = Stream([[1], [2], [3], [4]])
      with reset(rst):
        return prefix_sum(items, Stream([[]]))

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(prefix_sem=sem):
        g = build_graph(list_prefix, semantics=sem)
        expected = [[1], [1, 2], [3], [3, 4]]
        self.assertEqual(Scheduler.run_graph(g, horizon=4), expected)
        self.assertEqual(Scheduler.run_graph(g, horizon=4), expected)

  def test_streams_iterator_and_folded_op_name_roots(self):
    def prog_counter():
      return counter(10, name='x') + 1

    # Single-use iterator passed to streams= must work under optimize=True.
    for opt in (False, True):
      with self.subTest(optimize=opt):
        self.assertEqual(
            Scheduler.run(
                prog_counter,
                horizon=3,
                semantics=Semantics.SYNC,
                streams=iter(['x']),
                optimize=opt,
            ),
            {'x': [10, 11, 12]},
        )

    bad_streams: Any = 'x'
    with self.assertRaisesRegex(TypeError, 'not a bare str'):
      Scheduler.run(prog_counter, horizon=2, streams=bad_streams)

    # Folded default-op-named streams requested via streams= survive DCE/CSE.
    def prog_folded_ite():
      c = counter(0) == 0
      a = counter(1, name='a')
      return ifthenelse(c, a, a, name='ifthenelse')

    def prog_folded_switch():
      c = counter(0) == 0
      a = counter(10, name='a')
      return switch(c, a, a, name='switch')

    @psyre_node('sub')
    def sub_switch():
      c = counter(0) == 0
      v = counter(5, name='v')
      return switch(c, v, v, name='switch')

    for prog_fn, stream_name, expected in (
        (prog_folded_ite, 'ifthenelse', [1, 2, 3]),
        (prog_folded_switch, 'switch', [10, 11, 12]),
        (sub_switch, 'sub/switch', [5, 6, 7]),
        (sub_switch, 'switch', [5, 6, 7]),
        (
            lambda: ifthenelse(True, [1, 2, 3], [4, 5, 6], name='ifthenelse'),
            'ifthenelse',
            [1, 2, 3],
        ),
        (
            lambda: when(Stream(True), [10, 20, 30], name='when'),
            'when',
            [10, 20, 30],
        ),
    ):
      with self.subTest(stream_name=stream_name):
        self.assertEqual(
            Scheduler.run(prog_fn, horizon=3, streams=[stream_name]),
            {stream_name: expected},
        )

  def test_reentrant_build_and_scheduler_are_forbidden(self):
    ready_graph = build_graph(lambda: counter(0), semantics=Semantics.SYNC)
    reentrant_errors: list[str] = []

    def try_reenter_in_stage1_build():
      with self.assertRaisesRegex(RuntimeError, 'Re-entrant build_graph'):
        build_graph(lambda: counter(0))
      with self.assertRaisesRegex(RuntimeError, 'Re-entrant Scheduler'):
        Scheduler.run(lambda: counter(0), horizon=1)
      with self.assertRaisesRegex(RuntimeError, 'Re-entrant Scheduler'):
        Scheduler.run_graph(ready_graph, horizon=1)
      return Stream([1, 2])

    self.assertEqual(
        Scheduler.run(try_reenter_in_stage1_build, horizon=2), [1, 2]
    )

    @psyre_comb
    def reenter_stage2(x: int) -> int:
      for label, fn in (
          ('run', lambda: Scheduler.run(lambda: counter(0), horizon=1)),
          ('run_graph', lambda: Scheduler.run_graph(ready_graph, horizon=1)),
          ('build_graph', lambda: build_graph(lambda: counter(0))),
      ):
        try:
          fn()
        except RuntimeError as err:
          reentrant_errors.append(f'{label}:{err}')
      return x + 1

    # Even after catching all re-entry errors, outer active_scheduler remains
    # attached and post() EOF termination finishes cleanly without hanging.
    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        reentrant_errors.clear()
        out = Scheduler.run(
            lambda: post(reenter_stage2(counter(0)), 1),
            horizon=3,
            semantics=sem,
        )
        self.assertLen(reentrant_errors, 9)
        self.assertEqual(
            out[:2] if sem == Semantics.SYNC else out,
            [2, 3],
        )

  def test_push_unblocks_waiting_task_without_quadratic_deadlock_scan(self):
    def cyclic_chain():
      n = 30
      x = counter(0)
      prev = fby(0, lambda: prev + 1)  # pylint: disable=undefined-variable
      cur = x + prev
      for _ in range(n):
        cur = cur + 1
      return cur

    g = build_graph(cyclic_chain, semantics=Semantics.SYNC, optimize=False)
    orig_check = Scheduler._check_deadlock  # pylint: disable=protected-access
    full_scans = 0

    def counting_check(sched: Scheduler) -> None:
      nonlocal full_scans
      if sched._tasks and len(sched._blocked) >= len(sched._tasks):  # pylint: disable=protected-access
        full_scans += 1
      orig_check(sched)

    Scheduler._check_deadlock = counting_check  # type: ignore[method-assign]  # pylint: disable=protected-access
    try:
      res = Scheduler.run_graph(g, horizon=10)
    finally:
      Scheduler._check_deadlock = orig_check  # type: ignore[method-assign]  # pylint: disable=protected-access
    self.assertEqual(res[:4], [30, 32, 34, 36])
    # Without unblocking woken waiters inside Stream.push, full_scans is O(N*H).
    self.assertLessEqual(full_scans, 10)

  def test_static_stream_validation_prefix_period_and_indices(self):
    # Absent / Bottom rejected in BOTH prefix and period.
    for sentinel, name in ((Absent, 'Absent'), (Bottom, 'Bottom')):
      for section, make_bad in (
          ('period', lambda x=sentinel: Stream(x)),
          ('prefix', lambda x=sentinel: Stream([x, 1])),
          ('period', lambda x=sentinel: Stream([1, x])),
          ('prefix', lambda x=sentinel: Stream([x], period=[1])),
          ('period', lambda x=sentinel: Stream([1], period=[x])),
          ('period', lambda x=sentinel: Stream([], period=x)),
          ('period', lambda x=sentinel: Stream(period=x)),
          ('prefix', lambda x=sentinel: Stream.periodic([x], [1])),
          ('period', lambda x=sentinel: Stream.periodic([1], [x])),
      ):
        with self.assertRaisesRegex(
            TypeError, f'{section} cannot contain {name}'
        ):
          make_bad()

    # Empty static value lists and empty periods are rejected eagerly.
    with self.assertRaisesRegex(ValueError, 'at least one value'):
      Stream([])
    for make_empty_period in (
        lambda: Stream([], period=[]),
        lambda: Stream([1], period=[]),
        lambda: Stream(period=[]),
        lambda: Stream.periodic([], []),
        lambda: Stream.periodic([1], []),
    ):
      with self.assertRaisesRegex(ValueError, 'non-empty period'):
        make_empty_period()

    # Unknown kwargs rejected even on existing Stream pass-through.
    existing = Stream([1, 2])
    with self.assertRaisesRegex(TypeError, 'unexpected keyword argument'):
      Stream(1, bogus=2)
    with self.assertRaisesRegex(TypeError, 'unexpected keyword argument'):
      Stream(existing, periods=[3])

    # Non-negative index and static requirement checks on static_value_at().
    with self.assertRaisesRegex(ValueError, 'non-negative'):
      existing.static_value_at(-1)
    with self.assertRaisesRegex(RuntimeError, 'requires a static Stream'):
      Stream().static_value_at(0)

    # Empty prefix with non-empty period remains valid.
    per = Stream.periodic([], [4, 5])
    self.assertEqual(per.trace(5), [4, 5, 4, 5, 4])

    # Stream.once validates non-negative integer indices and works outside
    # Stage 1 as a static periodic impulse stream.
    self.assertEqual(Stream.once(0).trace(4), [True, False, False, False])
    self.assertEqual(Stream.once(2).trace(4), [False, False, True, False])
    for bad_k in (-1, True, False, 1.5, '0'):
      with self.assertRaisesRegex(ValueError, 'Stream.once index'):
        Stream.once(bad_k)  # type: ignore[arg-type]

    # Multi-argument variadic `Stream(a1, ..., an)` construction is rejected;
    # explicit list `Stream([a1, ..., an])` remains supported.
    for bad_call in (
        lambda: Stream(1, 2),  # type: ignore[call-arg]
        lambda: Stream(1, 2, 3),  # type: ignore[call-arg]
        lambda: Stream(False, True),  # type: ignore[call-arg]
        lambda: Stream(1, 2, period=[3]),  # type: ignore[call-arg]
    ):
      with self.assertRaises(TypeError):
        bad_call()
    s_list = Stream([1, 2, 3])
    self.assertEqual(s_list.trace(5), [1, 2, 3, 3, 3])
    self.assertEqual(list(s_list.prefix), [1, 2])
    self.assertEqual(list(s_list.period), [3])

  def test_resettable_static_streams_and_once_semantics(self):
    def reset_scopes():
      rst = Stream([False, False, True, False, False, True, False])
      outer_seq = Stream([10, 20, 30, 40, 50, 60, 70], name='outer_seq')
      with reset(rst):
        seq_r = Stream([10, 20, 30, 40], name='seq_r')
        per_r = Stream.periodic([1], [2, 3], name='per_r')
        once_r = Stream.once(1, name='once_r')
        const_r = Stream(42, name='const_r')
        nested_cnt = None
        with reset([False, False, True, False]):
          nested_cnt = counter(start=0)
        reused_outer = outer_seq + 0
      return seq_r, per_r, once_r, const_r, reused_outer, nested_cnt

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(case='reset_scopes', semantics=sem):
        g = build_graph(reset_scopes, semantics=sem)
        # 1-periodic `const_r` stays node-free (not wrapped in static_stream).
        const_stream = g.resolve_named_streams(['const_r'])['const_r']
        self.assertIsNone(g.producer(const_stream))
        clks = check_clocks(g)
        low = lower_resets(g, clks)
        self.assertIs(lower_resets(low, check_clocks(low)), low)
        # Verify repeated runs are clean and independent.
        expected = (
            [10, 20, 10, 20, 30, 10, 20],
            [1, 2, 1, 2, 3, 1, 2],
            [False, True, False, True, False, False, True],
            [42, 42, 42, 42, 42, 42, 42],
            [10, 20, 30, 40, 50, 60, 70],
            # Inner reset `[F, F, T, F]` resets when outer `rst` fires at t=2
            # and t=5, so the inner reset fires at t=4 (two ticks after t=2).
            [0, 1, 0, 1, 0, 0, 1],
        )
        self.assertEqual(
            Scheduler.run_graph(g, horizon=7, clocks=clks), expected
        )
        self.assertEqual(
            Scheduler.run_graph(g, horizon=7, clocks=clks), expected
        )

    # Manual subclocked reset (`node.arms == ()`) resets on `. on c`.
    def subclock_manual_reset():
      c = Stream([True, False, True, True, False, True])
      r = Stream([False, False, True, False, False, False])
      with reset(when(c, r)):
        s = Stream.periodic([], [1, 2, 3], name='sub_per')
      return s

    sync_sub = Scheduler.run(
        subclock_manual_reset, horizon=6, semantics=Semantics.SYNC
    )
    kahn_sub = Scheduler.run(
        subclock_manual_reset, horizon=6, semantics=Semantics.KAHN
    )
    self.assertEqual(sync_sub, [1, Absent, 1, 2, Absent, 3])
    self.assertEqual(kahn_sub, [1, 1, 2, 3])

    # Manual subclocked `static_stream` consumed directly inside a matching
    # `match(c)` arm must not be double-wrapped with a duplicate `when(c, s)`
    # (for both static and dynamic selector `c`).
    def subclock_manual_reset_in_match(dynamic_sel: bool):
      def prog():
        if dynamic_sel:
          c = (counter(start=0) % 2) == 0
        else:
          c = Stream([True, False, True, False, True, False])
        r = Stream([False, False, True, False, False, False])
        with reset(when(c, r)):
          s = Stream.periodic([], [10, 20, 30], name='sub_s')
        with match(c) as m:
          with case(True):
            m.y = s
          with case(False):
            m.y = -1
        return m.y

      return prog

    for dynamic_sel in (False, True):
      for sem in (Semantics.SYNC, Semantics.KAHN):
        with self.subTest(
            case='manual_reset_in_match', dynamic=dynamic_sel, sem=sem
        ):
          self.assertEqual(
              Scheduler.run(
                  subclock_manual_reset_in_match(dynamic_sel),
                  horizon=6,
                  semantics=sem,
              ),
              [10, -1, 10, -1, 20, -1],
          )

    # Resettable `static_stream` tagged selector with an unhandled tag keeps
    # `KeyError` under `optimize=True` instead of folding the mux away.
    for fn_name, op_fn in (('merge', merge), ('switch', switch)):
      def unhandled_resettable_tag(mux_op=op_fn):
        rst = Stream([False, False, False])
        with reset(rst):
          sel = Stream(['A', 'C'], name='sel')
        return mux_op(sel, {'A': 5, 'B': 5})

      for opt in (False, True):
        with self.subTest(
            case='unhandled_resettable_tag', op=fn_name, optimize=opt
        ):
          self._assert_mux_key_error(
              unhandled_resettable_tag, horizon=2, optimize=opt
          )

    # Bare list literal assigned inside `match` arm under supersampled `reset`.
    def supersampled_reset_match_literal():
      c = Stream([True, False, True, True, False, True])
      r = Stream([False, True, False, False, True, False])
      with reset(r):
        with match(c) as m:
          with case(True):
            m.y = [10, 20, 30]
          with case(False):
            m.y = -1
      return m.y

    self._assert_sync_kahn(
        supersampled_reset_match_literal, [10, -1, 10, 20, -1, 10], horizon=6
    )

    # Self-resetting static stream broken by unit-delayed `fby`.
    def self_reset_static():
      rst = fby(False, lambda: s == 20)  # pylint: disable=undefined-variable
      with reset(rst):
        s = Stream([10, 20, 30], name='s')
      return s

    self._assert_sync_kahn(
        self_reset_static, [10, 20, 10, 20, 10, 20], horizon=6
    )

  def test_stream_abs_tuple_subclass_gc_counter_horizon_and_exports(self):
    # Stream.__abs__ works and is CSE-deduplicated under optimize=True.
    def abs_prog():
      s = Stream([-3, 0, 5, -2])
      return abs(s), abs(s)

    for sem in (Semantics.SYNC, Semantics.KAHN):
      g = build_graph(abs_prog, semantics=sem)
      self.assertLen([n for n in g.nodes if n.op == '__abs__'], 1)
      self.assertEqual(
          Scheduler.run_graph(g, horizon=4),
          ([3, 0, 5, 2], [3, 0, 5, 2]),
      )

    # Plain tuple subclass (not a NamedTuple) does not crash output or match.
    class SubTuple(tuple):
      pass

    def subtuple_prog():
      c = Stream([True, False, True])
      with match(c) as m:
        with case(True):
          m.pair = SubTuple((Stream(1), Stream(2)))
        with case(False):
          m.pair = SubTuple((Stream(3), Stream(4)))
      return SubTuple((m.pair[0], m.pair[1]))

    g_sub = build_graph(subtuple_prog)
    self.assertTrue(all(n.merge_arms is None for n in g_sub.nodes))
    self.assertEqual(
        Scheduler.run_graph(g_sub, horizon=3), ([1, 3, 1], [2, 4, 2])
    )

    # counter(..., max_val=...) keeps clamping after a dynamic None tick.
    self._assert_sync_kahn(
        lambda: counter(0, max_val=Stream([5, None, 0, 0, 5])),
        [0, 1, 0, 0, 1],
        horizon=5,
    )

    # Scheduler rejects negative, bool, or non-int horizons uniformly.
    for bad_h in (-1, True, False, 1.5):
      with self.assertRaisesRegex(ValueError, 'horizon'):
        Scheduler.run(lambda: Stream(1), horizon=bad_h)  # type: ignore[arg-type]

    # Dynamic @psyre_comb / @psyre_sequ wrappers do not leak via global caches,
    # and unwrap_inst_func accepts unhashable callable objects.
    def make_wrappers():
      def raw_comb(x):
        return x + 1

      def raw_sequ(x):
        while True:
          x = yield x

      return psyre_comb(raw_comb), psyre_sequ(raw_sequ)

    w_comb, w_sequ = make_wrappers()
    ref_comb, ref_sequ = weakref.ref(w_comb), weakref.ref(w_sequ)
    del w_comb, w_sequ
    gc.collect()
    self.assertIsNone(ref_comb())
    self.assertIsNone(ref_sequ())

    class UnhashableCallable:
      __hash__ = None  # type: ignore[assignment]

      def __call__(self, x):
        return x

    uh = UnhashableCallable()
    self.assertIs(unwrap_inst_func(uh), uh)

    # @psyre_sequ closes inner Python generator on reset and task exit.
    closed: list[str] = []

    @psyre_sequ
    def closing_sequ(v):
      try:
        while True:
          v = yield v
      finally:
        closed.append('closed')

    def run_closing():
      rst = Stream([False, True, False])
      vals = Stream([1, 2, 3])
      with reset(rst):
        return closing_sequ(vals)

    self.assertEqual(Scheduler.run(run_closing, horizon=3), [1, 2, 3])
    self.assertLen(closed, 2)

    # Canonical package surface & removed aliases (`match`/`automaton` stay
    # submodules so `from psyre import match` never shadows `psyre.match`).
    for sym in (
        'match', 'automaton',
        'check_causality', 'CausalityError',
        'check_clocks', 'ClockError',
        'optimize_graph', 'eliminate_dead_code',
        'EMPTY_WORD', 'StaticSequence', 'StaticWord', 'concat', 'repeat',
        'random_stream',
    ):
      self.assertIn(sym, psyre.__all__)
      self.assertTrue(hasattr(psyre, sym))
    self.assertTrue(hasattr(psyre.match, 'match'))
    self.assertTrue(hasattr(psyre.automaton, 'automaton'))
    for exp_sym in ('Equations', 'psyre_equations', 'Thunk'):
      self.assertFalse(hasattr(psyre, exp_sym))
    self.assertFalse(hasattr(optimization, 'dce'))
    self.assertFalse(hasattr(clock, 'infer_clocks'))

  def test_p1_regressions_step_identity_mux_and_graph_hygiene(self):
    # 1. User @psyre_comb named like builtin ops must not collide in CSE,
    #    algebraic folding, or is_stateful; user @psyre_loop async generators
    #    are detected as stateful and resettable.
    @psyre_comb
    def max(a, b):  # pylint: disable=redefined-builtin
      return a - b

    @psyre_comb
    def scan(x):
      return x + 10

    def _make_custom_ite():
      @psyre_comb
      def ifthenelse(c, t, f):  # pylint: disable=redefined-outer-name
        del c, t, f
        return 99

      return ifthenelse

    custom_ite = _make_custom_ite()

    @psyre_loop
    async def acc_loop(o_stream, i_stream):
      total = 0
      while True:
        start = yield
        if start is Absent:
          i_stream.skip_absent()
          o_stream.push(Absent)
          continue
        if start:
          total = 0
        val = await pull(i_stream)
        total += val
        o_stream.push(total)

    def user_op_prog():
      a = Stream([10, 20, 30])
      b = Stream([3, 5, 7])
      r = Stream([False, False, True])
      with reset(r):
        loop_out = acc_loop(b)
      return (
          Stream.max(a, b),
          max(a, b),
          scan(b),
          custom_ite(True, a, b),
          loop_out,
      )

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(case='user_ops', semantics=sem):
        bi_max, usr_max, usr_scan, usr_ite, usr_loop = Scheduler.run(
            user_op_prog, horizon=3, semantics=sem, optimize=True
        )
        self.assertEqual(bi_max, [10, 20, 30])
        self.assertEqual(usr_max, [7, 15, 23])
        self.assertEqual(usr_scan, [13, 15, 17])
        self.assertEqual(usr_ite, [99, 99, 99])
        self.assertEqual(usr_loop, [3, 8, 7])

    # 2. Non-exhaustive dynamic mux folds preserve KeyError, and exact types
    #    (int/float/IntEnum) are preserved by identity mux folding.
    for opt in (False, True):
      for prog_fn in (
          lambda: switch(counter(0), {0: 0, 1: 1}),
          lambda: switch(counter(0), {0: 5, 1: 5}),
          lambda: merge(counter(0), {0: 0, 1: 1}),
      ):
        self._assert_mux_key_error(prog_fn, horizon=3, optimize=opt)

    class _Tag(enum.IntEnum):
      ZERO = 0
      ONE = 1

    for sel_vals, mapping, expected_type in (
        ([0.0, 1.0], {0: 0, 1: 1}, int),
        ([0, 1], {0: 0.0, 1: 1.0}, float),
        ([_Tag.ZERO, _Tag.ONE], {0: 0, 1: 1}, int),
    ):
      with self.subTest(sel_type=type(sel_vals[0]).__name__):
        res = Scheduler.run(
            lambda sv=sel_vals, mp=mapping: switch(Stream(sv), mp),
            horizon=2,
            semantics=Semantics.SYNC,
            optimize=True,
        )
        self.assertEqual([type(v) for v in res], [expected_type] * 2)

    # 3. same_stream_or_const never calls Stream.__eq__ on dict-of-Stream,
    #    rejects non-bool sequence __eq__, and preserves signed float zero.
    class _SeqEq:

      def __eq__(self, other: object) -> Any:
        del other
        return [False, False]

    self.assertFalse(same_stream_or_const(Stream(_SeqEq()), Stream(_SeqEq())))
    self.assertFalse(same_stream_or_const(Stream(-0.0), Stream(0.0)))

    def dict_of_stream_prog():
      sel = Stream([True, False])
      a = counter(0)
      b = counter(10)
      return ifthenelse(sel, {'s': a}, {'s': b})

    opt_dict_g = build_graph(dict_of_stream_prog, optimize=True)
    self.assertTrue(opt_dict_g.is_finalized)

    def match_dict_of_stream_prog():
      sel = Stream([True, False])
      a = counter(0)
      b = counter(10)
      with match(sel) as m:
        with case(True):
          m.out = {'s': a}
        with case(False):
          m.out = {'s': b}
      return m.out

    with self.assertRaisesRegex(
        TypeError, r'Unsupported match\(\) output container dict'
    ):
      build_graph(match_dict_of_stream_prog, optimize=True)

    # 4. Logger does not propagate to root stderr (avoids duplicate logs).
    self.assertFalse(logger.propagate)

    # 5. buffer positional `size` works via prepare_buffer & rejects invalid.
    def buf_pos():
      ip = Stream([True, True, False, False])
      op = Stream([False, False, True, True])
      data = when(ip, [10, 20, 30, 40])
      return buffer(data, ip, op, 2)

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(case='buffer_pos_size', semantics=sem):
        out = Scheduler.run(buf_pos, horizon=4, semantics=sem)
        expected = (
            [Absent, Absent, 10, 20] if sem == Semantics.SYNC else [10, 20]
        )
        self.assertEqual(out, expected)
    with self.assertRaisesRegex(ValueError, 'buffer size must be >= 1'):
      build_graph(lambda: buffer(when(True, 1), True, True, 0))
    with self.assertRaisesRegex(ValueError, 'buffer size must be >= 1'):
      build_graph(lambda: buffer(when(True, 1), True, True, True))

  def test_random_stream_defined_in_math_and_not_cse_eliminated(self):
    self.assertIs(psyre.random_stream, psyre_math.random_stream)
    self.assertFalse(hasattr(sync_primitives, 'random_stream'))
    self.assertFalse(hasattr(kahn_primitives, 'random_stream'))

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        g = build_graph(
            lambda: (Stream.random(), psyre_math.random_stream()),
            semantics=sem,
            optimize=True,
        )
        self.assertLen(g.nodes, 2)
        self.assertEqual([n.op for n in g.nodes], ['random_stream'] * 2)
        r1, r2 = Scheduler.run_graph(g, horizon=5, clocks=check_clocks(g))
        self.assertLen(r1, 5)
        self.assertLen(r2, 5)
        self.assertTrue(all(0.0 <= v < 1.0 for v in r1 + r2))
        self.assertGreater(len(set(r1)), 1)
        self.assertNotEqual(r1, r2)

  def test_static_word_grammar_and_algebra(self):
    self.assertEqual(StaticWord(), EMPTY_WORD)
    self.assertEqual(StaticWord([]), EMPTY_WORD)
    self.assertFalse(bool(EMPTY_WORD))
    self.assertEqual(EMPTY_WORD.length, 0)

    # Atom vs list unpacking and zero/one/multi repetition fusion.
    atom_list = StaticWord.atom([1, 2])
    self.assertEqual(atom_list.length, 1)
    self.assertEqual(atom_list[0], [1, 2])
    unpacked = StaticWord([1, 2])
    self.assertIsInstance(unpacked, LiteralWord)
    self.assertEqual(list(unpacked), [1, 2])
    self.assertIs(StaticWord(unpacked), unpacked)

    self.assertIs(repeat([1, 2], 0), EMPTY_WORD)
    self.assertEqual(repeat([1, 2], 1), unpacked)
    self.assertIs(unpacked * 0, EMPTY_WORD)
    self.assertIs(0 * unpacked, EMPTY_WORD)
    r3 = unpacked * 3
    self.assertIsInstance(r3, RepeatWord)
    self.assertEqual(r3.count_n, 3)
    self.assertEqual(list(r3), [1, 2, 1, 2, 1, 2])
    r6 = 2 * r3
    self.assertIsInstance(r6, RepeatWord)
    self.assertEqual(r6.count_n, 6)
    self.assertEqual(r6.length, 12)

    # Reject bool and negative repetition counts.
    for bad_k in (-1, -10, True, False, 1.5, '2'):
      with self.assertRaises((TypeError, ValueError)):
        repeat(0, bad_k)  # type: ignore[arg-type]

    # Concatenation flattens and merges adjacent literals and equal repeats.
    merged_lit = StaticWord([1, 2]) + [3, 4]
    self.assertIsInstance(merged_lit, LiteralWord)
    self.assertEqual(list(merged_lit), [1, 2, 3, 4])
    merged_rep = repeat(False, 5) + False + repeat(False, 3)
    self.assertIsInstance(merged_rep, RepeatWord)
    self.assertEqual(merged_rep.count_n, 9)
    self.assertEqual(merged_rep, repeat(False, 9))
    self.assertEqual(
        [True] + repeat(False, 2), StaticWord([True, False, False])
    )
    self.assertEqual(
        True + repeat(False, 2), StaticWord([True, False, False])
    )
    self.assertEqual(concat(), EMPTY_WORD)

    # Symbolic slicing, indexing, negative indexing, and O(|AST|) `in`.
    big = 10**8
    pattern = repeat(False, big) + [True, False] + repeat(True, 5)
    self.assertIsInstance(pattern, ConcatWord)
    self.assertEqual(pattern.length, big + 7)
    self.assertFalse(pattern[0])
    self.assertFalse(pattern[big - 1])
    self.assertTrue(pattern[big])
    self.assertFalse(pattern[big + 1])
    self.assertTrue(pattern[-1])
    self.assertFalse(pattern[-6])
    self.assertEqual(
        list(pattern[big - 2 : big + 3]),
        [False, False, True, False, True],
    )
    self.assertIn(True, pattern)
    self.assertNotIn(42, pattern)
    self.assertTrue(StaticSequence(pattern, StaticWord(False)).is_bool())
    self.assertFalse(
        StaticSequence(repeat(0, big) + 1, StaticWord(0)).is_bool()
    )

    # Type safety, signed zero, NaN, and mutable list identity preservation.
    self.assertNotEqual(StaticWord(0), StaticWord(False))
    self.assertNotEqual(StaticWord(0), StaticWord(0.0))
    self.assertNotEqual(StaticWord(0.0), StaticWord(-0.0))
    self.assertNotEqual(StaticWord(float('nan')), StaticWord(float('nan')))
    self.assertFalse(
        Stream.periodic([], [True, 1]).is_one_periodic
    )
    self.assertFalse(
        Stream.periodic([], [0.0, -0.0]).is_one_periodic
    )
    self.assertFalse(
        Stream.periodic([], [float('nan'), float('nan')]).is_one_periodic
    )
    a_mut, b_mut = [1], [1]
    self.assertFalse(Stream.periodic([], [a_mut, b_mut]).is_one_periodic)
    self.assertTrue(Stream.periodic([], [a_mut, a_mut]).is_one_periodic)

    # Reject Stream / LazyThunk side effects without triggering Stream.__eq__.
    dyn1, dyn2 = Stream(), Stream()
    self.assertTrue(static_stream_lib.same_const_value(dyn1, dyn1))
    self.assertFalse(static_stream_lib.same_const_value(dyn1, dyn2))
    self.assertFalse(static_stream_lib.same_const_value([dyn1], [dyn1]))
    self.assertNotIn(dyn1, StaticWord([1, 2]))
    with self.assertRaises(TypeError):
      _ = StaticWord([Absent])
    with self.assertRaises(TypeError):
      _ = StaticWord([Bottom])

  def test_static_sequence_canonicalization_and_o1_once_complexity(self):
    big_k = 10**8
    huge_k = 10**9

    # (v^k)^ω reduces to v^ω; trailing copies of 1-periodic period trim in O(1).
    seq_rep = StaticSequence(
        EMPTY_WORD, repeat(StaticWord([1, 2]), big_k)
    )
    self.assertEqual(seq_rep.period, StaticWord([1, 2]))
    self.assertEqual(
        StaticSequence.from_parts(
            repeat(False, big_k) + True + repeat(False, big_k), [False]
        ),
        StaticSequence.once(big_k),
    )
    self.assertTrue(
        StaticSequence.from_parts(repeat(7, big_k), [7]).is_one_periodic
    )
    self.assertEqual(
        StaticSequence.from_parts(repeat(7, big_k), [7]).const_value, 7
    )

    # Stream.once(k) builds, indexes, traces, formats, analyzes, optimizes,
    # visualizes, and executes in O(1) without expanding k elements.
    s_huge = Stream.once(huge_k)
    assert s_huge.static_sequence is not None
    self.assertEqual(s_huge.prefix.length, huge_k + 1)
    self.assertFalse(s_huge.static_value_at(0))
    self.assertFalse(s_huge.static_value_at(huge_k - 1))
    self.assertTrue(s_huge.static_value_at(huge_k))
    self.assertFalse(s_huge.static_value_at(huge_k + 1))
    self.assertEqual(s_huge.trace(6), [False] * 6)
    self.assertEqual(
        list(s_huge.prefix[huge_k - 2 : huge_k + 1]),
        [False, False, True],
    )
    self.assertIn(str(huge_k), str(s_huge))
    self.assertIn(str(huge_k), s_huge.static_sequence.format_compact())
    self.assertTrue(s_huge.static_sequence.is_bool())

    def big_once_prog():
      pulse = Stream.once(big_k)
      with match(pulse) as m:
        with case(True):
          m.out = 100
        with case(False):
          m.out = counter()
      with reset(Stream([False, False, True, False, False])):
        rst_pulse = Stream.once(big_k, name='rst_big_pulse')
      return pulse, m.out, rst_pulse

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        g = build_graph(big_once_prog, semantics=sem, optimize=True)
        self.assertTrue(g.is_bool_stream(g.output[0]))
        p_tr, m_tr, r_tr = Scheduler.run_graph(
            g, horizon=5, clocks=check_clocks(g)
        )
        self.assertEqual(p_tr, [False] * 5)
        self.assertEqual(m_tr, [0, 1, 2, 3, 4])
        self.assertEqual(r_tr, [False] * 5)

  def test_static_word_symbolic_eq_and_slice_bounds(self):
    n = 10**9

    # Symbolic equality handles mismatched AST splits, common divisors, and
    # rotated repeat phases without expanding elements.
    self.assertEqual(
        repeat(False, n - 1) + [False, True], repeat(False, n) + True
    )
    self.assertNotEqual(repeat(False, n) + 1, repeat(False, n) + 2)
    self.assertEqual(
        StaticSequence.once(n),
        StaticSequence.from_parts(repeat(False, n - 1) + [False, True], False),
    )
    self.assertEqual(repeat([1, 2], 2 * 10**8), repeat([1, 2, 1, 2], 10**8))
    self.assertNotEqual(repeat([1, 2, 1, 1, 2], 6), repeat([1, 2, 1], 10))
    self.assertEqual(
        repeat([0, 1], 10**8),
        concat([0], repeat([1, 0], 10**8 - 1), [1]),
    )
    self.assertNotEqual(
        repeat([0, 1], 10**8),
        concat([0], repeat([1, 0], 10**8 - 1), [2]),
    )
    a_mut, b_mut = [1], [1]
    self.assertEqual(StaticWord([a_mut, a_mut]), StaticWord([a_mut, b_mut]))
    self.assertNotEqual(repeat(float('nan'), n), repeat(float('nan'), n))
    self.assertFalse(static_stream_lib.same_const_value((1, 0.0), (1, -0.0)))

    # Strided/negative slicing and lengths exceeding 2**31 - 1 are rejected.
    word = StaticWord([1, 2, 3, 4])
    for step in (0, 2, -1, -2):
      with self.assertRaisesRegex(ValueError, 'step == 1'):
        _ = word[::step]
    with self.assertRaises(TypeError):
      _ = word[::True]  # type: ignore[misc]
    with self.assertRaisesRegex(ValueError, 'exceeds limit'):
      repeat(False, static_stream_lib.MAX_WORD_LENGTH + 1)
    with self.assertRaisesRegex(ValueError, 'exceeds limit'):
      Stream.once(static_stream_lib.MAX_WORD_LENGTH)
    with self.assertRaisesRegex(ValueError, 'exceeds limit'):
      concat(repeat(False, static_stream_lib.MAX_WORD_LENGTH), True)

  def test_static_word_eq_polynomial_on_nested_repeats_and_dags(self):
    # Both expand identically to `[0, 1] * 125_829_114` (`len = 251_658_228`).
    a = concat([0], repeat([1, 0], 2), [1])
    b = concat([0], repeat([1, 0], 3), [1])
    for _ in range(23):
      a = concat([0, 1], repeat(a, 2), [0, 1])
      b = concat([0, 1], repeat(b, 2), [0, 1])
    lhs = repeat(a, 3)
    pad = (lhs.length - 2 * b.length) // 2
    rhs = concat(repeat(b, 2), repeat([0, 1], pad))
    self.assertEqual(lhs.length, rhs.length)
    self.assertEqual(lhs, rhs)
    self.assertNotEqual(lhs, concat(rhs[:-1], [2]))

    # Identical deep RepeatWord/ConcatWord DAGs with distinct node identities
    # (`len ~ 10**8`, `O(depth)` AST nodes). Note bare `concat(lit, lit)`
    # flattens literals, so `repeat(..., 2)` keeps AST size logarithmic.
    u, v = StaticWord([0, 1]), StaticWord([0, 1])
    for idx in range(1, 25):
      u = concat([idx], repeat(u, 2), [idx + 1])
      v = concat([idx], repeat(v, 2), [idx + 1])
    self.assertGreater(u.length, 5 * 10**7)
    self.assertEqual(u, v)

    # Non-periodic deep symbolic words (`len > 10**8`) with different AST splits
    # (`_struct_eq` fails immediately, exercising SLP recompression directly).
    self.assertEqual(
        repeat(u, 3),
        concat(v[:7], v[7:], repeat(v, 2)),
    )
    self.assertNotEqual(
        concat(u, [0], u),
        concat(v, [1], v),
    )

    # Signed complex zeros, non-reflexive NaN tuples, and shared NaN leaves.
    self.assertNotEqual(
        repeat(complex(0.0, 0.0), 4),
        concat(repeat(complex(0.0, 0.0), 3), [complex(0.0, -0.0)]),
    )
    self.assertNotEqual(
        concat([(float('nan'),)], [1, 1]),
        concat([(float('nan'),)], repeat(1, 2)),
    )
    nan_val = float('nan')
    shared_nan = StaticWord(nan_val)
    # Flattening `shared_nan + [1, 1]` yields a new `LiteralWord`, so NaN stays
    # non-reflexive; unflattened shared `LiteralWord` leaves still compare equal
    # while distinct `LiteralWord`s wrapping the same NaN float remain unequal.
    self.assertNotEqual(
        concat(shared_nan, [1, 1]), concat(shared_nan, repeat(1, 2))
    )
    self.assertEqual(
        concat(shared_nan, repeat(1, 2), [1]),
        concat(shared_nan, repeat(1, 3)),
    )
    self.assertNotEqual(
        concat(StaticWord(nan_val), repeat(1, 2), [1]),
        concat(StaticWord(nan_val), repeat(1, 3)),
    )

  def test_optimize_mux_handles_value_error_on_hash_colliding_tag(self):
    class _RaisingTag:

      def __init__(self, tag: int, *, fail_eq: bool = False):
        self.tag = tag
        self.fail_eq = fail_eq

      def __hash__(self) -> int:
        return 7

      def __eq__(self, other: object) -> bool:
        if self.fail_eq or (isinstance(other, _RaisingTag) and other.fail_eq):
          raise ValueError('ambiguous truth value')
        return isinstance(other, _RaisingTag) and self.tag == other.tag

    good = _RaisingTag(1)
    bad = _RaisingTag(2, fail_eq=True)

    def prog():
      sel = Stream.periodic([], [bad])
      return merge(sel, {good: Stream(10)})

    graph = build_graph(prog, semantics=Semantics.SYNC)
    optimization.optimize_graph(graph)
    self.assertIsNotNone(graph)

  def test_repeated_non_one_periodic_pattern_sync_kahn_and_reset(self):
    def pattern_prog():
      # Periodic word built with grammar: ([False]*2 + [True])*ω
      cyc = Stream.periodic([], repeat(False, 2) + True)
      # Nested repetition prefix + periodic suffix inside reset scope.
      rst = Stream.periodic([], repeat(False, 4) + True)
      with reset(rst):
        nested = Stream.periodic(
            repeat(repeat(0, 2) + 1, 2), StaticWord([9, 8])
        )
        # Rewinds on ticks 4 and 9 (indices 0,1,2,3 -> 0,0,1,0; rst -> 0...).
        gated = Stream.periodic([], StaticWord(True) + repeat(False, 2))
      return cyc, nested, gated

    for sem in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=sem):
        cyc_tr, nested_tr, gated_tr = Scheduler.run(
            pattern_prog, horizon=10, semantics=sem
        )
        self.assertEqual(
            cyc_tr,
            [
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
            ],
        )
        self.assertEqual(nested_tr, [0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(
            gated_tr,
            [True, False, False, True, True, False, False, True, False, True],
        )

    # Also run without reset beyond prefix to cover transition into period.
    long_seq = Scheduler.run(
        lambda: Stream.periodic(
            repeat(repeat(0, 2) + 1, 2), StaticWord([9, 8])
        ),
        horizon=10,
        semantics=Semantics.SYNC,
    )
    self.assertEqual(long_seq, [0, 0, 1, 0, 0, 1, 9, 8, 9, 8])

    # Stream(word) stutters last value without expanding repetitions.
    stutter = Stream(repeat('a', 10**8) + 'b')
    self.assertEqual(stutter.static_value_at(0), 'a')
    self.assertEqual(stutter.static_value_at(10**8 - 1), 'a')
    self.assertEqual(stutter.static_value_at(10**8), 'b')
    self.assertEqual(stutter.static_value_at(10**8 + 99), 'b')
    self.assertEqual(list(stutter.period), ['b'])

  def test_kahn_stateful_primitives_reject_uninitialized_first_tick(self):
    async def check() -> None:
      sched = Scheduler(semantics=Semantics.KAHN, horizon=1)
      tok = active_scheduler.set(sched)
      try:
        for make_gen in (
            lambda o: kahn_primitives.fby_step_1(
                o, StreamPort(Stream(1)), StreamPort(Stream(2))
            ),
            lambda o: kahn_primitives.fby_step(
                o, StreamPort(Stream(1)), StreamPort(Stream(2)), n=2
            ),
            lambda o: kahn_primitives.current._psyre_step_func(  # pylint: disable=protected-access
                o,
                StreamPort(Stream(False)),
                StreamPort(Stream(1)),
                StreamPort(Stream(2)),
            ),
        ):
          out = Stream()
          gen = make_gen(StreamPort(out))
          await anext(gen)
          try:
            with self.assertRaisesRegex(TypeError, 'Bottom cannot be pushed'):
              await gen.asend(False)
          finally:
            await gen.aclose()
      finally:
        active_scheduler.reset(tok)

    asyncio.run(check())

  def test_replace_stream_alias_compression_and_prune(self):
    # 1. Chain of 200 aliases collapses via path compression in resolve_stream.
    def chain_prog():
      cur = counter(0)
      for _ in range(200):
        cur = cur + 1
      return cur

    g = build_graph(chain_prog, optimize=False)
    streams = [n.output for n in g.nodes]
    for i in range(len(streams) - 1):
      g.replace_stream(streams[i], streams[i + 1])
    root = streams[-1]
    self.assertIs(g.resolve_stream(streams[0]), root)
    self.assertIs(g.resolve_stream_id(streams[0].id), root)
    # Path compression collapses the entire chain directly onto `root`.
    self.assertIs(g._stream_aliases[streams[0].id], root)  # pylint: disable=protected-access
    self.assertIs(g._stream_aliases[streams[100].id], root)  # pylint: disable=protected-access

    # Clone keeps an independent `_stream_aliases` dict.
    cloned = g.clone_unfinalized()
    cloned.mark_finalized()
    extra_src = Stream()
    cloned.add_stream(extra_src)
    cloned.replace_stream(extra_src, root)
    self.assertNotIn(extra_src.id, g._stream_aliases)  # pylint: disable=protected-access

    # 2. `retain_nodes` preserves live chain `a -> b -> c` across dead `b`,
    # and transitively prunes `a`, `b`, and side branch `x -> b` when `c` dies.
    def quad_prog():
      s_x = counter(1)
      s_a = counter(2)
      s_b = counter(3)
      s_c = counter(4)
      return s_x, s_a, s_b, s_c

    qg = build_graph(quad_prog, optimize=False)
    nx, na, nb, nc = qg.nodes
    sx, sa, sb, sc = nx.output, na.output, nb.output, nc.output
    qg.replace_stream(sx, sb)
    qg.replace_stream(sa, sb)
    qg.replace_stream(sb, sc)
    # Compress only `sa` so `sx` stays indirect through `sb`.
    self.assertIs(qg.resolve_stream(sa), sc)
    # Drop `nx`, `na`, `nb` while keeping `nc` live -> aliases stay live!
    qg.retain_nodes([nc])
    self.assertIs(qg.resolve_stream(sx), sc)
    self.assertIs(qg.resolve_stream(sa), sc)
    self.assertIs(qg.resolve_stream_id(sb.id), sc)

    # Now drop `nc` -> every alias chain rooted at `sc` must be pruned.
    qg.retain_nodes([])
    self.assertEqual(qg._stream_aliases, {})  # pylint: disable=protected-access
    self.assertIsNone(qg.resolve_stream_id(sx.id))
    self.assertIsNone(qg.resolve_stream_id(sa.id))
    self.assertIsNone(qg.resolve_stream_id(sb.id))

    # 3. Re-aliasing an already-replaced stream raises ValueError; chain via
    # the active target instead so outputs and consumers stay consistent.
    def pair_prog():
      return counter(1), counter(2), counter(3)

    pg = build_graph(pair_prog, optimize=False)
    p0, p1, p2 = (n.output for n in pg.nodes)
    pg.replace_stream(p0, p1)
    with self.assertRaisesRegex(ValueError, 'already been replaced'):
      pg.replace_stream(p0, p2)
    pg.replace_stream(p1, p2)
    self.assertIs(pg.resolve_stream(p0), p2)
    self.assertEqual(pg.output, (p2, p2, p2))

    # 4. `lower_resets` short-circuits in O(1) when already lowered, and
    # `pin_base_clock` clears `_lowered_semantics`.
    def rst_prog():
      c = Stream([True, False])
      r = Stream([False, True])
      with reset(r):
        return when(c, counter(0))

    rg = build_graph(rst_prog, semantics=Semantics.SYNC)
    low = lower_resets(rg, check_clocks(rg))
    self.assertTrue(low.is_resets_lowered_for(Semantics.SYNC))
    self.assertIs(lower_resets(low, check_clocks(low)), low)
    with self.assertRaises(TypeError):
      lower_resets(low, None)  # type: ignore[arg-type]
    low.pin_base_clock(Stream(False))
    self.assertFalse(low.is_resets_lowered_for(Semantics.SYNC))

  def test_user_operators_colliding_with_builtin_names(self):
    @psyre_comb
    def merge(a, b):  # pylint: disable=redefined-outer-name
      return a * 10 + b

    @psyre_comb
    def when(a, b):  # pylint: disable=redefined-outer-name
      return a + b

    @psyre_comb
    def whennot(a, b):  # pylint: disable=redefined-outer-name
      return a - b

    @psyre_comb
    def buffer(a, b, c):  # pylint: disable=redefined-outer-name
      return a + b + c

    @psyre_comb
    def post(a):  # pylint: disable=redefined-outer-name
      return a + 100

    @psyre_comb
    def fby(a, b):  # pylint: disable=redefined-outer-name
      return a + b

    @psyre_sequ
    def pre(a):
      acc = 0
      while True:
        acc = acc + a
        a = yield acc

    # User comb `fby` / sequ `pre` do NOT break instantaneous causality cycles.
    def bad_comb_fby():
      x = fby(1, lambda: x)
      return x

    def bad_sequ_pre():
      x = pre(lambda: x)
      return x

    for bad_fn in (bad_comb_fby, bad_sequ_pre):
      with self.assertRaises(psyre.CausalityError):
        psyre.check_causality(build_graph(bad_fn, optimize=False))

    # Inside `match` / clocks / lowering / optimize / execution (SYNC & KAHN).
    def mixed_prog():
      sel = Stream.periodic([], [True, False])
      a = counter(1)
      b = counter(10)
      with match(sel) as m:
        with case(True):
          m.r = merge(a, b) + when(a, b) + buffer(a, b, 1)
        with case(False):
          m.r = whennot(b, a) + post(a) + pre(a)
      return m.r

    for opt in (False, True):
      for sem in (Semantics.SYNC, Semantics.KAHN):
        with self.subTest(optimize=opt, semantics=sem):
          g = build_graph(mixed_prog, semantics=sem, optimize=opt)
          clks = check_clocks(g)
          out = Scheduler.run_graph(g, horizon=4, clocks=clks)
          # tick 0 (True): a=1,b=10 -> (1*10+10) + (1+10) + (1+10+1) = 43
          # tick 1 (False): a=2,b=11 -> (11-2) + (2+100) + 2 = 113
          # tick 2 (True): a=3,b=12 -> (3*10+12) + (3+12) + (3+12+1) = 73
          # tick 3 (False): a=4,b=13 -> (13-4) + (4+100) + (2+4) = 119
          self.assertEqual(out, [43, 113, 73, 119])
          self.assertFalse(Scheduler.deadlocked)

    # Verify user `@psyre_comb def post` does NOT trigger KAHN EOF shortcut
    # when reading from a 0-token upstream `when(False, ...)`.
    def user_post_zero_tokens():
      gated = psyre.when(Stream(False), counter(0))
      return post(gated)

    Scheduler.run(
        user_post_zero_tokens,
        horizon=2,
        semantics=Semantics.KAHN,
        optimize=False,
    )
    self.assertTrue(Scheduler.deadlocked)

  def test_frozenset_cse_distinguishes_type_and_signed_zero(self):
    @psyre_comb(pure=True, static_argnames=('tags',))
    def tag_info(x, *, tags):
      val = next(iter(tags))
      if isinstance(val, float):
        return (type(val).__name__, math.copysign(1.0, val))
      if isinstance(val, complex):
        return (
            math.copysign(1.0, val.real),
            math.copysign(1.0, val.imag),
        )
      if isinstance(val, tuple):
        return (type(val[0]).__name__, math.copysign(1.0, val[1]))
      return (type(val).__name__, x)

    def distinct_prog():
      x = counter(0)
      return (
          tag_info(x, tags=frozenset({1})),
          tag_info(x, tags=frozenset({True})),
          tag_info(x, tags=frozenset({1.0})),
          tag_info(x, tags=frozenset({0.0})),
          tag_info(x, tags=frozenset({-0.0})),
          tag_info(x, tags=frozenset({0.0 + 0.0j})),
          tag_info(x, tags=frozenset({-0.0 + 0.0j})),
          tag_info(x, tags=frozenset({(1, 0.0)})),
          tag_info(x, tags=frozenset({(True, -0.0)})),
      )

    self.assertEqual(
        Scheduler.run(distinct_prog, horizon=2, optimize=True),
        Scheduler.run(distinct_prog, horizon=2, optimize=False),
    )

    def identical_prog():
      x = counter(0)
      return (
          tag_info(x, tags=frozenset({1, 2})),
          tag_info(x, tags=frozenset({1, 2})),
      )

    opt_g = build_graph(identical_prog, optimize=True)
    self.assertLen(opt_g.nodes, 2)

  def test_is_bool_stream_guards_comb_call_plan_and_forwards_current(self):
    class _FakeBox:

      def __init__(self, value):
        self.value = value

      def __radd__(self, other):
        del other
        return 'not_scalar'

      def __rand__(self, other):
        del other
        return 'not_bool'

    psyre.register_pytree_node(
        _FakeBox,
        lambda box: ((box.value,), None),
        lambda _, children: _FakeBox(children[0]),
    )
    try:
      and_g = build_graph(
          lambda: Stream(True) & _FakeBox(Stream(False)), optimize=True
      )
      self.assertFalse(and_g.is_bool_stream(and_g.output))
      self.assertEqual(Scheduler.run_graph(and_g, horizon=2), ['not_bool'] * 2)

      add_cmp_g = build_graph(
          lambda: (Stream(1) + _FakeBox(Stream(2))) == 1, optimize=False
      )
      self.assertFalse(add_cmp_g.is_bool_stream(add_cmp_g.output))
    finally:
      psyre.unregister_pytree_node(_FakeBox)

    bool_cur_g = build_graph(
        lambda: psyre.current(counter(0) == 0, Stream(True), default=False),
        optimize=False,
    )
    self.assertTrue(bool_cur_g.is_bool_stream(bool_cur_g.output))
    none_cur_g = build_graph(
        lambda: psyre.current(counter(0) == 0, Stream(True)),
        optimize=False,
    )
    self.assertFalse(none_cur_g.is_bool_stream(none_cur_g.output))

  def test_pure_user_op_named_like_builtin_not_misfolded(self):
    @psyre_comb(pure=True)
    def when(a, b):  # pylint: disable=redefined-outer-name
      return a + b

    @psyre_comb(pure=True)
    def __invert__(x):  # pylint: disable=invalid-name
      return x + 100

    @psyre_comb(pure=True)
    def ifthenelse(c, t, f):  # pylint: disable=redefined-outer-name
      return (c, t, f)

    def prog():
      w = when(Stream(True), Stream(5))
      inv = __invert__(__invert__(Stream(1)))
      ite = ifthenelse(Stream(True), Stream(1), Stream(1))
      return w, inv, ite

    g = build_graph(prog, optimize=True)
    self.assertEqual(
        [n.op for n in g.nodes],
        ['when', '__invert__', '__invert__', 'ifthenelse'],
    )
    self.assertEqual(
        Scheduler.run_graph(g, horizon=2),
        ([6, 6], [201, 201], [(True, 1, 1), (True, 1, 1)]),
    )

  def test_cse_dedups_distinct_equal_when_and_merge_selectors(self):
    def bool_prog():
      c = Stream([True, False], period=[True, False])
      x = counter(0)
      t = psyre.when(c, x)
      f = psyre.whennot(c, x + 10)
      n1 = psyre.when(c, x, negated=True)
      n2 = psyre.when(c, x, negated=True)
      m1 = psyre.merge(c, t, f)
      m2 = psyre.merge(c, t, f)
      return n1, n2, m1, m2

    bg = build_graph(bool_prog, optimize=True)
    n1, n2, m1, m2 = bg.output
    self.assertEqual(n1.id, n2.id)
    self.assertEqual(m1.id, m2.id)

    def tagged_prog():
      tag = Stream(['a', 'b'], period=['a', 'b'])
      x = counter(0)
      wa1 = psyre.when(tag, x, values='a')
      wa2 = psyre.when(tag, x, values='a')
      wb = psyre.when(tag, x, values='b')
      s1 = psyre.switch(tag, {'a': x, 'b': x + 1})
      s2 = psyre.switch(tag, {'a': x, 'b': x + 1})
      s_diff = psyre.switch(tag, {'a': x + 1, 'b': x})
      m1 = psyre.merge(tag, {'a': wa1, 'b': wb})
      m2 = psyre.merge(tag, {'a': wa2, 'b': wb})
      return wa1, wa2, wb, s1, s2, s_diff, m1, m2

    tg = build_graph(tagged_prog, optimize=True)
    wa1, wa2, wb, s1, s2, s_diff, m1, m2 = tg.output
    self.assertEqual(wa1.id, wa2.id)
    self.assertNotEqual(wa1.id, wb.id)
    self.assertEqual(s1.id, s2.id)
    self.assertNotEqual(s1.id, s_diff.id)
    self.assertEqual(m1.id, m2.id)
    self.assertEqual(
        Scheduler.run_graph(tg, horizon=4),
        Scheduler.run(tagged_prog, horizon=4, optimize=False),
    )


if __name__ == '__main__':
  absltest.main()
