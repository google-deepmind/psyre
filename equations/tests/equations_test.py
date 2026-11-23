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
"""Unit tests for lambda-free Psyre equations (`psyre_equations` / `Equations`)."""

# pylint: disable=undefined-variable,used-before-assignment
# pyrefly: ignore-errors[unknown-name]

import builtins
import collections
import contextlib
import logging
import threading
import types
from typing import Any, NamedTuple

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import check_causality
from psyre.automaton import automaton, state, unless, until
from psyre.clock import check_clocks
from psyre.common import (
    Absent,
    Semantics,
    fby,
    is_building,
    logger,
    psyre_comb,
    psyre_node,
    reset,
    scan,
)
from psyre.equations import (
    CircularEquationError,
    Equations,
    Thunk,
    psyre_equations,
)
from psyre.match import case, last, match
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


class _PairParams(NamedTuple):
  w: Any
  b: Any


class EquationsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def test_standalone_psyre_equations_defaults_to_sync_and_auto_names(self):
    orig_build_class = builtins.__build_class__
    with psyre_equations(extra_roots=('acc', 'tmp', 'data')) as prog:

      class prefix_sum(data=Stream([1, 2, 3, 4]), start=0):  # pylint: disable=invalid-name
        acc = fby(start, tmp)
        tmp = acc + data
        __output__ = tmp

    self.assertIs(builtins.__build_class__, orig_build_class)
    assert prog.graph is not None
    self.assertEqual(prog.graph.semantics, Semantics.SYNC)
    self.assertFalse(is_building())
    self.assertIs(prog.last, prefix_sum)
    self.assertIsInstance(prefix_sum, Equations)
    check_causality(prog.graph)
    clocks = check_clocks(prog.graph)

    self.assertEqual(
        Scheduler.run_graph(prog.graph, horizon=4, clocks=clocks),
        [1, 3, 6, 10],
    )
    self.assertEqual(
        Scheduler.run_graph(
            prog.graph, horizon=4, streams=('acc', 'tmp', 'data'), clocks=clocks
        ),
        {
            'acc': [0, 1, 3, 6],
            'tmp': [1, 3, 6, 10],
            'data': [1, 2, 3, 4],
        },
    )

  def test_single_equation_fby_counter(self):
    for semantics in (Semantics.SYNC, Semantics.KAHN):
      with self.subTest(semantics=semantics):
        with psyre_equations(semantics=semantics) as prog:

          class _:
            cnt = fby(0, cnt + 1)
            __output__ = cnt

        assert prog.graph is not None
        check_causality(prog.graph)
        clocks = check_clocks(prog.graph)
        self.assertEqual(
            Scheduler.run_graph(prog.graph, horizon=6, clocks=clocks),
            [0, 1, 2, 3, 4, 5],
        )

    # Single-equation resettable counter with default SYNC semantics.
    with psyre_equations() as prog:

      class _:
        with reset(Stream([False, False, True, False, False, True])):
          cnt = fby(0, cnt + 1)
        __output__ = cnt

    assert prog.graph is not None
    check_causality(prog.graph)
    clocks = check_clocks(prog.graph)
    self.assertEqual(
        Scheduler.run_graph(prog.graph, horizon=6, clocks=clocks),
        [0, 1, 0, 1, 2, 0],
    )

  def test_build_class_hook_is_scoped_to_direct_baseless_classes(self):
    def make_helper_class():
      class Helper:
        value = 42

      return Helper

    orig_build_class = builtins.__build_class__
    with psyre_equations() as prog:

      class LocalTuple(NamedTuple):
        x: Any

      class _:
        helper_cls = make_helper_class()
        x = Stream([1, 2, 3])
        __output__ = LocalTuple(x=x)

    self.assertIs(builtins.__build_class__, orig_build_class)
    self.assertNotIsInstance(LocalTuple, Equations)
    self.assertNotIsInstance(prog.last.helper_cls, Equations)
    self.assertIsInstance(prog.last, Equations)
    self.assertEqual(Scheduler.run_graph(prog.graph, horizon=3).x, [1, 2, 3])

  def test_auto_naming_preserves_explicit_names_and_inner_node_prefixes(self):
    @psyre_node('inner_box')
    def inner_box(x):
      with psyre_equations() as eq:

        class _:
          reg = fby(0, x, name='explicit_reg')
          doubled = reg * 2
          __output__ = doubled

      return eq.output

    with psyre_equations(optimize=False) as prog:

      class _:
        inp = Stream([1, 2, 3])
        out = inner_box(inp)
        alias = out
        __output__ = alias

    assert prog.graph is not None
    node_names = {node.name for node in prog.graph.nodes}
    self.assertIn('inner_box/explicit_reg', node_names)
    self.assertIn('inner_box/doubled', node_names)
    named = Scheduler.run_graph(
        prog.graph,
        horizon=3,
        streams=('inp', 'inner_box/explicit_reg', 'out', 'alias'),
    )
    self.assertEqual(named['inp'], [1, 2, 3])
    self.assertEqual(named['inner_box/explicit_reg'], [0, 1, 2])
    self.assertEqual(named['out'], [0, 2, 4])
    self.assertEqual(named['alias'], [0, 2, 4])

  def test_out_of_order_memoization_and_reflected_ops(self):
    with psyre_equations(optimize=False) as prog:

      class _:
        __output__ = (x, z1, z2, y, clamped, neg_reflected)
        x = y + 1
        z1 = x * 2
        z2 = x + 10
        y = fby(0, z1)
        clamped = fby(0, Stream.min(clamped + 2, 5))
        neg_reflected = 10 - clamped

    assert prog.graph is not None
    add_nodes = [n for n in prog.graph.nodes if n.op == '__add__']
    self.assertLen(add_nodes, 3)  # x = y + 1, z2 = x + 10, clamped + 2
    rx, rz1, rz2, ry, clamped, neg = Scheduler.run_graph(prog.graph, horizon=4)
    self.assertEqual(ry, [0, 2, 6, 14])
    self.assertEqual(rx, [1, 3, 7, 15])
    self.assertEqual(rz1, [2, 6, 14, 30])
    self.assertEqual(rz2, [11, 13, 17, 25])
    self.assertEqual(clamped, [0, 2, 4, 5])
    self.assertEqual(neg, [10, 8, 6, 5])

  def test_shadow_builtin_forward_indexing_and_loop_by_value_capture(self):
    with psyre_equations() as prog:

      class _:
        acc = fby(0, sum)
        sum = acc + pair[0] + pair[1]
        pair = [Stream([1, 2, 3]), Stream([10, 20, 30])]

        xs = [Stream([1, 2, 3]), Stream([10, 20, 30])]
        ys = []
        for i in range(2):
          ys.append(fby(0, zs[i]))
        zs = [ys[0] + xs[0], ys[1] + xs[1]]
        __output__ = (sum, zs[0], zs[1])

    s_tr, z0_tr, z1_tr = Scheduler.run_graph(prog.graph, horizon=3)
    self.assertEqual(s_tr, [11, 33, 66])
    self.assertEqual(z0_tr, [1, 3, 6])
    self.assertEqual(z1_tr, [10, 30, 60])

  def test_deferred_match_reset_and_lexical_captured_env(self):
    with psyre_equations(extra_roots=('y', 'doubled')) as prog:

      class _:
        with match(gate > 0):
          with case(True):
            with reset(rst):
              y = fby(0, y + 1)
              doubled = y * 2
          with case(False):
            pass
        gate = Stream([1, 0, 1, 1, 1])
        rst = Stream([False, False, False, True, False])

    assert prog.graph is not None
    res = Scheduler.run_graph(
        prog.graph,
        horizon=5,
        streams=('y', 'doubled'),
        clocks=check_clocks(prog.graph),
    )
    self.assertEqual(res['y'], [0, Absent, 1, 2, 0])
    self.assertEqual(res['doubled'], [0, Absent, 2, 4, 0])

    with psyre_equations() as prog_lex:

      class _:
        cond = Stream([True, False, True, False])
        outer_expr = late_x + 1
        with match(cond) as m:
          with case(True):
            inner_expr = late_x * 10
            m.a = outer_expr
            m.b = inner_expr
          with case(False):
            m.a = 0
            m.b = 0
        late_x = Stream([1, 2, 3, 4])
        __output__ = (m.a, m.b)

    assert prog_lex.graph is not None
    ra, rb = Scheduler.run_graph(
        prog_lex.graph, horizon=4, clocks=check_clocks(prog_lex.graph)
    )
    self.assertEqual(ra, [2, 0, 4, 0])
    self.assertEqual(rb, [10, 0, 20, 0])

  def test_automaton_last_strong_unless_namedtuple_and_forward_ref(self):
    def delay_params(p):
      return _PairParams(w=last(p.w, init=1), b=last(p.b, init=10))

    with psyre_equations() as prog:

      class _:
        toggle = Stream([False, True, False, False, True, False])
        inc = Stream([False, True, True, True, False, False])
        dec = Stream([False, False, False, False, True, False])

        with automaton() as fsm:
          with state('UP'):
            until(toggle, 'DOWN')
            fsm.v = last(fsm.v, 10) + 2
          with state('DOWN'):
            until(toggle, 'UP')
            fsm.v = last(fsm.v, 10) - 1

        with automaton() as strong_fsm:
          with state('IDLE', unless(inc, 'INCR'), unless(dec, 'DECR')):
            strong_fsm.o = last(strong_fsm.o, 0)
          with state(
              'INCR', unless(~inc | (last(strong_fsm.o, 0) >= 2), 'IDLE')
          ):
            strong_fsm.o = last(strong_fsm.o, 0) + 1
          with state('DECR', unless(~dec, 'IDLE')):
            strong_fsm.o = last(strong_fsm.o, 0) - 1

        with automaton() as train:
          prev = delay_params(train.params)
          with state('RUN'):
            train.params = _PairParams(w=prev.w + 1, b=prev.b + 10)

        scaled = Stream(2) * later_fsm.k
        with automaton() as later_fsm:
          with state('A'):
            later_fsm.k = 5

        __output__ = (fsm.v, strong_fsm.o, train.params, scaled)

    clocks = check_clocks(prog.graph)
    a1, a2, nt_res, scaled = Scheduler.run_graph(
        prog.graph, horizon=6, clocks=clocks
    )
    self.assertEqual(a1[:5], [12, 14, 13, 12, 14])
    self.assertEqual(a2, [0, 1, 2, 2, 1, 1])
    self.assertEqual(nt_res.w[:4], [2, 3, 4, 5])
    self.assertEqual(nt_res.b[:4], [20, 30, 40, 50])
    self.assertEqual(scaled, [10] * 6)

  def test_stream_init_guards_against_lazy_thunk(self):
    thunk = Thunk(lambda: Stream(1), desc='x')
    existing = Stream([1, 2])
    for bad_call in (
        lambda: Stream(thunk),
        lambda: Stream([thunk]),
        lambda: Stream(1, period=thunk),
        lambda: Stream(existing, period=thunk),
        lambda: Stream(existing, period=[thunk]),
    ):
      with self.assertRaisesRegex(TypeError, 'LazyThunk'):
        bad_call()

    orig_build_class = builtins.__build_class__
    with self.assertRaisesRegex(TypeError, 'LazyThunk'):
      with psyre_equations():

        class _:
          s = Stream(late)
          late = Stream([1, 2, 3])
          __output__ = s

    self.assertFalse(is_building())
    self.assertIs(builtins.__build_class__, orig_build_class)

  def test_error_diagnostics_and_cleanup(self):
    orig_build_class = builtins.__build_class__

    # 1. Immediate NameError on typo.
    with self.assertRaisesRegex(NameError, 'tpm'):
      with psyre_equations():

        class _:
          acc = fby(0, tpm)
          tmp = acc + 1
          __output__ = tmp

    self.assertFalse(is_building())
    self.assertIs(builtins.__build_class__, orig_build_class)

    # 2. Pure Thunk cycle caught cleanly.
    with self.assertRaises(CircularEquationError):
      with psyre_equations():

        class _:
          a = b + 1
          b = a + 1
          __output__ = a

    self.assertFalse(is_building())
    self.assertIs(builtins.__build_class__, orig_build_class)

    # 3. Double assignment of forward-referenced equation rejected.
    with self.assertRaisesRegex(RuntimeError, 'cannot be reassigned'):
      with psyre_equations():

        class _:
          acc = fby(0, tmp)
          tmp = acc + 1
          tmp = tmp * 2
          __output__ = tmp

    self.assertFalse(is_building())
    self.assertIs(builtins.__build_class__, orig_build_class)

    # 4. Unpacking / truthiness / mutation of unresolved Thunk rejected cleanly.
    t = Thunk(lambda: (1, 2), desc='pair')
    with self.assertRaisesRegex(TypeError, 'Truth value'):
      bool(t)
    with self.assertRaisesRegex(TypeError, 'Cannot iterate or unpack'):
      iter(t)
    with self.assertRaisesRegex(TypeError, 'Cannot mutate'):
      t[0] = 3

  def test_concurrent_class_definitions_with_psyre_equations(self):
    orig_build_class = builtins.__build_class__

    # Deterministic check for the exit race: a thread that loaded
    # builtins.__build_class__ before psyre_equations() exited must still
    # succeed when calling it afterward.
    with psyre_equations():
      hooked_build_class = builtins.__build_class__
    self.assertIs(builtins.__build_class__, orig_build_class)

    def _body():
      pass

    late_cls = hooked_build_class(_body, 'LatePlainClass')
    self.assertEqual(late_cls.__name__, 'LatePlainClass')
    self.assertNotIsInstance(late_cls, Equations)

    # Concurrent threads entering/exiting psyre_equations() alongside threads
    # defining plain classes must never raise or leak the hook.
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def equations_worker() -> None:
      try:
        barrier.wait()
        for _ in range(30):
          with psyre_equations() as prog:

            class _:
              cnt = fby(0, cnt + 1)
              __output__ = cnt

          assert prog.graph is not None
          self.assertEqual(
              Scheduler.run_graph(prog.graph, horizon=3), [0, 1, 2]
          )
      except BaseException as err:  # pylint: disable=broad-exception-caught
        errors.append(err)

    def plain_class_worker() -> None:
      try:
        barrier.wait()
        for i in range(150):

          class Plain:
            value = i

          self.assertNotIsInstance(Plain, Equations)
          self.assertEqual(Plain.value, i)
      except BaseException as err:  # pylint: disable=broad-exception-caught
        errors.append(err)

    threads = [
        threading.Thread(target=equations_worker),
        threading.Thread(target=equations_worker),
        threading.Thread(target=plain_class_worker),
        threading.Thread(target=plain_class_worker),
    ]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    self.assertEmpty(errors)
    self.assertIs(builtins.__build_class__, orig_build_class)

  def test_last_after_same_arm_assignment_reads_merged_output(self):
    # Arm-local stream output in automaton(): later last(fsm.v, 0) and fsm.v
    # references in the same state must target the merged stream cleanly.
    with psyre_equations() as local_auto:

      class _:
        with automaton() as fsm:
          with state('A'):
            fsm.v = last(fsm.v, 0) + 1
            until(last(fsm.v, 0) >= 2, 'B')
            fsm.delta = fsm.v - last(fsm.v, 0)
          with state('B'):
            fsm.v = 10
            fsm.delta = 0
        __output__ = (fsm.v, fsm.delta)

    assert local_auto.graph is not None
    check_causality(local_auto.graph)
    local_clocks = check_clocks(local_auto.graph)
    v_tr, d_tr = Scheduler.run_graph(
        local_auto.graph, horizon=4, clocks=local_clocks
    )
    self.assertEqual(v_tr, [1, 2, 3, 10])
    self.assertEqual(d_tr, [1, 1, 1, 0])

    # Same-arm last(m.v, 0) after m.v = ... inside match().
    with psyre_equations() as match_prog:

      class _:
        sel = Stream([True, False, True, False])
        with match(sel) as m:
          with case(True):
            m.v = 10
            m.prev = last(m.v, 0)
          with case(False):
            m.v = 20
            m.prev = last(m.v, 0)
        __output__ = (m.v, m.prev)

    assert match_prog.graph is not None
    check_causality(match_prog.graph)
    mv_tr, mp_tr = Scheduler.run_graph(match_prog.graph, horizon=4)
    self.assertEqual(mv_tr, [10, 20, 10, 20])
    self.assertEqual(mp_tr, [0, 10, 20, 10])

  def test_forward_referenced_structured_outputs_resolve_nested_thunks(self):
    with psyre_equations() as prog:

      class _:
        __output__ = out
        out = {
            'pair': (x + 1, x + 2),
            'named': _PairParams(w=y, b=[x * 10, y + 1]),
        }
        y = x + 3
        x = Stream([1, 2, 3])

    assert prog.graph is not None
    assert prog.last is not None
    # Verify both __output__ and the namespace attribute `out` are resolved.
    self.assertNotIsInstance(prog.last.out['pair'][0], Thunk)
    self.assertNotIsInstance(prog.last.out['named'].b[0], Thunk)
    res = Scheduler.run_graph(prog.graph, horizon=3)
    self.assertEqual(res['pair'], ([2, 3, 4], [3, 4, 5]))
    self.assertEqual(
        res['named'], _PairParams(w=[4, 5, 6], b=[[10, 20, 30], [5, 6, 7]])
    )

  def test_higher_order_forward_references_are_forbidden(self):
    # pylint: disable=no-self-argument,no-method-argument,self-cls-assignment
    orig_build_class = builtins.__build_class__

    def _assert_rejected(fn):
      with self.assertRaisesRegex(TypeError, 'define callables before use'):
        fn()
      self.assertFalse(is_building())
      self.assertIs(builtins.__build_class__, orig_build_class)

    # 1. Calling a forward-referenced callable or deferred method fails eagerly.
    def call_regular():
      with psyre_equations():

        class _:
          y = helper(x)
          x = Stream([1, 2])

          def helper(v):
            return v + 1

          __output__ = y

    def call_deferred_method():
      with psyre_equations():

        class _:
          y = (x + 1).min(5)
          x = Stream([1, 2])
          __output__ = y

    # 2. Binding a forward-referenced name to a callable without calling it
    #    (via EquationScope or container indexing via Thunk.__psyre_force__)
    #    must also fail cleanly.
    def bind_callable_in_fby():
      with psyre_equations():

        class _:
          y = fby(0, fn)

          def fn():
            return Stream([1, 2])

          __output__ = y

    def bind_callable_via_container_index():
      with psyre_equations():

        class _:
          y = ops[0]
          ops = [fby]
          __output__ = y

    for bad_case in (
        call_regular,
        call_deferred_method,
        bind_callable_in_fby,
        bind_callable_via_container_index,
    ):
      with self.subTest(case=bad_case.__name__):
        _assert_rejected(bad_case)

    # Passing an unhashable forward-referenced Thunk as scan(op=...) fails.
    with self.assertRaises(TypeError):
      with psyre_equations():

        class _:
          y = scan(x, op=add_op)
          x = Stream([1, 2, 3])

          def add_op(a, b):
            return a + b

          __output__ = y
    self.assertFalse(is_building())
    self.assertIs(builtins.__build_class__, orig_build_class)

    # Defining callables before use remains fully supported.
    with psyre_equations() as prog:

      class _:
        @psyre_comb
        def inc(v):
          return v + 1

        @psyre_node('box')
        def box(v):
          return v * 2

        y = box(inc(x))
        x = Stream([1, 2, 3])
        __output__ = y

    self.assertEqual(Scheduler.run_graph(prog.graph, horizon=3), [4, 6, 8])

    # Forward-referenced `abs(thunk)` resolves through `Stream.__abs__`.
    with psyre_equations() as prog_abs:

      class _:
        y = abs(x)
        x = Stream([-3, 0, 5])
        __output__ = y

    self.assertEqual(Scheduler.run_graph(prog_abs.graph, horizon=3), [3, 0, 5])

  def test_equations_metaclass_edge_cases_and_hygiene(self):
    # 1. Positional-only `/` allows name/bases/namespace/mcs/cls input kwargs.
    with psyre_equations() as kw_prog:

      class _(
          name=Stream([1, 2]),
          bases=Stream([10, 20]),
          namespace=Stream([100, 200]),
          mcs=Stream([1000, 2000]),
          cls=Stream([3, 4]),
      ):
        __output__ = name + bases + namespace + mcs + cls

    assert kw_prog.graph is not None
    self.assertEqual(
        Scheduler.run_graph(kw_prog.graph, horizon=2), [1114, 2226]
    )

    # 2. `contextlib.ExitStack.enter_context(psyre_equations())` hooks classes.
    with contextlib.ExitStack() as stack:
      es_prog = stack.enter_context(psyre_equations())

      class _:
        y = fby(0, y + 1)
        __output__ = y

    assert es_prog.graph is not None
    self.assertEqual(
        Scheduler.run_graph(es_prog.graph, horizon=4), [0, 1, 2, 3]
    )

    # 3. `_resolve_tree` handles arbitrary Mapping subclasses.
    with psyre_equations() as map_prog:

      class map_cls:  # pylint: disable=invalid-name
        ro = types.MappingProxyType({'a': x + 1})
        ud = collections.UserDict({'b': x + 2})
        x = Stream([10, 20])
        __output__ = (ro, ud)

    assert map_prog.graph is not None
    self.assertNotIsInstance(map_cls.ro['a'], Thunk)
    self.assertIsInstance(map_cls.ro, types.MappingProxyType)
    self.assertNotIsInstance(map_cls.ud['b'], Thunk)
    self.assertEqual(
        Scheduler.run_graph(map_prog.graph, horizon=2),
        ({'a': [11, 21]}, {'b': [12, 22]}),
    )

    # 4. Mutual container cycle raises CircularEquationError cleanly.
    with self.assertRaises(CircularEquationError):
      with psyre_equations():

        class _:
          a = (b, 1)
          b = (a, 2)
          __output__ = a

    # 5. Deep diamond Thunk._desc stays bounded instead of O(2^d) blowup.
    t = Thunk(lambda: 1, desc='x')
    for _ in range(25):
      t = t + t
    self.assertLessEqual(len(t._desc), 120)  # pylint: disable=protected-access

    # 6. `EquationScope.name_stream` never mutates `.name` on unnamed external
    #    Stream objects shared across graphs or outside Stage 1.
    ext = Stream([1, 2, 3])
    self.assertEqual(ext.name, '')
    with psyre_equations(extra_roots=('first_in',)) as p1:

      class _(first_in=ext):
        __output__ = first_in + 1

    self.assertEqual(ext.name, '')
    with psyre_equations(extra_roots=('second_in',)) as p2:

      class _:
        second_in = ext
        __output__ = second_in + 2

    self.assertEqual(ext.name, '')
    assert p1.graph is not None and p2.graph is not None
    self.assertEqual(p1.graph.stream_names(ext), ('first_in',))
    self.assertEqual(p2.graph.stream_names(ext), ('second_in',))

    class _(metaclass=Equations, leaked=ext):
      pass

    self.assertEqual(ext.name, '')

    # 7. Nested psyre_equations() rejects mismatched/standalone-only kwargs.
    with psyre_equations(semantics=Semantics.KAHN):
      with self.assertRaisesRegex(ValueError, 'conflicts with active'):
        with psyre_equations(semantics=Semantics.SYNC):
          pass
      with self.assertRaisesRegex(ValueError, 'optimize cannot be set'):
        with psyre_equations(optimize=False):
          pass
      with self.assertRaisesRegex(ValueError, 'extra_roots cannot be set'):
        with psyre_equations(extra_roots=('x',)):
          pass
      # Matching semantics is allowed when nested.
      with psyre_equations(semantics=Semantics.KAHN):

        class _:
          __output__ = Stream([1])

    with self.assertRaisesRegex(TypeError, 'Semantics'):
      with psyre_equations(semantics='SYNC'):  # type: ignore[arg-type]
        pass
    with self.assertRaisesRegex(TypeError, 'bool'):
      with psyre_equations(optimize='yes'):  # type: ignore[arg-type]
        pass


if __name__ == '__main__':
  absltest.main()
