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
"""Pure-Python tests for first-order payload, PyTree, and lifting contracts."""

# pylint: disable=used-before-assignment

import dataclasses
import enum
import functools
import logging
import sys
import types
from typing import Any, NamedTuple

from absl.testing import absltest

import psyre
# pylint: disable=g-importing-member,g-multiple-import
from psyre.automaton import automaton, state, until
from psyre.builder import build_graph
from psyre.clock import check_clocks
from psyre.common import (
    Absent,
    Bottom,
    LazyThunk,
    Semantics,
    format_tag,
    ifthenelse,
    is_pure_step_func,
    logger,
    make_comb_step_func,
    matches_tag,
    merge,
    psyre_comb,
    psyre_sequ,
    reset,
    scan,
    switch,
    when,
    when_selector_matches_const,
    whennot,
)
from psyre.common_payload import (
    CALL_PLAN_KWARG,
    is_array_object,
    is_nan_scalar,
    normalize_static_argnums,
    partition_comb_args,
    register_pytree_node,
    unregister_pytree_node,
)
from psyre.equations import psyre_equations
from psyre.match import case, default, match
from psyre.optimization import optimize_graph
from psyre.scheduler import Scheduler
from psyre.static_stream import StaticSequence, StaticWord, same_const_value
from psyre.stream import Stream, resolve_stream

# pylint: enable=g-importing-member,g-multiple-import


class _Color(enum.Enum):
  RED = 1
  BLUE = 2


class _CustomEqEnum(enum.Enum):
  A = 1
  B = 2

  def __eq__(self, other: Any) -> Any:
    return ('eq', self.name, getattr(other, 'name', other))


class _Pair(NamedTuple):
  first: Any
  second: Any


class _Box:
  """Custom container registered via `register_pytree_node`."""

  def __init__(self, values: tuple[Any, ...], tag: str):
    self.values = values
    self.tag = tag

  def __eq__(self, other: object) -> bool:
    return (
        isinstance(other, _Box)
        and self.tag == other.tag
        and self.values == other.values
    )


class _StrictArrayDuck:
  """Duck-typed array whose `__repr__`, `__bool__`, and `__eq__` fail."""

  def __init__(self, shape: tuple[int, ...], dtype: str = 'float32'):
    self.shape = shape
    self.dtype = dtype

  def __repr__(self) -> str:
    raise AssertionError('__repr__ must not be triggered on array objects')

  def __bool__(self) -> bool:
    raise ValueError('The truth value of an array is ambiguous.')

  def __eq__(self, other: object) -> Any:
    del other
    return _StrictArrayDuck(self.shape, dtype='bool')

  def __gt__(self, other: Any) -> Any:
    del other
    return _StrictArrayDuck(self.shape, dtype='bool')

  def __and__(self, other: Any) -> Any:
    return ('and', self.shape, other)

  def __or__(self, other: Any) -> Any:
    return ('or', self.shape, other)


class _RaisingEqObj:
  """Plain Python object whose `__eq__` raises `ValueError`."""

  def __init__(self, value: int):
    self.value = value

  def __eq__(self, other: object) -> bool:
    del other
    raise ValueError('ambiguous equality')


class _Vec:
  """Custom object supporting `+` with another `_Vec`, NOT with `int(0)`."""

  def __init__(self, x: int, y: int):
    self.x = x
    self.y = y

  def __add__(self, other: object) -> '_Vec':
    if not isinstance(other, _Vec):
      return NotImplemented
    return _Vec(self.x + other.x, self.y + other.y)

  def __eq__(self, other: object) -> bool:
    return isinstance(other, _Vec) and (self.x, self.y) == (other.x, other.y)


class _CallableObj:
  """Custom callable instance for testing strict first-order rejection."""

  def __call__(self, x: Any) -> Any:
    return x


class _MinimalThunk(LazyThunk):
  """Custom `LazyThunk` without `__array_ufunc__` / `__array_function__`."""

  def __init__(self, val: Any):
    self._val = val

  def __psyre_force__(self) -> Any:
    return self._val


class _Mode:
  pass


class _Payload:
  def __init__(self, label: str):
    self.label = label


class PayloadInteropTest(absltest.TestCase):
  """Zero-dependency tests for Psyre payload & lifting prerequisites."""

  def test_no_hard_numpy_or_jax_import_and_exports(self):
    self.assertNotIn('numpy', sys.modules)
    self.assertNotIn('jax', sys.modules)
    for name in (
        'is_pure_step_func',
        'register_pytree_node',
        'unregister_pytree_node',
    ):
      self.assertIn(name, psyre.__all__)
      self.assertTrue(hasattr(psyre, name))

  def test_reject_callable_strict_first_order(self):
    def fn(x):
      return x

    zero_lambda = lambda: 1
    one_lambda = lambda x: x + 1
    bound_method = 'abc'.upper
    partial_fn = functools.partial(fn, 1)
    callable_obj = _CallableObj()

    for bad in (
        fn,
        zero_lambda,
        one_lambda,
        len,
        bound_method,
        partial_fn,
        int,
        _Mode,
        callable_obj,
        (1, fn),
        (1, [2, one_lambda]),
        {'k': len},
        {fn: 'v'},
        types.MappingProxyType({'nested': callable_obj}),
        {fn},
        frozenset({len}),
    ):
      with self.subTest(bad=repr(bad)):
        with self.assertRaisesRegex(TypeError, 'first-order'):
          Stream.constant(bad)
        with self.assertRaisesRegex(TypeError, 'first-order'):
          StaticSequence.constant(bad)
      if not isinstance(bad, list):
        with self.assertRaisesRegex(TypeError, 'first-order'):
          Stream(bad)

    # Multi-element lists containing callables (directly or nested).
    for seq in ([1, fn], [1, (2, len)], [1, {'a': callable_obj}]):
      with self.assertRaisesRegex(TypeError, 'first-order'):
        Stream(seq)
    with self.assertRaisesRegex(TypeError, 'first-order'):
      StaticWord(fn)
    with self.assertRaisesRegex(TypeError, 'first-order'):
      resolve_stream(one_lambda)
    with self.assertRaisesRegex(TypeError, 'first-order'):
      resolve_stream(callable_obj)

    # Cycle inside container must terminate cleanly (benign & callable).
    cyclic_ok: list[Any] = [1]
    cyclic_ok.append(cyclic_ok)
    const_cyclic = Stream.constant(cyclic_ok)
    self.assertIs(const_cyclic.static_value_at(0), cyclic_ok)

    cyclic_bad: list[Any] = []
    cyclic_bad.append(cyclic_bad)
    cyclic_bad.append(fn)
    with self.assertRaisesRegex(TypeError, 'first-order'):
      Stream.constant(cyclic_bad)

    # LazyThunk keeps its distinct diagnostic message rather than 'first-order'.
    with self.assertRaisesRegex(TypeError, 'LazyThunk'):
      Stream(_MinimalThunk(1))

  def test_reject_callable_in_dynamic_push_and_selectors(self):
    @psyre_comb
    def emit_fn(x):
      return lambda y=x: y

    with self.assertRaises(ExceptionGroup) as ctx:
      Scheduler.run(lambda: emit_fn(Stream([1, 2])), horizon=2)
    self.assertRegex(
        str(ctx.exception.exceptions[0]), 'first-order'
    )
    self.assertIsInstance(ctx.exception.exceptions[0], TypeError)

    @psyre_comb
    def emit_nested_fn(x):
      return {'cb': (len, x)}

    with self.assertRaises(ExceptionGroup) as ctx2:
      Scheduler.run(lambda: emit_nested_fn(Stream([1, 2])), horizon=2)
    self.assertIsInstance(ctx2.exception.exceptions[0], TypeError)
    self.assertRegex(str(ctx2.exception.exceptions[0]), 'first-order')

    # Absent bypasses reject_callable cleanly on push.
    def active_prog():
      c = Stream([True, False, True])
      x = Stream([1, 2, 3])
      return when(c, x)

    out = Scheduler.run(active_prog, horizon=3, semantics=Semantics.SYNC)
    self.assertEqual(out, [1, Absent, 3])

    # Selector tags reject callables across when, merge, switch, match,
    # automaton.
    with self.assertRaisesRegex(TypeError, 'first-order'):
      build_graph(lambda: when(Stream(1), Stream(2), values=[len]))
    with self.assertRaisesRegex(TypeError, 'first-order'):
      build_graph(lambda: merge(Stream(1), {len: Stream(2)}))
    with self.assertRaisesRegex(TypeError, 'first-order'):
      build_graph(lambda: switch(Stream(1), {len: Stream(2)}))

    def bad_match():
      with match(Stream(1)) as m:
        with case(len):
          m.y = Stream(2)
        with default():
          m.y = Stream(3)
      return m.y

    with self.assertRaisesRegex(TypeError, 'first-order'):
      build_graph(bad_match)

    def bad_automaton():
      with automaton(init=len) as a:
        with state(len):
          a.y = Stream(1)
          until(Stream(False), len)
      return a.y

    with self.assertRaisesRegex(TypeError, 'first-order'):
      build_graph(bad_automaton)

    # `scan(data, op=fn)` continues to accept a callable reducer.
    res = Scheduler.run(
        lambda: scan(Stream([1, 2, 3, 4]), op=lambda a, b: a * b),
        horizon=4,
    )
    self.assertEqual(res, [1, 2, 6, 24])

  def test_stream_constant_and_plain_python_objects(self):
    payload_list = [1, 2, 3]
    word = StaticWord([10, 20])
    seq = StaticSequence.from_value([7, 8])
    custom = _Payload('hi')
    raising = _RaisingEqObj(5)

    def prog():
      a = Stream.constant(payload_list)
      b = Stream.constant(word)
      c = Stream.constant(seq)
      d = Stream.constant({'k': [1, 2]})
      e = Stream.constant(custom)
      f = Stream.constant(raising)
      return a, b, c, d, e, f

    ta, tb, tc, td, te, tf = Scheduler.run(prog, horizon=3)
    self.assertEqual(ta, [[1, 2, 3]] * 3)
    self.assertEqual(tb, [word] * 3)
    self.assertEqual(tc, [seq] * 3)
    self.assertEqual(td, [{'k': [1, 2]}] * 3)
    self.assertEqual(te, [custom] * 3)
    self.assertTrue(all(v is raising for v in tf))

    with self.assertRaisesRegex(TypeError, 'Stream instance'):
      Stream.constant(Stream([1, 2]))
    with self.assertRaises(TypeError):
      Stream.constant(Absent)
    with self.assertRaises(TypeError):
      Stream.constant(Bottom)
    with self.assertRaisesRegex(TypeError, 'LazyThunk'):
      Stream.constant(_MinimalThunk(1))

    # StaticSequence equality & matches_tag safely handle raising __eq__.
    self.assertTrue(same_const_value(raising, raising))
    self.assertFalse(same_const_value(_RaisingEqObj(1), _RaisingEqObj(1)))
    self.assertFalse(matches_tag(_RaisingEqObj(1), _RaisingEqObj(1)))

  def test_is_bool_stream_conservative_scalar_guard(self):
    g_scalar = build_graph(
        lambda: (Stream([1, 2, 3]) > 1, Stream([_Color.RED]) == _Color.RED),
        optimize=False,
    )
    s_cmp, e_cmp = g_scalar.output
    self.assertTrue(g_scalar.is_bool_stream(s_cmp))
    self.assertTrue(g_scalar.is_bool_stream(e_cmp))

    # Custom Enum overriding __eq__ cannot be assumed scalar bool.
    g_custom_enum = build_graph(
        lambda: Stream([_CustomEqEnum.A]) == _CustomEqEnum.A,
        optimize=False,
    )
    self.assertFalse(g_custom_enum.is_bool_stream(g_custom_enum.output))

    # Unbound dynamic Stream() comparison is NOT guaranteed scalar bool.
    g_unbound = build_graph(
        lambda: Stream(name='ext') > 0,
        optimize=False,
    )
    self.assertFalse(g_unbound.is_bool_stream(g_unbound.output))

    # Opaque @psyre_comb output comparison is NOT guaranteed scalar bool.
    @psyre_comb(pure=True)
    def opaque(x):
      return x

    g_opaque = build_graph(
        lambda: opaque(Stream([1, 2])) > 0,
        optimize=False,
    )
    self.assertFalse(g_opaque.is_bool_stream(g_opaque.output))

    # Duck-typed array comparison must NOT fold `(arr > 0) & False` to False.
    arr = _StrictArrayDuck((2, 3))

    def fold_prog():
      s = Stream.constant(arr)
      cmp_s = s > 0
      return cmp_s & False, cmp_s | True

    g_opt = build_graph(fold_prog, optimize=True)
    and_s, or_s = g_opt.output
    self.assertFalse(and_s.static)
    self.assertFalse(or_s.static)
    res_and, res_or = Scheduler.run_graph(g_opt, horizon=2)
    self.assertEqual(res_and, [('and', (2, 3), False)] * 2)
    self.assertEqual(res_or, [('or', (2, 3), True)] * 2)

    # Builtin scalar boolean still folds via `optimize_graph` and clocks `~p`.
    def clock_prog():
      x = Stream([1, 2, 3, 4])
      odd_p = (x & 1) == 1
      even_p = ~odd_p
      return when(odd_p, x), whennot(even_p, x)

    g_clk = build_graph(clock_prog, semantics=Semantics.SYNC, optimize=True)
    check_clocks(g_clk)
    o1, o2 = Scheduler.run_graph(g_clk, horizon=4)
    self.assertEqual(o1, [1, Absent, 3, Absent])
    self.assertEqual(o2, [1, Absent, 3, Absent])

  def test_format_tag_and_array_object_without_repr_or_bool(self):
    duck = _StrictArrayDuck((4, 8), dtype='bfloat16')
    self.assertTrue(is_array_object(duck))
    self.assertEqual(format_tag(duck), 'bfloat16[4,8]')
    scalar_duck = _StrictArrayDuck((), dtype='float32')
    self.assertEqual(format_tag(scalar_duck), 'float32[]')
    self.assertIsNone(when_selector_matches_const(None, duck))
    self.assertIsNone(when_selector_matches_const(None, scalar_duck))
    self.assertEqual(
        StaticSequence.constant(duck).format_compact(), 'bfloat16[4,8]'
    )
    self.assertFalse(is_array_object(42))
    self.assertFalse(is_array_object(Stream([1, 2])))
    self.assertFalse(is_array_object(_MinimalThunk(1)))
    self.assertFalse(is_array_object([1, 2]))
    self.assertFalse(is_array_object({'shape': (2,), 'dtype': 'int'}))

    # Equations Thunk desc and repr must not touch duck.__repr__.
    with psyre_equations(optimize=False) as eq:

      class eq_blk:  # pylint: disable=invalid-name
        out = a == duck
        repr_before = repr(out)
        a = Stream.constant(duck)
        __output__ = out

      self.assertIn('bfloat16[4,8]', eq_blk.repr_before)
    self.assertIsNotNone(eq.graph)

    # NaN scalar checks including complex NaN.
    self.assertTrue(is_nan_scalar(float('nan')))
    self.assertTrue(is_nan_scalar(complex(float('nan'), 0.0)))
    self.assertTrue(is_nan_scalar(complex(0.0, float('nan'))))
    self.assertFalse(is_nan_scalar(1.5))
    self.assertFalse(is_nan_scalar(complex(1.0, 2.0)))

  def test_ifthenelse_remains_strictly_scalar(self):
    duck = _StrictArrayDuck((2,), dtype='bool')
    with self.assertRaises(ExceptionGroup) as ctx:
      Scheduler.run(
          lambda: ifthenelse(Stream.constant(duck), Stream(1), Stream(0)),
          horizon=1,
      )
    self.assertIsInstance(ctx.exception.exceptions[0], ValueError)
    self.assertRegex(str(ctx.exception.exceptions[0]), 'ambiguous')

  def test_stream_getitem_setitem_iter_array_and_class_getitem(self):
    alias = Stream[int]
    self.assertIsInstance(alias, types.GenericAlias)
    self.assertEqual(alias.__origin__, Stream)
    self.assertEqual(alias.__args__, (int,))

    s = Stream([(10, 20, 30), (40, 50, 60)])
    with self.assertRaisesRegex(TypeError, 'not iterable'):
      iter(s)
    with self.assertRaisesRegex(TypeError, 'not iterable'):
      _, _ = s
    with self.assertRaisesRegex(TypeError, 'immutable'):
      s[0] = 99
    with self.assertRaisesRegex(TypeError, 'NumPy array'):
      s.__array__()

    def slice_prog():
      vecs = Stream([(10, 20, 30, 40), (1, 2, 3, 4)])
      idx = Stream([0, 2])
      lo = Stream([1, 0])
      hi = Stream([3, 2])
      dicts = Stream([{'k': 7}, {'k': 9}])
      return (
          vecs[0],
          vecs[-1],
          vecs[1:3],
          vecs[idx],
          vecs[lo:hi],
          dicts['k'],
      )

    first, last, sl, dyn, dyn_sl, dk = Scheduler.run(slice_prog, horizon=2)
    self.assertEqual(first, [10, 1])
    self.assertEqual(last, [40, 4])
    self.assertEqual(sl, [(20, 30), (2, 3)])
    self.assertEqual(dyn, [10, 3])
    self.assertEqual(dyn_sl, [(20, 30), (1, 2)])
    self.assertEqual(dk, [7, 9])

    # CSE deduplicates identical static and dynamic indexing projections.
    def cse_prog():
      v = Stream([(1, 2, 3), (4, 5, 6)])
      lo = Stream([0, 1])
      hi = Stream([2, 3])
      return v[0], v[0], v[1:3], v[1:3], v[lo:hi], v[lo:hi]

    g_cse = build_graph(cse_prog, optimize=True)
    a0, a1, s0, s1, d0, d1 = g_cse.output
    self.assertIs(a0, a1)
    self.assertIs(s0, s1)
    self.assertIs(d0, d1)

  def test_stream_sum_avoids_zero_seed_promotion(self):
    str_out = Scheduler.run(
        lambda: Stream(['a', 'x']).sum(Stream(['b', 'y']), Stream(['c', 'z'])),
        horizon=2,
    )
    self.assertEqual(str_out, ['abc', 'xyz'])

    vec_out = Scheduler.run(
        lambda: Stream([_Vec(1, 2), _Vec(3, 4)]).sum(
            Stream([_Vec(10, 20), _Vec(5, 6)])
        ),
        horizon=2,
    )
    self.assertEqual(vec_out, [_Vec(11, 22), _Vec(8, 10)])

    single = Scheduler.run(lambda: Stream([_Vec(7, 8)]).sum(), horizon=1)
    self.assertEqual(single, [_Vec(7, 8)])

  def test_psyre_comb_purity_and_cse(self):
    calls = 0

    @psyre_comb
    def impure_tick(x):
      nonlocal calls
      calls += 1
      return x + calls

    @psyre_comb(pure=True)
    def pure_scale(x, factor=2):
      return x * factor

    self.assertFalse(is_pure_step_func(impure_tick._psyre_step_func))  # pylint: disable=protected-access
    self.assertTrue(is_pure_step_func(pure_scale._psyre_step_func))  # pylint: disable=protected-access

    # Default pure=False prevents CSE on side-effecting @psyre_comb.
    g_imp = build_graph(
        lambda: (impure_tick(Stream([1, 2])), impure_tick(Stream([1, 2]))),
        optimize=True,
    )
    self.assertIsNot(g_imp.output[0], g_imp.output[1])
    t0, t1 = Scheduler.run_graph(g_imp, horizon=2)
    self.assertNotEqual(t0, t1)

    # Pure CSE with kwargs and CombCallPlan matching vs distinct kwargs.
    def pure_prog():
      x = Stream([1, 2, 3])
      a = pure_scale(x, factor=3)
      b = pure_scale(x, factor=3)
      c = pure_scale(x, factor=5)
      return a, b, c

    g_pure = build_graph(pure_prog, optimize=True)
    pa, pb, pc = g_pure.output
    self.assertIs(pa, pb)
    self.assertIsNot(pa, pc)

    # User pure @psyre_comb named 'when' or 'merge' must CSE with itself.
    @psyre_comb(pure=True)
    def when_shadow(x, y):
      return x + y

    when_shadow._psyre_step_func.__name__ = 'when'  # pylint: disable=protected-access

    def shadow_prog():
      a = Stream([1, 2])
      b = Stream([3, 4])
      return when_shadow(a, b), when_shadow(a, b)

    g_shadow = build_graph(shadow_prog, optimize=True)
    self.assertIs(g_shadow.output[0], g_shadow.output[1])

    # Freevar named `inst_func` is preserved for downstream lowering tools.
    def raw(a):
      return a + 1

    sf = make_comb_step_func(raw, pure=True)
    self.assertIn('inst_func', sf.__code__.co_freevars)

  def test_psyre_comb_nout_pytree_static_args_and_custom_node(self):
    @psyre_comb(pure=True, nout=2)
    def split_pair(x):
      return (x + 1, x * 10)

    def nout_prog():
      x = Stream([1, 2, 3])
      q, r = split_pair(x)
      return q, r

    qs, rs = Scheduler.run(nout_prog, horizon=3)
    self.assertEqual(qs, [2, 3, 4])
    self.assertEqual(rs, [10, 20, 30])

    # DCE drops the unused projected output stream and its __getitem__ node.
    def used_only_first():
      x = Stream([1, 2])
      q, unused_r = split_pair(x)
      del unused_r
      return q

    g_dce = build_graph(used_only_first, optimize=True)
    getitem_nodes = [n for n in g_dce.nodes if n.op == '__getitem__']
    self.assertLen(getitem_nodes, 1)

    with self.assertRaisesRegex(ValueError, 'o_stream'):
      build_graph(lambda: split_pair(Stream(1), o_stream=Stream()))
    with self.assertRaises(ValueError):
      psyre_comb(nout=0)
    with self.assertRaises(ValueError):
      psyre_comb(nout=True)

    # Zero-overhead fast path: flat positional args have NO CombCallPlan kwarg.
    @psyre_comb(pure=True)
    def flat_add(a, b, offset=1):
      return a + b + offset

    g_flat = build_graph(
        lambda: flat_add(Stream([1, 2]), Stream([3, 4]), offset=5),
        optimize=False,
    )
    self.assertNotIn(CALL_PLAN_KWARG, g_flat.nodes[0].kwargs)
    self.assertEqual(g_flat.nodes[0].kwargs, {'offset': 5})

    # Nested PyTree inputs (tuple, NamedTuple, list, dict, slice, dynamic kw).
    @psyre_comb(pure=True)
    def combine_tree(tup, pair, items, mapping, *, weight=1, sl=None):
      data = (0, 1, 2, 3, 4, 5)
      return (
          tup[0] + tup[1][0],
          pair.first + pair.second,
          sum(items),
          mapping['a'] * weight,
          len(data[sl]) if sl is not None else 0,
      )

    def tree_prog():
      x = Stream([1, 2])
      y = Stream([10, 20])
      z = Stream([100, 200])
      s_hi = Stream([3, 5])
      return combine_tree(
          (x, [y]),
          _Pair(x, lambda: y),
          [x, 5],
          {'a': z},
          weight=x,
          sl=slice(1, s_hi),
      )

    res = Scheduler.run(tree_prog, horizon=2)
    self.assertEqual(
        res,
        [(11, 11, 6, 100, 2), (22, 22, 7, 400, 4)],
    )

    # static_argnums / static_argnames keep callables & zero-arg lambdas static.
    @psyre_comb(pure=True, static_argnums=1, static_argnames='post_fn')
    def apply_with_fn(x, reducer, *, post_fn):
      return post_fn() + reducer(x)

    out_static = Scheduler.run(
        lambda: apply_with_fn(
            Stream([2, 5]), lambda v: v * 3, post_fn=lambda: 100
        ),
        horizon=2,
    )
    self.assertEqual(out_static, [106, 115])

    # static_literals=True keeps list arguments literal rather than temporal.
    @psyre_comb(pure=True, static_literals=True)
    def add_bias(x, bias):
      return [x + b for b in bias]

    self.assertEqual(
        Scheduler.run(lambda: add_bias(Stream([1, 10]), [2, 3]), horizon=2),
        [[3, 4], [12, 13]],
    )

    # prepare= normalizes arguments before partitioning.
    @psyre_comb(
        pure=True,
        prepare=lambda a, b, scale=1: ((a, b), {'scale': scale * 2}),
    )
    def scaled_add(a, b, *, scale):
      return (a + b) * scale

    self.assertEqual(
        Scheduler.run(
            lambda: scaled_add(Stream([1, 2]), Stream([3, 4]), scale=5),
            horizon=2,
        ),
        [40, 60],
    )

    # Bad static_argnums / static_argnames rejected cleanly.
    with self.assertRaises(ValueError):
      psyre_comb(static_argnums=-1)
    with self.assertRaises(TypeError):
      psyre_comb(static_argnums=True)
    with self.assertRaises(TypeError):
      psyre_comb(static_argnames=[123])  # type: ignore[list-item]

    # Extensible custom PyTree registration via register_pytree_node.
    register_pytree_node(
        _Box,
        lambda box: (box.values, box.tag),
        lambda tag, children: _Box(tuple(children), tag),
    )
    self.addCleanup(unregister_pytree_node, _Box)

    @psyre_comb(pure=True)
    def read_box(box: _Box):
      return (box.tag, box.values[0] + box.values[1])

    def box_prog():
      s1 = Stream([1, 2])
      s2 = Stream([10, 20])
      b1 = read_box(_Box((s1, s2), 'sum'))
      b2 = read_box(_Box((s1, s2), 'sum'))
      b3 = read_box(_Box((s1, s2), 'other'))
      return b1, b2, b3

    g_box = build_graph(box_prog, optimize=True)
    out1, out2, out3 = g_box.output
    self.assertIs(out1, out2)
    self.assertIsNot(out1, out3)
    t1, t2, t3 = Scheduler.run_graph(g_box, horizon=2)
    self.assertEqual(t1, [('sum', 11), ('sum', 22)])
    self.assertEqual(t2, t1)
    self.assertEqual(t3, [('other', 11), ('other', 22)])

    with self.assertRaises(TypeError):
      register_pytree_node('not_a_type', lambda x: ((), None), lambda a, c: a)  # type: ignore[arg-type]
    with self.assertRaises(TypeError):
      register_pytree_node(_Box, 'bad', lambda a, c: a)  # type: ignore[arg-type]

  def test_subclock_optimize_preserves_opaque_call_plan_nodes(self):
    @dataclasses.dataclass(frozen=True)
    class _Spec:
      k: int

    @psyre_comb(pure=True)
    def take_k(pair, *, spec: _Spec):
      return pair[0] + pair[1] + spec.k

    def prog():
      a = Stream([1, 2, 3, 4])
      b = Stream([10, 20, 30, 40])
      c = Stream([True, False, True, False])
      wa = when(c, a)
      wb = when(c, b)
      return take_k((wa, wb), spec=_Spec(5)), take_k((wa, wb), spec=_Spec(5))

    g = build_graph(prog, semantics=Semantics.SYNC, optimize=False)
    optimize_graph(g)
    self.assertIs(g.output[0], g.output[1])
    check_clocks(g)
    r0, r1 = Scheduler.run_graph(g, horizon=4)
    self.assertEqual(r0, [16, Absent, 38, Absent])
    self.assertEqual(r1, r0)

  def test_single_pass_pytree_debug_logging_and_stream_nesting_guards(self):
    # Custom PyTree `flatten_fn` runs strictly once per node (single-pass).
    flatten_calls = 0

    def _flatten_box(box: _Box) -> tuple[tuple[Any, ...], str]:
      nonlocal flatten_calls
      flatten_calls += 1
      return box.values, box.tag

    register_pytree_node(
        _Box,
        _flatten_box,
        lambda tag, children: _Box(tuple(children), tag),
    )
    self.addCleanup(unregister_pytree_node, _Box)
    nested_dyn = _Box((_Box((_Box((Stream([1, 2]),), 'c'),), 'b'),), 'a')
    dyn, _ = partition_comb_args((nested_dyn,), {})
    self.assertEqual(flatten_calls, 3)
    self.assertLen(dyn, 1)

    flatten_calls = 0
    nested_static = _Box((_Box((42,), 'inner'),), 'outer')
    static_in, static_kw = partition_comb_args((nested_static,), {})
    self.assertEqual(flatten_calls, 2)
    self.assertEqual(static_in, (nested_static,))
    self.assertEqual(static_kw, {})

    # Reserved `_psyre_call_plan` kwarg and non-int `static_argnums` fail fast.
    with self.assertRaisesRegex(ValueError, CALL_PLAN_KWARG):
      partition_comb_args((1,), {CALL_PLAN_KWARG: 123})
    for bad_argnums in ('abc', b'ab', bytearray(b'a'), True):
      with self.assertRaises(TypeError):
        normalize_static_argnums(bad_argnums)  # type: ignore[arg-type]

    # Complex signed zeros are distinguished in `same_const_value`.
    self.assertFalse(same_const_value(complex(-0.0, 0.0), complex(0.0, 0.0)))
    self.assertFalse(same_const_value(complex(0.0, -0.0), complex(0.0, 0.0)))
    self.assertTrue(same_const_value(complex(0.0, 0.0), complex(0.0, 0.0)))
    self.assertTrue(same_const_value(complex(-0.0, -0.0), complex(-0.0, -0.0)))

    # Non-identical duck arrays skip `__eq__`; DEBUG logging avoids `__repr__`.
    class _NoEqDuck(_StrictArrayDuck):

      def __eq__(self, other: object) -> Any:
        del other
        raise AssertionError('__eq__ must not run on distinct array objects')

    d1, d2 = _NoEqDuck((2, 3), dtype='bfloat16'), _NoEqDuck((2, 3))
    self.assertTrue(same_const_value(d1, d1))
    self.assertFalse(same_const_value(d1, d2))
    self.assertFalse(matches_tag(d1, d2))
    self.assertIn('bfloat16[2,3]', str(Stream.constant(d1)))

    existing = Stream([1, 2])
    thunk = _MinimalThunk(1)
    self.assertFalse(matches_tag(existing, existing))
    self.assertFalse(matches_tag(thunk, thunk))

    # Nested `Stream` rejection preserves `LazyThunk` error precedence.
    for bad_thunk_call in (
        lambda: Stream(existing, period=thunk),
        lambda: Stream(existing, period=[thunk]),
    ):
      with self.assertRaisesRegex(TypeError, 'LazyThunk'):
        bad_thunk_call()
    for bad_stream_call in (
        lambda: Stream(existing, period=[1, 2]),
        lambda: Stream(1, period=existing),
        lambda: Stream([existing]),
        lambda: Stream.periodic([existing], [0]),
        lambda: StaticWord([existing]),
        lambda: Stream.constant([existing]),
    ):
      with self.assertRaisesRegex(TypeError, 'Stream'):
        bad_stream_call()
    with self.assertRaisesRegex(TypeError, 'Stream'):
      Stream().push(existing, 0)
    with self.assertRaises(TypeError):
      StaticSequence([1], [2])  # type: ignore[arg-type]

    # DEBUG logging across `StreamPort`, `_comb`, and `Scheduler` stays safe.
    @psyre_comb(pure=True, static_argnames='aux')
    def pass_duck(x, aux=None):
      del aux
      return x

    records: list[logging.LogRecord] = []

    class _CapHandler(logging.Handler):

      def emit(self, record: logging.LogRecord) -> None:
        records.append(record)

    handler = _CapHandler()
    logger.addHandler(handler)
    orig_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
      res = Scheduler.run(
          lambda: pass_duck(Stream.constant(d1), aux=d2), horizon=2
      )
      self.assertTrue(all(v is d1 for v in res))
      self.assertTrue(any('bfloat16[2,3]' in r.getMessage() for r in records))
    finally:
      logger.removeHandler(handler)
      logger.setLevel(orig_level)

  def test_psyre_sequ_call_plan_pytrees_nout_and_kwarg_guard(self):
    # Flat calls keep the zero-overhead fast path (no `CombCallPlan` kwarg).
    @psyre_sequ
    def running_flat(x, *, init=0):
      total = init
      while True:
        total += x
        x = yield total

    flat_g = build_graph(lambda: running_flat(Stream([1, 2, 3]), init=10))
    self.assertNotIn(CALL_PLAN_KWARG, flat_g.nodes[0].kwargs)
    self.assertEqual(Scheduler.run_graph(flat_g, horizon=3), [11, 13, 16])

    # Positional PyTrees (`tuple`, `NamedTuple`, `dict`, custom registered node)
    # materialize identically on tick 0, reset ticks, and subsequent ticks under
    # both `SYNC` and `KAHN` semantics, including inside `match()`.
    @dataclasses.dataclass(frozen=True)
    class _SeqBox:
      a: Any
      b: Any
      bias: int

    register_pytree_node(
        _SeqBox,
        lambda v: ((v.a, v.b), v.bias),
        lambda bias, xs: _SeqBox(xs[0], xs[1], bias),
    )
    try:

      @psyre_sequ(nout=2)
      def accum_pair(pair, cfg):
        sum_a = 0
        sum_b = 0
        while True:
          box = cfg['box']
          sum_a += pair.first + box.a + box.bias
          sum_b += pair.second + box.b
          pair, cfg = yield (sum_a, sum_b)

      def prog():
        xs = Stream([1, 2, 3, 4, 5])
        ys = Stream([10, 20, 30, 40, 50])
        rst = Stream([False, False, True, False, False])
        gate = Stream([True, False, True, True, False])
        with reset(rst):
          sa, sb = accum_pair(
              _Pair(xs, ys), {'box': _SeqBox(xs * 2, ys // 10, bias=1)}
          )
        with match(gate) as m:
          with case(True):
            ga, _ = accum_pair(
                _Pair(xs, ys), {'box': _SeqBox(xs, ys, bias=0)}
            )
            m.val = ga
          with case(False):
            m.val = -1
        return sa, sb, m.val

      sync_res = Scheduler.run(prog, horizon=5, semantics=Semantics.SYNC)
      kahn_res = Scheduler.run(prog, horizon=5, semantics=Semantics.KAHN)
      self.assertEqual(sync_res, kahn_res)
      # Reset at t=2 restarts both accumulators from `xs[2]=3`, `ys[2]=30`.
      self.assertEqual(sync_res[0], [4, 11, 10, 23, 39])
      self.assertEqual(sync_res[1], [11, 33, 33, 77, 132])
      self.assertEqual(sync_res[2], [2, -1, 6, 12, -1])

      # Forward-ref `psyre_equations` thunk inside a positional PyTree.
      def eq_prog():
        with psyre_equations() as eq:

          class _(auto_name=False):
            # pylint: disable-next=undefined-variable
            out_a, _ = accum_pair(_Pair(a, b), {'box': _SeqBox(a, b, bias=0)})
            a = Stream([1, 2, 3])
            b = Stream([10, 20, 30])
            __output__ = out_a

        return eq.output

      self.assertEqual(Scheduler.run(eq_prog, horizon=3), [2, 6, 12])
    finally:
      unregister_pytree_node(_SeqBox)

    # `static_argnums`, `static_literals`, and `static_argnames`.
    @psyre_sequ(static_argnums=(1,), static_literals=True)
    def fold_with_fn(x, fn, scale):
      acc = 0
      while True:
        acc = fn(acc, x * scale)
        x, fn, scale = yield acc

    self.assertEqual(
        Scheduler.run(
            lambda: fold_with_fn(Stream([1, 2, 3]), lambda a, b: a + b, 10),
            horizon=3,
        ),
        [10, 30, 60],
    )

    # Dynamic `Stream` or thunk inside keyword args fails fast at Stage 1.
    @psyre_sequ
    def kw_sequ(x, scale=1):
      while True:
        x = yield x * scale

    with self.assertRaisesRegex(TypeError, "keyword argument 'scale'"):
      build_graph(lambda: kw_sequ(Stream([1, 2]), scale=Stream([2, 3])))
    with self.assertRaisesRegex(TypeError, "keyword argument 'scale'"):
      build_graph(
          lambda: kw_sequ(Stream([1, 2]), scale={'k': lambda: Stream([2, 3])})
      )
    with self.assertRaisesRegex(ValueError, 'o_stream='):
      @psyre_sequ(nout=2)
      def two_out(x):
        while True:
          x = yield (x, x)

      build_graph(lambda: two_out(Stream([1]), o_stream=Stream()))
    with self.assertRaisesRegex(ValueError, 'psyre_sequ nout'):
      psyre_sequ(nout=0)


if __name__ == '__main__':
  absltest.main()
