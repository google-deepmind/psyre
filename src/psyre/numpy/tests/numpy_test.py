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
"""Tests for the optional `psyre.numpy` stream-lifting module."""

import collections
import os
import subprocess
import sys

from absl.testing import absltest

# pylint: disable=g-import-not-at-top,g-importing-member,g-multiple-import
try:
  import numpy as np
except ImportError:  # pragma: no cover
  np = None

if np is not None:
  from psyre import numpy as pnp
else:  # pragma: no cover
  pnp = None

from psyre.automaton import automaton, state, until
from psyre.builder import build_graph
from psyre.clock import check_clocks
from psyre.common import (
    Absent,
    Semantics,
    fby,
    ifthenelse,
    psyre_comb,
    when,
)
from psyre.equations import psyre_equations
from psyre.match import case, match
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import

_Pair = collections.namedtuple('_Pair', ['total', 'scaled'])


def _node_count_by_name(graph, prefix: str) -> int:
  return sum(1 for n in graph.nodes if (n.name or '').startswith(prefix))


@absltest.skipIf(np is None, 'numpy is not installed')
class PsyreNumpyTest(absltest.TestCase):
  """Covers `psyre.numpy` lifting, ufuncs, CSE, `equations`, and `vmap`."""

  def test_core_import_does_not_load_numpy_in_fresh_python(self):
    # Skip inside embedded Python runtimes where `sys.executable` is None.
    if not sys.executable:
      self.skipTest('sys.executable is unavailable in embedded runtime.')
    code = (
        'import sys, psyre, psyre.equations; '
        'assert "numpy" not in sys.modules, "numpy unexpectedly imported"; '
        'assert "jax" not in sys.modules, "jax unexpectedly imported"'
    )
    # Ensure non-editable checkout / src paths remain importable in subprocess.
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(p for p in sys.path if p)
    res = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)

  def test_namespace_constants_dtypes_and_dir(self):
    self.assertAlmostEqual(pnp.pi, np.pi)
    self.assertIs(pnp.float32, np.float32)
    self.assertIs(pnp.newaxis, np.newaxis)
    self.assertIs(pnp.ndarray, np.ndarray)
    names = dir(pnp)
    for expected in ('lift', 'linalg', 'fft', 'random', 'sin', 'vmap'):
      self.assertIn(expected, names)
    with self.assertRaises(AttributeError):
      _ = pnp.definitely_nonexistent_attr_xyz
    self.assertIn('LiftedNamespace', repr(pnp.linalg))
    self.assertIn('LiftedUfunc', repr(pnp.sin))

  def test_ufunc_call_and_methods(self):
    def prog():
      a = Stream([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
      b = Stream([np.array([10.0, 20.0]), np.array([30.0, 40.0])])
      return {
          'sin': pnp.sin(a),
          'add': pnp.add(a, b),
          'add_same': pnp.add(a, a),
          'red': pnp.add.reduce(a),
          'acc': pnp.add.accumulate(a),
          'outer': pnp.multiply.outer(a, b),
          'redat': pnp.add.reduceat(b, [0, 1]),
          'div_q': pnp.divmod(b, a)[0],
          'div_r': pnp.divmod(b, a)[1],
      }

    traces = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(traces['sin'][0], np.sin([1.0, 2.0]))
    np.testing.assert_allclose(traces['sin'][1], np.sin([3.0, 4.0]))
    np.testing.assert_allclose(traces['add'][0], [11.0, 22.0])
    np.testing.assert_allclose(traces['add_same'][0], [2.0, 4.0])
    self.assertAlmostEqual(float(traces['red'][0]), 3.0)
    self.assertAlmostEqual(float(traces['red'][1]), 7.0)
    np.testing.assert_allclose(traces['acc'][1], [3.0, 7.0])
    np.testing.assert_allclose(
        traces['outer'][0], np.outer([1.0, 2.0], [10.0, 20.0])
    )
    np.testing.assert_allclose(traces['redat'][0], [10.0, 20.0])
    np.testing.assert_allclose(traces['div_q'][0], [10.0, 10.0])
    np.testing.assert_allclose(traces['div_r'][0], [0.0, 0.0])

  def test_ufunc_null_out_and_where_stream(self):
    def prog():
      a = Stream([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
      b = Stream([np.array([10.0, 20.0]), np.array([30.0, 40.0])])
      cond = Stream([np.array([True, False]), np.array([False, True])])
      return {
          'null_out': pnp.add(a, b, out=None),
          'where_s': pnp.where(cond, a, b),
          'masked': pnp.add(a, b, where=cond),
          'zeros_like': pnp.zeros_like(a),
      }

    traces = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(traces['null_out'][0], [11.0, 22.0])
    np.testing.assert_allclose(traces['where_s'][0], [1.0, 20.0])
    np.testing.assert_allclose(traces['where_s'][1], [30.0, 4.0])
    self.assertEqual(traces['masked'][0][0], 11.0)
    self.assertEqual(traces['masked'][1][1], 44.0)
    np.testing.assert_allclose(traces['zeros_like'][0], [0.0, 0.0])

  def test_ufunc_rejects_inplace_at_and_out(self):
    buf = np.zeros(2)

    def bad_at():
      x = Stream([np.array([1.0, 2.0])])
      return pnp.add.at(x, 0, 1.0)

    def bad_out_ufunc():
      x = Stream([np.array([1.0, 2.0])])
      return pnp.sin(x, out=buf)

    def bad_out_tuple():
      x = Stream([np.array([1.0, 2.0])])
      return pnp.divmod(x, 1.0, out=(buf, None))

    def bad_out_func():
      x = Stream([np.array([1.0, 2.0])])
      return pnp.sum(x, out=np.zeros(()))

    for prog in (bad_at, bad_out_ufunc, bad_out_tuple, bad_out_func):
      with self.assertRaises(TypeError):
        build_graph(prog)

  def test_mutating_numpy_functions_rejected(self):
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    calls = (
        lambda s: pnp.copyto(s, s),
        lambda s: pnp.fill_diagonal(s, 0.0),
        lambda s: pnp.place(s, s > 0, 0.0),
        lambda s: pnp.put(s, 0, 0.0),
        lambda s: pnp.put_along_axis(s, np.array([0]), 0.0, axis=0),
        lambda s: pnp.putmask(s, s > 0, 0.0),
        pnp.random.shuffle,
    )
    for fn_call in calls:
      with self.assertRaises(TypeError):
        build_graph(lambda fn=fn_call: fn(Stream([mat])))

  def test_array_coercion_rejected_on_stream_but_eager_on_literals(self):
    arr = pnp.array([1.0, 2.0])
    self.assertIsInstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, [1.0, 2.0])

    for fn in (
        pnp.array,
        pnp.asarray,
        pnp.asanyarray,
        pnp.ascontiguousarray,
        pnp.asfortranarray,
    ):
      with self.assertRaises(TypeError):
        build_graph(lambda f=fn: f(Stream([1.0, 2.0])))

    # User functions named like mutating NumPy APIs or taking `out=` work.
    def put(x, out=10):
      return x if out is None else x + out

    lifted_put = pnp.lift(put, pure=True)
    self.assertEqual(lifted_put(2, out=5), 7)
    self.assertEqual(lifted_put(2, out=None), 2)
    self.assertEqual(
        Scheduler.run(lambda: lifted_put(Stream([1, 2]), out=3), horizon=2),
        [4, 5],
    )

    # A user function merely named `array` wrapped with `lift` is NOT rejected.
    def array(x):
      return x + 1

    lifted_array = pnp.lift(array, pure=True)
    traces = Scheduler.run(
        lambda: {'y': lifted_array(Stream([1, 2]))}, horizon=2
    )
    self.assertEqual(traces['y'], [2, 3])

  def test_multi_output_unpacking_and_dce(self):
    mat = np.array([[4.0, 1.0], [1.0, 3.0]])

    def prog():
      a = Stream([np.array([5.0, 7.0])])
      b = Stream([np.array([2.0, 3.0])])
      q, r = pnp.divmod(a, b)
      frac, whole = pnp.modf(Stream([np.array([1.25, -2.5])]))
      mant, exp = pnp.frexp(Stream([np.array([8.0, 1.0])]))
      m = Stream([mat])
      sign, logdet = pnp.linalg.slogdet(m)
      vals, vecs = pnp.linalg.eigh(m)
      ev, _ = pnp.linalg.eig(m)
      qm, rm = pnp.linalg.qr(m)
      qh, qtau = pnp.linalg.qr(m, mode='raw')
      r_only = pnp.linalg.qr(m, mode='r')
      u, s, vt = pnp.linalg.svd(m)
      s_only = pnp.linalg.svd(m, compute_uv=False)
      sol, res, rank, sv = pnp.linalg.lstsq(
          m, Stream([np.array([1.0, 2.0])]), rcond=None
      )
      return {
          'q': q,
          'r': r,
          'frac': frac,
          'whole': whole,
          'mant': mant,
          'exp': exp,
          'sign': sign,
          'logdet': logdet,
          'vals': vals,
          'vecs': vecs,
          'ev': ev,
          'qm': qm,
          'rm': rm,
          'qh': qh,
          'qtau': qtau,
          'r_only': r_only,
          'u': u,
          's': s,
          'vt': vt,
          's_only': s_only,
          'sol': sol,
          'res': res,
          'rank': rank,
          'sv': sv,
      }

    traces = Scheduler.run(prog, horizon=1)
    np.testing.assert_allclose(traces['q'][0], [2.0, 2.0])
    np.testing.assert_allclose(traces['r'][0], [1.0, 1.0])
    np.testing.assert_allclose(traces['frac'][0], [0.25, -0.5])
    np.testing.assert_allclose(traces['whole'][0], [1.0, -2.0])
    np.testing.assert_allclose(traces['mant'][0], [0.5, 0.5])
    np.testing.assert_array_equal(traces['exp'][0], [4, 1])
    self.assertAlmostEqual(float(traces['sign'][0]), 1.0)
    self.assertAlmostEqual(float(traces['logdet'][0]), np.log(11.0))
    np.testing.assert_allclose(
        traces['qm'][0] @ traces['rm'][0], mat, atol=1e-6
    )
    np.testing.assert_allclose(traces['r_only'][0], traces['rm'][0], atol=1e-6)
    self.assertEqual(traces['qh'][0].shape, mat.shape)
    self.assertEqual(traces['qtau'][0].shape, (2,))
    np.testing.assert_allclose(
        traces['u'][0] @ np.diag(traces['s'][0]) @ traces['vt'][0],
        mat,
        atol=1e-6,
    )
    np.testing.assert_allclose(traces['s_only'][0], traces['s'][0], atol=1e-6)
    np.testing.assert_allclose(
        traces['vecs'][0] @ np.diag(traces['vals'][0]) @ traces['vecs'][0].T,
        mat,
        atol=1e-6,
    )
    self.assertEqual(int(traces['rank'][0]), 2)
    self.assertEqual(traces['res'][0].size, 0)
    np.testing.assert_allclose(traces['sv'][0], traces['s'][0], atol=1e-6)
    np.testing.assert_allclose(
        np.sort(traces['ev'][0]), np.sort(traces['vals'][0]), atol=1e-6
    )
    np.testing.assert_allclose(mat @ traces['sol'][0], [1.0, 2.0], atol=1e-6)

    # DCE strips unused projection nodes from multi-output ops.
    def keep_one():
      m = Stream([mat])
      _, unused_rm = pnp.linalg.qr(m)
      del unused_rm
      u, _, _ = pnp.linalg.svd(m)
      return {'u': u}

    opt = build_graph(keep_one, optimize=True)
    self.assertEqual(_node_count_by_name(opt, 'qr'), 0)
    self.assertEqual(_node_count_by_name(opt, 'svd'), 1)

  def test_zero_stream_eager_constants_and_polymorphism(self):
    # Outside or inside Stage 1, pure calls with no streams return plain values.
    z = pnp.zeros((2, 2))
    self.assertIsInstance(z, np.ndarray)
    self.assertAlmostEqual(float(pnp.sin(0.5)), float(np.sin(0.5)))

    def prog():
      mode = Stream([True, False, True, False])
      x = Stream([
          np.array([1.0, 2.0]),
          np.array([3.0, 4.0]),
          np.array([5.0, 6.0]),
          np.array([7.0, 8.0]),
      ])
      x_dyn = x + 0.0
      # Inside match arms, `pnp.zeros(2)` / `pnp.ones(2)` are clock-polymorphic.
      with match(mode) as m:
        with case(True):
          m.y = x_dyn + pnp.ones(2)
        with case(False):
          m.y = fby(pnp.zeros(2), x_dyn)
      return {'y': m.y}

    traces = Scheduler.run(prog, horizon=4)
    np.testing.assert_allclose(traces['y'][0], [2.0, 3.0])
    np.testing.assert_allclose(traces['y'][1], [0.0, 0.0])
    np.testing.assert_allclose(traces['y'][2], [6.0, 7.0])
    np.testing.assert_allclose(traces['y'][3], [3.0, 4.0])

  def test_nested_stream_containers_fft_and_zero_arg_thunks(self):
    def prog():
      a = Stream([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
      b = Stream([np.array([5.0, 6.0]), np.array([7.0, 8.0])])
      cat = pnp.concatenate([a, b])
      sel = pnp.select(
          [a > 2.0, b > 5.0],
          [a * 10.0, b * 100.0],
          default=0.0,
      )
      spec = pnp.fft.rfft(a)
      lazy_sin = pnp.sin(lambda: a)
      return {'cat': cat, 'sel': sel, 'spec': spec, 'lazy_sin': lazy_sin}

    traces = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(traces['cat'][0], [1.0, 2.0, 5.0, 6.0])
    np.testing.assert_allclose(traces['cat'][1], [3.0, 4.0, 7.0, 8.0])
    np.testing.assert_allclose(traces['sel'][0], [0.0, 600.0])
    np.testing.assert_allclose(traces['sel'][1], [30.0, 40.0])
    np.testing.assert_allclose(traces['spec'][0], np.fft.rfft([1.0, 2.0]))
    np.testing.assert_allclose(traces['lazy_sin'][0], np.sin([1.0, 2.0]))

  def test_cse_shares_ufunc_and_array_func_identity(self):
    def prog_ufunc():
      x = Stream([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
      return {'a': pnp.sin(x), 'b': pnp.sin(x), 'c': np.sin(x)}

    g_ufunc = build_graph(prog_ufunc, optimize=True)
    self.assertLen(g_ufunc.nodes, 1)
    self.assertIs(g_ufunc.output['a'], g_ufunc.output['b'])
    self.assertIs(g_ufunc.output['a'], g_ufunc.output['c'])

    def prog_sum():
      m = Stream([np.array([[1.0, 2.0], [3.0, 4.0]])])
      s0a = pnp.sum(m, axis=0)
      s0b = pnp.sum(m, axis=0)
      s1 = pnp.sum(m, axis=1)
      return {'s0a': s0a, 's0b': s0b, 's1': s1}

    g_sum = build_graph(prog_sum, optimize=True)
    self.assertIs(g_sum.output['s0a'], g_sum.output['s0b'])
    self.assertIsNot(g_sum.output['s0a'], g_sum.output['s1'])

  def test_impure_empty_and_random_not_cse_and_tick_per_step(self):
    def prog_empty():
      x = Stream([np.array([1.0, 2.0])])
      return {'a': pnp.empty_like(x), 'b': pnp.empty_like(x)}

    g_empty = build_graph(prog_empty, optimize=True)
    self.assertIsNot(g_empty.output['a'], g_empty.output['b'])

    # `default_rng` returns an eager `Generator`; its samplers lift as impure.
    rng = pnp.random.default_rng(123)
    self.assertIsInstance(rng, np.random.Generator)
    g_gen = build_graph(
        lambda: (pnp.lift(rng.random)(2), pnp.lift(rng.random)(2))
    )
    self.assertLen(g_gen.nodes, 2)
    with self.assertRaisesRegex(TypeError, 'In-place NumPy random'):
      pnp.lift(rng.shuffle)(np.arange(3))
    with self.assertRaises(TypeError):
      build_graph(lambda: pnp.sum(Stream([np.array([1.0])]), bogus_kwarg=1))

    def prog_rand():
      return {
          'a': pnp.random.random(4),
          'b': pnp.random.random(4),
      }

    g_rand = build_graph(prog_rand, optimize=True)
    self.assertIsNot(g_rand.output['a'], g_rand.output['b'])
    self.assertLen(g_rand.nodes, 2)

    traces = Scheduler.run(prog_rand, horizon=3)
    self.assertEqual(traces['a'][0].shape, (4,))
    self.assertFalse(np.array_equal(traces['a'][0], traces['a'][1]))
    self.assertFalse(np.array_equal(traces['a'][0], traces['b'][0]))

  def test_custom_lift_options(self):
    def weighted(x, scale, *, op):
      return op(x) * scale, x + scale

    lifted = pnp.lift(
        weighted,
        pure=True,
        nout=2,
        static_argnums=(),
        static_argnames=('op',),
        name='weighted_pair',
    )

    def prog():
      x = Stream([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
      a, b = lifted(x, 2.0, op=np.square)
      a2, _ = lifted(x, 2.0, op=np.square)
      return {'a': a, 'b': b, 'a2': a2}

    g = build_graph(prog, optimize=True)
    self.assertIs(g.output['a'], g.output['a2'])
    traces = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(traces['a'][0], [2.0, 8.0])
    np.testing.assert_allclose(traces['b'][1], [5.0, 6.0])

    # Eager zero-stream invocation returns raw tuple.
    y1, y2 = lifted(np.array([2.0, 3.0]), 10.0, op=np.square)
    np.testing.assert_allclose(y1, [40.0, 90.0])
    np.testing.assert_allclose(y2, [12.0, 13.0])

    with self.assertRaises(TypeError):
      pnp.lift(123)  # type: ignore[arg-type]
    with self.assertRaises(ValueError):
      pnp.lift(np.sin, nout=0)

  def test_equations_forward_references(self):
    mat = np.array([[3.0, 1.0], [1.0, 2.0]])

    # pylint: disable=used-before-assignment,invalid-name
    with psyre_equations() as prog:

      class block:
        s = pnp.sin(x)
        q, r = pnp.divmod(x, 2.0)
        cat = pnp.concatenate([x, prev_x])
        nrm = pnp.linalg.norm(x)
        u, sv, vt = pnp.linalg.svd(m)
        prev_x = fby(np.array([0.0, 0.0]), x)
        x = Stream([np.array([3.0, 4.0]), np.array([5.0, 12.0])])
        m = Stream([mat])
        __output__ = (s, q, r, cat, nrm, sv)

      self.assertIsNotNone(block)
    # pylint: enable=used-before-assignment,invalid-name

    s_t, q_t, r_t, cat_t, nrm_t, sv_t = Scheduler.run_graph(
        prog.graph, horizon=2
    )
    np.testing.assert_allclose(s_t[0], np.sin([3.0, 4.0]))
    np.testing.assert_allclose(q_t[1], [2.0, 6.0])
    np.testing.assert_allclose(r_t[0], [1.0, 0.0])
    np.testing.assert_allclose(cat_t[0], [3.0, 4.0, 0.0, 0.0])
    np.testing.assert_allclose(cat_t[1], [5.0, 12.0, 3.0, 4.0])
    self.assertAlmostEqual(float(nrm_t[0]), 5.0)
    self.assertAlmostEqual(float(nrm_t[1]), 13.0)
    np.testing.assert_allclose(sv_t[0], np.linalg.svd(mat, compute_uv=False))

  def test_sync_subclock_automaton_and_kahn_semantics(self):
    calls = []

    def tracked_sin(x):
      calls.append(x.copy())
      return np.sin(x)

    lifted_sin = pnp.lift(tracked_sin, pure=False)

    def prog():
      toggle = Stream([False, True, False, True])
      x = Stream([np.array([float(i), float(i + 1)]) for i in range(4)])
      x_dyn = x + 0.0
      with automaton(init='idle') as a:
        with state('idle'):
          a.val = Stream.constant(pnp.zeros(2))
          until(toggle, 'active')
        with state('active'):
          a.val = lifted_sin(x_dyn)
          until(toggle, 'idle')
      return {'out': a.val}

    t_sync = Scheduler.run(prog, horizon=4, semantics=Semantics.SYNC)
    # Ticks 0, 1 are idle (weak transition fires after tick 1); ticks 2, 3 are
    # active. `tracked_sin` must skip idle ticks completely.
    self.assertLen(calls, 2)
    np.testing.assert_allclose(t_sync['out'][0], [0.0, 0.0])
    np.testing.assert_allclose(t_sync['out'][2], np.sin([2.0, 3.0]))

    calls.clear()
    t_kahn = Scheduler.run(prog, horizon=4, semantics=Semantics.KAHN)
    for k in range(4):
      np.testing.assert_allclose(t_kahn['out'][k], t_sync['out'][k])

    # Verify subclocked Absent propagation stays clean in SYNC.
    def sub_prog():
      c = Stream([True, False, True])
      x = Stream([
          np.array([1.0, 2.0]),
          np.array([3.0, 4.0]),
          np.array([5.0, 6.0]),
      ])
      return {'sub': pnp.sin(when(c, x + 0.0))}

    g = build_graph(sub_prog, semantics=Semantics.SYNC, optimize=True)
    clocks = check_clocks(g)
    sub_traces = Scheduler.run_graph(g, horizon=3, clocks=clocks)
    self.assertIs(sub_traces['sub'][1], Absent)
    np.testing.assert_allclose(sub_traces['sub'][2], np.sin([5.0, 6.0]))

  def test_stack_streams_and_unstack_stream(self):
    def prog():
      a = Stream([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
      b = Stream([np.array([5.0, 6.0]), np.array([7.0, 8.0])])
      st0 = pnp.stack_streams(a, b, axis=0)
      st1 = pnp.stack_streams(a, b, axis=1)
      u0, u1 = pnp.unstack_stream(st0, size=2, axis=0)
      v0, v1 = pnp.unstack_stream(st1, size=2, axis=1)
      st_lit = pnp.stack_streams(a, [9.0, 10.0], axis=0)
      return {
          'st0': st0,
          'st1': st1,
          'u0': u0,
          'u1': u1,
          'v0': v0,
          'v1': v1,
          'st_lit': st_lit,
      }

    traces = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(traces['st0'][0], [[1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_allclose(traces['st1'][0], [[1.0, 5.0], [2.0, 6.0]])
    np.testing.assert_allclose(traces['u0'][1], [3.0, 4.0])
    np.testing.assert_allclose(traces['u1'][1], [7.0, 8.0])
    np.testing.assert_allclose(traces['v0'][0], [1.0, 2.0])
    np.testing.assert_allclose(traces['v1'][0], [5.0, 6.0])
    np.testing.assert_allclose(traces['st_lit'][0], [[1.0, 2.0], [9.0, 10.0]])

    # Identical `stack_streams` calls CSE cleanly.
    def dup_stack():
      a = Stream([np.ones(2)])
      b = Stream([np.zeros(2)])
      return {
          'x': pnp.stack_streams(a, b),
          'y': pnp.stack_streams(a, b),
      }

    g = build_graph(dup_stack, optimize=True)
    self.assertIs(g.output['x'], g.output['y'])

    with self.assertRaises(ValueError):
      pnp.stack_streams()
    with self.assertRaises(TypeError):
      pnp.stack_streams(Stream([1.0]), axis=True)
    with self.assertRaises(ValueError):
      pnp.unstack_stream(Stream([np.ones(2)]), size=0)
    with self.assertRaises(TypeError):
      pnp.unstack_stream(Stream([np.ones(2)]), size=2, axis=False)
    with self.assertRaises(ExceptionGroup) as ctx:
      Scheduler.run(
          lambda: {'u': pnp.unstack_stream(Stream([np.ones((3, 2))]), 2)[0]},
          horizon=1,
      )
    self.assertIsInstance(ctx.exception.exceptions[0], ValueError)

  def test_batch_streams_heterogeneous_per_lane_automata(self):
    def lane_fsm(x):
      with automaton(init='low') as a:
        with state('low'):
          a.val = fby(np.array(0.0), lambda: a.val + x)
          until(x > 5.0, 'high')
        with state('high'):
          a.val = Stream.constant(np.array(-1.0))
      return a.val

    def prog():
      xs = Stream([
          np.array([6.0, 1.0, 2.0]),
          np.array([1.0, 7.0, 2.0]),
          np.array([1.0, 1.0, 2.0]),
      ])
      return pnp.batch_streams(lane_fsm, xs, size=3, name='np_lanes')

    ts = Scheduler.run(prog, horizon=3, semantics=Semantics.SYNC)
    np.testing.assert_allclose(ts[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ts[1], [-1.0, 1.0, 2.0])
    np.testing.assert_allclose(ts[2], [-1.0, -1.0, 4.0])

  def test_vmap_axes_containers_and_errors(self):
    @psyre_comb(pure=True)
    def affine(x, scale, bias):
      return x * scale + bias

    def multi(x, w):
      return _Pair(total=np.sum(x * w), scaled=x * 2.0)

    def dict_out(x):
      return {'sq': [x * x, x + 1.0]}

    def prog():
      xs = Stream([
          np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
          np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
      ])
      bias = Stream([np.array([10.0, 20.0])])
      ys = pnp.vmap(affine, in_axes=(0, None, None))(xs, 2.0, bias)
      ys_t = pnp.vmap(lambda v: v * 10.0, in_axes=1, out_axes=1)(
          Stream([np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])])
      )
      pair_t, pair_s = pnp.vmap(multi, in_axes=(0, None), nout=2)(
          xs, np.array([1.0, 2.0])
      )
      dout = pnp.vmap(dict_out)(xs)
      return {
          'ys': ys,
          'ys_t': ys_t,
          'pair_t': pair_t,
          'pair_s': pair_s,
          'dout': dout,
      }

    traces = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(
        traces['ys'][0], [[12.0, 24.0], [16.0, 28.0], [20.0, 32.0]]
    )
    np.testing.assert_allclose(
        traces['ys_t'][0], [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    )
    np.testing.assert_allclose(traces['pair_t'][0], [5.0, 11.0, 17.0])
    np.testing.assert_allclose(
        traces['pair_s'][1], [[4.0, 0.0], [0.0, 4.0], [2.0, 2.0]]
    )
    self.assertIn('sq', traces['dout'][0])
    np.testing.assert_allclose(
        traces['dout'][0]['sq'][0], [[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]]
    )

    # Zero-stream eager call returns plain array.
    eager = pnp.vmap(lambda x: x + 1.0)(np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(eager, [11.0, 21.0, 31.0])

    # Validation & runtime errors.
    with self.assertRaises(TypeError):
      pnp.vmap(123)  # type: ignore[arg-type]
    with self.assertRaises(TypeError):
      pnp.vmap(lambda x: x, out_axes=True)  # type: ignore[arg-type]
    with self.assertRaises(ValueError):
      pnp.vmap(lambda x, y: x + y, in_axes=(0, 0))(
          np.ones((2, 3)), np.ones((4, 3))
      )
    with self.assertRaises(ValueError):
      pnp.vmap(lambda x: x, in_axes=None)(np.ones(3))
    with self.assertRaises(ValueError):
      pnp.vmap(lambda x: x)(np.array(5.0))
    with self.assertRaises(ValueError):
      pnp.vmap(lambda x: x)(np.zeros((0, 2)))
    with self.assertRaises(ValueError):
      pnp.vmap(lambda x, y: x + y, in_axes=(0,))(np.ones(2), np.ones(2))
    lane_idx = iter((0, 1))
    with self.assertRaisesRegex(ValueError, 'mismatched'):
      build_graph(
          lambda: pnp.batch_streams(
              lambda x: (x,) if next(lane_idx) == 0 else (x, x),
              Stream([np.ones(2)]),
              size=2,
          )
      )

  def test_ifthenelse_remains_strictly_scalar(self):
    # Scalar `ifthenelse` succeeds; vector predicate is rejected (use `where`).
    def ok_prog():
      a = Stream([np.array([1.0, 2.0])])
      b = Stream([np.array([3.0, 4.0])])
      return ifthenelse(Stream([True]), a, b)

    np.testing.assert_allclose(Scheduler.run(ok_prog, horizon=1)[0], [1.0, 2.0])

    def bad_prog():
      a = Stream([np.array([1.0, 2.0])])
      b = Stream([np.array([3.0, 4.0])])
      return ifthenelse(a < b, a, b)

    with self.assertRaises(ExceptionGroup) as ctx:
      Scheduler.run(bad_prog, horizon=1)
    self.assertIsInstance(ctx.exception.exceptions[0], ValueError)


if __name__ == '__main__':
  absltest.main()
