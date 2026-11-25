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
"""Tests for the optional `psyre.jax.lax` stream-lifting module."""

from absl.testing import absltest

# pylint: disable=g-import-not-at-top,g-importing-member,g-multiple-import
try:
  import jax
  from jax import lax as jax_lax
  import jax.numpy as jnp
  import numpy as np
except ImportError:  # pragma: no cover
  jax = None
  jax_lax = None
  jnp = None
  np = None

if jax is not None:
  from psyre.jax import lax as plax
else:  # pragma: no cover
  plax = None

from psyre.builder import build_graph
from psyre.clock import check_clocks
from psyre.common import Absent, Semantics, fby, when
from psyre.equations import psyre_equations
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import


@absltest.skipIf(jax is None, 'jax is not installed')
class PsyreLaxTest(absltest.TestCase):
  """Tests `psyre.jax.lax` lifting, `Primitive` tags, and control ops."""

  def test_primitive_metadata_and_namespace_dir(self):
    self.assertIs(plax.get_primitive(plax.add), jax_lax.add_p)
    self.assertIs(plax.get_primitive(plax.dot_general), jax_lax.dot_general_p)
    self.assertIs(plax.get_primitive(plax.sin), jax_lax.sin_p)
    self.assertIs(plax.get_primitive(plax.top_k), jax_lax.top_k_p)
    self.assertIs(plax.get_primitive(plax.select_n), jax_lax.select_n_p)
    self.assertIs(plax.get_primitive(plax.sort_key_val), jax_lax.sort_p)
    self.assertIs(plax.get_primitive(plax.linalg.qr), jax_lax.linalg.qr_p)
    self.assertIs(plax.get_primitive(plax.linalg.svd), jax_lax.linalg.svd_p)
    self.assertIs(plax.PRIMITIVES['add'], jax_lax.add_p)
    self.assertIs(plax.PRIMITIVES['linalg.qr'], jax_lax.linalg.qr_p)
    self.assertIn('dot_general', dir(plax))
    self.assertIn('linalg', dir(plax))

    # Multi-output `sort_key_val` and `linalg.lu` unpack cleanly.
    ks, vs = Scheduler.run(
        lambda: plax.sort_key_val(
            Stream([jnp.array([3, 1, 2])]), Stream([jnp.array([30, 10, 20])])
        ),
        horizon=1,
    )
    np.testing.assert_array_equal(ks[0], [1, 2, 3])
    np.testing.assert_array_equal(vs[0], [10, 20, 30])
    lu_f, pivots, perm = Scheduler.run(
        lambda: plax.linalg.lu(
            Stream([jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)])
        ),
        horizon=1,
    )
    self.assertEqual(lu_f[0].shape, (2, 2))
    self.assertEqual(pivots[0].shape, (2,))
    self.assertEqual(perm[0].shape, (2,))
    with self.assertRaises(AttributeError):
      _ = plax.nonexistent_lax_symbol_xyz

    def prog():
      x = Stream([jnp.array([1.0, 2.0])])
      y = Stream([jnp.array([3.0, 4.0])])
      return plax.add(x, y)

    g = build_graph(prog, optimize=True)
    self.assertLen(g.nodes, 1)
    self.assertIs(plax.get_primitive(g.nodes[0]), jax_lax.add_p)

  def test_arithmetic_dot_general_broadcast_and_select(self):
    def prog():
      x = Stream([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])])
      w = Stream([jnp.array([[1.0, 0.0], [0.0, 2.0]])])
      cond = Stream([jnp.array([True, False]), jnp.array([False, True])])
      dot = plax.dot_general(
          x, w, dimension_numbers=(((0,), (0,)), ((), ()))
      )
      bcast = plax.broadcast_in_dim(x, shape=(2, 2), broadcast_dimensions=(1,))
      sel = plax.select(cond, x, plax.neg(x))
      zero = plax.iota(jnp.float32, 2)
      return {
          'add': plax.add(x, zero),
          'dot': dot,
          'bcast': bcast,
          'sel': sel,
          'sin': plax.sin(x),
      }

    traces = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(traces['add'][0], [1.0, 3.0])
    np.testing.assert_allclose(traces['dot'][0], [1.0, 4.0])
    np.testing.assert_allclose(traces['dot'][1], [3.0, 8.0])
    np.testing.assert_allclose(traces['bcast'][0], [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(traces['sel'][0], [1.0, -2.0])
    np.testing.assert_allclose(traces['sel'][1], [-3.0, 4.0])
    np.testing.assert_allclose(traces['sin'][0], np.sin([1.0, 2.0]), rtol=1e-5)

    # CSE deduplicates identical pure lifted `lax` ops.
    g_cse = build_graph(
        lambda: (
            (x := Stream([jnp.array([1.0, 2.0])])),
            plax.sin(x),
            plax.sin(x),
            plax.cos(x),
        )[1:],
        optimize=True,
    )
    self.assertIs(g_cse.output[0], g_cse.output[1])
    self.assertIsNot(g_cse.output[0], g_cse.output[2])

  def test_multi_output_top_k_and_linalg_unpacking(self):
    mat = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)

    def prog():
      v = Stream([jnp.array([5.0, 1.0, 9.0, 2.0])])
      vals, idx = plax.top_k(v, 2)
      m = Stream([mat])
      e_vecs, e_vals = plax.linalg.eigh(m)
      qp, rp, perm = plax.linalg.qr(m, pivoting=True)
      lu_mat, piv, lu_perm = plax.linalg.lu(m)
      eig_w, eig_vr = plax.linalg.eig(m, compute_left_eigenvectors=False)
      schur_t, schur_s = plax.linalg.schur(m)
      return {
          'vals': vals,
          'idx': idx,
          'e_vals': e_vals,
          'e_vecs': e_vecs,
          'qp': qp,
          'rp': rp,
          'perm': perm,
          'lu_mat': lu_mat,
          'piv': piv,
          'lu_perm': lu_perm,
          'eig_w': eig_w,
          'eig_vr': eig_vr,
          'schur_t': schur_t,
          'schur_s': schur_s,
      }

    traces = Scheduler.run(prog, horizon=1)
    np.testing.assert_allclose(traces['vals'][0], [9.0, 5.0])
    np.testing.assert_array_equal(traces['idx'][0], [2, 0])
    np.testing.assert_allclose(
        traces['e_vecs'][0]
        @ jnp.diag(traces['e_vals'][0])
        @ traces['e_vecs'][0].T,
        mat,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        traces['qp'][0] @ traces['rp'][0], mat[:, traces['perm'][0]], atol=1e-5
    )
    self.assertEqual(traces['lu_mat'][0].shape, mat.shape)
    self.assertEqual(traces['piv'][0].shape, (2,))
    self.assertEqual(traces['lu_perm'][0].shape, (2,))
    np.testing.assert_allclose(
        np.sort(np.real(traces['eig_w'][0])),
        np.sort(traces['e_vals'][0]),
        atol=1e-5,
    )
    self.assertEqual(traces['eig_vr'][0].shape, mat.shape)
    np.testing.assert_allclose(
        traces['schur_s'][0] @ traces['schur_t'][0] @ traces['schur_s'][0].T,
        mat,
        atol=1e-5,
    )

  def test_higher_order_intra_tick_control_flow(self):
    # Static callables passed to higher-order `lax` control-flow primitives are
    # kept as static operator arguments (valid under first-order streams).
    def scan_step(carry, elem):
      nxt = carry + elem
      return nxt, nxt

    def prog():
      init = Stream([jnp.array(0.0), jnp.array(10.0)])
      xs = Stream([
          jnp.array([1.0, 2.0, 3.0]),
          jnp.array([4.0, 5.0, 6.0]),
      ])
      carry, ys = plax.scan(scan_step, init, xs)
      loop_out = plax.fori_loop(0, 3, lambda i, v: v + i, init)
      w_out = plax.while_loop(lambda v: v < 5.0, lambda v: v + 2.0, init)
      flag = Stream([True, False])
      c_out = plax.cond(
          flag, lambda a: a * 10.0, lambda a: a - 1.0, init
      )
      br_idx = Stream([0, 2])
      sw_out = plax.switch(
          br_idx,
          (lambda a: a + 1.0, lambda a: a + 2.0, lambda a: a + 3.0),
          init,
      )
      red_out = plax.reduce(xs, jnp.array(0.0), jax_lax.add, (0,))
      map_out = plax.map(lambda e: e * e, xs)
      ascan_out = plax.associative_scan(jax_lax.add, xs)
      return {
          'carry': carry,
          'ys': ys,
          'loop': loop_out,
          'while': w_out,
          'cond': c_out,
          'switch': sw_out,
          'reduce': red_out,
          'map': map_out,
          'ascan': ascan_out,
      }

    traces = Scheduler.run(prog, horizon=2)
    self.assertAlmostEqual(float(traces['carry'][0]), 6.0)
    self.assertAlmostEqual(float(traces['carry'][1]), 25.0)
    np.testing.assert_allclose(traces['ys'][0], [1.0, 3.0, 6.0])
    np.testing.assert_allclose(traces['ys'][1], [14.0, 19.0, 25.0])
    self.assertAlmostEqual(float(traces['loop'][0]), 3.0)
    self.assertAlmostEqual(float(traces['loop'][1]), 13.0)
    self.assertAlmostEqual(float(traces['while'][0]), 6.0)
    self.assertAlmostEqual(float(traces['while'][1]), 10.0)
    self.assertAlmostEqual(float(traces['cond'][0]), 0.0)
    self.assertAlmostEqual(float(traces['cond'][1]), 9.0)
    self.assertAlmostEqual(float(traces['switch'][0]), 1.0)
    self.assertAlmostEqual(float(traces['switch'][1]), 13.0)
    self.assertAlmostEqual(float(traces['reduce'][0]), 6.0)
    np.testing.assert_allclose(traces['map'][0], [1.0, 4.0, 9.0])
    np.testing.assert_allclose(traces['ascan'][0], [1.0, 3.0, 6.0])

  def test_subclock_and_equations_forward_refs(self):
    # pylint: disable=used-before-assignment,invalid-name
    with psyre_equations() as prog:

      class block:
        y = plax.add(x, prev_y)
        prev_y = fby(jnp.array([0.0, 0.0]), y)
        vals, idx = plax.top_k(y, 1)
        x = Stream([jnp.array([1.0, 3.0]), jnp.array([2.0, 1.0])])
        __output__ = (y, vals, idx)

      self.assertIsNotNone(block)
    # pylint: enable=used-before-assignment,invalid-name

    y_t, v_t, i_t = Scheduler.run_graph(prog.graph, horizon=2)
    np.testing.assert_allclose(y_t[0], [1.0, 3.0])
    np.testing.assert_allclose(y_t[1], [3.0, 4.0])
    np.testing.assert_allclose(v_t[1], [4.0])
    np.testing.assert_array_equal(i_t[1], [1])

    def sub_prog():
      c = Stream([True, False, True])
      x = Stream([jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)])
      return plax.exp(when(c, x + 0.0))

    g = build_graph(sub_prog, semantics=Semantics.SYNC, optimize=True)
    out = Scheduler.run_graph(g, horizon=3, clocks=check_clocks(g))
    self.assertIs(out[1], Absent)
    np.testing.assert_allclose(out[2], np.exp(3.0), rtol=1e-5)

    # KAHN execution elides Absent ticks cleanly on subclocked lax ops.
    k_out = Scheduler.run(sub_prog, horizon=3, semantics=Semantics.KAHN)
    self.assertLen(k_out, 2)
    np.testing.assert_allclose(k_out[0], np.exp(1.0), rtol=1e-5)
    np.testing.assert_allclose(k_out[1], np.exp(3.0), rtol=1e-5)

  def test_zero_stream_eager_returns_array(self):
    res = plax.add(jnp.array(2.0), jnp.array(3.0))
    self.assertIsInstance(res, jax.Array)
    self.assertAlmostEqual(float(res), 5.0)


if __name__ == '__main__':
  absltest.main()
