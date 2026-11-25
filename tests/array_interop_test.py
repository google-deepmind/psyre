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
"""Integration tests verifying NumPy (NEP-13/18) and JAX interop readiness."""

# pylint: disable=used-before-assignment

from absl.testing import absltest

# pylint: disable=g-import-not-at-top
try:
  import numpy as np
except ImportError:
  np = None

try:
  import jax
  from jax import lax
  import jax.numpy as jnp
except ImportError:
  jax = None
  lax = None
  jnp = None

# pylint: disable=g-importing-member,g-multiple-import
from psyre.automaton import automaton, state, until
from psyre.builder import build_graph
from psyre.clock import check_clocks
from psyre.common import (
    Absent,
    Semantics,
    fby,
    format_tag,
    ifthenelse,
    matches_tag,
    merge,
    psyre_comb,
    tag_key,
    when,
    when_selector_matches_const,
)
from psyre.equations import psyre_equations
from psyre.match import case, default, match
from psyre.payload import (
    is_array_object,
    is_nan_scalar,
    is_numpy_bool,
    is_numpy_scalar,
)
from psyre.scheduler import Scheduler
from psyre.static_stream import same_const_value
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import


@absltest.skipIf(np is None, 'numpy is not installed')
class NumpyInteropTest(absltest.TestCase):
  """Tests NEP-13/NEP-18 auto-lifting and NumPy scalar/array handling."""

  def test_numpy_ufunc_nep13_lifting_and_reflected_ops(self):
    bias = np.array([10.0, 20.0])
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])

    def prog():
      x = Stream([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
      y = Stream([np.array([2.0, 5.0]), np.array([1.0, 2.0])])
      return (
          np.sin(x),
          bias + x,
          x + bias,
          np.add(x, [100.0, 200.0]),
          mat @ x,
          np.add.reduce(x),
          np.add.accumulate(x),
          np.multiply.outer(x, y),
      )

    sin_t, bpx, xpb, lit, mv, red, acc, outer = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(sin_t[0], np.sin([1.0, 2.0]))
    np.testing.assert_allclose(sin_t[1], np.sin([3.0, 4.0]))
    np.testing.assert_allclose(bpx[0], [11.0, 22.0])
    np.testing.assert_allclose(xpb[1], [13.0, 24.0])
    # List literal `[100.0, 200.0]` broadcasts every tick (NOT period-2!).
    np.testing.assert_allclose(lit[0], [101.0, 202.0])
    np.testing.assert_allclose(lit[1], [103.0, 204.0])
    np.testing.assert_allclose(mv[0], [5.0, 11.0])
    np.testing.assert_allclose(red, [3.0, 7.0])
    np.testing.assert_allclose(acc[0], [1.0, 3.0])
    np.testing.assert_allclose(outer[0], np.outer([1.0, 2.0], [2.0, 5.0]))

  def test_numpy_ufunc_multi_output_unpacking(self):
    def prog():
      a = Stream([np.array([7, 10]), np.array([9, 11])])
      b = Stream([np.array([3, 4]), np.array([2, 5])])
      q, r = np.divmod(a, b)
      frac, whole = np.modf(Stream([np.array([1.25, 2.5])]))
      mant, exp = np.frexp(Stream([np.array([2.0, 8.0])]))
      return q, r, frac, whole, mant, exp

    qs, rs, fracs, wholes, mants, exps = Scheduler.run(prog, horizon=2)
    np.testing.assert_array_equal(qs[0], [2, 2])
    np.testing.assert_array_equal(rs[0], [1, 2])
    np.testing.assert_array_equal(qs[1], [4, 2])
    np.testing.assert_array_equal(rs[1], [1, 1])
    np.testing.assert_allclose(fracs[0], [0.25, 0.5])
    np.testing.assert_allclose(wholes[0], [1.0, 2.0])
    np.testing.assert_allclose(mants[0], [0.5, 0.5])
    np.testing.assert_array_equal(exps[0], [2, 4])

  def test_numpy_array_indexing_and_where_kwarg(self):
    def prog():
      mat = Stream([np.array([[1, 2, 3], [4, 5, 6]])])
      vec = Stream([np.array([10, 20, 30, 40])])
      x = Stream([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
      cond = Stream([np.array([True, False]), np.array([False, True])])
      return (
          mat[:, 0],
          vec[[0, 2]],
          np.select([x < 2.0, x > 3.0], [x * 10.0, x * 100.0], default=0.0),
          np.add(x, 10.0, where=cond),
      )

    col, fancy, sel, masked = Scheduler.run(prog, horizon=2)
    np.testing.assert_array_equal(col[0], [1, 4])
    np.testing.assert_array_equal(fancy[0], [10, 30])
    np.testing.assert_allclose(sel[0], [10.0, 0.0])
    np.testing.assert_allclose(sel[1], [0.0, 400.0])
    self.assertAlmostEqual(float(masked[0][0]), 11.0)
    self.assertAlmostEqual(float(masked[1][1]), 14.0)

  def test_numpy_array_function_nep18_lifting(self):
    def prog():
      a = Stream([np.array([[1.0, 2.0], [3.0, 4.0]])])
      b = Stream([np.array([[10.0, 20.0], [30.0, 40.0]])])
      mask = a > 2.0
      return (
          np.where(mask, b, a),
          np.concatenate([a, b], axis=0),
          np.stack([a, b], axis=0),
          np.sum(a, axis=-1, keepdims=True),
          np.clip(a, 1.5, 3.5),
          np.mean(a),
      )

    wh, cat, stk, sm, clp, mn = Scheduler.run(prog, horizon=1)
    np.testing.assert_allclose(wh[0], [[1.0, 2.0], [30.0, 40.0]])
    np.testing.assert_allclose(
        cat[0], [[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]]
    )
    np.testing.assert_allclose(
        stk[0], [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]
    )
    np.testing.assert_allclose(sm[0], [[3.0], [7.0]])
    np.testing.assert_allclose(clp[0], [[1.5, 2.0], [3.0, 3.5]])
    np.testing.assert_allclose(mn[0], 2.5)

  def test_numpy_cse_and_impure_rejection(self):
    def prog():
      x = Stream([np.array([[1.0, 2.0], [3.0, 4.0]])])
      s0 = np.sin(x)
      s1 = np.sin(x)
      r0 = np.sum(x, axis=0)
      r1 = np.sum(x, axis=0)
      r_other = np.sum(x, axis=1)
      e0 = np.empty_like(x)
      e1 = np.empty_like(x)
      return s0, s1, r0, r1, r_other, e0, e1

    g = build_graph(prog, optimize=True)
    s0, s1, r0, r1, r_other, e0, e1 = g.output
    self.assertIs(s0, s1)
    self.assertIs(r0, r1)
    self.assertIsNot(r0, r_other)
    self.assertIsNot(e0, e1)

  def test_numpy_inplace_and_asarray_rejected(self):
    buf = np.zeros(2)

    def check_rejections():
      s = Stream([np.array([1.0, 2.0])])
      with self.assertRaisesRegex(TypeError, 'NumPy array'):
        np.asarray(s)
      with self.assertRaisesRegex(TypeError, 'NumPy array'):
        np.array(s)
      with self.assertRaisesRegex(TypeError, 'out='):
        np.add(s, 1.0, out=buf)
      with self.assertRaisesRegex(TypeError, 'out='):
        tmp = np.zeros(2)
        tmp += s
      with self.assertRaisesRegex(TypeError, r'\.at'):
        np.add.at(s, 0, 1.0)
      with self.assertRaisesRegex(TypeError, 'out='):
        np.sum(s, out=np.zeros(()))
      with self.assertRaisesRegex(TypeError, 'copyto'):
        np.copyto(s, buf)
      with self.assertRaisesRegex(TypeError, 'put'):
        np.put(s, [0], [1.0])
      with self.assertRaisesRegex(TypeError, 'place'):
        np.place(s, [True, False], [1.0])
      with self.assertRaisesRegex(TypeError, 'putmask'):
        np.putmask(s, [True, False], [1.0])
      with self.assertRaisesRegex(TypeError, 'put_along_axis'):
        np.put_along_axis(s, np.array([0]), np.array([1.0]), axis=0)
      with self.assertRaisesRegex(TypeError, 'fill_diagonal'):
        np.fill_diagonal(s, 0.0)
      # out=None is allowed (stripped after validation).
      return np.add(s, 1.0, out=None)

    out = Scheduler.run(check_rejections, horizon=1)
    np.testing.assert_allclose(out[0], [2.0, 3.0])

  def test_numpy_ufunc_method_and_array_func_cse_without_cache(self):
    def prog():
      x = Stream([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
      return (
          np.sin(x),
          np.sin(x),
          np.add.reduce(x),
          np.add.reduce(x),
          np.sum(x),
          np.sum(x),
      )

    g = build_graph(prog, optimize=True)
    sin1, sin2, red1, red2, sum1, sum2 = g.output
    self.assertIs(sin1, sin2)
    self.assertIs(red1, red2)
    self.assertIs(sum1, sum2)
    out = Scheduler.run_graph(g, horizon=2)
    np.testing.assert_allclose(out[0][0], np.sin([1.0, 2.0]))
    np.testing.assert_allclose(out[2][0], 3.0)
    np.testing.assert_allclose(out[4][0], 3.0)

  def test_numpy_bool_stream_and_comparison_folds(self):
    def prog():
      x = Stream([np.array([-1.0, 2.0]), np.array([3.0, -4.0])])
      m = x > 0
      return m, m & False, m | True, np.array([1.0, 2.0]) == x

    g = build_graph(prog, optimize=True)
    m, mf, mt, eq = g.output
    self.assertFalse(g.is_bool_stream(m))
    self.assertFalse(g.is_bool_stream(eq))
    self.assertFalse(mf.static)
    self.assertFalse(mt.static)
    tm, tmf, tmt, teq = Scheduler.run_graph(g, horizon=2)
    np.testing.assert_array_equal(tm[0], [False, True])
    np.testing.assert_array_equal(tmf[0], [False, False])
    np.testing.assert_array_equal(tmt[0], [True, True])
    np.testing.assert_array_equal(teq[0], [False, True])

    # ifthenelse remains strictly scalar.
    with self.assertRaises(ExceptionGroup) as ctx:
      Scheduler.run(
          lambda: ifthenelse(
              Stream([np.array([True, False])]), Stream(1), Stream(0)
          ),
          horizon=1,
      )
    self.assertIsInstance(ctx.exception.exceptions[0], ValueError)

  def test_numpy_scalars_tags_signed_zero_nan_and_sum_dtype(self):
    nb_t = np.bool_(True)
    nb_f = np.bool_(False)
    self.assertTrue(is_numpy_bool(nb_t))
    self.assertTrue(is_numpy_scalar(nb_t))
    self.assertFalse(is_array_object(nb_t))
    self.assertEqual(tag_key(nb_t), (True, True))
    self.assertEqual(tag_key(nb_f), (True, False))
    self.assertTrue(matches_tag(nb_t, True))
    self.assertFalse(matches_tag(nb_t, 1))
    self.assertTrue(is_nan_scalar(np.float32('nan')))
    self.assertFalse(same_const_value(np.float32('nan'), np.float32('nan')))
    self.assertFalse(same_const_value(np.float32(-0.0), np.float32(0.0)))
    self.assertTrue(same_const_value(np.float32(0.0), np.float32(0.0)))
    self.assertFalse(
        same_const_value(
            np.complex64(complex(-0.0, 0.0)), np.complex64(complex(0.0, 0.0))
        )
    )
    self.assertFalse(
        same_const_value(
            np.complex64(complex(0.0, -0.0)), np.complex64(complex(0.0, 0.0))
        )
    )
    self.assertTrue(same_const_value(np.array(1), np.array(1)))
    self.assertFalse(
        same_const_value(np.array([1, 2]), np.array([1, 2]))
    )

    arr = np.zeros((2, 3), dtype=np.int64)
    self.assertTrue(is_array_object(arr))
    self.assertEqual(format_tag(arr), 'int64[2,3]')
    self.assertIsNone(when_selector_matches_const(None, np.array(True)))

    # Stream.sum preserves uint8 dtype without int(0) widening.
    u = Scheduler.run(
        lambda: Stream([np.array([1, 2], dtype=np.uint8)]).sum(
            Stream([np.array([3, 4], dtype=np.uint8)])
        ),
        horizon=1,
    )
    self.assertEqual(u[0].dtype, np.uint8)
    np.testing.assert_array_equal(u[0], [4, 6])

    # np.bool_ selector ticks route match/when correctly.
    def tag_prog():
      sel = Stream([np.bool_(True), np.bool_(False), np.bool_(True)])
      with match(sel) as m:
        with case(True):
          m.out = Stream(10)
        with default():
          m.out = Stream(20)
      return m.out

    self.assertEqual(Scheduler.run(tag_prog, horizon=3), [10, 20, 10])

  def test_numpy_subclocks_sync_and_kahn(self):
    def sync_prog():
      x = Stream([
          np.array([1.0, 2.0]),
          np.array([3.0, 4.0]),
          np.array([5.0, 6.0]),
      ])
      # Make `x_dyn` dynamic (base-clocked) so match() auto-samples it.
      x_dyn = x + 0.0
      c = Stream([True, False, True])
      with match(c) as m:
        with case(True):
          m.out = np.sin(x_dyn)
          m.cat = np.concatenate([x_dyn, x_dyn * 2.0], axis=0)
        with case(False):
          m.out = np.multiply(x_dyn, 10.0)
          m.cat = np.concatenate([x_dyn, x_dyn * 0.0], axis=0)
      return m.out, m.cat, when(c, np.sum(x))

    g = build_graph(sync_prog, semantics=Semantics.SYNC, optimize=True)
    clocks = check_clocks(g)
    m_out, m_cat, sub_sum = Scheduler.run_graph(g, horizon=3, clocks=clocks)
    np.testing.assert_allclose(m_out[0], np.sin([1.0, 2.0]))
    np.testing.assert_allclose(m_out[1], [30.0, 40.0])
    np.testing.assert_allclose(m_out[2], np.sin([5.0, 6.0]))
    np.testing.assert_allclose(m_cat[0], [1.0, 2.0, 2.0, 4.0])
    np.testing.assert_allclose(m_cat[1], [3.0, 4.0, 0.0, 0.0])
    np.testing.assert_allclose(m_cat[2], [5.0, 6.0, 10.0, 12.0])
    self.assertEqual(sub_sum[0], 3.0)
    self.assertIs(sub_sum[1], Absent)
    self.assertEqual(sub_sum[2], 11.0)

    def kahn_prog():
      x = Stream([np.array([1, 2]), np.array([3, 4]), np.array([5, 6])])
      c = Stream([True, False, True])
      return merge(c, np.square(when(c, x)), Stream([np.array([9, 9])]))

    k_out = Scheduler.run(kahn_prog, horizon=3, semantics=Semantics.KAHN)
    np.testing.assert_array_equal(k_out[0], [1, 4])
    np.testing.assert_array_equal(k_out[1], [9, 9])
    np.testing.assert_array_equal(k_out[2], [25, 36])

    # Automaton holding NumPy vectors across state switch.
    def auto_prog():
      flag = Stream([False, True, False])
      with automaton(init='A') as a:
        with state('A'):
          a.val = fby(
              np.array([1.0, 0.0]), lambda: a.val + np.array([1.0, 1.0])
          )
          until(flag, 'B')
        with state('B'):
          a.val = Stream.constant(np.array([0.0, 9.0]))
      return a.val

    ao = Scheduler.run(auto_prog, horizon=3, semantics=Semantics.SYNC)
    np.testing.assert_allclose(ao[0], [1.0, 0.0])
    np.testing.assert_allclose(ao[1], [2.0, 1.0])
    np.testing.assert_allclose(ao[2], [0.0, 9.0])

  def test_numpy_dispatch_in_equations_forward_refs(self):
    bias = np.array([1.0, 10.0])

    with psyre_equations() as prog:

      class block:  # pylint: disable=invalid-name
        acc = fby(np.array([0.0, 0.0]), next_acc)
        next_acc = np.tanh(acc + x)
        summed = np.add(s_stream, y)
        cat = np.concatenate([s_stream, y], axis=0)
        q, r = np.divmod(num, den)
        b_plus_x = bias + x
        sliced = big[lo:hi]
        s_stream = Stream([np.array([2.0, 3.0]), np.array([4.0, 5.0])])
        x = Stream([np.array([0.5, -0.5]), np.array([1.0, 0.0])])
        y = Stream([np.array([10.0, 20.0]), np.array([30.0, 40.0])])
        num = Stream([np.array([7, 9]), np.array([11, 13])])
        den = Stream([np.array([3, 4]), np.array([5, 6])])
        big = Stream([np.array([10, 20, 30, 40])])
        lo = Stream([1, 0])
        hi = Stream([3, 2])
        __output__ = (acc, summed, cat, q, r, b_plus_x, sliced)

      self.assertIsNotNone(block)

    acc_t, sum_t, cat_t, q_t, r_t, bpx_t, sl_t = Scheduler.run_graph(
        prog.graph, horizon=2
    )
    np.testing.assert_allclose(acc_t[0], [0.0, 0.0])
    np.testing.assert_allclose(acc_t[1], np.tanh([0.5, -0.5]))
    np.testing.assert_allclose(sum_t[0], [12.0, 23.0])
    np.testing.assert_allclose(cat_t[1], [4.0, 5.0, 30.0, 40.0])
    np.testing.assert_array_equal(q_t[0], [2, 2])
    np.testing.assert_array_equal(r_t[0], [1, 1])
    np.testing.assert_allclose(bpx_t[0], [1.5, 9.5])
    np.testing.assert_array_equal(sl_t[0], [20, 30])
    np.testing.assert_array_equal(sl_t[1], [10, 20])

    # Mutating / in-place ops rejected directly on unresolved Thunks.
    with self.assertRaisesRegex(TypeError, 'out='):
      with psyre_equations():

        class bad_out:  # pylint: disable=invalid-name
          _ = np.add(z, 1.0, out=np.zeros(2))
          z = Stream([np.array([1.0, 2.0])])
          __output__ = z

        del bad_out

    with self.assertRaisesRegex(TypeError, 'copyto'):
      with psyre_equations():

        class bad_copyto:  # pylint: disable=invalid-name
          _ = np.copyto(z, np.zeros(2))
          z = Stream([np.array([1.0, 2.0])])
          __output__ = z

        del bad_copyto

    with self.assertRaisesRegex(TypeError, 'NumPy array'):
      with psyre_equations():

        class bad_arr:  # pylint: disable=invalid-name
          _ = np.asarray(z)
          z = Stream([np.array([1.0, 2.0])])
          __output__ = z

        del bad_arr


@absltest.skipIf(jax is None, 'jax is not installed')
class JaxInteropTest(absltest.TestCase):
  """Tests forward-pass JAX array payloads and lifted `lax`/`jit`/`vmap`."""

  def test_jax_array_metadata_and_arithmetic(self):
    a = jnp.array([1.0, 2.0], dtype=jnp.float32)
    b = jnp.array([10.0, 20.0], dtype=jnp.float32)
    w = jnp.array([[1.0, 0.0], [2.0, 3.0]], dtype=jnp.float32)
    spec = jax.ShapeDtypeStruct((4, 8), jnp.bfloat16)

    self.assertTrue(is_array_object(a))
    self.assertTrue(is_array_object(spec))
    self.assertEqual(format_tag(a), 'float32[2]')
    self.assertEqual(format_tag(spec), 'bfloat16[4,8]')
    self.assertIsNone(when_selector_matches_const(None, jnp.array(True)))

    def prog():
      s = Stream([a, b])
      return a + s, s + b, s @ w, (s > 5.0) & False

    g = build_graph(prog, optimize=True)
    self.assertFalse(g.is_bool_stream(g.output[3]))
    self.assertFalse(g.output[3].static)
    t_ap, t_sb, t_mm, t_mask = Scheduler.run_graph(g, horizon=2)
    np.testing.assert_allclose(t_ap[0], [2.0, 4.0])
    np.testing.assert_allclose(t_sb[0], [11.0, 22.0])
    np.testing.assert_allclose(t_mm[0], [5.0, 6.0])
    np.testing.assert_array_equal(t_mask[0], [False, False])

    # Stream.sum keeps bfloat16 dtype (no Python int(0) widening).
    bf = Scheduler.run(
        lambda: Stream([jnp.array([1.0, 2.0], dtype=jnp.bfloat16)]).sum(
            Stream([jnp.array([3.0, 4.0], dtype=jnp.bfloat16)])
        ),
        horizon=1,
    )
    self.assertEqual(bf[0].dtype, jnp.bfloat16)

  def test_jax_lifted_lax_jit_vmap_and_absent_short_circuit(self):
    jit_calls = 0

    @jax.jit
    def kernel(x):
      return x * 2.0 + 1.0

    @psyre_comb(pure=True)
    def counted_kernel(x):
      nonlocal jit_calls
      jit_calls += 1
      return kernel(x)

    lifted_vmap = psyre_comb(pure=True)(
        jax.vmap(lambda row: jnp.sum(row * row))
    )
    lifted_topk = psyre_comb(pure=True, nout=2, static_argnums=1)(lax.top_k)
    lifted_dot = psyre_comb(pure=True, static_argnums=2)(lax.dot_general)
    lifted_select = psyre_comb(pure=True)(lax.select)

    def prog():
      c = Stream([True, False, True])
      x = Stream([
          jnp.array([1.0, 4.0, 2.0]),
          jnp.array([9.0, 8.0, 7.0]),
          jnp.array([3.0, 0.0, 5.0]),
      ])
      mat = Stream([
          jnp.array([[1.0, 2.0], [3.0, 4.0]]),
          jnp.array([[2.0, 0.0], [0.0, 2.0]]),
          jnp.array([[1.0, 1.0], [1.0, 1.0]]),
      ])
      sub = when(c, x)
      k_out = counted_kernel(sub)
      vals, idxs = lifted_topk(x, 2)
      v_out = lifted_vmap(mat)
      dot_out = lifted_dot(
          x,
          x,
          (((0,), (0,)), ((), ())),
      )
      sel_out = lifted_select(
          x > 2.5, x, jnp.zeros(3, dtype=jnp.float32)
      )
      return k_out, vals, idxs, v_out, dot_out, sel_out

    k_t, vals_t, idxs_t, v_t, d_t, sel_t = Scheduler.run(
        prog, horizon=3, semantics=Semantics.SYNC
    )
    # Inactive tick 1 short-circuits on Absent before calling `counted_kernel`.
    self.assertEqual(jit_calls, 2)
    self.assertIs(k_t[1], Absent)
    np.testing.assert_allclose(k_t[0], [3.0, 9.0, 5.0])
    np.testing.assert_allclose(k_t[2], [7.0, 1.0, 11.0])
    np.testing.assert_allclose(vals_t[0], [4.0, 2.0])
    np.testing.assert_array_equal(idxs_t[0], [1, 2])
    np.testing.assert_allclose(v_t[0], [5.0, 25.0])
    np.testing.assert_allclose(d_t[0], 21.0)
    np.testing.assert_allclose(sel_t[0], [0.0, 4.0, 0.0])

  def test_jax_equations_forward_reference(self):
    a = jnp.array([1.0, 2.0])
    with psyre_equations() as prog:

      class eq_jax:  # pylint: disable=invalid-name
        out = a + x
        x = Stream([jnp.array([3.0, 4.0])])
        __output__ = out

      self.assertIsNotNone(eq_jax)

    res = Scheduler.run_graph(prog.graph, horizon=1)
    np.testing.assert_allclose(res[0], [4.0, 6.0])


if __name__ == '__main__':
  absltest.main()
