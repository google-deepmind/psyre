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
"""Tests for `psyre.jax` lifting, `jax.numpy`/`nn`/`random`, and interop."""

import dataclasses
import importlib

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
  from psyre import jax as pjax
  from psyre.jax import lax as plax
else:  # pragma: no cover
  pjax = None
  plax = None

from psyre.automaton import automaton, state, until
from psyre.builder import build_graph
from psyre.clock import check_clocks
from psyre.common import Absent, Semantics, fby, ifthenelse, when
from psyre.equations import psyre_equations
from psyre.match import case, match
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import


@absltest.skipIf(jax is None, 'jax is not installed')
class PsyreJaxTest(absltest.TestCase):
  """Covers `psyre.jax` namespaces, PyTrees, conversions, and lifting API."""

  def test_lax_co_exposed_and_namespaces(self):
    self.assertIs(pjax.lax, plax)
    self.assertIs(importlib.import_module('psyre.jax.lax'), plax)
    self.assertIs(pjax.jnp, pjax.numpy)
    self.assertIn('sin', dir(pjax.jnp))
    self.assertIn('relu', dir(pjax.nn))
    self.assertIn('LiftedNamespace', repr(pjax.jnp))
    self.assertIsInstance(pjax.jnp, pjax.LiftedNamespace)
    self.assertIs(plax.get_primitive(plax.add), jax_lax.add_p)
    if pjax.scipy is not None:
      res = Scheduler.run(
          lambda: pjax.scipy.special.logsumexp(
              Stream([jnp.array([0.0, 0.0])])
          ),
          horizon=1,
      )
      self.assertAlmostEqual(float(res[0]), float(jnp.log(2.0)), places=5)

  def test_jnp_nn_fft_random_and_multi_output_unpacking(self):
    mat = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)

    def prog():
      x = Stream([jnp.array([-1.0, 2.0]), jnp.array([3.0, -4.0])])
      a = Stream([jnp.array([7.0, 9.0])])
      b = Stream([jnp.array([3.0, 4.0])])
      q, r = pjax.jnp.divmod(a, b)
      frac, whole = pjax.jnp.modf(Stream([jnp.array([1.25, -2.5])]))
      m = Stream([mat])
      sign, logdet = pjax.jnp.linalg.slogdet(m)
      qm, rm = pjax.jnp.linalg.qr(m)
      qh, qtau = pjax.jnp.linalg.qr(m, mode='raw')
      r_only = pjax.jnp.linalg.qr(m, mode='r')
      u, s, vt = pjax.jnp.linalg.svd(m)
      s_only = pjax.jnp.linalg.svd(m, compute_uv=False)
      key = Stream([jax.random.key(0), jax.random.key(1)])
      noise = pjax.random.normal(key, shape=(2,))
      return {
          'relu': pjax.nn.relu(x),
          'soft': pjax.nn.softmax(x),
          'sin': pjax.jnp.sin(x),
          'erf': pjax.scipy.special.erf(x),
          'rfft': pjax.jnp.fft.rfft(x),
          'q': q,
          'r': r,
          'frac': frac,
          'whole': whole,
          'sign': sign,
          'logdet': logdet,
          'qm': qm,
          'rm': rm,
          'qh': qh,
          'qtau': qtau,
          'r_only': r_only,
          'u': u,
          's': s,
          'vt': vt,
          's_only': s_only,
          'noise': noise,
      }

    traces = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(traces['relu'][0], [0.0, 2.0])
    np.testing.assert_allclose(traces['relu'][1], [3.0, 0.0])
    self.assertAlmostEqual(float(jnp.sum(traces['soft'][0])), 1.0, places=5)
    np.testing.assert_allclose(
        traces['sin'][0], np.sin([-1.0, 2.0]), rtol=1e-5
    )
    np.testing.assert_allclose(
        traces['erf'][0], jax.scipy.special.erf(jnp.array([-1.0, 2.0])),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        traces['rfft'][0], np.fft.rfft([-1.0, 2.0]), atol=1e-5
    )
    np.testing.assert_allclose(traces['q'][0], [2.0, 2.0])
    np.testing.assert_allclose(traces['r'][0], [1.0, 1.0])
    np.testing.assert_allclose(traces['frac'][0], [0.25, -0.5])
    np.testing.assert_allclose(traces['whole'][0], [1.0, -2.0])
    self.assertAlmostEqual(float(traces['sign'][0]), 1.0)
    self.assertAlmostEqual(float(traces['logdet'][0]), np.log(11.0), places=5)
    np.testing.assert_allclose(
        traces['qm'][0] @ traces['rm'][0], mat, atol=1e-5
    )
    np.testing.assert_allclose(traces['r_only'][0], traces['rm'][0], atol=1e-5)
    self.assertEqual(traces['qh'][0].shape, mat.shape)
    self.assertEqual(traces['qtau'][0].shape, (2,))
    np.testing.assert_allclose(
        traces['u'][0] @ jnp.diag(traces['s'][0]) @ traces['vt'][0],
        mat,
        atol=1e-5,
    )
    np.testing.assert_allclose(traces['s_only'][0], traces['s'][0], atol=1e-5)
    self.assertEqual(traces['noise'][0].shape, (2,))
    self.assertFalse(
        bool(jnp.array_equal(traces['noise'][0], traces['noise'][1]))
    )

    # Zero-stream eager call returns `jax.Array`.
    z = pjax.jnp.zeros((2, 3))
    self.assertIsInstance(z, jax.Array)
    self.assertEqual(z.shape, (2, 3))

  def test_backend_flags_and_cse(self):
    def prog():
      x = Stream([jnp.array([1.0, 2.0])])
      a = pjax.jnp.sin(x)
      b = pjax.jnp.sin(x)
      return {'a': a, 'b': b}

    g = build_graph(prog, optimize=True)
    self.assertLen(g.nodes, 1)
    self.assertIs(g.output['a'], g.output['b'])
    self.assertTrue(pjax.is_jax_step_func(g.nodes[0]))
    self.assertTrue(pjax.is_jax_step_func(pjax.jnp.sin))

  def test_to_jax_and_to_numpy_boundaries(self):
    # Eager conversions outside Stage 1.
    self.assertIsInstance(pjax.to_jax([1.0, 2.0], dtype=jnp.float32), jax.Array)
    self.assertIsInstance(pjax.to_numpy(jnp.array([1.0, 2.0])), np.ndarray)

    def prog():
      host = Stream([np.array([1.0, 2.0], dtype=np.float32)])
      dev = pjax.to_jax(host, dtype=jnp.float32)
      y = pjax.jnp.square(dev)
      back = pjax.to_numpy(y)
      return {'dev': dev, 'y': y, 'back': back}

    g = build_graph(prog, optimize=True)
    to_np_nodes = [n for n in g.nodes if n.op == 'to_numpy']
    self.assertLen(to_np_nodes, 1)
    # `to_numpy` leaves device islands cleanly (NOT tagged as JAX backend).
    self.assertFalse(pjax.is_jax_step_func(to_np_nodes[0]))

    traces = Scheduler.run_graph(g, horizon=1)
    self.assertIsInstance(traces['dev'][0], jax.Array)
    self.assertIsInstance(traces['back'][0], np.ndarray)
    np.testing.assert_allclose(traces['back'][0], [1.0, 4.0])

  def test_register_jax_pytree_bridges_custom_dataclass(self):
    @pjax.register_jax_pytree
    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class State:
      pos: object
      vel: object
      gain: float = dataclasses.field(metadata={'static': True})

    step_fn = pjax.lift(lambda st: st.pos + st.gain * st.vel, pure=True)

    def prog():
      p = Stream([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])])
      v = Stream([jnp.array([0.5, 1.0]), jnp.array([2.0, -1.0])])
      return {'out': step_fn(State(pos=p, vel=v, gain=2.0))}

    traces = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(traces['out'][0], [2.0, 4.0])
    np.testing.assert_allclose(traces['out'][1], [7.0, 2.0])

    with self.assertRaises(TypeError):
      pjax.register_jax_pytree(42)  # type: ignore[arg-type]

    class _Unreg:
      pass

    pjax.register_jax_pytree(_Unreg)
    try:
      with self.assertRaisesRegex(TypeError, 'not registered as a JAX PyTree'):
        build_graph(lambda: pjax.lift(lambda x: x)(_Unreg()))
    finally:
      from psyre.payload import unregister_pytree_node  # pylint: disable=g-import-not-at-top,g-importing-member
      unregister_pytree_node(_Unreg)

  def test_coercion_out_and_impure_rejections(self):
    with self.assertRaises(TypeError):
      build_graph(lambda: pjax.jnp.array(Stream([1.0])))
    with self.assertRaises(TypeError):
      build_graph(lambda: pjax.jnp.asarray(Stream([1.0])))
    with self.assertRaises(TypeError):
      build_graph(lambda: pjax.jnp.sin(Stream([1.0]), out=jnp.zeros(())))
    with self.assertRaises(TypeError):
      pjax.lift(123)  # type: ignore[arg-type]
    with self.assertRaises(ValueError):
      pjax.lift(jnp.sin, nout=0)
    with self.assertRaises(TypeError):
      pjax.lift(jnp.sin, primitive='not_a_prim')
    if hasattr(jax.lax, 'infeed'):
      with self.assertRaises(TypeError):
        plax.infeed(Stream([1.0]))

    with self.assertRaises(TypeError):
      pjax.lift(jax.debug.print)('x={}', Stream([1.0]))

    # User helper named `io_callback` / taking `out=` is NOT rejected.
    def io_callback(x, out=1.0):
      return x + out

    self.assertEqual(pjax.lift(io_callback)(2.0, out=3.0), 5.0)

    # Non-stream `pjax.jnp.array` works eagerly.
    self.assertIsInstance(pjax.jnp.array([1.0, 2.0]), jax.Array)

  def test_match_automaton_equations_and_scalar_ifthenelse(self):
    def prog():
      flag = Stream([True, False, True, False])
      x = Stream([jnp.array([float(i), float(i + 1)]) for i in range(4)])
      x_dyn = x + 0.0
      with match(flag) as m:
        with case(True):
          m.out = pjax.jnp.sin(x_dyn)
        with case(False):
          m.out = fby(pjax.jnp.zeros(2), x_dyn)
      with automaton(init='s0') as a:
        with state('s0'):
          a.v = pjax.nn.relu(x_dyn - 1.0)
          until(Stream([False, True, False, False]), 's1')
        with state('s1'):
          a.v = Stream.constant(jnp.array([9.0, 9.0]))
      return {'m': m.out, 'a': a.v, 'sub': pjax.jnp.cos(when(flag, x_dyn))}

    g = build_graph(prog, semantics=Semantics.SYNC, optimize=True)
    clocks = check_clocks(g)
    ts = Scheduler.run_graph(g, horizon=4, clocks=clocks)
    np.testing.assert_allclose(ts['m'][0], np.sin([0.0, 1.0]), rtol=1e-5)
    np.testing.assert_allclose(ts['m'][1], [0.0, 0.0])
    np.testing.assert_allclose(ts['m'][2], np.sin([2.0, 3.0]), rtol=1e-5)
    np.testing.assert_allclose(ts['m'][3], [1.0, 2.0])
    np.testing.assert_allclose(ts['a'][0], [0.0, 0.0])
    np.testing.assert_allclose(ts['a'][1], [0.0, 1.0])
    np.testing.assert_allclose(ts['a'][2], [9.0, 9.0])
    np.testing.assert_allclose(ts['a'][3], [9.0, 9.0])
    np.testing.assert_allclose(ts['sub'][0], np.cos([0.0, 1.0]), rtol=1e-5)
    self.assertIs(ts['sub'][1], Absent)
    np.testing.assert_allclose(ts['sub'][2], np.cos([2.0, 3.0]), rtol=1e-5)

    # Same-program KAHN equivalence for merged output.
    def kahn_prog():
      flag = Stream([True, False, True])
      x = Stream([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]),
                  jnp.array([5.0, 6.0])])
      x_dyn = x + 0.0
      with match(flag) as m:
        with case(True):
          m.out = pjax.jnp.square(x_dyn)
        with case(False):
          m.out = pjax.jnp.negative(x_dyn)
      return m.out

    k_out = Scheduler.run(kahn_prog, horizon=3, semantics=Semantics.KAHN)
    np.testing.assert_allclose(k_out[0], [1.0, 4.0])
    np.testing.assert_allclose(k_out[1], [-3.0, -4.0])
    np.testing.assert_allclose(k_out[2], [25.0, 36.0])

    # Forward references via `psyre.equations`.
    # pylint: disable=used-before-assignment,invalid-name
    with psyre_equations() as eq:

      class block:
        y = pjax.jnp.tanh(acc + x)
        acc = fby(jnp.zeros(2), y)
        x = Stream([jnp.array([1.0, -1.0]), jnp.array([0.0, 2.0])])
        __output__ = (acc, y)

      self.assertIsNotNone(block)
    # pylint: enable=used-before-assignment,invalid-name

    acc_t, y_t = Scheduler.run_graph(eq.graph, horizon=2)
    np.testing.assert_allclose(acc_t[0], [0.0, 0.0])
    np.testing.assert_allclose(acc_t[1], y_t[0], atol=1e-6)

    # `ifthenelse` remains strictly scalar outside `vmap`.
    with self.assertRaises(ExceptionGroup) as ctx:
      Scheduler.run(
          lambda: ifthenelse(
              Stream([jnp.array([True, False])]), Stream(1.0), Stream(0.0)
          ),
          horizon=1,
      )
    self.assertIsInstance(ctx.exception.exceptions[0], ValueError)


if __name__ == '__main__':
  absltest.main()
