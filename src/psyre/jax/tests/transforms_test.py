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
"""Tests for lifting-time JAX transformations (`psyre.jax`)."""

import dataclasses

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
from psyre.common import (
    Absent,
    Semantics,
    buffer,
    current,
    fby,
    ifthenelse,
    merge,
    morph,
    pre,
    prefix_sum,
    psyre_comb,
    psyre_sequ,
    reset,
    scan,
    switch,
    when,
)
from psyre.equations import psyre_equations
from psyre.match import case, match
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import


@absltest.skipIf(jax is None, 'jax is not installed')
class JaxTransformsTest(absltest.TestCase):
  """Covers `vmap`, `stack_streams`, `unstack_stream`, `batch_streams`, etc."""

  def test_stack_and_unstack_streams(self):
    def prog():
      a = Stream([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])])
      b = Stream([jnp.array([5.0, 6.0]), jnp.array([7.0, 8.0])])
      st0 = pjax.stack_streams(a, b, axis=0)
      st1 = pjax.stack_streams(a, b, axis=1)
      u0, u1 = pjax.unstack_stream(st0, size=2, axis=0)
      v0, v1 = pjax.unstack_stream(st1, size=2, axis=1)
      return {'st0': st0, 'st1': st1, 'u0': u0, 'u1': u1, 'v0': v0, 'v1': v1}

    ts = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(ts['st0'][0], [[1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_allclose(ts['st1'][0], [[1.0, 5.0], [2.0, 6.0]])
    np.testing.assert_allclose(ts['u0'][1], [3.0, 4.0])
    np.testing.assert_allclose(ts['u1'][1], [7.0, 8.0])
    np.testing.assert_allclose(ts['v0'][0], [1.0, 2.0])
    np.testing.assert_allclose(ts['v1'][0], [5.0, 6.0])

    with self.assertRaises(ValueError):
      pjax.stack_streams()
    with self.assertRaises(TypeError):
      pjax.stack_streams(Stream([1.0]), axis=True)
    with self.assertRaises(ValueError):
      pjax.unstack_stream(Stream([jnp.ones(2)]), size=0)

  def test_vmap_single_lifted_node_commutes_with_comb(self):
    @psyre_comb(pure=True)
    def affine(x, w, b):
      return jnp.dot(x, w) + b

    def prog():
      xs = Stream([
          jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
          jnp.array([[0.0, 1.0], [2.0, 0.0], [-1.0, 1.0]]),
      ])
      w = Stream([jnp.array([[1.0, 0.0], [0.0, 2.0]])])
      b = Stream([jnp.array([10.0, 20.0])])
      y_aff = pjax.vmap(affine, in_axes=(0, None, None))(xs, w, b)
      y_sin = pjax.vmap(pjax.jnp.sin)(xs)
      y_sum = pjax.vmap(pjax.jnp.sum)(xs, axis=-1)
      y_t = pjax.vmap(plax.mul, in_axes=(1, None), out_axes=1)(
          Stream([jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]),
          jnp.array(10.0),
      )
      vals, idx = pjax.vmap(plax.top_k, in_axes=(0, None))(xs, 1)
      # Axis name collective inside single-node `vmap`.
      y_norm = pjax.vmap(
          pjax.lift(lambda v: v / jax.lax.psum(v, 'batch')),
          axis_name='batch',
      )(Stream([jnp.array([1.0, 3.0])]))
      return {
          'aff': y_aff,
          'sin': y_sin,
          'sum': y_sum,
          't': y_t,
          'vals': vals,
          'idx': idx,
          'norm': y_norm,
      }

    ts = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(
        ts['aff'][0], [[11.0, 24.0], [13.0, 28.0], [15.0, 32.0]]
    )
    np.testing.assert_allclose(
        ts['sin'][0], np.sin([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), rtol=1e-5
    )
    np.testing.assert_allclose(ts['sum'][0], [3.0, 7.0, 11.0])
    np.testing.assert_allclose(
        ts['t'][0], [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    )
    np.testing.assert_allclose(ts['vals'][0], [[2.0], [4.0], [6.0]])
    np.testing.assert_array_equal(ts['idx'][0], [[1], [1], [1]])
    np.testing.assert_allclose(ts['norm'][0], [0.25, 0.75])

    # Outside Stage 1, eager call delegates directly to `jax.vmap`.
    eager = pjax.vmap(pjax.jnp.square)(jnp.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(eager, [4.0, 9.0, 16.0])

  def test_vmap_composite_feedback_fby_and_semantic_equivalence(self):
    def leaky_integrator(x, decay):
      acc = fby(jnp.zeros(2), lambda: decay * acc + x)
      return pjax.nn.relu(acc), acc

    x_ticks = [
        jnp.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.0]]),
        jnp.array([[2.0, 4.0], [-1.0, 1.0], [3.0, -2.0]]),
        jnp.array([[-3.0, 1.0], [2.0, -5.0], [0.0, 1.0]]),
    ]
    decay = jnp.array(0.5)

    def batched_prog():
      xs = Stream(x_ticks)
      d = Stream.constant(decay)
      return pjax.vmap(leaky_integrator, in_axes=(0, None))(xs, d)

    g = build_graph(batched_prog, optimize=True)
    clocks = check_clocks(g)
    b_relu, b_acc = Scheduler.run_graph(g, horizon=3, clocks=clocks)

    # Compare lane-by-lane against scalar execution `leaky_integrator(x_i, d)`.
    for lane in range(3):
      def single_prog(i=lane):
        xi = Stream([t[i] for t in x_ticks])
        return leaky_integrator(xi, Stream.constant(decay))

      s_relu, s_acc = Scheduler.run(single_prog, horizon=3)
      for tick in range(3):
        np.testing.assert_allclose(b_relu[tick][lane], s_relu[tick], atol=1e-5)
        np.testing.assert_allclose(b_acc[tick][lane], s_acc[tick], atol=1e-5)

  def test_vmap_composite_ifthenelse_and_switch(self):
    # (a) Batched `ifthenelse(c, t, f)` lowers to `jax.vmap(lax.select)`.
    def clamp_step(x):
      return ifthenelse(x > 0.0, x * 2.0, jnp.array(-1.0))

    xs_data = [
        jnp.array([-2.0, 0.0, 3.0, 5.0]),
        jnp.array([1.0, -4.0, 2.0, -1.0]),
    ]
    y = Scheduler.run(
        lambda: pjax.vmap(clamp_step)(Stream(xs_data)), horizon=2
    )
    np.testing.assert_allclose(y[0], [-1.0, -1.0, 6.0, 10.0])
    np.testing.assert_allclose(y[1], [2.0, -1.0, 4.0, -1.0])

    # (b) Unbatched scalar `ifthenelse(c, t, f)` picks branch and broadcasts.
    def scalar_mux(flag, x):
      return ifthenelse(flag, x + 10.0, Stream.constant(jnp.array(0.0)))

    flags = [True, False, True]
    ym = Scheduler.run(
        lambda: pjax.vmap(scalar_mux, in_axes=(None, 0))(
            Stream(flags), Stream([jnp.array([1.0, 2.0, 3.0])])
        ),
        horizon=3,
    )
    np.testing.assert_allclose(ym[0], [11.0, 12.0, 13.0])
    np.testing.assert_allclose(ym[1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ym[2], [11.0, 12.0, 13.0])

    # (c) Batched integer `switch` (including out-of-range tags -> `default`).
    def sw_fn(tag, x):
      return switch(
          tag, {0: x + 1.0, 1: x * 10.0, 2: Stream(0.0)}, default=Stream(-9.0)
      )

    tags = Stream([jnp.array([-1, 0, 1, 2, 7])])
    xv = Stream([jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])])
    ysw = Scheduler.run(
        lambda: pjax.vmap(sw_fn, in_axes=(0, 0))(tags, xv), horizon=1
    )
    np.testing.assert_allclose(ysw[0], [-9.0, 3.0, 30.0, 0.0, -9.0])

    # Non-integer tag dictionary under batched selector raises `BatchError`.
    def bad_sw(tag, x):
      return switch(tag, {'a': x, 'b': x * 2.0})

    with self.assertRaises(pjax.BatchError):
      build_graph(
          lambda: pjax.vmap(bad_sw)(
              Stream([jnp.array([0, 1])]), Stream([jnp.array([1.0, 2.0])])
          )
      )

  def test_vmap_composite_temporal_match_reset_and_out_axes(self):
    def cell(x, mode, rst):
      with reset(rst):
        s_fby = fby(jnp.array(0.0), lambda: s_fby + x)
      s_morph = morph(jnp.array(100.0), x)
      s_psum = prefix_sum(x, 0.0)
      s_scan = scan(x, op=lambda a, b: a + b * 2.0)
      s_pre = pre(x)
      x_dyn = x + 0.0
      with match(mode) as m:
        with case(True):
          m.res = x_dyn * 2.0
        with case(False):
          # Unbatched constant arm auto-broadcasts to batch size at subclock.
          m.res = Stream(0.0)
      c_hold = current(mode, when(mode, x_dyn), jnp.array(-5.0))
      return {
          'fby': s_fby,
          'morph': s_morph,
          'psum': s_psum,
          'scan': s_scan,
          'pre': s_pre,
          'match': m.res,
          'hold': c_hold,
          'tag': mode,
      }

    xs = [
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0]),
        jnp.array([5.0, 6.0]),
    ]
    modes = [True, False, True]
    rsts = [False, True, False]

    def prog():
      return pjax.vmap(
          cell,
          in_axes=(0, None, None),
          out_axes={
              'fby': 0,
              'morph': 0,
              'psum': 0,
              'scan': 0,
              'pre': 0,
              'match': 0,
              'hold': 0,
              'tag': None,
          },
      )(Stream(xs), Stream(modes), Stream(rsts))

    g = build_graph(prog, semantics=Semantics.SYNC, optimize=True)
    clocks = check_clocks(g)
    ts = Scheduler.run_graph(g, horizon=3, clocks=clocks)
    # tick 0: fby=0, tick 1: rst=True resets fby to 0, tick 2: 0 + [3, 4]
    np.testing.assert_allclose(ts['fby'][0], [0.0, 0.0])
    np.testing.assert_allclose(ts['fby'][1], [0.0, 0.0])
    np.testing.assert_allclose(ts['fby'][2], [3.0, 4.0])
    np.testing.assert_allclose(ts['morph'][0], [100.0, 100.0])
    np.testing.assert_allclose(ts['morph'][1], [3.0, 4.0])
    np.testing.assert_allclose(ts['psum'][2], [9.0, 12.0])
    np.testing.assert_allclose(ts['scan'][0], [1.0, 2.0])
    np.testing.assert_allclose(ts['scan'][1], [7.0, 10.0])
    self.assertIsNone(ts['pre'][0])
    np.testing.assert_allclose(ts['pre'][1], [1.0, 2.0])
    np.testing.assert_allclose(ts['match'][0], [2.0, 4.0])
    np.testing.assert_allclose(ts['match'][1], [0.0, 0.0])
    np.testing.assert_allclose(ts['match'][2], [10.0, 12.0])
    np.testing.assert_allclose(ts['hold'][0], [1.0, 2.0])
    np.testing.assert_allclose(ts['hold'][1], [1.0, 2.0])
    np.testing.assert_allclose(ts['hold'][2], [5.0, 6.0])
    self.assertEqual(ts['tag'], [True, False, True])

  def test_nested_vmap_and_equations_forward_refs(self):
    def inner(x):
      s = fby(jnp.array(0.0), lambda: s + x)
      return s + x * 2.0

    data = [
        jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
    ]
    out = Scheduler.run(
        lambda: pjax.vmap(pjax.vmap(inner))(Stream(data)), horizon=2
    )
    np.testing.assert_allclose(out[0], [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    np.testing.assert_allclose(
        out[1], [[21.0, 42.0, 63.0], [84.0, 105.0, 126.0]]
    )

    # Also test `out_axes` transposition on 2D outputs.
    moved = Scheduler.run(
        lambda: pjax.vmap(
            lambda x: pjax.jnp.sin(x) + 1.0, in_axes=0, out_axes=1
        )(Stream([jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])])),
        horizon=1,
    )
    np.testing.assert_allclose(
        moved[0],
        (np.sin([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]) + 1.0).T,
        rtol=1e-5,
    )

    # Composite `vmap` runs identically under `Semantics.KAHN`.
    k_vmap = Scheduler.run(
        lambda: pjax.vmap(lambda x: fby(jnp.zeros(2), x + 1.0))(
            Stream([
                jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                jnp.array([[5.0, 6.0], [7.0, 8.0]]),
            ])
        ),
        horizon=2,
        semantics=Semantics.KAHN,
    )
    np.testing.assert_allclose(k_vmap[0], [[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(k_vmap[1], [[2.0, 3.0], [4.0, 5.0]])

    # Nested single-op `vmap(vmap(op))` keeps kwargs and multi-output `nout`.
    ns_sum = pjax.vmap(pjax.vmap(pjax.jnp.sum))
    ns_topk = pjax.vmap(
        pjax.vmap(plax.top_k, in_axes=(0, None)), in_axes=(0, None)
    )
    s_ts, tv_ts, _ = Scheduler.run(
        lambda: (
            ns_sum(Stream([jnp.ones((2, 3, 4))]), axis=-1),
            *ns_topk(Stream([jnp.arange(12.0).reshape(2, 2, 3)]), 1),
        ),
        horizon=1,
    )
    np.testing.assert_allclose(s_ts[0], jnp.full((2, 3), 4.0))
    np.testing.assert_allclose(tv_ts[0], [[[2.0], [5.0]], [[8.0], [11.0]]])

    # Vector `(B, D)` `prefix_sum` inside `vmap` adds along batch axis `0`.
    ps_ts = Scheduler.run(
        lambda: pjax.vmap(lambda x: prefix_sum(x, 0.0))(
            Stream([jnp.ones((2, 3)), jnp.full((2, 3), 2.0)])
        ),
        horizon=2,
    )
    np.testing.assert_allclose(ps_ts[1], jnp.full((2, 3), 3.0))

    # Forward ref `equations.Thunk` passed into composite `vmap`.
    # pylint: disable=used-before-assignment,invalid-name
    with psyre_equations() as eq:

      class block:
        out_s = pjax.vmap(inner)(x)
        x = Stream([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])])
        __output__ = out_s

      self.assertIsNotNone(block)
    # pylint: enable=used-before-assignment,invalid-name

    ts = Scheduler.run_graph(eq.graph, horizon=2)
    np.testing.assert_allclose(ts[0], [2.0, 4.0])
    np.testing.assert_allclose(ts[1], [7.0, 10.0])

  def test_vmap_rejects_batched_control_clocks_and_bad_axes(self):
    xs = lambda: Stream([jnp.array([True, False])])
    vs = lambda: Stream([jnp.array([1.0, 2.0])])

    # Batched `when` selector.
    with self.assertRaises(pjax.BatchError):
      build_graph(lambda: pjax.vmap(when)(xs(), vs()))

    # Batched `merge` selector.
    with self.assertRaises(pjax.BatchError):
      build_graph(
          lambda: pjax.vmap(lambda c, v: merge(c, v, v))(xs(), vs())
      )

    # Batched `reset` signal.
    def bad_reset(r, v):
      with reset(r):
        return fby(jnp.array(0.0), v)

    with self.assertRaises(pjax.BatchError):
      build_graph(lambda: pjax.vmap(bad_reset)(xs(), vs()))

    # Batched `buffer` push predicate.
    with self.assertRaises(pjax.BatchError):
      build_graph(
          lambda: pjax.vmap(lambda c, v: buffer(v, c, Stream(True)))(xs(), vs())
      )

    # Batched `out_axes=None` when output actually depends on batched input.
    with self.assertRaises(pjax.BatchError):
      build_graph(
          lambda: pjax.vmap(lambda v: v + 1.0, out_axes=None)(vs())
      )

    # Arbitrary stateful `@psyre_sequ` node receiving a batched stream.
    @psyre_sequ
    def custom_sequ(x):
      s = x
      while True:
        x = yield s
        s = s + x

    with self.assertRaises(pjax.BatchError):
      build_graph(lambda: pjax.vmap(custom_sequ)(vs()))

  def test_batch_streams_heterogeneous_per_lane_automata(self):
    # Each lane triggers its OWN state machine transition independently!
    def lane_fsm(x):
      with automaton(init='low') as a:
        with state('low'):
          a.val = fby(jnp.array(0.0), lambda: a.val + x)
          until(x > 5.0, 'high')
        with state('high'):
          a.val = Stream.constant(jnp.array(-1.0))
      return a.val

    def prog():
      # Lane 0 crosses > 5 on tick 0; lane 1 crosses on tick 1; lane 2 never.
      xs = Stream([
          jnp.array([6.0, 1.0, 2.0]),
          jnp.array([1.0, 7.0, 2.0]),
          jnp.array([1.0, 1.0, 2.0]),
      ])
      return pjax.batch_streams(lane_fsm, xs, size=3, name='fsm_lanes')

    ts = Scheduler.run(prog, horizon=3, semantics=Semantics.SYNC)
    np.testing.assert_allclose(ts[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ts[1], [-1.0, 1.0, 2.0])
    np.testing.assert_allclose(ts[2], [-1.0, -1.0, 4.0])

  def test_rewrite_dtype_bfloat16_non_destructive(self):
    def prog():
      a = Stream([jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)])
      b = Stream([jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)])
      dot = plax.dot_general(
          a, b, dimension_numbers=(((1,), (0,)), ((), ()))
      )
      return {'dot': dot, 'sin': pjax.jnp.sin(a)}

    orig = build_graph(prog, optimize=True)
    bf16_g = pjax.rewrite_dtype(
        orig,
        target_dtype=jnp.bfloat16,
        primitives=(jax_lax.dot_general_p,),
    )
    accum_f32_g = pjax.rewrite_dtype(
        orig,
        target_dtype=jnp.bfloat16,
        ops=('dot_general',),
        preferred_element_type=jnp.float32,
    )

    orig_out = Scheduler.run_graph(orig, horizon=1)
    bf16_out = Scheduler.run_graph(bf16_g, horizon=1)
    f32_out = Scheduler.run_graph(accum_f32_g, horizon=1)

    self.assertEqual(orig_out['dot'][0].dtype, jnp.float32)
    self.assertEqual(bf16_out['dot'][0].dtype, jnp.bfloat16)
    self.assertEqual(bf16_out['sin'][0].dtype, jnp.float32)
    self.assertEqual(f32_out['dot'][0].dtype, jnp.float32)
    np.testing.assert_allclose(
        np.asarray(bf16_out['dot'][0], dtype=np.float32),
        np.asarray(orig_out['dot'][0], dtype=np.float32),
        rtol=2e-2,
    )

  def test_rewrite_dtype_ops_respects_include_stream_ops(self):
    def prog():
      a = Stream([jnp.array([1.0, 2.0], dtype=jnp.float32)])
      b = Stream([jnp.array([3.0, 4.0], dtype=jnp.float32)])
      return a + b

    orig = build_graph(prog, optimize=False)
    skip_g = pjax.rewrite_dtype(
        orig,
        target_dtype=jnp.bfloat16,
        ops=('__add__',),
        include_stream_ops=False,
    )
    cast_g = pjax.rewrite_dtype(
        orig,
        target_dtype=jnp.bfloat16,
        ops=('__add__',),
        include_stream_ops=True,
    )
    skip_out = Scheduler.run_graph(skip_g, horizon=1)[0]
    cast_out = Scheduler.run_graph(cast_g, horizon=1)[0]
    self.assertEqual(skip_out.dtype, jnp.float32)
    self.assertEqual(cast_out.dtype, jnp.bfloat16)

  def test_vmap_custom_registered_pytree_dataclass(self):
    @pjax.register_jax_pytree
    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class BodyState:
      pos: object
      vel: object
      gain: float = dataclasses.field(metadata={'static': True})

    step_single = pjax.lift(lambda st: st.pos + st.gain * st.vel, pure=True)

    def step_composite(st):
      acc = fby(jnp.array(0.0), lambda: acc + st.pos)
      return acc + st.gain * st.vel

    def prog():
      p = Stream([jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])])
      v_batched = Stream([
          jnp.array([10.0, 20.0, 30.0]),
          jnp.array([1.0, 2.0, 3.0]),
      ])
      v_shared = Stream([jnp.array(10.0), jnp.array(1.0)])
      s_all = BodyState(pos=p, vel=v_batched, gain=2.0)
      s_mix = BodyState(pos=p, vel=v_shared, gain=2.0)
      return {
          'single': pjax.vmap(step_single, in_axes=0)(s_all),
          'single_mix': pjax.vmap(
              step_single,
              in_axes=(BodyState(pos=0, vel=None, gain=2.0),),
          )(s_mix),
          'comp': pjax.vmap(step_composite, in_axes=0)(s_all),
          'comp_mix': pjax.vmap(
              step_composite,
              in_axes=(BodyState(pos=0, vel=None, gain=2.0),),
          )(s_mix),
      }

    ts = Scheduler.run(prog, horizon=2)
    np.testing.assert_allclose(ts['single'][0], [21.0, 42.0, 63.0])
    np.testing.assert_allclose(ts['single_mix'][0], [21.0, 22.0, 23.0])
    np.testing.assert_allclose(ts['comp'][0], [20.0, 40.0, 60.0])
    np.testing.assert_allclose(ts['comp'][1], [3.0, 6.0, 9.0])
    np.testing.assert_allclose(ts['comp_mix'][0], [20.0, 20.0, 20.0])
    np.testing.assert_allclose(ts['comp_mix'][1], [3.0, 4.0, 5.0])

  def test_vmap_match_mixed_axis_arm_confinement(self):
    # Covers chained multi-node arms, `automaton`, subclock `fby`, & `out_axes`.
    def pick(flag, a, b):
      with match(flag) as m:
        with case(True):
          t = a * 2.0
          acc = fby(jnp.zeros(2), t)
          m.out = acc + 1.0
        with case(False):
          m.out = b
      with automaton(init='s0') as au:
        with state('s0'):
          s = a + 1.0
          au.v = s * 3.0
          until(flag, 's1')
        with state('s1'):
          au.v = a * 0.0
      cur_b = current(flag, when(flag, b), jnp.zeros(2))
      return m.out, au.v, cur_b, when(flag, a)

    a_val = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2), axis=0
    b_val = jnp.array([[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]])  # (2, 3), ax=1

    def prog():
      flag = Stream([True, False, True])
      return pjax.vmap(pick, in_axes=(None, 0, 1), out_axes=(0, 0, 0, 1))(
          flag, Stream([a_val]), Stream([b_val])
      )

    g = build_graph(prog, semantics=Semantics.SYNC, optimize=False)
    clocks = check_clocks(g)
    m_ts, au_ts, cb_ts, wa_ts = Scheduler.run_graph(g, horizon=3, clocks=clocks)
    np.testing.assert_allclose(m_ts[0], jnp.ones((3, 2)))
    np.testing.assert_allclose(
        m_ts[1], [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    )
    np.testing.assert_allclose(m_ts[2], a_val * 2.0 + 1.0)
    np.testing.assert_allclose(au_ts[0], (a_val + 1.0) * 3.0)
    np.testing.assert_allclose(au_ts[1], jnp.zeros_like(a_val))
    np.testing.assert_allclose(
        cb_ts[0], [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    )
    np.testing.assert_allclose(cb_ts[1], cb_ts[0])
    np.testing.assert_allclose(wa_ts[0], a_val.T)
    self.assertIs(wa_ts[1], Absent)
    np.testing.assert_allclose(wa_ts[2], a_val.T)

    # Impure lifted op stays impure under vmap (no unsound CSE merge).
    imp = pjax.lift(lambda z: z + 1.0, pure=False)
    v_imp = pjax.vmap(imp)
    g_imp = build_graph(
        lambda: (v_imp(Stream([a_val])), v_imp(Stream([a_val]))), optimize=True
    )
    self.assertLen(g_imp.nodes, 2)

    # Composite vmap supports `scan` over a PyTree `(a, b)` CombCallPlan.
    res_scan = Scheduler.run(
        lambda: pjax.vmap(
            lambda a, b: scan(
                (a, b), op=lambda p, q: (p[0] + q[0], p[1] + q[1])
            ),
            in_axes=(0, None),
        )(Stream([jnp.array([1.0, 2.0])]), Stream([jnp.array(10.0)])),
        horizon=2,
    )
    np.testing.assert_allclose(res_scan[0][0], [1.0, 2.0])
    np.testing.assert_allclose(res_scan[1][0], [2.0, 4.0])
    np.testing.assert_allclose(res_scan[1][1], [20.0, 20.0])

    # Composite vmap runs under Semantics.KAHN too.
    kahn_out = Scheduler.run(
        lambda: pjax.vmap(lambda x: fby(jnp.zeros(()), x + 1.0))(
            Stream([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])])
        ),
        horizon=2,
        semantics=Semantics.KAHN,
    )
    np.testing.assert_allclose(kahn_out[0], [0.0, 0.0])
    np.testing.assert_allclose(kahn_out[1], [2.0, 3.0])

  def test_grad_and_value_and_grad_on_pure_jax_nodes_and_pytrees(self):
    # Eager mode (outside Stage 1) delegates directly to `jax.grad`.
    self.assertAlmostEqual(
        float(pjax.grad(lambda x: x * x * x)(jnp.array(3.0))), 27.0, places=5
    )

    # Higher-order differentiation (`pjax.grad(pjax.grad(f))`) and lifted ops.
    d2_cube = pjax.grad(pjax.grad(lambda x: x**3))
    d_sin = pjax.grad(pjax.jnp.sin)
    out = Scheduler.run(
        lambda: (
            d2_cube(Stream([jnp.array(2.0), jnp.array(4.0)])),
            d_sin(Stream([jnp.array(0.0), jnp.array(0.5)])),
        ),
        horizon=2,
    )
    np.testing.assert_allclose(out[0], [12.0, 24.0], rtol=1e-5)
    np.testing.assert_allclose(out[1], [1.0, np.cos(0.5)], rtol=1e-5)

    # PyTree of `Stream`s + `has_aux=True` + `fuse_jit_islands` composition.
    def quad_loss(params, x, y):
      pred = jnp.dot(x, params['w']) + params['b']
      err = pred - y
      return 0.5 * jnp.sum(err * err), pred

    vg = pjax.value_and_grad(quad_loss, argnums=0, has_aux=True)

    def prog():
      w = fby(jnp.array([1.0, -1.0]), lambda: w_next)
      b = fby(jnp.array(0.5), lambda: b_next)
      x = Stream([jnp.array([2.0, 3.0]), jnp.array([-1.0, 2.0])])
      y = Stream([jnp.array(1.0), jnp.array(0.0)])
      (loss, pred), grads = vg({'w': w, 'b': b}, x, y)
      w_next = pjax.jnp.subtract(w, pjax.jnp.multiply(0.1, grads['w']))
      b_next = pjax.jnp.subtract(b, pjax.jnp.multiply(0.1, grads['b']))
      return {'loss': loss, 'pred': pred, 'w': w, 'b': b, 'gw': grads['w']}

    g_eager = build_graph(prog, semantics=Semantics.SYNC, optimize=True)
    g_jit, islands = pjax.fuse_jit_islands(g_eager)
    self.assertLen(islands, 1)
    t_eager = Scheduler.run_graph(g_eager, horizon=3)
    t_jit = Scheduler.run_graph(g_jit, horizon=3)
    for k in ('loss', 'pred', 'w', 'b', 'gw'):
      for t in range(3):
        np.testing.assert_allclose(t_jit[k][t], t_eager[k][t], atol=1e-5)
    # Tick 0: pred = 2*1 + 3*(-1) + 0.5 = -0.5, err = -1.5, gw = [-3.0, -4.5].
    np.testing.assert_allclose(t_eager['pred'][0], -0.5, atol=1e-5)
    np.testing.assert_allclose(t_eager['gw'][0], [-3.0, -4.5], atol=1e-5)

    # Single `Stream` carrying a dict PyTree (uses `treedef.flatten_up_to`).
    def single_stream_pytree_prog():
      p_s = Stream([{'w': jnp.array([1.0, -1.0]), 'b': jnp.array(0.5)}])
      x_s = Stream([jnp.array([2.0, 3.0])])
      y_s = Stream([jnp.array(1.0)])
      (loss, pred), g_s = vg(p_s, x_s, y_s)
      return {'loss': loss, 'pred': pred, 'g': g_s}

    ts_single = Scheduler.run(single_stream_pytree_prog, horizon=1)
    np.testing.assert_allclose(ts_single['g'][0]['w'], [-3.0, -4.5], atol=1e-5)
    np.testing.assert_allclose(ts_single['g'][0]['b'], -1.5, atol=1e-5)

    # Multi-argument `argnums=(0, 1)` + `vmap(grad(...))` + `jit(grad(...))`.
    g_both = pjax.grad(lambda a, b: jnp.sum(a * b * b), argnums=(0, 1))
    ga, gb = Scheduler.run(
        lambda: g_both(
            Stream([jnp.array([2.0, 3.0])]), Stream([jnp.array([4.0, 5.0])])
        ),
        horizon=1,
    )
    np.testing.assert_allclose(ga[0], [16.0, 25.0], atol=1e-5)
    np.testing.assert_allclose(gb[0], [16.0, 30.0], atol=1e-5)

    j_grad = pjax.jit(pjax.grad(lambda x: jnp.sum(x * x)))
    v_grad = pjax.vmap(pjax.grad(lambda x: x * x))
    jg_out, vg_out = Scheduler.run(
        lambda: (
            j_grad(Stream([jnp.array([1.0, 2.0, 3.0])])),
            v_grad(Stream([jnp.array([1.0, 2.0, 3.0])])),
        ),
        horizon=1,
    )
    np.testing.assert_allclose(jg_out[0], [2.0, 4.0, 6.0], atol=1e-5)
    np.testing.assert_allclose(vg_out[0], [2.0, 4.0, 6.0], atol=1e-5)

    # `jit` and `vmap` preserve multi-arg, `has_aux`, and PyTree unpacking.
    j_both = pjax.jit(g_both)
    v_both = pjax.vmap(g_both)
    jv_both = pjax.jit(v_both)
    (jga, jgb), (vga, vgb), (jvga, jvgb) = Scheduler.run(
        lambda: (
            j_both(
                Stream([jnp.array([2.0, 3.0])]),
                Stream([jnp.array([4.0, 5.0])]),
            ),
            v_both(
                Stream([jnp.array([[2.0], [3.0]])]),
                Stream([jnp.array([[4.0], [5.0]])]),
            ),
            jv_both(
                Stream([jnp.array([[2.0], [3.0]])]),
                Stream([jnp.array([[4.0], [5.0]])]),
            ),
        ),
        horizon=1,
    )
    np.testing.assert_allclose(jga[0], [16.0, 25.0], atol=1e-5)
    np.testing.assert_allclose(jgb[0], [16.0, 30.0], atol=1e-5)
    np.testing.assert_allclose(vga[0], [[16.0], [25.0]], atol=1e-5)
    np.testing.assert_allclose(vgb[0], [[16.0], [30.0]], atol=1e-5)
    np.testing.assert_allclose(jvga[0], [[16.0], [25.0]], atol=1e-5)
    np.testing.assert_allclose(jvgb[0], [[16.0], [30.0]], atol=1e-5)

    j_vg = pjax.jit(vg)
    v_vg = pjax.vmap(vg, in_axes=(None, 0, 0))
    ((jl, jp), jg), ((vl, vp), vgr) = Scheduler.run(
        lambda: (
            j_vg(
                {
                    'w': Stream([jnp.array([1.0, -1.0])]),
                    'b': Stream([jnp.array(0.5)]),
                },
                Stream([jnp.array([2.0, 3.0])]),
                Stream([jnp.array(1.0)]),
            ),
            v_vg(
                {
                    'w': Stream([jnp.array([1.0, -1.0])]),
                    'b': Stream([jnp.array(0.5)]),
                },
                Stream([jnp.array([[2.0, 3.0], [-1.0, 2.0]])]),
                Stream([jnp.array([1.0, 0.0])]),
            ),
        ),
        horizon=1,
    )
    np.testing.assert_allclose(jl[0], 1.125, atol=1e-5)
    np.testing.assert_allclose(jp[0], -0.5, atol=1e-5)
    np.testing.assert_allclose(jg['w'][0], [-3.0, -4.5], atol=1e-5)
    np.testing.assert_allclose(jg['b'][0], -1.5, atol=1e-5)
    self.assertEqual(vl[0].shape, (2,))
    self.assertEqual(vp[0].shape, (2,))
    np.testing.assert_allclose(vgr['w'][0][0], [-3.0, -4.5], atol=1e-5)

    # `grad(..., has_aux=True)` and `value_and_grad(..., has_aux=False)`.
    g_aux = pjax.grad(lambda x: (x * x, x + 1.0), has_aux=True)
    vg_no_aux = pjax.value_and_grad(lambda x: x * x)
    (gx_a, aux_v), (val_v, gx_v) = Scheduler.run(
        lambda: (
            g_aux(Stream([jnp.array(3.0)])),
            vg_no_aux(Stream([jnp.array(3.0)])),
        ),
        horizon=1,
    )
    np.testing.assert_allclose(gx_a[0], 6.0, atol=1e-5)
    np.testing.assert_allclose(aux_v[0], 4.0, atol=1e-5)
    np.testing.assert_allclose(val_v[0], 9.0, atol=1e-5)
    np.testing.assert_allclose(gx_v[0], 6.0, atol=1e-5)

    # Identical `grad` calls deduplicate under `optimize_graph` CSE.
    g_sq = pjax.grad(lambda x: x * x)

    def cse_prog():
      s = Stream([jnp.array(2.0)])
      return g_sq(s), g_sq(s)

    self.assertLen(build_graph(cse_prog, optimize=True).nodes, 1)

    # User `Stream.__getitem__` stays outside JIT islands while multi-output
    # JAX ops fuse via JAX-tagged `jax_out_*` projections.
    def slice_prog():
      x = Stream([jnp.array([1.0, 2.0, 3.0])])
      y = pjax.jnp.add(x, 1.0)
      return pjax.jnp.multiply(y[1:3], 2.0)

    _, slice_islands = pjax.fuse_jit_islands(
        build_graph(slice_prog, optimize=True)
    )
    self.assertEmpty(slice_islands)

    # Reject stateful, non-pure (including impure `vmap`), and host-NumPy fns.
    @psyre_sequ
    def stateful_node(x):
      acc = 0.0
      while True:
        acc += x
        x = yield acc

    with self.assertRaisesRegex(TypeError, 'stateful'):
      pjax.grad(stateful_node)
    with self.assertRaises(TypeError):
      pjax.grad(pjax.lift(lambda x: x, pure=False))
    with self.assertRaises(TypeError):
      pjax.grad(pjax.vmap(pjax.lift(lambda x: x * x, pure=False)))
    with self.assertRaises(TypeError):
      pjax.grad(np.sin)
    with self.assertRaises(TypeError):
      pjax.grad(lambda x: x, argnums=True)  # type: ignore[arg-type]
    # Stage-1 builder is cleared while `jax.grad` traces even on eager calls.
    with self.assertRaises(RuntimeError):
      build_graph(lambda: pjax.grad(lambda x: fby(0.0, x))(jnp.array(1.0)))
    with self.assertRaises(ExceptionGroup) as eg:
      Scheduler.run(
          lambda: pjax.grad(lambda x: fby(0.0, x))(Stream([jnp.array(1.0)])),
          horizon=1,
      )
    self.assertTrue(
        any(isinstance(e, RuntimeError) for e in eg.exception.exceptions)
    )


if __name__ == '__main__':
  absltest.main()
