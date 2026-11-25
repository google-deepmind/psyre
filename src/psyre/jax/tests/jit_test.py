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
"""Tests for `psyre.jax.jit`, `fuse_jit_islands`, and `Jaxpr` expansion."""

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
  from psyre import numpy as pnp
  from psyre.jax import lax as plax
else:  # pragma: no cover
  pjax = None
  plax = None
  pnp = None

from psyre.analysis import check_causality
from psyre.automaton import automaton, state, until
from psyre.builder import build_graph
from psyre.clock import check_clocks
from psyre.common import (
    Absent,
    Semantics,
    fby,
    ifthenelse,
    psyre_comb,
    push_prefix,
    when,
)
from psyre.match import case, match
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-import-not-at-top,g-importing-member,g-multiple-import


@absltest.skipIf(jax is None, 'jax is not installed')
class PsyreJaxJitTest(absltest.TestCase):
  """Tests lazy `jit`, convex island fusion, and opt-in `Jaxpr` expansion."""

  def test_lazy_jit_zero_stage1_tracing_and_absent_short_circuit(self):
    trace_count = [0]

    def f(x, w, *, scale=1.0):
      trace_count[0] += 1
      return jax.nn.relu(x @ w) * scale

    jf = pjax.jit(f, static_argnames=('scale',), name='mlp_step')

    def prog():
      x = Stream([jnp.array([1.0, -2.0]), jnp.array([3.0, 4.0])])
      w = Stream([jnp.array([[1.0, 0.0], [0.0, 2.0]])])
      c = Stream([False, False])
      gx = when(c, x + 0.0)
      gw = when(c, w + jnp.zeros((2, 2)))
      return {'gated': jf(gx, gw, scale=2.0)}

    g = build_graph(prog, semantics=Semantics.SYNC, optimize=True)
    # Zero Stage-1 tracing on `trace_jaxpr=False`!
    self.assertEqual(trace_count[0], 0)
    clocks = check_clocks(g)
    ts = Scheduler.run_graph(g, horizon=2, clocks=clocks)
    # Every tick is `Absent`, so `jax.jit` is never even traced or invoked.
    self.assertEqual(ts['gated'], [Absent, Absent])
    self.assertEqual(trace_count[0], 0)

    # On active ticks, traces once and caches.
    def active_prog():
      x = Stream([jnp.array([1.0, -2.0]), jnp.array([3.0, 4.0])])
      w = Stream([jnp.array([[1.0, 0.0], [0.0, 2.0]])])
      return jf(x, w, scale=2.0)

    out = Scheduler.run(active_prog, horizon=2)
    self.assertEqual(trace_count[0], 1)
    np.testing.assert_allclose(out[0], [2.0, 0.0])
    np.testing.assert_allclose(out[1], [6.0, 16.0])

    # Eager invocation outside Stage 1 and `nout` unpacking.
    eager = jf(jnp.array([1.0, 2.0]), jnp.eye(2), scale=3.0)
    np.testing.assert_allclose(eager, [3.0, 6.0])

    pair_fn = pjax.jit(lambda a, b: (a + b, a - b), nout=2)
    p_s, p_d = Scheduler.run(
        lambda: pair_fn(
            Stream([jnp.array([5.0, 6.0])]), Stream([jnp.array([1.0, 2.0])])
        ),
        horizon=1,
    )
    np.testing.assert_allclose(p_s[0], [6.0, 8.0])
    np.testing.assert_allclose(p_d[0], [4.0, 4.0])

    # `static_argnums` support + strict `in_avals` rejection when disabled.
    scaled = pjax.jit(lambda x, k: x * k, static_argnums=1)
    rs = Scheduler.run(
        lambda: scaled(Stream([jnp.array([2.0, 3.0])]), 5.0), horizon=1
    )
    np.testing.assert_allclose(rs[0], [10.0, 15.0])

    with self.assertRaises(ValueError):
      pjax.jit(
          lambda x: x,
          trace_jaxpr=False,
          in_avals=jax.ShapeDtypeStruct((2,), jnp.float32),
      )

  def test_fuse_jit_islands_chain_feedback_and_multi_exit(self):
    def prog():
      u = Stream([jnp.array([1.0, 2.0]), jnp.array([-1.0, 0.5])])
      w = Stream([jnp.array([[0.5, 0.1], [-0.2, 0.4]])])
      acc = fby(jnp.zeros(2), lambda: nxt)
      h1 = pjax.jnp.matmul(acc, w)
      h2 = pjax.jnp.add(h1, u)
      h3 = pjax.jnp.tanh(h2)
      nxt = pjax.jnp.multiply(h3, 0.9)
      sq = pjax.jnp.square(h3)
      return {'acc': acc, 'nxt': nxt, 'sq': sq}

    orig = build_graph(prog, semantics=Semantics.SYNC, optimize=True)
    orig_count = len(orig.nodes)
    fused, islands = pjax.fuse_jit_islands(orig)
    self.assertLen(islands, 1)
    self.assertGreaterEqual(islands[0].num_nodes, 5)
    self.assertEqual(islands[0].num_outputs, 2)
    self.assertLess(len(fused.nodes), orig_count)
    check_causality(fused)
    f_clocks = check_clocks(fused)

    # Original graph is untouched and both match numerically.
    self.assertLen(orig.nodes, orig_count)
    t_orig = Scheduler.run_graph(orig, horizon=3)
    t_fused = Scheduler.run_graph(fused, horizon=3, clocks=f_clocks)
    for k in ('acc', 'nxt', 'sq'):
      for t in range(3):
        np.testing.assert_allclose(t_fused[k][t], t_orig[k][t], atol=1e-5)

    # `jit_graph` convenience returns the fused `Graph` directly, and re-fusing
    # a multi-exit island is idempotent (`island_out_*` does not re-wrap).
    self.assertEqual(len(pjax.jit_graph(orig).nodes), len(fused.nodes))
    _, second_islands = pjax.fuse_jit_islands(fused)
    self.assertEmpty(second_islands)

  def test_fuse_jit_islands_preserves_convexity_across_host_node(self):
    # Diamond: `a -> host_mul -> c` AND `a -> c`.
    # Fusing `{a, c}` would create a cycle `island -> host_mul -> island`!
    @psyre_comb(pure=False)
    def host_bridge(arr):
      return jnp.asarray(np.asarray(arr) + 10.0)

    def prog():
      x = Stream([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])])
      a0 = pjax.jnp.sin(x)
      a = pjax.jnp.multiply(a0, 2.0)
      mid = host_bridge(a)
      c = pjax.jnp.add(a, mid)
      out = pjax.jnp.square(c)
      return {'out': out}

    orig = build_graph(prog, semantics=Semantics.SYNC, optimize=True)
    fused, islands = pjax.fuse_jit_islands(orig)
    check_causality(fused)
    check_clocks(fused)
    # `a0, a` form one island; `c, out` form a SEPARATE island (not merged!).
    self.assertLen(islands, 2)
    t_orig = Scheduler.run_graph(orig, horizon=2)
    t_fused = Scheduler.run_graph(fused, horizon=2)
    for t in range(2):
      np.testing.assert_allclose(t_fused['out'][t], t_orig['out'][t], atol=1e-5)

  def test_fuse_jit_islands_separates_subclocks_and_skips_host_numpy(self):
    def prog():
      flag = Stream([True, False, True, False])
      x = Stream([jnp.array([float(i), float(i + 1)]) for i in range(4)])
      x_dyn = x + 0.0
      # Separate pure NumPy host chain (must NOT be fused into a JAX island).
      hn = Stream([np.array([1.0, 2.0])])
      host_out = pnp.cos(pnp.sin(hn))
      # Separate scalar `ifthenelse` stays unfused outside islands.
      tag_mux = ifthenelse(flag, Stream('a'), Stream('b'))
      # Double negation triggers `replace_stream` alias + DCE on `match` cond.
      with match(~~flag) as m:
        with case(True):
          m.res = pjax.jnp.square(pjax.jnp.sin(x_dyn))
        with case(False):
          m.res = pjax.jnp.abs(pjax.jnp.cos(x_dyn))
      # `Stream.max` on 0-D JAX scalars stays out of `jax.jit` (no bool trace).
      mx = pjax.jnp.sum(x).max(pjax.jnp.prod(x))
      return {'res': m.res, 'host': host_out, 'tag': tag_mux, 'mx': mx}

    orig = build_graph(prog, semantics=Semantics.SYNC, optimize=True)
    fused, islands = pjax.fuse_jit_islands(orig)
    check_causality(fused)
    check_clocks(fused)
    # Two separate subclock islands (one for `flag=True`, one for `flag=False`).
    self.assertLen(islands, 2)
    self.assertNotEqual(islands[0].clock, islands[1].clock)
    self.assertTrue(
        any(n.op in ('sin', 'cos') for n in fused.nodes),
        'Host psyre.numpy nodes must stay outside JAX islands.',
    )

    t_orig = Scheduler.run_graph(orig, horizon=4)
    t_fused = Scheduler.run_graph(fused, horizon=4)
    for t in range(4):
      np.testing.assert_allclose(t_fused['res'][t], t_orig['res'][t], atol=1e-5)
      np.testing.assert_allclose(
          t_fused['host'][t], t_orig['host'][t], atol=1e-5
      )
      np.testing.assert_allclose(t_fused['mx'][t], t_orig['mx'][t], atol=1e-5)
    self.assertEqual(t_fused['tag'], t_orig['tag'])

    # `extra_roots` keeps intermediate `Stream` readable in the fused `Graph`.
    def with_named():
      x = Stream([jnp.array([1.0, 2.0])])
      mid = pjax.jnp.sin(x)
      mid.name = 'mid_sin'
      y = pjax.jnp.cos(mid)
      return {'y': y}

    og = build_graph(with_named, optimize=True)
    fg, _ = pjax.fuse_jit_islands(og, extra_roots=('mid_sin',))
    self.assertIn('mid_sin', fg.resolve_named_streams(('mid_sin',)))

    with self.assertRaises(ValueError):
      pjax.fuse_jit_islands(og, min_nodes=0)
    with self.assertRaises(TypeError):
      pjax.fuse_jit_islands(og, extra_roots='mid_sin')

  def test_trace_jaxpr_requires_explicit_shape_dtype_struct_hints(self):
    def f(x, y):
      return x + y

    # Missing `in_avals` rejected.
    with self.assertRaises(ValueError):
      pjax.jit(f, trace_jaxpr=True)

    # Non-ShapeDtypeStruct hint rejected at wrap time.
    with self.assertRaises(TypeError):
      pjax.jit(f, trace_jaxpr=True, in_avals=(jnp.zeros(2), jnp.zeros(2)))
    with self.assertRaises(TypeError):
      pjax.jit(f, trace_jaxpr=True, in_avals=123)

    # Arity mismatch rejected at call time.
    bad_arity = pjax.jit(
        f,
        trace_jaxpr=True,
        in_avals=(jax.ShapeDtypeStruct((2,), jnp.float32),),
    )
    with self.assertRaises(ValueError):
      build_graph(
          lambda: bad_arity(
              Stream([jnp.zeros(2, jnp.float32)]),
              Stream([jnp.zeros(2, jnp.float32)]),
          )
      )

    # PyTree structure mismatch rejected at call time.
    bad_tree = pjax.jit(
        lambda d: d['a'],
        trace_jaxpr=True,
        in_avals=({'a': jax.ShapeDtypeStruct((2,), jnp.float32)},),
    )
    with self.assertRaises(ValueError):
      build_graph(
          lambda: bad_tree(Stream([jnp.zeros(2, jnp.float32)]))
      )

  def test_trace_jaxpr_expansion_primitives_dce_and_roundtrip_fusion(self):
    bias = jnp.array([0.5, -0.5], dtype=jnp.float32)

    def mlp(x, w):
      # Closed-over `bias` tests `constvars`; `2.0` tests `Literal`.
      h = jax.nn.relu(x @ w + bias)
      unused_branch = jnp.sin(x) * 2.0
      return {'h': h, 'sum': jnp.sum(h), 'aux': unused_branch}

    x_spec = jax.ShapeDtypeStruct((2,), jnp.float32)
    w_spec = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    traced_mlp = pjax.jit(
        mlp,
        trace_jaxpr=True,
        in_avals=(x_spec, w_spec),
        name='mlp',
    )

    x_val = jnp.array([1.0, -3.0], dtype=jnp.float32)
    w_val = jnp.array([[2.0, 1.0], [-1.0, 0.5]], dtype=jnp.float32)

    def prog():
      x = Stream([x_val, x_val * 2.0])
      w = Stream([w_val])
      out = traced_mlp(x, w)
      # Drop `'aux'` so DCE should eliminate the `sin` equation from `mlp`!
      return {'h': out['h'], 'sum': out['sum']}

    g_unopt = build_graph(prog, optimize=False)
    prim_names_unopt = {
        pjax.get_primitive(n).name
        for n in g_unopt.nodes
        if pjax.get_primitive(n) is not None
    }
    self.assertIn('dot_general', prim_names_unopt)
    self.assertIn('sin', prim_names_unopt)
    self.assertTrue(all(n.name.startswith('mlp/') for n in g_unopt.nodes))
    self.assertTrue(all(pjax.is_jax_step_func(n) for n in g_unopt.nodes))

    # DCE removes the unused `sin` node from the expanded `Jaxpr`!
    g_opt = build_graph(prog, optimize=True)
    prim_names_opt = {
        pjax.get_primitive(n).name
        for n in g_opt.nodes
        if pjax.get_primitive(n) is not None
    }
    self.assertIn('dot_general', prim_names_opt)
    self.assertNotIn('sin', prim_names_opt)
    self.assertLess(len(g_opt.nodes), len(g_unopt.nodes))

    # Execute fine-grained expanded graph and verify against eager JAX.
    ref0 = mlp(x_val, w_val)
    ref1 = mlp(x_val * 2.0, w_val)
    t_opt = Scheduler.run_graph(g_opt, horizon=2)
    np.testing.assert_allclose(t_opt['h'][0], ref0['h'], atol=1e-5)
    np.testing.assert_allclose(t_opt['sum'][1], ref1['sum'], atol=1e-5)

    # Round-trip the expanded `Jaxpr` back into ONE fused `jax.jit` super-node!
    fused, islands = pjax.fuse_jit_islands(g_opt)
    self.assertLen(islands, 1)
    self.assertLen(g_opt.nodes, islands[0].num_nodes)
    t_fused = Scheduler.run_graph(fused, horizon=2)
    np.testing.assert_allclose(t_fused['h'][0], ref0['h'], atol=1e-5)
    np.testing.assert_allclose(t_fused['sum'][1], ref1['sum'], atol=1e-5)

    # Outside Stage 1, calls `jax.jit(mlp)` eagerly.
    eager = traced_mlp(x_val, w_val)
    np.testing.assert_allclose(eager['h'], ref0['h'], atol=1e-5)

  def test_trace_jaxpr_multi_output_dropvar_and_nested_subjaxpr(self):
    @jax.jit
    def inner(v):
      return jnp.sin(v) + jnp.cos(v)

    def outer(v, k):
      # Exercises nested `jit`, `top_k` (`multiple_results=True`), & `DropVar`.
      h = inner(v)
      top_vals, _ = jax_lax.top_k(h, k)
      return top_vals

    spec = jax.ShapeDtypeStruct((4,), jnp.float32)
    vals = jnp.array([0.1, 1.2, -0.5, 2.3], dtype=jnp.float32)

    atomic_jit = pjax.jit(
        outer,
        trace_jaxpr=True,
        static_argnums=1,
        in_avals=spec,
        expand_subjaxprs=False,
        name='atomic',
    )
    flat_jit = pjax.jit(
        outer,
        trace_jaxpr=True,
        static_argnums=1,
        in_avals=spec,
        expand_subjaxprs=True,
        name='flat',
    )

    g_atomic = build_graph(
        lambda: atomic_jit(Stream([vals]), 2), optimize=False
    )
    g_flat = build_graph(lambda: flat_jit(Stream([vals]), 2), optimize=False)
    flat_prims = {
        pjax.get_primitive(n).name
        for n in g_flat.nodes
        if pjax.get_primitive(n) is not None
    }
    self.assertIn('sin', flat_prims)
    self.assertIn('cos', flat_prims)
    self.assertIn('top_k', flat_prims)
    self.assertGreater(len(g_flat.nodes), len(g_atomic.nodes))

    ref = outer(vals, 2)
    np.testing.assert_allclose(
        Scheduler.run_graph(g_atomic, horizon=1)[0], ref, atol=1e-5
    )
    np.testing.assert_allclose(
        Scheduler.run_graph(g_flat, horizon=1)[0], ref, atol=1e-5
    )

  def test_trace_jaxpr_rejects_impure_effects(self):
    def impure_fn(x):
      jax.debug.print('x={x}', x=x)
      return x + 1.0

    bad = pjax.jit(
        impure_fn,
        trace_jaxpr=True,
        in_avals=jax.ShapeDtypeStruct((2,), jnp.float32),
    )
    with self.assertRaises(pjax.ImpureJaxprError):
      build_graph(lambda: bad(Stream([jnp.ones(2, jnp.float32)])))

  def test_expand_jaxpr_direct_vmap_and_rewrite_dtype_composition(self):
    def step(x, w):
      return jnp.tanh(x @ w + 1.0)

    x_av = jax.ShapeDtypeStruct((2,), jnp.float32)
    w_av = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    closed = jax.make_jaxpr(step)(x_av, w_av)

    def prog_direct():
      x = Stream([jnp.array([1.001, 2.0], jnp.float32)])
      w = Stream([jnp.eye(2, dtype=jnp.float32) * 0.5])
      return pjax.expand_jaxpr(closed, x, w, name='direct')

    g = build_graph(prog_direct, optimize=True)
    out = Scheduler.run_graph(g, horizon=1)
    np.testing.assert_allclose(
        out[0],
        step(
            jnp.array([1.001, 2.0], jnp.float32),
            jnp.eye(2, dtype=jnp.float32) * 0.5,
        ),
        atol=1e-5,
    )

    # Compose `expand_jaxpr` inside composite Stage-1 `vmap` (rewrites each
    # expanded `JaxprEqn` node cleanly!).
    traced = pjax.jit(step, trace_jaxpr=True, in_avals=(x_av, w_av))
    xs = jnp.array([[1.0, 2.0], [-1.0, 0.5], [0.0, -2.0]], dtype=jnp.float32)
    w = jnp.array([[0.5, -0.2], [0.1, 0.4]], dtype=jnp.float32)
    # pylint: disable=unnecessary-lambda
    # Wrap in lambda so `vmap` runs its composite Stage-1 pass across the
    # expanded `JaxprEqn` nodes rather than fast-path `_psyre_inst_func`.
    v_out = Scheduler.run(
        lambda: pjax.vmap(
            lambda a, b: traced(a, b), in_axes=(0, None)
        )(Stream([xs]), Stream([w])),
        horizon=1,
    )
    # pylint: enable=unnecessary-lambda
    np.testing.assert_allclose(
        v_out[0], jax.vmap(step, in_axes=(0, None))(xs, w), atol=1e-5
    )

    # Compose `rewrite_dtype` (with `preferred_element_type`) & execute `bf_g`.
    bf_g = pjax.rewrite_dtype(
        g,
        target_dtype=jnp.bfloat16,
        primitives=(jax_lax.dot_general_p,),
        preferred_element_type=jnp.float32,
    )
    dot_nodes = [
        n for n in bf_g.nodes if pjax.get_primitive(n) is jax_lax.dot_general_p
    ]
    self.assertLen(dot_nodes, 1)
    bf_out = Scheduler.run_graph(bf_g, horizon=1)
    self.assertEqual(bf_out[0].dtype, jnp.float32)
    # 1.001 rounds to 1.0 in bfloat16, proving the operand cast executed.
    self.assertFalse(jnp.allclose(bf_out[0], out[0], atol=1e-5))
    exp_bf = step(
        jnp.array([1.001, 2.0], jnp.bfloat16).astype(jnp.float32),
        jnp.eye(2, dtype=jnp.bfloat16).astype(jnp.float32) * 0.5,
    )
    np.testing.assert_allclose(bf_out[0], exp_bf, atol=1e-5)

    # Constant folding of `iota`/`ones`, CSE across expanded eqns, & subclock.
    def with_consts(v):
      return jnp.sin(v) + jnp.sin(v) + jnp.arange(2, dtype=jnp.float32) + 1.0

    tr_c = pjax.jit(with_consts, trace_jaxpr=True, in_avals=x_av)

    def sub_prog():
      c = Stream([True, False, True])
      x = Stream([jnp.array([1.0, 2.0], jnp.float32)])
      return tr_c(when(c, x + 0.0))

    sg = build_graph(sub_prog, semantics=Semantics.SYNC, optimize=True)
    self.assertLen([n for n in sg.nodes if n.op == 'sin'], 1)
    s_out = Scheduler.run_graph(sg, horizon=3, clocks=check_clocks(sg))
    self.assertIs(s_out[1], Absent)
    np.testing.assert_allclose(
        s_out[0], with_consts(jnp.array([1.0, 2.0], jnp.float32)), atol=1e-5
    )

    # Inside an automaton, inactive arm stays short-circuited after fusion.
    def auto_prog():
      go_b = Stream([False, True, False])
      u = Stream([jnp.array([1.0, 2.0], jnp.float32)])
      u_dyn = u + 0.0
      with automaton(init='A') as a:
        with state('A'):
          a.val = plax.sin(plax.add(u_dyn, jnp.ones(2, jnp.float32)))
          until(go_b, 'B')
        with state('B'):
          a.val = plax.cos(plax.mul(u_dyn, jnp.array(2.0, jnp.float32)))
      return a.val

    ag = build_graph(auto_prog, semantics=Semantics.SYNC, optimize=True)
    af, a_islands = pjax.fuse_jit_islands(ag)
    self.assertLen(a_islands, 2)
    check_causality(af)
    check_clocks(af)
    got_tr = Scheduler.run_graph(af, horizon=3)
    ref_tr = Scheduler.run_graph(ag, horizon=3)
    for t in range(3):
      np.testing.assert_allclose(got_tr[t], ref_tr[t], atol=1e-5)

    # Island fusion under Semantics.KAHN preserves trace equivalence.
    kg = build_graph(
        lambda: plax.cos(
            plax.sin(Stream([jnp.array([1.0, 2.0], jnp.float32)]))
        ),
        semantics=Semantics.KAHN,
    )
    kf, k_islands = pjax.fuse_jit_islands(kg)
    self.assertLen(k_islands, 1)
    np.testing.assert_allclose(
        Scheduler.run_graph(kf, horizon=2)[0],
        Scheduler.run_graph(kg, horizon=2)[0],
        atol=1e-5,
    )

    # Single 1-element tuple arg with `in_avals=(spec,)` under `trace_jaxpr`.
    t1 = pjax.jit(
        lambda pair: pair[0] + 1.0, trace_jaxpr=True, in_avals=(x_av,)
    )
    t1_out = Scheduler.run(
        lambda: t1((Stream([jnp.array([1.0, 2.0], jnp.float32)]),)), horizon=1
    )
    np.testing.assert_allclose(t1_out[0], [2.0, 3.0])

  def test_fuse_jit_islands_common_prefix_across_scopes(self):
    def nested_scopes():
      x = Stream([jnp.array([1.0, 2.0])])
      with push_prefix('shared'):
        with push_prefix('enc'):
          h = pjax.jnp.sin(x)
        with push_prefix('dec'):
          y = pjax.jnp.cos(h)
      return y

    fg, islands = pjax.fuse_jit_islands(build_graph(nested_scopes))
    self.assertLen(islands, 1)
    self.assertEqual(fg.nodes[0].prefixes, (('shared', 0),))
    self.assertEqual(fg.nodes[0].name, 'shared/jit_island_0')

    def disjoint_scopes():
      x = Stream([jnp.array([1.0, 2.0])])
      with push_prefix('enc'):
        h = pjax.jnp.sin(x)
      with push_prefix('dec'):
        y = pjax.jnp.cos(h)
      return y

    fg_top, islands_top = pjax.fuse_jit_islands(build_graph(disjoint_scopes))
    self.assertLen(islands_top, 1)
    self.assertEqual(fg_top.nodes[0].prefixes, ())
    self.assertEqual(fg_top.nodes[0].name, 'jit_island_0')

    def same_scope():
      x = Stream([jnp.array([1.0, 2.0])])
      with push_prefix('block'):
        h = pjax.jnp.sin(x)
        y = pjax.jnp.cos(h)
      return y

    fg_same, islands_same = pjax.fuse_jit_islands(build_graph(same_scope))
    self.assertLen(islands_same, 1)
    self.assertEqual(fg_same.nodes[0].prefixes, (('block', 0),))
    self.assertEqual(fg_same.nodes[0].name, 'block/jit_island_0')


if __name__ == '__main__':
  absltest.main()
