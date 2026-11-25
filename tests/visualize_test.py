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
"""Unit tests for Psyre Graphviz/DOT layout and SVG schematic rendering."""

import base64
import gzip
import json
import logging
import re
import subprocess
from unittest import mock
import xml.etree.ElementTree as ET

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre import gui
from psyre import visualize
from psyre.automaton import (
    automaton,
    state,
    unless,
    until,
)
from psyre.builder import build_graph
from psyre.common import (
    counter,
    fby,
    ifthenelse,
    logger,
    pre,
    psyre_comb,
    psyre_node,
    reset,
    switch,
    when,
)
from psyre.match import (
    case,
    default,
    match,
)
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import

_SVG_NS = '{http://www.w3.org/2000/svg}'


def _parse_edge_metadata(svg: str) -> list[dict[str, str]]:
  """Extracts edge DOM metadata attributes from a rendered SVG string."""
  root = ET.fromstring(svg)
  edges = []
  for el in root.iter(f'{_SVG_NS}g'):
    if el.get('class') == 'edge':
      edges.append({
          'id': el.get('id', ''),
          'src': el.get('data-psyre-src', ''),
          'dst': el.get('data-psyre-dst', ''),
          'dst_port': el.get('data-psyre-dst-port', ''),
          'info': el.get('data-psyre-info', ''),
      })
  return edges


def _delayed_edges(svg: str) -> list[dict[str, str]]:
  """Returns rendered SVG edge metadata for delayed feedback edges."""
  return [e for e in _parse_edge_metadata(svg) if '.delayed_next' in e['info']]


class VisualizeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def _assert_back_edges(self, dot_src: str, count: int) -> None:
    dashed_lines = [
        line.strip()
        for line in dot_src.splitlines()
        if '#7C3AED' in line and 'dashed' in line
    ]
    self.assertLen(dashed_lines, count)
    for line in dashed_lines:
      self.assertIn('dir="back"', line)
      self.assertIn('constraint=true', line)

  def test_feedback_edges_use_dir_back_and_preserve_svg_metadata(self):
    def counter_prog():
      x = fby(0, lambda: x + 1, name='acc')
      return {'x': x}

    def pre_prog():
      p = pre(lambda: x, name='prev_x')
      x = ifthenelse(Stream.once(0), 0, p + 1)
      return {'x': x}

    def mutual_prog():
      x = fby(0, lambda: y + 1, name='x')
      y = fby(10, lambda: x + 2, name='y')
      return {'x': x, 'y': y}

    cases = (
        ('fby', counter_prog, 'fby', 1, ('LR', 'TB'), True),
        ('pre', pre_prog, 'pre', 1, ('LR',), False),
        ('mutual_fby', mutual_prog, 'fby', 2, ('LR',), False),
    )
    for label, prog, delay_op, count, rankdirs, check_src in cases:
      g = build_graph(prog)
      delay_keys = {f'n_{n.id}' for n in g.nodes if n.op == delay_op}
      with self.subTest(case=label):
        self._assert_back_edges(visualize.build_pydot(g).to_string(), count)
        for rankdir in rankdirs:
          svg = visualize.render_svg(g, rankdir=rankdir)
          delayed = _delayed_edges(svg)
          self.assertLen(delayed, count)
          self.assertEqual({e['dst'] for e in delayed}, delay_keys)
          if check_src:
            add_node = next(n for n in g.nodes if n.op != delay_op)
            self.assertEqual(delayed[0]['src'], f'n_{add_node.id}')

    # Verify the `render_html` convenience wrapper without re-running the FSM.
    self.assertIn('<svg', visualize.render_html(build_graph(counter_prog)))

  def test_forward_fby_stays_forward_and_self_loop_unconstrained(self):
    def prog():
      inp = Stream([1, 2, 3, 4], name='inp')
      fwd = fby(0, inp + 1, name='fwd')
      self_x = fby(0, lambda: self_x, name='self_x')
      return {'fwd': fwd, 'self_x': self_x}

    g = build_graph(prog)
    dot_lines = visualize.build_pydot(g).to_string().splitlines()
    for name, constraint, style, color in (
        ('fwd', 'constraint=true', 'style="solid"', 'color="#334155"'),
        ('self_x', 'constraint=false', 'style="dashed"', 'color="#7C3AED"'),
    ):
      node = next(n for n in g.nodes if n.name == name)
      matched = [
          l.strip() for l in dot_lines if f'psyre_e_{node.id}_p1' in l
      ]
      self.assertLen(matched, 1)
      self.assertNotIn('dir="back"', matched[0])
      self.assertIn(constraint, matched[0])
      self.assertIn(style, matched[0])
      self.assertIn(color, matched[0])

  def test_automaton_feedback_edges_stay_inside_cluster_and_render_html(self):
    def fsm_prog():
      a = Stream([False, True, False, True], name='a')
      r = Stream([False, False, True, False], name='r')
      with automaton(init='WAIT', name='await_a') as fsm:
        with state('WAIT', unless(r, 'WAIT')):
          fsm.y = counter(start=0)
          until(a, 'DONE')
        with state('DONE', unless(r, 'WAIT')):
          fsm.y = counter(start=10)
      return {'y': fsm.y}

    g = build_graph(fsm_prog)
    self._assert_back_edges(visualize.build_pydot(g).to_string(), 2)

    viz = visualize.visualize_graph(g, title='abro_box')
    self.assertIn('<svg', viz.svg)
    self.assertIn('psyre-viz-card', viz.html)
    self.assertIn('fsm_box_0', viz.dot)
    self.assertIn('⊞ await_a', viz.dot)

  def test_multi_valued_when_and_switch_visualization(self):
    def prog():
      sel = Stream(['A', 'B', 'C'], name='sel')
      x = Stream([1, 2, 3], name='x')
      w_eq = when(sel, x, values='A')
      w_ne = when(sel, x, values='A', negated=True)
      sw = switch(sel, {'A': x, 'B': x + 1}, default=x + 2, name='mux_sw')
      return {'w_eq': w_eq, 'w_ne': w_ne, 'sw': sw}

    g = build_graph(prog)
    dot_src = visualize.build_pydot(g).to_string()
    self.assertIn("='A'", dot_src)
    self.assertIn("≠'A'", dot_src)
    svg = visualize.render_svg(g)
    self.assertIn('data-psyre-shape="mux-trapezoid"', svg)
    self.assertIn('data-psyre-shape="selector-trapezoid"', svg)
    self.assertIn("match=='A'", svg)

  def test_large_graph_uses_net_ref_badges_and_renders_svg(self):
    def prog():
      x = Stream([1, 2, 3], name='x')
      acc = fby(0, lambda: cur, name='acc')
      cur = acc + x
      for _ in range(45):
        cur = (cur + x) + acc
      return {'out': cur}

    g = build_graph(prog)
    self.assertGreater(len(g.nodes), 80)
    dot_src = visualize.build_pydot(g).to_string()
    self.assertIn('ref_', dot_src)
    self.assertIn('↪ ', dot_src)
    svg = visualize.render_svg(g)
    self.assertIn('<svg', svg)

  def test_inline_static_stream_nodes_render_without_background_boxes(
      self,
  ):
    def prog():
      a = Stream([False, True, False, True], name='a')
      with automaton(init='S0', name='fsm') as fsm:
        with state('S0'):
          until(Stream.once(2), 'S1')
          fsm.y = a
        with state('S1'):
          fsm.y = True
      return {'y': fsm.y}

    g = build_graph(prog)
    svg = visualize.render_svg(g, collapse_automata=False)
    root = ET.fromstring(svg)
    checked = 0
    for el in root.iter(f'{_SVG_NS}g'):
      if (
          el.get('class') == 'node'
          and el.get('data-psyre-kind') == 'static_stream'
      ):
        checked += 1
        self.assertIsNone(el.find(f'{_SVG_NS}path'))
        self.assertIsNone(el.find(f'{_SVG_NS}polygon'))
    self.assertGreater(checked, 0)
    # Resettable `static_stream` node (`Stream.once(2)` inside `state('S0')`)
    # renders with its sequence metadata inside the SVG tooltip.
    self.assertIn('op=static_stream', svg)
    # Large symbolic repetition (`Stream.once(10**8)`) formats compactly in DOT.
    big_dot = visualize.build_pydot(
        build_graph(lambda: Stream.once(10**8))
    ).to_string()
    self.assertIn(str(10**8), big_dot)

  def test_stream_sum_renders_multi_port_node_and_lazy_gzip_views(self):
    def prog():
      a = Stream([1, 2, 3], name='a')
      b = Stream([4, 5, 6], name='b')
      c = Stream([7, 8, 9], name='c')
      return {'total': Stream.sum(a, b, c, name='total')}

    g = build_graph(prog)
    dot_src = visualize.build_pydot(g).to_string()
    self.assertIn('PORT="in2"', dot_src)
    svg = visualize.render_svg(g)
    self.assertIn('<svg', svg)

    # Verify lazy gzip-compressed view packaging when total raw SVG exceeds
    # `_LAZY_GZIP_THRESHOLD_BYTES`: only initial view is inline in DOM and
    # every remaining view decodes cleanly via gzip+base64.
    big_svg = svg.replace('</svg>', f'<desc>{"x" * 160_000}</desc></svg>')
    bundle = gui.SchematicBundle(
        views={
            '': gui.SchematicView(raw_svg=svg, dot_src=dot_src),
            'box': gui.SchematicView(raw_svg=big_svg, dot_src=dot_src),
            'box|box/sub': gui.SchematicView(raw_svg=big_svg, dot_src=dot_src),
        },
        initial_view_key='',
        fsm_prefixes=('box', 'box/sub'),
        title='lazy_gzip_test',
        node_count=len(g.nodes),
    )
    html_out = gui.render_bundle_html(bundle)
    self.assertEqual(html_out.count('class="psyre-viz-view'), 1)
    match_gz = re.search(r'const gzViews = (\{.*?\});', html_out)
    self.assertIsNotNone(match_gz)
    gz_map = json.loads(match_gz.group(1))
    self.assertEqual(set(gz_map.keys()), {'box', 'box|box/sub'})
    for b64_payload in gz_map.values():
      decoded = gzip.decompress(base64.b64decode(b64_payload)).decode('utf-8')
      self.assertIn('<svg', decoded)
      self.assertIn('</svg>', decoded)

  def test_dot_preserves_lowered_when_unique_activity_and_fsm_clusters(self):
    # 1. Nested match() under reset(): every edge endpoint `n_X` must have a
    #    matching node declaration (no dropped live `when` ghost ovals).
    def nested_match_with_reset():
      r = Stream([False, True, False, False], name='r')
      c1 = Stream([True, True, False, True], name='c1')
      c2 = Stream([1, 2, 1, 1], name='c2')
      with reset(r):
        with match(c1) as m1:
          with case(True):
            with match(c2) as m2:
              with case(1):
                m2.y = counter(0)
              with default():
                m2.y = 0
            m1.out = m2.y
          with case(False):
            m1.out = 0
      return m1.out

    g_nested = build_graph(nested_match_with_reset, optimize=False)
    dot_nested = visualize.build_pydot(g_nested).to_string()
    declared = set(re.findall(r'\bn_(\d+)\s*\[', dot_nested))
    referenced = set(re.findall(r'\bn_(\d+)(?::|\s*->)', dot_nested))
    self.assertTrue(referenced.issubset(declared), referenced - declared)

    # 2. Punctuation-stripped operators (`== 1` vs `!= 1`) must not collide
    #    into a single activity cluster or duplicate DOM ID.
    def match_case_vs_default():
      sel = Stream([1, 2, 1, 2], name='sel')
      with match(sel) as m:
        with case(1):
          m.o = counter(0) + 1
        with default():
          m.o = counter(10) + 2
      return m.o

    dot_act = visualize.build_pydot(
        build_graph(match_case_vs_default)
    ).to_string()
    self.assertIn('sel == 1', dot_act)
    self.assertIn('sel != 1', dot_act)
    act_names = re.findall(r'subgraph\s+(cluster_act_\S+)', dot_act)
    self.assertLen(act_names, 2)
    self.assertLen(set(act_names), 2)
    act_ids = re.findall(r'id="(psyre_act_cluster_[^"]+)"', dot_act)
    self.assertLen(act_ids, 2)
    self.assertLen(set(act_ids), 2)

    # 3. Subgraph with divergent equal-depth subclocks computes longest common
    #    prefix so the FSM/node cluster appears once, not duplicated.
    @psyre_node('grp')
    def divergent_box(w1: Stream, w2: Stream):
      return w1 + 10, w2 + 20

    def divergent_prog():
      c1 = Stream([True, False, True], name='c1')
      c2 = Stream([False, True, False], name='c2')
      x = Stream([1, 2, 3], name='x')
      return divergent_box(when(c1, x), when(c2, x))

    dot_div = visualize.build_pydot(build_graph(divergent_prog)).to_string()
    fsm_clusters = re.findall(r'subgraph\s+(cluster_fsm_0_\S+)', dot_div)
    self.assertLen(fsm_clusters, 1)
    fsm_ids = re.findall(r'id="(psyre_fsm_cluster_0)"', dot_div)
    self.assertLen(fsm_ids, 1)

    # 4. Non-NamedTuple tuple subclass output, `abs(Stream)`, and missing
    #    `initial_view_key` fallback in `gui.build_visualization`.
    class SubTuple(tuple):
      pass

    def subtuple_abs():
      return SubTuple((abs(Stream([-1, 2], name='x')),))

    svg_sub = visualize.render_svg(build_graph(subtuple_abs))
    self.assertIn('<svg', svg_sub)
    self.assertIn('|·|', svg_sub)
    view_only = gui.SchematicBundle(
        views={'only': gui.SchematicView(raw_svg=svg_sub, dot_src='digraph{}')},
        initial_view_key='missing',
        node_count=1,
    )
    self.assertIn('<svg', gui.build_visualization(view_only).svg)

    # 5. Inline HTML `<script>` JSON payloads escape `</script>` and `<!--`.
    evil_prefix = 'fsm</script><script>alert(1)</script><!--'
    evil_bundle = gui.SchematicBundle(
        views={
            evil_prefix: gui.SchematicView(
                raw_svg=svg_sub, dot_src='digraph{}'
            )
        },
        fsm_prefixes=(evil_prefix,),
        initial_view_key=evil_prefix,
        node_count=1,
    )
    html_out = gui.render_bundle_html(evil_bundle)
    self.assertNotIn('</script><script>', html_out)
    self.assertNotIn('<!--', html_out)
    self.assertIn('\\u003c/script\\u003e', html_out)

  def test_user_operators_colliding_with_builtin_names(self):
    # User operators colliding with builtin names (`merge`, `when`, `fby`,
    # `ifthenelse`, `prefix_sum`, `buffer`, `counter`) render as ordinary
    # combinational boxes rather than builtin schematic shapes/glyphs.
    @psyre_comb
    def merge(a, b):
      return a + b

    @psyre_comb
    def when(a, b):  # pylint: disable=redefined-outer-name
      return a + b

    @psyre_comb
    def fby(a, b):  # pylint: disable=redefined-outer-name
      return a + b

    @psyre_comb
    def ifthenelse(a, b, c):  # pylint: disable=redefined-outer-name
      return a + b + c

    @psyre_comb
    def prefix_sum(a):
      return a

    @psyre_comb
    def buffer(a):
      return a

    @psyre_comb
    def counter(a):  # pylint: disable=redefined-outer-name
      return a

    def user_named_prog():
      s = Stream([1, 2], name='s')
      return (
          merge(s, s)
          + when(s, s)
          + fby(s, s)
          + ifthenelse(s, s, s)
          + prefix_sum(s)
          + buffer(s)
          + counter(s)
      )

    user_svg = visualize.render_svg(build_graph(user_named_prog))
    self.assertIn('<svg', user_svg)
    self.assertNotIn('delayed_next', user_svg)
    self.assertNotIn('>Σ<', user_svg)
    self.assertNotIn('>042<', user_svg)

  def test_viewbox_contains_expanded_fby_square_and_validation_guards(self):
    # Expanded `fby` square with a long name must remain inside the SVG viewBox.
    def wide_fby_prog():
      x = Stream([1, 2, 3], name='x')
      return fby(0, x, name='very_long_state_register_name')

    svg = visualize.render_svg(build_graph(wide_fby_prog))
    root = ET.fromstring(svg)
    vx, vy, vw, vh = (float(v) for v in root.get('viewBox', '').split())
    graph_g = root.find(f'{_SVG_NS}g')
    self.assertIsNotNone(graph_g)
    m_tr = re.search(
        r'translate\(\s*(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)\s*\)',
        graph_g.get('transform', ''),
    )
    self.assertIsNotNone(m_tr)
    gx, gy = float(m_tr.group(1)), float(m_tr.group(2))
    found_square = False
    for poly in root.iter(f'{_SVG_NS}polygon'):
      if poly.get('data-psyre-shape') == 'fby-square':
        found_square = True
        for pair in poly.get('points', '').split():
          px, py = (float(c) for c in pair.split(','))
          self.assertGreaterEqual(px + gx, vx)
          self.assertLessEqual(px + gx, vx + vw)
          self.assertGreaterEqual(py + gy, vy)
          self.assertLessEqual(py + gy, vy + vh)
    self.assertTrue(found_square)

    # Actionable errors on missing dot binary or invalid SVG output.
    g_small = build_graph(lambda: counter(0))
    with self.assertRaisesRegex(FileNotFoundError, 'Install Graphviz'):
      visualize.render_svg(g_small, dot_prog='__nonexistent_dot_bin__')

    empty_proc = subprocess.CompletedProcess(
        ['dot'], 0, stdout=b'', stderr=b''
    )
    with mock.patch.object(
        visualize.subprocess, 'run', return_value=empty_proc
    ):
      with self.assertRaisesRegex(RuntimeError, 'missing <svg>'):
        visualize._run_dot_svg('digraph{}', 'dot')  # pylint: disable=protected-access

    # Non-zero exit code with valid SVG logs a warning instead of crashing.
    warn_proc = subprocess.CompletedProcess(
        ['dot'], 1, stdout=svg.encode('utf-8'), stderr=b'font warning'
    )
    with mock.patch.object(
        visualize.subprocess, 'run', return_value=warn_proc
    ):
      with self.assertLogs(logger, level='WARNING') as logs:
        out_svg = visualize._run_dot_svg('digraph{}', 'dot')  # pylint: disable=protected-access
      self.assertIn('<svg', out_svg)
      self.assertTrue(any('font warning' in m for m in logs.output))

    # Reject non-positive or non-int height_px.
    for bad_h, exc in ((0, ValueError), (-50, ValueError), (True, TypeError)):
      with self.assertRaises(exc):
        gui.SchematicBundle(
            views={'': gui.SchematicView(raw_svg=svg, dot_src='digraph{}')},
            height_px=bad_h,  # type: ignore[arg-type]
        )

    # Reject unknown expanded_automata prefix and materialize valid combos.
    def two_fsms():
      def mk(name):
        with automaton(init='A', name=name) as fsm:
          with state('A'):
            until(Stream.once(0), 'B')
            fsm.y = 1
          with state('B'):
            fsm.y = 2
        return fsm.y
      return tuple(mk(f'm{i}') for i in range(5))

    g_fsms = build_graph(two_fsms)
    with self.assertRaisesRegex(ValueError, 'Unknown expanded_automata'):
      visualize.build_pydot(g_fsms, expanded_automata=['typo'])
    with self.assertRaisesRegex(ValueError, 'Unknown expanded_automata'):
      visualize.visualize_graph(g_fsms, expanded_automata=['typo'])
    viz_pair = visualize.visualize_graph(
        g_fsms, expanded_automata=['m0', 'm2']
    )
    self.assertIn('⊟ m0', viz_pair.dot)
    self.assertIn('⊟ m2', viz_pair.dot)
    self.assertIn('⊞ m1', viz_pair.dot)

  def test_prefixes_pipe_keys_trapezoid_tips_cluster_paths_and_html_export(
      self,
  ):
    # 1. `_node_automaton_prefix` reads `node.prefixes`, not `node.name` '/'.
    def slash_and_unnamed_prog():
      x = Stream([1, 2, 3], name='x')
      top_slash = fby(0, x, name='foo/bar')
      with automaton(init='A', name='a|b') as fsm:
        with state('A'):
          until(Stream.once(0), 'B')
          fsm.y = fby(0, x, name='')
        with state('B'):
          fsm.y = 1
      return {'top': top_slash, 'y': fsm.y}

    g_pfx = build_graph(slash_and_unnamed_prog, optimize=False)
    dot_pfx = visualize.build_pydot(g_pfx, collapse_automata=True).to_string()
    self.assertNotIn('⊞ foo', dot_pfx)
    self.assertNotIn('⊟ foo', dot_pfx)
    self.assertIn('⊞ a|b', dot_pfx)
    # Unnamed `fby` inside `a|b` must be hidden inside the collapsed box.
    fby_in_fsm = [
        n for n in g_pfx.nodes if n.op == 'fby' and n.prefixes and not n.name
    ]
    self.assertLen(fby_in_fsm, 1)
    self.assertNotIn(f'n_{fby_in_fsm[0].id} [', dot_pfx)

    # 2. `'|'` in automaton name escapes to `%7C`, avoiding key collision with
    #    distinct `'a'` and `'b'` automata, and `save_html` includes UTF-8 meta.
    viz = visualize.visualize_graph(g_pfx, title='pipe_test')
    self.assertIn('a%7Cb', viz.html)
    self.assertIn('rawTarget.closest("g.edge")', viz.html)
    self.assertNotIn('<!DOCTYPE html>', viz._repr_html_())  # pylint: disable=protected-access
    saved_path = viz.save_html()
    with open(saved_path, encoding='utf-8') as f:
      saved_html = f.read()
    self.assertTrue(saved_html.startswith('<!DOCTYPE html>\n<meta charset='))

    # 3. LR `when`/`whennot` trapezoid selector arrow tip lands on the slanted
    #    top midpoint `0.5*(pts[0].y + pts[1].y)` (not `min(ys)`).
    def trap_prog():
      x = Stream([1, 2, 3], name='x')
      w = when(Stream([True, False], name='sel_w'), x)
      wn = when(Stream([True, False], name='sel_wn'), x, negated=True)
      return {'w': w, 'wn': wn}

    svg_trap = visualize.render_svg(build_graph(trap_prog), rankdir='LR')
    root_trap = ET.fromstring(svg_trap)
    nodes_by_key = {
        g.get('data-psyre-key', ''): g
        for g in root_trap.iter(f'{_SVG_NS}g')
        if g.get('class') == 'node'
    }
    checked_tips = 0
    for edge_g in root_trap.iter(f'{_SVG_NS}g'):
      if (
          edge_g.get('class') != 'edge'
          or edge_g.get('data-psyre-dst-port') != 'n'
      ):
        continue
      dst_g = nodes_by_key.get(edge_g.get('data-psyre-dst', ''))
      if dst_g is None or dst_g.get('data-psyre-kind') not in (
          'when',
          'whennot',
      ):
        continue
      dst_poly = dst_g.find(f'.//{_SVG_NS}polygon')
      edge_poly = edge_g.find(f'.//{_SVG_NS}polygon')
      self.assertIsNotNone(dst_poly)
      self.assertIsNotNone(edge_poly)
      dpts = [
          tuple(float(v) for v in p.split(','))
          for p in dst_poly.get('points', '').split()
      ]
      epts = [
          tuple(float(v) for v in p.split(','))
          for p in edge_poly.get('points', '').split()
      ]
      expected_top = 0.5 * (dpts[0][1] + dpts[1][1])
      if dst_g.get('data-psyre-kind') == 'whennot':
        expected_top -= 6.4
      # Arrow tip is the middle vertex (`epts[1]`) of the 4-point arrowhead.
      self.assertAlmostEqual(epts[1][1], expected_top, places=1)
      checked_tips += 1
    self.assertEqual(checked_tips, 2)

    # 4. Rounded `<path>` clusters widen leftward for long inline labels.
    synthetic_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
        '<g id="graph0" class="graph" transform="translate(0 100)">'
        '<polygon points="0,0 200,0 200,-100 0,-100"/>'
        '<g id="clust1" class="cluster">'
        '<path d="M52,-90 L188,-90 C194,-90 200,-84 200,-78 '
        'L200,-22 C200,-16 194,-10 188,-10 L52,-10 C46,-10 40,-16 40,-22 '
        'L40,-78 C40,-84 46,-90 52,-90"/>'
        '<g id="psyre_n_1" class="node"><title>n_1</title>'
        '<polygon points="50,-60 90,-60 90,-40 50,-40"/></g>'
        '<g id="psyre_c_1" class="node"><title>c_1</title>'
        '<text x="70" y="-75">very_long_inline_stream_label</text></g>'
        '<g id="psyre_e_1" class="edge"><title>c_1-&gt;n_1:n</title>'
        '<path d="M70,-70 L70,-60"/>'
        '<polygon points="67,-65 70,-60 73,-65 67,-65"/></g>'
        '</g></g></svg>'
    )
    post_root = ET.fromstring(
        gui.render_view_svg(
            gui.SchematicView(
                raw_svg=synthetic_svg,
                dot_src='digraph{}',
                static_edge_to_node={'psyre_e_1': 'psyre_c_1'},
            )
        )
    )
    coord_re = re.compile(r'(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)')
    post_min_xs = [
        min(float(x) for x, _ in coord_re.findall(p.get('d', '')))
        for g in post_root.iter(f'{_SVG_NS}g')
        if g.get('class') == 'cluster'
        and (p := g.find(f'{_SVG_NS}path')) is not None
    ]
    self.assertLen(post_min_xs, 1)
    self.assertLess(post_min_xs[0], 40.0)


if __name__ == '__main__':
  absltest.main()
