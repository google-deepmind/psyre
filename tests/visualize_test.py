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
    switch,
    when,
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


class VisualizeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def test_fby_feedback_uses_constrained_dir_back_and_preserves_svg_src_dst(
      self,
  ):
    def counter_prog():
      x = fby(0, lambda: x + 1, name='acc')
      return {'x': x}

    g = build_graph(counter_prog)
    dot_src = visualize.build_pydot(g).to_string()
    dashed_lines = [
        line.strip()
        for line in dot_src.splitlines()
        if '#7C3AED' in line and 'dashed' in line
    ]
    self.assertLen(dashed_lines, 1)
    self.assertIn('dir="back"', dashed_lines[0])
    self.assertIn('constraint=true', dashed_lines[0])

    for rankdir in ('LR', 'TB'):
      svg = visualize.render_svg(g, rankdir=rankdir)
      delayed = [
          e for e in _parse_edge_metadata(svg) if '.delayed_next' in e['info']
      ]
      self.assertLen(delayed, 1)
      fby_node = next(n for n in g.nodes if n.op == 'fby')
      add_node = next(n for n in g.nodes if n.op != 'fby')
      self.assertEqual(delayed[0]['src'], f'n_{add_node.id}')
      self.assertEqual(delayed[0]['dst'], f'n_{fby_node.id}')

  def test_pre_feedback_uses_constrained_dir_back(self):
    def pre_prog():
      p = pre(lambda: x, name='prev_x')
      x = ifthenelse(Stream.once(True), 0, p + 1)
      return {'x': x}

    g = build_graph(pre_prog)
    dot_src = visualize.build_pydot(g).to_string()
    dashed_lines = [
        line.strip()
        for line in dot_src.splitlines()
        if '#7C3AED' in line and 'dashed' in line
    ]
    self.assertLen(dashed_lines, 1)
    self.assertIn('dir="back"', dashed_lines[0])
    self.assertIn('constraint=true', dashed_lines[0])

    svg = visualize.render_svg(g)
    delayed = [
        e for e in _parse_edge_metadata(svg) if '.delayed_next' in e['info']
    ]
    self.assertLen(delayed, 1)
    pre_node = next(n for n in g.nodes if n.op == 'pre')
    self.assertEqual(delayed[0]['dst'], f'n_{pre_node.id}')

  def test_forward_fby_stays_forward_and_self_loop_unconstrained(self):
    def prog():
      inp = Stream([1, 2, 3, 4], name='inp')
      fwd = fby(0, inp + 1, name='fwd')
      self_x = fby(0, lambda: self_x, name='self_x')
      return {'fwd': fwd, 'self_x': self_x}

    g = build_graph(prog)
    dot_src = visualize.build_pydot(g).to_string()
    fwd_node = next(n for n in g.nodes if n.name == 'fwd')
    self_node = next(n for n in g.nodes if n.name == 'self_x')

    fwd_lines = [
        line.strip()
        for line in dot_src.splitlines()
        if f'psyre_e_{fwd_node.id}_p1' in line
    ]
    self.assertLen(fwd_lines, 1)
    self.assertNotIn('dir="back"', fwd_lines[0])
    self.assertIn('constraint=true', fwd_lines[0])
    self.assertIn('style="solid"', fwd_lines[0])
    self.assertIn('color="#334155"', fwd_lines[0])

    self_lines = [
        line.strip()
        for line in dot_src.splitlines()
        if f'psyre_e_{self_node.id}_p1' in line
    ]
    self.assertLen(self_lines, 1)
    self.assertNotIn('dir="back"', self_lines[0])
    self.assertIn('constraint=false', self_lines[0])
    self.assertIn('style="dashed"', self_lines[0])
    self.assertIn('color="#7C3AED"', self_lines[0])

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
    dot_src = visualize.build_pydot(g).to_string()
    dashed_lines = [
        line.strip()
        for line in dot_src.splitlines()
        if '#7C3AED' in line and 'dashed' in line
    ]
    self.assertLen(dashed_lines, 2)
    for line in dashed_lines:
      self.assertIn('dir="back"', line)
      self.assertIn('constraint=true', line)

    html_out = visualize.render_html(g, title='test_fsm')
    self.assertIn('<svg', html_out)

  def test_mutual_fby_cycle_renders_without_dot_cycles(self):
    def mutual_prog():
      x = fby(0, lambda: y + 1, name='x')
      y = fby(10, lambda: x + 2, name='y')
      return {'x': x, 'y': y}

    g = build_graph(mutual_prog)
    svg = visualize.render_svg(g)
    delayed = [
        e for e in _parse_edge_metadata(svg) if '.delayed_next' in e['info']
    ]
    self.assertLen(delayed, 2)
    fby_keys = {f'n_{n.id}' for n in g.nodes if n.op == 'fby'}
    self.assertEqual({e['dst'] for e in delayed}, fby_keys)

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

  def test_collapsed_automaton_and_visualize_graph_bundle(self):
    def prog():
      a = Stream([False, True, False], name='a')
      r = Stream([False, False, True], name='r')
      with automaton(init='WAIT', name='await_a') as fsm:
        with state('WAIT', unless(r, 'WAIT')):
          fsm.y = a
          until(a, 'DONE')
        with state('DONE', unless(r, 'WAIT')):
          fsm.y = True
      return {'y': fsm.y}

    g = build_graph(prog)
    collapsed_dot = visualize.build_pydot(g, collapse_automata=True).to_string()
    self.assertIn('fsm_box_0', collapsed_dot)
    self.assertIn('⊞ await_a', collapsed_dot)
    viz = visualize.visualize_graph(g, title='abro_box')
    self.assertIn('<svg', viz.svg)
    self.assertIn('psyre-viz-card', viz.html)
    self.assertIn('fsm_box_0', viz.dot)

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

  def test_inline_static_and_once_stream_nodes_render_without_background_boxes(
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
      if el.get('class') == 'node' and el.get('data-psyre-kind') in (
          'static_stream',
          'once_stream',
      ):
        checked += 1
        self.assertIsNone(el.find(f'{_SVG_NS}path'))
        self.assertIsNone(el.find(f'{_SVG_NS}polygon'))
    self.assertGreater(checked, 0)

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


if __name__ == '__main__':
  absltest.main()
