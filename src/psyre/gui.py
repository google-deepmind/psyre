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
"""SVG post-processing and interactive HTML/UX viewer for Psyre schematics."""

import atexit
import base64
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import gzip
import html
import json
import math
import os
import re
import tempfile
from typing import Any, Literal
import uuid
import xml.etree.ElementTree as ET

_LAZY_GZIP_THRESHOLD_BYTES = 256 * 1024


def _safe_inline_json(value: Any) -> str:
  """Serializes `value` as JSON escaped for inline `<script>` embedding."""
  return (
      json.dumps(value, separators=(',', ':'))
      .replace('<', '\\u003c')
      .replace('>', '\\u003e')
      .replace('&', '\\u0026')
  )

RankDir = Literal['LR', 'TB']

NodeKind = Literal[
    'when',
    'whennot',
    'merge',
    'fby',
    'post',
    'ifthenelse',
    'sigma',
    'fifo',
    'counter',
    'static_stream',
    'other',
]

# Schematic color palette (fill, border, text).
STYLE_SELECTOR = ('#FEF3C7', '#D97706', '#78350F')
STYLE_MUX = ('#CCFBF1', '#0D9488', '#134E4A')
STYLE_FBY = ('#EDE9FE', '#6D28D9', '#3B0764')
STYLE_STATEFUL = ('#F3E8FF', '#7C3AED', '#3B0764')
STYLE_COMB = ('#F8FAFC', '#475569', '#0F172A')
STYLE_INPUT = ('#ECFDF5', '#059669', '#065F46')
STYLE_CONST = ('#F1F5F9', '#94A3B8', '#334155')
STYLE_NET_REF = ('#F0F9FF', '#0284C7', '#0C4A6E')
STYLE_RST_REF = ('#FFF1F2', '#E11D48', '#881337')
STYLE_OUTPUT = ('#EFF6FF', '#2563EB', '#1E3A8A')
STYLE_CYCLE = ('#FEE2E2', '#DC2626', '#7F1D1D')

_SVG_NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', _SVG_NS)

_PATH_COORD_RE = re.compile(r'(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)')
_TRANSLATE_RE = re.compile(
    r'translate\(\s*(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)\s*\)'
)
_SVG_ANCHOR_RE = re.compile(r'<(?:[A-Za-z_][\w.-]*:)?a\b')


def validate_height_px(height_px: int) -> int:
  """Validates interactive viewer stage height as a positive integer."""
  if isinstance(height_px, bool) or not isinstance(height_px, int):
    raise TypeError(
        f'height_px must be a positive int, got {type(height_px).__name__}.'
    )
  if height_px <= 0:
    raise ValueError(f'height_px must be > 0, got {height_px}.')
  return height_px

_TEMP_EXPORT_DIR: tempfile.TemporaryDirectory[str] | None = None


@dataclasses.dataclass(frozen=True)
class NodeShapeSpec:
  """Metadata driving SVG post-processing for custom node schematic shapes."""

  kind: NodeKind
  border_color: str
  subscript: str = ''


@dataclasses.dataclass(frozen=True)
class SchematicView:
  """Raw Graphviz output and DOM metadata for a single expansion view."""

  raw_svg: str
  dot_src: str
  rankdir: RankDir = 'LR'
  shape_specs: Mapping[str, NodeShapeSpec] = dataclasses.field(
      default_factory=dict
  )
  info_by_dom_id: Mapping[str, str] = dataclasses.field(default_factory=dict)
  net_by_dom_id: Mapping[str, str] = dataclasses.field(default_factory=dict)
  fsm_box_by_dom_id: Mapping[str, str] = dataclasses.field(default_factory=dict)
  fsm_cluster_by_dom_id: Mapping[str, str] = dataclasses.field(
      default_factory=dict
  )
  static_edge_to_node: Mapping[str, str] = dataclasses.field(
      default_factory=dict
  )
  reversed_edge_ids: frozenset[str] = frozenset()


@dataclasses.dataclass(frozen=True)
class SchematicBundle:
  """Complete multi-view schematic payload communicated from visualize to gui."""

  views: Mapping[str, SchematicView]
  initial_view_key: str = ''
  fsm_prefixes: Sequence[str] = ()
  title: str = ''
  semantics_label: str = 'Psyre'
  node_count: int = 0
  height_px: int = 560

  def __post_init__(self) -> None:
    validate_height_px(self.height_px)


def _parse_polygon_points(points_attr: str) -> list[tuple[float, float]]:
  """Parses an SVG polygon `points` attribute into `(x, y)` float pairs."""
  pts: list[tuple[float, float]] = []
  for pair in points_attr.strip().split():
    x_str, y_str = pair.split(',', 1)
    pts.append((float(x_str), float(y_str)))
  return pts


def _points_bounds(
    pts: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float]:
  """Returns `(min_x, min_y, max_x, max_y)` for `(x, y)` pairs."""
  xs = [pt[0] for pt in pts]
  ys = [pt[1] for pt in pts]
  return min(xs), min(ys), max(xs), max(ys)


def _parse_polygon_bounds(
    points_attr: str,
) -> tuple[float, float, float, float]:
  """Extracts `(min_x, min_y, max_x, max_y)` from an SVG polygon points list."""
  return _points_bounds(_parse_polygon_points(points_attr))


def _format_pts(pts: Iterable[tuple[float, float]]) -> str:
  """Formats `(x, y)` pairs for an SVG `points` attribute."""
  return ' '.join(f'{x:.2f},{y:.2f}' for x, y in pts)


def _rect_pts(x0: float, y0: float, x1: float, y1: float) -> str:
  """Formats axis-aligned rectangle corners for an SVG `points` attribute."""
  return _format_pts([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _rounded_rect_path(
    x0: float, y0: float, x1: float, y1: float, radius: float = 12.0
) -> str:
  """Formats a rounded axis-aligned rectangle as an SVG `<path d>` string."""
  r = max(0.0, min(radius, 0.5 * (x1 - x0), 0.5 * (y1 - y0)))
  return (
      f'M{x0 + r:.2f},{y0:.2f} '
      f'L{x1 - r:.2f},{y0:.2f} '
      f'Q{x1:.2f},{y0:.2f} {x1:.2f},{y0 + r:.2f} '
      f'L{x1:.2f},{y1 - r:.2f} '
      f'Q{x1:.2f},{y1:.2f} {x1 - r:.2f},{y1:.2f} '
      f'L{x0 + r:.2f},{y1:.2f} '
      f'Q{x0:.2f},{y1:.2f} {x0:.2f},{y1 - r:.2f} '
      f'L{x0:.2f},{y0 + r:.2f} '
      f'Q{x0:.2f},{y0:.2f} {x0 + r:.2f},{y0:.2f} Z'
  )


def _svg_el(
    tag: str, attrs: Mapping[str, str], text: str | None = None
) -> ET.Element:
  """Creates an SVG-namespaced `ET.Element` with optional text content."""
  el = ET.Element(f'{{{_SVG_NS}}}{tag}', dict(attrs))
  if text is not None:
    el.text = text
  return el


def _append_subscript_tspan(text_el: ET.Element, subscript: str) -> None:
  """Replaces `text_el` children with its base text plus a subscript tspan."""
  base_text = ''.join(text_el.itertext())
  for child in list(text_el):
    text_el.remove(child)
  text_el.text = base_text
  sub_span = ET.SubElement(
      text_el,
      f'{{{_SVG_NS}}}tspan',
      {'dy': '0.25em', 'font-size': '0.72em'},
  )
  sub_span.text = subscript


def _apply_custom_glyph(
    group: ET.Element, spec: NodeShapeSpec, rankdir: RankDir
) -> None:
  """Transforms a Graphviz SVG `<g class="node">` into its schematic glyph."""
  if spec.kind == 'static_stream':
    drop_tags = {f'{{{_SVG_NS}}}polygon', f'{{{_SVG_NS}}}path'}
    for parent in group.iter():
      for child in list(parent):
        if child.tag in drop_tags:
          parent.remove(child)
    return

  if spec.kind == 'other':
    if spec.subscript:
      for text_el in group.findall(f'.//{{{_SVG_NS}}}text'):
        _append_subscript_tspan(text_el, spec.subscript)
    return

  poly_tag = f'{{{_SVG_NS}}}polygon'
  found = next(
      (
          (p, child)
          for p in group.iter()
          if (child := p.find(poly_tag)) is not None
      ),
      None,
  )
  if found is None:
    return
  poly_parent, poly = found
  orig_pts = _parse_polygon_points(poly.get('points', ''))
  if not orig_pts:
    return
  x0, y0, x1, y1 = _points_bounds(orig_pts)
  cx = 0.5 * (x0 + x1)
  cy = 0.5 * (y0 + y1)
  width = x1 - x0
  height = y1 - y0
  poly_index = list(poly_parent).index(poly)

  if spec.kind in ('fby', 'post'):
    # Expand register/lookahead boxes into squares and draw a directional
    # chevron hat: downward `v` inside the top edge for unit delay (`fby`/
    # `pre`) and upward `^` inside the bottom edge for lookahead (`post`).
    side = max(width, height)
    sx0, sx1 = cx - 0.5 * side, cx + 0.5 * side
    sy0, sy1 = cy - 0.5 * side, cy + 0.5 * side
    poly.set('points', _rect_pts(sx0, sy0, sx1, sy1))
    is_post = spec.kind == 'post'
    poly.set('data-psyre-shape', 'post-square' if is_post else 'fby-square')
    hat_size = min(9.0, 0.21 * side)
    hat_y, tip_y, text_dy = (
        (sy1, sy1 - hat_size, -0.35 * hat_size)
        if is_post
        else (sy0, sy0 + hat_size, 0.35 * hat_size)
    )
    hat = _svg_el(
        'polyline',
        {
            'class': 'psyre-post-hat' if is_post else 'psyre-fby-hat',
            'points': _format_pts([
                (cx - hat_size, hat_y),
                (cx, tip_y),
                (cx + hat_size, hat_y),
            ]),
            'fill': 'none',
            'stroke': spec.border_color,
            'stroke-width': '1.6',
            'stroke-linejoin': 'miter',
            'stroke-linecap': 'round',
        },
    )
    poly_parent.insert(poly_index + 1, hat)
    for text_el in group.findall(f'.//{{{_SVG_NS}}}text'):
      old_y = float(text_el.get('y', f'{cy:.2f}'))
      text_el.set('y', f'{old_y + text_dy:.2f}')
      if spec.subscript:
        _append_subscript_tspan(text_el, spec.subscript)
    return

  if spec.kind in ('when', 'whennot', 'merge'):
    if rankdir == 'LR':
      # In LR mode draw opposing trapezoids matching TB `trapezium` /
      # `invtrapezium`: `when` flares wider left-to-right (clock sampler), while
      # `merge` narrows left-to-right (fan-in multiplexer).
      slope = 0.18 * height
      s = -slope if spec.kind == 'merge' else slope
      poly.set(
          'points',
          _format_pts([
              (x0, y0 + s),
              (x1, y0 - s),
              (x1, y1 + s),
              (x0, y1 - s),
          ]),
      )
    poly.set(
        'data-psyre-shape',
        'mux-trapezoid' if spec.kind == 'merge' else 'selector-trapezoid',
    )
    if spec.kind == 'whennot':
      # Active-low inversion bubble placed on the selector entry edge (`n` in
      # LR, left slanted wall in TB).
      if rankdir == 'LR':
        bx, by = cx, y0 - 3.2
      else:
        left_xs = sorted(px for px, _ in set(orig_pts))[:2]
        left_mid_x = 0.5 * sum(left_xs) if left_xs else x0
        bx, by = left_mid_x - 3.2, cy
      poly_parent.insert(
          poly_index + 1,
          _svg_el(
              'circle',
              {
                  'class': 'psyre-whennot-bubble',
                  'cx': f'{bx:.2f}',
                  'cy': f'{by:.2f}',
                  'r': '3.2',
                  'fill': '#FFFFFF',
                  'stroke': spec.border_color,
                  'stroke-width': '1.5',
              },
          ),
      )
    return

  if spec.kind == 'ifthenelse':
    poly.set(
        'points', _format_pts([(cx, y0), (x1, cy), (cx, y1), (x0, cy)])
    )
    poly.set('data-psyre-shape', 'ifthenelse-diamond')
    return

  if spec.kind == 'sigma':
    poly.set('data-psyre-shape', 'sigma-box')
    return

  if spec.kind not in ('fifo', 'counter'):
    return
  for text_el in list(poly_parent.findall(f'{{{_SVG_NS}}}text')):
    poly_parent.remove(text_el)

  if spec.kind == 'fifo':
    # Open-ended shift-register frame (3 inner slot dividers, no leading bar).
    poly.set('data-psyre-shape', 'fifo-buffer')
    poly.set('stroke', 'none')
    if rankdir == 'TB':
      frame_pts = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
      dividers = [
          (x0, y0 + frac * height, x1, y0 + frac * height)
          for frac in (0.25, 0.50, 0.75)
      ]
    else:
      frame_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
      dividers = [
          (x0 + frac * width, y0, x0 + frac * width, y1)
          for frac in (0.25, 0.50, 0.75)
      ]
    elems = [
        _svg_el(
            'polyline',
            {
                'class': 'psyre-fifo-frame',
                'points': _format_pts(frame_pts),
                'fill': 'none',
                'stroke': spec.border_color,
                'stroke-width': '1.6',
                'stroke-linejoin': 'miter',
                'stroke-linecap': 'butt',
            },
        ),
        *(
            _svg_el(
                'line',
                {
                    'class': 'psyre-fifo-divider',
                    'x1': f'{lx0:.2f}',
                    'y1': f'{ly0:.2f}',
                    'x2': f'{lx1:.2f}',
                    'y2': f'{ly1:.2f}',
                    'stroke': spec.border_color,
                    'stroke-width': '1.4',
                },
            )
            for lx0, ly0, lx1, ly1 in dividers
        ),
    ]
  else:  # counter: 3-digit odometer housing replacing placeholder '042' label.
    fill_color = poly.get('fill', STYLE_STATEFUL[0])
    poly.set('data-psyre-shape', 'counter-odometer')
    poly.set('stroke', 'none')
    poly.set('fill', 'none')
    pad_x, pad_y = 4.2, 3.6
    wx0, wy0 = x0 + pad_x, y0 + pad_y
    win_w = max(12.0, width - 2.0 * pad_x)
    win_h = max(10.0, height - 2.0 * pad_y)
    slot_w = win_w / 3.0
    rx0, rx1 = wx0 + 2.0 * slot_w, wx0 + win_w
    elems = [
        _svg_el(
            'rect',
            {
                'class': 'psyre-counter-housing',
                'x': f'{x0:.2f}',
                'y': f'{y0:.2f}',
                'width': f'{width:.2f}',
                'height': f'{height:.2f}',
                'rx': '4.0',
                'ry': '4.0',
                'fill': fill_color,
                'stroke': spec.border_color,
                'stroke-width': '1.6',
            },
        ),
        _svg_el(
            'rect',
            {
                'class': 'psyre-counter-window',
                'x': f'{wx0:.2f}',
                'y': f'{wy0:.2f}',
                'width': f'{win_w:.2f}',
                'height': f'{win_h:.2f}',
                'rx': '2.0',
                'ry': '2.0',
                'fill': '#FFFFFF',
                'stroke': spec.border_color,
                'stroke-width': '1.2',
            },
        ),
        *(
            _svg_el(
                'line',
                {
                    'class': 'psyre-counter-divider',
                    'x1': f'{wx0 + div_idx * slot_w:.2f}',
                    'y1': f'{wy0:.2f}',
                    'x2': f'{wx0 + div_idx * slot_w:.2f}',
                    'y2': f'{wy0 + win_h:.2f}',
                    'stroke': spec.border_color,
                    'stroke-width': '1.1',
                },
            )
            for div_idx in (1, 2)
        ),
        *(
            _svg_el(
                'line',
                {
                    'class': 'psyre-counter-roll',
                    'x1': f'{rx0 + 2.0:.2f}',
                    'y1': f'{ry:.2f}',
                    'x2': f'{rx1 - 2.0:.2f}',
                    'y2': f'{ry:.2f}',
                    'stroke': spec.border_color,
                    'stroke-width': '0.9',
                    'stroke-opacity': '0.55',
                },
            )
            for ry in (wy0 + 2.8, wy0 + win_h - 2.8)
        ),
        *(
            _svg_el(
                'text',
                {
                    'class': 'psyre-counter-digit',
                    'x': f'{wx0 + (slot_idx + 0.5) * slot_w:.2f}',
                    'y': f'{cy + (2.1 if slot_idx == 2 else 3.6):.2f}',
                    'text-anchor': 'middle',
                    'font-family': 'Helvetica, monospace',
                    'font-size': '10.5',
                    'font-weight': '700',
                    'fill': STYLE_STATEFUL[2],
                },
                text=digit,
            )
            for slot_idx, digit in enumerate(('0', '4', '2'))
        ),
    ]
  for offset, el in enumerate(elems, start=1):
    poly_parent.insert(poly_index + offset, el)


def _shorten_static_stream_edges(
    static_edge_to_node: Mapping[str, str],
    group_by_id: Mapping[str, ET.Element],
    node_by_key: Mapping[str, ET.Element],
    cluster_shapes: Sequence[tuple[ET.Element, list[float]]],
) -> None:
  """Shortens inline stream arrows and places their label at the arrow tail."""
  for e_dom, c_dom in static_edge_to_node.items():
    edge_g = group_by_id.get(e_dom)
    node_g = group_by_id.get(c_dom)
    if edge_g is None or node_g is None:
      continue
    path_el = edge_g.find(f'.//{{{_SVG_NS}}}path')
    poly_el = edge_g.find(f'.//{{{_SVG_NS}}}polygon')
    text_el = node_g.find(f'.//{{{_SVG_NS}}}text')
    if path_el is None or poly_el is None or text_el is None:
      continue
    pts = [
        (float(m.group(1)), float(m.group(2)))
        for m in _PATH_COORD_RE.finditer(path_el.get('d', ''))
    ]
    poly_pts = _parse_polygon_points(poly_el.get('points', ''))
    if len(pts) < 2 or len(poly_pts) < 3:
      continue
    x_start, y_start = pts[0]
    x_end, y_end = pts[-1]
    tip_x, tip_y = max(
        poly_pts,
        key=lambda pt, xe=x_end, ye=y_end: math.hypot(pt[0] - xe, pt[1] - ye),
    )
    dst_port = edge_g.get('data-psyre-dst-port', '')
    if dst_port == 'n':
      dst_g = node_by_key.get(edge_g.get('data-psyre-dst', ''))
      if dst_g is not None:
        dst_poly = dst_g.find(f'.//{{{_SVG_NS}}}polygon')
        if dst_poly is not None and (dst_pts := dst_poly.get('points')):
          dst_pts_list = _parse_polygon_points(dst_pts)
          dx0, dy0, dx1, _ = _points_bounds(dst_pts_list)
          tip_x = 0.5 * (dx0 + dx1)
          # LR `when`/`whennot`/`merge` polygons have a slanted top edge
          # connecting `pts[0]` and `pts[1]`, whose midpoint is `0.5*(y0+y1)`
          # rather than `min(ys)`.
          if (
              dst_poly.get('data-psyre-shape')
              in ('selector-trapezoid', 'mux-trapezoid')
              and len(dst_pts_list) == 4
          ):
            top_y = 0.5 * (dst_pts_list[0][1] + dst_pts_list[1][1])
          else:
            top_y = dy0
          tip_y = (
              (top_y - 6.4)
              if dst_g.get('data-psyre-kind') == 'whennot'
              else top_y
          )
      ux, uy = 0.0, 1.0
      arrow_len = 5.5
      shaft_len = 8.5
    elif dst_port == 'w' or dst_port.endswith(':w'):
      ux, uy = 1.0, 0.0
      arrow_len = 5.5
      shaft_len = 11.0
    else:
      dx, dy = tip_x - x_end, tip_y - y_end
      tip_dist = math.hypot(dx, dy)
      dist = tip_dist
      if dist < 1e-3:
        dx, dy = tip_x - x_start, tip_y - y_start
        dist = math.hypot(dx, dy)
      if dist < 1e-3:
        continue
      ux, uy = dx / dist, dy / dist
      if abs(ux) >= 0.85:
        ux, uy = math.copysign(1.0, ux), 0.0
      elif abs(uy) >= 0.85:
        ux, uy = 0.0, math.copysign(1.0, uy)
      arrow_len = max(5.5, min(8.0, tip_dist))
      shaft_len = 14.0

    arrow_half_w = 0.36 * arrow_len
    base_x = tip_x - ux * arrow_len
    base_y = tip_y - uy * arrow_len
    start_x = base_x - ux * shaft_len
    start_y = base_y - uy * shaft_len
    path_el.set(
        'd', f'M{start_x:.2f},{start_y:.2f} L{base_x:.2f},{base_y:.2f}'
    )
    left_wing = (base_x - uy * arrow_half_w, base_y + ux * arrow_half_w)
    right_wing = (base_x + uy * arrow_half_w, base_y - ux * arrow_half_w)
    poly_el.set(
        'points',
        _format_pts([left_wing, (tip_x, tip_y), right_wing, left_wing]),
    )
    if dst_port == 'n':
      text_el.set('text-anchor', 'end')
      text_el.set('x', f'{tip_x - 3.8:.2f}')
      text_el.set('y', f'{tip_y - 7.0:.2f}')
    elif abs(ux) >= abs(uy):
      text_el.set('text-anchor', 'end' if ux > 0 else 'start')
      text_el.set(
          'x', f'{start_x - 3.5:.2f}' if ux > 0 else f'{start_x + 3.5:.2f}'
      )
      text_el.set('y', f'{start_y + 3.8:.2f}')
    else:
      text_el.set('text-anchor', 'middle')
      text_el.set('x', f'{start_x:.2f}')
      text_el.set(
          'y', f'{start_y - 4.0:.2f}' if uy > 0 else f'{start_y + 11.0:.2f}'
      )
    if text_el.get('text-anchor') == 'end':
      # Push any enclosing cluster rectangle leftward so tail labels attached
      # to shortened inline arrows never spill outside the cluster border.
      txt_str = ''.join(text_el.itertext()).strip()
      est_w = max(12.0, 6.8 * len(txt_str))
      left_edge = float(text_el.get('x', '0')) - est_w - 6.0
      for c_shape, bounds in cluster_shapes:
        bx0, by0, bx1, by1 = bounds
        if (
            bx0 - 2.0 <= tip_x <= bx1 + 2.0
            and by0 - 2.0 <= tip_y <= by1 + 2.0
            and left_edge < bx0
        ):
          bounds[0] = left_edge
          if c_shape.tag == f'{{{_SVG_NS}}}path':
            c_shape.set('d', _rounded_rect_path(left_edge, by0, bx1, by1))
          else:
            c_shape.set('points', _rect_pts(left_edge, by0, bx1, by1))
    for edge_text_el in edge_g.findall(f'{{{_SVG_NS}}}text'):
      edge_text_el.set('x', f'{0.5 * (start_x + base_x) - uy * 6.5:.2f}')
      edge_text_el.set(
          'y',
          f'{0.5 * (start_y + base_y) - abs(ux) * 3.2 + abs(uy) * 3.0:.2f}',
      )


def _expand_svg_viewbox(root: ET.Element) -> None:
  """Adds a safety margin around `<svg viewBox>` covering all glyphs/labels."""
  margin_pt = 10.0
  view_box = root.get('viewBox')
  if not view_box:
    return
  parts = [float(x) for x in view_box.split()]
  if len(parts) != 4:
    return
  vx, vy, vw, vh = parts
  min_x, min_y, max_x, max_y = vx, vy, vx + vw, vy + vh

  def include(x0: float, y0: float, x1: float, y1: float) -> None:
    nonlocal min_x, min_y, max_x, max_y
    min_x = min(min_x, x0)
    min_y = min(min_y, y0)
    max_x = max(max_x, x1)
    max_y = max(max_y, y1)

  # Graphviz places all nodes/clusters/edges inside a root `<g>` whose
  # `translate(gx gy)` maps negative-y local points into `viewBox` space.
  graph_group = root.find(f'{{{_SVG_NS}}}g')
  gx, gy = 0.0, 0.0
  bg_poly: ET.Element | None = None
  if graph_group is not None:
    m_tr = _TRANSLATE_RE.search(graph_group.get('transform', ''))
    if m_tr:
      gx, gy = float(m_tr.group(1)), float(m_tr.group(2))
    bg_poly = graph_group.find(f'{{{_SVG_NS}}}polygon')
  # Include custom-substituted node/cluster shapes (`fby`/`post` squares,
  # LR trapezoids, `whennot` inversion bubbles, widened cluster borders) so
  # standalone SVGs never clip expanded glyphs beyond Graphviz's initial box.
  stroke_pad = 2.0
  for poly_tag in (f'{{{_SVG_NS}}}polygon', f'{{{_SVG_NS}}}polyline'):
    for poly_el in root.iter(poly_tag):
      if poly_el is bg_poly:
        continue
      pts_attr = poly_el.get('points', '')
      if not pts_attr:
        continue
      pts = _parse_polygon_points(pts_attr)
      if not pts:
        continue
      px0, py0, px1, py1 = _points_bounds(pts)
      include(
          px0 + gx - stroke_pad,
          py0 + gy - stroke_pad,
          px1 + gx + stroke_pad,
          py1 + gy + stroke_pad,
      )
  for cluster_g in root.iter(f'{{{_SVG_NS}}}g'):
    if cluster_g.get('class') != 'cluster':
      continue
    c_path = cluster_g.find(f'{{{_SVG_NS}}}path')
    if c_path is None or not (d_attr := c_path.get('d', '')):
      continue
    path_pts = [
        (float(m.group(1)), float(m.group(2)))
        for m in _PATH_COORD_RE.finditer(d_attr)
    ]
    if path_pts:
      px0, py0, px1, py1 = _points_bounds(path_pts)
      include(
          px0 + gx - stroke_pad,
          py0 + gy - stroke_pad,
          px1 + gx + stroke_pad,
          py1 + gy + stroke_pad,
      )
  for circle_el in root.iter(f'{{{_SVG_NS}}}circle'):
    cx_attr = circle_el.get('cx')
    cy_attr = circle_el.get('cy')
    r_attr = circle_el.get('r')
    if cx_attr is None or cy_attr is None or r_attr is None:
      continue
    cx_v = float(cx_attr) + gx
    cy_v = float(cy_attr) + gy
    r_v = float(r_attr) + stroke_pad
    include(cx_v - r_v, cy_v - r_v, cx_v + r_v, cy_v + r_v)
  for rect_el in root.iter(f'{{{_SVG_NS}}}rect'):
    rx_attr = rect_el.get('x')
    ry_attr = rect_el.get('y')
    rw_attr = rect_el.get('width')
    rh_attr = rect_el.get('height')
    if rx_attr is None or ry_attr is None or rw_attr is None or rh_attr is None:
      continue
    rx0 = float(rx_attr) + gx
    ry0 = float(ry_attr) + gy
    include(
        rx0 - stroke_pad,
        ry0 - stroke_pad,
        rx0 + float(rw_attr) + stroke_pad,
        ry0 + float(rh_attr) + stroke_pad,
    )
  for text_el in root.iter(f'{{{_SVG_NS}}}text'):
    x_attr = text_el.get('x')
    y_attr = text_el.get('y')
    if not x_attr or not y_attr:
      continue
    tx = float(x_attr) + gx
    ty = float(y_attr) + gy
    txt = ''.join(text_el.itertext()).strip()
    est_w = max(10.0, 6.5 * len(txt))
    anchor = text_el.get('text-anchor', 'middle')
    if anchor == 'end':
      tx0, tx1 = tx - est_w, tx
    elif anchor == 'start':
      tx0, tx1 = tx, tx + est_w
    else:
      tx0, tx1 = tx - 0.5 * est_w, tx + 0.5 * est_w
    include(tx0, ty - 14.0, tx1, ty + 6.0)
  new_x = min_x - margin_pt
  new_y = min_y - margin_pt
  new_w = (max_x - min_x) + 2.0 * margin_pt
  new_h = (max_y - min_y) + 2.0 * margin_pt
  root.set('viewBox', f'{new_x:.2f} {new_y:.2f} {new_w:.2f} {new_h:.2f}')
  root.set('width', f'{new_w:.0f}pt')
  root.set('height', f'{new_h:.0f}pt')
  if bg_poly is not None and (bg_pts := bg_poly.get('points')):
    bx0, by0, bx1, by1 = _parse_polygon_bounds(bg_pts)
    dx_left = max(margin_pt, vx - min_x + margin_pt)
    dx_right = max(margin_pt, max_x - (vx + vw) + margin_pt)
    dy_top = max(margin_pt, vy - min_y + margin_pt)
    dy_bot = max(margin_pt, max_y - (vy + vh) + margin_pt)
    bg_poly.set(
        'points',
        _rect_pts(bx0 - dx_left, by0 - dy_top, bx1 + dx_right, by1 + dy_bot),
    )


def _unwrap_svg_anchors(root: ET.Element) -> None:
  """Unwraps any `<svg:a>` elements so Colab's HTML renderer never crashes."""
  anchor_tags = {f'{{{_SVG_NS}}}a', 'a'}
  for parent in root.iter():
    idx = 0
    while idx < len(parent):
      child = parent[idx]
      if child.tag in anchor_tags:
        grandchildren = list(child)
        parent.remove(child)
        for offset, grandchild in enumerate(grandchildren):
          parent.insert(idx + offset, grandchild)
        idx += len(grandchildren)
      else:
        idx += 1


def render_view_svg(view: SchematicView) -> str:
  """Post-processes a `SchematicView` into custom schematic SVG markup."""
  root = ET.fromstring(view.raw_svg)
  # Strip any Graphviz `<svg:a>` link wrappers emitted from tooltip/URL attrs.
  if _SVG_ANCHOR_RE.search(view.raw_svg):
    _unwrap_svg_anchors(root)
  collect_static = bool(view.static_edge_to_node)
  group_by_id: dict[str, ET.Element] = {}
  node_by_key: dict[str, ET.Element] = {}
  cluster_shapes: list[tuple[ET.Element, list[float]]] = []
  for group in root.iter(f'{{{_SVG_NS}}}g'):
    dom_id = group.get('id', '')
    group_cls = group.get('class', '')
    if collect_static:
      if dom_id:
        group_by_id[dom_id] = group
      if group_cls == 'cluster':
        c_poly = group.find(f'{{{_SVG_NS}}}polygon')
        if c_poly is not None and (pts_attr := c_poly.get('points', '')):
          cluster_shapes.append(
              (c_poly, list(_parse_polygon_bounds(pts_attr)))
          )
        else:
          c_path = group.find(f'{{{_SVG_NS}}}path')
          if c_path is not None and (d_attr := c_path.get('d', '')):
            path_pts = [
                (float(m.group(1)), float(m.group(2)))
                for m in _PATH_COORD_RE.finditer(d_attr)
            ]
            if path_pts:
              cluster_shapes.append((c_path, list(_points_bounds(path_pts))))
    title_el = group.find(f'{{{_SVG_NS}}}title')
    raw_key = (title_el.text or '').strip() if title_el is not None else ''
    if group_cls == 'node' and raw_key:
      group.set('data-psyre-key', raw_key)
      if collect_static:
        node_by_key[raw_key] = group
    elif group_cls == 'edge' and '->' in raw_key:
      src_raw, dst_raw = raw_key.split('->', 1)
      if dom_id in view.reversed_edge_ids:
        src_raw, dst_raw = dst_raw, src_raw
      group.set('data-psyre-src', src_raw.split(':', 1)[0].strip())
      group.set('data-psyre-dst', dst_raw.split(':', 1)[0].strip())
      if ':' in dst_raw:
        group.set('data-psyre-dst-port', dst_raw.split(':', 1)[1].strip())
    fsm_box = view.fsm_box_by_dom_id.get(dom_id)
    if fsm_box is not None:
      group.set('data-psyre-fsm-box', fsm_box)
    fsm_cluster = view.fsm_cluster_by_dom_id.get(dom_id)
    if fsm_cluster is not None:
      group.set('data-psyre-fsm-cluster', fsm_cluster)
    net_key = view.net_by_dom_id.get(dom_id)
    if net_key is not None:
      group.set('data-psyre-net', net_key)
    info = view.info_by_dom_id.get(dom_id)
    if info is not None:
      group.set('data-psyre-info', info)
      if title_el is not None:
        title_el.text = info
    spec = view.shape_specs.get(dom_id)
    if spec is not None:
      group.set('data-psyre-kind', spec.kind)
      _apply_custom_glyph(group, spec, view.rankdir)
  if collect_static:
    _shorten_static_stream_edges(
        view.static_edge_to_node, group_by_id, node_by_key, cluster_shapes
    )
  _expand_svg_viewbox(root)
  return ET.tostring(root, encoding='unicode')


def _wrap_interactive_html(
    svg_views: Mapping[str, str],
    *,
    title: str,
    semantics_label: str,
    node_count: int,
    height_px: int,
    initial_view_key: str,
    fsm_prefixes: Sequence[str],
) -> str:
  """Wraps schematic SVG views in a zoom/pan/expand/inspect HTML widget."""
  uid = f'psyre_viz_{uuid.uuid4().hex[:8]}'
  header_title = html.escape(title or 'Psyre Stage-1 Primitive Graph')
  sem_badge = html.escape(semantics_label)
  font_stack = (
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  )
  has_fsm = bool(fsm_prefixes)
  if has_fsm:
    default_hint = (
        'Pinch or use left slider to zoom · drag to pan · click ⊞/⊟ automaton '
        'box to expand/collapse · hover or click a node/edge to inspect'
    )
    fsm_buttons_html = """
      <button type="button" class="psyre-viz-btn"
        data-act="expand-all" title="Expand all automata">⊞ Expand All</button>
      <button type="button" class="psyre-viz-btn"
        data-act="collapse-all" title="Collapse all automata">⊟ Collapse All</button>"""
    fsm_legend_html = """
      <span><i class="psyre-swatch"
        style="background:#E0E7FF;border:1px solid #4338CA"></i>automaton (⊞/⊟)</span>"""
  else:
    default_hint = (
        'Pinch or use left slider to zoom · drag to pan · hover or click a '
        'node/edge to inspect'
    )
    fsm_buttons_html = ''
    fsm_legend_html = ''
  total_raw_bytes = sum(len(svg_markup) for svg_markup in svg_views.values())
  use_lazy_gzip = (
      len(svg_views) > 1 and total_raw_bytes > _LAZY_GZIP_THRESHOLD_BYTES
  )
  view_divs: list[str] = []
  gz_views: dict[str, str] = {}
  for view_key, svg_markup in svg_views.items():
    is_active = view_key == initial_view_key
    if use_lazy_gzip and not is_active:
      compressed = gzip.compress(svg_markup.encode('utf-8'), compresslevel=9)
      gz_views[view_key] = base64.b64encode(compressed).decode('ascii')
      continue
    active_cls = ' active' if is_active else ''
    escaped_key = html.escape(view_key, quote=True)
    view_divs.append(
        f'<div class="psyre-viz-view{active_cls}" '
        f'data-psyre-view="{escaped_key}">'
        f'{svg_markup}</div>'
    )
  views_html = '\n      '.join(view_divs)
  all_prefixes_json = _safe_inline_json(list(fsm_prefixes))
  available_keys_json = _safe_inline_json(list(svg_views.keys()))
  gz_views_json = _safe_inline_json(gz_views)
  initial_key_json = _safe_inline_json(initial_view_key)

  return f"""<div id="{uid}_root" class="psyre-viz-card">
  <style>
    #{uid}_root.psyre-viz-card {{
      font-family: {font_stack};
      border: 1px solid #CBD5E1;
      border-radius: 10px;
      background: #FFFFFF;
      color: #0F172A;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
      margin: 8px 0;
      user-select: none;
    }}
    #{uid}_root .psyre-viz-toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 12px;
      background: #F8FAFC;
      border-bottom: 1px solid #E2E8F0;
      font-size: 12.5px;
    }}
    #{uid}_root .psyre-viz-title {{
      font-weight: 600;
      color: #0F172A;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    #{uid}_root .psyre-viz-badge {{
      font-size: 11px;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 999px;
      background: #E2E8F0;
      color: #334155;
    }}
    #{uid}_root .psyre-viz-controls {{
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    #{uid}_root .psyre-viz-btn {{
      border: 1px solid #CBD5E1;
      background: #FFFFFF;
      color: #1E293B;
      border-radius: 6px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      line-height: 1.4;
    }}
    #{uid}_root .psyre-viz-btn:hover {{
      background: #F1F5F9;
      border-color: #94A3B8;
    }}
    #{uid}_root .psyre-viz-zoom-label {{
      min-width: 46px;
      text-align: center;
      font-variant-numeric: tabular-nums;
      color: #475569;
      font-size: 11.5px;
    }}
    #{uid}_root .psyre-viz-stage {{
      position: relative;
      width: 100%;
      height: {height_px}px;
      overflow: hidden;
      cursor: grab;
      background-color: #FFFFFF;
      background-image: radial-gradient(#E2E8F0 1px, transparent 1px);
      background-size: 18px 18px;
      touch-action: none;
    }}
    #{uid}_root .psyre-viz-stage.dragging {{
      cursor: grabbing;
    }}
    #{uid}_root .psyre-viz-zoom-rail {{
      position: absolute;
      left: 12px;
      top: 50%;
      transform: translateY(-50%);
      z-index: 10;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      padding: 6px 5px;
      background: rgba(248, 250, 252, 0.92);
      border: 1px solid #CBD5E1;
      border-radius: 8px;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
      backdrop-filter: blur(4px);
      cursor: default;
    }}
    #{uid}_root .psyre-viz-zoom-rail .psyre-viz-btn {{
      width: 24px;
      height: 22px;
      padding: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      font-weight: 600;
      line-height: 1;
    }}
    #{uid}_root .psyre-viz-zoom-track {{
      position: relative;
      width: 24px;
      height: 140px;
      cursor: ns-resize;
      touch-action: none;
    }}
    #{uid}_root .psyre-viz-zoom-bar {{
      position: absolute;
      left: 50%;
      top: 8px;
      bottom: 8px;
      width: 4px;
      transform: translateX(-50%);
      background: #CBD5E1;
      border-radius: 999px;
    }}
    #{uid}_root .psyre-viz-zoom-tick {{
      position: absolute;
      left: 4px;
      right: 4px;
      height: 1.5px;
      background: #94A3B8;
      pointer-events: none;
    }}
    #{uid}_root .psyre-viz-zoom-thumb {{
      position: absolute;
      left: 50%;
      top: 50%;
      width: 18px;
      height: 10px;
      transform: translate(-50%, -50%);
      background: #2563EB;
      border: 1.5px solid #FFFFFF;
      border-radius: 5px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.3);
      cursor: grab;
      transition: background 0.1s ease, transform 0.08s ease;
    }}
    #{uid}_root .psyre-viz-zoom-thumb:hover,
    #{uid}_root .psyre-viz-zoom-thumb.dragging {{
      background: #1D4ED8;
      transform: translate(-50%, -50%) scale(1.12);
    }}
    #{uid}_root .psyre-viz-zoom-thumb.dragging {{
      cursor: grabbing;
    }}
    #{uid}_root .psyre-viz-zoom-readout {{
      font-size: 10px;
      font-variant-numeric: tabular-nums;
      color: #475569;
      line-height: 1.2;
      min-width: 32px;
      text-align: center;
    }}
    #{uid}_root .psyre-viz-canvas {{
      transform-origin: 0 0;
      display: inline-block;
      will-change: transform;
    }}
    #{uid}_root .psyre-viz-view {{
      display: none;
    }}
    #{uid}_root .psyre-viz-view.active {{
      display: block;
    }}
    #{uid}_root .psyre-viz-view svg {{
      display: block;
      overflow: visible;
    }}
    #{uid}_root g.node, #{uid}_root g.edge {{
      cursor: pointer;
      transition: opacity 0.12s ease;
    }}
    #{uid}_root g.node[data-psyre-fsm-box]:hover polygon,
    #{uid}_root g.node[data-psyre-fsm-box]:hover path {{
      stroke-width: 2.6px;
      filter: brightness(0.96);
    }}
    #{uid}_root g.cluster[data-psyre-fsm-cluster] {{
      cursor: pointer;
    }}
    #{uid}_root g.cluster[data-psyre-fsm-cluster]:hover > polygon,
    #{uid}_root g.cluster[data-psyre-fsm-cluster]:hover > path {{
      stroke-width: 2.3px;
    }}
    #{uid}_root.has-selection g.node.dimmed,
    #{uid}_root.has-selection g.edge.dimmed {{
      opacity: 0.22;
    }}
    #{uid}_root .psyre-viz-footer {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 6px 12px;
      background: #F8FAFC;
      border-top: 1px solid #E2E8F0;
      font-size: 11.5px;
      color: #475569;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      min-height: 20px;
    }}
    #{uid}_root .psyre-viz-legend {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: {font_stack};
      white-space: nowrap;
      color: #64748B;
    }}
    #{uid}_root .psyre-swatch {{
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 2px;
      margin-right: 4px;
      vertical-align: middle;
    }}
  </style>
  <div class="psyre-viz-toolbar">
    <div class="psyre-viz-title">
      <span>{header_title}</span>
      <span class="psyre-viz-badge">{sem_badge} · {node_count} nodes</span>
    </div>
    <div class="psyre-viz-controls">{fsm_buttons_html}
      <button type="button" class="psyre-viz-btn"
        data-act="out" title="Zoom out">−</button>
      <span class="psyre-viz-zoom-label" id="{uid}_zoom">100%</span>
      <button type="button" class="psyre-viz-btn"
        data-act="in" title="Zoom in">+</button>
      <button type="button" class="psyre-viz-btn"
        data-act="fit" title="Fit graph to view">Fit</button>
      <button type="button" class="psyre-viz-btn"
        data-act="reset" title="Reset 1:1 scale">1:1</button>
    </div>
  </div>
  <div class="psyre-viz-stage" id="{uid}_stage">
    <div class="psyre-viz-zoom-rail" id="{uid}_zoom_rail"
      title="Drag slider to zoom">
      <button type="button" class="psyre-viz-btn"
        data-act="in" title="Zoom in">+</button>
      <div class="psyre-viz-zoom-track" id="{uid}_zoom_track"
        role="slider" aria-label="Zoom control cursor"
        aria-valuemin="3" aria-valuemax="1200" aria-valuenow="100">
        <div class="psyre-viz-zoom-bar"></div>
        <div class="psyre-viz-zoom-tick" id="{uid}_zoom_tick"
          title="100% (1:1)"></div>
        <div class="psyre-viz-zoom-thumb" id="{uid}_zoom_thumb"></div>
      </div>
      <button type="button" class="psyre-viz-btn"
        data-act="out" title="Zoom out">−</button>
      <span class="psyre-viz-zoom-readout" id="{uid}_zoom_rail_val">100%</span>
    </div>
    <div class="psyre-viz-canvas" id="{uid}_canvas">
      {views_html}
    </div>
  </div>
  <div class="psyre-viz-footer">
    <div id="{uid}_info">{default_hint}</div>
    <div class="psyre-viz-legend">{fsm_legend_html}
      <span><i class="psyre-swatch"
        style="background:{STYLE_SELECTOR[0]};border:1px solid {STYLE_SELECTOR[1]}"></i>when</span>
      <span><i class="psyre-swatch"
        style="background:{STYLE_MUX[0]};border:1px solid {STYLE_MUX[1]}"></i>merge</span>
      <span><i class="psyre-swatch"
        style="background:{STYLE_FBY[0]};border:1px solid {STYLE_FBY[1]}"></i>fby/pre (∨) · post (∧)</span>
      <span><i class="psyre-swatch"
        style="background:{STYLE_STATEFUL[0]};border:1px solid {STYLE_STATEFUL[1]}"></i>morph · Σ · FIFO</span>
    </div>
  </div>
  <script>
    (() => {{
      const root = document.getElementById("{uid}_root");
      const stage = document.getElementById("{uid}_stage");
      const canvas = document.getElementById("{uid}_canvas");
      const zoomEl = document.getElementById("{uid}_zoom");
      const zoomRail = document.getElementById("{uid}_zoom_rail");
      const zoomTrack = document.getElementById("{uid}_zoom_track");
      const zoomThumb = document.getElementById("{uid}_zoom_thumb");
      const zoomTick = document.getElementById("{uid}_zoom_tick");
      const zoomRailVal = document.getElementById("{uid}_zoom_rail_val");
      const infoEl = document.getElementById("{uid}_info");
      if (!root || !stage || !canvas) return;

      const allPrefixes = {all_prefixes_json};
      const availableKeys = {available_keys_json};
      const availableKeySet = new Set(availableKeys);
      const gzViews = {gz_views_json};
      const lazyMode = Object.keys(gzViews).length > 0;
      const viewCache = new Map();
      Array.from(canvas.querySelectorAll(".psyre-viz-view")).forEach((el) => {{
        viewCache.set(el.getAttribute("data-psyre-view") || "", el);
      }});
      let switchSeq = 0;
      const expandedSet = new Set();
      function encodeStateKey(prefixes) {{
        return Array.from(new Set(prefixes))
          .sort()
          .map((p) => p.replace(/%/g, "%25").replace(/\\|/g, "%7C"))
          .join("|");
      }}
      function decodeStateKey(key) {{
        if (!key) return [];
        return key
          .split("|")
          .filter(Boolean)
          .map((p) => p.replace(/%7C/g, "|").replace(/%25/g, "%"));
      }}
      function setExpandedFromKey(key) {{
        expandedSet.clear();
        decodeStateKey(key).forEach((p) => expandedSet.add(p));
      }}
      setExpandedFromKey({initial_key_json});

      function activeSvg() {{
        const v =
          canvas.querySelector(".psyre-viz-view.active") ||
          canvas.querySelector(".psyre-viz-view");
        return v ? v.querySelector("svg") : null;
      }}

      if (!activeSvg()) return;

      const defaultMsg = "{default_hint}";
      let scale = 1, tx = 0, ty = 0;
      let dragging = false, moved = false, startX = 0, startY = 0;
      let lockedInfo = null;
      let minScale = 0.03;
      const MAX_SCALE = 12.0;
      const TRACK_PAD = 8;

      function logScaleBounds() {{
        const logMin = Math.log(minScale);
        return {{ logMin, logSpan: Math.log(MAX_SCALE) - logMin }};
      }}

      function scaleToRatio(s) {{
        const {{ logMin, logSpan }} = logScaleBounds();
        const clamped = Math.min(MAX_SCALE, Math.max(minScale, s));
        return (Math.log(clamped) - logMin) / logSpan;
      }}

      function ratioToScale(r) {{
        const {{ logMin, logSpan }} = logScaleBounds();
        const clamped = Math.min(1, Math.max(0, r));
        return Math.exp(logMin + clamped * logSpan);
      }}

      function updateZoomCursor() {{
        const pctText = `${{Math.round(scale * 100)}}%`;
        zoomEl.textContent = pctText;
        if (zoomRailVal) zoomRailVal.textContent = pctText;
        if (!zoomTrack || !zoomThumb) return;
        const h = zoomTrack.clientHeight || 140;
        const usable = Math.max(1, h - 2 * TRACK_PAD);
        const ratio = scaleToRatio(scale);
        const topPx = TRACK_PAD + (1 - ratio) * usable;
        zoomThumb.style.top = `${{topPx.toFixed(1)}}px`;
        zoomTrack.setAttribute("aria-valuenow", String(Math.round(scale * 100)));
        if (zoomTick) {{
          const tickTop = TRACK_PAD + (1 - scaleToRatio(1.0)) * usable;
          zoomTick.style.top = `${{tickTop.toFixed(1)}}px`;
        }}
      }}

      function svgNaturalSize() {{
        const svg = activeSvg();
        if (!svg) return {{ w: 400, h: 300 }};
        const vb = svg.viewBox && svg.viewBox.baseVal;
        if (vb && vb.width > 0 && vb.height > 0) {{
          return {{ w: vb.width * (96 / 72), h: vb.height * (96 / 72) }};
        }}
        const r = svg.getBoundingClientRect();
        return {{ w: r.width || 400, h: r.height || 300 }};
      }}

      function applyTransform() {{
        const xStr = tx.toFixed(1);
        const yStr = ty.toFixed(1);
        const sStr = scale.toFixed(4);
        canvas.style.transform =
          `translate(${{xStr}}px, ${{yStr}}px) scale(${{sStr}})`;
        updateZoomCursor();
      }}

      function fitToStage() {{
        const sw = stage.clientWidth || 640;
        const sh = stage.clientHeight || {height_px};
        const {{ w, h }} = svgNaturalSize();
        const pad = 28;
        const fit = Math.min((sw - pad) / w, (sh - pad) / h);
        minScale = Math.min(0.03, Math.max(0.002, fit * 0.5));
        scale = Math.min(2.0, Math.max(minScale, fit));
        tx = (sw - w * scale) * 0.5;
        ty = (sh - h * scale) * 0.5;
        applyTransform();
      }}

      function setScaleAt(targetScale, clientX, clientY) {{
        const rect = stage.getBoundingClientRect();
        const px = clientX - rect.left;
        const py = clientY - rect.top;
        const next = Math.min(MAX_SCALE, Math.max(minScale, targetScale));
        const ratio = next / scale;
        tx = px - (px - tx) * ratio;
        ty = py - (py - ty) * ratio;
        scale = next;
        applyTransform();
      }}

      function zoomAt(factor, clientX, clientY) {{
        setScaleAt(scale * factor, clientX, clientY);
      }}

      function clearSelection() {{
        lockedInfo = null;
        root.classList.remove("has-selection");
        const svg = activeSvg();
        if (svg) {{
          svg.querySelectorAll("g.node.dimmed, g.edge.dimmed").forEach((el) => {{
            el.classList.remove("dimmed");
          }});
        }}
        infoEl.textContent = defaultMsg;
      }}

      async function decodeGzSvg(b64) {{
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        if (typeof DecompressionStream === "undefined") {{
          throw new Error("DecompressionStream unavailable");
        }}
        const ds = new DecompressionStream("gzip");
        const stream = new Blob([bytes]).stream().pipeThrough(ds);
        return await new Response(stream).text();
      }}

      async function ensureViewEl(key) {{
        const cached = viewCache.get(key);
        if (cached) return cached;
        const b64 = gzViews[key];
        if (!b64) return null;
        infoEl.textContent = "Decompressing view...";
        const svgMarkup = await decodeGzSvg(b64);
        const div = document.createElement("div");
        div.className = "psyre-viz-view";
        div.setAttribute("data-psyre-view", key);
        div.innerHTML = svgMarkup;
        viewCache.set(key, div);
        return div;
      }}

      async function switchView(nextKey, statusMsg) {{
        if (!availableKeySet.has(nextKey)) return;
        const seq = ++switchSeq;
        let targetEl;
        try {{
          targetEl = await ensureViewEl(nextKey);
        }} catch (err) {{
          infoEl.textContent = `Failed to render view: ${{err}}`;
          return;
        }}
        if (seq !== switchSeq || !targetEl) return;
        setExpandedFromKey(nextKey);
        if (lazyMode) {{
          Array.from(canvas.querySelectorAll(".psyre-viz-view")).forEach((v) => {{
            if (v !== targetEl) {{
              v.classList.remove("active");
              v.remove();
            }}
          }});
          if (targetEl.parentNode !== canvas) {{
            canvas.appendChild(targetEl);
          }}
          targetEl.classList.add("active");
        }} else {{
          viewCache.forEach((v, k) => {{
            v.classList.toggle("active", k === nextKey);
          }});
        }}
        clearSelection();
        if (statusMsg) infoEl.textContent = statusMsg;
        fitToStage();
      }}

      function ancestorPrefixes(pfx, includeSelf) {{
        const parts = pfx.split("/");
        const limit = includeSelf ? parts.length : parts.length - 1;
        const result = [];
        for (let i = 1; i <= limit; i++) {{
          const anc = parts.slice(0, i).join("/");
          if (allPrefixes.includes(anc)) result.push(anc);
        }}
        return result;
      }}

      function expandPrefix(pfx) {{
        const ancs = ancestorPrefixes(pfx, true);
        const candidate = new Set(expandedSet);
        ancs.forEach((a) => candidate.add(a));
        let nextKey = encodeStateKey(candidate);
        if (!availableKeySet.has(nextKey)) {{
          nextKey = encodeStateKey(ancs);
        }}
        if (!availableKeySet.has(nextKey)) {{
          let bestKey = encodeStateKey(allPrefixes);
          let bestLen = Infinity;
          availableKeys.forEach((k) => {{
            const ks = decodeStateKey(k);
            if (ancs.every((a) => ks.includes(a)) && ks.length < bestLen) {{
              bestKey = k;
              bestLen = ks.length;
            }}
          }});
          nextKey = bestKey;
        }}
        switchView(
          nextKey,
          `Expanded '${{pfx}}' (click cluster header/box to collapse)`
        );
      }}

      function collapsePrefix(pfx) {{
        const candidate = Array.from(expandedSet).filter(
          (item) => item !== pfx && !item.startsWith(pfx + "/")
        );
        let nextKey = encodeStateKey(candidate);
        if (!availableKeySet.has(nextKey)) {{
          nextKey = encodeStateKey(ancestorPrefixes(pfx, false));
        }}
        if (!availableKeySet.has(nextKey)) {{
          nextKey = "";
        }}
        switchView(
          nextKey,
          `Collapsed '${{pfx}}' (click ⊞ box to expand)`
        );
      }}

      stage.addEventListener("wheel", (ev) => {{
        if (!ev.ctrlKey && !ev.metaKey) return;
        ev.preventDefault();
        const modeScale = ev.deltaMode === 1 ? 32 : (ev.deltaMode === 2 ? 320 : 1);
        const dy = ev.deltaY * modeScale;
        const factor = Math.exp(-dy * 0.032);
        zoomAt(factor, ev.clientX, ev.clientY);
      }}, {{ passive: false }});

      let lastGestureScale = 1;
      stage.addEventListener("gesturestart", (ev) => {{
        ev.preventDefault();
        lastGestureScale = ev.scale || 1;
      }}, {{ passive: false }});
      stage.addEventListener("gesturechange", (ev) => {{
        ev.preventDefault();
        const curScale = ev.scale || 1;
        if (lastGestureScale > 0 && curScale > 0) {{
          const factor = Math.pow(curScale / lastGestureScale, 2.2);
          zoomAt(factor, ev.clientX, ev.clientY);
        }}
        lastGestureScale = curScale;
      }}, {{ passive: false }});

      if (zoomRail) {{
        ["pointerdown", "mousedown", "click", "dblclick"].forEach((evName) => {{
          zoomRail.addEventListener(evName, (ev) => ev.stopPropagation());
        }});
      }}

      if (zoomTrack && zoomThumb) {{
        let cursorDragging = false;
        function applyCursorY(clientY) {{
          const tRect = zoomTrack.getBoundingClientRect();
          const h = tRect.height || 140;
          const usable = Math.max(1, h - 2 * TRACK_PAD);
          const relY = clientY - tRect.top - TRACK_PAD;
          const ratio = 1 - Math.min(1, Math.max(0, relY / usable));
          const sRect = stage.getBoundingClientRect();
          setScaleAt(
            ratioToScale(ratio),
            sRect.left + sRect.width * 0.5,
            sRect.top + sRect.height * 0.5
          );
        }}
        zoomTrack.addEventListener("pointerdown", (ev) => {{
          if (ev.button !== 0) return;
          ev.preventDefault();
          ev.stopPropagation();
          cursorDragging = true;
          zoomThumb.classList.add("dragging");
          try {{ zoomTrack.setPointerCapture(ev.pointerId); }} catch (_) {{}}
          applyCursorY(ev.clientY);
        }});
        zoomTrack.addEventListener("pointermove", (ev) => {{
          if (!cursorDragging) return;
          ev.preventDefault();
          applyCursorY(ev.clientY);
        }});
        function endCursorDrag(ev) {{
          if (!cursorDragging) return;
          cursorDragging = false;
          zoomThumb.classList.remove("dragging");
          try {{ zoomTrack.releasePointerCapture(ev.pointerId); }} catch (_) {{}}
        }}
        zoomTrack.addEventListener("pointerup", endCursorDrag);
        zoomTrack.addEventListener("pointercancel", endCursorDrag);
      }}

      let downTarget = null;
      let downX = 0, downY = 0;
      const activePointers = new Map();
      let prevPinchDist = 0;

      stage.addEventListener("pointerdown", (ev) => {{
        activePointers.set(ev.pointerId, {{ x: ev.clientX, y: ev.clientY }});
        if (activePointers.size === 2) {{
          dragging = false;
          moved = true;
          stage.classList.remove("dragging");
          const pts = Array.from(activePointers.values());
          prevPinchDist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
          return;
        }}
        if (ev.button !== 0 || activePointers.size > 1) return;
        dragging = true;
        moved = false;
        downTarget = ev.target;
        downX = ev.clientX;
        downY = ev.clientY;
        startX = ev.clientX - tx;
        startY = ev.clientY - ty;
      }});

      stage.addEventListener("pointermove", (ev) => {{
        if (activePointers.has(ev.pointerId)) {{
          activePointers.set(ev.pointerId, {{ x: ev.clientX, y: ev.clientY }});
        }}
        if (activePointers.size === 2) {{
          const pts = Array.from(activePointers.values());
          const dist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
          if (prevPinchDist > 4 && dist > 4) {{
            const midX = 0.5 * (pts[0].x + pts[1].x);
            const midY = 0.5 * (pts[0].y + pts[1].y);
            const factor = Math.pow(dist / prevPinchDist, 1.8);
            zoomAt(factor, midX, midY);
          }}
          prevPinchDist = dist;
          return;
        }}
        if (!dragging) return;
        if (!moved && Math.hypot(ev.clientX - downX, ev.clientY - downY) > 4) {{
          moved = true;
          stage.classList.add("dragging");
          try {{ stage.setPointerCapture(ev.pointerId); }} catch (_) {{}}
        }}
        if (!moved) return;
        tx = ev.clientX - startX;
        ty = ev.clientY - startY;
        applyTransform();
      }});

      function endDrag(ev) {{
        activePointers.delete(ev.pointerId);
        if (activePointers.size < 2) prevPinchDist = 0;
        if (!dragging) return;
        dragging = false;
        stage.classList.remove("dragging");
        if (moved) {{
          try {{ stage.releasePointerCapture(ev.pointerId); }} catch (_) {{}}
        }}
      }}
      stage.addEventListener("pointerup", endDrag);
      stage.addEventListener("pointercancel", endDrag);
      stage.addEventListener("dblclick", () => fitToStage());

      root.querySelectorAll(".psyre-viz-btn").forEach((btn) => {{
        btn.addEventListener("click", () => {{
          const act = btn.getAttribute("data-act");
          const rect = stage.getBoundingClientRect();
          const cx = rect.left + rect.width * 0.5;
          const cy = rect.top + rect.height * 0.5;
          if (act === "in") zoomAt(1.4, cx, cy);
          else if (act === "out") zoomAt(1 / 1.4, cx, cy);
          else if (act === "fit") fitToStage();
          else if (act === "reset") {{
            scale = 1;
            const {{ w, h }} = svgNaturalSize();
            tx = Math.max(12, (rect.width - w) * 0.5);
            ty = Math.max(12, (rect.height - h) * 0.5);
            applyTransform();
          }} else if (act === "expand-all") {{
            const nextKey = encodeStateKey(allPrefixes);
            switchView(nextKey, "Expanded all automata");
          }} else if (act === "collapse-all") {{
            switchView("", "Collapsed all automata");
          }}
        }});
      }});

      canvas.addEventListener("mouseover", (ev) => {{
        if (lockedInfo) return;
        const g = ev.target.closest("g[data-psyre-info]");
        if (g) infoEl.textContent = g.getAttribute("data-psyre-info");
      }});
      canvas.addEventListener("mouseout", (ev) => {{
        if (lockedInfo) return;
        const g = ev.target.closest("g[data-psyre-info]");
        if (g) infoEl.textContent = defaultMsg;
      }});

      stage.addEventListener("click", (ev) => {{
        if (moved) return;
        const rawTarget = (
          ev.target &&
          ev.target !== stage &&
          ev.target !== canvas &&
          typeof ev.target.closest === "function"
        )
          ? ev.target
          : (
            downTarget && typeof downTarget.closest === "function"
              ? downTarget
              : document.elementFromPoint(ev.clientX, ev.clientY)
          );
        if (!rawTarget || typeof rawTarget.closest !== "function") {{
          clearSelection();
          return;
        }}
        const fsmBox = rawTarget.closest("g.node[data-psyre-fsm-box]");
        if (fsmBox) {{
          const pfx = fsmBox.getAttribute("data-psyre-fsm-box");
          if (pfx) expandPrefix(pfx);
          return;
        }}
        const g = rawTarget.closest("g.node");
        if (!g) {{
          const edgeEl = rawTarget.closest("g.edge");
          if (edgeEl) {{
            const svg = activeSvg();
            if (!svg) return;
            const allNodes = Array.from(svg.querySelectorAll("g.node"));
            const allEdges = Array.from(svg.querySelectorAll("g.edge"));
            const src = edgeEl.getAttribute("data-psyre-src") || "";
            const dst = edgeEl.getAttribute("data-psyre-dst") || "";
            lockedInfo =
              edgeEl.getAttribute("data-psyre-info") || `${{src}} -> ${{dst}}`;
            infoEl.textContent = lockedInfo;
            root.classList.add("has-selection");
            const keepNodes = new Set([src, dst].filter(Boolean));
            allNodes.forEach((n) => {{
              const k = n.getAttribute("data-psyre-key") || "";
              n.classList.toggle("dimmed", !keepNodes.has(k));
            }});
            allEdges.forEach((e) => {{
              e.classList.toggle("dimmed", e !== edgeEl);
            }});
            return;
          }}
          const fsmCluster = rawTarget.closest("g.cluster[data-psyre-fsm-cluster]");
          if (fsmCluster) {{
            const pfx = fsmCluster.getAttribute("data-psyre-fsm-cluster");
            if (pfx) {{
              collapsePrefix(pfx);
              return;
            }}
          }}
          clearSelection();
          return;
        }}
        const svg = activeSvg();
        if (!svg) return;
        const allNodes = Array.from(svg.querySelectorAll("g.node"));
        const allEdges = Array.from(svg.querySelectorAll("g.edge"));
        const dotKey = g.getAttribute("data-psyre-key") || "";
        const netKey = g.getAttribute("data-psyre-net") || "";
        lockedInfo = g.getAttribute("data-psyre-info") || dotKey;
        infoEl.textContent = lockedInfo;
        root.classList.add("has-selection");

        const seedKeys = new Set([dotKey]);
        if (netKey) {{
          allNodes.forEach((n) => {{
            if (n.getAttribute("data-psyre-net") === netKey) {{
              seedKeys.add(n.getAttribute("data-psyre-key") || "");
            }}
          }});
        }}
        const keepNodes = new Set(seedKeys);
        const keepEdges = new Set();
        allEdges.forEach((edgeEl) => {{
          const src = edgeEl.getAttribute("data-psyre-src") || "";
          const dst = edgeEl.getAttribute("data-psyre-dst") || "";
          if (!src || !dst) return;
          if (seedKeys.has(src) || seedKeys.has(dst)) {{
            keepEdges.add(edgeEl);
            keepNodes.add(src);
            keepNodes.add(dst);
          }}
        }});
        allNodes.forEach((n) => {{
          const k = n.getAttribute("data-psyre-key") || "";
          n.classList.toggle("dimmed", !keepNodes.has(k));
        }});
        allEdges.forEach((e) => {{
          e.classList.toggle("dimmed", !keepEdges.has(e));
        }});
      }});

      if (typeof ResizeObserver !== "undefined") {{
        const ro = new ResizeObserver((entries) => {{
          const w = entries[0] && entries[0].contentRect.width;
          if (w > 4) {{
            ro.disconnect();
            fitToStage();
          }}
        }});
        ro.observe(stage);
      }}
      fitToStage();
      requestAnimationFrame(fitToStage);
    }})();
  </script>
</div>"""


def _resolve_initial_view_key(bundle: SchematicBundle) -> str:
  """Returns the initial view key present in `bundle.views`."""
  if bundle.initial_view_key in bundle.views:
    return bundle.initial_view_key
  if '' in bundle.views:
    return ''
  if bundle.views:
    return next(iter(bundle.views))
  raise ValueError('SchematicBundle.views must contain at least one view.')


def render_bundle_html(bundle: SchematicBundle) -> str:
  """Renders a `SchematicBundle` into a self-contained interactive HTML viewer."""
  return build_visualization(bundle).html


def _get_temp_export_dir() -> str:
  """Returns a process-lifetime temporary directory for exported artifacts."""
  global _TEMP_EXPORT_DIR
  if _TEMP_EXPORT_DIR is None:
    _TEMP_EXPORT_DIR = tempfile.TemporaryDirectory(prefix='psyre_viz_')
    atexit.register(_TEMP_EXPORT_DIR.cleanup)
  return _TEMP_EXPORT_DIR.name


def _write_export_text(
    content: str,
    path: str | None,
    *,
    suffix: str,
) -> str:
  """Writes `content` to `path` or a session temporary file and returns path."""
  if path is None:
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=_get_temp_export_dir(),
        prefix='psyre_graph_',
        suffix=suffix,
        delete=False,
    ) as tmp_file:
      tmp_file.write(content)
      return tmp_file.name

  parent_dir = os.path.dirname(path)
  if parent_dir:
    os.makedirs(parent_dir, mode=0o755, exist_ok=True)
  with open(path, 'w', encoding='utf-8') as out_file:
    out_file.write(content)
  return path


@dataclasses.dataclass(frozen=True)
class GraphVisualization:
  """Interactive HTML, standalone SVG, and DOT bundle for a Psyre `Graph`."""

  svg: str
  html: str
  dot: str

  def __repr__(self) -> str:
    return (
        f'GraphVisualization(svg_chars={len(self.svg)}, '
        f'html_chars={len(self.html)}, dot_chars={len(self.dot)})'
    )

  def _repr_html_(self) -> str:
    """Renders the interactive zoomable schematic inline in Colab/Jupyter."""
    return self.html

  def save_svg(self, path: str | None = None) -> str:
    """Writes standalone SVG to `path` (or a temporary file if `None`)."""
    return _write_export_text(self.svg, path, suffix='.svg')

  def save_html(self, path: str | None = None) -> str:
    """Writes standalone UTF-8 interactive HTML to `path` (or a temp file)."""
    doc = (
        self.html
        if self.html.lstrip().startswith('<!DOCTYPE')
        else f'<!DOCTYPE html>\n<meta charset="utf-8">\n{self.html}'
    )
    return _write_export_text(doc, path, suffix='.html')


def build_visualization(bundle: SchematicBundle) -> GraphVisualization:
  """Builds a `GraphVisualization` from a `SchematicBundle`."""
  initial_key = _resolve_initial_view_key(bundle)
  svg_views = {key: render_view_svg(view) for key, view in bundle.views.items()}
  html_markup = _wrap_interactive_html(
      svg_views,
      title=bundle.title,
      semantics_label=bundle.semantics_label,
      node_count=bundle.node_count,
      height_px=bundle.height_px,
      initial_view_key=initial_key,
      fsm_prefixes=bundle.fsm_prefixes,
  )
  return GraphVisualization(
      svg=svg_views[initial_key],
      html=html_markup,
      dot=bundle.views[initial_key].dot_src,
  )
