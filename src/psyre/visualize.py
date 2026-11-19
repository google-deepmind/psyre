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
"""Graphviz/DOT schematic layout builder for Psyre Graphs."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
import concurrent.futures
import dataclasses
import html
import os
import re
import shutil
import subprocess
from typing import Any

from psyre import analysis
from psyre import clock
from psyre import common
from psyre import graph as graph_lib
from psyre import gui
from psyre import istructure
from psyre import stream as stream_lib
import pydot

RankDir = gui.RankDir
NodeKind = gui.NodeKind
GraphVisualization = gui.GraphVisualization

_SELECTOR_OPS = frozenset({'when', 'whennot'})
_FBY_LIKE_OPS = frozenset({'fby', 'pre'})
_FIFO_OPS = frozenset({'delay', 'buffer'})
_NET_REF_THRESHOLD = 80

_DUNDER_OP_GLYPHS: Mapping[str, str] = {
    '__add__': '+',
    '__radd__': '+',
    '__sub__': '-',
    '__rsub__': '-',
    '__mul__': '×',
    '__rmul__': '×',
    '__matmul__': '@',
    '__rmatmul__': '@',
    '__truediv__': '/',
    '__rtruediv__': '/',
    '__floordiv__': '//',
    '__rfloordiv__': '//',
    '__mod__': '%',
    '__rmod__': '%',
    '__pow__': '**',
    '__rpow__': '**',
    '__lshift__': '<<',
    '__rlshift__': '<<',
    '__rshift__': '>>',
    '__rrshift__': '>>',
    '__and__': '∧',
    '__rand__': '∧',
    '__or__': '∨',
    '__ror__': '∨',
    '__xor__': '⊕',
    '__rxor__': '⊕',
    '__pos__': '+',
    '__neg__': '-',
    '__invert__': '¬',
    '__lt__': '<',
    '__le__': '≤',
    '__eq__': '=',
    '__ne__': '≠',
    '__gt__': '>',
    '__ge__': '≥',
    '__getitem__': '[]',
}
_DUNDER_GLYPH_SET = frozenset(_DUNDER_OP_GLYPHS.values())

_DEFAULT_NODE_LOCAL_NAMES = frozenset({
    'buffer',
    'case',
    'counter',
    'current',
    'delay',
    'fby',
    'ifthenelse',
    'last',
    'merge',
    'morph',
    'once_stream',
    'pnr',
    'pns',
    'post',
    'pre',
    'prefix_sum',
    'random_stream',
    'scan',
    'state',
    'state_nr',
    'state_ns',
    'state_r',
    'state_s',
    'switch',
    'when',
    'whennot',
})


def _get_dot_binary(dot_prog: str | None = None) -> str:
  """Resolves the Graphviz `dot` executable from runfiles or system PATH."""
  if dot_prog is not None:
    return dot_prog
  bundled = os.path.abspath(
      os.path.join(os.path.dirname(__file__), '..', '..', 'graphviz', 'dot')
  )
  if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
    return bundled
  return shutil.which('dot') or 'dot'


def _run_dot_svg(dot_src: str, prog: str) -> str:
  """Runs Graphviz `dot -Tsvg` on `dot_src` and returns UTF-8 SVG markup."""
  proc = subprocess.run(
      [prog, '-Tsvg'],
      input=dot_src.encode('utf-8'),
      capture_output=True,
      check=False,
  )
  if proc.returncode != 0 and (
      b'<svg' not in proc.stdout or b'</svg>' not in proc.stdout
  ):
    err_msg = proc.stderr.decode('utf-8', errors='replace').strip()
    raise RuntimeError(
        f'Graphviz dot failed (code {proc.returncode}): {err_msg}'
    )
  return proc.stdout.decode('utf-8')


def _quote_dot(text: str) -> str:
  """Quotes an arbitrary string safely for pydot / Graphviz DOT syntax."""
  escaped = (
      text.replace('\\', '\\\\')
      .replace('"', '\\"')
      .replace('\r', '')
      .replace('\n', '\\n')
  )
  return f'"{escaped}"'


def _format_dot_label(
    label: str,
    num_left_inputs: int,
    rankdir: RankDir,
    *,
    min_width_in: float = 0.0,
    min_height_in: float = 0.0,
    pad_pt: int = 6,
) -> str:
  """Formats a DOT node label, adding vertical left input ports in LR mode."""
  if rankdir != 'LR' or num_left_inputs < 2:
    return _quote_dot(label)
  escaped = html.escape(label).replace('\n', '<BR/>')
  w_pt = max(0, round(min_width_in * 72.0))
  h_pt = max(14 * num_left_inputs, round(min_height_in * 72.0))
  row_h = max(12, h_pt // num_left_inputs)
  td_attrs = f'ROWSPAN="{num_left_inputs}" HEIGHT="{h_pt}"'
  if w_pt > 0:
    td_attrs += f' WIDTH="{w_pt}"'
  if label:
    td_attrs += f' CELLPADDING="{pad_pt}"'
  rows = [
      f'<TR><TD PORT="in0" WIDTH="0" HEIGHT="{row_h}"></TD>'
      f'<TD {td_attrs}>{escaped}</TD></TR>'
  ]
  for idx in range(1, num_left_inputs):
    rows.append(
        f'<TR><TD PORT="in{idx}" WIDTH="0" HEIGHT="{row_h}"></TD></TR>'
    )
  return (
      '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
      + ''.join(rows)
      + '</TABLE>>'
  )


def _format_scalar(value: Any, max_len: int = 24) -> str:
  """Formats a scalar or enum value concisely for node and edge labels."""
  return common.format_tag(value, max_len)


def _format_static_stream(istruct: istructure.IStruct) -> str:
  """Formats a static periodic IStruct concisely for input badges."""
  if istructure.is_one_periodic(istruct):
    return _format_scalar(istruct.period[0])
  prefix_part = ''
  if istruct.prefix:
    items = ', '.join(_format_scalar(v, 10) for v in istruct.prefix[:3])
    if len(istruct.prefix) > 3:
      items = f'{items}, …'
    prefix_part = f'[{items}]+'
  period_items = ', '.join(_format_scalar(v, 10) for v in istruct.period[:4])
  if len(istruct.period) > 4:
    period_items = f'{period_items}, …'
  return f'{prefix_part}[{period_items}]*ω'


def _node_automaton_prefix(node: graph_lib.Node) -> str:
  """Returns the slash-separated automaton scope prefix of `node` (or '')."""
  if '/' not in node.name:
    return ''
  return node.name.rsplit('/', 1)[0]


def _node_local_name(node: graph_lib.Node) -> str:
  """Returns the local name of `node` with any automaton prefix stripped."""
  return node.name.rsplit('/', 1)[-1] if node.name else ''


def _has_custom_node_name(node: graph_lib.Node) -> bool:
  """Returns True if `node` was assigned an explicit non-default local name."""
  local = _node_local_name(node)
  return (
      bool(local)
      and local != node.op
      and local not in _DEFAULT_NODE_LOCAL_NAMES
  )


def _explicit_stream_names(
    target_graph: graph_lib.Graph,
) -> dict[istructure.IStruct, str]:
  """Maps IStructs to explicit user-assigned Stream handle names when present."""
  names: dict[istructure.IStruct, str] = {}
  for handle in target_graph.handles:
    if not handle.name or handle.istruct in names:
      continue
    prod = target_graph.producer(handle.istruct)
    leaf = handle.name.rsplit('/', 1)[-1]
    if prod is None:
      names[handle.istruct] = handle.name
      continue
    if handle.name not in {prod.op, prod.name} and leaf != prod.op:
      names[handle.istruct] = handle.name
  for handle in target_graph.handles:
    if handle.name and handle.istruct not in names:
      prod = target_graph.producer(handle.istruct)
      leaf = handle.name.rsplit('/', 1)[-1]
      if prod is None or leaf != prod.op:
        names[handle.istruct] = handle.name
  return names


def _is_unnamed_scalar_constant(
    istruct: istructure.IStruct,
    target_graph: graph_lib.Graph,
    stream_names: Mapping[istructure.IStruct, str],
) -> bool:
  """Returns True for unnamed single-element static scalar constants."""
  return (
      target_graph.producer(istruct) is None
      and istructure.is_one_periodic(istruct)
      and istruct not in stream_names
  )


def _is_pre_uninitialized_init(
    node: graph_lib.Node,
    port_idx: int,
    istruct: istructure.IStruct,
    target_graph: graph_lib.Graph,
    stream_names: Mapping[istructure.IStruct, str],
) -> bool:
  """Returns True for the implicit None initial input of `pre(x)`."""
  return (
      node.op == 'fby'
      and port_idx == 0
      and _is_unnamed_scalar_constant(istruct, target_graph, stream_names)
      and istruct.period[0] is None
      and _node_local_name(node) == 'pre'
  )


def _unwrap_once_stream_arg(
    istruct: istructure.IStruct,
    target_graph: graph_lib.Graph,
) -> istructure.IStruct:
  """Unwraps match/automaton selector gating around a once_stream argument."""
  curr = istruct
  seen: set[int] = set()
  while curr.id not in seen:
    seen.add(curr.id)
    prod = target_graph.producer(curr)
    if prod is not None and prod.op in _SELECTOR_OPS and len(prod.inputs) >= 2:
      curr = prod.inputs[1].istruct
    else:
      break
  return curr


def _is_once_stream_constant_arg(
    node: graph_lib.Node,
    port_idx: int,
    istruct: istructure.IStruct,
    target_graph: graph_lib.Graph,
    stream_names: Mapping[istructure.IStruct, str],
) -> bool:
  """Returns True for an unnamed scalar constant `k` of `once_stream(k)`."""
  if node.op != 'once_stream' or port_idx != 0:
    return False
  unwrapped = _unwrap_once_stream_arg(istruct, target_graph)
  return _is_unnamed_scalar_constant(unwrapped, target_graph, stream_names)


def _once_stream_k_label(
    node: graph_lib.Node,
    target_graph: graph_lib.Graph,
    stream_names: Mapping[istructure.IStruct, str],
) -> str:
  """Extracts the subscript `k` label for a `once_stream(k)` node."""
  if not node.inputs:
    return 'k'
  k_istruct = _unwrap_once_stream_arg(node.inputs[0].istruct, target_graph)
  if k_istruct in stream_names:
    return stream_names[k_istruct].rsplit('/', 1)[-1]
  if istructure.is_one_periodic(k_istruct):
    return _format_scalar(k_istruct.period[0], 12)
  return 'k'


def _node_display_name(node: graph_lib.Node) -> str:
  """Returns the Stage-1 display name printed on a primitive Node."""
  key = _node_local_name(node) or node.op
  return _DUNDER_OP_GLYPHS.get(key, key)


def _node_full_display_name(node: graph_lib.Node) -> str:
  """Returns the full (automaton-prefixed) display name of `node`."""
  pfx = _node_automaton_prefix(node)
  local = _node_display_name(node)
  return f'{pfx}/{local}' if pfx else local


def _iter_output_paths(
    value: Any,
    prefix: str = 'out',
    stream_names: Mapping[istructure.IStruct, str] | None = None,
) -> Iterator[tuple[str, istructure.IStruct]]:
  """Yields `(label_path, IStruct)` pairs for leaves of `graph.output`."""
  if value is None:
    return
  is_anon_prefix = bool(re.fullmatch(r'out(\[\d+\])*', prefix))
  if isinstance(value, (istructure.IStruct, stream_lib.Stream)):
    istruct = istructure.as_istruct(value)
    if is_anon_prefix and stream_names and istruct in stream_names:
      yield stream_names[istruct].rsplit('/', 1)[-1], istruct
    else:
      yield prefix, istruct
    return
  if isinstance(value, tuple):
    cls = type(value)
    if cls is not tuple:
      for field_name, item in zip(value._fields, value):  # pylint: disable=protected-access
        sub = field_name if is_anon_prefix else f'{prefix}.{field_name}'
        yield from _iter_output_paths(item, sub, stream_names)
      return
  if isinstance(value, (tuple, list)):
    for idx, item in enumerate(value):
      yield from _iter_output_paths(item, f'{prefix}[{idx}]', stream_names)
    return
  if isinstance(value, Mapping):
    for key, item in value.items():
      key_str = str(key)
      sub = key_str if is_anon_prefix else f'{prefix}.{key_str}'
      yield from _iter_output_paths(item, sub, stream_names)
    return


def _merge_arm_labels(node: graph_lib.Node) -> dict[int, str]:
  """Extracts concise arm tag labels for a `merge`/`switch` node's inputs."""
  lookup = node.kwargs.get('lookup')
  if not isinstance(lookup, common.MergeSelector):
    return {}
  if lookup.is_boolean:
    return {1: 'T', 2: 'F'}
  by_arm: dict[int, list[str]] = {}
  for (_, tag), arm_idx in lookup.arms.items():
    by_arm.setdefault(arm_idx, []).append(_format_scalar(tag, 12))
  labels: dict[int, str] = {}
  for arm_idx, tags in by_arm.items():
    joined = ','.join(tags[:2])
    if len(tags) > 2:
      joined = f'{joined},…'
    labels[arm_idx + 1] = joined
  if lookup.default_index is not None:
    port = lookup.default_index + 1
    existing = labels.get(port)
    labels[port] = f'{existing}|_' if existing else '_'
  return labels


def _when_selector_label(node: graph_lib.Node) -> str:
  """Extracts a concise tag predicate label for a multi-valued `when` node."""
  selector = node.kwargs.get('selector')
  if not isinstance(selector, common.WhenSelector) or selector.is_boolean:
    return ''
  tags = [
      _format_scalar(val, 12)
      for _, val in sorted(selector.values, key=common.tag_sort_key)
  ]
  if not tags:
    return '*' if selector.negated else '∅'
  if len(tags) == 1:
    return f'≠{tags[0]}' if selector.negated else f'={tags[0]}'
  joined = ','.join(tags[:2])
  if len(tags) > 2:
    joined = f'{joined},…'
  return f'∉{{{joined}}}' if selector.negated else f'∈{{{joined}}}'


def _node_palette(node: graph_lib.Node, in_cycle: bool) -> tuple[str, str, str]:
  """Returns `(fill, border, text)` hex colors for a primitive Node."""
  if in_cycle:
    return gui.STYLE_CYCLE
  if node.op in _SELECTOR_OPS:
    return gui.STYLE_SELECTOR
  if node.op in {'merge', 'switch'}:
    return gui.STYLE_MUX
  if node.op in _FBY_LIKE_OPS or node.op == 'post':
    return gui.STYLE_FBY
  if (
      node.is_stateful
      or node.op == 'prefix_sum'
      or node.op in _FIFO_OPS
      or node.op == 'counter'
      or node.op == 'once_stream'
  ):
    return gui.STYLE_STATEFUL
  return gui.STYLE_COMB


def _node_tooltip(
    node: graph_lib.Node,
    clocks: clock.ClockAnalysis,
    stream_names: Mapping[istructure.IStruct, str],
    in_cycle: bool,
) -> str:
  """Builds a multi-field summary string for tooltips and node inspection."""
  disp_name = _node_display_name(node)
  full_name = _node_full_display_name(node)
  op_str = node.op
  clk_str = str(clocks.node_clock(node))
  out_name = stream_names.get(node.output_stream, '')
  parts: list[str] = []
  if full_name != op_str:
    parts.append(f'name={full_name}')
  parts.extend([
      f'op={op_str}',
      f'id=#{node.id}',
      f'clock={clk_str}',
  ])
  if out_name and out_name not in {disp_name, full_name, node.name}:
    parts.append(f'stream={out_name}')
  if node.op == 'when':
    sel_lbl = _when_selector_label(node)
    if sel_lbl:
      parts.append(f'match={sel_lbl}')
  if node.is_stateful and node.resets:
    parts.append(f'resets={len(node.resets)}')
  if in_cycle:
    parts.append('CAUSALITY_CYCLE=true')
  return ' | '.join(parts)


def _is_selector_input_port(node: graph_lib.Node, port_key: int | str) -> bool:
  """Returns True if `port_key` is a top-connected selector/control input."""
  return (
      isinstance(port_key, int)
      and port_key == 0
      and (
          node.op in _SELECTOR_OPS
          or node.op in {'merge', 'switch', 'ifthenelse'}
      )
  )


def _is_inline_source(
    istruct: istructure.IStruct,
    prod: graph_lib.Node | None,
) -> bool:
  """Returns True if `istruct` renders as a re-instantiated inline arrow."""
  return (prod is None and istruct.static) or (
      prod is not None and prod.op == 'once_stream'
  )


def _make_pydot_op_node(
    node: graph_lib.Node,
    *,
    target_graph: graph_lib.Graph,
    rankdir: RankDir,
    clocks: clock.ClockAnalysis,
    stream_names: Mapping[istructure.IStruct, str],
    in_cycle: bool,
    num_left_inputs: int = 1,
) -> tuple[pydot.Node, gui.NodeShapeSpec]:
  """Creates the pydot.Node and custom SVG shape spec for a Psyre Node."""
  dot_name = f'n_{node.id}'
  dom_id = f'psyre_n_{node.id}'
  label = _node_display_name(node)
  fill, border, text_color = _node_palette(node, in_cycle)
  tooltip = _node_tooltip(node, clocks, stream_names, in_cycle)
  use_table_ports = (
      rankdir == 'LR' and num_left_inputs >= 2 and node.op != 'ifthenelse'
  )
  min_left_h = (
      (0.22 * (num_left_inputs - 1) + 0.26)
      if (rankdir == 'LR' and num_left_inputs >= 2)
      else 0.0
  )

  attrs: dict[str, Any] = {
      'id': _quote_dot(dom_id),
      'fontname': _quote_dot('Helvetica'),
      'fontsize': '11',
      'fontcolor': _quote_dot(text_color),
      'style': _quote_dot('filled'),
      'fillcolor': _quote_dot(fill),
      'color': _quote_dot(border),
      'penwidth': '1.6',
  }

  kind: NodeKind = 'other'
  subscript = ''
  display_label = label
  min_w_in = 0.0
  min_h_in = min_left_h
  pad_pt = 6
  if node.op in _FBY_LIKE_OPS or node.op == 'post':
    kind = 'post' if node.op == 'post' else 'fby'
    n_val = node.kwargs.get('n', 1)
    if isinstance(n_val, int) and not isinstance(n_val, bool) and n_val > 1:
      subscript = str(n_val)
    side_in = max(
        0.60,
        min_left_h,
        min(1.45, 0.095 * (len(label) + len(subscript)) + 0.34),
    )
    attrs['shape'] = 'box'
    attrs['fixedsize'] = 'true'
    attrs['width'] = f'{side_in:.2f}'
    attrs['height'] = f'{side_in:.2f}'
    min_w_in = side_in
    min_h_in = side_in
    pad_pt = 4
  elif node.op in _SELECTOR_OPS:
    selector = node.kwargs.get('selector')
    is_neg_bool = (
        isinstance(selector, common.WhenSelector)
        and selector.is_boolean
        and selector.negated
    )
    kind = 'whennot' if (node.op == 'whennot' or is_neg_bool) else 'when'
    if rankdir == 'TB':
      attrs['shape'] = 'trapezium'
      attrs['margin'] = _quote_dot('0.14,0.08')
    else:
      attrs['shape'] = 'box'
      min_w_in = 0.68
      min_h_in = max(0.62, min_left_h)
      attrs['height'] = f'{min_h_in:.2f}'
      attrs['margin'] = _quote_dot('0.14,0.10')
      pad_pt = 8
  elif node.op in {'merge', 'switch'}:
    kind = 'merge'
    num_arms = max(2, len(node.inputs) - 1)
    if rankdir == 'TB':
      attrs['shape'] = 'invtrapezium'
      attrs['margin'] = _quote_dot('0.14,0.08')
    else:
      attrs['shape'] = 'box'
      min_w_in = 0.68
      min_h_in = max(0.64, 0.26 * num_arms, min_left_h)
      attrs['height'] = f'{min_h_in:.2f}'
      attrs['margin'] = _quote_dot('0.14,0.10')
      pad_pt = 8
  elif node.op == 'ifthenelse':
    kind = 'ifthenelse'
    attrs['shape'] = 'diamond'
    attrs['style'] = _quote_dot('filled')
    attrs['margin'] = _quote_dot('0.09,0.05')
    if min_left_h > 0.0:
      attrs['height'] = f'{max(0.56, min_left_h):.2f}'
  elif node.op == 'prefix_sum':
    kind = 'sigma'
    display_label = 'Σ'
    attrs['shape'] = 'box'
    attrs['style'] = _quote_dot('filled')
    attrs['fontsize'] = '18'
    attrs['margin'] = _quote_dot('0.12,0.06')
    min_w_in = 0.54
    min_h_in = max(0.42, min_left_h)
    if min_left_h > 0.0:
      attrs['height'] = f'{min_h_in:.2f}'
  elif node.op == 'once_stream':
    kind = 'once_stream'
    subscript = _once_stream_k_label(node, target_graph, stream_names)
    display_label = f'δ_{subscript}'
    attrs['shape'] = 'box'
    attrs['style'] = _quote_dot('filled')
    attrs['fontsize'] = '14'
    attrs['margin'] = _quote_dot('0.12,0.06')
  elif node.op in _FIFO_OPS:
    kind = 'fifo'
    display_label = ''
    attrs['shape'] = 'box'
    attrs['style'] = _quote_dot('filled')
    attrs['fixedsize'] = 'true'
    if rankdir == 'TB':
      attrs['width'] = '0.32'
      attrs['height'] = '0.68'
    else:
      min_w_in = 0.68
      min_h_in = max(0.32, min_left_h)
      attrs['width'] = f'{min_w_in:.2f}'
      attrs['height'] = f'{min_h_in:.2f}'
      pad_pt = 0
  elif node.op == 'counter':
    kind = 'counter'
    display_label = '042'
    attrs['shape'] = 'box'
    attrs['style'] = _quote_dot('filled')
    attrs['fixedsize'] = 'true'
    min_w_in = 0.72
    min_h_in = max(0.38, min_left_h)
    attrs['width'] = f'{min_w_in:.2f}'
    attrs['height'] = f'{min_h_in:.2f}'
  elif node.is_stateful:
    n_val = node.kwargs.get('n', 1)
    if isinstance(n_val, int) and not isinstance(n_val, bool) and n_val > 1:
      subscript = str(n_val)
    attrs['shape'] = 'box'
    attrs['peripheries'] = '2'
    attrs['margin'] = _quote_dot('0.12,0.08')
    min_w_in = 0.62
    min_h_in = max(0.42, min_left_h)
    if min_left_h > 0.0:
      attrs['height'] = f'{min_h_in:.2f}'
  elif label in _DUNDER_GLYPH_SET:
    attrs['shape'] = 'box'
    attrs['style'] = _quote_dot('rounded,filled')
    attrs['fontsize'] = '13'
    min_w_in = 0.42
    min_h_in = max(0.36, min_left_h)
    attrs['width'] = f'{min_w_in:.2f}'
    attrs['height'] = f'{min_h_in:.2f}'
    attrs['margin'] = _quote_dot('0.08,0.05')
    pad_pt = 5
  else:
    attrs['shape'] = 'box'
    attrs['style'] = _quote_dot('rounded,filled')
    attrs['margin'] = _quote_dot('0.12,0.08')
    min_w_in = 0.56
    min_h_in = max(0.38, min_left_h)
    if min_left_h > 0.0:
      attrs['height'] = f'{min_h_in:.2f}'

  if use_table_ports:
    attrs.pop('fixedsize', None)
    attrs['width'] = '0'
    attrs['height'] = '0'
    attrs['margin'] = _quote_dot('0,0')
    attrs['label'] = _format_dot_label(
        display_label,
        num_left_inputs,
        rankdir,
        min_width_in=min_w_in,
        min_height_in=min_h_in,
        pad_pt=pad_pt,
    )
  else:
    attrs['label'] = _quote_dot(display_label)
  spec = gui.NodeShapeSpec(
      dom_id=dom_id,
      kind=kind,
      rankdir=rankdir,
      border_color=border,
      info=tooltip,
      subscript=subscript,
  )
  return pydot.Node(dot_name, **attrs), spec


def _make_terminal_dot_node(
    dot_name: str,
    dom_id: str,
    label: str,
    palette: tuple[str, str, str],
    *,
    compact: bool = False,
) -> pydot.Node:
  """Creates a styled input, constant, or output terminal pydot.Node."""
  fill, border, text_color = palette
  return pydot.Node(
      dot_name,
      id=_quote_dot(dom_id),
      label=_quote_dot(label),
      shape='box',
      style=_quote_dot('rounded,filled'),
      fillcolor=_quote_dot(fill),
      color=_quote_dot(border),
      fontcolor=_quote_dot(text_color),
      fontname=_quote_dot('Helvetica'),
      fontsize='9.5' if compact else '10.5',
      penwidth='1.2',
      height='0.28' if compact else '0.36',
      margin=_quote_dot('0.07,0.04' if compact else '0.10,0.06'),
  )


def _make_static_stream_dot_node(
    dot_name: str,
    dom_id: str,
    label: str,
    text_color: str = '#334155',
    fontsize: str = '9.5',
    min_width: str = '0.36',
) -> pydot.Node:
  """Creates a boxless plain-text pydot.Node for an inline stream arrow."""
  return pydot.Node(
      dot_name,
      id=_quote_dot(dom_id),
      label=_quote_dot(label),
      shape='plaintext',
      fontcolor=_quote_dot(text_color),
      fontname=_quote_dot('Helvetica'),
      fontsize=fontsize,
      width=min_width,
      height='0.26',
      margin=_quote_dot('0.08,0.04'),
  )


def _activity_key(
    node: graph_lib.Node, clocks: clock.ClockAnalysis
) -> tuple[str, ...]:
  """Returns the tuple of enclosing ClockStep strings for scope clustering."""
  node_clk = clocks.node_clock(node)
  return tuple(str(step) for step in node_clk.steps)


def _automaton_prefixes(target_graph: graph_lib.Graph) -> list[str]:
  """Returns all automaton scope prefixes in first-seen hierarchical order."""
  seen: set[str] = set()
  ordered: list[str] = []
  for node in target_graph.nodes:
    pfx = _node_automaton_prefix(node)
    if not pfx:
      continue
    parts = pfx.split('/')
    for depth in range(1, len(parts) + 1):
      anc = '/'.join(parts[:depth])
      if anc not in seen:
        seen.add(anc)
        ordered.append(anc)
  return ordered


def _valid_expansion_states(
    fsm_prefixes: Sequence[str], max_states: int = 8
) -> list[frozenset[str]]:
  """Returns valid hierarchical sets of expanded automaton/node prefixes."""
  if not fsm_prefixes:
    return [frozenset()]
  children: dict[str, list[str]] = {'': []}
  for pfx in fsm_prefixes:
    children.setdefault(pfx, [])
    parent = pfx.rsplit('/', 1)[0] if '/' in pfx else ''
    children.setdefault(parent, []).append(pfx)

  def subtree_states(root: str) -> list[frozenset[str]]:
    direct_children = children.get(root, ())
    leaf_children = [c for c in direct_children if not children.get(c)]
    branch_children = [c for c in direct_children if children.get(c)]
    if len(leaf_children) > 4:
      child_lists = [subtree_states(c) for c in branch_children]
      child_lists.append([frozenset(), frozenset(leaf_children)])
    else:
      child_lists = [subtree_states(c) for c in direct_children]
    total_prod = 1
    for opts in child_lists:
      total_prod *= max(1, len(opts))
    if total_prod <= max_states:
      combos: list[frozenset[str]] = [frozenset()]
      for opts in child_lists:
        next_combos: list[frozenset[str]] = []
        for base in combos:
          for opt in opts:
            next_combos.append(base | opt)
        combos = next_combos
    else:
      combos = [frozenset()]
      seen_combos: set[frozenset[str]] = {frozenset()}
      for opts in child_lists:
        for opt in opts:
          if opt not in seen_combos:
            seen_combos.add(opt)
            combos.append(opt)
      all_direct = frozenset(direct_children)
      if all_direct and all_direct not in seen_combos:
        seen_combos.add(all_direct)
        combos.append(all_direct)
    if not root:
      return combos
    return [frozenset(), *(frozenset({root}) | c for c in combos)]

  states = subtree_states('')
  full = frozenset(fsm_prefixes)
  if full not in states:
    states.append(full)
  return states


@dataclasses.dataclass(frozen=True)
class _LayoutOptions:
  """View-level layout options shared across schematic builders."""

  rankdir: RankDir = 'LR'
  show_constants: bool = True
  group_by_activity: bool = True
  show_reset_edges: bool = True
  collapse_automata: bool = False
  expanded_automata: Iterable[str] | None = None


@dataclasses.dataclass(frozen=True)
class _DotMetadata:
  """DOM metadata maps produced alongside a `pydot.Dot` schematic."""

  shape_specs: dict[str, gui.NodeShapeSpec] = dataclasses.field(
      default_factory=dict
  )
  info_by_dom_id: dict[str, str] = dataclasses.field(default_factory=dict)
  net_by_dom_id: dict[str, str] = dataclasses.field(default_factory=dict)
  fsm_box_by_dom_id: dict[str, str] = dataclasses.field(default_factory=dict)
  fsm_cluster_by_dom_id: dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  static_edge_to_node: dict[str, str] = dataclasses.field(default_factory=dict)
  reversed_edge_ids: set[str] = dataclasses.field(default_factory=set)

  def to_schematic_view(self, raw_svg: str, dot_src: str) -> gui.SchematicView:
    """Packages Graphviz SVG and DOM metadata into a `gui.SchematicView`."""
    return gui.SchematicView(
        raw_svg=raw_svg,
        dot_src=dot_src,
        shape_specs=self.shape_specs,
        info_by_dom_id=self.info_by_dom_id,
        net_by_dom_id=self.net_by_dom_id,
        fsm_box_by_dom_id=self.fsm_box_by_dom_id,
        fsm_cluster_by_dom_id=self.fsm_cluster_by_dom_id,
        static_edge_to_node=self.static_edge_to_node,
        reversed_edge_ids=frozenset(self.reversed_edge_ids),
    )


def _net_ref_label(
    istruct: istructure.IStruct,
    prod: graph_lib.Node | None,
    stream_names: Mapping[istructure.IStruct, str],
    collapsed_pfx: str | None = None,
) -> str:
  """Formats a compact schematic net-reference badge label for `istruct`."""
  if collapsed_pfx:
    if prod is not None:
      short = stream_names.get(istruct, '') or _node_display_name(prod)
      if short.startswith(f'{collapsed_pfx}/'):
        short = short[len(collapsed_pfx) + 1 :]
      return f'↪ {collapsed_pfx}.{short}'
    return f'↪ {collapsed_pfx}'
  explicit = stream_names.get(istruct, '')
  if prod is not None:
    base = explicit or _node_display_name(prod)
    return f'↪ {base}#{prod.id}'
  if explicit:
    return f'↪ {explicit}'
  if istruct.static:
    return f'↪ {_format_static_stream(istruct)}'
  return f'↪ S_{istruct.id}'


def _can_reach(
    adj: Mapping[str, set[str]], start: str, target: str
) -> bool:
  """Returns True if `target` is reachable from `start` in `adj`."""
  if start == target:
    return True
  visited: set[str] = {start}
  stack: list[str] = [start]
  while stack:
    curr = stack.pop()
    for nxt in adj.get(curr, ()):
      if nxt == target:
        return True
      if nxt not in visited:
        visited.add(nxt)
        stack.append(nxt)
  return False


def _should_use_net_ref(
    node: graph_lib.Node,
    port_key: int | str,
    istruct: istructure.IStruct,
    prod: graph_lib.Node | None,
    *,
    is_delayed: bool,
    is_inline_src: bool,
    on_cycle: bool,
    is_fby_prod: bool,
    is_fby_immediate_edge: bool,
    same_cluster: bool,
    node_in_cluster: bool,
    large_graph: bool,
    stream_fanout: Mapping[istructure.IStruct, int],
) -> bool:
  """Returns True if an edge should route through a local net-ref badge."""
  if is_fby_immediate_edge:
    return False
  if not (large_graph and not on_cycle and not is_inline_src):
    return False
  fanout = stream_fanout.get(istruct, 0)
  if (
      is_fby_prod
      or (node.op == 'fby' and port_key == 0)
      or (is_delayed and (prod is None or prod.id >= node.id))
  ):
    return True
  if prod is None:
    return node_in_cluster or fanout > 6
  return (
      (not same_cluster)
      or prod.id >= node.id
      or (node.id - prod.id) > 16
      or fanout > 8
  )


@dataclasses.dataclass(frozen=True)
class _InputEdgeInfo:
  """Classification of a visible input edge shared by pre-pass and emission."""

  prod: graph_lib.Node | None
  is_inline_src: bool
  is_delayed: bool
  prod_collapsed_pfx: str | None
  prod_box_dot: str | None
  on_cycle: bool
  same_cluster: bool
  is_fby_immediate: bool
  use_net_ref: bool


@dataclasses.dataclass(frozen=True)
class _GraphVizContext:
  """Precomputed graph-wide analysis shared across expansion state views."""

  clocks: clock.ClockAnalysis
  stream_names: dict[istructure.IStruct, str]
  cycle_ids: set[int]
  cycle_edges: set[tuple[int, int]]
  stream_fanout: dict[istructure.IStruct, int]
  fsm_prefixes: list[str]
  descendant_nodes: dict[str, list[graph_lib.Node]]
  sample_by_pfx: dict[str, graph_lib.Node]
  base_act: dict[str, tuple[str, ...]]
  parent_fsm: dict[str, str]
  non_once_consumers: set[istructure.IStruct]


def _analyze_graph(target_graph: graph_lib.Graph) -> _GraphVizContext:
  """Computes view-independent clock, cycle, fanout, and hierarchy metadata."""
  target_graph.require_finalized()
  clocks = clock.infer_clocks(target_graph)
  stream_names = _explicit_stream_names(target_graph)
  cycle = analysis.find_instantaneous_cycle(target_graph)
  cycle_ids = {n.id for n in cycle} if cycle is not None else set()
  cycle_edges: set[tuple[int, int]] = set()
  if cycle is not None:
    for idx, src_node in enumerate(cycle):
      nxt_node = cycle[(idx + 1) % len(cycle)]
      cycle_edges.add((src_node.id, nxt_node.id))

  stream_fanout: dict[istructure.IStruct, int] = {}
  for c_node in target_graph.nodes:
    for h in c_node.inputs:
      stream_fanout[h.istruct] = stream_fanout.get(h.istruct, 0) + 1
  output_streams = [
      out_s
      for _, out_s in _iter_output_paths(
          target_graph.output, stream_names=stream_names
      )
  ]
  for out_s in output_streams:
    stream_fanout[out_s] = stream_fanout.get(out_s, 0) + 1
  non_once_consumers: set[istructure.IStruct] = set(output_streams)
  for c_node in reversed(target_graph.nodes):
    if c_node.op == 'once_stream':
      continue
    if (
        c_node.op in _SELECTOR_OPS
        and c_node.output_stream not in non_once_consumers
    ):
      continue
    for h in c_node.inputs:
      non_once_consumers.add(h.istruct)
    for r in c_node.resets:
      non_once_consumers.add(r.istruct)

  fsm_prefixes = _automaton_prefixes(target_graph)
  descendant_nodes: dict[str, list[graph_lib.Node]] = {}
  sample_by_pfx: dict[str, graph_lib.Node] = {}
  base_act: dict[str, tuple[str, ...]] = {}
  parent_fsm: dict[str, str] = {}
  for pfx in fsm_prefixes:
    members = [
        n
        for n in target_graph.nodes
        if (np := _node_automaton_prefix(n)) == pfx or np.startswith(f'{pfx}/')
    ]
    descendant_nodes[pfx] = members
    clocked_members = [
        n
        for n in members
        if (n.inputs or n.resets)
        and n.op != 'once_stream'
        and not (
            n.op in _SELECTOR_OPS and n.output_stream not in non_once_consumers
        )
    ] or members
    sample = min(clocked_members, key=lambda n: len(_activity_key(n, clocks)))
    sample_by_pfx[pfx] = sample
    base_act[pfx] = _activity_key(sample, clocks)
    parent_fsm[pfx] = pfx.rsplit('/', 1)[0] if '/' in pfx else ''

  return _GraphVizContext(
      clocks=clocks,
      stream_names=stream_names,
      cycle_ids=cycle_ids,
      cycle_edges=cycle_edges,
      stream_fanout=stream_fanout,
      fsm_prefixes=fsm_prefixes,
      descendant_nodes=descendant_nodes,
      sample_by_pfx=sample_by_pfx,
      base_act=base_act,
      parent_fsm=parent_fsm,
      non_once_consumers=non_once_consumers,
  )


class _DotBuilder:
  """Constructs a `pydot.Dot` schematic and DOM metadata for one view state."""

  def __init__(
      self,
      target_graph: graph_lib.Graph,
      ctx: _GraphVizContext,
      opts: _LayoutOptions,
  ) -> None:
    target_graph.require_finalized()
    if opts.rankdir not in ('LR', 'TB'):
      raise ValueError(
          f"rankdir must be 'LR' or 'TB', got {opts.rankdir!r}."
      )
    self.graph = target_graph
    self.ctx = ctx
    self.opts = opts
    self.out_port = 'e' if opts.rankdir == 'LR' else 's'
    self.in_port = 'w' if opts.rankdir == 'LR' else 'n'
    self.sel_port = 'n' if opts.rankdir == 'LR' else 'w'

    if opts.expanded_automata is not None:
      self.expanded_set = set(opts.expanded_automata)
    elif opts.collapse_automata:
      self.expanded_set = set()
    else:
      self.expanded_set = set(ctx.fsm_prefixes)

    self.collapsed_owner: dict[int, str | None] = {}
    for node in target_graph.nodes:
      np = _node_automaton_prefix(node)
      if not np:
        self.collapsed_owner[node.id] = None
        continue
      parts = np.split('/')
      owner: str | None = None
      for depth in range(1, len(parts) + 1):
        anc = '/'.join(parts[:depth])
        if anc not in self.expanded_set:
          owner = anc
          break
      self.collapsed_owner[node.id] = owner

    self.active_collapsed = [
        pfx
        for pfx in ctx.fsm_prefixes
        if any(self.collapsed_owner[n.id] == pfx for n in target_graph.nodes)
    ]
    self.visible_nodes = [
        n
        for n in target_graph.nodes
        if self.collapsed_owner[n.id] is None
        and n.op != 'once_stream'
        and not (
            n.op in _SELECTOR_OPS
            and n.output_stream not in ctx.non_once_consumers
        )
    ]
    self.large_graph = (
        len(self.visible_nodes) + len(self.active_collapsed)
    ) > _NET_REF_THRESHOLD

    self.dot = pydot.Dot(
        'psyre_graph',
        graph_type='digraph',
        rankdir=opts.rankdir,
        newrank='true',
        splines='true',
        nodesep='0.38',
        ranksep='0.58',
        pad='0.22',
        bgcolor=_quote_dot('#FFFFFF'),
        fontname=_quote_dot('Helvetica'),
    )
    self.meta = _DotMetadata()
    self.containers: dict[tuple[str, tuple[str, ...]], pydot.Cluster] = {}
    self.box_dot_by_prefix: dict[str, str] = {}
    self.box_container_by_prefix: dict[str, pydot.Dot | pydot.Cluster] = {}
    self.box_ports_by_prefix: dict[
        str, list[tuple[bool, istructure.IStruct, graph_lib.Node]]
    ] = {}
    self.node_containers: dict[int, pydot.Dot | pydot.Cluster] = {}
    self.pydot_nodes_by_id: dict[int, pydot.Node] = {}
    self.left_inputs_by_id: dict[int, int] = {}
    self.same_name_groups: dict[tuple[int, str], list[graph_lib.Node]] = {}
    self.fby_primary_consumer: dict[int, int] = {}
    self.ext_sources: dict[istructure.IStruct, str] = {}
    self.direct_out_producers: set[str] = set()
    self.net_ref_by_source: dict[str, dict[istructure.IStruct, int]] = {}
    self.constrained_adj: dict[str, set[str]] = {}

  def build(self) -> tuple[pydot.Dot, _DotMetadata]:
    """Populates the DOT graph and returns `(dot, metadata)`."""
    self._init_clusters()
    self._add_collapsed_boxes()
    self._add_visible_nodes()
    self._add_same_name_chains()
    self._compute_fby_primary_consumers()
    self._add_collapsed_box_edges()
    self._prepopulate_constrained_adj()
    self._add_visible_node_edges()
    self._add_output_terminals()
    self._add_net_out_badges()
    return self.dot, self.meta

  def _get_container(
      self,
      fsm_pfx: str,
      act_key: tuple[str, ...],
      sample_node: graph_lib.Node,
  ) -> pydot.Dot | pydot.Cluster:
    """Returns or creates the nested DOT cluster for `(fsm_pfx, act_key)`."""
    ctx = self.ctx
    group_act = self.opts.group_by_activity
    b_act = ctx.base_act.get(fsm_pfx, ()) if (fsm_pfx and group_act) else ()
    eff_act = act_key if group_act else ()
    if len(eff_act) < len(b_act):
      eff_act = b_act
      sample_node = ctx.sample_by_pfx[fsm_pfx]
    if not fsm_pfx and not eff_act:
      return self.dot
    key = (fsm_pfx, eff_act)
    existing = self.containers.get(key)
    if existing is not None:
      return existing
    if len(eff_act) > len(b_act):
      parent = self._get_container(fsm_pfx, eff_act[:-1], sample_node)
      act_clks = ctx.clocks.node_activity_clocks(sample_node)
      scope_clk = str(act_clks[len(eff_act)])
      fsm_slug = re.sub(r'[^A-Za-z0-9_]+', '_', fsm_pfx) if fsm_pfx else 'root'
      act_slug = re.sub(r'[^A-Za-z0-9_]+', '_', '_'.join(eff_act))
      act_dom = f'psyre_act_cluster_{fsm_slug}_{act_slug}'
      cluster = pydot.Cluster(
          f'act_{fsm_slug}_{act_slug}',
          id=_quote_dot(act_dom),
          label=_quote_dot(f'clock: {scope_clk}'),
          style=_quote_dot('rounded,dashed,filled'),
          color=_quote_dot('#CBD5E1'),
          fillcolor=_quote_dot('#F8FAFC'),
          fontname=_quote_dot('Helvetica'),
          fontsize='9.5',
          fontcolor=_quote_dot('#475569'),
          margin='10',
      )
      if fsm_pfx:
        self.meta.fsm_cluster_by_dom_id[act_dom] = fsm_pfx
        self.meta.info_by_dom_id[act_dom] = (
            f'automaton={fsm_pfx} | clock={scope_clk} | Click to collapse'
        )
      parent.add_subgraph(cluster)
      self.containers[key] = cluster
      return cluster

    parent = self._get_container(ctx.parent_fsm[fsm_pfx], b_act, sample_node)
    fsm_idx = ctx.fsm_prefixes.index(fsm_pfx)
    fsm_slug = re.sub(r'[^A-Za-z0-9_]+', '_', fsm_pfx)
    c_dom = f'psyre_fsm_cluster_{fsm_idx}'
    members = ctx.descendant_nodes[fsm_pfx]
    fsm_clk = str(ctx.clocks.node_activity_clocks(sample_node)[len(b_act)])
    cluster = pydot.Cluster(
        f'fsm_{fsm_idx}_{fsm_slug}',
        id=_quote_dot(c_dom),
        label=_quote_dot(f'⊟ {fsm_pfx} ({len(members)} nodes)'),
        style=_quote_dot('rounded,filled'),
        color=_quote_dot('#6366F1'),
        fillcolor=_quote_dot('#F5F7FF'),
        penwidth='1.8',
        fontname=_quote_dot('Helvetica-Bold'),
        fontsize='10.5',
        fontcolor=_quote_dot('#312E81'),
        margin='12',
    )
    self.meta.fsm_cluster_by_dom_id[c_dom] = fsm_pfx
    self.meta.info_by_dom_id[c_dom] = (
        f'automaton={fsm_pfx} | nodes={len(members)} | '
        f'clock={fsm_clk} | Click header or box to collapse'
    )
    parent.add_subgraph(cluster)
    self.containers[key] = cluster
    return cluster

  def _iter_visible_inputs(
      self, node: graph_lib.Node
  ) -> Iterator[tuple[int | str, istructure.IStruct, bool]]:
    """Yields visible `(port_key, istruct, is_delay_sched)` inputs of `node`."""
    stream_names = self.ctx.stream_names
    show_consts = self.opts.show_constants
    for port_idx, handle in enumerate(node.inputs):
      istruct = handle.istruct
      if _is_pre_uninitialized_init(
          node, port_idx, istruct, self.graph, stream_names
      ):
        continue
      if _is_once_stream_constant_arg(
          node, port_idx, istruct, self.graph, stream_names
      ):
        continue
      if not show_consts and _is_unnamed_scalar_constant(
          istruct, self.graph, stream_names
      ):
        continue
      yield port_idx, istruct, False
    if node.op == 'delay':
      p_handle = node.kwargs.get('p_stream')
      if isinstance(p_handle, (istructure.IStruct, stream_lib.Stream)):
        p_s = istructure.as_istruct(p_handle)
        if show_consts or not _is_unnamed_scalar_constant(
            p_s, self.graph, stream_names
        ):
          yield 'p', p_s, True

  def _visible_resets(
      self, node: graph_lib.Node
  ) -> list[tuple[int, istructure.IStruct]]:
    """Returns visible `(rst_idx, istruct)` reset inputs for `node`."""
    if not (self.opts.show_reset_edges and node.is_stateful):
      return []
    show_consts = self.opts.show_constants
    stream_names = self.ctx.stream_names
    return [
        (idx, h.istruct)
        for idx, h in enumerate(node.resets)
        if show_consts
        or not _is_unnamed_scalar_constant(h.istruct, self.graph, stream_names)
    ]

  def _count_left_inputs(self, node: graph_lib.Node) -> int:
    """Counts visible left-connected inputs (including schedule and resets)."""
    n_in = sum(
        1
        for port_key, _, _ in self._iter_visible_inputs(node)
        if not _is_selector_input_port(node, port_key)
    )
    return n_in + len(self._visible_resets(node))

  def _collect_collapsed_box_ports(
      self, box_pfx: str, members: Sequence[graph_lib.Node]
  ) -> list[tuple[bool, istructure.IStruct, graph_lib.Node]]:
    """Collects deduplicated external `(is_reset, istruct, member)` ports."""
    stream_names = self.ctx.stream_names
    seen_inputs: set[istructure.IStruct] = set()
    seen_resets: set[istructure.IStruct] = set()
    ports: list[tuple[bool, istructure.IStruct, graph_lib.Node]] = []
    for member in members:
      if member.op == 'once_stream':
        continue
      groups = [(False, member.inputs, seen_inputs)]
      if self.opts.show_reset_edges and member.is_stateful:
        groups.append((True, member.resets, seen_resets))
      for is_reset, handles, seen in groups:
        for handle in handles:
          istruct = handle.istruct
          if istruct in seen or _is_unnamed_scalar_constant(
              istruct, self.graph, stream_names
          ):
            continue
          prod = self.graph.producer(istruct)
          if prod is not None and self.collapsed_owner.get(prod.id) == box_pfx:
            continue
          seen.add(istruct)
          ports.append((is_reset, istruct, member))
    return ports

  def _collapsed_box_for_prod(
      self, prod: graph_lib.Node | None, is_inline: bool
  ) -> tuple[str | None, str | None]:
    """Returns `(collapsed_pfx, box_dot)` when `prod` lives inside a box."""
    if prod is None or is_inline:
      return None, None
    pfx = self.collapsed_owner.get(prod.id)
    if pfx is None:
      return None, None
    return pfx, self.box_dot_by_prefix.get(pfx)

  def _record_net_ref(
      self,
      istruct: istructure.IStruct,
      prod: graph_lib.Node | None,
      prod_box_dot: str | None,
  ) -> None:
    if prod is None:
      return
    src_key = prod_box_dot if prod_box_dot is not None else f'n_{prod.id}'
    src_streams = self.net_ref_by_source.setdefault(src_key, {})
    src_streams[istruct] = src_streams.get(istruct, 0) + 1

  def _record_direct_out(self, src_ep: str, dst_key: str) -> None:
    src_key = src_ep.split(':', 1)[0]
    if src_key != dst_key:
      self.direct_out_producers.add(src_key)

  def _emit_net_ref_endpoint(
      self,
      container: pydot.Dot | pydot.Cluster,
      *,
      ref_dot: str,
      istruct: istructure.IStruct,
      prod: graph_lib.Node | None,
      palette: tuple[str, str, str],
      collapsed_pfx: str | None = None,
      collapsed_box_dot: str | None = None,
      register_fsm_box: bool = True,
  ) -> str:
    """Adds a local schematic net-reference badge and returns its endpoint."""
    ref_dom = f'psyre_{ref_dot}'
    stream_names = self.ctx.stream_names
    ref_label = _net_ref_label(istruct, prod, stream_names, collapsed_pfx)
    s_clk = str(self.ctx.clocks.stream_clock(istruct))
    if collapsed_pfx and collapsed_box_dot:
      net_key = collapsed_box_dot
      src_desc = collapsed_pfx
      if register_fsm_box:
        self.meta.fsm_box_by_dom_id[ref_dom] = collapsed_pfx
    elif prod is not None:
      net_key = f'n_{prod.id}'
      src_desc = f'{_node_full_display_name(prod)}#{prod.id}'
    else:
      net_key = f'in_{istruct.id}'
      src_desc = stream_names.get(istruct) or f'S_{istruct.id}'
    self.meta.info_by_dom_id[ref_dom] = (
        f'net_ref={src_desc} | stream=S_{istruct.id} | clock={s_clk}'
    )
    self.meta.net_by_dom_id[ref_dom] = net_key
    container.add_node(
        _make_terminal_dot_node(
            ref_dot, ref_dom, ref_label, palette, compact=True
        )
    )
    return f'{ref_dot}:{self.out_port}'

  def _resolve_source(
      self,
      local_container: pydot.Dot | pydot.Cluster,
      istruct: istructure.IStruct,
      tag_suffix: str,
      *,
      show_constants: bool | None = None,
  ) -> str | None:
    """Returns the direct DOT source endpoint (`node:port`) for `istruct`."""
    stream_names = self.ctx.stream_names
    clocks = self.ctx.clocks
    rankdir = self.opts.rankdir
    eff_show_consts = (
        self.opts.show_constants if show_constants is None else show_constants
    )
    prod = self.graph.producer(istruct)
    if prod is not None and prod.op == 'once_stream':
      c_dot = f'c_{tag_suffix}'
      c_dom = f'psyre_{c_dot}'
      subscript = _once_stream_k_label(prod, self.graph, stream_names)
      c_tip = _node_tooltip(prod, clocks, stream_names, False)
      text_color = gui.STYLE_STATEFUL[2]
      local_container.add_node(
          _make_static_stream_dot_node(
              c_dot,
              c_dom,
              f'δ_{subscript}',
              text_color,
              fontsize='12.5',
              min_width='0.44',
          )
      )
      self.meta.shape_specs[c_dom] = gui.NodeShapeSpec(
          dom_id=c_dom,
          kind='once_stream',
          rankdir=rankdir,
          border_color=text_color,
          info=c_tip,
          subscript=subscript,
      )
      self.meta.info_by_dom_id[c_dom] = c_tip
      self.meta.net_by_dom_id[c_dom] = f'n_{prod.id}'
      return c_dot
    if prod is not None:
      owner_pfx = self.collapsed_owner.get(prod.id)
      if owner_pfx is not None:
        return f'{self.box_dot_by_prefix[owner_pfx]}:{self.out_port}'
      return f'n_{prod.id}:{self.out_port}'
    if istruct.static:
      is_unnamed = _is_unnamed_scalar_constant(
          istruct, self.graph, stream_names
      )
      if is_unnamed and not eff_show_consts:
        return None
      c_dot = f'c_{tag_suffix}'
      c_dom = f'psyre_{c_dot}'
      s_name = stream_names.get(istruct, '')
      seq_str = _format_static_stream(istruct)
      c_label = s_name or seq_str or f'S_{istruct.id}'
      s_clk = str(clocks.stream_clock(istruct))
      if is_unnamed:
        c_tip = f'constant={c_label} | stream=S_{istruct.id} | clock={s_clk}'
        text_color = gui.STYLE_CONST[2]
      else:
        c_tip = (
            f'static_stream={c_label} | stream=S_{istruct.id} | clock={s_clk}'
        )
        if s_name and seq_str:
          c_tip = f'{c_tip} | values={seq_str}'
        text_color = gui.STYLE_INPUT[2]
      local_container.add_node(
          _make_static_stream_dot_node(c_dot, c_dom, c_label, text_color)
      )
      self.meta.shape_specs[c_dom] = gui.NodeShapeSpec(
          dom_id=c_dom,
          kind='static_stream',
          rankdir=rankdir,
          border_color=text_color,
          info=c_tip,
      )
      self.meta.info_by_dom_id[c_dom] = c_tip
      self.meta.net_by_dom_id[c_dom] = f'in_{istruct.id}'
      return c_dot
    src_dot = self.ext_sources.get(istruct)
    if src_dot is None:
      src_dot = f'in_{istruct.id}'
      src_dom = f'psyre_{src_dot}'
      s_label = stream_names.get(istruct, '') or f'S_{istruct.id}'
      s_clk = str(clocks.stream_clock(istruct))
      self.dot.add_node(
          _make_terminal_dot_node(src_dot, src_dom, s_label, gui.STYLE_INPUT)
      )
      self.ext_sources[istruct] = src_dot
      self.meta.info_by_dom_id[src_dom] = (
          f'input={s_label} | stream=S_{istruct.id} | clock={s_clk}'
      )
      self.meta.net_by_dom_id[src_dom] = src_dot
    return f'{src_dot}:{self.out_port}'

  def _add_reset_edge(
      self,
      parent: pydot.Dot | pydot.Cluster,
      src_ep: str,
      dst_ep: str,
      e_dom: str,
      info_text: str,
      *,
      is_inline: bool,
      constraint: str,
  ) -> None:
    """Emits a dotted `rst` edge and registers its DOM metadata."""
    if is_inline:
      self.meta.static_edge_to_node[e_dom] = f"psyre_{src_ep.split(':', 1)[0]}"
    self.meta.info_by_dom_id[e_dom] = info_text
    parent.add_edge(
        pydot.Edge(
            src_ep,
            dst_ep,
            id=_quote_dot(e_dom),
            color=_quote_dot('#E11D48'),
            style=_quote_dot('dotted'),
            penwidth='1.2',
            arrowsize='0.6',
            constraint=constraint,
            label=_quote_dot(' rst '),
            fontname=_quote_dot('Helvetica'),
            fontsize='8.5',
            fontcolor=_quote_dot('#BE123C'),
        )
    )

  def _classify_input_edge(
      self,
      node: graph_lib.Node,
      port_key: int | str,
      istruct: istructure.IStruct,
  ) -> _InputEdgeInfo:
    """Classifies routing, cycle, cluster, and net-ref flags for an input."""
    is_delayed = isinstance(port_key, int) and (
        (node.op == 'fby' and port_key >= 1) or node.op == 'pre'
    )
    prod = self.graph.producer(istruct)
    is_inline_src = _is_inline_source(istruct, prod)
    prod_collapsed_pfx, prod_box_dot = self._collapsed_box_for_prod(
        prod, is_inline_src
    )
    on_cycle = (
        not is_delayed
        and not is_inline_src
        and prod is not None
        and prod_collapsed_pfx is None
        and (prod.id, node.id) in self.ctx.cycle_edges
    )
    node_cont = self.node_containers[node.id]
    same_cluster = False
    if prod is not None and not is_inline_src:
      src_cont = (
          self.box_container_by_prefix[prod_collapsed_pfx]
          if prod_collapsed_pfx is not None
          else self.node_containers.get(prod.id, self.dot)
      )
      same_cluster = src_cont is node_cont
    is_fby_prod = (
        prod is not None
        and not is_inline_src
        and prod.op in _FBY_LIKE_OPS
        and prod_collapsed_pfx is None
    )
    is_fby_immediate = (
        is_fby_prod
        and same_cluster
        and prod is not None
        and self.fby_primary_consumer.get(prod.id) == node.id
    )
    use_net_ref = _should_use_net_ref(
        node,
        port_key,
        istruct,
        prod,
        is_delayed=is_delayed,
        is_inline_src=is_inline_src,
        on_cycle=on_cycle,
        is_fby_prod=is_fby_prod,
        is_fby_immediate_edge=is_fby_immediate,
        same_cluster=same_cluster,
        node_in_cluster=node_cont is not self.dot,
        large_graph=self.large_graph,
        stream_fanout=self.ctx.stream_fanout,
    )
    return _InputEdgeInfo(
        prod=prod,
        is_inline_src=is_inline_src,
        is_delayed=is_delayed,
        prod_collapsed_pfx=prod_collapsed_pfx,
        prod_box_dot=prod_box_dot,
        on_cycle=on_cycle,
        same_cluster=same_cluster,
        is_fby_immediate=is_fby_immediate,
        use_net_ref=use_net_ref,
    )

  def _init_clusters(self) -> None:
    """Materializes expanded automaton and subclock clusters."""
    ctx = self.ctx
    for pfx in ctx.fsm_prefixes:
      parts = pfx.split('/')
      if all(
          '/'.join(parts[:d]) in self.expanded_set
          for d in range(1, len(parts) + 1)
      ):
        self._get_container(pfx, ctx.base_act[pfx], ctx.sample_by_pfx[pfx])
    for node in sorted(
        self.visible_nodes,
        key=lambda n: (
            len(_node_automaton_prefix(n).split('/'))
            if _node_automaton_prefix(n)
            else 0,
            len(_activity_key(n, ctx.clocks)),
        ),
    ):
      self._get_container(
          _node_automaton_prefix(node), _activity_key(node, ctx.clocks), node
      )

  def _add_collapsed_boxes(self) -> None:
    """Adds summary box nodes for collapsed automata."""
    ctx = self.ctx
    rankdir = self.opts.rankdir
    for pfx in self.active_collapsed:
      fsm_idx = ctx.fsm_prefixes.index(pfx)
      box_dot = f'fsm_box_{fsm_idx}'
      box_dom = f'psyre_{box_dot}'
      members = ctx.descendant_nodes[pfx]
      sample = ctx.sample_by_pfx[pfx]
      b_act = ctx.base_act[pfx]
      fsm_clk = str(ctx.clocks.node_activity_clocks(sample)[len(b_act)])
      box_container = self._get_container(ctx.parent_fsm[pfx], b_act, sample)
      ports = self._collect_collapsed_box_ports(pfx, members)
      self.box_ports_by_prefix[pfx] = ports
      box_left_in = len(ports)
      use_box_table = rankdir == 'LR' and box_left_in >= 2
      box_container.add_node(
          pydot.Node(
              box_dot,
              id=_quote_dot(box_dom),
              label=_format_dot_label(
                  f'⊞ {pfx}\n({len(members)} nodes)',
                  box_left_in,
                  rankdir,
                  min_width_in=1.10,
                  min_height_in=max(0.52, 0.22 * (box_left_in - 1) + 0.28),
                  pad_pt=10,
              ),
              shape='box',
              peripheries='2',
              style=_quote_dot('rounded,filled'),
              fillcolor=_quote_dot('#EEF2FF'),
              color=_quote_dot('#4F46E5'),
              fontcolor=_quote_dot('#1E1B4B'),
              fontname=_quote_dot('Helvetica-Bold'),
              fontsize='11',
              penwidth='1.8',
              width='0' if use_box_table else '1.10',
              height='0' if use_box_table else '0.52',
              margin=_quote_dot('0,0' if use_box_table else '0.16,0.10'),
          )
      )
      self.box_dot_by_prefix[pfx] = box_dot
      self.box_container_by_prefix[pfx] = box_container
      self.meta.fsm_box_by_dom_id[box_dom] = pfx
      self.meta.net_by_dom_id[box_dom] = box_dot
      self.meta.info_by_dom_id[box_dom] = (
          f'automaton={pfx} | nodes={len(members)} | '
          f'clock={fsm_clk} | Click to expand'
      )

  def _add_visible_nodes(self) -> None:
    """Adds primitive pydot.Nodes to their enclosing clusters."""
    ctx = self.ctx
    for node in self.visible_nodes:
      num_left_in = self._count_left_inputs(node)
      self.left_inputs_by_id[node.id] = num_left_in
      p_node, spec = _make_pydot_op_node(
          node,
          target_graph=self.graph,
          rankdir=self.opts.rankdir,
          clocks=ctx.clocks,
          stream_names=ctx.stream_names,
          in_cycle=node.id in ctx.cycle_ids,
          num_left_inputs=num_left_in,
      )
      self.meta.shape_specs[spec.dom_id] = spec
      self.meta.info_by_dom_id[spec.dom_id] = spec.info
      self.meta.net_by_dom_id[spec.dom_id] = f'n_{node.id}'
      container = self._get_container(
          _node_automaton_prefix(node), _activity_key(node, ctx.clocks), node
      )
      self.node_containers[node.id] = container
      self.pydot_nodes_by_id[node.id] = p_node
      if _has_custom_node_name(node):
        self.same_name_groups.setdefault((id(container), node.name), []).append(
            node
        )
      container.add_node(p_node)

  def _add_same_name_chains(self) -> None:
    """Aligns same-named stateful/operator nodes with invisible rank chains."""
    for (_, same_name), group_nodes in self.same_name_groups.items():
      if len(group_nodes) < 2:
        continue
      group_attr = _quote_dot(f'same_name_{same_name}')
      for g_node in group_nodes:
        self.pydot_nodes_by_id[g_node.id].set('group', group_attr)
      container = self.node_containers[group_nodes[0].id]
      for u_prev, u_curr in zip(group_nodes, group_nodes[1:]):
        e_dom = f'psyre_e_same_name_{u_prev.id}_{u_curr.id}'
        self.constrained_adj.setdefault(f'n_{u_prev.id}', set()).add(
            f'n_{u_curr.id}'
        )
        container.add_edge(
            pydot.Edge(
                f'n_{u_prev.id}',
                f'n_{u_curr.id}',
                id=_quote_dot(e_dom),
                style=_quote_dot('invis'),
                constraint='true',
                weight='20',
            )
        )

  def _compute_fby_primary_consumers(self) -> None:
    """Finds the first same-cluster forward consumer for each `fby`/`pre`."""
    for u_node in self.visible_nodes:
      if u_node.op not in _FBY_LIKE_OPS:
        continue
      u_cont = self.node_containers.get(u_node.id)
      fwd = [
          v.id
          for v in self.visible_nodes
          if v.id > u_node.id
          and self.node_containers.get(v.id) is u_cont
          and (u_node.id, v.id) not in self.ctx.cycle_edges
          and any(h.istruct is u_node.output_stream for h in v.inputs)
      ]
      if fwd:
        self.fby_primary_consumer[u_node.id] = min(fwd)

  def _add_collapsed_box_edges(self) -> None:
    """Adds external data and reset edges feeding collapsed automaton boxes."""
    for pfx in self.active_collapsed:
      box_dot = self.box_dot_by_prefix[pfx]
      box_container = self.box_container_by_prefix[pfx]
      ports = self.box_ports_by_prefix[pfx]
      use_left_ports = self.opts.rankdir == 'LR' and len(ports) >= 2
      left_idx = 0
      in_idx = 0
      rst_idx = 0
      for is_reset, istruct, member in ports:
        prod = self.graph.producer(istruct)
        is_inline = _is_inline_source(istruct, prod)
        if is_reset:
          rst_idx += 1
          suffix = f'{box_dot}_rst{rst_idx}'
        else:
          suffix = f'{box_dot}_in{in_idx}'
        src_ep = self._resolve_source(box_container, istruct, suffix)
        if src_ep is None:
          continue
        src_key = src_ep.split(':', 1)[0]
        self._record_direct_out(src_ep, box_dot)
        if use_left_ports:
          dst_ep = f'{box_dot}:in{left_idx}:w'
          left_idx += 1
        else:
          dst_ep = f'{box_dot}:{self.in_port}'
        edge_parent = box_container if is_inline else self.dot
        if is_reset:
          self._add_reset_edge(
              edge_parent,
              src_ep,
              dst_ep,
              f'psyre_e_{box_dot}_rst{rst_idx}',
              f'reset -> {pfx} | stream=S_{istruct.id}',
              is_inline=is_inline,
              constraint='true' if is_inline else 'false',
          )
        else:
          e_dom = f'psyre_e_{box_dot}_in{in_idx}'
          in_idx += 1
          s_clk = str(self.ctx.clocks.stream_clock(istruct))
          self.meta.info_by_dom_id[e_dom] = (
              f'edge -> {pfx} | stream=S_{istruct.id} | clock={s_clk}'
          )
          if is_inline:
            self.meta.static_edge_to_node[e_dom] = f'psyre_{src_key}'
          is_back = not is_inline and prod is not None and prod.id > member.id
          if not is_back and not is_inline and src_key != box_dot:
            self.constrained_adj.setdefault(src_key, set()).add(box_dot)
          edge_parent.add_edge(
              pydot.Edge(
                  src_ep,
                  dst_ep,
                  id=_quote_dot(e_dom),
                  color=_quote_dot('#7C3AED' if is_back else '#334155'),
                  style=_quote_dot('dashed' if is_back else 'solid'),
                  penwidth='1.5' if is_back else '1.4',
                  arrowsize='0.65' if is_inline else '0.7',
                  constraint='false' if is_back else 'true',
              )
          )

  def _prepopulate_constrained_adj(self) -> None:
    """Seeds constrained forward edges so feedback paths orient cleanly."""
    for node in self.visible_nodes:
      dst_key = f'n_{node.id}'
      for port_key, istruct, _ in self._iter_visible_inputs(node):
        info = self._classify_input_edge(node, port_key, istruct)
        if (
            info.prod is None
            or info.is_inline_src
            or info.on_cycle
            or info.use_net_ref
            or (info.is_delayed and not info.is_fby_immediate)
        ):
          continue
        src_key = info.prod_box_dot or f'n_{info.prod.id}'
        if src_key != dst_key:
          self.constrained_adj.setdefault(src_key, set()).add(dst_key)

  def _add_visible_node_edges(self) -> None:
    """Adds data, delayed, schedule, and reset edges feeding visible nodes."""
    for node in self.visible_nodes:
      self._add_edges_for_node(node)

  def _add_edges_for_node(self, node: graph_lib.Node) -> None:
    """Adds all visible incoming edges for a single `node`."""
    node_container = self.node_containers[node.id]
    num_left_inputs = self.left_inputs_by_id[node.id]
    dst_name = f'n_{node.id}'
    if node.op in {'merge', 'switch'}:
      arm_labels = _merge_arm_labels(node)
    elif node.op == 'ifthenelse':
      arm_labels = {1: 'T', 2: 'F'}
    elif node.op == 'when':
      sel_lbl = _when_selector_label(node)
      arm_labels = {0: sel_lbl} if sel_lbl else {}
    else:
      arm_labels = {}
    use_left_ports = self.opts.rankdir == 'LR' and num_left_inputs >= 2
    left_idx = 0

    def next_left_dst_ep() -> str:
      nonlocal left_idx
      if not use_left_ports:
        return f'{dst_name}:{self.in_port}'
      if node.op == 'ifthenelse' and num_left_inputs == 2:
        compass = 'nw' if left_idx == 0 else 'sw'
        left_idx += 1
        return f'{dst_name}:{compass}'
      ep = f'{dst_name}:in{left_idx}:w'
      left_idx += 1
      return ep

    for port_key, istruct, is_delay_sched in self._iter_visible_inputs(node):
      is_selector_port = _is_selector_input_port(node, port_key)
      info = self._classify_input_edge(node, port_key, istruct)
      prod_key = (
          (info.prod_box_dot or f'n_{info.prod.id}')
          if info.prod is not None
          else None
      )
      is_delayed_self_loop = (
          info.is_delayed and not info.is_inline_src and prod_key == dst_name
      )
      is_delayed_feedback = (
          info.is_delayed
          and not info.is_inline_src
          and not info.is_fby_immediate
          and info.prod is not None
          and prod_key is not None
          and not is_delayed_self_loop
          and (
              _can_reach(self.constrained_adj, dst_name, prod_key)
              or (
                  info.prod.id > node.id
                  and not _can_reach(self.constrained_adj, prod_key, dst_name)
              )
          )
      )
      is_back_edge = is_delayed_self_loop or is_delayed_feedback

      if info.use_net_ref:
        self._record_net_ref(istruct, info.prod, info.prod_box_dot)
        if info.prod is None:
          ref_palette = gui.STYLE_INPUT
        elif is_back_edge:
          ref_palette = gui.STYLE_FBY
        else:
          ref_palette = gui.STYLE_NET_REF
        src_ep = self._emit_net_ref_endpoint(
            node_container,
            ref_dot=f'ref_{node.id}_p{port_key}',
            istruct=istruct,
            prod=info.prod,
            palette=ref_palette,
            collapsed_pfx=info.prod_collapsed_pfx,
            collapsed_box_dot=info.prod_box_dot,
        )
      else:
        src_ep = self._resolve_source(
            node_container, istruct, f'{node.id}_p{port_key}'
        )
        if src_ep is None:
          continue
        self._record_direct_out(src_ep, dst_name)

      dst_ep = (
          f'{dst_name}:{self.sel_port}'
          if is_selector_port
          else next_left_dst_ep()
      )
      edge_label = (
          arm_labels[port_key]
          if (isinstance(port_key, int) and port_key in arm_labels)
          else ''
      )
      edge_color = '#334155'
      edge_style = 'solid'
      penwidth = '1.3'
      constraint = 'true'
      reverse_in_dot = False
      src_key = src_ep.split(':', 1)[0]
      if info.on_cycle:
        edge_color = '#DC2626'
        penwidth = '2.0'
      elif info.is_delayed:
        if is_back_edge:
          edge_color = '#7C3AED'
          edge_style = 'dashed'
          penwidth = '1.5'
        if (
            not info.use_net_ref
            and not info.is_inline_src
            and not info.is_fby_immediate
            and info.prod is not None
        ):
          if is_delayed_self_loop:
            constraint = 'false'
          elif is_delayed_feedback:
            # Emit feedback back-edges in DOT's forward rank direction with
            # dir="back" and constraint="true" so dot includes them in crossing
            # minimization and coordinate alignment instead of convoluted
            # unconstrained detours.
            reverse_in_dot = True
            self.constrained_adj.setdefault(dst_name, set()).add(src_key)
          else:
            self.constrained_adj.setdefault(src_key, set()).add(dst_name)
      elif is_selector_port:
        edge_color = '#B45309' if node.op in _SELECTOR_OPS else '#0F766E'

      e_dom = f'psyre_e_{node.id}_p{port_key}'
      s_clk = str(self.ctx.clocks.stream_clock(istruct))
      if is_delay_sched:
        role = 'p_stream'
      elif info.is_delayed:
        role = 'delayed_next'
      elif is_selector_port:
        role = 'selector'
      else:
        role = f'in[{port_key}]'
      self.meta.info_by_dom_id[e_dom] = (
          f'edge -> {_node_full_display_name(node)}.{role} | '
          f'stream=S_{istruct.id} | clock={s_clk}'
      )
      edge_attrs: dict[str, Any] = {
          'id': _quote_dot(e_dom),
          'color': _quote_dot(edge_color),
          'style': _quote_dot(edge_style),
          'penwidth': penwidth,
          'arrowsize': '0.65' if info.is_inline_src else '0.7',
          'constraint': constraint,
          'fontname': _quote_dot('Helvetica'),
          'fontsize': '9',
          'fontcolor': _quote_dot(edge_color),
      }
      if info.is_inline_src:
        self.meta.static_edge_to_node[e_dom] = f'psyre_{src_key}'
      elif info.is_fby_immediate:
        edge_attrs['weight'] = '10'
      elif reverse_in_dot:
        edge_attrs['dir'] = _quote_dot('back')
        edge_attrs['weight'] = '2'
      if edge_label:
        edge_attrs['label'] = _quote_dot(f' {edge_label} ')
      is_local = constraint == 'true' and (
          info.use_net_ref or info.is_inline_src or info.same_cluster
      )
      edge_parent = node_container if is_local else self.dot
      if reverse_in_dot:
        self.meta.reversed_edge_ids.add(e_dom)
        edge_parent.add_edge(pydot.Edge(dst_ep, src_ep, **edge_attrs))
      else:
        edge_parent.add_edge(pydot.Edge(src_ep, dst_ep, **edge_attrs))

    for rst_idx, rst_s in self._visible_resets(node):
      rst_prod = self.graph.producer(rst_s)
      rst_is_inline = _is_inline_source(rst_s, rst_prod)
      rst_collapsed_pfx, rst_box_dot = self._collapsed_box_for_prod(
          rst_prod, rst_is_inline
      )
      if self.large_graph and not rst_is_inline:
        self._record_net_ref(rst_s, rst_prod, rst_box_dot)
        src_ep = self._emit_net_ref_endpoint(
            node_container,
            ref_dot=f'ref_{node.id}_rst{rst_idx}',
            istruct=rst_s,
            prod=rst_prod,
            palette=gui.STYLE_RST_REF,
            collapsed_pfx=rst_collapsed_pfx,
            collapsed_box_dot=rst_box_dot,
            register_fsm_box=False,
        )
        rst_parent: pydot.Dot | pydot.Cluster = node_container
        rst_constraint = 'true'
      else:
        src_ep = self._resolve_source(
            node_container, rst_s, f'{node.id}_rst{rst_idx}'
        )
        if src_ep is None:
          continue
        self._record_direct_out(src_ep, dst_name)
        rst_parent = node_container if rst_is_inline else self.dot
        rst_constraint = 'true' if rst_is_inline else 'false'
      self._add_reset_edge(
          rst_parent,
          src_ep,
          next_left_dst_ep(),
          f'psyre_e_{node.id}_rst{rst_idx}',
          f'reset[{rst_idx}] -> {_node_full_display_name(node)} | '
          f'stream=S_{rst_s.id}',
          is_inline=rst_is_inline,
          constraint=rst_constraint,
      )

  def _add_output_terminals(self) -> None:
    """Adds graph output terminal badges and incoming edges."""
    ctx = self.ctx
    for out_idx, (out_label, out_s) in enumerate(
        _iter_output_paths(self.graph.output, stream_names=ctx.stream_names)
    ):
      out_dot = f'out_{out_idx}'
      out_dom = f'psyre_{out_dot}'
      out_clk = str(ctx.clocks.stream_clock(out_s))
      out_tip = f'output={out_label} | stream=S_{out_s.id} | clock={out_clk}'
      self.dot.add_node(
          _make_terminal_dot_node(out_dot, out_dom, out_label, gui.STYLE_OUTPUT)
      )
      self.meta.info_by_dom_id[out_dom] = out_tip
      prod = self.graph.producer(out_s)
      is_inline_out = _is_inline_source(out_s, prod)
      if prod is not None and not is_inline_out:
        prod_collapsed_pfx, prod_box_dot = self._collapsed_box_for_prod(
            prod, False
        )
        if self.large_graph and (
            (self.node_containers.get(prod.id, self.dot) is not self.dot)
            or prod_collapsed_pfx is not None
            or ctx.stream_fanout.get(out_s, 0) > 8
        ):
          self._record_net_ref(out_s, prod, prod_box_dot)
          src_ep = self._emit_net_ref_endpoint(
              self.dot,
              ref_dot=f'ref_out_{out_idx}',
              istruct=out_s,
              prod=prod,
              palette=gui.STYLE_NET_REF,
              collapsed_pfx=prod_collapsed_pfx,
              collapsed_box_dot=prod_box_dot,
          )
        else:
          src_key = prod_box_dot if prod_box_dot is not None else f'n_{prod.id}'
          self.direct_out_producers.add(src_key)
          src_ep = f'{src_key}:{self.out_port}'
      else:
        src_ep = self._resolve_source(
            self.dot, out_s, f'out_{out_idx}', show_constants=True
        )
        if src_ep is None:
          continue

      e_dom = f'psyre_e_out_{out_idx}'
      if is_inline_out:
        self.meta.static_edge_to_node[e_dom] = (
            f"psyre_{src_ep.split(':', 1)[0]}"
        )
      self.meta.info_by_dom_id[e_dom] = out_tip
      self.dot.add_edge(
          pydot.Edge(
              src_ep,
              f'{out_dot}:{self.in_port}',
              id=_quote_dot(e_dom),
              color=_quote_dot('#2563EB'),
              penwidth='1.4',
              arrowsize='0.65' if is_inline_out else '0.7',
          )
      )

  def _add_net_out_badge(
      self,
      container: pydot.Dot | pydot.Cluster,
      *,
      src_key: str,
      out_ref_dot: str,
      ref_label: str,
      palette: tuple[str, str, str],
      out_tip: str,
      edge_color: str,
      penwidth: str,
      fsm_pfx: str | None = None,
  ) -> None:
    """Attaches a net-reference fanout badge to a producer node or box."""
    out_ref_dom = f'psyre_{out_ref_dot}'
    container.add_node(
        _make_terminal_dot_node(
            out_ref_dot, out_ref_dom, ref_label, palette, compact=True
        )
    )
    self.meta.info_by_dom_id[out_ref_dom] = out_tip
    self.meta.net_by_dom_id[out_ref_dom] = src_key
    if fsm_pfx is not None:
      self.meta.fsm_box_by_dom_id[out_ref_dom] = fsm_pfx
    e_dom = f'psyre_e_{out_ref_dot}'
    self.meta.info_by_dom_id[e_dom] = out_tip
    container.add_edge(
        pydot.Edge(
            f'{src_key}:{self.out_port}',
            f'{out_ref_dot}:{self.in_port}',
            id=_quote_dot(e_dom),
            color=_quote_dot(edge_color),
            penwidth=penwidth,
            arrowsize='0.65',
        )
    )

  def _add_net_out_badges(self) -> None:
    """Adds net-reference output badges for nodes/boxes routed only via refs."""
    ctx = self.ctx
    for node in self.visible_nodes:
      src_key = f'n_{node.id}'
      if (
          src_key in self.direct_out_producers
          or src_key not in self.net_ref_by_source
      ):
        continue
      out_s = node.output_stream
      fanout_cnt = self.net_ref_by_source[src_key].get(out_s, 1)
      base = ctx.stream_names.get(out_s, '') or _node_display_name(node)
      ref_label = (
          f'↪ {base}#{node.id}'
          if fanout_cnt <= 1
          else f'↪ {base}#{node.id} (×{fanout_cnt})'
      )
      is_fby = node.op in _FBY_LIKE_OPS
      s_clk = str(ctx.clocks.stream_clock(out_s))
      self._add_net_out_badge(
          self.node_containers[node.id],
          src_key=src_key,
          out_ref_dot=f'out_ref_{node.id}',
          ref_label=ref_label,
          palette=gui.STYLE_FBY if is_fby else gui.STYLE_NET_REF,
          out_tip=(
              f'net_out={_node_full_display_name(node)}#{node.id} | '
              f'fanout={fanout_cnt} | stream=S_{out_s.id} | clock={s_clk}'
          ),
          edge_color='#7C3AED' if is_fby else '#475569',
          penwidth='1.3',
      )

    for pfx in self.active_collapsed:
      box_dot = self.box_dot_by_prefix[pfx]
      if (
          box_dot in self.direct_out_producers
          or box_dot not in self.net_ref_by_source
      ):
        continue
      streams_dict = self.net_ref_by_source[box_dot]
      total_refs = sum(streams_dict.values())
      num_streams = len(streams_dict)
      ref_label = (
          f'↪ {pfx} ({num_streams} nets, ×{total_refs})'
          if num_streams > 1
          else f'↪ {pfx} (×{total_refs})'
      )
      self._add_net_out_badge(
          self.box_container_by_prefix[pfx],
          src_key=box_dot,
          out_ref_dot=f'out_ref_{box_dot}',
          ref_label=ref_label,
          palette=gui.STYLE_NET_REF,
          out_tip=(
              f'net_out={pfx} | nets={num_streams} | '
              f'consumers={total_refs} | Click to expand'
          ),
          edge_color='#4F46E5',
          penwidth='1.4',
          fsm_pfx=pfx,
      )


def _build_dot_and_metadata(
    target_graph: graph_lib.Graph,
    opts: _LayoutOptions,
    *,
    viz_ctx: _GraphVizContext | None = None,
) -> tuple[pydot.Dot, _DotMetadata]:
  """Constructs `pydot.Dot` and DOM metadata maps for `target_graph`."""
  ctx = _analyze_graph(target_graph) if viz_ctx is None else viz_ctx
  return _DotBuilder(target_graph, ctx, opts).build()


def build_pydot(
    target_graph: graph_lib.Graph,
    *,
    rankdir: RankDir = 'LR',
    show_constants: bool = True,
    group_by_activity: bool = True,
    show_activity_edges: bool = False,
    show_reset_edges: bool = True,
    collapse_automata: bool = False,
    expanded_automata: Iterable[str] | None = None,
) -> pydot.Dot:
  """Builds the `pydot.Dot` representation of a Stage-1 finalized Psyre `Graph`.

  Args:
    target_graph: Finalized Stage-1 Psyre `Graph`.
    rankdir: Layout flow direction (`'LR'` left-to-right or `'TB'`
      top-to-bottom).
    show_constants: Whether to display unnamed scalar constant inputs.
    group_by_activity: Whether to group nodes sharing subclock steps into dashed
      synchronous clock clusters.
    show_activity_edges: Deprecated compatibility flag (unused).
    show_reset_edges: Whether to draw dotted `rst` control wires on stateful
      nodes.
    collapse_automata: When True and `expanded_automata` is None, collapses all
      automata into summary boxes.
    expanded_automata: Optional explicit iterable of automaton prefixes to
      render expanded.

  Returns:
    A configured `pydot.Dot` instance ready for Graphviz layout.
  """
  del show_activity_edges
  dot, _ = _build_dot_and_metadata(
      target_graph,
      _LayoutOptions(
          rankdir=rankdir,
          show_constants=show_constants,
          group_by_activity=group_by_activity,
          show_reset_edges=show_reset_edges,
          collapse_automata=collapse_automata,
          expanded_automata=expanded_automata,
      ),
  )
  return dot


def _build_schematic_view(
    target_graph: graph_lib.Graph,
    opts: _LayoutOptions,
    *,
    prog: str,
    viz_ctx: _GraphVizContext | None = None,
) -> gui.SchematicView:
  """Builds a `gui.SchematicView` by running Graphviz `dot -Tsvg`."""
  dot, meta = _build_dot_and_metadata(target_graph, opts, viz_ctx=viz_ctx)
  dot_src = dot.to_string()
  return meta.to_schematic_view(_run_dot_svg(dot_src, prog), dot_src)


def _canonical_state_key(expanded: Iterable[str]) -> str:
  """Returns the canonical `|`-joined state key for an expanded prefix set."""
  return '|'.join(sorted(set(expanded)))


def _build_schematic_bundle(
    target_graph: graph_lib.Graph,
    opts: _LayoutOptions,
    *,
    title: str,
    height_px: int,
    dot_prog: str | None,
) -> gui.SchematicBundle:
  """Builds a `gui.SchematicBundle` covering all automaton expansion states."""
  viz_ctx = _analyze_graph(target_graph)
  fsm_prefixes = viz_ctx.fsm_prefixes
  prog = _get_dot_binary(dot_prog)
  sem = (
      target_graph.semantics.name
      if target_graph.semantics is not None
      else 'Psyre'
  )
  if not fsm_prefixes:
    view = _build_schematic_view(
        target_graph,
        dataclasses.replace(
            opts, collapse_automata=False, expanded_automata=None
        ),
        prog=prog,
        viz_ctx=viz_ctx,
    )
    return gui.SchematicBundle(
        views={'': view},
        initial_view_key='',
        fsm_prefixes=(),
        title=title,
        semantics_label=sem,
        node_count=len(target_graph.nodes),
        height_px=height_px,
    )

  valid_states = _valid_expansion_states(fsm_prefixes)
  if opts.expanded_automata is not None:
    req = set(opts.expanded_automata)
    fsm_set = set(fsm_prefixes)
    closed_req: set[str] = set()
    for pfx in fsm_prefixes:
      if pfx in req:
        parts = pfx.split('/')
        for i in range(1, len(parts) + 1):
          anc = '/'.join(parts[:i])
          if anc in fsm_set:
            closed_req.add(anc)
    initial_key = _canonical_state_key(closed_req)
  elif opts.collapse_automata:
    initial_key = ''
  else:
    initial_key = _canonical_state_key(fsm_prefixes)

  def render_state(state: frozenset[str]) -> tuple[str, gui.SchematicView]:
    return (
        _canonical_state_key(state),
        _build_schematic_view(
            target_graph,
            dataclasses.replace(
                opts, collapse_automata=True, expanded_automata=state
            ),
            prog=prog,
            viz_ctx=viz_ctx,
        ),
    )

  if len(valid_states) > 1:
    max_workers = min(8, len(valid_states), os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      views = dict(executor.map(render_state, valid_states))
  else:
    views = dict(render_state(state) for state in valid_states)
  if initial_key not in views:
    initial_key = ''
  return gui.SchematicBundle(
      views=views,
      initial_view_key=initial_key,
      fsm_prefixes=fsm_prefixes,
      title=title,
      semantics_label=sem,
      node_count=len(target_graph.nodes),
      height_px=height_px,
  )


def render_svg(
    target_graph: graph_lib.Graph,
    *,
    rankdir: RankDir = 'LR',
    show_constants: bool = True,
    group_by_activity: bool = True,
    show_activity_edges: bool = False,
    show_reset_edges: bool = True,
    collapse_automata: bool = False,
    expanded_automata: Iterable[str] | None = None,
    dot_prog: str | None = None,
) -> str:
  """Renders a Stage-1 Psyre `Graph` as a custom schematic SVG string."""
  del show_activity_edges
  view = _build_schematic_view(
      target_graph,
      _LayoutOptions(
          rankdir=rankdir,
          show_constants=show_constants,
          group_by_activity=group_by_activity,
          show_reset_edges=show_reset_edges,
          collapse_automata=collapse_automata,
          expanded_automata=expanded_automata,
      ),
      prog=_get_dot_binary(dot_prog),
  )
  return gui.render_view_svg(view)


def render_html(
    target_graph: graph_lib.Graph,
    *,
    title: str = '',
    height_px: int = 560,
    rankdir: RankDir = 'LR',
    show_constants: bool = True,
    group_by_activity: bool = True,
    show_activity_edges: bool = False,
    show_reset_edges: bool = True,
    collapse_automata: bool = True,
    expanded_automata: Iterable[str] | None = None,
    dot_prog: str | None = None,
) -> str:
  """Renders a Stage-1 Psyre `Graph` as a self-contained interactive HTML viewer."""
  del show_activity_edges
  bundle = _build_schematic_bundle(
      target_graph,
      _LayoutOptions(
          rankdir=rankdir,
          show_constants=show_constants,
          group_by_activity=group_by_activity,
          show_reset_edges=show_reset_edges,
          collapse_automata=collapse_automata,
          expanded_automata=expanded_automata,
      ),
      title=title,
      height_px=height_px,
      dot_prog=dot_prog,
  )
  return gui.render_bundle_html(bundle)


def visualize_graph(
    target_graph: graph_lib.Graph,
    *,
    title: str = '',
    height_px: int = 560,
    rankdir: RankDir = 'LR',
    show_constants: bool = True,
    group_by_activity: bool = True,
    show_activity_edges: bool = False,
    show_reset_edges: bool = True,
    collapse_automata: bool = True,
    expanded_automata: Iterable[str] | None = None,
    dot_prog: str | None = None,
) -> GraphVisualization:
  """Builds an interactive `GraphVisualization` for a Stage-1 Psyre `Graph`."""
  del show_activity_edges
  bundle = _build_schematic_bundle(
      target_graph,
      _LayoutOptions(
          rankdir=rankdir,
          show_constants=show_constants,
          group_by_activity=group_by_activity,
          show_reset_edges=show_reset_edges,
          collapse_automata=collapse_automata,
          expanded_automata=expanded_automata,
      ),
      title=title,
      height_px=height_px,
      dot_prog=dot_prog,
  )
  return gui.build_visualization(bundle)
