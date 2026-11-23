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
"""Graphviz/DOT schematic layout builder for Psyre Graphs.

Clusters nodes hierarchically by automaton scope and Lustre-style `on` subclock
steps (Halbwachs et al., 1991), routing delayed feedback cycles via `dir=back`
and delegating custom glyph SVG post-processing to `psyre.gui`.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
import concurrent.futures
import dataclasses
import html
import itertools
import math
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
from psyre import stream as stream_lib
import pydot

RankDir = gui.RankDir
GraphVisualization = gui.GraphVisualization

_SELECTOR_OPS = ('when', 'whennot')
_MUX_OPS = ('merge', 'switch')
_FBY_LIKE_OPS = ('fby', 'pre')
_NET_REF_THRESHOLD = 80
_FONT = '"Helvetica"'
_SLUG_RE = re.compile(r'[^A-Za-z0-9_]+')

_BINARY_OP_GLYPHS: Mapping[str, str] = {
    'add': '+',
    'sub': '-',
    'mul': '×',
    'matmul': '@',
    'truediv': '/',
    'floordiv': '//',
    'mod': '%',
    'pow': '**',
    'lshift': '<<',
    'rshift': '>>',
    'and': '∧',
    'or': '∨',
    'xor': '⊕',
}
_DUNDER_OP_GLYPHS: Mapping[str, str] = {
    **{f'__{op}__': glyph for op, glyph in _BINARY_OP_GLYPHS.items()},
    **{f'__r{op}__': glyph for op, glyph in _BINARY_OP_GLYPHS.items()},
    '__pos__': '+',
    '__neg__': '-',
    '__invert__': '¬',
    '__abs__': '|·|',
    '__lt__': '<',
    '__le__': '≤',
    '__eq__': '=',
    '__ne__': '≠',
    '__gt__': '>',
    '__ge__': '≥',
}
_DUNDER_GLYPH_SET = frozenset(_DUNDER_OP_GLYPHS.values())

# Auto-generated or alias node local names that differ from `node.op`.
_DEFAULT_NODE_LOCAL_NAMES = frozenset({
    'case',
    'last',
    'pnr',
    'pns',
    'state',
    'state_nr',
    'state_ns',
    'state_r',
    'state_s',
    'switch',
    'whennot',
})


_DOT_INSTALL_HINT = (
    'Install Graphviz so `dot` is on PATH (e.g. `apt install graphviz` or '
    '`brew install graphviz`), or pass `dot_prog=/path/to/dot`.'
)


def _get_dot_binary(dot_prog: str | None = None) -> str:
  """Resolves the Graphviz `dot` executable from runfiles or system PATH."""
  if dot_prog is not None:
    return dot_prog
  bundled = os.path.abspath(
      os.path.join(os.path.dirname(__file__), '..', '..', 'graphviz', 'dot')
  )
  if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
    return bundled
  found = shutil.which('dot')
  if found is None:
    raise FileNotFoundError(
        f'Graphviz `dot` executable not found on PATH. {_DOT_INSTALL_HINT}'
    )
  return found


def _run_dot_svg(dot_src: str, prog: str) -> str:
  """Runs Graphviz `dot -Tsvg` on `dot_src` and returns UTF-8 SVG markup."""
  try:
    proc = subprocess.run(
        [prog, '-Tsvg'],
        input=dot_src.encode('utf-8'),
        capture_output=True,
        check=False,
    )
  except FileNotFoundError as err:
    raise FileNotFoundError(
        f'Graphviz executable {prog!r} not found. {_DOT_INSTALL_HINT}'
    ) from err
  err_msg = proc.stderr.decode('utf-8', errors='replace').strip()
  has_svg = b'<svg' in proc.stdout and b'</svg>' in proc.stdout
  if not has_svg:
    raise RuntimeError(
        f'Graphviz dot failed (code {proc.returncode}): '
        f'{err_msg or "missing <svg> output"}'
    )
  if proc.returncode != 0:
    # Retain Borg/Colab noisy font warnings when dot still produced valid SVG.
    common.logger.warning(
        'Graphviz dot exited with code %d while emitting SVG: %s',
        proc.returncode,
        err_msg[:240],
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
  # Use an invisible zero-width left column of `<TD PORT="in{idx}">` cells
  # alongside a single ROWSPAN label cell so multi-input edges attach vertically
  # spaced on the west border instead of overlapping at one compass point.
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


def _join_truncated(items: Sequence[str], limit: int, sep: str) -> str:
  """Joins up to `limit` items, appending an ellipsis when truncated."""
  joined = sep.join(items[:limit])
  return f'{joined}{sep}…' if len(items) > limit else joined


def _slug(text: str) -> str:
  """Sanitizes `text` into a DOT-safe identifier fragment."""
  return _SLUG_RE.sub('_', text)


def _parent_prefix(pfx: str) -> str:
  """Returns the parent slash-separated scope prefix of `pfx`, or `''`."""
  return pfx.rsplit('/', 1)[0] if '/' in pfx else ''


def _ancestor_prefixes(pfx: str) -> list[str]:
  """Returns non-empty cumulative slash-separated prefixes of `pfx`."""
  if not pfx:
    return []
  parts = pfx.split('/')
  return ['/'.join(parts[:depth]) for depth in range(1, len(parts) + 1)]


def _format_act_clock(act_key: tuple[str, ...]) -> str:
  """Formats a concrete activity clock step tuple in Lustre `on` notation."""
  return f". {' '.join(act_key)}" if act_key else '.'


def _format_static_stream(stream: stream_lib.Stream) -> str:
  """Formats a static periodic Stream concisely for input badges."""
  spec = stream.static_sequence
  return '' if spec is None else spec.format_compact()


def _node_automaton_prefix(node: graph_lib.Node) -> str:
  """Returns the slash-separated automaton scope prefix of `node` (or '')."""
  return _parent_prefix(node.name)


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
) -> dict[int, str]:
  """Maps stream IDs to explicit user-assigned Stream names when present."""
  names: dict[int, str] = target_graph.explicit_stream_names()
  # Prefer an explicit stream alias distinct from the producer node's own name
  # so tooltips expose `stream=<alias>` alongside `name=<node.name>`.
  for stream in target_graph.streams:
    prod = target_graph.producer(stream)
    if prod is None:
      continue
    for s_name in target_graph.stream_names(stream):
      if s_name != prod.name and target_graph.is_explicit_stream_name(
          stream, s_name
      ):
        names[stream.id] = s_name
        break
  return names


def _is_unnamed_scalar_constant(
    stream: stream_lib.Stream,
    target_graph: graph_lib.Graph,
    stream_names: Mapping[int, str],
) -> bool:
  """Returns True for unnamed single-element static scalar constants."""
  return (
      target_graph.producer(stream) is None
      and stream.is_one_periodic
      and stream.id not in stream_names
  )


def _node_display_name(node: graph_lib.Node) -> str:
  """Returns the Stage-1 display name printed on a primitive Node."""
  key = _node_local_name(node) or node.op
  if node.is_static_stream() and key == 'static_stream':
    src = node.kwargs.get('source')
    if isinstance(src, stream_lib.Stream):
      return _format_static_stream(src)
  return _DUNDER_OP_GLYPHS.get(key, key)


def _node_full_display_name(node: graph_lib.Node) -> str:
  """Returns the full (automaton-prefixed) display name of `node`."""
  pfx = _node_automaton_prefix(node)
  local = _node_display_name(node)
  return f'{pfx}/{local}' if pfx else local


def _iter_output_paths(
    value: Any,
    prefix: str = 'out',
    stream_names: Mapping[int, str] | None = None,
) -> Iterator[tuple[str, stream_lib.Stream]]:
  """Yields `(label_path, Stream)` pairs for leaves of `graph.output`."""
  if value is None:
    return
  is_anon_prefix = bool(re.fullmatch(r'out(\[\d+\])*', prefix))
  if isinstance(value, stream_lib.Stream):
    if is_anon_prefix and stream_names and value.id in stream_names:
      yield stream_names[value.id].rsplit('/', 1)[-1], value
    else:
      yield prefix, value
    return
  if common.is_namedtuple(value):
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
    by_arm.setdefault(arm_idx, []).append(common.format_tag(tag, 12))
  labels = {
      arm_idx + 1: _join_truncated(tags, 2, ',')
      for arm_idx, tags in by_arm.items()
  }
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
      common.format_tag(val, 12)
      for _, val in sorted(selector.values, key=common.tag_sort_key)
  ]
  if not tags:
    return '*' if selector.negated else '∅'
  if len(tags) == 1:
    return f'≠{tags[0]}' if selector.negated else f'={tags[0]}'
  joined = _join_truncated(tags, 2, ',')
  return f'∉{{{joined}}}' if selector.negated else f'∈{{{joined}}}'


def _node_palette(node: graph_lib.Node, in_cycle: bool) -> tuple[str, str, str]:
  """Returns `(fill, border, text)` hex colors for a primitive Node."""
  if in_cycle:
    return gui.STYLE_CYCLE
  if node.is_primitive(*_SELECTOR_OPS):
    return gui.STYLE_SELECTOR
  if node.is_primitive(*_MUX_OPS):
    return gui.STYLE_MUX
  if node.is_primitive(*_FBY_LIKE_OPS, 'post'):
    return gui.STYLE_FBY
  if node.is_stateful:
    return gui.STYLE_STATEFUL
  return gui.STYLE_COMB


def _node_tooltip(
    node: graph_lib.Node,
    clocks: clock.ClockAnalysis,
    stream_names: Mapping[int, str],
    in_cycle: bool,
) -> str:
  """Builds a multi-field summary string for tooltips and node inspection."""
  disp_name = _node_display_name(node)
  full_name = _node_full_display_name(node)
  op_str = node.op
  clk_str = str(clocks.node_clock(node))
  out_name = stream_names.get(node.output.id, '')
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
  if node.is_primitive('when'):
    sel_lbl = _when_selector_label(node)
    if sel_lbl:
      parts.append(f'match={sel_lbl}')
  if node.is_static_stream():
    src = node.kwargs.get('source')
    if isinstance(src, stream_lib.Stream):
      parts.append(f'values={_format_static_stream(src)}')
  if node.resets:
    parts.append(f'resets={len(node.resets)}')
  if in_cycle:
    parts.append('CAUSALITY_CYCLE=true')
  return ' | '.join(parts)


def _is_selector_input_port(node: graph_lib.Node, port_idx: int) -> bool:
  """Returns True if `port_idx` is a top-connected selector/control input."""
  return port_idx == 0 and node.is_primitive(
      *_SELECTOR_OPS, *_MUX_OPS, 'ifthenelse'
  )


def _is_inline_source(
    stream: stream_lib.Stream,
    prod: graph_lib.Node | None,
) -> bool:
  """Returns True if `stream` renders as a re-instantiated inline arrow."""
  return prod is None and stream.static


def _make_pydot_op_node(
    node: graph_lib.Node,
    *,
    rankdir: RankDir,
    clocks: clock.ClockAnalysis,
    stream_names: Mapping[int, str],
    in_cycle: bool,
    num_left_inputs: int,
) -> tuple[pydot.Node, gui.NodeShapeSpec, str]:
  """Creates the pydot.Node, SVG shape spec, and tooltip for a Psyre Node."""
  dot_name = f'n_{node.id}'
  dom_id = f'psyre_n_{node.id}'
  label = _node_display_name(node)
  fill, border, text_color = _node_palette(node, in_cycle)
  tooltip = _node_tooltip(node, clocks, stream_names, in_cycle)
  use_table_ports = (
      rankdir == 'LR'
      and num_left_inputs >= 2
      and not node.is_primitive('ifthenelse')
  )
  min_left_h = (
      (0.22 * (num_left_inputs - 1) + 0.26)
      if (rankdir == 'LR' and num_left_inputs >= 2)
      else 0.0
  )
  n_val = node.kwargs.get('n', 1)
  n_sub = (
      str(n_val)
      if isinstance(n_val, int) and not isinstance(n_val, bool) and n_val > 1
      else ''
  )

  attrs: dict[str, Any] = {
      'id': _quote_dot(dom_id),
      'shape': 'box',
      'fontname': _FONT,
      'fontsize': '11',
      'fontcolor': _quote_dot(text_color),
      'style': _quote_dot('filled'),
      'fillcolor': _quote_dot(fill),
      'color': _quote_dot(border),
      'penwidth': '1.6',
  }

  kind: gui.NodeKind = 'other'
  subscript = ''
  display_label = label
  min_w_in = 0.0
  min_h_in = min_left_h
  pad_pt = 6
  if node.is_primitive(*_FBY_LIKE_OPS, 'post'):
    kind = 'post' if node.op == 'post' else 'fby'
    subscript = n_sub
    side_in = max(
        0.60,
        min_left_h,
        min(1.45, 0.095 * (len(label) + len(subscript)) + 0.34),
    )
    attrs['fixedsize'] = 'true'
    attrs['width'] = attrs['height'] = f'{side_in:.2f}'
    min_w_in = min_h_in = side_in
    pad_pt = 4
  elif node.is_primitive(*_SELECTOR_OPS, *_MUX_OPS):
    if node.op in _MUX_OPS:
      kind = 'merge'
      tb_shape = 'invtrapezium'
      base_h = max(0.64, 0.26 * max(2, len(node.inputs) - 1))
    else:
      selector = node.kwargs.get('selector')
      is_neg_bool = (
          isinstance(selector, common.WhenSelector)
          and selector.is_boolean
          and selector.negated
      )
      kind = 'whennot' if (node.op == 'whennot' or is_neg_bool) else 'when'
      tb_shape = 'trapezium'
      base_h = 0.62
    if rankdir == 'TB':
      attrs['shape'] = tb_shape
      attrs['margin'] = _quote_dot('0.14,0.08')
    else:
      min_w_in = 0.68
      min_h_in = max(base_h, min_left_h)
      attrs['height'] = f'{min_h_in:.2f}'
      attrs['margin'] = _quote_dot('0.14,0.10')
      pad_pt = 8
  elif node.is_primitive('ifthenelse'):
    kind = 'ifthenelse'
    attrs['shape'] = 'diamond'
    attrs['margin'] = _quote_dot('0.09,0.05')
    if min_left_h > 0.0:
      attrs['height'] = f'{max(0.56, min_left_h):.2f}'
  elif node.is_primitive('prefix_sum'):
    kind = 'sigma'
    display_label = 'Σ'
    attrs['fontsize'] = '18'
    attrs['margin'] = _quote_dot('0.12,0.06')
    min_w_in = 0.54
    min_h_in = max(0.42, min_left_h)
  elif node.is_primitive('buffer'):
    kind = 'fifo'
    display_label = ''
    attrs['fixedsize'] = 'true'
    if rankdir == 'TB':
      attrs['width'] = '0.32'
      attrs['height'] = '0.68'
    else:
      min_w_in = 0.68
      min_h_in = max(0.32, min_left_h)
      attrs['width'] = f'{min_w_in:.2f}'
      attrs['height'] = f'{min_h_in:.2f}'
  elif node.is_primitive('counter'):
    kind = 'counter'
    display_label = '042'
    attrs['fixedsize'] = 'true'
    min_w_in = 0.72
    min_h_in = max(0.38, min_left_h)
    attrs['width'] = f'{min_w_in:.2f}'
    attrs['height'] = f'{min_h_in:.2f}'
  elif node.is_stateful:
    subscript = n_sub
    attrs['peripheries'] = '2'
    attrs['margin'] = _quote_dot('0.12,0.08')
    min_w_in = 0.62
    min_h_in = max(0.42, min_left_h)
  elif label in _DUNDER_GLYPH_SET:
    attrs['style'] = _quote_dot('rounded,filled')
    attrs['fontsize'] = '13'
    min_w_in = 0.42
    min_h_in = max(0.36, min_left_h)
    attrs['width'] = f'{min_w_in:.2f}'
    attrs['height'] = f'{min_h_in:.2f}'
    attrs['margin'] = _quote_dot('0.08,0.05')
    pad_pt = 5
  else:
    attrs['style'] = _quote_dot('rounded,filled')
    attrs['margin'] = _quote_dot('0.12,0.08')
    min_w_in = 0.56
    min_h_in = max(0.38, min_left_h)

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
      kind=kind,
      border_color=border,
      subscript=subscript,
  )
  return pydot.Node(dot_name, **attrs), spec, tooltip


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
      fontname=_FONT,
      fontsize='9.5' if compact else '10.5',
      penwidth='1.2',
      height='0.28' if compact else '0.36',
      margin=_quote_dot('0.07,0.04' if compact else '0.10,0.06'),
  )


def _activity_key(
    node: graph_lib.Node, clocks: clock.ClockAnalysis
) -> tuple[str, ...]:
  """Returns the tuple of enclosing ClockStep strings for scope clustering."""
  return tuple(str(step) for step in clocks.node_clock(node).steps)


def _valid_expansion_states(
    fsm_prefixes: Sequence[str],
) -> list[frozenset[str]]:
  """Returns valid hierarchical sets of expanded automaton/node prefixes."""
  # Take the Cartesian product of sibling subtree states while its size stays
  # within 8; beyond that threshold, intentionally fall back to the linear union
  # of individual subtree states plus expand-all/collapse-all (not a hard total
  # cap) so every automaton/PE box stays clickable.
  if not fsm_prefixes:
    return [frozenset()]
  children: dict[str, list[str]] = {'': []}
  for pfx in fsm_prefixes:
    children.setdefault(pfx, [])
    children.setdefault(_parent_prefix(pfx), []).append(pfx)

  def subtree_states(root: str) -> list[frozenset[str]]:
    direct_children = children.get(root, ())
    leaf_children = [c for c in direct_children if not children.get(c)]
    branch_children = [c for c in direct_children if children.get(c)]
    if len(leaf_children) > 4:
      child_lists = [subtree_states(c) for c in branch_children]
      child_lists.append([frozenset(), frozenset(leaf_children)])
    else:
      child_lists = [subtree_states(c) for c in direct_children]
    if math.prod(max(1, len(opts)) for opts in child_lists) <= 8:
      combos = [
          frozenset(itertools.chain.from_iterable(parts))
          for parts in itertools.product(*child_lists)
      ]
    else:
      combos = list(
          dict.fromkeys(
              itertools.chain(
                  (frozenset(),),
                  itertools.chain.from_iterable(child_lists),
                  (frozenset(direct_children),),
              )
          )
      )
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


def _net_ref_label(
    stream: stream_lib.Stream,
    prod: graph_lib.Node | None,
    stream_names: Mapping[int, str],
    collapsed_pfx: str | None = None,
) -> str:
  """Formats a compact schematic net-reference badge label for `stream`."""
  explicit = stream_names.get(stream.id, '')
  if prod is not None:
    base = explicit or _node_display_name(prod)
    if collapsed_pfx:
      return f'↪ {collapsed_pfx}.{base.removeprefix(f"{collapsed_pfx}/")}'
    return f'↪ {base}#{prod.id}'
  return f'↪ {explicit or f"S_{stream.id}"}'


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
  stream_names: dict[int, str]
  cycle_ids: set[int]
  cycle_edges: set[tuple[int, int]]
  stream_fanout: dict[int, int]
  fsm_prefixes: list[str]
  fsm_prefix_index: dict[str, int]
  descendant_nodes: dict[str, list[graph_lib.Node]]
  base_act: dict[str, tuple[str, ...]]


def _analyze_graph(target_graph: graph_lib.Graph) -> _GraphVizContext:
  """Computes view-independent clock, cycle, fanout, and hierarchy metadata."""
  target_graph.require_finalized()
  clocks = clock.check_clocks(target_graph)
  stream_names = _explicit_stream_names(target_graph)
  cycle = analysis.find_instantaneous_cycle(target_graph)
  cycle_ids = {n.id for n in cycle} if cycle is not None else set()
  cycle_edges: set[tuple[int, int]] = set()
  if cycle is not None:
    for idx, src_node in enumerate(cycle):
      cycle_edges.add((src_node.id, cycle[(idx + 1) % len(cycle)].id))

  stream_fanout: dict[int, int] = {}
  descendant_nodes: dict[str, list[graph_lib.Node]] = {}
  for node in target_graph.nodes:
    for s in node.inputs:
      stream_fanout[s.id] = stream_fanout.get(s.id, 0) + 1
    for anc in _ancestor_prefixes(_node_automaton_prefix(node)):
      descendant_nodes.setdefault(anc, []).append(node)
  for out_s in graph_lib.iter_output_leaves(target_graph.output):
    stream_fanout[out_s.id] = stream_fanout.get(out_s.id, 0) + 1

  fsm_prefixes = list(descendant_nodes)
  base_act: dict[str, tuple[str, ...]] = {}
  for pfx, members in descendant_nodes.items():
    clocked_members = [n for n in members if n.inputs or n.resets] or members
    common_key = _activity_key(clocked_members[0], clocks)
    for member in clocked_members[1:]:
      member_key = _activity_key(member, clocks)
      limit = min(len(common_key), len(member_key))
      lcp = next(
          (i for i in range(limit) if common_key[i] != member_key[i]), limit
      )
      common_key = common_key[:lcp]
    base_act[pfx] = common_key

  return _GraphVizContext(
      clocks=clocks,
      stream_names=stream_names,
      cycle_ids=cycle_ids,
      cycle_edges=cycle_edges,
      stream_fanout=stream_fanout,
      fsm_prefixes=fsm_prefixes,
      fsm_prefix_index={pfx: i for i, pfx in enumerate(fsm_prefixes)},
      descendant_nodes=descendant_nodes,
      base_act=base_act,
  )


def _validate_expanded_automata(
    expanded: Iterable[str], available: Iterable[str]
) -> set[str]:
  """Validates `expanded_automata` prefixes and returns them as a set."""
  req = set(expanded)
  avail_set = set(available)
  unknown = sorted(req - avail_set)
  if unknown:
    raise ValueError(
        f'Unknown expanded_automata prefix(es): {unknown!r}. '
        f'Available: {sorted(avail_set)!r}.'
    )
  return req


def _prod_key(prod: graph_lib.Node, box_dot: str | None) -> str:
  """Returns the DOT node identifier representing `prod` (or its box)."""
  return box_dot if box_dot is not None else f'n_{prod.id}'


class _DotBuilder:
  """Constructs a `pydot.Dot` schematic and DOM metadata for one view state."""

  def __init__(
      self,
      target_graph: graph_lib.Graph,
      ctx: _GraphVizContext,
      opts: _LayoutOptions,
  ) -> None:
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
      self.expanded_set = _validate_expanded_automata(
          opts.expanded_automata, ctx.fsm_prefix_index
      )
    elif opts.collapse_automata:
      self.expanded_set = set()
    else:
      self.expanded_set = set(ctx.fsm_prefixes)

    self.collapsed_owner: dict[int, str | None] = {
        node.id: next(
            (
                anc
                for anc in _ancestor_prefixes(_node_automaton_prefix(node))
                if anc not in self.expanded_set
            ),
            None,
        )
        for node in target_graph.nodes
    }
    owned_prefixes = set(self.collapsed_owner.values())
    self.active_collapsed = [
        pfx for pfx in ctx.fsm_prefixes if pfx in owned_prefixes
    ]
    self.visible_nodes = [
        n for n in target_graph.nodes if self.collapsed_owner[n.id] is None
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
        fontname=_FONT,
    )
    self.shape_specs: dict[str, gui.NodeShapeSpec] = {}
    self.info_by_dom_id: dict[str, str] = {}
    self.net_by_dom_id: dict[str, str] = {}
    self.fsm_box_by_dom_id: dict[str, str] = {}
    self.fsm_cluster_by_dom_id: dict[str, str] = {}
    self.static_edge_to_node: dict[str, str] = {}
    self.reversed_edge_ids: set[str] = set()
    self.containers: dict[tuple[str, tuple[str, ...]], pydot.Cluster] = {}
    self._next_act_idx = 0
    self.box_dot_by_prefix: dict[str, str] = {}
    self.box_container_by_prefix: dict[str, pydot.Dot | pydot.Cluster] = {}
    self.box_ports_by_prefix: dict[
        str, list[tuple[bool, stream_lib.Stream, graph_lib.Node]]
    ] = {}
    self.node_containers: dict[int, pydot.Dot | pydot.Cluster] = {}
    self.left_inputs_by_id: dict[int, int] = {}
    self.same_name_groups: dict[
        tuple[int, str], list[tuple[graph_lib.Node, pydot.Node]]
    ] = {}
    self.fby_primary_consumer: dict[int, int] = {}
    self.ext_sources: set[int] = set()
    self.direct_out_producers: set[str] = set()
    self.net_ref_by_source: dict[str, dict[int, int]] = {}
    self.constrained_adj: dict[str, set[str]] = {}
    self._edge_infos: dict[tuple[int, int], _InputEdgeInfo] = {}

  def build(self) -> pydot.Dot:
    """Populates and returns the `pydot.Dot` schematic graph."""
    self._init_clusters()
    self._add_collapsed_boxes()
    self._add_visible_nodes()
    self._add_same_name_chains()
    self._compute_fby_primary_consumers()
    self._add_collapsed_box_edges()
    self._prepopulate_constrained_adj()
    for node in self.visible_nodes:
      self._add_edges_for_node(node)
    self._add_output_terminals()
    self._add_net_out_badges()
    return self.dot

  def to_schematic_view(self, raw_svg: str, dot_src: str) -> gui.SchematicView:
    """Packages rendered SVG and collected DOM metadata into `SchematicView`."""
    return gui.SchematicView(
        raw_svg=raw_svg,
        dot_src=dot_src,
        rankdir=self.opts.rankdir,
        shape_specs=self.shape_specs,
        info_by_dom_id=self.info_by_dom_id,
        net_by_dom_id=self.net_by_dom_id,
        fsm_box_by_dom_id=self.fsm_box_by_dom_id,
        fsm_cluster_by_dom_id=self.fsm_cluster_by_dom_id,
        static_edge_to_node=self.static_edge_to_node,
        reversed_edge_ids=frozenset(self.reversed_edge_ids),
    )

  def _get_container(
      self,
      fsm_pfx: str,
      act_key: tuple[str, ...],
  ) -> pydot.Dot | pydot.Cluster:
    """Returns or creates the nested DOT cluster for `(fsm_pfx, act_key)`."""
    # Strict tree nesting invariant: each automaton cluster (`fsm_pfx`) lives
    # inside the longest-common-prefix subclock `base_act[fsm_pfx]` shared by
    # its members; deeper `act_key` subclock clusters nest inside `fsm_pfx`.
    ctx = self.ctx
    group_act = self.opts.group_by_activity
    b_act = ctx.base_act.get(fsm_pfx, ()) if (fsm_pfx and group_act) else ()
    eff_act = act_key if group_act else ()
    if eff_act[: len(b_act)] != b_act:
      eff_act = b_act
    if not fsm_pfx and not eff_act:
      return self.dot
    key = (fsm_pfx, eff_act)
    existing = self.containers.get(key)
    if existing is not None:
      return existing
    if len(eff_act) > len(b_act):
      parent = self._get_container(fsm_pfx, eff_act[:-1])
      scope_clk = _format_act_clock(eff_act)
      fsm_slug = _slug(fsm_pfx) if fsm_pfx else 'root'
      act_slug = _slug('_'.join(eff_act))
      act_idx = self._next_act_idx
      self._next_act_idx += 1
      act_dom = f'psyre_act_cluster_{act_idx}'
      cluster = pydot.Cluster(
          f'act_{act_idx}_{fsm_slug}_{act_slug}',
          id=_quote_dot(act_dom),
          label=_quote_dot(f'clock: {scope_clk}'),
          style=_quote_dot('rounded,dashed,filled'),
          color=_quote_dot('#CBD5E1'),
          fillcolor=_quote_dot('#F8FAFC'),
          fontname=_FONT,
          fontsize='9.5',
          fontcolor=_quote_dot('#475569'),
          margin='10',
      )
      if fsm_pfx:
        self.fsm_cluster_by_dom_id[act_dom] = fsm_pfx
        self.info_by_dom_id[act_dom] = (
            f'automaton={fsm_pfx} | clock={scope_clk} | Click to collapse'
        )
      parent.add_subgraph(cluster)
      self.containers[key] = cluster
      return cluster

    parent = self._get_container(_parent_prefix(fsm_pfx), b_act)
    fsm_idx = ctx.fsm_prefix_index[fsm_pfx]
    c_dom = f'psyre_fsm_cluster_{fsm_idx}'
    members = ctx.descendant_nodes[fsm_pfx]
    cluster = pydot.Cluster(
        f'fsm_{fsm_idx}_{_slug(fsm_pfx)}',
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
    self.fsm_cluster_by_dom_id[c_dom] = fsm_pfx
    self.info_by_dom_id[c_dom] = (
        f'automaton={fsm_pfx} | nodes={len(members)} | '
        f'clock={_format_act_clock(b_act)} | Click header or box to collapse'
    )
    parent.add_subgraph(cluster)
    self.containers[key] = cluster
    return cluster

  def _iter_visible_inputs(
      self, node: graph_lib.Node
  ) -> Iterator[tuple[int, stream_lib.Stream]]:
    """Yields visible `(port_idx, stream)` inputs of `node`."""
    stream_names = self.ctx.stream_names
    show_consts = self.opts.show_constants
    for port_idx, stream in enumerate(node.inputs):
      if not show_consts and _is_unnamed_scalar_constant(
          stream, self.graph, stream_names
      ):
        continue
      yield port_idx, stream

  def _visible_resets(
      self, node: graph_lib.Node
  ) -> list[tuple[int, stream_lib.Stream]]:
    """Returns visible `(rst_idx, stream)` reset inputs for `node`."""
    if not (self.opts.show_reset_edges and node.is_stateful):
      return []
    show_consts = self.opts.show_constants
    stream_names = self.ctx.stream_names
    return [
        (idx, r)
        for idx, r in enumerate(node.resets)
        if show_consts
        or not _is_unnamed_scalar_constant(r, self.graph, stream_names)
    ]

  def _count_left_inputs(self, node: graph_lib.Node) -> int:
    """Counts visible left-connected inputs (including resets)."""
    n_in = sum(
        1
        for port_idx, _ in self._iter_visible_inputs(node)
        if not _is_selector_input_port(node, port_idx)
    )
    return n_in + len(self._visible_resets(node))

  def _collect_collapsed_box_ports(
      self, box_pfx: str, members: Sequence[graph_lib.Node]
  ) -> list[tuple[bool, stream_lib.Stream, graph_lib.Node]]:
    """Collects deduplicated external `(is_reset, stream, member)` ports."""
    stream_names = self.ctx.stream_names
    seen_inputs: set[int] = set()
    seen_resets: set[int] = set()
    ports: list[tuple[bool, stream_lib.Stream, graph_lib.Node]] = []
    for member in members:
      groups = [(False, member.inputs, seen_inputs)]
      if self.opts.show_reset_edges and member.is_stateful:
        groups.append((True, member.resets, seen_resets))
      for is_reset, streams, seen in groups:
        for stream in streams:
          if stream.id in seen or _is_unnamed_scalar_constant(
              stream, self.graph, stream_names
          ):
            continue
          prod = self.graph.producer(stream)
          if prod is not None and self.collapsed_owner.get(prod.id) == box_pfx:
            continue
          seen.add(stream.id)
          ports.append((is_reset, stream, member))
    return ports

  def _collapsed_box_for_prod(
      self, prod: graph_lib.Node | None
  ) -> tuple[str | None, str | None]:
    """Returns `(collapsed_pfx, box_dot)` when `prod` lives inside a box."""
    if prod is None:
      return None, None
    pfx = self.collapsed_owner.get(prod.id)
    if pfx is None:
      return None, None
    return pfx, self.box_dot_by_prefix.get(pfx)

  def _record_direct_out(self, src_ep: str, dst_key: str) -> None:
    src_key = src_ep.split(':', 1)[0]
    if src_key != dst_key:
      self.direct_out_producers.add(src_key)

  def _emit_net_ref_endpoint(
      self,
      container: pydot.Dot | pydot.Cluster,
      *,
      ref_dot: str,
      stream: stream_lib.Stream,
      prod: graph_lib.Node | None,
      palette: tuple[str, str, str],
      collapsed_pfx: str | None = None,
      collapsed_box_dot: str | None = None,
      register_fsm_box: bool = True,
  ) -> str:
    """Adds a local schematic net-reference badge and returns its endpoint."""
    ref_dom = f'psyre_{ref_dot}'
    stream_names = self.ctx.stream_names
    ref_label = _net_ref_label(stream, prod, stream_names, collapsed_pfx)
    s_clk = str(self.ctx.clocks.stream_clock(stream))
    if prod is not None:
      src_key = _prod_key(prod, collapsed_box_dot)
      src_counts = self.net_ref_by_source.setdefault(src_key, {})
      src_counts[stream.id] = src_counts.get(stream.id, 0) + 1
    if collapsed_pfx and collapsed_box_dot:
      net_key = collapsed_box_dot
      src_desc = collapsed_pfx
      if register_fsm_box:
        self.fsm_box_by_dom_id[ref_dom] = collapsed_pfx
    elif prod is not None:
      net_key = f'n_{prod.id}'
      src_desc = f'{_node_full_display_name(prod)}#{prod.id}'
    else:
      net_key = f'in_{stream.id}'
      src_desc = stream_names.get(stream.id) or f'S_{stream.id}'
    self.info_by_dom_id[ref_dom] = (
        f'net_ref={src_desc} | stream=S_{stream.id} | clock={s_clk}'
    )
    self.net_by_dom_id[ref_dom] = net_key
    container.add_node(
        _make_terminal_dot_node(
            ref_dot, ref_dom, ref_label, palette, compact=True
        )
    )
    return f'{ref_dot}:{self.out_port}'

  def _resolve_source(
      self,
      local_container: pydot.Dot | pydot.Cluster,
      stream: stream_lib.Stream,
      tag_suffix: str,
  ) -> str:
    """Returns the direct DOT source endpoint (`node:port`) for `stream`."""
    stream_names = self.ctx.stream_names
    clocks = self.ctx.clocks
    prod = self.graph.producer(stream)
    if prod is not None:
      _, prod_box_dot = self._collapsed_box_for_prod(prod)
      return f'{_prod_key(prod, prod_box_dot)}:{self.out_port}'
    if stream.static:
      c_dot = f'c_{tag_suffix}'
      c_dom = f'psyre_{c_dot}'
      s_name = stream_names.get(stream.id, '')
      seq_str = _format_static_stream(stream)
      c_label = s_name or seq_str
      s_clk = str(clocks.stream_clock(stream))
      if _is_unnamed_scalar_constant(stream, self.graph, stream_names):
        c_tip = f'constant={c_label} | stream=S_{stream.id} | clock={s_clk}'
        text_color = gui.STYLE_CONST[2]
      else:
        c_tip = (
            f'static_stream={c_label} | stream=S_{stream.id} | clock={s_clk}'
        )
        if s_name:
          c_tip = f'{c_tip} | values={seq_str}'
        text_color = gui.STYLE_INPUT[2]
      local_container.add_node(
          pydot.Node(
              c_dot,
              id=_quote_dot(c_dom),
              label=_quote_dot(c_label),
              shape='plaintext',
              fontcolor=_quote_dot(text_color),
              fontname=_FONT,
              fontsize='9.5',
              width='0.36',
              height='0.26',
              margin=_quote_dot('0.08,0.04'),
          )
      )
      self.shape_specs[c_dom] = gui.NodeShapeSpec(
          kind='static_stream', border_color=text_color
      )
      self.info_by_dom_id[c_dom] = c_tip
      self.net_by_dom_id[c_dom] = f'in_{stream.id}'
      return c_dot
    src_dot = f'in_{stream.id}'
    if stream.id not in self.ext_sources:
      src_dom = f'psyre_{src_dot}'
      s_label = stream_names.get(stream.id, '') or f'S_{stream.id}'
      s_clk = str(clocks.stream_clock(stream))
      self.dot.add_node(
          _make_terminal_dot_node(src_dot, src_dom, s_label, gui.STYLE_INPUT)
      )
      self.ext_sources.add(stream.id)
      self.info_by_dom_id[src_dom] = (
          f'input={s_label} | stream=S_{stream.id} | clock={s_clk}'
      )
      self.net_by_dom_id[src_dom] = src_dot
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
      self.static_edge_to_node[e_dom] = f"psyre_{src_ep.split(':', 1)[0]}"
    self.info_by_dom_id[e_dom] = info_text
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
            fontname=_FONT,
            fontsize='8.5',
            fontcolor=_quote_dot('#BE123C'),
        )
    )

  def _classify_input_edge(
      self,
      node: graph_lib.Node,
      port_idx: int,
      stream: stream_lib.Stream,
  ) -> _InputEdgeInfo:
    """Classifies routing, cycle, cluster, and net-ref flags for an input."""
    is_delayed = (
        node.is_primitive('fby') and port_idx >= 1
    ) or node.is_primitive('pre')
    prod = self.graph.producer(stream)
    is_inline_src = _is_inline_source(stream, prod)
    prod_collapsed_pfx, prod_box_dot = self._collapsed_box_for_prod(prod)
    on_cycle = (
        not is_delayed
        and prod is not None
        and prod_collapsed_pfx is None
        and (prod.id, node.id) in self.ctx.cycle_edges
    )
    node_cont = self.node_containers[node.id]
    same_cluster = False
    if prod is not None:
      src_cont = (
          self.box_container_by_prefix[prod_collapsed_pfx]
          if prod_collapsed_pfx is not None
          else self.node_containers.get(prod.id, self.dot)
      )
      same_cluster = src_cont is node_cont
    is_fby_prod = (
        prod is not None
        and prod.is_primitive(*_FBY_LIKE_OPS)
        and prod_collapsed_pfx is None
    )
    is_fby_immediate = (
        is_fby_prod
        and same_cluster
        and prod is not None
        and self.fby_primary_consumer.get(prod.id) == node.id
    )
    # Break long cross-cluster, high-fanout (`>6` inputs / `>8` nodes), or
    # long-span (`node.id - prod.id > 16`) wires into local `↪` endpoints on
    # large graphs; keep causality cycles, inline arrows, and `fby`'s primary
    # edge direct.
    if (
        is_fby_immediate
        or not self.large_graph
        or on_cycle
        or is_inline_src
    ):
      use_net_ref = False
    elif is_fby_prod or (node.is_primitive('fby') and port_idx == 0):
      use_net_ref = True
    elif prod is None:
      fanout = self.ctx.stream_fanout.get(stream.id, 0)
      use_net_ref = is_delayed or (node_cont is not self.dot) or fanout > 6
    else:
      use_net_ref = (
          (not same_cluster)
          or prod.id >= node.id
          or (node.id - prod.id) > 16
          or self.ctx.stream_fanout.get(stream.id, 0) > 8
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
      if all(anc in self.expanded_set for anc in _ancestor_prefixes(pfx)):
        self._get_container(pfx, ctx.base_act[pfx])
    for node in sorted(
        self.visible_nodes,
        key=lambda n: (n.name.count('/'), len(ctx.clocks.node_clock(n))),
    ):
      self._get_container(
          _node_automaton_prefix(node), _activity_key(node, ctx.clocks)
      )

  def _add_collapsed_boxes(self) -> None:
    """Adds summary box nodes for collapsed automata."""
    ctx = self.ctx
    rankdir = self.opts.rankdir
    for pfx in self.active_collapsed:
      fsm_idx = ctx.fsm_prefix_index[pfx]
      box_dot = f'fsm_box_{fsm_idx}'
      box_dom = f'psyre_{box_dot}'
      members = ctx.descendant_nodes[pfx]
      b_act = ctx.base_act[pfx]
      box_container = self._get_container(_parent_prefix(pfx), b_act)
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
      self.fsm_box_by_dom_id[box_dom] = pfx
      self.net_by_dom_id[box_dom] = box_dot
      self.info_by_dom_id[box_dom] = (
          f'automaton={pfx} | nodes={len(members)} | '
          f'clock={_format_act_clock(b_act)} | Click to expand'
      )

  def _add_visible_nodes(self) -> None:
    """Adds primitive pydot.Nodes to their enclosing clusters."""
    ctx = self.ctx
    for node in self.visible_nodes:
      num_left_in = self._count_left_inputs(node)
      self.left_inputs_by_id[node.id] = num_left_in
      p_node, spec, tooltip = _make_pydot_op_node(
          node,
          rankdir=self.opts.rankdir,
          clocks=ctx.clocks,
          stream_names=ctx.stream_names,
          in_cycle=node.id in ctx.cycle_ids,
          num_left_inputs=num_left_in,
      )
      dom_id = f'psyre_n_{node.id}'
      self.shape_specs[dom_id] = spec
      self.info_by_dom_id[dom_id] = tooltip
      self.net_by_dom_id[dom_id] = f'n_{node.id}'
      container = self._get_container(
          _node_automaton_prefix(node), _activity_key(node, ctx.clocks)
      )
      self.node_containers[node.id] = container
      if _has_custom_node_name(node):
        self.same_name_groups.setdefault((id(container), node.name), []).append(
            (node, p_node)
        )
      container.add_node(p_node)

  def _add_same_name_chains(self) -> None:
    """Aligns same-named stateful/operator nodes with invisible rank chains.

    Sharing a DOT `group` attribute and high-weight invisible edges keeps
    unrolled/iterated stages with the same custom name collinear along rank.
    """
    for (_, same_name), group in self.same_name_groups.items():
      if len(group) < 2:
        continue
      group_attr = _quote_dot(f'same_name_{same_name}')
      for _, p_node in group:
        p_node.set('group', group_attr)
      container = self.node_containers[group[0][0].id]
      for (u_prev, _), (u_curr, _) in zip(group, group[1:]):
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
      if not u_node.is_primitive(*_FBY_LIKE_OPS):
        continue
      u_cont = self.node_containers[u_node.id]
      fwd = [
          v.id
          for v in self.graph.consumers(u_node.output)
          if v.id > u_node.id
          and self.node_containers.get(v.id) is u_cont
          and (u_node.id, v.id) not in self.ctx.cycle_edges
          and any(s is u_node.output for s in v.inputs)
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
      for is_reset, stream, member in ports:
        prod = self.graph.producer(stream)
        is_inline = _is_inline_source(stream, prod)
        if is_reset:
          rst_idx += 1
          suffix = f'{box_dot}_rst{rst_idx}'
        else:
          suffix = f'{box_dot}_in{in_idx}'
        src_ep = self._resolve_source(box_container, stream, suffix)
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
              f'reset -> {pfx} | stream=S_{stream.id}',
              is_inline=is_inline,
              constraint='true' if is_inline else 'false',
          )
        else:
          e_dom = f'psyre_e_{box_dot}_in{in_idx}'
          in_idx += 1
          s_clk = str(self.ctx.clocks.stream_clock(stream))
          self.info_by_dom_id[e_dom] = (
              f'edge -> {pfx} | stream=S_{stream.id} | clock={s_clk}'
          )
          if is_inline:
            self.static_edge_to_node[e_dom] = f'psyre_{src_key}'
          is_back = prod is not None and prod.id > member.id
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
    """Seeds constrained forward edges and caches input edge metadata."""
    for node in self.visible_nodes:
      dst_key = f'n_{node.id}'
      for port_idx, stream in self._iter_visible_inputs(node):
        info = self._classify_input_edge(node, port_idx, stream)
        self._edge_infos[(node.id, port_idx)] = info
        if (
            info.prod is None
            or info.on_cycle
            or info.use_net_ref
            or (info.is_delayed and not info.is_fby_immediate)
        ):
          continue
        src_key = _prod_key(info.prod, info.prod_box_dot)
        if src_key != dst_key:
          self.constrained_adj.setdefault(src_key, set()).add(dst_key)

  def _add_edges_for_node(self, node: graph_lib.Node) -> None:
    """Adds all visible incoming edges for a single `node`."""
    node_container = self.node_containers[node.id]
    num_left_inputs = self.left_inputs_by_id[node.id]
    dst_name = f'n_{node.id}'
    if node.is_primitive(*_MUX_OPS):
      arm_labels = _merge_arm_labels(node)
    elif node.is_primitive('ifthenelse'):
      arm_labels = {1: 'T', 2: 'F'}
    elif node.is_primitive('when'):
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
      if node.is_primitive('ifthenelse') and num_left_inputs == 2:
        compass = 'nw' if left_idx == 0 else 'sw'
        left_idx += 1
        return f'{dst_name}:{compass}'
      ep = f'{dst_name}:in{left_idx}:w'
      left_idx += 1
      return ep

    for port_idx, stream in self._iter_visible_inputs(node):
      is_selector_port = _is_selector_input_port(node, port_idx)
      info = self._edge_infos[(node.id, port_idx)]
      prod_key = (
          _prod_key(info.prod, info.prod_box_dot)
          if info.prod is not None
          else None
      )
      is_delayed_self_loop = info.is_delayed and prod_key == dst_name
      # A delayed `fby`/`pre` input is a feedback edge if `prod` is already
      # reachable forward from `node`, or (for mutual-feedback cycles where
      # neither direction is constrained yet) if `prod` was created later.
      is_delayed_feedback = (
          info.is_delayed
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
        if info.prod is None:
          ref_palette = gui.STYLE_INPUT
        elif is_back_edge:
          ref_palette = gui.STYLE_FBY
        else:
          ref_palette = gui.STYLE_NET_REF
        src_ep = self._emit_net_ref_endpoint(
            node_container,
            ref_dot=f'ref_{node.id}_p{port_idx}',
            stream=stream,
            prod=info.prod,
            palette=ref_palette,
            collapsed_pfx=info.prod_collapsed_pfx,
            collapsed_box_dot=info.prod_box_dot,
        )
      else:
        src_ep = self._resolve_source(
            node_container, stream, f'{node.id}_p{port_idx}'
        )
        self._record_direct_out(src_ep, dst_name)

      dst_ep = (
          f'{dst_name}:{self.sel_port}'
          if is_selector_port
          else next_left_dst_ep()
      )
      edge_label = arm_labels.get(port_idx, '')
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
        edge_color = (
            '#B45309' if node.is_primitive(*_SELECTOR_OPS) else '#0F766E'
        )

      e_dom = f'psyre_e_{node.id}_p{port_idx}'
      s_clk = str(self.ctx.clocks.stream_clock(stream))
      if info.is_delayed:
        role = 'delayed_next'
      elif is_selector_port:
        role = 'selector'
      else:
        role = f'in[{port_idx}]'
      self.info_by_dom_id[e_dom] = (
          f'edge -> {_node_full_display_name(node)}.{role} | '
          f'stream=S_{stream.id} | clock={s_clk}'
      )
      edge_attrs: dict[str, Any] = {
          'id': _quote_dot(e_dom),
          'color': _quote_dot(edge_color),
          'style': _quote_dot(edge_style),
          'penwidth': penwidth,
          'arrowsize': '0.65' if info.is_inline_src else '0.7',
          'constraint': constraint,
          'fontname': _FONT,
          'fontsize': '9',
          'fontcolor': _quote_dot(edge_color),
      }
      if info.is_inline_src:
        self.static_edge_to_node[e_dom] = f'psyre_{src_key}'
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
        self.reversed_edge_ids.add(e_dom)
        edge_parent.add_edge(pydot.Edge(dst_ep, src_ep, **edge_attrs))
      else:
        edge_parent.add_edge(pydot.Edge(src_ep, dst_ep, **edge_attrs))

    for rst_idx, rst_s in self._visible_resets(node):
      rst_prod = self.graph.producer(rst_s)
      rst_is_inline = _is_inline_source(rst_s, rst_prod)
      rst_collapsed_pfx, rst_box_dot = self._collapsed_box_for_prod(rst_prod)
      if self.large_graph and not rst_is_inline:
        src_ep = self._emit_net_ref_endpoint(
            node_container,
            ref_dot=f'ref_{node.id}_rst{rst_idx}',
            stream=rst_s,
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
      self.info_by_dom_id[out_dom] = out_tip
      prod = self.graph.producer(out_s)
      is_inline_out = _is_inline_source(out_s, prod)
      if prod is not None:
        prod_collapsed_pfx, prod_box_dot = self._collapsed_box_for_prod(prod)
        if self.large_graph and (
            (self.node_containers.get(prod.id, self.dot) is not self.dot)
            or prod_collapsed_pfx is not None
            or ctx.stream_fanout.get(out_s.id, 0) > 8
        ):
          src_ep = self._emit_net_ref_endpoint(
              self.dot,
              ref_dot=f'ref_out_{out_idx}',
              stream=out_s,
              prod=prod,
              palette=gui.STYLE_NET_REF,
              collapsed_pfx=prod_collapsed_pfx,
              collapsed_box_dot=prod_box_dot,
          )
        else:
          src_key = _prod_key(prod, prod_box_dot)
          self.direct_out_producers.add(src_key)
          src_ep = f'{src_key}:{self.out_port}'
      else:
        src_ep = self._resolve_source(self.dot, out_s, f'out_{out_idx}')

      e_dom = f'psyre_e_out_{out_idx}'
      if is_inline_out:
        self.static_edge_to_node[e_dom] = (
            f"psyre_{src_ep.split(':', 1)[0]}"
        )
      self.info_by_dom_id[e_dom] = out_tip
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
    self.info_by_dom_id[out_ref_dom] = out_tip
    self.net_by_dom_id[out_ref_dom] = src_key
    if fsm_pfx is not None:
      self.fsm_box_by_dom_id[out_ref_dom] = fsm_pfx
    e_dom = f'psyre_e_{out_ref_dot}'
    self.info_by_dom_id[e_dom] = out_tip
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
      out_s = node.output
      fanout_cnt = self.net_ref_by_source[src_key].get(out_s.id, 1)
      base_label = _net_ref_label(out_s, node, ctx.stream_names)
      ref_label = (
          base_label if fanout_cnt <= 1 else f'{base_label} (×{fanout_cnt})'
      )
      is_fby = node.is_primitive(*_FBY_LIKE_OPS)
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


def build_pydot(
    target_graph: graph_lib.Graph,
    *,
    rankdir: RankDir = 'LR',
    show_constants: bool = True,
    group_by_activity: bool = True,
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
    show_reset_edges: Whether to draw dotted `rst` control wires on stateful
      nodes.
    collapse_automata: When True and `expanded_automata` is None, collapses all
      automata into summary boxes (defaults to `False` for static DOT/SVG).
    expanded_automata: Optional explicit iterable of automaton prefixes to
      render expanded.

  Returns:
    A configured `pydot.Dot` instance ready for Graphviz layout.
  """
  ctx = _analyze_graph(target_graph)
  return _DotBuilder(
      target_graph,
      ctx,
      _LayoutOptions(
          rankdir=rankdir,
          show_constants=show_constants,
          group_by_activity=group_by_activity,
          show_reset_edges=show_reset_edges,
          collapse_automata=collapse_automata,
          expanded_automata=expanded_automata,
      ),
  ).build()


def _build_schematic_view(
    target_graph: graph_lib.Graph,
    opts: _LayoutOptions,
    *,
    prog: str,
    viz_ctx: _GraphVizContext | None = None,
) -> gui.SchematicView:
  """Builds a `gui.SchematicView` by running Graphviz `dot -Tsvg`."""
  ctx = _analyze_graph(target_graph) if viz_ctx is None else viz_ctx
  builder = _DotBuilder(target_graph, ctx, opts)
  dot_src = builder.build().to_string()
  return builder.to_schematic_view(_run_dot_svg(dot_src, prog), dot_src)


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
  gui.validate_height_px(height_px)
  viz_ctx = _analyze_graph(target_graph)
  fsm_prefixes = viz_ctx.fsm_prefixes
  valid_states = _valid_expansion_states(fsm_prefixes)
  if opts.expanded_automata is not None:
    req = _validate_expanded_automata(opts.expanded_automata, fsm_prefixes)
    fsm_set = set(fsm_prefixes)
    closed_req: set[str] = set()
    for pfx in req:
      closed_req.update(
          anc for anc in _ancestor_prefixes(pfx) if anc in fsm_set
      )
    req_state = frozenset(closed_req)
    # Materialize explicitly requested valid states omitted by the linear
    # product fallback in `_valid_expansion_states`.
    if req_state not in valid_states:
      valid_states.append(req_state)
    initial_key = _canonical_state_key(req_state)
  elif opts.collapse_automata or not fsm_prefixes:
    initial_key = ''
  else:
    initial_key = _canonical_state_key(fsm_prefixes)
  prog = _get_dot_binary(dot_prog)
  sem = (
      target_graph.semantics.name
      if target_graph.semantics is not None
      else 'Psyre'
  )

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
  return gui.SchematicBundle(
      views=views,
      initial_view_key=initial_key,
      fsm_prefixes=tuple(fsm_prefixes),
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
    show_reset_edges: bool = True,
    collapse_automata: bool = False,
    expanded_automata: Iterable[str] | None = None,
    dot_prog: str | None = None,
) -> str:
  """Renders a Stage-1 Psyre `Graph` as a custom schematic SVG string."""
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
    show_reset_edges: bool = True,
    collapse_automata: bool = True,
    expanded_automata: Iterable[str] | None = None,
    dot_prog: str | None = None,
) -> str:
  """Renders a Stage-1 Psyre `Graph` as a self-contained interactive HTML viewer."""
  return visualize_graph(
      target_graph,
      title=title,
      height_px=height_px,
      rankdir=rankdir,
      show_constants=show_constants,
      group_by_activity=group_by_activity,
      show_reset_edges=show_reset_edges,
      collapse_automata=collapse_automata,
      expanded_automata=expanded_automata,
      dot_prog=dot_prog,
  ).html


def visualize_graph(
    target_graph: graph_lib.Graph,
    *,
    title: str = '',
    height_px: int = 560,
    rankdir: RankDir = 'LR',
    show_constants: bool = True,
    group_by_activity: bool = True,
    show_reset_edges: bool = True,
    collapse_automata: bool = True,
    expanded_automata: Iterable[str] | None = None,
    dot_prog: str | None = None,
) -> GraphVisualization:
  """Builds an interactive `GraphVisualization` for a Stage-1 Psyre `Graph`."""
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
