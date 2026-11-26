# `psyre.gui`

Pure-stdlib SVG schematic post-processor and self-contained interactive
HTML/CSS/JS viewer for Psyre Stage-1 graphs.

## Overview & Role

`psyre.gui` decouples browser presentation and SVG geometry rewriting from
Graphviz invocation and core Psyre IR internals. It has **zero dependencies** on
`pydot`, Graphviz, or any other Psyre module: [`psyre.visualize`](visualize.md)
hands it immutable data transfer objects (`NodeShapeSpec`, `SchematicView`,
`SchematicBundle`), and `psyre.gui` transforms raw Graphviz SVGs into
publication-grade schematics and interactive pan/zoom HTML viewers suitable for
Jupyter, Colab, and standalone browser files.

## Design & Key Invariants

### SVG Post-Processing Pipeline (`render_view_svg`)

Raw Graphviz SVG output is parsed with `xml.etree.ElementTree` (with the default
SVG namespace registered to avoid `ns0:` prefixes) and rewritten in a single
pass:

1. **Notebook `<svg:a>` unwrapping**: Strips Graphviz `<svg:a>` anchor wrappers
   in place (re-parenting their children) so strict notebook HTML sanitizers do
   not fail on nested SVG links.
2. **Semantic DOM attributes & direction restoration**: Maps each `<g>` node and
   edge to `data-psyre-key`, `data-psyre-src`, `data-psyre-dst`,
   `data-psyre-dst-port`, `data-psyre-kind`, `data-psyre-info` (also updating
   `<title>` for native tooltips), `data-psyre-net`, `data-psyre-fsm-box`, and
   `data-psyre-fsm-cluster`. For edge IDs in `view.reversed_edge_ids` (emitted
   with `dir="back"` by [`psyre.visualize`](visualize.md) to keep feedback edges
   constrained), `src` and `dst` are swapped back to true producer-to-consumer
   direction.
3. **Custom schematic glyphs**: Replaces generic Graphviz boxes with domain
   symbols keyed by `NodeShapeSpec.kind`:
   - `fby` / `post`: Centered square with a downward `∨` chevron on the top wall
     (`fby`/`pre`) or upward `∧` chevron on the bottom wall (`post`), plus
     optional subscript for `n > 1`.
   - `when` / `whennot` / `merge`: Opposing trapezoids in `LR` layout (`when`
     flares left-to-right; `merge` narrows left-to-right), with an active-low
     white bubble (`r=3.2`) on the selector wall for `whennot`.
   - `ifthenelse`: Symmetric diamond.
   - `sigma`: Highlighted summation box (`Σ` for `prefix_sum`).
   - `fifo`: Open-ended 3-sided shift-register frame with 3 internal slot
     dividers (for `buffer`).
   - `counter`: 3-digit mechanical odometer glyph (`0`, `4`, `2`).
   - `static_stream`: Strips box `<polygon>`/`<path>` strokes, leaving clean
     boxless label text.
   - `other`: Appends subscript `<tspan>` when `spec.subscript` is non-empty.
4. **Inline static arrow shortening & cluster border widening**: Rewrites inline
   static-source wires (`view.static_edge_to_node`) into short straight arrows
   anchored cleanly to destination walls (accounting for LR trapezoid slant and
   `whennot`'s active-low bubble). When a left-aligned tail label extends past
   an enclosing cluster's left boundary, the enclosing cluster's `<path>`
   (rounded rectangle) or `<polygon>` border is widened automatically so labels
   never clip outside cluster boxes.
5. **Tight `viewBox` expansion**: Recomputes SVG bounds over all transformed
   polygons, polylines, cluster paths, circles, rects, and estimated text widths
   (accounting for Graphviz's root `translate(gx gy)` transform) and expands
   `viewBox`, `width`, `height`, and the background polygon with a 10pt margin.

### Interactive HTML Viewer Architecture

- **DOM scoping**: Each rendered viewer generates a random `psyre_viz_<hex>`
  prefix scoping all CSS rules and element IDs so multiple schematics coexist
  safely on one page.
- **Safe inline JSON**: Serialized view metadata escapes `<`, `>`, and `&`
  (`_safe_inline_json`) to prevent `</script>` injection inside inline script
  blocks.
- **Gzip-compressed lazy view mounting**: When a multi-view bundle's combined
  raw SVG markup exceeds 256 KiB, only `initial_view_key` is inlined into the
  initial HTML; inactive views are gzip-compressed (`compresslevel=9`) and
  base64-encoded, decompressed on demand in the browser via
  `DecompressionStream("gzip")`, and mounted one view at a time to keep DOM size
  bounded on large hierarchical models. Monotonic sequence counters discard
  stale async view switches.
- **Click precedence**: Clicking inside the interactive canvas resolves targets
  in strict priority order (ignoring drag gestures > 4px):
  1. Collapsed box (`g.node[data-psyre-fsm-box]`): expands that hierarchy prefix
     (percent-escaped state keys matching `visualize.py`).
  2. Regular node (`g.node`): locks tooltip info in the footer and highlights
     the node, all stubs sharing its `data-psyre-net`, incident edges, and
     immediate neighbors while dimming the rest (`opacity: 0.22`).
  3. Edge (`g.edge`): checked **before** cluster boxes so clicking a wire inside
     an expanded automaton inspects the edge instead of collapsing the cluster.
  4. Cluster (`g.cluster[data-psyre-fsm-cluster]`): collapses that hierarchy
     prefix and its descendants.
  5. Background: clears selection.
- **Navigation controls**: Logarithmic vertical zoom rail with a 1:1 marker,
  Ctrl/Meta+wheel zoom, two-finger pinch and Safari gesture zoom, drag pan,
  double-click fit, `Expand All`/`Collapse All` toolbar buttons (shown when
  hierarchies exist), and `ResizeObserver` auto-fit for initially hidden tabs.
- **Fragment vs. standalone HTML**: `GraphVisualization.html` and
  `_repr_html_()` return a bare `<div id="...">` fragment suitable for notebook
  cell output, while `GraphVisualization.save_html()` prepends
  `<!DOCTYPE html>\n<meta charset="utf-8">\n` so standalone files render UTF-8
  math and schematic glyphs reliably in browsers.

## Public API Reference

### Types & Palette Constants

- `RankDir = Literal['LR', 'TB']`: Layout orientation.
- `NodeKind = Literal['when', 'whennot', 'merge', 'fby', 'post', 'ifthenelse', 'sigma', 'fifo', 'counter', 'static_stream', 'other']`: Supported custom glyph
  categories.
- Palette constants (each a `(fill, border, text)` hex color 3-tuple):
  `STYLE_SELECTOR`, `STYLE_MUX`, `STYLE_FBY`, `STYLE_STATEFUL`, `STYLE_COMB`,
  `STYLE_INPUT`, `STYLE_CONST`, `STYLE_NET_REF`, `STYLE_RST_REF`,
  `STYLE_OUTPUT`, `STYLE_CYCLE`.

### Data Classes

#### `NodeShapeSpec`

```python
@dataclasses.dataclass(frozen=True)
class NodeShapeSpec:
    kind: NodeKind
    border_color: str
    subscript: str = ''
```

Per-node post-processing directive driving custom SVG glyph rewriting.

#### `SchematicView`

```python
@dataclasses.dataclass(frozen=True)
class SchematicView:
    raw_svg: str
    dot_src: str
    rankdir: RankDir = 'LR'
    shape_specs: Mapping[str, NodeShapeSpec] = ...
    info_by_dom_id: Mapping[str, str] = ...
    net_by_dom_id: Mapping[str, str] = ...
    fsm_box_by_dom_id: Mapping[str, str] = ...
    fsm_cluster_by_dom_id: Mapping[str, str] = ...
    static_edge_to_node: Mapping[str, str] = ...
    reversed_edge_ids: frozenset[str] = frozenset()
```

Raw Graphviz output and DOM metadata for one hierarchy expansion view.

#### `SchematicBundle`

```python
@dataclasses.dataclass(frozen=True)
class SchematicBundle:
    views: Mapping[str, SchematicView]
    initial_view_key: str = ''
    fsm_prefixes: Sequence[str] = ()
    title: str = ''
    semantics_label: str = 'Psyre'
    node_count: int = 0
    height_px: int = 560
```

Complete multi-view schematic payload transferred from `psyre.visualize` to
`psyre.gui`. Validates `height_px` in `__post_init__` via `validate_height_px`.

#### `GraphVisualization`

```python
@dataclasses.dataclass(frozen=True)
class GraphVisualization:
    svg: str
    html: str
    dot: str
```

Container returned by `build_visualization` and `visualize.visualize_graph`.
`svg` and `dot` hold the initial view; `html` embeds all views in the bundle.

- `_repr_html_() -> str`: Returns `self.html` for automatic rich display in
  Jupyter and Colab.
- `save_svg(path: str | None = None) -> str`: Writes `self.svg` as UTF-8 to
  `path` (creating missing parent directories) or to a process-scoped temporary
  file cleaned up at exit, returning the written file path.
- `save_html(path: str | None = None) -> str`: Writes `self.html` with a
  `<!DOCTYPE html>` + `<meta charset="utf-8">` document header if absent,
  returning the written file path.

### Functions

#### `validate_height_px`

```python
def validate_height_px(height_px: int) -> int
```

Validates that `height_px` is a positive plain `int` (rejecting `bool` with
`TypeError` and `<= 0` with `ValueError`).

#### `render_view_svg`

```python
def render_view_svg(view: SchematicView) -> str
```

Runs the complete SVG post-processing pipeline on a single `SchematicView` and
returns the transformed UTF-8 SVG markup.

#### `build_visualization`

```python
def build_visualization(bundle: SchematicBundle) -> GraphVisualization
```

Post-processes every view in `bundle.views`, resolves `initial_view_key`
(falling back to `''`, then the first available key), wraps the interactive HTML
viewer, and returns a `GraphVisualization`. Raises `ValueError` if
`bundle.views` is empty.

#### `render_bundle_html`

```python
def render_bundle_html(bundle: SchematicBundle) -> str
```

Convenience helper returning `build_visualization(bundle).html`.

## See Also

- [`psyre.visualize`](visualize.md) — Stage-1 DOT/SVG compilation and
  multi-view bundling
