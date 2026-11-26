# `psyre.visualize`

Stage-1 schematic compiler translating finalized Psyre dataflow graphs into
Graphviz DOT, custom-styled SVG schematics, and interactive multi-view HTML
viewers.

## Overview & Role

`psyre.visualize` bridges Stage-1 graph inspection (`psyre.graph`,
`psyre.analysis`, `psyre.clock`) and browser/notebook presentation
([`psyre.gui`](gui.md)). Because schematic generation depends on `pydot` and
the external Graphviz `dot` executable (installed via `pip install "psyre[viz]"`
plus system `graphviz`), top-level `psyre` does not import `psyre.visualize`
eagerly—keeping core execution pure Python stdlib.

The visualization pipeline is cleanly split across two modules:

1. **`psyre.visualize`** validates the finalized `Graph`, runs clock and
   causality analysis once, computes hierarchical cluster trees and collapsible
   expansion states, builds `pydot.Dot` graphs, invokes `dot -Tsvg` in parallel
   across expansion views, and packages results into `SchematicView` /
   `SchematicBundle` payloads.
2. **[`psyre.gui`](gui.md)** performs pure-stdlib SVG post-processing (custom
   schematic glyphs, inline arrow shortening, cluster border widening, viewBox
   normalization) and wraps views in a self-contained interactive HTML/JS
   viewer.

## Design & Key Invariants

### Single-Pass Graph Pre-Analysis

Before generating any view, the visualizer runs graph-wide analysis once and
shares it immutably across all rendered expansion states:

- **Finalization & clock checking**: Requires `target_graph.require_finalized()`
  and invokes [`check_clocks(target_graph)`](clock.md) upfront, propagating any
  `ClockError` before layout.
- **Causality cycles are rendered, not rejected**: Runs
  [`find_instantaneous_cycle(target_graph)`](analysis.md). Nodes and directed
  edges participating in an instantaneous causality cycle are highlighted in red
  (`gui.STYLE_CYCLE`, `CAUSALITY_CYCLE=true` in hover tooltips) and are never
  elided into net-reference stubs, making causality errors inspectable directly
  in the schematic.
- **Explicit stream aliases**: Preserves both the producer node name and any
  distinct explicit stream alias recorded via `Graph.stream_names()`, exposing
  both in tooltips.
- **`node.prefixes`-driven hierarchy**: Automaton and `@psyre_node` hierarchy
  prefixes are read directly from `node.prefixes` (joining segment names with
  `'/'`) rather than splitting `node.name` by `'/'`. This avoids misclassifying
  top-level streams containing slashes or unnamed nodes nested inside automata.
- **Base activity clock per hierarchy**: Each prefix's base subclock is the
  longest common `ClockStep` prefix across its clocked member nodes (nodes with
  at least one input or reset), preventing source-less constant nodes from
  collapsing an entire subclocked automaton back to the base clock `.`.

### Strict Tree-Nested Dual Clustering

Graphviz requires clusters to form a strict tree. `psyre.visualize` combines
hierarchy clusters (`@psyre_node` / `automaton` scopes) and synchronous `on`
subclock boxes (`group_by_activity=True`) into one tree:

- Each hierarchy cluster is placed inside its parent hierarchy at its base
  activity clock.
- Deeper subclocks nest step-by-step inside the hierarchy cluster; members whose
  activity clock diverges from the prefix base are clamped to preserve valid DOT
  cluster nesting.
- Inner subclock clusters inside a hierarchy inherit its collapse target
  (`fsm_cluster_by_dom_id`), so clicking an inner subclock box collapses the
  enclosing hierarchy.
- Visible nodes are sorted by `(len(node.prefixes), len(clock))` for
  deterministic cluster and layout ordering.

### Collapsible Expansion Views & State Enumeration

Interactive views (`visualize_graph` and `render_html`) pre-render multiple
hierarchy expansion states so users can expand and collapse automata/subgraphs
without a live Python kernel:

- **Combinatorial pruning**: For each hierarchy node, sibling combinations take
  a Cartesian product only while the total product is `<= 8`. Parents with more
  than 4 leaf children group those leaves into a single toggle, and larger
  branching factors fall back to a linear union (`empty`, individual child
  states, and `all direct children`). Both fully collapsed (`""`) and fully
  expanded states are always included.
- **Ancestor closure**: In `visualize_graph` / `render_html`, explicitly
  requested `expanded_automata` prefixes are automatically ancestor-closed and
  guaranteed to be included in the rendered view set. In direct single-view
  entry points (`build_pydot`, `render_svg`), `expanded_automata` is validated
  against known prefixes without automatic ancestor closure (passing a child
  without its parent leaves the parent collapsed).
- **State keys**: Multi-prefix state keys are sorted, percent-escaped
  (`%` → `%25`, `|` → `%7C`), and joined with `|`, ensuring prefix names
  containing `|` or `%` round-trip safely with `gui.py`.
- **Collapsed boxes**: A collapsed hierarchy renders as a double-bordered
  `⊞ <prefix> (N nodes)` box whose external input and reset ports deduplicate
  member boundary wires while omitting intra-box wires and unnamed scalar
  constants.

### Schematic Styling, Vertical Ports & Inline Static Sources

- **Operator glyphs & shapes**: Built-in `Stream` dunder operators map to math
  symbols (`×`, `∧`, `∨`, `⊕`, `¬`, `|·|`, `≤`, `=`, `≠`, `≥`, etc.). Primitive
  operators emit `gui.NodeShapeSpec` metadata for custom SVG shapes (`fby`/
  `pre`, `post`, `when`, `whennot`, `merge`/`switch`, `ifthenelse`,
  `prefix_sum`, `buffer`, `counter`). Primitive identity is checked via
  `node.is_primitive(...)` rather than `node.op` string comparison so user
  functions sharing builtin names are not misstyled.
- **LR vertical input ports**: In left-to-right layout (`rankdir='LR'`), nodes
  with 2 or more non-selector left inputs (excluding `ifthenelse`, which uses
  `:nw`/`:sw` compass ports) render as HTML-like `<TABLE>` labels with stacked
  `<TD PORT="in{idx}">` cells so incoming wires stay visually separated.
  Selector ports (`port 0` on `when`, `whennot`, `merge`, `switch`,
  `ifthenelse`) enter from the top (`:n` in LR, `:w` in TB).
- **Inline static sources**: Static constant/periodic streams without a producer
  node are duplicated per consumer as boxless `plaintext` nodes inside the
  consumer's container and shortened into compact labeled stubs during SVG
  post-processing, avoiding long cross-graph constant wires. Dynamic external
  inputs remain shared top-level input nodes.
- **Same-name collinear chains**: Visible nodes in the same cluster sharing a
  custom non-default local name are linked with high-weight invisible DOT edges
  (`group="same_name_..."`, `weight=20`) to align successive stages of the same
  signal horizontally.

### Constrained `dir="back"` Feedback Routing

Rather than marking all delayed feedback wires `constraint="false"` (which
excludes them from Graphviz crossing minimization and causes long spline
detours), `psyre.visualize` tracks a constrained-edge DAG during emission:

- Each visible `fby`/`pre` node designates its lowest-ID forward same-cluster
  consumer as its primary immediate successor (`weight=10`, always kept direct).
- Direct delayed feedback edges (`fby` port `>= 1` and `pre`, excluding
  self-loops and primary forward edges) that would close a cycle in the
  constrained DAG are emitted **reversed** in DOT with `dir="back"`,
  `constraint="true"`, `weight="2"`, and dashed purple styling (`#7C3AED`).
- Reversed edge DOM IDs are recorded in `SchematicView.reversed_edge_ids` so
  [`gui.render_view_svg`](gui.md) restores the true producer-to-consumer
  `data-psyre-src` / `data-psyre-dst` attributes for interactive highlighting.
- Direct self-loops remain `constraint="false"`.

### Large-Graph Net-Reference Stubs

When a view has more than 80 visible elements (`visible_nodes +
collapsed_boxes > 80`), long, cross-cluster, high-fanout, backward, or
non-inline reset wires are replaced by compact local `↪` net-reference stubs
(while keeping causality-cycle edges, inline static sources, and primary `fby`
edges direct). Producers whose outgoing wires are all replaced receive a
`↪ <name> (×N)` fanout badge sharing the same `net_by_dom_id` identifier, so
clicking any stub or producer highlights the full logical net in the viewer.

### Graphviz Subprocess & Parallel Rendering

- **Binary lookup**: Resolves `dot_prog` if explicitly supplied, otherwise
  checks for a bundled `dot` binary relative to the package, then falls back to
  `shutil.which('dot')`, raising `FileNotFoundError` with installation
  instructions if absent.
- **Non-fatal stderr handling**: If `dot` exits non-zero (for example, due to
  headless font warnings) but produces valid `<svg>...</svg>` output, a warning
  is logged via `psyre.common.logger` and the SVG is returned; missing SVG
  markers raise `RuntimeError`.
- **Concurrency**: Multi-view bundles run Graphviz `dot` concurrently across
  expansion states using a `ThreadPoolExecutor` (capped at `min(8, num_views,
  cpu_count)`).

## Public API Reference

### Re-Exports

- `RankDir`: Type alias `Literal['LR', 'TB']` re-exported from
  [`psyre.gui`](gui.md).
- `GraphVisualization`: Frozen dataclass bundling `.svg`, `.html`, and `.dot`
  re-exported from [`psyre.gui`](gui.md).

### Functions

#### `build_pydot`

```python
def build_pydot(
    target_graph: Graph,
    *,
    rankdir: RankDir = 'LR',
    show_constants: bool = True,
    group_by_activity: bool = True,
    show_reset_edges: bool = True,
    collapse_automata: bool = False,
    expanded_automata: Iterable[str] | None = None,
) -> pydot.Dot
```

Builds the single-view `pydot.Dot` representation of a finalized `Graph`.

- **Parameters**:
  - `target_graph`: Finalized Stage-1 `Graph`.
  - `rankdir`: Graphviz layout direction (`'LR'` or `'TB'`).
  - `show_constants`: When `False`, hides unnamed inline scalar constants.
  - `group_by_activity`: When `True`, wraps subclocked nodes in dashed clock
    clusters.
  - `show_reset_edges`: When `False`, omits reset edges from the schematic.
  - `collapse_automata`: When `True`, collapses hierarchy prefixes not listed in
    `expanded_automata` into summary boxes. Defaults to `False` for static
    single-view inspection.
  - `expanded_automata`: Optional hierarchy prefixes to keep expanded when
    `collapse_automata=True` (validated against known prefixes; not
    auto-ancestor-closed in `build_pydot`).
- **Raises**: `ValueError` on invalid `rankdir` or unknown prefix in
  `expanded_automata`; `RuntimeError` if `target_graph` is not finalized;
  `ClockError` if static clock analysis fails.

#### `render_svg`

```python
def render_svg(
    target_graph: Graph,
    *,
    rankdir: RankDir = 'LR',
    show_constants: bool = True,
    group_by_activity: bool = True,
    show_reset_edges: bool = True,
    collapse_automata: bool = False,
    expanded_automata: Iterable[str] | None = None,
    dot_prog: str | None = None,
) -> str
```

Compiles a single schematic view through Graphviz `dot -Tsvg` and applies
[`gui.render_view_svg`](gui.md) post-processing, returning a UTF-8 SVG string.

- **Parameters**: Same layout parameters as `build_pydot`, plus optional
  `dot_prog` path to the Graphviz `dot` executable.
- **Raises**: Same validation errors as `build_pydot`, plus `FileNotFoundError`
  if `dot` cannot be found and `RuntimeError` if `dot` fails to produce valid
  SVG.

#### `visualize_graph`

```python
def visualize_graph(
    target_graph: Graph,
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
) -> GraphVisualization
```

Builds a multi-view `SchematicBundle` (rendering expansion views in parallel)
and returns a [`GraphVisualization`](gui.md#graphvisualization) object that
renders inline in Jupyter/Colab via `_repr_html_()` and supports `.save_svg()`
and `.save_html()`.

- **Notes**: Defaults to `collapse_automata=True` for interactive drill-down;
  automatically ancestor-closes `expanded_automata` and guarantees the requested
  initial state is present in the bundle. Validates `height_px` as a positive
  non-boolean `int` (`TypeError` / `ValueError`).

#### `render_html`

```python
def render_html(
    target_graph: Graph,
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
) -> str
```

Convenience wrapper returning `visualize_graph(...).html` (an embeddable HTML
`<div>` fragment with scoped styles and interactive JavaScript).

## See Also

- [`psyre.gui`](gui.md) — SVG glyph post-processing and interactive viewer
- [`psyre.graph`](graph.md) — Stage-1 `Graph` and `Node` IR
- [`psyre.clock`](clock.md) — Synchronous subclock analysis
- [`psyre.analysis`](analysis.md) — Instantaneous causality cycle detection
