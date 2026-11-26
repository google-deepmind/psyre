# `psyre.analysis`

Whole-program static instantaneous causality checking and topological node
ordering for finalized Psyre graphs.

## Overview & Role

In synchronous dataflow languages (Lustre, Lucid Synchrone), a program is
statically causal when every feedback cycle passes through at least one unit
delay (`fby` or `pre`). `psyre.analysis` verifies this property on a finalized
[`Graph`](graph.md) before clock analysis or Stage-2 execution, and supplies a
deterministic causal node order reused by [`psyre.clock`](clock.md),
[`psyre.optimization`](optimization.md), [`psyre.jax.jit`](jax_jit.md), and
[`psyre.scheduler`](scheduler.md).

## Design & Key Invariants

### Which Edges Break Cycles

Instantaneous dependencies are read exclusively from
`Node.instantaneous_input_streams()`:

- **Cycle-breaking (delayed) edges**: `pre(x)` (`inputs[0]`) and the recursive
  tail of `fby(init, x)` (`inputs[1:]`).
- **Instantaneous edges (do *not* break cycles)**: `fby`'s initial value
  (`inputs[0]`), all enclosing `resets`, `post`, `when`, `whennot`, `merge`,
  `switch`, `current`, `morph`, `ifthenelse`, `buffer` (because `buffer`
  executes `pull` before `push`, so an empty buffer passes through on the same
  tick), and all combinational/sequential user nodes.

### Deterministic Peeling & Elementary Cycle Extraction

- **A. B. Kahn's (1962) topological peeling**: Computes instantaneous in-degrees
  across `graph.nodes` and peels zero-in-degree nodes in deterministic FIFO
  instantiation order (`Node.id` ascending). (Note: this is A. B. Kahn's 1962
  topological sort algorithm, distinct from Gilles Kahn's 1974 Process Networks.)
- **Elementary cycle extraction**: If any nodes remain unpeeled, the checker
  walks backward deterministically from `min(remaining_ids)` until revisiting a
  node `seed_id` guaranteed to lie on a cycle, then runs forward BFS restricted
  to `remaining_ids` from `seed_id` back to itself to extract a short elementary
  cycle. The cycle tuple is rotated to start at its minimum `Node.id` so
  `CausalityError` diagnostics are deterministic across runs.
- **Total ordering fallback (`causal_node_order`)**: Returns the peeled acyclic
  prefix followed by any remaining cyclic nodes in ascending `Node.id` order.
  This allows clock inference, optimization, visualization, and unguarded
  (`causality_analysis=False`) execution to operate on cyclic graphs without
  crashing during ordering.

## Public API Reference

All symbols except `topological_sort` are re-exported from top-level `psyre`.
Every function requires a finalized `Graph` (`RuntimeError` otherwise).

### `CausalityError`

```python
class CausalityError(RuntimeError):
    cycle: tuple[Node, ...]

    def __init__(self, cycle: Sequence[Node]) -> None
```

Raised when a `Graph` contains an instantaneous dependency cycle. Stores the
elementary cycle nodes in `.cycle` and formats them as
`'Instantaneous causality cycle detected: n0 -> n1 -> ... -> n0'`. Raises
`ValueError` if constructed with an empty `cycle`.

### `find_instantaneous_cycle`

```python
def find_instantaneous_cycle(graph: Graph) -> tuple[Node, ...] | None
```

Returns a deterministic elementary instantaneous cycle `tuple[Node, ...]`, or
`None` if the graph is statically causal. Used by
[`psyre.optimization`](optimization.md) and [`psyre.visualize`](visualize.md) to
inspect cycles without raising.

### `check_causality`

```python
def check_causality(graph: Graph) -> None
```

Validates that `graph` has no instantaneous cycles, raising `CausalityError` if
one exists.

### `topological_sort`

```python
def topological_sort(graph: Graph) -> tuple[Node, ...]
```

Returns all nodes of `graph` in deterministic causal topological order, raising
`CausalityError` on any instantaneous cycle.

### `causal_node_order`

```python
def causal_node_order(graph: Graph) -> tuple[Node, ...]
```

Returns the peeled causal prefix followed by any cyclic nodes in `Node.id`
order. Never raises `CausalityError`.

## See Also

- [`psyre.graph`](graph.md#builtin-identity--port-classification) — `Node.instantaneous_input_streams()` and `Node.delayed_input_streams()`
- [`psyre.clock`](clock.md) — Synchronous clock inference following causal order
- [`psyre.visualize`](visualize.md) — Rendering causality cycles in schematics
