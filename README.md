# Psyre: Reactive Dataflow in Python

**Psyre** is a two-stage Python domain-specific language (DSL) for
building, statically analyzing, visualizing, and executing synchronous (Lustre /
Lucid Synchrone style) and asynchronous (Kahn Process Network) reactive stream
programs.

## Key Features

- **Unified Synchronous & Kahn Semantics**: Write stream equations once using
  Python operators (`+`, `-`, `*`, `==`, `<`, `&`, `|`, ...), reactive
  combinators (`fby`, `pre`, `post`, `morph`, `when`, `whennot`, `merge`,
  `buffer`, `reset`, `current`), and sequential combinators (`scan`,
  `prefix_sum`), or lift custom Python functions with `@psyre_comb`,
  `@psyre_sequ`, and `@psyre_loop`, and evaluate these stream equations under
   either `Semantics.SYNC` or `Semantics.KAHN`.
- **Hierarchical Automata, Nodes & Pattern Matching**: Express nested state
  machines, hierarchical subgraphs (`@psyre_node`), and clocked mode switches
  directly in Python (`match`, `case`, `default`, `automaton`, `state`, `until`,
  `unless`, `last`) that compile down to primitive `when`, `merge`, `fby`, and
  `reset` nodes.
- **Stage-1 Whole-Program Graph & Dead-Code Elimination**:
  `Scheduler.build_graph()` constructs an inspectable dataflow `Graph` and
  eliminates unreachable nodes and self-referential dead feedback loops
  (`optimization.dce()`).
- **Static Causality Analysis & Synchronous Clock Calculus**:
  `analysis.check_causality()` verifies that every instantaneous dependency
  cycle is broken by a unit delay (`fby` / `pre`), and `clock.check_clocks()`
  infers hierarchical Lustre subclocks across `when`, `merge`, `buffer`,
  `reset`, `match`, and `automaton`.
- **Interactive Schematic Visualization**: `visualize.visualize_graph()` renders
  Stage-1 `Graph`s into interactive pan/zoom SVG schematics in Jupyter and Colab
  with collapsible hierarchical automaton and `@psyre_node` boxes, clock
  annotations, and neighborhood highlighting.

## Installation

```bash
pip install -e ".[test]"
```

Rendering schematics with `visualize.visualize_graph()` also requires the
Graphviz `dot` binary on your system `PATH` (e.g., `apt install graphviz` or
`brew install graphviz`).

## Running Tests

```bash
pytest
```

## Security

Eligibility for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security)
is determined by the
[Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
