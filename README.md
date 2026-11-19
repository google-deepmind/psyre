# Psyre: Reactive Dataflow in Python

**Psyre** is a two-stage Python domain-specific language (DSL) for authoring,
statically analyzing, optimizing, visualizing, and executing synchronous
(Lustre / Lucid Synchrone style) and asynchronous (Kahn Process Network)
reactive stream programs.

## Key Features

- **Unified Synchronous & Kahn Semantics**: Express stream equations once using
  standard Python operators, reactive delay and clock-sampling combinators
  (including multi-valued clock predicates, same-clock multiplexing, and
  asynchronous buffering), sequential reductions, and custom stateless or
  stateful primitives, then execute under either synchronous or Kahn Process
  Network semantics.
- **Hierarchical Automata, Pattern Matching & Subgraphs**: Write nested state
  machines with weak and strong transitions, scoped resets, inter-state memory,
  clocked pattern matching, and modular subgraph hierarchies that lower cleanly
  to primitive dataflow operators.
- **Two-Stage Architecture & Whole-Program Optimization**: Stage-1 elaboration
  builds a standalone, inspectable dataflow graph IR decoupled from Stage-2
  runtime scheduling and trace inspection. Fixed-point graph optimization
  performs clock-preserving algebraic and boolean simplification, constant and
  mux folding, pure combinational common-subexpression elimination, and
  dead-code elimination.
- **Static Causality & Synchronous Clock Calculus**: Static analyses verify that
  every instantaneous feedback cycle is broken by a unit delay and infer and
  validate hierarchical subclocks across sampling, merging, multiplexing,
  buffering, resets, pattern matches, and automata.
- **Interactive Schematic Visualization**: Render Stage-1 dataflow graphs into
  interactive pan/zoom SVG schematics in Jupyter and Colab with collapsible
  automaton and subgraph hierarchies, subclock annotations, feedback back-edge
  styling, and neighborhood highlighting.

## Repository Layout

- **Core package**: Stream representation, Stage-1 graph construction and pure
  dataflow IR, synchronous and Kahn primitives, pattern matching and
  hierarchical automata lowering, static causality and clock analyses,
  whole-program graph optimization, Stage-2 runtime scheduler, post-run trace
  inspection, and interactive SVG visualization.
- **Examples** (`examples/`):
  - *Recurrent Neural Network & Perceptron*: Variable-depth unrolled RNN and
    perceptron with bidirectional reverse-mode automatic differentiation and
    multi-mode training/inference control.
  - *Sparsely-Gated Mixture of Experts*: Dynamic subclock dispatch of sparse
    top-2 experts with reverse-mode AD and training—both standalone scalar
    experts and a chained configuration running heterogeneously unrolled RNN
    experts with inactive experts kept absent and frozen.
  - *Systolic Array*: Weight-stationary and output-stationary tiled matrix
    multiplication with distributed per-PE state machines and shared stream
    skew/deskew control.
- **Notebooks** (`notebooks/`): Interactive Colab/Jupyter walkthroughs covering
  canonical reactive idioms, the RNN model, the sparsely-gated Mixture of
  Experts, and the systolic array.
- **Tests** (`tests/` and `examples/`): Unit test suites covering semantics,
  lowering, static analyses, optimization, visualization, and all examples.

## Installation

```bash
pip install -e ".[test]"
```

Rendering interactive schematics also requires the Graphviz `dot` binary on your
system `PATH` (e.g., `apt install graphviz` or `brew install graphviz`).

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
