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

> **Note:** Psyre streams store values by reference without copying them or
> enforcing runtime immutability. Treat values carried on streams as immutable;
> the behavior of mutating values on streams is unspecified.

## Repository Layout

- **Documentation** (`docs/`): Per-module architecture and public API reference
  covering the core language, visualization, experimental `psyre.equations`, and
  optional `psyre.numpy` / `psyre.jax` packages—start at
  [docs/README.md](docs/README.md).
- **Core package**: Stream representation, Stage-1 graph construction and pure
  dataflow IR, synchronous and Kahn primitives, pattern matching and
  hierarchical automata lowering, static causality and clock analyses,
  whole-program graph optimization, Stage-2 runtime scheduler, post-run trace
  inspection, and interactive SVG visualization.
- **Examples** (`examples/src/`):
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
- **Tests** (`tests/` and `examples/tests/`): Unit test suites covering
  semantics, lowering, static analyses, optimization, visualization, and all
  examples.
- **Equations** (`psyre.equations` — *experimental*): An opt-in, lambda-free
  declarative syntax for recursive stream equations (`equations.py`, plus its
  dedicated `tests/` and `examples/{src,tests}/` inside the `equations/`
  subpackage — exported under `src/psyre/equations/` on GitHub). APIs,
  semantics, and internals may change without notice.
- **NumPy Lifting** (`psyre.numpy` — *optional*): Standalone optional stream-level
  lifting proxy for `numpy` ufuncs, array functions, `linalg`/`fft`/`random`
  submodules, `stack_streams`/`unstack_stream`/`batch_streams`, eager
  pointwise `vmap`, and a stride-group MNIST CNN example under
  `psyre.numpy.examples` (requires NumPy; core `psyre` stays pure Python
  stdlib).
- **JAX & `lax` Lifting** (`psyre.jax` — *optional*):
  Optional subpackage lifting pure `jax.lax` (`psyre.jax.lax`), `jax.numpy`
  (`pjax.jnp`), `jax.nn`, `jax.random`, and `jax.scipy` operations onto
  `Stream`s with canonical JAX `Primitive` metadata, lifting-time
  transformations (`vmap` over both single lifted ops and composite stateful
  Psyre subgraphs, instantaneous `grad` / `value_and_grad`, `stack_streams`,
  `unstack_stream`, `batch_streams`, and `rewrite_dtype`), `psyre.jax.jit`
  (lazy compilation, maximal same-clock convex island fusion via
  `fuse_jit_islands`, and opt-in Stage-1 `Jaxpr` expansion gated behind
  `trace_jaxpr=True` with explicit `ShapeDtypeStruct` hints), and an eager and
  JIT-compiled MNIST CNN example under `psyre.jax.examples`.

## Installation

Psyre requires **Python >= 3.12**. Install the core package in editable mode,
with optional extras for NumPy (`numpy`), JAX + `lax` (`jax`), Graphviz
visualization (`viz`), or the full test suite (`test`, which includes
visualization dependencies):

```bash
pip install -e .          # core runtime (pure Python stdlib)
pip install -e ".[numpy]" # + optional psyre.numpy stream-lifting module
pip install -e ".[jax]"   # + optional psyre.jax (including psyre.jax.lax)
pip install -e ".[viz]"   # + Graphviz/pydot schematic visualization
pip install -e ".[test]"  # + pytest / absl-py test suite (includes [viz])
pip install -e ".[test,jax]" # + full optional NumPy / JAX test suites
```

Rendering interactive schematics also requires the Graphviz `dot` binary on your
system `PATH` (e.g., `apt install graphviz` or `brew install graphviz`).

## Running Tests

```bash
pytest
```

## References

- **Halbwachs, Caspi, Raymond & Pilaud (1991)** — *The Synchronous Data Flow Programming Language LUSTRE*. *Proceedings of the IEEE*, 79(9):1305–1320. https://doi.org/10.1109/5.97300 (Origin: Caspi, Pilaud, Halbwachs & Plaice, *POPL '87*, pp. 178–188, https://doi.org/10.1145/41625.41641.)
- **Kahn (1974)** — *The Semantics of a Simple Language for Parallel Programming*. *Information Processing 74 (IFIP Congress '74)*, North-Holland, pp. 471–475. (Coroutine execution model: Kahn & MacQueen, *Coroutines and Networks of Parallel Processes*, *Information Processing 77 (IFIP '77)*, pp. 993–998.)
- **Berry & Gonthier (1992)** — *The ESTEREL Synchronous Programming Language: Design, Semantics, Implementation*. *Science of Computer Programming*, 19(2):87–152. https://doi.org/10.1016/0167-6423(92)90005-V (See also Berry, *The Foundations of Esterel*, MIT Press, 2000, pp. 425–454, for the canonical ABRO benchmark.)
- **Caspi & Pouzet (1996)** — *Synchronous Kahn Networks*. *ICFP '96*, pp. 226–238. https://doi.org/10.1145/232627.232651
- **Hamon & Pouzet (2000)** — *Modular Resetting of Synchronous Data-Flow Programs*. *PPDP '00*, pp. 289–300. https://doi.org/10.1145/351268.351300
- **Colaço & Pouzet (2003)** — *Clocks as First Class Abstract Types*. *EMSOFT '03*, LNCS 2855, Springer, pp. 134–155. https://doi.org/10.1007/978-3-540-45212-6_10
- **Maraninchi & Rémond (2003)** — *Mode-Automata: A New Domain-Specific Construct for the Development of Safe Critical Systems*. *Science of Computer Programming*, 46(3):219–254. https://doi.org/10.1016/S0167-6423(02)00093-X
- **Colaço, Pagano & Pouzet (2005)** — *A Conservative Extension of Synchronous Data-Flow with State Machines*. *EMSOFT '05*, pp. 173–182. https://doi.org/10.1145/1086228.1086261
- **Cohen, Duranton, Eisenbeis, Pagetti, Plateau & Pouzet (2006)** — *N-Synchronous Kahn Networks: A Relaxed Model of Synchrony for Real-Time Systems*. *POPL '06*, pp. 180–193. https://doi.org/10.1145/1111037.1111054
- **Colaço, Pagano & Pouzet (2017)** — *SCADE 6: A Formal Language for Embedded Critical Software Development (Invited Paper)*. *TASE 2017*, pp. 1–11. https://doi.org/10.1109/TASE.2017.8285623 (See also Pouzet, *Lucid Synchrone, version 3: Tutorial and Reference Manual*, 2006, https://www.di.ens.fr/~pouzet/lucid-synchrone/; Bourke, Brun & Pouzet, *PACMPL* 4(POPL), Art. 44, 2020, https://doi.org/10.1145/3371112; and Bourke, Pesin & Pouzet, *ACM TECS* 22(5s), Art. 137, 2023, https://doi.org/10.1145/3608102.)
- **Potop-Butucaru, Cohen, Plotkin & Pompougnac (2023)** — *Bidirectional Reactive Programming for Machine Learning*. Preprint [arXiv:2311.16977](https://arxiv.org/abs/2311.16977) [cs.PL], 2023 (HAL: `hal-04354071`).
- **Gaudelier, Cohen & Potop-Butucaru (2026)** — *Relaxed Activation Analysis of Dataflow Networks: A Clock Calculus for Machine Learning and Real-Time Scheduling*. *Proc. 29th Forum on Specification & Design Languages (FDL 2026)*, pp. 1–7. Extended preprint: [arXiv:2607.21797](https://arxiv.org/abs/2607.21797) [cs.PL].

## Security

Eligibility for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security)
is determined by the
[Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
