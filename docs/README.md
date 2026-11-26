# Psyre Documentation

Reference documentation for **Psyre**, a two-stage reactive dataflow DSL in
Python unifying synchronous Lustre / Lucid Synchrone semantics
(`Semantics.SYNC`) and asynchronous Kahn Process Networks (`Semantics.KAHN`),
together with optional NumPy and JAX array-lifting packages.

## Two-Stage Architecture

Psyre cleanly separates **Stage-1 elaboration and static compilation** from
**Stage-2 runtime execution**:

```text
User Python function (Stream ops, primitives, match/automaton, @psyre_comb/@psyre_sequ/@psyre_node)
  │
  ▼  Stage 1: GraphBuilder / build_graph()  [builder.py, match.py, automaton.py]
Finalized Graph IR (Nodes + single-assignment Streams)  [graph.py]
  ├─► Static causality analysis (check_causality)        [analysis.py]
  ├─► Colaço-Pouzet clock unification (check_clocks)     [clock.py]
  ├─► Whole-program optimization (optimize_graph)        [optimization.py]
  ├─► Optional JAX graph passes (rewrite_dtype, jit_graph) [jax/transforms.py, jax/jit.py]
  └─► Interactive SVG schematic visualization            [visualize.py, gui.py]
  │
  ▼  Pre-execution lowering: lower_resets(graph, clocks)  [builder.py]
  │
  ▼  Stage 2: Scheduler.run_graph()  [scheduler.py, sync_primitives.py, kahn_primitives.py]
Post-run traces (list[Any] / PyTree) + trace helpers     [trace.py]
```

## Core Design Principles

- **Zero-dependency pure-stdlib core**: Core `psyre` requires only Python >=
  3.12 with zero runtime dependencies. Optional array and visualization
  capabilities live behind extras (`psyre[numpy]`, `psyre[jax]`, `psyre[viz]`).
- **Strict first-order streams & immutability contract**: Streams carry
  first-order values—never callables or nested `Stream`s (enforced by
  [`psyre.payload.reject_callable`](payload.md#strict-first-order-payload-enforcement-reject_callable)).
  Values are stored by reference without deep-copying; treat stream payloads as
  immutable.
- **Distinct `Absent`, `Bottom`, and `None`**: `None` is an ordinary valid
  stream value, `Absent` marks inactive synchronous subclock ticks (`SYNC`
  only), and `Bottom` marks uninitialized internal slots and post-horizon EOF
  trace padding ([`psyre.common`](common.md#sentinels-absent-bottom-none-unset-and-unready)).
- **Uniform static clock polymorphism (`∀ 'a. 'a`)**: All unpinned
  producer-less static streams (both 1-periodic constants and multi-element
  periodic sequences backed by symbolic [`StaticWord`](static_stream.md)s)
  instantiate independently at each consumer's clock unless explicitly anchored
  (e.g. `when(Stream(True), seq)`).
- **Mandatory `SYNC` clock calculus & `SYNC`/`KAHN` parity**: Static clock
  unification and `lower_resets()` are mandatory under `Semantics.SYNC` (and
  automatically enforced under `Semantics.KAHN` whenever subclocked resets,
  static ports, or `current()` inputs require lowering), guaranteeing matching
  traces under `Absent`/`Bottom` elision (`present_values`).
- **Step-function identity over string matching**: Primitives and pure
  combinational operators are identified by step-function identity and metadata
  (`Node.is_primitive`, `common.same_step_func`) rather than `node.op` strings,
  keeping user functions safe from builtin-name collisions and eliminating
  closure caches.

## Top-Level `psyre` Package Exports

Importing `psyre` (`__version__ = '0.1.0'`) re-exports the core language facade:

- **Streams & static words**: `Stream`, `resolve_stream`, `StaticWord`,
  `StaticSequence`, `EMPTY_WORD`, `repeat`, `concat`, `Absent`, `Bottom`,
  `LazyThunk`, `is_thunk`, `force_thunk`
- **Primitives & math**: `fby`, `pre`, `post`, `morph`, `when`, `whennot`,
  `merge`, `switch`, `ifthenelse`, `current`, `buffer`, `counter`,
  `prefix_sum`, `scan`, `reset`, `exp`, `tanh`, `random_stream`
- **Control submodules & decorators**: `match`, `automaton`, `psyre_comb`,
  `psyre_sequ`, `psyre_loop`, `psyre_node`, `is_pure_step_func`,
  `register_pytree_node`, `unregister_pytree_node`
- **Stage-1 IR, analysis & optimization**: `Graph`, `Node`, `GraphBuilder`,
  `build_graph`, `lower_resets`, `is_building`, `Semantics`,
  `register_primitives`, `require_sync_semantics`, `CausalityError`,
  `check_causality`, `find_instantaneous_cycle`, `causal_node_order`,
  `BASE_CLOCK`, `Clock`, `ClockStep`, `ClockAnalysis`, `ClockError`,
  `check_clocks`, `optimize_graph`, `eliminate_dead_code`, `find_live_nodes`,
  `find_dead_nodes`
- **Stage-2 execution & trace inspection**: `Scheduler`, `is_present`,
  `is_absent`, `is_bottom`, `present_values`, `logger`

*(Visualization [`psyre.visualize`](visualize.md) / [`psyre.gui`](gui.md),
experimental [`psyre.equations`](equations.md), and optional
[`psyre.numpy`](numpy.md) / [`psyre.jax`](jax.md) are imported explicitly from
their respective submodules.)*

## Module Documentation Index

### Core Language & Runtime

| Module | Document | Summary |
|---|---|---|
| `psyre.payload` | [payload.md](payload.md) | Zero-dep payload validation, array inspection, PyTrees, `CombCallPlan`, and lifting factories |
| `psyre.common` | [common.md](common.md) | Sentinels, `Semantics`, selectors, Stage-1 stacks, `RingBuffer`, decorators, `same_step_func`, primitive dispatchers |
| `psyre.static_stream` | [static_stream.md](static_stream.md) | Symbolic `StaticWord` grammar, `StaticSequence`, `same_const_value`, resettable `static_stream` kernel |
| `psyre.stream` | [stream.md](stream.md) | Unified `Stream` channel, operators, NEP-13/NEP-18 hooks, `StreamPort`, sliding-window eviction |
| `psyre.math` | [math.md](math.md) | Scalar `exp`, `tanh`, and impure `random_stream` |
| `psyre.trace` | [trace.md](trace.md) | Post-run `is_present`, `is_absent`, `is_bottom`, `present_values` |
| `psyre.sync_primitives` | [sync_primitives.md](sync_primitives.md) | Synchronous `Absent`-propagating primitive kernels (`Semantics.SYNC`) |
| `psyre.kahn_primitives` | [kahn_primitives.md](kahn_primitives.md) | Asynchronous KPN primitive kernels (`Semantics.KAHN`) |
| `psyre.graph` | [graph.md](graph.md) | Pure Stage-1 `Graph` and `Node` IR, alias compression, selector canonicalization, cloning |
| `psyre.analysis` | [analysis.md](analysis.md) | Static instantaneous causality checking and topological ordering |
| `psyre.clock` | [clock.md](clock.md) | Colaço-Pouzet constraint-based clock unification, `Clock`, `ClockStep`, `ClockAnalysis` |
| `psyre.optimization` | [optimization.md](optimization.md) | Fixed-point algebraic/boolean/mux folding, cache-free pure CSE, and DCE |
| `psyre.builder` | [builder.md](builder.md) | Stage-1 `GraphBuilder`, `build_graph`, `lower_resets`, `requires_clock_analysis` |
| `psyre.match` | [match.md](match.md) | Clocked pattern matching (`match`/`case`/`default`), `last`, arm confinement & lowering |
| `psyre.automaton` | [automaton.md](automaton.md) | Hierarchical mode automata (`automaton`/`state`/`until`/`unless`) |
| `psyre.scheduler` | [scheduler.md](scheduler.md) | Stage-2 `asyncio` `Scheduler`, EOF/deadlock handling, eviction, trace extraction |

### Visualization & Experimental Equations

| Module | Document | Summary |
|---|---|---|
| `psyre.visualize` | [visualize.md](visualize.md) | Stage-1 Graphviz DOT/SVG compiler, collapsible hierarchies, constrained feedback routing |
| `psyre.gui` | [gui.md](gui.md) | Pure-stdlib SVG glyph post-processing and interactive HTML/JS viewer |
| `psyre.equations` | [equations.md](equations.md) | *(Experimental)* Lambda-free declarative recursive stream equations (`psyre_equations`, `Equations`, `Thunk`) |

### Optional Array Packages

| Module | Document | Summary |
|---|---|---|
| `psyre.numpy` | [numpy.md](numpy.md) | *(Optional)* NumPy stream-lifting proxy, multi-output unpacking, `batch_streams`, eager `vmap` |
| `psyre.jax` & `psyre.jax.ops` | [jax.md](jax.md) | *(Optional)* JAX package entry point, `jnp`/`nn`/`random`/`scipy`, `lift`, `to_jax`/`to_numpy`, `register_jax_pytree` |
| `psyre.jax.lax` | [jax_lax.md](jax_lax.md) | *(Optional)* Lifted `jax.lax` & `jax.lax.linalg` with canonical JAX `Primitive` metadata |
| `psyre.jax.transforms` | [jax_transforms.md](jax_transforms.md) | *(Optional)* Lifting-time `vmap` (single op & composite subgraphs), `grad`, `value_and_grad`, `rewrite_dtype` |
| `psyre.jax.jit` | [jax_jit.md](jax_jit.md) | *(Optional)* Lazy `jit`, convex same-clock `fuse_jit_islands`/`jit_graph`, opt-in `Jaxpr` expansion |

## See Also

- [Top-level README](../README.md) — Overview, installation extras, and academic references
