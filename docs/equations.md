# `psyre.equations` *(Experimental)*

> **Experimental:** `psyre.equations` is an opt-in declarative syntax layer
> whose APIs, semantics, and internals may change without notice.

Lambda-free declarative embedding for recursive Psyre stream equations and
forward `match()` / `automaton()` references, inspired by Lustre and Lucid
Synchrone.

## Overview & Role

In core Psyre, feedback loops and forward references to `match()` or
`automaton()` outputs require explicit zero-argument `lambda:` thunks (for
example, `x = fby(0, lambda: x + 1)`), which adds boilerplate and risks classic
Python late-binding loop-capture bugs.

`psyre.equations` (implemented in `psyre.equations.equations` and re-exported
from `psyre.equations`) eliminates manual `lambda:` thunks by turning Python
`class` bodies into declarative equation scopes:

```python
from psyre import Scheduler, fby
from psyre.equations import psyre_equations

with psyre_equations() as prog:
  class Counter:
    n = fby(0, n + 1)
    __output__ = n

traces = Scheduler.run_graph(prog.graph, horizon=5)
```

Core Psyre integrates with `psyre.equations` solely through the abstract
[`LazyThunk`](common.md#thunks-tags--validation) protocol in `psyre.common`—core modules never
import `psyre.equations`.

## Design & Key Invariants

### Bytecode-Guided Forward-Reference Interception (`EquationScope`)

When a class body runs with metaclass `Equations`, `Equations.__prepare__`
returns an `EquationScope(dict)` namespace pre-populated with any keyword
`inputs` passed in the class header:

- **Single-pass bytecode scan**: On the first lookup of a not-yet-bound name
  (`__missing__`), `EquationScope` inspects the class body frame's bytecode
  (`sys._getframe(2).f_code`) once via `dis.get_instructions` and records all
  non-dunder `STORE_NAME` targets.
- **Exact scoping**: If a missing name is assigned later in the class body,
  `__missing__` caches and returns a forward-reference `Thunk` for it. If the
  name is *not* assigned in the class body, `__missing__` raises `KeyError`,
  letting Python fall through to enclosing, global, and builtin scopes (so
  genuine typos still raise `NameError`).
- **Immutability & first-order guards**: Reassigning a name that was already
  forward-referenced raises `RuntimeError`; binding a forward-referenced name to
  a callable (other than a `LazyThunk`) raises `TypeError` immediately.
- **CAVEAT (CPython bytecode & untaken branches)**: Forward-reference detection
  relies on CPython frame inspection and `STORE_NAME` opcodes. A name assigned
  only inside an untaken branch (e.g. `if False: x = 1`) still produces a
  forward-reference `Thunk` during body execution and fails with `NameError` at
  class finalization.

### `Thunk` (`LazyThunk` Implementation)

`Thunk` subclasses [`psyre.common.LazyThunk`](common.md#thunks-tags--validation) and carries
five invariants:

- **Captured Stage-1 environment**: `__init__` snapshots
  [`CapturedEnv.capture()`](common.md#stage-1-contexts-scopes--control-records) (active `arm_stack`,
  `prefix_stack`, and `reset_stack`). Forcing `__psyre_force__()` restores that
  environment while `is_building()` is `True`, ensuring delayed Stage-1 node
  construction inherits the exact arm, reset, and `@psyre_node` scope where the
  equation was written.
- **Memoization & cycle detection**: Results are memoized using a private
  `_UNRESOLVED` sentinel (preserving `None`/falsy results). Re-entering
  `__psyre_force__()` while already resolving raises `CircularEquationError`
  (a subclass of `RecursionError`). If forcing fails with an internal deferral
  error (for example, an unfinalized enclosing `BlockHandle`), the thunk stays
  unmemoized and can be forced cleanly once the enclosing block exits.
- **Bounded diagnostic descriptions**: `Thunk._desc` is capped at 120 characters
  to prevent exponential string growth in deeply nested deferred expressions.
- **Deferred operators, indexing & NumPy dispatch**: `Thunk` supports deferred
  binary/reflected arithmetic, bitwise, comparison, and unary operators,
  `__getattr__`, `__getitem__` (resolving nested thunks inside tuples and
  `slice` objects), and NumPy NEP-13 (`__array_ufunc__`) / NEP-18
  (`__array_function__`) deferral. Mutating NumPy calls (`.at`, non-`None`
  `out=`, `copyto`, `put`, etc.) are rejected immediately; multi-output ufuncs
  (`divmod`, `modf`, `frexp`) return a tuple of indexed projections sharing one
  memoized `Thunk`.
- **Rejected eager protocols**: `__bool__`, `__iter__`, `__setitem__`,
  `__array__`, and `__call__` raise `TypeError` (`__hash__ = None`).

### Lazy `BlockHandle` Proxying & Nested Deferral

Inside an `EquationScope`, `match()` and `automaton()` `BlockHandle` objects are
wrapped in an internal proxy (`_LazyBlockHandle`). Reading attributes (such as
`m.y` or `last(m.y, 0)`) *before* the `match`/`automaton` block exits returns a
`Thunk` that defers until the block finalizes. When an inner `Equations` class
is nested inside an outer `match`/`automaton` or `Equations` block, references
to unfinalized outer handles remain deferred across inner class finalization,
whereas genuinely unbound names local to the class raise `NameError`.

### Frame-Scoped `builtins.__build_class__` Hook

`with psyre_equations(...) as prog:` avoids writing `metaclass=Equations` on
every class by temporarily wrapping `builtins.__build_class__`:

- **Reference-counted & thread-safe**: Protected by a `threading.Lock` and
  refcount; each wrapper closes over the previous `__build_class__` reference so
  concurrent threads finishing near hook teardown remain safe.
- **Strict injection predicate**: Automatically injects `metaclass=Equations`
  *only* for direct, base-less `class Foo:` statements executing in the exact
  caller frame of `with psyre_equations():` (walking past `contextlib` frames so
  `ExitStack.enter_context()` works) when no explicit `metaclass=` is supplied.
  Classes with base classes, explicit metaclasses (e.g. `metaclass=type`), or
  classes defined inside helper functions are untouched. The frame reference is
  cleared in `finally` to prevent leaks.

### Standalone vs. Nested Sessions & Two-Phase Resolution

- **Standalone mode** (`active_builder.get() is None`): `psyre_equations()`
  creates a [`GraphBuilder`](builder.md) defaulting to `Semantics.SYNC` and
  `optimize=True`, resets Stage-1 stacks, binds `session.output`, finalizes the
  `Graph`, runs [`optimize_graph`](optimization.md) if enabled, and stores the
  result on `session.graph`. Re-entering during an active Stage-2 `Scheduler`
  run raises `RuntimeError`.
- **Nested mode** (inside `build_graph`, `Scheduler.run`, or `@psyre_node`):
  Reuses the active `GraphBuilder` without finalizing or optimizing. Passing a
  conflicting `semantics`, or passing explicit `optimize` / `extra_roots`,
  raises `ValueError`.
- **Two-phase resolution & automatic naming**: On class exit (`Equations.__new__`),
  compound `Thunk` equations resolve before leaf forward-reference aliases and
  container bindings, so producer expressions claim the primary Stage-1 node and
  stream name before aliases attach secondary stream names via
  `Graph.add_stream_name()`. External/producer-less `Stream`s are never renamed
  in place. Container resolution tracks active recursion paths to detect
  thunk-induced cycles (`CircularEquationError`) while permitting shared DAGs.
- **Reserved class-header keywords**: `auto_name` (and Python's `metaclass`)
  cannot be used as equation input names in the class header.

## Public API Reference

All symbols below are exported in `psyre.equations.__all__`.

### Exceptions

#### `CircularEquationError`

```python
class CircularEquationError(RecursionError)
```

Raised when forcing a `Thunk` re-enters itself before producing a value (an
instantaneous unresolved equation cycle or a cyclic container reached through a
`Thunk`). Delaying the feedback path with `fby` or `pre` resolves the cycle.

### Functions

#### `resolve_value`

```python
def resolve_value(val: Any) -> Any
```

Forces `val` via [`force_thunk`](common.md#thunks-tags--validation) if `is_thunk(val)` is
`True`, otherwise returns `val` unchanged.

#### `psyre_equations`

```python
@contextlib.contextmanager
def psyre_equations(
    *,
    semantics: Semantics | None = None,
    optimize: bool | None = None,
    extra_roots: Iterable[Any] | None = None,
    name: str = '',
) -> Iterator[PsyreEquations]
```

Context manager hosting lambda-free `Equations` blocks in Stage 1.

- **Parameters**:
  - `semantics`: Execution semantics (`Semantics.SYNC` default in standalone
    mode; must match or be `None` when nested inside an active `GraphBuilder`).
  - `optimize`: Whether to run `optimize_graph` on exit (`True` default in
    standalone mode; must be `None` when nested).
  - `extra_roots`: Additional root streams/names preserved during optimization
    (standalone mode only; must be `None` when nested).
  - `name`: Optional `@psyre_node`-style hierarchy prefix pushed around the
    block.
- **Yields**: A [`PsyreEquations`](#psyreequations) session handle whose
  `.output`, `.graph`, and `.last` fields update as equation classes complete.
- **Raises**: `TypeError` on invalid argument types; `ValueError` on conflicting
  nested options; `RuntimeError` if entered during an active Stage-2 `Scheduler`
  run.

### Classes

#### `Thunk`

```python
class Thunk(LazyThunk):
    def __init__(
        self,
        fn: Callable[[], Any],
        *,
        desc: str = '<thunk>',
        env: CapturedEnv | None = None,
    ) -> None
```

Memoized, cycle-checked, Stage-1-environment-capturing `LazyThunk` proxy.

- **`__psyre_force__() -> Any`**: Evaluates `fn()` inside `env` (when building),
  forces one nested `LazyThunk` layer if returned, rejects non-thunk callables
  with `TypeError`, caches the result, and raises `CircularEquationError` on
  re-entry.
- **Operators & protocols**: Supports deferred arithmetic, bitwise, comparison,
  unary ops, attribute access (`__getattr__`), indexing (`__getitem__`), and
  NumPy `__array_ufunc__` / `__array_function__`. Unhashable (`__hash__ = None`);
  rejects `__bool__`, `__iter__`, `__setitem__`, `__array__`, and `__call__`.

#### `EquationScope`

```python
class EquationScope(dict):
    def __init__(
        self,
        initial: Mapping[str, Any] | None = None,
        *,
        auto_name: bool = True,
    ) -> None
```

Custom class-body mapping returned by `Equations.__prepare__` that intercepts
forward-referenced local assignments as `Thunk`s and wraps `BlockHandle`s.

- **`name_stream(key: str, stream: Stream) -> None`**: Auto-names an unnamed or
  default-op-named Stage-1 stream produced in the current prefix scope, or
  records a qualified alias via `Graph.add_stream_name()`.

#### `PsyreEquations`

```python
@dataclasses.dataclass
class PsyreEquations:
    output: Any = None
    graph: Graph | None = None
    last: type[Any] | None = None
```

Session handle yielded by `psyre_equations()`.

- `output`: Resolved `__output__` of the most recent `Equations` class defining
  `__output__`.
- `graph`: Finalized (and optionally optimized) `Graph` after exiting standalone
  mode, or the live `GraphBuilder.graph` while nested.
- `last`: Most recently finalized `Equations` class object.

#### `Equations`

```python
class Equations(type):
    @classmethod
    def __prepare__(
        mcs, name: str, bases: tuple[type[Any], ...], /,
        *, auto_name: bool = True, **inputs: Any,
    ) -> EquationScope

    def __new__(
        mcs, name: str, bases: tuple[type[Any], ...], namespace: EquationScope, /,
        *, auto_name: bool = True, **inputs: Any,
    ) -> Any
```

Metaclass turning a Python class body into a declarative Psyre equation block.
Can be used explicitly (`class System(metaclass=Equations, x=in_stream):`) or
implicitly via `with psyre_equations():`. Resolves deferred `Thunk` equations in
two phases, applies automatic naming when `auto_name=True`, resolves
`__output__`, and updates any active `PsyreEquations` session.

## See Also

- [`psyre.common`](common.md) — `LazyThunk`, `CapturedEnv`, `BlockHandle`
- [`psyre.builder`](builder.md) — Stage-1 `GraphBuilder` and `build_graph`
- [`psyre.match`](match.md) and [`psyre.automaton`](automaton.md) — Pattern
  matching and hierarchical state machines with `last()`
