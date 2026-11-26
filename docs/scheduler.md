# `psyre.scheduler`

Stage-2 demand-driven `asyncio` coroutine scheduler executing finalized Psyre
graphs under both `Semantics.SYNC` and `Semantics.KAHN` with sliding-window
eviction, deadlock/EOF distinction, and post-run trace extraction.

## Overview & Role

`psyre.scheduler` owns Stage 2 of Psyre's two-stage architecture. Given a
finalized [`Graph`](graph.md) (either passed to `Scheduler.run_graph` or built
end-to-end by `Scheduler.run`), `Scheduler` verifies and lowers the graph,
spawns one `asyncio.Task` per node inside an `asyncio.TaskGroup`, coordinates
blocking pulls and deadlock/EOF termination up to `horizon` ticks, reclaims
consumed intermediate stream buffers online, extracts requested traces, and
cleans up dynamic runtime buffers in a `finally:` block.

## Design & Key Invariants

### Mandatory `SYNC` Clock Analysis vs. Optional `KAHN`

- **`Semantics.SYNC` (default)**: Static clock analysis (`check_clocks`) and
  [`lower_resets`](builder.md#lower_resets) are **mandatory**. Passing
  `clock_calculus=False` to `Scheduler.run(..., semantics=Semantics.SYNC)`
  raises `ValueError`, and calling `Scheduler.run_graph(graph, clocks=None)`
  automatically invokes `check_clocks(graph)` and `lower_resets(graph, clocks)`.
- **`Semantics.KAHN`**: `clock_calculus=False` (or `clocks=None`) is permitted
  when [`requires_clock_analysis(graph)`](builder.md#conservative-kahn-lowering-gate-requires_clock_analysis)
  is `False`; otherwise `run_graph` raises `ClockError` requesting clock
  analysis so subclocked resets, static ports, or `current()` inputs are lowered
  soundly.

### Causal Task Spawning & Synchronous Fast Paths

- Node tasks are spawned in [`causal_node_order(exec_graph)`](analysis.md#causal_node_order)
  so producers run ahead of consumers on every tick, maximizing synchronous
  `StreamPort.try_pull_ready()` hits without suspending onto the `asyncio`
  event loop.
- `_run_node` drives stateless coroutines and stateful async generators up to
  `self._horizon` iterations:
  - Stateful nodes pull all enclosing `rst_ports` synchronously when ready,
    latching `start = True` across any `Absent` tick and forwarding
    `step_start = Absent` on inactive ticks so internal state stays untouched.
  - Async generators are always closed via `await agen.aclose()` in `finally:`.

### `O(1)` Deadlock Detection & Clean EOF Cancellation

- **Port-matched waiter eviction**: Whenever a task blocks in `Scheduler.pull`,
  it registers `self._blocked[task] = port`. When a producer writes to that
  stream, `Stream.push` calls `Scheduler.unblock_task(task, stream)` (checking
  `blocked_port.stream is stream`) before the next event-loop turn, avoiding
  quadratic deadlock scans.
- **EOF vs. true deadlock (`_check_deadlock`, `_handle_eof_cancel`)**:
  When every active task is blocked and no blocked port is ready (and no prior
  EOF cancellation is mid-unwind):
  - Tasks blocked on a **completed producer** are classified as clean horizon
    EOF (always under `SYNC`; under `KAHN`, when the consumer is arm-scoped,
    resettable, a `post` node, or has already consumed `>= 1` token, separating
    benign finite-horizon truncation from `index == 0` starvation like
    `when(False, counter()) + 1`). Those tasks receive a targeted EOF
    cancellation; `_handle_eof_cancel()` uses Python 3.12+ `task.uncancel() == 0`
    so concurrent sibling-exception cancellations from `asyncio.TaskGroup` are
    never swallowed.
  - If *no* task is waiting on a completed producer, `self._deadlocked = True`
    is set, a compact `DEBUG` summary is logged, and remaining tasks are
    cancelled.
- **Post-horizon `Bottom` padding**: After `TaskGroup` exits, under
  `Semantics.SYNC` when `not self._deadlocked`, `extract_traces` passes
  `pad_bottom=True` so producer-backed streams truncated early by `post`
  lookahead are padded with trailing [`Bottom`](common.md#sentinels-absent-bottom-none-unset-and-unready)
  entries up to `horizon` without mutating `Stream.seq`.

### Tier-2 Sliding-Window Slot Eviction

Before launching tasks, `Scheduler._execute` identifies all probed streams
(`retained_ids`: either `streams` or `graph.output_streams()`, plus any input of
a non-standard custom `@psyre_loop` node). Every other intermediate dynamic
stream receives `out_s.configure_eviction(readers)`:

- Monotonic readers register their `StreamPort`; resettable `post` nodes
  register `(in_port, o_port)` tuples so `min(in_port.index, o_port.index)`
  protects reset rewinds (`i_stream.index = o_stream.index`).
- Completed tasks unregister their reader entries in `finally:` via
  `stream.remove_reader(entry)`, allowing upstream buffers to reclaim slots as
  soon as downstream consumers finish.

### Re-Entrancy & Phase Guards

- Both `Scheduler.run` and `Scheduler.run_graph` reject re-entrant calls during
  an active Stage-1 `GraphBuilder` build or an active Stage-2 run
  (`RuntimeError`).
- `is_active_run()` remains `True` across the entire `_execute()` lifecycle
  (lowering, task loop, and `extract_traces()`), whereas `is_executing_tasks()`
  is `True` only while node tasks are live—allowing `extract_traces(pad_bottom=True)`
  right after `TaskGroup` exits while rejecting calls from inside running node
  tasks.
- `streams` rejects bare `str` (`TypeError`) and materializes iterables once to
  a `tuple` so single-use iterators are not exhausted before `run_graph`.

## Public API Reference

`Scheduler` is re-exported from top-level `psyre`.

### `Scheduler`

```python
class Scheduler:
    deadlocked: bool      # descriptor delegating to active instance on class access
    graph: Graph          # descriptor delegating to active instance on class access
    semantics: Semantics  # descriptor delegating to active instance on class access

    def __init__(
        self,
        semantics: Semantics = Semantics.SYNC,
        horizon: int = 4,
        graph: Graph | None = None,
    ) -> None
```

- **Class methods**:
  - `Scheduler.current() -> Scheduler`: Returns (or lazily initializes) the
    ambient `active_scheduler` `ContextVar` instance.
  - `Scheduler.run(main_func, *args, horizon=4, semantics=Semantics.SYNC, causality_analysis=True, clock_calculus=True, streams=None, optimize=True, **kwargs) -> Any`:
    End-to-end Stage-1 (`build_graph`, optional `check_causality`,
    `check_clocks`) + Stage-2 (`run_graph`) execution returning materialized
    traces matching `graph.output` or `dict[str, list[Any]]` when `streams` is
    specified.
  - `Scheduler.run_graph(graph=None, *, horizon=4, streams=None, clocks=None) -> Any`:
    Executes Stage 2 on a finalized `Graph` (defaulting to `cls.current().graph`)
    using a fresh `Scheduler` instance.
- **Instance methods**:
  - `is_active_run() -> bool`, `is_executing_tasks() -> bool`
  - `unblock_task(task: asyncio.Task[Any], stream: Stream) -> None`
  - `reset_runtime_state() -> None`: Clears dynamic stream buffers on
    `self._graph`.
  - `extract_traces(named: dict[str, Stream] | None = None, *, pad_bottom: bool = False) -> Any`
  - `async def pull(port: StreamPort) -> Any`: Deadlock-tracked pull used by
    [`common.pull`](common.md#runtime-execution--decorators).

## See Also

- [`psyre.builder`](builder.md) — `build_graph`, `lower_resets`, `requires_clock_analysis`
- [`psyre.stream`](stream.md#stage-2-storage-waiters--sliding-window-eviction) — `StreamPort` and eviction mechanics
- [`psyre.trace`](trace.md) — Post-run `is_present`, `is_absent`, `is_bottom`, `present_values`
