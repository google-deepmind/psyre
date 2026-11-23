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
"""Psyre Stage-2 demand-driven coroutine scheduler.

Executes finalized dataflow graphs under both `Semantics.SYNC` (synchronous
Lustre-style cycle stepping; Halbwachs et al., *Proc. IEEE* 1991) and
`Semantics.KAHN` (demand-driven Kahn Process Network coroutines; Kahn, *IFIP*
1974; Kahn & MacQueen, *IFIP* 1977).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Iterable
import inspect
import logging
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import (
    causal_node_order,
    check_causality,
)
from psyre.builder import (
    build_graph,
    lower_resets,
    requires_clock_analysis,
)
from psyre.clock import (
    ClockAnalysis,
    ClockError,
    check_clocks,
)
from psyre.common import (
    Absent,
    Semantics,
    UNREADY,
    active_builder,
    active_scheduler,
    is_builtin_primitive_step,
    logger,
    validate_non_negative_int,
)
from psyre.graph import (
    Graph,
    Node,
    map_output_leaves,
)
from psyre.stream import Stream, StreamPort

# pylint: enable=g-importing-member,g-multiple-import


def _stream_ready(port: StreamPort) -> bool:
  """Returns True if pulling the next element of port will not block."""
  stream = port.stream
  return stream.static or port.index < stream.pushed_count


def _is_standard_reader(node: Node) -> bool:
  """Returns True when `node` reads input ports with known cursor rules."""
  step_func = node.step_func
  return (
      is_builtin_primitive_step(node.op, step_func)
      or hasattr(step_func, '_psyre_inst_func')
      or bool(getattr(step_func, '_psyre_is_sequ', False))
  )


def _normalize_streams_arg(
    streams: Iterable[str] | None,
) -> tuple[str, ...] | None:
  """Validates `streams` (rejecting bare `str`) and copies it to a tuple."""
  if streams is None:
    return None
  if isinstance(streams, str):
    raise TypeError(
        'streams must be an iterable of stream names, not a bare str.'
    )
  return tuple(streams)


_DEFAULT_HORIZON = 4


class _SchedulerAttr[T]:
  """Descriptor delegating class access to the active Scheduler instance."""

  _attr: str

  def __set_name__(self, owner: type[Any], name: str) -> None:
    del owner
    self._attr = f'_{name}'

  def __get__(self, obj: Scheduler | None, objtype: Any = None) -> T:
    target = Scheduler.current() if obj is None else obj
    return getattr(target, self._attr)

  def __set__(self, obj: Scheduler, value: T) -> None:
    setattr(obj, self._attr, value)


class Scheduler:
  """Stage-2 coroutine scheduler executing finalized Psyre dataflow Graphs."""

  deadlocked = _SchedulerAttr[bool]()
  graph = _SchedulerAttr[Graph]()
  semantics = _SchedulerAttr[Semantics]()

  def __init__(
      self,
      semantics: Semantics = Semantics.SYNC,
      horizon: int = _DEFAULT_HORIZON,
      graph: Graph | None = None,
  ):
    """Initializes a Scheduler instance with its own Stage-2 runtime state."""
    validate_non_negative_int(horizon, 'horizon')
    self._semantics: Semantics = semantics
    self._horizon: int = horizon
    self._deadlocked: bool = False
    self._failed: bool = False
    self._running: bool = False
    self._graph: Graph = Graph(semantics=semantics) if graph is None else graph
    self._tasks: set[asyncio.Task[Any]] = set()
    self._blocked: dict[asyncio.Task[Any], StreamPort] = {}
    self._producer_tasks: dict[int, asyncio.Task[Any]] = {}
    self._task_nodes: dict[asyncio.Task[Any], Node] = {}
    self._eof_cancelled: set[asyncio.Task[Any]] = set()

  @classmethod
  def current(cls) -> Scheduler:
    """Returns the ambient active Scheduler instance."""
    sched = active_scheduler.get()
    if sched is None:
      sched = cls()
      active_scheduler.set(sched)
    return sched

  @classmethod
  def _require_not_reentrant(cls) -> None:
    """Rejects nested Stage-1 or Stage-2 Scheduler invocations."""
    if active_builder.get() is not None:
      raise RuntimeError(
          'Re-entrant Scheduler execution inside an active Stage-1 graph '
          'build is not allowed.'
      )
    curr = active_scheduler.get()
    if curr is not None and curr.is_active_run():
      raise RuntimeError(
          'Re-entrant Scheduler execution inside an active Stage-2 run is '
          'not allowed.'
      )

  @classmethod
  def run(
      cls,
      main_func: Callable[..., Any],
      *args: Any,
      horizon: int = _DEFAULT_HORIZON,
      semantics: Semantics = Semantics.SYNC,
      causality_analysis: bool = True,
      clock_calculus: bool = True,
      streams: Iterable[str] | None = None,
      optimize: bool = True,
      **kwargs: Any,
  ) -> Any:
    """Constructs and checks the Psyre graph (Stage 1), then runs it (Stage 2).

    Static clock analysis (`check_clocks`) is mandatory under `Semantics.SYNC`
    (`clock_calculus=False` raises `ValueError`). Under `Semantics.KAHN`,
    `clock_calculus=False` may skip static clock analysis when the graph does
    not require `lower_resets()`.
    """
    cls._require_not_reentrant()
    validate_non_negative_int(horizon, 'horizon')
    if not clock_calculus and semantics == Semantics.SYNC:
      raise ValueError(
          'Clock analysis is mandatory under Semantics.SYNC; '
          'clock_calculus=False is only supported under Semantics.KAHN.'
      )
    stream_names = _normalize_streams_arg(streams)
    graph = build_graph(
        main_func,
        *args,
        semantics=semantics,
        optimize=optimize,
        extra_roots=stream_names,
        **kwargs,
    )
    if causality_analysis:
      check_causality(graph)
    clocks = check_clocks(graph) if clock_calculus else None
    return cls.run_graph(
        graph, horizon=horizon, streams=stream_names, clocks=clocks
    )

  @classmethod
  def run_graph(
      cls,
      graph: Graph | None = None,
      *,
      horizon: int = _DEFAULT_HORIZON,
      streams: Iterable[str] | None = None,
      clocks: ClockAnalysis | None = None,
  ) -> Any:
    """Executes Stage 2 scheduling on a fresh Scheduler instance.

    Under `Semantics.SYNC`, static clock analysis (`check_clocks`) and
    `lower_resets()` always run when `clocks` is omitted. Under
    `Semantics.KAHN`, omitting `clocks` skips static clock checking unless the
    graph requires `lower_resets()`.
    """
    cls._require_not_reentrant()
    validate_non_negative_int(horizon, 'horizon')
    stream_names = _normalize_streams_arg(streams)
    if graph is None:
      graph = cls.current().graph
    graph.require_finalized()
    semantics = graph.semantics or Semantics.SYNC
    sched = cls(semantics=semantics, horizon=horizon, graph=graph)
    active_scheduler.set(sched)
    return sched._execute(streams=stream_names, clocks=clocks)

  def is_active_run(self) -> bool:
    """Returns True for the full `_execute` lifecycle (guards re-entrancy).

    Unlike `is_executing_tasks()`, stays `True` during graph lowering and final
    `extract_traces()` after `asyncio.TaskGroup` has emptied `_tasks`.
    """
    return self._running or bool(self._tasks)

  def is_executing_tasks(self) -> bool:
    """Returns True only while Stage-2 node tasks are actively running.

    Used by `extract_traces` and `Stream.trace` (`pad_bottom=True`) to allow
    post-run padding right after `asyncio.TaskGroup` exits, while `_running`
    is still `True`.
    """
    return bool(self._tasks)

  def unblock_task(self, task: asyncio.Task[Any], stream: Stream) -> None:
    """Removes a woken consumer task if it is blocked on `stream`."""
    blocked_port = self._blocked.get(task)
    if blocked_port is not None and blocked_port.stream is stream:
      self._blocked.pop(task, None)

  def reset_runtime_state(self) -> None:
    """Clears dynamic stream buffers for a fresh Stage-2 execution."""
    self._graph.require_finalized()
    for stream in self._graph.streams:
      stream.reset()

  def extract_traces(
      self,
      named: dict[str, Stream] | None = None,
      *,
      pad_bottom: bool = False,
  ) -> Any:
    """Materializes stream traces for named streams or graph output.

    Args:
      named: Optional mapping from stream name to `Stream`. When `None`,
        materializes traces in the shape of `graph.output`.
      pad_bottom: When True, pads dynamic producer-backed streams shorter than
        `self._horizon` with trailing `Bottom` entries.

    Returns:
      Either a `dict[str, list[Any]]` when `named` is provided, or a PyTree
      matching `graph.output` whose stream leaves are replaced by trace lists.

    Raises:
      RuntimeError: If `pad_bottom=True` is called while a Stage-1 build or
        Stage-2 task loop is still active, or if the graph is not finalized.
    """
    self._graph.require_finalized()
    if pad_bottom and (
        active_builder.get() is not None or self.is_executing_tasks()
    ):
      raise RuntimeError(
          'extract_traces(pad_bottom=True) can only be used after a Scheduler '
          'run completes.'
      )

    def trace_of(stream: Stream) -> list[Any]:
      resolved = self._graph.resolve_stream(stream)
      return resolved.trace(
          self._horizon,
          pad_bottom=(
              pad_bottom and self._graph.producer(resolved) is not None
          ),
      )

    if named is not None:
      return {name: trace_of(stream) for name, stream in named.items()}
    return map_output_leaves(trace_of, self._graph.output)

  def _execute(
      self,
      *,
      streams: tuple[str, ...] | None = None,
      clocks: ClockAnalysis | None = None,
  ) -> Any:
    """Runs Stage-2 node tasks for `self._graph` and extracts traces."""
    named = (
        None
        if streams is None
        else self._graph.resolve_named_streams(streams)
    )
    if clocks is not None:
      exec_graph = lower_resets(self._graph, clocks)
    elif self._semantics == Semantics.SYNC:
      exec_graph = lower_resets(self._graph, check_clocks(self._graph))
    else:
      if requires_clock_analysis(self._graph):
        raise ClockError(
            'KAHN execution with subclocked resets, static streams, or '
            'current() inputs requires clock analysis '
            '(pass clock_calculus=True or clocks=...).'
        )
      exec_graph = self._graph

    ordered_nodes = causal_node_order(exec_graph)
    if named is not None:
      retained_ids = {
          self._graph.resolve_stream(s).id for s in named.values()
      }
    else:
      retained_ids = {
          self._graph.resolve_stream(s).id
          for s in self._graph.output_streams()
      }
    node_ports: dict[
        int,
        tuple[
            StreamPort,
            tuple[StreamPort, ...],
            tuple[StreamPort, ...],
            list[tuple[Stream, Any]],
        ],
    ] = {}
    readers_by_stream: dict[int, list[Any]] = {}
    for node in ordered_nodes:
      o_port = StreamPort(node.output)
      in_ports = tuple(StreamPort(s) for s in node.inputs)
      rst_ports = tuple(StreamPort(s) for s in node.resets)
      is_std = _is_standard_reader(node)
      is_post = node.is_primitive('post')
      cleanups: list[tuple[Stream, Any]] = []
      for p_idx, in_port in enumerate(in_ports):
        s = in_port.stream
        if s.static:
          continue
        if not is_std:
          retained_ids.add(s.id)
          continue
        entry: Any = (
            (in_port, o_port)
            if (is_post and p_idx == 0 and rst_ports)
            else in_port
        )
        readers_by_stream.setdefault(s.id, []).append(entry)
        cleanups.append((s, entry))
      for r_port in rst_ports:
        rs = r_port.stream
        if rs.static:
          continue
        readers_by_stream.setdefault(rs.id, []).append(r_port)
        cleanups.append((rs, r_port))
      node_ports[node.id] = (o_port, in_ports, rst_ports, cleanups)

    async def inner() -> None:
      try:
        async with asyncio.TaskGroup() as tg:
          for node in ordered_nodes:
            o_port, in_ports, rst_ports, cleanups = node_ports[node.id]
            task = tg.create_task(
                self._run_node(node, o_port, in_ports, rst_ports, cleanups),
                name=f'{node.name}',
            )
            self._tasks.add(task)
            self._producer_tasks[node.output.id] = task
            self._task_nodes[task] = node
      finally:
        self._tasks.clear()
        self._blocked.clear()
        self._producer_tasks.clear()
        self._task_nodes.clear()
        self._eof_cancelled.clear()

    def reset_streams() -> None:
      self.reset_runtime_state()
      if exec_graph is not self._graph:
        for stream in exec_graph.streams:
          stream.reset()

    self._failed = False
    self._running = True
    reset_streams()
    for node in ordered_nodes:
      out_s = node.output
      if not out_s.static and out_s.id not in retained_ids:
        out_s.configure_eviction(readers_by_stream.get(out_s.id, []))
    try:
      asyncio.run(inner())
      pad_bottom = self._semantics == Semantics.SYNC and not self._deadlocked
      return self.extract_traces(named, pad_bottom=pad_bottom)
    finally:
      self._running = False
      reset_streams()

  async def pull(self, port: StreamPort) -> Any:
    """Pulls the next value from `port` with deadlock tracking."""
    ready = port.try_pull_ready()
    if ready is not UNREADY:
      return ready
    task = asyncio.current_task()
    assert task is not None
    try:
      self._blocked[task] = port
      self._check_deadlock()
      return await port.pull()
    finally:
      self._blocked.pop(task, None)

  def _complete_task(self) -> None:
    """Marks the current task completed and checks remaining tasks."""
    task = asyncio.current_task()
    if task is not None and task in self._tasks:
      self._tasks.discard(task)
      self._blocked.pop(task, None)
      self._check_deadlock()

  def _handle_eof_cancel(self) -> bool:
    """Returns True if the current task was cleanly EOF-cancelled."""
    curr_task = asyncio.current_task()
    if (
        self._deadlocked
        or self._failed
        or curr_task is None
        or curr_task not in self._eof_cancelled
    ):
      return False
    self._eof_cancelled.discard(curr_task)
    # Under Python 3.12+ `asyncio.TaskGroup`, decrement our single EOF-cancel
    # count via `uncancel()`. A non-zero remainder means `TaskGroup` also
    # cancelled this task (e.g. due to a concurrent sibling failure), so we
    # must re-raise `CancelledError` rather than swallow it.
    return curr_task.uncancel() == 0

  def _is_waiting_on_completed_producer(self, task: asyncio.Task[Any]) -> bool:
    """Returns True if a task waits on a completed producer (EOF)."""
    node = self._task_nodes.get(task)
    if node is None:
      return False
    port = self._blocked.get(task)
    if port is None:
      return False
    prod_task = self._producer_tasks.get(port.stream.id)
    if prod_task is None or prod_task in self._tasks:
      return False
    if self._semantics == Semantics.SYNC:
      return True
    # Under KAHN, separate benign horizon-EOF truncation from local starvation
    # (e.g. `when(False, counter()) + 1`, which sits un-reset at `index == 0`):
    # a blocked consumer is treated as clean EOF iff it is scoped by an arm,
    # resettable, a lookahead `post`, or has already consumed >= 1 token.
    return bool(
        node.arms
        or node.resets
        or node.is_primitive('post')
        or port.index > 0
    )

  def _check_deadlock(self) -> None:
    """Cancels blocked tasks when no active task is ready to make progress."""
    if (
        self._deadlocked
        or self._failed
        or not self._tasks
        or len(self._blocked) < len(self._tasks)
        # Wait while an already-issued EOF cancellation unwinds asynchronously.
        or any(task.cancelling() for task in self._tasks)
        or any(_stream_ready(port) for port in self._blocked.values())
    ):
      return
    eof_tasks = [
        task
        for task in self._tasks
        if self._is_waiting_on_completed_producer(task)
    ]
    if eof_tasks:
      for task in eof_tasks:
        if not task.done():
          self._eof_cancelled.add(task)
          task.cancel()
      return
    self._deadlocked = True
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug(
          'Deadlock detected: %d blocked task(s) with no ready producer: %s',
          len(self._tasks),
          ', '.join(sorted(t.get_name() for t in self._tasks)),
      )
    for task in list(self._tasks):
      if not task.done():
        task.cancel('Deadlock: no ready tasks')

  async def _run_node(
      self,
      node: Node,
      o_port: StreamPort,
      in_ports: tuple[StreamPort, ...],
      rst_ports: tuple[StreamPort, ...],
      reader_cleanups: list[tuple[Stream, Any]],
  ) -> None:
    """Executes node.step_func repeatedly up to the horizon (Stage 2)."""
    step_func = node.step_func
    kwargs = node.kwargs
    name = node.name
    agen: AsyncGenerator[Any, Any] | None = None
    start = True
    debug = logger.isEnabledFor(logging.DEBUG)
    if debug:
      logger.debug(
          'loop %s (%d inputs), o_stream=%s',
          name,
          len(in_ports),
          node.output._debug_tag(),  # pylint: disable=protected-access
      )
    try:
      if node.is_stateful:
        agen = (
            step_func(o_port, *in_ports, **kwargs)
            if kwargs
            else step_func(o_port, *in_ports)
        )
        await anext(agen)
      for k in range(self._horizon):
        if debug:
          logger.debug('loop %s.%s', name, k)
        try:
          if agen is None:
            if kwargs:
              await step_func(o_port, *in_ports, **kwargs)
            else:
              await step_func(o_port, *in_ports)
          else:
            if rst_ports:
              # Pull every enclosing reset port to keep cursors in sync. All
              # non-1-periodic resets are already lowered onto `node_clk` by
              # `lower_resets()`; if any dynamic reset yields `Absent` on an
              # inactive tick, keep `start` latched (preserving tick-0 `start`
              # or an unlowered constant `True` sibling reset) until active.
              has_absent = False
              for r_port in rst_ports:
                r = r_port.try_pull_ready()
                if r is UNREADY:
                  r = await self.pull(r_port)
                if r is Absent:
                  has_absent = True
                elif r:
                  start = True
              if has_absent:
                step_start: bool | object = Absent
              else:
                step_start = start
                start = False
            else:
              step_start = start
              start = False
            await agen.asend(step_start)
        except asyncio.CancelledError:
          if not self._handle_eof_cancel():
            raise
          break
    except BaseException as exc:
      if not isinstance(exc, asyncio.CancelledError):
        self._failed = True
      raise
    finally:
      try:
        if inspect.isasyncgen(agen):
          await agen.aclose()
      except BaseException as exc:
        if not isinstance(exc, asyncio.CancelledError):
          self._failed = True
        raise
      finally:
        for stream, entry in reader_cleanups:
          stream.remove_reader(entry)
        self._complete_task()
