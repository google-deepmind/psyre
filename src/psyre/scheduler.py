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
"""Psyre Stage-2 demand-driven coroutine scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Iterable
import traceback
from typing import Any, Generic, TypeVar

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import (
    check_causality,
)
from psyre.builder import (
    build_graph,
)
from psyre.clock import (
    ClockAnalysis,
    check_clocks,
    check_kahn_resets,
)
from psyre.common import (
    Absent,
    Semantics,
    active_scheduler,
    logger,
)
from psyre.graph import (
    Graph,
    Node,
    iter_output_leaves,
    map_output_leaves,
)
from psyre.istructure import (
    IStruct,
    as_istruct,
)
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


def _stream_ready(stream: Stream) -> bool:
  """Returns True if pulling the next element of stream will not block."""
  istruct = stream.istruct
  if istruct.static:
    return True
  return stream.index < len(istruct.seq) and istruct.seq[stream.index].done()


_DEFAULT_HORIZON = 4
_T = TypeVar('_T')


class _SchedulerAttr(Generic[_T]):
  """Descriptor delegating class access to the active Scheduler instance."""

  def __init__(self, attr: str):
    self._attr = f'_{attr}'

  def __get__(self, obj: Scheduler | None, objtype: Any = None) -> _T:
    target = Scheduler.current() if obj is None else obj
    return getattr(target, self._attr)

  def __set__(self, obj: Scheduler, value: _T) -> None:
    setattr(obj, self._attr, value)


class Scheduler:
  """Stage-2 coroutine scheduler executing finalized Psyre dataflow Graphs."""

  deadlocked = _SchedulerAttr[bool]('deadlocked')
  graph = _SchedulerAttr[Graph]('graph')
  semantics = _SchedulerAttr[Semantics]('semantics')

  def __init__(
      self,
      semantics: Semantics = Semantics.SYNC,
      horizon: int = _DEFAULT_HORIZON,
      graph: Graph | None = None,
  ):
    """Initializes a Scheduler instance with its own Stage-2 runtime state."""
    self._semantics: Semantics = semantics
    self._horizon: int = horizon
    self._deadlocked: bool = False
    self._graph: Graph = Graph(semantics=semantics) if graph is None else graph
    self._tasks: set[asyncio.Task[Any]] = set()
    self._blocked: dict[asyncio.Task[Any], Stream] = {}
    self._producer_tasks: dict[IStruct, asyncio.Task[Any]] = {}
    self._task_nodes: dict[asyncio.Task[Any], Node] = {}

  @classmethod
  def current(cls) -> Scheduler:
    """Returns the ambient active Scheduler instance."""
    sched = active_scheduler.get()
    if sched is None:
      sched = cls()
      active_scheduler.set(sched)
    return sched

  # pylint: disable=protected-access
  @classmethod
  async def pull(cls, stream: Stream) -> Any:
    """Pulls the next value from a single input stream."""
    return await cls.current()._pull(stream)

  @classmethod
  async def pull_all(cls, *args: Stream) -> tuple[Any, ...]:
    """Pulls one value from each input stream into a tuple."""
    return await cls.current()._pull_all(*args)

  # pylint: enable=protected-access

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
    """Constructs and checks the Psyre graph (Stage 1), then runs it (Stage 2)."""
    graph = build_graph(
        main_func,
        *args,
        semantics=semantics,
        optimize=optimize,
        extra_roots=streams,
        **kwargs,
    )
    if causality_analysis:
      check_causality(graph)
    clocks = check_clocks(graph) if clock_calculus else None
    return cls.run_graph(graph, horizon=horizon, streams=streams, clocks=clocks)

  @classmethod
  def run_graph(
      cls,
      graph: Graph | None = None,
      *,
      horizon: int = _DEFAULT_HORIZON,
      streams: Iterable[str] | None = None,
      clocks: ClockAnalysis | None = None,
  ) -> Any:
    """Executes Stage 2 scheduling on a fresh Scheduler instance."""
    if graph is None:
      graph = cls.current().graph
    graph.require_finalized()
    semantics = graph.semantics or Semantics.SYNC
    sched = cls(semantics=semantics, horizon=horizon, graph=graph)
    active_scheduler.set(sched)
    return sched._execute(streams=streams, clocks=clocks)

  def is_active_run(self) -> bool:
    """Returns True while this Scheduler is executing Stage-2 tasks."""
    return bool(self._tasks)

  def reset_runtime_state(self) -> None:
    """Clears dynamic stream buffers and resets handle cursors for Stage 2."""
    self._graph.require_finalized()
    for istruct in self._graph.streams:
      istruct.capture = False
      if not istruct.static:
        istruct.seq.clear()
        istruct.keep.clear()
    for node in self._graph.nodes:
      for handle in (node.output, *node.inputs, *node.resets):
        handle.index = 0

  def prepare_capture(
      self, streams: Iterable[str] | None = None
  ) -> dict[str, IStruct] | None:
    """Resets runtime state and enables capture only on selected streams.

    Args:
      streams: Optional iterable of stream names to capture instead of the
        default `output` PyTree. When `None`, captures leaves of `graph.output`.

    Returns:
      A mapping from stream name to `IStruct` when `streams` is provided, or
      `None` when capturing `graph.output`.
    """
    named = (
        None if streams is None else self._graph.resolve_named_streams(streams)
    )
    self.reset_runtime_state()
    if named is not None:
      for istruct in named.values():
        istruct.capture = True
    else:
      for leaf in iter_output_leaves(self._graph.output):
        as_istruct(leaf).capture = True
    return named

  def extract_traces(
      self, named: dict[str, IStruct] | None = None
  ) -> Any:
    """Materializes captured value traces for named streams or graph output.

    Args:
      named: Optional mapping returned by `prepare_capture`. When `None`,
        materializes traces in the shape of `graph.output`.

    Returns:
      Either a `dict[str, list[Any]]` when `named` is provided, or a PyTree
      matching `graph.output` whose stream leaves are replaced by trace lists.
    """
    self._graph.require_finalized()
    if named is not None:
      return {
          name: istruct.trace(self._horizon) for name, istruct in named.items()
      }
    return map_output_leaves(
        lambda h: as_istruct(h).trace(self._horizon), self._graph.output
    )

  def _execute(
      self,
      *,
      streams: Iterable[str] | None = None,
      clocks: ClockAnalysis | None = None,
  ) -> Any:
    """Runs Stage-2 node tasks for `self._graph` and extracts traces."""
    if self._semantics == Semantics.KAHN:
      check_kahn_resets(self._graph, clocks)
    named = self.prepare_capture(streams)

    async def inner() -> None:
      try:
        async with asyncio.TaskGroup() as tg:
          for node in self._graph.nodes:
            task = tg.create_task(self._run_node(node), name=f'{node.name}')
            self._tasks.add(task)
            self._producer_tasks[node.output_stream] = task
            self._task_nodes[task] = node
          self._check_deadlock()
      finally:
        self._tasks.clear()
        self._blocked.clear()
        self._producer_tasks.clear()
        self._task_nodes.clear()

    asyncio.run(inner())
    return self.extract_traces(named)

  async def _pull(self, stream: Stream) -> Any:
    """Pulls the next value from `stream` with deadlock tracking."""
    task = asyncio.current_task()
    assert task is not None
    try:
      if not _stream_ready(stream):
        self._blocked[task] = stream
        self._check_deadlock()
      return await stream.pull()
    finally:
      self._blocked.pop(task, None)

  async def _pull_all(self, *args: Stream) -> tuple[Any, ...]:
    """Pulls one value from each input stream into a tuple."""
    return tuple([await self._pull(stream) for stream in args])

  def _complete_task(self) -> None:
    """Marks the current task completed and checks remaining tasks."""
    task = asyncio.current_task()
    if task is not None and task in self._tasks:
      self._tasks.discard(task)
      self._blocked.pop(task, None)
      self._check_deadlock()

  def _is_waiting_on_completed_producer(self, task: asyncio.Task[Any]) -> bool:
    """Returns True if an arm or post task waits on a completed producer."""
    node = self._task_nodes.get(task)
    if node is None or not (node.arms or node.op == 'post'):
      return False
    stream = self._blocked.get(task)
    if stream is None or self._graph.producer(stream.istruct) is None:
      return False
    prod_task = self._producer_tasks.get(stream.istruct)
    return prod_task is None or prod_task not in self._tasks

  def _check_deadlock(self) -> None:
    """Cancels blocked tasks when no active task is ready to make progress."""
    if (
        self._deadlocked
        or not self._tasks
        or len(self._blocked) < len(self._tasks)
        or any(task.cancelling() for task in self._tasks)
        or any(_stream_ready(stream) for stream in self._blocked.values())
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
          task.cancel()
      return
    self._deadlocked = True
    for task in list(self._tasks):
      if not task.done():
        task.cancel('Deadlock: no ready tasks')

  async def _run_node(self, node: Node) -> None:
    """Executes node.step_func repeatedly up to the horizon (Stage 2)."""
    args = node.inputs
    resets = node.resets
    o_stream = node.output
    name = node.name
    kwargs = dict(node.kwargs)
    is_sync = self._semantics == Semantics.SYNC
    agen: AsyncGenerator[Any, Any] | None = (
        node.step_func(*args, **kwargs) if node.is_stateful else None
    )
    start = True
    logger.debug('loop %s %s, o_stream=%s', name, args, o_stream)
    try:
      if agen is not None:
        await anext(agen)
      for k in range(self._horizon):
        logger.debug('loop %s.%s', name, k)
        try:
          if agen is None:
            await node.step_func(*args, **kwargs)
          else:
            r_vals = await self._pull_all(*resets) if resets else ()
            if any(r is not Absent and r for r in r_vals):
              start = True
            if Absent in r_vals:
              step_start: bool | object = Absent
            else:
              step_start = start
              start = False
            await agen.asend(step_start)
        except asyncio.CancelledError as exc:
          if self._deadlocked:
            traceback.print_exception(exc)
            raise
          curr_task = asyncio.current_task()
          if curr_task is not None:
            curr_task.uncancel()
          break
    finally:
      if agen is not None:
        await agen.aclose()
    if is_sync:
      while o_stream.index < self._horizon:
        o_stream.push(Absent)
    logger.info('%-12s: %s', name, o_stream)
    self._complete_task()
