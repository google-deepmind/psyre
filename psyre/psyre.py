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
"""Psyre support types, classes, functions, and decorators."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Iterable, Sequence
import functools
import traceback
from typing import Any, Generic, TypeVar

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import (
    check_causality,
)
from psyre.clock import (
    ClockAnalysis,
    check_clocks,
    check_kahn_resets,
)
from psyre.common import (
    Absent,
    CapturedEnv,
    STATEFUL_PRIMITIVE_OPS,
    Semantics,
    active_scheduler,
    callable_name,
    is_thunk,
    logger,
    push_prefix,
    push_reset,
    register_inst_func,
    reset_stack,
    restore_resets,
)
from psyre.graph import (
    Graph,
    Node,
)
from psyre.istructure import (
    IStruct,
    IStructHandle,
)
from psyre.optimization import (
    eliminate_dead_code,
)

# pylint: enable=g-importing-member,g-multiple-import

reset = push_reset


def _stream_ready(stream: Stream) -> bool:
  """Returns True if pulling the next element of stream will not block."""
  istruct = stream.istruct
  if istruct.static:
    return True
  return stream.index < len(istruct.seq) and istruct.seq[stream.index].done()


def resolve_handle(stream_or_thunk: Any) -> Stream:
  """Resolves any zero-arg thunks and returns a fresh Stream handle."""
  while is_thunk(stream_or_thunk):
    stream_or_thunk = stream_or_thunk()
  return Stream(stream_or_thunk)


async def _comb(
    inst_func: Callable[..., Any],
    o_stream: Stream,
    *args: Stream,
    **kwargs: Any,
) -> None:
  """Executes a combinational (stateless) step from a pure function."""
  logger.debug('comb %s, %s, %s', callable_name(inst_func), args, kwargs)
  i_vals = await Scheduler.pull_all(*args) if args else ()
  o_val = Absent if Absent in i_vals else inst_func(*i_vals, **kwargs)
  o_stream.push(o_val)


async def _sequ(
    inst_func: Callable[..., Any],
    o_stream: Stream,
    *args: Stream,
    **kwargs: Any,
) -> AsyncGenerator[None, Any]:
  """Executes a sequential (stateful) generator function across ticks."""
  gen = None
  while True:
    start = yield
    logger.debug('sequ %s, %s, %s', callable_name(inst_func), args, kwargs)
    i_vals = await Scheduler.pull_all(*args) if args else ()
    if start is Absent:
      o_stream.push(Absent)
      continue
    if gen is None or (start and (not i_vals or i_vals[0] is not Absent)):
      gen = inst_func(*i_vals, **kwargs)
      o_val = next(gen)
    else:
      o_val = gen.send(i_vals[0] if len(i_vals) == 1 else i_vals)
    o_stream.push(o_val)


_STATEFUL_STEP_FUNCS: set[Callable[..., Any]] = set()
PrepareFn = Callable[..., tuple[tuple[Any, ...], dict[str, Any]]]


def psyre_loop(
    step_func: Callable[..., Any] | None = None,
    *,
    prepare: PrepareFn | None = None,
) -> Any:
  """@psyre_loop decorator for nodes."""

  def decorate(fn: Callable[..., Any]) -> Callable[..., Stream]:
    def wrapper(*args: Any, **kwargs: Any) -> Stream:
      return Scheduler.current()._loop(fn, *args, _prepare=prepare, **kwargs)  # pylint: disable=protected-access

    return wrapper

  return decorate if step_func is None else decorate(step_func)


def psyre_comb(inst_func: Callable[..., Any]) -> Callable[..., Stream]:
  """@psyre_comb decorator to define a node from a stateless function."""

  def step_func(o_stream: Stream, *args: Stream, **kwargs: Any) -> Any:
    return _comb(inst_func, o_stream, *args, **kwargs)

  step_func.__name__ = callable_name(inst_func)
  wrapper = psyre_loop(step_func)
  register_inst_func(wrapper, inst_func)
  return wrapper


def psyre_sequ(
    inst_func: Callable[..., Any] | None = None,
    *,
    prepare: PrepareFn | None = None,
) -> Any:
  """@psyre_sequ decorator to define a node from a stateful function."""

  def decorate(fn: Callable[..., Any]) -> Callable[..., Stream]:
    def step_func(o_stream: Stream, *args: Stream, **kwargs: Any) -> Any:
      return _sequ(fn, o_stream, *args, **kwargs)

    step_func.__name__ = callable_name(fn)
    _STATEFUL_STEP_FUNCS.add(step_func)
    return psyre_loop(step_func, prepare=prepare)

  return decorate if inst_func is None else decorate(inst_func)


def psyre_node(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """@psyre_node(name) decorator wrapping a Stage-1 function in a hierarchy."""
  if not isinstance(name, str) or not name:
    raise TypeError('psyre_node(name) requires a non-empty str argument.')

  def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      if not Scheduler.is_building():
        raise RuntimeError(
            'Psyre nodes can only be instantiated during Stage 1.'
        )
      scope_name = kwargs.pop('name', name)
      if not isinstance(scope_name, str) or not scope_name:
        raise TypeError('psyre_node name must be a non-empty str.')
      with push_prefix(scope_name):
        return fn(*args, **kwargs)

    return wrapper

  return decorator


class Stream(IStructHandle):
  """Stream handle with pointwise arithmetic, bitwise, and comparison operators."""

  def __init__(self, *args, name: str = '', **kwargs):
    """Initializes a Stream handle and registers it if Stage 1 is active."""
    super().__init__(*args, name=name, **kwargs)
    Scheduler.register_handle(self)

  def __bool__(self) -> bool:
    raise TypeError(
        'Truth value of a Stream is ambiguous; use Psyre combinators '
        '(e.g. ifthenelse, merge, when, Stream.max, or Stream.min).'
    )

  @psyre_comb
  def __add__(self, stream):
    return self + stream

  @psyre_comb
  def __radd__(self, stream):
    return stream + self

  @psyre_comb
  def __sub__(self, stream):
    return self - stream

  @psyre_comb
  def __rsub__(self, stream):
    return stream - self

  @psyre_comb
  def __mul__(self, stream):
    return self * stream

  @psyre_comb
  def __rmul__(self, stream):
    return stream * self

  @psyre_comb
  def __matmul__(self, stream):
    return self @ stream

  @psyre_comb
  def __rmatmul__(self, stream):
    return stream @ self

  @psyre_comb
  def __truediv__(self, stream):
    return self / stream

  @psyre_comb
  def __rtruediv__(self, stream):
    return stream / self

  @psyre_comb
  def __floordiv__(self, stream):
    return self // stream

  @psyre_comb
  def __rfloordiv__(self, stream):
    return stream // self

  @psyre_comb
  def __mod__(self, stream):
    return self % stream

  @psyre_comb
  def __rmod__(self, stream):
    return stream % self

  @psyre_comb
  def __pow__(self, stream):
    return self**stream

  @psyre_comb
  def __rpow__(self, stream):
    return stream**self

  @psyre_comb
  def __lshift__(self, stream):
    return self << stream

  @psyre_comb
  def __rlshift__(self, stream):
    return stream << self

  @psyre_comb
  def __rshift__(self, stream):
    return self >> stream

  @psyre_comb
  def __rrshift__(self, stream):
    return stream >> self

  @psyre_comb
  def __and__(self, stream):
    return self & stream

  @psyre_comb
  def __rand__(self, stream):
    return stream & self

  @psyre_comb
  def __or__(self, stream):
    return self | stream

  @psyre_comb
  def __ror__(self, stream):
    return stream | self

  @psyre_comb
  def __xor__(self, stream):
    return self ^ stream

  @psyre_comb
  def __rxor__(self, stream):
    return stream ^ self

  @psyre_comb
  def __pos__(self):
    return +self

  @psyre_comb
  def __neg__(self):
    return -self

  @psyre_comb
  def __invert__(self):
    return (not self) if isinstance(self, bool) else ~self

  @psyre_comb
  def __lt__(self, stream):
    return self < stream

  @psyre_comb
  def __le__(self, stream):
    return self <= stream

  @psyre_comb
  def __eq__(self, stream):
    return self == stream

  @psyre_comb
  def __ne__(self, stream):
    return self != stream

  @psyre_comb
  def __gt__(self, stream):
    return self > stream

  @psyre_comb
  def __ge__(self, stream):
    return self >= stream

  @psyre_comb
  def max(self, stream, *rest):
    """Returns the pointwise maximum across two or more inputs."""
    return max(self, stream, *rest)

  @psyre_comb
  def min(self, stream, *rest):
    """Returns the pointwise minimum across two or more inputs."""
    return min(self, stream, *rest)


def periodic_stream(
    prefix: Sequence[Any], period: Sequence[Any], name: str = ''
) -> Stream:
  """Returns a static periodic stream with the given prefix and period."""
  return Stream(list(prefix), period=list(period), name=name)


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
  """Scheduler and execution context for Psyre networks."""

  deadlocked = _SchedulerAttr[bool]('deadlocked')
  graph = _SchedulerAttr[Graph]('graph')
  semantics = _SchedulerAttr[Semantics]('semantics')

  _primitives_registry: dict[Semantics, dict[str, Any]] = {}

  def __init__(
      self,
      semantics: Semantics = Semantics.KAHN,
      horizon: int = _DEFAULT_HORIZON,
      graph: Graph | None = None,
  ):
    """Initializes a Scheduler instance with its own Stage-1/Stage-2 state."""
    self._semantics: Semantics = semantics
    self._primitives: dict[str, Any] = self._primitives_registry.get(
        semantics, {}
    )
    self._horizon: int = horizon
    self._deadlocked: bool = False
    self._building: bool = False
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
  def is_building(cls) -> bool:
    """Returns True while a Stage-1 Graph build is active."""
    return cls.current()._building

  @staticmethod
  def get_resets() -> tuple[Any, ...]:
    """Returns the active Stage-1 reset stack."""
    return reset_stack.get()

  restore_resets = staticmethod(restore_resets)

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
      semantics: Semantics = Semantics.KAHN,
      causality_analysis: bool = True,
      clock_calculus: bool = True,
      streams: Iterable[str] | None = None,
      optimize: bool = True,
      **kwargs: Any,
  ) -> Any:
    """Constructs and checks the Psyre graph (Stage 1), then runs it (Stage 2)."""
    graph = cls.build_graph(
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
  def build_graph(
      cls,
      main_func: Callable[..., Any],
      *args: Any,
      semantics: Semantics = Semantics.KAHN,
      optimize: bool = True,
      extra_roots: Iterable[Any] | None = None,
      **kwargs: Any,
  ) -> Graph:
    """Executes Stage 1 on a fresh Scheduler instance to build a Graph."""
    sched = cls(semantics=semantics)
    active_scheduler.set(sched)
    return sched._build(
        main_func, *args, optimize=optimize, extra_roots=extra_roots, **kwargs
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
    """Executes Stage 2 scheduling on a fresh Scheduler instance."""
    if graph is None:
      graph = cls.current().graph
    graph.require_finalized()
    semantics = graph.semantics or Semantics.KAHN
    sched = cls(semantics=semantics, horizon=horizon, graph=graph)
    active_scheduler.set(sched)
    return sched._execute(streams=streams, clocks=clocks)

  @classmethod
  def require_sync_semantics(cls, construct: str = '') -> None:
    """Raises RuntimeError if the active Scheduler is not using SYNC semantics."""
    if cls.current().semantics != Semantics.SYNC:
      subject = construct or 'Operation'
      raise RuntimeError(f'{subject} is only supported under SYNC semantics.')

  @classmethod
  def register_primitives(cls, semantics: Semantics, module: Any) -> None:
    """Registers a primitives module for the given semantics."""
    prims = vars(module)
    cls._primitives_registry[semantics] = prims
    active = active_scheduler.get()
    if active is not None and active.semantics == semantics:
      active._primitives = prims  # pylint: disable=protected-access

  @classmethod
  def register_handle(cls, handle: Stream) -> None:
    """Registers a Stream handle in the active Stage-1 Graph if building."""
    if cls.is_building():
      cls.current().graph.add_stream(handle)

  def _build(
      self,
      main_func: Callable[..., Any],
      *args: Any,
      optimize: bool = True,
      extra_roots: Iterable[Any] | None = None,
      **kwargs: Any,
  ) -> Graph:
    """Constructs, finalizes, and optionally optimizes `self._graph`."""
    self._building = True
    try:
      with CapturedEnv().restore():
        output = main_func(*args, **kwargs)
        self._graph.bind_output(output)
        self._graph.finalize(resolve_handle)
        if optimize:
          eliminate_dead_code(self._graph, extra_roots=extra_roots)
    finally:
      self._building = False
    return self._graph

  def _execute(
      self,
      *,
      streams: Iterable[str] | None = None,
      clocks: ClockAnalysis | None = None,
  ) -> Any:
    """Runs Stage-2 node tasks for `self._graph` and extracts traces."""
    if self._semantics == Semantics.KAHN:
      check_kahn_resets(self._graph, clocks)
    named = self._graph.prepare_capture(streams)

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
    return self._graph.extract_traces(self._horizon, named)

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
    if node is None or not (node.in_arm or node.op == 'post'):
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

  def _loop(
      self,
      step_func: Callable[..., Any],
      *args: Any,
      _prepare: PrepareFn | None = None,
      **kwargs: Any,
  ) -> Stream:
    """Registers a node in `self._graph` and returns its output Stream."""
    if not self._building:
      raise RuntimeError('Psyre nodes can only be instantiated during Stage 1.')
    op = callable_name(step_func)
    name = kwargs.pop('name', op)
    o_stream = kwargs.pop('o_stream', None)
    if _prepare is not None:
      args, kwargs = _prepare(*args, **kwargs)
    if o_stream is None:
      o_stream = Stream(name='' if name == '<lambda>' else name)
    is_stateful = (
        op in STATEFUL_PRIMITIVE_OPS or step_func in _STATEFUL_STEP_FUNCS
    )
    self._graph.add_node(
        op=op,
        name=name,
        output=o_stream,
        step_func=functools.partial(step_func, o_stream),
        raw_inputs=args,
        raw_resets=reset_stack.get() if is_stateful else (),
        kwargs=kwargs,
        is_stateful=is_stateful,
    )
    return o_stream


def register_primitives(semantics: Semantics, module: Any) -> None:
  """Registers a primitives module for the given semantics."""
  Scheduler.register_primitives(semantics, module)


def _validate_post_run_trace(trace: Any, caller: str) -> Sequence[Any]:
  """Validates that trace is a post-run Sequence inspected outside Scheduler."""
  sched = Scheduler.current()
  if sched._building or sched._tasks:  # pylint: disable=protected-access
    raise RuntimeError(f'{caller}() can only be used outside a Scheduler run.')
  if isinstance(trace, (IStruct, IStructHandle, str, bytes, bytearray)) or (
      not isinstance(trace, Sequence)
  ):
    raise TypeError(
        f'{caller}() expects a post-run trace Sequence, got '
        f'{type(trace).__name__}.'
    )
  return trace


def is_present(trace: Sequence[Any], index: int) -> bool:
  """Checks whether element `index` of a post-run trace is present.

  Args:
    trace: Completed trace `Sequence` returned after a Scheduler run.
    index: Tick index in `trace` (supports standard Python indexing).

  Returns:
    `True` if `trace[index]` is not `Absent`, `False` otherwise.

  Raises:
    RuntimeError: If called during a Scheduler Stage-1 build or Stage-2 run.
    TypeError: If `trace` is a Psyre `Stream`/`IStruct` or not a `Sequence`.
    IndexError: If `index` is out of range for `trace`.
  """
  validated = _validate_post_run_trace(trace, 'is_present')
  return validated[index] is not Absent


def is_absent(trace: Sequence[Any], index: int) -> bool:
  """Checks whether element `index` of a post-run trace is `Absent`.

  Args:
    trace: Completed trace `Sequence` returned after a Scheduler run.
    index: Tick index in `trace` (supports standard Python indexing).

  Returns:
    `True` if `trace[index]` is `Absent`, `False` otherwise.

  Raises:
    RuntimeError: If called during a Scheduler Stage-1 build or Stage-2 run.
    TypeError: If `trace` is a Psyre `Stream`/`IStruct` or not a `Sequence`.
    IndexError: If `index` is out of range for `trace`.
  """
  validated = _validate_post_run_trace(trace, 'is_absent')
  return validated[index] is Absent


def present_values(trace: Sequence[Any]) -> list[Any]:
  """Extracts all present (non-`Absent`) elements from a post-run trace.

  Args:
    trace: Completed trace `Sequence` returned after a Scheduler run.

  Returns:
    A list of all elements `v` in `trace` such that `v is not Absent`.

  Raises:
    RuntimeError: If called during a Scheduler Stage-1 build or Stage-2 run.
    TypeError: If `trace` is a Psyre `Stream`/`IStruct` or not a `Sequence`.
  """
  validated = _validate_post_run_trace(trace, 'present_values')
  return [item for item in validated if item is not Absent]


def _make_primitive(name: str) -> Callable[..., Stream]:
  """Creates a Stage-1 primitive dispatcher for `name`."""

  def dispatch(*args, **kwargs) -> Stream:
    sched = Scheduler.current()
    if not sched._building:  # pylint: disable=protected-access
      raise RuntimeError('Psyre nodes can only be instantiated during Stage 1.')
    prim = sched._primitives.get(name)  # pylint: disable=protected-access
    if prim is None:
      reg = Scheduler._primitives_registry  # pylint: disable=protected-access
      if name in reg.get(Semantics.SYNC, {}):
        Scheduler.require_sync_semantics(name)
      raise RuntimeError(
          f'{name} is not supported under {sched.semantics.name} semantics.'
      )
    return prim(*args, **kwargs)

  dispatch.__name__ = name
  return dispatch


buffer = _make_primitive('buffer')
counter = _make_primitive('counter')
current = _make_primitive('current')
delay = _make_primitive('delay')
fby = _make_primitive('fby')
ifthenelse = _make_primitive('ifthenelse')
merge = _make_primitive('merge')
morph = _make_primitive('morph')
once_stream = _make_primitive('once_stream')
post = _make_primitive('post')
pre = _make_primitive('pre')
prefix_sum = _make_primitive('prefix_sum')
random_stream = _make_primitive('random_stream')
scan = _make_primitive('scan')
when = _make_primitive('when')
whennot = _make_primitive('whennot')
