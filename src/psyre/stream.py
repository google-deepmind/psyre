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
"""Psyre Stream channel, Stage-2 StreamPort cursor, and pointwise operators.

A `Stream` is either:

* **Dynamic** (`Stream()`): an empty single-assignment channel populated at
  Stage-2 runtime by a single producer `Node` (pushing `Absent` on inactive
  ticks under `Semantics.SYNC`). Dynamic streams carry a concrete monomorphic
  synchronous clock (`.` or `. on ...`).
* **Static** (`Stream(v)`, `Stream([v0, ...])`, `Stream(word)`,
  `Stream(prefix, period=p)`, or `Stream.periodic(prefix, period)`): an
  eventually periodic infinite sequence consisting of a finite `prefix` followed
  by a repeating non-empty `period` (`prefix + period*ω`), both backed by
  symbolic `psyre.static_stream.StaticWord` expressions. Neither `prefix` nor
  `period` may contain `Absent` or `Bottom`.

All static streams — both 1-periodic constants (`Stream(v)`) and general
multi-element sequences (`Stream([v0, v1, ...])` / `Stream.periodic(...)`) —
are **uniformly clock-polymorphic** (`∀ 'a. 'a`): they belong to no single
fixed clock and are *never* implicitly downsampled by `match`/`automaton`.
See `psyre.clock` for full details on polymorphic clock instantiation.
"""

import asyncio
from collections.abc import Callable, Iterator
import functools
import logging
import operator
import types
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    Absent,
    Bottom,
    LazyThunk,
    UNREADY,
    UNSET,
    active_builder,
    active_scheduler,
    emit_comb_node,
    force_thunk,
    format_tag,
    is_thunk,
    logger,
    make_comb_step_func,
    psyre_comb,
    reset_stack,
    validate_non_negative_int,
)
from psyre.math import random_stream
from psyre.payload import (
    BUILTIN_SCALAR_TYPES,
    NUMPY_UFUNC_METHODS,
    is_pure_numpy_func,
    partition_comb_args,
    reject_callable,
    reject_numpy_inplace_ufunc,
    reject_numpy_mutating_func,
    strip_null_out,
    ufunc_nout,
)
from psyre.static_stream import (
    EMPTY_WORD,
    StaticSequence,
    StaticWord,
)

# pylint: enable=g-importing-member,g-multiple-import


_DEBUG = logging.DEBUG


def _lift_numpy_op(
    step_func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    nout: int = 1,
) -> Any:
  """Instantiates a Stage-1 lifted node for NEP-13/NEP-18 dispatch."""
  dyn_inputs, node_kwargs = partition_comb_args(
      args, strip_null_out(kwargs), static_literals=True
  )
  return emit_comb_node(
      step_func, dyn_inputs, node_kwargs, name=step_func.__name__, nout=nout
  )


def _make_stream_binop(
    name: str,
    op_fn: Callable[[Any, Any], Any],
    *,
    reflected: bool = False,
) -> Callable[[Any, Any], Any]:
  """Builds a `@psyre_comb(pure=True)` binary dunder wrapping `op_fn`."""

  def method(self, other):
    return op_fn(other, self) if reflected else op_fn(self, other)

  method.__name__ = name
  method.__qualname__ = f'Stream.{name}'
  return psyre_comb(method, pure=True)


def _make_stream_unaryop(
    name: str, op_fn: Callable[[Any], Any]
) -> Callable[[Any], Any]:
  """Builds a `@psyre_comb(pure=True)` unary dunder wrapping `op_fn`."""

  def method(self):
    return op_fn(self)

  method.__name__ = name
  method.__qualname__ = f'Stream.{name}'
  return psyre_comb(method, pure=True)


class Stream:
  """Single-assignment stream channel with pointwise operators.

  Note: Psyre stores references to stream values without defensively copying
  them or enforcing runtime immutability. The behavior of mutating values
  carried on streams in place is unspecified.

  Static streams (`static == True`) represent infinite, eventually periodic
  sequences of present values (`prefix + period*ω`) with no `Absent` or
  `Bottom` sentinels. In `psyre.clock`, every unpinned static stream is
  **clock-polymorphic** (`'a`, `'b`, ...): it is instantiated independently at
  each consumer port's clock (`input_clock`) and each `with reset(r):` scope's
  enclosing arm clock (`reset_clock`), always starting from index `0` of its
  dense sequence `prefix + period*ω`:

  * In `Semantics.SYNC`, a subclocked consumer reads the static stream as a
    gated emitter that produces `prefix`/`period` values on active ticks of
    that consumer's clock and `Absent` on inactive ticks.
  * In `Semantics.KAHN`, each consumer owns an independent `StreamPort` cursor
    that pulls from the static stream only on its own active ticks.
  * Unlike a dynamic stream produced by a `Node`, sharing one static `Stream`
    across multiple clocks (e.g. both `s + 1` on `.` and `merge(c, s, s)`) is
    valid: each distinct `(stream, clock)` site gets its own independent dense
    sequence starting from index `0`. When a single global base-clock timed
    waveform must be *sampled* across `match`/`automaton` arms (skipping values
    while an arm is inactive), anchor it onto the base clock as a dynamic node
    via `when(Stream(True), seq)`; `match`/`automaton` lowering inserts the
    per-arm `when` gates against that dynamic node before `optimize_graph`
    eliminates the outer `when(Stream(True), ...)` wrapper.
  """

  _id: int = 0
  _psyre_is_stream: bool = True
  __array_priority__: int = 1000
  __hash__ = None  # type: ignore[assignment]

  id: int
  name: str
  static: bool
  static_sequence: StaticSequence | None
  _is_const: bool
  _const_value: Any
  seq: list[Any]
  _seq_base: int
  _head: int
  _evict_enabled: bool
  _readers: list[Any] | None
  _single_reader: Any
  _evict_stride: int
  _evict_countdown: int
  _waiters: dict[
      int, list[tuple[asyncio.Future[Any], asyncio.Task[Any] | None]]
  ]

  def __new__(
      cls,
      value: Any = UNSET,
      *,
      name: str = '',
      period: Any = UNSET,
  ):
    del name
    if isinstance(value, Stream) and period is UNSET:
      return value
    return super().__new__(cls)

  def __init__(
      self,
      value: Any = UNSET,
      *,
      name: str = '',
      period: Any = UNSET,
  ):
    """Initializes a Stream channel and registers it if Stage 1 is active.

    With no positional arguments and no `period`, creates a dynamic stream
    (`static=False`). Otherwise, creates a **static** stream backed by a
    `StaticSequence` (`prefix + period*ω`):

    * `Stream(v)` → 1-periodic constant `prefix=EMPTY_WORD`, `period=[v]`.
    * `Stream([v0, ..., vk])` → finite list stuttering `vk` forever.
    * `Stream(word)` → symbolic `StaticWord` stuttering its final value forever
      (without expanding `S * k` repetitions).
    * `Stream(seq)` → wraps an existing `StaticSequence` directly.
    * `Stream(prefix, period=p)` / `Stream.periodic(prefix, p)` → finite
      `prefix` word expression followed by cyclic `period` (`StaticWord`,
      `list`, or scalar).

    Multi-argument `Stream(a1, ..., an)` construction is intentionally rejected;
    pass an explicit list `Stream([a1, ..., an])` or a `StaticWord` expression.

    Neither `prefix` nor `period` may contain `Absent` or `Bottom`. Static
    streams created outside an active `with reset(r):` scope (and all
    1-periodic constants, which are reset-invariant) are uniformly
    **clock-polymorphic** (`∀ 'a. 'a`): interpreted at each consumer's clock
    and activation scope, advancing from index `0` only on active ticks of that
    clock (inserting `Absent` on inactive ticks under `Semantics.SYNC`). A
    non-1-periodic static stream instantiated *inside* an active
    `with reset(r):` or `automaton` `state(...)` scope is emitted by a stateful
    resettable `static_stream` node whose cursor rewinds to index `0` whenever
    an enclosing reset fires (`Stream(existing)` preserves an
    already-constructed outer `Stream` without resetting it).

    Args:
      value: Omitted for a dynamic channel, an existing `Stream` to return
        as-is, a `StaticSequence`, a `StaticWord`, a `list` of values, or a
        scalar constant/prefix value.
      name: Optional human-readable stream name (or alias when wrapping an
        existing `Stream`).
      period: Optional repeating cycle (`StaticWord`, `list`, or scalar) of a
        static periodic stream.

    Raises:
      TypeError: On extra positional arguments, unexpected keyword arguments,
        `Absent`/`Bottom` sentinels, or unresolved `LazyThunk` arguments.
      ValueError: When a static stream has an empty value sequence or an empty
        `period`.
    """
    if isinstance(value, Stream) and period is UNSET:
      builder = active_builder.get()
      if builder is not None:
        builder.add_stream(self)
        if name and name != self.name:
          builder.graph.add_stream_name(self, name)
      elif name and not self.name:
        self.name = name
      return
    spec: StaticSequence | None
    if value is UNSET and period is UNSET:
      spec = None
    elif period is not UNSET:
      spec = StaticSequence.from_parts(value, period)
    else:
      spec = StaticSequence.from_value(value)
    self._init_fields(name=name, spec=spec)
    self._register_in_builder()

  def _init_fields(
      self, *, name: str, spec: StaticSequence | None
  ) -> None:
    """Initializes core Stream identity and backing static/dynamic storage."""
    self.id = Stream._id
    Stream._id += 1
    self.name = name
    self.static_sequence = spec
    self.static = spec is not None
    self._is_const = spec is not None and spec.is_one_periodic
    self._const_value = spec.const_value if self._is_const else UNSET
    self.seq = []
    self._seq_base = 0
    self._head = 0
    self._evict_enabled = False
    self._readers = None
    self._single_reader = None
    self._evict_stride = 1
    self._evict_countdown = 0
    self._waiters = {}

  def _register_in_builder(self) -> None:
    """Registers `self` with any active Stage-1 GraphBuilder."""
    builder = active_builder.get()
    if builder is None:
      return
    spec = self.static_sequence
    if spec is not None and not spec.is_one_periodic and reset_stack.get():
      source = Stream._from_static_sequence(spec, name=self.name)
      self.static = False
      self.static_sequence = None
      builder.add_static_stream_node(self, source, name=self.name)
    else:
      builder.add_stream(self)

  @classmethod
  def _new_detached(cls, *, name: str = '') -> 'Stream':
    """Creates an unregistered dynamic `Stream` without touching active builder."""
    stream = super().__new__(cls)
    stream._init_fields(name=name, spec=None)
    return stream

  @classmethod
  def _from_static_sequence(
      cls, spec: StaticSequence, *, name: str = ''
  ) -> 'Stream':
    """Creates an unregistered backing static `Stream` from `spec`."""
    stream = super().__new__(cls)
    stream._init_fields(name=name, spec=spec)
    return stream

  @property
  def prefix(self) -> StaticWord:
    """Symbolic finite prefix `StaticWord` (empty for dynamic streams)."""
    spec = self.static_sequence
    return EMPTY_WORD if spec is None else spec.prefix

  @property
  def period(self) -> StaticWord:
    """Symbolic periodic `StaticWord` (empty for dynamic streams)."""
    spec = self.static_sequence
    return EMPTY_WORD if spec is None else spec.period

  def _debug_tag(self) -> str:
    """Returns a compact `S<name>_<id>` label without touching payload repr."""
    return f'S{self.name}_{self.id}'

  def __str__(self) -> str:
    tag = self._debug_tag()
    if self.static_sequence is not None:
      return f'{tag}{self.static_sequence}'
    items = ', '.join(format_tag(v) for v in self.seq[self._head :])
    if self._seq_base or self._head:
      return f'{tag}[@{self._seq_base + self._head}:][{items}]'
    return f'{tag}[{items}]'

  @property
  def pushed_count(self) -> int:
    """Total logical slots pushed onto this dynamic `Stream`."""
    return self._seq_base + len(self.seq)

  @property
  def is_one_periodic(self) -> bool:
    """True when this static stream repeats a single constant value (`[v]*ω`).

    Used for Stage-1 constant checks and avoiding redundant `static_stream`
    nodes; clock inference treats all unpinned static streams uniformly as
    clock-polymorphic regardless of `is_one_periodic`.
    """
    return self._is_const

  @property
  def const_value(self) -> Any:
    """Returns the unique repeated value of a 1-periodic static `Stream`."""
    if not self._is_const:
      raise RuntimeError('const_value requires a 1-periodic static Stream.')
    return self._const_value

  def static_value_at(self, index: int) -> Any:
    """Returns the static periodic stream value at non-negative index."""
    if self.static_sequence is None:
      raise RuntimeError('static_value_at() requires a static Stream.')
    return self.static_sequence.value_at(index)

  def trace(self, horizon: int, *, pad_bottom: bool = False) -> list[Any]:
    """Returns dynamic values or the unclocked dense static sequence."""
    validate_non_negative_int(horizon, 'horizon')
    if pad_bottom:
      sched = active_scheduler.get()
      if active_builder.get() is not None or (
          sched is not None and sched.is_executing_tasks()
      ):
        raise RuntimeError(
            'Stream.trace(pad_bottom=True) can only be used after a Scheduler '
            'run completes.'
        )
    if self.static_sequence is not None:
      return self.static_sequence.trace(horizon)
    if self._seq_base or self._head:
      raise RuntimeError(
          'Full trace unavailable for evicted intermediate stream.'
      )
    if pad_bottom:
      if horizon < len(self.seq):
        raise ValueError(
            f'Cannot pad trace to horizon {horizon} < pushed count '
            f'{len(self.seq)}.'
        )
      if len(self.seq) < horizon:
        return [*self.seq, *([Bottom] * (horizon - len(self.seq)))]
    return list(self.seq)

  def _sync_eviction_mode(self, readers: list[Any]) -> None:
    """Updates cached single-reader pointer and sweep stride for `readers`."""
    # Rewindable `post` readers are stored as `(in_port, o_port)` tuples so
    # `isinstance(readers[0], StreamPort)` is `False`, disabling eager
    # single-reader eviction in favor of watermark `min(in_port, o_port)`.
    self._single_reader = (
        readers[0]
        if len(readers) == 1 and isinstance(readers[0], StreamPort)
        else None
    )
    self._evict_stride = min(max(8, len(readers) * 2), 64)

  def configure_eviction(self, readers: list[Any]) -> None:
    """Enables sliding-window slot eviction for an unprobed dynamic stream.

    Live dynamic entries map `logical_index == _seq_base + pos` over the window
    `[_seq_base + _head, _seq_base + len(seq))`. Cleared prefix slots in
    `seq[:_head]` hold `None` until amortized compaction removes them. Each
    entry in `readers` is either a consumer `StreamPort` (monotonic cursor) or
    a `(i_port, o_port)` tuple for a `post` node whose input cursor may rewind
    back to `o_port.index` on reset (watermark `min(i, o)`).

    Args:
      readers: Active consumer port registrations reading this stream.
    """
    if self.static:
      return
    self._evict_enabled = True
    self._readers = list(readers)
    self._sync_eviction_mode(self._readers)
    self._evict_countdown = self._evict_stride

  def _evict_before(self, target: int) -> None:
    """Reclaims stored dynamic stream slots with logical index `< target`."""
    rel = target - self._seq_base
    head = self._head
    if rel <= head:
      return
    seq = self.seq
    seq_len = len(seq)
    if rel >= seq_len:
      seq.clear()
      self._seq_base += seq_len
      self._head = 0
      return
    for pos in range(head, rel):
      seq[pos] = None
    self._head = rel
    # Hysteresis compaction avoids O(n) `list.pop(0)` / prefix shifts per tick.
    if rel >= 64 and rel * 2 >= seq_len:
      del seq[:rel]
      self._seq_base += rel
      self._head = 0

  def evict_ready(self) -> None:
    """Evicts dynamic stream slots already passed by every active reader."""
    if not self._evict_enabled:
      return
    readers = self._readers
    if not readers:
      if not self._waiters and self.seq:
        self._evict_before(self._seq_base + len(self.seq))
      return
    self._evict_before(
        min(
            min(entry[0].index, entry[1].index)
            if isinstance(entry, tuple)
            else entry.index
            for entry in readers
        )
    )

  def remove_reader(self, entry: Any) -> None:
    """Detaches a completed reader registration and reclaims passed slots."""
    readers = self._readers
    if not self._evict_enabled or readers is None:
      return
    for pos, candidate in enumerate(readers):
      if candidate is entry:
        readers.pop(pos)
        break
    self._sync_eviction_mode(readers)
    self.evict_ready()

  def reset(self) -> None:
    """Clears dynamic stream buffers and pending waiters for a fresh run."""
    if not self.static:
      self.seq.clear()
      self._seq_base = 0
      self._head = 0
      self._evict_enabled = False
      self._readers = None
      self._single_reader = None
      self._evict_stride = 1
      self._evict_countdown = 0
      self._waiters.clear()

  async def pull(self, index: int) -> Any:
    """Returns the value at the given stream index, waiting if not yet pushed."""
    if self.static:
      return self.static_value_at(index)
    if index < 0:
      raise ValueError(f'Stream index must be non-negative, got {index}.')
    base = self._seq_base
    if index < base + self._head:
      raise RuntimeError(f'Stream slot {index} was already evicted.')
    seq = self.seq
    if index < base + len(seq):
      value = seq[index - base]
    else:
      fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
      waiter = (fut, asyncio.current_task())
      self._waiters.setdefault(index, []).append(waiter)
      try:
        value = await fut
      finally:
        waiters = self._waiters.get(index)
        if waiters is not None:
          try:
            waiters.remove(waiter)
          except ValueError:
            pass
          if not waiters:
            self._waiters.pop(index, None)
    if value is Bottom:
      raise RuntimeError('pull() must never observe Bottom.')
    return value

  def push(self, value: Any, index: int) -> None:
    """Stores a value at the stream index and resolves any pending waiters."""
    if self.static:
      raise RuntimeError('Cannot push to a static stream.')
    if value is Bottom:
      raise TypeError(
          'Bottom cannot be pushed; only used to pad traces to horizon.'
      )
    if value is not Absent and type(value) not in BUILTIN_SCALAR_TYPES:
      reject_callable(value, label='Stream value', reject_streams=True)
    # Note: stream payload immutability is intentionally not checked or
    # enforced at runtime; the behavior of mutating values on streams is
    # unspecified.
    seq = self.seq
    pushed = self._seq_base + len(seq)
    if index != pushed:
      raise RuntimeError(
          f'Out-of-order push at slot {index}; expected {pushed}.'
      )
    waiters = self._waiters.pop(index, None) if self._waiters else None
    if self._evict_enabled:
      single = self._single_reader
      if not waiters and (
          single.index > pushed if single is not None else not self._readers
      ):
        if seq:
          seq.clear()
          self._head = 0
        self._seq_base = pushed + 1
        return
      seq.append(value)
      if single is None:
        self._evict_countdown -= 1
        if self._evict_countdown <= 0:
          self._evict_countdown = self._evict_stride
          self.evict_ready()
    else:
      seq.append(value)
    if waiters:
      sched = active_scheduler.get()
      for fut, task in waiters:
        if not fut.done():
          fut.set_result(value)
          # Eagerly clear `task` from `_blocked` before it resumes on the next
          # event-loop turn so `_check_deadlock` stays O(1) and cannot mistake
          # woken waiters for a deadlock.
          if sched is not None and task is not None:
            sched.unblock_task(task, self)

  @classmethod
  def constant(cls, value: Any, *, name: str = '') -> 'Stream':
    """Returns a 1-periodic constant `Stream` repeating `value` verbatim.

    Unlike `Stream([v0, v1])` (which constructs a temporal sequence), this wraps
    any first-order Python object verbatim—including a `list`, `StaticWord`,
    `StaticSequence`, `np.ndarray`, or `jax.Array`—as a single constant payload
    repeated at every active tick (`[value]*ω`).

    Args:
      value: First-order constant payload (must not be a `Stream`, `LazyThunk`,
        `Absent`/`Bottom` sentinel, or callable).
      name: Optional human-readable stream name.

    Returns:
      A 1-periodic, clock-polymorphic static `Stream`.

    Raises:
      TypeError: If `value` is a `Stream`, `LazyThunk`, sentinel, or callable.
    """
    if isinstance(value, Stream):
      raise TypeError(
          'Stream.constant() cannot wrap a Stream instance; use Stream(s) to '
          'alias an existing Stream.'
      )
    return cls(StaticSequence.constant(value), name=name)

  @classmethod
  def periodic(
      cls,
      prefix: Any,
      period: Any,
      name: str = '',
  ) -> 'Stream':
    """Returns a static periodic stream with the given prefix and period."""
    return cls(StaticSequence.from_parts(prefix, period), name=name)

  @classmethod
  def once(cls, k: int, *, name: str = '') -> 'Stream':
    """Returns an `O(1)` static impulse stream emitting `True` once at `k`.

    Symbolically represented as `StaticWord(False) * k + True` followed by
    `[False]*ω` (independent of `k` in both memory and construction time).
    Outside any `with reset(...)` / `automaton` state scope this yields a
    clock-polymorphic static `Stream`; inside a reset scope it yields a
    resettable `static_stream` node that rewinds to index `0` on reset.

    Args:
      k: Non-negative integer index of the single `True` tick (booleans are
        rejected).
      name: Optional human-readable stream/node name.

    Returns:
      A `Stream` yielding `True` at index `k` and `False` elsewhere.

    Raises:
      ValueError: If `k` is negative, a `bool`, or not an `int`.
    """
    return cls(StaticSequence.once(k), name=name)

  @classmethod
  def random(cls, **kwargs: Any) -> 'Stream':
    """Returns a stream of random floats in [0.0, 1.0)."""
    return random_stream(**kwargs)

  def __bool__(self) -> bool:
    raise TypeError(
        'Truth value of a Stream is ambiguous; use Psyre combinators '
        '(e.g. ifthenelse, merge, when, Stream.max, or Stream.min).'
    )

  def __iter__(self) -> Iterator[Any]:
    raise TypeError(
        'Stream is not iterable in Stage 1; index elements explicitly via '
        'stream[i] or unpack multi-output ops defined with nout=...'
    )

  def __setitem__(self, key: Any, value: Any) -> None:
    del key, value
    raise TypeError(
        'Stream payloads are immutable; item assignment is not supported.'
    )

  @psyre_comb(pure=True, static_literals=True)
  def __getitem__(self, key: Any) -> Any:
    """Returns a pointwise element/slice/key projection of each stream tick."""
    return self[key]

  def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
    del dtype, copy
    raise TypeError(
        'Cannot convert a Psyre Stream to a NumPy array; use lifted NumPy '
        'operations during Stage 1, or run Scheduler.run() to obtain traces.'
    )

  def __array_ufunc__(
      self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
  ) -> Any:
    """Auto-lifts NumPy ufunc calls (NEP-13) to `@psyre_comb(pure=True)`."""
    reject_numpy_inplace_ufunc(ufunc, method, kwargs)
    if method not in NUMPY_UFUNC_METHODS:
      return NotImplemented
    if any(
        isinstance(arg, LazyThunk)
        and getattr(type(arg), '__array_ufunc__', None) is not None
        for arg in (*inputs, *kwargs.values())
    ):
      return NotImplemented
    uname = getattr(ufunc, '__name__', 'ufunc')
    target = ufunc if method == '__call__' else getattr(ufunc, method)
    op_name = uname if method == '__call__' else f'{uname}.{method}'
    return _lift_numpy_op(
        make_comb_step_func(target, name=op_name, pure=True),
        inputs,
        kwargs,
        nout=ufunc_nout(ufunc, method),
    )

  def __array_function__(
      self,
      func: Any,
      types_arg: Any,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
  ) -> Any:
    """Auto-lifts NumPy array functions (NEP-18) to `@psyre_comb` nodes."""
    reject_numpy_mutating_func(func, kwargs)
    if any(
        issubclass(t, LazyThunk)
        and getattr(t, '__array_function__', None) is not None
        for t in types_arg
    ):
      return NotImplemented
    fname = getattr(func, '__name__', 'numpy_func')
    return _lift_numpy_op(
        make_comb_step_func(func, name=fname, pure=is_pure_numpy_func(func)),
        args,
        kwargs,
    )

  __add__ = _make_stream_binop('__add__', operator.add)
  __radd__ = _make_stream_binop('__radd__', operator.add, reflected=True)
  __sub__ = _make_stream_binop('__sub__', operator.sub)
  __rsub__ = _make_stream_binop('__rsub__', operator.sub, reflected=True)
  __mul__ = _make_stream_binop('__mul__', operator.mul)
  __rmul__ = _make_stream_binop('__rmul__', operator.mul, reflected=True)
  __matmul__ = _make_stream_binop('__matmul__', operator.matmul)
  __rmatmul__ = _make_stream_binop(
      '__rmatmul__', operator.matmul, reflected=True
  )
  __truediv__ = _make_stream_binop('__truediv__', operator.truediv)
  __rtruediv__ = _make_stream_binop(
      '__rtruediv__', operator.truediv, reflected=True
  )
  __floordiv__ = _make_stream_binop('__floordiv__', operator.floordiv)
  __rfloordiv__ = _make_stream_binop(
      '__rfloordiv__', operator.floordiv, reflected=True
  )
  __mod__ = _make_stream_binop('__mod__', operator.mod)
  __rmod__ = _make_stream_binop('__rmod__', operator.mod, reflected=True)
  __pow__ = _make_stream_binop('__pow__', operator.pow)
  __rpow__ = _make_stream_binop('__rpow__', operator.pow, reflected=True)
  __lshift__ = _make_stream_binop('__lshift__', operator.lshift)
  __rlshift__ = _make_stream_binop(
      '__rlshift__', operator.lshift, reflected=True
  )
  __rshift__ = _make_stream_binop('__rshift__', operator.rshift)
  __rrshift__ = _make_stream_binop(
      '__rrshift__', operator.rshift, reflected=True
  )
  __and__ = _make_stream_binop('__and__', operator.and_)
  __rand__ = _make_stream_binop('__rand__', operator.and_, reflected=True)
  __or__ = _make_stream_binop('__or__', operator.or_)
  __ror__ = _make_stream_binop('__ror__', operator.or_, reflected=True)
  __xor__ = _make_stream_binop('__xor__', operator.xor)
  __rxor__ = _make_stream_binop('__rxor__', operator.xor, reflected=True)
  __lt__ = _make_stream_binop('__lt__', operator.lt)
  __le__ = _make_stream_binop('__le__', operator.le)
  __eq__ = _make_stream_binop('__eq__', operator.eq)  # type: ignore[assignment]
  __ne__ = _make_stream_binop('__ne__', operator.ne)  # type: ignore[assignment]
  __gt__ = _make_stream_binop('__gt__', operator.gt)
  __ge__ = _make_stream_binop('__ge__', operator.ge)
  __pos__ = _make_stream_unaryop('__pos__', operator.pos)
  __neg__ = _make_stream_unaryop('__neg__', operator.neg)
  __abs__ = _make_stream_unaryop('__abs__', operator.abs)

  @psyre_comb(pure=True)
  def __invert__(self):
    # Python's `~bool` returns a signed int (`~True == -2`, `~False == -1`),
    # so map unary `~` on booleans to logical negation `not`.
    return (not self) if isinstance(self, bool) else ~self

  @psyre_comb(pure=True)
  def max(self, stream, *rest):
    """Returns the pointwise scalar maximum across two or more inputs."""
    return max(self, stream, *rest)

  @psyre_comb(pure=True)
  def min(self, stream, *rest):
    """Returns the pointwise scalar minimum across two or more inputs."""
    return min(self, stream, *rest)

  @psyre_comb(pure=True)
  def sum(self, *rest):
    """Returns the pointwise sum across inputs without a `0` seed."""
    return functools.reduce(operator.add, rest, self)


# Expose `Stream[T]` generic aliasing at runtime without forcing downstream
# bare `Stream` type annotations to be flagged as unparameterized generics.
setattr(Stream, '__class_getitem__', classmethod(types.GenericAlias))


class StreamPort:
  """Lightweight Stage-2 per-port read/write cursor over a Stream."""

  __slots__ = ('stream', 'index')

  stream: Stream
  index: int

  def __init__(self, stream: Stream):
    self.stream = stream
    self.index = 0

  def __str__(self) -> str:
    return f'{self.stream}@{self.index}'

  def try_pull_ready(self) -> Any:
    """Synchronously reads next slot if already available, else `UNREADY`."""
    stream = self.stream
    idx = self.index
    if stream.static:
      if stream._is_const:  # pylint: disable=protected-access
        result = stream._const_value  # pylint: disable=protected-access
      else:
        spec = stream.static_sequence
        assert spec is not None
        result = spec.value_at(idx)
      if logger.level <= _DEBUG and logger.isEnabledFor(_DEBUG):
        logger.debug(
            'pull %s, %s -> %s',
            stream._debug_tag(),  # pylint: disable=protected-access
            idx,
            format_tag(result),
        )
      self.index = idx + 1
      return result
    if idx < 0:
      raise ValueError(f'Stream index must be non-negative, got {idx}.')
    base = stream._seq_base  # pylint: disable=protected-access
    head = stream._head  # pylint: disable=protected-access
    if idx < base + head:
      raise RuntimeError(f'Stream slot {idx} was already evicted.')
    seq = stream.seq
    pos = idx - base
    seq_len = len(seq)
    if pos >= seq_len:
      return UNREADY
    result = seq[pos]
    if result is Bottom:
      raise RuntimeError('pull() must never observe Bottom.')
    if logger.level <= _DEBUG and logger.isEnabledFor(_DEBUG):
      logger.debug(
          'pull %s, %s -> %s',
          stream._debug_tag(),  # pylint: disable=protected-access
          idx,
          format_tag(result),
      )
    next_idx = idx + 1
    self.index = next_idx
    # Intentional hot-loop inline mirror of `Stream._evict_before` avoiding a
    # method call per single-reader pull.
    if stream._single_reader is self:  # pylint: disable=protected-access
      next_pos = pos + 1
      if next_pos == seq_len:
        seq.clear()
        stream._seq_base = next_idx  # pylint: disable=protected-access
        stream._head = 0  # pylint: disable=protected-access
      elif pos == head:
        seq[pos] = None
        stream._head = next_pos  # pylint: disable=protected-access
        if next_pos >= 64 and next_pos * 2 >= seq_len:
          del seq[:next_pos]
          stream._seq_base = base + next_pos  # pylint: disable=protected-access
          stream._head = 0  # pylint: disable=protected-access
      else:
        stream._evict_before(next_idx)  # pylint: disable=protected-access
    return result

  async def pull(self) -> Any:
    """Pulls the next value from the stream and advances the cursor."""
    idx = self.index
    stream = self.stream
    result = await stream.pull(idx)
    if logger.level <= _DEBUG and logger.isEnabledFor(_DEBUG):
      logger.debug(
          'pull %s, %s -> %s',
          stream._debug_tag(),  # pylint: disable=protected-access
          idx,
          format_tag(result),
      )
    self.index = idx + 1
    if stream._single_reader is self:  # pylint: disable=protected-access
      stream._evict_before(idx + 1)  # pylint: disable=protected-access
    return result

  def push(self, value: Any) -> None:
    """Pushes a value into the stream and advances the cursor."""
    idx = self.index
    if logger.level <= _DEBUG and logger.isEnabledFor(_DEBUG):
      logger.debug(
          'push %s, %s <- %s',
          self.stream._debug_tag(),  # pylint: disable=protected-access
          idx,
          format_tag(value),
      )
    self.stream.push(value, idx)
    self.index = idx + 1

  def skip(self) -> None:
    """Advances the read cursor past an active-tick discarded slot."""
    stream = self.stream
    idx = self.index
    if logger.level <= _DEBUG and logger.isEnabledFor(_DEBUG):
      logger.debug(
          'skip %s, %s',
          stream._debug_tag(),  # pylint: disable=protected-access
          idx,
      )
    self.index = idx + 1
    if stream._single_reader is self:  # pylint: disable=protected-access
      stream._evict_before(idx + 1)  # pylint: disable=protected-access

  def skip_absent(self) -> None:
    """Advances past an inactive (`Absent`) dynamic slot, preserving statics."""
    if not self.stream.static:
      self.skip()


def same_stream_or_const(a: Stream, b: Stream) -> bool:
  """Returns True when `a` and `b` are identical streams or equal constants."""
  if a is b:
    return True
  return (
      a.static_sequence is not None
      and b.static_sequence is not None
      and a.static_sequence.same_constant(b.static_sequence)
  )


def resolve_stream(stream_or_thunk: Any) -> Stream:
  """Resolves a deferred Stage-1 thunk or wraps a literal into a Stream.

  Stage-1 thunks (zero-argument functions and `LazyThunk` proxies) must resolve
  directly to a `Stream` instance (never to a callable, nested thunk, or bare
  non-`Stream` value). Non-thunk arguments are wrapped via `Stream(...)`.

  Args:
    stream_or_thunk: A `Stream`, a Stage-1 thunk evaluating to a `Stream`, or a
      literal/sequence wrapped via `Stream(...)`.

  Returns:
    The resolved `Stream` instance.

  Raises:
    TypeError: If a Stage-1 thunk resolves to a callable or non-`Stream` value.
  """
  if is_thunk(stream_or_thunk):
    resolved = force_thunk(stream_or_thunk)
    if not isinstance(resolved, Stream):
      raise TypeError(
          'Stage-1 thunk must resolve to a Stream instance, got '
          f'{type(resolved).__name__}.'
      )
    return resolved
  return Stream(stream_or_thunk)
