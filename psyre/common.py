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
"""Common Psyre definitions, sentinels, ring buffers, and Stage-1 scope stacks."""

from collections.abc import AsyncGenerator, Callable, Generator, Iterator, Mapping
import contextlib
import contextvars
import dataclasses
import enum
import logging
import operator
import sys
import types
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.istructure import (
    Absent,
    IStruct,
    IStructHandle,
    as_istruct,
)

# pylint: enable=g-importing-member,g-multiple-import

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('psyre')
logger.setLevel(logging.WARNING)

# Clear existing handlers to prevent duplicate or conflicting logs
for _handler in logger.handlers[:]:
  logger.removeHandler(_handler)

# Configure a logging StreamHandler directed explicitly to standard output
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(
    logging.Formatter('%(levelname)s:%(name)s:%(message)s')
)
logger.addHandler(_stdout_handler)

UNSET = object()


class Semantics(enum.Enum):
  """Execution semantics for Psyre node primitives."""

  KAHN = enum.auto()
  SYNC = enum.auto()


STATEFUL_PRIMITIVE_OPS = frozenset({
    'fby',
    'pre',
    'morph',
    'current',
    'counter',
    'prefix_sum',
    'scan',
    'once_stream',
    'buffer',
    'delay',
    'post',
})

BOOL_PRODUCER_OPS = frozenset({
    '__eq__',
    '__ne__',
    '__lt__',
    '__le__',
    '__gt__',
    '__ge__',
    'once_stream',
})

BOOL_BINARY_OPS = frozenset({
    '__and__',
    '__rand__',
    '__or__',
    '__ror__',
    '__xor__',
    '__rxor__',
})

TagKey = tuple[bool, Any]


def tag_key(tag: Any) -> TagKey:
  """Returns a type-disambiguated key separating bools from ints."""
  return (isinstance(tag, bool), tag)


def matches_tag(selector_value: Any, target_tag: Any) -> bool:
  """Checks whether selector_value matches target_tag without bool/int mixup."""
  if isinstance(target_tag, bool) or target_tag is None:
    return selector_value is target_tag
  if isinstance(selector_value, bool) or selector_value is None:
    return False
  return bool(selector_value == target_tag)


def format_tag(value: Any, max_len: int | None = None) -> str:
  """Formats a scalar or Enum tag value concisely for diagnostics and labels."""
  text = (
      f'{type(value).__name__}.{value.name}'
      if isinstance(value, enum.Enum)
      else repr(value)
  )
  if max_len is not None and len(text) > max_len:
    return f'{text[: max_len - 1]}…'
  return text


def is_thunk(val: Any) -> bool:
  """Returns True when val is a zero-argument Python function."""
  return isinstance(val, types.FunctionType) and val.__code__.co_argcount == 0


def callable_name(fn: Callable[..., Any], default: str = '') -> str:
  """Returns fn.__name__ when available, or default otherwise."""
  return getattr(fn, '__name__', default)


def validate_non_negative_int(k: int, op_name: str = 'delay k') -> int:
  """Validates that `k` is a non-negative integer (rejecting booleans)."""
  if isinstance(k, bool) or not isinstance(k, int) or k < 0:
    raise ValueError(f'{op_name} must be a non-negative integer, got {k!r}.')
  return k


def prepare_iterated_n(n: int, op_name: str = 'iterated op') -> int:
  """Validates that iteration count `n` is an integer >= 1."""
  if isinstance(n, bool) or not isinstance(n, int) or n < 1:
    raise ValueError(f'{op_name} n must be an integer >= 1, got {n!r}.')
  return n


@dataclasses.dataclass(frozen=True)
class MergeSelector:
  """Callable branch-index selector with inspectable arm tags for merge."""

  is_boolean: bool
  arms: Mapping[TagKey, int] = dataclasses.field(default_factory=dict)
  default_index: int | None = None

  def __call__(self, tag: Any) -> int:
    """Returns the zero-based arm stream index selected by tag."""
    if self.is_boolean:
      return 0 if tag else 1
    idx = self.arms.get(tag_key(tag), self.default_index)
    if idx is None:
      raise KeyError(f'Unhandled merge selector value: {tag!r}')
    return idx


def prepare_merge(
    selector: Any, *branches: Any, default: Any = None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates merge arguments and returns ((selector, *arm_streams), kwargs)."""
  if len(branches) == 2:
    return (selector, *branches), {'lookup': MergeSelector(is_boolean=True)}
  if len(branches) != 1 or not isinstance(branches[0], Mapping):
    raise TypeError(
        'merge expects either (c_stream, t_stream, f_stream) or '
        '(s_stream, branches_mapping, *, default=...)'
    )
  unique: dict[int, tuple[int, Any]] = {}

  def add_arm(arm: Any) -> int:
    return unique.setdefault(id(arm), (len(unique), arm))[0]

  arms = {tag_key(key): add_arm(arm) for key, arm in branches[0].items()}
  default_idx = None if default is None else add_arm(default)
  arm_streams = tuple(arm for _, arm in unique.values())
  lookup = MergeSelector(
      is_boolean=False, arms=arms, default_index=default_idx
  )
  return (selector, *arm_streams), {'lookup': lookup}


def prepare_fby(
    i_stream: Any, j_stream: Any, n: int = 1
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates fby arguments and returns ((i_stream, j_stream), {'n': n})."""
  return (i_stream, j_stream), {'n': prepare_iterated_n(n, 'fby')}


def prepare_pre(
    i_stream: Any, n: int = 1
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates pre arguments and returns ((i_stream,), {'n': n})."""
  return (i_stream,), {'n': prepare_iterated_n(n, 'pre')}


def prepare_morph(
    i_stream: Any, j_stream: Any, n: int = 1
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates morph arguments and returns ((i_stream, j_stream), {'n': n})."""
  return (i_stream, j_stream), {'n': prepare_iterated_n(n, 'morph')}


def prepare_post(
    i_stream: Any, n: int = 1
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates post arguments and returns ((i_stream,), {'n': n})."""
  return (i_stream,), {'n': prepare_iterated_n(n, 'post')}


def prepare_current(
    b_stream: Any, i_stream: Any, default: Any = None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes current arguments into ((b_stream, i_stream, default), {})."""
  return (b_stream, i_stream, default), {}


def prepare_counter(
    start: Any = 0, max_val: Any = None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes counter arguments, omitting max_val when None."""
  return ((start,) if max_val is None else (start, max_val)), {}


def prepare_prefix_sum(
    data: Any, start: Any = 0
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes prefix_sum arguments into ((data, start), {})."""
  return (data, start), {}


_INST_FUNCS: dict[Callable[..., Any], Callable[..., Any]] = {}


def register_inst_func(
    wrapper: Callable[..., Any], raw_func: Callable[..., Any]
) -> None:
  """Associates a `@psyre_comb` wrapper with its underlying Python function."""
  _INST_FUNCS[wrapper] = raw_func


def unwrap_inst_func(fn: Callable[..., Any]) -> Callable[..., Any]:
  """Returns the underlying Python function for a `@psyre_comb` wrapper."""
  return _INST_FUNCS.get(fn, fn)


def prepare_delay(
    i_stream: Any, k: int | None = None, *, period: Any = None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates delay arguments and returns ((i_stream,), kwargs)."""
  if period is not None and k is not None:
    raise ValueError('delay accepts either k or period, not both.')
  if period is None:
    steps = validate_non_negative_int(1 if k is None else k, 'delay k')
    sched = IStruct([False] * steps, period=[True])
    return (i_stream,), {'p_stream': IStructHandle(sched), 'size': steps + 2}
  if not isinstance(period, (IStruct, IStructHandle)):
    raise TypeError(
        'delay period must be a static periodic Stream or IStruct, got '
        f'{type(period).__name__}.'
    )
  sched = as_istruct(period)
  if not sched.static or not sched.period:
    raise ValueError('delay period must be a non-empty static periodic stream.')
  if not all(isinstance(v, bool) for v in (*sched.prefix, *sched.period)):
    raise ValueError('delay period stream must contain only boolean values.')
  size = len(sched.prefix) + len(sched.period) + 1
  return (i_stream,), {'p_stream': IStructHandle(sched), 'size': size}


def prepare_scan(
    data: Any, op: Callable[..., Any] = operator.add
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes scan arguments into ((data,), {'bin_op': raw_op})."""
  return (data,), {'bin_op': unwrap_inst_func(op)}


class RingBuffer:
  """Fixed-capacity ring buffer for buffer and delay primitives."""

  def __init__(self, size: int = 1):
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
      raise ValueError(f'RingBuffer size must be >= 1, got {size!r}.')
    self.size: int = size
    self.slots: list[Any] = [None] * size
    self.pull_idx: int = 0
    self.push_idx: int = 0

  def pull(self, val: Any) -> None:
    """Writes `val` at `pull_idx` and advances `pull_idx`."""
    self.slots[self.pull_idx] = val
    self.pull_idx = (self.pull_idx + 1) % self.size

  def push(self) -> Any:
    """Reads and clears the slot at `push_idx` and advances `push_idx`."""
    val = self.slots[self.push_idx]
    self.slots[self.push_idx] = None
    self.push_idx = (self.push_idx + 1) % self.size
    return val


active_scheduler: contextvars.ContextVar[Any] = contextvars.ContextVar(
    'psyre_active_scheduler', default=None
)


async def pull(stream: IStructHandle) -> Any:
  """Pulls the next value from a single input stream via the active Scheduler."""
  return await active_scheduler.get().pull(stream)


async def pull_all(*args: IStructHandle) -> tuple[Any, ...]:
  """Pulls one value from each input stream via the active Scheduler."""
  return await active_scheduler.get().pull_all(*args)


async def fby_step(
    o_stream: IStructHandle,
    i_stream: IStructHandle | None,
    j_stream: IStructHandle,
    *,
    n: int = 1,
) -> AsyncGenerator[None, Any]:
  """Executes an n-stage fby/pre delay across ticks."""
  stages: list[Any] = [Absent] * n
  pending: list[tuple[Any, Any]] = []

  def emit(m: int, s_m: Any, i_m: Any) -> Any:
    if s_m is Absent:
      return Absent
    if s_m and (i_m is not Absent or stages[m] is Absent):
      stages[m] = i_m
    return stages[m]

  def update(m: int, s_m: Any, val: Any) -> None:
    if s_m is not Absent and val is not Absent:
      stages[m] = val

  while True:
    start = yield
    init_val = None if i_stream is None else await pull(i_stream)
    if start is Absent and not pending:
      o_stream.push(Absent)
      await pull(j_stream)
      continue
    pending.append((start, init_val))
    if len(pending) < n:
      prev: Any = Absent
      for m in range(len(pending) - 1, n):
        out = emit(m, start, init_val)
        update(m, start, prev)
        prev = out
      o_stream.push(prev)
    else:
      o_stream.push(emit(n - 1, start, init_val))
      in_val = await pull(j_stream)
      s_0, _ = pending.pop(0)
      update(0, s_0, in_val)
      for m, (s_m, i_m) in enumerate(pending):
        update(m + 1, s_m, emit(m, s_m, i_m))


async def morph_step(
    o_stream: IStructHandle,
    i_stream: IStructHandle,
    j_stream: IStructHandle,
    *,
    n: int = 1,
) -> AsyncGenerator[None, Any]:
  """Executes an n-step initialization morph node across ticks."""
  rem = n
  while True:
    start = yield
    if start is Absent:
      o_stream.push(Absent)
      await pull_all(i_stream, j_stream)
      continue
    if start:
      rem = n
    if rem > 0:
      rem -= 1
      o_stream.push(await pull(i_stream))
      await pull(j_stream)
    else:
      o_stream.push(await pull(j_stream))
      await pull(i_stream)


reset_stack: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar(
    'psyre_reset_stack', default=()
)
prefix_stack: contextvars.ContextVar[tuple[tuple[str, int], ...]] = (
    contextvars.ContextVar('psyre_prefix_stack', default=())
)


@dataclasses.dataclass(eq=False)
class ArmScope:
  """Stage-1 arm condition and reset/prefix stacks."""

  cond: Any = None
  reset_depth: int = dataclasses.field(
      default_factory=lambda: len(reset_stack.get())
  )
  prefixes: tuple[tuple[str, int], ...] = dataclasses.field(
      default_factory=prefix_stack.get
  )


arm_stack: contextvars.ContextVar[tuple[ArmScope, ...]] = (
    contextvars.ContextVar('psyre_arm_stack', default=())
)


@dataclasses.dataclass(frozen=True)
class CapturedEnv:
  """Captured Stage-1 arm, prefix, and reset stacks for match/automaton."""

  arms: tuple[ArmScope, ...] = ()
  prefixes: tuple[tuple[str, int], ...] = ()
  resets: tuple[Any, ...] = ()

  @classmethod
  def capture(cls) -> 'CapturedEnv':
    """Captures the active Stage-1 arm, prefix, and reset stacks."""
    return cls(
        arms=arm_stack.get(),
        prefixes=prefix_stack.get(),
        resets=reset_stack.get(),
    )

  @contextlib.contextmanager
  def restore(self) -> Generator[None, None, None]:
    """Temporarily restores the captured arm, prefix, and reset stacks."""
    arm_token = arm_stack.set(self.arms)
    pfx_token = prefix_stack.set(self.prefixes)
    rst_token = reset_stack.set(self.resets)
    try:
      yield
    finally:
      reset_stack.reset(rst_token)
      prefix_stack.reset(pfx_token)
      arm_stack.reset(arm_token)


@contextlib.contextmanager
def restore_resets(resets: tuple[Any, ...]) -> Generator[None, None, None]:
  """Temporarily sets the active Stage-1 reset stack to `resets`."""
  token = reset_stack.set(resets)
  try:
    yield
  finally:
    reset_stack.reset(token)


@contextlib.contextmanager
def push_reset(r_stream: Any) -> Generator[None, None, None]:
  """Context manager resetting enclosed stateful nodes on `r_stream`."""
  with restore_resets(reset_stack.get() + (r_stream,)):
    yield


@contextlib.contextmanager
def push_prefix(prefix: str) -> Generator[None, None, None]:
  """Pushes a hierarchical node name prefix segment onto the active stack."""
  entry = (prefix, len(reset_stack.get()))
  token = prefix_stack.set(prefix_stack.get() + (entry,))
  try:
    yield
  finally:
    prefix_stack.reset(token)


@contextlib.contextmanager
def push_arm(arm: ArmScope) -> Generator[None, None, None]:
  """Pushes an ArmScope onto the active Stage-1 arm stack."""
  token = arm_stack.set(arm_stack.get() + (arm,))
  try:
    yield
  finally:
    arm_stack.reset(token)


@dataclasses.dataclass(frozen=True)
class MatchArm:
  """Descriptor for a single arm of a match() or automaton() block."""

  values: tuple[Any, ...] = ()
  is_wildcard: bool = False


@dataclasses.dataclass
class ArmRecord:
  """Collected transitions, arm scope, and named outputs for a single arm."""

  arm: MatchArm
  strong_transitions: list[Any] = dataclasses.field(default_factory=list)
  weak_transitions: list[Any] = dataclasses.field(default_factory=list)
  outputs: dict[str, Any] = dataclasses.field(default_factory=dict)
  arm_scope: ArmScope | None = None
  s_activity_stream: Any = None


@dataclasses.dataclass
class BlockContext:
  """Active Stage-1 construction state for a match() or automaton() block."""

  kind: str
  enclosing_env: CapturedEnv
  active_selector: Any = None
  init: Any = UNSET
  s_stream: Any = None
  r_stream: Any = None
  ns_stream: Any = None
  nr_stream: Any = None
  claimed_tags: list[Any] = dataclasses.field(default_factory=list)
  has_wildcard: bool = False
  arm_records: list[ArmRecord] = dataclasses.field(default_factory=list)
  active_arm: ArmRecord | None = None
  unconsumed_unless: list[Any] = dataclasses.field(default_factory=list)
  merged_outputs: dict[str, Any] = dataclasses.field(default_factory=dict)
  finalized: bool = False


active_block: contextvars.ContextVar[BlockContext | None] = (
    contextvars.ContextVar('psyre_active_block', default=None)
)


class BlockHandle(Mapping[str, Any]):
  """Namespace binding per-arm outputs and exposing merged output Streams."""

  _ctx: BlockContext

  def __init__(self, ctx: BlockContext):
    super().__setattr__('_ctx', ctx)

  def __getattr__(self, name: str) -> Any:
    ctx = self._ctx
    if ctx.finalized:
      if name in ctx.merged_outputs:
        return ctx.merged_outputs[name]
      raise AttributeError(f'{ctx.kind}() handle has no output {name!r}.')
    if ctx.active_arm is not None and name in ctx.active_arm.outputs:
      return ctx.active_arm.outputs[name]
    raise AttributeError(
        f'Output {name!r} is not finalized until the {ctx.kind}() block exits; '
        'wrap forward references in lambda: ...'
    )

  def __setattr__(self, name: str, value: Any) -> None:
    ctx = self._ctx
    if name.startswith('_'):
      raise AttributeError(
          f'Attribute {name!r} is reserved on {ctx.kind}() handles.'
      )
    if ctx.finalized:
      raise AttributeError(
          f'Cannot assign output {name!r} after {ctx.kind}() block has exited.'
      )
    arm_fn = 'case()' if ctx.kind == 'match' else 'state()'
    if active_block.get() is not ctx:
      raise RuntimeError(
          f'Cannot assign output {name!r} on enclosing {ctx.kind}() handle '
          'from another block.'
      )
    if ctx.active_arm is None:
      raise RuntimeError(
          f'Output {name!r} must be assigned inside a {arm_fn} block of '
          f'{ctx.kind}().'
      )
    ctx.active_arm.outputs[name] = value

  def _finalized_outputs(self) -> dict[str, Any]:
    ctx = self._ctx
    if not ctx.finalized:
      raise RuntimeError(
          f'{ctx.kind}() handle cannot be returned before its block exits.'
      )
    return ctx.merged_outputs

  def __getitem__(self, name: str) -> Any:
    return self._finalized_outputs()[name]

  def __iter__(self) -> Iterator[str]:
    return iter(self._finalized_outputs())

  def __len__(self) -> int:
    return len(self._finalized_outputs())


def require_open_arm_context(expected_kind: str, caller: str) -> BlockContext:
  """Returns the enclosing BlockContext for case(), default(), or state()."""
  ctx = active_block.get()
  article = 'an' if expected_kind == 'automaton' else 'a'
  if ctx is None or ctx.kind != expected_kind:
    hint = (
        'use state() inside automaton()'
        if expected_kind == 'match'
        else 'use case() or default() inside match()'
    )
    raise RuntimeError(
        f'{caller} can only be used directly inside {article} '
        f'{expected_kind}() block ({hint}).'
    )
  if ctx.active_arm is not None:
    raise RuntimeError(
        f'Nested {caller} requires an inner {expected_kind}() block.'
    )
  return ctx
