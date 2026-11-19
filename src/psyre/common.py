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
"""Common Psyre definitions, decorators, primitives, and scope stacks."""

from __future__ import annotations

from collections.abc import (
    AsyncGenerator,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
)
import contextlib
import contextvars
import dataclasses
import enum
import functools
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
    is_one_periodic,
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
BOOL_DOMAIN: frozenset[TagKey] = frozenset({(True, True), (True, False)})


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


def tag_sort_key(item: TagKey) -> tuple[str, str]:
  """Provides a deterministic sort key across heterogeneous discrete tags."""
  is_bool, val = item
  return ('0' if is_bool else '1', format_tag(val))


def same_stream_or_const(
    a: IStruct | IStructHandle, b: IStruct | IStructHandle
) -> bool:
  """Returns True when a and b are identical streams or equal constants."""
  sa, sb = as_istruct(a), as_istruct(b)
  if sa is sb:
    return True
  if is_one_periodic(sa) and is_one_periodic(sb):
    av, bv = sa.period[0], sb.period[0]
    if av is bv:
      return True
    if type(av) is type(bv):
      try:
        return bool(av == bv)
      except (TypeError, ValueError):
        return False
  return False


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
class WhenSelector:
  """Callable tag predicate with inspectable match tags for `when`."""

  is_boolean: bool = True
  values: frozenset[TagKey] = frozenset()
  negated: bool = False

  def __call__(self, tag: Any) -> bool:
    """Returns True when tag passes this when filter."""
    if self.is_boolean:
      return (not bool(tag)) if self.negated else bool(tag)
    matched = tag_key(tag) in self.values
    return (not matched) if self.negated else matched


def normalize_when_selector(
    selector: WhenSelector | None,
    *,
    is_bool: bool,
    op: str = 'when',
) -> tuple[bool, frozenset[TagKey], bool]:
  """Returns canonical (is_boolean, values, negated) for a when/whennot."""
  if op == 'whennot':
    return (is_bool, frozenset({(True, False)}), False)
  if selector is None:
    return (is_bool, frozenset({(True, True)}), False)
  if selector.is_boolean:
    return (is_bool, frozenset({(True, not selector.negated)}), False)
  if is_bool:
    relevant = selector.values & BOOL_DOMAIN
    effective = (BOOL_DOMAIN - relevant) if selector.negated else relevant
    return (True, effective, False)
  return (False, selector.values, selector.negated)


def is_tautological_selector(
    norm_sel: tuple[bool, frozenset[TagKey], bool],
) -> bool:
  """Returns True when a canonicalized selector matches every possible tag."""
  is_bool, vals, neg = norm_sel
  return (vals == BOOL_DOMAIN and not neg) if is_bool else (not vals and neg)


def prepare_when(
    c_stream: Any,
    i_stream: Any,
    *,
    values: Any = UNSET,
    negated: bool = False,
    selector: WhenSelector | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates when arguments and returns ((c_stream, i_stream), kwargs)."""
  if selector is not None:
    if values is not UNSET or negated:
      raise ValueError(
          'when accepts either selector or values/negated, not both.'
      )
    if not isinstance(selector, WhenSelector):
      raise TypeError(
          'when selector must be a WhenSelector, got '
          f'{type(selector).__name__}.'
      )
    if selector.is_boolean and not selector.negated:
      return (c_stream, i_stream), {}
    return (c_stream, i_stream), {'selector': selector}
  if values is UNSET:
    if negated:
      return (c_stream, i_stream), {
          'selector': WhenSelector(is_boolean=True, negated=True)
      }
    return (c_stream, i_stream), {}
  if isinstance(values, (str, bytes, bytearray)) or not isinstance(
      values, Iterable
  ):
    raw_tags = (values,)
  else:
    raw_tags = tuple(values)
  tag_keys = frozenset(tag_key(v) for v in raw_tags)
  return (c_stream, i_stream), {
      'selector': WhenSelector(
          is_boolean=False, values=tag_keys, negated=bool(negated)
      )
  }


@dataclasses.dataclass(frozen=True)
class MergeSelector:
  """Callable branch-index selector with inspectable arm tags."""

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

  def arm_selector(self, arm_index: int) -> WhenSelector:
    """Returns the WhenSelector matching tags routed to `arm_index`."""
    if self.is_boolean:
      return WhenSelector(is_boolean=True, negated=(arm_index != 0))
    is_default = arm_index == self.default_index
    tags = frozenset(
        k
        for k, idx in self.arms.items()
        if (idx != arm_index if is_default else idx == arm_index)
    )
    return WhenSelector(is_boolean=False, values=tags, negated=is_default)


def _prepare_mux(
    op_name: str, selector: Any, *branches: Any, default: Any = None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates merge/switch args and returns ((selector, *arms), kwargs)."""
  if len(branches) == 2:
    return (selector, *branches), {'lookup': MergeSelector(is_boolean=True)}
  if len(branches) != 1 or not isinstance(branches[0], Mapping):
    raise TypeError(
        f'{op_name} expects either (c_stream, t_stream, f_stream) or '
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


def prepare_merge(
    selector: Any, *branches: Any, default: Any = None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates merge arguments and returns ((selector, *arm_streams), kwargs)."""
  return _prepare_mux('merge', selector, *branches, default=default)


def prepare_switch(
    selector: Any, *branches: Any, default: Any = None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates switch arguments and returns ((selector, *arms), kwargs)."""
  return _prepare_mux('switch', selector, *branches, default=default)


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
    return (i_stream,), {'p_stream': sched, 'size': steps + 2}
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
  return (i_stream,), {'p_stream': sched, 'size': size}


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


active_builder: contextvars.ContextVar[Any] = contextvars.ContextVar(
    'psyre_active_builder', default=None
)
active_scheduler: contextvars.ContextVar[Any] = contextvars.ContextVar(
    'psyre_active_scheduler', default=None
)


async def pull(stream: IStructHandle) -> Any:
  """Pulls the next value from a stream via the active Scheduler."""
  return await active_scheduler.get().pull(stream)


async def pull_all(*args: IStructHandle) -> tuple[Any, ...]:
  """Pulls one value from each input stream via the active Scheduler."""
  return await active_scheduler.get().pull_all(*args)


reset_stack: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar(
    'psyre_reset_stack', default=()
)
prefix_stack: contextvars.ContextVar[tuple[tuple[str, int], ...]] = (
    contextvars.ContextVar('psyre_prefix_stack', default=())
)


@dataclasses.dataclass(eq=False)
class ArmScope:
  """Stage-1 arm condition, tag selector, and reset/prefix stacks."""

  cond: Any = None
  selector: WhenSelector | None = None
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
def push_reset(r_stream: Any) -> Generator[None, None, None]:
  """Context manager resetting enclosed stateful nodes on `r_stream`."""
  token = reset_stack.set(reset_stack.get() + (r_stream,))
  try:
    yield
  finally:
    reset_stack.reset(token)


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
  arm_scope: ArmScope
  strong_transitions: list[Any] = dataclasses.field(default_factory=list)
  weak_transitions: list[Any] = dataclasses.field(default_factory=list)
  outputs: dict[str, Any] = dataclasses.field(default_factory=dict)


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


reset = push_reset

PrepareFn = Callable[..., tuple[tuple[Any, ...], dict[str, Any]]]
_STATEFUL_STEP_FUNCS: set[Callable[..., Any]] = set()
primitives_registry: dict[Semantics, dict[str, Any]] = {}


def is_stateful_step_func(step_func: Callable[..., Any]) -> bool:
  """Returns True if `step_func` was registered via `@psyre_sequ`."""
  return step_func in _STATEFUL_STEP_FUNCS


def register_primitives(semantics: Semantics, module: Any) -> None:
  """Registers a primitives module for the given semantics."""
  primitives_registry[semantics] = vars(module)


def is_building() -> bool:
  """Returns True while a Stage-1 Graph build is active."""
  return active_builder.get() is not None


def require_sync_semantics(construct: str = '') -> None:
  """Raises RuntimeError unless the active GraphBuilder/Scheduler uses SYNC."""
  builder = active_builder.get()
  if builder is not None:
    sem = builder.semantics
  else:
    sched = active_scheduler.get()
    sem = Semantics.SYNC if sched is None else sched.semantics
  if sem != Semantics.SYNC:
    subject = construct or 'Operation'
    raise RuntimeError(f'{subject} is only supported under SYNC semantics.')


async def _comb(
    inst_func: Callable[..., Any],
    o_stream: IStructHandle,
    *args: IStructHandle,
    **kwargs: Any,
) -> None:
  """Executes a combinational (stateless) step from a pure function."""
  logger.debug('comb %s, %s, %s', callable_name(inst_func), args, kwargs)
  i_vals = await pull_all(*args) if args else ()
  o_val = Absent if Absent in i_vals else inst_func(*i_vals, **kwargs)
  o_stream.push(o_val)


async def _sequ(
    inst_func: Callable[..., Any],
    o_stream: IStructHandle,
    *args: IStructHandle,
    **kwargs: Any,
) -> AsyncGenerator[None, Any]:
  """Executes a sequential (stateful) generator function across ticks."""
  gen = None
  while True:
    start = yield
    logger.debug('sequ %s, %s, %s', callable_name(inst_func), args, kwargs)
    i_vals = await pull_all(*args) if args else ()
    if start is Absent:
      o_stream.push(Absent)
      continue
    if gen is None or (start and (not i_vals or i_vals[0] is not Absent)):
      gen = inst_func(*i_vals, **kwargs)
      o_val = next(gen)
    else:
      o_val = gen.send(i_vals[0] if len(i_vals) == 1 else i_vals)
    o_stream.push(o_val)


def _require_builder() -> Any:
  """Returns the active Stage-1 GraphBuilder or raises RuntimeError."""
  builder = active_builder.get()
  if builder is None:
    raise RuntimeError('Psyre nodes can only be instantiated during Stage 1.')
  return builder


def psyre_loop(
    step_func: Callable[..., Any] | None = None,
    *,
    prepare: PrepareFn | None = None,
) -> Any:
  """@psyre_loop decorator for nodes."""

  def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      builder = _require_builder()
      return builder.add_loop_node(fn, *args, _prepare=prepare, **kwargs)

    return wrapper

  return decorate if step_func is None else decorate(step_func)


def psyre_comb(inst_func: Callable[..., Any]) -> Callable[..., Any]:
  """@psyre_comb decorator to define a node from a stateless function."""

  def step_func(
      o_stream: IStructHandle, *args: IStructHandle, **kwargs: Any
  ) -> Any:
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

  def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
    def step_func(
        o_stream: IStructHandle, *args: IStructHandle, **kwargs: Any
    ) -> Any:
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
      _require_builder()
      scope_name = kwargs.pop('name', name)
      if not isinstance(scope_name, str) or not scope_name:
        raise TypeError('psyre_node name must be a non-empty str.')
      with push_prefix(scope_name):
        return fn(*args, **kwargs)

    return wrapper

  return decorator


def make_primitive(name: str) -> Callable[..., Any]:
  """Creates a Stage-1 primitive dispatcher for `name`."""

  def dispatch(*args, **kwargs) -> Any:
    builder = _require_builder()
    sem = builder.semantics or Semantics.SYNC
    prim = primitives_registry.get(sem, {}).get(name)
    if prim is None:
      if name in primitives_registry.get(Semantics.SYNC, {}):
        require_sync_semantics(name)
      raise RuntimeError(
          f'{name} is not supported under {sem.name} semantics.'
      )
    return prim(*args, **kwargs)

  dispatch.__name__ = name
  return dispatch


buffer = make_primitive('buffer')
counter = make_primitive('counter')
current = make_primitive('current')
delay = make_primitive('delay')
fby = make_primitive('fby')
ifthenelse = make_primitive('ifthenelse')
merge = make_primitive('merge')
morph = make_primitive('morph')
post = make_primitive('post')
pre = make_primitive('pre')
prefix_sum = make_primitive('prefix_sum')
scan = make_primitive('scan')
switch = make_primitive('switch')
when = make_primitive('when')
whennot = make_primitive('whennot')
