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
"""Common Psyre definitions, decorators, primitives, and scope stacks.

Defines the two execution semantics (`Semantics.SYNC` following Lustre —
Halbwachs et al., *Proc. IEEE* 1991 — and `Semantics.KAHN` following Kahn
Process Networks — Kahn, *IFIP* 1974), the two non-value sentinels (`Absent`
for synchronous clock absence and `Bottom` for undefined/uninitialized slots
or horizon-truncated traces), combinational/sequential node wrappers, and
shared Stage-1 scope stacks (`reset`, `prefix`, `match`/`automaton` arms).
"""

from __future__ import annotations

import abc
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
import inspect
import logging
import operator
import sys
import types
from typing import Any, Protocol

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common_payload import (
    CALL_PLAN_KWARG,
    PlanNode,
    is_array_object,
    is_int_ge,
    is_numpy_bool,
    is_zero_arg_function,
    normalize_static_argnames,
    normalize_static_argnums,
    partition_comb_args,
    reject_callable,
    safe_scalar_eq,
    validate_positive_int,
)
# Re-exported for internal/downstream modules (`graph`, `match`, `visualize`,
# `static_stream`, `scheduler`, examples).
from psyre.common_payload import (  # pylint: disable=unused-import
    is_namedtuple,
    validate_non_negative_int,
)

# pylint: enable=g-importing-member,g-multiple-import


class _Sentinel:
  """Singleton sentinel for non-value stream entries (`Absent`, `Bottom`).

  Both sentinels return `False` from `__eq__` (which also leaves them
  unhashable via `__hash__ = None`), forcing callers to use identity checks
  (`x is Absent` / `x is Bottom`) rather than `==` so sentinels can never
  accidentally match arbitrary payload values or act as mapping keys. Note that
  Python's `list.__eq__` compares element identity before calling `__eq__`, so
  `[Absent] == [Absent]` remains `True` in unit-test assertions.
  """

  _psyre_is_sentinel: bool = True
  __slots__ = ('_name',)

  def __init__(self, name: str):
    self._name = name

  def __bool__(self) -> bool:
    return False

  def __eq__(self, _) -> bool:
    # Always False (even for self): enforce `is Absent` / `is Bottom`.
    return False

  def __repr__(self) -> str:
    return self._name


Absent = _Sentinel('Absent')  # pylint: disable=invalid-name
Bottom = _Sentinel('Bottom')  # pylint: disable=invalid-name


def reject_sentinel(value: Any, *, label: str = 'Stream value') -> None:
  """Raises TypeError if `value` is the `Absent` or `Bottom` sentinel.

  Psyre stores references to stream values without defensively copying them or
  enforcing runtime immutability; the behavior of mutating values carried on
  streams in place is unspecified.

  Args:
    value: Candidate payload value to check.
    label: Subject label prefix used in `TypeError` messages.

  Raises:
    TypeError: If `value` is `Absent` or `Bottom`.
  """
  if value is Absent:
    raise TypeError(f'{label} cannot contain Absent.')
  if value is Bottom:
    raise TypeError(f'{label} cannot contain Bottom.')


logger = logging.getLogger('psyre')
logger.setLevel(logging.WARNING)
logger.propagate = False

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
UNREADY = object()
_DEBUG = logging.DEBUG


class Semantics(enum.Enum):
  """Execution semantics for Psyre node primitives."""

  KAHN = enum.auto()
  SYNC = enum.auto()


BOOL_PRODUCER_OPS = frozenset({
    '__eq__',
    '__ne__',
    '__lt__',
    '__le__',
    '__gt__',
    '__ge__',
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
  if is_numpy_bool(tag):
    return (True, bool(tag))
  return (isinstance(tag, bool), tag)


def matches_tag(selector_value: Any, target_tag: Any) -> bool:
  """Checks whether selector_value matches target_tag without bool/int mixup."""
  if is_numpy_bool(target_tag):
    target_tag = bool(target_tag)
  if is_numpy_bool(selector_value):
    selector_value = bool(selector_value)
  if isinstance(target_tag, bool) or target_tag is None:
    return selector_value is target_tag
  if isinstance(selector_value, bool) or selector_value is None:
    return False
  return safe_scalar_eq(selector_value, target_tag)


def format_tag(value: Any, max_len: int | None = None) -> str:
  """Formats a scalar, Enum, or array descriptor concisely for diagnostics."""
  if isinstance(value, enum.Enum):
    text = f'{type(value).__name__}.{value.name}'
  elif is_array_object(value):
    dims = ','.join(str(d) for d in value.shape)
    text = f'{value.dtype}[{dims}]'
  else:
    text = repr(value)
  if max_len is not None and len(text) > max_len:
    return f'{text[: max_len - 1]}…'
  return text


def tag_sort_key(item: TagKey) -> tuple[str, str]:
  """Provides a deterministic sort key across heterogeneous discrete tags."""
  is_bool, val = item
  return ('0' if is_bool else '1', format_tag(val))


class _StreamPort(Protocol):
  """Structural protocol for Stage-2 `StreamPort` cursors used by primitives."""

  def try_pull_ready(self) -> Any:
    ...

  def push(self, value: Any) -> None:
    ...

  def skip_absent(self) -> None:
    ...


class LazyThunk(abc.ABC):
  """Abstract base class for non-lambda deferred Stage-1 expression proxies."""

  _psyre_is_lazy_thunk: bool = True
  __slots__ = ()

  @abc.abstractmethod
  def __psyre_force__(self) -> Any:
    """Evaluates and returns the concrete Stage-1 value of this thunk."""


def is_thunk(val: Any) -> bool:
  """Returns True when val is a zero-argument Python function or LazyThunk."""
  return isinstance(val, LazyThunk) or is_zero_arg_function(val)


def force_thunk(val: Any) -> Any:
  """Forces a zero-arg function thunk or LazyThunk to a non-callable value.

  Args:
    val: Deferred `LazyThunk` proxy or zero-argument Python function.

  Returns:
    The non-callable, non-`LazyThunk` value produced by forcing `val`.

  Raises:
    TypeError: If `val` is not a thunk, or if forcing it yields a callable or
      another `LazyThunk`.
  """
  if isinstance(val, LazyThunk):
    result = val.__psyre_force__()
  elif is_zero_arg_function(val):
    result = val()
  else:
    raise TypeError(
        'Expected a LazyThunk or zero-argument function, got '
        f'{type(val).__name__}.'
    )
  if is_thunk(result):
    raise TypeError(
        'Nested Stage-1 thunks are not allowed; thunk must evaluate to a '
        f'non-callable value, got {type(result).__name__}.'
    )
  if callable(result):
    raise TypeError(
        'Thunk must evaluate to a non-callable value, got '
        f'{type(result).__name__}.'
    )
  return result


def callable_name(fn: Callable[..., Any], default: str = '') -> str:
  """Returns fn.__name__ when available, or default otherwise."""
  return getattr(fn, '__name__', default)


def prepare_iterated_n(n: int, op_name: str = 'iterated op') -> int:
  """Validates that iteration count `n` is an integer >= 1."""
  return validate_positive_int(n, f'{op_name} n')


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
    return (True, frozenset({(True, False)}), False)
  if selector is None:
    return (True, frozenset({(True, True)}), False)
  if selector.is_boolean:
    return (True, frozenset({(True, not selector.negated)}), False)
  if is_bool:
    relevant = selector.values & BOOL_DOMAIN
    effective = (BOOL_DOMAIN - relevant) if selector.negated else relevant
    if effective:
      return (True, effective, False)
    return (False, frozenset(), False)
  return (False, selector.values, selector.negated)


def selector_from_norm(
    norm_sel: tuple[bool, frozenset[TagKey], bool],
    fallback: WhenSelector | None = None,
) -> WhenSelector | None:
  """Converts a normalized `when` selector tuple into a canonical `WhenSelector`."""
  is_bool, vals, neg = norm_sel
  if is_bool and not neg:
    if vals == frozenset({(True, True)}):
      return None
    if vals == frozenset({(True, False)}):
      return WhenSelector(is_boolean=True, negated=True)
  if fallback is not None:
    return fallback
  return WhenSelector(is_boolean=False, values=vals, negated=neg)


def is_tautological_selector(
    norm_sel: tuple[bool, frozenset[TagKey], bool],
) -> bool:
  """Returns True when a canonicalized selector matches every possible tag."""
  is_bool, vals, neg = norm_sel
  return (vals == BOOL_DOMAIN and not neg) if is_bool else (not vals and neg)


def when_selector_matches_const(
    selector: WhenSelector | None,
    value: Any,
    *,
    op: str = 'when',
) -> bool | None:
  """Returns whether `when`/`whennot` selects constant `value`, or `None`.

  Returns `None` if `value` cannot safely be evaluated for truthiness or
  hashed for tag lookup (e.g. ambiguous array `__bool__` or unhashable payload).

  Args:
    selector: Optional `WhenSelector` predicate (`None` for plain boolean).
    value: Constant scalar/tag value from a 1-periodic static `Stream`.
    op: Operator name (`'when'` or `'whennot'`).

  Returns:
    `True` or `False` if statically decidable, otherwise `None`.
  """
  if is_array_object(value):
    return None
  try:
    if op == 'whennot':
      return not bool(value)
    if selector is None:
      return bool(value)
    return bool(selector(value))
  except (TypeError, ValueError):
    return None


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
  for tag in raw_tags:
    reject_sentinel(tag, label='when selector value')
    reject_callable(tag, label='when selector value')
  tag_keys = frozenset(tag_key(v) for v in raw_tags)
  return (c_stream, i_stream), {
      'selector': WhenSelector(
          is_boolean=False, values=tag_keys, negated=bool(negated)
      )
  }


_EMPTY_TAG_MAP: Mapping[TagKey, int] = types.MappingProxyType({})


@dataclasses.dataclass(frozen=True)
class MergeSelector:
  """Callable branch-index selector with inspectable arm tags."""

  is_boolean: bool
  arms: Mapping[TagKey, int] = _EMPTY_TAG_MAP
  default_index: int | None = None

  def __post_init__(self) -> None:
    if not isinstance(self.arms, types.MappingProxyType):
      frozen = types.MappingProxyType(dict(self.arms))
      object.__setattr__(self, 'arms', frozen)

  def __call__(self, tag: Any) -> int:
    """Returns the zero-based arm stream index selected by tag."""
    if self.is_boolean:
      return 0 if tag else 1
    idx = self.arms.get(tag_key(tag), self.default_index)
    if idx is None:
      raise KeyError(f'Unhandled merge selector value: {format_tag(tag)}')
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
    op_name: str,
    selector: Any,
    *branches: Any,
    default: Any = UNSET,
    lookup: MergeSelector | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates merge/switch args and returns ((selector, *arms), kwargs)."""
  if lookup is not None:
    if default is not UNSET:
      raise ValueError(
          f'{op_name} accepts either lookup or default, not both.'
      )
    if not isinstance(lookup, MergeSelector):
      raise TypeError(
          f'{op_name} lookup must be a MergeSelector, got '
          f'{type(lookup).__name__}.'
      )
    if not branches:
      raise ValueError(
          f'{op_name} with lookup requires at least one branch stream.'
      )
    if lookup.is_boolean:
      if len(branches) != 2:
        raise ValueError(
            f'Boolean {op_name} lookup expects 2 branch streams, got '
            f'{len(branches)}.'
        )
    else:
      n_branches = len(branches)
      indices = list(lookup.arms.values())
      if lookup.default_index is not None:
        indices.append(lookup.default_index)
      if any(idx < 0 or idx >= n_branches for idx in indices):
        raise ValueError(
            f'{op_name} lookup arm index out of range for {n_branches} '
            'branch streams.'
        )
    return (selector, *branches), {'lookup': lookup}
  if len(branches) == 2:
    if default is not UNSET:
      raise ValueError(
          f'Boolean {op_name}(c_stream, t_stream, f_stream) does not accept '
          'default=.'
      )
    return (selector, *branches), {'lookup': MergeSelector(is_boolean=True)}
  if len(branches) != 1 or not isinstance(branches[0], Mapping):
    raise TypeError(
        f'{op_name} expects either (c_stream, t_stream, f_stream) or '
        '(s_stream, branches_mapping, *, default=...)'
    )
  unique: dict[int, tuple[int, Any]] = {}

  def add_arm(arm: Any) -> int:
    return unique.setdefault(id(arm), (len(unique), arm))[0]

  arms: dict[TagKey, int] = {}
  for key, arm in branches[0].items():
    reject_sentinel(key, label=f'{op_name} branch tag')
    reject_callable(key, label=f'{op_name} branch tag')
    arms[tag_key(key)] = add_arm(arm)
  default_idx = None if default is UNSET else add_arm(default)
  arm_streams = tuple(arm for _, arm in unique.values())
  if not arm_streams:
    raise ValueError(f'{op_name} requires at least one branch stream.')
  resolved_lookup = MergeSelector(
      is_boolean=False, arms=arms, default_index=default_idx
  )
  return (selector, *arm_streams), {'lookup': resolved_lookup}


prepare_merge = functools.partial(_prepare_mux, 'merge')
prepare_switch = functools.partial(_prepare_mux, 'switch')


def _make_iterated_prepare(
    op_name: str, arity: int
) -> Callable[..., tuple[tuple[Any, ...], dict[str, Any]]]:
  """Builds a `prepare` helper validating positional args and iteration `n`."""
  if arity == 1:

    def prepare_unary(
        i_stream: Any, n: int = 1
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
      return (i_stream,), {'n': prepare_iterated_n(n, op_name)}

    return prepare_unary

  def prepare_binary(
      i_stream: Any, j_stream: Any, n: int = 1
  ) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (i_stream, j_stream), {'n': prepare_iterated_n(n, op_name)}

  return prepare_binary


prepare_fby = _make_iterated_prepare('fby', 2)
prepare_pre = _make_iterated_prepare('pre', 1)
prepare_morph = _make_iterated_prepare('morph', 2)
prepare_post = _make_iterated_prepare('post', 1)


def prepare_buffer(
    i_stream: Any,
    i_p_stream: Any,
    o_p_stream: Any,
    size: int = 1,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Validates `buffer` arguments and returns positional stream ports."""
  if not is_int_ge(size, 1):
    raise ValueError(f'buffer size must be >= 1, got {size!r}.')
  return (i_stream, i_p_stream, o_p_stream), {'size': size}


def prepare_current(
    b_stream: Any, i_stream: Any, default: Any = None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes current arguments into ((b_stream, i_stream, default), {})."""
  return (b_stream, i_stream, default), {}


def prepare_counter(
    start: Any = 0, max_val: Any = None, tick: Any = True
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes counter arguments, omitting max_val when None."""
  if max_val is None:
    return (start, tick), {}
  return (start, tick, max_val), {}


def prepare_prefix_sum(
    data: Any, start: Any = 0
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes prefix_sum arguments into ((data, start), {})."""
  return (data, start), {}


def unwrap_inst_func(fn: Callable[..., Any]) -> Callable[..., Any]:
  """Returns the underlying Python function for a `@psyre_comb` wrapper."""
  return getattr(fn, '_psyre_inst_func', fn)


def prepare_scan(
    data: Any, op: Callable[..., Any] = operator.add
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes scan arguments into ((data,), {'bin_op': raw_op})."""
  raw_op = unwrap_inst_func(op)
  if isinstance(raw_op, LazyThunk) or not callable(raw_op):
    raise TypeError(
        f'scan op must be callable, got {type(op).__name__}.'
    )
  return (data,), {'bin_op': raw_op}


class RingBuffer:
  """Strict fixed-capacity ring buffer for the `buffer` primitive.

  Circular-slice invariant: live buffered items (`slot is not Bottom`) always
  occupy the contiguous modular range `[push_idx, pull_idx) mod size`. Storing
  `Bottom` in empty/drained slots separates an empty slot from a valid buffered
  `None` payload *and* distinguishes empty (`slots[push_idx] is Bottom`) from
  full (`slots[pull_idx] is not Bottom`) when `pull_idx == push_idx` without a
  separate occupancy counter.

  Both `sync_primitives.buffer` and `kahn_primitives.buffer` execute `pull`
  *before* `push` on each active tick, so a full buffer of capacity `size`
  cannot accept a simultaneous `pull` and `push` (raises `BufferError`).
  `pull()` rejects `Absent`/`Bottom` and raises `BufferError` on overflow;
  `push()` resets the popped slot to `Bottom` and raises `BufferError` on
  underflow so `Bottom` never escapes the buffer.
  """

  def __init__(self, size: int = 1):
    if not is_int_ge(size, 1):
      raise ValueError(f'RingBuffer size must be >= 1, got {size!r}.')
    self.size: int = size
    self.slots: list[Any] = [Bottom] * size
    self.pull_idx: int = 0
    self.push_idx: int = 0

  def pull(self, val: Any) -> None:
    """Writes `val` at `pull_idx` and advances `pull_idx`."""
    reject_sentinel(val, label='RingBuffer value')
    if self.slots[self.pull_idx] is not Bottom:
      raise BufferError(
          f'RingBuffer overflow: pull into occupied slot {self.pull_idx} '
          f'(size={self.size}).'
      )
    self.slots[self.pull_idx] = val
    self.pull_idx = (self.pull_idx + 1) % self.size

  def push(self) -> Any:
    """Reads `push_idx`, resets it to `Bottom`, and advances `push_idx`."""
    val = self.slots[self.push_idx]
    if val is Bottom:
      raise BufferError(
          f'RingBuffer underflow: push from empty slot {self.push_idx} '
          'before matching pull.'
      )
    self.slots[self.push_idx] = Bottom
    self.push_idx = (self.push_idx + 1) % self.size
    return val


active_builder: contextvars.ContextVar[Any] = contextvars.ContextVar(
    'psyre_active_builder', default=None
)
active_scheduler: contextvars.ContextVar[Any] = contextvars.ContextVar(
    'psyre_active_scheduler', default=None
)


def _require_scheduler(caller: str = 'pull()') -> Any:
  """Returns the active Stage-2 Scheduler or raises RuntimeError."""
  sched = active_scheduler.get()
  if sched is None:
    raise RuntimeError(f'{caller} requires an active Scheduler run.')
  return sched


async def pull(stream: _StreamPort) -> Any:
  """Pulls the next value from a stream via the active Scheduler."""
  val = stream.try_pull_ready()
  if val is not UNREADY:
    return val
  return await _require_scheduler('pull()').pull(stream)


async def pull_all(*args: _StreamPort) -> tuple[Any, ...]:
  """Pulls one value from each input stream via the active Scheduler."""
  if not args:
    return ()
  sched = None
  out: list[Any] = []
  for port in args:
    val = port.try_pull_ready()
    if val is UNREADY:
      if sched is None:
        sched = _require_scheduler('pull_all()')
      val = await sched.pull(port)
    out.append(val)
  return tuple(out)


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
def _push_stack(
    var: contextvars.ContextVar[tuple[Any, ...]], item: Any
) -> Generator[None, None, None]:
  """Temporarily pushes `item` onto a tuple-valued `ContextVar` stack."""
  token = var.set(var.get() + (item,))
  try:
    yield
  finally:
    var.reset(token)


def push_reset(r_stream: Any) -> contextlib.AbstractContextManager[None]:
  """Context manager resetting enclosed stateful nodes on `r_stream`."""
  return _push_stack(reset_stack, r_stream)


def push_prefix(prefix: str) -> contextlib.AbstractContextManager[None]:
  """Pushes a hierarchical node name prefix segment onto the active stack."""
  return _push_stack(prefix_stack, (prefix, len(reset_stack.get())))


def push_arm(arm: ArmScope) -> contextlib.AbstractContextManager[None]:
  """Pushes an ArmScope onto the active Stage-1 arm stack."""
  return _push_stack(arm_stack, arm)


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
  output_envs: dict[str, CapturedEnv] = dataclasses.field(default_factory=dict)


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
    ctx.active_arm.output_envs[name] = CapturedEnv.capture()

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
primitives_registry: dict[Semantics, dict[str, Any]] = {}


def is_pure_step_func(step_func: Callable[..., Any]) -> bool:
  """Returns True if `step_func` was marked pure via `@psyre_comb`."""
  return bool(getattr(step_func, '_psyre_pure', False))


def is_builtin_scalar_step_func(step_func: Callable[..., Any]) -> bool:
  """Returns True if `step_func` is guaranteed to return a scalar."""
  return bool(getattr(step_func, '_psyre_returns_scalar', False))


def mark_builtin_scalar_op(wrapper: Any) -> None:
  """Marks a `@psyre_comb` wrapper's `step_func` as scalar-returning."""
  step_func = getattr(wrapper, '_psyre_step_func', None)
  if step_func is not None:
    setattr(step_func, '_psyre_returns_scalar', True)


def is_stateful_step_func(step_func: Callable[..., Any]) -> bool:
  """Returns True if `step_func` produces a stateful async generator."""
  return bool(
      getattr(step_func, '_psyre_stateful', False)
      or inspect.isasyncgenfunction(step_func)
  )


def is_step_of(owner: Any, op: str, step_func: Callable[..., Any]) -> bool:
  """Returns True if `step_func` matches the `@psyre_loop` wrapper `owner.<op>`."""
  return step_func is getattr(
      getattr(owner, op, None), '_psyre_step_func', None
  )


def is_builtin_primitive_step(
    op: str, step_func: Callable[..., Any]
) -> bool:
  """Returns True if `step_func` matches a registered builtin primitive `op`."""
  return any(
      step_func is getattr(mod.get(op), '_psyre_step_func', None)
      for mod in primitives_registry.values()
  )


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
    o_stream: _StreamPort,
    *args: _StreamPort,
    **kwargs: Any,
) -> None:
  """Executes a combinational (stateless) step from a pure function."""
  if logger.level <= _DEBUG and logger.isEnabledFor(_DEBUG):
    logger.debug(
        'comb %s, arity=%d, kwargs=%s',
        callable_name(inst_func),
        len(args),
        tuple(kwargs),
    )
  # Materialized-template slow path (nested PyTrees / dynamic kwargs /
  # static_argnums / static_literals): handled once up front before the
  # arity-specialized flat fast path.
  if kwargs and (plan := kwargs.get(CALL_PLAN_KWARG)) is not None:
    i_vals = await pull_all(*args) if args else ()
    if any(val is Absent for val in i_vals):
      o_stream.push(Absent)
      return
    call_args, call_kwargs = plan.materialize(i_vals)
    o_val = (
        inst_func(*call_args, **call_kwargs)
        if call_kwargs
        else inst_func(*call_args)
    )
  else:
    # Fast flat-positional path: arities 0, 1, and 2 are unrolled inline to
    # avoid per-tick coroutine/list/tuple allocation from `pull_all` on hot
    # scalar/array operators. Under `Semantics.SYNC`, every port is still
    # pulled before checking `Absent` so port cursors and EOF barriers align.
    n = len(args)
    if n == 0:
      o_val = inst_func(**kwargs) if kwargs else inst_func()
    elif n == 1:
      v0 = args[0].try_pull_ready()
      if v0 is UNREADY:
        v0 = await _require_scheduler('_comb()').pull(args[0])
      if v0 is Absent:
        o_stream.push(Absent)
        return
      o_val = inst_func(v0, **kwargs) if kwargs else inst_func(v0)
    elif n == 2:
      p0, p1 = args
      v0 = p0.try_pull_ready()
      if v0 is UNREADY:
        v0 = await _require_scheduler('_comb()').pull(p0)
      v1 = p1.try_pull_ready()
      if v1 is UNREADY:
        v1 = await _require_scheduler('_comb()').pull(p1)
      if v0 is Absent or v1 is Absent:
        o_stream.push(Absent)
        return
      o_val = inst_func(v0, v1, **kwargs) if kwargs else inst_func(v0, v1)
    else:
      i_vals = await pull_all(*args)
      if any(val is Absent for val in i_vals):
        o_stream.push(Absent)
        return
      o_val = inst_func(*i_vals, **kwargs) if kwargs else inst_func(*i_vals)
  if o_val is Absent:
    reject_sentinel(
        o_val, label=f'@psyre_comb {callable_name(inst_func, "<callable>")}'
    )
  o_stream.push(o_val)


async def _sequ(
    inst_func: Callable[..., Any],
    o_stream: _StreamPort,
    *args: _StreamPort,
    **kwargs: Any,
) -> AsyncGenerator[None, Any]:
  """Executes a sequential (stateful) generator function across ticks."""
  # Generator protocol:
  # - Initial/reset ticks construct `inst_func(*call_args, **call_kwargs)` and
  #   call `next(gen)`; later ticks call `gen.send(call_args[0])` for 1
  #   positional arg, `gen.send(call_args)` for 2+ args, or `gen.send(())` for
  #   0 args (with nested PyTree args reconstructed via `CombCallPlan` when
  #   active). Static kwargs stay unchanged in the generator frame across ticks.
  # - Under mandatory `SYNC` clock analysis (`lower_resets`), every subclocked
  #   inactive tick delivers `start is Absent`, skipping inputs via
  #   `skip_absent()` and keeping the generator paused (`Scheduler` latches any
  #   pending reset). On active ticks all inputs are present (`not Absent`).
  plan = kwargs.get(CALL_PLAN_KWARG) if kwargs else None
  gen = None
  try:
    while True:
      start = yield
      if logger.level <= _DEBUG and logger.isEnabledFor(_DEBUG):
        logger.debug(
            'sequ %s, arity=%d, kwargs=%s',
            callable_name(inst_func),
            len(args),
            tuple(kwargs),
        )
      if start is Absent:
        for stream in args:
          stream.skip_absent()
        o_stream.push(Absent)
        continue
      i_vals = await pull_all(*args) if args else ()
      if plan is not None:
        call_args, call_kwargs = plan.materialize(i_vals)
      else:
        call_args, call_kwargs = i_vals, kwargs
      if start or gen is None:
        if gen is not None:
          gen.close()
        gen = (
            inst_func(*call_args, **call_kwargs)
            if call_kwargs
            else inst_func(*call_args)
        )
        o_val = next(gen)
      else:
        o_val = gen.send(call_args[0] if len(call_args) == 1 else call_args)
      if o_val is Absent:
        reject_sentinel(
            o_val, label=f'@psyre_sequ {callable_name(inst_func, "<callable>")}'
        )
      o_stream.push(o_val)
  finally:
    if gen is not None:
      gen.close()


def require_builder() -> Any:
  """Returns the active Stage-1 GraphBuilder or raises RuntimeError."""
  builder = active_builder.get()
  if builder is None:
    raise RuntimeError('Psyre nodes can only be instantiated during Stage 1.')
  return builder


def _require_callable_target(target: Any, decorator_name: str) -> None:
  """Rejects non-callable targets passed to `@psyre_*` decorators."""
  if not callable(target):
    raise TypeError(
        f'{decorator_name} expects a callable target, got '
        f'{type(target).__name__}.'
    )


def psyre_loop(
    step_func: Callable[..., Any] | None = None,
    *,
    prepare: PrepareFn | None = None,
    stateful: bool = False,
) -> Any:
  """@psyre_loop decorator for nodes."""

  def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
    _require_callable_target(fn, 'psyre_loop')
    if stateful:
      setattr(fn, '_psyre_stateful', True)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
      return require_builder().add_loop_node(
          fn, *args, _prepare=prepare, **kwargs
      )

    setattr(wrapper, '_psyre_step_func', fn)
    return wrapper

  return decorate if step_func is None else decorate(step_func)


def make_comb_step_func(
    inst_func: Callable[..., Any],
    *,
    name: str = '',
    pure: bool = False,
) -> Callable[..., Any]:
  """Builds a combinational `step_func` closure wrapping `inst_func`."""
  # Keep closure freevar named `inst_func` for downstream tooling inspection.
  def step_func(
      o_stream: _StreamPort, *args: _StreamPort, **kwargs: Any
  ) -> Any:
    return _comb(inst_func, o_stream, *args, **kwargs)

  step_func.__name__ = name or callable_name(inst_func)
  if pure:
    setattr(step_func, '_psyre_pure', True)
  setattr(step_func, '_psyre_inst_func', inst_func)
  return step_func


def _wrap_partitioned_step(
    step_func: Callable[..., Any],
    *,
    prepare: PrepareFn | None,
    argnums: frozenset[int],
    argnames: frozenset[str],
    static_literals: bool,
    nout: int | None,
    decorator_name: str,
    reject_dynamic_kwargs: bool = False,
) -> Callable[..., Any]:
  """Wraps `step_func` with `CombCallPlan` arg partitioning and `nout` unpack."""

  def partitioned_prepare(
      *args: Any, **kwargs: Any
  ) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if prepare is not None:
      args, kwargs = prepare(*args, **kwargs)
    dyn_inputs, node_kwargs = partition_comb_args(
        args,
        kwargs,
        static_argnums=argnums,
        static_argnames=argnames,
        static_literals=static_literals,
    )
    if reject_dynamic_kwargs and (
        (plan := node_kwargs.get(CALL_PLAN_KWARG)) is not None
    ):
      for key, tmpl in plan.kwargs_template:
        if isinstance(tmpl, PlanNode):
          raise TypeError(
              f'@{decorator_name} does not support dynamic Stream/thunk '
              f'values in keyword argument {key!r} (generator.send() only '
              'accepts positional values); pass dynamic streams positionally '
              'or normalize them with prepare=.'
          )
    return dyn_inputs, node_kwargs

  base = psyre_loop(step_func, prepare=partitioned_prepare)
  if nout is None:
    return base

  def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    if 'o_stream' in kwargs and kwargs['o_stream'] is not None:
      raise ValueError(
          f'o_stream= is not supported on multi-output '
          f'@{decorator_name}(nout=...).'
      )
    raw_out = base(*args, **kwargs)
    return tuple(raw_out[i] for i in range(nout))

  setattr(wrapper, '_psyre_step_func', step_func)
  return wrapper


def psyre_comb(
    func: Callable[..., Any] | None = None,
    *,
    prepare: PrepareFn | None = None,
    pure: bool = False,
    nout: int | None = None,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    static_literals: bool = False,
) -> Any:
  """`@psyre_comb` decorator defining a stateless node from a pure function.

  Supports both bare `@psyre_comb` and parameterized
  `@psyre_comb(pure=True, nout=..., static_argnums=..., ...)` usage. Dynamic
  `Stream` and Stage-1 thunk leaves passed via keyword arguments or nested
  inside PyTrees (`tuple`, `NamedTuple`, `list`, `dict`/`Mapping`, `slice`,
  and types registered via `register_pytree_node`) are automatically extracted
  into the node's `inputs` and reconstructed at each Stage-2 tick.

  Args:
    func: Instantaneous callable to wrap (omitted when called with keyword
      options as a decorator factory).
    prepare: Optional Stage-1 argument normalizer `(args, kwargs) -> (args,
      kwargs)` run before dynamic-vs.-static partitioning.
    pure: When True, marks the node as referentially transparent so
      `optimize_graph` may apply clock-preserving CSE across identical inputs
      and static `kwargs`. Defaults to `False` (safe for impure functions such
      as `random_stream`).
    nout: Optional positive integer output arity. When set, calling the wrapped
      function unpacks its tuple-valued output stream into an `nout`-tuple of
      projected `Stream`s via pointwise indexing (`raw[0], ..., raw[nout - 1]`).
    static_argnums: Positional argument index or iterable of non-negative
      indices kept verbatim as static per-tick arguments (never converted to
      `Stream`s or rejected for being callable).
    static_argnames: Keyword argument name or iterable of names kept verbatim
      as static per-tick arguments.
    static_literals: When True, positional arguments containing no `Stream` or
      Stage-1 thunk stay verbatim inside the static call template instead of
      being promoted to constant/periodic `Stream`s via `resolve_stream`.

  Returns:
    A wrapped callable (or decorator) that emits a Stage-1 `Node` and returns a
    `Stream` (or an `nout`-tuple of `Stream`s when `nout` is specified).
  """
  if nout is not None:
    validate_positive_int(nout, 'psyre_comb nout')
  argnums = normalize_static_argnums(static_argnums)
  argnames = normalize_static_argnames(static_argnames)

  def decorate(inst_func: Callable[..., Any]) -> Callable[..., Any]:
    _require_callable_target(inst_func, 'psyre_comb')
    step_func = make_comb_step_func(inst_func, pure=pure)
    wrapper = _wrap_partitioned_step(
        step_func,
        prepare=prepare,
        argnums=argnums,
        argnames=argnames,
        static_literals=static_literals,
        nout=nout,
        decorator_name='psyre_comb',
    )
    if pure:
      setattr(wrapper, '_psyre_pure', True)
    setattr(wrapper, '_psyre_inst_func', inst_func)
    return wrapper

  return decorate if func is None else decorate(func)


def psyre_sequ(
    inst_func: Callable[..., Any] | None = None,
    *,
    prepare: PrepareFn | None = None,
    nout: int | None = None,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    static_literals: bool = False,
) -> Any:
  """`@psyre_sequ` decorator defining a stateful node from a generator function.

  Supports both bare `@psyre_sequ` and parameterized
  `@psyre_sequ(nout=..., static_argnums=..., ...)` usage. Dynamic `Stream` and
  Stage-1 thunk leaves nested inside positional PyTree arguments (`tuple`,
  `NamedTuple`, `list`, `dict`/`Mapping`, `slice`, and types registered via
  `register_pytree_node`) are extracted into the node's `inputs` and
  reconstructed on each tick (passed to `fn(*args, **kwargs)` on start/reset
  and delivered via `gen.send(arg)` or `gen.send(args)` on later ticks).

  Because Python's `generator.send()` carries a single positional value rather
  than keyword arguments, dynamic `Stream`/thunk leaves inside keyword arguments
  are rejected with `TypeError` unless converted to positional slots via
  `prepare=` or kept verbatim via `static_argnames=`.

  Args:
    inst_func: Generator callable to wrap (omitted when called with keyword
      options as a decorator factory).
    prepare: Optional Stage-1 argument normalizer `(args, kwargs) -> (args,
      kwargs)` run before dynamic-vs.-static partitioning.
    nout: Optional positive integer output arity unpacking tuple-valued outputs
      into an `nout`-tuple of `Stream`s via pointwise indexing.
    static_argnums: Positional argument index or iterable of non-negative
      indices kept verbatim as static per-tick arguments.
    static_argnames: Keyword argument name or iterable of names kept verbatim
      as static per-tick arguments (passed when instantiating the generator).
    static_literals: When True, positional arguments containing no `Stream` or
      Stage-1 thunk stay verbatim inside the static call template instead of
      being promoted to constant/periodic `Stream`s via `resolve_stream`.

  Returns:
    A wrapped callable (or decorator) emitting a stateful Stage-1 `Node` and
    returning a `Stream` (or an `nout`-tuple of `Stream`s when `nout` is set).
  """
  if nout is not None:
    validate_positive_int(nout, 'psyre_sequ nout')
  argnums = normalize_static_argnums(static_argnums)
  argnames = normalize_static_argnames(static_argnames)

  def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
    _require_callable_target(fn, 'psyre_sequ')

    def step_func(
        o_stream: _StreamPort, *args: _StreamPort, **kwargs: Any
    ) -> Any:
      return _sequ(fn, o_stream, *args, **kwargs)

    step_func.__name__ = callable_name(fn)
    setattr(step_func, '_psyre_stateful', True)
    setattr(step_func, '_psyre_is_sequ', True)
    return _wrap_partitioned_step(
        step_func,
        prepare=prepare,
        argnums=argnums,
        argnames=argnames,
        static_literals=static_literals,
        nout=nout,
        decorator_name='psyre_sequ',
        reject_dynamic_kwargs=True,
    )

  return decorate if inst_func is None else decorate(inst_func)


def psyre_node(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """@psyre_node(name) decorator wrapping a Stage-1 function in a hierarchy."""
  if not isinstance(name, str) or not name:
    raise TypeError('psyre_node(name) requires a non-empty str argument.')

  def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      require_builder()
      scope_name = kwargs.pop('name', name)
      if not isinstance(scope_name, str) or not scope_name:
        raise TypeError('psyre_node name must be a non-empty str.')
      with push_prefix(scope_name):
        return fn(*args, **kwargs)

    return wrapper

  return decorator


def make_primitive(name: str, *, doc: str = '') -> Callable[..., Any]:
  """Creates a Stage-1 primitive dispatcher for `name`."""

  def dispatch(*args, **kwargs) -> Any:
    sem = require_builder().semantics or Semantics.SYNC
    prim = primitives_registry.get(sem, {}).get(name)
    if prim is None:
      raise RuntimeError(
          f'{name} is not supported under {sem.name} semantics.'
      )
    return prim(*args, **kwargs)

  dispatch.__name__ = name
  if doc:
    dispatch.__doc__ = doc
  return dispatch


buffer = make_primitive(
    'buffer',
    doc=(
        'Bounded FIFO buffer `buffer(i, pull_p, push_p, size=1)` with '
        'pull-before-push ordering; raises `BufferError` on '
        'overflow/underflow.'
    ),
)
counter = make_primitive(
    'counter',
    doc=(
        'Resettable integer counter `counter(start=0, max_val=None, '
        'tick=True)` incrementing on ticks where `tick` is truthy.'
    ),
)
current = make_primitive(
    'current',
    doc=(
        'Sample-and-hold latch `current(b, i, default=None)` projecting a '
        "subclocked or equal-clock stream `i` onto `b`'s clock."
    ),
)
fby = make_primitive(
    'fby',
    doc=(
        'Followed-by delay `fby(init, x, n=1)`: emits `init` for `n` active '
        'ticks after start/reset, then `x` delayed by `n` ticks.'
    ),
)
ifthenelse = make_primitive(
    'ifthenelse',
    doc=(
        'Pointwise strictly scalar multiplexer `ifthenelse(c, t, f)` '
        'evaluating `t if c else f` (use `np.where`/`jnp.where` for arrays).'
    ),
)
merge = make_primitive(
    'merge',
    doc=(
        'Combines complementary subclocked streams `merge(c, t, f)` or '
        '`merge(sel, {tag: arm, ...}, default=...)` without dropping tokens.'
    ),
)
morph = make_primitive(
    'morph',
    doc=(
        'Initialization operator `morph(i, j, n=1)` (Lustre `->`): emits the '
        'first `n` active ticks of `i` after start/reset, then `j`.'
    ),
)
post = make_primitive(
    'post',
    doc=(
        'Lookahead operator `post(i, n=1)`: drops `n` active ticks and emits '
        'future values `i[t + n]` (Potop-Butucaru et al., arXiv:2311.16977).'
    ),
)
pre = make_primitive(
    'pre',
    doc=(
        'Unit delay `pre(x, n=1)` initialized to `None` for the first `n` '
        'active ticks after start/reset.'
    ),
)
prefix_sum = make_primitive(
    'prefix_sum',
    doc=(
        'Resettable cumulative sum `prefix_sum(data, start=0)` accumulating '
        'elements of `data` starting from `start`.'
    ),
)
scan = make_primitive(
    'scan',
    doc=(
        'Cumulative binary fold `scan(data, op=operator.add)` initialized '
        'with the first element of `data`.'
    ),
)
switch = make_primitive(
    'switch',
    doc=(
        'Same-clock selector `switch(c, t, f)` or '
        '`switch(sel, {tag: arm, ...}, default=...)` pulling all arms.'
    ),
)
when = make_primitive(
    'when',
    doc=(
        'Downsamples `when(c, i, *, values=..., negated=False)` keeping '
        'active ticks where `c` is truthy (or matches `values`).'
    ),
)
whennot = make_primitive(
    'whennot',
    doc='Downsamples `whennot(c, i)` keeping active ticks where `c` is falsy.',
)


def emit_when(
    sel: Any,
    val: Any,
    selector: WhenSelector | None = None,
    *,
    name: str = '',
) -> Any:
  """Emits a `when` or `whennot` node matching `selector` (boolean or tagged)."""
  kwargs = {'name': name} if name else {}
  if selector is None:
    return when(sel, val, **kwargs)
  if selector.is_boolean:
    if selector.negated:
      return whennot(sel, val, **kwargs)
    return when(sel, val, **kwargs)
  return when(sel, val, selector=selector, **kwargs)
