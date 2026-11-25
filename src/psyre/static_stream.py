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
"""Symbolic finite words and eventually periodic static stream sequences.

Finite-word grammar: `S ::= w | S * k | S1 + S2` (`k >= 0` integer, total word
length bounded by `MAX_WORD_LENGTH = 2**31 - 1`).
Infinite sequences (`StaticSequence`): `prefix + period*ω` where both `prefix`
and the non-empty `period` are `StaticWord` expressions evaluated symbolically
without expanding `S * k` repetitions (`O(1)` for `once(k)` impulses).
`StaticWord.__eq__` and `StaticSequence.__eq__` compare normalized ASTs
structurally in `O(|DAG|)` time.
"""

from __future__ import annotations

import bisect
import collections
from collections.abc import (
    AsyncGenerator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
import itertools
from typing import Any, overload

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    Absent,
    UNSET,
    format_tag,
    reject_sentinel,
    validate_non_negative_int,
)
from psyre.payload import (
    exact_value_key,
    is_array_object,
    is_lazy_thunk_obj,
    is_nan_scalar,
    is_stream_obj,
    reject_callable,
    safe_scalar_eq,
    zero_sign_key,
)

# pylint: enable=g-importing-member,g-multiple-import

_STATIC_STREAM_STEP_ATTR = '_psyre_static_stream'
MAX_WORD_LENGTH = (1 << 31) - 1

_MUTABLE_CONTAINERS = (list, set, Mapping, bytearray, collections.deque)


def _format_atom_str(value: Any) -> str:
  """Formats an atom without invoking `str()`/`repr()` on dense arrays."""
  return format_tag(value) if is_array_object(value) else str(value)


def _safe_eq(av: Any, bv: Any) -> bool:
  """Safe `==` guarding against `Stream`/`LazyThunk`/array side effects."""
  return av is bv or safe_scalar_eq(av, bv)


def same_const_value(av: Any, bv: Any) -> bool:
  """Returns True when `av` and `bv` share exact type, value, and zero sign.

  Mutable containers (`list`, `set`, `dict`/`Mapping`, `bytearray`, `deque`)
  and non-0-D, non-NumPy, or object-dtype arrays compare by object identity
  (`is`) so distinct mutable references are never folded or aliased together.
  NaN scalars always compare unequal.

  Args:
    av: First constant payload value.
    bv: Second constant payload value.

  Returns:
    True iff `av` and `bv` are considered identical static constants.
  """
  if is_nan_scalar(av) or type(av) is not type(bv):
    return False
  if isinstance(av, tuple):
    return len(av) == len(bv) and all(
        same_const_value(x, y) for x, y in zip(av, bv)
    )
  if isinstance(av, frozenset):
    if len(av) != len(bv):
      return False
    ka, kb = exact_value_key(av), exact_value_key(bv)
    if ka is None or kb is None:
      return False
    try:
      return bool(ka == kb)
    except (TypeError, ValueError, RuntimeError):
      return False
  if is_array_object(av):
    if av.shape != bv.shape or av.dtype != bv.dtype:
      return False
    if (
        not av.shape
        and type(av).__module__ == 'numpy'
        and getattr(av.dtype, 'kind', None) != 'O'
    ):
      return same_const_value(av.item(), bv.item())
    return av is bv
  if isinstance(av, _MUTABLE_CONTAINERS):
    return av is bv
  return _safe_eq(av, bv) and zero_sign_key(av) == zero_sign_key(bv)


def _validate_atom(value: Any, label: str) -> None:
  """Rejects sentinels, `LazyThunk`s, callables, and nested `Stream`s."""
  reject_sentinel(value, label=label)
  reject_callable(value, label=label, reject_streams=True)


def _validate_items(items: Iterable[Any], label: str) -> tuple[Any, ...]:
  """Copies and validates every element of a literal static word."""
  copied = tuple(items)
  for item in copied:
    _validate_atom(item, label)
  return copied


def _check_int(value: Any, what: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f'{what} must be an int, not {type(value).__name__}.')
  return value


def _clamp_bound(bound: int | None, length: int, default: int) -> int:
  """Normalizes an endpoint for an ascending `step == 1` slice."""
  if bound is None:
    return default
  pos = _check_int(bound, 'Slice index')
  return max(0, min(length, pos + length if pos < 0 else pos))


def _is_stream_or_thunk(value: Any) -> bool:
  return is_stream_obj(value) or is_lazy_thunk_obj(value)


def _struct_eq(
    a: StaticWord, b: StaticWord, memo: set[tuple[int, int]] | None = None
) -> bool:
  """True when two `StaticWord` ASTs have identical structure and atoms."""
  # Fast identity check short-circuits identical AST nodes (consistent with
  # Python container equality, a shared `StaticWord` compares equal to itself
  # even if it carries a NaN atom). `memo` avoids O(2^depth) walks on distinct
  # shared DAG nodes.
  if a is b:
    return True
  if type(a) is not type(b) or a.length != b.length:
    return False
  key = (id(a), id(b))
  if memo is not None and key in memo:
    return True
  if isinstance(a, LiteralWord):
    assert isinstance(b, LiteralWord)
    ok = all(same_const_value(x, y) for x, y in zip(a.items, b.items))
  elif isinstance(a, RepeatWord):
    assert isinstance(b, RepeatWord)
    if memo is None:
      memo = set()
    ok = a.count_n == b.count_n and _struct_eq(a.inner, b.inner, memo)
  else:
    assert isinstance(a, ConcatWord) and isinstance(b, ConcatWord)
    if len(a.parts) != len(b.parts):
      return False
    if memo is None:
      memo = set()
    ok = all(_struct_eq(x, y, memo) for x, y in zip(a.parts, b.parts))
  if ok and memo is not None:
    memo.add(key)
  return ok


class StaticWord:
  """Immutable finite word in the grammar `S ::= w | S * k | S1 + S2`.

  Concrete variants are `LiteralWord` (`w`), `RepeatWord` (`S * k`, `k >= 2`),
  and `ConcatWord` (`S1 + S2 + ...`, flattened). `StaticWord(v)` constructs a
  canonical word without unpacking non-`list` values (use `StaticWord.atom(v)`
  to keep a `list` as a single element).
  """

  __slots__ = ()
  __hash__ = None  # type: ignore[assignment]
  length: int

  def __new__(cls, value: Any = UNSET) -> StaticWord:
    if cls is not StaticWord:
      return super().__new__(cls)
    return EMPTY_WORD if value is UNSET else _as_word(value)

  @staticmethod
  def atom(value: Any, *, label: str = 'Static word') -> StaticWord:
    """Wraps a single value into a length-1 word without unpacking `list`s."""
    _validate_atom(value, label)
    return LiteralWord((value,))

  def __len__(self) -> int:
    return self.length

  def __bool__(self) -> bool:
    return self.length > 0

  def value_at(self, index: int) -> Any:
    """Returns the element at `0 <= index < self.length` without expansion."""
    raise NotImplementedError

  def __iter__(self) -> Iterator[Any]:
    raise NotImplementedError

  def _slice_nonempty(self, start: int, stop: int) -> StaticWord:
    raise NotImplementedError

  def slice_word(self, start: int, stop: int) -> StaticWord:
    """Returns the symbolic sub-word for `0 <= start <= stop <= self.length`."""
    if start >= stop:
      return EMPTY_WORD
    if start == 0 and stop == self.length:
      return self
    return self._slice_nonempty(start, stop)

  def format_word(
      self,
      *,
      item_fmt: Callable[[Any], str] = _format_atom_str,
      max_items: int | None = None,
      parenthesize_concat: bool = False,
  ) -> str:
    """Formats this word expression compactly without expanding `S * k`."""
    raise NotImplementedError

  def iter_atoms(self) -> Iterator[Any]:
    """Yields leaf elements across the AST DAG in `O(|DAG|)` time."""
    seen: set[int] = set()
    stack: list[StaticWord] = [self]
    while stack:
      node = stack.pop()
      nid = id(node)
      if nid in seen:
        continue
      seen.add(nid)
      if isinstance(node, LiteralWord):
        yield from node.items
      elif isinstance(node, RepeatWord):
        stack.append(node.inner)
      else:
        assert isinstance(node, ConcatWord)
        stack.extend(reversed(node.parts))

  @overload
  def __getitem__(self, index: int) -> Any: ...

  @overload
  def __getitem__(self, index: slice) -> StaticWord: ...

  def __getitem__(self, index: int | slice) -> Any:
    n = self.length
    if isinstance(index, slice):
      if index.step is not None and _check_int(index.step, 'Slice step') != 1:
        raise ValueError(
            f'StaticWord slicing only supports step == 1, got {index.step!r}.'
        )
      lo = _clamp_bound(index.start, n, 0)
      hi = _clamp_bound(index.stop, n, n)
      return self.slice_word(lo, hi)
    idx = _check_int(index, 'StaticWord index')
    pos = idx + n if idx < 0 else idx
    if not 0 <= pos < n:
      raise IndexError(f'StaticWord index {index} out of range [0, {n}).')
    return self.value_at(pos)

  def uniform_value(self) -> tuple[bool, Any]:
    """Returns `(True, v)` iff non-empty and every atom shares exact value."""
    if self.length == 0:
      return False, None
    atoms = self.iter_atoms()
    first = next(atoms)
    for atom in atoms:
      if not same_const_value(first, atom):
        return False, None
    return True, first

  def __contains__(self, value: Any) -> bool:
    return any(_safe_eq(atom, value) for atom in self.iter_atoms())

  def __mul__(self, count: Any) -> StaticWord:
    return NotImplemented if _is_stream_or_thunk(count) else repeat(self, count)

  def __rmul__(self, count: Any) -> StaticWord:
    return NotImplemented if _is_stream_or_thunk(count) else repeat(self, count)

  def __add__(self, other: Any) -> StaticWord:
    return NotImplemented if _is_stream_or_thunk(other) else concat(self, other)

  def __radd__(self, other: Any) -> StaticWord:
    return NotImplemented if _is_stream_or_thunk(other) else concat(other, self)

  def __eq__(self, other: Any) -> bool:
    """Structural equality over normalized `StaticWord` ASTs (`O(|DAG|)`)."""
    if not isinstance(other, StaticWord):
      return NotImplemented
    return _struct_eq(self, other)

  def __str__(self) -> str:
    return self.format_word()

  def __repr__(self) -> str:
    return f'StaticWord({self.format_word(item_fmt=repr)})'


class LiteralWord(StaticWord):
  """Finite literal word `w` storing an immutable tuple of elements."""

  __slots__ = ('items', 'length')
  items: tuple[Any, ...]

  def __new__(cls, items: tuple[Any, ...]) -> LiteralWord:
    obj = super(StaticWord, cls).__new__(cls)
    obj.items = items
    obj.length = len(items)
    return obj

  def value_at(self, index: int) -> Any:
    return self.items[index]

  def __iter__(self) -> Iterator[Any]:
    yield from self.items

  def _slice_nonempty(self, start: int, stop: int) -> StaticWord:
    return LiteralWord(self.items[start:stop])

  def format_word(
      self,
      *,
      item_fmt: Callable[[Any], str] = _format_atom_str,
      max_items: int | None = None,
      parenthesize_concat: bool = False,
  ) -> str:
    del parenthesize_concat
    shown = self.items[:max_items] if max_items is not None else self.items
    body = ', '.join(item_fmt(v) for v in shown)
    if max_items is not None and len(self.items) > max_items:
      return f'[{body}, …]'
    return f'[{body}]'


EMPTY_WORD: StaticWord = LiteralWord(())


class RepeatWord(StaticWord):
  """Repeated word `S * k` storing `inner` and integer count `k >= 2`."""

  __slots__ = ('inner', 'count_n', 'length')
  inner: StaticWord
  count_n: int

  def __new__(cls, inner: StaticWord, count: int) -> RepeatWord:
    obj = super(StaticWord, cls).__new__(cls)
    obj.inner = inner
    obj.count_n = count
    obj.length = inner.length * count
    return obj

  def value_at(self, index: int) -> Any:
    return self.inner.value_at(index % self.inner.length)

  def __iter__(self) -> Iterator[Any]:
    if isinstance(self.inner, LiteralWord) and len(self.inner.items) == 1:
      yield from itertools.repeat(self.inner.items[0], self.count_n)
      return
    for _ in range(self.count_n):
      yield from self.inner

  def _slice_nonempty(self, start: int, stop: int) -> StaticWord:
    p = self.inner.length
    q0, r0 = divmod(start, p)
    q1, r1 = divmod(stop, p)
    if q0 == q1:
      return self.inner.slice_word(r0, r1)
    head = (self.inner.slice_word(r0, p),) if r0 > 0 else ()
    full = q1 - q0 - (1 if r0 > 0 else 0)
    mid = (repeat(self.inner, full),) if full > 0 else ()
    tail = (self.inner.slice_word(0, r1),) if r1 > 0 else ()
    return concat(*head, *mid, *tail)

  def format_word(
      self,
      *,
      item_fmt: Callable[[Any], str] = _format_atom_str,
      max_items: int | None = None,
      parenthesize_concat: bool = False,
  ) -> str:
    del parenthesize_concat
    inner = self.inner.format_word(
        item_fmt=item_fmt, max_items=max_items, parenthesize_concat=True
    )
    return f'{inner}*{self.count_n}'


class ConcatWord(StaticWord):
  """Flattened concatenation `S1 + S2 + ...` with binary-search offsets."""

  __slots__ = ('parts', 'offsets', 'length')
  parts: tuple[StaticWord, ...]
  offsets: tuple[int, ...]

  def __new__(cls, parts: tuple[StaticWord, ...]) -> ConcatWord:
    obj = super(StaticWord, cls).__new__(cls)
    obj.parts = parts
    obj.offsets = tuple(itertools.accumulate(p.length for p in parts))
    obj.length = obj.offsets[-1]
    return obj

  def value_at(self, index: int) -> Any:
    slot = bisect.bisect_right(self.offsets, index)
    base = 0 if slot == 0 else self.offsets[slot - 1]
    return self.parts[slot].value_at(index - base)

  def __iter__(self) -> Iterator[Any]:
    for part in self.parts:
      yield from part

  def _slice_nonempty(self, start: int, stop: int) -> StaticWord:
    first = bisect.bisect_right(self.offsets, start)
    last = bisect.bisect_right(self.offsets, stop - 1)
    sliced: list[StaticWord] = []
    for slot in range(first, last + 1):
      base = 0 if slot == 0 else self.offsets[slot - 1]
      part = self.parts[slot]
      sliced.append(
          part.slice_word(max(0, start - base), min(part.length, stop - base))
      )
    return concat(*sliced)

  def format_word(
      self,
      *,
      item_fmt: Callable[[Any], str] = _format_atom_str,
      max_items: int | None = None,
      parenthesize_concat: bool = False,
  ) -> str:
    shown_p = self.parts[:max_items] if max_items is not None else self.parts
    body = '+'.join(
        p.format_word(item_fmt=item_fmt, max_items=max_items) for p in shown_p
    )
    if max_items is not None and len(self.parts) > max_items:
      body = f'{body}+…'
    return f'({body})' if parenthesize_concat else body


def _as_word(value: Any, *, label: str = 'Static word') -> StaticWord:
  """Converts a `StaticWord`, `list`, or scalar atom to a `StaticWord`."""
  if isinstance(value, StaticWord):
    return value
  if isinstance(value, list):
    items = _validate_items(value, label)
    return LiteralWord(items) if items else EMPTY_WORD
  _validate_atom(value, label)
  return LiteralWord((value,))


def repeat(
    value: Any, count: int, *, label: str = 'Static word'
) -> StaticWord:
  """Builds `S * count` symbolically without expanding `count` copies."""
  validate_non_negative_int(count, 'StaticWord repetition count')
  word = _as_word(value, label=label)
  if count == 0 or word.length == 0:
    return EMPTY_WORD
  if count == 1:
    return word
  if word.length * count > MAX_WORD_LENGTH:
    raise ValueError(
        f'StaticWord length exceeds limit {MAX_WORD_LENGTH} '
        f'(got {word.length * count}).'
    )
  if isinstance(word, RepeatWord):
    return RepeatWord(word.inner, word.count_n * count)
  return RepeatWord(word, count)


def _single_atom_repeat(
    word: StaticWord,
) -> tuple[LiteralWord, Any, int] | None:
  """Returns `(inner, atom, count)` for a single-atom `RepeatWord`."""
  if (
      isinstance(word, RepeatWord)
      and isinstance(word.inner, LiteralWord)
      and len(word.inner.items) == 1
  ):
    return word.inner, word.inner.items[0], word.count_n
  return None


def _append_part(flat: list[StaticWord], part: StaticWord) -> None:
  """Appends `part`, fusing adjacent literals, boundary runs, and repeats."""
  if not flat:
    flat.append(part)
    return
  prev = flat[-1]
  if isinstance(prev, LiteralWord) and isinstance(part, LiteralWord):
    flat[-1] = LiteralWord(prev.items + part.items)
    return
  if (
      isinstance(part, LiteralWord)
      and (rep := _single_atom_repeat(prev)) is not None
  ):
    inner, atom, cnt = rep
    items = part.items
    i = 0
    while i < len(items) and same_const_value(items[i], atom):
      i += 1
    if i:
      flat[-1] = repeat(inner, cnt + i)
      if i < len(items):
        flat.append(LiteralWord(items[i:]))
      return
  if (
      isinstance(prev, LiteralWord)
      and (rep := _single_atom_repeat(part)) is not None
  ):
    inner, atom, cnt = rep
    items = prev.items
    i = len(items)
    while i > 0 and same_const_value(items[i - 1], atom):
      i -= 1
    if i < len(items):
      merged = repeat(inner, cnt + (len(items) - i))
      if i == 0:
        flat[-1] = merged
      else:
        flat[-1] = LiteralWord(items[:i])
        flat.append(merged)
      return
  if (
      isinstance(prev, RepeatWord)
      and isinstance(part, RepeatWord)
      and _struct_eq(prev.inner, part.inner)
  ):
    flat[-1] = repeat(prev.inner, prev.count_n + part.count_n)
    return
  flat.append(part)


def concat(*parts: Any, label: str = 'Static word') -> StaticWord:
  """Flattens and fuses zero or more finite word expressions into one."""
  flat: list[StaticWord] = []
  for raw in parts:
    word = _as_word(raw, label=label)
    for part in word.parts if isinstance(word, ConcatWord) else (word,):
      if part.length > 0:
        _append_part(flat, part)
  if not flat:
    return EMPTY_WORD
  total = sum(part.length for part in flat)
  if total > MAX_WORD_LENGTH:
    raise ValueError(
        f'StaticWord length exceeds limit {MAX_WORD_LENGTH} (got {total}).'
    )
  return flat[0] if len(flat) == 1 else ConcatWord(tuple(flat))


def _trim_trailing(
    word: StaticWord,
    target: Any,
    memo: dict[int, StaticWord] | None = None,
) -> StaticWord:
  """Removes trailing copies matching `target` via `same_const_value`."""
  if isinstance(word, LiteralWord):
    end = len(word.items)
    while end > 0 and same_const_value(word.items[end - 1], target):
      end -= 1
    if end == len(word.items):
      return word
    return LiteralWord(word.items[:end]) if end > 0 else EMPTY_WORD
  wid = id(word)
  if memo is not None and wid in memo:
    return memo[wid]
  if memo is None:
    memo = {}
  if isinstance(word, RepeatWord):
    trimmed = _trim_trailing(word.inner, target, memo)
    if trimmed.length == 0:
      res = EMPTY_WORD
    elif trimmed.length == word.inner.length:
      res = word
    else:
      res = concat(repeat(word.inner, word.count_n - 1), trimmed)
  else:
    assert isinstance(word, ConcatWord)
    parts = list(word.parts)
    while parts:
      trimmed = _trim_trailing(parts[-1], target, memo)
      if trimmed.length == 0:
        parts.pop()
      else:
        parts[-1] = trimmed
        break
    res = concat(*parts)
  memo[wid] = res
  return res


class StaticSequence:
  """Eventually periodic infinite sequence `prefix + period*ω`."""

  __hash__ = None  # type: ignore[assignment]
  __slots__ = ('prefix', 'period', '_is_one_periodic', '_const_value')

  prefix: StaticWord
  period: StaticWord
  _is_one_periodic: bool
  _const_value: Any

  def __init__(self, prefix: StaticWord, period: StaticWord):
    if not isinstance(prefix, StaticWord) or not isinstance(period, StaticWord):
      raise TypeError(
          'StaticSequence expects StaticWord prefix and period (use '
          'StaticSequence.from_parts for raw values).'
      )
    if period.length == 0:
      raise ValueError('Static periodic stream requires a non-empty period.')
    while isinstance(period, RepeatWord):
      period = period.inner
    is_unif, val = period.uniform_value()
    if is_unif and period.length > 1:
      period = LiteralWord((val,))
    if period.length == 1:
      prefix = _trim_trailing(prefix, period.value_at(0))
    self.prefix = prefix
    self.period = period
    self._is_one_periodic = prefix.length == 0 and period.length == 1
    self._const_value = period.value_at(0) if self._is_one_periodic else UNSET

  @classmethod
  def from_parts(cls, prefix: Any, period: Any) -> StaticSequence:
    """Builds `prefix + period*ω` validating both words."""
    # Validate `period` before `prefix` so `Stream(existing, period=thunk)`
    # reports the unresolved `LazyThunk` before rejecting the `Stream` prefix.
    per = _as_word(period, label='Static stream period')
    pfx = (
        EMPTY_WORD
        if prefix is UNSET
        else _as_word(prefix, label='Static stream prefix')
    )
    return cls(pfx, per)

  @classmethod
  def from_value(cls, value: Any) -> StaticSequence:
    """Builds a `StaticSequence` from sequence, `StaticWord`, `list`, or atom."""
    if isinstance(value, StaticSequence):
      return value
    if isinstance(value, StaticWord):
      n = value.length
      if n == 0:
        raise ValueError('Static stream requires at least one value.')
      last = LiteralWord((value.value_at(n - 1),))
      return cls(value.slice_word(0, n - 1), last)
    if isinstance(value, list):
      if not value:
        raise ValueError('Static stream requires at least one value.')
      pfx_items = _validate_items(value[:-1], 'Static stream prefix')
      last_val = value[-1]
      _validate_atom(last_val, 'Static stream period')
      pfx = LiteralWord(pfx_items) if pfx_items else EMPTY_WORD
      return cls(pfx, LiteralWord((last_val,)))
    _validate_atom(value, 'Static stream period')
    return cls(EMPTY_WORD, LiteralWord((value,)))

  @classmethod
  def constant(cls, value: Any) -> StaticSequence:
    """Builds a 1-periodic constant sequence `[value]*ω` without unpacking."""
    _validate_atom(value, 'Static stream period')
    return cls(EMPTY_WORD, LiteralWord((value,)))

  @classmethod
  def once(cls, k: int) -> StaticSequence:
    """Builds an `O(1)` impulse sequence emitting `True` only at index `k`."""
    validate_non_negative_int(k, 'Stream.once index')
    return cls(
        concat(repeat(False, k), LiteralWord((True,))), LiteralWord((False,))
    )

  @property
  def is_one_periodic(self) -> bool:
    """True when this sequence repeats a single constant value (`[v]*ω`)."""
    return self._is_one_periodic

  @property
  def const_value(self) -> Any:
    """Returns the unique repeated value of a 1-periodic `StaticSequence`."""
    if not self._is_one_periodic:
      raise RuntimeError('const_value requires a 1-periodic StaticSequence.')
    return self._const_value

  def same_constant(self, other: StaticSequence) -> bool:
    """True iff both sequences are 1-periodic with equal constant values."""
    return (
        self.is_one_periodic
        and other.is_one_periodic
        and same_const_value(self.const_value, other.const_value)
    )

  def value_at(self, index: int) -> Any:
    """Returns the value at non-negative `index` in `O(depth)` time."""
    if index < 0:
      raise ValueError(f'Stream index must be non-negative, got {index}.')
    if self._is_one_periodic:
      return self._const_value
    pfx_len = self.prefix.length
    if index < pfx_len:
      return self.prefix.value_at(index)
    return self.period.value_at((index - pfx_len) % self.period.length)

  def iter_values(self) -> Iterator[Any]:
    """Returns an infinite `O(1)`-amortized iterator over the sequence."""
    if self.prefix.length > 0:
      yield from self.prefix
    if isinstance(self.period, LiteralWord) and len(self.period.items) == 1:
      yield from itertools.repeat(self.period.items[0])
      return
    while True:
      yield from self.period

  def trace(self, horizon: int) -> list[Any]:
    """Materializes the first `horizon` elements as a Python `list`."""
    validate_non_negative_int(horizon, 'horizon')
    if self._is_one_periodic:
      return [self._const_value] * horizon
    return list(itertools.islice(self.iter_values(), horizon))

  def iter_atoms(self) -> Iterator[Any]:
    """Yields leaf elements across `prefix` and `period` in `O(|DAG|)` time."""
    yield from self.prefix.iter_atoms()
    yield from self.period.iter_atoms()

  def is_bool(self) -> bool:
    """Returns True when every element is a Python `bool` (`O(|DAG|)`)."""
    return all(isinstance(v, bool) for v in self.iter_atoms())

  def _format(
      self,
      *,
      item_fmt: Callable[[Any], str] = _format_atom_str,
      max_items: int | None = None,
  ) -> str:
    """Formats `prefix + period*ω` using `item_fmt` and optional `max_items`."""
    pfx = ''
    if self.prefix.length > 0:
      pfx_str = self.prefix.format_word(
          item_fmt=item_fmt, max_items=max_items
      )
      pfx = f'{pfx_str}+'
    per = self.period.format_word(
        item_fmt=item_fmt, max_items=max_items, parenthesize_concat=True
    )
    return f'{pfx}{per}*ω'

  def format_compact(
      self, *, max_tag_len: int = 10, max_items: int = 4
  ) -> str:
    """Formats this sequence concisely for schematic badges and tooltips."""
    if self._is_one_periodic:
      return format_tag(self._const_value, 24)
    return self._format(
        item_fmt=lambda v: format_tag(v, max_tag_len), max_items=max_items
    )

  def __str__(self) -> str:
    return self._format()

  def __repr__(self) -> str:
    return f'StaticSequence({self.prefix!r}, {self.period!r})'

  def __eq__(self, other: Any) -> bool:
    """Structural equality over normalized `(prefix, period)`."""
    if not isinstance(other, StaticSequence):
      return NotImplemented
    return self.prefix == other.prefix and self.period == other.period


def is_static_stream_step(step_func: Any) -> bool:
  """Returns True when `step_func` is the builtin `static_stream` coroutine."""
  return bool(getattr(step_func, _STATIC_STREAM_STEP_ATTR, False))


async def static_stream(
    o_stream: Any,
    *,
    source: Any,
) -> AsyncGenerator[None, Any]:
  """Emits `source`'s dense static values on active ticks, rewinding on reset."""
  spec = source.static_sequence
  assert spec is not None
  cursor: Iterator[Any] | None = None
  while True:
    start = yield
    if start is Absent:
      o_stream.push(Absent)
    else:
      if start or cursor is None:
        cursor = spec.iter_values()
      o_stream.push(next(cursor))


setattr(static_stream, _STATIC_STREAM_STEP_ATTR, True)
