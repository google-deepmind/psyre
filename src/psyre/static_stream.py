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
without expanding `S * k` repetitions (`O(1)` for `once(k)` impulses). Symbolic
word equality (`StaticWord.__eq__`) uses deterministic Straight-Line Program
recompression (Jeż, *J. ACM* 2016) so equality stays polynomial in DAG size even
for nested repetitions up to `MAX_WORD_LENGTH`.
"""

from __future__ import annotations

import bisect
from collections.abc import (
    AsyncGenerator,
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
import enum
import itertools
import math
from typing import Any, overload

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    Absent,
    LazyThunk,
    UNSET,
    format_tag,
    reject_sentinel,
    validate_non_negative_int,
)
from psyre.common_payload import (
    contains_stream_or_thunk,
    is_array_object,
    is_nan_scalar,
    is_numpy_scalar,
    is_stream_obj,
    reject_callable,
    safe_scalar_eq,
)

# pylint: enable=g-importing-member,g-multiple-import

_STATIC_STREAM_STEP_ATTR = '_psyre_static_stream'
MAX_WORD_LENGTH = (1 << 31) - 1


def _format_atom_str(value: Any) -> str:
  """Formats an atom without invoking `str()`/`repr()` on dense arrays."""
  return format_tag(value) if is_array_object(value) else str(value)


def _safe_eq(av: Any, bv: Any) -> bool:
  """Safe `==` guarding against `Stream`/`LazyThunk`/array side effects."""
  return av is bv or safe_scalar_eq(av, bv)


def _same_float_zero_sign(av: float, bv: float) -> bool:
  """Distinguishes IEEE-754 `-0.0` from `+0.0`."""
  return av != 0.0 or math.copysign(1.0, av) == math.copysign(1.0, bv)


def same_const_value(av: Any, bv: Any) -> bool:
  """Returns True when `av` and `bv` share exact type, value, and zero sign."""
  # Value equality (used by `StaticWord.__eq__` and constant dedup). Contrast
  # with `_same_atom`, which keeps identity (`is`) for mutable payloads so
  # distinct mutable references are never folded or aliased together.
  if is_nan_scalar(av):
    return False
  if type(av) is not type(bv):
    return False
  if isinstance(av, tuple):
    return len(av) == len(bv) and all(
        same_const_value(x, y) for x, y in zip(av, bv)
    )
  if not _safe_eq(av, bv):
    return False
  if is_numpy_scalar(av):
    av, bv = av.item(), bv.item()
  if isinstance(av, float):
    return _same_float_zero_sign(av, bv)
  if isinstance(av, complex):
    return _same_float_zero_sign(av.real, bv.real) and _same_float_zero_sign(
        av.imag, bv.imag
    )
  return True


def _is_immutable(value: Any) -> bool:
  """Returns True when `value` is deeply immutable for safe folding."""
  if (
      value is None
      or isinstance(
          value, (bool, int, float, complex, str, bytes, enum.Enum)
      )
      or is_numpy_scalar(value)
  ):
    return True
  if type(value) is tuple:  # pylint: disable=unidiomatic-typecheck
    return all(_is_immutable(item) for item in value)
  return False


def _same_atom(av: Any, bv: Any) -> bool:
  """Identity-safe equality preserving distinct mutable object references."""
  if is_nan_scalar(av):
    return False
  return av is bv or (_is_immutable(av) and same_const_value(av, bv))


def _validate_atom(value: Any, label: str) -> None:
  """Rejects sentinels, `LazyThunk`s, callables, and nested `Stream`s."""
  reject_sentinel(value, label=label)
  # Check callables/`LazyThunk` before nested `Stream`s so unresolved thunks
  # always surface their dedicated diagnostic message.
  reject_callable(value, label=label)
  if contains_stream_or_thunk(value):
    raise TypeError(
        f'{label} cannot contain a Stream instance; Psyre streams are '
        'first-order.'
    )


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
    ok = all(_same_atom(x, y) for x, y in zip(a.items, b.items))
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


_HASHABLE_EXACT_SCALARS = frozenset({
    type(None),
    bool,
    int,
    str,
    bytes,
})


def _compile_word_slp(
    a: StaticWord, b: StaticWord
) -> tuple[int, int, list[list[int]], int]:
  """Compiles `a` and `b` into one acyclic Straight-Line Program (SLP).

  Positive ints are terminal alphabet IDs (modulo `same_const_value`); negative
  ints `~nt` reference earlier non-terminal rules (`rules[nt]`) in strict
  topological order. Two dedicated in-degree-0 sink rules hold the roots for
  `a` and `b`.

  Args:
    a: First non-empty `StaticWord`.
    b: Second non-empty `StaticWord`.

  Returns:
    Tuple `(ra_root, rb_root, rules, next_sym)`.
  """
  next_sym = 1
  scalar_ids: dict[Any, int] = {}
  buckets: dict[type[Any], list[tuple[Any, int]]] = {}

  def atom_id(val: Any) -> int:
    nonlocal next_sym

    def intern_scalar(key: Any) -> int:
      nonlocal next_sym
      sid = scalar_ids.get(key)
      if sid is None:
        sid = next_sym
        next_sym += 1
        scalar_ids[key] = sid
      return sid

    cls = type(val)  # pylint: disable=unidiomatic-typecheck
    if cls in _HASHABLE_EXACT_SCALARS:
      return intern_scalar((cls, val))
    if cls is float and not math.isnan(val):
      return intern_scalar(
          (cls, val, math.copysign(1.0, val) if val == 0.0 else 1.0)
      )
    if cls is complex and not (math.isnan(val.real) or math.isnan(val.imag)):
      return intern_scalar((
          cls,
          val,
          math.copysign(1.0, val.real) if val.real == 0.0 else 1.0,
          math.copysign(1.0, val.imag) if val.imag == 0.0 else 1.0,
      ))
    if not same_const_value(val, val):
      # Non-reflexive atoms (e.g. NaN, `(nan,)`, complex NaN) get a fresh ID
      # per `LiteralWord` slot; `compiled` memoization keeps shared leaf nodes
      # equal across both words without aliasing distinct `LiteralWord`s.
      sid = next_sym
      next_sym += 1
      return sid
    entries = buckets.setdefault(cls, [])
    for rep, sid in entries:
      if same_const_value(rep, val):
        return sid
    sid = next_sym
    next_sym += 1
    entries.append((val, sid))
    return sid

  rules: list[list[int]] = []
  compiled: dict[int, int] = {}

  def compile_node(word: StaticWord) -> int:
    wid = id(word)
    cached = compiled.get(wid)
    if cached is not None:
      return cached
    if isinstance(word, LiteralWord):
      syms = [atom_id(item) for item in word.items]
      if len(syms) == 1:
        res = syms[0]
      else:
        rules.append(syms)
        res = ~(len(rules) - 1)
    elif isinstance(word, ConcatWord):
      rules.append([compile_node(part) for part in word.parts])
      res = ~(len(rules) - 1)
    else:
      assert isinstance(word, RepeatWord)
      inner_sym = compile_node(word.inner)
      cnt = word.count_n
      pows = [inner_sym]
      bit = 1
      while (bit << 1) <= cnt:
        rules.append([pows[-1], pows[-1]])
        pows.append(~(len(rules) - 1))
        bit <<= 1
      parts = [pows[j] for j in range(len(pows)) if (cnt >> j) & 1]
      if len(parts) == 1:
        res = parts[0]
      else:
        rules.append(parts)
        res = ~(len(rules) - 1)
    compiled[wid] = res
    return res

  ref_a = compile_node(a)
  ref_b = compile_node(b)
  rules.append([ref_a])
  ra_root = len(rules) - 1
  rules.append([ref_b])
  rb_root = len(rules) - 1
  return ra_root, rb_root, rules, next_sym


def _recompress_rle_phase(
    rules: list[list[int]],
    root_set: set[int],
    next_sym: int,
) -> tuple[list[list[int]], int]:
  """Block-compresses all maximal repeated runs `c^k` across an SLP DAG."""
  n = len(rules)
  pref_rle: list[tuple[int, int] | None] = [None] * n
  suff_rle: list[tuple[int, int] | None] = [None] * n
  core_rle: list[list[Any]] = [[] for _ in range(n)]
  has_run = False

  def emit_run(buf: list[Any], sym: int, cnt: int) -> None:
    nonlocal has_run
    if buf and isinstance(buf[-1], tuple) and buf[-1][0] == sym:
      buf[-1] = (sym, buf[-1][1] + cnt)
      has_run = True
    else:
      buf.append((sym, cnt))
      if cnt > 1:
        has_run = True

  for i, r in enumerate(rules):
    if not r:
      continue
    out: list[Any] = []
    for e in r:
      if e > 0:
        emit_run(out, e, 1)
      else:
        c = ~e
        pr = pref_rle[c]
        if pr is not None:
          emit_run(out, pr[0], pr[1])
        if core_rle[c]:
          out.append(e)
        sr = suff_rle[c]
        if sr is not None:
          emit_run(out, sr[0], sr[1])
    if i not in root_set and out:
      lo, hi = 0, len(out)
      if isinstance(out[lo], tuple):
        pref_rle[i] = out[lo]
        lo += 1
      if lo < hi and isinstance(out[hi - 1], tuple):
        suff_rle[i] = out[hi - 1]
        hi -= 1
      core_rle[i] = out[lo:hi]
    else:
      core_rle[i] = out

  if not has_run:
    return (
        [[e[0] if isinstance(e, tuple) else e for e in r] for r in core_rle],
        next_sym,
    )

  sym_map: dict[tuple[int, int], int] = {}
  next_id = 1
  remapped: list[list[int]] = []
  for r in core_rle:
    nr: list[int] = []
    for e in r:
      if isinstance(e, tuple):
        sid = sym_map.get(e)
        if sid is None:
          sid = next_id
          next_id += 1
          sym_map[e] = sid
        nr.append(sid)
      else:
        nr.append(e)
    remapped.append(nr)
  return remapped, next_id


def _words_eq(a: StaticWord, b: StaticWord) -> bool:
  """Symbolically checks element-wise equality of two equal-length words."""
  # Deterministic Straight-Line Program (SLP) recompression (Jeż, J. ACM 2016):
  # alternating maximal-run (RLE) compression and non-overlapping `L -> R`
  # pair compression via a 1/2-approx greedy undirected max-cut (eliminating
  # at least 1/4 of adjacent distinct pairs per round). Runs in polynomial time
  # in DAG size and O(log L) rounds (L <= MAX_WORD_LENGTH), preventing
  # exponential blowup on nested RepeatWords with differing periods.
  if a.length == 0:
    return True
  if isinstance(a, LiteralWord) and isinstance(b, LiteralWord):
    return all(same_const_value(x, y) for x, y in zip(a.items, b.items))

  ra_root, rb_root, rules, next_sym = _compile_word_slp(a, b)
  root_set = {ra_root, rb_root}
  n = len(rules)

  while True:
    rules, next_sym = _recompress_rle_phase(rules, root_set, next_sym)
    ra, rb = rules[ra_root], rules[rb_root]
    if len(ra) == 1 and ra[0] > 0:
      return len(rb) == 1 and rb[0] == ra[0]
    if len(rb) == 1 and rb[0] > 0:
      return False

    # Compute first/last terminal per non-empty rule and exact multiplicities.
    first_sym = [0] * n
    last_sym = [0] * n
    for i, r in enumerate(rules):
      if r:
        first_sym[i] = r[0] if r[0] > 0 else first_sym[~r[0]]
        last_sym[i] = r[-1] if r[-1] > 0 else last_sym[~r[-1]]
    if (
        first_sym[ra_root] != first_sym[rb_root]
        or last_sym[ra_root] != last_sym[rb_root]
    ):
      return False

    mult = [0] * n
    mult[ra_root] = 1
    mult[rb_root] = 1
    for i in range(n - 1, -1, -1):
      m = mult[i]
      if m:
        for e in rules[i]:
          if e < 0:
            mult[~e] += m

    # Weighted adjacency of adjacent distinct terminal pairs across all rules.
    pair_w: dict[tuple[int, int], int] = {}
    undir_adj: dict[int, dict[int, int]] = {}
    active_syms: set[int] = set()
    for i, r in enumerate(rules):
      m = mult[i]
      if not m or len(r) < 2:
        continue
      for left_e, right_e in zip(r, r[1:]):
        u = left_e if left_e > 0 else last_sym[~left_e]
        v = right_e if right_e > 0 else first_sym[~right_e]
        active_syms.add(u)
        active_syms.add(v)
        pair_w[(u, v)] = pair_w.get((u, v), 0) + m
        u_map = undir_adj.setdefault(u, {})
        u_map[v] = u_map.get(v, 0) + m
        v_map = undir_adj.setdefault(v, {})
        v_map[u] = v_map.get(u, 0) + m

    # Greedy undirected 1/2 max-cut, oriented so `L -> R` carries >= 1/4 weight.
    part_a: set[int] = set()
    part_b: set[int] = set()
    for s in sorted(active_syms):
      adj = undir_adj[s]
      w_a = sum(w for nb, w in adj.items() if nb in part_a)
      w_b = sum(w for nb, w in adj.items() if nb in part_b)
      if w_a >= w_b:
        part_b.add(s)
      else:
        part_a.add(s)
    w_ab = sum(w for (u, v), w in pair_w.items() if u in part_a and v in part_b)
    w_ba = sum(w for (u, v), w in pair_w.items() if u in part_b and v in part_a)
    left_set, right_set = (part_a, part_b) if w_ab >= w_ba else (part_b, part_a)

    # Bottom-up boundary popping (`R` prefix / `L` suffix) and pair collapse.
    pop_l = [0] * n
    pop_r = [0] * n
    pair_map: dict[tuple[int, int], int] = {}
    new_rules: list[list[int]] = [[] for _ in range(n)]
    for i, r in enumerate(rules):
      if not mult[i]:
        continue
      expanded: list[int] = []
      for e in r:
        if e > 0:
          expanded.append(e)
        else:
          c = ~e
          pl = pop_l[c]
          if pl:
            expanded.append(pl)
          if new_rules[c]:
            expanded.append(e)
          pr = pop_r[c]
          if pr:
            expanded.append(pr)
      comp: list[int] = []
      j = 0
      ln = len(expanded)
      while j < ln:
        cur = expanded[j]
        if cur > 0 and cur in left_set and j + 1 < ln:
          nxt = expanded[j + 1]
          if nxt > 0 and nxt in right_set:
            key = (cur, nxt)
            pid = pair_map.get(key)
            if pid is None:
              pid = next_sym
              next_sym += 1
              pair_map[key] = pid
            comp.append(pid)
            j += 2
            continue
        comp.append(cur)
        j += 1
      if i not in root_set and comp:
        lo, hi = 0, len(comp)
        if comp[lo] > 0 and comp[lo] in right_set:
          pop_l[i] = comp[lo]
          lo += 1
        if lo < hi and comp[hi - 1] > 0 and comp[hi - 1] in left_set:
          pop_r[i] = comp[hi - 1]
          hi -= 1
        new_rules[i] = comp[lo:hi]
      else:
        new_rules[i] = comp
    rules = new_rules


class StaticWord(Sequence[Any]):
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
    return EMPTY_WORD if value is UNSET else as_word(value)

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
    if isinstance(self, LiteralWord):
      return self.items[index]
    if isinstance(self, RepeatWord):
      return self.inner.value_at(index % self.inner.length)
    assert isinstance(self, ConcatWord)
    slot = bisect.bisect_right(self.offsets, index)
    base = 0 if slot == 0 else self.offsets[slot - 1]
    return self.parts[slot].value_at(index - base)

  def iter_atoms(self) -> Iterator[Any]:
    """Yields each AST leaf element once without expanding `* k`."""
    if isinstance(self, LiteralWord):
      yield from self.items
    elif isinstance(self, RepeatWord):
      yield from self.inner.iter_atoms()
    elif isinstance(self, ConcatWord):
      for part in self.parts:
        yield from part.iter_atoms()

  def __iter__(self) -> Iterator[Any]:
    if isinstance(self, LiteralWord):
      yield from self.items
    elif isinstance(self, RepeatWord):
      if isinstance(self.inner, LiteralWord) and len(self.inner.items) == 1:
        yield from itertools.repeat(self.inner.items[0], self.count_n)
      else:
        for _ in range(self.count_n):
          yield from self.inner
    elif isinstance(self, ConcatWord):
      for part in self.parts:
        yield from part

  def slice_word(self, start: int, stop: int) -> StaticWord:
    """Returns the symbolic sub-word for `0 <= start <= stop <= self.length`."""
    if start >= stop:
      return EMPTY_WORD
    if start == 0 and stop == self.length:
      return self
    if isinstance(self, LiteralWord):
      return LiteralWord(self.items[start:stop])
    if isinstance(self, RepeatWord):
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
    assert isinstance(self, ConcatWord)
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
    """Formats this word expression compactly without expanding `S * k`."""
    if isinstance(self, LiteralWord):
      shown = self.items[:max_items] if max_items is not None else self.items
      body = ', '.join(item_fmt(v) for v in shown)
      if max_items is not None and len(self.items) > max_items:
        return f'[{body}, …]'
      return f'[{body}]'
    if isinstance(self, RepeatWord):
      inner = self.inner.format_word(
          item_fmt=item_fmt, max_items=max_items, parenthesize_concat=True
      )
      return f'{inner}*{self.count_n}'
    assert isinstance(self, ConcatWord)
    shown_p = self.parts[:max_items] if max_items is not None else self.parts
    body = '+'.join(
        p.format_word(item_fmt=item_fmt, max_items=max_items) for p in shown_p
    )
    if max_items is not None and len(self.parts) > max_items:
      body = f'{body}+…'
    return f'({body})' if parenthesize_concat else body

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
      if not _same_atom(first, atom):
        return False, None
    return True, first

  def __contains__(self, value: Any) -> bool:
    return any(_safe_eq(atom, value) for atom in self.iter_atoms())

  def __mul__(self, count: Any) -> StaticWord:
    return NotImplemented if is_stream_obj(count) else repeat(self, count)

  def __rmul__(self, count: Any) -> StaticWord:
    return NotImplemented if is_stream_obj(count) else repeat(self, count)

  def __add__(self, other: Any) -> StaticWord:
    return NotImplemented if is_stream_obj(other) else concat(self, other)

  def __radd__(self, other: Any) -> StaticWord:
    return NotImplemented if is_stream_obj(other) else concat(other, self)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, StaticWord):
      return NotImplemented
    if self.length != other.length:
      return False
    if _struct_eq(self, other):
      return True
    u_self, v_self = self.uniform_value()
    u_other, v_other = other.uniform_value()
    if u_self and u_other:
      return same_const_value(v_self, v_other)
    return _words_eq(self, other)

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


def as_word(
    value: Any,
    *,
    label: str = 'Static word',
    accept_sequence: bool = False,
) -> StaticWord:
  """Converts a `StaticWord`, `list`, sequence, or scalar to a `StaticWord`."""
  if isinstance(value, StaticWord):
    return value
  is_seq = isinstance(value, list) or (
      accept_sequence
      and isinstance(value, Sequence)
      and not isinstance(value, (str, bytes, bytearray, LazyThunk))
  )
  if is_seq:
    items = _validate_items(value, label)
    return LiteralWord(items) if items else EMPTY_WORD
  _validate_atom(value, label)
  return LiteralWord((value,))


def repeat(
    value: Any, count: int, *, label: str = 'Static word'
) -> StaticWord:
  """Builds `S * count` symbolically without expanding `count` copies."""
  validate_non_negative_int(count, 'StaticWord repetition count')
  word = as_word(value, label=label)
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


def _merge_adjacent(prev: StaticWord, nxt: StaticWord) -> StaticWord | None:
  """Fuses adjacent `LiteralWord`s or identical `RepeatWord` runs."""
  if isinstance(prev, LiteralWord) and isinstance(nxt, LiteralWord):
    return LiteralWord(prev.items + nxt.items)
  if isinstance(prev, RepeatWord) or isinstance(nxt, RepeatWord):
    if isinstance(prev, RepeatWord):
      p_base, p_cnt = prev.inner, prev.count_n
    else:
      p_base, p_cnt = prev, 1
    if isinstance(nxt, RepeatWord):
      n_base, n_cnt = nxt.inner, nxt.count_n
    else:
      n_base, n_cnt = nxt, 1
    if _struct_eq(p_base, n_base):
      return repeat(p_base, p_cnt + n_cnt)
  return None


def concat(*parts: Any, label: str = 'Static word') -> StaticWord:
  """Flattens and fuses zero or more finite word expressions into one."""
  flat: list[StaticWord] = []
  for raw in parts:
    word = as_word(raw, label=label)
    for part in word.parts if isinstance(word, ConcatWord) else (word,):
      if part.length == 0:
        continue
      if flat and (merged := _merge_adjacent(flat[-1], part)) is not None:
        flat[-1] = merged
      else:
        flat.append(part)
  if not flat:
    return EMPTY_WORD
  total = sum(part.length for part in flat)
  if total > MAX_WORD_LENGTH:
    raise ValueError(
        f'StaticWord length exceeds limit {MAX_WORD_LENGTH} (got {total}).'
    )
  return flat[0] if len(flat) == 1 else ConcatWord(tuple(flat))


def _trim_trailing(word: StaticWord, target: Any) -> StaticWord:
  """Removes trailing copies identical to immutable/same-id `target`."""
  if isinstance(word, LiteralWord):
    end = len(word.items)
    while end > 0 and _same_atom(word.items[end - 1], target):
      end -= 1
    if end == len(word.items):
      return word
    return LiteralWord(word.items[:end]) if end > 0 else EMPTY_WORD
  if isinstance(word, RepeatWord):
    is_unif, val = word.inner.uniform_value()
    if is_unif and _same_atom(val, target):
      return EMPTY_WORD
    trimmed = _trim_trailing(word.inner, target)
    if trimmed.length == word.inner.length:
      return word
    return concat(repeat(word.inner, word.count_n - 1), trimmed)
  assert isinstance(word, ConcatWord)
  parts = list(word.parts)
  while parts:
    trimmed = _trim_trailing(parts[-1], target)
    if trimmed.length == 0:
      parts.pop()
    else:
      parts[-1] = trimmed
      break
  return concat(*parts)


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
  def from_parts(
      cls, prefix: Any, period: Any, *, accept_sequence: bool = False
  ) -> StaticSequence:
    """Builds `prefix + period*ω` validating both words."""
    # Validate `period` before `prefix` so `Stream(existing, period=thunk)`
    # reports the unresolved `LazyThunk` before rejecting the `Stream` prefix.
    per = as_word(
        period,
        label='Static stream period',
        accept_sequence=accept_sequence,
    )
    if prefix is UNSET:
      pfx = EMPTY_WORD
    else:
      pfx = as_word(
          prefix,
          label='Static stream prefix',
          accept_sequence=accept_sequence,
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
    """Yields leaf elements across `prefix` and `period` in `O(|AST|)` time."""
    yield from self.prefix.iter_atoms()
    yield from self.period.iter_atoms()

  def is_bool(self) -> bool:
    """Returns True when every element is a Python `bool` (`O(|AST|)`)."""
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
    """Structural equality over partially normalized `(prefix, period)`."""
    # Normalizes `(v^k)*ω` -> `v*ω`, uniform periods to length 1, and trims
    # trailing repeats when `len(period) == 1`. Does *not* test extensional
    # ω-word equality for multi-element periods (`[1, 2] + [1, 2]*ω` compares
    # unequal to `[1, 2]*ω`).
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
