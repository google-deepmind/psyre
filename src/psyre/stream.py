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
"""Psyre Stream handle, pointwise operators, and stream constructors."""

from collections.abc import Sequence
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    active_builder,
    is_thunk,
    make_primitive,
    psyre_comb,
)
from psyre.istructure import IStructHandle

# pylint: enable=g-importing-member,g-multiple-import

_once_stream = make_primitive('once_stream')
_random_stream = make_primitive('random_stream')


class Stream(IStructHandle):
  """Stream handle with pointwise operators and stream constructors."""

  def __init__(self, *args, name: str = '', **kwargs):
    """Initializes a Stream handle and registers it if Stage 1 is active."""
    super().__init__(*args, name=name, **kwargs)
    builder = active_builder.get()
    if builder is not None:
      builder.add_stream(self)

  @classmethod
  def periodic(
      cls, prefix: Sequence[Any], period: Sequence[Any], name: str = ''
  ) -> 'Stream':
    """Returns a static periodic stream with the given prefix and period."""
    return cls(list(prefix), period=list(period), name=name)

  @classmethod
  def once(cls, k: Any, **kwargs: Any) -> 'Stream':
    """Returns a resettable stream emitting True once at position k."""
    return _once_stream(k, **kwargs)

  @classmethod
  def random(cls, **kwargs: Any) -> 'Stream':
    """Returns a stream of random floats in [0.0, 1.0)."""
    return _random_stream(**kwargs)

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

  @psyre_comb
  def sum(self, *rest):
    """Returns the pointwise sum across one or more scalar inputs."""
    return sum((self, *rest))


def resolve_handle(stream_or_thunk: Any) -> Stream:
  """Resolves any zero-arg thunks and returns a fresh Stream handle."""
  while is_thunk(stream_or_thunk):
    stream_or_thunk = stream_or_thunk()
  return Stream(stream_or_thunk)
