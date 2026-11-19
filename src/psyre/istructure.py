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
"""Psyre I-structure and stream handle definitions."""

import asyncio
import logging
from typing import Any

_logger = logging.getLogger('psyre')


class _Absent:
  """Singleton representing an absent value in a synchronous stream."""

  def __bool__(self) -> bool:
    return False

  def __eq__(self, _) -> bool:
    return False

  def __repr__(self) -> str:
    return 'Absent'

  def __str__(self) -> str:
    return 'Absent'


Absent = _Absent()  # pylint: disable=invalid-name


class IStruct:
  """Single-assignment concurrent structure for static and dynamic streams."""

  _id: int = 0

  id: int
  static: bool
  capture: bool
  prefix: list[Any]
  period: list[Any]
  seq: list[asyncio.Future[Any]]
  keep: list[Any]

  def __init__(self, *args, **kwargs):
    self.id = IStruct._id
    IStruct._id += 1
    self.capture = False
    self.prefix = []
    self.period = []
    self.seq = []
    self.keep = []
    if len(args) == 0 and 'period' not in kwargs:
      self.static = False
    else:
      self.static = True
      if len(args) == 1 and isinstance(args[0], list):
        seq = args[0]
      else:
        seq = list(args)
      if 'period' in kwargs:
        self.prefix = seq
        period = kwargs.pop('period')
        self.period = period if isinstance(period, list) else [period]
      else:
        self.prefix = seq[0:-1]
        self.period = [seq[-1]]
      if any(item is Absent for item in (*self.prefix, *self.period)):
        raise TypeError('Periodic stream cannot contain Absent.')

  def __str__(self):
    if self.static:
      if len(self.prefix) == 0:
        prefix = ''
      else:
        prefix = ', '.join([str(value) for value in self.prefix])
        prefix = f'[{prefix}]+'
      period = ', '.join([str(value) for value in self.period])
      return f'S_{self.id}{prefix}[{period}]*ω'
    else:
      return f'S_{self.id}{self.keep}'

  def static_value_at(self, index: int) -> Any:
    """Returns the periodic static stream value at index."""
    if index < len(self.prefix):
      return self.prefix[index]
    return self.period[(index - len(self.prefix)) % len(self.period)]

  def trace(self, horizon: int) -> list[Any]:
    """Returns captured dynamic values or horizon static elements."""
    if self.static:
      return [self.static_value_at(i) for i in range(horizon)]
    return list(self.keep)

  async def pull(self, index):
    """Returns the value at the given stream index, waiting if not yet pushed."""
    if self.static:
      return self.static_value_at(index)
    else:
      while len(self.seq) <= index:
        self.seq.append(asyncio.Future())
      value = await self.seq[index]  # Blocks until matching stream push
      return value

  def push(self, value, index):
    """Stores a value at the stream index and resolves any waiting future."""
    if self.static:
      raise RuntimeError('Cannot push to a static stream.')
    if len(self.seq) == index or self.seq[index].done():
      value_future = asyncio.Future()
      self.seq.append(value_future)
    else:
      value_future = self.seq[index]
    value_future.set_result(value)
    if self.capture or _logger.isEnabledFor(logging.INFO):
      self.keep.append(value)


class IStructHandle:
  """Stream handle indexing over a shared I-structure."""

  _id: int = 0

  id: int
  name: str
  index: int
  istruct: IStruct

  def __init__(self, *args, name: str = '', **kwargs):
    self.id = IStructHandle._id
    IStructHandle._id += 1
    self.index = 0
    if len(args) == 1 and isinstance(args[0], IStructHandle):
      self.istruct = args[0].istruct  # Shared I-structure
      self.name = name or args[0].name
    elif len(args) == 1 and isinstance(args[0], IStruct):
      self.istruct = args[0]  # Existing I-structure
      self.name = name
    else:
      self.istruct = IStruct(*args, **kwargs)  # New I-structure
      self.name = name

  def __str__(self) -> str:
    return f'S{self.name}_{self.id}({self.istruct})'

  async def pull(self) -> Any:
    """Pulls the next value from the I-structure and advances the index."""
    result = await self.istruct.pull(self.index)
    _logger.debug('pull %s, %s -> %s', self, self.index, result)
    self.index += 1
    return result

  def push(self, value: Any) -> None:
    """Pushes a value into the underlying I-structure and advances the index."""
    _logger.debug('push %s, %s <- %s', self, self.index, value)
    self.istruct.push(value, self.index)
    self.index += 1


def as_istruct(stream: IStruct | IStructHandle) -> IStruct:
  """Extracts the underlying IStruct from an IStruct or IStructHandle."""
  if isinstance(stream, IStruct):
    return stream
  if isinstance(stream, IStructHandle):
    return stream.istruct
  raise TypeError(
      f'Expected IStruct or IStructHandle, got {type(stream).__name__}.'
  )


def is_one_periodic(stream: IStruct | IStructHandle) -> bool:
  """Returns True when stream is a single-element periodic constant."""
  istruct = as_istruct(stream)
  return istruct.static and not istruct.prefix and len(istruct.period) == 1

