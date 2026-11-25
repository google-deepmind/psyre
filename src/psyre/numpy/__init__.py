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
"""Optional NumPy stream-lifting module (`psyre.numpy`).

Install NumPy support via ``pip install "psyre[numpy]"``. Importing this
package lifts pure `numpy` functions and ufuncs onto Psyre `Stream`s without
adding a hard `numpy` dependency to core `psyre`.

Example::

  from psyre import Scheduler, Stream
  from psyre import numpy as pnp

  def prog():
    x = Stream([pnp.array([1.0, 2.0]), pnp.array([3.0, 4.0])])
    return pnp.sin(x) + pnp.sum(x, axis=-1, keepdims=True)

  traces = Scheduler.run(prog, horizon=2)
"""

from __future__ import annotations

# pylint: disable=g-importing-member,g-multiple-import
from psyre.numpy.numpy import (
    dir_names as _dir_names,
    get_attr as _get_attr,
)
from psyre.numpy.numpy import (
    LiftedNamespace,
    LiftedUfunc,
    batch_streams,
    fft,
    lift,
    linalg,
    random,
    stack_streams,
    unstack_stream,
    vmap,
)

# pylint: enable=g-importing-member,g-multiple-import

__all__ = (
    'LiftedNamespace',
    'LiftedUfunc',
    'batch_streams',
    'fft',
    'lift',
    'linalg',
    'random',
    'stack_streams',
    'unstack_stream',
    'vmap',
)


def __getattr__(name: str):  # pylint: disable=invalid-name
  return _get_attr(name)


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  return sorted(set(__all__) | set(_dir_names()))
