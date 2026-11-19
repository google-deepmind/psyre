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
"""Post-run trace inspection helpers for Psyre stream outputs."""

from collections.abc import Sequence
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    active_builder,
    active_scheduler,
)
from psyre.istructure import Absent

# pylint: enable=g-importing-member,g-multiple-import


def _validate_post_run_trace(trace: Any, caller: str) -> Sequence[Any]:
  """Validates that trace is a post-run Sequence inspected outside Scheduler."""
  sched = active_scheduler.get()
  if active_builder.get() is not None or (
      sched is not None and sched.is_active_run()
  ):
    raise RuntimeError(f'{caller}() can only be used outside a Scheduler run.')
  # str/bytes/bytearray implement Sequence, so reject them explicitly.
  if isinstance(trace, (str, bytes, bytearray)) or not isinstance(
      trace, Sequence
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
