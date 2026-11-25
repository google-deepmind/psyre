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
"""Shared pure-stdlib test helpers, enums, and fakes for Psyre test suites."""

from collections.abc import Callable, Sequence
import dataclasses
import enum
from typing import Any
import unittest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    Absent,
    Bottom,
    Semantics,
)
from psyre.scheduler import Scheduler
from psyre.trace import present_values

# pylint: enable=g-importing-member,g-multiple-import


class Mode(enum.Enum):
  """Two-state discrete enum shared across clock, match, and automaton tests."""

  FAST = 1
  SLOW = 2


class TriMode(enum.Enum):
  """Three-state discrete enum shared across match and automaton tests."""

  IDLE = enum.auto()
  RUN = enum.auto()
  STOP = enum.auto()


@dataclasses.dataclass(frozen=True)
class EqAll:
  """Test double whose `__eq__` returns True for every object."""

  tag: str = ''

  def __eq__(self, other: object) -> bool:
    del other
    return True


class FakeArray:
  """Minimal array-like duck type (no NumPy/JAX dependency) for guard tests."""

  shape = (1,)
  dtype = 'bool'


def assert_no_sentinels(
    test_case: unittest.TestCase, trace: Sequence[Any]
) -> None:
  """Asserts an unpadded trace contains neither `Absent` nor `Bottom`."""
  test_case.assertTrue(
      all(v is not Absent and v is not Bottom for v in trace),
      f'Unexpected Absent/Bottom sentinel in trace: {trace!r}',
  )


def assert_absent_bottom_elision(
    test_case: unittest.TestCase,
    sync_vals: Sequence[Any],
    kahn_vals: Sequence[Any],
) -> None:
  """Asserts `kahn_vals` contains no sentinels and matches `present_values`."""
  assert_no_sentinels(test_case, kahn_vals)
  test_case.assertEqual(present_values(sync_vals), kahn_vals)


def assert_psyre_error(
    test_case: unittest.TestCase,
    main_fn: Callable[[], Any],
    expected_exc_type: type[Exception],
    expected_regex: str,
    semantics: Semantics = Semantics.SYNC,
) -> None:
  """Asserts running `main_fn` for 1 tick raises `expected_exc_type`."""
  with test_case.assertRaisesRegex(expected_exc_type, expected_regex):
    Scheduler.run(main_fn, horizon=1, semantics=semantics)


def assert_buffer_error(
    test_case: unittest.TestCase,
    prog: Callable[[], Any],
    horizon: int,
    pattern: str,
    semantics: Semantics = Semantics.SYNC,
) -> None:
  """Asserts `prog` fails with an `ExceptionGroup` containing `BufferError`."""
  with test_case.assertRaises(ExceptionGroup) as eg:
    Scheduler.run(prog, horizon=horizon, semantics=semantics)
  test_case.assertTrue(
      any(
          isinstance(err, BufferError) and pattern in str(err)
          for err in eg.exception.exceptions
      ),
      f'Expected BufferError matching {pattern!r}, got '
      f'{eg.exception.exceptions!r}',
  )
