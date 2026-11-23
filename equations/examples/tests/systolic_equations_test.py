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
"""Unit tests for the `psyre_equations`-based 4x4 systolic array."""

from absl.testing import absltest

from equations.examples.src import systolic_equations
from examples.tests import systolic_test


class SystolicEquationsTest(systolic_test.SystolicTest):
  """Runs the shared systolic test suite against `systolic_equations`."""

  MOD = systolic_equations


if __name__ == '__main__':
  absltest.main()
