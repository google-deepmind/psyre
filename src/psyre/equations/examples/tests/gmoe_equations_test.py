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
"""Unit tests for the `psyre_equations`-based GMoE and GMoE-RNN models."""

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.equations.examples.src import gmoe_equations
from psyre.equations.examples.src import rnn_equations
from examples.tests import gmoe_test

# pylint: enable=g-importing-member,g-multiple-import


class GmoeEquationsTest(gmoe_test.GmoeTest):
  """Runs the shared GMoE test suite against `gmoe_equations` + `rnn_equations`."""

  GMOE = gmoe_equations
  RNN = rnn_equations


if __name__ == '__main__':
  absltest.main()
