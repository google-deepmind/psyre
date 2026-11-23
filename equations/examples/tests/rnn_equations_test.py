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
"""Unit tests for the `psyre_equations`-based RNN and perceptron model."""

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from equations import psyre_equations
from equations.examples.src import rnn_equations
from examples.tests import rnn_test
from psyre.scheduler import Scheduler
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


class ModelEquationsTest(rnn_test.ModelTest):
  """Runs the shared RNN test suite against `rnn_equations`."""

  MOD = rnn_equations

  def test_relu_supports_forward_ref_thunk(self):
    with psyre_equations() as prog:

      class _:
        # Forward-referenced Thunk must resolve via Stream.max without v.max().
        # pylint: disable-next=undefined-variable,used-before-assignment
        fwd_out = rnn_equations.relu(late_vals)  # pyrefly: ignore[unknown-name]
        late_vals = Stream([-1.5, 0.0, 2.5])
        __output__ = fwd_out

    assert prog.graph is not None
    self.assertEqual(
        Scheduler.run_graph(prog.graph, horizon=3), [0.0, 0.0, 2.5]
    )


if __name__ == '__main__':
  absltest.main()
