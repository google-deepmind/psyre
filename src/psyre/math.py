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
"""Pointwise mathematical stream primitives lifted via `@psyre_comb`."""

import math

from psyre.common import psyre_comb  # pylint: disable=g-importing-member


@psyre_comb
def exp(v):
  """Pointwise exponential stream function."""
  return math.exp(v)


@psyre_comb
def tanh(v):
  """Pointwise hyperbolic tangent stream function."""
  return math.tanh(v)
