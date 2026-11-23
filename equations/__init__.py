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
"""Lambda-free declarative Psyre equations package (experimental, WIP).

Note: `psyre.equations` is highly experimental and a work in progress; APIs,
semantics, and internals may change or be removed without notice. Inspired by
Lustre/Lucid Synchrone recursive equations (Halbwachs et al., 1991; Caspi &
Pouzet, 1996; Pouzet, 2006).
"""

# pylint: disable=g-importing-member,g-multiple-import
from equations.equations import (
    CircularEquationError,
    EquationScope,
    Equations,
    PsyreEquations,
    Thunk,
    psyre_equations,
    resolve_value,
)

# pylint: enable=g-importing-member,g-multiple-import

__all__ = (
    'CircularEquationError',
    'EquationScope',
    'Equations',
    'PsyreEquations',
    'Thunk',
    'psyre_equations',
    'resolve_value',
)
