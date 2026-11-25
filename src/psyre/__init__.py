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
"""Psyre reactive dataflow DSL in Python.

Unifies synchronous Lustre / Lucid Synchrone semantics (`Semantics.SYNC`;
Halbwachs et al., *Proc. IEEE* 1991; Caspi & Pouzet, *ICFP* 1996; Hamon &
Pouzet, *PPDP* 2000; Cohen et al., *POPL* 2006) with asynchronous deterministic
Kahn Process Networks (`Semantics.KAHN`; Kahn, *IFIP* 1974; Kahn & MacQueen,
*IFIP* 1977) and bidirectional ML dataflow extensions (`post` lookahead and
bounded buffers; Potop-Butucaru, Cohen, Plotkin & Pompougnac, 2023,
arXiv:2311.16977).
"""

__version__ = '0.1.0'

# pylint: disable=g-importing-member,g-multiple-import
from psyre import automaton
from psyre import kahn_primitives as _kahn_primitives  # pylint: disable=unused-import
from psyre import match
from psyre import sync_primitives as _sync_primitives  # pylint: disable=unused-import
from psyre.analysis import (
    CausalityError,
    causal_node_order,
    check_causality,
    find_instantaneous_cycle,
)
from psyre.builder import GraphBuilder, build_graph, lower_resets
from psyre.clock import (
    BASE_CLOCK,
    Clock,
    ClockAnalysis,
    ClockError,
    ClockStep,
    check_clocks,
)
from psyre.common import (
    Absent,
    Bottom,
    LazyThunk,
    Semantics,
    buffer,
    counter,
    current,
    fby,
    force_thunk,
    ifthenelse,
    is_building,
    is_pure_step_func,
    is_thunk,
    logger,
    merge,
    morph,
    post,
    pre,
    prefix_sum,
    psyre_comb,
    psyre_loop,
    psyre_node,
    psyre_sequ,
    register_primitives,
    require_sync_semantics,
    reset,
    scan,
    switch,
    when,
    whennot,
)
from psyre.graph import Graph, Node
from psyre.math import (
    exp,
    random_stream,
    tanh,
)
from psyre.optimization import (
    eliminate_dead_code,
    find_dead_nodes,
    find_live_nodes,
    optimize_graph,
)
from psyre.payload import (
    register_pytree_node,
    unregister_pytree_node,
)
from psyre.scheduler import Scheduler
from psyre.static_stream import (
    EMPTY_WORD,
    StaticSequence,
    StaticWord,
    concat,
    repeat,
)
from psyre.stream import (
    Stream,
    resolve_stream,
)
from psyre.trace import (
    is_absent,
    is_bottom,
    is_present,
    present_values,
)

# pylint: enable=g-importing-member,g-multiple-import

__all__ = (
    'Absent',
    'BASE_CLOCK',
    'Bottom',
    'CausalityError',
    'Clock',
    'ClockAnalysis',
    'ClockError',
    'ClockStep',
    'EMPTY_WORD',
    'Graph',
    'GraphBuilder',
    'LazyThunk',
    'Node',
    'Scheduler',
    'Semantics',
    'StaticSequence',
    'StaticWord',
    'Stream',
    '__version__',
    'automaton',
    'buffer',
    'build_graph',
    'causal_node_order',
    'check_causality',
    'check_clocks',
    'concat',
    'counter',
    'current',
    'eliminate_dead_code',
    'exp',
    'fby',
    'find_dead_nodes',
    'find_instantaneous_cycle',
    'find_live_nodes',
    'force_thunk',
    'ifthenelse',
    'is_absent',
    'is_bottom',
    'is_building',
    'is_present',
    'is_pure_step_func',
    'is_thunk',
    'logger',
    'lower_resets',
    'match',
    'merge',
    'morph',
    'optimize_graph',
    'post',
    'pre',
    'prefix_sum',
    'present_values',
    'psyre_comb',
    'psyre_loop',
    'psyre_node',
    'psyre_sequ',
    'random_stream',
    'register_primitives',
    'register_pytree_node',
    'repeat',
    'require_sync_semantics',
    'reset',
    'resolve_stream',
    'scan',
    'switch',
    'tanh',
    'unregister_pytree_node',
    'when',
    'whennot',
)
