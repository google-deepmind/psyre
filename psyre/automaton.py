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
"""Psyre hierarchical state-machine (automaton) context managers."""

from collections.abc import Iterator, Sequence
import contextlib
import dataclasses
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre.common import (
    ArmRecord,
    ArmScope,
    BlockContext,
    BlockHandle,
    CapturedEnv,
    MatchArm,
    UNSET,
    active_block,
    is_thunk,
    matches_tag,
    push_arm,
    push_prefix,
    push_reset,
    require_open_arm_context,
)
from psyre.istructure import IStructHandle
from psyre.match import (
    finalize_named_outputs,
    make_arm_activity,
    merge_arm_outputs,
    validate_discrete_tag,
    validate_exhaustiveness,
)
from psyre.psyre import (
    Scheduler,
    Stream,
    fby,
    ifthenelse,
    when,
)

# pylint: enable=g-importing-member,g-multiple-import


@dataclasses.dataclass
class _Transition:
  """Guarded state transition specification (for unless() and until())."""

  cond: Any
  target: Any
  reset: Any
  consumed: bool = False


def until(
    cond: Any, target: Any, *, reset: Any = True  # pylint: disable=redefined-outer-name
) -> None:
  """Registers a weak transition evaluated after the current state step."""
  Scheduler.require_sync_semantics('until')
  ctx = active_block.get()
  if ctx is None or ctx.kind != 'automaton' or ctx.active_arm is None:
    raise RuntimeError(
        'until() can only be called directly inside a state() block of '
        'automaton().'
    )
  ctx.active_arm.weak_transitions.append(
      _Transition(cond=cond, target=target, reset=reset)
  )


def unless(
    cond: Any, target: Any, *, reset: Any = True  # pylint: disable=redefined-outer-name
) -> _Transition:
  """Creates a strong transition clause for `state(tag, unless(...), ...)`."""
  Scheduler.require_sync_semantics('unless')
  ctx = active_block.get()
  if ctx is None or ctx.kind != 'automaton':
    raise RuntimeError(
        'unless() can only be used in state(tag, unless(...)) inside '
        'automaton().'
    )
  if ctx.active_arm is not None:
    raise RuntimeError(
        'unless() cannot be called inside a state() body; pass unless(...) '
        'directly to state(tag, unless(...)).'
    )
  clause = _Transition(cond=cond, target=target, reset=reset)
  ctx.unconsumed_unless.append(clause)
  return clause


def _validate_tag_in_arms(
    target_tag: Any,
    arms: Sequence[MatchArm],
    *,
    role: str = 'Transition target',
) -> None:
  """Validates that a discrete state tag matches at least one arm."""
  if (
      isinstance(target_tag, IStructHandle)
      or is_thunk(target_tag)
      or any(arm.is_wildcard for arm in arms)
  ):
    return
  for arm in arms:
    if any(matches_tag(target_tag, tag) for tag in arm.values):
      return
  raise ValueError(
      f'{role} {target_tag!r} does not match any state in automaton().'
  )


def _fold_transitions(
    transitions: Sequence[_Transition],
    fallback_mode: Any,
    fallback_reset: Any,
    arms: Sequence[MatchArm],
) -> tuple[Any, Any]:
  """Folds prioritized transitions into (mode, reset) streams or values."""
  mode_acc = fallback_mode
  reset_acc = fallback_reset
  for trans in reversed(transitions):
    _validate_tag_in_arms(trans.target, arms)
    mode_acc = ifthenelse(trans.cond, trans.target, mode_acc)
    reset_acc = ifthenelse(trans.cond, trans.reset, reset_acc)
  return mode_acc, reset_acc


@contextlib.contextmanager
def automaton(
    init: Any = UNSET,
    name: str = '',
) -> Iterator[BlockHandle]:
  """Context manager for stateful hierarchical automata with transitions."""
  Scheduler.require_sync_semantics('automaton')
  if not isinstance(name, str):
    raise TypeError(
        f'automaton(name=...) must be a str, got {type(name).__name__}.'
    )
  if init is not UNSET and not isinstance(init, IStructHandle):
    validate_discrete_tag(init, 'automaton(init=...)')
  fsm_prefix = name if name else f'fsm_{Scheduler.graph.next_fsm_id()}'
  with push_prefix(fsm_prefix):
    enclosing_env = CapturedEnv.capture()
    ctx = BlockContext(
        kind='automaton',
        enclosing_env=enclosing_env,
        init=init,
    )
    handle = BlockHandle(ctx)
    block_token = active_block.set(ctx)
    try:
      yield handle
    finally:
      active_block.reset(block_token)

    if any(not clause.consumed for clause in ctx.unconsumed_unless):
      raise RuntimeError(
          'unless() clause must be passed directly to state(tag, unless(...)).'
      )
    if not ctx.arm_records:
      raise ValueError('automaton() requires at least one state().')

    arms = [rec.arm for rec in ctx.arm_records]
    if ctx.init is UNSET:
      init_state = arms[0].values[0]
    else:
      init_state = ctx.init
      _validate_tag_in_arms(init_state, arms, role='Initial state')
    validate_exhaustiveness(arms, init_state, construct='automaton')

    pns_stream = fby(init_state, lambda: ctx.ns_stream, name='pns')
    pnr_stream = fby(False, lambda: ctx.nr_stream, name='pnr')

    if any(rec.strong_transitions for rec in ctx.arm_records):
      strong_states: list[Any] = []
      strong_resets: list[Any] = []
      strong_scopes: list[ArmScope] = []
      for rec in ctx.arm_records:
        pns_scope = ArmScope(cond=make_arm_activity(pns_stream, rec.arm, ()))
        with push_arm(pns_scope):
          if rec.strong_transitions:
            s_next, r_next = _fold_transitions(
                rec.strong_transitions, pns_stream, pnr_stream, arms
            )
          else:
            s_next, r_next = pns_stream, pnr_stream
        strong_states.append(s_next)
        strong_resets.append(r_next)
        strong_scopes.append(pns_scope)
      s_stream = merge_arm_outputs(
          pns_stream,
          arms,
          strong_states,
          name='state_s',
          arm_scopes=strong_scopes,
      )
      r_stream = merge_arm_outputs(
          pns_stream,
          arms,
          strong_resets,
          name='state_r',
          arm_scopes=strong_scopes,
      )
    else:
      s_stream = pns_stream
      r_stream = pnr_stream

    ctx.s_stream = s_stream
    ctx.r_stream = r_stream

    weak_states: list[Any] = []
    weak_resets: list[Any] = []
    weak_scopes: list[ArmScope] = []
    for rec in ctx.arm_records:
      s_activity = make_arm_activity(s_stream, rec.arm, ())
      rec.s_activity_stream = s_activity
      s_scope = rec.arm_scope or ArmScope(cond=s_activity)
      s_scope.cond = s_activity
      with push_arm(s_scope):
        if rec.weak_transitions:
          ns_next, nr_next = _fold_transitions(
              rec.weak_transitions, s_stream, False, arms
          )
        else:
          ns_next, nr_next = s_stream, False
      weak_states.append(ns_next)
      weak_resets.append(nr_next)
      weak_scopes.append(s_scope)

    ctx.ns_stream = merge_arm_outputs(
        s_stream, arms, weak_states, name='state_ns', arm_scopes=weak_scopes
    )
    ctx.nr_stream = merge_arm_outputs(
        s_stream, arms, weak_resets, name='state_nr', arm_scopes=weak_scopes
    )
    finalize_named_outputs(ctx, s_stream, arms)


@contextlib.contextmanager
def state(
    tag: Any,
    *unless_clauses: _Transition,
) -> Iterator[Stream]:
  """Context manager declaring a discrete state inside `with automaton(...)`."""
  Scheduler.require_sync_semantics('state')
  ctx = require_open_arm_context('automaton', 'state()')
  validate_discrete_tag(tag, 'state()')
  if any(matches_tag(tag, existing) for existing in ctx.claimed_tags):
    raise ValueError(f'Duplicate state {tag!r} in automaton().')
  for clause in unless_clauses:
    if not isinstance(clause, _Transition):
      raise TypeError(
          'Extra arguments to state() must be unless(...) clauses, got '
          f'{type(clause).__name__}.'
      )
    clause.consumed = True

  ctx.claimed_tags.append(tag)
  arm = MatchArm(values=(tag,), is_wildcard=False)
  arm_record = ArmRecord(arm=arm, strong_transitions=list(unless_clauses))
  s_act_thunk = lambda: arm_record.s_activity_stream
  arm_scope = ArmScope()
  arm_record.arm_scope = arm_scope
  sampled = when(s_act_thunk, lambda: ctx.s_stream, name='state')
  ctx.active_arm = arm_record
  try:
    with push_arm(arm_scope):
      with push_reset(lambda: ctx.r_stream):
        yield sampled
  finally:
    ctx.active_arm = None
  ctx.arm_records.append(arm_record)
