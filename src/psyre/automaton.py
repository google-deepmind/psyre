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
from psyre.builder import GraphBuilder
from psyre.common import (
    ArmRecord,
    ArmScope,
    BlockContext,
    BlockHandle,
    CapturedEnv,
    MatchArm,
    UNSET,
    active_block,
    fby,
    ifthenelse,
    is_thunk,
    matches_tag,
    push_arm,
    push_prefix,
    push_reset,
    require_open_arm_context,
    require_sync_semantics,
    switch,
    when,
)
from psyre.istructure import (
    Absent,
    IStruct,
    as_istruct,
    is_one_periodic,
)
from psyre.match import (
    finalize_named_outputs,
    make_arm_selector,
    merge_arm_outputs,
    validate_discrete_tag,
    validate_exhaustiveness,
)
from psyre.stream import Stream

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
  require_sync_semantics('until')
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
  require_sync_semantics('unless')
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
      isinstance(target_tag, Stream)
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


def _static_const_value(val: Any) -> tuple[bool, Any]:
  """Returns (True, scalar) if val is a static constant, else (False, None)."""
  if val is Absent:
    raise TypeError('Periodic stream cannot contain Absent.')
  if isinstance(val, list):
    if any(item is Absent for item in val):
      raise TypeError('Periodic stream cannot contain Absent.')
    if len(val) == 1:
      return True, val[0]
    return False, None
  if is_thunk(val):
    return False, None
  if isinstance(val, (IStruct, Stream)):
    istruct = as_istruct(val)
    if is_one_periodic(istruct):
      return True, istruct.period[0]
    return False, None
  return True, val


def _is_static_false(val: Any) -> bool:
  """Returns True when val is statically constant and falsy."""
  is_const, c_val = _static_const_value(val)
  return is_const and not c_val


def _same_static_value(a: Any, b: Any) -> bool:
  """Returns True when a and b are identical streams or equal static tags."""
  if a is b:
    return True
  if isinstance(a, (IStruct, Stream)) and isinstance(
      b, (IStruct, Stream)
  ):
    if as_istruct(a) is as_istruct(b):
      return True
  a_const, a_val = _static_const_value(a)
  b_const, b_val = _static_const_value(b)
  return a_const and b_const and matches_tag(a_val, b_val)


def _select_transition_value(
    cond: Any,
    then_val: Any,
    else_val: Any,
    *,
    is_reset: bool = False,
) -> Any:
  """Selects between then_val and else_val under cond with constant folding."""
  if _same_static_value(then_val, else_val):
    return then_val
  if is_reset:
    t_const, t_val = _static_const_value(then_val)
    e_const, e_val = _static_const_value(else_val)
    if t_const and e_const:
      t_b, e_b = bool(t_val), bool(e_val)
      if t_b == e_b:
        return t_b
      if (
          isinstance(cond, (IStruct, Stream))
          and (curr_builder := GraphBuilder.current()) is not None
          and curr_builder.is_bool_stream(cond)
      ):
        return cond if t_b else ~cond
  return ifthenelse(cond, then_val, else_val)


def _fold_transitions(
    transitions: Sequence[_Transition],
    fallback_mode: Any,
    fallback_reset: Any,
    arms: Sequence[MatchArm],
) -> tuple[Any, Any]:
  """Folds prioritized transitions into (mode, reset) streams or values."""
  guarded: list[_Transition] = []
  mode_acc = fallback_mode
  reset_acc = fallback_reset
  reached_const_true = False
  for trans in transitions:
    _validate_tag_in_arms(trans.target, arms)
    is_const, c_val = _static_const_value(trans.cond)
    _static_const_value(trans.reset)
    if reached_const_true:
      continue
    if is_const:
      if c_val:
        mode_acc, reset_acc = trans.target, trans.reset
        reached_const_true = True
    else:
      guarded.append(trans)
  for trans in reversed(guarded):
    mode_acc = _select_transition_value(trans.cond, trans.target, mode_acc)
    reset_acc = _select_transition_value(
        trans.cond, trans.reset, reset_acc, is_reset=True
    )
  return mode_acc, reset_acc


@contextlib.contextmanager
def automaton(
    init: Any = UNSET,
    name: str = '',
) -> Iterator[BlockHandle]:
  """Context manager for stateful hierarchical automata with transitions."""
  require_sync_semantics('automaton')
  if not isinstance(name, str):
    raise TypeError(
        f'automaton(name=...) must be a str, got {type(name).__name__}.'
    )
  if init is not UNSET and not isinstance(init, Stream):
    validate_discrete_tag(init, 'automaton(init=...)')
  curr_builder = GraphBuilder.current()
  if curr_builder is None:
    raise RuntimeError('Psyre nodes can only be instantiated during Stage 1.')
  fsm_prefix = name if name else f'fsm_{curr_builder.next_fsm_id()}'
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

    has_strong = any(rec.strong_transitions for rec in ctx.arm_records)
    has_weak = any(rec.weak_transitions for rec in ctx.arm_records)
    has_weak_reset = any(
        any(not _is_static_false(t.reset) for t in rec.weak_transitions)
        for rec in ctx.arm_records
    )

    pns_stream = fby(init_state, lambda: ctx.ns_stream, name='pns')
    pnr_stream: Any = (
        fby(False, lambda: ctx.nr_stream, name='pnr')
        if has_weak_reset
        else False
    )

    if has_strong:
      strong_states: dict[Any, Any] = {}
      strong_resets: dict[Any, Any] = {}
      for rec in ctx.arm_records:
        tag = rec.arm.values[0]
        strong_states[tag], strong_resets[tag] = _fold_transitions(
            rec.strong_transitions, tag, pnr_stream, arms
        )
      s_stream = switch(pns_stream, strong_states, name='state_s')
      if all(_is_static_false(v) for v in strong_resets.values()):
        r_stream: Any = False
      else:
        r_stream = switch(pns_stream, strong_resets, name='state_r')
    else:
      s_stream, r_stream = pns_stream, pnr_stream

    ctx.s_stream = s_stream
    ctx.r_stream = r_stream

    weak_scopes: list[ArmScope] = []
    weak_states: list[Any] = []
    weak_resets: list[Any] = []
    for rec in ctx.arm_records:
      rec.arm_scope.cond = s_stream
      weak_scopes.append(rec.arm_scope)
      if has_weak:
        with push_arm(rec.arm_scope):
          ns_next, nr_next = _fold_transitions(
              rec.weak_transitions, rec.arm.values[0], False, arms
          )
        weak_states.append(ns_next)
        weak_resets.append(nr_next)

    if has_weak:
      ctx.ns_stream = merge_arm_outputs(
          s_stream, arms, weak_states, name='state_ns', arm_scopes=weak_scopes
      )
    else:
      ctx.ns_stream = s_stream
    if has_weak_reset:
      ctx.nr_stream = merge_arm_outputs(
          s_stream, arms, weak_resets, name='state_nr', arm_scopes=weak_scopes
      )
    else:
      ctx.nr_stream = False
    finalize_named_outputs(ctx, s_stream, arms)


@contextlib.contextmanager
def state(
    tag: Any,
    *unless_clauses: _Transition,
) -> Iterator[Stream]:
  """Context manager declaring a discrete state inside `with automaton(...)`."""
  require_sync_semantics('state')
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
  arm_scope = ArmScope(selector=make_arm_selector(arm))
  arm_record = ArmRecord(
      arm=arm, arm_scope=arm_scope, strong_transitions=list(unless_clauses)
  )
  sampled = when(
      lambda: ctx.s_stream,
      lambda: ctx.s_stream,
      selector=arm_scope.selector,
      name='state',
  )
  ctx.active_arm = arm_record
  try:
    with push_arm(arm_scope):
      with push_reset(lambda: ctx.r_stream):
        yield sampled
  finally:
    ctx.active_arm = None
  ctx.arm_records.append(arm_record)
