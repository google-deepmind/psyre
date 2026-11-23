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
"""Psyre hierarchical state-machine (automaton) context managers.

Lowers hierarchical mode automata with strong (`unless`) and weak (`until`)
guarded transitions and reset vs. history re-entry (`reset=True`/`False`) into
core synchronous dataflow operators (`fby`, `switch`, `merge`, `when`, `reset`):

  * Mode automata and strong/weak transitions: Maraninchi & Rémond, *Mode-
    automata: About modes and states for reactive systems*, ESOP 1998 / *SCP*
    46(3), 2003.
  * Conservative extension with shared `last` memory: Colaço, Pagano & Pouzet,
    *A conservative extension of synchronous data-flow with state machines*,
    EMSOFT 2005.
  * Modular reset compilation: Hamon & Pouzet, PPDP 2000; mechanized compilation
    of automata and resets in Bourke, Brun & Pouzet (POPL 2020) and Bourke,
    Pesin & Pouzet (*ACM TECS*, 2023).
"""

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
    MergeSelector,
    TagKey,
    UNSET,
    active_block,
    fby,
    ifthenelse,
    is_thunk,
    matches_tag,
    push_arm,
    push_prefix,
    push_reset,
    require_builder,
    require_open_arm_context,
    switch,
    tag_key,
    when,
)
from psyre.common_payload import is_array_object
from psyre.match import (
    bind_thunk_env,
    finalize_named_outputs,
    make_arm_selector,
    merge_arm_outputs,
    validate_discrete_tag,
    validate_exhaustiveness,
)
from psyre.static_stream import StaticSequence
from psyre.stream import Stream

# pylint: enable=g-importing-member,g-multiple-import


@dataclasses.dataclass
class _Transition:
  """Guarded state transition specification (for unless() and until())."""

  cond: Any
  target: Any
  reset: Any
  env: CapturedEnv = dataclasses.field(default_factory=CapturedEnv.capture)
  consumed: bool = False


def until(
    cond: Any, target: Any, *, reset: Any = True  # pylint: disable=redefined-outer-name
) -> None:
  """Registers a weak transition evaluated after the current state step."""
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
  if isinstance(target_tag, Stream) or is_thunk(target_tag):
    return
  if any(matches_tag(target_tag, arm.values[0]) for arm in arms):
    return
  raise ValueError(
      f'{role} {target_tag!r} does not match any state in automaton().'
  )


def _static_const_value(val: Any) -> tuple[bool, Any]:
  """Returns (True, scalar) if val is a static constant, else (False, None)."""
  if is_thunk(val):
    return False, None
  if isinstance(val, Stream):
    return (True, val.const_value) if val.is_one_periodic else (False, None)
  spec = StaticSequence.from_value(val)
  return (True, spec.const_value) if spec.is_one_periodic else (False, None)


def _scalar_bool(val: Any, role: str) -> bool:
  """Converts a scalar constant to `bool`, rejecting array objects."""
  if is_array_object(val):
    raise TypeError(
        f'automaton() {role} must be a scalar truth value, got '
        f'{type(val).__name__}.'
    )
  return bool(val)


def _is_static_false(val: Any) -> bool:
  """Returns True when val is a statically constant, falsy reset."""
  is_const, c_val = _static_const_value(val)
  return is_const and not _scalar_bool(c_val, 'reset')


def _select_transition_value(
    cond: Any,
    then_val: Any,
    else_val: Any,
    *,
    is_reset: bool = False,
) -> Any:
  """Selects between then_val and else_val under cond with constant folding."""
  if then_val is else_val:
    return then_val
  t_const, t_val = _static_const_value(then_val)
  e_const, e_val = _static_const_value(else_val)
  if t_const and e_const:
    if matches_tag(t_val, e_val):
      return then_val
    if is_reset:
      t_b = _scalar_bool(t_val, 'reset')
      e_b = _scalar_bool(e_val, 'reset')
      if t_b == e_b:
        return t_b
      if (
          isinstance(cond, Stream)
          and (curr_builder := GraphBuilder.current()) is not None
          and curr_builder.is_bool_stream(cond)
      ):
        return cond if t_b else ~cond
  return ifthenelse(cond, then_val, else_val)


def _bind_transition_value(val: Any, env: CapturedEnv) -> Any:
  """Binds a transition target/reset to `env` so `reset(...)` scopes apply."""
  if is_thunk(val):
    return bind_thunk_env(val, env)
  if isinstance(val, Stream) or not env.resets:
    return val
  is_const, _ = _static_const_value(val)
  if is_const:
    # Keep 1-periodic constants raw so `_is_static_false` can fold them.
    return val
  # Non-1-periodic literals (e.g. `[False, True, False]`) must be wrapped inside
  # `env.restore()` so enclosing `with reset(r):` scopes make them resettable.
  with env.restore():
    return Stream(val)


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
    # Validate `trans.reset` eagerly even for unreachable clauses after a
    # constant-True guard (rejects sentinels, callables, empty lists, arrays).
    r_const, r_val = _static_const_value(trans.reset)
    if r_const:
      _scalar_bool(r_val, 'reset')
    if is_const:
      cond_bool = _scalar_bool(c_val, 'transition condition')
    else:
      cond_bool = False
    if reached_const_true:
      continue
    if is_const:
      if cond_bool:
        mode_acc = _bind_transition_value(trans.target, trans.env)
        reset_acc = _bind_transition_value(trans.reset, trans.env)
        reached_const_true = True
    else:
      guarded.append(trans)
  for trans in reversed(guarded):
    with trans.env.restore():
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
  if not isinstance(name, str):
    raise TypeError(
        f'automaton(name=...) must be a str, got {type(name).__name__}.'
    )
  if init is not UNSET and not isinstance(init, Stream):
    validate_discrete_tag(init, 'automaton(init=...)')
  curr_builder = require_builder()
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
    validate_exhaustiveness(arms, construct='automaton')

    # Two-phase Moore/Mealy encoding (Colaço, Pagano & Pouzet, EMSOFT 2005):
    #   pns = fby(init, ns)   pnr = fby(False, nr)   (prev next-state / reset)
    #   s_stream / r_stream = strong `unless` switch on `pns` (this tick)
    #   state bodies + `last()` run on `(s_stream == tag)` reset by `r_stream`
    #   ns_stream / nr_stream = weak `until` merge on `s_stream` (next tick)
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
      strong_arms: dict[TagKey, int] = {}
      strong_states: list[Any] = []
      strong_resets: list[Any] = []
      for idx, rec in enumerate(ctx.arm_records):
        tag = rec.arm.values[0]
        strong_arms[tag_key(tag)] = idx
        st_val, rst_val = _fold_transitions(
            rec.strong_transitions, tag, pnr_stream, arms
        )
        strong_states.append(st_val)
        strong_resets.append(rst_val)
      strong_lookup = MergeSelector(is_boolean=False, arms=strong_arms)
      s_stream = switch(
          pns_stream, *strong_states, lookup=strong_lookup, name='state_s'
      )
      if all(_is_static_false(v) for v in strong_resets):
        r_stream: Any = False
      else:
        r_stream = switch(
            pns_stream, *strong_resets, lookup=strong_lookup, name='state_r'
        )
    else:
      s_stream, r_stream = pns_stream, pnr_stream

    ctx.s_stream = s_stream
    ctx.r_stream = Stream(r_stream)

    weak_scopes: list[ArmScope] = []
    weak_envs: list[CapturedEnv] = []
    weak_states: list[Any] = []
    weak_resets: list[Any] = []
    for rec in ctx.arm_records:
      rec.arm_scope.cond = s_stream
      weak_scopes.append(rec.arm_scope)
      if has_weak:
        with push_arm(rec.arm_scope):
          with push_reset(lambda: ctx.r_stream):
            weak_envs.append(CapturedEnv.capture())
            ns_next, nr_next = _fold_transitions(
                rec.weak_transitions, rec.arm.values[0], False, arms
            )
        weak_states.append(ns_next)
        weak_resets.append(nr_next)

    if has_weak:
      ctx.ns_stream = merge_arm_outputs(
          s_stream,
          arms,
          weak_states,
          name='state_ns',
          arm_scopes=weak_scopes,
          arm_envs=weak_envs,
      )
    else:
      ctx.ns_stream = s_stream
    if has_weak_reset:
      ctx.nr_stream = merge_arm_outputs(
          s_stream,
          arms,
          weak_resets,
          name='state_nr',
          arm_scopes=weak_scopes,
          arm_envs=weak_envs,
      )
    finalize_named_outputs(ctx, s_stream, arms)


@contextlib.contextmanager
def state(
    tag: Any,
    *unless_clauses: _Transition,
) -> Iterator[Stream]:
  """Context manager declaring a discrete state inside `with automaton(...)`."""
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
