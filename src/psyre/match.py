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
"""Psyre match context manager and shared arm-lowering infrastructure."""

from collections.abc import Callable, Iterator, Sequence
import contextlib
import enum
from typing import Any

# pylint: disable=g-importing-member,g-multiple-import
from psyre import kahn_primitives
from psyre import sync_primitives
from psyre.builder import GraphBuilder
from psyre.common import (
    ArmRecord,
    ArmScope,
    BlockContext,
    BlockHandle,
    CapturedEnv,
    MatchArm,
    Semantics,
    TagKey,
    UNSET,
    WhenSelector,
    active_block,
    arm_stack,
    fby,
    is_tautological_selector,
    is_thunk,
    matches_tag,
    merge,
    normalize_when_selector,
    prepare_iterated_n,
    push_arm,
    register_primitives,
    require_open_arm_context,
    same_stream_or_const,
    tag_key,
    when,
    whennot,
)
from psyre.graph import Graph, Node
from psyre.istructure import IStruct, is_one_periodic
from psyre.stream import Stream, resolve_handle

# pylint: enable=g-importing-member,g-multiple-import

register_primitives(Semantics.KAHN, kahn_primitives)
register_primitives(Semantics.SYNC, sync_primitives)


def _require_arm_scope(
    port_stack: tuple[ArmScope, ...],
    req_arms: tuple[ArmScope, ...],
    label: str,
    consumer: Node,
) -> None:
  """Raises RuntimeError if port_stack does not start with req_arms."""
  if port_stack[: len(req_arms)] != req_arms:
    raise RuntimeError(
        f'{label} cannot escape its case()/state() scope '
        f'(consumed by {consumer}).'
    )


def _check_arm_confinement(builder: GraphBuilder) -> None:
  """Validates that arm-local streams do not escape their arm scope."""
  graph = builder.graph
  confined: list[tuple[IStruct, tuple[ArmScope, ...], str]] = [
      (handle.istruct, req_arms, 'Arm-local last()')
      for handle, req_arms in builder.scoped_lasts
  ]
  confined.extend(
      (node.output_stream, node.arms, 'Arm-local stream')
      for node in graph.nodes
      if node.arms
  )
  if not confined:
    return
  output_istructs = set(graph.output_streams())
  for istruct, req_arms, label in confined:
    if istruct in output_istructs:
      raise RuntimeError(
          f'{label} cannot escape its case()/state() scope '
          'via return or program output.'
      )
    for consumer in graph.consumers(istruct):
      merge_stacks = consumer.merge_arms
      c_arms = consumer.arms
      for port_idx, in_h in enumerate(consumer.inputs):
        if in_h.istruct is not istruct:
          continue
        has_merge = (
            merge_stacks is not None and 0 < port_idx <= len(merge_stacks)
        )
        _require_arm_scope(
            merge_stacks[port_idx - 1] if has_merge else c_arms,
            req_arms,
            label,
            consumer,
        )
      for r_idx, r_h in enumerate(consumer.resets):
        if r_h.istruct is istruct:
          r_arms = tuple(a for a in c_arms if a.reset_depth <= r_idx)
          _require_arm_scope(r_arms, req_arms, label, consumer)


_StepKey = tuple[IStruct, bool, frozenset[TagKey], bool]


def _streams_differ(
    a: Sequence[Stream], b: Sequence[Stream]
) -> bool:
  """Returns True if two handle sequences differ in length or stream id."""
  return len(a) != len(b) or any(
      x.istruct is not y.istruct for x, y in zip(a, b)
  )


def _lower_arms(graph: Graph) -> None:
  """Lowers enclosing-clock arm inputs and resets to `when` subclock nodes."""
  if not any(node.arms or node.merge_arms for node in graph.nodes):
    return

  def resolve_scope_cond(scope: ArmScope) -> Stream:
    if is_thunk(scope.cond):
      scope.cond = scope.cond()
    if not isinstance(scope.cond, Stream):
      scope.cond = Stream(scope.cond)
    return scope.cond

  def selector_step(
      sel_s: IStruct, selector: WhenSelector | None, *, op: str = 'when'
  ) -> _StepKey | None:
    norm_sel = normalize_when_selector(
        selector, is_bool=graph.is_bool_stream(sel_s), op=op
    )
    return None if is_tautological_selector(norm_sel) else (sel_s, *norm_sel)

  def arm_step(scope: ArmScope) -> _StepKey | None:
    return selector_step(resolve_scope_cond(scope).istruct, scope.selector)

  def scopes_to_steps(scopes: Sequence[ArmScope]) -> tuple[_StepKey, ...]:
    steps = [arm_step(scope) for scope in scopes]
    return tuple(step for step in steps if step is not None)

  stream_steps: dict[IStruct, tuple[_StepKey, ...]] = {
      node.output_stream: scopes_to_steps(node.arms) for node in graph.nodes
  }
  when_cache: dict[
      IStruct, list[tuple[Any, IStruct, tuple[str, ...], IStruct]]
  ] = {}
  base_false: Stream | None = None
  initial_nodes = list(graph.nodes)
  visited: set[int] = set()

  def visit_stream(s: IStruct) -> None:
    prod = graph.producer(s)
    if prod is not None and prod.id not in visited:
      visit(prod)

  def emit_when(
      arm: ArmScope,
      val_handle: Stream,
      arm_prefix: tuple[ArmScope, ...],
      raw_pfxs: tuple[tuple[str, int], ...],
  ) -> Stream:
    raw_cond_h = resolve_scope_cond(arm)
    raw_cond_s = raw_cond_h.istruct
    visit_stream(raw_cond_s)
    norm_sel = normalize_when_selector(
        arm.selector, is_bool=graph.is_bool_stream(raw_cond_s)
    )
    if is_tautological_selector(norm_sel):
      return val_handle
    cond_h = (
        lower_handle(raw_cond_h, arm_prefix, raw_pfxs)
        if arm_prefix
        else raw_cond_h
    )
    pfx_names = tuple(p for p, _ in raw_pfxs)
    val_s = val_handle.istruct
    bucket = when_cache.setdefault(cond_h.istruct, [])
    for c_sel, c_val, c_pfxs, c_out in bucket:
      if (
          c_sel == norm_sel
          and c_pfxs == pfx_names
          and same_stream_or_const(c_val, val_s)
      ):
        return Stream(c_out)
    is_bool, vals, _ = norm_sel
    with CapturedEnv(arms=arm_prefix, prefixes=raw_pfxs).restore():
      cond_s = Stream(cond_h.istruct)
      val_stream = Stream(val_s)
      if is_bool and vals == frozenset({(True, False)}):
        out_stream = whennot(cond_s, val_stream, name='whennot')
      elif arm.selector is None or (
          is_bool and vals == frozenset({(True, True)})
      ):
        out_stream = when(cond_s, val_stream, name='when')
      else:
        out_stream = when(
            cond_s, val_stream, selector=arm.selector, name='when'
        )
      out_istruct = out_stream.istruct
      stream_steps[out_istruct] = scopes_to_steps((*arm_prefix, arm))
      bucket.append((norm_sel, val_s, pfx_names, out_istruct))
      return Stream(out_istruct)

  def lower_handle(
      handle: Stream,
      target_arms: tuple[ArmScope, ...],
      raw_pfxs: tuple[tuple[str, int], ...],
  ) -> Stream:
    visit_stream(handle.istruct)
    steps = stream_steps.get(handle.istruct, ())
    active_arms = [
        (idx, arm, step)
        for idx, arm in enumerate(target_arms)
        if (step := arm_step(arm)) is not None
    ]
    target_steps = tuple(step for _, _, step in active_arms)
    d = len(steps)
    if d < len(target_steps) and target_steps[:d] == steps:
      prod = graph.producer(handle.istruct)
      prod_pfxs = prod.prefixes if prod is not None else ()
      for idx, arm, _ in active_arms[d:]:
        arm_pfxs = (
            tuple(p for p in arm.prefixes if p in raw_pfxs) or arm.prefixes
        )
        if (
            len(prod_pfxs) > len(arm_pfxs)
            and raw_pfxs[: len(prod_pfxs)] == prod_pfxs
        ):
          arm_pfxs = prod_pfxs
        handle = emit_when(arm, handle, target_arms[:idx], arm_pfxs)
    return handle

  def visit(node: Node) -> None:
    nonlocal base_false
    if node.id in visited:
      return
    visited.add(node.id)

    node_arms = node.arms
    merge_stacks = node.merge_arms
    raw_pfxs = node.prefixes
    for s in node.instantaneous_input_streams():
      visit_stream(s)

    orig_inputs = node.inputs
    new_inputs = list(orig_inputs)
    if merge_stacks is not None and node.op == 'merge':
      if new_inputs and not is_one_periodic(new_inputs[0]):
        new_inputs[0] = lower_handle(new_inputs[0], node_arms, raw_pfxs)
      for arm_idx, arm_stack_entry in enumerate(merge_stacks):
        port = arm_idx + 1
        if port < len(new_inputs) and not is_one_periodic(new_inputs[port]):
          new_inputs[port] = lower_handle(
              new_inputs[port], arm_stack_entry, raw_pfxs
          )
    elif node_arms and node.op != 'pre':
      inst_inputs = new_inputs[: 1 if node.op == 'fby' else len(new_inputs)]
      ports = [i for i, h in enumerate(inst_inputs) if not is_one_periodic(h)]
      if not ports and inst_inputs and not node.is_stateful:
        ports = [0]
      for i in ports:
        new_inputs[i] = lower_handle(new_inputs[i], node_arms, raw_pfxs)

    new_resets = list(node.resets)
    if node.is_stateful and (node.resets or node_arms):
      new_resets = []
      has_full_depth = False
      for r_idx, r_h in enumerate(node.resets):
        r_arms = tuple(a for a in node_arms if a.reset_depth <= r_idx)
        r_pfxs = tuple(p for p in raw_pfxs if p[1] <= r_idx)
        new_resets.append(lower_handle(r_h, r_arms, r_pfxs))
        has_full_depth = has_full_depth or len(r_arms) == len(node_arms)
      if node_arms and not has_full_depth:
        if base_false is None:
          base_false = Stream([False], period=[False])
        new_resets.append(lower_handle(base_false, node_arms, raw_pfxs))

    if _streams_differ(new_inputs, orig_inputs) or _streams_differ(
        new_resets, node.resets
    ):
      graph.rewire_node(
          node, inputs=tuple(new_inputs), resets=tuple(new_resets)
      )

    if node.op == 'merge':
      rate_inputs: Sequence[Stream] = new_inputs[:1]
    elif node.op == 'buffer':
      rate_inputs = new_inputs[1:3]
    else:
      rate_inputs = new_inputs
    base_steps = scopes_to_steps(node_arms) if node.is_stateful else ()
    for h in rate_inputs:
      if not is_one_periodic(h):
        h_steps = stream_steps.get(h.istruct, ())
        if len(h_steps) >= len(base_steps):
          base_steps = h_steps
          break
    extra_step: _StepKey | None = None
    if node.op in {'when', 'whennot'} and orig_inputs:
      extra_step = selector_step(
          orig_inputs[0].istruct, node.kwargs.get('selector'), op=node.op
      )
    elif node.op == 'buffer' and len(orig_inputs) >= 3:
      extra_step = selector_step(orig_inputs[2].istruct, None)
    stream_steps[node.output_stream] = (
        base_steps if extra_step is None else base_steps + (extra_step,)
    )

  for node in initial_nodes:
    visit(node)

  # Pass 2: lower delayed inputs of `fby` and `pre`:
  for node in initial_nodes:
    if node.op not in {'fby', 'pre'} or not node.arms:
      continue
    delayed_port = 0 if node.op == 'pre' else 1
    if delayed_port >= len(node.inputs):
      continue
    delayed_h = node.inputs[delayed_port]
    if is_one_periodic(delayed_h):
      continue
    delayed_new = lower_handle(delayed_h, node.arms, node.prefixes)
    if delayed_new.istruct is not delayed_h.istruct:
      new_inputs = list(node.inputs)
      new_inputs[delayed_port] = delayed_new
      graph.rewire_node(node, inputs=tuple(new_inputs), resets=node.resets)


def _on_finalize(builder: GraphBuilder) -> None:
  """Validates arm confinement and lowers arm inputs/resets to `when`."""
  _check_arm_confinement(builder)
  _lower_arms(builder.graph)


GraphBuilder.on_finalize = _on_finalize


def validate_discrete_tag(tag: Any, caller: str) -> None:
  """Validates that tag is a discrete, hashable pattern tag."""
  if isinstance(tag, Stream) or is_thunk(tag):
    raise TypeError(
        f'{caller} pattern tag must be a discrete constant value, got {tag!r}.'
    )
  try:
    hash(tag)
  except TypeError as err:
    raise TypeError(f'{caller} pattern tag {tag!r} must be hashable.') from err


def last(
    source: Any,
    init: Any = None,
    n: int = 1,
    *,
    name: str = 'last',
) -> Stream:
  """Returns a Stream yielding the n-th previous defined value of `source`."""
  prepare_iterated_n(n, 'last')
  ctx = active_block.get()
  if ctx is None:
    raise RuntimeError('last() can only be used inside match() or automaton().')
  last_env = ctx.enclosing_env
  caller_arms = arm_stack.get()
  with last_env.restore():
    mem_stream = fby(init, source, n=n, name=name)
  if caller_arms is not last_env.arms and (
      curr_builder := GraphBuilder.current()
  ) is not None:
    curr_builder.scoped_lasts.append((mem_stream, caller_arms))
  return mem_stream


def validate_exhaustiveness(
    arms: Sequence[MatchArm],
    init_mode: Any,
    *,
    construct: str,
) -> None:
  """Validates finite Enum/bool domain coverage when no wildcard arm exists."""
  if any(arm.is_wildcard for arm in arms):
    return
  claimed: list[Any] = []
  for arm in arms:
    claimed.extend(arm.values)
  tags = (
      [*claimed, init_mode]
      if init_mode is not UNSET and not isinstance(init_mode, Stream)
      else claimed
  )
  if not tags:
    return
  domain: Sequence[Any] = ()
  if (
      all(isinstance(t, enum.Enum) for t in tags)
      and len({type(t) for t in tags}) == 1
  ):
    domain = list(type(tags[0]))
  elif all(isinstance(t, bool) for t in tags):
    domain = (True, False)
  missing = [d for d in domain if not any(matches_tag(d, c) for c in claimed)]
  if missing:
    raise ValueError(
        f'Non-exhaustive {construct}(): missing cases {missing!r}.'
    )


def make_arm_selector(
    arm: MatchArm,
    excluded_for_wildcard: tuple[Any, ...] = (),
) -> WhenSelector:
  """Creates the WhenSelector predicate matching a single arm."""
  raw_tags = excluded_for_wildcard if arm.is_wildcard else arm.values
  return WhenSelector(
      is_boolean=False,
      values=frozenset(tag_key(t) for t in raw_tags),
      negated=arm.is_wildcard,
  )


def _wrap_arm_thunk(
    fn: Callable[[], Any], arm_scope: ArmScope
) -> Callable[[], Any]:
  """Returns a distinct zero-arg thunk evaluating fn under arm_scope."""

  def evaluate() -> Any:
    with push_arm(arm_scope):
      return fn()

  return evaluate


def merge_arm_outputs(
    selector_stream: Any,
    arms: Sequence[MatchArm],
    arm_values: Sequence[Any],
    name: str,
    arm_scopes: Sequence[ArmScope],
) -> Stream:
  """Merges one output stream per arm back onto the enclosing clock."""
  branches: dict[Any, Any] = {}
  default_val: Any = None
  encl_arms = arm_stack.get()
  for arm, out_val, arm_scope in zip(arms, arm_values, arm_scopes, strict=True):
    aligned: Any = (
        _wrap_arm_thunk(out_val, arm_scope)
        if is_thunk(out_val)
        else Stream(out_val)
    )
    if arm.is_wildcard:
      default_val = aligned
    else:
      for tag in arm.values:
        branches[tag] = aligned
  out_stream = merge(selector_stream, branches, default=default_val, name=name)
  if (curr_builder := GraphBuilder.current()) is not None:
    merge_node = curr_builder.graph.producer(out_stream)
    if merge_node is not None:
      merge_node.merge_arms = tuple(
          encl_arms + (scope,) for scope in arm_scopes
      )
  return out_stream


def _merge_arm_values(
    out_name: str,
    selector_stream: Any,
    arms: Sequence[MatchArm],
    arm_values: Sequence[Any],
    *,
    construct: str,
    arm_scopes: Sequence[ArmScope],
) -> Any:
  """Validates and recursively merges an output attribute across all arms."""
  first = arm_values[0]
  if isinstance(first, tuple):
    cls: Any = type(first)
    if {type(val) for val in arm_values} != {cls} or any(
        len(val) != len(first) for val in arm_values
    ):
      kind = 'tuple' if cls is tuple else 'NamedTuple'
      raise ValueError(
          f'All {construct}() arms must assign matching {kind} types of '
          f'identical length to {out_name!r}.'
      )
    suffixes = (
        (str(i) for i in range(len(first)))
        if cls is tuple
        else first._fields  # pylint: disable=protected-access
    )
    merged = (
        _merge_arm_values(
            f'{out_name}_{sfx}',
            selector_stream,
            arms,
            col,
            construct=construct,
            arm_scopes=arm_scopes,
        )
        for sfx, col in zip(suffixes, zip(*arm_values))
    )
    return tuple(merged) if cls is tuple else cls(*merged)
  if any(isinstance(val, tuple) for val in arm_values):
    raise ValueError(
        f'Inconsistent {construct}() arms for {out_name!r}: mixing scalar '
        'Stream and structured values.'
    )
  return merge_arm_outputs(
      selector_stream, arms, arm_values, name=out_name, arm_scopes=arm_scopes
  )


def finalize_named_outputs(
    ctx: BlockContext,
    selector_stream: Any,
    arms: Sequence[MatchArm],
) -> None:
  """Validates consistent output names across arms and merges each output."""
  expected_keys = list(ctx.arm_records[0].outputs.keys())
  expected_set = set(expected_keys)
  for rec in ctx.arm_records[1:]:
    actual_set = set(rec.outputs.keys())
    if actual_set != expected_set:
      missing = [key for key in expected_keys if key not in actual_set]
      extra = [key for key in rec.outputs if key not in expected_set]
      arm_desc = (
          'default()'
          if rec.arm.is_wildcard
          else ', '.join(repr(val) for val in rec.arm.values)
      )
      raise ValueError(
          f'Inconsistent {ctx.kind}() outputs across arms: arm '
          f'{arm_desc} missing {missing!r}, unexpected {extra!r} '
          f'(expected {expected_keys!r}).'
      )
  arm_scopes = [rec.arm_scope for rec in ctx.arm_records]
  for out_name in expected_keys:
    ctx.merged_outputs[out_name] = _merge_arm_values(
        out_name,
        selector_stream,
        arms,
        [rec.outputs[out_name] for rec in ctx.arm_records],
        construct=ctx.kind,
        arm_scopes=arm_scopes,
    )
  ctx.finalized = True


def _deferred_selector(
    selector: Any, env: CapturedEnv
) -> Callable[[], Stream]:
  """Lazily resolves selector once in env across all match consumers."""
  handle: Stream | None = None

  def get_handle() -> Stream:
    nonlocal handle
    if handle is None:
      with env.restore():
        handle = resolve_handle(selector)
    return handle

  return get_handle


@contextlib.contextmanager
def match(selector: Any) -> Iterator[BlockHandle]:
  """Context manager for plain combinational stream pattern matching."""
  enclosing_env = CapturedEnv.capture()
  ctx = BlockContext(
      kind='match',
      enclosing_env=enclosing_env,
      active_selector=_deferred_selector(selector, enclosing_env),
  )
  handle = BlockHandle(ctx)
  block_token = active_block.set(ctx)
  try:
    yield handle
  finally:
    active_block.reset(block_token)

  if not ctx.arm_records:
    raise ValueError('match() requires at least one case().')
  arms = [rec.arm for rec in ctx.arm_records]
  validate_exhaustiveness(arms, UNSET, construct='match')
  finalize_named_outputs(ctx, ctx.active_selector, arms)


@contextlib.contextmanager
def case(*tags: Any) -> Iterator[Stream]:
  """Context manager declaring a combinational arm inside `with match(...)`."""
  ctx = require_open_arm_context('match', 'case()')
  if ctx.has_wildcard:
    raise ValueError('Unreachable case() after wildcard arm in match().')

  if not tags:
    arm = MatchArm(values=(), is_wildcard=True)
    excluded = tuple(ctx.claimed_tags)
    ctx.has_wildcard = True
  else:
    unclaimed: list[Any] = []
    for tag in tags:
      validate_discrete_tag(tag, 'case()')
      if not any(matches_tag(tag, c) for c in (*ctx.claimed_tags, *unclaimed)):
        unclaimed.append(tag)
    if not unclaimed:
      raise ValueError(f'Duplicate or unreachable case {tags!r} in match().')
    ctx.claimed_tags.extend(unclaimed)
    arm = MatchArm(values=tuple(unclaimed), is_wildcard=False)
    excluded = ()

  arm_sel = make_arm_selector(arm, excluded)
  arm_scope = ArmScope(cond=ctx.active_selector, selector=arm_sel)
  sampled = when(
      ctx.active_selector,
      ctx.active_selector,
      selector=arm_sel,
      name='case',
  )
  arm_record = ArmRecord(arm=arm, arm_scope=arm_scope)
  ctx.active_arm = arm_record
  try:
    with push_arm(arm_scope):
      yield sampled
  finally:
    ctx.active_arm = None
  ctx.arm_records.append(arm_record)


@contextlib.contextmanager
def default() -> Iterator[Stream]:
  """Context manager declaring the trailing wildcard arm of `with match(...)`."""
  require_open_arm_context('match', 'default()')
  with case() as sampled:
    yield sampled
