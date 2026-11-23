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
"""Psyre match context manager and shared arm-lowering infrastructure.

Provides clocked pattern matching (`match`/`case`/`default`), shared inter-arm
memory (`last`), and the Stage-1 arm lowering pass (`_lower_arms`) translating
scoped branches to complementary `when`/`whennot` and `merge` equations
following Lucid Synchrone v3 and Heptagon/BZR (Colaço, Pagano & Pouzet,
*EMSOFT* 2005).
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import enum
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
    WhenSelector,
    active_block,
    arm_stack,
    emit_when as emit_when_node,
    fby,
    is_namedtuple,
    is_thunk,
    matches_tag,
    merge,
    prepare_iterated_n,
    push_arm,
    reject_sentinel,
    require_open_arm_context,
    selector_from_norm,
    tag_key,
    when,
)
from psyre.common_payload import (
    contains_stream_or_thunk,
    is_nan_scalar,
    is_numpy_bool,
    reject_callable,
)
from psyre.graph import Graph, Node
from psyre.stream import Stream, resolve_stream, same_stream_or_const

# pylint: enable=g-importing-member,g-multiple-import


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
  confined: list[tuple[Stream, tuple[ArmScope, ...], str]] = [
      (stream, req_arms, 'Arm-local last()')
      for stream, req_arms in builder.scoped_lasts
  ]
  confined.extend(
      (node.output, node.arms, 'Arm-local stream')
      for node in graph.nodes
      if node.arms
  )
  if not confined:
    return
  output_ids = {s.id for s in graph.output_streams()}
  for stream, req_arms, label in confined:
    if stream.id in output_ids:
      raise RuntimeError(
          f'{label} cannot escape its case()/state() scope '
          'via return or program output.'
      )
    for consumer in graph.consumers(stream):
      merge_stacks = consumer.merge_arms
      c_arms = consumer.arms
      for port_idx, in_s in enumerate(consumer.inputs):
        if in_s is not stream:
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
      for r_idx, r_s in enumerate(consumer.resets):
        if r_s is stream:
          _require_arm_scope(
              consumer.arms_at_reset(r_idx), req_arms, label, consumer
          )


_StepKey = tuple[int, bool, frozenset[TagKey], bool]


def _streams_differ(
    a: Sequence[Stream], b: Sequence[Stream]
) -> bool:
  """Returns True if two stream sequences differ in length or stream identity."""
  return len(a) != len(b) or any(x is not y for x, y in zip(a, b))


def _lower_arms(graph: Graph) -> None:
  """Lowers enclosing-clock arm inputs and resets to `when` subclock nodes."""
  # Three passes over the finalized (thunk-resolved) Stage-1 graph:
  #   1. Instantaneous DFS over initial nodes, sampling slower-clock inputs via
  #      `when`/`whennot` (`when_cache`) and anchoring each subclocked stateful
  #      node with `base_false` when no reset carries its full arm depth.
  #   2. Delayed back-edges of `fby` (`inputs[1]`) and `pre` (`inputs[0]`),
  #      deferred until Pass 1 records `stream_steps` for every node.
  #   3. Nested `ArmScope` condition alignment so `node.arms` stays synced with
  #      inner `merge` selector streams after `optimize_graph` folds.
  if not any(node.arms or node.merge_arms for node in graph.nodes):
    return

  def resolve_scope_cond(scope: ArmScope) -> Stream:
    scope.cond = resolve_stream(scope.cond)
    return scope.cond

  def selector_step(
      sel_s: Stream, selector: WhenSelector | None, *, op: str = 'when'
  ) -> _StepKey | None:
    norm = graph.normalize_when_selector(sel_s, selector, op=op)
    if graph.is_always_active_when(sel_s, selector, op=op, norm=norm):
      return None
    return (sel_s.id, *norm)

  def arm_step(scope: ArmScope) -> _StepKey | None:
    return selector_step(resolve_scope_cond(scope), scope.selector)

  def scopes_to_steps(scopes: Sequence[ArmScope]) -> tuple[_StepKey, ...]:
    steps = [arm_step(scope) for scope in scopes]
    return tuple(step for step in steps if step is not None)

  stream_steps: dict[int, tuple[_StepKey, ...]] = {
      node.output.id: scopes_to_steps(node.arms) for node in graph.nodes
  }
  when_cache: dict[
      tuple[int, tuple[_StepKey, ...]],
      list[tuple[Any, Stream, tuple[str, ...], Stream]],
  ] = {}
  base_false: Stream | None = None
  initial_nodes = list(graph.nodes)
  visited: set[int] = set()

  def visit_stream(s: Stream) -> None:
    prod = graph.producer(s)
    if prod is not None and prod.id not in visited:
      visit(prod)

  def emit_when(
      arm: ArmScope,
      norm_sel: tuple[bool, frozenset[TagKey], bool],
      val_stream: Stream,
      arm_prefix: tuple[ArmScope, ...],
      raw_pfxs: tuple[tuple[str, int], ...],
  ) -> Stream:
    raw_cond = resolve_scope_cond(arm)
    visit_stream(raw_cond)
    cond_s = (
        lower_stream(raw_cond, arm_prefix, raw_pfxs)
        if arm_prefix
        else raw_cond
    )
    pfx_names = tuple(p for p, _ in raw_pfxs)
    bucket = when_cache.setdefault((cond_s.id, scopes_to_steps(arm_prefix)), [])
    for c_sel, c_val, c_pfxs, c_out in bucket:
      if (
          c_sel == norm_sel
          and c_pfxs == pfx_names
          and same_stream_or_const(c_val, val_stream)
      ):
        return c_out
    sel_obj = selector_from_norm(norm_sel, fallback=arm.selector)
    with CapturedEnv(arms=arm_prefix, prefixes=raw_pfxs).restore():
      out_stream = emit_when_node(cond_s, val_stream, sel_obj)
      stream_steps[out_stream.id] = scopes_to_steps((*arm_prefix, arm))
      bucket.append((norm_sel, val_stream, pfx_names, out_stream))
      return out_stream

  def lower_stream(
      stream: Stream,
      target_arms: tuple[ArmScope, ...],
      raw_pfxs: tuple[tuple[str, int], ...],
  ) -> Stream:
    if graph.is_polymorphic_static(stream):
      return stream
    visit_stream(stream)
    steps = stream_steps.get(stream.id, ())
    active_arms = [
        (idx, arm, step)
        for idx, arm in enumerate(target_arms)
        if (step := arm_step(arm)) is not None
    ]
    target_steps = tuple(step for _, _, step in active_arms)
    d = len(steps)
    if d < len(target_steps) and target_steps[:d] == steps:
      prod = graph.producer(stream)
      prod_pfxs = prod.prefixes if prod is not None else ()
      for idx, arm, step in active_arms[d:]:
        # Keep the synthetic `when` node placed inside the narrowest enclosing
        # visualization hierarchy shared by the consumer (`raw_pfxs`) and either
        # the arm or producer (`prod_pfxs`).
        arm_pfxs = (
            tuple(p for p in arm.prefixes if p in raw_pfxs) or arm.prefixes
        )
        if (
            len(prod_pfxs) > len(arm_pfxs)
            and raw_pfxs[: len(prod_pfxs)] == prod_pfxs
        ):
          arm_pfxs = prod_pfxs
        stream = emit_when(
            arm,
            (step[1], step[2], step[3]),
            stream,
            target_arms[:idx],
            arm_pfxs,
        )
    return stream

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
    if merge_stacks is not None and node.is_primitive('merge'):
      if new_inputs:
        new_inputs[0] = lower_stream(new_inputs[0], node_arms, raw_pfxs)
      for arm_idx, arm_stack_entry in enumerate(merge_stacks):
        port = arm_idx + 1
        if port < len(new_inputs):
          new_inputs[port] = lower_stream(
              new_inputs[port], arm_stack_entry, raw_pfxs
          )
    elif node_arms and not node.is_primitive('pre'):
      inst_len = 1 if node.is_primitive('fby') else len(new_inputs)
      for i in range(min(inst_len, len(new_inputs))):
        new_inputs[i] = lower_stream(new_inputs[i], node_arms, raw_pfxs)

    new_resets: list[Stream] = []
    if node.is_stateful and (node.resets or node_arms):
      has_full_depth = False
      for r_idx, r_s in enumerate(node.resets):
        r_arms = node.arms_at_reset(r_idx)
        r_pfxs = node.prefixes_at_reset(r_idx)
        new_resets.append(lower_stream(r_s, r_arms, r_pfxs))
        has_full_depth = has_full_depth or (
            not graph.is_polymorphic_static(r_s)
            and len(r_arms) == len(node_arms)
        )
      if node_arms and not has_full_depth and scopes_to_steps(node_arms):
        # Anchor activation-clock cadence on stateful subclocked nodes so
        # `Scheduler` delivers `Absent` (SYNC) / finite EOF (KAHN) even when
        # all user resets live on outer clocks or are polymorphic static.
        if base_false is None:
          base_false = Stream(False)
          graph.pin_base_clock(base_false)
        new_resets.append(lower_stream(base_false, node_arms, raw_pfxs))

    if _streams_differ(new_inputs, orig_inputs) or _streams_differ(
        new_resets, node.resets
    ):
      graph.rewire_node(
          node, inputs=tuple(new_inputs), resets=tuple(new_resets)
      )

    # Zero-input stateless nodes (e.g., `random_stream()`) carry no input to
    # emit `Absent`, so they run on `BASE_CLOCK` and are sampled downstream.
    base_steps = (
        scopes_to_steps(node_arms) if (node.is_stateful or node.inputs) else ()
    )
    has_concrete_input = False
    for h in node.rate_inputs():
      if not graph.is_polymorphic_static(h):
        has_concrete_input = True
        h_steps = stream_steps.get(h.id, ())
        if len(h_steps) >= len(base_steps):
          base_steps = h_steps
          break
    if not has_concrete_input and node.inherits_reset_clock() and new_resets:
      rst_steps = [
          r_steps
          for r in new_resets
          if not graph.is_polymorphic_static(r)
          and len(r_steps := stream_steps.get(r.id, ())) > len(base_steps)
          and r_steps[: len(base_steps)] == base_steps
      ]
      if rst_steps:
        base_steps = max(rst_steps, key=len)
    extra_step: _StepKey | None = None
    if node.is_primitive('when', 'whennot') and orig_inputs:
      extra_step = selector_step(
          orig_inputs[0], node.kwargs.get('selector'), op=node.op
      )
    elif node.is_primitive('buffer') and len(orig_inputs) >= 3:
      extra_step = selector_step(orig_inputs[2], None)
    stream_steps[node.output.id] = (
        base_steps if extra_step is None else base_steps + (extra_step,)
    )

  # Pass 1: lower instantaneous inputs/resets in topological-like DFS order.
  for node in initial_nodes:
    visit(node)

  # Pass 2: lower delayed inputs of `fby` and `pre`:
  for node in initial_nodes:
    if not node.is_primitive('fby', 'pre') or not node.arms:
      continue
    delayed_port = 0 if node.is_primitive('pre') else 1
    if delayed_port >= len(node.inputs):
      continue
    delayed_s = node.inputs[delayed_port]
    delayed_new = lower_stream(delayed_s, node.arms, node.prefixes)
    if delayed_new is not delayed_s:
      new_inputs = list(node.inputs)
      new_inputs[delayed_port] = delayed_new
      graph.rewire_node(node, inputs=tuple(new_inputs), resets=node.resets)

  # Pass 3: lower nested dynamic `ArmScope` selectors onto their enclosing arm
  # prefix clocks so later optimizations (such as folding
  # `when(Stream(True), seq)` to polymorphic `seq`) keep `node.arms` aligned
  # with the corresponding inner `merge` selector stream.
  scope_cache: dict[ArmScope, ArmScope] = {}
  node_idx = 0
  while node_idx < len(graph.nodes):
    node = graph.nodes[node_idx]
    node_idx += 1
    if len(node.arms) <= 1:
      continue
    orig_arms = node.arms
    new_arms: list[ArmScope] = [orig_arms[0]]
    arms_changed = False
    for idx in range(1, len(orig_arms)):
      scope = orig_arms[idx]
      cached = scope_cache.get(scope)
      if cached is None:
        raw_cond = resolve_scope_cond(scope)
        cond_s = lower_stream(raw_cond, orig_arms[:idx], scope.prefixes)
        cached = (
            scope
            if cond_s is raw_cond
            else dataclasses.replace(scope, cond=cond_s)
        )
        scope_cache[scope] = cached
      if cached is not scope:
        arms_changed = True
      new_arms.append(cached)
    if arms_changed:
      node.arms = tuple(new_arms)

  # All arm merges have been lowered to explicit `when`/`whennot` nodes; clear
  # temporary `merge_arms` stacks so finalized nodes drop stale references.
  for node in graph.nodes:
    node.merge_arms = None


def _on_finalize(builder: GraphBuilder) -> None:
  """Validates arm confinement and lowers arm inputs/resets to `when`."""
  _check_arm_confinement(builder)
  _lower_arms(builder.graph)


GraphBuilder.on_finalize = _on_finalize


def validate_discrete_tag(tag: Any, caller: str) -> None:
  """Validates that tag is a discrete, hashable, non-NaN pattern tag."""
  if isinstance(tag, Stream) or is_thunk(tag):
    raise TypeError(
        f'{caller} pattern tag must be a discrete constant value, got {tag!r}.'
    )
  reject_sentinel(tag, label=f'{caller} pattern tag')
  reject_callable(tag, label=f'{caller} pattern tag')
  if is_nan_scalar(tag):
    raise ValueError(
        f'{caller} pattern tag cannot be NaN (never matches via ==).'
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
  if caller_arms != last_env.arms and (
      curr_builder := GraphBuilder.current()
  ) is not None:
    curr_builder.scoped_lasts.append((mem_stream, caller_arms))
  return mem_stream


def validate_exhaustiveness(
    arms: Sequence[MatchArm],
    *,
    construct: str,
) -> None:
  """Validates finite Enum/bool domain coverage when no wildcard arm exists."""
  if any(arm.is_wildcard for arm in arms):
    return
  claimed: list[Any] = []
  for arm in arms:
    claimed.extend(arm.values)
  if not claimed:
    return
  domain: Sequence[Any] = ()
  if (
      all(isinstance(t, enum.Enum) for t in claimed)
      and len({type(t) for t in claimed}) == 1
  ):
    domain = list(type(claimed[0]))
  elif all(isinstance(t, bool) or is_numpy_bool(t) for t in claimed):
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


def bind_thunk_env(
    val: Any,
    env: CapturedEnv,
) -> Callable[[], Stream]:
  """Wraps a Stage-1 thunk so forcing it restores `env` and returns a Stream."""

  def evaluate() -> Stream:
    with env.restore():
      return resolve_stream(val)

  return evaluate


def merge_arm_outputs(
    selector_stream: Any,
    arms: Sequence[MatchArm],
    arm_values: Sequence[Any],
    name: str,
    arm_scopes: Sequence[ArmScope],
    arm_envs: Sequence[CapturedEnv],
) -> Stream:
  """Merges one output stream per arm back onto the enclosing clock."""
  arm_streams: list[Any] = []
  arm_map: dict[TagKey, int] = {}
  default_idx: int | None = None
  encl_arms = arm_stack.get()
  for idx, (arm, out_val, env) in enumerate(
      zip(arms, arm_values, arm_envs, strict=True)
  ):
    if is_thunk(out_val):
      arm_streams.append(bind_thunk_env(out_val, env))
    else:
      with env.restore():
        arm_streams.append(Stream(out_val))
    if arm.is_wildcard:
      default_idx = idx
    else:
      for tag in arm.values:
        arm_map[tag_key(tag)] = idx
  lookup = MergeSelector(
      is_boolean=False, arms=arm_map, default_index=default_idx
  )
  out_stream = merge(selector_stream, *arm_streams, lookup=lookup, name=name)
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
    arm_envs: Sequence[CapturedEnv],
) -> Any:
  """Validates and recursively merges an output attribute across all arms."""
  first = arm_values[0]
  if isinstance(first, tuple):
    cls: Any = type(first)
    is_nt = is_namedtuple(first)
    if {type(val) for val in arm_values} != {cls} or any(
        len(val) != len(first) for val in arm_values
    ):
      kind = 'NamedTuple' if is_nt else 'tuple'
      raise ValueError(
          f'All {construct}() arms must assign matching {kind} types of '
          f'identical length to {out_name!r}.'
      )
    suffixes = (
        first._fields  # pylint: disable=protected-access
        if is_nt
        else (str(i) for i in range(len(first)))
    )
    merged = (
        _merge_arm_values(
            f'{out_name}_{sfx}',
            selector_stream,
            arms,
            col,
            construct=construct,
            arm_scopes=arm_scopes,
            arm_envs=arm_envs,
        )
        for sfx, col in zip(suffixes, zip(*arm_values))
    )
    return cls(*merged) if is_nt else tuple(merged)
  if any(isinstance(val, tuple) for val in arm_values):
    raise ValueError(
        f'Inconsistent {construct}() arms for {out_name!r}: mixing scalar '
        'Stream and structured values.'
    )
  for val in arm_values:
    if isinstance(val, (list, set, frozenset, Mapping)) and (
        contains_stream_or_thunk(val, include_zero_arg_fn=True)
    ):
      raise TypeError(
          f'Unsupported {construct}() output container {type(val).__name__} '
          f'for {out_name!r} containing Stream or thunk leaves; use a tuple '
          'or NamedTuple instead.'
      )
  return merge_arm_outputs(
      selector_stream,
      arms,
      arm_values,
      name=out_name,
      arm_scopes=arm_scopes,
      arm_envs=arm_envs,
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
        arm_envs=[rec.output_envs[out_name] for rec in ctx.arm_records],
    )
  ctx.finalized = True


def _deferred_selector(
    selector: Any, env: CapturedEnv
) -> Callable[[], Stream]:
  """Lazily resolves `selector` once in `env` across all `match` consumers."""
  # Memoizing ensures a raw list selector inside `with reset(r):` emits one
  # resettable `static_stream` node shared by all `case()` guards and `merge`s.
  handle: Stream | None = None

  def get_handle() -> Stream:
    nonlocal handle
    if handle is None:
      with env.restore():
        handle = resolve_stream(selector)
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
  validate_exhaustiveness(arms, construct='match')
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
      if any(matches_tag(tag, c) for c in (*ctx.claimed_tags, *unclaimed)):
        raise ValueError(f'Duplicate case tag {tag!r} in match().')
      unclaimed.append(tag)
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
