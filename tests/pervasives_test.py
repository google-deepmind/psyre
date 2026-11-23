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
"""Equivalence tests comparing Psyre primitive ops with derived pervasives."""

from collections.abc import Callable, Sequence
import dataclasses
import logging
import operator
from typing import Any

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import (
    CausalityError,
    check_causality,
)
from psyre.automaton import (
    automaton,
    state,
    until,
)
from psyre.builder import build_graph
from psyre.clock import (
    ClockError,
    check_clocks,
)
from psyre.common import (
    Absent,
    Bottom,
    Semantics,
    buffer,
    counter,
    current,
    fby,
    ifthenelse,
    is_thunk,
    logger,
    merge,
    morph,
    post,
    pre,
    prefix_sum,
    prepare_iterated_n,
    psyre_comb,
    reset,
    scan,
    switch,
    when,
    whennot,
)
from psyre.match import (
    case,
    match,
)
from psyre.scheduler import Scheduler
from psyre.static_stream import StaticWord
from psyre.stream import Stream
from psyre.trace import present_values

# pylint: enable=g-importing-member,g-multiple-import


def share_stream(spec: Any) -> Any:
  """Shares a single Stream across multiple internal consumer ports."""
  if is_thunk(spec):
    return spec
  return spec if isinstance(spec, Stream) else Stream(spec)


def _iterate(
    n: int,
    op_name: str,
    val: Any,
    step: Callable[[Any, str], Stream],
    name: str,
) -> Stream:
  """Applies `step(acc, step_name)` n times, naming the final node `name`."""
  prepare_iterated_n(n, op_name)
  out: Any = val
  for i in range(n):
    out = step(out, name if i == n - 1 else op_name)
  return out


def periodic_stream_pervasive(
    prefix: Sequence[Any], period: Sequence[Any], name: str = ''
) -> Stream:
  """Derived periodic stream constructed from Stream(prefix, period=...)."""
  return Stream(
      prefix if isinstance(prefix, StaticWord) else list(prefix),
      period=period if isinstance(period, StaticWord) else list(period),
      name=name,
  )


def pre_pervasive(x: Any, n: int = 1, name: str = 'pre') -> Stream:
  """Derived stateful delay initialized to None, implemented via fby."""
  return _iterate(n, 'pre', x, lambda acc, nm: fby(None, acc, name=nm), name)


def fby_pervasive(
    i_stream: Any, j_stream: Any, n: int = 1, name: str = 'fby'
) -> Stream:
  """Derived iterated followed-by delay implemented by chaining `fby` n times."""
  shared_init = share_stream(i_stream)
  return _iterate(
      n, 'fby', j_stream, lambda acc, nm: fby(shared_init, acc, name=nm), name
  )


def post_pervasive(x: Any, n: int = 1, name: str = 'post') -> Stream:
  """Derived iterated lookahead implemented by chaining `post` n times."""
  return _iterate(n, 'post', x, lambda acc, nm: post(acc, name=nm), name)


def morph_pervasive(
    x: Any, y: Any, n: int = 1, name: str = 'morph'
) -> Stream:
  """Derived initialization operator emitting first n values of x then y."""
  prepare_iterated_n(n, 'morph')
  first = fby(True, False, n=n)
  return merge(first, when(first, x), whennot(first, y), name=name)


def current_pervasive(
    b: Any, i: Any, default: Any = None, name: str = 'current'
) -> Stream:
  """Derived same-clock sample-and-hold of stream i gated by boolean b."""
  c = merge(b, when(b, i), whennot(b, fby(default, lambda: c)), name=name)
  return c


def counter_pervasive(
    start: Any = 0,
    max_val: Any = None,
    tick: Any = True,
    name: str = 'counter',
) -> Stream:
  """Derived resettable incrementing integer stream enabled by tick."""
  cur = fby(
      start,
      lambda: acc,
      name=name if max_val is None else (f'{name}_fby' if name else 'fby'),
  )
  if max_val is not None:
    cur = cur.min(max_val, name=name)
  acc = (
      cur + 1
      if isinstance(tick, bool) and tick
      else ifthenelse(tick, cur + 1, cur)
  )
  return cur


def prefix_sum_pervasive(
    data: Any, start: Any = 0, name: str = 'prefix_sum'
) -> Stream:
  """Derived resettable running sum of data starting from start."""
  acc = fby(start, lambda: tmp, name=f'{name}_acc' if name else 'fby')
  tmp = acc + data
  tmp.name = name
  return tmp


def scan_pervasive(
    data: Any, op: Callable[[Any, Any], Any] = operator.add, name: str = 'scan'
) -> Stream:
  """Derived cumulative fold of binary operator op across data."""
  tmp = fby(data, lambda: acc)
  acc = morph_pervasive(data, op(tmp, data), name=name)
  return acc


def _slot_thunk(slots: Sequence[Any], idx: int) -> Callable[[], Any]:
  """Returns a zero-argument thunk reading slots[idx] after loop binding."""
  return lambda: slots[idx]


@psyre_comb
def check_buffer_pull(overflow: bool, val: Any) -> Any:
  """Raises `BufferError` on overflow; otherwise forwards `val`."""
  if overflow:
    raise BufferError('RingBuffer overflow: pull into occupied slot.')
  return val


@psyre_comb
def check_buffer_push(valid: bool, val: Any) -> Any:
  """Raises `BufferError` on underflow; otherwise forwards `val`."""
  if not valid:
    raise BufferError(
        'RingBuffer underflow: push from empty slot before matching pull.'
    )
  return val


def buffer_pervasive(
    i_stream: Any,
    i_p_stream: Any,
    o_p_stream: Any,
    size: int = 1,
    name: str = 'buffer',
) -> Stream:
  """Derived bounded lossless FIFO buffer implemented from core primitives."""
  if size < 1:
    raise ValueError(f'buffer size must be >= 1, got {size}.')
  i_p = share_stream(i_p_stream)
  o_p = share_stream(o_p_stream)
  in_val = merge(i_p, i_stream, None)

  if size == 1:
    slot_next: list[Any] = [None, None]
    prev_slot = fby(None, lambda: slot_next[0])
    prev_valid = fby(False, lambda: slot_next[1])
    checked_in = check_buffer_pull(
        ifthenelse(i_p, prev_valid, False), in_val
    )
    after_pull = ifthenelse(i_p, checked_in, prev_slot)
    after_valid = ifthenelse(i_p, True, prev_valid)
    slot_next[0] = ifthenelse(o_p, None, after_pull)
    slot_next[1] = ifthenelse(o_p, False, after_valid)
    return check_buffer_push(
        when(o_p, after_valid), when(o_p, after_pull), name=name
    )

  idx_next: list[Any] = [None, None]
  pull_idx = fby(0, lambda: idx_next[0])
  idx_next[0] = ifthenelse(i_p, (pull_idx + 1) % size, pull_idx)
  push_idx = fby(0, lambda: idx_next[1])
  idx_next[1] = ifthenelse(o_p, (push_idx + 1) % size, push_idx)

  next_slots: list[Any] = [None] * size
  next_valids: list[Any] = [None] * size
  prev_valids: list[Stream] = [
      fby(False, _slot_thunk(next_valids, slot_idx))
      for slot_idx in range(size)
  ]
  pulled_occupied = switch(pull_idx, dict(enumerate(prev_valids)))
  checked_in = check_buffer_pull(
      ifthenelse(i_p, pulled_occupied, False), in_val
  )
  after_slots: list[Stream] = []
  after_valids: list[Stream] = []
  for slot_idx in range(size):
    prev_s = fby(None, _slot_thunk(next_slots, slot_idx))
    prev_v = prev_valids[slot_idx]
    write_here = ifthenelse(i_p, pull_idx == slot_idx, False)
    after_s = ifthenelse(write_here, checked_in, prev_s)
    after_v = ifthenelse(write_here, True, prev_v)
    read_here = ifthenelse(o_p, push_idx == slot_idx, False)
    next_slots[slot_idx] = ifthenelse(read_here, None, after_s)
    next_valids[slot_idx] = ifthenelse(read_here, False, after_v)
    after_slots.append(after_s)
    after_valids.append(after_v)

  rd_val = switch(push_idx, dict(enumerate(after_slots)))
  rd_valid = switch(push_idx, dict(enumerate(after_valids)))
  return check_buffer_push(
      when(o_p, rd_valid), when(o_p, rd_val), name=name
  )


@dataclasses.dataclass(frozen=True)
class _Ops:
  """Primitive or pervasive operator bundle for equivalence tests."""

  periodic_stream: Callable[..., Stream] = Stream.periodic
  pre: Callable[..., Stream] = pre
  fby: Callable[..., Stream] = fby
  post: Callable[..., Stream] = post
  morph: Callable[..., Stream] = morph
  current: Callable[..., Stream] = current
  counter: Callable[..., Stream] = counter
  prefix_sum: Callable[..., Stream] = prefix_sum
  scan: Callable[..., Stream] = scan
  buffer: Callable[..., Stream] = buffer


_PRIMITIVE_OPS = _Ops()
PERVASIVE_OPS = _Ops(
    periodic_stream=periodic_stream_pervasive,
    pre=pre_pervasive,
    fby=fby_pervasive,
    post=post_pervasive,
    morph=morph_pervasive,
    current=current_pervasive,
    counter=counter_pervasive,
    prefix_sum=prefix_sum_pervasive,
    scan=scan_pervasive,
    buffer=buffer_pervasive,
)


class PervasivesTest(absltest.TestCase):

  OPS = PERVASIVE_OPS

  def get_ops(self, use_primitives: bool) -> _Ops:
    """Returns either the primitive or derived pervasive operator bundle."""
    return _PRIMITIVE_OPS if use_primitives else self.OPS

  def setUp(self):
    super().setUp()
    logger.setLevel(logging.WARNING)

  def _assert_equivalent(
      self,
      make_prog: Callable[[bool], Callable[[], dict[str, Any]]],
      *,
      horizon: int,
      semantics: tuple[Semantics, ...] = (Semantics.KAHN, Semantics.SYNC),
      clock_calculus: bool = True,
      causality_analysis: bool = True,
      streams: Sequence[str] | None = None,
  ) -> None:
    """Asserts that primitive and pervasive program variants match exactly."""
    stream_batches: tuple[Sequence[str] | None, ...] = (
        (None,) if streams is None else (None, streams)
    )
    for sem in semantics:
      for cap_streams in stream_batches:
        with self.subTest(semantics=sem, streams=cap_streams):
          prim_res = Scheduler.run(
              make_prog(True),
              horizon=horizon,
              semantics=sem,
              causality_analysis=causality_analysis,
              clock_calculus=clock_calculus,
              streams=cap_streams,
          )
          self.assertFalse(Scheduler.deadlocked)
          perv_res = Scheduler.run(
              make_prog(False),
              horizon=horizon,
              semantics=sem,
              causality_analysis=causality_analysis,
              clock_calculus=clock_calculus,
              streams=cap_streams,
          )
          self.assertFalse(Scheduler.deadlocked)
          for key, prim_trace in prim_res.items():
            with self.subTest(semantics=sem, output=key):
              self.assertEqual(prim_trace, perv_res[key])

  def test_primitive_node_count_reduction(self):
    def build_variant(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        xs = Stream([3, 1, 4, 1, 5, 9])
        ys = Stream([10, 20, 30, 40, 50, 60])
        cond = Stream([True, False, True, True, False, True])
        return (
            ops.periodic_stream([0], [1, 2]),
            ops.pre(xs),
            ops.pre(xs, 3),
            ops.fby(ys, xs, 3),
            ops.post(xs, 3),
            ops.morph(xs, ys),
            ops.current(cond, xs, default=-1),
            ops.counter(start=5, max_val=7),
            ops.prefix_sum(xs, start=100),
            ops.scan(xs, op=Stream.max),
        )

      return prog

    for sem in (Semantics.KAHN, Semantics.SYNC):
      prim_graph = build_graph(build_variant(True), semantics=sem)
      perv_graph = build_graph(build_variant(False), semantics=sem)
      check_causality(prim_graph)
      check_causality(perv_graph)
      check_clocks(prim_graph)
      check_clocks(perv_graph)
      # 9 single-node primitives = 9 nodes, vs expanded pervasive nodes.
      self.assertLen(prim_graph.nodes, 9)
      self.assertLess(len(prim_graph.nodes), len(perv_graph.nodes))

  def test_base_clock_resets_and_operators_match_pervasives(self):
    def make_prog(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        xs = Stream([3, 1, 4, 1, 5, 9, 2, 6])
        ys = Stream([10, 20, 30, 40, 50, 60, 70, 80])
        starts = Stream([100, 200, 300, 400, 500, 600, 700, 800])
        cond = Stream([False, True, False, False, True, True, False, True])
        cond2 = Stream([True, False, False, True, False, True, False, False])
        rst = Stream([False, False, True, False, False, True, False, False])

        per_a = ops.periodic_stream([7, 8], [1, 2, 3], name='per_a')
        per_b = ops.periodic_stream([], [4, 5], name='per_b')
        pre_x = ops.pre(xs)
        arr_xy = ops.morph(xs * 10, ys)
        cur_def = ops.current(cond, xs, default=-1)
        cur_stream_def = ops.current(cond2, ys, default=starts)
        cnt_0 = ops.counter()
        cnt_s = ops.counter(start=starts)
        cnt_max = ops.counter(start=1, max_val=4)
        cnt_over = ops.counter(start=10, max_val=4)
        cnt_dyn_max = ops.counter(start=starts, max_val=ys * 5)
        psum_0 = ops.prefix_sum(xs)
        psum_s = ops.prefix_sum(xs, start=starts)
        scan_add = ops.scan(xs)
        scan_mul = ops.scan(xs, op=operator.mul)
        scan_max = ops.scan(xs, op=Stream.max)
        scan_min = ops.scan(xs, op=Stream.min)
        scan_custom = ops.scan(xs, op=lambda acc, val: acc * 2 + val)

        with reset(rst):
          pre_rst = ops.pre(xs)
          arr_rst = ops.morph(starts, ys)
          cur_rst = ops.current(cond, xs, default=starts)
          cnt_rst = ops.counter(start=starts)
          cnt_max_rst = ops.counter(start=2, max_val=3)
          cnt_over_rst = ops.counter(start=starts, max_val=250)
          psum_rst = ops.prefix_sum(xs, start=starts)
          scan_rst = ops.scan(xs, op=Stream.max)

        return {
            'per_a': per_a,
            'per_b': per_b,
            'pre_x': pre_x,
            'arr_xy': arr_xy,
            'cur_def': cur_def,
            'cur_stream_def': cur_stream_def,
            'cnt_0': cnt_0,
            'cnt_s': cnt_s,
            'cnt_max': cnt_max,
            'cnt_over': cnt_over,
            'cnt_dyn_max': cnt_dyn_max,
            'psum_0': psum_0,
            'psum_s': psum_s,
            'scan_add': scan_add,
            'scan_mul': scan_mul,
            'scan_max': scan_max,
            'scan_min': scan_min,
            'scan_custom': scan_custom,
            'pre_rst': pre_rst,
            'arr_rst': arr_rst,
            'cur_rst': cur_rst,
            'cnt_rst': cnt_rst,
            'cnt_max_rst': cnt_max_rst,
            'cnt_over_rst': cnt_over_rst,
            'psum_rst': psum_rst,
            'scan_rst': scan_rst,
        }

      return prog

    self._assert_equivalent(make_prog, horizon=8)
    for sem in (Semantics.KAHN, Semantics.SYNC):
      res = Scheduler.run(make_prog(True), horizon=8, semantics=sem)
      self.assertEqual(res['cnt_0'], [0, 1, 2, 3, 4, 5, 6, 7])
      self.assertEqual(res['cnt_max'], [1, 2, 3, 4, 4, 4, 4, 4])
      self.assertEqual(res['cnt_over'], [4, 4, 4, 4, 4, 4, 4, 4])
      self.assertEqual(res['cnt_max_rst'], [2, 3, 2, 3, 3, 2, 3, 3])
      self.assertEqual(
          res['cnt_over_rst'], [100, 101, 250, 250, 250, 250, 250, 250]
      )

  def test_subclocks_match_and_automaton_match_pervasives(self):
    def make_match_prog(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        xs = Stream([2, 5, 1, 8, 3, 7, 4, 6, 9, 2])
        ys = Stream([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        cond = Stream(
            [True, False, True, True, False, True, False, True, True, False]
        )
        inner_b = Stream(
            [False, True, True, False, True, False, True, True, False, True]
        )
        arm_rst = Stream(
            [False, False, False, True, False, False, False, True, False, False]
        )

        with match(cond) as sub_m:
          with case(True):
            sub_m.pre = ops.pre(xs)
            sub_arr = ops.morph(xs * 10, ys)
            sub_cur = ops.current(inner_b, xs, default=99)
            with reset(arm_rst):
              sub_cnt = ops.counter(start=1)
              sub_cnt_max = ops.counter(start=1, max_val=2)
              sub_cnt_over = ops.counter(start=10, max_val=3)
              sub_psum = ops.prefix_sum(xs, start=10)
              sub_scan = ops.scan(xs, op=Stream.max)
            sub_m.val = (
                sub_arr
                + sub_cur
                + sub_cnt
                + sub_cnt_max
                + sub_cnt_over
                + sub_psum
                + sub_scan
            )
          with case(False):
            sub_m.pre = -1
            sub_m.val = -1

        return {
            'sub_pre': sub_m.pre,
            'sub_val': sub_m.val,
        }

      return prog

    def make_automaton_prog(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        xs = Stream([2, 5, 1, 8, 3, 7, 4, 6, 9, 2])
        ys = Stream([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        with automaton(init='A') as fsm:
          with state('A'):
            c_a = ops.counter(start=0)
            s_a = ops.scan(xs)
            ps_a = ops.prefix_sum(ys, start=5)
            ar_a = ops.morph(100, s_a)
            until(c_a >= 1, 'B')
            fsm.out = ar_a + ps_a
          with state('B'):
            c_b = ops.counter(start=10)
            until(c_b >= 11, 'A')
            fsm.out = c_b
        return {'fsm_out': fsm.out}

      return prog

    self._assert_equivalent(make_match_prog, horizon=10)
    self._assert_equivalent(make_automaton_prog, horizon=10)

  def test_sync_absent_inputs_and_supersampled_resets_match_pervasives(self):
    def make_prog(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        rst = Stream([False, False, True, False, False, True, False, False])
        cond = Stream([True, False, True, False, True, False, True, True])
        super_rst = Stream(
            [False, True, False, False, False, True, False, False]
        )
        dense_x = Stream([3, 4, 5, 6, 7, 8, 9, 10])
        dense_y = Stream([10, 20, 30, 40, 50, 60, 70, 80])
        dense_def = Stream([99, 99, 88, 88, 77, 66, 55, 44])
        inner_b = Stream([True, True, False, True, True, False, False, True])

        with reset(super_rst):
          sup_when_cnt = when(cond, ops.counter(start=1, max_val=2))
          with match(cond) as sup_m:
            with case(True):
              ops.pre(dense_x, name='sup_pre')
              sup_arr = ops.morph(dense_x * 10, dense_y)
              sup_cur = ops.current(inner_b, dense_x, default=-5)
              sup_cnt = ops.counter(start=10)
              sup_cnt_max = ops.counter(start=10, max_val=11)
              sup_cnt_over = ops.counter(start=20, max_val=12)
              sup_psum = ops.prefix_sum(dense_x, start=100)
              sup_scan = ops.scan(dense_x, op=Stream.max)
              with reset(rst):
                ops.pre(dense_x, name='abs_pre')
                ops.morph(dense_x, dense_y, name='abs_arr')
                ops.current(inner_b, dense_x, default=dense_def, name='abs_cur')
                ops.counter(start=dense_def, name='abs_cnt')
                ops.counter(start=dense_def, max_val=70, name='abs_cnt_max')
                ops.prefix_sum(dense_x, start=dense_def, name='abs_psum')
                ops.scan(dense_x, name='abs_scan')
              sup_out = (
                  sup_arr
                  + sup_cur
                  + sup_cnt
                  + sup_cnt_max
                  + sup_cnt_over
                  + sup_psum
                  + sup_scan
              )
              sup_out.name = 'sup_out'
              sup_m.out = sup_out
            with case(False):
              sup_m.out = 0

        return {
            'sup_when_cnt': sup_when_cnt,
            'sup_merged': sup_m.out,
        }

      return prog

    self._assert_equivalent(
        make_prog,
        horizon=8,
        streams=(
            'abs_pre',
            'abs_arr',
            'abs_cur',
            'abs_cnt',
            'abs_cnt_max',
            'abs_psum',
            'abs_scan',
            'sup_pre',
            'sup_out',
        ),
    )

  def test_buffer_pervasive_matches_primitive_in_kahn_and_sync(self):
    def make_prog(use_primitives: bool):
      p_buf = buffer if use_primitives else self.OPS.buffer

      def prog():
        xs = Stream([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        xs_none = Stream([None, 20, None, 40, None, 60, None, 80, None, 100])
        # Single-slot lossless FIFO + same-tick cut-through:
        ip_1 = Stream(
            [True, False, True, False, True, False, True, False, True, False]
        )
        op_1 = Stream(
            [False, True, True, False, False, True, True, False, False, True]
        )
        # Depth-2 lossless FIFO + cut-through and wrap-around:
        ip_a = Stream(
            [True, True, False, False, True, True, False, True, False, False]
        )
        op_a = Stream(
            [False, False, True, True, False, True, True, True, False, False]
        )
        # Depth-3 lossless FIFO filling all 3 slots before draining:
        ip_3 = Stream(
            [True, True, True, False, False, True, True, False, False, False]
        )
        op_3 = Stream(
            [False, False, False, True, True, False, False, True, True, True]
        )
        # Reset discarding unread buffered items before new causal pull/push:
        rst = Stream(
            [False, False, False, True, False, False, False, True, False, False]
        )
        ip_r = Stream(
            [True, True, False, True, True, False, True, True, False, False]
        )
        op_r = Stream(
            [False, False, True, False, True, True, False, True, False, False]
        )
        gate = Stream(
            [True, False, True, True, False, True, True, False, True, True]
        )

        b1_a = p_buf(when(ip_1, xs), ip_1, op_1, size=1)
        b2_a = p_buf(when(ip_a, xs), ip_a, op_a, size=2)
        b3_a = p_buf(when(ip_a, xs), ip_a, op_a, size=3)
        b2_none = p_buf(when(ip_a, xs_none), ip_a, op_a, size=2)
        b3_full = p_buf(when(ip_3, xs), ip_3, op_3, size=3)

        with reset(rst):
          b2_rst = p_buf(when(ip_r, xs), ip_r, op_r, size=2)
          b3_rst = p_buf(when(ip_r, xs), ip_r, op_r, size=3)

        # Delayed feedback through fby + buffer:
        fb_buf = p_buf(
            when(ip_a, fby(1, lambda: fb_state)),
            ip_a,
            op_a,
            size=2,
        )
        with match(op_a) as fb_m:
          with case(True):
            fb_m.val = fb_buf
          with case(False):
            fb_m.val = 5
        fb_state = fb_m.val + 1

        # Subclock buffer inside match/case:
        with match(gate) as sub_m:
          with case(True):
            sub_buf = p_buf(when(ip_a, xs), ip_a, op_a, size=2, name='sub_buf')
            with match(op_a) as use_m:
              with case(True):
                use_m.val = sub_buf
              with case(False):
                use_m.val = 0
            sub_m.val = use_m.val
          with case(False):
            sub_m.val = -9

        return {
            'b1_a': b1_a,
            'b2_a': b2_a,
            'b3_a': b3_a,
            'b2_none': b2_none,
            'b3_full': b3_full,
            'b2_rst': b2_rst,
            'b3_rst': b3_rst,
            'fb_state': fb_state,
            'sub_val': sub_m.val,
        }

      return prog

    self._assert_equivalent(make_prog, horizon=10, streams=('sub_buf',))

    # Also verify Absent-elision equivalence on buffer_pervasive directly:
    sync_traces = Scheduler.run(
        make_prog(False),
        horizon=10,
        semantics=Semantics.SYNC,
        clock_calculus=True,
    )
    kahn_traces = Scheduler.run(
        make_prog(False),
        horizon=10,
        semantics=Semantics.KAHN,
        clock_calculus=True,
    )
    for key, sync_vals in sync_traces.items():
      with self.subTest(absent_elision=key):
        kahn_vals = kahn_traces[key]
        self.assertTrue(
            all(v is not Absent and v is not Bottom for v in kahn_vals)
        )
        self.assertEqual(present_values(sync_vals), kahn_vals)

    # Verify overflow and underflow fail fast with BufferError for both
    # primitive `buffer` and derived `buffer_pervasive` (including `None`).
    bad_cases = (
        (
            'overflow_s1',
            lambda fn: fn([10, 20], [True, True], [False, False], size=1),
            2,
            'overflow',
        ),
        (
            'overflow_s1_none',
            lambda fn: fn([None, 20], [True, True], [False, False], size=1),
            2,
            'overflow',
        ),
        (
            'overflow_s2',
            lambda fn: fn(
                [None, 20, 30],
                [True, True, True],
                [False, False, False],
                size=2,
            ),
            3,
            'overflow',
        ),
        (
            'underflow_s1',
            lambda fn: fn([10], [False], [True], size=1),
            1,
            'underflow',
        ),
        (
            'underflow_s2_after_none',
            lambda fn: fn(
                [None], [True, False, False], [False, True, True], size=2
            ),
            3,
            'underflow',
        ),
    )

    def reset_underflow(op_fn: Any) -> Stream:
      rst = Stream([False, True])
      ip = Stream([True, False])
      op = Stream([False, True])
      with reset(rst):
        return op_fn([10], ip, op, size=2)

    bad_cases += (('underflow_after_reset', reset_underflow, 2, 'underflow'),)
    for label, builder_fn, hz, word in bad_cases:
      for use_prim in (True, False):
        op_fn = buffer if use_prim else self.OPS.buffer
        for sem in (Semantics.SYNC, Semantics.KAHN):
          with self.subTest(case=label, primitive=use_prim, semantics=sem):
            with self.assertRaises(ExceptionGroup) as eg:
              Scheduler.run(
                  lambda fn=builder_fn, op=op_fn: fn(op),
                  horizon=hz,
                  semantics=sem,
              )
            self.assertTrue(
                any(
                    isinstance(err, BufferError) and word in str(err)
                    for err in eg.exception.exceptions
                ),
                f'Expected BufferError({word!r}), got '
                f'{eg.exception.exceptions!r}',
            )

  def test_buffer_pervasive_supersampled_reset_and_validation(self):
    def make_super_prog(use_primitives: bool):
      p_buf = buffer if use_primitives else self.OPS.buffer

      def prog():
        ip1 = Stream([True, False, True, False, True, False, True, False])
        op1 = Stream([False, True, True, False, False, True, False, True])
        ip2 = Stream([True, True, False, True, False, False, True, False])
        op2 = Stream([False, False, True, False, True, True, False, True])
        ip_sub = Stream([True, True, False, True, False, True])
        op_sub = Stream([False, False, True, False, True, True])
        xs = Stream([10, 20, 30, 40, 50, 60, 70, 80])
        cond = Stream([True, False, True, True, False, True, True, True])
        super_rst = Stream(
            [False, True, False, False, True, False, False, False]
        )

        raw1 = p_buf(when(ip1, xs), ip1, op1, size=1)
        raw2 = p_buf(when(ip2, xs), ip2, op2, size=2)
        with reset(super_rst):
          with match(cond) as m:
            with case(True):
              buf_out = p_buf(
                  when(ip_sub, [1, 2, 3, 4, 5, 6]), ip_sub, op_sub, size=2
              )
              m.out = merge(op_sub, buf_out, -1)
            with case(False):
              m.out = 0
        return {'raw1': raw1, 'raw2': raw2, 'super_match': m.out}

      return prog

    self._assert_equivalent(make_super_prog, horizon=8)

    with self.assertRaisesRegex(ValueError, 'buffer size must be >= 1'):
      build_graph(
          lambda: self.OPS.buffer(when(True, 1), True, True, size=0),
          semantics=Semantics.SYNC,
      )

    # Unsampled base-clock producer (`counter()`) instead of `. on ip` is
    # rejected by check_clocks just like primitive `buffer`:
    def bad_clock():
      ip = Stream([True, False, True, False])
      op = Stream([False, True, False, True])
      return self.OPS.buffer(counter(), ip, op, size=2)

    with self.assertRaises(ClockError):
      check_clocks(build_graph(bad_clock, semantics=Semantics.SYNC))

    # Instantaneous cut-through self-loop is rejected by check_causality just
    # like primitive `buffer`:
    def bad_cycle():
      cyc = self.OPS.buffer(when(True, lambda: cyc), True, True, size=2)  # pylint: disable=undefined-variable
      return cyc

    with self.assertRaises(CausalityError):
      check_causality(
          build_graph(bad_cycle, semantics=Semantics.SYNC)
      )

  def test_iterated_pre_fby_post_match_pervasives(self):
    def make_prog(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        xs = Stream([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ys = Stream([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rst = Stream(
            [False, False, False, False, True, False, False, False, True, False]
        )
        rst_dense = Stream(
            [False, True, True, False, False, True, False, True, False, False]
        )
        cond = Stream(
            [True, False, True, True, False, True, True, False, True, True]
        )

        pre1 = ops.pre(xs, 1)
        pre3 = ops.pre(xs, 3)
        fby1 = ops.fby(99, xs, 1)
        fby3 = ops.fby(ys * 100, xs, 3)
        post1 = ops.post(xs, 1)
        post3 = ops.post(xs + 0, 3)
        morph1 = ops.morph(ys * 100, xs, 1)
        morph3 = ops.morph(ys * 100, xs, 3)
        post_fby = ops.post(ops.fby(0, xs, n=3), n=2)

        fib3 = ops.fby(1, lambda: fib3_next, 3)  # pylint: disable=undefined-variable
        fib3_next = fib3 + ys

        with reset(rst):
          pre3_rst = ops.pre(xs, 2)
          fby3_rst = ops.fby(ys * 100, xs, 3)
          post2_rst = ops.post(xs, 1)
          morph3_rst = ops.morph(ys * 100, xs, 3)

        with reset(rst_dense):
          pre1_dense = ops.pre(xs, 1)
          pre3_dense = ops.pre(xs, 3)
          fby1_dense = ops.fby(ys * 100, xs, 1)
          fby3_dense = ops.fby(ys * 100, xs, 3)
          morph1_dense = ops.morph(ys * 100, xs, 1)
          morph3_dense = ops.morph(ys * 100, xs, 3)

        with match(cond) as m:
          with case(True):
            m.pre_sub = ops.pre(xs, 2)
            m.fby_sub = ops.fby(ys * 10, xs, 2)
            m.morph_sub = ops.morph(ys * 10, xs, 2)
            ops.post(xs, 2, name='post_sub')
            with reset(rst):
              m.fby_sub_rst = ops.fby(ys * 10, xs, 2)
              m.morph_sub_rst = ops.morph(ys * 10, xs, 2)
              ops.post(xs, 1, name='post_sub_rst')
          with case(False):
            m.pre_sub = -1
            m.fby_sub = -1
            m.morph_sub = -1
            m.fby_sub_rst = -1
            m.morph_sub_rst = -1

        return {
            'pre1': pre1,
            'pre3': pre3,
            'fby1': fby1,
            'fby3': fby3,
            'post1': post1,
            'post3': post3,
            'morph1': morph1,
            'morph3': morph3,
            'post_fby': post_fby,
            'fib3': fib3,
            'pre3_rst': pre3_rst,
            'fby3_rst': fby3_rst,
            'post2_rst': post2_rst,
            'morph3_rst': morph3_rst,
            'pre1_dense': pre1_dense,
            'pre3_dense': pre3_dense,
            'fby1_dense': fby1_dense,
            'fby3_dense': fby3_dense,
            'morph1_dense': morph1_dense,
            'morph3_dense': morph3_dense,
            'pre_sub': m.pre_sub,
            'fby_sub': m.fby_sub,
            'morph_sub': m.morph_sub,
            'fby_sub_rst': m.fby_sub_rst,
            'morph_sub_rst': m.morph_sub_rst,
        }

      return prog

    self._assert_equivalent(
        make_prog, horizon=10, streams=('post_sub', 'post_sub_rst')
    )

    def make_super_and_automaton_prog(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        cond = Stream(
            [True, False, True, True, False, True, True, False, True, True]
        )
        super_rst = Stream(
            [False, True, False, False, True, False, False, True, False, False]
        )
        with reset(super_rst):
          with match(cond) as m:
            with case(True):
              m.super_pre1 = ops.pre([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
              m.super_pre = ops.pre([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
              m.super_fby1 = ops.fby(99, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
              m.super_fby = ops.fby(99, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
              ops.post([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, name='super_post')
              m.super_morph = ops.morph(99, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
            with case(False):
              m.super_pre1 = -1
              m.super_pre = -1
              m.super_fby1 = -1
              m.super_fby = -1
              m.super_morph = -1

        with automaton(init='A') as fsm:
          with state('A'):
            c_a = counter(start=0)
            until(c_a >= 3, 'B')
            fsm.f1 = ops.fby(50, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 1)
            fsm.f = ops.fby(50, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 2)
            ops.post([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, name='fsm_p')
            fsm.m = ops.morph(50, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 2)
          with state('B'):
            c_b = counter(start=0)
            until(c_b >= 1, 'A')
            fsm.f1 = -1
            fsm.f = -1
            fsm.m = -1

        return {
            'super_pre1': m.super_pre1,
            'super_pre': m.super_pre,
            'super_fby1': m.super_fby1,
            'super_fby': m.super_fby,
            'super_morph': m.super_morph,
            'fsm_f1': fsm.f1,
            'fsm_f': fsm.f,
            'fsm_m': fsm.m,
        }

      return prog

    self._assert_equivalent(
        make_super_and_automaton_prog,
        horizon=10,
        streams=('super_post', 'fsm_p'),
    )

    def make_subclock_iterated_prog(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        # Include leading False ticks so tick 0 is `Absent` in SYNC.
        sched = Stream([
            False, True, False, True, True, False, True, True, False, True,
        ])
        raw_x = when(
            sched,
            Stream([9, 10, 9, 20, 30, 9, 40, 50, 9, 60]),
        )
        raw_inits = when(
            sched,
            Stream([9, 100, 9, 200, 300, 9, 400, 500, 9, 600]),
        )
        rst = Stream(
            [False, False, False, False, True, False, True, False, False, False]
        )

        with match(sched) as m:
          with case(True):
            sub_pre1 = ops.pre(raw_x, 1)
            sub_pre = ops.pre(raw_x, 2)
            sub_fby1 = ops.fby(raw_inits, raw_x, 1)
            sub_fby = ops.fby(raw_inits, raw_x, 3)
            ops.post(raw_x, 2, name='sub_post')
            m.sub_pre1 = sub_pre1
            m.sub_pre = sub_pre
            m.sub_fby1 = sub_fby1
            m.sub_fby = sub_fby
            m.sub_morph = ops.morph(raw_inits, raw_x, 3)
          with case(False):
            m.sub_pre1 = -1
            m.sub_pre = -1
            m.sub_fby1 = -1
            m.sub_fby = -1
            m.sub_morph = -1
        fby_post_loop = ops.fby(
            1, lambda: ops.post(fby_post_loop, 2) + 1, 3
        )  # pylint: disable=undefined-variable
        with reset(rst):
          with match(sched) as rm:
            with case(True):
              rm.pre1 = ops.pre(raw_x, 1)
              rm.pre = ops.pre(raw_x, 2)
              rm.fby1 = ops.fby(raw_inits, raw_x, 1)
              rm.fby = ops.fby(raw_inits, raw_x, 2)
              ops.post(raw_x, 1, name='rst_post')
              rm.morph = ops.morph(raw_inits, raw_x, 2)
            with case(False):
              rm.pre1 = -1
              rm.pre = -1
              rm.fby1 = -1
              rm.fby = -1
              rm.morph = -1

        return {
            'sub_pre1': m.sub_pre1,
            'sub_pre': m.sub_pre,
            'sub_fby1': m.sub_fby1,
            'sub_fby': m.sub_fby,
            'sub_morph': m.sub_morph,
            'fby_post_loop': fby_post_loop,
            'rst_pre1': rm.pre1,
            'rst_pre': rm.pre,
            'rst_fby1': rm.fby1,
            'rst_fby': rm.fby,
            'rst_morph': rm.morph,
        }

      return prog

    self._assert_equivalent(
        make_subclock_iterated_prog,
        horizon=10,
        semantics=(Semantics.SYNC, Semantics.KAHN),
        streams=('sub_post', 'rst_post'),
    )

    for bad_n in (0, -2, True, 1.5):
      for fn in (
          lambda n=bad_n: pre(1, n),
          lambda n=bad_n: self.OPS.pre(1, n),
          lambda n=bad_n: fby(0, 1, n),
          lambda n=bad_n: self.OPS.fby(0, 1, n),
          lambda n=bad_n: post(1, n),
          lambda n=bad_n: self.OPS.post(1, n),
          lambda n=bad_n: morph(0, 1, n),
          lambda n=bad_n: self.OPS.morph(0, 1, n),
      ):
        with self.assertRaisesRegex(ValueError, 'integer >= 1'):
          build_graph(fn, semantics=Semantics.SYNC)

  def test_counter_tick_and_subclock_current_equivalence(self):
    def make_prog(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        tick_seq = Stream([False, True, False, True, True, False, True, True])
        dyn_tick = (counter(0) % 2) == 1
        starts = Stream([10, 20, 30, 40, 50, 60, 70, 80])
        maxs = Stream([1, 2, 10, 10, 10, 3, 10, 10])
        rst = Stream([False, False, False, True, False, False, True, False])
        gate = Stream([True, False, True, True, False, True, False, True])

        base_t = ops.counter(start=0, tick=tick_seq)
        stall_t = ops.counter(start=5, tick=False)
        sat_t = ops.counter(start=0, max_val=2, tick=tick_seq)
        dyn_t = ops.counter(start=0, max_val=maxs, tick=dyn_tick)
        over_t = ops.counter(start=5, max_val=maxs, tick=tick_seq)
        with reset(rst):
          rst_t = ops.counter(start=starts, tick=tick_seq)
          rst_max = ops.counter(start=0, max_val=2, tick=tick_seq)
        with match(gate) as m:
          with case(True):
            m.sub_cnt = ops.counter(
                start=1, tick=Stream([False, True, False, True, True])
            )
          with case(False):
            m.sub_cnt = -1
        with automaton(init='A') as fsm:
          with state('A'):
            until(rst, 'B')
            fsm.cnt = ops.counter(start=0, tick=dyn_tick)
          with state('B'):
            until(rst, 'A')
            fsm.cnt = ops.counter(start=100, tick=tick_seq)

        # Direct subclock projection `current(b, when(b, x), default)` matches
        # canonical Lustre `c = merge(b, sub, whennot(b, fby(default, c)))`.
        nat = counter(0)
        sub_odd = when(dyn_tick, nat)
        if use_primitives:
          cur_sub = current(dyn_tick, sub_odd, default=-1)
          with reset(rst):
            cur_sub_rst = current(dyn_tick, sub_odd, default=starts)
        else:
          cur_sub = merge(
              dyn_tick,
              sub_odd,
              whennot(
                  dyn_tick,
                  fby(-1, lambda: cur_sub),  # pylint: disable=undefined-variable
              ),
          )
          with reset(rst):
            cur_sub_rst = merge(
                dyn_tick,
                sub_odd,
                whennot(
                    dyn_tick,
                    fby(
                        starts,
                        lambda: cur_sub_rst,  # pylint: disable=undefined-variable
                    ),
                ),
            )

        return {
            'base_t': base_t,
            'stall_t': stall_t,
            'sat_t': sat_t,
            'dyn_t': dyn_t,
            'over_t': over_t,
            'rst_t': rst_t,
            'rst_max': rst_max,
            'sub_cnt': m.sub_cnt,
            'fsm_cnt': fsm.cnt,
            'cur_sub': cur_sub,
            'cur_sub_rst': cur_sub_rst,
        }

      return prog

    self._assert_equivalent(make_prog, horizon=8)


if __name__ == '__main__':
  absltest.main()
