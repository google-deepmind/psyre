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
import logging
import operator
import sys
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
from psyre.clock import (
    ClockError,
    check_clocks,
)
from psyre.common import (
    prepare_delay,
    prepare_iterated_n,
)
from psyre.match import (
    case,
    match,
)
from psyre.psyre import (
    Scheduler,
    Semantics,
    Stream,
    buffer,
    counter,
    current,
    delay,
    fby,
    ifthenelse,
    logger,
    merge,
    morph,
    periodic_stream,
    post,
    pre,
    prefix_sum,
    present_values,
    reset,
    scan,
    when,
    whennot,
)

# pylint: enable=g-importing-member,g-multiple-import


def periodic_stream_pervasive(
    prefix: Sequence[Any], period: Sequence[Any], name: str = ''
) -> Stream:
  """Derived periodic stream constructed from Stream(prefix, period=...)."""
  return Stream(list(prefix), period=list(period), name=name)


def pre_pervasive(x: Any, n: int = 1, name: str = 'pre') -> Stream:
  """Derived stateful delay initialized to None, implemented via fby."""
  prepare_iterated_n(n, 'pre')
  out: Any = x
  for i in range(n):
    out = fby(None, out, name=name if i == n - 1 else 'pre')
  return out


def fby_pervasive(
    i_stream: Any, j_stream: Any, n: int = 1, name: str = 'fby'
) -> Stream:
  """Derived iterated followed-by delay implemented by chaining `fby` n times."""
  prepare_iterated_n(n, 'fby')
  shared_init = _share_stream(i_stream)
  out: Any = j_stream
  for i in range(n):
    out = fby(shared_init, out, name=name if i == n - 1 else 'fby')
  return out


def post_pervasive(x: Any, n: int = 1, name: str = 'post') -> Stream:
  """Derived iterated lookahead implemented by chaining `post` n times."""
  prepare_iterated_n(n, 'post')
  out: Any = x
  for i in range(n):
    out = post(out, name=name if i == n - 1 else 'post')
  return out


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
  """Derived sample-and-hold of stream i downsampled by boolean stream b."""
  c = merge(b, when(b, i), whennot(b, fby(default, lambda: t)), name=name)
  t = c  # Bind feedback variable after stream construction.
  return c


def counter_pervasive(
    start: Any = 0, max_val: Any = None, name: str = 'counter'
) -> Stream:
  """Derived resettable incrementing integer stream starting from start."""
  cur = fby(
      start,
      lambda: acc,
      name=name if max_val is None else (f'{name}_fby' if name else 'fby'),
  )
  if max_val is not None:
    cur = cur.min(max_val, name=name)
  acc = cur + 1
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


def _share_stream(spec: Any) -> Any:
  """Shares a single Stream/IStruct across multiple internal consumer ports."""
  if callable(spec):
    return spec
  return spec if isinstance(spec, Stream) else Stream(spec)


def _slot_thunk(slots: Sequence[Any], idx: int) -> Callable[[], Any]:
  """Returns a zero-argument thunk reading slots[idx] after loop binding."""
  return lambda: slots[idx]


def buffer_pervasive(
    i_stream: Any,
    i_p_stream: Any,
    o_p_stream: Any,
    size: int = 1,
    name: str = 'buffer',
) -> Stream:
  """Derived bounded ring buffer implemented purely from core primitives."""
  if size < 1:
    raise ValueError(f'buffer size must be >= 1, got {size}.')
  i_p = _share_stream(i_p_stream)
  o_p = _share_stream(o_p_stream)
  in_val = merge(i_p, i_stream, None)

  if size == 1:
    slot_next: list[Any] = [None]
    prev_slot = fby(None, lambda: slot_next[0])
    after_pull = ifthenelse(i_p, in_val, prev_slot)
    slot_next[0] = ifthenelse(o_p, None, after_pull)
    return when(o_p, after_pull, name=name)

  idx_next: list[Any] = [None, None]
  pull_idx = fby(0, lambda: idx_next[0])
  idx_next[0] = ifthenelse(i_p, (pull_idx + 1) % size, pull_idx)
  push_idx = fby(0, lambda: idx_next[1])
  idx_next[1] = ifthenelse(o_p, (push_idx + 1) % size, push_idx)

  next_slots: list[Any] = [None] * size
  after_slots: list[Stream] = []
  for slot_idx in range(size):
    prev_s = fby(None, _slot_thunk(next_slots, slot_idx))
    after_s = ifthenelse((pull_idx == slot_idx) & i_p, in_val, prev_s)
    next_slots[slot_idx] = ifthenelse(
        (push_idx == slot_idx) & o_p, None, after_s
    )
    after_slots.append(after_s)

  out_val: Any = after_slots[0]
  for slot_idx in range(1, size):
    out_val = ifthenelse(push_idx == slot_idx, after_slots[slot_idx], out_val)
  return when(o_p, out_val, name=name)


def delay_pervasive(
    i_stream: Any,
    k: int | None = None,
    *,
    period: Any = None,
    name: str = 'delay',
) -> Stream:
  """Derived delay implemented via buffer in SYNC and ifthenelse in KAHN."""
  _, delay_kwargs = prepare_delay(i_stream, k, period=period)
  p_istruct = delay_kwargs['p_stream'].istruct
  size = delay_kwargs['size']
  if Scheduler.semantics == Semantics.KAHN:
    return ifthenelse(True, i_stream, i_stream, name=name)
  shared_in = _share_stream(i_stream)
  tick = counter()
  if period is None:
    steps = 1 if k is None else k
    o_p = _share_stream(tick >= steps)
  else:
    prefix = tuple(p_istruct.prefix)
    period_seq = tuple(p_istruct.period)
    p_len = len(prefix)
    t_len = len(period_seq)
    idx = (
        tick % t_len
        if p_len == 0
        else ifthenelse(tick < p_len, tick, p_len + ((tick - p_len) % t_len))
    )
    table = {i: val for i, val in enumerate((*prefix, *period_seq))}
    o_p = _share_stream(merge(idx, table, default=False))
  ip = _share_stream(ifthenelse(o_p, shared_in, shared_in) == shared_in)
  buf = buffer(when(ip, shared_in), ip, o_p, size=size)
  absent_s = post(ip, sys.maxsize)
  return merge(o_p, buf, whennot(o_p, absent_s), name=name)


class PervasivesTest(absltest.TestCase):

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
      def prog():
        xs = Stream([3, 1, 4, 1, 5, 9])
        ys = Stream([10, 20, 30, 40, 50, 60])
        cond = Stream([True, False, True, True, False, True])
        if use_primitives:
          return (
              periodic_stream([0], [1, 2]),
              pre(xs),
              pre(xs, 3),
              fby(ys, xs, 3),
              post(xs, 3),
              morph(xs, ys),
              current(cond, xs, default=-1),
              counter(start=5, max_val=7),
              prefix_sum(xs, start=100),
              scan(xs, op=Stream.max),
              delay(xs, 2),
          )
        return (
            periodic_stream_pervasive([0], [1, 2]),
            pre_pervasive(xs),
            pre_pervasive(xs, 3),
            fby_pervasive(ys, xs, 3),
            post_pervasive(xs, 3),
            morph_pervasive(xs, ys),
            current_pervasive(cond, xs, default=-1),
            counter_pervasive(start=5, max_val=7),
            prefix_sum_pervasive(xs, start=100),
            scan_pervasive(xs, op=Stream.max),
            delay_pervasive(xs, 2),
        )

      return prog

    for sem in (Semantics.KAHN, Semantics.SYNC):
      prim_graph = Scheduler.build_graph(build_variant(True), semantics=sem)
      perv_graph = Scheduler.build_graph(build_variant(False), semantics=sem)
      check_causality(prim_graph)
      check_causality(perv_graph)
      check_clocks(prim_graph)
      check_clocks(perv_graph)
      # 10 single-node primitives = 10 nodes, vs expanded pervasive nodes.
      self.assertLen(prim_graph.nodes, 10)
      self.assertLess(len(prim_graph.nodes), len(perv_graph.nodes))

  def test_base_clock_resets_and_operators_match_pervasives(self):
    def make_prog(use_primitives: bool):
      p_stream = (
          periodic_stream if use_primitives else periodic_stream_pervasive
      )
      p_pre = pre if use_primitives else pre_pervasive
      p_morph = morph if use_primitives else morph_pervasive
      p_current = current if use_primitives else current_pervasive
      p_counter = counter if use_primitives else counter_pervasive
      p_psum = prefix_sum if use_primitives else prefix_sum_pervasive
      p_scan = scan if use_primitives else scan_pervasive

      def prog():
        xs = Stream([3, 1, 4, 1, 5, 9, 2, 6])
        ys = Stream([10, 20, 30, 40, 50, 60, 70, 80])
        starts = Stream([100, 200, 300, 400, 500, 600, 700, 800])
        cond = Stream([False, True, False, False, True, True, False, True])
        cond2 = Stream([True, False, False, True, False, True, False, False])
        rst = Stream([False, False, True, False, False, True, False, False])

        per_a = p_stream([7, 8], [1, 2, 3], name='per_a')
        per_b = p_stream([], [4, 5], name='per_b')
        pre_x = p_pre(xs)
        arr_xy = p_morph(xs * 10, ys)
        cur_def = p_current(cond, xs, default=-1)
        cur_stream_def = p_current(cond2, ys, default=starts)
        cnt_0 = p_counter()
        cnt_s = p_counter(start=starts)
        cnt_max = p_counter(start=1, max_val=4)
        cnt_over = p_counter(start=10, max_val=4)
        cnt_dyn_max = p_counter(start=starts, max_val=ys * 5)
        psum_0 = p_psum(xs)
        psum_s = p_psum(xs, start=starts)
        scan_add = p_scan(xs)
        scan_mul = p_scan(xs, op=operator.mul)
        scan_max = p_scan(xs, op=Stream.max)
        scan_min = p_scan(xs, op=Stream.min)
        scan_custom = p_scan(xs, op=lambda acc, val: acc * 2 + val)

        with reset(rst):
          pre_rst = p_pre(xs)
          arr_rst = p_morph(starts, ys)
          cur_rst = p_current(cond, xs, default=starts)
          cnt_rst = p_counter(start=starts)
          cnt_max_rst = p_counter(start=2, max_val=3)
          cnt_over_rst = p_counter(start=starts, max_val=250)
          psum_rst = p_psum(xs, start=starts)
          scan_rst = p_scan(xs, op=Stream.max)

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
      p_pre = pre if use_primitives else pre_pervasive
      p_morph = morph if use_primitives else morph_pervasive
      p_current = current if use_primitives else current_pervasive
      p_counter = counter if use_primitives else counter_pervasive
      p_psum = prefix_sum if use_primitives else prefix_sum_pervasive
      p_scan = scan if use_primitives else scan_pervasive

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
            sub_m.pre = p_pre(xs)
            sub_arr = p_morph(xs * 10, ys)
            sub_cur = p_current(inner_b, xs, default=99)
            with reset(arm_rst):
              sub_cnt = p_counter(start=1)
              sub_cnt_max = p_counter(start=1, max_val=2)
              sub_cnt_over = p_counter(start=10, max_val=3)
              sub_psum = p_psum(xs, start=10)
              sub_scan = p_scan(xs, op=Stream.max)
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
      p_morph = morph if use_primitives else morph_pervasive
      p_counter = counter if use_primitives else counter_pervasive
      p_psum = prefix_sum if use_primitives else prefix_sum_pervasive
      p_scan = scan if use_primitives else scan_pervasive

      def prog():
        xs = Stream([2, 5, 1, 8, 3, 7, 4, 6, 9, 2])
        ys = Stream([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        with automaton(init='A') as fsm:
          with state('A'):
            c_a = p_counter(start=0)
            s_a = p_scan(xs)
            ps_a = p_psum(ys, start=5)
            ar_a = p_morph(100, s_a)
            until(c_a >= 1, 'B')
            fsm.out = ar_a + ps_a
          with state('B'):
            c_b = p_counter(start=10)
            until(c_b >= 11, 'A')
            fsm.out = c_b
        return {'fsm_out': fsm.out}

      return prog

    self._assert_equivalent(make_match_prog, horizon=10)
    self._assert_equivalent(
        make_automaton_prog, horizon=10, semantics=(Semantics.SYNC,)
    )

  def test_sync_absent_inputs_and_supersampled_resets_match_pervasives(self):
    def make_prog(use_primitives: bool):
      p_pre = pre if use_primitives else pre_pervasive
      p_morph = morph if use_primitives else morph_pervasive
      p_current = current if use_primitives else current_pervasive
      p_counter = counter if use_primitives else counter_pervasive
      p_psum = prefix_sum if use_primitives else prefix_sum_pervasive
      p_scan = scan if use_primitives else scan_pervasive

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
          sup_when_cnt = when(cond, p_counter(start=1, max_val=2))
          with match(cond) as sup_m:
            with case(True):
              p_pre(dense_x, name='sup_pre')
              sup_arr = p_morph(dense_x * 10, dense_y)
              sup_cur = p_current(inner_b, dense_x, default=-5)
              sup_cnt = p_counter(start=10)
              sup_cnt_max = p_counter(start=10, max_val=11)
              sup_cnt_over = p_counter(start=20, max_val=12)
              sup_psum = p_psum(dense_x, start=100)
              sup_scan = p_scan(dense_x, op=Stream.max)
              with reset(rst):
                p_pre(dense_x, name='abs_pre')
                p_morph(dense_x, dense_y, name='abs_arr')
                p_current(inner_b, dense_x, default=dense_def, name='abs_cur')
                p_counter(start=dense_def, name='abs_cnt')
                p_counter(start=dense_def, max_val=70, name='abs_cnt_max')
                p_psum(dense_x, start=dense_def, name='abs_psum')
                p_scan(dense_x, name='abs_scan')
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
        semantics=(Semantics.SYNC,),
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
      p_buf = buffer if use_primitives else buffer_pervasive

      def prog():
        xs = Stream([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # Lossless FIFO, underflow (push when empty -> None), and same-tick
        # cut-through (pull & push simultaneously):
        ip_a = Stream(
            [True, True, False, False, False, True, False, True, True, False]
        )
        op_a = Stream(
            [False, False, True, True, True, True, False, False, True, True]
        )
        # Ring-buffer overwrite (3 consecutive pulls before pushes):
        ip_ov = Stream(
            [True, True, True, False, False, True, True, True, True, False]
        )
        op_ov = Stream(
            [False, False, False, True, True, False, False, False, True, True]
        )
        rst = Stream(
            [False, False, False, True, False, False, False, True, False, False]
        )
        gate = Stream(
            [True, False, True, True, False, True, True, False, True, True]
        )

        b1_a = p_buf(when(ip_a, xs), ip_a, op_a, size=1)
        b2_a = p_buf(when(ip_a, xs), ip_a, op_a, size=2)
        b3_a = p_buf(when(ip_a, xs), ip_a, op_a, size=3)

        b1_ov = p_buf(when(ip_ov, xs), ip_ov, op_ov, size=1)
        b2_ov = p_buf(when(ip_ov, xs), ip_ov, op_ov, size=2)
        b3_ov = p_buf(when(ip_ov, xs), ip_ov, op_ov, size=3)

        with reset(rst):
          b2_rst = p_buf(when(ip_ov, xs), ip_ov, op_ov, size=2)
          b3_rst = p_buf(when(ip_a, xs), ip_a, op_a, size=3)

        # Delayed feedback through fby + buffer:
        fb_buf = p_buf(
            when(ip_a, fby(1, lambda: fb_state)),
            ip_a,
            op_a,
            size=2,
        )
        with match(op_a) as fb_m:
          with case(True):
            fb_m.val = ifthenelse(fb_buf == None, 0, fb_buf)  # pylint: disable=singleton-comparison
          with case(False):
            fb_m.val = 5
        fb_state = fb_m.val + 1

        # Subclock buffer inside match/case:
        with match(gate) as sub_m:
          with case(True):
            sub_buf = p_buf(when(ip_a, xs), ip_a, op_a, size=2, name='sub_buf')
            with match(op_a) as use_m:
              with case(True):
                use_m.val = ifthenelse(sub_buf == None, -1, sub_buf)  # pylint: disable=singleton-comparison
              with case(False):
                use_m.val = 0
            sub_m.val = use_m.val
          with case(False):
            sub_m.val = -9

        return {
            'b1_a': b1_a,
            'b2_a': b2_a,
            'b3_a': b3_a,
            'b1_ov': b1_ov,
            'b2_ov': b2_ov,
            'b3_ov': b3_ov,
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
        self.assertEqual(present_values(sync_vals), kahn_traces[key])

  def test_buffer_pervasive_supersampled_reset_and_validation(self):
    def make_super_prog(use_primitives: bool):
      p_buf = buffer if use_primitives else buffer_pervasive

      def prog():
        ip = Stream([True, True, False, True, False, False, True, False])
        xs = when(ip, [10, 20, 30, 40, 50, 60, 70, 80])
        op = Stream([False, False, True, False, True, True, False, True])
        cond = Stream([True, False, True, True, False, True, True, True])
        super_rst = Stream(
            [False, True, False, False, True, False, False, False]
        )

        raw1 = p_buf(xs, ip, op, size=1)
        raw2 = p_buf(xs, ip, op, size=2)
        with reset(super_rst):
          with match(cond) as m:
            with case(True):
              buf_out = p_buf(
                  when(ip, [1, 2, 3, 4, 5, 6, 7, 8]), ip, op, size=2
              )
              m.out = merge(op, buf_out, -1)
            with case(False):
              m.out = 0
        return {'raw1': raw1, 'raw2': raw2, 'super_match': m.out}

      return prog

    self._assert_equivalent(
        make_super_prog,
        horizon=8,
        semantics=(Semantics.SYNC,),
    )

    with self.assertRaisesRegex(ValueError, 'buffer size must be >= 1'):
      Scheduler.build_graph(
          lambda: buffer_pervasive(when(True, 1), True, True, size=0),
          semantics=Semantics.SYNC,
      )

    # Unsampled base-clock producer (`counter()`) instead of `. on ip` is
    # rejected by check_clocks just like primitive `buffer`:
    def bad_clock():
      ip = Stream([True, False, True, False])
      op = Stream([False, True, False, True])
      return buffer_pervasive(counter(), ip, op, size=2)

    with self.assertRaises(ClockError):
      check_clocks(Scheduler.build_graph(bad_clock, semantics=Semantics.SYNC))

    # Instantaneous cut-through self-loop is rejected by check_causality just
    # like primitive `buffer`:
    def bad_cycle():
      cyc = buffer_pervasive(when(True, lambda: cyc), True, True, size=2)  # pylint: disable=undefined-variable
      return cyc

    with self.assertRaises(CausalityError):
      check_causality(
          Scheduler.build_graph(bad_cycle, semantics=Semantics.SYNC)
      )

  def test_delay_pervasive_matches_primitive_in_kahn_and_sync(self):
    def make_prog(use_primitives: bool):
      p_delay = delay if use_primitives else delay_pervasive

      def prog():
        xs = Stream([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ys = Stream([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rst = Stream(
            [False, False, False, True, False, False, False, True, False, False]
        )
        gate = Stream(
            [True, False, True, True, False, True, True, False, True, True]
        )
        arm_rst = Stream([
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ])
        sched_a = periodic_stream([False, False], [True])
        sched_b = periodic_stream([False], [True, False])
        sched_c = periodic_stream([], [False, True, True])

        d_def = p_delay(xs)
        d_k0 = p_delay(xs, 0)
        d_k2 = p_delay(xs, k=2)
        d_pa = p_delay(xs, period=sched_a)
        d_pb = p_delay(xs, period=sched_b)
        d_pc = p_delay(xs, period=sched_c)

        with reset(rst):
          d_k1_rst = p_delay(xs)
          d_k2_rst = p_delay(xs, 2)
          d_pb_rst = p_delay(xs, period=sched_b)

        with match(gate) as sub_m:
          with case(True):
            sub_d1 = p_delay(xs, 1, name='sub_d1')
            sub_dp = p_delay(ys, period=sched_b, name='sub_dp')
            with reset(arm_rst):
              p_delay(xs, 2, name='sub_d_rst')
              p_delay(ys, period=sched_c, name='sub_dp_rst')
            sub_m.val = sub_d1 + sub_dp
          with case(False):
            sub_m.val = -1

        return {
            'd_def': d_def,
            'd_k0': d_k0,
            'd_k2': d_k2,
            'd_pa': d_pa,
            'd_pb': d_pb,
            'd_pc': d_pc,
            'd_k1_rst': d_k1_rst,
            'd_k2_rst': d_k2_rst,
            'd_pb_rst': d_pb_rst,
            'sub_val': sub_m.val,
        }

      return prog

    self._assert_equivalent(
        make_prog,
        horizon=10,
        streams=('sub_d1', 'sub_dp', 'sub_d_rst', 'sub_dp_rst'),
    )

    def make_automaton_prog(use_primitives: bool):
      p_delay = delay if use_primitives else delay_pervasive

      def prog():
        xs = Stream([10, 20, 30, 40, 50, 60, 70, 80])
        sched = periodic_stream([False], [True, False])
        with automaton(init='A') as fsm:
          with state('A'):
            c_a = counter(start=0)
            until(c_a >= 2, 'B')
            fsm.d_k = p_delay(xs, 1)
            fsm.d_p = p_delay(xs, period=sched)
          with state('B'):
            c_b = counter(start=0)
            until(c_b >= 1, 'A')
            fsm.d_k = p_delay(xs, 0)
            fsm.d_p = -1
        return {'fsm_dk': fsm.d_k, 'fsm_dp': fsm.d_p}

      return prog

    self._assert_equivalent(
        make_automaton_prog, horizon=8, semantics=(Semantics.SYNC,)
    )

    def make_absent_and_super_prog(use_primitives: bool):
      p_delay = delay if use_primitives else delay_pervasive

      def prog():
        sparse_sched = periodic_stream(
            [], [True, False, True, False, True, True, False, True]
        )
        sparse_x = delay(Stream([10, 20, 30, 40, 50]), period=sparse_sched)
        rst = Stream([False, False, False, True, False, False, False, False])
        cond = Stream([True, False, True, True, False, True, True, True])
        super_rst = Stream(
            [False, True, False, False, True, False, False, False]
        )
        sched = periodic_stream([False], [True, False])

        raw_k1 = p_delay(sparse_x, 1)
        raw_p = p_delay(sparse_x, period=sched)
        with reset(rst):
          rst_k1 = p_delay(sparse_x, 1)
          rst_p = p_delay(sparse_x, period=sched)
        with reset(super_rst):
          with match(cond) as m:
            with case(True):
              m.dk = p_delay([1, 2, 3, 4, 5, 6, 7, 8], 1)
              m.dp = p_delay([1, 2, 3, 4, 5, 6, 7, 8], period=sched)
            with case(False):
              m.dk = -1
              m.dp = -1
        return {
            'raw_k1': raw_k1,
            'raw_p': raw_p,
            'rst_k1': rst_k1,
            'rst_p': rst_p,
            'super_dk': m.dk,
            'super_dp': m.dp,
        }

      return prog

    self._assert_equivalent(
        make_absent_and_super_prog,
        horizon=8,
        semantics=(Semantics.SYNC,),
    )

    for fn in (delay, delay_pervasive):
      with self.assertRaisesRegex(ValueError, 'non-negative integer'):
        Scheduler.build_graph(lambda f=fn: f(1, -1), semantics=Semantics.SYNC)
      with self.assertRaisesRegex(ValueError, 'non-negative integer'):
        Scheduler.build_graph(lambda f=fn: f(1, True), semantics=Semantics.SYNC)
      with self.assertRaisesRegex(ValueError, 'either k or period, not both'):
        Scheduler.build_graph(
            lambda f=fn: f(1, 2, period=periodic_stream([False], [True])),
            semantics=Semantics.SYNC,
        )
      with self.assertRaisesRegex(TypeError, 'static periodic'):
        Scheduler.build_graph(
            lambda f=fn: f(1, period=[False, True]),
            semantics=Semantics.SYNC,
        )
      with self.assertRaisesRegex(ValueError, 'static periodic'):
        Scheduler.build_graph(
            lambda f=fn: f(1, period=counter()),
            semantics=Semantics.SYNC,
        )
      with self.assertRaisesRegex(ValueError, 'boolean values'):
        Scheduler.build_graph(
            lambda f=fn: f(1, period=periodic_stream([0], [1])),
            semantics=Semantics.SYNC,
        )

  def test_iterated_pre_fby_post_match_pervasives(self):
    def make_prog(use_primitives: bool):
      p_pre = pre if use_primitives else pre_pervasive
      p_fby = fby if use_primitives else fby_pervasive
      p_post = post if use_primitives else post_pervasive
      p_morph = morph if use_primitives else morph_pervasive

      def prog():
        xs = Stream([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ys = Stream([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rst = Stream(
            [False, False, False, False, True, False, False, False, True, False]
        )
        cond = Stream(
            [True, False, True, True, False, True, True, False, True, True]
        )

        pre1 = p_pre(xs, 1)
        pre3 = p_pre(xs, 3)
        fby1 = p_fby(99, xs, 1)
        fby3 = p_fby(ys * 100, xs, 3)
        post1 = p_post(xs, 1)
        post3 = p_post(xs + 0, 3)
        morph1 = p_morph(ys * 100, xs, 1)
        morph3 = p_morph(ys * 100, xs, 3)
        post_fby = p_post(p_fby(0, xs, n=3), n=2)

        fib3 = p_fby(1, lambda: fib3_next, 3)  # pylint: disable=undefined-variable
        fib3_next = fib3 + ys

        with reset(rst):
          pre3_rst = p_pre(xs, 2)
          fby3_rst = p_fby(ys * 100, xs, 3)
          post2_rst = p_post(xs, 1)
          morph3_rst = p_morph(ys * 100, xs, 3)

        with match(cond) as m:
          with case(True):
            m.pre_sub = p_pre(xs, 2)
            m.fby_sub = p_fby(ys * 10, xs, 2)
            m.morph_sub = p_morph(ys * 10, xs, 2)
            p_post(xs, 2, name='post_sub')
            with reset(rst):
              m.fby_sub_rst = p_fby(ys * 10, xs, 2)
              m.morph_sub_rst = p_morph(ys * 10, xs, 2)
              p_post(xs, 1, name='post_sub_rst')
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

    def make_sync_only_prog(use_primitives: bool):
      p_pre = pre if use_primitives else pre_pervasive
      p_fby = fby if use_primitives else fby_pervasive
      p_post = post if use_primitives else post_pervasive
      p_morph = morph if use_primitives else morph_pervasive

      def prog():
        sparse_sched = periodic_stream(
            [], [True, False, True, True, False, True, True, False, True, True]
        )
        init_sched = periodic_stream(
            [], [True, True, False, True, True, True, False, True, True, True]
        )
        sparse_x = delay(
            Stream([10, 20, 30, 40, 50, 60, 70]), period=sparse_sched
        )
        inits = delay(
            Stream([100, 200, 400, 500, 600, 800, 900, 10]), period=init_sched
        )
        rst = Stream(
            [False, False, False, False, True, False, False, False, False, True]
        )
        cond = Stream(
            [True, False, True, True, False, True, True, False, True, True]
        )
        super_rst = Stream(
            [False, True, False, False, True, False, False, True, False, False]
        )

        raw_pre = p_pre(sparse_x, 2)
        raw_fby = p_fby(inits, sparse_x, 3)
        raw_post = p_post(sparse_x, 2)
        raw_morph = p_morph(inits, sparse_x, 3)
        with reset(rst):
          rst_pre = p_pre(sparse_x, 2)
          rst_fby = p_fby(inits, sparse_x, 2)
          rst_post = p_post(sparse_x, 1)
          rst_morph = p_morph(inits, sparse_x, 2)
        with reset(super_rst):
          with match(cond) as m:
            with case(True):
              m.super_pre = p_pre([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
              m.super_fby = p_fby(99, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
              m.super_post = p_post([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
              m.super_morph = p_morph(99, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
            with case(False):
              m.super_pre = -1
              m.super_fby = -1
              m.super_post = -1
              m.super_morph = -1

        with automaton(init='A') as fsm:
          with state('A'):
            c_a = counter(start=0)
            until(c_a >= 3, 'B')
            fsm.f = p_fby(50, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 2)
            fsm.p = p_post([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
            fsm.m = p_morph(50, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 2)
          with state('B'):
            c_b = counter(start=0)
            until(c_b >= 1, 'A')
            fsm.f = -1
            fsm.p = -1
            fsm.m = -1

        return {
            'raw_pre': raw_pre,
            'raw_fby': raw_fby,
            'raw_post': raw_post,
            'raw_morph': raw_morph,
            'rst_pre': rst_pre,
            'rst_fby': rst_fby,
            'rst_post': rst_post,
            'rst_morph': rst_morph,
            'super_pre': m.super_pre,
            'super_fby': m.super_fby,
            'super_post': m.super_post,
            'super_morph': m.super_morph,
            'fsm_f': fsm.f,
            'fsm_p': fsm.p,
            'fsm_m': fsm.m,
        }

      return prog

    self._assert_equivalent(
        make_sync_only_prog, horizon=10, semantics=(Semantics.SYNC,)
    )

    for bad_n in (0, -2, True, 1.5):
      for fn in (
          lambda n=bad_n: pre(1, n),
          lambda n=bad_n: pre_pervasive(1, n),
          lambda n=bad_n: fby(0, 1, n),
          lambda n=bad_n: fby_pervasive(0, 1, n),
          lambda n=bad_n: post(1, n),
          lambda n=bad_n: post_pervasive(1, n),
          lambda n=bad_n: morph(0, 1, n),
          lambda n=bad_n: morph_pervasive(0, 1, n),
      ):
        with self.assertRaisesRegex(ValueError, 'integer >= 1'):
          Scheduler.build_graph(fn, semantics=Semantics.SYNC)


if __name__ == '__main__':
  absltest.main()
