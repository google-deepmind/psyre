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
"""Equivalence tests for derived Psyre pervasives using `psyre_equations`."""

# pylint: disable=undefined-variable,used-before-assignment
# pyrefly: ignore-errors[unknown-name]

from collections.abc import Callable
import dataclasses
import operator
from typing import Any

from absl.testing import absltest

# pylint: disable=g-importing-member,g-multiple-import
from psyre.analysis import CausalityError, check_causality
from psyre.common import (
    Semantics,
    fby,
    ifthenelse,
    merge,
    prepare_iterated_n,
    switch,
    when,
    whennot,
)
from psyre.equations import psyre_equations
from psyre.match import case, match
from psyre.stream import Stream
from tests import pervasives_test

# pylint: enable=g-importing-member,g-multiple-import


def morph_pervasive(
    x: Any, y: Any, n: int = 1, name: str = 'morph'
) -> Stream:
  """Derived initialization operator emitting first n values of x then y."""
  prepare_iterated_n(n, 'morph')
  with psyre_equations() as eq:

    class _(auto_name=False):
      first = fby(True, False, n=n)
      __output__ = merge(first, when(first, x), whennot(first, y), name=name)

  return eq.output


def current_pervasive(
    b: Any, i: Any, default: Any = None, name: str = 'current'
) -> Stream:
  """Derived same-clock sample-and-hold of stream i gated by boolean b."""
  with psyre_equations() as eq:

    class _(auto_name=False):
      c = merge(b, when(b, i), whennot(b, fby(default, c)), name=name)
      __output__ = c

  return eq.output


def counter_pervasive(
    start: Any = 0,
    max_val: Any = None,
    tick: Any = True,
    name: str = 'counter',
) -> Stream:
  """Derived resettable incrementing integer stream enabled by tick."""
  with psyre_equations() as eq:

    class _(auto_name=False):
      raw = fby(
          start,
          acc,
          name=name if max_val is None else (f'{name}_fby' if name else 'fby'),
      )
      cur = raw if max_val is None else raw.min(max_val, name=name)
      acc = (
          cur + 1
          if isinstance(tick, bool) and tick
          else ifthenelse(tick, cur + 1, cur)
      )
      __output__ = cur

  return eq.output


def prefix_sum_pervasive(
    data: Any, start: Any = 0, name: str = 'prefix_sum'
) -> Stream:
  """Derived resettable running sum of data starting from start."""
  with psyre_equations() as eq:

    class _(auto_name=False):
      acc = fby(start, tmp, name=f'{name}_acc' if name else 'fby')
      tmp = acc + data
      tmp.name = name
      __output__ = tmp

  return eq.output


def scan_pervasive(
    data: Any, op: Callable[[Any, Any], Any] = operator.add, name: str = 'scan'
) -> Stream:
  """Derived cumulative fold of binary operator op across data."""
  with psyre_equations() as eq:

    class _(auto_name=False):
      tmp = fby(data, acc)
      acc = morph_pervasive(data, op(tmp, data), name=name)
      __output__ = acc

  return eq.output


def buffer_pervasive(
    i_stream: Any,
    i_p_stream: Any,
    o_p_stream: Any,
    size: int = 1,
    name: str = 'buffer',
) -> Stream:
  """Derived bounded ring buffer implemented with lambda-free equations."""
  if size < 1:
    raise ValueError(f'buffer size must be >= 1, got {size}.')
  i_p = pervasives_test.share_stream(i_p_stream)
  o_p = pervasives_test.share_stream(o_p_stream)
  in_val = merge(i_p, i_stream, None)

  with psyre_equations() as eq:
    if size == 1:

      class _(auto_name=False):
        prev_slot = fby(None, slot_next)
        prev_valid = fby(False, valid_next)
        checked_in = pervasives_test.check_buffer_pull(
            ifthenelse(i_p, prev_valid, False), in_val
        )
        after_pull = ifthenelse(i_p, checked_in, prev_slot)
        after_valid = ifthenelse(i_p, True, prev_valid)
        slot_next = ifthenelse(o_p, None, after_pull)
        valid_next = ifthenelse(o_p, False, after_valid)
        __output__ = pervasives_test.check_buffer_push(
            when(o_p, after_valid), when(o_p, after_pull), name=name
        )

    else:

      class _(auto_name=False):
        """Multi-slot bounded lossless FIFO buffer equations."""

        pull_idx = fby(0, next_pull_idx)
        next_pull_idx = ifthenelse(i_p, (pull_idx + 1) % size, pull_idx)
        push_idx = fby(0, next_push_idx)
        next_push_idx = ifthenelse(o_p, (push_idx + 1) % size, push_idx)

        prev_valids: list[Stream] = []
        for slot_idx in range(size):
          prev_valids.append(fby(False, next_valids[slot_idx]))
        pulled_occupied = switch(pull_idx, dict(enumerate(prev_valids)))
        checked_in = pervasives_test.check_buffer_pull(
            ifthenelse(i_p, pulled_occupied, False), in_val
        )

        after_slots: list[Stream] = []
        after_valids: list[Stream] = []
        next_slots_list: list[Stream] = []
        next_valids_list: list[Stream] = []
        for slot_idx in range(size):
          prev_s = fby(None, next_slots[slot_idx])
          prev_v = prev_valids[slot_idx]
          write_here = ifthenelse(i_p, pull_idx == slot_idx, False)
          after_s = ifthenelse(write_here, checked_in, prev_s)
          after_v = ifthenelse(write_here, True, prev_v)
          read_here = ifthenelse(o_p, push_idx == slot_idx, False)
          next_slots_list.append(ifthenelse(read_here, None, after_s))
          next_valids_list.append(ifthenelse(read_here, False, after_v))
          after_slots.append(after_s)
          after_valids.append(after_v)
        next_slots = tuple(next_slots_list)
        next_valids = tuple(next_valids_list)
        rd_val = switch(push_idx, dict(enumerate(after_slots)))
        rd_valid = switch(push_idx, dict(enumerate(after_valids)))
        __output__ = pervasives_test.check_buffer_push(
            when(o_p, rd_valid), when(o_p, rd_val), name=name
        )

  return eq.output


class PervasivesEquationsTest(pervasives_test.PervasivesTest):
  """Runs the full pervasive suite against `psyre_equations` pervasives."""

  OPS = dataclasses.replace(
      pervasives_test.PERVASIVE_OPS,
      morph=morph_pervasive,
      current=current_pervasive,
      counter=counter_pervasive,
      prefix_sum=prefix_sum_pervasive,
      scan=scan_pervasive,
      buffer=buffer_pervasive,
  )

  def test_equations_forward_refs_with_pervasives(self):
    def make_prog(use_primitives: bool):
      ops = self.get_ops(use_primitives)

      def prog():
        with psyre_equations() as eq:

          class _:
            ip_a = Stream([
                True, True, False, False, True,
                False, True, True, False, False,
            ])
            op_a = Stream([
                False, True, True, False, False,
                True, True, False, True, False,
            ])
            ys = Stream([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

            # Delayed feedback through fby + buffer without `lambda:`.
            fb_buf = ops.buffer(
                when(ip_a, fby(1, fb_state)), ip_a, op_a, size=2
            )
            with match(op_a) as fb_m:
              with case(True):
                fb_m.val = fb_buf
              with case(False):
                fb_m.val = 5
            fb_state = fb_m.val + 1

            # Mutually recursive iterated fby without `lambda:`.
            fib3 = ops.fby(1, fib3_next, 3)
            fib3_next = fib3 + ys

            # Delayed feedback through iterated fby + post without `lambda:`.
            fby_post_loop = ops.fby(1, ops.post(fby_post_loop, 2) + 1, 3)

            __output__ = {
                'fb_buf': fb_buf,
                'fb_state': fb_state,
                'fib3': fib3,
                'fby_post_loop': fby_post_loop,
            }

        return eq.output

      return prog

    self._assert_equivalent(make_prog, horizon=10)

    # Instantaneous cut-through self-loop without `lambda:` triggers causality.
    with psyre_equations(semantics=Semantics.SYNC) as bad_eq:

      class _:
        cyc = buffer_pervasive(when(True, cyc), True, True, size=2)
        __output__ = cyc

    with self.assertRaises(CausalityError):
      check_causality(bad_eq.graph)


if __name__ == '__main__':
  absltest.main()
