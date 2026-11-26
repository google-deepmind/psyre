# `psyre.math`

Minimal built-in scalar mathematical and random stream primitives built on
Python's standard library.

## Overview & Role

`psyre.math` provides three lightweight zero-dependency `@psyre_comb` stream
operators built directly on Python's `math` and `random` modules, all
re-exported from top-level `psyre`. Moving `random_stream` out of the
stateful/synchronous primitive registries into `psyre.math` keeps primitive
registries focused on reactive control/temporal operators while preserving clean
purity separation for CSE.

## Design & Key Invariants

- **Purity separation**: `exp` and `tanh` are decorated with
  `@psyre_comb(pure=True)` so [`optimize_graph`](optimization.md) can fold and
  CSE identical invocations. `random_stream` is intentionally left impure
  (`@psyre_comb`, `pure=False`) so CSE never merges independent random streams.
- **Scalar-return tagging**: All three wrappers are marked with
  `mark_builtin_scalar_op` (`_psyre_returns_scalar = True`), allowing
  [`Graph.is_bool_stream`](graph.md) to prove that downstream comparisons on
  their outputs return Python `bool`s.
- **Scalar scope**: These functions operate on Python scalar streams. For dense
  NumPy or JAX array streams, use NEP-13/NEP-18 dispatch,
  [`psyre.numpy`](numpy.md), or [`psyre.jax`](jax.md).

## Public API Reference

All functions below are re-exported from top-level `psyre`.

### `exp`

```python
@psyre_comb(pure=True)
def exp(v)
```

Pointwise exponential stream operator computing `math.exp(v)` on each active
tick. Eligible for pure combinational CSE.

### `tanh`

```python
@psyre_comb(pure=True)
def tanh(v)
```

Pointwise hyperbolic tangent stream operator computing `math.tanh(v)` on each
active tick. Eligible for pure combinational CSE.

### `random_stream`

```python
@psyre_comb
def random_stream()
```

Zero-input impure stream generator emitting `random.random()` in `[0.0, 1.0)` on
each active tick. Also exposed via `Stream.random()`. Never CSE-eliminated.

## See Also

- [`psyre.stream`](stream.md) — `Stream.random()` and built-in arithmetic operators
- [`psyre.numpy`](numpy.md) and [`psyre.jax`](jax.md) — Array-valued math operations
