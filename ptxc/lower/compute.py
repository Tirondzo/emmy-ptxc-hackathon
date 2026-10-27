"""Scalar compute: Assign, Let, Init, Accum, Load, Write."""

from __future__ import annotations

from collections.abc import Sequence

from .. import ir
from ..ir import Node
from ..ptx import Addr, Reg, Value
from ..types import Ty
from .lowerer import Lowerer, rule
from .symbols import Array


@rule("Assign")
def assign(low: Lowerer, s: Node) -> None:
    """`name = op(args)`. The reference rule: read operands, apply the op, bind the result.
    `dtype` is None in every combine_states body, so the fallback is load-bearing."""
    args = [low.read(name) for name in s.args]
    ty = ir.ty(s.dtype) or args[0].ty
    low.assign(s.name, low.apply(s.op, ty, args))


@rule("Let")
def let(low: Lowerer, s: Node) -> None:
    """`name = value`: a pure expression (a literal, a precomputed index), converted to `dtype`."""
    ...


@rule("Init")
def init(low: Lowerer, s: Node) -> None:
    """Carried state seeded with a literal: declare a Variable. seed=False loops rely on it."""
    ...


@rule("Accum")
def accum(low: Lowerer, s: Node) -> None:
    """`name <- op(base, value)`, where `base` defaults to `name` itself."""
    ...


@rule("Load")
def load(low: Lowerer, s: Node) -> None:
    """`names` <- consecutive elements of `input` from `index`; dtype defaults to the buffer's."""
    ...


@rule("Write")
def write(low: Lowerer, s: Node) -> None:
    """`output[index..]` <- `values`; `atomic` adds instead of storing. swizzle != NONE -> Unsupported."""
    ...


def load_values(low: Lowerer, arr: Array, addr: Addr, ty: Ty, count: int) -> list[Reg]:
    """`ld.{space}[.v2|.v4].{ty}` for 1, 2 or 4 values; any other count as scalar loads at `+i*nbytes`."""
    ...


def store_values(
    low: Lowerer,
    arr: Array,
    addr: Addr,
    ty: Ty,
    values: Sequence[Value],
    atomic: bool,
) -> None:
    """`st.{space}[.vN].{ty}` likewise, immediates materialized first;
    atomic -> one `red.{space}.add.{ty}` per element."""
    ...
