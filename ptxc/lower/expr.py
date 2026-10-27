"""Expressions -> operands.

Indices, predicates and loop bounds are small trees: Var, Literal, BinaryExpr, TernaryExpr,
and the documented but unseen CastExpr, FuncCallExpr and Builtin. Each function returns
where its result lives: a register, or an immediate when every leaf is constant, so
constants fold on the way up.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..ir import Node
from ..ptx import Addr, Reg, Value
from .symbols import Array

if TYPE_CHECKING:
    from .lowerer import Lowerer


def value(low: Lowerer, e: Node | int | float) -> Value:
    """Compile `e`; a bare Python number (loop bounds, an Init identity) becomes an Imm."""
    ...


def predicate(low: Lowerer, e: Node) -> Reg:
    """A comparison or logical expression -> a predicate register."""
    ...


def binary(low: Lowerer, e: Node) -> Value:
    """+ - * / % -> arithmetic in the unified type; == != < <= > >= -> setp;
    and / or -> and.pred / or.pred. Two immediates fold."""
    ...


def ternary(low: Lowerer, e: Node) -> Value:
    """`selp.{ty} d, if_true, if_false, cond`."""
    ...


def unify(low: Lowerer, a: Value, b: Value) -> tuple[Value, Value]:
    """A common type for a binary op: widen ints, promote int to float."""
    ...


def linear(low: Lowerer, e: Node) -> tuple[Value, int]:
    """Split `e` into (variable part, constant addend): `(a0 * 4) + 3` -> (a0 * 4, 3).

    What turns an unrolled body's neighbouring loads into `[base+12]` displacements and
    a grouped write into one vector store at the group's base.
    """
    ...


def address(low: Lowerer, arr: Array, index: Sequence[Node]) -> Addr:
    """The byte address of `arr[index]`, row-major over `arr.shape`.

    Variable parts are scaled by their stride and summed (mad.lo.s32); constant parts and
    `arr.offset` become the Addr's displacement. 64-bit arithmetic (mul.wide.s32) for
    global, 32-bit for shared. An empty index (a scalar buffer) is displacement 0.
    """
    ...
