"""Expressions -> operands.

Indices, predicates and loop bounds are small trees: Var, Literal, BinaryExpr, TernaryExpr,
and the documented but unseen CastExpr, FuncCallExpr and Builtin. Each function returns
where its result lives: a register, or an immediate when every leaf is constant, so
constants fold on the way up.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .. import ir, ops
from ..errors import InvalidKernel, Unsupported
from ..ir import Node
from ..ptx import Addr, Imm, Reg, Sym, Value
from ..types import F32, INDEX, PRED, S64, U32, U64, Ty
from .symbols import Array

if TYPE_CHECKING:
    from .lowerer import Lowerer

ARITH = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide", "%": "remainder"}
COMPARE = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
LOGIC = {"&&": "and", "||": "or", "and": "and", "or": "or"}
SPECIAL = {"threadIdx": "tid", "blockIdx": "ctaid", "blockDim": "ntid", "gridDim": "nctaid"}


def value(low: Lowerer, e: Node | int | float) -> Value:
    if isinstance(e, (int, float)):
        return Imm(e, F32 if isinstance(e, float) else INDEX)
    match ir.kind(e):
        case "Var":
            return low.read(e.name)
        case "Literal":
            return Imm(e.value, ir.ty(e.dtype) or value(low, e.value).ty)
        case "BinaryExpr":
            return binary(low, e)
        case "TernaryExpr":
            return ternary(low, e)
        case "CastExpr":
            return low.b.cvt(value(low, e.expr), ir.ty(e.dtype))
        case "FuncCallExpr":
            args = [value(low, a) for a in e.args]
            ty = next((a.ty for a in args if a.ty.is_float), args[0].ty)
            return ops.apply(low.b, e.name, ty, [low.b.cvt(a, ty) for a in args])
        case "Builtin":
            space, axis = e.name.split(".")
            return low.b.cvt(low.b.op("mov", U32, Sym(f"%{SPECIAL[space]}.{axis}")), INDEX)
        case k:
            raise Unsupported(f"expression {k}")


def predicate(low: Lowerer, e: Node) -> Reg:
    v = value(low, e)
    return v if v.ty == PRED else low.b.setp("ne", v, Imm(0, v.ty))


def binary(low: Lowerer, e: Node) -> Value:
    if e.op in ARITH:
        return arith(low, e.op, value(low, e.left), value(low, e.right))
    if e.op in COMPARE:
        a, b = unify(low, value(low, e.left), value(low, e.right))
        return low.b.setp(COMPARE[e.op], a, b)
    if e.op in LOGIC:
        return low.b.op(LOGIC[e.op], PRED, predicate(low, e.left), predicate(low, e.right))
    raise Unsupported(f"operator {e.op!r}")


def arith(low: Lowerer, op: str, a: Value, b: Value) -> Value:
    a, b = unify(low, a, b)
    if isinstance(a, Imm) and isinstance(b, Imm):
        return Imm(fold(op, a.value, b.value), a.ty)
    return ops.apply(low.b, ARITH[op], a.ty, [a, b])


def fold(op: str, x: int | float, y: int | float) -> int | float:
    match op:
        case "+":
            return x + y
        case "-":
            return x - y
        case "*":
            return x * y
        case "/":
            return x / y if isinstance(x, float) or isinstance(y, float) else math.trunc(x / y)  # C division
        case _:
            return x - y * math.trunc(x / y)  # C remainder


def ternary(low: Lowerer, e: Node) -> Value:
    p = predicate(low, e.cond)
    a, b = unify(low, value(low, e.if_true), value(low, e.if_false))
    return low.b.op("selp", a.ty, a, b, p)


def unify(low: Lowerer, a: Value, b: Value) -> tuple[Value, Value]:
    """A common type for a binary op: float wins, then the wider."""
    ty = max(a.ty, b.ty, key=lambda t: (t.is_float, t.bits))
    return low.b.cvt(a, ty), low.b.cvt(b, ty)


def linear(low: Lowerer, e: Node) -> tuple[Value | None, int]:
    """`(a0 * 4) + 3` -> (a0 * 4, 3); a constant expression -> (None, 3).

    What turns an unrolled body's neighbouring loads into `[base+12]` displacements and a
    grouped write into one vector store at the group's base.
    """
    if ir.kind(e) == "BinaryExpr" and e.op in ("+", "-"):
        (lv, lc), (rv, rc) = linear(low, e.left), linear(low, e.right)
        c = lc + rc if e.op == "+" else lc - rc
        match lv, rv:
            case None, None:
                return None, c
            case _, None:
                return lv, c
            case None, _:
                return (rv if e.op == "+" else ops.apply(low.b, "negative", rv.ty, [rv])), c
            case _:
                return arith(low, e.op, lv, rv), c
    v = value(low, e)
    if isinstance(v, Imm) and not v.ty.is_float:
        return None, int(v.value)
    return v, 0


def address(low: Lowerer, arr: Array, index: Sequence[Node]) -> Addr:
    """`arr[index]`, row-major; the constant part of every coordinate becomes the displacement."""
    if len(index) > len(arr.shape):
        raise InvalidKernel(f"{len(index)} coordinates into {arr.name}{list(arr.shape)}")
    flat: Value | None = None
    elems = 0
    for e, stride in zip(index, arr.strides):  # fewer coordinates than axes: the rest are 0 (`sum_1[]`)
        var, c = linear(low, e)
        elems += c * stride
        if var is None:
            continue
        var = low.b.cvt(var, INDEX)
        if flat is None:
            flat = var if stride == 1 else low.b.op("mul.lo", INDEX, var, Imm(stride, INDEX))
        elif stride == 1:
            flat = low.b.op("add", INDEX, flat, var)
        else:
            flat = low.b.op("mad.lo", INDEX, var, Imm(stride, INDEX), flat)
    return locate(low, arr, flat, elems)


def locate(low: Lowerer, arr: Array, flat: Value | None, elems: int = 0) -> Addr:
    """The byte address of element `flat + elems`: 64-bit arithmetic for global, 32-bit for shared."""
    offset = arr.offset + elems * arr.ty.nbytes
    if flat is None:
        return Addr(arr.base, offset)
    if arr.space == "global":
        nbytes = low.b.reg(S64)
        low.b.emit("mul.wide.s32", nbytes, flat, Imm(arr.ty.nbytes, INDEX))
        return Addr(low.b.op("add", U64, arr.base, low.b.cvt(nbytes, U64)), offset)
    nbytes = low.b.op("mul.lo", U32, low.b.cvt(flat, U32), Imm(arr.ty.nbytes, U32))
    return Addr(low.b.op("add", U32, arr.base, nbytes), offset)
