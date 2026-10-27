"""Expressions -> operands.

Indices, predicates and loop bounds are small trees: Var, Literal, BinaryExpr, TernaryExpr,
and the documented but unseen CastExpr, FuncCallExpr and Builtin. Each function returns
where its result lives: a register, or an immediate when every leaf is constant, so
constants fold on the way up.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .. import ir, ops
from ..errors import InvalidKernel, Unsupported
from ..ir import Node
from ..ptx import Addr, Imm, Reg, Sym, Value, cvt, emit, op, reg, setp
from ..types import F32, INDEX, PRED, S64, U32, U64
from .lowerer import read
from .symbols import Array

ARITH = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide", "%": "remainder"}
COMPARE = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
LOGIC = {"&&": "and", "||": "or", "and": "and", "or": "or"}
SPECIAL = {"threadIdx": "tid", "blockIdx": "ctaid", "blockDim": "ntid", "gridDim": "nctaid"}


def value(e: Node | int | float) -> Value:
    if isinstance(e, (int, float)):
        return Imm(e, F32 if isinstance(e, float) else INDEX)
    match ir.kind(e):
        case "Var":
            return read(e.name)
        case "Literal":
            return Imm(e.value, ir.ty(e.dtype) or value(e.value).ty)
        case "BinaryExpr":
            return binary(e)
        case "TernaryExpr":
            return ternary(e)
        case "CastExpr":
            return cvt(value(e.expr), ir.ty(e.dtype))
        case "FuncCallExpr":
            args = [value(a) for a in e.args]
            ty = next((a.ty for a in args if a.ty.is_float), args[0].ty)
            return ops.apply(e.name, ty, [cvt(a, ty) for a in args])
        case "Builtin":
            space, axis = e.name.split(".")
            return cvt(op("mov", U32, Sym(f"%{SPECIAL[space]}.{axis}")), INDEX)
        case k:
            raise Unsupported(f"expression {k}")


def predicate(e: Node) -> Reg:
    v = value(e)
    return v if v.ty == PRED else setp("ne", v, v.ty(0))


def binary(e: Node) -> Value:
    if e.op in ARITH:
        return arith(e.op, value(e.left), value(e.right))
    if e.op in COMPARE:
        a, b = unify(value(e.left), value(e.right))
        return setp(COMPARE[e.op], a, b)
    if e.op in LOGIC:
        return op(LOGIC[e.op], PRED, predicate(e.left), predicate(e.right))
    raise Unsupported(f"operator {e.op!r}")


def arith(operator: str, a: Value, b: Value) -> Value:
    a, b = unify(a, b)
    if isinstance(a, Imm) and isinstance(b, Imm):
        return a.ty(fold(operator, a.value, b.value))
    return ops.apply(ARITH[operator], a.ty, [a, b])


def fold(operator: str, x: int | float, y: int | float) -> int | float:
    match operator:
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


def ternary(e: Node) -> Value:
    p = predicate(e.cond)
    a, b = unify(value(e.if_true), value(e.if_false))
    return op("selp", a.ty, a, b, p)


def unify(a: Value, b: Value) -> tuple[Value, Value]:
    """A common type for a binary op: float wins, then the wider."""
    ty = max(a.ty, b.ty, key=lambda t: (t.is_float, t.bits))
    return cvt(a, ty), cvt(b, ty)


def linear(e: Node) -> tuple[Value | None, int]:
    """`(a0 * 4) + 3` -> (a0 * 4, 3); a constant expression -> (None, 3).

    What turns an unrolled body's neighbouring loads into `[base+12]` displacements and a
    grouped write into one vector store at the group's base.
    """
    if ir.kind(e) == "BinaryExpr" and e.op in ("+", "-"):
        (lv, lc), (rv, rc) = linear(e.left), linear(e.right)
        c = lc + rc if e.op == "+" else lc - rc
        match lv, rv:
            case None, None:
                return None, c
            case _, None:
                return lv, c
            case None, _:
                return (rv if e.op == "+" else ops.apply("negative", rv.ty, [rv])), c
            case _:
                return arith(e.op, lv, rv), c
    v = value(e)
    if isinstance(v, Imm) and not v.ty.is_float:
        return None, int(v.value)
    return v, 0


def address(arr: Array, index: Sequence[Node]) -> Addr:
    """`arr[index]`, row-major; the constant part of every coordinate becomes the displacement."""
    if len(index) > len(arr.shape):
        raise InvalidKernel(f"{len(index)} coordinates into {arr.name}{list(arr.shape)}")
    flat: Value | None = None
    elems = 0
    for e, stride in zip(index, arr.strides):  # fewer coordinates than axes: the rest are 0 (`sum_1[]`)
        var, c = linear(e)
        elems += c * stride
        if var is None:
            continue
        var = cvt(var, INDEX)
        if flat is None:
            flat = var if stride == 1 else op("mul.lo", INDEX, var, INDEX(stride))
        elif stride == 1:
            flat = op("add", INDEX, flat, var)
        else:
            flat = op("mad.lo", INDEX, var, INDEX(stride), flat)
    return locate(arr, flat, elems)


def locate(arr: Array, flat: Value | None, elems: int = 0) -> Addr:
    """The byte address of element `flat + elems`: 64-bit arithmetic for global, 32-bit for shared."""
    offset = arr.offset + elems * arr.ty.nbytes
    if flat is None:
        return Addr(arr.base, offset)
    if arr.space == "global":
        nbytes = reg(S64)
        emit("mul.wide.s32", nbytes, flat, INDEX(arr.ty.nbytes))
        base = op("add", U64, arr.base, cvt(nbytes, U64))
        return Addr(base, offset)
    nbytes = op("mul.lo", U32, cvt(flat, U32), U32(arr.ty.nbytes))
    base = op("add", U32, arr.base, nbytes)
    return Addr(base, offset)
