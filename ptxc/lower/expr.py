"""Expressions -> operands.

Indices, predicates and loop bounds are small trees: Var, Literal, BinaryExpr, TernaryExpr,
and the documented but unseen CastExpr, FuncCallExpr and Builtin. Each function returns
where its result lives: a register, or an immediate when every leaf is constant, so
constants fold on the way up.

Indices are where a kernel's integer work is: `affine` keeps one as sum(coefficient * register)
+ constant, so constants become displacements and equal terms cancel, which ptxas does not do
for a loop it sees rolled.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

from .. import ir, ops
from ..errors import InvalidKernel, Unsupported
from ..ir import Node
from ..ptx import Addr, Imm, Reg, Sym, Value, cvt, emit, materialize, op, reg, select, truth, unify
from ..types import F32, INDEX, PRED, S64, U32, U64, ty_of
from .lowerer import lowering, options, read, target
from .symbols import Array, Bound, Variable

ARITH = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide", "//": "divide", "%": "remainder"}
COMPARE = {"==": "equal", "!=": "not_equal", "<": "less", "<=": "less_equal", ">": "greater", ">=": "greater_equal"}
LOGIC = {"&&": "and", "||": "or", "and": "and", "or": "or"}
SPECIAL = {"thread_idx": "tid", "block_idx": "ctaid", "block_dim": "ntid", "grid_dim": "nctaid"}  # as emmy spells them


def value(e: Node | int | float) -> Value:
    if isinstance(e, (int, float)):
        return Imm(e, F32 if isinstance(e, float) else INDEX)
    match ir.kind(e):
        case "Var":
            return read(e.name)
        case "Literal":
            return Imm(e.value, ty_of(e.dtype))
        case "BinaryExpr":
            return binary(e)
        case "TernaryExpr":
            return ternary(e)
        case "CastExpr":
            return cvt(value(e.expr), ir.ty(e.dtype))
        case "FuncCallExpr":
            args = [value(a) for a in e.args]
            ty = next((a.ty for a in args if a.ty.is_float), args[0].ty)
            return ops.apply(e.name, ty, args)
        case "Builtin":
            return builtin(e.name)
        case k:
            raise Unsupported(f"expression {k}")


def builtin(name: str) -> Value:
    """`thread_idx.x` and friends -> the special register; `warp_size` is the target's."""
    if name == "warp_size":
        return INDEX(target().warp)
    space, _, axis = name.partition(".")
    if space not in SPECIAL or axis not in ("x", "y", "z"):
        raise Unsupported(f"builtin {name!r}")
    return cvt(op("mov", U32, Sym(f"%{SPECIAL[space]}.{axis}")), INDEX)


def predicate(e: Node) -> Reg:
    return truth(value(e))


def binary(e: Node) -> Value:
    if e.op in ARITH:
        return arith(e.op, value(e.left), value(e.right))
    if e.op in COMPARE:
        return ops.apply(COMPARE[e.op], PRED, [value(e.left), value(e.right)])  # the one table of comparisons
    if e.op in LOGIC:
        return op(LOGIC[e.op], PRED, predicate(e.left), predicate(e.right))
    raise Unsupported(f"operator {e.op!r}")


def arith(operator: str, a: Value, b: Value) -> Value:
    a, b = unify(a, b)
    if isinstance(a, Imm) and isinstance(b, Imm):
        return a.ty(fold(operator, a.value, b.value))
    return ops.apply(ARITH[operator], a.ty, [a, b])


def fold(operator: str, x: int | float, y: int | float) -> int | float:
    """Both operands already of one type (`unify`), so an int result means an int operation."""
    if operator in ("/", "//", "%") and y == 0:
        raise InvalidKernel("division by a constant zero")
    match operator:
        case "+":
            return x + y
        case "-":
            return x - y
        case "*":
            return x * y
        case "/" | "//":  # C division, which is what `//` renders to in emmy too: toward zero, like div.s32
            return x / y if isinstance(x, float) else trunc_div(x, y)
        case _:
            return x - y * trunc_div(x, y)  # C remainder


def trunc_div(x: int | float, y: int | float) -> int:
    """Toward zero; in integers, since `x / y` rounds past 2^53."""
    if isinstance(x, float):
        return math.trunc(x / y)
    q = abs(x) // abs(y)
    return q if (x < 0) == (y < 0) else -q


def ternary(e: Node) -> Value:
    p = predicate(e.cond)
    a, b = unify(value(e.if_true), value(e.if_false))
    return select(a.ty, a, b, p)


# ---- indices ------------------------------------------------------------------------


type Terms = dict[str, tuple[Reg, int]]  # register name -> (register, coefficient), coefficients never zero


def affine(e: Node) -> tuple[Terms, int]:
    """`e` as sum(coefficient * register) + constant: `(a1_b * 64 + a1_u * 2 + 1) - a1_b * 64` is
    `{a1_u: 2}, 1`. `+`, `-`, `*` by a constant and an unrolled axis's offset distribute; anything
    else (`/`, `%`, a ternary) is one opaque term."""
    match ir.kind(e):
        case "Var":
            match lowering().syms.lookup(e.name):
                case Bound(Imm() as v, offset) if not v.ty.is_float:
                    return {}, int(v.value) + offset
                case Bound(Reg() as r, offset) if not r.ty.is_float:
                    return {r.name: (r, 1)}, offset
                case Variable(reg=r) if not r.ty.is_float:
                    return {r.name: (r, 1)}, 0
        case "Literal" if not ty_of(e.dtype).is_float:
            return {}, int(e.value)
        case "BinaryExpr" if e.op in ("+", "-"):
            (lt, lc), (rt, rc) = affine(e.left), affine(e.right)
            sign = 1 if e.op == "+" else -1
            return combine(lt, scale(rt, sign)), lc + sign * rc
        case "BinaryExpr" if e.op == "*":
            (lt, lc), (rt, rc) = affine(e.left), affine(e.right)
            if not lt:
                return scale(rt, lc), lc * rc
            if not rt:
                return scale(lt, rc), lc * rc
            return term(arith("*", realize(lt, lc), realize(rt, rc)))
    return term(cached(e))


def term(v: Value) -> tuple[Terms, int]:
    """A value as an affine form: a constant, or one register with coefficient 1."""
    if isinstance(v, Imm):
        return {}, int(v.value)
    r = materialize(cvt(v, INDEX))
    return {r.name: (r, 1)}, 0


def scale(terms: Terms, k: int) -> Terms:
    return {} if k == 0 else {n: (r, c * k) for n, (r, c) in terms.items()}


def combine(a: Terms, b: Terms) -> Terms:
    out = dict(a)
    for n, (r, c) in b.items():
        c += out.get(n, (r, 0))[1]
        if c:
            out[n] = (r, c)
        else:
            out.pop(n, None)
    return out


def realize(terms: Terms, const: int = 0) -> Value:
    """The affine form as one value: a mul / mad / add chain, or an immediate."""
    acc: Value | None = None
    for _, (r, c) in sorted(terms.items()):
        if acc is None:
            acc = r if c == 1 else op("mul.lo", INDEX, r, INDEX(c))
        elif c == 1:
            acc = op("add", INDEX, acc, r)
        else:
            acc = op("mad.lo", INDEX, r, INDEX(c), acc)
    if acc is None:
        return INDEX(const)
    return op("add", INDEX, acc, INDEX(const)) if const else acc


def cached(e: Node) -> Value:
    """`value(e)`, reused within the scope when `e` reads only registers that never change
    (`Options.cse`; ptxas would merge the duplicates itself)."""
    syms = lowering().syms
    if not options().cse or (key := key_of(e)) is None:
        return value(e)
    if (v := syms.recall(key)) is None:
        v = value(e)
        syms.remember(key, v)
    return v


def key_of(e: Node) -> Hashable | None:
    """A structural key for `e` under the current bindings; None when it reads a mutable register."""
    syms = lowering().syms
    match ir.kind(e):
        case "Var":
            match syms.get(e.name):
                case Bound(Imm() as v, offset):
                    return ("imm", v.value, offset)
                case Bound(Reg() as r, offset) if not syms.is_variable(r):
                    return ("reg", r.name, offset)
                case _:
                    return None
        case "Literal":
            return ("lit", e.value, str(e.dtype))
        case "BinaryExpr":
            return keys("bin", e.op, e.left, e.right)
        case "TernaryExpr":
            return keys("tern", None, e.cond, e.if_true, e.if_false)
        case "CastExpr":
            return keys("cast", str(e.dtype), e.expr)
        case "FuncCallExpr":
            return keys("call", e.name, *e.args)
        case "Builtin":
            return ("builtin", e.name)
        case _:
            return None


def keys(tag: str, info: object, *parts: Node) -> Hashable | None:
    ks = [key_of(p) for p in parts]
    return None if any(k is None for k in ks) else (tag, info, *ks)


def address(arr: Array, index: Sequence[Node]) -> Addr:
    """`arr[index]`, row-major; the constant part of every coordinate becomes the displacement.
    One coordinate per axis, or none for element 0 (`sum_1[]`): any other count is refused
    rather than flattened by a guess."""
    if index and len(index) != len(arr.shape):
        raise InvalidKernel(f"{len(index)} coordinates into {arr.name}{list(arr.shape)}")
    terms: Terms = {}
    elems = 0
    for e, stride in zip(index, arr.strides):
        t, c = affine(e)
        terms = combine(terms, scale(t, stride))
        elems += c * stride
    return locate(arr, terms, elems)


def locate_value(arr: Array, flat: Value | None, elems: int = 0) -> Addr:
    """The byte address of element `flat + elems`, for callers holding a value rather than an index."""
    if flat is None:
        return locate(arr, {}, elems)
    t, c = term(flat)
    return locate(arr, t, elems + c)


def locate(arr: Array, terms: Terms, elems: int) -> Addr:
    """`[base+displacement]`: the constant part in bytes (plus the array's place in its pool) is the
    displacement; the terms make the base, 64-bit for global, 32-bit for shared, reused within the
    scope under `Options.cse`."""
    offset = arr.offset + elems * arr.ty.nbytes
    if not terms:
        return Addr(arr.base, offset)
    syms = lowering().syms
    key: Hashable | None = ("base", arr.base.name, arr.space, tuple((n, c) for n, (_, c) in sorted(terms.items())))
    if not options().cse or any(syms.is_variable(r) for r, _ in terms.values()):
        key = None  # a mutable term: the base is not a function of the key
    if key is not None and (base := syms.recall(key)) is not None:
        return Addr(base, offset)
    flat = realize(terms)
    if arr.space == "global":
        nbytes = reg(S64)
        emit("mul.wide.s32", nbytes, flat, INDEX(arr.ty.nbytes))
        base = op("add", U64, arr.base, cvt(nbytes, U64))
    else:
        base = op("add", U32, arr.base, op("mul.lo", U32, cvt(flat, U32), U32(arr.ty.nbytes)))
    if key is not None:
        syms.remember(key, base)
    return Addr(base, offset)
