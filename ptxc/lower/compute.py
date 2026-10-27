"""Scalar compute: Assign, Let, Init, Accum, Load, Write."""

from __future__ import annotations

from .. import ir, ops
from ..errors import InvalidKernel, Unsupported
from ..ir import Node
from ..ptx import Addr, Reg, Vec, cvt, dotted, emit, materialize, reg
from ..types import Ty
from . import expr
from .lowerer import array, assign, declare, lowering, read, rule
from .symbols import Array, Variable


@rule("Assign")
def assign_(s: Node) -> None:
    """`name = op(args)`. `dtype` is None in every combine_states body, so the fallback is load-bearing."""
    args = [read(name) for name in s.args]
    ty = ir.ty(s.dtype) or args[0].ty
    assign(s.name, ops.apply(ir.op_name(s.op), ty, [cvt(a, ty) for a in args]))


@rule("Let")
def let(s: Node) -> None:
    v = expr.value(s.value)
    assign(s.name, cvt(v, ir.ty(s.dtype) or v.ty))


@rule("Init")
def init(s: Node) -> None:
    v = expr.value(s.identity)
    declare(s.name, cvt(v, ir.ty(s.dtype) or v.ty))


@rule("Accum")
def accum(s: Node) -> None:
    """`name <- op(base, value)`, where `base` defaults to `name` itself."""
    if not isinstance(lowering().syms.get(s.name), Variable):
        raise InvalidKernel(f"accumulator {s.name!r} is not seeded by its loop or an Init")
    args = [read(s.base or s.name), read(s.value)]
    ty = ir.ty(s.dtype) or args[0].ty
    assign(s.name, ops.apply(ir.op_name(s.op), ty, [cvt(a, ty) for a in args]))


@rule("Load")
def load(s: Node) -> None:
    arr = array(s.input)
    ty = ir.ty(s.dtype) or arr.ty
    regs = load_values(arr, expr.address(arr, s.index), ty, len(s.names))
    for name, r in zip(s.names, regs, strict=True):
        assign(name, r)


@rule("Write")
def write(s: Node) -> None:
    if s.swizzle != "NONE":
        raise Unsupported(f"swizzle {s.swizzle}")
    arr = array(s.output)
    ty = ir.ty(s.value_dtype) or arr.ty
    regs = [materialize(cvt(read(name), ty)) for name in s.values]
    addr = expr.address(arr, s.index)
    if s.atomic:
        add_values(arr, addr, regs)
    else:
        store_values(arr, addr, regs)


def load_values(arr: Array, addr: Addr, ty: Ty, count: int) -> list[Reg]:
    regs = [reg(ty) for _ in range(count)]
    for vec, at, dst in accesses(addr, regs):
        emit(dotted("ld", arr.space, vec, ty.ptx), dst, at)
    return regs


def store_values(arr: Array, addr: Addr, regs: list[Reg]) -> None:
    ty = regs[0].ty
    for vec, at, src in accesses(addr, regs):
        emit(dotted("st", arr.space, vec, ty.ptx), at, src)


def add_values(arr: Array, addr: Addr, regs: list[Reg]) -> None:
    """`red.add` one element at a time: no vector form before sm_90."""
    for i, src in enumerate(regs):
        emit(f"red.{arr.space}.add.{src.ty.ptx}", Addr(addr.base, addr.offset + i * src.ty.nbytes), src)


def accesses(addr: Addr, regs: list[Reg]) -> list[tuple[str | None, Addr, Reg | Vec]]:
    """One `.v2` / `.v4` access when the registers form a legal 8- or 16-byte vector, else one per register."""
    n, ty = len(regs), regs[0].ty
    if n == 1:
        return [(None, addr, regs[0])]
    if n in (2, 4) and n * ty.nbytes <= 16:
        return [(f"v{n}", addr, Vec(tuple(regs)))]
    return [(None, Addr(addr.base, addr.offset + i * ty.nbytes), r) for i, r in enumerate(regs)]
