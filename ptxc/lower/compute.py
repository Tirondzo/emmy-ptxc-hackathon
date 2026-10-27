"""Scalar compute: Assign, Let, Init, Accum, Load, Write."""

from __future__ import annotations

from collections.abc import Sequence

from .. import ir
from ..errors import InvalidKernel, Unsupported
from ..ir import Node
from ..ptx import Addr, Reg, Value, Vec, opcode
from ..types import Ty
from . import expr
from .lowerer import Lowerer, rule
from .symbols import Array, Variable


@rule("Assign")
def assign(low: Lowerer, s: Node) -> None:
    """`name = op(args)`. `dtype` is None in every combine_states body, so the fallback is load-bearing."""
    args = [low.read(name) for name in s.args]
    ty = ir.ty(s.dtype) or args[0].ty
    low.assign(s.name, low.apply(s.op, ty, [low.b.cvt(a, ty) for a in args]))


@rule("Let")
def let(low: Lowerer, s: Node) -> None:
    v = expr.value(low, s.value)
    low.assign(s.name, low.b.cvt(v, ir.ty(s.dtype) or v.ty))


@rule("Init")
def init(low: Lowerer, s: Node) -> None:
    v = expr.value(low, s.identity)
    low.declare(s.name, low.b.cvt(v, ir.ty(s.dtype) or v.ty))


@rule("Accum")
def accum(low: Lowerer, s: Node) -> None:
    """`name <- op(base, value)`, where `base` defaults to `name` itself."""
    if not isinstance(low.syms.get(s.name), Variable):
        raise InvalidKernel(f"accumulator {s.name!r} is not seeded by its loop or an Init")
    args = [low.read(s.base or s.name), low.read(s.value)]
    ty = ir.ty(s.dtype) or args[0].ty
    low.assign(s.name, low.apply(s.op, ty, [low.b.cvt(a, ty) for a in args]))


@rule("Load")
def load(low: Lowerer, s: Node) -> None:
    arr = low.array(s.input)
    ty = ir.ty(s.dtype) or arr.ty
    regs = load_values(low, arr, expr.address(low, arr, s.index), ty, len(s.names))
    for name, reg in zip(s.names, regs, strict=True):
        low.assign(name, reg)


@rule("Write")
def write(low: Lowerer, s: Node) -> None:
    if s.swizzle != "NONE":
        raise Unsupported(f"swizzle {s.swizzle}")
    arr = low.array(s.output)
    ty = ir.ty(s.value_dtype) or arr.ty
    values = [low.b.cvt(low.read(name), ty) for name in s.values]
    store_values(low, arr, expr.address(low, arr, s.index), ty, values, s.atomic)


def load_values(low: Lowerer, arr: Array, addr: Addr, ty: Ty, count: int) -> list[Reg]:
    regs = [low.b.reg(ty) for _ in range(count)]
    for vec, at, dst in accesses(addr, ty, regs):
        low.b.emit(opcode("ld", arr.space, vec, ty.ptx), dst, at)
    return regs


def store_values(low: Lowerer, arr: Array, addr: Addr, ty: Ty, values: Sequence[Value], atomic: bool) -> None:
    regs = [low.b.materialize(v) for v in values]
    if atomic:
        for i, src in enumerate(regs):
            low.b.emit(opcode("red", arr.space, "add", ty.ptx), Addr(addr.base, addr.offset + i * ty.nbytes), src)
        return
    for vec, at, src in accesses(addr, ty, regs):
        low.b.emit(opcode("st", arr.space, vec, ty.ptx), at, src)


def accesses(addr: Addr, ty: Ty, regs: list[Reg]) -> list[tuple[str | None, Addr, Reg | Vec]]:
    """One `.v2` / `.v4` access when the registers form a legal 8- or 16-byte vector, else one per register."""
    n = len(regs)
    if n == 1:
        return [(None, addr, regs[0])]
    if n in (2, 4) and n * ty.nbytes <= 16:
        return [(f"v{n}", addr, Vec(tuple(regs)))]
    return [(None, Addr(addr.base, addr.offset + i * ty.nbytes), reg) for i, reg in enumerate(regs)]
