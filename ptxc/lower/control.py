"""Control flow: Tile, Loop, StridedLoop, Cond."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import nullcontext

from .. import ir
from ..errors import InvalidKernel, Unsupported
from ..ir import Body, Node
from ..ptx import Imm, Reg, Sym, Value
from ..types import INDEX, U32
from . import expr
from .lowerer import Lowerer, rule
from .symbols import Bound


@rule("Tile")
def tile(low: Lowerer, s: Node) -> None:
    """The block: decode this thread's cell, guard, run the body once.

    `raster_*` are ignored on purpose: rasterization permutes which block owns which tile,
    and every bijection is correct.
    """
    launch = low.kernel.launch
    if s.aux_threads:
        raise Unsupported("aux_threads")
    if s.block_threads not in (None, launch.block[0]):
        raise InvalidKernel(f"Tile wants {s.block_threads} threads per block, meta.json launches {launch.block[0]}")
    cells = math.prod(ir.extent(a) for a in s.axes)
    if launch.threads < cells:
        raise InvalidKernel(f"{cells} cells for {launch.threads} launched threads")
    gid = global_id(low)
    guard = low.b.if_(low.b.setp("lt", gid, Imm(cells, INDEX))) if launch.threads > cells else nullcontext()
    with low.syms.scope(), guard:
        decode_axes(low, gid, s.axes)
        low.block(s.body)


def global_id(low: Lowerer) -> Reg:
    ctaid = low.b.op("mov", U32, Sym("%ctaid.x"))
    ntid = low.b.op("mov", U32, Sym("%ntid.x"))
    tid = low.b.op("mov", U32, Sym("%tid.x"))
    return low.b.cvt(low.b.op("mad.lo", U32, ctaid, ntid, tid), INDEX)


def decode_axes(low: Lowerer, gid: Reg, axes: Body) -> None:
    """axis_i = (gid / stride_i) % extent_i, row-major; the leading axis is bounded by the guard."""
    extents = [ir.extent(a) for a in axes]
    for i, axis in enumerate(axes):
        stride = math.prod(extents[i + 1 :])
        v: Value = gid if stride == 1 else low.b.op("div", INDEX, gid, Imm(stride, INDEX))
        if i:
            v = low.b.op("rem", INDEX, v, Imm(extents[i], INDEX))
        low.assign(axis.name, v)


@rule("Loop")
def loop(low: Lowerer, s: Node) -> None:
    """`for axis in 0..extent`. `unroll` is ignored: a runtime loop is always correct."""
    end = Imm(ir.extent(s.axis), INDEX)
    counted_loop(low, s.axis.name, Imm(0, INDEX), end, Imm(1, INDEX), s.body, s.seed)


@rule("StridedLoop")
def strided_loop(low: Lowerer, s: Node) -> None:
    end = Imm(ir.extent(s.axis), INDEX) if s.end is None else expr.value(low, s.end)
    counted_loop(low, s.axis.name, expr.value(low, s.start), end, expr.value(low, s.step), s.body, s.seed)


def counted_loop(low: Lowerer, axis: str, start: Value, end: Value, step: Value, body: Body, seed: bool) -> None:
    """Seed first, in the enclosing scope; then a fresh counter bound to `axis` in a new scope."""
    if seed:
        seed_accumulators(low, body)
    var = low.b.reg(INDEX)
    low.b.mov(var, low.b.cvt(start, INDEX))
    with low.syms.scope(), low.b.loop(var, low.b.cvt(end, INDEX), low.b.cvt(step, INDEX)):
        low.syms.bind(axis, Bound(var))
        low.block(body)


def seed_accumulators(low: Lowerer, body: Body) -> None:
    """Seed each Accum of the loop's immediate body (through Cond, not nested loops) at its identity."""
    accums = {s.name: s for s in immediate(body) if ir.kind(s) == "Accum"}
    for name, s in accums.items():
        if (ty := ir.ty(s.dtype)) is None:
            raise Unsupported(f"untyped accumulator {name!r}")
        low.seed(name, Imm(ir.op_identity(s.op), ty))


def immediate(body: Body) -> Iterator[Node]:
    for s in body:
        yield s
        if ir.kind(s) == "Cond":
            yield from immediate(s.body)
            yield from immediate(s.else_body)


@rule("Cond")
def cond(low: Lowerer, s: Node) -> None:
    pred = expr.predicate(low, s.cond)
    with low.b.if_(pred) as branch:
        with low.syms.scope():
            low.block(s.body)
        if s.else_body:
            branch.else_()
            with low.syms.scope():
                low.block(s.else_body)
