"""Control flow: Tile, Loop, StridedLoop, Cond."""

from __future__ import annotations

import math
from contextlib import nullcontext

from .. import ir
from ..errors import InvalidKernel, Unsupported
from ..ir import Body, Node
from ..ptx import Reg, Sym, Value, cvt, for_, if_, mov, op, reg, setp
from ..types import INDEX, U32
from . import expr
from .lowerer import assign, bind, declare, lower, lowering, rule, scope, tid
from .symbols import Bound


@rule("Tile")
def tile(s: Node) -> None:
    """The block: decode this thread's cell, guard, run the body once.

    `raster_*` are ignored on purpose: rasterization permutes which block owns which tile,
    and every bijection is correct.
    """
    launch = lowering().kernel.launch
    if s.aux_threads:
        raise Unsupported("aux_threads")
    if s.block_threads not in (None, launch.block):
        raise InvalidKernel(f"Tile wants {s.block_threads} threads per block, meta.json launches {launch.block}")
    cells = math.prod(ir.extent(a) for a in s.axes)
    if launch.threads < cells:
        raise InvalidKernel(f"{cells} cells for {launch.threads} launched threads")
    gid = global_id()
    guard = if_(setp("lt", gid, U32(cells))) if launch.threads > cells else nullcontext()
    with scope(), guard:
        decode_axes(gid, s.axes)
        lower(s.body)


def global_id() -> Reg:
    ctaid = op("mov", U32, Sym("%ctaid.x"))
    ntid = op("mov", U32, Sym("%ntid.x"))
    return op("mad.lo", U32, ctaid, ntid, tid())


def decode_axes(gid: Reg, axes: Body) -> None:
    """axis_i = (gid / stride_i) % extent_i, row-major, unsigned (gid is never negative, and a signed
    division by a constant pays a sign fix-up); the leading axis is bounded by the guard."""
    extents = [ir.extent(a) for a in axes]
    for i, axis in enumerate(axes):
        stride = math.prod(extents[i + 1 :])
        v: Value = gid if stride == 1 else op("div", U32, gid, U32(stride))
        if i:
            v = op("rem", U32, v, U32(extents[i]))
        assign(axis.name, cvt(v, INDEX))


@rule("Loop")
def loop(s: Node) -> None:
    """`for axis in 0..extent`. `unroll` is ignored: a runtime loop is always correct."""
    counted_loop(s.axis.name, INDEX(0), INDEX(ir.extent(s.axis)), INDEX(1), s.body, s.seed)


@rule("StridedLoop")
def strided_loop(s: Node) -> None:
    end = INDEX(ir.extent(s.axis)) if s.end is None else expr.value(s.end)
    counted_loop(s.axis.name, expr.value(s.start), end, expr.value(s.step), s.body, s.seed)


def counted_loop(axis: str, start: Value, end: Value, step: Value, body: Body, seeded: bool) -> None:
    """Seed first, in the enclosing scope; then a fresh counter bound to `axis` in a new scope."""
    if seeded:
        seed_accumulators(body)
    var = reg(INDEX)
    mov(var, cvt(start, INDEX))
    with scope(), for_(var, cvt(end, INDEX), cvt(step, INDEX)):
        bind(axis, Bound(var))
        lower(body)


def seed_accumulators(body: Body) -> None:
    """Declare each Accum of the loop's own body at its identity: a fresh Variable in the enclosing
    scope, shadowing an outer name of the same spelling. An Accum inside a nested block belongs
    to whatever that name denotes there."""
    accums = {s.name: s for s in body if ir.kind(s) == "Accum"}
    for name, s in accums.items():
        if (ty := ir.ty(s.dtype)) is None:
            raise Unsupported(f"untyped accumulator {name!r}")
        declare(name, ty(ir.op_identity(s.op)))


@rule("Cond")
def cond(s: Node) -> None:
    with if_(expr.predicate(s.cond)) as branch:
        with scope():
            lower(s.body)
        if s.else_body:
            branch.else_()
            with scope():
                lower(s.else_body)
