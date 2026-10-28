"""Control flow: Tile and Cond. The loops are in `loops.py`."""

from __future__ import annotations

import math
from contextlib import nullcontext

from .. import ir
from ..errors import InvalidKernel, Unsupported
from ..ir import Body, Node
from ..ptx import Reg, Sym, Value, cvt, if_, op, setp
from ..types import INDEX, U32
from . import expr
from .lowerer import assign, lower, lowering, rule, scope, tid


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
    division by a constant pays a sign fix-up); the leading axis is bounded by the guard. Every axis
    is then known to lie in [0, extent), which `loops.trip_count` uses."""
    extents = [ir.extent(a) for a in axes]
    for i, axis in enumerate(axes):
        stride = math.prod(extents[i + 1 :])
        v: Value = gid if stride == 1 else op("div", U32, gid, U32(stride))
        if i:
            v = op("rem", U32, v, U32(extents[i]))
        assign(axis.name, cvt(v, INDEX))
        lowering().axes[axis.name] = extents[i]


@rule("Cond")
def cond(s: Node) -> None:
    with if_(expr.predicate(s.cond)) as branch:
        with scope():
            lower(s.body)
        if s.else_body:
            branch.else_()
            with scope():
                lower(s.else_body)
