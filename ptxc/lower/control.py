"""Control flow: Tile, Loop, StridedLoop, Cond."""

from __future__ import annotations

from ..ir import Body, Node
from ..ptx import Reg, Value
from . import expr
from .lowerer import Lowerer, rule


@rule("Tile")
def tile(low: Lowerer, s: Node) -> None:
    """The block: decode this thread's cell, guard, run the body once.

    cells = product of the axes' extents; gid = ctaid.x * ntid.x + tid.x. Every listed axis
    is decoded (`window` only records a block axis's parent). The guard `gid < cells` is not
    in the IR: emit it when the launch has spare threads; a launch with fewer threads than
    cells is InvalidKernel. `raster_*` are ignored on purpose: rasterization permutes which
    block owns which tile, and every bijection is correct. aux_threads != 0, or
    block_threads disagreeing with the launch block -> Unsupported.
    """
    ...


def global_id(low: Lowerer) -> Reg: ...


def decode_axes(low: Lowerer, gid: Reg, axes: Body) -> None:
    """axis_i = (gid / stride_i) % extent_i, stride_i the product of the extents after it;
    bind each axis name. The leading axis needs no `%`, the trailing one no `/`."""
    ...


@rule("Loop")
def loop(low: Lowerer, s: Node) -> None:
    """`for axis in 0..extent`. `unroll` is ignored: a runtime loop is always correct."""
    ...


@rule("StridedLoop")
def strided_loop(low: Lowerer, s: Node) -> None:
    """`for axis in start..end:step`; `end` None means the axis extent."""
    ...


def counted_loop(
    low: Lowerer,
    axis: str,
    start: Value,
    end: Value,
    step: Value,
    body: Body,
    seed: bool,
) -> None:
    """Seed first, in the enclosing scope; then a fresh counter bound to `axis` in a new scope."""
    ...


def seed_accumulators(low: Lowerer, body: Body) -> None:
    """Seed each distinct Accum of the loop's *immediate* body at its op's identity.

    Immediate means: descending into Cond bodies, never into nested loops, which seed their
    own. It runs before the loop opens its scope, because the state outlives the loop: a01
    reads an inner loop's accumulator in the outer body right after that loop. A seed=True
    loop whose Accums all sit in nested loops (matmul `_ks`) seeds nothing; seed=False
    loops (matmul `_ki`) rely on an Init.
    """
    ...


@rule("Cond")
def cond(low: Lowerer, s: Node) -> None:
    """`if cond: body [else: else_body]`, each body in its own scope."""
    pred = expr.predicate(low, s.cond)
    with low.b.if_(pred) as branch:
        with low.syms.scope():
            low.block(s.body)
        if s.else_body:
            branch.else_()
            with low.syms.scope():
                low.block(s.else_body)
