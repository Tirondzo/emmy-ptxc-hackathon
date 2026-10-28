"""Loop and StridedLoop, and ptxc's one performance decision: unrolling.

A loop is lowered in one of three shapes, all the same computation in the same order (memory is
touched as the schedule laid it out, a barrier in the body still meets every thread):

  straight   no loop: `trip` copies of the body, the axis a constant
  rolled     the plain `for (var = start; var < end; var += step)`
  partial    `factor` copies per iteration while a whole group is in range, then what is left

The win is in `expr.affine`: a constant axis folds into `[base+imm]` displacements, which ptxas
does not do for a rolled loop. The decision is made where the loop is met, but `sizes` predicts
what the nested loops become, so an outer loop knows the size of its body.
"""

from __future__ import annotations

import math

from .. import ir
from ..errors import Unsupported
from ..ir import Body, Node
from ..options import Options
from ..ptx import Imm, Reg, Value, cvt, for_, mov, reg
from ..types import INDEX, ty_of
from . import expr
from .lowerer import bind, declare, lower, lowering, options, rule, scope, trace
from .symbols import Bound


@rule("Loop")
def loop(s: Node) -> None:
    counted_loop(s.axis.name, INDEX(0), INDEX(ir.extent(s.axis)), INDEX(1), s)


@rule("StridedLoop")
def strided_loop(s: Node) -> None:
    end = INDEX(ir.extent(s.axis)) if s.end is None else expr.value(s.end)
    counted_loop(s.axis.name, expr.value(s.start), end, expr.value(s.step), s)


def counted_loop(axis: str, start: Value, end: Value, step: Value, s: Node) -> None:
    """Seed first, in the enclosing scope; then the loop in the shape `unroll_factor` chose."""
    if s.seed:
        seed_accumulators(s.body)
    trip, (own, expanded) = trip_count(s), sizes(s.body)
    factor = unroll_factor(s.unroll, trip, own, expanded, options()) if isinstance(step, Imm) else 1  # offsets need a constant step
    straight_ = trip is not None and factor >= trip
    trace("unroll", f"{axis}: trip={trip} hinted={s.unroll} body={own}->{expanded} -> {'straight' if straight_ else f'x{factor}'}")
    start = cvt(start, INDEX)
    if straight_:
        straight(axis, start, step.value, trip, s.body)
    elif factor == 1:
        rolled(axis, start, end, step, s.body)
    else:
        partial(axis, start, end, step.value, factor, trip, s.body)


# ---- the three shapes -------------------------------------------------------------------


def straight(axis: str, start: Value, step: int, trip: int, body: Body) -> None:
    with scope():
        for k in range(trip):
            copy(axis, at(start, k * step), body)


def rolled(axis: str, start: Value, end: Value, step: Value, body: Body) -> None:
    var = counter(start)
    with scope(), for_(var, cvt(end, INDEX), cvt(step, INDEX)):
        copy(axis, Bound(var), body)


def partial(axis: str, start: Value, end: Value, step: int, factor: int, trip: int | None, body: Body) -> None:
    """The main loop runs while a whole group of `factor` iterations fits before `end`. What is
    left is `trip % factor` copies when the trip count is known, else the plain loop."""
    var = counter(start)
    last_group = cvt(expr.arith("-", end, INDEX((factor - 1) * step)), INDEX)
    with for_(var, last_group, INDEX(factor * step)), scope():
        for j in range(factor):
            copy(axis, Bound(var, j * step), body)
    if trip is not None:  # var is now the first iteration left
        with scope():
            for k in range(trip % factor):
                copy(axis, Bound(var, k * step), body)
    else:
        with scope(), for_(var, cvt(end, INDEX), INDEX(step)):
            copy(axis, Bound(var), body)


def counter(start: Value) -> Reg:
    var = reg(INDEX)
    mov(var, start)
    return var


def at(start: Value, offset: int) -> Bound:
    """`start + offset`: a constant, or the start register with an offset `expr.affine` folds."""
    if isinstance(start, Imm):
        return Bound(INDEX(start.value + offset))
    return Bound(start, offset)


def copy(axis: str, where: Bound, body: Body) -> None:
    """Copies share a scope: straight-line code, where rebinding a name is the SSA rule anyway."""
    bind(axis, where)
    lower(body)


def seed_accumulators(body: Body) -> None:
    """Each Accum of the loop's own body starts at its identity, as a Variable of the enclosing
    scope; one in a nested block belongs to whatever that name denotes there."""
    accums = {s.name: s for s in body if ir.kind(s) == "Accum"}
    for name, s in accums.items():
        if (ty := ir.ty(s.dtype)) is None:
            raise Unsupported(f"untyped accumulator {name!r}")
        declare(name, ty(ir.op_identity(s.op)))


# ---- the decision -----------------------------------------------------------------------


def unroll_factor(hinted: bool, trip: int | None, own: int, expanded: int, o: Options) -> int:
    """Copies of the body per iteration; a factor >= trip means no loop. `own` is the body as
    written, `expanded` what it becomes with its nested loops unrolled; the quota bounds the latter."""
    if o.unroll == "none" or expanded == 0:
        return 1
    if o.unroll == "force" and trip is not None:
        return trip
    if hinted and trip is not None:
        if trip * expanded <= o.unroll_quota:
            return trip
        return max((d for d in range(1, trip) if trip % d == 0 and d * expanded <= o.unroll_quota), default=1)
    if trip is not None and trip * expanded <= min(o.unroll_auto_full, o.unroll_quota):
        return trip
    if own > o.unroll_auto_body:
        return 1
    factor = max(1, o.unroll_auto)
    while factor > 1 and factor * expanded > o.unroll_quota:
        factor //= 2
    return min(factor, trip) if trip is not None else factor


def sizes(body: Body) -> tuple[int, int]:
    """(own, expanded): the statements of `body` with nested loop bodies counted once, and what it
    becomes once each nested loop is unrolled as `unroll_factor` will shape it."""
    own = expanded = 0
    for s in body:
        own += 1
        expanded += 1
        if ir.kind(s) in ("Loop", "StridedLoop"):
            (inner_own, inner), trip = sizes(s.body), trip_count(s)
            factor = unroll_factor(s.unroll, trip, inner_own, inner, options())
            own += inner_own
            expanded += inner * (trip if trip is not None and factor >= trip else factor)
        else:
            for field in ("body", "else_body"):
                if isinstance(nested := getattr(s, field, None), (list, tuple)):
                    nested_own, nested_expanded = sizes(nested)
                    own += nested_own
                    expanded += nested_expanded
    return own, expanded


def trip_count(s: Node) -> int | None:
    """Iterations when the IR fixes them: a Loop's extent; a StridedLoop's constant bounds, or the
    cooperative stride `for i in tid..N:T` with `tid` a Tile axis below T and T dividing N."""
    if ir.kind(s) == "Loop":
        return ir.extent(s.axis)
    step = static(s.step)
    end = ir.extent(s.axis) if s.end is None else static(s.end)
    if step is None or end is None or step <= 0:
        return None
    if (start := static(s.start)) is not None:
        return max(0, math.ceil((end - start) / step))
    if ir.kind(s.start) == "Var" and end % step == 0:
        extent = lowering().axes.get(s.start.name)
        if extent is not None and extent <= step:
            return end // step
    return None


def static(e: Node) -> int | None:
    """An integer expression's value when the IR fixes it: literals, names bound to a constant,
    arithmetic on those."""
    match ir.kind(e):
        case "Literal" if not ty_of(e.dtype).is_float:
            return int(e.value)
        case "Var":
            match lowering().syms.get(e.name):
                case Bound(Imm() as v, offset) if not v.ty.is_float:
                    return int(v.value) + offset
            return None
        case "BinaryExpr" if e.op in expr.ARITH:
            left, right = static(e.left), static(e.right)
            if left is None or right is None or (e.op in ("/", "//", "%") and right == 0):
                return None
            return int(expr.fold(e.op, left, right))
    return None
