"""The builder: the one emission API the lowering talks to.

`with PtxBuilder(module):` makes `module` the target of every function below, and the
functions are the vocabulary rules compose. They own everything about *producing*
instructions and nothing about what the IR means:
  names         virtual registers per bank, unique labels (ptxas allocates for real)
  emission      raw instructions, and the value-returning helpers rules compose
  control flow  `if_` and `for_` are the only places that spell setp / bra / labels
Emission is eager: nothing reads the current builder after its `with` has closed.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from ..errors import Unsupported
from ..types import B32, PRED, U32, Imm, Kind, Ty
from .code import PREFIX, Instr, Label, Operand, PtxModule, Reg, Sym, Value, dotted

_current: ContextVar[PtxBuilder] = ContextVar("builder")


class PtxBuilder:
    def __init__(self, module: PtxModule) -> None:
        self.module = module
        self.counts: dict[str, int] = {}  # bank -> registers handed out so far
        self.labels = 0

    def __enter__(self) -> PtxBuilder:
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        self.module.registers = dict(self.counts)
        _current.reset(self._token)


def builder() -> PtxBuilder:
    return _current.get()


# ---- names -------------------------------------------------------------------


def reg(ty: Ty) -> Reg:
    b = builder()
    n = b.counts[ty.bank] = b.counts.get(ty.bank, 0) + 1
    return Reg(f"{PREFIX[ty.bank]}{n}", ty)


def label(hint: str) -> Sym:
    b = builder()
    b.labels += 1
    return Sym(f"L_{hint}_{b.labels}")


def place(sym: Sym) -> None:
    builder().module.body.append(Label(sym.name))


# ---- emission ----------------------------------------------------------------


def emit(opcode: str, *operands: Operand, guard: Reg | None = None, negate: bool = False) -> None:
    builder().module.body.append(Instr(opcode, operands, guard, negate))


def op(opcode: str, ty: Ty, *srcs: Operand) -> Reg:
    """`opcode.{ty} dst, srcs...` into a fresh register: the type suffix is appended here."""
    dst = reg(ty)
    emit(f"{opcode}.{ty.ptx}", dst, *srcs)
    return dst


def mov(dst: Reg, src: Value) -> None:
    emit(f"mov.{dst.ty.ptx}", dst, src)


def materialize(v: Value) -> Reg:
    if isinstance(v, Reg):
        return v
    if v.ty.bits == 8:  # no 8-bit registers: the byte rides in a .b32, as it does after a load
        return Reg(op("mov", U32, U32(int(v.value))).name, v.ty)
    return op("mov", v.ty, v)


def cvt(src: Value, to: Ty) -> Value:
    """Numeric conversion; same bank (s32 <-> u32) is a retype without an instruction.
    To and from a predicate: C's truth rules, nonzero (NaN included) is true and true is 1."""
    if src.ty == to:
        return src
    if isinstance(src, Imm):
        if to.kind is Kind.PRED or to.bits == 8:
            return to(int(src.value != 0))
        return to(src.value)
    if to.bits == 8:  # a bool in memory: C's conversion to bool, then the byte
        return Reg(op("selp", U32, U32(1), U32(0), materialize(cvt(src, PRED))).name, to)
    if src.ty.bits == 8:  # zero-extended by the load: widen for free, then convert
        return cvt(Reg(src.name, U32), to)
    if to.kind is Kind.PRED:
        return setp("neu" if src.ty.is_float else "ne", src, src.ty(0))
    if src.ty.kind is Kind.PRED:
        return op("selp", to, to(1), to(0), src)
    if src.ty.bank == to.bank:
        return Reg(src.name, to)
    if Kind.BITS in (src.ty.kind, to.kind):
        raise Unsupported(f"cvt.{to.ptx}.{src.ty.ptx}: raw bits have no numeric value")
    if src.ty.is_float and not to.is_float:
        rnd = "rzi"  # C truncation
    elif to.is_float and (not src.ty.is_float or src.ty.bits > to.bits):
        rnd = "rn"
    else:
        rnd = None
    dst = reg(to)
    emit(dotted("cvt", rnd, to.ptx, src.ty.ptx), dst, src)
    return dst


def bitcast(src: Value, to: Ty) -> Value:
    """Reinterpret across banks (`mov.b32 %r, %f`): what shuffles need."""
    if src.ty.bank == to.bank:
        return cvt(src, to)
    dst = reg(to)
    emit(f"mov.b{to.bits}", dst, materialize(src))
    return dst


COMPARE = {"eq": operator.eq, "ne": operator.ne, "neu": operator.ne, "lt": operator.lt, "le": operator.le,
           "gt": operator.gt, "ge": operator.ge, "nan": lambda a, b: math.isnan(a) or math.isnan(b)}


def setp(cmp: str, a: Value, b: Value) -> Value:
    """`a cmp b` as a predicate; two immediates fold."""
    if isinstance(a, Imm) and isinstance(b, Imm):
        return PRED(int(COMPARE[cmp](a.value, b.value)))
    if a.ty.kind is Kind.PRED or (a.ty.kind is Kind.BITS and cmp not in ("eq", "ne")):
        raise Unsupported(f"setp.{cmp}.{a.ty.ptx}")
    p = reg(PRED)
    emit(f"setp.{cmp}.{a.ty.ptx}", p, a, b)
    return p


def truth(v: Value) -> Reg:
    """`v` as a predicate register: a branch guard, a `selp` selector."""
    return materialize(cvt(v, PRED))


def unify(a: Value, b: Value) -> tuple[Value, Value]:
    """A common type for a binary op: float wins, then the wider."""
    ty = max(a.ty, b.ty, key=lambda t: (t.is_float, t.bits))
    return cvt(a, ty), cvt(b, ty)


# ---- structured control flow -------------------------------------------------


@contextmanager
def if_(pred: Reg, negate: bool = False) -> Generator[Branch]:
    branch = Branch(label("skip"))
    emit("bra", branch.skip, guard=pred, negate=not negate)
    yield branch
    place(branch.end)


@contextmanager
def for_(var: Reg, end: Value, step: Value) -> Generator[None]:
    """`for (; var < end; var += step)`; `var` already holds the start value."""
    head, done = label("loop"), label("done")
    place(head)
    emit("bra", done, guard=setp("ge", var, end))
    yield
    emit(f"add.{var.ty.ptx}", var, var, step)
    emit("bra", head)
    place(done)


def barrier(barrier_id: int = 0, count: int | None = None, warp: bool = False) -> None:
    if warp:
        emit("bar.warp.sync", activemask())
    else:
        emit("bar.sync", U32(barrier_id), *([U32(count)] if count else []))


def activemask() -> Reg:
    """The lanes of this warp that are executing: the member mask of every warp-level sync."""
    return op("activemask", B32)


class Branch:
    """Yielded by `if_`: `@!p bra skip` was emitted; `else_()` once, inside the block."""

    def __init__(self, skip: Sym) -> None:
        self.skip = self.end = skip

    def else_(self) -> None:
        self.end = label("endif")
        emit("bra", self.end)
        place(self.skip)
