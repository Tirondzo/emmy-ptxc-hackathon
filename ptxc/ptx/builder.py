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

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from ..types import B32, PRED, U32, Imm, Ty
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
    return v if isinstance(v, Reg) else op("mov", v.ty, v)


def cvt(src: Value, to: Ty) -> Value:
    """Numeric conversion; same bank (s32 <-> u32) is a retype without an instruction."""
    if src.ty == to:
        return src
    if isinstance(src, Imm):
        return to(float(src.value) if to.is_float else int(src.value))
    if src.ty.bank == to.bank:
        return Reg(src.name, to)
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


def setp(cmp: str, a: Value, b: Value) -> Reg:
    p = reg(PRED)
    emit(f"setp.{cmp}.{a.ty.ptx}", p, a, b)
    return p


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
        emit("bar.warp.sync", B32(0xFFFFFFFF))
    else:
        emit("bar.sync", U32(barrier_id), *([U32(count)] if count else []))


class Branch:
    """Yielded by `if_`: `@!p bra skip` was emitted; `else_()` once, inside the block."""

    def __init__(self, skip: Sym) -> None:
        self.skip = self.end = skip

    def else_(self) -> None:
        self.end = label("endif")
        emit("bra", self.end)
        place(self.skip)
