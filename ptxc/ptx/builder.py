"""PtxBuilder: the one emission API the lowering talks to.

Owns everything about *producing* instructions and nothing about what the IR means:
  names         virtual registers per bank, unique labels (ptxas allocates for real)
  emission      raw instructions, and the value-returning helpers rules compose
  control flow  `if_` and `loop` are the only places that spell setp / bra / labels
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from ..types import B32, PRED, U32, Ty
from .code import PREFIX, Imm, Instr, Label, Operand, PtxModule, Reg, Sym, Value, opcode


class PtxBuilder:
    def __init__(self, module: PtxModule) -> None:
        self.module = module
        self._counts: dict[str, int] = {}  # bank -> registers handed out so far
        self._labels = 0

    # ---- names -------------------------------------------------------------------

    def reg(self, ty: Ty) -> Reg:
        n = self._counts[ty.bank] = self._counts.get(ty.bank, 0) + 1
        return Reg(f"{PREFIX[ty.bank]}{n}", ty)

    def label(self, hint: str) -> Sym:
        self._labels += 1
        return Sym(f"L_{hint}_{self._labels}")

    def place(self, label: Sym) -> None:
        self.module.body.append(Label(label.name))

    # ---- emission ----------------------------------------------------------------

    def emit(
        self,
        opcode: str,
        *operands: Operand,
        guard: Reg | None = None,
        negate: bool = False,
    ) -> None:
        self.module.body.append(Instr(opcode, operands, guard, negate))

    def op(self, opcode: str, ty: Ty, *srcs: Operand) -> Reg:
        """`opcode.{ty} dst, srcs...` into a fresh register: the type suffix is appended here."""
        dst = self.reg(ty)
        self.emit(f"{opcode}.{ty.ptx}", dst, *srcs)
        return dst

    def mov(self, dst: Reg, src: Value) -> None:
        self.emit(f"mov.{dst.ty.ptx}", dst, src)

    def materialize(self, v: Value) -> Reg:
        return v if isinstance(v, Reg) else self.op("mov", v.ty, v)

    def cvt(self, src: Value, to: Ty) -> Value:
        """Numeric conversion; same bank (s32 <-> u32) is a retype without an instruction."""
        if src.ty == to:
            return src
        if isinstance(src, Imm):
            return Imm(float(src.value) if to.is_float else int(src.value), to)
        if src.ty.bank == to.bank:
            return Reg(src.name, to)
        if src.ty.is_float and not to.is_float:
            rnd = "rzi"  # C truncation
        elif to.is_float and (not src.ty.is_float or src.ty.bits > to.bits):
            rnd = "rn"
        else:
            rnd = None
        dst = self.reg(to)
        self.emit(opcode("cvt", rnd, to.ptx, src.ty.ptx), dst, src)
        return dst

    def bitcast(self, src: Value, to: Ty) -> Value:
        """Reinterpret across banks (`mov.b32 %r, %f`): what shuffles need."""
        if src.ty.bank == to.bank:
            return self.cvt(src, to)
        dst = self.reg(to)
        self.emit(f"mov.b{to.bits}", dst, self.materialize(src))
        return dst

    def setp(self, cmp: str, a: Value, b: Value) -> Reg:
        p = self.reg(PRED)
        self.emit(f"setp.{cmp}.{a.ty.ptx}", p, a, b)
        return p

    # ---- structured control flow -------------------------------------------------

    @contextmanager
    def if_(self, pred: Reg, negate: bool = False) -> Generator[Branch]:
        branch = Branch(self, self.label("skip"))
        self.emit("bra", branch.skip, guard=pred, negate=not negate)
        yield branch
        self.place(branch.end)

    @contextmanager
    def loop(self, var: Reg, end: Value, step: Value) -> Generator[None]:
        """`for (; var < end; var += step)`; `var` already holds the start value."""
        head, done = self.label("loop"), self.label("done")
        self.place(head)
        self.emit("bra", done, guard=self.setp("ge", var, end))
        yield
        self.emit(f"add.{var.ty.ptx}", var, var, step)
        self.emit("bra", head)
        self.place(done)

    def barrier(self, barrier_id: int = 0, count: int | None = None, warp: bool = False) -> None:
        if warp:
            self.emit("bar.warp.sync", Imm(0xFFFFFFFF, B32))
        else:
            self.emit("bar.sync", Imm(barrier_id, U32), *([Imm(count, U32)] if count else []))

    # ---- completion --------------------------------------------------------------

    def finish(self) -> PtxModule:
        self.module.registers = dict(self._counts)
        return self.module


class Branch:
    """Yielded by `PtxBuilder.if_`: `@!p bra skip` was emitted; `else_()` once, inside the block."""

    def __init__(self, b: PtxBuilder, skip: Sym) -> None:
        self.b, self.skip, self.end = b, skip, skip

    def else_(self) -> None:
        self.end = self.b.label("endif")
        self.b.emit("bra", self.end)
        self.b.place(self.skip)
