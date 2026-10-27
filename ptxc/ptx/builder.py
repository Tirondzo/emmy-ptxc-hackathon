"""PtxBuilder: the one emission API the lowering talks to.

Owns everything about *producing* instructions and nothing about what the IR means:
  names         virtual registers per bank, unique labels (ptxas allocates for real)
  emission      raw instructions, and the value-returning helpers rules compose
  control flow  `if_` and `loop` are the only places that spell setp / bra / labels
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from ..types import Ty
from .code import Operand, PtxModule, Reg, Sym, Value

PREFIX = {"pred": "p", "f32": "f", "f64": "fd", "b32": "r", "b64": "rd"}  # bank -> register name


class PtxBuilder:
    def __init__(self, module: PtxModule) -> None:
        self.module = module
        self._counts: dict[str, int] = {}  # bank -> registers handed out so far
        self._labels = 0

    # ---- names -------------------------------------------------------------------

    def reg(self, ty: Ty) -> Reg:
        """A fresh register `%{PREFIX[bank]}{n}` in `ty.bank`; the Reg keeps `ty`."""
        ...

    def label(self, hint: str) -> Sym: ...

    def place(self, label: Sym) -> None: ...

    # ---- emission ----------------------------------------------------------------

    def emit(
        self,
        opcode: str,
        *operands: Operand,
        guard: Reg | None = None,
        negate: bool = False,
    ) -> None: ...

    def op(self, opcode: str, ty: Ty, *srcs: Operand) -> Reg:
        """Emit `opcode dst, srcs...` into a fresh register of `ty` and return it."""
        ...

    def mov(self, dst: Reg, src: Value) -> None: ...

    def cvt(self, src: Value, to: Ty) -> Value:
        """Numeric conversion: `cvt.rn.f32.s32`, `cvt.s64.s32`, ...

        An Imm converts at compile time. A Reg of the same bank and width (s32 <-> u32)
        is retyped without an instruction.
        """
        ...

    def bitcast(self, src: Value, to: Ty) -> Value:
        """Reinterpret across banks (`mov.b32 %r, %f`): what shuffles and 64-bit splits need."""
        ...

    def setp(self, cmp: str, a: Value, b: Value) -> Reg:
        """`setp.{cmp}.{ty} %p, a, b` -> the predicate register."""
        ...

    # ---- structured control flow -------------------------------------------------

    @contextmanager
    def if_(self, pred: Reg, negate: bool = False) -> Generator[Branch]:
        """Run the block only where `pred` holds.

            @!%p bra SKIP;   <block>   SKIP:

        The yielded Branch's `else_()` turns the tail into `bra END; SKIP: <else> END:`.
        """
        ...

    @contextmanager
    def loop(self, var: Reg, end: Value, step: Value) -> Generator[None]:
        """`for (; var < end; var += step)`; `var` already holds the start value.

            HEAD: setp.ge %p, var, end; @%p bra END;   <body>   add var, var, step; bra HEAD; END:
        """
        ...

    def barrier(self, barrier_id: int = 0, count: int | None = None, warp: bool = False) -> None:
        """`bar.sync id[, count]`, or `bar.warp.sync 0xffffffff` for a warp-scope barrier."""
        ...

    # ---- completion --------------------------------------------------------------

    def finish(self) -> PtxModule:
        """Record the per-bank register counts the printer declares."""
        ...


class Branch:
    """Handle yielded by `PtxBuilder.if_`; `else_()` may be called once, inside the block."""

    def __init__(self, b: PtxBuilder, skip: Sym) -> None: ...

    def else_(self) -> None: ...
