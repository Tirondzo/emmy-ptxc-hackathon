"""Lowerer: the statement visitor, the main flow of the compiler.

Holds what every rule shares (the kernel, the builder, the symbol table) and dispatches
each statement to its rule. Attribute flow: inherited top-down (kernel facts, the current
scope), synthesized bottom-up (operands returned by `expr.value` and `apply`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .. import ir, ops
from ..errors import InvalidKernel, Unsupported
from ..ir import Body, Kernel, Node
from ..ptx import Addr, Imm, PtxBuilder, Reg, Sym, Value
from ..types import B32, INDEX, U32, U64, Ty
from .symbols import Array, Bound, Symbols, Variable

type Rule = Callable[["Lowerer", Node], None]

RULES: dict[str, Rule] = {}


def rule(*kinds: str) -> Callable[[Rule], Rule]:
    def register(fn: Rule) -> Rule:
        for k in kinds:
            RULES[k] = fn
        return fn

    return register


class Lowerer:
    def __init__(self, kernel: Kernel, b: PtxBuilder) -> None:
        self.kernel = kernel
        self.b = b
        self.syms = Symbols()
        self.smem_used = 0  # bytes carved from the pool so far, when shared memory is pooled

    def run(self) -> None:
        self.prologue()
        self.block(self.kernel.body)

    def prologue(self) -> None:
        """Parameters -> global Arrays, plus `lane` and `warp`, which the IR reads but never declares."""
        for buf in self.kernel.params:
            param = self.b.op("ld.param", U64, Addr(Sym(buf.param_name)))
            base = self.b.op("cvta.to.global", U64, param)
            self.syms.bind(buf.name, Array(buf.name, base, buf.ty, buf.shape, "global"))
        tid = self.b.op("mov", U32, Sym("%tid.x"))
        lane = self.b.op("and", B32, tid, Imm(31, B32))
        warp = self.b.op("shr", U32, tid, Imm(5, U32))
        self.syms.bind("lane", Bound(self.b.cvt(lane, INDEX)))
        self.syms.bind("warp", Bound(self.b.cvt(warp, INDEX)))

    def block(self, body: Body) -> None:
        for node in body:
            self.stmt(node)

    def stmt(self, node: Node) -> None:
        if (k := ir.kind(node)) not in RULES:
            raise Unsupported(f"statement {k}")
        RULES[k](self, node)

    # ---- names -------------------------------------------------------------------

    def read(self, name: str) -> Value:
        match self.syms.lookup(name):
            case Bound(value):
                return value
            case Variable(reg):
                return reg
            case _:
                raise InvalidKernel(f"{name!r} is a buffer, not a value")

    def declare(self, name: str, init: Value) -> Reg:
        """A fresh Variable holding `init`, bound in the innermost scope."""
        reg = self.b.reg(init.ty)
        self.b.mov(reg, init)
        self.syms.bind(name, Variable(reg))
        return reg

    def assign(self, name: str, value: Value) -> None:
        """Variable -> `mov` into its register. Unbound, or bound in the innermost scope ->
        rebind as Bound. Bound in an enclosing scope -> InvalidKernel.

        The IR is trusted to be SSA. A name redefined from inside a nested block (a loop
        body, a Cond branch, a combine instantiation) is either loop-carried state, which
        emmy marks with Accum or Init, or a value defined on one path only. Rebinding it
        would silently read a stale register, so the compiler refuses instead of analysing
        data flow.
        """
        match self.syms.get(name):
            case Variable(reg):
                if value != reg:  # TODO: the producing instruction could target `reg` directly; ignored for now
                    self.b.mov(reg, value)
            case None:
                self.syms.bind(name, Bound(value))
            case _ if self.syms.local(name):
                self.syms.bind(name, Bound(value))
            case _:
                raise InvalidKernel(f"{name!r} is redefined from a nested block")

    def seed(self, name: str, value: Value) -> None:
        """Loop-carried state at its start value: assign if already a Variable, else declare."""
        if isinstance(self.syms.get(name), Variable):
            self.assign(name, value)
        else:
            self.declare(name, value)

    def array(self, name: str) -> Array:
        match self.syms.lookup(name):
            case Array() as arr:
                return arr
            case _:
                raise InvalidKernel(f"{name!r} is not a buffer")

    def apply(self, op: Node, ty: Ty, args: Sequence[Value]) -> Value:
        return ops.apply(self.b, ir.op_name(op), ty, args)
