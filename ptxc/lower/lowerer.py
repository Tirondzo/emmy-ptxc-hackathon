"""Lowerer: the statement visitor, the main flow of the compiler.

Holds what every rule shares (the kernel, the builder, the symbol table) and dispatches
each statement to its rule. Attribute flow: inherited top-down (kernel facts, the current
scope), synthesized bottom-up (operands returned by `expr.value` and `apply`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .. import ir, ops
from ..errors import Unsupported
from ..ir import Body, Kernel, Node
from ..ptx import PtxBuilder, Reg, Value
from ..types import Ty
from .symbols import Array, Symbols

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
        """Parameters -> global Arrays (`ld.param.u64`, `cvta.to.global.u64`), and the names
        the IR reads but never declares: `lane` = tid.x & 31 and `warp` = tid.x >> 5.
        Bound unconditionally; ptxas drops them when unused."""
        ...

    def block(self, body: Body) -> None:
        for node in body:
            self.stmt(node)

    def stmt(self, node: Node) -> None:
        if (k := ir.kind(node)) not in RULES:
            raise Unsupported(f"statement {k}")
        RULES[k](self, node)

    # ---- names -------------------------------------------------------------------

    def read(self, name: str) -> Value:
        """A Bound's value or a Variable's register. An Array is not a value -> InvalidKernel."""
        ...

    def declare(self, name: str, init: Value) -> Reg:
        """A fresh Variable holding `init`, bound in the innermost scope."""
        ...

    def assign(self, name: str, value: Value) -> None:
        """Variable -> `mov` into its register. Unbound, or bound in the innermost scope ->
        rebind as Bound. Bound in an enclosing scope -> InvalidKernel.

        The IR is trusted to be SSA. A name redefined from inside a nested block (a loop
        body, a Cond branch, a combine instantiation) is either loop-carried state, which
        emmy marks with Accum or Init, or a value defined on one path only. Rebinding it
        would silently read a stale register, so the compiler refuses instead of analysing
        data flow.
        """
        ...

    def seed(self, name: str, value: Value) -> None:
        """Loop-carried state at its start value: assign if already a Variable, else declare."""
        ...

    def array(self, name: str) -> Array:
        """The Array bound to `name`; anything else -> InvalidKernel."""
        ...

    def apply(self, op: Node, ty: Ty, args: Sequence[Value]) -> Value:
        return ops.apply(self.b, ir.op_name(op), ty, args)
