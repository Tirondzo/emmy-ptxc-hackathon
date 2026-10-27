"""The lowering: the statement visitor, the main flow of the compiler.

`with Lowering(kernel):` makes the kernel's names and launch the context of every function
below, the way `with PtxBuilder(module):` makes the module the target of every emission.
Rules read the IR top-down and get operands back bottom-up (`expr.value`, `ops.apply`).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .. import ir
from ..errors import InvalidKernel, Unsupported
from ..ir import Body, Kernel, Node
from ..ptx import Addr, Reg, Sym, Value, cvt, mov, op, reg
from ..types import B32, INDEX, U32, U64
from .symbols import Array, Bound, Symbol, Symbols, Variable

type Rule = Callable[[Node], None]

RULES: dict[str, Rule] = {}
_current: ContextVar[Lowering] = ContextVar("lowering")


def rule(*kinds: str) -> Callable[[Rule], Rule]:
    def register(fn: Rule) -> Rule:
        for k in kinds:
            RULES[k] = fn
        return fn

    return register


@dataclass
class Lowering:
    kernel: Kernel
    syms: Symbols = field(default_factory=Symbols)
    smem_used: int = 0  # bytes carved from the pool so far, when shared memory is pooled

    def __enter__(self) -> Lowering:
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        _current.reset(self._token)


def lowering() -> Lowering:
    return _current.get()


def prologue() -> None:
    """Parameters -> global Arrays, plus `lane` and `warp`, which the IR reads but never declares."""
    for buf in lowering().kernel.params:
        param = op("ld.param", U64, Addr(Sym(buf.param_name)))
        base = op("cvta.to.global", U64, param)
        bind(buf.name, Array(buf.name, base, buf.ty, buf.shape, "global"))
    tid = op("mov", U32, Sym("%tid.x"))
    lane = op("and", B32, tid, B32(31))
    warp = op("shr", U32, tid, U32(5))
    bind("lane", Bound(cvt(lane, INDEX)))
    bind("warp", Bound(cvt(warp, INDEX)))


def lower(body: Body) -> None:
    for node in body:
        if (k := ir.kind(node)) not in RULES:
            raise Unsupported(f"statement {k}")
        RULES[k](node)


# ---- names -------------------------------------------------------------------


def scope() -> AbstractContextManager[None]:
    return lowering().syms.scope()


def bind(name: str, symbol: Symbol) -> None:
    lowering().syms.bind(name, symbol)


def read(name: str) -> Value:
    match lowering().syms.lookup(name):
        case Bound(value):
            return value
        case Variable(reg=r):
            return r
        case _:
            raise InvalidKernel(f"{name!r} is a buffer, not a value")


def declare(name: str, init: Value) -> Reg:
    """A fresh Variable holding `init`, bound in the innermost scope."""
    r = reg(init.ty)
    mov(r, init)
    bind(name, Variable(r))
    return r


def assign(name: str, value: Value) -> None:
    """Variable -> `mov` into its register. Unbound, or bound in the innermost scope ->
    rebind as Bound. Bound in an enclosing scope -> InvalidKernel.

    The IR is trusted to be SSA. A name redefined from inside a nested block (a loop
    body, a Cond branch, a combine instantiation) is either loop-carried state, which
    emmy marks with Accum or Init, or a value defined on one path only. Rebinding it
    would silently read a stale register, so the compiler refuses instead of analysing
    data flow.
    """
    syms = lowering().syms
    match syms.get(name):
        case Variable(reg=r):
            if value != r:  # TODO: the producing instruction could target `r` directly; ignored for now
                mov(r, value)
        case None:
            syms.bind(name, Bound(value))
        case _ if syms.local(name):
            syms.bind(name, Bound(value))
        case _:
            raise InvalidKernel(f"{name!r} is redefined from a nested block")


def seed(name: str, value: Value) -> None:
    """Loop-carried state at its start value: assign if already a Variable, else declare."""
    if isinstance(lowering().syms.get(name), Variable):
        assign(name, value)
    else:
        declare(name, value)


def array(name: str) -> Array:
    match lowering().syms.lookup(name):
        case Array() as arr:
            return arr
        case _:
            raise InvalidKernel(f"{name!r} is not a buffer")
