"""The lowering: the statement visitor, the main flow of the compiler.

`with Lowering(kernel, target):` makes the kernel's names and launch, and the machine, the context
of every function below, the way `with PtxBuilder(module):` makes the module the target of every
emission.
Rules read the IR top-down and get operands back bottom-up (`expr.value`, `ops.apply`).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .. import ir
from ..config import Target
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
    target: Target
    syms: Symbols = field(default_factory=Symbols)
    smem_used: int = 0  # bytes carved from the pool so far, when shared memory is pooled
    tid: Reg | None = None  # %tid.x once `tid()` has read it

    def __enter__(self) -> Lowering:
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        _current.reset(self._token)


def lowering() -> Lowering:
    return _current.get()


def target() -> Target:
    return lowering().target


def prologue() -> None:
    """Parameters -> global Arrays, plus `lane` and `warp`, which the IR reads but never declares."""
    for buf in lowering().kernel.params:
        param = op("ld.param", U64, Addr(Sym(buf.param_name)))
        base = op("cvta.to.global", U64, param)
        bind(buf.name, Array(buf.name, base, buf.ty, buf.shape, "global"))
    lane = op("and", B32, tid(), B32(target().lane_mask))
    warp = op("shr", U32, tid(), U32(target().warp_shift))
    bind("lane", Bound(cvt(lane, INDEX)))
    bind("warp", Bound(cvt(warp, INDEX)))


def tid() -> Reg:
    """`%tid.x`, read once per kernel."""
    cur = lowering()
    if cur.tid is None:
        cur.tid = op("mov", U32, Sym("%tid.x"))
    return cur.tid


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
    """Variable -> `mov` into its register. Anything else -> Bound in the innermost scope,
    shadowing an outer binding until the block closes, as a C block would. A Bound never
    aliases a Variable's register: that value is copied first, so later mutation of the
    Variable cannot reach it. The IR's Accum / Init marks say what is loop-carried; nothing
    here analyses data flow.
    """
    syms = lowering().syms
    match syms.get(name):
        case Variable(reg=r):
            if not same(value, r):  # TODO: the producing instruction could target `r` directly
                mov(r, value)
        case _:
            syms.bind(name, Bound(op("mov", value.ty, value) if syms.is_variable(value) else value))


def same(a: Value, b: Reg) -> bool:
    """One register, whatever type `cvt` retyped it to."""
    return isinstance(a, Reg) and a.name == b.name


def mutable(name: str) -> None:
    """A collective's state, which its combine mutates in place: a Variable already; a Bound of
    this scope becomes one; a Bound of an enclosing scope would turn into a shadow that dies
    with the block, so it is refused rather than silently lost."""
    syms = lowering().syms
    match syms.get(name):
        case Variable():
            return
        case Bound(value) if syms.local(name):
            declare(name, value)
        case Bound():
            raise Unsupported(f"collective state {name!r} is defined outside its block")
        case _:
            raise InvalidKernel(f"{name!r} is not a value")


def array(name: str) -> Array:
    match lowering().syms.lookup(name):
        case Array() as arr:
            return arr
        case _:
            raise InvalidKernel(f"{name!r} is not a buffer")
