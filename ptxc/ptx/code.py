"""PTX held as data until it is printed.

An entry's header (register counts, shared declarations) is known only after its body is
built, so `PtxModule` keeps the sections apart and the printer lays them out in PTX order.
Declarations are appended whenever a rule meets them; nothing needs a pre-pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Imm, Ty

PREFIX = {"pred": "p", "f32": "f", "f64": "fd", "b32": "r", "b64": "rd"}  # bank -> register name


@dataclass(frozen=True)
class Reg:
    name: str  # without the leading "%"
    ty: Ty

    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass(frozen=True)
class Sym:
    """A named PTX entity: a `.param`, a `.shared` array, a label, a special register."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Addr:
    base: Reg | Sym
    offset: int = 0  # bytes

    def __str__(self) -> str:
        return f"[{self.base}{self.offset:+d}]" if self.offset else f"[{self.base}]"


@dataclass(frozen=True)
class Vec:
    """`{a, b, c, d}`: the operand of vector loads and stores, and of 64-bit splits."""

    elems: tuple[Reg, ...]

    def __str__(self) -> str:
        return "{" + ", ".join(map(str, self.elems)) + "}"


type Value = Reg | Imm
type Operand = Reg | Imm | Sym | Addr | Vec


@dataclass(frozen=True)
class Instr:
    opcode: str
    operands: tuple[Operand, ...] = ()
    guard: Reg | None = None  # `@%p`, or `@!%p` when negated
    negate: bool = False


@dataclass(frozen=True)
class Label:
    name: str


type Item = Instr | Label


@dataclass(frozen=True)
class Param:
    name: str
    ty: Ty


@dataclass
class Shared:
    name: str
    nbytes: int
    align: int  # the dynamic pool's grows: declared once, carved by every Smem
    extern: bool = False  # the dynamic pool, sized at launch: `.extern .shared .b8 name[]`


@dataclass
class PtxModule:
    entry: str
    params: tuple[Param, ...]
    target: str  # `.target`: the machine, as ptxas spells it
    version: str  # `.version`: the PTX ISA the body speaks
    max_threads: int | None = None  # `.maxntid`: the launch's block, so ptxas bounds registers to it
    shared: list[Shared] = field(default_factory=list)
    body: list[Item] = field(default_factory=list)
    registers: dict[str, int] = field(default_factory=dict)  # bank -> count, set when the builder exits


def dotted(*parts: str | None) -> str:
    """An opcode with an optional modifier: dotted("ld", space, None, "f32") == "ld.global.f32".
    Fixed shapes are spelled as f-strings instead."""
    return ".".join(p for p in parts if p)
