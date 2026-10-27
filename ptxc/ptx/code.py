"""PTX held as data until it is printed.

An entry's header (register counts, shared declarations) is known only after its body is
built, so `PtxModule` keeps the sections apart and the printer lays them out in PTX order.
Declarations are appended whenever a rule meets them; nothing needs a pre-pass.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ..types import Kind, Ty

PREFIX = {"pred": "p", "f32": "f", "f64": "fd", "b32": "r", "b64": "rd"}  # bank -> register name


@dataclass(frozen=True)
class Reg:
    name: str  # without the leading "%"
    ty: Ty

    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass(frozen=True)
class Imm:  # Const
    value: int | float
    ty: Ty

    def __str__(self) -> str:
        if self.ty.kind is Kind.BITS:
            return hex(int(self.value))
        if not self.ty.is_float:
            return str(int(self.value))
        if self.ty.bits == 32:  # floats are written as exact bit patterns, never decimal
            return "0f%08X" % struct.unpack(">I", struct.pack(">f", self.value))[0]
        return "0d%016X" % struct.unpack(">Q", struct.pack(">d", self.value))[0]


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


@dataclass(frozen=True)
class Shared:
    name: str
    nbytes: int
    align: int
    extern: bool = False  # the dynamic pool, sized at launch: `.extern .shared .b8 name[]`


@dataclass
class PtxModule:
    entry: str
    params: tuple[Param, ...]
    target: str = "sm_89"
    version: str = "8.0"  # sm_89 needs PTX ISA 7.8 or later
    shared: list[Shared] = field(default_factory=list)
    body: list[Item] = field(default_factory=list)
    registers: dict[str, int] = field(default_factory=dict)  # bank -> count, set by finish()


def opcode(*parts: str | None) -> str:
    """Join opcode parts, dropping absent modifiers: opcode("add", None, "f32") == "add.f32"."""
    return ".".join(p for p in parts if p)
