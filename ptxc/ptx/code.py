"""PTX held as data until it is printed.

An entry's header (register counts, shared declarations) is known only after its body is
built, so `PtxModule` keeps the sections apart and the printer lays them out in PTX order.
Declarations are appended whenever a rule meets them; nothing needs a pre-pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Ty


@dataclass(frozen=True)
class Reg:
    name: str  # without the leading "%"
    ty: Ty

    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass(frozen=True)
class Imm: # Const
    value: int | float
    ty: Ty

    def __str__(self) -> str:
        """Ints in decimal; floats as exact bit patterns (`0f3F800000`, `0d...`), never decimal."""
        ...


@dataclass(frozen=True)
class Sym:
    """A named PTX entity: a `.param`, a `.shared` array, a label, a special register."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Addr:
    """`[base+offset]`, the offset in bytes."""

    base: Reg | Sym
    offset: int = 0

    def __str__(self) -> str: ...


@dataclass(frozen=True)
class Vec:
    """`{a, b, c, d}`: the operand of vector loads and stores, and of 64-bit splits."""

    elems: tuple[Reg, ...]

    def __str__(self) -> str: ...


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
