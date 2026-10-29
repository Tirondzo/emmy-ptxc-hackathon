from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import Enum

from .errors import Unsupported


class Kind(Enum):
    PRED = "pred"
    FLOAT = "f"
    SIGNED = "s"
    UNSIGNED = "u"
    BITS = "b"


@dataclass(frozen=True)
class Ty:
    kind: Kind
    bits: int

    @property
    def ptx(self) -> str:
        return "pred" if self.kind is Kind.PRED else f"{self.kind.value}{self.bits}"

    @property
    def nbytes(self) -> int:
        """Size in memory; a predicate has none (`storable` gives the byte a bool is stored as)."""
        if self.kind is Kind.PRED:
            raise TypeError("a predicate has no size in memory")
        return self.bits // 8

    @property
    def is_float(self) -> bool:
        return self.kind is Kind.FLOAT

    @property
    def bank(self) -> str:
        """The `.reg` declaration this type lives in; s32, u32 and b32 share "b32". A 16-bit value is
        bits in a "b16" (PTX has no half register), an 8-bit one rides in a "b32"."""
        if self.kind is Kind.PRED:
            return "pred"
        if self.bits == 16:
            return "b16"
        return f"{'f' if self.is_float else 'b'}{max(self.bits, 32)}"

    @property
    def raw(self) -> str:
        """The spelling for `ld`, `st` and `mov`, which move bits and know no `.f16`."""
        return "b16" if self.bits == 16 else self.ptx

    def __call__(self, value: int | float) -> Imm:
        """A typed immediate: `INDEX(4)`, `B32(0x1f)`, `ty(0)`."""
        return Imm(value, self)


@dataclass(frozen=True)
class Imm:  # Const
    value: int | float
    ty: Ty

    def __post_init__(self) -> None:  # the value is of the type: folding can then go by the Python type
        object.__setattr__(self, "value", float(self.value) if self.ty.is_float else int(self.value))

    def __str__(self) -> str:
        if self.ty.kind is Kind.BITS:
            return hex(self.value)
        if not self.ty.is_float:
            return str(self.value)
        if self.ty.bits == 16:  # PTX has no half constant: the bits, for the `mov.b16` that materializes it
            return "0x%04X" % float_bits(self.value, 16)
        if self.ty.bits == 32:  # floats are written as exact bit patterns, never decimal
            return "0f%08X" % float_bits(self.value, 32)
        return "0d%016X" % float_bits(self.value, 64)


FORMATS = {16: ("e", "H"), 32: ("f", "I"), 64: ("d", "Q")}  # struct codes: the float of that width, its bits


def float_bits(value: float, bits: int) -> int:
    """`value` rounded to the float of `bits` bits, as an integer; a magnitude beyond its range is inf,
    as it would be after any operation of that type."""
    pack, unpack = FORMATS[bits]
    try:
        return struct.unpack(">" + unpack, struct.pack(">" + pack, value))[0]
    except OverflowError:
        return float_bits(math.copysign(math.inf, value), bits)


PRED = Ty(Kind.PRED, 1)
F16 = Ty(Kind.FLOAT, 16)
F32 = Ty(Kind.FLOAT, 32)
F64 = Ty(Kind.FLOAT, 64)
S32 = Ty(Kind.SIGNED, 32)
S64 = Ty(Kind.SIGNED, 64)
U16 = Ty(Kind.UNSIGNED, 16)
U32 = Ty(Kind.UNSIGNED, 32)
U64 = Ty(Kind.UNSIGNED, 64)
B32 = Ty(Kind.BITS, 32)
INDEX = S32
BOOL = Ty(Kind.UNSIGNED, 8)  # a C bool in memory: one byte, 0 or 1; in registers a bool is PRED

NAMES = {  # emmy's dtype names and the C spellings Smem uses
    "f16": F16,
    "half": F16,
    "__half": F16,
    "f32": F32,
    "float": F32,
    "f64": F64,
    "double": F64,
    "i32": S32,
    "s32": S32,
    "int": S32,
    "i64": S64,
    "s64": S64,
    "long long": S64,
    "u32": U32,
    "unsigned int": U32,
    "u64": U64,
    "unsigned long long": U64,
    "bool": PRED,  # a predicate register; `storable` makes it a byte in memory
}


def ty_of(name: str) -> Ty:
    if name not in NAMES:
        raise Unsupported(f"dtype {name!r}")
    return NAMES[name]


def storable(ty: Ty) -> Ty:
    """The element type of a buffer or shared array: a bool is stored as C stores it, one byte of 0 or 1."""
    return BOOL if ty.kind is Kind.PRED else ty
