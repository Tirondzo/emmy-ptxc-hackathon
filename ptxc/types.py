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
        """The `.reg` declaration this type lives in; s32, u32 and b32 share "b32"."""
        if self.kind is Kind.PRED:
            return "pred"
        return f"{'f' if self.is_float else 'b'}{max(self.bits, 32)}"  # 8-bit values ride in 32-bit registers

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
        if self.ty.bits == 32:  # floats are written as exact bit patterns, never decimal
            return "0f%08X" % f32_bits(self.value)
        return "0d%016X" % struct.unpack(">Q", struct.pack(">d", self.value))[0]


def f32_bits(value: float) -> int:
    """The nearest f32; a magnitude beyond f32 rounds to inf, as every f32 operation does."""
    try:
        return struct.unpack(">I", struct.pack(">f", value))[0]
    except OverflowError:
        return f32_bits(math.copysign(math.inf, value))


PRED = Ty(Kind.PRED, 1)
F32 = Ty(Kind.FLOAT, 32)
F64 = Ty(Kind.FLOAT, 64)
S32 = Ty(Kind.SIGNED, 32)
S64 = Ty(Kind.SIGNED, 64)
U32 = Ty(Kind.UNSIGNED, 32)
U64 = Ty(Kind.UNSIGNED, 64)
B32 = Ty(Kind.BITS, 32)
INDEX = S32
BOOL = Ty(Kind.UNSIGNED, 8)  # a C bool in memory: one byte, 0 or 1; in registers a bool is PRED

NAMES = {  # emmy's dtype names and the C spellings Smem uses; 16-bit types absent on purpose
    "f32": F32,
    "float": F32,
    "f64": F64,
    "double": F64,
    "i32": S32,
    "s32": S32,
    "int": S32,
    "i64": S64,
    "s64": S64,
    "u32": U32,
    "u64": U64,
    "bool": PRED,  # a predicate register; `storable` makes it a byte in memory
}


def ty_of(name: str) -> Ty:
    if name not in NAMES:
        raise Unsupported(f"dtype {name!r}")
    return NAMES[name]


def storable(ty: Ty) -> Ty:
    """The element type of a buffer or shared array: a bool is stored as C stores it, one byte of 0 or 1."""
    return BOOL if ty.kind is Kind.PRED else ty
