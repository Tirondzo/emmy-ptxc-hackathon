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
        return self.bits // 8

    @property
    def is_float(self) -> bool:
        return self.kind is Kind.FLOAT

    @property
    def bank(self) -> str:
        """The `.reg` declaration this type lives in; s32, u32 and b32 share "b32"."""
        if self.kind is Kind.PRED:
            return "pred"
        return f"{'f' if self.is_float else 'b'}{self.bits}"


PRED = Ty(Kind.PRED, 1)
F32 = Ty(Kind.FLOAT, 32)
F64 = Ty(Kind.FLOAT, 64)
S32 = Ty(Kind.SIGNED, 32)
S64 = Ty(Kind.SIGNED, 64)
U32 = Ty(Kind.UNSIGNED, 32)
U64 = Ty(Kind.UNSIGNED, 64)
B32 = Ty(Kind.BITS, 32)
B64 = Ty(Kind.BITS, 64)
INDEX = S32

NAMES = {  # emmy's dtype names and the C spellings Smem uses; 16- and 8-bit types absent on purpose
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
}


def ty_of(name: str) -> Ty:
    if name not in NAMES:
        raise Unsupported(f"dtype {name!r}")
    return NAMES[name]
