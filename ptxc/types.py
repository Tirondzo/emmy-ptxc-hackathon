from dataclasses import dataclass
from enum import Enum


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
        """Instruction type suffix: "f32", "s32", "pred"."""
        ...

    @property
    def nbytes(self) -> int: ...

    @property
    def is_float(self) -> bool: ...

    @property
    def bank(self) -> str:
        """The `.reg` declaration this type lives in: "pred", "f32", "f64", "b32" or "b64".

        s32, u32 and b32 share one bank. Register names and counts are per bank.
        """
        ...


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


def ty_of(name: str) -> Ty:
    """Emmy's dtype name ("f32") or C spelling ("float", "int") -> Ty.

    16- and 8-bit types raise Unsupported: they need their own banks and conversions,
    and no recipe handles them yet.
    """
    ...
