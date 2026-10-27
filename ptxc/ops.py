"""Elementwise ops: name -> recipe.

Numerics are pinned, not configured: `.rn` on float add/sub/mul (IEEE, and it keeps ptxas
from contracting mul+add into fma), `.approx` transcendentals through ex2 / lg2 / tanh.
"""

from collections.abc import Callable, Sequence

from .errors import Unsupported
from .ptx import Imm, PtxBuilder, Value
from .types import Ty

type Recipe = Callable[..., Value]  # (b: PtxBuilder, ty: Ty, *args: Value) -> Value

OPS: dict[str, Recipe] = {}

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def recipe(*names: str) -> Callable[[Recipe], Recipe]:
    def register(fn: Recipe) -> Recipe:
        for name in names:
            OPS[name] = fn
        return fn

    return register


def apply(b: PtxBuilder, name: str, ty: Ty, args: Sequence[Value]) -> Value:
    if name not in OPS:
        raise Unsupported(f"elementwise op {name!r}")
    return OPS[name](b, ty, *args)


def floating(name: str, ty: Ty, f64: bool = True) -> str:
    """`name` for a float type; f64 only where PTX spells it."""
    if ty.is_float and (ty.bits == 32 or f64):
        return name
    raise Unsupported(f"{name} on {ty.ptx}")


# ---- int and float; the int spelling differs (mul.lo, div, no .rn) --------------------


@recipe("add")
def add(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value:
    return b.op("add.rn" if ty.is_float else "add", ty, x, y)


@recipe("subtract")
def subtract(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value:
    return b.op("sub.rn" if ty.is_float else "sub", ty, x, y)


@recipe("multiply")
def multiply(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value:
    return b.op("mul.rn" if ty.is_float else "mul.lo", ty, x, y)


@recipe("divide", "true_divide")
def divide(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value:
    return b.op("div.rn" if ty.is_float else "div", ty, x, y)


@recipe("remainder", "mod")
def remainder(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value:
    if ty.is_float:
        raise Unsupported("float remainder")
    return b.op("rem", ty, x, y)


@recipe("maximum")
def maximum(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value:
    return b.op("max", ty, x, y)


@recipe("minimum")
def minimum(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value:
    return b.op("min", ty, x, y)


@recipe("negative")
def negative(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    return b.op("neg", ty, x)


@recipe("abs", "absolute")
def absolute(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    return b.op("abs", ty, x)


@recipe("copy")
def copy(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    return x


@recipe("relu")
def relu(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    return b.op("max", ty, x, Imm(0, ty))


# ---- float only --------------------------------------------------------------------


@recipe("reciprocal")
def reciprocal(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    return b.op(floating("rcp.rn", ty), ty, x)


@recipe("sqrt")
def sqrt(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    return b.op(floating("sqrt.rn", ty), ty, x)


@recipe("rsqrt")
def rsqrt(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    return b.op(floating("rsqrt.approx", ty), ty, x)


@recipe("exp")
def exp(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    """`ex2.approx.f32(x * log2 e)`."""
    scaled = b.op("mul.rn", ty, x, Imm(LOG2E, ty))
    return b.op(floating("ex2.approx", ty, f64=False), ty, scaled)


@recipe("log")
def log(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    """`lg2.approx.f32(x) * ln 2`."""
    lg2 = b.op(floating("lg2.approx", ty, f64=False), ty, x)
    return b.op("mul.rn", ty, lg2, Imm(LN2, ty))


@recipe("tanh")
def tanh(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    return b.op(floating("tanh.approx", ty, f64=False), ty, x)


@recipe("sigmoid")
def sigmoid(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    """`1 / (1 + exp(-x))`, composed from the recipes above."""
    return reciprocal(b, ty, add(b, ty, Imm(1, ty), exp(b, ty, negative(b, ty, x))))
