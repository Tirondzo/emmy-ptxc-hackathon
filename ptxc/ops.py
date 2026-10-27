"""Elementwise ops: name -> recipe.

Numerics are pinned, not configured: `.rn` on float add/sub/mul (IEEE, and it keeps ptxas
from contracting mul+add into fma), `.approx` transcendentals through ex2 / lg2 / tanh.
"""

from collections.abc import Callable, Sequence

from .errors import Unsupported
from .ptx import Value, op
from .types import Ty

type Recipe = Callable[..., Value]  # (ty: Ty, *args: Value) -> Value

OPS: dict[str, Recipe] = {}

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def recipe(*names: str) -> Callable[[Recipe], Recipe]:
    def register(fn: Recipe) -> Recipe:
        for name in names:
            OPS[name] = fn
        return fn

    return register


def apply(name: str, ty: Ty, args: Sequence[Value]) -> Value:
    if name not in OPS:
        raise Unsupported(f"elementwise op {name!r}")
    return OPS[name](ty, *args)


def floating(name: str, ty: Ty, f64: bool = True) -> str:
    """`name` for a float type; f64 only where PTX spells it."""
    if ty.is_float and (ty.bits == 32 or f64):
        return name
    raise Unsupported(f"{name} on {ty.ptx}")


# ---- int and float; the int spelling differs (mul.lo, div, no .rn) --------------------


@recipe("add")
def add(ty: Ty, x: Value, y: Value) -> Value:
    return op("add.rn" if ty.is_float else "add", ty, x, y)


@recipe("subtract")
def subtract(ty: Ty, x: Value, y: Value) -> Value:
    return op("sub.rn" if ty.is_float else "sub", ty, x, y)


@recipe("multiply")
def multiply(ty: Ty, x: Value, y: Value) -> Value:
    return op("mul.rn" if ty.is_float else "mul.lo", ty, x, y)


@recipe("divide", "true_divide")
def divide(ty: Ty, x: Value, y: Value) -> Value:
    return op("div.rn" if ty.is_float else "div", ty, x, y)


@recipe("remainder", "mod")
def remainder(ty: Ty, x: Value, y: Value) -> Value:
    if ty.is_float:
        raise Unsupported("float remainder")
    return op("rem", ty, x, y)


@recipe("maximum")
def maximum(ty: Ty, x: Value, y: Value) -> Value:
    return op("max", ty, x, y)


@recipe("minimum")
def minimum(ty: Ty, x: Value, y: Value) -> Value:
    return op("min", ty, x, y)


@recipe("negative")
def negative(ty: Ty, x: Value) -> Value:
    return op("neg", ty, x)


@recipe("abs", "absolute")
def absolute(ty: Ty, x: Value) -> Value:
    return op("abs", ty, x)


@recipe("copy")
def copy(ty: Ty, x: Value) -> Value:
    return x


@recipe("relu")
def relu(ty: Ty, x: Value) -> Value:
    return op("max", ty, x, ty(0))


# ---- float only --------------------------------------------------------------------


@recipe("reciprocal")
def reciprocal(ty: Ty, x: Value) -> Value:
    return op(floating("rcp.rn", ty), ty, x)


@recipe("sqrt")
def sqrt(ty: Ty, x: Value) -> Value:
    return op(floating("sqrt.rn", ty), ty, x)


@recipe("rsqrt")
def rsqrt(ty: Ty, x: Value) -> Value:
    return op(floating("rsqrt.approx", ty), ty, x)


@recipe("exp")
def exp(ty: Ty, x: Value) -> Value:
    scaled = op("mul.rn", ty, x, ty(LOG2E))
    return op(floating("ex2.approx", ty, f64=False), ty, scaled)


@recipe("log")
def log(ty: Ty, x: Value) -> Value:
    lg2 = op(floating("lg2.approx", ty, f64=False), ty, x)
    return op("mul.rn", ty, lg2, ty(LN2))


@recipe("tanh")
def tanh(ty: Ty, x: Value) -> Value:
    return op(floating("tanh.approx", ty, f64=False), ty, x)


@recipe("sigmoid")
def sigmoid(ty: Ty, x: Value) -> Value:
    return reciprocal(ty, add(ty, ty(1), exp(ty, negative(ty, x))))
