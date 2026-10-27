"""Elementwise ops: name -> recipe."""

from collections.abc import Callable, Sequence

from .errors import Unsupported
from .ptx import PtxBuilder, Value
from .types import Ty

type Recipe = Callable[..., Value]  # (b: PtxBuilder, ty: Ty, *args: Value) -> Value

OPS: dict[str, Recipe] = {}


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


# ---- arithmetic: int and float; the int spelling differs (mul.lo, div, rem, no .rn) ----


@recipe("add")
def add(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value: ...


@recipe("subtract")
def subtract(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value: ...


@recipe("multiply")
def multiply(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value: ...


@recipe("divide", "true_divide")
def divide(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value: ...


@recipe("maximum")
def maximum(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value: ...


@recipe("minimum")
def minimum(b: PtxBuilder, ty: Ty, x: Value, y: Value) -> Value: ...


@recipe("negative")
def negative(b: PtxBuilder, ty: Ty, x: Value) -> Value: ...


@recipe("abs", "absolute")
def absolute(b: PtxBuilder, ty: Ty, x: Value) -> Value: ...


@recipe("copy")
def copy(b: PtxBuilder, ty: Ty, x: Value) -> Value: ...


@recipe("relu")
def relu(b: PtxBuilder, ty: Ty, x: Value) -> Value: ...


# ---- float only --------------------------------------------------------------------


@recipe("reciprocal")
def reciprocal(b: PtxBuilder, ty: Ty, x: Value) -> Value: ...


@recipe("sqrt")
def sqrt(b: PtxBuilder, ty: Ty, x: Value) -> Value: ...


@recipe("rsqrt")
def rsqrt(b: PtxBuilder, ty: Ty, x: Value) -> Value: ...


@recipe("exp")
def exp(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    """`ex2.approx.f32(x * log2 e)`."""
    ...


@recipe("log")
def log(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    """`lg2.approx.f32(x) * ln 2`."""
    ...


@recipe("tanh")
def tanh(b: PtxBuilder, ty: Ty, x: Value) -> Value: ...


@recipe("sigmoid")
def sigmoid(b: PtxBuilder, ty: Ty, x: Value) -> Value:
    """`1 / (1 + exp(-x))`, composed from the recipes above."""
    ...
