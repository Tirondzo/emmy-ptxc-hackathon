"""Kernel semantics the problem sets never exercise, run on one warp: simulated (simwarp.py),
and on the GPU when there is one. Integer data, so a sum is exact in any order."""

from __future__ import annotations

import numpy as np
import pytest
from emmy.compiler import dtype as D
from emmy.compiler.ir.elementwise import ElementwiseImpl as Op
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt.leaves import Assign, Let
from kernels import arg, load, ptx_of, shuffle, spec, tile, write

NAN = float("nan")


def run(warp, body, args: list[dict], block: int = 32) -> dict[str, np.ndarray]:
    return warp(ptx_of(body, args, block=block), spec(args, block=block))


def gt(a: str, b) -> BinaryExpr:
    return BinaryExpr(">", Var(a), b if isinstance(b, Literal) else Var(b))


@pytest.mark.parametrize("length", range(2, 33))
def test_shuffle_sums_every_length(warp, length: int) -> None:
    """A power of two: 32 lanes in groups of `length`. Anything else: a block of `length` lanes.
    Lane i holds 2^i, so every subset of lanes has its own sum (mod 2^32, which `add.s32` is)."""
    lanes = 32 if length & (length - 1) == 0 else length
    x = (np.uint32(1) << np.arange(lanes, dtype=np.uint32)).view(np.int32)
    args = [arg("x", "i32", x), arg("out", "i32", n=lanes)]
    out = run(warp, [tile(lanes, load("v", "x"), shuffle(length, v="add"), write("out", "v"))], args, block=lanes)["out"]
    groups = x.reshape(-1, length).sum(axis=1, dtype=np.int32)
    assert out.tolist() == np.repeat(groups, length).tolist()


def test_shuffle_carries_predicates(warp) -> None:
    """A bool state crosses the shuffle as 1 / 0: any-of over groups of 8."""
    x = np.zeros(32, np.float32)
    x[[3, 20]] = 1.0
    args = [arg("x", "f32", x), arg("out", "bool")]
    body = [tile(32, load("x", "x"), Let("v", gt("x", Literal(0.5, "float"))), shuffle(8, v="bitwise_or"), write("out", "v"))]
    assert run(warp, body, args)["out"].tolist() == [lane // 8 in (0, 2) for lane in range(32)]


def test_reduce_aliases(warp) -> None:
    """emmy's spellings of a combine: `sum` is add, `amax` is maximum."""
    x = np.arange(32, dtype=np.float32) - 10
    args = [arg("x", "f32", x), arg("sum", "f32"), arg("max", "f32")]
    body = [tile(32, load("s", "x"), load("m", "x"), shuffle(32, s="sum", m="amax"), write("sum", "s"), write("max", "m"))]
    out = run(warp, body, args)
    assert out["sum"].tolist() == [x.sum()] * 32 and out["max"].tolist() == [x.max()] * 32


def test_select_on_predicates(warp) -> None:
    """`where` and `?:` with bool arms, typed or not; `where` untyped with float arms takes their type."""
    rng = np.random.default_rng(0)
    x, y = (rng.standard_normal(32).astype(np.float32) for _ in range(2))
    args = [arg("x", "f32", x), arg("y", "f32", y), arg("typed", "bool"), arg("untyped", "bool"), arg("ternary", "bool"), arg("floats", "f32")]
    body = [tile(32, load("x", "x"), load("y", "y"),
                 Let("c", gt("x", Literal(0.0, "float"))), Let("p", gt("y", Literal(0.0, "float"))), Let("q", gt("x", "y")),
                 Assign("w", Op("where"), ("c", "p", "q"), dtype=D.BOOL),
                 Assign("u", Op("where"), ("c", "p", "q")),  # untyped: the arms are predicates
                 Let("t", TernaryExpr(Var("c"), Var("p"), Var("q"))),
                 Assign("f", Op("where"), ("c", "x", "y")),  # untyped: the arms are f32, the condition does not decide
                 write("typed", "w"), write("untyped", "u"), write("ternary", "t"), write("floats", "f"))]
    out = run(warp, body, args)
    selected = np.where(x > 0, y > 0, x > y).tolist()
    assert out["typed"].tolist() == out["untyped"].tolist() == out["ternary"].tolist() == selected
    assert out["floats"].tolist() == np.where(x > 0, x, y).tolist()


def test_not_equal_is_unordered(warp) -> None:
    """C's `!=` is true when either side is NaN, in an expression and as the op; `==` is false."""
    x = np.array([1, 2, NAN, NAN, 5, 6, 7, 8] * 4, np.float32)
    y = np.array([1, 3, NAN, 1, NAN, 6, 7.5, 8] * 4, np.float32)
    args = [arg("x", "f32", x), arg("y", "f32", y), arg("ne", "bool"), arg("ne_op", "bool"), arg("eq", "bool")]
    body = [tile(32, load("x", "x"), load("y", "y"),
                 Let("a", BinaryExpr("!=", Var("x"), Var("y"))), Assign("b", Op("not_equal"), ("x", "y"), dtype=D.BOOL),
                 Let("c", BinaryExpr("==", Var("x"), Var("y"))),
                 write("ne", "a"), write("ne_op", "b"), write("eq", "c"))]
    out = run(warp, body, args)
    assert out["ne"].tolist() == out["ne_op"].tolist() == (x != y).tolist()
    assert out["eq"].tolist() == (x == y).tolist()


def test_negative_of_unsigned(warp) -> None:
    """Two's complement: `neg` of the bits read as unsigned is `-x` read back as signed."""
    x = np.array([0, 1, 2, 7, -7, 2**31 - 1, -(2**31) + 1, 12345] * 4, np.int32)
    body = [tile(32, load("u", "x", dtype=D.U32), Assign("n", Op("negative"), ("u",), dtype=D.U32), write("out", "n"))]
    assert run(warp, body, [arg("x", "i32", x), arg("out", "i32")])["out"].tolist() == (-x).tolist()


def test_floor_divide_of_integers(warp) -> None:
    """Non-negative operands, where floor and truncation agree: the domain emmy renders `//` as C's `/` for."""
    x = (np.arange(32, dtype=np.int32) * 7) % 50
    y = np.arange(32, dtype=np.int32) % 6 + 1
    body = [tile(32, load("x", "x"), load("y", "y"), Assign("q", Op("floor_divide"), ("x", "y"), dtype=D.I32), write("out", "q"))]
    assert run(warp, body, [arg("x", "i32", x), arg("y", "i32", y), arg("out", "i32")])["out"].tolist() == (x // y).tolist()
