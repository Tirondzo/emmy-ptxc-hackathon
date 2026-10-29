"""Constant folding must agree with the instructions it replaces.

`/` is C's: toward zero, like `div.s32`. `//` and `%` are Python's, floored, as torch and emmy's own
folder define them; emmy's CUDA renders them as C's `/` and `%`, which agree only for non-negative
operands. The GPU test runs the `/` pairs through `div.s32` / `rem.s32` and compares bit for bit;
test_warp.py does the same for the floored recipes.
"""

from __future__ import annotations

import numpy as np
import pytest
from gpu_runner import Gpu, assemble, launch

from ptxc.lower.expr import arith, fold
from ptxc.types import INDEX, Imm

PAIRS = [(x, y) for x in (-7, -6, -5, -1, 0, 1, 5, 6, 7) for y in (-3, -2, -1, 1, 2, 3)]

DIV_REM_PTX = """\
.version 8.0
.target sm_89
.address_size 64

.visible .entry k_div_rem(
	.param .u64 param_x,
	.param .u64 param_y,
	.param .u64 param_q,
	.param .u64 param_r
)
{
	.reg .b32 %r<8>;
	.reg .b64 %rd<16>;
	ld.param.u64 %rd1, [param_x];
	ld.param.u64 %rd2, [param_y];
	ld.param.u64 %rd3, [param_q];
	ld.param.u64 %rd4, [param_r];
	cvta.to.global.u64 %rd5, %rd1;
	cvta.to.global.u64 %rd6, %rd2;
	cvta.to.global.u64 %rd7, %rd3;
	cvta.to.global.u64 %rd8, %rd4;
	mov.u32 %r1, %tid.x;
	mul.wide.s32 %rd9, %r1, 4;
	add.u64 %rd10, %rd5, %rd9;
	add.u64 %rd11, %rd6, %rd9;
	add.u64 %rd12, %rd7, %rd9;
	add.u64 %rd13, %rd8, %rd9;
	ld.global.s32 %r2, [%rd10];
	ld.global.s32 %r3, [%rd11];
	div.s32 %r4, %r2, %r3;
	rem.s32 %r5, %r2, %r3;
	st.global.s32 [%rd12], %r4;
	st.global.s32 [%rd13], %r5;
	ret;
}
"""


@pytest.mark.parametrize(("x", "y"), PAIRS)
def test_fold_divisions(x: int, y: int) -> None:
    q, r = fold("//", x, y), fold("%", x, y)
    assert q * y + r == x, "`//` and `%` are one division"
    assert q <= x / y < q + 1, "the quotient is floored"
    assert r == 0 or (r < 0) == (y < 0), "the remainder takes the divisor's sign"
    assert abs(fold("/", x, y)) == abs(x) // abs(y), "`/` is truncated toward zero"


def test_arith_folds_immediates() -> None:
    """Two constants fold to an immediate without emitting anything (no builder is active here)."""
    assert arith("//", INDEX(-7), INDEX(2)) == Imm(-4, INDEX)
    assert arith("%", INDEX(-7), INDEX(2)) == Imm(1, INDEX)
    assert arith("//", INDEX(7), INDEX(-2)) == Imm(-4, INDEX)
    assert arith("/", INDEX(-7), INDEX(2)) == Imm(-3, INDEX)


def test_fold_is_exact_past_2_53() -> None:
    """Folding divides in integers: `x / y` in floats would round a 64-bit quotient."""
    big = 2**60 + 1
    assert fold("/", big, 3) == big // 3 and fold("/", -big, 3) == -(big // 3)
    assert fold("/", 7.0, 2.0) == 3.5


def test_fold_matches_div_rem_s32(gpu: Gpu) -> None:
    """The folded `/` is what `div.s32` computes for the same operands, and `rem.s32` what is left."""
    xs, ys = (np.array(v, np.int32) for v in zip(*PAIRS, strict=True))
    n = len(PAIRS)
    spec = {
        "entry": "k_div_rem", "arg_order": ["x", "y", "q", "r"],
        "grid": [1, 1, 1], "block": [n, 1, 1], "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "i32", "shape": [n], "data": xs.tolist()},
            {"name": "y", "dtype": "i32", "shape": [n], "data": ys.tolist()},
            {"name": "q", "dtype": "i32", "shape": [n], "output": True},
            {"name": "r", "dtype": "i32", "shape": [n], "output": True},
        ],
    }
    out = launch(gpu, assemble(DIV_REM_PTX, gpu.arch).cubin, spec, {}).outputs
    assert out["q"].tolist() == [fold("/", x, y) for x, y in PAIRS]
    assert out["r"].tolist() == [x - y * fold("/", x, y) for x, y in PAIRS]
