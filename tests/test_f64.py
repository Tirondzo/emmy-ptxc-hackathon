"""The f64 transcendentals on the GPU against numpy in float64: PTX has no f64 `ex2` / `lg2` / `sin` /
`tanh`, so each recipe is a reduction and an `fma.rn.f64` series (ops.py, the f64 section), and these
bounds are what tests/FINDINGS.md records for them. The warp simulator does not interpret f64, so GPU only."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from emmy.compiler import dtype as D
from emmy.compiler.ir.elementwise import ElementwiseImpl as Op
from emmy.compiler.ir.stmt.leaves import Assign
from gpu_runner import Gpu, assemble, launch
from kernels import arg, load, ptx_of, spec, tile, write
from scipy.special import erf

N = 1024
RNG = np.random.default_rng(0)
X = RNG.standard_normal(N)
WIDE = RNG.uniform(-700, 700, N)


def run(gpu: Gpu, op: str, **inputs: np.ndarray) -> np.ndarray:
    """`out = op(inputs)` elementwise in f64, one element per thread."""
    body = [load(f"v_{n}", n) for n in inputs] + [Assign("r", Op(op), tuple(f"v_{n}" for n in inputs), dtype=D.F64), write("out", "r")]
    args = [arg(n, "f64", v) for n, v in inputs.items()] + [arg("out", "f64", n=N)]
    ptx = ptx_of([tile(N, *body)], args, block=128, grid=N // 128)
    return launch(gpu, assemble(ptx, gpu.arch).cubin, spec(args, block=128, grid=N // 128), {}).outputs["out"]


def ulps(got: np.ndarray, ref: np.ndarray) -> float:
    assert np.array_equal(np.isnan(got), np.isnan(ref)) and np.array_equal(np.isinf(got), np.isinf(ref))
    ok = np.isfinite(ref)
    return float(np.max(np.abs(got[ok] - ref[ok]) / np.spacing(np.abs(ref[ok]))))


CASES: list[tuple[str, dict[str, np.ndarray], Callable, float]] = [
    ("exp", {"x": X}, np.exp, 1),
    ("exp", {"x": WIDE}, np.exp, 1),
    ("log", {"x": np.abs(X) + 0.5}, np.log, 2),
    ("log", {"x": np.exp(WIDE)}, np.log, 2),
    ("sin", {"x": X * 3}, np.sin, 2),
    ("cos", {"x": X * 3}, np.cos, 2),
    ("sin", {"x": WIDE * 100}, np.sin, 3),
    ("tanh", {"x": X * 3}, np.tanh, 3),
    ("tanh", {"x": X * 1e-3}, np.tanh, 3),
    ("erf", {"x": X * 2}, erf, 3),
    ("erf", {"x": X * 0.1}, erf, 3),
    ("sigmoid", {"x": X * 5}, lambda v: 1 / (1 + np.exp(-v)), 3),
    ("pow", {"x": np.abs(X) + 0.1, "y": RNG.uniform(-3, 3, N)}, np.power, 3),
    ("pow", {"x": X, "y": np.full(N, 2.0)}, np.power, 2),
    ("pow", {"x": X, "y": np.full(N, 3.0)}, np.power, 2),
]


@pytest.mark.parametrize(("op", "inputs", "ref", "bound"), CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_within_a_few_ulp(gpu: Gpu, op: str, inputs: dict[str, np.ndarray], ref: Callable, bound: float) -> None:
    assert ulps(run(gpu, op, **inputs), ref(*inputs.values())) <= bound


SPECIALS = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-310, 709.78, 710.0, -745.0, -746.0, 1.0, -1.0])


@pytest.mark.parametrize(("op", "expected"), [
    ("exp", [np.nan, np.inf, 0.0, 1.0, 1.0, 1.0, 1.7928e308, np.inf, 4.94e-324, 0.0, np.e, 1 / np.e]),
    ("log", [np.nan, np.inf, np.nan, -np.inf, -np.inf, np.log(1e-310), np.log(709.78), np.log(710.0), np.nan, np.nan, 0.0, np.nan]),
    ("tanh", [np.nan, 1.0, -1.0, 0.0, -0.0, 1e-310, 1.0, 1.0, -1.0, -1.0, np.tanh(1.0), -np.tanh(1.0)]),
    ("erf", [np.nan, 1.0, -1.0, 0.0, -0.0, 1e-310 * 2 / np.sqrt(np.pi), 1.0, 1.0, -1.0, -1.0, erf(1.0), -erf(1.0)]),
    ("sigmoid", [np.nan, 1.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1 / (1 + np.exp(-1.0)), 1 / (1 + np.e)]),
])
def test_special_values(gpu: Gpu, op: str, expected: list[float]) -> None:
    """NaN through, the infinities and zeros as IEEE and C have them, the ends of exp's range (inf, the
    smallest subnormal, 0), a subnormal argument."""
    got = run(gpu, op, x=np.concatenate([SPECIALS, np.zeros(N - len(SPECIALS))]))[: len(SPECIALS)]
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=0, equal_nan=True)
    if op in ("tanh", "erf"):  # odd functions keep the sign of zero
        assert np.array_equal(np.signbit(got[3:5]), [False, True])


def test_pow_special_values(gpu: Gpu) -> None:
    """C's pow: 0^y, inf^-1, a negative base to an odd integer and to a fraction, y = 0 and x = 1 against anything."""
    x = np.array([0.0, 0.0, np.inf, -2.0, -2.0, -8.0, 5.0, np.nan, 1.0, 2.0, 0.5])
    y = np.array([2.0, -1.0, -1.0, 3.0, 0.5, 1 / 3, 0.0, 2.0, np.nan, 1024.0, -1075.0])
    pad = np.ones(N - len(x))
    got = run(gpu, "pow", x=np.concatenate([x, pad]), y=np.concatenate([y, pad]))[: len(x)]
    expected = [0.0, np.inf, 0.0, -8.0, np.nan, np.nan, 1.0, np.nan, 1.0, np.inf, np.inf]
    np.testing.assert_array_equal(got, expected)
