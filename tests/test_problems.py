"""Every problem folder, end to end: compile.py -> ptxas (sm_89 and the local GPU) -> launch ->
expected.npz. The failure message carries the runner's report for that problem."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from gpu_runner import Gpu, report, run_problem


KNOWN_GAPS = {  # problem id -> why ptxc fails it (2026-10-28); strict xfail, so a fix shows up as XPASS
    "p11_mul_2d": "operator '//' (floor division) in index expressions of 2-D and 3-D tiles",
    "p25_3d": "operator '//'",
    "p60_2d_odd_width": "operator '//'",
    "p61_2d_narrow": "operator '//'",
    "p62_4d": "operator '//'",
    "p63_2d_large": "operator '//'",
    "r19_3d_last": "operator '//'",
    "f27_softmax_3d": "operator '//'",
    "p64_pow3": "elementwise op 'pow'",
    "p65_pow_tensor": "elementwise op 'pow'",
    "p68_compare_mask": "dtype 'bool' (comparison ops)",
    "p41_gelu": "elementwise op 'erf'",
    "p50_erf": "elementwise op 'erf'",
    "p43_sin_cos": "elementwise ops 'sin' / 'cos'",
    "p52_square_fn": "elementwise op 'square'",
    "p46_where": "dtype 'bool' (comparison ops and 'where')",
    "p42_gelu_tanh": "tanh.approx is ~1e-5 relative, outside this problem's 1e-5 tolerance",
    "a16_relu_mm": "Select statement (documented as absent from the hidden set)",
    "m23_mm_mm": "Select statement (documented as absent from the hidden set)",
}


def describe(res: dict) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(res)
    return buf.getvalue()


def test_problem(problem_dir: Path, gpu: Gpu, compiler: str, repeat: int, request: pytest.FixtureRequest) -> None:
    if problem_dir.name in KNOWN_GAPS:
        request.applymarker(pytest.mark.xfail(reason=KNOWN_GAPS[problem_dir.name], strict=True))
    res = run_problem(problem_dir, compiler=compiler, gpu=gpu, repeat=repeat)
    assert res["ok"], f"{res['problem']}:\n{describe(res)}"
    if not res["checked"]:
        pytest.skip(f"{res['problem']} assembled and ran, but has no expected.npz to check against")
