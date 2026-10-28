"""Every problem folder, end to end: compile.py -> ptxas (sm_89 and the local GPU) -> launch ->
expected.npz. The failure message carries the runner's report for that problem."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from gpu_runner import Gpu, report, run_problem

KNOWN_GAPS = {  # problem id -> why ptxc fails it; strict xfail, so a fix shows up as XPASS
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
