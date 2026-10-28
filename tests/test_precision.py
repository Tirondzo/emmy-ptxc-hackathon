"""Precision picks between programs that pass the problem's check: the budget over rtol / atol, the CLI
options, the spelling each mode emits, and the open problems on the GPU under every mode."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from gpu_runner import PROBLEM_DIRS, Gpu, all_problems, run_problem

from ptxc import Options, compile_file
from ptxc.driver import compile_kernel
from ptxc.ir import Kernel, Tolerance
from ptxc.ops import SUBNORMAL, ULP, Budget

OPEN = PROBLEM_DIRS[0]
M01 = OPEN / "m01_matmul_512"


def problem(name: str) -> Path:
    for d in all_problems(PROBLEM_DIRS):
        if d.name == name:
            return d
    pytest.skip(f"problem {name} is not checked out")


def compiled(pdir: Path, options: Options = Options(), tolerance: Tolerance | None = None) -> str:
    """The problem's PTX; `tolerance` replaces the one meta.json gives it."""
    ir = next(pdir.glob("*.ir.json"))
    if tolerance is None:
        return compile_file(ir, options)
    return compile_kernel(replace(Kernel.load(ir), tolerance=tolerance), options)


def spellings(ptx: str) -> set[str]:
    """The float opcodes used, without operands: {'add.f32', 'rcp.approx.ftz.f32', 'cvt.rni.f32.f32', ...}."""
    return set(re.findall(r"^\s+((?:add|sub|mul|div|rcp|sqrt|rsqrt|ex2|lg2|fma|tanh|sin|cos|cvt)\.[a-z.0-9]*f32)\b", ptx, re.M))


class TestBudget:
    """`Budget.decide`: a quarter of the tolerance, in the tolerance's own shape; a spelling fits when
    its relative error (in ulps) and its absolute error both do."""

    def test_zero_tolerance_is_exact(self) -> None:
        b = Budget.decide("auto", Tolerance(0, 0))
        assert b == Budget() and str(b) == "exact"
        assert not b.allows(ulps=1) and not b.allows(absolute=SUBNORMAL) and b.allows()
        assert Budget.decide("auto", Tolerance(0, 1e-6)).allows(absolute=SUBNORMAL)  # rtol 0: nothing relative may move, a subnormal may

    def test_open_set_tolerances(self) -> None:
        b = Budget.decide("auto", Tolerance(1e-3, 1e-4))
        assert b == Budget(2.5e-4, 2.5e-5)
        assert b.allows(ulps=1) and b.allows(ulps=2, absolute=SUBNORMAL)
        assert not b.allows(ulps=2**12)  # tanh.approx, 2^-11 relative, needs rtol >= 2e-3
        assert Budget.decide("auto", Tolerance(2e-3, 1e-4)).allows(ulps=2**12)
        assert b.allows(absolute=2**-20.4) and not b.allows(absolute=2**-14.7)  # sin.approx reduced, not raw
        assert Budget.decide("auto", Tolerance(1e-3, 1e-3)).allows(absolute=2**-14.7)
        for rtol, atol in [(1e-4, 1e-4), (1e-6, 1e-6), (1e-5, 1e-6)]:
            assert Budget.decide("auto", Tolerance(rtol, atol)).allows(ulps=2, absolute=SUBNORMAL), (rtol, atol)

    def test_a_quarter_of_rtol(self) -> None:
        assert Budget.decide("auto", Tolerance(4 * ULP, 1e-6)).allows(ulps=1)  # 1 ulp fits
        assert not Budget.decide("auto", Tolerance(4 * ULP, 1e-6)).allows(ulps=2)  # 2 do not
        assert Budget.decide("auto", Tolerance(8 * ULP, 1e-6)).allows(ulps=2)
        assert not Budget.decide("auto", Tolerance(3 * ULP, 1e-6)).allows(ulps=1)
        assert not Budget.decide("auto", Tolerance(1e-3, 1e-40)).allows(absolute=SUBNORMAL)  # atol under 2^-126

    def test_the_ends_ignore_the_tolerance(self) -> None:
        assert Budget.decide("exact", Tolerance(1e-3, 1e-3)) == Budget()
        fast = Budget.decide("fast", Tolerance(0, 0))
        assert fast.allows(ulps=math.inf, absolute=1.0) and str(fast) == "no limit"


@pytest.mark.skipif(not M01.exists(), reason="the open set is not checked out next to this repo")
class TestCli:
    @staticmethod
    def cli(compiler: str, *flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, compiler, str(next(M01.glob("*.ir.json"))), *flags], capture_output=True, text=True)

    @pytest.mark.parametrize(("flags", "options"), [(["--precision=auto"], Options()), (["--precision=exact"], Options(precision="exact")),
                                                    (["--precision=fast"], Options(precision="fast")), (["--raw-trig"], Options(raw_trig=True)),
                                                    (["--precision=exact", "--unroll=none"], Options(precision="exact", unroll="none"))])
    def test_options_match(self, compiler: str, flags: list[str], options: Options) -> None:
        r = self.cli(compiler, *flags)
        assert r.returncode == 0, r.stderr
        assert r.stdout == compiled(M01, options)

    def test_bad_mode_is_a_usage_error(self, compiler: str) -> None:
        r = self.cli(compiler, "--precision=sloppy")
        assert r.returncode == 2 and r.stdout == "" and "usage" in r.stderr


@pytest.mark.skipif(not M01.exists(), reason="the open set is not checked out next to this repo")
class TestSpelling:
    """What the budget does to the PTX text. The recipes' own numerics (`fma.rn`, the `mul.rn` by log2e
    inside exp) are pinned in every mode."""

    def test_matmul_lets_ptxas_fuse(self) -> None:
        assert {"mul.f32", "add.f32"} <= spellings(compiled(M01))  # rtol 1e-3: no rounding modifier to stop the fma
        assert {"mul.rn.f32", "add.rn.f32"} <= spellings(compiled(M01, Options(precision="exact")))

    def test_exact_problem_keeps_rn(self) -> None:
        ptx = compiled(problem("p20_three_inputs"))  # x * y + z at rtol = atol = 0: fma would change bits
        assert {"mul.rn.f32", "add.rn.f32"} <= spellings(ptx)
        assert not {"mul.f32", "add.f32"} & spellings(ptx)
        assert {"mul.f32", "add.f32"} <= spellings(compiled(problem("p20_three_inputs"), Options(precision="fast")))

    def test_approximate_division_within_tolerance(self) -> None:
        assert {"rcp.approx.ftz.f32", "sqrt.approx.ftz.f32"} <= spellings(compiled(problem("f01_rmsnorm")))
        assert {"rcp.rn.f32", "sqrt.rn.f32"} <= spellings(compiled(problem("f01_rmsnorm"), Options(precision="exact")))
        assert "div.rn.f32" in spellings(compiled(problem("p12_div_abs")))  # exact problem
        assert "div.full.ftz.f32" in spellings(compiled(problem("p12_div_abs"), Options(precision="fast")))

    def test_ftz_needs_an_absolute_budget(self) -> None:
        assert "rcp.approx.f32" in spellings(compiled(problem("f01_rmsnorm"), tolerance=Tolerance(1e-3, 0)))

    def test_transcendentals_get_ftz_only(self) -> None:
        fast, exact = spellings(compiled(problem("p13_exp"))), spellings(compiled(problem("p13_exp"), Options(precision="exact")))
        assert "ex2.approx.ftz.f32" in fast and "ex2.approx.f32" in exact
        assert "mul.rn.f32" in fast  # the recipe's own x * log2e stays pinned

    def test_tanh_takes_the_instruction_when_rtol_allows(self) -> None:
        p16 = problem("p16_tanh")  # rtol 1e-5
        assert "tanh.approx.f32" not in spellings(compiled(p16))
        assert "tanh.approx.f32" in spellings(compiled(p16, tolerance=Tolerance(2e-3, 1e-4)))
        assert "tanh.approx.f32" not in spellings(compiled(p16, tolerance=Tolerance(1e-3, 1e-4)))
        assert "tanh.approx.f32" in spellings(compiled(p16, Options(precision="fast")))

    def test_sin_has_three_spellings(self) -> None:
        """Polynomial at the 1e-5 / 1e-6 tier; `sin.approx` on a reduced argument (one `cvt.rni` per call)
        once atol has room for 2^-20.4; raw only with `raw_trig` and room for 2^-14.7."""
        p43 = problem("p43_sin_cos")
        polynomial, reduced, raw = {"cvt.rni.f32.f32"}, {"sin.approx.ftz.f32", "cvt.rni.f32.f32"}, {"sin.approx.ftz.f32"}
        assert not {"sin.approx.ftz.f32", "sin.approx.f32"} & spellings(compiled(p43)) and polynomial <= spellings(compiled(p43))
        assert reduced <= spellings(compiled(p43, Options(precision="fast")))
        assert reduced <= spellings(compiled(p43, Options(raw_trig=True), tolerance=Tolerance(1e-3, 1e-4)))  # 2.5e-5 < 2^-14.7
        fast_raw = spellings(compiled(p43, Options(precision="fast", raw_trig=True)))
        assert raw <= fast_raw and "cvt.rni.f32.f32" not in fast_raw
        assert "cvt.rni.f32.f32" not in spellings(compiled(p43, Options(raw_trig=True), tolerance=Tolerance(1e-3, 1e-3)))

    def test_trace_prints_the_budget(self, capsys: pytest.CaptureFixture[str]) -> None:
        compiled(M01, Options(trace=("precision",)))
        compiled(problem("p20_three_inputs"), Options(trace=("precision",)))
        compiled(problem("p16_tanh"), Options(trace=("precision",), precision="fast"))
        lines = [l for l in capsys.readouterr().err.splitlines() if l.startswith("ptxc: precision")]
        assert lines == ["ptxc: precision: rtol=0.001 atol=0.001: budget 2097 ulp relative, 2.5e-04 absolute",
                         "ptxc: precision: rtol=0 atol=0: exact",
                         "ptxc: precision: rtol=1e-05 atol=1e-06: no limit"]


class TestOnGpu:
    """Every mode on one problem per kind of arithmetic: a matmul (fma), rmsnorm (rcp, sqrt), attention
    (exp, rcp), a plain division and a reciprocal checked exactly, x * y + z checked exactly."""

    PROBLEMS = ["m01_matmul_512", "f01_rmsnorm", "a02_attention_128", "p12_div_abs", "p26_reciprocal", "p20_three_inputs"]

    @staticmethod
    def run(gpu: Gpu, name: str, mode: str, **options: object) -> dict:
        opts = Options(precision=mode, **options)  # type: ignore[arg-type]
        return run_problem(problem(name), compiler=lambda ir: compile_file(ir, opts), gpu=gpu)

    @staticmethod
    def output(res: dict) -> dict:
        (c,) = res["kernels"][0]["outputs"].values()
        return c

    @pytest.mark.parametrize("mode", ["auto", "exact"])
    @pytest.mark.parametrize("name", PROBLEMS)
    def test_auto_and_exact_pass(self, gpu: Gpu, name: str, mode: str) -> None:
        res = self.run(gpu, name, mode)
        assert res["ok"], json.dumps(res, indent=1, default=str)

    @pytest.mark.parametrize("name", PROBLEMS)
    def test_fast_passes_where_there_is_tolerance(self, gpu: Gpu, name: str) -> None:
        res = self.run(gpu, name, "fast")
        if res["rtol"] == 0:
            pytest.skip("an exact problem: fast may legitimately miss it")
        assert res["ok"], json.dumps(res, indent=1, default=str)

    def test_contraction_changes_bits(self, gpu: Gpu) -> None:
        """Why auto keeps `.rn` at tolerance zero: fused, x * y + z differs from numpy's two roundings
        in about a quarter of the elements, by half an ulp of the product (x * y reaches about 20
        here), which is many ulps of the result where the sum cancels."""
        res = self.run(gpu, "p20_three_inputs", "fast")
        c = self.output(res)
        assert not res["ok"] and 0 < c["mismatches"] < c["size"]
        assert c["max_abs_err"] < 32 * 2.0**-24 and c["max_ulp"] > 1

    def test_tanh_approx_stays_within_its_bound(self, gpu: Gpu) -> None:
        """The numbers the recipes ask for are claims the GPU can check: `tanh.approx` over |x| up to 16
        within 2^12 ulp."""
        c = self.output(self.run(gpu, "p84_tanh_wide", "fast"))
        assert 1 < c["max_ulp"] <= 2**12, c

    def test_sin_approx_reduced_holds_for_wide_arguments(self, gpu: Gpu) -> None:
        """|x| up to 120: reduced into [-pi, pi], `sin.approx` keeps its 2^-20.4 and passes the 1e-5 / 1e-6
        problem; on the raw argument the ISA promises only 2^-14.7, which is what it delivers."""
        res = self.run(gpu, "p81_sin_wide", "fast")
        assert res["ok"], json.dumps(res, indent=1, default=str)
        assert self.output(res)["max_abs_err"] <= 2**-20.4
        raw = self.output(self.run(gpu, "p81_sin_wide", "fast", raw_trig=True))
        assert 2**-20.4 < raw["max_abs_err"] <= 2**-14.7, raw
