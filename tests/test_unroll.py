"""Unrolling picks between correct programs: the policy, the CLI option, the shape each mode
emits, and every emission path on the GPU under every mode (set in-process through `Options`)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from gpu_runner import PROBLEM_DIRS, Gpu, all_problems, run_problem

from ptxc import Options, compile_file
from ptxc.lower.loops import unroll_factor

OPEN = PROBLEM_DIRS[0]
M01 = OPEN / "m01_matmul_512"


def problem(name: str) -> Path:
    for d in all_problems(PROBLEM_DIRS):
        if d.name == name:
            return d
    pytest.skip(f"problem {name} is not checked out")


def compiled(pdir: Path, options: Options) -> str:
    return compile_file(next(pdir.glob("*.ir.json")), options)


def factor(hinted: bool, trip: int | None, body: int, expanded: int | None = None, **options: object) -> int:
    """`unroll_factor` for a body of `body` statements that expands to `expanded` (itself, when it holds no loop)."""
    return unroll_factor(hinted, trip, body, body if expanded is None else expanded, Options(**options))  # type: ignore[arg-type]


class TestPolicy:
    """The numbers are the open set's loops: a body's own length, and what it becomes once its
    nested loops are unrolled (`loops.sizes`)."""

    def test_hinted_loop_in_full_when_it_fits(self) -> None:
        assert factor(True, 64, 11) == 64  # m01's k-loop

    def test_hinted_loop_by_a_divisor_when_it_does_not(self) -> None:
        assert factor(True, 64, 133, 1145) == 1  # a01's key loop around 4 straight dot products: no divisor fits
        assert factor(True, 64, 133, 1145, unroll_quota=4096) == 2
        assert factor(True, 50, 37) == 50  # m13: 1850 fits
        assert factor(True, 64, 11, unroll_quota=100) == 8

    def test_unhinted_loop_gets_the_auto_factor(self) -> None:
        assert factor(False, 1024, 2) == 8  # a strided reduction over 128K elements
        assert factor(False, None, 2) == 8  # a trip count that depends on the thread
        assert factor(False, 128, 20, 144) == 8  # a02's key loop: short as written, its inner loop straight
        assert factor(False, 3, 2, unroll_auto_full=0) == 3  # shorter than the factor: in full
        assert factor(False, 64, 200) == 1  # a long body stays rolled
        assert factor(False, 1024, 2, unroll_auto=4) == 4

    def test_small_unhinted_loop_goes_straight(self) -> None:
        assert factor(False, 64, 4) == 64  # a01's inner dot products: 256 statements
        assert factor(False, 512, 2) == 512  # r02's strided reduction: 1024 statements
        assert factor(False, 512, 3) == 8  # one statement more per iteration: the factor
        assert factor(False, 64, 4, unroll_auto_full=0) == 8
        assert factor(False, 64, 4, unroll_quota=100) == 8  # the quota caps it too

    def test_auto_factor_shrinks_into_the_quota(self) -> None:
        assert factor(False, 64, 60, unroll_quota=130) == 2
        assert factor(False, 8, 20, 776) == 2  # m01's stage loop around its straight k-loop: 4 x 776 > 2048

    def test_modes(self) -> None:
        assert factor(True, 64, 11, unroll="none") == 1
        assert factor(False, 8, 20, 776, unroll="force") == 8  # force ignores the quota
        assert factor(False, None, 20, 776, unroll="force") == 2  # ... but needs a trip count: else auto's factor under the quota
        assert factor(False, 16, 100) == 1  # auto leaves a long body rolled

    def test_empty_body_is_never_unrolled(self) -> None:
        assert factor(True, 64, 0) == 1


@pytest.mark.skipif(not M01.exists(), reason="the open set is not checked out next to this repo")
class TestCli:
    """`--unroll auto|none|force|QUOTA`, the one option; the same PTX as the `Options` it stands for."""

    @staticmethod
    def cli(compiler: str, *flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, compiler, str(next(M01.glob("*.ir.json"))), *flags], capture_output=True, text=True)

    @pytest.mark.parametrize(("flag", "options"), [("none", Options(unroll="none")), ("force", Options(unroll="force")),
                                                   ("100", Options(unroll_quota=100)), ("auto", Options())])
    def test_option_matches_options(self, compiler: str, flag: str, options: Options) -> None:
        r = self.cli(compiler, f"--unroll={flag}")
        assert r.returncode == 0, r.stderr
        assert r.stdout == compiled(M01, options)

    @pytest.mark.parametrize("bad", [["--bogus"], ["--unroll=maybe"], ["--unroll"], ["--no-unroll"]])
    def test_bad_flags_are_usage_errors(self, compiler: str, bad: list[str]) -> None:
        r = self.cli(compiler, *bad)
        assert r.returncode == 2 and r.stdout == "" and "usage" in r.stderr


@pytest.mark.skipif(not M01.exists(), reason="the open set is not checked out next to this repo")
class TestShape:
    """What each mode does to the open matmul: two rolled loops; the k-loop straight inside a stage
    loop unrolled by 2 (4 x 776 statements is over the quota, 2 x 776 under it); no loop under force."""

    @pytest.mark.parametrize(("mode", "loops"), [("none", 2), ("auto", 1), ("force", 0)])
    def test_loops_left(self, mode: str, loops: int) -> None:
        assert len(re.findall(r"^L_loop_\d+:", compiled(M01, Options(unroll=mode)), re.M)) == loops

    def test_decisions_see_the_nest(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The stage loop is decided knowing its k-loop will be straight: a body of 22 that expands to 726."""
        compiled(M01, Options(trace=("unroll",)))
        lines = [l for l in capsys.readouterr().err.splitlines() if l.startswith("ptxc: unroll")]
        assert lines == ["ptxc: unroll: _ks: trip=8 hinted=False body=20->776 -> x2"] + \
                        ["ptxc: unroll: _ki: trip=64 hinted=True body=12->12 -> straight"] * 2

    def test_unrolled_copies_share_one_base(self) -> None:
        """2 stages x 64 copies of the k-loop body, four loads each (two rows of `_a_smem`, two columns
        of `_b_smem`), through displacements off a base computed once per stage: 512 shared loads
        and no integer division past the stage indices."""
        ptx = compiled(M01, Options())
        assert ptx.count("ld.shared.f32") == 2 * 64 * 4
        assert ptx.count("div.") <= 8 and ptx.count("rem.") <= 8  # the tile decode, and two stage indices per copy

    def test_without_cse_the_lowering_is_raw(self) -> None:
        ptx = compiled(M01, Options(cse=False))
        assert ptx.count("ld.shared.f32") == 2 * 64 * 4
        assert ptx.count("div.") > 64  # one per copy per access: what ptxas then merges itself


class TestOnGpu:
    """One problem per emission path, under every option set: hinted full unroll (m01), a uniform
    strided trip count (r02: 65536 / 128 per thread), a thread-dependent one with a tail loop
    (r10: 1000 / 128), nested unhinted loops (a02), and a remainder after a partial unroll (m13)."""

    OPTIONS = {"none": Options(unroll="none"), "auto": Options(), "force": Options(unroll="force"), "no-cse": Options(cse=False)}

    @pytest.mark.parametrize("variant", list(OPTIONS))
    @pytest.mark.parametrize("name", ["m01_matmul_512", "r02_big_sum", "r10_sum_1000", "a02_attention_128", "m13_odd_100"])
    def test_every_variant_is_correct(self, gpu: Gpu, name: str, variant: str) -> None:
        options = self.OPTIONS[variant]
        res = run_problem(problem(name), compiler=lambda ir: compile_file(ir, options), gpu=gpu)
        assert res["ok"], json.dumps(res, indent=1, default=str)
