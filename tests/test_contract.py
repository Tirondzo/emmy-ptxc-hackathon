"""The submission contract, independent of any GPU result:

    python compile.py path/to/kernel.ir.json > out.ptx

PTX only on stdout, ASCII, the entry named as the IR names it, the parameters in meta.json's
`arg_order` (not the IR node's inputs), deterministic, and quick.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from gpu_runner import PROBLEM_DIRS, Gpu, all_problems, run_problem

OPEN = PROBLEM_DIRS[0]
P00 = OPEN / "p00_scalar_add"


def compile_ir(compiler: str, ir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, compiler, str(ir)], capture_output=True, text=True, timeout=120)


def ir_of(pdir: Path) -> Path:
    return next(pdir.glob("*.ir.json"))


def signature(ptx: str) -> tuple[str, list[str]]:
    """(entry name, parameter names) from the `.visible .entry` declaration."""
    m = re.search(r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S)
    assert m, "no .visible .entry in the PTX"
    params = re.findall(r"\.param\s+\.\w+\s+(\w+)", m.group(2))
    return m.group(1), params


@pytest.mark.skipif(not P00.exists(), reason="the open set is not checked out next to this repo")
class TestCli:
    def test_no_args_is_usage(self, compiler: str) -> None:
        r = subprocess.run([sys.executable, compiler], capture_output=True, text=True)
        assert r.returncode == 2
        assert "usage" in r.stderr.lower()
        assert r.stdout == ""

    def test_missing_file_fails_cleanly(self, compiler: str, tmp_path: Path) -> None:
        r = compile_ir(compiler, tmp_path / "nope.ir.json")
        assert r.returncode != 0
        assert r.stdout == "", "nothing may reach stdout when compilation fails"

    def test_stdout_is_ascii_ptx(self, compiler: str) -> None:
        r = compile_ir(compiler, ir_of(P00))
        assert r.returncode == 0, r.stderr
        ptx = r.stdout
        assert ptx.isascii()
        first = next(l for l in ptx.splitlines() if l.strip() and not l.strip().startswith("//"))
        assert first.startswith(".version"), first
        version = float(first.split()[1])
        assert version >= 7.8, "sm_89 needs PTX ISA 7.8 or later"
        assert re.search(r"^\.target\s+sm_89\s*$", ptx, re.M), "the problems are scheduled for sm_89"
        assert re.search(r"^\.address_size\s+64\s*$", ptx, re.M)

    def test_entry_and_params_follow_meta(self, compiler: str) -> None:
        for pdir in all_problems([OPEN]):
            meta = json.loads((pdir / "meta.json").read_text())
            for k in meta["kernels"]:
                r = compile_ir(compiler, pdir / f"{k['entry']}.ir.json")
                assert r.returncode == 0, f"{pdir.name}: {r.stderr}"
                entry, params = signature(r.stdout)
                assert entry == k["entry"], pdir.name
                assert len(params) == len(k["arg_order"]), pdir.name
                for p, name in zip(params, k["arg_order"]):
                    assert p.endswith(name), f"{pdir.name}: param {p!r} is not for arg {name!r}"

    def test_deterministic(self, compiler: str) -> None:
        a = compile_ir(compiler, ir_of(OPEN / "f02_row_softmax")).stdout
        b = compile_ir(compiler, ir_of(OPEN / "f02_row_softmax")).stdout
        assert a == b

    def test_compiles_in_seconds(self, compiler: str) -> None:
        for pdir in all_problems([OPEN]):
            t0 = time.perf_counter()
            r = compile_ir(compiler, ir_of(pdir))
            dt = time.perf_counter() - t0
            assert r.returncode == 0, r.stderr
            assert dt < 5.0, f"{pdir.name} took {dt:.1f} s"


@pytest.mark.skipif(not P00.exists(), reason="the open set is not checked out next to this repo")
class TestArgOrder:
    """meta.json's arg_order is the contract: a permuted order must be honoured, not the IR's inputs."""

    @staticmethod
    def permuted_copy(src: Path, dst: Path, order: list[str]) -> Path:
        shutil.copytree(src, dst)
        meta = json.loads((dst / "meta.json").read_text())
        for k in meta["kernels"]:
            by_name = {a["name"]: a for a in k["args"]}
            assert set(order) == set(k["arg_order"])
            k["arg_order"] = list(order)
            k["args"] = [by_name[n] for n in order]  # the runner passes args positionally, as the hosted one does
        (dst / "meta.json").write_text(json.dumps(meta, indent=1))
        return dst

    def test_outputs_first(self, compiler: str, gpu: Gpu, tmp_path: Path) -> None:
        pdir = self.permuted_copy(P00, tmp_path / "p00_outputs_first", ["add", "x1", "x0"])
        r = compile_ir(compiler, ir_of(pdir))
        assert r.returncode == 0, r.stderr
        _, params = signature(r.stdout)
        assert [p.split("_", 1)[1] for p in params] == ["add", "x1", "x0"]
        res = run_problem(pdir, compiler=compiler, gpu=gpu)
        assert res["ok"], json.dumps(res, indent=1, default=str)

    def test_constants_last(self, compiler: str, gpu: Gpu, tmp_path: Path) -> None:
        src = OPEN / "p02_scale_shift"
        pdir = self.permuted_copy(src, tmp_path / "p02_constants_last", ["x", "add", "add_c1", "mul_c1"])
        res = run_problem(pdir, compiler=compiler, gpu=gpu)
        assert res["ok"], json.dumps(res, indent=1, default=str)
