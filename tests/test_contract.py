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

from ptxc.errors import InvalidKernel, Unsupported
from ptxc.ir import Kernel

OPEN = PROBLEM_DIRS[0]
P00 = OPEN / "p00_scalar_add"
TINY = PROBLEM_DIRS[1] / "p23_tiny"  # one pointwise kernel, k_add_pointwise, checked in with this repo


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
                assert f".maxntid {k['block'][0]}, 1, 1" in r.stdout, f"{pdir.name}: the launch bound is meta.json's block"

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


class TestInput:
    """Whatever is wrong with the two input files is one InvalidKernel line, never a traceback; valid
    IR ptxc does not lower is Unsupported instead. Both files are read as UTF-8 whatever the locale."""

    @pytest.fixture
    def ir(self, tmp_path: Path) -> Path:
        shutil.copytree(TINY, tmp_path / "p")
        return next((tmp_path / "p").glob("*.ir.json"))

    @staticmethod
    def meta(ir: Path, edit) -> None:
        path = ir.with_name("meta.json")
        meta = json.loads(path.read_text(encoding="utf-8"))
        edit(meta["kernels"][0])
        path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def one_line(ir: Path) -> str:
        with pytest.raises(InvalidKernel) as info:
            Kernel.load(ir)
        assert "\n" not in str(info.value)
        return str(info.value)

    @pytest.mark.parametrize("text", ["{}", '{"nodes": 1}', '{"inputs": ["x0"', "[]"], ids=["empty", "no-nodes", "truncated", "list"])
    def test_broken_json(self, ir: Path, text: str) -> None:
        ir.write_text(text)
        assert ir.name in self.one_line(ir)

    def test_not_utf8(self, ir: Path) -> None:
        ir.write_bytes(b'{"x": "\xff"}')
        assert ir.name in self.one_line(ir)

    @pytest.mark.parametrize(("old", "new"), [("Tile(", "Tyle("), ("Load(", "Load(zzz=1, ")], ids=["unknown-statement", "unknown-field"])
    def test_broken_body(self, ir: Path, old: str, new: str) -> None:
        ir.write_text(ir.read_text().replace(old, new, 1))
        assert ir.name in self.one_line(ir)

    def test_no_meta(self, ir: Path) -> None:
        ir.with_name("meta.json").unlink()
        assert "meta.json" in self.one_line(ir)

    def test_meta_without_the_entry(self, ir: Path) -> None:
        self.meta(ir, lambda k: k.update(entry="k_other"))
        assert "k_add_pointwise" in self.one_line(ir)

    def test_meta_names_a_missing_arg(self, ir: Path) -> None:
        self.meta(ir, lambda k: k["arg_order"].append("ghost"))
        assert "ghost" in self.one_line(ir)

    def test_unsupported_is_not_malformed(self, ir: Path) -> None:
        self.meta(ir, lambda k: k["args"][0].update(dtype="f16"))
        with pytest.raises(Unsupported, match="f16"):
            Kernel.load(ir)

    def test_scalar_parameter_is_unsupported_by_name(self, ir: Path) -> None:
        """A `"scalar": true` argument (the runner's by-value form, which only serial axes use) is refused with
        its name, not reported as a malformed meta.json."""
        self.meta(ir, lambda k: k["args"][0].update(scalar=True, value=1.0))
        with pytest.raises(Unsupported, match=r"scalar parameters \['x"):
            Kernel.load(ir)

    def test_reads_utf8(self, ir: Path) -> None:
        self.meta(ir, lambda k: k.update(code="x \u2218 \u03c3(y)"))
        assert Kernel.load(ir).entry == "k_add_pointwise"

    def test_cli_reports_one_line(self, compiler: str, tmp_path: Path) -> None:
        bad = tmp_path / "k.ir.json"
        bad.write_text("{}")
        r = compile_ir(compiler, bad)
        assert r.returncode == 1 and r.stdout == ""
        assert r.stderr.startswith("ptxc: InvalidKernel: k.ir.json") and r.stderr.count("\n") == 1
