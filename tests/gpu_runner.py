#!/usr/bin/env python3
"""A local stand-in for the hosted runner: assemble PTX with ptxas, launch it on this
machine's GPU through the CUDA driver API, and compare the outputs with expected.npz.

    python tests/gpu_runner.py r01_array_sum --compiler compile.py     # like submit.py --compiler
    python tests/gpu_runner.py r01_array_sum path/to/out.ptx            # a fixed PTX file
    python tests/gpu_runner.py r01_array_sum --ref                      # the problem's .ref.cu through nvrtc
    python tests/gpu_runner.py --all [DIR ...] --compiler compile.py    # every problem under DIR(s)

The pipeline is the one submit.py drives remotely: PTX -> ptxas (for the contract arch sm_89
*and* this GPU's arch) -> cubin -> one launch under meta.json's grid/block/smem_bytes with the
args meta.json describes (seeded inputs, constant data, zeroed outputs, intermediates chained
from earlier kernels) -> np.allclose against expected.npz within the problem's tolerance.

On top of what the hosted runner reports, every buffer is fenced with sentinel bytes, so an
out-of-bounds write is detected, and inputs are checked to be unchanged after the launch.

Requirements (all pip): numpy, cuda-python, nvidia-cuda-nvcc (ships ptxas), nvidia-cuda-nvrtc
(only for --ref). `PTXAS=path` overrides the ptxas lookup.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_COMPILER = REPO / "compile.py"
PROBLEM_DIRS = [REPO.parent / "emmy-hackathon-2026" / "problems", HERE / "problems"]

DT = {"f32": np.float32, "f16": np.float16, "f64": np.float64, "i32": np.int32, "i64": np.int64, "bool": np.bool_}
GUARD = 256  # sentinel bytes before and after every buffer; keeps data 256-byte aligned like cudaMalloc
SENTINEL = 0xA5
CONTRACT_ARCH = "sm_89"  # the arch every submission must assemble for
STATIC_SMEM_LIMIT = 48 * 1024


class RunError(Exception):
    """One stage failed. `stage` mirrors the hosted runner: compile, ptx, ptxas, entry, run."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


# ---- ptxas ---------------------------------------------------------------------------


def find_ptxas() -> str:
    if p := os.environ.get("PTXAS"):
        return p
    if p := shutil.which("ptxas"):
        return p
    hits = sorted(Path(sys.prefix).glob("Lib/site-packages/nvidia/cu*/bin/ptxas*"))
    hits += sorted(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cu*/bin/ptxas"))
    if hits:
        return str(hits[-1])
    raise RunError("ptxas", "ptxas not found: pip install nvidia-cuda-nvcc, or set PTXAS=/path/to/ptxas")


@dataclass
class Assembled:
    arch: str
    cubin: bytes
    registers: dict[str, int]  # entry -> registers
    spills: dict[str, tuple[int, int]]  # entry -> (spill store bytes, spill load bytes)
    warnings: list[str]


def assemble(ptx: str, arch: str, flags: tuple[str, ...] = ()) -> Assembled:
    if not ptx.isascii():
        raise RunError("ptxas", "PTX contains non-ASCII text")
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "k.ptx", Path(td) / "k.cubin"
        src.write_text(ptx, encoding="ascii")
        r = subprocess.run(
            [find_ptxas(), f"-arch={arch}", "-v", *flags, "-o", str(out), str(src)],
            capture_output=True, text=True,
        )
        text = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            raise RunError("ptxas", text.strip())
        regs, spills, warnings = parse_ptxas_info(text)
        return Assembled(arch, out.read_bytes(), regs, spills, warnings)


def parse_ptxas_info(text: str) -> tuple[dict[str, int], dict[str, tuple[int, int]], list[str]]:
    regs: dict[str, int] = {}
    spills: dict[str, tuple[int, int]] = {}
    warnings: list[str] = []
    entry = "?"
    for line in text.splitlines():
        if m := re.search(r"Compiling entry function '(\w+)'", line):
            entry = m.group(1)
        elif m := re.search(r"Used (\d+) registers", line):
            regs[entry] = int(m.group(1))
        elif m := re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line):
            spills[entry] = (int(m.group(1)), int(m.group(2)))
        elif "warning" in line.lower():
            warnings.append(line.strip())
    return regs, spills, warnings


def not_ptx(ptx: str) -> str | None:
    """submit.py's pre-check: why this cannot be PTX, or None."""
    lines = [l.strip() for l in ptx.splitlines() if l.strip() and not l.strip().startswith(("//", "/*", "*"))]
    if not lines:
        return "it is empty"
    if not lines[0].startswith("."):
        return f"it starts with {lines[0][:60]!r} instead of a directive like .version"
    return None


# ---- nvrtc, for the reference .cu ------------------------------------------------------


def compile_cuda(source: str, arch: str) -> str:
    """CUDA C++ -> PTX through nvrtc; `arch` like sm_89 (compiled as compute_89)."""
    from cuda.bindings import nvrtc

    def check(res):
        err, *rest = res
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RunError("nvrtc", nvrtc.nvrtcGetErrorString(err)[1].decode())
        return rest[0] if len(rest) == 1 else rest

    prog = check(nvrtc.nvrtcCreateProgram(source.encode(), b"ref.cu", 0, [], []))
    opts = [f"--gpu-architecture=compute_{arch.split('_')[1]}".encode(), b"-std=c++17"]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    size = check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RunError("nvrtc", log.decode(errors="replace").strip())
    size = check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * size
    check(nvrtc.nvrtcGetPTX(prog, ptx))
    return ptx.decode().rstrip("\x00")


# ---- the GPU --------------------------------------------------------------------------


class Gpu:
    def __init__(self, device: int = 0) -> None:
        from cuda.bindings import driver as cu  # lazy: --help works without a GPU

        self.cu = cu
        self.device = device
        self.poisoned = False  # a sticky error (illegal address, timeout) killed the context
        self.check(cu.cuInit(0))
        self.dev = self.check(cu.cuDeviceGet(device))
        self.name = self.check(cu.cuDeviceGetName(256, self.dev)).split(b"\x00")[0].decode(errors="replace")
        A = cu.CUdevice_attribute
        major = self.check(cu.cuDeviceGetAttribute(A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.dev))
        minor = self.check(cu.cuDeviceGetAttribute(A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.dev))
        self.arch = f"sm_{major}{minor}"
        self.smem_optin = self.check(cu.cuDeviceGetAttribute(A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, self.dev))
        self.ensure()

    def ensure(self) -> None:
        """A usable context: the primary one, reset first if an earlier launch poisoned it."""
        if self.poisoned:
            self.cu.cuDevicePrimaryCtxReset(self.dev)
            self.poisoned = False
        self.ctx = self.check(self.cu.cuDevicePrimaryCtxRetain(self.dev))
        self.check(self.cu.cuCtxSetCurrent(self.ctx))

    def check(self, result):
        err, *rest = result
        if err != self.cu.CUresult.CUDA_SUCCESS:
            name = self.cu.cuGetErrorName(err)[1].decode()
            desc = self.cu.cuGetErrorString(err)[1].decode()
            if err not in (self.cu.CUresult.CUDA_ERROR_INVALID_VALUE, self.cu.CUresult.CUDA_ERROR_OUT_OF_MEMORY,
                           self.cu.CUresult.CUDA_ERROR_INVALID_IMAGE, self.cu.CUresult.CUDA_ERROR_NOT_FOUND,
                           self.cu.CUresult.CUDA_ERROR_INVALID_PTX, self.cu.CUresult.CUDA_ERROR_NO_BINARY_FOR_GPU):
                self.poisoned = True  # everything else is treated as sticky; the next ensure() resets
            raise RunError("run", f"{name}: {desc}")
        return rest[0] if len(rest) == 1 else (tuple(rest) or None)

    def load(self, image: bytes):
        buf = ctypes.create_string_buffer(image, len(image) + 1)  # NUL-terminated, for PTX text too
        return self.check(self.cu.cuModuleLoadData(ctypes.addressof(buf)))

    def function(self, module, entry: str):
        return self.check(self.cu.cuModuleGetFunction(module, entry.encode()))


@dataclass
class DevBuf:
    name: str
    base: object  # CUdeviceptr of the whole allocation
    ptr: int  # where the data starts: base + GUARD
    host: np.ndarray  # what was uploaded (outputs: zeros)
    is_output: bool

    @property
    def nbytes(self) -> int:
        return self.host.nbytes

    @classmethod
    def upload(cls, gpu: Gpu, name: str, host: np.ndarray, is_output: bool) -> DevBuf:
        host = np.ascontiguousarray(host)
        total = host.nbytes + 2 * GUARD
        base = gpu.check(gpu.cu.cuMemAlloc(total))
        gpu.check(gpu.cu.cuMemsetD8(base, SENTINEL, total))
        ptr = int(base) + GUARD
        if host.nbytes:
            gpu.check(gpu.cu.cuMemcpyHtoD(gpu.cu.CUdeviceptr(ptr), host.ctypes.data, host.nbytes))
        return cls(name, base, ptr, host, is_output)

    def download(self, gpu: Gpu) -> tuple[np.ndarray, int, int]:
        """(data, bytes of the front guard clobbered, bytes of the back guard clobbered)."""
        total = self.nbytes + 2 * GUARD
        whole = np.empty(total, np.uint8)
        gpu.check(gpu.cu.cuMemcpyDtoH(whole.ctypes.data, self.base, total))
        front = int((whole[:GUARD] != SENTINEL).sum())
        back = int((whole[GUARD + self.nbytes:] != SENTINEL).sum())
        data = whole[GUARD:GUARD + self.nbytes].view(self.host.dtype).reshape(self.host.shape).copy()
        return data, front, back

    def zero(self, gpu: Gpu) -> None:
        gpu.check(gpu.cu.cuMemsetD8(gpu.cu.CUdeviceptr(self.ptr), 0, self.nbytes))

    def free(self, gpu: Gpu) -> None:
        gpu.cu.cuMemFree(self.base)


def make_input(a: dict, produced: dict[str, np.ndarray]) -> np.ndarray:
    """The hosted runner's argument forms: seed, data, intermediate (chained), output (zeros)."""
    dtype, shape = DT[a["dtype"]], tuple(a["shape"]) or (1,)
    if a.get("output"):
        return np.zeros(shape, dtype)
    if a.get("intermediate"):
        if a["name"] not in produced:
            raise RunError("run", f"needs {a['name']} from an earlier kernel - kernels run in meta.json order")
        return np.asarray(produced[a["name"]], dtype).reshape(shape)
    if "data" in a:
        return np.asarray(a["data"], dtype).reshape(shape)
    if "seed" in a:
        return np.random.default_rng(a["seed"]).standard_normal(shape).astype(dtype)
    raise RunError("run", f"argument {a['name']!r} has no seed, data, output or intermediate form")


@dataclass
class Launched:
    outputs: dict[str, np.ndarray]
    guard_violations: dict[str, tuple[int, int]]  # buffer -> (front bytes, back bytes)
    clobbered_inputs: list[str]
    median_us: float
    times_us: list[float] = field(default_factory=list)
    nondeterministic: list[str] = field(default_factory=list)  # outputs that differed bitwise between repeats


def launch(gpu: Gpu, image: bytes, spec: dict, produced: dict[str, np.ndarray],
           warmup: int = 0, iters: int = 1, repeat: int = 1) -> Launched:
    """Upload the args, launch `warmup + iters` times (timed), read everything back.

    `repeat` > 1 launches again that many times on fresh output buffers and compares the
    outputs bitwise: a race between threads tends to show up as run-to-run differences.
    """
    cu = gpu.cu
    module = gpu.load(image)
    fn = gpu.function(module, spec["entry"])
    bufs: list[DevBuf] = []
    holders: list[np.ndarray] = []  # one small array per parameter; the launch takes pointers to them
    try:
        for a in spec["args"]:
            if a.get("scalar"):
                holders.append(np.array([a["value"]], DT[a["dtype"]]))
                continue
            buf = DevBuf.upload(gpu, a["name"], make_input(a, produced), bool(a.get("output")))
            bufs.append(buf)
            holders.append(np.array([buf.ptr], np.uint64))
        params = np.array([h.ctypes.data for h in holders], np.uint64)
        smem = int(spec.get("smem_bytes", 0))
        if smem > STATIC_SMEM_LIMIT:
            if smem > gpu.smem_optin:
                raise RunError("run", f"smem_bytes {smem} exceeds the per-block limit {gpu.smem_optin}")
            gpu.check(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
        zero = [b for b in bufs if b.name in spec.get("zero_outputs", [])]
        grid, block = spec["grid"], spec["block"]
        stream = cu.CUstream(0)
        start, stop = gpu.check(cu.cuEventCreate(0)), gpu.check(cu.cuEventCreate(0))

        def run_once() -> float:
            for b in zero:
                b.zero(gpu)
            gpu.check(cu.cuEventRecord(start, stream))
            gpu.check(cu.cuLaunchKernel(fn, *grid, *block, smem, stream, params.ctypes.data, 0))
            gpu.check(cu.cuEventRecord(stop, stream))
            gpu.check(cu.cuCtxSynchronize())
            return gpu.check(cu.cuEventElapsedTime(start, stop)) * 1000.0

        times = [run_once() for _ in range(warmup + iters)][warmup:]
        outputs, guards, clobbered = {}, {}, []
        for b in bufs:
            data, front, back = b.download(gpu)
            if front or back:
                guards[b.name] = (front, back)
            if b.is_output:
                outputs[b.name] = data
            elif not np.array_equal(data.view(np.uint8), b.host.view(np.uint8)):
                clobbered.append(b.name)
        nondeterministic: set[str] = set()
        for _ in range(repeat - 1):
            for b in bufs:
                if b.is_output:
                    b.zero(gpu)
            run_once()
            for b in bufs:
                if b.is_output and not np.array_equal(b.download(gpu)[0].view(np.uint8), outputs[b.name].view(np.uint8)):
                    nondeterministic.add(b.name)
        return Launched(outputs, guards, clobbered, float(np.median(times)), times, sorted(nondeterministic))
    finally:
        for b in bufs:
            b.free(gpu)
        cu.cuModuleUnload(module)


# ---- one problem, the way submit.py drives it ---------------------------------------------


def compare(got: np.ndarray, ref: np.ndarray, rtol: float, atol: float) -> dict:
    got = np.asarray(got, dtype=np.float32).reshape(ref.shape)
    g, r = got.astype(np.float64), ref.astype(np.float64)
    diff = np.abs(g - r)
    bad = ~(diff <= atol + rtol * np.abs(r))  # NaN compares false -> bad, like np.allclose
    finite = diff[np.isfinite(diff)]
    out = {
        "ok": not bool(bad.any()),
        "max_abs_err": float(finite.max()) if finite.size else float("nan"),
        "mismatches": int(bad.sum()),
        "size": int(got.size),
        "nan": int(np.isnan(got).sum()),
    }
    if out["mismatches"]:
        i = int(np.argmax(bad.ravel()))
        out["first_bad"] = {"index": i, "got": float(g.ravel()[i]), "expected": float(r.ravel()[i])}
    return out


def get_ptx(pdir: Path, k: dict, compiler: str | None, ptx_file: str | None, ref: bool, timeout: float) -> tuple[str, str]:
    if ref:
        cu_file = pdir / f"{k['entry']}.ref.cu"
        if not cu_file.exists():
            raise RunError("compile", f"no {cu_file.name} (hidden-style problem)")
        return compile_cuda(cu_file.read_text(), CONTRACT_ARCH), "ref.cu via nvrtc"
    if compiler:
        ir = next(pdir.glob(f"{k['entry']}.ir.json"), None)
        if ir is None:
            raise RunError("compile", f"no {k['entry']}.ir.json in {pdir}")
        try:
            r = subprocess.run([sys.executable, compiler, str(ir)], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RunError("compile", f"compiler exceeded {timeout:.0f} s") from None
        if r.returncode != 0:
            raise RunError("compile", f"exit code {r.returncode}:\n{r.stderr.strip() or '(nothing on stderr)'}")
        return r.stdout, f"{Path(compiler).name} {ir.name}"
    if ptx_file:
        return Path(ptx_file).read_text(), ptx_file
    raise RunError("compile", "give a PTX file, --compiler or --ref")


def run_problem(pdir: Path, *, compiler: str | None = None, ptx_file: str | None = None, ref: bool = False,
                gpu: Gpu | None = None, warmup: int = 0, iters: int = 1, repeat: int = 1,
                contract_arch: str | None = CONTRACT_ARCH, compile_timeout: float = 120.0) -> dict:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz")) if (pdir / "expected.npz").exists() else {}
    gpu = gpu or Gpu()
    gpu.ensure()
    produced: dict[str, np.ndarray] = {}
    result = {"problem": meta.get("id", pdir.name), "dir": str(pdir), "gpu": gpu.name, "arch": gpu.arch,
              "rtol": meta["rtol"], "atol": meta["atol"], "ok": True, "kernels": []}
    for k in meta["kernels"]:
        kr: dict = {"entry": k["entry"], "ok": True, "outputs": {}}
        result["kernels"].append(kr)
        try:
            t0 = time.perf_counter()
            ptx, kr["source"] = get_ptx(pdir, k, compiler, ptx_file, ref, compile_timeout)
            kr["compile_s"] = round(time.perf_counter() - t0, 3)
            kr["ptx_chars"] = len(ptx)
            if why := not_ptx(ptx):
                raise RunError("ptx", f"that is not PTX: {why}")
            if contract_arch and contract_arch != gpu.arch:
                kr[f"registers_{contract_arch}"] = assemble(ptx, contract_arch).registers.get(k["entry"])
            asm = assemble(ptx, gpu.arch)
            kr["registers"] = asm.registers.get(k["entry"])
            kr["spills"] = asm.spills.get(k["entry"])
            kr["ptxas_warnings"] = asm.warnings
            if k["entry"] not in asm.registers:
                raise RunError("entry", f"compiled, but entry points are {list(asm.registers)} - the entry must be named {k['entry']!r}")
            launched = launch(gpu, asm.cubin, k, produced, warmup, iters, repeat)
            kr["median_us"] = round(launched.median_us, 2)
            kr["guard_violations"] = launched.guard_violations
            kr["clobbered_inputs"] = launched.clobbered_inputs
            kr["nondeterministic"] = launched.nondeterministic
            if launched.guard_violations or launched.clobbered_inputs:
                kr["ok"] = False
            if launched.nondeterministic and not k.get("zero_outputs"):
                kr["ok"] = False  # float atomics may legitimately differ run to run; anything else may not
            for name, arr in launched.outputs.items():
                produced[name] = arr
                if name in expected:
                    kr["outputs"][name] = c = compare(arr, expected[name], meta["rtol"], meta["atol"])
                    kr["ok"] = kr["ok"] and c["ok"]
        except RunError as e:
            kr["ok"] = False
            kr["error"] = {"stage": e.stage, "message": str(e)}
        result["ok"] = result["ok"] and kr["ok"]
        if "error" in kr:
            break
    missing = [n for n in expected if n not in produced]
    if missing and result["ok"]:
        result["ok"] = False
        result["error"] = f"expected outputs never produced: {missing}"
    result["checked"] = bool(expected)  # False: hidden-style problem, only assembled and launched
    return result


# ---- CLI ---------------------------------------------------------------------------------


def find_problem(arg: str, roots: list[Path]) -> Path:
    p = Path(arg)
    if (p / "meta.json").exists():
        return p.resolve()
    for root in roots:
        if (root / arg / "meta.json").exists():
            return (root / arg).resolve()
    known = sorted(d.name for root in roots if root.exists() for d in root.iterdir() if (d / "meta.json").exists())
    sys.exit(f"unknown problem {arg!r}; choose from: {', '.join(known)}")


def all_problems(roots: list[Path]) -> list[Path]:
    out = []
    for root in roots:
        if (root / "meta.json").exists():
            out.append(root.resolve())
        elif root.exists():
            out += sorted(d.resolve() for d in root.iterdir() if (d / "meta.json").exists())
    return out


def report(res: dict) -> None:
    for kr in res["kernels"]:
        e = kr["entry"]
        if "registers" in kr:
            spill = kr.get("spills") or (0, 0)
            extra = f", {spill[0]}B spill" if spill[0] else ""
            r89 = kr.get("registers_sm_89")
            arch_note = f"sm_89: {r89} regs; " if r89 is not None else ""
            print(f"[{e}] {arch_note}{res['arch']}: {kr['registers']} registers{extra}  (compile {kr.get('compile_s', '?')} s)")
            for w in kr.get("ptxas_warnings", []):
                print(f"[{e}] ptxas: {w}")
        for name, c in kr["outputs"].items():
            tag = "PASS" if c["ok"] else "FAIL"
            line = f"[{e}] {tag}  {name}: max abs err {c['max_abs_err']:.3e}  ({kr.get('median_us', '?')} us)"
            if not c["ok"]:
                line += (f"\n    {c['mismatches']}/{c['size']} outside rtol={res['rtol']} atol={res['atol']}, {c['nan']} NaN;"
                         f" first bad [{c['first_bad']['index']}]: got {c['first_bad']['got']:.6g}, expected {c['first_bad']['expected']:.6g}")
            print(line)
        for name, (front, back) in kr.get("guard_violations", {}).items():
            print(f"[{e}] OUT-OF-BOUNDS WRITE on {name!r}: {front} bytes before, {back} bytes after the buffer")
        for name in kr.get("clobbered_inputs", []):
            print(f"[{e}] INPUT CLOBBERED: {name!r} changed during the launch")
        for name in kr.get("nondeterministic", []):
            print(f"[{e}] NONDETERMINISTIC: {name!r} differed between repeated launches")
        if "error" in kr:
            print(f"[{e}] {kr['error']['stage']} failed: {kr['error']['message']}")
    if "error" in res:
        print(f"[{res['problem']}] {res['error']}")
    if res["ok"] and not res.get("checked", True):
        print(f"[{res['problem']}] ran, but there is no expected.npz to check against")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", nargs="?", help="problem id or folder (omit with --all)")
    ap.add_argument("ptx", nargs="?", help="a PTX file (single-kernel problems)")
    ap.add_argument("--compiler", help=f"your compile.py (default when neither a PTX file nor --ref is given: {DEFAULT_COMPILER})")
    ap.add_argument("--ref", action="store_true", help="run the problem's .ref.cu through nvrtc instead (runner self-check)")
    ap.add_argument("--all", nargs="*", metavar="DIR", help="run every problem under DIR(s) (default: the open set and tests/problems)")
    ap.add_argument("--problems", action="append", default=[], metavar="DIR", help="extra folders to resolve problem ids in")
    ap.add_argument("--iters", type=int, default=1, help="timed launches; the median is reported")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="extra launches whose outputs must match bitwise (race check)")
    ap.add_argument("--no-contract-arch", action="store_true", help=f"skip the extra {CONTRACT_ARCH} assembly")
    ap.add_argument("--json", action="store_true", help="print one JSON document instead of the report")
    args = ap.parse_args()

    roots = [Path(p) for p in args.problems] + PROBLEM_DIRS
    if args.all is not None:
        pdirs = all_problems([Path(p) for p in args.all] or PROBLEM_DIRS)
    elif args.problem:
        pdirs = [find_problem(args.problem, roots)]
    else:
        ap.error("give a problem or --all")
    compiler = args.compiler
    if not compiler and not args.ptx and not args.ref:
        compiler = str(DEFAULT_COMPILER)

    gpu = Gpu()
    results = []
    for pdir in pdirs:
        res = run_problem(pdir, compiler=compiler, ptx_file=args.ptx, ref=args.ref, gpu=gpu,
                          warmup=args.warmup, iters=args.iters, repeat=args.repeat,
                          contract_arch=None if args.no_contract_arch else CONTRACT_ARCH)
        results.append(res)
        if not args.json:
            if len(pdirs) > 1:
                print(f"== {res['problem']}")
            report(res)
    if args.json:
        print(json.dumps(results if args.all is not None else results[0], indent=1, default=str))
    elif len(results) > 1:
        passed = sum(r["ok"] for r in results)
        print(f"\n{passed}/{len(results)} problems pass on {gpu.name} ({gpu.arch})")
        for r in results:
            if not r["ok"]:
                k = next((k for k in r["kernels"] if not k["ok"]), None)
                why = (k or {}).get("error", {}).get("message", "outputs differ").splitlines()[0] if k else r.get("error", "")
                print(f"  FAIL {r['problem']}: {why[:120]}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
