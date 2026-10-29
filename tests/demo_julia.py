#!/usr/bin/env python3
"""A Julia set drawn by ptxc: emmy unrolls 32 steps of z = z^2 + c into one pointwise kernel, ptxc compiles it.

    python tests/demo_julia.py                  # julia.png in the current directory, and a preview here
    python tests/demo_julia.py --out pic.png

The kernel is tests/problems/p99_julia (the spec is in expressions_extra.py; regenerate with
`python tests/generate_problems.py --only p99_julia --force`, about 80 s). Its two inputs are the
coordinate grids: the test fills them with seeded noise, this script with the complex plane.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from expressions_extra import julia
from gpu_runner import CONTRACT_ARCH, DEFAULT_COMPILER, Gpu, assemble, compile_cuda, get_ptx, launch

PROBLEM = Path(__file__).resolve().parent / "problems" / "p99_julia"
COLOURS = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                    [245, 125, 21], [250, 193, 39], [252, 255, 164]], float)  # dark to light, like inferno
RAMP = " .:-=+*#%@"


def write_png(path: Path, rgb: np.ndarray) -> None:
    """8-bit RGB PNG with the standard library: a filter byte per row, zlib, a CRC per chunk."""
    h, w, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + row.tobytes() for row in rgb)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def shade(n: np.ndarray, steps: int) -> np.ndarray:
    """Steps survived -> colour; points that never escape are black."""
    t = np.where(n >= steps, 0.0, n / steps) * (len(COLOURS) - 1)
    i = np.minimum(t.astype(int), len(COLOURS) - 2)
    f = (t - i)[..., None]
    return (COLOURS[i] * (1 - f) + COLOURS[i + 1] * f).astype(np.uint8)


def preview(n: np.ndarray, steps: int, cols: int = 96) -> str:
    """The picture in characters, a block mean per cell; a terminal cell is about twice as tall as wide."""
    h, w = n.shape
    rows = round(cols * h / w / 2)
    cells = n[: h // rows * rows, : w // cols * cols].reshape(rows, h // rows, cols, w // cols).mean((1, 3))
    idx = np.minimum((cells / steps * len(RAMP)).astype(int), len(RAMP) - 1)
    return "\n".join("".join(RAMP[i] for i in row) for row in idx)


def opcodes(ptx: str) -> Counter:
    return Counter(m.group(1) for m in re.finditer(r"^\s+(?:@!?%p\d+\s+)?([a-z][\w.]*)[^;]*;", ptx, re.M))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("julia.png"), help="where to write the picture (default: julia.png)")
    args = ap.parse_args()

    meta = json.loads((PROBLEM / "meta.json").read_text())
    k = meta["kernels"][0]
    out = meta["graph_outputs"][0]
    h, w = next(a["shape"] for a in k["args"] if a["name"] == "re")
    steps = 32  # julia()'s default, as in the torch code
    print(f"torch, traced by emmy into one kernel, {k['entry']} ({PROBLEM.name}):\n")
    print("    " + meta["code"].replace("\n", "\n    ") + "\n")

    t0 = time.perf_counter()
    ptx, _ = get_ptx(PROBLEM, k, str(DEFAULT_COMPILER), None, False, 120.0)
    seconds = time.perf_counter() - t0
    ops = opcodes(ptx)
    per_thread = h * w // (k["grid"][0] * k["block"][0])
    arith = sum(c for op, c in ops.items() if op.split(".")[0] in ("mul", "add", "sub", "fma") and op.endswith(".f32"))
    compares = sum(c for op, c in ops.items() if op.startswith("setp") and op.endswith(".f32"))
    print(f"ptxc   compile.py: {len(ptx.splitlines())} lines of PTX in {seconds:.2f} s; each thread draws {per_thread} pixels,")
    print(f"       {arith / per_thread:.0f} f32 mul / add / sub and {compares / per_thread:.0f} compares per pixel")

    gpu = Gpu()
    gpu.ensure()
    asm = assemble(ptx, gpu.arch)
    plane = {"re": np.broadcast_to(np.linspace(-1.6, 1.6, w, dtype=np.float32), (h, w)).copy(),
             "im": np.broadcast_to(np.linspace(0.9, -0.9, h, dtype=np.float32)[:, None], (h, w)).copy()}
    spec = dict(k, args=[dict(a, intermediate=True) if a["name"] in plane else a for a in k["args"]])  # read from `plane`
    ran = launch(gpu, asm.cubin, spec, plane, warmup=3, iters=20)
    n = ran.outputs[out].reshape(h, w)
    ref = julia(plane["re"], plane["im"])
    print(f"GPU    {gpu.name} ({gpu.arch}): {h}x{w} pixels in {ran.median_us:.1f} us, {asm.registers[k['entry']]} registers")
    print(f"check  numpy f32, same steps: {int((n != ref).sum())} of {n.size} pixels differ")

    emmy_ptx = compile_cuda((PROBLEM / f"{k['entry']}.ref.cu").read_text(), CONTRACT_ARCH)
    emmy = launch(gpu, assemble(emmy_ptx, gpu.arch).cubin, spec, plane, warmup=3, iters=20)
    off = int((emmy.outputs[out].reshape(h, w) != ref).sum())
    print(f"emmy   its own CUDA (fma contraction on): {emmy.median_us:.1f} us, {off} pixels differ from numpy")

    write_png(args.out, shade(n, steps))
    print(f"\n{preview(n, steps)}\n\npicture: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
