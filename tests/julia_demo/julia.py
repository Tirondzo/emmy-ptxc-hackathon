#!/usr/bin/env python3
"""The Julia-set demo's two ends: the complex plane as the kernel's inputs, and the picture from its output.

    python julia.py plane plane.npz     # re, im: the kernel's inputs, a grid over the complex plane
    python julia.py draw julia.npz      # its output -> julia.png, a preview here, a check against numpy

The kernel is emmy's trace of a 32-step loop of z = z^2 + c (tests/problems/p99_julia, spec in
tests/expressions_extra.py). The steps in between, ptxc, ptxas and the launch, are in README.md here.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
from expressions_extra import julia  # noqa: E402  the numpy reference, next to the torch code it mirrors

PROBLEM = TESTS / "problems" / "p99_julia"
STEPS = 32  # julia()'s default, as in the torch code
COLOURS = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                    [245, 125, 21], [250, 193, 39], [252, 255, 164]], float)  # dark to light, like inferno
RAMP = " .:-=+*#%@"


def plane() -> dict[str, np.ndarray]:
    """x from -1.6 to 1.6 across, y from 0.9 down to -0.9, in the shape the kernel was traced for."""
    k = json.loads((PROBLEM / "meta.json").read_text())["kernels"][0]
    h, w = next(a["shape"] for a in k["args"] if a["name"] == "re")
    return {"re": np.broadcast_to(np.linspace(-1.6, 1.6, w, dtype=np.float32), (h, w)).copy(),
            "im": np.broadcast_to(np.linspace(0.9, -0.9, h, dtype=np.float32)[:, None], (h, w)).copy()}


def write_png(path: Path, rgb: np.ndarray) -> None:
    """8-bit RGB PNG with the standard library: a filter byte per row, zlib, a CRC per chunk."""
    h, w, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + row.tobytes() for row in rgb)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def shade(n: np.ndarray) -> np.ndarray:
    """Steps survived -> colour; points that never escape are black."""
    t = np.where(n >= STEPS, 0.0, n / STEPS) * (len(COLOURS) - 1)
    i = np.minimum(t.astype(int), len(COLOURS) - 2)
    f = (t - i)[..., None]
    return (COLOURS[i] * (1 - f) + COLOURS[i + 1] * f).astype(np.uint8)


def preview(n: np.ndarray, cols: int = 96) -> str:
    """The picture in characters, a block mean per cell; a terminal cell is about twice as tall as wide."""
    h, w = n.shape
    rows = round(cols * h / w / 2)
    cells = n[: h // rows * rows, : w // cols * cols].reshape(rows, h // rows, cols, w // cols).mean((1, 3))
    idx = np.minimum((cells / STEPS * len(RAMP)).astype(int), len(RAMP) - 1)
    return "\n".join("".join(RAMP[i] for i in row) for row in idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("plane", help="write the kernel's inputs")
    p.add_argument("npz", type=Path)
    d = sub.add_parser("draw", help="turn the kernel's output into julia.png")
    d.add_argument("npz", type=Path, help="tests/launch.py --out")
    d.add_argument("--out", type=Path, default=Path("julia.png"), help="where to write the picture (default: julia.png)")
    args = ap.parse_args()

    xy = plane()
    if args.step == "plane":
        np.savez(args.npz, **xy)
        print(f"re, im: {xy['re'].shape[0]}x{xy['re'].shape[1]} f32 saved to {args.npz}")
        return 0
    (n,) = np.load(args.npz).values()
    differ = int((n != julia(xy["re"], xy["im"])).sum())
    write_png(args.out, shade(n))
    print(f"{preview(n)}\n\ncheck  numpy f32, same steps: {differ} of {n.size} pixels differ\npicture: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
