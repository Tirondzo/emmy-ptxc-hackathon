#!/usr/bin/env python3
"""Launch a cubin's kernel the way meta.json describes it and save the outputs: the host side, by hand.

    python launch.py julia.cubin ../problems/p99_julia/meta.json --inputs plane.npz --out julia.npz

Parameters follow meta.json's `arg_order`: inputs from --inputs by name (the rest seeded, as the runner
makes them), constants from their `data`, outputs zeroed; grid and block from meta.json. Prints the
median of 20 timed launches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_runner import Gpu, launch  # noqa: E402  tests/gpu_runner.py: the driver API, sentinels around every buffer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cubin", type=Path, help="ptxas output for this GPU")
    ap.add_argument("meta", type=Path, help="the problem's meta.json: entry, grid, block, arg_order, constants")
    ap.add_argument("--inputs", type=Path, help=".npz of input arrays by name (default: the seeded inputs)")
    ap.add_argument("--out", type=Path, default=Path("out.npz"), help="where to save the outputs (default: out.npz)")
    args = ap.parse_args()

    kernels = json.loads(args.meta.read_text())["kernels"]
    if len(kernels) != 1:
        ap.error(f"{args.meta} lists {len(kernels)} kernels; this launches one")
    (k,) = kernels
    given = dict(np.load(args.inputs)) if args.inputs else {}
    # `intermediate` is the runner's form for an array handed over by name instead of generated
    spec = dict(k, args=[dict(a, intermediate=True) if a["name"] in given else a for a in k["args"]])
    gpu = Gpu()
    gpu.ensure()
    ran = launch(gpu, args.cubin.read_bytes(), spec, given, warmup=3, iters=20)
    if ran.guard_violations or ran.clobbered_inputs:
        print(f"warning: writes out of bounds {ran.guard_violations}, inputs modified {ran.clobbered_inputs}")
    np.savez(args.out, **ran.outputs)
    print(f"{k['entry']} on {gpu.name}: grid {k['grid'][0]} x block {k['block'][0]}, {ran.median_us:.1f} us; "
          f"{', '.join(ran.outputs)} saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
