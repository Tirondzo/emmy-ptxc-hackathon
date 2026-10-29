"""Time ptxc's kernels against emmy's reference under several `Options`, on the local GPU.

    python tests/bench.py m01_matmul_512 r02_big_sum          # the three unroll modes and the reference
    python tests/bench.py --all                                # the open set
    python tests/bench.py a01_attention -c x4=unroll_auto=4 -c q512=unroll_quota=512 -c raw=cse=0

A configuration is `name=field=value,...` over the fields of `Options`; `name=` is the defaults.
A kernel's time is the minimum over `--repeats` runs of the median of `--iters` launches, since
noise on a shared GPU only ever makes a run slower.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_runner import PROBLEM_DIRS, Gpu, all_problems, find_problem, run_problem  # noqa: E402

from ptxc import Options, compile_file  # noqa: E402
from ptxc.errors import CompileError  # noqa: E402

DEFAULT_CONFIGS = ["auto=", "none=unroll=none", "force=unroll=force", "exact=precision=exact"]
TYPES = {f.name: f.type for f in fields(Options)}


def parse_config(text: str) -> tuple[str, Options]:
    name, _, assignments = text.partition("=")
    values: dict[str, object] = {}
    for a in filter(None, assignments.split(",")):
        field, _, v = a.partition("=")
        if field not in TYPES:
            sys.exit(f"{field!r} is not an Options field; choose from {', '.join(TYPES)}")
        match TYPES[field]:
            case "int":
                values[field] = int(v)
            case "bool":
                values[field] = v.lower() in ("1", "true", "on", "yes")
            case "tuple[str, ...]":
                values[field] = tuple(v.split("+"))
            case _:
                values[field] = v
    return name, Options(**values)  # type: ignore[arg-type]


def measure(pdir: Path, gpu: Gpu, options: Options | None, repeats: int, iters: int) -> dict:
    """Minimum over repeats of the per-launch median, summed over the problem's kernels; `options`
    None times emmy's reference kernel instead."""
    compiler = None if options is None else (lambda ir: compile_file(ir, options))
    best: dict = {"us": float("inf")}
    for _ in range(repeats):
        try:
            res = run_problem(pdir, compiler=compiler, ref=options is None, gpu=gpu, warmup=5, iters=iters)
        except CompileError as e:  # the in-process compiler raises what compile.py prints as one line
            return {"error": str(e)[:60]}
        if not res["ok"]:
            err = next((k.get("error", {}).get("message") for k in res["kernels"] if not k["ok"]), res.get("error"))
            return {"error": str(err or "outputs differ")[:60]}
        us = sum(k["median_us"] for k in res["kernels"])
        if us < best["us"]:
            spills = [k.get("spills") or (0, 0) for k in res["kernels"]]
            best = {"us": us, "regs": max(k.get("registers") or 0 for k in res["kernels"]),
                    "spill": sum(s[0] for s in spills), "ptx_chars": sum(k["ptx_chars"] for k in res["kernels"])}
    return best


def cell(m: dict) -> str:
    if "error" in m:
        return f"FAIL {m['error']}"
    spill = f" {m['spill']}B" if m["spill"] else ""
    return f"{m['us']:.1f} us ({m['regs']}r{spill})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="*", help="problem ids or folders")
    ap.add_argument("--all", action="store_true", help="every problem of the open set")
    ap.add_argument("--problems-dir", action="append", default=[], metavar="DIR", help="extra folders to resolve ids in")
    ap.add_argument("-c", "--config", action="append", metavar="NAME=FIELD=VALUE,...", help=f"default: {' '.join(DEFAULT_CONFIGS)}")
    ap.add_argument("--no-ref", action="store_true", help="skip emmy's reference kernel")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    roots = [Path(p) for p in args.problems_dir] + list(PROBLEM_DIRS)
    dirs = all_problems([PROBLEM_DIRS[0]]) if args.all else [find_problem(p, roots) for p in args.problems]
    if not dirs:
        ap.error("name a problem or pass --all")
    configs = [parse_config(c) for c in (args.config or DEFAULT_CONFIGS)]
    gpu = Gpu()
    columns = [name for name, _ in configs] + ([] if args.no_ref else ["reference"])
    print(f"GPU: {gpu.name} ({gpu.arch}); min over {args.repeats} runs of the median of {args.iters} launches\n")
    print("| problem | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for pdir in dirs:
        cells = [cell(measure(pdir, gpu, options, args.repeats, args.iters)) for _, options in configs]
        if not args.no_ref:
            cells.append(cell(measure(pdir, gpu, None, args.repeats, args.iters)))
        print(f"| {pdir.name} | " + " | ".join(cells) + " |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
