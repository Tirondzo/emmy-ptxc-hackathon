#!/usr/bin/env python3
"""Generate tests/problems from tests/expressions_extra.py with the hackathon's tools/generate.py.

    python tests/generate_problems.py              # every spec whose folder does not exist yet
    python tests/generate_problems.py --only f25_sigmoid_norm
    python tests/generate_problems.py --force      # regenerate everything

Finds `emmy` in the running interpreter's environment and sets PYTHONUTF8=1 for it (its help text
is not cp1252-encodable on Windows). About 7 s per expression.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent.parent / "emmy-hackathon-2026" / "tools"
OUT = HERE / "problems"
SPECS = HERE / "expressions_extra.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--emmy", default=shutil.which("emmy", path=str(Path(sys.executable).parent)) or "emmy")
    ap.add_argument("--target", default="sm_89")
    args = ap.parse_args()

    os.environ["PYTHONUTF8"] = "1"
    gen = load(TOOLS / "generate.py", "generate")
    specs = load(SPECS, "expressions_extra").OPEN
    if args.only:
        specs = [s for s in specs if s["id"] == args.only]
    todo = [s for s in specs if args.force or not (OUT / s["id"] / "meta.json").exists()]
    print(f"{len(todo)} of {len(specs)} specs to generate into {OUT}")
    failed = []
    for s in todo:
        with tempfile.TemporaryDirectory() as td:
            dump = Path(td) / "dump"
            try:
                gen.trace(args.emmy, s, dump, args.target)
                row = gen.package(s, dump, OUT / s["id"])
                print(f"  OK    {s['id']:<24} {', '.join(row['entries'])}")
            except gen.subprocess.CalledProcessError as e:
                failed.append(s["id"])
                tail = (e.stderr or "").strip().splitlines()[-1:] or ["?"]
                print(f"  FAIL  {s['id']:<24} trace: {tail[0][:120]}")
            except Exception as e:  # noqa: BLE001
                failed.append(s["id"])
                print(f"  FAIL  {s['id']:<24} package: {e}")
    print(f"{len(todo) - len(failed)} ok, {len(failed)} failed" + (f": {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
