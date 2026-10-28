import argparse
import sys

from ptxc import CompileError, Options, compile_file


def unroll(value: str) -> str | int:
    """`auto`, `none`, `force`, or a quota in statements for the auto policy."""
    if value in ("auto", "none", "force"):
        return value
    if value.isdigit():
        return int(value)
    raise argparse.ArgumentTypeError(f"{value!r}: expected auto, none, force or a number")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="compile.py", usage="python compile.py path/to/kernel.ir.json [--unroll MODE] > out.ptx",
                                 description="Emmy Kernel IR -> PTX. meta.json is read from next to the IR.")
    ap.add_argument("ir", help="the kernel's .ir.json")
    ap.add_argument("--unroll", type=unroll, default="auto", metavar="auto|none|force|QUOTA",
                    help="loop unrolling: the IR's hints and the compiler's policy (default), no loop unrolled, "
                         "every loop with a known trip count unrolled in full, or the policy under a quota of "
                         "QUOTA statements per loop nest (default 2048)")
    args = ap.parse_args(argv[1:])
    options = Options(unroll_quota=args.unroll) if isinstance(args.unroll, int) else Options(unroll=args.unroll)
    try:
        sys.stdout.write(compile_file(args.ir, options))
    except CompileError as err:
        print(f"ptxc: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
