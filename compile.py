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
    ap = argparse.ArgumentParser(prog="compile.py",
                                 description="Emmy Kernel IR -> PTX on stdout. meta.json is read from next to the IR.",
                                 epilog="example: python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx")
    ap.add_argument("ir", help="the kernel's .ir.json")
    ap.add_argument("--unroll", type=unroll, default="auto", metavar="auto|none|force|QUOTA",
                    help="loop unrolling: the IR's hints and the compiler's policy (default), no loop unrolled, "
                         "every loop with a known trip count unrolled in full, or the policy under a quota of "
                         "QUOTA statements per loop nest (default 2048)")
    ap.add_argument("--precision", choices=["auto", "exact", "fast"], default="auto", metavar="auto|exact|fast",
                    help="float arithmetic: the cheapest spellings whose error fits meta.json's rtol / atol (default), "
                         "IEEE rounding everywhere (no fma contraction), or fma contraction, approximate div / rcp / sqrt "
                         "and flush-to-zero whatever the tolerance")
    ap.add_argument("--raw-trig", action="store_true",
                    help="sin / cos arguments stay within 100 pi: sin.approx may skip the argument reduction")
    args = ap.parse_args(argv[1:])
    unroll_option = {"unroll_quota": args.unroll} if isinstance(args.unroll, int) else {"unroll": args.unroll}
    options = Options(precision=args.precision, raw_trig=args.raw_trig, **unroll_option)
    try:
        sys.stdout.write(compile_file(args.ir, options))
    except CompileError as err:
        print(f"ptxc: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
