import sys

from ptxc import CompileError, compile_file


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(compile_file(argv[1]))
    except CompileError as err:
        print(f"ptxc: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
