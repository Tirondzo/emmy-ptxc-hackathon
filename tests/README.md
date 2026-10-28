# Testing ptxc

Everything here runs on a local NVIDIA GPU instead of the hackathon's hosted runner, with the
same pipeline: `compile.py` -> `ptxas` -> launch under `meta.json` -> compare with
`expected.npz`. The open set is read from `../emmy-hackathon-2026/problems` (the hackathon
repo checked out next to this one); our own problems live in `tests/problems`.

```
tests/
  gpu_runner.py          the local runner (CLI and library): ptxas + CUDA driver API
  conftest.py            pytest options: --problems DIR, --compiler PATH, --repeat N
  test_problems.py       one test per problem folder, end to end
  test_contract.py       the CLI contract: stdout is ASCII PTX, entry name, arg_order, determinism, speed,
                         and malformed input as one error line
  kernels.py             emmy IR written in Python and lowered in-process, for shapes the problems never produce
  simwarp.py             one warp of the emitted PTX interpreted in Python: the semantics tests without a GPU
  test_lowering.py       lowering decisions read off the PTX text (builtins, dtypes, selects, shuffles, decode)
  test_warp.py           kernel semantics on a simulated warp and on the GPU: shuffles of every length, selects, NaN
  test_unroll.py         the unroll policy and options, the shape each mode emits, every emission path on the GPU
  test_precision.py      the precision policy over rtol / atol, the spelling each mode emits, every mode on the GPU
  bench.py               ptxc under several option sets against emmy's reference: min over repeats, registers, spills
  expressions_extra.py   torch expressions for our own problems, one list per category
  problems/              generated from expressions_extra.py (IR + meta + expected, like the open set)
```

## Setup (Windows, done once)

```
python -m venv .venv                      # at the workspace root, next to both repos
.venv/Scripts/pip install -r emmy-ptxc-hackathon/tests/requirements.txt
```

`tests/requirements.txt` pins everything to the versions FINDINGS.md was produced with: emmy-ml and
numpy, pytest, cuda-python (driver API), nvidia-cuda-nvcc (ptxas) and nvidia-cuda-nvrtc (`--ref`) to
run the tests; torch (cu128) and scipy only to generate problems; requests only for `submit.py`.

Two Windows quirks of the `emmy` CLI (only needed to *generate* problems, not to run tests):

- it imports the Unix-only `fcntl` module; a no-op shim `fcntl.py` dropped into
  `.venv/Lib/site-packages` makes it import (locks are only used by its golden cache);
- its help text is not cp1252-encodable, so run it with `PYTHONUTF8=1`.

## Running

```
python tests/gpu_runner.py r01_array_sum                      # compile.py on one problem
python tests/gpu_runner.py --all                              # the open set and tests/problems
python tests/gpu_runner.py r01_array_sum --ref                # emmy's .ref.cu through nvrtc: runner self-check
python tests/gpu_runner.py m01_matmul_512 --iters 20 --warmup 3 --repeat 3
python -m pytest tests -q                                     # everything
python -m pytest tests -q -k "p4 or a1"                       # by problem id
python -m pytest tests -q -k "lowering or warp or fold"       # no GPU needed: the gpu variants skip
python tests/bench.py m01_matmul_512 r02_big_sum              # auto / none / force / exact vs the reference
python tests/bench.py --all -c x4=unroll_auto=4 -c q512=unroll_quota=512 -c raw=cse=0 -c fast=precision=fast
python -c "from ptxc import *; compile_file('<ir>', Options(trace=('unroll', 'precision')))" > /dev/null   # one line per decision
```

Beyond pass/fail the runner reports registers for sm_89 and the local arch, spills, ptxas
warnings, median launch time, out-of-bounds writes (every buffer is fenced with sentinel bytes),
inputs modified by the kernel, and outputs that differ between repeated launches (a race;
not counted as a failure when the kernel accumulates with atomics). For every checked output it
also prints how much precision was given away: the largest distance from the expected value in
f32 ulps, and the share of the tolerance (`atol + rtol * |expected|`) the worst element used.

## Generating more problems

Add a spec to `tests/expressions_extra.py`, then:

```
python tests/generate_problems.py              # only specs whose folder does not exist yet
python tests/generate_problems.py --only ID    # one, regenerated
```

Each spec is a torch expression, a numpy reference and a tolerance; the hackathon's
`tools/generate.py` traces it with emmy for `sm_89` and writes the Kernel IR, `meta.json` and
`expected.npz`. Expressions emmy cannot trace are reported and skipped. About 7 s each.
Keep `code` free of intermediate variables: emmy's inline tracer takes every named tensor of
the last statement as a graph input.

Problems the compiler is known to fail are listed in `KNOWN_GAPS` in `test_problems.py` as
strict xfails: once a gap is fixed, the test reports XPASS and the entry must be removed.
