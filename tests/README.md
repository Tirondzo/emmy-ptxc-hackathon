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
  test_contract.py       the CLI contract: stdout is ASCII PTX, entry name, arg_order, determinism, speed
  expressions_extra.py   torch expressions for our own problems, one list per category
  problems/              generated from expressions_extra.py (IR + meta + expected, like the open set)
```

## Setup (Windows, done once)

```
python -m venv .venv                      # at the workspace root, next to both repos
.venv/Scripts/pip install emmy-ml==0.3.11 numpy requests pytest
.venv/Scripts/pip install cuda-python nvidia-cuda-nvcc nvidia-cuda-nvrtc   # driver API, ptxas, nvrtc
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128   # only to generate IRs
```

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
```

Beyond pass/fail the runner reports registers for sm_89 and the local arch, spills, ptxas
warnings, median launch time, out-of-bounds writes (every buffer is fenced with sentinel bytes),
inputs modified by the kernel, and outputs that differ between repeated launches (a race;
not counted as a failure when the kernel accumulates with atomics).

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
