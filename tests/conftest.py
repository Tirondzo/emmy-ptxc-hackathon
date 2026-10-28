"""pytest wiring: one `problem_dir` parameter per problem folder, one GPU for the session.

    pytest tests                                   # the open set + tests/problems
    pytest tests --problems path/to/dir            # only there (repeatable)
    pytest tests -k "p1 or r2"                     # the usual pytest selection, by problem id
    pytest tests --compiler other/compile.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # gpu_runner, simwarp, kernels
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the ptxc package under test

import simwarp  # noqa: E402
from gpu_runner import CONTRACT_ARCH, DEFAULT_COMPILER, PROBLEM_DIRS, Gpu, all_problems, assemble, launch  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--problems", action="append", default=[], metavar="DIR",
                     help="problem folder(s); default: the hackathon's open set and tests/problems")
    parser.addoption("--compiler", default=str(DEFAULT_COMPILER), help="the compile.py under test")
    parser.addoption("--repeat", type=int, default=3, help="launches per kernel whose outputs must agree bitwise")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "problem_dir" in metafunc.fixturenames:
        roots = [Path(p) for p in metafunc.config.getoption("--problems")] or PROBLEM_DIRS
        dirs = all_problems(roots)
        metafunc.parametrize("problem_dir", dirs, ids=[d.name for d in dirs])


@pytest.fixture(scope="session")
def gpu() -> Gpu:
    try:
        return Gpu()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no usable CUDA device: {e}")


@pytest.fixture(params=["simulated", "gpu"])
def warp(request: pytest.FixtureRequest) -> Callable[[str, dict], dict[str, np.ndarray]]:
    """`warp(ptx, spec) -> outputs` for one block of one warp: on simwarp, and on the GPU when there is one."""
    if request.param == "simulated":
        return simwarp.run
    gpu: Gpu = request.getfixturevalue("gpu")

    def on_gpu(ptx: str, spec: dict) -> dict[str, np.ndarray]:
        if gpu.arch != CONTRACT_ARCH:
            assemble(ptx, CONTRACT_ARCH)  # the contract's arch first, as the hosted runner would
        return launch(gpu, assemble(ptx, gpu.arch).cubin, spec, {}).outputs

    return on_gpu


@pytest.fixture(scope="session")
def compiler(request: pytest.FixtureRequest) -> str:
    return str(Path(request.config.getoption("--compiler")).resolve())


@pytest.fixture(scope="session")
def repeat(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--repeat"))
