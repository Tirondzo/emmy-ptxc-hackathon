"""The pipeline: load -> lower -> render, for one target."""

from pathlib import Path

from .config import SM89, Target
from .ir import Kernel
from .lower import Lowering, lower, prologue
from .ptx import Param, PtxBuilder, PtxModule, render
from .types import U64


def compile_file(ir_path: str | Path, target: Target = SM89) -> str:
    return compile_kernel(Kernel.load(Path(ir_path)), target)


def compile_kernel(kernel: Kernel, target: Target = SM89) -> str:
    target.check(kernel.launch)
    params = tuple(Param(buf.param_name, U64) for buf in kernel.params)
    module = PtxModule(kernel.entry, params, target.name, target.isa, max_threads=kernel.launch.block)
    with PtxBuilder(module), Lowering(kernel, target):
        prologue()
        lower(kernel.body)
    return render(module)
