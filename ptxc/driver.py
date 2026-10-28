"""The pipeline: load -> lower -> render, for one target under one set of options."""

from pathlib import Path

from .config import SM89, Target
from .context import lowering, trace
from .ir import Kernel
from .lower import Lowering, lower, prologue
from .options import Options
from .ptx import Param, PtxBuilder, PtxModule, render
from .types import U64


def compile_file(ir_path: str | Path, options: Options | None = None, target: Target = SM89) -> str:
    return compile_kernel(Kernel.load(Path(ir_path)), options, target)


def compile_kernel(kernel: Kernel, options: Options | None = None, target: Target = SM89) -> str:
    options = options or Options()
    target.check(kernel.launch)
    params = tuple(Param(buf.param_name, U64) for buf in kernel.params)
    module = PtxModule(kernel.entry, params, target.name, target.isa, max_threads=kernel.launch.block)
    with PtxBuilder(module), Lowering(kernel, target, options):
        trace("precision", f"rtol={kernel.tolerance.rtol:g} atol={kernel.tolerance.atol:g}: {lowering().budget}")
        prologue()
        lower(kernel.body)
    return render(module)
