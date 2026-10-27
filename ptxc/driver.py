"""The pipeline: load -> lower -> render."""

from pathlib import Path

from .ir import Kernel
from .lower import Lowerer
from .ptx import Param, PtxBuilder, PtxModule, render
from .types import U64


def compile_file(ir_path: str | Path) -> str:
    kernel = Kernel.load(Path(ir_path))
    params = tuple(Param(buf.param_name, U64) for buf in kernel.params)
    b = PtxBuilder(PtxModule(kernel.entry, params, kernel.target))
    Lowerer(kernel, b).run()
    return render(b.finish())
