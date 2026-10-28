"""Kernels written as emmy IR in Python, for the shapes the problem sets never produce.

`ptx_of(body, args)` lowers one in-process; `spec(args)` is the launch `gpu_runner.launch` and
`simwarp.run` take for it. `args` are spelled as meta.json spells them: name, dtype, shape, and
`data` for an input or `output` for an output.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from emmy.compiler.ir.axis import Axis, Dim
from emmy.compiler.ir.elementwise import ElementwiseImpl as Op
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Tile, WarpShuffle
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write

from ptxc.driver import compile_kernel
from ptxc.ir import Buffer, Kernel, Launch
from ptxc.types import storable, ty_of

ENTRY = "k_test"


def ptx_of(body: Sequence, args: Sequence[dict], block: int = 32, grid: int = 1, smem_bytes: int = 0) -> str:
    params = tuple(Buffer(a["name"], tuple(a["shape"]), storable(ty_of(a["dtype"]))) for a in args)
    return compile_kernel(Kernel(ENTRY, tuple(body), params, Launch(grid, block, smem_bytes)))


def spec(args: Sequence[dict], block: int = 32, grid: int = 1, smem_bytes: int = 0) -> dict:
    return {"entry": ENTRY, "arg_order": [a["name"] for a in args], "grid": [grid, 1, 1], "block": [block, 1, 1],
            "smem_bytes": smem_bytes, "zero_outputs": [], "args": list(args)}


# ---- the pieces: one Tile over `a0`, one element per thread ---------------------------------


def arg(name: str, dtype: str, data: np.ndarray | None = None, n: int = 32) -> dict:
    """An input holding `data`, or an output of `n` elements."""
    if data is None:
        return {"name": name, "dtype": dtype, "shape": [n], "output": True}
    return {"name": name, "dtype": dtype, "shape": list(np.shape(data)), "data": np.asarray(data).tolist()}


def tile(n: int, *body) -> Tile:
    return Tile(axes=(Axis("a0", Dim(n)),), body=tuple(body))


def load(name: str, buf: str, dtype=None) -> Load:
    return Load(names=(name,), input=buf, index=(Var("a0"),), dtype=dtype)


def write(buf: str, name: str) -> Write:
    return Write(output=buf, index=(Var("a0"),), values=(name,), value_dtype=None, atomic=False, swizzle="NONE")


def shuffle(length: int, **combine: str) -> WarpShuffle:
    """`shuffle(8, v="add", m="amax")`: states `v` and `m`, each folded with its op over `length` lanes."""
    return WarpShuffle(state=tuple(combine), state_b=tuple(f"{s}_b" for s in combine),
                       combine_states=tuple(Assign(s, Op(op), (s, f"{s}_b")) for s, op in combine.items()),
                       length=length)
