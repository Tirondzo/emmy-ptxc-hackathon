"""The input: emmy's IR object model and the launch contract from meta.json."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import InvalidKernel, Unsupported
from .types import Ty, storable, ty_of

if TYPE_CHECKING:  # annotations only: ptxc still imports without emmy installed
    from emmy.compiler.dtype import DataType
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.expr import Expr
    from emmy.compiler.ir.stmt.base import Stmt

type Node = Stmt | Expr  # any IR node; rules read its fields, the IR's public vocabulary
type Body = Sequence[Stmt]  # emmy's Body is a tuple subclass; Sequence also admits our own lists


# ---- emmy's object model ----------------------------------------------------------


def kind(node: Node) -> str:
    """The node's class name: the dispatch key of every rule."""
    return type(node).__name__


def extent(axis: Axis) -> int:
    if not axis.extent.is_static:
        raise Unsupported(f"symbolic extent of axis {axis.name!r}")
    return axis.extent.as_static()


def ty(dtype: DataType | str | None) -> Ty | None:
    """A DataType, or the C spelling Smem uses ("float"); None means "infer it"."""
    if dtype is None:
        return None
    return ty_of(dtype if isinstance(dtype, str) else dtype.name)


def plus(a: Expr, b: Expr) -> Expr:
    from emmy.compiler.ir.expr import BinaryExpr

    return BinaryExpr(op="+", left=a, right=b)


def op_name(op: ElementwiseImpl) -> str:
    return op.name


def op_identity(op: ElementwiseImpl) -> float:
    """emmy's neutral element: maximum is -1e30, not -inf, so inf - inf never appears."""
    if not op.has_identity:
        raise InvalidKernel(f"{op.name} has no identity to seed with")
    return op.identity


# ---- the launch contract ------------------------------------------------------------


def row_major(shape: Sequence[int]) -> tuple[int, ...]:
    """Element strides, the last axis fastest."""
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    ty: Ty
    is_output: bool

    @property
    def param_name(self) -> str:
        return f"param_{self.name}"  # a buffer named `add` would otherwise read as an opcode

    @property
    def strides(self) -> tuple[int, ...]:
        return row_major(self.shape)


@dataclass(frozen=True)
class Launch:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem_bytes: int  # dynamic shared memory requested at launch; 0 means every Smem is static

    @property
    def threads(self) -> int:
        if self.grid[1:] != (1, 1) or self.block[1:] != (1, 1):
            raise Unsupported("multi-dimensional launch")
        return self.grid[0] * self.block[0]


@dataclass(frozen=True)
class Kernel:
    entry: str
    body: Body
    params: tuple[Buffer, ...]  # in meta.json `arg_order`: the signature contract
    launch: Launch
    target: str = "sm_89"

    def buffer(self, name: str) -> Buffer:
        for buf in self.params:
            if buf.name == name:
                return buf
        raise InvalidKernel(f"buffer {name!r} is not a kernel parameter")

    @classmethod
    def load(cls, ir_path: Path) -> Kernel:
        """The IR through `Graph.from_dict`, meta.json beside it, one KernelOp per file."""
        from emmy.compiler.graph import Graph  # the one runtime import of emmy

        graph = Graph.from_dict(json.loads(ir_path.read_text()))
        kernels = [node.op for node in graph.nodes.values() if kind(node.op) == "KernelOp"]
        if len(kernels) != 1:
            raise Unsupported(f"{len(kernels)} KernelOps in one file")
        (op,) = kernels
        if op.serial:
            raise Unsupported("serial axes")

        meta = json.loads(ir_path.with_name("meta.json").read_text())
        spec = next((k for k in meta["kernels"] if k["entry"] == op.name), None)
        if spec is None:
            raise InvalidKernel(f"meta.json has no kernel with entry {op.name!r}")
        args = {a["name"]: a for a in spec["args"]}
        return cls(
            entry=op.name,
            body=op.body,
            params=tuple(
                Buffer(n, tuple(args[n]["shape"]), storable(ty_of(args[n]["dtype"])), bool(args[n].get("output")))
                for n in spec["arg_order"]
            ),
            launch=Launch(tuple(spec["grid"]), tuple(spec["block"]), spec.get("smem_bytes", 0)),
        )
