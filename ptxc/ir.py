"""The input: emmy's IR object model and the launch contract from meta.json."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import IDENTITY
from .errors import CompileError, InvalidKernel, Unsupported
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
    """Through emmy's own alias map: sum -> add, prod -> multiply, amax / fmax -> maximum, fmin -> minimum."""
    return op.reduce_canon


def op_identity(op: ElementwiseImpl) -> float:
    """The neutral element a seeded accumulator starts from: `config.IDENTITY` by canonical name."""
    if (name := op_name(op)) not in IDENTITY:
        raise InvalidKernel(f"{op.name} has no identity to seed with")
    return IDENTITY[name]


# ---- the launch contract ------------------------------------------------------------


def row_major(shape: Sequence[int]) -> tuple[int, ...]:
    """Element strides, the last axis fastest."""
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


def identifier(name: str) -> str:
    """A name PTX takes as is; anything else would fail in ptxas, far from its cause."""
    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name):
        raise InvalidKernel(f"{name!r} is not a PTX identifier")
    return name


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    ty: Ty

    @property
    def param_name(self) -> str:
        return f"param_{self.name}"  # a buffer named `add` would otherwise read as an opcode


@dataclass(frozen=True)
class Launch:
    """One-dimensional by construction: `load` refuses a grid or block with a second dimension."""

    grid: int
    block: int
    smem_bytes: int  # dynamic shared memory requested at launch; 0 means every Smem is static

    @property
    def threads(self) -> int:
        return self.grid * self.block


@dataclass(frozen=True)
class Kernel:
    entry: str
    body: Body
    params: tuple[Buffer, ...]  # in meta.json `arg_order`: the signature contract
    launch: Launch

    @classmethod
    def load(cls, ir_path: Path) -> Kernel:
        """The IR through `Graph.from_dict`, meta.json beside it, one KernelOp per file. Whatever
        the two files get wrong (missing, malformed, a key absent) is one InvalidKernel line."""
        from emmy.compiler.graph import Graph  # the one runtime import of emmy

        try:
            graph = Graph.from_dict(json.loads(ir_path.read_text(encoding="utf-8")))
            meta = json.loads(ir_path.with_name("meta.json").read_text(encoding="utf-8"))
            kernels = [node.op for node in graph.nodes.values() if kind(node.op) == "KernelOp"]
            if len(kernels) != 1:
                raise Unsupported(f"{len(kernels)} KernelOps in one file")
            (op,) = kernels
            if op.serial:
                raise Unsupported("serial axes")
            spec = next((k for k in meta["kernels"] if k["entry"] == op.name), None)
            if spec is None:
                raise InvalidKernel(f"meta.json has no kernel with entry {op.name!r}")
            args = {a["name"]: a for a in spec["args"]}
            params = tuple(
                Buffer(identifier(n), tuple(args[n]["shape"]), storable(ty_of(args[n]["dtype"]))) for n in spec["arg_order"]
            )
            (grid, *rest), (block, *rest_b) = spec["grid"], spec["block"]
            if any(d != 1 for d in rest + rest_b):
                raise Unsupported("multi-dimensional launch")
            launch = Launch(grid, block, spec.get("smem_bytes", 0))
        except CompileError:
            raise
        except Exception as err:  # emmy's parser is not ours: whatever it raises is a malformed input
            raise InvalidKernel(f"{ir_path.name}: {type(err).__name__}: {err}") from err
        return cls(identifier(op.name), op.body, params, launch)
