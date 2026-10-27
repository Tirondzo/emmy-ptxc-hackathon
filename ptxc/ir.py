"""The input: emmy's IR object model and the launch contract from meta.json."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .types import Ty

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
    """`Axis.extent.as_static()`. Every problem is static; `is_static` false -> Unsupported."""
    ...


def ty(dtype: DataType | str | None) -> Ty | None:
    """A DataType, or the C spelling Smem uses ("float"), -> Ty. None means "infer it"."""
    ...


def op_name(op: ElementwiseImpl) -> str:
    """ElementwiseImpl -> its name: "add", "maximum", "exp"."""
    ...


def op_identity(op: ElementwiseImpl) -> float:
    """`ElementwiseImpl.identity`, the neutral element as emmy defines it; `has_identity`
    false -> InvalidKernel.
    """
    ...


# ---- the launch contract ------------------------------------------------------------


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
        """Row-major element strides, the last axis fastest."""
        ...


@dataclass(frozen=True)
class Launch:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem_bytes: int  # dynamic shared memory requested at launch; 0 means every Smem is static

    @property
    def threads(self) -> int:
        """grid.x * block.x. Launches are one-dimensional; anything else -> Unsupported."""
        ...


@dataclass(frozen=True)
class Kernel:
    entry: str
    body: Body
    params: tuple[Buffer, ...]  # in meta.json `arg_order`: the signature contract
    launch: Launch
    target: str = "sm_89"

    def buffer(self, name: str) -> Buffer:
        """The parameter buffer called `name`; not a parameter -> InvalidKernel."""
        ...

    @classmethod
    def load(cls, ir_path: Path) -> Kernel:
        """`Graph.from_dict` on the IR, meta.json beside it, the kernel spec matched by entry name.
        One KernelOp per file; several -> Unsupported
        Non-empty `serial` axes -> Unsupported.
        """
        ...
