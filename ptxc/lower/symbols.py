"""Names: what an IR name denotes while lowering, with lexical scopes.

The rule that matters is rebinding versus mutation. An SSA name is `Bound` to a value and
may be rebound freely: in straight-line code the next read sees the new value. A
loop-carried name must be a `Variable`, a register that is *mutated* in place, because the
next iteration reads the register the loop header saw. Accumulators, `Init` state and the
states of WarpShuffle / TreeHalve are Variables declared before their loop. This is why the
compiler needs no phi nodes. The one invariant between the two: a Bound never aliases a
Variable's register, or mutation would reach a value that is supposed to be fixed.

The table is pure data; the Lowerer emits the movs.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from ..errors import InvalidKernel
from ..ir import row_major
from ..ptx import Reg, Value
from ..types import Ty


@dataclass(frozen=True)
class Bound:
    """An SSA value."""

    value: Value


@dataclass(frozen=True)
class Variable:
    """A loop-carried register, assigned in place."""

    reg: Reg


@dataclass(frozen=True)
class Array:
    """A global or shared buffer: where element 0 lives, its element type and shape."""

    name: str
    base: Reg  # u64 generic address for global; u32 shared-window address for shared
    ty: Ty
    shape: tuple[int, ...]
    space: Literal["global", "shared"]
    offset: int = 0  # byte offset inside the pool, when shared memory is pooled

    @property
    def strides(self) -> tuple[int, ...]:
        return row_major(self.shape)


type Symbol = Bound | Variable | Array


class Symbols:
    def __init__(self) -> None:
        self._scopes: ChainMap[str, Symbol] = ChainMap()
        self._variables: set[str] = set()  # every register a Variable ever owned

    @contextmanager
    def scope(self) -> Generator[None]:
        self._scopes = self._scopes.new_child()
        try:
            yield
        finally:
            self._scopes = self._scopes.parents

    def bind(self, name: str, symbol: Symbol) -> None:
        self._scopes[name] = symbol
        if isinstance(symbol, Variable):
            self._variables.add(symbol.reg.name)

    def is_variable(self, value: Value) -> bool:
        """`value` is a Variable's register, under whatever type `cvt` retyped it to."""
        return isinstance(value, Reg) and value.name in self._variables

    def get(self, name: str) -> Symbol | None:
        return self._scopes.get(name)

    def local(self, name: str) -> bool:
        """Bound in the innermost scope: rebinding it is straight-line and safe."""
        return name in self._scopes.maps[0]

    def lookup(self, name: str) -> Symbol:
        if (symbol := self.get(name)) is None:
            raise InvalidKernel(f"{name!r} is read before it is defined")
        return symbol
