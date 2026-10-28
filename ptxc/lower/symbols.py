"""Names: what an IR name denotes while lowering, with lexical scopes.

The rule that matters is rebinding versus mutation. An SSA name is `Bound` to a value and
may be rebound freely: in straight-line code the next read sees the new value. A
loop-carried name must be a `Variable`, a register that is *mutated* in place, because the
next iteration reads the register the loop header saw. Accumulators, `Init` state and the
states of WarpShuffle / TreeHalve are Variables declared before their loop. This is why the
compiler needs no phi nodes. The one invariant between the two: a Bound never aliases a
Variable's register, or mutation would reach a value that is supposed to be fixed.

The same scopes remember pure index values (`remember` / `recall`), forgotten with the scope,
so nothing is reused where its register is not defined.

The table is pure data; the Lowerer emits the movs.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Generator, Hashable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from ..errors import InvalidKernel
from ..ir import row_major
from ..ptx import Reg, Value
from ..types import Ty


@dataclass(frozen=True)
class Bound:
    """An SSA value, plus the constant an unrolled copy adds to it (`Bound(var, 8)`), which
    `expr.affine` folds into a displacement; a read outside an index pays the `add`."""

    value: Value
    offset: int = 0


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
        self._memo: ChainMap[Hashable, Value] = ChainMap()  # pure values already computed in a scope
        self._variables: set[str] = set()  # every register a Variable ever owned

    @contextmanager
    def scope(self) -> Generator[None]:
        self._scopes, self._memo = self._scopes.new_child(), self._memo.new_child()
        try:
            yield
        finally:
            self._scopes, self._memo = self._scopes.parents, self._memo.parents

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

    def recall(self, key: Hashable) -> Value | None:
        return self._memo.get(key)

    def remember(self, key: Hashable, value: Value) -> None:
        self._memo[key] = value
