"""Names: what an IR name denotes while lowering, with lexical scopes.

The rule that matters is rebinding versus mutation. An SSA name is `Bound` to a value and
may be rebound freely: in straight-line code the next read sees the new value. A
loop-carried name must be a `Variable`, a register that is *mutated* in place, because the
next iteration reads the register the loop header saw. Accumulators, `Init` state and the
states of WarpShuffle / TreeHalve are Variables declared before their loop. This is why the
compiler needs no phi nodes.

The table is pure data; the Lowerer emits the movs.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

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


type Symbol = Bound | Variable | Array


class Symbols:
    def __init__(self) -> None:
        self._scopes: ChainMap[str, Symbol] = ChainMap()

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Bindings made inside are dropped on exit; outer bindings stay visible."""
        ...

    def bind(self, name: str, symbol: Symbol) -> None: ...

    def get(self, name: str) -> Symbol | None: ...

    def local(self, name: str) -> bool:
        """Bound in the innermost scope: rebinding it is straight-line and safe."""
        ...

    def lookup(self, name: str) -> Symbol:
        """Unbound -> InvalidKernel: a name read before it is defined."""
        ...
