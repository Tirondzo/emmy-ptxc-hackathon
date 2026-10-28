"""The ambient lowering. `with Lowering(kernel, target, options):` in the driver makes one kernel, one
machine and one set of options the context of every rule and every recipe, through a ContextVar
(scoped, reset on exit, per thread). The accessors live here, outside `lower/`, so `ops.py` can read
them without importing the rules that import it."""

from __future__ import annotations

import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Target
    from .lower.lowerer import Lowering
    from .options import Options

current: ContextVar[Lowering] = ContextVar("lowering")


def lowering() -> Lowering:
    return current.get()


def target() -> Target:
    return lowering().target


def options() -> Options:
    return lowering().options


def trace(topic: str, message: str) -> None:
    """One line on stderr when `topic` is in `Options.trace`: how a decision was made."""
    if topic in options().trace:
        print(f"ptxc: {topic}: {message}", file=sys.stderr)
