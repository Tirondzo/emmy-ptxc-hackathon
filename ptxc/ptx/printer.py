"""Printer: PtxModule -> text. Layout only; every decision was made upstream."""

from ..errors import InvalidKernel
from .code import Item, PtxModule


def render(m: PtxModule) -> str:
    lines = [
        *header(m),
        *shared_decls(m),
        *signature(m),
        "{",
        *register_decls(m),
        *(item(i) for i in m.body),
        "\tret;",
        "}",
    ]
    text = "\n".join(lines) + "\n"
    if not text.isascii():  # ptxas rejects non-ASCII anywhere, comments included
        raise InvalidKernel("PTX output contains non-ASCII text")
    return text


def header(m: PtxModule) -> list[str]:
    """`.version`, `.target`, `.address_size 64`."""
    ...


def shared_decls(m: PtxModule) -> list[str]:
    """`.shared .align A .b8 name[N];`, or for the pool `.extern .shared .align 16 .b8 name[];`."""
    ...


def signature(m: PtxModule) -> list[str]:
    """`.visible .entry name(` one `.param .u64 name` per line `)`."""
    ...


def register_decls(m: PtxModule) -> list[str]:
    """One `.reg .{bank} %{prefix}<count+1>;` per bank in use."""
    ...


def item(i: Item) -> str:
    """A label as `name:`; an instruction as `\\t[@[!]%p ]opcode a, b, c;`."""
    ...
