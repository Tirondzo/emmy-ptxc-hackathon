"""Printer: PtxModule -> text. Layout only; every decision was made upstream."""

from .code import PREFIX, Item, Label, PtxModule


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
    return "\n".join(lines) + "\n"


def header(m: PtxModule) -> list[str]:
    return [f".version {m.version}", f".target {m.target}", ".address_size 64", ""]


def shared_decls(m: PtxModule) -> list[str]:
    return [
        f".extern .shared .align {s.align} .b8 {s.name}[];"
        if s.extern
        else f".shared .align {s.align} .b8 {s.name}[{s.nbytes}];"
        for s in m.shared
    ]


def signature(m: PtxModule) -> list[str]:
    params = ",\n".join(f"\t.param .{p.ty.ptx} {p.name}" for p in m.params)
    bound = [f".maxntid {m.max_threads}, 1, 1"] if m.max_threads else []
    return [f".visible .entry {m.entry}(", params, ")", *bound]


def register_decls(m: PtxModule) -> list[str]:
    return [f"\t.reg .{bank} %{PREFIX[bank]}<{m.registers[bank] + 1}>;" for bank in PREFIX if bank in m.registers]


def item(i: Item) -> str:
    if isinstance(i, Label):
        return f"{i.name}:"
    guard = f"@{'!' if i.negate else ''}{i.guard} " if i.guard else ""
    operands = ", ".join(map(str, i.operands))
    return f"\t{guard}{i.opcode}{' ' + operands if operands else ''};"
