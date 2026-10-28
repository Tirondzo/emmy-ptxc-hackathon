"""One warp of the PTX ptxc emits, in Python. The lanes run in lockstep; a lane that takes a
branch sits out until its label comes round, so `if` blocks work and loops do not. `run(ptx,
spec)` takes gpu_runner's launch spec (one block of at most 32 threads) and returns the outputs
as `launch` does.

Values are Python numbers, rounded to f32 or wrapped to 32 bits where the opcode says so;
comparisons are ordered unless the opcode is `neu`; a shuffle from a lane that does not exist
reads NaN, so a schedule that leans on one is caught by its sums. 32-bit values only (no
`{lo, hi}` splits), no shared memory.
"""

from __future__ import annotations

import math
import operator
import re
import struct
from collections.abc import Callable

import numpy as np
from gpu_runner import make_input

ITEM = re.compile(r"^(?:@(!?)(%\w+)\s+)?([\w.]+)\s*(.*)$")  # @!%p opcode a, b, c
ADDR = re.compile(r"^\[([%\w]+)([+-]\d+)?\]$")
SPECIAL = {"%tid.x": lambda lane, lanes: lane, "%laneid": lambda lane, lanes: lane, "%ctaid.x": lambda lane, lanes: 0,
           "%ntid.x": lambda lane, lanes: lanes, "%nctaid.x": lambda lane, lanes: 1}


def ordered(cmp: Callable) -> Callable:
    return lambda a, b: not (math.isnan(a) or math.isnan(b)) and cmp(a, b)


COMPARE = {"eq": ordered(operator.eq), "ne": ordered(operator.ne), "lt": ordered(operator.lt), "le": ordered(operator.le),
           "gt": ordered(operator.gt), "ge": ordered(operator.ge), "neu": operator.ne,
           "nan": lambda a, b: math.isnan(a) or math.isnan(b)}
ARITH = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "mad": lambda a, b, c: a * b + c,
         "div": lambda a, b: a / b if isinstance(a, float) else math.trunc(a / b), "rem": lambda a, b: int(math.fmod(a, b)),
         "neg": operator.neg, "abs": abs, "max": max, "min": min, "and": operator.and_, "or": operator.or_,
         "xor": operator.xor, "not": lambda a: not a if isinstance(a, bool) else ~a,
         "shl": operator.lshift, "shr": operator.rshift}


def run(ptx: str, spec: dict) -> dict[str, np.ndarray]:
    (lanes, *_), (blocks, *_) = spec["block"], spec["grid"]
    if blocks != 1 or lanes > 32:
        raise ValueError("one block of at most 32 threads")
    arrays = {a["name"]: make_input(a, {}).ravel() for a in spec["args"]}
    base = {name: (k + 1) << 20 for k, name in enumerate(arrays)}
    mem = {base[name] + i * arr.itemsize: v for name, arr in arrays.items() for i, v in enumerate(arr.tolist())}
    Warp(ptx, lanes, mem, {f"param_{name}": b for name, b in base.items()}).run()
    return {a["name"]: np.array([mem[base[a["name"]] + i * arr.itemsize] for i in range(arr.size)], arr.dtype).reshape(a["shape"])
            for a in spec["args"] if a.get("output") and (arr := arrays[a["name"]]) is not None}


def wrap32(x: int, signed: bool) -> int:
    x &= 0xFFFFFFFF
    return x - (1 << 32) if signed and x >> 31 else x


class Warp:
    def __init__(self, ptx: str, lanes: int, mem: dict[int, object], base: dict[str, int]) -> None:
        body = ptx[ptx.index("{") + 1 : ptx.rindex("}")]
        lines = [line.strip().rstrip(";") for line in body.splitlines()]
        self.items = [line for line in lines if line and not line.startswith(".reg")]
        self.lanes, self.mem, self.base = lanes, mem, base
        self.regs: list[dict[str, object]] = [{} for _ in range(lanes)]
        self.waiting: list[str | None] = [None] * lanes  # the label a lane that took a branch rejoins at

    def active(self) -> list[int]:
        return [lane for lane in range(self.lanes) if self.waiting[lane] is None]

    def run(self) -> None:
        for item in self.items:
            if item.endswith(":"):
                self.waiting = [None if w == item[:-1] else w for w in self.waiting]
                continue
            negate, guard, opcode, rest = ITEM.match(item).groups()
            args = [a.strip() for a in rest.split(",")] if rest else []
            for lane in self.active():
                if guard is None or bool(self.regs[lane][guard]) != bool(negate):
                    self.step(lane, opcode, args)

    def step(self, lane: int, opcode: str, args: list[str]) -> None:
        r = self.regs[lane]
        root, *mods = opcode.split(".")
        v = lambda a: self.value(lane, a)  # noqa: E731
        match root:
            case "ret" | "bar":
                pass
            case "bra":
                self.waiting[lane] = args[0]
            case "ld" if mods[0] == "param":
                r[args[0]] = self.base[ADDR.match(args[1])[1]]
            case "ld":
                r[args[0]] = self.mem[self.address(lane, args[1])]
            case "st":
                self.mem[self.address(lane, args[0])] = v(args[1])
            case "mov" | "cvta":
                r[args[0]] = v(args[1])
            case "cvt":
                x, to = v(args[1]), mods[-2]
                r[args[0]] = float(np.float32(x)) if to == "f32" else float(x) if to.startswith("f") else math.trunc(x)
            case "setp":
                r[args[0]] = COMPARE[mods[0]](v(args[1]), v(args[2]))
            case "selp":
                r[args[0]] = v(args[1]) if v(args[3]) else v(args[2])
            case "activemask":
                r[args[0]] = sum(1 << lane for lane in self.active())
            case "shfl":
                r[args[0]] = self.shuffle(mods[1], lane, args[1], v(args[2]), v(args[3]))
            case _:
                x = ARITH[root](*map(v, args[1:]))
                if mods[-1] == "f32":
                    x = float(np.float32(x))
                elif mods[-1] in ("s32", "u32", "b32") and isinstance(x, int):
                    x = wrap32(x, signed=mods[-1] == "s32")
                r[args[0]] = x

    def value(self, lane: int, a: str) -> object:
        if a in SPECIAL:
            return SPECIAL[a](lane, self.lanes)
        if a.startswith("%"):
            return self.regs[lane][a]
        if a.startswith(("0f", "0d")):
            return struct.unpack(">f" if a[1] == "f" else ">d", bytes.fromhex(a[2:]))[0]
        return int(a, 0)

    def address(self, lane: int, a: str) -> int:
        m = ADDR.match(a)
        return self.regs[lane][m[1]] + int(m[2] or 0)

    def shuffle(self, mode: str, lane: int, src: str, b: int, c: int) -> object:
        """The ISA's source-lane rule: `c` packs the segment mask and the clamp; out of range reads the lane itself."""
        mask, cval = (c >> 8) & 0x1F, c & 0x1F
        max_lane, min_lane = (lane & mask) | (cval & ~mask & 0x1F), lane & mask
        match mode:
            case "up":
                j, ok = lane - b, lane - b >= max_lane
            case "down":
                j, ok = lane + b, lane + b <= max_lane
            case "bfly":
                j, ok = lane ^ b, lane ^ b <= max_lane
            case _:
                j = min_lane | (b & ~mask & 0x1F)
                ok = j <= max_lane
        if not ok:
            j = lane
        return self.regs[j][src] if j < self.lanes else math.nan
