"""Tensor cores: RegFragment, LdmatrixLoad, MmaSyncPtx, RegStore, emmy's `mma.sync` path.

A fragment is the PTX ISA's per-lane share of one m16n8k16 operand: four `.b32` of packed halves
for A, two for B, four `.f32` for C. The lane maps are the ISA's, its figures for `ldmatrix` and
`mma.m16n8k16`, spelled as index expressions over the prologue's `lane`, so `expr.address` folds
them like any other index. Whatever emmy's nodes can carry beyond what its f16 matmul emits
(gmem-direct loads, byte slabs, guards, epilogues, other atoms and dtypes) is `Unsupported`
before a wrong PTX is written.
"""

from __future__ import annotations

import math
import re

from .. import ir
from ..errors import InvalidKernel, Unsupported
from ..ir import Node
from ..ptx import Vec, cvt, emit, materialize, mov, reg
from ..types import B32, F16, F32, Imm
from . import expr
from .block import slab
from .compute import store_values
from .lowerer import array, bind, lowering, rule, target
from .symbols import Bound, Fragment

ATOM = (16, 8, 16)
LAYOUT = "m16n8k16"
SLOTS = {"a": (4, B32), "b": (2, B32), "c": (4, F32)}  # role -> registers per lane and their type


def lane(op: str, k: int) -> Node:
    """`lane % k` or `lane // k`: a lane's place in the ISA's maps, as an index expression."""
    return ir.binary(op, ir.var("lane"), ir.literal(k))


@rule("RegFragment")
def reg_fragment(s: Node) -> None:
    """One register per slot; a family (`count` > 1, emmy's re-rolled `name[k]`) is one binding per
    member under its spelled name. C starts at zero."""
    if s.shape != ATOM or s.nregs is not None or s.role not in SLOTS:
        raise Unsupported(f"fragment {s.name!r}: {s.role} of {'x'.join(map(str, s.shape))}")
    if ir.ty(s.dtype) != (F32 if s.role == "c" else F16):
        raise Unsupported(f"fragment {s.name!r}: {s.role}:{s.dtype.name}")
    whole_warps()
    n, ty = SLOTS[s.role]
    for k in range(s.count):
        regs = tuple(reg(ty) for _ in range(n))
        if s.role == "c":
            for r in regs:
                mov(r, ty(0))
        bind(s.name if s.count == 1 else f"{s.name}[{k}]", Fragment(regs))


def whole_warps() -> None:
    """`ldmatrix` and `mma.sync.aligned` take every lane of the warp: whole warps in the block, no
    guarded tile (its spare threads would skip the instruction), and a machine that has them."""
    cur = lowering()
    if target().sm < 80:
        raise Unsupported(f"mma.sync.m16n8k16 on {target().name}: sm_80 and later")
    if cur.kernel.launch.block % target().warp:
        raise InvalidKernel(f"mma.sync needs whole warps: a block of {cur.kernel.launch.block} threads")
    if cur.kernel.launch.threads > math.prod(cur.axes.values()):
        raise InvalidKernel("mma.sync in a guarded tile: the spare threads leave lanes out of a warp-wide instruction")


def fragment(ref: str) -> Fragment:
    """The fragment a node names: `_a0`, `_b[2]`, or `_b[_r0]` with `_r0` an unrolled loop's constant."""
    if (m := re.fullmatch(r"(\w+)\[(\w+)\]", ref)) and not m[2].isdigit():
        match lowering().syms.get(m[2]):
            case Bound(Imm() as v, offset):
                ref = f"{m[1]}[{int(v.value) + offset}]"
            case _:
                raise Unsupported(f"fragment {ref}: the subscript is not a constant (its loop is not unrolled)")
    match lowering().syms.lookup(ref):
        case Fragment() as f:
            return f
        case _:
            raise InvalidKernel(f"{ref!r} is not a fragment")


@rule("LdmatrixLoad")
def ldmatrix_load(s: Node) -> None:
    """`ldmatrix.sync.aligned.m8n8.x{2,4}[.trans].shared.b16`: A as x4, B as x2.trans, or two B fragments
    at once (`pair_frag`, x4.trans). Lane l addresses row l % 16 of the tile and, with four matrices,
    the 8-column block l // 16: the ISA's order of the A slots (rows 0-7, 8-15; k 0-7, 8-15)."""
    if not s.staged or s.byte_slab or s.scale_buffer is not None or s.gmem_guard or s.k_zero or s.b_trans:
        raise Unsupported(f"ldmatrix of {s.frag}: only a staged, unguarded 16-bit slab")
    if s.fragment_layout != LAYOUT:
        raise Unsupported(f"fragment layout {s.fragment_layout}")
    arr = array(s.src_buffer)
    if arr.space != "shared" or arr.ty.bits != 16 or len(arr.shape) < 2:
        raise Unsupported(f"ldmatrix from {arr.name}: a 16-bit shared matrix")
    if s.ldm not in (0, arr.strides[-2]):
        raise Unsupported(f"ldmatrix from {arr.name}{list(arr.shape)} with ldm {s.ldm}")
    regs = fragment(s.frag).regs + (fragment(s.pair_frag).regs if s.pair_frag else ())
    row = ir.plus(s.src_index[-2], lane("%", 16))
    col = s.src_index[-1] if len(regs) == 2 else ir.plus(s.src_index[-1], ir.binary("*", lane("//", 16), ir.literal(8)))
    trans = ".trans" if s.role == "b" else ""
    emit(f"ldmatrix.sync.aligned.m8n8.x{len(regs)}{trans}.shared.b16", Vec(regs), slab(arr, (*s.src_index[:-2], row, col), s.swizzle))


@rule("MmaSyncPtx")
def mma_sync(s: Node) -> None:
    """`c = a . b + c` over the fragments; `d` and `c` are the same registers."""
    if s.shape != ATOM or (s.ab_dtype, s.c_dtype) != ("f16", "f32") or s.b_row_major or s.sfa_frag is not None:
        raise Unsupported(f"mma m{s.shape[0]}n{s.shape[1]}k{s.shape[2]} {s.ab_dtype} into {s.c_dtype}")
    c, a, b = (fragment(f).regs for f in (s.c_frag, s.a_frag, s.b_frag))
    emit("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", Vec(c), Vec(a), Vec(b), Vec(c))


@rule("RegStore")
def reg_store(s: Node) -> None:
    """Lane l holds rows g and g + 8 (g = l // 4) at columns 2t and 2t + 1 (t = l % 4): one store of a
    pair per row, converted to the buffer's type (`store_values` makes it a `.v2`)."""
    if s.epilogue is not None or s.extra_frags or s.m_guard or s.n_guard or s.atomic or s.swizzle != "NONE":
        raise Unsupported(f"RegStore to {s.dst_buffer}: epilogues, guards, atomics and swizzled slabs")
    if s.fragment_layout != LAYOUT or s.volta_interleaved:
        raise Unsupported(f"fragment layout {s.fragment_layout}")
    arr = array(s.dst_buffer)
    if len(arr.shape) < 2:
        raise InvalidKernel(f"RegStore into {arr.name}{list(arr.shape)}: a fragment is two-dimensional")
    if s.row_dim not in (None, len(arr.shape) - 2) or s.col_dim not in (None, len(arr.shape) - 1) or s.ldm not in (0, arr.strides[-2]):
        raise Unsupported(f"RegStore to {arr.name}: the fragment's rows are not the next-to-last axis")
    regs = fragment(s.frag).regs
    col = ir.plus(s.dst_index[-1], ir.binary("*", lane("%", 4), ir.literal(2)))
    for half, pair in enumerate((regs[:2], regs[2:])):
        row = ir.plus(s.dst_index[-2], ir.plus(lane("//", 4), ir.literal(8 * half)))
        addr = expr.address(arr, (*s.dst_index[:-2], row, col))
        store_values(arr, addr, [materialize(cvt(r, arr.ty)) for r in pair])
