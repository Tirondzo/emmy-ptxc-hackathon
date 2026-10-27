"""Block-level: Smem, Sync, WarpShuffle, TreeHalve, CpAsyncCopy / Commit / Wait."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from .. import ir
from ..errors import InvalidKernel, Unsupported
from ..ir import Node
from ..ptx import Addr, Imm, Reg, Shared, Sym, Value, Vec
from ..types import B32, INDEX, PRED, U32
from . import expr
from .lowerer import Lowerer, rule
from .symbols import Array, Bound

POOL = "_smem_pool"  # the one `.extern .shared` array when meta.json launches dynamic shared memory


@rule("Smem")
def smem(low: Lowerer, s: Node) -> None:
    """Static: its own `.shared` declaration. Pooled: an offset carved from `POOL` by a running cursor."""
    ty = ir.ty(s.dtype)
    nbytes = math.prod(s.extents) * ty.nbytes
    align = max(s.align, ty.nbytes)
    shared = low.b.module.shared
    if pool := low.kernel.launch.smem_bytes:
        if not shared:
            shared.append(Shared(POOL, 0, 16, extern=True))
        offset = -(-low.smem_used // align) * align
        low.smem_used = offset + nbytes
        if low.smem_used > pool:
            raise InvalidKernel(f"shared arrays need {low.smem_used} bytes, meta.json launches {pool}")
        base = Sym(POOL)
    else:
        shared.append(Shared(s.name, nbytes, align))
        offset, base = 0, Sym(s.name)
    reg = low.b.op("mov", U32, base)
    low.syms.bind(s.name, Array(s.name, reg, ty, tuple(s.extents), "shared", offset))


@rule("Sync")
def sync(low: Lowerer, s: Node) -> None:
    low.b.barrier(s.barrier_id, s.count, s.warp)


@rule("WarpShuffle")
def warp_shuffle(low: Lowerer, s: Node) -> None:
    """Butterfly: for each delta, every state meets its lane ^ delta partner and `combine` folds it in."""
    for name in s.state:
        low.seed(name, low.read(name))
    for delta in halving(s.length):
        combine(low, s, [shuffle(low, low.read(name), delta, s.length) for name in s.state])


@rule("TreeHalve")
def tree_halve(low: Lowerer, s: Node) -> None:
    """Halving tree over `bufs`, as emmy renders it: `if tid < stride: bufs[tid] = combine(bufs[tid],
    bufs[tid + stride])`, a barrier after each step, then every thread takes `bufs[0]`."""
    if s.inner is not None or s.barrier_count is not None:
        raise Unsupported("segmented TreeHalve")
    for name in s.state:
        low.seed(name, low.read(name))
    bufs = [low.array(name) for name in s.bufs]
    tid = low.read(s.tid_var)
    slots = [expr.locate(low, buf, tid) for buf in bufs]
    for stride in halving(s.length):
        with low.b.if_(active(low, tid, stride, s.length)):
            partners = [Addr(slot.base, slot.offset + stride * buf.ty.nbytes) for buf, slot in zip(bufs, slots)]
            for name, buf, slot in zip(s.state, bufs, slots, strict=True):
                low.assign(name, low.b.op(f"ld.{buf.space}", buf.ty, slot))
            combine(low, s, [low.b.op(f"ld.{buf.space}", buf.ty, at) for buf, at in zip(bufs, partners)])
            for name, buf, slot in zip(s.state, bufs, slots, strict=True):
                low.b.emit(f"st.{buf.space}.{buf.ty.ptx}", slot, low.read(name))
        low.b.barrier(s.barrier_id)
    for name, buf in zip(s.state, bufs, strict=True):
        low.assign(name, low.b.op(f"ld.{buf.space}", buf.ty, expr.locate(low, buf, None)))


def halving(n: int) -> Iterator[int]:
    """Strides of a butterfly or tree over `n` lanes: the powers of two below `n`, descending."""
    stride = 1 << max(n - 1, 0).bit_length()
    while stride > 1:
        stride //= 2
        yield stride


def active(low: Lowerer, tid: Value, stride: int, length: int) -> Reg:
    """`tid < stride`, and for a length that is not a power of two, `tid + stride < length` too."""
    p = low.b.setp("lt", tid, Imm(stride, INDEX))
    if 2 * stride <= length:
        return p
    partner = low.b.op("add", INDEX, tid, Imm(stride, INDEX))
    return low.b.op("and", PRED, p, low.b.setp("lt", partner, Imm(length, INDEX)))


def shuffle(low: Lowerer, x: Value, delta: int, length: int) -> Value:
    """`shfl.sync.bfly.b32` of one value; a 64-bit value goes as two halves."""
    clamp = Imm(((32 - length) << 8) | 0x1F, B32)

    def bfly(bits: Value) -> Reg:
        return low.b.op("shfl.sync.bfly", B32, bits, Imm(delta, INDEX), clamp, Imm(0xFFFFFFFF, B32))

    if x.ty.bits == 32:
        return low.b.bitcast(bfly(low.b.bitcast(x, B32)), x.ty)
    lo, hi = low.b.reg(B32), low.b.reg(B32)
    low.b.emit("mov.b64", Vec((lo, hi)), low.b.materialize(x))
    out = low.b.reg(x.ty)
    low.b.emit("mov.b64", out, Vec((bfly(lo), bfly(hi))))
    return out


def combine(low: Lowerer, s: Node, others: Sequence[Value]) -> None:
    """Fold `others` into `state`: instantiate `combine_states` in a fresh scope with `state_b` bound to them."""
    with low.syms.scope():
        for name, other in zip(s.state_b, others, strict=True):
            low.syms.bind(name, Bound(other))
        low.block(s.combine_states)


@rule("CpAsyncCopy")
def cp_async_copy(low: Lowerer, s: Node) -> None:
    if s.swizzle != "NONE":
        raise Unsupported(f"swizzle {s.swizzle}")
    if s.nbytes not in (4, 8, 16):
        raise Unsupported(f"cp.async of {s.nbytes} bytes")
    index = s.smem_index if s.lane_index is None else [ir.plus(a, b) for a, b in zip(s.smem_index, s.lane_index)]
    dst = expr.address(low, low.array(s.smem), index)
    src = expr.address(low, low.array(s.src), s.src_index)
    cache = "cg" if s.nbytes == 16 else "ca"
    low.b.emit(f"cp.async.{cache}.shared.global", dst, src, Imm(s.nbytes, INDEX))


@rule("CpAsyncCommit")
def cp_async_commit(low: Lowerer, s: Node) -> None:
    low.b.emit("cp.async.commit_group")


@rule("CpAsyncWait")
def cp_async_wait(low: Lowerer, s: Node) -> None:
    low.b.emit("cp.async.wait_group", Imm(s.group, INDEX))
