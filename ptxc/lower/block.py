"""Block-level: Smem, Sync, WarpShuffle, TreeHalve, CpAsyncCopy / Commit / Wait."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from .. import ir
from ..errors import InvalidKernel, Unsupported
from ..ir import Node
from ..ptx import (
    Addr,
    Reg,
    Shared,
    Sym,
    Value,
    Vec,
    activemask,
    barrier,
    bitcast,
    builder,
    cvt,
    emit,
    if_,
    materialize,
    op,
    reg,
    setp,
)
from ..types import B32, INDEX, PRED, U32, storable
from . import expr
from .lowerer import array, assign, bind, lower, lowering, mutable, read, rule, scope, target
from .symbols import Array, Bound

POOL = "_smem_pool"  # the one `.extern .shared` array when meta.json launches dynamic shared memory


@rule("Smem")
def smem(s: Node) -> None:
    """Static: its own `.shared` declaration. Pooled: an offset carved from `POOL` by a running cursor."""
    ty = storable(ir.ty(s.dtype))
    nbytes = math.prod(s.extents) * ty.nbytes
    align = max(s.align, ty.nbytes)
    shared = builder().module.shared
    cur = lowering()
    if pool := cur.kernel.launch.smem_bytes:
        if not shared:
            shared.append(Shared(POOL, 0, 16, extern=True))  # declared once; its alignment grows with what it holds
        shared[0].align = max(shared[0].align, align)
        offset = -(-cur.smem_used // align) * align
        cur.smem_used = offset + nbytes
        if cur.smem_used > pool:
            raise InvalidKernel(f"shared arrays need {cur.smem_used} bytes, meta.json launches {pool}")
        base = Sym(POOL)
    else:
        shared.append(Shared(ir.identifier(s.name), nbytes, align))
        offset, base = 0, Sym(s.name)
    bind(s.name, Array(s.name, op("mov", U32, base), ty, tuple(s.extents), "shared", offset))


@rule("Sync")
def sync(s: Node) -> None:
    barrier(s.barrier_id, s.count, s.warp)


@rule("WarpShuffle")
def warp_shuffle(s: Node) -> None:
    """A power-of-two `length`: the butterfly, every lane meets lane ^ delta and ends with the result.
    Any other length (emmy never makes one): a butterfly cannot reach every lane, so the states fold
    down onto lane 0, each step guarded by `lane + delta < length`, and lane 0 is broadcast. The
    member mask is `activemask`: a sub-warp block or a guarded tile leaves lanes that never arrive."""
    warp = target().warp
    if s.length > warp:
        raise InvalidKernel(f"WarpShuffle over {s.length} lanes: a warp has {warp}")
    for name in s.state:
        mutable(name)
    mask = activemask()
    if s.length & (s.length - 1) == 0:
        for delta in halving(s.length):
            combine(s, [shuffle("bfly", read(name), delta, s.length, mask) for name in s.state])
        return
    lane = read("lane")
    for delta in halving(s.length):
        others = [shuffle("down", read(name), delta, warp, mask) for name in s.state]
        with if_(setp("lt", op("add", INDEX, lane, INDEX(delta)), INDEX(s.length))):
            combine(s, others)
    for name in s.state:
        assign(name, shuffle("idx", read(name), 0, warp, mask))


@rule("TreeHalve")
def tree_halve(s: Node) -> None:
    """Halving tree over `bufs`, as emmy renders it: `if tid < stride: bufs[tid] = combine(bufs[tid],
    bufs[tid + stride])`, a barrier after each step, then every thread takes `bufs[0]`."""
    if s.inner is not None or s.barrier_count is not None:
        raise Unsupported("segmented TreeHalve")
    for name in s.state:
        mutable(name)
    bufs = [array(name) for name in s.bufs]
    tid = read(s.tid_var)
    slots = [expr.locate(buf, tid) for buf in bufs]
    for stride in halving(s.length):
        with if_(active(tid, stride, s.length)):
            partners = [Addr(slot.base, slot.offset + stride * buf.ty.nbytes) for buf, slot in zip(bufs, slots)]
            for name, buf, slot in zip(s.state, bufs, slots, strict=True):
                assign(name, op(f"ld.{buf.space}", buf.ty, slot))
            combine(s, [op(f"ld.{buf.space}", buf.ty, at) for buf, at in zip(bufs, partners)])
            for name, buf, slot in zip(s.state, bufs, slots, strict=True):
                emit(f"st.{buf.space}.{buf.ty.ptx}", slot, read(name))
        barrier(s.barrier_id)
    for name, buf in zip(s.state, bufs, strict=True):
        assign(name, op(f"ld.{buf.space}", buf.ty, expr.locate(buf, None)))


def halving(n: int) -> Iterator[int]:
    """Strides of a butterfly or tree over `n` lanes, the powers of two below `n`, descending:
    halving(32) -> 16, 8, 4, 2, 1 and halving(6) -> 4, 2, 1."""
    stride = 1 << max(n - 1, 0).bit_length()
    while stride > 1:
        stride //= 2
        yield stride


def active(tid: Value, stride: int, length: int) -> Reg:
    """`tid < stride`, and for a length that is not a power of two, `tid + stride < length` too."""
    p = setp("lt", tid, INDEX(stride))
    if 2 * stride <= length:
        return p
    partner = op("add", INDEX, tid, INDEX(stride))
    return op("and", PRED, p, setp("lt", partner, INDEX(length)))


def shuffle(mode: str, x: Value, lane: int, length: int, mask: Reg) -> Value:
    """`shfl.sync.{mode}.b32` of one value, `lane` its delta or source; the clamp keeps it inside aligned
    `length`-lane groups. A 64-bit value goes as two halves, a predicate as 1 / 0."""
    if x.ty == PRED:
        return cvt(shuffle(mode, cvt(x, U32), lane, length, mask), PRED)
    clamp = B32(((target().warp - length) << 8) | target().lane_mask)

    def shfl(bits: Value) -> Reg:
        return op(f"shfl.sync.{mode}", B32, bits, INDEX(lane), clamp, mask)

    if x.ty.bits == 32:
        return bitcast(shfl(bitcast(x, B32)), x.ty)
    lo, hi = reg(B32), reg(B32)
    emit("mov.b64", Vec((lo, hi)), materialize(x))
    out = reg(x.ty)
    emit("mov.b64", out, Vec((shfl(lo), shfl(hi))))
    return out


def combine(s: Node, others: Sequence[Value]) -> None:
    """Fold `others` into `state`: instantiate `combine_states` in a fresh scope with `state_b` bound to them."""
    with scope():
        for name, other in zip(s.state_b, others, strict=True):
            bind(name, Bound(other))
        lower(s.combine_states)


@rule("CpAsyncCopy")
def cp_async_copy(s: Node) -> None:
    if s.swizzle != "NONE":
        raise Unsupported(f"swizzle {s.swizzle}")
    if s.nbytes not in (4, 8, 16):
        raise Unsupported(f"cp.async of {s.nbytes} bytes")
    index = s.smem_index if s.lane_index is None else [ir.plus(a, b) for a, b in zip(s.smem_index, s.lane_index)]
    dst = expr.address(array(s.smem), index)
    src = expr.address(array(s.src), s.src_index)
    cache = "cg" if s.nbytes == 16 else "ca"
    emit(f"cp.async.{cache}.shared.global", dst, src, INDEX(s.nbytes))


@rule("CpAsyncCommit")
def cp_async_commit(s: Node) -> None:
    emit("cp.async.commit_group")


@rule("CpAsyncWait")
def cp_async_wait(s: Node) -> None:
    emit("cp.async.wait_group", INDEX(s.group))
