"""Block-level: Smem, Sync, WarpShuffle, TreeHalve, CpAsyncCopy / Commit / Wait."""

from __future__ import annotations

from ..ir import Node
from ..ptx import Value
from .lowerer import Lowerer, rule


@rule("Smem")
def smem(low: Lowerer, s: Node) -> None:
    """One `__shared__` array, bound as an Array(space="shared").

    launch.smem_bytes == 0: append a static `Shared` declaration to the module (the printer
    places it). Otherwise carve `offset` from one `.extern .shared` pool with the running
    cursor `low.smem_used`, aligned up; the pool is declared on first use and the total is
    checked against smem_bytes. `align` 0 means the element's natural alignment.
    """
    ...


@rule("Sync")
def sync(low: Lowerer, s: Node) -> None:
    low.b.barrier(s.barrier_id, s.count, s.warp)


@rule("WarpShuffle")
def warp_shuffle(low: Lowerer, s: Node) -> None:
    """Butterfly over `length` lanes. `seed` every state first (a Bound becomes a Variable
    holding its value); then for delta in length/2, ..., 1: each state_b =
    shfl.sync.bfly(state, delta), and `combine` folds it in. Narrower than a warp: the
    shuffle's clamp operand is ((32 - length) << 8) | 0x1f."""
    ...


@rule("TreeHalve")
def tree_halve(low: Lowerer, s: Node) -> None:
    """Halving tree over `bufs`, `length` partials indexed by `tid_var`, as emmy renders it:

        for stride in length/2, ..., 1:
            if tid < stride:  state = bufs[tid]; state_b = bufs[tid + stride]; combine; bufs[tid] = state
            barrier           (outside the branch: every thread reaches it)
        state = bufs[0]       (broadcast to every thread)

    States are `seed`ed first, as in WarpShuffle. length not a power of two: also guard
    `tid + stride < length`. `inner` or `barrier_count` set -> Unsupported.
    """
    ...


def shuffle(low: Lowerer, x: Value, delta: int, length: int) -> Value:
    """`shfl.sync.bfly.b32` of one value: bitcast to b32 and back; 64-bit values as two halves."""
    ...


def combine(low: Lowerer, s: Node) -> None:
    """Instantiate `combine_states` in a fresh scope with `state_b` already bound. Its Assigns
    to the `state` names mutate their Variables, which is the whole mechanism. The caller
    has `seed`ed the states, so an Assign never meets a Bound from the enclosing scope,
    which `Lowerer.assign` would refuse."""
    ...


@rule("CpAsyncCopy")
def cp_async_copy(low: Lowerer, s: Node) -> None:
    """`cp.async.{cg|ca}.shared.global [dst], [src], nbytes`: cg for 16 bytes, ca for 4 and 8.
    dst = smem[smem_index + lane_index] per dimension (lane_index may be None);
    src = global[src_index]. `lane_rows` is informational. swizzle != NONE -> Unsupported."""
    ...


@rule("CpAsyncCommit")
def cp_async_commit(low: Lowerer, s: Node) -> None: ...


@rule("CpAsyncWait")
def cp_async_wait(low: Lowerer, s: Node) -> None:
    """`cp.async.wait_group {group}`."""
    ...
