"""Tensor cores, read off the PTX text: the fragment registers, the lane maps of ldmatrix and the
C store, the swizzle, the family subscripts, the whole-warp contract; and emmy's f16 matmul end to end."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from emmy.compiler import dtype as D
from emmy.compiler.ir.axis import Axis, Dim
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCopy, LdmatrixLoad, MmaSyncPtx, RegFragment, RegStore, Smem
from emmy.compiler.ir.stmt.blocks import Loop
from gpu_runner import PROBLEM_DIRS, all_problems
from kernels import ENTRY, tile

from ptxc import Options, compile_file
from ptxc.config import SM89, Target
from ptxc.driver import compile_kernel
from ptxc.errors import InvalidKernel, Unsupported
from ptxc.ir import Buffer, Kernel, Launch
from ptxc.types import storable, ty_of

SHAPE = (16, 8, 16)
ZERO = (Literal(0, "int"), Literal(0, "int"))
OUT16 = {"name": "out", "dtype": "f16", "shape": [16, 8], "output": True}
OUT32 = {"name": "out", "dtype": "f32", "shape": [16, 8], "output": True}
X16 = {"name": "x", "dtype": "f16", "shape": [16, 16], "data": [[0.0] * 16] * 16}
X32 = {"name": "x", "dtype": "f32", "shape": [16, 16], "data": [[0.0] * 16] * 16}


def frag(name: str, role: str, count: int = 1) -> RegFragment:
    return RegFragment(name=name, role=role, shape=SHAPE, dtype=D.F32 if role == "c" else D.F16, count=count)


def mma(c: str = "_c", a: str = "_a", b: str = "_b") -> MmaSyncPtx:
    return MmaSyncPtx(c_frag=c, a_frag=a, b_frag=b, shape=SHAPE)


def slab(name: str = "s", extents=(16, 16), dtype: str = "__half") -> Smem:
    return Smem(name=name, extents=extents, dtype=dtype, align=16)


def ldmatrix(role: str = "a", frag: str = "_a", **fields) -> LdmatrixLoad:
    return LdmatrixLoad(frag=frag, src_buffer="s", src_index=ZERO, role=role, ldm=16, **fields)


def lowered(*body, args=(OUT16,), block: int = 32, grid: int = 1, options: Options | None = None, target: Target = SM89) -> str:
    """One warp over a tile of 32 cells: no guard, whole warps."""
    params = tuple(Buffer(a["name"], tuple(a["shape"]), storable(ty_of(a["dtype"]))) for a in args)
    return compile_kernel(Kernel(ENTRY, (tile(32, *body),), params, Launch(grid, block, 0)), options, target)


def problem(name: str) -> Path:
    for d in all_problems(PROBLEM_DIRS):
        if d.name == name:
            return d
    pytest.skip(f"problem {name} is not checked out")


class TestFragments:
    def test_mma_accumulates_in_place(self) -> None:
        ptx = lowered(frag("_a", "a"), frag("_b", "b"), frag("_c", "c"), mma())
        assert ptx.count("mov.f32 %f") == 4 and "0f00000000" in ptx  # C starts at zero
        m = re.search(r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.f16\.f16\.f32 (\{[^}]*\}), (\{[^}]*\}), (\{[^}]*\}), (\{[^}]*\});", ptx)
        assert m and m[1] == m[4] == "{%f1, %f2, %f3, %f4}"  # d is c
        assert m[2].count("%r") == 4 and m[3].count("%r") == 2  # four packed-half registers for A, two for B

    def test_family_members_by_subscript(self) -> None:
        ptx = lowered(frag("_a", "a"), frag("_b", "b", count=2), frag("_c", "c", count=2), mma("_c[0]", b="_b[0]"), mma("_c[1]", b="_b[1]"))
        mmas = re.findall(r"mma\.sync\S+ (\{[^}]*\}), \{[^}]*\}, (\{[^}]*\})", ptx)
        assert len(mmas) == 2 and mmas[0] != mmas[1]  # each member its own registers

    def test_subscript_by_an_unrolled_loop(self) -> None:
        body = [frag("_a", "a"), frag("_b", "b", count=2), frag("_c", "c", count=2),
                Loop(axis=Axis("_r0", Dim(2)), body=(mma("_c[_r0]", b="_b[_r0]"),), unroll=True, seed=False)]
        assert lowered(*body).count("mma.sync") == 2
        with pytest.raises(Unsupported, match="not unrolled"):
            lowered(*body, options=Options(unroll="none"))

    def test_other_atoms_are_unsupported(self) -> None:
        with pytest.raises(Unsupported, match="16x8x8"):
            lowered(RegFragment(name="_a", role="a", shape=(16, 8, 8), dtype=D.F16))
        with pytest.raises(Unsupported, match="bf16"):
            lowered(frag("_a", "a"), frag("_b", "b"), frag("_c", "c"), MmaSyncPtx(c_frag="_c", a_frag="_a", b_frag="_b", shape=SHAPE, ab_dtype="bf16"))


class TestLdmatrix:
    def test_a_is_x4_at_row_lane_mod_16_and_column_block_lane_div_16(self) -> None:
        ptx = lowered(slab(), frag("_a", "a"), ldmatrix())
        assert re.search(r"ldmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16 \{%r\d+, %r\d+, %r\d+, %r\d+\}, \[%r\d+\];", ptx)
        assert re.search(r"and\.b32 %r\d+, %r\d+, 0xf;", ptx) and re.search(r"shr\.s32 %r\d+, %r\d+, 4;", ptx)
        assert "ld.shared" not in ptx

    def test_b_is_x2_trans_and_a_pair_x4_trans(self) -> None:
        single = lowered(slab(), frag("_b", "b"), ldmatrix("b", "_b"))
        assert re.search(r"ldmatrix\.sync\.aligned\.m8n8\.x2\.trans\.shared\.b16 \{%r\d+, %r\d+\}, ", single)
        pair = lowered(slab(), frag("_b", "b", count=2), ldmatrix("b", "_b[0]", pair_frag="_b[1]"))
        assert re.search(r"ldmatrix\.sync\.aligned\.m8n8\.x4\.trans\.shared\.b16 \{%r\d+, %r\d+, %r\d+, %r\d+\}, ", pair)

    def test_swizzle_is_the_xor_on_the_element_index(self) -> None:
        ptx = lowered(slab(), frag("_a", "a"), ldmatrix(swizzle="B128"))
        assert re.search(r"shr\.b32 (%r\d+), (%r\d+), 6;\n\tand\.b32 (%r\d+), \1, 0x7;\n\tshl\.b32 (%r\d+), \3, 3;\n\txor\.b32 %r\d+, \2, \4;", ptx)

    def test_only_a_staged_half_slab(self) -> None:
        with pytest.raises(Unsupported, match="staged"):
            lowered(slab(), frag("_a", "a"), ldmatrix(staged=False))
        with pytest.raises(Unsupported, match="16-bit"):
            lowered(slab(dtype="float"), frag("_a", "a"), ldmatrix())


class TestSwizzledFill:
    def test_cp_async_fills_through_the_same_xor(self) -> None:
        copy = CpAsyncCopy(smem="s", smem_index=ZERO, src="x", src_index=ZERO, nbytes=16, swizzle="B64",
                           lane_index=(Var("a0"), Literal(0, "int")), lane_rows=None)
        ptx = lowered(slab(), copy, args=[X16, OUT16])
        assert re.search(r"and\.b32 %r\d+, %r\d+, 0x3;\n\tshl\.b32", ptx) and "cp.async.cg.shared.global" in ptx

    def test_a_32_bit_slab_cannot_be_swizzled(self) -> None:
        copy = CpAsyncCopy(smem="s", smem_index=ZERO, src="x", src_index=ZERO, nbytes=16, swizzle="B128", lane_index=None, lane_rows=None)
        with pytest.raises(Unsupported, match="f32 elements"):
            lowered(slab(dtype="float"), copy, args=[X32, OUT32])


class TestRegStore:
    def test_two_pairs_per_lane_rows_g_and_g_plus_8(self) -> None:
        store = RegStore(dst_buffer="out", dst_index=ZERO, frag="_c", shape=SHAPE)
        ptx = lowered(frag("_c", "c"), store)
        assert ptx.count("cvt.rn.f16.f32") == 4 and ptx.count("st.global.v2.b16") == 2
        assert re.search(r"st\.global\.v2\.b16 \[%rd\d+\+128\]", ptx)  # row g + 8 of an 8-wide half matrix: 8 * 8 * 2 bytes
        wide = lowered(frag("_c", "c"), store, args=[OUT32])
        assert wide.count("st.global.v2.f32") == 2 and re.search(r"st\.global\.v2\.f32 \[%rd\d+\+256\]", wide)

    def test_epilogues_and_guards_are_unsupported(self) -> None:
        guarded = RegStore(dst_buffer="out", dst_index=ZERO, frag="_c", shape=SHAPE, m_guard=(Literal(0, "int"), Literal(16, "int")))
        with pytest.raises(Unsupported, match="guards"):
            lowered(frag("_c", "c"), guarded)


class TestWholeWarps:
    def test_partial_warps_and_guarded_tiles_are_refused(self) -> None:
        with pytest.raises(InvalidKernel, match="whole warps"):
            lowered(frag("_c", "c"), block=16, grid=2)
        with pytest.raises(InvalidKernel, match="guarded tile"):
            lowered(frag("_c", "c"), block=64)

    def test_needs_sm_80(self) -> None:
        with pytest.raises(Unsupported, match="sm_80"):
            lowered(frag("_c", "c"), target=Target(sm=75, isa="8.0"))


class TestEmmyMatmul:
    """emmy's f16 matmul (tests/problems/m50_f16_mm): the whole path, as the GPU test runs it."""

    def test_is_all_tensor_cores(self) -> None:
        ptx = compile_file(next(problem("m50_f16_mm").glob("*.ir.json")))
        assert ptx.count("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32") == 64  # 4 k-steps x 4 k16 x 4 column fragments
        assert ptx.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 16 and ptx.count("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16") == 32
        assert ptx.count("st.global.v2.b16") == 8 and "ld.shared" not in ptx and "fma" not in ptx

    def test_rolled_fragment_loops_are_refused(self) -> None:
        with pytest.raises(Unsupported, match="not unrolled"):
            compile_file(next(problem("m50_f16_mm").glob("*.ir.json")), Options(unroll="none"))
