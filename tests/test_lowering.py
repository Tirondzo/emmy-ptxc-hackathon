"""Lowering decisions read off the PTX text, no GPU. Each kernel is emmy IR built in Python
(kernels.py): the shapes the IR admits but the problem sets never produce."""

from __future__ import annotations

import re

import pytest
from emmy.compiler import dtype as D
from emmy.compiler.ir.axis import Axis, Dim
from emmy.compiler.ir.elementwise import ElementwiseImpl as Op
from emmy.compiler.ir.expr import BinaryExpr, Builtin, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Tile
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let, Load, Write
from kernels import arg, load, ptx_of, shuffle, tile, write

from ptxc import ir
from ptxc.errors import InvalidKernel, Unsupported
from ptxc.lower.block import halving
from ptxc.types import F32, F64, S32, S64, U32, U64, ty_of

X, Y, OUT = arg("x", "f32"), arg("y", "f32"), arg("out", "f32")  # names and types only: nothing here is launched


def lowered(*body, args=(X, Y, OUT), **launch) -> str:
    return ptx_of([tile(32, *body)], args, **launch)


class TestBuiltins:
    @pytest.mark.parametrize(("name", "sreg"), [("thread_idx.x", "%tid.x"), ("block_idx.x", "%ctaid.x"),
                                                ("block_dim.x", "%ntid.x"), ("grid_dim.x", "%nctaid.x")])
    def test_emmy_spellings(self, name: str, sreg: str) -> None:
        ptx = lowered(Let("t", Builtin(name), dtype=D.I32), write("out", "t"), args=[arg("out", "i32")])
        assert re.search(rf"mov\.u32 %r\d+, {re.escape(sreg)};", ptx)

    def test_warp_size_is_a_constant(self) -> None:
        ptx = lowered(Let("t", BinaryExpr("*", Builtin("warp_size"), Literal(2, "int")), dtype=D.I32), write("out", "t"),
                      args=[arg("out", "i32")])
        assert re.search(r"mov\.s32 %r\d+, 64;", ptx)

    def test_cuda_spellings_are_refused(self) -> None:
        with pytest.raises(Unsupported, match="blockIdx.x"):
            lowered(Let("t", Builtin("blockIdx.x")), args=[OUT])


class TestTypes:
    @pytest.mark.parametrize(("spelling", "ty"), [("long long", S64), ("unsigned int", U32), ("unsigned long long", U64),
                                                  ("int", S32), ("float", F32), ("double", F64)])
    def test_c_spellings(self, spelling: str, ty) -> None:
        assert ty_of(spelling) is ty

    def test_smem_in_a_c_spelling(self) -> None:
        ptx = lowered(Smem("scratch", (32,), "long long"), args=[OUT])
        assert ".shared .align 8 .b8 scratch[256];" in ptx

    def test_pool_is_declared_once_at_the_largest_alignment(self) -> None:
        """Two Smems in a dynamic pool: one `.extern` declaration, aligned as the most demanding array
        asks, the second array at the first's rounded end."""
        ptx = lowered(Smem("a", (32,), "float", align=32), Smem("b", (4,), "double"),
                      Load(names=("v",), input="b", index=(Literal(0, "int"),), dtype=None), write("out", "v"),
                      args=[OUT], smem_bytes=256)
        assert re.findall(r"^\.(?:extern )?\.shared .*$", ptx, re.M) == [".extern .shared .align 32 .b8 _smem_pool[];"]
        assert re.search(r"ld\.shared\.f64 %fd\d+, \[%r\d+\+128\];", ptx)


class TestTile:
    def test_decode_is_unsigned(self) -> None:
        """64 threads over 4 x 8 cells: the guard and the axis decode are `u32`; `%tid.x` is read once."""
        cells = Tile(axes=(Axis("i", Dim(4)), Axis("j", Dim(8))), body=(
            Let("t", BinaryExpr("+", Var("i"), Var("j")), dtype=D.I32),
            Write(output="out", index=(Var("i"), Var("j")), values=("t",), value_dtype=None, atomic=False, swizzle="NONE")))
        ptx = ptx_of([cells], [{"name": "out", "dtype": "i32", "shape": [4, 8], "output": True}], block=64)
        assert re.search(r"setp\.lt\.u32 %p\d+, %r\d+, 32;", ptx)
        assert re.search(r"div\.u32 %r\d+, %r\d+, 8;", ptx) and re.search(r"rem\.u32 %r\d+, %r\d+, 8;", ptx)
        assert "div.s32" not in ptx and "rem.s32" not in ptx
        assert ptx.count("%tid.x") == 1

    def test_maxntid_is_the_block(self) -> None:
        assert ".maxntid 128, 1, 1\n{" in lowered(args=[OUT], block=128)


class TestOps:
    @pytest.mark.parametrize(("alias", "canonical"), [("sum", "add"), ("prod", "multiply"), ("amax", "maximum"),
                                                      ("fmax", "maximum"), ("fmin", "minimum")])
    def test_reduce_aliases(self, alias: str, canonical: str) -> None:
        assert ir.op_name(Op(alias)) == canonical

    @pytest.mark.parametrize(("alias", "opcode"), [("sum", "add.rn.f32"), ("prod", "mul.rn.f32"), ("amax", "max.f32")])
    def test_accum_through_an_alias(self, alias: str, opcode: str) -> None:
        """`Accum(op=amax)` as a trace leaves it: seeded with the alias's identity, combined with the canonical op."""
        loop = Loop(axis=Axis("k", Dim(4)), body=(
            Load(names=("v",), input="x", index=(Var("a0"), Var("k")), dtype=None), Accum("acc", "v", Op(alias), dtype=D.F32)))
        ptx = lowered(loop, write("out", "acc"), args=[{"name": "x", "dtype": "f32", "shape": [32, 4], "output": True}, OUT])
        assert opcode in ptx

    def test_floor_divide_is_integer_division(self) -> None:
        ints = [arg("x", "i32"), arg("y", "i32"), arg("out", "i32")]
        ptx = lowered(load("x", "x"), load("y", "y"), Assign("q", Op("floor_divide"), ("x", "y"), dtype=D.I32), write("out", "q"), args=ints)
        assert re.search(r"div\.s32 %r\d+, %r\d+, %r\d+;", ptx)
        with pytest.raises(Unsupported, match="floor_divide"):
            lowered(load("x", "x"), load("y", "y"), Assign("q", Op("floor_divide"), ("x", "y"), dtype=D.F32), write("out", "q"))

    @pytest.mark.parametrize(("operator", "dtype", "cmp"), [("!=", "f32", "neu.f32"), ("!=", "i32", "ne.s32"), ("==", "f32", "eq.f32"),
                                                            ("<=", "f32", "le.f32"), (">", "i32", "gt.s32")])
    def test_comparisons_go_through_one_table(self, operator: str, dtype: str, cmp: str) -> None:
        """An expression's `!=` on floats is unordered like the `not_equal` op: C's `!=`."""
        args = [arg("x", dtype), arg("y", dtype), arg("out", "bool")]
        ptx = lowered(load("x", "x"), load("y", "y"), Let("c", BinaryExpr(operator, Var("x"), Var("y"))), write("out", "c"), args=args)
        assert f"setp.{cmp} " in ptx

    def test_nothing_selects_or_shuffles_a_predicate(self) -> None:
        """`selp` and `shfl` have no `.pred` form: bool arms and states ride as 1 / 0."""
        ptx = lowered(load("x", "x"), load("y", "y"),
                      Let("c", BinaryExpr(">", Var("x"), Var("y"))), Let("p", BinaryExpr("<", Var("x"), Literal(0.0, "float"))),
                      Assign("w", Op("where"), ("c", "p", "c"), dtype=D.BOOL), shuffle(32, w="bitwise_or"), write("out", "w"),
                      args=[X, Y, arg("out", "bool")])
        assert "selp.u32" in ptx and "shfl.sync.bfly.b32" in ptx
        assert not re.search(r"(selp|shfl)[\w.]*\.pred", ptx)

    def test_negative_of_unsigned_is_the_signed_one(self) -> None:
        ptx = lowered(load("u", "x", dtype=D.U32), Assign("n", Op("negative"), ("u",), dtype=D.U32), write("out", "n"),
                      args=[arg("x", "i32"), arg("out", "i32")])
        assert "neg.s32" in ptx and "neg.u32" not in ptx


class TestShuffle:
    @staticmethod
    def lowered(length: int) -> str:
        return lowered(load("v", "x"), shuffle(length, v="add"), write("out", "v"), args=[X, OUT])

    @pytest.mark.parametrize("length", [2, 4, 8, 16, 32])
    def test_power_of_two_is_a_butterfly(self, length: int) -> None:
        ptx = self.lowered(length)
        steps = re.findall(r"shfl\.sync\.bfly\.b32 %r\d+, %r\d+, (\d+), (0x[0-9a-f]+)", ptx)
        assert [int(delta) for delta, _ in steps] == list(halving(length))
        assert {clamp for _, clamp in steps} == {hex(((32 - length) << 8) | 0x1F)}  # every step stays inside its group
        assert "shfl.sync.down" not in ptx and "shfl.sync.idx" not in ptx

    @pytest.mark.parametrize("length", [3, 5, 6, 7, 12, 24, 31])
    def test_other_lengths_fold_onto_lane_0(self, length: int) -> None:
        """No butterfly reaches every lane: `shfl.down` by each power of two, taken only where
        `lane + delta < length`, then lane 0 is broadcast."""
        ptx = self.lowered(length)
        deltas = [int(d) for d in re.findall(r"shfl\.sync\.down\.b32 %r\d+, %r\d+, (\d+), 0x1f", ptx)]
        assert deltas == list(halving(length))
        assert re.findall(r"setp\.lt\.s32 %p\d+, %r\d+, (\d+);", ptx) == [str(length)] * len(deltas)
        assert len(re.findall(r"shfl\.sync\.idx\.b32 %r\d+, %r\d+, 0, 0x1f", ptx)) == 1
        assert "bfly" not in ptx

    def test_longer_than_a_warp_is_refused(self) -> None:
        with pytest.raises(InvalidKernel, match="64"):
            self.lowered(64)
