"""Lowering decisions read off the PTX text, no GPU. Each kernel is emmy IR built in Python
(kernels.py): the shapes the IR admits but the problem sets never produce."""

from __future__ import annotations

import re

import pytest
from emmy.compiler import dtype as D
from emmy.compiler.ir.axis import Axis, Dim
from emmy.compiler.ir.elementwise import ElementwiseImpl as Op
from emmy.compiler.ir.expr import BinaryExpr, Builtin, FlatIndex, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Tile
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Select, SelectBranch, Write
from kernels import arg, load, ptx_of, shuffle, tile, write

from ptxc import ir
from ptxc.errors import InvalidKernel, Unsupported
from ptxc.lower.block import halving
from ptxc.types import F16, F32, F64, S32, S64, U32, U64, ty_of

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
                                                  ("int", S32), ("float", F32), ("double", F64), ("__half", F16)])
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
        with pytest.raises(Unsupported, match="floor division"):
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


class TestHalf:
    """f16 is a memory type: `.b16` registers between `ld` / `st` and a `cvt`, every op computed in f32 and
    rounded once where the IR says f16, no `.f16` arithmetic (which takes no constants)."""

    X, OUT = arg("x", "f16"), arg("out", "f16")

    def test_ops_compute_in_f32_and_round_once(self) -> None:
        ptx = lowered(load("v", "x"), Assign("e", Op("exp"), ("v",), dtype=D.F16), write("out", "e"), args=[self.X, self.OUT])
        assert re.search(r"ld\.global\.b16 %rs\d+, \[%rd\d+\];", ptx)
        assert "cvt.f32.f16" in ptx and "ex2.approx.f32" in ptx and "cvt.rn.f16.f32" in ptx
        assert re.search(r"st\.global\.b16 \[%rd\d+\], %rs\d+;", ptx)
        assert all(line.lstrip().startswith("cvt.") for line in ptx.splitlines() if ".f16" in line)

    def test_pairs_move_as_one_vector(self) -> None:
        pair = Write(output="out", index=(BinaryExpr("*", Var("a0"), Literal(2, "int")),), values=("v", "w"),
                     value_dtype=None, atomic=False, swizzle="NONE")
        ptx = lowered(load("v", "x"), Assign("w", Op("negative"), ("v",), dtype=D.F16), pair, args=[self.X, arg("out", "f16", n=64)])
        assert re.search(r"st\.global\.v2\.b16 \[%rd\d+\], \{%rs\d+, %rs\d+\};", ptx)

    def test_constants_are_moved_as_bits(self) -> None:
        """PTX has no half constant: the identity of a max over halves is the bits of -inf, which is what
        emmy's -1e30 rounds to in f16."""
        loop = Loop(axis=Axis("k", Dim(4)), body=(
            Load(names=("v",), input="x", index=(Var("a0"), Var("k")), dtype=None), Accum("m", "v", Op("amax"), dtype=D.F16)))
        ptx = lowered(loop, write("out", "m"), args=[{"name": "x", "dtype": "f16", "shape": [32, 4], "output": True}, self.OUT])
        assert re.search(r"mov\.b16 %rs\d+, 0xFC00;", ptx) and "max.f32" in ptx

    def test_shuffles_ride_in_a_b32(self) -> None:
        ptx = lowered(load("m", "x"), shuffle(32, m="amax"), write("out", "m"), args=[self.X, self.OUT])
        assert "cvt.u32.u16" in ptx and "shfl.sync.bfly.b32" in ptx and "cvt.u16.u32" in ptx

    def test_predicates_come_through_f32(self) -> None:
        ptx = lowered(load("v", "x"), Let("p", BinaryExpr(">", Var("v"), Literal(0.0, "float"))),
                      Assign("w", Op("where"), ("p", "v", "v"), dtype=D.F16), write("out", "w"), args=[self.X, self.OUT])
        assert re.search(r"setp\.gt\.f32 %p\d+, %f\d+, 0f00000000;", ptx) and not re.search(r"setp\.\w+\.f16", ptx)


class TestBeyondTheContract:
    """The starred nodes and operators of the IR reference that ptxc lowers anyway."""

    def test_select_chains_from_the_default(self) -> None:
        """`Select` is emmy's chain of ternaries: one `selp` per branch but the last, which is the default."""
        branches = (SelectBranch("v", BinaryExpr("<", Var("a0"), Literal(8, "int"))),
                    SelectBranch("w", BinaryExpr("<", Var("a0"), Literal(16, "int"))), SelectBranch("z", Literal(1, "int")))
        ptx = lowered(load("v", "x"), load("w", "y"), Init("z", 0.0, dtype=D.F32), Select("m", branches), write("out", "m"))
        assert len(re.findall(r"selp\.f32 ", ptx)) == 2
        assert re.search(r"setp\.lt\.s32 %p\d+, %r\d+, 8;", ptx) and re.search(r"setp\.lt\.s32 %p\d+, %r\d+, 16;", ptx)
        assert not re.search(r"setp\.\w+\.s32 %p\d+, 1, ", ptx), "the default's predicate is not read"

    def test_flat_index_is_the_row_major_offset(self) -> None:
        """`FlatIndex(x, (i, 3))` into a [32, 4] buffer is `i * 4 + 3`, an affine form like any index."""
        x = {"name": "x", "dtype": "f32", "shape": [32, 4], "output": True}
        ptx = lowered(Let("f", FlatIndex("x", (Var("a0"), Literal(3, "int"))), dtype=D.I32), write("out", "f"), args=[x, arg("out", "i32")])
        assert re.search(r"mul\.lo\.s32 %r\d+, %r\d+, 4;", ptx) and re.search(r"add\.s32 %r\d+, %r\d+, 3;", ptx)

    @pytest.mark.parametrize(("operator", "opcode"), [("&", "and.b32"), ("|", "or.b32"), ("^", "xor.b32"), ("<<", "shl.b32"), (">>", "shr.s32")])
    def test_bitwise_index_operators(self, operator: str, opcode: str) -> None:
        ptx = lowered(Let("t", BinaryExpr(operator, Var("a0"), Literal(3, "int")), dtype=D.I32), write("out", "t"), args=[arg("out", "i32")])
        assert re.search(rf"{re.escape(opcode)} %r\d+, %r\d+, (0x)?3;", ptx)


class TestConversions:
    @pytest.mark.parametrize(("left", "right"), [("u", "s"), ("s", "u")])
    def test_mixed_signs_compare_unsigned(self, left: str, right: str) -> None:
        """C's usual arithmetic conversions: at one width unsigned wins, whichever side it is on."""
        ptx = lowered(load("u", "x", dtype=D.U32), load("s", "y", dtype=D.I32), Let("c", BinaryExpr("<", Var(left), Var(right))),
                      write("out", "c"), args=[arg("x", "i32"), arg("y", "i32"), arg("out", "bool")])
        assert "setp.lt.u32" in ptx and "setp.lt.s32" not in ptx

    @pytest.mark.parametrize("name", ["exp", "log", "sin", "cos", "tanh", "erf", "sigmoid"])
    def test_f64_transcendentals_are_polynomials(self, name: str) -> None:
        """No `.approx` instruction exists for f64: the recipes are reductions and `fma.rn.f64` series."""
        ptx = lowered(load("v", "x"), Assign("r", Op(name), ("v",), dtype=D.F64), write("out", "r"), args=[arg("x", "f64"), arg("out", "f64")])
        assert "fma.rn.f64" in ptx and ".approx" not in ptx


class TestSmem:
    def test_declared_once_in_an_unrolled_loop(self) -> None:
        """A `Smem` met in every copy of a straight loop is one `__shared__` array: declared once, rebound each time."""
        scratch = Smem("scratch", (32,), "float")
        loop = Loop(axis=Axis("k", Dim(2)), body=(
            scratch, Load(names=("v",), input="scratch", index=(Var("a0"),), dtype=None),
            Assign("w", Op("add"), ("v", "v"), dtype=D.F32),
            Write(output="scratch", index=(Var("a0"),), values=("w",), value_dtype=None, atomic=False, swizzle="NONE")), unroll=True)
        ptx = lowered(loop, args=[OUT])
        assert ptx.count(".shared .align 4 .b8 scratch[128];") == 1
        assert len(re.findall(r"mov\.u32 %r\d+, scratch;", ptx)) == 2

    def test_another_shape_under_the_same_name_is_refused(self) -> None:
        with pytest.raises(InvalidKernel, match="scratch"):
            lowered(Smem("scratch", (32,), "float"), Smem("scratch", (64,), "float"), args=[OUT])
