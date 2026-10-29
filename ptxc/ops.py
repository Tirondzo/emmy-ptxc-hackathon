"""Elementwise ops: name -> recipe, and the error budget the tolerance leaves them.

A recipe with more than one spelling asks the ambient `Budget` whether the faster one still fits,
`allows(ulps, absolute)`, with the worst case the PTX ISA states for it: add / sub / mul without
`.rn` (so ptxas may fuse mul + add into fma), `div.full` / `rcp.approx` / `sqrt.approx`,
`tanh.approx`, `sin.approx` on a reduced or a raw argument, `.ftz` on the approximate instructions.
Everything else is pinned: explicit `fma.rn` inside the polynomial recipes, exp / log / rsqrt through
the `.approx` instructions (2 ulp at the instruction; exp's error grows with |x|, about |x| ulp, as
`__expf`'s does, since x log2e is rounded before `ex2`), erf as a polynomial at about 1 ulp.

A recipe takes the result type and its arguments already converted to it. A `raw` recipe takes
the arguments as they come and converts what it needs: comparisons unify their operands and
return a predicate, `where` selects on one, masks follow C's truth rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .context import lowering, options
from .errors import Unsupported
from .ir import Tolerance
from .options import Precision
from .ptx import Reg, Value, bitcast, cvt, emit, op, reg, select, setp, truth, unify
from .types import B32, F16, F32, PRED, S32, U32, Imm, Kind, Ty

type Recipe = Callable[..., Value]  # (ty: Ty, *args: Value) -> Value

OPS: dict[str, Recipe] = {}
RAW: set[str] = set()


# ---- the error budget the tolerance leaves -----------------------------------------------------

ULP = 2.0**-23  # one f32 ulp as a relative error, at its largest (just above a power of two)
SHARE = 4  # the recipes may spend 1/SHARE of the tolerance; the rest stays for the computation's own rounding
SUBNORMAL = 2.0**-126  # every f32 below this in magnitude is subnormal: what `.ftz` flushes to zero


@dataclass(frozen=True)
class Budget:
    """What a kernel's float recipes may spend on faster spellings, in the tolerance's own shape. A
    recipe asks `allows(ulps, absolute)` with the worst case the PTX ISA states for a spelling and
    takes it when the answer is yes; the IEEE spelling is the fallback. f32 only: f64 is always `.rn`.
    The kernel's budget sits on the ambient `Lowering`."""

    rtol: float = 0.0
    atol: float = 0.0

    @classmethod
    def decide(cls, precision: Precision, tolerance: Tolerance) -> Budget:
        """`exact` spends nothing, `fast` has no limit, `auto` 1/SHARE of the tolerance."""
        match precision:
            case "exact":
                return cls()
            case "fast":
                return cls(math.inf, math.inf)
        return cls(tolerance.rtol / SHARE, tolerance.atol / SHARE)

    def allows(self, ulps: float = 0.0, absolute: float = 0.0) -> bool:
        return ulps * ULP <= self.rtol and absolute <= self.atol

    def __str__(self) -> str:
        if self.rtol == math.inf:
            return "no limit"
        if not (self.rtol or self.atol):
            return "exact"
        return f"budget {self.rtol / ULP:.0f} ulp relative, {self.atol:.1e} absolute"


def allows(ulps: float = 0.0, absolute: float = 0.0) -> bool:
    """Whether a spelling that is off by up to `ulps` relative and `absolute` fits the kernel's budget."""
    return lowering().budget.allows(ulps, absolute)


def rounded(name: str, ty: Ty) -> str:
    """`name.rn`, or `name` when the budget allows the 1 ulp of fma contraction; integers as they are."""
    if not ty.is_float:
        return name
    return name if ty == F32 and allows(ulps=1) else f"{name}.rn"


def ftz(ty: Ty) -> str:
    """`.ftz` flushes subnormals to zero, the fast path of the approximate instructions."""
    return ".ftz" if ty == F32 and allows(absolute=SUBNORMAL) else ""


LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
TWO_OVER_PI = 0.6366197723675814
PIO2 = (1.5707962512969971e00, 7.5497894158615964e-08, 5.3903029534742385e-15)  # pi/2 in three f32 parts


def recipe(*names: str, raw: bool = False) -> Callable[[Recipe], Recipe]:
    def register(fn: Recipe) -> Recipe:
        for name in names:
            OPS[name] = fn
            if raw: # a raw recipe returns what it computed: a predicate, raw bits
                RAW.add(name)
        return fn

    return register


def apply(name: str, ty: Ty, args: Sequence[Value]) -> Value:
    """`name(args)` with a result of type `ty`. An f16 result is computed in f32 and rounded once at the
    end: PTX half arithmetic takes no constants, and f32 is wide enough that + - * / round the same."""
    if name not in OPS:
        raise Unsupported(f"elementwise op {name!r}")
    compute = F32 if ty == F16 else ty
    if name not in RAW:
        args = [cvt(a, compute) for a in args]
    return cvt(OPS[name](compute, *args), ty)  # a raw recipe returns what it computed: a predicate, raw bits


def floating(name: str, ty: Ty, f64: bool = True) -> str:
    """`name` for a float type; f64 only where PTX spells it."""
    if ty.is_float and (ty.bits == 32 or f64):
        return name
    raise Unsupported(f"{name} on {ty.ptx}")


def require_f32(what: str, ty: Ty) -> None:
    """The polynomial recipes and the `.approx` instructions they lean on exist for f32 only."""
    if ty != F32:
        raise Unsupported(f"{what} on {ty.ptx}")


def fma(ty: Ty, a: Value, b: Value, c: Value) -> Reg:
    return op(floating("fma.rn", ty), ty, a, b, c)


def horner(ty: Ty, x: Value, coeffs: Sequence[float]) -> Value:
    """`coeffs` from the highest power down, one fma per step."""
    p: Value = ty(coeffs[0])
    for c in coeffs[1:]:
        p = fma(ty, p, x, ty(c))
    return p


def rint(ty: Ty, x: Value) -> Reg:
    """Round to the nearest integer, still a float."""
    r = reg(ty)
    emit(f"cvt.rni.{ty.ptx}.{ty.ptx}", r, x)
    return r


# ---- int and float; the int spelling differs (mul.lo, div, no .rn) --------------------


@recipe("add")
def add(ty: Ty, x: Value, y: Value) -> Value:
    return op(rounded("add", ty), ty, x, y)


@recipe("subtract")
def subtract(ty: Ty, x: Value, y: Value) -> Value:
    return op(rounded("sub", ty), ty, x, y)


@recipe("multiply")
def multiply(ty: Ty, x: Value, y: Value) -> Value:
    """Integers keep the low half of the product (`mul.lo`), as C does."""
    return op(rounded("mul", ty) if ty.is_float else "mul.lo", ty, x, y)


@recipe("divide", "true_divide")
def divide(ty: Ty, x: Value, y: Value) -> Value:
    if ty == F32 and allows(ulps=2):  # div.full: 2 ulp over the whole range (div.approx only for |y| in [2^-126, 2^126])
        return op(f"div.full{ftz(ty)}", ty, x, y)
    return op("div.rn" if ty.is_float else "div", ty, x, y)


@recipe("floor_divide")
def floor_divide(ty: Ty, x: Value, y: Value) -> Value:
    """Python's `//`, as torch and emmy's own folder define it (its CUDA renders C's `/`, the same only for x >= 0)."""
    if k := shift(ty, y):
        return op("shr", ty, x, U32(k))
    q = op("div", ty, x, y)
    return op("sub", ty, q, cvt(down(op("rem", ty, x, y), y), ty)) if ty.kind is Kind.SIGNED else q


@recipe("remainder", "mod")
def remainder(ty: Ty, x: Value, y: Value) -> Value:
    """Python's `%`: the remainder takes the divisor's sign."""
    if shift(ty, y):
        bits = Ty(Kind.BITS, ty.bits)
        return cvt(op("and", bits, cvt(x, bits), bits(y.value - 1)), ty)
    r = op("rem", ty, x, y)
    return op("add", ty, r, select(ty, y, ty(0), down(r, y))) if ty.kind is Kind.SIGNED else r


def shift(ty: Ty, y: Value) -> int:
    """k when y is the literal 2^k, k > 0: in two's complement the floor is then a shift and a mask,
    cheaper than C's `/` and right for either sign. Every power-of-two index decode lands here."""
    if ty.is_float:
        raise Unsupported(f"float floor division on {ty.ptx}")
    if not isinstance(y, Imm) or y.value <= 1 or y.value & (y.value - 1):
        return 0
    return int(y.value).bit_length() - 1


def down(r: Reg, y: Value) -> Reg:
    """Where flooring steps one below C's quotient: a nonzero remainder of the other sign than the divisor."""
    if isinstance(y, Imm):
        return setp("lt" if y.value > 0 else "gt", r, r.ty(0))
    return op("and", PRED, setp("ne", r, r.ty(0)), op("xor", PRED, setp("lt", r, r.ty(0)), setp("lt", y, r.ty(0))))


@recipe("maximum")
def maximum(ty: Ty, x: Value, y: Value) -> Value:
    return op("max", ty, x, y)


@recipe("minimum")
def minimum(ty: Ty, x: Value, y: Value) -> Value:
    return op("min", ty, x, y)


@recipe("negative")
def negative(ty: Ty, x: Value) -> Value:
    """`neg` has no unsigned form; in two's complement the signed one is bit-identical."""
    signed = Ty(Kind.SIGNED, ty.bits) if ty.kind is Kind.UNSIGNED else ty
    return cvt(op("neg", signed, x), ty)


@recipe("abs", "absolute")
def absolute(ty: Ty, x: Value) -> Value:
    return op("abs", ty, x)


@recipe("copy", "pad")
def copy(ty: Ty, x: Value) -> Value:
    return x


@recipe("square")
def square(ty: Ty, x: Value) -> Value:
    return multiply(ty, x, x)


@recipe("relu")
def relu(ty: Ty, x: Value) -> Value:
    return op("max", ty, x, ty(0))


# ---- predicates: comparisons, where, masks ------------------------------------------------


def compare(cmp: str, unordered: str | None = None) -> Recipe:
    def fn(ty: Ty, a: Value, b: Value) -> Value:
        a, b = unify(a, b)
        return setp(unordered if unordered and a.ty.is_float else cmp, a, b)

    return fn


for _name, _cmp in {"equal": "eq", "greater": "gt", "less": "lt", "greater_equal": "ge", "less_equal": "le"}.items():
    recipe(_name, raw=True)(compare(_cmp))
recipe("not_equal", raw=True)(compare("ne", unordered="neu"))  # C's !=: true when either side is NaN


@recipe("where", raw=True)
def where(ty: Ty, cond: Value, a: Value, b: Value) -> Value:
    return select(ty, cvt(a, ty), cvt(b, ty), truth(cond))


@recipe("bitwise_not", raw=True)
def bitwise_not(ty: Ty, x: Value) -> Value:
    """Logical not, which is what emmy means by it on masks (`x == 0`)."""
    return op("not", PRED, truth(x))


def logical(opcode: str) -> Recipe:
    """`&&` / `||` on masks (bool or f32), the bitwise op on integers."""

    def fn(ty: Ty, a: Value, b: Value) -> Value:
        if ty.is_float or ty.kind is Kind.PRED:
            return op(opcode, PRED, truth(a), truth(b))
        bits = Ty(Kind.BITS, ty.bits)
        return op(opcode, bits, cvt(a, bits), cvt(b, bits))

    return fn


for _name, _opcode in {"bitwise_and": "and", "bitwise_or": "or", "bitwise_xor": "xor"}.items():
    recipe(_name, raw=True)(logical(_opcode))


# ---- float only --------------------------------------------------------------------


@recipe("reciprocal")
def reciprocal(ty: Ty, x: Value) -> Value:
    if ty == F32 and allows(ulps=1):  # rcp.approx: 1 ulp
        return op(f"rcp.approx{ftz(ty)}", ty, x)
    return op(floating("rcp.rn", ty), ty, x)


@recipe("sqrt")
def sqrt(ty: Ty, x: Value) -> Value:
    if ty == F32 and allows(ulps=1):  # sqrt.approx: 2^-23 relative
        return op(f"sqrt.approx{ftz(ty)}", ty, x)
    return op(floating("sqrt.rn", ty), ty, x)


@recipe("rsqrt")
def rsqrt(ty: Ty, x: Value) -> Value:
    return op(floating(f"rsqrt.approx{ftz(ty)}", ty), ty, x)


@recipe("exp", "exp_fast")
def exp(ty: Ty, x: Value) -> Value:
    scaled = op("mul.rn", ty, x, ty(LOG2E))
    return op(floating(f"ex2.approx{ftz(ty)}", ty, f64=False), ty, scaled)


@recipe("log")
def log(ty: Ty, x: Value) -> Value:
    lg2 = op(floating(f"lg2.approx{ftz(ty)}", ty, f64=False), ty, x)
    return op("mul.rn", ty, lg2, ty(LN2))


@recipe("sigmoid")
def sigmoid(ty: Ty, x: Value) -> Value:
    return reciprocal(ty, add(ty, ty(1), exp(ty, negative(ty, x))))


SQRT_HALF_BITS = 0x3F3504F3  # sqrt(1/2) as f32 bits: the mantissa window is [sqrt 1/2, sqrt 2)
LOG2_SERIES = (0.32059889797532519, 0.41219857740970382, 0.5770780163555854, 0.9617966939259756)  # 2 log2e / (2k+1), k = 4..1
TWO_LOG2E = 2.8853900817779268


def log2_split(ty: Ty, x: Value) -> tuple[Value, Value]:
    """log2 of a positive normal x as an unevaluated sum hi + lo, accurate to about 2^-30 absolute:
    the exact exponent plus log2 of the mantissa m in [sqrt 1/2, sqrt 2) from the atanh series in
    t = (m - 1) / (m + 1). `lg2.approx` is 2^-22.6, which pow would multiply by its exponent."""
    i = op("sub", S32, cvt(bitcast(x, B32), S32), S32(SQRT_HALF_BITS))
    e = cvt(op("shr", S32, i, U32(23)), ty)
    m = bitcast(op("add", S32, cvt(op("and", B32, cvt(i, B32), B32(0x007FFFFF)), S32), S32(SQRT_HALF_BITS)), ty)
    t = op("div.rn", ty, op("sub.rn", ty, m, ty(1)), op("add.rn", ty, m, ty(1)))
    s = op("mul.rn", ty, t, t)
    lm = fma(ty, op("mul.rn", ty, t, s), horner(ty, s, LOG2_SERIES), op("mul.rn", ty, t, ty(TWO_LOG2E)))
    hi = op("add.rn", ty, e, lm)  # Fast2Sum: |e| >= |lm| unless e == 0, so hi + lo == e + lm exactly
    lo = op("sub.rn", ty, lm, op("sub.rn", ty, hi, e))
    return hi, lo


@recipe("pow", "power")
def power(ty: Ty, x: Value, y: Value) -> Value:
    """powf as C defines it: 2^(y log2|x|) with the product kept to about 2^-30, so the result is
    a few ulp for any exponent; then the special values the bit games do not see (0, inf, NaN),
    the sign for an odd integer y, NaN for a negative base with a fractional y, 1 for y = 0."""
    require_f32("pow", ty)
    ax = op("abs", ty, x)
    hi, lo = log2_split(ty, ax)
    p = op("mul.rn", ty, y, hi)
    p_lo = fma(ty, y, lo, fma(ty, y, hi, op("neg", ty, p)))
    r = op("mul.rn", ty, op("ex2.approx", ty, p), fma(ty, p_lo, ty(LN2), ty(1)))
    y_negative = setp("lt", y, ty(0))
    r = op("selp", ty, op("selp", ty, ty(math.inf), ty(0), y_negative), r, setp("eq", ax, ty(0)))
    r = op("selp", ty, op("selp", ty, ty(0), ty(math.inf), y_negative), r, setp("eq", ax, ty(math.inf)))
    r = op("selp", ty, x, r, setp("nan", x, x))
    yi = rint(ty, y)
    integer = setp("eq", yi, y)
    half = op("mul.rn", ty, y, ty(0.5))
    odd = op("and", PRED, integer, setp("ne", rint(ty, half), half))
    negative_base = setp("lt", x, ty(0))
    r = op("selp", ty, op("neg", ty, r), r, op("and", PRED, negative_base, odd))
    r = op("selp", ty, ty(math.nan), r, op("and", PRED, negative_base, op("not", PRED, integer)))
    return op("selp", ty, ty(1), r, setp("eq", y, ty(0)))


TANH = (-0.006287672556936741, 0.021081697195768356, -0.05385489761829376, 0.13332612812519073, -0.33333319425582886)


@recipe("tanh")
def tanh(ty: Ty, x: Value) -> Value:
    """`tanh.approx` (one instruction, 2^-11 relative) when the budget has room for it, which needs
    rtol >= 2e-3. Else |x| < 0.55: x + x^3 P(x^2), a minimax fit; above: 1 - 2 / (exp(2|x|) + 1).
    About 2 ulp."""
    require_f32("tanh", ty)
    if allows(ulps=2**12):
        return op("tanh.approx", ty, x)
    t = op("abs", ty, x)
    e = exp(ty, op("mul.rn", ty, t, ty(2)))
    tail = fma(ty, ty(-2), op("rcp.rn", ty, op("add.rn", ty, e, ty(1))), ty(1))
    tail = op("copysign", ty, x, tail)
    s = op("mul.rn", ty, x, x)
    core = fma(ty, op("mul.rn", ty, horner(ty, s, TANH), s), x, x)
    return op("selp", ty, tail, core, setp("ge", t, ty(0.55)))


ERF_CORE = (-5.96761703e-4, 4.99119423e-3, -2.67681349e-2, 1.12819925e-1, -3.76125336e-1, 1.28379166e-1)


@recipe("erf")
def erf(ty: Ty, x: Value) -> Value:
    """Faithfully rounded erff (N. Juffa): odd polynomial in x below 0.921875, 1 - exp(p(|x|)) above."""
    require_f32("erf", ty)
    t = op("abs", ty, x)
    s = op("mul.rn", ty, x, x)
    r = fma(ty, ty(-1.72853470e-5), t, ty(3.83197126e-4))
    u = fma(ty, ty(-3.88396438e-3), t, ty(2.42546219e-2))
    r = fma(ty, r, s, u)
    r = fma(ty, r, t, ty(-1.06777877e-1))
    r = fma(ty, r, t, ty(-6.34846687e-1))
    r = fma(ty, r, t, ty(-1.28717512e-1))
    r = fma(ty, r, t, op("neg", ty, t))
    tail = op("copysign", ty, x, op("sub.rn", ty, ty(1), exp(ty, r)))
    core = fma(ty, horner(ty, s, ERF_CORE), x, x)
    return op("selp", ty, tail, core, setp("gt", t, ty(0.921875)))


SIN = (-1.95152959e-4, 8.33216087e-3, -1.66666546e-1)
COS = (2.44331571e-5, -1.38873163e-3, 4.16666418e-2, -0.5)


def reduce_trig(ty: Ty, x: Value) -> tuple[Value, Reg]:
    """x = r + q * pi/2 with |r| <= pi/4: Cody-Waite in three parts, exact to about |x| < 1e5."""
    j = rint(ty, op("mul.rn", ty, x, ty(TWO_OVER_PI)))
    r: Value = x
    for part in PIO2:
        r = fma(ty, j, ty(-part), r)
    q = reg(S32)
    emit(f"cvt.rzi.s32.{ty.ptx}", q, j)
    return r, q


TWO_PI = (6.2831854820251465, -1.7484555314695172e-07)  # 2 pi in two f32 parts


def reduce_turns(ty: Ty, x: Value) -> Value:
    """x - rint(x / 2pi) * 2pi, into [-pi, pi]; exact to about |x| < 1e5 with 2 pi in two parts."""
    k = rint(ty, op("mul.rn", ty, x, ty(1 / (2 * math.pi))))
    return fma(ty, k, ty(-TWO_PI[1]), fma(ty, k, ty(-TWO_PI[0]), x))


def trig(ty: Ty, x: Value, cosine: bool) -> Value:
    """Three spellings, the cheapest that fits first. `sin.approx` on the raw argument: absolute 2^-14.7,
    which the ISA promises only within 100 pi, so only when the caller vouches for the range
    (`Options.raw_trig`). `sin.approx` after reducing the argument into [-pi, pi]: absolute 2^-20.4
    (cos's bound, the looser), three instructions more. Else Cody-Waite reduction and the polynomials:
    the quadrant picks the polynomial and the sign, about 1 ulp."""
    name = "cos" if cosine else "sin"
    require_f32(name, ty)
    if options().raw_trig and allows(absolute=2**-14.7):
        return op(f"{name}.approx{ftz(ty)}", ty, x)
    if allows(absolute=2**-20.4):
        return op(f"{name}.approx{ftz(ty)}", ty, reduce_turns(ty, x))
    r, q = reduce_trig(ty, x)
    s = op("mul.rn", ty, r, r)
    sin_r = fma(ty, op("mul.rn", ty, horner(ty, s, SIN), s), r, r)
    cos_r = fma(ty, horner(ty, s, COS), s, ty(1))
    quadrant = cvt(q, B32)
    odd = setp("ne", op("and", B32, quadrant, B32(1)), B32(0))
    v = op("selp", ty, sin_r, cos_r, odd) if cosine else op("selp", ty, cos_r, sin_r, odd)
    phase = cvt(op("add", S32, q, S32(1)), B32) if cosine else quadrant  # sign flips in quadrants 2, 3 (cos: 1, 2)
    flip = setp("ne", op("and", B32, phase, B32(2)), B32(0))
    return op("selp", ty, op("neg", ty, v), v, flip)


@recipe("sin")
def sin(ty: Ty, x: Value) -> Value:
    return trig(ty, x, cosine=False)


@recipe("cos")
def cos(ty: Ty, x: Value) -> Value:
    return trig(ty, x, cosine=True)
