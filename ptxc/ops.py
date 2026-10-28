"""Elementwise ops: name -> recipe.

Numerics are pinned, not configured: `.rn` on float add / sub / mul (IEEE, and it keeps ptxas
from contracting mul+add into fma), explicit `fma.rn` inside the polynomial recipes, exp / log /
rsqrt through the `.approx` instructions (a few ulp), erf / sin / cos / tanh as polynomials at
about 2 ulp (`tanh.approx` is only 2^-11 accurate and `sin.approx` has no range reduction).

A recipe takes the result type and its arguments already converted to it. A `raw` recipe takes
the arguments as they come and converts what it needs: comparisons unify their operands and
return a predicate, `where` selects on one, masks follow C's truth rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .errors import Unsupported
from .ptx import Reg, Value, bitcast, cvt, emit, op, reg, setp, truth, unify
from .types import B32, F32, PRED, S32, U32, Kind, Ty

type Recipe = Callable[..., Value]  # (ty: Ty, *args: Value) -> Value

OPS: dict[str, Recipe] = {}
RAW: set[str] = set()

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
TWO_OVER_PI = 0.6366197723675814
PIO2 = (1.5707962512969971e00, 7.5497894158615964e-08, 5.3903029534742385e-15)  # pi/2 in three f32 parts


def recipe(*names: str, raw: bool = False) -> Callable[[Recipe], Recipe]:
    def register(fn: Recipe) -> Recipe:
        for name in names:
            OPS[name] = fn
            if raw:
                RAW.add(name)
        return fn

    return register


def apply(name: str, ty: Ty, args: Sequence[Value]) -> Value:
    """`name(args)` with a result of type `ty`."""
    if name not in OPS:
        raise Unsupported(f"elementwise op {name!r}")
    if name not in RAW:
        args = [cvt(a, ty) for a in args]
    return cvt(OPS[name](ty, *args), ty)  # a raw recipe returns what it computed: a predicate, raw bits


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
    return op("add.rn" if ty.is_float else "add", ty, x, y)


@recipe("subtract")
def subtract(ty: Ty, x: Value, y: Value) -> Value:
    return op("sub.rn" if ty.is_float else "sub", ty, x, y)


@recipe("multiply")
def multiply(ty: Ty, x: Value, y: Value) -> Value:
    return op("mul.rn" if ty.is_float else "mul.lo", ty, x, y)


@recipe("divide", "true_divide")
def divide(ty: Ty, x: Value, y: Value) -> Value:
    return op("div.rn" if ty.is_float else "div", ty, x, y)


@recipe("remainder", "mod")
def remainder(ty: Ty, x: Value, y: Value) -> Value:
    if ty.is_float:
        raise Unsupported("float remainder")
    return op("rem", ty, x, y)


@recipe("maximum")
def maximum(ty: Ty, x: Value, y: Value) -> Value:
    return op("max", ty, x, y)


@recipe("minimum")
def minimum(ty: Ty, x: Value, y: Value) -> Value:
    return op("min", ty, x, y)


@recipe("negative")
def negative(ty: Ty, x: Value) -> Value:
    return op("neg", ty, x)


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
    return op("selp", ty, cvt(a, ty), cvt(b, ty), truth(cond))


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
    return op(floating("rcp.rn", ty), ty, x)


@recipe("sqrt")
def sqrt(ty: Ty, x: Value) -> Value:
    return op(floating("sqrt.rn", ty), ty, x)


@recipe("rsqrt")
def rsqrt(ty: Ty, x: Value) -> Value:
    return op(floating("rsqrt.approx", ty), ty, x)


@recipe("exp", "exp_fast")
def exp(ty: Ty, x: Value) -> Value:
    scaled = op("mul.rn", ty, x, ty(LOG2E))
    return op(floating("ex2.approx", ty, f64=False), ty, scaled)


@recipe("log")
def log(ty: Ty, x: Value) -> Value:
    lg2 = op(floating("lg2.approx", ty, f64=False), ty, x)
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
    """|x| < 0.55: x + x^3 P(x^2), a minimax fit; above: 1 - 2 / (exp(2|x|) + 1). About 2 ulp."""
    require_f32("tanh", ty)
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


def trig(ty: Ty, x: Value, cosine: bool) -> Value:
    """sin or cos from the reduced argument: the quadrant picks the polynomial and the sign."""
    require_f32("cos" if cosine else "sin", ty)
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
