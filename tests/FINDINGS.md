# ptxc test findings, 2026-10-28

Two passes on one day: the first built the local test bench and found the gaps, the second fixed
them in the compiler. Local GPU: RTX 5080 (sm_120), ptxas 13.4 from the `nvidia-cuda-nvcc` wheel,
emmy-ml 0.3.11, torch 2.11+cu128. Baseline: commit `b4c3ea3`.

## Status after the fixes

| set | problems | pass | fail |
|---|---|---|---|
| open set (`emmy-hackathon-2026/problems`) | 15 | 15 | 0 |
| own set (`tests/problems`, generated with emmy from `tests/expressions_extra.py`) | 145 | 143 | 2 (`Select`, documented out of scope) |
| total | 160 | 158 | 2, both strict xfails in `KNOWN_GAPS` |

`python -m pytest tests -q`: 166 passed, 2 xfailed in 81 s (160 problems + 8 contract tests). No
problem that compiles produces a wrong result, writes out of bounds, modifies an input, or differs
between repeated launches (float atomics excepted). The hosted runner (RTX 5090) agrees with the
local runner on all 24 problems cross-checked through `submit.py`, 10 of them today with the new
ops (`//` tiles, masks, erf, sin, tanh, pow, 3-D softmax).

## What changed in the compiler

1. **`//` in index expressions** (`lower/expr.py`): emmy decodes a flattened tile index with
   `(i // w, i % w)` for every pointwise tile of rank >= 2 and for 3-D reductions and softmax.
   `//` now lowers and folds on constants. 8 problems. (Later: `//` and `%` floor as torch does,
   p91 to p94, where emmy's own CUDA truncates and fails.)
2. **Elementwise ops** (`ops.py`): `square`, `erf`, `sin`, `cos`, `pow`, and an accurate `tanh`;
   aliases `exp_fast`, `pad`, `power`, `absolute`. The transcendentals are polynomials with
   explicit `fma.rn`, because `tanh.approx.f32` is only 2^-11 (125 ulp measured) and
   `sin.approx` has no range reduction. `pow` computes log2 as an exact exponent plus the atanh
   series of the mantissa and keeps `y * log2|x|` as a double-float, so its error no longer grows
   with the exponent; it follows C's `powf` for 0, inf, NaN, negative bases and `y = 0`.
3. **bool** (`types.py`, `ptx/builder.py`, `ops.py`): a predicate register. The six comparisons
   produce it (`setp`, `!=` unordered as in C), `where` selects on it (`selp`), `bitwise_not` /
   `_and` / `_or` are C's logical `!`, `&&`, `||` on masks (bool or f32, as emmy renders them) and
   bitwise on integers. Conversions follow C: nonzero (NaN included) is true, true is 1. A
   comparison of two constants folds.
4. **Buffer accesses convert at the buffer's element type** (`lower/compute.py`), as a C
   assignment does: `(x > 0).float()` stores 1.0 / 0.0 into an f32 buffer instead of
   `st.global.v2.pred`; a bool buffer is one byte of 0 / 1 (`st.global.u8`), like C's `bool`.
   The hosted runner rejects dtype `bool` (`known: f16, f32, f64, i32, i64`), so the hidden set
   cannot contain one; the support is verified locally only.

## Accuracy (ulp against the correctly rounded f32 expected values)

| problem | op | max ulp | max abs | note |
|---|---|---|---|---|
| p79_erf_wide, p50_erf | erf | 1.0 | 6e-8 | N. Juffa's faithfully rounded erff |
| p81_sin_wide, p82_cos_wide | sin, cos of |x| up to 120 | 1.0 | 6e-8 | Cody-Waite in three parts, exact to about |x| < 1e5 |
| p84_tanh_wide, p16_tanh | tanh | 1.0 | 6e-8 | was 125 ulp with `tanh.approx` |
| p64, p85, p86, p87, p88 | pow (y = 3, 5, 4, -2, tensor in [-9, 9]) | 3 to 5 | | was 18 to 62 ulp with `lg2.approx` |
| p13_exp, p89_exp_wide | exp, |x| up to 32 | 2, 19 | | `ex2.approx(x log2e)`: relative error 2^-24 |x|, 5e-6 at the f32 limit |
| p14_log, p90_log_wide | log | 2 | 2e-7 | `lg2.approx` |
| p17_sigmoid, p40_silu | sigmoid, silu | 2 | | |
| p41_gelu, p42_gelu_tanh, p80_gelu_2d | gelu | | 2.4e-7 | ulp large only where 1 + erf cancels, as in CUDA |

Every tolerance in the open set is 1e-5 relative or looser for ops like these.

## Still open

- `Select` statements (matmul fused into matmul, a16 / m23): documented as absent from the hidden
  set; `Unsupported`.
- `exp` keeps the fast `ex2.approx` form: accurate enough for any f32 input at 1e-5 relative, and
  a Cody-Waite exp would add about five instructions to every softmax and attention inner loop.
- Float `remainder` / `mod`: emmy's own renderer rejects it, so it cannot be in the hidden set.
- Adjacent loads are not merged into `v2` / `v4` (the `optimize(module)` slot, see Performance).

## Performance (2026-10-28, evening)

Before: the open matmul ran 6.7x slower than emmy's CUDA (204 vs 30 us), big reductions 2x, the
attention kernels 1.1 to 1.3x. The gap was decomposed by hand-editing PTX and reading the
reference PTX out of nvrtc:

- the k-loop stayed rolled: ptxas gets no unroll hint from PTX and left a 64-iteration loop of
  ~70 instructions as it was; unrolling it in the compiler (axis bound to an immediate) gave 1.6x;
- every address was recomputed per access: `(_ks / 64) % 3` and `a1_b * 64 + a1_u * 2 - a1_b * 64`
  were lowered from scratch for each of the 4 loads of each of the 64 copies, ~3400 integer
  instructions per stage against ~10 in the reference. This was the remaining 4x and ptxas did not
  recover it either;
- the reference also merges adjacent shared loads into `v2` / `v4` (98 loads instead of 256) and
  contracts mul + add into fma. Neither is done yet.

What changed: `expr.affine` keeps an index as sum(coefficient * register) + constant (constants
fold into displacements, equal terms cancel); `lower/loops.py` unrolls by the policy in
`unroll_factor`, with the knobs in `ptxc/options.py` (`compile.py --unroll auto|none|force|QUOTA`;
the rest through `Options(...)`).
Rolled loops alone gained from the addressing: m01 204 -> 77 us with `--unroll=none`.

`python tests/bench.py --all` (RTX 5080, min over 3 runs of the median of 30 launches; small
kernels are launch-bound at about 7 us and noisy by 1 to 2 us; `none` is every loop rolled):

| problem | auto (default) | none | emmy reference |
|---|---|---|---|
| a01_attention | 65.2 us (255 regs, 352 B spill) | 87.9 us (128 regs) | 79.6 us (255 regs, 236 B spill) |
| a02_attention_128 | 25.3 us (80 regs) | 45.9 us | 37.2 us (77 regs) |
| m01_matmul_512 | 32.5 us (40 regs) | 76.7 us (30 regs) | 30.4 us (40 regs) |
| m02_matmul_rect | 9.2 us (40 regs) | 9.0 us | 9.4 us |
| r02_big_sum | 13.3 us (38 regs) | 25.5 us (19 regs) | 15.2 us (38 regs) |
| r03_mean | 9.2 us (38 regs) | 16.4 us | 11.2 us |
| f01 / f02 / f03, r01 / r04, p00 to p03 | 7 to 9 us, within noise of the reference in every mode | | |

Tuning notes, for the knobs' defaults:

- the decision is made where the loop is met, in the one pass, but the body is measured after
  its nested loops are unrolled (`loops.sizes`), so an outer loop is decided knowing how big it
  becomes, as a bottom-up pass would decide. This is what made a01 faster than the reference: its
  key loop (133 statements around four 64-step dot products) stays rolled around straight inner
  loops, 65 us; unrolled by 2 it is 76 us, in full as nvcc does it 92 to 107 us with 2.7 KB spilled;
- the quota (2048 statements per nest) is the register lever: ptxc's registers are virtual,
  ptxas allocates under `.maxntid`, and what makes it spill is how many copies it can interleave;
- unhinted loops: straight when the expansion fits 1024 statements, else x8, unless the body as
  written is over 64 statements. The attention inner dot products (64 x 4) run 1.6x faster
  straight than x4 or x8 inside the outer body; r02 / r03 gain 15% straight over x8; x4 is 10%
  slower than x8 on r02 and x32 is 2x slower. The length limit is on the body as written, not
  expanded: a02's 16-statement key loop around a straight 128-statement dot product is worth x8
  (25 us against 37 rolled);
- the `cse` memo (opaque index subexpressions and address bases computed once per scope) changes
  no timing, ptxas merges the duplicates itself; it halves the PTX text. `Options(cse=False)` shows
  the raw lowering;
- binding an unrolled copy's axis as `Bound(var, j * step)` instead of an `add` is worth 14% on
  a02 and on a 1M-element reduction: ptxas does not fold the add through the widening multiply;
- correctness does not depend on any of it: the suite runs the loops of every emission path under
  `none`, `auto`, `force` and `cse=False` (396 passed, 2 xfailed).

## Precision (2026-10-29)

The check every problem is scored with is `np.allclose`: an output element passes when
`|got - expected| <= atol + rtol * |expected|`. `rtol` is the relative part, a share of the
value itself (1e-3 is about 8400 f32 ulps, 1e-6 about 8); `atol` is the absolute floor that takes
over where the expected value is near zero and any relative bound would collapse (softmax tails,
differences that cancel). The expected values are computed in f64 and rounded to f32, so a
tolerance of zero demands the correctly rounded f32 answer; the open set uses it only for single
pointwise ops (add, mul, relu).

Emmy's own reference kernels are built with `--use_fast_math` (`emmy/compiler/backend/cuda/nvcc.py`):
fma contraction, approximate division, square root and reciprocal, flush-to-zero, `__expf`. The
tolerances were set with that reference passing, so they have room for exactly those spellings.
ptxc pinned `.rn` on every float add / sub / mul, which the PTX ISA defines as the thing that stops
ptxas from fusing: "mul/add sequences with no rounding modifiers may be optimized to use
fused-multiply-add instructions on the target device". Every dot product ran as mul + add.

What changed: `ptxc/ops.py` gives the float recipes an error budget, `Budget`, decided once per
kernel from meta.json's tolerance: a quarter of `rtol` and of `atol` (`auto`), nothing (`exact`) or
no limit (`fast`; `Options.precision`, `compile.py --precision`, `trace=("precision",)` prints it).
A recipe with a faster spelling asks `allows(ulps, absolute)` with the worst case the PTX ISA states
for it, in the tolerance's own shape, and takes it on yes; the IEEE spelling is the fallback. The
other three quarters of the tolerance stay for the computation's own rounding and the summation
order, which it must absorb in any mode.

| spelling | worst case, PTX ISA | asks for | fits from |
|---|---|---|---|
| `add.f32`, `sub.f32`, `mul.f32` without `.rn` | ptxas fuses mul + add; against the two-rounding result the fma moves by the product's rounding error, under 1 ulp of the product (and closer to the exact value) | 1 ulp | rtol 4.8e-7 |
| `div.full.f32` | 2 ulp over the whole range (`div.approx` only for divisors in [2^-126, 2^126]) | 2 ulp | rtol 9.5e-7 |
| `rcp.approx.f32`, `sqrt.approx.f32` | 1 ulp, 2^-23 relative | 1 ulp | rtol 4.8e-7 |
| `tanh.approx.f32` | 2^-11 relative over the whole range (125 ulp measured on randn) | 2^12 ulp | rtol 2e-3 |
| `sin.approx.f32` / `cos.approx.f32` after `x - rint(x / 2pi) * 2pi` (3 instructions, 2 pi in two parts) | absolute 2^-20.5 / 2^-20.4 within [-2pi, 2pi] | 2^-20.4 absolute | atol 2.9e-6 |
| the same on the raw argument, only with `Options.raw_trig` | absolute 2^-14.7 within 100 pi, best effort beyond | 2^-14.7 absolute | atol 1.5e-4 |
| `.ftz` on `ex2` / `lg2` / `rsqrt.approx` and on the approximate rcp / sqrt / div | a subnormal result flushes to zero | 2^-126 absolute | atol 4.7e-38 |

A spelling asks for a relative error where the ISA's bound is relative and an absolute one where it
is absolute, which is what makes tanh and sin fall out differently: `tanh.approx` needs rtol >= 2e-3
and every tolerance in both sets is 1e-3 or tighter, so the polynomial wins everywhere today and a
looser hidden problem would get the single instruction. `sin.approx` is bounded only on a range, so
the argument is reduced into [-pi, pi] first, where the bound holds for any input; the reduction is
three instructions against the polynomial path's fifteen, and the 1e-3 / 1e-4 tiers take it. Skipping
the reduction is the one assumption about inputs the IR cannot state, so it sits behind `raw_trig`
(default off): with it, and atol room for 2^-14.7, `sin.approx` goes straight on the argument, as
emmy's `__sinf` does. The recipes' own numerics stay pinned in every mode: `fma.rn` and `mul.rn` inside
the polynomials, pow's double-float `y * log2|x|`, the `mul.rn` by log2e inside exp. f64 arithmetic
is always `.rn`. Every tolerance in both problem sets is either zero or at least 1e-6, so `auto`
spends nothing on the former and buys everything but `tanh.approx` and `sin.approx` on the latter
(our sin problems sit at atol 1e-6); the budget only bites in between.

Measured over all 160 problems (`tests/gpu_runner.py` now reports, per output, the largest distance
in f32 ulps and the share of the tolerance the worst element used):

| mode | pass | tolerance used, median / max | notes |
|---|---|---|---|
| exact | 158 / 160 | 0.19% / 12.25% | the two Select problems, as before |
| auto (default) | 158 / 160 | 0.19% / 12.25% | identical pass set; 36 problems change bits |
| fast forced | 153 / 160 | 0.19% / 12.25% | loses p12_div_abs (1 ulp), p19_relu_affine `x * 3 - 1` (64 ulp), p20_three_inputs `x * y + z` (704 ulp), p26_reciprocal (1 ulp), all at rtol = atol = 0, and p42_gelu_tanh at 1e-5 (`tanh.approx` inside `1 + tanh(...)`, 270% of the tolerance); p16_tanh and p84_tanh_wide survive `tanh.approx` at 83% and 88% of theirs, which is what a cost of 2^12 ulp against a budget of 2 ulp predicts |

- The worst users of their tolerance are the same in every mode and come from the pinned
  transcendentals: p89_exp_wide 12.2% (`ex2.approx` at |x| up to 32), p14_log 11.7%, p47_softplus
  8.0%, p43_sin_cos 7.6%, p80_gelu_2d 6.4%. Nothing the relaxations touch comes near them.
- Where `auto` moves the error it moves it by a fraction of a percent of the tolerance, in both
  directions: m25_rect_small_n 1.18% -> 1.50%, m02_matmul_rect 0.40% -> 0.55%, m11 0.34% -> 0.45%,
  m14_1024 0.71% -> 0.75%; m16_bias 1.03% -> 0.91%, m18_relu 0.54% -> 0.46%, a17_scaled_v 0.36% ->
  0.31%, r31_sum_two_inputs 0.22% -> 0.07%. The approximate instructions cost one ulp where they
  appear alone: p15_sqrt_rsqrt 1 -> 2 ulp, p51_pow_half 0 -> 1, p17_sigmoid 1.16% -> 1.23%.
- Ulp counts on matmul and attention outputs are in the thousands to millions in every mode
  (m01 3.0M exact, 2.9M auto) and are not a measure of anything: a dot product of 512 random terms
  lands near zero often, and one ulp of a result near zero is tiny. The share of the tolerance is
  the measure, which is why `atol` exists.
- p20 under `fast` shows what contraction does at tolerance zero: 236 of 1024 elements differ, by
  at most 9.5e-7 absolute (half an ulp of a product of magnitude up to 20) and 704 ulps of a result
  where x * y and z cancelled. `auto` keeps `.rn` there.

What it buys (`python tests/bench.py ... -c auto= -c exact=precision=exact --repeats 5`, RTX 5080,
min over 5 runs of the median of 30 launches; the kernels under 10 us are launch-bound and unchanged):

| problem | auto (default) | exact | emmy reference |
|---|---|---|---|
| a12_256 | 380.7 us (179 regs) | 600.8 us (128 regs) | 985.4 us (255 regs, 816 B spill) |
| m14_1024 | 156.3 us (104 regs) | 217.8 us (80 regs) | 146.2 us (96 regs) |
| a19_long_keys | 384.8 us (255 regs, 208 B spill) | 445.7 us (255 regs, 348 B spill) | 553.5 us |
| a01_attention | 55.0 us (255 regs, 212 B spill) | 65.3 us (255 regs, 352 B spill) | 78.6 us (255 regs, 236 B spill) |
| m01_matmul_512 | 30.5 us (42 regs) | 32.5 us (40 regs) | 29.9 us (40 regs) |
| a02_attention_128 | 25.2 us | 25.2 us | 35.6 us |
| r02_big_sum, r03_mean | 11.0, 9.1 us | 11.4, 9.3 us | 13.5, 11.7 us |
| m02, m21, f01, f02, f03, f22 | within 0.5 us of exact and of the reference | | |

The gain is the fma: half the arithmetic instructions in a dot product, and fewer live registers
between the mul and the add, which is where a01 and a19 shed a third of their spills. The approximate
rcp / sqrt / div and `.ftz` are worth about 2 us on a01 and nothing measurable elsewhere; they are
kept because the reference uses them and the tolerance pays for them. Where `exact` matches `auto`
(a02, m21) ptxas had nothing to fuse that the schedule left adjacent, or the kernel is bound elsewhere.

## f16 (2026-10-29)

Emmy traces half tensors into the same IR: `f16` buffers, `f16` as the result dtype of every
pointwise Assign (its CUDA uses `__half` operators and `hexp`, so each op rounds to half), `f32`
accumulators rounded to f16 once at the store, and a softmax whose running max stays `f16` through
WarpShuffle, a `float` Smem and TreeHalve.

ptxc treats f16 as a memory type. A value is in a `.b16` register only between a load or store and
a `cvt`; every op computes in f32 and `cvt.rn.f16.f32` rounds the result where the IR's dtype says
f16. That is the correctly rounded half result for add, sub, mul, div and sqrt (f32 carries 24 bits,
at least 2 x 11 + 2, so rounding twice equals rounding once) and within an f16 ulp of the `hexp`-style
intrinsics elsewhere. Half arithmetic is never spelled: PTX takes no half constant operand (ptxas:
"Arguments mismatch"), has no `mov.f16` or `ld.f16` (halves move as `.b16`), and `ex2.approx.f16`
would be one more spelling for kernels that are launch-bound. Two or four halves load and store as
`.v2.b16` / `.v4.b16`; a half in a shuffle rides in the low half of a `.b32`; the max identity
`-1e30` is `mov.b16 %rs, 0xFC00`, -inf, which is also what `__float2half(-1e30f)` gives emmy.

Seven problems in tests/problems (p95-p98, r50, r51, f50), expected values from numpy's float16,
which rounds per op like the IR. On the RTX 5080:

| problem | ptxc | emmy's reference | note |
|---|---|---|---|
| p95_f16_mul_add | exact (rtol = atol = 0) | 247 / 1024 off by one ulp | its `mul.f16` + `add.f16` carry no `.rn`, so ptxas fuses them into one rounding |
| p96_f16_where, p97_f16_to_f32, r51_f16_amax | exact | exact | |
| p98_f16_relu_exp, r50_f16_sum | exact | exact | tolerances 2e-3 / 1e-3 and 2e-3 / 2e-2 leave an ulp of slack, unused |
| f50_f16_softmax | 5 f16 ulp, 16% of the tolerance | 5 ulp, 12% | |

The same bits under `--precision exact` and `fast`: the budget only changes f32 spellings, and a
conversion sits between every two of them, so ptxas has nothing to fuse. Time (bench.py, min of 5):
7.8 to 9.3 us for ptxc, 7.6 to 9.5 us for the reference, launch-bound either way; the two
conversions per op cost nothing measurable. The runner now counts ulps in the output's own width
(an f16 ulp read as 8192 f32 ulps before) and hands nvrtc the pip CUDA headers, which the reference
needs for `cuda_fp16.h`. Not covered: bf16 and i16 (the runner accepts neither), and emmy's f16
matmul, which is a tensor-core kernel (ldmatrix, mma.sync, swizzled cp.async) at every size.

## Coverage

152 own problems: pointwise sizes 7 to 1M, ranks 1 to 4, odd and prime widths, broadcasts, every
op in emmy's CUDA table (add, subtract, multiply, divide, comparisons, maximum, minimum, pow,
negative, copy, square, reciprocal, relu, sigmoid, exp, log, rsqrt, sin, cos, tanh, sqrt, erf, abs,
where, bitwise_not / and / or) over wide ranges; reductions of 5 to 1M elements, rows of 8 to
10007, columns, middle axes, 3-D, unpinned schedules with atomics; matmul 32 to 1024 square and
rectangular, split-K, epilogues, bmm; softmax over 16 to 4096 and dim 0, layer / rms / l1 / l2
norms, cosine similarity; attention 16 to 256 queries, 16 to 1000 keys, batched; f16 pointwise, reductions and softmax.

Expressions emmy itself cannot trace (floor, exp2, expm1, log_softmax, var, prod, whole-tensor
2-D max / min / sum, amin, reduction followed by a scalar epilogue, matmul followed by sum,
torch.remainder) stay listed under `UNTRACEABLE`.

## Reproduce

```
python -m pytest tests -q                 # 455 passed, 3 skipped, 2 xfailed
python tests/bench.py m01_matmul_512 -c auto= -c exact=precision=exact   # what the tolerance buys
python tests/gpu_runner.py --all          # the same, as a report with registers and timings
python tests/gpu_runner.py p85_pow5       # one problem
python tests/bench.py --all               # the performance table above
```
