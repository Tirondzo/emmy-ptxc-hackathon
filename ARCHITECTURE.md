# ptxc architecture

`ptxc` compiles one Emmy Kernel IR kernel to PTX:

```bash
python compile.py path/to/kernel.ir.json > out.ptx     # meta.json sits next to the IR
```

Design fundamental : The compiler grows by adding a rule or a recipe, not by editing. Core parts are encapsulated.

## Pipeline

```
kernel.ir.json + meta.json -> load -> lower -> render -> PTX
```

| Stage | Module | Owns | Must not |
|---|---|---|---|
| load | `ir.py` | emmy's object model (the only module that imports emmy); the launch config and the tolerance from meta.json | interpret statements |
| lower | `lower/` | core business logic. One rule per statement kind, names and scopes, expression trees | spell PTX text, pick registers or labels |
| ops | `ops.py` | legalization of elementwise ops and every numeric choice, including the one the tolerance leaves open (`Numerics`) | know which statement used the op |
| build | `ptx/builder.py` | registers, labels, structured control flow | know the IR |
| render | `ptx/printer.py` | PTX layout | make decisions |
| config | `config.py` | every choice made rather than read from the IR: the `Target` (sm, ISA version, warp, launch limits) and the numeric conventions (`HUGE`, `IDENTITY`); for the developer, never an option | hold opcode text, or a recipe's own numbers |
| context | `context.py` | the ambient `Lowering` for rules and recipes alike: `lowering()`, `options()`, `target()`, `trace()` | hold state of its own |

`ptx/` knows PTX and nothing about Emmy. Everything in between speaks the IR's field
vocabulary (`s.body`, `s.index`) and the PTX layer's operands.

We avoid pre-pass analysis to simplify the solution and overall approach to architecture. It's very easy to fall down to multi-pass solution that both blurs responsibility and brings over-complication.
The IR and meta.json fix every choice correctness depends on. What they leave open is pinned
once: the machine and the identities in `config.py`, the recipes in `ops.py`. The rounding is the
one choice meta.json decides indirectly: its `rtol` / `atol` say how much a result may move, and
`ops.Numerics` spends part of that on the faster spellings.

## Contracts

- **The context is ambient.** `with PtxBuilder(module), Lowering(kernel, target, options):` in the
  driver makes the module the target of every emission, the kernel the source of every name, the
  `Target` the machine and the options (with `Lowering.budget`, what every float recipe may spend) the
  policy, through a `ContextVar` (`context.py`; scoped, reset on exit, per thread). The verbs are plain functions
  over it: `emit`, `op`, `cvt`, `if_`, `for_` from `ptx`; `read`, `assign`, `array`, `scope`,
  `lower` from `lower`. Emission is eager: nothing reads the context after the `with` closes.
- **Emit returns an operand.** `expr.value`, `ops.apply` and the builder's `op` / `cvt` /
  `setp` return a typed `Reg` or `Imm`. Callers never care how a value was produced, and
  constants fold on the way up: `expr.affine` keeps an index as sum(coefficient * register) +
  constant, so `(a0 * 4) + 3` is a `[base+12]` displacement and `a1_b * 64 - a1_b * 64` costs
  nothing; `expr.address` computes one base register per distinct (array, terms) in a scope.
  A type applied to a number is an immediate: `INDEX(4)`, `ty(0)`.
- **One rule per node kind.** `@rule("Kind")` in `lower/{compute,control,loops,block,fragments}.py`. An
  unknown kind, op, field value or dtype raises `Unsupported` compilation error.
- **Names are lexically scoped.** An SSA name (`Load`, `Assign`, `Let`) is `Bound` in the
  innermost scope and shadows an outer binding until the block closes, as in C. Loop-carried
  state (`Accum`, `Init`, a collective's states) is a `Variable`: one register mutated by `mov`.
  Three invariants in `lower/lowerer.py` keep the two apart without analysing data flow: a
  Bound never aliases a Variable's register (the value is copied first); a `seed=True` loop
  *declares* its accumulators in the enclosing scope; a collective's state is a Variable or a
  Bound of the current scope, anything else is `Unsupported`.
- **Registers are per bank.** `Ty.bank` maps s32 / u32 / b32 to one `.b32` bank; register
  names and counts are per bank, never per type.
- **Every input fault is one line.** Whatever the IR or meta.json gets wrong is a `CompileError`
  (`InvalidKernel`: breaks the IR's rules or disagrees with meta.json; `Unsupported`: valid IR
  ptxc does not lower yet), printed as one line on stderr with exit 1 and nothing on stdout.
  `ir.Kernel.load` turns whatever the two files get wrong, including anything emmy's parser
  raises, into `InvalidKernel`; a Python traceback is a compiler bug, never an input.

## Decided by the IR's semantics

| Topic | Rule | Where |
|---|---|---|
| Tile decode | row-major over every listed axis, in unsigned arithmetic (gid is never negative); guard `gid < cells` when the launch has spare threads | `control.tile` |
| Launch bound | `.maxntid` from meta.json's block: ptxas bounds registers to what the launch needs | `driver`, `printer` |
| Launch limits | a block beyond `Target.max_block` or `smem_bytes` beyond `Target.max_smem` is `InvalidKernel` at compile time, not a launch failure | `config.Target.check` |
| Rasterization | ignored for now. Doesn't affect correctness, only cache hits | `control.tile` |
| Loop unrolling | the one performance decision, between correct programs only (same order, same barriers). Three shapes: straight (no loop, the axis a constant), rolled, partial (`factor` copies per iteration, then the known remainder as copies or a tail loop). A loop the IR marks `unroll=True` goes straight within the quota, else by the largest divisor of its trip count; an unmarked loop goes straight when small, else by the compiler's factor. The quota is measured on the nest: `expansion` predicts what nested loops become, so an outer loop is decided knowing its body's size, as a bottom-up pass would, in the one pass. The trip count of `for i in tid..N:T` is `N / T` for every thread when the Tile axis `tid` has extent <= T and T divides N. Copies bind the axis to a constant or to `Bound(var, j * step)`, which `expr.affine` folds into displacements, so an unrolled body is loads and FMAs off one base register (binding through an `add` instead costs 14% on the attention kernel) | `loops.py`, `options.py` |
| Accumulator seed | a `seed=True` loop declares each Accum of its *own* body (not of nested blocks) at its identity, in the enclosing scope, before opening its own; an outer name of the same spelling is shadowed, not reset | `control.seed_accumulators` |
| Accum | `name <- op(base or name, value)`, `op` through emmy's alias map (`amax` is `maximum`): online softmax rescales the accumulator and passes it as `base` | `compute.accum` |
| Identities | `config.IDENTITY` by canonical op; a maximum seeds at `-HUGE`, emmy's `-1e30` rather than `-inf`, so online softmax never sees `inf - inf = NaN`; `HUGE` is the one number to change (`math.inf` for IEEE) | `ir.op_identity` |
| Implicit names | `lane = tid & (warp - 1)`, `warp = tid >> log2 warp`, from `Target.warp`, bound in the prologue | `lowerer.prologue` |
| Shared memory | `smem_bytes > 0` -> one `.extern` pool carved by a running cursor; else static arrays | `block.smem` |
| Combines | the states are made Variables first (`lowerer.mutable`); `combine_states` is instantiated per step in a fresh scope with `state_b` bound, and its Assigns mutate them | `block.combine` |
| Warp sync masks | `activemask` for `shfl.sync` and `bar.warp.sync`: a sub-warp block (a22 launches 16 threads) or a guarded tile leaves lanes that never arrive, and naming them is undefined | `block.shuffle`, `builder.barrier` |
| WarpShuffle | a power-of-two length is the butterfly (emmy's only case); any other length folds down onto lane 0 under a `lane + delta < length` guard and broadcasts it, since a butterfly cannot reach every lane; a length beyond `Target.warp` is `InvalidKernel` | `block.warp_shuffle` |
| TreeHalve | halving tree, barrier outside the branch, root broadcast to every thread | `block.tree_halve` |
| Rounding | an error budget per kernel, `Lowering.budget` (`ops.Budget`): a quarter of meta.json's `rtol` / `atol` (`auto`), nothing (`exact`) or no limit (`fast`). A recipe with a faster spelling asks `allows(ulps, absolute)` with the worst case the PTX ISA states for it and takes it on yes, else the IEEE spelling: add / sub / mul without `.rn` so ptxas fuses mul + add into fma (1 ulp), `div.full` (2), `rcp.approx` / `sqrt.approx` (1), `tanh.approx` (2^12, so from rtol 2e-3), `sin.approx` on an argument reduced into [-pi, pi] (2^-20.4 absolute) or, with `Options.raw_trig` vouching that inputs stay within 100 pi, on the raw one (2^-14.7), `.ftz` on the approximate instructions (2^-126 absolute). A tolerance of zero keeps IEEE `.rn` everywhere: the open set's exact problems are single pointwise ops, where there is nothing to contract. Recipes keep their own `fma.rn` / `mul.rn`; f64 is always `.rn` | `ops.py`, `driver` |
| Transcendentals | exp / log / rsqrt through `ex2` / `lg2` / `rsqrt.approx` (a few ulp); erf, sin, cos, tanh as polynomials at about 2 ulp, because `tanh.approx` is 2^-11 and `sin.approx` has no range reduction; pow as `ex2(y * lg2(abs x))` with C's sign and NaN rules | `ops.py` |
| bool | a `.pred` register: comparisons produce it (`setp`), `where` selects on it (`selp`), arithmetic converts it to 1 / 0 and anything nonzero (NaN included) back to true, as C does; in a buffer it is one byte of 0 / 1, as C stores it; `selp` has no predicate form, so a select between predicates goes through 1 / 0 | `builder.cvt`, `builder.select`, `types.storable`, `ops.py` |
| f16 | a memory type: `.b16` registers between `ld` / `st` and a `cvt`; every op computes in f32 and rounds once where the IR's dtype says f16, which is the IEEE half result for + - * / (f32 carries 2p+2 bits, so the double rounding is innocuous) and within an ulp of CUDA's `hexp`-style intrinsics elsewhere. PTX half arithmetic takes no constants, so none is spelled; a half in a shuffle rides in a `.b32`; emmy's own CUDA emits `mul.f16` + `add.f16`, which ptxas fuses, and so misses numpy's per-op rounding by an ulp where ptxc hits it | `types.Ty.bank` / `Ty.raw`, `ops.apply`, `builder.unify` / `cvt`, `block.shuffle` |
| Tensor cores | emmy's f16 matmul. `RegFragment` is one register per ISA slot per lane (A four `.b32` of packed halves, B two, C four `.f32` zeroed); a family (`count`) is one binding per member under its spelled name, so `_b[_r0]` resolves once its loop is unrolled and is `Unsupported` while it is a register. `LdmatrixLoad` is `ldmatrix.x4` (A), `x2.trans` (B) or `x4.trans` (two B fragments, `pair_frag`): lane l addresses row l % 16 of the tile and column block l // 16. `MmaSyncPtx` is `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` with d = c. `RegStore` writes rows g and g + 8 at columns 2t, 2t + 1 (g = l // 4, t = l % 4), one `.v2` pair per row, converted to the buffer's type. The lane maps are index expressions over the prologue's `lane`, so `expr.address` folds them like any index. Whole warps only: a block not a multiple of 32 or a guarded tile is `InvalidKernel`, below sm_80 `Unsupported`; every field emmy's nodes carry beyond the f16 matmul (gmem-direct loads, byte slabs, guards, epilogues, other atoms and dtypes) is `Unsupported` first | `fragments.py` |
| Swizzled slabs | emmy's software swizzle on a 16-bit slab: `e ^ (((e >> 6) & bits) << 3)` on the element index, bits 7 / 3 / 1 for B128 / B64 / B32, so the rows one `ldmatrix` reads sit in different banks. The cp.async fill and the ldmatrix drain go through one function and agree by construction; the swizzled index is one value, so it forgoes the displacement | `block.slab` |
| Floor division | `//` and `%` are Python's, floored, as torch and emmy's own folder define them; emmy's CUDA renders C's `/` and `%`, the same only for non-negative operands (every index it builds). By a literal 2^k they are `shr` and `and`, cheaper than C's and right for either sign. `/` is C's | `ops.shift`, `ops.down`, `expr.fold` |
| Beyond the contract | `CastExpr`, `FuncCallExpr` and `Builtin` are starred in TASK.md (in no open or hidden problem) but lowered anyway, a line each: a cast is `cvt`, a call is the elementwise recipe at the first float argument's type (else the first argument's), a builtin (`thread_idx.x`, as emmy spells it) is its special register and `warp_size` is `Target.warp` | `expr.value` |
| Indexing | one coordinate per axis, row-major; an empty index is element 0 (`sum_1[]`); any other count is `InvalidKernel` rather than a guessed flattening | `expr.address` |
| Type mixes | `cvt` between raw bits and numbers, `setp` on predicates or ordered on bits: `Unsupported` before ptxas sees them | `builder.cvt`, `builder.setp` |

## Options: the trade-offs

`config.py` holds what ptxc assumes about the machine and its numeric conventions, never a
command-line option. `options.py` holds the performance trade-offs the compiler decides on
its own: the unroll mode and the code-size quota on a loop nest (the register lever: ptxas
spills when it can interleave too many copies), the factor and thresholds for loops the IR does
not mark, the `precision` mode (`auto` spends a quarter of the tolerance on fma contraction and
the approximate instructions; `exact` and `fast` are the two ends), `raw_trig` (the one assumption
about inputs the IR cannot state: sin / cos arguments within 100 pi), the `cse` memo (ptxas merges
the duplicates itself; the memo keeps the PTX short) and `trace` topics. The command-line options
are `--unroll auto|none|force|QUOTA`, `--precision auto|exact|fast` and `--raw-trig`; everything else is set by
constructing `Options(...)` (tests, `tests/bench.py`) or by changing a default. The measurements
behind the defaults are in `tests/FINDINGS.md`. `Options` travels on `Lowering`, like everything
ambient; there are no environment variables.

## Extending

| To add | Do |
|---|---|
| an elementwise op | one `@recipe("name")` in `ops.py`, composing others by calling them |
| a node kind | one `@rule("Kind")` in the matching `lower/` module |
| a dtype | its names (emmy's, and the C spelling `Smem` carries) in `types.NAMES`, a bank in `Ty.bank` and `PREFIX`, conversions in `builder.cvt`; a type `ld` / `st` / `mov` cannot spell gets its spelling in `Ty.raw` |
| a kernel-wide value | bind it in `lowerer.prologue` |
| a target | one more `Target` in `config.py`, passed to `compile_file`; then `Unsupported` where an instruction is younger than the machine: `cp.async` (sm_80), `shfl.sync` / `activemask` / `bar.warp.sync` (sm_70), vector `red` (sm_90); `mma.sync.m16n8k16` already is (`fragments.whole_warps`) |
| a tensor-core node | one `@rule` in `lower/fragments.py`: a fragment is a `Fragment` symbol (its registers in ISA slot order), a lane map an index expression, a slab address `block.slab`; whatever field the rule does not read is `Unsupported` first |
| a numeric convention | `config.py` (`HUGE`, `IDENTITY`); a recipe's own numbers stay in `ops.py` |
| a faster spelling | one `if allows(ulps=..., absolute=...)` in the recipe, with the bound the ISA states (or one measured), and a test that the GPU keeps it |

## Deferred, on purpose

Each needs a concrete case before it earns its place:

- an `optimize(module)` pass between lower and render (mov-chain folding, hoisting): the
  module is already data, so it slots in without touching the rules;
- several `KernelOp`s per file (their shared arrays would need entry-qualified names);
- bf16 and the 16-bit integers (the runner accepts neither), `aux_threads`, `serial` axes, and the
  starred nodes of the IR reference beyond the f16 matmul's: TMA, mbarrier, wgmma, and the fragment
  epilogues (`FragmentApply`, `FragmentRowReduce`, `FragmentRepack`, `FragmentPromote`), which emmy
  emits only for its flash-attention mma schedule. All raise `Unsupported`.

## Useful links

https://docs.nvidia.com/cuda/parallel-thread-execution/#instruction-set
