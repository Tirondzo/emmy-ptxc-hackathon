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
| load | [`ir.py`](./ptxc/ir.py) | emmy's object model (the only module that imports emmy); the launch config from meta.json | interpret statements |
| lower | [`lower/`](./ptxc/lower/) | core business logic. One rule per statement kind, names and scopes, expression trees | spell PTX text, pick registers or labels |
| ops | [`ops.py`](./ptxc/ops.py) | legalization of elementwise ops and every numeric choice | know which statement used the op |
| build | [`ptx/builder.py`](./ptxc/ptx/builder.py) | registers, labels, structured control flow | know the IR |
| render | [`ptx/printer.py`](./ptxc/ptx/printer.py) | PTX layout | make decisions |
| config | [`config.py`](./ptxc/config.py) | every choice made rather than read from the IR: the `Target` (sm, ISA version, warp, launch limits) and the numeric conventions (`HUGE`, `IDENTITY`); for the developer, never an option | hold opcode text, or a recipe's own numbers |

`ptx/` knows PTX and nothing about Emmy. Everything in between speaks the IR's field
vocabulary (`s.body`, `s.index`) and the PTX layer's operands.

We avoid pre-pass analysis to simplify the solution and overall approach to architecture. It's very easy to fall down to multi-pass solution that both blurs responsibility and brings over-complication.
The IR and meta.json fix every choice correctness depends on. What they leave open is pinned
once: the machine and the identities in `config.py`, the rounding and the recipes in `ops.py`.

## Contracts

- **The context is ambient.** `with PtxBuilder(module), Lowering(kernel, target):` in the
  driver makes the module the target of every emission, the kernel the source of every name
  and the `Target` the machine, through a `ContextVar` (scoped, reset on exit, per thread). The verbs are plain functions
  over it: `emit`, `op`, `cvt`, `if_`, `for_` from `ptx`; `read`, `assign`, `array`, `scope`,
  `lower` from `lower`. Emission is eager: nothing reads the context after the `with` closes.
- **Emit returns an operand.** `expr.value`, `ops.apply` and the builder's `op` / `cvt` /
  `setp` return a typed `Reg` or `Imm`. Callers never care how a value was produced, and
  constants fold on the way up: `expr.linear` turns `(a0 * 4) + 3` into a `[base+12]`
  displacement. A type applied to a number is an immediate: `INDEX(4)`, `ty(0)`.
- **One rule per node kind.** `@rule("Kind")` in `lower/{compute,control,block}.py`. An
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
| Accumulator seed | a `seed=True` loop declares each Accum of its *own* body (not of nested blocks) at its identity, in the enclosing scope, before opening its own; an outer name of the same spelling is shadowed, not reset | `control.seed_accumulators` |
| Accum | `name <- op(base or name, value)`, `op` through emmy's alias map (`amax` is `maximum`): online softmax rescales the accumulator and passes it as `base` | `compute.accum` |
| Identities | `config.IDENTITY` by canonical op; a maximum seeds at `-HUGE`, emmy's `-1e30` rather than `-inf`, so online softmax never sees `inf - inf = NaN`; `HUGE` is the one number to change (`math.inf` for IEEE) | `ir.op_identity` |
| Implicit names | `lane = tid & (warp - 1)`, `warp = tid >> log2 warp`, from `Target.warp`, bound in the prologue | `lowerer.prologue` |
| Shared memory | `smem_bytes > 0` -> one `.extern` pool carved by a running cursor; else static arrays | `block.smem` |
| Combines | the states are made Variables first (`lowerer.mutable`); `combine_states` is instantiated per step in a fresh scope with `state_b` bound, and its Assigns mutate them | `block.combine` |
| Warp sync masks | `activemask` for `shfl.sync` and `bar.warp.sync`: a sub-warp block (a22 launches 16 threads) or a guarded tile leaves lanes that never arrive, and naming them is undefined | `block.shuffle`, `builder.barrier` |
| WarpShuffle | a power-of-two length is the butterfly (emmy's only case); any other length folds down onto lane 0 under a `lane + delta < length` guard and broadcasts it, since a butterfly cannot reach every lane; a length beyond `Target.warp` is `InvalidKernel` | `block.warp_shuffle` |
| TreeHalve | halving tree, barrier outside the branch, root broadcast to every thread | `block.tree_halve` |
| Rounding | `.rn` on float add / sub / mul (also blocks fma contraction); explicit `fma.rn` inside recipes | `ops.py` |
| Transcendentals | exp / log / rsqrt through `ex2` / `lg2` / `rsqrt.approx`, 2 ulp at the instruction; exp's error grows with the argument, about \|x\| ulp, because x log2e is rounded before `ex2` (`__expf`'s behaviour), and log's relative error grows as x nears 1, because `lg2.approx` is 2^-22.6 absolute; erf, sin, cos, tanh as polynomials at about 2 ulp, because `tanh.approx` is 2^-11 and `sin.approx` has no range reduction; pow as `ex2(y * lg2(abs x))` with C's sign and NaN rules, its error proportional to \|y log x\| | `ops.py` |
| bool | a `.pred` register: comparisons produce it (`setp`), `where` selects on it (`selp`), arithmetic converts it to 1 / 0 and anything nonzero (NaN included) back to true, as C does; in a buffer it is one byte of 0 / 1, as C stores it; `selp` has no predicate form, so a select between predicates goes through 1 / 0 | `builder.cvt`, `builder.select`, `types.storable`, `ops.py` |
| Coordinates | `//` in a flattened tile index (every pointwise tile of rank >= 2) lowers to `div.s32`: operands are non-negative | `expr.arith` |
| Beyond the contract | `CastExpr`, `FuncCallExpr` and `Builtin` are starred in TASK.md (in no open or hidden problem) but lowered anyway, a line each: a cast is `cvt`, a call is the elementwise recipe at the first float argument's type (else the first argument's), a builtin (`thread_idx.x`, as emmy spells it) is its special register and `warp_size` is `Target.warp` | `expr.value` |
| Indexing | one coordinate per axis, row-major; an empty index is element 0 (`sum_1[]`); any other count is `InvalidKernel` rather than a guessed flattening | `expr.address` |
| Type mixes | a binary op takes C's usual arithmetic conversions: float over integer, the wider width, and at one width unsigned over signed whichever side it is on (`u32 < s32` compares unsigned). `cvt` between raw bits and numbers, `setp` on predicates or ordered on bits: `Unsupported` before ptxas sees them | `builder.unify`, `builder.cvt`, `builder.setp` |
| Parameters | every parameter is a `.u64` buffer pointer; a `"scalar": true` argument in meta.json (only `serial` axes pass one) is `Unsupported` by name | `ir.Kernel.load` |

## Extending

| To add | Do |
|---|---|
| an elementwise op | one `@recipe("name")` in `ops.py`, composing others by calling them |
| a node kind | one `@rule("Kind")` in the matching `lower/` module |
| a dtype | its names (emmy's, and the C spelling `Smem` carries) in `types.NAMES`, a bank in `Ty.bank` and `PREFIX`, conversions in `builder.cvt` |
| a kernel-wide value | bind it in `lowerer.prologue` |
| a target | one more `Target` in `config.py`, passed to `compile_file`; then `Unsupported` where an instruction is younger than the machine: `cp.async` (sm_80), `shfl.sync` / `activemask` / `bar.warp.sync` (sm_70), vector `red` (sm_90) |
| a numeric convention | `config.py` (`HUGE`, `IDENTITY`); rounding and a recipe's own numbers stay in `ops.py` |

## Deferred, on purpose

Each needs a concrete case before it earns its place:

- an `optimize(module)` pass between lower and render (mov-chain folding, hoisting): the
  module is already data, so it slots in without touching the rules;
- several `KernelOp`s per file (their shared arrays would need entry-qualified names);
- 16-bit dtypes, swizzled shared layouts, `aux_threads`, `serial` axes, and the starred
  nodes of the IR reference (TMA, mbarrier, tensor-core fragments). All raise `Unsupported`.

## Useful links

https://docs.nvidia.com/cuda/parallel-thread-execution/#instruction-set
