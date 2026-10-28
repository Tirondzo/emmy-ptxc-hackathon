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
| load | `ir.py` | emmy's object model (the only module that imports emmy); the launch config from meta.json | interpret statements |
| lower | `lower/` | core business logic. One rule per statement kind, names and scopes, expression trees | spell PTX text, pick registers or labels |
| ops | `ops.py` | legalization of elementwise ops and every numeric choice | know which statement used the op |
| build | `ptx/builder.py` | registers, labels, structured control flow | know the IR |
| render | `ptx/printer.py` | PTX layout | make decisions |

`ptx/` knows PTX and nothing about Emmy. Everything in between speaks the IR's field
vocabulary (`s.body`, `s.index`) and the PTX layer's operands.

We avoid pre-pass analysis to simplify the solution and overall approach to architecture. It's very easy to fall down to multi-pass solution that both blurs responsibility and brings over-complication.
The IR and meta.json fix every choice correctness depends on; the numeric choices they
leave open are pinned once in `ops.py`.

## Contracts

- **The context is ambient.** `with PtxBuilder(module), Lowering(kernel):` in the driver
  makes the module the target of every emission and the kernel the source of every name,
  through a `ContextVar` (scoped, reset on exit, per thread). The verbs are plain functions
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

## Decided by the IR's semantics

| Topic | Rule | Where |
|---|---|---|
| Tile decode | row-major over every listed axis; guard `gid < cells` when the launch has spare threads | `control.tile` |
| Rasterization | ignored for now. Doesn't affect correctness, only cache hits | `control.tile` |
| Accumulator seed | a `seed=True` loop declares each Accum of its *own* body (not of nested blocks) at its identity, in the enclosing scope, before opening its own; an outer name of the same spelling is shadowed, not reset | `control.seed_accumulators` |
| Accum | `name <- op(base or name, value)`: online softmax rescales the accumulator and passes it as `base` | `compute.accum` |
| Max identity | emmy's `-1e30`, not `-inf`: avoids `inf - inf = NaN` in online softmax | `ir.op_identity` |
| Implicit names | `lane`, `warp` bound in the prologue | `lowerer.prologue` |
| Shared memory | `smem_bytes > 0` -> one `.extern` pool carved by a running cursor; else static arrays | `block.smem` |
| Combines | the states are made Variables first (`lowerer.mutable`); `combine_states` is instantiated per step in a fresh scope with `state_b` bound, and its Assigns mutate them | `block.combine` |
| Warp sync masks | `activemask` for `shfl.sync` and `bar.warp.sync`: a sub-warp block (a22 launches 16 threads) or a guarded tile leaves lanes that never arrive, and naming them is undefined | `block.shuffle`, `builder.barrier` |
| TreeHalve | halving tree, barrier outside the branch, root broadcast to every thread | `block.tree_halve` |
| Rounding | `.rn` on float add / sub / mul (also blocks fma contraction); explicit `fma.rn` inside recipes | `ops.py` |
| Transcendentals | exp / log / rsqrt through `ex2` / `lg2` / `rsqrt.approx` (a few ulp); erf, sin, cos, tanh as polynomials at about 2 ulp, because `tanh.approx` is 2^-11 and `sin.approx` has no range reduction; pow as `ex2(y * lg2(abs x))` with C's sign and NaN rules | `ops.py` |
| bool | a `.pred` register: comparisons produce it (`setp`), `where` selects on it (`selp`), arithmetic converts it to 1 / 0 and anything nonzero (NaN included) back to true, as C does; in a buffer it is one byte of 0 / 1, as C stores it | `builder.cvt`, `types.storable`, `ops.py` |
| Coordinates | `//` in a flattened tile index (every pointwise tile of rank >= 2) lowers to `div.s32`: operands are non-negative | `expr.arith` |
| Indexing | one coordinate per axis, row-major; an empty index is element 0 (`sum_1[]`); any other count is `InvalidKernel` rather than a guessed flattening | `expr.address` |
| Type mixes | `cvt` between raw bits and numbers, `setp` on predicates or ordered on bits: `Unsupported` before ptxas sees them | `builder.cvt`, `builder.setp` |

## Extending

| To add | Do |
|---|---|
| an elementwise op | one `@recipe("name")` in `ops.py`, composing others by calling them |
| a node kind | one `@rule("Kind")` in the matching `lower/` module |
| a dtype | a bank in `Ty.bank` and `PREFIX`, conversions in `builder.cvt` |
| a kernel-wide value | bind it in `lowerer.prologue` |

## Deferred, on purpose

Each needs a concrete case before it earns its place:

- an `optimize(module)` pass between lower and render (mov-chain folding, hoisting): the
  module is already data, so it slots in without touching the rules;
- several `KernelOp`s per file (their shared arrays would need entry-qualified names);
- 16-bit dtypes, swizzled shared layouts, `aux_threads`, `serial` axes, and the starred
  nodes of the IR reference (TMA, mbarrier, tensor-core fragments). All raise `Unsupported`.

## Useful links

https://docs.nvidia.com/cuda/parallel-thread-execution/#instruction-set
