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

- **Emit returns an operand.** `expr.value`, `Lowerer.apply` and the builder's `op` / `cvt` /
  `setp` return a typed `Reg` or `Imm`. Callers never care how a value was produced, and
  constants fold on the way up: `expr.linear` turns `(a0 * 4) + 3` into a `[base+12]`
  displacement.
- **One rule per node kind.** `@rule("Kind")` in `lower/{compute,control,block}.py`. An
  unknown kind, op, field value or dtype raises `Unsupported` compilation error.
- **Rebinding vs mutation.** SSA names are `Bound` and rebound freely. Assume `Load`, `Assign`, `Let` are bound to SSA. An SSA name redefined from inside a nested block is `InvalidKernel`. We trust the IR's `Accum` / `Init` marks instead of analysing data flow.
- **Registers are per bank.** `Ty.bank` maps s32 / u32 / b32 to one `.b32` bank; register
  names and counts are per bank, never per type.

## Decided by the IR's semantics

| Topic | Rule | Where |
|---|---|---|
| Tile decode | row-major over every listed axis; guard `gid < cells` when the launch has spare threads | `control.tile` |
| Rasterization | ignored for now. Doesn't affect correctness, only cache hits | `control.tile` |
| Accumulator seed | a `seed=True` loop seeds the Accums of its *immediate* body (through Cond, not nested loops), in the enclosing scope, before opening its own | `control.seed_accumulators` |
| Accum | `name <- op(base or name, value)`: online softmax rescales the accumulator and passes it as `base` | `compute.accum` |
| Max identity | emmy's `-1e30`, not `-inf`: avoids `inf - inf = NaN` in online softmax | `ir.op_identity` |
| Implicit names | `lane`, `warp` bound in the prologue | `Lowerer.prologue` |
| Shared memory | `smem_bytes > 0` -> one `.extern` pool carved by a running cursor; else static arrays | `block.smem` |
| Combines | `combine_states` instantiated per step in a fresh scope with `state_b` bound; its Assigns mutate the states | `block.combine` |
| TreeHalve | halving tree, barrier outside the branch, root broadcast to every thread | `block.tree_halve` |
| Rounding | `.rn` on float add / sub / mul (also blocks fma contraction), `.approx` ignored | `ops.py` |

## Extending

| To add | Do |
|---|---|
| an elementwise op | one `@recipe("name")` in `ops.py`, composing others by calling them |
| a node kind | one `@rule("Kind")` in the matching `lower/` module |
| a dtype | a bank in `Ty.bank` and `PREFIX`, conversions in `PtxBuilder.cvt` |
| a kernel-wide value | bind it in `Lowerer.prologue` |

## Deferred, on purpose

Each needs a concrete case before it earns its place:

- an `optimize(module)` pass between lower and render (mov-chain folding, hoisting): the
  module is already data, so it slots in without touching the rules;
- several `KernelOp`s per file (their shared arrays would need entry-qualified names);
- 16-bit dtypes, swizzled shared layouts, `aux_threads`, `serial` axes, and the starred
  nodes of the IR reference (TMA, mbarrier, tensor-core fragments). All raise `Unsupported`.
