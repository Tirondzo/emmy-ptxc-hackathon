# IR2PTX Python compiler
Made as a submission to [emmy-hackathon-2026](https://github.com/cloudrift-ai/emmy-hackathon-2026) with submission **[Video](https://youtu.be/Ej83R-K1Aqg)**.

```sh
python compile.py path/to/kernel.ir.json > out.ptx
```

`pip install -r requirements.txt` Requires Python 3.12 or newer.

## Detailed docs (ai-friendly)
### [Architecture](./ARCHITECTURE.md) with all major decisions and details.
### [Tests Toolkit](./tests/README.md) with edge cases and [Findings](./tests/FINDINGS.md).

## Short description (human-friendly)
> [!NOTE]
> This is the [`simple`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/simple) branch: correctness only, no performance optimizations or trade-offs, no cli options.
>
> [`main`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/main): a balance of required correctness, simplicity and performance, without non-required [`extra`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/extra) features.
>
> [`extra`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/extra) extends [`main`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/main) with more features, demonstrating the compiler's extensibility.


### Correctness

Correctness was the main focus, covered by lots of synthetic tests (> 100) generated from the real code using `emmy`.

Edge cases found during development became tests too.


### Architecture

Architecture is where most of the manual work went.

- The fundamental: the compiler grows by adding a `@rule` or a `@recipe`.
- Readability: lowering state lives in a `ContextVar`, so rules call primitives directly instead of threading a context through every function. MLIR, Modular MAX, PyTorch Inductor, JAX, rustc and GCC do the same, for example [MLIR's context management](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Bindings/Python.md#context-management).
- Configs and assumptions live in one place: [`config.py`](./ptxc/config.py).
- Single pass. Multi-pass stays the last resort, not yet required.
- No phi nodes: we trust the IR's `Accum` and `Init` semantics instead.

### Performance

Optimizations are not a priority for this branch: every float op is IEEE `.rn`, every loop stays rolled.
Loop unrolling and the error budget, with measurements, are on [`main`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/main).

### Submission [Video](https://youtu.be/Ej83R-K1Aqg)
