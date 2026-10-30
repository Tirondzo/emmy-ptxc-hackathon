# IR2PTX Python compiler
Made as a submission to [emmy-hackathon-2026](https://github.com/cloudrift-ai/emmy-hackathon-2026) with submission **[Video](https://youtu.be/Ej83R-K1Aqg)**.

```sh
python compile.py path/to/kernel.ir.json > out.ptx
```

`pip install -r requirements.txt` Requires Python 3.12 or newer. \
`python compile.py -h` lists more cli options: `--unroll`, `--precision` and others...

## Detailed docs (ai-friendly)
### [Architecture](./ARCHITECTURE.md) with all major decisions and details.
### [Tests Toolkit](./tests/README.md) with edge cases and [Findings](./tests/FINDINGS.md).

## Short description (human-friendly)
> [!NOTE]
> This is the [`extra`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/extra) branch: [`main`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/main) plus non-required features, demonstrating the compiler's extensibility.
>
> [`main`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/main): a balance of required correctness, simplicity and performance, without the extras.
>
> [`simple`](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/simple): correctness only, no performance optimizations or trade-offs.

### What `extra` adds

- Tensor cores: emmy's f16 matmul through `cp.async` into swizzled slabs, `ldmatrix` and `mma.sync`, within 3% of emmy's CUDA. One new rule module, [`fragments.py`](./ptxc/lower/fragments.py).
- f64 transcendentals: exp, log, sin, cos, tanh, erf and pow as `fma.rn.f64` series, 1 to 3 ulp against numpy ([test_f64.py](./tests/test_f64.py)).
- `Select`, `FlatIndex` and the bitwise index operators: starred nodes no open problem uses.
- A [Julia set demo](./tests/julia_demo/README.md): a 32-step loop emmy unrolls into one pointwise kernel, compiled by ptxc and drawn as a picture.


### Correctness

Correctness was the main focus, covered by lots of synthetic tests (> 100) generated from the real code using `emmy`.

Edge cases found during development became tests too. A few of them show that emmy's own CUDA disagrees with its expected outputs (p91, p92, p93, p94, p95).

A few are proposed for the hidden set, to check other participants' submissions as well. (See [hacks.py](./tests/hacks.py)).


### Architecture

Architecture is where most of the manual work went.

- The fundamental: the compiler grows by adding a `@rule` or a `@recipe`.
- Readability: lowering state lives in a `ContextVar`, so rules call primitives directly instead of threading a context through every function. MLIR, Modular MAX, PyTorch Inductor, JAX, rustc and GCC do the same, for example [MLIR's context management](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Bindings/Python.md#context-management).
- Configs and assumptions live in one place: [`config.py`](./ptxc/config.py) and [`options.py`](./ptxc/options.py).
- Single pass. Multi-pass stays the last resort, not yet required.
- No phi nodes: we trust the IR's `Accum` and `Init` semantics instead.

### Performance

RTX 5080, the minimum over 3 to 5 runs of the median of 30 launches. \
Method and more kernels in [FINDINGS.md](./tests/FINDINGS.md). Gain is After over Before.

Loop unrolling and partial unrolling: rolled vs unrolled.

| Kernel | Before | After | emmy's CUDA | Gain |
|:---|---:|---:|---:|---:|
| a01_attention | 87.9 µs | 65.2 µs | 79.6 µs | 1.35× |
| a02_attention_128 | 45.9 µs | 25.3 µs | 37.2 µs | 1.81× |
| m01_matmul_512 | 76.7 µs | 32.5 µs | 30.4 µs | 2.36× |
| r02_big_sum | 25.5 µs | 13.3 µs | 15.2 µs | 1.92× |
| r03_mean | 16.4 µs | 9.2 µs | 11.2 µs | 1.78× |

Approx precision + loop unrolling: exact `.rn` vs auto budget, both unrolled.

| Kernel | Before | After | emmy's CUDA | Gain |
|:---|---:|---:|---:|---:|
| a12_256 | 600.8 µs | 380.7 µs | 985.4 µs | 1.58× |
| m14_1024 | 217.8 µs | 156.3 µs | 146.2 µs | 1.39× |
| a01_attention | 65.3 µs | 55.0 µs | 78.6 µs | 1.19× |
| a19_long_keys | 445.7 µs | 384.8 µs | 553.5 µs | 1.16× |
| m01_matmul_512 | 32.5 µs | 30.5 µs | 29.9 µs | 1.07× |

### Submission [Video](https://youtu.be/Ej83R-K1Aqg)