# Julia set demo

emmy unrolls a 32-step loop of `z = z^2 + c` into one pointwise kernel
([`../problems/p99_julia`](../problems/p99_julia)); ptxc compiles it. Run from this folder.

Once:

```
python julia.py plane plane.npz          # the inputs: a 720x1280 grid over the complex plane
```

Once per PowerShell session (ptxas comes with the nvidia-cuda-nvcc wheel and reads ASCII only):

```
$env:PATH += ";$PWD\..\..\..\.venv\Lib\site-packages\nvidia\cu13\bin"
$PSDefaultParameterValues['Out-File:Encoding'] = 'ascii'
```

Each run:

```
python ../../compile.py ../problems/p99_julia/k_add_125_pointwise.ir.json > julia.ptx   # ptxc: IR -> PTX
ptxas -arch=sm_120 -v julia.ptx -o julia.cubin                                          # PTX -> machine code (RTX 50xx)
python launch.py julia.cubin ../problems/p99_julia/meta.json --inputs plane.npz --out julia.npz
python julia.py draw julia.npz                                                          # julia.png, checked against numpy
start julia.png
```
