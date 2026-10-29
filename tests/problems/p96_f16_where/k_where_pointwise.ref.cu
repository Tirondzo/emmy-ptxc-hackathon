#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const __half* mul_c1, const float* gt_c1, const __half* x, __half* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = gt_c1[0];
        __half in2__u0 = x[a0 * 2];
        bool v0__u0 = __half2float(in2__u0) > in0__u0;
        __half in1__u0 = mul_c1[0];
        __half v2__u0 = __float2half(((v0__u0) ? (__half2float(in2__u0)) : (__half2float((in1__u0 * in2__u0)))));
        float in0__u1 = gt_c1[0];
        __half in2__u1 = x[a0 * 2 + 1];
        bool v0__u1 = __half2float(in2__u1) > in0__u1;
        __half in1__u1 = mul_c1[0];
        __half v2__u1 = __float2half(((v0__u1) ? (__half2float(in2__u1)) : (__half2float((in1__u1 * in2__u1)))));
        __half2 _vs_v2__u0_445605f3 = __halves2half2(v2__u0, v2__u1);
        *reinterpret_cast<__half2*>(&where[a0 * 2]) = _vs_v2__u0_445605f3;
}
