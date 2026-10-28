extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const float* x, const float* y, float* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = x[a0 * 2];
        float in1__u0 = y[a0 * 2];
        float v1__u0 = powf(fabsf(in0__u0), in1__u0);
        float in0__u1 = x[a0 * 2 + 1];
        float in1__u1 = y[a0 * 2 + 1];
        float v1__u1 = powf(fabsf(in0__u1), in1__u1);
        float2 _vs_v1__u0_8ac6ffff = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&pow_1[a0 * 2]) = _vs_v1__u0_8ac6ffff;
}
