extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const float* mul_c1, const float* add_c1, const float* x, const float* y, float* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in2__u0 = x[a0 * 2];
        float in0__u0 = add_c1[0];
        float in1__u0 = mul_c1[0];
        float in3__u0 = y[a0 * 2];
        float v3__u0 = powf((in0__u0 + fabsf(in2__u0)), (in1__u0 * in3__u0));
        float in2__u1 = x[a0 * 2 + 1];
        float in0__u1 = add_c1[0];
        float in1__u1 = mul_c1[0];
        float in3__u1 = y[a0 * 2 + 1];
        float v3__u1 = powf((in0__u1 + fabsf(in2__u1)), (in1__u1 * in3__u1));
        float2 _vs_v3__u0_10ff7ba9 = make_float2(v3__u0, v3__u1);
        *reinterpret_cast<float2*>(&pow_1[a0 * 2]) = _vs_v3__u0_10ff7ba9;
}
