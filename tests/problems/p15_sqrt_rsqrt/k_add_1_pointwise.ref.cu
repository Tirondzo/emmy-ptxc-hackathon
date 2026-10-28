extern "C" __global__
__launch_bounds__(256) void k_add_1_pointwise(const float* add_c1, const float* x, float* add_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 2];
        float in0__u0 = add_c1[0];
        float v5__u0 = rsqrtf((in0__u0 + fabsf(in1__u0))) + sqrtf(fabsf(in1__u0));
        float in1__u1 = x[a0 * 2 + 1];
        float in0__u1 = add_c1[0];
        float v5__u1 = rsqrtf((in0__u1 + fabsf(in1__u1))) + sqrtf(fabsf(in1__u1));
        float2 _vs_v5__u0_47688133 = make_float2(v5__u0, v5__u1);
        *reinterpret_cast<float2*>(&add_1[a0 * 2]) = _vs_v5__u0_47688133;
}
