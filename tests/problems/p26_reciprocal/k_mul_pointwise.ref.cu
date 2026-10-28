extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* add_c1, const float* mul_c1, const float* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in2__u0 = x[a0 * 2];
        float in0__u0 = add_c1[0];
        float in1__u0 = mul_c1[0];
        float v3__u0 = in1__u0 * (1.0f / (in0__u0 + fabsf(in2__u0)));
        float in2__u1 = x[a0 * 2 + 1];
        float in0__u1 = add_c1[0];
        float in1__u1 = mul_c1[0];
        float v3__u1 = in1__u1 * (1.0f / (in0__u1 + fabsf(in2__u1)));
        float2 _vs_v3__u0_01b2bbc2 = make_float2(v3__u0, v3__u1);
        *reinterpret_cast<float2*>(&mul[a0 * 2]) = _vs_v3__u0_01b2bbc2;
}
