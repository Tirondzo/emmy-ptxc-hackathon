extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* add_c1, const float* y, const float* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 128) {
        int a0 = _gid;
        float in2__u0 = y[a0 * 4];
        float in0__u0 = add_c1[0];
        float in1__u0 = x[a0 * 4];
        float v2__u0 = in1__u0 / (in0__u0 + fabsf(in2__u0));
        float in2__u1 = y[a0 * 4 + 1];
        float in0__u1 = add_c1[0];
        float in1__u1 = x[a0 * 4 + 1];
        float v2__u1 = in1__u1 / (in0__u1 + fabsf(in2__u1));
        float in2__u2 = y[a0 * 4 + 2];
        float in0__u2 = add_c1[0];
        float in1__u2 = x[a0 * 4 + 2];
        float v2__u2 = in1__u2 / (in0__u2 + fabsf(in2__u2));
        float in2__u3 = y[a0 * 4 + 3];
        float in0__u3 = add_c1[0];
        float in1__u3 = x[a0 * 4 + 3];
        float v2__u3 = in1__u3 / (in0__u3 + fabsf(in2__u3));
        float4 _vs_v2__u0_a17c1487 = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&div[a0 * 4]) = _vs_v2__u0_a17c1487;
    }
}
