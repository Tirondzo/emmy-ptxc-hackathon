extern "C" __global__
__launch_bounds__(256) void k_log_pointwise(const float* mul_c1, const float* add_c1, const float* x, float* log) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in2__u0 = x[a0 * 4];
        float in1__u0 = mul_c1[0];
        float in0__u0 = add_c1[0];
        float v3__u0 = log((in0__u0 + (in1__u0 * fabsf(in2__u0))));
        float in2__u1 = x[a0 * 4 + 1];
        float in1__u1 = mul_c1[0];
        float in0__u1 = add_c1[0];
        float v3__u1 = log((in0__u1 + (in1__u1 * fabsf(in2__u1))));
        float in2__u2 = x[a0 * 4 + 2];
        float in1__u2 = mul_c1[0];
        float in0__u2 = add_c1[0];
        float v3__u2 = log((in0__u2 + (in1__u2 * fabsf(in2__u2))));
        float in2__u3 = x[a0 * 4 + 3];
        float in1__u3 = mul_c1[0];
        float in0__u3 = add_c1[0];
        float v3__u3 = log((in0__u3 + (in1__u3 * fabsf(in2__u3))));
        float4 _vs_v3__u0_0506ec0d = make_float4(v3__u0, v3__u1, v3__u2, v3__u3);
        *reinterpret_cast<float4*>(&log[a0 * 4]) = _vs_v3__u0_0506ec0d;
}
