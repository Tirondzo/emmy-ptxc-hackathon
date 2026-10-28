extern "C" __global__
__launch_bounds__(256) void k_cos_pointwise(const float* mul_c1, const float* x, float* cos) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = mul_c1[0];
        float in1__u0 = x[a0 * 4];
        float v1__u0 = cosf((in0__u0 * in1__u0));
        float in0__u1 = mul_c1[0];
        float in1__u1 = x[a0 * 4 + 1];
        float v1__u1 = cosf((in0__u1 * in1__u1));
        float in0__u2 = mul_c1[0];
        float in1__u2 = x[a0 * 4 + 2];
        float v1__u2 = cosf((in0__u2 * in1__u2));
        float in0__u3 = mul_c1[0];
        float in1__u3 = x[a0 * 4 + 3];
        float v1__u3 = cosf((in0__u3 * in1__u3));
        float4 _vs_v1__u0_d31bd730 = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&cos[a0 * 4]) = _vs_v1__u0_d31bd730;
}
