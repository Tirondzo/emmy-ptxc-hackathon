extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* add_c1, const float* add_1_c1, const float* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in2__u0 = x[a0 * 4];
        float in1__u0 = add_c1[0];
        float in0__u0 = add_1_c1[0];
        float v7__u0 = ((in1__u0 + in2__u0) * expf((0.0f - fabsf(in2__u0)))) / (in0__u0 + (in2__u0 * in2__u0));
        float in2__u1 = x[a0 * 4 + 1];
        float in1__u1 = add_c1[0];
        float in0__u1 = add_1_c1[0];
        float v7__u1 = ((in1__u1 + in2__u1) * expf((0.0f - fabsf(in2__u1)))) / (in0__u1 + (in2__u1 * in2__u1));
        float in2__u2 = x[a0 * 4 + 2];
        float in1__u2 = add_c1[0];
        float in0__u2 = add_1_c1[0];
        float v7__u2 = ((in1__u2 + in2__u2) * expf((0.0f - fabsf(in2__u2)))) / (in0__u2 + (in2__u2 * in2__u2));
        float in2__u3 = x[a0 * 4 + 3];
        float in1__u3 = add_c1[0];
        float in0__u3 = add_1_c1[0];
        float v7__u3 = ((in1__u3 + in2__u3) * expf((0.0f - fabsf(in2__u3)))) / (in0__u3 + (in2__u3 * in2__u3));
        float4 _vs_v7__u0_24d22975 = make_float4(v7__u0, v7__u1, v7__u2, v7__u3);
        *reinterpret_cast<float4*>(&div[a0 * 4]) = _vs_v7__u0_24d22975;
}
