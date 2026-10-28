extern "C" __global__
__launch_bounds__(256) void k_erf_pointwise(const float* x, float* erf) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float2 _v_in0__u0 = *reinterpret_cast<const float2*>(&x[a0 * 2]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float v0__u0 = erff(in0__u0);
        float v0__u1 = erff(in0__u1);
        float2 _vs_v0__u0_fe3bfacc = make_float2(v0__u0, v0__u1);
        *reinterpret_cast<float2*>(&erf[a0 * 2]) = _vs_v0__u0_fe3bfacc;
}
