extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* y, const float* x, const float* z, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = x[a0 * 2];
        float in1__u0 = y[a0 * 2];
        float in2__u0 = z[a0 * 2];
        float v1__u0 = in2__u0 + (in0__u0 * in1__u0);
        float in0__u1 = x[a0 * 2 + 1];
        float in1__u1 = y[a0 * 2 + 1];
        float in2__u1 = z[a0 * 2 + 1];
        float v1__u1 = in2__u1 + (in0__u1 * in1__u1);
        float2 _vs_v1__u0_13c38830 = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&add[a0 * 2]) = _vs_v1__u0_13c38830;
}
