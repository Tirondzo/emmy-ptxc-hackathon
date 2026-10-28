extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const float* lt_c1, const float* gt_c1, const float* x, float* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = gt_c1[0];
        float in2__u0 = x[a0 * 2];
        bool v0__u0 = in2__u0 > in0__u0;
        float in1__u0 = lt_c1[0];
        bool v1__u0 = in2__u0 < in1__u0;
        float v4__u0 = (((v0__u0 && v1__u0)) ? (in2__u0) : ((0.0f - in2__u0)));
        float in0__u1 = gt_c1[0];
        float in2__u1 = x[a0 * 2 + 1];
        bool v0__u1 = in2__u1 > in0__u1;
        float in1__u1 = lt_c1[0];
        bool v1__u1 = in2__u1 < in1__u1;
        float v4__u1 = (((v0__u1 && v1__u1)) ? (in2__u1) : ((0.0f - in2__u1)));
        float2 _vs_v4__u0_7b24a16e = make_float2(v4__u0, v4__u1);
        *reinterpret_cast<float2*>(&where[a0 * 2]) = _vs_v4__u0_7b24a16e;
}
