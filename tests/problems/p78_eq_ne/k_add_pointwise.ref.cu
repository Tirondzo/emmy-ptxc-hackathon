extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float2 _v_in0__u0 = *reinterpret_cast<const float2*>(&x[a0 * 2]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        bool v2__u0 = in0__u0 == fabsf(in0__u0);
        bool v5__u0 = in0__u0 != fabsf(in0__u0);
        float v6__u0 = ((v2__u0) ? (in0__u0) : ((0.0f - in0__u0)));
        float v7__u0 = ((v5__u0) ? ((0.0f - in0__u0)) : (in0__u0));
        float v8__u0 = v6__u0 + v7__u0;
        bool v2__u1 = in0__u1 == fabsf(in0__u1);
        bool v5__u1 = in0__u1 != fabsf(in0__u1);
        float v6__u1 = ((v2__u1) ? (in0__u1) : ((0.0f - in0__u1)));
        float v7__u1 = ((v5__u1) ? ((0.0f - in0__u1)) : (in0__u1));
        float v8__u1 = v6__u1 + v7__u1;
        float2 _vs_v8__u0_8c89bfb7 = make_float2(v8__u0, v8__u1);
        *reinterpret_cast<float2*>(&add[a0 * 2]) = _vs_v8__u0_8c89bfb7;
}
