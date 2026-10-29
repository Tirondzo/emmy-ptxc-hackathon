extern "C" __global__
__launch_bounds__(256) void k_floor_divide_pointwise(const float* mul_c1, const int* floor_divide_c1, const float* x, int* floor_divide) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = mul_c1[0];
        float in2__u0 = x[a0 * 2];
        int v1__u0 = (in1__u0 * in2__u0);
        int in0__u0 = floor_divide_c1[0];
        int v2__u0 = v1__u0 / in0__u0;
        float in1__u1 = mul_c1[0];
        float in2__u1 = x[a0 * 2 + 1];
        int v1__u1 = (in1__u1 * in2__u1);
        int in0__u1 = floor_divide_c1[0];
        int v2__u1 = v1__u1 / in0__u1;
        floor_divide[a0 * 2] = v2__u0;
        floor_divide[a0 * 2 + 1] = v2__u1;
}
