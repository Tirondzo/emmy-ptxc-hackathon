extern "C" __global__
__launch_bounds__(256) void k_gt_pointwise(const float* gt_c1, const float* x, bool* gt) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = gt_c1[0];
        float in1__u0 = x[a0 * 2];
        bool v0__u0 = in1__u0 > in0__u0;
        float in0__u1 = gt_c1[0];
        float in1__u1 = x[a0 * 2 + 1];
        bool v0__u1 = in1__u1 > in0__u1;
        gt[a0 * 2] = v0__u0;
        gt[a0 * 2 + 1] = v0__u1;
}
