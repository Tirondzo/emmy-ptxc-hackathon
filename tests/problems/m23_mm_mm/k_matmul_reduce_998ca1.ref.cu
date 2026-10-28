extern "C" __global__
__launch_bounds__(256) void k_matmul_reduce_998ca1(const float* x2, const float* x1, const float* x0, float* matmul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 16384;
        int a1 = (_gid / 256) % 64;
        int a2_co = (_gid) % 256;
        float acc1 = 0.0f;
        float acc1__r1 = 0.0f;
        for (int a2 = a2_co; a2 < 64; a2 += 512) {
            float acc0 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in0 = x0[a0 * 64 + a3];
                float in1 = x1[a3 * 64 + a2];
                float v0 = in0 * in1;
                acc0 += v0;
            }
            float in2 = x2[a2 * 64 + a1];
            float v1 = acc0 * in2;
            acc1 += v1;
            float acc0__r1 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in0__r1 = x0[a0 * 64 + a3];
                float in1__r1 = x1[a3 * 64 + (a2 + 256) % 64];
                float v0__r1 = in0__r1 * in1__r1;
                acc0__r1 += v0__r1;
            }
            float in2__r1 = x2[(a2 + 256) % 64 * 64 + a1];
            float v1__r1 = acc0__r1 * in2__r1;
            float v1__r1__id = 0.0f;
            float v1__r1__m = ((a2 + 256 < 64) ? (((float)(v1__r1))) : (((float)(v1__r1__id))));
            acc1__r1 += v1__r1__m;
        }
        acc1 += acc1__r1;
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            acc1 = acc1 + acc1__o;
        }
        __shared__ float acc1_smem[8];
        if (lane == 0) {
            acc1_smem[warp] = acc1;
        }
        __syncthreads();
        for (int s = 4; s > 0; s >>= 1) {
            if (warp < s) {
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                acc1 = acc1 + acc1__o;
                acc1_smem[warp] = acc1;
            }
            __syncthreads();
        }
        acc1 = acc1_smem[0];
        if (a2_co == 0) {
            matmul_1[a0 * 64 + a1] = acc1;
        }
}
