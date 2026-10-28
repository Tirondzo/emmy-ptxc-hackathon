extern "C" __global__
__launch_bounds__(128) void k_div_reduce(const float* y, const float* x, float* div) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        for (int a1 = a1_co; a1 < 256; a1 += 128) {
            float in0 = x[a0 * 256 + a1];
            float v0 = in0 * in0;
            float in1 = y[a0 * 256 + a1];
            float v1 = in1 * in1;
            float v2 = in0 * in1;
            acc0 += v0;
            acc1 += v1;
            acc2 += v2;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 16);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 8);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 4);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 2);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 1);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
            acc2 = acc2 + acc2__o;
        }
        __shared__ float acc0_smem[4];
        __shared__ float acc1_smem[4];
        __shared__ float acc2_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
            acc1_smem[warp] = acc1;
            acc2_smem[warp] = acc2;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                float acc2 = acc2_smem[warp];
                float acc2__o = acc2_smem[warp + s];
                acc0 = acc0 + acc0__o;
                acc1 = acc1 + acc1__o;
                acc2 = acc2 + acc2__o;
                acc0_smem[warp] = acc0;
                acc1_smem[warp] = acc1;
                acc2_smem[warp] = acc2;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        acc1 = acc1_smem[0];
        acc2 = acc2_smem[0];
        if (a1_co == 0) {
            float v6 = acc2 / (sqrtf(acc1) * sqrtf(acc0));
            div[a0] = v6;
        }
}
