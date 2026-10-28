extern "C" __global__
__launch_bounds__(64) void k_matmul_reduce_0d13ea(const float* q, const float* kt, const float* v, float* matmul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 4096;
        int a1 = (_gid / 64) % 64;
        int a2_co = (_gid) % 64;
        float acc1 = 0.0f;
        float acc1__r1 = 0.0f;
        float acc1__r2 = 0.0f;
        float acc1__r3 = 0.0f;
        for (int a2 = a2_co; a2 < 64; a2 += 256) {
            float acc0 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in1 = q[a0 * 64 + a3];
                float in0 = kt[a3 * 64 + a2];
                float v0 = in0 * in1;
                acc0 += v0;
            }
            float in2 = v[a2 * 64 + a1];
            float v2 = in2 * fmaxf(0.0f, acc0);
            acc1 += v2;
            float acc0__r1 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in1__r1 = q[a0 * 64 + a3];
                float in0__r1 = kt[a3 * 64 + (a2 + 64) % 64];
                float v0__r1 = in0__r1 * in1__r1;
                acc0__r1 += v0__r1;
            }
            float in2__r1 = v[(a2 + 64) % 64 * 64 + a1];
            float v2__r1 = in2__r1 * fmaxf(0.0f, acc0__r1);
            float v2__r1__id = 0.0f;
            float v2__r1__m = ((a2 + 64 < 64) ? (((float)(v2__r1))) : (((float)(v2__r1__id))));
            acc1__r1 += v2__r1__m;
            float acc0__r2 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in1__r2 = q[a0 * 64 + a3];
                float in0__r2 = kt[a3 * 64 + (a2 + 128) % 64];
                float v0__r2 = in0__r2 * in1__r2;
                acc0__r2 += v0__r2;
            }
            float in2__r2 = v[(a2 + 128) % 64 * 64 + a1];
            float v2__r2 = in2__r2 * fmaxf(0.0f, acc0__r2);
            float v2__r2__id = 0.0f;
            float v2__r2__m = ((a2 + 128 < 64) ? (((float)(v2__r2))) : (((float)(v2__r2__id))));
            acc1__r2 += v2__r2__m;
            float acc0__r3 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in1__r3 = q[a0 * 64 + a3];
                float in0__r3 = kt[a3 * 64 + (a2 + 192) % 64];
                float v0__r3 = in0__r3 * in1__r3;
                acc0__r3 += v0__r3;
            }
            float in2__r3 = v[(a2 + 192) % 64 * 64 + a1];
            float v2__r3 = in2__r3 * fmaxf(0.0f, acc0__r3);
            float v2__r3__id = 0.0f;
            float v2__r3__m = ((a2 + 192 < 64) ? (((float)(v2__r3))) : (((float)(v2__r3__id))));
            acc1__r3 += v2__r3__m;
        }
        acc1 += acc1__r1;
        acc1 += acc1__r2;
        acc1 += acc1__r3;
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
        __shared__ float acc1_smem[2];
        if (lane == 0) {
            acc1_smem[warp] = acc1;
        }
        __syncthreads();
        for (int s = 1; s > 0; s >>= 1) {
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
