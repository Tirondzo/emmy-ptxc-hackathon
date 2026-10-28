static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem) : "memory");
}

template <int Bytes>
static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

extern "C" __global__
__launch_bounds__(128) void k_bmm(const float* x1, const float* x0, float* bmm) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a3_ks = _gid / 1024;
        int a0 = (_gid / 256) % 4;
        int a1_b = (_gid / 256) % 1;
        int a2_b = (_gid / 128) % 2;
        int a1_u = (_gid / 16) % 8;
        int a2_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c4_1 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c5_1 = 0.0f;
        float acc0__c6_0 = 0.0f;
        float acc0__c6_1 = 0.0f;
        float acc0__c7_0 = 0.0f;
        float acc0__c7_1 = 0.0f;
        __shared__ __align__(16) float _a_smem[256];
        __shared__ __align__(16) float _b_smem[128];
        for (int _ks = 0; _ks < 4; _ks += 4) {
            for (int _fa = a1_u * 16 + a2_u; _fa < 64; _fa += 128) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[a0 * 4096 + (a1_b * 64 + _fa) * 64 + (_a3_ks * 4 + _ks)]);
            }
            for (int _fb = a1_u * 16 + a2_u; _fb < 32; _fb += 128) {
                emmy_cp_async_cg(&_b_smem[_fb * 4], &x1[a0 * 4096 + (_a3_ks * 4 + (_ks + _fb / 8)) * 64 + (a2_b * 32 + 4 * (_fb % 8))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 4; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 32 + (a2_b * 32 + a2_u * 2 - a2_b * 32)];
                float in0__c0_0 = _a_smem[(a1_b * 64 + a1_u * 8 - a1_b * 64) * 4 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c0_1 = _b_smem[_ki * 32 + (a2_b * 32 + a2_u * 2 + 1 - a2_b * 32)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                float in0__c1_0 = _a_smem[(a1_b * 64 + a1_u * 8 + 1 - a1_b * 64) * 4 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
                float in0__c2_0 = _a_smem[(a1_b * 64 + a1_u * 8 + 2 - a1_b * 64) * 4 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float acc0__v__c2_1 = in1__c0_1 * in0__c2_0;
                acc0__c2_1 += acc0__v__c2_1;
                float in0__c3_0 = _a_smem[(a1_b * 64 + a1_u * 8 + 3 - a1_b * 64) * 4 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float acc0__v__c3_1 = in1__c0_1 * in0__c3_0;
                acc0__c3_1 += acc0__v__c3_1;
                float in0__c4_0 = _a_smem[(a1_b * 64 + a1_u * 8 + 4 - a1_b * 64) * 4 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                float acc0__v__c4_1 = in1__c0_1 * in0__c4_0;
                acc0__c4_1 += acc0__v__c4_1;
                float in0__c5_0 = _a_smem[(a1_b * 64 + a1_u * 8 + 5 - a1_b * 64) * 4 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                float acc0__v__c5_1 = in1__c0_1 * in0__c5_0;
                acc0__c5_1 += acc0__v__c5_1;
                float in0__c6_0 = _a_smem[(a1_b * 64 + a1_u * 8 + 6 - a1_b * 64) * 4 + _ki];
                float acc0__v__c6_0 = in1__c0_0 * in0__c6_0;
                acc0__c6_0 += acc0__v__c6_0;
                float acc0__v__c6_1 = in1__c0_1 * in0__c6_0;
                acc0__c6_1 += acc0__v__c6_1;
                float in0__c7_0 = _a_smem[(a1_b * 64 + a1_u * 8 + 7 - a1_b * 64) * 4 + _ki];
                float acc0__v__c7_0 = in1__c0_0 * in0__c7_0;
                acc0__c7_0 += acc0__v__c7_0;
                float acc0__v__c7_1 = in1__c0_1 * in0__c7_0;
                acc0__c7_1 += acc0__v__c7_1;
            }
            __syncthreads();
        }
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8) * 64 + (a2_b * 32 + a2_u * 2)], acc0__c0_0);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8) * 64 + (a2_b * 32 + a2_u * 2 + 1)], acc0__c0_1);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 1) * 64 + (a2_b * 32 + a2_u * 2)], acc0__c1_0);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 1) * 64 + (a2_b * 32 + a2_u * 2 + 1)], acc0__c1_1);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 2) * 64 + (a2_b * 32 + a2_u * 2)], acc0__c2_0);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 2) * 64 + (a2_b * 32 + a2_u * 2 + 1)], acc0__c2_1);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 3) * 64 + (a2_b * 32 + a2_u * 2)], acc0__c3_0);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 3) * 64 + (a2_b * 32 + a2_u * 2 + 1)], acc0__c3_1);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 4) * 64 + (a2_b * 32 + a2_u * 2)], acc0__c4_0);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 4) * 64 + (a2_b * 32 + a2_u * 2 + 1)], acc0__c4_1);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 5) * 64 + (a2_b * 32 + a2_u * 2)], acc0__c5_0);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 5) * 64 + (a2_b * 32 + a2_u * 2 + 1)], acc0__c5_1);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 6) * 64 + (a2_b * 32 + a2_u * 2)], acc0__c6_0);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 6) * 64 + (a2_b * 32 + a2_u * 2 + 1)], acc0__c6_1);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 7) * 64 + (a2_b * 32 + a2_u * 2)], acc0__c7_0);
        atomicAdd(&bmm[a0 * 4096 + (a1_b * 64 + a1_u * 8 + 7) * 64 + (a2_b * 32 + a2_u * 2 + 1)], acc0__c7_1);
}
