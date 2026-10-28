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
__launch_bounds__(128) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 6144;
        int a0_b = (_gid / 1024) % 6;
        int a1_b = (_gid / 128) % 8;
        int a0_u = (_gid / 16) % 8;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        float acc0__c0_3 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c1_2 = 0.0f;
        float acc0__c1_3 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c2_2 = 0.0f;
        float acc0__c2_3 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        float acc0__c3_2 = 0.0f;
        float acc0__c3_3 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c4_1 = 0.0f;
        float acc0__c4_2 = 0.0f;
        float acc0__c4_3 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c5_1 = 0.0f;
        float acc0__c5_2 = 0.0f;
        float acc0__c5_3 = 0.0f;
        float acc0__c6_0 = 0.0f;
        float acc0__c6_1 = 0.0f;
        float acc0__c6_2 = 0.0f;
        float acc0__c6_3 = 0.0f;
        float acc0__c7_0 = 0.0f;
        float acc0__c7_1 = 0.0f;
        float acc0__c7_2 = 0.0f;
        float acc0__c7_3 = 0.0f;
        float acc0__c8_0 = 0.0f;
        float acc0__c8_1 = 0.0f;
        float acc0__c8_2 = 0.0f;
        float acc0__c8_3 = 0.0f;
        float acc0__c9_0 = 0.0f;
        float acc0__c9_1 = 0.0f;
        float acc0__c9_2 = 0.0f;
        float acc0__c9_3 = 0.0f;
        float acc0__c10_0 = 0.0f;
        float acc0__c10_1 = 0.0f;
        float acc0__c10_2 = 0.0f;
        float acc0__c10_3 = 0.0f;
        float acc0__c11_0 = 0.0f;
        float acc0__c11_1 = 0.0f;
        float acc0__c11_2 = 0.0f;
        float acc0__c11_3 = 0.0f;
        __shared__ __align__(16) float _a_smem[384];
        __shared__ __align__(16) float _b_smem[256];
        for (int _ks = 0; _ks < 4; _ks += 4) {
            for (int _fa = a0_u * 16 + a1_u; _fa < 96; _fa += 128) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[((a0_b * 96 + _fa < 512) ? (a0_b * 96 + _fa) : (511)) * 64 + (_a2_ks * 4 + _ks)]);
            }
            for (int _fb = a0_u * 16 + a1_u; _fb < 64; _fb += 128) {
                emmy_cp_async_cg(&_b_smem[_fb * 4], &x1[(_a2_ks * 4 + (_ks + _fb / 16)) * 512 + (a1_b * 64 + 4 * (_fb % 16))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 4; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u * 4 - a1_b * 64)];
                float in0__c0_0 = _a_smem[(a0_b * 96 + a0_u * 12 - a0_b * 96) * 4 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c0_1 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u * 4 + 1 - a1_b * 64)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                float in1__c0_2 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u * 4 + 2 - a1_b * 64)];
                float acc0__v__c0_2 = in1__c0_2 * in0__c0_0;
                acc0__c0_2 += acc0__v__c0_2;
                float in1__c0_3 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u * 4 + 3 - a1_b * 64)];
                float acc0__v__c0_3 = in1__c0_3 * in0__c0_0;
                acc0__c0_3 += acc0__v__c0_3;
                float in0__c1_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 1 - a0_b * 96) * 4 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
                float acc0__v__c1_2 = in1__c0_2 * in0__c1_0;
                acc0__c1_2 += acc0__v__c1_2;
                float acc0__v__c1_3 = in1__c0_3 * in0__c1_0;
                acc0__c1_3 += acc0__v__c1_3;
                float in0__c2_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 2 - a0_b * 96) * 4 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float acc0__v__c2_1 = in1__c0_1 * in0__c2_0;
                acc0__c2_1 += acc0__v__c2_1;
                float acc0__v__c2_2 = in1__c0_2 * in0__c2_0;
                acc0__c2_2 += acc0__v__c2_2;
                float acc0__v__c2_3 = in1__c0_3 * in0__c2_0;
                acc0__c2_3 += acc0__v__c2_3;
                float in0__c3_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 3 - a0_b * 96) * 4 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float acc0__v__c3_1 = in1__c0_1 * in0__c3_0;
                acc0__c3_1 += acc0__v__c3_1;
                float acc0__v__c3_2 = in1__c0_2 * in0__c3_0;
                acc0__c3_2 += acc0__v__c3_2;
                float acc0__v__c3_3 = in1__c0_3 * in0__c3_0;
                acc0__c3_3 += acc0__v__c3_3;
                float in0__c4_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 4 - a0_b * 96) * 4 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                float acc0__v__c4_1 = in1__c0_1 * in0__c4_0;
                acc0__c4_1 += acc0__v__c4_1;
                float acc0__v__c4_2 = in1__c0_2 * in0__c4_0;
                acc0__c4_2 += acc0__v__c4_2;
                float acc0__v__c4_3 = in1__c0_3 * in0__c4_0;
                acc0__c4_3 += acc0__v__c4_3;
                float in0__c5_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 5 - a0_b * 96) * 4 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                float acc0__v__c5_1 = in1__c0_1 * in0__c5_0;
                acc0__c5_1 += acc0__v__c5_1;
                float acc0__v__c5_2 = in1__c0_2 * in0__c5_0;
                acc0__c5_2 += acc0__v__c5_2;
                float acc0__v__c5_3 = in1__c0_3 * in0__c5_0;
                acc0__c5_3 += acc0__v__c5_3;
                float in0__c6_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 6 - a0_b * 96) * 4 + _ki];
                float acc0__v__c6_0 = in1__c0_0 * in0__c6_0;
                acc0__c6_0 += acc0__v__c6_0;
                float acc0__v__c6_1 = in1__c0_1 * in0__c6_0;
                acc0__c6_1 += acc0__v__c6_1;
                float acc0__v__c6_2 = in1__c0_2 * in0__c6_0;
                acc0__c6_2 += acc0__v__c6_2;
                float acc0__v__c6_3 = in1__c0_3 * in0__c6_0;
                acc0__c6_3 += acc0__v__c6_3;
                float in0__c7_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 7 - a0_b * 96) * 4 + _ki];
                float acc0__v__c7_0 = in1__c0_0 * in0__c7_0;
                acc0__c7_0 += acc0__v__c7_0;
                float acc0__v__c7_1 = in1__c0_1 * in0__c7_0;
                acc0__c7_1 += acc0__v__c7_1;
                float acc0__v__c7_2 = in1__c0_2 * in0__c7_0;
                acc0__c7_2 += acc0__v__c7_2;
                float acc0__v__c7_3 = in1__c0_3 * in0__c7_0;
                acc0__c7_3 += acc0__v__c7_3;
                float in0__c8_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 8 - a0_b * 96) * 4 + _ki];
                float acc0__v__c8_0 = in1__c0_0 * in0__c8_0;
                acc0__c8_0 += acc0__v__c8_0;
                float acc0__v__c8_1 = in1__c0_1 * in0__c8_0;
                acc0__c8_1 += acc0__v__c8_1;
                float acc0__v__c8_2 = in1__c0_2 * in0__c8_0;
                acc0__c8_2 += acc0__v__c8_2;
                float acc0__v__c8_3 = in1__c0_3 * in0__c8_0;
                acc0__c8_3 += acc0__v__c8_3;
                float in0__c9_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 9 - a0_b * 96) * 4 + _ki];
                float acc0__v__c9_0 = in1__c0_0 * in0__c9_0;
                acc0__c9_0 += acc0__v__c9_0;
                float acc0__v__c9_1 = in1__c0_1 * in0__c9_0;
                acc0__c9_1 += acc0__v__c9_1;
                float acc0__v__c9_2 = in1__c0_2 * in0__c9_0;
                acc0__c9_2 += acc0__v__c9_2;
                float acc0__v__c9_3 = in1__c0_3 * in0__c9_0;
                acc0__c9_3 += acc0__v__c9_3;
                float in0__c10_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 10 - a0_b * 96) * 4 + _ki];
                float acc0__v__c10_0 = in1__c0_0 * in0__c10_0;
                acc0__c10_0 += acc0__v__c10_0;
                float acc0__v__c10_1 = in1__c0_1 * in0__c10_0;
                acc0__c10_1 += acc0__v__c10_1;
                float acc0__v__c10_2 = in1__c0_2 * in0__c10_0;
                acc0__c10_2 += acc0__v__c10_2;
                float acc0__v__c10_3 = in1__c0_3 * in0__c10_0;
                acc0__c10_3 += acc0__v__c10_3;
                float in0__c11_0 = _a_smem[(a0_b * 96 + a0_u * 12 + 11 - a0_b * 96) * 4 + _ki];
                float acc0__v__c11_0 = in1__c0_0 * in0__c11_0;
                acc0__c11_0 += acc0__v__c11_0;
                float acc0__v__c11_1 = in1__c0_1 * in0__c11_0;
                acc0__c11_1 += acc0__v__c11_1;
                float acc0__v__c11_2 = in1__c0_2 * in0__c11_0;
                acc0__c11_2 += acc0__v__c11_2;
                float acc0__v__c11_3 = in1__c0_3 * in0__c11_0;
                acc0__c11_3 += acc0__v__c11_3;
            }
            __syncthreads();
        }
        if (a0_b * 96 + a0_u * 12 + 0 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c0_0);
        }
        if (a0_b * 96 + a0_u * 12 + 0 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c0_1);
        }
        if (a0_b * 96 + a0_u * 12 + 0 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c0_2);
        }
        if (a0_b * 96 + a0_u * 12 + 0 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c0_3);
        }
        if (a0_b * 96 + a0_u * 12 + 1 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 1) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c1_0);
        }
        if (a0_b * 96 + a0_u * 12 + 1 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 1) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c1_1);
        }
        if (a0_b * 96 + a0_u * 12 + 1 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 1) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c1_2);
        }
        if (a0_b * 96 + a0_u * 12 + 1 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 1) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c1_3);
        }
        if (a0_b * 96 + a0_u * 12 + 2 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 2) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c2_0);
        }
        if (a0_b * 96 + a0_u * 12 + 2 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 2) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c2_1);
        }
        if (a0_b * 96 + a0_u * 12 + 2 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 2) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c2_2);
        }
        if (a0_b * 96 + a0_u * 12 + 2 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 2) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c2_3);
        }
        if (a0_b * 96 + a0_u * 12 + 3 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 3) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c3_0);
        }
        if (a0_b * 96 + a0_u * 12 + 3 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 3) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c3_1);
        }
        if (a0_b * 96 + a0_u * 12 + 3 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 3) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c3_2);
        }
        if (a0_b * 96 + a0_u * 12 + 3 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 3) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c3_3);
        }
        if (a0_b * 96 + a0_u * 12 + 4 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 4) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c4_0);
        }
        if (a0_b * 96 + a0_u * 12 + 4 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 4) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c4_1);
        }
        if (a0_b * 96 + a0_u * 12 + 4 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 4) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c4_2);
        }
        if (a0_b * 96 + a0_u * 12 + 4 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 4) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c4_3);
        }
        if (a0_b * 96 + a0_u * 12 + 5 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 5) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c5_0);
        }
        if (a0_b * 96 + a0_u * 12 + 5 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 5) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c5_1);
        }
        if (a0_b * 96 + a0_u * 12 + 5 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 5) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c5_2);
        }
        if (a0_b * 96 + a0_u * 12 + 5 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 5) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c5_3);
        }
        if (a0_b * 96 + a0_u * 12 + 6 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 6) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c6_0);
        }
        if (a0_b * 96 + a0_u * 12 + 6 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 6) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c6_1);
        }
        if (a0_b * 96 + a0_u * 12 + 6 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 6) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c6_2);
        }
        if (a0_b * 96 + a0_u * 12 + 6 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 6) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c6_3);
        }
        if (a0_b * 96 + a0_u * 12 + 7 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 7) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c7_0);
        }
        if (a0_b * 96 + a0_u * 12 + 7 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 7) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c7_1);
        }
        if (a0_b * 96 + a0_u * 12 + 7 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 7) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c7_2);
        }
        if (a0_b * 96 + a0_u * 12 + 7 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 7) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c7_3);
        }
        if (a0_b * 96 + a0_u * 12 + 8 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 8) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c8_0);
        }
        if (a0_b * 96 + a0_u * 12 + 8 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 8) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c8_1);
        }
        if (a0_b * 96 + a0_u * 12 + 8 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 8) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c8_2);
        }
        if (a0_b * 96 + a0_u * 12 + 8 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 8) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c8_3);
        }
        if (a0_b * 96 + a0_u * 12 + 9 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 9) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c9_0);
        }
        if (a0_b * 96 + a0_u * 12 + 9 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 9) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c9_1);
        }
        if (a0_b * 96 + a0_u * 12 + 9 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 9) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c9_2);
        }
        if (a0_b * 96 + a0_u * 12 + 9 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 9) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c9_3);
        }
        if (a0_b * 96 + a0_u * 12 + 10 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 10) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c10_0);
        }
        if (a0_b * 96 + a0_u * 12 + 10 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 10) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c10_1);
        }
        if (a0_b * 96 + a0_u * 12 + 10 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 10) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c10_2);
        }
        if (a0_b * 96 + a0_u * 12 + 10 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 10) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c10_3);
        }
        if (a0_b * 96 + a0_u * 12 + 11 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 11) % 512 * 512 + (a1_b * 64 + a1_u * 4)], acc0__c11_0);
        }
        if (a0_b * 96 + a0_u * 12 + 11 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 11) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 1)], acc0__c11_1);
        }
        if (a0_b * 96 + a0_u * 12 + 11 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 11) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 2)], acc0__c11_2);
        }
        if (a0_b * 96 + a0_u * 12 + 11 < 512) {
            atomicAdd(&matmul[(a0_b * 96 + a0_u * 12 + 11) % 512 * 512 + (a1_b * 64 + a1_u * 4 + 3)], acc0__c11_3);
        }
}
