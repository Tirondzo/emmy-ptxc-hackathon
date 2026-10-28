extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_257df3(const float* div_c1, const float* mul_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 512;
        int a5_b = (_gid / 256) % 2;
        int a0_u = (_gid / 16) % 16;
        int a5_u = (_gid) % 16;
        float in0 = div_c1[0];
        float v1 = 1.0f / in0;
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        #pragma unroll
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in3__ar0 = q[(a0_b * 16 + a0_u) * 64 + a2];
                float in2__ar0 = kt[a2 * 64 + a1];
                float v3__ar0 = in2__ar0 * in3__ar0;
                acc0__ar0 += v3__ar0;
            }
            float in8__bc0 = v[a1 * 64 + (a5_b * 32 + a5_u * 2)];
            float in1__bc0 = mul_c1[0];
            float in8__bc1 = v[a1 * 64 + (a5_b * 32 + a5_u * 2 + 1)];
            float in1__bc1 = mul_c1[0];
            float v4__c0_0 = acc0__ar0 * v1;
            float one__c0_0 = 1.0f;
            float v4__gn__c0_0 = fmaxf(acc1__c0_0, v4__c0_0);
            float v4__alpha__c0_0 = expf((acc1__c0_0 - v4__gn__c0_0));
            float v4__beta__c0_0 = expf((v4__c0_0 - v4__gn__c0_0));
            float v4__acc3_sa__c0_0 = acc3__c0_0 * v4__alpha__c0_0;
            float v4__acc3_sb__c0_0 = one__c0_0 * v4__beta__c0_0;
            acc3__c0_0 = v4__acc3_sa__c0_0 + v4__acc3_sb__c0_0;
            float v4__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v4__alpha__c0_0;
            float v4__acc5__sum_sb__c0_0 = (in1__bc0 * in8__bc0) * v4__beta__c0_0;
            acc5__sum__c0_0 = v4__acc5__sum_sa__c0_0 + v4__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v4__c0_0);
            float v4__c0_1 = acc0__ar0 * v1;
            float one__c0_1 = 1.0f;
            float v4__gn__c0_1 = fmaxf(acc1__c0_1, v4__c0_1);
            float v4__alpha__c0_1 = expf((acc1__c0_1 - v4__gn__c0_1));
            float v4__beta__c0_1 = expf((v4__c0_1 - v4__gn__c0_1));
            float v4__acc3_sa__c0_1 = acc3__c0_1 * v4__alpha__c0_1;
            float v4__acc3_sb__c0_1 = one__c0_1 * v4__beta__c0_1;
            acc3__c0_1 = v4__acc3_sa__c0_1 + v4__acc3_sb__c0_1;
            float v4__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v4__alpha__c0_1;
            float v4__acc5__sum_sb__c0_1 = (in1__bc1 * in8__bc1) * v4__beta__c0_1;
            acc5__sum__c0_1 = v4__acc5__sum_sa__c0_1 + v4__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v4__c0_1);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        matmul_1[(a0_b * 16 + a0_u) * 64 + (a5_b * 32 + a5_u * 2)] = acc5__c0_0;
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        matmul_1[(a0_b * 16 + a0_u) * 64 + (a5_b * 32 + a5_u * 2 + 1)] = acc5__c0_1;
}
