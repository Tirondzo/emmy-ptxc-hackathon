extern "C" __global__
__launch_bounds__(128) void k_matmul_softmax_reduce_ad7c0b(const float* div_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 256;
        int a5_b = (_gid / 128) % 2;
        int a0_u = (_gid / 16) % 8;
        int a5_u = (_gid) % 16;
        float in0 = div_c1[0];
        float v1 = 1.0f / in0;
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        float acc3__c0_2 = 0.0f;
        float acc5__sum__c0_2 = 0.0f;
        float acc1__c0_2 = -1e+30f;
        float acc3__c0_3 = 0.0f;
        float acc5__sum__c0_3 = 0.0f;
        float acc1__c0_3 = -1e+30f;
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        float acc1__c1_0 = -1e+30f;
        float acc3__c1_1 = 0.0f;
        float acc5__sum__c1_1 = 0.0f;
        float acc1__c1_1 = -1e+30f;
        float acc3__c1_2 = 0.0f;
        float acc5__sum__c1_2 = 0.0f;
        float acc1__c1_2 = -1e+30f;
        float acc3__c1_3 = 0.0f;
        float acc5__sum__c1_3 = 0.0f;
        float acc1__c1_3 = -1e+30f;
        float acc3__c2_0 = 0.0f;
        float acc5__sum__c2_0 = 0.0f;
        float acc1__c2_0 = -1e+30f;
        float acc3__c2_1 = 0.0f;
        float acc5__sum__c2_1 = 0.0f;
        float acc1__c2_1 = -1e+30f;
        float acc3__c2_2 = 0.0f;
        float acc5__sum__c2_2 = 0.0f;
        float acc1__c2_2 = -1e+30f;
        float acc3__c2_3 = 0.0f;
        float acc5__sum__c2_3 = 0.0f;
        float acc1__c2_3 = -1e+30f;
        float acc3__c3_0 = 0.0f;
        float acc5__sum__c3_0 = 0.0f;
        float acc1__c3_0 = -1e+30f;
        float acc3__c3_1 = 0.0f;
        float acc5__sum__c3_1 = 0.0f;
        float acc1__c3_1 = -1e+30f;
        float acc3__c3_2 = 0.0f;
        float acc5__sum__c3_2 = 0.0f;
        float acc1__c3_2 = -1e+30f;
        float acc3__c3_3 = 0.0f;
        float acc5__sum__c3_3 = 0.0f;
        float acc1__c3_3 = -1e+30f;
        float acc3__c4_0 = 0.0f;
        float acc5__sum__c4_0 = 0.0f;
        float acc1__c4_0 = -1e+30f;
        float acc3__c4_1 = 0.0f;
        float acc5__sum__c4_1 = 0.0f;
        float acc1__c4_1 = -1e+30f;
        float acc3__c4_2 = 0.0f;
        float acc5__sum__c4_2 = 0.0f;
        float acc1__c4_2 = -1e+30f;
        float acc3__c4_3 = 0.0f;
        float acc5__sum__c4_3 = 0.0f;
        float acc1__c4_3 = -1e+30f;
        float acc3__c5_0 = 0.0f;
        float acc5__sum__c5_0 = 0.0f;
        float acc1__c5_0 = -1e+30f;
        float acc3__c5_1 = 0.0f;
        float acc5__sum__c5_1 = 0.0f;
        float acc1__c5_1 = -1e+30f;
        float acc3__c5_2 = 0.0f;
        float acc5__sum__c5_2 = 0.0f;
        float acc1__c5_2 = -1e+30f;
        float acc3__c5_3 = 0.0f;
        float acc5__sum__c5_3 = 0.0f;
        float acc1__c5_3 = -1e+30f;
        float acc3__c6_0 = 0.0f;
        float acc5__sum__c6_0 = 0.0f;
        float acc1__c6_0 = -1e+30f;
        float acc3__c6_1 = 0.0f;
        float acc5__sum__c6_1 = 0.0f;
        float acc1__c6_1 = -1e+30f;
        float acc3__c6_2 = 0.0f;
        float acc5__sum__c6_2 = 0.0f;
        float acc1__c6_2 = -1e+30f;
        float acc3__c6_3 = 0.0f;
        float acc5__sum__c6_3 = 0.0f;
        float acc1__c6_3 = -1e+30f;
        float acc3__c7_0 = 0.0f;
        float acc5__sum__c7_0 = 0.0f;
        float acc1__c7_0 = -1e+30f;
        float acc3__c7_1 = 0.0f;
        float acc5__sum__c7_1 = 0.0f;
        float acc1__c7_1 = -1e+30f;
        float acc3__c7_2 = 0.0f;
        float acc5__sum__c7_2 = 0.0f;
        float acc1__c7_2 = -1e+30f;
        float acc3__c7_3 = 0.0f;
        float acc5__sum__c7_3 = 0.0f;
        float acc1__c7_3 = -1e+30f;
        float acc3__c8_0 = 0.0f;
        float acc5__sum__c8_0 = 0.0f;
        float acc1__c8_0 = -1e+30f;
        float acc3__c8_1 = 0.0f;
        float acc5__sum__c8_1 = 0.0f;
        float acc1__c8_1 = -1e+30f;
        float acc3__c8_2 = 0.0f;
        float acc5__sum__c8_2 = 0.0f;
        float acc1__c8_2 = -1e+30f;
        float acc3__c8_3 = 0.0f;
        float acc5__sum__c8_3 = 0.0f;
        float acc1__c8_3 = -1e+30f;
        float acc3__c9_0 = 0.0f;
        float acc5__sum__c9_0 = 0.0f;
        float acc1__c9_0 = -1e+30f;
        float acc3__c9_1 = 0.0f;
        float acc5__sum__c9_1 = 0.0f;
        float acc1__c9_1 = -1e+30f;
        float acc3__c9_2 = 0.0f;
        float acc5__sum__c9_2 = 0.0f;
        float acc1__c9_2 = -1e+30f;
        float acc3__c9_3 = 0.0f;
        float acc5__sum__c9_3 = 0.0f;
        float acc1__c9_3 = -1e+30f;
        for (int a1 = 0; a1 < 128; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar0 = q[(a0_b * 80 + a0_u * 10) % 128 * 128 + a2];
                float in1__ar0 = kt[a2 * 128 + a1];
                float v3__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v3__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar1 = q[(a0_b * 80 + a0_u * 10 + 1) % 128 * 128 + a2];
                float in1__ar1 = kt[a2 * 128 + a1];
                float v3__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v3__ar1;
            }
            float acc0__ar2 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar2 = q[(a0_b * 80 + a0_u * 10 + 2) % 128 * 128 + a2];
                float in1__ar2 = kt[a2 * 128 + a1];
                float v3__ar2 = in1__ar2 * in2__ar2;
                acc0__ar2 += v3__ar2;
            }
            float acc0__ar3 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar3 = q[(a0_b * 80 + a0_u * 10 + 3) % 128 * 128 + a2];
                float in1__ar3 = kt[a2 * 128 + a1];
                float v3__ar3 = in1__ar3 * in2__ar3;
                acc0__ar3 += v3__ar3;
            }
            float acc0__ar4 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar4 = q[(a0_b * 80 + a0_u * 10 + 4) % 128 * 128 + a2];
                float in1__ar4 = kt[a2 * 128 + a1];
                float v3__ar4 = in1__ar4 * in2__ar4;
                acc0__ar4 += v3__ar4;
            }
            float acc0__ar5 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar5 = q[(a0_b * 80 + a0_u * 10 + 5) % 128 * 128 + a2];
                float in1__ar5 = kt[a2 * 128 + a1];
                float v3__ar5 = in1__ar5 * in2__ar5;
                acc0__ar5 += v3__ar5;
            }
            float acc0__ar6 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar6 = q[(a0_b * 80 + a0_u * 10 + 6) % 128 * 128 + a2];
                float in1__ar6 = kt[a2 * 128 + a1];
                float v3__ar6 = in1__ar6 * in2__ar6;
                acc0__ar6 += v3__ar6;
            }
            float acc0__ar7 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar7 = q[(a0_b * 80 + a0_u * 10 + 7) % 128 * 128 + a2];
                float in1__ar7 = kt[a2 * 128 + a1];
                float v3__ar7 = in1__ar7 * in2__ar7;
                acc0__ar7 += v3__ar7;
            }
            float acc0__ar8 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar8 = q[(a0_b * 80 + a0_u * 10 + 8) % 128 * 128 + a2];
                float in1__ar8 = kt[a2 * 128 + a1];
                float v3__ar8 = in1__ar8 * in2__ar8;
                acc0__ar8 += v3__ar8;
            }
            float acc0__ar9 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__ar9 = q[(a0_b * 80 + a0_u * 10 + 9) % 128 * 128 + a2];
                float in1__ar9 = kt[a2 * 128 + a1];
                float v3__ar9 = in1__ar9 * in2__ar9;
                acc0__ar9 += v3__ar9;
            }
            float v4__c0_0 = acc0__ar0 * v1;
            float one__c0_0 = 1.0f;
            float v4__gn__c0_0 = fmaxf(acc1__c0_0, v4__c0_0);
            float v4__alpha__c0_0 = expf((acc1__c0_0 - v4__gn__c0_0));
            float v4__beta__c0_0 = expf((v4__c0_0 - v4__gn__c0_0));
            float v4__acc3_sa__c0_0 = acc3__c0_0 * v4__alpha__c0_0;
            float v4__acc3_sb__c0_0 = one__c0_0 * v4__beta__c0_0;
            acc3__c0_0 = v4__acc3_sa__c0_0 + v4__acc3_sb__c0_0;
            float v4__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v4__alpha__c0_0;
            float4 _v_in7__bc0 = *reinterpret_cast<const float4*>(&v[a1 * 128 + (a5_b * 64 + a5_u * 4)]);
            float in7__bc0 = _v_in7__bc0.x;
            float in7__bc1 = _v_in7__bc0.y;
            float in7__bc2 = _v_in7__bc0.z;
            float in7__bc3 = _v_in7__bc0.w;
            float v4__acc5__sum_sb__c0_0 = in7__bc0 * v4__beta__c0_0;
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
            float v4__acc5__sum_sb__c0_1 = in7__bc1 * v4__beta__c0_1;
            acc5__sum__c0_1 = v4__acc5__sum_sa__c0_1 + v4__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v4__c0_1);
            float v4__c0_2 = acc0__ar0 * v1;
            float one__c0_2 = 1.0f;
            float v4__gn__c0_2 = fmaxf(acc1__c0_2, v4__c0_2);
            float v4__alpha__c0_2 = expf((acc1__c0_2 - v4__gn__c0_2));
            float v4__beta__c0_2 = expf((v4__c0_2 - v4__gn__c0_2));
            float v4__acc3_sa__c0_2 = acc3__c0_2 * v4__alpha__c0_2;
            float v4__acc3_sb__c0_2 = one__c0_2 * v4__beta__c0_2;
            acc3__c0_2 = v4__acc3_sa__c0_2 + v4__acc3_sb__c0_2;
            float v4__acc5__sum_sa__c0_2 = acc5__sum__c0_2 * v4__alpha__c0_2;
            float v4__acc5__sum_sb__c0_2 = in7__bc2 * v4__beta__c0_2;
            acc5__sum__c0_2 = v4__acc5__sum_sa__c0_2 + v4__acc5__sum_sb__c0_2;
            acc1__c0_2 = fmaxf(acc1__c0_2, v4__c0_2);
            float v4__c0_3 = acc0__ar0 * v1;
            float one__c0_3 = 1.0f;
            float v4__gn__c0_3 = fmaxf(acc1__c0_3, v4__c0_3);
            float v4__alpha__c0_3 = expf((acc1__c0_3 - v4__gn__c0_3));
            float v4__beta__c0_3 = expf((v4__c0_3 - v4__gn__c0_3));
            float v4__acc3_sa__c0_3 = acc3__c0_3 * v4__alpha__c0_3;
            float v4__acc3_sb__c0_3 = one__c0_3 * v4__beta__c0_3;
            acc3__c0_3 = v4__acc3_sa__c0_3 + v4__acc3_sb__c0_3;
            float v4__acc5__sum_sa__c0_3 = acc5__sum__c0_3 * v4__alpha__c0_3;
            float v4__acc5__sum_sb__c0_3 = in7__bc3 * v4__beta__c0_3;
            acc5__sum__c0_3 = v4__acc5__sum_sa__c0_3 + v4__acc5__sum_sb__c0_3;
            acc1__c0_3 = fmaxf(acc1__c0_3, v4__c0_3);
            float v4__c1_0 = acc0__ar1 * v1;
            float one__c1_0 = 1.0f;
            float v4__gn__c1_0 = fmaxf(acc1__c1_0, v4__c1_0);
            float v4__alpha__c1_0 = expf((acc1__c1_0 - v4__gn__c1_0));
            float v4__beta__c1_0 = expf((v4__c1_0 - v4__gn__c1_0));
            float v4__acc3_sa__c1_0 = acc3__c1_0 * v4__alpha__c1_0;
            float v4__acc3_sb__c1_0 = one__c1_0 * v4__beta__c1_0;
            acc3__c1_0 = v4__acc3_sa__c1_0 + v4__acc3_sb__c1_0;
            float v4__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v4__alpha__c1_0;
            float v4__acc5__sum_sb__c1_0 = in7__bc0 * v4__beta__c1_0;
            acc5__sum__c1_0 = v4__acc5__sum_sa__c1_0 + v4__acc5__sum_sb__c1_0;
            acc1__c1_0 = fmaxf(acc1__c1_0, v4__c1_0);
            float v4__c1_1 = acc0__ar1 * v1;
            float one__c1_1 = 1.0f;
            float v4__gn__c1_1 = fmaxf(acc1__c1_1, v4__c1_1);
            float v4__alpha__c1_1 = expf((acc1__c1_1 - v4__gn__c1_1));
            float v4__beta__c1_1 = expf((v4__c1_1 - v4__gn__c1_1));
            float v4__acc3_sa__c1_1 = acc3__c1_1 * v4__alpha__c1_1;
            float v4__acc3_sb__c1_1 = one__c1_1 * v4__beta__c1_1;
            acc3__c1_1 = v4__acc3_sa__c1_1 + v4__acc3_sb__c1_1;
            float v4__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v4__alpha__c1_1;
            float v4__acc5__sum_sb__c1_1 = in7__bc1 * v4__beta__c1_1;
            acc5__sum__c1_1 = v4__acc5__sum_sa__c1_1 + v4__acc5__sum_sb__c1_1;
            acc1__c1_1 = fmaxf(acc1__c1_1, v4__c1_1);
            float v4__c1_2 = acc0__ar1 * v1;
            float one__c1_2 = 1.0f;
            float v4__gn__c1_2 = fmaxf(acc1__c1_2, v4__c1_2);
            float v4__alpha__c1_2 = expf((acc1__c1_2 - v4__gn__c1_2));
            float v4__beta__c1_2 = expf((v4__c1_2 - v4__gn__c1_2));
            float v4__acc3_sa__c1_2 = acc3__c1_2 * v4__alpha__c1_2;
            float v4__acc3_sb__c1_2 = one__c1_2 * v4__beta__c1_2;
            acc3__c1_2 = v4__acc3_sa__c1_2 + v4__acc3_sb__c1_2;
            float v4__acc5__sum_sa__c1_2 = acc5__sum__c1_2 * v4__alpha__c1_2;
            float v4__acc5__sum_sb__c1_2 = in7__bc2 * v4__beta__c1_2;
            acc5__sum__c1_2 = v4__acc5__sum_sa__c1_2 + v4__acc5__sum_sb__c1_2;
            acc1__c1_2 = fmaxf(acc1__c1_2, v4__c1_2);
            float v4__c1_3 = acc0__ar1 * v1;
            float one__c1_3 = 1.0f;
            float v4__gn__c1_3 = fmaxf(acc1__c1_3, v4__c1_3);
            float v4__alpha__c1_3 = expf((acc1__c1_3 - v4__gn__c1_3));
            float v4__beta__c1_3 = expf((v4__c1_3 - v4__gn__c1_3));
            float v4__acc3_sa__c1_3 = acc3__c1_3 * v4__alpha__c1_3;
            float v4__acc3_sb__c1_3 = one__c1_3 * v4__beta__c1_3;
            acc3__c1_3 = v4__acc3_sa__c1_3 + v4__acc3_sb__c1_3;
            float v4__acc5__sum_sa__c1_3 = acc5__sum__c1_3 * v4__alpha__c1_3;
            float v4__acc5__sum_sb__c1_3 = in7__bc3 * v4__beta__c1_3;
            acc5__sum__c1_3 = v4__acc5__sum_sa__c1_3 + v4__acc5__sum_sb__c1_3;
            acc1__c1_3 = fmaxf(acc1__c1_3, v4__c1_3);
            float v4__c2_0 = acc0__ar2 * v1;
            float one__c2_0 = 1.0f;
            float v4__gn__c2_0 = fmaxf(acc1__c2_0, v4__c2_0);
            float v4__alpha__c2_0 = expf((acc1__c2_0 - v4__gn__c2_0));
            float v4__beta__c2_0 = expf((v4__c2_0 - v4__gn__c2_0));
            float v4__acc3_sa__c2_0 = acc3__c2_0 * v4__alpha__c2_0;
            float v4__acc3_sb__c2_0 = one__c2_0 * v4__beta__c2_0;
            acc3__c2_0 = v4__acc3_sa__c2_0 + v4__acc3_sb__c2_0;
            float v4__acc5__sum_sa__c2_0 = acc5__sum__c2_0 * v4__alpha__c2_0;
            float v4__acc5__sum_sb__c2_0 = in7__bc0 * v4__beta__c2_0;
            acc5__sum__c2_0 = v4__acc5__sum_sa__c2_0 + v4__acc5__sum_sb__c2_0;
            acc1__c2_0 = fmaxf(acc1__c2_0, v4__c2_0);
            float v4__c2_1 = acc0__ar2 * v1;
            float one__c2_1 = 1.0f;
            float v4__gn__c2_1 = fmaxf(acc1__c2_1, v4__c2_1);
            float v4__alpha__c2_1 = expf((acc1__c2_1 - v4__gn__c2_1));
            float v4__beta__c2_1 = expf((v4__c2_1 - v4__gn__c2_1));
            float v4__acc3_sa__c2_1 = acc3__c2_1 * v4__alpha__c2_1;
            float v4__acc3_sb__c2_1 = one__c2_1 * v4__beta__c2_1;
            acc3__c2_1 = v4__acc3_sa__c2_1 + v4__acc3_sb__c2_1;
            float v4__acc5__sum_sa__c2_1 = acc5__sum__c2_1 * v4__alpha__c2_1;
            float v4__acc5__sum_sb__c2_1 = in7__bc1 * v4__beta__c2_1;
            acc5__sum__c2_1 = v4__acc5__sum_sa__c2_1 + v4__acc5__sum_sb__c2_1;
            acc1__c2_1 = fmaxf(acc1__c2_1, v4__c2_1);
            float v4__c2_2 = acc0__ar2 * v1;
            float one__c2_2 = 1.0f;
            float v4__gn__c2_2 = fmaxf(acc1__c2_2, v4__c2_2);
            float v4__alpha__c2_2 = expf((acc1__c2_2 - v4__gn__c2_2));
            float v4__beta__c2_2 = expf((v4__c2_2 - v4__gn__c2_2));
            float v4__acc3_sa__c2_2 = acc3__c2_2 * v4__alpha__c2_2;
            float v4__acc3_sb__c2_2 = one__c2_2 * v4__beta__c2_2;
            acc3__c2_2 = v4__acc3_sa__c2_2 + v4__acc3_sb__c2_2;
            float v4__acc5__sum_sa__c2_2 = acc5__sum__c2_2 * v4__alpha__c2_2;
            float v4__acc5__sum_sb__c2_2 = in7__bc2 * v4__beta__c2_2;
            acc5__sum__c2_2 = v4__acc5__sum_sa__c2_2 + v4__acc5__sum_sb__c2_2;
            acc1__c2_2 = fmaxf(acc1__c2_2, v4__c2_2);
            float v4__c2_3 = acc0__ar2 * v1;
            float one__c2_3 = 1.0f;
            float v4__gn__c2_3 = fmaxf(acc1__c2_3, v4__c2_3);
            float v4__alpha__c2_3 = expf((acc1__c2_3 - v4__gn__c2_3));
            float v4__beta__c2_3 = expf((v4__c2_3 - v4__gn__c2_3));
            float v4__acc3_sa__c2_3 = acc3__c2_3 * v4__alpha__c2_3;
            float v4__acc3_sb__c2_3 = one__c2_3 * v4__beta__c2_3;
            acc3__c2_3 = v4__acc3_sa__c2_3 + v4__acc3_sb__c2_3;
            float v4__acc5__sum_sa__c2_3 = acc5__sum__c2_3 * v4__alpha__c2_3;
            float v4__acc5__sum_sb__c2_3 = in7__bc3 * v4__beta__c2_3;
            acc5__sum__c2_3 = v4__acc5__sum_sa__c2_3 + v4__acc5__sum_sb__c2_3;
            acc1__c2_3 = fmaxf(acc1__c2_3, v4__c2_3);
            float v4__c3_0 = acc0__ar3 * v1;
            float one__c3_0 = 1.0f;
            float v4__gn__c3_0 = fmaxf(acc1__c3_0, v4__c3_0);
            float v4__alpha__c3_0 = expf((acc1__c3_0 - v4__gn__c3_0));
            float v4__beta__c3_0 = expf((v4__c3_0 - v4__gn__c3_0));
            float v4__acc3_sa__c3_0 = acc3__c3_0 * v4__alpha__c3_0;
            float v4__acc3_sb__c3_0 = one__c3_0 * v4__beta__c3_0;
            acc3__c3_0 = v4__acc3_sa__c3_0 + v4__acc3_sb__c3_0;
            float v4__acc5__sum_sa__c3_0 = acc5__sum__c3_0 * v4__alpha__c3_0;
            float v4__acc5__sum_sb__c3_0 = in7__bc0 * v4__beta__c3_0;
            acc5__sum__c3_0 = v4__acc5__sum_sa__c3_0 + v4__acc5__sum_sb__c3_0;
            acc1__c3_0 = fmaxf(acc1__c3_0, v4__c3_0);
            float v4__c3_1 = acc0__ar3 * v1;
            float one__c3_1 = 1.0f;
            float v4__gn__c3_1 = fmaxf(acc1__c3_1, v4__c3_1);
            float v4__alpha__c3_1 = expf((acc1__c3_1 - v4__gn__c3_1));
            float v4__beta__c3_1 = expf((v4__c3_1 - v4__gn__c3_1));
            float v4__acc3_sa__c3_1 = acc3__c3_1 * v4__alpha__c3_1;
            float v4__acc3_sb__c3_1 = one__c3_1 * v4__beta__c3_1;
            acc3__c3_1 = v4__acc3_sa__c3_1 + v4__acc3_sb__c3_1;
            float v4__acc5__sum_sa__c3_1 = acc5__sum__c3_1 * v4__alpha__c3_1;
            float v4__acc5__sum_sb__c3_1 = in7__bc1 * v4__beta__c3_1;
            acc5__sum__c3_1 = v4__acc5__sum_sa__c3_1 + v4__acc5__sum_sb__c3_1;
            acc1__c3_1 = fmaxf(acc1__c3_1, v4__c3_1);
            float v4__c3_2 = acc0__ar3 * v1;
            float one__c3_2 = 1.0f;
            float v4__gn__c3_2 = fmaxf(acc1__c3_2, v4__c3_2);
            float v4__alpha__c3_2 = expf((acc1__c3_2 - v4__gn__c3_2));
            float v4__beta__c3_2 = expf((v4__c3_2 - v4__gn__c3_2));
            float v4__acc3_sa__c3_2 = acc3__c3_2 * v4__alpha__c3_2;
            float v4__acc3_sb__c3_2 = one__c3_2 * v4__beta__c3_2;
            acc3__c3_2 = v4__acc3_sa__c3_2 + v4__acc3_sb__c3_2;
            float v4__acc5__sum_sa__c3_2 = acc5__sum__c3_2 * v4__alpha__c3_2;
            float v4__acc5__sum_sb__c3_2 = in7__bc2 * v4__beta__c3_2;
            acc5__sum__c3_2 = v4__acc5__sum_sa__c3_2 + v4__acc5__sum_sb__c3_2;
            acc1__c3_2 = fmaxf(acc1__c3_2, v4__c3_2);
            float v4__c3_3 = acc0__ar3 * v1;
            float one__c3_3 = 1.0f;
            float v4__gn__c3_3 = fmaxf(acc1__c3_3, v4__c3_3);
            float v4__alpha__c3_3 = expf((acc1__c3_3 - v4__gn__c3_3));
            float v4__beta__c3_3 = expf((v4__c3_3 - v4__gn__c3_3));
            float v4__acc3_sa__c3_3 = acc3__c3_3 * v4__alpha__c3_3;
            float v4__acc3_sb__c3_3 = one__c3_3 * v4__beta__c3_3;
            acc3__c3_3 = v4__acc3_sa__c3_3 + v4__acc3_sb__c3_3;
            float v4__acc5__sum_sa__c3_3 = acc5__sum__c3_3 * v4__alpha__c3_3;
            float v4__acc5__sum_sb__c3_3 = in7__bc3 * v4__beta__c3_3;
            acc5__sum__c3_3 = v4__acc5__sum_sa__c3_3 + v4__acc5__sum_sb__c3_3;
            acc1__c3_3 = fmaxf(acc1__c3_3, v4__c3_3);
            float v4__c4_0 = acc0__ar4 * v1;
            float one__c4_0 = 1.0f;
            float v4__gn__c4_0 = fmaxf(acc1__c4_0, v4__c4_0);
            float v4__alpha__c4_0 = expf((acc1__c4_0 - v4__gn__c4_0));
            float v4__beta__c4_0 = expf((v4__c4_0 - v4__gn__c4_0));
            float v4__acc3_sa__c4_0 = acc3__c4_0 * v4__alpha__c4_0;
            float v4__acc3_sb__c4_0 = one__c4_0 * v4__beta__c4_0;
            acc3__c4_0 = v4__acc3_sa__c4_0 + v4__acc3_sb__c4_0;
            float v4__acc5__sum_sa__c4_0 = acc5__sum__c4_0 * v4__alpha__c4_0;
            float v4__acc5__sum_sb__c4_0 = in7__bc0 * v4__beta__c4_0;
            acc5__sum__c4_0 = v4__acc5__sum_sa__c4_0 + v4__acc5__sum_sb__c4_0;
            acc1__c4_0 = fmaxf(acc1__c4_0, v4__c4_0);
            float v4__c4_1 = acc0__ar4 * v1;
            float one__c4_1 = 1.0f;
            float v4__gn__c4_1 = fmaxf(acc1__c4_1, v4__c4_1);
            float v4__alpha__c4_1 = expf((acc1__c4_1 - v4__gn__c4_1));
            float v4__beta__c4_1 = expf((v4__c4_1 - v4__gn__c4_1));
            float v4__acc3_sa__c4_1 = acc3__c4_1 * v4__alpha__c4_1;
            float v4__acc3_sb__c4_1 = one__c4_1 * v4__beta__c4_1;
            acc3__c4_1 = v4__acc3_sa__c4_1 + v4__acc3_sb__c4_1;
            float v4__acc5__sum_sa__c4_1 = acc5__sum__c4_1 * v4__alpha__c4_1;
            float v4__acc5__sum_sb__c4_1 = in7__bc1 * v4__beta__c4_1;
            acc5__sum__c4_1 = v4__acc5__sum_sa__c4_1 + v4__acc5__sum_sb__c4_1;
            acc1__c4_1 = fmaxf(acc1__c4_1, v4__c4_1);
            float v4__c4_2 = acc0__ar4 * v1;
            float one__c4_2 = 1.0f;
            float v4__gn__c4_2 = fmaxf(acc1__c4_2, v4__c4_2);
            float v4__alpha__c4_2 = expf((acc1__c4_2 - v4__gn__c4_2));
            float v4__beta__c4_2 = expf((v4__c4_2 - v4__gn__c4_2));
            float v4__acc3_sa__c4_2 = acc3__c4_2 * v4__alpha__c4_2;
            float v4__acc3_sb__c4_2 = one__c4_2 * v4__beta__c4_2;
            acc3__c4_2 = v4__acc3_sa__c4_2 + v4__acc3_sb__c4_2;
            float v4__acc5__sum_sa__c4_2 = acc5__sum__c4_2 * v4__alpha__c4_2;
            float v4__acc5__sum_sb__c4_2 = in7__bc2 * v4__beta__c4_2;
            acc5__sum__c4_2 = v4__acc5__sum_sa__c4_2 + v4__acc5__sum_sb__c4_2;
            acc1__c4_2 = fmaxf(acc1__c4_2, v4__c4_2);
            float v4__c4_3 = acc0__ar4 * v1;
            float one__c4_3 = 1.0f;
            float v4__gn__c4_3 = fmaxf(acc1__c4_3, v4__c4_3);
            float v4__alpha__c4_3 = expf((acc1__c4_3 - v4__gn__c4_3));
            float v4__beta__c4_3 = expf((v4__c4_3 - v4__gn__c4_3));
            float v4__acc3_sa__c4_3 = acc3__c4_3 * v4__alpha__c4_3;
            float v4__acc3_sb__c4_3 = one__c4_3 * v4__beta__c4_3;
            acc3__c4_3 = v4__acc3_sa__c4_3 + v4__acc3_sb__c4_3;
            float v4__acc5__sum_sa__c4_3 = acc5__sum__c4_3 * v4__alpha__c4_3;
            float v4__acc5__sum_sb__c4_3 = in7__bc3 * v4__beta__c4_3;
            acc5__sum__c4_3 = v4__acc5__sum_sa__c4_3 + v4__acc5__sum_sb__c4_3;
            acc1__c4_3 = fmaxf(acc1__c4_3, v4__c4_3);
            float v4__c5_0 = acc0__ar5 * v1;
            float one__c5_0 = 1.0f;
            float v4__gn__c5_0 = fmaxf(acc1__c5_0, v4__c5_0);
            float v4__alpha__c5_0 = expf((acc1__c5_0 - v4__gn__c5_0));
            float v4__beta__c5_0 = expf((v4__c5_0 - v4__gn__c5_0));
            float v4__acc3_sa__c5_0 = acc3__c5_0 * v4__alpha__c5_0;
            float v4__acc3_sb__c5_0 = one__c5_0 * v4__beta__c5_0;
            acc3__c5_0 = v4__acc3_sa__c5_0 + v4__acc3_sb__c5_0;
            float v4__acc5__sum_sa__c5_0 = acc5__sum__c5_0 * v4__alpha__c5_0;
            float v4__acc5__sum_sb__c5_0 = in7__bc0 * v4__beta__c5_0;
            acc5__sum__c5_0 = v4__acc5__sum_sa__c5_0 + v4__acc5__sum_sb__c5_0;
            acc1__c5_0 = fmaxf(acc1__c5_0, v4__c5_0);
            float v4__c5_1 = acc0__ar5 * v1;
            float one__c5_1 = 1.0f;
            float v4__gn__c5_1 = fmaxf(acc1__c5_1, v4__c5_1);
            float v4__alpha__c5_1 = expf((acc1__c5_1 - v4__gn__c5_1));
            float v4__beta__c5_1 = expf((v4__c5_1 - v4__gn__c5_1));
            float v4__acc3_sa__c5_1 = acc3__c5_1 * v4__alpha__c5_1;
            float v4__acc3_sb__c5_1 = one__c5_1 * v4__beta__c5_1;
            acc3__c5_1 = v4__acc3_sa__c5_1 + v4__acc3_sb__c5_1;
            float v4__acc5__sum_sa__c5_1 = acc5__sum__c5_1 * v4__alpha__c5_1;
            float v4__acc5__sum_sb__c5_1 = in7__bc1 * v4__beta__c5_1;
            acc5__sum__c5_1 = v4__acc5__sum_sa__c5_1 + v4__acc5__sum_sb__c5_1;
            acc1__c5_1 = fmaxf(acc1__c5_1, v4__c5_1);
            float v4__c5_2 = acc0__ar5 * v1;
            float one__c5_2 = 1.0f;
            float v4__gn__c5_2 = fmaxf(acc1__c5_2, v4__c5_2);
            float v4__alpha__c5_2 = expf((acc1__c5_2 - v4__gn__c5_2));
            float v4__beta__c5_2 = expf((v4__c5_2 - v4__gn__c5_2));
            float v4__acc3_sa__c5_2 = acc3__c5_2 * v4__alpha__c5_2;
            float v4__acc3_sb__c5_2 = one__c5_2 * v4__beta__c5_2;
            acc3__c5_2 = v4__acc3_sa__c5_2 + v4__acc3_sb__c5_2;
            float v4__acc5__sum_sa__c5_2 = acc5__sum__c5_2 * v4__alpha__c5_2;
            float v4__acc5__sum_sb__c5_2 = in7__bc2 * v4__beta__c5_2;
            acc5__sum__c5_2 = v4__acc5__sum_sa__c5_2 + v4__acc5__sum_sb__c5_2;
            acc1__c5_2 = fmaxf(acc1__c5_2, v4__c5_2);
            float v4__c5_3 = acc0__ar5 * v1;
            float one__c5_3 = 1.0f;
            float v4__gn__c5_3 = fmaxf(acc1__c5_3, v4__c5_3);
            float v4__alpha__c5_3 = expf((acc1__c5_3 - v4__gn__c5_3));
            float v4__beta__c5_3 = expf((v4__c5_3 - v4__gn__c5_3));
            float v4__acc3_sa__c5_3 = acc3__c5_3 * v4__alpha__c5_3;
            float v4__acc3_sb__c5_3 = one__c5_3 * v4__beta__c5_3;
            acc3__c5_3 = v4__acc3_sa__c5_3 + v4__acc3_sb__c5_3;
            float v4__acc5__sum_sa__c5_3 = acc5__sum__c5_3 * v4__alpha__c5_3;
            float v4__acc5__sum_sb__c5_3 = in7__bc3 * v4__beta__c5_3;
            acc5__sum__c5_3 = v4__acc5__sum_sa__c5_3 + v4__acc5__sum_sb__c5_3;
            acc1__c5_3 = fmaxf(acc1__c5_3, v4__c5_3);
            float v4__c6_0 = acc0__ar6 * v1;
            float one__c6_0 = 1.0f;
            float v4__gn__c6_0 = fmaxf(acc1__c6_0, v4__c6_0);
            float v4__alpha__c6_0 = expf((acc1__c6_0 - v4__gn__c6_0));
            float v4__beta__c6_0 = expf((v4__c6_0 - v4__gn__c6_0));
            float v4__acc3_sa__c6_0 = acc3__c6_0 * v4__alpha__c6_0;
            float v4__acc3_sb__c6_0 = one__c6_0 * v4__beta__c6_0;
            acc3__c6_0 = v4__acc3_sa__c6_0 + v4__acc3_sb__c6_0;
            float v4__acc5__sum_sa__c6_0 = acc5__sum__c6_0 * v4__alpha__c6_0;
            float v4__acc5__sum_sb__c6_0 = in7__bc0 * v4__beta__c6_0;
            acc5__sum__c6_0 = v4__acc5__sum_sa__c6_0 + v4__acc5__sum_sb__c6_0;
            acc1__c6_0 = fmaxf(acc1__c6_0, v4__c6_0);
            float v4__c6_1 = acc0__ar6 * v1;
            float one__c6_1 = 1.0f;
            float v4__gn__c6_1 = fmaxf(acc1__c6_1, v4__c6_1);
            float v4__alpha__c6_1 = expf((acc1__c6_1 - v4__gn__c6_1));
            float v4__beta__c6_1 = expf((v4__c6_1 - v4__gn__c6_1));
            float v4__acc3_sa__c6_1 = acc3__c6_1 * v4__alpha__c6_1;
            float v4__acc3_sb__c6_1 = one__c6_1 * v4__beta__c6_1;
            acc3__c6_1 = v4__acc3_sa__c6_1 + v4__acc3_sb__c6_1;
            float v4__acc5__sum_sa__c6_1 = acc5__sum__c6_1 * v4__alpha__c6_1;
            float v4__acc5__sum_sb__c6_1 = in7__bc1 * v4__beta__c6_1;
            acc5__sum__c6_1 = v4__acc5__sum_sa__c6_1 + v4__acc5__sum_sb__c6_1;
            acc1__c6_1 = fmaxf(acc1__c6_1, v4__c6_1);
            float v4__c6_2 = acc0__ar6 * v1;
            float one__c6_2 = 1.0f;
            float v4__gn__c6_2 = fmaxf(acc1__c6_2, v4__c6_2);
            float v4__alpha__c6_2 = expf((acc1__c6_2 - v4__gn__c6_2));
            float v4__beta__c6_2 = expf((v4__c6_2 - v4__gn__c6_2));
            float v4__acc3_sa__c6_2 = acc3__c6_2 * v4__alpha__c6_2;
            float v4__acc3_sb__c6_2 = one__c6_2 * v4__beta__c6_2;
            acc3__c6_2 = v4__acc3_sa__c6_2 + v4__acc3_sb__c6_2;
            float v4__acc5__sum_sa__c6_2 = acc5__sum__c6_2 * v4__alpha__c6_2;
            float v4__acc5__sum_sb__c6_2 = in7__bc2 * v4__beta__c6_2;
            acc5__sum__c6_2 = v4__acc5__sum_sa__c6_2 + v4__acc5__sum_sb__c6_2;
            acc1__c6_2 = fmaxf(acc1__c6_2, v4__c6_2);
            float v4__c6_3 = acc0__ar6 * v1;
            float one__c6_3 = 1.0f;
            float v4__gn__c6_3 = fmaxf(acc1__c6_3, v4__c6_3);
            float v4__alpha__c6_3 = expf((acc1__c6_3 - v4__gn__c6_3));
            float v4__beta__c6_3 = expf((v4__c6_3 - v4__gn__c6_3));
            float v4__acc3_sa__c6_3 = acc3__c6_3 * v4__alpha__c6_3;
            float v4__acc3_sb__c6_3 = one__c6_3 * v4__beta__c6_3;
            acc3__c6_3 = v4__acc3_sa__c6_3 + v4__acc3_sb__c6_3;
            float v4__acc5__sum_sa__c6_3 = acc5__sum__c6_3 * v4__alpha__c6_3;
            float v4__acc5__sum_sb__c6_3 = in7__bc3 * v4__beta__c6_3;
            acc5__sum__c6_3 = v4__acc5__sum_sa__c6_3 + v4__acc5__sum_sb__c6_3;
            acc1__c6_3 = fmaxf(acc1__c6_3, v4__c6_3);
            float v4__c7_0 = acc0__ar7 * v1;
            float one__c7_0 = 1.0f;
            float v4__gn__c7_0 = fmaxf(acc1__c7_0, v4__c7_0);
            float v4__alpha__c7_0 = expf((acc1__c7_0 - v4__gn__c7_0));
            float v4__beta__c7_0 = expf((v4__c7_0 - v4__gn__c7_0));
            float v4__acc3_sa__c7_0 = acc3__c7_0 * v4__alpha__c7_0;
            float v4__acc3_sb__c7_0 = one__c7_0 * v4__beta__c7_0;
            acc3__c7_0 = v4__acc3_sa__c7_0 + v4__acc3_sb__c7_0;
            float v4__acc5__sum_sa__c7_0 = acc5__sum__c7_0 * v4__alpha__c7_0;
            float v4__acc5__sum_sb__c7_0 = in7__bc0 * v4__beta__c7_0;
            acc5__sum__c7_0 = v4__acc5__sum_sa__c7_0 + v4__acc5__sum_sb__c7_0;
            acc1__c7_0 = fmaxf(acc1__c7_0, v4__c7_0);
            float v4__c7_1 = acc0__ar7 * v1;
            float one__c7_1 = 1.0f;
            float v4__gn__c7_1 = fmaxf(acc1__c7_1, v4__c7_1);
            float v4__alpha__c7_1 = expf((acc1__c7_1 - v4__gn__c7_1));
            float v4__beta__c7_1 = expf((v4__c7_1 - v4__gn__c7_1));
            float v4__acc3_sa__c7_1 = acc3__c7_1 * v4__alpha__c7_1;
            float v4__acc3_sb__c7_1 = one__c7_1 * v4__beta__c7_1;
            acc3__c7_1 = v4__acc3_sa__c7_1 + v4__acc3_sb__c7_1;
            float v4__acc5__sum_sa__c7_1 = acc5__sum__c7_1 * v4__alpha__c7_1;
            float v4__acc5__sum_sb__c7_1 = in7__bc1 * v4__beta__c7_1;
            acc5__sum__c7_1 = v4__acc5__sum_sa__c7_1 + v4__acc5__sum_sb__c7_1;
            acc1__c7_1 = fmaxf(acc1__c7_1, v4__c7_1);
            float v4__c7_2 = acc0__ar7 * v1;
            float one__c7_2 = 1.0f;
            float v4__gn__c7_2 = fmaxf(acc1__c7_2, v4__c7_2);
            float v4__alpha__c7_2 = expf((acc1__c7_2 - v4__gn__c7_2));
            float v4__beta__c7_2 = expf((v4__c7_2 - v4__gn__c7_2));
            float v4__acc3_sa__c7_2 = acc3__c7_2 * v4__alpha__c7_2;
            float v4__acc3_sb__c7_2 = one__c7_2 * v4__beta__c7_2;
            acc3__c7_2 = v4__acc3_sa__c7_2 + v4__acc3_sb__c7_2;
            float v4__acc5__sum_sa__c7_2 = acc5__sum__c7_2 * v4__alpha__c7_2;
            float v4__acc5__sum_sb__c7_2 = in7__bc2 * v4__beta__c7_2;
            acc5__sum__c7_2 = v4__acc5__sum_sa__c7_2 + v4__acc5__sum_sb__c7_2;
            acc1__c7_2 = fmaxf(acc1__c7_2, v4__c7_2);
            float v4__c7_3 = acc0__ar7 * v1;
            float one__c7_3 = 1.0f;
            float v4__gn__c7_3 = fmaxf(acc1__c7_3, v4__c7_3);
            float v4__alpha__c7_3 = expf((acc1__c7_3 - v4__gn__c7_3));
            float v4__beta__c7_3 = expf((v4__c7_3 - v4__gn__c7_3));
            float v4__acc3_sa__c7_3 = acc3__c7_3 * v4__alpha__c7_3;
            float v4__acc3_sb__c7_3 = one__c7_3 * v4__beta__c7_3;
            acc3__c7_3 = v4__acc3_sa__c7_3 + v4__acc3_sb__c7_3;
            float v4__acc5__sum_sa__c7_3 = acc5__sum__c7_3 * v4__alpha__c7_3;
            float v4__acc5__sum_sb__c7_3 = in7__bc3 * v4__beta__c7_3;
            acc5__sum__c7_3 = v4__acc5__sum_sa__c7_3 + v4__acc5__sum_sb__c7_3;
            acc1__c7_3 = fmaxf(acc1__c7_3, v4__c7_3);
            float v4__c8_0 = acc0__ar8 * v1;
            float one__c8_0 = 1.0f;
            float v4__gn__c8_0 = fmaxf(acc1__c8_0, v4__c8_0);
            float v4__alpha__c8_0 = expf((acc1__c8_0 - v4__gn__c8_0));
            float v4__beta__c8_0 = expf((v4__c8_0 - v4__gn__c8_0));
            float v4__acc3_sa__c8_0 = acc3__c8_0 * v4__alpha__c8_0;
            float v4__acc3_sb__c8_0 = one__c8_0 * v4__beta__c8_0;
            acc3__c8_0 = v4__acc3_sa__c8_0 + v4__acc3_sb__c8_0;
            float v4__acc5__sum_sa__c8_0 = acc5__sum__c8_0 * v4__alpha__c8_0;
            float v4__acc5__sum_sb__c8_0 = in7__bc0 * v4__beta__c8_0;
            acc5__sum__c8_0 = v4__acc5__sum_sa__c8_0 + v4__acc5__sum_sb__c8_0;
            acc1__c8_0 = fmaxf(acc1__c8_0, v4__c8_0);
            float v4__c8_1 = acc0__ar8 * v1;
            float one__c8_1 = 1.0f;
            float v4__gn__c8_1 = fmaxf(acc1__c8_1, v4__c8_1);
            float v4__alpha__c8_1 = expf((acc1__c8_1 - v4__gn__c8_1));
            float v4__beta__c8_1 = expf((v4__c8_1 - v4__gn__c8_1));
            float v4__acc3_sa__c8_1 = acc3__c8_1 * v4__alpha__c8_1;
            float v4__acc3_sb__c8_1 = one__c8_1 * v4__beta__c8_1;
            acc3__c8_1 = v4__acc3_sa__c8_1 + v4__acc3_sb__c8_1;
            float v4__acc5__sum_sa__c8_1 = acc5__sum__c8_1 * v4__alpha__c8_1;
            float v4__acc5__sum_sb__c8_1 = in7__bc1 * v4__beta__c8_1;
            acc5__sum__c8_1 = v4__acc5__sum_sa__c8_1 + v4__acc5__sum_sb__c8_1;
            acc1__c8_1 = fmaxf(acc1__c8_1, v4__c8_1);
            float v4__c8_2 = acc0__ar8 * v1;
            float one__c8_2 = 1.0f;
            float v4__gn__c8_2 = fmaxf(acc1__c8_2, v4__c8_2);
            float v4__alpha__c8_2 = expf((acc1__c8_2 - v4__gn__c8_2));
            float v4__beta__c8_2 = expf((v4__c8_2 - v4__gn__c8_2));
            float v4__acc3_sa__c8_2 = acc3__c8_2 * v4__alpha__c8_2;
            float v4__acc3_sb__c8_2 = one__c8_2 * v4__beta__c8_2;
            acc3__c8_2 = v4__acc3_sa__c8_2 + v4__acc3_sb__c8_2;
            float v4__acc5__sum_sa__c8_2 = acc5__sum__c8_2 * v4__alpha__c8_2;
            float v4__acc5__sum_sb__c8_2 = in7__bc2 * v4__beta__c8_2;
            acc5__sum__c8_2 = v4__acc5__sum_sa__c8_2 + v4__acc5__sum_sb__c8_2;
            acc1__c8_2 = fmaxf(acc1__c8_2, v4__c8_2);
            float v4__c8_3 = acc0__ar8 * v1;
            float one__c8_3 = 1.0f;
            float v4__gn__c8_3 = fmaxf(acc1__c8_3, v4__c8_3);
            float v4__alpha__c8_3 = expf((acc1__c8_3 - v4__gn__c8_3));
            float v4__beta__c8_3 = expf((v4__c8_3 - v4__gn__c8_3));
            float v4__acc3_sa__c8_3 = acc3__c8_3 * v4__alpha__c8_3;
            float v4__acc3_sb__c8_3 = one__c8_3 * v4__beta__c8_3;
            acc3__c8_3 = v4__acc3_sa__c8_3 + v4__acc3_sb__c8_3;
            float v4__acc5__sum_sa__c8_3 = acc5__sum__c8_3 * v4__alpha__c8_3;
            float v4__acc5__sum_sb__c8_3 = in7__bc3 * v4__beta__c8_3;
            acc5__sum__c8_3 = v4__acc5__sum_sa__c8_3 + v4__acc5__sum_sb__c8_3;
            acc1__c8_3 = fmaxf(acc1__c8_3, v4__c8_3);
            float v4__c9_0 = acc0__ar9 * v1;
            float one__c9_0 = 1.0f;
            float v4__gn__c9_0 = fmaxf(acc1__c9_0, v4__c9_0);
            float v4__alpha__c9_0 = expf((acc1__c9_0 - v4__gn__c9_0));
            float v4__beta__c9_0 = expf((v4__c9_0 - v4__gn__c9_0));
            float v4__acc3_sa__c9_0 = acc3__c9_0 * v4__alpha__c9_0;
            float v4__acc3_sb__c9_0 = one__c9_0 * v4__beta__c9_0;
            acc3__c9_0 = v4__acc3_sa__c9_0 + v4__acc3_sb__c9_0;
            float v4__acc5__sum_sa__c9_0 = acc5__sum__c9_0 * v4__alpha__c9_0;
            float v4__acc5__sum_sb__c9_0 = in7__bc0 * v4__beta__c9_0;
            acc5__sum__c9_0 = v4__acc5__sum_sa__c9_0 + v4__acc5__sum_sb__c9_0;
            acc1__c9_0 = fmaxf(acc1__c9_0, v4__c9_0);
            float v4__c9_1 = acc0__ar9 * v1;
            float one__c9_1 = 1.0f;
            float v4__gn__c9_1 = fmaxf(acc1__c9_1, v4__c9_1);
            float v4__alpha__c9_1 = expf((acc1__c9_1 - v4__gn__c9_1));
            float v4__beta__c9_1 = expf((v4__c9_1 - v4__gn__c9_1));
            float v4__acc3_sa__c9_1 = acc3__c9_1 * v4__alpha__c9_1;
            float v4__acc3_sb__c9_1 = one__c9_1 * v4__beta__c9_1;
            acc3__c9_1 = v4__acc3_sa__c9_1 + v4__acc3_sb__c9_1;
            float v4__acc5__sum_sa__c9_1 = acc5__sum__c9_1 * v4__alpha__c9_1;
            float v4__acc5__sum_sb__c9_1 = in7__bc1 * v4__beta__c9_1;
            acc5__sum__c9_1 = v4__acc5__sum_sa__c9_1 + v4__acc5__sum_sb__c9_1;
            acc1__c9_1 = fmaxf(acc1__c9_1, v4__c9_1);
            float v4__c9_2 = acc0__ar9 * v1;
            float one__c9_2 = 1.0f;
            float v4__gn__c9_2 = fmaxf(acc1__c9_2, v4__c9_2);
            float v4__alpha__c9_2 = expf((acc1__c9_2 - v4__gn__c9_2));
            float v4__beta__c9_2 = expf((v4__c9_2 - v4__gn__c9_2));
            float v4__acc3_sa__c9_2 = acc3__c9_2 * v4__alpha__c9_2;
            float v4__acc3_sb__c9_2 = one__c9_2 * v4__beta__c9_2;
            acc3__c9_2 = v4__acc3_sa__c9_2 + v4__acc3_sb__c9_2;
            float v4__acc5__sum_sa__c9_2 = acc5__sum__c9_2 * v4__alpha__c9_2;
            float v4__acc5__sum_sb__c9_2 = in7__bc2 * v4__beta__c9_2;
            acc5__sum__c9_2 = v4__acc5__sum_sa__c9_2 + v4__acc5__sum_sb__c9_2;
            acc1__c9_2 = fmaxf(acc1__c9_2, v4__c9_2);
            float v4__c9_3 = acc0__ar9 * v1;
            float one__c9_3 = 1.0f;
            float v4__gn__c9_3 = fmaxf(acc1__c9_3, v4__c9_3);
            float v4__alpha__c9_3 = expf((acc1__c9_3 - v4__gn__c9_3));
            float v4__beta__c9_3 = expf((v4__c9_3 - v4__gn__c9_3));
            float v4__acc3_sa__c9_3 = acc3__c9_3 * v4__alpha__c9_3;
            float v4__acc3_sb__c9_3 = one__c9_3 * v4__beta__c9_3;
            acc3__c9_3 = v4__acc3_sa__c9_3 + v4__acc3_sb__c9_3;
            float v4__acc5__sum_sa__c9_3 = acc5__sum__c9_3 * v4__alpha__c9_3;
            float v4__acc5__sum_sb__c9_3 = in7__bc3 * v4__beta__c9_3;
            acc5__sum__c9_3 = v4__acc5__sum_sa__c9_3 + v4__acc5__sum_sb__c9_3;
            acc1__c9_3 = fmaxf(acc1__c9_3, v4__c9_3);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        if (a0_b * 80 + a0_u * 10 + 0 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c0_0;
        }
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        if (a0_b * 80 + a0_u * 10 + 0 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c0_1;
        }
        float acc5__c0_2 = acc5__sum__c0_2 * (1.0f / acc3__c0_2);
        if (a0_b * 80 + a0_u * 10 + 0 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c0_2;
        }
        float acc5__c0_3 = acc5__sum__c0_3 * (1.0f / acc3__c0_3);
        if (a0_b * 80 + a0_u * 10 + 0 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c0_3;
        }
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        if (a0_b * 80 + a0_u * 10 + 1 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 1) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c1_0;
        }
        float acc5__c1_1 = acc5__sum__c1_1 * (1.0f / acc3__c1_1);
        if (a0_b * 80 + a0_u * 10 + 1 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 1) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c1_1;
        }
        float acc5__c1_2 = acc5__sum__c1_2 * (1.0f / acc3__c1_2);
        if (a0_b * 80 + a0_u * 10 + 1 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 1) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c1_2;
        }
        float acc5__c1_3 = acc5__sum__c1_3 * (1.0f / acc3__c1_3);
        if (a0_b * 80 + a0_u * 10 + 1 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 1) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c1_3;
        }
        float acc5__c2_0 = acc5__sum__c2_0 * (1.0f / acc3__c2_0);
        if (a0_b * 80 + a0_u * 10 + 2 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 2) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c2_0;
        }
        float acc5__c2_1 = acc5__sum__c2_1 * (1.0f / acc3__c2_1);
        if (a0_b * 80 + a0_u * 10 + 2 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 2) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c2_1;
        }
        float acc5__c2_2 = acc5__sum__c2_2 * (1.0f / acc3__c2_2);
        if (a0_b * 80 + a0_u * 10 + 2 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 2) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c2_2;
        }
        float acc5__c2_3 = acc5__sum__c2_3 * (1.0f / acc3__c2_3);
        if (a0_b * 80 + a0_u * 10 + 2 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 2) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c2_3;
        }
        float acc5__c3_0 = acc5__sum__c3_0 * (1.0f / acc3__c3_0);
        if (a0_b * 80 + a0_u * 10 + 3 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 3) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c3_0;
        }
        float acc5__c3_1 = acc5__sum__c3_1 * (1.0f / acc3__c3_1);
        if (a0_b * 80 + a0_u * 10 + 3 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 3) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c3_1;
        }
        float acc5__c3_2 = acc5__sum__c3_2 * (1.0f / acc3__c3_2);
        if (a0_b * 80 + a0_u * 10 + 3 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 3) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c3_2;
        }
        float acc5__c3_3 = acc5__sum__c3_3 * (1.0f / acc3__c3_3);
        if (a0_b * 80 + a0_u * 10 + 3 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 3) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c3_3;
        }
        float acc5__c4_0 = acc5__sum__c4_0 * (1.0f / acc3__c4_0);
        if (a0_b * 80 + a0_u * 10 + 4 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 4) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c4_0;
        }
        float acc5__c4_1 = acc5__sum__c4_1 * (1.0f / acc3__c4_1);
        if (a0_b * 80 + a0_u * 10 + 4 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 4) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c4_1;
        }
        float acc5__c4_2 = acc5__sum__c4_2 * (1.0f / acc3__c4_2);
        if (a0_b * 80 + a0_u * 10 + 4 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 4) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c4_2;
        }
        float acc5__c4_3 = acc5__sum__c4_3 * (1.0f / acc3__c4_3);
        if (a0_b * 80 + a0_u * 10 + 4 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 4) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c4_3;
        }
        float acc5__c5_0 = acc5__sum__c5_0 * (1.0f / acc3__c5_0);
        if (a0_b * 80 + a0_u * 10 + 5 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 5) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c5_0;
        }
        float acc5__c5_1 = acc5__sum__c5_1 * (1.0f / acc3__c5_1);
        if (a0_b * 80 + a0_u * 10 + 5 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 5) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c5_1;
        }
        float acc5__c5_2 = acc5__sum__c5_2 * (1.0f / acc3__c5_2);
        if (a0_b * 80 + a0_u * 10 + 5 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 5) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c5_2;
        }
        float acc5__c5_3 = acc5__sum__c5_3 * (1.0f / acc3__c5_3);
        if (a0_b * 80 + a0_u * 10 + 5 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 5) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c5_3;
        }
        float acc5__c6_0 = acc5__sum__c6_0 * (1.0f / acc3__c6_0);
        if (a0_b * 80 + a0_u * 10 + 6 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 6) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c6_0;
        }
        float acc5__c6_1 = acc5__sum__c6_1 * (1.0f / acc3__c6_1);
        if (a0_b * 80 + a0_u * 10 + 6 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 6) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c6_1;
        }
        float acc5__c6_2 = acc5__sum__c6_2 * (1.0f / acc3__c6_2);
        if (a0_b * 80 + a0_u * 10 + 6 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 6) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c6_2;
        }
        float acc5__c6_3 = acc5__sum__c6_3 * (1.0f / acc3__c6_3);
        if (a0_b * 80 + a0_u * 10 + 6 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 6) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c6_3;
        }
        float acc5__c7_0 = acc5__sum__c7_0 * (1.0f / acc3__c7_0);
        if (a0_b * 80 + a0_u * 10 + 7 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 7) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c7_0;
        }
        float acc5__c7_1 = acc5__sum__c7_1 * (1.0f / acc3__c7_1);
        if (a0_b * 80 + a0_u * 10 + 7 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 7) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c7_1;
        }
        float acc5__c7_2 = acc5__sum__c7_2 * (1.0f / acc3__c7_2);
        if (a0_b * 80 + a0_u * 10 + 7 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 7) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c7_2;
        }
        float acc5__c7_3 = acc5__sum__c7_3 * (1.0f / acc3__c7_3);
        if (a0_b * 80 + a0_u * 10 + 7 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 7) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c7_3;
        }
        float acc5__c8_0 = acc5__sum__c8_0 * (1.0f / acc3__c8_0);
        if (a0_b * 80 + a0_u * 10 + 8 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 8) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c8_0;
        }
        float acc5__c8_1 = acc5__sum__c8_1 * (1.0f / acc3__c8_1);
        if (a0_b * 80 + a0_u * 10 + 8 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 8) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c8_1;
        }
        float acc5__c8_2 = acc5__sum__c8_2 * (1.0f / acc3__c8_2);
        if (a0_b * 80 + a0_u * 10 + 8 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 8) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c8_2;
        }
        float acc5__c8_3 = acc5__sum__c8_3 * (1.0f / acc3__c8_3);
        if (a0_b * 80 + a0_u * 10 + 8 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 8) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c8_3;
        }
        float acc5__c9_0 = acc5__sum__c9_0 * (1.0f / acc3__c9_0);
        if (a0_b * 80 + a0_u * 10 + 9 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 9) % 128 * 128 + (a5_b * 64 + a5_u * 4)] = acc5__c9_0;
        }
        float acc5__c9_1 = acc5__sum__c9_1 * (1.0f / acc3__c9_1);
        if (a0_b * 80 + a0_u * 10 + 9 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 9) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 1)] = acc5__c9_1;
        }
        float acc5__c9_2 = acc5__sum__c9_2 * (1.0f / acc3__c9_2);
        if (a0_b * 80 + a0_u * 10 + 9 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 9) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 2)] = acc5__c9_2;
        }
        float acc5__c9_3 = acc5__sum__c9_3 * (1.0f / acc3__c9_3);
        if (a0_b * 80 + a0_u * 10 + 9 < 128) {
            matmul_1[(a0_b * 80 + a0_u * 10 + 9) % 128 * 128 + (a5_b * 64 + a5_u * 4 + 3)] = acc5__c9_3;
        }
}
