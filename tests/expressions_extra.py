"""Extra problems for ptxc: variations inside the five hackathon categories.

Same spec format as the hackathon's tools/expressions_open.py (an `OPEN` list of dicts with
id / code / ref / rtol / atol / knobs), so its tools/generate.py packages them unchanged:

    PYTHONUTF8=1 python ../emmy-hackathon-2026/tools/generate.py tests/expressions_extra.py \
        --emmy ../.venv/Scripts/emmy.exe --out tests/problems

Ids are numbered from 10 so they never collide with the open set. `ref` takes the seeded
float32 inputs in traced-graph order and returns the expected outputs; exact ops are checked
exactly (rtol = atol = 0), everything that goes through an .approx instruction or a different
summation order gets a tolerance in the range the open set uses.
"""

import numpy as np
from scipy.special import erf

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"  # one block of 128 cooperating threads, as the open reductions are pinned
F, D = np.float32, np.float64

EXACT = dict(rtol=0, atol=0)
APPROX = dict(rtol=1e-5, atol=1e-6)  # ex2 / lg2 / rsqrt .approx: a few ulp
REDUCE = dict(rtol=1e-4, atol=1e-4)
FUSED = dict(rtol=1e-3, atol=1e-4)
MATMUL = dict(rtol=1e-3, atol=1e-3)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = x - m
    return e - np.log(np.exp(e).sum(axis=axis, keepdims=True))


def mm(a, b):
    return (a.astype(D) @ b.astype(D)).astype(F)


def attn(q, kt, v, scale):
    return (_softmax((q.astype(D) @ kt.astype(D)) / scale) @ v.astype(D)).astype(F)


def spec(id, code, ref, knobs=None, **tol):
    d = dict(id=id, code=code, ref=ref, **tol)
    if knobs:
        d["knobs"] = knobs
    return d


OPEN = [
    # ---- category 1: pointwise ---------------------------------------------------------
    spec("p10_sub_odd", "torch.randn(1000) - torch.randn(1000)", lambda a, b: [a - b], **EXACT),
    spec("p11_mul_2d", "torch.randn(32, 48) * torch.randn(32, 48)", lambda a, b: [a * b], **EXACT),
    spec("p12_div_abs", "x = torch.randn(512); y = torch.randn(512); x / (torch.abs(y) + 1.0)",
         lambda x, y: [x / (np.abs(y) + F(1))], **EXACT),
    spec("p13_exp", "torch.exp(torch.randn(2048))", lambda x: [np.exp(x.astype(D)).astype(F)], **APPROX),
    spec("p14_log", "torch.log(torch.abs(torch.randn(1024)) + 0.5)",
         lambda x: [np.log(np.abs(x.astype(D)) + 0.5).astype(F)], **APPROX),
    spec("p15_sqrt_rsqrt", "x = torch.randn(1024); torch.sqrt(torch.abs(x)) + torch.rsqrt(torch.abs(x) + 1.0)",
         lambda x: [(np.sqrt(np.abs(x.astype(D))) + 1.0 / np.sqrt(np.abs(x.astype(D)) + 1.0)).astype(F)], **APPROX),
    spec("p16_tanh", "torch.tanh(torch.randn(1024))", lambda x: [np.tanh(x.astype(D)).astype(F)], **APPROX),
    spec("p17_sigmoid", "torch.sigmoid(torch.randn(1024))", lambda x: [(1.0 / (1.0 + np.exp(-x.astype(D)))).astype(F)], **APPROX),
    spec("p18_min_max_neg", "x = torch.randn(2048); y = torch.randn(2048); torch.maximum(-x, y) + torch.minimum(x, y)",
         lambda x, y: [np.maximum(-x, y) + np.minimum(x, y)], **EXACT),
    spec("p19_relu_affine", "x = torch.randn(4096); y = torch.randn(4096); torch.relu(x * 3.0 - 1.0) + torch.relu(y)",
         lambda x, y: [np.maximum(x * F(3) - F(1), F(0)) + np.maximum(y, F(0))], **EXACT),
    spec("p20_three_inputs", "x = torch.randn(1024); y = torch.randn(1024); z = torch.randn(1024); x * y + z",
         lambda x, y, z: [x * y + z], **EXACT),
    spec("p21_many_consts", "(torch.randn(3072) + 1.0) * 0.5 - 2.0", lambda x: [(x + F(1)) * F(0.5) - F(2)], **EXACT),
    spec("p22_square_same_input", "x = torch.randn(1024); x * x", lambda x: [x * x], **EXACT),
    spec("p23_tiny", "torch.randn(7) + torch.randn(7)", lambda a, b: [a + b], **EXACT),
    spec("p24_large", "torch.randn(1048576) * 2.0", lambda x: [x * F(2)], **EXACT),
    spec("p25_3d", "torch.randn(4, 32, 64) + torch.randn(4, 32, 64)", lambda a, b: [a + b], **EXACT),
    spec("p26_reciprocal", "1.0 / (torch.abs(torch.randn(1024)) + 0.5)", lambda x: [F(1) / (np.abs(x) + F(0.5))], **EXACT),
    spec("p27_abs_neg", "x = torch.randn(1024); -torch.abs(x) * x", lambda x: [-np.abs(x) * x], **EXACT),
    spec("p28_broadcast_row", "torch.randn(64, 128) + torch.randn(128)", lambda a, b: [a + b], **EXACT),
    spec("p29_broadcast_col", "torch.randn(64, 128) * torch.randn(64, 1)", lambda a, b: [a * b], **EXACT),
    spec("p30_prime_size", "torch.randn(4099) * torch.randn(4099)", lambda a, b: [a * b], **EXACT),
    spec("p31_long_chain", "x = torch.randn(2048); torch.exp(-torch.abs(x)) * (x + 2.0) / (x * x + 1.0)",
         lambda x: [(np.exp(-np.abs(x.astype(D))) * (x.astype(D) + 2.0) / (x.astype(D) ** 2 + 1.0)).astype(F)], **APPROX),
    # ops the open set never used: do they trace, and does ptxc lower them?
    spec("p40_silu", "F.silu(torch.randn(1024))", lambda x: [(x.astype(D) / (1.0 + np.exp(-x.astype(D)))).astype(F)], **APPROX),
    spec("p41_gelu", "F.gelu(torch.randn(1024))",
         lambda x: [(0.5 * x.astype(D) * (1.0 + erf(x.astype(D) / np.sqrt(2.0)))).astype(F)], **APPROX),
    spec("p42_gelu_tanh", "F.gelu(torch.randn(1024), approximate='tanh')",
         lambda x: [(0.5 * x.astype(D) * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x.astype(D) + 0.044715 * x.astype(D) ** 3)))).astype(F)],
         **APPROX),
    spec("p43_sin_cos", "x = torch.randn(1024); torch.sin(x) + torch.cos(x)",
         lambda x: [(np.sin(x.astype(D)) + np.cos(x.astype(D))).astype(F)], **APPROX),
    spec("p44_pow2", "torch.randn(1024) ** 2", lambda x: [x * x], **APPROX),
    spec("p45_clamp", "torch.clamp(torch.randn(1024), -0.5, 0.5)", lambda x: [np.clip(x, F(-0.5), F(0.5))], **EXACT),
    spec("p46_where", "x = torch.randn(1024); y = torch.randn(1024); torch.where(x > 0, x, y)",
         lambda x, y: [np.where(x > 0, x, y)], **EXACT),
    spec("p47_softplus", "F.softplus(torch.randn(1024))", lambda x: [np.logaddexp(0.0, x.astype(D)).astype(F)], **APPROX),
    spec("p48_floor_sign", "x = torch.randn(1024) * 3.0; torch.floor(x) + torch.sign(x)",
         lambda x: [np.floor(x * F(3)) + np.sign(x * F(3))], **EXACT),
    spec("p49_exp2_log2", "x = torch.randn(1024); torch.exp2(x) + torch.log2(torch.abs(x) + 1.0)",
         lambda x: [(np.exp2(x.astype(D)) + np.log2(np.abs(x.astype(D)) + 1.0)).astype(F)], **APPROX),
    spec("p50_erf", "torch.erf(torch.randn(1024))", lambda x: [erf(x.astype(D)).astype(F)], **APPROX),
    spec("p51_pow_half", "torch.abs(torch.randn(1024)) ** 0.5", lambda x: [np.sqrt(np.abs(x.astype(D))).astype(F)], **APPROX),
    spec("p52_square_fn", "torch.square(torch.randn(1024))", lambda x: [x * x], **EXACT),
    spec("p53_log1p_expm1", "x = torch.randn(1024); torch.log1p(torch.abs(x)) + torch.expm1(x)",
         lambda x: [(np.log1p(np.abs(x.astype(D))) + np.expm1(x.astype(D))).astype(F)], **APPROX),

    # ---- category 2: reductions ---------------------------------------------------------
    spec("r10_sum_1000", "torch.randn(1000).sum()", lambda x: [x.astype(D).sum().astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r11_sum_2d_all", "torch.randn(64, 64).sum()", lambda x: [x.astype(D).sum().astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r12_max", "torch.randn(4096).max()", lambda x: [x.max()], REDUCE_KNOBS, **EXACT),
    spec("r13_min", "torch.randn(4096).min()", lambda x: [x.min()], REDUCE_KNOBS, **EXACT),
    spec("r14_mean_rows_1000", "torch.randn(32, 1000).mean(-1)", lambda x: [x.astype(D).mean(-1).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r15_col_sum", "torch.randn(64, 512).sum(0)", lambda x: [x.astype(D).sum(0).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r16_sum_sq", "x = torch.randn(8192); (x * x).sum()", lambda x: [(x.astype(D) ** 2).sum().astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r17_abs_sum", "torch.abs(torch.randn(4096)).sum()", lambda x: [np.abs(x.astype(D)).sum().astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r18_row_max", "torch.randn(64, 256).amax(-1)", lambda x: [x.max(-1)], REDUCE_KNOBS, **EXACT),
    spec("r19_3d_last", "torch.randn(8, 16, 64).sum(-1)", lambda x: [x.astype(D).sum(-1).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r20_sum_tiny", "torch.randn(32).sum()", lambda x: [x.astype(D).sum().astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r21_prod", "(torch.randn(32) * 0.1 + 1.0).prod()",
         lambda x: [np.prod((x * F(0.1) + F(1)).astype(D)).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r22_sum_1M", "torch.randn(1048576).sum()", lambda x: [x.astype(D).sum().astype(F)], REDUCE_KNOBS, rtol=1e-3, atol=1e-2),
    spec("r23_unpinned_sum", "torch.randn(65536).sum()", lambda x: [x.astype(D).sum().astype(F)], rtol=1e-3, atol=1e-3),
    spec("r24_unpinned_rows", "torch.randn(64, 2048).sum(-1)", lambda x: [x.astype(D).sum(-1).astype(F)], rtol=1e-3, atol=1e-3),
    spec("r25_mean_dim0", "torch.randn(128, 32).mean(0)", lambda x: [x.astype(D).mean(0).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r26_keepdim", "torch.randn(16, 128).sum(-1, keepdim=True)",
         lambda x: [x.astype(D).sum(-1, keepdims=True).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r27_mid_axis", "torch.randn(8, 64, 16).sum(1)", lambda x: [x.astype(D).sum(1).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r28_scaled_sum", "(torch.randn(4096) * 2.0).sum()", lambda x: [(x.astype(D) * 2.0).sum().astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r29_mean_epilogue", "torch.randn(2048).mean() * 4.0 + 1.0",
         lambda x: [(x.astype(D).mean() * 4.0 + 1.0).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r30_row_min", "torch.randn(32, 1000).amin(-1)", lambda x: [x.min(-1)], REDUCE_KNOBS, **EXACT),
    spec("r31_sum_two_inputs", "x = torch.randn(4096); y = torch.randn(4096); (x * y).sum()",
         lambda x, y: [(x.astype(D) * y.astype(D)).sum().astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r32_rows_wide_odd", "torch.randn(3, 10007).sum(-1)", lambda x: [x.astype(D).sum(-1).astype(F)], REDUCE_KNOBS, **REDUCE),

    # ---- category 3: matmul -------------------------------------------------------------
    spec("m10_64", "torch.randn(64, 64) @ torch.randn(64, 64)", lambda a, b: [mm(a, b)], **MATMUL),
    spec("m11_256x128x512", "torch.randn(256, 128) @ torch.randn(128, 512)", lambda a, b: [mm(a, b)], **MATMUL),
    spec("m12_tall", "torch.randn(1024, 64) @ torch.randn(64, 32)", lambda a, b: [mm(a, b)], **MATMUL),
    spec("m13_odd_100", "torch.randn(100, 100) @ torch.randn(100, 100)", lambda a, b: [mm(a, b)], **MATMUL),
    spec("m14_1024", "torch.randn(1024, 1024) @ torch.randn(1024, 1024)", lambda a, b: [mm(a, b)], rtol=1e-3, atol=1e-2),
    spec("m15_scaled", "(torch.randn(128, 128) @ torch.randn(128, 128)) * 0.5", lambda a, b: [mm(a, b) * F(0.5)], **MATMUL),
    spec("m16_bias", "a = torch.randn(128, 256); b = torch.randn(256, 64); c = torch.randn(64); a @ b + c",
         lambda a, b, c: [(a.astype(D) @ b.astype(D) + c.astype(D)).astype(F)], **MATMUL),
    spec("m17_k16", "torch.randn(128, 16) @ torch.randn(16, 128)", lambda a, b: [mm(a, b)], **MATMUL),
    spec("m18_relu", "torch.relu(torch.randn(128, 128) @ torch.randn(128, 128))", lambda a, b: [np.maximum(mm(a, b), F(0))], **MATMUL),
    spec("m19_bmm", "torch.bmm(torch.randn(4, 64, 64), torch.randn(4, 64, 64))",
         lambda a, b: [np.einsum("bik,bkj->bij", a.astype(D), b.astype(D)).astype(F)], **MATMUL),
    spec("m20_chain_sum", "(torch.randn(128, 128) @ torch.randn(128, 128)).sum()",
         lambda a, b: [(a.astype(D) @ b.astype(D)).sum().astype(F)], rtol=1e-3, atol=1e-1),
    spec("m21_512x64x512", "torch.randn(512, 64) @ torch.randn(64, 512)", lambda a, b: [mm(a, b)], **MATMUL),
    spec("m22_wide_k", "torch.randn(64, 2048) @ torch.randn(2048, 64)", lambda a, b: [mm(a, b)], rtol=1e-3, atol=1e-2),
    spec("m23_mm_mm", "(torch.randn(64, 64) @ torch.randn(64, 64)) @ torch.randn(64, 64)",
         lambda a, b, c: [(a.astype(D) @ b.astype(D) @ c.astype(D)).astype(F)], rtol=1e-3, atol=1e-2),
    spec("m24_32", "torch.randn(32, 32) @ torch.randn(32, 32)", lambda a, b: [mm(a, b)], **MATMUL),
    spec("m25_rect_small_n", "torch.randn(256, 512) @ torch.randn(512, 16)", lambda a, b: [mm(a, b)], **MATMUL),

    # ---- category 4: fused --------------------------------------------------------------
    spec("f10_softmax_1000", "torch.softmax(torch.randn(32, 1000), dim=-1)", lambda x: [_softmax(x.astype(D)).astype(F)], rtol=1e-3, atol=1e-5),
    spec("f11_softmax_small", "torch.softmax(torch.randn(16, 32), dim=-1)", lambda x: [_softmax(x.astype(D)).astype(F)], rtol=1e-3, atol=1e-5),
    spec("f12_log_softmax", "torch.log_softmax(torch.randn(64, 256), dim=-1)", lambda x: [_log_softmax(x.astype(D)).astype(F)], **FUSED),
    spec("f13_layer_norm", "F.layer_norm(torch.randn(64, 1024), (1024,))",
         lambda x: [((x.astype(D) - x.astype(D).mean(-1, keepdims=True)) / np.sqrt(x.astype(D).var(-1, keepdims=True) + 1e-5)).astype(F)],
         **FUSED),
    spec("f14_rmsnorm_300", "x = torch.randn(32, 300); w = torch.randn(300); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         lambda x, w: [(x / np.sqrt((x.astype(D) ** 2).mean(-1, keepdims=True) + 1e-5) * w).astype(F)], **FUSED),
    spec("f15_l1norm", "x = torch.randn(64, 512); x / torch.abs(x).sum(-1, keepdim=True)",
         lambda x: [(x / np.abs(x.astype(D)).sum(-1, keepdims=True)).astype(F)], **FUSED),
    spec("f16_softmax_scaled", "torch.softmax(torch.randn(64, 256) * 0.1, dim=-1)",
         lambda x: [_softmax(x.astype(D) * 0.1).astype(F)], rtol=1e-3, atol=1e-5),
    spec("f17_sub_max", "x = torch.randn(64, 512); x - x.amax(-1, keepdim=True)", lambda x: [x - x.max(-1, keepdims=True)], **EXACT),
    spec("f18_center", "x = torch.randn(64, 512); x - x.mean(-1, keepdim=True)",
         lambda x: [(x.astype(D) - x.astype(D).mean(-1, keepdims=True)).astype(F)], **FUSED),
    spec("f19_softmax_dim0", "torch.softmax(torch.randn(256, 64), dim=0)", lambda x: [_softmax(x.astype(D), axis=0).astype(F)], rtol=1e-3, atol=1e-5),
    spec("f20_row_dot", "x = torch.randn(64, 512); w = torch.randn(512); (x * w).sum(-1)",
         lambda x, w: [(x.astype(D) * w.astype(D)).sum(-1).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("f21_cosine", "x = torch.randn(64, 256); y = torch.randn(64, 256); "
                       "(x * y).sum(-1) / (torch.sqrt((x * x).sum(-1)) * torch.sqrt((y * y).sum(-1)))",
         lambda x, y: [((x.astype(D) * y.astype(D)).sum(-1) / (np.sqrt((x.astype(D) ** 2).sum(-1)) * np.sqrt((y.astype(D) ** 2).sum(-1)))).astype(F)],
         **FUSED),
    spec("f22_rmsnorm_4096", "x = torch.randn(32, 4096); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)",
         lambda x: [(x / np.sqrt((x.astype(D) ** 2).mean(-1, keepdims=True) + 1e-5)).astype(F)], **FUSED),
    spec("f23_softmax_1row", "torch.softmax(torch.randn(1, 4096), dim=-1)", lambda x: [_softmax(x.astype(D)).astype(F)], rtol=1e-3, atol=1e-6),
    spec("f24_var", "torch.randn(64, 512).var(-1)", lambda x: [x.astype(D).var(-1, ddof=1).astype(F)], **FUSED),
    # no intermediate variables in `code`: emmy's inline tracer takes every named tensor of the last
    # statement as a graph input (`s = sigmoid(x); s / s.sum()` would make `s` the input)
    spec("f25_sigmoid_norm", "x = torch.randn(64, 256); torch.sigmoid(x) / torch.sigmoid(x).sum(-1, keepdim=True)",
         lambda x: [(lambda s: s / s.sum(-1, keepdims=True))(1.0 / (1.0 + np.exp(-x.astype(D)))).astype(F)], **FUSED),
    spec("f26_softmax_2048", "torch.softmax(torch.randn(16, 2048), dim=-1)", lambda x: [_softmax(x.astype(D)).astype(F)], rtol=1e-3, atol=1e-6),
    spec("f27_softmax_3d", "torch.softmax(torch.randn(4, 32, 128), dim=-1)", lambda x: [_softmax(x.astype(D)).astype(F)], rtol=1e-3, atol=1e-5),
    spec("f28_rms_only", "x = torch.randn(64, 512); torch.sqrt((x * x).mean(-1))",
         lambda x: [np.sqrt((x.astype(D) ** 2).mean(-1)).astype(F)], **FUSED),
    spec("f29_softmax_scale_const", "torch.softmax(torch.randn(64, 128) / 8.0, dim=-1)",
         lambda x: [_softmax(x.astype(D) / 8.0).astype(F)], rtol=1e-3, atol=1e-5),
    spec("f30_rmsnorm_odd_rows", "x = torch.randn(7, 1024); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)",
         lambda x: [(x / np.sqrt((x.astype(D) ** 2).mean(-1, keepdims=True) + 1e-5)).astype(F)], **FUSED),

    # ---- category 5: attention-like -------------------------------------------------------
    spec("a10_tiny_noscale", "q = torch.randn(32, 32); kt = torch.randn(32, 32); v = torch.randn(32, 32); torch.softmax(q @ kt, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 1.0)], **FUSED),
    spec("a11_rect", "q = torch.randn(64, 32); kt = torch.randn(32, 128); v = torch.randn(128, 64); torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 5.656854)], **FUSED),
    spec("a12_256", "q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 8.0)], **FUSED),
    spec("a13_mul_scale", "q = torch.randn(64, 64); kt = torch.randn(64, 64); v = torch.randn(64, 64); torch.softmax(q @ kt * 0.125, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 8.0)], **FUSED),
    spec("a14_128sq", "q = torch.randn(128, 128); kt = torch.randn(128, 128); v = torch.randn(128, 128); torch.softmax(q @ kt / 11.313708, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 11.313708)], **FUSED),
    spec("a15_odd", "q = torch.randn(48, 16); kt = torch.randn(16, 48); v = torch.randn(48, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 4.0)], **FUSED),
    spec("a16_relu_mm", "q = torch.randn(64, 64); kt = torch.randn(64, 64); v = torch.randn(64, 64); torch.relu(q @ kt) @ v",
         lambda q, kt, v: [(np.maximum(q.astype(D) @ kt.astype(D), 0.0) @ v.astype(D)).astype(F)], rtol=1e-3, atol=1e-2),
    spec("a17_scaled_v", "q = torch.randn(64, 64); kt = torch.randn(64, 64); v = torch.randn(64, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ (v * 2.0)",
         lambda q, kt, v: [attn(q, kt, v * F(2), 8.0)], **FUSED),
    spec("a18_small_rows", "q = torch.randn(16, 64); kt = torch.randn(64, 16); v = torch.randn(16, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 8.0)], **FUSED),
    spec("a19_long_keys", "q = torch.randn(32, 64); kt = torch.randn(64, 512); v = torch.randn(512, 32); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 8.0)], **FUSED),

    # ---- batch 2: narrow rows, odd 2-D widths, higher rank, pow, masks, unpinned max, batched attention ----
    spec("p60_2d_odd_width", "torch.randn(32, 1001) + torch.randn(32, 1001)", lambda a, b: [a + b], **EXACT),
    spec("p61_2d_narrow", "torch.randn(16, 10) * 2.0", lambda x: [x * F(2)], **EXACT),
    spec("p62_4d", "torch.randn(8, 8, 8, 8) * torch.randn(8, 8, 8, 8)", lambda a, b: [a * b], **EXACT),
    spec("p63_2d_large", "torch.randn(1024, 1024) + 1.0", lambda x: [x + F(1)], **EXACT),
    spec("p64_pow3", "torch.randn(1024) ** 3", lambda x: [(x.astype(D) ** 3).astype(F)], **APPROX),
    spec("p65_pow_tensor", "x = torch.randn(1024); y = torch.randn(1024); torch.abs(x) ** y",
         lambda x, y: [(np.abs(x.astype(D)) ** y.astype(D)).astype(F)], rtol=1e-4, atol=1e-5),
    spec("p66_exp_ratio", "x = torch.randn(1024); torch.exp(x) / (1.0 + torch.exp(x))",
         lambda x: [(np.exp(x.astype(D)) / (1.0 + np.exp(x.astype(D)))).astype(F)], **APPROX),
    spec("p67_broadcast_both", "torch.randn(64, 1) + torch.randn(1, 128)", lambda a, b: [a + b], **EXACT),
    spec("p68_compare_mask", "x = torch.randn(1024); y = torch.randn(1024); (x > y) * x", lambda x, y: [(x > y) * x], **EXACT),
    spec("r40_rows_16", "torch.randn(128, 16).sum(-1)", lambda x: [x.astype(D).sum(-1).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r41_rows_32", "torch.randn(256, 32).sum(-1)", lambda x: [x.astype(D).sum(-1).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r42_rows_64", "torch.randn(64, 64).sum(-1)", lambda x: [x.astype(D).sum(-1).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r43_unpinned_row_max", "torch.randn(64, 2048).amax(-1)", lambda x: [x.max(-1)], **EXACT),
    spec("r44_unpinned_sum_odd", "torch.randn(100003).sum()", lambda x: [x.astype(D).sum().astype(F)], rtol=1e-3, atol=1e-3),
    spec("r45_rows_8", "torch.randn(512, 8).sum(-1)", lambda x: [x.astype(D).sum(-1).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r46_sum_5", "torch.randn(5).sum()", lambda x: [x.astype(D).sum().astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("r47_mean_3d_mid", "torch.randn(4, 100, 8).mean(1)", lambda x: [x.astype(D).mean(1).astype(F)], REDUCE_KNOBS, **REDUCE),
    spec("f40_softmax_16", "torch.softmax(torch.randn(128, 16), dim=-1)", lambda x: [_softmax(x.astype(D)).astype(F)], rtol=1e-3, atol=1e-5),
    spec("f41_softmax_64", "torch.softmax(torch.randn(64, 64), dim=-1)", lambda x: [_softmax(x.astype(D)).astype(F)], rtol=1e-3, atol=1e-5),
    spec("f42_manual_softmax", "x = torch.randn(64, 256); torch.exp(x) / torch.exp(x).sum(-1, keepdim=True)",
         lambda x: [(np.exp(x.astype(D)) / np.exp(x.astype(D)).sum(-1, keepdims=True)).astype(F)], **FUSED),
    spec("f43_scale_by_sum", "x = torch.randn(64, 256); x * x.sum(-1, keepdim=True)",
         lambda x: [(x.astype(D) * x.astype(D).sum(-1, keepdims=True)).astype(F)], **FUSED),
    spec("f44_center_sq", "x = torch.randn(64, 256); (x - x.mean(-1, keepdim=True)) ** 2",
         lambda x: [((x.astype(D) - x.astype(D).mean(-1, keepdims=True)) ** 2).astype(F)], **FUSED),
    spec("f45_layernorm_odd", "F.layer_norm(torch.randn(16, 300), (300,))",
         lambda x: [((x.astype(D) - x.astype(D).mean(-1, keepdims=True)) / np.sqrt(x.astype(D).var(-1, keepdims=True) + 1e-5)).astype(F)],
         **FUSED),
    spec("f46_rmsnorm_rows_32", "x = torch.randn(256, 32); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)",
         lambda x: [(x / np.sqrt((x.astype(D) ** 2).mean(-1, keepdims=True) + 1e-5)).astype(F)], **FUSED),
    spec("f47_softmax_dim0_odd", "torch.softmax(torch.randn(1000, 32), dim=0)", lambda x: [_softmax(x.astype(D), axis=0).astype(F)], rtol=1e-3, atol=1e-5),
    spec("a20_narrow", "q = torch.randn(64, 16); kt = torch.randn(16, 64); v = torch.randn(64, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 4.0)], **FUSED),
    spec("a21_batched", "q = torch.randn(2, 64, 32); kt = torch.randn(2, 32, 64); v = torch.randn(2, 64, 32); "
                        "torch.bmm(torch.softmax(torch.bmm(q, kt) / 5.656854, dim=-1), v)",
         lambda q, kt, v: [np.stack([attn(q[i], kt[i], v[i], 5.656854) for i in range(q.shape[0])])], **FUSED),
    spec("a22_keys_1000", "q = torch.randn(16, 32); kt = torch.randn(32, 1000); v = torch.randn(1000, 16); torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         lambda q, kt, v: [attn(q, kt, v, 5.656854)], **FUSED),
]

# emmy 0.3.11 cannot trace these (2026-10-28: its frontend or CUDA renderer rejects floor, exp2,
# expm1, log_softmax, var, prod, whole-tensor max/min/sum of a 2-D tensor, amin, a reduction
# followed by a scalar epilogue, and matmul -> sum). They cannot be in the hidden set either, so
# they are kept for the record and left out of OPEN.
UNTRACEABLE = {
    "p48_floor_sign", "p49_exp2_log2", "p53_log1p_expm1", "r11_sum_2d_all", "r12_max", "r13_min",
    "r21_prod", "r29_mean_epilogue", "r30_row_min", "m20_chain_sum", "f12_log_softmax", "f24_var",
}
OPEN = [s for s in OPEN if s["id"] not in UNTRACEABLE]


# ---- round 2 (2026-10-28): the ops and index forms added to ptxc after the first findings -----------
# masks in every form emmy renders them, the polynomial transcendentals over wide ranges (tail and
# core branches, range reduction), pow with negative bases and negative / tensor exponents.
OPEN += [
    spec("p70_mask_float", "(torch.randn(1024) > 0).float()", lambda x: [(x > 0).astype(F)], **EXACT),
    spec("p71_mask_and", "x = torch.randn(1024); torch.where((x > 0) & (x < 1), x, -x)",
         lambda x: [np.where((x > 0) & (x < 1), x, -x)], **EXACT),
    spec("p72_mask_not", "x = torch.randn(1024); torch.where(~(x > 0), x, -x)",
         lambda x: [np.where(~(x > 0), x, -x)], **EXACT),
    spec("p73_mask_or_mul", "x = torch.randn(1024); ((x > 1) | (x < -1)) * x",
         lambda x: [((x > 1) | (x < -1)) * x], **EXACT),
    spec("p74_bool_out", "torch.randn(1024) > 0", lambda x: [x > 0], **EXACT),
    spec("p75_where_consts", "x = torch.randn(1024); torch.where(x > 0.5, 1.0, 0.0)",
         lambda x: [np.where(x > 0.5, F(1), F(0))], **EXACT),
    spec("p76_sigmoid_mask", "x = torch.randn(1024); torch.sigmoid(x) * (x > -1)",
         lambda x: [(1.0 / (1.0 + np.exp(-x.astype(D)))).astype(F) * (x > -1)], **APPROX),
    spec("p77_le_ge", "x = torch.randn(1024); y = torch.randn(1024); torch.where(x <= y, x, y) + torch.where(x >= y, x, y)",
         lambda x, y: [np.minimum(x, y) + np.maximum(x, y)], **EXACT),
    spec("p78_eq_ne", "x = torch.randn(1024); torch.where(x != torch.abs(x), -x, x) + torch.where(x == torch.abs(x), x, -x)",
         lambda x: [np.abs(x) * F(2)], **EXACT),
    spec("p79_erf_wide", "torch.erf(torch.randn(8192) * 3.0)", lambda x: [erf((x * F(3)).astype(D)).astype(F)], **APPROX),
    spec("p80_gelu_2d", "F.gelu(torch.randn(64, 96))",
         lambda x: [(0.5 * x.astype(D) * (1 + erf(x.astype(D) / np.sqrt(2)))).astype(F)], **APPROX),
    spec("p81_sin_wide", "torch.sin(torch.randn(8192) * 30.0)", lambda x: [np.sin((x * F(30)).astype(D)).astype(F)], **APPROX),
    spec("p82_cos_wide", "torch.cos(torch.randn(8192) * 30.0)", lambda x: [np.cos((x * F(30)).astype(D)).astype(F)], **APPROX),
    spec("p83_sin_small", "torch.sin(torch.randn(4096) * 0.01)", lambda x: [np.sin((x * F(0.01)).astype(D)).astype(F)], **APPROX),
    spec("p84_tanh_wide", "torch.tanh(torch.randn(8192) * 4.0)", lambda x: [np.tanh((x * F(4)).astype(D)).astype(F)], **APPROX),
    spec("p85_pow5", "torch.randn(1024) ** 5", lambda x: [(x.astype(D) ** 5).astype(F)], **APPROX),
    spec("p86_pow4", "torch.randn(1024) ** 4", lambda x: [(x.astype(D) ** 4).astype(F)], **APPROX),
    spec("p87_pow_neg", "(torch.abs(torch.randn(1024)) + 0.5) ** -2.0",
         lambda x: [((np.abs(x) + F(0.5)).astype(D) ** -2.0).astype(F)], **APPROX),
    spec("p88_pow_tensor_wide", "x = torch.randn(1024); y = torch.randn(1024); (torch.abs(x) + 0.1) ** (y * 3.0)",
         lambda x, y: [((np.abs(x) + F(0.1)).astype(D) ** (y * F(3)).astype(D)).astype(F)], rtol=1e-4, atol=1e-5),
    spec("p89_exp_wide", "torch.exp(torch.randn(4096) * 8.0)", lambda x: [np.exp((x * F(8)).astype(D)).astype(F)], **APPROX),
    spec("p90_log_wide", "torch.log(torch.abs(torch.randn(4096)) * 1000.0 + 1.0)",
         lambda x: [np.log((np.abs(x) * F(1000) + F(1)).astype(D)).astype(F)], **APPROX),
    spec("r48_sum_erf", "torch.erf(torch.randn(128, 256)).sum(-1)",
         lambda x: [erf(x.astype(D)).astype(F).astype(D).sum(-1).astype(F)], knobs=REDUCE_KNOBS, **REDUCE),
    spec("f48_softmax_tanh", "torch.softmax(torch.tanh(torch.randn(64, 256)), -1)",
         lambda x: [_softmax(np.tanh(x.astype(D)).astype(F).astype(D)).astype(F)], **FUSED),
]
