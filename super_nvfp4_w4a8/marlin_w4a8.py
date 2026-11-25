"""Fused NVFP4 x FP8 grouped MoE GEMM that reads vLLM's *Marlin* NVFP4 tensors directly.

Same numerics as the decode + CUTLASS path of dense_fp8.py (W8 = fp8_rne(e2m1 * S_rel), per-token
FP8 activations, FP32 accumulation promoted every BLOCK_K), but the FP8 weight tile is produced in
registers from the 4-bit Marlin codes and fed straight into the tensor-core dot, so nothing is
materialised in HBM and no extra copy of the weights exists (the Marlin W4A16 path owns them).

Swap-AB: acc[BLOCK_N, BLOCK_M] = W8[BLOCK_N, BLOCK_K] @ X8^T[BLOCK_K, BLOCK_M]; weight rows are the
MMA M dimension (64 per warpgroup), tokens of one expert the N dimension (>= 16), which suits the
decode regime where an expert sees only a handful of tokens.

Marlin layout (tests/probe_marlin_layout.py): codes int32 [E, K/16, 2N]; a 64n x 16k tile is 128
words, word = j + 4kq + 16r, nibble slot = b0 + 2b1 + 4b2, n = 16j + 8b1 + r, k = 8b0 + 2kq + b2;
tile (kt, nt) starts at word kt*2N + nt*128. Scale byte position: dense_fp8._marlin_scale_pos.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .dense_fp8 import _e2m1x_srel_to_fp8, _marlin_scale_pos


@triton.jit
def marlin_w4a8_moe_gemm(
    a_ptr, a_scale_ptr,                 # fp8 [rows, K], fp32 [rows]
    q_ptr, ms_ptr, shift_ptr, bscale_ptr,  # i32 [E,K/16,2N], u8 [E,K/16,N], i8 [E,N], fp32 [E,N]
    c_ptr, topk_w_ptr,                  # out [slots, N], fp32 [slots]
    sorted_ids_ptr, expert_ids_ptr, num_post_pad_ptr,
    N, K, num_valid_slots, stride_am, stride_cm,
    A_ROW_DIV: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr, NT: tl.constexpr, T: tl.constexpr, GROUP_N: tl.constexpr,
):
    BLOCK_N: tl.constexpr = 64 * NT
    BLOCK_K: tl.constexpr = 16 * T
    pid = tl.program_id(0)
    num_pid_n = N // BLOCK_N
    num_post_pad = tl.load(num_post_pad_ptr)
    num_pid_m = tl.cdiv(num_post_pad, BLOCK_M)
    if pid >= num_pid_n * num_pid_m:
        return
    num_in_group = GROUP_N * num_pid_m
    group_id = pid // num_in_group
    first_n = group_id * GROUP_N
    gsize = tl.minimum(num_pid_n - first_n, GROUP_N)
    pid_n = first_n + ((pid % num_in_group) % gsize)
    pid_m = (pid % num_in_group) // gsize

    offs_tok = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slot = tl.load(sorted_ids_ptr + offs_tok).to(tl.int64)
    valid = slot < num_valid_slots
    slot = tl.where(valid, slot, 0)
    arow = slot // A_ROW_DIV
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    c_ptrs = c_ptr + slot[None, :] * stride_cm + n[:, None]
    c_mask = valid[None, :]
    if e < 0:
        tl.store(c_ptrs, tl.zeros((BLOCK_N, BLOCK_M), dtype=c_ptr.dtype.element_ty), mask=c_mask)
        return

    KT = K // 16
    nti = tl.arange(0, NT)
    t = tl.arange(0, T)
    r = tl.arange(0, 8)
    kq = tl.arange(0, 4)
    j = tl.arange(0, 4)
    inner = j[None, None, :] + 4 * kq[None, :, None] + 16 * r[:, None, None]                 # [8,4,4]
    tile_off = ((pid_n * NT + nti) * 128)[:, None, None, None, None] + inner[None, None, :, :, :]  # [NT,1,8,4,4]
    q_base = q_ptr + e * KT * (2 * N)
    sh = 4 * tl.arange(0, 8)
    spos = _marlin_scale_pos(n)
    ms_base = ms_ptr + e * KT * N
    sft = tl.load(shift_ptr + e * N + n).to(tl.int32)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + arow[None, :] * stride_am + offs_k[:, None]

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for ks in range(0, KT // T):
        kt = ks * T + t
        w = tl.load(q_base + (kt.to(tl.int64) * (2 * N))[None, :, None, None, None] + tile_off)  # [NT,T,8,4,4]
        nib = ((w[:, :, :, :, :, None] >> sh[None, None, None, None, None, :]) & 0xF).to(tl.uint8)
        nib = tl.reshape(nib, (NT, T, 8, 4, 4, 2, 2, 2))            # nt,t,r,kq,j,b2,b1,b0
        codes = tl.reshape(tl.permute(nib, (0, 4, 6, 2, 1, 7, 3, 5)), (BLOCK_N, T, 16))
        b = tl.load(ms_base + (kt * N)[None, :] + spos[:, None]).to(tl.int32)              # [BN,T]
        ex = (b >> 3) + (127 - 15) - sft[:, None]
        bits = (ex << 23) | ((b & 7) << 20)
        bits = tl.where((b == 0) | (ex < 1), 0, bits)
        sr = bits.to(tl.float32, bitcast=True).to(tl.float8e4nv).to(tl.float16)
        w8 = tl.reshape(_e2m1x_srel_to_fp8(codes, sr[:, :, None]), (BLOCK_N, BLOCK_K))
        aT = tl.load(a_ptrs, mask=valid[None, :], other=0.0)
        acc += tl.dot(w8, aT)            # fresh FP8 accumulator per BLOCK_K, promoted into FP32
        a_ptrs += BLOCK_K

    bsc = tl.load(bscale_ptr + e * N + n)
    asc = tl.load(a_scale_ptr + arow, mask=valid, other=0.0)
    acc = acc * bsc[:, None] * asc[None, :]
    if MUL_ROUTED_WEIGHT:
        tw = tl.load(topk_w_ptr + slot, mask=valid, other=0.0)
        acc = acc * tw[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# keyed by upper bound of average slots per expert (M * top_k / E)
FUSED_CFGS = {
    8: dict(BLOCK_M=16, NT=2, T=8, GROUP_N=8, num_warps=8, num_stages=3),
    24: dict(BLOCK_M=32, NT=2, T=8, GROUP_N=8, num_warps=8, num_stages=3),
    1 << 30: dict(BLOCK_M=64, NT=2, T=8, GROUP_N=8, num_warps=8, num_stages=3),
}


def _cfg(avg, N, K, override=None):
    c = dict(override) if override else dict(next(v for ub, v in sorted(FUSED_CFGS.items()) if avg <= ub))
    while (N // 64) % c["NT"]:
        c["NT"] //= 2
    while (K // 16) % c["T"]:
        c["T"] //= 2
    return c


def _align(topk_ids, block_m, E):
    try:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        sid, eid, npp = moe_align_block_size(topk_ids, block_m, E)
        return sid, eid, npp
    except Exception:  # noqa: BLE001  (older/newer vLLM): pure torch fallback
        from .moe import moe_align
        return moe_align(topk_ids, block_m, E)


def _gemm(a8, a_s, q, ms, shift, bscale, out, tw, sid, eid, npp, num_valid, N, K, a_row_div, mul, c):
    bm = c["BLOCK_M"]
    grid = (triton.cdiv(sid.numel(), bm) * (N // (64 * c["NT"])),)
    marlin_w4a8_moe_gemm[grid](
        a8, a_s, q, ms, shift, bscale, out, tw, sid, eid, npp, N, K, num_valid,
        a8.stride(0), out.stride(0), A_ROW_DIV=a_row_div, MUL_ROUTED_WEIGHT=mul,
        BLOCK_M=bm, NT=c["NT"], T=c["T"], GROUP_N=c.get("GROUP_N", 8),
        num_warps=c.get("num_warps", 8), num_stages=c.get("num_stages", 3))


def fused_marlin_w4a8_moe(dense, x, topk_weights, topk_ids, cfg1=None, cfg2=None):
    """dense: dense_fp8.Fp8DenseMoE built with marlin_codes + marlin_scales (share_codes)."""
    from .moe import rowwise_quant
    M, H = x.shape
    top_k = topk_ids.shape[1]
    E, I = dense.E, dense.I
    ns = M * top_k
    avg = ns / E
    q1, q2 = dense.mq
    s1, s2 = dense.ms
    c1 = _cfg(avg, I, H, cfg1)
    c2 = _cfg(avg, H, I, cfg2)
    tw = topk_weights.reshape(-1).to(torch.float32)
    x8, xs = rowwise_quant(x.contiguous(), relu2=False)
    sid, eid, npp = _align(topk_ids, c1["BLOCK_M"], E)
    h = torch.empty((ns, I), dtype=torch.bfloat16, device=x.device)
    _gemm(x8, xs, q1, s1.view(torch.uint8), dense.w1[1], dense.b1, h, tw, sid, eid, npp, ns, I, H,
          top_k, False, c1)
    h8, hs = rowwise_quant(h, relu2=True)
    if c2["BLOCK_M"] != c1["BLOCK_M"]:
        sid, eid, npp = _align(topk_ids, c2["BLOCK_M"], E)
    y = torch.empty((ns, H), dtype=torch.float32, device=x.device)
    _gemm(h8, hs, q2, s2.view(torch.uint8), dense.w2[1], dense.b2, y, tw, sid, eid, npp, ns, H, I,
          1, True, c2)
    return y.view(M, top_k, H).sum(dim=1).to(x.dtype)


def cfg_from_env(prefix):
    raw = os.environ.get(prefix)
    if not raw:
        return None
    k = ("BLOCK_M", "NT", "T", "GROUP_N", "num_warps", "num_stages")
    return dict(zip(k, (int(v) for v in raw.split("_"))))
