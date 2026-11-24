"""Host-side orchestration of the NVFP4-W4A8 routed-expert MoE (non-gated ReLU2).

    Y[t] = sum_k w[t,k] * FC2_e( q8( relu(FC1_e(q8(X[t])))^2 ) ),  e = ids[t,k]

Numerical boundaries (identical in ``reference.moe_r2``):
  1. X (bf16) -> per-token dynamic FP8 E4M3 (amax/448, RNE, zero-row guard)
  2. FC1: FP8 x W8 (fp8(e2m1*S_rel)) -> FP32 acc -> * B1[e,n] * xs[t] -> bf16 slot buffer
  3. ReLU2 in fp32 on the fully scaled FC1 output -> per-(t,k)-row dynamic FP8
  4. FC2: FP8 x W8 -> FP32 acc -> * B2[e,n] * hs[t,k] * w[t,k] -> fp32 slot buffer
  5. sum over k in fp32 (fixed order) -> bf16
Routing, routed_scaling_factor, latent projections and shared experts stay in vLLM.
Everything is CUDA-graph capturable (no host sync, fixed grids).
"""
from __future__ import annotations

import os
import json
import functools
import torch
import triton

from .kernels import nvfp4_w4a8_moe_gemm, rowwise_fp8_quant


def moe_align(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    """Sort (token, k) slots by expert and pad each expert segment to block_m.

    Returns sorted_ids [max_padded] (pad value = num_slots), expert_ids [max_blocks],
    num_post_pad [1]. Pure torch, deterministic, no host synchronisation.
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    num_slots = flat.numel()
    dev = flat.device
    max_padded = num_slots + num_experts * (block_m - 1)
    max_padded = triton.cdiv(max_padded, block_m) * block_m
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(num_experts, dtype=torch.int64, device=dev).scatter_add_(
        0, flat, torch.ones_like(flat))  # (bincount would host-sync on CUDA)
    padded = (counts + block_m - 1) // block_m * block_m
    start = torch.cumsum(counts, 0) - counts
    pstart = torch.cumsum(padded, 0) - padded
    e_sorted = flat[order]
    rank = torch.arange(num_slots, device=dev) - start[e_sorted]
    pos = pstart[e_sorted] + rank
    sorted_ids = torch.full((max_padded,), num_slots, dtype=torch.int64, device=dev)
    sorted_ids.scatter_(0, pos, order)
    block_starts = torch.arange(0, max_padded, block_m, device=dev)
    expert_ids = torch.searchsorted(torch.cumsum(padded, 0), block_starts, right=True)
    expert_ids = expert_ids.clamp_(max=num_experts - 1).to(torch.int32)
    num_post_pad = padded.sum().reshape(1).to(torch.int32)
    return sorted_ids, expert_ids, num_post_pad


def rowwise_quant(x: torch.Tensor, relu2: bool = False):
    R, K = x.shape
    q = torch.empty((R, K), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((R,), dtype=torch.float32, device=x.device)
    if R:
        rowwise_fp8_quant[(R,)](x, q, s, K, x.stride(0), q.stride(0),
                                APPLY_RELU2=relu2, BLOCK=triton.next_power_of_2(K),
                                num_warps=8 if K > 2048 else 4)
    return q, s


# ----------------------------------------------------------------------------- configs
_DEFAULT = {  # keyed by upper bound of average slots per expert (M*top_k/E)
    # All spill-free on sm_90a per `ptxas -v` (CPU gate). The original 4-warp 128x128 tiles
    # spilled 6-8 KB/thread and ran 20-200x slower than bandwidth on H200.
    16: dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=128, GROUP_N=8, num_warps=8, num_stages=3),
    48: dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, GROUP_N=8, num_warps=8, num_stages=3),
    128: dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, GROUP_N=8, num_warps=8, num_stages=3),
    1 << 30: dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, GROUP_N=8, num_warps=8, num_stages=3),
}


@functools.lru_cache(maxsize=None)
def _load_table(path: str | None):
    if not path:
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def pick_config(num_tokens: int, top_k: int, num_experts: int, which: str) -> dict:
    table = _load_table(os.environ.get("SUPER_W4A8_CONFIG"))
    if table and which in {"fc1", "fc2"} and isinstance(next(iter(table.values())), dict) \
            and which in next(iter(table.values())):
        # tuned table: {M_upper: {"fc1": cfg, "fc2": cfg}}
        for ub in sorted(table):
            if num_tokens <= ub:
                return dict(table[ub][which])
        return dict(table[max(table)][which])
    avg = num_tokens * top_k / max(num_experts, 1)
    for ub in sorted(_DEFAULT):
        if avg <= ub:
            return dict(_DEFAULT[ub])
    return dict(_DEFAULT[max(_DEFAULT)])


def _promote(cfg: dict) -> bool:
    """FP8 WGMMA on SM90 accumulates in reduced precision; unpromoted accumulation over the
    whole K measured 1-2% rel-L2 vs exact at K=1024-2688 on H200. Default: promote into FP32
    every BLOCK_K (kernel PROMOTE_ACC). Override: cfg["PROMOTE_ACC"] or env SUPER_W4A8_PROMOTE=0
    (diagnostic only)."""
    v = cfg.get("PROMOTE_ACC", os.environ.get("SUPER_W4A8_PROMOTE", "1"))
    return str(v).lower() not in ("0", "false", "no", "off")


# compiled-kernel statistics (registers / spills / shared memory), keyed by launch config
KSTATS: dict = {}


def _gemm(a8, a_s, packed, s_rel, bscale, out, topk_w, sorted_ids, expert_ids, npp,
          num_valid, a_row_div, mul_routed, cfg):
    E, N, Kh = packed.shape
    K = 2 * Kh
    if a8.shape[1] != K:
        raise ValueError(f"activation K={a8.shape[1]} != weight K={K}")
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    if bk % 32:
        raise ValueError("BLOCK_K must be a multiple of 32 (fp8 MMA K and 16-group alignment)")
    grid = (triton.cdiv(sorted_ids.numel(), bm) * triton.cdiv(N, bn),)
    promote = _promote(cfg)
    ck = nvfp4_w4a8_moe_gemm[grid](
        a8, a_s, packed, s_rel, bscale, out, topk_w,
        sorted_ids, expert_ids, npp,
        N, K, num_valid,
        a8.stride(0), a8.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        s_rel.stride(0), s_rel.stride(1), s_rel.stride(2),
        bscale.stride(0), bscale.stride(1),
        out.stride(0), out.stride(1),
        A_ROW_DIV=a_row_div, MUL_ROUTED_WEIGHT=mul_routed,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_N=cfg.get("GROUP_N", 8),
        EVEN_K=(K % bk == 0),
        PROMOTE_ACC=promote,
        num_warps=cfg.get("num_warps", 4), num_stages=cfg.get("num_stages", 3),
    )
    key = (N, K, a_row_div > 1, bm, bn, bk, cfg.get("num_warps", 4), cfg.get("num_stages", 3), promote)
    if key not in KSTATS and ck is not None:
        KSTATS[key] = {k: getattr(ck, k, None) for k in ("n_regs", "n_spills", "n_max_threads")}
        KSTATS[key]["shared"] = getattr(getattr(ck, "metadata", None), "shared", None)


def nvfp4_w4a8_moe(
    x: torch.Tensor,             # [M, H] bf16 (latent width for Super)
    topk_weights: torch.Tensor,  # [M, top_k]
    topk_ids: torch.Tensor,      # [M, top_k] int
    w1: tuple,                   # (packed [E,I,H/2] u8, s_rel [E,I,H/16] fp8, bscale [E,I] fp32)
    w2: tuple,                   # (packed [E,H,I/2] u8, s_rel [E,H,I/16] fp8, bscale [E,H] fp32)
    out: torch.Tensor | None = None,
    chunk_tokens: int | None = None,
    return_intermediates: bool = False,
):
    M, H = x.shape
    top_k = topk_ids.shape[1]
    E, I, _ = w1[0].shape
    if w2[0].shape[1] != H or 2 * w2[0].shape[2] != I or 2 * w1[0].shape[2] != H:
        raise ValueError(f"shape mismatch x={tuple(x.shape)} w1={tuple(w1[0].shape)} w2={tuple(w2[0].shape)}")
    if out is None:
        out = torch.empty((M, H), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    chunk = chunk_tokens or int(os.environ.get("SUPER_W4A8_CHUNK", "8192"))
    inter = {}
    for t0 in range(0, M, chunk):
        t1 = min(M, t0 + chunk)
        xc, idc, twc = x[t0:t1], topk_ids[t0:t1], topk_weights[t0:t1]
        m = t1 - t0
        ns = m * top_k
        tw = twc.reshape(-1).to(torch.float32).contiguous()

        x8, xs = rowwise_quant(xc.contiguous(), relu2=False)
        cfg1 = pick_config(m, top_k, E, "fc1")
        sid, eid, npp = moe_align(idc, cfg1["BLOCK_M"], E)
        h = torch.empty((ns, I), dtype=torch.bfloat16, device=x.device)
        _gemm(x8, xs, *w1, h, tw, sid, eid, npp, ns, top_k, False, cfg1)

        h8, hs = rowwise_quant(h, relu2=True)
        cfg2 = pick_config(m, top_k, E, "fc2")
        if cfg2["BLOCK_M"] != cfg1["BLOCK_M"]:
            sid, eid, npp = moe_align(idc, cfg2["BLOCK_M"], E)
        y = torch.empty((ns, H), dtype=torch.float32, device=x.device)
        _gemm(h8, hs, *w2, y, tw, sid, eid, npp, ns, 1, True, cfg2)
        out[t0:t1] = y.view(m, top_k, H).sum(dim=1).to(out.dtype)
        if return_intermediates and t0 == 0:
            inter = dict(x8=x8, xs=xs, h=h, h8=h8, hs=hs, y_slots=y)
    return (out, inter) if return_intermediates else out
