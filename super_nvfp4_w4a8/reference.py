"""High-precision references that separate quantization loss from implementation bugs.

R0: original BF16 expert weights, BF16 activations, exact math         (needs BF16 ckpt)
R1: faithfully dequantized NVFP4 weights, BF16 activations/intermediate (stock A16 policy)
R2: the EXACT W4A8 algorithm of moe.py (fp8 operands, scales, roundings), exact matmul
DUT: Triton kernel.   R1-R0 = PTQ weight loss, R2-R1 = A8 + weight-operand loss,
                      DUT-R2 = implementation error (should be accumulation-order only).

All matmuls run in float64 on CPU by default (no TF32 possible); on GPU they run in
fp32 with TF32 disabled.
"""
from __future__ import annotations

import contextlib
import torch

from .fmt import dequant_nvfp4, w8_reference

FP8_MAX = 448.0


@contextlib.contextmanager
def exact_matmul():
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[:2]
        torch.set_float32_matmul_precision(old[2])


def rowwise_fp8_ref(x: torch.Tensor, relu2: bool = False):
    """Bit-exact mirror of kernels.rowwise_fp8_quant (fp32 math, IEEE div)."""
    x = x.float()
    if relu2:
        x = torch.clamp_min(x, 0.0)
        x = x * x
    amax = x.abs().amax(dim=-1, keepdim=True)
    nz = amax > 0
    safe = torch.where(nz, amax, torch.ones_like(amax))
    inv = torch.where(nz, 448.0 / safe, torch.ones_like(amax))
    scale = torch.where(nz, safe / 448.0, torch.ones_like(amax))
    q = torch.clamp(x * inv, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.squeeze(-1)


class _lazy:
    """Per-expert weight materialisation (only experts that are routed to)."""
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, e):
        return self.fn(e)


def _expert_view(W):
    return W if isinstance(W, _lazy) else _lazy(lambda e: W[e])


def moe_r2(x, topk_w, topk_ids, w1, w2, acc_dtype=torch.float64):
    """Exact executable model of the W4A8 algorithm (mirrors moe.py boundaries)."""
    (p1, s1, b1), (p2, s2, b2) = w1, w2
    dev = x.device
    x8, xs = rowwise_fp8_ref(x)
    W1 = _lazy(lambda e: w8_reference(p1[e:e + 1], s1[e:e + 1], acc_dtype)[0])
    W2 = _lazy(lambda e: w8_reference(p2[e:e + 1], s2[e:e + 1], acc_dtype)[0])
    M, top_k = topk_ids.shape
    H = p2.shape[1]
    I = p1.shape[1]
    # FC1 per slot -> bf16
    h = torch.zeros((M, top_k, I), dtype=torch.bfloat16, device=dev)
    for e in torch.unique(topk_ids).tolist():
        t, k = (topk_ids == e).nonzero(as_tuple=True)
        acc = x8[t].to(acc_dtype) @ W1[e].T
        acc = acc.float() * b1[e].float()[None, :]
        acc = acc * xs[t][:, None]
        h[t, k] = acc.to(torch.bfloat16)
    h8, hs = rowwise_fp8_ref(h.reshape(M * top_k, I), relu2=True)
    h8 = h8.reshape(M, top_k, I)
    hs = hs.reshape(M, top_k)
    y = torch.zeros((M, top_k, H), dtype=torch.float32, device=dev)
    for e in torch.unique(topk_ids).tolist():
        t, k = (topk_ids == e).nonzero(as_tuple=True)
        acc = h8[t, k].to(acc_dtype) @ W2[e].T
        acc = acc.float() * b2[e].float()[None, :]
        acc = acc * hs[t, k][:, None]
        acc = acc * topk_w[t, k].float()[:, None]
        y[t, k] = acc
    return y.sum(dim=1).to(x.dtype), dict(x8=x8, xs=xs, h=h, h8=h8, hs=hs, y_slots=y)


def moe_dense_ref(x, topk_w, topk_ids, W1, W2, acc_dtype=torch.float64, bf16_intermediate=True):
    """R0/R1: dense weights [E,I,H], [E,H,I]; BF16 activations, exact matmul."""
    M, top_k = topk_ids.shape
    H = W2[int(topk_ids[0, 0])].shape[0]
    W1, W2 = _expert_view(W1), _expert_view(W2)
    xd = x.to(acc_dtype)
    y = torch.zeros((M, top_k, H), dtype=acc_dtype, device=x.device)
    for e in torch.unique(topk_ids).tolist():
        t, k = (topk_ids == e).nonzero(as_tuple=True)
        h = xd[t] @ W1[e].to(acc_dtype).T
        if bf16_intermediate:
            h = h.to(torch.bfloat16).to(acc_dtype)
        h = torch.clamp_min(h, 0) ** 2
        if bf16_intermediate:
            h = h.to(torch.bfloat16).to(acc_dtype)
        y[t, k] = (h @ W2[e].to(acc_dtype).T) * topk_w[t, k].to(acc_dtype)[:, None]
    return y.sum(dim=1).to(x.dtype)


def moe_r1(x, topk_w, topk_ids, nv1, nv2, **kw):
    """nv = (packed, scale_e4m3, weight_scale_2) straight from the checkpoint."""
    def deq(nv):
        p, sc, g = nv
        g = g.reshape(-1)
        return _lazy(lambda e: dequant_nvfp4(p[e:e + 1], sc[e:e + 1], g[e:e + 1] if g.numel() > 1 else g)[0])
    return moe_dense_ref(x, topk_w, topk_ids, deq(nv1), deq(nv2), **kw)


def compare(a: torch.Tensor, b: torch.Tensor, abs_floor: float = 1e-6) -> dict:
    a = a.double()
    b = b.double()
    d = a - b
    fin = bool(torch.isfinite(a).all())
    row_rel = d.norm(dim=-1) / b.norm(dim=-1).clamp_min(abs_floor)
    return {
        "finite": fin,
        "max_abs": d.abs().max().item(),
        "mae": d.abs().mean().item(),
        "rel_l2": (d.norm() / b.norm().clamp_min(abs_floor)).item(),
        "cosine": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
        "row_rel_p99": torch.quantile(row_rel.float(), 0.99).item() if row_rel.numel() > 1 else row_rel.max().item(),
        "row_rel_max": row_rel.max().item(),
    }
