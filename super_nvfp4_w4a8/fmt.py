"""NVFP4 checkpoint tensors -> W4A8 kernel representation (no persistent FP8 expansion).

Canonical NVFP4 (ModelOpt) contract, per expert e and matrix j:
    W[e, n, k] = G[e] * float(S[e, n, k // 16]) * e2m1(code[e, n, k])
      code : packed uint8 [E, N, K/2]; low nibble = even k, high nibble = odd k
      S    : float8_e4m3fn [E, N, K/16]   (``weight_scale``)
      G    : float32 [E]                  (``weight_scale_2``; multiply to dequantize)

W4A8 kernel representation (what this module produces):
    r[e, n]      = 2 ** ceil(log2(6 * max_g S[e, n, g] / 448))     (power of two)
    S_rel[e,n,g] = S[e, n, g] / r[e, n]        stored as float8_e4m3fn (exact unless
                                               it falls in the E4M3 subnormal range)
    B[e, n]      = G[e] * r[e, n]              float32, applied in the GEMM epilogue
    W8[e, n, k]  = fp8_e4m3_rne(e2m1(code) * S_rel)   built in registers by the kernel
    W[e, n, k]  ~= B[e, n] * W8[e, n, k]

The packed codes are kept untouched (4 bit/weight in HBM). Only the 1-byte group
scales are rewritten in place of the originals, plus one fp32 per output row.
The *only* new weight-side rounding is the final fp8 RNE of (e2m1 * S_rel), which
needs up to 6 significant bits (2 from e2m1, 4 from the E4M3 scale) while E4M3 has 4. ``weight_conversion_error`` measures it.
"""
from __future__ import annotations

import torch

FP8_MAX = 448.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_lut(device=None, dtype=torch.float32) -> torch.Tensor:
    pos = torch.tensor(E2M1_VALUES, dtype=dtype, device=device)
    return torch.cat([pos, -pos])  # code 8..15 are negative (code 8 == -0.0)


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> uint8 codes [..., K] (even k = low nibble)."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed NVFP4 weights must be uint8, got {packed.dtype}")
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def canonical_global_scale(g: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Accept [E], [E,1], [E,shards] (all shards must agree) or a scalar."""
    g = g.detach().float()
    if g.numel() == 1:
        return g.reshape(1).expand(num_experts).contiguous()
    g = g.reshape(num_experts, -1)
    if g.shape[1] > 1 and not torch.equal(g, g[:, :1].expand_as(g)):
        raise ValueError("non-gated expert has differing per-shard weight_scale_2 values; "
                         "refusing to merge them silently")
    return g[:, 0].contiguous()


def dequant_nvfp4(packed, scale, gscale) -> torch.Tensor:
    """R1 weights: faithful float32 dequantization [E, N, K]."""
    E, N, Kh = packed.shape
    codes = unpack_codes(packed).long()
    vals = e2m1_lut(packed.device)[codes]
    s = scale.float().repeat_interleave(16, dim=-1)
    if s.shape[-1] != 2 * Kh:
        raise ValueError(f"scale K/16 mismatch: {tuple(scale.shape)} vs packed {tuple(packed.shape)}")
    g = canonical_global_scale(gscale, E)
    return vals * s * g[:, None, None]


def prepare_w4a8(packed: torch.Tensor, scale: torch.Tensor, gscale: torch.Tensor):
    """Returns (packed, s_rel_fp8 [E,N,K/16], row_scale fp32 [E,N])."""
    if scale.dtype != torch.float8_e4m3fn:
        raise TypeError(f"NVFP4 block scale must be float8_e4m3fn, got {scale.dtype}")
    E, N, Kh = packed.shape
    if scale.shape != (E, N, Kh // 8):
        raise ValueError(f"expected scale shape {(E, N, Kh // 8)}, got {tuple(scale.shape)}")
    s = scale.float()
    if not torch.isfinite(s).all():
        raise ValueError("non-finite NVFP4 block scale")
    smax = s.amax(dim=-1)
    # rows whose scales are all zero: r = 1 (row is all zeros anyway)
    r = torch.where(smax > 0, torch.exp2(torch.ceil(torch.log2(6.0 * smax / FP8_MAX))),
                    torch.ones_like(smax))
    s_rel = (s / r[..., None]).to(torch.float8_e4m3fn)
    g = canonical_global_scale(gscale, E).to(s.device)
    row_scale = (g[:, None] * r).float().contiguous()
    return packed.contiguous(), s_rel.contiguous(), row_scale


def w8_reference(packed, s_rel, dtype=torch.float32) -> torch.Tensor:
    """R2 weight operand exactly as the kernel builds it: fp8(e2m1 * S_rel)."""
    codes = unpack_codes(packed).long()
    vals = e2m1_lut(packed.device)[codes]
    sr = s_rel.float().repeat_interleave(16, dim=-1)
    prod = vals * sr                       # exact in fp32 (and in fp16): <= 6 sig. bits
    return prod.to(torch.float8_e4m3fn).to(dtype)


def weight_conversion_error(packed, scale, gscale, s_rel, row_scale) -> dict:
    w1 = dequant_nvfp4(packed, scale, gscale)
    w2 = w8_reference(packed, s_rel) * row_scale[..., None]
    d = (w2 - w1)
    return {
        "rel_l2": (d.norm() / w1.norm().clamp_min(1e-30)).item(),
        "max_abs": d.abs().max().item(),
        "frac_changed": (d != 0).float().mean().item(),
        "s_rel_subnormal_frac": (s_rel.float() < 2 ** -6).logical_and(s_rel.float() > 0).float().mean().item(),
    }
