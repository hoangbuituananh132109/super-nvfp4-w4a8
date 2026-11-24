"""Recover the nibble permutation that vLLM's Marlin NVFP4 MoE prep applies to the FP4 codes.

Goal: a decode kernel that reads Marlin-layout codes directly, so the hybrid path does not need a
second copy of the 4-bit weights. The permutation is measured, not assumed: for every bit b of the
standard element index p = n*K + k, a one-expert bank whose nibble at p equals bit_b(p) is pushed
through prepare_nvfp4_moe_layer_for_marlin; reading output nibble q gives bit_b(src(q)).
Value preservation is checked separately with constant banks (all 16 codes).

Returns per matrix: shape, output dtype/shape, whether it is a pure permutation, whether codes are
preserved, and inv[p] (output nibble index of standard element p) as int32 for offline analysis.
"""
from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn


def _nibbles(t: torch.Tensor) -> torch.Tensor:
    b = t.contiguous().view(torch.uint8).reshape(-1)
    return torch.stack([b & 0xF, b >> 4], 1).reshape(-1)   # little endian: low nibble first


def _pack(codes: torch.Tensor) -> torch.Tensor:  # [..., K] codes 0..15 -> [..., K/2] u8 (low=even k)
    c = codes.to(torch.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()


def _prep(c1, c2, dev):
    """c1: [N1,K1] codes (fc1, N1=I, K1=H), c2: [N2,K2] codes (fc2). Returns Marlin w13, w2."""
    from super_nvfp4_w4a8.dense_fp8 import _marlin_prepare
    (n1, k1), (n2, k2) = c1.shape, c2.shape
    one = torch.ones((), device=dev)
    nv1 = (_pack(c1)[None], torch.ones((1, n1, k1 // 16), device=dev).to(FP8), one.reshape(1))
    nv2 = (_pack(c2)[None], torch.ones((1, n2, k2 // 16), device=dev).to(FP8), one.reshape(1))
    w13, _, _, w2, _, _, _ = _marlin_prepare(nv1, nv2)
    return w13, w2


def probe(I: int, H: int, dev) -> dict:
    shapes = {"w13": (I, H), "w2": (H, I)}
    res = {k: {"shape_NK": v} for k, v in shapes.items()}
    idx = {k: torch.arange(n * kk, device=dev, dtype=torch.int64).reshape(n, kk) for k, (n, kk) in shapes.items()}
    # value preservation
    preserved = {"w13": True, "w2": True}
    for c in range(16):
        o1, o2 = _prep(torch.full(shapes["w13"], c, device=dev), torch.full(shapes["w2"], c, device=dev), dev)
        for k, o in (("w13", o1), ("w2", o2)):
            nb = _nibbles(o)
            vals = torch.unique(nb).tolist()
            if vals != [c]:
                preserved[k] = False
                res[k].setdefault("value_map_issue", {})[c] = vals[:8]
    for k in shapes:
        res[k]["codes_preserved"] = preserved[k]
    # bit planes
    nbits = {k: int(idx[k].numel() - 1).bit_length() for k in shapes}
    src = {}
    for b in range(max(nbits.values())):
        # codes 1 + bit: output nibbles that stay 0 are padding, not element 0
        o1, o2 = _prep(1 + ((idx["w13"] >> b) & 1), 1 + ((idx["w2"] >> b) & 1), dev)
        for k, o in (("w13", o1), ("w2", o2)):
            if b == 0:
                res[k]["out_dtype"], res[k]["out_shape"] = str(o.dtype), list(o.shape)
                src[k] = torch.zeros(_nibbles(o).numel(), device=dev, dtype=torch.int64)
                res[k]["_valid"] = _nibbles(o) != 0
            nb = _nibbles(o).to(torch.int64)
            if b < nbits[k]:
                src[k] |= ((nb - 1) & 1) << b
    for k in shapes:
        n_el = idx[k].numel()
        valid = res[k].pop("_valid")
        res[k]["out_nibbles"] = src[k].numel()
        res[k]["padding_nibbles"] = int((~valid).sum())
        q = torch.nonzero(valid).reshape(-1)
        s = src[k][q]
        ok = s.numel() == n_el and torch.equal(torch.sort(s)[0], torch.arange(n_el, device=dev))
        res[k]["is_permutation"] = bool(ok)
        if ok:
            inv = torch.empty(n_el, device=dev, dtype=torch.int64)
            inv[s] = q
            res[k]["inv"] = inv.to(torch.int32).cpu()
            # quick structure hints: first rows/cols of inv as (n,k) grid
            g = inv.reshape(shapes[k])
            res[k]["inv_corner_16x16"] = g[:16, :16].cpu().tolist()
            res[k]["inv_row0_step"] = (g[0, 1:33] - g[0, :32]).cpu().tolist()
            res[k]["inv_col0_step"] = (g[1:33, 0] - g[:32, 0]).cpu().tolist()
    return res
