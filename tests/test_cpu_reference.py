"""CPU-only checks of the NVFP4 -> W4A8 representation and of the R2 reference.

Run:  python -m pytest -q tests/test_cpu_reference.py     (torch CPU, no GPU/triton needed
for everything except the import of moe.moe_align, which only needs triton.cdiv)
"""
import math
import itertools
import pytest
import torch

from super_nvfp4_w4a8.fmt import (
    E2M1_VALUES, unpack_codes, dequant_nvfp4, prepare_w4a8, w8_reference,
    canonical_global_scale, weight_conversion_error,
)
from super_nvfp4_w4a8.reference import rowwise_fp8_ref, moe_r2, moe_r1, compare

torch.manual_seed(0)
FP8 = torch.float8_e4m3fn


def quantize_nvfp4(w: torch.Tensor):
    """ModelOpt-style NVFP4 PTQ of [E, N, K] for synthetic fixtures."""
    E, N, K = w.shape
    g = w.abs().amax(dim=(1, 2)) / (6 * 448)                      # weight_scale_2 [E]
    blk = w.reshape(E, N, K // 16, 16)
    s = (blk.abs().amax(-1) / 6 / g[:, None, None]).to(FP8)        # E4M3 block scale
    sf = s.float().clamp_min(2 ** -9)
    x = blk / (sf[..., None] * g[:, None, None, None])
    grid = torch.tensor(E2M1_VALUES)
    idx = (x.abs()[..., None] - grid).abs().argmin(-1)
    code = idx | ((x < 0).to(torch.int64) << 3)
    code = code.reshape(E, N, K).to(torch.uint8)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous(), s.contiguous(), g


def test_unpack_order_and_lut():
    packed = torch.tensor([[[0x21, 0xF8]]], dtype=torch.uint8)
    assert unpack_codes(packed).tolist() == [[[1, 2, 8, 15]]]
    s = torch.ones((1, 1, 1), dtype=FP8)
    p = torch.zeros((1, 1, 8), dtype=torch.uint8)
    p[0, 0, :2] = torch.tensor([0x21, 0xF8], dtype=torch.uint8)
    w = dequant_nvfp4(p, s, torch.ones(1))
    assert w[0, 0, :4].tolist() == [0.5, 1.0, -0.0, -6.0]


def test_k32_tile_two_different_group_scales():
    # one K32 region: first 16 elements scale 0.25, next 16 scale 3.5 -> must not merge
    codes = torch.tensor([7] * 16 + [3] * 16, dtype=torch.uint8).reshape(1, 1, 32)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    s = torch.tensor([[[0.25, 3.5]]]).to(FP8)
    g = torch.tensor([0.01])
    w1 = dequant_nvfp4(packed, s, g)
    assert torch.allclose(w1[0, 0, :16], torch.full((16,), 6 * 0.25 * 0.01))
    assert torch.allclose(w1[0, 0, 16:], torch.full((16,), 1.5 * 3.5 * 0.01))
    p, s_rel, b = prepare_w4a8(packed, s, g)
    w2 = w8_reference(p, s_rel) * b[..., None]
    # r = 2^-4 -> S_rel = 4 and 56. 6*4 = 24 is exact in E4M3; 1.5*56 = 84 = 1.0101b*2^6
    # needs 4 mantissa bits, so the single FP8 operand rounds it (RNE tie -> 80).
    # This is the documented second rounding, not a merged-scale bug.
    assert torch.equal(w2[0, 0, :16].float(), w1[0, 0, :16].float())
    assert torch.allclose(w2[0, 0, 16:].float(), torch.full((16,), 80 * 2 ** -4 * 0.01))
    assert ((w2[0, 0, 16:] - w1[0, 0, 16:]).abs() / w1[0, 0, 16:].abs()).max() <= 2 ** -4
    # merged-scale bug detector: the halves must keep their own scale (ratio 80/24, not 1)
    assert not torch.allclose(w2[0, 0, :16], w2[0, 0, 16:])
    # an exactly representable second half (code 1.0 * 56) must stay exact
    codes2 = torch.tensor([7] * 16 + [2] * 16, dtype=torch.uint8).reshape(1, 1, 32)
    packed2 = codes2[..., 0::2] | (codes2[..., 1::2] << 4)
    p2, s_rel2, b2 = prepare_w4a8(packed2, s, g)
    assert torch.equal((w8_reference(p2, s_rel2) * b2[..., None]).float(),
                       dequant_nvfp4(packed2, s, g).float())


@pytest.mark.parametrize("gshape", [(), (1,), (3,), (3, 1)])
def test_global_scale_shapes(gshape):
    g = torch.full(gshape, 0.5) if gshape else torch.tensor(0.5)
    assert canonical_global_scale(g, 3).shape == (3,)
    with pytest.raises(ValueError):
        canonical_global_scale(torch.tensor([[1.0, 2.0]] * 3), 3)


def test_prepare_range_and_single_rounding():
    w = torch.randn(4, 64, 256) * torch.exp(torch.randn(4, 64, 1))
    packed, s, g = quantize_nvfp4(w)
    p, s_rel, b = prepare_w4a8(packed, s, g)
    w8 = w8_reference(p, s_rel)
    assert w8.abs().max() <= 448
    # r is a power of two
    r = b / g[:, None]
    assert torch.equal(torch.exp2(torch.round(torch.log2(r))), r)
    w1 = dequant_nvfp4(packed, s, g)
    w2 = w8 * b[..., None]
    # one E4M3 RNE: relative error <= 2^-4 for normal results
    normal = (w8.abs() >= 2 ** -6)
    rel = ((w2 - w1).abs() / w1.abs().clamp_min(1e-30))[normal]
    assert rel.max() <= 2 ** -4 + 1e-7
    err = weight_conversion_error(packed, s, g, s_rel, b)
    assert err["rel_l2"] < 0.04, err


def test_all_code_scale_pairs_exact_in_fp16():
    e4 = torch.arange(256, dtype=torch.uint8).view(FP8).float()
    e4 = e4[torch.isfinite(e4) & (e4 >= 0) & (e4 <= 448 / 6)]
    q = torch.tensor(E2M1_VALUES)
    prod32 = q[:, None] * e4[None, :]
    prod16 = (q.half()[:, None] * e4.half()[None, :]).float()
    assert torch.equal(prod16, prod32)


def test_rowwise_quant_zero_rows_and_relu2():
    x = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 1.0, 3.0]], dtype=torch.bfloat16)
    q, s = rowwise_fp8_ref(x)
    assert s[0] == 1.0 and q[0].float().abs().sum() == 0
    q, s = rowwise_fp8_ref(x, relu2=True)
    assert torch.isclose(s[1], torch.tensor(9.0 / 448))
    assert q[1].float().tolist()[0] == 0.0


def test_relu2_applied_after_fc1_scale():
    # FC1 output = row_scale * acc; ReLU2 must see the scaled value (scale appears squared)
    E, I, H = 1, 32, 32
    w = torch.randn(E, I, H)
    w2 = torch.randn(E, H, I)
    nv1, nv2 = quantize_nvfp4(w), quantize_nvfp4(w2)
    x = torch.randn(4, H, dtype=torch.bfloat16)
    ids = torch.zeros(4, 1, dtype=torch.int64)
    tw = torch.ones(4, 1)
    p1 = prepare_w4a8(*nv1)
    p2 = prepare_w4a8(*nv2)
    y2, _ = moe_r2(x, tw, ids, p1, p2)
    y1 = moe_r1(x, tw, ids, nv1, nv2)
    c = compare(y2, y1)
    assert c["cosine"] > 0.99, c
    # scaling W1's global scale by 4 must scale the output by 16 (ReLU2 after scale)
    nv1b = (nv1[0], nv1[1], nv1[2] * 4)
    y2b, _ = moe_r2(x, tw, ids, prepare_w4a8(*nv1b), p2)
    assert compare(y2b.float(), (y2.float() * 16)).get("rel_l2") < 0.02


def test_moe_align_matches_definition():
    pytest.importorskip("triton")
    from super_nvfp4_w4a8.moe import moe_align
    E, M, k, bm = 7, 13, 3, 4
    ids = torch.randint(0, E, (M, k))
    ids[:, 0] = 2  # one popular expert, some unused
    sid, eid, npp = moe_align(ids, bm, E)
    n = int(npp)
    flat = ids.reshape(-1)
    seen = []
    for b in range(n // bm):
        e = int(eid[b])
        blk = sid[b * bm:(b + 1) * bm]
        real = blk[blk < M * k]
        assert all(int(flat[i]) == e for i in real)
        seen += real.tolist()
    assert sorted(seen) == list(range(M * k))
    assert (sid[n:] == M * k).all()
