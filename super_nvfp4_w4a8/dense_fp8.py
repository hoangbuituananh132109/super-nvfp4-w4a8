"""Large-M path: decode NVFP4 -> FP8 per layer into a reusable scratch, then run a tuned
FP8 grouped GEMM (per-token activation scale x per-output-channel weight scale).

Weights stay 4-bit in HBM. Per forward of a layer, the routed experts' FP4 codes are decoded
once into an FP8 scratch (bit-identical to fmt.w8_reference, i.e. the same W8 as the fused
kernel and the R2 reference), then:

    CUTLASS : vllm ops.cutlass_moe_mm (SM90 grouped FP8)        -> run_cutlass()
    TRITON  : vllm fused_experts(use_fp8_w8a8, per-channel)      -> run_triton()

The scratch is shared across layers (one layer at a time), sized E*max(N*K) bytes per matrix.
For Super (E=512, 2688x1024) that is 2 x 1.41 GB. Decode cost is ~read 0.5 B + write 1 B per
weight, amortised over all tokens of the forward; it only pays off for large M (prefill).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1x_srel_to_fp8(code, sr):
    code16 = code.to(tl.uint16)
    bits = ((code16 & 8) << 12) | ((code16 & 7) << 9)
    v = bits.to(tl.float16, bitcast=True) * 16384.0      # exact e2m1
    return (v * sr).to(tl.float8e4nv)                      # exact product, single RNE


@triton.jit
def _nvfp4_to_fp8_kernel(packed_ptr, srel_ptr, out_ptr, n_rows, K,
                         BLOCK_R: tl.constexpr, BLOCK_G: tl.constexpr):
    """packed uint8 [R, K/2], s_rel fp8 [R, K/16] -> fp8 [R, K] = fp8_rne(e2m1(code) * s_rel).

    Each program handles BLOCK_R rows x BLOCK_G groups of 16. Every packed byte and every scale
    byte is read exactly once (8 bytes/group), the output row segment is written contiguously."""
    pid_r = tl.program_id(0)
    pid_g = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    gs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    j = tl.arange(0, 8)
    G = K // 16
    rmask = rows < n_rows
    gmask = gs < G
    m3 = rmask[:, None, None] & gmask[None, :, None]
    # bytes [R, G, 8] (byte b of a group holds elements 2b (low) and 2b+1 (high))
    bptr = packed_ptr + rows[:, None, None] * (K // 2) + gs[None, :, None] * 8 + j[None, None, :]
    byte = tl.load(bptr, mask=m3, other=0)
    sr = tl.load(srel_ptr + rows[:, None] * G + gs[None, :], mask=rmask[:, None] & gmask[None, :],
                 other=0.0).to(tl.float16)
    lo = _e2m1x_srel_to_fp8(byte & 0xF, sr[:, :, None])
    hi = _e2m1x_srel_to_fp8((byte >> 4) & 0xF, sr[:, :, None])
    out = tl.reshape(tl.join(lo, hi), (BLOCK_R, BLOCK_G * 16))     # interleave -> k order
    ks = pid_g * BLOCK_G * 16 + tl.arange(0, BLOCK_G * 16)
    tl.store(out_ptr + rows[:, None] * K + ks[None, :], out,
             mask=rmask[:, None] & (ks < K)[None, :])


@triton.jit
def _nvfp4_to_fp8_flat_kernel(packed_ptr, srel_ptr, out16_ptr, n_bytes, BLOCK: tl.constexpr):
    """Flat decode. Packed bytes [R, K/2] and s_rel [R, K/16] are both row-major and K/2 is a
    multiple of 8, so byte i belongs to group i // 8 of the flattened scale array. Byte i holds
    elements 2i (low nibble) and 2i+1 (high nibble); their FP8 codes are written as one uint16
    (little endian: low byte = element 2i), so input and output share the same index -- no
    interleave/shuffle, fully coalesced 1 B in / 2 B out per lane."""
    pid = tl.program_id(0).to(tl.int64)
    i = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = i < n_bytes
    byte = tl.load(packed_ptr + i, mask=m, other=0)
    sr = tl.load(srel_ptr + (i // 8), mask=m, other=0.0).to(tl.float16)
    lo = _e2m1x_srel_to_fp8(byte & 0xF, sr).to(tl.uint8, bitcast=True).to(tl.uint16)
    hi = _e2m1x_srel_to_fp8((byte >> 4) & 0xF, sr).to(tl.uint8, bitcast=True).to(tl.uint16)
    tl.store(out16_ptr + i, lo | (hi << 8), mask=m)


@triton.jit
def _nib(w, s):
    return (w >> (4 * s)) & 0xF


@triton.jit
def _marlin_nvfp4_to_fp8_kernel(q_ptr, srel_ptr, out_ptr, N, K,
                                BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, COMBINE: tl.constexpr):
    """Decode straight from vLLM Marlin NVFP4 codes (int32 [E, K/16, 2N]) to FP8 [E, N, K].

    Marlin tile = 64 n x 16 k = 128 int32 words. Measured layout (tests/probe_marlin_layout.py;
    closed form verified on the full Super fc1/fc2 permutations): element (n, k) of a tile with
    nn = n % 64, kin = k % 16 sits in word  wi = nn//16 + 4*((kin%8)//2) + 16*(nn%8)  at nibble
    slot  (kin//8) + 2*((nn//8)%2) + 4*(kin%2);  tile (kt, nt) starts at word kt*2N + nt*128.
    Each lane emits 4 consecutive k of one row (k, k+1 from word A; k+2, k+3 from word A+4).
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    e = tl.program_id(2).to(tl.int64)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (n < N)[:, None] & (c < K // 4)[None, :]
    k4 = 4 * c
    kt = k4 // 16
    kin = k4 % 16
    nt = n // 64
    nn = n % 64
    wi = (nn // 16 + 16 * (nn % 8))[:, None] + (4 * ((kin % 8) // 2))[None, :]
    s0 = (2 * ((nn // 8) % 2))[:, None] + (kin // 8)[None, :]
    word = (e * (K // 16) * (2 * N) + (kt * (2 * N)).to(tl.int64)[None, :]
            + (nt * 128).to(tl.int64)[:, None] + wi)
    wA = tl.load(q_ptr + word, mask=mask, other=0)
    wB = tl.load(q_ptr + word + 4, mask=mask, other=0)
    G = K // 16
    sr = tl.load(srel_ptr + e * N * G + n.to(tl.int64)[:, None] * G + kt[None, :], mask=mask,
                 other=0.0).to(tl.float16)
    v0 = _e2m1x_srel_to_fp8(_nib(wA, s0), sr)
    v1 = _e2m1x_srel_to_fp8(_nib(wA, s0 + 4), sr)
    v2 = _e2m1x_srel_to_fp8(_nib(wB, s0), sr)
    v3 = _e2m1x_srel_to_fp8(_nib(wB, s0 + 4), sr)
    row = e * N * K + n.to(tl.int64)[:, None] * K
    if COMBINE:
        u0 = v0.to(tl.uint8, bitcast=True).to(tl.uint32)
        u1 = v1.to(tl.uint8, bitcast=True).to(tl.uint32)
        u2 = v2.to(tl.uint8, bitcast=True).to(tl.uint32)
        u3 = v3.to(tl.uint8, bitcast=True).to(tl.uint32)
        quad = u0 | (u1 << 8) | (u2 << 16) | (u3 << 24)
        tl.store(out_ptr + (row // 4) + c[None, :], quad, mask=mask)  # out_ptr: uint32 view
    else:
        base = out_ptr + row + k4[None, :]
        tl.store(base, v0, mask=mask)
        tl.store(base + 1, v1, mask=mask)
        tl.store(base + 2, v2, mask=mask)
        tl.store(base + 3, v3, mask=mask)


@triton.jit
def _marlin_tile_decode_kernel(q_ptr, srel_ptr, out_ptr, N, K, T: tl.constexpr):
    """Coalesced variant: one program = one 64-row n-tile x T consecutive Marlin k-tiles.

    The 128 words of a tile are read contiguously as [r=8, kq=4, j=4] (word = j + 4kq + 16r), the
    8 nibbles of a word as [b2, b1, b0] (slot = b0 + 2b1 + 4b2); the standard order is
    n = 16j + 8b1 + r, k = 16t + 8b0 + 2kq + b2, i.e. a pure permute (j,b1,r | t,b0,kq,b2)."""
    nt = tl.program_id(0)
    ktg = tl.program_id(1)
    e = tl.program_id(2).to(tl.int64)
    KT = K // 16
    t = tl.arange(0, T)
    r = tl.arange(0, 8)
    kq = tl.arange(0, 4)
    j = tl.arange(0, 4)
    kt = ktg * T + t
    inner = (j[None, None, :] + 4 * kq[None, :, None] + 16 * r[:, None, None])           # [8,4,4]
    word = (e * KT * (2 * N) + (kt.to(tl.int64) * (2 * N))[:, None, None, None]
            + nt * 128 + inner[None, :, :, :])                                             # [T,8,4,4]
    w = tl.load(q_ptr + word, mask=(kt < KT)[:, None, None, None], other=0)
    sh = 4 * tl.arange(0, 8)
    nib = ((w[:, :, :, :, None] >> sh[None, None, None, None, :]) & 0xF).to(tl.uint8)    # [T,8,4,4,8]
    nib = tl.reshape(nib, (T, 8, 4, 4, 2, 2, 2))                                          # t,r,kq,j,b2,b1,b0
    codes = tl.reshape(tl.permute(nib, (3, 5, 1, 0, 6, 2, 4)), (64, T * 16))
    n = nt * 64 + tl.arange(0, 64)
    k = ktg * T * 16 + tl.arange(0, T * 16)
    km = (k < K)[None, :]
    sr = tl.load(srel_ptr + e * N * KT + n.to(tl.int64)[:, None] * KT + (k // 16)[None, :], mask=km,
                 other=0.0).to(tl.float16)
    v = _e2m1x_srel_to_fp8(codes, sr)
    tl.store(out_ptr + e * N * K + n.to(tl.int64)[:, None] * K + k[None, :], v, mask=km)


# tuned on H200, Super layer (bench-0d26c3bf2e): coalesced tile kernel T=8 w2 = 1.61 ms for both matrices
MARLIN_DECODE_CFG = {"impl": "tile", "block_n": 64, "block_c": 32, "num_warps": 2, "combine": False, "T": 8}
MARLIN_DECODE_SWEEP = [(bn, bc, w, cb) for bn in (64, 128) for bc in (16, 32) for w in (4, 8)
                       for cb in (False, True)]
MARLIN_TILE_SWEEP = [(T, w) for T in (1, 2, 4, 8) for w in (2, 4, 8)]


def decode_marlin_to_fp8(q: torch.Tensor, s_rel: torch.Tensor, N: int, K: int, out: torch.Tensor | None = None,
                         block_n=None, block_c=None, num_warps=None, combine=None, impl=None, T=None):
    """Marlin NVFP4 codes [E, K/16, 2N] int32 + s_rel [E, N, K/16] fp8 -> FP8 [E, N, K]."""
    E = q.shape[0]
    if N % 64 or K % 16 or tuple(q.shape) != (E, K // 16, 2 * N) or not q.is_contiguous():
        raise ValueError(f"unexpected Marlin code tensor {tuple(q.shape)} for N={N} K={K}")
    if out is None or out.shape != (E, N, K):
        out = torch.empty((E, N, K), dtype=torch.float8_e4m3fn, device=q.device)
    cfg = MARLIN_DECODE_CFG
    if (impl or cfg["impl"]) == "tile":
        T = T or cfg["T"]
        _marlin_tile_decode_kernel[(N // 64, triton.cdiv(K // 16, T), E)](
            q, s_rel, out, N, K, T=T, num_warps=num_warps or cfg["num_warps"])
        return out
    bn, bc = block_n or cfg["block_n"], block_c or cfg["block_c"]
    cb = cfg["combine"] if combine is None else combine
    dst = out.view(torch.uint32) if cb else out
    _marlin_nvfp4_to_fp8_kernel[(triton.cdiv(N, bn), triton.cdiv(K // 4, bc), E)](
        q, s_rel, dst, N, K, BLOCK_N=bn, BLOCK_C=bc, COMBINE=cb, num_warps=num_warps or cfg["num_warps"])
    return out


# "tile" is the verified-exact default; "flat" was faster but NOT bit-exact on H200 (bench-33fefbcca0)
DECODE_CFG = {"impl": "tile", "block": 4096, "num_warps": 4}
DECODE_SWEEP = [(b, w) for b in (1024, 2048, 4096, 8192) for w in (4, 8)]


def decode_to_fp8(packed: torch.Tensor, s_rel: torch.Tensor, out: torch.Tensor | None = None,
                  impl: str | None = None, block: int | None = None, num_warps: int | None = None,
                  block_r: int = 8, block_g: int = 16):
    """[E, N, K/2] u8 + [E, N, K/16] fp8 -> [E, N, K] fp8 (scratch reused if given)."""
    E, N, Kh = packed.shape
    K = 2 * Kh
    if K % 16:
        raise ValueError("K must be a multiple of 16")
    if out is None or out.shape != (E, N, K):
        out = torch.empty((E, N, K), dtype=torch.float8_e4m3fn, device=packed.device)
    impl = impl or DECODE_CFG["impl"]
    if impl == "flat":
        if not (packed.is_contiguous() and s_rel.is_contiguous() and out.is_contiguous()):
            raise ValueError("flat decode needs contiguous packed / s_rel / out")
        nb = packed.numel()
        blk = block or DECODE_CFG["block"]
        _nvfp4_to_fp8_flat_kernel[(triton.cdiv(nb, blk),)](
            packed, s_rel, out.view(torch.uint16), nb, BLOCK=blk,
            num_warps=num_warps or DECODE_CFG["num_warps"])
    else:  # "tile": v1 kernel (join/reshape interleave)
        R = E * N
        _nvfp4_to_fp8_kernel[(triton.cdiv(R, block_r), triton.cdiv(K // 16, block_g))](
            packed, s_rel, out, R, K, BLOCK_R=block_r, BLOCK_G=block_g, num_warps=num_warps or 4)
    return out


class Fp8DenseMoE:
    """Holds the NVFP4 W4A8 bank (4-bit) and a shared FP8 scratch; runs FP8 grouped GEMMs."""

    _scratch: dict = {}

    def __init__(self, w1, w2, marlin_codes=None):
        # w = (packed [E,N,K/2] u8, s_rel [E,N,K/16] fp8, bscale [E,N] fp32) from fmt.prepare_w4a8
        # marlin_codes = (w13 int32 [E,H/16,2I], w2 int32 [E,I/16,2H]): decode reads these instead
        # of packed (w[0] may be None), so the hybrid path keeps a single copy of the 4-bit codes.
        self.w1, self.w2 = w1, w2
        self.mq = marlin_codes
        E, I, G1 = w1[1].shape
        self.E, self.I, self.H = E, I, 16 * G1
        dev = w1[1].device
        self.b1 = w1[2].float().unsqueeze(-1).contiguous()  # [E, I, 1]
        self.b2 = w2[2].float().unsqueeze(-1).contiguous()  # [E, H, 1]
        self.ab_strides1 = torch.full((E,), self.H, device=dev, dtype=torch.int64)
        self.c_strides1 = torch.full((E,), self.I, device=dev, dtype=torch.int64)
        self.ab_strides2 = torch.full((E,), self.I, device=dev, dtype=torch.int64)
        self.c_strides2 = torch.full((E,), self.H, device=dev, dtype=torch.int64)

    def _buf(self, key, shape):
        b = Fp8DenseMoE._scratch.get(key)
        dev = self.w1[1].device
        if b is None or b.numel() < int(torch.Size(shape).numel()) or b.device != dev:
            b = torch.empty(int(torch.Size(shape).numel()), dtype=torch.float8_e4m3fn, device=dev)
            Fp8DenseMoE._scratch[key] = b
        return b[: int(torch.Size(shape).numel())].view(shape)

    def decode(self):
        if self.mq is not None:
            w8_1 = decode_marlin_to_fp8(self.mq[0], self.w1[1], self.I, self.H,
                                        self._buf("w1", (self.E, self.I, self.H)))
            w8_2 = decode_marlin_to_fp8(self.mq[1], self.w2[1], self.H, self.I,
                                        self._buf("w2", (self.E, self.H, self.I)))
            return w8_1, w8_2
        w8_1 = decode_to_fp8(self.w1[0], self.w1[1], self._buf("w1", (self.E, self.I, self.H)))
        w8_2 = decode_to_fp8(self.w2[0], self.w2[1], self._buf("w2", (self.E, self.H, self.I)))
        return w8_1, w8_2

    # ------------------------------------------------------------------ CUTLASS
    def run_cutlass(self, x, topk_weights, topk_ids, w8=None):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_permute, moe_unpermute
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
        w8_1, w8_2 = w8 if w8 is not None else self.decode()
        M, H = x.shape
        topk = topk_ids.shape[1]
        E, I = self.E, self.I
        a1q, a1s = ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
        a1q, a1s, efto, inv_perm, _ = moe_permute(a1q, a1s, topk_ids, E, E, None)
        counts = (efto[1:] - efto[:-1]).to(torch.int32)
        cI, cH = torch.full_like(counts, I), torch.full_like(counts, H)
        # vLLM's cutlass_moe_mm dispatches a swap-AB kernel when the permuted row count is <= 64
        # (same rule as run_cutlass_moe_fp8); it then expects problem sizes as (N, M, K).
        if M * topk <= 64:
            ps1 = torch.stack([cI, counts, cH], 1).contiguous()
            ps2 = torch.stack([cH, counts, cI], 1).contiguous()
        else:
            ps1 = torch.stack([counts, cI, cH], 1).contiguous()
            ps2 = torch.stack([counts, cH, cI], 1).contiguous()
        offs = efto[:-1]
        mm1 = torch.empty((M * topk, I), dtype=x.dtype, device=x.device)
        ops.cutlass_moe_mm(mm1, a1q, w8_1, a1s, self.b1, offs, ps1, self.ab_strides1, self.ab_strides1,
                           self.c_strides1, True, True)
        act = torch.empty_like(mm1)
        apply_moe_activation(MoEActivation.RELU2_NO_MUL, act, mm1)
        a2q, a2s = ops.scaled_fp8_quant(act, None, use_per_token_if_dynamic=True)
        mm2 = torch.empty((M * topk, H), dtype=x.dtype, device=x.device)
        ops.cutlass_moe_mm(mm2, a2q, w8_2, a2s, self.b2, offs, ps2, self.ab_strides2, self.ab_strides2,
                           self.c_strides2, True, True)
        out = torch.empty((M, H), dtype=x.dtype, device=x.device)
        moe_unpermute(out, mm2, topk_weights.float(), inv_perm, efto)
        return out

    # ------------------------------------------------------------------ Triton (vLLM fused_moe)
    def run_triton(self, x, topk_weights, topk_ids, w8=None):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        w8_1, w8_2 = w8 if w8 is not None else self.decode()
        qc = fp8_w8a8_moe_quant_config(w1_scale=self.b1, w2_scale=self.b2,
                                       per_act_token_quant=True, per_out_ch_quant=True)
        return fused_experts(x, w8_1, w8_2, topk_weights.float(), topk_ids,
                             activation=MoEActivation.RELU2_NO_MUL, quant_config=qc)


# ---------------------------------------------------------------------------------------- hybrid
def _marlin_prepare(nv1, nv2):
    """Stock vLLM Marlin NVFP4 (W4A16) weights for the small-M path. nv = (packed u8 [E,N,K/2],
    scale fp8 [E,N,K/16], weight_scale_2 [E]) exactly as in the ModelOpt checkpoint."""
    from types import SimpleNamespace
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    E, I, Hh = nv1[0].shape
    layer = SimpleNamespace(num_experts=E, hidden_size=2 * Hh, intermediate_size_per_partition=I,
                            params_dtype=torch.bfloat16)
    w13, s13, g13, w2, s2, g2 = prepare_nvfp4_moe_layer_for_marlin(
        layer, nv1[0], nv1[1], nv1[2].float().reshape(E), nv2[0], nv2[1], nv2[2].float().reshape(E),
        is_act_and_mul=False)
    ws = marlin_make_workspace_new(nv1[0].device, 4)
    return w13, s13, g13, w2, s2, g2, ws


class HybridNvfp4MoE:
    """Token-count dispatch between two exact NVFP4 consumers of the same checkpoint:

        M <  dense_min_tokens : stock vLLM Marlin NVFP4 W4A16 (bandwidth-bound decode regime)
        M >= dense_min_tokens : NVFP4 -> FP8 decode (bit-exact W8 of the R2 algorithm) + CUTLASS
                                grouped FP8 GEMM, per-token FP8 activations (prefill regime)

    The crossover is measured per model by tests/bench_baselines.py (Super H200: ~2k tokens).
    Memory: with share_codes (default) the decode reads Marlin's repacked codes directly, so the
    extra state is s_rel (fp8, 1 B per 16 weights) + one fp32 row scale, plus the per-layer FP8
    scratch shared by all layers (Super: 2.8 GB).
    """

    def __init__(self, nv1, nv2, dense_min_tokens: int | None = None, share_codes: bool | None = None):
        import os
        from .fmt import prepare_w4a8
        self.thr = int(dense_min_tokens if dense_min_tokens is not None
                       else os.environ.get("SUPER_W4A8_DENSE_MIN_TOKENS", "3072"))
        share = (os.environ.get("SUPER_W4A8_SHARE_CODES", "1") == "1") if share_codes is None else share_codes
        p1, p2 = prepare_w4a8(*nv1), prepare_w4a8(*nv2)
        self.marlin = _marlin_prepare(nv1, nv2)
        if share:  # decode reads Marlin's codes: extra state is only s_rel + row scale
            self.dense = Fp8DenseMoE((None, p1[1], p1[2]), (None, p2[1], p2[2]),
                                     marlin_codes=(self.marlin[0], self.marlin[3]))
        else:
            self.dense = Fp8DenseMoE(p1, p2)
        self.share_codes = share

    def run_marlin(self, x, topk_weights, topk_ids):
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.scalar_type import scalar_types
        w13, s13, g13, w2, s2, g2, ws = self.marlin
        return fused_marlin_moe(x, w13, w2, None, None, s13, s2, topk_weights, topk_ids,
                                scalar_types.float4_e2m1f.id, activation=MoEActivation.RELU2_NO_MUL,
                                global_scale1=g13, global_scale2=g2, workspace=ws)

    def __call__(self, x, topk_weights, topk_ids):
        if x.shape[0] >= self.thr:
            return self.dense.run_cutlass(x, topk_weights, topk_ids)
        return self.run_marlin(x, topk_weights, topk_ids)
