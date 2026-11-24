"""Triton kernels: NVFP4(group16, E4M3 scale) weight x FP8 activation grouped MoE GEMM.

Hopper has no FP4 MMA. Each program decodes a [BLOCK_N, BLOCK_K] tile of 4-bit codes
in registers, multiplies by the per-16 relative scale (exact in fp16), rounds once to
E4M3 and feeds an FP8 x FP8 -> FP32 tensor-core dot (WGMMA on SM90). The weights stay
4-bit in HBM; no FP8 copy of the model is materialised.

Swap-AB layout: acc[BLOCK_N, BLOCK_M] = W8[BLOCK_N, BLOCK_K] @ X8^T[BLOCK_K, BLOCK_M].
The register-produced weight tile is the A operand (WGMMA allows A from registers),
so it does not need a round-trip through shared memory, and BLOCK_M (tokens per
expert tile) can be as small as 16 for decode.
"""
import triton
import triton.language as tl


@triton.jit
def _decode_w8(packed, kk, s_rel):
    """packed: uint8 [BN, BK] (byte holding element kk), s_rel: fp8 [BN, BK] -> fp8 [BN, BK]."""
    shift = ((kk % 2) * 4).to(tl.uint8)
    code = (packed >> shift[None, :]) & 0xF
    code16 = code.to(tl.uint16)
    # fp16 bit pattern (sign<<15)|(mag<<9) == e2m1(code) * 2**-14 exactly (incl. 0.5 subnormal)
    bits = ((code16 & 8) << 12) | ((code16 & 7) << 9)
    v = bits.to(tl.float16, bitcast=True) * 16384.0          # exact e2m1 value, |v| <= 6
    prod = v * s_rel.to(tl.float16)                           # exact: <= 6 significant bits (fp16 has 11), <= 448
    return prod.to(tl.float8e4nv)                             # single RNE (cvt.rn.satfinite)


@triton.jit
def nvfp4_w4a8_moe_gemm(
    a_ptr, a_scale_ptr,            # fp8 [num_a_rows, K], fp32 [num_a_rows]
    b_ptr, s_ptr, bscale_ptr,      # uint8 [E, N, K/2], fp8 [E, N, K/16], fp32 [E, N]
    c_ptr, topk_w_ptr,             # out [num_slots, N] (bf16/fp32), fp32 [num_slots]
    sorted_ids_ptr, expert_ids_ptr, num_post_pad_ptr,
    N, K, num_valid_slots,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_se, stride_sn, stride_sk,
    stride_bse, stride_bsn,
    stride_cm, stride_cn,
    A_ROW_DIV: tl.constexpr,       # top_k for FC1 (row = slot // top_k), 1 for FC2
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    PROMOTE_ACC: tl.constexpr = True,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_post_pad = tl.load(num_post_pad_ptr)
    num_pid_m = tl.cdiv(num_post_pad, BLOCK_M)
    if pid >= num_pid_n * num_pid_m:  # grid is sized for the worst-case padding
        return
    # grouped ordering over (n, m): consecutive programs share token tiles
    num_in_group = GROUP_N * num_pid_m
    group_id = pid // num_in_group
    first_n = group_id * GROUP_N
    gsize = tl.minimum(num_pid_n - first_n, GROUP_N)
    pid_n = first_n + ((pid % num_in_group) % gsize)
    pid_m = (pid % num_in_group) // gsize

    offs_tok = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    slot = tl.load(sorted_ids_ptr + offs_tok)
    valid = slot < num_valid_slots
    slot = tl.where(valid, slot, 0)
    arow = slot // A_ROW_DIV

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    c_ptrs = c_ptr + slot[None, :] * stride_cm + offs_n[:, None] * stride_cn
    c_mask = valid[None, :] & n_mask[:, None]
    if e < 0:  # expert not local (expert-parallel map); write zeros like vLLM
        tl.store(c_ptrs, tl.zeros((BLOCK_N, BLOCK_M), dtype=c_ptr.dtype.element_ty), mask=c_mask)
        return

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + arow[None, :] * stride_am + offs_k[:, None] * stride_ak
    b_ptrs = b_ptr + e * stride_be + offs_n[:, None] * stride_bn + (offs_k[None, :] // 2) * stride_bk
    s_ptrs = s_ptr + e * stride_se + offs_n[:, None] * stride_sn + (offs_k[None, :] // 16) * stride_sk

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            aT = tl.load(a_ptrs, mask=valid[None, :], other=0.0)
            bp = tl.load(b_ptrs, mask=n_mask[:, None], other=0)
            sr = tl.load(s_ptrs, mask=n_mask[:, None], other=0.0)
        else:
            k_mask = offs_k < K - k0 * BLOCK_K
            aT = tl.load(a_ptrs, mask=valid[None, :] & k_mask[:, None], other=0.0)
            bp = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0)
            sr = tl.load(s_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w8 = _decode_w8(bp, offs_k, sr)
        if PROMOTE_ACC:
            # SM90 FP8 WGMMA accumulates in reduced precision. Start each BLOCK_K chunk from a
            # fresh zero accumulator and add it into the FP32 accumulator (DeepGEMM-style
            # promotion every BLOCK_K). Triton's max_num_imprecise_acc fails to compile here
            # (register-A operand, Triton 3.8: make_ttgir PassManager failure on H200).
            acc += tl.dot(w8, aT)
        else:
            acc = tl.dot(w8, aT, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        s_ptrs += (BLOCK_K // 16) * stride_sk

    # epilogue: per-row weight scale (G*r) and per-token activation scale, then router weight
    bsc = tl.load(bscale_ptr + e * stride_bse + offs_n * stride_bsn, mask=n_mask, other=0.0)
    asc = tl.load(a_scale_ptr + arow, mask=valid, other=0.0)
    acc = acc * bsc[:, None]
    acc = acc * asc[None, :]
    if MUL_ROUTED_WEIGHT:
        tw = tl.load(topk_w_ptr + slot, mask=valid, other=0.0)
        acc = acc * tw[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def rowwise_fp8_quant(
    x_ptr, q_ptr, s_ptr, K,
    stride_xm, stride_qm,
    APPLY_RELU2: tl.constexpr, BLOCK: tl.constexpr,
):
    """Per-row dynamic FP8: s = amax/448 (amax==0 -> s=1), q = rne(clamp(x * (448/amax))).

    With APPLY_RELU2 the row is first mapped to max(x,0)**2 in fp32 (FC1 output is
    already fully scaled, so the FC1 scale is applied before the square)."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=m, other=0.0).to(tl.float32)
    if APPLY_RELU2:
        x = tl.maximum(x, 0.0)
        x = x * x
    amax = tl.max(tl.abs(x), axis=0)
    nz = amax > 0.0
    safe = tl.where(nz, amax, 1.0)
    inv = tl.where(nz, tl.div_rn(448.0, safe), 1.0)
    scale = tl.where(nz, tl.div_rn(safe, 448.0), 1.0)
    q = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
    tl.store(q_ptr + row * stride_qm + offs, q.to(tl.float8e4nv), mask=m)
    tl.store(s_ptr + row, scale)
