"""CUDA kernels (JIT-built with torch.utils.cpp_extension) for the NVFP4 -> FP8 path.

marlin_decode: vLLM Marlin NVFP4 codes + processed Marlin scales -> FP8 W8 [E, N, K] row-major,
bit-identical to dense_fp8._marlin_tile_decode_ms_kernel (Triton), without any cross-thread shuffle:

    Marlin 64n x 16k tile = 128 int32 words, word = j + 4kq + 16r, nibble slot = b0 + 2b1 + 4b2,
    n = 16j + 8b1 + r, k = 8b0 + 2kq + b2.
    => nibble slots (s, s+4) of one word are the two k-adjacent elements (2kq+8b0, 2kq+8b0+1) of row
       16j + 8b1 + r, so one cvt.e4m3x2 produces 2 output bytes in order. A lane owning (j, r) reads
       the 4 words kq = 0..3 and holds 16 consecutive k of rows 16j+r and 16j+8+r -> two 16 B stores.

Build: once per container (cached in $TORCH_EXTENSIONS_DIR); the image build pre-compiles it.
"""
from __future__ import annotations

import os

import torch

_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ int marlin_scale_pos(int n) {
  int s = n & 63;
  int cp = (s & 7) * 8 + (s >> 3);
  int d = cp & 3;
  int sw = d == 1 ? 2 : (d == 2 ? 1 : d);
  return (n & ~63) + cp - d + sw;
}

// Marlin processed scale byte (unsigned E5M3 of S*sf*2^7) + per-row shift -> S_rel as fp16 (exact
// value of the fp8 E4M3 RNE rounding of S_rel), identical to the Triton decode.
__device__ __forceinline__ __half srel_from_marlin(uint32_t b, int sft) {
  int ex = int(b >> 3) + (127 - 15) - sft;
  uint32_t bits = (uint32_t(ex) << 23) | ((b & 7u) << 20);
  if (b == 0 || ex < 1) bits = 0;
  float f = __uint_as_float(bits);
  __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
  __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  return __half(h);
}

// two nibbles (bits [4s, 4s+4) and [4s+16, 4s+20) of w) -> fp8x2 of e2m1 * srel, lo byte = slot s
__device__ __forceinline__ uint16_t pair_to_fp8x2(uint32_t w, int s, __half2 sr2) {
  uint32_t x = w >> (4 * s);
  uint32_t hb = ((x & 0x00070007u) << 9) | ((x & 0x00080008u) << 12);   // e2m1 * 2^-14 as fp16x2
  __half2 v = *reinterpret_cast<__half2*>(&hb);
  v = __hmul2(v, __float2half2_rn(16384.f));                             // exact e2m1
  v = __hmul2(v, sr2);                                                   // exact product (<= 6 bits)
  __half2_raw r = v;
  return (uint16_t)__nv_cvt_halfraw2_to_fp8x2(r, __NV_SATFINITE, __NV_E4M3);  // single RNE
}

// One block = one 64-row Marlin n-tile x W consecutive k-tiles (W warps, warp w decodes k-tile
// kt0 + w). Lanes write their two 16 B row pieces to a padded shared tile; the block then stores
// full 16*W-byte row segments (W = 8: 128 B lines) coalesced.
template <int W>
__global__ void __launch_bounds__(32 * W) marlin_decode_kernel(
    const int32_t* __restrict__ q, const uint8_t* __restrict__ ms, const int8_t* __restrict__ shift,
    uint8_t* __restrict__ out, int N, int K) {
  constexpr int ROWB = 16 * W + 16;                 // padded row stride (bytes): conflict-free uint4
  __shared__ __align__(16) uint8_t tile[64 * ROWB];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nt = blockIdx.x;
  const int64_t e = blockIdx.z;
  const int KT = K >> 4;
  const int kt0 = blockIdx.y * W;
  const int kt = kt0 + warp;
  const int j = lane & 3, r = lane >> 2;
  const int rA = 16 * j + r, rB = rA + 8;
  if (kt < KT) {
    const int nA = nt * 64 + rA, nB = nt * 64 + rB;
    const int32_t* qt = q + e * KT * (int64_t)(2 * N) + (int64_t)kt * (2 * N) + nt * 128 + j + 16 * r;
    uint32_t w0 = __ldg(qt), w1 = __ldg(qt + 4), w2 = __ldg(qt + 8), w3 = __ldg(qt + 12);
    const uint8_t* mk = ms + e * KT * (int64_t)N + (int64_t)kt * N;
    __half2 srA = __half2half2(srel_from_marlin(__ldg(mk + marlin_scale_pos(nA)), __ldg(shift + e * N + nA)));
    __half2 srB = __half2half2(srel_from_marlin(__ldg(mk + marlin_scale_pos(nB)), __ldg(shift + e * N + nB)));
    uint4 a, b;
    // row A (b1 = 0): k = 2kq (+1) from slots (0,4); k = 8 + 2kq (+1) from slots (1,5)
    a.x = pair_to_fp8x2(w0, 0, srA) | (uint32_t(pair_to_fp8x2(w1, 0, srA)) << 16);
    a.y = pair_to_fp8x2(w2, 0, srA) | (uint32_t(pair_to_fp8x2(w3, 0, srA)) << 16);
    a.z = pair_to_fp8x2(w0, 1, srA) | (uint32_t(pair_to_fp8x2(w1, 1, srA)) << 16);
    a.w = pair_to_fp8x2(w2, 1, srA) | (uint32_t(pair_to_fp8x2(w3, 1, srA)) << 16);
    // row B (b1 = 1): slots (2,6) and (3,7)
    b.x = pair_to_fp8x2(w0, 2, srB) | (uint32_t(pair_to_fp8x2(w1, 2, srB)) << 16);
    b.y = pair_to_fp8x2(w2, 2, srB) | (uint32_t(pair_to_fp8x2(w3, 2, srB)) << 16);
    b.z = pair_to_fp8x2(w0, 3, srB) | (uint32_t(pair_to_fp8x2(w1, 3, srB)) << 16);
    b.w = pair_to_fp8x2(w2, 3, srB) | (uint32_t(pair_to_fp8x2(w3, 3, srB)) << 16);
    *reinterpret_cast<uint4*>(tile + rA * ROWB + warp * 16) = a;
    *reinterpret_cast<uint4*>(tile + rB * ROWB + warp * 16) = b;
  }
  __syncthreads();
  const int ncols = min(W, KT - kt0);
  uint8_t* ob = out + (e * N + nt * 64) * (int64_t)K + kt0 * 16;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = threadIdx.x + i * 32 * W;       // 64 rows x W uint4
    const int row = idx / W, col = idx % W;
    if (col < ncols)
      *reinterpret_cast<uint4*>(ob + row * (int64_t)K + col * 16) =
          *reinterpret_cast<const uint4*>(tile + row * ROWB + col * 16);
  }
}

void marlin_decode(torch::Tensor q, torch::Tensor ms, torch::Tensor shift, torch::Tensor out,
                   int64_t N, int64_t K, int64_t T, int64_t warps) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kInt32 && q.is_contiguous());
  TORCH_CHECK(ms.scalar_type() == torch::kUInt8 && ms.is_contiguous());
  TORCH_CHECK(shift.scalar_type() == torch::kInt8 && shift.is_contiguous());
  TORCH_CHECK(out.element_size() == 1 && out.is_contiguous());
  TORCH_CHECK(N % 64 == 0 && K % 16 == 0);
  const int E = q.size(0);
  const int KT = K / 16;
  const int W = (int)T;                              // k-tiles (= warps) per block
  dim3 grid(N / 64, (KT + W - 1) / W, E);
  dim3 block(32 * W);
  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
  auto* o = reinterpret_cast<uint8_t*>(out.data_ptr());
  const auto* qp = q.data_ptr<int32_t>();
  const auto* mp = ms.data_ptr<uint8_t>();
  const auto* sp = shift.data_ptr<int8_t>();
  switch (W) {
    case 2: marlin_decode_kernel<2><<<grid, block, 0, st>>>(qp, mp, sp, o, N, K); break;
    case 4: marlin_decode_kernel<4><<<grid, block, 0, st>>>(qp, mp, sp, o, N, K); break;
    case 8: marlin_decode_kernel<8><<<grid, block, 0, st>>>(qp, mp, sp, o, N, K); break;
    case 16: marlin_decode_kernel<16><<<grid, block, 0, st>>>(qp, mp, sp, o, N, K); break;
    default: TORCH_CHECK(false, "T (k-tiles per block) must be 2, 4, 8 or 16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------------
// Small-M fused W4A8 MoE GEMM (decode regime), mma.sync m16n8k32 e4m3, swap-AB.
//   acc[row n, token] = sum_k W8[e, n, k] * X8[token, k]
// Weights come straight from the Marlin tensors. Lane (g = lane/4, t = lane%4) of a warp that owns
// Marlin n-tile nt loads, per Marlin k-tile, the uint4 = words 4t + 16g + {0..3} = word (j, kq=t, r=g)
// for the 4 M-tiles j (rows 64nt + 16j + {g, g+8}). Nibble pairs (0,4),(1,5) of a word are row g at
// k = 2t,2t+1 / 8+2t,9+2t; pairs (2,6),(3,7) are row g+8. So A fragment a0 (row g, mma k 4t..4t+3)
// = [k 2t, 2t+1, 8+2t, 9+2t] of that k-tile: the activations are stored with the same within-16
// permutation  p = 4t + i  <->  k = {2t, 2t+1, 8+2t, 9+2t}[i]  (quant_perm), so B needs no shuffle.
// The 8 Marlin scale bytes a lane needs per k-tile are ms[e, kt, 64nt + 8g .. +7]; byte p belongs
// to row (j, b1) with 2j + b1 = PERM8[p], PERM8 = {0,2,1,3,4,6,5,7} (self-inverse).
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ __half2 srel_fast(uint32_t b, int sft) {
  int e16 = int(b >> 3) - sft;
  uint32_t m = b & 7u;
  unsigned short h;
  if (b == 0) h = 0;
  else if (e16 >= 9 && (e16 < 23 || (e16 == 23 && m < 7))) h = (unsigned short)((e16 << 10) | (m << 7));
  else { __half x = srel_from_marlin(b, sft); h = __half_as_ushort(x); }
  __half hh = __ushort_as_half(h);
  return __half2half2(hh);
}

__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void st_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

template <int NB, bool MUL_W, typename OutT>
__global__ void __launch_bounds__(128) w4a8_small_kernel(
    const uint8_t* __restrict__ x8p, const float* __restrict__ xs, const int32_t* __restrict__ q,
    const uint8_t* __restrict__ ms, const int8_t* __restrict__ shift, const float* __restrict__ bscale,
    OutT* __restrict__ out, const float* __restrict__ tw, const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ npp, int N, int K,
    int num_valid, int a_row_div) {
  constexpr int BT = 8 * NB;
  const int mb = blockIdx.y;
  if (mb * BT >= __ldg(npp)) return;
  const int e = __ldg(expert_ids + mb);
  if (e < 0) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nt = blockIdx.x * 4 + warp;
  if (nt * 64 >= N) return;
  const int g = lane >> 2, t = lane & 3;
  const int KT = K >> 4;
  const uint8_t* xr[NB];
  bool xv[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    int s = __ldg(sorted_ids + mb * BT + b * 8 + g);
    xv[b] = s < num_valid;
    xr[b] = x8p + (int64_t)(xv[b] ? s / a_row_div : 0) * K + 4 * t;
  }
  int sft[8];
#pragma unroll
  for (int u = 0; u < 8; ++u) sft[u] = __ldg(shift + (int64_t)e * N + nt * 64 + 16 * (u >> 1) + 8 * (u & 1) + g);
  const int64_t q_stride = 2 * (int64_t)N;
  const int32_t* qb = q + (int64_t)e * KT * q_stride + nt * 128 + 4 * t + 16 * g;
  const uint8_t* mbp = ms + (int64_t)e * KT * N + nt * 64 + 8 * g;
  float acc[4][NB][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int b = 0; b < NB; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[j][b][c] = 0.f;

  uint4 W0 = __ldg(reinterpret_cast<const uint4*>(qb));
  uint4 W1 = __ldg(reinterpret_cast<const uint4*>(qb + q_stride));
  uint2 S0 = __ldg(reinterpret_cast<const uint2*>(mbp));
  uint2 S1 = __ldg(reinterpret_cast<const uint2*>(mbp + N));
  for (int kt = 0; kt < KT; kt += 2) {
    uint4 nW0 = W0, nW1 = W1;
    uint2 nS0 = S0, nS1 = S1;
    const bool more = kt + 2 < KT;
    if (more) {  // prefetch next step
      nW0 = __ldg(reinterpret_cast<const uint4*>(qb + (kt + 2) * q_stride));
      nW1 = __ldg(reinterpret_cast<const uint4*>(qb + (kt + 3) * q_stride));
      nS0 = __ldg(reinterpret_cast<const uint2*>(mbp + (int64_t)(kt + 2) * N));
      nS1 = __ldg(reinterpret_cast<const uint2*>(mbp + (int64_t)(kt + 3) * N));
    }
    uint32_t bf0[NB], bf1[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      bf0[b] = xv[b] ? __ldg(reinterpret_cast<const uint32_t*>(xr[b] + kt * 16)) : 0u;
      bf1[b] = xv[b] ? __ldg(reinterpret_cast<const uint32_t*>(xr[b] + kt * 16 + 16)) : 0u;
    }
    const uint32_t w0[4] = {W0.x, W0.y, W0.z, W0.w};
    const uint32_t w1[4] = {W1.x, W1.y, W1.z, W1.w};
    const uint64_t s0 = (uint64_t(S0.y) << 32) | S0.x;
    const uint64_t s1 = (uint64_t(S1.y) << 32) | S1.x;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int pA = (2 * j) == 2 ? 1 : ((2 * j) == 6 ? 5 : 2 * j);          // PERM8[2j]
      const int pB = (2 * j + 1) == 1 ? 2 : ((2 * j + 1) == 5 ? 6 : 2 * j + 1);  // PERM8[2j+1]
      const __half2 sA0 = srel_fast((uint32_t)(s0 >> (8 * pA)) & 0xFFu, sft[2 * j]);
      const __half2 sB0 = srel_fast((uint32_t)(s0 >> (8 * pB)) & 0xFFu, sft[2 * j + 1]);
      const __half2 sA1 = srel_fast((uint32_t)(s1 >> (8 * pA)) & 0xFFu, sft[2 * j]);
      const __half2 sB1 = srel_fast((uint32_t)(s1 >> (8 * pB)) & 0xFFu, sft[2 * j + 1]);
      const uint32_t a0 = pair_to_fp8x2(w0[j], 0, sA0) | (uint32_t(pair_to_fp8x2(w0[j], 1, sA0)) << 16);
      const uint32_t a1 = pair_to_fp8x2(w0[j], 2, sB0) | (uint32_t(pair_to_fp8x2(w0[j], 3, sB0)) << 16);
      const uint32_t a2 = pair_to_fp8x2(w1[j], 0, sA1) | (uint32_t(pair_to_fp8x2(w1[j], 1, sA1)) << 16);
      const uint32_t a3 = pair_to_fp8x2(w1[j], 2, sB1) | (uint32_t(pair_to_fp8x2(w1[j], 3, sB1)) << 16);
#pragma unroll
      for (int b = 0; b < NB; ++b) mma_e4m3(acc[j][b], a0, a1, a2, a3, bf0[b], bf1[b]);
    }
    W0 = nW0; W1 = nW1; S0 = nS0; S1 = nS1;
  }
  // epilogue: c0,c1 = row g, tokens 2t, 2t+1; c2,c3 = row g+8
#pragma unroll
  for (int b = 0; b < NB; ++b) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int s = __ldg(sorted_ids + mb * BT + b * 8 + 2 * t + c);
      if (s >= num_valid) continue;
      const float as = __ldg(xs + s / a_row_div) * (MUL_W ? __ldg(tw + s) : 1.f);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int nA = nt * 64 + 16 * j + g;
        const float vA = acc[j][b][c] * __ldg(bscale + (int64_t)e * N + nA) * as;
        const float vB = acc[j][b][2 + c] * __ldg(bscale + (int64_t)e * N + nA + 8) * as;
        st_out(out + (int64_t)s * N + nA, vA);
        st_out(out + (int64_t)s * N + nA + 8, vB);
      }
    }
  }
}

// per-row dynamic FP8 quant (same numerics as moe.rowwise_fp8_quant) + within-16 K permutation
// p = 4t + i <- k = {2t, 2t+1, 8+2t, 9+2t}[i]; optional relu(x)^2 in fp32 first.
template <bool RELU2>
__global__ void quant_perm_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ q,
                                  float* __restrict__ s, int K, int64_t ldx) {
  const int64_t row = blockIdx.x;
  const __nv_bfloat16* xr = x + row * ldx;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = __bfloat162float(xr[k]);
    if (RELU2) { v = fmaxf(v, 0.f); v = v * v; }
    amax = fmaxf(amax, fabsf(v));
  }
  __shared__ float red[32];
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  amax = red[0];
  const bool nz = amax > 0.f;
  const float safe = nz ? amax : 1.f;
  const float inv = nz ? __fdiv_rn(448.f, safe) : 1.f;
  if (threadIdx.x == 0) s[row] = nz ? __fdiv_rn(safe, 448.f) : 1.f;
  for (int p = threadIdx.x; p < K; p += blockDim.x) {
    const int blk = p & ~15, pi = p & 15, tt = pi >> 2, i = pi & 3;
    const int k = blk + ((i < 2) ? (2 * tt + i) : (8 + 2 * tt + (i - 2)));
    float v = __bfloat162float(xr[k]);
    if (RELU2) { v = fmaxf(v, 0.f); v = v * v; }
    v = fminf(fmaxf(__fmul_rn(v, inv), -448.f), 448.f);
    q[row * K + p] = (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  }
}

void quant_perm(torch::Tensor x, torch::Tensor q, torch::Tensor s, bool relu2) {
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.stride(1) == 1);
  const int R = x.size(0), K = x.size(1);
  if (R == 0) return;
  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
  const int th = K >= 2048 ? 512 : 256;
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* qp = reinterpret_cast<uint8_t*>(q.data_ptr());
  if (relu2) quant_perm_kernel<true><<<R, th, 0, st>>>(xp, qp, s.data_ptr<float>(), K, x.stride(0));
  else quant_perm_kernel<false><<<R, th, 0, st>>>(xp, qp, s.data_ptr<float>(), K, x.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w4a8_small_gemm(torch::Tensor x8p, torch::Tensor xs, torch::Tensor q, torch::Tensor ms,
                     torch::Tensor shift, torch::Tensor bscale, torch::Tensor out, torch::Tensor tw,
                     torch::Tensor sorted_ids, torch::Tensor expert_ids, torch::Tensor npp,
                     int64_t N, int64_t K, int64_t num_valid, int64_t a_row_div, bool mul_w, int64_t nb) {
  TORCH_CHECK(K % 32 == 0 && N % 64 == 0, "K % 32, N % 64");
  TORCH_CHECK(sorted_ids.scalar_type() == torch::kInt32 && expert_ids.scalar_type() == torch::kInt32);
  const int BT = 8 * nb;
  const int nblocks = (sorted_ids.size(0) + BT - 1) / BT;
  dim3 grid((N / 64 + 3) / 4, nblocks);
  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
  auto* xp = reinterpret_cast<const uint8_t*>(x8p.data_ptr());
  auto* qp = q.data_ptr<int32_t>();
  auto* mp = reinterpret_cast<const uint8_t*>(ms.data_ptr());
  auto* sp = shift.data_ptr<int8_t>();
#define W4A8_LAUNCH(NB_, MW_, T_)                                                                    \
  w4a8_small_kernel<NB_, MW_, T_><<<grid, 128, 0, st>>>(                                             \
      xp, xs.data_ptr<float>(), qp, mp, sp, bscale.data_ptr<float>(),                                \
      reinterpret_cast<T_*>(out.data_ptr()), tw.data_ptr<float>(), sorted_ids.data_ptr<int32_t>(),   \
      expert_ids.data_ptr<int32_t>(), npp.data_ptr<int32_t>(), N, K, num_valid, a_row_div)
  const bool f32 = out.scalar_type() == torch::kFloat32;
  TORCH_CHECK(f32 || out.scalar_type() == torch::kBFloat16);
  if (nb == 1) {
    if (f32) { if (mul_w) W4A8_LAUNCH(1, true, float); else W4A8_LAUNCH(1, false, float); }
    else { if (mul_w) W4A8_LAUNCH(1, true, __nv_bfloat16); else W4A8_LAUNCH(1, false, __nv_bfloat16); }
  } else if (nb == 2) {
    if (f32) { if (mul_w) W4A8_LAUNCH(2, true, float); else W4A8_LAUNCH(2, false, float); }
    else { if (mul_w) W4A8_LAUNCH(2, true, __nv_bfloat16); else W4A8_LAUNCH(2, false, __nv_bfloat16); }
  } else if (nb == 4) {
    if (f32) { if (mul_w) W4A8_LAUNCH(4, true, float); else W4A8_LAUNCH(4, false, float); }
    else { if (mul_w) W4A8_LAUNCH(4, true, __nv_bfloat16); else W4A8_LAUNCH(4, false, __nv_bfloat16); }
  } else TORCH_CHECK(false, "nb must be 1, 2 or 4");
#undef W4A8_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP = ("void marlin_decode(torch::Tensor q, torch::Tensor ms, torch::Tensor shift, torch::Tensor out, "
        "int64_t N, int64_t K, int64_t T, int64_t warps);\n"
        "void quant_perm(torch::Tensor x, torch::Tensor q, torch::Tensor s, bool relu2);\n"
        "void w4a8_small_gemm(torch::Tensor x8p, torch::Tensor xs, torch::Tensor q, torch::Tensor ms, "
        "torch::Tensor shift, torch::Tensor bscale, torch::Tensor out, torch::Tensor tw, torch::Tensor sorted_ids, "
        "torch::Tensor expert_ids, torch::Tensor npp, int64_t N, int64_t K, int64_t num_valid, int64_t a_row_div, "
        "bool mul_w, int64_t nb);")

_MOD = None
DECODE_CFG = {"T": int(os.environ.get("SUPER_W4A8_CUDA_DECODE_T", "8")),
              "warps": int(os.environ.get("SUPER_W4A8_CUDA_DECODE_WARPS", "4"))}


def module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        # the vLLM image exports a multi-arch list (incl. sm_75, no FP8 mma): build for Hopper only
        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            _MOD = load_inline(name="super_w4a8_cuda_ops_v3", cpp_sources=[_CPP], cuda_sources=[_SRC],
                               functions=["marlin_decode", "quant_perm", "w4a8_small_gemm"], extra_cuda_cflags=["-O3"],
                               verbose=False)
        finally:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _MOD


def available() -> bool:
    try:
        module()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"SUPER_W4A8_WARN CUDA ops unavailable, falling back to Triton decode: {e!r}"[:2000], flush=True)
        return False


def marlin_decode_to_fp8(q, ms, shift, N, K, out=None, T=None, warps=None):
    E = q.shape[0]
    if out is None:
        out = torch.empty((E, N, K), dtype=torch.float8_e4m3fn, device=q.device)
    module().marlin_decode(q, ms.view(torch.uint8), shift, out.view(torch.uint8), N, K,
                           T or DECODE_CFG["T"], warps or DECODE_CFG["warps"])
    return out


def _align(topk_ids, block_m, E):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    sid, eid, npp = moe_align_block_size(topk_ids, block_m, E)
    return sid.int(), eid.int(), npp.int()


def small_cfg(avg):
    nb = int(os.environ.get("SUPER_W4A8_SMALL_NB", "0"))
    if nb:
        return nb
    return 1 if avg <= 8 else (2 if avg <= 24 else 4)


def small_w4a8_moe(dense, x, topk_weights, topk_ids, nb=None):
    """Decode-regime fused W4A8 MoE on the Marlin tensors (cuda w4a8_small_kernel).
    dense = dense_fp8.Fp8DenseMoE with marlin codes/scales (share_codes); numerics = the dense path's
    (W8 = fp8_rne(e2m1 * S_rel), per-token FP8 activations, fp32 relu^2, per-(token,k) FP8)."""
    m = module()
    M, H = x.shape
    top_k = topk_ids.shape[1]
    E, I = dense.E, dense.I
    ns = M * top_k
    nb = nb or small_cfg(ns / E)
    sid, eid, npp = _align(topk_ids, 8 * nb, E)
    tw = topk_weights.reshape(-1).to(torch.float32).contiguous()
    x8 = torch.empty((M, H), dtype=torch.uint8, device=x.device)
    xs = torch.empty((M,), dtype=torch.float32, device=x.device)
    m.quant_perm(x.contiguous(), x8, xs, False)
    h = torch.empty((ns, I), dtype=torch.bfloat16, device=x.device)
    m.w4a8_small_gemm(x8, xs, dense.mq[0], dense.ms[0].view(torch.uint8), dense.w1[1], dense.b1, h, tw,
                      sid, eid, npp, I, H, ns, top_k, False, nb)
    h8 = torch.empty((ns, I), dtype=torch.uint8, device=x.device)
    hs = torch.empty((ns,), dtype=torch.float32, device=x.device)
    m.quant_perm(h, h8, hs, True)
    y = torch.empty((ns, H), dtype=torch.float32, device=x.device)
    m.w4a8_small_gemm(h8, hs, dense.mq[1], dense.ms[1].view(torch.uint8), dense.w2[1], dense.b2, y, tw,
                      sid, eid, npp, H, I, ns, 1, True, nb)
    return y.view(M, top_k, H).sum(dim=1).to(x.dtype)
