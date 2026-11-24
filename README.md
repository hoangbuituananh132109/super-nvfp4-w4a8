# super-nvfp4-w4a8

**NVFP4 (block 16, E4M3 scales) weight × FP8 activation MoE for NVIDIA Nemotron-3 Super 120B
on H200 (SM90), served through vLLM.**

> Status (2026-11-24): **research prototype.** Operator-level results on real checkpoint layers
> are below; a full-model vLLM serve benchmark of the hybrid path has **not** been run yet.
> Not production-ready.

## Why

- MXFP4 (block 32, E8M0 scales) W4A8 is available on Hopper, but its weight error is clearly
  higher than NVFP4 (block 16, E4M3 scales) on this model.
- vLLM 0.33 only offers NVFP4 as weight-only W4A16 on Hopper (Marlin). It explicitly rejects
  `NVFP4 weight + FP8 activation`, so no stock path exists.
- Goal: NVFP4 accuracy with W4A8 speed, i.e. **faster than Marlin NVFP4 W4A16 and at least as
  fast as MXFP4 W4A8**, using the unmodified ModelOpt NVFP4 checkpoint.

## Numerics

`W[e,n,k] = G[e] · S[e,n,k/16] · e2m1(code)`. An FP8 tensor-core operand is built per weight as
`W8 = fp8_rne(e2m1 · S_rel)`, with `S_rel = S / r`. Here `r[e,n]` is a power of two chosen per row
so that `6·max S/r ≤ 448`, and `B[e,n] = G·r` is applied in the epilogue. The product is exact in
fp16, so the **only** added weight error is that one RNE rounding (`e2m1·S` needs up to
6 significant bits, E4M3 has 4).

Error is decomposed against exact references, using real layer 1, M=256, rel-L2:

- R0: BF16 weights.
- R1: NVFP4 A16.
- R2: exact W4A8 algorithm (float64).
- DUT: the kernel.

| Super 120B layer 1 (E512, top-22, H1024, I2688) | rel-L2 vs BF16 |
|---|---:|
| Marlin NVFP4 W4A16 (stock vLLM) | 4.55% |
| **NVFP4 W4A8, this repo** (fused kernel / dense path) | 5.05% / 5.18% |
| MXFP4 W4A8 (FlashInfer CUTLASS SM90 mixed-input) | 6.19% |

Fused-kernel DUT−R2 is 0.42% on both Super and Lightning 30B. That kernel promotes the FP8
WGMMA accumulator into FP32 every BLOCK_K.

## Kernels / paths

1. **Fused Triton W4A8 MoE** (`kernels.py`, `moe.py`). Decodes FP4 in registers and runs
   FP8 WGMMA. It is correct and batch-invariant, but slow (layout round-trip every K step).
   Kept as the reference DUT.
2. **Hybrid dispatcher** (`dense_fp8.py: HybridNvfp4MoE`), the current fast path:
   - `M < 3072` tokens (decode, bandwidth-bound): stock vLLM **Marlin NVFP4 W4A16**.
   - `M ≥ 3072` tokens (prefill): decode NVFP4 → FP8 for the layer into a shared scratch, then
     run **CUTLASS grouped FP8 GEMM** (`vllm ops.cutlass_moe_mm`) with per-token FP8 activations
     and per-channel weight scales.
   - The decode reads **Marlin's repacked codes directly**, so the 4-bit weights are stored once.
     The Marlin nibble permutation was measured on GPU (`tests/probe_marlin_layout.py`) and has
     a closed form. It is a 64n × 16k tile of 128 int32 words,
     `word = j + 4kq + 16r`, `slot = b0 + 2b1 + 4b2`,
     `n = 16j + 8b1 + r`, `k = 8b0 + 2kq + b2`.
     So the decode is a coalesced tile load plus a register permute. It is bit-exact against the
     R2 weight operand: 1.61 ms for both matrices of a Super layer.

## Operator results (1× H200, real Super layer, ms)

| tokens M | Marlin NVFP4 A16 | MXFP4 A8 (Humming) | **Hybrid NVFP4 W4A8** |
|---:|---:|---:|---:|
| 1 (CUDA graph) | 0.056 | 0.072 | **0.056** |
| 512 | 1.10 | 1.24 | 1.12 |
| 2048 | 3.22 | 3.49 | 3.47 (Marlin side at the default threshold) |
| 4096 | — | — | **4.39** |
| 8192 | 9.73 | 12.61 | **6.94** |
| 16384 | 18.6 | — | **11.70** |

On this layer, prefill at M = 8192 is 29% faster than Marlin A16 and 45% faster than MXFP4 A8.
Decode matches Marlin, which is the bandwidth-bound ceiling. The FP8 GEMM alone takes 5.3 ms at
M = 8192; the decode adds a fixed ~1.6 ms per layer.

## Layout

| Path | Content |
|---|---|
| `super_nvfp4_w4a8/fmt.py` | checkpoint decoding, `prepare_w4a8` (S_rel, G·r), conversion-error metrics |
| `super_nvfp4_w4a8/kernels.py` | fused Triton W4A8 grouped GEMM (swap-AB), ReLU² + row-wise FP8 quant |
| `super_nvfp4_w4a8/moe.py` | token→expert alignment (pure torch, graph-safe), chunking, tile tables |
| `super_nvfp4_w4a8/dense_fp8.py` | NVFP4→FP8 decode kernels (plain and Marlin-layout), CUTLASS / Triton FP8 MoE, hybrid dispatcher |
| `super_nvfp4_w4a8/reference.py` | exact R0/R1/R2 references and comparison metrics |
| `super_nvfp4_w4a8/vllm_plugin.py` | `--quantization super_nvfp4_w4a8`; wraps the stock ModelOpt config, replaces routed experts only |
| `tests/` | CPU unit tests, GPU operator gate, baseline benchmark, Marlin layout probe |
| `bench/` | tile tuner, numpy numerics simulation |
| `modal_runs/` | Modal launcher with CPU gate (pytest, AOT `sm_90a` compile, `ptxas` spill check) bound to source hashes; GPU stages `op` / `bench` / `serve` |

## Use in vLLM

```bash
SUPER_W4A8_MODE=hybrid vllm serve <nvfp4-checkpoint> --quantization super_nvfp4_w4a8 \
  --trust-remote-code --tensor-parallel-size 1 --kv-cache-dtype fp8 --enable-chunked-prefill
```

`SUPER_W4A8_DENSE_MIN_TOKENS` sets the switch point (default 3072). The default mode is
`fused` (path 1). Router, shared expert, attention, Mamba and other linears stay on stock vLLM.

## Running the gates on Modal

```bash
export MODAL_PROFILE=<profile> W4A8_VOLUME=<volume with checkpoints>
modal run modal_runs/run_w4a8.py --stage cpu     # CPU only: tests, plugin import, AOT compile + spill check
modal run modal_runs/run_w4a8.py --stage op      # 1×H200 operator gate (R0/R1/R2/DUT on real layers)
modal run modal_runs/run_w4a8.py --stage bench   # 1×H200 baselines: Marlin A16, MXFP4 A8, ours
```

GPU stages refuse to start unless the CPU gate passed for the exact current source hashes.

## Known limitations / next steps

- No full-model serve benchmark of the hybrid path yet. Still to check: end-to-end throughput,
  CUDA-graph capture inside vLLM, chunked-prefill interaction, accuracy evals.
- Accuracy cost of FP8 activations vs A16: +0.6 pp rel-L2 on the layer test. Model-level
  evals pending.
- TP1 / EP1 only. Non-gated ReLU² experts without bias (the Super architecture). SM89+.
- A flat uint16-store decode variant was faster but not bit-exact on H200 (Triton codegen issue
  under investigation), so it is not used.
