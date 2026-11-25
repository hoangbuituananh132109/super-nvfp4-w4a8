# super-nvfp4-w4a8

**NVFP4 (block 16, E4M3 scales) weight × FP8 activation MoE for NVIDIA Nemotron-3 Super 120B
on H200 (SM90), served through vLLM.**

> Status (2026-11-25): **research prototype, full-model serve measured.** On Super 120B (1× H200,
> vLLM 0.33, CUDA graphs) the hybrid path is faster than stock Marlin NVFP4 W4A16 everywhere except
> batch-1 decode (parity), matches MXFP4 W4A8 on prefill, beats it at batch 1, and is still behind it
> on batched decode. GSM8K matches Marlin W4A16 within stderr. Not production-hardened.

## Why

- MXFP4 (block 32, E8M0 scales) W4A8 is available on Hopper, but its weight error is clearly
  higher than NVFP4 (block 16, E4M3 scales) on this model.
- vLLM 0.33 only offers NVFP4 as weight-only W4A16 on Hopper (Marlin). It explicitly rejects
  `NVFP4 weight + FP8 activation`, so no stock path exists.
- Goal: NVFP4 accuracy with W4A8 speed, i.e. **faster than Marlin NVFP4 W4A16 and at least as
  fast as MXFP4 W4A8**, using the unmodified ModelOpt NVFP4 checkpoint.

## Full-model serve results (Super 120B, 1× H200, total tok/s)

`modal_runs/serve_bench.py`: `vllm serve` + `vllm bench serve` (random data, `--ignore-eos`),
`max-num-batched-tokens 8192`, fp8 KV cache, CUDA graphs on.

| workload | Marlin NVFP4 W4A16 | **NVFP4 W4A8 hybrid** | MXFP4 W4A8 (FlashInfer, BF16 ckpt) |
|---|---:|---:|---:|
| prefill (8192 in / 1 out, 64 req) | 14,697 | **21,754** (+48%) | 21,688 |
| mixed (1024 / 256, 256 req) | 9,362 | **11,399** (+22%) | 12,206 |
| decode (128 / 512, 128 concurrent) | 3,275 | **3,349** (+2%) | 3,936 |
| batch 1 (512 / 256), TPOT ms | 6.41 | 6.49 | 8.19 |

All variants answer 8/8 arithmetic QA probes. First-token top-20 symmetric KL vs Marlin A16:
hybrid 0.015, MXFP4 0.021.

**GSM8K** (lm-eval-harness 0.4.9, 5-shot, greedy, completions API, full 1319 questions,
`SERVE_EVALS=gsm8k`). Stderr is about ±0.7.

| Super 120B | strict-match | flexible-extract |
|---|---:|---:|
| Marlin NVFP4 W4A16 | 92.80 | 93.40 |
| **NVFP4 W4A8 hybrid** | **93.10** | **93.86** |
| MXFP4 W4A8 | 92.27 | 92.95 |

| Lightning 30B (stderr about ±1.0) | strict-match | flexible-extract |
|---|---:|---:|
| Marlin NVFP4 W4A16 | 83.47 | 84.53 |
| **NVFP4 W4A8 hybrid** | **82.49** | **84.15** |

With 128 concurrent 5-shot requests, most steps exceed 2048 tokens, so the FP8 prefill path is
exercised. Decode steps use Marlin.

Nemotron-3 Lightning 30B (E128, top-6, H2688, I1856), same harness, total tok/s:

| workload | Marlin NVFP4 W4A16 | **NVFP4 W4A8 hybrid** | MXFP4 W4A8 |
|---|---:|---:|---:|
| prefill | 49,829 | **73,862** (+48%) | not supported* |
| mixed | 27,279 | **37,236** (+36%) | — |
| decode | 10,772 | **10,842** (+1%) | — |
| batch 1 | 919 | **979** (+7%) | — |

\* The FlashInfer SM90 MXFP4×FP8 kernel requires K/32 divisible by 4; Lightning's I = 1856 gives
58, so it fails at load.

Marlin's NVFP4 scale compression (unsigned E5M3, a common power-of-two factor per layer) flushes
scales below its range to 0. On Lightning that is up to 1.4 M blocks in one layer (layer 3). This
affects stock Marlin A16 and the hybrid equally, because the hybrid reads Marlin's scales. Model-level
evals should cover it.

Where the time goes (torch-profiler kernel traces of each phase, `SERVE_PROFILE=1`):

- **Prefill:** the hybrid MoE layer costs 4.9 ms at 8192 tokens, against 3.6 ms for autotuned
  MXFP4. Its non-expert layers, however, are FP8-static in the NVFP4 checkpoint, against BF16 in
  the MXFP4 baseline (about 0.83 s vs 1.17 s per 64k-token prefill). Net result: parity.
- **Batched decode:** this regime uses Marlin W4A16 (bs128: 1.34 s vs 0.73 s for MXFP4 in the
  profiled window). This gap is a **format property**. NVFP4 scales are E4M3 with a mantissa, so
  every weight needs a multiply and an RNE rounding to reach an FP8 operand. MXFP4's E8M0 scale is
  just an exponent add. A fused W4A8 decode-regime kernel was built (Triton and CUDA,
  `cuda_ops.w4a8_small_gemm`) and is numerically exact (0.17% vs R2). It is ALU-bound at about 2×
  Marlin's time, so Marlin W4A16 stays on that side: it is also more accurate there.
- **Batch 1:** MXFP4 is slower here because of its BF16 dense layers.

## Numerics

`W[e,n,k] = G[e] · S[e,n,k/16] · e2m1(code)`. An FP8 tensor-core operand is built per weight as
`W8 = fp8_rne(e2m1 · S_rel)`, with `S_rel = S / r`. Here `r[e,n]` is a power of two chosen per row
so that `6·max S/r ≤ 448`, and `B[e,n] = G·r` is applied in the epilogue. The product is exact in
fp16, so the **only** added weight error is that one RNE rounding.

| Super 120B layer 1 (E512, top-22, H1024, I2688), M = 256 | rel-L2 vs BF16 |
|---|---:|
| Marlin NVFP4 W4A16 (stock vLLM) | 4.54% |
| **NVFP4 W4A8, this repo** (precise acc / CUTLASS fast-acc) | 5.05% / 5.18% |
| MXFP4 W4A8 (FlashInfer CUTLASS SM90 mixed-input) | 6.19% |

vLLM's `cutlass_moe_mm` uses FP8 fast accumulation, which adds 0.6% vs R2. A precise-accumulation
grouped GEMM (`torch._scaled_grouped_mm`, `use_fast_accum=False`) removes it, but costs 3.4× GEMM
time, so it is optional (`SUPER_W4A8_GEMM=torch`).

## Paths

**Hybrid dispatcher** (`dense_fp8.py: HybridNvfp4MoE`, the default):

- `M < 2048` tokens: stock vLLM **Marlin NVFP4 W4A16**.
- `M ≥ 2048` tokens, in three steps:
  1. CUDA kernel `cuda_ops.marlin_decode` decodes NVFP4 → FP8 straight from **Marlin's repacked
     codes and processed scales**, so the 4-bit weights exist only once. It is bit-exact and takes
     1.13 ms per Super layer, both matrices.
  2. **CUTLASS grouped FP8 GEMM** (`vllm ops.cutlass_moe_mm`) runs with per-token activations and
     per-channel weight scales.
  3. A fused ReLU² (fp32) plus per-row FP8 quant feeds the second GEMM.

The decode works because the Marlin nibble layout has a closed form, measured in
`tests/probe_marlin_layout.py`. A 64n × 16k tile is 128 int32 words:
`word = j + 4kq + 16r`, `slot = b0 + 2b1 + 4b2`, `n = 16j + 8b1 + r`, `k = 8b0 + 2kq + b2`.
Nibble slots `(s, s+4)` are k-adjacent, so one `cvt.e4m3x2` emits two output bytes in order.

Experimental, not on the default path:

- **fused Triton W4A8** (`kernels.py`/`moe.py`, `SUPER_W4A8_MODE=fused`);
- **Triton Marlin-layout fused** (`marlin_w4a8.py`);
- **CUDA decode-regime W4A8** (`cuda_ops.small_w4a8_moe`).

All three are exact. The Triton ones are 8–40× slower: Triton round-trips the register-decoded tile
through shared memory every K step.

## Operator results (1× H200, real Super layer, eager ms)

| tokens M | Marlin NVFP4 A16 | MXFP4 A8 (autotuned) | **Hybrid NVFP4 W4A8** |
|---:|---:|---:|---:|
| 1 (CUDA graph) | 0.055 | 0.069 | 0.055 (Marlin) |
| 128 (CUDA graph) | 0.68 | 0.53 | 0.68 (Marlin) |
| 2048 | 3.22 | 1.24 | **2.49** |
| 4096 | 5.04 | 1.99 | **3.09** |
| 8192 | 9.64 | 3.62 | **4.76** |

At 8192 tokens the hybrid splits into GEMM 2.79 ms, decode 1.14 ms, ReLU²+quant 0.42 ms and
routing 0.3 ms.

## Layout

| Path | Content |
|---|---|
| `super_nvfp4_w4a8/fmt.py` | checkpoint decoding, `prepare_w4a8` (S_rel, G·r), conversion-error metrics |
| `super_nvfp4_w4a8/dense_fp8.py` | NVFP4→FP8 decode (Triton), CUTLASS / torch / Triton FP8 MoE, hybrid dispatcher |
| `super_nvfp4_w4a8/cuda_ops.py` | JIT CUDA: Marlin→FP8 decode, permuted FP8 quant, decode-regime W4A8 MMA kernel |
| `super_nvfp4_w4a8/kernels.py`, `moe.py` | fused Triton W4A8 reference kernel, ReLU² + row-wise FP8 quant, alignment |
| `super_nvfp4_w4a8/marlin_w4a8.py` | Triton fused GEMM on Marlin tensors (experimental, slow) |
| `super_nvfp4_w4a8/reference.py` | exact R0/R1/R2 references and comparison metrics |
| `super_nvfp4_w4a8/vllm_plugin.py` | `--quantization super_nvfp4_w4a8`; wraps the stock ModelOpt config, replaces routed experts only |
| `tests/` | CPU unit tests, GPU operator gate, baseline benchmark (`BENCH_KPROF` per-kernel profile), Marlin layout probe |
| `modal_runs/` | Modal launcher: CPU gate (pytest, AOT `sm_90a` compile, ptxas spills, nvcc build) bound to source hashes; GPU stages `op` / `bench` / `serve2` (full vLLM serve, `SERVE_PROFILE=1` for kernel traces) |
| `third_party/super_mxfp4a8/` | MXFP4 W4A8 baseline adapter used for the comparisons (not included in this repo) |

## Use in vLLM

```bash
vllm serve <nvfp4-checkpoint> --quantization super_nvfp4_w4a8 \
  --trust-remote-code --tensor-parallel-size 1 --kv-cache-dtype fp8 --enable-chunked-prefill
```

Environment knobs:

- `SUPER_W4A8_DENSE_MIN_TOKENS` (default 2048; 1536–3072 measured equal on Super).
- `SUPER_W4A8_CUDA_DECODE=0` falls back to the Triton decode.
- `SUPER_W4A8_GEMM=cutlass|torch|torch_fast`.
- `SUPER_W4A8_FUSED_ACT=0` uses the unfused activation.

The CUDA kernels are JIT-built once per container with `nvcc` (CUDA 13 in the vLLM 0.33 image),
about 1 min, cached in `$TORCH_EXTENSIONS_DIR`. Router, shared expert, attention, Mamba and other
linears stay on stock vLLM. The extra memory is one FP8 scratch for one layer (2.8 GB on Super);
checkpoint-layout expert tensors are freed after Marlin repacking.

## Known limitations / next steps

- **Accuracy evals:** GSM8K is done (above). Broader evals (MMLU, long-context, code) are not done yet. Before that, only layer rel-L2, QA
  probes and first-token KL have been measured.
- **Batched decode:** this regime stays at Marlin W4A16 speed (see above).
- **Prefill:** remaining ideas are a better-tuned FP8 grouped GEMM (vLLM's runs at about 36% of
  H200 FP8 peak on these shapes), and decoding the next layer's weights on a side stream to hide
  the 1.1 ms decode.
- **Scope:** TP1 / EP1 only, non-gated ReLU² experts without bias (the Super architecture), SM90.
