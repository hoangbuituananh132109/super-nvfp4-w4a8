# super-nvfp4-w4a8

**NVFP4 (block 16, E4M3 scales) weight × FP8 activation MoE for NVIDIA Nemotron-3 Super 120B
on H200 (SM90), served through vLLM.**

> Status (2026-11-26): **research prototype; recommendation: use vLLM's default instead.** On
> Hopper, vLLM (0.33 and main) runs NVFP4 MoE with **Humming W4A16 by default**. Against that
> default, this W4A8 hybrid is only +11% on pure prefill and +6% on mixed load, at parity on
> decode, with slightly higher layer error (FP8 activations). For production, serve the NVFP4
> checkpoint with `--moe-backend auto` (Humming W4A16). Earlier tables in this repo compared
> against Marlin forced with `--moe-backend marlin`, which is **not** vLLM's default on H200.

## Why

- MXFP4 (block 32, E8M0 scales) W4A8 is available on Hopper, but its weight error is clearly
  higher than NVFP4 (block 16, E4M3 scales) on this model.
- vLLM has no `NVFP4 weight + FP8 activation` MoE path on Hopper: Marlin W4A8 was removed, and
  Humming's schema resolver falls back to W4A16 for group-wise weights with 8-bit activations
  (`VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'` fails with
  `No compatible Humming input schema` for group size 16).
- Goal: NVFP4 accuracy with W4A8 speed, using the unmodified ModelOpt NVFP4 checkpoint.

## Full-model serve results (Super 120B, 1× H200, total tok/s)

`modal_runs/serve_bench.py`: `vllm serve` + `vllm bench serve` (random data, `--ignore-eos`),
`max-num-batched-tokens 8192`, fp8 KV cache, CUDA graphs on. Single runs; run-to-run spread is a
few percent.

**vLLM 0.33** (startup log: `Using 'HUMMING' NvFp4 MoE backend`):

| workload | Marlin W4A16 (forced) | **Humming W4A16 (default)** | NVFP4 W4A8 hybrid | MXFP4 W4A8 (FlashInfer, BF16 ckpt) |
|---|---:|---:|---:|---:|
| prefill (8192 in / 1 out, 64 req) | 14,697 | **20,038** | 22,278 (+11%) | 21,688 |
| mixed (1024 / 256, 256 req) | 9,362 | **10,954** | 11,644 (+6%) | 12,206 |
| decode (128 / 512, 128 concurrent) | 3,275 | **3,353** | 3,398 (+1%) | 3,936 |
| batch 1 (512 / 256), TPOT ms | 6.41 | **6.43** | 6.49 | 8.19 |

Percentages are against the Humming default. The Humming and hybrid columns come from the same
run; Marlin and MXFP4 come from an earlier run with the same harness.

**vLLM main** (nightly `c2cc003c`, 2026-11-25): Marlin 14,800 / 9,393 / 3,257 / 454; Humming
19,565 / 10,632 / 3,285 / 450; hybrid plugin 21,973 / 11,245 / 3,302 / 445 (same order of rows).

All variants answer 8/8 arithmetic QA probes. First-token top-20 symmetric KL vs Marlin A16:
hybrid 0.015, MXFP4 0.021.

**GSM8K** (lm-eval-harness 0.4.9, 5-shot, greedy, completions API, full 1319 questions,
`SERVE_EVALS=gsm8k`). Stderr is about ±0.7.

| Super 120B | strict-match | flexible-extract |
|---|---:|---:|
| Marlin NVFP4 W4A16 (forced, 0.33) | 92.80 | 93.40 |
| Humming NVFP4 W4A16 (default, main) | 91.96 | 93.18 |
| **NVFP4 W4A8 hybrid** (0.33) | **93.10** | **93.86** |
| MXFP4 W4A8 (0.33) | 92.27 | 92.95 |

| Lightning 30B (stderr about ±1.0) | strict-match | flexible-extract |
|---|---:|---:|
| Marlin NVFP4 W4A16 (forced, 0.33) | 83.47 | 84.53 |
| Humming NVFP4 W4A16 (default, main) | 82.11 | 83.78 |
| **NVFP4 W4A8 hybrid** (0.33) | **82.49** | **84.15** |

With 128 concurrent 5-shot requests, most steps exceed 2048 tokens, so the FP8 prefill path is
exercised. Decode steps use Marlin.

Nemotron-3 Lightning 30B (E128, top-6, H2688, I1856), vLLM 0.33, total tok/s. Here the baseline
is Marlin **forced**; against the Humming default on vLLM main the hybrid-style FP8 prefill
measured only +1% prefill, +3% mixed:

| workload | Marlin W4A16 (forced) | **NVFP4 W4A8 hybrid** | MXFP4 W4A8 |
|---|---:|---:|---:|
| prefill | 49,829 | **73,862** | not supported* |
| mixed | 27,279 | **37,236** | — |
| decode | 10,772 | **10,842** | — |
| batch 1 | 919 | **979** | — |

\* The FlashInfer SM90 MXFP4×FP8 kernel requires K/32 divisible by 4; Lightning's I = 1856 gives
58, so it fails at load.

Marlin's NVFP4 scale compression (unsigned E5M3, a common power-of-two factor per layer) flushes
scales below its range to 0. On Lightning that is up to 1.4 M blocks in one layer (layer 3). This
affects stock Marlin A16 and the hybrid equally, because the hybrid reads Marlin's scales. Model-level
evals should cover it.

Where the time goes (torch-profiler kernel traces of each phase, `SERVE_PROFILE=1`):

(The profile below was taken on vLLM 0.33 against forced Marlin; Humming closes most of the
prefill MoE gap.)

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
| Marlin NVFP4 W4A16 (`--moe-backend marlin`) | 4.54% |
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

## Quickstart on a GPU box (scripts/)

```bash
bash scripts/install.sh                      # pip install into the vLLM env, check versions, pre-build CUDA ops
MODEL_PATH=/path/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 GPUS=0,1 VARIANT=auto bash scripts/serve.sh   # TP1 replica per GPU, ports 8000, 8001 (VARIANT=w4a8: this plugin)
PORT=8000 bash scripts/smoke_http.sh         # /v1/models, chat (thinking off/on), completions
PORT=8000 TAG=w4a8 TOKENIZER_PATH=$MODEL_PATH bash scripts/bench.sh   # prefill / mixed / decode / batch-1
```

`VARIANT=auto` serves the same checkpoint with vLLM's default (Humming W4A16 on H200), which is
the recommended production setting; `VARIANT=marlin` forces Marlin W4A16.

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
- **Against vLLM's default:** Humming W4A16 is within 0–11% of this path on every workload
  measured, with A16 accuracy, so the W4A8 trade-off is not worth it for production.
- **Batched decode:** this regime stays at Marlin W4A16 speed (see above).
- **Prefill:** remaining ideas are a better-tuned FP8 grouped GEMM (vLLM's runs at about 36% of
  H200 FP8 peak on these shapes), and decoding the next layer's weights on a side stream to hide
  the 1.1 ms decode.
- **Scope:** TP1 / EP1 only, non-gated ReLU² experts without bias (the Super architecture), SM90.
