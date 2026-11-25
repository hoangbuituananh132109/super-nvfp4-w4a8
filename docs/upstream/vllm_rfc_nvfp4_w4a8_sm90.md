# [RFC] Memory-neutral NVFP4 x FP8 MoE on Hopper: on-the-fly decode from Marlin tensors

*(Posted as https://github.com/vllm-project/vllm/issues/69876)*

## Motivation

On SM90, NVFP4 MoE checkpoints run through Marlin W4A16. That path is accurate, but prefill is
compute-bound on FP16/BF16 tensor cores. Two other approaches exist, and neither covers the
single-GPU large-MoE case:

- **#38728** converts NVFP4 to FP8 at load time. That doubles the expert weight memory. For a
  120B MoE on one 141 GB H200, the FP8 copy of the experts (about 112 GB) does not fit next to the
  rest of the model and a KV cache.
- **FlashInfer SM90 NVFP4 x FP8 MegaMoE** (flashinfer#6343, #6999) targets expert-parallel MoE
  on NVLink-connected GPUs: EP4/EP8, 256/384 experts, SwiGLU. It does not serve TP1/EP1 replicas
  on GPUs without NVLink.

## Proposal

Add an opt-in SM90 NVFP4 MoE backend that keeps the 4-bit weights exactly once, in Marlin's
repacked layout, and dispatches per forward step by token count M:

- **`M < threshold`** (decode; default 2048): unchanged Marlin W4A16.
- **`M >= threshold`** (prefill), in three steps:
  1. A CUDA kernel decodes the layer's NVFP4 experts to FP8 into one shared per-layer scratch
     (2.8 GB for E512 / H1024 / I2688). It reads **Marlin's packed codes and processed scales
     directly**, using a closed-form Marlin nibble/scale layout. The decoded operand is
     `W8 = fp8_rne(e2m1 * S / r)`, with a power-of-two `r` per row folded into the epilogue scale.
  2. Existing `cutlass_moe_mm` grouped FP8 GEMMs run with per-token dynamic activations.
  3. One fused kernel applies the activation and the per-row FP8 quant between the two GEMMs.

The extra memory is one layer's FP8 scratch. Checkpoint-layout expert tensors are freed after
Marlin repacking.

## Results

Nemotron-3 Super 120B (NVFP4, E512, top-22) and Nemotron-3 Lightning 30B. Setup: 1x H200,
vLLM 0.33, TP1, CUDA graphs, fp8 KV, `max-num-batched-tokens 8192`. Metric: total tok/s from
`vllm bench serve`.

| Super 120B | Marlin W4A16 | **this** | MXFP4 W4A8 (FlashInfer Humming, from BF16) |
|---|---:|---:|---:|
| prefill 8192/1 | 14,697 | **21,754** (+48%) | 21,688 |
| mixed 1024/256 | 9,362 | **11,399** (+22%) | 12,206 |
| decode 128/512 @128 | 3,275 | 3,349 | 3,936 |
| batch-1 TPOT (ms) | 6.41 | 6.49 | 8.19 |

| Lightning 30B | Marlin W4A16 | **this** |
|---|---:|---:|
| prefill | 49,829 | **73,862** (+48%) |
| mixed | 27,279 | **37,236** (+36%) |
| decode | 10,772 | 10,842 |

Accuracy on a real Super layer (M = 256), rel-L2 vs BF16 weights:

- Marlin A16: 4.54%
- this: 5.18% (CUTLASS fast-accumulation); 5.05% with precise accumulation
- MXFP4 W4A8: 6.19%

The decode kernel is bit-exact against a float64 reference of `W8`. QA probes are unchanged, and
the first-token top-20 symmetric KL vs Marlin is 0.015. *(Model-level lm-eval: TODO before
the PR.)*

Operator, Super layer at M = 8192: Marlin 9.64 ms, this 4.76 ms. The 4.76 ms breaks down into
GEMM 2.79, decode 1.14, activation+quant 0.42 and routing 0.3.

## Questions for maintainers

1. Is this acceptable as an opt-in Marlin-family backend, `--moe-backend marlin_fp8_prefill` or
   similar? Or should it extend the #38728 `VLLM_NVFP4_GEMM_BACKEND` switch as its memory-neutral
   mode?
2. Should the CUDA decode kernel live in `csrc/moe/marlin_moe_wna16/` next to Marlin, since it
   depends on Marlin's layout? Or should we ask FlashInfer to host it?
3. Is a configurable token threshold OK, or do you prefer a fixed one or a heuristic based on
   shape and SM count?
4. Do you want gated SwiGLU (`silu_and_mul`) and TP/EP support in the first PR, or is
   non-gated ReLU² (the Nemotron-3 MoE) acceptable first, with SwiGLU as a follow-up?

## Notes found along the way

- Marlin's NVFP4 scale compression (unsigned E5M3 plus one power-of-two factor per layer) flushes
  out-of-range block scales to 0. On Nemotron-3 Lightning that is up to 1.44 M blocks in a
  single layer. It affects the stock W4A16 path as well. Is it worth a warning or a fix?
- `cutlass_moe_mm` uses FP8 fast accumulation. Against an exact W4A8 reference it adds about 0.6%
  rel-L2 compared with promoted accumulation.

Reference implementation (an out-of-tree vLLM plugin, used for the numbers above):
`<repo link>`.
