"""Operator-level baseline benchmark on one MoE layer of a real checkpoint (H200).

    python tests/bench_baselines.py --nvfp4 <ckpt> --bf16 <ckpt> --out r.json

Variants, same routed layer, same routing, same inputs:
  A16_MARLIN_NVFP4   vLLM stock Marlin NVFP4 weight-only (BF16 activations)   <- product baseline
  A8_HUMMING_MXFP4   FlashInfer CUTLASS SM90 mixed-input MXFP4 x FP8 (Humming), MXFP4 made from BF16
  A8_OURS_NVFP4      this repo's Triton NVFP4(g16) x FP8 kernel (default tiles)
For each: accuracy vs BF16 dense reference (R0) at M=256, and latency for several M
(CUDA-graph replay = decode-like; eager = prefill-like). Each variant is isolated: a crash in one
is recorded and the others still run. Also records real NVFP4 scale statistics relevant to the
exponent-add (two-base) decode: fraction of S_rel below 2^-5 (slow path) and per-row spread.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from super_nvfp4_w4a8.fmt import prepare_w4a8  # noqa: E402
from super_nvfp4_w4a8.moe import nvfp4_w4a8_moe  # noqa: E402
from super_nvfp4_w4a8.reference import compare, moe_dense_ref, exact_matmul  # noqa: E402
from gpu_operator_check import load_bank, find_moe_layer, stack, act, routing  # noqa: E402

FP8 = torch.float8_e4m3fn


def bench(fn, iters=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return {"median_ms": ts[len(ts) // 2], "p10_ms": ts[len(ts) // 10], "p90_ms": ts[(9 * len(ts)) // 10]}


def graph_bench(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return bench(g.replay)


# ----------------------------------------------------------------------------- variants
def make_marlin(nv1, nv2, E, H, I, dev):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.scalar_type import scalar_types
    layer = SimpleNamespace(num_experts=E, hidden_size=H, intermediate_size_per_partition=I,
                            params_dtype=torch.bfloat16)
    w13, s13, g13, w2, s2, g2 = prepare_nvfp4_moe_layer_for_marlin(
        layer, nv1[0], nv1[1], nv1[2].float(), nv2[0], nv2[1], nv2[2].float(), is_act_and_mul=False)
    ws = marlin_make_workspace_new(dev, 4)
    qid = scalar_types.float4_e2m1f.id

    def run(x, tw, ids):
        return fused_marlin_moe(x, w13, w2, None, None, s13, s2, tw, ids, qid,
                                activation=MoEActivation.RELU2_NO_MUL, global_scale1=g13,
                                global_scale2=g2, workspace=ws)
    return run


def make_humming_mxfp4(W1, W2, dev, tune=False):
    from flashinfer.fused_moe import ReLU2, cutlass_fused_moe
    from flashinfer.fused_moe.prepare import (_quantize_mxfp4_linear,
                                              preprocess_moe_weights_for_sm90_mixed_gemm_humming)

    def q(W):
        ps, ss = [], []
        for e in range(W.shape[0]):
            p, s = _quantize_mxfp4_linear(W[e].to(dev).contiguous())
            ps.append(p); ss.append(s)
        return torch.stack(ps), torch.stack(ss)
    q1, s1 = q(W1)
    q2, s2 = q(W2)
    w1, sc1, r1 = preprocess_moe_weights_for_sm90_mixed_gemm_humming(q1, s1)
    w2, sc2, r2 = preprocess_moe_weights_for_sm90_mixed_gemm_humming(q2, s2)
    one = torch.ones((), device=dev, dtype=torch.float32)
    scales = [sc1.view(torch.int32), r1 * 64.0, one, sc2.view(torch.int32), r2 * 64.0]

    def run(x, tw, ids):
        out = torch.empty_like(x, dtype=torch.bfloat16)
        cutlass_fused_moe(x, ids.int().contiguous(), tw.contiguous(), w1, w2, output_dtype=torch.bfloat16,
                          quant_scales=scales, activation_type=ReLU2().type, use_w4_group_scaling=True,
                          use_wfp4afp8_humming=True, use_fused_finalize=True, enable_pdl=False,
                          tune_max_num_tokens=8192, output=out)
        return out
    if tune:  # what vLLM does at warm-up: FlashInfer autotuner picks tactics per token bucket
        from flashinfer.autotuner import autotune
        gen = torch.Generator(device=dev).manual_seed(7)
        E, top_k = W1.shape[0], 22
        with autotune(True):
            for M in (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
                x = act(M, W2.shape[1], dev, gen)
                tw, ids = routing(M, E, top_k, dev, gen=gen)
                run(x, tw, ids)
        torch.cuda.synchronize()
    return run


def make_ours(nv1, nv2):
    p1, p2 = prepare_w4a8(*nv1), prepare_w4a8(*nv2)

    def run(x, tw, ids):
        return nvfp4_w4a8_moe(x, tw, ids, p1, p2)
    return run


_DENSE = {}


def _mismatch_diag(got, ref, packed, s_rel, n=8):
    """Where a decode differs from the exact reference: count + first positions with operands."""
    g, r = got.view(torch.uint8).reshape(-1), ref.view(torch.uint8).reshape(-1)
    bad = torch.nonzero(g != r).reshape(-1)
    out = {"n_mismatch": int(bad.numel()), "n_total": int(g.numel())}
    if bad.numel():
        idx = bad[:n]
        out["first"] = idx.tolist()
        out["got"] = g[idx].tolist()
        out["ref"] = r[idx].tolist()
        out["srel_u8"] = s_rel.view(torch.uint8).reshape(-1)[idx // 16].tolist()
        if packed is not None:
            out["packed_byte"] = packed.reshape(-1)[idx // 2].tolist()
        # periodicity hints
        out["mod16_hist"] = torch.bincount(bad % 16, minlength=16).tolist()
        out["frac_rows_hit"] = float(torch.unique(bad // ref.shape[-1]).numel() / (g.numel() / ref.shape[-1]))
    return out


def make_dense(nv1, nv2, backend, include_decode, res_extra):
    from super_nvfp4_w4a8.dense_fp8 import Fp8DenseMoE
    from super_nvfp4_w4a8.fmt import w8_reference
    key = id(nv1)
    if key not in _DENSE:
        from super_nvfp4_w4a8 import dense_fp8 as DF
        d = Fp8DenseMoE(prepare_w4a8(*nv1), prepare_w4a8(*nv2))
        # decode sweep: v1 tile kernel (already verified exact) vs flat uint16-store configs
        ref = (DF.decode_to_fp8(d.w1[0], d.w1[1], impl="tile"), DF.decode_to_fp8(d.w2[0], d.w2[1], impl="tile"))
        sweep = {"tile_v1": {"ms": bench(lambda: (DF.decode_to_fp8(d.w1[0], d.w1[1], ref[0], impl="tile"),
                                                   DF.decode_to_fp8(d.w2[0], d.w2[1], ref[1], impl="tile")))
                             ["median_ms"], "exact": True}}
        o1, o2 = torch.empty_like(ref[0]), torch.empty_like(ref[1])
        for blk, w in DF.DECODE_SWEEP:
            nm = f"flat_b{blk}_w{w}"
            try:
                f = (lambda blk=blk, w=w: (DF.decode_to_fp8(d.w1[0], d.w1[1], o1, impl="flat", block=blk, num_warps=w),
                                           DF.decode_to_fp8(d.w2[0], d.w2[1], o2, impl="flat", block=blk, num_warps=w)))
                o1.zero_(); o2.zero_(); f(); torch.cuda.synchronize()
                ex = bool(torch.equal(o1.view(torch.uint8), ref[0].view(torch.uint8)) and
                          torch.equal(o2.view(torch.uint8), ref[1].view(torch.uint8)))
                sweep[nm] = {"ms": bench(f)["median_ms"], "exact": ex}
                if not ex and "diag" not in res_extra:
                    res_extra["diag"] = _mismatch_diag(o1, ref[0], d.w1[0], d.w1[1])
            except Exception as e:  # noqa: BLE001
                sweep[nm] = {"err": repr(e)[:300]}
        # decode straight from Marlin's repacked codes (shared-code hybrid)
        try:
            from super_nvfp4_w4a8.dense_fp8 import _marlin_prepare
            mq = _marlin_prepare(nv1, nv2)
            q1, q2 = mq[0], mq[3]
            res_extra["marlin_code_shapes"] = [list(q1.shape), list(q2.shape), str(q1.dtype)]
            for bn, bc, w, cb in DF.MARLIN_DECODE_SWEEP:
                nm = f"marlin_bn{bn}_bc{bc}_w{w}_c{int(cb)}"
                try:
                    f = (lambda bn=bn, bc=bc, w=w, cb=cb: (
                        DF.decode_marlin_to_fp8(q1, d.w1[1], d.I, d.H, o1, block_n=bn, block_c=bc, num_warps=w, combine=cb),
                        DF.decode_marlin_to_fp8(q2, d.w2[1], d.H, d.I, o2, block_n=bn, block_c=bc, num_warps=w, combine=cb)))
                    o1.zero_(); o2.zero_(); f(); torch.cuda.synchronize()
                    ex = bool(torch.equal(o1.view(torch.uint8), ref[0].view(torch.uint8)) and
                              torch.equal(o2.view(torch.uint8), ref[1].view(torch.uint8)))
                    sweep[nm] = {"ms": bench(f)["median_ms"], "exact": ex}
                    if not ex and "diag_marlin" not in res_extra:
                        res_extra["diag_marlin"] = _mismatch_diag(o1, ref[0], None, d.w1[1])
                except Exception as e:  # noqa: BLE001
                    sweep[nm] = {"err": repr(e)[:300]}
            cfgs = {f"marlin_bn{a}_bc{b}_w{c}_c{int(e)}": dict(impl="lane", block_n=a, block_c=b, num_warps=c,
                                                                 combine=e) for a, b, c, e in DF.MARLIN_DECODE_SWEEP}
            for T, w in DF.MARLIN_TILE_SWEEP:
                nm = f"mtile_T{T}_w{w}"
                cfgs[nm] = dict(impl="tile", T=T, num_warps=w)
                try:
                    f = (lambda T=T, w=w: (
                        DF.decode_marlin_to_fp8(q1, d.w1[1], d.I, d.H, o1, impl="tile", T=T, num_warps=w),
                        DF.decode_marlin_to_fp8(q2, d.w2[1], d.H, d.I, o2, impl="tile", T=T, num_warps=w)))
                    o1.zero_(); o2.zero_(); f(); torch.cuda.synchronize()
                    ex = bool(torch.equal(o1.view(torch.uint8), ref[0].view(torch.uint8)) and
                              torch.equal(o2.view(torch.uint8), ref[1].view(torch.uint8)))
                    sweep[nm] = {"ms": bench(f)["median_ms"], "exact": ex}
                    if not ex and "diag_mtile" not in res_extra:
                        res_extra["diag_mtile"] = _mismatch_diag(o1, ref[0], None, d.w1[1])
                except Exception as e:  # noqa: BLE001
                    sweep[nm] = {"err": repr(e)[:300]}
            del mq, q1, q2
            mbest = min((k for k, v in sweep.items() if v.get("exact") and k in cfgs),
                        key=lambda k: sweep[k]["ms"], default=None)
            if mbest:
                DF.MARLIN_DECODE_CFG.update(cfgs[mbest])
            res_extra["marlin_decode_cfg"] = dict(DF.MARLIN_DECODE_CFG) if mbest else "NO_EXACT_CONFIG"
        except Exception:  # noqa: BLE001
            res_extra["marlin_decode_error"] = traceback.format_exc()[-2000:]
        del ref, o1, o2
        torch.cuda.empty_cache()
        best = min((k for k, v in sweep.items() if v.get("exact") and k.startswith("flat")),
                   key=lambda k: sweep[k]["ms"], default=None)
        if best:
            b_, w_ = best[len("flat_b"):].split("_w")
            DF.DECODE_CFG.update(impl="flat", block=int(b_), num_warps=int(w_))
        else:
            DF.DECODE_CFG.update(impl="tile")
        res_extra["decode_sweep"] = sweep
        res_extra["decode_cfg"] = dict(DF.DECODE_CFG)
        w8 = d.decode()
        torch.cuda.synchronize()
        # bit-exactness of the decode vs the R2 weight operand (sampled experts)
        chk = {}
        for e in sorted({0, d.E // 2, d.E - 1}):
            ref1 = w8_reference(d.w1[0][e:e + 1], d.w1[1][e:e + 1]).to(FP8)[0]
            ref2 = w8_reference(d.w2[0][e:e + 1], d.w2[1][e:e + 1]).to(FP8)[0]
            chk[e] = bool(torch.equal(w8[0][e].view(torch.uint8), ref1.view(torch.uint8)) and
                          torch.equal(w8[1][e].view(torch.uint8), ref2.view(torch.uint8)))
        res_extra["decode_bit_exact"] = chk
        res_extra["decode_ms"] = bench(lambda: d.decode())
        res_extra["scratch_GB"] = (w8[0].numel() + w8[1].numel()) / 1e9
        _DENSE[key] = (d, w8)
    d, w8 = _DENSE[key]
    fn = d.run_cutlass if backend == "cutlass" else d.run_triton

    def run(x, tw, ids):
        return fn(x, tw, ids, None if include_decode else w8)
    return run


def make_hybrid(nv1, nv2, thr, res_extra):
    from super_nvfp4_w4a8.dense_fp8 import HybridNvfp4MoE
    from super_nvfp4_w4a8.fmt import w8_reference
    h = HybridNvfp4MoE(nv1, nv2, dense_min_tokens=thr)
    w8 = h.dense.decode()
    torch.cuda.synchronize()
    p1, p2 = prepare_w4a8(*nv1), prepare_w4a8(*nv2)
    chk = {}
    for e in sorted({0, h.dense.E // 2, h.dense.E - 1}):
        r1 = w8_reference(p1[0][e:e + 1], p1[1][e:e + 1]).to(FP8)[0]
        r2 = w8_reference(p2[0][e:e + 1], p2[1][e:e + 1]).to(FP8)[0]
        chk[e] = bool(torch.equal(w8[0][e].view(torch.uint8), r1.view(torch.uint8)) and
                      torch.equal(w8[1][e].view(torch.uint8), r2.view(torch.uint8)))
    res_extra["hybrid_share_codes"] = h.share_codes
    res_extra["hybrid_flushed_scales"] = h.flushed_scales
    # whole-bank bit-exactness vs the verified packed-layout decode (all experts, both matrices)
    from super_nvfp4_w4a8 import dense_fp8 as DF
    f1 = DF.decode_to_fp8(p1[0], p1[1], impl="tile")
    full = bool(torch.equal(f1.view(torch.uint8), w8[0].view(torch.uint8)))
    del f1
    f2 = DF.decode_to_fp8(p2[0], p2[1], impl="tile")
    full = full and bool(torch.equal(f2.view(torch.uint8), w8[1].view(torch.uint8)))
    if not full and "diag_hybrid" not in res_extra:
        res_extra["diag_hybrid"] = _mismatch_diag(w8[1], f2, None, p2[1])
    del f2
    res_extra["hybrid_decode_full_bank_exact"] = full
    res_extra["hybrid_decode_bit_exact"] = chk
    res_extra["hybrid_decode_ms"] = bench(lambda: h.dense.decode())
    try:
        from super_nvfp4_w4a8.cuda_ops import marlin_decode_to_fp8
        d = h.dense
        t1 = DF.decode_marlin_ms_to_fp8(d.mq[0], d.ms[0], d.w1[1], d.I, d.H)
        t2 = DF.decode_marlin_ms_to_fp8(d.mq[1], d.ms[1], d.w2[1], d.H, d.I)
        cu = {}
        for T in (2, 4, 8, 16):
            for wp in (0,):
                c1 = marlin_decode_to_fp8(d.mq[0], d.ms[0], d.w1[1], d.I, d.H, T=T, warps=wp)
                c2 = marlin_decode_to_fp8(d.mq[1], d.ms[1], d.w2[1], d.H, d.I, T=T, warps=wp)
                ex = bool(torch.equal(c1.view(torch.uint8), t1.view(torch.uint8)) and
                          torch.equal(c2.view(torch.uint8), t2.view(torch.uint8)))
                ms_ = bench(lambda: (marlin_decode_to_fp8(d.mq[0], d.ms[0], d.w1[1], d.I, d.H, out=c1, T=T, warps=wp),
                                     marlin_decode_to_fp8(d.mq[1], d.ms[1], d.w2[1], d.H, d.I, out=c2, T=T, warps=wp)))
                cu[f"T{T}_w{wp}"] = {"exact_vs_triton": ex, "ms": ms_["median_ms"]}
                del c1, c2
        res_extra["cuda_decode"] = cu
        del t1, t2
    except Exception:  # noqa: BLE001
        res_extra["cuda_decode_error"] = traceback.format_exc()[-2500:]
    res_extra["hybrid_thr"] = h.thr

    def run(x, tw, ids):
        return h(x, tw, ids)
    return run


def make_fused_marlin(nv1, nv2, cfg):
    """marlin_w4a8.py: fused NVFP4xFP8 GEMM reading the Marlin tensors; cfg 'bm_nt_T_gn_w_st' or ''."""
    from super_nvfp4_w4a8.dense_fp8 import HybridNvfp4MoE
    from super_nvfp4_w4a8.marlin_w4a8 import fused_marlin_w4a8_moe
    h = HybridNvfp4MoE(nv1, nv2, dense_min_tokens=1 << 30, share_codes=True)
    c = None
    if cfg:
        k = ("BLOCK_M", "NT", "T", "GROUP_N", "num_warps", "num_stages")
        c = dict(zip(k, (int(v) for v in cfg.split("_"))))

    def run(x, tw, ids):
        return fused_marlin_w4a8_moe(h.dense, x, tw, ids, c, c)
    run.dense = h.dense
    return run


def make_hybrid_gemm(nv1, nv2, gemm, thr=1):
    from super_nvfp4_w4a8.dense_fp8 import HybridNvfp4MoE
    h = HybridNvfp4MoE(nv1, nv2, dense_min_tokens=thr)
    h.gemm = gemm

    def run(x, tw, ids):
        return h(x, tw, ids)
    return run


def make_small_cuda(nv1, nv2, nb):
    from super_nvfp4_w4a8.dense_fp8 import HybridNvfp4MoE
    from super_nvfp4_w4a8.cuda_ops import small_w4a8_moe
    h = HybridNvfp4MoE(nv1, nv2, dense_min_tokens=1 << 30, share_codes=True)

    def run(x, tw, ids):
        return small_w4a8_moe(h.dense, x, tw, ids, nb=nb or None)
    return run


def scale_stats(nv):
    """Real-checkpoint S_rel statistics for the exponent-add decode."""
    _, s_rel, _ = prepare_w4a8(*nv)
    s = s_rel.float()
    nz = s > 0
    out = {"groups": s.numel(), "frac_zero": 1 - nz.float().mean().item(),
           "frac_below_2^-5": ((s < 2 ** -5) & nz).float().mean().item(),
           "frac_below_2^-6": ((s < 2 ** -6) & nz).float().mean().item()}
    smax = s.amax(-1)
    smin = torch.where(nz, s, torch.full_like(s, float("inf"))).amin(-1)
    spread = torch.log2(smax / smin)[torch.isfinite(smin) & (smax > 0)]
    out["row_log2_spread_p50"] = spread.median().item()
    out["row_log2_spread_p99"] = torch.quantile(spread.float()[:1_000_000], 0.99).item()
    out["row_log2_spread_max"] = spread.max().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvfp4", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ms", default="1,8,32,128,512,2048,8192,16384")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"device": torch.cuda.get_device_name(), "ckpt": a.nvfp4, "variants": {}, "errors": {}}

    t_start = time.time()

    def dump(msg=None):
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        if msg:
            print(f"BENCH_PROGRESS t={time.time() - t_start:.0f}s {msg}", flush=True)

    layer = find_moe_layer(a.nvfp4)
    cfg = json.load(open(os.path.join(a.nvfp4, "config.json")))
    top_k = cfg.get("num_experts_per_tok", 22)
    bank = load_bank(a.nvfp4, layer, 512)
    E = 1 + max(e for e, _, _ in bank)
    nv1 = tuple(stack(bank, E, "up_proj", f).to(dev) for f in ("weight", "weight_scale", "weight_scale_2"))
    nv2 = tuple(stack(bank, E, "down_proj", f).to(dev) for f in ("weight", "weight_scale", "weight_scale_2"))
    nv1 = (nv1[0], nv1[1].view(FP8) if nv1[1].dtype != FP8 else nv1[1], nv1[2].reshape(E))
    nv2 = (nv2[0], nv2[1].view(FP8) if nv2[1].dtype != FP8 else nv2[1], nv2[2].reshape(E))
    I, H = nv1[0].shape[1], nv2[0].shape[1]
    bb = load_bank(a.bf16, layer, E)
    W1 = stack(bb, E, "up_proj", "weight")
    W2 = stack(bb, E, "down_proj", "weight")
    res.update(layer=layer, E=E, top_k=top_k, H=H, I=I)
    print(f"BENCH_PROGRESS t={time.time() - t_start:.0f}s banks loaded E={E} H={H} I={I} "
          f"gpu_mem={torch.cuda.memory_allocated() / 2**30:.1f}GiB", flush=True)
    try:
        res["scale_stats"] = {"fc1": scale_stats(nv1), "fc2": scale_stats(nv2)}
    except Exception:  # noqa: BLE001
        res["errors"]["scale_stats"] = traceback.format_exc()[-2000:]
    dump()

    gen = torch.Generator(device=dev).manual_seed(4321)
    M0 = 256
    x0 = act(M0, H, dev, gen)
    tw0, ids0 = routing(M0, E, top_k, dev, gen=gen)
    with exact_matmul():
        r0 = moe_dense_ref(x0, tw0, ids0, W1.to(dev), W2.to(dev), acc_dtype=torch.float32)

    if os.environ.get("BENCH_PROBE_MARLIN") == "1":
        try:
            from probe_marlin_layout import probe
            t0 = time.time()
            pr = probe(I, H, dev)
            torch.save({k: v.pop("inv", None) for k, v in pr.items()}, a.out + ".marlin_inv.pt")
            pr["probe_s"] = time.time() - t0
            res["marlin_probe"] = pr
        except Exception:  # noqa: BLE001
            res["errors"]["marlin_probe"] = traceback.format_exc()[-3000:]
        dump("marlin probe done")

    res["dense_decode"] = {}
    only = os.environ.get("BENCH_VARIANTS")
    builders = [("A16_MARLIN_NVFP4", lambda: make_marlin(nv1, nv2, E, H, I, dev)),
                ("A8_DEQ_CUTLASS_NVFP4", lambda: make_dense(nv1, nv2, "cutlass", True, res["dense_decode"])),
                ("A8_DEQ_CUTLASS_gemm_only", lambda: make_dense(nv1, nv2, "cutlass", False, res["dense_decode"])),
                ("A8_DEQ_TRITON_NVFP4", lambda: make_dense(nv1, nv2, "triton", True, res["dense_decode"])),
                ("A8_DEQ_TRITON_gemm_only", lambda: make_dense(nv1, nv2, "triton", False, res["dense_decode"])),
                ("A8_OURS_NVFP4", lambda: make_ours(nv1, nv2)),
                ("HYBRID_NVFP4", lambda: make_hybrid(nv1, nv2, int(os.environ.get("BENCH_HYBRID_THR", "2048")), res["dense_decode"])),
                ("A8_HUMMING_MXFP4", lambda: make_humming_mxfp4(W1, W2, dev)),
                ("A8_HUMMING_MXFP4_TUNED", lambda: make_humming_mxfp4(W1, W2, dev, tune=True)),
                ("A8_FMARLIN", lambda: make_fused_marlin(nv1, nv2, "")),
                ("HYB_CUTLASS", lambda: make_hybrid_gemm(nv1, nv2, "cutlass")),
                ("HYB_TORCH", lambda: make_hybrid_gemm(nv1, nv2, "torch")),
                ("HYB_TORCH_FAST", lambda: make_hybrid_gemm(nv1, nv2, "torch_fast")),
                ("A8_SMALL_CUDA", lambda: make_small_cuda(nv1, nv2, 0)),
                ("A8_SMALL_CUDA_nb1", lambda: make_small_cuda(nv1, nv2, 1)),
                ("A8_SMALL_CUDA_nb2", lambda: make_small_cuda(nv1, nv2, 2)),
                ("A8_SMALL_CUDA_nb4", lambda: make_small_cuda(nv1, nv2, 4))]
    for cfg in os.environ.get("BENCH_FM_CFGS", "").split(","):
        if cfg:
            builders.append((f"A8_FMARLIN_{cfg}", lambda cfg=cfg: make_fused_marlin(nv1, nv2, cfg)))
    if only:
        builders = [b for b in builders if b[0] in only.split(",") or
                    (b[0].startswith("A8_FMARLIN_") and "A8_FMARLIN_*" in only.split(","))]
    # R2 (exact W4A8 algorithm of the fused kernel) for DUT-R2 of the dense variants
    from gpu_operator_check import run_ref_r2
    with exact_matmul():
        r2 = run_ref_r2(x0, tw0, ids0, prepare_w4a8(*nv1), prepare_w4a8(*nv2))
    for name, build in builders:
        v = {"timing_eager": {}, "timing_graph": {}}
        res["variants"][name] = v
        try:
            t0 = time.time()
            run = build()
            torch.cuda.synchronize()
            v["prepare_s"] = time.time() - t0
            y0 = run(x0, tw0, ids0)
            torch.cuda.synchronize()
            v["acc_vs_R0_M256"] = compare(y0.float(), r0.float())
            v["acc_vs_R2_M256"] = compare(y0.float(), r2.float())
            dump(f"{name} built in {v['prepare_s']:.1f}s, accuracy done")
            if os.environ.get("BENCH_KPROF"):
                v["kprof"] = {}
                for M in (int(m) for m in os.environ["BENCH_KPROF"].split(",")):
                    x = act(M, H, dev, gen)
                    tw, ids = routing(M, E, top_k, dev, gen=gen)
                    for _ in range(3):
                        run(x, tw, ids)
                    torch.cuda.synchronize()
                    from torch.profiler import profile, ProfilerActivity
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(5):
                            run(x, tw, ids)
                        torch.cuda.synchronize()
                    dt = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)  # noqa: E731
                    rows = [(e.key, dt(e) / 5 / 1e3, e.count // 5) for e in prof.key_averages() if dt(e) > 0]
                    rows.sort(key=lambda r: -r[1])
                    v["kprof"][M] = [{"k": k[:160], "ms": round(t, 4), "n": n} for k, t, n in rows[:12]]
                dump(f"{name} kprof {json.dumps(v['kprof'])[:1500]}")
            for M in sorted((int(m) for m in a.Ms.split(",")), reverse=True):  # large M first
                x = act(M, H, dev, gen)
                tw, ids = routing(M, E, top_k, dev, gen=gen)
                try:
                    v["timing_eager"][M] = bench(lambda: run(x, tw, ids))
                except Exception as e:  # noqa: BLE001
                    v["timing_eager"][M] = f"err {e!r}"[:300]
                if M <= 512:
                    try:
                        v["timing_graph"][M] = graph_bench(lambda: run(x, tw, ids))
                    except Exception as e:  # noqa: BLE001
                        v["timing_graph"][M] = f"err {e!r}"[:300]
                dump(f"{name} M={M} eager={v['timing_eager'][M]}")
        except Exception:  # noqa: BLE001
            res["errors"][name] = traceback.format_exc()[-3000:]
            dump()
        torch.cuda.empty_cache()
    res["status"] = "DONE"
    dump()
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
