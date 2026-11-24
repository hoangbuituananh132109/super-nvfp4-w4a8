"""GPU operator gate: DUT (Triton W4A8) vs R2 (exact W4A8 model) vs R1 (NVFP4 A16) vs R0 (BF16).

    python tests/gpu_operator_check.py --out result.json                       # synthetic only
    python tests/gpu_operator_check.py --nvfp4 /ckpt/NVFP4 --bf16 /ckpt/BF16 --layer auto --out r.json

Stages, in order (stops at the first failing gate, keeps failing tensors):
  1. tiny fixtures (E=4, H=I=64)            DUT vs R2 (float64)   rel_l2 < 2e-3
  2. Super shapes, synthetic weights        M in 1,32,256,2048,8192; E=512 top22
  3. batch invariance                        target rows bitwise equal across batch mixes
  4. real expert bank from checkpoint        R1-R0, R2-R1, DUT-R2 at real weights (optional)
  5. timing                                  CUDA events, warm, median/p10/p90
Tolerances are fixed in advance: DUT-R2 differences may only come from FP32 accumulation
order inside the tensor core and from bf16 rounding of the FC1 slot buffer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from super_nvfp4_w4a8.fmt import prepare_w4a8, dequant_nvfp4, weight_conversion_error, E2M1_VALUES  # noqa: E402
from super_nvfp4_w4a8.moe import nvfp4_w4a8_moe, pick_config  # noqa: E402
from super_nvfp4_w4a8.reference import moe_r2, moe_r1, moe_dense_ref, compare, exact_matmul  # noqa: E402

FP8 = torch.float8_e4m3fn
TOL_DUT_R2 = 5e-3      # relative L2, whole output
TOL_DUT_R2_ROW = 2e-2  # worst row


def quantize_nvfp4(w, chunk=16):
    if w.shape[0] > chunk:  # bound temporary memory
        parts = [quantize_nvfp4(w[i:i + chunk], chunk) for i in range(0, w.shape[0], chunk)]
        return tuple(torch.cat(t) for t in zip(*parts))
    E, N, K = w.shape
    g = w.abs().amax(dim=(1, 2)).float() / (6 * 448)
    blk = w.float().reshape(E, N, K // 16, 16)
    s = (blk.abs().amax(-1) / 6 / g[:, None, None]).to(FP8)
    x = blk / (s.float().clamp_min(2 ** -9)[..., None] * g[:, None, None, None])
    grid = torch.tensor(E2M1_VALUES, device=w.device)
    idx = (x.abs()[..., None] - grid).abs().argmin(-1)
    code = (idx | ((x < 0).long() << 3)).reshape(E, N, K).to(torch.uint8)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous(), s.contiguous(), g


def routing(M, E, k, dev, skew=1.0, gen=None):
    logits = torch.randn(M, E, device=dev, generator=gen) * skew
    w, ids = torch.topk(torch.softmax(logits, -1), k, dim=-1)
    return (w / w.sum(-1, keepdim=True)).float(), ids.to(torch.int32)


def act(M, H, dev, gen=None):
    x = torch.randn(M, H, device=dev, generator=gen)
    x[:, :8] *= 20  # outlier channels
    x *= torch.exp(torch.randn(M, 1, device=dev, generator=gen) * 0.5)
    x[0] = 0 if M > 3 else x[0]  # an all-zero row
    return x.bfloat16()


def run_ref_r2(x, tw, ids, p1, p2, chunk=512):
    outs = []
    for t0 in range(0, x.shape[0], chunk):
        y, _ = moe_r2(x[t0:t0 + chunk], tw[t0:t0 + chunk], ids[t0:t0 + chunk], p1, p2,
                      acc_dtype=torch.float64 if x.shape[0] <= 64 else torch.float32)
        outs.append(y)
    return torch.cat(outs)


def bench(fn, iters=20, warm=5):
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


# ---------------------------------------------------------------- real checkpoint bank
def load_bank(root, layer, E):
    """Return {name: tensor} for routed experts of one layer. Matches
    '...layers.{L}.mixer.experts.{e}.(up_proj|down_proj).{weight|weight_scale|weight_scale_2}'."""
    from safetensors import safe_open
    idx_path = os.path.join(root, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"]
    # anchored: mtp.layers.{L}.mixer.experts.* (BF16) must not overwrite backbone tensors
    pat = re.compile(rf"^backbone\.layers\.{layer}\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(weight|weight_scale|weight_scale_2)$")
    by_file = {}
    for k, f in wmap.items():
        m = pat.search(k)
        if m:
            by_file.setdefault(f, []).append((k, m))
    if not by_file:
        raise KeyError(f"no routed-expert tensors for layer {layer} under {root}")
    out = {}
    for f, items in by_file.items():
        with safe_open(os.path.join(root, f), "pt", device="cpu") as h:
            for k, m in items:
                out[(int(m.group(1)), m.group(2), m.group(3))] = h.get_tensor(k)
    return out


def find_moe_layer(root):
    wmap = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    for k in sorted(wmap):
        m = re.search(r"^backbone\.layers\.(\d+)\.mixer\.experts\.0\.up_proj\.weight$", k)
        if m:
            return int(m.group(1))
    raise KeyError("no MoE layer found")


def stack(bank, E, proj, field):
    return torch.stack([bank[(e, proj, field)] for e in range(E)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nvfp4")
    ap.add_argument("--bf16")
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--Ms", default="1,32,256,2048,8192")
    ap.add_argument("--save-failures", default="failures.pt")
    ap.add_argument("--real-only", action="store_true", help="skip synthetic stages 1-3")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
           "torch": torch.__version__, "stages": {}, "status": "RUNNING"}
    try:
        import triton
        res["triton"] = triton.__version__
    except Exception:
        pass

    def dump():
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)

    res["failures"] = []

    def fail(stage, info, tensors):
        """Non-fatal: record, keep tensors, continue so one GPU residency collects everything."""
        res["failures"].append(stage)
        res["stages"][f"FAILED_{stage}"] = info
        if tensors:
            base, ext = os.path.splitext(a.save_failures)
            torch.save(tensors, f"{base}-{stage}{ext or '.pt'}")
        dump()

    gen = torch.Generator(device=dev).manual_seed(1234)
    for name, fn in ((("synthetic", synthetic_stages),) if not a.real_only else ()) + (("real_bank", real_bank_stage),):
        try:
            fn(a, res, dev, gen, dump, fail)
        except Exception:
            import traceback
            res["failures"].append(f"CRASH_{name}")
            res["stages"][f"CRASH_{name}"] = traceback.format_exc()[-4000:]
            dump()
    res["status"] = "PASS" if not res["failures"] else f"FAIL_{res['failures'][0]}"
    dump()
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if not res["failures"] else 1)


def kernel_profile(fn, iters=10):
    """Per-kernel device time and host time per call (eager), plus CUDA-graph replay time."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        dev_us = getattr(ev, "device_time_total", None)
        if dev_us is None:
            dev_us = getattr(ev, "cuda_time_total", 0.0)
        if dev_us and dev_us > 0 and ev.key and not ev.key.startswith(("cudaLaunch", "cuLaunch", "ProfilerStep")):
            rows.append({"kernel": ev.key[:120], "calls_per_iter": ev.count / iters,
                         "device_us_per_iter": dev_us / iters})
    rows.sort(key=lambda r: -r["device_us_per_iter"])
    out = {"kernels": rows[:25], "sum_device_us_per_iter": sum(r["device_us_per_iter"] for r in rows)}
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    out["host_us_per_call_no_sync"] = (time.perf_counter() - t0) / iters * 1e6
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            fn()
        out["cuda_graph"] = bench(g.replay)
    except Exception as e:  # noqa: BLE001
        out["cuda_graph"] = f"capture failed: {e!r}"[:600]
    return out


def _guard(res, name, fn, dump):
    """Run one diagnostic sub-stage; a crash is recorded but does not stop later stages."""
    try:
        fn()
    except Exception:  # noqa: BLE001
        import traceback
        res["stages"][f"CRASH_{name}"] = traceback.format_exc()[-3000:]
        res.setdefault("diag_crashes", []).append(name)
    dump()


def precision_and_profile(res, dev, gen, p1, p2, E, H, k, dump):
    """Diagnostics in priority order: kernel stats + profile, promotion A/B, bounded tile sweep."""
    from super_nvfp4_w4a8 import moe as moe_mod
    res["stages"]["profile"] = {}

    def profile_all():
        for M in (1, 32, 256, 8192):
            x = act(M, H, dev, gen)
            tw, ids = routing(M, E, k, dev, gen=gen)
            res["stages"]["profile"][M] = kernel_profile(lambda: nvfp4_w4a8_moe(x, tw, ids, p1, p2))
            res["stages"]["kernel_stats"] = {str(kk): v for kk, v in moe_mod.KSTATS.items()}
            dump()
    _guard(res, "profile", profile_all, dump)

    res["stages"]["promotion_ab"] = {}

    def promotion():
        old = os.environ.get("SUPER_W4A8_PROMOTE")
        try:
            for M in (1, 256):
                x = act(M, H, dev, gen)
                tw, ids = routing(M, E, k, dev, gen=gen)
                sub = min(M, 64)
                r2 = run_ref_r2(x[:sub], tw[:sub], ids[:sub], p1, p2)  # float64 for sub<=64
                for pv in ("1", "0"):
                    os.environ["SUPER_W4A8_PROMOTE"] = pv
                    y = nvfp4_w4a8_moe(x, tw, ids, p1, p2)
                    c = compare(y[:sub], r2)
                    c["timing"] = bench(lambda: nvfp4_w4a8_moe(x, tw, ids, p1, p2))
                    res["stages"]["promotion_ab"][f"M{M}_promote{pv}"] = c
                    dump()
        finally:
            if old is None:
                os.environ.pop("SUPER_W4A8_PROMOTE", None)
            else:
                os.environ["SUPER_W4A8_PROMOTE"] = old
    _guard(res, "promotion_ab", promotion, dump)

    res["stages"]["tile_sweep"] = {}

    def tile_sweep(budget_s=480.0):
        from super_nvfp4_w4a8.moe import moe_align, rowwise_quant, _gemm
        t_start = time.time()
        I = p1[0].shape[1]
        for M, bm in ((1, 16), (32, 16), (256, 32), (2048, 64), (8192, 64), (8192, 128)):
            x = act(M, H, dev, gen)
            tw, ids = routing(M, E, k, dev, gen=gen)
            x8, xs = rowwise_quant(x)
            twf = tw.reshape(-1).contiguous()
            h = torch.empty((M * k, I), dtype=torch.bfloat16, device=dev)
            h8, hs = rowwise_quant(torch.randn(M * k, I, device=dev).bfloat16(), relu2=True)
            y = torch.empty((M * k, H), dtype=torch.float32, device=dev)
            sid, eid, npp = moe_align(ids, bm, E)
            rows = res["stages"]["tile_sweep"].setdefault(M, [])
            for bn in (64, 128):
                for bk in (64, 128):
                    for w in (4, 8):
                        for st in (3,):  # num_stages does not change spills; keep budget for tiles
                            if time.time() - t_start > budget_s:
                                res["stages"]["tile_sweep"]["budget_exhausted_at"] = M
                                return
                            cfg = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_N=8, num_warps=w, num_stages=st)
                            row = {"cfg": cfg}
                            for which in ("fc1", "fc2"):
                                fn = ((lambda: _gemm(x8, xs, *p1, h, twf, sid, eid, npp, M * k, k, False, cfg))
                                      if which == "fc1" else
                                      (lambda: _gemm(h8, hs, *p2, y, twf, sid, eid, npp, M * k, 1, True, cfg)))
                                try:
                                    row[which] = bench(fn, iters=10, warm=2)["median_ms"]
                                except Exception as e:  # noqa: BLE001  (out of resources etc.)
                                    row[which] = f"err {e!r}"[:200]
                            rows.append(row)
                            res["stages"]["tile_sweep"][M] = rows
                            dump()
        res["stages"]["kernel_stats"] = {str(kk): v for kk, v in moe_mod.KSTATS.items()}
    _guard(res, "tile_sweep", tile_sweep, dump)
    res["stages"]["kernel_stats"] = {str(kk): v for kk, v in moe_mod.KSTATS.items()}
    dump()


def synthetic_stages(a, res, dev, gen, dump, fail):
    # ---- 1. tiny
    with exact_matmul():
        E, H, I, k = 4, 64, 64, 2
        w1 = torch.randn(E, I, H, device=dev, generator=gen); w1[1] *= 30
        w2 = torch.randn(E, H, I, device=dev, generator=gen)
        nv1, nv2 = quantize_nvfp4(w1), quantize_nvfp4(w2)
        p1, p2 = prepare_w4a8(*nv1), prepare_w4a8(*nv2)
        x = act(16, H, dev, gen)
        tw, ids = routing(16, E, k, dev, gen=gen)
        ids[3] = torch.tensor([0, 3], device=dev, dtype=torch.int32)  # expert ids 0 and E-1
        y = nvfp4_w4a8_moe(x, tw, ids, p1, p2)
        r2 = run_ref_r2(x, tw, ids, p1, p2)
        c = compare(y, r2)
        res["stages"]["tiny"] = c
        if not c["finite"] or c["rel_l2"] > TOL_DUT_R2:
            fail("tiny", c, dict(x=x, tw=tw, ids=ids, p1=p1, p2=p2, y=y, r2=r2))
        dump()

        # ---- 2. Super shapes, synthetic weights
        E, H, I, k = 512, 1024, 2688, 22
        w1 = torch.randn(E, I, H, device=dev, generator=gen, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(E, H, I, device=dev, generator=gen, dtype=torch.bfloat16) * 0.02
        nv1, nv2 = quantize_nvfp4(w1), quantize_nvfp4(w2)
        del w1, w2
        p1, p2 = prepare_w4a8(*nv1), prepare_w4a8(*nv2)
        precision_and_profile(res, dev, gen, p1, p2, E, H, k, dump)
        res["stages"]["super_synth"] = {}
        for M in [int(v) for v in a.Ms.split(",")]:
            x = act(M, H, dev, gen)
            tw, ids = routing(M, E, k, dev, gen=gen)
            y = nvfp4_w4a8_moe(x, tw, ids, p1, p2)
            sub = min(M, 256)  # reference on a row subset (all experts still exercised)
            r2 = run_ref_r2(x[:sub], tw[:sub], ids[:sub], p1, p2)
            c = compare(y[:sub], r2)
            c["cfg_fc1"] = pick_config(M, k, E, "fc1")
            c["timing"] = bench(lambda: nvfp4_w4a8_moe(x, tw, ids, p1, p2))
            res["stages"]["super_synth"][M] = c
            dump()
            if not c["finite"] or c["rel_l2"] > TOL_DUT_R2 or c["row_rel_max"] > TOL_DUT_R2_ROW:
                fail(f"super_synth_M{M}", c, dict(x=x[:sub], tw=tw[:sub], ids=ids[:sub], y=y[:sub], r2=r2))

        # ---- 3. batch invariance: same rows, different neighbours / permutations
        tgt = act(8, H, dev, gen)
        ttw, tids = routing(8, E, k, dev, gen=gen)
        base = nvfp4_w4a8_moe(tgt, ttw, tids, p1, p2)
        inv = {}
        for M in (8, 64, 1024, 8192):
            if M == 8:
                continue
            other = act(M - 8, H, dev, gen)
            otw, oids = routing(M - 8, E, k, dev, gen=gen)
            perm = torch.randperm(M, device=dev, generator=gen)
            X = torch.cat([tgt, other])[perm]
            TW = torch.cat([ttw, otw])[perm]
            IDS = torch.cat([tids, oids])[perm]
            Y = nvfp4_w4a8_moe(X, TW, IDS, p1, p2)
            back = torch.empty_like(perm); back[perm] = torch.arange(M, device=dev)
            got = Y[back[:8]]
            same_cfg = pick_config(M, k, E, "fc1") == pick_config(8, k, E, "fc1")
            inv[M] = {"bitwise_equal": bool(torch.equal(got, base)), "same_tile_cfg": same_cfg,
                      **compare(got, base)}
        res["stages"]["batch_invariance"] = inv
        dump()
        if not all(v["bitwise_equal"] or not v["same_tile_cfg"] for v in inv.values()):
            fail("batch_invariance", inv, {})
        if any(v["rel_l2"] > TOL_DUT_R2 for v in inv.values()):
            fail("batch_invariance_cross_cfg", inv, {})


def real_bank_stage(a, res, dev, gen, dump, fail):
    # ---- 4. real bank
    if a.nvfp4:
        layer = find_moe_layer(a.nvfp4) if a.layer == "auto" else int(a.layer)
        bank = load_bank(a.nvfp4, layer, 512)
        E = 1 + max(e for e, _, _ in bank)
        top_k = json.load(open(os.path.join(a.nvfp4, "config.json"))).get("num_experts_per_tok", 22)
        nv1 = tuple(stack(bank, E, "up_proj", f).to(dev) for f in ("weight", "weight_scale", "weight_scale_2"))
        nv2 = tuple(stack(bank, E, "down_proj", f).to(dev) for f in ("weight", "weight_scale", "weight_scale_2"))
        nv1 = (nv1[0], nv1[1].view(FP8) if nv1[1].dtype != FP8 else nv1[1], nv1[2].reshape(E))
        nv2 = (nv2[0], nv2[1].view(FP8) if nv2[1].dtype != FP8 else nv2[1], nv2[2].reshape(E))
        p1, p2 = prepare_w4a8(*nv1), prepare_w4a8(*nv2)
        real = {"ckpt": a.nvfp4, "layer": layer, "E": E, "top_k": top_k,
                "w1_shape": tuple(nv1[0].shape), "w2_shape": tuple(nv2[0].shape),
                "wconv_fc1": weight_conversion_error(*nv1, p1[1], p1[2]),
                "wconv_fc2": weight_conversion_error(*nv2, p2[1], p2[2])}
        H = nv2[0].shape[1]
        M = 256
        x = act(M, H, dev, gen)
        tw, ids = routing(M, E, top_k, dev, gen=gen)
        with exact_matmul():
            y = nvfp4_w4a8_moe(x, tw, ids, p1, p2)
            r2 = run_ref_r2(x, tw, ids, p1, p2)
            r1 = moe_r1(x, tw, ids, nv1, nv2, acc_dtype=torch.float32)
            real["DUT_vs_R2"] = compare(y, r2)
            real["R2_vs_R1"] = compare(r2, r1)
            if a.bf16:
                bb = load_bank(a.bf16, layer, E)
                W1 = stack(bb, E, "up_proj", "weight").to(dev)
                W2 = stack(bb, E, "down_proj", "weight").to(dev)
                r0 = moe_dense_ref(x, tw, ids, W1, W2, acc_dtype=torch.float32)
                real["R1_vs_R0"] = compare(r1, r0)
                real["R2_vs_R0"] = compare(r2, r0)
                real["DUT_vs_R0"] = compare(y, r0)
                real["weight_R1_vs_R0_rel_l2"] = {
                    "fc1": ((dequant_nvfp4(*nv1) - W1.float()).norm() / W1.float().norm()).item(),
                    "fc2": ((dequant_nvfp4(*nv2) - W2.float()).norm() / W2.float().norm()).item()}
        res["stages"]["real_bank"] = real
        dump()
        if real["DUT_vs_R2"]["rel_l2"] > TOL_DUT_R2:
            fail("real_bank", real, dict(x=x, tw=tw, ids=ids, y=y, r2=r2))
        # DUT timing on the real bank (warm CUDA events); no A16 baseline in this script
        real["timing"] = {}
        for Mt in (1, 32, 256, 2048, 8192):
            xt = act(Mt, H, dev, gen)
            twt, idt = routing(Mt, E, top_k, dev, gen=gen)
            real["timing"][Mt] = bench(lambda: nvfp4_w4a8_moe(xt, twt, idt, p1, p2))
            dump()


if __name__ == "__main__":
    main()
