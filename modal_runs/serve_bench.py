"""Full-model vLLM serve comparison: stock Marlin NVFP4 W4A16 vs this repo's hybrid W4A8.

    python modal_runs/serve_bench.py --model <nvfp4 ckpt> --out <dir> [--variants A16,HYBRID]

Per variant: start `vllm serve` (CUDA graphs on; falls back to --enforce-eager and records it),
then run the same probes and `vllm bench serve` workloads (random tokens, --ignore-eos):
    prefill   in 8192 / out 1,   64 prompts, unbounded rate
    mixed     in 1024 / out 256, 256 prompts, unbounded rate
    decode    in 128  / out 512, 128 prompts, 128 concurrent
    latency1  in 512  / out 256, 8 prompts, 1 concurrent
Accuracy probes: greedy answers to fixed arithmetic questions and first-token top-20 logprobs of
fixed prompts, compared across variants (same weights; only the MoE expert path differs).
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
NAME = "probe"
QUESTIONS = [("Only output the number: 17 + 25.", "42"), ("Only output the number: 12 multiplied by 13.", "156"),
             ("Only output the number: 144 divided by 12.", "12"), ("Only output the number: 19 squared.", "361"),
             ("Only output the number: area of a 15 by 8 rectangle.", "120"),
             ("Only output the number: 250 minus 87.", "163"), ("Only output the number: 7 times 8 plus 9.", "65"),
             ("Only output the number: a dozen plus two dozen.", "36")]
LOGPROB_PROMPTS = ["The capital of France is", "def fibonacci(n):\n    ", "In 1969, the first humans to walk on",
                   "Water boils at a temperature of", "The quick brown fox", "Photosynthesis is the process by which",
                   "Một năm có", "The derivative of x^2 is"]
WORKLOADS = [("prefill", 8192, 1, 64, None), ("mixed", 1024, 256, 256, None), ("mixed2", 1024, 256, 256, None),
             ("decode", 128, 512, 128, 128), ("latency1", 512, 256, 8, 1)]
VARIANTS = {"A16": (["--moe-backend", "marlin"], {"VLLM_PLUGINS": ""}),
            "AUTO": ([], {"VLLM_PLUGINS": ""}),   # vLLM's default NVFP4 MoE backend (Humming on SM90 in main)
            "HUMMING": (["--moe-backend", "humming"], {"VLLM_PLUGINS": ""}),
            # the vLLM PR branch backend (needs W4A8_VLLM_OVERLAY at image build)
            "PR": (["--moe-backend", "marlin_fp8_prefill"], {"VLLM_PLUGINS": ""}),
            # vLLM main's own NVFP4 x FP8 path on Hopper: Humming with FP8 activations
            "HUMMING_FP8": (["--moe-backend", "humming"], {"VLLM_PLUGINS": "",
                            "VLLM_HUMMING_INPUT_QUANT_CONFIG": '{"dtype":"float8e4m3"}'}),
            "HYBRID": (["--quantization", "super_nvfp4_w4a8"], {"SUPER_W4A8_MODE": "hybrid",
                                                               "VLLM_PLUGINS": "super_nvfp4_w4a8"}),
            "FUSED": (["--quantization", "super_nvfp4_w4a8"], {"SUPER_W4A8_MODE": "fused"}),
            # MXFP4 W4A8 baseline: FlashInfer SM90 "Humming" kernel, quantised online from the BF16
            # checkpoint by the handoff adapter (third_party/super_mxfp4a8)
            "MXFP4": (["--quantization", "super_mxfp4a8_smoke"], {})}


PROFILE_PHASES = [  # (name, in, out, prompts, concurrency); each is first run unprofiled as warm-up
    ("prefill", 8192, 1, 8, None), ("decode", 16, 48, 128, 128), ("bs1", 512, 32, 1, 1)]


def serve_cmd(model, extra, eager, max_len, profile_dir=None):
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model, "--served-model-name", NAME,
            "--trust-remote-code", "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.92",
            "--max-model-len", str(max_len), "--max-num-seqs", "256", "--max-num-batched-tokens", "8192",
            "--kv-cache-dtype", "fp8", "--mamba-ssm-cache-dtype", "float32", "--enable-chunked-prefill",
            "--no-enable-prefix-caching", "--max-logprobs", "20", "--port", "8000",
            *(["--enforce-eager"] if eager else []), *extra,
            *(["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": str(profile_dir),
                                                "torch_profiler_with_stack": False, "ignore_frontend": True,
                                                "torch_profiler_use_gzip": True})] if profile_dir else [])]


def start(model, extra, env, eager, max_len, log_path, ready_s, profile_dir=None):
    cmd = serve_cmd(model, extra, eager, max_len, profile_dir)
    log = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                         env={**os.environ, "VLLM_PLUGINS": "super_nvfp4_w4a8,super_mxfp4a8", **env})

    def pump():
        for line in p.stdout:
            log.write(line); log.flush()
            if any(t in line for t in ("SUPER_", "Error", "error", "Traceback", "KV cache", "memory", "backend",
                                       "Application startup complete", "CUDA graph", "Loading weights took")):
                print("[serve]", line.rstrip()[:400], flush=True)
    threading.Thread(target=pump, daemon=True).start()
    t0 = time.time()
    while time.time() - t0 < ready_s:
        if p.poll() is not None:
            return p, cmd, None
        try:
            if requests.get(BASE + "/health", timeout=2).status_code == 200:
                return p, cmd, time.time() - t0
        except requests.RequestException:
            pass
        time.sleep(3)
    return p, cmd, None


def stop(p):
    if p is not None and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait()


def probes():
    out = {"qa": [], "logprobs": []}
    for q, ans in QUESTIONS:
        r = requests.post(BASE + "/v1/chat/completions", timeout=120, json={
            "model": NAME, "messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 48,
            "chat_template_kwargs": {"enable_thinking": False}}).json()
        txt = (r.get("choices") or [{}])[0].get("message", {}).get("content") or json.dumps(r)[:200]
        out["qa"].append({"q": q, "expected": ans, "answer": txt[:200], "correct": ans in (txt or "")})
    for pr in LOGPROB_PROMPTS:
        r = requests.post(BASE + "/v1/completions", timeout=120, json={
            "model": NAME, "prompt": pr, "temperature": 0, "max_tokens": 1, "logprobs": 20}).json()
        lp = ((r.get("choices") or [{}])[0].get("logprobs") or {}).get("top_logprobs") or [{}]
        out["logprobs"].append(lp[0])
    out["qa_correct"] = sum(x["correct"] for x in out["qa"])
    return out


def bench(model, wl, result_dir, tag=""):
    name, il, ol, n, conc = wl
    fn = f"{name}{tag}.json"
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--backend", "vllm",
           "--base-url", BASE, "--model", NAME, "--tokenizer", model, "--trust-remote-code",
           "--dataset-name", "random", "--random-input-len", str(il), "--random-output-len", str(ol),
           "--num-prompts", str(n), "--request-rate", "inf", "--ignore-eos",
           "--seed", "4321" if name.endswith("2") else "1234",
           "--save-result", "--result-dir", str(result_dir), "--result-filename", fn,
           *(["--max-concurrency", str(conc)] if conc else [])]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    res = {"cmd": cmd, "rc": r.returncode, "wall_s": time.time() - t, "tail": (r.stdout + r.stderr)[-1500:]}
    f = Path(result_dir) / fn
    if f.exists():
        d = json.loads(f.read_text())
        res["metrics"] = {k: d.get(k) for k in (
            "duration", "completed", "total_input_tokens", "total_output_tokens", "request_throughput",
            "input_throughput", "output_throughput", "total_token_throughput", "mean_ttft_ms", "median_ttft_ms",
            "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms")}
    return res


def _category(name):
    n = name.lower()
    rules = [("moe_fp8_decode", ("marlin_tile_decode", "nvfp4_to_fp8")),
             ("moe_w4a16_marlin", ("marlin",)),
             ("moe_grouped_fp8", ("cutlass_moe", "grouped", "kernelgroup", "ptrarray")),
             ("moe_mxfp4", ("humming", "mxfp4", "trtllm", "fp4", "moe_gemm", "fused_moe")),
             ("moe_routing", ("moe_align", "topk", "moe_sum", "moe_permute", "unpermute", "expand",
                              "count_and_sort", "routing", "shuffle_rows", "compute_arg_sorts", "gather", "scatter")),
             ("fp8_quant", ("quant",)),
             ("attention", ("flash", "fmha", "attention", "attn", "paged", "reshape_and_cache", "merge_")),
             ("mamba", ("mamba", "ssd", "selective", "causal_conv", "chunk_scan", "chunk_state", "state_passing",
                        "bmm_chunk", "cumsum", "ssm")),
             ("gemm_dense", ("gemm", "cutlass", "sm90", "nvjet", "cublas", "xmma", "scaled_mm", "wgmma")),
             ("memcpy", ("memcpy", "memset")),
             ("norm_act", ("norm", "relu", "silu", "elementwise", "vectorized", "reduce"))]
    for cat, keys in rules:
        if any(k in n for k in keys):
            return cat
    return "other"


def _kernel_table(files):
    """Sum GPU kernel time by name over the worker traces of one profiled phase."""
    import gzip
    agg, n_ev, t0, t1 = {}, 0, None, None
    for f in files:
        op = gzip.open if str(f).endswith(".gz") else open
        with op(f, "rt") as fh:
            d = json.load(fh)
        for e in d.get("traceEvents", []):
            if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            k = e.get("name", "?")
            a = agg.setdefault(k, [0.0, 0])
            a[0] += float(e.get("dur", 0)); a[1] += 1; n_ev += 1
            ts = float(e.get("ts", 0)); t0 = ts if t0 is None else min(t0, ts)
            t1 = max(t1 or 0, ts + float(e.get("dur", 0)))
    total = sum(v[0] for v in agg.values())
    rows = sorted(({"name": k[:220], "us": v[0], "calls": v[1], "pct": 100 * v[0] / max(total, 1e-9),
                    "cat": _category(k)} for k, v in agg.items()), key=lambda r: -r["us"])
    cats = {}
    for r in rows:
        cats[r["cat"]] = cats.get(r["cat"], 0.0) + r["us"]
    return {"gpu_kernel_us": total, "gpu_span_us": (t1 - t0) if t0 is not None else None, "events": n_ev,
            "categories": {c: {"us": v, "pct": 100 * v / max(total, 1e-9)}
                           for c, v in sorted(cats.items(), key=lambda kv: -kv[1])},
            "top": rows[:60]}


def profile_phases(model, out, pdir):
    res = {}
    for ph in PROFILE_PHASES:
        bench(model, ph, out, tag="-warm")                         # JIT / autotune / graph warm-up
        before = set(pdir.rglob("*"))
        r0 = requests.post(BASE + "/start_profile", timeout=600)
        b = bench(model, ph, out, tag="-prof")
        r1 = requests.post(BASE + "/stop_profile", timeout=1800)
        files, last = [], -1
        for _ in range(120):                                       # traces are flushed asynchronously
            files = [f for f in pdir.rglob("*") if f.is_file() and f not in before and "trace" in f.name]
            size = sum(f.stat().st_size for f in files)
            if files and size == last:
                break
            last = size; time.sleep(5)
        row = {"start_rc": r0.status_code, "stop_rc": r1.status_code, "files": [str(f) for f in files],
               "bench": b.get("metrics"), "bench_rc": b.get("rc"), "bench_tail": b.get("tail", "")[-600:]}
        try:
            row.update(_kernel_table(files))
        except Exception:  # noqa: BLE001
            row["parse_error"] = traceback.format_exc()[-1500:]
        res[ph[0]] = row
        cats = {k: round(v["pct"], 1) for k, v in (row.get("categories") or {}).items()}
        print(f"PROFILE {ph[0]} kernel_ms={row.get('gpu_kernel_us', 0) / 1e3:.1f} events={row.get('events')} "
              f"files={len(files)} {cats}", flush=True)
    return res


def lm_eval(model, task, out_dir, fewshot=5, limit=None):
    """lm-eval-harness (separate venv) against the running server: greedy, completions API."""
    od = Path(out_dir) / task
    od.mkdir(parents=True, exist_ok=True)
    margs = (f"model={NAME},base_url={BASE}/v1/completions,num_concurrent=128,max_retries=3,"
             f"tokenized_requests=False,tokenizer={model},tokenizer_backend=huggingface,trust_remote_code=True")
    cmd = ["/opt/lmeval/bin/lm_eval", "--model", "local-completions", "--model_args", margs, "--tasks", task,
           "--num_fewshot", str(fewshot), "--batch_size", "1", "--output_path", str(od), "--seed", "1234"]
    if limit:
        cmd += ["--limit", str(limit)]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    res = {"cmd": cmd, "rc": r.returncode, "wall_s": time.time() - t, "tail": (r.stdout + r.stderr)[-2000:]}
    files = sorted(od.rglob("results*.json"), key=lambda f: f.stat().st_mtime)
    if files:
        d = json.loads(files[-1].read_text())
        res["metrics"] = d.get("results", {}).get(task)
        res["n_samples"] = d.get("n-samples", {}).get(task)
    return res


def compare_logprobs(a, b):
    """Per prompt: argmax agreement and symmetric KL on the union of top-20 (missing = min logprob)."""
    rows = []
    for x, y in zip(a, b):
        if not x or not y:
            rows.append(None); continue
        keys = set(x) | set(y)
        fx, fy = min(x.values()), min(y.values())
        px = {k: math.exp(x.get(k, fx)) for k in keys}
        py = {k: math.exp(y.get(k, fy)) for k in keys}
        sx, sy = sum(px.values()), sum(py.values())
        kl = sum(px[k] / sx * math.log((px[k] / sx) / (py[k] / sy)) for k in keys)
        kl += sum(py[k] / sy * math.log((py[k] / sy) / (px[k] / sx)) for k in keys)
        rows.append({"argmax_same": max(x, key=x.get) == max(y, key=y.get), "sym_kl_top20": kl})
    ok = [r for r in rows if r]
    return {"per_prompt": rows, "argmax_agree": sum(r["argmax_same"] for r in ok), "n": len(ok),
            "mean_sym_kl": sum(r["sym_kl_top20"] for r in ok) / max(len(ok), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bf16", default=None, help="BF16 checkpoint for the MXFP4 variant")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="A16,HYBRID")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--ready-s", type=int, default=1200)
    ap.add_argument("--workloads", default="prefill,mixed,decode,latency1")
    ap.add_argument("--profile", action="store_true", help="kernel profile per phase instead of benchmarks")
    ap.add_argument("--evals", default="", help="comma list of lm-eval tasks, e.g. gsm8k")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"model": a.model, "variants": {}, "status": "RUNNING",
           "device": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                    capture_output=True, text=True).stdout.strip()}
    save = lambda: (out / "result.json").write_text(json.dumps(res, indent=1, default=str))  # noqa: E731
    wls = [w for w in WORKLOADS if w[0] in a.workloads.split(",")]
    for v in a.variants.split(","):
        base, _, thr = v.partition("@")   # e.g. HYBRID@4096 -> dense path from 4096 tokens
        extra, env = VARIANTS[base]
        env = {**env, **({"SUPER_W4A8_DENSE_MIN_TOKENS": thr} if thr else {})}
        row = {"extra": extra, "env": env}
        res["variants"][v] = row
        p = None
        try:
            for eager in (False, True):
                model = a.bf16 if base == "MXFP4" else a.model
                if base == "MXFP4" and not (a.bf16 and Path(a.bf16, "config.json").exists()):
                    raise FileNotFoundError(f"MXFP4 needs the BF16 checkpoint, missing: {a.bf16}")
                pdir = (out / f"{v}-trace") if a.profile else None
                if pdir:
                    pdir.mkdir(parents=True, exist_ok=True)
                p, cmd, ready = start(model, extra, env, eager, a.max_len, out / f"{v}-server{'-eager' if eager else ''}.log",
                                      a.ready_s, pdir)
                row.update(cmd=cmd, eager=eager, ready_s=ready)
                if ready is not None:
                    break
                row.setdefault("start_failures", []).append({"eager": eager, "rc": p.poll()})
                stop(p); p = None
                save()
            if p is None:
                row["status"] = "START_FAILED"; save(); continue
            print(f"BENCH_PROGRESS {v} ready in {row['ready_s']:.0f}s eager={row['eager']}", flush=True)
            if a.profile:
                row["profile"] = profile_phases(a.model, out / v, pdir); row["status"] = "PROFILED"; save()
                continue
            row["probes"] = probes(); save()
            print(f"BENCH_PROGRESS {v} probes qa={row['probes']['qa_correct']}/{len(QUESTIONS)}", flush=True)
            row["evals"] = {}
            for task in [t for t in a.evals.split(",") if t]:
                row["evals"][task] = lm_eval(model, task, out / v / "lm_eval")
                m = row["evals"][task].get("metrics") or {}
                print(f"BENCH_PROGRESS {v} eval {task} rc={row['evals'][task]['rc']} "
                      f"{ {k: v_ for k, v_ in m.items() if 'exact_match' in k} }", flush=True)
                save()
            row["bench"] = {}
            for wl in wls:
                row["bench"][wl[0]] = bench(a.model, wl, out / v)
                m = row["bench"][wl[0]].get("metrics") or {}
                print(f"BENCH_PROGRESS {v} {wl[0]} rc={row['bench'][wl[0]]['rc']} in_tok/s={m.get('input_throughput')} "
                      f"out_tok/s={m.get('output_throughput')} ttft={m.get('median_ttft_ms')} tpot={m.get('median_tpot_ms')}",
                      flush=True)
                save()
            row["status"] = "MEASURED"
        except Exception:  # noqa: BLE001
            row["status"] = "FAIL"; row["error"] = traceback.format_exc()[-3000:]
            print(row["error"], flush=True)
        finally:
            stop(p); save()
    vs = [v for v in res["variants"] if res["variants"][v].get("probes")]
    if "A16" in vs:
        res["logprob_vs_A16"] = {v: compare_logprobs(res["variants"]["A16"]["probes"]["logprobs"],
                                                     res["variants"][v]["probes"]["logprobs"]) for v in vs if v != "A16"}
    res["status"] = "DONE"; save()
    print("SERVE_RESULT", json.dumps({v: {"status": r.get("status"), "eager": r.get("eager"),
                                         "qa": (r.get("probes") or {}).get("qa_correct"),
                                         **{k: (b.get("metrics") or {}).get("total_token_throughput")
                                            for k, b in (r.get("bench") or {}).items()}}
                                     for v, r in res["variants"].items()}), flush=True)


if __name__ == "__main__":
    main()
