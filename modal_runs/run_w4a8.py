"""Modal launcher, three gated stages. One H200, no retries, no GPU fallback.

    modal run --env main modal_runs/run_w4a8.py --stage cpu     # CPU only, writes cpu-PASS.json
    modal run --env main modal_runs/run_w4a8.py --stage op      # H200 operator gate  (~10-15 min)
    modal run --detach --env main modal_runs/run_w4a8.py --stage serve   # full model (~25-35 min)

op and serve refuse to start unless cpu-PASS.json matches the current source hashes;
serve additionally requires op-PASS.json for the same hashes. Evidence goes to the volume
under /probe-evidence/nvfp4-w4a8-*/<attempt>/ and the JSON result is written locally.
"""
import hashlib
import json
import os
import signal
import subprocess
import threading
import uuid
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASE = os.environ.get("W4A8_BASE_IMAGE", "registry:vllm/vllm-openai:v0.33.0")  # or an existing Modal image id
VOLUME = os.environ.get("W4A8_VOLUME", "nemotron-models")
# Checkpoint directory layout of the models volume (two layouts in use); paths are resolved locally and passed to the functions
# (container env does not see local env vars).
_LAYOUT = {
    "flat": {"super": ("/cache/models/NVFP4/ff433f5493e25d631c9f12b5d55c674229923d02",
                         "/cache/models/BF16/2dc98e2afe4face0e4ce40972a915c45368bd34a"),
               "lightning": ("/cache/models/lightning-nvfp4/bee7596271d1495f6992ae224aefde4410e816b8",
                             "/cache/models/lightning-bf16/a9904d24bcc1d289a1950fa9d2b978c47cf903b9")},
    "split": {"super": ("/cache/models/SuperNVFP4/ff433f5493e25d631c9f12b5d55c674229923d02",
                         "/cache/models/SuperBF16/2dc98e2afe4face0e4ce40972a915c45368bd34a"),
               "lightning": ("/cache/models/NVFP4/bee7596271d1495f6992ae224aefde4410e816b8",
                             "/cache/models/BF16/a9904d24bcc1d289a1950fa9d2b978c47cf903b9")},
}
LAYOUT = _LAYOUT[os.environ.get("W4A8_LAYOUT", "split")]
NVFP4, BF16 = LAYOUT["super"]
LIGHT_NVFP4, LIGHT_BF16 = LAYOUT["lightning"]
BENCH_GROUPS = os.environ.get("BENCH_GROUPS", "A16_MARLIN_NVFP4,A8_DEQ_TRITON_gemm_only,A8_DEQ_TRITON_NVFP4;"
                              "A8_DEQ_CUTLASS_gemm_only,A8_DEQ_CUTLASS_NVFP4")
BENCH_MODELS = os.environ.get("BENCH_MODELS", "super,lightning")
BENCH_MS = os.environ.get("BENCH_MS", "1,8,32,128,512,2048,8192,16384")
BENCH_TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "2400"))

app = modal.App("super-nvfp4-w4a8")
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
image = ((modal.Image.from_registry(BASE[len("registry:"):]).entrypoint([])
           .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python && python --version")
          if BASE.startswith("registry:") else modal.Image.from_id(BASE))
         .run_commands("/usr/bin/python3 -m pip install -q pytest safetensors")
         # lm-eval in its own venv (talks to the vLLM server over HTTP; must not touch vLLM's deps)
         .run_commands("/usr/bin/python3 -m venv /opt/lmeval && /opt/lmeval/bin/pip install -q 'lm-eval[api]==0.4.9'")
         .add_local_dir(str(REPO), "/probe/w4a8", copy=True,
                        ignore=["**/__pycache__/**", "**/*.json", "**/*.pt", "**/*.log", "**/*.ptx",
                                ".git/**", ".pytest_cache/**"])
         .run_commands("/usr/bin/python3 -m pip install --no-deps --no-build-isolation /probe/w4a8 "
                       "/probe/w4a8/third_party/super_mxfp4a8"))


def binding():
    files = sorted(p for p in REPO.rglob("*.py") if "__pycache__" not in p.parts)
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def native(cmd, deadline, env=None, cwd="/probe/w4a8"):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True, cwd=cwd, env={**os.environ, **(env or {})})
    timer = threading.Timer(deadline, lambda: os.killpg(p.pid, signal.SIGTERM))
    timer.start()
    try:
        for line in p.stdout:
            print(line.rstrip(), flush=True)
        return p.wait()
    finally:
        timer.cancel()
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


def _evidence(kind, attempt):
    d = Path(f"/cache/probe-evidence/nvfp4-w4a8-{kind}/{attempt}")
    d.mkdir(parents=True, exist_ok=False)
    return d


@app.function(image=image, cpu=8, memory=32768, timeout=1200, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/cache": volume})
def cpu_stage(attempt, cfg):
    d = _evidence("cpu", attempt)
    try:
        native(["/usr/bin/python3", "-u", "modal_runs/cpu_gate.py", str(d / "result.json")], 1100,
               env={"NVFP4_CKPT": cfg["layout"]["super"][0], "CUDA_VISIBLE_DEVICES": ""})
        for f in ("gemm_sm90.ptx", "fused_marlin.ttgir"):
            src = Path("/probe/w4a8") / f
            if src.exists():
                (d / f).write_bytes(src.read_bytes())
        return json.loads((d / "result.json").read_text())
    finally:
        volume.commit()


@app.function(image=image, gpu="H200", cpu=8, memory=196608, timeout=1800, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/cache": volume})
def op_stage(attempt, cfg):
    NVFP4, BF16 = cfg["layout"]["super"]
    LIGHT_NVFP4, LIGHT_BF16 = cfg["layout"]["lightning"]
    d = _evidence("op", attempt)
    try:
        subprocess.run(["nvidia-smi"], check=False)
        rc = native(["/usr/bin/python3", "-u", "tests/gpu_operator_check.py", "--out", str(d / "result.json"),
                     "--nvfp4", NVFP4, "--bf16", BF16, "--save-failures", str(d / "failures.pt")], 1300)
        r = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {"status": "HARNESS_FAIL"}
        r["returncode"] = rc
        # Second model in the same container (no extra cold start): Lightning 30B real bank only.
        # Different shapes (E128 top6 H2688 I1856: masked K/N tails). Runs even if Super failed.
        rc2 = native(["/usr/bin/python3", "-u", "tests/gpu_operator_check.py", "--real-only",
                      "--out", str(d / "lightning.json"), "--nvfp4", LIGHT_NVFP4, "--bf16", LIGHT_BF16,
                      "--save-failures", str(d / "lightning-failures.pt")], 400)
        r["lightning"] = (json.loads((d / "lightning.json").read_text())
                          if (d / "lightning.json").exists() else {"status": "HARNESS_FAIL"})
        r["lightning"]["returncode"] = rc2
        if r["lightning"].get("status") != "PASS" and r.get("status") == "PASS":
            r["status"] = "FAIL_lightning"
        return r
    finally:
        volume.commit()


@app.function(image=image, gpu="H200", cpu=8, memory=196608, timeout=BENCH_TIMEOUT, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/cache": volume})
def bench_stage(attempt, cfg):
    """Operator-level baselines on real layers: Marlin NVFP4 A16 vs Humming MXFP4 A8 vs ours."""
    d = _evidence("bench", attempt)
    out = {"status": "RUNNING", "cfg": cfg}
    try:
        subprocess.run(["nvidia-smi"], check=False)
        # one process per variant group: a sticky CUDA fault in one group cannot erase the others
        groups = [g for g in cfg["groups"].split(";") if g]
        for tag in cfg["models"].split(","):
            nv, bf = cfg["layout"][tag]
            dl = cfg["deadline"][tag]
            out[tag] = {}
            for gi, grp in enumerate(groups):
                f = d / f"{tag}-g{gi}.json"
                rc = native(["/usr/bin/python3", "-u", "tests/bench_baselines.py", "--nvfp4", nv, "--bf16", bf,
                             "--out", str(f), "--Ms", cfg["Ms"]], dl,
                            env={"BENCH_VARIANTS": grp, **cfg.get("env", {})})
                out[tag][f"g{gi}"] = json.loads(f.read_text()) if f.exists() else {"status": "HARNESS_FAIL"}
                out[tag][f"g{gi}"]["returncode"] = rc
                volume.commit()  # keep finished groups even if the container dies later
        out["status"] = "DONE"
        return out
    finally:
        volume.commit()


@app.function(image=image, gpu="H200", cpu=8, memory=196608, timeout=2700, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/cache": volume})
def serve_stage(attempt, variants):
    os.environ["PROBE_MODAL_CALL_ID"] = modal.current_function_call_id()
    try:
        target = Path("/tmp") / f"{attempt}.json"
        native(["/usr/bin/python3", "-u", "modal_runs/serve_worker.py", "gpu", attempt, str(target)], 2600,
               env={"SERVE_VARIANTS": variants})
        return json.loads(target.read_text()) if target.exists() else {"status": "HARNESS_FAIL"}
    finally:
        volume.commit()


SERVE2_TIMEOUT = int(os.environ.get("SERVE2_TIMEOUT", "3600"))


@app.function(image=image, gpu="H200", cpu=16, memory=262144, timeout=SERVE2_TIMEOUT, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/cache": volume})
def serve2_stage(attempt, cfg):
    """Full-model vLLM comparison (modal_runs/serve_bench.py): stock Marlin A16 vs hybrid W4A8."""
    d = _evidence("serve2", attempt)
    out = {"status": "RUNNING", "cfg": cfg}
    try:
        for tag in cfg["serve_models"].split(","):
            nv = cfg["layout"][tag][0]
            od = d / tag
            native(["/usr/bin/python3", "-u", "modal_runs/serve_bench.py", "--model", nv, "--out", str(od),
                    "--bf16", cfg["layout"][tag][1],
                    "--variants", cfg["serve_variants"], "--workloads", cfg["serve_workloads"],
                    "--max-len", str(cfg["serve_max_len"]), *(["--profile"] if cfg.get("serve_profile") else []),
                    "--evals", cfg.get("serve_evals", "")],
                   cfg["serve_deadline"][tag],
                   env=cfg.get("env", {}))
            f = od / "result.json"
            out[tag] = json.loads(f.read_text()) if f.exists() else {"status": "HARNESS_FAIL"}
            volume.commit()
        out["status"] = "DONE"
        return out
    finally:
        volume.commit()


@app.local_entrypoint()
def main(stage: str = "cpu", variants: str = "NVFP4_W4A8"):
    assert stage in ("cpu", "op", "bench", "serve", "serve2")
    # every script a GPU stage launches must exist at its repo-relative path (cwd=/probe/w4a8)
    for rel in ("tests/gpu_operator_check.py", "tests/bench_baselines.py", "modal_runs/serve_worker.py",
                "modal_runs/serve_bench.py"):
        assert (REPO / rel).is_file(), f"missing {rel}"
    b = binding()
    if stage in ("op", "bench", "serve", "serve2"):
        gate = json.loads((HERE / "cpu-PASS.json").read_text())
        assert gate["status"] == "CPU_PASS" and gate["binding"] == b, "re-run --stage cpu for this source"
    if stage == "serve":
        og = json.loads((HERE / "op-PASS.json").read_text())
        assert og["status"] == "PASS" and og.get("binding") == b, "operator gate must pass first"
    attempt = f"{stage}-{uuid.uuid4().hex[:10]}"
    fn = {"cpu": cpu_stage, "op": op_stage, "bench": bench_stage, "serve": serve_stage,
          "serve2": serve2_stage}[stage]
    cfg = {"layout": LAYOUT, "groups": BENCH_GROUPS, "models": BENCH_MODELS, "Ms": BENCH_MS,
           "env": {k: os.environ[k] for k in ("BENCH_PROBE_MARLIN", "BENCH_HYBRID_THR", "SUPER_W4A8_MODE",
                             "SUPER_W4A8_DENSE_MIN_TOKENS", "SUPER_W4A8_SHARE_CODES", "BENCH_FM_CFGS",
                             "BENCH_KPROF", "SUPER_W4A8_FUSED_ACT", "SUPER_W4A8_FUSED_MIN_TOKENS")
                   if k in os.environ},
           "deadline": {"super": int(os.environ.get("BENCH_DL_SUPER", "700")),
                        "lightning": int(os.environ.get("BENCH_DL_LIGHT", "400"))},
           "serve_models": os.environ.get("SERVE_MODELS", "lightning"),
           "serve_variants": os.environ.get("SERVE_VARIANTS", "A16,HYBRID"),
           "serve_workloads": os.environ.get("SERVE_WORKLOADS", "prefill,mixed,decode,latency1"),
           "serve_max_len": int(os.environ.get("SERVE_MAX_LEN", "32768")),
           "serve_profile": os.environ.get("SERVE_PROFILE") == "1",
           "serve_evals": os.environ.get("SERVE_EVALS", ""),
           "serve_deadline": {"lightning": int(os.environ.get("SERVE_DL_LIGHT", "1500")),
                              "super": int(os.environ.get("SERVE_DL_SUPER", "2400"))}}
    call = fn.spawn(attempt, variants) if stage == "serve" else fn.spawn(attempt, cfg)
    meta = {"attempt": attempt, "call_id": call.object_id, "base_image": BASE,
            "profile": os.environ.get("MODAL_PROFILE"), "binding": b, "cfg": cfg}
    (HERE / f"{attempt}-submission.json").write_text(json.dumps(meta, indent=2))
    print("SUBMITTED", json.dumps(meta), flush=True)
    r = call.get()
    r["binding"] = b
    (HERE / f"{attempt}-result.json").write_text(json.dumps(r, indent=2, default=str))
    if stage == "cpu" and r.get("status") == "CPU_PASS":
        (HERE / "cpu-PASS.json").write_text(json.dumps(r, indent=2))
    if stage == "op" and r.get("status") == "PASS":
        (HERE / "op-PASS.json").write_text(json.dumps(r, indent=2, default=str))
    print("RESULT", r.get("status"), flush=True)
