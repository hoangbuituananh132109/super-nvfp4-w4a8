"""CPU-only Modal job: pinned Hugging Face snapshots straight into the models volume.

    modal run modal_runs/fetch_ckpt.py --which lightning-nvfp4,super-nvfp4

Layout written ("split" layout of run_w4a8.py):
    /cache/models/NVFP4/<rev>        Lightning NVFP4      /cache/models/BF16/<rev>       Lightning BF16
    /cache/models/SuperNVFP4/<rev>   Super NVFP4          /cache/models/SuperBF16/<rev>  Super BF16
File list and sizes are checked against the Hub metadata of the pinned revision. No GPU, nothing
is downloaded to the local machine.
"""
import json
import os
import time

import modal

CKPTS = {
    "lightning-nvfp4": ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                        "bee7596271d1495f6992ae224aefde4410e816b8", "NVFP4"),
    "lightning-bf16": ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
                       "a9904d24bcc1d289a1950fa9d2b978c47cf903b9", "BF16"),
    "super-nvfp4": ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
                    "ff433f5493e25d631c9f12b5d55c674229923d02", "SuperNVFP4"),
    "super-bf16": ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
                   "2dc98e2afe4face0e4ce40972a915c45368bd34a", "SuperBF16"),
    "qwen3-30b-nvfp4": ("nvidia/Qwen3-30B-A3B-NVFP4", "2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3", "Qwen3-30B-A3B-NVFP4"),
}
VOLUME = os.environ.get("W4A8_VOLUME", "nemotron-models")

app = modal.App("super-w4a8-fetch")
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.0", "hf_xet")


@app.function(image=image, cpu=8, memory=16384, timeout=7200, retries=0, max_containers=4,
              scaledown_window=2, volumes={"/cache": vol}, ephemeral_disk=524288)
def fetch(key: str):
    from huggingface_hub import HfApi, snapshot_download
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    repo, rev, sub = CKPTS[key]
    dest = f"/cache/models/{sub}/{rev}"
    t0 = time.time()
    info = HfApi().model_info(repo, revision=rev, files_metadata=True)
    want = {s.rfilename: s.size for s in info.siblings}
    snapshot_download(repo, revision=rev, local_dir=dest, max_workers=16)
    got = {}
    for root, _, files in os.walk(dest):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, dest)
            if not rel.startswith(".cache"):
                got[rel] = os.path.getsize(p)
    bad = {f: (s, got.get(f)) for f, s in want.items() if s is not None and got.get(f) != s}
    vol.commit()
    return {"key": key, "repo": repo, "rev": rev, "dest": dest, "files": len(want),
            "GiB": round(sum(v or 0 for v in want.values()) / 2**30, 2), "mismatch": bad,
            "seconds": round(time.time() - t0, 1), "status": "OK" if not bad else "MISMATCH"}


@app.local_entrypoint()
def main(which: str = "lightning-nvfp4,super-nvfp4"):
    keys = [k for k in which.split(",") if k]
    for r in fetch.map(keys):
        print("FETCH", json.dumps(r), flush=True)
