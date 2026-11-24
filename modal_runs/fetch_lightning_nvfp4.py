"""CPU-only Modal job: pin-download Nemotron 3.5 Lightning 30B-A3B NVFP4 into the models volume ($W4A8_VOLUME).

    python -m modal run --env main tools/fetch_lightning_nvfp4.py

Revision is pinned; a mismatch in file list or sizes is an error. No GPU. Nothing is downloaded to the local machine.
"""
import json
import os
import time

import modal

REPO = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
REV = "bee7596271d1495f6992ae224aefde4410e816b8"
VOLUME = os.environ.get("W4A8_VOLUME", "nemotron-models")
DEST = f"/cache/models/lightning-nvfp4/{REV}"

app = modal.App("super-w4a8-fetch-lightning")
vol = modal.Volume.from_name(VOLUME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.0", "hf_xet")


@app.function(image=image, cpu=4, memory=8192, timeout=3000, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/cache": vol})
def fetch():
    from huggingface_hub import HfApi, snapshot_download
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    t0 = time.time()
    info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
    want = {s.rfilename: s.size for s in info.siblings if not s.rfilename.endswith((".png",))}
    snapshot_download(REPO, revision=REV, local_dir=DEST, allow_patterns=list(want), max_workers=16)
    got = {}
    for f in want:
        p = os.path.join(DEST, f)
        got[f] = os.path.getsize(p) if os.path.exists(p) else None
    bad = {f: (want[f], got[f]) for f in want if want[f] is not None and got[f] != want[f]}
    res = {"repo": REPO, "rev": REV, "dest": DEST, "files": len(want),
           "bytes": sum(v or 0 for v in got.values()), "size_mismatch": bad,
           "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(DEST, "FETCH_MANIFEST.json"), "w") as fh:
        json.dump({"result": res, "sizes": got}, fh, indent=1)
    vol.commit()
    if bad:
        raise RuntimeError(f"size mismatch: {bad}")
    return res


@app.local_entrypoint()
def main():
    print("RESULT", json.dumps(fetch.remote()))
