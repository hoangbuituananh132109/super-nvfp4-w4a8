"""Modal checks for the vLLM PR branch (work/vllm-main, `nvfp4-marlin-fp8-prefill`).

    modal run modal_runs/vllm_pr.py --stage lint     # CPU: pre-commit on the changed files
    modal run modal_runs/vllm_pr.py --stage test     # H200: pytest of the new test file

The GPU stage overlays the branch's changed Python files on the pinned vLLM nightly
image. The image commit and the branch base differ, but every file the branch
touches is byte-identical between the two (checked locally before running).
"""
import json
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VLLM_SRC = (HERE.parent.parent.parent / "vllm-main").resolve()
NIGHTLY = "vllm/vllm-openai:nightly-c2cc003c5b644095cabcd5212d2967546ce344f8"
BASE_REF = "13c850cb622ffb97a067863f470486de8fdfa5a7"

app = modal.App("vllm-pr-marlin-fp8-prefill")

lint_image = (modal.Image.debian_slim(python_version="3.12")
              .apt_install("git", "curl")
              .pip_install("pre-commit", "uv")
              .add_local_dir(str(VLLM_SRC), "/src/vllm", copy=True))

gpu_image = (modal.Image.from_registry(NIGHTLY).entrypoint([])
             .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python",
                           "python3 -m pip install -q 'pytest>=9.2' pytest-asyncio pytest-timeout pytest-rerunfailures tblib pillow")
             .add_local_dir(str(VLLM_SRC / "vllm"), "/src/vllm/vllm", copy=True)
             .add_local_dir(str(VLLM_SRC / "tests"), "/src/vllm/tests", copy=True)
             .add_local_file(str(VLLM_SRC / "pyproject.toml"), "/src/vllm/pyproject.toml", copy=True))


def sh(cmd, cwd="/src/vllm", timeout=3000, env=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, **(env or {})})
    return {"cmd": cmd, "rc": r.returncode, "out": (r.stdout + r.stderr)[-60000:]}


@app.function(image=lint_image, cpu=8, memory=16384, timeout=2400)
def lint(files):
    # the tree was copied from a Windows checkout (CRLF): restore HEAD's LF blobs first
    out = [sh("git config --global --add safe.directory '*' && git config core.autocrlf false && "
              "git rm -q --cached -r . && git reset -q --hard HEAD && git status --short | head"),
           sh("pre-commit install-hooks", timeout=2000),
           sh("pre-commit run --files " + " ".join(files), timeout=2000),
           sh("git diff --stat"),
           sh("git diff")]
    return out


@app.function(image=gpu_image, gpu="H200", cpu=8, memory=65536, timeout=3000)
def gpu_test(changed, pytest_args):
    import vllm
    site = Path(vllm.__file__).parent
    res = {"site": str(site)}
    # overlay the branch's changed vllm/ files on the installed package
    for rel in changed:
        if rel.startswith("vllm/"):
            src = Path("/src/vllm") / rel
            dst = site / rel[len("vllm/"):]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    res["overlaid"] = [r for r in changed if r.startswith("vllm/")]
    res["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")["out"]
    # run from a dir where `import vllm` resolves to the installed (overlaid) package
    res["pytest"] = sh(f"cd /src/vllm && mv vllm vllm_src_unused && python3 -m pytest {pytest_args}",
                       cwd="/", timeout=2800)
    return res


@app.local_entrypoint()
def main(stage: str = "test", pytest_args: str = "tests/kernels/moe/test_marlin_fp8_prefill_moe.py -v"):
    changed = subprocess.run(["git", "diff", "--name-only", BASE_REF, "HEAD"], cwd=VLLM_SRC, capture_output=True,
                             text=True).stdout.split()
    changed += subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=VLLM_SRC,
                              capture_output=True, text=True).stdout.split()
    print("changed:", changed)
    if stage == "lint":
        r = lint.remote(changed)
    else:
        r = gpu_test.remote(changed, pytest_args)
    out = HERE / f"vllm_pr_{stage}.json"
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1)[-9000:])
