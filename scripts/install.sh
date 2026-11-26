#!/usr/bin/env bash
# Install the plugin into the Python env that runs vLLM and check the stack.
#   bash scripts/install.sh            # from the repo root (the hfd --local-dir)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

$PY -m pip install --no-deps --no-build-isolation -q .
$PY - <<'EOF'
import importlib.metadata as md, shutil, os, torch
v = md.version("vllm")
print(f"vllm {v} | torch {torch.__version__} | cuda {torch.version.cuda}")
if not v.startswith("0.33."):
    print("WARN: tested only on vLLM 0.33.x; other versions may break the plugin (internal MoE APIs)")
eps = [e.name for e in md.entry_points(group="vllm.general_plugins")]
print("vllm plugins:", eps)
assert "super_nvfp4_w4a8" in eps, "plugin entry point not registered"
nv = shutil.which("nvcc") or (os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"))
print("nvcc:", nv if os.path.exists(nv) else "NOT FOUND -> CUDA decode falls back to Triton (slower prefill)")
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
EOF
# pre-build the CUDA kernels once (cached in $TORCH_EXTENSIONS_DIR, default ~/.cache/torch_extensions)
$PY -c "from super_nvfp4_w4a8.cuda_ops import available; print('cuda ops built:', available())"
