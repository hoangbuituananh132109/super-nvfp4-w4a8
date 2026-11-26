#!/usr/bin/env bash
# Host Nemotron-3 Super NVFP4 with vLLM: one TP1 replica per GPU (the plugin is TP1/EP1 only).
#
#   MODEL_PATH=/path/to/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 GPUS=0,1 VARIANT=auto bash scripts/serve.sh
#
# VARIANT=w4a8   : this repo (--quantization super_nvfp4_w4a8, hybrid Marlin / FP8 prefill)
# VARIANT=auto   : vLLM's default (Humming W4A16 on H200), recommended for production
# VARIANT=marlin : vLLM NVFP4 W4A16 with Marlin forced
# Replica i listens on BASE_PORT + i; logs go to $LOG_DIR/serve_<variant>_gpu<id>.log
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH to the ModelOpt NVFP4 checkpoint}"
VARIANT=${VARIANT:-auto}
GPUS=${GPUS:-0}
BASE_PORT=${BASE_PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Nemotron-3-Super-120B-A12B-NVFP4}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH}
# the CUDA kernels are JIT-built once into this cache; it must be writable (set it if HOME is read-only)
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}
GPU_MEM=${GPU_MEM:-0.95}
MAX_BATCHED=${MAX_BATCHED:-16384}
MAX_SEQS=${MAX_SEQS:-512}
MAX_LEN=${MAX_LEN:-auto}
KV_DTYPE=${KV_DTYPE:-auto}
LOG_DIR=${LOG_DIR:-./logs}
mkdir -p "$LOG_DIR"

case "$VARIANT" in
  w4a8)   QARGS=(--quantization super_nvfp4_w4a8) ;;
  marlin) QARGS=(--moe-backend marlin) ;;
  auto)   QARGS=() ;;
  *) echo "VARIANT must be w4a8, auto or marlin"; exit 1 ;;
esac

IFS=',' read -ra GPU_LIST <<< "$GPUS"
pids=()
for i in "${!GPU_LIST[@]}"; do
  g=${GPU_LIST[$i]}; port=$((BASE_PORT + i))
  log="$LOG_DIR/serve_${VARIANT}_gpu${g}.log"
  echo "starting $VARIANT on GPU $g -> http://127.0.0.1:$port  (log: $log)"
  CUDA_VISIBLE_DEVICES=$g VLLM_SERVER_DEV_MODE=1 \
  vllm serve "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    "${QARGS[@]}" \
    --gpu-memory-utilization "$GPU_MEM" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --kv-cache-dtype "$KV_DTYPE" \
    --port "$port" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --enable-tokenizer-info-endpoint \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser nemotron_v3 \
    --enable-prefix-caching \
    ${EXTRA_ARGS:-} > "$log" 2>&1 &
  pids+=($!)
done
echo "pids: ${pids[*]}   (stop: kill ${pids[*]})"
echo "readiness: curl -s http://127.0.0.1:$BASE_PORT/health ; the plugin logs SUPER_NVFP4_W4A8_READY in the log"
wait
