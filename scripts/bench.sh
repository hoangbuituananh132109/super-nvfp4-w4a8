#!/usr/bin/env bash
# Throughput / latency of one running server (same workloads as the README tables):
#   PORT=8000 TAG=w4a8 TOKENIZER_PATH=/path/to/ckpt bash scripts/bench.sh
# Results: $OUT_DIR/<TAG>/<workload>.json, plus a summary line per workload.
set -euo pipefail
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8000}; TAG=${TAG:-run}
OUT_DIR=${OUT_DIR:-./bench_results}
: "${TOKENIZER_PATH:?set TOKENIZER_PATH (usually the checkpoint dir)}"
URL="http://$HOST:$PORT"
MODEL=$(curl -s "$URL/v1/models" | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
mkdir -p "$OUT_DIR/$TAG"
# name  in   out  prompts  concurrency(0 = unbounded)
WORKLOADS=${WORKLOADS:-"prefill:8192:1:64:0 mixed:1024:256:256:0 decode:128:512:128:128 latency1:512:256:8:1"}
for w in $WORKLOADS; do
  IFS=: read -r name il ol n conc <<< "$w"
  extra=(); [ "$conc" != 0 ] && extra=(--max-concurrency "$conc")
  vllm bench serve --backend vllm --base-url "$URL" --model "$MODEL" --tokenizer "$TOKENIZER_PATH" \
    --trust-remote-code --dataset-name random --random-input-len "$il" --random-output-len "$ol" \
    --num-prompts "$n" --request-rate inf --ignore-eos --seed 1234 \
    --save-result --result-dir "$OUT_DIR/$TAG" --result-filename "$name.json" "${extra[@]}" > "$OUT_DIR/$TAG/$name.log" 2>&1
  python3 - "$OUT_DIR/$TAG/$name.json" "$name" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]:9s} total_tok/s={d['total_token_throughput']:9.0f}  out_tok/s={d['output_throughput']:8.0f}  "
      f"TTFT_med={d['median_ttft_ms']:8.0f}ms  TPOT_med={d['median_tpot_ms']:6.2f}ms")
EOF
done
