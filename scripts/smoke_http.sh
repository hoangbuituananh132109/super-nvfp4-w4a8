#!/usr/bin/env bash
# HTTP smoke test against a running server:  PORT=8000 bash scripts/smoke_http.sh
set -euo pipefail
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8000}
URL="http://$HOST:$PORT"
MODEL=$(curl -s "$URL/v1/models" | python -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "model: $MODEL"
echo "--- chat (reasoning off)"
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 64,
  \"messages\": [{\"role\": \"user\", \"content\": \"Only output the number: 17 + 25.\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": false}}" | python -c "import sys, json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'], '| usage', r.get('usage'))"
echo "--- chat (reasoning on)"
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 512,
  \"messages\": [{\"role\": \"user\", \"content\": \"Một năm có bao nhiêu ngày? Trả lời ngắn.\"}]}" | python -c "
import sys, json; m=json.load(sys.stdin)['choices'][0]['message']
print('reasoning:', (m.get('reasoning_content') or m.get('reasoning') or '')[:300].replace(chr(10),' '))
print('answer   :', m.get('content'))"
echo "--- completions"
curl -s "$URL/v1/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 8, \"temperature\": 0}" | python -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])"
