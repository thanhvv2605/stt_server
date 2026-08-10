#!/usr/bin/env bash
set -euo pipefail

AUDIO_FILE="${1:-sample.wav}"
N="${2:-4}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

for i in $(seq 1 "$N"); do
  curl -s -X POST "$BASE_URL/v1/audio/transcriptions" \
    -H "Authorization: Bearer $STT_API_KEY" \
    -F "file=@${AUDIO_FILE}" \
    -F "response_format=verbose_json" \
    -o "result_${i}.json" &
done
wait

echo "Requests completed. Checking for non-empty results:"
for i in $(seq 1 "$N"); do
  size=$(wc -c < "result_${i}.json")
  echo "result_${i}.json: ${size} bytes"
done
