#!/usr/bin/env bash
set -euo pipefail

AUDIO_FILE="${1:-sample.wav}"
N="${2:-4}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

SECONDS=0
for i in $(seq 1 "$N"); do
  curl -s -X POST "$BASE_URL/v1/audio/transcriptions" \
    -H "Authorization: Bearer $STT_API_KEY" \
    -F "file=@${AUDIO_FILE}" \
    -F "response_format=verbose_json" \
    -o "result_${i}.json" \
    -w "%{http_code}" \
    > "status_${i}.txt" &
done
wait
elapsed=$SECONDS

echo "Requests completed in ${elapsed}s. Results:"
failures=0
for i in $(seq 1 "$N"); do
  size=$(wc -c < "result_${i}.json")
  status=$(cat "status_${i}.txt")
  if [ "$status" = "200" ]; then
    echo "result_${i}.json: ${size} bytes, status ${status} (OK)"
  else
    echo "result_${i}.json: ${size} bytes, status ${status} (FAIL)"
    failures=$((failures + 1))
  fi
done

echo "---"
echo "Total wall-clock time for ${N} concurrent requests: ${elapsed}s"
if [ "$failures" -gt 0 ]; then
  echo "FAIL: ${failures}/${N} requests did not return 200"
  exit 1
fi
echo "All ${N} requests returned 200 OK"
