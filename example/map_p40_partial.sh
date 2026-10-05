#!/bin/bash
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=$HOME/demo_1127
for t in "$@"; do
  for k in 1 2 3 4 5 6; do
    out=$(timeout 3000 ./.venv38/bin/python eval_map_npu.py "$t" --limit 500 2>&1 | grep "^RESULT")
    if [ -n "$out" ]; then
      ap=$(echo "$out" | cut -f3)
      [ "$(echo "$ap > 1" | bc 2>/dev/null || echo 0)" = 1 ] && { echo "$out"; break; }
    fi
    sleep 3
  done
done
echo "DONE $(date '+%H:%M:%S')"
