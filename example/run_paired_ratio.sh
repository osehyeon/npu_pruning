#!/bin/bash
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=$HOME/demo_1127
for c in lw lwhalf egs egshalf; do
  ok=0
  for t in 1 2 3 4 5 6 7 8; do
    ./.venv38/bin/python measure_paired_ratio.py X "$c" 10 20 > "/tmp/pr_${c}.txt" 2>&1
    grep -q "paired t" "/tmp/pr_${c}.txt" && { ok=1; break; }
    sleep 3
  done
  [ $ok -eq 1 ] && echo "OK $c" || echo "FAIL $c"
done
echo RATIODONE
