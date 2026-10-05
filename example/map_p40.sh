#!/bin/bash
# 재학습 40 % 모델의 조합별 mAP. LoadModel 간헐 실패에 재시도.
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=$HOME/demo_1127
for t in p40_lw p40_lwhalf p40_egs p40_egshalf; do
  ok=0
  for k in 1 2 3 4 5 6; do
    out=$(timeout 3000 ./.venv38/bin/python eval_map_npu.py "$t" --limit 500 2>&1 | grep "^RESULT")
    if [ -n "$out" ]; then
      ap=$(echo "$out" | cut -f3)
      if [ "$(echo "$ap > 1" | bc 2>/dev/null || echo 0)" = 1 ]; then
        echo "$out"; ok=1; break
      fi
    fi
    sleep 3
  done
  [ $ok = 0 ] && echo "NG $t"
done
echo "DONE $(date '+%H:%M:%S')"
