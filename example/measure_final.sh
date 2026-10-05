#!/bin/bash
# 전 조합 측정. 40회 반복 중 중앙값의 10배를 넘는 회차는 호스트 대기로 보고 제외한다.
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=$HOME/demo_1127
OUT=grid_latency.tsv
printf "model\tcombo\ttotal_ms\tstd\tlaunch_wait_ms\n" > "$OUT"
for m in original sw10sc8 sw10sc16 sw10sc32 sw10sc64 \
         sw30sc8 sw30sc16 sw30sc32 sw30sc64 \
         sw50sc8 sw50sc16 sw50sc32 sw50sc64; do
  for c in lw lwhalf egs egshalf; do
    r=""
    for t in 1 2 3 4 5; do
      r=$(timeout 400 ./.venv38/bin/python measure_filtered.py "$m" "$c" 2>/dev/null | grep "^RESULT")
      [ -n "$r" ] && break
      sleep 2
    done
    if [ -n "$r" ]; then
      echo "$r" | cut -f2- >> "$OUT"
      echo "$r" | awk -F'\t' '{printf "OK %-10s %-8s %7.2f  sd %.2f\n",$2,$3,$4,$5}'
    else echo "NG $m $c"; fi
  done
done
echo "DONE $(date '+%H:%M:%S')"
