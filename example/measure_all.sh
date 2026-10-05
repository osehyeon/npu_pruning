#!/bin/bash
# 확보된 바이너리를 전수 측정한다. 모델당 프로세스를 새로 띄우고 크래시에 재시도.
# 결과: grid_latency.tsv
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=$HOME/demo_1127
OUT=grid_latency.tsv
[ -f "$OUT" ] || printf "model\tcombo\ttotal_ms\tstd\tlaunch_wait_ms\n" > "$OUT"
for m in original sw50sc8 sw50sc16 sw50sc32 sw50sc64 \
         sw30sc8 sw30sc16 sw30sc32 sw30sc64 \
         sw10sc8 sw10sc16 sw10sc32 sw10sc64; do
  for c in lw lwhalf egs egshalf; do
    grep -qP "^$m\t$c\t" "$OUT" && continue
    s=$m; [ "$m" = original ] && [ "$c" = lw ] && s=original_maD20
    [ -f "${s}_${c}.o" ] || { echo "· $m $c 바이너리 없음"; continue; }
    for t in 1 2 3 4; do
      r=$(timeout 400 ./.venv38/bin/python measure_grid.py "$m" --combo "$c" 2>/dev/null | grep "^RESULT")
      [ -n "$r" ] && break
      sleep 2
    done
    if [ -n "$r" ]; then
      echo "$r" | cut -f2- >> "$OUT"
      echo "✓ $(echo "$r" | awk -F'\t' '{printf "%-10s %-8s %7.2f ms", $2,$3,$4}')"
    else
      echo "✗ $m $c 4회 실패"
    fi
  done
done
echo "════ 측정 완료 $(date '+%m-%d %H:%M:%S') ════"
