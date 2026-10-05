#!/bin/bash
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=$HOME/demo_1127
OUT=grid_latency.tsv
while read -r m c; do
  [ -z "${m:-}" ] && continue
  best=""
  for t in 1 2 3 4 5; do
    r=$(timeout 400 ./.venv38/bin/python measure_grid.py "$m" --combo "$c" 2>/dev/null | grep "^RESULT")
    if [ -n "$r" ]; then
      sd=$(echo "$r" | cut -f5)
      ok=$(echo "$sd < 3" | bc 2>/dev/null || echo 0)
      best="$r"
      [ "$ok" = 1 ] && break
    fi
    sleep 2
  done
  if [ -n "$best" ]; then
    grep -v -P "^$m\t$c\t" "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
    echo "$best" | cut -f2- >> "$OUT"
    echo "$best" | awk -F'\t' '{printf "OK %-10s %-8s %7.2f ms  sd %.2f\n",$2,$3,$4,$5}'
  else
    echo "NG $m $c"
  fi
done < todo.txt
echo "DONE $(date '+%H:%M:%S')"
