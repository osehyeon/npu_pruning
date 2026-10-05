#!/bin/bash
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=$HOME/demo_1127 REPS=200
OUT=grid_latency_hirep.tsv
printf "model\tcombo\ttotal_ms\tstd\tlaunch_wait_ms\n" > "$OUT"
while read -r m c; do
  r=""
  for t in 1 2 3 4 5; do
    r=$(timeout 900 ./.venv38/bin/python measure_hirep.py "$m" "$c" 2>/dev/null | grep "^RESULT")
    [ -n "$r" ] && break; sleep 2
  done
  if [ -n "$r" ]; then echo "$r" | cut -f2- >> "$OUT"; echo "OK $m $c $(echo "$r"|cut -f4,5)"
  else echo "FAIL $m $c"; fi
done <<EOF
sw10sc8 lw
sw10sc8 egshalf
sw10sc16 lwhalf
sw10sc16 egs
sw10sc64 lwhalf
sw10sc64 egshalf
sw30sc8 egs
sw30sc64 egshalf
sw50sc64 lwhalf
sw50sc8 lw
sw50sc64 lw
EOF
echo HIREPDONE
