#!/bin/bash
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=$HOME/demo_1127
OUT=paired_summary.tsv
printf "ratio\tcombo\tsc\tmean_ms\tsample_std\tround_std\tsem\n" > "$OUT"
for r in 10 30 50; do
  for c in lw lwhalf egs egshalf; do
    ok=0
    for t in 1 2 3 4 5; do
      if ./.venv38/bin/python measure_paired.py "$r" "$c" 10 20 > "/tmp/p_${r}_${c}.txt" 2>&1; then
        grep -q "paired t" "/tmp/p_${r}_${c}.txt" && { ok=1; break; }
      fi
      sleep 3
    done
    if [ $ok -eq 1 ]; then
      awk -v r="$r" -v c="$c" '/^ *(8|16|32|64) /{printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",r,c,$1,$2,$3,$4,$5}' \
        "/tmp/p_${r}_${c}.txt" >> "$OUT"
      echo "OK  $r% $c  $(grep 'sc64 ' /tmp/p_${r}_${c}.txt | tail -1)"
    else
      echo "FAIL $r% $c"
    fi
  done
done
echo PAIREDDONE
