#!/bin/bash
# 동일 조건 uint8 세 모델 mAP — 모델당 프로세스 분리
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$HOME/demo_1127"
PY=./.venv38/bin/python
LIMIT="${LIMIT:-0}"
OUT=$(mktemp)
for t in ${@:-original pruned40 pruned50}; do
    $PY eval_u8_final.py "$t" --limit "$LIMIT" 2>&1 | tee -a "$OUT" | grep -vE "^RESULT"
done
echo
printf "%-10s %7s %7s %7s %7s %7s %7s\n" 모델 AP AP50 AP75 APs APm APl
awk -F'\t' '/^RESULT/{printf "%-10s %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f\n",$2,$3,$4,$5,$6,$7,$8}' "$OUT"
rm -f "$OUT"
