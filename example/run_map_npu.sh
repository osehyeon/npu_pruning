#!/bin/bash
# NPU 실행 기준 COCO mAP — 모델마다 프로세스를 새로 띄웁니다 (DMA 제약)
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$HOME/demo_1127"
PY=./.venv38/bin/python
LIMIT="${LIMIT:-0}"
OUT=$(mktemp)
for t in ${@:-orig_egs 40_egs 50_egs}; do
    timeout 3600 $PY eval_map_npu.py "$t" --limit "$LIMIT" 2>&1 | tee -a "$OUT" | grep -vE "^RESULT"
done
echo
printf "%-10s %7s %7s %7s %7s %7s %7s %9s\n" 모델 AP AP50 AP75 APs APm APl "NPU ms"
awk -F'\t' '/^RESULT/{printf "%-10s %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %9.1f\n",$2,$3,$4,$5,$6,$7,$8,$9}' "$OUT"
rm -f "$OUT"
