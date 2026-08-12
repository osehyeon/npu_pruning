#!/bin/bash
# 전체 COCO val2017 (5,000장) 평가 — 논문 수치와 직접 대조하기 위한 측정
cd "$(dirname "$(readlink -f "$0")")"
PY=$HOME/miniforge3/envs/nota/bin/python
echo "════ fp32 시작 $(date '+%H:%M:%S') ════"
$PY eval_map.py    --limit 0 --out map_full_fp32.json
echo "════ uint8 시작 $(date '+%H:%M:%S') ════"
$PY eval_map_u8.py --limit 0 --out map_full_u8.json
echo "════ 전체 완료 $(date '+%H:%M:%S') ════"
