#!/bin/bash
# NPU 속도 벤치마크 — 모델마다 프로세스를 새로 띄웁니다.
# 한 프로세스에서 두 번째 모델을 로드하면 DMA 오류가 납니다.
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$HOME/demo_1127"
PY=./.venv38/bin/python
# 첫 번째 바이너리가 가속 계산의 기준(원본)이 됩니다.
BINS=("$@")
[ ${#BINS[@]} -eq 0 ] && BINS=(orig_egs.o 40_egs.o 50_egs.o)

missing=()
for b in "${BINS[@]}"; do [ -f "$b" ] || missing+=("$b"); done
if [ ${#missing[@]} -gt 0 ]; then
    echo "✗ 바이너리 없음: ${missing[*]}" >&2
    echo "  사용 가능: $(ls *.o 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi

OUT=$(mktemp)
for b in "${BINS[@]}"; do
    echo "── $b ──"
    timeout 600 $PY bench_npu.py "$b" 2>&1 | tee -a "$OUT" | grep -vE "^RESULT"
done
echo
printf "%-12s %9s %8s %12s %14s %10s\n" "모델" "총 ms" "가속" "SendInput" "Launch+Wait" "Receive"
awk -F'\t' '/^RESULT/{if(!b)b=$3; printf "%-12s %9.1f %7.2fx %12.1f %14.1f %10.1f\n",$2,$3,b/$3,$7,$8,$9}' "$OUT"
echo
awk -F'\t' '/^RESULT/{if(!c)c=$8; printf "  %s 연산구간 %.1f ms (원본 대비 %.2fx)\n",$2,$8,c/$8}' "$OUT"
rm -f "$OUT"
