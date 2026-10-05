#!/bin/bash
# 스윕 12종을 컴파일하면서 nbc 중간 산출물을 모델별로 보관한다.
#   ./compile_sweep_keep.sh <lw|egs>
# nbc 작업 디렉터리를 공유하므로 반드시 순차 실행.
set -u
cd "$(dirname "$(readlink -f "$0")")"
GS="${1:-lw}"
U8=$HOME/workspace_docs/model/u8
NB=$HOME/demo_0211/nbc/neubla
KEEP=$HOME/workspace_docs/example/nbc_artifacts/$GS
mkdir -p "$KEEP"
for r in 10 30 50; do for s in 8 16 32 64; do
  tag="sw${r}sc${s}"
  out="$PWD/${tag}_${GS}.o"
  echo "════ $tag · $GS · $(date '+%H:%M:%S') ════"
  rm -f "$NB"/NeublaExecutionProvider_*.onnx "$NB"/NeublaExecutionProvider_*_group_*.json
  if ./compile_variant.sh "$U8/${tag}_u8_maD20.onnx" input.1 "$out" "$GS" single 2>&1 | tail -2; then
    d="$KEEP/$tag"; mkdir -p "$d"
    cp -f "$NB"/NeublaExecutionProvider_*.onnx "$d/" 2>/dev/null
    cp -f "$NB"/NeublaExecutionProvider_*_group_*.json "$d/" 2>/dev/null
    n=$(ls "$d"/*_group_*.json 2>/dev/null | wc -l)
    sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    printf "     → %.2f MB · group %d개 · %s\n" "$(echo "scale=4;$sz/1000000"|bc)" "$n" "$(date '+%H:%M:%S')"
  else
    echo "     ✗ 실패"
  fi
done; done
echo "════ 전체 완료 $(date '+%H:%M:%S') ════"
