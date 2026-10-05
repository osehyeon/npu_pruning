#!/bin/bash
# tx_ofm_step=half 격자 완성.
#   A단계  lw · half    모델당 3~4분 · 원본 포함 13종
#   B단계  egs · half   모델당 1.5~2.5시간 · 필수 순서대로
# nbc 작업 디렉터리를 공유하므로 반드시 순차 실행.
set -u
cd "$(dirname "$(readlink -f "$0")")"
U8=$HOME/workspace_docs/model/u8
NB=$HOME/demo_0211/nbc/neubla
KEEP=$PWD/nbc_artifacts

# 우선순위 — 원본이 기준이고, 50%가 정렬 대비가 가장 큼
ORDER="original sw50sc64 sw50sc8 sw50sc32 sw50sc16 sw30sc64 sw30sc8 sw30sc32 sw30sc16 sw10sc64 sw10sc8 sw10sc32 sw10sc16"

run() {                       # $1=tag  $2=gs
  local tag=$1 gs=$2
  local u8="$U8/${tag}_u8_maD20.onnx"
  local out="$PWD/${tag}_${gs}half.o"
  [ -f "$u8" ] || { echo "  ✗ $tag 모델 없음"; return; }
  [ -f "$out" ] && { echo "── $tag · $gs·half 이미 있음"; return; }
  echo "════ $tag · $gs·half · $(date '+%m-%d %H:%M:%S') ════"
  rm -f "$NB"/gen_inline_assembly.o
  rm -f "$NB"/NeublaExecutionProvider_*.onnx "$NB"/NeublaExecutionProvider_*_group_*.json
  ./compile_variant_skipconcat.sh "$u8" input.1 "$out" "$gs" half \
      > "logs_${gs}half_${tag}.log" 2>&1
  if [ ! -f "$out" ] && [ -f "$NB"/gen_inline_assembly.o ]; then
    mv "$NB"/gen_inline_assembly.o "$out"; echo "     ↳ 검증 예외 후 회수"
  fi
  local d="$KEEP/${gs}half/$tag"; mkdir -p "$d"
  cp -f "$NB"/NeublaExecutionProvider_*.onnx "$d/" 2>/dev/null
  cp -f "$NB"/NeublaExecutionProvider_*_group_*.json "$d/" 2>/dev/null
  local n=$(ls "$d"/*_group_*.json 2>/dev/null | wc -l)
  if [ -f "$out" ]; then
    printf "     → %.2f MB · group %d개 · %s\n" \
      "$(echo "scale=4;$(stat -c%s "$out")/1000000"|bc)" "$n" "$(date '+%H:%M:%S')"
  else
    echo "     ✗ .o 미생성 · group ${n}개 · $(date '+%H:%M:%S')"
  fi
}

echo "########## A단계 · lw·half 13종 ##########"
for t in $ORDER; do run "$t" lw; done
echo "########## B단계 · egs·half 13종 ##########"
for t in $ORDER; do run "$t" egs; done
echo "════ 전체 완료 $(date '+%m-%d %H:%M:%S') ════"
