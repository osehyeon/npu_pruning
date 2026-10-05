#!/bin/bash
# 격자의 남은 칸을 채운다.
#   egs·half   30% 세 종 + 10% 네 종
#   egs·single original 1종
# nbc 작업 디렉터리를 공유하므로 순차 실행.
set -u
cd "$(dirname "$(readlink -f "$0")")"
U8=$HOME/workspace_docs/model/u8
NB=$HOME/demo_0211/nbc/neubla
KEEP=$PWD/nbc_artifacts

run() {   # $1=tag $2=gs $3=step
  local tag=$1 gs=$2 step=$3
  local sfx=$gs; [ "$step" = half ] && sfx="${gs}half"
  local out="$PWD/${tag}_${sfx}.o"
  [ -f "$out" ] && { echo "── $tag · $sfx 이미 있음"; return; }
  echo "════ $tag · $gs·$step · $(date '+%m-%d %H:%M:%S') ════"
  rm -f "$NB"/gen_inline_assembly.o
  rm -f "$NB"/NeublaExecutionProvider_*.onnx "$NB"/NeublaExecutionProvider_*_group_*.json
  ./compile_variant_skipconcat.sh "$U8/${tag}_u8_maD20.onnx" input.1 "$out" "$gs" "$step" \
      > "logs_${sfx}_${tag}.log" 2>&1
  [ ! -f "$out" ] && [ -f "$NB"/gen_inline_assembly.o ] && {
    mv "$NB"/gen_inline_assembly.o "$out"; echo "     ↳ 검증 예외 후 회수"; }
  local d="$KEEP/$sfx/$tag"; mkdir -p "$d"
  cp -f "$NB"/NeublaExecutionProvider_*.onnx "$d/" 2>/dev/null
  cp -f "$NB"/NeublaExecutionProvider_*_group_*.json "$d/" 2>/dev/null
  local n=$(ls "$d"/*_group_*.json 2>/dev/null | wc -l)
  [ -f "$out" ] \
    && printf "     → %.2f MB · group %d개 · %s\n" \
         "$(echo "scale=4;$(stat -c%s "$out")/1000000"|bc)" "$n" "$(date '+%H:%M:%S')" \
    || echo "     ✗ 미생성 · group ${n}개 · $(date '+%H:%M:%S')"
}

# 우선순위 — 30% 를 먼저 채워 압축률 축을 완성
for t in sw30sc64 sw30sc8 sw30sc32 sw30sc16; do run "$t" egs half; done
for t in sw10sc64 sw10sc8 sw10sc32 sw10sc16; do run "$t" egs half; done
run original egs single
echo "════ 전체 완료 $(date '+%m-%d %H:%M:%S') ════"
