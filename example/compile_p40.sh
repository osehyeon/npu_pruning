#!/bin/bash
# 재학습 40 % 모델의 나머지 세 조합. 입력 텐서 이름이 images 로 다르다.
set -u
cd "$(dirname "$(readlink -f "$0")")"
U8=$HOME/workspace_docs/model/u8/p40ep10_u8_maD20.onnx
NB=$HOME/demo_0211/nbc/neubla
KEEP=$PWD/nbc_artifacts
run() {
  local gs=$1 step=$2 sfx=$3
  local out="$PWD/p40ep10_${sfx}.o"
  [ -f "$out" ] && { echo "-- $sfx 이미 있음"; return; }
  echo "==== p40ep10 $gs $step $(date '+%m-%d %H:%M:%S') ===="
  rm -f "$NB"/gen_inline_assembly.o "$NB"/NeublaExecutionProvider_*
  ./compile_variant_skipconcat.sh "$U8" images "$out" "$gs" "$step" > "logs_p40_${sfx}.log" 2>&1
  [ ! -f "$out" ] && [ -f "$NB"/gen_inline_assembly.o ] && { mv "$NB"/gen_inline_assembly.o "$out"; echo "   회수"; }
  d="$KEEP/${sfx}/p40ep10"; mkdir -p "$d"
  cp -f "$NB"/NeublaExecutionProvider_* "$d/" 2>/dev/null
  [ -f "$out" ] && printf "   -> %.2f MB  %s\n" "$(echo "scale=4;$(stat -c%s "$out")/1000000"|bc)" "$(date '+%H:%M:%S')" \
                || echo "   x 미생성"
}
run lw   half   lwhalf
run egs  half   egshalf
run egs  single egs
echo "DONE $(date '+%m-%d %H:%M:%S')"
