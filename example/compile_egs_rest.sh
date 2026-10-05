#!/bin/bash
# egs 미확보 9종을 컴파일한다.
# verify 단계에서 예외가 나도 gen_inline_assembly.o 는 이미 생성돼 있으므로 회수한다.
#   1차 실패: util.py 의 skip_nodes 가 원본 모델 이름으로 하드코딩됨 → 런타임 패치
#   2차 실패: verify.py:99 te.inst() 의 pybind 바인딩 결함 → 회수로 우회
set -u
cd "$(dirname "$(readlink -f "$0")")"
U8=$HOME/workspace_docs/model/u8
NB=$HOME/demo_0211/nbc/neubla
KEEP=$PWD/nbc_artifacts/egs
for r in 10 30 50; do for s in 8 16 32 64; do
  tag="sw${r}sc${s}"; out="$PWD/${tag}_egs.o"
  [ -f "$out" ] && { echo "── $tag 이미 확보 · 건너뜀"; continue; }
  echo "════ $tag · egs · $(date '+%m-%d %H:%M:%S') ════"
  rm -f "$NB"/gen_inline_assembly.o
  rm -f "$NB"/NeublaExecutionProvider_*.onnx "$NB"/NeublaExecutionProvider_*_group_*.json
  ./compile_variant_skipconcat.sh "$U8/${tag}_u8_maD20.onnx" input.1 "$out" egs single \
      > "logs_egs_${tag}.log" 2>&1
  # 스크립트가 죽었더라도 작업 디렉터리에 .o 가 남아 있으면 회수
  if [ ! -f "$out" ] && [ -f "$NB"/gen_inline_assembly.o ]; then
    mv "$NB"/gen_inline_assembly.o "$out"
    echo "     ↳ 검증 예외 후 회수"
  fi
  d="$KEEP/$tag"; mkdir -p "$d"
  cp -f "$NB"/NeublaExecutionProvider_*.onnx "$d/" 2>/dev/null
  cp -f "$NB"/NeublaExecutionProvider_*_group_*.json "$d/" 2>/dev/null
  n=$(ls "$d"/*_group_*.json 2>/dev/null | wc -l)
  if [ -f "$out" ]; then
    printf "     → %.2f MB · group %d개 · %s\n" \
      "$(echo "scale=4;$(stat -c%s "$out")/1000000"|bc)" "$n" "$(date '+%H:%M:%S')"
  else
    echo "     ✗ .o 미생성 · group ${n}개 · $(date '+%H:%M:%S')"
  fi
done; done
echo "════ 전체 완료 $(date '+%m-%d %H:%M:%S') ════"
