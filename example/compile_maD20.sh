#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")"
U8=$HOME/workspace_docs/model/u8
for spec in "original:input.1" "pruned40:input_1" "pruned50:input_1"; do
  m="${spec%%:*}"; inp="${spec##*:}"
  echo "════ $m ($(date '+%H:%M:%S')) ════"
  ./compile_variant.sh "$U8/${m}_u8_maD20.onnx" "$inp" "$PWD/${m}_maD20_lw.o" lw single 2>&1 | tail -2
done
echo "════ 완료 $(date '+%H:%M:%S') ════"
