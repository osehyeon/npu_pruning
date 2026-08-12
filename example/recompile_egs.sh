#!/bin/bash
# egs 로 재컴파일 — nbc 작업 디렉토리를 공유하므로 반드시 순차 실행
cd "$(dirname "$(readlink -f "$0")")"
set -u
run() {
  echo "════ $1  ($(date '+%H:%M:%S')) ════"
  ./compile_variant.sh "$2" "$3" "$PWD/$4" egs single 2>&1 | tail -3
  echo "     → $(ls -la "$4" 2>/dev/null | awk '{printf "%.1f MB", $5/1e6}' || echo '실패')  $(date '+%H:%M:%S')"
}
run "50% 프루닝" "$HOME/demo_0211/pruning_model/50.onnx" input_1 50_egs.o
run "원본"       "$HOME/demo_1127/yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx" input orig_egs.o
echo "════ 전체 완료 $(date '+%H:%M:%S') ════"
