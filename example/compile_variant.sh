#!/bin/bash
# 양자화된 uint8 ONNX 를 원하는 nbc 옵션 조합으로 컴파일합니다.
#
#   ./compile_variant.sh <uint8.onnx> <입력텐서이름> <출력.o> [groupsearch] [tx_ofm_step]
#
# groupsearch : none | lw | egs        (기본 none = built-in group)
# tx_ofm_step : single | half          (기본 single)
#
# 예)
#   ./compile_variant.sh ~/demo_0211/pruning_model/40.onnx input_1 ~/40_lwhalf.o lw half
#
# 주의: groupsearch=none 은 컴파일러에 **모델별 그룹 설정이 하드코딩**되어 있어야 합니다
#       (getBestGroup_yolov3 / getBestGroup_yolov3Reduced 두 개뿐).
#       모르는 이름이면 "config is not yet implemented(NYI)" 로 죽습니다.
#       6번째 인자로 이름을 지정할 수 있습니다.
#
# 근거: demo_pruning/compile.sh + demo_0211/nbc/neubla/{nn_test,util}.py
set -e

MODEL=$(readlink -f "$1")
IN_NAME="${2:-input_1}"
OUT=$(readlink -f "$3")
GS="${4:-none}"
STEP="${5:-single}"
NAME="${6:-variant}"

NBC_ROOT="$HOME/demo_0211/nbc"
[ -f "$MODEL" ] || { echo "✗ 모델 없음: $MODEL"; exit 1; }

# nn_test.py 를 건드리지 않고 옵션을 바꾸기 위해 임시 러너를 만듭니다
RUNNER="$NBC_ROOT/neubla/_variant_run.py"
cat > "$RUNNER" <<EOF
import util, sys
gs = None if "$GS" == "none" else "$GS"
util.Inference("$NAME", sys.argv[1], {sys.argv[2]: sys.argv[3]},
               gs=gs, tx_ofm_step="$STEP")
EOF

echo "── 컴파일: groupsearch=$GS · tx_ofm_step=$STEP · name=$NAME"
echo "   모델: $MODEL"

cd "$NBC_ROOT/dev"
# shellcheck disable=SC1091
source set_env.sh >/dev/null 2>&1

cd "$NBC_ROOT/neubla"
INPUT_BIN="$NBC_ROOT/models/vision/object_detection/yolov3/input.bin"
rm -f gen_inline_assembly.o
python3 _variant_run.py "$MODEL" "$IN_NAME" "$INPUT_BIN"

[ -f gen_inline_assembly.o ] || { echo "✗ gen_inline_assembly.o 가 생성되지 않았습니다"; exit 1; }
mv gen_inline_assembly.o "$OUT"
rm -f "$RUNNER"
echo "✓ $OUT  ($(du -h "$OUT" | cut -f1))"
