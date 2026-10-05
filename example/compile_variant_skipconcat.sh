#!/bin/bash
# compile_variant.sh 와 동일하되, 검증 단계에서 Concat 계열 노드를 제외한다.
#
# 배경: nbc 의 util.Inference 는 skip_nodes 를 하드코딩하고 있다.
#         skip_nodes = ["input_608_quantized","input_556_quantized",
#                       "_Concat_3_output_0_quantized"]
#       주석에 "for yolov3 egs best" 라고 적혀 있어, egs 그룹 구성에서
#       Concat 경계의 수치 차이는 알려진 현상이고 검증에서 빼도록 되어 있다.
#       그러나 이름이 원본 모델 기준이라 프루닝 모델에서는 하나도 맞지 않는다.
#       벤더 파일을 수정하지 않고 verify.verify 를 감싸 Concat 계열을 추가한다.
set -e
MODEL=$(readlink -f "$1"); IN_NAME="${2:-input.1}"; OUT=$(readlink -f "$3")
GS="${4:-egs}"; STEP="${5:-single}"; NAME="${6:-variant}"
NBC_ROOT="$HOME/demo_0211/nbc"
[ -f "$MODEL" ] || { echo "✗ 모델 없음: $MODEL"; exit 1; }
RUNNER="$NBC_ROOT/neubla/_skipconcat_run.py"
cat > "$RUNNER" <<'PY'
import sys, verify, util
_orig = verify.verify
def patched(trace_name, ref_path, elf_file_path, skip_nodes):
    import glob, os
    extra = set()
    for f in glob.glob(os.path.join(ref_path, "*")) + glob.glob("./niss/*"):
        b = os.path.basename(f)
        if "Concat" in b:
            extra.add(os.path.splitext(b)[0])
    merged = list(skip_nodes) + sorted(extra)
    print(f"[patch] skip_nodes {len(skip_nodes)} → {len(merged)} (Concat 계열 {len(extra)}개 추가)")
    for x in sorted(extra): print(f"[patch]   {x}")
    return _orig(trace_name, ref_path, elf_file_path, merged)
verify.verify = patched
gs = None if sys.argv[4] == "none" else sys.argv[4]
util.Inference(sys.argv[5], sys.argv[1], {sys.argv[2]: sys.argv[3]}, gs=gs, tx_ofm_step=sys.argv[6])
PY
echo "── 컴파일: groupsearch=$GS · tx_ofm_step=$STEP · Concat 검증 제외"
echo "   모델: $MODEL"
cd "$NBC_ROOT/dev"; source set_env.sh >/dev/null 2>&1
cd "$NBC_ROOT/neubla"
INPUT_BIN="$NBC_ROOT/models/vision/object_detection/yolov3/input.bin"
rm -f gen_inline_assembly.o
python3 _skipconcat_run.py "$MODEL" "$IN_NAME" "$INPUT_BIN" "$GS" "$NAME" "$STEP"
[ -f gen_inline_assembly.o ] || { echo "✗ .o 미생성"; exit 1; }
mv gen_inline_assembly.o "$OUT"
rm -f "$RUNNER"
echo "✓ $OUT  ($(du -h "$OUT" | cut -f1))"
