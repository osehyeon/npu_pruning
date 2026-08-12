#!/bin/bash
# NPU 모드 실행 래퍼
#
#   ./run_npu.sh              → --mode npu
#   ./run_npu.sh both         → --mode both  (CPU 와 비교)
#   ./run_npu.sh npu --image 내사진.jpg
#
# sudo 가 필요한 이유: /dev/plda0 이 crw------- root root 이기 때문입니다.
# 매번 sudo 가 싫으면 아래 udev 규칙을 한 번 설치하면 됩니다.
#
#   sudo cp ~/workdir/antara-software/antara-pcie-sdk/PCIeDriver/51-plda-pcie-udev.rules \
#           /etc/udev/rules.d/
#   sudo udevadm control --reload-rules
#   # 드라이버 재로드 후 /dev/plda0 이 0666 으로 생성됩니다

set -e
cd "$(dirname "$(readlink -f "$0")")"

MODE="${1:-npu}"
shift || true

PY="./.venv38/bin/python"
DEMO="$HOME/demo_1127"

[ -x "$PY" ] || { echo "✗ $PY 없음 — README 의 venv 생성 절차를 먼저 실행하세요"; exit 1; }
[ -d "$DEMO" ] || { echo "✗ $DEMO 없음"; exit 1; }

# /dev/plda0 을 이미 열 수 있으면 sudo 없이 실행
if [ -r /dev/plda0 ]; then
    echo "[ wrapper] /dev/plda0 접근 가능 — sudo 없이 실행합니다"
    exec env PYTHONPATH="$DEMO" "$PY" run_yolov3.py --mode "$MODE" "$@"
fi

echo "[ wrapper] /dev/plda0 이 root 전용이라 sudo 로 실행합니다"
# sudo 는 PYTHONPATH 를 지우므로 명시적으로 전달합니다
exec sudo PYTHONPATH="$DEMO" "$(readlink -f "$PY")" run_yolov3.py --mode "$MODE" "$@"
