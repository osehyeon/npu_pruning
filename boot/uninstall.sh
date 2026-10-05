#!/bin/bash
# 부팅 자동 초기화를 제거한다. 설치 전 상태로 되돌린다.
#   sudo /home/osehn/workspace_docs/boot/uninstall.sh
set -e
[ "$(id -u)" = 0 ] || { echo "✗ sudo 로 실행하세요"; exit 1; }
echo "── systemd 유닛 제거 ──"
systemctl disable npu-board-init.service 2>/dev/null || true
systemctl stop npu-board-init.service 2>/dev/null || true
rm -f /etc/systemd/system/npu-board-init.service
systemctl daemon-reload
systemctl reset-failed 2>/dev/null || true
echo "  제거됨"
echo
echo "로그 파일은 남겨둡니다: /var/log/npu-board-init.log"
echo "지우려면: sudo rm -f /var/log/npu-board-init.log"
echo
echo "확인:"
systemctl list-unit-files 2>/dev/null | grep npu-board-init || echo "  유닛 없음 · 정상"
