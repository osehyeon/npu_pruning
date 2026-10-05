#!/bin/bash
# 부팅 자동 초기화를 설치한다. root 권한이 필요하다.
#   sudo /home/osehn/workspace_docs/boot/install.sh
# 원복은 uninstall.sh
set -e
[ "$(id -u)" = 0 ] || { echo "✗ sudo 로 실행하세요"; exit 1; }
B=/home/osehn/workspace_docs/boot
echo "── systemd 유닛 설치 ──"
cp -f "$B/npu-board-init.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable npu-board-init.service
echo "  enabled"
echo
echo "── 지금 한 번 실행 ──"
systemctl start npu-board-init.service || true
sleep 3
systemctl --no-pager --lines=0 status npu-board-init.service | head -4
echo
echo "설치 완료. 로그: /var/log/npu-board-init.log"
echo "원복: sudo $B/uninstall.sh"
