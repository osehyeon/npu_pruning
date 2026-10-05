#!/bin/bash
# 재부팅 후 보드 복구를 한 번에 수행한다.
#   ~/workspace_docs/after-reboot.sh
#
# 하는 일
#   1. PLDA 드라이버 적재                  o_dbg=0
#   2. UART 로 dbgprint uart off      측정 왜곡 방지 · 반드시 필요
#   3. 상태 확인
#
# 순서가 중요하다. 드라이버 적재가 dbgprint 설정을 초기화하므로
# UART 를 먼저 끄면 무효가 된다.
#
# 시리얼과 드라이버 모두 sudo 가 필요해 처음 한 번 암호를 묻는다.
set -u
TTY=/dev/ttyUSB0
echo "═══ 재부팅 후 복구 · $(date '+%m-%d %H:%M:%S') ═══"

echo
echo "── sudo 인증 (한 번만) ──"
sudo -v || { echo "✗ sudo 실패"; exit 1; }

echo
echo "── 1. 드라이버 적재 ──"
~/workdir/load_driver.sh

echo
echo "── 2. UART · dbgprint 끄기 ──"
if [ -e "$TTY" ]; then
  sudo stty -F "$TTY" 115200 cs8 -cstopb -parenb clocal -crtscts -hupcl raw -echo 2>/dev/null
  sudo bash -c "exec 3<>$TTY
    ( timeout 12 cat <&3 > /tmp/uart_rx.txt ) &
    RD=\$!
    sleep 1; printf '\\r' >&3
    sleep 1; printf 'dbgprint uart off\\r' >&3
    sleep 4; printf 'dbgprint\\r' >&3
    wait \$RD 2>/dev/null
    exec 3<&-" 2>/dev/null
  tr -d '\r' < /tmp/uart_rx.txt | tail -8
  if grep -q "uart: OFF" /tmp/uart_rx.txt 2>/dev/null; then
    echo "   ✓ dbgprint uart OFF 확인됨"
  else
    echo "   ✗ 확인 실패 — minicom -D $TTY 로 직접 확인할 것"
  fi
else
  echo "   ✗ $TTY 없음 — 보드 UART 케이블 확인"
fi

echo
echo "── 3. 상태 확인 ──"
printf "   plda 모듈   : %s\n" "$(lsmod | grep -c '^plda ' | sed 's/^0$/없음/;s/^1$/적재됨/')"
printf "   장치 노드    : %s\n" "$(ls /dev/plda0 2>/dev/null || echo '없음')"
printf "   o_dbg       : %s\n" "$(cat /sys/module/plda/parameters/o_dbg 2>/dev/null || echo '?')"
printf "   루트 여유    : %s\n" "$(df -h / | awk 'NR==2{print $4" ("$5" 사용)"}')"

echo
echo "═══ 다음 ═══"
echo "  cd ~/workspace_docs && claude --continue"
echo
echo "  이어서 할 일은 REBOOT-CHECKLIST.md 의 5절 참조"
echo "  1순위는 egs 바이너리 수치 검증임:"
echo "    cd ~/workspace_docs/example"
echo "    LIMIT=200 ./run_map_npu.sh sw50sc64_lw sw50sc64_egs"
