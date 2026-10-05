#!/bin/bash
# 부팅 시 Antara 보드를 사용 가능 상태로 만든다. root 로 실행된다.
#   1. PLDA 드라이버 적재              o_dbg=0 · max_irq=32
#   2. UART 로 dbgprint uart off   측정 왜곡 방지
# 순서 중요 — 드라이버 적재가 dbgprint 설정을 초기화한다.
# 로그: /var/log/npu-board-init.log
exec >>/var/log/npu-board-init.log 2>&1
echo "═══ $(date '+%F %T') 부팅 초기화 ═══"

TTY=/dev/ttyUSB0
WD=/home/osehn/workdir
KO="$WD/antara-software/antara-pcie-sdk/PCIeDriver/plda.ko"

# 보드 UART 가 올라올 때까지 잠시 대기
for i in $(seq 1 30); do [ -e "$TTY" ] && break; sleep 2; done

echo "── 드라이버 적재 ──"
if lsmod | grep -q '^plda '; then
  echo "  이미 적재됨"
else
  [ -f "$KO" ] && insmod "$KO" o_dbg=0 max_irq=32 && echo "  insmod 성공" || echo "  ✗ insmod 실패"
  udevadm settle 2>/dev/null || sleep 1
fi

echo "── UART dbgprint off ──"
if [ -e "$TTY" ]; then
  stty -F "$TTY" 115200 cs8 -cstopb -parenb clocal -crtscts -hupcl raw -echo 2>/dev/null
  bash -c "exec 3<>$TTY
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
  echo "  ✗ $TTY 없음"
fi

echo "── 상태 ──"
echo "  lsmod  : $(lsmod | grep '^plda ' || echo '미로드')"
echo "  o_dbg  : $(cat /sys/module/plda/parameters/o_dbg 2>/dev/null || echo '?')"
echo "  node   : $(ls -la /dev/plda0 2>/dev/null || echo '없음')"
echo "  disk   : $(df -h / | awk 'NR==2{print $4" free ("$5")"}')"
echo "═══ 완료 $(date '+%T') ═══"
