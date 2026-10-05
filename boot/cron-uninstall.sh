#!/bin/bash
# @reboot 세션 자동 복구 항목만 제거한다. sudo 불필요.
crontab -l 2>/dev/null | grep -v "session-autostart.sh" | crontab -
echo "제거 완료. 남은 crontab:"
crontab -l 2>/dev/null || echo "  (비어 있음)"
