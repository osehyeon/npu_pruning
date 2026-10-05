#!/bin/bash
# 부팅 후 tmux 안에서 claude 세션을 자동으로 되살린다. 일반 사용자로 실행된다.
# 로그: ~/workspace_docs/boot/session-autostart.log
exec >>"$HOME/workspace_docs/boot/session-autostart.log" 2>&1
echo "═══ $(date '+%F %T') 세션 자동 복구 ═══"
sleep 20                                    # 네트워크와 보드 초기화 대기
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
command -v tmux >/dev/null || { echo "✗ tmux 없음"; exit 1; }
command -v claude >/dev/null || { echo "✗ claude 없음"; exit 1; }
tmux has-session -t npu 2>/dev/null && { echo "이미 있음"; exit 0; }
tmux new-session -d -s npu -c "$HOME/workspace_docs"
tmux send-keys -t npu 'claude --continue' C-m
echo "tmux 세션 npu 생성 · claude --continue 실행"
echo "  접속: tmux attach -t npu"
