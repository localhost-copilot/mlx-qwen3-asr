#!/usr/bin/env bash
# Install a LaunchAgent that keeps the folder watcher running at login.
#
#   ./install-launchd.sh "/Volumes/backup/小鹅通课程"
#   ./install-launchd.sh "/path/a" "/path/b"
#
# Re-running replaces the existing agent.
set -euo pipefail

LABEL="com.mlx-asr.watcher"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
SCRIPT="$HERE/watch_transcribe.py"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs"

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <要监听的目录> [更多目录…]" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "找不到 $PY，请先在仓库根目录执行：uv pip install -e \".[serve]\"" >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

# Build ProgramArguments XML: interpreter, script, then one --root per path.
ARGS_XML="    <string>$PY</string>
    <string>$SCRIPT</string>"
for root in "$@"; do
  ARGS_XML="$ARGS_XML
    <string>--root</string>
    <string>$(cd "$root" && pwd)</string>"
done

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
$ARGS_XML
    <string>--poll</string>
    <string>30</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>120</integer>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/mlx-asr-watcher.out.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/mlx-asr-watcher.err.log</string>
</dict>
</plist>
PLIST_EOF

UID_NUM="$(id -u)"
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST"
launchctl enable "gui/$UID_NUM/$LABEL"

echo "已安装并启动 $LABEL"
echo "  监听: $*"
echo "  日志: $LOG_DIR/mlx-asr-watcher.out.log"
echo "  停止: launchctl bootout gui/$UID_NUM/$LABEL"
