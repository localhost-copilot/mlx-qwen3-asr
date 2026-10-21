#!/usr/bin/env bash
# Convenience launcher: use this repo's .venv to run the folder watcher.
#
#   ./start-watcher.sh --root "/path/to/downloads"
#   ./start-watcher.sh --root "/path/to/downloads" --once --dry-run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "找不到 $PY，请先按 README 安装：uv pip install -e \".[serve]\"" >&2
  exit 1
fi

exec "$PY" "$HERE/watch_transcribe.py" "$@"
