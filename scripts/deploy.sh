#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-maxb@192.168.0.12}"
REMOTE_DIR="${REMOTE_DIR:-/home/maxb/naves-calc}"
SERVICE="${SERVICE:-naves-calc.service}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8501/}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

printf 'Deploying %s to %s:%s\n' "$ROOT_DIR" "$HOST" "$REMOTE_DIR"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.pytest_cache/' \
  --exclude '.idea/' \
  --exclude '__pycache__/' \
  --exclude 'logs/' \
  "$ROOT_DIR/" "$HOST:$REMOTE_DIR/"

ssh "$HOST" bash -s <<EOF_REMOTE
set -euo pipefail
export PATH="\$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd "$REMOTE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="\$HOME/.local/bin:\$PATH"
fi

uv sync --python 3.12
systemctl --user daemon-reload
systemctl --user restart "$SERVICE"
sleep 3
systemctl --user --no-pager --full status "$SERVICE" | sed -n '1,14p'
curl -fsS -I "$HEALTH_URL" | head -n 5
EOF_REMOTE

printf 'Deploy complete: %s\n' "$HEALTH_URL"
