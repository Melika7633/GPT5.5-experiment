#!/bin/bash
# Resolve one 200-paper block on a local machine and keep a timestamped log, e.g.
#   bash generation/run-local-resolver.sh low 0
# Exit status 2 means some searches are still pending: rerun the same command later.
set -euo pipefail
cd "$(dirname "$0")/.."

CONDITION=$1
START=$2
LOG_DIR=$(printf "outputs/%s/resolved/%04d-%04d/runs" "$CONDITION" "$START" $((START + 199)))
mkdir -p "$LOG_DIR"
python3 -u -m generation.resolve --condition "$CONDITION" --start "$START" --count 200 2>&1 \
    | tee "$LOG_DIR/resolve-$(date -u +%Y%m%dT%H%M%SZ).log"
