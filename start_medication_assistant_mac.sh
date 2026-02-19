#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_header() {
  echo "========================================"
  echo "Start Medication Assistant (macOS)"
  echo "========================================"
  echo
}

print_header

cd "$PROJECT_DIR"

echo "[1/2] Starting API Server..."
API_CMD="cd \"$PROJECT_DIR\"; echo 'Starting API Server...'; uvicorn API:app --host 0.0.0.0 --port 8000 --reload"
osascript <<APPLESCRIPT
tell application "Terminal"
  do script "$API_CMD"
end tell
APPLESCRIPT
sleep 3

echo "[2/2] Starting UI Server..."
UI_CMD="cd \"$PROJECT_DIR/rag-ui\"; echo 'Starting UI Server...'; npm start"
osascript <<APPLESCRIPT
tell application "Terminal"
  do script "$UI_CMD"
end tell
APPLESCRIPT

sleep 5

echo "Opening browser..."
open http://localhost:3000

cat <<EOF

========================================
RAG System Started Successfully!
========================================
API Server:  http://localhost:8000
UI Server:   http://localhost:3000
API Docs:    http://localhost:8000/docs

Two Terminal windows were opened:
 - RAG API Server (FastAPI with uvicorn)
 - RAG UI Server (Next.js)

Close those windows to stop the servers.
EOF

