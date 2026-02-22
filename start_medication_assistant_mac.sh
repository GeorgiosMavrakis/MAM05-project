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

# Fix execute permissions on node_modules binaries (in case they were lost)
if [[ -d "rag-ui/node_modules/.bin" ]]; then
  chmod +x rag-ui/node_modules/.bin/* 2>/dev/null || true
fi

echo "[1/2] Starting API Server..."
osascript - "$PROJECT_DIR" <<'APPLESCRIPT'
on run argv
  set projectDir to item 1 of argv
  tell application "Terminal"
    do script "cd " & projectDir & " && echo 'Starting API Server...' && uvicorn API:app --host 0.0.0.0 --port 8000 --reload"
  end tell
end run
APPLESCRIPT
sleep 3

echo "[2/2] Starting UI Server..."
osascript - "$PROJECT_DIR" <<'APPLESCRIPT'
on run argv
  set projectDir to item 1 of argv
  tell application "Terminal"
    do script "cd " & projectDir & "/rag-ui && echo 'Starting UI Server...' && npm start"
  end tell
end run
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

