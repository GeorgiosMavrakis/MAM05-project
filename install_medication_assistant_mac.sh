#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_header() {
  echo "========================================"
  echo "Start Medication Assistant (macOS)"
  echo "========================================"
  echo
}

require_cmd() {
  local name="$1"
  local hint="$2"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "ERROR: $name is not installed."
    echo "$hint"
    echo
    exit 1
  fi
}

print_header

cd "$PROJECT_DIR"

echo "[1/6] Checking Node.js installation..."
require_cmd node "Please install Node.js from https://nodejs.org/ or via Homebrew/nvm."
node --version

echo "[2/6] Checking Python installation..."
require_cmd python3 "Please install Python 3.10+ from https://www.python.org/ or via Homebrew."
python3 --version

echo "[3/6] Installing/Checking Python dependencies..."
if [[ ! -f "requirements.txt" ]]; then
  echo "WARNING: requirements.txt not found. Skipping Python dependency check."
else
  echo "Installing Python packages from requirements.txt..."
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
  echo "Python dependencies installed successfully!"
fi

echo "[4/6] Installing/Checking Node.js dependencies..."
if [[ ! -d "rag-ui" ]]; then
  echo "ERROR: rag-ui directory not found."
  exit 1
fi
pushd "rag-ui" >/dev/null
if [[ ! -d "node_modules" ]]; then
  echo "node_modules not found. Running npm install..."
  npm install
else
  echo "node_modules found. Checking for updates..."
  npm install
fi
popd >/dev/null

echo "[5/6] Starting API Server..."
API_CMD="cd \"$PROJECT_DIR\"; echo 'Starting API Server...'; uvicorn API:app --host 0.0.0.0 --port 8000 --reload"
osascript <<APPLESCRIPT
tell application "Terminal"
  do script "$API_CMD"
end tell
APPLESCRIPT
sleep 3

echo "[6/6] Starting UI Server..."
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

