#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

python -m edumind.cli ocr-api &
OCR_PID=$!
sleep 2

python -m edumind.cli rag-api &
RAG_PID=$!
sleep 2

cleanup() {
  kill "$OCR_PID" "$RAG_PID" 2>/dev/null || true
}

trap cleanup EXIT

echo "Services launched:"
echo "  OCR API: http://localhost:8000/docs"
echo "  RAG API: http://localhost:8001/docs"
echo "Run 'python -m edumind.cli ui' separately for the maintained local UI."

wait
