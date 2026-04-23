#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

echo "Starting FinQA Chatbot UI …"
echo "Open http://localhost:7860 in your browser."
"$PY" app.py
