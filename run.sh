#!/usr/bin/env bash
# Launch the Gradio app (assumes Ollama server is already running).

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Prefer the project venv if it exists
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

echo "Starting FinQA Chatbot UI …"
echo "Open http://localhost:7860 in your browser."
"$PY" app.py
