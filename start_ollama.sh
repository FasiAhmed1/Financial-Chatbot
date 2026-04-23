#!/usr/bin/env bash
# Start Ollama and pull the required LLM.

set -euo pipefail

LLM_MODEL="${MODEL_NAME:-qwen2.5:7b}"

if ! command -v ollama &>/dev/null; then
    echo "ERROR: ollama not found."
    echo "Install it from https://ollama.com/download and re-run this script."
    exit 1
fi

echo "Pulling LLM : $LLM_MODEL"
ollama pull "$LLM_MODEL"

echo ""
echo "Starting Ollama server on http://localhost:11434 …"
echo "(Press Ctrl-C to stop)"
ollama serve
