#!/usr/bin/env bash

set -euo pipefail

echo " FinQA Chatbot — Setup"

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done
[ -z "$PYTHON" ] && { echo "ERROR: Python not found."; exit 1; }
echo "Python : $("$PYTHON" --version)  ($(which "$PYTHON"))"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo ""
    echo " No virtual environment detected."
    echo "   Recommended:"
    echo "     $PYTHON -m venv .venv && source .venv/bin/activate"
    echo "   Continuing without venv …"
fi

echo ""
echo "[1/3] Installing Python dependencies …"
"$PYTHON" -m pip install --upgrade pip --quiet
"$PYTHON" -m pip install -r requirements.txt

echo ""
echo "[2/3] Downloading and processing FinQA dataset …"
"$PYTHON" -m data.prepare_finqa

echo ""
echo "[3/3] Building ChromaDB vector index …"
echo "     (embeddings run locally via sentence-transformers — no server needed)"
"$PYTHON" -m indexing.build_index

echo ""
echo " Setup complete!"
echo ""
echo " To launch:"
echo "   Terminal 1 → ./start_ollama.sh     (keep running)"
echo "   Terminal 2 → $PYTHON app.py"
echo "   Browser    → http://localhost:7860"
