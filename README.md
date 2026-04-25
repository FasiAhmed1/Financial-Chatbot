=odel
3. **Retrieval tuning**: Experiment with `top_k`, `chunk_size`, and hybrid BM25+vector retrieval
4. **Few-shot prompting**: Add worked examples to `agent/prompts.py` for low-accuracy question types
5. **Feedback loop**: Collect user thumbs-up/down in the UI → re-label and add to eval set

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `MODEL_NAME` | `qwen2.5:7b` | Ollama model tag |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformer embedding model |
| `OLLAMA_API_KEY` | `ollama` | Placeholder key (Ollama ignores it) |
