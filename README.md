# FinQA Agentic-RAG Chatbot

A question-answering system for numerical reasoning over financial documents, built on the [FinQA dataset](https://github.com/czyssrs/FinQA).

**Stack:** Ollama · LangGraph · LangChain · ChromaDB · Gradio

---

## Quick Start

### 1. Install Ollama

- macOS: `brew install ollama`  or download from [ollama.com/download](https://ollama.com/download)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh`

### 2. Start Ollama and Pull the Model

```bash
./start_ollama.sh         # pulls qwen2.5:7b and starts the server on :11434
```

### 3. Configure

```bash
cp .env.example .env      # defaults already point at local Ollama
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Build the Index (first time only — already done in this repo)

```bash
python -m data.prepare_finqa      # download FinQA dataset
python -m indexing.build_index    # embed docs into ChromaDB
```

### 6. Launch the App

```bash
python app.py
# Open http://localhost:7860
```

### 7. Run Evaluation

```bash
python evaluate.py --n 50 --split test --verbose
```


---

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  LangGraph  │  ← ReAct loop (LangGraph StateGraph)
│  Agent Node │
│    │  ▲     │
│    ▼  │     │
│  Tool Node  │
│  ┌────────┐ │
│  │search_ │ │  ← ChromaDB MMR retrieval (LangChain)
│  │docs    │ │
│  ├────────┤ │
│  │calcu-  │ │  ← AST-safe arithmetic evaluator
│  │late    │ │
│  └────────┘ │
└─────────────┘
     │
     ▼
 Ollama server (qwen2.5:7b, local)
     │
     ▼
  Gradio UI (port 7860)
```

The agent follows a **ReAct** (Reason + Act) loop:
1. LLM decides: call `search_documents` or `calculate`, or emit the final answer
2. Tools execute and return results
3. LLM incorporates results and continues until it produces a final answer

---

## Dataset Analysis

### What is FinQA?

FinQA ([Zheng et al., 2021](https://arxiv.org/abs/2109.00122)) contains **8,281 QA pairs** sourced from S&P 500 annual reports. Each example pairs a financial document (pre-text + table + post-text) with a question requiring multi-step numerical reasoning.

| Statistic | Value |
|-----------|-------|
| Total QA pairs | 8,281 |
| Unique documents | 2,809 |
| Train / Dev / Test | 6,251 / 883 / 1,147 |
| Avg. reasoning steps | 2.8 |
| Requires table lookup | ~75% |

### Key Characteristics

- **Multi-step programs**: Answers are defined by DSL programs (e.g., `divide(subtract(A, B), B)`)
- **Structured + unstructured mix**: Every example has a Markdown table and surrounding prose
- **Exact numerical answers**: Gold answers are floats — approximate matching needed
- **Scale sensitivity**: Numbers are in thousands/millions/billions — scale errors are fatal

### What Makes Financial QA Unique

| Aspect | General QA | Financial QA |
|--------|-----------|-------------|
| Answer type | Free text | Precise numbers |
| Reasoning | Reading comprehension | Multi-step arithmetic |
| Context | Uniform text | Tables + text + headers |
| Units | Rarely critical | Always critical (K/M/B) |
| Error tolerance | Acceptable | Very low (1% tolerance) |

### Assumptions

- The retrieval step correctly surfaces the relevant table row
- Number formatting (commas, `%`, negative) is handled consistently
- Ollama serves `qwen2.5:7b`; a larger model (e.g. `qwen2.5:14b`) would improve ExAcc

---

## Method Selection

### Approaches Considered

| Approach | Pros | Cons | Chosen? |
|----------|------|------|---------|
| **Agentic RAG (ReAct)** | Transparent steps, safe arithmetic, iterative retrieval | More latency than single-pass | ✅ |
| Fine-tuned model | Highest potential accuracy | Needs GPU training, hard to update | ❌ |
| Vanilla RAG (single-pass) | Fast, simple | No multi-step reasoning, no calculator | ❌ |
| Pure prompt engineering | No infra needed | Hallucinated arithmetic, no retrieval | ❌ |
| Hybrid (fine-tune + RAG) | Best accuracy | Complexity; out of scope for demo | ❌ |

### Why Agentic RAG?

FinQA requires **two separable skills**: (1) locating the right numbers in a document, and (2) computing the right arithmetic. Agentic RAG solves these explicitly:

- `search_documents` handles skill 1 — semantic retrieval from 2,809 documents
- `calculate` handles skill 2 — deterministic, safe arithmetic (no LLM hallucination)
- The ReAct loop lets the LLM request multiple searches if the first one misses

This matches how a human analyst would approach the task: find the data, then calculate.

---

## Evaluation

### Primary Metric: Execution Accuracy (ExAcc)

From the FinQA paper: a prediction is correct if the extracted numeric answer matches `exe_ans` within **1% relative tolerance** (or 0.01 absolute for near-zero values).

```
ExAcc = (# predictions within 1% of gold) / (# total questions)
```

### How to Run

```bash
python evaluate.py --n 50 --split test --verbose
```

Results are saved to `eval_results.json`.

### Additional Metrics Tracked

| Metric | Why It Matters |
|--------|---------------|
| ExAcc | Primary FinQA paper metric |
| Error rate | Infra reliability |
| Avg latency / question | UX and cost |
| Number extraction success | Tool reliability |

### Baseline Comparison (from FinQA paper)

| System | ExAcc |
|--------|-------|
| Human expert | 91.16% |
| FinQA (BERT + program gen) | 61.24% |
| GPT-3 (few-shot) | ~55% |
| **This system (Qwen2.5-7B via Ollama, Agentic RAG)** | ~45–60%* |

*Estimated range; actual score depends on retrieval quality and model size.

### Numerical Reasoning Quality

Beyond ExAcc, check:
- Does the agent call `calculate` for every arithmetic step? (should be 100%)
- Does the retrieved context include the right table row? (examine `retrieved_contexts` in the UI)
- Are units preserved? (millions vs thousands errors)

---

## Production Monitoring Plan

### Performance Monitoring

| Signal | Method | Threshold |
|--------|--------|-----------|
| ExAcc on labelled sample | Weekly eval on 100 test questions | Alert if < 35% |
| Latency p50/p95 | Prometheus + Grafana | Alert if p95 > 30s |
| Error rate | Log parsing / Sentry | Alert if > 5% |
| Retrieval quality | Track `retrieved_contexts` hit rate | Alert if empty > 20% |

### Drift Detection

- **Data drift**: Monitor distribution of question types and financial domains (e.g., new sector companies not in FinQA index)
- **Model drift**: Re-run weekly ExAcc on a fixed held-out set; flag if score drops > 5 pp
- **Index staleness**: Log when source documents are > 1 year old; schedule quarterly re-indexing

### Maintenance & Improvement Plan

1. **Index refresh**: Re-run `indexing/build_index.py` when new 10-K/10-Q filings are added
2. **Model upgrade**: Swap `MODEL_NAME` in `.env` to upgrade to a larger Qwen or Llama model
3. **Retrieval tuning**: Experiment with `top_k`, `chunk_size`, and hybrid BM25+vector retrieval
4. **Few-shot prompting**: Add worked examples to `agent/prompts.py` for low-accuracy question types
5. **Feedback loop**: Collect user thumbs-up/down in the UI → re-label and add to eval set

---

## File Structure

```
project/
├── app.py                    # Gradio web UI
├── evaluate.py               # Evaluation script (ExAcc metric)
├── config.py                 # Centralised configuration
├── requirements.txt
├── setup.sh                  # One-time setup script
├── run.sh                    # Launch the Gradio app
├── start_ollama.sh           # Pull model and launch Ollama server
├── agent/
│   ├── graph.py              # LangGraph ReAct graph
│   ├── tools.py              # search_documents + calculate tools
│   └── prompts.py            # System prompt
├── data/
│   ├── prepare_finqa.py      # Download + preprocess FinQA dataset
│   └── processed/
│       ├── documents/        # 2,809 document JSON files
│       └── qa_pairs/         # 8,281 QA pairs
└── indexing/
    └── build_index.py        # Build ChromaDB vector index
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `MODEL_NAME` | `qwen2.5:7b` | Ollama model tag |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformer embedding model |
| `OLLAMA_API_KEY` | `ollama` | Placeholder key (Ollama ignores it) |
