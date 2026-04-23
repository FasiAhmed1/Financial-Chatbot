# FinQA Agentic-RAG Chatbot

A question-answering system for numerical reasoning over financial documents, built on the [FinQA dataset](https://github.com/czyssrs/FinQA).

**Stack:** Ollama · LangGraph · LangChain · ChromaDB · Gradio

---

The agent follows a **ReAct** (Reason + Act) loop:
1. LLM decides: call `search_documents` or `calculate`, or emit the final answer
2. Tools execute and return results
3. LLM incorporates results and continues until it produces a final answer

---

## Dataset Analysis

### What is FinQA?

FinQA ([Zheng et al., 2021](https://arxiv.org/abs/2109.00122)) contains **8,281 QA pairs** sourced from S&P 500 annual reports. Each example pairs a financial document (pre-text + table + post-text) with a question requiring multi-step numerical reasoning.


### What Makes Financial QA Unique

Answer type | Free text | Precise numbers 
Reasoning | Reading comprehension | Multi-step arithmetic 
Context | Uniform text | Tables + text + headers 

### Assumptions

- The retrieval step correctly surfaces the relevant table row
- Number formatting (commas, `%`, negative) is handled consistently
- Ollama serves `qwen2.5:7b`; a larger model (e.g. `qwen2.5:14b`) would improve ExAcc

---

## Method Selection

### Approaches Considered

| **Agentic RAG (ReAct)** | Transparent steps, safe arithmetic, iterative retrieval | More latency than single-pass 
Fine-tuned model | Highest potential accuracy | Needs GPU training, hard to update
Vanilla RAG (single-pass) | Fast, simple | No multi-step reasoning, no calculator 

### Why Agentic RAG?

FinQA requires **two separable skills**: (1) locating the right numbers in a document and (2) computing the right arithmetic. Agentic RAG solves these explicitly:

- `search_documents` handles skill 1 — semantic retrieval from 2,809 documents
- `calculate` handles skill 2 — deterministic, safe arithmetic (no LLM hallucination)
- The ReAct loop lets the LLM request multiple searches if the first one misses

This matches how a human analyst would approach the task - find the data, then calculate.

---

## Evaluation

### Primary Metric: Execution Accuracy 

From the FinQA paper: a prediction is correct if the extracted numeric answer matches `exe_ans` within **1% relative tolerance** (or 0.01 absolute for near-zero values).

### How to Run

Results are saved to `eval_results.json`.

### Additional Metrics Tracked

ExAcc | Primary FinQA paper metric 
Error rate | Infra reliability 
Avg latency / question | UX and cost 
Number extraction success | Tool reliability 

### Baseline Comparison (from FinQA paper)

Human expert | 91.16% 
FinQA (BERT + program gen) | 61.24% 
GPT-3 (few-shot) | ~55% 
**This system (Qwen2.5-7B via Ollama, Agentic RAG)** | ~45–60%* 

*Estimated range- actual score depends on retrieval quality and model size.

### Numerical Reasoning Quality

Beyond ExAcc, check:
- Does the agent call `calculate` for every arithmetic step? (should be 100%)
- Does the retrieved context include the right table row? (examine `retrieved_contexts` in the UI)
- Are units preserved? (millions vs thousands errors)

---


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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `MODEL_NAME` | `qwen2.5:7b` | Ollama model tag |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformer embedding model |
| `OLLAMA_API_KEY` | `ollama` | Placeholder key (Ollama ignores it) |
