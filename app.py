"""
Gradio web UI for the FinQA Agentic-RAG chatbot.

Layout:
  ┌─────────────────────────────────────┬──────────────────────┐
  │  Chat  (2/3 width)                  │  Agent Reasoning     │
  │  [history]                          │  (tool calls + docs) │
  │  [input box]  [Ask]  [Clear]        │                      │
  └─────────────────────────────────────┴──────────────────────┘
  [Example questions]

Run:
    python app.py
"""

import traceback

import gradio as gr
from langchain_core.messages import HumanMessage

from agent.graph import build_finqa_agent, extract_final_answer, format_reasoning_trace
from config import config

# ---------------------------------------------------------------------------
# Initialise agent (once at startup)
# ---------------------------------------------------------------------------

print("Initialising FinQA agent …")
try:
    agent = build_finqa_agent()
    AGENT_READY = True
    INIT_ERROR = ""
    print("Agent ready.")
except Exception as exc:
    AGENT_READY = False
    INIT_ERROR = str(exc)
    print(f"[WARNING] Agent init failed: {exc}\nCheck that Ollama is running and ChromaDB is built.")

# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------

def answer_question(question: str, history: list) -> tuple[list, str, str]:
    """Called by Gradio on submit. Returns updated history, reasoning, status."""
    if not question.strip():
        return history, "", "⚠️ Please enter a question."

    if not AGENT_READY:
        err_msg = f"❌ Agent not ready: {INIT_ERROR}"
        return history + [{"role": "user", "content": question}, {"role": "assistant", "content": err_msg}], "", err_msg

    try:
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "question": question,
                "retrieved_contexts": [],
            }
        )
        messages = result["messages"]
        final_answer = extract_final_answer(messages)
        reasoning = format_reasoning_trace(messages)
        return history + [{"role": "user", "content": question}, {"role": "assistant", "content": final_answer}], reasoning, "✅ Done"

    except Exception:
        tb = traceback.format_exc()
        return (
            history + [{"role": "user", "content": question}, {"role": "assistant", "content": "Sorry, an error occurred while processing your question."}],
            f"Error:\n{tb}",
            "❌ Error",
        )


# ---------------------------------------------------------------------------
# Example questions (sampled from FinQA)
# ---------------------------------------------------------------------------

EXAMPLES = [
    "What was the percentage change in net revenue from 2017 to 2018?",
    "How much did total assets increase between the two most recent reporting periods?",
    "What percentage of total revenue was represented by non-interest income?",
    "Calculate the year-over-year growth rate in operating expenses.",
    "What is the ratio of tier 1 capital to risk-weighted assets?",
    "By how many basis points did the net interest margin change?",
    "What share of total loans were commercial real estate loans?",
]

# ---------------------------------------------------------------------------
# Gradio UI  (Gradio 6 compatible)
# ---------------------------------------------------------------------------

CSS = """
#chat-col { min-height: 550px; }
#reasoning-box textarea { font-family: monospace; font-size: 0.82rem; }
footer { display: none !important; }
"""

with gr.Blocks(title="FinQA Chatbot") as demo:

    gr.Markdown(
        """
        # 📊 FinQA: Financial Question-Answering Chatbot
        **Agentic RAG** · Ollama · LangGraph · LangChain · ChromaDB
        > Ask questions requiring multi-step numerical reasoning over SEC filings and annual reports.
        > The agent retrieves relevant passages, performs calculations, and shows its reasoning.
        """
    )

    with gr.Row():
        # ── Left: Chat panel ──────────────────────────────────────────────
        with gr.Column(scale=3, elem_id="chat-col"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=480,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="Ask a financial question…",
                    lines=2,
                    scale=5,
                    show_label=False,
                )
                ask_btn = gr.Button("Ask ▶", variant="primary", scale=1, min_width=80)

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear chat", size="sm")
                status_bar = gr.Textbox(
                    value="Ready" if AGENT_READY else "⚠️ Agent not ready",
                    label="",
                    interactive=False,
                    scale=4,
                    show_label=False,
                )

        # ── Right: Reasoning panel ────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 🤖 Agent Reasoning Trace")
            reasoning_box = gr.Textbox(
                label="Tool calls · Retrieved context · Calculations",
                elem_id="reasoning-box",
                lines=26,
                max_lines=40,
                interactive=False,
            )

    # ── Examples ─────────────────────────────────────────────────────────
    gr.Examples(
        examples=[[ex] for ex in EXAMPLES],
        inputs=[msg_input],
        label="📌 Example questions (click to load)",
    )

    # ── Model info accordion ──────────────────────────────────────────────
    with gr.Accordion("ℹ️ System Info", open=False):
        gr.Markdown(
            f"""
            | Component | Value |
            |-----------|-------|
            | LLM | `{config.model_name}` |
            | Serving | Ollama @ `{config.ollama_base_url}` |
            | Embeddings | `{config.embedding_model}` (local sentence-transformers) |
            | Vector store | ChromaDB (`{config.collection_name}`) |
            | Retrieval k | {config.top_k} chunks (MMR) |
            | Framework | LangGraph agentic-RAG |
            """
        )

    # ── Wiring ────────────────────────────────────────────────────────────
    def _submit(question, history):
        return answer_question(question, history)

    ask_btn.click(
        fn=_submit,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, reasoning_box, status_bar],
    ).then(fn=lambda: "", outputs=msg_input)

    msg_input.submit(
        fn=_submit,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, reasoning_box, status_bar],
    ).then(fn=lambda: "", outputs=msg_input)

    clear_btn.click(
        fn=lambda: ([], "", "Ready"),
        outputs=[chatbot, reasoning_box, status_bar],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name=config.app_host,
        server_port=config.app_port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS,
    )
