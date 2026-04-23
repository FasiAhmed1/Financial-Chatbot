"""
LangGraph Agentic-RAG graph for FinQA.

Architecture (ReAct loop):
    START
      ↓
    [agent]  ← LLM decides: call a tool OR produce final answer
      ↓  (if tool_calls present)
    [tools]  ← ToolNode executes search_documents / calculate
      ↓
    [agent]  ← LLM sees tool results, continues reasoning
      ↓  (when no tool_calls)
    END

State fields:
    messages          : conversation history (LangGraph managed)
    question          : the current user question (metadata only)
    retrieved_contexts: snippets collected from tool calls (for UI display)
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS
from config import config


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FinQAState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    retrieved_contexts: list[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _make_llm() -> ChatOpenAI:
    """Connect to the Ollama OpenAI-compatible server."""
    return ChatOpenAI(
        model=config.model_name,
        base_url=config.ollama_base_url,
        api_key=config.ollama_api_key,   # Ollama ignores the key but client requires one
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=False,
    )


def build_agent_node(llm_with_tools):
    """Return a node function that calls the LLM (with tool bindings)."""

    def agent_node(state: FinQAState) -> dict:
        messages = list(state["messages"])

        # Inject system prompt once at the start of every graph invocation
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)

        # Collect any new retrieved text for the UI
        new_contexts: list[str] = []
        if isinstance(response, AIMessage) and response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] == "search_documents":
                    new_contexts.append(f"[Searching: {tc['args'].get('query', '')}]")

        return {
            "messages": [response],
            "retrieved_contexts": state.get("retrieved_contexts", []) + new_contexts,
        }

    return agent_node


def collect_tool_results(state: FinQAState) -> dict:
    """After ToolNode runs, pull the search results into retrieved_contexts."""
    from langchain_core.messages import ToolMessage

    last_msgs = state["messages"]
    new_contexts: list[str] = []
    for msg in reversed(last_msgs):
        if isinstance(msg, ToolMessage) and msg.name == "search_documents":
            new_contexts.insert(0, msg.content)
            break  # only the latest search result

    return {"retrieved_contexts": state.get("retrieved_contexts", []) + new_contexts}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue(state: FinQAState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_finqa_agent():
    """Compile and return the LangGraph agentic-RAG graph."""
    llm = _make_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    agent_node = build_agent_node(llm_with_tools)
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(FinQAState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END},
    )
    # After tools run, optionally harvest contexts then loop back to agent
    graph.add_node("harvest", collect_tool_results)
    graph.add_edge("tools", "harvest")
    graph.add_edge("harvest", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience helpers used by the UI
# ---------------------------------------------------------------------------

def extract_final_answer(messages: list[BaseMessage]) -> str:
    """Return the last non-tool-call AI message content."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return msg.content
    return ""


def format_reasoning_trace(messages: list[BaseMessage]) -> str:
    """Build a human-readable trace of the agent's tool interactions."""
    from langchain_core.messages import HumanMessage, ToolMessage

    lines: list[str] = []
    step = 0
    for msg in messages:
        if isinstance(msg, HumanMessage):
            continue
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                step += 1
                name = tc["name"]
                args = tc["args"]
                if name == "search_documents":
                    lines.append(f"🔍 Step {step}: Searching documents")
                    lines.append(f"   Query: \"{args.get('query', '')}\"")
                elif name == "calculate":
                    lines.append(f"🧮 Step {step}: Calculating")
                    lines.append(f"   Expression: {args.get('expression', '')}")
                else:
                    lines.append(f"🔧 Step {step}: {name}({args})")
        elif isinstance(msg, ToolMessage):
            content = msg.content or ""
            preview = content[:800] + ("…" if len(content) > 800 else "")
            lines.append(f"   → Result:\n{preview}")
            lines.append("")

    return "\n".join(lines) if lines else "No tool calls made."
