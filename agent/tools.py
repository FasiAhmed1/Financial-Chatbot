"""
LangChain tools used by the FinQA agent:
  - search_documents : semantic retrieval from ChromaDB
  - calculate        : safe AST-based arithmetic evaluator
"""

import ast
import math
import operator
from functools import lru_cache

from langchain_core.tools import tool

from config import config

# ---------------------------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        fn = _SAFE_NAMES.get(node.func.id)
        if fn is None:
            raise ValueError(f"Unsupported function: {node.func.id!r}")
        args = [_eval_node(a) for a in node.args]
        return fn(*args)

    if isinstance(node, ast.Name):
        val = _SAFE_NAMES.get(node.id)
        if val is None or not isinstance(val, (int, float)):
            raise ValueError(f"Unknown name: {node.id!r}")
        return float(val)

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Parse and evaluate a purely arithmetic expression safely."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc
    return _eval_node(tree.body)


# ---------------------------------------------------------------------------
# Retriever singleton (lazy init)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_retriever():
    """Load ChromaDB + local HuggingFace embeddings and return a LangChain retriever (cached)."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_dir,
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.top_k, "fetch_k": config.top_k * 3},
    )


# ---------------------------------------------------------------------------
# LangChain @tool definitions
# ---------------------------------------------------------------------------

@tool
def search_documents(query: str) -> str:
    """Search the FinQA financial document database for passages and tables
    relevant to the query. Use this to retrieve numbers, percentages, or
    context needed to answer a financial question.

    Args:
        query: A natural-language search query describing what financial data
               you need (e.g., "net revenue 2017 2018 JPMorgan").

    Returns:
        A numbered list of the most relevant document excerpts.
    """
    retriever = _get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found. Try rephrasing the query."

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        doc_id = doc.metadata.get("doc_id", "unknown")
        parts.append(f"[{i}] Source: {doc_id}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Supports: +, -, *, /, **, %, //, abs(), round(), sqrt(), log().

    Args:
        expression: A valid arithmetic expression string, e.g.:
                    "(109.0 - 105.5) / 105.5 * 100"
                    "round(1234.567, 2)"
                    "sqrt(144)"

    Returns:
        A string showing the expression and its computed result.
    """
    try:
        result = safe_eval(expression)
        # Format: avoid unnecessary .0 for integers
        if result == int(result) and abs(result) < 1e15:
            formatted = str(int(result))
        else:
            formatted = f"{result:.6g}"
        return f"{expression} = {formatted}"
    except ZeroDivisionError:
        return f"Error: division by zero in '{expression}'"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


TOOLS = [search_documents, calculate]
