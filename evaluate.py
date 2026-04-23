"""
FinQA Evaluation Script
"""

import argparse
import json
import math
import re
import random
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.graph import build_finqa_agent, extract_final_answer
from config import config


_NUMBER_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|-?\d+(?:\.\d+)?%?"
)


def parse_number(text: str) -> float | None:
    """Extract the last number (possibly a percentage) from a text string."""
    text = text.replace(",", "")
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].rstrip("%")
    try:
        return float(raw)
    except ValueError:
        return None


def execution_match(predicted: str, gold_exe_ans) -> bool:
    """
    True if the numeric answer extracted from `predicted` matches `gold_exe_ans`.

    Tolerances:
      - relative 1 % for values with abs > 1
      - absolute 0.01 for near-zero values
    """
    if gold_exe_ans is None:
        return False
    try:
        gold = float(gold_exe_ans)
    except (TypeError, ValueError):
        # gold is a string answer (e.g. "yes" / "no")
        gold_str = str(gold_exe_ans).strip().lower()
        pred_str = predicted.strip().lower()
        return gold_str in pred_str

    pred = parse_number(predicted)
    if pred is None:
        return False

    if math.isnan(gold) or math.isinf(gold):
        return False

    abs_gold = abs(gold)
    if abs_gold < 1e-3:
        return abs(pred - gold) < 0.01
    return abs(pred - gold) / abs_gold < 0.01


def program_accuracy(predicted: str, gold_program: str) -> bool:
    """Very loose check: does the predicted answer contain the right operator sequence."""
    if not gold_program:
        return False
    ops = re.findall(r"[a-z_]+\(", gold_program)
    if not ops:
        return False
    return all(op[:-1] in predicted.lower() for op in ops)



def load_qa_pairs(split: str, n: int, seed: int = 42) -> list[dict]:
    qa_path = Path(config.processed_data_dir) / "qa_pairs" / "qa_pairs.json"
    if not qa_path.exists():
        raise FileNotFoundError(f"QA pairs not found at {qa_path}. Run setup.sh first.")
    all_pairs = json.loads(qa_path.read_text())
    split_pairs = [p for p in all_pairs if p.get("split") == split]
    if not split_pairs:
        raise ValueError(f"No QA pairs found for split='{split}'")
    random.seed(seed)
    return random.sample(split_pairs, min(n, len(split_pairs)))



def evaluate(n: int = 50, split: str = "test", verbose: bool = False) -> dict:
    print(f"\n{'='*60}")
    print(f"  FinQA Evaluation  |  split={split}  |  n={n}")
    print(f"  LLM: {config.model_name}")
    print(f"  Ollama: {config.ollama_base_url}")
    print(f"{'='*60}\n")

    agent = build_finqa_agent()
    qa_pairs = load_qa_pairs(split, n)

    results = []
    correct_exec = 0
    total_latency = 0.0
    errors = 0

    for i, qp in enumerate(qa_pairs, 1):
        question = qp["question"]
        gold_exe  = qp.get("exe_ans")
        gold_ans  = qp.get("answer", "")

        if verbose:
            print(f"\n[{i}/{n}] Q: {question}")
            print(f"        Gold: {gold_exe} (answer={gold_ans})")

        t0 = time.time()
        try:
            result = agent.invoke({
                "messages": [HumanMessage(content=question)],
                "question": question,
                "retrieved_contexts": [],
            })
            predicted = extract_final_answer(result["messages"])
            latency   = time.time() - t0
            exec_ok   = execution_match(predicted, gold_exe)

            if exec_ok:
                correct_exec += 1
            total_latency += latency

            results.append({
                "id":        qp["id"],
                "question":  question,
                "gold":      gold_exe,
                "predicted": predicted,
                "exec_match": exec_ok,
                "latency":   round(latency, 2),
            })

            if verbose:
                mark = "✅" if exec_ok else "❌"
                print(f"        Pred: {predicted[:120]}")
                print(f"        {mark}  latency={latency:.1f}s")

        except Exception as exc:
            errors += 1
            results.append({
                "id":       qp["id"],
                "question": question,
                "gold":     gold_exe,
                "error":    str(exc),
                "exec_match": False,
                "latency":    time.time() - t0,
            })
            if verbose:
                print(f"        ❌ ERROR: {exc}")

        # Print running accuracy every 10 questions
        if i % 10 == 0 or i == n:
            acc = correct_exec / i * 100
            avg_lat = total_latency / max(1, i - errors)
            print(f"  [{i:3d}/{n}]  ExAcc={acc:.1f}%  avg_latency={avg_lat:.1f}s")

    total     = len(results)
    exec_acc  = correct_exec / total * 100 if total else 0.0
    avg_lat   = total_latency / max(1, total - errors)

    print(f"\n{'='*60}")
    print(f"  RESULTS  (split={split}, n={total})")
    print(f"  Execution Accuracy (ExAcc) : {exec_acc:.1f}%")
    print(f"  Errors                     : {errors}")
    print(f"  Avg latency / question     : {avg_lat:.2f}s")
    print(f"{'='*60}\n")

    # Save results to JSON
    out_path = Path("eval_results.json")
    out_path.write_text(json.dumps({
        "config":    {"model": config.model_name, "split": split, "n": total},
        "exec_acc":  round(exec_acc, 2),
        "errors":    errors,
        "avg_latency_s": round(avg_lat, 2),
        "results":   results,
    }, indent=2))
    print(f"Full results saved to {out_path}")

    return {"exec_acc": exec_acc, "errors": errors, "avg_latency": avg_lat}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the FinQA agent.")
    parser.add_argument("--n",       type=int,  default=50,         help="Number of questions to evaluate")
    parser.add_argument("--split",   type=str,  default="test",     help="Dataset split: train / validation / test")
    parser.add_argument("--verbose", action="store_true",           help="Print each question and prediction")
    parser.add_argument("--seed",    type=int,  default=42,         help="Random seed for sampling")
    args = parser.parse_args()

    evaluate(n=args.n, split=args.split, verbose=args.verbose)
