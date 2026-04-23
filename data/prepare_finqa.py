"""
Download and preprocess the FinQA dataset.

"""

import json
from pathlib import Path

import httpx

from tabulate import tabulate
from tqdm import tqdm

from config import config

# Raw JSON files from the official FinQA GitHub repository
_FINQA_URLS = {
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "validation": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}



def format_table(table: list[list]) -> str:
    """Convert nested-list table to a Markdown-formatted string."""
    if not table:
        return ""
    headers = [str(h) for h in table[0]]
    rows = [[str(c) for c in row] for row in table[1:]]
    return tabulate(rows, headers=headers, tablefmt="pipe")


def build_document_text(example: dict) -> str:
    """Combine pre_text, table, and post_text into a single context string."""
    parts: list[str] = []

    pre_text = example.get("pre_text") or []
    if isinstance(pre_text, str):
        pre_text = [pre_text]
    if pre_text:
        parts.append("\n".join(str(p) for p in pre_text))

    table = example.get("table") or []
    if table:
        parts.append("\n[TABLE]\n" + format_table(table) + "\n[/TABLE]")

    post_text = example.get("post_text") or []
    if isinstance(post_text, str):
        post_text = [post_text]
    if post_text:
        parts.append("\n".join(str(p) for p in post_text))

    return "\n\n".join(parts)


def extract_doc_id(example: dict) -> str:
    """Return the document-level ID, stripping the trailing QA-pair suffix."""
    raw_id: str = str(example.get("id", ""))
    # e.g. "Single_JPMORGAN_2018_page_27.pdf-0" → "Single_JPMORGAN_2018_page_27.pdf"
    for suffix in ("-0", "-1", "-2", "-3", "-4", "-5"):
        if raw_id.endswith(suffix):
            return raw_id[: -len(suffix)]
    return raw_id


def _download_split(name: str, url: str) -> list[dict]:
    """Download a FinQA JSON split and return parsed records."""
    print(f"  Downloading {name}: {url}")
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    print(f"    → {len(data)} examples")
    return data


def prepare(output_dir: str | None = None) -> Path:
    """Download FinQA and write deduplicated document + QA-pair JSON files."""
    out_dir = Path(output_dir or config.processed_data_dir)
    docs_dir = out_dir / "documents"
    qa_dir = out_dir / "qa_pairs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    seen_doc_ids: set[str] = set()
    qa_pairs: list[dict] = []

    for split, url in _FINQA_URLS.items():
        examples = _download_split(split, url)

        for example in tqdm(examples, desc=f"Processing {split}"):
            doc_id = extract_doc_id(example)
            doc_text = build_document_text(example)

            # Save document once (deduplicated)
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                doc_record = {
                    "doc_id": doc_id,
                    "text": doc_text,
                    "pre_text": example.get("pre_text", []),
                    "post_text": example.get("post_text", []),
                    "table": example.get("table", []),
                }
                safe_name = doc_id.replace("/", "_").replace(" ", "_")
                doc_path = docs_dir / f"{safe_name}.json"
                doc_path.write_text(json.dumps(doc_record, ensure_ascii=False, indent=2))

            # Save QA pair
            qa = example.get("qa") or {}
            qa_pairs.append(
                {
                    "id": example.get("id", ""),
                    "doc_id": doc_id,
                    "question": qa.get("question", example.get("question", "")),
                    "answer": qa.get("answer", example.get("answer", "")),
                    "exe_ans": qa.get("exe_ans"),
                    "program": qa.get("program", ""),
                    "split": split,
                }
            )

    qa_path = qa_dir / "qa_pairs.json"
    qa_path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2))

    print(f"\nDone.")
    print(f"  Unique documents : {len(seen_doc_ids)}")
    print(f"  QA pairs         : {len(qa_pairs)}")
    print(f"  Output dir       : {out_dir}")
    return out_dir


if __name__ == "__main__":
    prepare()
