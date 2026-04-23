

import argparse
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

from config import config


def load_embedding_model() -> HuggingFaceEmbeddings:
    print(f"Loading embedding model: {config.embedding_model} (local sentence-transformers, CPU)")
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(docs_dir: Path) -> list[Document]:
    """Read all processed document JSON files and return LangChain Documents."""
    doc_files = sorted(docs_dir.glob("*.json"))
    if not doc_files:
        raise FileNotFoundError(
            f"No document JSON files found in {docs_dir}.\n"
            "Run `python -m data.prepare_finqa` first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    lc_docs: list[Document] = []
    for fp in tqdm(doc_files, desc="Loading docs"):
        record = json.loads(fp.read_text())
        doc_id = record["doc_id"]
        full_text = record["text"]

        chunks = splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            lc_docs.append(
                Document(
                    page_content=chunk,
                    metadata={"doc_id": doc_id, "chunk": i, "source": fp.name},
                )
            )

    return lc_docs


def build_index(reset: bool = False) -> Chroma:
    embeddings = load_embedding_model()
    persist_dir = config.chroma_persist_dir

    if reset and Path(persist_dir).exists():
        import shutil
        shutil.rmtree(persist_dir)
        print(f"Cleared existing index at {persist_dir}")

    docs_dir = Path(config.processed_data_dir) / "documents"
    print(f"Reading documents from {docs_dir}")
    documents = load_documents(docs_dir)
    print(f"Total chunks to index: {len(documents)}")

    print("Building ChromaDB vector store …")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=config.collection_name,
        persist_directory=persist_dir,
    )

    print(f"Index saved to {persist_dir}")
    print(f"Collection '{config.collection_name}' has {vectorstore._collection.count()} chunks.")
    return vectorstore


def load_index() -> Chroma:
    """Load an existing ChromaDB index without re-embedding."""
    embeddings = load_embedding_model()
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild the index")
    args = parser.parse_args()
    build_index(reset=args.reset)
