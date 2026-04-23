import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Ollama server (local inference, OpenAI-compatible API)
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "qwen2.5:7b"))

    # Local sentence-transformers embedding model (runs on CPU; no server needed)
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # Ollama ignores the key, but the OpenAI client requires a non-empty string
    ollama_api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY", "ollama"))

    # ChromaDB persistence
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "finqa_documents"

    # Retrieval
    top_k: int = 4

    # LLM generation
    temperature: float = 0.0
    max_tokens: int = 2048

    # Dataset
    processed_data_dir: str = "./data/processed"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 7860


config = Config()
