"""
Configuration file for the RAG app.
"""

from pathlib import Path


class AppConfig:
    PROJECT_ROOT = Path(__file__).parent

    CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_data"
    TEMP_PDF_DIR = PROJECT_ROOT / "temp_pdfs"

    COLLECTION_PREFIX = "rag_session_"

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL = "llama-3.3-70b-versatile"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_PDF_SIZE_MB = 20

    TOP_K_RESULTS = 8
    MAX_CHAT_HISTORY = 1000

    def __init__(self):
        self.CHROMA_PERSIST_DIR.mkdir(exist_ok=True)
        self.TEMP_PDF_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_collection_name(cls, session_id: str) -> str:
        return f"{cls.COLLECTION_PREFIX}{session_id}".replace(" ", "_").replace("-", "_")

    @classmethod
    def get_max_pdf_size_bytes(cls) -> int:
        return cls.MAX_PDF_SIZE_MB * 1024 * 1024