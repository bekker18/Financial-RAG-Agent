import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str

    gemini_api_key: str
    gemini_embedding_model: str
    embedding_dim: int

    sec_user_agent: str

    chroma_path: str
    chroma_collection: str

    chunk_size: int
    chunk_overlap: int


def get_settings() -> Settings:
    return Settings(
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        groq_model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_embedding_model=os.environ.get(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-2",
        ),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "768")),
        sec_user_agent=os.environ.get(
            "SEC_USER_AGENT",
            "FinancialRAGAgent example@example.com",
        ),
        chroma_path=os.environ.get("CHROMA_PATH", "data/chroma"),
        chroma_collection=os.environ.get(
            "CHROMA_COLLECTION",
            "financial_filings",
        ),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "1800")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "250")),
    )
