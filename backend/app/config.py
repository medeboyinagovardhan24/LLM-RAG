
from functools import lru_cache
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Load .env file
load_dotenv()

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LLMConfig:
    api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
    )


@dataclass
class RAGConfig:
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", 800))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", 100))
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", 5))
    )
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", 60))
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be a positive integer")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be non-negative and less than CHUNK_SIZE"
            )
        if self.top_k <= 0:
            raise ValueError("TOP_K must be a positive integer")
        if self.rrf_k <= 0:
            raise ValueError("RRF_K must be a positive integer")


@dataclass
class StorageConfig:
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", REPO_ROOT))
    )

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"


_DEFAULT_DEV_ORIGINS: List[str] = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5888",
    "http://127.0.0.1:5888",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:9001",
    "http://127.0.0.1:9001",
    "http://localhost:9091",
    "http://127.0.0.1:9091",
    "http://localhost:8010",
    "http://127.0.0.1:8010",
    "null",
]


def _resolve_allowed_origins() -> List[str]:
    """Resolve CORS origins from the ALLOWED_ORIGINS env var.

    Uses comma-separated origins when configured. Otherwise, falls back
    to local-development origins.
    """
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return list(_DEFAULT_DEV_ORIGINS)


@dataclass
class CORSConfig:
    allowed_origins: List[str] = field(
        default_factory=_resolve_allowed_origins
    )

    allowed_origin_regex: str = field(
        default_factory=lambda: os.getenv(
            "ALLOWED_ORIGIN_REGEX",
            r"https?://(localhost|127\.0\.0\.1):(\d+)",
        )
    )


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    cors: CORSConfig = field(default_factory=CORSConfig)
    max_upload_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", 20))
    )

    def __post_init__(self) -> None:
        if self.max_upload_mb <= 0:
            raise ValueError("MAX_UPLOAD_MB must be a positive integer")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origins(self) -> List[str]:
        return self.cors.allowed_origins

    @property
    def cors_origin_regex(self) -> str:
        return self.cors.allowed_origin_regex


@lru_cache
def get_settings() -> AppConfig:
    return AppConfig()
