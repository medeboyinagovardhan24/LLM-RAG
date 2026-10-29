"""Shared pytest fixtures.

Tests run with no network access and no real Gemini key: the collection is a
throwaway ChromaDB in a temp directory and the generator is a fake injected via
FastAPI's ``dependency_overrides``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# =========================================================================
# A tiny, dependency-free valid PDF builder (with a correct xref table) so
# upload tests exercise the real pypdf extraction path.
# =========================================================================

def make_pdf_bytes(lines: list[str]) -> bytes:
    ops = ["BT", "/F1 18 Tf", "72 720 Td", "18 TL"]
    for i, line in enumerate(lines):
        esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if i:
            ops.append("T*")
        ops.append(f"({esc}) Tj")
    ops.append("ET")
    content = "\n".join(ops).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_start = len(pdf)
    size = len(objects) + 1
    pdf += f"xref\n0 {size}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += b"trailer\n" + f"<< /Size {size} /Root 1 0 R >>\n".encode()
    pdf += b"startxref\n" + f"{xref_start}\n".encode() + b"%%EOF"
    return bytes(pdf)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Isolate settings + caches to a temp data directory for each test."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from app.config import get_settings
    from app.services import clear_collection_cache, get_generator

    for cached in (get_settings, get_generator):
        cached.cache_clear()
    clear_collection_cache()

    yield tmp_path

    for cached in (get_settings, get_generator):
        cached.cache_clear()
    clear_collection_cache()


@pytest.fixture()
def collection(data_dir):
    import chromadb

    chroma_dir = data_dir / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    db = chromadb.PersistentClient(path=str(chroma_dir))
    return db.get_or_create_collection("domain_documents")


class FakeGenerator:
    """Stand-in for :class:`app.services.GeminiGenerator`."""

    def __init__(self, answer: str = "This is a grounded test answer about RAG.") -> None:
        self.answer = answer
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.answer


@pytest.fixture()
def fake_generator():
    return FakeGenerator()


@pytest.fixture()
def client(data_dir, collection, fake_generator):
    from app.main import app
    from app.services import get_collection, get_generator

    app.dependency_overrides[get_collection] = lambda: collection
    app.dependency_overrides[get_generator] = lambda: fake_generator

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def sample_pdf() -> bytes:
    return make_pdf_bytes(
        [
            "Retrieval Augmented Generation Overview",
            "RAG combines semantic retrieval with a large language model.",
            "It grounds answers in uploaded domain documents.",
        ]
    )
