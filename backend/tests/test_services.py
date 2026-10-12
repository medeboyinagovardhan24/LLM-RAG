"""Unit tests for the pure service-layer helpers."""

from __future__ import annotations

from app.services import (
    build_prompt,
    build_sources,
    calculate_file_hash,
    looks_like_pdf,
    split_text,
)


def test_split_text_respects_overlap():
    text = "abcdefghij"  # 10 chars
    chunks = split_text(text, chunk_size=4, overlap=1)
    # step = 3 -> starts at 0,3,6,9
    assert chunks == ["abcd", "defg", "ghij", "j"]


def test_split_text_empty():
    assert split_text("", chunk_size=100, overlap=10) == []


def test_calculate_file_hash_is_deterministic():
    assert calculate_file_hash(b"hello") == calculate_file_hash(b"hello")
    assert calculate_file_hash(b"hello") != calculate_file_hash(b"world")


def test_looks_like_pdf():
    assert looks_like_pdf(b"%PDF-1.7 ...")
    assert not looks_like_pdf(b"PK\x03\x04")
    assert not looks_like_pdf(b"")


def test_build_prompt_includes_context_and_question():
    docs = ["RAG grounds answers in documents."]
    metas = [{"filename": "rag.pdf", "page": 1, "chunk": 2}]
    prompt = build_prompt("What is RAG?", docs, metas)
    assert "RAG grounds answers in documents." in prompt
    assert "rag.pdf" in prompt
    assert "What is RAG?" in prompt
    assert "RETRIEVED CONTEXT" in prompt


def test_build_sources_dedups():
    metas = [
        {"filename": "a.pdf", "file_hash": "h1", "page": 1, "chunk": 1},
        {"filename": "a.pdf", "file_hash": "h1", "page": 1, "chunk": 1},  # dup
        {"filename": "a.pdf", "file_hash": "h1", "page": 2, "chunk": 1},
    ]
    sources = build_sources(metas)
    assert len(sources) == 2
    assert sources[0] == {"filename": "a.pdf", "page": 1, "content_type": "text", "chunk": 1}


def test_get_collection_serializes_concurrent_initialization(data_dir, monkeypatch):
    import chromadb
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    from types import SimpleNamespace

    import app.services as services

    calls = 0
    calls_lock = Lock()
    start_barrier = Barrier(8)
    collection = object()

    class FakeClient:
        def get_or_create_collection(self, **kwargs):
            return collection

    def create_client(**kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return FakeClient()

    monkeypatch.setattr(
        services,
        "get_settings",
        lambda: SimpleNamespace(
            storage=SimpleNamespace(chroma_dir=data_dir / "chroma_db"),
            llm=SimpleNamespace(api_key="test-probe", embedding_model="unused"),
        ),
    )
    monkeypatch.setattr(chromadb, "PersistentClient", create_client)

    def load_collection():
        start_barrier.wait()
        return services.get_collection()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: load_collection(), range(8)))

    assert calls == 1
    assert all(result is collection for result in results)


def test_generate_stream_retries_server_error_before_first_token(monkeypatch):
    from types import SimpleNamespace

    from google.genai.errors import ServerError

    import app.services as services

    attempts = 0

    def generate_content_stream(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            def fail_before_first_token():
                raise ServerError(503, {"error": {"message": "unavailable"}})
                yield

            return fail_before_first_token()
        return iter([SimpleNamespace(text="Recovered answer")])

    generator = services.GeminiGenerator.__new__(services.GeminiGenerator)
    generator._client = SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=generate_content_stream)
    )
    generator._model = "test-model"
    monkeypatch.setattr(services.time, "sleep", lambda _delay: None)

    assert list(generator.generate_stream("prompt")) == ["Recovered answer"]
    assert attempts == 2


def test_generate_stream_does_not_retry_after_emitting_tokens(monkeypatch):
    import pytest
    from types import SimpleNamespace

    from google.genai.errors import ServerError

    import app.services as services

    attempts = 0

    def generate_content_stream(**kwargs):
        nonlocal attempts
        attempts += 1

        def fail_after_first_token():
            yield SimpleNamespace(text="Partial answer")
            raise ServerError(503, {"error": {"message": "unavailable"}})

        return fail_after_first_token()

    generator = services.GeminiGenerator.__new__(services.GeminiGenerator)
    generator._client = SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=generate_content_stream)
    )
    generator._model = "test-model"
    monkeypatch.setattr(services.time, "sleep", lambda _delay: None)

    with pytest.raises(ServerError):
        list(generator.generate_stream("prompt"))
    assert attempts == 1
