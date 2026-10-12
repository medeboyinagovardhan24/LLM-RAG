"""Tests for the Q&A endpoint."""

from __future__ import annotations


def _upload(client, data: bytes, name: str = "doc.pdf"):
    return client.post("/api/upload", files={"file": (name, data, "application/pdf")})


def test_ask_without_documents_returns_400(client):
    resp = client.post("/api/ask", json={"question": "What is RAG?"})
    assert resp.status_code == 400
    assert "upload" in resp.json()["detail"].lower()


def test_ask_blank_question_rejected(client):
    # Empty string fails Pydantic min_length validation.
    resp = client.post("/api/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_whitespace_question_rejected(client, sample_pdf):
    _upload(client, sample_pdf)
    resp = client.post("/api/ask", json={"question": "   "})
    assert resp.status_code == 422


def test_ask_with_documents_returns_answer_and_sources(client, sample_pdf, fake_generator):
    _upload(client, sample_pdf, name="rag.pdf")

    resp = client.post("/api/ask", json={"question": "What is RAG?"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    assert body["answer"] == fake_generator.answer
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["filename"] == "rag.pdf"

    # The generator must have received grounded context, not the bare question.
    assert "RETRIEVED CONTEXT" in fake_generator.last_prompt
    assert "What is RAG?" in fake_generator.last_prompt


def test_ask_generation_failure_returns_502(client, sample_pdf):
    from app.main import app
    from app.services import get_generator
    from app.cache_monitor import response_cache

    class BoomGenerator:
        def generate(self, prompt):
            raise RuntimeError("model unavailable")

    _upload(client, sample_pdf)
    response_cache.clear()  # ensure cache doesn't mask the error
    app.dependency_overrides[get_generator] = lambda: BoomGenerator()

    resp = client.post("/api/ask", json={"question": "What is RAG?"})
    assert resp.status_code == 502


def test_stream_generation_failure_returns_temporary_error_event(client, sample_pdf):
    from app.main import app
    from app.services import get_generator
    from google.genai.errors import ServerError

    class UnavailableGenerator:
        def generate_stream(self, prompt):
            raise ServerError(503, {"error": {"message": "temporarily unavailable"}})

    _upload(client, sample_pdf)
    app.dependency_overrides[get_generator] = lambda: UnavailableGenerator()

    response = client.post(
        "/api/ask/stream",
        json={
            "question": "What is RAG?",
            "use_query_rewriting": False,
            "use_guard": False,
        },
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "Gemini is temporarily overloaded" in response.text


def test_stream_503_falls_back_to_non_stream_generation(client, sample_pdf):
    from app.main import app
    from app.services import get_generator
    from google.genai.errors import ServerError

    class FallbackGenerator:
        def generate_stream(self, prompt):
            raise ServerError(503, {"error": {"message": "temporarily unavailable"}})

        def generate(self, prompt):
            return "Recovered answer from the non-streaming endpoint."

    _upload(client, sample_pdf)
    app.dependency_overrides[get_generator] = lambda: FallbackGenerator()

    response = client.post(
        "/api/ask/stream",
        json={
            "question": "What is RAG?",
            "use_query_rewriting": False,
            "use_guard": False,
        },
    )

    assert response.status_code == 200
    assert "event: done" in response.text
    assert "Recovered answer from the non-streaming endpoint." in response.text
