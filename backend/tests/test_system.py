"""Tests for the System endpoints (root, health, status)."""

from __future__ import annotations

import pytest


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "running" in resp.json()["message"].lower()


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_status_empty(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "online"
    assert body["documents"] == 0
    assert body["chunks_indexed"] == 0
    assert body["llm_configured"] is False


def test_status_after_upload(client, sample_pdf):
    client.post("/api/upload", files={"file": ("doc.pdf", sample_pdf, "application/pdf")})
    body = client.get("/api/status").json()
    assert body["documents"] == 1
    assert body["chunks_indexed"] >= 1


def test_default_cors_origins_exclude_null(monkeypatch):
    from app.config import _resolve_allowed_origins

    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    assert "null" not in _resolve_allowed_origins()


def test_reset_analytics_preserves_sessions_and_documents(client, sample_pdf, monkeypatch, tmp_path):
    import app.main as main
    import app.memory as memory

    monkeypatch.setattr(memory, "PERSIST_FILE", tmp_path / "memory_store.json")
    store = memory.MemoryStore()
    store.add_turn("session-1", "A saved question", "A saved answer", latency_ms=250)
    store.record_feedback("up", "A saved question")
    monkeypatch.setattr(main, "memory_store", store)

    upload = client.post("/api/upload", files={"file": ("doc.pdf", sample_pdf, "application/pdf")})
    assert upload.status_code == 200

    response = client.post("/api/analytics/reset")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert store.sessions["session-1"] == [{
        "user": "A saved question",
        "assistant": "A saved answer",
        "ts": store.sessions["session-1"][0]["ts"],
    }]

    analytics = client.get("/api/analytics").json()
    assert analytics["total_queries"] == 0
    assert analytics["thumbs_up"] == 0
    assert analytics["thumbs_down"] == 0
    assert analytics["avg_latency_ms"] == 0
    assert analytics["recent_queries"] == []
    assert analytics["popular_documents"] == [{"filename": "doc.pdf", "chunks": 1}]


def test_memory_sessions_are_isolated_and_bounded(monkeypatch, tmp_path):
    import app.memory as memory

    monkeypatch.setattr(memory, "PERSIST_FILE", tmp_path / "memory_store.json")
    monkeypatch.setattr(memory, "MAX_SESSIONS", 2)
    monkeypatch.setattr(memory, "MAX_TURNS_PER_SESSION", 1)
    store = memory.MemoryStore()
    store.add_turn("session-a", "first", "answer 1")
    store.add_turn("session-a", "second", "answer 2")
    assert store.get_session("session-a")[-1]["user"] == "second"
    store.add_turn("session-b", "other", "answer 3")
    store.add_turn("session-c", "newer", "answer 4")

    assert store.get_session("session-a") == []
    assert store.get_session("session-b")[-1]["user"] == "other"
    assert store.get_session("session-c")[-1]["user"] == "newer"
    assert len(store.queries) == 4


def test_memory_store_persists_and_restores_bounded_data(monkeypatch, tmp_path):
    import app.memory as memory

    monkeypatch.setattr(memory, "PERSIST_FILE", tmp_path / "memory_store.json")
    store = memory.MemoryStore()
    store.add_turn("session-a", "question", "answer", latency_ms=12)
    store.record_feedback("up", "question")

    restored = memory.MemoryStore()

    assert restored.get_session("session-a")[0]["user"] == "question"
    assert restored.analytics_snapshot()["thumbs_up"] == 1
    assert restored.analytics_snapshot()["avg_latency_ms"] == 12


def test_request_monitor_counts_concurrent_requests():
    from concurrent.futures import ThreadPoolExecutor

    from app.cache_monitor import RequestMonitor

    monitor = RequestMonitor()
    total = 500

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(
            lambda index: monitor.record_request(
                "/api/ask", "POST", 500 if index % 5 == 0 else 200, 10.0
            ),
            range(total),
        ))

    metrics = monitor.get_metrics()
    assert metrics["total_requests"] == total
    assert metrics["endpoint_breakdown"]["/api/ask"]["count"] == total
    assert metrics["endpoint_breakdown"]["/api/ask"]["errors"] == total // 5


def test_response_cache_key_includes_conversation_and_options():
    from app.cache_monitor import ResponseCache

    first = ResponseCache.cache_key(
        "question",
        request_context={"session_id": "session-a", "use_guard": True},
    )
    other_session = ResponseCache.cache_key(
        "question",
        request_context={"session_id": "session-b", "use_guard": True},
    )
    other_options = ResponseCache.cache_key(
        "question",
        request_context={"session_id": "session-a", "use_guard": False},
    )

    assert first != other_session
    assert first != other_options


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("MAX_UPLOAD_MB", "-1", "MAX_UPLOAD_MB"),
        ("CHUNK_SIZE", "0", "CHUNK_SIZE"),
        ("CHUNK_OVERLAP", "800", "CHUNK_OVERLAP"),
        ("TOP_K", "0", "TOP_K"),
    ],
)
def test_invalid_resource_configuration_is_rejected(monkeypatch, variable, value, message):
    from app.config import AppConfig

    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=message):
        AppConfig()
