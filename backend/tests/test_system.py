"""Tests for the System endpoints (root, health, status)."""

from __future__ import annotations


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


def test_status_after_upload(client, sample_pdf):
    client.post("/api/upload", files={"file": ("doc.pdf", sample_pdf, "application/pdf")})
    body = client.get("/api/status").json()
    assert body["documents"] == 1
    assert body["chunks_indexed"] >= 1


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
