"""Tests for the Documents endpoints (upload, list, get, delete)."""

from __future__ import annotations


def _upload(client, data: bytes, name: str = "doc.pdf", content_type: str = "application/pdf"):
    return client.post("/api/upload", files={"file": (name, data, content_type)})


# --- Upload ---------------------------------------------------------------

def test_upload_valid_pdf(client, sample_pdf):
    resp = _upload(client, sample_pdf)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["duplicate"] is False
    assert body["filename"] == "doc.pdf"
    assert body["pages"] == 1
    assert body["chunks"] >= 1


def test_upload_rejects_non_pdf_extension(client):
    resp = _upload(client, b"hello", name="notes.txt", content_type="text/plain")
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


def test_upload_rejects_empty_file(client):
    resp = _upload(client, b"")
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_upload_rejects_fake_pdf_content(client):
    resp = _upload(client, b"this is not really a pdf")
    assert resp.status_code == 400
    assert "valid PDF" in resp.json()["detail"]


def test_upload_rejects_unreadable_pdf(client):
    # Correct magic bytes but not a parseable document.
    resp = _upload(client, b"%PDF-1.4\nnot a real structure")
    assert resp.status_code == 422


def test_upload_rejects_oversized(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()

    big = b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024)  # ~2 MB, valid magic bytes
    resp = _upload(client, big, name="big.pdf")
    assert resp.status_code == 413
    assert "limit" in resp.json()["detail"].lower()


def test_upload_detects_duplicate(client, sample_pdf):
    first = _upload(client, sample_pdf)
    assert first.json()["success"] is True

    second = _upload(client, sample_pdf)
    assert second.status_code == 200
    assert second.json()["duplicate"] is True


# --- List / get / delete --------------------------------------------------

def test_list_documents(client, sample_pdf):
    assert client.get("/api/documents").json()["count"] == 0

    _upload(client, sample_pdf, name="rag.pdf")

    body = client.get("/api/documents").json()
    assert body["count"] == 1
    assert body["documents"][0]["filename"] == "rag.pdf"
    assert body["documents"][0]["chunks"] >= 1


def test_get_single_document(client, sample_pdf):
    _upload(client, sample_pdf, name="rag.pdf")
    resp = client.get("/api/documents/rag.pdf")
    assert resp.status_code == 200
    assert resp.json()["document"]["filename"] == "rag.pdf"


def test_get_missing_document_404(client):
    resp = client.get("/api/documents/nope.pdf")
    assert resp.status_code == 404


def test_delete_document(client, sample_pdf):
    _upload(client, sample_pdf, name="rag.pdf")

    resp = client.delete("/api/documents/rag.pdf")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["chunks_deleted"] >= 1

    assert client.get("/api/documents").json()["count"] == 0


def test_delete_missing_document_404(client):
    resp = client.delete("/api/documents/nope.pdf")
    assert resp.status_code == 404
