"""Unit tests for multimodal uploads (images, tables), source content types, and multilingual support."""

from __future__ import annotations

import io
import json
from PIL import Image
from app.services import (
    extract_tables_as_markdown,
    extract_tables_structured,
    looks_like_image,
    build_sources,
    CT_TABLE,
    CT_IMAGE,
    CT_TEXT,
    CT_OCR,
)
from app.hybrid_search import _tokenize


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def test_looks_like_image():
    buf = io.BytesIO()
    img = Image.new("RGB", (30, 30), color="red")
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    assert looks_like_image(png_bytes) is True
    assert looks_like_image(b"not an image file") is False


def test_upload_valid_image(client):
    buf = io.BytesIO()
    img = Image.new("RGB", (60, 60), color="green")
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    resp = client.post(
        "/api/upload",
        files={"file": ("chart.png", png_bytes, "image/png")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["filename"] == "chart.png"
    assert body["pages"] == 1
    assert body["chunks"] >= 1


# ---------------------------------------------------------------------------
# Table extraction — pdfplumber helper
# ---------------------------------------------------------------------------

def test_extract_tables_as_markdown():
    class DummyPage:
        def extract_tables(self):
            return [
                [
                    ["Product", "Price", "Stock"],
                    ["Laptop",  "$999",  "15"],
                    ["Mouse",   "$25",   "100"],
                ]
            ]

    md = extract_tables_as_markdown(DummyPage())
    assert "| Product | Price | Stock |" in md
    assert "| --- | --- | --- |" in md
    assert "| Laptop | $999 | 15 |" in md
    assert "| Mouse | $25 | 100 |" in md


def test_extract_tables_structured():
    class DummyPage:
        def extract_tables(self):
            return [
                [
                    ["Name",    "Department", "Score"],
                    ["Alice",   "CS",         "92"],
                    ["Bob",     "EE",         "88"],
                ]
            ]

    tables = extract_tables_structured(DummyPage(), page_num=3)
    assert len(tables) == 1
    t = tables[0]
    assert t["table_number"] == 1
    assert t["page"] == 3
    assert "Name" in t["headers"]
    assert t["row_count"] == 2
    assert "Alice" in t["markdown"]
    # JSON data should be parseable
    rows = json.loads(t["json_data"])
    assert rows[0]["Name"] == "Alice"
    assert rows[0]["Score"] == "92"


def test_extract_tables_structured_empty():
    class EmptyPage:
        def extract_tables(self):
            return []

    tables = extract_tables_structured(EmptyPage(), page_num=1)
    assert tables == []


# ---------------------------------------------------------------------------
# Source building — content_type propagation
# ---------------------------------------------------------------------------

def test_build_sources_text():
    metadatas = [
        {"filename": "doc.pdf", "page": 1, "chunk": 1, "content_type": CT_TEXT},
        {"filename": "doc.pdf", "page": 1, "chunk": 2, "content_type": CT_TEXT},
    ]
    sources = build_sources(metadatas)
    assert len(sources) == 2
    assert sources[0]["content_type"] == CT_TEXT
    assert sources[0]["chunk"] == 1


def test_build_sources_table():
    metadatas = [
        {
            "filename": "doc.pdf",
            "page": 2,
            "chunk": 1,
            "content_type": CT_TABLE,
            "table_number": 1,
            "table_markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |",
            "table_headers": "A, B",
        }
    ]
    sources = build_sources(metadatas)
    assert len(sources) == 1
    src = sources[0]
    assert src["content_type"] == CT_TABLE
    assert src["table_number"] == 1
    assert "| A | B |" in src["table_markdown"]
    assert src["table_headers"] == "A, B"
    # chunk should NOT be in table sources
    assert "chunk" not in src


def test_build_sources_image():
    metadatas = [
        {
            "filename": "doc.pdf",
            "page": 5,
            "chunk": 1,
            "content_type": CT_IMAGE,
            "image_id": "abc123_p5_img1",
            "image_filename": "abc123_p5_img1.png",
            "image_description": "A flowchart showing the system architecture.",
            "ocr_text": "System Architecture",
        }
    ]
    sources = build_sources(metadatas)
    assert len(sources) == 1
    src = sources[0]
    assert src["content_type"] == CT_IMAGE
    assert src["image_id"] == "abc123_p5_img1"
    assert src["image_filename"] == "abc123_p5_img1.png"
    assert "flowchart" in src["image_description"]


def test_build_sources_deduplication():
    metadatas = [
        {"filename": "doc.pdf", "page": 1, "chunk": 1, "content_type": CT_TEXT},
        {"filename": "doc.pdf", "page": 1, "chunk": 1, "content_type": CT_TEXT},  # duplicate
        {"filename": "doc.pdf", "page": 1, "chunk": 2, "content_type": CT_TEXT},
    ]
    sources = build_sources(metadatas)
    assert len(sources) == 2  # deduplicated


# ---------------------------------------------------------------------------
# Multilingual BM25 tokenization
# ---------------------------------------------------------------------------

def test_multilingual_bm25_tokenization():
    spanish = "¿Cuál es el precio del café?"
    tokens_es = _tokenize(spanish)
    assert "precio" in tokens_es
    assert "café" in tokens_es

    german = "Die Überwachung und Größe"
    tokens_de = _tokenize(german)
    assert "überwachung" in tokens_de
    assert "größe" in tokens_de

    french = "L'intelligence artificielle et l'éducation"
    tokens_fr = _tokenize(french)
    assert "intelligence" in tokens_fr
    assert "artificielle" in tokens_fr
