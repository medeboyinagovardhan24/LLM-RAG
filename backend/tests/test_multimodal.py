"""Unit tests for multimodal uploads (images, tables), source content types, and multilingual support."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from PIL import Image
from app import services
from app.services import (
    extract_tables_as_markdown,
    extract_tables_structured,
    extract_embedded_images_from_page,
    looks_like_image,
    read_pdf_into_chunks,
    build_sources,
    build_sources_with_citation_map,
    CT_TABLE,
    CT_IMAGE,
    CT_TEXT,
)
from app.hybrid_search import _tokenize
from app.config import get_settings


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


def test_extract_embedded_images_uses_pdfium_image_object_type(data_dir):
    class FakeImageObject:
        type = services.pdfium.raw.FPDF_PAGEOBJ_IMAGE

        def get_bounds(self):
            return (0, 0, 25, 25)

    class FakeTextObject:
        type = 1

        def get_pos(self):
            return (0, 0, 50, 50)

    class FakePage:
        def render(self, scale):
            assert scale == 2.0
            return self

        def to_pil(self):
            return Image.new("RGB", (100, 100), color="red")

        def get_objects(self):
            return [FakeTextObject(), FakeImageObject()]

        def get_width(self):
            return 50

        def get_height(self):
            return 50

    extracted = extract_embedded_images_from_page(
        [FakePage()], 0, 1, "image-test-hash", get_settings()
    )

    assert len(extracted) == 1
    assert extracted[0]["image_filename"].endswith("_p1_img1.png")
    assert (data_dir / "uploads" / "images" / extracted[0]["image_filename"]).is_file()


def test_pdf_images_are_indexed_without_gemini_key(
    data_dir, sample_pdf, tmp_path, monkeypatch
):
    file_path = tmp_path / "images.pdf"
    file_path.write_bytes(sample_pdf)
    extracted_image = {
        "image_id": "image-without-key",
        "image_filename": "image-without-key.png",
        "page": 1,
    }
    monkeypatch.setattr(
        services,
        "extract_embedded_images_from_page",
        lambda *args: [extracted_image],
    )

    def fail_if_vision_called(*_args, **_kwargs):
        raise AssertionError("Gemini Vision must not be called without an API key")

    monkeypatch.setattr(services, "describe_image_with_gemini", fail_if_vision_called)
    assert get_settings().llm.api_key.startswith("test-")

    _, _, chunks, metadatas, _ = read_pdf_into_chunks(
        file_path, "images.pdf", "test-file-hash", get_settings()
    )

    image_index = next(i for i, metadata in enumerate(metadatas) if metadata["content_type"] == CT_IMAGE)
    assert metadatas[image_index]["image_filename"] == "image-without-key.png"
    assert "GEMINI_API_KEY" in chunks[image_index]


def test_upload_valid_image_uses_local_ocr(client, monkeypatch, data_dir):
    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(image_to_string=lambda _image: "A chart with readable labels."),
    )
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

    chunks_response = client.get("/api/documents/chart.png/chunks")
    assert chunks_response.status_code == 200
    image_filename = chunks_response.json()["chunks"][0]["image_filename"]
    image_response = client.get(f"/api/images/{image_filename}")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"

    delete_response = client.delete("/api/documents/chart.png")
    assert delete_response.status_code == 200
    assert delete_response.json()["images_deleted"] == 1
    assert not (data_dir / "uploads" / "chart.png").exists()
    assert client.get(f"/api/images/{image_filename}").status_code == 404


def test_upload_image_ocr_failure_does_not_index_placeholder(
    client, monkeypatch, data_dir, caplog
):
    def fail_ocr(_image):
        raise RuntimeError("Tesseract executable unavailable")

    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(image_to_string=fail_ocr),
    )
    buf = io.BytesIO()
    Image.new("RGB", (60, 60), color="green").save(buf, format="PNG")

    response = client.post(
        "/api/upload",
        files={"file": ("blank.png", buf.getvalue(), "image/png")},
    )

    assert response.status_code == 422
    assert "Gemini Vision" in response.json()["detail"]
    assert "Local OCR failed" in caplog.text
    assert client.get("/api/documents").json()["count"] == 0
    assert not (data_dir / "uploads" / "blank.png").exists()


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


def test_build_sources_with_citation_map_preserves_dedup_keys():
    metadatas = [
        {"filename": "doc.pdf", "page": 1, "chunk": 1, "content_type": CT_TEXT},
        {"filename": "doc.pdf", "page": 1, "chunk": 1, "content_type": CT_TEXT},
        {"filename": "doc.pdf", "page": 1, "chunk": 2, "content_type": CT_TEXT},
        {"filename": "other.pdf", "page": 1, "chunk": 1, "content_type": CT_TEXT},
        {
            "filename": "other.pdf",
            "page": 1,
            "content_type": CT_TABLE,
            "table_number": 1,
            "table_markdown": "| A |",
        },
        {
            "filename": "other.pdf",
            "page": 1,
            "content_type": CT_TABLE,
            "table_number": 1,
            "table_markdown": "| A |",
        },
        {
            "filename": "other.pdf",
            "page": 1,
            "content_type": CT_TABLE,
            "table_number": 2,
            "table_markdown": "| B |",
        },
        {
            "filename": "other.pdf",
            "page": 1,
            "content_type": CT_IMAGE,
            "image_id": "image-1",
            "image_filename": "image-1.png",
        },
        {
            "filename": "other.pdf",
            "page": 1,
            "content_type": CT_IMAGE,
            "image_id": "image-1",
            "image_filename": "image-1.png",
        },
        {
            "filename": "other.pdf",
            "page": 1,
            "content_type": CT_IMAGE,
            "image_id": "image-2",
            "image_filename": "image-2.png",
        },
        {
            "filename": "doc.pdf",
            "page": 1,
            "chunk": 1,
            "content_type": "ocr",
        },
    ]

    sources, citation_map = build_sources_with_citation_map(metadatas)

    assert citation_map == {
        1: 1,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 4,
        7: 5,
        8: 6,
        9: 6,
        10: 7,
        11: 8,
    }
    assert sources == build_sources(metadatas)
    assert len(sources) == 8
    assert [source["table_number"] for source in sources if source["content_type"] == CT_TABLE] == [1, 2]
    assert [source["image_id"] for source in sources if source["content_type"] == CT_IMAGE] == [
        "image-1",
        "image-2",
    ]


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
