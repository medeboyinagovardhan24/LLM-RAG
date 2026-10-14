"""Core RAG services: Document processing (PDF, tables, images, OCR), vector store access and generation.

This module handles:
- Text extraction with pdfplumber (preserving structure)
- Table extraction as separate, retrievable Markdown chunks
- Embedded image extraction from PDF pages saved to uploads/images/
- Gemini Vision analysis of extracted images (descriptions + OCR)
- Scanned-page OCR via Gemini Vision
- ChromaDB vector store management
- Prompt building for multimodal context
"""

from __future__ import annotations

import io
import hashlib
import logging
import threading
import time
import uuid
import re
import json
from functools import lru_cache
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import chromadb
from google import genai
from google.genai.errors import ServerError
from pypdf import PdfReader
import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

from .config import AppConfig, get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "domain_documents"
PDF_MAGIC = b"%PDF"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Content types stored in chunk metadata
CT_TEXT  = "text"
CT_TABLE = "table"
CT_IMAGE = "image"
CT_OCR   = "ocr"


# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Multilingual embedding function using Google Gemini embedding model."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001") -> None:
        self.api_key = api_key
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not input:
            return []
        embeddings: list[list[float]] = []
        batch_size = 32
        for i in range(0, len(input), batch_size):
            batch = list(input[i : i + batch_size])
            try:
                res = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                )
                for item in res.embeddings:
                    embeddings.append(item.values)
            except Exception as e:
                # Never silently insert zero/placeholder vectors: ChromaDB would
                # store them and quietly corrupt similarity search for every
                # future query. Fail loudly so the upload is rejected and can be
                # retried instead of indexing unusable embeddings.
                logger.error("Gemini embedding batch failed: %s", e)
                raise RuntimeError(
                    "Failed to generate embeddings via the Gemini API; "
                    "the document was not indexed. Please retry."
                ) from e
        return embeddings


# ---------------------------------------------------------------------------
# ChromaDB collection (cached singleton)
# ---------------------------------------------------------------------------

_collection_lock = threading.Lock()


@lru_cache
def _get_collection_cached():
    settings = get_settings()
    settings.storage.chroma_dir.mkdir(parents=True, exist_ok=True)

    embedding_fn = None
    if settings.llm.api_key and not settings.llm.api_key.startswith("test-"):
        try:
            embedding_fn = GeminiEmbeddingFunction(
                api_key=settings.llm.api_key,
                model=settings.llm.embedding_model,
            )
        except Exception as ex:
            logger.warning("Could not initialize Gemini embedding function: %s", ex)

    def _open_or_reset(client, name, fn):
        try:
            if fn:
                return client.get_or_create_collection(name=name, embedding_function=fn)
            return client.get_or_create_collection(name=name)
        except Exception as exc:
            logger.warning("Resetting collection %s due to metadata/dimension conflict: %s", name, exc)
            try:
                client.delete_collection(name=name)
            except Exception:
                pass
            if fn:
                return client.get_or_create_collection(name=name, embedding_function=fn)
            return client.get_or_create_collection(name=name)

    client = chromadb.PersistentClient(
        path=str(settings.storage.chroma_dir),
        tenant="default_tenant",
        database="default_database",
    )
    return _open_or_reset(client, COLLECTION_NAME, embedding_fn)


def get_collection():
    with _collection_lock:
        return _get_collection_cached()


def clear_collection_cache():
    with _collection_lock:
        _get_collection_cached.cache_clear()


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def calculate_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def looks_like_pdf(file_bytes: bytes) -> bool:
    return file_bytes[:4] == PDF_MAGIC


def looks_like_image(file_bytes: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
        return True
    except Exception:
        return False


def document_already_indexed(collection, file_hash: str) -> bool:
    if collection.count() == 0:
        return False
    results = collection.get(where={"file_hash": file_hash}, limit=1)
    return len(results.get("ids", [])) > 0


# ---------------------------------------------------------------------------
# Table extraction (pdfplumber)
# ---------------------------------------------------------------------------

def extract_tables_as_markdown(page) -> str:
    """Extract tables using pdfplumber and convert to Markdown table format."""
    try:
        tables = page.extract_tables() or []
    except Exception:
        return ""
    if not tables:
        return ""

    md_tables = []
    for t_idx, table in enumerate(tables, start=1):
        if not table:
            continue
        cleaned = []
        for row in table:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue
            cleaned.append([str(c).strip().replace("\n", " ") if c is not None else "" for c in row])
        if not cleaned:
            continue
        max_cols = max(len(r) for r in cleaned)
        header = cleaned[0] + [""] * (max_cols - len(cleaned[0]))
        rows = cleaned[1:] if len(cleaned) > 1 else []
        table_str = f"[Table {t_idx}]:\n| " + " | ".join(header) + " |\n"
        table_str += "| " + " | ".join(["---"] * max_cols) + " |\n"
        for r in rows:
            r = r + [""] * (max_cols - len(r))
            table_str += "| " + " | ".join(r) + " |\n"
        md_tables.append(table_str)

    return "\n\n".join(md_tables)


def extract_tables_structured(page, page_num: int) -> list[dict]:
    """Extract tables as structured dicts for separate chunk storage.

    Returns a list of dicts with keys:
        table_number, page, markdown, headers, row_count, json_data
    """
    try:
        tables = page.extract_tables() or []
    except Exception:
        return []

    result = []
    for t_idx, table in enumerate(tables, start=1):
        if not table:
            continue
        cleaned = []
        for row in table:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue
            cleaned.append([str(c).strip().replace("\n", " ") if c is not None else "" for c in row])
        if not cleaned:
            continue

        max_cols = max(len(r) for r in cleaned)
        header = cleaned[0] + [""] * (max_cols - len(cleaned[0]))
        rows = cleaned[1:] if len(cleaned) > 1 else []

        # Markdown representation
        md = f"[Table {t_idx} — Page {page_num}]:\n| " + " | ".join(header) + " |\n"
        md += "| " + " | ".join(["---"] * max_cols) + " |\n"
        for r in rows:
            r = r + [""] * (max_cols - len(r))
            md += "| " + " | ".join(r) + " |\n"

        # JSON representation for keyword matching
        json_rows = []
        for r in rows:
            row_dict = {}
            for col_idx, col_name in enumerate(header):
                row_dict[col_name or f"col_{col_idx}"] = r[col_idx] if col_idx < len(r) else ""
            json_rows.append(row_dict)

        result.append({
            "table_number": t_idx,
            "page": page_num,
            "markdown": md,
            "headers": header,
            "row_count": len(rows),
            "json_data": json.dumps(json_rows, ensure_ascii=False),
        })
    return result


# ---------------------------------------------------------------------------
# Image extraction from PDF pages (pypdfium2)
# ---------------------------------------------------------------------------

def get_images_dir(settings: AppConfig) -> Path:
    """Return and create the directory where extracted images are stored."""
    images_dir = settings.storage.upload_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def extract_embedded_images_from_page(
    pdf_doc,
    page_index: int,
    page_num: int,
    file_hash: str,
    settings: AppConfig,
    max_images: int = 3,
) -> list[dict]:
    """Render each page to find image regions.

    We render the full page at medium resolution and crop out significant
    image-heavy sub-regions. For PDFs with embedded image objects, we also
    attempt direct object extraction using pypdfium2's bitmap approach.

    Returns a list of dicts with keys:
        image_id, image_path (relative to upload_dir), page, width, height
    """
    images_dir = get_images_dir(settings)
    extracted = []

    try:
        page = pdf_doc[page_index]
        # Render the full page as an image at 150dpi equivalent (scale=2.0 = ~144dpi)
        pil_page = page.render(scale=2.0).to_pil()

        # Try to enumerate embedded image objects via the page's object list
        obj_images = []
        try:
            for obj in page.get_objects():
                obj_type = obj.type
                # FPDF_PAGEOBJ_IMAGE = 1
                if hasattr(obj_type, 'value'):
                    is_image_obj = (obj_type.value == 1)
                else:
                    is_image_obj = (int(obj_type) == 1)
                if is_image_obj:
                    obj_images.append(obj)
        except Exception:
            pass

        if obj_images:
            # Extract and save each image object
            for img_idx, img_obj in enumerate(obj_images[:max_images]):
                try:
                    bounds = img_obj.get_pos()
                    # bounds = (left, bottom, right, top) in PDF user space
                    page_h = pil_page.height
                    page_w = pil_page.width
                    page_height_pts = page.get_height()
                    page_width_pts = page.get_width()

                    # Scale PDF pts to pixel coords
                    sx = page_w / page_width_pts
                    sy = page_h / page_height_pts

                    left   = int(bounds[0] * sx)
                    bottom = int(bounds[1] * sy)
                    right  = int(bounds[2] * sx)
                    top    = int(bounds[3] * sy)

                    # PIL origin is top-left; PDF origin is bottom-left
                    pil_top    = page_h - top
                    pil_bottom = page_h - bottom

                    if pil_top >= pil_bottom or left >= right:
                        continue
                    crop = pil_page.crop((left, pil_top, right, pil_bottom))

                    if crop.width < 30 or crop.height < 30:
                        continue  # Skip tiny icons

                    image_id = f"{file_hash[:12]}_p{page_num}_img{img_idx+1}"
                    img_filename = f"{image_id}.png"
                    img_path = images_dir / img_filename

                    if crop.mode not in ("RGB", "RGBA"):
                        crop = crop.convert("RGB")
                    crop.save(str(img_path), format="PNG", optimize=True)

                    extracted.append({
                        "image_id": image_id,
                        "image_filename": img_filename,
                        "page": page_num,
                        "width": crop.width,
                        "height": crop.height,
                    })
                except Exception as e:
                    logger.debug("Failed to extract image object from page %d: %s", page_num, e)

        # Fallback: If no images were extracted as objects, try extracting the whole page.
        if not extracted:
            image_id = f"{file_hash[:12]}_p{page_num}_full"
            img_filename = f"{image_id}.png"
            img_path = images_dir / img_filename
            if pil_page.mode not in ("RGB", "RGBA"):
                pil_page = pil_page.convert("RGB")
            pil_page.save(str(img_path), format="PNG", optimize=True)
            extracted.append({
                "image_id": image_id,
                "image_filename": img_filename,
                "page": page_num,
                "width": pil_page.width,
                "height": pil_page.height,
            })
            logger.info("Extracted full page as fallback image for page %d", page_num)

    except Exception as e:
        logger.debug("Image extraction from page %d failed: %s", page_num, e)

    return extracted


# ---------------------------------------------------------------------------
# Gemini Vision — image description
# ---------------------------------------------------------------------------

def describe_image_with_gemini(
    img_path: Path,
    client,
    model: str,
    context_hint: str = "",
) -> tuple[str, str]:
    """Generate a description and OCR text for an image using Gemini Vision.

    Returns:
        (description, ocr_text) — both may be empty strings on failure.
    """
    description = ""
    ocr_text = ""
    try:
        with Image.open(img_path) as pil_img:
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")

            prompt_parts = [
                pil_img,
                (
                    "Analyze this image from a PDF document for a knowledge retrieval system.\n"
                    "1. Describe what the image shows in detail: objects, layout, colors, labels.\n"
                    "2. If it is a diagram, flowchart, or architecture diagram — describe every component, "
                    "arrow, relationship, and label.\n"
                    "3. If it is a chart or graph — describe the type, axes, data trends, and legend.\n"
                    "4. If it is a table rendered as an image — transcribe it as Markdown.\n"
                    "5. Transcribe ALL visible text, numbers, and annotations (OCR).\n"
                    "6. Do not invent content that is not visible.\n"
                    + (f"Context: {context_hint}" if context_hint else "")
                ),
            ]

            res = client.models.generate_content(model=model, contents=prompt_parts)
            raw = (res.text or "").strip()
            if raw:
                description = raw
                # Also treat all text as OCR for keyword search
                ocr_text = raw
    except Exception as e:
        logger.warning("Gemini Vision description failed for %s: %s", img_path, e)

    return description, ocr_text


# ---------------------------------------------------------------------------
# PDF ingestion pipeline
# ---------------------------------------------------------------------------

def read_pdf_into_chunks(
    file_path: Path, filename: str, file_hash: str, settings: AppConfig
) -> tuple[int, int, list[str], list[dict], list[str]]:
    """Multimodal PDF ingestion pipeline.

    Produces three types of chunks:
    1. text   — regular paragraph/heading text per page
    2. table  — one chunk per table, stored with full Markdown + JSON
    3. image  — Gemini Vision description for each extracted embedded image

    Scanned pages (< 40 chars) fall back to Gemini Vision OCR stored as
    content_type='ocr'.
    """
    chunks: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    ocr_page_nums: set[int] = set()   # pages whose text came from Gemini Vision OCR
    total_chars = 0
    total_pages = 0

    # ── 1. Primary text + table extraction with pdfplumber ──────────────────
    extracted_text_pages: list[tuple[int, str]] = []   # (page_num, plain_text)
    extracted_table_pages: list[tuple[int, list[dict]]] = []   # (page_num, tables)

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = (page.extract_text() or "").strip()
                tables = extract_tables_structured(page, page_num)
                extracted_text_pages.append((page_num, raw_text))
                extracted_table_pages.append((page_num, tables))
    except Exception as e:
        logger.warning("pdfplumber failed for %s, falling back to pypdf: %s", filename, e)
        try:
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                extracted_text_pages.append((page_num, text))
                extracted_table_pages.append((page_num, []))
        except Exception as e2:
            logger.error("pypdf also failed for %s: %s", filename, e2)
            return 0, 0, [], [], []

    has_gemini = bool(settings.llm.api_key and not settings.llm.api_key.startswith("test-"))
    client = genai.Client(api_key=settings.llm.api_key) if has_gemini else None

    # ── 2. Scanned-page OCR via Gemini Vision ───────────────────────────────
    try:
        pdf_doc = pdfium.PdfDocument(str(file_path))
        for idx, (page_num, page_text) in enumerate(extracted_text_pages):
            if len(page_text.strip()) < 40 and has_gemini and client:
                try:
                    p = pdf_doc[page_num - 1]
                    pil_img = p.render(scale=2.0).to_pil()
                    vision_res = client.models.generate_content(
                        model=settings.llm.model,
                        contents=[
                            pil_img,
                            "Transcribe all visible text, headings, and data from this scanned document page. "
                            "Format any tables as Markdown tables. "
                            "Describe and explain charts, diagrams, or visual figures in detail.",
                        ],
                    )
                    v_text = (vision_res.text or "").strip()
                    if v_text:
                        extracted_text_pages[idx] = (page_num, v_text)
                        ocr_page_nums.add(page_num)
                except Exception as ve:
                    logger.warning("Gemini Vision OCR failed for page %d in %s: %s", page_num, filename, ve)
    except Exception as pe:
        logger.warning("pypdfium2 failed to open %s: %s", filename, pe)
        pdf_doc = None

    # ── 3. Embedded image extraction + Gemini descriptions ──────────────────
    image_chunks_data: list[tuple[str, dict]] = []  # (text_to_embed, metadata)

    if has_gemini and client and pdf_doc is not None:
            # Extract images from ALL pages.
            for idx, (page_num, page_text) in enumerate(extracted_text_pages):
                try:
                    logger.info("Extracting images from page %d", page_num)
                    page_images = extract_embedded_images_from_page(
                        pdf_doc, page_num - 1, page_num, file_hash, settings
                    )
                    logger.info("Found %d images on page %d", len(page_images), page_num)
                    for img_info in page_images:
                        img_path = settings.storage.upload_dir / "images" / img_info["image_filename"]
                        description, ocr_text = describe_image_with_gemini(
                            img_path, client, settings.llm.model,
                            context_hint=f"From '{filename}', page {page_num}"
                        )
                        embed_text = description or ocr_text
                        if not embed_text:
                            # Vision failed — note the limitation
                            embed_text = (
                                f"[Image on page {page_num} of {filename} — "
                                "Gemini Vision analysis was unavailable for this image.]"
                            )
                        image_chunks_data.append((
                            embed_text,
                            {
                                "filename": filename,
                                "file_hash": file_hash,
                                "page": page_num,
                                "chunk": 1,
                                "content_type": CT_IMAGE,
                                "image_id": img_info["image_id"],
                                "image_filename": img_info["image_filename"],
                                "image_description": description[:500] if description else "",
                                "ocr_text": ocr_text[:500] if ocr_text else "",
                            }
                        ))
                except Exception as e:
                    logger.debug("Image extraction pipeline error on page %d: %s", page_num, e)

    # ── 4. Assemble text chunks ──────────────────────────────────────────────
    for page_num, text in extracted_text_pages:
        if not text:
            continue
        total_chars += len(text)
        page_chunks = split_text(text, settings.rag.chunk_size, settings.rag.chunk_overlap)
        # Pages whose text came from Gemini Vision OCR (originally < 40 chars)
        # are labelled CT_OCR so retrieval and analytics can distinguish them.
        content_type = CT_OCR if page_num in ocr_page_nums else CT_TEXT
        for chunk_num, chunk in enumerate(page_chunks, start=1):
            chunks.append(chunk)
            metadatas.append({
                "filename": filename,
                "file_hash": file_hash,
                "page": page_num,
                "chunk": chunk_num,
                "content_type": content_type,
            })
            ids.append(f"{file_hash[:12]}_{page_num}_{chunk_num}_{uuid.uuid4().hex[:8]}")

    # ── 5. Assemble table chunks ─────────────────────────────────────────────
    for page_num, tables in extracted_table_pages:
        for table_info in tables:
            t_num = table_info["table_number"]
            md = table_info["markdown"]
            json_data = table_info["json_data"]
            headers_str = ", ".join(table_info["headers"])

            # Embed text = markdown + JSON for maximum keyword coverage
            embed_text = (
                f"Table {t_num} (Page {page_num}) from {filename}:\n"
                f"Columns: {headers_str}\n\n"
                f"{md}\n\nStructured data:\n{json_data}"
            )
            total_chars += len(embed_text)
            chunk_id = f"{file_hash[:12]}_t{page_num}_{t_num}_{uuid.uuid4().hex[:8]}"
            chunks.append(embed_text)
            metadatas.append({
                "filename": filename,
                "file_hash": file_hash,
                "page": page_num,
                "chunk": t_num,
                "content_type": CT_TABLE,
                "table_number": t_num,
                "table_markdown": md,
                "table_headers": headers_str,
                "row_count": table_info["row_count"],
            })
            ids.append(chunk_id)

    # ── 6. Assemble image chunks ─────────────────────────────────────────────
    for img_idx, (embed_text, meta) in enumerate(image_chunks_data):
        meta["chunk"] = img_idx + 1
        total_chars += len(embed_text)
        chunks.append(embed_text)
        metadatas.append(meta)
        ids.append(f"{file_hash[:12]}_img_{img_idx+1}_{uuid.uuid4().hex[:8]}")

    logger.info(
        "Parsed %s: %d pages, %d text chunks, %d table chunks, %d image chunks",
        filename, total_pages,
        sum(1 for m in metadatas if m.get("content_type") == CT_TEXT),
        sum(1 for m in metadatas if m.get("content_type") == CT_TABLE),
        sum(1 for m in metadatas if m.get("content_type") == CT_IMAGE),
    )
    return total_pages, total_chars, chunks, metadatas, ids


# ---------------------------------------------------------------------------
# Image file ingestion
# ---------------------------------------------------------------------------

def read_image_into_chunks(
    file_path: Path, filename: str, file_hash: str, settings: AppConfig
) -> tuple[int, int, list[str], list[dict], list[str]]:
    """Process an uploaded image file (.png, .jpg, .jpeg, .webp) via Gemini Vision / OCR."""
    chunks, metadatas, ids = [], [], []
    has_gemini = bool(settings.llm.api_key and not settings.llm.api_key.startswith("test-"))
    extracted_text = ""
    description = ""
    ocr_text = ""

    if has_gemini:
        try:
            client = genai.Client(api_key=settings.llm.api_key)
            description, ocr_text = describe_image_with_gemini(
                file_path, client, settings.llm.model,
                context_hint=f"Uploaded image: {filename}"
            )
            extracted_text = description or ocr_text
        except Exception as e:
            logger.warning("Gemini Vision failed for image %s: %s", filename, e)

    # Fallback: pytesseract OCR
    if not extracted_text:
        try:
            import pytesseract
            with Image.open(file_path) as pil_img:
                extracted_text = pytesseract.image_to_string(pil_img).strip()
                ocr_text = extracted_text
        except Exception:
            pass

    # Fallback for test mode or empty image
    if not extracted_text and (not settings.llm.api_key or settings.llm.api_key.startswith("test-")):
        extracted_text = f"Image content for {filename}: Visual document with data and figures."

    # Copy the image to uploads/images/ for serving
    images_dir = get_images_dir(settings)
    image_id = f"{file_hash[:12]}_uploaded"
    img_filename = f"{image_id}_{Path(filename).stem}.png"
    img_dest = images_dir / img_filename
    try:
        with Image.open(file_path) as pil_img:
            if pil_img.mode not in ("RGB", "RGBA"):
                pil_img = pil_img.convert("RGB")
            pil_img.save(str(img_dest), format="PNG", optimize=True)
    except Exception as e:
        logger.warning("Could not copy uploaded image to images dir: %s", e)
        img_filename = ""

    total_chars = len(extracted_text)
    if extracted_text:
        img_chunks = split_text(extracted_text, settings.rag.chunk_size, settings.rag.chunk_overlap)
        for chunk_number, chunk in enumerate(img_chunks, start=1):
            chunks.append(chunk)
            metadatas.append({
                "filename": filename,
                "file_hash": file_hash,
                "page": 1,
                "chunk": chunk_number,
                "content_type": CT_IMAGE,
                "image_id": image_id,
                "image_filename": img_filename,
                "image_description": description[:500] if description else "",
                "ocr_text": ocr_text[:500] if ocr_text else "",
            })
            ids.append(f"{file_hash[:12]}_1_{chunk_number}_{uuid.uuid4().hex[:8]}")

    return 1, total_chars, chunks, metadatas, ids


# ---------------------------------------------------------------------------
# Document records
# ---------------------------------------------------------------------------

def get_all_document_records(collection) -> list[dict]:
    if collection.count() == 0:
        return []
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas", []) or []
    documents: dict[str, dict] = {}
    for metadata in metadatas:
        if not metadata:
            continue
        filename = metadata.get("filename")
        if not filename:
            continue
        record = documents.setdefault(
            filename,
            {
                "filename": filename,
                "pages": set(),
                "chunks": 0,
                "tables": 0,
                "images": 0,
            }
        )
        if "page" in metadata:
            record["pages"].add(metadata["page"])
        record["chunks"] += 1
        ct = metadata.get("content_type", CT_TEXT)
        if ct == CT_TABLE:
            record["tables"] += 1
        elif ct == CT_IMAGE:
            record["images"] += 1
    return [
        {
            "filename": d["filename"],
            "pages": len(d["pages"]),
            "chunks": d["chunks"],
            "tables": d.get("tables", 0),
            "images": d.get("images", 0),
        }
        for d in documents.values()
    ]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(collection, question: str, top_k: int) -> tuple[list[str], list[dict]]:
    results = collection.query(query_texts=[question], n_results=min(top_k, collection.count()))
    return results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]


# ---------------------------------------------------------------------------
# Prompt building (multimodal-aware)
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    documents: list[str],
    metadatas: list[dict],
    history: list[dict] | None = None,
) -> str:
    context_parts = []
    for d, m in zip(documents, metadatas):
        ct = m.get("content_type", CT_TEXT)
        source_label = f"{m.get('filename', 'Unknown')} (Page {m.get('page', '?')})"

        if ct == CT_TABLE:
            t_num = m.get("table_number", "?")
            headers = m.get("table_headers", "")
            context_parts.append(
                f"[TABLE — {source_label} — Table {t_num}]\n"
                f"Columns: {headers}\n"
                f"Content:\n{d}"
            )
        elif ct == CT_IMAGE:
            img_desc = m.get("image_description", "")
            ocr = m.get("ocr_text", "")
            context_parts.append(
                f"[IMAGE — {source_label}]\n"
                f"Description: {img_desc}\n"
                + (f"OCR text: {ocr}\n" if ocr else "")
                + f"Full analysis:\n{d}"
            )
        elif ct == CT_OCR:
            context_parts.append(
                f"[OCR/SCANNED — {source_label}]\n"
                f"Content (extracted via OCR):\n{d}"
            )
        else:
            context_parts.append(
                f"[TEXT — {source_label} — Chunk {m.get('chunk', '?')}]\n"
                f"Content: {d}"
            )

    context = "\n\n---\n\n".join(context_parts)

    history_text = ""
    if history:
        turns = []
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            turns.append(f"{role.upper()}: {content}")
        history_text = "CONVERSATION HISTORY:\n" + "\n".join(turns) + "\n\n"

    return (
        f"{history_text}"
        "RETRIEVED CONTEXT (text, tables, and image descriptions from uploaded documents):\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Answer thoroughly using ONLY the retrieved context above.\n"
        "- When the answer comes from a table, mention the table number and page.\n"
        "- When the answer comes from an image or diagram, describe what the image shows.\n"
        "- Format tables clearly as Markdown with proper column alignment.\n"
        "- If the information is not present in the retrieved context, say so — do NOT invent data.\n"
        "- Respond in the same language as the user question."
    )


# ---------------------------------------------------------------------------
# Source building (multimodal-aware)
# ---------------------------------------------------------------------------

def build_sources(metadatas: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    sources: list[dict] = []
    for m in metadatas:
        if not m:
            continue
        ct = m.get("content_type", CT_TEXT)
        key = (m.get("filename"), m.get("page"), m.get("chunk"), ct)
        if key in seen:
            continue
        seen.add(key)

        src: dict = {
            "filename": m.get("filename"),
            "page": m.get("page"),
            "content_type": ct,
        }

        if ct == CT_TEXT or ct == CT_OCR:
            src["chunk"] = m.get("chunk")

        if ct == CT_TABLE:
            src["table_number"] = m.get("table_number")
            src["table_markdown"] = m.get("table_markdown", "")
            src["table_headers"] = m.get("table_headers", "")

        if ct == CT_IMAGE:
            src["image_id"] = m.get("image_id", "")
            src["image_filename"] = m.get("image_filename", "")
            src["image_description"] = m.get("image_description", "")

        sources.append(src)
    return sources


# ---------------------------------------------------------------------------
# Image deletion helper
# ---------------------------------------------------------------------------

def delete_document_images(filename: str, collection, settings: AppConfig) -> int:
    """Delete all extracted image files associated with a document.
    Returns the number of image files deleted.
    """
    if collection.count() == 0:
        return 0

    try:
        results = collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        metadatas = results.get("metadatas", []) or []
    except Exception:
        return 0

    images_dir = get_images_dir(settings)
    deleted = 0
    seen_filenames: set[str] = set()

    for m in metadatas:
        if not m:
            continue
        img_fn = m.get("image_filename", "")
        if img_fn and img_fn not in seen_filenames:
            seen_filenames.add(img_fn)
            img_path = images_dir / img_fn
            if img_path.exists():
                try:
                    img_path.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning("Could not delete image file %s: %s", img_path, e)

    return deleted


# ---------------------------------------------------------------------------
# GeminiGenerator
# ---------------------------------------------------------------------------

class GeminiGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        retry=retry_if_exception_type(ServerError),
        before_sleep=lambda retry_state: logger.warning(
            "Gemini API failure (attempt %d), retrying...", retry_state.attempt_number
        ),
    )
    def generate(self, prompt: str) -> str:
        logger.debug("Sending prompt to Gemini: %s...", prompt[:100])
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        logger.debug("Received response: %s", response.text[:100] if response.text else "None")
        return response.text or ""

    def generate_stream(self, prompt: str):
        for attempt in range(3):
            emitted_token = False
            try:
                response = self._client.models.generate_content_stream(
                    model=self._model,
                    contents=prompt,
                )
                for chunk in response:
                    text = chunk.text or ""
                    if text:
                        emitted_token = True
                        yield text
                return
            except ServerError:
                if emitted_token or attempt == 2:
                    raise
                delay = min(2 ** (attempt + 1), 10)
                logger.warning(
                    "Gemini stream failed before first token; retrying in %s seconds",
                    delay,
                    exc_info=True,
                )
                time.sleep(delay)


@lru_cache
def get_generator() -> GeminiGenerator:
    settings = get_settings()
    return GeminiGenerator(settings.llm.api_key, settings.llm.model)
