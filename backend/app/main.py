"""FastAPI application for LLM-RAG: Domain-Specific Question Answering System.

Wires configuration, the vector store and the Gemini generator together and
exposes the REST API consumed by the frontend. Domain logic lives in
:mod:`app.services`; this module is the HTTP layer.

Multimodal additions (v2.1):
- POST /api/upload   — indexes PDF, DOCX, and image text/content
- GET  /api/images/{image_id} — serves extracted image files securely
- DELETE /api/documents/{filename} — also removes associated image files
- POST /api/ask / POST /api/ask/stream — sources now carry content_type,
  table_markdown, image_url for frontend rendering
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .config import AppConfig, get_settings
from .schemas import (
    AskRequest,
    AskResponse,
    ConfidenceInfo,
    DeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    EvaluationInfo,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    RetrievalInfo,
    StatusResponse,
    UploadResponse,
    AnalyticsResponse,
)
from .services import (
    GeminiGenerator,
    build_prompt,
    build_sources_with_citation_map,
    calculate_file_hash,
    delete_document_images,
    document_already_indexed,
    get_all_document_records,
    get_collection,
    get_generator,
    looks_like_docx,
    get_images_dir,
    GeminiConfigurationError,
    GeminiTimeoutError,
    looks_like_image,
    looks_like_pdf,
    read_image_into_chunks,
    read_docx_into_chunks,
    read_pdf_into_chunks,
    retrieve_context,
    CT_TABLE,
    CT_IMAGE,
    CT_OCR,
    CT_TEXT,
)
from .memory import store as memory_store

# --- Advanced feature modules ---
from .hybrid_search import hybrid_search
from .guard import detect_hallucination, build_guarded_prompt
from .reranker import rewrite_query
from .evaluator import evaluate_answer_faithfulness, compute_answer_relevance
from .cache_monitor import response_cache, request_monitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("domainqa")

SYSTEM_NAME = "LLM-RAG: Domain-Specific Question Answering System"

# Load and validate local settings at import; Gemini is required for model-backed operations.
_settings = get_settings()

app = FastAPI(
    title=SYSTEM_NAME,
    version="2.1.0",
    description=(
        "A local Retrieval-Augmented Generation (RAG) API with "
        "hybrid search (vector + BM25), hallucination detection, query rewriting, "
        "answer faithfulness evaluation, response caching, and monitoring. "
        "Upload PDF and DOCX documents or supported images and receive grounded answers with source citations "
        "from text, tables, embedded images, and diagrams."
    ),
    openapi_tags=[
        {"name": "System", "description": "Health, status, cache stats, and monitoring metrics."},
        {"name": "Documents", "description": "Upload, list and delete indexed PDFs, DOCX files, and images."},
        {"name": "Q&A", "description": "Ask grounded questions with hybrid search and hallucination guard."},
        {"name": "Evaluation", "description": "RAG evaluation and faithfulness analysis."},
        {"name": "Feedback", "description": "Rate answer quality."},
        {"name": "Analytics", "description": "Usage analytics and system metrics."},
        {"name": "Media", "description": "Serve extracted image files."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_origin_regex=_settings.cors_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# MIDDLEWARE — request monitoring
# =========================================================================

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 2)
    request_monitor.record_request(
        endpoint=getattr(request.scope.get("route"), "path", "unmatched"),
        method=request.method,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    return response


@app.exception_handler(GeminiConfigurationError)
async def handle_gemini_configuration_error(
    _request: Request,
    exc: GeminiConfigurationError,
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(GeminiTimeoutError)
async def handle_gemini_timeout_error(
    _request: Request,
    exc: GeminiTimeoutError,
) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": str(exc)})


# =========================================================================
# SYSTEM
# =========================================================================

@app.get("/", tags=["System"])
def root() -> dict:
    return {"message": f"{SYSTEM_NAME} is running"}


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    return HealthResponse()


def _remove_unindexed_images(metadatas: list[dict], settings: AppConfig) -> None:
    images_dir = get_images_dir(settings)
    for metadata in metadatas:
        image_filename = metadata.get("image_filename")
        if not image_filename:
            continue
        if Path(image_filename).name != image_filename:
            logger.error("Refusing to remove unsafe extracted image path")
            continue
        image_path = images_dir / image_filename
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not remove unindexed image %s", image_filename)


@app.get("/api/status", response_model=StatusResponse, tags=["System"])
def status(
    collection=Depends(get_collection),
    settings: AppConfig = Depends(get_settings),
) -> StatusResponse:
    documents = get_all_document_records(collection)
    return StatusResponse(
        system=SYSTEM_NAME,
        documents=len(documents),
        chunks_indexed=collection.count(),
        total_queries=memory_store.analytics_snapshot()["total_queries"],
        llm_configured=bool(
            settings.llm.api_key and not settings.llm.api_key.startswith("test-")
        ),
    )


@app.get("/api/cache-stats", tags=["System"])
def cache_stats() -> dict:
    return response_cache.stats()


@app.get("/api/monitor", tags=["System"])
def monitor_metrics() -> dict:
    metrics = request_monitor.get_metrics()
    metrics["slow_requests"] = request_monitor.get_slow_requests()
    return metrics


# =========================================================================
# MEDIA — serve extracted image files securely
# =========================================================================

@app.get("/api/images/{image_filename}", tags=["Media"])
def serve_image(
    image_filename: str,
    settings: AppConfig = Depends(get_settings),
):
    """Serve an extracted image file from uploads/images/.

    Only allows alphanumeric characters, hyphens, underscores, and dots
    to prevent path traversal attacks.
    """
    # Security: reject any path traversal
    safe_name = Path(image_filename).name
    if safe_name != image_filename or ".." in image_filename or "/" in image_filename or "\\" in image_filename:
        raise HTTPException(status_code=400, detail="Invalid image filename")

    # Only allow known image extensions
    allowed_exts = {".png", ".jpg", ".jpeg", ".webp"}
    if Path(safe_name).suffix.lower() not in allowed_exts:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    images_dir = get_images_dir(settings)
    img_path = images_dir / safe_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(Path(safe_name).suffix.lower(), "image/png")
    return FileResponse(str(img_path), media_type=media_type)


# =========================================================================
# DOCUMENTS
# =========================================================================

@app.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents(collection=Depends(get_collection)) -> DocumentListResponse:
    documents = get_all_document_records(collection)
    return DocumentListResponse(count=len(documents), documents=documents)


@app.get(
    "/api/documents/{filename}",
    response_model=DocumentDetailResponse,
    tags=["Documents"],
)
def get_document(filename: str, collection=Depends(get_collection)) -> DocumentDetailResponse:
    for document in get_all_document_records(collection):
        if document["filename"] == filename:
            return DocumentDetailResponse(document=document)
    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/api/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(
    file: UploadFile = File(...),
    collection=Depends(get_collection),
    settings: AppConfig = Depends(get_settings),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    filename_lower = file.filename.lower()
    is_pdf = filename_lower.endswith(".pdf")
    is_docx = filename_lower.endswith(".docx")
    is_image = any(filename_lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp"))

    if not is_pdf and not is_docx and not is_image:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and image files (.png, .jpg, .jpeg, .webp) are allowed",
        )

    contents = await file.read(settings.max_upload_bytes + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit",
        )

    if is_pdf:
        if not looks_like_pdf(contents):
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF")
    elif is_docx:
        if not looks_like_docx(contents):
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid DOCX document")
    else:
        if not looks_like_image(contents):
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image")

    file_hash = calculate_file_hash(contents)
    safe_filename = Path(file.filename.replace("\\", "/")).name
    if safe_filename in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if document_already_indexed(collection, file_hash):
        return UploadResponse(
            success=False,
            duplicate=True,
            message=f"This {'PDF' if is_pdf else 'DOCX' if is_docx else 'image'} has already been indexed",
            filename=safe_filename,
        )

    settings.storage.upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.storage.upload_dir / safe_filename
    created_upload = False
    file_already_exists = file_path.exists()
    if file_already_exists:
        existing_hash = hashlib.sha256()
        try:
            if file_path.stat().st_size == len(contents):
                with file_path.open("rb") as existing_file:
                    for block in iter(lambda: existing_file.read(1024 * 1024), b""):
                        existing_hash.update(block)
        except OSError as exc:
            logger.exception("Could not inspect existing upload %s", safe_filename)
            raise HTTPException(
                status_code=500,
                detail="Could not verify the existing upload file.",
            ) from exc
        if existing_hash.hexdigest() != file_hash:
            raise HTTPException(
                status_code=409,
                detail="A different file with this name already exists; rename the upload and try again.",
            )
    else:
        try:
            with file_path.open("xb") as uploaded_file:
                created_upload = True
                uploaded_file.write(contents)
        except FileExistsError as exc:
            raise HTTPException(
                status_code=409,
                detail="A different file with this name already exists; rename the upload and try again.",
            ) from exc
        except OSError as exc:
            if created_upload and file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    logger.exception("Could not remove partial upload %s", safe_filename)
            logger.exception("Could not save uploaded file %s", safe_filename)
            raise HTTPException(
                status_code=500,
                detail="Could not save the uploaded file.",
            ) from exc

    try:
        metadatas: list[dict] = []
        if is_pdf:
            total_pages, total_characters, chunks, metadatas, ids = read_pdf_into_chunks(
                file_path, safe_filename, file_hash, settings
            )
        elif is_docx:
            total_pages, total_characters, chunks, metadatas, ids = read_docx_into_chunks(
                file_path, safe_filename, file_hash, settings
            )
        else:
            total_pages, total_characters, chunks, metadatas, ids = read_image_into_chunks(
                file_path, safe_filename, file_hash, settings
            )
    except GeminiTimeoutError as exc:
        logger.exception("Gemini request timed out while parsing %s", safe_filename)
        _remove_unindexed_images(metadatas, settings)
        if created_upload and file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=504,
            detail="Gemini request timed out while processing the document. Please retry.",
        ) from exc
    except Exception:
        logger.exception("Failed to parse file %s", safe_filename)
        _remove_unindexed_images(metadatas, settings)
        if created_upload and file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=422,
            detail="Could not read the document. It may be corrupted or password-protected.",
        )

    if not chunks:
        _remove_unindexed_images(metadatas, settings)
        if created_upload and file_path.exists():
            file_path.unlink()
        detail = (
            "No searchable text could be extracted from the image. "
            "Configure Gemini Vision or install Tesseract OCR for text-bearing images."
            if is_image
            else "No readable text or content found in the document."
        )
        raise HTTPException(status_code=422, detail=detail)

    try:
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    except Exception as exc:
        logger.exception("Failed to index %s", safe_filename)
        _remove_unindexed_images(metadatas, settings)
        try:
            collection.delete(ids=ids)
        except Exception:
            logger.exception("Could not roll back partially indexed chunks for %s", safe_filename)
        if created_upload and file_path.exists():
            file_path.unlink()
        if isinstance(exc, GeminiTimeoutError):
            raise HTTPException(
                status_code=504,
                detail="Gemini request timed out while indexing the document. Please retry.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="Failed to generate embeddings for the document. Please try again.",
        )

    response_cache.clear()

    # Count chunk types for response
    ct_counter = Counter(m.get("content_type", CT_TEXT) for m in metadatas)
    table_count = ct_counter.get(CT_TABLE, 0)
    image_count = ct_counter.get(CT_IMAGE, 0)

    logger.info(
        "Indexed %s (%s pages, %d text chunks, %d table chunks, %d image chunks)",
        safe_filename, total_pages,
        ct_counter.get(CT_TEXT, 0) + ct_counter.get(CT_OCR, 0),
        table_count, image_count,
    )

    return UploadResponse(
        success=True,
        duplicate=False,
        message=f"{'PDF' if is_pdf else 'DOCX' if is_docx else 'Image'} uploaded and indexed successfully",
        filename=safe_filename,
        pages=total_pages,
        characters=total_characters,
        chunks=len(chunks),
        tables=table_count,
        images=image_count,
        document_hash=file_hash[:16],
    )


@app.delete("/api/documents/{filename}", response_model=DeleteResponse, tags=["Documents"])
def delete_document(
    filename: str,
    collection=Depends(get_collection),
    settings: AppConfig = Depends(get_settings),
) -> DeleteResponse:
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    results = collection.get(where={"filename": filename}, include=["metadatas"])
    ids = results.get("ids", [])
    if not ids:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete extracted image files BEFORE removing from collection
    images_deleted = delete_document_images(filename, collection, settings)

    collection.delete(ids=ids)

    # Delete original uploaded file
    file_path = settings.storage.upload_dir / Path(filename).name
    file_deleted = False
    if file_path.exists():
        file_path.unlink()
        file_deleted = True

    # Invalidate cache
    response_cache.clear()

    logger.info("Deleted %s (%d chunks, %d images)", filename, len(ids), images_deleted)
    return DeleteResponse(
        message="Document deleted successfully",
        filename=filename,
        chunks_deleted=len(ids),
        file_deleted=file_deleted,
        images_deleted=images_deleted,
    )


@app.get("/api/documents/{filename}/chunks", tags=["Documents"])
def get_document_chunks(
    filename: str,
    collection=Depends(get_collection),
    settings: AppConfig = Depends(get_settings),
):
    """Return all chunks for a specific document with their metadata and content."""
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    results = collection.get(
        where={"filename": filename},
        include=["documents", "metadatas"]
    )

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    if not ids:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = []
    for doc_id, document, metadata in zip(ids, documents, metadatas):
        chunk_entry = {
            "id": doc_id,
            "content": document,
            "page": metadata.get("page"),
            "chunk_number": metadata.get("chunk", 0),
            "filename": metadata.get("filename", filename),
            "content_type": metadata.get("content_type", CT_TEXT),
            "section": metadata.get("section"),
        }
        # Attach table/image metadata for richer chunk viewer
        if metadata.get("content_type") == CT_TABLE:
            chunk_entry["table_number"] = metadata.get("table_number")
            chunk_entry["table_markdown"] = metadata.get("table_markdown", "")
        if metadata.get("content_type") == CT_IMAGE:
            chunk_entry["image_id"] = metadata.get("image_id", "")
            chunk_entry["image_filename"] = metadata.get("image_filename", "")
            chunk_entry["image_description"] = metadata.get("image_description", "")
        chunks.append(chunk_entry)

    # Sort by page then chunk number
    chunks.sort(
        key=lambda c: (
            c["page"] is None,
            c["page"] if c["page"] is not None else 0,
            c["chunk_number"],
        )
    )

    return {
        "filename": filename,
        "chunk_size": settings.rag.chunk_size,
        "chunk_overlap": settings.rag.chunk_overlap,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


# =========================================================================
# Q&A — helper: enrich sources with image_url
# =========================================================================

def _enrich_sources_with_image_urls(sources: list[dict], base_url: str) -> list[dict]:
    """Add image_url to image-type sources so the frontend can display them."""
    enriched = []
    for src in sources:
        if src.get("content_type") == CT_IMAGE and src.get("image_filename"):
            src = dict(src)
            img_fn = src["image_filename"]
            src["image_url"] = f"{base_url}/api/images/{img_fn}"
        enriched.append(src)
    return enriched


def _count_content_types(metadatas: list[dict]) -> dict[str, int]:
    counter = Counter(m.get("content_type", CT_TEXT) for m in metadatas if m)
    return dict(counter)


def _map_source_citations(answer: str, citation_map: dict[int, int]) -> str:
    """Map prompt citation numbers to the deduplicated source array."""
    citation_pattern = re.compile(r"\[SOURCE\s*(\d+)\]", re.IGNORECASE)
    return citation_pattern.sub(
        lambda match: (
            f"[SOURCE {citation_map[int(match.group(1))]}]"
            if int(match.group(1)) in citation_map
            else ""
        ),
        answer,
    )


def _iter_mapped_citation_tokens(
    tokens: Iterable[str],
    citation_map: dict[int, int],
) -> Iterator[str]:
    """Map complete citations while retaining a possible split marker between tokens."""
    pending = ""
    source_prefix = "[SOURCE"
    for token in tokens:
        pending += token
        hold_from = len(pending)
        open_bracket = pending.rfind("[")
        if open_bracket >= 0:
            suffix = pending[open_bracket:].upper()
            if source_prefix.startswith(suffix) or (
                suffix.startswith(source_prefix)
                and all(
                    character.isspace() or character.isdigit()
                    for character in suffix[len(source_prefix):]
                )
            ):
                hold_from = open_bracket

        ready, pending = pending[:hold_from], pending[hold_from:]
        mapped = _map_source_citations(ready, citation_map)
        if mapped:
            yield mapped

    if pending:
        mapped = _map_source_citations(pending, citation_map)
        if mapped:
            yield mapped


# =========================================================================
# Q&A — Main endpoint
# =========================================================================

@app.post("/api/ask", response_model=AskResponse, tags=["Q&A"])
def ask_question(
    request: Request,
    payload: AskRequest,
    collection=Depends(get_collection),
    generator: GeminiGenerator = Depends(get_generator),
    settings: AppConfig = Depends(get_settings),
) -> AskResponse:
    start = time.time()

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    if collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload a PDF first.",
        )

    # --- CACHE CHECK ---
    cache_key = response_cache.cache_key(
        question,
        payload.filter_filename,
        payload.model_dump(mode="json"),
    )
    cached_result = response_cache.get(cache_key)
    if cached_result:
        logger.info("Cache hit for: %s", question[:60])
        cached_result["cached"] = True
        cached_result["latency_ms"] = round((time.time() - start) * 1000, 1)
        memory_store.add_turn(
            session_id=payload.session_id or uuid.uuid4().hex,
            user=question,
            assistant=cached_result["answer"],
            latency_ms=cached_result["latency_ms"],
        )
        return AskResponse(**cached_result)

    # --- QUERY REWRITING ---
    rewritten_queries = None
    search_query = question
    if payload.use_query_rewriting:
        try:
            rewritten_queries = rewrite_query(question, generator)
            if rewritten_queries and len(rewritten_queries) > 1:
                search_query = " ".join(rewritten_queries)
                logger.info("Query rewritten: %s -> %s", question[:60], search_query[:60])
        except Exception:
            logger.warning("Query rewriting failed, using original query")

    # --- HYBRID SEARCH + METADATA FILTERING ---
    search_type = "hybrid" if payload.use_hybrid_search else "vector"
    if payload.use_hybrid_search:
        try:
            documents, metadatas, scores = hybrid_search(
                collection, search_query, top_k=settings.rag.top_k,
                filter_filename=payload.filter_filename,
                rrf_k=settings.rag.rrf_k,
            )
        except Exception as exc:
            logger.exception("Hybrid document retrieval failed")
            raise HTTPException(
                status_code=502,
                detail="Document retrieval failed. Please try again.",
            ) from exc
    else:
        try:
            documents, metadatas = retrieve_context(
                collection, search_query, settings.rag.top_k,
                filter_filename=payload.filter_filename,
            )
        except Exception as exc:
            logger.exception("Vector document retrieval failed")
            raise HTTPException(
                status_code=502,
                detail="Document retrieval failed. Please try again.",
            ) from exc
        scores = [0.0] * len(documents)

    if not documents:
        return AskResponse(
            question=question,
            answer="I could not find the answer in the uploaded documents.",
            sources=[],
            latency_ms=round((time.time() - start) * 1000, 1),
            retrieval=RetrievalInfo(
                search_type=search_type,
                query_rewrite=search_query if search_query != question else None,
                total_chunks_retrieved=0,
            ),
        )

    # --- HALLUCINATION GUARD ---
    history = [turn.model_dump() for turn in payload.history] if payload.history else None
    if payload.use_guard:
        prompt = build_guarded_prompt(question, documents, metadatas, history=history)
    else:
        prompt = build_prompt(question, documents, metadatas, history=history)

    try:
        answer = generator.generate(prompt)
    except GeminiTimeoutError as exc:
        logger.exception("Answer generation timed out")
        raise HTTPException(
            status_code=504,
            detail="Gemini request timed out. Please try again.",
        ) from exc
    except Exception:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=502,
            detail="The language model failed to generate an answer. Please try again.",
        )

    if not answer.strip():
        answer = "I could not generate an answer from the uploaded documents."
    sources, citation_map = build_sources_with_citation_map(metadatas)
    answer = _map_source_citations(answer, citation_map)

    # Enrich image sources with serving URLs
    base_url = str(request.base_url).rstrip("/")
    sources = _enrich_sources_with_image_urls(sources, base_url)

    followups = _generate_followups(question, answer)

    # --- CONFIDENCE SCORING ---
    confidence_info = None
    if payload.use_guard:
        guard_result = detect_hallucination(answer, documents, metadatas=metadatas)
        confidence_info = ConfidenceInfo(**guard_result)

    # --- FAITHFULNESS EVALUATION ---
    eval_info = None
    if payload.evaluate:
        faith = evaluate_answer_faithfulness(answer, sources, metadatas)
        relevance = compute_answer_relevance(question, answer, generator)
        eval_info = EvaluationInfo(
            faithfulness_score=faith["faithfulness_score"],
            unsupported_claims=faith["unsupported_claims"],
            coverage_score=faith["coverage_score"],
            relevance_score=relevance,
        )

    ct_counts = _count_content_types(metadatas)

    response_data = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "suggested_followups": followups,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "confidence": confidence_info.model_dump() if confidence_info else None,
        "retrieval": RetrievalInfo(
            search_type=search_type,
            query_rewrite=search_query if search_query != question else None,
            total_chunks_retrieved=len(documents),
            scores=[round(s, 4) for s in scores[:5]],
            content_types=ct_counts,
        ).model_dump(),
        "evaluation": eval_info.model_dump() if eval_info else None,
        "cached": False,
    }

    response_cache.set(cache_key, response_data)
    memory_store.add_turn(
        session_id=payload.session_id or uuid.uuid4().hex,
        user=question,
        assistant=answer,
        latency_ms=response_data["latency_ms"],
    )

    return AskResponse(**response_data)


# =========================================================================
# Q&A — Streaming endpoint
# =========================================================================

@app.post("/api/ask/stream", tags=["Q&A"])
async def ask_question_stream(
    request: Request,
    payload: AskRequest,
    collection=Depends(get_collection),
    generator: GeminiGenerator = Depends(get_generator),
    settings: AppConfig = Depends(get_settings),
):
    """Streaming SSE endpoint — tokens arrive in real-time."""
    start = time.time()

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    if collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet.",
        )

    # --- QUERY REWRITING ---
    search_query = question
    if payload.use_query_rewriting:
        try:
            rewritten_queries = rewrite_query(question, generator)
            if rewritten_queries and len(rewritten_queries) > 1:
                search_query = " ".join(rewritten_queries)
        except Exception as exc:
            logger.exception("Streaming query rewriting failed")
            raise HTTPException(
                status_code=502,
                detail="Query preparation failed. Please try again.",
            ) from exc

    # --- HYBRID SEARCH ---
    if payload.use_hybrid_search:
        try:
            documents, metadatas, _ = hybrid_search(
                collection, search_query, top_k=settings.rag.top_k,
                filter_filename=payload.filter_filename,
                rrf_k=settings.rag.rrf_k,
            )
        except Exception as exc:
            logger.exception("Streaming hybrid document retrieval failed")
            raise HTTPException(
                status_code=502,
                detail="Document retrieval failed. Please try again.",
            ) from exc
    else:
        try:
            documents, metadatas = retrieve_context(
                collection, search_query, settings.rag.top_k,
                filter_filename=payload.filter_filename,
            )
        except Exception as exc:
            logger.exception("Streaming vector document retrieval failed")
            raise HTTPException(
                status_code=502,
                detail="Document retrieval failed. Please try again.",
            ) from exc
    sources, citation_map = build_sources_with_citation_map(metadatas)

    # Enrich image sources with serving URLs
    base_url = str(request.base_url).rstrip("/")
    sources = _enrich_sources_with_image_urls(sources, base_url)

    search_type = "hybrid" if payload.use_hybrid_search else "vector"
    ct_counts = _count_content_types(metadatas)

    if not documents:
        async def empty():
            yield "event: done\ndata: " + json.dumps({
                "answer": "I could not find the answer in the uploaded documents.",
                "sources": [],
                "latency_ms": round((time.time() - start) * 1000, 1),
            }) + "\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    # --- PROMPT ---
    history = [turn.model_dump() for turn in payload.history] if payload.history else None
    if payload.use_guard:
        prompt = build_guarded_prompt(question, documents, metadatas, history=history)
    else:
        prompt = build_prompt(question, documents, metadatas, history=history)

    async def event_stream():
        full_answer = ""
        logger.info("Sending prompt to Gemini: %d characters", len(prompt))
        yield f"event: sources\ndata: {json.dumps({
            'sources': sources,
        })}\n\n"

        def tracked_tokens() -> Iterator[str]:
            nonlocal full_answer
            for token in generator.generate_stream(prompt):
                if token:
                    full_answer += token
                    yield token

        try:
            for token in _iter_mapped_citation_tokens(
                tracked_tokens(),
                citation_map,
            ):
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            logger.exception("Streaming failed")
            exc_str = str(exc).lower()
            if not full_answer and ("503" in exc_str or "unavailable" in exc_str):
                try:
                    logger.warning("Gemini streaming unavailable; trying non-streaming")
                    full_answer = generator.generate(prompt)
                except Exception:
                    logger.exception("Non-streaming fallback failed")

            if not full_answer:
                detail = "Generation failed"
                if isinstance(exc, GeminiTimeoutError):
                    detail = "Gemini request timed out. Please try again."
                elif "503" in exc_str or "unavailable" in exc_str:
                    detail = "Gemini is temporarily overloaded. Please try again shortly."
                elif "429" in exc_str or "resource_exhausted" in exc_str or "quota" in exc_str:
                    detail = "Gemini API quota exceeded. Please wait a minute and try again."
                elif "api_key" in exc_str or "authentication" in exc_str or "401" in exc_str:
                    detail = "Invalid API key. Please check your Gemini API configuration."
                yield f"event: error\ndata: {json.dumps({'detail': detail})}\n\n"
                return

        if not full_answer.strip():
            full_answer = "I could not generate an answer from the uploaded documents."
        full_answer = _map_source_citations(full_answer, citation_map)

        # Confidence
        confidence = None
        if payload.use_guard:
            guard_result = detect_hallucination(full_answer, documents, metadatas=metadatas)
            confidence = guard_result

        # Faithfulness eval
        evaluation = None
        if payload.evaluate:
            faith = evaluate_answer_faithfulness(full_answer, sources, metadatas)
            relevance = compute_answer_relevance(question, full_answer, generator)
            evaluation = {
                "faithfulness_score": faith["faithfulness_score"],
                "unsupported_claims": faith["unsupported_claims"],
                "coverage_score": faith["coverage_score"],
                "relevance_score": relevance,
            }

        memory_store.add_turn(
            session_id=payload.session_id or uuid.uuid4().hex,
            user=question,
            assistant=full_answer,
            latency_ms=round((time.time() - start) * 1000, 1),
        )

        yield f"event: done\ndata: {json.dumps({
            'answer': full_answer,
            'sources': sources,
            'latency_ms': round((time.time() - start) * 1000, 1),
            'confidence': confidence,
            'evaluation': evaluation,
            'retrieval': {
                'search_type': search_type,
                'query_rewrite': search_query if search_query != question else None,
                'total_chunks_retrieved': len(documents),
                'content_types': ct_counts,
            },
        })}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _generate_followups(question: str, answer: str) -> list[str]:
    """Generate 3 suggested follow-up questions using Gemini (best-effort)."""
    try:
        gen = get_generator()
        followup_prompt = (
            f"Based on this Q&A:\nQ: {question}\nA: {answer}\n\n"
            "Suggest exactly 3 short, specific follow-up questions (one per line, no numbering). "
            "Each should be answerable from the same document context. "
            "Return ONLY the 3 questions, one per line."
        )
        raw = gen.generate(followup_prompt)
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[:3]
    except Exception:
        logger.warning("Follow-up question generation failed", exc_info=True)
        return []


# =========================================================================
# FEEDBACK
# =========================================================================

@app.post("/api/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def record_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    memory_store.record_feedback(payload.rating, payload.question)
    return FeedbackResponse(message="Feedback recorded. Thank you!")


# =========================================================================
# ANALYTICS
# =========================================================================

@app.get("/api/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics(collection=Depends(get_collection)) -> AnalyticsResponse:
    docs = get_all_document_records(collection)
    popular = sorted(docs, key=lambda d: d.get("chunks", 0), reverse=True)[:5]
    snapshot = memory_store.analytics_snapshot()
    return AnalyticsResponse(
        total_queries=snapshot["total_queries"],
        thumbs_up=snapshot["thumbs_up"],
        thumbs_down=snapshot["thumbs_down"],
        avg_latency_ms=snapshot["avg_latency_ms"],
        popular_documents=[{"filename": d["filename"], "chunks": d["chunks"]} for d in popular],
        recent_queries=snapshot["recent_queries"],
    )


@app.post("/api/analytics/reset", tags=["Analytics"])
def reset_analytics() -> dict:
    memory_store.reset_analytics()
    return {
        "success": True,
        "message": "Analytics reset. Documents and conversation history were preserved.",
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001)
