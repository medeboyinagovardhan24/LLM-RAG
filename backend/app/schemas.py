"""Pydantic request/response models.

These give the API a typed contract and make the auto-generated OpenAPI docs
at ``/docs`` accurate and self-describing.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# --- Chat & History -------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


# --- Requests -------------------------------------------------------------

class AskRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Conversation session identifier.")
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question about the indexed documents.",
        examples=["What is Retrieval-Augmented Generation?"],
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Optional prior conversation turns for contextual multi-turn Q&A.",
    )
    filter_filename: str | None = Field(
        default=None,
        description="Optional filter to restrict retrieval to a single document.",
    )
    use_hybrid_search: bool = Field(
        default=True,
        description="Use hybrid search (vector + BM25) instead of pure vector search.",
    )
    use_query_rewriting: bool = Field(
        default=True,
        description="Rewrite query with LLM before retrieval for better results.",
    )
    use_guard: bool = Field(
        default=True,
        description="Enable hallucination detection and confidence scoring.",
    )
    evaluate: bool = Field(
        default=False,
        description="Include RAG evaluation metrics in the response.",
    )


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    comment: str | None = None


# --- Shared pieces --------------------------------------------------------

class Source(BaseModel):
    """A single retrieved evidence source with multimodal metadata."""
    filename: Optional[str] = None
    page: Optional[int] = None

    # Content type: text | table | image | ocr
    content_type: str = "text"

    # Text/OCR sources
    chunk: Optional[int] = None
    snippet: Optional[str] = None

    # Table sources
    table_number: Optional[int] = None
    table_markdown: Optional[str] = None
    table_headers: Optional[str] = None

    # Image/diagram sources
    image_id: Optional[str] = None
    image_filename: Optional[str] = None
    image_description: Optional[str] = None
    image_url: Optional[str] = None   # Set by main.py using the serve endpoint


class DocumentRecord(BaseModel):
    filename: str
    file_hash: str | None = None
    pages: int
    chunks: int
    tables: int = 0
    images: int = 0


# --- Responses ------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"


class StatusResponse(BaseModel):
    status: str = "online"
    system: str
    documents: int
    chunks_indexed: int
    total_queries: int = 0


class DocumentListResponse(BaseModel):
    success: bool = True
    count: int
    documents: list[DocumentRecord]


class DocumentDetailResponse(BaseModel):
    success: bool = True
    document: DocumentRecord


class UploadResponse(BaseModel):
    success: bool
    duplicate: bool = False
    message: str
    filename: str | None = None
    pages: int | None = None
    characters: int | None = None
    chunks: int | None = None
    tables: int | None = None
    images: int | None = None
    document_hash: str | None = None


class DeleteResponse(BaseModel):
    success: bool = True
    message: str
    filename: str
    chunks_deleted: int
    file_deleted: bool
    images_deleted: int = 0


class ConfidenceInfo(BaseModel):
    is_reliable: bool
    confidence: float = Field(ge=0, le=1)
    warning: str | None = None


class EvaluationInfo(BaseModel):
    faithfulness_score: float = Field(ge=0, le=1, default=0.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    coverage_score: float = Field(ge=0, le=1, default=0.0)
    relevance_score: float = Field(ge=0, le=5, default=0.0)


class RetrievalInfo(BaseModel):
    search_type: str = "hybrid"
    query_rewrite: str | None = None
    total_chunks_retrieved: int = 0
    scores: list[float] = Field(default_factory=list)
    content_types: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each content_type retrieved (text, table, image, ocr).",
    )


class AskResponse(BaseModel):
    success: bool = True
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    confidence: ConfidenceInfo | None = None
    retrieval: RetrievalInfo | None = None
    evaluation: EvaluationInfo | None = None
    cached: bool = False


class FeedbackResponse(BaseModel):
    success: bool = True
    message: str = "Feedback recorded successfully."


class AnalyticsResponse(BaseModel):
    total_queries: int
    thumbs_up: int
    thumbs_down: int
    avg_latency_ms: float
    popular_documents: list[dict]
    recent_queries: list[dict]
