
"""Hybrid search combining vector similarity with BM25 keyword scoring.

Blends ChromaDB vector retrieval with BM25 keyword retrieval using a
configurable weighting parameter (alpha). Includes optional reranking that
boosts documents containing exact keyword matches.

Multimodal additions:
- Table chunks receive a score bonus for table-related queries.
- Image/diagram chunks receive a bonus for image-related queries.
- Exact keyword matches can improve relevance through reranking.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Content-type-aware boosting keywords

_TABLE_KEYWORDS: set[str] = {
    "table", "tables", "row", "rows", "column", "columns", "record", "records",
    "data", "student", "students", "teacher", "teachers", "department",
    "departments", "marks", "grade", "grades", "list", "entries", "entry",
    "cell", "header", "value", "values", "field", "fields", "spreadsheet",
    "tabular",
}

_IMAGE_KEYWORDS: set[str] = {
    "image", "diagram", "figure", "chart", "graph", "photo", "picture",
    "flowchart", "architecture", "illustration", "schematic", "screenshot",
    "plot", "visualization", "visual", "drawing", "sketch", "component",
    "components", "arrows", "legend", "caption", "axes",
}

_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need", "must",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "what", "which", "who", "whom", "where", "when", "why",
    "how", "not", "no", "nor", "so", "too", "very", "just", "about",
}


def reciprocal_rank_fusion(
    ranked_results: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Fuse ranked stable chunk identifiers using reciprocal rank fusion.

    Identifiers missing from a result list receive no contribution. Duplicate
    identifiers within one list are counted only at their first rank.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if weights is not None and len(weights) != len(ranked_results):
        raise ValueError("weights must match ranked_results")

    fused: dict[str, float] = {}
    for result_index, result in enumerate(ranked_results):
        weight = weights[result_index] if weights is not None else 1.0
        if weight < 0:
            raise ValueError("weights must be non-negative")
        seen: set[str] = set()
        for rank, identifier in enumerate(result, start=1):
            if identifier in seen:
                continue
            seen.add(identifier)
            fused[identifier] = fused.get(identifier, 0.0) + weight / (k + rank)
    return fused


def _tokenize(text: str) -> list[str]:
    """Lowercase text, remove punctuation and filter common stopwords."""
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    return [token for token in tokens if token not in _STOPWORDS]


def hybrid_search(
    collection,
    query: str,
    top_k: int = 10,
    alpha: float = 0.7,
    filter_filename: Optional[str] = None,
    rrf_k: int = 60,
) -> tuple[list[str], list[dict], list[float]]:
    """Combine ChromaDB vector retrieval with BM25 keyword retrieval."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    if top_k <= 0:
        return [], [], []
    if rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")

    query_lower = query.lower().strip()
    if not query_lower:
        return [], [], []

    where_filter: dict | None = None
    if filter_filename:
        where_filter = {"filename": filter_filename}

    # FIX: ChromaDB Collection.count() does not accept `where`
    # in the installed version. Use collection.get() for filtered counts.
    if where_filter:
        matching_records = collection.get(
            where=where_filter,
            include=["metadatas"],
        )
        collection_count = len(matching_records.get("ids", []))
    else:
        collection_count = collection.count()

    if collection_count == 0:
        return [], [], []

    # Vector search
    vector_ids: list[str] = []
    vec_documents: list[str] = []
    vec_metadatas: list[dict] = []

    if alpha > 0.0:
        n_results = min(top_k * 2, collection_count)

        vector_results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        vector_ids = vector_results.get("ids", [[]])[0]
        vec_documents = vector_results.get("documents", [[]])[0]
        vec_metadatas = vector_results.get("metadatas", [[]])[0]

    candidates: dict[str, tuple[str, dict]] = {
        doc_id: (doc, meta)
        for doc_id, doc, meta in zip(
            vector_ids, vec_documents, vec_metadatas
        )
    }

    # BM25 keyword search
    bm25_ranks: dict[str, int] = {}

    query_tokens = _tokenize(query_lower)

    if alpha < 1.0 and query_tokens:
        get_kwargs: dict = {
            "include": ["documents", "metadatas"]
        }

        if where_filter:
            get_kwargs["where"] = where_filter

        all_data = collection.get(**get_kwargs)

        corpus_ids = all_data.get("ids", [])
        corpus_docs = all_data.get("documents", [])
        corpus_metas = all_data.get("metadatas", [])

        if corpus_docs:
            tokenized_corpus = [
                _tokenize(doc or "")
                for doc in corpus_docs
            ]

            bm25 = BM25Okapi(tokenized_corpus)
            raw_bm25 = bm25.get_scores(query_tokens).tolist()

            ranked_indices = sorted(
                (
                    idx
                    for idx, score in enumerate(raw_bm25)
                    if score > 0 or set(query_tokens).intersection(tokenized_corpus[idx])
                ),
                key=lambda idx: (-raw_bm25[idx], idx),
            )

            for rank, idx in enumerate(ranked_indices, start=1):
                doc_id = corpus_ids[idx]
                bm25_ranks[doc_id] = rank

                if doc_id not in candidates:
                    candidates[doc_id] = (
                        corpus_docs[idx],
                        corpus_metas[idx],
                    )

    # Content-type-aware boosting
    query_tokens_set = set(query_tokens)

    wants_table = bool(query_tokens_set & _TABLE_KEYWORDS)
    wants_image = bool(query_tokens_set & _IMAGE_KEYWORDS)

    # Combine vector and BM25 ranks. Alpha remains a compatibility weighting
    # control; the underlying rank fusion is reciprocal-rank based.
    fusion_weights = (alpha, 1.0 - alpha)
    fused_scores = reciprocal_rank_fusion(
        (vector_ids, tuple(bm25_ranks)),
        k=rrf_k,
        weights=fusion_weights,
    )
    content_type_boost = 0.5 / (rrf_k + 1)
    combined: list[tuple[str, dict, float]] = []

    for doc_id, (doc, meta) in candidates.items():
        combined_score = fused_scores.get(doc_id, 0.0)

        content_type = meta.get("content_type", "text")

        if alpha > 0.0 and content_type == "table" and wants_table:
            combined_score += content_type_boost
        elif (
            alpha > 0.0
            and
            content_type in ("image", "ocr")
            and wants_image
        ):
            combined_score += content_type_boost

        combined.append((doc, meta, combined_score))

    combined.sort(key=lambda item: item[2], reverse=True)
    top = combined[:top_k]

    documents = [item[0] for item in top]
    metadatas = [item[1] for item in top]
    scores = [round(item[2], 6) for item in top]

    return documents, metadatas, scores


def rerank_by_relevance(
    query: str,
    documents: list[str],
    metadatas: list[dict],
    scores: list[float],
) -> tuple[list[str], list[dict], list[float]]:
    """Rerank retrieved chunks by boosting exact query keyword matches."""

    if not documents:
        return [], [], []

    keywords = _tokenize(query)

    if not keywords:
        return documents, metadatas, scores

    boosted: list[tuple[str, dict, float]] = []

    for doc, meta, score in zip(documents, metadatas, scores):
        doc_lower = doc.lower()

        hits = sum(
            1 for keyword in keywords
            if keyword in doc_lower
        )

        bonus = hits / len(keywords)
        boosted.append((doc, meta, score + bonus))

    boosted.sort(key=lambda item: item[2], reverse=True)

    return (
        [item[0] for item in boosted],
        [item[1] for item in boosted],
        [round(item[2], 6) for item in boosted],
    )
