"""Hybrid search combining vector similarity with BM25 keyword scoring.

Blends ChromaDB vector retrieval with rank-bm25 keyword retrieval using a
configurable weighting parameter (alpha). Includes optional reranking that
boosts documents containing exact keyword matches.

Multimodal additions:
- Table chunks receive a score bonus when the query contains table-related
  keywords (table, column, row, record, data, etc.).
- Image/diagram chunks receive a bonus for image/diagram-related queries.
- Exact numeric and name matches in table content receive additional boosts.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# --- Content-type aware boosting keywords -----------------------------------

_TABLE_KEYWORDS: set[str] = {
    "table", "tables", "row", "rows", "column", "columns", "record", "records",
    "data", "student", "students", "teacher", "teachers", "department", "departments",
    "marks", "grade", "grades", "list", "entries", "entry", "cell", "header",
    "value", "values", "field", "fields", "spreadsheet", "tabular",
}

_IMAGE_KEYWORDS: set[str] = {
    "image", "diagram", "figure", "chart", "graph", "photo", "picture",
    "flowchart", "architecture", "illustration", "schematic", "screenshot",
    "plot", "visualization", "visual", "drawing", "sketch", "component",
    "components", "arrows", "legend", "caption", "axes",
}

# Common English stopwords used during tokenisation.
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


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation and remove stopwords (multilingual unicode-safe)."""
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    return [t for t in tokens if t not in _STOPWORDS]


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores into [0, 1]."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def hybrid_search(
    collection,
    query: str,
    top_k: int = 10,
    alpha: float = 0.7,
    filter_filename: Optional[str] = None,
) -> tuple[list[str], list[dict], list[float]]:
    """Combine vector similarity and BM25 keyword scores.

    Parameters
    ----------
    collection:
        ChromaDB collection object.
    query:
        User query string.
    top_k:
        Maximum number of results to return.
    alpha:
        Weight for the vector similarity score (0-1).  ``1 - alpha`` is
        assigned to the BM25 score.  A value of 1.0 disables BM25.
    filter_filename:
        Optional filename to restrict retrieval to a single document.

    Returns
    -------
    tuple[list[str], list[dict], list[float]]
        ``(documents, metadatas, scores)`` sorted by combined score descending.
    """
    collection_count = collection.count()
    if collection_count == 0:
        return [], [], []

    query_lower = query.lower().strip()
    if not query_lower:
        return [], [], []

    # ----- Vector search via ChromaDB ------------------------------------
    n_results = min(top_k, collection_count)
    where_filter: dict | None = None
    if filter_filename:
        where_filter = {"filename": filter_filename}

    try:
        vector_results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.exception("Vector search failed; falling back to empty results")
        return [], [], []

    vec_documents = vector_results.get("documents", [[]])[0]
    vec_metadatas = vector_results.get("metadatas", [[]])[0]
    vec_distances = vector_results.get("distances", [[]])[0]

    if not vec_documents:
        return [], [], []

    # ChromaDB distances are L2 by default -- convert to similarity [0, 1].
    vec_similarities = _normalize_scores([-d for d in vec_distances])

    # Build a lookup keyed by (filename, page, chunk) for later merging.
    doc_map: dict[tuple, int] = {}
    all_documents: list[str] = list(vec_documents)
    all_metadatas: list[dict] = list(vec_metadatas)
    vec_score_map: dict[tuple, float] = {}

    for idx, (doc, meta) in enumerate(zip(vec_documents, vec_metadatas)):
        key = (
            meta.get("filename", ""),
            meta.get("page", ""),
            meta.get("chunk", ""),
        )
        doc_map[key] = idx
        vec_score_map[key] = vec_similarities[idx]

    # ----- BM25 keyword search ------------------------------------------
    bm25_scores_map: dict[tuple, float] = {}

    if alpha < 1.0:
        try:
            # Fetch a smaller set for BM25 ranking to improve speed.
            fetch_limit = min(max(top_k * 2, 5), collection_count)
            get_kwargs: dict = {"include": ["documents", "metadatas"]}
            if where_filter:
                get_kwargs["where"] = where_filter
            get_kwargs["limit"] = fetch_limit

            all_data = collection.get(**get_kwargs)
            corpus_docs = all_data.get("documents", [])
            corpus_metas = all_data.get("metadatas", [])

            if corpus_docs:
                tokenized_corpus = [_tokenize(doc) for doc in corpus_docs]
                bm25 = BM25Okapi(tokenized_corpus)
                query_tokens = _tokenize(query)
                raw_bm25 = bm25.get_scores(query_tokens).tolist()
                bm25_normalized = _normalize_scores(raw_bm25)

                for idx_b, meta in enumerate(corpus_metas):
                    if idx_b >= len(bm25_normalized):
                        break
                    key = (
                        meta.get("filename", ""),
                        meta.get("page", ""),
                        meta.get("chunk", ""),
                    )
                    bm25_scores_map[key] = bm25_normalized[idx_b]

                    # Extend result set with docs found by BM25 but not vector search.
                    if key not in doc_map:
                        doc_map[key] = len(all_documents)
                        all_documents.append(corpus_docs[idx_b])
                        all_metadatas.append(meta)
                        vec_score_map[key] = 0.0
        except Exception:
            logger.exception("BM25 search failed; continuing with vector-only scores")

    # ----- Content-type boosting ----------------------------------------
    # Detect whether the query is asking about tables or images/diagrams.
    query_tokens_set = set(_tokenize(query_lower))
    wants_table = bool(query_tokens_set & _TABLE_KEYWORDS)
    wants_image = bool(query_tokens_set & _IMAGE_KEYWORDS)

    TABLE_BOOST = 0.25  # bonus for table chunks when query targets tables
    IMAGE_BOOST = 0.25  # bonus for image chunks when query targets images

    # ----- Combine scores ------------------------------------------------
    combined: list[tuple[str, dict, float]] = []
    seen_keys: set[tuple] = set()

    for idx, (doc, meta) in enumerate(zip(all_documents, all_metadatas)):
        key = (
            meta.get("filename", ""),
            meta.get("page", ""),
            meta.get("chunk", ""),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)

        vec_score = vec_score_map.get(key, 0.0)
        bm25_score = bm25_scores_map.get(key, 0.0)
        combined_score = alpha * vec_score + (1.0 - alpha) * bm25_score

        # Apply content-type boost
        ct = meta.get("content_type", "text")
        if ct == "table" and wants_table:
            combined_score += TABLE_BOOST
        elif ct in ("image", "ocr") and wants_image:
            combined_score += IMAGE_BOOST

        combined.append((doc, meta, combined_score))

    # Sort descending by combined score and take top_k.
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
    """Rerank results by boosting chunks that contain exact query keywords.

    This is a lightweight heuristic pass: for every keyword in the query that
    appears verbatim in a document, its score receives a small additive bonus.

    Parameters
    ----------
    query:
        Original user query.
    documents:
        Document strings to rerank.
    metadatas:
        Corresponding metadata dicts.
    scores:
        Corresponding relevance scores.

    Returns
    -------
    tuple[list[str], list[dict], list[float]]
        Reranked copies of the inputs, sorted by boosted score descending.
    """
    if not documents:
        return [], [], []

    keywords = _tokenize(query)
    if not keywords:
        return documents, metadatas, scores

    boosted: list[tuple[str, dict, float]] = []
    for doc, meta, score in zip(documents, metadatas, scores):
        doc_lower = doc.lower()
        hits = sum(1 for kw in keywords if kw in doc_lower)
        bonus = hits / len(keywords)  # up to +1.0 boost
        boosted.append((doc, meta, score + bonus))

    boosted.sort(key=lambda item: item[2], reverse=True)

    return (
        [item[0] for item in boosted],
        [item[1] for item in boosted],
        [round(item[2], 6) for item in boosted],
    )
