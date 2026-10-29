"""RAG evaluation: retrieval quality, answer faithfulness, relevance, and coverage metrics."""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


def _tokenize_simple(text: str) -> set[str]:
    """Lowercase tokenisation producing a set of words."""
    return set(re.findall(r"[a-z]+", text.lower()))


def evaluate_ranked_retrieval(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> dict[str, float]:
    """Calculate identifier-based Precision@K, Recall@K and reciprocal rank."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    relevant = set(relevant_ids)
    if not relevant:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0}

    ranked = retrieved_ids[:k]
    relevant_hits = sum(identifier in relevant for identifier in ranked)
    first_relevant_rank = next(
        (rank for rank, identifier in enumerate(ranked, start=1) if identifier in relevant),
        None,
    )
    return {
        "precision_at_k": round(relevant_hits / k, 4),
        "recall_at_k": round(relevant_hits / len(relevant), 4),
        "mrr_at_k": round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
    }


def evaluate_retrieval(
    query: str,
    retrieved_docs: list[str],
    expected_keywords: list[str],
) -> dict:
    """Evaluate the quality of a retrieval step against expected keywords.

    Parameters
    ----------
    query:
        The user query (used for novelty computation).
    retrieved_docs:
        Document strings returned by the retriever.
    expected_keywords:
        Keywords or short phrases expected to appear in relevant documents.

    Returns
    -------
    dict with keys:
        ``precision`` (float): fraction of retrieved docs that contain at
            least one expected keyword.
        ``recall`` (float): fraction of expected keywords found across
            retrieved docs.
        ``novelty`` (float): 1 - (overlap between query and retrieved docs).
            Higher means the retriever found new content beyond the query.
        ``total_docs`` (int): number of retrieved documents.
        ``matched_docs`` (int): docs containing at least one keyword.
    """
    if not retrieved_docs:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "novelty": 0.0,
            "total_docs": 0,
            "matched_docs": 0,
        }

    query_tokens = _tokenize_simple(query)
    expected_lower = [kw.lower() for kw in expected_keywords]

    matched = 0
    all_retrieved_tokens: set[str] = set()

    for doc in retrieved_docs:
        doc_lower = doc.lower()
        all_retrieved_tokens.update(_tokenize_simple(doc))
        if any(kw in doc_lower for kw in expected_lower):
            matched += 1

    precision = matched / len(retrieved_docs) if retrieved_docs else 0.0

    if expected_keywords:
        found = sum(
            1 for kw in expected_lower
            if any(kw in doc.lower() for doc in retrieved_docs)
        )
        recall = found / len(expected_keywords)
    else:
        recall = 0.0

    if query_tokens and all_retrieved_tokens:
        overlap = query_tokens & all_retrieved_tokens
        novelty = 1.0 - (len(overlap) / len(all_retrieved_tokens))
    else:
        novelty = 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "novelty": round(max(0.0, novelty), 4),
        "total_docs": len(retrieved_docs),
        "matched_docs": matched,
    }


def evaluate_answer_faithfulness(
    answer: str,
    sources: list[dict],
    metadatas: list[dict],
) -> dict:
    """Evaluate how faithful the answer is to the retrieved sources.

    Returns a dict with:
      - faithfulness_score (0-1)
      - unsupported_claims (list of strings)
      - coverage_score (0-1)
    """
    if not answer or not sources:
        return {
            "faithfulness_score": 0.0,
            "unsupported_claims": [],
            "coverage_score": 0.0,
        }

    # Extract source snippets for comparison
    source_texts = []
    for s in sources:
        snippet = s.get("snippet", "")
        if snippet:
            source_texts.append(snippet.lower())

    all_source_text = " ".join(source_texts)

    # Split answer into sentences as "claims"
    sentences = re.split(r"[.!?]+", answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    unsupported = []
    supported_count = 0

    for sentence in sentences:
        # Check if key content words from the sentence appear in source text
        content_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", sentence.lower()))
        content_words -= {
            "this", "that", "with", "from", "have", "been", "were", "they",
            "their", "there", "about", "which", "would", "could", "should",
            "more", "also", "than", "into", "only", "very", "some", "when",
        }
        if not content_words:
            supported_count += 1
            continue

        words_in_source = sum(1 for w in content_words if w in all_source_text)
        support_ratio = words_in_source / len(content_words)

        if support_ratio >= 0.4:
            supported_count += 1
        else:
            unsupported.append(sentence.strip())

    faithfulness = supported_count / max(len(sentences), 1)

    # Coverage: how many source documents are actually referenced in the answer
    answer_lower = answer.lower()
    referenced = 0
    for s in sources:
        filename = s.get("filename", "")
        if filename and filename.lower() in answer_lower:
            referenced += 1
        elif s.get("page") and f"page {s['page']}" in answer_lower:
            referenced += 1

    # Also check SOURCE N citations
    citation_matches = re.findall(r"\[SOURCE\s*(\d+)\]", answer, re.IGNORECASE)
    referenced = max(referenced, len(set(citation_matches)))

    coverage = min(referenced / max(len(sources), 1), 1.0)

    return {
        "faithfulness_score": round(faithfulness, 3),
        "unsupported_claims": unsupported[:5],  # cap at 5
        "coverage_score": round(coverage, 3),
    }


def compute_answer_relevance(
    question: str,
    answer: str,
    generator,
    scale: int = 5,
) -> float:
    """Use the LLM to rate answer relevance on a 1-{scale} scale.

    Returns a float score.
    """
    if not answer or not question:
        return 0.0

    try:
        eval_prompt = (
            f"Rate how relevant this answer is to the question on a scale of 1 to {scale}.\n"
            f"1 = completely irrelevant, {scale} = perfectly relevant and complete.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer[:1000]}\n\n"
            f"Respond with ONLY a number between 1 and {scale}."
        )
        raw = generator.generate(eval_prompt)
        raw = raw.strip()
        # Extract number
        match = re.search(r"(\d+(?:\.\d+)?)", raw)
        if match:
            score = float(match.group(1))
            return round(min(max(score, 1.0), float(scale)), 2)
    except Exception:
        pass

    return 3.0  # default neutral score


def batch_evaluate(
    qa_pairs: list[dict],
    generator,
) -> dict:
    """Evaluate a batch of question-answer pairs.

    Each item in qa_pairs should have 'question', 'answer', and 'sources'.
    Returns aggregate metrics.
    """
    results = []
    for pair in qa_pairs:
        faith = evaluate_answer_faithfulness(
            pair["answer"],
            pair.get("sources", []),
            pair.get("metadatas", []),
        )
        relevance = compute_answer_relevance(
            pair["question"],
            pair["answer"],
            generator,
        )
        results.append({**faith, "relevance_score": relevance})

    if not results:
        return {
            "count": 0,
            "avg_faithfulness": 0.0,
            "avg_coverage": 0.0,
            "avg_relevance": 0.0,
        }

    return {
        "count": len(results),
        "avg_faithfulness": round(
            sum(r["faithfulness_score"] for r in results) / len(results), 3
        ),
        "avg_coverage": round(
            sum(r["coverage_score"] for r in results) / len(results), 3
        ),
        "avg_relevance": round(
            sum(r["relevance_score"] for r in results) / len(results), 2
        ),
    }
