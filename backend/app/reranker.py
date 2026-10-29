"""Query rewriting, expansion, and keyword extraction for enhanced retrieval."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def rewrite_query(question: str, generator, max_rewrites: int = 3) -> list[str]:
    """Use the LLM to generate alternative phrasings of the query for better retrieval.

    Returns a list of rewritten queries (including the original).
    """
    try:
        rewrite_prompt = (
            f"Given this user question, generate exactly {max_rewrites} alternative "
            f"phrasings that could help find relevant information in a document store.\n\n"
            f"Original question: {question}\n\n"
            "Return ONLY the alternative phrasings, one per line. "
            "Do not include the original question. Do not number them. "
            "Keep each phrasing concise (under 20 words)."
        )
        raw = generator.generate(rewrite_prompt)
        rewrites = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        # Clean up numbering if present
        cleaned = []
        for r in rewrites[:max_rewrites]:
            r = re.sub(r"^\d+[\.\)]\s*", "", r).strip()
            if r and r.lower() != question.lower():
                cleaned.append(r)
        return [question] + cleaned
    except Exception:
        logger.warning("Query rewriting failed; using the original question", exc_info=True)
        return [question]


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """Extract the most important keywords from text using simple heuristics."""
    # Common stop words to filter out
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "about", "also", "what", "which", "who",
        "whom", "this", "that", "these", "those", "how", "when", "where", "why",
        "if", "then", "else", "it", "its", "i", "me", "my", "we", "our",
        "you", "your", "he", "she", "they", "them", "his", "her", "their",
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    word_freq: dict[str, int] = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def multi_doc_analysis(
    question: str,
    documents: list[str],
    metadatas: list[dict],
    generator,
) -> dict:
    """Synthesize a cross-document answer using the LLM.

    For questions that require information from multiple documents, this
    function assembles all sources and asks the model to produce a structured
    response that identifies which documents contributed to each part of the
    answer.

    Parameters
    ----------
    question:
        The user question.
    documents:
        Retrieved document strings.
    metadatas:
        Corresponding metadata dicts.
    generator:
        A GeminiGenerator instance with a ``.generate`` method.

    Returns
    -------
    dict with keys:
        ``answer`` (str): the synthesized answer.
        ``document_contributions`` (list[dict]): per-document contribution
            with keys ``filename``, ``page``, ``contribution``.
        ``requires_multiple_docs`` (bool): whether the answer spans more
            than one source.
    """
    if not documents:
        return {
            "answer": "No documents were retrieved for this question.",
            "document_contributions": [],
            "requires_multiple_docs": False,
        }

    source_parts: list[str] = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source_parts.append(
            f"[SOURCE {idx}] File: {meta.get('filename', 'Unknown')} "
            f"(Page {meta.get('page', '?')}):\n{doc}"
        )
    all_sources = "\n\n".join(source_parts)

    analysis_prompt = (
        f"You are analyzing multiple documents to answer a question.\n\n"
        f"QUESTION: {question}\n\n"
        f"SOURCES:\n{all_sources}\n\n"
        f"TASK:\n"
        f"1. Read all sources carefully.\n"
        f"2. Synthesize a comprehensive answer.\n"
        f"3. For each source that contributed to the answer, describe its "
        f"contribution in one sentence.\n\n"
        f"Respond in the following EXACT format (no extra text):\n\n"
        f"ANSWER:\n"
        f"<your synthesized answer>\n\n"
        f"CONTRIBUTIONS:\n"
        f"SOURCE 1: <one sentence describing what SOURCE 1 contributed>\n"
        f"SOURCE 2: <one sentence describing what SOURCE 2 contributed>\n"
        f"...\n"
    )

    try:
        response = generator.generate(analysis_prompt)
    except Exception:
        fallback_answer = " ".join(doc[:200] for doc in documents[:3])
        return {
            "answer": f"Based on the retrieved documents: {fallback_answer}",
            "document_contributions": [
                {
                    "filename": meta.get("filename", "Unknown"),
                    "page": meta.get("page", 0),
                    "contribution": "Provided contextual information.",
                }
                for meta in metadatas
            ],
            "requires_multiple_docs": len(documents) > 1,
        }

    # Parse the structured response.
    answer_text = ""
    contributions: list[dict] = []

    answer_match = re.search(
        r"ANSWER:\s*\n(.*?)(?=\nCONTRIBUTIONS:|\Z)", response, re.DOTALL
    )
    if answer_match:
        answer_text = answer_match.group(1).strip()

    contrib_section = re.search(r"CONTRIBUTIONS:\s*\n(.*)", response, re.DOTALL)
    if contrib_section:
        contrib_lines = contrib_section.group(1).strip().split("\n")
        for line in contrib_lines:
            line = line.strip()
            if not line:
                continue
            source_match = re.match(
                r"SOURCE\s+(\d+):\s*(.*)", line, re.IGNORECASE
            )
            if source_match:
                src_idx = int(source_match.group(1)) - 1
                desc = source_match.group(2).strip()
                if 0 <= src_idx < len(metadatas):
                    contributions.append({
                        "filename": metadatas[src_idx].get("filename", "Unknown"),
                        "page": metadatas[src_idx].get("page", 0),
                        "contribution": desc,
                    })

    if not answer_text:
        answer_text = response.strip()

    unique_files = {meta.get("filename") for meta in metadatas}

    return {
        "answer": answer_text,
        "document_contributions": contributions,
        "requires_multiple_docs": len(unique_files) > 1,
    }
