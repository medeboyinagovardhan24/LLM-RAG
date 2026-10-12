"""Hallucination detection and answer confidence scoring."""
from __future__ import annotations

import re
from collections import Counter


def compute_confidence(answer: str, documents: list[str], metadatas: list[dict]) -> float:
    """Compute a 0-1 confidence score for the answer based on source grounding."""
    if not answer or not documents:
        return 0.0

    answer_lower = answer.lower()
    answer_words = set(re.findall(r"\w+", answer_lower, re.UNICODE))

    # --- Source overlap (40%) ---
    source_words = set()
    for doc in documents:
        source_words.update(re.findall(r"\w+", doc.lower(), re.UNICODE))
    overlap = len(answer_words & source_words) / max(len(answer_words), 1)
    source_score = min(overlap * 2, 1.0)  # amplify overlap

    # --- Answer length penalty (20%) ---
    word_count = len(answer.split())
    if word_count < 5:
        length_score = 0.3  # too short — might be a refusal
    elif word_count > 800:
        length_score = 0.5  # too long — might be rambling
    else:
        length_score = 1.0

    # --- Source diversity (20%) ---
    unique_sources = len(set(m.get("filename") for m in metadatas if m))
    diversity_score = min(unique_sources / 2, 1.0)

    # --- Citation density (20%) ---
    citation_pattern = re.compile(r"\[SOURCE\s*\d+\]|source\s*\d+", re.IGNORECASE)
    citations = len(citation_pattern.findall(answer))
    citation_score = min(citations / max(len(documents), 1), 1.0)

    confidence = (
        source_score * 0.40
        + length_score * 0.20
        + diversity_score * 0.20
        + citation_score * 0.20
    )
    return round(min(confidence, 1.0), 3)


def detect_hallucination(
    answer: str,
    source_texts: list[str],
    threshold: float = 0.3,
) -> dict:
    """Detect potential hallucination and return confidence info."""
    confidence = compute_confidence(answer, source_texts, [])

    is_reliable = confidence >= threshold
    warning = None

    if not is_reliable:
        if confidence < 0.15:
            warning = "Very low confidence. The answer may not be grounded in the provided documents."
        elif confidence < 0.3:
            warning = "Low confidence. Some parts of the answer may not be supported by the sources."

    # Check for common hallucination patterns
    hedging = re.compile(
        r"\b(I think|I believe|probably|generally|usually|typically|"
        r"it is known that|it is well known)\b",
        re.IGNORECASE,
    )
    if hedging.search(answer):
        warning = warning or "Answer contains hedging language that may indicate uncertainty."

    return {
        "is_reliable": is_reliable,
        "confidence": confidence,
        "warning": warning,
    }


def build_guarded_prompt(
    question: str,
    documents: list[str],
    metadatas: list[dict],
    history: list[dict] | None = None,
) -> str:
    """Build an enhanced RAG prompt with hallucination guardrails and inline citation instructions."""
    context_parts = []
    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        context_parts.append(
            f"[SOURCE {index}] File: {metadata.get('filename', 'Unknown')} "
            f"(Page {metadata.get('page', '?')}, Chunk {metadata.get('chunk', '?')})\n"
            f"{document}"
        )
    context = "\n\n".join(context_parts)

    history_text = ""
    if history:
        for turn in history[-3:]:
            history_text += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n\n"

    return f"""You are a helpful, grounded question answering system with source citations.

Rules:
1. Answer the user's question thoroughly using the sources provided below.
2. You MUST cite your sources inline using [SOURCE N] notation when referencing specific content.
3. If the answer is not in the sources, say EXACTLY: "The answer is not found in the provided documents."
4. Provide a clear, well-structured answer — do not just list citations.
5. Summarize and explain the information from the sources in a helpful way.
6. Table Support: When the sources contain tables or structured data, format the information clearly as structured Markdown tables.
7. Multilingual Support: Always respond in the same language as the user's question (e.g. if the user asks in Spanish, answer in Spanish; if in Hindi, answer in Hindi; if in German, answer in German), translating and synthesizing information from the retrieved context as needed.
8. Visuals & Photos: If the sources describe photos, diagrams, charts, or images, explain their visual content accurately.
9. If multiple sources agree, cite all relevant sources.
10. If sources conflict, note the conflict and cite both sides.

========================================
RETRIEVED CONTEXT
========================================

{context}

========================================
CONVERSATION HISTORY
========================================

{history_text}

========================================
USER QUESTION
========================================

{question}

========================================
INSTRUCTIONS
========================================

Provide a clear, comprehensive answer grounded in the sources above.
Use [SOURCE N] citations when quoting or referencing specific details.
If uncertain, say so explicitly. Do not invent information.

ANSWER:
"""
