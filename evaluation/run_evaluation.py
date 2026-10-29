
"""Research-style evaluation runner for LLM-RAG.

Supports five configurations:
1. Gemini without retrieval
2. ChromaDB semantic search + Gemini
3. BM25 keyword search + Gemini
4. Hybrid retrieval + Gemini
5. Hybrid retrieval + reranking + Gemini

Produces per-run CSV, JSON, and Markdown reports.
Mock mode uses synthetic fixtures and is only a pipeline diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import email.utils
import json
import logging
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.genai.errors import APIError

backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import get_settings
from app.evaluator import evaluate_ranked_retrieval
from app.hybrid_search import hybrid_search, rerank_by_relevance
from app.services import (
    build_prompt,
    get_all_document_records,
    get_collection,
    get_generator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAMES = {
    "gemini_no_retrieval": "Gemini without retrieval",
    "chromadb_semantic": "ChromaDB Semantic Search + Gemini",
    "bm25_keyword": "BM25 Keyword Search + Gemini",
    "hybrid_retrieval": "Hybrid Retrieval + Gemini",
    "hybrid_rerank": "Hybrid Retrieval + Reranking + Gemini",
}

CONFIG_KEYS = list(CONFIG_NAMES)
DEFAULT_RETRY_COUNT = 2
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_RETRY_MAX_DELAY = 60.0
DEFAULT_REQUEST_INTERVAL = 1.0


def load_evaluation_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSON or JSONL evaluation dataset.

    The public schema uses ``question_id``, ``relevant_document``,
    ``relevant_page``, and ``relevant_chunk_ids``. Existing fixture aliases
    are normalized so older datasets remain runnable.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("questions", [])

    if not isinstance(records, list) or not records:
        raise ValueError("The evaluation dataset must contain a non-empty list of questions.")

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {index} must be an object.")
        question = str(record.get("question", "")).strip()
        if not question:
            raise ValueError(f"Dataset record {index} is missing question.")
        answerable = record.get(
            "answerable",
            not bool(record.get("is_unanswerable", False)),
        )
        if not isinstance(answerable, bool):
            raise ValueError(f"Dataset record {index} answerable must be boolean.")
        relevant_document = record.get(
            "relevant_document", record.get("ground_truth_filename")
        )
        chunk_ids = record.get(
            "relevant_chunk_ids", record.get("ground_truth_chunk_ids", [])
        )
        if chunk_ids is None:
            chunk_ids = []
        if not isinstance(chunk_ids, list) or not all(
            isinstance(chunk_id, (str, int)) for chunk_id in chunk_ids
        ):
            raise ValueError(
                f"Dataset record {index} relevant_chunk_ids must be a list of identifiers."
            )
        normalized.append({
            **record,
            "id": record.get("id", record.get("question_id", index)),
            "question": question,
            "expected_answer": record.get("expected_answer", ""),
            "ground_truth_filename": relevant_document,
            "ground_truth_page": record.get("relevant_page", record.get("ground_truth_page")),
            "ground_truth_chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
            "is_unanswerable": not answerable,
        })
    return normalized


class GeminiQuotaError(RuntimeError):
    """A Gemini quota/rate-limit failure that should stop further evaluation."""

    def __init__(
        self,
        message: str,
        *,
        daily: bool,
        hard_quota: bool,
        attempts: int,
        retry_after: float | None,
    ) -> None:
        super().__init__(message)
        self.daily = daily
        self.hard_quota = hard_quota
        self.attempts = attempts
        self.retry_after = retry_after


def _safe_error_detail(value: Any, limit: int = 500) -> str:
    """Bound error details and redact common API-key/authorization formats."""
    text = str(value)
    text = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", text)
    text = re.sub(
        r"(?i)([\"']?(?:authorization|api[_ -]?key|gemini_api_key|key)"
        r"[\"']?\s*[:=]\s*[\"']?)(?:bearer\s+)?[^\s,;\"'}]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer [REDACTED]",
        text,
    )
    return text[:limit]


def _exception_text(exc: BaseException) -> str:
    values = []
    current: BaseException | None = exc
    while current is not None:
        values.extend((
            getattr(current, "status", ""),
            getattr(current, "message", ""),
            getattr(current, "details", ""),
            str(current),
        ))
        current = current.__cause__ or current.__context__
    return " ".join(str(value) for value in values if value).lower()


def _is_quota_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        response = getattr(current, "response", None)
        code = (
            getattr(current, "code", None)
            or getattr(current, "status_code", None)
            or getattr(response, "status_code", None)
        )
        status = str(getattr(current, "status", "") or "").upper()
        text = " ".join((
            str(getattr(current, "message", "") or ""),
            str(current),
        )).lower()
        if (
            code == 429
            or str(code) == "429"
            or status == "RESOURCE_EXHAUSTED"
            or (
                isinstance(current, APIError)
                and ("resource_exhausted" in text or "too many requests" in text)
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_daily_quota_error(exc: BaseException) -> bool:
    text = _exception_text(exc)
    return bool(
        re.search(r"\bper[_ -]?day\b|\bdaily\b|\bday quota\b", text)
    )


def _is_hard_quota_error(exc: BaseException) -> bool:
    text = _exception_text(exc)
    if re.search(
        r"\bper[_ -]?(minute|second|hour)\b|requests_per_(minute|second|hour)",
        text,
    ):
        return False
    return bool(
        re.search(r"quota.{0,40}(exhausted|exceeded|limit)", text)
        or re.search(r"\blimit\s*[:=]\s*0\b", text)
    )


def _quota_exception(exc: BaseException) -> BaseException | None:
    current: BaseException | None = exc
    while current is not None:
        if _is_quota_error(current):
            return current
        current = current.__cause__ or current.__context__
    return None


def _has_api_failure(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        response = getattr(current, "response", None)
        if (
            isinstance(current, (APIError, TimeoutError))
            or getattr(current, "status_code", None) is not None
            or getattr(response, "status_code", None) is not None
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _safe_exception_detail(exc: BaseException, limit: int = 500) -> str:
    status_parts = [
        str(getattr(exc, "code", "") or ""),
        str(getattr(exc, "status", "") or ""),
        str(getattr(exc, "message", "") or ""),
    ]
    detail = " ".join(part for part in status_parts if part)
    if not detail:
        detail = str(exc)
    return _safe_error_detail(detail, limit=limit)


def _parse_retry_delay(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        return max(0.0, parsed) if math.isfinite(parsed) else None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s?\s*", value, re.I)
        if match:
            parsed = float(match.group(1))
            return parsed if math.isfinite(parsed) else None
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    current: BaseException | None = exc
    while current is not None:
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = next(
            (value for key, value in headers.items() if key.lower() == "retry-after"),
            None,
        )
        if retry_after is not None:
            delay = _parse_retry_delay(retry_after)
            if delay is not None:
                return delay
            try:
                retry_at = email.utils.parsedate_to_datetime(retry_after)
                return max(
                    0.0,
                    (retry_at - datetime.now(retry_at.tzinfo)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                pass

        pending = [getattr(current, "details", None)]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                for key, nested in value.items():
                    if str(key).lower().replace("_", "") in {
                        "retrydelay", "retryafter",
                    }:
                        delay = _parse_retry_delay(nested)
                        if delay is not None:
                            return delay
                    pending.append(nested)
            elif isinstance(value, list):
                pending.extend(value)
        current = current.__cause__ or current.__context__
    return None


class RequestController:
    """Pace Gemini evaluation calls and apply bounded retries for HTTP 429."""

    def __init__(
        self,
        generator: Any,
        *,
        enabled: bool,
        max_retries: int,
        retry_base_delay: float,
        retry_max_delay: float,
        request_interval: float,
    ) -> None:
        self.generator = generator
        self.enabled = enabled
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.request_interval = request_interval
        self.api_requests = 0
        self.api_failures = 0
        self.quota_responses = 0
        self.retries = 0
        self.vector_query_slots = 0
        self.judge_cache: dict[str, tuple[float | None, str]] = {}
        self.judge_cache_hits = 0
        self._last_request_at: float | None = None

    def _pace(self) -> None:
        if not self.enabled:
            return
        if self._last_request_at is not None:
            wait = self.request_interval - (
                time.monotonic() - self._last_request_at
            )
            if wait > 0:
                time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _retry_delay(self, exc: BaseException, attempt: int, operation: str) -> float:
        daily = _is_daily_quota_error(exc)
        hard_quota = _is_hard_quota_error(exc)
        retry_after = _retry_after_seconds(exc)
        safe_detail = _safe_exception_detail(exc)
        if daily or hard_quota:
            raise GeminiQuotaError(
                f"{'Daily ' if daily else ''}Gemini quota exhausted: {safe_detail}",
                daily=daily,
                hard_quota=hard_quota,
                attempts=attempt + 1,
                retry_after=retry_after,
            ) from exc

        if attempt >= self.max_retries:
            raise GeminiQuotaError(
                f"Gemini rate limit persisted after {attempt + 1} attempt(s): "
                f"{safe_detail}",
                daily=False,
                hard_quota=False,
                attempts=attempt + 1,
                retry_after=retry_after,
            ) from exc

        if retry_after is not None:
            delay = retry_after
            if delay > self.retry_max_delay:
                raise GeminiQuotaError(
                    "Gemini retry delay exceeds the configured retry-delay "
                    f"budget ({self.retry_max_delay:g}s); stopping: {safe_detail}",
                    daily=False,
                    hard_quota=False,
                    attempts=attempt + 1,
                    retry_after=retry_after,
                ) from exc
        else:
            delay = min(
                self.retry_base_delay * (2 ** attempt)
                + random.uniform(0.0, self.retry_base_delay),
                self.retry_max_delay,
            )
        self.retries += 1
        logger.warning(
            "Gemini rate limit during %s; retry %d/%d in %.2f seconds",
            operation,
            attempt + 1,
            self.max_retries,
            delay,
        )
        return delay

    def vector_query(self, operation: Any) -> Any:
        """Run a Chroma vector query with bounded 429 handling."""
        if not self.enabled:
            return operation()
        for attempt in range(self.max_retries + 1):
            self._pace()
            self.vector_query_slots += 1
            try:
                return operation()
            except Exception as exc:
                if _has_api_failure(exc):
                    self.api_failures += 1
                if not _is_quota_error(exc):
                    raise
                self.quota_responses += 1
                time.sleep(self._retry_delay(exc, attempt, "embedding query"))
        raise AssertionError("Vector-query retry loop exited unexpectedly")

    def cached_judge(
        self,
        cache_key: str,
        operation: Any,
    ) -> tuple[tuple[float | None, str], bool]:
        if cache_key in self.judge_cache:
            self.judge_cache_hits += 1
            return self.judge_cache[cache_key], True
        result = operation()
        if result[0] is not None:
            self.judge_cache[cache_key] = result
        return result, False

    def generate(self, prompt: str) -> str:
        if not self.enabled:
            return self.generator.generate(prompt)

        for attempt in range(self.max_retries + 1):
            self._pace()
            self.api_requests += 1
            try:
                return self.generator.generate(prompt)
            except Exception as exc:
                self.api_failures += 1
                if not _is_quota_error(exc):
                    raise
                self.quota_responses += 1
                time.sleep(self._retry_delay(exc, attempt, "generation/judging"))

        raise AssertionError("Request retry loop exited unexpectedly")


def _extract_source_blocks(prompt: str) -> list[tuple[int, str]]:
    """Extract [SOURCE N] blocks from a RAG prompt."""
    matches = list(re.finditer(r"\[SOURCE\s*(\d+)\]", prompt, re.I))
    blocks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        blocks.append((int(match.group(1)), prompt[match.start():end].lower()))
    return blocks


class MockEvaluatorGenerator:
    """Synthetic generator used only to verify offline pipeline execution."""

    def __init__(self, collection: Any | None = None) -> None:
        self.collection = collection

    def generate(self, prompt: str) -> str:
        p = prompt.lower()

        if (
            "evaluate the answer against the reference answer" in p
            or "evaluate the generated answer" in p
        ):
            fields = re.search(
                r"Question:\s*(.*?)\n"
                r"Expected Answer:\s*(.*?)\n"
                r"Generated Answer:\s*(.*?)\n\n"
                r"Return only JSON:",
                prompt,
                re.IGNORECASE | re.DOTALL,
            )
            if fields is None:
                return json.dumps({
                    "score": None,
                    "reason": "Mock judge prompt was malformed.",
                })
            question, expected, generated = fields.groups()
            score = compute_deterministic_relevance(
                question.strip(), expected.strip(), generated.strip()
            )
            return json.dumps({
                "score": score,
                "reason": (
                    "Synthetic lexical-overlap diagnostic; not a real "
                    "semantic quality assessment."
                ),
            })

        if "answer concisely based on your general knowledge" in p:
            if "france" in p:
                return "The capital of France is Paris and its currency is the Euro."
            if "royal mart" in p:
                return "Royal Mart is a supermarket in Bengaluru."
            if "govardhan" in p:
                return "Medeboyina Govardhan is an engineering student."
            return "I do not have access to those specific document records."

        blocks = _extract_source_blocks(prompt)

        def source_for(*terms: str) -> int | None:
            for number, block in blocks:
                if any(term in block for term in terms):
                    return number
            return None

        if "where is the visited royal mart outlet located" in p:
            src = source_for("kothihosahalli", "kodigehalli", "maruthi nagar")
            if src:
                return (
                    "The visited Royal Mart outlet is located at No. 90, Aura, "
                    "Kothihosahalli Extension, Maruthi Nagar, Kodigehalli, "
                    f"Bengaluru, Karnataka 560092 [SOURCE {src}]."
                )

        if "direct export activities" in p:
            src = source_for("direct export", "domestic retail", "retail point")
            if src:
                return (
                    "The evidence describes a domestic retail outlet and does "
                    f"not establish direct export activities [SOURCE {src}]."
                )

        if "products and customer services" in p or "products/services" in p:
            src = source_for("groceries", "vegetables", "doorstep delivery", "cereals")
            if src:
                return (
                    "Products include groceries, vegetables, cereals, packaged "
                    f"food and household items [SOURCE {src}]."
                )

        if "operational activities table" in p or "handle purchasing" in p:
            src = source_for("purchasing", "suppliers", "distributors")
            if src:
                return (
                    "Products are sourced through suppliers, distributors, and "
                    f"other supply channels [SOURCE {src}]."
                )

        if "cold chain failure" in p or "logistics challenges table" in p:
            src = source_for("cold chain", "perishables", "temperature")
            if src:
                return (
                    "Cold-chain failure can lead to quality loss for perishables "
                    f"when temperature is not maintained [SOURCE {src}]."
                )

        if "govardhan" in p and (
            "specialization" in p or "academic" in p or "email" in p
        ):
            src = source_for("artificial intelligence", "machine learning")
            if src:
                return (
                    "Medeboyina Govardhan is an Artificial Intelligence and "
                    f"Machine Learning engineering student [SOURCE {src}]."
                )

        return "The answer is not found in the provided documents."


class OfflineFixtureCollection:
    """Small synthetic corpus for offline pipeline checks."""

    def __init__(self, questions: list[dict]):
        self.records = []
        for q in questions:
            filename = q.get("ground_truth_filename")
            if not filename or q.get("is_unanswerable", False):
                continue

            page = q.get("ground_truth_page") or 1
            metadata = {
                "filename": filename,
                "page": page,
                "chunk": q["id"],
                "content_type": (
                    "table" if q.get("category") == "table_retrieval" else "text"
                ),
                "fixture": True,
            }
            content = q["expected_answer"]
            rid = hashlib.sha256(
                f"{filename}:{page}:{q['id']}:{content}".encode()
            ).hexdigest()[:16]
            self.records.append((rid, content, metadata))

        self.fixture_filenames = sorted({
            meta["filename"] for _, _, meta in self.records
        })

    def _filtered(self, where=None):
        if not where:
            return list(self.records)
        return [
            row for row in self.records
            if all(row[2].get(k) == v for k, v in where.items())
        ]

    def count(self, where=None):
        return len(self._filtered(where))

    def get(self, include=None, where=None):
        rows = self._filtered(where)
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [r[2] for r in rows],
        }

    def query(self, query_texts, n_results, where=None, include=None):
        terms = set(re.findall(r"\w+", query_texts[0].lower()))
        rows = self._filtered(where)
        rows.sort(
            key=lambda r: (
                -len(terms & set(re.findall(r"\w+", r[1].lower()))),
                r[0],
            )
        )
        rows = rows[:n_results]
        overlap = [
            len(terms & set(re.findall(r"\w+", r[1].lower())))
            for r in rows
        ]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[r[2] for r in rows]],
            "distances": [[1.0 / (1 + n) for n in overlap]],
        }


def compute_deterministic_relevance(
    question: str, expected: str, generated: str
) -> float:
    """Lexical overlap diagnostic; not a semantic quality metric."""
    if not generated:
        return 0.0

    expected_tokens = set(re.findall(r"\w+", expected.lower()))
    generated_tokens = set(re.findall(r"\w+", generated.lower()))

    stopwords = {
        "the", "a", "an", "is", "are", "and", "or", "in", "on", "of",
        "to", "for", "at", "by", "with",
    }
    expected_tokens -= stopwords
    generated_tokens -= stopwords

    if not expected_tokens:
        return 1.0 if not generated_tokens else 0.5

    intersection = expected_tokens & generated_tokens
    recall = len(intersection) / len(expected_tokens)
    precision = len(intersection) / len(generated_tokens) if generated_tokens else 0.0

    if recall + precision == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(max(0.0, min(1.0, 0.7 * recall + 0.3 * f1)), 3)


def evaluate_answer_relevance(
    question: str,
    expected: str,
    generated: str | None,
    generator: Any,
    is_mock: bool = False,
    request_controller: RequestController | None = None,
) -> tuple[float | None, str]:
    """Score an answer; preserve judge failures instead of inventing scores."""
    if not generated:
        return None, "Generation failed; relevance was not evaluated."

    prompt = f"""Evaluate the answer against the reference answer.
Score from 0.0 to 1.0:
0.0 = incorrect, irrelevant, or contradictory.
0.5 = partially correct or incomplete.
1.0 = correct and relevant.

If the expected answer says the information is not found in the documents,
award 1.0 when the generated answer correctly says the evidence is unavailable.

Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}

Return only JSON:
{{"score": 0.0, "reason": "Brief explanation"}}
"""

    try:
        response = (
            request_controller.generate(prompt)
            if request_controller is not None
            else generator.generate(prompt)
        )
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Judge returned an empty response")

        start = response.find("{")
        end = response.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("Judge returned an invalid response without JSON")

        result = json.loads(response[start:end])
        value = result.get("score")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Judge score must be numeric")

        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError("Judge score must be between 0 and 1")

        reason = result.get("reason", "Evaluated by LLM judge.")
        if is_mock:
            reason = f"Synthetic diagnostic only: {reason}"
        return score, _safe_error_detail(reason, limit=300)

    except GeminiQuotaError:
        raise
    except Exception as exc:
        detail = _safe_error_detail(
            _safe_exception_detail(exc, limit=300), limit=300
        )
        logger.warning(
            "Relevance judge failed (%s): %s", type(exc).__name__, detail
        )
        return None, f"Judge error ({type(exc).__name__}): {detail}"


def evaluate_citation_accuracy(
    answer: str | None,
    retrieved_metas: list[dict],
    ground_truth_filename: str | None,
    ground_truth_page: int | None,
    is_unanswerable: bool = False,
) -> float | None:
    """Check source index attribution to expected filename/page."""
    if answer is None or is_unanswerable or not ground_truth_filename:
        return None

    citations = [
        int(n) for n in re.findall(r"\[SOURCE\s*(\d+)\]", answer, re.I)
    ]
    if not citations:
        return 0.0

    valid = 0
    for number in citations:
        index = number - 1
        if 0 <= index < len(retrieved_metas):
            meta = retrieved_metas[index]
            same_file = (
                str(meta.get("filename", "")).lower()
                == ground_truth_filename.lower()
            )
            same_page = (
                ground_truth_page is None
                or meta.get("page") == ground_truth_page
            )
            if same_file and same_page:
                valid += 1

    return round(valid / len(citations), 3)


def run_configuration_evaluation(
    config_key: str,
    q: dict,
    collection: Any,
    generator: Any,
    top_k: int = 5,
    is_mock: bool = False,
    request_controller: RequestController | None = None,
) -> dict[str, Any]:
    """Run retrieval, generation, and scoring for one configuration/question."""
    start_time = time.perf_counter()
    request_counts_before = (
        (
            request_controller.api_requests,
            request_controller.api_failures,
            request_controller.retries,
            request_controller.vector_query_slots,
            request_controller.quota_responses,
        )
        if request_controller is not None
        else (0, 0, 0)
    )
    retrieval_error = None
    generation_error = None
    quota_failure = None
    retrieval_api_failure_count = 0
    retrieved_docs: list[str] = []
    retrieved_metas: list[dict] = []
    retrieved_scores: list[float] = []
    score_type = None

    # Retrieval phase
    try:
        if config_key == "gemini_no_retrieval":
            pass

        elif config_key == "chromadb_semantic":
            count = collection.count()
            if count > 0:
                query_operation = lambda: collection.query(
                    query_texts=[q["question"]],
                    n_results=min(top_k, count),
                    include=["documents", "metadatas", "distances"],
                )
                result = (
                    request_controller.vector_query(query_operation)
                    if request_controller is not None
                    else query_operation()
                )
                retrieved_docs = result.get("documents", [[]])[0]
                retrieved_metas = result.get("metadatas", [[]])[0]
                retrieved_scores = result.get("distances", [[]])[0]
            score_type = "chroma_distance"

        elif config_key == "bm25_keyword":
            retrieved_docs, retrieved_metas, retrieved_scores = hybrid_search(
                collection, q["question"], top_k=top_k, alpha=0.0
            )
            score_type = "bm25_rank_fusion_score"

        elif config_key == "hybrid_retrieval":
            search_operation = lambda: hybrid_search(
                collection, q["question"], top_k=top_k, alpha=0.7,
                rrf_k=getattr(getattr(get_settings(), "rag", None), "rrf_k", 60),
            )
            retrieved_docs, retrieved_metas, retrieved_scores = (
                request_controller.vector_query(search_operation)
                if request_controller is not None
                else search_operation()
            )
            score_type = "hybrid_rank_fusion_score"

        elif config_key == "hybrid_rerank":
            search_operation = lambda: hybrid_search(
                collection, q["question"], top_k=top_k, alpha=0.7,
                rrf_k=getattr(getattr(get_settings(), "rag", None), "rrf_k", 60),
            )
            docs, metas, scores = (
                request_controller.vector_query(search_operation)
                if request_controller is not None
                else search_operation()
            )
            retrieved_docs, retrieved_metas, retrieved_scores = rerank_by_relevance(
                q["question"], docs, metas, scores
            )
            score_type = "hybrid_rerank_score"

        else:
            raise ValueError(f"Unknown configuration: {config_key}")

    except Exception as exc:
        retrieval_api_failure_count = int(
            request_controller is None and _has_api_failure(exc)
        )
        retrieval_error = f"{type(exc).__name__}: {_safe_exception_detail(exc, 300)}"
        quota_exc = _quota_exception(exc)
        if quota_exc is not None:
            daily = _is_daily_quota_error(quota_exc)
            hard_quota = _is_hard_quota_error(quota_exc)
            quota_failure = {
                "kind": (
                    "daily_quota_exhausted" if daily
                    else "quota_exhausted" if hard_quota
                    else "rate_limit_exhausted"
                ),
                "operation": "retrieval_embedding",
                "attempts": getattr(quota_exc, "attempts", 1),
                "retry_after_seconds": _retry_after_seconds(quota_exc),
                "message": _safe_exception_detail(quota_exc),
            }
        logger.warning(
            "Retrieval failed for %s, question %s: %s",
            config_key, q.get("id"), retrieval_error,
        )

    # Generation phase
    generated_answer = None
    judge_status = "skipped_generation_failed"
    if config_key == "gemini_no_retrieval":
        prompt = (
            "Answer concisely based on your general knowledge:\n"
            f"Question: {q['question']}\nAnswer:"
        )
    else:
        prompt = build_prompt(q["question"], retrieved_docs, retrieved_metas)

    if retrieval_error:
        generation_error = "Skipped because retrieval failed."
    else:
        try:
            generated_answer = (
                request_controller.generate(prompt)
                if request_controller is not None
                else generator.generate(prompt)
            )
            if not isinstance(generated_answer, str) or not generated_answer.strip():
                generated_answer = None
                generation_error = "generation_error: EmptyResponse"
        except GeminiQuotaError as exc:
            generation_error = _safe_error_detail(exc)
            quota_failure = {
                "kind": (
                    "daily_quota_exhausted" if exc.daily
                    else "quota_exhausted" if exc.hard_quota
                    else "rate_limit_exhausted"
                ),
                "operation": "generation",
                "attempts": exc.attempts,
                "retry_after_seconds": exc.retry_after,
                "message": _safe_error_detail(exc),
            }
        except Exception as exc:
            detail = _safe_exception_detail(exc, limit=300)
            generation_error = f"{type(exc).__name__}: {detail}"
            logger.warning(
                "Generation failed for %s, question %s (%s): %s",
                config_key, q.get("id"), type(exc).__name__, detail,
            )

    latency = None if is_mock else round(time.perf_counter() - start_time, 3)

    # Relevance scoring
    relevance_score = None
    relevance_reason = "Not evaluated."
    relevance_error = None
    mock_diagnostic_relevance = None
    judge_cache_hit = False

    if is_mock:
        if generated_answer is not None:
            relevance_score, relevance_reason = evaluate_answer_relevance(
                q["question"],
                q["expected_answer"],
                generated_answer,
                generator,
                is_mock=True,
            )
            mock_diagnostic_relevance = relevance_score
            if relevance_score is None:
                relevance_error = relevance_reason
                judge_status = "failed"
            else:
                judge_status = "synthetic_diagnostic"
        else:
            relevance_reason = "Synthetic diagnostic skipped because generation failed."
    elif generated_answer is not None:
        try:
            def judge_answer() -> tuple[float | None, str]:
                return evaluate_answer_relevance(
                    q["question"],
                    q["expected_answer"],
                    generated_answer,
                    generator,
                    request_controller=request_controller,
                )

            if request_controller is not None:
                cache_key = hashlib.sha256(
                    json.dumps(
                        [q["question"], q["expected_answer"], generated_answer],
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                (relevance_score, relevance_reason), judge_cache_hit = (
                    request_controller.cached_judge(cache_key, judge_answer)
                )
            else:
                relevance_score, relevance_reason = judge_answer()
            if relevance_score is None:
                relevance_error = relevance_reason
                judge_status = "failed"
            else:
                judge_status = "scored"
        except GeminiQuotaError as exc:
            relevance_error = _safe_error_detail(exc)
            judge_status = "failed"
            quota_failure = {
                "kind": (
                    "daily_quota_exhausted" if exc.daily
                    else "quota_exhausted" if exc.hard_quota
                    else "rate_limit_exhausted"
                ),
                "operation": "relevance_judge",
                "attempts": exc.attempts,
                "retry_after_seconds": exc.retry_after,
                "message": _safe_error_detail(exc),
            }
    else:
        relevance_reason = "Generation failed; relevance was not evaluated."

    api_requests_used = (
        request_controller.api_requests - request_counts_before[0]
        if request_controller is not None else 0
    )
    api_failures_used = (
        request_controller.api_failures - request_counts_before[1]
        if request_controller is not None else 0
    )
    api_failures_used += retrieval_api_failure_count
    retries_used = (
        request_controller.retries - request_counts_before[2]
        if request_controller is not None else 0
    )
    vector_queries_used = (
        request_controller.vector_query_slots - request_counts_before[3]
        if request_controller is not None else 0
    )
    quota_responses_used = (
        request_controller.quota_responses - request_counts_before[4]
        if request_controller is not None else int(quota_failure is not None)
    )

    gt_file = q.get("ground_truth_filename")
    retrieved_filenames = [
        meta.get("filename")
        for meta in retrieved_metas
        if meta.get("filename")
    ]
    unique_filenames = list(dict.fromkeys(retrieved_filenames))

    retrieval_metrics = {
        "recall_at_1": None,
        "recall_at_3": None,
        "recall_at_5": None,
        "precision_at_5": None,
        "mrr_at_5": None,
    }

    if (
        config_key != "gemini_no_retrieval"
        and gt_file
        and not is_mock
        and not q.get("is_unanswerable", False)
        and not retrieval_error
    ):
        metrics = evaluate_ranked_retrieval(unique_filenames, [gt_file], k=5)
        retrieval_metrics["precision_at_5"] = metrics["precision_at_k"]
        retrieval_metrics["mrr_at_5"] = metrics["mrr_at_k"]
        for k in (1, 3, 5):
            retrieval_metrics[f"recall_at_{k}"] = (
                1.0 if gt_file in unique_filenames[:k] else 0.0
            )

    citation_accuracy = None
    if (
        not is_mock
        and config_key != "gemini_no_retrieval"
        and generated_answer is not None
        and not q.get("is_unanswerable", False)
    ):
        citation_accuracy = evaluate_citation_accuracy(
            generated_answer,
            retrieved_metas,
            gt_file,
            q.get("ground_truth_page"),
        )

    retrieved_chunks = []
    for rank, (document, metadata) in enumerate(
        zip(retrieved_docs, retrieved_metas), start=1
    ):
        identity = {
            key: metadata.get(key)
            for key in (
                "filename", "file_hash", "page", "chunk", "content_type",
                "table_number", "image_id",
            )
            if metadata.get(key) is not None
        }
        chunk_id = hashlib.sha256(
            (json.dumps(identity, sort_keys=True) + "\0" + document).encode()
        ).hexdigest()[:16]
        retrieved_chunks.append({
            "chunk_id": chunk_id,
            "rank": rank,
            "metadata": identity,
            "score": (
                retrieved_scores[rank - 1]
                if rank <= len(retrieved_scores) else None
            ),
            "score_type": score_type,
        })

    generation_status = (
        "succeeded"
        if generated_answer is not None and generation_error is None
        else "skipped"
        if retrieval_error is not None
        else "failed"
    )
    evaluation_status = (
        "failed"
        if generation_status == "failed"
        or retrieval_error is not None
        or relevance_error is not None
        else "succeeded"
    )

    return {
        "question_id": q["id"],
        "category": q.get("category", "general"),
        "config_key": config_key,
        "config_name": CONFIG_NAMES[config_key],
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "generated_answer": generated_answer,
        "success": generated_answer is not None and generation_error is None,
        "evaluation_status": evaluation_status,
        "generation_status": generation_status,
        "judge_status": judge_status,
        "judge_cache_hit": judge_cache_hit,
        "retrieval_error": retrieval_error,
        "generation_error": generation_error,
        "relevance_score": relevance_score,
        "relevance_reason": relevance_reason,
        "relevance_error": relevance_error,
        "quota_failure": json.dumps(quota_failure) if quota_failure else None,
        "api_request_count": api_requests_used,
        "api_failure_count": api_failures_used,
        "retry_count": retries_used,
        "vector_query_count": vector_queries_used,
        "quota_response_count": quota_responses_used,
        "skipped_reason": None,
        "mock_diagnostic_relevance": mock_diagnostic_relevance,
        "faithfulness_score": None,
        "faithfulness_note": "Not measured; claim-level verification is not implemented.",
        "citation_accuracy": citation_accuracy,
        "latency_sec": latency,
        "retrieved_sources": json.dumps(retrieved_filenames),
        "retrieved_chunks": json.dumps(retrieved_chunks),
        **retrieval_metrics,
    }


def _build_results_matrix(report: dict[str, Any]) -> str:
    configs = report["configurations"]

    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"

    rows = []
    for config in configs.values():
        rows.append(
            f"| {config['name']} | {fmt(config['mean_relevance'])} | "
            f"{fmt(config['mean_faithfulness'])} | {fmt(config['recall_at_1'])} | "
            f"{fmt(config['recall_at_3'])} | {fmt(config['recall_at_5'])} | "
            f"{fmt(config['precision_at_5'])} | {fmt(config['mrr_at_5'])} | "
            f"{fmt(config['mean_citation_accuracy'])} | "
            f"{fmt(config['mean_latency_sec'])} | "
            f"{config['generation_successes']}/{config['generation_attempts']} "
            f"(failed {config['failed_evaluations']}, "
            f"skipped {config['skipped_evaluations']}) |"
        )

    mock = report["execution_mode"] == "MOCK/OFFLINE"
    relevance_heading = (
        "Synthetic lexical relevance diagnostic"
        if mock
        else "Gemini relevance"
    )
    return f"""# LLM-RAG Evaluation Results

- **Mode:** {report['execution_mode']}
- **Timestamp:** {report['evaluation_timestamp']}
- **Run ID:** {report['run_id']}
- **Generation model:** {report['generation_model'] or 'N/A (mock)'}
- **Embedding model:** {report['embedding_model'] or 'N/A'}
- **Top-k:** {report['top_k']}
- **Dataset SHA-256:** `{report['dataset_sha256']}`
- **Corpus type:** {report['corpus_kind']}

| Configuration | {relevance_heading} | Faithfulness | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR@5 | Citation accuracy | Mean latency (s) | Generation success / attempted (failed, skipped) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Limitations

- {"Mock scores are synthetic lexical diagnostics, not real answer-quality measurements." if mock else "Relevance scores are from the configured Gemini judge; failed judge calls remain unscored."}
- Generation success, evaluation failure, and quota-triggered skipped configurations are counted separately in `evaluation_report.json`.
- Retry/pacing does not guarantee quota availability. Daily quota failures stop immediately; unresolved temporary 429 failures stop the remaining run to avoid extra requests.
- The printed request estimate covers generation and judge calls, not possible embedding calls initiated by ChromaDB's persisted embedding function.
- Retrieval metrics are filename-level, not chunk-level relevance judgments.
- Faithfulness is not measured because claim-level evidence verification is not implemented.
- Citation accuracy checks source index attribution, not whether every claim is supported.
- Latency measures retrieval and generation, excluding relevance judging.
"""


def _print_summary(report: dict[str, Any], run_dir: Path) -> None:
    print("\n" + "=" * 76)
    print("LLM-RAG RESEARCH EVALUATION SUMMARY")
    print("=" * 76)
    print(
        f"Mode: {report['execution_mode']} | "
        f"Questions: {report['total_test_questions']}"
    )
    print("-" * 76)
    relevance_heading = (
        "Synthetic"
        if report["execution_mode"] == "MOCK/OFFLINE"
        else "Relevance"
    )
    print(
        f"{'Configuration':<43} | {relevance_heading:<10} | {'Generation':<18}"
    )
    print("-" * 76)

    for config in report["configurations"].values():
        relevance = config["mean_relevance"]
        relevance_text = f"{relevance:.3f}" if relevance is not None else "N/A"
        generation_text = (
            f"{config['generation_successes']}/"
            f"{config['generation_attempts']} "
            f"(failed {config['generation_failures']}, "
            f"skip {config['generation_skipped']})"
        )
        print(
            f"{config['name']:<43} | {relevance_text:<10} | "
            f"{generation_text:<18}"
        )
    totals = report["run_counts"]
    print(
        "Run totals: "
        f"attempted={totals['attempted_evaluations']}, "
        f"failed={totals['failed_evaluations']}, "
        f"skipped={totals['skipped_evaluations']}, "
        f"terminal_quota_failures={totals['quota_failures']}, "
        f"quota_rate_limit_responses={totals['quota_responses']}, "
        f"runner_API_calls={totals['runner_api_requests']}, "
        f"API failures={totals['api_failures']}"
    )

    print("=" * 76)
    print(f"Artifacts saved to: {run_dir}")
    print(" - results.csv")
    print(" - evaluation_report.json")
    print(" - results_matrix.md")


def _validate_prerequisites(
    questions: list[dict],
    api_key: str,
    indexed_filenames: set[str] | None,
    is_mock: bool = False,
) -> None:
    if is_mock:
        return
    if (
        not api_key
        or api_key.startswith("test-")
        or api_key == "your-gemini-api-key-here"
    ):
        raise RuntimeError(
            "GEMINI_API_KEY is required for live evaluation. "
            "Use --mock for offline pipeline testing."
        )
    if indexed_filenames is None:
        return
    expected_files = {
        question["ground_truth_filename"]
        for question in questions
        if question.get("ground_truth_filename")
    }
    missing = sorted(expected_files - indexed_filenames)
    if missing:
        raise RuntimeError(
            "Index the required ground-truth documents before evaluation: "
            + ", ".join(missing)
        )


def _skipped_result(q: dict, config_key: str, reason: str) -> dict[str, Any]:
    return {
        "question_id": q["id"],
        "category": q.get("category", "general"),
        "config_key": config_key,
        "config_name": CONFIG_NAMES[config_key],
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "generated_answer": None,
        "success": False,
        "evaluation_status": "skipped",
        "generation_status": "skipped",
        "judge_status": "skipped",
        "judge_cache_hit": False,
        "retrieval_error": None,
        "generation_error": None,
        "relevance_score": None,
        "relevance_reason": None,
        "relevance_error": None,
        "quota_failure": None,
        "api_request_count": 0,
        "api_failure_count": 0,
        "retry_count": 0,
        "vector_query_count": 0,
        "quota_response_count": 0,
        "skipped_reason": reason,
        "mock_diagnostic_relevance": None,
        "faithfulness_score": None,
        "faithfulness_note": "Not measured; claim-level verification is not implemented.",
        "citation_accuracy": None,
        "latency_sec": None,
        "retrieved_sources": "[]",
        "retrieved_chunks": "[]",
        "recall_at_1": None,
        "recall_at_3": None,
        "recall_at_5": None,
        "precision_at_5": None,
        "mrr_at_5": None,
    }


def _quota_kind(row: dict[str, Any]) -> str | None:
    failure = row.get("quota_failure")
    if not failure:
        return None
    return json.loads(failure)["kind"]


def run_evaluation(
    mock: bool = False,
    output_dir: Path | None = None,
    dataset_path: Path | None = None,
    top_k: int = 5,
    limit: int | None = None,
    config_keys: list[str] | None = None,
    max_retries: int = DEFAULT_RETRY_COUNT,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
) -> dict[str, Any]:
    """Run the evaluation dataset using mock fixtures or live Gemini."""
    evaluation_dir = Path(__file__).resolve().parent
    test_file = dataset_path or evaluation_dir / "test_questions.json"
    all_questions = load_evaluation_dataset(test_file)
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError("max_retries must be a non-negative integer.")
    retry_settings = (retry_base_delay, retry_max_delay, request_interval)
    if any(not math.isfinite(value) or value < 0 for value in retry_settings):
        raise ValueError("Retry delays and request interval must be non-negative.")

    selected_config_keys = list(config_keys or CONFIG_KEYS)
    if not selected_config_keys:
        raise ValueError("Select at least one evaluation configuration.")
    if len(selected_config_keys) != len(set(selected_config_keys)):
        raise ValueError("Evaluation configurations must not be repeated.")
    unknown_configs = set(selected_config_keys) - set(CONFIG_KEYS)
    if unknown_configs:
        raise ValueError(
            "Unknown evaluation configuration(s): "
            + ", ".join(sorted(unknown_configs))
        )
    questions = all_questions[:limit] if limit is not None else all_questions

    settings = get_settings()
    api_key = settings.llm.api_key

    if mock:
        collection = OfflineFixtureCollection(questions)
        indexed_filenames = set(collection.fixture_filenames)
        generator = MockEvaluatorGenerator()
        corpus_kind = "synthetic_offline_fixtures"
        model_name = None
        embedding_model = None
        print("MOCK MODE: scores are synthetic diagnostics, not live quality measurements.")
    else:
        _validate_prerequisites(questions, api_key, None)
        base_model_requests = len(questions) * len(selected_config_keys) * 2
        vector_query_slots = len(questions) * sum(
            key in {"chromadb_semantic", "hybrid_retrieval", "hybrid_rerank"}
            for key in selected_config_keys
        )
        print(
            "LIVE request estimate: at most "
            f"{base_model_requests} generation/judge calls before retries "
            f"(bounded runner ceiling {base_model_requests * (max_retries + 1)}); "
            "generation failures skip their judge call. Semantic/hybrid retrieval "
            f"has up to {vector_query_slots} vector-query attempt(s) before retries "
            f"(bounded runner ceiling {vector_query_slots * (max_retries + 1)}); "
            "these may invoke Gemini embeddings depending on the persisted ChromaDB "
            "embedding-function configuration. "
            "These runner ceilings exclude Gemini service-internal retries for "
            "server errors and do not guarantee quota availability."
        )
        collection = get_collection()
        indexed_docs = get_all_document_records(collection)
        indexed_filenames = {doc["filename"] for doc in indexed_docs}
        _validate_prerequisites(questions, api_key, indexed_filenames)

        generator = get_generator()
        corpus_kind = "configured_local_chroma"
        model_name = settings.llm.model
        embedding_model = settings.llm.embedding_model
        print(f"LIVE MODE: Gemini model = {model_name}")

    request_controller = RequestController(
        generator,
        enabled=not mock,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
        request_interval=request_interval,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_dir or evaluation_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    expected_outputs = (
        "results.csv", "evaluation_report.json", "results_matrix.md"
    )
    if any((run_dir / name).exists() for name in expected_outputs):
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation artifacts in {run_dir}"
        )

    results = []
    print(
        f"\nEvaluating {len(questions)} questions across "
        f"{len(selected_config_keys)} configurations...\n"
    )

    stop_reason = None
    for question in questions:
        print(f"Question {question['id']}: {question['question'][:65]}")
        for config_key in selected_config_keys:
            if stop_reason:
                results.append(_skipped_result(question, config_key, stop_reason))
                continue
            result = run_configuration_evaluation(
                config_key,
                question,
                collection,
                generator,
                top_k=top_k,
                is_mock=mock,
                request_controller=request_controller,
            )
            results.append(result)

            status = result["evaluation_status"].upper()
            relevance = result["relevance_score"]
            relevance_text = (
                f"{relevance:.3f}" if relevance is not None else "UNSCORED"
            )
            latency_text = (
                f"{result['latency_sec']}s"
                if result["latency_sec"] is not None
                else "N/A"
            )
            print(
                f"  [{status}] {CONFIG_NAMES[config_key]}: "
                f"{'Synthetic relevance' if mock else 'Relevance'}="
                f"{relevance_text}, "
                f"Latency={latency_text}"
            )
            if result["generation_error"]:
                print(f"    Generation error: {result['generation_error']}")
            if result["retrieval_error"]:
                print(f"    Retrieval error: {result['retrieval_error']}")
            if result["relevance_error"]:
                print(f"    Judge note: {result['relevance_error']}")
            if result["quota_failure"]:
                quota = json.loads(result["quota_failure"])
                stop_reason = (
                    f"Stopped after {quota['kind']} during {quota['operation']}; "
                    "remaining configurations/questions were skipped."
                )
                print(f"    {stop_reason}")

    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = {}
    for key in selected_config_keys:
        rows = [row for row in results if row["config_key"] == key]
        attempted_rows = [
            row for row in rows if row["evaluation_status"] != "skipped"
        ]
        skipped_count = len(rows) - len(attempted_rows)
        generation_successes = sum(
            row["generation_status"] == "succeeded" for row in attempted_rows
        )
        generation_failures = sum(
            row["generation_status"] == "failed" for row in attempted_rows
        )
        generation_attempts = generation_successes + generation_failures
        evaluation_failures = sum(
            row["evaluation_status"] == "failed" for row in attempted_rows
        )
        relevance_values = [
            row["relevance_score"] for row in attempted_rows
            if row["relevance_score"] is not None
        ]
        faith_values = [
            row["faithfulness_score"] for row in attempted_rows
            if row["faithfulness_score"] is not None
        ]
        citation_values = [
            row["citation_accuracy"] for row in attempted_rows
            if row["citation_accuracy"] is not None
        ]
        latencies = [
            row["latency_sec"] for row in attempted_rows
            if row["latency_sec"] is not None
        ] if not mock else []

        def average(values):
            return round(sum(values) / len(values), 3) if values else None

        recall_lists = {}
        for k in (1, 3, 5):
            values = [
                row[f"recall_at_{k}"] for row in attempted_rows
                if row[f"recall_at_{k}"] is not None
            ]
            recall_lists[f"recall_at_{k}"] = average(values)

        precision_values = [
            row["precision_at_5"] for row in attempted_rows
            if row["precision_at_5"] is not None
        ]
        mrr_values = [
            row["mrr_at_5"] for row in attempted_rows
            if row["mrr_at_5"] is not None
        ]
        quota_kinds = [_quota_kind(row) for row in attempted_rows]

        summary[key] = {
            "name": CONFIG_NAMES[key],
            "attempted_evaluations": len(attempted_rows),
            "successful_evaluations": sum(
                row["evaluation_status"] == "succeeded"
                for row in attempted_rows
            ),
            "failed_evaluations": evaluation_failures,
            "skipped_evaluations": skipped_count,
            "generation_successes": generation_successes,
            "generation_failures": generation_failures,
            "generation_attempts": generation_attempts,
            "generation_skipped": sum(
                row["generation_status"] == "skipped" for row in rows
            ),
            "generation_success_rate": (
                round(generation_successes / generation_attempts, 3)
                if generation_attempts else None
            ),
            "success_rate": (
                round(
                    sum(row["evaluation_status"] == "succeeded"
                        for row in attempted_rows) / len(attempted_rows),
                    3,
                )
                if attempted_rows else None
            ),
            "judge_failures": sum(
                row["judge_status"] == "failed" for row in attempted_rows
            ),
            "duplicate_judges_avoided": sum(
                row["judge_cache_hit"] for row in attempted_rows
            ),
            "judge_skipped": sum(
                row["judge_status"] in {
                    "skipped", "skipped_generation_failed",
                }
                for row in rows
            ),
            "quota_failures": sum(
                kind is not None for kind in quota_kinds
            ),
            "quota_responses": sum(
                row["quota_response_count"] for row in attempted_rows
            ),
            "daily_quota_failures": quota_kinds.count("daily_quota_exhausted"),
            "hard_quota_failures": quota_kinds.count("quota_exhausted"),
            "rate_limit_failures": quota_kinds.count("rate_limit_exhausted"),
            "api_failures": sum(
                row["api_failure_count"] for row in attempted_rows
            ),
            "runner_api_requests": sum(
                row["api_request_count"] for row in attempted_rows
            ),
            "vector_query_slots": sum(
                row["vector_query_count"] for row in attempted_rows
            ),
            "retries": sum(
                row["retry_count"] for row in attempted_rows
            ),
            "mean_relevance": average(relevance_values),
            "relevance_scored_evaluations": len(relevance_values),
            "relevance_unscored_evaluations": (
                len(attempted_rows) - len(relevance_values)
            ),
            "mean_faithfulness": average(faith_values),
            "faithfulness_scored_evaluations": len(faith_values),
            "mean_citation_accuracy": average(citation_values),
            "citation_scored_evaluations": len(citation_values),
            "mean_latency_sec": average(latencies),
            **recall_lists,
            "precision_at_5": average(precision_values),
            "mrr_at_5": average(mrr_values),
            "retrieval_metric_evaluations": sum(
                row["recall_at_5"] is not None for row in attempted_rows
            ),
        }

    attempted_total = sum(
        row["evaluation_status"] != "skipped" for row in results
    )
    skipped_total = len(results) - attempted_total
    generation_attempts_total = sum(
        row["generation_status"] in {"succeeded", "failed"}
        for row in results
    )
    generation_successes_total = sum(
        row["generation_status"] == "succeeded" for row in results
    )
    quota_kinds = [_quota_kind(row) for row in results]
    run_counts = {
        "attempted_evaluations": attempted_total,
        "successful_evaluations": sum(
            row["evaluation_status"] == "succeeded" for row in results
        ),
        "failed_evaluations": sum(
            row["evaluation_status"] == "failed" for row in results
        ),
        "skipped_evaluations": skipped_total,
        "generation_successes": generation_successes_total,
        "generation_failures": sum(
            row["generation_status"] == "failed" for row in results
        ),
        "generation_skipped": sum(
            row["generation_status"] == "skipped" for row in results
        ),
        "generation_success_rate": (
            round(generation_successes_total / generation_attempts_total, 3)
            if generation_attempts_total else None
        ),
        "generation_attempts": generation_attempts_total,
        "judge_failures": sum(row["judge_status"] == "failed" for row in results),
        "duplicate_judges_avoided": sum(
            row["judge_cache_hit"] for row in results
        ),
        "judge_skipped": sum(
            row["judge_status"] in {"skipped", "skipped_generation_failed"}
            for row in results
        ),
        "quota_failures": sum(kind is not None for kind in quota_kinds),
        "quota_responses": sum(
            row["quota_response_count"] for row in results
        ),
        "daily_quota_failures": quota_kinds.count("daily_quota_exhausted"),
        "hard_quota_failures": quota_kinds.count("quota_exhausted"),
        "rate_limit_failures": quota_kinds.count("rate_limit_exhausted"),
        "api_failures": sum(row["api_failure_count"] for row in results),
        "runner_api_requests": sum(row["api_request_count"] for row in results),
        "vector_query_slots": sum(row["vector_query_count"] for row in results),
        "retries": sum(row["retry_count"] for row in results),
    }

    report = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "execution_mode": "MOCK/OFFLINE" if mock else "LIVE_GEMINI",
        "total_test_questions": len(questions),
        "dataset_question_count": len(all_questions),
        "questions_excluded_by_limit": len(all_questions) - len(questions),
        "selected_configurations": selected_config_keys,
        "total_configurations_tested": len(selected_config_keys),
        "total_configuration_question_pairs": len(results),
        "indexed_documents": sorted(indexed_filenames),
        "corpus_kind": corpus_kind,
        "generation_model": model_name,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "retry_policy": {
            "max_retries_per_request": max_retries,
            "base_delay_seconds": retry_base_delay,
            "maximum_retry_delay_seconds": retry_max_delay,
            "request_interval_seconds": request_interval,
            "policy": (
                "Daily quota errors stop immediately; temporary 429 errors "
                "respect Retry-After or use bounded exponential backoff with "
                "jitter. Evaluation stops after an unresolved 429."
            ),
        },
        "latency_definition": (
            "Per attempted configuration: retrieval plus generation elapsed "
            "time; excludes relevance judging. Offline latency is not aggregated."
        ),
        "relevance_metric_definition": (
            "Synthetic lexical-overlap diagnostic; not comparable to Gemini "
            "relevance scores."
            if mock
            else "Gemini judge score against the reference answer; not independent "
            "ground truth."
        ),
        "embedding_request_note": (
            "ChromaDB semantic/hybrid retrieval may call the persisted embedding "
            "function; vector-query pacing slots are counted, but external embedding "
            "requests are not included in runner API request counts."
        ),
        "request_count_scope": (
            "Counts runner-managed generation/judge calls, their 429 retries, and "
            "vector-query attempts. It excludes GeminiGenerator's internal retries "
            "for server errors and does not assume every Chroma query makes an "
            "external embedding request."
        ),
        "run_counts": run_counts,
        "stop_reason": stop_reason,
        "dataset_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
        "dataset_path": str(test_file),
        "configurations": summary,
    }

    with open(
        run_dir / "evaluation_report.json", "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    with open(
        run_dir / "results_matrix.md", "w", encoding="utf-8"
    ) as file:
        file.write(_build_results_matrix(report))

    _print_summary(report, run_dir)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-RAG retrieval and answer generation."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--mock",
        action="store_true",
        help="Run the synthetic offline pipeline diagnostic.",
    )
    modes.add_argument(
        "--live",
        action="store_true",
        help="Run live Gemini evaluation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks (default: 5).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to a verified JSON or JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N questions.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=CONFIG_KEYS,
        help="Configurations to run (default: all five).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_RETRY_COUNT,
        help=f"Retries after temporary 429 responses (default: {DEFAULT_RETRY_COUNT}).",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=DEFAULT_RETRY_BASE_DELAY,
        help=f"Base exponential retry delay in seconds (default: {DEFAULT_RETRY_BASE_DELAY}).",
    )
    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=DEFAULT_RETRY_MAX_DELAY,
        help=f"Maximum retry delay/wait budget in seconds (default: {DEFAULT_RETRY_MAX_DELAY}).",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL,
        help=f"Minimum seconds between runner-managed requests (default: {DEFAULT_REQUEST_INTERVAL}).",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    return run_evaluation(
        mock=args.mock,
        dataset_path=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        config_keys=args.configs,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        retry_max_delay=args.retry_max_delay,
        request_interval=args.request_interval,
    )


if __name__ == "__main__":
    main()
