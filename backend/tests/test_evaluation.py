"""Tests for the research-style evaluation harness."""

from __future__ import annotations

import pytest
from google.genai.errors import ClientError

import sys
from pathlib import Path

# Add project root to sys.path so evaluation module is importable
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from evaluation.run_evaluation import (
    GeminiQuotaError,
    RequestController,
    _validate_prerequisites,
    compute_deterministic_relevance,
    evaluate_citation_accuracy,
    evaluate_answer_relevance,
    run_configuration_evaluation,
    run_evaluation,
    MockEvaluatorGenerator,
    CONFIG_NAMES,
    load_evaluation_dataset,
)


def test_evaluation_requires_gemini_api_key():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        _validate_prerequisites([], "", set(), is_mock=False)


def test_evaluation_mock_mode_allows_empty_api_key():
    # In mock mode, missing API key should be allowed
    _validate_prerequisites([], "", set(), is_mock=True)


def test_evaluation_dataset_loader_normalizes_verified_json(tmp_path):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        '{"question_id":"q1","question":"Where?","expected_answer":"Here",'
        '"relevant_document":"source.pdf","relevant_page":2,'
        '"relevant_chunk_ids":["chunk-1"],"answerable":true}\n',
        encoding="utf-8",
    )

    questions = load_evaluation_dataset(dataset)

    assert questions[0]["id"] == "q1"
    assert questions[0]["ground_truth_filename"] == "source.pdf"
    assert questions[0]["ground_truth_page"] == 2
    assert questions[0]["ground_truth_chunk_ids"] == ["chunk-1"]
    assert questions[0]["is_unanswerable"] is False


def test_evaluation_requires_indexed_ground_truth_documents():
    questions = [{"ground_truth_filename": "expected.pdf"}]

    with pytest.raises(RuntimeError, match="expected.pdf"):
        _validate_prerequisites(questions, "configured", {"other.pdf"})


def test_evaluation_prerequisites_allow_indexed_ground_truth():
    _validate_prerequisites(
        [{"ground_truth_filename": "expected.pdf"}],
        "configured",
        {"expected.pdf"},
    )


def test_deterministic_relevance_identical_text():
    score = compute_deterministic_relevance("What is X?", "Royal Mart in Bengaluru", "Royal Mart in Bengaluru")
    assert score >= 0.95


def test_deterministic_relevance_fallback_refusal():
    fallback = "The answer is not found in the provided documents."
    score = compute_deterministic_relevance("Question?", fallback, fallback)
    assert score == 1.0


def test_citation_accuracy():
    metadatas = [
        {"filename": "doc1.pdf", "page": 2},
        {"filename": "doc2.pdf", "page": 5},
    ]

    # Valid citation matching doc1.pdf page 2
    acc = evaluate_citation_accuracy(
        "According to [SOURCE 1], Royal Mart operates in Bengaluru.",
        metadatas,
        ground_truth_filename="doc1.pdf",
        ground_truth_page=2,
    )
    assert acc == 1.0

    # Incorrect citation
    acc_bad = evaluate_citation_accuracy(
        "According to [SOURCE 2], Royal Mart operates in Bengaluru.",
        metadatas,
        ground_truth_filename="doc1.pdf",
        ground_truth_page=2,
    )
    assert acc_bad == 0.0

    # Mentioning a filename is not a source citation.
    assert evaluate_citation_accuracy(
        "The answer is in doc1.pdf.",
        metadatas,
        ground_truth_filename="doc1.pdf",
        ground_truth_page=2,
    ) == 0.0

    # Missing page metadata cannot verify a page-specific citation.
    assert evaluate_citation_accuracy(
        "According to [SOURCE 1], the claim is supported.",
        [{"filename": "doc1.pdf"}],
        ground_truth_filename="doc1.pdf",
        ground_truth_page=2,
    ) == 0.0

    # Unanswerable questions have no citation-accuracy target.
    acc_unanswerable = evaluate_citation_accuracy(
        "The answer is not found in the provided documents.",
        metadatas,
        ground_truth_filename="doc1.pdf",
        ground_truth_page=2,
        is_unanswerable=True,
    )
    assert acc_unanswerable is None


def test_all_five_configurations_offline_execution(collection):
    # Add dummy document to collection
    collection.add(
        ids=["doc_1"],
        documents=["Royal Mart is a supermarket in Bengaluru."],
        metadatas=[{"filename": "93444.pdf", "page": 1, "chunk": 1}],
    )

    q = {
        "id": 1,
        "question": "Where is the visited Royal Mart outlet located?",
        "expected_answer": "No.90, Aura, Kothihosahalli, Bengaluru.",
        "ground_truth_filename": "93444.pdf",
        "ground_truth_page": 1,
        "is_unanswerable": False,
    }

    mock_gen = MockEvaluatorGenerator(collection=collection)

    for config_key in CONFIG_NAMES:
        res = run_configuration_evaluation(
            config_key=config_key,
            q=q,
            collection=collection,
            generator=mock_gen,
            top_k=3,
            is_mock=True,
        )
        assert res["success"] is True
        assert res["config_key"] == config_key
        assert res["latency_sec"] is None
        assert isinstance(res["relevance_score"], float)
        assert 0.0 <= res["relevance_score"] <= 1.0
        assert res["mock_diagnostic_relevance"] == res["relevance_score"]
        assert res["judge_status"] == "synthetic_diagnostic"
        assert res["recall_at_5"] is None


def test_mock_generator_recognizes_actual_judge_prompt():
    import json

    prompt = """Evaluate the answer against the reference answer.
Score from 0.0 to 1.0:
Question: Where is Royal Mart located?
Expected Answer: Royal Mart is in Bengaluru.
Generated Answer: Royal Mart is in Bengaluru.

Return only JSON:
{"score": 0.0, "reason": "Brief explanation"}
"""
    generator = MockEvaluatorGenerator()
    response = generator.generate(prompt)
    parsed = json.loads(response)
    assert parsed["score"] == 1.0
    assert isinstance(parsed["score"], (int, float))
    assert "Synthetic lexical-overlap diagnostic" in parsed["reason"]
    assert generator.generate(prompt) == response


def test_mock_relevance_judge_validates_malformed_response():
    class MalformedMockJudge:
        def generate(self, prompt):
            return '{"score": "not-a-number", "reason": "malformed"}'

    score, reason = evaluate_answer_relevance(
        "Question?",
        "Expected answer.",
        "Generated answer.",
        MalformedMockJudge(),
        is_mock=True,
    )
    assert score is None
    assert "Judge score must be numeric" in reason


def test_successful_live_generation_and_judging_are_recorded():
    class SuccessfulGenerator:
        def __init__(self):
            self.responses = [
                "The evidence supports this answer.",
                '{"score": 0.9, "reason": "Matches the reference."}',
            ]

        def generate(self, prompt):
            return self.responses.pop(0)

    controller = RequestController(
        SuccessfulGenerator(),
        enabled=True,
        max_retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        request_interval=0,
    )
    result = run_configuration_evaluation(
        "gemini_no_retrieval",
        {"id": 1, "question": "Question?", "expected_answer": "Expected."},
        collection=None,
        generator=controller.generator,
        request_controller=controller,
    )
    assert result["generation_status"] == "succeeded"
    assert result["judge_status"] == "scored"
    assert result["success"] is True
    assert result["relevance_score"] == 0.9
    assert result["api_request_count"] == 2


def test_identical_answer_reuses_judge_result_without_duplicate_api_call():
    class IdenticalAnswers:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if "Evaluate the answer against the reference answer." in prompt:
                return '{"score": 0.8, "reason": "same answer"}'
            return "Identical generated answer."

    generator = IdenticalAnswers()
    controller = RequestController(
        generator,
        enabled=True,
        max_retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        request_interval=0,
    )
    question = {
        "id": 1,
        "question": "Question?",
        "expected_answer": "Expected.",
    }
    first = run_configuration_evaluation(
        "gemini_no_retrieval",
        question,
        collection=None,
        generator=generator,
        request_controller=controller,
    )
    second = run_configuration_evaluation(
        "gemini_no_retrieval",
        question,
        collection=None,
        generator=generator,
        request_controller=controller,
    )
    assert first["judge_cache_hit"] is False
    assert second["judge_cache_hit"] is True
    assert first["relevance_score"] == second["relevance_score"] == 0.8
    assert first["api_request_count"] == 2
    assert second["api_request_count"] == 1
    assert generator.calls == 3


def test_failed_judge_result_is_not_cached_for_identical_answer():
    class JudgeFailsOnce:
        def __init__(self):
            self.judge_calls = 0

        def generate(self, prompt):
            if "Evaluate the answer against the reference answer." in prompt:
                self.judge_calls += 1
                if self.judge_calls == 1:
                    raise ClientError(503, {"error": {"message": "temporary failure"}})
                return '{"score": 0.8, "reason": "same answer"}'
            return "Identical generated answer."

    generator = JudgeFailsOnce()
    controller = RequestController(
        generator,
        enabled=True,
        max_retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        request_interval=0,
    )
    question = {
        "id": 1,
        "question": "Question?",
        "expected_answer": "Expected.",
    }

    first = run_configuration_evaluation(
        "gemini_no_retrieval",
        question,
        collection=None,
        generator=generator,
        request_controller=controller,
    )
    second = run_configuration_evaluation(
        "gemini_no_retrieval",
        question,
        collection=None,
        generator=generator,
        request_controller=controller,
    )

    assert first["judge_status"] == "failed"
    assert first["judge_cache_hit"] is False
    assert second["judge_status"] == "scored"
    assert second["judge_cache_hit"] is False
    assert second["relevance_score"] == 0.8
    assert generator.judge_calls == 2


def test_generation_failure_skips_relevance_judge():
    class FailedGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            raise RuntimeError("generation unavailable")

    generator = FailedGenerator()
    result = run_configuration_evaluation(
        "gemini_no_retrieval",
        {"id": 1, "question": "Question?", "expected_answer": "Expected."},
        collection=None,
        generator=generator,
    )
    assert generator.calls == 1
    assert result["generation_status"] == "failed"
    assert result["judge_status"] == "skipped_generation_failed"
    assert result["relevance_score"] is None


def test_judge_failure_after_successful_generation_keeps_score_missing():
    class GeneratorWithJudgeFailure:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "A generated response."
            raise ClientError(503, {"error": {"message": "judge unavailable"}})

    result = run_configuration_evaluation(
        "gemini_no_retrieval",
        {"id": 1, "question": "Question?", "expected_answer": "Expected."},
        collection=None,
        generator=GeneratorWithJudgeFailure(),
    )
    assert result["generation_status"] == "succeeded"
    assert result["judge_status"] == "failed"
    assert result["relevance_score"] is None
    assert result["evaluation_status"] == "failed"


def test_temporary_rate_limit_respects_retry_after_and_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("evaluation.run_evaluation.time.sleep", sleeps.append)

    class RetryOnce:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise ClientError(
                    429,
                    {
                        "error": {
                            "status": "RESOURCE_EXHAUSTED",
                            "message": "Quota exceeded for requests_per_minute",
                            "details": [{"retryDelay": "7s"}],
                        }
                    },
                )
            return "success"

    generator = RetryOnce()
    controller = RequestController(
        generator,
        enabled=True,
        max_retries=2,
        retry_base_delay=0.1,
        retry_max_delay=10,
        request_interval=0,
    )
    assert controller.generate("prompt") == "success"
    assert generator.calls == 2
    assert controller.retries == 1
    assert controller.quota_responses == 1
    assert sleeps == [7.0]


def test_request_interval_paces_runner_managed_requests(monkeypatch):
    sleeps = []
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(
        "evaluation.run_evaluation.time.monotonic", lambda: next(ticks)
    )
    monkeypatch.setattr("evaluation.run_evaluation.time.sleep", sleeps.append)

    class Generator:
        def generate(self, prompt):
            return "ok"

    controller = RequestController(
        Generator(),
        enabled=True,
        max_retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        request_interval=1,
    )
    assert controller.generate("first") == "ok"
    assert controller.generate("second") == "ok"
    assert sleeps == [1.0]


def test_temporary_rate_limit_retries_are_bounded(monkeypatch):
    monkeypatch.setattr("evaluation.run_evaluation.time.sleep", lambda _: None)

    class AlwaysLimited:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            raise ClientError(
                429,
                {
                    "error": {
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "Quota exceeded for requests_per_minute",
                    }
                },
            )

    generator = AlwaysLimited()
    controller = RequestController(
        generator,
        enabled=True,
        max_retries=2,
        retry_base_delay=0,
        retry_max_delay=0,
        request_interval=0,
    )
    with pytest.raises(GeminiQuotaError) as error:
        controller.generate("prompt")
    assert error.value.daily is False
    assert generator.calls == 3
    assert controller.retries == 2


def test_hard_quota_error_is_not_retried():
    class QuotaExhausted:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            raise ClientError(
                429,
                {
                    "error": {
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "Quota exceeded for the current billing period",
                    }
                },
            )

    generator = QuotaExhausted()
    controller = RequestController(
        generator,
        enabled=True,
        max_retries=4,
        retry_base_delay=0,
        retry_max_delay=0,
        request_interval=0,
    )
    with pytest.raises(GeminiQuotaError) as error:
        controller.generate("prompt")
    assert error.value.daily is False
    assert error.value.hard_quota is True
    assert generator.calls == 1


def test_error_details_redact_api_keys_and_bearer_tokens():
    from evaluation.run_evaluation import _safe_error_detail

    detail = _safe_error_detail(
        "GEMINI_API_KEY=secret-value Authorization: Bearer token-value"
    )
    assert "secret-value" not in detail
    assert "token-value" not in detail
    assert "[REDACTED]" in detail


def test_daily_quota_failure_stops_live_run_and_marks_remaining_rows_skipped(
    monkeypatch, tmp_path, capsys
):
    from types import SimpleNamespace
    import evaluation.run_evaluation as evaluation

    class DailyQuotaGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            raise ClientError(
                429,
                {
                    "error": {
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "Quota exceeded for requests_per_day",
                    }
                },
            )

    generator = DailyQuotaGenerator()
    monkeypatch.setattr(
        evaluation,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(
                api_key="configured-key",
                model="test-model",
                embedding_model="test-embedding",
            )
        ),
    )
    monkeypatch.setattr(evaluation, "get_collection", lambda: object())
    monkeypatch.setattr(
        evaluation,
        "get_all_document_records",
        lambda collection: [
            {"filename": "93444.pdf"},
            {"filename": "Govardhan_Resume.pdf"},
        ],
    )
    monkeypatch.setattr(evaluation, "get_generator", lambda: generator)

    report = evaluation.run_evaluation(
        mock=False,
        output_dir=tmp_path / "quota-run",
        limit=1,
        max_retries=3,
        request_interval=0,
    )
    capsys.readouterr()
    assert generator.calls == 1
    assert report["run_counts"]["attempted_evaluations"] == 1
    assert report["run_counts"]["generation_failures"] == 1
    assert report["run_counts"]["skipped_evaluations"] == 4
    assert report["run_counts"]["quota_failures"] == 1
    assert report["run_counts"]["quota_responses"] == 1
    assert report["stop_reason"]


def test_cli_limit_and_configuration_options_are_forwarded(monkeypatch):
    import evaluation.run_evaluation as evaluation

    captured = {}

    def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(evaluation, "run_evaluation", fake_run_evaluation)
    evaluation.main([
        "--live",
        "--limit", "1",
        "--configs", "bm25_keyword", "gemini_no_retrieval",
        "--max-retries", "4",
        "--request-interval", "0.25",
    ])
    assert captured["limit"] == 1
    assert captured["config_keys"] == ["bm25_keyword", "gemini_no_retrieval"]
    assert captured["max_retries"] == 4
    assert captured["request_interval"] == 0.25


def test_mock_mode_abstains_for_out_of_domain_rag_prompt():
    from evaluation.run_evaluation import build_prompt

    generator = MockEvaluatorGenerator()
    prompt = build_prompt(
        "What is the capital of France?",
        ["An unrelated document about groceries."],
        [{"filename": "groceries.pdf", "page": 1}],
    )

    assert generator.generate(prompt) == "The answer is not found in the provided documents."


def test_live_relevance_judge_failure_is_missing_not_deterministic_fallback():
    class InvalidJudge:
        def generate(self, prompt):
            return "not valid JSON"

    score, reason = evaluate_answer_relevance(
        "Question?", "Reference answer", "Generated answer", InvalidJudge()
    )

    assert score is None
    assert "invalid response" in reason.lower()


@pytest.mark.parametrize("response", [None, "   "])
def test_empty_generation_is_reported_as_failure(response):
    class EmptyGenerator:
        def generate(self, prompt):
            return response

    result = run_configuration_evaluation(
        config_key="gemini_no_retrieval",
        q={"id": "empty", "question": "A question?", "expected_answer": "An answer."},
        collection=None,
        generator=EmptyGenerator(),
        top_k=5,
        is_mock=False,
    )

    assert result["success"] is False
    assert result["generated_answer"] is None
    assert result["generation_error"] == "generation_error: EmptyResponse"


def test_live_mode_without_api_key_fails_before_opening_local_index(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import evaluation.run_evaluation as evaluation

    monkeypatch.setattr(
        evaluation,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(api_key="", model="test-model", embedding_model="test-embed")
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "get_collection",
        lambda: pytest.fail("Live mode without a key must stop before opening ChromaDB"),
    )

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        evaluation.run_evaluation(mock=False, output_dir=tmp_path)


def test_mock_run_uses_offline_fixtures_and_synthetic_relevance_without_api_calls(
    monkeypatch, tmp_path, capsys
):
    from types import SimpleNamespace
    import csv
    import evaluation.run_evaluation as evaluation

    monkeypatch.setattr(
        evaluation,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(api_key="", model="unused", embedding_model="unused")
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "get_collection",
        lambda: pytest.fail("Offline mock mode must not open the local ChromaDB index"),
    )
    monkeypatch.setattr(
        evaluation,
        "get_generator",
        lambda: pytest.fail("Offline mock mode must not construct the live Gemini generator"),
    )
    output_dir = tmp_path / "mock-run"

    report = evaluation.run_evaluation(mock=True, output_dir=output_dir)
    capsys.readouterr()

    assert report["execution_mode"] == "MOCK/OFFLINE"
    assert report["corpus_kind"] == "synthetic_offline_fixtures"
    assert report["generation_model"] is None
    assert "Synthetic lexical-overlap diagnostic" in report[
        "relevance_metric_definition"
    ]
    assert all(
        isinstance(config["mean_relevance"], float)
        and config["mean_faithfulness"] is None
        and config["mean_citation_accuracy"] is None
        and config["mean_latency_sec"] is None
        and config["recall_at_5"] is None
        for config in report["configurations"].values()
    )
    assert (output_dir / "results.csv").is_file()
    assert (output_dir / "evaluation_report.json").is_file()
    assert (output_dir / "results_matrix.md").is_file()
    with (output_dir / "results.csv").open(encoding="utf-8", newline="") as results_file:
        rows = list(csv.DictReader(results_file))
    assert len(rows) == 40
    assert all(row["retrieved_chunks"] for row in rows)
    assert all(row["relevance_score"] for row in rows)
    assert all(row["judge_status"] == "synthetic_diagnostic" for row in rows)
    assert report["run_counts"]["runner_api_requests"] == 0
    matrix = (output_dir / "results_matrix.md").read_text(encoding="utf-8")
    assert "Synthetic lexical relevance diagnostic" in matrix
    assert "not real answer-quality measurements" in matrix


def test_mock_question_limit_runs_only_selected_questions_and_configurations(
    monkeypatch, tmp_path, capsys
):
    from types import SimpleNamespace
    import evaluation.run_evaluation as evaluation

    monkeypatch.setattr(
        evaluation,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(api_key="", model="unused", embedding_model="unused")
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "get_collection",
        lambda: pytest.fail("Mock mode must not open persistent ChromaDB"),
    )
    monkeypatch.setattr(
        evaluation,
        "get_generator",
        lambda: pytest.fail("Mock mode must not create the live generator"),
    )
    report = evaluation.run_evaluation(
        mock=True,
        output_dir=tmp_path / "limited-mock",
        limit=1,
        config_keys=["gemini_no_retrieval"],
    )
    capsys.readouterr()
    assert report["total_test_questions"] == 1
    assert report["questions_excluded_by_limit"] == 7
    assert report["total_configuration_question_pairs"] == 1
    assert report["run_counts"]["generation_successes"] == 1
    assert isinstance(
        report["configurations"]["gemini_no_retrieval"]["mean_relevance"], float
    )


def test_daily_quota_during_judging_keeps_generation_success_and_score_missing():
    class JudgeQuotaGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "Generated answer."
            raise ClientError(
                429,
                {
                    "error": {
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "Quota exceeded for requests_per_day",
                    }
                },
            )

    generator = JudgeQuotaGenerator()
    controller = RequestController(
        generator,
        enabled=True,
        max_retries=2,
        retry_base_delay=0,
        retry_max_delay=0,
        request_interval=0,
    )
    result = run_configuration_evaluation(
        "gemini_no_retrieval",
        {"id": 1, "question": "Question?", "expected_answer": "Expected."},
        collection=None,
        generator=generator,
        request_controller=controller,
    )
    assert result["success"] is True
    assert result["generation_status"] == "succeeded"
    assert result["judge_status"] == "failed"
    assert result["relevance_score"] is None
    assert '"operation": "relevance_judge"' in result["quota_failure"]
    assert generator.calls == 2


def test_quota_error_wrapped_by_chroma_embedding_stops_generation():
    class EmbeddingFailureCollection:
        def count(self):
            return 1

        def query(self, **kwargs):
            try:
                raise ClientError(
                    429,
                    {
                        "error": {
                            "status": "RESOURCE_EXHAUSTED",
                            "message": "Quota exceeded for requests_per_day",
                        }
                    },
                )
            except ClientError as exc:
                raise RuntimeError("Failed to generate embeddings via Gemini") from exc

    class UnusedGenerator:
        def generate(self, prompt):
            pytest.fail("Generation must be skipped after retrieval quota failure")

    controller = RequestController(
        UnusedGenerator(),
        enabled=True,
        max_retries=1,
        retry_base_delay=0,
        retry_max_delay=0,
        request_interval=0,
    )
    result = run_configuration_evaluation(
        "chromadb_semantic",
        {"id": 1, "question": "Question?", "expected_answer": "Expected."},
        collection=EmbeddingFailureCollection(),
        generator=controller.generator,
        request_controller=controller,
    )
    assert result["generation_status"] == "skipped"
    assert result["judge_status"] == "skipped_generation_failed"
    assert result["quota_failure"] is not None
    assert '"operation": "retrieval_embedding"' in result["quota_failure"]
    assert result["api_failure_count"] == 1


def test_temporary_vector_embedding_limit_retries_bounded_operation(monkeypatch):
    monkeypatch.setattr("evaluation.run_evaluation.time.sleep", lambda _: None)
    calls = 0

    def query():
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                raise ClientError(
                    429,
                    {
                        "error": {
                            "status": "RESOURCE_EXHAUSTED",
                            "message": "per_minute rate limit",
                        }
                    },
                )
            except ClientError as exc:
                raise RuntimeError("Embedding request failed") from exc
        return ["retrieved"]

    controller = RequestController(
        generator=object(),
        enabled=True,
        max_retries=1,
        retry_base_delay=0,
        retry_max_delay=0,
        request_interval=0,
    )
    assert controller.vector_query(query) == ["retrieved"]
    assert calls == 2
    assert controller.retries == 1
    assert controller.quota_responses == 1
    assert controller.vector_query_slots == 2
