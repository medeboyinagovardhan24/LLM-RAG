"""Regression tests for hybrid retrieval and retrieval evaluation."""

from __future__ import annotations

from app.evaluator import evaluate_ranked_retrieval
from app.hybrid_search import (
    hybrid_search,
    reciprocal_rank_fusion,
    rerank_by_relevance,
)


class FakeCollection:
    def __init__(self, records: list[tuple[str, str, dict]]) -> None:
        self.records = records

    def count(self, where=None):
        return sum(
            1 for _, _, metadata in self.records
            if not where or all(metadata.get(key) == value for key, value in where.items())
        )

    def query(self, query_texts, n_results, where=None, include=None):
        records = [
            row for row in self.records
            if not where or all(row[2].get(key) == value for key, value in where.items())
        ][:n_results]
        return {
            "ids": [[row[0] for row in records]],
            "documents": [[row[1] for row in records]],
            "metadatas": [[row[2] for row in records]],
            "distances": [[0.1 for _ in records]],
        }

    def get(self, include=None, where=None):
        records = [
            row for row in self.records
            if not where or all(row[2].get(key) == value for key, value in where.items())
        ]
        return {
            "ids": [row[0] for row in records],
            "documents": [row[1] for row in records],
            "metadatas": [row[2] for row in records],
        }


def test_hybrid_search_uses_full_bm25_corpus_and_unique_chunk_ids():
    text_metadata = {
        "filename": "doc.pdf",
        "page": 1,
        "chunk": 1,
        "content_type": "text",
    }
    records = [
        ("vector-text", "ordinary page text", text_metadata),
        *[
            (
                f"other-{index}",
                f"ordinary material {index}",
                {"filename": "doc.pdf", "page": index + 2, "chunk": 1, "content_type": "text"},
            )
            for index in range(10)
        ],
        (
            "bm25-image",
            "A diagram image contains the uniquevisualword label",
            {
                "filename": "doc.pdf",
                "page": 1,
                "chunk": 1,
                "content_type": "image",
            },
        ),
    ]

    documents, metadatas, _ = hybrid_search(
        FakeCollection(records),
        "uniquevisualword image",
        top_k=2,
        alpha=0.3,
    )

    assert "A diagram image contains the uniquevisualword label" in documents
    assert {metadata["content_type"] for metadata in metadatas} == {"text", "image"}


def test_bm25_only_does_not_run_vector_search_or_return_vector_only_candidates():
    class BM25OnlyCollection(FakeCollection):
        def query(self, **kwargs):
            raise AssertionError("BM25-only search must not query vectors")

    collection = BM25OnlyCollection([
        ("vector-only", "ordinary semantic material", {"filename": "a.pdf", "page": 1}),
        ("lexical-hit", "needle appears in this passage", {"filename": "b.pdf", "page": 2}),
        ("other-1", "different alpha material", {"filename": "c.pdf", "page": 3}),
        ("other-2", "different beta material", {"filename": "d.pdf", "page": 4}),
    ])

    documents, metadatas, scores = hybrid_search(
        collection, "needle", top_k=2, alpha=0.0
    )

    assert documents == ["needle appears in this passage"]
    assert metadatas[0]["filename"] == "b.pdf"
    assert len(scores) == 1


def test_bm25_only_does_not_apply_content_type_boosts():
    table_first = FakeCollection([
        ("table", "needle common passage", {"filename": "a.pdf", "content_type": "table"}),
        ("text", "needle common passage", {"filename": "b.pdf", "content_type": "text"}),
        *[
            (f"other-{index}", f"unrelated passage {index}", {"filename": f"other-{index}.pdf"})
            for index in range(8)
        ],
    ])
    text_first = FakeCollection([
        ("table", "needle common passage", {"filename": "a.pdf", "content_type": "text"}),
        ("text", "needle common passage", {"filename": "b.pdf", "content_type": "table"}),
        *[
            (f"other-{index}", f"unrelated passage {index}", {"filename": f"other-{index}.pdf"})
            for index in range(8)
        ],
    ])

    table_docs, table_metas, table_scores = hybrid_search(
        table_first, "needle rows", top_k=2, alpha=0.0
    )
    text_docs, text_metas, text_scores = hybrid_search(
        text_first, "needle rows", top_k=2, alpha=0.0
    )

    assert table_docs == text_docs
    assert table_scores == text_scores
    assert [meta["filename"] for meta in table_metas] == [
        meta["filename"] for meta in text_metas
    ]


def test_exact_term_reranker_changes_order_when_keyword_evidence_is_stronger():
    documents = ["an unrelated passage", "the requested raretoken appears here"]
    metadatas = [{"chunk": 1}, {"chunk": 2}]

    reranked_docs, reranked_metas, reranked_scores = rerank_by_relevance(
        "raretoken", documents, metadatas, [0.7, 0.1]
    )

    assert reranked_docs[0] == documents[1]
    assert reranked_metas[0]["chunk"] == 2
    assert reranked_scores[0] > reranked_scores[1]


def test_hybrid_search_applies_document_filter():
    collection = FakeCollection([
        ("a", "alpha content", {"filename": "a.pdf", "page": 1, "chunk": 1}),
        ("b", "beta content", {"filename": "b.pdf", "page": 1, "chunk": 1}),
    ])

    documents, metadatas, _ = hybrid_search(
        collection, "alpha", top_k=2, filter_filename="b.pdf"
    )

    assert documents == ["beta content"]
    assert metadatas[0]["filename"] == "b.pdf"


def test_hybrid_search_rejects_alpha_outside_unit_interval():
    import pytest

    with pytest.raises(ValueError, match="alpha"):
        hybrid_search(FakeCollection([]), "query", alpha=1.1)


def test_reciprocal_rank_fusion_merges_and_deduplicates_rankings():
    scores = reciprocal_rank_fusion(
        [["semantic-a", "shared", "semantic-b"], ["keyword-a", "shared"]],
        k=60,
    )

    assert set(scores) == {"semantic-a", "shared", "semantic-b", "keyword-a"}
    assert scores["shared"] == 1 / 62 + 1 / 62
    assert scores["shared"] > scores["semantic-b"]


def test_reciprocal_rank_fusion_handles_empty_result_sets():
    scores = reciprocal_rank_fusion([[], ["keyword-only"]], k=10)

    assert scores == {"keyword-only": 1 / 11}


def test_hybrid_search_keeps_keyword_results_when_vector_search_is_empty():
    class EmptyVectorCollection(FakeCollection):
        def query(self, **kwargs):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

    documents, metadatas, _ = hybrid_search(
        EmptyVectorCollection([
            ("keyword-hit", "needle appears here", {"filename": "doc.pdf"}),
        ]),
        "needle",
        top_k=1,
        alpha=0.7,
    )

    assert documents == ["needle appears here"]
    assert metadatas == [{"filename": "doc.pdf"}]


def test_ranked_retrieval_metrics_use_relevant_identifier_rank():
    assert evaluate_ranked_retrieval(
        ["wrong.pdf", "source.pdf", "also-wrong.pdf"],
        ["source.pdf"],
        k=3,
    ) == {
        "precision_at_k": 0.3333,
        "recall_at_k": 1.0,
        "mrr_at_k": 0.5,
    }


def test_ranked_retrieval_requires_positive_k():
    import pytest

    with pytest.raises(ValueError, match="positive"):
        evaluate_ranked_retrieval([], ["source.pdf"], k=0)
