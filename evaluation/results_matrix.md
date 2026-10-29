# LLM-RAG Experimental Evaluation Matrix

> **ARCHIVED — DO NOT PRESENT AS MEASURED RESULTS.** This historical matrix is preserved for traceability, but its offline relevance, faithfulness, citation, and latency values came from a deterministic mock pipeline and heuristic scoring, not a live model evaluation. Its retrieval scores also predate the audited metric handling. The narrative below is not a verified comparison. The corrected `--mock` mode reports quality metrics as N/A; a live run requires an explicit `--live` selection and configured Gemini key. See [`README.md`](./README.md) for the current methodology.

**Evaluation Mode:** Offline Pipeline Verification (Mock Evaluator)  
**Date:** 2026-10-18T13:27:48Z  
**Test Set Size:** 8 questions  
**Indexed Corpus Documents:** `93444.pdf`, `Govardhan_Resume.pdf`

---

### Table: Generation Relevance on the LLM-RAG Test Set

| Model / Retrieval Configuration | Live Gemini Relevance | Offline Pipeline Test Baseline | Status |
|---|---:|---:|---|
| **Gemini without retrieval** | *Pending Live Key* | 0.096 | Pipeline Verified |
| **ChromaDB Semantic Search + Gemini** | *Pending Live Key* | 0.865 | Pipeline Verified |
| **BM25 Keyword Search + Gemini** | *Pending Live Key* | 0.864 | Pipeline Verified |
| **Hybrid Retrieval + Gemini** | *Pending Live Key* | 0.864 | Pipeline Verified |
| **Hybrid Retrieval + Reranking + Gemini** | *Pending Live Key* | 0.864 | Pipeline Verified |

> **Audit & Academic Integrity Note:** Live Gemini API evaluation is currently marked **Pending** because `GEMINI_API_KEY` is not configured in `.env`. In accordance with rigorous scientific evaluation standards, live scores are **never fabricated, simulated, or presented as live measurements**. The offline test baseline verifies pipeline mechanics, retrieval routing, and scoring functions deterministically.


---

### Table 7 (Expanded): Multi-Metric Evaluation Matrix (Offline Verification Run)

| Configuration | Relevance (0–1) | Faithfulness (0–1) | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR@5 | Citation Accuracy | Avg Latency (s) | Success Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gemini without retrieval** | 0.096 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.000s | 8/8 |
| **ChromaDB Semantic Search + Gemini** | 0.865 | 0.938 | 1.000 | 1.000 | 1.000 | 0.971 | 1.000 | 1.000 | 0.505s | 8/8 |
| **BM25 Keyword Search + Gemini** | 0.864 | 0.938 | 1.000 | 1.000 | 1.000 | 0.914 | 1.000 | 1.000 | 0.488s | 8/8 |
| **Hybrid Retrieval + Gemini** | 0.864 | 0.938 | 1.000 | 1.000 | 1.000 | 0.971 | 1.000 | 1.000 | 0.450s | 8/8 |
| **Hybrid Retrieval + Reranking + Gemini** | 0.864 | 0.938 | 1.000 | 1.000 | 1.000 | 0.971 | 1.000 | 1.000 | 0.444s | 8/8 |

### Key Findings & Interpretation

1. **Impact of Retrieval on Domain-Specific Questions:**
   - The ungrounded baseline (*Gemini without retrieval*) cannot accurately answer proprietary facts (such as the specific address of the Royal Mart store or details of private resumes) and is prone to hallucination or generic guessing.
   - Grounding Gemini with retrieved document contexts substantially elevates generation relevance, faithfulness, and citation accuracy.

2. **ChromaDB Vector vs. BM25 Keyword Search:**
   - ChromaDB semantic retrieval captures conceptual and paraphrased queries effectively.
   - BM25 excels at exact keyword and numeric lookups (e.g., store numbers, email addresses, table cell values).

3. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF):**
   - Combining ChromaDB and BM25 (alpha = 0.7) achieves the most robust balance across both descriptive questions and structured table lookups.

4. **Effect of Exact Keyword Reranking:**
   - Lightweight reranking (`rerank_by_relevance`) promotes chunks containing exact query terms to top positions, ensuring high Recall@1 and improved citation grounding.
