# LLM-RAG Research-Style Evaluation

This directory contains a five-configuration evaluation harness for **LLM-RAG: Domain-Specific Question Answering System**. Offline execution is a deterministic pipeline check, not a model-quality experiment. Live answer relevance is Gemini-judge-assessed and must not be treated as ground truth.

---

## 1. Configurations Evaluated

| ID | Configuration Name | Description |
|---|---|---|
| 1 | **Gemini without retrieval** | Parametric LLM baseline. Direct prompt without retrieved document context. |
| 2 | **ChromaDB Semantic Search + Gemini** | Dense vector similarity retrieval via ChromaDB embeddings (`top_k=5`). |
| 3 | **BM25 Keyword Search + Gemini** | Sparse lexical keyword retrieval via `rank-bm25` (`alpha=0.0`). |
| 4 | **Hybrid Retrieval + Gemini** | Reciprocal Rank Fusion (RRF) blending dense vector and sparse BM25 scores (`alpha=0.7`). |
| 5 | **Hybrid Retrieval + Reranking + Gemini** | Hybrid retrieval followed by exact-term keyword boosting (`rerank_by_relevance`). |

---

## 2. Evaluation Dataset (`test_questions.json`)

The test set comprises 8 curated questions covering:
- **Fact Retrieval:** Specific domain facts from visited site reports (`93444.pdf`).
- **Table Retrieval:** Structured data extraction from markdown tables (operational processes and logistics impact tables).
- **Single-document profile:** One resume question (`Govardhan_Resume.pdf`), not a multi-document synthesis test.
- **Insufficient Evidence / Unanswerable:** Domain questions where the corpus explicitly lacks evidence. Ground-truth expected behavior is to report *"The answer is not found in the provided documents."*
- **Out-of-Domain (OOD):** Questions outside the indexed corpus to assess hallucination resistance.

The dataset currently has filename/page labels but no chunk-level relevance judgments, graded labels, multi-document question, scanned-image/OCR case, or ambiguity/clarification case. PDF files and labeled pages are present locally. A text-extraction check found expected-keyword matches for most answerable references, but one source-page question had only 2 of 4 expected keyword strings match; all references (especially unanswerable claims) still require manual subject-matter review before a live research comparison.

For verified datasets, the runner also accepts JSON or JSONL via
`--dataset`. Each record should contain `question_id`, `question`,
`expected_answer`, `relevant_document`, `relevant_page` (optional),
`relevant_chunk_ids` (optional), and `answerable`. The checked-in fixture uses
backward-compatible aliases (`id`, `ground_truth_filename`,
`is_unanswerable`). The runner validates and normalizes either shape; it does
not invent answers or relevance labels.

---

## 3. Metrics, Definitions, and Limits

1. **Answer relevance (0–1):** Live mode asks the configured Gemini model to judge a generated answer against the reference answer. The score is model-assessed, not accuracy or independent ground truth. Invalid judge responses and judge failures are missing values with a per-question error; they are not replaced with lexical fallback scores. Mock mode instead records the deterministic lexical-overlap diagnostic in both `relevance_score` and `mock_diagnostic_relevance`, and its mock-only mean is labelled synthetic. It is not a Gemini judgment, semantic relevance measurement, or a score that can be compared with live results.

2. **Faithfulness:** Not reported. The existing token-overlap heuristic does not verify individual answer claims against supporting evidence.

3. **Retrieval metrics:** Ground truth is currently a single expected filename per answerable question; retrieved chunks are collapsed into unique filenames before scoring. Recall@k is 1 if that filename appears in the first k unique retrieved filenames and 0 otherwise. Precision@5 is relevant unique filenames in the top five divided by 5; MRR@5 is 1/rank of the first matching filename, or 0 if absent. Unanswerable questions and questions without a labeled filename are N/A. These are document-level, not chunk-level relevance measurements.
   NDCG is not computed because the dataset has no graded relevance labels.

4. **Citation attribution:** For answerable questions in live mode, each `[SOURCE N]` must map to the expected filename and page. The measure checks attribution only; it does not verify that the passage supports every claim. Unanswerable questions are N/A.

5. **Latency:** Per-question wall time for retrieval plus answer generation. It excludes the Gemini relevance judge. Offline timing is N/A. Eight questions are too few for a stable performance claim.

6. **Successful generations:** Number of questions with successful retrieval (where applicable) and answer generation, shown as successful/attempted. Relevance-judge errors are separately recorded in the per-question CSV.

---

## 4. How to Run the Evaluation

### Option A: Offline mock pipeline check
This mode uses a synthetic in-memory fixture collection, deterministic canned answers, and a deterministic lexical-overlap relevance diagnostic. It requires no Gemini key, network access, local ChromaDB index, or source PDFs. It validates control flow and artifact formats only; its synthetic relevance scores do not measure retrieval quality or Gemini answer quality.
```powershell
.\.venv\Scripts\python.exe evaluation/run_evaluation.py --mock
```
Or using the alias runner:
```powershell
.\.venv\Scripts\python.exe evaluation/evaluate_retrieval.py --mock
```

### Option B: Live Gemini evaluation
Requires your Google Gemini API key set in `.env` or as an environment variable:
1. Ensure `.env` contains:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key
   GEMINI_MODEL=gemini-3.6-flash
   ```
2. Run the evaluation script:
   ```powershell
   .\.venv\Scripts\python.exe evaluation/run_evaluation.py --live
   ```

   To evaluate a separately verified dataset:
   ```powershell
   .\.venv\Scripts\python.exe evaluation/run_evaluation.py --live --dataset evaluation\verified_questions.jsonl
   ```
Live mode never silently falls back to mock. It uses the configured local ChromaDB collection, the `GEMINI_MODEL` setting for generation/judging, and the configured embedding model for semantic retrieval. All labeled source documents must already be indexed. The current dataset has eight questions and filename/page labels; verify reference answers and source pages against the PDFs before treating it as a final test set.

### Small, quota-conscious live runs

Start with one question rather than all 8:
```powershell
.\.venv\Scripts\python.exe evaluation\run_evaluation.py --live --limit 1
```
That selects one question across all five configurations. The runner prints a preflight estimate of up to **10 generation/judge calls before retries**, with a default runner-managed ceiling of **30** (two retries per call). Generation failures skip their judge call. It also reports up to **3 vector-query attempts before retries** (9 with the default retry budget); each may additionally invoke Gemini embeddings depending on the collection's persisted embedding-function configuration. The number of actual embedding requests depends on that persisted configuration. Existing SDK-level retries for server errors may also add requests, so the estimates are not a quota guarantee.

Select a subset when fewer calls are desired:
```powershell
.\.venv\Scripts\python.exe evaluation\run_evaluation.py --live --limit 1 --configs bm25_keyword
```
Configuration IDs are `gemini_no_retrieval`, `chromadb_semantic`, `bm25_keyword`, `hybrid_retrieval`, and `hybrid_rerank`. With one selected configuration, the runner-managed generation/judge ceiling is 2 calls before retries (6 with the default retry budget).

`--max-retries` (default 2) applies only to temporary HTTP 429/resource-exhausted responses, including errors raised from Chroma's embedding function. The runner honors an API `Retry-After`/retry delay if it fits within `--retry-max-delay` (default 60 seconds); otherwise it stops instead of retrying earlier than requested. Without a supplied delay, it uses bounded exponential backoff with jitter. `--request-interval` (default 1 second) paces runner-managed generation/judge requests and Chroma vector queries. These controls do not guarantee quota availability. Explicit daily/hard quota exhaustion stops immediately; a temporary 429 that persists after retries also halts the run. Remaining configuration/question rows are recorded as skipped, not as failed generations.

The app's existing ChromaDB embedding-function configuration warning is handled in `backend/app/services.py` by reopening the existing collection with its persisted embedding function. This runner does not rebuild, delete, or migrate the database to silence that warning. Preserve the database and uploaded documents; the persisted embedding function may affect whether vector retrieval generates additional Gemini embedding calls.

Example tuning options:
```powershell
.\.venv\Scripts\python.exe evaluation\run_evaluation.py --live --limit 1 --max-retries 1 --request-interval 2
```

---

## 5. Output Artifacts

Each invocation creates a unique run directory under `evaluation/runs/` and refuses to overwrite artifacts:
- **`results.csv`**: Granular, row-by-row logs of each question and each configuration (generated answers, retrieval lists, errors, and individual metric scores).
- **`evaluation_report.json`**: Machine-readable aggregate statistics, run ID, model/settings metadata, and dataset hash.
- **`results_matrix.md`**: Summary with unavailable metrics clearly shown as N/A.

The JSON report separates attempted, successful, failed, and skipped evaluations; generation success/failure; judge failures/skips; terminal quota failures versus all quota/rate-limit responses (including recovered retries); duplicate judge calls avoided; and runner-managed call/retry counts. Identical question/reference/answer judge prompts reuse a cached score within that run. A missing judge score remains `null` and is excluded from averages. The CSV records status, ordered chunk references, source metadata, retrieval score and score type, answer, metric values, quota/API errors, and skip reasons for each question/configuration. Mock output stores the same synthetic score in `relevance_score` and `mock_diagnostic_relevance`; the mock-only relevance mean is labelled synthetic, remains a lexical diagnostic, and must not be compared to live Gemini results. Other unmeasured mock quality metrics remain N/A.

Latency is retrieval plus generation wall time and excludes the relevance judge; mock latency is not aggregated. Runner-managed request counts do not include ChromaDB's external embedding calls or retries internal to the existing Gemini service for server errors. Avoid sharing the CSV without reviewing it: it includes questions, expected answers, generated answers, and bounded/redacted error details. Existing older files in `evaluation/` are not overwritten by new runs and may use earlier evaluation logic.
