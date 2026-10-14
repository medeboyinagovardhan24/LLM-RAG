# LLM-RAG: Domain-Specific Question Answering System

A Retrieval-Augmented Generation (RAG) application that answers questions from your own documents. It retrieves the most relevant passages (text, tables, and image descriptions) from uploaded PDFs and images, then uses Google's Gemini models to generate grounded answers with source citations.

## Features

- **Document processing** — extracts text and tables from PDFs; runs OCR / vision analysis on scanned pages and uploaded images.
- **Hybrid retrieval** — combines ChromaDB vector similarity with BM25 keyword scoring, plus content-type–aware boosting for tables and images.
- **Grounded answers with citations** — responses cite the source file, page, and chunk.
- **Hallucination guard** — confidence scoring and low-grounding warnings.
- **Query rewriting** — optional LLM-based query expansion for better recall.
- **Streaming** — token-by-token answers over Server-Sent Events.
- **Evaluation framework** — offline harness for retrieval recall and answer faithfulness/relevance.
- **Caching & monitoring** — in-memory response cache with TTL and per-endpoint request metrics.

## Technology stack

- Python · FastAPI · Uvicorn
- ChromaDB (vector store)
- Google Gemini API (`google-genai`) — generation, embeddings, and vision
- pypdf · pdfplumber · pypdfium2 · Pillow · pytesseract (document parsing / OCR)
- rank-bm25 (keyword retrieval)
- Static HTML/CSS/JS frontend (served by nginx in Docker)

## Project structure

```text
LLM-RAG/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI HTTP layer (routes, middleware, streaming)
│   │   ├── services.py      # RAG core: ingestion, ChromaDB, Gemini generator
│   │   ├── hybrid_search.py # vector + BM25 retrieval
│   │   ├── guard.py         # hallucination detection / confidence scoring
│   │   ├── reranker.py      # LLM query rewriting & expansion
│   │   ├── evaluator.py     # faithfulness / relevance / coverage metrics
│   │   ├── cache_monitor.py # response cache + request monitoring
│   │   ├── memory.py        # conversation history & analytics
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── config.py        # environment-driven settings
│   ├── tests/               # offline pytest suite (no network / no real key)
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── Dockerfile
├── frontend/                # static UI; nginx reverse-proxies /api in Docker
├── evaluation/              # offline RAG evaluation harness
├── docs/                    # architecture diagrams
├── run.py                   # one-command local launcher (API + UI)
├── docker-compose.yml
├── Makefile
├── .env.example
└── README.md
```

## Setup

### 1. Clone

```bash
git clone https://github.com/medeboyinagovardhan24/LLM-RAG.git
cd LLM-RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
# For running the test suite as well:
pip install -r backend/requirements-dev.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and set your Google Gemini API key (get one at https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your-gemini-api-key-here
```

The model and RAG parameters are configurable there too (`GEMINI_MODEL`, `CHUNK_SIZE`, `TOP_K`, `MAX_UPLOAD_MB`, `ALLOWED_ORIGINS`, …). **Never commit your `.env`.**

## Running

### Option A — one command (backend + frontend)

```bash
python run.py
```

This starts the API and serves the static UI, printing both URLs (it auto-selects free ports).

### Option B — backend only

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Then open the interactive API docs at http://127.0.0.1:8000/docs.

### Option C — Docker (full stack)

```bash
docker compose up --build
```

The UI is served at http://localhost:8080 and proxies `/api` to the backend. `GEMINI_API_KEY` is read from your `.env`.

> On systems with `make`, the `Makefile` wraps these: `make install`, `make dev`, `make test`, `make docker-up`.

## Testing

The suite runs fully offline — it uses a throwaway ChromaDB in a temp directory and a fake generator, so no network access or real API key is required:

```bash
cd backend
pytest
```

CI runs this automatically on every push and pull request (see `.github/workflows/ci.yml`).

## Evaluation

The evaluation harness measures answer relevance, retrieval Recall@K, RAG vs. non-RAG performance, and out-of-domain handling:

```bash
python evaluation/run_evaluation.py
```

It reads `evaluation/test_questions.json` and writes `evaluation/evaluation_report.json` and `results.csv` (both git-ignored, since they are generated). Results depend on the indexed documents, retrieval configuration, and Gemini API availability.

## Limitations

- Answer quality depends on document quality and the retrieved context.
- Multilingual generation does not guarantee equally effective multilingual retrieval.
- Scanned documents, tables, and embedded images depend on extraction/vision quality.
- Gemini API availability, rate limits, and credentials affect generation and evaluation.

## Future enhancements

- Improved reranking and cross-encoder scoring
- More reliable table and embedded-image extraction
- Expanded evaluation datasets and metrics

## License

MIT — see [LICENSE](LICENSE).

## Repository

GitHub: https://github.com/medeboyinagovardhan24/LLM-RAG
