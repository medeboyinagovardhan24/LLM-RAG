# LLM-RAG: Domain-Specific Question Answering System

A Retrieval-Augmented Generation (RAG) application that answers questions from your own documents. It retrieves the most relevant passages (text, tables, and image descriptions) from uploaded PDF, DOCX, and image files, then uses Google's Gemini models to generate grounded answers with source citations.

## Features

- **Document processing** — extracts text and tables from PDF and DOCX files, plus embedded image previews from PDFs; runs OCR and vision analysis on scanned pages and uploaded images when Gemini is configured.
- **Hybrid retrieval** — combines ChromaDB vector similarity with BM25 keyword scoring, plus content-type-aware boosting for tables and images.
- **Grounded answers with citations** — responses cite the source file, page, and chunk.
- **Hallucination guard** — confidence scoring and low-grounding warnings.
- **Query rewriting** — optional LLM-based query expansion for better recall.
- **Streaming** — token-by-token answers over Server-Sent Events (SSE).
- **Evaluation framework** — document-level retrieval metrics and Gemini-judged relevance; faithfulness is not measured because claim-level verification is not implemented. Live evaluation requires an API key and indexed ground-truth documents.
- **Caching and monitoring** — in-memory response cache with TTL and per-endpoint request metrics.
- **Presentation and viva guide** — architecture, demo sequence, setup, limitations, troubleshooting, and viva questions in [`docs/presentation-viva-guide.md`](./docs/presentation-viva-guide.md).

## Technology Stack

- Python, FastAPI, Uvicorn
- ChromaDB — vector store
- Google Gemini API (`google-genai`) — generation, embeddings, and vision
- pypdf, pdfplumber, pypdfium2, Pillow, pytesseract — document parsing and OCR
- rank-bm25 — keyword retrieval
- Static HTML, CSS, and JavaScript frontend
- Nginx — frontend serving and API reverse proxy in Docker

## Project Structure

```text
LLM-RAG/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── services.py
│   │   ├── hybrid_search.py
│   │   ├── guard.py
│   │   ├── reranker.py
│   │   ├── evaluator.py
│   │   ├── cache_monitor.py
│   │   ├── memory.py
│   │   ├── schemas.py
│   │   └── config.py
│   ├── tests/
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── Dockerfile
├── frontend/
├── evaluation/
├── docs/
├── run.py
├── docker-compose.yml
├── Makefile
├── .env.example
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/medeboyinagovardhan24/LLM-RAG.git
cd LLM-RAG
```

### 2. Create a Virtual Environment

Create the Python virtual environment (Python 3.12 or newer is recommended):

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
```

**macOS or Linux:**

```bash
source .venv/bin/activate
python -m pip install -r backend/requirements-dev.txt
```

### 3. Install Runtime Dependencies Only (Optional)

For a runtime-only installation, install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

`backend/requirements-dev.txt` includes the runtime requirements, so use that
single file for development and testing.

### 4. Configure the Gemini API Key

Create a `.env` file from the example:

**Windows PowerShell:**

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

**macOS or Linux:**

```bash
cp .env.example .env
```

Open `.env` and configure your Google Gemini API key:

```env
GEMINI_API_KEY=your-gemini-api-key-here
```

Get an API key from:

https://aistudio.google.com/apikey

The model and RAG parameters can also be configured using environment variables, including `GEMINI_MODEL`, `CHUNK_SIZE`, `TOP_K`, `RRF_K`, `MAX_UPLOAD_MB`, and `ALLOWED_ORIGINS`, depending on the settings supported by the application. Hybrid retrieval uses weighted reciprocal rank fusion; `RRF_K` controls the rank-saturation constant and defaults to 60.

**Important:** Never commit your actual `.env` file or expose your API key publicly.
The application has no user authentication or server-side tenant isolation.
Run it only on a trusted machine/network; CORS settings do not provide access
control. The included local launchers and Docker frontend bind to loopback by
default. Do not change those bindings or otherwise expose the unauthenticated
API to an untrusted network. Treat uploaded documents, extracted images,
conversation history, and evaluation output as potentially sensitive local data.

## Running the Application

### Option A — Run the Backend and Frontend Together

From the project root, run:

```powershell
.\.venv\Scripts\python.exe run.py
```

The launcher starts the backend API and frontend, then prints the URLs to open in your browser. It can select available ports automatically.

### Option B — Run the Backend Only

From the project root:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### Option C — Run with Docker

Ensure Docker Desktop is running and your `.env` file contains the required configuration.

From the project root:

```bash
docker compose up --build
```

The frontend is configured to be served at:

http://localhost:8080

The backend API is accessible through the frontend's configured `/api` reverse proxy.

The exact ports and service configuration depend on `docker-compose.yml`.

On Windows, `.\.venv\Scripts\python.exe run.py` serves the frontend and API on selected local
ports. The launcher serves the matching API configuration dynamically and
does not rewrite the tracked `frontend/config.js`.

### Makefile Commands

On systems with `make` installed, the following convenience commands may be available:

```bash
make install
make dev
make test
make docker-up
```

Refer to the project's `Makefile` for the exact behavior of each command.

## Testing

The project includes an offline test suite. It uses a temporary ChromaDB instance and a mock generator so tests can run without contacting the Gemini API or requiring a real API key.

From the project root, run the tests manually:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests
.\.venv\Scripts\python.exe -m pip check
```

The tests are run manually; this repository does not configure a CI workflow.

## Evaluation

The evaluation suite defines five configurations. Run an explicit mode from
the project root:

```powershell
.\.venv\Scripts\python.exe evaluation/run_evaluation.py --mock
.\.venv\Scripts\python.exe evaluation/run_evaluation.py --live
```

`--mock` runs a deterministic, no-network fixture check without an API key or
the local vector database. Its canned answers and synthetic corpus do not
measure model or retrieval quality; the report leaves quality metrics
unavailable. `--live` requires `GEMINI_API_KEY`, an available Gemini service,
and all labeled source PDFs indexed in the configured ChromaDB store. It uses
the configured Gemini generation/judge model and reports judge relevance
separately from retrieval metrics. It never silently switches to mock mode.

For a quota-conscious live smoke test, use
`.\.venv\Scripts\python.exe evaluation\run_evaluation.py --live --limit 1`.
This runs one question across the five configurations (up to 10 generation and
judge calls before retries); the runner prints the bounded retry estimate before
starting. ChromaDB semantic/hybrid retrieval may make additional Gemini
embedding calls depending on the persisted collection configuration. A daily
quota failure stops immediately, and an unresolved temporary 429 stops the
remaining run; unattempted rows are recorded as skipped.

Live retrieval measures are filename-level (not chunk-level), and relevance
is assessed by an LLM judge rather than ground truth. Faithfulness is not
reported because the existing lexical heuristic does not verify claims against
evidence. Inspect [evaluation/README.md](./evaluation/README.md) for definitions,
limitations, and dataset notes. Each run writes CSV, JSON, and Markdown artifacts
to a unique `evaluation/runs/<run-id>/` directory so previous results are not
overwritten. The CSV contains questions, references, and generated answers;
review it before sharing.

## Limitations

- Answer quality depends on the quality of the source documents and retrieved context.
- Multilingual generation does not guarantee equally effective multilingual retrieval.
- Scanned documents, tables, and embedded images depend on extraction and vision quality. Image descriptions require the configured Gemini vision service; without it, images can be stored but are not semantically described.
- `pytesseract` is a Python wrapper; the Tesseract executable must also be installed and available on `PATH` for local OCR of uploaded images.
- Gemini API availability, rate limits, and credentials can affect generation and evaluation.
- Retrieval and answer quality may vary with document structure and query complexity.
- The reported confidence is a heuristic source-overlap indicator, not a calibrated probability or a guarantee that an answer is correct.
- There is no authentication, authorization, or server-side session isolation. Do not expose the service to untrusted networks without adding and validating an access-control layer.
- Hybrid BM25 scoring examines the full filtered collection; latency and memory use can increase as the corpus grows. Benchmark with the expected production corpus before scaling.

## Future Enhancements

- Production-grade authentication and per-user data isolation.
- Improved reranking and cross-encoder scoring.
- More reliable table and embedded-image extraction.
- Expanded evaluation datasets and metrics.
- Improved multilingual document retrieval.
- Enhanced monitoring and retrieval diagnostics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Repository

GitHub: https://github.com/medeboyinagovardhan24/LLM-RAG
