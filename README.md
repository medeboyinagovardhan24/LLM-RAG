# LLM-RAG: Domain-Specific Question Answering System

A Retrieval-Augmented Generation (RAG) application that answers questions from your own documents. It retrieves the most relevant passages (text, tables, and image descriptions) from uploaded PDFs and images, then uses Google's Gemini models to generate grounded answers with source citations.

## Features

- **Document processing** — extracts text and tables from PDFs; runs OCR and vision analysis on scanned pages and uploaded images.
- **Hybrid retrieval** — combines ChromaDB vector similarity with BM25 keyword scoring, plus content-type-aware boosting for tables and images.
- **Grounded answers with citations** — responses cite the source file, page, and chunk.
- **Hallucination guard** — confidence scoring and low-grounding warnings.
- **Query rewriting** — optional LLM-based query expansion for better recall.
- **Streaming** — token-by-token answers over Server-Sent Events (SSE).
- **Evaluation framework** — offline harness for retrieval recall and answer faithfulness and relevance.
- **Caching and monitoring** — in-memory response cache with TTL and per-endpoint request metrics.

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

Create the Python virtual environment:

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS or Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the application dependencies:

```bash
pip install -r backend/requirements.txt
```

To install the development dependencies required for testing:

```bash
pip install -r backend/requirements-dev.txt
```

### 4. Configure the Gemini API Key

Create a `.env` file from the example:

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
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

The model and RAG parameters can also be configured using environment variables, including `GEMINI_MODEL`, `CHUNK_SIZE`, `TOP_K`, `MAX_UPLOAD_MB`, and `ALLOWED_ORIGINS`, depending on the settings supported by the application.

**Important:** Never commit your actual `.env` file or expose your API key publicly.

## Running the Application

### Option A — Run the Backend and Frontend Together

From the project root, run:

```bash
python run.py
```

The launcher starts the backend API and frontend, then prints the URLs to open in your browser. It can select available ports automatically.

### Option B — Run the Backend Only

From the project root:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
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

To run the tests manually:

```bash
cd backend
pytest
```

Run the test suite whenever you want to verify the application after making code changes.

## Evaluation

The evaluation framework measures answer relevance, retrieval Recall@K, comparisons between RAG and non-RAG responses, and out-of-domain handling.

From the project root, run:

```bash
python evaluation/run_evaluation.py
```

The evaluation uses the configured test questions and produces evaluation reports and CSV results when supported by the current implementation.

Generated evaluation outputs are excluded from version control when configured in `.gitignore`.

Results depend on the indexed documents, retrieval settings, available evaluation data, and Gemini API availability.

## Limitations

- Answer quality depends on the quality of the source documents and retrieved context.
- Multilingual generation does not guarantee equally effective multilingual retrieval.
- Scanned documents, tables, and embedded images depend on extraction and vision quality.
- Gemini API availability, rate limits, and credentials can affect generation and evaluation.
- Retrieval and answer quality may vary with document structure and query complexity.

## Future Enhancements

- Improved reranking and cross-encoder scoring.
- More reliable table and embedded-image extraction.
- Expanded evaluation datasets and metrics.
- Improved multilingual document retrieval.
- Enhanced monitoring and retrieval diagnostics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Repository

GitHub: https://github.com/medeboyinagovardhan24/LLM-RAG
