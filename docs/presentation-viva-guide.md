# LLM-RAG: Domain-Specific Question Answering System

## Architecture in one minute

The application has a static HTML/CSS/JavaScript frontend and a FastAPI backend. On upload, the backend validates the file and size, hashes it to identify duplicates, extracts page text and tables, and optionally extracts and analyzes images. It splits the extracted content into overlapping chunks with filename, page, chunk, and content-type metadata. ChromaDB persists the chunks and their embeddings. For a question, the backend retrieves candidates using ChromaDB semantic search and BM25 keyword search, combines their rankings, and sends the retrieved evidence to Gemini with instructions to answer from that evidence and cite the numbered sources. The UI streams the answer and displays the source metadata returned by the API.

The API also supports a heuristic confidence indicator, optional query rewriting and evaluation, response caching, analytics, and document deletion. The confidence value is a heuristic, not a calibrated probability. The lightweight `rerank_by_relevance` helper exists in the code but is not currently called by the API request path.

## End-to-end workflow

### Document ingestion

1. The frontend posts a PDF or supported image to `POST /api/upload`.
2. The backend enforces the configured byte limit, checks the file extension and content, and uses SHA-256 to detect duplicates.
3. PDFs are parsed with pdfplumber, with pypdf as a text-extraction fallback. Tables are stored as separate Markdown/structured-data chunks. pypdfium2 renders pages for embedded-image extraction; scanned-page OCR and image descriptions use Gemini Vision when configured.
4. Uploaded image files use Gemini Vision when configured, then try pytesseract as a local OCR fallback. The Python package alone is not enough: the Tesseract executable must also be installed and available on `PATH`. If no text can be extracted, the upload is rejected instead of indexing fabricated placeholder text.
5. The backend splits text into character-based chunks (defaults: 800 characters with 100 characters of overlap) and stores metadata including filename, page, chunk, and content type.
6. ChromaDB persists chunks and embeddings under `DATA_DIR/chroma_db`; original uploads and extracted image files are stored under `DATA_DIR/uploads`.

### Question answering

1. The frontend sends a question to `POST /api/ask/stream`; the non-streaming `POST /api/ask` endpoint is also available.
2. Hybrid retrieval gets semantic candidates from ChromaDB and ranks the filtered corpus with BM25. The current implementation combines ranks with reciprocal-rank fusion (`alpha=0.7` for the vector rank) and applies small table/image relevance boosts.
3. The backend sends the retrieved chunks, metadata, question, and recent conversation context to Gemini. The guarded prompt treats retrieved text as untrusted and asks Gemini to decline when evidence is insufficient.
4. The API builds a deduplicated source list from retrieved metadata and maps each prompt citation index to its displayed source index. Citations without a retrieved source are removed, and filenames/pages are never inferred from the model's answer.
5. The stream sends its source list before answer tokens so citations can be rendered against the same numbering used in the final response. Document deletion removes the indexed chunks, uploaded file, and associated extracted image files when those files can be removed.

## Technologies and their roles

| Technology | Role in this project |
|---|---|
| FastAPI, Uvicorn | REST/SSE API and local application server |
| ChromaDB | Persistent local vector collection |
| Google `google-genai` | Gemini text generation, embeddings (when configured), and vision |
| `rank-bm25` | Keyword-based retrieval combined with vector ranking |
| pdfplumber, pypdf | PDF text and table extraction; pypdf also provides the text fallback |
| pypdfium2 | PDF page rendering and embedded image handling |
| Pillow | Image validation, conversion, and storage |
| pytesseract + Tesseract executable | Optional local OCR fallback for uploaded images |
| pytest, HTTPX | Offline backend and API tests using temporary ChromaDB and mocked Gemini |
| HTML, CSS, JavaScript | Static browser interface, upload flow, streaming responses, and source rendering |
| Nginx, Docker Compose | Optional containerized frontend/API deployment |

`GEMINI_MODEL` selects the generation/vision model (default: `gemini-3.6-flash`); `GEMINI_EMBEDDING_MODEL` selects the embedding model (default: `gemini-embedding-001`). Confirm that the configured model names are enabled for the API key before the presentation. Without a real Gemini key, local indexing can use ChromaDB's default embedding function, but Gemini answer generation is unavailable.

## Windows and VS Code setup

From the repository root in PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` and set `GEMINI_API_KEY` to your own key. Keep `.env` private. In VS Code, select `.venv\Scripts\python.exe` with **Python: Select Interpreter**. Start the full local UI and API from the repository root:

```powershell
.\.venv\Scripts\python.exe run.py
```

Open the frontend URL printed by the launcher. The API documentation is available at the printed `/docs` URL. To run only the API:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/docs`. The repository has no bundled presentation sample PDF; prepare a small, readable PDF containing material you are permitted to demonstrate.

For tests, from the repository root run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests
.\.venv\Scripts\python.exe -m pip check
```

The live evaluation script does not need the backend process to be running. It does require a real Gemini API key and the expected `royal_mart_report.pdf` already indexed in the configured ChromaDB store; that PDF is not included in this repository.

## Suggested live-demo sequence

1. Before presentation day, use a small, text-based PDF on a topic you understand. Prepare two facts with known page numbers, one table or diagram if available, and one question whose answer is absent.
2. Start the application and confirm the backend status is connected and Gemini is configured.
3. Upload the prepared PDF once. Show the reported page/chunk/table/image counts and the document list.
4. Ask a direct question answerable from one page. Explain how the answer is generated from retrieved context and point to the displayed filename/page source.
5. Ask a question answerable from a table or diagram only if your chosen PDF has one; show the content type and page metadata.
6. Ask the prepared out-of-document question. Check that the answer says the evidence is insufficient; do not present this as a mathematical guarantee against hallucinations.
7. Open the document chunk view if useful, then delete the demonstration document and confirm it disappears from the list.
8. Keep a backup copy of the PDF and screenshots, but never rely on screenshots as proof of live API behavior.

## Likely viva questions and answers

1. **What is Retrieval-Augmented Generation?**  
   RAG retrieves relevant external evidence at question time and supplies it to a language model to generate a response. This project indexes uploaded documents and includes retrieved chunks in the Gemini prompt.

2. **Why use RAG instead of fine-tuning?**  
   Uploaded content can be indexed and replaced without training a new model. Retrieval also provides evidence that can be shown as sources. RAG does not guarantee that the model will always follow that evidence.

3. **What happens from upload to answer?**  
   The backend validates and extracts the upload, chunks and embeds content, stores it in ChromaDB, retrieves by semantic and keyword methods, and asks Gemini to answer from the retrieved context with source labels.

4. **What is a chunk and why is overlap used?**  
   A chunk is a bounded text segment stored and retrieved independently. Character-based chunks limit context size; overlap preserves some context across adjacent boundaries. Defaults are 800 and 100 characters.

5. **What embedding model is used?**  
   The configured Gemini embedding model defaults to `gemini-embedding-001`. When no real Gemini key is set, ChromaDB's default embedding function is used for local indexing; that does not make Gemini generation available.

6. **Why use ChromaDB?**  
   It provides a persistent local vector collection with metadata filtering and similarity queries, which fits this single-machine demonstration. It is not a substitute for multi-tenant production storage.

7. **What is hybrid retrieval in this project?**  
   It combines ChromaDB's vector candidate ranking with BM25's lexical ranking using reciprocal-rank fusion, then adds small boosts for table/image chunks on matching query types.

8. **How does BM25 help?**  
   BM25 ranks documents by query-term relevance and can retrieve exact names, numbers, or terms that semantic similarity may under-rank. It complements, rather than replaces, vector search.

9. **How are the semantic and lexical rankings blended?**  
   The implementation uses reciprocal-rank fusion, not a direct comparison of incomparable raw distance and BM25 scores. The vector rank weight defaults to `alpha=0.7`.

10. **Does the current API run a dedicated reranker?**  
    No. A lightweight `rerank_by_relevance` helper is defined, but the API's question path does not call it. Current retrieval is vector + BM25 rank fusion with content-type boosts.

11. **How are citations produced?**  
    Each retrieved chunk has filename/page/chunk metadata. The prompt labels retrieved chunks `[SOURCE 1]`, `[SOURCE 2]`, and so on. The API builds a mapping alongside the deduplicated source list, rewrites citations to that list's numbering, and removes citations that have no retrieved source.

12. **Can citation formatting guarantee factual correctness?**  
    No. Metadata citations identify retrieved evidence, but a model can still make an unsupported claim or cite relevant but insufficient context. Evaluation with real documents is needed.

13. **How are tables handled?**  
    pdfplumber extracts tables into separate Markdown and structured-data chunks, with page and table metadata. Extraction quality depends on the PDF's layout.

14. **How are scanned PDFs handled?**  
    Pages with fewer than 40 extracted text characters can be rendered and sent to Gemini Vision for transcription when a key is configured. Without Gemini, this project does not provide a local Tesseract OCR path for scanned PDF pages.

15. **How are uploaded images handled?**  
    Gemini Vision is attempted when configured. If it does not produce text, pytesseract can invoke a local Tesseract executable for OCR. If neither yields searchable text, the upload is rejected rather than indexing generic placeholder content.

16. **What is the hallucination guard?**  
    The guarded prompt labels retrieved content and history as untrusted, instructs the model to answer only from the evidence, and asks it to state when evidence is insufficient. It is a prompt-level safeguard, not a formal proof.

17. **Is the confidence value a probability?**  
    No. It is a heuristic based on answer/source word overlap, answer length, source diversity, and citation density. It is not calibrated and must not be described as a probability.

18. **What happens when no chunks are retrieved?**  
    The API returns a fixed “not found in uploaded documents” style response without asking Gemini to invent an answer.

19. **How are duplicate uploads detected?**  
    The upload bytes are hashed with SHA-256. A matching hash already in the collection is reported as a duplicate and is not indexed again.

20. **What happens when a document is deleted?**  
    The API removes the document's indexed chunk IDs, uploaded file, and extracted image files associated with its metadata, then clears cached answers.

21. **How does the response cache stay consistent after changes?**  
    Cache keys include the question, filter, and request context. Upload and delete operations clear the cache so answers do not outlive corpus changes.

22. **What does the evaluation script measure?**  
    It computes filename-level Recall@1/3/5, Precision@5, and MRR@5 for vector and hybrid retrieval. It also asks Gemini to judge answers on a small example set; those model-judged scores are not deterministic benchmark results.

23. **Why is the evaluation dataset not enough to prove quality?**  
    It has only three examples, including two tied to a PDF absent from the repository. A credible evaluation needs a representative, reviewed dataset, indexed ground truth, and repeated tests.

24. **Where is the API key stored?**  
    In the local `.env` as `GEMINI_API_KEY`, read by the backend configuration. `.env` is ignored by Git; the key must not be committed or included in screenshots.

25. **Is the application production-ready for multiple users?**  
    No. It has no authentication, authorization, tenant isolation, or production-grade access control. It is intended for trusted local demonstrations unless those controls are added.

## Known limitations and realistic next steps

- There is no authentication, authorization, or per-user document/session isolation.
- Scanned PDF OCR and visual descriptions depend on Gemini availability; local Tesseract fallback is only used for uploaded images and requires an external executable.
- Table and embedded-image extraction depend on document layout; only a limited number of embedded images are extracted per page.
- The confidence score and out-of-domain evaluation heuristic are not calibrated safety metrics.
- The evaluation set is tiny and the expected Royal Mart PDF is not distributed with the project.
- Query rewriting is available in the API but disabled by the frontend to avoid an additional Gemini request during the live demo.
- `rerank_by_relevance` is not wired into API retrieval.
- The current hybrid implementation scans the filtered corpus for BM25, so latency and memory use grow with corpus size.
- Future work could add authentication/tenant isolation, a reviewed evaluation corpus, scalable BM25 indexing, a tested cross-encoder reranker, and OCR for scanned PDFs without external vision calls.

## Troubleshooting

| Symptom | Check |
|---|---|
| `ModuleNotFoundError` at startup | Select `.venv\Scripts\python.exe` in VS Code and install `backend\requirements-dev.txt` from the repository root. |
| UI says backend offline | Use the URL printed by `run.py`; check that its API `/api/health` endpoint responds and that the selected ports are not blocked. |
| Gemini unavailable / HTTP 503 | Check `.env`, restart the backend after changing it, verify model access and account quota, then retry. Do not paste the API key into logs or presentation material. |
| HTTP 429 or temporary Gemini failure | Wait for quota/rate limits to clear. The demo depends on an external service and cannot promise availability. |
| Image upload returns “No searchable text” | Configure Gemini Vision or install Tesseract OCR and ensure `tesseract.exe` is on `PATH`. |
| Scanned PDF has no chunks | Configure a working Gemini Vision model; the local Tesseract fallback applies to uploaded images, not scanned PDF pages. |
| PDF upload reports no readable content | Try an unlocked, valid PDF with selectable text. For a scanned file, use Gemini OCR; malformed/password-protected files can fail. |
| Upload is rejected for size | The default maximum is 20 MB; change `MAX_UPLOAD_MB` only if you have verified the machine can process the larger file. |
| “Already indexed” appears | The same file bytes are already indexed. Delete the old record first if you intentionally need to re-index it. |
| Evaluation stops before running | The local ChromaDB must contain `royal_mart_report.pdf`, and `GEMINI_API_KEY` must be configured. The dataset PDF is not bundled. |
| Docker Compose exits before startup | Add `GEMINI_API_KEY` to the project-root `.env` and confirm Docker Desktop is running. |

## Presentation-day checklist

- [ ] Run the offline tests and `python -m pip check` in the intended virtual environment.
- [ ] Verify the Gemini key and selected model with a real, non-sensitive test document; confirm account quota.
- [ ] Confirm the prepared PDF is readable, on-topic, small enough, and has known page references.
- [ ] Test the exact live-demo questions and the absent-evidence question.
- [ ] Confirm upload, answer streaming, source display, and delete all work in the browser.
- [ ] Keep a local backup of the demo PDF and a fallback presentation if Gemini is unavailable.
- [ ] Close unrelated tabs and never show `.env`, API keys, private documents, or generated evaluation claims.
- [ ] Describe confidence and evaluation scores as heuristics/model judgments, not guarantees.
