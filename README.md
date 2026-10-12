# LLM-RAG: Domain-Specific Question Answering System

A Retrieval-Augmented Generation (RAG) application that answers questions using uploaded documents, retrieves relevant context, and generates grounded answers with source references.

## Features

- **Document Processing:** Extracts text from PDF documents.
- **Semantic Retrieval:** Uses ChromaDB for vector-based document search.
- **AI-Powered Answers:** Uses Google's Gemini API to generate answers from retrieved context.
- **Source Citations:** Provides document references to help users verify answers.
- **Conversation Memory:** Supports retaining conversation history.
- **Evaluation Framework:** Evaluates question-answering performance and retrieval metrics.
- **Multilingual Support:** Gemini supports multiple languages, while retrieval quality depends on the embedding model and document language.
- **Table and Image Retrieval:** Intended to support richer document content, depending on the implemented extraction and retrieval features.

## Technology Stack

- Python
- FastAPI
- ChromaDB
- Google Gemini API
- PyPDF
- pdfplumber
- HTML, CSS, and JavaScript

## Project Structure

```text
llm/
├── backend/
├── frontend/
├── evaluation/
├── uploads/
├── chroma_db/
├── .env.example
├── .gitignore
├── README.md
└── run.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/medeboyinagovardhan24/LLM-RAG.git
cd LLM-RAG
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages required by your project. If you maintain a `requirements.txt` file, run:

```powershell
pip install -r requirements.txt
```

### 4. Configure your API key

Create a local `.env` file and configure your Gemini API key using the environment variable expected by your application.

**Never commit your `.env` file or expose your API key.**

### 5. Start the backend

Use the startup command configured for your project. For example:

```powershell
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to access the FastAPI documentation.

## Evaluation

The evaluation framework can measure:

- Answer relevance and correctness
- Retrieval Recall@K
- RAG versus non-RAG performance
- Out-of-domain question handling
- Unsupported-answer behavior

Evaluation results depend on the test dataset, retrieval configuration, API availability, and evaluation implementation.

## Limitations

- Answer quality depends on document quality and retrieved context.
- Multilingual generation does not guarantee equally effective multilingual retrieval.
- Scanned documents, tables, and embedded images require suitable extraction and indexing support.
- Gemini API availability, rate limits, and credentials can affect evaluation results.

## Future Enhancements

- Improved hybrid retrieval and reranking
- More reliable table and embedded-image extraction
- Richer source citations
- Streaming responses
- Expanded evaluation datasets and metrics

## Repository

GitHub: https://github.com/medeboyinagovardhan24/LLM-RAG
