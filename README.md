# DocuFlow: Semantic Embedding & Retrieval Studio

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-3.0+-orange.svg)](https://www.sbert.net/)
[![Tests](https://img.shields.io/badge/Tests-44%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

> **DocuFlow** is a complete, production-ready semantic embedding, vector retrieval, and Retrieval-Augmented Generation (RAG) platform. It empowers users to index heterogeneous documents (PDFs, Word documents, raw text, and live webpages), convert them into dense mathematical vector representations, perform ultra-fast semantic search, and synthesize grounded answers with citations.

---

## 🌟 Key Features

- 🧠 **Dense Semantic Retrieval**: Utilizes 384-dimensional dense neural embeddings (`all-MiniLM-L6-v2`) to capture deep conceptual meaning, resolving vocabulary mismatch and synonym blindness.
- 📐 **Rigorous Mathematical Core**: Pure-Python and vectorized implementations of Dot Product, Euclidean L2 Norm, Cosine Similarity, and Reciprocal Rank Fusion (RRF).
- 📄 **Multi-Format Ingestion**: Native parsing for **PDF** (via PyMuPDF & PyPDF), **Word .docx** (via python-docx), **plain text**, **Markdown**, and **CSV**.
- 🌐 **Live Web Scraping**: Ingest any public web article or documentation page directly via URL with automatic boilerplate cleaning and metadata extraction.
- 💾 **Persistent Vector Store**: Lightweight, self-contained, file-backed vector database with cosine similarity search, metadata filtering, chunk IDs, and zero external database dependencies.
- 🤖 **Retrieval-Augmented Generation (RAG)**: Integrates with OpenAI models for generative Q&A, with an intelligent offline extractive synthesis engine fallback when no API key is provided.
- 💻 **Interactive Web Studio**: Sleek, modern web application featuring semantic search with similarity meters, drag-and-drop file ingestion, RAG Q&A, and a knowledge base explorer.
- 🔬 **Educational Experiments**: Four self-contained interactive scripts demonstrating vectors, cosine angles, embedding clustering, and keyword-vs-semantic search benchmarks.
- 🧪 **Exhaustive Test Suite**: 44 unit and integration tests covering chunking, embeddings, similarity math, ingestion, vector store, and REST APIs.

---

## 🏗️ System Architecture

```
                 +-------------------------------------------------+
                 |                User Interfaces                  |
                 |  Web Studio (/app)  |  REST API (/api/search)   |
                 +------------------------+------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
        +---------------------------+           +---------------------------+
        |   Ingestion Pipeline      |           |     Query Pipeline        |
        | - Document Parser (PDF/DOCX)           | - User Search Query       |
        | - Web Scraper (HTML)      |           | - RAG Q&A Question        |
        | - Cleaner & Normalizer    |           +-------------+-------------+
        +-------------+-------------+                         |
                      |                                       |
                      v                                       v
        +---------------------------+           +---------------------------+
        |      Core Chunking        |           |    Embedding Model        |
        | - Sliding window overlap  |           | - all-MiniLM-L6-v2        |
        | - Paragraph boundary split|           | - 384-dim dense vectors   |
        +-------------+-------------+           +-------------+-------------+
                      |                                       |
                      v                                       |
        +---------------------------+                         |
        |    Embedding Model        |                         |
        | - Generate 384-d vectors  |                         |
        +-------------+-------------+                         |
                      |                                       |
                      v                                       v
        +-----------------------------------------------------------+
        |             DocuFlow Persistent Vector Store              |
        |  - Cosine Similarity Matching: cos(theta) = (A.B)/(||A||*||B||)
        |  - Metadata Filtering & Top-K Ranking                     |
        |  - File-backed storage (data/embeddings/vector_store.json)|
        +-----------------------------+-----------------------------+
                                      |
                                      v
                        +---------------------------+
                        |   RAG Synthesis Engine    |
                        | - OpenAI GPT-4o-mini      |
                        | - Offline Extractive Mode |
                        +---------------------------+
```

---

## 📁 Project Directory Structure

```
DocuFlow/
├── .env                       # Local active environment configuration
├── .env.example               # Template environment configuration
├── .gitignore                 # Version control exclusions
├── conftest.py                # Pytest path discovery configuration
├── main.py                    # Server entrypoint with static UI mounting
├── README.md                  # Project overview and quickstart
├── requirements.txt           # Python package dependencies
│
├── api/                       # REST API Layer
│   ├── __init__.py
│   └── routes.py              # Endpoints: /search, /ingest, /ask, /stats
│
├── core/                      # Mathematical and Vector Processing Core
│   ├── __init__.py
│   ├── chunking.py            # Sliding window, paragraph, and metadata chunking
│   ├── embeddings.py          # SentenceTransformers model wrapper
│   ├── ranking.py             # Top-K sorting, thresholding, and RRF
│   └── similarity.py          # Dot product, L2 magnitude, cosine similarity
│
├── data/                      # Data Storage Directory
│   ├── embeddings/            # Persisted vector index JSON files
│   ├── processed/             # Processed chunk caches
│   └── raw/                   # Raw documents and scraped webpage archives
│       └── documents/         # Sample files (AI guide, solar system)
│
├── experiments/               # Educational and Benchmark Scripts
│   ├── 01_vectors.py          # Vector math from scratch
│   ├── 02_cosine_similarity.py# Angular relationships and length invariance
│   ├── 03_embeddings.py       # Domain semantic clustering matrix
│   └── 04_semantic_search.py  # Semantic search vs keyword matching benchmark
│
├── frontend/                  # Web Studio User Interface
│   ├── index.html             # Single-page application HTML5 interface
│   ├── script.js              # Reactive UI client with async API calls
│   └── style.css              # Dark-mode styling, glassmorphism, responsive grid
│
├── ingestion/                 # Document Parsing & Web Scraping Pipeline
│   ├── __init__.py
│   ├── cleaner.py             # Unicode normalizer, HTML cleaner, metadata stats
│   ├── document_parser.py     # PDF, DOCX, TXT, MD, CSV parser
│   └── web_scraper.py         # HTTP fetcher with BeautifulSoup extraction
│
├── llm/                       # Retrieval-Augmented Generation
│   ├── __init__.py
│   └── client.py              # LLM client with OpenAI and offline fallback
│
├── tests/                     # Automated Pytest Suite (44 tests)
│   ├── test_api.py            # Integration tests for FastAPI endpoints
│   ├── test_chunking.py       # Unit tests for text chunking
│   ├── test_embeddings.py     # Embeddings shape and semantic tests
│   ├── test_ingestion.py      # Cleaner and document parser tests
│   ├── test_retrieval.py      # VectorStore and SemanticSearch tests
│   └── test_similarity.py     # Vector math and cosine similarity tests
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or 3.11 installed
- Git (optional)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/DocuFlow.git
cd DocuFlow
pip install -r requirements.txt
```

### 3. Configure Environment
Copy the example `.env` file:
```bash
cp .env.example .env
```
*(Optional: Add your `OPENAI_API_KEY` to `.env` if you want OpenAI-powered generative answers for RAG. If omitted, DocuFlow automatically operates in offline synthesis mode).*

### 4. Run the Server
Launch the DocuFlow server using `uvicorn`:
```bash
python main.py
```
Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Web Studio
Open your browser and navigate to:
- **Web Studio UI**: [http://localhost:8000/app](http://localhost:8000/app)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 REST API Reference

### 1. Semantic Search
**`POST /api/search`**
```json
{
  "query": "How do computers learn patterns from data?",
  "top_k": 3,
  "min_score": 0.40
}
```

### 2. Ingest Document File
**`POST /api/ingest/file`** (Multipart Form)
- `file`: Upload PDF, DOCX, TXT, MD, or CSV file.
- `chunk_size`: Maximum character length per chunk (default: 500).
- `overlap`: Overlap characters between chunks (default: 50).

### 3. Ingest Webpage URL
**`POST /api/ingest/url`**
```json
{
  "url": "https://en.wikipedia.org/wiki/Machine_learning",
  "chunk_size": 500,
  "overlap": 50
}
```

### 4. Direct Text Ingestion
**`POST /api/ingest/text`**
```json
{
  "title": "Quantum Computing Notes",
  "text": "Quantum computers leverage superposition and entanglement...",
  "chunk_size": 500,
  "overlap": 50
}
```

### 5. Ask AI (RAG Q&A)
**`POST /api/ask`**
```json
{
  "question": "What is the difference between supervised and unsupervised learning?",
  "top_k": 3
}
```

### 6. Knowledge Base Management
- **`GET /api/stats`**: Retrieve chunk count, model name, and storage footprint.
- **`GET /api/documents`**: List all stored chunk metadata.
- **`DELETE /api/documents`**: Wipe the vector store.

---

## 🔬 Educational Experiments

Run the interactive experiment scripts to explore vector math and semantic behavior:

```bash
# Experiment 01: Vector representations and geometric operations
python experiments/01_vectors.py

# Experiment 02: Cosine similarity and length invariance
python experiments/02_cosine_similarity.py

# Experiment 03: Dense embeddings and cross-domain similarity matrix
python experiments/03_embeddings.py

# Experiment 04: Semantic search vs keyword matching benchmark
python experiments/04_semantic_search.py
```

---

## 🧪 Running Automated Tests

Run the complete test suite:
```bash
python -m pytest -v tests/
```

To run with code coverage:
```bash
python -m pytest --cov=core --cov=retrieval --cov=ingestion tests/
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
