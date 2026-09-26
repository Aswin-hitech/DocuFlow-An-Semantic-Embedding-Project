import os
import sys
import shutil
import tempfile
from typing import List, Dict, Any, Optional

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.search import SemanticSearch
from ingestion.document_parser import DocumentParser
from ingestion.web_scraper import WebScraper
from ingestion.cleaner import clean_text
from llm.client import LLMClient

router = APIRouter()

# Global engine instances
search_engine = SemanticSearch(storage_path="data/embeddings/vector_store.json")
web_scraper = WebScraper()
llm_client = LLMClient()


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=50)
    min_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    source: Optional[str] = None


class TextIngestRequest(BaseModel):
    text: str
    title: Optional[str] = "Manual Input"
    chunk_size: int = Field(default=500, ge=50, le=5000)
    overlap: int = Field(default=50, ge=0, le=1000)
    metadata: Optional[Dict[str, Any]] = None


class UrlIngestRequest(BaseModel):
    url: str
    chunk_size: int = Field(default=500, ge=50, le=5000)
    overlap: int = Field(default=50, ge=0, le=1000)


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=20)
    source: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@router.get("/")
def home():
    """
    Health check and service status.
    """
    stats = search_engine.get_stats()
    return {
        "status": "healthy",
        "service": "DocuFlow Semantic Search & Embedding API",
        "version": "1.0.0",
        "total_indexed_chunks": stats.get("total_chunks", 0),
        "model": stats.get("model_name", "all-MiniLM-L6-v2")
    }


@router.post("/search")
def search(request: SearchRequest):
    """
    Semantic search across indexed documents.
    """
    filter_meta = {"source": request.source} if request.source and request.source != "all" else None
    results = search_engine.search(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        filter_metadata=filter_meta
    )
    return {
        "query": request.query,
        "top_k": request.top_k,
        "source": request.source or "all",
        "count": len(results),
        "results": results
    }


@router.post("/ingest/text")
def ingest_text(request: TextIngestRequest):
    """
    Ingest, clean, chunk, and embed raw text.
    """
    cleaned = clean_text(request.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Provided text is empty or invalid.")

    meta = request.metadata or {}
    meta["title"] = request.title

    chunk_ids = search_engine.index_document(
        text=cleaned,
        source=request.title,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        metadata=meta
    )

    return {
        "status": "success",
        "message": f"Successfully indexed {len(chunk_ids)} chunks.",
        "title": request.title,
        "chunks_indexed": len(chunk_ids),
        "chunk_ids": chunk_ids
    }


@router.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    overlap: int = Form(50)
):
    """
    Upload and parse documents (PDF, DOCX, TXT, MD, CSV) and index chunks.
    """
    filename = file.filename or "uploaded_file"
    suffix = os.path.splitext(filename)[1].lower()

    # Save to temporary file to parse
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        parsed = DocumentParser.parse_file(tmp_path)
        extracted_text = parsed.get("text", "")
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail=f"No readable text could be extracted from {filename}.")

        # Persist raw file into data/raw/documents/
        dest_dir = "data/raw/documents"
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy(tmp_path, dest_path)

        metadata = parsed.get("metadata", {})
        metadata["original_filename"] = filename

        chunk_ids = search_engine.index_document(
            text=extracted_text,
            doc_id=filename,
            source=filename,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata=metadata
        )

        return {
            "status": "success",
            "message": f"Successfully parsed and indexed {len(chunk_ids)} chunks from '{filename}'.",
            "filename": filename,
            "chunks_indexed": len(chunk_ids),
            "metadata": metadata
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/ingest/url")
def ingest_url(request: UrlIngestRequest):
    """
    Scrape a webpage URL, extract clean content, chunk, and index into vector store.
    """
    try:
        scraped = web_scraper.scrape_url(request.url, save_raw=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scrape URL: {str(e)}")

    text = scraped.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Webpage returned no text content.")

    title = scraped.get("title") or request.url
    meta = scraped.get("metadata", {})

    chunk_ids = search_engine.index_document(
        text=text,
        doc_id=title,
        source=request.url,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        metadata=meta
    )

    return {
        "status": "success",
        "message": f"Successfully scraped and indexed {len(chunk_ids)} chunks from {request.url}.",
        "url": request.url,
        "title": title,
        "chunks_indexed": len(chunk_ids)
    }


@router.post("/ask")
@router.post("/rag")
@router.post("/chat")
def ask_question(request: AskRequest):
    """
    RAG & Conversational Assistant endpoint:
    Retrieves most relevant chunks (optionally scoped to a specific document or scrapped URL)
    and synthesizes an intelligent, grounded response with citations.
    """
    filter_meta = {"source": request.source} if request.source and request.source != "all" else None
    hits = search_engine.search(
        query=request.question,
        top_k=request.top_k,
        filter_metadata=filter_meta
    )
    synthesis = llm_client.generate_answer(
        query=request.question,
        retrieved_chunks=hits,
        history=request.history,
        source_filter=request.source
    )

    return {
        "question": request.question,
        "answer": synthesis.get("answer"),
        "model": synthesis.get("model"),
        "source_filter": request.source or "all",
        "retrieved_count": len(hits),
        "sources": hits
    }


@router.get("/sources")
def get_sources():
    """
    List distinct sources (documents, scraped URLs, text snippets) currently indexed in DocuFlow.
    """
    sources_dict = {}
    docs = search_engine.vector_store.documents if search_engine.vector_store else []
    for d in docs:
        meta = d.get("metadata", {})
        source_id = meta.get("source") or meta.get("doc_id") or "Direct Input"
        title = meta.get("title") or meta.get("original_filename") or source_id

        if str(source_id).startswith("http://") or str(source_id).startswith("https://"):
            doc_type = "web"
        elif any(str(source_id).lower().endswith(ext) for ext in [".pdf", ".docx", ".doc", ".txt", ".md", ".csv"]):
            doc_type = "document"
        else:
            doc_type = "text"

        if source_id not in sources_dict:
            sources_dict[source_id] = {
                "id": source_id,
                "title": title,
                "type": doc_type,
                "chunks_count": 0
            }
        sources_dict[source_id]["chunks_count"] += 1

    return {
        "total_sources": len(sources_dict),
        "sources": list(sources_dict.values())
    }


@router.get("/health")
def health():
    """
    Health endpoint alias.
    """
    return home()


@router.get("/documents")
def get_documents(limit: int = Query(default=100, ge=1, le=1000)):
    """
    List indexed chunks and their metadata.
    """
    if search_engine.vector_store:
        docs = search_engine.vector_store.get_all(limit=limit)
        return {
            "total_count": search_engine.vector_store.count(),
            "returned_count": len(docs),
            "documents": docs
        }
    return {
        "total_count": len(search_engine.chunks),
        "returned_count": len(search_engine.chunks[:limit]),
        "documents": [{"text": c} for c in search_engine.chunks[:limit]]
    }


@router.get("/stats")
def get_stats():
    """
    Get system-wide statistics for DocuFlow.
    """
    return search_engine.get_stats()


@router.delete("/documents")
def clear_documents():
    """
    Reset and wipe the vector index.
    """
    search_engine.clear()
    return {
        "status": "success",
        "message": "All indexed documents and vectors have been cleared."
    }