import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/api/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "DocuFlow" in data["service"]


def test_search_endpoint_empty():
    response = client.post("/api/search", json={"query": "test query", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert isinstance(data["results"], list)


def test_ingest_text_and_search():
    ingest_payload = {
        "text": "Deep residual networks revolutionized convolutional image recognition with skip connections.",
        "title": "ResNet Paper Notes",
        "chunk_size": 200,
        "overlap": 20
    }
    ingest_res = client.post("/api/ingest/text", json=ingest_payload)
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"

    # Search for this ingested content
    search_res = client.post("/api/search", json={"query": "convolutional neural networks residual connections", "top_k": 2})
    assert search_res.status_code == 200
    hits = search_res.json()["results"]
    assert len(hits) >= 1
    assert "residual networks" in hits[0]["chunk"].lower()


def test_rag_ask_endpoint():
    res = client.post("/api/ask", json={"question": "What did residual networks revolutionize?", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["sources"]) >= 1


def test_stats_and_documents_endpoint():
    res = client.get("/api/stats")
    assert res.status_code == 200
    assert "total_chunks" in res.json()

    res_docs = client.get("/api/documents")
    assert res_docs.status_code == 200
    assert "documents" in res_docs.json()
