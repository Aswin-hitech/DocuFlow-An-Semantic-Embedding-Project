import os
import tempfile
import pytest
from retrieval.vector_store import VectorStore
from retrieval.search import SemanticSearch


class TestVectorStore:
    def setup_method(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.store = VectorStore(storage_path=self.temp_file.name)
        self.store.clear()

    def teardown_method(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_add_and_search(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        v3 = [0.0, 0.0, 1.0]

        self.store.add("Doc X", v1, {"axis": "x"})
        self.store.add("Doc Y", v2, {"axis": "y"})
        self.store.add("Doc Z", v3, {"axis": "z"})

        assert self.store.count() == 3

        # Search for vector close to X
        hits = self.store.search([0.9, 0.1, 0.0], top_k=2)
        assert len(hits) == 2
        assert hits[0]["text"] == "Doc X"
        assert hits[0]["score"] > 0.8

    def test_metadata_filtering(self):
        self.store.add("Doc A", [1.0, 0.0], {"category": "tech"})
        self.store.add("Doc B", [1.0, 0.0], {"category": "finance"})

        hits = self.store.search([1.0, 0.0], top_k=5, filter_metadata={"category": "finance"})
        assert len(hits) == 1
        assert hits[0]["text"] == "Doc B"

    def test_persistence_save_load(self):
        self.store.add("Persistent Doc", [0.5, 0.5], {"source": "disk"})
        self.store.save()

        # Create new instance pointing to same file
        new_store = VectorStore(storage_path=self.temp_file.name)
        assert new_store.count() == 1
        assert new_store.documents[0]["text"] == "Persistent Doc"

    def test_delete_and_clear(self):
        doc_id = self.store.add("To be deleted", [0.1, 0.2])
        assert self.store.count() == 1
        removed = self.store.delete(doc_id)
        assert removed == 1
        assert self.store.count() == 0


class TestSemanticSearch:
    def test_semantic_search_retrieval(self):
        searcher = SemanticSearch(storage_path=None)  # In-memory
        docs = [
            "Neural networks and deep learning are transformative.",
            "Cooking authentic Italian pizza requires high oven temperatures.",
            "Cardiovascular health improves with regular exercise."
        ]
        searcher.add_chunks(docs)

        # Query related to machine learning
        results = searcher.search("artificial intelligence and neural models", top_k=1)
        assert len(results) == 1
        assert "Neural networks" in results[0]["chunk"]

    def test_index_document(self):
        searcher = SemanticSearch(storage_path=None)
        long_text = "Python is great. " * 50  # Over 500 chars
        ids = searcher.index_document(long_text, chunk_size=200, overlap=30, source="test_doc")
        assert len(ids) > 1
        assert len(searcher.chunks) == len(ids)
