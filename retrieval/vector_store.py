import os
import json
import uuid
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np

from core.similarity import cosine_similarity
from core.ranking import rank_results


class VectorStore:
    """
    Persistent file-backed vector store for DocuFlow.
    Stores chunk text, embedding vectors, and associated metadata with disk persistence.
    """

    def __init__(self, storage_path: str = "data/embeddings/vector_store.json"):
        self.storage_path = storage_path
        self.documents: List[Dict[str, Any]] = []
        self._ensure_storage_dir()
        self.load()

    def _ensure_storage_dir(self):
        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def add(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a single document chunk and embedding to the vector store.
        """
        record_id = doc_id or str(uuid.uuid4())
        record = {
            "id": record_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": time.time()
        }
        self.documents.append(record)
        return record_id

    def add_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple document chunks and embeddings in batch.
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings.")

        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]

        added_ids = []
        for text, embedding, meta, doc_id in zip(texts, embeddings, metadatas, ids):
            record = {
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": meta,
                "created_at": time.time()
            }
            self.documents.append(record)
            added_ids.append(doc_id)

        return added_ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k closest chunks using cosine similarity.
        Optionally filter results by metadata key-values.
        """
        if not self.documents:
            return []

        # Filter documents by metadata if filter provided
        candidate_docs = self.documents
        if filter_metadata:
            target_source = filter_metadata.get("source")
            if target_source and target_source != "all":
                target_str = str(target_source).strip().lower()
                candidate_docs = [
                    d for d in self.documents
                    if (
                        target_str in str(d.get("metadata", {}).get("source", "")).lower()
                        or target_str in str(d.get("metadata", {}).get("doc_id", "")).lower()
                        or target_str in str(d.get("metadata", {}).get("original_filename", "")).lower()
                        or target_str in str(d.get("metadata", {}).get("title", "")).lower()
                    )
                ]
            else:
                candidate_docs = [
                    d for d in self.documents
                    if all(d.get("metadata", {}).get(k) == v for k, v in filter_metadata.items())
                ]

        if not candidate_docs:
            return []

        # Vectorized similarity calculation if numpy available
        try:
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return []

            embeddings_matrix = np.array([doc["embedding"] for doc in candidate_docs], dtype=np.float32)
            norms = np.linalg.norm(embeddings_matrix, axis=1)
            # Avoid division by zero
            norms = np.where(norms == 0, 1e-10, norms)
            scores = np.dot(embeddings_matrix, query_vec) / (norms * query_norm)

            results = []
            for doc, score in zip(candidate_docs, scores):
                results.append({
                    "id": doc["id"],
                    "chunk": doc["text"],
                    "text": doc["text"],
                    "score": float(score),
                    "metadata": doc.get("metadata", {})
                })
        except Exception:
            # Fallback to pure Python similarity
            results = []
            for doc in candidate_docs:
                score = cosine_similarity(query_embedding, doc["embedding"])
                results.append({
                    "id": doc["id"],
                    "chunk": doc["text"],
                    "text": doc["text"],
                    "score": score,
                    "metadata": doc.get("metadata", {})
                })

        return rank_results(results, top_k=top_k, min_score=min_score)

    def delete(self, doc_ids: Union[str, List[str]]) -> int:
        """
        Delete document chunks by ID. Returns number of removed chunks.
        """
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        target_set = set(doc_ids)

        initial_len = len(self.documents)
        self.documents = [d for d in self.documents if d["id"] not in target_set]
        removed_count = initial_len - len(self.documents)
        return removed_count

    def clear(self):
        """
        Clear all documents in the vector store.
        """
        self.documents = []
        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except OSError:
                self.save()

    def save(self, path: Optional[str] = None):
        """
        Persist vector store to JSON file.
        """
        target = path or self.storage_path
        self._ensure_storage_dir()
        payload = {
            "version": "1.0",
            "count": len(self.documents),
            "updated_at": time.time(),
            "documents": self.documents
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: Optional[str] = None):
        """
        Load vector store from JSON file if it exists.
        """
        target = path or self.storage_path
        if os.path.exists(target):
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
            except Exception as e:
                print(f"Warning: Failed to load vector store from {target}: {e}")
                self.documents = []
        else:
            self.documents = []

    def count(self) -> int:
        """
        Return the total number of stored vectors.
        """
        return len(self.documents)

    def get_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return all records (excluding large raw embedding vectors for compact display).
        """
        records = [
            {
                "id": d["id"],
                "text": d["text"],
                "metadata": d.get("metadata", {}),
                "created_at": d.get("created_at")
            }
            for d in self.documents
        ]
        if limit:
            return records[:limit]
        return records

    def get_stats(self) -> Dict[str, Any]:
        """
        Return vector store statistics.
        """
        dimension = len(self.documents[0]["embedding"]) if self.documents else 0
        sources = set()
        for doc in self.documents:
            src = doc.get("metadata", {}).get("source")
            if src:
                sources.add(src)

        file_size_kb = 0.0
        if os.path.exists(self.storage_path):
            file_size_kb = round(os.path.getsize(self.storage_path) / 1024, 2)

        return {
            "total_chunks": len(self.documents),
            "dimension": dimension,
            "distinct_sources": len(sources),
            "sources": list(sources),
            "storage_path": self.storage_path,
            "file_size_kb": file_size_kb
        }


if __name__ == "__main__":
    store = VectorStore(storage_path="data/embeddings/test_store.json")
    store.clear()

    # Dummy test vectors
    store.add("Sample doc 1", [1.0, 0.0, 0.0], {"source": "manual"})
    store.add("Sample doc 2", [0.0, 1.0, 0.0], {"source": "manual"})
    store.add("Sample doc 3", [0.8, 0.2, 0.0], {"source": "manual"})

    hits = store.search([1.0, 0.1, 0.0], top_k=2)
    print("Search hits:", hits)
    print("Store stats:", store.get_stats())
    store.save()
    store.clear()
    if os.path.exists("data/embeddings/test_store.json"):
        os.remove("data/embeddings/test_store.json")
