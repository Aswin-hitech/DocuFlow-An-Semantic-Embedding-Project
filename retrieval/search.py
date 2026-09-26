import os
import sys
from typing import List, Dict, Any, Optional

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.embeddings import EmbeddingModel
from core.similarity import cosine_similarity
from core.ranking import rank_results
from core.chunking import chunk_text, chunk_text_with_metadata
from retrieval.vector_store import VectorStore


class SemanticSearch:
    """
    High-level Semantic Search Engine.
    Coordinates embedding generation, chunking, persistent vector storage,
    and similarity-based retrieval.
    """

    def __init__(self, storage_path: Optional[str] = "data/embeddings/vector_store.json"):
        self.embedding_model = EmbeddingModel()
        self.storage_path = storage_path
        self.vector_store = VectorStore(storage_path=storage_path) if storage_path else None

        # In-memory fallback / backward-compatibility buffers
        self.chunks: List[str] = []
        self.chunk_embeddings: List[List[float]] = []

    def add_chunks(
        self,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        source: Optional[str] = None
    ):
        """
        Add text chunks, generate embeddings, and store them.
        Maintains backward compatibility with original add_chunks(self, chunks).
        """
        if not chunks:
            return

        self.chunks.extend(chunks)
        new_embeddings = self.embedding_model.generate_embeddings(chunks)
        self.chunk_embeddings.extend(new_embeddings)

        if self.vector_store:
            built_metadatas = []
            for i, chunk in enumerate(chunks):
                meta = {"source": source or "direct_input", "index": i}
                if metadatas and i < len(metadatas):
                    meta.update(metadatas[i])
                built_metadatas.append(meta)

            self.vector_store.add_batch(
                texts=chunks,
                embeddings=new_embeddings,
                metadatas=built_metadatas
            )
            self.vector_store.save()

    def index_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
        chunk_size: int = 500,
        overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Chunk a long document, embed all chunks, and index into the store.
        """
        enriched_chunks = chunk_text_with_metadata(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            doc_id=doc_id,
            source=source,
            extra_metadata=metadata
        )

        texts = [c["text"] for c in enriched_chunks]
        metadatas = [c["metadata"] for c in enriched_chunks]
        ids = [c["chunk_id"] for c in enriched_chunks]

        if not texts:
            return []

        embeddings = self.embedding_model.generate_embeddings(texts)

        # Update in-memory lists
        self.chunks.extend(texts)
        self.chunk_embeddings.extend(embeddings)

        # Store persistently
        if self.vector_store:
            self.vector_store.add_batch(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            self.vector_store.save()

        return ids

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for the most semantically similar chunks for a given query.
        """
        if not query or not query.strip():
            return []

        # 1. Convert user query into an embedding
        query_embedding = self.embedding_model.generate_embedding(query)

        # If persistent vector store has records, use its optimized search
        if self.vector_store and self.vector_store.count() > 0:
            return self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
                filter_metadata=filter_metadata
            )

        # Fallback to in-memory chunks
        if not self.chunks:
            return []

        results = []
        for chunk, chunk_embedding in zip(self.chunks, self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            results.append({
                "chunk": chunk,
                "text": chunk,
                "score": score
            })

        return rank_results(results, top_k=top_k, min_score=min_score)

    def clear(self):
        """
        Clear all indexed chunks from memory and persistent store.
        """
        self.chunks = []
        self.chunk_embeddings = []
        if self.vector_store:
            self.vector_store.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the search index.
        """
        if self.vector_store:
            stats = self.vector_store.get_stats()
            stats["model_name"] = self.embedding_model.model_name
            return stats

        return {
            "total_chunks": len(self.chunks),
            "model_name": self.embedding_model.model_name,
            "dimension": self.embedding_model.dimension if self.chunk_embeddings else 0
        }


if __name__ == "__main__":
    documents = [
        "Machine learning allows computers to learn patterns from data.",
        "Python is a popular programming language used for software development.",
        "Supervised learning uses labelled data to train machine learning models.",
        "Neural networks are computational models inspired by the human brain.",
        "The capital of France is Paris."
    ]

    search_engine = SemanticSearch(storage_path=None)  # In-memory for demo run

    search_engine.add_chunks(documents)

    query = "How do computers learn from data?"

    results = search_engine.search(
        query,
        top_k=3
    )

    print("\nQuery:")
    print(query)

    print("\nTop Results:")

    for i, result in enumerate(results, start=1):
        print(f"\nRank {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['chunk']}")