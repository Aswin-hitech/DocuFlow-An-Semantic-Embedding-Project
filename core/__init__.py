"""
Core modules for DocuFlow: text chunking, embedding generation,
vector similarity, and semantic ranking.
"""

from core.chunking import chunk_text
from core.embeddings import EmbeddingModel
from core.similarity import dot_product, magnitude, cosine_similarity
from core.ranking import rank_results

__all__ = [
    "chunk_text",
    "EmbeddingModel",
    "dot_product",
    "magnitude",
    "cosine_similarity",
    "rank_results"
]
