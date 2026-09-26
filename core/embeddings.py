import os
# Prevent transformers from attempting to load TensorFlow/Keras 3
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import List, Union
from sentence_transformers import SentenceTransformer
from core.similarity import cosine_similarity


class EmbeddingModel:
    """
    Embedding Model wrapper using HuggingFace sentence-transformers.
    Defaults to all-MiniLM-L6-v2 which yields high-performance 384-dimensional embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text string.

        Args:
            text: Input string.

        Returns:
            List of floats representing the embedding vector.
        """
        if not isinstance(text, str):
            text = str(text)

        embedding = self.model.encode(text)
        return embedding.tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of text strings.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Ensure all items are strings
        cleaned_texts = [str(t) for t in texts]
        embeddings = self.model.encode(cleaned_texts)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """
        Return the vector dimension of the underlying model.
        """
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    model = EmbeddingModel()

    texts = [
        "I love programming.",
        "I enjoy writing code.",
        "The weather is very cold today."
    ]

    embeddings = model.generate_embeddings(texts)

    for text, embedding in zip(texts, embeddings):
        print("\nText:", text)
        print("Dimensions:", len(embedding))

    print("\n--- Similarities ---")

    similarity_1 = cosine_similarity(
        embeddings[0],
        embeddings[1]
    )

    similarity_2 = cosine_similarity(
        embeddings[0],
        embeddings[2]
    )

    print("Programming vs Coding:", similarity_1)
    print("Programming vs Weather:", similarity_2)