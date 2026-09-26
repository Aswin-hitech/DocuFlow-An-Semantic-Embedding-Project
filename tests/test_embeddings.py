import pytest
from core.embeddings import EmbeddingModel
from core.similarity import cosine_similarity


class TestEmbeddingModel:
    @classmethod
    def setup_class(cls):
        cls.model = EmbeddingModel()

    def test_model_dimension(self):
        assert self.model.dimension == 384

    def test_single_embedding(self):
        text = "Hello world of natural language processing."
        emb = self.model.generate_embedding(text)
        assert isinstance(emb, list)
        assert len(emb) == 384
        assert all(isinstance(x, float) for x in emb)

    def test_batch_embeddings(self):
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = self.model.generate_embeddings(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384

    def test_empty_batch(self):
        assert self.model.generate_embeddings([]) == []

    def test_semantic_consistency(self):
        # "cats" and "felines" should be closer than "cats" and "quantum mechanics"
        t1 = "Felines and domesticated cats make quiet pets."
        t2 = "Cats are popular household feline animals."
        t3 = "Quantum entanglement governs subatomic particles."

        emb1 = self.model.generate_embedding(t1)
        emb2 = self.model.generate_embedding(t2)
        emb3 = self.model.generate_embedding(t3)

        sim_synonym = cosine_similarity(emb1, emb2)
        sim_divergent = cosine_similarity(emb1, emb3)

        assert sim_synonym > sim_divergent
        assert sim_synonym > 0.6
