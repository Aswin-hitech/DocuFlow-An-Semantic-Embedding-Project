import pytest
from core.chunking import chunk_text, chunk_text_with_metadata, chunk_by_paragraphs, BoundaryAwareChunker


class TestChunkText:
    def test_basic_chunking(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        # Expected:
        # 1: text[0:10] = "abcdefghij", next start = 10 - 2 = 8
        # 2: text[8:18] = "ijklmnopqr", next start = 18 - 2 = 16
        # 3: text[16:26] = "qrstuvwxyz", next start = 26 - 2 = 24
        # 4: text[24:34] = "yz"
        assert len(chunks) == 4
        assert chunks[0] == "abcdefghij"
        assert chunks[1] == "ijklmnopqr"
        assert chunks[2] == "qrstuvwxyz"
        assert chunks[3] == "yz"

    def test_empty_text(self):
        assert chunk_text("", chunk_size=100, overlap=10) == []

    def test_chunk_size_larger_than_text(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            chunk_text("Sample", chunk_size=0, overlap=0)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="overlap cannot be negative"):
            chunk_text("Sample", chunk_size=50, overlap=-5)

    def test_overlap_greater_or_equal_chunk_size(self):
        with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
            chunk_text("Sample", chunk_size=50, overlap=50)

    def test_boundary_aware_never_cuts_words(self):
        text = "The umpire considered that the bowler derived an unfair advantage during the delivery."
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        assert len(chunks) > 1
        for chunk in chunks:
            # Verify words are not cut into non-words like 'advan' or 'bowle'
            words = chunk.split()
            for w in words:
                # Every word in the chunk must exist as an intact word in the source text
                clean_w = w.strip(".,;:!?")
                assert clean_w in text.split() or clean_w in [t.strip(".,;:!?") for t in text.split()]

    def test_sentence_boundary_preservation(self):
        text = "Machine learning is powerful. Deep learning extends neural networks. RAG retrieval grounds facts."
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) >= 2
        # Check that sentences remain intact and end with proper punctuation
        for chunk in chunks:
            assert chunk.endswith(".")


class TestChunkTextWithMetadata:
    def test_metadata_structure(self):
        text = "First block of text. Second block of text. Third block of text."
        enriched = chunk_text_with_metadata(
            text,
            chunk_size=30,
            overlap=5,
            doc_id="doc_1",
            source="test_source.txt"
        )
        assert len(enriched) > 0
        for i, item in enumerate(enriched):
            assert "chunk_id" in item
            assert "text" in item
            assert "metadata" in item
            assert item["metadata"]["doc_id"] == "doc_1"
            assert item["metadata"]["source"] == "test_source.txt"
            assert item["metadata"]["chunk_index"] == i
            assert item["metadata"]["total_chunks"] == len(enriched)

    def test_exact_character_offsets(self):
        text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india."
        enriched = chunk_text_with_metadata(
            text,
            chunk_size=30,
            overlap=5,
            doc_id="doc_2",
            source="test_offsets.txt"
        )
        for item in enriched:
            s = item["metadata"]["char_start"]
            e = item["metadata"]["char_end"]
            expected_text = item["text"]
            assert text[s:e] == expected_text


class TestChunkByParagraphs:
    def test_split_by_double_newlines(self):
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = chunk_by_paragraphs(text, max_chunk_size=15)
        assert len(chunks) == 3
        assert "Paragraph 1." in chunks[0]
        assert "Paragraph 2." in chunks[1]
        assert "Paragraph 3." in chunks[2]

    def test_combine_short_paragraphs(self):
        text = "Short 1.\n\nShort 2."
        chunks = chunk_by_paragraphs(text, max_chunk_size=500)
        assert len(chunks) == 1
        assert "Short 1.\n\nShort 2." == chunks[0]
