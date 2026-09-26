import pytest
from core.chunking import chunk_text, chunk_text_with_metadata, chunk_by_paragraphs


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
