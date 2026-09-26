import os
import tempfile
import pytest
from ingestion.cleaner import clean_text, clean_html, extract_metadata, remove_urls
from ingestion.document_parser import DocumentParser


class TestCleaner:
    def test_clean_text_normalizes_spaces_and_quotes(self):
        dirty = "  “Smart quotes”   and    spaces   \n\n\n\nNew line  "
        cleaned = clean_text(dirty)
        assert '"Smart quotes"' in cleaned
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_remove_urls(self):
        text = "Visit https://google.com or www.example.org for more info."
        no_url = remove_urls(text)
        assert "https://" not in no_url
        assert "www." not in no_url

    def test_clean_html(self):
        html = "<html><body><h1>Title</h1><p>DocuFlow body.</p><script>alert('bad');</script></body></html>"
        text = clean_html(html)
        assert "Title" in text
        assert "DocuFlow body." in text
        assert "alert" not in text

    def test_extract_metadata(self):
        sample = "One two three four five six seven eight nine ten."
        meta = extract_metadata(sample)
        assert meta["word_count"] == 10
        assert meta["char_count"] == len(sample)


class TestDocumentParser:
    def test_parse_txt(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
            f.write("Line 1\nLine 2\nLine 3")
            path = f.name

        try:
            res = DocumentParser.parse_file(path)
            assert "Line 1" in res["text"]
            assert res["metadata"]["file_type"] == "txt"
            assert res["metadata"]["filename"] == os.path.basename(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
