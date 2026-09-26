"""
Ingestion module for DocuFlow: text cleaner, document parser, and web scraper.
"""

from ingestion.cleaner import clean_text, clean_html, extract_metadata, remove_urls
from ingestion.document_parser import DocumentParser
from ingestion.web_scraper import WebScraper

__all__ = [
    "clean_text",
    "clean_html",
    "extract_metadata",
    "remove_urls",
    "DocumentParser",
    "WebScraper"
]
