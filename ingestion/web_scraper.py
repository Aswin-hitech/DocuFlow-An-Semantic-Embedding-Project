import os
import time
import json
import urllib.parse
from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup

from ingestion.cleaner import clean_html, extract_metadata


class WebScraper:
    """
    Robust web scraper for document ingestion into DocuFlow.
    Extracts text, titles, headings, and metadata from web pages.
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 DocuFlow/1.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }

    def __init__(self, timeout: int = 15, raw_storage_dir: str = "data/raw/webpages"):
        self.timeout = timeout
        self.raw_storage_dir = raw_storage_dir
        if raw_storage_dir:
            os.makedirs(raw_storage_dir, exist_ok=True)

    def scrape_url(self, url: str, save_raw: bool = False) -> Dict[str, Any]:
        """
        Scrape and extract content from a target URL.

        Args:
            url: The webpage URL.
            save_raw: Whether to save raw HTML to data/raw/webpages/.

        Returns:
            Dict containing title, text, headings, url, metadata, and status.
        """
        parsed_url = urllib.parse.urlparse(url)
        if not parsed_url.scheme or parsed_url.scheme not in ["http", "https"]:
            raise ValueError(f"Invalid URL scheme in '{url}'. Must start with http:// or https://")

        try:
            response = requests.get(url, headers=self.DEFAULT_HEADERS, timeout=self.timeout)
            response.raise_for_status()
            html_content = response.text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}")

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        else:
            title = parsed_url.netloc

        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"].strip()

        # Extract headings for hierarchy context
        headings = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            h_text = h.get_text().strip()
            if h_text:
                headings.append({"level": h.name, "text": h_text})

        # Extract cleaned body text
        cleaned_text = clean_html(html_content)

        metadata = {
            "source_type": "webpage",
            "url": url,
            "domain": parsed_url.netloc,
            "title": title,
            "meta_description": meta_desc,
            "scraped_at": time.time(),
            "status_code": response.status_code
        }
        metadata.update(extract_metadata(cleaned_text))

        if save_raw and self.raw_storage_dir:
            safe_filename = "".join([c if c.isalnum() else "_" for c in parsed_url.netloc + parsed_url.path])[:80] + ".json"
            save_path = os.path.join(self.raw_storage_dir, safe_filename)
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({"url": url, "title": title, "text": cleaned_text, "metadata": metadata}, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save raw scraped webpage: {e}")

        return {
            "title": title,
            "text": cleaned_text,
            "headings": headings,
            "url": url,
            "metadata": metadata
        }

    def scrape_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape a batch of URLs sequentially with error handling.
        """
        results = []
        for u in urls:
            try:
                res = self.scrape_url(u)
                results.append(res)
            except Exception as e:
                print(f"Error scraping {u}: {e}")
        return results


if __name__ == "__main__":
    scraper = WebScraper()
    print("WebScraper initialized successfully.")
