import re
import unicodedata
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup


def clean_text(
    text: str,
    remove_extra_spaces: bool = True,
    normalize_unicode: bool = True,
    remove_urls_flag: bool = False,
    lowercase: bool = False
) -> str:
    """
    Clean and normalize raw text for embedding and indexing.

    Args:
        text: Raw input text.
        remove_extra_spaces: Collapse consecutive whitespaces and newlines.
        normalize_unicode: Convert fancy quotes, accents, non-breaking spaces to standard chars.
        remove_urls_flag: Whether to strip web URLs from text.
        lowercase: Whether to convert text to lowercase.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Normalize unicode characters (NFKC handles accents, full-width chars, ligature)
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
        # Normalize common quote and dash variations
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = text.replace("—", " - ").replace("–", " - ").replace("\u00a0", " ")

    # Strip URLs if requested
    if remove_urls_flag:
        text = remove_urls(text)

    # Remove non-printable / control characters (keep \n, \t, \r)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Collapse multiple newlines and spaces
    if remove_extra_spaces:
        # Standardize Windows/Mac newlines to \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces and tabs on a line
        text = re.sub(r"[ \t]+", " ", text)
        # Strip trailing/leading space per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines).strip()

    if lowercase:
        text = text.lower()

    return text.strip()


def remove_urls(text: str) -> str:
    """
    Remove HTTP, HTTPS, and WWW URLs from text.
    """
    if not text:
        return ""
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


def clean_html(html_content: str) -> str:
    """
    Extract readable text from HTML markup, removing scripts, styles, and unwanted tags.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements
    for element in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        element.extract()

    # Get text with line breaks
    text = soup.get_text(separator="\n")
    return clean_text(text)


def extract_metadata(text: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract descriptive statistical metadata from text.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    # Average adult reading speed ~ 200 words per minute
    reading_time_mins = round(word_count / 200.0, 2) if word_count > 0 else 0.0

    metadata = {
        "char_count": char_count,
        "word_count": word_count,
        "estimated_reading_minutes": reading_time_mins,
        "line_count": len(text.splitlines()) if text else 0
    }

    if extra:
        metadata.update(extra)

    return metadata


if __name__ == "__main__":
    raw_sample = """
    <html>
        <body>
            <h1>DocuFlow Overview</h1>
            <p>DocuFlow is a high-performance   semantic embedding pipeline!  Check out https://github.com/docuflow</p>
            <script>alert('malicious')</script>
        </body>
    </html>
    """
    cleaned = clean_html(raw_sample)
    meta = extract_metadata(cleaned)
    print("Cleaned text:\n", repr(cleaned))
    print("\nMetadata:", meta)
