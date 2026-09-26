import re
from typing import List, Dict, Any, Optional


class BoundaryAwareChunker:
    """
    Intelligent hierarchical boundary-aware text chunker.
    Splits text along semantic boundaries (paragraphs, lines, sentences, clauses, words)
    without truncating words or shredding sentence boundaries.
    """

    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " "
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or list(self.DEFAULT_SEPARATORS)

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of maximum length chunk_size, respecting semantic boundaries.
        """
        if not text:
            return []

        # If text is already within chunk size
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # If text has no whitespace/delimiters at all (e.g. synthetic test string 'abcdefghijklmnopqrstuvwxyz'),
        # use character slicing for exact backward compatibility with unit tests.
        if not re.search(r"\s", text):
            chunks = []
            start = 0
            text_length = len(text)
            while start < text_length:
                end = start + self.chunk_size
                chunk = text[start:end]
                if chunk:
                    chunks.append(chunk)
                start = end - self.overlap
            return chunks

        return self._recursive_split(text, self.separators)

    def _split_pieces(self, text: str, sep: str) -> List[str]:
        """Split text while preserving separators on preceding pieces."""
        if not sep:
            return list(text)

        raw_parts = text.split(sep)
        pieces = []
        for i, part in enumerate(raw_parts):
            if i < len(raw_parts) - 1:
                pieces.append(part + sep)
            else:
                if part:
                    pieces.append(part)
        return pieces

    def _recursive_split(self, text: str, seps: List[str]) -> List[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Find first matching separator present in text
        selected_sep = None
        for s in seps:
            if s in text:
                selected_sep = s
                break

        if selected_sep is None:
            # No separators left. If text has spaces, split on space
            if " " in text:
                return self._recursive_split(text, [" "])
            # Pure unbroken string longer than chunk_size
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                if chunk:
                    chunks.append(chunk)
                start = end - self.overlap
            return chunks

        next_seps = seps[seps.index(selected_sep) + 1:]
        raw_pieces = self._split_pieces(text, selected_sep)

        # Recursively split any piece that alone exceeds chunk_size
        pieces = []
        for p in raw_pieces:
            if len(p) <= self.chunk_size or not next_seps:
                pieces.append(p)
            else:
                sub = self._recursive_split(p, next_seps)
                pieces.extend(sub)

        # Merge pieces into chunks up to chunk_size with boundary-aware overlap
        chunks = []
        current_chunk: List[str] = []
        current_len = 0

        for piece in pieces:
            if not piece.strip() and not current_chunk:
                continue

            piece_len = len(piece)
            if current_len + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_len += piece_len
            else:
                # Flush current_chunk
                if current_chunk:
                    chunk_text = "".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                    # Compute overlap pieces
                    overlap_pieces = []
                    overlap_len = 0
                    if self.overlap > 0:
                        for prev_p in reversed(current_chunk):
                            if overlap_len + len(prev_p) <= self.overlap:
                                overlap_pieces.insert(0, prev_p)
                                overlap_len += len(prev_p)
                            else:
                                break

                    current_chunk = overlap_pieces
                    current_len = overlap_len

                # Add piece
                if current_len + piece_len <= self.chunk_size:
                    current_chunk.append(piece)
                    current_len += piece_len
                else:
                    if current_chunk:
                        chunk_text = "".join(current_chunk).strip()
                        if chunk_text:
                            chunks.append(chunk_text)
                    current_chunk = [piece]
                    current_len = piece_len

        if current_chunk:
            chunk_text = "".join(current_chunk).strip()
            if chunk_text and (not chunks or chunks[-1] != chunk_text):
                chunks.append(chunk_text)

        return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into boundary-aware chunks of maximum size `chunk_size`
    with `overlap` character buffer, preserving complete words and sentences.
    """
    chunker = BoundaryAwareChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_text(text)


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate boundary-aware text chunks enriched with detailed metadata
    including precise character offsets, chunk index, and document info.
    """
    raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    enriched_chunks = []

    cursor = 0
    for i, chunk in enumerate(raw_chunks):
        # Locate chunk span within source text
        c_start = text.find(chunk, cursor)
        if c_start == -1:
            # Fallback search if stripped
            probe = chunk[:min(30, len(chunk))]
            c_start = text.find(probe, cursor)
            if c_start == -1:
                c_start = text.find(probe)
                if c_start == -1:
                    c_start = cursor

        c_end = min(len(text), c_start + len(chunk))
        cursor = max(cursor, c_start + 1)

        meta = {
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "char_start": c_start,
            "char_end": c_end,
            "char_count": len(chunk),
            "doc_id": doc_id or "default_doc",
            "source": source or "direct_input"
        }
        if extra_metadata:
            meta.update(extra_metadata)

        enriched_chunks.append({
            "chunk_id": f"{doc_id or 'doc'}_chunk_{i}",
            "text": chunk,
            "metadata": meta
        })

    return enriched_chunks


def chunk_by_paragraphs(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Split text by natural paragraph boundaries (double newlines), grouping small
    paragraphs up to max_chunk_size. Large paragraphs are split along sentence/word boundaries.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for p in paragraphs:
        p_len = len(p)
        if current_length + p_len + 2 <= max_chunk_size:
            current_chunk.append(p)
            current_length += p_len + 2
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            if p_len > max_chunk_size:
                # If a single paragraph is too large, split it with boundary-aware chunk_text
                sub_chunks = chunk_text(p, chunk_size=max_chunk_size, overlap=min(100, max_chunk_size // 5))
                chunks.extend(sub_chunks)
                current_chunk = []
                current_length = 0
            else:
                current_chunk = [p]
                current_length = p_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


if __name__ == "__main__":
    sample_text = """
    Machine learning is a branch of artificial intelligence.
    It allows computers to learn patterns from data without
    being explicitly programmed for every task.

    There are several types of machine learning.
    Supervised learning uses labelled data for training.
    Unsupervised learning works with unlabeled data.
    Reinforcement learning learns through rewards and penalties.

    Neural networks are an important part of modern machine
    learning and are widely used in computer vision,
    natural language processing and speech recognition.
    """

    chunks = chunk_text(
        sample_text,
        chunk_size=200,
        overlap=40
    )

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ({len(chunk)} chars) ---")
        print(chunk)