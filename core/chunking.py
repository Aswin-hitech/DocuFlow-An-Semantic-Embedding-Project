import re
from typing import List, Dict, Any, Optional


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

    return chunks


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    enriched_chunks = []

    start = 0
    step = chunk_size - overlap

    for i, chunk in enumerate(raw_chunks):
        c_start = i * step
        c_end = min(c_start + chunk_size, len(text))
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
    paragraphs up to max_chunk_size.
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
                # If a single paragraph is too large, split it with chunk_text
                sub_chunks = chunk_text(p, chunk_size=max_chunk_size, overlap=100)
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
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)