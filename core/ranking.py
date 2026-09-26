from typing import List, Dict, Any, Optional


def rank_results(
    results: List[Dict[str, Any]],
    top_k: int = 3,
    min_score: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Rank search results based on similarity score.

    Args:
        results: List of dictionaries containing:
                 - chunk (or text)
                 - score
        top_k: Number of results to return.
        min_score: Optional minimum score threshold to filter out low-relevance results.

    Returns:
        Top-k results sorted by descending similarity.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if not results:
        return []

    # Filter by minimum score if specified
    filtered = results
    if min_score is not None:
        filtered = [r for r in results if r.get("score", 0.0) >= min_score]

    sorted_results = sorted(
        filtered,
        key=lambda result: result["score"],
        reverse=True
    )

    return sorted_results[:top_k]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Combine multiple ranked result lists (e.g. dense semantic search + BM25 keyword search)
    using Reciprocal Rank Fusion (RRF).

    Score formula: RRF_score(d) = sum(1 / (k + rank(d)))
    """
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("chunk") or item.get("text") or str(item)
            if key not in doc_lookup:
                doc_lookup[key] = item
            scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))

    fused_results = []
    for key, rrf_score in scores.items():
        entry = dict(doc_lookup[key])
        entry["rrf_score"] = rrf_score
        fused_results.append(entry)

    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_results[:top_k]


if __name__ == "__main__":
    results = [
        {
            "chunk": "Machine learning learns patterns from data.",
            "score": 0.72
        },
        {
            "chunk": "Neural networks are used in deep learning.",
            "score": 0.91
        },
        {
            "chunk": "Python is a programming language.",
            "score": 0.43
        },
        {
            "chunk": "Supervised learning uses labelled data.",
            "score": 0.84
        }
    ]

    top_results = rank_results(results, top_k=3)

    for i, result in enumerate(top_results, start=1):
        print(f"\nRank {i}")
        print("Score:", result["score"])
        print("Chunk:", result["chunk"])