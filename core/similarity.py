import math
from typing import List, Sequence, Union


def dot_product(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """
    Compute the dot product of two vectors of equal dimension.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension.")

    result = 0.0
    for a, b in zip(vector_a, vector_b):
        result += float(a) * float(b)

    return result


def magnitude(vector: Sequence[float]) -> float:
    """
    Compute the Euclidean magnitude (L2 norm) of a vector.
    """
    result = 0.0
    for value in vector:
        result += float(value) * float(value)

    return math.sqrt(result)


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """
    Compute the cosine similarity between two vectors:
    cos(theta) = (A . B) / (||A|| * ||B||)
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension.")

    magnitude_a = magnitude(vector_a)
    magnitude_b = magnitude(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    dot = dot_product(vector_a, vector_b)
    # Clamp due to floating point precision issues
    score = dot / (magnitude_a * magnitude_b)
    return max(-1.0, min(1.0, score))


def euclidean_distance(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """
    Compute the Euclidean distance (L2 distance) between two vectors.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension.")

    squared_diff_sum = sum((float(a) - float(b)) ** 2 for a, b in zip(vector_a, vector_b))
    return math.sqrt(squared_diff_sum)


def manhattan_distance(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """
    Compute the Manhattan distance (L1 distance) between two vectors.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension.")

    return sum(abs(float(a) - float(b)) for a, b in zip(vector_a, vector_b))


if __name__ == "__main__":
    vector_a = [1, 2, 3]
    vector_b = [2, 4, 6]

    similarity = cosine_similarity(vector_a, vector_b)
    dist = euclidean_distance(vector_a, vector_b)

    print("Vector A:", vector_a)
    print("Vector B:", vector_b)
    print("Cosine Similarity:", similarity)
    print("Euclidean Distance:", dist)