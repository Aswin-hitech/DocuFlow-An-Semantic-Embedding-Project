"""
DocuFlow Experiment 01: Mathematical Vectors & Representations
============================================================
This experiment demonstrates fundamental vector concepts:
1. Vector representation as multi-dimensional coordinates
2. Vector addition, subtraction, and scalar multiplication
3. Vector magnitude (Euclidean / L2 norm)
4. Unit vector normalization
5. Geometric intuition behind dense vector embeddings
"""

import math
import sys
import os

# Ensure project root is accessible
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.similarity import dot_product, magnitude, euclidean_distance


def vector_add(v1, v2):
    """Component-wise addition of two vectors."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension.")
    return [a + b for a, b in zip(v1, v2)]


def vector_scale(v, scalar):
    """Multiply a vector by a scalar value."""
    return [x * scalar for x in v]


def normalize_vector(v):
    """Transform vector into a unit vector (norm = 1.0)."""
    mag = magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return [x / mag for x in v]


def run_experiment():
    print("=" * 60)
    print("  DOCUFLOW EXPERIMENT 01: VECTORS & GEOMETRIC OPERATIONS")
    print("=" * 60)

    # 1. 2D & 3D Vectors
    v1 = [3.0, 4.0]
    v2 = [1.0, 2.0]
    print(f"\n[1] Vector Definitions:")
    print(f"    v1 = {v1}")
    print(f"    v2 = {v2}")

    # 2. Vector Addition
    v_sum = vector_add(v1, v2)
    print(f"\n[2] Vector Addition (v1 + v2):")
    print(f"    {v1} + {v2} = {v_sum}")

    # 3. Scalar Multiplication
    v_scaled = vector_scale(v1, 2.5)
    print(f"\n[3] Scalar Multiplication (v1 * 2.5):")
    print(f"    {v1} * 2.5 = {v_scaled}")

    # 4. Magnitude (L2 Norm)
    mag_v1 = magnitude(v1)
    print(f"\n[4] Vector Magnitude (L2 Norm):")
    print(f"    ||v1|| = sqrt(3^2 + 4^2) = {mag_v1:.4f}")

    # 5. Normalization
    v1_unit = normalize_vector(v1)
    mag_unit = magnitude(v1_unit)
    print(f"\n[5] Unit Vector Normalization:")
    print(f"    Normalized v1 = {v1_unit}")
    print(f"    Length of unit vector = {mag_unit:.6f}")

    # 6. Euclidean Distance
    dist = euclidean_distance(v1, v2)
    print(f"\n[6] Euclidean Distance between v1 and v2:")
    print(f"    dist(v1, v2) = {dist:.4f}")

    # 7. High-dimensional Embeddings Concept
    print(f"\n[7] High-Dimensional Vectors in DocuFlow:")
    print("    - MiniLM embeddings have 384 dimensions.")
    print("    - Each dimension captures subtle semantic features.")
    print("    - In high dimensions, angle (cosine) is far more meaningful than absolute distance.")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
