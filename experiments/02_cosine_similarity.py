"""
DocuFlow Experiment 02: Cosine Similarity & Angular Relationships
==============================================================
This experiment explores:
1. Mathematical definition: cos(theta) = (A . B) / (||A|| * ||B||)
2. Invariance to vector magnitude (document length invariance)
3. Testing identical, orthogonal, and opposite vectors
4. Converting cosine similarity to geometric angular degrees
5. Why Cosine Similarity is superior to Euclidean Distance for NLP
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.similarity import dot_product, magnitude, cosine_similarity, euclidean_distance


def angle_in_degrees(v1, v2):
    """Calculate angular difference in degrees between two vectors."""
    cos_sim = cosine_similarity(v1, v2)
    # Clamp for floating point edge cases
    clamped = max(-1.0, min(1.0, cos_sim))
    radians = math.acos(clamped)
    return math.degrees(radians)


def run_experiment():
    print("=" * 65)
    print("  DOCUFLOW EXPERIMENT 02: COSINE SIMILARITY & ANGLES")
    print("=" * 65)

    cases = [
        ("Identical Vectors", [2.0, 3.0, 5.0], [2.0, 3.0, 5.0]),
        ("Scaled Identical (Length Invariance)", [1.0, 2.0], [100.0, 200.0]),
        ("Similar Direction (Small Angle)", [1.0, 2.0, 3.0], [1.2, 2.1, 2.9]),
        ("Orthogonal (Independent)", [1.0, 0.0], [0.0, 1.0]),
        ("Opposite Direction", [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]),
    ]

    for label, v_a, v_b in cases:
        sim = cosine_similarity(v_a, v_b)
        ang = angle_in_degrees(v_a, v_b)
        dist = euclidean_distance(v_a, v_b)
        print(f"\nCase: {label}")
        print(f"  Vector A : {v_a}")
        print(f"  Vector B : {v_b}")
        print(f"  Cosine Sim: {sim:+.4f} (Angle: {ang:5.1f}°)")
        print(f"  Euclidean : {dist:.4f}")

    print("\n" + "=" * 65)
    print("KEY TAKEAWAYS FOR NATURAL LANGUAGE PROCESSING:")
    print("  1. Scaled Identical: Euclidean distance is huge (222.49),")
    print("     but Cosine Similarity is exactly 1.0000!")
    print("  2. In document search, a 500-word essay on AI and a 50-word")
    print("     summary have similar directional vectors, but differing magnitudes.")
    print("  3. Cosine similarity correctly matches them based on topic, not length.")
    print("=" * 65)


if __name__ == "__main__":
    run_experiment()
