import math
import pytest
from core.similarity import (
    dot_product,
    magnitude,
    cosine_similarity,
    euclidean_distance,
    manhattan_distance
)


class TestDotProduct:
    def test_standard_dot_product(self):
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert dot_product(v1, v2) == pytest.approx(32.0)

    def test_orthogonal_dot_product(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert dot_product(v1, v2) == pytest.approx(0.0)

    def test_dimension_mismatch_raises(self):
        v1 = [1.0, 2.0]
        v2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="same dimension"):
            dot_product(v1, v2)


class TestMagnitude:
    def test_known_pythagorean_triple(self):
        # 3, 4, 5
        v = [3.0, 4.0]
        assert magnitude(v) == pytest.approx(5.0)

    def test_zero_vector(self):
        v = [0.0, 0.0, 0.0]
        assert magnitude(v) == pytest.approx(0.0)

    def test_negative_elements(self):
        v = [-3.0, -4.0]
        assert magnitude(v) == pytest.approx(5.0)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_scaled_vectors_length_invariance(self):
        v1 = [1.0, 2.0]
        v2 = [50.0, 100.0]
        assert cosine_similarity(v1, v2) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        v1 = [2.0, 3.0]
        v2 = [-2.0, -3.0]
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector_raises(self):
        v1 = [1.0, 2.0]
        v2 = [0.0, 0.0]
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity(v1, v2)

    def test_dimension_mismatch_raises(self):
        v1 = [1.0, 2.0]
        v2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="same dimension"):
            cosine_similarity(v1, v2)


class TestDistanceMetrics:
    def test_euclidean_distance(self):
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        assert euclidean_distance(v1, v2) == pytest.approx(5.0)

    def test_manhattan_distance(self):
        v1 = [1.0, 2.0]
        v2 = [4.0, 6.0]
        # |1-4| + |2-6| = 3 + 4 = 7
        assert manhattan_distance(v1, v2) == pytest.approx(7.0)
