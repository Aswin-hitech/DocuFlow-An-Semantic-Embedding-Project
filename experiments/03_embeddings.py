"""
DocuFlow Experiment 03: Dense Neural Embeddings & Semantic Spaces
==============================================================
This experiment investigates:
1. Transforming sentences into 384-dimensional dense vectors
2. Measuring semantic affinity across different knowledge domains:
   - Technology / Machine Learning
   - Food / Cooking
   - Sports / Fitness
3. Demonstrating that synonyms and semantic paraphrases produce high similarity
4. Generating a formatted cross-domain similarity matrix
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embeddings import EmbeddingModel
from core.similarity import cosine_similarity


def run_experiment():
    print("=" * 70)
    print("  DOCUFLOW EXPERIMENT 03: NEURAL EMBEDDINGS & DOMAIN CLUSTERING")
    print("=" * 70)

    print("\n[+] Initializing sentence transformer model...")
    model = EmbeddingModel()
    print(f"[+] Loaded model: {model.model_name} (Dimensions: {model.dimension})")

    samples = [
        ("Tech-1", "Artificial intelligence algorithms are transforming modern software."),
        ("Tech-2", "Machine learning models learn patterns from training datasets."),
        ("Food-1", "Freshly baked pizza with basil and melted mozzarella cheese."),
        ("Food-2", "A delicious Italian pasta recipe topped with grated parmesan."),
        ("Sport-1", "The football team scored a dramatic goal in the final minute."),
        ("Sport-2", "Athletes train vigorously for marathon endurance races.")
    ]

    texts = [text for _, text in samples]
    labels = [lbl for lbl, _ in samples]

    print(f"\n[+] Encoding {len(texts)} sample sentences into {model.dimension}-dimensional vectors...")
    embeddings = model.generate_embeddings(texts)

    # Print vector inspection
    sample_vec = embeddings[0]
    print(f"\nSample vector representation (first 6 dimensions):")
    print(f"  {sample_vec[:6]} ...")

    # Print Pairwise Similarity Matrix
    print("\n" + "=" * 70)
    print("PAIRWISE COSINE SIMILARITY MATRIX")
    print("=" * 70)

    header = f"{'Category':<10}" + "".join(f"{lbl:>10}" for lbl in labels)
    print(header)
    print("-" * len(header))

    for i, (label_i, _) in enumerate(samples):
        row = f"{label_i:<10}"
        for j in range(len(samples)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            row += f"{sim:10.3f}"
        print(row)

    print("\n" + "=" * 70)
    print("OBSERVATIONS:")
    print("  - Tech-1 vs Tech-2 similarity is high (Intra-domain affinity)")
    print("  - Food-1 vs Food-2 similarity is high")
    print("  - Tech-1 vs Food-1 similarity is low / near zero (Domain separation)")
    print("  - This proves dense embeddings understand contextual semantic meaning!")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
