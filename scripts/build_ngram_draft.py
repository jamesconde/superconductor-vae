#!/usr/bin/env python3
"""
Build N-gram + Structural Hybrid Draft Model from Training Data.

This script builds the draft model used for speculative decoding during
REINFORCE training. The draft model predicts likely next tokens to speed
up autoregressive generation.

Usage:
    python scripts/build_ngram_draft.py

Output:
    data/processed/draft_model.pkl (+ .ngram.pkl, .struct.pkl)

The draft model is automatically loaded by train_v12_clean.py when available.
"""

import sys
import json
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from superconductor.models.ngram_draft import (
    HybridDraft,
    build_draft_model,
    load_or_build_draft_model,
)


def load_formulas(data_path: Path, holdout_path: Path) -> list:
    """
    Load training formulas (excluding holdout set).

    Args:
        data_path: Path to CSV with formulas
        holdout_path: Path to holdout JSON

    Returns:
        List of formula strings
    """
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} total samples from {data_path.name}")

    # Load holdout formulas
    holdout_formulas = set()
    if holdout_path.exists():
        with open(holdout_path, 'r') as f:
            data = json.load(f)
        holdout_formulas = {s['formula'] for s in data['holdout_samples']}
        print(f"Loaded {len(holdout_formulas)} holdout formulas to exclude")

    # Filter to training formulas only
    formulas = [f for f in df['formula'].tolist() if f not in holdout_formulas]
    print(f"Training formulas: {len(formulas)}")

    return formulas


def analyze_formulas(formulas: list):
    """Print analysis of formula patterns."""
    import re
    from collections import Counter

    print("\n" + "=" * 60)
    print("Formula Pattern Analysis")
    print("=" * 60)

    # Count denominators
    all_denoms = []
    for formula in formulas:
        denoms = re.findall(r'/(\d+)', formula)
        all_denoms.extend(denoms)

    denom_counts = Counter(all_denoms)
    print(f"\nUnique denominators: {len(denom_counts)}")
    print("Top 10 denominators:")
    for denom, count in denom_counts.most_common(10):
        pct = 100 * count / len(all_denoms)
        print(f"  /{denom}: {count:,} ({pct:.1f}%)")

    # Count elements
    from superconductor.models.autoregressive_decoder import tokenize_formula, ELEMENTS

    element_counts = Counter()
    for formula in formulas:
        tokens = tokenize_formula(formula)
        for tok in tokens:
            if tok in ELEMENTS[1:]:
                element_counts[tok] += 1

    print(f"\nUnique elements: {len(element_counts)}")
    print("Top 10 elements:")
    for elem, count in element_counts.most_common(10):
        pct = 100 * count / sum(element_counts.values())
        print(f"  {elem}: {count:,} ({pct:.1f}%)")

    # Token sequence length distribution
    lengths = []
    for formula in formulas:
        tokens = tokenize_formula(formula)
        lengths.append(len(tokens))

    print(f"\nSequence lengths:")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.1f}")


def test_draft_model(model: HybridDraft, formulas: list, n_tests: int = 10):
    """Test draft model predictions on sample formulas."""
    import random
    from superconductor.models.autoregressive_decoder import (
        tokenize_formula, TOKEN_TO_IDX, IDX_TO_TOKEN, START_IDX, END_IDX
    )

    print("\n" + "=" * 60)
    print("Draft Model Test")
    print("=" * 60)

    random.seed(42)
    test_formulas = random.sample(formulas, min(n_tests, len(formulas)))

    for formula in test_formulas:
        tokens = tokenize_formula(formula)
        indices = [START_IDX] + [TOKEN_TO_IDX.get(t, 0) for t in tokens] + [END_IDX]

        print(f"\nFormula: {formula}")
        print(f"  Tokens: {len(indices)}")

        # Test draft at different positions
        for pos in [1, 3, 5, 10]:
            if pos >= len(indices):
                break

            context = indices[:pos]
            drafted = model.draft_k_tokens(context, k=5)
            drafted_str = [IDX_TO_TOKEN.get(d, '?') for d in drafted]

            # Check how many match actual sequence
            actual = indices[pos:pos+5] if pos + 5 <= len(indices) else indices[pos:]
            n_match = sum(1 for d, a in zip(drafted, actual) if d == a)

            print(f"  @pos {pos}: draft={drafted_str}, match={n_match}/5")


def main():
    # Paths
    data_path = PROJECT_ROOT / 'data/processed/supercon_fractions_combined.csv'
    holdout_path = PROJECT_ROOT / 'data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json'
    output_path = PROJECT_ROOT / 'data/processed/draft_model.pkl'

    # Check for contrastive data (larger dataset)
    contrastive_path = PROJECT_ROOT / 'data/processed/supercon_fractions_contrastive.csv'
    if contrastive_path.exists():
        print(f"Using contrastive dataset (larger): {contrastive_path.name}")
        data_path = contrastive_path

    # Load formulas
    formulas = load_formulas(data_path, holdout_path)

    # Analyze patterns
    analyze_formulas(formulas)

    # Build draft model
    print("\n" + "=" * 60)
    print("Building Draft Model")
    print("=" * 60)

    model = build_draft_model(
        formulas=formulas,
        max_len=60,
        save_path=output_path,
    )

    # Test model
    test_draft_model(model, formulas)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Draft model saved to: {output_path}")
    print(f"Associated files:")
    print(f"  - {output_path.with_suffix('.ngram.pkl')}")
    print(f"  - {output_path.with_suffix('.struct.pkl')}")


if __name__ == '__main__':
    main()
