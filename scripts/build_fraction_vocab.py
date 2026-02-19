#!/usr/bin/env python3
"""
Build Fraction Vocabulary for V13.0 Semantic Fraction Tokenization.

Scans the training CSV, extracts all unique (canonicalized) fractions,
and builds a definitive vocabulary file at data/fraction_vocab.json.

Usage:
    python scripts/build_fraction_vocab.py

Output:
    data/fraction_vocab.json — The fraction vocabulary used by FractionAwareTokenizer
"""

import re
import json
import math
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_contrastive.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "fraction_vocab.json"

# Physics-important fractions that MUST be in vocabulary even if not in training data
# (from holdout set and common superconductor stoichiometries)
PHYSICS_IMPORTANT_FRACTIONS = [
    (1, 2), (1, 3), (2, 3), (1, 4), (3, 4),
    (1, 5), (2, 5), (3, 5), (4, 5),
    (1, 10), (3, 10), (7, 10), (9, 10),
    (1, 20), (3, 20), (17, 20), (19, 20),
    (9, 5), (1, 50), (1, 100),
    (1, 500), (499, 500),  # Trace doping
    (137, 20),  # YBCO oxygen: O(137/20) = O6.85
]


def canonicalize_fraction(num: int, den: int) -> tuple:
    """Reduce a fraction to lowest terms via GCD."""
    g = math.gcd(num, den)
    return (num // g, den // g)


def extract_fractions_from_formula(formula: str) -> list:
    """Extract all (numerator, denominator) pairs from a formula string."""
    pattern = r'\((\d+)/(\d+)\)'
    return [(int(m[0]), int(m[1])) for m in re.findall(pattern, formula)]


def main():
    import pandas as pd

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    formulas = df['formula'].tolist()
    print(f"  Total formulas: {len(formulas)}")

    # Extract and canonicalize all fractions
    print("Extracting fractions...")
    fraction_counts = Counter()
    for formula in formulas:
        fracs = extract_fractions_from_formula(formula)
        for num, den in fracs:
            canon = canonicalize_fraction(num, den)
            fraction_counts[canon] += 1

    print(f"  Unique canonicalized fractions from data: {len(fraction_counts)}")

    # Add physics-important fractions (with count 0 if not seen)
    added = 0
    for frac in PHYSICS_IMPORTANT_FRACTIONS:
        canon = canonicalize_fraction(*frac)
        if canon not in fraction_counts:
            fraction_counts[canon] = 0
            added += 1
    if added:
        print(f"  Added {added} physics-important fractions not in training data")

    # Sort by frequency (descending), then by (den, num) for stability
    sorted_fracs = sorted(
        fraction_counts.items(),
        key=lambda x: (-x[1], x[0][1], x[0][0])
    )

    # Build the vocabulary
    fractions_list = []
    fraction_to_id = {}
    for i, ((num, den), count) in enumerate(sorted_fracs):
        frac_str = f"{num}/{den}"
        fractions_list.append(frac_str)
        fraction_to_id[frac_str] = i

    # Compute coverage stats
    total_occurrences = sum(fraction_counts.values())
    cumulative = 0
    coverage_at = {}
    for i, ((num, den), count) in enumerate(sorted_fracs):
        cumulative += count
        pct = 100 * cumulative / total_occurrences if total_occurrences > 0 else 0
        for threshold in [50, 90, 95, 99, 100]:
            if threshold not in coverage_at and pct >= threshold:
                coverage_at[threshold] = i + 1

    # Build output
    vocab = {
        "version": "V13.0",
        "description": "Semantic fraction vocabulary for V13.0 tokenizer",
        "source_dataset": str(DATA_PATH.name),
        "n_formulas": len(formulas),
        "n_fractions": len(fractions_list),
        "total_fraction_occurrences": total_occurrences,
        "coverage": {
            f"top_{k}pct": v for k, v in sorted(coverage_at.items())
        },
        "max_numerator": max(num for (num, den) in fraction_counts.keys()),
        "max_denominator": max(den for (num, den) in fraction_counts.keys()),
        "fractions": fractions_list,
        "fraction_to_id": fraction_to_id,
        "fraction_counts": {f"{num}/{den}": count for (num, den), count in sorted_fracs},
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"\nFraction vocabulary saved to {OUTPUT_PATH}")
    print(f"  Total fractions: {len(fractions_list)}")
    print(f"  Coverage: {coverage_at}")
    print(f"  Estimated total vocab size: 5 (special) + 118 (elements) + 20 (integers) + {len(fractions_list)} (fractions) = {5 + 118 + 20 + len(fractions_list)}")


if __name__ == '__main__':
    main()
