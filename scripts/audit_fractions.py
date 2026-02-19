#!/usr/bin/env python3
"""
Fraction Vocabulary Audit Script (V13.0 Preparation)

Scans the training CSV to extract all unique fractions, compute coverage statistics,
and determine whether a closed fraction vocabulary is feasible.

Usage:
    python scripts/audit_fractions.py

Output:
    Prints statistics to stdout and saves detailed report to scratch/fraction_audit_report.txt
"""

import re
import math
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_contrastive.csv"
REPORT_PATH = PROJECT_ROOT / "scratch" / "fraction_audit_report.txt"


def canonicalize_fraction(num: int, den: int) -> tuple:
    """Reduce a fraction to lowest terms via GCD."""
    g = math.gcd(num, den)
    return (num // g, den // g)


def extract_fractions_from_formula(formula: str) -> list:
    """Extract all (numerator, denominator) pairs from a formula string."""
    pattern = r'\((\d+)/(\d+)\)'
    return [(int(m[0]), int(m[1])) for m in re.findall(pattern, formula)]


def extract_integers_from_formula(formula: str) -> list:
    """Extract standalone integer subscripts from a formula (not inside fractions)."""
    # Match element followed by integer (not inside parentheses)
    # Pattern: element symbol followed by digits, but not preceded by / or (
    pattern = r'([A-Z][a-z]?)(\d+)(?![/)])'
    return [int(m[1]) for m in re.findall(pattern, formula)]


def main():
    import pandas as pd

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    formulas = df['formula'].tolist()
    print(f"  Total formulas: {len(formulas)}")

    # =========================================================================
    # Phase 1: Raw fraction extraction (before canonicalization)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Raw Fraction Analysis (before canonicalization)")
    print("="*70)

    raw_fractions = Counter()
    formulas_with_fractions = 0
    total_fraction_occurrences = 0
    max_num = 0
    max_den = 0
    fractions_per_formula = []

    for formula in formulas:
        fracs = extract_fractions_from_formula(formula)
        if fracs:
            formulas_with_fractions += 1
            fractions_per_formula.append(len(fracs))
            for num, den in fracs:
                raw_fractions[(num, den)] += 1
                total_fraction_occurrences += 1
                max_num = max(max_num, num)
                max_den = max(max_den, den)
        else:
            fractions_per_formula.append(0)

    print(f"\n  Formulas with fractions: {formulas_with_fractions}/{len(formulas)} "
          f"({100*formulas_with_fractions/len(formulas):.1f}%)")
    print(f"  Unique raw fractions: {len(raw_fractions)}")
    print(f"  Total fraction occurrences: {total_fraction_occurrences}")
    print(f"  Max numerator: {max_num}")
    print(f"  Max denominator: {max_den}")

    if fractions_per_formula:
        has_fracs = [f for f in fractions_per_formula if f > 0]
        if has_fracs:
            avg_fracs = sum(has_fracs) / len(has_fracs)
            print(f"  Mean fractions per formula (where >0): {avg_fracs:.2f}")
            print(f"  Max fractions per formula: {max(fractions_per_formula)}")

    # =========================================================================
    # Phase 2: Canonicalized fraction analysis
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: Canonicalized Fraction Analysis (GCD-reduced)")
    print("="*70)

    canon_fractions = Counter()
    for (num, den), count in raw_fractions.items():
        canon = canonicalize_fraction(num, den)
        canon_fractions[canon] += count

    n_reduced = len(raw_fractions) - len(canon_fractions)
    print(f"\n  Unique canonicalized fractions: {len(canon_fractions)}")
    print(f"  Fractions eliminated by GCD reduction: {n_reduced}")

    # Coverage analysis
    sorted_fracs = canon_fractions.most_common()
    cumulative = 0
    coverage_thresholds = [50, 90, 95, 99, 99.9, 100]
    threshold_idx = 0

    print(f"\n  Coverage by top-N fractions:")
    for i, ((num, den), count) in enumerate(sorted_fracs):
        cumulative += count
        coverage = 100 * cumulative / total_fraction_occurrences
        while threshold_idx < len(coverage_thresholds) and coverage >= coverage_thresholds[threshold_idx]:
            print(f"    Top {i+1:4d} fractions → {coverage_thresholds[threshold_idx]:5.1f}% coverage")
            threshold_idx += 1

    # Top 30 most common fractions
    print(f"\n  Top 30 most common fractions:")
    for i, ((num, den), count) in enumerate(sorted_fracs[:30]):
        pct = 100 * count / total_fraction_occurrences
        print(f"    {i+1:3d}. ({num}/{den}) = {num/den:.6f}  count={count:5d}  ({pct:.1f}%)")

    # =========================================================================
    # Phase 3: Integer subscript analysis
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 3: Integer Subscript Analysis")
    print("="*70)

    integer_counts = Counter()
    for formula in formulas:
        ints = extract_integers_from_formula(formula)
        for val in ints:
            integer_counts[val] += 1

    print(f"\n  Unique integer subscripts: {len(integer_counts)}")
    print(f"  Max integer: {max(integer_counts.keys()) if integer_counts else 0}")
    sorted_ints = integer_counts.most_common()
    print(f"\n  Integer subscript distribution:")
    for val, count in sorted_ints[:25]:
        print(f"    {val:5d}: count={count:5d}")

    ints_over_20 = {v: c for v, c in integer_counts.items() if v > 20}
    if ints_over_20:
        total_over_20 = sum(ints_over_20.values())
        total_all_ints = sum(integer_counts.values())
        print(f"\n  Integers > 20: {len(ints_over_20)} unique, "
              f"{total_over_20} occurrences ({100*total_over_20/total_all_ints:.1f}% of all integer subscripts)")
    else:
        print(f"\n  No integers > 20 found")

    # =========================================================================
    # Phase 4: Sequence length reduction estimate
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 4: Sequence Length Reduction Estimate")
    print("="*70)

    # Import existing tokenizer for comparison
    from superconductor.models.autoregressive_decoder import tokenize_formula

    old_lengths = []
    new_lengths = []
    for formula in formulas:
        old_tokens = tokenize_formula(formula)
        old_lengths.append(len(old_tokens) + 2)  # +2 for BOS/EOS

        # New estimate: each fraction(n/d) becomes 1 token instead of 2+len(n)+1+len(d)+2
        fracs = extract_fractions_from_formula(formula)
        old_frac_tokens = 0
        for num, den in fracs:
            # Old: ( + digits(num) + / + digits(den) + ) = 3 + len(str(num)) + len(str(den))
            old_frac_tokens += 3 + len(str(num)) + len(str(den))
        new_frac_tokens = len(fracs)  # Each fraction = 1 token
        saved = old_frac_tokens - new_frac_tokens
        new_lengths.append(len(old_tokens) + 2 - saved)

    import numpy as np
    old_arr = np.array(old_lengths)
    new_arr = np.array(new_lengths)
    reduction = (old_arr - new_arr) / old_arr * 100

    print(f"\n  Old mean sequence length: {old_arr.mean():.1f} tokens")
    print(f"  New mean sequence length: {new_arr.mean():.1f} tokens")
    print(f"  Mean reduction: {reduction.mean():.1f}%")
    print(f"  Median reduction: {np.median(reduction):.1f}%")
    print(f"  Max old length: {old_arr.max()}")
    print(f"  Max new length: {new_arr.max()}")

    # =========================================================================
    # Phase 5: Decision gate
    # =========================================================================
    print("\n" + "="*70)
    print("DECISION GATE")
    print("="*70)
    n_unique = len(canon_fractions)
    if n_unique < 2000:
        print(f"\n  ✓ {n_unique} unique fractions < 2000 threshold")
        print(f"  → Proceed with CLOSED VOCABULARY (Option A)")
        print(f"  → Total vocab size: 5 (special) + 118 (elements) + 20 (integers) + {n_unique} (fractions) = {5 + 118 + 20 + n_unique}")
    else:
        print(f"\n  ✗ {n_unique} unique fractions ≥ 2000 threshold")
        print(f"  → Consider decomposition tokens (Appendix A)")

    # =========================================================================
    # Save detailed report
    # =========================================================================
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write("Fraction Vocabulary Audit Report\n")
        f.write(f"Dataset: {DATA_PATH}\n")
        f.write(f"Total formulas: {len(formulas)}\n")
        f.write(f"Unique raw fractions: {len(raw_fractions)}\n")
        f.write(f"Unique canonicalized fractions: {len(canon_fractions)}\n\n")

        f.write("All canonicalized fractions (sorted by frequency):\n")
        for (num, den), count in sorted_fracs:
            f.write(f"  ({num}/{den}) = {num/den:.8f}  count={count}\n")

        f.write(f"\nAll integer subscripts (sorted by frequency):\n")
        for val, count in sorted_ints:
            f.write(f"  {val}: count={count}\n")

    print(f"\n  Detailed report saved to: {REPORT_PATH}")


if __name__ == '__main__':
    main()
