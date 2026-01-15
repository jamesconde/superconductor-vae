"""
Stoichiometry-Aware Fraction Corrector for Chemical Formulas.

Key Insights from Data Analysis:
1. Only 28 unique denominators exist in the dataset
2. Fractions on the same "site" sum to nice numbers (1, 2, 3, etc.)
3. Denominators cluster: /5, /10, /20, /100 cover 60%

Correction Strategies:
1. DENOMINATOR SNAPPING: Rare denominators -> nearest common one
2. STOICHIOMETRY CONSTRAINT: Adjust numerators so fractions sum to target
3. DENOMINATOR CONSISTENCY: If one frac uses /1000, others likely do too
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter

# Common denominators in order of frequency
COMMON_DENOMINATORS = [5, 10, 20, 100, 2, 50, 25, 4, 1000, 200, 500, 40, 250, 125, 8]

# Valid denominators (all 28 found in data)
VALID_DENOMINATORS = {
    2, 4, 5, 8, 10, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500,
    1000, 2000, 5000, 10000, 2500, 625, 3125  # Adding some derived ones
}

# Common stoichiometry sums (fractions on same site sum to these)
COMMON_SUMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@dataclass
class FractionCorrectionResult:
    """Result of fraction correction."""
    original: str
    corrected: str
    corrections_made: List[str] = field(default_factory=list)

    @property
    def was_corrected(self) -> bool:
        return self.original != self.corrected


class FractionCorrector:
    """
    Corrects fraction errors using stoichiometry constraints.

    Example:
        corrector = FractionCorrector()
        result = corrector.correct("La(1877/1000)Ba(119/1000)Cu1O4")
        # -> "La(1881/1000)Ba(119/1000)Cu1O4" (sum adjusted to 2000)
    """

    def __init__(
        self,
        snap_denominators: bool = True,
        enforce_stoichiometry: bool = True,
        denominator_consistency: bool = True,
        tolerance: int = 10,  # Max adjustment to numerator
    ):
        self.snap_denominators = snap_denominators
        self.enforce_stoichiometry = enforce_stoichiometry
        self.denominator_consistency = denominator_consistency
        self.tolerance = tolerance

    def correct(self, formula: str) -> FractionCorrectionResult:
        """Apply all fraction corrections."""
        corrections = []
        current = formula

        # 1. Snap invalid denominators to valid ones
        if self.snap_denominators:
            current, changed = self._snap_denominators(current)
            if changed:
                corrections.append("snapped_denominators")

        # 2. Enforce denominator consistency within formula
        if self.denominator_consistency:
            current, changed = self._enforce_denominator_consistency(current)
            if changed:
                corrections.append("denominator_consistency")

        # 3. Enforce stoichiometry constraints
        if self.enforce_stoichiometry:
            current, changed = self._enforce_stoichiometry(current)
            if changed:
                corrections.append("stoichiometry_adjusted")

        return FractionCorrectionResult(
            original=formula,
            corrected=current,
            corrections_made=corrections
        )

    def _snap_denominators(self, formula: str) -> Tuple[str, bool]:
        """Snap invalid denominators to nearest valid one."""
        changed = False

        def snap_fraction(match):
            nonlocal changed
            num, denom = int(match.group(1)), int(match.group(2))

            if denom in VALID_DENOMINATORS:
                return match.group(0)

            # Find nearest valid denominator
            nearest = min(VALID_DENOMINATORS, key=lambda d: abs(d - denom))

            # Adjust numerator proportionally
            new_num = round(num * nearest / denom)
            changed = True
            return f"({new_num}/{nearest})"

        result = re.sub(r'\((\d+)/(\d+)\)', snap_fraction, formula)
        return result, changed

    def _enforce_denominator_consistency(self, formula: str) -> Tuple[str, bool]:
        """Make all fractions use the most common denominator in the formula."""
        fractions = re.findall(r'\((\d+)/(\d+)\)', formula)
        if len(fractions) < 2:
            return formula, False

        # Find most common denominator
        denoms = [int(f[1]) for f in fractions]
        denom_counts = Counter(denoms)
        most_common_denom = denom_counts.most_common(1)[0][0]

        # Check if all already consistent
        if len(set(denoms)) == 1:
            return formula, False

        # Convert all to most common denominator
        changed = False
        def convert_fraction(match):
            nonlocal changed
            num, denom = int(match.group(1)), int(match.group(2))
            if denom == most_common_denom:
                return match.group(0)

            # Convert: num/denom -> new_num/most_common_denom
            new_num = round(num * most_common_denom / denom)
            changed = True
            return f"({new_num}/{most_common_denom})"

        result = re.sub(r'\((\d+)/(\d+)\)', convert_fraction, formula)
        return result, changed

    def _enforce_stoichiometry(self, formula: str) -> Tuple[str, bool]:
        """
        Adjust fractions so they sum to a nice stoichiometry number.

        Strategy:
        1. Group fractions by their likely "site" (consecutive fractions)
        2. Check if they sum close to a nice number (1, 2, 3, ...)
        3. Adjust the largest fraction to make the sum exact
        """
        fractions = list(re.finditer(r'\((\d+)/(\d+)\)', formula))
        if len(fractions) < 2:
            return formula, False

        # Group consecutive fractions (assume they're on same site)
        groups = self._group_consecutive_fractions(fractions, formula)

        changed = False
        result = formula

        for group in groups:
            if len(group) < 2:
                continue

            # Calculate current sum
            total_num = 0
            common_denom = None

            for match in group:
                num, denom = int(match.group(1)), int(match.group(2))
                if common_denom is None:
                    common_denom = denom
                elif denom != common_denom:
                    # Mixed denominators - skip this group
                    continue
                total_num += num

            if common_denom is None:
                continue

            # Check if close to a nice sum
            current_sum = total_num / common_denom
            target_sum = self._find_nearest_nice_sum(current_sum)

            if target_sum is None:
                continue

            target_num = int(target_sum * common_denom)
            diff = target_num - total_num

            # Only adjust if within tolerance
            if abs(diff) > self.tolerance:
                continue

            # Adjust the largest fraction
            largest_match = max(group, key=lambda m: int(m.group(1)))
            old_num = int(largest_match.group(1))
            new_num = old_num + diff

            if new_num <= 0:
                continue

            # Replace in formula
            old_frac = largest_match.group(0)
            new_frac = f"({new_num}/{common_denom})"
            result = result.replace(old_frac, new_frac, 1)
            changed = True

        return result, changed

    def _group_consecutive_fractions(
        self,
        fractions: List[re.Match],
        formula: str
    ) -> List[List[re.Match]]:
        """Group fractions that are likely on the same crystallographic site."""
        if not fractions:
            return []

        groups = []
        current_group = [fractions[0]]

        for i in range(1, len(fractions)):
            prev_end = fractions[i-1].end()
            curr_start = fractions[i].start()

            # Check what's between the fractions
            between = formula[prev_end:curr_start]

            # If only element symbols between, they're on same site
            # e.g., "La(7/10)Sr(3/10)" - La and Sr share A-site
            if re.match(r'^[A-Z][a-z]?$', between.strip()):
                current_group.append(fractions[i])
            else:
                groups.append(current_group)
                current_group = [fractions[i]]

        groups.append(current_group)
        return groups

    def _find_nearest_nice_sum(self, value: float) -> Optional[int]:
        """Find nearest nice sum (1, 2, 3, ...) within tolerance."""
        for target in COMMON_SUMS:
            if abs(value - target) < 0.02:  # Within 2%
                return target
        return None


def correct_fractions(formula: str) -> str:
    """Convenience function to correct fractions in a formula."""
    corrector = FractionCorrector()
    result = corrector.correct(formula)
    return result.corrected


def test_fraction_corrector():
    """Test the fraction corrector with real examples."""
    corrector = FractionCorrector()

    test_cases = [
        # (input, expected_behavior)
        ("La(1877/1000)Ba(119/1000)Cu1O4", "sum should adjust to 2000"),
        ("La(7/10)Sr(3/10)CuO4", "already correct, sum=1"),
        ("Bi(1/2)Sr(1/2)CuO3", "already correct, sum=1"),
        ("La(1877/1000)Ba(123/1000)Cu1O4", "already sums to 2000"),
        ("Y(8/45)Ba(2/10)Cu3O7", "weird denominator 45 should snap"),
    ]

    print("=" * 70)
    print("FRACTION CORRECTOR TEST")
    print("=" * 70)

    for formula, expected in test_cases:
        result = corrector.correct(formula)
        print(f"\nInput:    {formula}")
        print(f"Output:   {result.corrected}")
        print(f"Changed:  {result.was_corrected}")
        if result.corrections_made:
            print(f"Applied:  {result.corrections_made}")
        print(f"Expected: {expected}")


if __name__ == "__main__":
    test_fraction_corrector()
