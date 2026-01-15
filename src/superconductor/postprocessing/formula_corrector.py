"""
Post-Processing Corrector for Generated Chemical Formulas.

Fixes STRUCTURAL errors that are correctable:
- Unbalanced parentheses
- Double/missing slashes in fractions
- Empty fractions
- Invalid element symbols (fuzzy matching)
- Trailing garbage characters

Does NOT fix SEMANTIC errors (uncorrectable):
- Wrong element choice (Cu instead of La)
- Wrong digit values (8/10 instead of 7/10)
- Missing/extra elements

Philosophy:
    If errors are CONSISTENT and PREDICTABLE, we can fix them in post-processing.
    This saves model capacity for learning the hard semantic correctness.
"""

import re
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from difflib import get_close_matches

# Valid element symbols (1 and 2 character)
VALID_ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
}

# Elements commonly found in superconductors (for fuzzy matching priority)
COMMON_SUPERCONDUCTOR_ELEMENTS = {
    'La', 'Sr', 'Cu', 'O', 'Y', 'Ba', 'Ca', 'Bi', 'Tl', 'Hg',
    'Pb', 'Fe', 'As', 'Se', 'Te', 'Mg', 'B', 'Nb', 'Ti', 'V',
    'Zr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Pr', 'Ce', 'Th', 'U', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'K', 'Rb', 'Cs', 'Na', 'Al', 'Ga', 'Ge',
}


class CorrectionType(Enum):
    """Types of corrections applied."""
    BALANCED_PARENS = auto()      # Added/removed parentheses
    FIXED_SLASH = auto()          # Fixed double slash or added missing slash
    REMOVED_EMPTY_FRACTION = auto()  # Removed ()
    FIXED_ELEMENT = auto()        # Fuzzy matched invalid element
    REMOVED_INVALID = auto()      # Removed unrecognized characters
    TRUNCATED_GARBAGE = auto()    # Removed trailing garbage
    FIXED_FRACTION_FORMAT = auto()  # Fixed malformed fraction like (710) -> (7/10)


@dataclass
class CorrectionResult:
    """Result of formula correction."""
    original: str
    corrected: str
    is_valid: bool
    corrections_applied: List[CorrectionType] = field(default_factory=list)
    correction_details: List[str] = field(default_factory=list)

    @property
    def was_corrected(self) -> bool:
        return len(self.corrections_applied) > 0

    @property
    def n_corrections(self) -> int:
        return len(self.corrections_applied)

    def __str__(self) -> str:
        if not self.was_corrected:
            return f"'{self.original}' - no corrections needed"
        return (
            f"'{self.original}' -> '{self.corrected}'\n"
            f"  Corrections: {[c.name for c in self.corrections_applied]}\n"
            f"  Details: {self.correction_details}"
        )


class FormulaCorrector:
    """
    Post-processing corrector for generated chemical formulas.

    Fixes consistent structural errors that the model makes,
    saving model capacity for learning semantic correctness.

    Usage:
        corrector = FormulaCorrector()
        result = corrector.correct("La(7/10Sr(3/10)CuO4")
        print(result.corrected)  # "La(7/10)Sr(3/10)CuO4"
        print(result.corrections_applied)  # [CorrectionType.BALANCED_PARENS]
    """

    def __init__(
        self,
        fix_parentheses: bool = True,
        fix_slashes: bool = True,
        fix_empty_fractions: bool = True,
        fix_elements: bool = True,
        fix_fraction_format: bool = True,
        remove_invalid: bool = True,
        element_fuzzy_threshold: float = 0.6,
    ):
        self.fix_parentheses = fix_parentheses
        self.fix_slashes = fix_slashes
        self.fix_empty_fractions = fix_empty_fractions
        self.fix_elements = fix_elements
        self.fix_fraction_format = fix_fraction_format
        self.remove_invalid = remove_invalid
        self.element_fuzzy_threshold = element_fuzzy_threshold

        # Build element lookup for fuzzy matching
        self.elements_lower = {e.lower(): e for e in VALID_ELEMENTS}
        self.all_elements_list = list(VALID_ELEMENTS)

    def correct(self, formula: str) -> CorrectionResult:
        """
        Apply all corrections to a formula.

        Args:
            formula: Generated formula string

        Returns:
            CorrectionResult with corrected formula and details
        """
        corrections = []
        details = []
        current = formula.strip()

        # 1. Remove trailing garbage (non-formula characters at end)
        current, changed, detail = self._truncate_garbage(current)
        if changed:
            corrections.append(CorrectionType.TRUNCATED_GARBAGE)
            details.append(detail)

        # 2. Fix double/triple slashes
        if self.fix_slashes:
            current, changed, detail = self._fix_multiple_slashes(current)
            if changed:
                corrections.append(CorrectionType.FIXED_SLASH)
                details.append(detail)

        # 3. Remove empty fractions ()
        if self.fix_empty_fractions:
            current, changed, detail = self._remove_empty_fractions(current)
            if changed:
                corrections.append(CorrectionType.REMOVED_EMPTY_FRACTION)
                details.append(detail)

        # 4. Fix fraction format: (710) -> (7/10), (31000) -> (3/1000)
        if self.fix_fraction_format:
            current, changed, detail = self._fix_fraction_format(current)
            if changed:
                corrections.append(CorrectionType.FIXED_FRACTION_FORMAT)
                details.append(detail)

        # 5. Balance parentheses
        if self.fix_parentheses:
            current, changed, detail = self._balance_parentheses(current)
            if changed:
                corrections.append(CorrectionType.BALANCED_PARENS)
                details.append(detail)

        # 6. Fix invalid element symbols
        if self.fix_elements:
            current, changed, detail = self._fix_elements(current)
            if changed:
                corrections.append(CorrectionType.FIXED_ELEMENT)
                details.append(detail)

        # 7. Remove any remaining invalid characters
        if self.remove_invalid:
            current, changed, detail = self._remove_invalid_chars(current)
            if changed:
                corrections.append(CorrectionType.REMOVED_INVALID)
                details.append(detail)

        # Validate final result
        is_valid = self._validate(current)

        return CorrectionResult(
            original=formula,
            corrected=current,
            is_valid=is_valid,
            corrections_applied=corrections,
            correction_details=details
        )

    def _truncate_garbage(self, formula: str) -> Tuple[str, bool, str]:
        """Remove trailing garbage characters."""
        # Valid characters in formula
        valid_pattern = re.compile(r'^[A-Za-z0-9()/]+')
        match = valid_pattern.match(formula)

        if match and match.end() < len(formula):
            truncated = formula[:match.end()]
            garbage = formula[match.end():]
            return truncated, True, f"Removed trailing: '{garbage}'"

        return formula, False, ""

    def _fix_multiple_slashes(self, formula: str) -> Tuple[str, bool, str]:
        """Replace multiple slashes with single slash."""
        fixed = re.sub(r'/+', '/', formula)
        if fixed != formula:
            return fixed, True, "Fixed multiple slashes"
        return formula, False, ""

    def _remove_empty_fractions(self, formula: str) -> Tuple[str, bool, str]:
        """Remove empty parentheses ()."""
        fixed = re.sub(r'\(\s*\)', '', formula)
        if fixed != formula:
            count = formula.count('()') - fixed.count('()')
            return fixed, True, f"Removed {count} empty fraction(s)"
        return formula, False, ""

    def _fix_fraction_format(self, formula: str) -> Tuple[str, bool, str]:
        """
        Fix malformed fractions.

        Common errors:
        - (710) -> (7/10)     # Missing slash
        - (31000) -> (3/1000) # Missing slash in large fraction
        - (7/1/0) -> (7/10)   # Extra slash
        """
        changed = False
        details = []

        # Pattern: digits only in parens (missing slash)
        # Heuristic: first digit(s) is numerator, rest is denominator
        def fix_no_slash(match):
            nonlocal changed, details
            digits = match.group(1)
            if len(digits) >= 2:
                # Try common patterns: 1 digit / rest, or 2 digits / rest
                # Most common: single digit numerator
                if len(digits) <= 4:
                    # (710) -> (7/10), (320) -> (3/20)
                    fixed = f"({digits[0]}/{digits[1:]})"
                else:
                    # (31000) -> (3/1000), (999100000) -> ???
                    # Heuristic: if starts with small digits, probably 1-2 digit numerator
                    if int(digits[0]) < 5:
                        fixed = f"({digits[0]}/{digits[1:]})"
                    else:
                        # Could be like (99100) meaning (99/100)
                        fixed = f"({digits[:2]}/{digits[2:]})"
                changed = True
                details.append(f"({digits})->{fixed}")
                return fixed
            return match.group(0)

        # Fix (digits) without slash
        formula = re.sub(r'\((\d{2,})\)', fix_no_slash, formula)

        # Fix extra slashes: (7/1/0) -> (7/10)
        def fix_extra_slash(match):
            nonlocal changed, details
            content = match.group(1)
            parts = content.split('/')
            if len(parts) > 2:
                # Keep first part as numerator, join rest as denominator
                fixed = f"({parts[0]}/{''.join(parts[1:])})"
                changed = True
                details.append(f"({content})->{fixed}")
                return fixed
            return match.group(0)

        formula = re.sub(r'\(([0-9/]+)\)', fix_extra_slash, formula)

        detail = "; ".join(details) if details else ""
        return formula, changed, detail

    def _balance_parentheses(self, formula: str) -> Tuple[str, bool, str]:
        """Balance parentheses by adding missing ones."""
        open_count = formula.count('(')
        close_count = formula.count(')')

        if open_count == close_count:
            return formula, False, ""

        if open_count > close_count:
            # Add missing closing parens at end
            diff = open_count - close_count
            fixed = formula + ')' * diff
            return fixed, True, f"Added {diff} closing paren(s)"
        else:
            # Remove extra closing parens from end
            diff = close_count - open_count
            # Only remove from end if they're trailing
            fixed = formula
            removed = 0
            while fixed.endswith(')') and fixed.count(')') > fixed.count('('):
                fixed = fixed[:-1]
                removed += 1
            if removed > 0:
                return fixed, True, f"Removed {removed} extra closing paren(s)"
            # If extra parens are in middle, try to find and remove
            # This is harder - for now just note it
            return formula, False, ""

    def _fix_elements(self, formula: str) -> Tuple[str, bool, str]:
        """Fix invalid element symbols using fuzzy matching."""
        # Extract potential element symbols (capital letter optionally followed by lowercase)
        pattern = r'([A-Z][a-z]?)'

        changed = False
        details = []

        def fix_element(match):
            nonlocal changed, details
            elem = match.group(1)

            if elem in VALID_ELEMENTS:
                return elem

            # Try fuzzy matching
            # First try exact lowercase match (case error)
            if elem.lower() in self.elements_lower:
                fixed = self.elements_lower[elem.lower()]
                if fixed != elem:
                    changed = True
                    details.append(f"{elem}->{fixed}")
                    return fixed

            # Try close matches
            matches = get_close_matches(
                elem,
                self.all_elements_list,
                n=1,
                cutoff=self.element_fuzzy_threshold
            )
            if matches:
                fixed = matches[0]
                changed = True
                details.append(f"{elem}->{fixed}")
                return fixed

            # No good match - keep as is (will be flagged as invalid)
            return elem

        fixed = re.sub(pattern, fix_element, formula)
        detail = "; ".join(details) if details else ""
        return fixed, changed, detail

    def _remove_invalid_chars(self, formula: str) -> Tuple[str, bool, str]:
        """Remove characters that aren't valid in formulas."""
        # Valid: A-Z, a-z, 0-9, (, ), /
        valid_pattern = r'[^A-Za-z0-9()/]'
        invalid_chars = set(re.findall(valid_pattern, formula))

        if invalid_chars:
            fixed = re.sub(valid_pattern, '', formula)
            return fixed, True, f"Removed invalid: {invalid_chars}"

        return formula, False, ""

    def _validate(self, formula: str) -> bool:
        """Check if formula is valid after corrections."""
        if not formula:
            return False

        # Check balanced parentheses
        if formula.count('(') != formula.count(')'):
            return False

        # Check has at least one element
        if not re.search(r'[A-Z]', formula):
            return False

        # Check all fractions have proper format
        fractions = re.findall(r'\(([^)]+)\)', formula)
        for frac in fractions:
            if '/' not in frac:
                return False
            parts = frac.split('/')
            if len(parts) != 2:
                return False
            if not (parts[0].isdigit() and parts[1].isdigit()):
                return False

        return True


def correct_formula(formula: str) -> str:
    """
    Convenience function to correct a formula.

    Args:
        formula: Generated formula string

    Returns:
        Corrected formula string
    """
    corrector = FormulaCorrector()
    result = corrector.correct(formula)
    return result.corrected


def validate_formula(formula: str) -> bool:
    """
    Check if a formula is valid.

    Args:
        formula: Formula string

    Returns:
        True if valid
    """
    corrector = FormulaCorrector()
    return corrector._validate(formula)


# =============================================================================
# Batch Processing and Statistics
# =============================================================================

@dataclass
class CorrectionStatistics:
    """Statistics about corrections applied to a batch of formulas."""
    total: int
    corrected: int
    valid_after_correction: int
    correction_counts: Dict[CorrectionType, int] = field(default_factory=dict)

    @property
    def correction_rate(self) -> float:
        return self.corrected / self.total if self.total > 0 else 0.0

    @property
    def validity_rate(self) -> float:
        return self.valid_after_correction / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:
        lines = [
            f"Correction Statistics ({self.total} formulas):",
            f"  Needed correction: {self.corrected} ({self.correction_rate:.1%})",
            f"  Valid after correction: {self.valid_after_correction} ({self.validity_rate:.1%})",
            "  Correction types:"
        ]
        for ctype, count in sorted(self.correction_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {ctype.name}: {count}")
        return "\n".join(lines)


def correct_batch(
    formulas: List[str],
    corrector: Optional[FormulaCorrector] = None
) -> Tuple[List[str], CorrectionStatistics]:
    """
    Correct a batch of formulas and gather statistics.

    Args:
        formulas: List of generated formulas
        corrector: Optional FormulaCorrector instance

    Returns:
        Tuple of (corrected_formulas, statistics)
    """
    if corrector is None:
        corrector = FormulaCorrector()

    corrected = []
    stats = CorrectionStatistics(
        total=len(formulas),
        corrected=0,
        valid_after_correction=0,
        correction_counts={ct: 0 for ct in CorrectionType}
    )

    for formula in formulas:
        result = corrector.correct(formula)
        corrected.append(result.corrected)

        if result.was_corrected:
            stats.corrected += 1
        if result.is_valid:
            stats.valid_after_correction += 1

        for ctype in result.corrections_applied:
            stats.correction_counts[ctype] += 1

    return corrected, stats
