"""
Canonical Ordering for Chemical Formulas.

Standardizes element ordering in chemical formulas for consistent
training and inference. Chemical formulas are inherently order-agnostic
(YBa2Cu3O7 = Ba2Cu3O7Y chemically), but autoregressive decoders learn
a specific ordering.

Available orderings:
1. Electronegativity: Cations first, anions last (most chemically intuitive)
2. Alphabetical: A-Z ordering (simple, reproducible)
3. Abundance: Most abundant elements first
4. Hill System: C first, H second, then alphabetical (organic chemistry standard)

February 2026
"""

import re
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


class OrderingMethod(Enum):
    """Available element ordering methods."""
    ELECTRONEGATIVITY = 'electronegativity'
    ALPHABETICAL = 'alphabetical'
    ABUNDANCE = 'abundance'
    HILL_SYSTEM = 'hill'
    ATOMIC_NUMBER = 'atomic_number'


# Pauling electronegativity values (higher = more electronegative)
ELECTRONEGATIVITY = {
    'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
    'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0.00, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0.00,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
    'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.00,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16,
    'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10, 'I': 2.66, 'Xe': 2.60,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
    'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.10, 'Dy': 1.22,
    'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Hf': 1.30,
    'Ta': 1.50, 'W': 2.36, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28,
    'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.00,
    'At': 2.20, 'Rn': 0.00, 'Fr': 0.70, 'Ra': 0.90, 'Ac': 1.10, 'Th': 1.30,
    'Pa': 1.50, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.30, 'Cm': 1.30,
    'Bk': 1.30, 'Cf': 1.30, 'Es': 1.30, 'Fm': 1.30, 'Md': 1.30, 'No': 1.30,
    'Lr': 1.30, 'Rf': 0.00, 'Db': 0.00, 'Sg': 0.00, 'Bh': 0.00, 'Hs': 0.00,
    'Mt': 0.00, 'Ds': 0.00, 'Rg': 0.00, 'Cn': 0.00, 'Nh': 0.00, 'Fl': 0.00,
    'Mc': 0.00, 'Lv': 0.00, 'Ts': 0.00, 'Og': 0.00,
}

# Atomic numbers for atomic number ordering
ATOMIC_NUMBER = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
    'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
    'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
    'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
    'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111,
    'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118,
}


@dataclass
class ElementWithFraction:
    """An element with its stoichiometric fraction."""
    element: str
    fraction_str: str  # Original fraction string, e.g., "(7/10)" or "2"
    numerator: Optional[int] = None
    denominator: Optional[int] = None

    @property
    def fraction_value(self) -> float:
        """Compute numerical fraction value."""
        if self.numerator is not None and self.denominator is not None:
            return self.numerator / self.denominator
        # Try to parse as integer
        try:
            return float(self.fraction_str.strip('()'))
        except ValueError:
            return 1.0


class CanonicalOrderer:
    """
    Standardizes element ordering in chemical formulas.

    Supports the fraction format: Element(numerator/denominator)
    Example: "La(7/10)Sr(3/10)CuO4" -> parsed and reordered
    """

    def __init__(self, method: OrderingMethod = OrderingMethod.ELECTRONEGATIVITY):
        self.method = method

    def parse_formula(self, formula: str) -> List[ElementWithFraction]:
        """
        Parse fraction-format formula into list of (element, fraction) pairs.

        Handles formats:
        - "La(7/10)Sr(3/10)CuO4" - fraction format
        - "YBa2Cu3O7" - integer stoichiometry
        - "Mg0.9Al0.1B2" - decimal format (legacy)

        Returns:
            List of ElementWithFraction objects
        """
        elements = []

        # Pattern for element with optional fraction/number
        # Matches: Element or Element(num/denom) or Element(digits) or ElementDigits
        pattern = r'([A-Z][a-z]?)(?:\((\d+)/(\d+)\)|\((\d+)\)|(\d+(?:\.\d+)?))?'

        for match in re.finditer(pattern, formula):
            elem = match.group(1)
            if not elem:
                continue

            # Determine fraction format
            if match.group(2) and match.group(3):
                # Fraction format: Element(num/denom)
                num = int(match.group(2))
                denom = int(match.group(3))
                frac_str = f"({num}/{denom})"
                elements.append(ElementWithFraction(elem, frac_str, num, denom))
            elif match.group(4):
                # Parenthesized integer: Element(n)
                n = match.group(4)
                elements.append(ElementWithFraction(elem, f"({n})", int(n), 1))
            elif match.group(5):
                # Simple number: ElementN or Element0.N
                n = match.group(5)
                elements.append(ElementWithFraction(elem, n))
            else:
                # No stoichiometry (implicit 1)
                elements.append(ElementWithFraction(elem, ""))

        return elements

    def elements_to_formula(self, elements: List[ElementWithFraction]) -> str:
        """
        Convert list of ElementWithFraction back to formula string.

        Returns:
            Formula string with elements in given order
        """
        parts = []
        for ef in elements:
            if ef.fraction_str:
                parts.append(f"{ef.element}{ef.fraction_str}")
            else:
                parts.append(ef.element)
        return ''.join(parts)

    def get_sort_key(self, ef: ElementWithFraction) -> tuple:
        """
        Get sort key for an element based on ordering method.

        Returns tuple for stable multi-key sorting.
        """
        elem = ef.element

        if self.method == OrderingMethod.ELECTRONEGATIVITY:
            # Lower electronegativity first (cations before anions)
            en = ELECTRONEGATIVITY.get(elem, 2.0)
            return (en, elem)

        elif self.method == OrderingMethod.ALPHABETICAL:
            return (elem,)

        elif self.method == OrderingMethod.ABUNDANCE:
            # Higher abundance (fraction) first
            return (-ef.fraction_value, elem)

        elif self.method == OrderingMethod.HILL_SYSTEM:
            # C first, H second, then alphabetical
            if elem == 'C':
                return (0, elem)
            elif elem == 'H':
                return (1, elem)
            else:
                return (2, elem)

        elif self.method == OrderingMethod.ATOMIC_NUMBER:
            return (ATOMIC_NUMBER.get(elem, 999), elem)

        else:
            return (elem,)

    def canonicalize(self, formula: str) -> str:
        """
        Convert formula to canonical ordering.

        Args:
            formula: Chemical formula string

        Returns:
            Formula with elements in canonical order
        """
        elements = self.parse_formula(formula)
        if not elements:
            return formula  # Return unchanged if parsing fails

        # Sort by ordering method
        elements.sort(key=self.get_sort_key)

        return self.elements_to_formula(elements)

    def canonicalize_batch(self, formulas: List[str]) -> List[str]:
        """Canonicalize a batch of formulas."""
        return [self.canonicalize(f) for f in formulas]


class OrderAugmentation:
    """
    Data augmentation by shuffling element orders.

    Since chemical formulas are order-agnostic, training on multiple
    orderings can improve model robustness.
    """

    def __init__(self, n_augmentations: int = 2, seed: Optional[int] = None):
        """
        Args:
            n_augmentations: Number of shuffled versions to generate
            seed: Random seed for reproducibility
        """
        self.n_augmentations = n_augmentations
        self.rng = __import__('random').Random(seed)
        self.orderer = CanonicalOrderer()

    def augment(self, formula: str, include_original: bool = True) -> List[str]:
        """
        Generate order-shuffled versions of formula.

        Args:
            formula: Original formula
            include_original: Whether to include the original in output

        Returns:
            List of formula strings with different element orderings
        """
        elements = self.orderer.parse_formula(formula)
        if len(elements) <= 1:
            return [formula]  # Nothing to shuffle

        augmented = []
        if include_original:
            augmented.append(formula)

        seen = {formula}  # Avoid duplicates

        for _ in range(self.n_augmentations * 2):  # Extra attempts for uniqueness
            if len(augmented) >= (self.n_augmentations + (1 if include_original else 0)):
                break

            shuffled = elements.copy()
            self.rng.shuffle(shuffled)
            shuffled_formula = self.orderer.elements_to_formula(shuffled)

            if shuffled_formula not in seen:
                seen.add(shuffled_formula)
                augmented.append(shuffled_formula)

        return augmented

    def augment_batch(
        self,
        formulas: List[str],
        include_original: bool = True,
    ) -> List[str]:
        """
        Augment a batch of formulas.

        Returns flattened list of all augmented formulas.
        """
        augmented = []
        for formula in formulas:
            augmented.extend(self.augment(formula, include_original))
        return augmented


# Convenience functions for common orderings
def to_electronegativity_order(formula: str) -> str:
    """Convert formula to electronegativity ordering (cations first)."""
    return CanonicalOrderer(OrderingMethod.ELECTRONEGATIVITY).canonicalize(formula)


def to_alphabetical_order(formula: str) -> str:
    """Convert formula to alphabetical ordering."""
    return CanonicalOrderer(OrderingMethod.ALPHABETICAL).canonicalize(formula)


def to_abundance_order(formula: str) -> str:
    """Convert formula to abundance ordering (major elements first)."""
    return CanonicalOrderer(OrderingMethod.ABUNDANCE).canonicalize(formula)


def shuffle_element_order(formula: str, n: int = 2) -> List[str]:
    """Generate n shuffled orderings of a formula."""
    return OrderAugmentation(n_augmentations=n).augment(formula)


# Validation
def validate_ordering_consistency(
    original: str,
    reordered: str,
    orderer: Optional[CanonicalOrderer] = None,
) -> bool:
    """
    Validate that reordered formula has same composition as original.

    Returns True if compositions match.
    """
    if orderer is None:
        orderer = CanonicalOrderer()

    orig_elements = orderer.parse_formula(original)
    reord_elements = orderer.parse_formula(reordered)

    # Compare element sets with fractions
    orig_comp = {(e.element, e.fraction_value) for e in orig_elements}
    reord_comp = {(e.element, e.fraction_value) for e in reord_elements}

    return orig_comp == reord_comp
