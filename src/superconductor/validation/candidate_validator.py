"""
Rule-based validation for superconductor candidates.

Validates chemical plausibility without DFT calculations:
1. Charge balance (oxidation states)
2. Electronegativity spread
3. Element compatibility
4. Stoichiometry validity
5. Superconductor family heuristics

Note: For thermodynamic stability validation, DFT calculations
or ML surrogates would be needed (separate project).
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import re
from itertools import product

from ..encoders.element_properties import (
    ELEMENT_PROPERTIES,
    get_element_property,
    get_oxidation_states,
    SUPERCONDUCTOR_ELEMENTS,
)
from ..encoders.composition_encoder import CompositionEncoder


@dataclass
class ValidationResult:
    """Container for validation results."""
    formula: str
    is_valid: bool
    overall_score: float  # 0-1 score

    # Individual checks
    charge_balance: bool
    charge_balance_score: float
    electronegativity_spread: bool
    electronegativity_score: float
    element_compatibility: bool
    compatibility_score: float
    stoichiometry_valid: bool
    stoichiometry_score: float
    superconductor_likelihood: float

    # Details
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    inferred_structure: Optional[str] = None
    likely_family: Optional[str] = None


class CandidateValidator:
    """
    Rule-based validator for superconductor candidates.

    Checks chemical plausibility using domain knowledge
    without expensive DFT calculations.

    Example:
        validator = CandidateValidator()
        result = validator.validate("YBa2Cu3O7")
        print(f"Valid: {result.is_valid}, Score: {result.overall_score:.2f}")
    """

    # Known incompatible element pairs
    INCOMPATIBLE_PAIRS = [
        {'Na', 'K'},  # Alkali metals compete for same sites
        {'F', 'Cl', 'Br', 'I'},  # Multiple halogens typically don't mix well
    ]

    # Known good element combinations for superconductors
    FAVORABLE_COMBINATIONS = {
        'cuprate': [
            {'Cu', 'O'},
            {'Cu', 'O', 'Ba'},
            {'Cu', 'O', 'Sr'},
            {'Cu', 'O', 'Y'},
            {'Cu', 'O', 'La'},
        ],
        'iron_based': [
            {'Fe', 'As'},
            {'Fe', 'Se'},
            {'Fe', 'As', 'La'},
            {'Fe', 'As', 'Ba'},
        ],
        'MgB2': [
            {'Mg', 'B'},
        ],
        'A15': [
            {'Nb', 'Sn'},
            {'Nb', 'Ge'},
            {'V', 'Si'},
        ],
    }

    # Perovskite tolerance factor ranges
    PEROVSKITE_TOLERANCE = (0.8, 1.1)  # Acceptable range

    def __init__(
        self,
        strict_mode: bool = False,
        min_score_threshold: float = 0.5,
        require_known_family: bool = False,
        min_sc_likelihood: float = 0.0
    ):
        """
        Initialize validator.

        Args:
            strict_mode: If True, require all checks to pass
            min_score_threshold: Minimum overall score for validity
            require_known_family: If True, reject candidates not matching known SC families
            min_sc_likelihood: Minimum superconductor likelihood score (0.0-1.0)
        """
        self.strict_mode = strict_mode
        self.min_score_threshold = min_score_threshold
        self.require_known_family = require_known_family
        self.min_sc_likelihood = min_sc_likelihood
        self.encoder = CompositionEncoder()

    def validate(self, formula: str) -> ValidationResult:
        """
        Validate a candidate formula.

        Args:
            formula: Chemical formula string

        Returns:
            ValidationResult with all check results
        """
        # Parse formula
        stoichiometry = self.encoder.parse_formula(formula)

        if not stoichiometry:
            return ValidationResult(
                formula=formula,
                is_valid=False,
                overall_score=0.0,
                charge_balance=False,
                charge_balance_score=0.0,
                electronegativity_spread=False,
                electronegativity_score=0.0,
                element_compatibility=False,
                compatibility_score=0.0,
                stoichiometry_valid=False,
                stoichiometry_score=0.0,
                superconductor_likelihood=0.0,
                warnings=["Could not parse formula"]
            )

        warnings = []
        suggestions = []

        # 1. Check charge balance
        charge_ok, charge_score, charge_warning = self._check_charge_balance(stoichiometry)
        if charge_warning:
            warnings.append(charge_warning)

        # 2. Check electronegativity spread
        en_ok, en_score, en_warning = self._check_electronegativity_spread(stoichiometry)
        if en_warning:
            warnings.append(en_warning)

        # 3. Check element compatibility
        compat_ok, compat_score, compat_warning = self._check_element_compatibility(stoichiometry)
        if compat_warning:
            warnings.append(compat_warning)

        # 4. Check stoichiometry
        stoich_ok, stoich_score, stoich_warning = self._check_stoichiometry(stoichiometry)
        if stoich_warning:
            warnings.append(stoich_warning)

        # 5. Estimate superconductor likelihood
        sc_likelihood, likely_family = self._estimate_superconductor_likelihood(stoichiometry)

        # 6. Infer structure type
        inferred_structure = self._infer_structure_type(stoichiometry)

        # Calculate overall score
        overall_score = (
            0.3 * charge_score +
            0.2 * en_score +
            0.2 * compat_score +
            0.15 * stoich_score +
            0.15 * sc_likelihood
        )

        # Determine validity
        if self.strict_mode:
            is_valid = all([charge_ok, en_ok, compat_ok, stoich_ok])
        else:
            is_valid = overall_score >= self.min_score_threshold

        # Additional filters
        if self.require_known_family and likely_family is None:
            is_valid = False
            warnings.append("Does not match any known superconductor family")

        if sc_likelihood < self.min_sc_likelihood:
            is_valid = False
            warnings.append(f"SC likelihood {sc_likelihood:.2f} below threshold {self.min_sc_likelihood:.2f}")

        # Generate suggestions
        if not charge_ok:
            suggestions.append("Consider adjusting oxygen content for charge balance")
        if sc_likelihood < 0.3:
            suggestions.append("Formula doesn't match known superconductor patterns")
        if likely_family:
            suggestions.append(f"Resembles {likely_family} superconductors")

        return ValidationResult(
            formula=formula,
            is_valid=is_valid,
            overall_score=overall_score,
            charge_balance=charge_ok,
            charge_balance_score=charge_score,
            electronegativity_spread=en_ok,
            electronegativity_score=en_score,
            element_compatibility=compat_ok,
            compatibility_score=compat_score,
            stoichiometry_valid=stoich_ok,
            stoichiometry_score=stoich_score,
            superconductor_likelihood=sc_likelihood,
            warnings=warnings,
            suggestions=suggestions,
            inferred_structure=inferred_structure,
            likely_family=likely_family
        )

    def _check_charge_balance(
        self,
        stoichiometry: Dict[str, float]
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check if formula has plausible charge balance.

        Tries common oxidation state combinations to see if
        total charge can be zero.
        """
        elements = list(stoichiometry.keys())
        counts = [stoichiometry[e] for e in elements]

        # Get possible oxidation states for each element
        ox_states = []
        for elem in elements:
            states = get_oxidation_states(elem)
            if not states:
                states = [0]  # Unknown element, assume neutral
            ox_states.append(states)

        # Try all combinations
        best_imbalance = float('inf')

        for state_combo in product(*ox_states):
            total_charge = sum(
                count * state
                for count, state in zip(counts, state_combo)
            )
            imbalance = abs(total_charge)
            if imbalance < best_imbalance:
                best_imbalance = imbalance

            if imbalance < 0.5:  # Allow small numerical error
                return True, 1.0, None

        # Partial credit for near-balanced
        score = max(0, 1.0 - best_imbalance / 10.0)

        warning = None
        if best_imbalance > 2:
            warning = f"Charge imbalance of {best_imbalance:.1f} detected"

        return best_imbalance < 2.0, score, warning

    def _check_electronegativity_spread(
        self,
        stoichiometry: Dict[str, float]
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check for reasonable electronegativity difference.

        Superconductors typically have ionic character (EN spread)
        but not too extreme.
        """
        en_values = []
        for elem in stoichiometry.keys():
            en = get_element_property(elem, 'electronegativity')
            if en and en > 0:
                en_values.append(en)

        if len(en_values) < 2:
            return True, 0.5, "Single element or missing EN data"

        spread = max(en_values) - min(en_values)

        # Ideal range for superconductors: 0.5 to 2.5
        if 0.5 <= spread <= 2.5:
            score = 1.0
            return True, score, None
        elif spread < 0.5:
            score = spread / 0.5
            return False, score, f"Low electronegativity spread ({spread:.2f})"
        else:
            score = max(0, 1.0 - (spread - 2.5) / 1.0)
            return False, score, f"High electronegativity spread ({spread:.2f})"

    def _check_element_compatibility(
        self,
        stoichiometry: Dict[str, float]
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check for known incompatible element pairs.
        """
        elements = set(stoichiometry.keys())

        for incompatible_set in self.INCOMPATIBLE_PAIRS:
            intersection = elements & incompatible_set
            if len(intersection) >= 2:
                return False, 0.3, f"Incompatible elements: {intersection}"

        # Check for favorable combinations
        for family, combos in self.FAVORABLE_COMBINATIONS.items():
            for combo in combos:
                if combo.issubset(elements):
                    return True, 1.0, None

        return True, 0.7, None  # Neutral - no known issues

    def _check_stoichiometry(
        self,
        stoichiometry: Dict[str, float]
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check stoichiometry validity.

        - Counts should be reasonable (not too large)
        - Should not have too many different elements
        """
        total = sum(stoichiometry.values())
        n_elements = len(stoichiometry)
        counts = list(stoichiometry.values())

        warnings = []

        # Check total atoms
        if total > 50:
            warnings.append(f"Very large formula unit ({total:.0f} atoms)")
            total_score = 0.5
        else:
            total_score = 1.0

        # Check element count
        if n_elements > 8:
            warnings.append(f"Many elements ({n_elements})")
            elem_score = 0.5
        else:
            elem_score = 1.0

        # Check for very small fractions
        min_count = min(counts)
        if min_count < 0.1:
            warnings.append(f"Very small stoichiometry ({min_count:.3f})")
            frac_score = 0.5
        else:
            frac_score = 1.0

        score = (total_score + elem_score + frac_score) / 3
        warning = "; ".join(warnings) if warnings else None

        return score > 0.6, score, warning

    def _estimate_superconductor_likelihood(
        self,
        stoichiometry: Dict[str, float]
    ) -> Tuple[float, Optional[str]]:
        """
        Estimate likelihood this is a superconductor.

        Based on presence of elements common in superconductors.
        """
        elements = set(stoichiometry.keys())

        # Check against known superconductor families
        best_match = 0.0
        best_family = None

        # Cuprate check
        if 'Cu' in elements and 'O' in elements:
            cuprate_elements = {'Ba', 'Sr', 'Ca', 'Y', 'La', 'Nd', 'Bi', 'Tl', 'Hg'}
            if elements & cuprate_elements:
                match = len(elements & (cuprate_elements | {'Cu', 'O'})) / len(elements)
                if match > best_match:
                    best_match = match
                    best_family = 'cuprate'

        # Iron-based check
        if 'Fe' in elements:
            iron_elements = {'As', 'P', 'Se', 'Te', 'S', 'La', 'Ba', 'Sr', 'K', 'Na'}
            if elements & iron_elements:
                match = len(elements & (iron_elements | {'Fe'})) / len(elements)
                if match > best_match:
                    best_match = match
                    best_family = 'iron_based'

        # MgB2 check
        if 'Mg' in elements and 'B' in elements:
            if len(elements) <= 3:
                match = 0.9
                if match > best_match:
                    best_match = match
                    best_family = 'MgB2'

        # A15 check
        if 'Nb' in elements or 'V' in elements:
            a15_elements = {'Sn', 'Ge', 'Si', 'Al', 'Ga'}
            if elements & a15_elements:
                match = 0.85
                if match > best_match:
                    best_match = match
                    best_family = 'A15'

        # Conventional check (elemental or simple binary)
        conventional_sc = {'Al', 'Pb', 'Sn', 'In', 'Nb', 'V', 'Ta', 'Hg', 'Ti'}
        if len(elements) <= 2 and elements & conventional_sc:
            match = 0.7
            if match > best_match:
                best_match = match
                best_family = 'conventional'

        # Hydride check
        if 'H' in elements:
            h_count = stoichiometry.get('H', 0)
            total = sum(stoichiometry.values())
            if h_count / total > 0.5:
                hydride_hosts = {'La', 'Y', 'Ca', 'S', 'C'}
                if elements & hydride_hosts:
                    match = 0.6
                    if match > best_match:
                        best_match = match
                        best_family = 'hydride'

        return best_match, best_family

    def _infer_structure_type(
        self,
        stoichiometry: Dict[str, float]
    ) -> Optional[str]:
        """
        Infer likely crystal structure type.
        """
        elements = set(stoichiometry.keys())
        total = sum(stoichiometry.values())

        # Perovskite: ABO3 or A2BO4 (Ruddlesden-Popper)
        if 'O' in elements:
            o_count = stoichiometry.get('O', 0)
            o_ratio = o_count / total

            # ABO3 type
            if 0.55 <= o_ratio <= 0.65:
                return 'perovskite'

            # Layered perovskite
            if 0.45 <= o_ratio <= 0.55:
                return 'layered_perovskite'

        # Check for layered cuprate
        if 'Cu' in elements and 'O' in elements:
            if any(e in elements for e in ['Ba', 'Sr', 'Ca', 'Y', 'La']):
                return 'layered_cuprate'

        # ThCr2Si2 type (iron-based)
        if 'Fe' in elements and any(e in elements for e in ['As', 'P']):
            return 'ThCr2Si2'

        return None

    def validate_batch(
        self,
        formulas: List[str]
    ) -> List[ValidationResult]:
        """Validate multiple formulas."""
        return [self.validate(f) for f in formulas]


def validate_candidate(
    formula: str,
    strict: bool = False
) -> ValidationResult:
    """
    Convenience function for single validation.

    Args:
        formula: Chemical formula string
        strict: Whether to use strict validation mode

    Returns:
        ValidationResult
    """
    validator = CandidateValidator(strict_mode=strict)
    return validator.validate(formula)
