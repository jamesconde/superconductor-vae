"""
Physics-based validation for superconductor candidates.

Validates using physical principles, NOT by matching known families:
1. Hume-Rothery rules for alloy/solid solution formation
2. Immiscible element pairs (elements that don't mix)
3. Bond type compatibility (metallic, ionic, covalent)
4. Thermodynamic stability heuristics
5. Known impossible combinations

This allows discovery of NEW superconductor families while filtering
physically impossible compounds.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from ..encoders.element_properties import (
    ELEMENT_PROPERTIES,
    get_element_property,
    get_oxidation_states,
)


@dataclass
class PhysicsValidationResult:
    """Result of physics-based validation."""
    formula: str
    is_plausible: bool
    plausibility_score: float  # 0-1

    # Individual physics checks
    hume_rothery_compatible: bool
    hume_rothery_score: float
    no_immiscible_pairs: bool
    bond_types_compatible: bool
    no_impossible_combinations: bool
    reasonable_composition: bool

    # Details
    issues: List[str] = field(default_factory=list)
    element_analysis: Dict[str, str] = field(default_factory=dict)

    # Synthesis conditions that might enable exotic phases
    # (e.g., high pressure, rapid quench, thin film deposition)
    potential_synthesis_routes: List[str] = field(default_factory=list)


class PhysicsValidator:
    """
    Validate candidates using physics principles.

    Does NOT filter based on known superconductor families - that would
    prevent discovery of new types. Instead, filters based on whether
    the compound could physically exist.
    """

    # Known immiscible metal pairs (limited solid solubility)
    # Based on binary phase diagrams - these pairs violate Hume-Rothery rules
    # (size difference >15%, different crystal structures, electronegativity mismatch)
    # Note: Rapid solidification, thin films, or extreme pressure can sometimes
    # create metastable alloys even from immiscible pairs
    IMMISCIBLE_PAIRS = {
        # Cu-based: Cu has fcc structure, these are bcc with large size mismatch
        frozenset({'Cu', 'W'}),    # 25% size diff, fcc vs bcc, immiscible
        frozenset({'Cu', 'Mo'}),   # 18% size diff, fcc vs bcc
        frozenset({'Cr', 'Cu'}),   # Different crystal structure, limited
        frozenset({'V', 'Cu'}),    # 15% size diff, fcc vs bcc
        # Noble metals don't mix with refractory metals
        frozenset({'Ag', 'Ni'}),   # Large EN diff (0.3), immiscible
        frozenset({'Ag', 'W'}),    # Completely immiscible, no intermetallics
        frozenset({'Au', 'W'}),    # Immiscible, large atomic size mismatch
        frozenset({'Au', 'Re'}),   # Immiscible
        frozenset({'Au', 'Mo'}),   # Very limited, different structures
        # Iron-based: Fe doesn't alloy with certain elements
        frozenset({'Fe', 'Ag'}),   # Completely immiscible (no solid solubility)
        frozenset({'Fe', 'Cu'}),   # <5% solubility at room temp
        # Lead-based: Pb is large and soft metal
        frozenset({'Pb', 'Fe'}),   # Immiscible (different bonding character)
        frozenset({'Pb', 'Cu'}),   # Very limited mutual solubility
        frozenset({'Bi', 'Cu'}),   # Limited solubility
    }

    # Elements that don't form stable compounds with metals
    # (noble gases, some metalloids in certain contexts)
    INERT_ELEMENTS = {'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'}

    # Alkali metals - having multiple is unusual/unstable
    ALKALI_METALS = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'}

    # Alkaline earth metals
    ALKALINE_EARTH = {'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'}

    # Halogens - multiple halogens in same compound is rare
    HALOGENS = {'F', 'Cl', 'Br', 'I', 'At'}

    # Transition metals (for metallic bonding)
    TRANSITION_METALS = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    }

    # Lanthanides
    LANTHANIDES = {
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
        'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'
    }

    # Actinides
    ACTINIDES = {
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
    }

    # Elements that commonly form anions (for ionic compounds)
    COMMON_ANIONS = {'O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'F', 'Cl', 'Br', 'I'}

    def __init__(
        self,
        max_atomic_size_diff: float = 0.15,  # Hume-Rothery: 15%
        max_en_diff_for_metallic: float = 0.5,  # For metallic bonding
        allow_radioactive: bool = False
    ):
        """
        Args:
            max_atomic_size_diff: Maximum atomic radius difference for solid solution
            max_en_diff_for_metallic: Max electronegativity diff for metallic bonds
            allow_radioactive: Whether to allow radioactive elements
        """
        self.max_atomic_size_diff = max_atomic_size_diff
        self.max_en_diff_for_metallic = max_en_diff_for_metallic
        self.allow_radioactive = allow_radioactive

        # Radioactive elements (short-lived or dangerous)
        self.radioactive_elements = {
            'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
            'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
        }
        # Th and U are sometimes used, so we don't include them by default

    def validate(self, formula: str, stoichiometry: Dict[str, float]) -> PhysicsValidationResult:
        """
        Validate a candidate using physics principles.

        Args:
            formula: Chemical formula string
            stoichiometry: Dict of element -> count

        Returns:
            PhysicsValidationResult
        """
        elements = set(stoichiometry.keys())
        issues = []
        element_analysis = {}

        # Classify elements
        for elem in elements:
            elem_type = self._classify_element(elem)
            element_analysis[elem] = elem_type

        # 1. Check for noble gases (can't form compounds)
        noble_in_formula = elements & self.INERT_ELEMENTS
        if noble_in_formula:
            issues.append(f"Contains noble gas(es) {noble_in_formula} which don't form compounds")
            no_impossible = False
        else:
            no_impossible = True

        # 2. Check for radioactive elements
        if not self.allow_radioactive:
            radioactive = elements & self.radioactive_elements
            if radioactive:
                issues.append(f"Contains unstable radioactive element(s): {radioactive}")
                # This is a warning, not necessarily impossible

        # 3. Check for immiscible pairs
        no_immiscible = True
        for pair in self.IMMISCIBLE_PAIRS:
            if pair.issubset(elements):
                issues.append(f"Contains immiscible pair {pair} - won't form stable alloy")
                no_immiscible = False

        # 4. Check Hume-Rothery rules for metallic elements
        hr_compatible, hr_score, hr_issues = self._check_hume_rothery(elements, stoichiometry)
        issues.extend(hr_issues)

        # 5. Check bond type compatibility
        bond_compatible, bond_issues = self._check_bond_compatibility(elements, stoichiometry)
        issues.extend(bond_issues)

        # 6. Check for problematic element combinations
        combo_ok, combo_issues = self._check_element_combinations(elements, stoichiometry)
        issues.extend(combo_issues)
        no_impossible = no_impossible and combo_ok

        # 7. Check composition reasonableness
        reasonable, reason_issues = self._check_composition_reasonable(stoichiometry)
        issues.extend(reason_issues)

        # 8. Suggest potential synthesis routes for exotic compositions
        synthesis_routes = self._suggest_synthesis_routes(
            elements, no_immiscible, bond_compatible, hr_compatible
        )

        # Calculate overall plausibility
        scores = [
            hr_score,
            1.0 if no_immiscible else 0.3,
            1.0 if bond_compatible else 0.5,
            1.0 if no_impossible else 0.0,
            1.0 if reasonable else 0.5
        ]
        plausibility_score = np.mean(scores)

        # Determine if plausible
        is_plausible = (
            no_impossible and
            no_immiscible and
            plausibility_score >= 0.5
        )

        return PhysicsValidationResult(
            formula=formula,
            is_plausible=is_plausible,
            plausibility_score=plausibility_score,
            hume_rothery_compatible=hr_compatible,
            hume_rothery_score=hr_score,
            no_immiscible_pairs=no_immiscible,
            bond_types_compatible=bond_compatible,
            no_impossible_combinations=no_impossible,
            reasonable_composition=reasonable,
            issues=issues,
            element_analysis=element_analysis,
            potential_synthesis_routes=synthesis_routes
        )

    def _classify_element(self, elem: str) -> str:
        """Classify element by type."""
        if elem in self.ALKALI_METALS:
            return 'alkali_metal'
        elif elem in self.ALKALINE_EARTH:
            return 'alkaline_earth'
        elif elem in self.TRANSITION_METALS:
            return 'transition_metal'
        elif elem in self.LANTHANIDES:
            return 'lanthanide'
        elif elem in self.ACTINIDES:
            return 'actinide'
        elif elem in self.HALOGENS:
            return 'halogen'
        elif elem in self.INERT_ELEMENTS:
            return 'noble_gas'
        elif elem in {'B', 'Si', 'Ge', 'As', 'Sb', 'Te'}:
            return 'metalloid'
        elif elem in {'C', 'N', 'O', 'P', 'S', 'Se'}:
            return 'nonmetal'
        elif elem == 'H':
            return 'hydrogen'
        else:
            return 'other_metal'

    def _check_hume_rothery(
        self,
        elements: Set[str],
        stoichiometry: Dict[str, float]
    ) -> Tuple[bool, float, List[str]]:
        """
        Check Hume-Rothery rules for solid solution formation.

        Rules:
        1. Atomic size difference < 15%
        2. Similar electronegativity
        3. Same crystal structure (we approximate)
        4. Similar valence
        """
        issues = []
        metal_elements = elements & (self.TRANSITION_METALS | self.LANTHANIDES |
                                     self.ALKALI_METALS | self.ALKALINE_EARTH)

        if len(metal_elements) < 2:
            # Not enough metals to check Hume-Rothery
            return True, 1.0, []

        # Get atomic radii
        radii = {}
        for elem in metal_elements:
            r = get_element_property(elem, 'atomic_radius')
            if r and r > 0:
                radii[elem] = r

        if len(radii) < 2:
            return True, 0.8, []

        # Check size differences
        radii_vals = list(radii.values())
        max_r, min_r = max(radii_vals), min(radii_vals)
        size_diff = (max_r - min_r) / max_r if max_r > 0 else 0

        if size_diff > self.max_atomic_size_diff:
            issues.append(f"Atomic size difference {size_diff:.1%} > {self.max_atomic_size_diff:.0%} (Hume-Rothery)")
            size_score = max(0, 1 - (size_diff - self.max_atomic_size_diff) / 0.15)
        else:
            size_score = 1.0

        # Check electronegativity difference
        ens = {}
        for elem in metal_elements:
            en = get_element_property(elem, 'electronegativity')
            if en and en > 0:
                ens[elem] = en

        if len(ens) >= 2:
            en_vals = list(ens.values())
            en_diff = max(en_vals) - min(en_vals)
            if en_diff > 1.0:  # Large EN diff suggests ionic, not metallic
                issues.append(f"Large electronegativity difference ({en_diff:.2f}) between metals")
                en_score = max(0, 1 - (en_diff - 1.0) / 1.0)
            else:
                en_score = 1.0
        else:
            en_score = 0.8

        overall_score = (size_score + en_score) / 2
        compatible = overall_score >= 0.6

        return compatible, overall_score, issues

    def _check_bond_compatibility(
        self,
        elements: Set[str],
        stoichiometry: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check if bond types are compatible.

        - Metals + metals = metallic bonding (alloys) ✓
        - Metals + nonmetals = ionic/covalent (compounds) ✓
        - Multiple alkali metals = problematic
        - Nonmetals only = molecular compounds (less likely SC)
        """
        issues = []

        # Count by type
        alkali = elements & self.ALKALI_METALS
        alkaline = elements & self.ALKALINE_EARTH
        transition = elements & self.TRANSITION_METALS
        halogens = elements & self.HALOGENS
        nonmetals = elements & self.COMMON_ANIONS

        # Multiple alkali metals is thermodynamically unstable under normal conditions
        # Physics: Alkali metals have identical +1 valence, similar bcc structures,
        # and nearly identical ionic radii. They compete for the same crystallographic
        # sites and show almost complete immiscibility (see Na-K, K-Rb phase diagrams).
        # The mixing enthalpy is positive due to electronic structure similarity.
        # NOTE: Under extreme pressure (>100 GPa), behavior can change dramatically.
        multiple_alkali = len(alkali) > 1
        if multiple_alkali:
            issues.append(
                f"Multiple alkali metals {alkali} - thermodynamically immiscible at ambient pressure "
                f"(identical +1 valence, similar radii → compete for same sites, positive mixing enthalpy). "
                f"Extreme pressure synthesis may enable exotic phases."
            )

        # Multiple halogens in same compound is rare
        if len(halogens) > 1:
            issues.append(f"Multiple halogens {halogens} - unusual combination")

        # All nonmetals (no metals) - unlikely superconductor
        metals = alkali | alkaline | transition | self.LANTHANIDES
        if not metals and elements - self.COMMON_ANIONS - {'H', 'C'}:
            issues.append("No metal elements - unlikely to be superconductor")

        # Bond compatibility fails if multiple alkali metals (unstable)
        compatible = not multiple_alkali
        return compatible, issues

    def _check_element_combinations(
        self,
        elements: Set[str],
        stoichiometry: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check for known problematic combinations.
        """
        issues = []

        # Alkali + halogen without other elements = salt, not SC
        alkali = elements & self.ALKALI_METALS
        halogens = elements & self.HALOGENS
        if alkali and halogens and len(elements) == 2:
            issues.append(f"Simple alkali halide salt ({alkali}, {halogens}) - not a superconductor")
            return False, issues

        # Mercury + thallium compounds can be toxic/unstable
        if 'Hg' in elements and 'Tl' in elements:
            issues.append("Hg-Tl combination - toxic and potentially unstable")

        # Certain actinides are dangerous
        dangerous_actinides = elements & {'Pu', 'Am', 'Cm'}
        if dangerous_actinides:
            issues.append(f"Contains highly radioactive {dangerous_actinides}")
            return False, issues

        return True, issues

    def _check_composition_reasonable(
        self,
        stoichiometry: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check if composition is reasonable.

        Note: High-entropy alloys (HEAs) can have 15+ principal elements,
        so we don't limit element count too strictly. The CrMnFeCoNi Cantor
        alloy has 5 elements, and complex HEAs with 10+ elements exist.
        """
        issues = []

        total_atoms = sum(stoichiometry.values())
        n_elements = len(stoichiometry)

        # High-entropy alloys can have many elements - only flag extreme cases
        # 20+ elements would be unprecedented, but 15 is within possibility
        if n_elements > 20:
            issues.append(f"Extremely high element count ({n_elements}) - unprecedented even for HEAs")
            return False, issues
        elif n_elements > 15:
            issues.append(f"Very high element count ({n_elements}) - would require careful synthesis (HEA-like)")
            # Warning but not invalid - HEAs exist with 10+ elements

        # Very large formula unit
        if total_atoms > 200:
            issues.append(f"Very large formula unit ({total_atoms} atoms) - may indicate parsing error")
            return False, issues
        elif total_atoms > 100:
            issues.append(f"Large formula unit ({total_atoms} atoms) - complex structure")
            # Warning but not necessarily invalid

        # Check for fractional stoichiometry (should be integers or simple fractions)
        for elem, count in stoichiometry.items():
            if count > 0 and count < 0.5:
                issues.append(f"Very small stoichiometry for {elem}: {count:.2f}")

        return len(issues) == 0 or n_elements <= 20, issues

    def _suggest_synthesis_routes(
        self,
        elements: Set[str],
        no_immiscible: bool,
        bond_compatible: bool,
        hr_compatible: bool
    ) -> List[str]:
        """
        Suggest potential synthesis routes for exotic compositions.

        When a composition has issues, there may still be exotic synthesis
        routes that could enable metastable phases or new compounds.
        """
        routes = []

        # Check for conditions that might benefit from special synthesis
        alkali = elements & self.ALKALI_METALS
        has_h = 'H' in elements
        has_lanthanides = bool(elements & self.LANTHANIDES)

        # Multiple alkali metals - extreme pressure might help
        if len(alkali) > 1:
            routes.append(
                "High-pressure synthesis (>50-100 GPa): Pressure can stabilize "
                "unusual alkali metal arrangements by modifying electronic structure"
            )

        # Immiscible pairs - rapid quenching or thin films
        if not no_immiscible:
            routes.append(
                "Rapid solidification / melt spinning: Can trap metastable phases "
                "by kinetically avoiding equilibrium immiscibility"
            )
            routes.append(
                "Thin film deposition (PLD, MBE, sputtering): Non-equilibrium growth "
                "can enable metastable alloys from immiscible components"
            )

        # Hume-Rothery violations - mechanical alloying
        if not hr_compatible:
            routes.append(
                "Mechanical alloying / ball milling: Can force solid solution "
                "formation even when Hume-Rothery rules are violated"
            )

        # Hydrogen-rich - high pressure hydrides
        if has_h and sum(1 for e in elements if e != 'H') <= 3:
            routes.append(
                "High-pressure hydrogen loading (DAC): Superhydrides like LaH10 "
                "are synthesized at >100 GPa; may stabilize unusual stoichiometries"
            )

        # Lanthanide compounds - may need special atmosphere
        if has_lanthanides:
            routes.append(
                "Inert atmosphere (Ar glovebox): Lanthanides oxidize rapidly; "
                "arc melting in Ar or vacuum synthesis may be required"
            )

        # High-entropy alloy route for many elements
        if len(elements) >= 5:
            routes.append(
                "High-entropy alloy (HEA) synthesis: Arc melting, induction melting, "
                "or powder metallurgy for equiatomic multi-element alloys"
            )

        return routes


def validate_physics(formula: str, stoichiometry: Dict[str, float]) -> PhysicsValidationResult:
    """Convenience function for physics validation."""
    validator = PhysicsValidator()
    return validator.validate(formula, stoichiometry)
