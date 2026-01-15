"""
Isotope-aware composition encoder for superconductors.

Encodes chemical formulas with isotope-specific features including:
- Exact atomic masses (not average)
- Nuclear spin contributions
- Natural abundance weighting
- Mass deviation from natural average

Example:
    encoder = IsotopeEncoder()

    # Standard formula (uses natural abundance weighted isotopes)
    result = encoder.encode("YBa2Cu3O7")

    # Isotope-specific formula
    result = encoder.encode("LaD10")  # Deuterium
    result = encoder.encode("Y(18O)Ba2Cu3O6")  # Oxygen-18
"""

import torch
import torch.nn as nn
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .isotope_properties import (
    ISOTOPE_DATABASE,
    get_isotope,
    get_most_abundant_isotope,
    get_all_isotopes,
    parse_isotope_notation,
    IsotopeData,
)
from .element_properties import ELEMENT_PROPERTIES


@dataclass
class IsotopeEncodedComposition:
    """Result of isotope-aware encoding."""

    # Element-level features (one per element in formula)
    element_indices: np.ndarray         # [n_elements] atomic numbers
    element_fractions: np.ndarray       # [n_elements] molar fractions
    element_masses: np.ndarray          # [n_elements] isotope-specific masses
    element_spins: np.ndarray           # [n_elements] nuclear spins
    element_abundances: np.ndarray      # [n_elements] isotope abundances
    mass_deviations: np.ndarray         # [n_elements] deviation from natural mass

    # Aggregated features
    total_mass: float                   # Total formula mass
    avg_mass: float                     # Average atomic mass
    weighted_spin: float                # Molar-fraction weighted spin
    isotope_effect_factor: float        # Estimated isotope effect (M^-0.5)

    # Full composition vector (for compatibility)
    composition_vector: np.ndarray      # [118] fractional composition
    isotope_features: np.ndarray        # [4] aggregated isotope features

    # Metadata
    formula: str                        # Original formula
    parsed_elements: Dict[str, float]   # Element -> count mapping
    isotope_info: Dict[str, Tuple[int, float]]  # Element -> (mass_number, fraction)


class IsotopeFormulaParser:
    """
    Parse chemical formulas with isotope notation.

    Supports:
    - Standard: YBa2Cu3O7
    - Superscript: Y(¹⁸O)Ba₂Cu₃O₆
    - Prefix mass: Y(18O)Ba2Cu3O6
    - Special: D (deuterium), T (tritium)
    - Mixed: LaD10, Y(18O)7
    """

    # Superscript digit mapping
    SUPERSCRIPT_MAP = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }

    # Subscript digit mapping
    SUBSCRIPT_MAP = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }

    # Atomic numbers
    ATOMIC_NUMBERS = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
        'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
        'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
        'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118,
        # Special isotope names
        'D': 1,  # Deuterium (hydrogen-2)
        'T': 1,  # Tritium (hydrogen-3)
    }

    def __init__(self):
        pass

    def normalize_formula(self, formula: str) -> str:
        """Convert superscript/subscript notation to standard ASCII."""
        result = formula

        # Convert superscripts (isotope masses)
        for sup, digit in self.SUPERSCRIPT_MAP.items():
            result = result.replace(sup, digit)

        # Convert subscripts (element counts)
        for sub, digit in self.SUBSCRIPT_MAP.items():
            result = result.replace(sub, digit)

        return result

    def parse(self, formula: str) -> Tuple[Dict[str, float], Dict[str, Tuple[int, float]]]:
        """
        Parse formula into element counts and isotope info.

        Returns:
            elements: Dict[element_symbol, count]
            isotopes: Dict[element_symbol, (mass_number, fraction)]
                      mass_number=0 means natural abundance
        """
        formula = self.normalize_formula(formula)
        elements = {}
        isotopes = {}

        # Step 1: Handle parenthesized groups with isotopes: (18O), (2H), etc.
        paren_pattern = r'\((\d+)([A-Z][a-z]?)\)(\d*\.?\d*)'
        for match in re.finditer(paren_pattern, formula):
            mass_num = int(match.group(1))
            symbol = match.group(2)
            count = float(match.group(3)) if match.group(3) else 1.0

            elements[symbol] = elements.get(symbol, 0) + count
            isotopes[symbol] = (mass_num, count)

        # Remove parenthesized groups
        formula_clean = re.sub(paren_pattern, '', formula)

        # Step 2: Handle special isotope names D and T
        # These are special: D = deuterium (H-2), T = tritium (H-3)
        # They appear as standalone symbols in formulas like "LaD10"
        # We need to handle them BEFORE standard element parsing
        # because "D" would otherwise not match as an element

        # First, temporarily replace D and T with placeholders
        # Pattern: D or T followed by digits (LaD10 -> La{D}10)
        # We match D/T when they're uppercase and followed by digit or end
        d_pattern = r'D(\d+\.?\d*)?(?![a-z])'  # D not followed by lowercase (so not Dy)
        t_pattern = r'T(\d+\.?\d*)?(?![a-z])'  # T not followed by lowercase (so not Ti, Tm, etc)

        for match in re.finditer(d_pattern, formula_clean):
            count = float(match.group(1)) if match.group(1) else 1.0
            elements['H'] = elements.get('H', 0) + count
            isotopes['H'] = (2, elements['H'])  # Deuterium

        for match in re.finditer(t_pattern, formula_clean):
            count = float(match.group(1)) if match.group(1) else 1.0
            elements['H'] = elements.get('H', 0) + count
            isotopes['H'] = (3, elements['H'])  # Tritium

        # Remove D and T from formula
        formula_clean = re.sub(d_pattern, '', formula_clean)
        formula_clean = re.sub(t_pattern, '', formula_clean)

        # Step 3: Handle explicit isotope notation at START of formula: 13C6H12O6
        # Pattern: digits (2+) followed by element at formula start
        iso_start_pattern = r'^(\d{2,})([A-Z][a-z]?)(\d*\.?\d*)'
        match = re.match(iso_start_pattern, formula_clean)
        if match:
            mass_num = int(match.group(1))
            symbol = match.group(2)
            count = float(match.group(3)) if match.group(3) else 1.0
            elements[symbol] = elements.get(symbol, 0) + count
            isotopes[symbol] = (mass_num, count)
            formula_clean = formula_clean[match.end():]

        # Step 4: Parse standard elements: Cu3, O7, Y, Ba2
        # Match element symbol (1 uppercase + optional lowercase) followed by optional count
        std_pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        for match in re.finditer(std_pattern, formula_clean):
            symbol = match.group(1)
            count_str = match.group(2)
            count = float(count_str) if count_str else 1.0

            if symbol:
                # Add to elements (may already have isotope-specified amount)
                if symbol in elements and symbol in isotopes and isotopes[symbol][0] > 0:
                    # This element already has isotope info - ADD to count
                    # (e.g., Y(18O)Ba2Cu3O6 - O already has 1 from 18O, add 6 more)
                    elements[symbol] = elements.get(symbol, 0) + count
                    # Keep isotope info but update count
                    mass_num = isotopes[symbol][0]
                    isotopes[symbol] = (mass_num, elements[symbol])
                else:
                    elements[symbol] = elements.get(symbol, 0) + count
                    if symbol not in isotopes:
                        isotopes[symbol] = (0, count)  # 0 = natural abundance
                    else:
                        # Update count in isotopes
                        mass_num = isotopes[symbol][0]
                        isotopes[symbol] = (mass_num, elements[symbol])

        return elements, isotopes

    def get_atomic_number(self, symbol: str) -> int:
        """Get atomic number for element symbol."""
        return self.ATOMIC_NUMBERS.get(symbol, 0)


class IsotopeEncoder:
    """
    Encode chemical compositions with isotope-aware features.

    For each element in a formula, extracts:
    - Exact isotope mass (or abundance-weighted if not specified)
    - Nuclear spin
    - Mass deviation from natural average
    - Abundance (1.0 if specific isotope, otherwise natural)

    These features enable the model to learn isotope effects on Tc.
    """

    def __init__(
        self,
        max_elements: int = 118,
        include_element_properties: bool = True,
        default_to_abundant: bool = True
    ):
        """
        Args:
            max_elements: Maximum atomic number to support
            include_element_properties: Include elemental properties (electronegativity, etc.)
            default_to_abundant: Use most abundant isotope when not specified
        """
        self.max_elements = max_elements
        self.include_element_properties = include_element_properties
        self.default_to_abundant = default_to_abundant
        self.parser = IsotopeFormulaParser()

        # Precompute natural average masses for each element
        self._natural_masses = self._compute_natural_masses()

    def _compute_natural_masses(self) -> Dict[str, float]:
        """Compute natural abundance weighted masses."""
        natural_masses = {}

        for symbol, isotopes in ISOTOPE_DATABASE.items():
            total_mass = 0.0
            total_abundance = 0.0

            for mass_num, iso_data in isotopes.items():
                if iso_data.natural_abundance > 0:
                    total_mass += iso_data.atomic_mass * iso_data.natural_abundance
                    total_abundance += iso_data.natural_abundance

            if total_abundance > 0:
                natural_masses[symbol] = total_mass / total_abundance
            elif isotopes:
                # No natural isotopes, use most abundant or first
                most_abundant = max(isotopes.values(), key=lambda x: x.natural_abundance)
                natural_masses[symbol] = most_abundant.atomic_mass

        return natural_masses

    def _get_isotope_data(
        self,
        symbol: str,
        mass_number: int
    ) -> Tuple[float, float, float]:
        """
        Get isotope data: (mass, spin, abundance).

        Args:
            symbol: Element symbol
            mass_number: Isotope mass number (0 for natural abundance)

        Returns:
            (atomic_mass, nuclear_spin, abundance)
        """
        if mass_number > 0:
            # Specific isotope requested
            isotope = get_isotope(symbol, mass_number)
            if isotope:
                return (isotope.atomic_mass, isotope.nuclear_spin, isotope.natural_abundance)

        # Use most abundant isotope or natural average
        if self.default_to_abundant:
            abundant = get_most_abundant_isotope(symbol)
            if abundant:
                return (abundant.atomic_mass, abundant.nuclear_spin, abundant.natural_abundance)

        # Fallback to natural mass with spin=0
        natural_mass = self._natural_masses.get(symbol, 0.0)
        return (natural_mass, 0.0, 1.0)

    def encode(self, formula: str) -> IsotopeEncodedComposition:
        """
        Encode a chemical formula with isotope features.

        Args:
            formula: Chemical formula (e.g., "YBa2Cu3O7", "LaD10", "Y(18O)Ba2Cu3O6")

        Returns:
            IsotopeEncodedComposition with element-level and aggregated features
        """
        # Parse formula
        elements, isotope_info = self.parser.parse(formula)

        if not elements:
            raise ValueError(f"Could not parse formula: {formula}")

        # Compute total atoms for normalization
        total_atoms = sum(elements.values())

        # Collect element-level data
        n_elements = len(elements)
        element_indices = np.zeros(n_elements, dtype=np.int32)
        element_fractions = np.zeros(n_elements, dtype=np.float32)
        element_masses = np.zeros(n_elements, dtype=np.float32)
        element_spins = np.zeros(n_elements, dtype=np.float32)
        element_abundances = np.zeros(n_elements, dtype=np.float32)
        mass_deviations = np.zeros(n_elements, dtype=np.float32)

        for i, (symbol, count) in enumerate(elements.items()):
            # Get atomic number
            atomic_num = self.parser.get_atomic_number(symbol)
            element_indices[i] = atomic_num

            # Get fraction
            element_fractions[i] = count / total_atoms

            # Get isotope info
            mass_num, _ = isotope_info.get(symbol, (0, count))
            mass, spin, abundance = self._get_isotope_data(symbol, mass_num)

            element_masses[i] = mass
            element_spins[i] = spin
            element_abundances[i] = abundance if mass_num > 0 else 1.0

            # Mass deviation from natural
            natural = self._natural_masses.get(symbol, mass)
            if natural > 0:
                mass_deviations[i] = (mass - natural) / natural

        # Compute aggregated features
        total_mass = np.sum(element_masses * element_fractions * total_atoms)
        avg_mass = np.mean(element_masses)
        weighted_spin = np.sum(element_spins * element_fractions)

        # Isotope effect factor: M^(-0.5) normalized to natural
        # Higher = lighter isotopes (should increase Tc for BCS)
        natural_mass_sum = sum(
            self._natural_masses.get(s, 0) * c
            for s, c in elements.items()
        )
        if natural_mass_sum > 0 and total_mass > 0:
            isotope_effect_factor = (natural_mass_sum / total_mass) ** 0.5
        else:
            isotope_effect_factor = 1.0

        # Create full composition vector [118]
        composition_vector = np.zeros(self.max_elements, dtype=np.float32)
        for symbol, count in elements.items():
            atomic_num = self.parser.get_atomic_number(symbol)
            if 1 <= atomic_num <= self.max_elements:
                composition_vector[atomic_num - 1] = count / total_atoms

        # Aggregated isotope features [4]
        isotope_features = np.array([
            avg_mass / 200.0,  # Normalized average mass
            weighted_spin,     # Weighted nuclear spin
            np.mean(mass_deviations),  # Average mass deviation
            isotope_effect_factor - 1.0  # Isotope effect (0 = natural)
        ], dtype=np.float32)

        return IsotopeEncodedComposition(
            element_indices=element_indices,
            element_fractions=element_fractions,
            element_masses=element_masses,
            element_spins=element_spins,
            element_abundances=element_abundances,
            mass_deviations=mass_deviations,
            total_mass=float(total_mass),
            avg_mass=float(avg_mass),
            weighted_spin=float(weighted_spin),
            isotope_effect_factor=float(isotope_effect_factor),
            composition_vector=composition_vector,
            isotope_features=isotope_features,
            formula=formula,
            parsed_elements=elements,
            isotope_info=isotope_info
        )

    def encode_batch(
        self,
        formulas: List[str],
        max_elements_per_formula: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple formulas as batch tensors.

        Args:
            formulas: List of chemical formulas
            max_elements_per_formula: Padding size for element arrays

        Returns:
            Dict of tensors ready for model input
        """
        batch_size = len(formulas)

        # Initialize tensors
        element_indices = torch.zeros(batch_size, max_elements_per_formula, dtype=torch.long)
        element_fractions = torch.zeros(batch_size, max_elements_per_formula)
        element_masses = torch.zeros(batch_size, max_elements_per_formula)
        element_spins = torch.zeros(batch_size, max_elements_per_formula)
        element_mask = torch.zeros(batch_size, max_elements_per_formula, dtype=torch.bool)
        composition_vectors = torch.zeros(batch_size, self.max_elements)
        isotope_features = torch.zeros(batch_size, 4)

        for i, formula in enumerate(formulas):
            try:
                encoded = self.encode(formula)
                n_elem = len(encoded.element_indices)

                element_indices[i, :n_elem] = torch.tensor(encoded.element_indices)
                element_fractions[i, :n_elem] = torch.tensor(encoded.element_fractions)
                element_masses[i, :n_elem] = torch.tensor(encoded.element_masses)
                element_spins[i, :n_elem] = torch.tensor(encoded.element_spins)
                element_mask[i, :n_elem] = True
                composition_vectors[i] = torch.tensor(encoded.composition_vector)
                isotope_features[i] = torch.tensor(encoded.isotope_features)

            except Exception as e:
                # Log but continue - allow partial batch encoding
                print(f"Warning: Could not encode {formula}: {e}")

        return {
            'element_indices': element_indices,
            'element_fractions': element_fractions,
            'element_masses': element_masses,
            'element_spins': element_spins,
            'element_mask': element_mask,
            'composition_vector': composition_vectors,
            'isotope_features': isotope_features
        }


class IsotopeAwareEncoder(nn.Module):
    """
    Neural network encoder that incorporates isotope information.

    Combines:
    - Composition vector encoding
    - Isotope-specific features
    - Element-level attention

    Output is a feature vector suitable for BidirectionalVAE or other models.
    """

    def __init__(
        self,
        composition_dim: int = 118,
        isotope_feature_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 64,
        n_attention_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            composition_dim: Dimension of composition vector (usually 118)
            isotope_feature_dim: Dimension of isotope features
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            n_attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.composition_dim = composition_dim
        self.isotope_feature_dim = isotope_feature_dim

        # Composition encoder
        self.composition_encoder = nn.Sequential(
            nn.Linear(composition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Isotope feature encoder
        self.isotope_encoder = nn.Sequential(
            nn.Linear(isotope_feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Element-level attention (imported from element_attention.py)
        from .element_attention import ElementEmbedding, ElementAttention

        self.element_embedding = ElementEmbedding(
            n_elements=composition_dim,
            embedding_dim=hidden_dim // 2,
            use_properties=False  # Properties included in isotope features
        )

        self.element_attention = ElementAttention(
            hidden_dim=hidden_dim // 2,
            n_heads=n_attention_heads,
            dropout=dropout
        )

        # Combine all features
        # composition (hidden) + isotope (hidden) + attention (hidden/2)
        combined_dim = hidden_dim * 2 + hidden_dim // 2

        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_dim = output_dim

    def forward(
        self,
        composition_vector: torch.Tensor,
        isotope_features: torch.Tensor,
        element_indices: Optional[torch.Tensor] = None,
        element_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode composition with isotope awareness.

        Args:
            composition_vector: [batch, 118] fractional composition
            isotope_features: [batch, 4] aggregated isotope features
            element_indices: [batch, n_elements] atomic numbers (optional)
            element_mask: [batch, n_elements] valid element mask (optional)

        Returns:
            features: [batch, output_dim] encoded features
            attention_weights: [batch, n_elements] or None
        """
        # Encode composition
        comp_features = self.composition_encoder(composition_vector)

        # Encode isotope features
        iso_features = self.isotope_encoder(isotope_features)

        # Element attention (if indices provided)
        attention_weights = None
        if element_indices is not None:
            elem_embeds = self.element_embedding(element_indices)
            attn_output = self.element_attention(elem_embeds, element_mask)
            attn_features = attn_output.weighted_representation
            attention_weights = attn_output.attention_weights
        else:
            # Default: zero attention features
            batch_size = composition_vector.shape[0]
            attn_features = torch.zeros(
                batch_size, self.element_embedding.embedding_dim,
                device=composition_vector.device
            )

        # Combine all features
        combined = torch.cat([comp_features, iso_features, attn_features], dim=-1)
        output = self.combiner(combined)

        return output, attention_weights


def create_isotope_encoder() -> IsotopeEncoder:
    """Factory function to create default isotope encoder."""
    return IsotopeEncoder(
        max_elements=118,
        include_element_properties=True,
        default_to_abundant=True
    )
