"""
Chemical composition encoder for superconductor materials.

Converts chemical formulas (e.g., "YBa2Cu3O7") into numerical feature vectors
for machine learning. Supports:
- Composition vectors (element fractions)
- Weighted element statistics (mean/std of properties)
- Stoichiometry-aware encoding
- Doping notation (e.g., "YBa2Cu3O7-δ")
"""

import re
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from .element_properties import (
    ELEMENT_PROPERTIES,
    PROPERTY_KEYS,
    SYMBOL_TO_Z,
    ELEMENT_SYMBOLS,
    get_element_property,
    get_atomic_number,
)


@dataclass
class EncodedComposition:
    """Container for encoded chemical composition."""
    formula: str
    composition_vector: torch.Tensor  # [max_elements] element fractions
    element_stats: torch.Tensor  # [n_properties * 2] mean/std stats
    element_count: int  # Number of unique elements
    total_atoms: float  # Total stoichiometry
    elements_present: List[str]  # List of element symbols
    stoichiometry: Dict[str, float]  # Element -> count mapping


class CompositionEncoder:
    """
    Encode chemical formulas to numerical feature vectors.

    Example:
        encoder = CompositionEncoder()
        encoded = encoder.encode("YBa2Cu3O7")
        print(encoded.composition_vector.shape)  # [118]
        print(encoded.element_stats.shape)  # [22]  (11 props * 2 for mean/std)
    """

    def __init__(
        self,
        max_elements: int = 118,
        properties: Optional[List[str]] = None,
        normalize_composition: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize composition encoder.

        Args:
            max_elements: Maximum atomic number to support (default 118)
            properties: List of property keys to use for statistics
            normalize_composition: Whether to normalize composition to sum to 1
            device: Torch device for tensors
        """
        self.max_elements = max_elements
        self.properties = properties or PROPERTY_KEYS
        self.normalize_composition = normalize_composition
        self.device = device

        # Precompute property normalization statistics
        self._compute_property_normalization()

    def _compute_property_normalization(self):
        """Compute mean/std for each property across all elements."""
        self.property_means = {}
        self.property_stds = {}

        for prop in self.properties:
            values = []
            for symbol, props in ELEMENT_PROPERTIES.items():
                if prop in props and props[prop] is not None:
                    val = props[prop]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        values.append(val)

            if values:
                self.property_means[prop] = np.mean(values)
                self.property_stds[prop] = np.std(values) + 1e-8
            else:
                self.property_means[prop] = 0.0
                self.property_stds[prop] = 1.0

    def parse_formula(self, formula: str) -> Dict[str, float]:
        """
        Parse chemical formula to element-count dictionary.

        Handles:
        - Basic formulas: "NaCl", "H2O", "YBa2Cu3O7"
        - Parentheses: "Ca(OH)2", "(NH4)2SO4"
        - Decimals: "La0.85Sr0.15CuO4"
        - Doping notation: "YBa2Cu3O7-x" (treats x as small value)

        Args:
            formula: Chemical formula string

        Returns:
            Dictionary mapping element symbols to counts
        """
        # Remove spaces and common annotations
        formula = formula.strip()
        formula = re.sub(r'\s+', '', formula)

        # Handle doping notation (δ, x, y, z at end)
        formula = re.sub(r'[-+]?[δxyzn]$', '', formula, flags=re.IGNORECASE)
        formula = re.sub(r'[-+]?\d*\.\d+$', '', formula)  # Remove trailing decimals

        return self._parse_formula_recursive(formula)

    def _parse_formula_recursive(self, formula: str) -> Dict[str, float]:
        """Recursively parse formula handling parentheses."""
        counts: Dict[str, float] = {}

        i = 0
        while i < len(formula):
            if formula[i] == '(':
                # Find matching closing parenthesis
                depth = 1
                j = i + 1
                while j < len(formula) and depth > 0:
                    if formula[j] == '(':
                        depth += 1
                    elif formula[j] == ')':
                        depth -= 1
                    j += 1

                # Get content inside parentheses
                inner = formula[i + 1:j - 1]
                inner_counts = self._parse_formula_recursive(inner)

                # Get multiplier after closing parenthesis
                k = j
                while k < len(formula) and (formula[k].isdigit() or formula[k] == '.'):
                    k += 1
                multiplier = float(formula[j:k]) if j < k else 1.0

                # Add to counts
                for elem, count in inner_counts.items():
                    counts[elem] = counts.get(elem, 0) + count * multiplier

                i = k
            elif formula[i].isupper():
                # Element symbol
                j = i + 1
                while j < len(formula) and formula[j].islower():
                    j += 1
                element = formula[i:j]

                # Get count
                k = j
                while k < len(formula) and (formula[k].isdigit() or formula[k] == '.'):
                    k += 1
                count = float(formula[j:k]) if j < k else 1.0

                if element in SYMBOL_TO_Z:
                    counts[element] = counts.get(element, 0) + count
                # Skip unknown elements silently

                i = k
            else:
                i += 1

        return counts

    def encode(self, formula: str) -> EncodedComposition:
        """
        Encode a chemical formula to feature vectors.

        Args:
            formula: Chemical formula string (e.g., "YBa2Cu3O7")

        Returns:
            EncodedComposition with composition vector and element statistics
        """
        # Parse formula
        stoichiometry = self.parse_formula(formula)

        if not stoichiometry:
            # Return zero encoding for unparseable formulas
            return EncodedComposition(
                formula=formula,
                composition_vector=torch.zeros(self.max_elements, device=self.device),
                element_stats=torch.zeros(len(self.properties) * 2, device=self.device),
                element_count=0,
                total_atoms=0.0,
                elements_present=[],
                stoichiometry={}
            )

        # Compute total atoms
        total_atoms = sum(stoichiometry.values())

        # Build composition vector
        composition_vector = torch.zeros(self.max_elements, device=self.device)
        for element, count in stoichiometry.items():
            z = SYMBOL_TO_Z.get(element)
            if z and z <= self.max_elements:
                if self.normalize_composition:
                    composition_vector[z - 1] = count / total_atoms
                else:
                    composition_vector[z - 1] = count

        # Compute weighted element statistics
        element_stats = self._compute_weighted_stats(stoichiometry, total_atoms)

        return EncodedComposition(
            formula=formula,
            composition_vector=composition_vector,
            element_stats=element_stats,
            element_count=len(stoichiometry),
            total_atoms=total_atoms,
            elements_present=list(stoichiometry.keys()),
            stoichiometry=stoichiometry
        )

    def _compute_weighted_stats(
        self,
        stoichiometry: Dict[str, float],
        total_atoms: float
    ) -> torch.Tensor:
        """
        Compute weighted mean and std of element properties.

        Args:
            stoichiometry: Element -> count mapping
            total_atoms: Total stoichiometry for normalization

        Returns:
            Tensor of [mean_prop1, mean_prop2, ..., std_prop1, std_prop2, ...]
        """
        n_props = len(self.properties)
        stats = torch.zeros(n_props * 2, device=self.device)

        for i, prop in enumerate(self.properties):
            values = []
            weights = []

            for element, count in stoichiometry.items():
                props = ELEMENT_PROPERTIES.get(element, {})
                val = props.get(prop)

                if val is not None and isinstance(val, (int, float)):
                    # Normalize property value
                    normalized_val = (val - self.property_means[prop]) / self.property_stds[prop]
                    values.append(normalized_val)
                    weights.append(count / total_atoms)

            if values:
                values = np.array(values)
                weights = np.array(weights)
                weights = weights / weights.sum()  # Ensure weights sum to 1

                # Weighted mean
                mean = np.sum(values * weights)

                # Weighted std
                variance = np.sum(weights * (values - mean) ** 2)
                std = np.sqrt(variance)

                stats[i] = mean
                stats[n_props + i] = std

        return stats

    def encode_batch(
        self,
        formulas: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[EncodedComposition]]:
        """
        Encode a batch of chemical formulas.

        Args:
            formulas: List of chemical formula strings

        Returns:
            Tuple of:
            - composition_vectors: [batch, max_elements]
            - element_stats: [batch, n_properties * 2]
            - encoded_list: List of EncodedComposition objects
        """
        encoded_list = [self.encode(f) for f in formulas]

        composition_vectors = torch.stack([e.composition_vector for e in encoded_list])
        element_stats = torch.stack([e.element_stats for e in encoded_list])

        return composition_vectors, element_stats, encoded_list

    def get_feature_dim(self) -> int:
        """Get total feature dimension (composition + stats)."""
        return self.max_elements + len(self.properties) * 2

    def get_composition_dim(self) -> int:
        """Get composition vector dimension."""
        return self.max_elements

    def get_stats_dim(self) -> int:
        """Get element statistics dimension."""
        return len(self.properties) * 2


class CompositionDecoder:
    """
    Decode composition vectors back to chemical formulas.

    Used for candidate generation from latent space.
    """

    def __init__(
        self,
        max_elements: int = 118,
        threshold: float = 0.01,
        max_stoichiometry: int = 12
    ):
        """
        Initialize decoder.

        Args:
            max_elements: Maximum atomic number
            threshold: Minimum fraction to include element
            max_stoichiometry: Maximum stoichiometric coefficient
        """
        self.max_elements = max_elements
        self.threshold = threshold
        self.max_stoichiometry = max_stoichiometry

        # Electronegativity order for conventional formula writing
        self._build_electronegativity_order()

    def _build_electronegativity_order(self):
        """Build ordering for conventional formula writing."""
        self.electronegativity = {}
        for symbol, props in ELEMENT_PROPERTIES.items():
            en = props.get('electronegativity', 2.0)
            self.electronegativity[symbol] = en if en else 2.0

    def decode(
        self,
        composition_vector: Union[torch.Tensor, np.ndarray],
        round_stoichiometry: bool = True
    ) -> str:
        """
        Decode composition vector to chemical formula.

        Args:
            composition_vector: [max_elements] tensor of element fractions
            round_stoichiometry: Whether to round to integer stoichiometry

        Returns:
            Chemical formula string
        """
        if isinstance(composition_vector, torch.Tensor):
            composition_vector = composition_vector.detach().cpu().numpy()

        # Ensure positive
        composition_vector = np.abs(composition_vector)

        # Find significant elements
        elements = []
        for i in range(min(len(composition_vector), self.max_elements)):
            frac = composition_vector[i]
            if frac > self.threshold:
                symbol = ELEMENT_SYMBOLS[i + 1]  # 1-indexed
                if symbol:
                    elements.append((symbol, frac))

        if not elements:
            return ""

        # Convert fractions to stoichiometry
        fractions = [e[1] for e in elements]
        min_frac = min(fractions)

        stoichiometry = []
        for symbol, frac in elements:
            ratio = frac / min_frac
            if round_stoichiometry:
                count = max(1, min(self.max_stoichiometry, round(ratio)))
            else:
                count = ratio
            stoichiometry.append((symbol, count, frac))

        # Sort by electronegativity (cation before anion)
        stoichiometry.sort(key=lambda x: self.electronegativity.get(x[0], 2.0))

        # Build formula string
        formula = ""
        for symbol, count, _ in stoichiometry:
            formula += symbol
            if round_stoichiometry:
                if count != 1:
                    formula += str(int(count))
            else:
                if abs(count - round(count)) < 0.01:
                    if round(count) != 1:
                        formula += str(int(round(count)))
                else:
                    formula += f"{count:.2f}"

        return formula

    def decode_batch(
        self,
        composition_vectors: Union[torch.Tensor, np.ndarray],
        round_stoichiometry: bool = True
    ) -> List[str]:
        """Decode batch of composition vectors."""
        if isinstance(composition_vectors, torch.Tensor):
            composition_vectors = composition_vectors.detach().cpu().numpy()

        return [self.decode(cv, round_stoichiometry) for cv in composition_vectors]


# Convenience functions
def encode_formula(formula: str, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quick encoding of a single formula.

    Returns:
        Tuple of (composition_vector, element_stats)
    """
    encoder = CompositionEncoder(device=device)
    encoded = encoder.encode(formula)
    return encoded.composition_vector, encoded.element_stats


def decode_composition(
    composition_vector: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.01
) -> str:
    """Quick decoding of composition vector to formula."""
    decoder = CompositionDecoder(threshold=threshold)
    return decoder.decode(composition_vector)
