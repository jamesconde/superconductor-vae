"""
Superconductor Family Classifier.

Classifies superconductors into families based on their dominant
physical mechanism/theory:
- BCS Conventional (phonon-mediated)
- Cuprate (YBCO, LSCO, Bi-based, Tl-based, Hg-based)
- Iron-based (pnictides, chalcogenides)
- MgB2-type (two-gap BCS)
- Heavy Fermion (f-electron)
- Other/Unknown

Can operate in two modes:
1. Rule-based: Deterministic classification from formula composition
2. Learned: Neural network predicts family from latent/Magpie features

February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import IntEnum


class SuperconductorFamily(IntEnum):
    """Superconductor family/mechanism categories."""

    # Non-superconductor (for completeness)
    NOT_SUPERCONDUCTOR = 0

    # BCS/conventional (phonon-mediated, s-wave)
    BCS_CONVENTIONAL = 1

    # Cuprates (d-wave, CuO2 planes)
    CUPRATE_YBCO = 2      # YBa2Cu3O7 family
    CUPRATE_LSCO = 3      # La2-xSrxCuO4 family
    CUPRATE_BSCCO = 4     # Bi2Sr2Can-1CunO2n+4 family
    CUPRATE_TBCCO = 5     # Tl-based cuprates
    CUPRATE_HBCCO = 6     # Hg-based cuprates
    CUPRATE_OTHER = 7     # Other cuprates

    # Iron-based (multi-band, Fermi surface nesting)
    IRON_PNICTIDE = 8     # FeAs-based (1111, 122, 111)
    IRON_CHALCOGENIDE = 9  # FeSe, FeTe-based

    # Other known mechanisms
    MGB2_TYPE = 10        # MgB2 and related (two-gap BCS)
    HEAVY_FERMION = 11    # f-electron systems (UPt3, CeCoIn5, etc.)
    ORGANIC = 12          # Organic superconductors (BEDT-TTF, etc.)

    # Unknown/other - for materials that don't fit known categories
    OTHER_UNKNOWN = 13


# Mapping for simpler 5-class version (for initial experiments)
FAMILY_TO_SIMPLE = {
    SuperconductorFamily.NOT_SUPERCONDUCTOR: 0,
    SuperconductorFamily.BCS_CONVENTIONAL: 1,
    SuperconductorFamily.CUPRATE_YBCO: 2,
    SuperconductorFamily.CUPRATE_LSCO: 2,
    SuperconductorFamily.CUPRATE_BSCCO: 2,
    SuperconductorFamily.CUPRATE_TBCCO: 2,
    SuperconductorFamily.CUPRATE_HBCCO: 2,
    SuperconductorFamily.CUPRATE_OTHER: 2,
    SuperconductorFamily.IRON_PNICTIDE: 3,
    SuperconductorFamily.IRON_CHALCOGENIDE: 3,
    SuperconductorFamily.MGB2_TYPE: 1,  # BCS-like
    SuperconductorFamily.HEAVY_FERMION: 4,
    SuperconductorFamily.ORGANIC: 4,
    SuperconductorFamily.OTHER_UNKNOWN: 5,
}

SIMPLE_FAMILY_NAMES = [
    'not_superconductor',
    'bcs_conventional',
    'cuprate',
    'iron_based',
    'other_mechanism',
    'unknown',
]


@dataclass
class FamilyClassifierConfig:
    """Configuration for family classifier."""

    # Number of classes (14 for full, 6 for simple)
    num_classes: int = 14

    # Use simple 6-class version
    use_simple_classes: bool = False

    # Hidden dimension for learned classifier
    hidden_dim: int = 256

    # Dropout rate
    dropout: float = 0.1


# Element sets for rule-based classification
CUPRATE_ELEMENTS = {'Cu', 'O'}
CUPRATE_YBCO_ELEMENTS = {'Y', 'Ba', 'Cu', 'O'}
CUPRATE_LSCO_ELEMENTS = {'La', 'Sr', 'Cu', 'O'}
CUPRATE_BSCCO_ELEMENTS = {'Bi', 'Sr', 'Ca', 'Cu', 'O'}
CUPRATE_TBCCO_ELEMENTS = {'Tl', 'Ba', 'Ca', 'Cu', 'O'}
CUPRATE_HBCCO_ELEMENTS = {'Hg', 'Ba', 'Ca', 'Cu', 'O'}

IRON_PNICTIDE_ELEMENTS = {'Fe', 'As'}
IRON_CHALCOGENIDE_ELEMENTS = {'Fe', 'Se', 'Te'}

MGB2_ELEMENTS = {'Mg', 'B'}

HEAVY_FERMION_ELEMENTS = {'U', 'Ce', 'Yb', 'Pu'}  # f-block elements

ORGANIC_ELEMENTS = {'C', 'H', 'N', 'S'}  # Characteristic organic elements


class RuleBasedFamilyClassifier:
    """
    Rule-based superconductor family classifier.

    Uses elemental composition to classify into families.
    Deterministic - no learned parameters.
    """

    def __init__(self, config: Optional[FamilyClassifierConfig] = None):
        self.config = config or FamilyClassifierConfig()

    def classify_from_elements(
        self,
        elements: Set[str],
        fractions: Optional[Dict[str, float]] = None,
    ) -> SuperconductorFamily:
        """
        Classify from element set.

        Args:
            elements: Set of element symbols in the formula
            fractions: Optional dict of element -> molar fraction

        Returns:
            SuperconductorFamily enum value
        """
        # Check for cuprates (Cu + O required)
        if CUPRATE_ELEMENTS.issubset(elements):
            # Determine specific cuprate family
            if 'Y' in elements and 'Ba' in elements:
                return SuperconductorFamily.CUPRATE_YBCO
            elif 'La' in elements and ('Sr' in elements or 'Ba' in elements):
                return SuperconductorFamily.CUPRATE_LSCO
            elif 'Bi' in elements and 'Sr' in elements:
                return SuperconductorFamily.CUPRATE_BSCCO
            elif 'Tl' in elements and 'Ba' in elements:
                return SuperconductorFamily.CUPRATE_TBCCO
            elif 'Hg' in elements and 'Ba' in elements:
                return SuperconductorFamily.CUPRATE_HBCCO
            else:
                return SuperconductorFamily.CUPRATE_OTHER

        # Check for iron-based
        if 'Fe' in elements:
            if 'As' in elements or 'P' in elements:
                return SuperconductorFamily.IRON_PNICTIDE
            elif 'Se' in elements or 'Te' in elements:
                return SuperconductorFamily.IRON_CHALCOGENIDE

        # Check for MgB2-type
        if 'Mg' in elements and 'B' in elements:
            return SuperconductorFamily.MGB2_TYPE

        # Check for heavy fermion (f-block elements)
        if elements & HEAVY_FERMION_ELEMENTS:
            return SuperconductorFamily.HEAVY_FERMION

        # Check for organic (predominantly C, H, N, S)
        organic_fraction = len(elements & ORGANIC_ELEMENTS) / max(len(elements), 1)
        if organic_fraction > 0.5 and 'C' in elements:
            return SuperconductorFamily.ORGANIC

        # Default to BCS conventional for simple metallic compounds
        # Most remaining superconductors are BCS-type
        if len(elements) <= 4:
            return SuperconductorFamily.BCS_CONVENTIONAL

        # Unknown/complex materials
        return SuperconductorFamily.OTHER_UNKNOWN

    def classify_from_formula(self, formula: str) -> SuperconductorFamily:
        """
        Classify from formula string.

        Args:
            formula: Chemical formula string (e.g., "YBa2Cu3O7")

        Returns:
            SuperconductorFamily enum value
        """
        elements = self._parse_elements(formula)
        return self.classify_from_elements(elements)

    def _parse_elements(self, formula: str) -> Set[str]:
        """Extract element set from formula string."""
        import re
        # Match element symbols (capital letter optionally followed by lowercase)
        pattern = r'([A-Z][a-z]?)'
        matches = re.findall(pattern, formula)
        return set(matches)

    def classify_batch_from_tokens(
        self,
        tokens: torch.Tensor,
        idx_to_token: Dict[int, str],
        element_set: Set[str],
    ) -> torch.Tensor:
        """
        Classify batch of token sequences.

        Args:
            tokens: [batch, seq_len] token indices
            idx_to_token: Token index to string mapping
            element_set: Set of valid element symbols

        Returns:
            [batch] tensor of family indices
        """
        batch_size = tokens.size(0)
        families = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)

        for i in range(batch_size):
            # Extract elements from tokens
            elements = set()
            for idx in tokens[i].tolist():
                token = idx_to_token.get(idx, '')
                if token in element_set:
                    elements.add(token)

            family = self.classify_from_elements(elements)
            families[i] = family.value

        return families

    def get_simple_class(self, family: SuperconductorFamily) -> int:
        """Map detailed family to simple 6-class version."""
        return FAMILY_TO_SIMPLE[family]


class LearnedFamilyClassifier(nn.Module):
    """
    Neural network classifier for superconductor families.

    Can predict family from:
    - Latent vector z
    - Magpie features
    - Composition features

    Outputs class logits that can be used for:
    - Hard classification (argmax)
    - Soft classification (probabilities)
    - Auxiliary training loss
    """

    def __init__(
        self,
        input_dim: int,  # Latent dim or feature dim
        config: Optional[FamilyClassifierConfig] = None,
    ):
        super().__init__()
        self.config = config or FamilyClassifierConfig()

        num_classes = 6 if self.config.use_simple_classes else 14

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, num_classes),
        )

        # For getting family names
        self.family_names = SIMPLE_FAMILY_NAMES if self.config.use_simple_classes else [
            f.name.lower() for f in SuperconductorFamily
        ]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict family logits.

        Args:
            features: [batch, input_dim] latent or Magpie features

        Returns:
            [batch, num_classes] logits
        """
        return self.classifier(features)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Get hard predictions (class indices)."""
        logits = self.forward(features)
        return logits.argmax(dim=-1)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(features)
        return F.softmax(logits, dim=-1)

    def get_family_name(self, class_idx: int) -> str:
        """Get family name from class index."""
        return self.family_names[class_idx]


class HybridFamilyClassifier(nn.Module):
    """
    Hybrid classifier combining rule-based and learned approaches.

    - Uses rule-based classification as ground truth labels
    - Learns to predict family from latent/Magpie features
    - Can be used to condition generation on family
    """

    def __init__(
        self,
        latent_dim: int = 512,
        magpie_dim: int = 145,
        config: Optional[FamilyClassifierConfig] = None,
    ):
        super().__init__()
        self.config = config or FamilyClassifierConfig()

        # Rule-based classifier for ground truth
        self.rule_based = RuleBasedFamilyClassifier(config)

        # Learned classifier from latent
        self.from_latent = LearnedFamilyClassifier(
            input_dim=latent_dim,
            config=config,
        )

        # Learned classifier from Magpie
        self.from_magpie = LearnedFamilyClassifier(
            input_dim=magpie_dim,
            config=config,
        )

        # Combined classifier (latent + magpie)
        self.from_combined = LearnedFamilyClassifier(
            input_dim=latent_dim + magpie_dim,
            config=config,
        )

    def forward(
        self,
        latent: Optional[torch.Tensor] = None,
        magpie: Optional[torch.Tensor] = None,
        mode: str = 'combined',
    ) -> torch.Tensor:
        """
        Predict family logits.

        Args:
            latent: [batch, latent_dim] latent vectors
            magpie: [batch, magpie_dim] Magpie features
            mode: 'latent', 'magpie', or 'combined'

        Returns:
            [batch, num_classes] logits
        """
        if mode == 'latent':
            assert latent is not None
            return self.from_latent(latent)
        elif mode == 'magpie':
            assert magpie is not None
            return self.from_magpie(magpie)
        elif mode == 'combined':
            assert latent is not None and magpie is not None
            combined = torch.cat([latent, magpie], dim=-1)
            return self.from_combined(combined)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_loss(
        self,
        latent: torch.Tensor,
        magpie: torch.Tensor,
        target_families: torch.Tensor,
        mode: str = 'combined',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss.

        Args:
            latent: [batch, latent_dim]
            magpie: [batch, magpie_dim]
            target_families: [batch] ground truth family indices
            mode: Which classifier to train

        Returns:
            Dict with 'loss', 'accuracy'
        """
        logits = self.forward(latent, magpie, mode=mode)
        loss = F.cross_entropy(logits, target_families)

        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == target_families).float().mean()

        return {
            'loss': loss,
            'accuracy': accuracy,
            'logits': logits,
        }


def get_theory_applicable(family: SuperconductorFamily) -> List[str]:
    """
    Get list of theories applicable to a superconductor family.

    Returns theory names that can be used for regularization.
    """
    if family == SuperconductorFamily.BCS_CONVENTIONAL:
        return ['bcs', 'mcmillan']
    elif family in [SuperconductorFamily.MGB2_TYPE]:
        return ['bcs', 'mcmillan', 'two_gap']
    elif family.name.startswith('CUPRATE'):
        return ['cuprate_dome', 'd_wave', 'zhang_rice']
    elif family in [SuperconductorFamily.IRON_PNICTIDE, SuperconductorFamily.IRON_CHALCOGENIDE]:
        return ['iron_based', 's_pm', 'multiband']
    elif family == SuperconductorFamily.HEAVY_FERMION:
        return ['heavy_fermion', 'kondo']
    elif family == SuperconductorFamily.ORGANIC:
        return ['organic']
    else:
        # Unknown - no specific theory constraints
        return ['none']
