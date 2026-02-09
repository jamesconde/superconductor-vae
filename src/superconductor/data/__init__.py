"""
Superconductor data loading and processing.

Provides dataset classes for:
- SuperCon database loading
- Contrastive learning with negative samples
- Materials API integration
- Attention-compatible datasets (for AttentionBidirectionalVAE)
- Canonical ordering for formula standardization
"""

from .dataset import (
    SuperconductorDataset,
    ContrastiveDataset,
    SuperconductorSample,
)
from .supercon_loader import SuperConLoader

from .attention_dataset import (
    AttentionSuperconductorDataset,
    create_attention_dataloaders,
)

from .canonical_ordering import (
    OrderingMethod,
    CanonicalOrderer,
    OrderAugmentation,
    to_electronegativity_order,
    to_alphabetical_order,
    to_abundance_order,
    shuffle_element_order,
)

__all__ = [
    # Original datasets
    'SuperconductorDataset',
    'ContrastiveDataset',
    'SuperconductorSample',
    'SuperConLoader',

    # Attention-compatible datasets
    'AttentionSuperconductorDataset',
    'create_attention_dataloaders',

    # Canonical ordering
    'OrderingMethod',
    'CanonicalOrderer',
    'OrderAugmentation',
    'to_electronegativity_order',
    'to_alphabetical_order',
    'to_abundance_order',
    'shuffle_element_order',
]
