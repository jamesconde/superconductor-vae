"""
Superconductor data loading and processing.

Provides dataset classes for:
- SuperCon database loading
- Contrastive learning with negative samples
- Materials API integration
- Attention-compatible datasets (for AttentionBidirectionalVAE)
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

__all__ = [
    # Original datasets
    'SuperconductorDataset',
    'ContrastiveDataset',
    'SuperconductorSample',
    'SuperConLoader',

    # Attention-compatible datasets
    'AttentionSuperconductorDataset',
    'create_attention_dataloaders',
]
