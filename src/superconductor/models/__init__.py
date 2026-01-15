"""
Superconductor prediction and generation models.

Provides:
- BidirectionalVAE: VAE for prediction and generation (uses Magpie features)
- AttentionBidirectionalVAE: VAE with integrated element attention (recommended)
- TcPredictor: Expert network for Tc prediction
- SuperconductorRMENN: R-MENN adapted for superconductors

Architecture Comparison:

    BidirectionalVAE (original):
        Magpie features (144) → VAE encoder → Latent → Tc predictor
        - Fast, uses pre-computed features
        - No element-level interpretability

    AttentionBidirectionalVAE (recommended for discovery):
        Element indices/fractions → Element Attention → VAE encoder → Latent → Tc predictor
        - Attention weights show which elements matter for Tc
        - Supports isotope features
        - Element contributions decompose Tc prediction
"""

from .bidirectional_vae import (
    BidirectionalVAE,
    BidirectionalEncoder,
    FeatureDecoder,
    TcPredictor,
    CompetenceHead,
    BidirectionalVAELoss,
)

from .attention_vae import (
    AttentionBidirectionalVAE,
    AttentionVAELoss,
    ElementEncoder,
    create_attention_vae,
)

from .autoregressive_decoder import (
    AutoregressiveFormulaDecoder,
    FormulaVAEWithDecoder,
    tokenize_formula,
    tokens_to_indices,
    indices_to_formula,
    create_formula_tokenizer,
    VOCAB_SIZE,
)

__all__ = [
    # Original VAE (Magpie features)
    'BidirectionalVAE',
    'BidirectionalEncoder',
    'FeatureDecoder',
    'TcPredictor',
    'CompetenceHead',
    'BidirectionalVAELoss',

    # Attention VAE (element-level, recommended)
    'AttentionBidirectionalVAE',
    'AttentionVAELoss',
    'ElementEncoder',
    'create_attention_vae',

    # Autoregressive decoder (for novel formula generation)
    'AutoregressiveFormulaDecoder',
    'FormulaVAEWithDecoder',
    'tokenize_formula',
    'tokens_to_indices',
    'indices_to_formula',
    'create_formula_tokenizer',
    'VOCAB_SIZE',
]
