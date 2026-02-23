"""
Superconductor prediction and generation models.

Provides:
- FullMaterialsVAE: V12+ full materials VAE (element attention + Magpie + Tc)
- ElementEncoder: Element-level attention encoder (used by FullMaterialsVAE)
- BidirectionalVAE: Original VAE for prediction and generation (Magpie features)
- TcPredictor: Expert network for Tc prediction
- FamilyClassifier: Classify superconductors by physical mechanism

Active Architecture (V12+):
    FullMaterialsVAE:
        Element attention + Magpie features + Tc → Fusion → VAE encoder → Latent z (2048)
        z → Multi-head decoder (Tc, Magpie, formula, SC class, family, etc.)
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
    ElementEncoder,
)

from .autoregressive_decoder import (
    tokenize_formula,
    tokens_to_indices,
    indices_to_formula,
    create_formula_tokenizer,
    VOCAB_SIZE,
)

from .physics_z import PhysicsZ

from .family_classifier import (
    SuperconductorFamily,
    FamilyClassifierConfig,
    RuleBasedFamilyClassifier,
    LearnedFamilyClassifier,
    HybridFamilyClassifier,
    get_theory_applicable,
)

__all__ = [
    # Original VAE (Magpie features)
    'BidirectionalVAE',
    'BidirectionalEncoder',
    'FeatureDecoder',
    'TcPredictor',
    'CompetenceHead',
    'BidirectionalVAELoss',

    # Attention VAE (element-level)
    'ElementEncoder',

    # Autoregressive decoder (for novel formula generation)
    'tokenize_formula',
    'tokens_to_indices',
    'indices_to_formula',
    'create_formula_tokenizer',
    'VOCAB_SIZE',

    # Physics Z coordinate map (V12.31)
    'PhysicsZ',

    # Family classifier (for theory-based regularization)
    'SuperconductorFamily',
    'FamilyClassifierConfig',
    'RuleBasedFamilyClassifier',
    'LearnedFamilyClassifier',
    'HybridFamilyClassifier',
    'get_theory_applicable',
]
