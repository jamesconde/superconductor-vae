"""
Superconductor VAE — Generative model for superconductor formula discovery.

Multi-task autoencoder that encodes superconductor compositions into a rich
latent space (z=2048) and decodes back to formulas, Tc, Magpie features,
SC classification, and hierarchical family labels.

Active Architecture (V12+/V14):
    FullMaterialsVAE (encoder):
        Element attention + Magpie features (145) + Tc → Fusion → z (2048)
    EnhancedTransformerDecoder (formula decoder):
        z → 24 cross-attention memory tokens → autoregressive formula generation
    Multi-head decoder:
        z → Tc, Magpie, SC class, family, high-pressure, Tc bucket, fractions

Key Features:
- Element-level attention for learning element importance
- Semantic fraction tokenization (FRAC:p/q tokens) for exact stoichiometry
- Contrastive learning with non-superconductors (46K dataset)
- REINFORCE/RLOO for autoregressive formula generation
- Token type classifier with hard vocab masking
- Isotope-aware encoding with 291 isotopes for 84 elements
- Latent space analysis and candidate generation
- Rule-based validation (charge balance, element compatibility)

Isotope Support:
    from superconductor import IsotopeEncoder, estimate_isotope_effect

    encoder = IsotopeEncoder()
    result = encoder.encode('LaD10')  # Deuterium (H-2)
    tc_h = 250.0  # LaH10 at ~250K
    tc_d = estimate_isotope_effect('H', 1, 2, tc_h)  # ~177K for LaD10

Element Attention:
    from superconductor import ElementAttention, interpret_attention_weights

    attention = ElementAttention(hidden_dim=64, n_heads=4)
    output = attention(element_embeddings, element_mask)
    # output.attention_weights shows element importance (e.g., Cu/O high in YBCO)

Data Directory:
    data/processed/supercon_fractions_contrastive.csv (46,645 samples)

Training:
    cd superconductor-vae && PYTHONPATH=src python scripts/train_v12_clean.py
"""

# Encoders
from .encoders.composition_encoder import (
    CompositionEncoder,
    CompositionDecoder,
    EncodedComposition,
    encode_formula,
    decode_composition,
)
from .encoders.element_properties import (
    ELEMENT_PROPERTIES,
    ELEMENT_SYMBOLS,
    SYMBOL_TO_Z,
    get_element_property,
    get_atomic_number,
    get_element_symbol,
    get_oxidation_states,
    is_superconductor_relevant,
    SUPERCONDUCTOR_ELEMENTS,
)
from .encoders.feature_groups import (
    FeatureGroups,
    GroupedFeatureEncoder,
    ExpertAttentionHead,
    AttentiveExpert,
    ContrastiveFeatureEncoder,
    DEFAULT_GROUP_DIMS,
)

# Isotope support
from .encoders.isotope_properties import (
    ISOTOPE_DATABASE,
    IsotopeData,
    get_isotope,
    get_all_isotopes,
    get_stable_isotopes,
    get_most_abundant_isotope,
    parse_isotope_notation,
    estimate_isotope_effect,
    get_isotope_mass_ratio,
)

# Element attention
from .encoders.element_attention import (
    ElementEmbedding,
    ElementAttention,
    IsotopeAwareElementAttention,
    MultiHeadElementAttention,
    AttentionOutput,
    interpret_attention_weights,
)

# Isotope-aware encoding
from .encoders.isotope_encoder import (
    IsotopeEncoder,
    IsotopeEncodedComposition,
    IsotopeFormulaParser,
    IsotopeAwareEncoder,
    create_isotope_encoder,
)

# Data
from .data.dataset import (
    SuperconductorDataset,
    ContrastiveDataset,
    SuperconductorSample,
)
from .data.supercon_loader import (
    SuperConLoader,
    load_supercon,
)

# Models
from .models.bidirectional_vae import (
    BidirectionalVAE,
    BidirectionalEncoder,
    FeatureDecoder,
    TcPredictor,
    CompetenceHead,
    BidirectionalVAELoss,
    create_bidirectional_vae,
)
from .models.attention_vae import (
    ElementEncoder,
)

# Validation
from .validation.candidate_validator import (
    CandidateValidator,
    ValidationResult,
    validate_candidate,
)

# Generation
from .generation.latent_analyzer import (
    LatentSpaceAnalyzer,
    ClusterInfo,
)
from .generation.candidate_generator import (
    CandidateGenerator,
    GeneratedCandidate,
)
from .generation.discovery_pipeline import (
    SuperconductorDiscoveryPipeline,
    DiscoveryConfig,
)

# Utils - Materials API
from .utils.materials_api import (
    MaterialsProjectClient,
    get_api_key,
    fetch_negative_samples,
    DEFAULT_NON_SUPERCONDUCTORS,
    DEFAULT_MAGNETIC_MATERIALS,
)

__all__ = [
    # Encoders
    'CompositionEncoder',
    'CompositionDecoder',
    'EncodedComposition',
    'encode_formula',
    'decode_composition',
    'ELEMENT_PROPERTIES',
    'ELEMENT_SYMBOLS',
    'SYMBOL_TO_Z',
    'get_element_property',
    'get_atomic_number',
    'get_element_symbol',
    'get_oxidation_states',
    'is_superconductor_relevant',
    'SUPERCONDUCTOR_ELEMENTS',
    'FeatureGroups',
    'GroupedFeatureEncoder',
    'ExpertAttentionHead',
    'AttentiveExpert',
    'ContrastiveFeatureEncoder',
    'DEFAULT_GROUP_DIMS',

    # Isotope support
    'ISOTOPE_DATABASE',
    'IsotopeData',
    'get_isotope',
    'get_all_isotopes',
    'get_stable_isotopes',
    'get_most_abundant_isotope',
    'parse_isotope_notation',
    'estimate_isotope_effect',
    'get_isotope_mass_ratio',

    # Element attention
    'ElementEmbedding',
    'ElementAttention',
    'IsotopeAwareElementAttention',
    'MultiHeadElementAttention',
    'AttentionOutput',
    'interpret_attention_weights',

    # Isotope-aware encoding
    'IsotopeEncoder',
    'IsotopeEncodedComposition',
    'IsotopeFormulaParser',
    'IsotopeAwareEncoder',
    'create_isotope_encoder',

    # Data
    'SuperconductorDataset',
    'ContrastiveDataset',
    'SuperconductorSample',
    'SuperConLoader',
    'load_supercon',

    # Models - Original VAE (Magpie features)
    'BidirectionalVAE',
    'BidirectionalEncoder',
    'FeatureDecoder',
    'TcPredictor',
    'CompetenceHead',
    'BidirectionalVAELoss',
    'create_bidirectional_vae',

    # Models - Attention VAE (element-level)
    'ElementEncoder',

    # Validation
    'CandidateValidator',
    'ValidationResult',
    'validate_candidate',

    # Generation
    'LatentSpaceAnalyzer',
    'ClusterInfo',
    'CandidateGenerator',
    'GeneratedCandidate',
    'SuperconductorDiscoveryPipeline',
    'DiscoveryConfig',

    # Utils - Materials API
    'MaterialsProjectClient',
    'get_api_key',
    'fetch_negative_samples',
    'DEFAULT_NON_SUPERCONDUCTORS',
    'DEFAULT_MAGNETIC_MATERIALS',
]

__version__ = '0.2.0'
