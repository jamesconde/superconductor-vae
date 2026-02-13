"""
R-MENN Superconductor Discovery Module

A comprehensive module for predicting critical temperature (Tc) of superconductors
and discovering new superconductor candidates using R-MENN architecture.

Key Features:
- Chemical formula encoding with elemental property statistics
- Isotope-aware encoding with 291 isotopes for 84 elements
- Element-level attention for learning element importance
- Bidirectional VAE for prediction and generation
- Contrastive learning with non-superconductors and magnetic materials
- Latent space analysis and visualization
- Multiple candidate generation strategies
- Rule-based validation (charge balance, element compatibility)

Isotope Support:
    The module supports isotope-specific encoding for exploring isotope effects
    on superconductivity (Tc ∝ M^(-0.5) in BCS superconductors):

    from superconductor import IsotopeEncoder, estimate_isotope_effect

    encoder = IsotopeEncoder()
    result = encoder.encode('LaD10')  # Deuterium (H-2)
    result = encoder.encode('Y(18O)Ba2Cu3O6')  # Oxygen-18

    # Estimate isotope effect
    tc_h = 250.0  # LaH10 at ~250K
    tc_d = estimate_isotope_effect('H', 1, 2, tc_h)  # ~177K for LaD10

Element Attention:
    Learn which elements are most important for Tc prediction:

    from superconductor import ElementAttention, interpret_attention_weights

    attention = ElementAttention(hidden_dim=64, n_heads=4)
    output = attention(element_embeddings, element_mask)
    # output.attention_weights shows element importance (e.g., Cu/O high in YBCO)

Example Usage:
    from superconductor import (
        SuperconductorDataset,
        SuperconductorDiscoveryPipeline,
        DiscoveryConfig
    )

    # Load data
    dataset = SuperconductorDataset.from_supercon('path/to/supercon.csv')

    # Configure pipeline
    config = DiscoveryConfig(
        latent_dim=64,
        epochs=200,
        min_tc_threshold=77.0  # Liquid nitrogen temperature
    )

    # Run discovery
    pipeline = SuperconductorDiscoveryPipeline(dataset, config)
    candidates, results = pipeline.run()

    # Top candidates
    print(results.head(10))

Data Directory:
    Place SuperCon dataset in:
    RMENN-Signature-Clean/data/superconductor/raw/

Architecture Philosophy:
    This module follows R-MENN's core philosophy of COMPETENCY over accuracy:
    - Models learn to predict confidently on materials they understand
    - Models abstain/hand off on materials they are uncertain about
    - Validation ensures chemically plausible candidates
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
    AttentionBidirectionalVAE,
    AttentionVAELoss,
    ElementEncoder,
    create_attention_vae,
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

    # Models - Attention VAE (element-level, recommended)
    'AttentionBidirectionalVAE',
    'AttentionVAELoss',
    'ElementEncoder',
    'create_attention_vae',

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
