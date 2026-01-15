"""
Superconductor encoding modules.

Provides chemical formula encoding and feature extraction for superconductor materials.
Includes isotope-aware encoding and element-level attention mechanisms.
"""

from .composition_encoder import CompositionEncoder
from .element_properties import ELEMENT_PROPERTIES, get_element_property
from .feature_groups import FeatureGroups, GroupedFeatureEncoder

# Isotope support
from .isotope_properties import (
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
from .element_attention import (
    ElementEmbedding,
    ElementAttention,
    IsotopeAwareElementAttention,
    MultiHeadElementAttention,
    AttentionOutput,
    interpret_attention_weights,
)

# Isotope-aware encoding
from .isotope_encoder import (
    IsotopeEncoder,
    IsotopeEncodedComposition,
    IsotopeFormulaParser,
    IsotopeAwareEncoder,
    create_isotope_encoder,
)

__all__ = [
    # Composition encoding
    'CompositionEncoder',
    'ELEMENT_PROPERTIES',
    'get_element_property',
    'FeatureGroups',
    'GroupedFeatureEncoder',

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
]
