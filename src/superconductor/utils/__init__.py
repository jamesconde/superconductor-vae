"""
Superconductor utility functions.

Includes:
- Materials Project API client for negative samples
- Pre-compiled material lists for fallback
"""

from .materials_api import (
    MaterialsProjectClient,
    get_api_key,
    fetch_negative_samples,
    DEFAULT_NON_SUPERCONDUCTORS,
    DEFAULT_MAGNETIC_MATERIALS,
)

__all__ = [
    'MaterialsProjectClient',
    'get_api_key',
    'fetch_negative_samples',
    'DEFAULT_NON_SUPERCONDUCTORS',
    'DEFAULT_MAGNETIC_MATERIALS',
]
