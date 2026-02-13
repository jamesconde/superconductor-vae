"""
Superconductor utility functions.

Includes:
- Materials Project API client for negative samples
- Pre-compiled material lists for fallback
- Training manifest system (V12.29) for checkpoint version tracking
"""

from .materials_api import (
    MaterialsProjectClient,
    get_api_key,
    fetch_negative_samples,
    DEFAULT_NON_SUPERCONDUCTORS,
    DEFAULT_MAGNETIC_MATERIALS,
)
from .env_config import detect_environment
from .manifest import (
    build_manifest,
    check_config_drift,
    compute_config_hash,
    get_git_info,
    get_environment_info,
    get_model_architecture_fingerprint,
)

__all__ = [
    'MaterialsProjectClient',
    'get_api_key',
    'fetch_negative_samples',
    'DEFAULT_NON_SUPERCONDUCTORS',
    'DEFAULT_MAGNETIC_MATERIALS',
    'detect_environment',
    'build_manifest',
    'check_config_drift',
    'compute_config_hash',
    'get_git_info',
    'get_environment_info',
    'get_model_architecture_fingerprint',
]
