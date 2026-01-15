"""
Validation modules for superconductor candidates.

Provides:
- CandidateValidator: Family-based heuristics for superconductor likelihood
- PhysicsValidator: Physics-based validation (Hume-Rothery rules, immiscible pairs)

The PhysicsValidator is the PRIMARY check for physical plausibility - it validates
whether compounds could physically exist (alloy formation, bond compatibility).
It does NOT require matching known superconductor families.
"""

from .candidate_validator import (
    CandidateValidator,
    ValidationResult,
    validate_candidate,
)

from .physics_validator import (
    PhysicsValidator,
    PhysicsValidationResult,
    validate_physics,
)

__all__ = [
    # Family-based validation (less critical)
    'CandidateValidator',
    'ValidationResult',
    'validate_candidate',
    # Physics-based validation (PRIMARY for plausibility)
    'PhysicsValidator',
    'PhysicsValidationResult',
    'validate_physics',
]
