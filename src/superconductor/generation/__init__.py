"""
Superconductor candidate generation modules.

Provides:
- LatentSpaceAnalyzer: Analyze and visualize learned latent space
- CandidateGenerator: Generate new superconductor candidates (Magpie features)
- AttentionCandidateGenerator: Generate candidates with attention (recommended)
- SuperconductorDiscoveryPipeline: End-to-end discovery pipeline
"""

from .latent_analyzer import LatentSpaceAnalyzer
from .candidate_generator import CandidateGenerator, GeneratedCandidate
from .discovery_pipeline import SuperconductorDiscoveryPipeline

from .attention_generator import (
    AttentionCandidateGenerator,
    AttentionCandidate,
)

__all__ = [
    # Original generators
    'LatentSpaceAnalyzer',
    'CandidateGenerator',
    'GeneratedCandidate',
    'SuperconductorDiscoveryPipeline',

    # Attention-based generators
    'AttentionCandidateGenerator',
    'AttentionCandidate',
]
