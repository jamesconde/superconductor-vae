"""
Superconductor candidate generation modules.

Provides:
- LatentSpaceAnalyzer: Analyze and visualize learned latent space
- CandidateGenerator: Generate new superconductor candidates (Magpie features)
- SuperconductorDiscoveryPipeline: End-to-end discovery pipeline
"""

from .latent_analyzer import LatentSpaceAnalyzer
from .candidate_generator import CandidateGenerator, GeneratedCandidate
from .discovery_pipeline import SuperconductorDiscoveryPipeline

__all__ = [
    'LatentSpaceAnalyzer',
    'CandidateGenerator',
    'GeneratedCandidate',
    'SuperconductorDiscoveryPipeline',
]
