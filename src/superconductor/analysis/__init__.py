"""
Latent space topology analysis for superconductor VAE.

Provides intrinsic dimensionality estimation, local density analysis,
SC/non-SC boundary detection, cluster topology (KMeans + HDBSCAN),
and longitudinal tracking.
"""

from .topology_analyzer import TopologyAnalyzer, TopologySnapshot

__all__ = ['TopologyAnalyzer', 'TopologySnapshot']
