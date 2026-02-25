"""
SC/non-SC boundary detection in latent space.

Identifies boundary samples as those with high label heterogeneity in their
k-NN neighborhood. Computes boundary thickness, centroid separation distance,
and Fisher separation ratio.

Tests the theoretical prediction of a "saturation boundary" separating
superconductors from non-superconductors in latent space.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.neighbors import NearestNeighbors


def compute_boundary_metrics(
    z: np.ndarray,
    is_sc: np.ndarray,
    k: int = 20,
    knn_model: Optional[NearestNeighbors] = None,
    knn_distances: Optional[np.ndarray] = None,
    knn_indices: Optional[np.ndarray] = None,
    heterogeneity_threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Detect SC/non-SC boundary and compute separation metrics.

    Boundary samples: those whose k-NN neighborhood has > threshold fraction
    of opposite-label neighbors.

    Args:
        z: [N, D] latent vectors
        is_sc: [N] boolean/int array (1=SC, 0=non-SC)
        k: Number of neighbors
        knn_model: Precomputed NearestNeighbors (used only if distances/indices not provided)
        knn_distances: [N, k+1] precomputed distances (col 0 = self)
        knn_indices: [N, k+1] precomputed indices (col 0 = self)
        heterogeneity_threshold: Fraction threshold for boundary classification

    Returns:
        Dict with boundary_thickness, boundary_n_samples,
        sc_nonsc_centroid_distance, sc_nonsc_separation_ratio
    """
    sc_mask = is_sc.astype(bool)

    if sc_mask.sum() == 0 or (~sc_mask).sum() == 0:
        return {
            'boundary_thickness': 0.0,
            'boundary_n_samples': 0,
            'sc_nonsc_centroid_distance': 0.0,
            'sc_nonsc_separation_ratio': 0.0,
        }

    # Get k-NN distances and indices (prefer precomputed)
    if knn_distances is None or knn_indices is None:
        if knn_model is None:
            knn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric='euclidean')
            knn_model.fit(z)
        knn_distances, knn_indices = knn_model.kneighbors(z)

    # Skip self (col 0)
    neighbor_indices = knn_indices[:, 1:k + 1]  # [N, k]
    neighbor_distances = knn_distances[:, 1:k + 1]  # [N, k]

    # Compute per-sample heterogeneity: fraction of neighbors with different label
    neighbor_labels = is_sc[neighbor_indices]  # [N, k]
    own_labels = is_sc[:, np.newaxis]  # [N, 1]
    heterogeneity = np.mean(neighbor_labels != own_labels, axis=1)  # [N]

    # Boundary samples
    boundary_mask = heterogeneity > heterogeneity_threshold
    n_boundary = int(boundary_mask.sum())

    # Boundary thickness: mean distance from boundary samples to nearest opposite-label neighbor
    if n_boundary > 0:
        boundary_thicknesses = []
        for i in np.where(boundary_mask)[0]:
            opp_mask = neighbor_labels[i] != is_sc[i]
            if opp_mask.any():
                boundary_thicknesses.append(float(neighbor_distances[i][opp_mask].min()))
        boundary_thickness = float(np.mean(boundary_thicknesses)) if boundary_thicknesses else 0.0
    else:
        boundary_thickness = 0.0

    # Centroid separation
    sc_centroid = z[sc_mask].mean(axis=0)
    nonsc_centroid = z[~sc_mask].mean(axis=0)
    centroid_distance = float(np.linalg.norm(sc_centroid - nonsc_centroid))

    # Fisher separation ratio: centroid_dist / sqrt(0.5 * (var_SC + var_nonSC))
    var_sc = float(np.mean(np.sum((z[sc_mask] - sc_centroid) ** 2, axis=1)))
    var_nonsc = float(np.mean(np.sum((z[~sc_mask] - nonsc_centroid) ** 2, axis=1)))

    pooled_std = np.sqrt(0.5 * (var_sc + var_nonsc))
    if pooled_std > 1e-10:
        separation_ratio = centroid_distance / pooled_std
    else:
        separation_ratio = 0.0

    return {
        'boundary_thickness': boundary_thickness,
        'boundary_n_samples': n_boundary,
        'sc_nonsc_centroid_distance': centroid_distance,
        'sc_nonsc_separation_ratio': separation_ratio,
    }


def compute_per_sample_boundary(
    z: np.ndarray,
    is_sc: np.ndarray,
    k: int = 20,
    knn_model: Optional[NearestNeighbors] = None,
    knn_distances: Optional[np.ndarray] = None,
    knn_indices: Optional[np.ndarray] = None,
    heterogeneity_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample boundary metrics for full metadata.

    Args:
        z: [N, D] latent vectors
        is_sc: [N] labels
        k: Number of neighbors
        knn_model: Precomputed model (fallback)
        knn_distances: [N, k+1] precomputed distances
        knn_indices: [N, k+1] precomputed indices
        heterogeneity_threshold: Fraction threshold for boundary classification

    Returns:
        (heterogeneity [N], boundary_flag [N] bool)
    """
    if knn_indices is None:
        if knn_model is None:
            knn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric='euclidean')
            knn_model.fit(z)
        _, knn_indices = knn_model.kneighbors(z)

    neighbor_indices = knn_indices[:, 1:k + 1]
    neighbor_labels = is_sc[neighbor_indices]
    own_labels = is_sc[:, np.newaxis]
    heterogeneity = np.mean(neighbor_labels != own_labels, axis=1)

    boundary_flag = heterogeneity > heterogeneity_threshold

    return heterogeneity, boundary_flag
