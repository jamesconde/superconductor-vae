"""
k-NN local density estimation for latent space analysis.

Computes per-sample local density as 1 / r_k^d where d is the intrinsic
dimensionality estimate. Provides density contrast between SC and non-SC
populations.

Reuses k-NN distances from intrinsic dimension computation to avoid
redundant neighbor searches.
"""

import numpy as np
from typing import Dict, Optional


def compute_knn_density(
    knn_distances: np.ndarray,
    is_sc: np.ndarray,
    intrinsic_dim: float,
    k: int = 20,
) -> Dict[str, float]:
    """
    Compute k-NN density metrics.

    Args:
        knn_distances: [N, k+1] distances from NearestNeighbors (col 0 = self)
        is_sc: [N] boolean/int array
        intrinsic_dim: Global intrinsic dimensionality estimate for density scaling
        k: Number of neighbors used

    Returns:
        Dict with knn_mean_distance, knn_median_distance, knn_distance_std,
        density_contrast_sc_nonsc
    """
    # Distance to k-th neighbor (last column, skipping self)
    r_k = knn_distances[:, min(k, knn_distances.shape[1] - 1)]

    metrics = {
        'knn_mean_distance': float(np.mean(r_k)),
        'knn_median_distance': float(np.median(r_k)),
        'knn_distance_std': float(np.std(r_k)),
    }

    # Density contrast between SC and non-SC
    sc_mask = is_sc.astype(bool)

    if sc_mask.sum() > 0 and (~sc_mask).sum() > 0:
        r_k_sc = r_k[sc_mask]
        r_k_nonsc = r_k[~sc_mask]

        # Mean k-NN distance (inverse proxy for density)
        # Lower r_k = higher density
        mean_r_sc = np.mean(r_k_sc)
        mean_r_nonsc = np.mean(r_k_nonsc)

        # Density contrast: ratio of mean inverse distances
        # > 1 means SC is denser than non-SC
        if mean_r_sc > 1e-10:
            metrics['density_contrast_sc_nonsc'] = float(mean_r_nonsc / mean_r_sc)
        else:
            metrics['density_contrast_sc_nonsc'] = 0.0
    else:
        metrics['density_contrast_sc_nonsc'] = 0.0

    return metrics


def compute_per_sample_density(
    knn_distances: np.ndarray,
    intrinsic_dim: float,
    k: int = 20,
) -> np.ndarray:
    """
    Compute per-sample local density estimate.

    density_i = 1 / r_k^d where d = intrinsic_dim

    Args:
        knn_distances: [N, k+1] distances from NearestNeighbors
        intrinsic_dim: Estimated intrinsic dimensionality
        k: Number of neighbors

    Returns:
        [N] array of per-sample density estimates
    """
    r_k = knn_distances[:, min(k, knn_distances.shape[1] - 1)]
    r_k_safe = np.maximum(r_k, 1e-10)

    # Use capped intrinsic dim for density estimation
    d = max(intrinsic_dim, 1.0)

    density = 1.0 / (r_k_safe ** d)

    # Normalize to [0, 1] range for comparability across epochs
    d_max = density.max()
    if d_max > 0:
        density = density / d_max

    return density
