"""
Intrinsic dimensionality estimation for latent space.

Implements:
- MLE estimator (Levina-Bickel 2004): Per-point estimate from k-NN distances,
  global estimate = harmonic mean. Fast, reliable for high-D data.
- Correlation dimension (Grassberger-Procaccia): Log-log slope of C(r) on subsample.
  Slower but provides independent validation.

Uses sklearn.neighbors.NearestNeighbors (brute force at 2048 dims).
"""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.neighbors import NearestNeighbors


def compute_intrinsic_dimension_mle(
    z: np.ndarray,
    k: int = 20,
    knn_distances: Optional[np.ndarray] = None,
    knn_model: Optional[NearestNeighbors] = None,
) -> Tuple[Dict[str, float], np.ndarray, Optional[NearestNeighbors]]:
    """
    Estimate intrinsic dimensionality using MLE (Levina-Bickel 2004).

    For each point i:
        d_hat_i = [(1/(k-1)) * sum_{j=1}^{k-1} log(r_k / r_j)]^{-1}

    Global estimate = harmonic mean of per-point estimates.

    Args:
        z: [N, D] array of latent vectors
        k: Number of nearest neighbors
        knn_distances: [N, k+1] precomputed distances (optional, skips fitting)
        knn_model: Precomputed NearestNeighbors model (optional)

    Returns:
        (metrics_dict, per_sample_dim_estimates, fitted_knn_model)
    """
    if knn_distances is None:
        if knn_model is None:
            knn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric='euclidean')
            knn_model.fit(z)
        knn_distances, _ = knn_model.kneighbors(z)

    # knn_distances[:, 0] is distance to self (0), so use [:, 1:]
    dists = knn_distances[:, 1:k + 1]  # [N, k]

    # r_k is the k-th nearest neighbor distance (last column)
    r_k = dists[:, -1]  # [N]

    # Compute per-point MLE estimate
    # d_hat_i = [(1/(k-1)) * sum_{j=1}^{k-1} log(r_k / r_j)]^{-1}
    # Using r_j for j=1..k-1 (columns 0..k-2 of dists)
    r_j = dists[:, :-1]  # [N, k-1]

    # Guard against zero distances
    r_j_safe = np.maximum(r_j, 1e-10)
    r_k_safe = np.maximum(r_k, 1e-10)

    log_ratios = np.log(r_k_safe[:, np.newaxis] / r_j_safe)  # [N, k-1]
    log_ratios = np.maximum(log_ratios, 1e-10)  # Guard against log(1)=0

    mean_log_ratio = log_ratios.mean(axis=1)  # [N]
    per_point_dim = 1.0 / np.maximum(mean_log_ratio, 1e-10)  # [N]

    # Filter out extreme outliers (numerical artifacts)
    valid_mask = (per_point_dim > 0) & (per_point_dim < 10 * z.shape[1])
    valid_dims = per_point_dim[valid_mask]

    if len(valid_dims) == 0:
        global_dim = 0.0
    else:
        # Harmonic mean for global estimate
        global_dim = float(len(valid_dims) / np.sum(1.0 / valid_dims))

    return {
        'intrinsic_dim_mle': global_dim,
    }, per_point_dim, knn_model


def compute_intrinsic_dimension_mle_subset(
    z: np.ndarray,
    is_sc: np.ndarray,
    k: int = 20,
    knn_distances: Optional[np.ndarray] = None,
    knn_model: Optional[NearestNeighbors] = None,
) -> Tuple[Dict[str, float], np.ndarray, NearestNeighbors]:
    """
    Compute MLE intrinsic dim globally and for SC/non-SC subsets.

    Args:
        z: [N, D] latent vectors
        is_sc: [N] boolean/int array (1=SC, 0=non-SC)
        k: Number of neighbors
        knn_distances: Precomputed distances
        knn_model: Precomputed model

    Returns:
        (metrics_dict, per_sample_dim, knn_model)
    """
    metrics, per_point_dim, knn_model = compute_intrinsic_dimension_mle(
        z, k=k, knn_distances=knn_distances, knn_model=knn_model
    )

    sc_mask = is_sc.astype(bool)
    nonsc_mask = ~sc_mask

    # SC subset
    sc_dims = per_point_dim[sc_mask]
    valid_sc = sc_dims[(sc_dims > 0) & (sc_dims < 10 * z.shape[1])]
    if len(valid_sc) > 0:
        metrics['intrinsic_dim_mle_sc'] = float(len(valid_sc) / np.sum(1.0 / valid_sc))
    else:
        metrics['intrinsic_dim_mle_sc'] = 0.0

    # Non-SC subset
    nonsc_dims = per_point_dim[nonsc_mask]
    valid_nonsc = nonsc_dims[(nonsc_dims > 0) & (nonsc_dims < 10 * z.shape[1])]
    if len(valid_nonsc) > 0:
        metrics['intrinsic_dim_mle_nonsc'] = float(len(valid_nonsc) / np.sum(1.0 / valid_nonsc))
    else:
        metrics['intrinsic_dim_mle_nonsc'] = 0.0

    return metrics, per_point_dim, knn_model


def compute_correlation_dimension(
    z: np.ndarray,
    n_subsample: int = 5000,
    n_radii: int = 50,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, float]:
    """
    Estimate correlation dimension via Grassberger-Procaccia algorithm.

    Computes C(r) = fraction of pairs closer than r, then fits log-log slope.
    Uses subsample for computational feasibility.

    Args:
        z: [N, D] latent vectors
        n_subsample: Number of points to subsample (pairwise cost)
        n_radii: Number of radius values to evaluate
        rng: Random state for reproducibility

    Returns:
        Dict with 'intrinsic_dim_correlation'
    """
    from scipy.spatial.distance import pdist

    if rng is None:
        rng = np.random.RandomState(42)

    # Subsample if needed
    if len(z) > n_subsample:
        idx = rng.choice(len(z), n_subsample, replace=False)
        z_sub = z[idx]
    else:
        z_sub = z

    # Compute pairwise distances
    dists = pdist(z_sub, metric='euclidean')
    n_pairs = len(dists)

    if n_pairs == 0:
        return {'intrinsic_dim_correlation': 0.0}

    # Compute C(r) for a range of radii
    d_min = np.percentile(dists, 1)
    d_max = np.percentile(dists, 99)

    if d_min <= 0 or d_max <= d_min:
        return {'intrinsic_dim_correlation': 0.0}

    radii = np.logspace(np.log10(d_min), np.log10(d_max), n_radii)
    c_r = np.array([np.sum(dists < r) / n_pairs for r in radii])

    # Fit log-log slope in the scaling region (middle 60% of radii)
    valid = c_r > 0
    if valid.sum() < 5:
        return {'intrinsic_dim_correlation': 0.0}

    log_r = np.log(radii[valid])
    log_c = np.log(c_r[valid])

    # Use middle portion for fitting (avoid edge effects)
    n_valid = len(log_r)
    start = n_valid // 5
    end = 4 * n_valid // 5
    if end - start < 3:
        start = 0
        end = n_valid

    # Linear regression on log-log
    coeffs = np.polyfit(log_r[start:end], log_c[start:end], 1)
    corr_dim = float(coeffs[0])

    # Sanity check
    if corr_dim < 0 or corr_dim > z.shape[1]:
        corr_dim = 0.0

    return {'intrinsic_dim_correlation': corr_dim}
