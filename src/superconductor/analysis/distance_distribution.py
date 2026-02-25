"""
Pairwise distance distribution analysis for latent space.

Computes mean, std, skewness, and kurtosis of the pairwise distance
distribution on a stratified subsample. These statistics characterize
the overall geometry of the latent space.

Uses scipy.spatial.distance.pdist on a 5K stratified subsample
(proportional SC/non-SC) to keep computation feasible (~12.5M distances).
"""

import numpy as np
from typing import Dict, Optional
from scipy.spatial.distance import pdist
from scipy.stats import skew, kurtosis


def compute_distance_distribution(
    z: np.ndarray,
    is_sc: np.ndarray,
    n_subsample: int = 5000,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, float]:
    """
    Compute pairwise distance distribution statistics.

    Args:
        z: [N, D] latent vectors
        is_sc: [N] boolean/int array
        n_subsample: Max samples for pairwise computation
        rng: Random state for reproducibility

    Returns:
        Dict with pairwise_dist_mean, pairwise_dist_std,
        pairwise_dist_skewness, pairwise_dist_kurtosis
    """
    if rng is None:
        rng = np.random.RandomState(42)

    sc_mask = is_sc.astype(bool)

    # Stratified subsample (proportional SC/non-SC)
    if len(z) > n_subsample:
        n_sc = sc_mask.sum()
        n_nonsc = (~sc_mask).sum()
        total = n_sc + n_nonsc

        n_sc_sample = max(1, int(n_subsample * n_sc / total))
        n_nonsc_sample = max(1, n_subsample - n_sc_sample)

        sc_indices = np.where(sc_mask)[0]
        nonsc_indices = np.where(~sc_mask)[0]

        sc_sample = rng.choice(sc_indices, min(n_sc_sample, len(sc_indices)), replace=False)
        nonsc_sample = rng.choice(nonsc_indices, min(n_nonsc_sample, len(nonsc_indices)), replace=False)

        sample_indices = np.concatenate([sc_sample, nonsc_sample])
        z_sub = z[sample_indices]
    else:
        z_sub = z

    # Compute pairwise distances
    dists = pdist(z_sub, metric='euclidean')

    if len(dists) == 0:
        return {
            'pairwise_dist_mean': 0.0,
            'pairwise_dist_std': 0.0,
            'pairwise_dist_skewness': 0.0,
            'pairwise_dist_kurtosis': 0.0,
        }

    return {
        'pairwise_dist_mean': float(np.mean(dists)),
        'pairwise_dist_std': float(np.std(dists)),
        'pairwise_dist_skewness': float(skew(dists)),
        'pairwise_dist_kurtosis': float(kurtosis(dists)),
    }
