"""
HDBSCAN density-based clustering for latent space analysis.

Provides an alternative to KMeans that:
- Discovers natural cluster count (no need to specify k)
- Identifies noise/outlier points (label = -1)
- Handles non-convex cluster shapes
- Finds clusters of varying density

Requires PCA pre-reduction (2048D -> ~20D) for computational feasibility
and to avoid the curse of dimensionality. The PCA step is built-in.

Key differences from KMeans (cluster_topology.py):
    KMeans: Forces k=9 clusters, assigns every point, assumes spherical clusters.
            Best for: tracking known SC family cluster metrics over training.
    HDBSCAN: Finds natural cluster count, labels outliers as noise, handles
             arbitrary shapes. Best for: discovering structure, identifying
             outlier materials, understanding natural groupings.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def compute_hdbscan_metrics(
    z: np.ndarray,
    is_sc: np.ndarray,
    min_cluster_size: int = 100,
    pca_dims: int = 20,
    tc_values: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Run HDBSCAN clustering and compute topology metrics.

    Applies PCA pre-reduction before HDBSCAN to avoid curse of dimensionality
    and keep runtime feasible (~35-40s on 50K samples).

    Args:
        z: [N, D] latent vectors (full dataset)
        is_sc: [N] boolean/int array
        min_cluster_size: Minimum samples to form a cluster.
            Lower = more granular clusters + more noise.
            Higher = fewer, larger clusters + less noise.
            Recommended: 100 for ~50K samples.
        pca_dims: Number of PCA components for pre-reduction.
            20 is the sweet spot (fast, good silhouette on this data).
        tc_values: [N] Tc values for per-cluster statistics
        random_state: PCA seed

    Returns:
        Dict with hdbscan_n_clusters, hdbscan_noise_fraction,
        hdbscan_silhouette, hdbscan_largest_cluster_fraction,
        hdbscan_tc_range_largest
    """
    sc_mask = is_sc.astype(bool)
    z_sc = z[sc_mask]

    if len(z_sc) < min_cluster_size * 2:
        return _empty_metrics()

    # PCA pre-reduction
    z_pca = PCA(n_components=min(pca_dims, z_sc.shape[1]),
                random_state=random_state).fit_transform(z_sc)

    # HDBSCAN
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        n_jobs=-1,
        store_centers='centroid',
    )
    labels = hdb.fit_predict(z_pca)

    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    n_noise = int((labels == -1).sum())
    noise_fraction = n_noise / len(labels)

    if n_clusters < 2:
        return {
            'hdbscan_n_clusters': n_clusters,
            'hdbscan_noise_fraction': noise_fraction,
            'hdbscan_silhouette': 0.0,
            'hdbscan_largest_cluster_fraction': 0.0,
            'hdbscan_tc_range_largest': 0.0,
        }

    # Silhouette score (on non-noise points, subsample for speed)
    valid = labels != -1
    rng = np.random.RandomState(random_state)
    if valid.sum() > 10000:
        sil_idx = rng.choice(np.where(valid)[0], 10000, replace=False)
        sil_score = float(silhouette_score(z_pca[sil_idx], labels[sil_idx]))
    else:
        sil_score = float(silhouette_score(z_pca[valid], labels[valid]))

    # Largest cluster fraction
    cluster_sizes = [(labels == c).sum() for c in cluster_ids]
    largest_fraction = max(cluster_sizes) / len(labels) if cluster_sizes else 0.0

    # Tc range of largest cluster
    tc_range_largest = 0.0
    if tc_values is not None:
        tc_sc = tc_values[sc_mask]
        largest_id = cluster_ids[np.argmax(cluster_sizes)]
        largest_tc = tc_sc[labels == largest_id]
        tc_range_largest = float(largest_tc.max() - largest_tc.min())

    return {
        'hdbscan_n_clusters': n_clusters,
        'hdbscan_noise_fraction': noise_fraction,
        'hdbscan_silhouette': sil_score,
        'hdbscan_largest_cluster_fraction': largest_fraction,
        'hdbscan_tc_range_largest': tc_range_largest,
    }


def compute_hdbscan_full(
    z: np.ndarray,
    is_sc: np.ndarray,
    min_cluster_size: int = 100,
    pca_dims: int = 20,
    tc_values: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Tuple[Dict[str, float], np.ndarray, Optional[Dict], np.ndarray]:
    """
    Full HDBSCAN analysis with per-sample labels and cluster statistics.

    Args:
        z: [N, D] latent vectors
        is_sc: [N] labels
        min_cluster_size: HDBSCAN min_cluster_size
        pca_dims: PCA dimensions for pre-reduction
        tc_values: [N] Tc values (optional)
        random_state: Seed

    Returns:
        (metrics, labels_full [N] with -2=non-SC -1=noise 0..k=cluster,
         cluster_stats or None, z_pca_sc [N_sc, pca_dims])
    """
    sc_mask = is_sc.astype(bool)
    z_sc = z[sc_mask]

    if len(z_sc) < min_cluster_size * 2:
        labels_full = np.full(len(z), -2, dtype=np.int32)
        z_pca = np.zeros((len(z_sc), min(pca_dims, z.shape[1])), dtype=np.float32)
        return _empty_metrics(), labels_full, None, z_pca

    # PCA pre-reduction
    pca = PCA(n_components=min(pca_dims, z_sc.shape[1]), random_state=random_state)
    z_pca = pca.fit_transform(z_sc)

    # HDBSCAN
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        n_jobs=-1,
        store_centers='centroid',
    )
    sc_labels = hdb.fit_predict(z_pca)

    # Map to full dataset: -2 = non-SC, -1 = noise, 0+ = cluster
    labels_full = np.full(len(z), -2, dtype=np.int32)
    labels_full[sc_mask] = sc_labels

    # Compute scalar metrics inline (avoids re-fitting HDBSCAN)
    cluster_ids = sorted(set(sc_labels) - {-1})
    n_clusters = len(cluster_ids)
    n_noise = int((sc_labels == -1).sum())
    noise_fraction = n_noise / len(sc_labels)

    if n_clusters >= 2:
        valid = sc_labels != -1
        rng = np.random.RandomState(random_state)
        if valid.sum() > 10000:
            sil_idx = rng.choice(np.where(valid)[0], 10000, replace=False)
            sil_score = float(silhouette_score(z_pca[sil_idx], sc_labels[sil_idx]))
        else:
            sil_score = float(silhouette_score(z_pca[valid], sc_labels[valid]))
    else:
        sil_score = 0.0

    cluster_sizes = [(sc_labels == c).sum() for c in cluster_ids]
    largest_fraction = max(cluster_sizes) / len(sc_labels) if cluster_sizes else 0.0

    tc_range_largest = 0.0
    if tc_values is not None and cluster_ids:
        tc_sc_flat = tc_values[sc_mask]
        largest_id = cluster_ids[np.argmax(cluster_sizes)]
        largest_tc = tc_sc_flat[sc_labels == largest_id]
        tc_range_largest = float(largest_tc.max() - largest_tc.min())

    metrics = {
        'hdbscan_n_clusters': n_clusters,
        'hdbscan_noise_fraction': noise_fraction,
        'hdbscan_silhouette': sil_score,
        'hdbscan_largest_cluster_fraction': largest_fraction,
        'hdbscan_tc_range_largest': tc_range_largest,
    }

    # Per-cluster statistics
    cluster_stats = None
    if cluster_ids and tc_values is not None:
        tc_sc = tc_values[sc_mask]
        cluster_stats = {}

        for c in cluster_ids:
            cmask = sc_labels == c
            cluster_z = z_pca[cmask]
            cluster_stats[f'cluster_{c}'] = {
                'count': int(cmask.sum()),
                'tc_mean': float(tc_sc[cmask].mean()),
                'tc_std': float(tc_sc[cmask].std()),
                'tc_min': float(tc_sc[cmask].min()),
                'tc_max': float(tc_sc[cmask].max()),
                'centroid': hdb.centroids_[c].tolist() if hasattr(hdb, 'centroids_') else None,
                'pca_spread': float(np.linalg.norm(cluster_z - cluster_z.mean(axis=0), axis=1).mean()),
            }

        # Noise stats
        noise_mask = sc_labels == -1
        if noise_mask.sum() > 0:
            cluster_stats['noise'] = {
                'count': int(noise_mask.sum()),
                'tc_mean': float(tc_sc[noise_mask].mean()),
                'tc_std': float(tc_sc[noise_mask].std()),
                'tc_min': float(tc_sc[noise_mask].min()),
                'tc_max': float(tc_sc[noise_mask].max()),
            }

    return metrics, labels_full, cluster_stats, z_pca


def _empty_metrics() -> Dict[str, float]:
    return {
        'hdbscan_n_clusters': 0,
        'hdbscan_noise_fraction': 0.0,
        'hdbscan_silhouette': 0.0,
        'hdbscan_largest_cluster_fraction': 0.0,
        'hdbscan_tc_range_largest': 0.0,
    }
