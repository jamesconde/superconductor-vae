"""
Cluster topology analysis for superconductor latent space.

Performs KMeans clustering on SC subset (k=9 matching 9 SC families),
computes silhouette scores, inter/intra-cluster distances, and per-cluster
Tc statistics.

Tests whether SC families (YBCO, LSCO, Bi-cuprate, etc.) form distinct
clusters in the learned latent space.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def compute_cluster_metrics(
    z: np.ndarray,
    is_sc: np.ndarray,
    n_clusters: int = 9,
    tc_values: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Cluster SC subset and compute topology metrics.

    Args:
        z: [N, D] latent vectors (full dataset)
        is_sc: [N] boolean/int array
        n_clusters: Number of clusters (9 = SC family count)
        tc_values: [N] Tc values for per-cluster statistics
        random_state: KMeans seed

    Returns:
        Dict with n_clusters_sc, silhouette_score_sc,
        inter_cluster_distance_mean, intra_cluster_distance_mean
    """
    sc_mask = is_sc.astype(bool)
    z_sc = z[sc_mask]

    if len(z_sc) < n_clusters + 1:
        return {
            'n_clusters_sc': 0,
            'silhouette_score_sc': 0.0,
            'inter_cluster_distance_mean': 0.0,
            'intra_cluster_distance_mean': 0.0,
        }

    # Fit MiniBatchKMeans on SC subset (faster than full KMeans for large N)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3, batch_size=4096)
    labels = kmeans.fit_predict(z_sc)
    centroids = kmeans.cluster_centers_  # [k, D]

    # Silhouette score (subsample if large for speed)
    if len(z_sc) > 10000:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(z_sc), 10000, replace=False)
        sil_score = float(silhouette_score(z_sc[idx], labels[idx], metric='euclidean'))
    else:
        sil_score = float(silhouette_score(z_sc, labels, metric='euclidean'))

    # Inter-cluster distance: mean pairwise centroid distance
    n_c = len(centroids)
    inter_dists = []
    for i in range(n_c):
        for j in range(i + 1, n_c):
            inter_dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    inter_mean = float(np.mean(inter_dists)) if inter_dists else 0.0

    # Intra-cluster distance: mean distance to own centroid
    intra_dists = []
    for c in range(n_c):
        cluster_points = z_sc[labels == c]
        if len(cluster_points) > 0:
            dists_to_centroid = np.linalg.norm(cluster_points - centroids[c], axis=1)
            intra_dists.append(float(np.mean(dists_to_centroid)))
    intra_mean = float(np.mean(intra_dists)) if intra_dists else 0.0

    return {
        'n_clusters_sc': n_clusters,
        'silhouette_score_sc': sil_score,
        'inter_cluster_distance_mean': inter_mean,
        'intra_cluster_distance_mean': intra_mean,
    }


def compute_cluster_full(
    z: np.ndarray,
    is_sc: np.ndarray,
    n_clusters: int = 9,
    tc_values: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Tuple[Dict[str, float], np.ndarray, Optional[Dict]]:
    """
    Full cluster analysis with per-sample labels and Tc statistics.

    Args:
        z: [N, D] latent vectors
        is_sc: [N] labels
        n_clusters: Number of clusters
        tc_values: [N] Tc values (optional)
        random_state: Seed

    Returns:
        (metrics, cluster_labels_sc_only [N_sc], cluster_tc_stats or None)
        cluster_labels_sc_only has -1 for non-SC samples
    """
    sc_mask = is_sc.astype(bool)
    z_sc = z[sc_mask]

    if len(z_sc) < n_clusters + 1:
        labels_full = np.full(len(z), -1, dtype=np.int32)
        return {
            'n_clusters_sc': 0,
            'silhouette_score_sc': 0.0,
            'inter_cluster_distance_mean': 0.0,
            'intra_cluster_distance_mean': 0.0,
        }, labels_full, None

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3, batch_size=4096)
    sc_labels = kmeans.fit_predict(z_sc)

    # Map back to full dataset (-1 for non-SC)
    labels_full = np.full(len(z), -1, dtype=np.int32)
    labels_full[sc_mask] = sc_labels

    metrics = compute_cluster_metrics(z, is_sc, n_clusters, tc_values, random_state)

    # Per-cluster Tc statistics (if Tc provided)
    tc_stats = None
    if tc_values is not None:
        tc_sc = tc_values[sc_mask]
        tc_stats = {}
        for c in range(n_clusters):
            cluster_tc = tc_sc[sc_labels == c]
            if len(cluster_tc) > 0:
                tc_stats[f'cluster_{c}'] = {
                    'count': int(len(cluster_tc)),
                    'tc_mean': float(np.mean(cluster_tc)),
                    'tc_std': float(np.std(cluster_tc)),
                    'tc_min': float(np.min(cluster_tc)),
                    'tc_max': float(np.max(cluster_tc)),
                }

    return metrics, labels_full, tc_stats
