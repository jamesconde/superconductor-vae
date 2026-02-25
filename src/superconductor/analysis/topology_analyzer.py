"""
Main orchestrator for latent space topology analysis.

Provides TopologyAnalyzer class that wires all sub-analyzers and produces:
1. TopologySnapshot: ~30 scalar metrics (fast, ~10-15s) for per-epoch monitoring
2. Full metadata: per-sample tensors for detailed analysis on best checkpoints

Two-tier output:
- Compact: outputs/topology_summary.jsonl (JSONL, one dict per line)
- Full: outputs/topology_metadata_epochNNNN.pt (torch tensors)
"""

import json
import time
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.neighbors import NearestNeighbors

from .pca_spectrum import compute_pca_spectrum
from .intrinsic_dimension import (
    compute_intrinsic_dimension_mle_subset,
    compute_correlation_dimension,
)
from .density_estimator import compute_knn_density, compute_per_sample_density
from .boundary_detector import compute_boundary_metrics, compute_per_sample_boundary
from .cluster_topology import compute_cluster_metrics, compute_cluster_full
from .hdbscan_topology import compute_hdbscan_metrics, compute_hdbscan_full
from .distance_distribution import compute_distance_distribution


@dataclass
class TopologySnapshot:
    """~30 scalar metrics characterizing latent space topology at one epoch."""

    epoch: int = -1
    timestamp: str = ''
    n_samples: int = 0
    compute_time_seconds: float = 0.0

    # Intrinsic dimensionality
    intrinsic_dim_mle: float = 0.0
    intrinsic_dim_mle_sc: float = 0.0
    intrinsic_dim_mle_nonsc: float = 0.0
    intrinsic_dim_correlation: float = 0.0

    # PCA spectrum
    pca_effective_rank: float = 0.0
    pca_variance_top10: float = 0.0
    pca_variance_top50: float = 0.0
    pca_anisotropy: float = 0.0

    # k-NN density
    knn_mean_distance: float = 0.0
    knn_median_distance: float = 0.0
    knn_distance_std: float = 0.0
    density_contrast_sc_nonsc: float = 0.0

    # SC/non-SC boundary
    boundary_thickness: float = 0.0
    boundary_n_samples: int = 0
    sc_nonsc_centroid_distance: float = 0.0
    sc_nonsc_separation_ratio: float = 0.0

    # KMeans cluster topology (SC subset, fixed k=9)
    n_clusters_sc: int = 0
    silhouette_score_sc: float = 0.0
    inter_cluster_distance_mean: float = 0.0
    intra_cluster_distance_mean: float = 0.0

    # HDBSCAN density-based clustering (SC subset, auto k)
    hdbscan_n_clusters: int = 0
    hdbscan_noise_fraction: float = 0.0
    hdbscan_silhouette: float = 0.0
    hdbscan_largest_cluster_fraction: float = 0.0
    hdbscan_tc_range_largest: float = 0.0

    # Pairwise distance distribution
    pairwise_dist_mean: float = 0.0
    pairwise_dist_std: float = 0.0
    pairwise_dist_skewness: float = 0.0
    pairwise_dist_kurtosis: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to plain dict for JSON serialization."""
        return asdict(self)

    def summary_str(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"=== Topology Snapshot (epoch {self.epoch}) ===",
            f"  Samples: {self.n_samples}, Compute time: {self.compute_time_seconds:.1f}s",
            f"",
            f"  Intrinsic Dim (MLE):      {self.intrinsic_dim_mle:.1f}  (SC: {self.intrinsic_dim_mle_sc:.1f}, non-SC: {self.intrinsic_dim_mle_nonsc:.1f})",
            f"  Intrinsic Dim (Corr):     {self.intrinsic_dim_correlation:.1f}",
            f"  PCA Effective Rank:       {self.pca_effective_rank:.1f}",
            f"  PCA Var Top 10/50:        {self.pca_variance_top10*100:.1f}% / {self.pca_variance_top50*100:.1f}%",
            f"  PCA Anisotropy:           {self.pca_anisotropy:.1f}",
            f"",
            f"  k-NN Distance (mean):     {self.knn_mean_distance:.4f}",
            f"  Density Contrast SC/nSC:  {self.density_contrast_sc_nonsc:.3f}",
            f"",
            f"  Boundary Samples:         {self.boundary_n_samples}",
            f"  Boundary Thickness:       {self.boundary_thickness:.4f}",
            f"  Centroid Distance SC/nSC: {self.sc_nonsc_centroid_distance:.4f}",
            f"  Separation Ratio:         {self.sc_nonsc_separation_ratio:.4f}",
            f"",
            f"  KMeans (k=9) Silhouette:  {self.silhouette_score_sc:.4f}",
            f"  Inter/Intra Cluster:      {self.inter_cluster_distance_mean:.4f} / {self.intra_cluster_distance_mean:.4f}",
            f"",
            f"  HDBSCAN Clusters:         {self.hdbscan_n_clusters}  (noise: {self.hdbscan_noise_fraction*100:.1f}%)",
            f"  HDBSCAN Silhouette:       {self.hdbscan_silhouette:.4f}",
            f"  HDBSCAN Largest Cluster:  {self.hdbscan_largest_cluster_fraction*100:.1f}%",
            f"",
            f"  Pairwise Dist Mean/Std:   {self.pairwise_dist_mean:.4f} / {self.pairwise_dist_std:.4f}",
            f"  Pairwise Skew/Kurt:       {self.pairwise_dist_skewness:.4f} / {self.pairwise_dist_kurtosis:.4f}",
        ]
        return '\n'.join(lines)


class TopologyAnalyzer:
    """
    Orchestrates all latent space topology analyses.

    Usage:
        analyzer = TopologyAnalyzer(k=20, use_gpu=True)

        # Fast compact analysis (~10-15s)
        snapshot = analyzer.analyze_compact(z_vectors, is_sc, epoch=100)
        analyzer.save_snapshot(snapshot, 'outputs/topology_summary.jsonl')

        # Full analysis with per-sample metadata (~60-70s)
        snapshot, metadata = analyzer.analyze_full(z_vectors, is_sc, tc_values, epoch=100)
        analyzer.save_full_metadata(metadata, 'outputs/topology_metadata_epoch0100.pt')
    """

    def __init__(
        self,
        k: int = 20,
        n_clusters: int = 9,
        use_gpu: bool = True,
        random_state: int = 42,
        hdbscan_min_cluster_size: int = 100,
        hdbscan_pca_dims: int = 20,
    ):
        self.k = k
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_pca_dims = hdbscan_pca_dims

    def _fit_knn(self, z_vectors: np.ndarray) -> Tuple[Optional[NearestNeighbors], np.ndarray, np.ndarray]:
        """
        Compute k-NN distances and indices once. All sub-analyzers reuse these.

        Uses GPU-accelerated chunked torch.cdist when available (~2-3s),
        falling back to sklearn brute force on CPU (~65s).

        Returns:
            (knn_model_or_None, knn_distances [N, k+1], knn_indices [N, k+1])
        """
        k_total = self.k + 1  # +1 for self

        if self.use_gpu and torch.cuda.is_available():
            try:
                return self._fit_knn_gpu(z_vectors, k_total)
            except RuntimeError:
                pass  # OOM, fall back to CPU

        # CPU fallback via sklearn
        knn_model = NearestNeighbors(
            n_neighbors=k_total, algorithm='brute', metric='euclidean'
        )
        knn_model.fit(z_vectors)
        knn_distances, knn_indices = knn_model.kneighbors(z_vectors)
        return knn_model, knn_distances, knn_indices

    def _fit_knn_gpu(
        self, z_vectors: np.ndarray, k_total: int
    ) -> Tuple[None, np.ndarray, np.ndarray]:
        """GPU k-NN via chunked torch.cdist + topk. ~25x faster than CPU sklearn."""
        device = torch.device('cuda')
        z_gpu = torch.from_numpy(z_vectors).float().to(device)
        n = len(z_gpu)

        # Chunk size tuned for 8GB VRAM: chunk_size * N * 4 bytes < ~2GB
        # 2048 * 50K * 4 = ~400MB per chunk (safe)
        chunk_size = 2048
        all_dists = []
        all_indices = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = z_gpu[start:end]
            d = torch.cdist(chunk, z_gpu)  # [chunk_size, N]
            topk_vals, topk_idx = d.topk(k_total, dim=1, largest=False)
            all_dists.append(topk_vals.cpu())
            all_indices.append(topk_idx.cpu())
            del d

        knn_distances = torch.cat(all_dists, dim=0).numpy()
        knn_indices = torch.cat(all_indices, dim=0).numpy()

        del z_gpu
        torch.cuda.empty_cache()

        return None, knn_distances, knn_indices

    def analyze_compact(
        self,
        z_vectors: np.ndarray,
        is_sc: np.ndarray,
        epoch: int = -1,
        include_hdbscan: bool = False,
    ) -> TopologySnapshot:
        """
        Fast compact analysis producing ~25-30 scalar metrics.

        Skips correlation dimension and distance distribution for speed.
        HDBSCAN is off by default in compact mode (~35s extra); use
        include_hdbscan=True or --hdbscan CLI flag to enable.
        Target: ~10s without HDBSCAN, ~45s with.

        Args:
            z_vectors: [N, D] latent vectors (numpy)
            is_sc: [N] SC labels (0/1)
            epoch: Training epoch number
            include_hdbscan: Include HDBSCAN clustering (~35s extra)

        Returns:
            TopologySnapshot with all scalar metrics
        """
        t_start = time.time()
        snapshot = TopologySnapshot(
            epoch=epoch,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            n_samples=len(z_vectors),
        )

        # 1. PCA spectrum (GPU, ~2-5s)
        pca_metrics = compute_pca_spectrum(z_vectors, use_gpu=self.use_gpu)
        _update_snapshot(snapshot, pca_metrics)

        # 2. k-NN fit + query — SINGLE call, reused by all downstream analyses
        knn_model, knn_distances, knn_indices = self._fit_knn(z_vectors)

        # 3. Intrinsic dim MLE (reuses precomputed knn_distances)
        id_metrics, per_point_dim, _ = compute_intrinsic_dimension_mle_subset(
            z_vectors, is_sc, k=self.k, knn_distances=knn_distances
        )
        _update_snapshot(snapshot, id_metrics)

        # 4. k-NN density (reuses knn_distances)
        density_metrics = compute_knn_density(
            knn_distances, is_sc,
            intrinsic_dim=snapshot.intrinsic_dim_mle,
            k=self.k,
        )
        _update_snapshot(snapshot, density_metrics)

        # 5. Boundary detection (reuses precomputed knn_distances + indices)
        boundary_metrics = compute_boundary_metrics(
            z_vectors, is_sc, k=self.k,
            knn_distances=knn_distances, knn_indices=knn_indices,
        )
        _update_snapshot(snapshot, boundary_metrics)

        # 6. KMeans cluster topology (SC subset, ~2s)
        cluster_metrics = compute_cluster_metrics(
            z_vectors, is_sc, n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        _update_snapshot(snapshot, cluster_metrics)

        # 7. HDBSCAN density-based clustering (optional, ~35s)
        if include_hdbscan:
            hdbscan_metrics = compute_hdbscan_metrics(
                z_vectors, is_sc,
                min_cluster_size=self.hdbscan_min_cluster_size,
                pca_dims=self.hdbscan_pca_dims,
                random_state=self.random_state,
            )
            _update_snapshot(snapshot, hdbscan_metrics)

        snapshot.compute_time_seconds = time.time() - t_start
        return snapshot

    def analyze_full(
        self,
        z_vectors: np.ndarray,
        is_sc: np.ndarray,
        tc_values: Optional[np.ndarray] = None,
        epoch: int = -1,
    ) -> Tuple[TopologySnapshot, Dict]:
        """
        Full analysis producing scalar metrics + per-sample metadata.

        Includes correlation dimension, distance distribution, and HDBSCAN
        with per-sample labels and cluster statistics.
        Target: ~70s on 50K x 2048 data with GPU.

        Args:
            z_vectors: [N, D] latent vectors (numpy)
            is_sc: [N] SC labels
            tc_values: [N] Tc values (optional, for cluster Tc stats)
            epoch: Training epoch

        Returns:
            (TopologySnapshot, full_metadata_dict)
        """
        t_start = time.time()
        snapshot = TopologySnapshot(
            epoch=epoch,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            n_samples=len(z_vectors),
        )

        # 1. PCA spectrum
        pca_metrics = compute_pca_spectrum(z_vectors, use_gpu=self.use_gpu)
        _update_snapshot(snapshot, pca_metrics)

        # 2. k-NN fit + query — SINGLE call
        knn_model, knn_distances, knn_indices = self._fit_knn(z_vectors)

        # 3. Intrinsic dim MLE (reuses knn_distances)
        id_metrics, per_point_dim, _ = compute_intrinsic_dimension_mle_subset(
            z_vectors, is_sc, k=self.k, knn_distances=knn_distances
        )
        _update_snapshot(snapshot, id_metrics)

        # 4. Correlation dimension (slower, on subsample — independent of k-NN)
        corr_metrics = compute_correlation_dimension(
            z_vectors, n_subsample=5000,
            rng=np.random.RandomState(self.random_state),
        )
        _update_snapshot(snapshot, corr_metrics)

        # 5. k-NN density
        density_metrics = compute_knn_density(
            knn_distances, is_sc,
            intrinsic_dim=snapshot.intrinsic_dim_mle,
            k=self.k,
        )
        _update_snapshot(snapshot, density_metrics)
        per_sample_density = compute_per_sample_density(
            knn_distances, snapshot.intrinsic_dim_mle, k=self.k,
        )

        # 6. Boundary detection (reuses precomputed distances + indices)
        boundary_metrics = compute_boundary_metrics(
            z_vectors, is_sc, k=self.k,
            knn_distances=knn_distances, knn_indices=knn_indices,
        )
        _update_snapshot(snapshot, boundary_metrics)
        heterogeneity, boundary_flag = compute_per_sample_boundary(
            z_vectors, is_sc, k=self.k,
            knn_distances=knn_distances, knn_indices=knn_indices,
        )

        # 7. KMeans cluster topology
        cluster_metrics, cluster_labels, cluster_tc_stats = compute_cluster_full(
            z_vectors, is_sc, n_clusters=self.n_clusters,
            tc_values=tc_values, random_state=self.random_state,
        )
        _update_snapshot(snapshot, cluster_metrics)

        # 8. HDBSCAN density-based clustering
        hdbscan_metrics, hdbscan_labels, hdbscan_cluster_stats, _ = compute_hdbscan_full(
            z_vectors, is_sc,
            min_cluster_size=self.hdbscan_min_cluster_size,
            pca_dims=self.hdbscan_pca_dims,
            tc_values=tc_values, random_state=self.random_state,
        )
        _update_snapshot(snapshot, hdbscan_metrics)

        # 9. Distance distribution
        dist_metrics = compute_distance_distribution(
            z_vectors, is_sc, n_subsample=5000,
            rng=np.random.RandomState(self.random_state),
        )
        _update_snapshot(snapshot, dist_metrics)

        snapshot.compute_time_seconds = time.time() - t_start

        # Build per-sample metadata dict
        metadata = {
            'epoch': epoch,
            'per_point_intrinsic_dim': torch.from_numpy(per_point_dim).float(),
            'per_sample_density': torch.from_numpy(per_sample_density).float(),
            'heterogeneity': torch.from_numpy(heterogeneity).float(),
            'boundary_flag': torch.from_numpy(boundary_flag.astype(np.int8)),
            'cluster_labels_kmeans': torch.from_numpy(cluster_labels),
            'cluster_labels_hdbscan': torch.from_numpy(hdbscan_labels),
            'snapshot': snapshot.to_dict(),
        }
        if cluster_tc_stats is not None:
            metadata['cluster_tc_stats_kmeans'] = cluster_tc_stats
        if hdbscan_cluster_stats is not None:
            metadata['cluster_tc_stats_hdbscan'] = hdbscan_cluster_stats

        return snapshot, metadata

    @staticmethod
    def load_z_cache(cache_path: str | Path) -> Dict:
        """
        Load z-cache from disk (same format as training saves).

        Returns dict with z_vectors, tc_values, is_sc, epoch, etc.
        """
        cache_path = Path(cache_path)
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        return cache

    @staticmethod
    def save_snapshot(snapshot: TopologySnapshot, jsonl_path: str | Path):
        """Append topology snapshot to JSONL file."""
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(snapshot.to_dict()) + '\n')

    @staticmethod
    def save_full_metadata(metadata: Dict, pt_path: str | Path):
        """Save full per-sample metadata as .pt file."""
        pt_path = Path(pt_path)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(metadata, pt_path)


def _update_snapshot(snapshot: TopologySnapshot, metrics: Dict[str, float]):
    """Update snapshot fields from a metrics dict."""
    for key, value in metrics.items():
        if hasattr(snapshot, key):
            setattr(snapshot, key, value)
