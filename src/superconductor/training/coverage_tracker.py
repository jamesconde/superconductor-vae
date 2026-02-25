"""
Phase 2: Latent Space Coverage Tracking.

Partitions the 2048-dim z-space into k clusters (default 64) using MiniBatchKMeans,
tracks per-cluster visit counts, and computes inverse-visit-count sampling weights
to bias future sampling toward underexplored regions.

Also tracks per-cluster quality metrics (valid formula rate, garbage rate) to identify:
  - Regions where the model produces garbage → latent geometry bounds
  - Regions with moderate quality → targets for additional self-supervised training

February 2026
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class CoverageTracker:
    """K-means coverage tracking for z-space sampling.

    Partitions z-space into k clusters, tracks visits per cluster, and produces
    inverse-visit-count sampling weights so underexplored regions get prioritized.

    Quality tracking: records per-cluster valid/total counts from filtering.
    Clusters with low valid rates are flagged as either:
      - 'boundary' (some valid, <50% rate) → extra self-supervised training targets
      - 'out_of_bounds' (near-zero valid rate) → model can't produce valid formulas here

    Usage:
        tracker = CoverageTracker(k=64, temperature=1.0, decay=0.995)
        tracker.fit(z_cache)  # Once per z-cache load

        # Each sub-epoch:
        weights = tracker.get_sampling_weights()  # [N_cached] for torch.multinomial
        z_sampled = z_cache[torch.multinomial(weights, n, replacement=True)]
        tracker.record_visits(z_sampled)
        tracker.record_quality(z_sampled, n_valid=..., valid_mask=...)
        tracker.decay_visits()
        metrics = tracker.get_metrics()

    State persistence follows EntropyManager.get_state()/load_state() pattern.
    """

    def __init__(
        self,
        k: int = 64,
        temperature: float = 1.0,
        decay: float = 0.995,
        device: Optional[torch.device] = None,
    ):
        """Initialize coverage tracker.

        Args:
            k: Number of clusters. 46K samples / 64 = ~720 per cluster.
            temperature: Sampling weight exponent. Higher = more uniform.
                w[cluster] = 1 / (1 + visit_count)^temperature
            decay: Visit count decay per sub-epoch. 0.995 = half-life ~139 sub-epochs.
            device: GPU device for centroid operations.
        """
        self.k = k
        self.temperature = temperature
        self.decay = decay
        self.device = device or torch.device('cpu')

        # State (populated by fit())
        self._fitted = False
        self._centroids = None        # [k, latent_dim] on GPU
        self._centroid_sq_norms = None # [k] precomputed for fast assignment
        self._cluster_labels = None   # [N_cached] cluster ID per training point
        self._n_cached = 0

        # Visit tracking
        self._visit_counts = None     # [k] float tensor

        # Quality tracking: per-cluster valid/total counts
        self._cluster_total = None    # [k] total samples generated from this cluster
        self._cluster_valid = None    # [k] valid samples (passed all filters)

        # Novelty tracking: per-cluster unique formula discovery
        self._cluster_seen_formulas = None     # list[set[str]] — cumulative unique formulas per cluster
        self._cluster_novel_this_epoch = None  # [k] NEW unique formulas this sub-epoch
        self._cluster_total_this_epoch = None  # [k] ALL valid formulas this sub-epoch

    def fit(self, z_cache: torch.Tensor):
        """Fit k-means on training z-vectors and assign cluster labels.

        Args:
            z_cache: [N, latent_dim] cached training z-vectors.
                     Typically 46K x 2048. Runs MiniBatchKMeans on CPU (~2-3s).
        """
        from sklearn.cluster import MiniBatchKMeans

        n_samples = z_cache.shape[0]
        self._n_cached = n_samples

        # Clamp k to actual sample count
        effective_k = min(self.k, n_samples)

        # Fit on CPU numpy
        z_np = z_cache.detach().cpu().float().numpy()
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            random_state=42,
            n_init=3,
            batch_size=min(4096, n_samples),
        )
        kmeans.fit(z_np)

        # Store centroids on GPU for fast assignment
        self._centroids = torch.from_numpy(
            kmeans.cluster_centers_
        ).float().to(self.device)  # [k, latent_dim]
        self._centroid_sq_norms = (self._centroids ** 2).sum(dim=1)  # [k]

        # Assign training points to clusters
        self._cluster_labels = torch.from_numpy(
            kmeans.labels_
        ).long().to(self.device)  # [N]

        # Initialize visit counts (zero = never visited)
        self._visit_counts = torch.zeros(effective_k, device=self.device)

        # Initialize quality counters
        self._cluster_total = torch.zeros(effective_k, device=self.device)
        self._cluster_valid = torch.zeros(effective_k, device=self.device)

        # Initialize novelty tracking
        self._cluster_seen_formulas = [set() for _ in range(effective_k)]
        self._cluster_novel_this_epoch = torch.zeros(effective_k, device=self.device)
        self._cluster_total_this_epoch = torch.zeros(effective_k, device=self.device)

        self._fitted = True
        self.k = effective_k  # Update in case clamped

    def assign(self, z: torch.Tensor) -> torch.Tensor:
        """Assign z-vectors to nearest cluster centroid (GPU matmul).

        Uses ||z-c||^2 = ||z||^2 + ||c||^2 - 2*z@c^T to avoid explicit
        distance computation. Single matmul, <0.1ms for typical batch sizes.

        Args:
            z: [N, latent_dim] z-vectors on GPU

        Returns:
            [N] cluster IDs (long tensor)
        """
        if not self._fitted:
            raise RuntimeError("CoverageTracker not fitted. Call fit() first.")

        z = z.to(self.device)
        # ||z-c||^2 = ||z||^2 + ||c||^2 - 2*z@c^T
        z_sq_norms = (z ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        # [N, k] = [N, 1] + [1, k] - 2*[N, latent_dim]@[latent_dim, k]
        dists = z_sq_norms + self._centroid_sq_norms.unsqueeze(0) - 2.0 * z @ self._centroids.T
        return dists.argmin(dim=1)  # [N]

    def record_visits(self, z: torch.Tensor):
        """Record visits for sampled z-vectors.

        Args:
            z: [N, latent_dim] sampled z-vectors
        """
        if not self._fitted:
            return

        cluster_ids = self.assign(z)
        # bincount to increment per-cluster
        counts = torch.bincount(cluster_ids, minlength=self.k).float()
        self._visit_counts += counts

    def record_quality(
        self,
        z_sampled: torch.Tensor,
        valid_mask: torch.Tensor,
        valid_formulas: Optional[List[str]] = None,
        valid_idx: Optional[List[int]] = None,
    ):
        """Record per-cluster quality and novelty from filtering results.

        Args:
            z_sampled: [N, latent_dim] all sampled z-vectors (before filtering)
            valid_mask: [N] boolean tensor — True for samples that passed all filters
            valid_formulas: List of valid formula strings (same order as valid_idx)
            valid_idx: Indices into z_sampled for valid formulas
        """
        if not self._fitted:
            return

        cluster_ids = self.assign(z_sampled)
        n_total = torch.bincount(cluster_ids, minlength=self.k).float()
        n_valid = torch.bincount(
            cluster_ids[valid_mask], minlength=self.k
        ).float()
        self._cluster_total += n_total
        self._cluster_valid += n_valid

        # Track per-cluster novelty (new unique formulas)
        if (valid_formulas is not None and valid_idx is not None
                and self._cluster_seen_formulas is not None):
            for i, formula in enumerate(valid_formulas):
                if i >= len(valid_idx):
                    break
                sample_idx = valid_idx[i]
                cid = cluster_ids[sample_idx].item()
                self._cluster_total_this_epoch[cid] += 1
                if formula not in self._cluster_seen_formulas[cid]:
                    self._cluster_seen_formulas[cid].add(formula)
                    self._cluster_novel_this_epoch[cid] += 1

    def reset_epoch_novelty(self):
        """Reset per-sub-epoch novelty counters. Call at start of each sub-epoch.

        The cumulative _cluster_seen_formulas persists — only the epoch counters reset.
        """
        if self._cluster_novel_this_epoch is not None:
            self._cluster_novel_this_epoch.zero_()
        if self._cluster_total_this_epoch is not None:
            self._cluster_total_this_epoch.zero_()

    def get_sampling_weights(self) -> torch.Tensor:
        """Compute per-training-point sampling weights for torch.multinomial.

        Maps cluster-level inverse-visit-count weights to individual training points
        via their cluster assignments.

        Returns:
            [N_cached] normalized weights (sums to 1.0). For torch.multinomial.
        """
        if not self._fitted:
            # Uniform fallback
            return torch.ones(self._n_cached, device=self.device) / self._n_cached

        # Cluster-level weights: w = novelty_factor / (1 + visits)^temperature
        # novelty_factor = max(0.1, novel / max(1, total)) per cluster
        # Saturated clusters get floor of 10% weight (not zero — model may improve)
        # Clusters with no data yet default to 1.0 (max priority for unexplored)
        cluster_weights = 1.0 / (1.0 + self._visit_counts) ** self.temperature

        if (self._cluster_novel_this_epoch is not None
                and self._cluster_total_this_epoch is not None):
            total_epoch = self._cluster_total_this_epoch.clamp(min=1.0)
            novelty_rate = self._cluster_novel_this_epoch / total_epoch
            # Clusters with no data this epoch → novelty_factor = 1.0 (unexplored)
            no_data = (self._cluster_total_this_epoch < 0.5)
            novelty_factor = novelty_rate.clamp(min=0.1)
            novelty_factor[no_data] = 1.0
            cluster_weights = cluster_weights * novelty_factor

        # Map to per-point weights
        point_weights = cluster_weights[self._cluster_labels]  # [N_cached]

        # Normalize for torch.multinomial
        total = point_weights.sum()
        if total > 0:
            point_weights = point_weights / total
        else:
            point_weights = torch.ones_like(point_weights) / len(point_weights)

        return point_weights

    def decay_visits(self):
        """Apply exponential decay to visit counts. Call once per sub-epoch.

        Decay factor 0.995 → half-life ~139 sub-epochs.
        Prevents early visits from permanently "satisfying" a cluster.
        """
        if self._fitted and self._visit_counts is not None:
            self._visit_counts *= self.decay

    def get_metrics(self) -> Dict[str, float]:
        """Compute coverage metrics for logging.

        Returns:
            Dict with coverage metrics:
              - coverage_fraction: clusters visited / total (0-1)
              - coverage_clusters_visited: count of visited clusters
              - coverage_visit_gini: Gini coefficient (0=uniform, 1=concentrated)
              - coverage_visit_entropy_normalized: entropy / log(k), 1.0 = uniform
              - coverage_total_visits: cumulative z-vectors sampled (after decay)
              - coverage_min_visits / coverage_max_visits: cluster visit range
              - coverage_garbage_clusters: clusters with >10 samples and <10% valid rate
              - coverage_boundary_clusters: clusters with >10 samples and 10-50% valid rate
        """
        if not self._fitted:
            return {}

        visits = self._visit_counts.cpu().numpy()
        k = len(visits)

        # Clusters visited (visit_count > 0.5 after decay)
        visited = (visits > 0.5).sum()

        # Gini coefficient
        sorted_v = np.sort(visits)
        n = len(sorted_v)
        if sorted_v.sum() > 0:
            index = np.arange(1, n + 1)
            gini = (2 * (index * sorted_v).sum()) / (n * sorted_v.sum()) - (n + 1) / n
        else:
            gini = 0.0

        # Normalized entropy
        total_v = visits.sum()
        if total_v > 0:
            probs = visits / total_v
            probs = probs[probs > 0]
            entropy = -(probs * np.log(probs)).sum()
            max_entropy = math.log(k) if k > 1 else 1.0
            norm_entropy = entropy / max_entropy
        else:
            norm_entropy = 0.0

        # Quality metrics
        totals = self._cluster_total.cpu().numpy()
        valids = self._cluster_valid.cpu().numpy()
        has_samples = totals > 10  # Need enough samples for meaningful rate
        if has_samples.any():
            rates = np.zeros_like(totals)
            rates[has_samples] = valids[has_samples] / totals[has_samples]
            garbage_clusters = int(((rates < 0.10) & has_samples).sum())
            boundary_clusters = int(((rates >= 0.10) & (rates < 0.50) & has_samples).sum())
        else:
            garbage_clusters = 0
            boundary_clusters = 0

        # Novelty metrics
        avg_novelty = 0.0
        saturated_clusters = 0
        productive_clusters = 0
        total_unique_formulas = 0

        if self._cluster_seen_formulas is not None:
            total_unique_formulas = sum(len(s) for s in self._cluster_seen_formulas)

        if (self._cluster_novel_this_epoch is not None
                and self._cluster_total_this_epoch is not None):
            novel_np = self._cluster_novel_this_epoch.cpu().numpy()
            total_ep_np = self._cluster_total_this_epoch.cpu().numpy()
            has_data = total_ep_np > 0.5
            if has_data.any():
                rates_nov = novel_np[has_data] / total_ep_np[has_data]
                avg_novelty = float(rates_nov.mean())
                saturated_clusters = int((rates_nov < 0.05).sum())
                productive_clusters = int((rates_nov >= 0.20).sum())

        return {
            'coverage_fraction': float(visited / k),
            'coverage_clusters_visited': int(visited),
            'coverage_visit_gini': float(gini),
            'coverage_visit_entropy_normalized': float(norm_entropy),
            'coverage_total_visits': float(total_v),
            'coverage_min_visits': float(visits.min()),
            'coverage_max_visits': float(visits.max()),
            'coverage_garbage_clusters': garbage_clusters,
            'coverage_boundary_clusters': boundary_clusters,
            'coverage_avg_novelty': avg_novelty,
            'coverage_saturated_clusters': saturated_clusters,
            'coverage_productive_clusters': productive_clusters,
            'coverage_total_unique_formulas': total_unique_formulas,
        }

    def get_cluster_quality_report(self) -> Dict[str, object]:
        """Detailed per-cluster quality report for analysis.

        Returns dict with:
          - cluster_rates: [k] valid rate per cluster
          - garbage_ids: list of cluster IDs with <10% valid rate (latent bounds)
          - boundary_ids: list of cluster IDs with 10-50% valid rate (training targets)
          - good_ids: list of cluster IDs with >=50% valid rate
        """
        if not self._fitted:
            return {'cluster_rates': [], 'garbage_ids': [], 'boundary_ids': [], 'good_ids': []}

        totals = self._cluster_total.cpu().numpy()
        valids = self._cluster_valid.cpu().numpy()
        rates = np.zeros_like(totals)
        has_samples = totals > 10
        rates[has_samples] = valids[has_samples] / totals[has_samples]

        garbage_ids = [int(i) for i in range(self.k) if has_samples[i] and rates[i] < 0.10]
        boundary_ids = [int(i) for i in range(self.k) if has_samples[i] and 0.10 <= rates[i] < 0.50]
        good_ids = [int(i) for i in range(self.k) if has_samples[i] and rates[i] >= 0.50]

        return {
            'cluster_rates': rates.tolist(),
            'garbage_ids': garbage_ids,
            'boundary_ids': boundary_ids,
            'good_ids': good_ids,
        }

    def get_state(self) -> Dict:
        """Get tracker state for checkpoint serialization.

        Returns dict safe for torch.save(). Follows EntropyManager pattern.
        """
        if not self._fitted:
            return {'fitted': False}

        state = {
            'fitted': True,
            'k': self.k,
            'temperature': self.temperature,
            'decay': self.decay,
            'centroids': self._centroids.cpu(),          # [k, latent_dim]
            'cluster_labels': self._cluster_labels.cpu(), # [N_cached]
            'visit_counts': self._visit_counts.cpu(),     # [k]
            'cluster_total': self._cluster_total.cpu(),   # [k]
            'cluster_valid': self._cluster_valid.cpu(),   # [k]
            'n_cached': self._n_cached,
        }

        # Novelty tracking: sets aren't weights_only=True safe, convert to list[list[str]]
        if self._cluster_seen_formulas is not None:
            state['cluster_seen_formulas'] = [
                list(s) for s in self._cluster_seen_formulas
            ]
        if self._cluster_novel_this_epoch is not None:
            state['cluster_novel_this_epoch'] = self._cluster_novel_this_epoch.cpu()
        if self._cluster_total_this_epoch is not None:
            state['cluster_total_this_epoch'] = self._cluster_total_this_epoch.cpu()

        return state

    def load_state(self, state: Dict):
        """Restore tracker state from checkpoint.

        Args:
            state: Dict from get_state(), loaded from checkpoint.
        """
        if not state.get('fitted', False):
            self._fitted = False
            return

        self.k = state['k']
        self.temperature = state.get('temperature', self.temperature)
        self.decay = state.get('decay', self.decay)
        self._n_cached = state['n_cached']

        self._centroids = state['centroids'].float().to(self.device)
        self._centroid_sq_norms = (self._centroids ** 2).sum(dim=1)
        self._cluster_labels = state['cluster_labels'].long().to(self.device)
        self._visit_counts = state['visit_counts'].float().to(self.device)
        self._cluster_total = state.get(
            'cluster_total', torch.zeros(self.k, device=self.device)
        ).float().to(self.device)
        self._cluster_valid = state.get(
            'cluster_valid', torch.zeros(self.k, device=self.device)
        ).float().to(self.device)

        # Restore novelty tracking (backward compat: old checkpoints lack these)
        seen_raw = state.get('cluster_seen_formulas')
        if seen_raw is not None:
            self._cluster_seen_formulas = [set(lst) for lst in seen_raw]
        else:
            self._cluster_seen_formulas = [set() for _ in range(self.k)]

        self._cluster_novel_this_epoch = state.get(
            'cluster_novel_this_epoch',
            torch.zeros(self.k, device=self.device),
        )
        if isinstance(self._cluster_novel_this_epoch, torch.Tensor):
            self._cluster_novel_this_epoch = self._cluster_novel_this_epoch.float().to(self.device)

        self._cluster_total_this_epoch = state.get(
            'cluster_total_this_epoch',
            torch.zeros(self.k, device=self.device),
        )
        if isinstance(self._cluster_total_this_epoch, torch.Tensor):
            self._cluster_total_this_epoch = self._cluster_total_this_epoch.float().to(self.device)

        self._fitted = True
