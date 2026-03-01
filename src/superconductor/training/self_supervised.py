"""
Phase 2: Self-Supervised Training for Superconductor VAE.

PURPOSE: Improve the model's generalization ability — close the gap between
training exact match (86.5%) and holdout exact match (22.2%). This is a
MODEL ENHANCEMENT algorithm, NOT a discovery algorithm. Novel superconductor
discovery is a separate effort (holdout search, z-space exploration).

However, during Phase 2 generation, the model may produce formulas that are
NOT in the training data and pass all validators. These are flagged and saved
as opportunistic discoveries (formula + z-vector + predicted Tc) but they
are a side effect, not the goal.

Architecture:
  1. Sample z-vectors from latent space (4 strategies: perturbation, element-anchored, SLERP, PCA walk)
  2. Generate formulas via decoder (greedy + temperature)
  3. Filter candidates (parse, chemical validation, physics validation)
  4. Compute 4 self-supervised loss signals on valid candidates
  5. Update encoder/decoder with safety-gated gradients
  6. Flag and save any novel plausible superconductors encountered

Reference: docs/PHASE2_SELF_SUPERVISED_DESIGN.md

February 2026
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from superconductor.losses.round_trip_loss import (
    RoundTripConsistencyLoss,
    _formula_to_encoder_input,
    _ensure_imports,
)
from superconductor.losses.constraint_zoo import (
    SiteOccupancySumLoss,
    ChargeBalanceLoss,
)
from superconductor.validation.candidate_validator import CandidateValidator
from superconductor.validation.physics_validator import PhysicsValidator
from superconductor.training.coverage_tracker import CoverageTracker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SelfSupervisedConfig:
    """Configuration for Phase 2 self-supervised training."""

    # Master toggle and activation
    enabled: bool = False
    start: str = 'auto'          # Epoch number or 'auto'
    auto_min_exact: float = 0.80  # Min exact match to auto-activate
    interval: int = 2            # Run every N supervised epochs
    min_resume_epochs: int = 0   # Suppress activation for N epochs after training starts
                                 # (e.g., 50 for post-expansion recovery)

    # Weight ramping
    max_weight: float = 0.1      # Max total Phase 2 loss weight
    warmup_epochs: int = 50      # Epochs to ramp from 0 to max_weight

    # Sampling budget
    # n_samples: How many z-vectors to sample per sub-epoch.
    # Set to 0 for auto-scaling based on GPU VRAM (continuous formula).
    # Duplicate formulas from different z-vectors are NOT deduplicated — each z
    # is a distinct latent point and gets its own round-trip loss. Degeneracy
    # (many z → same formula) is tracked as a diagnostic of z-space topology.
    n_samples: int = 0            # 0 = auto-scale from VRAM
    noise_schedule: List[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.08, 0.1]
    )
    noise_warmup_epochs: int = 200  # Epochs to ramp noise from min to max

    # Generation
    greedy_fraction: float = 0.5    # Fraction of samples decoded greedily
    explore_temp_min: float = 0.1   # Min temp for exploratory decoding
    explore_temp_max: float = 0.3   # Max temp for exploratory decoding

    # Learning rate
    lr_factor: float = 0.1      # Phase 2 LR = main_LR * this
    max_grad_norm: float = 0.5  # Phase 2 gradient clip

    # Loss weights (relative within Phase 2)
    round_trip_weight: float = 1.0     # Loss 1: Extended round-trip consistency
    consistency_weight: float = 0.5    # Loss 2: Multi-head self-consistency
    physics_weight: float = 0.3        # Loss 3: Physics constraints
    reinforce_weight: float = 0.5      # Loss 4: REINFORCE round-trip reward

    # Mode collapse detection
    diversity_bonus: float = 5.0       # REINFORCE bonus for unique formulas
    collapse_threshold: float = 0.3    # unique/total < this triggers intervention
    collapse_temp_boost: float = 0.5   # Temp increase on collapse detection
    collapse_rt_weight_mult: float = 2.0  # Round-trip weight multiplier on collapse

    # Coverage tracking
    coverage_k: int = 64               # Number of k-means clusters for z-space partitioning
    coverage_temperature: float = 1.0  # Sampling weight exponent (higher = more uniform)
    coverage_decay: float = 0.995      # Visit count decay per sub-epoch (half-life ~139)

    # Strategy 4: Element-anchored sampling
    element_anchored: bool = True                 # Enable element-anchored sampling
    element_anchored_fraction: float = 0.20       # Fraction of sampling budget
    element_min_shared: int = 2                   # Min shared elements for neighbor status
    element_perturb_sigma: float = 0.05           # Perturbation sigma for centroid blends
    element_interpolate_fraction: float = 0.3     # Fraction using SLERP (rest = centroid blend)

    # Safety guards
    exact_drop_threshold: float = 0.02  # Halve weight if exact drops this much
    exact_drop_window: int = 4         # Epochs to compare for drop detection

    # Filtering thresholds
    candidate_min_score: float = 0.5   # CandidateValidator threshold
    physics_min_score: float = 0.4     # PhysicsValidator threshold

    @classmethod
    def from_train_config(cls, config: dict) -> 'SelfSupervisedConfig':
        """Create from TRAIN_CONFIG dictionary."""
        raw_n = config.get('phase2_n_samples', 'auto')
        if raw_n == 'auto' or raw_n == 0:
            n = 0  # Will be resolved at runtime from VRAM
        else:
            n = int(raw_n)
        return cls(
            enabled=config.get('phase2_enabled', False),
            start=config.get('phase2_start', 'auto'),
            auto_min_exact=config.get('phase2_auto_min_exact', 0.80),
            interval=config.get('phase2_interval', 2),
            min_resume_epochs=config.get('phase2_min_resume_epochs', 0),
            max_weight=config.get('phase2_max_weight', 0.1),
            warmup_epochs=config.get('phase2_warmup', 50),
            n_samples=n,
            noise_schedule=config.get('phase2_noise_schedule', [0.02, 0.05, 0.08, 0.1]),
            noise_warmup_epochs=config.get('phase2_noise_warmup_epochs', 200),
            lr_factor=config.get('phase2_lr_factor', 0.1),
            max_grad_norm=config.get('phase2_max_grad_norm', 0.5),
            diversity_bonus=config.get('phase2_diversity_bonus', 5.0),
            collapse_threshold=config.get('phase2_collapse_threshold', 0.3),
            coverage_k=config.get('phase2_coverage_k', 64),
            coverage_temperature=config.get('phase2_coverage_temperature', 1.0),
            coverage_decay=config.get('phase2_coverage_decay', 0.995),
            element_anchored=config.get('phase2_element_anchored', True),
            element_anchored_fraction=config.get('phase2_element_anchored_fraction', 0.20),
            element_min_shared=config.get('phase2_element_min_shared', 2),
            element_perturb_sigma=config.get('phase2_element_perturb_sigma', 0.05),
            element_interpolate_fraction=config.get('phase2_element_interpolate_fraction', 0.3),
        )

    def resolve_n_samples(self, device: torch.device) -> int:
        """Resolve z-vector count from GPU VRAM if auto (0).

        Uses a continuous formula: n = clamp(3.2 * vram_gb, 32, 512)
        This scales smoothly across any GPU without hardcoded tiers:
          8GB  → 25   (RTX 4060 Laptop)
          16GB → 51
          24GB → 77   (RTX 3090)
          40GB → 128  (A100 40GB)
          80GB → 256  (A100 80GB)
        """
        if self.n_samples > 0:
            return self.n_samples
        if not torch.cuda.is_available():
            return 32
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        return int(max(32, min(512, 3.2 * vram_gb)))


# ---------------------------------------------------------------------------
# Z-Space Sampler (3 strategies)
# ---------------------------------------------------------------------------

def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two z-vectors.

    Preserves magnitude (stays on hypersphere) unlike linear interpolation.
    """
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)

    cos_omega = (z1_norm * z2_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(cos_omega)

    # Handle near-parallel vectors (sin(omega) ≈ 0)
    sin_omega = torch.sin(omega)
    near_parallel = (sin_omega.abs() < 1e-6).squeeze(-1)

    # SLERP formula
    coeff1 = torch.sin((1 - t) * omega) / sin_omega
    coeff2 = torch.sin(t * omega) / sin_omega

    result = coeff1 * z1 + coeff2 * z2

    # Fallback to lerp for near-parallel
    if near_parallel.any():
        lerp_result = (1 - t) * z1 + t * z2
        if result.dim() == 1:
            if near_parallel.item():
                result = lerp_result
        else:
            result[near_parallel] = lerp_result[near_parallel]

    # Preserve average magnitude
    avg_mag = (z1.norm(dim=-1, keepdim=True) + z2.norm(dim=-1, keepdim=True)) / 2
    result = F.normalize(result, dim=-1) * avg_mag

    return result


class ZSpaceSampler:
    """Samples z-vectors from latent space using 3-4 strategies.

    When element data is available and config.element_anchored is True:
        Strategy 1 (40%): Training z neighborhood (perturbation)
        Strategy 2 (20%): Element-anchored sampling (chemically-guided neighborhoods)
        Strategy 3 (25%): SLERP interpolation between same-family pairs
        Strategy 4 (15%): PCA-directed walks

    Otherwise (backward compatible):
        Strategy 1 (60%): Training z neighborhood (perturbation)
        Strategy 2 (25%): SLERP interpolation between same-family pairs
        Strategy 3 (15%): PCA-directed walks
    """

    def __init__(
        self,
        config: SelfSupervisedConfig,
        device: torch.device,
        coverage_tracker: Optional[CoverageTracker] = None,
    ):
        self.config = config
        self.device = device
        self.coverage_tracker = coverage_tracker

        # PCA components (computed lazily on first call)
        self._pca_components = None  # [n_components, 2048]
        self._pca_mean = None        # [2048]
        self._pca_std = None         # [n_components]

        # Family index for SLERP pairs (built from z-cache)
        self._family_indices = {}    # family_id -> list of indices

        # Element index for element-anchored sampling (Strategy 4)
        self._element_sets = None          # list[frozenset] per sample — element atomic numbers
        self._element_to_samples = None    # dict: atomic_number -> set of sample indices
        self._all_elements = None          # sorted list of all unique atomic numbers
        self._element_visit_counts = None  # dict: atomic_number -> float visit count (for cycling)
        self._element_seen_formulas = None # dict[int, set[str]] — unique formulas per element
        self._has_element_data = False

    def update_cache(
        self,
        z_cache: torch.Tensor,      # [N, 2048] cached training z-vectors
        is_sc: torch.Tensor,         # [N] SC labels (1/0)
        family_labels: Optional[torch.Tensor] = None,  # [N] family indices
        element_indices: Optional[torch.Tensor] = None,  # [N, 12] atomic numbers
        element_mask: Optional[torch.Tensor] = None,     # [N, 12] active element slots
    ):
        """Update internal state from latest z-cache."""
        self._z_cache = z_cache.to(self.device)
        self._is_sc = is_sc.to(self.device)
        self._n_cached = z_cache.shape[0]

        # Build family index for SLERP
        self._family_indices = {}
        if family_labels is not None:
            for fam_id in family_labels.unique().tolist():
                mask = (family_labels == fam_id)
                self._family_indices[int(fam_id)] = torch.where(mask)[0]
        else:
            # All in one group if no family labels
            self._family_indices[0] = torch.arange(self._n_cached)

        # Compute PCA (SVD on centered z-vectors)
        self._compute_pca(z_cache)

        # Build element index for Strategy 4 (element-anchored sampling)
        if (element_indices is not None and element_mask is not None
                and self.config.element_anchored):
            self._build_element_index(element_indices, element_mask)
        else:
            self._has_element_data = False

        # Fit coverage tracker on z-cache (~2-3s for 46K x 2048)
        if self.coverage_tracker is not None:
            try:
                self.coverage_tracker.fit(z_cache)
            except Exception as e:
                print(f"  [Coverage] Warning: fit failed: {e}", flush=True)

    def _compute_pca(self, z_cache: torch.Tensor, n_components: int = 20):
        """Compute PCA components from training z-vectors."""
        z_np = z_cache.cpu().float().numpy()
        self._pca_mean = z_np.mean(axis=0)
        centered = z_np - self._pca_mean

        # Truncated SVD for efficiency (only need top-k components)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self._pca_components = torch.from_numpy(
                Vt[:n_components]
            ).float().to(self.device)  # [20, 2048]
            self._pca_std = torch.from_numpy(
                S[:n_components] / np.sqrt(max(len(z_np) - 1, 1))
            ).float().to(self.device)  # [20]
            self._pca_mean_t = torch.from_numpy(
                self._pca_mean
            ).float().to(self.device)  # [2048]
        except np.linalg.LinAlgError:
            # SVD failed — disable PCA sampling
            self._pca_components = None

    def _build_element_index(
        self,
        element_indices: torch.Tensor,  # [N, 12]
        element_mask: torch.Tensor,     # [N, 12]
    ):
        """Build inverted index: atomic_number -> set of sample indices.

        Also builds per-sample element sets and initializes element visit counts
        for cycling through all elements (rare elements get priority).

        O(N × 12) = O(46K × 12) < 1 second.
        """
        t0 = time.time()
        N = element_indices.shape[0]

        elem_idx_np = element_indices.cpu().numpy()
        elem_mask_np = element_mask.cpu().bool().numpy()

        self._element_sets = []
        self._element_to_samples = {}

        for i in range(N):
            active = elem_idx_np[i][elem_mask_np[i]]
            elem_set = frozenset(int(a) for a in active if a > 0)
            self._element_sets.append(elem_set)

            for a in elem_set:
                if a not in self._element_to_samples:
                    self._element_to_samples[a] = set()
                self._element_to_samples[a].add(i)

        self._all_elements = sorted(self._element_to_samples.keys())

        # Initialize element visit counts for cycling (all start at 0)
        # Preserved across update_cache calls only if elements match
        if self._element_visit_counts is None:
            self._element_visit_counts = {a: 0.0 for a in self._all_elements}
        else:
            # Merge: keep existing counts for known elements, init new ones
            new_counts = {}
            for a in self._all_elements:
                new_counts[a] = self._element_visit_counts.get(a, 0.0)
            self._element_visit_counts = new_counts

        # Initialize element seen formulas for novelty tracking
        if self._element_seen_formulas is None:
            self._element_seen_formulas = {a: set() for a in self._all_elements}
        else:
            new_seen = {}
            for a in self._all_elements:
                new_seen[a] = self._element_seen_formulas.get(a, set())
            self._element_seen_formulas = new_seen

        self._has_element_data = True

        # Diagnostic
        n_unique_elems = len(self._all_elements)
        avg_elems_per_sample = np.mean([len(s) for s in self._element_sets]) if N > 0 else 0
        avg_samples_per_elem = (
            np.mean([len(v) for v in self._element_to_samples.values()])
            if self._element_to_samples else 0
        )
        print(f"  [Element Index] Built in {time.time() - t0:.2f}s: "
              f"{n_unique_elems} unique elements, "
              f"avg {avg_elems_per_sample:.1f} elements/sample, "
              f"avg {avg_samples_per_elem:.0f} samples/element", flush=True)

    def _weighted_sample_idx(self, n: int) -> torch.Tensor:
        """Sample n indices from z-cache, weighted by coverage if available.

        Falls back to uniform torch.randint if coverage tracker is not fitted.
        """
        if (self.coverage_tracker is not None
                and self.coverage_tracker._fitted
                and self._n_cached > 0):
            weights = self.coverage_tracker.get_sampling_weights()
            return torch.multinomial(weights, n, replacement=True)
        return torch.randint(0, self._n_cached, (n,), device=self.device)

    def sample(
        self,
        n: int,
        epoch: int,
        phase2_epoch: int,  # Epochs since Phase 2 activation
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Sample n z-vectors using 3 or 4 strategies.

        When element data is available and config.element_anchored:
            40% perturbation, 20% element-anchored, 25% SLERP, 15% PCA
        Otherwise (backward compatible):
            60% perturbation, 25% SLERP, 15% PCA

        Args:
            n: Number of z-vectors to sample this wave
            epoch: Current training epoch (for noise schedule)
            phase2_epoch: Epochs since Phase 2 activation (for noise ramp)

        Returns:
            z_sampled: [n, 2048]
            stats: Dict with counts per strategy
        """
        use_element = (self._has_element_data and self.config.element_anchored)

        if use_element:
            # 4-strategy split: 40% perturbation, 20% element-anchored, 25% SLERP, 15% PCA
            elem_frac = self.config.element_anchored_fraction
            n_element = int(n * elem_frac)
            n_perturb = int(n * (0.60 - elem_frac))  # Reduce perturbation to make room
            n_slerp = int(n * 0.25)
            n_pca = n - n_perturb - n_element - n_slerp
        else:
            # Original 3-strategy split: 60% perturbation, 25% SLERP, 15% PCA
            n_element = 0
            n_perturb = int(n * 0.60)
            n_slerp = int(n * 0.25)
            n_pca = n - n_perturb - n_slerp

        parts = []
        stats = {'n_perturb': 0, 'n_slerp': 0, 'n_pca': 0, 'n_element_anchored': 0}

        # Strategy 1: Training Z neighborhood perturbation
        z_perturbed = self._sample_perturbation(n_perturb, phase2_epoch)
        parts.append(z_perturbed)
        stats['n_perturb'] = z_perturbed.shape[0]

        # Strategy 2: Element-anchored sampling (only when element data available)
        if n_element > 0:
            z_elem = self._sample_element_anchored(n_element, phase2_epoch)
            parts.append(z_elem)
            stats['n_element_anchored'] = z_elem.shape[0]

        # Strategy 3: SLERP interpolation
        z_slerp = self._sample_slerp(n_slerp)
        parts.append(z_slerp)
        stats['n_slerp'] = z_slerp.shape[0]

        # Strategy 4: PCA-directed walk
        z_pca = self._sample_pca_walk(n_pca)
        parts.append(z_pca)
        stats['n_pca'] = z_pca.shape[0]

        z_sampled = torch.cat(parts, dim=0)  # [n_samples, 2048]

        # Record visits for coverage tracking
        if self.coverage_tracker is not None and self.coverage_tracker._fitted:
            self.coverage_tracker.record_visits(z_sampled)

        return z_sampled, stats

    def _sample_perturbation(
        self, n: int, phase2_epoch: int
    ) -> torch.Tensor:
        """Strategy 1: Perturb training z-vectors with scheduled noise."""
        # Noise schedule: ramp from min to max over warmup epochs
        schedule = self.config.noise_schedule
        progress = min(1.0, phase2_epoch / max(1, self.config.noise_warmup_epochs))
        sigma = schedule[0] + progress * (schedule[-1] - schedule[0])

        # Random training z-vectors (coverage-weighted if tracker available)
        idx = self._weighted_sample_idx(n)
        z_base = self._z_cache[idx]  # [n, 2048]

        # Additive Gaussian noise
        epsilon = torch.randn_like(z_base)
        z_perturbed = z_base + sigma * epsilon

        return z_perturbed

    def _sample_element_anchored(
        self, n: int, phase2_epoch: int
    ) -> torch.Tensor:
        """Strategy 2: Element-anchored sampling.

        Explores z-neighborhoods of chemically similar materials by:
        1. Selecting anchor samples biased toward under-explored ELEMENTS (cycling)
        2. Finding neighbors that share >= min_shared elements with the anchor
        3. Either SLERPing between anchor and neighbor (30%) or blending toward
           the neighborhood centroid with noise (70%)

        This keeps perturbations in chemically meaningful regions — the same insight
        that makes holdout_search_targeted so effective.

        Element cycling ensures every element gets explored over time, even rare ones.
        """
        if not self._has_element_data or n <= 0:
            return self._sample_perturbation(n, phase2_epoch)

        sigma = self.config.element_perturb_sigma
        min_shared = self.config.element_min_shared
        interp_frac = self.config.element_interpolate_fraction
        n_interp_budget = int(n * interp_frac)  # SLERP budget (counter-based, not index-based)

        results = []
        n_slerp_used = 0  # Track actual SLERP usage to preserve budget

        # --- Pick anchors biased toward under-explored elements ---
        # Weight elements by inverse visit count × novelty factor (cycling):
        # discovery_rate = unique formulas per visit. Higher = more productive.
        # Never-explored elements get 1.0 (max priority).
        # Saturated elements (many visits, few unique) get low rate.
        elem_weights = {}
        for a in self._all_elements:
            visits = self._element_visit_counts.get(a, 0.0)
            n_unique = len(self._element_seen_formulas.get(a, set())) if self._element_seen_formulas else 0
            discovery_rate = n_unique / max(1.0, visits) if visits > 0 else 1.0
            novelty_factor = max(0.1, min(1.0, discovery_rate))
            visit_weight = 1.0 / (1.0 + visits)
            elem_weights[a] = visit_weight * novelty_factor

        # Select n target elements with probability proportional to weight
        weight_arr = np.array([elem_weights[a] for a in self._all_elements])
        weight_arr /= weight_arr.sum()
        target_elements = np.random.choice(
            self._all_elements, size=n, replace=True, p=weight_arr
        )

        for i in range(n):
            target_elem = int(target_elements[i])

            # Pick a random anchor sample containing this element
            anchor_candidates = list(self._element_to_samples.get(target_elem, set()))
            if not anchor_candidates:
                # Fallback: random sample
                anchor_idx = torch.randint(0, self._n_cached, (1,)).item()
            else:
                anchor_idx = anchor_candidates[np.random.randint(len(anchor_candidates))]

            anchor_z = self._z_cache[anchor_idx]  # [2048]
            anchor_elems = self._element_sets[anchor_idx]

            # Find neighbors sharing >= min_shared elements
            neighbor_indices = self._find_element_neighbors(
                anchor_elems, anchor_idx, min_shared
            )

            if not neighbor_indices:
                # Fallback: plain perturbation from anchor
                epsilon = torch.randn_like(anchor_z)
                results.append(anchor_z + sigma * epsilon)
            elif n_slerp_used < n_interp_budget:
                # SLERP between anchor and a random neighbor (budget-tracked)
                nb_idx = neighbor_indices[np.random.randint(len(neighbor_indices))]
                nb_z = self._z_cache[nb_idx]
                t = np.random.uniform(0.1, 0.9)
                results.append(slerp(anchor_z, nb_z, t))
                n_slerp_used += 1
            else:
                # Centroid blend: move toward neighborhood centroid + noise
                nb_sample = neighbor_indices
                if len(nb_sample) > 16:
                    nb_sample = [neighbor_indices[j] for j in
                                 np.random.choice(len(neighbor_indices), 16, replace=False)]
                nb_zs = self._z_cache[nb_sample]  # [k, 2048]
                centroid = nb_zs.mean(dim=0)  # [2048]

                # Blend: 70% anchor, 30% centroid + noise
                blend = 0.7 * anchor_z + 0.3 * centroid
                epsilon = torch.randn_like(blend)
                results.append(blend + sigma * epsilon)

            # Record element visits for cycling
            for a in anchor_elems:
                if a in self._element_visit_counts:
                    self._element_visit_counts[a] += 1.0

        # Decay all element visit counts (similar to coverage decay)
        decay = self.config.coverage_decay if hasattr(self.config, 'coverage_decay') else 0.995
        for a in self._element_visit_counts:
            self._element_visit_counts[a] *= decay

        return torch.stack(results)  # [n, 2048]

    def _find_element_neighbors(
        self,
        anchor_elems: frozenset,
        anchor_idx: int,
        min_shared: int,
        max_per_element: int = 2000,
    ) -> List[int]:
        """Find samples sharing >= min_shared elements with anchor.

        Uses inverted index for fast lookup. Falls back to min_shared=1 if
        no candidates found at the requested threshold.

        For common elements (e.g., O in ~90% of 46K samples), we cap iteration
        to max_per_element random entries to bound worst-case to O(n_elems * 2000).
        """
        # Count overlaps per candidate using inverted index
        overlap_counts = {}
        for a in anchor_elems:
            members = self._element_to_samples.get(a, set())
            # Cap iteration for common elements (O, Cu, etc.) to avoid
            # iterating 40K+ entries in pure Python
            if len(members) > max_per_element:
                members = set(np.random.choice(list(members), max_per_element, replace=False))
            for sample_idx in members:
                if sample_idx == anchor_idx:
                    continue
                overlap_counts[sample_idx] = overlap_counts.get(sample_idx, 0) + 1

        # Filter by min_shared threshold
        neighbors = [idx for idx, cnt in overlap_counts.items() if cnt >= min_shared]

        # Fallback chain: try min_shared=1 if nothing found at threshold
        if not neighbors and min_shared > 1:
            neighbors = [idx for idx, cnt in overlap_counts.items() if cnt >= 1]

        return neighbors

    def record_element_novelty(
        self,
        valid_formulas: List[str],
        valid_parsed: List[Dict],
    ):
        """Record per-element novelty from valid generated formulas.

        For each valid formula, extracts its elements from parsed['element_indices']
        + parsed['element_mask'] and adds the formula to _element_seen_formulas[atomic_num].
        Strategy-agnostic: any valid formula with Cu counts as Cu exploration.

        Args:
            valid_formulas: List of valid formula strings
            valid_parsed: List of parsed encoder input dicts (with element_indices, element_mask)
        """
        if self._element_seen_formulas is None or not self._has_element_data:
            return

        for i, formula in enumerate(valid_formulas):
            if i >= len(valid_parsed):
                break
            parsed = valid_parsed[i]
            elem_idx = parsed.get('element_indices')
            elem_mask = parsed.get('element_mask')
            if elem_idx is None or elem_mask is None:
                continue

            # Extract active element atomic numbers
            idx_np = elem_idx.cpu().numpy() if hasattr(elem_idx, 'cpu') else elem_idx
            mask_np = elem_mask.cpu().bool().numpy() if hasattr(elem_mask, 'cpu') else elem_mask
            for j in range(len(idx_np)):
                if mask_np[j]:
                    a = int(idx_np[j])
                    if a > 0 and a in self._element_seen_formulas:
                        self._element_seen_formulas[a].add(formula)

    def _sample_slerp(self, n: int) -> torch.Tensor:
        """Strategy 3: SLERP between same-family pairs."""
        results = []
        t_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Distribute across families
        families = list(self._family_indices.keys())
        per_family = max(1, n // max(1, len(families)))
        remaining = n

        for fam_id in families:
            if remaining <= 0:
                break
            fam_idx = self._family_indices[fam_id]
            if len(fam_idx) < 2:
                continue

            n_pairs = min(per_family, remaining)
            # Random pairs within family
            perm = torch.randperm(len(fam_idx), device=self.device)
            for p in range(0, min(n_pairs, len(fam_idx) - 1)):
                i1, i2 = fam_idx[perm[p]], fam_idx[perm[(p + 1) % len(fam_idx)]]
                z1, z2 = self._z_cache[i1], self._z_cache[i2]
                t = t_values[p % len(t_values)]
                z_interp = slerp(z1, z2, t)
                results.append(z_interp)
                remaining -= 1
                if remaining <= 0:
                    break

        if len(results) == 0:
            # Fallback: perturb random pairs
            return self._sample_perturbation(n, 0)

        # Pad with more SLERP if we didn't fill quota
        while len(results) < n:
            pair_idx = self._weighted_sample_idx(2)
            idx1, idx2 = pair_idx[0], pair_idx[1]
            t = np.random.choice(t_values)
            z_interp = slerp(self._z_cache[idx1], self._z_cache[idx2], t)
            results.append(z_interp)

        return torch.stack(results[:n])  # [n, 2048]

    def _sample_pca_walk(self, n: int) -> torch.Tensor:
        """Strategy 4: Walk along top PCA components."""
        if self._pca_components is None:
            return self._sample_perturbation(n, 0)

        results = []
        sigma_steps = [0.3, 0.5, 1.0, 1.5, 2.0]
        n_components = self._pca_components.shape[0]

        # Walk along each PC at +/- sigma steps (coverage-weighted base selection)
        base_idx = self._weighted_sample_idx(n)
        z_bases = self._z_cache[base_idx]  # [n, 2048]

        for i in range(n):
            pc_idx = i % n_components
            step_idx = (i // n_components) % len(sigma_steps)
            sign = 1.0 if (i // (n_components * len(sigma_steps))) % 2 == 0 else -1.0

            sigma_step = sigma_steps[step_idx]
            direction = self._pca_components[pc_idx]  # [2048]
            scale = sign * sigma_step * self._pca_std[pc_idx]
            z_walked = z_bases[i] + scale * direction
            results.append(z_walked)

        return torch.stack(results)  # [n, 2048]


# ---------------------------------------------------------------------------
# Candidate Filter
# ---------------------------------------------------------------------------

class CandidateFilter:
    """4-stage filtering pipeline for generated formulas.

    Stage 1: Parse check (_formula_to_encoder_input)
    Stage 2: CandidateValidator (overall_score >= threshold)
    Stage 3: PhysicsValidator (plausibility_score >= threshold)
    Stage 4: Constraint rewards A1, A4, A7 (must pass all three)
    """

    def __init__(
        self,
        config: SelfSupervisedConfig,
        vocab_config=None,
    ):
        self.config = config
        self.candidate_validator = CandidateValidator(
            min_score_threshold=config.candidate_min_score,
        )
        self.physics_validator = PhysicsValidator()
        self._vocab_config = vocab_config

    def filter(
        self,
        formulas: List[str],
        tokens: Optional[torch.Tensor] = None,  # [batch, seq_len] for A1/A4/A7
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[List[int], List[str], List[Dict], Dict[str, int]]:
        """Filter generated formulas through 4-stage pipeline.

        Returns:
            valid_indices: indices into original formulas list
            valid_formulas: formulas that passed all stages
            parsed_inputs: encoder input dicts for valid formulas
            stats: dict with pass counts per stage
        """
        _ensure_imports()
        stats = {
            'total': len(formulas),
            'pass_parse': 0,
            'pass_candidate': 0,
            'pass_physics': 0,
            'pass_constraints': 0,
        }

        valid_indices = []
        valid_formulas = []
        parsed_inputs = []

        for i, formula in enumerate(formulas):
            if not formula or len(formula.strip()) == 0:
                continue

            # Stage 1: Parse check
            parsed = _formula_to_encoder_input(formula, device=device)
            if parsed is None:
                continue
            stats['pass_parse'] += 1

            # Stage 2: CandidateValidator
            try:
                val_result = self.candidate_validator.validate(formula)
                if val_result.overall_score < self.config.candidate_min_score:
                    continue
            except Exception:
                continue
            stats['pass_candidate'] += 1

            # Stage 3: PhysicsValidator
            try:
                # Build stoichiometry dict from parsed elements
                from superconductor.losses.round_trip_loss import _parse_fraction_formula
                _ensure_imports()
                stoich = _parse_fraction_formula(formula)
                if stoich is None:
                    stoich = {}
                phys_result = self.physics_validator.validate(formula, stoich)
                if phys_result.plausibility_score < self.config.physics_min_score:
                    continue
            except Exception:
                continue
            stats['pass_physics'] += 1

            # Stage 4: Constraint rewards A1, A4, A7 (using token-level checks)
            # These are non-differentiable rewards, so we use them as binary gates
            if tokens is not None and i < tokens.shape[0]:
                tok = tokens[i:i+1]  # [1, seq_len]
                mask = (tok != 0).float()  # Non-padding mask
                try:
                    from superconductor.losses.constraint_rewards import (
                        compute_duplicate_element_penalty,
                        compute_stoich_normalization_penalty,
                        compute_impossible_element_penalty,
                    )
                    a1 = compute_duplicate_element_penalty(tok, mask, penalty=-1.0)
                    a4 = compute_stoich_normalization_penalty(tok, mask, penalty=-1.0)
                    a7 = compute_impossible_element_penalty(tok, mask, penalty=-1.0)
                    # All must pass (penalty = 0 means pass)
                    if (a1.item() < 0) or (a4.item() < 0) or (a7.item() < 0):
                        continue
                except Exception:
                    pass  # Skip constraint check if imports fail
            stats['pass_constraints'] += 1

            valid_indices.append(i)
            valid_formulas.append(formula)
            parsed_inputs.append(parsed)

        return valid_indices, valid_formulas, parsed_inputs, stats


# ---------------------------------------------------------------------------
# Novel Discovery Tracker
# ---------------------------------------------------------------------------

class NovelDiscoveryTracker:
    """Flags and saves generated formulas that are NOT in the training data.

    Phase 2 is a model enhancement algorithm, not a discovery algorithm.
    But when a novel plausible superconductor appears during generation,
    we save it opportunistically (formula + z-vector + predicted Tc).

    Distinguishes three categories:
      - Known: formula exists in training data (ignored)
      - Holdout recovery: formula matches one of the 45 held-out targets (logged)
      - Novel: formula not in training or holdout data (logged + saved)
    """

    def __init__(
        self,
        known_formulas: Set[str],
        holdout_formulas: Set[str],
        output_path: str = 'outputs/phase2_discoveries.jsonl',
    ):
        self._known = known_formulas
        self._holdout = holdout_formulas
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # Avoid logging the same formula twice across sub-epochs
        self._seen_novel: Set[str] = set()
        self._seen_holdout: Set[str] = set()

        # Counters
        self.n_novel = 0
        self.n_holdout_recovered = 0

    def check_batch(
        self,
        formulas: List[str],
        z_vectors: torch.Tensor,      # [N, latent_dim]
        tc_preds: Optional[torch.Tensor],  # [N] predicted Tc (normalized)
        epoch: int,
        tc_mean: float = 2.725,       # log1p z-score stats for denormalization
        tc_std: float = 1.353,
    ) -> Dict[str, int]:
        """Check a batch of valid formulas for novel discoveries.

        Args:
            formulas: Valid formula strings (already passed all filters)
            z_vectors: Corresponding z-vectors
            tc_preds: Encoder's Tc predictions (normalized log1p z-score)
            epoch: Current training epoch
            tc_mean, tc_std: For denormalizing Tc predictions

        Returns:
            Dict with counts: n_novel, n_holdout_recovered
        """
        batch_novel = 0
        batch_holdout = 0

        for i, formula in enumerate(formulas):
            if formula in self._known:
                continue

            z = z_vectors[i].detach().cpu()
            tc_norm = tc_preds[i].item() if tc_preds is not None else None
            # Denormalize: Tc = exp(tc_norm * tc_std + tc_mean) - 1
            tc_kelvin = None
            if tc_norm is not None:
                import math as _math
                tc_kelvin = _math.expm1(tc_norm * tc_std + tc_mean)

            if formula in self._holdout:
                if formula not in self._seen_holdout:
                    self._seen_holdout.add(formula)
                    self.n_holdout_recovered += 1
                    batch_holdout += 1
                    print(f"  [Phase2 HOLDOUT RECOVERY] {formula} "
                          f"(Tc_pred={tc_kelvin:.1f}K)" if tc_kelvin else
                          f"  [Phase2 HOLDOUT RECOVERY] {formula}",
                          flush=True)
                    self._save_record(
                        formula, z, tc_kelvin, epoch, category='holdout_recovery',
                    )
            else:
                # Novel formula — not in training or holdout
                if formula not in self._seen_novel:
                    self._seen_novel.add(formula)
                    self.n_novel += 1
                    batch_novel += 1
                    tc_str = f"Tc_pred={tc_kelvin:.1f}K" if tc_kelvin else "Tc=?"
                    print(f"  [Phase2 NOVEL] {formula} ({tc_str})", flush=True)
                    self._save_record(
                        formula, z, tc_kelvin, epoch, category='novel',
                    )

        return {'n_novel': batch_novel, 'n_holdout_recovered': batch_holdout}

    def _save_record(
        self,
        formula: str,
        z_vector: torch.Tensor,
        tc_kelvin: Optional[float],
        epoch: int,
        category: str,
    ):
        """Append a discovery record to the JSONL file."""
        record = {
            'formula': formula,
            'tc_kelvin': round(tc_kelvin, 2) if tc_kelvin is not None else None,
            'epoch': epoch,
            'category': category,
            'z_vector_norm': round(z_vector.norm().item(), 4),
            # Store z-vector as list for reproducibility (can re-decode later)
            'z_vector': z_vector.tolist(),
        }
        with open(self._output_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


# ---------------------------------------------------------------------------
# Phase 2 Loss Computer
# ---------------------------------------------------------------------------

class Phase2LossComputer:
    """Computes all 4 self-supervised loss signals.

    Loss 1: Extended round-trip consistency (encoder gradients)
    Loss 2: Multi-head self-consistency (encoder head gradients)
    Loss 3: Physics constraints on generated formulas (encoder gradients)
    Loss 4: REINFORCE round-trip reward (decoder gradients)
    """

    def __init__(
        self,
        config: SelfSupervisedConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device

        # Loss 3 modules
        self.site_occupancy_loss = SiteOccupancySumLoss(weight=1.0)
        self.charge_balance_loss = ChargeBalanceLoss(weight=1.0)

    def compute(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        z_sampled: torch.Tensor,           # [N_valid, 2048] original sampled z
        parsed_inputs: List[Dict],         # Encoder inputs for valid formulas
        valid_formulas: List[str],         # String formulas for logging
        log_probs: Optional[torch.Tensor] = None,  # [N_valid, seq_len] for REINFORCE
        token_mask: Optional[torch.Tensor] = None,  # [N_valid, seq_len]
        all_unique_formulas: Optional[set] = None,   # For diversity bonus
    ) -> Dict[str, torch.Tensor]:
        """Compute all 4 Phase 2 loss signals.

        Returns dict with:
            'phase2_total': weighted sum of all losses
            'loss1_round_trip': extended round-trip consistency loss
            'loss2_consistency': multi-head self-consistency loss
            'loss3_physics': physics constraint loss
            'loss4_reinforce': REINFORCE reward-based loss
            'n_valid': number of valid candidates processed
            'z_mse': round-trip Z reconstruction error
            'tc_mse': round-trip Tc reconstruction error
        """
        n_valid = len(parsed_inputs)
        zero = torch.tensor(0.0, device=self.device, requires_grad=True)

        if n_valid == 0:
            return {
                'phase2_total': zero,
                'loss1_round_trip': zero.detach(),
                'loss2_consistency': zero.detach(),
                'loss3_physics': zero.detach(),
                'loss4_reinforce': zero.detach(),
                'n_valid': 0,
                'z_mse': torch.tensor(0.0, device=self.device),
                'tc_mse': torch.tensor(0.0, device=self.device),
            }

        # Build batched encoder inputs
        elem_indices = torch.stack([p['element_indices'] for p in parsed_inputs]).to(self.device)
        elem_fractions = torch.stack([p['element_fractions'] for p in parsed_inputs]).to(self.device)
        elem_mask = torch.stack([p['element_mask'] for p in parsed_inputs]).to(self.device)

        # === Loss 1: Extended Round-Trip Consistency ===
        loss1, z_mse, tc_mse = self._compute_round_trip_loss(
            encoder, z_sampled, elem_indices, elem_fractions, elem_mask,
        )

        # === Loss 2: Multi-Head Self-Consistency ===
        loss2 = self._compute_consistency_loss(
            encoder, z_sampled, elem_indices, elem_fractions, elem_mask,
        )

        # === Loss 3: Physics Constraints ===
        loss3 = self._compute_physics_loss(
            elem_indices, elem_fractions, elem_mask,
        )

        # === Loss 4: REINFORCE Round-Trip Reward ===
        loss4 = self._compute_reinforce_loss(
            encoder, z_sampled, elem_indices, elem_fractions, elem_mask,
            log_probs, token_mask, valid_formulas, all_unique_formulas,
        )

        # Weighted combination
        total = (
            self.config.round_trip_weight * loss1
            + self.config.consistency_weight * loss2
            + self.config.physics_weight * loss3
            + self.config.reinforce_weight * loss4
        )

        return {
            'phase2_total': total,
            'loss1_round_trip': loss1.detach(),
            'loss2_consistency': loss2.detach(),
            'loss3_physics': loss3.detach(),
            'loss4_reinforce': loss4.detach(),
            'n_valid': n_valid,
            'z_mse': z_mse.detach(),
            'tc_mse': tc_mse.detach(),
        }

    def _compute_round_trip_loss(
        self,
        encoder: nn.Module,
        z_original: torch.Tensor,      # [N, 2048]
        elem_indices: torch.Tensor,    # [N, 12]
        elem_fractions: torch.Tensor,  # [N, 12]
        elem_mask: torch.Tensor,       # [N, 12]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Loss 1: Extended round-trip consistency.

        Re-encode parsed formulas and compare z-space + Tc.
        Gradients flow through encoder. Decoder was under no_grad during generation.
        """
        # Decode z to get Tc and Magpie predictions
        with torch.no_grad():
            decode_result = encoder.decode(z_original)
            magpie_proxy = decode_result['magpie_pred']  # [N, 145]
            tc_proxy = decode_result['tc_pred']          # [N]

        # Re-encode WITH gradients (key: magpie_proxy has no grad, but encoder
        # parameters do, so gradients flow through encoder weights)
        encode_result = encoder.encode(
            element_indices=elem_indices,
            element_fractions=elem_fractions,
            element_mask=elem_mask,
            magpie_features=magpie_proxy,
            tc=tc_proxy.detach(),
        )
        z_recon = encode_result['z']  # [N, 2048]

        # Detach original z for comparison target
        z_target = z_original.detach()
        z_mse = F.mse_loss(z_recon, z_target)

        # Also check Tc consistency
        decode_recon = encoder.decode(z_recon)
        tc_recon = decode_recon['tc_pred']
        tc_target = tc_proxy.detach()
        tc_mse = F.mse_loss(tc_recon, tc_target)

        loss = z_mse + 5.0 * tc_mse  # Tc weighted 5x (same as A5)

        return loss, z_mse.detach(), tc_mse.detach()

    def _compute_consistency_loss(
        self,
        encoder: nn.Module,
        z_sampled: torch.Tensor,       # [N, 2048]
        elem_indices: torch.Tensor,    # [N, 12]
        elem_fractions: torch.Tensor,  # [N, 12]
        elem_mask: torch.Tensor,       # [N, 12]
    ) -> torch.Tensor:
        """Loss 2: Multi-head self-consistency.

        Ensure encoder heads agree: SC classifier, Tc, Tc bucket, family.
        """
        # Get all head predictions from z
        decode_result = encoder.decode(z_sampled)
        tc_pred = decode_result['tc_pred']              # [N]
        tc_class_logits = decode_result.get('tc_class_logits')  # [N, 5] or None

        # Get SC prediction from encoder's sc_head if it exists
        # sc_head takes concatenated [z, tc_pred, magpie_pred, hp_pred, fraction_pred,
        #   element_count_pred, competence, tc_class_logits] = 2048+1+magpie+1+12+1+1+5
        sc_logit = None
        if hasattr(encoder, 'sc_head') and encoder.sc_head is not None:
            hp_pred = encoder.hp_head(z_sampled).squeeze(-1) if hasattr(encoder, 'hp_head') else torch.zeros(z_sampled.shape[0], device=z_sampled.device)
            fraction_output = encoder.fraction_head(z_sampled)
            fraction_pred = fraction_output[:, :encoder.max_elements]
            element_count_pred = fraction_output[:, -1]
            competence = encoder.competence_head(z_sampled).squeeze(-1) if hasattr(encoder, 'competence_head') else torch.zeros(z_sampled.shape[0], device=z_sampled.device)
            sc_input = torch.cat([
                z_sampled,                                    # 2048
                tc_pred.unsqueeze(-1),                        # 1
                decode_result['magpie_pred'],                 # magpie_dim
                hp_pred.unsqueeze(-1),                        # 1
                fraction_pred,                                # 12
                element_count_pred.unsqueeze(-1),             # 1
                competence.unsqueeze(-1),                     # 1
                tc_class_logits if tc_class_logits is not None else torch.zeros(z_sampled.shape[0], 5, device=z_sampled.device),  # 5
            ], dim=-1)
            sc_logit = encoder.sc_head(sc_input).squeeze(-1)  # [N]

        losses = []

        # Rule 1: SC prob <-> Tc agreement
        # If Tc > 0 (unnormalized), model should predict SC
        if sc_logit is not None:
            # tc_pred is normalized (log1p + z-score). Tc > 0K roughly means
            # normalized > some threshold, but we use a soft target:
            # positive tc_pred => likely SC
            should_be_sc = torch.sigmoid(tc_pred * 2.0)  # Soft target
            sc_loss = F.binary_cross_entropy_with_logits(
                sc_logit, should_be_sc.detach()
            )
            losses.append(sc_loss)

        # Rule 2: Tc value <-> Tc bucket agreement
        if tc_class_logits is not None:
            # Compute which bucket tc_pred falls in
            # Buckets: [0, 0-10, 10-50, 50-100, 100+] in Kelvin
            # tc_pred is normalized, so we use the bucket logits directly
            # and enforce consistency via soft cross-entropy
            tc_bucket_probs = F.softmax(tc_class_logits, dim=-1)  # [N, 5]
            # KL divergence: bucket prediction should be peaked
            # (low entropy = high confidence = consistent)
            bucket_entropy = -(tc_bucket_probs * (tc_bucket_probs + 1e-8).log()).sum(dim=-1)
            losses.append(bucket_entropy.mean() * 0.1)  # Encourage peaked predictions

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return sum(losses) / len(losses)

    def _compute_physics_loss(
        self,
        elem_indices: torch.Tensor,    # [N, 12]
        elem_fractions: torch.Tensor,  # [N, 12]
        elem_mask: torch.Tensor,       # [N, 12]
    ) -> torch.Tensor:
        """Loss 3: Differentiable physics constraints (A3 + A6)."""
        losses = []

        # A3: Site occupancy sum
        try:
            a3_result = self.site_occupancy_loss(
                elem_indices, elem_fractions, elem_mask
            )
            if a3_result['site_occupancy_loss'].requires_grad:
                losses.append(a3_result['site_occupancy_loss'])
        except Exception:
            pass

        # A6: Charge balance
        try:
            a6_result = self.charge_balance_loss(
                elem_indices, elem_fractions, elem_mask
            )
            if a6_result['charge_balance_loss'].requires_grad:
                losses.append(a6_result['charge_balance_loss'])
        except Exception:
            pass

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return sum(losses) / len(losses)

    def _compute_reinforce_loss(
        self,
        encoder: nn.Module,
        z_original: torch.Tensor,       # [N, 2048]
        elem_indices: torch.Tensor,     # [N, 12]
        elem_fractions: torch.Tensor,   # [N, 12]
        elem_mask: torch.Tensor,        # [N, 12]
        log_probs: Optional[torch.Tensor] = None,  # [N, seq_len]
        token_mask: Optional[torch.Tensor] = None,  # [N, seq_len]
        valid_formulas: Optional[List[str]] = None,
        all_unique_formulas: Optional[set] = None,
    ) -> torch.Tensor:
        """Loss 4: REINFORCE with round-trip reward.

        Reward = cosine_sim(z_sampled, z_reconstructed) * physics_score + diversity_bonus.
        This is the only signal that reaches the decoder.
        """
        if log_probs is None or token_mask is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        n_valid = z_original.shape[0]
        if n_valid == 0 or log_probs.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Compute round-trip z for reward (no_grad — reward is non-differentiable)
        with torch.no_grad():
            decode_result = encoder.decode(z_original)
            magpie_proxy = decode_result['magpie_pred']
            tc_proxy = decode_result['tc_pred']

            encode_result = encoder.encode(
                element_indices=elem_indices,
                element_fractions=elem_fractions,
                element_mask=elem_mask,
                magpie_features=magpie_proxy,
                tc=tc_proxy,
            )
            z_recon = encode_result['z']

            # Cosine similarity reward
            cos_sim = F.cosine_similarity(z_original, z_recon, dim=-1)  # [N]
            reward = cos_sim.clamp(0, 1)  # [N]

            # Diversity bonus: extra reward for unique formulas
            if valid_formulas is not None and all_unique_formulas is not None:
                for i, f in enumerate(valid_formulas):
                    if f not in all_unique_formulas:
                        reward[i] += self.config.diversity_bonus
                        all_unique_formulas.add(f)

            # Baseline subtraction (mean reward)
            baseline = reward.mean()
            advantage = reward - baseline  # [N]

        # REINFORCE loss: -advantage * sum(log_probs)
        # log_probs: [N, seq_len], token_mask: [N, seq_len]
        # Truncate/pad to match
        min_len = min(log_probs.shape[1], token_mask.shape[1])
        lp = log_probs[:n_valid, :min_len]
        tm = token_mask[:n_valid, :min_len]

        per_sample_log_prob = (lp * tm).sum(dim=-1)  # [N]
        reinforce_loss = -(advantage * per_sample_log_prob).mean()

        return reinforce_loss


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

class SelfSupervisedEpoch:
    """Phase 2 orchestrator: runs one self-supervised sub-epoch.

    Call flow:
        epoch_runner = SelfSupervisedEpoch(config, encoder, decoder, device)
        epoch_runner.load_z_cache(z_cache_path)
        metrics = epoch_runner.run(epoch, phase2_epoch, main_lr)
    """

    def __init__(
        self,
        config: SelfSupervisedConfig,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        v13_tokenizer=None,
        max_formula_len: int = 60,
        known_formulas: Optional[Set[str]] = None,
        holdout_formulas: Optional[Set[str]] = None,
        discovery_output_path: str = 'outputs/phase2_discoveries.jsonl',
    ):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.v13_tokenizer = v13_tokenizer
        self.max_formula_len = max_formula_len

        # Resolve sample count from GPU VRAM if auto
        self._n_samples = config.resolve_n_samples(device)
        vram_str = ""
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            vram_str = f" ({vram_gb:.1f}GB VRAM)"
        if config.n_samples == 0:
            print(f"  [Phase 2] Auto n_samples{vram_str}: {self._n_samples}")
        else:
            print(f"  [Phase 2] Explicit n_samples: {self._n_samples}")

        # Sub-components
        self.coverage_tracker = CoverageTracker(
            k=config.coverage_k,
            temperature=config.coverage_temperature,
            decay=config.coverage_decay,
            device=device,
        )
        self.sampler = ZSpaceSampler(config, device, coverage_tracker=self.coverage_tracker)
        self.candidate_filter = CandidateFilter(config)
        self.loss_computer = Phase2LossComputer(config, device)

        # Novel discovery tracker (opportunistic — Phase 2 is for model
        # enhancement, not discovery, but we save interesting finds)
        self.discovery_tracker = None
        if known_formulas is not None:
            holdout = holdout_formulas or set()
            self.discovery_tracker = NovelDiscoveryTracker(
                known_formulas=known_formulas,
                holdout_formulas=holdout,
                output_path=discovery_output_path,
            )
            print(f"  [Phase 2] Discovery tracker: {len(known_formulas)} known, "
                  f"{len(holdout)} holdout")

        # State tracking
        self._phase2_activation_epoch = None
        self._activation_exact = None
        self._current_weight = 0.0
        self._collapse_active = False
        self._collapse_epochs_remaining = 0
        self._exact_history = []

        # Track unique formulas across sub-epochs for diversity
        self._all_unique_formulas = set()

    def should_activate(self, epoch: int, current_exact: float,
                        start_epoch: int = 0) -> bool:
        """Check if Phase 2 should activate this epoch.

        Args:
            epoch: Current training epoch
            current_exact: Current exact match rate
            start_epoch: Epoch training resumed from (for min_resume_epochs guard)
        """
        if not self.config.enabled:
            return False

        # Already activated?
        if self._phase2_activation_epoch is not None:
            return True

        # Suppress for N epochs after training starts/resumes (post-expansion recovery)
        if self.config.min_resume_epochs > 0:
            epochs_since_resume = epoch - start_epoch
            if epochs_since_resume < self.config.min_resume_epochs:
                return False

        # Check activation condition
        if self.config.start == 'auto':
            if current_exact >= self.config.auto_min_exact:
                self._phase2_activation_epoch = epoch
                self._activation_exact = current_exact
                return True
            return False
        else:
            # Explicit epoch
            start_ep = int(self.config.start)
            if epoch >= start_ep:
                self._phase2_activation_epoch = epoch
                self._activation_exact = current_exact
                return True
            return False

    def should_run_this_epoch(self, epoch: int) -> bool:
        """Check if Phase 2 sub-epoch should run this specific epoch."""
        if self._phase2_activation_epoch is None:
            return False
        return (epoch - self._phase2_activation_epoch) % self.config.interval == 0

    def get_current_weight(self, phase2_epoch: int) -> float:
        """Get current Phase 2 weight with warmup ramp."""
        if self.config.warmup_epochs <= 0:
            return self.config.max_weight
        progress = min(1.0, phase2_epoch / self.config.warmup_epochs)
        return self.config.max_weight * progress

    def load_z_cache(self, cache_path: str):
        """Load z-cache from disk to initialize sampler.

        Reads element_indices and element_mask if present (for element-anchored
        sampling). Backward compatible: old caches without element data use
        cache.get(..., None) and fall back to 3-strategy sampling.
        """
        # weights_only=False: z-cache is our own training output, trusted.
        # PyTorch 2.6 changed default to True, which rejects TorchVersion objects
        # embedded in cache metadata.
        cache = torch.load(cache_path, map_location=self.device, weights_only=False)
        z_vectors = cache['z_vectors']  # [N, 2048]
        is_sc = cache.get('is_sc', torch.ones(z_vectors.shape[0]))

        # Try to get family labels from cache
        family_labels = cache.get('family_labels', None)

        # Element data for element-anchored sampling (Strategy 4)
        element_indices = cache.get('element_indices', None)
        element_mask = cache.get('element_mask', None)

        if element_indices is None and self.config.element_anchored:
            print("  [Phase 2] Warning: z-cache missing element data — "
                  "element-anchored sampling disabled until cache refresh. "
                  "Delete outputs/latent_cache.pt to force regeneration.",
                  flush=True)

        self.sampler.update_cache(
            z_vectors, is_sc, family_labels,
            element_indices=element_indices,
            element_mask=element_mask,
        )

    def update_z_cache(
        self,
        z_vectors: torch.Tensor,
        is_sc: torch.Tensor,
        family_labels: Optional[torch.Tensor] = None,
        element_indices: Optional[torch.Tensor] = None,
        element_mask: Optional[torch.Tensor] = None,
    ):
        """Update sampler directly with z-vectors (avoids disk I/O)."""
        self.sampler.update_cache(
            z_vectors, is_sc, family_labels,
            element_indices=element_indices,
            element_mask=element_mask,
        )

    def check_safety(
        self,
        current_exact: float,
        phase2_epoch: int,
    ) -> float:
        """Check safety guards and return adjusted Phase 2 weight.

        Returns 0.0 if Phase 2 should be suppressed this epoch.
        """
        self._exact_history.append(current_exact)

        # Weight ramp
        weight = self.get_current_weight(phase2_epoch)

        # Exact match monitor: halve weight if training exact drops
        if len(self._exact_history) > self.config.exact_drop_window:
            recent_max = max(self._exact_history[-self.config.exact_drop_window:])
            if current_exact < recent_max - self.config.exact_drop_threshold:
                weight *= 0.5
                print(f"  [Phase2 SAFETY] Exact dropped {(recent_max - current_exact)*100:.1f}% "
                      f"— halving weight to {weight:.4f}", flush=True)

        # Mode collapse intervention
        if self._collapse_active and self._collapse_epochs_remaining > 0:
            weight *= self.config.collapse_rt_weight_mult  # Boost round-trip
            self._collapse_epochs_remaining -= 1
            if self._collapse_epochs_remaining == 0:
                self._collapse_active = False

        self._current_weight = weight
        return weight

    def run(
        self,
        epoch: int,
        current_exact: float,
        enc_opt: torch.optim.Optimizer,
        dec_opt: torch.optim.Optimizer,
        main_lr: float,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        scaler: Optional[torch.amp.GradScaler] = None,
        stop_boost: float = 0.0,
        hard_stop_threshold: float = 0.0,
        heads_pred_template: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Run one Phase 2 self-supervised sub-epoch.

        Args:
            epoch: Current training epoch
            current_exact: Current training exact match
            enc_opt: Encoder optimizer
            dec_opt: Decoder optimizer
            main_lr: Current main learning rate
            use_amp: Whether to use mixed precision
            amp_dtype: AMP dtype
            scaler: GradScaler for AMP
            stop_boost: Stop prediction boost
            hard_stop_threshold: Hard stop threshold
            heads_pred_template: Template for heads_pred (not used, reserved)

        Returns:
            Dict of Phase 2 metrics for logging
        """
        t_start = time.time()

        phase2_epoch = epoch - self._phase2_activation_epoch
        weight = self.check_safety(current_exact, phase2_epoch)

        if weight <= 1e-8:
            return {'phase2_weight': 0.0, 'phase2_skipped': True}

        # Temporarily adjust LR for Phase 2
        phase2_lr = main_lr * self.config.lr_factor
        original_lrs_enc = []
        original_lrs_dec = []
        for pg in enc_opt.param_groups:
            original_lrs_enc.append(pg['lr'])
            pg['lr'] = phase2_lr
        for pg in dec_opt.param_groups:
            original_lrs_dec.append(pg['lr'])
            pg['lr'] = phase2_lr

        try:
            metrics = self._run_inner(
                epoch, phase2_epoch, weight,
                enc_opt, dec_opt,
                use_amp, amp_dtype, scaler,
                stop_boost, hard_stop_threshold,
            )
        finally:
            # Restore original LR
            for i, pg in enumerate(enc_opt.param_groups):
                pg['lr'] = original_lrs_enc[i]
            for i, pg in enumerate(dec_opt.param_groups):
                pg['lr'] = original_lrs_dec[i]

        metrics['phase2_time'] = time.time() - t_start
        return metrics

    @torch.no_grad()
    def _build_stoich_pred(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Build stoich_pred from encoder heads for decoder conditioning.

        V12 mode (37-dim): fractions(12) + numden(24) + count(1)
        V13 mode (13-dim): fractions(12) + count(1)

        Returns None if encoder lacks fraction_head.
        Pads to decoder's expected stoich_input_dim if needed (V12 decoder + V13 encoder).
        """
        if not hasattr(self.encoder, 'fraction_head'):
            return None
        fraction_output = self.encoder.fraction_head(z)
        fraction_pred = fraction_output[:, :12]
        element_count_pred = fraction_output[:, 12]
        if hasattr(self.encoder, 'numden_head') and self.encoder.use_numden_head:
            numden_pred = self.encoder.numden_head(z)
            stoich = torch.cat([fraction_pred, numden_pred, element_count_pred.unsqueeze(-1)], dim=-1)
        else:
            stoich = torch.cat([fraction_pred, element_count_pred.unsqueeze(-1)], dim=-1)
        # Pad if decoder expects more dims (V12 checkpoint: 37, V13 encoder: 13)
        expected = getattr(self.decoder, 'stoich_input_dim', stoich.shape[-1])
        if stoich.shape[-1] < expected:
            stoich = torch.cat([stoich, torch.zeros(stoich.shape[0], expected - stoich.shape[-1], device=stoich.device)], dim=-1)
        return stoich

    def _empty_metrics(self) -> Dict[str, float]:
        """Return zero-valued Phase 2 metrics dict (used when Phase 2 is skipped)."""
        return {
            'phase2_weight': 0.0,
            'phase2_n_sampled': 0,
            'phase2_n_valid': 0,
            'phase2_valid_rate': 0.0,
            'phase2_total_loss': 0.0,
            'phase2_z_mse': 0.0,
            'phase2_tc_mse': 0.0,
            'phase2_unique_rate': 0.0,
            'phase2_n_degenerate': 0,
            'phase2_n_novel': 0,
            'phase2_n_holdout_recovered': 0,
            'phase2_collapse_active': False,
        }

    def _run_inner(
        self,
        epoch: int,
        phase2_epoch: int,
        weight: float,
        enc_opt: torch.optim.Optimizer,
        dec_opt: torch.optim.Optimizer,
        use_amp: bool,
        amp_dtype: torch.dtype,
        scaler: Optional[torch.amp.GradScaler],
        stop_boost: float,
        hard_stop_threshold: float,
    ) -> Dict[str, float]:
        """Single-pass: sample z → generate → filter → compute losses on ALL valid.

        Returns dict of Phase 2 metrics (prefixed `phase2_*`).

        Duplicate formulas from different z-vectors are kept (each z is a distinct
        latent point needing its own round-trip loss). Degeneracy is tracked as a
        diagnostic metric — it tells us about z-space topology (many z → same formula
        = traveling in a loop).
        """
        self.encoder.train()
        self.decoder.train()

        _ensure_imports()
        from superconductor.models.autoregressive_decoder import indices_to_formula

        # Safety: bail if sampler has no z-cache loaded (e.g. z-cache load failed)
        if not hasattr(self.sampler, '_n_cached') or self.sampler._n_cached == 0:
            print("  [Phase 2] Skipped: sampler has no z-cache loaded", flush=True)
            return self._empty_metrics()

        n_samples = self._n_samples

        # Reset per-sub-epoch novelty counters (cumulative seen_formulas persists)
        self.coverage_tracker.reset_epoch_novelty()

        # Step 1: SAMPLE z-vectors
        z_sampled, sample_stats = self.sampler.sample(n_samples, epoch, phase2_epoch)

        # Step 2: GENERATE formulas
        n_greedy = int(z_sampled.shape[0] * self.config.greedy_fraction)
        all_tokens = []
        all_lp = []
        all_masks = []
        all_formulas = []

        # Greedy decode (no log_probs needed)
        with torch.no_grad():
            if n_greedy > 0:
                z_g = z_sampled[:n_greedy]
                stoich_g = self._build_stoich_pred(z_g)
                tok_g, _, _ = self.decoder.generate_with_kv_cache(
                    z=z_g, stoich_pred=stoich_g, temperature=0.0,
                    max_len=self.max_formula_len,
                    stop_boost=stop_boost,
                    hard_stop_threshold=hard_stop_threshold,
                )
                all_tokens.append(tok_g)
                all_lp.append(torch.zeros_like(tok_g, dtype=torch.float32))
                all_masks.append((tok_g != 0).float())
                for i in range(tok_g.shape[0]):
                    f = (self.v13_tokenizer.decode(tok_g[i].tolist())
                         if self.v13_tokenizer else indices_to_formula(tok_g[i]))
                    all_formulas.append(f)

        # Exploratory decode (with log_probs for REINFORCE)
        n_explore = z_sampled.shape[0] - n_greedy
        if n_explore > 0:
            z_e = z_sampled[n_greedy:]
            temp = self.config.explore_temp_min + (
                self.config.explore_temp_max - self.config.explore_temp_min
            ) * np.random.random()
            if self._collapse_active:
                temp = self.config.collapse_temp_boost

            with torch.no_grad():
                stoich_e = self._build_stoich_pred(z_e)

            tok_e, lp_e, _, mask_e = self.decoder.sample_for_reinforce(
                z=z_e, stoich_pred=stoich_e, temperature=temp,
                max_len=self.max_formula_len,
                stop_boost=stop_boost,
                hard_stop_threshold=hard_stop_threshold,
            )
            all_tokens.append(tok_e)
            all_lp.append(lp_e)
            all_masks.append(mask_e)
            for i in range(tok_e.shape[0]):
                f = (self.v13_tokenizer.decode(tok_e[i].tolist())
                     if self.v13_tokenizer else indices_to_formula(tok_e[i]))
                all_formulas.append(f)

        n_generated = len(all_formulas)

        # Pad to same seq_len
        if not all_tokens:
            return {'phase2_weight': weight, 'phase2_n_sampled': n_samples,
                    'phase2_n_valid': 0, 'phase2_valid_rate': 0.0}
        max_sl = max(t.shape[1] for t in all_tokens)
        pad_t, pad_l, pad_m = [], [], []
        for t, l, m in zip(all_tokens, all_lp, all_masks):
            if t.shape[1] < max_sl:
                p = max_sl - t.shape[1]
                t = F.pad(t, (0, p), value=0)
                l = F.pad(l, (0, p), value=0.0)
                m = F.pad(m, (0, p), value=0.0)
            pad_t.append(t); pad_l.append(l); pad_m.append(m)
        all_tok_t = torch.cat(pad_t, dim=0)   # [n_generated, seq_len]
        all_lp_t = torch.cat(pad_l, dim=0)    # [n_generated, seq_len]
        all_mask_t = torch.cat(pad_m, dim=0)   # [n_generated, seq_len]

        # Step 3: FILTER candidates (parse, chemical, physics, constraints)
        valid_idx, valid_formulas, valid_parsed, filter_stats = \
            self.candidate_filter.filter(
                all_formulas, tokens=all_tok_t, device=self.device,
            )
        n_valid = len(valid_idx)

        # Record per-cluster quality and novelty (which clusters produce garbage vs valid)
        if self.coverage_tracker._fitted and n_valid >= 0:
            valid_mask = torch.zeros(n_generated, dtype=torch.bool, device=self.device)
            for vi in valid_idx:
                valid_mask[vi] = True
            self.coverage_tracker.record_quality(
                z_sampled, valid_mask,
                valid_formulas=valid_formulas,
                valid_idx=valid_idx,
            )

        # Record per-element novelty (strategy-agnostic)
        if self.sampler._has_element_data and valid_formulas:
            self.sampler.record_element_novelty(valid_formulas, valid_parsed)

        # Diagnostic: degeneracy — how many valid z→formula pairs share a formula
        formula_counts = {}
        for f in valid_formulas:
            formula_counts[f] = formula_counts.get(f, 0) + 1
        n_unique_formulas = len(formula_counts)
        n_degenerate = sum(c - 1 for c in formula_counts.values())  # Extra copies
        unique_rate = n_unique_formulas / max(1, n_valid)

        # Collect example formulas for logging (up to 5 valid + 5 invalid)
        _example_valid = list(dict.fromkeys(valid_formulas))[:5]  # Unique, preserving order
        _rejected_set = set(valid_formulas)
        _example_invalid = [f for f in all_formulas if f not in _rejected_set][:5]

        # Mode collapse detection
        if n_valid > 0 and unique_rate < self.config.collapse_threshold:
            if not self._collapse_active:
                print(f"  [Phase2 COLLAPSE] Unique rate {unique_rate:.2f} < "
                      f"{self.config.collapse_threshold} — activating intervention",
                      flush=True)
                self._collapse_active = True
                self._collapse_epochs_remaining = 2

        if n_valid == 0:
            return {
                'phase2_weight': weight,
                'phase2_n_sampled': n_samples,
                'phase2_n_valid': 0,
                'phase2_n_unique_formulas': 0,
                'phase2_n_degenerate': 0,
                'phase2_valid_rate': 0.0,
                'phase2_unique_rate': 0.0,
                'phase2_example_valid': [],
                'phase2_example_invalid': _example_invalid,
                **{f'phase2_filter_{k}': v for k, v in filter_stats.items()},
                **{f'phase2_sample_{k}': v for k, v in sample_stats.items()},
            }

        # Gather valid z-vectors and log_probs (keeping ALL, including duplicates)
        z_valid = z_sampled[valid_idx]                   # [n_valid, 2048]
        lp_valid = all_lp_t[valid_idx]                   # [n_valid, seq_len]
        mask_valid = all_mask_t[valid_idx]                # [n_valid, seq_len]

        # Step 3.5: FLAG novel discoveries (opportunistic — not the goal)
        discovery_stats = {'n_novel': 0, 'n_holdout_recovered': 0}
        if self.discovery_tracker is not None:
            with torch.no_grad():
                dec_result = self.encoder.decode(z_valid)
                tc_preds = dec_result.get('tc_pred')     # [n_valid] normalized
            discovery_stats = self.discovery_tracker.check_batch(
                valid_formulas, z_valid, tc_preds, epoch,
            )

        # Step 4: COMPUTE losses on all valid candidates
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss_result = self.loss_computer.compute(
                    self.encoder, self.decoder,
                    z_valid, valid_parsed, valid_formulas,
                    log_probs=lp_valid, token_mask=mask_valid,
                    all_unique_formulas=self._all_unique_formulas,
                )
        else:
            loss_result = self.loss_computer.compute(
                self.encoder, self.decoder,
                z_valid, valid_parsed, valid_formulas,
                log_probs=lp_valid, token_mask=mask_valid,
                all_unique_formulas=self._all_unique_formulas,
            )

        # Step 5: UPDATE with safety-gated gradients
        total_loss = weight * loss_result['phase2_total']

        if total_loss.requires_grad and total_loss.item() != 0.0:
            enc_opt.zero_grad()
            dec_opt.zero_grad()

            if scaler is not None and use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(enc_opt)
                scaler.unscale_(dec_opt)
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.config.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.decoder.parameters(), self.config.max_grad_norm
                )
                scaler.step(enc_opt)
                scaler.step(dec_opt)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.config.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.decoder.parameters(), self.config.max_grad_norm
                )
                enc_opt.step()
                dec_opt.step()

        # Step 6: Coverage decay + metrics
        self.coverage_tracker.decay_visits()
        coverage_metrics = self.coverage_tracker.get_metrics()

        # Step 7: Build metrics
        metrics = {
            'phase2_weight': weight,
            'phase2_n_sampled': n_samples,
            'phase2_n_valid': n_valid,
            'phase2_n_unique_formulas': n_unique_formulas,
            'phase2_n_degenerate': n_degenerate,
            'phase2_valid_rate': n_valid / max(1, n_generated),
            'phase2_unique_rate': unique_rate,
            'phase2_total_loss': loss_result['phase2_total'].item(),
            'phase2_loss1_rt': loss_result['loss1_round_trip'].item(),
            'phase2_loss2_consist': loss_result['loss2_consistency'].item(),
            'phase2_loss3_physics': loss_result['loss3_physics'].item(),
            'phase2_loss4_reinforce': loss_result['loss4_reinforce'].item(),
            'phase2_z_mse': loss_result['z_mse'].item(),
            'phase2_tc_mse': loss_result['tc_mse'].item(),
            'phase2_collapse_active': int(self._collapse_active),
            'phase2_n_novel': discovery_stats['n_novel'],
            'phase2_n_holdout_recovered': discovery_stats['n_holdout_recovered'],
            'phase2_example_valid': _example_valid,
            'phase2_example_invalid': _example_invalid,
            'phase2_elem_total_unique': (
                sum(len(s) for s in self.sampler._element_seen_formulas.values())
                if self.sampler._element_seen_formulas else 0
            ),
            **{f'phase2_filter_{k}': v for k, v in filter_stats.items()},
            **{f'phase2_sample_{k}': v for k, v in sample_stats.items()},
            **coverage_metrics,
        }

        return metrics

    def get_state(self) -> Dict:
        """Get full Phase 2 state for checkpoint serialization.

        Saves activation epoch, collapse flags, exact history, and coverage tracker.
        Follows EntropyManager.get_state()/load_state() pattern.
        """
        state = {
            'activation_epoch': self._phase2_activation_epoch,
            'activation_exact': self._activation_exact,
            'current_weight': self._current_weight,
            'collapse_active': self._collapse_active,
            'collapse_epochs_remaining': self._collapse_epochs_remaining,
            'exact_history': self._exact_history.copy(),
            'all_unique_formulas': list(self._all_unique_formulas),
            'coverage_tracker': self.coverage_tracker.get_state(),
        }
        # Discovery tracker counters
        if self.discovery_tracker is not None:
            state['discovery_n_novel'] = self.discovery_tracker.n_novel
            state['discovery_n_holdout_recovered'] = self.discovery_tracker.n_holdout_recovered
            state['discovery_seen_novel'] = list(self.discovery_tracker._seen_novel)
            state['discovery_seen_holdout'] = list(self.discovery_tracker._seen_holdout)

        # Sampler state: element visit counts + element seen formulas
        # BUG FIX: These were not persisted before, causing reset on training resume
        state['sampler_element_visit_counts'] = (
            dict(self.sampler._element_visit_counts)
            if self.sampler._element_visit_counts else None
        )
        state['sampler_element_seen_formulas'] = (
            {a: list(s) for a, s in self.sampler._element_seen_formulas.items()}
            if self.sampler._element_seen_formulas else None
        )

        return state

    def load_state(self, state: Dict):
        """Restore Phase 2 state from checkpoint.

        Args:
            state: Dict from get_state(), loaded from checkpoint.
        """
        self._phase2_activation_epoch = state.get('activation_epoch')
        self._activation_exact = state.get('activation_exact')
        self._current_weight = state.get('current_weight', 0.0)
        self._collapse_active = state.get('collapse_active', False)
        self._collapse_epochs_remaining = state.get('collapse_epochs_remaining', 0)
        self._exact_history = state.get('exact_history', [])
        self._all_unique_formulas = set(state.get('all_unique_formulas', []))

        # Restore coverage tracker
        coverage_state = state.get('coverage_tracker')
        if coverage_state is not None:
            self.coverage_tracker.load_state(coverage_state)

        # Restore discovery tracker counters
        if self.discovery_tracker is not None:
            self.discovery_tracker.n_novel = state.get('discovery_n_novel', 0)
            self.discovery_tracker.n_holdout_recovered = state.get('discovery_n_holdout_recovered', 0)
            self.discovery_tracker._seen_novel = set(state.get('discovery_seen_novel', []))
            self.discovery_tracker._seen_holdout = set(state.get('discovery_seen_holdout', []))

        # Restore sampler state (element visit counts + element seen formulas)
        # BUG FIX: These were not persisted before, causing reset on training resume
        elem_visits = state.get('sampler_element_visit_counts')
        if elem_visits is not None:
            self.sampler._element_visit_counts = elem_visits

        elem_seen = state.get('sampler_element_seen_formulas')
        if elem_seen is not None:
            self.sampler._element_seen_formulas = {
                int(a): set(formulas) for a, formulas in elem_seen.items()
            }

        if self._phase2_activation_epoch is not None:
            print(f"  [Phase 2] Restored state: activation_epoch={self._phase2_activation_epoch}, "
                  f"weight={self._current_weight:.4f}, "
                  f"collapse={self._collapse_active}, "
                  f"coverage_fitted={self.coverage_tracker._fitted}", flush=True)

    @property
    def is_active(self) -> bool:
        """Whether Phase 2 has been activated."""
        return self._phase2_activation_epoch is not None

    @property
    def activation_epoch(self) -> Optional[int]:
        """Epoch when Phase 2 was activated."""
        return self._phase2_activation_epoch
