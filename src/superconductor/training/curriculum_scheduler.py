"""
V15.3: Curriculum-Based AR Warmup Scheduler

Progressively focuses training on formulas the model can already generate
autoregressively, starting with short/simple formulas and advancing to
longer ones as AR exact match improves per bucket.

The model uses V12.41 mode (vocab=148) where fractions are generated
token-by-token (digit-by-digit), making formula sequences longer and
the curriculum more important.

Formula length distribution (52,813 samples, max_len=60):
  Min: 3, Max: 60, Mean: 16.4, Median: 14
  P10: 6, P25: 8, P50: 14, P75: 23, P90: 31

Bucket edges [3, 7, 11, 16, 24, 32, 61] create 6 buckets aligned with
distribution percentiles.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class CurriculumScheduler:
    """Curriculum scheduler that boosts sampling weight for the active length bucket.

    Weights MULTIPLY (not replace) existing sampler weights (class balance,
    Tc-binned, hard-sequence oversampling). This prevents catastrophic
    forgetting of previously learned formulas.

    Lifecycle:
        1. Init with per-sample token counts
        2. Each epoch: get_sample_weights() → multiply into sampler
        3. After AR eval: step(ar_exact_per_bucket) → 'hold'/'advance'/'complete'
        4. On advance: recompute sampler weights with new phase
        5. Save/restore via state_dict()/load_state_dict()
    """

    def __init__(
        self,
        seq_lengths: np.ndarray,
        bucket_edges: List[int] = None,
        advance_threshold: float = 0.50,
        advance_patience: int = 3,
        active_boost: float = 3.0,
        frontier_boost: float = 1.5,
        floor_weight: float = 0.2,
        graduated_weight: float = 0.5,
    ):
        """
        Args:
            seq_lengths: Per-sample non-PAD token counts, shape [N].
            bucket_edges: Bin edges. Creates len(edges)-1 buckets.
                Default: [3, 7, 11, 16, 24, 32, 61] (6 buckets).
            advance_threshold: AR exact match in active bucket to advance.
            advance_patience: Consecutive evals above threshold to advance.
            active_boost: Sampling weight multiplier for active bucket.
            frontier_boost: Sampling weight multiplier for next bucket.
            floor_weight: Min weight for far-future buckets (still sampled).
            graduated_weight: Weight for mastered (past) buckets.
        """
        if bucket_edges is None:
            bucket_edges = [3, 7, 11, 16, 24, 32, 61]

        self.bucket_edges = bucket_edges
        self.n_buckets = len(bucket_edges) - 1
        self.advance_threshold = advance_threshold
        self.advance_patience = advance_patience
        self.active_boost = active_boost
        self.frontier_boost = frontier_boost
        self.floor_weight = floor_weight
        self.graduated_weight = graduated_weight

        # Assign each sample to a bucket
        self.seq_lengths = np.asarray(seq_lengths, dtype=np.float32)
        self.bucket_ids = np.full(len(seq_lengths), -1, dtype=np.int32)
        for i in range(self.n_buckets):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            mask = (self.seq_lengths >= lo) & (self.seq_lengths < hi)
            self.bucket_ids[mask] = i

        # Samples outside all buckets (shouldn't happen with proper edges) get floor weight
        # but still participate in training

        # State
        self.current_phase = 0       # Index of currently active bucket
        self.advance_counter = 0     # Consecutive evals above threshold
        self.epoch = 0               # Curriculum step counter
        self.ar_history: Dict[int, List[float]] = {i: [] for i in range(self.n_buckets)}

        # Print bucket distribution
        self._print_distribution()

    def _print_distribution(self):
        """Print bucket distribution summary."""
        print(f"  [Curriculum AR] {self.n_buckets} buckets, phase={self.current_phase}, "
              f"advance@{self.advance_threshold:.0%} x{self.advance_patience}")
        for i in range(self.n_buckets):
            lo, hi = self.bucket_edges[i], self.bucket_edges[i + 1]
            n = int((self.bucket_ids == i).sum())
            pct = n / len(self.bucket_ids) * 100
            label = f"{lo}-{hi-1}" if i < self.n_buckets - 1 else f"{lo}+"
            status = ("ACTIVE" if i == self.current_phase
                      else "frontier" if i == self.current_phase + 1
                      else "graduated" if i < self.current_phase
                      else "future")
            print(f"    Bucket {i} [{label:>5s}]: {n:5d} samples ({pct:5.1f}%) — {status}")

    def get_sample_weights(self) -> np.ndarray:
        """Return per-sample weight multipliers for the current phase.

        Returns:
            np.ndarray of shape [N] with weight multipliers.
            Graduated buckets → graduated_weight (0.5),
            Active bucket → active_boost (3.0),
            Frontier (next) → frontier_boost (1.5),
            Future buckets → floor_weight (0.2),
            Samples outside buckets → floor_weight.
        """
        weights = np.full(len(self.seq_lengths), self.floor_weight, dtype=np.float64)

        for i in range(self.n_buckets):
            mask = self.bucket_ids == i
            if i < self.current_phase:
                # Graduated — mastered but still sampled
                weights[mask] = self.graduated_weight
            elif i == self.current_phase:
                # Active — boosted
                weights[mask] = self.active_boost
            elif i == self.current_phase + 1:
                # Frontier — next up, moderate boost
                weights[mask] = self.frontier_boost
            else:
                # Future — floor weight (already set)
                pass

        return weights

    def step(self, ar_exact_per_bucket: Dict[int, float]) -> str:
        """Update state after an AR evaluation.

        Args:
            ar_exact_per_bucket: {bucket_index: ar_exact_match_fraction}
                Only buckets with data need to be present.

        Returns:
            'hold': Staying at current phase.
            'advance': Advanced to next phase.
            'complete': All buckets graduated.
        """
        self.epoch += 1

        # Record history
        for bucket_id, ar_exact in ar_exact_per_bucket.items():
            if bucket_id in self.ar_history:
                self.ar_history[bucket_id].append(ar_exact)

        # Check active bucket performance
        active_ar = ar_exact_per_bucket.get(self.current_phase)
        if active_ar is None:
            # No data for active bucket (shouldn't happen) — hold
            return 'hold'

        if active_ar >= self.advance_threshold:
            self.advance_counter += 1
        else:
            self.advance_counter = 0

        if self.advance_counter >= self.advance_patience:
            # Advance to next phase
            self.advance_counter = 0
            self.current_phase += 1

            if self.current_phase >= self.n_buckets:
                # All buckets graduated
                self.current_phase = self.n_buckets - 1  # Stay at last bucket
                return 'complete'

            return 'advance'

        return 'hold'

    def get_status_string(self) -> str:
        """One-liner for epoch logging."""
        lo = self.bucket_edges[self.current_phase]
        hi = self.bucket_edges[self.current_phase + 1]
        label = f"{lo}-{hi-1}" if self.current_phase < self.n_buckets - 1 else f"{lo}+"
        return (f"phase={self.current_phase}/{self.n_buckets-1} ([{label}]), "
                f"patience={self.advance_counter}/{self.advance_patience}")

    def state_dict(self) -> dict:
        """Serialize state for checkpoint saving."""
        return {
            'current_phase': self.current_phase,
            'advance_counter': self.advance_counter,
            'epoch': self.epoch,
            'ar_history': {k: list(v) for k, v in self.ar_history.items()},
            # Config for validation on restore
            'bucket_edges': self.bucket_edges,
            'advance_threshold': self.advance_threshold,
            'advance_patience': self.advance_patience,
        }

    def load_state_dict(self, state: dict):
        """Restore state from checkpoint."""
        self.current_phase = state['current_phase']
        self.advance_counter = state['advance_counter']
        self.epoch = state.get('epoch', 0)

        # Restore AR history (convert string keys back to int if needed)
        saved_history = state.get('ar_history', {})
        for k, v in saved_history.items():
            bucket_id = int(k)
            if bucket_id in self.ar_history:
                self.ar_history[bucket_id] = list(v)

        # Validate config consistency
        saved_edges = state.get('bucket_edges')
        if saved_edges and saved_edges != self.bucket_edges:
            print(f"  [Curriculum AR] WARNING: bucket_edges changed "
                  f"{saved_edges} → {self.bucket_edges}. Resetting to phase 0.")
            self.current_phase = 0
            self.advance_counter = 0
