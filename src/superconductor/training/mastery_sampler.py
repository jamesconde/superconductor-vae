"""
Mastery-Aware Training: Focus on Weak Examples, Retire Mastered Ones.

This module implements adaptive sampling that:
1. Tracks per-sample mastery (rolling accuracy over epochs)
2. Samples hard/struggling examples more frequently
3. Maintains minimum replay probability for mastered samples
4. Detects regression when mastered samples start failing

Philosophy:
    Don't waste compute on samples the model has already learned.
    Focus capacity on weaknesses while monitoring for catastrophic forgetting.

Usage:
    tracker = MasteryTracker(n_samples=16000, window_size=5)
    sampler = MasteryAwareSampler(tracker, min_replay_prob=0.1)

    for epoch in range(epochs):
        # Train with weighted sampling
        loader = DataLoader(dataset, sampler=sampler)

        # After epoch, update mastery with predictions
        tracker.update_epoch(sample_indices, correct_mask)
        sampler.update_weights()

        # Check for regression
        regressions = tracker.detect_regressions()
"""

import torch
import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from collections import deque
import warnings


@dataclass
class MasteryStats:
    """Statistics about sample mastery distribution."""
    n_mastered: int           # Samples with mastery >= threshold
    n_struggling: int         # Samples with mastery < 0.5
    n_medium: int             # Samples in between
    n_regressed: int          # Previously mastered, now failing
    n_newly_mastered: int     # Newly reached mastery this epoch

    mastery_mean: float       # Mean mastery score
    mastery_std: float        # Std of mastery scores

    # Distribution buckets
    mastery_histogram: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Mastery Distribution:\n"
            f"  Mastered (>=0.8):  {self.n_mastered:5d} ({self.n_mastered / (self.n_mastered + self.n_medium + self.n_struggling) * 100:.1f}%)\n"
            f"  Medium (0.5-0.8):  {self.n_medium:5d}\n"
            f"  Struggling (<0.5): {self.n_struggling:5d}\n"
            f"  Mean mastery: {self.mastery_mean:.3f} +/- {self.mastery_std:.3f}\n"
            f"  Regressions: {self.n_regressed}, Newly mastered: {self.n_newly_mastered}"
        )


class MasteryTracker:
    """
    Track per-sample mastery over a rolling window of epochs.

    Mastery score = (correct predictions in last N epochs) / N

    A sample is "mastered" when its mastery score exceeds a threshold,
    meaning the model consistently predicts it correctly.

    Args:
        n_samples: Total number of training samples
        window_size: Number of recent epochs to consider for mastery
        mastery_threshold: Score above which a sample is "mastered" (default 0.8)
        regression_threshold: Score drop that indicates regression (default 0.3)
    """

    def __init__(
        self,
        n_samples: int,
        window_size: int = 5,
        mastery_threshold: float = 0.8,
        regression_threshold: float = 0.3
    ):
        self.n_samples = n_samples
        self.window_size = window_size
        self.mastery_threshold = mastery_threshold
        self.regression_threshold = regression_threshold

        # Rolling history: list of deques, one per sample
        # Each deque stores bool (correct/incorrect) for last N epochs
        self.history: List[deque] = [
            deque(maxlen=window_size) for _ in range(n_samples)
        ]

        # Current mastery scores (updated after each epoch)
        self.mastery_scores = np.zeros(n_samples, dtype=np.float32)

        # Track peak mastery (for regression detection)
        self.peak_mastery = np.zeros(n_samples, dtype=np.float32)

        # Track which samples were mastered last epoch
        self.was_mastered = np.zeros(n_samples, dtype=bool)

        # Epoch counter
        self.current_epoch = 0

    def update_epoch(
        self,
        sample_indices: torch.Tensor,
        correct_mask: torch.Tensor
    ) -> MasteryStats:
        """
        Update mastery tracking after an epoch.

        Args:
            sample_indices: Indices of samples that were in this epoch's batches
            correct_mask: Boolean tensor, True if sample was predicted correctly
                         (exact match for formulas)

        Returns:
            MasteryStats for this epoch
        """
        sample_indices = sample_indices.cpu().numpy()
        correct_mask = correct_mask.cpu().numpy()

        # Track newly mastered and regressions
        previously_mastered = self.was_mastered.copy()

        # Update history for each sample
        for idx, correct in zip(sample_indices, correct_mask):
            self.history[idx].append(bool(correct))

        # Recompute mastery scores
        for i in range(self.n_samples):
            if len(self.history[i]) > 0:
                self.mastery_scores[i] = sum(self.history[i]) / len(self.history[i])
            # else: keep at 0 (sample not seen yet)

        # Update peak mastery
        self.peak_mastery = np.maximum(self.peak_mastery, self.mastery_scores)

        # Determine current mastery status
        self.was_mastered = self.mastery_scores >= self.mastery_threshold

        # Count statistics
        n_mastered = np.sum(self.was_mastered)
        n_struggling = np.sum(self.mastery_scores < 0.5)
        n_medium = self.n_samples - n_mastered - n_struggling

        # Detect regressions: was mastered AND score dropped significantly
        regression_mask = (
            previously_mastered &
            (self.peak_mastery - self.mastery_scores > self.regression_threshold)
        )
        n_regressed = np.sum(regression_mask)

        # Detect newly mastered
        newly_mastered = self.was_mastered & ~previously_mastered
        n_newly_mastered = np.sum(newly_mastered)

        # Build histogram
        histogram = {
            '0.0-0.2': np.sum(self.mastery_scores < 0.2),
            '0.2-0.4': np.sum((self.mastery_scores >= 0.2) & (self.mastery_scores < 0.4)),
            '0.4-0.6': np.sum((self.mastery_scores >= 0.4) & (self.mastery_scores < 0.6)),
            '0.6-0.8': np.sum((self.mastery_scores >= 0.6) & (self.mastery_scores < 0.8)),
            '0.8-1.0': np.sum(self.mastery_scores >= 0.8),
        }

        self.current_epoch += 1

        return MasteryStats(
            n_mastered=int(n_mastered),
            n_struggling=int(n_struggling),
            n_medium=int(n_medium),
            n_regressed=int(n_regressed),
            n_newly_mastered=int(n_newly_mastered),
            mastery_mean=float(np.mean(self.mastery_scores)),
            mastery_std=float(np.std(self.mastery_scores)),
            mastery_histogram=histogram
        )

    def get_mastery_scores(self) -> np.ndarray:
        """Get current mastery scores for all samples."""
        return self.mastery_scores.copy()

    def get_struggling_indices(self, threshold: float = 0.5) -> np.ndarray:
        """Get indices of samples with mastery below threshold."""
        return np.where(self.mastery_scores < threshold)[0]

    def get_mastered_indices(self) -> np.ndarray:
        """Get indices of mastered samples."""
        return np.where(self.was_mastered)[0]

    def detect_regressions(self) -> List[int]:
        """
        Detect samples that were mastered but are now failing.

        Returns:
            List of sample indices that have regressed
        """
        regression_mask = (
            (self.peak_mastery >= self.mastery_threshold) &
            (self.peak_mastery - self.mastery_scores > self.regression_threshold)
        )
        return list(np.where(regression_mask)[0])

    def get_sample_history(self, sample_idx: int) -> List[bool]:
        """Get prediction history for a specific sample."""
        return list(self.history[sample_idx])

    def state_dict(self) -> dict:
        """Get state for checkpointing. Allows resuming training with mastery history preserved."""
        return {
            'history': [list(h) for h in self.history],  # Convert deques to lists
            'mastery_scores': self.mastery_scores.tolist(),
            'peak_mastery': self.peak_mastery.tolist(),
            'was_mastered': self.was_mastered.tolist(),
            'current_epoch': self.current_epoch,
        }

    def load_state_dict(self, state: dict):
        """Restore state from checkpoint."""
        # Restore history deques
        for i, h in enumerate(state['history']):
            self.history[i] = deque(h, maxlen=self.window_size)
        self.mastery_scores = np.array(state['mastery_scores'], dtype=np.float32)
        self.peak_mastery = np.array(state['peak_mastery'], dtype=np.float32)
        self.was_mastered = np.array(state['was_mastered'], dtype=bool)
        self.current_epoch = state['current_epoch']

    def reset(self):
        """Reset all tracking (start fresh)."""
        self.history = [deque(maxlen=self.window_size) for _ in range(self.n_samples)]
        self.mastery_scores = np.zeros(self.n_samples, dtype=np.float32)
        self.peak_mastery = np.zeros(self.n_samples, dtype=np.float32)
        self.was_mastered = np.zeros(self.n_samples, dtype=bool)
        self.current_epoch = 0


class MasteryAwareSampler(Sampler):
    """
    Weighted sampler that focuses on struggling samples while maintaining
    minimum replay probability for PERFECTED samples.

    IMPORTANT: min_replay_prob only applies to samples at perfection_threshold (default 0.95),
    NOT at mastery_threshold (default 0.8). This ensures samples at 80-95% mastery still get
    meaningful training to push toward 100% convergence.

    Sampling weight formula:
        if mastery >= perfection_threshold:
            weight = min_replay_prob  # Truly mastered, minimal replay
        else:
            weight = (1 - mastery)^focus_factor  # Still learning, proportional weight

    This means:
        - Struggling samples (mastery=0): weight = 1.0 (always sampled)
        - Medium samples (mastery=0.5): weight = 0.35 (with focus_factor=1.5)
        - Good samples (mastery=0.8): weight = 0.09 (still meaningful!)
        - Near-perfect (mastery=0.95+): weight = min_replay_prob (e.g., 0.1)

    Args:
        tracker: MasteryTracker instance
        min_replay_prob: Minimum sampling probability for perfected samples
        focus_factor: How much to emphasize struggling samples (1.0 = linear, 2.0 = quadratic)
        perfection_threshold: Mastery level at which min_replay_prob kicks in (default 0.95)
        num_samples: Number of samples per epoch (default: full dataset)
    """

    def __init__(
        self,
        tracker: MasteryTracker,
        min_replay_prob: float = 0.1,
        focus_factor: float = 1.5,
        perfection_threshold: float = 0.95,  # Only truly perfected samples get min_replay
        num_samples: Optional[int] = None,
        base_weights: Optional[np.ndarray] = None,  # V12.2: Curriculum-based base weights
    ):
        self.tracker = tracker
        self.min_replay_prob = min_replay_prob
        self.focus_factor = focus_factor
        self.perfection_threshold = perfection_threshold
        self.num_samples = num_samples or tracker.n_samples

        # V12.2: Store curriculum base weights (for 1-fraction formulas, difficult elements)
        if base_weights is not None:
            self.base_weights = base_weights.astype(np.float32)
        else:
            self.base_weights = np.ones(tracker.n_samples, dtype=np.float32)

        # Initialize uniform weights
        self.weights = np.ones(tracker.n_samples, dtype=np.float32)
        self._update_weights()

    def _update_weights(self):
        """Recompute sampling weights from mastery scores."""
        mastery = self.tracker.get_mastery_scores()

        # Weight = (1 - mastery)^focus_factor for samples still learning
        raw_weights = np.power(1.0 - mastery, self.focus_factor)

        # ONLY apply min_replay_prob floor for samples at perfection_threshold
        # This ensures samples at 80% mastery still get trained toward 100%
        is_perfected = mastery >= self.perfection_threshold
        n_perfected = np.sum(is_perfected)

        self.weights = np.where(
            is_perfected,
            self.min_replay_prob,  # Perfected: minimal replay
            np.maximum(raw_weights, 0.01)  # Still learning: use formula (floor at 1% to avoid zero)
        )

        # V12.2: Apply curriculum base weights (1-fraction formulas, difficult elements)
        # This multiplies mastery-based weights by curriculum weights
        self.weights = self.weights * self.base_weights

        # Normalize to sum to num_samples (expected samples per epoch)
        # NOTE: This normalization ensures training NEVER stops even if ALL samples are perfected.
        # When all samples have equal weight (all at 95%+), normalization distributes them uniformly.
        # Example: 16000 samples all at 0.1 weight → each gets normalized to 1.0 → uniform sampling
        self.weights = self.weights / self.weights.sum() * self.num_samples

        # Track perfection ratio for monitoring
        self._perfection_ratio = n_perfected / len(mastery) if len(mastery) > 0 else 0.0

    def update_weights(self):
        """Public method to update weights after tracker is updated."""
        self._update_weights()

    def __iter__(self) -> Iterator[int]:
        """Generate sample indices weighted by difficulty."""
        # Normalize weights to probabilities
        probs = self.weights / self.weights.sum()

        # Sample with replacement according to weights
        indices = np.random.choice(
            self.tracker.n_samples,
            size=self.num_samples,
            replace=True,
            p=probs
        )

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def get_weight_stats(self) -> Dict[str, float]:
        """Get statistics about current sampling weights."""
        mastered_mask = self.tracker.was_mastered
        struggling_mask = self.tracker.mastery_scores < 0.5

        return {
            'mean_weight': float(np.mean(self.weights)),
            'mastered_mean_weight': float(np.mean(self.weights[mastered_mask])) if mastered_mask.any() else 0.0,
            'struggling_mean_weight': float(np.mean(self.weights[struggling_mask])) if struggling_mask.any() else 0.0,
            'weight_ratio': float(np.mean(self.weights[struggling_mask]) / np.mean(self.weights[mastered_mask]))
                           if mastered_mask.any() and struggling_mask.any() else 1.0,
            'perfection_ratio': getattr(self, '_perfection_ratio', 0.0),  # V12.1: % at perfection
        }

    def get_perfection_ratio(self) -> float:
        """Get fraction of samples at perfection_threshold (eligible for retirement)."""
        return getattr(self, '_perfection_ratio', 0.0)


class MasteryAwareDataLoader:
    """
    Convenience wrapper that combines MasteryTracker and MasteryAwareSampler
    with a DataLoader-like interface.

    Handles the epoch-by-epoch workflow:
    1. Iterate through batches with mastery-weighted sampling
    2. Collect predictions during iteration
    3. Update mastery after epoch completes

    Usage:
        loader = MasteryAwareDataLoader(dataset, batch_size=32)

        for epoch in range(epochs):
            for batch_idx, (inputs, targets, sample_indices) in enumerate(loader):
                outputs = model(inputs)
                predictions = outputs.argmax(-1)

                # Register predictions for mastery tracking
                correct = (predictions == targets).all(dim=1)  # Exact match
                loader.register_predictions(sample_indices, correct)

            # End of epoch - update mastery and get stats
            stats = loader.end_epoch()
            print(stats)
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        window_size: int = 5,
        mastery_threshold: float = 0.8,
        min_replay_prob: float = 0.1,
        focus_factor: float = 1.5,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False
    ):
        from torch.utils.data import DataLoader

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples = len(dataset)

        # Create tracker and sampler
        self.tracker = MasteryTracker(
            n_samples=self.n_samples,
            window_size=window_size,
            mastery_threshold=mastery_threshold
        )
        self.sampler = MasteryAwareSampler(
            tracker=self.tracker,
            min_replay_prob=min_replay_prob,
            focus_factor=focus_factor,
            num_samples=self.n_samples
        )

        # Create underlying DataLoader
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )

        # Epoch state
        self._epoch_indices = []
        self._epoch_correct = []

    def __iter__(self):
        """Iterate through batches, yielding (data, indices)."""
        self._epoch_indices = []
        self._epoch_correct = []

        # We need to track which indices were sampled
        # This requires a custom collate or index tracking
        for batch in self._loader:
            yield batch

    def __len__(self):
        return len(self._loader)

    def register_predictions(
        self,
        sample_indices: torch.Tensor,
        correct_mask: torch.Tensor
    ):
        """
        Register predictions for mastery tracking.

        Call this during iteration to record which samples were correct.

        Args:
            sample_indices: Indices of samples in this batch
            correct_mask: Boolean tensor, True if prediction was correct
        """
        self._epoch_indices.append(sample_indices.cpu())
        self._epoch_correct.append(correct_mask.cpu())

    def end_epoch(self) -> MasteryStats:
        """
        Call at end of epoch to update mastery tracking.

        Returns:
            MasteryStats for this epoch
        """
        if not self._epoch_indices:
            warnings.warn("No predictions registered this epoch. Call register_predictions() during iteration.")
            return None

        # Concatenate all batch results
        all_indices = torch.cat(self._epoch_indices)
        all_correct = torch.cat(self._epoch_correct)

        # Update tracker
        stats = self.tracker.update_epoch(all_indices, all_correct)

        # Update sampler weights for next epoch
        self.sampler.update_weights()

        # Clear epoch state
        self._epoch_indices = []
        self._epoch_correct = []

        return stats

    def get_tracker(self) -> MasteryTracker:
        """Get the underlying MasteryTracker."""
        return self.tracker

    def get_sampler(self) -> MasteryAwareSampler:
        """Get the underlying MasteryAwareSampler."""
        return self.sampler


def create_mastery_aware_training_components(
    n_samples: int,
    window_size: int = 5,
    mastery_threshold: float = 0.8,
    min_replay_prob: float = 0.1,
    focus_factor: float = 1.5,
    perfection_threshold: float = 0.95
) -> Tuple[MasteryTracker, MasteryAwareSampler]:
    """
    Factory function to create tracker and sampler.

    Args:
        n_samples: Number of training samples
        window_size: Rolling window for mastery computation
        mastery_threshold: Score above which sample is "mastered" (for stats)
        min_replay_prob: Minimum sampling probability for PERFECTED samples
        focus_factor: Emphasis on struggling samples (1.0=linear, 2.0=quadratic)
        perfection_threshold: Mastery level for retirement (default 0.95, NOT 0.8!)
            - Samples at 80% mastery still get trained toward 100%
            - Only samples at 95%+ get reduced to min_replay_prob

    Returns:
        Tuple of (MasteryTracker, MasteryAwareSampler)
    """
    tracker = MasteryTracker(
        n_samples=n_samples,
        window_size=window_size,
        mastery_threshold=mastery_threshold
    )
    sampler = MasteryAwareSampler(
        tracker=tracker,
        min_replay_prob=min_replay_prob,
        focus_factor=focus_factor,
        perfection_threshold=perfection_threshold,
        num_samples=n_samples
    )
    return tracker, sampler


# =============================================================================
# MULTI-TASK MASTERY TRACKER (V12.1)
# =============================================================================

@dataclass
class MultiTaskMasteryStats:
    """Statistics for multi-task mastery (formula + Tc + Magpie)."""
    n_mastered: int               # Samples mastered on ALL tasks
    n_formula_mastered: int       # Mastered on formula only
    n_tc_mastered: int            # Mastered on Tc only
    n_magpie_mastered: int        # Mastered on Magpie only
    n_struggling: int             # Struggling on any task

    formula_mastery_mean: float
    tc_mastery_mean: float
    magpie_mastery_mean: float
    combined_mastery_mean: float

    def __str__(self) -> str:
        total = self.n_mastered + self.n_struggling + (
            self.n_formula_mastered + self.n_tc_mastered + self.n_magpie_mastered -
            3 * self.n_mastered  # Avoid double counting
        )
        return (
            f"Multi-Task Mastery:\n"
            f"  All tasks mastered: {self.n_mastered:5d}\n"
            f"  Formula mastery: {self.formula_mastery_mean:.3f} ({self.n_formula_mastered} mastered)\n"
            f"  Tc mastery:      {self.tc_mastery_mean:.3f} ({self.n_tc_mastered} mastered)\n"
            f"  Magpie mastery:  {self.magpie_mastery_mean:.3f} ({self.n_magpie_mastered} mastered)\n"
            f"  Combined mean:   {self.combined_mastery_mean:.3f}"
        )


class MultiTaskMasteryTracker:
    """
    Track mastery across multiple tasks: formula reconstruction, Tc prediction, Magpie prediction.

    A sample is considered "mastered" only when ALL tasks exceed their thresholds.
    This prevents sample retirement when only one aspect is learned.

    Philosophy: Focus compute on samples where ANY task is struggling, retire only
    when the model has truly learned all aspects of a sample.

    Args:
        n_samples: Total number of training samples
        window_size: Number of recent epochs to consider for mastery
        formula_threshold: Mastery threshold for formula exact match (default 0.8)
        tc_threshold: Mastery threshold for Tc accuracy (default 0.8)
        magpie_threshold: Mastery threshold for Magpie accuracy (default 0.8)
        task_weights: Optional weights for combining task mastery (default equal)
    """

    def __init__(
        self,
        n_samples: int,
        window_size: int = 5,
        formula_threshold: float = 0.8,
        tc_threshold: float = 0.8,
        magpie_threshold: float = 0.8,
        task_weights: Optional[Dict[str, float]] = None
    ):
        self.n_samples = n_samples
        self.window_size = window_size
        self.formula_threshold = formula_threshold
        self.tc_threshold = tc_threshold
        self.magpie_threshold = magpie_threshold

        # Task weights for combined mastery (default: formula most important)
        self.task_weights = task_weights or {
            'formula': 0.6,  # Formula reconstruction is primary task
            'tc': 0.2,
            'magpie': 0.2
        }

        # Rolling history for each task: list of deques
        self.formula_history: List[deque] = [
            deque(maxlen=window_size) for _ in range(n_samples)
        ]
        self.tc_history: List[deque] = [
            deque(maxlen=window_size) for _ in range(n_samples)
        ]
        self.magpie_history: List[deque] = [
            deque(maxlen=window_size) for _ in range(n_samples)
        ]

        # Current mastery scores per task
        self.formula_mastery = np.zeros(n_samples, dtype=np.float32)
        self.tc_mastery = np.zeros(n_samples, dtype=np.float32)
        self.magpie_mastery = np.zeros(n_samples, dtype=np.float32)

        # Combined mastery score (weighted average)
        self.combined_mastery = np.zeros(n_samples, dtype=np.float32)

        # Track which samples are mastered (ALL tasks above threshold)
        self.was_mastered = np.zeros(n_samples, dtype=bool)

        self.current_epoch = 0

    def update_epoch(
        self,
        sample_indices: torch.Tensor,
        formula_correct: torch.Tensor,
        tc_correct: torch.Tensor,
        magpie_correct: torch.Tensor
    ) -> MultiTaskMasteryStats:
        """
        Update mastery tracking after an epoch.

        Args:
            sample_indices: Indices of samples in this epoch
            formula_correct: Boolean, True if formula exact match
            tc_correct: Boolean, True if Tc error below threshold
            magpie_correct: Boolean, True if Magpie error below threshold

        Returns:
            MultiTaskMasteryStats for this epoch
        """
        indices = sample_indices.cpu().numpy()
        formula_c = formula_correct.cpu().numpy()
        tc_c = tc_correct.cpu().numpy()
        magpie_c = magpie_correct.cpu().numpy()

        # Update history for each sample
        for i, (idx, f_ok, t_ok, m_ok) in enumerate(zip(indices, formula_c, tc_c, magpie_c)):
            self.formula_history[idx].append(bool(f_ok))
            self.tc_history[idx].append(bool(t_ok))
            self.magpie_history[idx].append(bool(m_ok))

        # Recompute mastery scores
        for i in range(self.n_samples):
            if len(self.formula_history[i]) > 0:
                self.formula_mastery[i] = sum(self.formula_history[i]) / len(self.formula_history[i])
            if len(self.tc_history[i]) > 0:
                self.tc_mastery[i] = sum(self.tc_history[i]) / len(self.tc_history[i])
            if len(self.magpie_history[i]) > 0:
                self.magpie_mastery[i] = sum(self.magpie_history[i]) / len(self.magpie_history[i])

        # Compute combined mastery (weighted average)
        self.combined_mastery = (
            self.task_weights['formula'] * self.formula_mastery +
            self.task_weights['tc'] * self.tc_mastery +
            self.task_weights['magpie'] * self.magpie_mastery
        )

        # A sample is mastered only if ALL tasks are above their thresholds
        formula_mastered = self.formula_mastery >= self.formula_threshold
        tc_mastered = self.tc_mastery >= self.tc_threshold
        magpie_mastered = self.magpie_mastery >= self.magpie_threshold

        self.was_mastered = formula_mastered & tc_mastered & magpie_mastered

        self.current_epoch += 1

        # Compute stats
        return MultiTaskMasteryStats(
            n_mastered=int(np.sum(self.was_mastered)),
            n_formula_mastered=int(np.sum(formula_mastered)),
            n_tc_mastered=int(np.sum(tc_mastered)),
            n_magpie_mastered=int(np.sum(magpie_mastered)),
            n_struggling=int(np.sum(self.combined_mastery < 0.5)),
            formula_mastery_mean=float(np.mean(self.formula_mastery)),
            tc_mastery_mean=float(np.mean(self.tc_mastery)),
            magpie_mastery_mean=float(np.mean(self.magpie_mastery)),
            combined_mastery_mean=float(np.mean(self.combined_mastery))
        )

    def get_mastery_scores(self) -> np.ndarray:
        """Get combined mastery scores for sampling."""
        return self.combined_mastery.copy()

    @property
    def mastery_scores(self) -> np.ndarray:
        """Alias for compatibility with MasteryAwareSampler."""
        return self.combined_mastery

    def get_task_mastery(self, task: str) -> np.ndarray:
        """Get mastery scores for a specific task."""
        if task == 'formula':
            return self.formula_mastery.copy()
        elif task == 'tc':
            return self.tc_mastery.copy()
        elif task == 'magpie':
            return self.magpie_mastery.copy()
        else:
            raise ValueError(f"Unknown task: {task}")

    def get_struggling_indices(self, threshold: float = 0.5) -> np.ndarray:
        """Get indices of samples struggling on ANY task."""
        any_struggling = (
            (self.formula_mastery < threshold) |
            (self.tc_mastery < threshold) |
            (self.magpie_mastery < threshold)
        )
        return np.where(any_struggling)[0]

    def get_mastered_indices(self) -> np.ndarray:
        """Get indices of samples mastered on ALL tasks."""
        return np.where(self.was_mastered)[0]

    def state_dict(self) -> dict:
        """Get state for checkpointing. Allows resuming training with mastery history preserved."""
        return {
            'formula_history': [list(h) for h in self.formula_history],
            'tc_history': [list(h) for h in self.tc_history],
            'magpie_history': [list(h) for h in self.magpie_history],
            'formula_mastery': self.formula_mastery.tolist(),
            'tc_mastery': self.tc_mastery.tolist(),
            'magpie_mastery': self.magpie_mastery.tolist(),
            'combined_mastery': self.combined_mastery.tolist(),
            'was_mastered': self.was_mastered.tolist(),
            'current_epoch': self.current_epoch,
        }

    def load_state_dict(self, state: dict):
        """Restore state from checkpoint."""
        for i, h in enumerate(state['formula_history']):
            self.formula_history[i] = deque(h, maxlen=self.window_size)
        for i, h in enumerate(state['tc_history']):
            self.tc_history[i] = deque(h, maxlen=self.window_size)
        for i, h in enumerate(state['magpie_history']):
            self.magpie_history[i] = deque(h, maxlen=self.window_size)
        self.formula_mastery = np.array(state['formula_mastery'], dtype=np.float32)
        self.tc_mastery = np.array(state['tc_mastery'], dtype=np.float32)
        self.magpie_mastery = np.array(state['magpie_mastery'], dtype=np.float32)
        self.combined_mastery = np.array(state['combined_mastery'], dtype=np.float32)
        self.was_mastered = np.array(state['was_mastered'], dtype=bool)
        self.current_epoch = state['current_epoch']

    def reset(self):
        """Reset all tracking."""
        self.formula_history = [deque(maxlen=self.window_size) for _ in range(self.n_samples)]
        self.tc_history = [deque(maxlen=self.window_size) for _ in range(self.n_samples)]
        self.magpie_history = [deque(maxlen=self.window_size) for _ in range(self.n_samples)]
        self.formula_mastery = np.zeros(self.n_samples, dtype=np.float32)
        self.tc_mastery = np.zeros(self.n_samples, dtype=np.float32)
        self.magpie_mastery = np.zeros(self.n_samples, dtype=np.float32)
        self.combined_mastery = np.zeros(self.n_samples, dtype=np.float32)
        self.was_mastered = np.zeros(self.n_samples, dtype=bool)
        self.current_epoch = 0


def create_multitask_mastery_components(
    n_samples: int,
    window_size: int = 5,
    formula_threshold: float = 0.8,
    tc_threshold: float = 0.8,
    magpie_threshold: float = 0.8,
    min_replay_prob: float = 0.1,
    focus_factor: float = 1.5,
    perfection_threshold: float = 0.95,
    task_weights: Optional[Dict[str, float]] = None,
    base_weights: Optional[np.ndarray] = None,  # V12.2: Curriculum-based base weights
) -> Tuple[MultiTaskMasteryTracker, MasteryAwareSampler]:
    """
    Factory function to create multi-task tracker and sampler.

    The sampler uses combined_mastery from the tracker, which is a weighted
    average of all task masteries. Sampling focuses on samples struggling
    on ANY task.

    IMPORTANT: perfection_threshold (default 0.95) determines when samples
    get retired, NOT the per-task thresholds. This ensures samples at 80%
    mastery still get trained toward 100%.

    Args:
        n_samples: Number of training samples
        window_size: Rolling window for mastery computation
        formula_threshold: Threshold for formula mastery (for stats)
        tc_threshold: Threshold for Tc mastery (for stats)
        magpie_threshold: Threshold for Magpie mastery (for stats)
        min_replay_prob: Minimum sampling probability for PERFECTED samples
        focus_factor: Emphasis on struggling samples
        perfection_threshold: Combined mastery level for retirement (default 0.95)
        task_weights: Optional weights for combining task mastery

    Returns:
        Tuple of (MultiTaskMasteryTracker, MasteryAwareSampler)
    """
    tracker = MultiTaskMasteryTracker(
        n_samples=n_samples,
        window_size=window_size,
        formula_threshold=formula_threshold,
        tc_threshold=tc_threshold,
        magpie_threshold=magpie_threshold,
        task_weights=task_weights
    )

    # Create a wrapper that makes MultiTaskMasteryTracker compatible with MasteryAwareSampler
    # The sampler expects a tracker with get_mastery_scores() - which we have
    sampler = MasteryAwareSampler(
        tracker=tracker,  # Works because MultiTaskMasteryTracker has same interface
        min_replay_prob=min_replay_prob,
        focus_factor=focus_factor,
        perfection_threshold=perfection_threshold,
        num_samples=n_samples,
        base_weights=base_weights,  # V12.2: Curriculum-based base weights
    )

    return tracker, sampler
