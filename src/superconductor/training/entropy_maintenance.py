"""
Entropy Maintenance Strategies for REINFORCE Training.

Prevents entropy collapse (policy becoming too deterministic) during RL training
by dynamically adjusting exploration incentives. This module implements multiple
strategies based on recent research:

- Adaptive Entropy Regularization (AER) - dynamically adjust entropy coefficient
- Temperature Warm Restarts - cyclically boost sampling temperature
- Per-Position Entropy Weighting - focus exploration on error-prone positions
- Novelty Bonus - reward diverse generations
- Uncertainty-Guided Exploration - use reward variance to guide exploration

References:
- "Adaptive Entropy Regularization for RLVR" (2024) - arxiv.org/abs/2510.10959
- "EPO: Entropy-regularized Policy Optimization" (2024) - arxiv.org/abs/2509.22576
- "Entropy Collapse in Large Language Models" (2024)

Usage:
    from superconductor.training.entropy_maintenance import EntropyManager

    entropy_manager = EntropyManager(
        strategy='adaptive',
        min_entropy=0.1,
        target_entropy=0.5,
    )

    for epoch in range(n_epochs):
        # Get current entropy weight and temperature
        entropy_weight = entropy_manager.get_entropy_weight(epoch, current_entropy)
        temperature = entropy_manager.get_temperature(epoch, current_entropy)

        # Train with adjusted parameters
        ...

        # Update manager with training metrics
        entropy_manager.update(
            epoch=epoch,
            entropy=current_entropy,
            reward=mean_reward,
            exact_match=exact_match,
        )
"""

import numpy as np
from typing import Optional, List, Dict, Literal, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import math


class EntropyStrategy(Enum):
    """Available entropy maintenance strategies."""
    CONSTANT = 'constant'                    # Fixed entropy weight (baseline)
    ADAPTIVE = 'adaptive'                    # AER-style adaptive coefficient
    CYCLICAL = 'cyclical'                    # Temperature warm restarts
    POSITION_WEIGHTED = 'position_weighted'  # Per-position weighting
    NOVELTY_BONUS = 'novelty_bonus'          # Reward diversity bonus
    UNCERTAINTY_GUIDED = 'uncertainty'       # Variance-based exploration
    COMPOSITE = 'composite'                  # Combine multiple strategies
    CAUSAL = 'causal'                        # Diagnose plateau cause before boosting


@dataclass
class EntropyConfig:
    """Configuration for entropy maintenance."""
    strategy: str = 'adaptive'

    # Target entropy levels
    min_entropy: float = 0.1                 # Minimum acceptable entropy (nats)
    target_entropy: float = 0.5              # Target entropy to maintain
    max_entropy: float = 2.0                 # Maximum entropy (near-uniform)

    # Adaptive entropy (AER) parameters
    entropy_weight_base: float = 0.2         # Base entropy bonus weight
    entropy_weight_min: float = 0.05         # Minimum entropy weight
    entropy_weight_max: float = 1.0          # Maximum entropy weight
    adaptation_rate: float = 0.1             # How fast to adapt (learning rate)
    plateau_window: int = 10                 # Epochs to detect plateau
    plateau_threshold: float = 0.01          # Improvement threshold for plateau (relative by default)
    plateau_relative: bool = True            # If True, threshold is relative to current performance

    # Temperature parameters
    temperature_base: float = 0.8            # Base sampling temperature
    temperature_min: float = 0.5             # Minimum temperature
    temperature_max: float = 1.5             # Maximum temperature
    temperature_restart_period: int = 50     # Epochs between warm restarts
    temperature_restart_boost: float = 0.3   # Temperature boost at restart

    # Per-position parameters
    position_decay: float = 0.9              # Decay factor for position weights
    error_position_boost: float = 2.0        # Extra weight for error-prone positions

    # Novelty bonus parameters
    novelty_weight: float = 0.1              # Weight for novelty bonus
    novelty_buffer_size: int = 1000          # Size of generation history buffer
    novelty_distance_metric: str = 'edit'    # 'edit', 'jaccard', 'embedding'

    # Uncertainty-guided parameters
    uncertainty_window: int = 5              # Window for reward variance estimation
    uncertainty_weight: float = 0.2          # Weight for uncertainty bonus
    variance_threshold: float = 0.1          # High variance = uncertain = explore

    # Causal diagnosis parameters
    causal_diagnosis_window: int = 10        # Epochs to check entropy trend before plateau
    causal_followup_window: int = 10         # Epochs to check if boost helped
    causal_entropy_drop_threshold: float = 0.1  # 10% drop = entropy dropped
    causal_min_success_rate: float = 0.3     # Min success rate to trust boosts
    causal_strong_boost: float = 2.0         # Multiplier for strong evidence
    causal_weak_boost: float = 1.3           # Multiplier for weak evidence
    causal_minimal_boost: float = 1.1        # Multiplier for poor history


class AdaptiveEntropyScheduler:
    """
    Adaptive Entropy Regularization (AER) scheduler.

    Dynamically adjusts entropy coefficient based on:
    1. Current entropy level vs target
    2. Training plateau detection
    3. Improvement rate in rewards/accuracy

    Based on: "Adaptive Entropy Regularization for RLVR" (2024)
    """

    def __init__(
        self,
        target_entropy: float = 0.5,
        min_entropy: float = 0.1,
        base_weight: float = 0.2,
        min_weight: float = 0.05,
        max_weight: float = 1.0,
        adaptation_rate: float = 0.1,
        plateau_window: int = 10,
        plateau_threshold: float = 0.01,
        plateau_relative: bool = True,
    ):
        self.target_entropy = target_entropy
        self.min_entropy = min_entropy
        self.base_weight = base_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adaptation_rate = adaptation_rate
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.plateau_relative = plateau_relative

        # Current entropy weight
        self.current_weight = base_weight

        # History tracking
        self.entropy_history: Deque[float] = deque(maxlen=plateau_window * 2)
        self.reward_history: Deque[float] = deque(maxlen=plateau_window * 2)
        self.exact_match_history: Deque[float] = deque(maxlen=plateau_window * 2)

        # Plateau detection state
        self._plateau_detected = False
        self._plateau_epochs = 0

    def update(
        self,
        entropy: float,
        reward: Optional[float] = None,
        exact_match: Optional[float] = None,
    ) -> float:
        """
        Update entropy weight based on current metrics.

        Args:
            entropy: Current policy entropy (nats)
            reward: Current mean reward (optional)
            exact_match: Current exact match rate (optional)

        Returns:
            Updated entropy weight
        """
        self.entropy_history.append(entropy)
        if reward is not None:
            self.reward_history.append(reward)
        if exact_match is not None:
            self.exact_match_history.append(exact_match)

        # Compute weight adjustment
        adjustment = 0.0

        # 1. Entropy gap adjustment
        entropy_gap = self.target_entropy - entropy
        if entropy < self.min_entropy:
            # Critical: entropy too low, strong boost
            adjustment += self.adaptation_rate * 2.0 * (self.min_entropy - entropy) / self.min_entropy
        elif entropy_gap > 0:
            # Below target: increase weight
            adjustment += self.adaptation_rate * entropy_gap / self.target_entropy
        else:
            # Above target: slight decrease
            adjustment -= self.adaptation_rate * 0.5 * abs(entropy_gap) / self.target_entropy

        # 2. Plateau detection adjustment
        if self._detect_plateau():
            # Plateau detected: boost entropy to encourage exploration
            adjustment += self.adaptation_rate * 0.5
            self._plateau_detected = True
            self._plateau_epochs += 1
        else:
            self._plateau_detected = False
            self._plateau_epochs = 0

        # 3. Apply adjustment with bounds
        self.current_weight = np.clip(
            self.current_weight + adjustment,
            self.min_weight,
            self.max_weight
        )

        return self.current_weight

    def _detect_plateau(self) -> bool:
        """
        Detect if training is in a plateau (no improvement).

        Uses relative threshold by default: improvement must exceed
        plateau_threshold * current_performance to not be a plateau.

        Example with plateau_threshold=0.01 (1%):
        - At 80% exact match: need >0.8% improvement (0.008 absolute)
        - At 20% exact match: need >0.2% improvement (0.002 absolute)
        """
        if len(self.exact_match_history) < self.plateau_window:
            return False

        # Check exact match improvement over window
        recent = list(self.exact_match_history)[-self.plateau_window:]
        older = list(self.exact_match_history)[-2 * self.plateau_window:-self.plateau_window]

        if len(older) < self.plateau_window:
            return False

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        improvement = recent_mean - older_mean

        if self.plateau_relative:
            # Relative threshold: improvement must be > threshold * current_performance
            # Use older_mean as baseline to avoid division issues when improving
            baseline = max(older_mean, 0.01)  # Avoid division by zero
            relative_improvement = improvement / baseline
            return relative_improvement < self.plateau_threshold
        else:
            # Absolute threshold (original behavior)
            return improvement < self.plateau_threshold

    def get_weight(self) -> float:
        """Get current entropy weight."""
        return self.current_weight

    def get_state(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            'current_weight': self.current_weight,
            'entropy_history': list(self.entropy_history),
            'reward_history': list(self.reward_history),
            'exact_match_history': list(self.exact_match_history),
            'plateau_detected': self._plateau_detected,
            'plateau_epochs': self._plateau_epochs,
        }

    def load_state(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.current_weight = state['current_weight']
        self.entropy_history = deque(state['entropy_history'], maxlen=self.plateau_window * 2)
        self.reward_history = deque(state['reward_history'], maxlen=self.plateau_window * 2)
        self.exact_match_history = deque(state['exact_match_history'], maxlen=self.plateau_window * 2)
        self._plateau_detected = state.get('plateau_detected', False)
        self._plateau_epochs = state.get('plateau_epochs', 0)


class CausalEntropyScheduler:
    """
    Causal Entropy Scheduler - diagnoses plateau causes before intervening.

    Unlike the naive approach of "plateau → boost entropy", this scheduler:
    1. Checks if entropy actually dropped BEFORE the plateau started
    2. Checks if entropy is currently below minimum threshold
    3. Uses tiered response based on evidence strength
    4. Tracks intervention success and adjusts confidence accordingly

    Evidence Scoring:
    - STRONG: Both entropy dropped AND entropy is low
    - WEAK: Either entropy dropped OR entropy is low
    - NONE: Neither condition met → no boost

    The boost amount is further modulated by historical success rate.
    """

    def __init__(
        self,
        base_weight: float = 0.2,
        min_weight: float = 0.05,
        max_weight: float = 1.0,
        min_entropy: float = 0.1,
        diagnosis_window: int = 10,
        followup_window: int = 10,
        plateau_window: int = 10,
        plateau_threshold: float = 0.01,
        plateau_relative: bool = True,
        entropy_drop_threshold: float = 0.1,
        min_success_rate: float = 0.3,
        strong_boost: float = 2.0,
        weak_boost: float = 1.3,
        minimal_boost: float = 1.1,
    ):
        self.base_weight = base_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_entropy = min_entropy
        self.diagnosis_window = diagnosis_window
        self.followup_window = followup_window
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.plateau_relative = plateau_relative
        self.entropy_drop_threshold = entropy_drop_threshold
        self.min_success_rate = min_success_rate
        self.strong_boost = strong_boost
        self.weak_boost = weak_boost
        self.minimal_boost = minimal_boost

        # Current state
        self.current_weight = base_weight

        # History tracking (longer for causal analysis)
        history_len = max(diagnosis_window * 3, plateau_window * 3)
        self.entropy_history: Deque[float] = deque(maxlen=history_len)
        self.exact_match_history: Deque[float] = deque(maxlen=history_len)

        # Intervention tracking
        self.interventions: List[Dict] = []

        # State machine
        self.state = 'MONITORING'  # MONITORING or BOOSTING
        self.boost_start_epoch: Optional[int] = None
        self.pre_boost_exact_match: Optional[float] = None
        self.last_diagnosis: Optional[Dict] = None

    def update(
        self,
        epoch: int,
        entropy: float,
        exact_match: float,
        reward: Optional[float] = None,
    ) -> float:
        """
        Update scheduler and return entropy weight to use.

        Args:
            epoch: Current training epoch
            entropy: Current policy entropy
            exact_match: Current exact match rate
            reward: Current mean reward (optional, for logging)

        Returns:
            Entropy weight to use
        """
        self.entropy_history.append(entropy)
        self.exact_match_history.append(exact_match)

        if self.state == 'MONITORING':
            if self._detect_plateau():
                self.last_diagnosis = self._diagnose_plateau_cause()

                if self.last_diagnosis['evidence'] != 'none':
                    self.state = 'BOOSTING'
                    self.boost_start_epoch = epoch
                    self.pre_boost_exact_match = exact_match
                    self.current_weight = self._compute_boost_amount(self.last_diagnosis)
                    print(f"  [Entropy] Plateau + {self.last_diagnosis['evidence']} evidence "
                          f"→ {self.current_weight/self.base_weight:.1f}x boost "
                          f"(dropped={self.last_diagnosis['entropy_dropped']}, "
                          f"low={self.last_diagnosis['entropy_low']})")
                    return self.current_weight
                else:
                    # Plateau detected but entropy not implicated
                    print(f"  [Entropy] Plateau detected but entropy not implicated "
                          f"(no drop, not low) - no boost")
                    return self.base_weight
            return self.current_weight

        elif self.state == 'BOOSTING':
            # Check if we should exit boosting
            if epoch - self.boost_start_epoch >= self.followup_window:
                success = self._evaluate_intervention_success()
                self._record_intervention(success)
                status = "SUCCESS - plateau broken" if success else "FAILURE - plateau persists"
                print(f"  [Entropy] Boost ended: {status} "
                      f"(history: {len(self.interventions)} interventions, "
                      f"{sum(i['success'] for i in self.interventions)}/{len(self.interventions)} successful)")
                self.state = 'MONITORING'
                self.current_weight = self.base_weight
                return self.base_weight
            return self.current_weight

        return self.current_weight

    def _detect_plateau(self) -> bool:
        """Detect if training is in a plateau (no improvement)."""
        if len(self.exact_match_history) < self.plateau_window * 2:
            return False

        recent = list(self.exact_match_history)[-self.plateau_window:]
        older = list(self.exact_match_history)[-2 * self.plateau_window:-self.plateau_window]

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        improvement = recent_mean - older_mean

        if self.plateau_relative:
            baseline = max(older_mean, 0.01)
            relative_improvement = improvement / baseline
            return relative_improvement < self.plateau_threshold
        else:
            return improvement < self.plateau_threshold

    def _diagnose_plateau_cause(self) -> Dict:
        """
        Analyze whether entropy drop likely caused the plateau.

        Returns dict with:
        - evidence: 'strong', 'weak', or 'none'
        - entropy_dropped: bool
        - entropy_low: bool
        - success_rate: float (from history)
        - pre_plateau_entropy: float
        - current_entropy: float
        """
        # Get entropy before and during plateau
        if len(self.entropy_history) < self.diagnosis_window * 2:
            return {'evidence': 'none', 'reason': 'insufficient_history',
                    'entropy_dropped': False, 'entropy_low': False,
                    'success_rate': 0.5, 'pre_plateau_entropy': 0, 'current_entropy': 0}

        pre_plateau_entropy = list(self.entropy_history)[-2 * self.diagnosis_window:-self.diagnosis_window]
        during_plateau_entropy = list(self.entropy_history)[-self.diagnosis_window:]

        pre_mean = np.mean(pre_plateau_entropy)
        during_mean = np.mean(during_plateau_entropy)

        # Condition A: Did entropy drop before/during plateau?
        if pre_mean > 0.01:
            entropy_drop_ratio = (pre_mean - during_mean) / pre_mean
            entropy_dropped = entropy_drop_ratio > self.entropy_drop_threshold
        else:
            entropy_dropped = False

        # Condition B: Is entropy currently low?
        entropy_low = during_mean < self.min_entropy

        # Evidence scoring (tiered)
        if entropy_dropped and entropy_low:
            evidence = 'strong'
        elif entropy_dropped or entropy_low:
            evidence = 'weak'
        else:
            evidence = 'none'

        # Historical success rate
        if len(self.interventions) >= 3:
            recent = self.interventions[-5:] if len(self.interventions) >= 5 else self.interventions
            success_rate = sum(i['success'] for i in recent) / len(recent)
        else:
            success_rate = 0.5  # Unknown, assume moderate

        return {
            'evidence': evidence,
            'entropy_dropped': entropy_dropped,
            'entropy_low': entropy_low,
            'success_rate': success_rate,
            'pre_plateau_entropy': pre_mean,
            'current_entropy': during_mean,
        }

    def _compute_boost_amount(self, diagnosis: Dict) -> float:
        """
        Compute entropy weight based on evidence level and historical success.

        Tiered response:
        - Strong evidence + good history → strong_boost (2.0x)
        - Weak evidence + good history   → weak_boost (1.3x)
        - Any evidence + poor history    → minimal_boost (1.1x)
        - No evidence                    → 1.0x (no boost)
        """
        evidence = diagnosis['evidence']
        success_rate = diagnosis['success_rate']

        if evidence == 'none':
            return self.base_weight  # No boost

        # Discount based on historical success
        if success_rate < self.min_success_rate:
            # Poor history: minimal boost regardless of evidence
            multiplier = self.minimal_boost
        elif evidence == 'strong':
            multiplier = self.strong_boost
        else:  # weak
            multiplier = self.weak_boost

        return np.clip(
            self.base_weight * multiplier,
            self.min_weight,
            self.max_weight
        )

    def _evaluate_intervention_success(self) -> bool:
        """Check if the entropy boost broke the plateau."""
        if self.pre_boost_exact_match is None:
            return False

        current_exact = self.exact_match_history[-1]
        improvement = current_exact - self.pre_boost_exact_match

        # Success = meaningful improvement during boost period (relative threshold)
        threshold = self.plateau_threshold * self.pre_boost_exact_match
        return improvement > threshold

    def _record_intervention(self, success: bool):
        """Record intervention outcome for learning."""
        self.interventions.append({
            'epoch': self.boost_start_epoch,
            'evidence': self.last_diagnosis['evidence'] if self.last_diagnosis else 'unknown',
            'boost_multiplier': self.current_weight / self.base_weight,
            'success': success,
            'pre_exact': self.pre_boost_exact_match,
            'post_exact': self.exact_match_history[-1] if self.exact_match_history else None,
        })

    def get_weight(self) -> float:
        """Get current entropy weight."""
        return self.current_weight

    def get_state(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            'current_weight': self.current_weight,
            'entropy_history': list(self.entropy_history),
            'exact_match_history': list(self.exact_match_history),
            'interventions': self.interventions.copy(),
            'state': self.state,
            'boost_start_epoch': self.boost_start_epoch,
            'pre_boost_exact_match': self.pre_boost_exact_match,
            'last_diagnosis': self.last_diagnosis,
        }

    def load_state(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.current_weight = state['current_weight']
        history_len = max(self.diagnosis_window * 3, self.plateau_window * 3)
        self.entropy_history = deque(state['entropy_history'], maxlen=history_len)
        self.exact_match_history = deque(state['exact_match_history'], maxlen=history_len)
        self.interventions = state.get('interventions', [])
        self.state = state.get('state', 'MONITORING')
        self.boost_start_epoch = state.get('boost_start_epoch')
        self.pre_boost_exact_match = state.get('pre_boost_exact_match')
        self.last_diagnosis = state.get('last_diagnosis')

    def get_info(self) -> Dict:
        """Get detailed info about current state."""
        success_count = sum(i['success'] for i in self.interventions) if self.interventions else 0
        total_count = len(self.interventions)

        return {
            'state': self.state,
            'current_weight': self.current_weight,
            'base_weight': self.base_weight,
            'boost_multiplier': self.current_weight / self.base_weight,
            'total_interventions': total_count,
            'successful_interventions': success_count,
            'success_rate': success_count / total_count if total_count > 0 else 0.5,
            'current_entropy': self.entropy_history[-1] if self.entropy_history else None,
            'last_diagnosis': self.last_diagnosis,
        }


class TemperatureWarmRestartScheduler:
    """
    Temperature warm restart scheduler for exploration.

    Periodically boosts sampling temperature to encourage exploration,
    similar to learning rate warm restarts but for the policy temperature.

    Schedule: T(t) = T_min + boost * decay^(t mod period)
    """

    def __init__(
        self,
        base_temperature: float = 0.8,
        min_temperature: float = 0.5,
        max_temperature: float = 1.5,
        restart_period: int = 50,
        restart_boost: float = 0.3,
        decay_rate: float = 0.95,
    ):
        self.base_temperature = base_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.restart_period = restart_period
        self.restart_boost = restart_boost
        self.decay_rate = decay_rate

        self.current_temperature = base_temperature

    def get_temperature(self, epoch: int, current_entropy: Optional[float] = None) -> float:
        """
        Get temperature for current epoch.

        Args:
            epoch: Current training epoch
            current_entropy: Current entropy (optional, for adaptive boost)

        Returns:
            Sampling temperature
        """
        # Position within current restart period
        position = epoch % self.restart_period

        # Compute decayed boost
        if position == 0 and epoch > 0:
            # Restart point: apply boost
            boost = self.restart_boost
        else:
            # Decay from last restart
            boost = self.restart_boost * (self.decay_rate ** position)

        # Adaptive boost if entropy is very low
        if current_entropy is not None and current_entropy < 0.1:
            boost *= 1.5  # Extra boost when entropy collapses

        temperature = self.base_temperature + boost
        self.current_temperature = np.clip(temperature, self.min_temperature, self.max_temperature)

        return self.current_temperature

    def get_state(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {'current_temperature': self.current_temperature}

    def load_state(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.current_temperature = state['current_temperature']


class PerPositionEntropyWeighter:
    """
    Per-position entropy weighting for sequence generation.

    Tracks error rates at each position and weights entropy bonus accordingly:
    - Positions with high error rates get higher entropy weight (more exploration)
    - Positions with low error rates get lower weight (exploitation)

    This focuses exploration where it matters most.
    """

    def __init__(
        self,
        max_len: int = 60,
        base_weight: float = 1.0,
        error_boost: float = 2.0,
        decay: float = 0.99,
        smoothing: float = 0.1,
    ):
        self.max_len = max_len
        self.base_weight = base_weight
        self.error_boost = error_boost
        self.decay = decay
        self.smoothing = smoothing

        # Error rate tracking per position (EMA)
        self.position_error_rates = np.ones(max_len) * 0.5  # Start at 50%
        self.position_counts = np.zeros(max_len)

    def update(self, position_errors: np.ndarray, position_mask: np.ndarray):
        """
        Update error rates with new batch data.

        Args:
            position_errors: Binary array of errors per position (batch, seq_len)
            position_mask: Valid positions mask (batch, seq_len)
        """
        # Compute error rate per position for this batch
        batch_errors = position_errors.sum(axis=0)
        batch_counts = position_mask.sum(axis=0)

        # Update EMA for each position
        for pos in range(min(len(batch_counts), self.max_len)):
            if batch_counts[pos] > 0:
                batch_rate = batch_errors[pos] / batch_counts[pos]
                # Exponential moving average
                self.position_error_rates[pos] = (
                    self.decay * self.position_error_rates[pos] +
                    (1 - self.decay) * batch_rate
                )
                self.position_counts[pos] += batch_counts[pos]

    def get_weights(self) -> np.ndarray:
        """
        Get entropy weights per position.

        Returns:
            Array of weights (max_len,)
        """
        # Higher error rate -> higher weight
        weights = self.base_weight + self.error_boost * self.position_error_rates

        # Apply smoothing to avoid drastic changes
        smoothed = np.convolve(weights, np.ones(3) / 3, mode='same')
        return smoothed

    def get_weight_for_position(self, position: int) -> float:
        """Get entropy weight for specific position."""
        if position >= self.max_len:
            return self.base_weight
        return self.base_weight + self.error_boost * self.position_error_rates[position]

    def get_state(self) -> Dict:
        """Get state for checkpointing."""
        return {
            'position_error_rates': self.position_error_rates.tolist(),
            'position_counts': self.position_counts.tolist(),
        }

    def load_state(self, state: Dict):
        """Load state from checkpoint."""
        self.position_error_rates = np.array(state['position_error_rates'])
        self.position_counts = np.array(state['position_counts'])


class NoveltyBonus:
    """
    Novelty bonus for encouraging diverse generations.

    Tracks recent generations and rewards novel outputs that differ
    from the history. This prevents the model from collapsing to
    a small set of "safe" outputs.

    Novelty metrics:
    - 'edit': Levenshtein edit distance
    - 'jaccard': Jaccard similarity of token sets
    - 'unique_tokens': Count of unique tokens generated
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        novelty_weight: float = 0.1,
        distance_metric: str = 'edit',
        k_nearest: int = 5,
    ):
        self.buffer_size = buffer_size
        self.novelty_weight = novelty_weight
        self.distance_metric = distance_metric
        self.k_nearest = k_nearest

        # Generation history buffer
        self.history: Deque[tuple] = deque(maxlen=buffer_size)

    def compute_novelty(self, generated_tokens: List[List[int]]) -> np.ndarray:
        """
        Compute novelty bonus for each generation.

        Args:
            generated_tokens: List of generated token sequences

        Returns:
            Novelty scores for each sequence
        """
        if len(self.history) == 0:
            # No history yet, maximum novelty
            return np.ones(len(generated_tokens)) * self.novelty_weight

        novelty_scores = []
        for tokens in generated_tokens:
            token_tuple = tuple(tokens)

            if self.distance_metric == 'edit':
                score = self._edit_distance_novelty(token_tuple)
            elif self.distance_metric == 'jaccard':
                score = self._jaccard_novelty(token_tuple)
            elif self.distance_metric == 'unique_tokens':
                score = self._unique_token_novelty(token_tuple)
            else:
                score = self._edit_distance_novelty(token_tuple)

            novelty_scores.append(score)

        return np.array(novelty_scores) * self.novelty_weight

    def _edit_distance_novelty(self, tokens: tuple) -> float:
        """Compute novelty based on edit distance to nearest neighbors."""
        if len(self.history) == 0:
            return 1.0

        # Sample from history if too large
        history_sample = list(self.history)
        if len(history_sample) > 100:
            indices = np.random.choice(len(history_sample), 100, replace=False)
            history_sample = [history_sample[i] for i in indices]

        # Compute distances to history
        distances = []
        for hist_tokens in history_sample:
            dist = self._levenshtein_distance(tokens, hist_tokens)
            # Normalize by max length
            norm_dist = dist / max(len(tokens), len(hist_tokens), 1)
            distances.append(norm_dist)

        # Average of k-nearest distances (lower = less novel)
        distances.sort()
        k_distances = distances[:min(self.k_nearest, len(distances))]
        return np.mean(k_distances)

    def _jaccard_novelty(self, tokens: tuple) -> float:
        """Compute novelty based on Jaccard distance to history."""
        if len(self.history) == 0:
            return 1.0

        token_set = set(tokens)
        similarities = []

        for hist_tokens in list(self.history)[-100:]:  # Use recent history
            hist_set = set(hist_tokens)
            intersection = len(token_set & hist_set)
            union = len(token_set | hist_set)
            if union > 0:
                jaccard = intersection / union
                similarities.append(jaccard)

        if not similarities:
            return 1.0

        # Novelty = 1 - average similarity
        return 1.0 - np.mean(similarities)

    def _unique_token_novelty(self, tokens: tuple) -> float:
        """Compute novelty based on unique tokens."""
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        return unique_ratio

    @staticmethod
    def _levenshtein_distance(s1: tuple, s2: tuple) -> int:
        """Compute Levenshtein edit distance between two sequences."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def add_to_history(self, generated_tokens: List[List[int]]):
        """Add generations to history buffer."""
        for tokens in generated_tokens:
            self.history.append(tuple(tokens))

    def get_state(self) -> Dict:
        """Get state for checkpointing."""
        return {'history': [list(t) for t in self.history]}

    def load_state(self, state: Dict):
        """Load state from checkpoint."""
        self.history = deque([tuple(t) for t in state['history']], maxlen=self.buffer_size)


class UncertaintyGuidedExploration:
    """
    Uncertainty-guided exploration based on reward variance.

    High reward variance suggests the model is uncertain about certain
    regions of the output space. We increase exploration (entropy) in
    these uncertain regions.

    Strategy:
    - Track reward variance over recent batches
    - Higher variance -> more exploration needed
    - Adjusts entropy weight based on uncertainty
    """

    def __init__(
        self,
        window_size: int = 10,
        base_weight: float = 0.2,
        variance_weight: float = 0.5,
        variance_threshold: float = 0.1,
        max_boost: float = 2.0,
    ):
        self.window_size = window_size
        self.base_weight = base_weight
        self.variance_weight = variance_weight
        self.variance_threshold = variance_threshold
        self.max_boost = max_boost

        # Reward tracking
        self.reward_history: Deque[float] = deque(maxlen=window_size)
        self.variance_history: Deque[float] = deque(maxlen=window_size)

    def update(self, rewards: np.ndarray) -> float:
        """
        Update with batch rewards and compute exploration weight.

        Args:
            rewards: Array of rewards from current batch

        Returns:
            Exploration weight multiplier
        """
        # Compute batch statistics
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)

        self.reward_history.append(batch_mean)
        self.variance_history.append(batch_var)

        return self.get_exploration_weight()

    def get_exploration_weight(self) -> float:
        """
        Get current exploration weight based on uncertainty.

        Returns:
            Weight multiplier for entropy bonus
        """
        if len(self.variance_history) < 3:
            return self.base_weight

        # Average recent variance
        avg_variance = np.mean(list(self.variance_history))

        # Compute boost based on variance
        if avg_variance > self.variance_threshold:
            # High variance: boost exploration
            variance_ratio = avg_variance / self.variance_threshold
            boost = min(self.variance_weight * variance_ratio, self.max_boost)
            return self.base_weight * (1 + boost)
        else:
            return self.base_weight

    def get_state(self) -> Dict:
        """Get state for checkpointing."""
        return {
            'reward_history': list(self.reward_history),
            'variance_history': list(self.variance_history),
        }

    def load_state(self, state: Dict):
        """Load state from checkpoint."""
        self.reward_history = deque(state['reward_history'], maxlen=self.window_size)
        self.variance_history = deque(state['variance_history'], maxlen=self.window_size)


class EntropyManager:
    """
    Unified entropy maintenance manager.

    Combines multiple strategies for maintaining healthy exploration
    during REINFORCE training. Provides a single interface for:
    - Getting entropy weights
    - Getting sampling temperatures
    - Computing novelty bonuses
    - Tracking and updating all strategies

    Usage:
        manager = EntropyManager(strategy='adaptive')

        for epoch in range(n_epochs):
            entropy_weight = manager.get_entropy_weight(epoch, current_entropy)
            temperature = manager.get_temperature(epoch, current_entropy)

            # ... training ...

            manager.update(
                epoch=epoch,
                entropy=current_entropy,
                reward=mean_reward,
                exact_match=exact_match,
                generated_tokens=batch_tokens,
                position_errors=errors,
                position_mask=mask,
            )
    """

    def __init__(
        self,
        strategy: str = 'adaptive',
        config: Optional[EntropyConfig] = None,
        max_len: int = 60,
    ):
        """
        Initialize entropy manager.

        Args:
            strategy: One of 'constant', 'adaptive', 'cyclical', 'position_weighted',
                     'novelty_bonus', 'uncertainty', 'composite'
            config: Optional configuration (uses defaults if None)
            max_len: Maximum sequence length for position-based weighting
        """
        self.config = config or EntropyConfig(strategy=strategy)
        self.strategy = EntropyStrategy(self.config.strategy)
        self.max_len = max_len

        # Initialize component schedulers based on strategy
        self._init_schedulers()

    def _init_schedulers(self):
        """Initialize component schedulers."""
        cfg = self.config

        # Adaptive entropy scheduler
        self.adaptive_scheduler = AdaptiveEntropyScheduler(
            target_entropy=cfg.target_entropy,
            min_entropy=cfg.min_entropy,
            base_weight=cfg.entropy_weight_base,
            min_weight=cfg.entropy_weight_min,
            max_weight=cfg.entropy_weight_max,
            adaptation_rate=cfg.adaptation_rate,
            plateau_window=cfg.plateau_window,
            plateau_threshold=cfg.plateau_threshold,
            plateau_relative=cfg.plateau_relative,
        )

        # Temperature scheduler
        self.temperature_scheduler = TemperatureWarmRestartScheduler(
            base_temperature=cfg.temperature_base,
            min_temperature=cfg.temperature_min,
            max_temperature=cfg.temperature_max,
            restart_period=cfg.temperature_restart_period,
            restart_boost=cfg.temperature_restart_boost,
        )

        # Per-position weighter
        self.position_weighter = PerPositionEntropyWeighter(
            max_len=self.max_len,
            base_weight=1.0,
            error_boost=cfg.error_position_boost,
            decay=cfg.position_decay,
        )

        # Novelty bonus
        self.novelty_bonus = NoveltyBonus(
            buffer_size=cfg.novelty_buffer_size,
            novelty_weight=cfg.novelty_weight,
            distance_metric=cfg.novelty_distance_metric,
        )

        # Uncertainty-guided exploration
        self.uncertainty_explorer = UncertaintyGuidedExploration(
            window_size=cfg.uncertainty_window,
            base_weight=cfg.entropy_weight_base,
            variance_weight=cfg.uncertainty_weight,
            variance_threshold=cfg.variance_threshold,
        )

        # Causal entropy scheduler (diagnoses plateau cause before boosting)
        self.causal_scheduler = CausalEntropyScheduler(
            base_weight=cfg.entropy_weight_base,
            min_weight=cfg.entropy_weight_min,
            max_weight=cfg.entropy_weight_max,
            min_entropy=cfg.min_entropy,
            diagnosis_window=cfg.causal_diagnosis_window,
            followup_window=cfg.causal_followup_window,
            plateau_window=cfg.plateau_window,
            plateau_threshold=cfg.plateau_threshold,
            plateau_relative=cfg.plateau_relative,
            entropy_drop_threshold=cfg.causal_entropy_drop_threshold,
            min_success_rate=cfg.causal_min_success_rate,
            strong_boost=cfg.causal_strong_boost,
            weak_boost=cfg.causal_weak_boost,
            minimal_boost=cfg.causal_minimal_boost,
        )

    def get_entropy_weight(
        self,
        epoch: int,
        current_entropy: Optional[float] = None,
    ) -> float:
        """
        Get entropy bonus weight for current state.

        Args:
            epoch: Current training epoch
            current_entropy: Current policy entropy (optional but recommended)

        Returns:
            Entropy weight to use in loss computation
        """
        if self.strategy == EntropyStrategy.CONSTANT:
            return self.config.entropy_weight_base

        elif self.strategy == EntropyStrategy.ADAPTIVE:
            return self.adaptive_scheduler.get_weight()

        elif self.strategy == EntropyStrategy.CYCLICAL:
            # Use temperature-based implicit entropy boost
            return self.config.entropy_weight_base

        elif self.strategy == EntropyStrategy.UNCERTAINTY_GUIDED:
            return self.uncertainty_explorer.get_exploration_weight()

        elif self.strategy == EntropyStrategy.COMPOSITE:
            # Combine adaptive and uncertainty
            adaptive_weight = self.adaptive_scheduler.get_weight()
            uncertainty_weight = self.uncertainty_explorer.get_exploration_weight()
            return (adaptive_weight + uncertainty_weight) / 2

        elif self.strategy == EntropyStrategy.CAUSAL:
            # Causal scheduler returns weight directly from update()
            # If not updated yet, return base weight
            return self.causal_scheduler.get_weight()

        else:
            return self.config.entropy_weight_base

    def get_temperature(
        self,
        epoch: int,
        current_entropy: Optional[float] = None,
    ) -> float:
        """
        Get sampling temperature for current state.

        Args:
            epoch: Current training epoch
            current_entropy: Current policy entropy

        Returns:
            Temperature for sampling
        """
        if self.strategy in [EntropyStrategy.CYCLICAL, EntropyStrategy.COMPOSITE]:
            return self.temperature_scheduler.get_temperature(epoch, current_entropy)
        else:
            # Adaptive temperature based on entropy
            if current_entropy is not None and current_entropy < self.config.min_entropy:
                # Boost temperature when entropy is too low
                boost = (self.config.min_entropy - current_entropy) / self.config.min_entropy
                return self.config.temperature_base + boost * 0.2
            return self.config.temperature_base

    def get_position_weights(self) -> np.ndarray:
        """Get entropy weights per sequence position."""
        if self.strategy in [EntropyStrategy.POSITION_WEIGHTED, EntropyStrategy.COMPOSITE]:
            return self.position_weighter.get_weights()
        else:
            return np.ones(self.max_len)

    def compute_novelty_bonus(self, generated_tokens: List[List[int]]) -> np.ndarray:
        """
        Compute novelty bonus for generated sequences.

        Args:
            generated_tokens: List of generated token sequences

        Returns:
            Novelty bonus for each sequence
        """
        if self.strategy in [EntropyStrategy.NOVELTY_BONUS, EntropyStrategy.COMPOSITE]:
            return self.novelty_bonus.compute_novelty(generated_tokens)
        else:
            return np.zeros(len(generated_tokens))

    def update(
        self,
        epoch: int,
        entropy: Optional[float] = None,
        reward: Optional[float] = None,
        exact_match: Optional[float] = None,
        rewards_batch: Optional[np.ndarray] = None,
        generated_tokens: Optional[List[List[int]]] = None,
        position_errors: Optional[np.ndarray] = None,
        position_mask: Optional[np.ndarray] = None,
    ):
        """
        Update all schedulers with current training metrics.

        Args:
            epoch: Current epoch
            entropy: Current mean entropy
            reward: Current mean reward
            exact_match: Current exact match rate
            rewards_batch: Array of rewards (for variance estimation)
            generated_tokens: List of generated sequences (for novelty)
            position_errors: Binary error array per position (for position weighting)
            position_mask: Valid positions mask
        """
        # Update adaptive scheduler
        if entropy is not None:
            self.adaptive_scheduler.update(
                entropy=entropy,
                reward=reward,
                exact_match=exact_match,
            )

        # Update causal scheduler (needs epoch, entropy, and exact_match)
        if self.strategy == EntropyStrategy.CAUSAL and entropy is not None and exact_match is not None:
            self.causal_scheduler.update(
                epoch=epoch,
                entropy=entropy,
                exact_match=exact_match,
                reward=reward,
            )

        # Update uncertainty explorer
        if rewards_batch is not None:
            self.uncertainty_explorer.update(rewards_batch)

        # Update novelty history
        if generated_tokens is not None:
            self.novelty_bonus.add_to_history(generated_tokens)

        # Update position weighter
        if position_errors is not None and position_mask is not None:
            self.position_weighter.update(position_errors, position_mask)

    def get_info(self, epoch: int, current_entropy: Optional[float] = None) -> Dict:
        """
        Get detailed info about current entropy maintenance state.

        Args:
            epoch: Current epoch
            current_entropy: Current entropy

        Returns:
            Dictionary with detailed state info
        """
        info = {
            'strategy': self.strategy.value,
            'epoch': epoch,
            'entropy_weight': self.get_entropy_weight(epoch, current_entropy),
            'temperature': self.get_temperature(epoch, current_entropy),
            'current_entropy': current_entropy,
            'target_entropy': self.config.target_entropy,
            'min_entropy': self.config.min_entropy,
        }

        # Add adaptive scheduler info
        if self.strategy in [EntropyStrategy.ADAPTIVE, EntropyStrategy.COMPOSITE]:
            info['adaptive_weight'] = self.adaptive_scheduler.get_weight()
            info['plateau_detected'] = self.adaptive_scheduler._plateau_detected
            info['plateau_epochs'] = self.adaptive_scheduler._plateau_epochs

        # Add uncertainty info
        if self.strategy in [EntropyStrategy.UNCERTAINTY_GUIDED, EntropyStrategy.COMPOSITE]:
            if len(self.uncertainty_explorer.variance_history) > 0:
                info['reward_variance'] = np.mean(list(self.uncertainty_explorer.variance_history))

        # Add causal scheduler info
        if self.strategy == EntropyStrategy.CAUSAL:
            causal_info = self.causal_scheduler.get_info()
            info['causal_state'] = causal_info['state']
            info['boost_multiplier'] = causal_info['boost_multiplier']
            info['total_interventions'] = causal_info['total_interventions']
            info['intervention_success_rate'] = causal_info['success_rate']
            info['last_diagnosis'] = causal_info['last_diagnosis']

        return info

    def get_state(self) -> Dict:
        """Get full state for checkpointing."""
        return {
            'strategy': self.strategy.value,
            'config': vars(self.config),
            'adaptive_scheduler': self.adaptive_scheduler.get_state(),
            'temperature_scheduler': self.temperature_scheduler.get_state(),
            'position_weighter': self.position_weighter.get_state(),
            'novelty_bonus': self.novelty_bonus.get_state(),
            'uncertainty_explorer': self.uncertainty_explorer.get_state(),
            'causal_scheduler': self.causal_scheduler.get_state(),
        }

    def load_state(self, state: Dict):
        """Load full state from checkpoint."""
        self.adaptive_scheduler.load_state(state['adaptive_scheduler'])
        self.temperature_scheduler.load_state(state['temperature_scheduler'])
        self.position_weighter.load_state(state['position_weighter'])
        self.novelty_bonus.load_state(state['novelty_bonus'])
        self.uncertainty_explorer.load_state(state['uncertainty_explorer'])
        if 'causal_scheduler' in state:
            self.causal_scheduler.load_state(state['causal_scheduler'])


def create_entropy_manager(
    strategy: str = 'adaptive',
    max_len: int = 60,
    **kwargs
) -> EntropyManager:
    """
    Factory function to create an EntropyManager.

    Args:
        strategy: Strategy name ('constant', 'adaptive', 'cyclical',
                  'position_weighted', 'novelty_bonus', 'uncertainty', 'composite',
                  'causal'). The 'causal' strategy diagnoses plateau causes before
                  boosting entropy - it checks if entropy dropped before plateau
                  and uses tiered response based on evidence strength.
        max_len: Maximum sequence length
        **kwargs: Override config parameters

    Returns:
        Configured EntropyManager
    """
    config = EntropyConfig(strategy=strategy, **kwargs)
    return EntropyManager(strategy=strategy, config=config, max_len=max_len)


# Convenience functions for quick access

def get_adaptive_entropy_weight(
    current_entropy: float,
    target_entropy: float = 0.5,
    min_entropy: float = 0.1,
    base_weight: float = 0.2,
) -> float:
    """
    Quick computation of adaptive entropy weight.

    Args:
        current_entropy: Current policy entropy
        target_entropy: Target entropy level
        min_entropy: Minimum acceptable entropy
        base_weight: Base entropy weight

    Returns:
        Adjusted entropy weight
    """
    if current_entropy < min_entropy:
        # Critical: strong boost
        boost = 2.0 * (min_entropy - current_entropy) / min_entropy
        return min(base_weight * (1 + boost), 1.0)
    elif current_entropy < target_entropy:
        # Below target: moderate boost
        gap = (target_entropy - current_entropy) / target_entropy
        return base_weight * (1 + gap)
    else:
        return base_weight


def compute_position_weighted_entropy_loss(
    entropy_per_position: 'torch.Tensor',
    position_weights: np.ndarray,
    mask: 'torch.Tensor',
) -> 'torch.Tensor':
    """
    Compute position-weighted entropy loss.

    Args:
        entropy_per_position: Entropy at each position (batch, seq_len)
        position_weights: Weights per position (seq_len,)
        mask: Valid positions mask (batch, seq_len)

    Returns:
        Weighted mean entropy loss
    """
    import torch

    # Convert weights to tensor
    weights = torch.tensor(position_weights, device=entropy_per_position.device, dtype=entropy_per_position.dtype)

    # Truncate or pad weights to match sequence length
    seq_len = entropy_per_position.size(1)
    if len(weights) < seq_len:
        weights = torch.cat([weights, torch.ones(seq_len - len(weights), device=weights.device)])
    weights = weights[:seq_len]

    # Apply weights and mask
    weighted_entropy = entropy_per_position * weights.unsqueeze(0) * mask
    total_weight = (mask * weights.unsqueeze(0)).sum()

    if total_weight > 0:
        return -weighted_entropy.sum() / total_weight  # Negative because we maximize entropy
    else:
        return torch.tensor(0.0, device=entropy_per_position.device)
