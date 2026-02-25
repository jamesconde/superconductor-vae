"""
Training utilities for superconductor formula generation.
"""

from .mastery_sampler import (
    MasteryTracker,
    MasteryAwareSampler,
    MasteryAwareDataLoader,
    MasteryStats,
    create_mastery_aware_training_components,
)

from .kl_annealing import (
    CyclicalKLScheduler,
    KLSchedulerConfig,
    KLLossWithAnnealing,
    compute_kl_loss,
    compute_kl_loss_with_free_bits,
)

from .soft_token_sampling import (
    SoftTokenScheduler,
    SoftTokenMixer,
    SoftTokenDecoder,
    create_soft_token_training_components,
)

from .entropy_maintenance import (
    EntropyManager,
    EntropyConfig,
    EntropyStrategy,
    AdaptiveEntropyScheduler,
    CausalEntropyScheduler,
    TemperatureWarmRestartScheduler,
    PerPositionEntropyWeighter,
    NoveltyBonus,
    UncertaintyGuidedExploration,
    create_entropy_manager,
    get_adaptive_entropy_weight,
    compute_position_weighted_entropy_loss,
)

from .self_supervised import (
    SelfSupervisedConfig,
    SelfSupervisedEpoch,
    ZSpaceSampler,
    CandidateFilter,
    Phase2LossComputer,
    NovelDiscoveryTracker,
)

from .coverage_tracker import CoverageTracker

__all__ = [
    # Mastery-aware sampling
    'MasteryTracker',
    'MasteryAwareSampler',
    'MasteryAwareDataLoader',
    'MasteryStats',
    'create_mastery_aware_training_components',
    # KL annealing
    'CyclicalKLScheduler',
    'KLSchedulerConfig',
    'KLLossWithAnnealing',
    'compute_kl_loss',
    'compute_kl_loss_with_free_bits',
    # Soft-token sampling
    'SoftTokenScheduler',
    'SoftTokenMixer',
    'SoftTokenDecoder',
    'create_soft_token_training_components',
    # Entropy maintenance
    'EntropyManager',
    'EntropyConfig',
    'EntropyStrategy',
    'AdaptiveEntropyScheduler',
    'CausalEntropyScheduler',
    'TemperatureWarmRestartScheduler',
    'PerPositionEntropyWeighter',
    'NoveltyBonus',
    'UncertaintyGuidedExploration',
    'create_entropy_manager',
    'get_adaptive_entropy_weight',
    'compute_position_weighted_entropy_loss',
    # Phase 2: Self-supervised training
    'SelfSupervisedConfig',
    'SelfSupervisedEpoch',
    'ZSpaceSampler',
    'CandidateFilter',
    'Phase2LossComputer',
    'NovelDiscoveryTracker',
    # Phase 2: Coverage tracking
    'CoverageTracker',
]
