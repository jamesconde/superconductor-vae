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
]
