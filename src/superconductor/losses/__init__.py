"""
Loss functions for superconductor formula generation.

Includes:
- formula_loss: Token-weighted cross-entropy with accuracy tracking
- semantic_unit_loss: Semantic-level loss (elements, fractions, subscripts)
- reinforce_loss: REINFORCE-based sequence-level optimization with RLOO baseline
- consistency_losses: Self-consistency and bidirectional consistency losses
- theory_losses: BCS, cuprate, and other theory-based regularization
- z_supervision_loss: V12.31 Physics Z coordinate supervision
- constraint_rewards: V12.43 REINFORCE reward modifiers (A1, A2, A4, A7, B1-B8)
- round_trip_loss: V12.43 Round-trip cycle consistency (A5)
- constraint_zoo: V12.43 Differentiable physics constraints (A3, A6)
"""

from .formula_loss import (
    TokenType,
    get_token_type,
    build_token_type_mask,
    build_loss_weights,
    FormulaAccuracyMetrics,
    FormulaAccuracyTracker,
    WeightedFormulaLoss,
    FormulaLossWithAccuracy,
    format_accuracy_log,
)

from .semantic_unit_loss import (
    SemanticUnit,
    parse_tokens_to_semantic_units,
    compute_semantic_loss,
    SemanticUnitLoss,
)

from .reinforce_loss import (
    RewardConfig,
    compute_exact_match_reward,
    compute_semantic_reward,
    REINFORCELoss,
    MixedCEReinforce,
)

from .consistency_losses import (
    ConsistencyLossConfig,
    SelfConsistencyLoss,
    BidirectionalConsistencyLoss,
    CombinedConsistencyLoss,
    compute_consistency_reward,
)

from .theory_losses import (
    TheoryLossConfig,
    BCSTheoryLoss,
    CuprateTheoryLoss,
    IronBasedTheoryLoss,
    UnknownTheoryLoss,
    TheoryRegularizationLoss,
    compute_theory_regularization,
)

from .z_supervision_loss import (
    CompositionalSupervisionLoss,
    MagpieEncodingLoss,
    GLConsistencyLoss,
    BCSConsistencyLoss,
    CobordismConsistencyLoss,
    DimensionlessRatioConsistencyLoss,
    DirectSupervisionLoss,
    PhysicsZLoss,
)

# V12.43: SC Constraint Zoo
from .constraint_rewards import (
    ConstraintRewardConfig,
    FamilyConstraintConfig,
    compute_constraint_rewards,
    compute_duplicate_element_penalty,
    compute_gcd_canonicality_penalty,
    compute_stoich_normalization_penalty,
    compute_impossible_element_penalty,
    compute_family_constraint_rewards,
)

from .round_trip_loss import (
    RoundTripConsistencyLoss,
)

from .constraint_zoo import (
    SiteOccupancySumLoss,
    ChargeBalanceLoss,
)

__all__ = [
    # formula_loss
    'TokenType',
    'get_token_type',
    'build_token_type_mask',
    'build_loss_weights',
    'FormulaAccuracyMetrics',
    'FormulaAccuracyTracker',
    'WeightedFormulaLoss',
    'FormulaLossWithAccuracy',
    'format_accuracy_log',
    # semantic_unit_loss
    'SemanticUnit',
    'parse_tokens_to_semantic_units',
    'compute_semantic_loss',
    'SemanticUnitLoss',
    # reinforce_loss
    'RewardConfig',
    'compute_exact_match_reward',
    'compute_semantic_reward',
    'REINFORCELoss',
    'MixedCEReinforce',
    # consistency_losses
    'ConsistencyLossConfig',
    'SelfConsistencyLoss',
    'BidirectionalConsistencyLoss',
    'CombinedConsistencyLoss',
    'compute_consistency_reward',
    # theory_losses
    'TheoryLossConfig',
    'BCSTheoryLoss',
    'CuprateTheoryLoss',
    'IronBasedTheoryLoss',
    'UnknownTheoryLoss',
    'TheoryRegularizationLoss',
    'compute_theory_regularization',
    # z_supervision_loss (V12.31)
    'CompositionalSupervisionLoss',
    'MagpieEncodingLoss',
    'GLConsistencyLoss',
    'BCSConsistencyLoss',
    'CobordismConsistencyLoss',
    'DimensionlessRatioConsistencyLoss',
    'DirectSupervisionLoss',
    'PhysicsZLoss',
    # constraint_rewards (V12.43)
    'ConstraintRewardConfig',
    'FamilyConstraintConfig',
    'compute_constraint_rewards',
    'compute_duplicate_element_penalty',
    'compute_gcd_canonicality_penalty',
    'compute_stoich_normalization_penalty',
    'compute_impossible_element_penalty',
    'compute_family_constraint_rewards',
    # round_trip_loss (V12.43)
    'RoundTripConsistencyLoss',
    # constraint_zoo (V12.43)
    'SiteOccupancySumLoss',
    'ChargeBalanceLoss',
]
