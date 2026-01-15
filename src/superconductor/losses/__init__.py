"""
Loss functions for superconductor formula generation.

Includes:
- formula_loss: Token-weighted cross-entropy with accuracy tracking
- semantic_unit_loss: Semantic-level loss (elements, fractions, subscripts)
- reinforce_loss: REINFORCE-based sequence-level optimization with RLOO baseline
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
]
