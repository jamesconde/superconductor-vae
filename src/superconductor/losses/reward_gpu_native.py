"""
GPU-Native Reward Function for Superconductor VAE.

This reward computation is 100% tensor-based - NO string parsing.
Runs entirely on GPU for maximum speed.

Key insight: We don't need to parse chemical formulas to compute useful rewards.
Token-level comparison is sufficient for training and is orders of magnitude faster.

V12.1 UPDATE: Added fraction-aware penalties. Errors within fractions (numerator/
denominator digits) are penalized more heavily because they affect stoichiometry.

V13.0 UPDATE: When use_semantic_fractions=True, fraction-aware penalties are
disabled because fractions are single semantic tokens (no digit-by-digit errors
possible). A new fraction-value penalty replaces the old digit penalty — it
scales by the difference in float values between predicted and target fraction
tokens. Basic token-match and exact-match rewards are tokenizer-agnostic.

December 2025
"""

import torch
from dataclasses import dataclass
from typing import Optional

# V12 Token indices for fraction detection (from old vocabulary)
# These are ONLY used when use_semantic_fractions=False
LPAREN_IDX = 4      # '('
RPAREN_IDX = 5      # ')'
SLASH_IDX = 16      # '/'
DIGIT_START = 138   # '0'
DIGIT_END = 147     # '9'


@dataclass
class GPURewardConfig:
    """Configuration for GPU-native rewards."""

    # Exact match bonus (sequence-level)
    exact_match: float = 100.0

    # Near-exact bonuses (by number of wrong tokens)
    near_exact_1: float = 50.0   # 1 token wrong
    near_exact_2: float = 25.0   # 2 tokens wrong
    near_exact_3: float = 10.0   # 3 tokens wrong

    # Per-token rewards for partial credit
    token_correct: float = 1.0   # Reward per correct token
    token_penalty: float = -0.5  # Penalty per wrong token

    # Length penalty
    length_mismatch_penalty: float = -2.0  # Per token of length difference

    # FRACTION-AWARE PENALTIES (V12.2 - AGGRESSIVE for digit accuracy)
    # Analysis showed 90% of errors are digit errors in fractions
    fraction_digit_penalty: float = -10.0   # AGGRESSIVE: -3.0 → -10.0 (digit errors are #1 problem)
    fraction_structure_penalty: float = -5.0  # Penalty for missing/extra ( ) /

    # V12.5: SEMANTIC DIGIT PENALTY - scale by HOW WRONG the digit is
    # Predicting 3 instead of 8 should be penalized MORE than 7 instead of 8
    # penalty = base_penalty * (1 + semantic_scale * |pred_digit - target_digit| / 9)
    # With scale=2.0: 8→3 gets 2.1x penalty, 8→7 gets 1.2x penalty
    use_semantic_digit_penalty: bool = True
    semantic_digit_scale: float = 2.0  # Multiplier for digit difference scaling

    # V12.37: Length-only error reward (perfect formula prefix, just too long)
    # Addresses extra-append failure (12.75% of errors): model outputs correct formula
    # then appends extra tokens instead of stopping. Give high reward to encourage
    # learning to stop earlier.
    length_only_base_reward: float = 50.0   # Base reward for length-only errors
    length_only_per_extra: float = 5.0      # Penalty per extra token beyond END
    length_only_floor: float = 10.0         # Minimum reward for length-only errors


def get_default_gpu_reward_config() -> GPURewardConfig:
    """Get default GPU-native reward config."""
    return GPURewardConfig()


@torch.no_grad()
def detect_fraction_positions(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Detect which positions are inside fractions (between '(' and ')').

    Returns a boolean mask where True = position is inside a fraction.
    This is GPU-native using cumsum to track parenthesis depth.

    Args:
        tokens: [batch, seq_len] token indices
        mask: [batch, seq_len] valid position mask

    Returns:
        in_fraction: [batch, seq_len] boolean mask
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    mask = mask.bool()

    # Find opening and closing parentheses
    is_lparen = (tokens == LPAREN_IDX) & mask
    is_rparen = (tokens == RPAREN_IDX) & mask

    # Use cumsum to track depth: +1 at '(', -1 at ')'
    # Position is "in fraction" if cumsum > 0 (after '(' but before matching ')')
    depth_delta = is_lparen.long() - is_rparen.long()
    depth = depth_delta.cumsum(dim=1)

    # In fraction = depth > 0 (we're inside at least one parenthesis)
    # Also include the opening '(' itself but not the closing ')'
    in_fraction = (depth > 0) | is_lparen

    return in_fraction & mask


@torch.no_grad()
def detect_digit_positions(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Detect which positions are digits (0-9).

    Args:
        tokens: [batch, seq_len] token indices
        mask: [batch, seq_len] valid position mask

    Returns:
        is_digit: [batch, seq_len] boolean mask
    """
    mask = mask.bool()
    is_digit = (tokens >= DIGIT_START) & (tokens <= DIGIT_END) & mask
    return is_digit


@torch.no_grad()
def detect_fraction_structure(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Detect fraction structure tokens: ( ) /

    Args:
        tokens: [batch, seq_len] token indices
        mask: [batch, seq_len] valid position mask

    Returns:
        is_structure: [batch, seq_len] boolean mask
    """
    mask = mask.bool()
    is_lparen = tokens == LPAREN_IDX
    is_rparen = tokens == RPAREN_IDX
    is_slash = tokens == SLASH_IDX
    return (is_lparen | is_rparen | is_slash) & mask


@torch.no_grad()
def compute_semantic_digit_penalty(
    sampled_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor,
    in_fraction: torch.Tensor,
    base_penalty: float = -10.0,
    semantic_scale: float = 2.0,
) -> torch.Tensor:
    """
    V12.5: Compute semantic digit penalty that scales by digit difference.

    Instead of flat penalty per wrong digit, this penalizes based on HOW WRONG:
    - Predicting 3 instead of 8: |8-3|=5, scaled penalty = base * (1 + scale * 5/9)
    - Predicting 7 instead of 8: |8-7|=1, scaled penalty = base * (1 + scale * 1/9)

    This gives the decoder direct signal that "close" digit errors are less bad
    than "far" digit errors, mirroring the stoichiometry MSE loss concept.

    Args:
        sampled_tokens: [batch, seq_len] predicted tokens
        target_tokens: [batch, seq_len] ground truth tokens
        mask: [batch, seq_len] valid position mask
        in_fraction: [batch, seq_len] positions inside fractions
        base_penalty: Base penalty per digit error (default -10.0)
        semantic_scale: How much to scale by digit difference (default 2.0)

    Returns:
        total_penalty: [batch] total semantic digit penalty per sample
    """
    device = sampled_tokens.device
    mask = mask.bool()

    # Identify digit positions in target that are in fractions
    target_is_digit = (target_tokens >= DIGIT_START) & (target_tokens <= DIGIT_END)
    digit_in_fraction = target_is_digit & in_fraction & mask

    # Identify mismatches at digit-in-fraction positions
    mismatches = (sampled_tokens != target_tokens)
    digit_mismatches = mismatches & digit_in_fraction

    # Convert tokens to actual digit values (0-9)
    # Token indices: DIGIT_START=138 is '0', DIGIT_END=147 is '9'
    sampled_digit_values = (sampled_tokens - DIGIT_START).clamp(0, 9).float()
    target_digit_values = (target_tokens - DIGIT_START).clamp(0, 9).float()

    # Compute absolute digit difference at mismatch positions
    digit_diff = torch.abs(sampled_digit_values - target_digit_values)

    # Compute scaled penalty: base * (1 + scale * diff / 9)
    # diff=0: penalty = base * 1.0 (no difference, but this shouldn't happen at mismatches)
    # diff=5: penalty = base * (1 + scale * 5/9) = base * 2.1 with scale=2
    # diff=9: penalty = base * (1 + scale * 9/9) = base * 3.0 with scale=2
    penalty_scale = 1.0 + semantic_scale * digit_diff / 9.0

    # Apply penalty only at digit mismatch positions
    position_penalty = digit_mismatches.float() * base_penalty * penalty_scale

    # Sum penalty per sample
    total_penalty = position_penalty.sum(dim=1)

    return total_penalty


@torch.no_grad()
def compute_fraction_value_penalty(
    sampled_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor,
    fraction_token_start: int,
    fraction_values: torch.Tensor,
    base_penalty: float = -10.0,
    semantic_scale: float = 2.0,
) -> torch.Tensor:
    """V13.0: Compute penalty based on fraction VALUE difference.

    When the target is a fraction token and the prediction is wrong, scale
    penalty by how far the predicted value is from the target value.
    E.g., predicting FRAC:1/4 (0.25) when target is FRAC:3/4 (0.75) gets
    a larger penalty than predicting FRAC:17/20 (0.85) when target is FRAC:4/5 (0.80).

    Args:
        sampled_tokens: [batch, seq_len] predicted tokens
        target_tokens: [batch, seq_len] ground truth tokens
        mask: [batch, seq_len] valid position mask
        fraction_token_start: First fraction token index (143 for V13)
        fraction_values: [vocab_size] float value for each fraction token (0.0 for non-fraction)
        base_penalty: Base penalty per fraction error
        semantic_scale: How much to scale by value difference

    Returns:
        total_penalty: [batch] total fraction value penalty per sample
    """
    device = sampled_tokens.device
    mask = mask.bool()

    # Ensure fraction_values is on the right device
    if fraction_values.device != device:
        fraction_values = fraction_values.to(device)

    # Identify fraction positions in target
    target_is_frac = (target_tokens >= fraction_token_start) & mask
    mismatches = (sampled_tokens != target_tokens)
    frac_mismatches = mismatches & target_is_frac

    if not frac_mismatches.any():
        return torch.zeros(sampled_tokens.shape[0], device=device)

    # Look up float values for sampled and target fraction tokens
    # Clamp indices to valid range for the lookup table
    vocab_size = fraction_values.shape[0]
    sampled_clamped = sampled_tokens.clamp(0, vocab_size - 1)
    target_clamped = target_tokens.clamp(0, vocab_size - 1)

    sampled_vals = fraction_values[sampled_clamped]  # [batch, seq_len]
    target_vals = fraction_values[target_clamped]    # [batch, seq_len]

    # Compute value difference at mismatch positions
    val_diff = torch.abs(sampled_vals - target_vals)

    # Scale: base * (1 + scale * diff). diff is typically 0-20 for stoichiometry
    # Normalize by max reasonable diff (20) to keep scale similar to V12 digit penalty
    penalty_scale = 1.0 + semantic_scale * val_diff.clamp(max=20.0) / 20.0

    # Apply penalty only at fraction mismatch positions
    position_penalty = frac_mismatches.float() * base_penalty * penalty_scale

    return position_penalty.sum(dim=1)


@torch.no_grad()
def compute_reward_gpu_native(
    sampled_tokens: torch.Tensor,      # [batch, seq_len]
    target_tokens: torch.Tensor,       # [batch, seq_len]
    mask: torch.Tensor,                # [batch, seq_len] - True where valid
    config: Optional[GPURewardConfig] = None,
    pad_idx: int = 0,
    end_idx: int = 2,
    use_semantic_fractions: bool = False,
    fraction_token_start: int = 0,
    fraction_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute rewards using pure tensor operations on GPU.

    100x faster than string-parsing reward because:
    1. No CPU-GPU data transfer
    2. No Python string operations
    3. Fully batched/vectorized

    Args:
        sampled_tokens: Sampled token indices [batch, seq_len]
        target_tokens: Target token indices [batch, seq_len]
        mask: Boolean mask for valid positions [batch, seq_len]
        config: Reward configuration
        pad_idx: Padding token index
        end_idx: End token index
        use_semantic_fractions: V13.0 flag — disables V12 digit-level fraction
            penalties and enables fraction-value penalty instead
        fraction_token_start: First fraction token index (V13.0 only)
        fraction_values: [vocab_size] float values per token (V13.0 only)

    Returns:
        rewards: [batch] tensor of rewards
    """
    if config is None:
        config = get_default_gpu_reward_config()

    batch_size = sampled_tokens.shape[0]
    device = sampled_tokens.device
    mask = mask.bool()

    # 1. Token-level comparison (all on GPU)
    matches = (sampled_tokens == target_tokens) & mask  # [batch, seq_len]
    mismatches = (sampled_tokens != target_tokens) & mask

    n_matches = matches.sum(dim=1).float()      # [batch]
    n_mismatches = mismatches.sum(dim=1).float()  # [batch]
    n_valid = mask.sum(dim=1).float()           # [batch]

    # 2. Exact sequence match (highest reward)
    exact_match = (n_mismatches == 0)

    # 3. Near-exact matches (1-3 tokens wrong)
    near_exact_1 = (n_mismatches == 1)
    near_exact_2 = (n_mismatches == 2)
    near_exact_3 = (n_mismatches == 3)

    # 4. Compute length mismatch (find actual sequence lengths)
    # Find position of first END token or last valid token
    sampled_has_end = (sampled_tokens == end_idx) & mask
    target_has_end = (target_tokens == end_idx) & mask

    # Get sequence lengths (position of END token, or total valid length)
    sampled_end_pos = torch.where(
        sampled_has_end.any(dim=1),
        sampled_has_end.float().argmax(dim=1),
        mask.sum(dim=1).float()
    )
    target_end_pos = torch.where(
        target_has_end.any(dim=1),
        target_has_end.float().argmax(dim=1),
        mask.sum(dim=1).float()
    )
    length_diff = torch.abs(sampled_end_pos - target_end_pos)

    # 5. Compute rewards (all tensor ops)
    # ========== COMPUTE FRACTION PENALTY (for all non-exact) ==========
    if use_semantic_fractions and fraction_values is not None:
        # V13.0: Fractions are single tokens — no digit/structure penalties.
        # Instead, penalize by float value difference between predicted and target fractions.
        fraction_penalty = compute_fraction_value_penalty(
            sampled_tokens, target_tokens, mask,
            fraction_token_start=fraction_token_start,
            fraction_values=fraction_values,
            base_penalty=config.fraction_digit_penalty,
            semantic_scale=config.semantic_digit_scale,
        )
    else:
        # V12.x: Character-level fraction penalties (digit-by-digit)
        # Detect fraction regions in TARGET (ground truth structure)
        target_in_fraction = detect_fraction_positions(target_tokens, mask)
        target_is_structure = detect_fraction_structure(target_tokens, mask)

        # Count structural errors (missing/wrong parentheses or slashes)
        structure_errors = (mismatches & target_is_structure).sum(dim=1).float()

        # V12.5: Semantic digit penalty - scales by digit difference
        if config.use_semantic_digit_penalty:
            digit_penalty = compute_semantic_digit_penalty(
                sampled_tokens, target_tokens, mask, target_in_fraction,
                base_penalty=config.fraction_digit_penalty,
                semantic_scale=config.semantic_digit_scale,
            )
        else:
            target_is_digit = detect_digit_positions(target_tokens, mask)
            digit_in_fraction = target_is_digit & target_in_fraction
            digit_errors = (mismatches & digit_in_fraction).sum(dim=1).float()
            digit_penalty = digit_errors * config.fraction_digit_penalty

        fraction_penalty = digit_penalty + structure_errors * config.fraction_structure_penalty
    # ========== END FRACTION PENALTY COMPUTATION ==========

    rewards = torch.zeros(batch_size, device=device)

    # Exact match gets full bonus
    rewards = torch.where(exact_match,
                          torch.full_like(rewards, config.exact_match),
                          rewards)

    # V12.37: Length-only error detection — "perfect formula, just too long"
    # All tokens up to target END are correct, but sampled sequence continues past END.
    # This is the #1 non-exact error type (12.75% of dataset). Give high reward
    # to provide gradient signal to "just stop one token earlier."
    not_exact = ~exact_match
    target_end_col = target_end_pos.unsqueeze(1).long()  # [batch, 1]
    seq_len = sampled_tokens.size(1)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    before_target_end = positions < target_end_col  # [batch, seq_len]
    # Check if all tokens before target END match
    prefix_correct = ((sampled_tokens == target_tokens) | ~before_target_end | ~mask).all(dim=1)
    too_long = sampled_end_pos > target_end_pos
    length_only_error = prefix_correct & too_long & not_exact

    # Reward: high base minus mild penalty per extra token, with a floor
    extra_tokens = (sampled_end_pos - target_end_pos).clamp(min=0)
    length_only_reward = config.length_only_base_reward - extra_tokens * config.length_only_per_extra
    length_only_reward = length_only_reward.clamp(min=config.length_only_floor)

    rewards = torch.where(length_only_error, length_only_reward, rewards)

    # Near-exact bonuses (only if not exact and not length-only)
    # V12.5: Apply semantic fraction penalty to near-exact cases too!
    # V12.30: Apply length_mismatch_penalty to near-exact tiers (previously only applied to partial_credit).
    # This prevents RL from rewarding "perfect formula + extra token" with full near_exact bonus.
    not_handled = not_exact & ~length_only_error  # V12.37: Skip length-only errors (already rewarded)
    length_penalty = length_diff * config.length_mismatch_penalty
    rewards = torch.where(not_handled & near_exact_1,
                          torch.full_like(rewards, config.near_exact_1) + fraction_penalty + length_penalty,
                          rewards)
    rewards = torch.where(not_handled & near_exact_2,
                          torch.full_like(rewards, config.near_exact_2) + fraction_penalty + length_penalty,
                          rewards)
    rewards = torch.where(not_handled & near_exact_3,
                          torch.full_like(rewards, config.near_exact_3) + fraction_penalty + length_penalty,
                          rewards)

    # For samples with 4+ mismatches, use token-level partial credit
    partial_credit_mask = not_exact & ~length_only_error & (n_mismatches > 3)
    token_reward = n_matches * config.token_correct + n_mismatches * config.token_penalty
    token_reward = token_reward + length_diff * config.length_mismatch_penalty

    # V12.5: Apply already-computed fraction penalty to partial credit
    token_reward = token_reward + fraction_penalty
    token_reward = torch.clamp(token_reward, min=-100, max=5)  # Allow larger negative for semantic penalties

    rewards = torch.where(partial_credit_mask, token_reward, rewards)

    return rewards


@torch.no_grad()
def compute_reward_gpu_simple(
    sampled_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor,
    exact_match_bonus: float = 100.0,
    per_token_bonus: float = 2.0,
) -> torch.Tensor:
    """
    Simplified GPU reward - even faster.

    reward = exact_match_bonus if all tokens match
             else per_token_bonus * (num_correct_tokens)
    """
    mask = mask.bool()
    matches = (sampled_tokens == target_tokens) & mask
    n_matches = matches.sum(dim=1).float()
    n_mismatches = ((sampled_tokens != target_tokens) & mask).sum(dim=1)

    exact_match = (n_mismatches == 0)

    rewards = torch.where(
        exact_match,
        torch.full_like(n_matches, exact_match_bonus),
        n_matches * per_token_bonus
    )

    return rewards
