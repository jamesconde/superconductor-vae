"""
V10 Discriminative Reward Function for Superconductor VAE.

Key changes from V8:
1. EXACT MATCH BONUS: 100 (was 10) - 10x stronger signal
2. REDUCED PARTIAL CREDIT: Remove generous bonuses that let imperfect samples score 9/10
3. STEEPER NEAR-EXACT PENALTIES: Make the drop from exact to near-exact much sharper

Problem with V8: Reward saturated at ~8.7 while exact match was only 27%
Solution: Make exact match dramatically more rewarding than "almost correct"

December 2025
"""

import torch
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RewardConfigV10:
    """V10 Discriminative reward configuration."""

    # EXACT MATCH - the only thing that really matters
    exact_match: float = 100.0          # Was 10.0 - now 10x stronger

    # Near-exact: steep dropoff (was 9.5, 9.0, 8.5 - too generous!)
    near_exact_1_token: float = 50.0    # 1 token wrong = half credit
    near_exact_2_tokens: float = 25.0   # 2 tokens wrong = quarter credit
    near_exact_3_tokens: float = 10.0   # 3 tokens wrong = minimal credit

    # Element errors - HARSH (unchanged from V8)
    wrong_element: float = -20.0        # Was -10.0
    missing_element: float = -15.0      # Was -8.0
    extra_element: float = -15.0        # Was -8.0
    wrong_element_order: float = -10.0  # Was -5.0

    # Fraction errors - MODERATE
    missing_fraction: float = -5.0      # Was -3.0
    extra_fraction: float = -5.0        # Was -3.0
    wrong_fraction: float = -3.0        # Was -2.0

    # Other errors
    wrong_subscript: float = -1.0       # Was -0.5
    unbalanced_parens: float = -5.0     # Was -2.0

    # REDUCED partial credit bonuses (the key fix!)
    # These were letting imperfect samples score 9/10
    all_elements_correct_bonus: float = 2.0   # Was 5.0
    all_fractions_correct_bonus: float = 1.0  # Was 3.0
    correct_length_bonus: float = 0.5         # Was 1.0

    # Max partial credit possible: 2.0 + 1.0 + 0.5 = 3.5 (was 9.0!)
    # This creates a HUGE gap between exact (100) and partial (max ~3.5)


def get_default_reward_config_v10() -> RewardConfigV10:
    """Get default V10 discriminative reward config."""
    return RewardConfigV10()


# Regex patterns for parsing
ELEMENT_PATTERN = re.compile(r'([A-Z][a-z]?)')
FRACTION_PATTERN = re.compile(r'\((\d+)/(\d+)\)')
SUBSCRIPT_PATTERN = re.compile(r'(?<=[A-Z]|[a-z]|\))(\d+)(?!\d*[/)])')


def parse_formula_components(formula: str) -> Tuple[List[str], List[str], List[str]]:
    """Parse formula into elements, fractions, and subscripts."""
    elements = ELEMENT_PATTERN.findall(formula)
    fractions = FRACTION_PATTERN.findall(formula)
    fractions = [f"({n}/{d})" for n, d in fractions]
    subscripts = SUBSCRIPT_PATTERN.findall(formula)
    return elements, fractions, subscripts


def compute_token_edit_distance(pred_tokens: List[int], target_tokens: List[int],
                                 pad_idx: int, end_idx: int) -> int:
    """Compute simple token-level edit distance (substitutions + length diff)."""
    # Filter out padding
    pred_clean = [t for t in pred_tokens if t != pad_idx and t != end_idx]
    target_clean = [t for t in target_tokens if t != pad_idx and t != end_idx]

    # Count substitutions in overlapping region
    min_len = min(len(pred_clean), len(target_clean))
    substitutions = sum(1 for i in range(min_len) if pred_clean[i] != target_clean[i])

    # Add length difference
    length_diff = abs(len(pred_clean) - len(target_clean))

    return substitutions + length_diff


class TargetCacheV10:
    """Cache for pre-parsed target formulas."""

    def __init__(self, formulas: List[str], idx_to_token: dict):
        self.formulas = formulas
        self.idx_to_token = idx_to_token
        self.components = {}

        for formula in formulas:
            if formula not in self.components:
                self.components[formula] = parse_formula_components(formula)

    def get_components(self, formula: str) -> Tuple[List[str], List[str], List[str]]:
        if formula in self.components:
            return self.components[formula]
        return parse_formula_components(formula)


def tokens_to_string(tokens, idx_to_token, pad_idx=0, start_idx=1, end_idx=2):
    """Convert token indices to string."""
    result = []
    for idx in tokens:
        idx = idx.item() if isinstance(idx, torch.Tensor) else idx
        if idx in [pad_idx, start_idx, end_idx]:
            continue
        token = idx_to_token.get(idx, '')
        if token:
            result.append(token)
    return ''.join(result)


def compute_reward_v10(
    sampled_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    idx_to_token: dict,
    mask: torch.Tensor,
    config: Optional[RewardConfigV10] = None,
    target_cache: Optional[TargetCacheV10] = None,
    pad_idx: int = 0,
    start_idx: int = 1,
    end_idx: int = 2,
) -> torch.Tensor:
    """
    Compute V10 discriminative rewards.

    Key difference from V8: HUGE gap between exact match (100) and partial credit (max ~3.5)
    This forces the model to optimize for exact match, not "almost correct".
    """
    if config is None:
        config = get_default_reward_config_v10()

    batch_size = sampled_tokens.shape[0]
    rewards = torch.zeros(batch_size, device=sampled_tokens.device)

    for i in range(batch_size):
        pred_str = tokens_to_string(sampled_tokens[i], idx_to_token, pad_idx, start_idx, end_idx)
        target_str = tokens_to_string(target_tokens[i], idx_to_token, pad_idx, start_idx, end_idx)

        # EXACT MATCH - the big prize
        if pred_str == target_str:
            rewards[i] = config.exact_match  # 100.0!
            continue

        # Near-exact check (token edit distance)
        pred_list = sampled_tokens[i].tolist()
        target_list = target_tokens[i].tolist()
        edit_dist = compute_token_edit_distance(pred_list, target_list, pad_idx, end_idx)

        if edit_dist == 1:
            rewards[i] = config.near_exact_1_token  # 50.0
            continue
        elif edit_dist == 2:
            rewards[i] = config.near_exact_2_tokens  # 25.0
            continue
        elif edit_dist == 3:
            rewards[i] = config.near_exact_3_tokens  # 10.0
            continue

        # More than 3 tokens wrong - compute partial credit (max ~3.5)
        reward = 0.0

        # Parse components
        if target_cache is not None:
            tgt_elems, tgt_fracs, tgt_subs = target_cache.get_components(target_str)
        else:
            tgt_elems, tgt_fracs, tgt_subs = parse_formula_components(target_str)

        pred_elems, pred_fracs, pred_subs = parse_formula_components(pred_str)

        # Element analysis
        elements_correct = (pred_elems == tgt_elems)
        if elements_correct:
            reward += config.all_elements_correct_bonus  # +2.0
        else:
            # Penalties for element errors
            pred_set = set(pred_elems)
            tgt_set = set(tgt_elems)

            missing = tgt_set - pred_set
            extra = pred_set - tgt_set

            reward += len(missing) * config.missing_element
            reward += len(extra) * config.extra_element

            if pred_set == tgt_set and pred_elems != tgt_elems:
                reward += config.wrong_element_order

        # Fraction analysis
        fractions_correct = (pred_fracs == tgt_fracs)
        if fractions_correct:
            reward += config.all_fractions_correct_bonus  # +1.0
        else:
            pred_frac_set = set(pred_fracs)
            tgt_frac_set = set(tgt_fracs)

            missing_fracs = len(tgt_frac_set - pred_frac_set)
            extra_fracs = len(pred_frac_set - tgt_frac_set)

            reward += missing_fracs * config.missing_fraction
            reward += extra_fracs * config.extra_fraction

            # Wrong fractions (present in both but different)
            common_count = len(pred_frac_set & tgt_frac_set)
            if not fractions_correct and common_count < len(tgt_fracs):
                wrong_count = len(tgt_fracs) - common_count
                reward += wrong_count * config.wrong_fraction

        # Length bonus
        if len(pred_str) == len(target_str):
            reward += config.correct_length_bonus  # +0.5

        # Subscript check
        if pred_subs != tgt_subs:
            reward += config.wrong_subscript * abs(len(pred_subs) - len(tgt_subs) + 1)

        # Parentheses balance check
        if pred_str.count('(') != pred_str.count(')'):
            reward += config.unbalanced_parens

        rewards[i] = reward

    return rewards
