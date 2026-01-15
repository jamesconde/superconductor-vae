"""
V8 Enhanced Reward Function with Targeted Penalties.

Key Design Principles:
1. HARSH penalties for element errors (model must NEVER confuse elements)
2. MODERATE penalties for structural errors (wrong counts, order)
3. MILD penalties for digit-level errors in fractions
4. SCALED rewards (0-10) for stronger gradients

December 2025
"""

import torch
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Token indices (imported from decoder)
PAD_IDX = 0
START_IDX = 1
END_IDX = 2


@dataclass
class RewardConfigV8:
    """Configuration for V8 reward function."""
    # Exact match bonus
    exact_match: float = 10.0
    near_exact_1_token: float = 9.5
    near_exact_2_tokens: float = 9.0
    near_exact_3_tokens: float = 8.5

    # Element penalties (HARSH - these are fundamental errors)
    wrong_element: float = -10.0      # Predicted wrong element
    missing_element: float = -8.0     # Dropped an element
    extra_element: float = -8.0       # Hallucinated an element
    wrong_element_order: float = -5.0 # Elements in wrong order

    # Fraction penalties (MODERATE)
    missing_fraction: float = -3.0    # Expected fraction, got none
    extra_fraction: float = -3.0      # Hallucinated fraction
    completely_wrong_fraction: float = -2.0  # Different stoichiometry

    # Digit-level penalties (MILD)
    one_digit_off: float = -0.3
    two_digits_off: float = -1.0
    three_plus_digits_off: float = -1.5

    # Structural penalties
    unbalanced_parens: float = -2.0
    wrong_subscript: float = -0.5

    # Bonuses for partial correctness
    all_elements_correct_bonus: float = 5.0
    all_fractions_correct_bonus: float = 3.0
    correct_length_bonus: float = 1.0


# Element symbols for parsing
ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
}


def tokens_to_string(tokens: torch.Tensor, idx_to_token: dict) -> str:
    """Convert token indices to string."""
    result = []
    for idx in tokens:
        idx = idx.item() if isinstance(idx, torch.Tensor) else idx
        if idx in [PAD_IDX, START_IDX, END_IDX]:
            continue
        token = idx_to_token.get(idx, '')
        if token:
            result.append(token)
    return ''.join(result)


def parse_formula_components(formula: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse formula into elements, fractions, and subscripts.

    Args:
        formula: Chemical formula string like "La(7/10)Sr(3/10)CuO4"

    Returns:
        elements: List of element symbols in order ['La', 'Sr', 'Cu', 'O']
        fractions: List of fraction strings ['(7/10)', '(3/10)']
        subscripts: List of subscript numbers ['4']
    """
    elements = []
    fractions = []
    subscripts = []

    i = 0
    while i < len(formula):
        # Check for fraction: (num/denom)
        if formula[i] == '(':
            # Find matching closing paren
            j = i + 1
            depth = 1
            while j < len(formula) and depth > 0:
                if formula[j] == '(':
                    depth += 1
                elif formula[j] == ')':
                    depth -= 1
                j += 1
            fraction = formula[i:j]
            if '/' in fraction:
                fractions.append(fraction)
            i = j
            continue

        # Check for two-letter element
        if i + 1 < len(formula):
            two_letter = formula[i:i+2]
            if two_letter in ELEMENTS:
                elements.append(two_letter)
                i += 2
                continue

        # Check for one-letter element
        if formula[i] in ELEMENTS:
            elements.append(formula[i])
            i += 1
            continue

        # Check for digit (subscript)
        if formula[i].isdigit():
            # Collect all consecutive digits
            j = i
            while j < len(formula) and formula[j].isdigit():
                j += 1
            subscripts.append(formula[i:j])
            i = j
            continue

        # Skip other characters
        i += 1

    return elements, fractions, subscripts


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

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


def compute_reward_v8(
    sampled_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    idx_to_token: dict,
    mask: Optional[torch.Tensor] = None,
    config: Optional[RewardConfigV8] = None
) -> torch.Tensor:
    """
    Compute V8 reward with targeted penalties.

    This reward function:
    1. Gives maximum reward (10.0) for exact match
    2. Harshly penalizes element errors (-10.0 per wrong element)
    3. Moderately penalizes structural errors
    4. Mildly penalizes digit-level fraction errors

    Args:
        sampled_tokens: [batch, seq_len] sampled token indices
        target_tokens: [batch, seq_len] target token indices
        idx_to_token: Dictionary mapping indices to token strings
        mask: [batch, seq_len] valid positions
        config: Reward configuration

    Returns:
        rewards: [batch] reward for each sequence
    """
    if config is None:
        config = RewardConfigV8()

    if mask is None:
        mask = target_tokens != PAD_IDX

    batch_size = sampled_tokens.shape[0]
    device = sampled_tokens.device
    rewards = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        # Extract valid tokens
        valid_mask = mask[i]
        sampled = sampled_tokens[i][valid_mask]
        target = target_tokens[i][valid_mask]

        # Convert to strings
        sampled_str = tokens_to_string(sampled, idx_to_token)
        target_str = tokens_to_string(target, idx_to_token)

        # =============================================
        # EXACT MATCH CHECK (maximum reward)
        # =============================================
        if sampled_str == target_str:
            rewards[i] = config.exact_match
            continue

        # =============================================
        # NEAR-EXACT CHECK (token-level edit distance)
        # =============================================
        sampled_list = sampled.tolist()
        target_list = target.tolist()

        # Count token differences
        token_diff = sum(1 for s, t in zip(sampled_list, target_list) if s != t)
        token_diff += abs(len(sampled_list) - len(target_list))

        if token_diff == 1:
            rewards[i] = config.near_exact_1_token
            continue
        elif token_diff == 2:
            rewards[i] = config.near_exact_2_tokens
            continue
        elif token_diff == 3:
            rewards[i] = config.near_exact_3_tokens
            continue

        # =============================================
        # PARSE INTO COMPONENTS
        # =============================================
        sampled_elems, sampled_fracs, sampled_subs = parse_formula_components(sampled_str)
        target_elems, target_fracs, target_subs = parse_formula_components(target_str)

        reward = 0.0

        # =============================================
        # ELEMENT ANALYSIS (HARSH PENALTIES)
        # =============================================
        n_sampled_elems = len(sampled_elems)
        n_target_elems = len(target_elems)

        # Check for missing elements
        if n_sampled_elems < n_target_elems:
            reward += config.missing_element * (n_target_elems - n_sampled_elems)

        # Check for extra elements
        if n_sampled_elems > n_target_elems:
            reward += config.extra_element * (n_sampled_elems - n_target_elems)

        # Check element correctness (in order)
        elements_correct = 0
        elements_wrong = 0
        order_errors = 0

        min_elems = min(n_sampled_elems, n_target_elems)
        for j in range(min_elems):
            if sampled_elems[j] == target_elems[j]:
                elements_correct += 1
            elif sampled_elems[j] in target_elems:
                # Element exists but in wrong position
                order_errors += 1
            else:
                # Completely wrong element
                elements_wrong += 1

        reward += config.wrong_element * elements_wrong
        reward += config.wrong_element_order * order_errors

        # Bonus if all elements correct
        all_elements_correct = (elements_correct == n_target_elems and
                                n_sampled_elems == n_target_elems)
        if all_elements_correct:
            reward += config.all_elements_correct_bonus

        # =============================================
        # FRACTION ANALYSIS (MODERATE PENALTIES)
        # =============================================
        n_sampled_fracs = len(sampled_fracs)
        n_target_fracs = len(target_fracs)

        # Check for missing/extra fractions
        if n_sampled_fracs < n_target_fracs:
            reward += config.missing_fraction * (n_target_fracs - n_sampled_fracs)
        if n_sampled_fracs > n_target_fracs:
            reward += config.extra_fraction * (n_sampled_fracs - n_target_fracs)

        # Compare fractions digit-by-digit
        fractions_correct = 0
        min_fracs = min(n_sampled_fracs, n_target_fracs)

        for j in range(min_fracs):
            s_frac = sampled_fracs[j]
            t_frac = target_fracs[j]

            if s_frac == t_frac:
                fractions_correct += 1
            else:
                # Compute character-level edit distance
                edit_dist = levenshtein_distance(s_frac, t_frac)

                if edit_dist == 1:
                    reward += config.one_digit_off
                elif edit_dist == 2:
                    reward += config.two_digits_off
                else:
                    reward += config.three_plus_digits_off

        # Bonus if all fractions correct
        all_fractions_correct = (fractions_correct == n_target_fracs and
                                  n_sampled_fracs == n_target_fracs)
        if all_fractions_correct:
            reward += config.all_fractions_correct_bonus

        # =============================================
        # SUBSCRIPT ANALYSIS (MILD PENALTIES)
        # =============================================
        for j in range(min(len(sampled_subs), len(target_subs))):
            if sampled_subs[j] != target_subs[j]:
                reward += config.wrong_subscript

        # =============================================
        # LENGTH BONUS
        # =============================================
        if len(sampled_str) == len(target_str):
            reward += config.correct_length_bonus

        # =============================================
        # STRUCTURAL CHECK (parentheses balance)
        # =============================================
        open_parens = sampled_str.count('(')
        close_parens = sampled_str.count(')')
        if open_parens != close_parens:
            reward += config.unbalanced_parens

        rewards[i] = reward

    return rewards


def compute_reward_v8_batch(
    sampled_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    idx_to_token: dict,
    mask: Optional[torch.Tensor] = None,
    config: Optional[RewardConfigV8] = None
) -> torch.Tensor:
    """
    Batch-friendly wrapper for V8 reward computation.

    This is the main entry point for the training loop.
    """
    return compute_reward_v8(sampled_tokens, target_tokens, idx_to_token, mask, config)


# Convenience function to create default config
def get_default_reward_config() -> RewardConfigV8:
    """Get default V8 reward configuration."""
    return RewardConfigV8()


# Test function
def test_reward_v8():
    """Test the V8 reward function with examples."""
    from superconductor.models.autoregressive_decoder import (
        tokenize_formula, tokens_to_indices, IDX_TO_TOKEN
    )

    test_cases = [
        # (sampled, target, expected_behavior)
        ("La(7/10)Sr(3/10)CuO4", "La(7/10)Sr(3/10)CuO4", "exact match = 10.0"),
        ("La(7/10)Sr(3/10)CuO4", "La(7/10)Sr(3/10)CuO5", "1 token off = 9.5"),
        ("Ba(7/10)Sr(3/10)CuO4", "La(7/10)Sr(3/10)CuO4", "wrong element = -10.0"),
        ("La(7/10)CuO4", "La(7/10)Sr(3/10)CuO4", "missing element = -8.0"),
        ("La(8/10)Sr(3/10)CuO4", "La(7/10)Sr(3/10)CuO4", "1 digit off = ~9.0"),
    ]

    print("=" * 60)
    print("V8 REWARD FUNCTION TEST")
    print("=" * 60)

    config = RewardConfigV8()

    for sampled_str, target_str, expected in test_cases:
        # Tokenize
        sampled_tokens = tokens_to_indices(tokenize_formula(sampled_str), max_len=80)
        target_tokens = tokens_to_indices(tokenize_formula(target_str), max_len=80)

        # Compute reward
        reward = compute_reward_v8(
            sampled_tokens.unsqueeze(0),
            target_tokens.unsqueeze(0),
            IDX_TO_TOKEN,
            config=config
        )

        print(f"\nSampled:  '{sampled_str}'")
        print(f"Target:   '{target_str}'")
        print(f"Reward:   {reward.item():.2f}")
        print(f"Expected: {expected}")


if __name__ == "__main__":
    test_reward_v8()
