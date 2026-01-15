"""
V8.1 Optimized Reward Function - FAST version.

Optimizations over V8:
1. NO Levenshtein - simple equality for fractions (much faster)
2. Caching support - pre-parse targets once, reuse
3. Vectorized operations where possible
4. Simplified logic for speed

December 2025
"""

import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# Token indices
PAD_IDX = 0
START_IDX = 1
END_IDX = 2


@dataclass
class RewardConfigV8Optimized:
    """Configuration for V8.1 optimized reward function."""
    # Exact match bonus (scaled 0-10)
    exact_match: float = 10.0
    near_exact_1_token: float = 9.5
    near_exact_2_tokens: float = 9.0
    near_exact_3_tokens: float = 8.5

    # Element penalties (HARSH)
    wrong_element: float = -10.0
    missing_element: float = -8.0
    extra_element: float = -8.0
    wrong_element_order: float = -5.0

    # Fraction penalties (MODERATE - simplified, no Levenshtein)
    missing_fraction: float = -3.0
    extra_fraction: float = -3.0
    wrong_fraction: float = -2.0  # Single penalty for any wrong fraction

    # Subscript penalties (MILD)
    wrong_subscript: float = -0.5

    # Structural penalties
    unbalanced_parens: float = -2.0

    # Bonuses
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
            j = i
            while j < len(formula) and formula[j].isdigit():
                j += 1
            subscripts.append(formula[i:j])
            i = j
            continue

        # Skip other characters
        i += 1

    return elements, fractions, subscripts


class TargetCache:
    """Cache for pre-parsed target formulas."""

    def __init__(self, formulas: List[str], idx_to_token: dict):
        """Pre-parse all target formulas once."""
        self.strings = {}  # token_tuple -> string
        self.components = {}  # token_tuple -> (elements, fractions, subscripts)
        self.idx_to_token = idx_to_token

        # Pre-parse unique formulas
        for formula in formulas:
            if formula not in self.components:
                self.components[formula] = parse_formula_components(formula)

    def get_string(self, tokens: torch.Tensor) -> str:
        """Get string for tokens (with caching)."""
        # Convert to tuple for hashing
        key = tuple(tokens.tolist())
        if key not in self.strings:
            self.strings[key] = tokens_to_string(tokens, self.idx_to_token)
        return self.strings[key]

    def get_components(self, formula: str) -> Tuple[List[str], List[str], List[str]]:
        """Get parsed components for formula (with caching)."""
        if formula not in self.components:
            self.components[formula] = parse_formula_components(formula)
        return self.components[formula]


def compute_reward_v8_optimized(
    sampled_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    idx_to_token: dict,
    mask: Optional[torch.Tensor] = None,
    config: Optional[RewardConfigV8Optimized] = None,
    target_cache: Optional[TargetCache] = None,
) -> torch.Tensor:
    """
    Compute V8.1 optimized reward with targeted penalties.

    Optimizations:
    - No Levenshtein distance (simple equality)
    - Optional caching for target parsing
    - Simplified fraction comparison

    Args:
        sampled_tokens: [batch, seq_len] sampled token indices
        target_tokens: [batch, seq_len] target token indices
        idx_to_token: Dictionary mapping indices to token strings
        mask: [batch, seq_len] valid positions
        config: Reward configuration
        target_cache: Optional cache for pre-parsed targets

    Returns:
        rewards: [batch] reward for each sequence
    """
    if config is None:
        config = RewardConfigV8Optimized()

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
        if target_cache:
            target_str = target_cache.get_string(target)
        else:
            target_str = tokens_to_string(target, idx_to_token)

        # =============================================
        # EXACT MATCH CHECK
        # =============================================
        if sampled_str == target_str:
            rewards[i] = config.exact_match
            continue

        # =============================================
        # NEAR-EXACT CHECK (fast token comparison)
        # =============================================
        sampled_list = sampled.tolist()
        target_list = target.tolist()

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

        if target_cache:
            target_elems, target_fracs, target_subs = target_cache.get_components(target_str)
        else:
            target_elems, target_fracs, target_subs = parse_formula_components(target_str)

        reward = 0.0

        # =============================================
        # ELEMENT ANALYSIS (HARSH PENALTIES)
        # =============================================
        n_sampled_elems = len(sampled_elems)
        n_target_elems = len(target_elems)

        if n_sampled_elems < n_target_elems:
            reward += config.missing_element * (n_target_elems - n_sampled_elems)
        if n_sampled_elems > n_target_elems:
            reward += config.extra_element * (n_sampled_elems - n_target_elems)

        elements_correct = 0
        elements_wrong = 0
        order_errors = 0

        min_elems = min(n_sampled_elems, n_target_elems)
        for j in range(min_elems):
            if sampled_elems[j] == target_elems[j]:
                elements_correct += 1
            elif sampled_elems[j] in target_elems:
                order_errors += 1
            else:
                elements_wrong += 1

        reward += config.wrong_element * elements_wrong
        reward += config.wrong_element_order * order_errors

        all_elements_correct = (elements_correct == n_target_elems and
                                n_sampled_elems == n_target_elems)
        if all_elements_correct:
            reward += config.all_elements_correct_bonus

        # =============================================
        # FRACTION ANALYSIS (SIMPLIFIED - NO LEVENSHTEIN)
        # =============================================
        n_sampled_fracs = len(sampled_fracs)
        n_target_fracs = len(target_fracs)

        if n_sampled_fracs < n_target_fracs:
            reward += config.missing_fraction * (n_target_fracs - n_sampled_fracs)
        if n_sampled_fracs > n_target_fracs:
            reward += config.extra_fraction * (n_sampled_fracs - n_target_fracs)

        fractions_correct = 0
        min_fracs = min(n_sampled_fracs, n_target_fracs)

        for j in range(min_fracs):
            if sampled_fracs[j] == target_fracs[j]:
                fractions_correct += 1
            else:
                # Simple wrong penalty (no Levenshtein)
                reward += config.wrong_fraction

        all_fractions_correct = (fractions_correct == n_target_fracs and
                                  n_sampled_fracs == n_target_fracs)
        if all_fractions_correct:
            reward += config.all_fractions_correct_bonus

        # =============================================
        # SUBSCRIPT ANALYSIS (MILD)
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
        # STRUCTURAL CHECK
        # =============================================
        open_parens = sampled_str.count('(')
        close_parens = sampled_str.count(')')
        if open_parens != close_parens:
            reward += config.unbalanced_parens

        rewards[i] = reward

    return rewards


def get_default_reward_config_optimized() -> RewardConfigV8Optimized:
    """Get default V8.1 optimized reward configuration."""
    return RewardConfigV8Optimized()
