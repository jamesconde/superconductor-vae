"""
V15.x: Site Duplication Loss — targets and stats for crystallographic site duplication head.

Only 0.35% (185/52,813) of training formulas legitimately have duplicate elements
(same element on different Wyckoff positions, e.g. Cu on dopant + CuO plane sites).
The model duplicates elements in ~34% of error records. This module provides:

  compute_site_dup_targets(): Vectorized [batch, seq_len] binary targets
      (1.0 where a token is a duplicate element, 0.0 elsewhere)

  compute_site_dup_stats(): Debug helper for inspecting target distributions
"""

import torch


def compute_site_dup_targets(
    formula_targets: torch.Tensor,
    element_start: int = 20,
    element_end: int = 137,
    pad_idx: int = 0,
) -> torch.Tensor:
    """Compute binary targets for the site duplication head.

    Returns [batch, seq_len] float tensor where 1.0 marks positions that are a
    *repeated* occurrence of an element token (i.e. this element already appeared
    earlier in the same sequence). First occurrences and non-element tokens are 0.0.

    Fully vectorized — no Python loops over batch or sequence.

    Args:
        formula_targets: [batch, seq_len] int64 token IDs (ground-truth, after START removal)
        element_start: First element token ID (inclusive). V12=20, V13/V14=5.
        element_end: Last element token ID (inclusive). V12=137, V13/V14=122.
        pad_idx: PAD token ID (excluded from element detection).

    Returns:
        targets: [batch, seq_len] float32 tensor. 1.0 at duplicate element positions.
    """
    batch, seq_len = formula_targets.shape
    device = formula_targets.device
    n_elements = element_end - element_start + 1  # 118

    # Step 1: Identify element tokens
    is_element = (
        (formula_targets >= element_start)
        & (formula_targets <= element_end)
        & (formula_targets != pad_idx)
    )  # [batch, seq_len]

    # Step 2: Map element tokens to local indices [0, n_elements)
    # Non-element positions are clamped to 0 but masked out by is_element
    elem_local = (formula_targets - element_start).clamp(0, n_elements - 1)  # [batch, seq_len]

    # Step 3: Build one-hot encoding for element positions only
    # [batch, seq_len, n_elements]
    one_hot = torch.zeros(batch, seq_len, n_elements, device=device)
    one_hot.scatter_(2, elem_local.unsqueeze(-1), is_element.unsqueeze(-1).float())

    # Step 4: Cumulative sum along sequence dimension gives running count per element
    cum_count = one_hot.cumsum(dim=1)  # [batch, seq_len, n_elements]

    # Step 5: Gather the count at each position for its element
    # counts_at_pos[b, t] = cum_count[b, t, elem_local[b, t]]
    counts_at_pos = cum_count.gather(2, elem_local.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]

    # Step 6: Duplicate = count > 1 AND is an element token
    targets = ((counts_at_pos > 1) & is_element).float()

    return targets


def compute_site_dup_stats(
    formula_targets: torch.Tensor,
    element_start: int = 20,
    element_end: int = 137,
    pad_idx: int = 0,
) -> dict:
    """Debug helper: compute statistics about site duplication targets.

    Returns dict with:
        n_positive: Total number of duplicate-element positions across batch
        n_element: Total number of element positions
        n_total: Total number of non-PAD positions
        positive_rate: Fraction of element positions that are duplicates
        n_samples_with_dups: Number of sequences containing at least one duplicate
        batch_size: Number of sequences in the batch
    """
    targets = compute_site_dup_targets(formula_targets, element_start, element_end, pad_idx)
    is_element = (
        (formula_targets >= element_start)
        & (formula_targets <= element_end)
        & (formula_targets != pad_idx)
    )
    non_pad = formula_targets != pad_idx

    n_positive = targets.sum().item()
    n_element = is_element.sum().item()
    n_total = non_pad.sum().item()

    # Per-sequence: does this sequence have any duplicates?
    has_dup = targets.sum(dim=1) > 0  # [batch]
    n_samples_with_dups = has_dup.sum().item()

    return {
        'n_positive': int(n_positive),
        'n_element': int(n_element),
        'n_total': int(n_total),
        'positive_rate': n_positive / max(n_element, 1),
        'n_samples_with_dups': int(n_samples_with_dups),
        'batch_size': formula_targets.shape[0],
    }
