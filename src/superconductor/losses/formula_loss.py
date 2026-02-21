"""
Weighted Loss and Per-Token-Type Accuracy for Chemical Formula Generation.

Chemical formulas require 100% accuracy - there is no "close enough".
La(7/10)Sr(3/10)CuO4 and La(8/10)Sr(3/10)CuO4 are completely different compounds.

This module provides:
1. Weighted cross-entropy loss (elements penalized more than digits)
2. Per-token-type accuracy tracking (elements, digits, structure, fractions)
3. Exact match tracking

Token Types:
    - ELEMENT: Chemical element symbols (La, Sr, Cu, O, etc.) - CRITICAL
    - DIGIT: Numbers 0-9 for stoichiometry - IMPORTANT
    - STRUCTURE: ( ) / for fraction notation - IMPORTANT
    - CONTROL: <PAD>, <START>, <END> - ignored in accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import vocabulary from autoregressive decoder
from ..models.autoregressive_decoder import (
    VOCAB, TOKEN_TO_IDX, IDX_TO_TOKEN, VOCAB_SIZE,
    PAD_IDX, START_IDX, END_IDX,
    ELEMENTS, DIGITS, SPECIAL_CHARS
)

# V13.0: Support for semantic fraction tokenizer with larger vocab
# When V13 is active, _active_vocab_size may be > VOCAB_SIZE (148)
_active_vocab_size: int = VOCAB_SIZE
_v13_tokenizer = None


def set_formula_loss_tokenizer(tokenizer) -> None:
    """Configure formula_loss for V13 semantic fraction tokenizer.

    Call this during training setup if using V13. Updates TOKEN_TYPE_MASK
    to cover the full V13 vocab (elements + integers + fraction tokens).
    """
    global _active_vocab_size, _v13_tokenizer, TOKEN_TYPE_MASK
    _v13_tokenizer = tokenizer
    _active_vocab_size = tokenizer.vocab_size
    TOKEN_TYPE_MASK = _build_token_type_mask_v13(tokenizer)


def _build_token_type_mask_v13(tokenizer) -> torch.Tensor:
    """Build TOKEN_TYPE_MASK for V13 semantic fraction tokenizer."""
    type_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.long)
    # PAD=0, BOS=1, EOS=2, UNK=3, FRAC_UNK=4 -> CONTROL
    for idx in [0, 1, 2, 3, 4]:
        type_mask[idx] = TokenType.CONTROL.value
    # Elements (5-122) -> ELEMENT
    for idx in range(5, 123):
        type_mask[idx] = TokenType.ELEMENT.value
    # Integers 1-20 (123-142) -> DIGIT
    for idx in range(123, 143):
        type_mask[idx] = TokenType.DIGIT.value
    # Fraction tokens -> STRUCTURE (closest semantic category)
    frac_end = tokenizer.fraction_token_start + tokenizer.n_fraction_tokens
    for idx in range(tokenizer.fraction_token_start, frac_end):
        type_mask[idx] = TokenType.STRUCTURE.value
    return type_mask


class TokenType(Enum):
    """Categories of tokens in chemical formulas."""
    CONTROL = 0    # <PAD>, <START>, <END>
    ELEMENT = 1    # Chemical elements (La, Sr, Cu, O, etc.)
    DIGIT = 2      # Numbers 0-9
    STRUCTURE = 3  # ( ) / for fractions
    OTHER = 4      # Other special characters


def get_token_type(token_idx: int) -> TokenType:
    """Classify a token index into its type."""
    if token_idx in [PAD_IDX, START_IDX, END_IDX]:
        return TokenType.CONTROL

    token = IDX_TO_TOKEN.get(token_idx, '')

    if token in ELEMENTS[1:]:  # Skip empty string at index 0
        return TokenType.ELEMENT
    elif token in DIGITS:
        return TokenType.DIGIT
    elif token in ['(', ')', '/']:
        return TokenType.STRUCTURE
    else:
        return TokenType.OTHER


def build_token_type_mask() -> torch.Tensor:
    """Build a tensor mapping token indices to their types (V12 layout)."""
    type_mask = torch.zeros(VOCAB_SIZE, dtype=torch.long)
    for idx in range(VOCAB_SIZE):
        type_mask[idx] = get_token_type(idx).value
    return type_mask


# Pre-compute token type mask
TOKEN_TYPE_MASK = build_token_type_mask()


def build_loss_weights(
    element_weight: float = 5.0,
    digit_weight: float = 2.0,
    structure_weight: float = 3.0,
    other_weight: float = 1.0,
    vocab_size: int = 0,
    fraction_weight: float = 0.0,
) -> torch.Tensor:
    """
    Build per-token loss weights.

    Elements are most critical - wrong element = wrong compound.
    Structure tokens (parentheses, slash) define the formula syntax.
    Digits define stoichiometry - equally important for correctness.

    Args:
        element_weight: Weight for element tokens (default 5.0)
        digit_weight: Weight for digit tokens (default 2.0)
        structure_weight: Weight for ( ) / tokens (default 3.0)
        other_weight: Weight for other tokens (default 1.0)
        vocab_size: Override vocab size (0 = use active vocab size)
        fraction_weight: Override weight for V13 fraction tokens (0 = use structure_weight)

    Returns:
        Tensor of shape [vocab_size] with per-token weights
    """
    vs = vocab_size if vocab_size > 0 else _active_vocab_size
    weights = torch.ones(vs)

    # V13: use the V13-aware TOKEN_TYPE_MASK directly
    if vs > VOCAB_SIZE and _v13_tokenizer is not None:
        frac_w = fraction_weight if fraction_weight > 0 else structure_weight
        for idx in range(vs):
            if idx < len(TOKEN_TYPE_MASK):
                tt = TOKEN_TYPE_MASK[idx].item()
            else:
                tt = TokenType.OTHER.value
            if tt == TokenType.ELEMENT.value:
                weights[idx] = element_weight
            elif tt == TokenType.DIGIT.value:
                weights[idx] = digit_weight
            elif tt == TokenType.STRUCTURE.value:
                weights[idx] = frac_w
            elif tt == TokenType.OTHER.value:
                weights[idx] = other_weight
        return weights

    # V12: original path
    for idx in range(vs):
        token_type = get_token_type(idx)
        if token_type == TokenType.ELEMENT:
            weights[idx] = element_weight
        elif token_type == TokenType.DIGIT:
            weights[idx] = digit_weight
        elif token_type == TokenType.STRUCTURE:
            weights[idx] = structure_weight
        elif token_type == TokenType.OTHER:
            weights[idx] = other_weight
        # CONTROL tokens keep weight 1.0

    return weights


@dataclass
class FormulaAccuracyMetrics:
    """Per-token-type accuracy metrics."""
    overall: float           # Overall token accuracy
    element: float           # Accuracy on element tokens
    digit: float             # Accuracy on digit tokens
    structure: float         # Accuracy on ( ) / tokens
    exact_match: float       # Full sequence exact match rate

    # Counts for debugging
    n_element_tokens: int
    n_digit_tokens: int
    n_structure_tokens: int
    n_total_tokens: int
    n_sequences: int
    n_exact_matches: int

    def __str__(self) -> str:
        return (
            f"Accuracy Breakdown:\n"
            f"  Overall:    {self.overall:.1%} ({self.n_total_tokens} tokens)\n"
            f"  Elements:   {self.element:.1%} ({self.n_element_tokens} tokens)\n"
            f"  Digits:     {self.digit:.1%} ({self.n_digit_tokens} tokens)\n"
            f"  Structure:  {self.structure:.1%} ({self.n_structure_tokens} tokens)\n"
            f"  Exact Match: {self.exact_match:.1%} ({self.n_exact_matches}/{self.n_sequences})"
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            'overall': self.overall,
            'element': self.element,
            'digit': self.digit,
            'structure': self.structure,
            'exact_match': self.exact_match,
        }


class FormulaAccuracyTracker:
    """
    Track per-token-type accuracy during training/evaluation.

    Usage:
        tracker = FormulaAccuracyTracker()

        for batch in dataloader:
            predictions = model(batch)
            tracker.update(predictions, targets, mask)

        metrics = tracker.compute()
        print(metrics)
        tracker.reset()
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.token_type_mask = TOKEN_TYPE_MASK.to(self.device)
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.correct = {t: 0 for t in TokenType}
        self.total = {t: 0 for t in TokenType}
        self.n_sequences = 0
        self.n_exact_matches = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        Update accuracy counters with a batch.

        Args:
            predictions: [batch, seq_len] predicted token indices
            targets: [batch, seq_len] target token indices
            mask: [batch, seq_len] True for valid (non-padding) positions
        """
        if mask is None:
            mask = targets != PAD_IDX

        # Move to correct device
        if self.token_type_mask.device != predictions.device:
            self.token_type_mask = self.token_type_mask.to(predictions.device)

        batch_size, seq_len = predictions.shape

        # If targets contain indices beyond the mask size (V13 vocab), resize
        max_target = targets.max().item()
        if max_target >= self.token_type_mask.shape[0]:
            new_mask = TOKEN_TYPE_MASK.to(predictions.device)
            if new_mask.shape[0] > self.token_type_mask.shape[0]:
                self.token_type_mask = new_mask

        # Per-token correctness
        correct_mask = (predictions == targets) & mask

        # Get token types for targets — clamp to avoid out-of-bounds
        clamped_targets = targets.clamp(max=self.token_type_mask.shape[0] - 1)
        target_types = self.token_type_mask[clamped_targets]  # [batch, seq_len]

        # Count per type
        for token_type in TokenType:
            if token_type == TokenType.CONTROL:
                continue  # Skip control tokens

            type_mask = (target_types == token_type.value) & mask
            type_correct = correct_mask & type_mask

            self.correct[token_type] += type_correct.sum().item()
            self.total[token_type] += type_mask.sum().item()

        # Exact match: all non-padding tokens correct
        sequence_correct = (correct_mask == mask).all(dim=1)
        self.n_exact_matches += sequence_correct.sum().item()
        self.n_sequences += batch_size

    def compute(self) -> FormulaAccuracyMetrics:
        """Compute final metrics."""
        def safe_div(a, b):
            return a / b if b > 0 else 0.0

        total_correct = sum(self.correct[t] for t in TokenType if t != TokenType.CONTROL)
        total_count = sum(self.total[t] for t in TokenType if t != TokenType.CONTROL)

        return FormulaAccuracyMetrics(
            overall=safe_div(total_correct, total_count),
            element=safe_div(self.correct[TokenType.ELEMENT], self.total[TokenType.ELEMENT]),
            digit=safe_div(self.correct[TokenType.DIGIT], self.total[TokenType.DIGIT]),
            structure=safe_div(self.correct[TokenType.STRUCTURE], self.total[TokenType.STRUCTURE]),
            exact_match=safe_div(self.n_exact_matches, self.n_sequences),
            n_element_tokens=self.total[TokenType.ELEMENT],
            n_digit_tokens=self.total[TokenType.DIGIT],
            n_structure_tokens=self.total[TokenType.STRUCTURE],
            n_total_tokens=total_count,
            n_sequences=self.n_sequences,
            n_exact_matches=self.n_exact_matches,
        )


class WeightedFormulaLoss(nn.Module):
    """
    Weighted cross-entropy loss for chemical formulas.

    Applies higher weights to:
    - Element tokens (wrong element = wrong compound)
    - Structure tokens (wrong syntax = unparseable)
    - Digit tokens (wrong stoichiometry = wrong compound)

    Args:
        element_weight: Weight for element tokens (default 5.0)
        digit_weight: Weight for digit tokens (default 2.0)
        structure_weight: Weight for ( ) / tokens (default 3.0)
        label_smoothing: Label smoothing factor (default 0.0)
        exact_match_weight: Weight for exact match auxiliary loss (default 0.0)
    """

    def __init__(
        self,
        element_weight: float = 5.0,
        digit_weight: float = 2.0,
        structure_weight: float = 3.0,
        label_smoothing: float = 0.0,
        exact_match_weight: float = 0.0,
        reduction: str = 'sum_over_positions'  # 'mean', 'sum', or 'sum_over_positions' (sum positions, mean batch)
    ):
        super().__init__()

        self.element_weight = element_weight
        self.digit_weight = digit_weight
        self.structure_weight = structure_weight
        self.exact_match_weight = exact_match_weight
        self.reduction = reduction

        # Build weight tensor
        weights = build_loss_weights(
            element_weight=element_weight,
            digit_weight=digit_weight,
            structure_weight=structure_weight
        )
        self.register_buffer('weights', weights)

        # For 'sum_over_positions', we need per-element loss
        if reduction == 'sum_over_positions':
            ce_reduction = 'none'
        else:
            ce_reduction = reduction

        # Cross-entropy with weights
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=PAD_IDX,
            label_smoothing=label_smoothing,
            reduction=ce_reduction
        )

        # Token type mask for exact match computation
        self.register_buffer('token_type_mask', TOKEN_TYPE_MASK)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss.

        Args:
            logits: [batch, seq_len, vocab_size] model predictions
            targets: [batch, seq_len] target token indices
            mask: [batch, seq_len] optional valid position mask

        Returns:
            Dict with 'total', 'ce', and optionally 'exact_match' losses
        """
        batch_size, seq_len, vocab_size = logits.shape

        # V13: rebuild CE loss if vocab size changed (e.g., 148 -> 4355)
        if vocab_size != self.weights.shape[0]:
            new_weights = build_loss_weights(
                element_weight=self.element_weight,
                digit_weight=self.digit_weight,
                structure_weight=self.structure_weight,
                vocab_size=vocab_size,
            ).to(logits.device)
            self.weights = new_weights
            ce_reduction = 'none' if self.reduction == 'sum_over_positions' else self.reduction
            self.ce_loss = nn.CrossEntropyLoss(
                weight=new_weights,
                ignore_index=PAD_IDX,
                reduction=ce_reduction,
            )

        # Reshape for cross-entropy: [batch*seq, vocab] vs [batch*seq]
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # Weighted cross-entropy
        ce_loss_raw = self.ce_loss(logits_flat, targets_flat)

        # Handle 'sum_over_positions' reduction:
        # - Sum losses within each sequence (every error counts equally)
        # - Mean over sequences in batch (for consistent gradient magnitude)
        if self.reduction == 'sum_over_positions':
            # ce_loss_raw is [batch*seq] with 0 for PAD positions
            ce_loss_per_seq = ce_loss_raw.view(batch_size, seq_len)  # [batch, seq]
            ce_loss_sum_per_seq = ce_loss_per_seq.sum(dim=1)  # [batch] - sum over positions
            ce_loss = ce_loss_sum_per_seq.mean()  # scalar - mean over batch
        else:
            ce_loss = ce_loss_raw

        total_loss = ce_loss
        result = {'ce': ce_loss, 'total': total_loss}

        # Optional exact match auxiliary loss
        if self.exact_match_weight > 0:
            if mask is None:
                mask = targets != PAD_IDX

            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets) | ~mask
            sequence_correct = correct.all(dim=1).float()
            exact_match_loss = 1.0 - sequence_correct.mean()

            total_loss = ce_loss + self.exact_match_weight * exact_match_loss
            result['exact_match'] = exact_match_loss
            result['total'] = total_loss

        return result


class FormulaLossWithAccuracy(nn.Module):
    """
    Combined loss and accuracy tracking for training.

    Provides:
    - Weighted cross-entropy loss
    - Per-token-type accuracy
    - Exact match tracking

    Usage:
        criterion = FormulaLossWithAccuracy(element_weight=5.0)

        for batch in dataloader:
            logits = model(inputs)
            loss_dict, acc_dict = criterion(logits, targets)
            loss_dict['total'].backward()
    """

    def __init__(
        self,
        element_weight: float = 5.0,
        digit_weight: float = 2.0,
        structure_weight: float = 3.0,
        label_smoothing: float = 0.0,
        exact_match_weight: float = 0.1,
        reduction: str = 'sum_over_positions'  # 'mean', 'sum', or 'sum_over_positions' (recommended)
    ):
        super().__init__()

        self.loss_fn = WeightedFormulaLoss(
            element_weight=element_weight,
            digit_weight=digit_weight,
            structure_weight=structure_weight,
            label_smoothing=label_smoothing,
            exact_match_weight=exact_match_weight,
            reduction=reduction
        )

        self.accuracy_tracker = FormulaAccuracyTracker()
        self.register_buffer('token_type_mask', TOKEN_TYPE_MASK)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        track_accuracy: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Compute loss and accuracy.

        Returns:
            loss_dict: {'total', 'ce', 'exact_match'}
            acc_dict: {'overall', 'element', 'digit', 'structure', 'exact_match'}
        """
        if mask is None:
            mask = targets != PAD_IDX

        # Compute loss
        loss_dict = self.loss_fn(logits, targets, mask)

        # Compute accuracy
        acc_dict = {}
        if track_accuracy:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)

                # Move token type mask to correct device
                if self.token_type_mask.device != predictions.device:
                    self.token_type_mask = self.token_type_mask.to(predictions.device)

                # V13: resize token type mask if targets exceed it
                max_target = targets.max().item()
                if max_target >= self.token_type_mask.shape[0]:
                    new_mask = TOKEN_TYPE_MASK.to(predictions.device)
                    if new_mask.shape[0] > self.token_type_mask.shape[0]:
                        self.token_type_mask = new_mask

                # Per-type accuracy
                correct = (predictions == targets) & mask
                clamped_targets = targets.clamp(max=self.token_type_mask.shape[0] - 1)
                target_types = self.token_type_mask[clamped_targets]

                for token_type in [TokenType.ELEMENT, TokenType.DIGIT, TokenType.STRUCTURE]:
                    type_mask = (target_types == token_type.value) & mask
                    if type_mask.sum() > 0:
                        type_acc = (correct & type_mask).sum().float() / type_mask.sum()
                        acc_dict[token_type.name.lower()] = type_acc.item()
                    else:
                        acc_dict[token_type.name.lower()] = 0.0

                # Overall accuracy
                if mask.sum() > 0:
                    acc_dict['overall'] = correct.sum().float().item() / mask.sum().item()
                else:
                    acc_dict['overall'] = 0.0

                # Exact match
                seq_correct = ((predictions == targets) | ~mask).all(dim=1)
                acc_dict['exact_match'] = seq_correct.float().mean().item()

        return loss_dict, acc_dict

    def reset_accuracy(self):
        """Reset accuracy tracker (call at start of epoch)."""
        self.accuracy_tracker.reset()


def format_accuracy_log(acc_dict: Dict[str, float], prefix: str = "") -> str:
    """Format accuracy dict for logging."""
    return (
        f"{prefix}acc: {acc_dict.get('overall', 0):.1%} | "
        f"elem: {acc_dict.get('element', 0):.1%} | "
        f"digit: {acc_dict.get('digit', 0):.1%} | "
        f"struct: {acc_dict.get('structure', 0):.1%} | "
        f"exact: {acc_dict.get('exact_match', 0):.1%}"
    )
