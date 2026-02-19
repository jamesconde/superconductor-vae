"""
Semantic Unit Loss for Chemical Formula Reconstruction.

The Problem:
Token-level loss treats "Ni" -> "N" as one token error.
But semantically, this is a COMPLETE element failure - Nickel vs Nitrogen.

The Solution:
Parse token sequences into semantic units (elements, fractions) and apply
additional loss at the unit level. This teaches the model that partial
element predictions are catastrophic failures.

Semantic Units:
1. Elements: "La", "Sr", "Cu", "O", "Ni" etc.
2. Fractions: "(7/10)", "(1/500)" etc.
3. Subscripts: standalone numbers like "2", "3", "4"

Loss Components:
1. Token-level CE loss (existing) - per-token accuracy
2. Element-level loss - are the elements correct in order?
3. Fraction-level loss - is the stoichiometry correct?
4. Sequence-level loss - exact match penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import re

from ..models.autoregressive_decoder import (
    IDX_TO_TOKEN, TOKEN_TO_IDX, PAD_IDX, START_IDX, END_IDX,
    ELEMENTS
)

# V13.0: Optional semantic fraction tokenizer
_v13_tokenizer = None


def set_semantic_unit_tokenizer(tokenizer) -> None:
    """Configure semantic_unit_loss for V13 semantic fraction tokenizer."""
    global _v13_tokenizer
    _v13_tokenizer = tokenizer


@dataclass
class SemanticUnit:
    """A semantic unit in a chemical formula."""
    unit_type: str  # 'element', 'fraction', 'subscript', 'structure'
    tokens: List[str]  # The tokens that make up this unit
    positions: List[int]  # Positions of these tokens in the sequence
    value: str  # The semantic value (e.g., "Ni", "(7/10)", "2")


def parse_tokens_to_semantic_units(token_indices: torch.Tensor) -> List[SemanticUnit]:
    """
    Parse a sequence of token indices into semantic units.

    Supports both V12 (character-level) and V13 (semantic fraction) tokenizers.

    Example (V12):
        [La, (, 7, /, 1, 0, ), Sr, (, 3, /, 1, 0, ), Cu, O, 4]
        ->
        [Element("La"), Fraction("(7/10)"), Element("Sr"), Fraction("(3/10)"),
         Element("Cu"), Element("O"), Subscript("4")]

    Example (V13):
        [La, FRAC:7/10, Sr, FRAC:3/10, Cu, O, 4]
        ->
        [Element("La"), Fraction("(7/10)"), Element("Sr"), Fraction("(3/10)"),
         Element("Cu"), Element("O"), Subscript("4")]
    """
    # V13 path: use the semantic fraction tokenizer
    if _v13_tokenizer is not None:
        return _parse_tokens_v13(token_indices)

    # V12 path: original character-level parsing
    return _parse_tokens_v12(token_indices)


def _parse_tokens_v13(token_indices: torch.Tensor) -> List[SemanticUnit]:
    """Parse V13 semantic fraction token sequence into semantic units."""
    units = []
    tok = _v13_tokenizer

    for idx, token_idx in enumerate(token_indices.tolist()):
        # Skip special tokens (PAD=0, BOS=1, EOS=2, UNK=3, FRAC_UNK=4)
        if token_idx <= 4:
            continue

        # Element tokens (5-122)
        if tok.is_element_token(token_idx):
            name = tok.decode([token_idx])
            units.append(SemanticUnit(
                unit_type='element',
                tokens=[name],
                positions=[idx],
                value=name
            ))

        # Integer tokens (123-142)
        elif tok.is_integer_token(token_idx):
            val = tok.decode([token_idx])
            units.append(SemanticUnit(
                unit_type='subscript',
                tokens=[val],
                positions=[idx],
                value=val
            ))

        # Fraction tokens (143+)
        elif tok.is_fraction_token(token_idx):
            frac_str = tok.decode([token_idx])  # e.g., "(7/10)"
            units.append(SemanticUnit(
                unit_type='fraction',
                tokens=[frac_str],
                positions=[idx],
                value=frac_str
            ))

    return units


def _parse_tokens_v12(token_indices: torch.Tensor) -> List[SemanticUnit]:
    """Parse V12 character-level token sequence into semantic units."""
    units = []
    tokens = []

    # Convert indices to tokens
    for idx, token_idx in enumerate(token_indices.tolist()):
        if token_idx in [PAD_IDX, START_IDX, END_IDX]:
            continue
        token = IDX_TO_TOKEN.get(token_idx, '')
        if token:
            tokens.append((idx, token))

    if not tokens:
        return []

    i = 0
    while i < len(tokens):
        pos, token = tokens[i]

        # Check for element (1 or 2 character chemical symbol)
        if token in ELEMENTS[1:]:  # Skip empty string
            units.append(SemanticUnit(
                unit_type='element',
                tokens=[token],
                positions=[pos],
                value=token
            ))
            i += 1
            continue

        # Check for fraction start: (
        if token == '(':
            # Collect until we find )
            fraction_tokens = [token]
            fraction_positions = [pos]
            j = i + 1
            while j < len(tokens) and tokens[j][1] != ')':
                fraction_tokens.append(tokens[j][1])
                fraction_positions.append(tokens[j][0])
                j += 1

            if j < len(tokens):
                # Found closing )
                fraction_tokens.append(tokens[j][1])
                fraction_positions.append(tokens[j][0])
                j += 1

            units.append(SemanticUnit(
                unit_type='fraction',
                tokens=fraction_tokens,
                positions=fraction_positions,
                value=''.join(fraction_tokens)
            ))
            i = j
            continue

        # Check for standalone digit (subscript)
        if token.isdigit():
            units.append(SemanticUnit(
                unit_type='subscript',
                tokens=[token],
                positions=[pos],
                value=token
            ))
            i += 1
            continue

        # Other structure tokens
        units.append(SemanticUnit(
            unit_type='structure',
            tokens=[token],
            positions=[pos],
            value=token
        ))
        i += 1

    return units


def compute_semantic_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    element_penalty: float = 5.0,
    fraction_penalty: float = 3.0,
    exact_match_penalty: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute loss at the semantic unit level.

    Args:
        predictions: [batch, seq_len] predicted token indices
        targets: [batch, seq_len] target token indices
        element_penalty: Penalty multiplier for wrong elements
        fraction_penalty: Penalty multiplier for wrong fractions
        exact_match_penalty: Penalty for non-exact match

    Returns:
        Dict with 'element_loss', 'fraction_loss', 'exact_match_loss', 'total'
    """
    batch_size = predictions.shape[0]
    device = predictions.device

    element_losses = []
    fraction_losses = []
    exact_losses = []

    for i in range(batch_size):
        pred_units = parse_tokens_to_semantic_units(predictions[i])
        target_units = parse_tokens_to_semantic_units(targets[i])

        # Separate by type
        pred_elements = [u for u in pred_units if u.unit_type == 'element']
        target_elements = [u for u in target_units if u.unit_type == 'element']

        pred_fractions = [u for u in pred_units if u.unit_type == 'fraction']
        target_fractions = [u for u in target_units if u.unit_type == 'fraction']

        # Element loss: compare element sequences
        n_elements = max(len(pred_elements), len(target_elements))
        element_errors = 0
        if n_elements > 0:
            for j in range(min(len(pred_elements), len(target_elements))):
                if pred_elements[j].value != target_elements[j].value:
                    element_errors += 1
            # Count missing or extra elements
            element_errors += abs(len(pred_elements) - len(target_elements))
            element_losses.append(element_errors / n_elements)
        else:
            element_losses.append(0.0)

        # Fraction loss: compare fraction sequences
        n_fractions = max(len(pred_fractions), len(target_fractions))
        fraction_errors = 0
        if n_fractions > 0:
            for j in range(min(len(pred_fractions), len(target_fractions))):
                if pred_fractions[j].value != target_fractions[j].value:
                    fraction_errors += 1
            # Count missing or extra fractions
            fraction_errors += abs(len(pred_fractions) - len(target_fractions))
            fraction_losses.append(fraction_errors / n_fractions)
        else:
            fraction_losses.append(0.0)

        # Exact match loss
        pred_str = ''.join([u.value for u in pred_units])
        target_str = ''.join([u.value for u in target_units])
        if pred_str != target_str:
            exact_losses.append(1.0)
        else:
            exact_losses.append(0.0)

    element_loss = torch.tensor(element_losses, device=device).mean() * element_penalty
    fraction_loss = torch.tensor(fraction_losses, device=device).mean() * fraction_penalty
    exact_match_loss = torch.tensor(exact_losses, device=device).mean() * exact_match_penalty

    total_loss = element_loss + fraction_loss + exact_match_loss

    return {
        'element_loss': element_loss,
        'fraction_loss': fraction_loss,
        'exact_match_loss': exact_match_loss,
        'total': total_loss
    }


class SemanticUnitLoss(nn.Module):
    """
    Combined token-level and semantic unit loss.

    Total Loss = token_ce_loss + element_penalty * element_loss
                 + fraction_penalty * fraction_loss + exact_penalty * exact_loss
    """

    def __init__(
        self,
        element_weight: float = 5.0,
        digit_weight: float = 4.0,
        structure_weight: float = 1.5,
        element_penalty: float = 5.0,
        fraction_penalty: float = 3.0,
        exact_match_penalty: float = 1.0,
        reduction: str = 'sum_over_positions'
    ):
        super().__init__()

        self.element_penalty = element_penalty
        self.fraction_penalty = fraction_penalty
        self.exact_match_penalty = exact_match_penalty

        # Import the token-level loss
        from .formula_loss import FormulaLossWithAccuracy
        self.token_loss_fn = FormulaLossWithAccuracy(
            element_weight=element_weight,
            digit_weight=digit_weight,
            structure_weight=structure_weight,
            exact_match_weight=0.0,  # We compute our own exact match loss
            label_smoothing=0.0,
            reduction=reduction
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        track_accuracy: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Compute combined token and semantic unit loss.

        Args:
            logits: [batch, seq_len, vocab_size] model predictions
            targets: [batch, seq_len] target token indices
            mask: optional mask
            track_accuracy: whether to compute accuracy metrics

        Returns:
            loss_dict: {'total', 'token_ce', 'element', 'fraction', 'exact_match'}
            acc_dict: accuracy metrics from token-level evaluation
        """
        # Token-level loss and accuracy
        token_loss_dict, acc_dict = self.token_loss_fn(
            logits, targets, mask, track_accuracy
        )

        # Get predictions for semantic loss
        predictions = logits.argmax(dim=-1)

        # Semantic unit loss (no gradient through argmax, but still useful as penalty)
        semantic_loss_dict = compute_semantic_loss(
            predictions, targets,
            element_penalty=self.element_penalty,
            fraction_penalty=self.fraction_penalty,
            exact_match_penalty=self.exact_match_penalty
        )

        # Combined loss
        total_loss = (
            token_loss_dict['total'] +
            semantic_loss_dict['element_loss'] +
            semantic_loss_dict['fraction_loss'] +
            semantic_loss_dict['exact_match_loss']
        )

        return {
            'total': total_loss,
            'token_ce': token_loss_dict['total'],
            'element_loss': semantic_loss_dict['element_loss'],
            'fraction_loss': semantic_loss_dict['fraction_loss'],
            'exact_match_loss': semantic_loss_dict['exact_match_loss']
        }, acc_dict


def test_semantic_parsing():
    """Test the semantic unit parsing."""
    # Create a test sequence: La(7/10)Sr(3/10)CuO4
    tokens = ['La', '(', '7', '/', '1', '0', ')', 'Sr', '(', '3', '/', '1', '0', ')', 'Cu', 'O', '4']

    # Convert to indices
    indices = torch.tensor([TOKEN_TO_IDX.get(t, 0) for t in tokens])

    # Parse
    units = parse_tokens_to_semantic_units(indices)

    print("Parsed semantic units:")
    for unit in units:
        print(f"  {unit.unit_type}: '{unit.value}' at positions {unit.positions}")

    return units


if __name__ == '__main__':
    test_semantic_parsing()
