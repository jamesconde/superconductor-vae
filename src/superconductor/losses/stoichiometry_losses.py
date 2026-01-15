"""
V12.4 Stoichiometry-Aware Losses for Superconductor VAE.

These losses help the model understand that fractions have NUMERICAL MEANING:
- 1/5 (0.2) and 9/5 (1.8) are NOT just a single token difference
- The stoichiometry error is proportional to the numerical difference

This provides gradient signals that scale with the magnitude of stoichiometry errors,
helping the model learn precise stoichiometry rather than treating all digit errors equally.

Key insight: Token-level cross-entropy treats '1' vs '9' as equally wrong as '1' vs '2'.
But stoichiometrically, the difference between Fe(1/5) and Fe(9/5) is 9x larger than
the difference between Fe(1/5) and Fe(2/5). This loss captures that semantic meaning.

December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StoichiometryLossConfig:
    """Configuration for stoichiometry-aware losses."""

    # Fraction MSE weight (main stoichiometry loss)
    fraction_mse_weight: float = 5.0

    # Element count prediction weight
    element_count_weight: float = 1.0

    # Composition vector similarity weight (for REINFORCE reward)
    composition_similarity_weight: float = 2.0

    # Scaling factor for fraction values (stoichiometry is typically 0-10)
    fraction_scale: float = 1.0

    # Whether to use log-scale for large fractions (handles outliers better)
    use_log_scale: bool = False

    # Mask threshold - only penalize predictions where ground truth has a valid element
    mask_threshold: float = 0.001


class StoichiometryMSELoss(nn.Module):
    """
    MSE loss on predicted element fractions.

    This loss directly compares the numerical values of predicted stoichiometry
    to ground truth, providing gradients proportional to the magnitude of errors.

    Key benefit: Unlike token-level CE where predicting '1' vs '9' is a single
    token error, this loss penalizes by |0.2 - 1.8| = 1.6 (8x larger than |0.2 - 0.4|).
    """

    def __init__(self, config: Optional[StoichiometryLossConfig] = None):
        super().__init__()
        self.config = config or StoichiometryLossConfig()

    def forward(
        self,
        fraction_pred: torch.Tensor,     # [batch, max_elements] predicted fractions
        element_fractions: torch.Tensor,  # [batch, max_elements] ground truth fractions
        element_mask: torch.Tensor,       # [batch, max_elements] valid element mask
        element_count_pred: Optional[torch.Tensor] = None,  # [batch] predicted count
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stoichiometry-aware MSE loss.

        Args:
            fraction_pred: Predicted element fractions from FractionValueHead
            element_fractions: Ground truth element fractions (from input data)
            element_mask: Boolean mask indicating valid element slots
            element_count_pred: Optional predicted element count

        Returns:
            Dict with 'stoich_mse', 'element_count_loss', 'stoich_total'
        """
        # Ensure mask is float for multiplication
        mask_float = element_mask.float()

        # Apply optional log scaling for large fractions
        if self.config.use_log_scale:
            # log(1 + x) transformation for better gradient flow on large values
            pred_scaled = torch.log1p(fraction_pred.abs()) * torch.sign(fraction_pred)
            target_scaled = torch.log1p(element_fractions)
        else:
            pred_scaled = fraction_pred
            target_scaled = element_fractions

        # Masked MSE: only compute loss where we have valid elements
        # This prevents the model from being penalized for predicting non-zero
        # values in empty element slots
        squared_error = (pred_scaled - target_scaled) ** 2
        masked_squared_error = squared_error * mask_float

        # Average over valid elements only
        n_valid = mask_float.sum(dim=1, keepdim=True).clamp(min=1)
        per_sample_mse = masked_squared_error.sum(dim=1) / n_valid.squeeze(-1)

        stoich_mse = per_sample_mse.mean() * self.config.fraction_mse_weight

        # Element count loss (how many elements in formula)
        losses = {'stoich_mse': stoich_mse}

        if element_count_pred is not None:
            # Ground truth element count = sum of mask
            element_count_target = element_mask.sum(dim=1).float()
            element_count_loss = F.mse_loss(element_count_pred, element_count_target)
            losses['element_count_loss'] = element_count_loss * self.config.element_count_weight
        else:
            losses['element_count_loss'] = torch.tensor(0.0, device=fraction_pred.device)

        # Total stoichiometry loss
        losses['stoich_total'] = losses['stoich_mse'] + losses['element_count_loss']

        return losses


class CompositionVectorLoss(nn.Module):
    """
    Loss based on composition vector similarity.

    Converts formulas to 118-dimensional composition vectors (element fractions)
    and computes cosine similarity or MSE. This captures the overall chemical
    composition rather than individual token positions.

    Useful for:
    - REINFORCE rewards (composition similarity bonus)
    - Auxiliary training signal
    - Evaluation metric
    """

    def __init__(self, n_elements: int = 118, config: Optional[StoichiometryLossConfig] = None):
        super().__init__()
        self.n_elements = n_elements
        self.config = config or StoichiometryLossConfig()

    def elements_to_composition_vector(
        self,
        element_indices: torch.Tensor,    # [batch, max_elements] atomic numbers
        element_fractions: torch.Tensor,  # [batch, max_elements] molar fractions
        element_mask: torch.Tensor,       # [batch, max_elements] valid element mask
    ) -> torch.Tensor:
        """
        Convert sparse element representation to dense 118-dim composition vector.

        Args:
            element_indices: Atomic numbers (1-118) for each element slot
            element_fractions: Molar fraction for each element slot
            element_mask: Boolean mask for valid elements

        Returns:
            composition_vector: [batch, 118] dense composition vector
        """
        batch_size = element_indices.size(0)
        device = element_indices.device

        # Initialize dense composition vector
        composition = torch.zeros(batch_size, self.n_elements, device=device)

        # Scatter element fractions into composition vector
        # element_indices are 1-indexed (atomic numbers), so subtract 1
        valid_indices = element_indices.clamp(min=1) - 1  # Convert to 0-indexed

        # Apply mask to fractions
        masked_fractions = element_fractions * element_mask.float()

        # Scatter into composition vector
        # Use scatter_add in case of duplicate indices (shouldn't happen but safe)
        composition.scatter_add_(1, valid_indices.long(), masked_fractions)

        return composition

    def forward(
        self,
        pred_element_indices: torch.Tensor,
        pred_element_fractions: torch.Tensor,
        pred_element_mask: torch.Tensor,
        target_element_indices: torch.Tensor,
        target_element_fractions: torch.Tensor,
        target_element_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composition vector similarity.

        Returns:
            Dict with 'cosine_similarity', 'composition_mse', 'composition_loss'
        """
        # Convert to dense composition vectors
        pred_comp = self.elements_to_composition_vector(
            pred_element_indices, pred_element_fractions, pred_element_mask
        )
        target_comp = self.elements_to_composition_vector(
            target_element_indices, target_element_fractions, target_element_mask
        )

        # Cosine similarity (1 = identical, 0 = orthogonal)
        cosine_sim = F.cosine_similarity(pred_comp, target_comp, dim=1)

        # MSE on composition vectors
        composition_mse = F.mse_loss(pred_comp, target_comp)

        # Combined loss (1 - similarity to make it minimizable)
        composition_loss = (1 - cosine_sim.mean()) * self.config.composition_similarity_weight

        return {
            'cosine_similarity': cosine_sim.mean(),
            'composition_mse': composition_mse,
            'composition_loss': composition_loss,
        }


@torch.no_grad()
def compute_stoichiometry_reward(
    pred_fractions: torch.Tensor,        # [batch, max_elements]
    target_fractions: torch.Tensor,      # [batch, max_elements]
    element_mask: torch.Tensor,          # [batch, max_elements]
    max_reward: float = 20.0,
    error_scale: float = 5.0,
) -> torch.Tensor:
    """
    Compute REINFORCE reward based on stoichiometry accuracy.

    This reward is ADDITIVE to the existing token-level rewards. It provides
    extra signal for getting stoichiometry numerically correct.

    Reward formula: max_reward * exp(-error_scale * mean_abs_error)

    Args:
        pred_fractions: Predicted element fractions
        target_fractions: Ground truth element fractions
        element_mask: Valid element mask
        max_reward: Maximum possible stoichiometry reward
        error_scale: Scaling factor (higher = more sensitive to errors)

    Returns:
        rewards: [batch] tensor of stoichiometry rewards
    """
    mask_float = element_mask.float()

    # Absolute error per element
    abs_error = (pred_fractions - target_fractions).abs()
    masked_abs_error = abs_error * mask_float

    # Mean absolute error per sample
    n_valid = mask_float.sum(dim=1).clamp(min=1)
    mean_abs_error = masked_abs_error.sum(dim=1) / n_valid

    # Exponential reward: perfect stoichiometry = max_reward, errors decrease reward
    rewards = max_reward * torch.exp(-error_scale * mean_abs_error)

    return rewards


class CombinedStoichiometryLoss(nn.Module):
    """
    Combined stoichiometry-aware loss for training.

    Aggregates:
    1. StoichiometryMSELoss - direct fraction prediction
    2. CompositionVectorLoss - overall composition similarity (optional)

    Use this in the training loop alongside token-level CE and REINFORCE.
    """

    def __init__(
        self,
        config: Optional[StoichiometryLossConfig] = None,
        use_composition_loss: bool = False,  # Optional composition vector loss
        n_elements: int = 118,
    ):
        super().__init__()
        self.config = config or StoichiometryLossConfig()
        self.use_composition_loss = use_composition_loss

        self.stoich_mse = StoichiometryMSELoss(config=self.config)

        if use_composition_loss:
            self.composition_loss = CompositionVectorLoss(n_elements, config=self.config)
        else:
            self.composition_loss = None

    def forward(
        self,
        fraction_pred: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        element_count_pred: Optional[torch.Tensor] = None,
        element_indices: Optional[torch.Tensor] = None,  # For composition loss
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined stoichiometry loss.

        Args:
            fraction_pred: From encoder's fraction_head
            element_fractions: Ground truth (input to encoder)
            element_mask: Valid element mask
            element_count_pred: Optional element count prediction
            element_indices: Required if use_composition_loss=True

        Returns:
            Dict with all loss components and 'total' combined loss
        """
        # Main stoichiometry MSE loss
        losses = self.stoich_mse(
            fraction_pred, element_fractions, element_mask, element_count_pred
        )

        # Optional composition vector loss
        if self.use_composition_loss and element_indices is not None:
            # For composition loss, we compare predicted fractions against target
            # Assuming pred uses same element indices as target (just different fractions)
            comp_losses = self.composition_loss(
                element_indices, fraction_pred, element_mask,
                element_indices, element_fractions, element_mask,
            )
            losses.update(comp_losses)
            losses['total'] = losses['stoich_total'] + losses['composition_loss']
        else:
            losses['total'] = losses['stoich_total']

        return losses
