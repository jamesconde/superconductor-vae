"""
Consistency Losses for Superconductor VAE.

These losses ensure the encoder-decoder loop produces coherent property predictions:
- Self-consistency: Original properties match reconstructed properties
- Bidirectional: Forward-backward validation of property predictions

February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ConsistencyLossConfig:
    """Configuration for consistency losses."""

    # Weight for Tc consistency (primary property)
    tc_weight: float = 1.0

    # Weight for Magpie feature consistency (secondary)
    magpie_weight: float = 0.1

    # Whether to normalize Magpie features before comparison
    normalize_magpie: bool = True

    # Huber loss delta (for robustness to outliers)
    huber_delta: float = 1.0

    # Whether to use Huber loss instead of MSE
    use_huber: bool = False


class SelfConsistencyLoss(nn.Module):
    """
    Self-consistency loss comparing original vs reconstructed properties.

    Ensures that when we encode a material and decode it back, the predicted
    properties (Tc, Magpie features) remain consistent with the originals.

    This encourages the latent space to preserve property information
    throughout the encode-decode cycle.
    """

    def __init__(self, config: Optional[ConsistencyLossConfig] = None):
        super().__init__()
        self.config = config or ConsistencyLossConfig()

    def forward(
        self,
        original_tc: torch.Tensor,          # [batch] or [batch, 1]
        reconstructed_tc: torch.Tensor,     # [batch] or [batch, 1]
        original_magpie: Optional[torch.Tensor] = None,    # [batch, 145]
        reconstructed_magpie: Optional[torch.Tensor] = None,  # [batch, 145]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute self-consistency loss.

        Args:
            original_tc: Ground truth Tc values
            reconstructed_tc: Tc predicted from reconstructed formula
            original_magpie: Ground truth Magpie features
            reconstructed_magpie: Magpie features from reconstruction

        Returns:
            Dict with 'tc_consistency', 'magpie_consistency', 'total'
        """
        # Ensure tensors are properly shaped
        if original_tc.dim() == 2:
            original_tc = original_tc.squeeze(-1)
        if reconstructed_tc.dim() == 2:
            reconstructed_tc = reconstructed_tc.squeeze(-1)

        # Tc consistency loss
        if self.config.use_huber:
            tc_loss = F.huber_loss(
                reconstructed_tc, original_tc,
                delta=self.config.huber_delta
            )
        else:
            tc_loss = F.mse_loss(reconstructed_tc, original_tc)

        tc_loss = tc_loss * self.config.tc_weight

        losses = {'tc_consistency': tc_loss}

        # Magpie consistency loss (optional)
        if original_magpie is not None and reconstructed_magpie is not None:
            if self.config.normalize_magpie:
                # Normalize to unit vectors for cosine-like comparison
                original_norm = F.normalize(original_magpie, p=2, dim=-1)
                recon_norm = F.normalize(reconstructed_magpie, p=2, dim=-1)
                magpie_loss = F.mse_loss(recon_norm, original_norm)
            else:
                magpie_loss = F.mse_loss(reconstructed_magpie, original_magpie)

            magpie_loss = magpie_loss * self.config.magpie_weight
            losses['magpie_consistency'] = magpie_loss
        else:
            losses['magpie_consistency'] = torch.tensor(0.0, device=original_tc.device)

        losses['total'] = losses['tc_consistency'] + losses['magpie_consistency']
        return losses


class BidirectionalConsistencyLoss(nn.Module):
    """
    Forward-backward consistency loss.

    Validates that:
    1. Original formula -> encode -> z -> predict properties
    2. z -> decode -> reconstructed formula -> predict properties
    3. Properties from (1) and (2) should match

    This is stronger than self-consistency because it validates
    through the full generation pipeline.
    """

    def __init__(
        self,
        tc_predictor: Optional[nn.Module] = None,
        config: Optional[ConsistencyLossConfig] = None,
    ):
        """
        Args:
            tc_predictor: A frozen model that predicts Tc from formula tokens.
                         If None, must provide predicted Tc externally.
            config: Loss configuration
        """
        super().__init__()
        self.tc_predictor = tc_predictor
        self.config = config or ConsistencyLossConfig()

        # Freeze the predictor if provided
        if self.tc_predictor is not None:
            for param in self.tc_predictor.parameters():
                param.requires_grad = False

    def forward(
        self,
        original_tc: torch.Tensor,                  # [batch]
        reconstructed_formula_tokens: torch.Tensor,  # [batch, seq_len]
        formula_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
        pred_tc_from_reconstruction: Optional[torch.Tensor] = None,  # [batch]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute bidirectional consistency loss.

        Args:
            original_tc: Ground truth Tc
            reconstructed_formula_tokens: Tokens from decoder
            formula_mask: Mask for valid tokens (non-padding)
            pred_tc_from_reconstruction: If provided, skip internal prediction

        Returns:
            Dict with 'bidirectional_consistency', 'tc_error_mean', 'tc_error_std'
        """
        if pred_tc_from_reconstruction is None:
            if self.tc_predictor is None:
                raise ValueError(
                    "Either provide tc_predictor in __init__ or "
                    "pred_tc_from_reconstruction in forward()"
                )
            # Predict Tc from reconstructed formula
            with torch.no_grad():
                pred_tc_from_reconstruction = self.tc_predictor(
                    reconstructed_formula_tokens, formula_mask
                )

        # Ensure proper shape
        if original_tc.dim() == 2:
            original_tc = original_tc.squeeze(-1)
        if pred_tc_from_reconstruction.dim() == 2:
            pred_tc_from_reconstruction = pred_tc_from_reconstruction.squeeze(-1)

        # Compute consistency loss
        tc_error = (original_tc - pred_tc_from_reconstruction).abs()

        if self.config.use_huber:
            consistency_loss = F.huber_loss(
                pred_tc_from_reconstruction, original_tc,
                delta=self.config.huber_delta
            )
        else:
            consistency_loss = F.mse_loss(
                pred_tc_from_reconstruction, original_tc
            )

        consistency_loss = consistency_loss * self.config.tc_weight

        return {
            'bidirectional_consistency': consistency_loss,
            'tc_error_mean': tc_error.mean(),
            'tc_error_std': tc_error.std() if tc_error.numel() > 1 else torch.tensor(0.0),
        }


class CombinedConsistencyLoss(nn.Module):
    """
    Combined self-consistency and bidirectional consistency loss.

    Use this as the main consistency loss in training.
    """

    def __init__(
        self,
        config: Optional[ConsistencyLossConfig] = None,
        use_bidirectional: bool = True,
        tc_predictor: Optional[nn.Module] = None,
        bidirectional_weight: float = 0.5,
    ):
        super().__init__()
        self.config = config or ConsistencyLossConfig()
        self.use_bidirectional = use_bidirectional
        self.bidirectional_weight = bidirectional_weight

        self.self_consistency = SelfConsistencyLoss(config=self.config)

        if use_bidirectional:
            self.bidirectional = BidirectionalConsistencyLoss(
                tc_predictor=tc_predictor,
                config=self.config,
            )
        else:
            self.bidirectional = None

    def forward(
        self,
        original_tc: torch.Tensor,
        reconstructed_tc: torch.Tensor,
        original_magpie: Optional[torch.Tensor] = None,
        reconstructed_magpie: Optional[torch.Tensor] = None,
        reconstructed_formula_tokens: Optional[torch.Tensor] = None,
        formula_mask: Optional[torch.Tensor] = None,
        pred_tc_from_reconstruction: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined consistency loss.

        Returns:
            Dict with all component losses and 'total'
        """
        # Self-consistency
        self_losses = self.self_consistency(
            original_tc, reconstructed_tc,
            original_magpie, reconstructed_magpie,
        )

        losses = {
            'self_tc_consistency': self_losses['tc_consistency'],
            'self_magpie_consistency': self_losses['magpie_consistency'],
        }

        total = self_losses['total']

        # Bidirectional consistency (optional)
        if self.use_bidirectional and reconstructed_formula_tokens is not None:
            bidir_losses = self.bidirectional(
                original_tc,
                reconstructed_formula_tokens,
                formula_mask,
                pred_tc_from_reconstruction,
            )
            losses['bidirectional_consistency'] = bidir_losses['bidirectional_consistency']
            losses['tc_error_mean'] = bidir_losses['tc_error_mean']
            total = total + self.bidirectional_weight * bidir_losses['bidirectional_consistency']
        else:
            losses['bidirectional_consistency'] = torch.tensor(0.0, device=original_tc.device)

        losses['total'] = total
        return losses


# Convenience function for training integration
def compute_consistency_reward(
    original_tc: torch.Tensor,
    predicted_tc: torch.Tensor,
    max_reward: float = 10.0,
    error_scale: float = 0.1,
) -> torch.Tensor:
    """
    Compute REINFORCE reward based on Tc consistency.

    Reward formula: max_reward * exp(-error_scale * |original - predicted|)

    Args:
        original_tc: Ground truth Tc
        predicted_tc: Tc predicted from generated formula
        max_reward: Maximum possible reward
        error_scale: Scaling factor (higher = more sensitive)

    Returns:
        rewards: [batch] tensor of consistency rewards
    """
    with torch.no_grad():
        if original_tc.dim() == 2:
            original_tc = original_tc.squeeze(-1)
        if predicted_tc.dim() == 2:
            predicted_tc = predicted_tc.squeeze(-1)

        tc_error = (original_tc - predicted_tc).abs()
        rewards = max_reward * torch.exp(-error_scale * tc_error)

    return rewards
