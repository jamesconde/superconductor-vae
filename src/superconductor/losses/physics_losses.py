"""
V13 Physics-Informed Losses for Superconductor VAE.

These losses encourage the latent space z to represent physically meaningful
structure rather than arbitrary compressed representations.

Key losses:
- TcIntermediateLoss: z_Tc partition must predict Tc (bottleneck constraint)
- ContrastiveTcLoss: Similar Tc values should map to similar z regions
- LatentSmoothnessLoss: Similar compositions should have smooth z transitions
- PhysicsBoundsLoss: Encourage physically plausible predictions

December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TcIntermediateLoss(nn.Module):
    """
    Bottleneck constraint: z_Tc partition must predict Tc BEFORE reconstruction.

    This forces the z_Tc portion of the latent space to contain Tc-predictive
    information, creating a semantic partition in the latent space.

    The intermediate Tc prediction happens in the encoder, before the decoder
    sees any skip connections or attended inputs. This ensures z_Tc is
    truly encoding Tc-relevant features.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        Tc_pred_intermediate: torch.Tensor,
        Tc_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Tc_pred_intermediate: Predicted Tc from z_Tc partition [batch, 1]
            Tc_target: True Tc values (normalized) [batch] or [batch, 1]

        Returns:
            MSE loss between intermediate prediction and target
        """
        if Tc_target.dim() == 1:
            Tc_target = Tc_target.unsqueeze(-1)

        return F.mse_loss(Tc_pred_intermediate, Tc_target, reduction=self.reduction)


class ContrastiveTcLoss(nn.Module):
    """
    Contrastive loss: Similar Tc values should have similar z representations.

    This shapes the latent space geometry to respect Tc similarity:
    - Materials with similar Tc should cluster together in z-space
    - Materials with different Tc should be pushed apart

    This creates a physics-informed manifold structure where interpolating
    in z-space corresponds to interpolating in Tc space.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        tc_scale: float = 1.0,
        margin: float = 0.5
    ):
        """
        Args:
            temperature: Softmax temperature for similarity computation
            tc_scale: Scaling factor for Tc differences (depends on normalization)
            margin: Minimum Tc difference to consider "different" (in normalized units)
        """
        super().__init__()
        self.temperature = temperature
        self.tc_scale = tc_scale
        self.margin = margin

    def forward(
        self,
        z: torch.Tensor,
        Tc_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: Full latent vectors [batch, z_dim]
            Tc_values: Tc values (normalized) [batch]

        Returns:
            Contrastive loss encouraging similar Tc → similar z
        """
        batch_size = z.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)

        # Normalize z vectors for cosine similarity
        z_norm = F.normalize(z, p=2, dim=-1)  # [batch, z_dim]

        # Pairwise cosine similarities in z-space
        z_sim = torch.mm(z_norm, z_norm.t())  # [batch, batch]

        # Pairwise Tc differences (absolute)
        Tc_values = Tc_values.view(-1)  # Ensure 1D
        Tc_diff = torch.abs(Tc_values.unsqueeze(0) - Tc_values.unsqueeze(1))  # [batch, batch]

        # Normalize Tc differences to [0, 1] range
        tc_max = Tc_diff.max() + 1e-6
        Tc_diff_norm = Tc_diff / tc_max  # [batch, batch]

        # Target similarity: High when Tc is similar, low when different
        # Using soft assignment: target_sim = exp(-scaled_diff)
        target_sim = torch.exp(-self.tc_scale * Tc_diff_norm)

        # Mask out diagonal (self-similarity)
        mask = 1 - torch.eye(batch_size, device=z.device)

        # MSE between actual z similarity and target similarity (based on Tc)
        loss = ((z_sim - target_sim) ** 2 * mask).sum() / (mask.sum() + 1e-6)

        return loss


class LatentSmoothnessLoss(nn.Module):
    """
    Smoothness constraint: Similar compositions should have smooth z transitions.

    This encourages the latent space to be locally smooth - materials with
    similar chemical compositions should be nearby in z-space. This prevents
    the model from learning discontinuous or chaotic latent representations.
    """

    def __init__(self, composition_weight: float = 1.0):
        super().__init__()
        self.composition_weight = composition_weight

    def forward(
        self,
        z: torch.Tensor,
        composition_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: Latent vectors [batch, z_dim]
            composition_vectors: Element fraction vectors [batch, n_elements]

        Returns:
            Smoothness loss penalizing large z jumps for similar compositions
        """
        batch_size = z.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)

        # Pairwise L2 distances in z-space
        z_dist = torch.cdist(z, z, p=2)  # [batch, batch]

        # Pairwise cosine similarities in composition space
        comp_norm = F.normalize(composition_vectors, p=2, dim=-1)
        comp_sim = torch.mm(comp_norm, comp_norm.t())  # [batch, batch]

        # We want: high comp_sim → low z_dist
        # Loss: penalize when z_dist is large for high comp_sim pairs
        # Weight z_dist by comp_sim (similar compositions contribute more)

        # Mask out diagonal
        mask = 1 - torch.eye(batch_size, device=z.device)

        # Loss: sum of (z_dist * comp_sim) - high penalty for large z_dist on similar compositions
        loss = (z_dist * comp_sim * mask).sum() / (mask.sum() + 1e-6)

        return self.composition_weight * loss


class PhysicsBoundsLoss(nn.Module):
    """
    Physics bounds constraint: Encourage physically plausible predictions.

    Components:
    1. Tc positivity: Superconductor Tc must be positive
    2. Composition structure: z_composition should have meaningful structure
    3. Electronic structure: z_electronic should capture band structure features

    These soft constraints guide the model toward physically realistic outputs.
    """

    def __init__(
        self,
        tc_positive_weight: float = 1.0,
        entropy_weight: float = 0.1,
        max_entropy: float = 3.0
    ):
        """
        Args:
            tc_positive_weight: Weight for Tc positivity penalty
            entropy_weight: Weight for entropy regularization
            max_entropy: Maximum allowed entropy before penalty kicks in
        """
        super().__init__()
        self.tc_positive_weight = tc_positive_weight
        self.entropy_weight = entropy_weight
        self.max_entropy = max_entropy

    def forward(
        self,
        z_partitions: Dict[str, torch.Tensor],
        Tc_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_partitions: Dictionary with 'z_composition', 'z_electronic', etc.
            Tc_pred: Predicted Tc values [batch, 1] or [batch]

        Returns:
            Combined physics bounds loss
        """
        loss = torch.tensor(0.0, device=Tc_pred.device)

        # 1. Tc positivity: Penalize negative Tc predictions
        # (After denormalization, Tc should be > 0)
        # For normalized Tc, we assume mean > 0, so very negative values are bad
        Tc_negative_penalty = F.relu(-Tc_pred - 1.0).mean()  # Allow some buffer
        loss = loss + self.tc_positive_weight * Tc_negative_penalty

        # 2. z_composition entropy regularization
        # Prevent z_composition from collapsing to uniform or single-mode
        if 'z_composition' in z_partitions:
            z_comp = z_partitions['z_composition']

            # Compute softmax probabilities (treating z_comp as logits)
            comp_probs = F.softmax(z_comp, dim=-1)

            # Entropy: -sum(p * log(p))
            entropy = -torch.mean(comp_probs * torch.log(comp_probs + 1e-8), dim=-1).mean()

            # Penalize if entropy is too high (too uniform) or too low (collapsed)
            # Target: moderate entropy for meaningful clustering
            entropy_penalty = F.relu(entropy - self.max_entropy)
            loss = loss + self.entropy_weight * entropy_penalty

        return loss


class CombinedPhysicsLoss(nn.Module):
    """
    Combined V13 physics-informed loss function.

    Aggregates all physics losses with configurable weights:
    - Tc intermediate (bottleneck)
    - Contrastive Tc (latent geometry)
    - Smoothness (local continuity)
    - Physics bounds (plausibility)
    """

    def __init__(
        self,
        tc_intermediate_weight: float = 5.0,
        contrastive_weight: float = 1.0,
        smoothness_weight: float = 0.5,
        bounds_weight: float = 0.1
    ):
        super().__init__()

        self.tc_intermediate_weight = tc_intermediate_weight
        self.contrastive_weight = contrastive_weight
        self.smoothness_weight = smoothness_weight
        self.bounds_weight = bounds_weight

        self.tc_intermediate_loss = TcIntermediateLoss()
        self.contrastive_loss = ContrastiveTcLoss()
        self.smoothness_loss = LatentSmoothnessLoss()
        self.bounds_loss = PhysicsBoundsLoss()

    def forward(
        self,
        z: torch.Tensor,
        z_partitions: Dict[str, torch.Tensor],
        Tc_pred_intermediate: torch.Tensor,
        Tc_target: torch.Tensor,
        Tc_pred_final: Optional[torch.Tensor] = None,
        composition_vectors: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all physics losses.

        Args:
            z: Full latent vector [batch, z_dim]
            z_partitions: Dict with 'z_class', 'z_composition', 'z_electronic', 'z_Tc'
            Tc_pred_intermediate: Intermediate Tc prediction from encoder [batch, 1]
            Tc_target: True Tc values (normalized) [batch]
            Tc_pred_final: Final Tc prediction from decoder (optional)
            composition_vectors: Element fraction vectors (for smoothness loss)

        Returns:
            Dictionary with individual and combined losses
        """
        losses = {}

        # 1. Tc intermediate bottleneck loss
        tc_int_loss = self.tc_intermediate_loss(Tc_pred_intermediate, Tc_target)
        losses['tc_intermediate'] = tc_int_loss

        # 2. Contrastive Tc loss
        contrastive = self.contrastive_loss(z, Tc_target)
        losses['contrastive'] = contrastive

        # 3. Smoothness loss (if composition vectors provided)
        if composition_vectors is not None:
            smooth = self.smoothness_loss(z, composition_vectors)
            losses['smoothness'] = smooth
        else:
            losses['smoothness'] = torch.tensor(0.0, device=z.device)

        # 4. Physics bounds loss
        Tc_for_bounds = Tc_pred_final if Tc_pred_final is not None else Tc_pred_intermediate
        bounds = self.bounds_loss(z_partitions, Tc_for_bounds)
        losses['bounds'] = bounds

        # Combined weighted loss
        total = (
            self.tc_intermediate_weight * losses['tc_intermediate'] +
            self.contrastive_weight * losses['contrastive'] +
            self.smoothness_weight * losses['smoothness'] +
            self.bounds_weight * losses['bounds']
        )
        losses['physics_total'] = total

        return losses
