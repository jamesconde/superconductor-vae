"""
Attention-augmented BidirectionalVAE for superconductor discovery.

Integrates element-level attention INTO the VAE encoding path so that:
1. The model learns which elements are important for Tc prediction
2. The latent space captures element-weighted relationships
3. Attention weights are interpretable and grounded in Tc prediction

Architecture:
    Formula → Element embeddings → Attention → VAE encoder → Latent z → Tc predictor
                                      ↓
                              Attention weights (interpretable)

This differs from adding attention as a separate module - here attention
directly influences what the VAE's latent space learns about Tc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..encoders.element_attention import (
    ElementEmbedding,
    ElementAttention,
    AttentionOutput,
)
from ..encoders.isotope_encoder import IsotopeEncoder


@dataclass
class AttentionVAEOutput:
    """Output from AttentionVAE forward pass."""
    tc_pred: torch.Tensor              # [batch] predicted Tc
    z: torch.Tensor                    # [batch, latent_dim] latent representation
    z_mean: torch.Tensor               # [batch, latent_dim] latent mean
    z_logvar: torch.Tensor             # [batch, latent_dim] latent log variance
    reconstruction: torch.Tensor       # [batch, input_dim] reconstructed input
    competence: torch.Tensor           # [batch] model confidence
    attention_weights: torch.Tensor    # [batch, n_elements] element importance
    element_contributions: torch.Tensor  # [batch, n_elements] per-element Tc contribution


class ElementEncoder(nn.Module):
    """
    Encode elements with learnable embeddings and attention.

    This is the KEY integration point - attention happens HERE, before the VAE,
    so the latent space learns from attention-weighted representations.
    """

    def __init__(
        self,
        n_elements: int = 118,
        element_embed_dim: int = 64,
        n_attention_heads: int = 4,
        output_dim: int = 128,
        dropout: float = 0.1,
        use_isotope_features: bool = True
    ):
        super().__init__()

        self.n_elements = n_elements
        self.element_embed_dim = element_embed_dim
        self.use_isotope_features = use_isotope_features

        # Element embeddings (learnable)
        self.element_embedding = ElementEmbedding(
            n_elements=n_elements,
            embedding_dim=element_embed_dim,
            property_dim=11,  # Standard element properties
            use_properties=True
        )

        # Attention over elements
        self.element_attention = ElementAttention(
            hidden_dim=element_embed_dim,
            n_heads=n_attention_heads,
            dropout=dropout,
            temperature=1.0
        )

        # Isotope feature integration
        if use_isotope_features:
            # Isotope features: [mass_deviation, spin, abundance, isotope_effect]
            self.isotope_mlp = nn.Sequential(
                nn.Linear(4, element_embed_dim // 2),
                nn.GELU(),
                nn.Linear(element_embed_dim // 2, element_embed_dim)
            )

        # Project attended representation to output
        # Input: attended (embed_dim) + optional isotope (embed_dim)
        proj_input_dim = element_embed_dim * (2 if use_isotope_features else 1)
        self.output_projection = nn.Sequential(
            nn.Linear(proj_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.output_dim = output_dim

    def forward(
        self,
        element_indices: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        element_properties: Optional[torch.Tensor] = None,
        isotope_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode composition with attention.

        Args:
            element_indices: [batch, max_elements] atomic numbers (0 = padding)
            element_fractions: [batch, max_elements] molar fractions
            element_mask: [batch, max_elements] True for present elements
            element_properties: [batch, max_elements, prop_dim] optional properties
            isotope_features: [batch, 4] aggregated isotope features

        Returns:
            output: [batch, output_dim] attention-weighted representation
            attention_weights: [batch, max_elements] element importance
            element_embeddings: [batch, max_elements, embed_dim] for analysis
        """
        batch_size, max_elements = element_indices.shape

        # Get element embeddings
        element_embeds = self.element_embedding(element_indices, element_properties)

        # Weight by molar fraction (stoichiometry matters!)
        # This is CRITICAL - Cu3 should contribute 3x more than Y1
        fraction_weights = element_fractions.unsqueeze(-1)  # [batch, max_elem, 1]
        weighted_embeds = element_embeds * fraction_weights

        # Apply attention to learn which elements are MOST important
        attn_output = self.element_attention(weighted_embeds, element_mask)
        attended = attn_output.weighted_representation  # [batch, embed_dim]
        attention_weights = attn_output.attention_weights  # [batch, max_elem]

        # Add isotope features if available
        if self.use_isotope_features and isotope_features is not None:
            iso_embed = self.isotope_mlp(isotope_features)  # [batch, embed_dim]
            attended = torch.cat([attended, iso_embed], dim=-1)

        # Project to output dimension
        output = self.output_projection(attended)

        return output, attention_weights, element_embeds


class AttentionVAEEncoder(nn.Module):
    """VAE encoder that takes attention-weighted element representation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        deterministic: bool = False
    ):
        super().__init__()
        self.deterministic = deterministic

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(prev_dim, latent_dim)
        if not deterministic:
            self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        if self.deterministic:
            return self.fc_mean(h), None
        return self.fc_mean(h), self.fc_logvar(h)


class AttentionVAEDecoder(nn.Module):
    """VAE decoder for reconstruction."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ):
        super().__init__()

        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class TcPredictorWithContributions(nn.Module):
    """
    Predict Tc from latent space AND compute per-element contributions.

    This is key for interpretability - we can see how each element
    contributes to the final Tc prediction.
    """

    def __init__(
        self,
        latent_dim: int,
        element_embed_dim: int,
        hidden_dims: List[int]
    ):
        super().__init__()

        # Main Tc predictor from latent
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.predictor = nn.Sequential(*layers)

        # Per-element contribution head
        # Maps element embedding to scalar contribution
        self.element_contribution = nn.Sequential(
            nn.Linear(element_embed_dim, element_embed_dim // 2),
            nn.GELU(),
            nn.Linear(element_embed_dim // 2, 1)
        )

    def forward(
        self,
        z: torch.Tensor,
        element_embeddings: torch.Tensor,
        element_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict Tc and element contributions.

        Args:
            z: [batch, latent_dim] latent representation
            element_embeddings: [batch, max_elements, embed_dim]
            element_mask: [batch, max_elements]

        Returns:
            tc_pred: [batch] predicted Tc
            element_contributions: [batch, max_elements] per-element contributions
        """
        # Main prediction
        tc_pred = self.predictor(z).squeeze(-1)

        # Per-element contributions (for interpretability)
        contributions = self.element_contribution(element_embeddings).squeeze(-1)
        # Mask out padding
        contributions = contributions.masked_fill(~element_mask.bool(), 0.0)

        return tc_pred, contributions


class AttentionBidirectionalVAE(nn.Module):
    """
    Bidirectional VAE with integrated element attention.

    This model properly integrates attention into the VAE pipeline:
    1. Element embeddings capture per-element information
    2. Attention learns which elements are important for Tc
    3. Attended representation feeds into VAE encoder
    4. Latent space learns Tc-relevant features from attended input
    5. Tc predictor uses latent space for final prediction

    The attention weights are now GROUNDED in Tc prediction - they reflect
    what the model learned is important, not just arbitrary weights.
    """

    def __init__(
        self,
        n_elements: int = 118,
        element_embed_dim: int = 64,
        n_attention_heads: int = 4,
        encoder_input_dim: int = 128,
        encoder_hidden: List[int] = [128, 64],
        latent_dim: int = 32,
        decoder_hidden: List[int] = [64, 128],
        predictor_hidden: List[int] = [32, 16],
        use_isotope_features: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_elements = n_elements
        self.latent_dim = latent_dim
        self.use_isotope_features = use_isotope_features

        # Element encoder with attention
        self.element_encoder = ElementEncoder(
            n_elements=n_elements,
            element_embed_dim=element_embed_dim,
            n_attention_heads=n_attention_heads,
            output_dim=encoder_input_dim,
            dropout=dropout,
            use_isotope_features=use_isotope_features
        )

        # VAE encoder
        self.vae_encoder = AttentionVAEEncoder(
            input_dim=encoder_input_dim,
            hidden_dims=encoder_hidden,
            latent_dim=latent_dim
        )

        # VAE decoder (reconstructs the attended representation)
        self.vae_decoder = AttentionVAEDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            output_dim=encoder_input_dim
        )

        # Tc predictor with element contributions
        self.tc_predictor = TcPredictorWithContributions(
            latent_dim=latent_dim,
            element_embed_dim=element_embed_dim,
            hidden_dims=predictor_hidden
        )

        # Competence head (model confidence)
        self.competence_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        element_indices: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        element_properties: Optional[torch.Tensor] = None,
        isotope_features: Optional[torch.Tensor] = None,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with attention-weighted encoding.

        Args:
            element_indices: [batch, max_elements] atomic numbers
            element_fractions: [batch, max_elements] molar fractions
            element_mask: [batch, max_elements] valid element mask
            element_properties: [batch, max_elements, prop_dim] optional
            isotope_features: [batch, 4] optional isotope features
            return_all: Whether to return all outputs

        Returns:
            Dict with tc_pred, z, attention_weights, element_contributions, etc.
        """
        # Encode with attention
        attended, attention_weights, element_embeds = self.element_encoder(
            element_indices, element_fractions, element_mask,
            element_properties, isotope_features
        )

        # VAE encoding
        z_mean, z_logvar = self.vae_encoder(attended)
        z = self.reparameterize(z_mean, z_logvar)

        # Decode (reconstruct attended representation)
        reconstruction = self.vae_decoder(z)

        # Predict Tc with element contributions
        tc_pred, element_contributions = self.tc_predictor(
            z, element_embeds, element_mask
        )

        # Competence
        competence = self.competence_head(z).squeeze(-1)

        if return_all:
            return {
                'tc_pred': tc_pred,
                'z': z,
                'z_mean': z_mean,
                'z_logvar': z_logvar,
                'reconstruction': reconstruction,
                'attended_input': attended,
                'competence': competence,
                'attention_weights': attention_weights,
                'element_contributions': element_contributions,
                'element_embeddings': element_embeds,
            }
        else:
            return {'tc_pred': tc_pred, 'competence': competence}

    def encode(
        self,
        element_indices: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        element_properties: Optional[torch.Tensor] = None,
        isotope_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode to latent space."""
        attended, _, _ = self.element_encoder(
            element_indices, element_fractions, element_mask,
            element_properties, isotope_features
        )
        z_mean, _ = self.vae_encoder(attended)
        return z_mean

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space (returns attended representation)."""
        return self.vae_decoder(z)

    def predict_tc_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Predict Tc directly from latent (without element contributions)."""
        return self.tc_predictor.predictor(z).squeeze(-1)


class AttentionVAELoss(nn.Module):
    """
    Loss function for AttentionBidirectionalVAE.

    Components:
    1. Tc prediction loss (MSE)
    2. Reconstruction loss (MSE on attended representation)
    3. KL divergence (VAE regularization)
    4. Attention entropy (optional - encourage sparse attention)
    5. Contribution consistency (element contributions should sum to prediction)
    """

    def __init__(
        self,
        prediction_weight: float = 1.0,
        reconstruction_weight: float = 0.1,
        kl_weight: float = 0.01,
        attention_entropy_weight: float = 0.01,
        contribution_weight: float = 0.1
    ):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.attention_entropy_weight = attention_entropy_weight
        self.contribution_weight = contribution_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        tc_true: torch.Tensor,
        element_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            outputs: Dict from AttentionBidirectionalVAE.forward()
            tc_true: [batch] true Tc values (normalized)
            element_mask: [batch, max_elements] valid element mask

        Returns:
            Dict with total loss and components
        """
        # Prediction loss
        pred_loss = F.mse_loss(outputs['tc_pred'], tc_true)

        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['reconstruction'], outputs['attended_input'])

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + outputs['z_logvar'] - outputs['z_mean'].pow(2) - outputs['z_logvar'].exp()
        )

        # Attention entropy (encourage learning, not uniform attention)
        # Low entropy = model learned to focus on specific elements
        attn = outputs['attention_weights']
        attn_entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean()

        # Element contribution consistency
        # Sum of element contributions should relate to predicted Tc
        contributions = outputs['element_contributions']
        contrib_sum = contributions.sum(dim=-1)
        contrib_loss = F.mse_loss(contrib_sum, outputs['tc_pred'])

        # Total loss
        total = (
            self.prediction_weight * pred_loss +
            self.reconstruction_weight * recon_loss +
            self.kl_weight * kl_loss +
            self.attention_entropy_weight * attn_entropy +
            self.contribution_weight * contrib_loss
        )

        return {
            'total': total,
            'prediction': pred_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss,
            'attention_entropy': attn_entropy,
            'contribution': contrib_loss
        }


def create_attention_vae(
    n_elements: int = 118,
    latent_dim: int = 32,
    use_isotope_features: bool = True
) -> AttentionBidirectionalVAE:
    """Factory function for AttentionBidirectionalVAE with good defaults."""
    return AttentionBidirectionalVAE(
        n_elements=n_elements,
        element_embed_dim=64,
        n_attention_heads=4,
        encoder_input_dim=128,
        encoder_hidden=[128, 64],
        latent_dim=latent_dim,
        decoder_hidden=[64, 128],
        predictor_hidden=[32, 16],
        use_isotope_features=use_isotope_features,
        dropout=0.1
    )


# ============================================================================
# V12.33: HIERARCHICAL FAMILY CLASSIFICATION HEAD
# ============================================================================

class HierarchicalFamilyHead(nn.Module):
    """
    V12.33: 3-level hierarchical family classification conditioned on sc_pred.

    Replaces the flat 14-class family_head with a tree structure:
        Level 0: NOT_SC = 1 - P(SC)                     (from sc_head, not learned here)
        Level 1: Coarse family (7 classes, SC samples only)
            0: BCS_CONVENTIONAL  (fine class 1)
            1: CUPRATE           (fine classes 2-7)
            2: IRON              (fine classes 8-9)
            3: MGB2              (fine class 10)
            4: HEAVY_FERMION     (fine class 11)
            5: ORGANIC           (fine class 12)
            6: OTHER_UNKNOWN     (fine class 13)
        Level 2a: Cuprate sub-family (6 classes, cuprate samples only)
            0: YBCO(2), 1: LSCO(3), 2: BSCCO(4), 3: TBCCO(5), 4: HBCCO(6), 5: OTHER(7)
        Level 2b: Iron sub-family (2 classes, iron samples only)
            0: PNICTIDE(8), 1: CHALCOGENIDE(9)

    Composed 14-class probability:
        P(NOT_SC)   = 1 - P(SC)
        P(BCS)      = P(SC) * P(BCS|SC)
        P(YBCO)     = P(SC) * P(Cuprate|SC) * P(YBCO|Cuprate)
        P(PNICTIDE) = P(SC) * P(Iron|SC)    * P(Pnictide|Iron)
        etc.
    """

    def __init__(self, backbone_dim: int = 512, dropout: float = 0.1):
        super().__init__()

        # Coarse head: h(backbone_dim) + sc_prob(1) → 256 → 128 → 7
        self.coarse_head = nn.Sequential(
            nn.Linear(backbone_dim + 1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 7),
        )

        # Cuprate sub-head: h(backbone_dim) + sc_prob(1) → 128 → 64 → 6
        self.cuprate_sub_head = nn.Sequential(
            nn.Linear(backbone_dim + 1, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )

        # Iron sub-head: h(backbone_dim) + sc_prob(1) → 64 → 2
        self.iron_sub_head = nn.Sequential(
            nn.Linear(backbone_dim + 1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        h: torch.Tensor,
        sc_pred_detached: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical family classification conditioned on sc_pred.

        Args:
            h: [batch, backbone_dim] decoder backbone output
            sc_pred_detached: [batch] sc_head logits (DETACHED — no gradient to sc_head)

        Returns:
            Dict with coarse_logits [batch, 7], cuprate_sub_logits [batch, 6],
            iron_sub_logits [batch, 2], composed_14 [batch, 14] probabilities.
        """
        batch_size = h.shape[0]
        device = h.device

        # Condition on sc_pred probability (sigmoid of detached logit)
        sc_prob = torch.sigmoid(sc_pred_detached).unsqueeze(-1)  # [batch, 1]
        conditioned = torch.cat([h, sc_prob], dim=-1)  # [batch, backbone_dim + 1]

        # Level 1: Coarse family logits (7 classes, for SC samples)
        coarse_logits = self.coarse_head(conditioned)  # [batch, 7]

        # Level 2a: Cuprate sub-family logits (6 classes)
        cuprate_sub_logits = self.cuprate_sub_head(conditioned)  # [batch, 6]

        # Level 2b: Iron sub-family logits (2 classes)
        iron_sub_logits = self.iron_sub_head(conditioned)  # [batch, 2]

        # Compose 14-class probability distribution
        coarse_probs = F.softmax(coarse_logits, dim=-1)  # [batch, 7]
        cuprate_sub_probs = F.softmax(cuprate_sub_logits, dim=-1)  # [batch, 6]
        iron_sub_probs = F.softmax(iron_sub_logits, dim=-1)  # [batch, 2]

        composed = torch.zeros(batch_size, 14, device=device)
        sc_prob_squeezed = sc_prob.squeeze(-1)  # [batch]

        # Class 0: NOT_SC = 1 - P(SC)
        composed[:, 0] = 1.0 - sc_prob_squeezed

        # Class 1: BCS_CONVENTIONAL = P(SC) * P(BCS|SC)
        composed[:, 1] = sc_prob_squeezed * coarse_probs[:, 0]

        # Classes 2-7: Cuprate sub-families = P(SC) * P(Cuprate|SC) * P(sub|Cuprate)
        cuprate_prob = sc_prob_squeezed * coarse_probs[:, 1]  # [batch]
        composed[:, 2:8] = cuprate_prob.unsqueeze(-1) * cuprate_sub_probs  # [batch, 6]

        # Classes 8-9: Iron sub-families = P(SC) * P(Iron|SC) * P(sub|Iron)
        iron_prob = sc_prob_squeezed * coarse_probs[:, 2]  # [batch]
        composed[:, 8:10] = iron_prob.unsqueeze(-1) * iron_sub_probs  # [batch, 2]

        # Class 10: MGB2 = P(SC) * P(MGB2|SC)
        composed[:, 10] = sc_prob_squeezed * coarse_probs[:, 3]

        # Class 11: HEAVY_FERMION = P(SC) * P(HF|SC)
        composed[:, 11] = sc_prob_squeezed * coarse_probs[:, 4]

        # Class 12: ORGANIC = P(SC) * P(Organic|SC)
        composed[:, 12] = sc_prob_squeezed * coarse_probs[:, 5]

        # Class 13: OTHER_UNKNOWN = P(SC) * P(Other|SC)
        composed[:, 13] = sc_prob_squeezed * coarse_probs[:, 6]

        return {
            'coarse_logits': coarse_logits,
            'cuprate_sub_logits': cuprate_sub_logits,
            'iron_sub_logits': iron_sub_logits,
            'composed_14': composed,
        }


# ============================================================================
# V12: FULL MATERIALS VAE - Encodes entire supercon dataset
# ============================================================================

class FullMaterialsVAE(nn.Module):
    """
    V12 Full Materials VAE - Encodes the ENTIRE superconductor dataset.

    This is the foundation for combining with theory networks. The latent space
    should encode everything knowable about a superconductor from empirical data.

    ENCODER INPUTS:
        - Element composition (indices, fractions, mask) - structural info
        - Magpie features (145 dims) - derived material properties
        - Tc (1 dim) - critical temperature

    DECODER OUTPUTS:
        - Tc prediction (1 dim) - reconstruct critical temperature
        - Magpie features (145 dims) - reconstruct material properties
        - Formula tokens (handled by external EnhancedTransformerDecoder)

    The latent space z (2048 dims) encodes a COMPLETE representation of the
    superconductor that can be:
        1. Decoded back to observables (formula, Tc, properties)
        2. Combined with theory networks (BCS, etc.)
        3. Interpolated/sampled to discover new materials

    Architecture:
        ┌─ Element attention (256 dim) ─┐
        │                                │
        ├─ Magpie MLP (256 dim) ────────┼→ Fusion (768) → VAE Encoder → z (2048)
        │                                │
        └─ Tc embedding (256 dim) ──────┘
                                          ↓
                                   z (2048) → Multi-head Decoder
                                          ├→ Tc head (1)
                                          ├→ Magpie head (145)
                                          └→ attended_input for formula decoder
    """

    def __init__(
        self,
        n_elements: int = 118,
        element_embed_dim: int = 128,
        n_attention_heads: int = 8,
        magpie_dim: int = 145,
        fusion_dim: int = 256,  # Each branch outputs this dim
        encoder_hidden: List[int] = [512, 256],
        latent_dim: int = 2048,
        decoder_hidden: List[int] = [256, 512],
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_elements = n_elements
        self.latent_dim = latent_dim
        self.magpie_dim = magpie_dim
        self.fusion_dim = fusion_dim

        # =====================================================================
        # ENCODER BRANCHES
        # =====================================================================

        # Branch 1: Element composition with attention (existing architecture)
        self.element_encoder = ElementEncoder(
            n_elements=n_elements,
            element_embed_dim=element_embed_dim,
            n_attention_heads=n_attention_heads,
            output_dim=fusion_dim,
            dropout=dropout,
            use_isotope_features=False  # We have Magpie features instead
        )

        # Branch 2: Magpie features MLP
        self.magpie_encoder = nn.Sequential(
            nn.Linear(magpie_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Branch 3: Tc embedding (single value to representation)
        self.tc_encoder = nn.Sequential(
            nn.Linear(1, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Fusion layer: combine all branches
        # Input: element (fusion_dim) + magpie (fusion_dim) + tc (fusion_dim)
        total_fusion_dim = fusion_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(total_fusion_dim, total_fusion_dim),
            nn.LayerNorm(total_fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # VAE encoder (from fused representation to latent)
        # Deterministic mode: z = fc_mean(h) directly, no reparameterization noise
        self.vae_encoder = AttentionVAEEncoder(
            input_dim=total_fusion_dim,
            hidden_dims=encoder_hidden,
            latent_dim=latent_dim,
            deterministic=True
        )

        # =====================================================================
        # DECODER HEADS
        # =====================================================================

        # Shared decoder backbone
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.decoder_backbone = nn.Sequential(*decoder_layers)

        # Head 1: Tc prediction (V12.28 — deeper residual MLP with Net2Net transfer)
        # Old: tc_head = Linear(512→256) → GELU → Linear(256→1)
        # New: tc_proj(512→256) + residual_block(256→256) + tc_out(256→128→1)
        # Net2Net: tc_proj inherits tc_head.0 weights; residual block identity-init;
        #          tc_out.2 (Linear 256→128) is new; tc_out.4 (Linear 128→1) gets
        #          tc_head.2 weights via net2net wider expansion in checkpoint loading.
        self.tc_proj = nn.Linear(prev_dim, 256)  # Project to residual dim
        self.tc_res_block = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
        )
        # Initialize residual block as near-identity so initial behavior ≈ old tc_head
        with torch.no_grad():
            nn.init.eye_(self.tc_res_block[0].weight)
            nn.init.zeros_(self.tc_res_block[0].bias)
            nn.init.eye_(self.tc_res_block[4].weight)
            nn.init.zeros_(self.tc_res_block[4].bias)
        self.tc_out = nn.Sequential(
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Head 2: Magpie feature reconstruction
        self.magpie_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.GELU(),
            nn.Linear(prev_dim, magpie_dim)
        )

        # Head 3: Attended representation for formula decoder
        # This outputs the "attended_input" that EnhancedTransformerDecoder uses
        self.attended_head = nn.Sequential(
            nn.Linear(prev_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Competence head (model confidence)
        self.competence_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, 1),
            nn.Sigmoid()
        )

        # =====================================================================
        # FRACTION VALUE HEAD (V12.4 - Stoichiometry-Aware Learning)
        # =====================================================================
        # Predicts the numerical value of element fractions directly from z.
        # This provides a gradient signal that distinguishes:
        #   - 1/5 vs 2/5 (small error: |0.2 - 0.4| = 0.2)
        #   - 1/5 vs 9/5 (large error: |0.2 - 1.8| = 1.6)
        #
        # The loss is MSE between predicted and actual fractions, so larger
        # stoichiometry errors produce proportionally larger gradients.
        #
        # Architecture: z → hidden → [max_elements fractions + count]
        # Output shape: [batch, max_elements + 1] where:
        #   - [:, :max_elements] = predicted fraction for each element slot
        #   - [:, -1] = predicted number of elements
        # =====================================================================
        self.max_elements = 12  # Maximum elements per formula
        self.fraction_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.max_elements + 1)  # +1 for element count
        )

        # =====================================================================
        # NUMERATOR/DENOMINATOR PREDICTION HEAD — REMOVED in V13.0
        # =====================================================================
        # V12.38-V12.41: numden_head predicted raw (num, den) in log1p space for
        # decoder conditioning. V13.0 removes this because fraction prediction is
        # now handled by the decoder's cross-entropy over semantic fraction tokens.
        # The numden signal is implicit in the fraction tokens themselves.
        # Old checkpoints with numden_head weights can be loaded with strict=False.
        # =====================================================================

        # =====================================================================
        # HIGH-PRESSURE PREDICTION HEAD (V12.19)
        # =====================================================================
        # Binary classifier: P(requires_high_pressure | z)
        # Predicts whether a superconductor requires high pressure to exhibit SC.
        # ~1% of SC are HP, so use pos_weight in BCEWithLogitsLoss during training.
        # Architecture: z → hidden → 1 (logit, no sigmoid — BCE handles it)
        # =====================================================================
        self.hp_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # =====================================================================
        # Tc BUCKET CLASSIFICATION HEAD (V12.28 — auxiliary signal)
        # =====================================================================
        # Classifies Tc into 5 buckets for coarse-grained signal:
        #   0=non-SC(Tc=0), 1=low(0-10K), 2=medium(10-50K),
        #   3=high(50-100K), 4=very-high(100K+)
        # This auxiliary loss helps the Tc head learn bucket boundaries.
        # =====================================================================
        self.tc_class_head = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 5),  # 5 Tc buckets
        )

        # =====================================================================
        # SC/NON-SC CLASSIFICATION HEAD (V12.21/V12.28 — Cross-Head Consistency)
        # =====================================================================
        # Binary classifier: P(is_superconductor | z, head_predictions)
        # Unlike other heads that read only z, this head receives z CONCATENATED
        # with the predictions of ALL other heads. This creates a cross-head
        # consistency check: the SC classifier learns patterns like "high Tc +
        # certain Magpie features → likely SC" and "Tc ≈ 0 + non-SC Magpie → not SC".
        # Gradients flow through the concatenated predictions back into the other
        # heads, so the SC loss also improves the other heads' accuracy.
        #
        # V12.28: Added tc_class_logits(5) to input
        # Input: z(2048) + tc_pred(1) + magpie_pred(magpie_dim) + hp_pred(1)
        #        + fraction_pred(12) + element_count_pred(1) + competence(1)
        #        + tc_class_logits(5) = 2069 + magpie_dim
        # =====================================================================
        sc_input_dim = latent_dim + 1 + magpie_dim + 1 + self.max_elements + 1 + 1 + 5  # +5 for tc_class
        self.sc_head = nn.Sequential(
            nn.Linear(sc_input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),  # logits — apply sigmoid for probability
        )

        # =====================================================================
        # HIERARCHICAL FAMILY CLASSIFICATION HEAD (V12.33)
        # =====================================================================
        # Replaces flat 14-class family_head with 3-level hierarchical tree:
        #   sc_pred → coarse (7 SC families) → sub-family (cuprate: 6, iron: 2)
        # Conditioned on sc_pred.detach() so P(NOT_SC) = 1 - P(SC).
        # Loss: separate CE at each level on appropriate subsets.
        # =====================================================================
        self.hierarchical_family_head = HierarchicalFamilyHead(
            backbone_dim=prev_dim, dropout=dropout
        )

    def get_config(self) -> Dict:
        """Return the constructor parameters for manifest embedding."""
        return {
            'n_elements': self.n_elements,
            'latent_dim': self.latent_dim,
            'magpie_dim': self.magpie_dim,
            'fusion_dim': self.fusion_dim,
        }

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE, or passthrough in deterministic mode."""
        if logvar is None:
            return mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(
        self,
        element_indices: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        magpie_features: torch.Tensor,
        tc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all inputs to latent space.

        Args:
            element_indices: [batch, max_elements] atomic numbers
            element_fractions: [batch, max_elements] molar fractions
            element_mask: [batch, max_elements] valid element mask
            magpie_features: [batch, 145] Magpie-derived features
            tc: [batch, 1] or [batch] critical temperature (normalized)

        Returns:
            Dict with z, z_mean, z_logvar, attention_weights, fused_repr
        """
        # Ensure tc is [batch, 1]
        if tc.dim() == 1:
            tc = tc.unsqueeze(-1)

        # Branch 1: Element composition
        element_repr, attention_weights, element_embeds = self.element_encoder(
            element_indices, element_fractions, element_mask
        )

        # Branch 2: Magpie features
        magpie_repr = self.magpie_encoder(magpie_features)

        # Branch 3: Tc embedding
        tc_repr = self.tc_encoder(tc)

        # Fuse all branches
        fused = torch.cat([element_repr, magpie_repr, tc_repr], dim=-1)
        fused = self.fusion(fused)

        # VAE encoding
        z_mean, z_logvar = self.vae_encoder(fused)
        z = self.reparameterize(z_mean, z_logvar)

        return {
            'z': z,
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'attention_weights': attention_weights,
            'element_embeddings': element_embeds,
            'fused_repr': fused,
        }

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent to all outputs.

        Args:
            z: [batch, latent_dim] latent representation

        Returns:
            Dict with tc_pred, magpie_pred, attended_input
        """
        # Shared backbone
        h = self.decoder_backbone(z)

        # Multi-head outputs
        # V12.28: Deeper residual Tc head
        tc_h = self.tc_proj(h)
        tc_h = tc_h + self.tc_res_block(tc_h)  # Residual connection
        tc_pred = self.tc_out(tc_h).squeeze(-1)

        magpie_pred = self.magpie_head(h)
        attended_input = self.attended_head(h)

        # V12.28: Tc bucket classification
        tc_class_logits = self.tc_class_head(h)

        return {
            'tc_pred': tc_pred,
            'magpie_pred': magpie_pred,
            'attended_input': attended_input,
            'tc_class_logits': tc_class_logits,
            'backbone_h': h,  # V12.33: Exposed for hierarchical family head
        }

    def forward(
        self,
        element_indices: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        magpie_features: torch.Tensor,
        tc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode inputs, decode to outputs.

        Returns dict with all latent and output tensors for loss computation.
        """
        # Encode
        enc_out = self.encode(
            element_indices, element_fractions, element_mask,
            magpie_features, tc
        )

        # Decode
        dec_out = self.decode(enc_out['z'])

        # Competence
        competence = self.competence_head(enc_out['z']).squeeze(-1)

        # Fraction value prediction (V12.4 stoichiometry-aware)
        # Predicts element fractions directly from z for stoichiometry loss
        fraction_output = self.fraction_head(enc_out['z'])
        fraction_pred = fraction_output[:, :self.max_elements]  # [batch, max_elements]
        element_count_pred = fraction_output[:, -1]  # [batch]

        # V13.0: numden_head removed — fraction info now in semantic fraction tokens
        # numden_pred kept as None for backward compatibility with loss function signatures
        numden_pred = None

        # High-pressure prediction (V12.19)
        hp_pred = self.hp_head(enc_out['z']).squeeze(-1)  # [batch] logits

        # V12.28: Tc bucket classification logits
        tc_class_logits = dec_out['tc_class_logits']

        # SC/non-SC classification with cross-head consistency (V12.21/V12.28)
        # Feed z + all other head predictions so the classifier learns
        # cross-head consistency patterns (e.g., high Tc + SC Magpie → SC)
        # V12.28: Added tc_class_logits(5) to input
        sc_input = torch.cat([
            enc_out['z'],                          # 2048
            dec_out['tc_pred'].unsqueeze(-1),       # 1
            dec_out['magpie_pred'],                 # magpie_dim
            hp_pred.unsqueeze(-1),                  # 1
            fraction_pred,                          # 12
            element_count_pred.unsqueeze(-1),       # 1
            competence.unsqueeze(-1),               # 1
            tc_class_logits,                        # 5 (V12.28)
        ], dim=-1)
        sc_pred = self.sc_head(sc_input).squeeze(-1)  # [batch] logits

        # V12.33: Hierarchical family classification (conditioned on sc_pred)
        # sc_pred.detach() prevents family gradients from flowing back through sc_head
        family_out = self.hierarchical_family_head(dec_out['backbone_h'], sc_pred.detach())

        # Latent regularization: L2 on z (deterministic mode) or KL (VAE mode)
        # IMPORTANT (2026-02-02): 'kl_loss' key is INTENTIONALLY reused for L2 reg.
        # The entire downstream pipeline (CombinedLossWithREINFORCE, train_v12_clean.py,
        # loss logging, checkpoint saving) reads loss via the 'kl_loss' key and multiplies
        # by config['kl_weight']. Renaming would require touching 10+ callsites for no
        # functional benefit. The value is L2 reg (mean(z²)) NOT KL divergence when
        # deterministic=True. Config comment in train_v12_clean.py documents this too.
        if enc_out['z_logvar'] is None:
            # Deterministic mode: light L2 regularization keeps z bounded
            z_reg = torch.mean(enc_out['z'].pow(2))
        else:
            # VAE mode: standard KL divergence
            z_reg = -0.5 * torch.mean(
                1 + enc_out['z_logvar'] - enc_out['z_mean'].pow(2) - enc_out['z_logvar'].exp()
            )

        return {
            # Latent
            'z': enc_out['z'],
            'z_mean': enc_out['z_mean'],
            'z_logvar': enc_out['z_logvar'],  # None in deterministic mode
            # NOTE: 'kl_loss' key contains L2 reg (mean(z²)) in deterministic mode,
            # NOT KL divergence. Key name kept for downstream compatibility.
            'kl_loss': z_reg,
            # Encoder outputs
            'attention_weights': enc_out['attention_weights'],
            'element_embeddings': enc_out['element_embeddings'],
            # Decoder outputs
            'tc_pred': dec_out['tc_pred'],
            'magpie_pred': dec_out['magpie_pred'],
            'attended_input': dec_out['attended_input'],
            # Competence
            'competence': competence,
            # Stoichiometry prediction (V12.4)
            'fraction_pred': fraction_pred,
            'element_count_pred': element_count_pred,
            # Numerator/denominator prediction — V13.0: always None (numden_head removed)
            'numden_pred': numden_pred,
            # High-pressure prediction (V12.19) — logits, apply sigmoid for probability
            'hp_pred': hp_pred,
            # SC/non-SC classification (V12.21) — logits, apply sigmoid for probability
            # Cross-head consistency: uses z + all other head predictions as input
            'sc_pred': sc_pred,
            # Tc bucket classification (V12.28) — 5 classes, cross-entropy loss
            'tc_class_logits': tc_class_logits,
            # V12.33: Hierarchical family classification (replaces flat family_logits)
            'family_coarse_logits': family_out['coarse_logits'],        # [batch, 7]
            'family_cuprate_sub_logits': family_out['cuprate_sub_logits'],  # [batch, 6]
            'family_iron_sub_logits': family_out['iron_sub_logits'],    # [batch, 2]
            'family_composed_14': family_out['composed_14'],            # [batch, 14] probs
        }

    def predict_tc_mc(self, z: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout: run decode N times with dropout active, average Tc predictions.

        This gives both a refined prediction (mean) and an uncertainty estimate (std).
        Higher std indicates the model is less confident about this sample.

        Args:
            z: [batch, latent_dim] latent representation
            n_samples: Number of forward passes (default 10)

        Returns:
            Tuple of (tc_mean, tc_std) each [batch]
        """
        was_training = self.training
        self.decoder_backbone.train()  # Enable dropout
        self.tc_res_block.train()      # Enable dropout in residual block

        tc_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                h = self.decoder_backbone(z)
                tc_h = self.tc_proj(h)
                tc_h = tc_h + self.tc_res_block(tc_h)
                tc_preds.append(self.tc_out(tc_h).squeeze(-1))

        # Restore original mode
        if not was_training:
            self.decoder_backbone.eval()
            self.tc_res_block.eval()

        tc_stack = torch.stack(tc_preds)
        return tc_stack.mean(dim=0), tc_stack.std(dim=0)

    def upgrade_tc_head_from_checkpoint(self, old_state_dict: dict):
        """
        Net2Net weight transfer from old tc_head to new tc_proj/tc_res_block/tc_out.

        Old architecture: tc_head = Sequential(Linear(512,256), GELU, Linear(256,1))
        New architecture: tc_proj(512→256) + tc_res_block(256→256) + tc_out(LN,GELU,256→128,GELU,128→1)

        Transfer strategy:
          - tc_head.0.weight/bias → tc_proj.weight/bias (direct copy, same shape)
          - tc_head.2.weight/bias → used to initialize tc_out final layer via Net2Net wider
          - tc_res_block stays identity-initialized (from __init__)
        """
        from .net2net_expansion import expand_linear_wider

        with torch.no_grad():
            # Transfer tc_head.0 → tc_proj (Linear 512→256, exact match)
            if 'tc_head.0.weight' in old_state_dict:
                self.tc_proj.weight.copy_(old_state_dict['tc_head.0.weight'])
                self.tc_proj.bias.copy_(old_state_dict['tc_head.0.bias'])
                print("  [Net2Net] tc_head.0 → tc_proj: direct weight transfer")

            # Transfer tc_head.2 → tc_out final layer via Net2Net wider expansion
            # Old: Linear(256→1), New: Linear(128→1)
            # The old 256→1 layer's knowledge gets compressed into the new 128→1 layer.
            # We use the old weights for the first 128 input connections and zero the rest,
            # letting the new intermediate 256→128 layer learn the projection.
            if 'tc_head.2.weight' in old_state_dict:
                old_w = old_state_dict['tc_head.2.weight']  # [1, 256]
                old_b = old_state_dict['tc_head.2.bias']    # [1]
                # tc_out[4] is Linear(128→1)
                # Copy first 128 columns of old [1,256] weight
                self.tc_out[4].weight[:, :min(128, old_w.shape[1])].copy_(
                    old_w[:, :min(128, old_w.shape[1])]
                )
                self.tc_out[4].bias.copy_(old_b)
                # tc_out[2] is Linear(256→128) — initialize to pass through first 128 dims
                nn.init.zeros_(self.tc_out[2].weight)
                nn.init.zeros_(self.tc_out[2].bias)
                # Identity-like: first 128 outputs = first 128 inputs
                for i in range(128):
                    self.tc_out[2].weight[i, i] = 1.0
                print("  [Net2Net] tc_head.2 → tc_out: wider expansion with identity projection")


class FullMaterialsLoss(nn.Module):
    """
    Loss function for FullMaterialsVAE.

    Components:
        1. Tc reconstruction (MSE)
        2. Magpie reconstruction (MSE)
        3. Formula reconstruction (CE from external decoder - passed in)
        4. KL divergence (VAE regularization)
    """

    def __init__(
        self,
        tc_weight: float = 1.0,
        magpie_weight: float = 1.0,
        formula_weight: float = 1.0,
        kl_weight: float = 0.0,  # Keep 0 for jagged latent
        tc_huber_delta: float = 0.0,  # V12.20: 0 = MSE, >0 = Huber
    ):
        super().__init__()
        self.tc_weight = tc_weight
        self.magpie_weight = magpie_weight
        self.formula_weight = formula_weight
        self.kl_weight = kl_weight
        self.tc_huber_delta = tc_huber_delta

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        tc_true: torch.Tensor,
        magpie_true: torch.Tensor,
        formula_loss: torch.Tensor,  # From external formula decoder
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full materials reconstruction loss.
        """
        # Tc reconstruction (V12.20: Huber loss for outlier robustness)
        if self.tc_huber_delta > 0:
            tc_loss = F.huber_loss(outputs['tc_pred'], tc_true, delta=self.tc_huber_delta)
        else:
            tc_loss = F.mse_loss(outputs['tc_pred'], tc_true)

        # Magpie reconstruction
        magpie_loss = F.mse_loss(outputs['magpie_pred'], magpie_true)

        # KL divergence
        kl_loss = outputs['kl_loss']

        # Total loss
        total = (
            self.tc_weight * tc_loss +
            self.magpie_weight * magpie_loss +
            self.formula_weight * formula_loss +
            self.kl_weight * kl_loss
        )

        return {
            'total': total,
            'tc_loss': tc_loss,
            'magpie_loss': magpie_loss,
            'formula_loss': formula_loss,
            'kl_loss': kl_loss,
        }


# =============================================================================
# V13: STRUCTURED LATENT SPACE ENCODER
# =============================================================================

class StructuredLatentEncoder(nn.Module):
    """
    V13 Structured Latent Space Encoder.

    Replaces the VAE encoder section with semantically partitioned latent space:
    - z_class (64): Superconductor family classification
    - z_composition (128): Stoichiometry/element ratios
    - z_electronic (128): Electronic structure features
    - z_Tc (64): Temperature-dependent behavior

    Total z_dim = 384 (down from 2048)

    Key feature: z_Tc partition has an INTERMEDIATE Tc prediction head that
    creates a bottleneck constraint - z_Tc must encode Tc-predictive information.
    """

    def __init__(
        self,
        input_dim: int = 768,  # 3 * fusion_dim from encoder branches
        z_class_dim: int = 64,
        z_composition_dim: int = 128,
        z_electronic_dim: int = 128,
        z_Tc_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.z_dims = {
            'class': z_class_dim,
            'composition': z_composition_dim,
            'electronic': z_electronic_dim,
            'Tc': z_Tc_dim
        }
        self.total_z_dim = sum(self.z_dims.values())  # 384

        # Shared encoder backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Partition heads - each produces a semantically meaningful subspace
        self.z_class_head = nn.Linear(256, z_class_dim)
        self.z_composition_head = nn.Linear(256, z_composition_dim)
        self.z_electronic_head = nn.Linear(256, z_electronic_dim)
        self.z_Tc_head = nn.Linear(256, z_Tc_dim)

        # CRITICAL: Intermediate Tc predictor from z_Tc partition
        # This creates a bottleneck constraint - z_Tc MUST encode Tc-predictive features
        self.Tc_intermediate_predictor = nn.Sequential(
            nn.Linear(z_Tc_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fused_repr: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode fused representation to structured latent space.

        Args:
            fused_repr: [batch, input_dim] concatenated encoder branch outputs

        Returns:
            z: [batch, total_z_dim] full latent vector
            partitions: Dict with individual z partitions and intermediate predictions
        """
        h = self.backbone(fused_repr)

        # Compute each partition
        z_class = self.z_class_head(h)
        z_composition = self.z_composition_head(h)
        z_electronic = self.z_electronic_head(h)
        z_Tc = self.z_Tc_head(h)

        # Concatenate for full z (order matters for decoder)
        z = torch.cat([z_class, z_composition, z_electronic, z_Tc], dim=-1)

        # Intermediate Tc prediction from z_Tc partition (bottleneck constraint)
        Tc_pred_intermediate = self.Tc_intermediate_predictor(z_Tc)

        partitions = {
            'z_class': z_class,
            'z_composition': z_composition,
            'z_electronic': z_electronic,
            'z_Tc': z_Tc,
            'Tc_pred_intermediate': Tc_pred_intermediate
        }

        return z, partitions


class FullMaterialsV13(nn.Module):
    """
    V13 Full Materials Encoder with Structured Latent Space.

    This is V12 with the VAE encoder replaced by StructuredLatentEncoder.
    The encoder branches (element, Magpie, Tc) are preserved from V12.

    KEY CHANGES from V12:
    1. StructuredLatentEncoder replaces AttentionVAEEncoder (z = 384 vs 2048)
    2. No reparameterization trick (deterministic latent space)
    3. Intermediate Tc prediction from z_Tc partition (bottleneck)
    4. Skip connections DISABLED by default

    Weight preservation from V12:
    - element_encoder: FULLY PRESERVED (copy weights)
    - magpie_encoder: FULLY PRESERVED (copy weights)
    - tc_encoder: FULLY PRESERVED (copy weights)
    - fusion: FULLY PRESERVED (copy weights)
    - structured_encoder: NEW (random init)
    - decoder_backbone: ADAPTED (smaller input dim)
    - tc_head, magpie_head, attended_head: ADAPTED
    """

    def __init__(
        self,
        n_elements: int = 118,
        element_embed_dim: int = 128,
        n_attention_heads: int = 8,
        magpie_dim: int = 145,
        fusion_dim: int = 256,
        z_class_dim: int = 64,
        z_composition_dim: int = 128,
        z_electronic_dim: int = 128,
        z_Tc_dim: int = 64,
        decoder_hidden: List[int] = [256, 512],
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_elements = n_elements
        self.magpie_dim = magpie_dim
        self.fusion_dim = fusion_dim

        # Compute latent dim from partitions
        self.z_dims = {
            'class': z_class_dim,
            'composition': z_composition_dim,
            'electronic': z_electronic_dim,
            'Tc': z_Tc_dim
        }
        self.latent_dim = sum(self.z_dims.values())  # 384

        # =====================================================================
        # ENCODER BRANCHES (SAME AS V12 - weights will be transferred)
        # =====================================================================

        # Branch 1: Element composition with attention
        self.element_encoder = ElementEncoder(
            n_elements=n_elements,
            element_embed_dim=element_embed_dim,
            n_attention_heads=n_attention_heads,
            output_dim=fusion_dim,
            dropout=dropout,
            use_isotope_features=False
        )

        # Branch 2: Magpie features MLP
        self.magpie_encoder = nn.Sequential(
            nn.Linear(magpie_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Branch 3: Tc embedding
        self.tc_encoder = nn.Sequential(
            nn.Linear(1, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Fusion layer (SAME AS V12)
        total_fusion_dim = fusion_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(total_fusion_dim, total_fusion_dim),
            nn.LayerNorm(total_fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # =====================================================================
        # V13: STRUCTURED LATENT ENCODER (replaces VAE encoder)
        # =====================================================================

        self.structured_encoder = StructuredLatentEncoder(
            input_dim=total_fusion_dim,
            z_class_dim=z_class_dim,
            z_composition_dim=z_composition_dim,
            z_electronic_dim=z_electronic_dim,
            z_Tc_dim=z_Tc_dim,
            dropout=dropout
        )

        # =====================================================================
        # DECODER HEADS (adapted for smaller latent dim)
        # =====================================================================

        # Shared decoder backbone
        decoder_layers = []
        prev_dim = self.latent_dim  # 384 instead of 2048
        for hidden_dim in decoder_hidden:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.decoder_backbone = nn.Sequential(*decoder_layers)

        # Head 1: Tc prediction (final)
        self.tc_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.GELU(),
            nn.Linear(prev_dim // 2, 1)
        )

        # Head 2: Magpie feature reconstruction
        self.magpie_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.GELU(),
            nn.Linear(prev_dim, magpie_dim)
        )

        # Head 3: Attended representation for formula decoder
        self.attended_head = nn.Sequential(
            nn.Linear(prev_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Competence head
        self.competence_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 4),
            nn.GELU(),
            nn.Linear(self.latent_dim // 4, 1),
            nn.Sigmoid()
        )

    def encode(
        self,
        element_indices: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        magpie_features: torch.Tensor,
        tc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode all inputs to structured latent space."""
        # Ensure tc is [batch, 1]
        if tc.dim() == 1:
            tc = tc.unsqueeze(-1)

        # Branch 1: Element composition
        element_repr, attention_weights, element_embeds = self.element_encoder(
            element_indices, element_fractions, element_mask
        )

        # Branch 2: Magpie features
        magpie_repr = self.magpie_encoder(magpie_features)

        # Branch 3: Tc embedding
        tc_repr = self.tc_encoder(tc)

        # Fuse all branches
        fused = torch.cat([element_repr, magpie_repr, tc_repr], dim=-1)
        fused = self.fusion(fused)

        # V13: Structured encoding (no reparameterization)
        z, partitions = self.structured_encoder(fused)

        return {
            'z': z,
            'z_partitions': partitions,
            'attention_weights': attention_weights,
            'element_embeddings': element_embeds,
            'fused_repr': fused,
            'Tc_pred_intermediate': partitions['Tc_pred_intermediate'],
        }

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode latent to all outputs."""
        h = self.decoder_backbone(z)

        tc_pred = self.tc_head(h).squeeze(-1)
        magpie_pred = self.magpie_head(h)
        attended_input = self.attended_head(h)

        return {
            'tc_pred': tc_pred,
            'magpie_pred': magpie_pred,
            'attended_input': attended_input,
        }

    def forward(
        self,
        element_indices: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        magpie_features: torch.Tensor,
        tc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass."""
        # Encode
        enc_out = self.encode(
            element_indices, element_fractions, element_mask,
            magpie_features, tc
        )

        # Decode
        dec_out = self.decode(enc_out['z'])

        # Competence
        competence = self.competence_head(enc_out['z']).squeeze(-1)

        # V13: No KL loss (deterministic latent)
        kl_loss = torch.tensor(0.0, device=enc_out['z'].device)

        return {
            # Latent
            'z': enc_out['z'],
            'z_partitions': enc_out['z_partitions'],
            'kl_loss': kl_loss,  # Always 0 for V13
            # Encoder outputs
            'attention_weights': enc_out['attention_weights'],
            'element_embeddings': enc_out['element_embeddings'],
            # Intermediate prediction (for bottleneck loss)
            'Tc_pred_intermediate': enc_out['Tc_pred_intermediate'],
            # Decoder outputs
            'tc_pred': dec_out['tc_pred'],
            'magpie_pred': dec_out['magpie_pred'],
            'attended_input': dec_out['attended_input'],
            # Competence
            'competence': competence,
        }

    @classmethod
    def from_v12_checkpoint(
        cls,
        v12_checkpoint_path: str,
        device: torch.device = torch.device('cuda'),
        magpie_dim: int = 145,
        **kwargs
    ) -> 'FullMaterialsV13':
        """
        Create V13 model with weights transferred from V12 checkpoint.

        Transfers:
        - element_encoder: FULL (all weights)
        - magpie_encoder: FULL (all weights)
        - tc_encoder: FULL (all weights)
        - fusion: FULL (all weights)
        - structured_encoder: NEW (random init)
        - decoder_*: NEW (random init, different input dim)

        Args:
            v12_checkpoint_path: Path to V12 best_model.pt
            device: Target device
            magpie_dim: Number of Magpie features
            **kwargs: Override default V13 hyperparameters

        Returns:
            FullMaterialsV13 with transferred encoder branch weights
        """
        import torch

        # Load V12 checkpoint
        checkpoint = torch.load(v12_checkpoint_path, map_location=device)
        v12_encoder_state = checkpoint.get('encoder_state_dict', {})

        # Create V13 model with default or overridden params
        model = cls(magpie_dim=magpie_dim, **kwargs).to(device)

        # Transfer encoder branch weights
        transferred = 0
        skipped = 0

        # Weights to transfer (exactly matching layers)
        transfer_prefixes = [
            'element_encoder.',
            'magpie_encoder.',
            'tc_encoder.',
            'fusion.',
        ]

        for name, param in model.named_parameters():
            # Check if this parameter should be transferred
            should_transfer = any(name.startswith(prefix) for prefix in transfer_prefixes)

            if should_transfer and name in v12_encoder_state:
                v12_param = v12_encoder_state[name]
                if param.shape == v12_param.shape:
                    param.data.copy_(v12_param)
                    transferred += 1
                else:
                    print(f"  Shape mismatch for {name}: V12={v12_param.shape}, V13={param.shape}")
                    skipped += 1
            elif should_transfer:
                print(f"  Missing in V12: {name}")
                skipped += 1

        print(f"V12 → V13 weight transfer: {transferred} transferred, {skipped} skipped/new")

        return model
