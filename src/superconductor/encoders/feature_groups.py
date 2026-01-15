"""
Grouped feature representation for superconductor materials.

Organizes features into semantic groups for attention-based processing:
- Composition: stoichiometry and element fractions
- Element Statistics: weighted mean/std of atomic properties
- Structure: crystal structure parameters (if available)
- Electronic: band structure features (if available)
- Thermodynamic: formation energy, stability (if available)
- Experimental: synthesis conditions (if available)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class FeatureGroups:
    """
    Structured grouping of superconductor features.

    Each group can be None if data is not available.
    The grouped encoder will handle missing groups gracefully.
    """
    # Group 1: Composition (required)
    composition: torch.Tensor  # [batch, comp_dim] - element fractions

    # Group 2: Element Statistics (required)
    element_stats: torch.Tensor  # [batch, stat_dim] - mean/std of properties

    # Group 3: Structure (optional)
    structure: Optional[torch.Tensor] = None  # [batch, struct_dim] - lattice params

    # Group 4: Electronic Properties (optional)
    electronic: Optional[torch.Tensor] = None  # [batch, elec_dim] - band structure

    # Group 5: Thermodynamic Properties (optional)
    thermodynamic: Optional[torch.Tensor] = None  # [batch, thermo_dim] - formation energy

    # Group 6: Experimental Conditions (optional)
    experimental: Optional[torch.Tensor] = None  # [batch, exp_dim] - synthesis conditions

    # Metadata
    batch_size: int = 0
    device: str = 'cpu'

    def __post_init__(self):
        """Compute batch size and device from composition tensor."""
        if self.composition is not None:
            self.batch_size = self.composition.shape[0]
            self.device = str(self.composition.device)

    def to(self, device: str) -> 'FeatureGroups':
        """Move all tensors to specified device."""
        return FeatureGroups(
            composition=self.composition.to(device) if self.composition is not None else None,
            element_stats=self.element_stats.to(device) if self.element_stats is not None else None,
            structure=self.structure.to(device) if self.structure is not None else None,
            electronic=self.electronic.to(device) if self.electronic is not None else None,
            thermodynamic=self.thermodynamic.to(device) if self.thermodynamic is not None else None,
            experimental=self.experimental.to(device) if self.experimental is not None else None,
        )

    def get_available_groups(self) -> List[str]:
        """Return list of available (non-None) group names."""
        groups = []
        if self.composition is not None:
            groups.append('composition')
        if self.element_stats is not None:
            groups.append('element_stats')
        if self.structure is not None:
            groups.append('structure')
        if self.electronic is not None:
            groups.append('electronic')
        if self.thermodynamic is not None:
            groups.append('thermodynamic')
        if self.experimental is not None:
            groups.append('experimental')
        return groups

    def get_group(self, name: str) -> Optional[torch.Tensor]:
        """Get feature group by name."""
        return getattr(self, name, None)

    def to_flat_tensor(self) -> torch.Tensor:
        """Concatenate all available groups into single tensor."""
        tensors = []
        for name in ['composition', 'element_stats', 'structure',
                     'electronic', 'thermodynamic', 'experimental']:
            tensor = getattr(self, name)
            if tensor is not None:
                tensors.append(tensor)
        return torch.cat(tensors, dim=-1)


class GroupedFeatureEncoder(nn.Module):
    """
    Encodes grouped features using cross-group attention.

    Each feature group is first projected to a common hidden dimension,
    then cross-group attention is applied to learn group interactions.
    """

    def __init__(
        self,
        group_dims: Dict[str, int],
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize grouped feature encoder.

        Args:
            group_dims: Dictionary mapping group names to input dimensions
            hidden_dim: Hidden dimension for attention
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.group_dims = group_dims
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.group_names = list(group_dims.keys())

        # Per-group encoders (project each group to hidden_dim)
        self.group_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for name, dim in group_dims.items()
        })

        # Cross-group attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        # Layer norm for attention output
        self.attention_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # Final projection (flatten groups and project)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * len(group_dims), hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.output_dim = hidden_dim

    def forward(
        self,
        groups: FeatureGroups,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode grouped features.

        Args:
            groups: FeatureGroups object with feature tensors
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (encoded features [batch, hidden_dim], optional attention weights)
        """
        batch_size = groups.batch_size
        device = groups.device

        # Encode each available group
        encoded_groups = []
        for name in self.group_names:
            group_tensor = groups.get_group(name)
            if group_tensor is not None:
                encoded = self.group_encoders[name](group_tensor)
                encoded_groups.append(encoded)
            else:
                # Use zeros for missing groups
                zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
                encoded_groups.append(zeros)

        # Stack: [batch, n_groups, hidden_dim]
        group_embeddings = torch.stack(encoded_groups, dim=1)

        # Cross-group attention
        attended, attention_weights = self.cross_attention(
            group_embeddings, group_embeddings, group_embeddings
        )

        # Residual connection + norm
        attended = self.attention_norm(attended + group_embeddings)

        # Flatten and project
        flat = attended.reshape(batch_size, -1)
        output = self.output_proj(flat)

        if return_attention:
            return output, attention_weights
        return output, None


class ExpertAttentionHead(nn.Module):
    """
    Expert-specific attention over feature groups.

    Each expert learns which feature groups to attend to,
    enabling interpretable expert specialization.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_groups: int,
        temperature: float = 1.0
    ):
        """
        Initialize expert attention head.

        Args:
            hidden_dim: Hidden dimension of group embeddings
            n_groups: Number of feature groups
            temperature: Softmax temperature (lower = sharper attention)
        """
        super().__init__()

        # Learnable query for this expert
        self.query = nn.Parameter(torch.randn(hidden_dim))

        # Key projection
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.n_groups = n_groups

    def forward(self, group_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights over feature groups.

        Args:
            group_embeddings: [batch, n_groups, hidden_dim] or [n_groups, hidden_dim]

        Returns:
            attention_weights: [batch, n_groups] or [n_groups] soft weights over groups
        """
        # Handle both batched and unbatched input
        if group_embeddings.dim() == 2:
            # [n_groups, hidden_dim]
            keys = self.key_proj(group_embeddings)  # [n_groups, hidden_dim]
            scores = torch.matmul(keys, self.query) / self.temperature  # [n_groups]
            return F.softmax(scores, dim=0)
        else:
            # [batch, n_groups, hidden_dim]
            keys = self.key_proj(group_embeddings)  # [batch, n_groups, hidden_dim]
            scores = torch.matmul(keys, self.query) / self.temperature  # [batch, n_groups]
            return F.softmax(scores, dim=-1)


class AttentiveExpert(nn.Module):
    """
    Expert network with attention over feature groups.

    Each expert learns to attend to specific feature groups,
    enabling interpretable specialization.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_groups: int,
        output_dim: int = 1,
        temperature: float = 1.0
    ):
        """
        Initialize attentive expert.

        Args:
            hidden_dim: Hidden dimension of group embeddings
            n_groups: Number of feature groups
            output_dim: Output dimension (1 for Tc prediction)
            temperature: Attention temperature
        """
        super().__init__()

        self.attention = ExpertAttentionHead(hidden_dim, n_groups, temperature)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.hidden_dim = hidden_dim
        self.n_groups = n_groups
        self.output_dim = output_dim

    def forward(
        self,
        group_embeddings: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with attention-weighted group combination.

        Args:
            group_embeddings: [batch, n_groups, hidden_dim]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output [batch, output_dim], optional attention weights)
        """
        # Get attention weights for this expert
        attn_weights = self.attention(group_embeddings)  # [batch, n_groups]

        # Weighted combination of group embeddings
        # [batch, n_groups, 1] * [batch, n_groups, hidden_dim] -> sum -> [batch, hidden_dim]
        weighted = (group_embeddings * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Predict
        output = self.mlp(weighted)

        if return_attention:
            return output, attn_weights
        return output, None


class ContrastiveFeatureEncoder(nn.Module):
    """
    Feature encoder with contrastive learning support.

    Learns representations that:
    - Pull superconductors with similar Tc together
    - Push superconductors away from non-superconductors
    - Push superconductors away from magnetic materials

    This is critical for learning what makes materials superconduct.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        """
        Initialize contrastive encoder.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            temperature: InfoNCE temperature
            dropout: Dropout rate
        """
        super().__init__()

        self.temperature = temperature
        self.latent_dim = latent_dim

        # Build encoder
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features to latent space."""
        return self.encoder(x)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent to contrastive space."""
        return F.normalize(self.projection_head(z), dim=-1)

    def contrastive_loss(
        self,
        z_superconductor: torch.Tensor,
        z_negative: torch.Tensor,
        tc_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            z_superconductor: [batch_sc, latent_dim] superconductor embeddings
            z_negative: [batch_neg, latent_dim] negative sample embeddings
                       (non-superconductors, magnetic materials)
            tc_values: Optional [batch_sc] Tc values for similarity weighting

        Returns:
            Contrastive loss value
        """
        # Project to contrastive space
        proj_sc = self.project(z_superconductor)  # [batch_sc, latent_dim]
        proj_neg = self.project(z_negative)  # [batch_neg, latent_dim]

        # Compute similarities
        # Superconductor-to-superconductor (positives based on Tc similarity)
        sim_sc_sc = torch.matmul(proj_sc, proj_sc.T) / self.temperature  # [batch_sc, batch_sc]

        # Superconductor-to-negative (all negatives)
        sim_sc_neg = torch.matmul(proj_sc, proj_neg.T) / self.temperature  # [batch_sc, batch_neg]

        # If Tc values provided, use them to weight positive pairs
        if tc_values is not None:
            # Materials with similar Tc should be closer
            tc_diff = torch.abs(tc_values.unsqueeze(1) - tc_values.unsqueeze(0))
            tc_weights = torch.exp(-tc_diff / 50.0)  # Decay scale of 50K
            # Mask diagonal
            mask = ~torch.eye(len(tc_values), dtype=torch.bool, device=tc_values.device)
            tc_weights = tc_weights * mask.float()
        else:
            tc_weights = None

        # InfoNCE loss
        # For each superconductor, treat other superconductors as positives
        # and non-superconductors as negatives

        # Concatenate positive and negative similarities
        # [batch_sc, batch_sc + batch_neg]
        all_sim = torch.cat([sim_sc_sc, sim_sc_neg], dim=1)

        # Labels: positives are indices 0 to batch_sc-1 (excluding self)
        batch_sc = z_superconductor.shape[0]
        batch_neg = z_negative.shape[0]

        # Mask out self-similarity
        mask = torch.eye(batch_sc, dtype=torch.bool, device=z_superconductor.device)
        all_sim[:, :batch_sc].masked_fill_(mask, float('-inf'))

        # Compute loss
        # Each row should have high similarity with other superconductors
        # and low similarity with negatives

        # Use cross-entropy where all superconductors are "correct" classes
        # Simplified: maximize similarity with closest superconductor, minimize with negatives

        # Max similarity among positives (other superconductors)
        pos_sim = sim_sc_sc.clone()
        pos_sim.masked_fill_(mask, float('-inf'))
        max_pos_sim = pos_sim.max(dim=1)[0]  # [batch_sc]

        # LogSumExp over negatives
        neg_logsumexp = torch.logsumexp(sim_sc_neg, dim=1)  # [batch_sc]

        # Contrastive loss: push positives above negatives
        loss = -max_pos_sim + neg_logsumexp

        return loss.mean()


# Default feature group dimensions for superconductors
DEFAULT_GROUP_DIMS = {
    'composition': 118,  # One-hot element fractions (up to Og)
    'element_stats': 22,  # 11 properties * 2 (mean/std)
}

# Extended group dimensions when additional data is available
EXTENDED_GROUP_DIMS = {
    'composition': 118,
    'element_stats': 22,
    'structure': 12,  # a, b, c, alpha, beta, gamma, volume, density, space_group_enc...
    'electronic': 8,  # band_gap, dos_at_fermi, electron_density...
    'thermodynamic': 4,  # formation_energy, stability, hull_distance, decomposition_energy
    'experimental': 6,  # synthesis_temp, pressure, atmosphere, time...
}
