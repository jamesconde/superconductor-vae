"""
V16.0: DETR-style Set Prediction Decoder for formula generation.

Predicts (element, fraction) pairs as an UNORDERED SET using fixed parallel slots
with Hungarian matching loss. Eliminates the sequential decomposition bottleneck
of the AR decoder — chemical formulas are sets, not sequences.

Architecture:
    z [B, 2048] → z_proj → memory [B, n_z_tokens, d_model]
    slot_queries [12, d_model] (learned) → self-attn + cross-attn layers
    → element_head [B, 12, 119] (118 elements + empty)
    → fraction_head [B, 12] (softplus, non-negative)
    → presence_head [B, 12] (occupied/empty logit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class SetDecoderLayer(nn.Module):
    """Pre-norm self-attention + cross-attention + FFN (DETR-style).

    Self-attention between slots lets them coordinate to avoid
    predicting duplicate elements. Cross-attention reads from z memory.
    """

    def __init__(self, d_model: int = 512, nhead: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()

        # Self-attention (NO causal mask — slots see each other)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (slots attend to z memory tokens)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, slots: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: [B, n_slots, d_model] — current slot representations
            memory: [B, n_z_tokens, d_model] — projected latent z tokens

        Returns:
            Updated slots [B, n_slots, d_model]
        """
        # Pre-norm self-attention
        x = self.norm1(slots)
        x, _ = self.self_attn(x, x, x)
        slots = slots + x

        # Pre-norm cross-attention
        x = self.norm2(slots)
        x, _ = self.cross_attn(x, memory, memory)
        slots = slots + x

        # Pre-norm FFN
        x = self.norm3(slots)
        x = self.ffn(x)
        slots = slots + x

        return slots


class SetFormulaDecoder(nn.Module):
    """DETR-style fixed-slot set prediction decoder.

    Predicts up to n_slots (element, fraction) pairs from latent z in a single
    parallel forward pass. During training, Hungarian matching assigns predicted
    slots to ground truth pairs optimally.

    Args:
        latent_dim: Encoder latent dimension (2048)
        d_model: Internal transformer dimension (512)
        nhead: Number of attention heads (8)
        num_layers: Number of SetDecoderLayers (3)
        dim_feedforward: FFN hidden dim (1024)
        n_slots: Max elements per formula (12)
        n_elements: Number of chemical elements (118)
        n_z_tokens: Number of memory tokens from z projection (4)
        dropout: Dropout rate (0.1)
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        n_slots: int = 12,
        n_elements: int = 118,
        n_z_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_slots = n_slots
        self.n_elements = n_elements
        self.n_z_tokens = n_z_tokens
        self.d_model = d_model

        # Project latent z into memory tokens for cross-attention
        self.z_proj = nn.Linear(latent_dim, n_z_tokens * d_model)

        # Learned slot queries (Xavier-initialized)
        _slot_init = torch.empty(1, n_slots, d_model)
        nn.init.xavier_uniform_(_slot_init)
        self.slot_queries = nn.Parameter(_slot_init.squeeze(0))

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            SetDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Element classification head: 119 classes (0=empty/no-object, 1-118=elements)
        self.element_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, n_elements + 1),  # +1 for no-object class 0
        )

        # Fraction regression head (non-negative via Softplus)
        self.fraction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.fraction_activation = nn.Softplus()

        # Presence head (binary: is this slot occupied?)
        self.presence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: z → set of (element, fraction) predictions.

        Args:
            z: [B, latent_dim] latent vector from encoder

        Returns:
            Dict with:
                element_logits: [B, n_slots, n_elements+1] — class 0 = empty
                fraction_pred: [B, n_slots] — predicted fractions (non-negative)
                presence_logits: [B, n_slots] — occupied/empty logits
        """
        B = z.shape[0]

        # Project z into memory tokens: [B, n_z_tokens, d_model]
        memory = self.z_proj(z).view(B, self.n_z_tokens, self.d_model)

        # Expand learned slot queries to batch: [B, n_slots, d_model]
        slots = self.slot_queries.unsqueeze(0).expand(B, -1, -1)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            slots = layer(slots, memory)

        # Prediction heads
        element_logits = self.element_head(slots)         # [B, n_slots, n_elements+1]
        fraction_pred = self.fraction_activation(
            self.fraction_head(slots).squeeze(-1)          # [B, n_slots]
        )
        presence_logits = self.presence_head(slots).squeeze(-1)  # [B, n_slots]

        return {
            'element_logits': element_logits,
            'fraction_pred': fraction_pred,
            'presence_logits': presence_logits,
        }

    @torch.no_grad()
    def generate(self, z: torch.Tensor, presence_threshold: float = 0.5
                 ) -> List[List[Tuple[int, float]]]:
        """
        Generate formula predictions as sets of (atomic_number, fraction) pairs.

        Args:
            z: [B, latent_dim] latent vectors
            presence_threshold: sigmoid threshold for slot occupancy

        Returns:
            List of B formulas, each a list of (atomic_number, fraction) tuples
        """
        out = self.forward(z)
        element_ids = out['element_logits'].argmax(dim=-1)  # [B, n_slots]
        fractions = out['fraction_pred']                     # [B, n_slots]
        presence = torch.sigmoid(out['presence_logits'])     # [B, n_slots]

        B = z.shape[0]
        results = []
        for b in range(B):
            formula = []
            for s in range(self.n_slots):
                elem = element_ids[b, s].item()
                if elem != 0 and presence[b, s].item() > presence_threshold:
                    formula.append((elem, fractions[b, s].item()))
            results.append(formula)
        return results
