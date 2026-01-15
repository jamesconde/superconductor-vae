"""
Element-level attention for superconductor encoding.

Learns which elements in a chemical composition are most important
for superconductivity prediction. Provides interpretable attention
weights showing what the model focuses on.

Example:
    For YBa2Cu3O7 (YBCO):
    - Cu might get attention 0.45 (Cu-O planes are key)
    - O might get attention 0.35 (oxygen content matters)
    - Ba, Y get lower attention (structural spacers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class AttentionOutput:
    """Output from element attention module."""
    weighted_representation: torch.Tensor  # [batch, hidden_dim]
    attention_weights: torch.Tensor        # [batch, n_elements]
    element_embeddings: torch.Tensor       # [batch, n_elements, hidden_dim]


class ElementEmbedding(nn.Module):
    """
    Embed individual elements with their properties.

    Each element gets an embedding based on:
    - Learnable element-specific vector
    - Property-based features (electronegativity, radius, etc.)
    """

    def __init__(
        self,
        n_elements: int = 118,
        embedding_dim: int = 64,
        property_dim: int = 11,  # Number of element properties
        use_properties: bool = True
    ):
        """
        Args:
            n_elements: Maximum number of elements (default: full periodic table)
            embedding_dim: Dimension of element embeddings
            property_dim: Dimension of property features
            use_properties: Whether to combine with property features
        """
        super().__init__()

        self.n_elements = n_elements
        self.embedding_dim = embedding_dim
        self.use_properties = use_properties

        # Learnable element embeddings
        self.element_embed = nn.Embedding(n_elements + 1, embedding_dim, padding_idx=0)

        # Property encoder (if using properties)
        if use_properties:
            self.property_encoder = nn.Sequential(
                nn.Linear(property_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU()
            )
            # Combine learned + property embeddings
            self.combiner = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(
        self,
        element_indices: torch.Tensor,
        element_properties: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed elements.

        Args:
            element_indices: [batch, n_elements] atomic numbers (0 = padding)
            element_properties: [batch, n_elements, property_dim] optional properties

        Returns:
            [batch, n_elements, embedding_dim] element embeddings
        """
        # Learnable embeddings
        embeds = self.element_embed(element_indices)  # [batch, n_elem, embed_dim]

        if self.use_properties and element_properties is not None:
            # Encode properties
            prop_embeds = self.property_encoder(element_properties)
            # Combine
            combined = torch.cat([embeds, prop_embeds], dim=-1)
            embeds = self.combiner(combined)

        return embeds


class ElementAttention(nn.Module):
    """
    Attention mechanism over elements in a chemical composition.

    Learns which elements are most important for the prediction task.
    Uses a learnable query that represents "what matters for superconductivity"
    and attends over element embeddings.

    Example:
        attention = ElementAttention(hidden_dim=64)
        output = attention(element_embeddings, element_mask)
        # output.attention_weights shows importance of each element
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Args:
            hidden_dim: Dimension of element embeddings
            n_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Softmax temperature (lower = sharper attention)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.temperature = temperature

        # Learnable query: "what matters for superconductivity?"
        self.query = nn.Parameter(torch.randn(n_heads, self.head_dim))

        # Project elements to keys and values
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize
        nn.init.xavier_uniform_(self.query)

    def forward(
        self,
        element_embeddings: torch.Tensor,
        element_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> AttentionOutput:
        """
        Apply attention over elements.

        Args:
            element_embeddings: [batch, n_elements, hidden_dim]
            element_mask: [batch, n_elements] True for valid elements
            return_attention: Whether to return attention weights

        Returns:
            AttentionOutput with weighted representation and attention weights
        """
        batch_size, n_elements, _ = element_embeddings.shape

        # Project to keys and values
        keys = self.key_proj(element_embeddings)  # [batch, n_elem, hidden]
        values = self.value_proj(element_embeddings)  # [batch, n_elem, hidden]

        # Reshape for multi-head attention
        keys = keys.view(batch_size, n_elements, self.n_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)  # [batch, heads, n_elem, head_dim]

        values = values.view(batch_size, n_elements, self.n_heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)  # [batch, heads, n_elem, head_dim]

        # Query: [n_heads, head_dim] -> [1, n_heads, 1, head_dim]
        query = self.query.unsqueeze(0).unsqueeze(2)

        # Compute attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1))  # [batch, heads, 1, n_elem]
        scores = scores / (self.head_dim ** 0.5 * self.temperature)

        # Apply mask (set padding to -inf)
        if element_mask is not None:
            mask = element_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, n_elem]
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax to get attention weights
        attention = F.softmax(scores, dim=-1)  # [batch, heads, 1, n_elem]
        attention = self.dropout(attention)

        # Apply attention to values
        attended = torch.matmul(attention, values)  # [batch, heads, 1, head_dim]
        attended = attended.squeeze(2)  # [batch, heads, head_dim]
        attended = attended.reshape(batch_size, self.hidden_dim)  # [batch, hidden]

        # Output projection
        output = self.output_proj(attended)
        output = self.layer_norm(output)

        # Average attention across heads for interpretability
        avg_attention = attention.mean(dim=1).squeeze(1)  # [batch, n_elem]

        return AttentionOutput(
            weighted_representation=output,
            attention_weights=avg_attention,
            element_embeddings=element_embeddings
        )


class IsotopeAwareElementAttention(nn.Module):
    """
    Element attention that also considers isotope information.

    For each element, considers:
    - Base element properties
    - Isotope-specific mass and nuclear spin
    - Mass deviation from natural abundance average

    This allows the model to learn isotope effects on superconductivity.
    """

    def __init__(
        self,
        n_elements: int = 118,
        hidden_dim: int = 64,
        n_heads: int = 4,
        isotope_feature_dim: int = 4,  # mass, spin, abundance, mass_deviation
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_elements = n_elements
        self.hidden_dim = hidden_dim

        # Element embedding (base properties)
        self.element_embedding = ElementEmbedding(
            n_elements=n_elements,
            embedding_dim=hidden_dim,
            use_properties=True
        )

        # Isotope feature encoder
        self.isotope_encoder = nn.Sequential(
            nn.Linear(isotope_feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Combine element + isotope embeddings
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Main attention mechanism
        self.attention = ElementAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(
        self,
        element_indices: torch.Tensor,
        element_properties: torch.Tensor,
        isotope_features: torch.Tensor,
        element_mask: Optional[torch.Tensor] = None
    ) -> AttentionOutput:
        """
        Apply isotope-aware attention.

        Args:
            element_indices: [batch, n_elements] atomic numbers
            element_properties: [batch, n_elements, property_dim] element properties
            isotope_features: [batch, n_elements, isotope_dim] isotope features
                              (mass, spin, abundance, mass_deviation)
            element_mask: [batch, n_elements] True for present elements

        Returns:
            AttentionOutput with isotope-aware weighted representation
        """
        # Get base element embeddings
        elem_embeds = self.element_embedding(element_indices, element_properties)

        # Encode isotope features
        iso_embeds = self.isotope_encoder(isotope_features)

        # Combine
        combined = torch.cat([elem_embeds, iso_embeds], dim=-1)
        combined = self.combiner(combined)

        # Apply attention
        return self.attention(combined, element_mask)


class MultiHeadElementAttention(nn.Module):
    """
    Multiple attention heads that can learn different aspects of element importance.

    For example:
    - Head 1: Focus on transition metals
    - Head 2: Focus on oxygen/chalcogen content
    - Head 3: Focus on rare earth elements

    Each head has its own learnable query.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_queries: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of element embeddings
            n_queries: Number of independent queries (attention heads)
            dropout: Dropout rate
        """
        super().__init__()

        self.n_queries = n_queries
        self.hidden_dim = hidden_dim

        # Multiple learnable queries
        self.queries = nn.Parameter(torch.randn(n_queries, hidden_dim))

        # Shared key/value projections
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output: combine multiple heads
        self.output_proj = nn.Linear(hidden_dim * n_queries, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.queries)

    def forward(
        self,
        element_embeddings: torch.Tensor,
        element_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-query attention.

        Args:
            element_embeddings: [batch, n_elements, hidden_dim]
            element_mask: [batch, n_elements]

        Returns:
            output: [batch, hidden_dim] weighted representation
            attention: [batch, n_queries, n_elements] per-query attention
        """
        batch_size, n_elements, _ = element_embeddings.shape

        # Project
        keys = self.key_proj(element_embeddings)  # [batch, n_elem, hidden]
        values = self.value_proj(element_embeddings)

        # Compute attention for each query
        # queries: [n_queries, hidden] -> [1, n_queries, hidden, 1]
        queries = self.queries.unsqueeze(0).unsqueeze(-1)

        # keys: [batch, n_elem, hidden] -> [batch, 1, hidden, n_elem]
        keys_t = keys.unsqueeze(1).transpose(-2, -1)

        # Scores: [batch, n_queries, 1, n_elem]
        scores = torch.matmul(queries.transpose(-2, -1), keys_t)
        scores = scores.squeeze(-2) / (self.hidden_dim ** 0.5)  # [batch, n_queries, n_elem]

        # Mask
        if element_mask is not None:
            mask = element_mask.unsqueeze(1)  # [batch, 1, n_elem]
            scores = scores.masked_fill(~mask, float('-inf'))

        # Attention weights
        attention = F.softmax(scores, dim=-1)  # [batch, n_queries, n_elem]
        attention = self.dropout(attention)

        # Apply attention
        # attention: [batch, n_queries, n_elem]
        # values: [batch, n_elem, hidden]
        attended = torch.bmm(
            attention.view(batch_size * self.n_queries, 1, n_elements),
            values.unsqueeze(1).expand(-1, self.n_queries, -1, -1).reshape(
                batch_size * self.n_queries, n_elements, self.hidden_dim
            )
        )  # [batch * n_queries, 1, hidden]

        attended = attended.view(batch_size, self.n_queries * self.hidden_dim)

        # Output
        output = self.output_proj(attended)
        output = self.layer_norm(output)

        return output, attention


def interpret_attention_weights(
    attention_weights: torch.Tensor,
    element_symbols: List[str],
    top_k: int = 5
) -> List[Dict[str, float]]:
    """
    Convert attention weights to interpretable element importance.

    Args:
        attention_weights: [batch, n_elements] attention weights
        element_symbols: List of element symbols for each position
        top_k: Number of top elements to return

    Returns:
        List of dicts mapping element symbol to attention weight
    """
    batch_size = attention_weights.shape[0]
    results = []

    for i in range(batch_size):
        weights = attention_weights[i].cpu().numpy()

        # Get top-k elements
        top_indices = np.argsort(weights)[::-1][:top_k]

        element_importance = {}
        for idx in top_indices:
            if idx < len(element_symbols) and element_symbols[idx]:
                element_importance[element_symbols[idx]] = float(weights[idx])

        results.append(element_importance)

    return results
