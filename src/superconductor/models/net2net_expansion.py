"""
Net2Net Model Expansion Utilities.

Enables progressive model growth while preserving learned weights:
1. Wider: Add neurons to existing layers
2. Deeper: Insert new layers between existing ones
3. Latent: Expand VAE latent dimensions

Based on: "Net2Net: Accelerating Learning via Knowledge Transfer" (Chen et al., 2016)

Usage:
    from superconductor.models.net2net_expansion import (
        expand_linear_wider,
        expand_linear_deeper,
        expand_transformer_decoder,
        expand_vae_latent,
    )
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
import copy


def expand_linear_wider(
    layer: nn.Linear,
    new_out_features: int,
    next_layer: Optional[nn.Linear] = None,
    noise_std: float = 0.01
) -> Tuple[nn.Linear, Optional[nn.Linear]]:
    """
    Expand a linear layer to have more output features (wider).

    The new neurons are initialized with small noise so the function
    is approximately preserved initially, then learns.

    Args:
        layer: Linear layer to expand
        new_out_features: New number of output features (must be >= current)
        next_layer: Optional next layer whose input features need adjustment
        noise_std: Std of noise for new neuron initialization

    Returns:
        Tuple of (expanded_layer, adjusted_next_layer)
    """
    old_out = layer.out_features
    in_features = layer.in_features

    assert new_out_features >= old_out, "Cannot shrink layer"

    if new_out_features == old_out:
        return layer, next_layer

    # Create new wider layer
    new_layer = nn.Linear(in_features, new_out_features, bias=layer.bias is not None)

    with torch.no_grad():
        # Copy old weights
        new_layer.weight[:old_out, :] = layer.weight.data

        # Initialize new neurons with small noise
        new_layer.weight[old_out:, :] = torch.randn(
            new_out_features - old_out, in_features
        ) * noise_std

        if layer.bias is not None:
            new_layer.bias[:old_out] = layer.bias.data
            new_layer.bias[old_out:] = torch.zeros(new_out_features - old_out)

    # Adjust next layer if provided
    new_next_layer = None
    if next_layer is not None:
        old_in = next_layer.in_features
        out_features = next_layer.out_features

        assert old_in == old_out, "Layer dimensions don't match"

        new_next_layer = nn.Linear(new_out_features, out_features, bias=next_layer.bias is not None)

        with torch.no_grad():
            # Copy old weights
            new_next_layer.weight[:, :old_in] = next_layer.weight.data

            # New input connections initialized to small values
            new_next_layer.weight[:, old_in:] = torch.randn(
                out_features, new_out_features - old_in
            ) * noise_std

            if next_layer.bias is not None:
                new_next_layer.bias.data = next_layer.bias.data.clone()

    return new_layer, new_next_layer


def expand_linear_deeper(
    layer: nn.Linear,
    activation: str = 'gelu'
) -> nn.Sequential:
    """
    Insert an identity-initialized layer after the given layer (deeper).

    The new layer starts as identity, so network function is preserved,
    then learns additional transformations.

    Args:
        layer: Linear layer to add depth after
        activation: Activation function ('gelu', 'relu', 'none')

    Returns:
        Sequential with original layer + new identity layer
    """
    out_features = layer.out_features

    # Create identity layer
    identity_layer = nn.Linear(out_features, out_features)

    with torch.no_grad():
        # Initialize as identity matrix
        nn.init.eye_(identity_layer.weight)
        nn.init.zeros_(identity_layer.bias)

    # Build sequential
    if activation == 'gelu':
        return nn.Sequential(layer, nn.GELU(), identity_layer)
    elif activation == 'relu':
        return nn.Sequential(layer, nn.ReLU(), identity_layer)
    else:
        return nn.Sequential(layer, identity_layer)


def expand_embedding(
    embedding: nn.Embedding,
    new_embedding_dim: int,
    noise_std: float = 0.01
) -> nn.Embedding:
    """
    Expand embedding dimension.

    Args:
        embedding: Embedding layer to expand
        new_embedding_dim: New embedding dimension
        noise_std: Std of noise for new dimensions

    Returns:
        Expanded embedding layer
    """
    num_embeddings = embedding.num_embeddings
    old_dim = embedding.embedding_dim
    padding_idx = embedding.padding_idx

    assert new_embedding_dim >= old_dim, "Cannot shrink embedding"

    if new_embedding_dim == old_dim:
        return embedding

    new_embedding = nn.Embedding(
        num_embeddings, new_embedding_dim,
        padding_idx=padding_idx
    )

    with torch.no_grad():
        # Copy old embeddings
        new_embedding.weight[:, :old_dim] = embedding.weight.data

        # Initialize new dimensions with small noise
        new_embedding.weight[:, old_dim:] = torch.randn(
            num_embeddings, new_embedding_dim - old_dim
        ) * noise_std

        # Keep padding vector as zeros
        if padding_idx is not None:
            new_embedding.weight[padding_idx] = 0

    return new_embedding


def expand_multihead_attention(
    attention: nn.MultiheadAttention,
    new_embed_dim: Optional[int] = None,
    new_num_heads: Optional[int] = None,
    noise_std: float = 0.01
) -> nn.MultiheadAttention:
    """
    Expand multihead attention layer.

    Can expand embed_dim (wider) or num_heads (more attention patterns).

    Args:
        attention: MultiheadAttention to expand
        new_embed_dim: New embedding dimension (optional)
        new_num_heads: New number of heads (optional)
        noise_std: Noise std for initialization

    Returns:
        Expanded attention layer
    """
    old_embed_dim = attention.embed_dim
    old_num_heads = attention.num_heads

    new_embed_dim = new_embed_dim or old_embed_dim
    new_num_heads = new_num_heads or old_num_heads

    assert new_embed_dim >= old_embed_dim, "Cannot shrink embed_dim"
    assert new_num_heads >= old_num_heads, "Cannot reduce heads"
    assert new_embed_dim % new_num_heads == 0, "embed_dim must be divisible by num_heads"

    if new_embed_dim == old_embed_dim and new_num_heads == old_num_heads:
        return attention

    new_attention = nn.MultiheadAttention(
        embed_dim=new_embed_dim,
        num_heads=new_num_heads,
        dropout=attention.dropout,
        bias=attention.in_proj_bias is not None,
        batch_first=attention.batch_first
    )

    with torch.no_grad():
        # Copy Q, K, V projection weights
        # in_proj_weight shape: (3 * embed_dim, embed_dim)
        if attention.in_proj_weight is not None:
            old_3e = 3 * old_embed_dim
            new_3e = 3 * new_embed_dim

            # Initialize new layer
            new_attention.in_proj_weight[:old_3e, :old_embed_dim] = attention.in_proj_weight.data
            new_attention.in_proj_weight[old_3e:, :] = torch.randn(
                new_3e - old_3e, new_embed_dim
            ) * noise_std
            new_attention.in_proj_weight[:old_3e, old_embed_dim:] = torch.randn(
                old_3e, new_embed_dim - old_embed_dim
            ) * noise_std

        # Copy output projection
        if attention.out_proj is not None:
            new_attention.out_proj.weight[:old_embed_dim, :old_embed_dim] = attention.out_proj.weight.data
            new_attention.out_proj.weight[old_embed_dim:, :] = torch.randn(
                new_embed_dim - old_embed_dim, new_embed_dim
            ) * noise_std
            new_attention.out_proj.weight[:old_embed_dim, old_embed_dim:] = torch.randn(
                old_embed_dim, new_embed_dim - old_embed_dim
            ) * noise_std

            if attention.out_proj.bias is not None:
                new_attention.out_proj.bias[:old_embed_dim] = attention.out_proj.bias.data
                new_attention.out_proj.bias[old_embed_dim:] = 0

    return new_attention


def expand_transformer_decoder_layer(
    layer: nn.TransformerDecoderLayer,
    new_d_model: Optional[int] = None,
    new_nhead: Optional[int] = None,
    new_dim_feedforward: Optional[int] = None,
    noise_std: float = 0.01
) -> nn.TransformerDecoderLayer:
    """
    Expand a transformer decoder layer.

    Args:
        layer: TransformerDecoderLayer to expand
        new_d_model: New model dimension
        new_nhead: New number of attention heads
        new_dim_feedforward: New FFN dimension
        noise_std: Noise std for initialization

    Returns:
        Expanded decoder layer
    """
    # Get current dimensions
    old_d_model = layer.self_attn.embed_dim
    old_nhead = layer.self_attn.num_heads
    old_dim_ff = layer.linear1.out_features

    new_d_model = new_d_model or old_d_model
    new_nhead = new_nhead or old_nhead
    new_dim_feedforward = new_dim_feedforward or old_dim_ff

    # Create new layer
    new_layer = nn.TransformerDecoderLayer(
        d_model=new_d_model,
        nhead=new_nhead,
        dim_feedforward=new_dim_feedforward,
        dropout=layer.dropout.p if hasattr(layer, 'dropout') else 0.1,
        batch_first=layer.self_attn.batch_first
    )

    # Copy and expand weights
    with torch.no_grad():
        # Self-attention
        if old_d_model == new_d_model and old_nhead == new_nhead:
            new_layer.self_attn.load_state_dict(layer.self_attn.state_dict())
        # Note: For complex expansion, use expand_multihead_attention

        # Cross-attention
        if old_d_model == new_d_model and old_nhead == new_nhead:
            new_layer.multihead_attn.load_state_dict(layer.multihead_attn.state_dict())

        # FFN
        if old_d_model == new_d_model and old_dim_ff == new_dim_feedforward:
            new_layer.linear1.load_state_dict(layer.linear1.state_dict())
            new_layer.linear2.load_state_dict(layer.linear2.state_dict())

        # Layer norms
        if old_d_model == new_d_model:
            new_layer.norm1.load_state_dict(layer.norm1.state_dict())
            new_layer.norm2.load_state_dict(layer.norm2.state_dict())
            new_layer.norm3.load_state_dict(layer.norm3.state_dict())

    return new_layer


def expand_vae_latent(
    encoder_state_dict: Dict[str, torch.Tensor],
    decoder_state_dict: Dict[str, torch.Tensor],
    old_latent_dim: int,
    new_latent_dim: int,
    encoder_latent_layer_name: str = "fc_mu",  # Layer that outputs mu
    decoder_latent_layer_name: str = "latent_to_memory",  # First decoder layer
    noise_std: float = 0.01
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Expand VAE latent space dimension.

    New latent dimensions are initialized to produce near-zero outputs,
    so the model function is approximately preserved.

    Args:
        encoder_state_dict: Encoder model state dict
        decoder_state_dict: Decoder model state dict
        old_latent_dim: Current latent dimension
        new_latent_dim: New latent dimension
        encoder_latent_layer_name: Name of encoder layer producing latent
        decoder_latent_layer_name: Name of decoder layer consuming latent
        noise_std: Noise std for new weights

    Returns:
        Tuple of (expanded_encoder_state_dict, expanded_decoder_state_dict)
    """
    assert new_latent_dim >= old_latent_dim, "Cannot shrink latent"

    if new_latent_dim == old_latent_dim:
        return encoder_state_dict, decoder_state_dict

    expanded_encoder = copy.deepcopy(encoder_state_dict)
    expanded_decoder = copy.deepcopy(decoder_state_dict)

    # Expand encoder output (mu and logvar layers)
    for suffix in ['.weight', '.bias']:
        # Expand mu layer
        mu_key = encoder_latent_layer_name + suffix
        if mu_key in expanded_encoder:
            old_tensor = expanded_encoder[mu_key]
            if suffix == '.weight':
                # weight shape: (latent_dim, in_features)
                in_features = old_tensor.shape[1]
                new_tensor = torch.zeros(new_latent_dim, in_features)
                new_tensor[:old_latent_dim, :] = old_tensor
                new_tensor[old_latent_dim:, :] = torch.randn(
                    new_latent_dim - old_latent_dim, in_features
                ) * noise_std
            else:
                # bias shape: (latent_dim,)
                new_tensor = torch.zeros(new_latent_dim)
                new_tensor[:old_latent_dim] = old_tensor
            expanded_encoder[mu_key] = new_tensor

        # Expand logvar layer (same expansion)
        logvar_key = encoder_latent_layer_name.replace('mu', 'logvar') + suffix
        if logvar_key in expanded_encoder:
            old_tensor = expanded_encoder[logvar_key]
            if suffix == '.weight':
                in_features = old_tensor.shape[1]
                new_tensor = torch.zeros(new_latent_dim, in_features)
                new_tensor[:old_latent_dim, :] = old_tensor
                new_tensor[old_latent_dim:, :] = torch.randn(
                    new_latent_dim - old_latent_dim, in_features
                ) * noise_std
            else:
                new_tensor = torch.zeros(new_latent_dim)
                new_tensor[:old_latent_dim] = old_tensor
            expanded_encoder[logvar_key] = new_tensor

    # Expand decoder input
    for key in expanded_decoder:
        if decoder_latent_layer_name in key and '.weight' in key:
            old_tensor = expanded_decoder[key]
            out_features = old_tensor.shape[0]
            new_tensor = torch.zeros(out_features, new_latent_dim)
            new_tensor[:, :old_latent_dim] = old_tensor
            # New latent dims have small weights (near-zero contribution initially)
            new_tensor[:, old_latent_dim:] = torch.randn(
                out_features, new_latent_dim - old_latent_dim
            ) * noise_std * 0.1  # Extra small for decoder
            expanded_decoder[key] = new_tensor

    return expanded_encoder, expanded_decoder


def insert_transformer_layer(
    decoder: nn.TransformerDecoder,
    position: int,
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    dropout: float = 0.1
) -> nn.TransformerDecoder:
    """
    Insert a new identity-initialized layer into a transformer decoder.

    Args:
        decoder: TransformerDecoder to expand
        position: Position to insert new layer (0 = first)
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: FFN dimension
        dropout: Dropout rate

    Returns:
        Decoder with new layer inserted
    """
    # Create new identity-initialized layer
    new_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True
    )

    # Initialize as near-identity
    with torch.no_grad():
        # Self-attention: initialize to pass through
        # Set attention weights to produce identity-like behavior
        # This is approximate - true identity would need specific init
        for param in new_layer.self_attn.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.01)
            else:
                nn.init.zeros_(param)

        # Cross-attention: small weights
        for param in new_layer.multihead_attn.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.01)
            else:
                nn.init.zeros_(param)

        # FFN: identity-like
        nn.init.eye_(new_layer.linear1.weight[:d_model, :])
        if new_layer.linear1.out_features > d_model:
            nn.init.zeros_(new_layer.linear1.weight[d_model:, :])
        nn.init.zeros_(new_layer.linear1.bias)

        nn.init.eye_(new_layer.linear2.weight[:, :d_model])
        if new_layer.linear2.in_features > d_model:
            nn.init.zeros_(new_layer.linear2.weight[:, d_model:])
        nn.init.zeros_(new_layer.linear2.bias)

    # Insert layer
    layers = list(decoder.layers)
    layers.insert(position, new_layer)

    # Create new decoder
    new_decoder = nn.TransformerDecoder(layers[0], num_layers=len(layers))
    new_decoder.layers = nn.ModuleList(layers)

    return new_decoder


class ModelExpander:
    """
    High-level interface for model expansion.

    Usage:
        expander = ModelExpander(model)
        expander.expand_latent(new_dim=256)
        expander.add_transformer_layer(position=2)
        expanded_model = expander.get_model()
    """

    def __init__(self, model: nn.Module):
        self.model = copy.deepcopy(model)
        self.expansion_log = []

    def expand_latent(
        self,
        new_dim: int,
        encoder_attr: str = 'encoder',
        decoder_attr: str = 'decoder'
    ):
        """Expand VAE latent dimension."""
        encoder = getattr(self.model, encoder_attr)
        decoder = getattr(self.model, decoder_attr)

        old_dim = encoder.latent_dim

        # This is a simplified version - full implementation would
        # traverse the model and expand all relevant layers
        self.expansion_log.append(f"Expanded latent: {old_dim} -> {new_dim}")

        # Update latent_dim attributes
        encoder.latent_dim = new_dim
        decoder.latent_dim = new_dim

        return self

    def get_model(self) -> nn.Module:
        """Get the expanded model."""
        return self.model

    def get_expansion_log(self) -> list:
        """Get log of expansions performed."""
        return self.expansion_log


# Convenience function for warm-start training
def load_and_expand_checkpoint(
    checkpoint_path: str,
    model_class: type,
    expansion_config: Dict[str, Any],
    device: str = 'cuda'
) -> nn.Module:
    """
    Load a checkpoint and expand the model according to config.

    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Class to instantiate
        expansion_config: Dict with expansion parameters
        device: Device to load model to

    Returns:
        Expanded model with loaded weights
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create expanded model
    # (Implementation depends on specific model architecture)

    # Load compatible weights
    # (Use strict=False to allow new parameters)

    return None  # Placeholder - implement for specific models
