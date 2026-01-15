"""
Pointer-Generator Transformer Decoder for Chemical Formulas.

This decoder can either:
1. GENERATE tokens from vocabulary (standard)
2. COPY tokens from the input formula (for exact reproduction)

The copy mechanism is especially important for preserving exact
stoichiometry like "0.85" instead of generating "0.8".

Based on: "Get To The Point: Summarization with Pointer-Generator Networks"
(See et al., 2017)

Architecture:
    - Standard Transformer decoder (self-attention + cross-attention to z)
    - Additional cross-attention to INPUT tokens (not just latent z)
    - Copy gate that decides generate vs copy at each step
    - Final distribution blends vocab and copy distributions

Usage:
    decoder = PointerGeneratorDecoder(latent_dim=128, d_model=256, ...)
    loss = decoder.compute_loss(z, input_tokens, target_tokens)
    generated = decoder.generate(z, input_tokens)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from .autoregressive_decoder import (
    VOCAB_SIZE, PAD_IDX, START_IDX, END_IDX,
    TOKEN_TO_IDX, IDX_TO_TOKEN,
    indices_to_formula,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CopyAttention(nn.Module):
    """
    Attention mechanism for copying from source sequence.

    Computes attention over source tokens to determine which tokens
    to copy at each decoding step.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,      # (batch, tgt_len, d_model)
        key: torch.Tensor,        # (batch, src_len, d_model)
        value: torch.Tensor,      # (batch, src_len, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (batch, src_len)
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention over source for copying.

        Returns:
            output: Attended values (batch, tgt_len, d_model)
            attn_weights: Attention weights (batch, tgt_len, src_len) if return_attn
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.size(1)

        # Project
        q = self.q_proj(query).view(batch_size, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, heads, tgt, src)

        # Mask padding
        if key_padding_mask is not None:
            # Expand mask: (batch, src_len) -> (batch, 1, 1, src_len)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        output = torch.matmul(attn_weights, v)  # (batch, heads, tgt, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        output = self.out_proj(output)

        # Average attention across heads for copy distribution
        if return_attn:
            attn_for_copy = attn_weights.mean(dim=1)  # (batch, tgt_len, src_len)
            return output, attn_for_copy
        return output, None


class PointerGeneratorDecoder(nn.Module):
    """
    Transformer decoder with pointer-generator mechanism.

    Can either generate from vocabulary or copy from input sequence.
    This is crucial for preserving exact stoichiometry values.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 50
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.vocab_size = VOCAB_SIZE

        # Token embeddings (shared for input and output)
        self.token_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=d_model,
            padding_idx=PAD_IDX
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Project latent to memory for cross-attention
        self.latent_to_memory = nn.Sequential(
            nn.Linear(latent_dim, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 8),
        )
        self.n_memory_tokens = 8

        # Transformer decoder (attends to latent memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Copy attention (attends to input tokens)
        self.copy_attention = CopyAttention(d_model, nhead, dropout)

        # Copy gate: decides generate vs copy
        # Input: decoder state + copy context
        self.copy_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # Vocabulary projection
        self.vocab_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, VOCAB_SIZE)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _create_memory(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        memory = self.latent_to_memory(z)
        memory = memory.view(batch_size, self.n_memory_tokens, self.d_model)
        return memory

    def forward(
        self,
        z: torch.Tensor,
        input_tokens: torch.Tensor,   # Source formula tokens
        target_tokens: torch.Tensor,  # Target formula tokens (for teacher forcing)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with pointer-generator.

        Args:
            z: Latent vectors (batch, latent_dim)
            input_tokens: Source tokens for copying (batch, src_len)
            target_tokens: Target tokens for teacher forcing (batch, tgt_len)

        Returns:
            vocab_logits: Vocabulary distribution (batch, tgt_len-1, vocab_size)
            copy_attn: Copy attention weights (batch, tgt_len-1, src_len)
            copy_gate: Copy probability (batch, tgt_len-1, 1)
        """
        batch_size = z.size(0)
        device = z.device

        # Create memory from latent
        latent_memory = self._create_memory(z)

        # Embed source tokens for copy attention
        src_embedded = self.token_embedding(input_tokens)
        src_padding_mask = (input_tokens == PAD_IDX)

        # Embed target tokens (all but last)
        tgt_tokens = target_tokens[:, :-1]
        tgt_embedded = self.token_embedding(tgt_tokens)
        tgt_embedded = self.pos_encoding(tgt_embedded)

        # Causal mask for self-attention
        tgt_len = tgt_tokens.size(1)
        causal_mask = self._generate_causal_mask(tgt_len, device)
        tgt_padding_mask = (tgt_tokens == PAD_IDX)

        # Transformer decoder (attends to latent memory)
        decoder_output = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=latent_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # Copy attention (attends to source tokens)
        copy_context, copy_attn = self.copy_attention(
            query=decoder_output,
            key=src_embedded,
            value=src_embedded,
            key_padding_mask=src_padding_mask,
            return_attn=True
        )

        # Copy gate
        gate_input = torch.cat([decoder_output, copy_context], dim=-1)
        copy_prob = self.copy_gate(gate_input)  # (batch, tgt_len, 1)

        # Vocabulary distribution
        vocab_logits = self.vocab_proj(decoder_output)  # (batch, tgt_len, vocab_size)

        return vocab_logits, copy_attn, copy_prob

    def compute_loss(
        self,
        z: torch.Tensor,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pointer-generator loss.

        The loss encourages the model to either:
        - Generate the correct token from vocabulary, OR
        - Copy the correct token from source
        """
        vocab_logits, copy_attn, copy_prob = self.forward(z, input_tokens, target_tokens)

        # Target tokens (shifted)
        target = target_tokens[:, 1:vocab_logits.size(1)+1]
        batch_size, tgt_len = target.shape

        # Vocabulary probability
        vocab_prob = F.softmax(vocab_logits, dim=-1)  # (batch, tgt_len, vocab_size)

        # Build copy distribution over vocabulary
        # copy_attn: (batch, tgt_len, src_len)
        # input_tokens: (batch, src_len)
        # We need to scatter copy_attn into vocab-sized tensor

        copy_dist = torch.zeros_like(vocab_prob)  # (batch, tgt_len, vocab_size)

        # Expand input_tokens for scatter
        src_len = input_tokens.size(1)
        input_expanded = input_tokens.unsqueeze(1).expand(-1, tgt_len, -1)  # (batch, tgt_len, src_len)

        # Scatter copy attention into vocabulary positions
        copy_dist.scatter_add_(2, input_expanded, copy_attn)

        # Blend distributions
        # p_final = (1 - p_copy) * p_vocab + p_copy * p_copy_dist
        final_prob = (1 - copy_prob) * vocab_prob + copy_prob * copy_dist

        # Compute loss (negative log likelihood)
        # Gather probability of target tokens
        target_flat = target.reshape(-1)
        final_prob_flat = final_prob.reshape(-1, self.vocab_size)

        # Get probability of correct token
        target_prob = final_prob_flat.gather(1, target_flat.unsqueeze(1)).squeeze(1)

        # Mask padding
        mask = (target_flat != PAD_IDX).float()

        # Negative log likelihood with small epsilon for stability
        nll = -torch.log(target_prob + 1e-10)
        loss = (nll * mask).sum() / (mask.sum() + 1e-10)

        # Compute accuracy
        predictions = final_prob.argmax(dim=-1)
        mask_2d = target != PAD_IDX
        correct = ((predictions == target) & mask_2d).sum()
        total = mask_2d.sum()
        accuracy = correct.float() / (total.float() + 1e-10)

        # Copy usage statistics
        copy_usage = (copy_prob.squeeze(-1) * mask.view(batch_size, tgt_len)).sum() / (mask.sum() + 1e-10)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'perplexity': torch.exp(loss),
            'copy_usage': copy_usage,  # How often copy is used
        }

    def generate(
        self,
        z: torch.Tensor,
        input_tokens: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        max_len: Optional[int] = None
    ) -> List[str]:
        """
        Generate formulas with pointer-generator.

        Args:
            z: Latent vectors (batch, latent_dim)
            input_tokens: Source tokens for copying (batch, src_len)
            temperature: Sampling temperature
            top_k: Top-k sampling
            max_len: Maximum length

        Returns:
            List of generated formula strings
        """
        self.eval()
        batch_size = z.size(0)
        device = z.device
        max_len = max_len or self.max_len

        with torch.no_grad():
            # Prepare memory and source embeddings
            latent_memory = self._create_memory(z)
            src_embedded = self.token_embedding(input_tokens)
            src_padding_mask = (input_tokens == PAD_IDX)

            # Start with START token
            generated = torch.full(
                (batch_size, 1), START_IDX, dtype=torch.long, device=device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(max_len - 1):
                # Embed generated sequence
                tgt_embedded = self.token_embedding(generated)
                tgt_embedded = self.pos_encoding(tgt_embedded)

                # Causal mask
                tgt_len = generated.size(1)
                causal_mask = self._generate_causal_mask(tgt_len, device)

                # Transformer decoder
                decoder_output = self.transformer_decoder(
                    tgt=tgt_embedded,
                    memory=latent_memory,
                    tgt_mask=causal_mask
                )

                # Get last position
                last_output = decoder_output[:, -1:, :]

                # Copy attention
                copy_context, copy_attn = self.copy_attention(
                    query=last_output,
                    key=src_embedded,
                    value=src_embedded,
                    key_padding_mask=src_padding_mask,
                    return_attn=True
                )

                # Copy gate
                gate_input = torch.cat([last_output, copy_context], dim=-1)
                copy_prob = self.copy_gate(gate_input)

                # Vocabulary distribution
                vocab_logits = self.vocab_proj(last_output).squeeze(1)

                # Apply temperature
                if temperature != 1.0:
                    vocab_logits = vocab_logits / temperature

                vocab_prob = F.softmax(vocab_logits, dim=-1)

                # Copy distribution
                copy_dist = torch.zeros(batch_size, self.vocab_size, device=device)
                copy_attn_squeezed = copy_attn.squeeze(1)  # (batch, src_len)
                copy_dist.scatter_add_(1, input_tokens, copy_attn_squeezed)

                # Blend
                copy_prob_squeezed = copy_prob.squeeze(1).squeeze(1)  # (batch,)
                final_prob = (1 - copy_prob_squeezed.unsqueeze(1)) * vocab_prob + \
                             copy_prob_squeezed.unsqueeze(1) * copy_dist

                # Top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = final_prob < torch.topk(final_prob, top_k)[0][..., -1, None]
                    final_prob[indices_to_remove] = 0
                    final_prob = final_prob / final_prob.sum(dim=-1, keepdim=True)

                # Sample or argmax
                if temperature < 0.01:
                    next_token = final_prob.argmax(dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(final_prob, num_samples=1)

                # Update finished
                finished = finished | (next_token.squeeze(-1) == END_IDX)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

                if finished.all():
                    break

            # Convert to formulas
            formulas = []
            for i in range(batch_size):
                formula = indices_to_formula(generated[i, 1:])
                formulas.append(formula)

        return formulas


def create_pointer_generator_decoder(
    latent_dim: int = 128,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    max_len: int = 50
) -> PointerGeneratorDecoder:
    """Factory function for creating pointer-generator decoder."""
    return PointerGeneratorDecoder(
        latent_dim=latent_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len
    )
