"""
Soft-Token Scheduled Sampling for Autoregressive Decoders.

Addresses exposure bias by training with soft (probabilistic) tokens
instead of hard (discrete) tokens. Unlike standard scheduled sampling
which samples discrete tokens (breaking differentiability), soft-token
sampling propagates probability distributions through the decoder.

Key insight: Instead of embedding(argmax(logits)), we use softmax(logits) @ embedding.weight
This creates a "soft embedding" that's a weighted average of all token embeddings.

Benefits:
1. Fully differentiable (gradients flow through soft tokens)
2. Preserves uncertainty (doesn't commit to single token)
3. Smoother training signal than hard sampling
4. Natural interpolation between teacher forcing and free generation

Based on: "Soft-Token Trajectory Forecasting" concepts and
"Scheduled Sampling for Transformers" (with soft modification)

Usage:
    soft_sampler = SoftTokenScheduler(n_epochs=300, start_ratio=0.0, end_ratio=0.5)
    mixer = SoftTokenMixer(embedding_layer)

    for epoch in range(n_epochs):
        soft_ratio = soft_sampler.get_ratio(epoch)

        # During forward pass
        soft_embeddings = mixer.mix_embeddings(
            ground_truth_tokens,
            predicted_logits,
            soft_ratio,
            temperature=1.0
        )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SoftTokenScheduler:
    """
    Schedule for soft-token sampling ratio during training.

    Starts with teacher forcing (soft_ratio=0) and gradually increases
    the use of soft predictions (soft_ratio→end_ratio).

    Args:
        n_epochs: Total training epochs
        start_ratio: Initial soft-token ratio (0 = full teacher forcing)
        end_ratio: Final soft-token ratio (0.5 = 50% soft, 50% teacher)
        warmup_epochs: Epochs before starting to increase ratio
        schedule: 'linear', 'cosine', or 'exponential'
    """

    def __init__(
        self,
        n_epochs: int = 300,
        start_ratio: float = 0.0,
        end_ratio: float = 0.5,
        warmup_epochs: int = 10,
        schedule: str = 'linear'
    ):
        self.n_epochs = n_epochs
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule

        self.effective_epochs = n_epochs - warmup_epochs

    def get_ratio(self, epoch: int) -> float:
        """
        Get soft-token ratio for given epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Soft-token ratio in [start_ratio, end_ratio]
        """
        if epoch < self.warmup_epochs:
            return self.start_ratio

        # Progress through post-warmup epochs
        progress = (epoch - self.warmup_epochs) / max(1, self.effective_epochs)
        progress = min(1.0, progress)

        if self.schedule == 'linear':
            ratio = self.start_ratio + progress * (self.end_ratio - self.start_ratio)

        elif self.schedule == 'cosine':
            # Cosine schedule (slower start, faster middle, slower end)
            cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
            ratio = self.start_ratio + cosine_progress * (self.end_ratio - self.start_ratio)

        elif self.schedule == 'exponential':
            # Exponential (slow start, accelerating)
            exp_progress = (np.exp(progress) - 1) / (np.e - 1)
            ratio = self.start_ratio + exp_progress * (self.end_ratio - self.start_ratio)

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return ratio


class SoftTokenMixer(nn.Module):
    """
    Mix teacher-forced embeddings with soft predicted embeddings.

    Given ground truth tokens and model predictions (logits), creates
    a mixture of:
    - Hard embeddings: embedding(ground_truth)
    - Soft embeddings: softmax(logits/temp) @ embedding.weight

    The mixture ratio determines exposure to model's own predictions.

    Args:
        embedding: Token embedding layer
        temperature: Temperature for softmax (lower = sharper, higher = softer)
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        temperature: float = 1.0
    ):
        super().__init__()
        self.embedding = embedding
        self.temperature = temperature
        self.vocab_size = embedding.num_embeddings
        self.embed_dim = embedding.embedding_dim

    def soft_embed(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Create soft embeddings from logits.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            temperature: Optional temperature override

        Returns:
            Soft embeddings (batch, seq_len, embed_dim)
        """
        temp = temperature or self.temperature

        # Softmax over vocabulary
        probs = F.softmax(logits / temp, dim=-1)  # (batch, seq_len, vocab_size)

        # Weighted sum of embeddings: probs @ embedding.weight
        # embedding.weight is (vocab_size, embed_dim)
        soft_emb = torch.matmul(probs, self.embedding.weight)  # (batch, seq_len, embed_dim)

        return soft_emb

    def hard_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Create hard embeddings from token indices.

        Args:
            tokens: Token indices (batch, seq_len)

        Returns:
            Hard embeddings (batch, seq_len, embed_dim)
        """
        return self.embedding(tokens)

    def mix_embeddings(
        self,
        ground_truth_tokens: torch.Tensor,
        predicted_logits: torch.Tensor,
        soft_ratio: float,
        temperature: Optional[float] = None,
        position_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Mix hard (teacher) and soft (predicted) embeddings.

        Args:
            ground_truth_tokens: Target token indices (batch, seq_len)
            predicted_logits: Model predictions (batch, seq_len, vocab_size)
            soft_ratio: Ratio of soft embeddings to use (0=teacher, 1=soft)
            temperature: Temperature for soft embeddings
            position_mask: Optional mask for which positions to apply soft tokens
                          (batch, seq_len) True = use soft, False = use hard

        Returns:
            Mixed embeddings (batch, seq_len, embed_dim)
        """
        hard_emb = self.hard_embed(ground_truth_tokens)

        if soft_ratio <= 0:
            return hard_emb

        soft_emb = self.soft_embed(predicted_logits, temperature)

        if position_mask is not None:
            # Apply soft tokens only at masked positions
            position_mask = position_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            mixed = hard_emb * (1 - position_mask) + soft_emb * position_mask
            # Then apply ratio
            mixed = hard_emb * (1 - soft_ratio) + mixed * soft_ratio
        else:
            # Simple linear interpolation
            mixed = hard_emb * (1 - soft_ratio) + soft_emb * soft_ratio

        return mixed

    def forward(
        self,
        ground_truth_tokens: torch.Tensor,
        predicted_logits: torch.Tensor,
        soft_ratio: float,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """Forward pass - alias for mix_embeddings."""
        return self.mix_embeddings(
            ground_truth_tokens, predicted_logits, soft_ratio, temperature
        )


class SoftTokenDecoder(nn.Module):
    """
    Wrapper that adds soft-token training capability to a transformer decoder.

    During training:
    1. First pass: Get predictions with teacher forcing
    2. Second pass: Use soft mixture of predictions and ground truth

    This is slightly more expensive (2 forward passes) but gives much
    better exposure bias handling than hard scheduled sampling.

    Args:
        decoder: The underlying transformer decoder
        soft_scheduler: SoftTokenScheduler for ratio scheduling
        temperature: Temperature for soft embeddings
    """

    def __init__(
        self,
        decoder: nn.Module,
        soft_scheduler: Optional[SoftTokenScheduler] = None,
        temperature: float = 1.0
    ):
        super().__init__()
        self.decoder = decoder
        self.soft_scheduler = soft_scheduler
        self.temperature = temperature

        # Create soft token mixer using decoder's embedding
        self.mixer = SoftTokenMixer(
            decoder.token_embedding,
            temperature=temperature
        )

    def forward_with_soft_tokens(
        self,
        z: torch.Tensor,
        target_tokens: torch.Tensor,
        soft_ratio: float,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with soft-token mixing.

        Args:
            z: Latent vectors (batch, latent_dim)
            target_tokens: Target token indices (batch, seq_len)
            soft_ratio: Ratio of soft tokens to use
            temperature: Temperature for soft embeddings

        Returns:
            Tuple of (logits, predictions)
        """
        temp = temperature or self.temperature

        if soft_ratio <= 0:
            # Pure teacher forcing — return only (logits, predictions) for API compat
            result = self.decoder(z, target_tokens)
            return result[0], result[1]

        # First pass: get predictions with teacher forcing
        with torch.no_grad():
            first_logits, *_extra = self.decoder(z, target_tokens)

        # Create soft embeddings for input (all but last token)
        input_tokens = target_tokens[:, :-1]

        # Shift logits to align with inputs (predict position i uses logits from i-1)
        # first_logits predicts positions 1 to seq_len-1
        # We need logits for positions 0 to seq_len-2 to create soft inputs
        # Use a simple approach: for position i, use prediction from position i (which predicted i+1)
        # But we need to be careful about the shift

        # Actually simpler: just use the logits as-is for soft embedding
        # first_logits[i] = prediction for position i+1 (given inputs 0..i)
        # For soft input at position i (i>0), we use softmax(first_logits[i-1])

        # Pad logits at start (first position always uses hard START token)
        batch_size, seq_len_minus_1, vocab_size = first_logits.shape

        # Create soft embeddings
        # For position 0: always use hard START token
        # For position i>0: mix hard ground_truth[i] with soft(logits[i-1])

        # Get hard embeddings
        hard_emb = self.decoder.token_embedding(input_tokens)

        # Get soft embeddings from shifted logits
        # Pad with zeros at start (won't be used due to position 0 being hard)
        padded_logits = F.pad(first_logits[:, :-1, :], (0, 0, 1, 0), value=0)
        soft_emb = self.mixer.soft_embed(padded_logits, temp)

        # Create position mask: position 0 is always hard
        position_mask = torch.ones_like(input_tokens, dtype=torch.bool)
        position_mask[:, 0] = False  # First position always teacher-forced

        # Mix embeddings
        position_mask_expanded = position_mask.unsqueeze(-1).float()
        mixed_emb = torch.where(
            position_mask_expanded.bool().expand_as(hard_emb),
            hard_emb * (1 - soft_ratio) + soft_emb * soft_ratio,
            hard_emb
        )

        # Second pass: forward with mixed embeddings
        # We need to bypass the embedding layer and feed directly
        mixed_emb = self.decoder.pos_encoding(mixed_emb)

        # Create memory from latent
        memory = self.decoder._create_memory(z)

        # Create causal mask
        seq_len = input_tokens.size(1)
        causal_mask = self.decoder._generate_causal_mask(seq_len, z.device)

        # Create padding mask
        tgt_key_padding_mask = (input_tokens == self.decoder.token_embedding.padding_idx)

        # Transformer forward
        output = self.decoder.transformer_decoder(
            tgt=mixed_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Project to vocabulary
        logits = self.decoder.output_proj(output)
        predictions = logits.argmax(dim=-1)

        return logits, predictions

    def forward(
        self,
        z: torch.Tensor,
        target_tokens: torch.Tensor,
        epoch: Optional[int] = None,
        soft_ratio: Optional[float] = None,
        teacher_forcing_ratio: float = 1.0  # For API compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional soft-token mixing.

        Args:
            z: Latent vectors
            target_tokens: Target token indices
            epoch: Current epoch (for scheduler)
            soft_ratio: Override soft ratio (if None, uses scheduler)
            teacher_forcing_ratio: Ignored (for API compatibility)

        Returns:
            Tuple of (logits, predictions)
        """
        # Determine soft ratio
        if soft_ratio is None:
            if self.soft_scheduler is not None and epoch is not None:
                soft_ratio = self.soft_scheduler.get_ratio(epoch)
            else:
                soft_ratio = 0.0

        return self.forward_with_soft_tokens(z, target_tokens, soft_ratio)


def create_soft_token_training_components(
    decoder: nn.Module,
    n_epochs: int = 300,
    start_ratio: float = 0.0,
    end_ratio: float = 0.5,
    warmup_epochs: int = 10,
    temperature: float = 1.0,
    schedule: str = 'linear'
) -> Tuple[SoftTokenScheduler, SoftTokenMixer]:
    """
    Factory function to create soft-token training components.

    Args:
        decoder: The transformer decoder
        n_epochs: Total training epochs
        start_ratio: Initial soft-token ratio
        end_ratio: Final soft-token ratio
        warmup_epochs: Warmup epochs before increasing ratio
        temperature: Temperature for soft embeddings
        schedule: Schedule type ('linear', 'cosine', 'exponential')

    Returns:
        Tuple of (SoftTokenScheduler, SoftTokenMixer)
    """
    scheduler = SoftTokenScheduler(
        n_epochs=n_epochs,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
        warmup_epochs=warmup_epochs,
        schedule=schedule
    )

    mixer = SoftTokenMixer(
        embedding=decoder.token_embedding,
        temperature=temperature
    )

    return scheduler, mixer
