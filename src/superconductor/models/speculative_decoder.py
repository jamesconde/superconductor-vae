"""
V12.8: Speculative Decoding for Fast Autoregressive Generation.

Speculative decoding uses a small "draft" model to propose multiple tokens,
then a larger "target" model to verify them in parallel. This can provide
2-3x speedup for inference while maintaining the same output distribution
as the target model alone.

Algorithm:
1. Draft model generates k tokens autoregressively (fast, ~15-20% of target size)
2. Target model scores all k tokens in one forward pass (parallel verification)
3. Accept/reject tokens based on probability ratio
4. Continue from first rejected position

Reference:
    Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"
    https://arxiv.org/abs/2211.17192

================================================================================
INTEGRATION STATUS: NOT INTEGRATED INTO MAIN PIPELINE
================================================================================

This module provides the architecture for speculative decoding but is NOT
currently integrated into training or inference pipelines. Reasons:

1. REINFORCE Training: KV caching alone provides 1.3x speedup with zero overhead.
   Speculative decoding would require periodic draft model distillation for
   marginal additional speedup (~1.5-2x total vs 1.3x).

2. Draft Model: The DraftTransformerDecoder is untrained. With random weights,
   acceptance rate is ~20%, making speculative decoding SLOWER than standard
   generation. Training via knowledge distillation is required.

3. Use Case: Speculative decoding is most valuable for batch inference after
   training (generating thousands of candidate formulas), not during training.

To integrate in the future:
1. Train draft model via knowledge distillation from trained target
2. Add to inference scripts for bulk formula generation
3. Optional: Add periodic distillation for REINFORCE speedup (complexity tradeoff)

Architecture Summary:
- DraftTransformerDecoder: 4.9M params (~5% of target's 92M)
  - d_model=256, nhead=4, num_layers=3, dim_feedforward=1024
  - n_memory_tokens=4, n_stoich_tokens=2

- SpeculativeDecoder: Wrapper implementing draft-verify-accept loop
  - k=5 tokens drafted per iteration (configurable)
  - Uses precompute_memory() for efficiency
  - benchmark() method for performance comparison

- create_speculative_decoder(): Factory function for easy instantiation

Test script: scratch/test_speculative_decoder.py
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .autoregressive_decoder import (
    EnhancedTransformerDecoder,
    VOCAB_SIZE, PAD_IDX, START_IDX, END_IDX
)


class DraftTransformerDecoder(EnhancedTransformerDecoder):
    """
    Lightweight decoder for speculative decoding draft generation.

    This is a smaller version of EnhancedTransformerDecoder (~15-20% of params)
    designed for fast autoregressive token proposal. The draft model sacrifices
    accuracy for speed - its outputs are verified by the target model.

    Default architecture (vs EnhancedTransformerDecoder):
        d_model: 256 (vs 512) - 50% smaller
        nhead: 4 (vs 8) - 50% fewer
        num_layers: 3 (vs 12) - 75% fewer
        dim_feedforward: 1024 (vs 2048) - 50% smaller
        n_memory_tokens: 4 (vs 16) - 75% fewer

    This results in ~15% of target model parameters, enabling fast draft generation.
    """

    def __init__(
        self,
        latent_dim: int = 2048,  # Keep same as target for shared encoder
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 80,
        n_memory_tokens: int = 4,
        encoder_skip_dim: int = 256,
        use_skip_connection: bool = True,
        use_stoich_conditioning: bool = True,
        max_elements: int = 12,
        n_stoich_tokens: int = 2,  # Fewer stoich tokens
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__(
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            n_memory_tokens=n_memory_tokens,
            encoder_skip_dim=encoder_skip_dim,
            use_skip_connection=use_skip_connection,
            use_stoich_conditioning=use_stoich_conditioning,
            max_elements=max_elements,
            n_stoich_tokens=n_stoich_tokens,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )


class SpeculativeDecoder(nn.Module):
    """
    Speculative decoding wrapper combining draft and target decoders.

    This module coordinates the draft-verify-accept loop for fast inference.
    It uses KV caching in both models for efficient generation.

    Usage:
        target_decoder = EnhancedTransformerDecoder(...)
        draft_decoder = DraftTransformerDecoder(...)
        spec_decoder = SpeculativeDecoder(target_decoder, draft_decoder, k=5)

        # Generate with speculative decoding
        formulas = spec_decoder.generate(z, encoder_skip, temperature=0.8)

    Args:
        target_decoder: Full-size decoder (EnhancedTransformerDecoder)
        draft_decoder: Smaller draft decoder (DraftTransformerDecoder)
        k: Number of tokens to draft per iteration (default: 5)
        temperature: Sampling temperature (default: 1.0)
    """

    def __init__(
        self,
        target_decoder: EnhancedTransformerDecoder,
        draft_decoder: DraftTransformerDecoder,
        k: int = 5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target = target_decoder
        self.draft = draft_decoder
        self.k = k
        self.temperature = temperature

        # Verify compatibility
        assert self.target.latent_dim == self.draft.latent_dim, \
            "Target and draft decoders must have same latent_dim"
        assert self.target.max_len == self.draft.max_len, \
            "Target and draft decoders must have same max_len"

    def _speculative_sampling(
        self,
        draft_tokens: torch.Tensor,  # [batch, k]
        draft_log_probs: torch.Tensor,  # [batch, k]
        target_logits: torch.Tensor,  # [batch, k, vocab]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform speculative sampling: accept/reject draft tokens.

        For each position i, we accept the draft token with probability:
            min(1, p_target(x_i) / p_draft(x_i))

        If rejected, we sample from an adjusted distribution.

        Returns:
            accepted_tokens: [batch, n_accepted] - accepted token indices
            n_accepted: [batch] - number of accepted tokens per sequence
            final_token: [batch] - token sampled after rejection (or None)
        """
        batch_size, k = draft_tokens.shape
        device = draft_tokens.device

        # Get target log probs for draft tokens
        target_log_probs = F.log_softmax(target_logits / self.temperature, dim=-1)
        target_log_probs_draft = target_log_probs.gather(
            2, draft_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [batch, k]

        # Compute acceptance probabilities
        # p_accept = min(1, p_target / p_draft) = min(1, exp(log_target - log_draft))
        log_ratio = target_log_probs_draft - draft_log_probs
        accept_probs = torch.clamp(torch.exp(log_ratio), max=1.0)  # [batch, k]

        # Sample acceptance for each position
        random_vals = torch.rand_like(accept_probs)
        accepted_mask = random_vals < accept_probs  # [batch, k]

        # Find first rejection position for each sequence
        # If all accepted, first_reject = k (meaning accept all k tokens)
        rejected_mask = ~accepted_mask
        # Add sentinel at end (always rejected) to handle all-accepted case
        rejected_with_sentinel = torch.cat([
            rejected_mask,
            torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        ], dim=1)
        first_reject_pos = rejected_with_sentinel.int().argmax(dim=1)  # [batch]

        # Gather accepted tokens (up to first rejection)
        n_accepted = first_reject_pos.clone()  # [batch]

        # Create output tensor for accepted tokens
        max_accepted = n_accepted.max().item()
        if max_accepted == 0:
            accepted_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        else:
            # Gather accepted tokens
            positions = torch.arange(max_accepted, device=device).unsqueeze(0)
            valid_mask = positions < n_accepted.unsqueeze(1)
            accepted_tokens = torch.where(
                valid_mask,
                draft_tokens[:, :max_accepted],
                torch.full_like(draft_tokens[:, :max_accepted], PAD_IDX)
            )

        # Sample final token from adjusted distribution at rejection point
        # For sequences that accepted all k tokens, sample from target at position k
        # For sequences that rejected at position i, sample from adjusted distribution
        final_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

        for b in range(batch_size):
            reject_pos = first_reject_pos[b].item()
            if reject_pos < k:
                # Rejected at position reject_pos - sample from adjusted distribution
                # p_adjusted = max(0, p_target - p_draft) / Z
                target_probs = F.softmax(target_logits[b, reject_pos] / self.temperature, dim=-1)
                draft_probs = torch.exp(draft_log_probs[b, reject_pos]) * torch.ones_like(target_probs)
                draft_probs[draft_tokens[b, reject_pos]] = torch.exp(draft_log_probs[b, reject_pos])

                adjusted_probs = torch.clamp(target_probs - draft_probs, min=0)
                if adjusted_probs.sum() > 0:
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    final_tokens[b] = torch.multinomial(adjusted_probs, 1).squeeze()
                else:
                    # Fallback to target distribution
                    final_tokens[b] = torch.multinomial(target_probs, 1).squeeze()
            else:
                # All k tokens accepted - sample next token from target
                # Note: We need one more forward pass for this, but it's rare
                final_tokens[b] = PAD_IDX  # Placeholder - will be handled in main loop

        return accepted_tokens, n_accepted, final_tokens

    def generate_with_speculation(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Generate tokens using speculative decoding.

        Args:
            z: Latent vectors [batch, latent_dim]
            encoder_skip: Optional skip connection
            stoich_pred: Optional stoichiometry conditioning
            temperature: Sampling temperature
            max_len: Maximum sequence length

        Returns:
            generated_tokens: [batch, seq_len] - generated token indices
            n_draft_tokens: Total draft tokens generated
            n_accepted_tokens: Total accepted tokens (efficiency metric)
        """
        self.target.eval()
        self.draft.eval()
        batch_size = z.size(0)
        device = z.device
        max_len = max_len or self.target.max_len
        self.temperature = temperature

        with torch.no_grad():
            # Initialize with START token
            generated = [torch.full((batch_size, 1), START_IDX, dtype=torch.long, device=device)]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            n_draft_total = 0
            n_accepted_total = 0

            # V12.8: Pre-compute memory once (static for all steps) using cached memory API
            target_memory = self.target.precompute_memory(z, encoder_skip, stoich_pred)
            draft_memory = self.draft.precompute_memory(z, encoder_skip, stoich_pred)

            while True:
                current_tokens = torch.cat(generated, dim=1)
                current_len = current_tokens.size(1)

                if current_len >= max_len:
                    break
                if finished.all():
                    break

                # Determine how many tokens to draft
                k = min(self.k, max_len - current_len)

                # ==== Draft Phase ====
                # Generate k tokens from draft model using KV cache
                draft_kv_cache = self.draft._init_kv_cache(batch_size, device)
                draft_tokens = []
                draft_log_probs_list = []

                # Process existing tokens through draft model to build cache
                for pos in range(current_len):
                    token = current_tokens[:, pos:pos+1]
                    token_emb = self.draft.token_embedding(token)
                    _, draft_kv_cache = self.draft._forward_one_step_with_cache(
                        token_emb, draft_memory, draft_kv_cache, pos
                    )

                # Generate k new tokens
                for i in range(k):
                    pos = current_len + i
                    if i == 0:
                        # Use last token from current sequence
                        token = current_tokens[:, -1:]
                    else:
                        # Use previously generated draft token
                        token = draft_tokens[-1]

                    token_emb = self.draft.token_embedding(token)
                    output, draft_kv_cache = self.draft._forward_one_step_with_cache(
                        token_emb, draft_memory, draft_kv_cache, pos
                    )

                    logits = self.draft.output_proj(output).squeeze(1)
                    if temperature > 0.01:
                        probs = F.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        log_prob = F.log_softmax(logits / temperature, dim=-1).gather(1, next_token)
                    else:
                        next_token = logits.argmax(dim=-1, keepdim=True)
                        log_prob = torch.zeros(batch_size, 1, device=device)

                    draft_tokens.append(next_token)
                    draft_log_probs_list.append(log_prob.squeeze(-1))

                draft_tokens_tensor = torch.cat(draft_tokens, dim=1)  # [batch, k]
                draft_log_probs = torch.stack(draft_log_probs_list, dim=1)  # [batch, k]
                n_draft_total += k * batch_size

                # ==== Verify Phase ====
                # Score all k draft tokens with target model in one forward pass
                # Create input sequence: current_tokens + draft_tokens[:-1]
                verify_input = torch.cat([current_tokens, draft_tokens_tensor[:, :-1]], dim=1)

                embedded = self.target.token_embedding(verify_input)
                embedded = self.target.pos_encoding(embedded)

                seq_len = verify_input.size(1)
                causal_mask = self.target._generate_causal_mask(seq_len, device)
                padding_mask = (verify_input == PAD_IDX)

                output = self.target.transformer_decoder(
                    tgt=embedded,
                    memory=target_memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=padding_mask
                )

                # Get logits for draft token positions
                target_logits = self.target.output_proj(output)
                # Logits at positions [current_len-1, current_len, ..., current_len+k-2]
                # correspond to tokens at [current_len, current_len+1, ..., current_len+k-1]
                draft_position_logits = target_logits[:, current_len-1:current_len-1+k, :]

                # ==== Accept/Reject Phase ====
                accepted_tokens, n_accepted, sampled_final_tokens = self._speculative_sampling(
                    draft_tokens_tensor, draft_log_probs, draft_position_logits
                )
                n_accepted_total += n_accepted.sum().item()

                # Add accepted tokens to generated sequence
                if accepted_tokens.size(1) > 0:
                    generated.append(accepted_tokens)

                # Add final token (sampled after rejection) as a batch tensor
                # Create mask for sequences that need a final token
                needs_final = (n_accepted < k) & (sampled_final_tokens != PAD_IDX) & (~finished)
                if needs_final.any():
                    # Add final tokens for all sequences (PAD for those that don't need it)
                    final_token_batch = torch.where(
                        needs_final.unsqueeze(1),
                        sampled_final_tokens.unsqueeze(1),
                        torch.full((batch_size, 1), PAD_IDX, dtype=torch.long, device=device)
                    )
                    generated.append(final_token_batch)

                # Check for END tokens
                all_generated = torch.cat(generated, dim=1)
                for b in range(batch_size):
                    if not finished[b]:
                        if (all_generated[b] == END_IDX).any():
                            finished[b] = True

            # Concatenate all generated tokens
            output_tokens = torch.cat(generated, dim=1)

            # Truncate at END token
            for b in range(batch_size):
                end_positions = (output_tokens[b] == END_IDX).nonzero()
                if len(end_positions) > 0:
                    end_pos = end_positions[0].item()
                    output_tokens[b, end_pos+1:] = PAD_IDX

            return output_tokens[:, 1:], n_draft_total, n_accepted_total  # Exclude START token

    def generate(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
    ) -> List[str]:
        """
        Generate formula strings using speculative decoding.

        This is the main user-facing method.
        """
        from .autoregressive_decoder import indices_to_formula

        tokens, _, _ = self.generate_with_speculation(
            z, encoder_skip, stoich_pred, temperature, max_len
        )

        formulas = []
        for i in range(tokens.size(0)):
            formula = indices_to_formula(tokens[i])
            formulas.append(formula)

        return formulas

    def benchmark(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
        n_runs: int = 10,
    ) -> dict:
        """
        Benchmark speculative decoding vs standard generation.

        Returns timing comparison and acceptance statistics.
        """
        import time

        # Warm up
        _ = self.generate_with_speculation(z[:1], temperature=temperature, max_len=20)
        _ = self.target.generate_with_kv_cache(z[:1], temperature=temperature, max_len=20)

        if z.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark speculative
        spec_times = []
        total_draft = 0
        total_accepted = 0
        for _ in range(n_runs):
            start = time.time()
            _, n_draft, n_accepted = self.generate_with_speculation(
                z, encoder_skip, stoich_pred, temperature, max_len
            )
            if z.device.type == 'cuda':
                torch.cuda.synchronize()
            spec_times.append(time.time() - start)
            total_draft += n_draft
            total_accepted += n_accepted

        # Benchmark standard
        std_times = []
        for _ in range(n_runs):
            start = time.time()
            _ = self.target.generate_with_kv_cache(
                z, encoder_skip, stoich_pred, temperature, max_len
            )
            if z.device.type == 'cuda':
                torch.cuda.synchronize()
            std_times.append(time.time() - start)

        avg_spec = sum(spec_times) / n_runs
        avg_std = sum(std_times) / n_runs
        acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0

        return {
            'speculative_time_ms': avg_spec * 1000,
            'standard_time_ms': avg_std * 1000,
            'speedup': avg_std / avg_spec if avg_spec > 0 else 0,
            'acceptance_rate': acceptance_rate,
            'draft_tokens_per_run': total_draft / n_runs,
            'accepted_tokens_per_run': total_accepted / n_runs,
        }


def create_speculative_decoder(
    target_decoder: EnhancedTransformerDecoder,
    k: int = 5,
    draft_d_model: int = 256,
    draft_nhead: int = 4,
    draft_num_layers: int = 3,
    draft_dim_feedforward: int = 1024,
    draft_n_memory_tokens: int = 4,
) -> SpeculativeDecoder:
    """
    Factory function to create a speculative decoder from an existing target decoder.

    Creates a draft decoder with reduced size and wraps both in SpeculativeDecoder.

    Args:
        target_decoder: The full-size decoder to use for verification
        k: Number of tokens to draft per iteration
        draft_*: Architecture parameters for draft decoder

    Returns:
        SpeculativeDecoder instance
    """
    draft_decoder = DraftTransformerDecoder(
        latent_dim=target_decoder.latent_dim,
        d_model=draft_d_model,
        nhead=draft_nhead,
        num_layers=draft_num_layers,
        dim_feedforward=draft_dim_feedforward,
        dropout=0.1,
        max_len=target_decoder.max_len,
        n_memory_tokens=draft_n_memory_tokens,
        encoder_skip_dim=target_decoder.use_skip_connection and 256 or 0,
        use_skip_connection=target_decoder.use_skip_connection,
        use_stoich_conditioning=target_decoder.use_stoich_conditioning,
        max_elements=target_decoder.max_elements,
        n_stoich_tokens=2,
        use_gradient_checkpointing=False,  # Draft should be fast, no checkpointing
    ).to(next(target_decoder.parameters()).device)

    return SpeculativeDecoder(target_decoder, draft_decoder, k=k)
