"""
REINFORCE-based Loss for Chemical Formula Reconstruction.

Standard cross-entropy loss optimizes per-token accuracy, but what we really
want is EXACT SEQUENCE MATCH. REINFORCE allows training on non-differentiable
objectives like exact match or semantic correctness.

Key Components:
1. Sample tokens from the model's softmax distribution
2. Compute a reward (exact match = 1, else = 0)
3. Use policy gradient: loss = -reward * log_prob(sampled_tokens)

This directly optimizes for exact match, not just per-token accuracy!

Variance Reduction Techniques:
- Baseline subtraction: R - baseline reduces variance
- Entropy regularization: Encourages exploration
- Mixed loss: Combine with supervised CE for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from ..models.autoregressive_decoder import PAD_IDX, START_IDX, END_IDX


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    exact_match_reward: float = 1.0      # Reward for exact sequence match
    partial_match_reward: float = 0.0    # Base reward for partial matches
    per_token_bonus: float = 0.01        # Small bonus per correct token
    element_correct_bonus: float = 0.1   # Bonus for each correct element
    length_penalty: float = 0.0          # Penalty for wrong length


def compute_exact_match_reward(
    sampled_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    config: Optional[RewardConfig] = None
) -> torch.Tensor:
    """
    Compute reward based on exact sequence match.

    Args:
        sampled_tokens: [batch, seq_len] sampled token indices
        target_tokens: [batch, seq_len] target token indices
        mask: [batch, seq_len] valid positions (non-PAD)
        config: reward configuration

    Returns:
        rewards: [batch] reward for each sequence
    """
    if config is None:
        config = RewardConfig()

    if mask is None:
        mask = target_tokens != PAD_IDX

    batch_size = sampled_tokens.shape[0]
    device = sampled_tokens.device

    rewards = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        valid_mask = mask[i]
        sampled = sampled_tokens[i][valid_mask]
        target = target_tokens[i][valid_mask]

        # Exact match check
        if torch.equal(sampled, target):
            rewards[i] = config.exact_match_reward
        else:
            # Partial credit based on token accuracy
            correct = (sampled == target).float().sum()
            total = valid_mask.sum().float()
            token_accuracy = correct / total

            rewards[i] = (
                config.partial_match_reward +
                config.per_token_bonus * correct
            )

    return rewards


def compute_semantic_reward(
    sampled_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    exact_match_bonus: float = 0.2,  # Bonus for exact match on top of partial
) -> torch.Tensor:
    """
    Compute HIERARCHICAL reward based on semantic correctness.

    Reward Structure (designed to point toward exact match):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1.0   │ Exact match (all tokens correct)
    ──────┼─────────────────────────────────────────────────
    0.8   │ All elements correct + all fractions correct
    0.7   │ All elements correct + most fractions correct
    0.6   │ All elements correct + some fraction errors
    ──────┼─────────────────────────────────────────────────
    0.4   │ Most elements correct (in order)
    0.2   │ Some elements correct (in order)
    0.1   │ Elements present but wrong order
    ──────┼─────────────────────────────────────────────────
    0.0   │ Wrong elements or garbage
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    This creates a smooth gradient toward exact match while heavily
    penalizing structural errors (wrong elements, wrong order).

    Args:
        sampled_tokens: [batch, seq_len] sampled token indices
        target_tokens: [batch, seq_len] target token indices
        mask: [batch, seq_len] valid positions
        exact_match_bonus: additional reward for exact match

    Returns:
        rewards: [batch] semantic reward for each sequence
    """
    from .semantic_unit_loss import parse_tokens_to_semantic_units

    if mask is None:
        mask = target_tokens != PAD_IDX

    batch_size = sampled_tokens.shape[0]
    device = sampled_tokens.device
    rewards = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        sampled_units = parse_tokens_to_semantic_units(sampled_tokens[i])
        target_units = parse_tokens_to_semantic_units(target_tokens[i])

        # Build full strings for exact match check
        sampled_str = ''.join(u.value for u in sampled_units)
        target_str = ''.join(u.value for u in target_units)

        # =============================================
        # LEVEL 1: Check for exact match (reward = 1.0)
        # =============================================
        if sampled_str == target_str:
            rewards[i] = 1.0
            continue

        # Extract semantic units by type
        sampled_elements = [u.value for u in sampled_units if u.unit_type == 'element']
        target_elements = [u.value for u in target_units if u.unit_type == 'element']

        sampled_fractions = [u.value for u in sampled_units if u.unit_type == 'fraction']
        target_fractions = [u.value for u in target_units if u.unit_type == 'fraction']

        sampled_subscripts = [u.value for u in sampled_units if u.unit_type == 'subscript']
        target_subscripts = [u.value for u in target_units if u.unit_type == 'subscript']

        # =============================================
        # LEVEL 2: Check element correctness (most important!)
        # =============================================
        n_target_elements = len(target_elements)
        n_sampled_elements = len(sampled_elements)

        if n_target_elements == 0:
            element_score = 1.0 if n_sampled_elements == 0 else 0.0
        else:
            # Count elements correct IN ORDER (position matters!)
            elements_correct_in_order = 0
            for j in range(min(n_sampled_elements, n_target_elements)):
                if sampled_elements[j] == target_elements[j]:
                    elements_correct_in_order += 1
                else:
                    break  # Stop at first mismatch - order matters!

            # Penalize wrong number of elements
            length_penalty = abs(n_sampled_elements - n_target_elements) / max(n_target_elements, 1)

            element_score = (elements_correct_in_order / n_target_elements) * (1 - 0.5 * length_penalty)

        # If elements are completely wrong, give very low reward
        if element_score < 0.5:
            rewards[i] = element_score * 0.4  # Max 0.2 for bad elements
            continue

        # =============================================
        # LEVEL 3: Check fraction correctness
        # =============================================
        n_target_fractions = len(target_fractions)

        if n_target_fractions == 0:
            fraction_score = 1.0
        else:
            fractions_correct = 0
            for j in range(min(len(sampled_fractions), n_target_fractions)):
                if sampled_fractions[j] == target_fractions[j]:
                    fractions_correct += 1

            # Penalize wrong number of fractions
            length_penalty = abs(len(sampled_fractions) - n_target_fractions) / max(n_target_fractions, 1)
            fraction_score = (fractions_correct / n_target_fractions) * (1 - 0.3 * length_penalty)

        # =============================================
        # LEVEL 4: Check subscript correctness
        # =============================================
        n_target_subscripts = len(target_subscripts)

        if n_target_subscripts == 0:
            subscript_score = 1.0
        else:
            subscripts_correct = sum(
                1 for j in range(min(len(sampled_subscripts), n_target_subscripts))
                if sampled_subscripts[j] == target_subscripts[j]
            )
            subscript_score = subscripts_correct / n_target_subscripts

        # =============================================
        # Combine scores with hierarchy
        # =============================================
        # Elements are gate-keeping: must be mostly right to get high reward
        if element_score >= 1.0:
            # All elements correct - now fractions matter most
            # Reward range: 0.6 to 0.8 (need exact match for 1.0)
            base_reward = 0.6
            fraction_bonus = 0.15 * fraction_score
            subscript_bonus = 0.05 * subscript_score
            rewards[i] = base_reward + fraction_bonus + subscript_bonus
        else:
            # Some element errors - cap reward
            # Reward range: 0.2 to 0.5
            rewards[i] = 0.2 + 0.3 * element_score

    return rewards


class REINFORCELoss(nn.Module):
    """
    REINFORCE-based loss for sequence-level optimization.

    Based on best practices from recent LLM research (2024-2025):
    - Ahmadian et al. "Back to Basics" - REINFORCE outperforms PPO
    - RLOO (Leave-One-Out) baseline for variance reduction
    - KL regularization to prevent policy drift
    - Mixed CE + RL (MIXER approach from Ranzato et al.)

    Combines:
    1. Supervised CE loss (for stability)
    2. REINFORCE loss with RLOO baseline (for exact match optimization)
    3. KL regularization (to stay close to reference policy)
    4. Entropy regularization (for exploration)

    The total loss is:
        L = ce_weight * CE_loss + rl_weight * REINFORCE_loss
            + kl_weight * KL_divergence - entropy_weight * entropy

    Where REINFORCE_loss uses Leave-One-Out baseline from multiple samples.
    """

    def __init__(
        self,
        ce_weight: float = 0.5,          # Weight for supervised CE loss
        rl_weight: float = 0.5,          # Weight for REINFORCE loss
        entropy_weight: float = 0.01,    # Weight for entropy bonus
        kl_weight: float = 0.1,          # Weight for KL regularization
        baseline_momentum: float = 0.99,  # Momentum for baseline update
        use_semantic_reward: bool = True, # Use semantic or exact match reward
        temperature: float = 1.0,         # Sampling temperature
        min_temperature: float = 0.5,     # Minimum temperature (for annealing)
        n_samples_rloo: int = 4,          # Number of samples for RLOO baseline
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.rl_weight = rl_weight
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.baseline_momentum = baseline_momentum
        self.use_semantic_reward = use_semantic_reward
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.n_samples_rloo = n_samples_rloo

        # Running baseline for variance reduction (used when RLOO not available)
        self.register_buffer('baseline', torch.tensor(0.0))
        self.register_buffer('baseline_count', torch.tensor(0))

        # CE loss for supervised component
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='none')

        # Store reference log probs for KL computation (set during first forward)
        self.reference_log_probs = None

    def update_baseline(self, reward: torch.Tensor):
        """Update running baseline with exponential moving average."""
        batch_mean = reward.mean()
        if self.baseline_count == 0:
            self.baseline = batch_mean
        else:
            self.baseline = (
                self.baseline_momentum * self.baseline +
                (1 - self.baseline_momentum) * batch_mean
            )
        self.baseline_count += 1

    def sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample tokens from logits using temperature-scaled softmax.

        Args:
            logits: [batch, seq_len, vocab_size]
            temperature: sampling temperature (higher = more random)

        Returns:
            sampled_tokens: [batch, seq_len]
            log_probs: [batch, seq_len] log probabilities of sampled tokens
        """
        if temperature is None:
            temperature = self.temperature

        batch_size, seq_len, vocab_size = logits.shape

        # Temperature scaling
        scaled_logits = logits / temperature

        # Softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Sample from categorical distribution
        # Reshape for sampling: [batch*seq_len, vocab_size]
        probs_flat = probs.view(-1, vocab_size)
        sampled_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)

        # Reshape back: [batch, seq_len]
        sampled_tokens = sampled_flat.view(batch_size, seq_len)

        # Get log probabilities of sampled tokens
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        sampled_log_probs = log_probs.gather(
            -1, sampled_tokens.unsqueeze(-1)
        ).squeeze(-1)

        return sampled_tokens, sampled_log_probs

    def compute_rloo_baseline(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute RLOO (Leave-One-Out) baseline.

        Sample n_samples_rloo sequences per input, compute reward for each,
        and use mean of OTHER samples as baseline for each sample.

        Returns:
            advantages: [batch] advantage for each sample (reward - baseline)
            mean_log_probs: [batch] mean log prob across samples
            mean_rewards: [batch] mean reward (for logging)
        """
        batch_size = logits.shape[0]
        n_samples = self.n_samples_rloo

        all_rewards = []
        all_log_probs = []

        # Sample n times
        for _ in range(n_samples):
            sampled_tokens, sampled_log_probs = self.sample_from_logits(
                logits, temperature
            )

            # Compute reward for this sample
            if self.use_semantic_reward:
                rewards = compute_semantic_reward(sampled_tokens, targets, mask)
            else:
                rewards = compute_exact_match_reward(sampled_tokens, targets, mask)

            # Sequence-level log prob
            masked_log_probs = sampled_log_probs * mask.float()
            seq_log_prob = masked_log_probs.sum(dim=1)

            all_rewards.append(rewards)
            all_log_probs.append(seq_log_prob)

        # Stack: [n_samples, batch]
        rewards_stack = torch.stack(all_rewards, dim=0)
        log_probs_stack = torch.stack(all_log_probs, dim=0)

        # RLOO baseline: for each sample, baseline is mean of OTHER samples
        # For sample i: baseline_i = (sum of all rewards - reward_i) / (n_samples - 1)
        total_reward = rewards_stack.sum(dim=0)  # [batch]

        advantages_list = []
        for i in range(n_samples):
            # Leave-one-out baseline for sample i
            baseline_i = (total_reward - rewards_stack[i]) / (n_samples - 1)
            advantage_i = rewards_stack[i] - baseline_i
            advantages_list.append(advantage_i)

        # Use mean advantage and log prob across samples
        advantages = torch.stack(advantages_list, dim=0).mean(dim=0)  # [batch]
        mean_log_probs = log_probs_stack.mean(dim=0)  # [batch]
        mean_rewards = rewards_stack.mean(dim=0)  # [batch]

        return advantages, mean_log_probs, mean_rewards

    def compute_kl_divergence(
        self,
        current_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and reference policy.

        KL(current || reference) = sum_x current(x) * log(current(x) / reference(x))

        This regularizes toward the reference policy to prevent drift.
        """
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)

        # KL divergence per position
        current_probs = F.softmax(current_logits, dim=-1)
        kl_per_position = (current_probs * (current_log_probs - reference_log_probs)).sum(dim=-1)

        # Mask and average
        kl_per_position = kl_per_position * mask.float()
        kl_div = kl_per_position.sum(dim=1).mean()  # Sum over positions, mean over batch

        return kl_div

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        reference_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined CE + REINFORCE loss with RLOO baseline and KL regularization.

        Args:
            logits: [batch, seq_len, vocab_size] model predictions
            targets: [batch, seq_len] target token indices
            mask: [batch, seq_len] valid positions
            temperature: sampling temperature (overrides self.temperature)
            reference_logits: [batch, seq_len, vocab_size] reference policy for KL
                              (typically from initial/frozen model)

        Returns:
            Dict with loss components and metrics
        """
        batch_size, seq_len, vocab_size = logits.shape

        if mask is None:
            mask = targets != PAD_IDX

        # ===============================
        # 1. Supervised CE Loss
        # ===============================
        logits_flat = logits.contiguous().view(-1, vocab_size)
        targets_flat = targets.contiguous().view(-1)
        ce_loss_per_token = self.ce_loss(logits_flat, targets_flat)
        ce_loss_per_token = ce_loss_per_token.view(batch_size, seq_len)

        # Sum over positions, mean over batch
        ce_loss = (ce_loss_per_token * mask.float()).sum(dim=1).mean()

        # ===============================
        # 2. REINFORCE Loss with RLOO
        # ===============================
        if self.n_samples_rloo > 1:
            # Use RLOO baseline (sample multiple times)
            advantages, sequence_log_probs, rewards = self.compute_rloo_baseline(
                logits, targets, mask, temperature
            )
        else:
            # Single sample with EMA baseline
            sampled_tokens, sampled_log_probs = self.sample_from_logits(
                logits, temperature
            )

            # Compute reward
            if self.use_semantic_reward:
                rewards = compute_semantic_reward(sampled_tokens, targets, mask)
            else:
                rewards = compute_exact_match_reward(sampled_tokens, targets, mask)

            # Baseline subtraction for variance reduction
            self.update_baseline(rewards)
            advantages = rewards - self.baseline.detach()

            # Sequence log prob
            masked_log_probs = sampled_log_probs * mask.float()
            sequence_log_probs = masked_log_probs.sum(dim=1)

        # REINFORCE gradient: -advantage * log_prob
        # We want to maximize reward, so minimize negative advantage * log_prob
        reinforce_loss = -(advantages * sequence_log_probs).mean()

        # ===============================
        # 3. Entropy Regularization
        # ===============================
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]
        entropy = (entropy * mask.float()).sum(dim=1).mean()  # Average entropy

        # ===============================
        # 4. KL Regularization (optional)
        # ===============================
        if reference_logits is not None and self.kl_weight > 0:
            kl_div = self.compute_kl_divergence(logits, reference_logits, mask)
        else:
            kl_div = torch.tensor(0.0, device=logits.device)

        # ===============================
        # Combined Loss
        # ===============================
        total_loss = (
            self.ce_weight * ce_loss +
            self.rl_weight * reinforce_loss +
            self.kl_weight * kl_div -
            self.entropy_weight * entropy
        )

        # Compute metrics
        with torch.no_grad():
            # Argmax accuracy (for comparison)
            argmax_tokens = logits.argmax(dim=-1)
            argmax_correct = ((argmax_tokens == targets) & mask).float()
            argmax_accuracy = argmax_correct.sum() / mask.sum()

            # Exact match rate
            exact_matches = (
                ((argmax_tokens == targets) | ~mask).all(dim=1).float().mean()
            )

        return {
            'total': total_loss,
            'ce_loss': ce_loss,
            'reinforce_loss': reinforce_loss,
            'kl_div': kl_div,
            'entropy': entropy,
            'mean_reward': rewards.mean(),
            'baseline': self.baseline,
            'argmax_accuracy': argmax_accuracy,
            'exact_match': exact_matches,
        }


class MixedCEReinforce(nn.Module):
    """
    Mixed training strategy: Start with CE, gradually add REINFORCE.

    Based on MIXER approach (Ranzato et al. 2016) with improvements from
    recent LLM RLHF research (Ahmadian et al. 2024):
    - RLOO baseline for variance reduction
    - KL regularization to prevent policy drift
    - Curriculum from supervised to RL

    Schedule:
        - Epochs 0-warmup: Pure CE (rl_weight = 0)
        - Epochs warmup-end: Linear increase of rl_weight to final_rl_weight
    """

    def __init__(
        self,
        warmup_epochs: int = 20,
        final_rl_weight: float = 0.5,
        use_semantic_reward: bool = True,
        temperature_start: float = 1.0,
        temperature_end: float = 0.5,
        temperature_decay_epochs: int = 50,
        n_samples_rloo: int = 4,  # Number of samples for RLOO
        kl_weight: float = 0.1,   # KL regularization weight
    ):
        super().__init__()

        self.warmup_epochs = warmup_epochs
        self.final_rl_weight = final_rl_weight
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay_epochs = temperature_decay_epochs

        self.reinforce_loss = REINFORCELoss(
            ce_weight=1.0,
            rl_weight=0.0,
            use_semantic_reward=use_semantic_reward,
            temperature=temperature_start,
            n_samples_rloo=n_samples_rloo,
            kl_weight=kl_weight,
        )

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update epoch and adjust weights accordingly."""
        self.current_epoch = epoch

        # RL weight schedule
        if epoch < self.warmup_epochs:
            rl_weight = 0.0
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.warmup_epochs)
            rl_weight = min(self.final_rl_weight, self.final_rl_weight * progress)

        self.reinforce_loss.rl_weight = rl_weight
        self.reinforce_loss.ce_weight = 1.0 - rl_weight * 0.5  # Reduce CE as RL increases

        # Temperature schedule
        temp_progress = min(1.0, epoch / self.temperature_decay_epochs)
        temperature = (
            self.temperature_start +
            (self.temperature_end - self.temperature_start) * temp_progress
        )
        self.reinforce_loss.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reference_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with current epoch's weights.

        Args:
            logits: Current model predictions
            targets: Target token indices
            mask: Valid positions mask
            reference_logits: Reference policy for KL regularization
                              (typically from initial model before RL)
        """
        result = self.reinforce_loss(logits, targets, mask, reference_logits=reference_logits)
        result['rl_weight'] = torch.tensor(self.reinforce_loss.rl_weight)
        result['temperature'] = torch.tensor(self.reinforce_loss.temperature)
        return result


def test_reinforce():
    """Test REINFORCE loss computation."""
    batch_size = 4
    seq_len = 20
    vocab_size = 100

    # Random logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Make some sequences match (for testing reward)
    targets[0] = logits[0].argmax(dim=-1)  # This one will be exact match

    # Create mask
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -5:] = False  # Last 5 positions are padding

    # Test REINFORCE loss
    criterion = REINFORCELoss(use_semantic_reward=False)
    result = criterion(logits, targets, mask)

    print("REINFORCE Loss Test:")
    print(f"  Total loss: {result['total'].item():.4f}")
    print(f"  CE loss: {result['ce_loss'].item():.4f}")
    print(f"  REINFORCE loss: {result['reinforce_loss'].item():.4f}")
    print(f"  Entropy: {result['entropy'].item():.4f}")
    print(f"  Mean reward: {result['mean_reward'].item():.4f}")
    print(f"  Argmax accuracy: {result['argmax_accuracy'].item():.2%}")
    print(f"  Exact match: {result['exact_match'].item():.2%}")

    return result


def test_hierarchical_reward():
    """
    Test the hierarchical reward function with real formula examples.

    Shows how different types of errors get different rewards.
    """
    from ..models.autoregressive_decoder import TOKEN_TO_IDX, tokenize_formula, tokens_to_indices

    print("\n" + "=" * 60)
    print("HIERARCHICAL REWARD EXAMPLES")
    print("=" * 60)

    # Target formula: La(7/10)Sr(3/10)CuO4
    target_formula = "La(7/10)Sr(3/10)CuO4"

    test_cases = [
        ("La(7/10)Sr(3/10)CuO4", "Exact match"),
        ("La(7/10)Sr(3/10)CuO3", "Wrong subscript (4->3)"),
        ("La(7/10)Sr(3/10)CuO", "Missing subscript"),
        ("La(8/10)Sr(2/10)CuO4", "Wrong fractions, elements correct"),
        ("La(7/10)Sr(3/10)CuO4Fe", "Extra element at end"),
        ("La(7/10)Cu(3/10)SrO4", "Elements out of order"),
        ("Ni(7/10)Sr(3/10)CuO4", "Wrong first element (La->Ni)"),
        ("Fe(1/2)Mn(1/2)O3", "Completely different formula"),
    ]

    # Convert target to tokens
    target_tokens_list = tokenize_formula(target_formula)
    target_indices = tokens_to_indices(target_tokens_list, max_len=40)
    target_tensor = target_indices.unsqueeze(0)  # [1, seq_len]

    print(f"\nTarget: {target_formula}")
    print("-" * 60)

    for sample_formula, description in test_cases:
        sample_tokens_list = tokenize_formula(sample_formula)
        sample_indices = tokens_to_indices(sample_tokens_list, max_len=40)
        sample_tensor = sample_indices.unsqueeze(0)  # [1, seq_len]

        # Compute reward
        reward = compute_semantic_reward(sample_tensor, target_tensor)

        print(f"{description:40s} | Reward: {reward.item():.3f} | {sample_formula}")

    print("-" * 60)
    print("\nReward Hierarchy:")
    print("  1.0       = Exact match")
    print("  0.6-0.8   = All elements correct, fraction/subscript errors")
    print("  0.2-0.5   = Some elements correct")
    print("  0.0-0.2   = Wrong elements or garbage")


if __name__ == '__main__':
    test_reinforce()
    test_hierarchical_reward()
