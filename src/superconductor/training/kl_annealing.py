"""
Cyclical KL Annealing for VAE Training.

Prevents KL vanishing / posterior collapse by cycling the KL weight (β)
from 0 to 1 multiple times during training. This allows the model to:
1. First learn good reconstructions (β≈0, autoencoder mode)
2. Then regularize the latent space (β→1, VAE mode)
3. Repeat to progressively improve latent codes

Based on: "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
https://arxiv.org/abs/1903.10145

Usage:
    scheduler = CyclicalKLScheduler(n_epochs=300, n_cycles=4, warmup_epochs=10)

    for epoch in range(n_epochs):
        beta = scheduler.get_beta(epoch)
        loss = reconstruction_loss + beta * kl_loss
"""

import numpy as np
from typing import Literal
from dataclasses import dataclass


@dataclass
class KLSchedulerConfig:
    """Configuration for KL annealing scheduler."""
    n_epochs: int = 300
    n_cycles: int = 4              # Number of annealing cycles
    warmup_epochs: int = 10        # Pure autoencoder warmup (β=0)
    cycle_ratio: float = 0.5       # Fraction of cycle for ramping (rest is β=1)
    min_beta: float = 0.0          # Minimum β value
    max_beta: float = 1.0          # Maximum β value
    schedule_type: str = 'cyclical'  # 'cyclical', 'monotonic', 'constant'


class CyclicalKLScheduler:
    """
    Cyclical annealing scheduler for VAE KL weight (β).

    Schedule types:
        - 'cyclical': Repeat 0→1 ramp multiple times (recommended)
        - 'monotonic': Single 0→1 ramp then constant
        - 'constant': Fixed β=1 throughout (baseline)

    Args:
        n_epochs: Total training epochs
        n_cycles: Number of annealing cycles (for cyclical)
        warmup_epochs: Initial epochs with β=0 (pure autoencoder)
        cycle_ratio: Fraction of each cycle spent ramping (0.5 = ramp half, plateau half)
        min_beta: Minimum β value during ramp
        max_beta: Maximum β value at plateau
        schedule_type: 'cyclical', 'monotonic', or 'constant'
    """

    def __init__(
        self,
        n_epochs: int = 300,
        n_cycles: int = 4,
        warmup_epochs: int = 10,
        cycle_ratio: float = 0.5,
        min_beta: float = 0.0,
        max_beta: float = 1.0,
        schedule_type: str = 'cyclical'
    ):
        self.n_epochs = n_epochs
        self.n_cycles = n_cycles
        self.warmup_epochs = warmup_epochs
        self.cycle_ratio = cycle_ratio
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.schedule_type = schedule_type

        # Compute cycle length
        self.effective_epochs = n_epochs - warmup_epochs
        self.cycle_length = self.effective_epochs / n_cycles if n_cycles > 0 else self.effective_epochs

    def get_beta(self, epoch: int) -> float:
        """
        Get KL weight (β) for given epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            β value in [min_beta, max_beta]
        """
        if self.schedule_type == 'constant':
            return self.max_beta

        # Warmup phase: pure autoencoder
        if epoch < self.warmup_epochs:
            return self.min_beta

        # Adjust epoch for post-warmup
        adjusted_epoch = epoch - self.warmup_epochs

        if self.schedule_type == 'monotonic':
            # Single ramp over all post-warmup epochs
            ramp_epochs = self.effective_epochs * self.cycle_ratio
            if adjusted_epoch >= ramp_epochs:
                return self.max_beta
            else:
                progress = adjusted_epoch / ramp_epochs
                return self.min_beta + progress * (self.max_beta - self.min_beta)

        elif self.schedule_type == 'cyclical':
            # Cyclical: repeat ramp multiple times
            cycle_position = adjusted_epoch % self.cycle_length
            ramp_length = self.cycle_length * self.cycle_ratio

            if cycle_position >= ramp_length:
                # Plateau phase
                return self.max_beta
            else:
                # Ramp phase
                progress = cycle_position / ramp_length
                return self.min_beta + progress * (self.max_beta - self.min_beta)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def get_cycle_info(self, epoch: int) -> dict:
        """Get detailed info about current position in schedule."""
        beta = self.get_beta(epoch)

        if epoch < self.warmup_epochs:
            phase = 'warmup'
            cycle = 0
            cycle_progress = 0.0
        else:
            adjusted = epoch - self.warmup_epochs
            cycle = int(adjusted // self.cycle_length) + 1
            cycle_progress = (adjusted % self.cycle_length) / self.cycle_length

            ramp_ratio = self.cycle_ratio
            if cycle_progress < ramp_ratio:
                phase = 'ramp'
            else:
                phase = 'plateau'

        return {
            'epoch': epoch,
            'beta': beta,
            'phase': phase,
            'cycle': cycle,
            'cycle_progress': cycle_progress
        }

    def get_full_schedule(self) -> np.ndarray:
        """Get β values for all epochs (for visualization)."""
        return np.array([self.get_beta(e) for e in range(self.n_epochs)])


def compute_kl_loss(mu: 'torch.Tensor', logvar: 'torch.Tensor') -> 'torch.Tensor':
    """
    Compute KL divergence loss for VAE: KL(q(z|x) || p(z)).

    Assumes p(z) = N(0, I) (standard normal prior).

    KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)

    Args:
        mu: Mean of approximate posterior (batch, latent_dim)
        logvar: Log variance of approximate posterior (batch, latent_dim)

    Returns:
        KL divergence, summed over latent dimensions, averaged over batch
    """
    import torch
    # KL divergence per sample
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # Average over batch
    return kl_per_sample.mean()


def compute_kl_loss_with_free_bits(
    mu: 'torch.Tensor',
    logvar: 'torch.Tensor',
    free_bits: float = 0.5
) -> 'torch.Tensor':
    """
    Compute KL loss with free bits (minimum KL per dimension).

    Free bits prevents complete posterior collapse by ensuring each
    latent dimension has at least `free_bits` nats of information.

    Args:
        mu: Mean of approximate posterior
        logvar: Log variance of approximate posterior
        free_bits: Minimum KL per dimension (default 0.5 nats)

    Returns:
        KL divergence with free bits constraint
    """
    import torch
    # KL per dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Apply free bits: max(kl, free_bits) per dimension
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    # Sum over dimensions, average over batch
    return kl_per_dim.sum(dim=1).mean()


class KLLossWithAnnealing:
    """
    Combined KL loss computation with cyclical annealing.

    Usage:
        kl_module = KLLossWithAnnealing(n_epochs=300, n_cycles=4)

        for epoch in range(n_epochs):
            kl_loss = kl_module(mu, logvar, epoch)
    """

    def __init__(
        self,
        n_epochs: int = 300,
        n_cycles: int = 4,
        warmup_epochs: int = 10,
        cycle_ratio: float = 0.5,
        max_beta: float = 1.0,
        use_free_bits: bool = False,
        free_bits: float = 0.5,
        schedule_type: str = 'cyclical'
    ):
        self.scheduler = CyclicalKLScheduler(
            n_epochs=n_epochs,
            n_cycles=n_cycles,
            warmup_epochs=warmup_epochs,
            cycle_ratio=cycle_ratio,
            max_beta=max_beta,
            schedule_type=schedule_type
        )
        self.use_free_bits = use_free_bits
        self.free_bits = free_bits

    def __call__(
        self,
        mu: 'torch.Tensor',
        logvar: 'torch.Tensor',
        epoch: int
    ) -> tuple:
        """
        Compute annealed KL loss.

        Args:
            mu: Mean of approximate posterior
            logvar: Log variance of approximate posterior
            epoch: Current training epoch

        Returns:
            Tuple of (weighted_kl_loss, raw_kl_loss, beta)
        """
        # Compute raw KL
        if self.use_free_bits:
            raw_kl = compute_kl_loss_with_free_bits(mu, logvar, self.free_bits)
        else:
            raw_kl = compute_kl_loss(mu, logvar)

        # Get annealing weight
        beta = self.scheduler.get_beta(epoch)

        # Weighted KL
        weighted_kl = beta * raw_kl

        return weighted_kl, raw_kl, beta

    def get_beta(self, epoch: int) -> float:
        """Get current β value."""
        return self.scheduler.get_beta(epoch)
