#!/usr/bin/env python3
"""
V12.6 Clean Training Script - Simplified but with all optimizations.

Removes complexity to isolate CUDA crash issue while keeping:
- Mixed precision (AMP)
- Multi-worker DataLoader
- Full model architecture
- Proper loss function
- Checkpointing
"""

import os
import sys
import json
import math
import hashlib
import signal
import re
import time
from collections import defaultdict

# --- Ensure C compiler is available for torch.compile (triton/inductor) ---
# This fixes the recurring issue where torch.compile fails with
# "Failed to find C compiler" when the conda env isn't fully activated.
# Works across local conda envs, Google Colab, and standard Linux installs.
import shutil as _shutil
if 'CC' not in os.environ:
    # Priority: conda env gcc → system gcc
    _conda_gcc = os.path.join(sys.prefix, 'bin', 'gcc')
    _found_cc = _conda_gcc if os.path.isfile(_conda_gcc) else _shutil.which('gcc')
    if _found_cc:
        os.environ['CC'] = _found_cc
        _cc_dir = os.path.dirname(_found_cc)
        _path = os.environ.get('PATH', '')
        if _cc_dir not in _path:
            os.environ['PATH'] = _cc_dir + os.pathsep + _path

import numpy as np
import pandas as pd
from scipy.stats import rankdata as _rankdata  # V12.20: quantile transform for skewed Magpie features
from scipy.special import ndtri as _ndtri       # V12.20: inverse normal CDF (rank → Gaussian)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
# PyTorch version compatibility for AMP (Automatic Mixed Precision)
# PyTorch 2.0+ has unified torch.amp with device_type argument
# Older versions use torch.cuda.amp without device_type
import torch.cuda.amp as _cuda_amp
try:
    from torch.amp import GradScaler
    # Test if new API works (device_type argument)
    _test_autocast = torch.amp.autocast(device_type='cuda', enabled=False)
    from torch.amp import autocast
    PYTORCH_NEW_AMP = True
except (ImportError, TypeError):
    from torch.cuda.amp import autocast as _old_autocast, GradScaler
    PYTORCH_NEW_AMP = False
    # Wrapper to match new API signature
    def autocast(device_type='cuda', dtype=None, enabled=True):
        # Old API doesn't support dtype directly, use enabled only
        return _old_autocast(enabled=enabled)
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superconductor.utils.env_config import detect_environment
from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, tokenize_formula, tokens_to_indices,
    VOCAB_SIZE, PAD_IDX, START_IDX, END_IDX, IDX_TO_TOKEN, indices_to_formula
)
from superconductor.encoders.element_properties import get_atomic_number

# V12.8: REINFORCE support with GPU-native rewards
from superconductor.losses.reward_gpu_native import (
    compute_reward_gpu_native, GPURewardConfig, get_default_gpu_reward_config
)

# V12.9: Entropy maintenance for REINFORCE (prevents entropy collapse)
from superconductor.training.entropy_maintenance import (
    EntropyManager, create_entropy_manager, EntropyConfig
)

# V12.12: Contrastive learning for SC vs non-SC separation
from superconductor.losses.contrastive import (
    SuperconductorContrastiveLoss, category_to_label
)
from torch.utils.data import WeightedRandomSampler

# V12.9: N-gram + Structural hybrid draft model for speculative decoding
from superconductor.models.ngram_draft import (
    HybridDraft, load_or_build_draft_model
)

# V12.16: Theory-guided consistency feedback
from superconductor.losses.consistency_losses import (
    ConsistencyLossConfig, CombinedConsistencyLoss, compute_consistency_reward
)
from superconductor.losses.theory_losses import (
    TheoryLossConfig, TheoryRegularizationLoss
)
from superconductor.models.family_classifier import (
    SuperconductorFamily, RuleBasedFamilyClassifier, HybridFamilyClassifier
)


# ============================================================================
# V12.15: TIMING INSTRUMENTATION
# ============================================================================

class TimingStats:
    """
    Track timing for different phases of training.

    Phases tracked:
    - data_load: Time to fetch batch from DataLoader
    - encoder_fwd: Encoder forward pass
    - decoder_fwd: Decoder forward pass (teacher forcing)
    - loss_compute: Loss computation (excluding REINFORCE)
    - reinforce_sample: REINFORCE autoregressive sampling (the expensive part)
    - backward: Backward pass (gradient computation)
    - optimizer: Optimizer step (including gradient clipping)
    - other: Everything else (metrics accumulation, etc.)
    """

    PHASES = [
        'data_load',
        'encoder_fwd',
        'decoder_fwd',
        'loss_compute',
        'reinforce_sample',
        'backward',
        'optimizer',
        'other'
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all timing accumulators."""
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self._phase_start = None
        self._current_phase = None
        self._epoch_start = None

    def start_epoch(self):
        """Mark the start of an epoch."""
        self._epoch_start = time.perf_counter()

    def start(self, phase: str):
        """Start timing a phase."""
        # Ensure CUDA operations are complete before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._phase_start = time.perf_counter()
        self._current_phase = phase

    def stop(self, phase: str = None):
        """Stop timing the current phase."""
        if self._phase_start is None:
            return
        # Ensure CUDA operations are complete before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._phase_start
        phase = phase or self._current_phase
        if phase:
            self.times[phase] += elapsed
            self.counts[phase] += 1
        self._phase_start = None
        self._current_phase = None

    def get_epoch_time(self) -> float:
        """Get total epoch time in seconds."""
        if self._epoch_start is None:
            return 0.0
        return time.perf_counter() - self._epoch_start

    def get_total_tracked(self) -> float:
        """Get sum of all tracked phase times."""
        return sum(self.times.values())

    def get_breakdown(self) -> Dict[str, float]:
        """
        Get timing breakdown as percentages.

        Returns:
            Dict mapping phase name to percentage of total time.
        """
        total = self.get_total_tracked()
        if total == 0:
            return {phase: 0.0 for phase in self.PHASES}
        return {phase: (self.times[phase] / total) * 100 for phase in self.PHASES}

    def get_absolute(self) -> Dict[str, float]:
        """Get absolute times in seconds for each phase."""
        return dict(self.times)

    def get_per_batch(self) -> Dict[str, float]:
        """Get average time per batch for each phase (in ms)."""
        result = {}
        for phase in self.PHASES:
            if self.counts[phase] > 0:
                result[phase] = (self.times[phase] / self.counts[phase]) * 1000  # ms
            else:
                result[phase] = 0.0
        return result

    def format_summary(self, epoch_time: float = None) -> str:
        """
        Format timing summary for display.

        Args:
            epoch_time: Optional total epoch time (if not using start_epoch)

        Returns:
            Formatted string like "Fwd 28% | Sample 45% | Bwd 22% | Data 5%"
        """
        breakdown = self.get_breakdown()
        epoch_time = epoch_time or self.get_epoch_time()

        # Combine encoder + decoder forward
        fwd_pct = breakdown['encoder_fwd'] + breakdown['decoder_fwd']

        parts = []

        # Main phases (sorted by typical importance)
        if breakdown['reinforce_sample'] > 1:
            parts.append(f"Sample {breakdown['reinforce_sample']:.0f}%")
        if fwd_pct > 1:
            parts.append(f"Fwd {fwd_pct:.0f}%")
        if breakdown['backward'] > 1:
            parts.append(f"Bwd {breakdown['backward']:.0f}%")
        if breakdown['loss_compute'] > 1:
            parts.append(f"Loss {breakdown['loss_compute']:.0f}%")
        if breakdown['optimizer'] > 1:
            parts.append(f"Opt {breakdown['optimizer']:.0f}%")
        if breakdown['data_load'] > 1:
            parts.append(f"Data {breakdown['data_load']:.0f}%")

        # Calculate untracked time
        tracked = self.get_total_tracked()
        if epoch_time > 0:
            untracked_pct = max(0, (epoch_time - tracked) / epoch_time * 100)
            if untracked_pct > 5:
                parts.append(f"Other {untracked_pct:.0f}%")

        return " | ".join(parts) if parts else "No timing data"

    def format_detailed(self) -> str:
        """
        Format detailed timing report.

        Returns:
            Multi-line string with detailed breakdown.
        """
        breakdown = self.get_breakdown()
        absolute = self.get_absolute()
        per_batch = self.get_per_batch()
        epoch_time = self.get_epoch_time()

        lines = [
            "=" * 60,
            "TIMING BREAKDOWN",
            "=" * 60,
            f"Total epoch time: {epoch_time:.1f}s",
            f"Tracked time: {self.get_total_tracked():.1f}s ({self.get_total_tracked()/epoch_time*100:.1f}%)" if epoch_time > 0 else "",
            "",
            f"{'Phase':<20} {'Time (s)':<12} {'%':<8} {'ms/batch':<12}",
            "-" * 60,
        ]

        for phase in self.PHASES:
            if absolute[phase] > 0 or self.counts[phase] > 0:
                lines.append(
                    f"{phase:<20} {absolute[phase]:<12.2f} {breakdown[phase]:<8.1f} {per_batch[phase]:<12.2f}"
                )

        lines.append("-" * 60)
        lines.append("")

        return "\n".join(lines)


# Global timing stats instance (set per-epoch in train_epoch)
_timing_stats: Optional[TimingStats] = None


def get_timing_stats() -> Optional[TimingStats]:
    """Get the current timing stats instance."""
    global _timing_stats
    return _timing_stats


# ============================================================================
# CONFIG
# ============================================================================

MODEL_CONFIG = {
    'latent_dim': 2048,
    'fusion_dim': 256,
    'magpie_dim': 145,
    'encoder_hidden': [512, 256],
    'decoder_hidden': [256, 512],
    'd_model': 512,
    'nhead': 8,
    'num_layers': 12,
    'dim_feedforward': 2048,
    'n_memory_tokens': 16,
    'element_embed_dim': 128,
}

TRAIN_CONFIG = {
    'num_epochs': 4000,
    'learning_rate': 3e-5,      # Reduced for stable fine-tuning (was 1e-4)
    'max_formula_len': 60,
    'checkpoint_interval': 50,

    # =========================================================================
    # BATCH SIZE CONFIGURATION
    # =========================================================================
    # Set to 'auto' to automatically scale based on GPU memory:
    #   - 8GB  (RTX 4060):      batch_size=48
    #   - 11GB (GTX 1080 Ti):   batch_size=48
    #   - 16GB (V100 16GB):     batch_size=64
    #   - 24GB (RTX 3090/4090): batch_size=96
    #   - 40GB+ (A100):         batch_size=128
    #
    # Effective batch size = batch_size * accumulation_steps
    # Larger effective batch = smoother gradients, but may need LR scaling
    # =========================================================================
    'batch_size': 42,            # Full GPU budget (no other GPU apps running)
    'accumulation_steps': 2,    # Gradient accumulation (increase for larger effective batch) V12.12: effective batch=64

    # V12.8: Data Loading Optimizations (defaults — overridden by detect_environment())
    'num_workers': 2,
    'pin_memory': False,
    'prefetch_factor': 1,
    'persistent_workers': False,

    # V12.8: Mixed Precision Optimizations
    'use_amp': True,
    'amp_dtype': 'auto',        # 'auto' detects GPU capability, or force 'bfloat16'/'float16'
    'matmul_precision': 'medium',  # 'highest', 'high', 'medium' for TF32

    # V12.8: Learning Rate Schedule
    'lr_scheduler': 'cosine',  # Plain cosine decay - no restarts for stable fine-tuning
    'lr_restart_period': 100,   # T_0 for warm restarts
    'lr_restart_mult': 2,       # T_mult - double period after each restart
    'lr_min_factor': 0.01,      # eta_min = lr * this factor

    # V12.8: Compute Optimizations
    'use_torch_compile': True,   # V12.10: Works with resume (gcc installed in conda env)
    'compile_mode': 'reduce-overhead',  # 'default', 'reduce-overhead', 'max-autotune'
    'use_gradient_checkpointing': False,  # Incompatible with torch.compile (tensor count mismatch during recomputation)
    'enable_flash_sdp': True,   # Enable Flash Attention via SDPA

    # Loss weights (final values - curriculum ramps up to these)
    'formula_weight': 1.0,
    'tc_weight': 10.0,
    'magpie_weight': 2.0,
    'stoich_weight': 2.0,  # V12.4: Stoichiometry loss weight
    'kl_weight': 0.0001,  # Now L2 regularization weight on z (deterministic encoder)
    'hp_loss_weight': 0.05,  # V12.19: High-pressure prediction loss weight (BCE)
    'sc_loss_weight': 0.5,   # V12.21: SC/non-SC classification loss weight (BCE, all samples)

    # V12.20: Tc loss improvements (log-transform + Huber)
    'tc_log_transform': True,   # Apply log1p(Tc) before z-score normalization (reduces skew 2.18→-0.17)
    'tc_huber_delta': 1.0,      # Huber loss delta (clips gradient for outliers; 1.0 = ~1 std dev in normalized space)

    # V12.20: Magpie normalization improvements
    'magpie_skew_threshold': 3.0,   # Features with |skew| > this get quantile-transformed to Gaussian
    'magpie_sc_only_norm': True,    # Use SC-only mean/std for z-score (avoids non-SC distribution bias)

    # Teacher forcing decay
    'tf_start': 1.0,           # Start with full teacher forcing
    'tf_end': 0.0,             # Decay to full autoregressive
    'tf_decay_epochs': 100,    # Decay over 100 epochs

    # Curriculum phases
    'curriculum_phase1_end': 30,  # Ramp up Tc/Magpie weights

    # Focal loss and label smoothing (to break accuracy plateau)
    'focal_gamma': 2.0,          # Focal loss focusing parameter (0=standard CE, 2=typical)
    'label_smoothing': 0.1,      # Label smoothing factor (0.0=none, 0.1=typical)

    # =========================================================================
    # V12.8: REINFORCE Settings
    # =========================================================================
    # REINFORCE uses RLOO (Leave-One-Out) baseline for variance reduction.
    # With KV caching, autoregressive sampling is ~60x faster than naive.
    #
    # Recommended settings:
    #   - Start with rl_weight=0 until model converges on CE loss
    #   - Then enable rl_weight=1.0-2.5 for fine-tuning
    #   - Higher n_samples_rloo = lower variance but slower
    # =========================================================================
    'rl_weight': 1.5,            # REINFORCE weight (0=disabled, 1.0-2.5=typical) V12.12: Enabled for fine-tuning
    'ce_weight': 1.0,            # Cross-entropy weight (keep at 1.0)
    'n_samples_rloo': 2,         # Number of samples for RLOO baseline (2-4)
    'rl_temperature': 0.8,       # Sampling temperature for REINFORCE
    'entropy_weight': 0.2,       # Entropy bonus in REINFORCE reward (encourages exploration)
    'use_autoregressive_reinforce': True,  # Use KV-cached autoregressive sampling (recommended)

    # =========================================================================
    # V12.9: Speculative Decoding Settings
    # =========================================================================
    # Uses n-gram + structural draft model to predict tokens ahead, then verifies
    # in batch. Speeds up autoregressive sampling by ~1.8-2.2x.
    #
    # The draft model is built from training data and cached to disk.
    # Enable for REINFORCE training to reduce sampling overhead.
    # V12.16: Fixed per-sequence tracking - now works with large batches.
    # V12.16.1: DISABLED - N-gram draft model doesn't match VAE decoder well.
    #   The draft model predicts based on corpus statistics, but the VAE decoder
    #   predicts based on latent z. Acceptance rate is only 1-4%, making spec
    #   decoding 5x SLOWER than standard sampling. For spec decoding to work,
    #   we'd need a z-conditioned draft model (mini-decoder).
    # =========================================================================
    'use_speculative_decoding': False,      # V12.16.1: Disabled - doesn't match VAE architecture
    'speculative_k': 5,                     # Number of tokens to draft at once
    'draft_model_path': 'data/processed/draft_model_v2.pkl',  # Path to V2 draft model cache (position-aware)

    # =========================================================================
    # V12.17: Latent Z Caching & Prediction Logging
    # =========================================================================
    # Cache latent z vectors and decoder predictions for analysis.
    # Useful for:
    #   - Analyzing z-space structure (correlations with Tc, family, etc.)
    #   - Training z-conditioned draft model for speculative decoding
    #   - Debugging encoder/decoder behavior
    #   - Post-training analysis of all predictions
    #
    # Cache modes:
    #   - 'z_only': Just z vectors (fast, ~50MB per cache)
    #   - 'z_and_predictions': Z + decoder predictions (slower, ~200MB per cache)
    #   - 'full': Z + predictions + token log_probs (slow, ~500MB per cache)
    # =========================================================================
    'cache_z_vectors': True,                # Enable z-vector caching
    'z_cache_interval': 50,                 # Cache every N epochs (0 = only on best checkpoint)
    'z_cache_path': 'outputs/latent_cache.pt',  # Path to save z cache
    'z_cache_mode': 'z_and_predictions',    # 'z_only', 'z_and_predictions', or 'full'
    'z_cache_every_epoch': True,            # Cache EVERY epoch for full error analysis over time

    # =========================================================================
    # V12.15: Timing Instrumentation
    # =========================================================================
    # Track compute time breakdown to identify bottlenecks.
    # Phases tracked: data_load, encoder_fwd, decoder_fwd, loss_compute,
    #                 reinforce_sample, backward, optimizer
    # =========================================================================
    'enable_timing': True,                  # Enable timing instrumentation

    # =========================================================================
    # V12.9: Entropy Maintenance Settings
    # =========================================================================
    # Prevents entropy collapse during REINFORCE training by dynamically
    # adjusting entropy weight and temperature based on training progress.
    #
    # Strategies:
    #   - 'constant': Fixed entropy weight (baseline)
    #   - 'adaptive': AER-style dynamic coefficient
    #   - 'causal': Diagnoses plateau cause before boosting (recommended)
    #               Only boosts if entropy dropped before plateau AND/OR entropy is low
    #               Tracks intervention success and adjusts confidence over time
    #   - 'cyclical': Temperature warm restarts
    #   - 'composite': Combines adaptive + cyclical + uncertainty
    # =========================================================================
    'entropy_strategy': 'causal',        # Entropy maintenance strategy (causal = smarter diagnosis)
    'entropy_target': 0.5,               # Target entropy to maintain (nats)
    'entropy_min': 0.1,                  # Critical threshold - strong boost below this
    'entropy_weight_min': 0.05,          # Minimum entropy weight
    'entropy_weight_max': 1.0,           # Maximum entropy weight
    'entropy_plateau_window': 10,        # Epochs to detect training plateau
    'entropy_plateau_threshold': 0.01,   # Improvement threshold (1% relative by default)
    'entropy_plateau_relative': True,    # If True, threshold scales with performance

    # Resume from checkpoint (set to None to train from scratch)
    'resume_checkpoint': 'outputs/checkpoint_best.pt',  # V12.10: Resume from best checkpoint

    # V12.12: Retrain on new/combined data - resets catastrophic drop detector
    # Set to True when training data has changed (new normalization stats)
    'retrain_new_data': True,

    # =========================================================================
    # V12.12: Contrastive Learning Settings
    # =========================================================================
    # Trains on 46K mixed SC + non-SC data with contrastive loss to:
    # 1. Improve decoder on rare elements (more formula examples)
    # 2. Separate SC vs non-SC in latent space
    # 3. Cluster SC families together
    # =========================================================================
    'contrastive_mode': True,          # Enable contrastive training with non-SC data
    'contrastive_weight': 0.1,         # Weight for contrastive loss
    'contrastive_warmup_epochs': 100,  # Warm up contrastive loss over N epochs
    'contrastive_temperature': 0.07,   # SupCon temperature (0.07 = standard)
    'non_sc_formula_weight': 0.5,      # Formula loss weight for non-SC samples (lower)
    'use_extended_labels': True,       # Use per-family labels (vs binary SC/non-SC)
    'balanced_sampling': True,         # Balanced sampling: ~50/50 SC/non-SC per batch

    # V12.12: Selective Backpropagation
    # Skip backward pass on batches where loss is well below running average.
    # Saves compute on "easy" batches the model has already learned.
    'selective_backprop': True,
    'selective_backprop_threshold': 0.33,  # Skip if batch loss < threshold * running_avg_loss

    # =========================================================================
    # V12.16: Theory-Guided Consistency Feedback
    # =========================================================================
    # Applies physics-based constraints during training:
    # - Consistency loss: Ensures encoder-decoder property coherence
    # - Theory loss: BCS, cuprate, iron-based constraints by family
    # - Unknown materials get NO theory constraints (intentional)
    # =========================================================================
    'use_consistency_loss': False,         # Enable consistency feedback (disabled by default until tested)
    'consistency_weight': 0.1,             # Weight for consistency loss
    'consistency_tc_weight': 1.0,          # Weight for Tc consistency
    'consistency_magpie_weight': 0.1,      # Weight for Magpie consistency

    'use_theory_loss': True,               # V12.22: Enable theory-based regularization
    'theory_weight': 0.5,                  # V12.22: Strong signal (was 0.05)
    'theory_warmup_epochs': 50,            # V12.22: Ramp up over 50 epochs
    'theory_use_soft_constraints': True,   # V12.22: Soft quadratic penalties (no hard caps)

    'use_family_classifier': False,        # Train learned family classifier alongside
    'family_classifier_weight': 0.01,      # Weight for family classification loss
}

CONTRASTIVE_DATA_PATH = PROJECT_ROOT / 'data/processed/supercon_fractions_contrastive.csv'

DATA_PATH = PROJECT_ROOT / 'data/processed/supercon_fractions_combined.csv'
HOLDOUT_PATH = PROJECT_ROOT / 'data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# ============================================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ============================================================================

class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss with optional Label Smoothing for sequence generation.

    Focal Loss: Focuses training on hard examples by down-weighting easy ones.
        FL(p) = -alpha * (1-p)^gamma * log(p)

    Label Smoothing: Prevents overconfidence by softening targets.
        soft_target = (1 - smoothing) * one_hot + smoothing / num_classes

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is standard CE, gamma=2 is typical.
        alpha: Class weight (not used here, kept for API compatibility)
        smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 typical)
        ignore_index: Index to ignore in loss computation (e.g., PAD_IDX)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0,
                 smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch*seq_len, vocab_size] or [batch, seq_len, vocab_size]
            targets: [batch*seq_len] or [batch, seq_len]

        Returns:
            Scalar loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

        num_classes = logits.size(-1)

        # Create mask for valid tokens (not padding)
        valid_mask = (targets != self.ignore_index)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Get valid logits and targets
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        # Compute log probabilities
        log_probs = F.log_softmax(valid_logits, dim=-1)
        probs = torch.exp(log_probs)

        # Get probability of correct class
        target_log_probs = log_probs.gather(dim=-1, index=valid_targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=valid_targets.unsqueeze(-1)).squeeze(-1)

        # Focal weight: (1 - p)^gamma
        focal_weight = (1.0 - target_probs) ** self.gamma

        # Label smoothing: mix target log prob with uniform distribution
        if self.smoothing > 0:
            # Smoothed loss = (1-s)*CE + s*uniform_CE
            # uniform_CE = -log(1/num_classes) = log(num_classes)
            smooth_loss = -log_probs.mean(dim=-1)  # Mean over all classes
            focal_loss = focal_weight * (
                (1.0 - self.smoothing) * (-target_log_probs) +
                self.smoothing * smooth_loss
            )
        else:
            focal_loss = focal_weight * (-target_log_probs)

        return focal_loss.mean()


# ============================================================================
# CURRICULUM & TEACHER FORCING
# ============================================================================

def get_curriculum_weights(epoch: int) -> Tuple[float, float]:
    """
    Curriculum learning for loss weights.

    Phase 1 (epochs 0-30): Ramp up from moderate to full
        - tc_weight: 5.0 → 10.0
        - magpie_weight: 1.0 → 2.0

    Phase 2 (epochs 30+): Full strength
        - tc_weight: 10.0
        - magpie_weight: 2.0
    """
    phase1_end = TRAIN_CONFIG['curriculum_phase1_end']

    if epoch < phase1_end:
        progress = epoch / phase1_end
        tc_weight = 5.0 + 5.0 * progress       # 5.0 → 10.0
        magpie_weight = 1.0 + 1.0 * progress   # 1.0 → 2.0
    else:
        tc_weight = TRAIN_CONFIG['tc_weight']        # 10.0
        magpie_weight = TRAIN_CONFIG['magpie_weight']  # 2.0

    return tc_weight, magpie_weight


def get_teacher_forcing_ratio(exact_match: float) -> float:
    """
    Adaptive teacher forcing based on model competence (exact match %).

    V12.6: Now uses optimized 2-pass scheduled sampling instead of 60-pass sequential.
    The decoder's forward method was updated to use a parallel approach that's
    only 2x slower than full teacher forcing instead of 60x.

    Formula: TF = 1.0 - exact_match (no floor)
    - TF decreases linearly as exact_match increases
    - At 100% exact match, TF = 0 (pure autoregressive)

    Examples:
        - exact_match = 0%   → TF = 1.0 (full teacher forcing)
        - exact_match = 50%  → TF = 0.5 (half teacher forcing)
        - exact_match = 80%  → TF = 0.2 (mostly autoregressive)
        - exact_match = 100% → TF = 0.0 (pure autoregressive)
    """
    # Linear decay with no floor - allows pure autoregressive at high competence
    tf_ratio = 1.0 - exact_match
    return max(0.0, tf_ratio)  # Clamp at 0 (exact_match can't exceed 1.0 anyway)


# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

_shutdown_state = {
    'should_stop': False,
    'encoder': None, 'decoder': None,
    'epoch': 0,
    'entropy_manager': None,
    # V12.10: Additional state for proper checkpoint saving
    'enc_opt': None, 'dec_opt': None,
    'enc_scheduler': None, 'dec_scheduler': None,
    'prev_exact': 0, 'best_exact': 0,
    'theory_loss_fn': None,  # V12.22
}

def signal_handler(signum, frame):
    print(f"\n*** Interrupt received - saving and exiting... ***")
    _shutdown_state['should_stop'] = True
    if _shutdown_state['encoder'] is not None:
        save_checkpoint(
            _shutdown_state['encoder'], _shutdown_state['decoder'],
            _shutdown_state['epoch'], 'interrupt',
            entropy_manager=_shutdown_state.get('entropy_manager'),
            enc_opt=_shutdown_state.get('enc_opt'),
            dec_opt=_shutdown_state.get('dec_opt'),
            enc_scheduler=_shutdown_state.get('enc_scheduler'),
            dec_scheduler=_shutdown_state.get('dec_scheduler'),
            prev_exact=_shutdown_state.get('prev_exact', 0),
            best_exact=_shutdown_state.get('best_exact', 0),
            theory_loss_fn=_shutdown_state.get('theory_loss_fn'),  # V12.22
        )
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# DATA LOADING
# ============================================================================

def parse_fraction_formula(formula: str) -> Optional[Dict[str, float]]:
    """Parse formula like 'Ag(1/500)Al(499/500)' to element fractions."""
    pattern = r'([A-Z][a-z]?)(?:\((\d+)/(\d+)\)|(\d*\.?\d+))?'
    matches = re.findall(pattern, formula)

    result = {}
    for match in matches:
        element = match[0]
        if not element:
            continue
        if match[1] and match[2]:
            result[element] = float(match[1]) / float(match[2])
        elif match[3]:
            result[element] = float(match[3])
        else:
            result[element] = 1.0
    return result if result else None


def load_holdout_indices(holdout_path: Path, formulas: list) -> set:
    """Load holdout sample indices by matching formulas (robust to row reordering).

    Args:
        holdout_path: Path to GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json
        formulas: List of formula strings from the loaded CSV

    Returns:
        Set of integer indices into the formulas list that are holdout samples.
    """
    with open(holdout_path, 'r') as f:
        data = json.load(f)
    holdout_formulas = {s['formula'] for s in data['holdout_samples']}
    indices = {i for i, f in enumerate(formulas) if f in holdout_formulas}
    if len(indices) != len(holdout_formulas):
        print(f"  WARNING: Found {len(indices)}/{len(holdout_formulas)} holdout samples in data")
    return indices


def _get_cache_dir():
    """Get the data preprocessing cache directory."""
    return PROJECT_ROOT / 'data' / 'processed' / 'cache'


def _compute_csv_hash(csv_path: Path) -> str:
    """Compute a fast hash of the CSV file for cache invalidation.

    Uses file size + mtime as a proxy (much faster than hashing 29MB).
    Falls back to MD5 of first+last 8KB if mtime is unreliable (e.g., Drive).
    """
    stat = csv_path.stat()
    # size + mtime is fast and reliable for local files
    quick_hash = f"{stat.st_size}_{stat.st_mtime_ns}"
    return hashlib.md5(quick_hash.encode()).hexdigest()[:16]


def _try_load_cache(csv_hash: str, max_formula_len: int):
    """Try to load preprocessed tensors from cache.

    Returns (tensors_dict, norm_stats, n_magpie_cols, train_indices) or None if cache miss.
    """
    cache_dir = _get_cache_dir()
    meta_path = cache_dir / 'cache_meta.json'

    if not meta_path.exists():
        return None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Validate cache matches current data and config
        if meta.get('csv_hash') != csv_hash:
            print("  Cache stale: CSV changed")
            return None
        if meta.get('max_formula_len') != max_formula_len:
            print("  Cache stale: max_formula_len changed")
            return None
        # V12.20: Invalidate cache if tc_log_transform setting changed
        tc_log_transform = TRAIN_CONFIG.get('tc_log_transform', False)
        if meta.get('tc_log_transform', False) != tc_log_transform:
            print(f"  Cache stale: tc_log_transform changed ({meta.get('tc_log_transform', False)} → {tc_log_transform})")
            return None
        # V12.20: Invalidate cache if Magpie normalization settings changed
        magpie_skew_threshold = TRAIN_CONFIG.get('magpie_skew_threshold', 0.0)
        if meta.get('magpie_skew_threshold', 0.0) != magpie_skew_threshold:
            print(f"  Cache stale: magpie_skew_threshold changed ({meta.get('magpie_skew_threshold', 0.0)} → {magpie_skew_threshold})")
            return None
        magpie_sc_only = TRAIN_CONFIG.get('magpie_sc_only_norm', False)
        if meta.get('magpie_sc_only_norm', False) != magpie_sc_only:
            print(f"  Cache stale: magpie_sc_only_norm changed ({meta.get('magpie_sc_only_norm', False)} → {magpie_sc_only})")
            return None

        # Load tensors
        tensors = {
            'formula_tokens': torch.load(cache_dir / 'formula_tokens.pt', weights_only=True),
            'element_indices': torch.load(cache_dir / 'element_indices.pt', weights_only=True),
            'element_fractions': torch.load(cache_dir / 'element_fractions.pt', weights_only=True),
            'element_mask': torch.load(cache_dir / 'element_mask.pt', weights_only=True),
            'tc_tensor': torch.load(cache_dir / 'tc_tensor.pt', weights_only=True),
            'magpie_tensor': torch.load(cache_dir / 'magpie_tensor.pt', weights_only=True),
        }
        # V12.19: Load HP tensor if available
        hp_cache_path = cache_dir / 'hp_tensor.pt'
        if hp_cache_path.exists():
            tensors['hp_tensor'] = torch.load(hp_cache_path, weights_only=True)

        # V12.22: Load family tensor if available
        family_cache_path = cache_dir / 'family_tensor.pt'
        if family_cache_path.exists():
            tensors['family_tensor'] = torch.load(family_cache_path, weights_only=True)

        norm_stats = {
            'tc_mean': meta['tc_mean'],
            'tc_std': meta['tc_std'],
            'tc_log_transform': meta.get('tc_log_transform', False),  # V12.20
            'magpie_mean': meta['magpie_mean'],
            'magpie_std': meta['magpie_std'],
            # V12.20: Magpie transform metadata
            'magpie_skewed_indices': meta.get('magpie_skewed_indices', []),
            'magpie_sc_only_norm': meta.get('magpie_sc_only_norm', False),
        }

        train_indices = meta['train_indices']
        n_magpie_cols = meta['n_magpie_cols']

        print(f"  Loaded from cache ({len(train_indices)} train samples, {n_magpie_cols} Magpie features)")
        return tensors, norm_stats, n_magpie_cols, train_indices

    except Exception as e:
        print(f"  Cache load failed: {e}")
        return None


def _save_cache(tensors: dict, norm_stats: dict, n_magpie_cols: int,
                train_indices: list, csv_hash: str, max_formula_len: int):
    """Save preprocessed tensors to cache for fast reload."""
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save tensors
    for name, tensor in tensors.items():
        torch.save(tensor, cache_dir / f'{name}.pt')

    # Save metadata
    meta = {
        'csv_hash': csv_hash,
        'max_formula_len': max_formula_len,
        'n_magpie_cols': n_magpie_cols,
        'train_indices': train_indices,
        'tc_mean': norm_stats['tc_mean'],
        'tc_std': norm_stats['tc_std'],
        'tc_log_transform': norm_stats.get('tc_log_transform', False),  # V12.20
        'magpie_mean': norm_stats['magpie_mean'],
        'magpie_std': norm_stats['magpie_std'],
        # V12.20: Magpie transform config for cache invalidation
        'magpie_skew_threshold': TRAIN_CONFIG.get('magpie_skew_threshold', 0.0),
        'magpie_sc_only_norm': norm_stats.get('magpie_sc_only_norm', False),
        'magpie_skewed_indices': norm_stats.get('magpie_skewed_indices', []),
    }
    with open(cache_dir / 'cache_meta.json', 'w') as f:
        json.dump(meta, f)

    print(f"  Saved preprocessing cache to {cache_dir}")


def load_and_prepare_data():
    """Load and prepare full materials dataset.

    V12.12: Supports contrastive mode with mixed SC/non-SC data.
    In contrastive mode, loads 46K samples and adds is_superconductor + category labels.
    Uses balanced sampling to ensure ~50/50 SC/non-SC per batch.

    Uses a tensor cache to skip tokenization/normalization on subsequent runs.
    Cache is invalidated when the CSV file changes or max_formula_len changes.
    """
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)

    max_len = TRAIN_CONFIG['max_formula_len']
    contrastive = TRAIN_CONFIG.get('contrastive_mode', False)

    # V12.12: Select data path based on mode
    data_path = CONTRASTIVE_DATA_PATH if contrastive else DATA_PATH
    if contrastive:
        print(f"CONTRASTIVE MODE: Loading SC + non-SC data from {data_path.name}")
    else:
        print(f"SC-only mode: Loading from {data_path.name}")

    # Check for valid cache (contrastive mode invalidates SC-only cache and vice versa)
    csv_hash = _compute_csv_hash(data_path)
    # Add mode to hash so contrastive and non-contrastive have separate caches
    mode_suffix = '_contrastive' if contrastive else '_sc_only'
    cached = _try_load_cache(csv_hash + mode_suffix, max_len)

    is_sc_tensor = None  # Will be set below
    label_tensor = None  # Will be set below
    n_sc_train = 0
    n_non_sc_train = 0

    if cached is not None:
        tensors, norm_stats, n_magpie_cols, train_indices = cached

        # Rebuild dataset from cached tensors
        tensor_list = [
            tensors['element_indices'], tensors['element_fractions'],
            tensors['element_mask'], tensors['formula_tokens'],
            tensors['tc_tensor'], tensors['magpie_tensor'],
        ]
        if 'is_sc_tensor' in tensors and 'label_tensor' in tensors:
            is_sc_tensor = tensors['is_sc_tensor']
            label_tensor = tensors['label_tensor']
            tensor_list.extend([is_sc_tensor, label_tensor])
        else:
            # Legacy cache without contrastive tensors — add dummy tensors (all SC)
            n = tensors['element_indices'].size(0)
            is_sc_tensor = torch.ones(n, dtype=torch.long)
            label_tensor = torch.zeros(n, dtype=torch.long)
            tensor_list.extend([is_sc_tensor, label_tensor])

        # V12.19: HP tensor (backward compatible — default 0 if missing)
        if 'hp_tensor' in tensors:
            tensor_list.append(tensors['hp_tensor'])
        else:
            n = tensors['element_indices'].size(0)
            tensor_list.append(torch.zeros(n, dtype=torch.float32))

        # V12.22: Family tensor (backward compatible — default zeros triggers recomputation)
        if 'family_tensor' in tensors:
            tensor_list.append(tensors['family_tensor'])
        else:
            # Not in cache — will be recomputed below after dataset creation
            n = tensors['element_indices'].size(0)
            tensor_list.append(torch.zeros(n, dtype=torch.long))

        dataset = TensorDataset(*tensor_list)
        train_dataset = Subset(dataset, train_indices)
        is_sc_train = is_sc_tensor[train_indices]
        n_sc_train = int(is_sc_train.sum().item())
        n_non_sc_train = len(train_indices) - n_sc_train
        print(f"Training samples: {len(train_indices)} (from cache)")

    else:
        # Full preprocessing path
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples")

        formulas = df['formula'].tolist()
        tc_values = df['Tc'].values

        # V12.12: Extract SC labels and category labels
        if contrastive and 'is_superconductor' in df.columns:
            is_sc_values = df['is_superconductor'].values.astype(np.int64)
            print(f"  SC: {is_sc_values.sum()}, Non-SC: {(1 - is_sc_values).sum()}")
        else:
            is_sc_values = np.ones(len(df), dtype=np.int64)

        # V12.19: Extract high-pressure labels (default 0 for backward compat)
        if 'requires_high_pressure' in df.columns:
            hp_values = df['requires_high_pressure'].values.astype(np.float32)
            n_hp = int(hp_values.sum())
            print(f"  High-pressure SC: {n_hp} ({n_hp/max(is_sc_values.sum(),1)*100:.1f}% of SC)")
        else:
            hp_values = np.zeros(len(df), dtype=np.float32)
            print("  No requires_high_pressure column — defaulting to 0")

        # V12.12: Contrastive labels from category column
        # V12.19: Pass HP flag to assign class 12 for non-hydride HP-SC
        use_extended = TRAIN_CONFIG.get('use_extended_labels', True)
        if 'category' in df.columns:
            label_values = np.array([
                category_to_label(cat, use_extended=use_extended,
                                  requires_high_pressure=int(hp))
                for cat, hp in zip(df['category'].values, hp_values)
            ], dtype=np.int64)
            n_labels = len(set(label_values))
            print(f"  Contrastive labels: {n_labels} classes (extended={use_extended})")
        else:
            label_values = np.zeros(len(df), dtype=np.int64)

        # Normalize Tc (use SC samples only for Tc stats to avoid non-SC Tc=0 skewing)
        # V12.20: Optional log-transform before z-score (reduces skewness from 2.18 to -0.17)
        tc_log_transform = TRAIN_CONFIG.get('tc_log_transform', False)
        sc_mask_np = is_sc_values == 1
        if tc_log_transform:
            tc_for_norm = np.log1p(tc_values)  # log(1 + Tc) — handles Tc=0 safely
            print(f"Tc log-transform: log1p applied (raw range [{tc_values.min():.1f}, {tc_values.max():.1f}]K → [{tc_for_norm.min():.2f}, {tc_for_norm.max():.2f}])")
        else:
            tc_for_norm = tc_values
        tc_mean = float(tc_for_norm[sc_mask_np].mean()) if sc_mask_np.any() else float(tc_for_norm.mean())
        tc_std = float(tc_for_norm[sc_mask_np].std()) if sc_mask_np.any() else float(tc_for_norm.std())
        tc_normalized = (tc_for_norm - tc_mean) / tc_std
        print(f"Tc (SC-only stats): mean={tc_mean:.2f}, std={tc_std:.2f}" + (" [log1p space]" if tc_log_transform else " [K]"))

        # Get Magpie features
        exclude = ['formula', 'Tc', 'composition', 'category', 'is_superconductor', 'compound possible', 'formula_original', 'requires_high_pressure']
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        magpie_cols = [c for c in numeric_cols if c not in exclude]
        n_magpie_cols = len(magpie_cols)
        print(f"Found {n_magpie_cols} Magpie features")

        magpie_data = df[magpie_cols].values.astype(np.float32)

        # Handle NaN
        nan_mask = np.isnan(magpie_data)
        if nan_mask.any():
            col_means = np.nanmean(magpie_data, axis=0)
            for col_idx in range(magpie_data.shape[1]):
                magpie_data[nan_mask[:, col_idx], col_idx] = col_means[col_idx]

        # V12.20: Quantile-transform skewed features before z-score
        # Zero-inflated features (94-100% zeros) can't be fixed with log/sqrt.
        # Rank-based Gaussian transform handles any distribution shape.
        skew_threshold = TRAIN_CONFIG.get('magpie_skew_threshold', 0.0)
        magpie_skewed_indices = []
        if skew_threshold > 0:
            from scipy.stats import skew as _skew_func
            skewness = np.array([_skew_func(magpie_data[:, i]) for i in range(magpie_data.shape[1])])
            skewed_mask = np.abs(skewness) > skew_threshold
            magpie_skewed_indices = np.where(skewed_mask)[0].tolist()
            if magpie_skewed_indices:
                print(f"  Quantile-transforming {len(magpie_skewed_indices)} skewed Magpie features (|skew| > {skew_threshold}):")
                # Fixed RNG for reproducibility — jitter breaks ties in zero-inflated features
                _jitter_rng = np.random.default_rng(42)
                for idx in magpie_skewed_indices:
                    old_skew = skewness[idx]
                    col_data = magpie_data[:, idx]
                    # Add tiny jitter to break ties (99%+ zeros share same rank otherwise)
                    jittered = col_data + _jitter_rng.normal(0, 1e-6, len(col_data)).astype(np.float32)
                    # Rank → uniform (0,1) → Gaussian via inverse normal CDF
                    ranks = _rankdata(jittered, method='average')
                    uniform = (ranks - 0.5) / len(ranks)
                    magpie_data[:, idx] = _ndtri(uniform).astype(np.float32)
                    new_skew = _skew_func(magpie_data[:, idx])
                    print(f"    {magpie_cols[idx]}: skew {old_skew:.1f} → {new_skew:.2f}")

        # V12.20: SC-only normalization (avoids non-SC distribution bias)
        use_sc_only = TRAIN_CONFIG.get('magpie_sc_only_norm', False)
        if use_sc_only and sc_mask_np.any():
            magpie_mean = magpie_data[sc_mask_np].mean(axis=0)
            magpie_std = magpie_data[sc_mask_np].std(axis=0) + 1e-8
            print(f"  Magpie z-score: SC-only stats ({sc_mask_np.sum()} samples)")
        else:
            magpie_mean = magpie_data.mean(axis=0)
            magpie_std = magpie_data.std(axis=0) + 1e-8
            print(f"  Magpie z-score: all-sample stats ({len(magpie_data)} samples)")
        magpie_normalized = (magpie_data - magpie_mean) / magpie_std

        # Tokenize formulas
        print("Tokenizing formulas...")
        all_tokens = []
        for formula in formulas:
            tokens = tokenize_formula(formula)
            indices = tokens_to_indices(tokens, max_len=max_len)
            all_tokens.append(indices)
        formula_tokens = torch.stack(all_tokens)

        # Parse element compositions
        print("Parsing element compositions...")
        MAX_ELEMENTS = 12
        element_indices = torch.zeros(len(formulas), MAX_ELEMENTS, dtype=torch.long)
        element_fractions = torch.zeros(len(formulas), MAX_ELEMENTS, dtype=torch.float32)
        element_mask = torch.zeros(len(formulas), MAX_ELEMENTS, dtype=torch.bool)

        for i, formula in enumerate(formulas):
            parsed = parse_fraction_formula(formula)
            if not parsed:
                continue
            total = sum(parsed.values())
            for j, (element, frac) in enumerate(parsed.items()):
                if j >= MAX_ELEMENTS:
                    break
                try:
                    atomic_num = get_atomic_number(element)
                    element_indices[i, j] = atomic_num
                    element_fractions[i, j] = frac / total if total > 0 else frac
                    element_mask[i, j] = True
                except:
                    continue

        # Create tensors
        tc_tensor = torch.tensor(tc_normalized, dtype=torch.float32).unsqueeze(1)
        magpie_tensor = torch.tensor(magpie_normalized, dtype=torch.float32)
        is_sc_tensor = torch.tensor(is_sc_values, dtype=torch.long)
        label_tensor = torch.tensor(label_values, dtype=torch.long)
        hp_tensor = torch.tensor(hp_values, dtype=torch.float32)  # V12.19

        # V12.22: Classify SC families for theory-guided loss
        family_classifier = RuleBasedFamilyClassifier()
        family_values = np.zeros(len(formulas), dtype=np.int64)
        for i, formula in enumerate(formulas):
            if is_sc_values[i] == 1:
                parsed = parse_fraction_formula(formula)
                if parsed:
                    elements = set(parsed.keys())
                    family_values[i] = family_classifier.classify_from_elements(elements).value
                else:
                    family_values[i] = SuperconductorFamily.OTHER_UNKNOWN.value
            else:
                family_values[i] = SuperconductorFamily.NOT_SUPERCONDUCTOR.value
        family_tensor = torch.tensor(family_values, dtype=torch.long)
        # Print family distribution
        from collections import Counter as _Counter
        _fam_counts = _Counter(family_values.tolist())
        _fam_names = {f.value: f.name for f in SuperconductorFamily}
        print(f"  Family classification:")
        for fam_val, count in sorted(_fam_counts.items()):
            print(f"    {_fam_names.get(fam_val, 'UNKNOWN')}: {count}")

        # Get train indices (exclude holdout — matched by formula, not position)
        holdout_indices = load_holdout_indices(HOLDOUT_PATH, formulas)
        all_indices = set(range(len(formulas)))
        train_indices = sorted(all_indices - holdout_indices)

        is_sc_train = is_sc_tensor[train_indices]
        n_sc_train = int(is_sc_train.sum().item())
        n_non_sc_train = len(train_indices) - n_sc_train

        print(f"Training samples: {len(train_indices)} (SC: {n_sc_train}, Non-SC: {n_non_sc_train})")
        print(f"Holdout samples (NEVER TRAIN): {len(holdout_indices)}")

        norm_stats = {
            'tc_mean': tc_mean, 'tc_std': tc_std,
            'tc_log_transform': tc_log_transform,  # V12.20: True = tc_mean/std are in log1p space
            'magpie_mean': magpie_mean.tolist(),
            'magpie_std': magpie_std.tolist(),
            # V12.20: Magpie transform metadata for inference reproducibility
            'magpie_skewed_indices': magpie_skewed_indices,  # Which features were quantile-transformed
            'magpie_sc_only_norm': use_sc_only,
        }

        # Save cache for next run
        _save_cache(
            tensors={
                'formula_tokens': formula_tokens,
                'element_indices': element_indices,
                'element_fractions': element_fractions,
                'element_mask': element_mask,
                'tc_tensor': tc_tensor,
                'magpie_tensor': magpie_tensor,
                'is_sc_tensor': is_sc_tensor,
                'label_tensor': label_tensor,
                'hp_tensor': hp_tensor,  # V12.19
                'family_tensor': family_tensor,  # V12.22
            },
            norm_stats=norm_stats,
            n_magpie_cols=n_magpie_cols,
            train_indices=train_indices,
            csv_hash=csv_hash + mode_suffix,
            max_formula_len=max_len,
        )

        # Create dataset (10 tensors: 6 original + is_sc + label + hp + family)
        dataset = TensorDataset(
            element_indices, element_fractions, element_mask,
            formula_tokens, tc_tensor, magpie_tensor,
            is_sc_tensor, label_tensor, hp_tensor, family_tensor,
        )
        train_dataset = Subset(dataset, train_indices)

    # V12.11: Auto batch size based on GPU memory
    batch_size = TRAIN_CONFIG['batch_size']
    if batch_size == 'auto':
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_mem_gb >= 38:
                batch_size = 128
            elif gpu_mem_gb >= 22:
                batch_size = 96
            elif gpu_mem_gb >= 15:
                batch_size = 64
            else:
                # RTX 4060 (8GB) and similar - batch_size=48 works well with REINFORCE
                batch_size = 48
            print(f"Auto batch size: {batch_size} (GPU memory: {gpu_mem_gb:.1f}GB)")
        else:
            batch_size = 16
            print(f"Auto batch size: {batch_size} (CPU mode)")

    # Create DataLoader with V12.8 optimizations
    use_workers = TRAIN_CONFIG['num_workers']
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': use_workers,
    }
    if use_workers > 0:
        loader_kwargs['pin_memory'] = TRAIN_CONFIG['pin_memory']
        loader_kwargs['prefetch_factor'] = TRAIN_CONFIG['prefetch_factor']
        loader_kwargs['persistent_workers'] = TRAIN_CONFIG.get('persistent_workers', True)
    else:
        loader_kwargs['pin_memory'] = False

    # V12.12: Balanced sampling for contrastive mode
    if contrastive and TRAIN_CONFIG.get('balanced_sampling', True) and n_non_sc_train > 0:
        # Assign sampling weights: oversample minority class for ~50/50 batches
        is_sc_train_arr = is_sc_tensor[train_indices].numpy()
        sc_weight = 1.0 / max(n_sc_train, 1)
        non_sc_weight = 1.0 / max(n_non_sc_train, 1)
        sample_weights = np.where(is_sc_train_arr == 1, sc_weight, non_sc_weight)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_indices),
            replacement=True,
        )
        loader_kwargs['sampler'] = sampler
        # Cannot use shuffle with sampler
        print(f"Balanced sampling: SC weight={sc_weight:.6f}, Non-SC weight={non_sc_weight:.6f}")
    else:
        loader_kwargs['shuffle'] = True

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    print(f"DataLoader: workers={use_workers}, pin_memory={loader_kwargs.get('pin_memory', False)}")

    # V12.12: Save norm_stats.json for inference (always update when data changes)
    norm_stats_path = OUTPUT_DIR / 'norm_stats.json'
    norm_stats_with_cols = dict(norm_stats)
    # Include magpie column names for reference
    if 'magpie_cols' not in norm_stats_with_cols:
        # Reconstruct column names from the data
        try:
            df_cols = pd.read_csv(DATA_PATH, nrows=0)
            exclude = ['formula', 'Tc', 'composition', 'category', 'is_superconductor', 'compound possible', 'formula_original', 'requires_high_pressure']
            numeric_cols = [c for c in df_cols.columns if c not in exclude]
            # Filter to only numeric columns that were actually used
            norm_stats_with_cols['magpie_cols'] = [c for c in numeric_cols
                                                    if c in df_cols.select_dtypes(include=['number']).columns]
        except Exception:
            pass
    with open(norm_stats_path, 'w') as f:
        json.dump(norm_stats_with_cols, f)
    print(f"  Saved norm_stats to {norm_stats_path}")

    return train_loader, norm_stats, n_magpie_cols


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_models(magpie_dim: int, device: torch.device):
    """Create encoder and decoder models."""
    print("\n" + "=" * 60)
    print("Creating Models")
    print("=" * 60)

    encoder = FullMaterialsVAE(
        n_elements=118,
        element_embed_dim=MODEL_CONFIG['element_embed_dim'],
        n_attention_heads=8,
        magpie_dim=magpie_dim,
        fusion_dim=MODEL_CONFIG['fusion_dim'],
        encoder_hidden=MODEL_CONFIG['encoder_hidden'],
        latent_dim=MODEL_CONFIG['latent_dim'],
        decoder_hidden=MODEL_CONFIG['decoder_hidden'],
        dropout=0.1
    ).to(device)

    decoder = EnhancedTransformerDecoder(
        latent_dim=MODEL_CONFIG['latent_dim'],
        d_model=MODEL_CONFIG['d_model'],
        nhead=MODEL_CONFIG['nhead'],
        num_layers=MODEL_CONFIG['num_layers'],
        dim_feedforward=MODEL_CONFIG['dim_feedforward'],
        max_len=TRAIN_CONFIG['max_formula_len'],
        n_memory_tokens=MODEL_CONFIG['n_memory_tokens'],
        encoder_skip_dim=MODEL_CONFIG['fusion_dim'],
        use_skip_connection=True,
        use_stoich_conditioning=True,
        max_elements=12,
        n_stoich_tokens=4,
        dropout=0.1,
        # V12.8: Gradient checkpointing for memory optimization
        use_gradient_checkpointing=TRAIN_CONFIG.get('use_gradient_checkpointing', False),
    ).to(device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder: {enc_params:,} parameters")
    print(f"Decoder: {dec_params:,} parameters")
    print(f"Total: {enc_params + dec_params:,} parameters")

    return encoder, decoder


# ============================================================================
# LOSS FUNCTION WITH REINFORCE SUPPORT (V12.8)
# ============================================================================

class CombinedLossWithREINFORCE(nn.Module):
    """
    V12.8 Combined Loss with REINFORCE support.

    Components:
    1. Formula reconstruction (CE + optional REINFORCE with RLOO)
    2. Tc reconstruction (Huber loss on log-transformed Tc, V12.20)
    3. Magpie reconstruction (MSE)
    4. KL divergence
    5. Stoichiometry MSE (V12.4)

    REINFORCE uses GPU-native reward computation for fast training.
    KV caching enables efficient autoregressive sampling (~60x faster).
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        rl_weight: float = 0.0,
        tc_weight: float = 10.0,
        magpie_weight: float = 2.0,
        kl_weight: float = 0.0001,
        stoich_weight: float = 2.0,
        entropy_weight: float = 0.01,
        n_samples_rloo: int = 2,
        temperature: float = 0.8,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        use_autoregressive_reinforce: bool = True,
        tc_huber_delta: float = 0.0,  # V12.20: 0 = MSE (backward compat), >0 = Huber
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.rl_weight = rl_weight
        self.tc_weight = tc_weight
        self.magpie_weight = magpie_weight
        self.kl_weight = kl_weight
        self.stoich_weight = stoich_weight
        self.entropy_weight = entropy_weight
        self.n_samples_rloo = n_samples_rloo
        self.temperature = temperature
        self.tc_huber_delta = tc_huber_delta  # V12.20

        # V12.8: Autoregressive REINFORCE with KV caching
        self.use_autoregressive_reinforce = use_autoregressive_reinforce
        self._decoder = None
        self._max_len = 60

        # GPU-native reward config
        self.gpu_reward_config = get_default_gpu_reward_config()

        # Focal loss or standard CE
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.ce_loss = FocalLossWithLabelSmoothing(
                gamma=focal_gamma,
                smoothing=label_smoothing,
                ignore_index=PAD_IDX
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_IDX)

    def set_decoder(self, decoder, max_len: int = 60, draft_model=None):
        """Set decoder for autoregressive REINFORCE sampling (V12.8).

        Args:
            decoder: The EnhancedTransformerDecoder
            max_len: Maximum sequence length
            draft_model: Optional HybridDraft for speculative decoding (V12.9)
        """
        self._decoder = decoder
        self._max_len = max_len
        self._draft_model = draft_model

    def set_draft_model(self, draft_model):
        """Set draft model for speculative decoding (V12.9)."""
        self._draft_model = draft_model

    def compute_rloo_autoregressive(
        self,
        z: torch.Tensor,
        targets: torch.Tensor,
        encoder_skip: torch.Tensor = None,
        stoich_pred: torch.Tensor = None,
    ):
        """
        V12.8: Compute RLOO advantages using autoregressive sampling with KV cache.
        V12.9: Supports speculative decoding when draft_model is set.

        This samples from the model's actual autoregressive behavior (not teacher-forced
        logits), which is more accurate for REINFORCE training.

        Entropy bonus is added to the reward signal (not subtracted from loss) to
        encourage exploration in a principled RL manner.
        """
        if self._decoder is None:
            raise RuntimeError("Decoder not set. Call set_decoder() first.")

        batch_size = z.shape[0]
        n_samples = self.n_samples_rloo

        # V12.8 OPTIMIZATION: Batch all RLOO samples together for parallel generation
        # Instead of generating n_samples times sequentially, we expand the batch
        # and generate all samples in ONE forward pass (much better GPU utilization)

        # Expand inputs: [batch, ...] -> [batch * n_samples, ...]
        z_expanded = z.repeat(n_samples, 1)  # [batch * n_samples, latent_dim]
        encoder_skip_expanded = encoder_skip.repeat(n_samples, 1) if encoder_skip is not None else None
        stoich_pred_expanded = stoich_pred.repeat(n_samples, 1) if stoich_pred is not None else None
        targets_expanded = targets.repeat(n_samples, 1)  # [batch * n_samples, seq_len]

        # V12.15: Track REINFORCE sampling time separately (this is the expensive part)
        timing = get_timing_stats()
        if timing:
            # Stop loss_compute timing, start reinforce_sample timing
            timing.stop('loss_compute')
            timing.start('reinforce_sample')

        # V12.9: Use speculative decoding if draft model available
        if self._draft_model is not None:
            sampled_tokens, log_probs, entropy, mask, spec_stats = self._decoder.speculative_sample_for_reinforce(
                z=z_expanded,
                draft_model=self._draft_model,
                encoder_skip=encoder_skip_expanded,
                stoich_pred=stoich_pred_expanded,
                temperature=self.temperature,
                max_len=self._max_len,
                k=5,  # Draft 5 tokens at a time
            )
            # Store stats for logging (accessible via self._last_spec_stats)
            self._last_spec_stats = spec_stats
        else:
            # Generate ALL samples in one batched call (2x faster than sequential!)
            sampled_tokens, log_probs, entropy, mask = self._decoder.sample_for_reinforce(
                z=z_expanded,
                encoder_skip=encoder_skip_expanded,
                stoich_pred=stoich_pred_expanded,
                temperature=self.temperature,
                max_len=self._max_len,
            )
            self._last_spec_stats = None

        # V12.15: Stop sampling timing, resume loss_compute
        if timing:
            timing.stop('reinforce_sample')
            timing.start('loss_compute')

        # Pad/truncate to match target length
        if sampled_tokens.size(1) < targets.size(1):
            pad_len = targets.size(1) - sampled_tokens.size(1)
            sampled_tokens = F.pad(sampled_tokens, (0, pad_len), value=PAD_IDX)
            log_probs = F.pad(log_probs, (0, pad_len), value=0.0)
            entropy = F.pad(entropy, (0, pad_len), value=0.0)
            mask = F.pad(mask, (0, pad_len), value=0.0)
        elif sampled_tokens.size(1) > targets.size(1):
            sampled_tokens = sampled_tokens[:, :targets.size(1)]
            log_probs = log_probs[:, :targets.size(1)]
            entropy = entropy[:, :targets.size(1)]
            mask = mask[:, :targets.size(1)]

        # Compute task rewards for all samples at once
        task_rewards = compute_reward_gpu_native(
            sampled_tokens, targets_expanded, mask.bool(),
            config=self.gpu_reward_config,
            pad_idx=PAD_IDX, end_idx=END_IDX
        )

        # Compute entropy bonus per sample
        seq_entropy = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Combined reward = task_reward + entropy_bonus
        rewards = task_rewards + self.entropy_weight * seq_entropy

        # Sum log probs over sequence
        seq_log_prob = (log_probs * mask).sum(dim=1)

        # Reshape from [batch * n_samples] -> [n_samples, batch] for RLOO computation
        rewards_stack = rewards.view(n_samples, batch_size)
        log_probs_stack = seq_log_prob.view(n_samples, batch_size)
        entropy_stack = seq_entropy.view(n_samples, batch_size)

        # RLOO baseline computation
        total_reward = rewards_stack.sum(dim=0)

        advantages_list = []
        for i in range(n_samples):
            baseline_i = (total_reward - rewards_stack[i]) / (n_samples - 1)
            advantage_i = rewards_stack[i] - baseline_i
            advantages_list.append(advantage_i)

        advantages = torch.stack(advantages_list, dim=0).mean(dim=0)
        mean_log_probs = log_probs_stack.mean(dim=0)
        mean_rewards = rewards_stack.mean(dim=0)
        mean_entropy = entropy_stack.mean()  # Scalar for logging

        return advantages, mean_log_probs, mean_rewards, mean_entropy

    def sample_from_logits(self, logits: torch.Tensor, temperature: float):
        """Sample tokens from logits (fallback when not using autoregressive)."""
        batch_size, seq_len, vocab_size = logits.shape
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        flat_probs = probs.view(-1, vocab_size)
        sampled_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        sampled_tokens = sampled_flat.view(batch_size, seq_len)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        sampled_log_probs = log_probs.gather(2, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        return sampled_tokens, sampled_log_probs

    def compute_rloo_from_logits(self, logits, targets, mask):
        """Compute RLOO from teacher-forced logits (faster but less accurate)."""
        n_samples = self.n_samples_rloo
        all_rewards = []
        all_log_probs = []
        all_entropies = []

        # V12.8: Compute proper entropy from full logits (before temperature)
        # H(p) = -sum(p * log(p)) over vocabulary, then average over sequence
        probs_for_entropy = F.softmax(logits, dim=-1)  # [batch, seq_len, vocab]
        log_probs_for_entropy = F.log_softmax(logits, dim=-1)
        entropy_per_pos = -(probs_for_entropy * log_probs_for_entropy).sum(dim=-1)  # [batch, seq_len]

        for _ in range(n_samples):
            sampled_tokens, sampled_log_probs = self.sample_from_logits(logits, self.temperature)

            # Task rewards
            task_rewards = compute_reward_gpu_native(
                sampled_tokens, targets, mask,
                config=self.gpu_reward_config,
                pad_idx=PAD_IDX, end_idx=END_IDX
            )

            # V12.8: Use proper entropy H(p) = -sum(p * log(p)) from full distribution
            # (computed once outside loop since logits don't change per sample)
            seq_entropy = (entropy_per_pos * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)

            # Combined reward = task_reward + entropy_bonus
            rewards = task_rewards + self.entropy_weight * seq_entropy

            masked_log_probs = sampled_log_probs * mask.float()
            seq_log_prob = masked_log_probs.sum(dim=1)
            all_rewards.append(rewards)
            all_log_probs.append(seq_log_prob)
            all_entropies.append(seq_entropy)

        rewards_stack = torch.stack(all_rewards, dim=0)
        log_probs_stack = torch.stack(all_log_probs, dim=0)
        entropy_stack = torch.stack(all_entropies, dim=0)
        total_reward = rewards_stack.sum(dim=0)

        advantages_list = []
        for i in range(n_samples):
            baseline_i = (total_reward - rewards_stack[i]) / (n_samples - 1)
            advantage_i = rewards_stack[i] - baseline_i
            advantages_list.append(advantage_i)

        advantages = torch.stack(advantages_list, dim=0).mean(dim=0)
        mean_log_probs = log_probs_stack.mean(dim=0)
        mean_rewards = rewards_stack.mean(dim=0)
        mean_entropy = entropy_stack.mean()

        return advantages, mean_log_probs, mean_rewards, mean_entropy

    def forward(
        self,
        formula_logits: torch.Tensor,
        formula_targets: torch.Tensor,
        tc_pred: torch.Tensor,
        tc_true: torch.Tensor,
        magpie_pred: torch.Tensor,
        magpie_true: torch.Tensor,
        kl_loss: torch.Tensor,
        tc_weight_override: float = None,
        magpie_weight_override: float = None,
        # V12.4: Stoichiometry loss inputs
        fraction_pred: torch.Tensor = None,
        element_fractions: torch.Tensor = None,
        element_mask: torch.Tensor = None,
        element_count_pred: torch.Tensor = None,
        # V12.8: Autoregressive REINFORCE inputs
        z: torch.Tensor = None,
        encoder_skip: torch.Tensor = None,
        stoich_pred_for_reinforce: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss with optional REINFORCE."""

        batch_size, seq_len, vocab_size = formula_logits.shape
        mask = (formula_targets != PAD_IDX)

        # Use curriculum weights if provided
        tc_w = tc_weight_override if tc_weight_override is not None else self.tc_weight
        magpie_w = magpie_weight_override if magpie_weight_override is not None else self.magpie_weight

        # 1. Formula CE Loss
        if self.use_focal_loss:
            targets_with_pad = formula_targets.clone()
            targets_with_pad[~mask] = PAD_IDX
            formula_ce_loss = self.ce_loss(formula_logits, targets_with_pad)
        else:
            logits_flat = formula_logits.contiguous().view(-1, vocab_size)
            targets_flat = formula_targets.contiguous().view(-1)
            ce_loss_per_token = self.ce_loss(logits_flat, targets_flat)
            ce_loss_per_token = ce_loss_per_token.view(batch_size, seq_len)
            formula_ce_loss = (ce_loss_per_token * mask.float()).sum(dim=1).mean()

        # 2. REINFORCE Loss (skip if rl_weight=0)
        # V12.8: Entropy bonus is now part of the reward signal, not direct loss subtraction
        rl_entropy = torch.tensor(0.0, device=formula_logits.device)
        if self.rl_weight > 0:
            if self.use_autoregressive_reinforce and z is not None and self._decoder is not None:
                # V12.8: True autoregressive sampling with KV cache
                advantages, mean_log_probs, mean_rewards, rl_entropy = self.compute_rloo_autoregressive(
                    z=z,
                    targets=formula_targets,
                    encoder_skip=encoder_skip,
                    stoich_pred=stoich_pred_for_reinforce,
                )
            else:
                # Fallback: sample from teacher-forced logits
                advantages, mean_log_probs, mean_rewards, rl_entropy = self.compute_rloo_from_logits(
                    formula_logits, formula_targets, mask
                )
            reinforce_loss = -(advantages * mean_log_probs).mean()
        else:
            reinforce_loss = torch.tensor(0.0, device=formula_logits.device)
            mean_rewards = torch.tensor(0.0, device=formula_logits.device)

        # 3. Entropy computation (for logging only when REINFORCE disabled)
        # When REINFORCE is enabled, entropy is part of the reward signal
        if self.rl_weight > 0:
            entropy = rl_entropy  # Use RL entropy for logging
        else:
            # Compute from logits for monitoring (not used in loss)
            probs = F.softmax(formula_logits, dim=-1)
            log_probs = F.log_softmax(formula_logits, dim=-1)
            entropy_per_position = -(probs * log_probs).sum(dim=-1)
            entropy = (entropy_per_position * mask.float()).sum(dim=1).mean()

        # 4. Tc Loss (V12.20: Huber loss clips gradient from outliers)
        if self.tc_huber_delta > 0:
            tc_loss = F.huber_loss(tc_pred.squeeze(), tc_true.squeeze(), delta=self.tc_huber_delta)
        else:
            tc_loss = F.mse_loss(tc_pred.squeeze(), tc_true.squeeze())

        # 5. Magpie Loss
        magpie_loss = F.mse_loss(magpie_pred, magpie_true)

        # 6. Stoichiometry Loss (V12.4)
        stoich_loss = torch.tensor(0.0, device=formula_logits.device)
        element_count_loss = torch.tensor(0.0, device=formula_logits.device)

        if fraction_pred is not None and element_fractions is not None and element_mask is not None:
            elem_mask_float = element_mask.float()
            squared_error = (fraction_pred - element_fractions) ** 2
            masked_squared_error = squared_error * elem_mask_float
            n_valid = elem_mask_float.sum(dim=1, keepdim=True).clamp(min=1)
            per_sample_mse = masked_squared_error.sum(dim=1) / n_valid.squeeze(-1)
            stoich_loss = per_sample_mse.mean()

            if element_count_pred is not None:
                element_count_target = element_mask.sum(dim=1).float()
                element_count_loss = F.mse_loss(element_count_pred, element_count_target)

        # 7. Combined Loss
        # V12.8: Entropy bonus is now part of REINFORCE reward, NOT subtracted here
        formula_loss = self.ce_weight * formula_ce_loss + self.rl_weight * reinforce_loss
        total = (
            formula_loss +
            tc_w * tc_loss +
            magpie_w * magpie_loss +
            self.kl_weight * kl_loss +
            self.stoich_weight * stoich_loss +
            0.5 * element_count_loss
            # Entropy bonus removed - now in REINFORCE reward signal
        )

        # Compute accuracy metrics
        predictions = formula_logits.argmax(dim=-1)
        correct = (predictions == formula_targets) & mask
        token_accuracy = correct.sum().float() / mask.sum().float()
        seq_correct = (correct | ~mask).all(dim=1)
        exact_match = seq_correct.float().mean()

        result = {
            'total': total,
            'formula_loss': formula_ce_loss,  # CE component only for logging
            'reinforce_loss': reinforce_loss,
            'tc_loss': tc_loss,
            'magpie_loss': magpie_loss,
            'stoich_loss': stoich_loss,
            'kl_loss': kl_loss,
            'entropy': entropy,
            'mean_reward': mean_rewards.mean() if torch.is_tensor(mean_rewards) else mean_rewards,
            'token_accuracy': token_accuracy,
            'exact_match': exact_match,
        }

        # V12.9: Add speculative decoding stats if available
        if hasattr(self, '_last_spec_stats') and self._last_spec_stats is not None:
            result['spec_acceptance_rate'] = self._last_spec_stats.get('acceptance_rate', 0)
            result['spec_tokens_per_step'] = self._last_spec_stats.get('avg_tokens_per_step', 1)

        return result


# Backward compatibility alias
CombinedLoss = CombinedLossWithREINFORCE


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(encoder, decoder, epoch, suffix='', entropy_manager=None,
                    enc_opt=None, dec_opt=None, enc_scheduler=None, dec_scheduler=None,
                    prev_exact=None, best_exact=None, theory_loss_fn=None):
    """Save model checkpoint with full training state for proper resumption."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if suffix:
        path = OUTPUT_DIR / f'checkpoint_{suffix}.pt'
    else:
        path = OUTPUT_DIR / f'checkpoint_epoch_{epoch:04d}.pt'

    checkpoint_data = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }

    # V12.9: Save entropy manager state if available
    if entropy_manager is not None:
        checkpoint_data['entropy_manager_state'] = entropy_manager.get_state()

    # V12.10: Save optimizer and scheduler state for proper resumption
    if enc_opt is not None:
        checkpoint_data['enc_optimizer_state_dict'] = enc_opt.state_dict()
    if dec_opt is not None:
        checkpoint_data['dec_optimizer_state_dict'] = dec_opt.state_dict()
    if enc_scheduler is not None:
        checkpoint_data['enc_scheduler_state_dict'] = enc_scheduler.state_dict()
        checkpoint_data['scheduler_type'] = type(enc_scheduler).__name__  # V12.11: Track scheduler type
    if dec_scheduler is not None:
        checkpoint_data['dec_scheduler_state_dict'] = dec_scheduler.state_dict()

    # V12.22: Save theory loss function state (learnable BCS/cuprate predictors)
    if theory_loss_fn is not None:
        checkpoint_data['theory_loss_fn_state_dict'] = theory_loss_fn.state_dict()

    # V12.10: Save training state variables
    if prev_exact is not None:
        checkpoint_data['prev_exact'] = prev_exact
    if best_exact is not None:
        checkpoint_data['best_exact'] = best_exact

    torch.save(checkpoint_data, path)
    print(f"  Saved checkpoint: {path.name}", flush=True)


# ============================================================================
# V12.17: LATENT Z CACHING & PREDICTION LOGGING
# ============================================================================

def cache_z_vectors(encoder, loader, device, epoch, cache_path, dataset_info=None,
                    decoder=None, mode='z_only'):
    """
    Compute and cache latent z vectors and optionally decoder predictions.

    This enables:
    - Z-space analysis (correlations with Tc, family, element presence)
    - Training z-conditioned mini-decoder for speculative decoding
    - Debugging encoder/decoder behavior over training
    - Post-training analysis linking z coordinates to predictions

    Args:
        encoder: Trained encoder model
        loader: DataLoader for the dataset
        device: Torch device
        epoch: Current epoch number
        cache_path: Path to save the cache
        dataset_info: Optional dict with formulas, Tc values, etc.
        decoder: Optional decoder for generating predictions (required for 'z_and_predictions' mode)
        mode: Cache mode - 'z_only', 'z_and_predictions', or 'full'

    Cache modes:
        - 'z_only': Just z vectors (~50MB) - fast
        - 'z_and_predictions': Z + generated formulas (~200MB) - includes decoder output
        - 'full': Z + predictions + per-token log_probs (~500MB) - for detailed analysis

    Saves:
        {
            'z_vectors': [N, latent_dim] tensor,
            'tc_values': [N] tensor,
            'tc_pred': [N] tensor (encoder's Tc prediction),
            'is_sc': [N] tensor (1=superconductor, 0=non-SC),
            'target_tokens': [N, seq_len] tensor (ground truth),
            'generated_tokens': [N, seq_len] tensor (decoder output, if mode != 'z_only'),
            'exact_match': [N] tensor (1=match, 0=mismatch, if mode != 'z_only'),
            'log_probs': [N, seq_len] tensor (per-token log probs, if mode == 'full'),
            'epoch': int,
            'timestamp': str,
            'mode': str,
            'stats': dict,
        }
    """
    import datetime
    from superconductor.models.autoregressive_decoder import IDX_TO_TOKEN, END_IDX, PAD_IDX

    encoder.eval()
    if decoder is not None:
        decoder.eval()

    all_z = []
    all_tc = []
    all_tc_pred = []
    all_is_sc = []
    all_target_tokens = []
    all_generated_tokens = []
    all_exact_match = []
    all_log_probs = []
    all_attended_input = []

    sample_idx = 0
    include_predictions = mode in ['z_and_predictions', 'full'] and decoder is not None
    include_log_probs = mode == 'full' and decoder is not None

    with torch.no_grad():
        for batch in loader:
            # Unpack batch (same as training loop — V12.19: 9 tensors)
            batch_tensors = [b.to(device) for b in batch]
            elem_idx, elem_frac, elem_mask, tokens, tc, magpie, is_sc, labels = batch_tensors[:8]

            # Encode
            encoder_out = encoder(elem_idx, elem_frac, elem_mask, magpie, tc)
            z = encoder_out['z']

            # Skip NaN batches
            if torch.isnan(z).any():
                sample_idx += z.size(0)
                continue

            all_z.append(z.cpu())
            all_tc.append(tc.cpu())
            all_tc_pred.append(encoder_out['tc_pred'].cpu())
            all_is_sc.append(is_sc.cpu())
            all_target_tokens.append(tokens.cpu())

            if include_predictions:
                attended_input = encoder_out.get('attended_input')
                all_attended_input.append(attended_input.cpu() if attended_input is not None else None)

                # Generate predictions using decoder
                if include_log_probs:
                    gen_tokens, log_probs, _, _ = decoder.sample_for_reinforce(
                        z=z, encoder_skip=attended_input, temperature=0.0,  # Greedy
                        max_len=tokens.size(1)
                    )
                    all_log_probs.append(log_probs.cpu())
                else:
                    gen_tokens = decoder.generate_with_kv_cache(
                        z=z, encoder_skip=attended_input, temperature=0.0,
                        max_len=tokens.size(1), return_log_probs=False, return_entropy=False
                    )[0]

                all_generated_tokens.append(gen_tokens.cpu())

                # Compute exact match (same method as evaluate_true_autoregressive)
                targets = tokens[:, 1:]  # Skip START token
                gen_len = gen_tokens.size(1)
                tgt_len = targets.size(1)

                # Pad or truncate generated tokens to match target length
                if gen_len < tgt_len:
                    gen_compare = F.pad(gen_tokens, (0, tgt_len - gen_len), value=PAD_IDX)
                elif gen_len > tgt_len:
                    gen_compare = gen_tokens[:, :tgt_len]
                else:
                    gen_compare = gen_tokens

                # Compare only non-PAD positions
                mask = (targets != PAD_IDX)
                mismatches_per_seq = ((gen_compare != targets) & mask).sum(dim=1)
                exact_match = (mismatches_per_seq == 0).long().cpu()

                all_exact_match.append(exact_match)

            sample_idx += z.size(0)

    encoder.train()
    if decoder is not None:
        decoder.train()

    # Concatenate all batches
    z_vectors = torch.cat(all_z, dim=0)
    tc_values = torch.cat(all_tc, dim=0)
    tc_pred_values = torch.cat(all_tc_pred, dim=0)
    is_sc_values = torch.cat(all_is_sc, dim=0)
    target_tokens = torch.cat(all_target_tokens, dim=0)

    # Build cache dict
    cache_data = {
        'z_vectors': z_vectors,
        'tc_values': tc_values,
        'tc_pred': tc_pred_values,
        'is_sc': is_sc_values,
        'target_tokens': target_tokens,
        'epoch': epoch,
        'timestamp': datetime.datetime.now().isoformat(),
        'latent_dim': z_vectors.size(1),
        'n_samples': z_vectors.size(0),
        'mode': mode,
    }

    if include_predictions:
        # Pad generated_tokens to same length before concatenating
        max_gen_len = max(gt.size(1) for gt in all_generated_tokens)
        padded_gen_tokens = []
        for gt in all_generated_tokens:
            if gt.size(1) < max_gen_len:
                pad = torch.full((gt.size(0), max_gen_len - gt.size(1)), PAD_IDX, dtype=gt.dtype)
                gt = torch.cat([gt, pad], dim=1)
            padded_gen_tokens.append(gt)
        generated_tokens = torch.cat(padded_gen_tokens, dim=0)
        exact_match = torch.cat(all_exact_match, dim=0)
        cache_data['generated_tokens'] = generated_tokens
        cache_data['exact_match'] = exact_match
        cache_data['exact_match_pct'] = exact_match.float().mean().item() * 100

    if include_log_probs:
        # Pad log_probs to same length
        max_len = max(lp.size(1) for lp in all_log_probs)
        padded_log_probs = []
        for lp in all_log_probs:
            if lp.size(1) < max_len:
                pad = torch.zeros(lp.size(0), max_len - lp.size(1))
                lp = torch.cat([lp, pad], dim=1)
            padded_log_probs.append(lp)
        cache_data['log_probs'] = torch.cat(padded_log_probs, dim=0)

    # Compute basic statistics
    sc_mask_cache = is_sc_values.bool()
    # Squeeze tc_values from [N,1] to [N] to avoid broadcast mismatch with tc_pred_values [N]
    tc_values_flat = tc_values.squeeze(-1)
    tc_mae_all = (tc_pred_values - tc_values_flat).abs().mean().item()

    # V12.21: SC-only Tc MAE (primary metric — non-SC Tc predictions are meaningless)
    tc_mae_sc = 0.0
    tc_mae_non_sc = 0.0
    tc_kelvin_breakdown = {}
    if sc_mask_cache.any():
        sc_tc_err = (tc_pred_values[sc_mask_cache] - tc_values_flat[sc_mask_cache]).abs()
        tc_mae_sc = sc_tc_err.mean().item()

        # Convert to Kelvin for breakdown if norm_stats available
        if dataset_info and 'tc_mean' in dataset_info:
            tc_m = dataset_info['tc_mean']
            tc_s = dataset_info['tc_std']
            tc_log = dataset_info.get('tc_log_transform', False)
            # Denormalize: tc_norm * std + mean -> log1p space (or K)
            # Squeeze to 1D — tc_values is [N,1] from TensorDataset, tc_pred_values is [N]
            sc_tc_true_flat = tc_values[sc_mask_cache].squeeze(-1)
            sc_tc_pred_flat = tc_pred_values[sc_mask_cache].squeeze(-1)
            sc_tc_true_denorm = sc_tc_true_flat * tc_s + tc_m
            sc_tc_pred_denorm = sc_tc_pred_flat * tc_s + tc_m
            if tc_log:
                # expm1 to convert from log1p space to Kelvin
                sc_tc_true_K = torch.expm1(sc_tc_true_denorm)
                sc_tc_pred_K = torch.expm1(sc_tc_pred_denorm)
            else:
                sc_tc_true_K = sc_tc_true_denorm
                sc_tc_pred_K = sc_tc_pred_denorm
            tc_err_K = (sc_tc_pred_K - sc_tc_true_K).abs()

            # Breakdown by Tc range in Kelvin
            ranges = [(0, 10, '0-10K'), (10, 50, '10-50K'), (50, 100, '50-100K'), (100, float('inf'), '100K+')]
            for lo, hi, label in ranges:
                range_mask = (sc_tc_true_K >= lo) & (sc_tc_true_K < hi)
                if range_mask.any():
                    tc_kelvin_breakdown[label] = {
                        'mae_K': tc_err_K[range_mask].mean().item(),
                        'count': range_mask.sum().item(),
                    }
            tc_kelvin_breakdown['overall_mae_K'] = tc_err_K.mean().item()

    if (~sc_mask_cache).any():
        tc_mae_non_sc = (tc_pred_values[~sc_mask_cache] - tc_values_flat[~sc_mask_cache]).abs().mean().item()

    cache_data['stats'] = {
        'z_mean': z_vectors.mean(dim=0),
        'z_std': z_vectors.std(dim=0),
        'z_min': z_vectors.min(dim=0).values,
        'z_max': z_vectors.max(dim=0).values,
        'z_norm_mean': z_vectors.norm(dim=1).mean().item(),
        'z_norm_std': z_vectors.norm(dim=1).std().item(),
        'tc_mean': tc_values.mean().item(),
        'tc_std': tc_values.std().item(),
        'tc_pred_mae': tc_mae_all,          # Legacy: all-sample MAE (inflated by non-SC)
        'tc_pred_mae_sc': tc_mae_sc,        # V12.21: SC-only MAE (primary metric)
        'tc_pred_mae_non_sc': tc_mae_non_sc,  # V12.21: Non-SC MAE (expect high - untrained)
        'tc_kelvin_breakdown': tc_kelvin_breakdown,  # V12.21: MAE by Tc range in Kelvin
        'n_superconductors': is_sc_values.sum().item(),
        'n_non_sc': (1 - is_sc_values).sum().item(),
    }

    # Save cache
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(exist_ok=True)
    torch.save(cache_data, cache_path)

    print(f"  Cached {z_vectors.size(0)} samples to {cache_path.name} (mode={mode})")
    print(f"    Z: {z_vectors.size(1)} dims, norm={z_vectors.norm(dim=1).mean():.2f}±{z_vectors.norm(dim=1).std():.2f}")
    if include_predictions:
        print(f"    Exact match: {cache_data['exact_match_pct']:.1f}%")
    # V12.21: Show SC-only Tc MAE as primary metric
    tc_stats = cache_data['stats']
    print(f"    Tc MAE (SC-only): {tc_stats['tc_pred_mae_sc']:.4f} | "
          f"(all: {tc_stats['tc_pred_mae']:.4f}, non-SC: {tc_stats['tc_pred_mae_non_sc']:.4f})")
    if tc_stats.get('tc_kelvin_breakdown'):
        kb = tc_stats['tc_kelvin_breakdown']
        parts = []
        for label in ['0-10K', '10-50K', '50-100K', '100K+']:
            if label in kb:
                parts.append(f"{label}: {kb[label]['mae_K']:.1f}K (n={kb[label]['count']})")
        if parts:
            print(f"    Tc Kelvin: {' | '.join(parts)}")
        if 'overall_mae_K' in kb:
            print(f"    Tc overall MAE: {kb['overall_mae_K']:.1f}K")

    return cache_data


def log_training_metrics(epoch, metrics, log_path, true_eval=None):
    """
    Append training metrics to a CSV log file for post-training analysis.

    Creates/appends to a CSV with columns:
        epoch, exact_match, accuracy, loss, tc_loss, magpie_loss, stoich_loss,
        rl_loss, reward, entropy, entropy_weight, z_norm, tf_ratio, lr,
        true_exact (every 10 epochs), epoch_time, timestamp

    Args:
        epoch: Current epoch number
        metrics: Dict of training metrics from train_epoch()
        log_path: Path to CSV log file
        true_eval: Optional dict from evaluate_true_autoregressive()
    """
    import csv
    import datetime

    log_path = Path(log_path)
    file_exists = log_path.exists()

    # Define columns
    columns = [
        'epoch', 'exact_match', 'accuracy', 'loss', 'tc_loss', 'magpie_loss',
        'stoich_loss', 'rl_loss', 'reward', 'entropy', 'entropy_weight',
        'z_norm', 'tf_ratio', 'contrastive_loss', 'hp_loss', 'sc_loss',
        'theory_loss',  # V12.22
        'true_exact', 'epoch_time', 'timestamp'
    ]

    # Extract values
    row = {
        'epoch': epoch,
        'exact_match': metrics.get('exact_match', 0),
        'accuracy': metrics.get('accuracy', 0),
        'loss': metrics.get('loss', 0),
        'tc_loss': metrics.get('tc_loss', 0),
        'magpie_loss': metrics.get('magpie_loss', 0),
        'stoich_loss': metrics.get('stoich_loss', 0),
        'rl_loss': metrics.get('reinforce_loss', 0),  # V12.21: Fix key mismatch (was 'rl_loss')
        'reward': metrics.get('mean_reward', 0),     # V12.21: Fix key mismatch (was 'reward')
        'entropy': metrics.get('entropy', 0),
        'entropy_weight': metrics.get('entropy_weight', 0),  # Set by train() if entropy_manager exists
        'z_norm': metrics.get('z_norm', 0),
        'tf_ratio': metrics.get('tf_ratio', 0),
        'contrastive_loss': metrics.get('contrastive_loss', 0),
        'hp_loss': metrics.get('hp_loss', 0),
        'sc_loss': metrics.get('sc_loss', 0),
        'theory_loss': metrics.get('theory_loss', 0),  # V12.22
        'true_exact': true_eval['true_exact_match'] if true_eval else '',
        'epoch_time': metrics.get('epoch_time', 0),
        'timestamp': datetime.datetime.now().isoformat(),
    }

    # Write to CSV
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _strip_compiled_prefix(state_dict):
    """Strip '_orig_mod.' prefix from compiled model state dict keys.

    Handles both top-level prefixes (encoder) and nested prefixes (decoder's
    transformer_decoder was compiled separately, creating keys like
    'transformer_decoder._orig_mod.layers.0...').
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Handle top-level _orig_mod. prefix
        if new_key.startswith('_orig_mod.'):
            new_key = new_key[len('_orig_mod.'):]
        # Handle nested _orig_mod. prefixes (from separately compiled submodules)
        new_key = new_key.replace('._orig_mod.', '.')
        new_state_dict[new_key] = value
    return new_state_dict


def load_checkpoint(encoder, decoder, checkpoint_path, entropy_manager=None,
                    enc_opt=None, dec_opt=None, enc_scheduler=None, dec_scheduler=None,
                    theory_loss_fn=None):
    """Load model checkpoint with full training state for proper resumption.

    Returns:
        dict with keys: 'start_epoch', 'prev_exact', 'best_exact'
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # V12.10: Handle checkpoints saved with torch.compile (have '_orig_mod.' prefix)
    enc_state = checkpoint['encoder_state_dict']
    dec_state = checkpoint['decoder_state_dict']

    # V12.13: Detect whether target models are compiled (for rollback into compiled models)
    target_is_compiled = any(k.startswith('_orig_mod.') for k in encoder.state_dict().keys())

    # Check if checkpoint was saved with compiled model
    checkpoint_is_compiled = (
        any(k.startswith('_orig_mod.') for k in enc_state.keys()) or
        any('._orig_mod.' in k for k in dec_state.keys())
    )

    if checkpoint_is_compiled and not target_is_compiled:
        # Loading compiled checkpoint into uncompiled model (initial resume)
        print("  Detected compiled checkpoint - stripping '_orig_mod.' prefixes", flush=True)
        enc_state = _strip_compiled_prefix(enc_state)
        dec_state = _strip_compiled_prefix(dec_state)
    elif not checkpoint_is_compiled and target_is_compiled:
        # Loading uncompiled checkpoint into compiled model (shouldn't happen, but handle it)
        print("  Adding '_orig_mod.' prefixes for compiled model target", flush=True)
        enc_state = {'_orig_mod.' + k: v for k, v in enc_state.items()}
        # Decoder: only transformer_decoder is compiled (nested prefix)
        dec_new = {}
        for k, v in dec_state.items():
            if k.startswith('transformer_decoder.'):
                dec_new['transformer_decoder._orig_mod.' + k[len('transformer_decoder.'):]] = v
            else:
                dec_new[k] = v
        dec_state = dec_new
    elif checkpoint_is_compiled and target_is_compiled:
        # Both compiled — keys should match directly
        pass
    # else: both uncompiled — keys match directly

    # strict=False: deterministic encoder drops fc_logvar, so old checkpoint has extra keys
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        print(f"  [Checkpoint] Missing keys in encoder: {missing}", flush=True)
    if unexpected:
        print(f"  [Checkpoint] Unexpected keys in encoder (ignored): {unexpected}", flush=True)
    decoder.load_state_dict(dec_state)
    start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}", flush=True)

    # V12.9: Load entropy manager state if available
    if entropy_manager is not None and 'entropy_manager_state' in checkpoint:
        entropy_manager.load_state(checkpoint['entropy_manager_state'])
        print(f"  Restored entropy manager state", flush=True)

    # V12.22: Load theory loss function state if available
    if theory_loss_fn is not None and 'theory_loss_fn_state_dict' in checkpoint:
        try:
            theory_loss_fn.load_state_dict(checkpoint['theory_loss_fn_state_dict'])
            print(f"  Restored theory loss function state (BCS/cuprate predictors)", flush=True)
        except (RuntimeError, KeyError) as e:
            print(f"  [Checkpoint] Theory loss state incompatible, starting fresh: {e}", flush=True)
    elif theory_loss_fn is not None:
        print(f"  [Checkpoint] No theory_loss_fn state in checkpoint — starting fresh", flush=True)

    # V12.10: Load optimizer state if available
    # Encoder optimizer may fail to restore if parameter count changed (e.g., fc_logvar removed)
    if enc_opt is not None and 'enc_optimizer_state_dict' in checkpoint:
        try:
            enc_opt.load_state_dict(checkpoint['enc_optimizer_state_dict'])
            print(f"  Restored encoder optimizer state", flush=True)
        except (ValueError, KeyError) as e:
            print(f"  [Checkpoint] Encoder optimizer state incompatible (param count changed), "
                  f"using fresh optimizer: {e}", flush=True)
    if dec_opt is not None and 'dec_optimizer_state_dict' in checkpoint:
        dec_opt.load_state_dict(checkpoint['dec_optimizer_state_dict'])
        print(f"  Restored decoder optimizer state", flush=True)

    # V12.11: Load scheduler state if available AND scheduler type matches
    # Incompatible scheduler types (e.g., CosineAnnealingWarmRestarts vs CosineAnnealingLR)
    # have different internal states and loading across types causes training instability
    saved_scheduler_type = checkpoint.get('scheduler_type', None)
    current_scheduler_type = type(enc_scheduler).__name__ if enc_scheduler else None

    if enc_scheduler is not None and 'enc_scheduler_state_dict' in checkpoint:
        if saved_scheduler_type is None:
            # Old checkpoint without scheduler type - skip to be safe
            print(f"  Skipping scheduler restore (old checkpoint, type unknown)", flush=True)
        elif saved_scheduler_type == current_scheduler_type:
            enc_scheduler.load_state_dict(checkpoint['enc_scheduler_state_dict'])
            print(f"  Restored encoder scheduler state", flush=True)
        else:
            print(f"  Skipping scheduler restore (type changed: {saved_scheduler_type} → {current_scheduler_type})", flush=True)
    if dec_scheduler is not None and 'dec_scheduler_state_dict' in checkpoint:
        if saved_scheduler_type is not None and saved_scheduler_type == current_scheduler_type:
            dec_scheduler.load_state_dict(checkpoint['dec_scheduler_state_dict'])
            print(f"  Restored decoder scheduler state", flush=True)
        # Skip silently if types don't match (warning printed above)

    # V12.10: Return training state variables
    result = {
        'start_epoch': start_epoch,
        'prev_exact': checkpoint.get('prev_exact', 0),
        'best_exact': checkpoint.get('best_exact', 0),
    }

    if 'prev_exact' in checkpoint:
        print(f"  Restored prev_exact={result['prev_exact']:.3f}, best_exact={result['best_exact']:.3f}", flush=True)

    return result


# ============================================================================
# TRAINING LOOP
# ============================================================================

@torch.no_grad()
def evaluate_true_autoregressive(encoder, decoder, loader, device, max_samples=1000,
                                  log_errors=True, epoch=None):
    """
    V12.18: Enhanced evaluation with full Z-space diagnostics and reconstruction metrics.

    Evaluates TRUE autoregressive exact match (no teacher forcing) AND captures
    per-sample Z norms, Tc/Magpie/stoichiometry reconstruction quality to correlate
    with error patterns.

    Args:
        encoder: The encoder model
        decoder: The decoder model
        loader: DataLoader with evaluation data
        device: torch device
        max_samples: Maximum samples to evaluate (for speed)
        log_errors: Whether to log errors to file (default True)
        epoch: Current epoch number for error log filename

    Returns:
        dict with true_exact_match, sample statistics, and Z diagnostics
    """
    import numpy as np

    encoder.eval()
    decoder.eval()

    total_exact = 0
    total_samples = 0
    n_by_errors = {0: 0, 1: 0, 2: 0, 3: 0, 'more': 0}  # Track error distribution

    # V12.15: Collect error records for analysis
    error_records = []

    # V12.18: Collect per-sample reconstruction diagnostics
    all_z_norms = []          # Per-sample Z vector L2 norm
    all_z_max_dims = []       # Per-sample max absolute Z dimension
    all_tc_true = []          # Ground truth Tc
    all_tc_pred = []          # Encoder's Tc prediction
    all_magpie_mse = []       # Per-sample Magpie reconstruction MSE
    all_stoich_mse = []       # Per-sample stoichiometry reconstruction MSE
    all_n_errors = []         # Per-sample token error count (0 = exact match)
    all_seq_lens = []         # Per-sample sequence length
    all_is_sc = []            # Per-sample SC flag
    all_n_elements = []       # Per-sample number of elements (from elem_mask)

    for batch in loader:
        if total_samples >= max_samples:
            break

        # Unpack batch (V12.12: 8 tensors with contrastive data)
        batch_tensors = [b.to(device) for b in batch]
        elem_idx, elem_frac, elem_mask, formula_tokens, tc, magpie = batch_tensors[:6]
        # Get is_sc flag if available (for error analysis)
        is_sc = batch_tensors[6] if len(batch_tensors) > 6 else None

        batch_size = formula_tokens.size(0)

        # Encode (same order as train_epoch) - get ALL encoder outputs
        encoder_out = encoder(elem_idx, elem_frac, elem_mask, magpie, tc)
        z = encoder_out['z']
        encoder_skip = encoder_out['attended_input']
        tc_pred = encoder_out['tc_pred']
        magpie_pred = encoder_out['magpie_pred']
        fraction_pred = encoder_out.get('fraction_pred')
        element_count_pred = encoder_out.get('element_count_pred')

        # V12.18: Compute per-sample reconstruction metrics
        z_norms = z.float().norm(dim=1)                       # [B]
        z_max_dim = z.float().abs().max(dim=1).values         # [B]
        tc_pred_vals = tc_pred.squeeze().float()               # [B]
        tc_true_vals = tc.squeeze().float()                    # [B]
        magpie_mse = (magpie_pred.float() - magpie.float()).pow(2).mean(dim=1)  # [B]

        # Stoichiometry MSE (fraction + element count predictions vs input)
        stoich_mse_vals = torch.zeros(batch_size, device=device)
        if fraction_pred is not None:
            # fraction_pred: [B, max_elems, 1] or similar, elem_frac: [B, max_elems]
            frac_target = elem_frac.float()
            frac_pred_squeezed = fraction_pred.squeeze(-1).float() if fraction_pred.dim() == 3 else fraction_pred.float()
            # Mask by valid elements
            valid_mask = elem_mask.bool()
            for si in range(batch_size):
                vm = valid_mask[si]
                if vm.any():
                    stoich_mse_vals[si] = (frac_pred_squeezed[si][vm] - frac_target[si][vm]).pow(2).mean()

        # Number of elements per sample
        n_elements = elem_mask.bool().sum(dim=1)  # [B]

        all_z_norms.extend(z_norms.cpu().tolist())
        all_z_max_dims.extend(z_max_dim.cpu().tolist())
        all_tc_true.extend(tc_true_vals.cpu().tolist())
        all_tc_pred.extend(tc_pred_vals.cpu().tolist())
        all_magpie_mse.extend(magpie_mse.cpu().tolist())
        all_stoich_mse.extend(stoich_mse_vals.cpu().tolist())
        all_n_elements.extend(n_elements.cpu().tolist())
        if is_sc is not None:
            all_is_sc.extend(is_sc.cpu().tolist())

        # Generate autoregressively (TRUE inference - no teacher forcing)
        generated_tokens, _, _ = decoder.generate_with_kv_cache(
            z=z,
            encoder_skip=encoder_skip,
            temperature=0.001,  # Near-greedy for evaluation
            max_len=decoder.max_len,
        )

        # Compare with targets (excluding START token from targets)
        targets = formula_tokens[:, 1:]  # Remove START token

        # Pad/truncate to match lengths
        gen_len = generated_tokens.size(1)
        tgt_len = targets.size(1)

        if gen_len < tgt_len:
            generated_tokens = F.pad(generated_tokens, (0, tgt_len - gen_len), value=PAD_IDX)
        elif gen_len > tgt_len:
            generated_tokens = generated_tokens[:, :tgt_len]

        # Count matches
        mask = (targets != PAD_IDX)
        matches = (generated_tokens == targets) & mask
        mismatches_per_seq = ((generated_tokens != targets) & mask).sum(dim=1)

        # Track exact matches and error distribution
        for i in range(batch_size):
            n_errors = mismatches_per_seq[i].item()
            seq_len = int(mask[i].sum().item())
            all_n_errors.append(n_errors)
            all_seq_lens.append(seq_len)

            if n_errors == 0:
                total_exact += 1
                n_by_errors[0] += 1
            elif n_errors == 1:
                n_by_errors[1] += 1
            elif n_errors == 2:
                n_by_errors[2] += 1
            elif n_errors == 3:
                n_by_errors[3] += 1
            else:
                n_by_errors['more'] += 1

            # V12.15: Record errors for analysis (with V12.18 Z diagnostics)
            if log_errors and n_errors > 0:
                target_formula = indices_to_formula(formula_tokens[i])
                generated_formula = indices_to_formula(generated_tokens[i])

                # Find specific token mismatches
                target_seq = targets[i]
                gen_seq = generated_tokens[i]
                seq_mask = mask[i]
                mismatch_positions = []
                for pos in range(len(seq_mask)):
                    if seq_mask[pos] and target_seq[pos] != gen_seq[pos]:
                        target_token = IDX_TO_TOKEN.get(target_seq[pos].item(), '?')
                        gen_token = IDX_TO_TOKEN.get(gen_seq[pos].item(), '?')
                        mismatch_positions.append(f"pos{pos}:{target_token}->{gen_token}")

                sample_global_idx = total_samples + i
                error_records.append({
                    'sample_idx': sample_global_idx,
                    'n_errors': int(n_errors),
                    'target': target_formula,
                    'generated': generated_formula,
                    'is_sc': bool(is_sc[i].item()) if is_sc is not None else None,
                    'mismatches': mismatch_positions,
                    'seq_len': seq_len,
                    # V12.18: Per-sample reconstruction diagnostics
                    'z_norm': round(z_norms[i].item(), 4),
                    'z_max_dim': round(z_max_dim[i].item(), 4),
                    'tc_true': round(tc_true_vals[i].item(), 4),
                    'tc_pred': round(tc_pred_vals[i].item(), 4),
                    'tc_error': round(abs(tc_pred_vals[i].item() - tc_true_vals[i].item()), 4),
                    'magpie_mse': round(magpie_mse[i].item(), 6),
                    'stoich_mse': round(stoich_mse_vals[i].item(), 6),
                    'n_elements': int(n_elements[i].item()),
                })

        total_samples += batch_size

    true_exact_match = total_exact / total_samples if total_samples > 0 else 0

    # V12.18: Compute Z-space diagnostic statistics
    all_z_norms_np = np.array(all_z_norms[:total_samples])
    all_n_errors_np = np.array(all_n_errors[:total_samples])
    all_tc_true_np = np.array(all_tc_true[:total_samples])
    all_tc_pred_np = np.array(all_tc_pred[:total_samples])
    all_magpie_mse_np = np.array(all_magpie_mse[:total_samples])
    all_stoich_mse_np = np.array(all_stoich_mse[:total_samples])
    all_seq_lens_np = np.array(all_seq_lens[:total_samples])
    all_z_max_dims_np = np.array(all_z_max_dims[:total_samples])
    all_n_elements_np = np.array(all_n_elements[:total_samples])

    exact_mask = all_n_errors_np == 0
    error_mask = all_n_errors_np > 0

    # Correlations (handle edge cases)
    def safe_corrcoef(a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    z_diagnostics = {
        # Z-norm statistics: exact vs error samples
        'z_norm_overall': {'mean': float(all_z_norms_np.mean()), 'std': float(all_z_norms_np.std())},
        'z_norm_exact': {'mean': float(all_z_norms_np[exact_mask].mean()), 'std': float(all_z_norms_np[exact_mask].std())} if exact_mask.any() else None,
        'z_norm_errors': {'mean': float(all_z_norms_np[error_mask].mean()), 'std': float(all_z_norms_np[error_mask].std())} if error_mask.any() else None,

        # Z max dimension (outlier detector)
        'z_max_dim_overall': {'mean': float(all_z_max_dims_np.mean()), 'std': float(all_z_max_dims_np.std())},
        'z_max_dim_exact': float(all_z_max_dims_np[exact_mask].mean()) if exact_mask.any() else None,
        'z_max_dim_errors': float(all_z_max_dims_np[error_mask].mean()) if error_mask.any() else None,

        # Tc reconstruction quality
        'tc_mae_overall': float(np.abs(all_tc_pred_np - all_tc_true_np).mean()),
        'tc_mae_exact': float(np.abs(all_tc_pred_np[exact_mask] - all_tc_true_np[exact_mask]).mean()) if exact_mask.any() else None,
        'tc_mae_errors': float(np.abs(all_tc_pred_np[error_mask] - all_tc_true_np[error_mask]).mean()) if error_mask.any() else None,
        'tc_r2': float(1 - np.sum((all_tc_pred_np - all_tc_true_np)**2) / max(np.sum((all_tc_true_np - all_tc_true_np.mean())**2), 1e-8)),

        # Magpie reconstruction quality
        'magpie_mse_overall': float(all_magpie_mse_np.mean()),
        'magpie_mse_exact': float(all_magpie_mse_np[exact_mask].mean()) if exact_mask.any() else None,
        'magpie_mse_errors': float(all_magpie_mse_np[error_mask].mean()) if error_mask.any() else None,

        # Stoichiometry reconstruction quality
        'stoich_mse_overall': float(all_stoich_mse_np.mean()),
        'stoich_mse_exact': float(all_stoich_mse_np[exact_mask].mean()) if exact_mask.any() else None,
        'stoich_mse_errors': float(all_stoich_mse_np[error_mask].mean()) if error_mask.any() else None,

        # Sequence length vs errors
        'seq_len_exact': float(all_seq_lens_np[exact_mask].mean()) if exact_mask.any() else None,
        'seq_len_errors': float(all_seq_lens_np[error_mask].mean()) if error_mask.any() else None,

        # Element count vs errors
        'n_elements_exact': float(all_n_elements_np[exact_mask].mean()) if exact_mask.any() else None,
        'n_elements_errors': float(all_n_elements_np[error_mask].mean()) if error_mask.any() else None,

        # V12.21: SC-only Tc MAE (primary metric — non-SC Tc is meaningless)
        'tc_mae_sc_only': None,
        'tc_mae_non_sc': None,

        # Correlations: what predicts errors?
        'corr_z_norm_vs_errors': safe_corrcoef(all_z_norms_np, all_n_errors_np),
        'corr_tc_error_vs_formula_errors': safe_corrcoef(np.abs(all_tc_pred_np - all_tc_true_np), all_n_errors_np),
        'corr_magpie_mse_vs_errors': safe_corrcoef(all_magpie_mse_np, all_n_errors_np),
        'corr_stoich_mse_vs_errors': safe_corrcoef(all_stoich_mse_np, all_n_errors_np),
        'corr_seq_len_vs_errors': safe_corrcoef(all_seq_lens_np, all_n_errors_np),
        'corr_n_elements_vs_errors': safe_corrcoef(all_n_elements_np, all_n_errors_np),
        'corr_tc_true_vs_errors': safe_corrcoef(all_tc_true_np, all_n_errors_np),

        # Error breakdown by Z-norm quartile
        'errors_by_z_norm_quartile': {},
        # Error breakdown by Tc range
        'errors_by_tc_range': {},
        # Error breakdown by sequence length bucket
        'errors_by_seq_len_bucket': {},
    }

    # V12.21: SC-only Tc MAE
    if all_is_sc:
        all_is_sc_np = np.array(all_is_sc[:total_samples])
        sc_eval_mask = all_is_sc_np > 0
        if sc_eval_mask.any():
            z_diagnostics['tc_mae_sc_only'] = float(np.abs(
                all_tc_pred_np[sc_eval_mask] - all_tc_true_np[sc_eval_mask]).mean())
        if (~sc_eval_mask).any():
            z_diagnostics['tc_mae_non_sc'] = float(np.abs(
                all_tc_pred_np[~sc_eval_mask] - all_tc_true_np[~sc_eval_mask]).mean())

    # Z-norm quartile analysis
    if len(all_z_norms_np) > 4:
        quartiles = np.percentile(all_z_norms_np, [25, 50, 75])
        for qi, (lo, hi, label) in enumerate([
            (0, quartiles[0], 'Q1_lowest'),
            (quartiles[0], quartiles[1], 'Q2'),
            (quartiles[1], quartiles[2], 'Q3'),
            (quartiles[2], float('inf'), 'Q4_highest'),
        ]):
            qmask = (all_z_norms_np >= lo) & (all_z_norms_np < hi) if label != 'Q4_highest' else (all_z_norms_np >= lo)
            if qmask.any():
                z_diagnostics['errors_by_z_norm_quartile'][label] = {
                    'n_samples': int(qmask.sum()),
                    'exact_pct': float((all_n_errors_np[qmask] == 0).mean() * 100),
                    'avg_errors': float(all_n_errors_np[qmask].mean()),
                    'z_norm_range': [float(lo), float(hi) if label != 'Q4_highest' else float(all_z_norms_np.max())],
                }

    # Tc range analysis
    tc_ranges = [(0, 10, '0-10K'), (10, 30, '10-30K'), (30, 77, '30-77K'),
                 (77, 120, '77-120K'), (120, 200, '120-200K'), (200, float('inf'), '>200K')]
    for lo, hi, label in tc_ranges:
        tc_mask = (all_tc_true_np >= lo) & (all_tc_true_np < hi)
        if tc_mask.any():
            z_diagnostics['errors_by_tc_range'][label] = {
                'n_samples': int(tc_mask.sum()),
                'exact_pct': float((all_n_errors_np[tc_mask] == 0).mean() * 100),
                'avg_errors': float(all_n_errors_np[tc_mask].mean()),
                'avg_z_norm': float(all_z_norms_np[tc_mask].mean()),
            }

    # Sequence length bucket analysis
    seq_buckets = [(1, 10, '1-10'), (11, 20, '11-20'), (21, 30, '21-30'),
                   (31, 40, '31-40'), (41, 60, '41-60')]
    for lo, hi, label in seq_buckets:
        sl_mask = (all_seq_lens_np >= lo) & (all_seq_lens_np <= hi)
        if sl_mask.any():
            z_diagnostics['errors_by_seq_len_bucket'][label] = {
                'n_samples': int(sl_mask.sum()),
                'exact_pct': float((all_n_errors_np[sl_mask] == 0).mean() * 100),
                'avg_errors': float(all_n_errors_np[sl_mask].mean()),
                'avg_z_norm': float(all_z_norms_np[sl_mask].mean()),
            }

    # V12.18: Print Z diagnostic summary to console
    zd = z_diagnostics
    z_exact = zd['z_norm_exact']
    z_err = zd['z_norm_errors']
    print(f"  Z-diag: norm exact={z_exact['mean']:.1f}±{z_exact['std']:.1f}" if z_exact else "  Z-diag: no exact samples", end='')
    print(f" | err={z_err['mean']:.1f}±{z_err['std']:.1f}" if z_err else " | no error samples", end='')
    tc_sc_str = f", SC={zd['tc_mae_sc_only']:.4f}" if zd.get('tc_mae_sc_only') is not None else ""
    print(f" | Tc MAE={zd['tc_mae_overall']:.4f}{tc_sc_str} (exact={zd['tc_mae_exact']:.4f}, err={zd['tc_mae_errors']:.4f})" if zd['tc_mae_exact'] is not None and zd['tc_mae_errors'] is not None else '')
    print(f"  Correlations: z_norm→err={zd['corr_z_norm_vs_errors']:.3f} | tc_err→err={zd['corr_tc_error_vs_formula_errors']:.3f} | "
          f"magpie→err={zd['corr_magpie_mse_vs_errors']:.3f} | seq_len→err={zd['corr_seq_len_vs_errors']:.3f} | "
          f"n_elem→err={zd['corr_n_elements_vs_errors']:.3f}")

    # V12.15: Write error log to file (V12.18: enriched with Z diagnostics)
    if log_errors:
        error_log_path = OUTPUT_DIR / f'error_analysis_epoch_{epoch if epoch else "latest"}.json'

        # Sort by number of errors (worst first) then by sequence length
        error_records.sort(key=lambda x: (-x['n_errors'], -x['seq_len']))

        error_summary = {
            'epoch': epoch,
            'total_samples': total_samples,
            'total_errors': len(error_records),
            'exact_match_pct': true_exact_match * 100,
            'error_distribution': {str(k): v for k, v in n_by_errors.items()},
            'errors_by_type': {
                'sc_errors': sum(1 for e in error_records if e['is_sc'] is True),
                'non_sc_errors': sum(1 for e in error_records if e['is_sc'] is False),
                'unknown_errors': sum(1 for e in error_records if e['is_sc'] is None),
            },
            'avg_errors_per_failed': sum(e['n_errors'] for e in error_records) / len(error_records) if error_records else 0,
            # V12.18: Full Z-space diagnostics
            'z_diagnostics': z_diagnostics,
            'error_records': error_records,
        }

        import json
        with open(error_log_path, 'w') as f:
            json.dump(error_summary, f, indent=2)

        print(f"  Error analysis saved: {error_log_path.name} ({len(error_records)} errors)")

    return {
        'true_exact_match': true_exact_match,
        'total_samples': total_samples,
        'exact_count': total_exact,
        'error_distribution': n_by_errors,
        'z_diagnostics': z_diagnostics,
    }


def train_epoch(encoder, decoder, loader, loss_fn, enc_opt, dec_opt, scaler, device, use_amp,
                tf_ratio=1.0, tc_weight=10.0, magpie_weight=2.0,
                focal_loss_fn=None, amp_dtype=torch.float16, accumulation_steps=1,
                contrastive_loss_fn=None, contrastive_weight=0.0, non_sc_formula_weight=0.5,
                selective_backprop=False, selective_backprop_threshold=0.33,
                enable_timing=True, hp_loss_weight=0.0, sc_loss_weight=0.0,
                theory_loss_fn=None, theory_weight=0.0, norm_stats=None):
    """Train for one epoch with curriculum weights and teacher forcing.

    V12.8: Added amp_dtype for configurable mixed precision (bfloat16/float16)
           Added accumulation_steps for gradient accumulation (larger effective batch)
           Added REINFORCE support via CombinedLossWithREINFORCE
    V12.12: Added contrastive loss for mixed SC/non-SC training
    V12.12: Added selective backpropagation - skip backward on easy batches
    V12.15: Added enable_timing for compute/time profiling
    V12.19: Added hp_loss_weight for high-pressure prediction head
    V12.21: Added sc_loss_weight for SC/non-SC classification head
    V12.22: Added theory_loss_fn, theory_weight, norm_stats for theory-guided consistency
    """
    global _timing_stats

    encoder.train()
    decoder.train()

    # V12.15: Initialize timing stats
    timing = TimingStats() if enable_timing else None
    _timing_stats = timing
    if timing:
        timing.start_epoch()

    total_loss = 0
    total_acc = 0
    total_exact = 0
    total_tc = 0
    total_magpie = 0
    total_stoich = 0
    total_reinforce = 0
    total_reward = 0
    total_entropy = 0
    total_z_norm = 0
    total_contrastive = 0
    total_hp_loss = 0  # V12.19
    total_sc_loss = 0  # V12.21
    total_theory_loss = 0  # V12.22
    total_sc_exact = 0
    total_non_sc_exact = 0
    total_spec_accept = 0  # V12.9: Speculative decoding acceptance rate
    total_spec_tokens = 0  # V12.9: Tokens per step with speculation
    n_spec_batches = 0     # V12.9: Batches with speculative decoding stats
    n_batches = 0
    n_sc_batches = 0
    n_non_sc_batches = 0
    n_skipped = 0  # V12.12: Selective backprop skip counter

    # V12.12: Running average loss for selective backpropagation
    running_avg_loss = None
    sb_momentum = 0.95  # EMA momentum for running average

    # Zero gradients at start of accumulation
    enc_opt.zero_grad()
    dec_opt.zero_grad()

    # V12.15: Start timing for data loading
    if timing:
        timing.start('data_load')

    for batch_idx, batch in enumerate(loader):
        # V12.15: Stop data load timing, start batch processing
        if timing:
            timing.stop('data_load')

        # V12.22: Unpack 10 tensors (6 original + is_sc + label + hp + family)
        batch_tensors = [b.to(device) for b in batch]
        elem_idx, elem_frac, elem_mask, tokens, tc, magpie, is_sc, labels = batch_tensors[:8]
        hp_labels = batch_tensors[8] if len(batch_tensors) > 8 else torch.zeros_like(is_sc, dtype=torch.float32)
        family_labels = batch_tensors[9] if len(batch_tensors) > 9 else None

        sc_mask = is_sc.bool()  # True = superconductor

        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            # V12.15: Time encoder forward
            if timing:
                timing.start('encoder_fwd')

            # Encoder forward (all samples — SC and non-SC both get encoded)
            encoder_out = encoder(elem_idx, elem_frac, elem_mask, magpie, tc)
            z = encoder_out['z']

            if timing:
                timing.stop('encoder_fwd')

            # V12.14: Early NaN detection on latent z — skip batch before decoder forward
            # Sporadic NaN in z can corrupt optimizer momentum buffers even when total loss
            # appears finite (NaN doesn't always propagate through all loss components)
            if torch.isnan(z).any():
                n_skipped += 1
                enc_opt.zero_grad()
                dec_opt.zero_grad()
                if timing:
                    timing.start('data_load')  # Resume data load timing
                continue

            tc_pred = encoder_out['tc_pred']
            magpie_pred = encoder_out['magpie_pred']
            attended_input = encoder_out['attended_input']
            # NOTE: 'kl_loss' is actually L2 reg (mean(z²)) since deterministic encoder.
            # Key name kept for compatibility — see attention_vae.py forward() comment.
            kl_loss = encoder_out['kl_loss']

            # Stoichiometry predictions for loss and conditioning
            fraction_pred = encoder_out.get('fraction_pred')
            element_count_pred = encoder_out.get('element_count_pred')
            if fraction_pred is not None and element_count_pred is not None:
                stoich_pred = torch.cat([fraction_pred, element_count_pred.unsqueeze(-1)], dim=-1)
            else:
                stoich_pred = None

            # V12.15: Time decoder forward
            if timing:
                timing.start('decoder_fwd')

            # Decoder forward with teacher forcing ratio (all samples)
            formula_logits, _ = decoder(
                z, tokens, encoder_skip=attended_input,
                stoich_pred=stoich_pred, teacher_forcing_ratio=tf_ratio
            )
            formula_targets = tokens[:, 1:]

            if timing:
                timing.stop('decoder_fwd')

            # V12.12: Contrastive loss on full batch (SC + non-SC)
            contrastive_loss_val = torch.tensor(0.0, device=device)
            if contrastive_loss_fn is not None and contrastive_weight > 0:
                contrastive_loss_val = contrastive_loss_fn(z, labels)

            # V12.19: High-pressure prediction loss (only on SC samples)
            hp_loss_val = torch.tensor(0.0, device=device)
            if hp_loss_weight > 0:
                hp_pred = encoder_out.get('hp_pred')
                if hp_pred is not None and sc_mask.any():
                    # Only compute HP loss on superconductor samples
                    sc_hp_pred = hp_pred[sc_mask]
                    sc_hp_labels = hp_labels[sc_mask]
                    # pos_weight handles class imbalance (~100:1 non-HP to HP among SC)
                    n_pos = sc_hp_labels.sum().clamp(min=1)
                    n_neg = (1 - sc_hp_labels).sum().clamp(min=1)
                    pos_weight = (n_neg / n_pos).clamp(max=50.0)  # Cap at 50x
                    hp_loss_val = F.binary_cross_entropy_with_logits(
                        sc_hp_pred, sc_hp_labels,
                        pos_weight=pos_weight,
                    )

            # V12.21: SC/non-SC classification loss (on ALL samples)
            # Cross-head consistency: sc_head receives z + other head predictions
            sc_loss_val = torch.tensor(0.0, device=device)
            if sc_loss_weight > 0:
                sc_pred = encoder_out.get('sc_pred')
                if sc_pred is not None:
                    sc_loss_val = F.binary_cross_entropy_with_logits(
                        sc_pred, is_sc.float(),
                    )

            # V12.22: Theory-guided consistency loss (SC samples only)
            theory_loss_val = torch.tensor(0.0, device=device)
            if theory_loss_fn is not None and theory_weight > 0 and sc_mask.any():
                theory_result = theory_loss_fn(
                    formula_tokens=tokens[sc_mask],
                    predicted_tc=tc_pred[sc_mask],
                    magpie_features=magpie[sc_mask],
                    families=family_labels[sc_mask] if family_labels is not None else None,
                    element_fractions=elem_frac[sc_mask],
                    tc_is_normalized=True,
                    tc_mean=norm_stats['tc_mean'] if norm_stats else 32.0,
                    tc_std=norm_stats['tc_std'] if norm_stats else 35.0,
                )
                theory_loss_val = theory_result['total']

            # V12.15: Time loss computation (includes REINFORCE sampling)
            if timing:
                timing.start('loss_compute')

            # V12.12: Compute main loss differently for SC vs non-SC
            if sc_mask.all():
                # Pure SC batch — standard loss (no masking needed)
                loss_dict = loss_fn(
                    formula_logits=formula_logits,
                    formula_targets=formula_targets,
                    tc_pred=tc_pred,
                    tc_true=tc,
                    magpie_pred=magpie_pred,
                    magpie_true=magpie,
                    kl_loss=kl_loss,
                    tc_weight_override=tc_weight,
                    magpie_weight_override=magpie_weight,
                    fraction_pred=fraction_pred,
                    element_fractions=elem_frac,
                    element_mask=elem_mask,
                    element_count_pred=element_count_pred,
                    z=z,
                    encoder_skip=attended_input,
                    stoich_pred_for_reinforce=stoich_pred,
                )
                loss = (loss_dict['total'] + contrastive_weight * contrastive_loss_val
                        + hp_loss_weight * hp_loss_val
                        + sc_loss_weight * sc_loss_val
                        + theory_weight * theory_loss_val)  # V12.22

            elif (~sc_mask).all():
                # Pure non-SC batch — formula loss only (lower weight), no Tc/Magpie/REINFORCE
                loss_dict = loss_fn(
                    formula_logits=formula_logits,
                    formula_targets=formula_targets,
                    tc_pred=tc_pred,
                    tc_true=tc,
                    magpie_pred=magpie_pred,
                    magpie_true=magpie,
                    kl_loss=kl_loss,
                    tc_weight_override=0.0,     # No Tc loss for non-SC
                    magpie_weight_override=0.0,  # No Magpie loss for non-SC
                    fraction_pred=fraction_pred,
                    element_fractions=elem_frac,
                    element_mask=elem_mask,
                    element_count_pred=element_count_pred,
                    z=None,  # No REINFORCE for non-SC
                    encoder_skip=attended_input,
                    stoich_pred_for_reinforce=None,
                )
                # Scale formula loss by non_sc_formula_weight
                loss = (non_sc_formula_weight * loss_dict['total'] + contrastive_weight * contrastive_loss_val
                        + sc_loss_weight * sc_loss_val
                        + theory_weight * theory_loss_val)  # V12.22

            else:
                # Mixed batch — compute SC and non-SC losses separately
                n_sc = sc_mask.sum().item()
                n_non_sc = (~sc_mask).sum().item()

                # SC portion: full loss
                sc_loss_dict = loss_fn(
                    formula_logits=formula_logits[sc_mask],
                    formula_targets=formula_targets[sc_mask],
                    tc_pred=tc_pred[sc_mask],
                    tc_true=tc[sc_mask],
                    magpie_pred=magpie_pred[sc_mask],
                    magpie_true=magpie[sc_mask],
                    kl_loss=kl_loss,  # KL is already a scalar over full batch
                    tc_weight_override=tc_weight,
                    magpie_weight_override=magpie_weight,
                    fraction_pred=fraction_pred[sc_mask] if fraction_pred is not None else None,
                    element_fractions=elem_frac[sc_mask],
                    element_mask=elem_mask[sc_mask],
                    element_count_pred=element_count_pred[sc_mask] if element_count_pred is not None else None,
                    z=z[sc_mask],
                    encoder_skip=attended_input[sc_mask],
                    stoich_pred_for_reinforce=stoich_pred[sc_mask] if stoich_pred is not None else None,
                )

                # Non-SC portion: formula-only, lower weight
                non_sc_loss_dict = loss_fn(
                    formula_logits=formula_logits[~sc_mask],
                    formula_targets=formula_targets[~sc_mask],
                    tc_pred=tc_pred[~sc_mask],
                    tc_true=tc[~sc_mask],
                    magpie_pred=magpie_pred[~sc_mask],
                    magpie_true=magpie[~sc_mask],
                    kl_loss=torch.tensor(0.0, device=device),
                    tc_weight_override=0.0,
                    magpie_weight_override=0.0,
                    fraction_pred=fraction_pred[~sc_mask] if fraction_pred is not None else None,
                    element_fractions=elem_frac[~sc_mask],
                    element_mask=elem_mask[~sc_mask],
                    element_count_pred=element_count_pred[~sc_mask] if element_count_pred is not None else None,
                    z=None,  # No REINFORCE for non-SC
                    encoder_skip=attended_input[~sc_mask],
                    stoich_pred_for_reinforce=None,
                )

                # Weighted combination: SC at full weight, non-SC at reduced weight
                batch_total = n_sc + n_non_sc
                sc_frac = n_sc / batch_total
                non_sc_frac = n_non_sc / batch_total
                loss_dict = sc_loss_dict  # Use SC metrics for logging
                loss = (sc_frac * sc_loss_dict['total'] +
                        non_sc_frac * non_sc_formula_weight * non_sc_loss_dict['total'] +
                        contrastive_weight * contrastive_loss_val +
                        hp_loss_weight * hp_loss_val +
                        sc_loss_weight * sc_loss_val +
                        theory_weight * theory_loss_val)  # V12.22

        # V12.15: Stop loss computation timing
        if timing:
            timing.stop('loss_compute')

        # V12.12: Selective backpropagation — skip backward on easy batches
        current_loss = loss.item()

        # V12.13: NaN guard — skip batch entirely if loss is NaN to prevent corruption
        if math.isnan(current_loss) or math.isinf(current_loss):
            n_skipped += 1
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            if timing:
                timing.start('data_load')  # Resume data load timing
            continue

        # Update running average loss (EMA)
        if running_avg_loss is None:
            running_avg_loss = current_loss
        else:
            running_avg_loss = sb_momentum * running_avg_loss + (1 - sb_momentum) * current_loss

        # Skip backward pass if batch loss is well below running average
        # (only after we have a stable estimate — skip first 10 batches worth of warmup)
        skip_backward = (
            selective_backprop
            and n_batches >= 10
            and current_loss < selective_backprop_threshold * running_avg_loss
        )

        if skip_backward:
            n_skipped += 1
        else:
            # V12.15: Time backward pass
            if timing:
                timing.start('backward')

            # V12.8: Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps

            # Backward with AMP
            scaler.scale(scaled_loss).backward()

            if timing:
                timing.stop('backward')

            # Step optimizer every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                # V12.15: Time optimizer step
                if timing:
                    timing.start('optimizer')

                scaler.unscale_(enc_opt)
                scaler.unscale_(dec_opt)

                # Gradient clipping (V12.22: include theory loss params in encoder clip)
                _enc_params = list(encoder.parameters())
                if theory_loss_fn is not None:
                    _enc_params = _enc_params + list(theory_loss_fn.parameters())
                enc_grad_norm = torch.nn.utils.clip_grad_norm_(_enc_params, 1.0)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

                # V12.14: NaN gradient guard — skip optimizer step if gradients are NaN/Inf
                # This prevents NaN from poisoning Adam momentum buffers (exp_avg, exp_avg_sq)
                # which causes permanent corruption that persists across checkpoint resumes
                if math.isnan(enc_grad_norm) or math.isinf(enc_grad_norm):
                    enc_opt.zero_grad()
                    n_skipped += 1
                else:
                    scaler.step(enc_opt)

                if math.isnan(dec_grad_norm) or math.isinf(dec_grad_norm):
                    dec_opt.zero_grad()
                else:
                    scaler.step(dec_opt)

                scaler.update()

                # Zero gradients for next accumulation
                enc_opt.zero_grad()
                dec_opt.zero_grad()

                if timing:
                    timing.stop('optimizer')

        # Accumulate metrics
        total_loss += loss_dict['total'].item()
        total_acc += loss_dict['token_accuracy'].item()
        total_exact += loss_dict['exact_match'].item()
        total_tc += loss_dict['tc_loss'].item()
        total_magpie += loss_dict['magpie_loss'].item()
        total_stoich += loss_dict['stoich_loss'].item()
        total_reinforce += loss_dict['reinforce_loss'].item()
        total_reward += loss_dict['mean_reward'].item() if torch.is_tensor(loss_dict['mean_reward']) else loss_dict['mean_reward']
        total_entropy += loss_dict['entropy'].item()
        total_contrastive += contrastive_loss_val.item() if torch.is_tensor(contrastive_loss_val) else contrastive_loss_val
        total_hp_loss += hp_loss_val.item() if torch.is_tensor(hp_loss_val) else hp_loss_val
        total_sc_loss += sc_loss_val.item() if torch.is_tensor(sc_loss_val) else sc_loss_val  # V12.21
        total_theory_loss += theory_loss_val.item() if torch.is_tensor(theory_loss_val) else theory_loss_val  # V12.22
        z_norm_val = z.detach().float().norm(dim=1).mean().item()
        if not math.isnan(z_norm_val):
            total_z_norm += z_norm_val

        # V12.9: Track speculative decoding stats if available
        if 'spec_acceptance_rate' in loss_dict:
            total_spec_accept += loss_dict['spec_acceptance_rate']
            total_spec_tokens += loss_dict['spec_tokens_per_step']
            n_spec_batches += 1

        n_batches += 1

        # V12.12: Track SC vs non-SC accuracy separately
        if sc_mask.any():
            sc_preds = formula_logits[sc_mask].argmax(dim=-1)
            sc_targets = formula_targets[sc_mask]
            sc_seq_mask = (sc_targets != PAD_IDX)
            sc_correct = ((sc_preds == sc_targets) | ~sc_seq_mask).all(dim=1)
            total_sc_exact += sc_correct.float().mean().item()
            n_sc_batches += 1
        if (~sc_mask).any():
            nsc_preds = formula_logits[~sc_mask].argmax(dim=-1)
            nsc_targets = formula_targets[~sc_mask]
            nsc_seq_mask = (nsc_targets != PAD_IDX)
            nsc_correct = ((nsc_preds == nsc_targets) | ~nsc_seq_mask).all(dim=1)
            total_non_sc_exact += nsc_correct.float().mean().item()
            n_non_sc_batches += 1

        # V12.15: Resume data load timing for next batch
        if timing:
            timing.start('data_load')

    # V12.15: Stop final data_load timing (for last batch's "next" that never comes)
    if timing:
        timing.stop('data_load')

    # Guard against zero-batch epochs (all batches NaN-skipped)
    if n_batches == 0:
        print(f"  [WARNING] Entire epoch skipped ({n_skipped} NaN/Inf batches)", flush=True)
        return {
            'loss': float('nan'), 'accuracy': 0, 'exact_match': 0,
            'tc_loss': 0, 'magpie_loss': 0, 'stoich_loss': 0,
            'reinforce_loss': 0, 'mean_reward': 0, 'entropy': 0,
            'contrastive_loss': 0, 'hp_loss': 0, 'sc_loss': 0, 'theory_loss': 0,
            'z_norm': 0, 'n_skipped': n_skipped, 'n_total_batches': 0,
        }

    results = {
        'loss': total_loss / n_batches,
        'accuracy': total_acc / n_batches,
        'exact_match': total_exact / n_batches,
        'tc_loss': total_tc / n_batches,
        'magpie_loss': total_magpie / n_batches,
        'stoich_loss': total_stoich / n_batches,
        'reinforce_loss': total_reinforce / n_batches,
        'mean_reward': total_reward / n_batches,
        'entropy': total_entropy / n_batches,
        'contrastive_loss': total_contrastive / n_batches,
        'hp_loss': total_hp_loss / n_batches,  # V12.19
        'sc_loss': total_sc_loss / n_batches,  # V12.21
        'theory_loss': total_theory_loss / n_batches,  # V12.22
        'z_norm': total_z_norm / n_batches,
        'n_skipped': n_skipped,  # V12.12: Selective backprop skips
        'n_total_batches': n_batches,
    }

    # V12.12: Add SC/non-SC breakdown
    if n_sc_batches > 0:
        results['sc_exact_match'] = total_sc_exact / n_sc_batches
    if n_non_sc_batches > 0:
        results['non_sc_exact_match'] = total_non_sc_exact / n_non_sc_batches

    # V12.9: Add speculative decoding stats
    if n_spec_batches > 0:
        results['spec_acceptance_rate'] = total_spec_accept / n_spec_batches
        results['spec_tokens_per_step'] = total_spec_tokens / n_spec_batches

    # V12.15: Add timing stats
    if timing:
        results['timing'] = timing
        results['epoch_time'] = timing.get_epoch_time()
        results['timing_breakdown'] = timing.get_breakdown()

    return results


def train():
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================================================
    # Environment-aware DataLoader tuning (overrides TRAIN_CONFIG defaults)
    # ========================================================================
    env = detect_environment()
    TRAIN_CONFIG['num_workers'] = env['num_workers']
    TRAIN_CONFIG['pin_memory'] = env['pin_memory']
    TRAIN_CONFIG['persistent_workers'] = env['persistent_workers']
    TRAIN_CONFIG['prefetch_factor'] = env['prefetch_factor']
    TRAIN_CONFIG['use_torch_compile'] = env['use_torch_compile']
    # V12.20: Environment can override compile_mode (Colab uses 'default' to avoid
    # reduce-overhead memory leak; local uses TRAIN_CONFIG default)
    if env.get('compile_mode') is not None:
        TRAIN_CONFIG['compile_mode'] = env['compile_mode']
        print(f"[env] compile_mode overridden: '{env['compile_mode']}'")

    # Apply batch_size_multiplier when batch_size is a fixed int (not 'auto')
    if env['batch_size_multiplier'] != 1.0 and isinstance(TRAIN_CONFIG['batch_size'], int):
        original_bs = TRAIN_CONFIG['batch_size']
        TRAIN_CONFIG['batch_size'] = max(1, int(original_bs * env['batch_size_multiplier']))
        if TRAIN_CONFIG['batch_size'] != original_bs:
            print(f"[env] Batch size adjusted: {original_bs} -> {TRAIN_CONFIG['batch_size']} "
                  f"(x{env['batch_size_multiplier']})")

    # Override accumulation_steps for large-VRAM GPUs (A100: single-step with big batch)
    if env.get('accumulation_steps') is not None:
        old_accum = TRAIN_CONFIG.get('accumulation_steps', 2)
        TRAIN_CONFIG['accumulation_steps'] = env['accumulation_steps']
        if env['accumulation_steps'] != old_accum:
            print(f"[env] Accumulation steps: {old_accum} -> {env['accumulation_steps']}")

    # Override REINFORCE samples for large-VRAM GPUs (A100: more samples = better gradients)
    if env.get('n_samples_rloo') is not None:
        old_ns = TRAIN_CONFIG.get('n_samples_rloo', 2)
        TRAIN_CONFIG['n_samples_rloo'] = env['n_samples_rloo']
        if env['n_samples_rloo'] != old_ns:
            print(f"[env] REINFORCE n_samples: {old_ns} -> {env['n_samples_rloo']}")

    # Override selective backprop for large-VRAM GPUs (A100: all samples get full gradients)
    if env.get('selective_backprop') is not None:
        old_sb = TRAIN_CONFIG.get('selective_backprop', True)
        TRAIN_CONFIG['selective_backprop'] = env['selective_backprop']
        if env['selective_backprop'] != old_sb:
            print(f"[env] Selective backprop: {old_sb} -> {env['selective_backprop']}")

    # ========================================================================
    # V12.8: Apply optimizations before model creation
    # ========================================================================

    # Set matmul precision for TF32 (only on Ampere+ GPUs)
    matmul_precision = TRAIN_CONFIG.get('matmul_precision', 'highest')
    torch.set_float32_matmul_precision(matmul_precision)
    print(f"Matmul precision: {matmul_precision}")

    # Enable Flash Attention via SDPA backends
    if TRAIN_CONFIG.get('enable_flash_sdp', True):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        print("Flash Attention (SDPA): enabled")

    # Load data
    train_loader, norm_stats, magpie_dim = load_and_prepare_data()

    # Create models
    encoder, decoder = create_models(magpie_dim, device)

    # NOTE: torch.compile moved to AFTER checkpoint loading (see below)
    # This allows loading checkpoints saved without compile into models that will be compiled

    _shutdown_state['encoder'] = encoder
    _shutdown_state['decoder'] = decoder

    # V12.8: Loss function with REINFORCE support
    loss_fn = CombinedLossWithREINFORCE(
        ce_weight=TRAIN_CONFIG.get('ce_weight', 1.0),
        rl_weight=TRAIN_CONFIG.get('rl_weight', 0.0),
        tc_weight=TRAIN_CONFIG['tc_weight'],
        magpie_weight=TRAIN_CONFIG['magpie_weight'],
        stoich_weight=TRAIN_CONFIG['stoich_weight'],
        kl_weight=TRAIN_CONFIG['kl_weight'],
        entropy_weight=TRAIN_CONFIG.get('entropy_weight', 0.01),
        n_samples_rloo=TRAIN_CONFIG.get('n_samples_rloo', 2),
        temperature=TRAIN_CONFIG.get('rl_temperature', 0.8),
        use_focal_loss=True,
        focal_gamma=TRAIN_CONFIG['focal_gamma'],
        label_smoothing=TRAIN_CONFIG['label_smoothing'],
        use_autoregressive_reinforce=TRAIN_CONFIG.get('use_autoregressive_reinforce', True),
        tc_huber_delta=TRAIN_CONFIG.get('tc_huber_delta', 0.0),  # V12.20
    )

    # V12.9: Build or load draft model for speculative decoding
    draft_model = None
    use_speculative = (
        TRAIN_CONFIG.get('use_speculative_decoding', True) and
        TRAIN_CONFIG.get('rl_weight', 0.0) > 0 and
        TRAIN_CONFIG.get('use_autoregressive_reinforce', True)
    )
    if use_speculative:
        draft_model_path = PROJECT_ROOT / TRAIN_CONFIG.get('draft_model_path', 'data/processed/draft_model.pkl')
        try:
            # Load training formulas for building draft model
            import pandas as pd
            contrastive_path = PROJECT_ROOT / 'data/processed/supercon_fractions_contrastive.csv'
            data_path = contrastive_path if contrastive_path.exists() else DATA_PATH
            df = pd.read_csv(data_path)
            formulas = df['formula'].tolist()

            # Load or build draft model
            draft_model = load_or_build_draft_model(
                formulas=formulas,
                cache_path=draft_model_path,
                max_len=TRAIN_CONFIG['max_formula_len'],
            )
            print(f"Speculative Decoding: Enabled (k={TRAIN_CONFIG.get('speculative_k', 5)} tokens)")
        except Exception as e:
            print(f"Speculative Decoding: Failed to load draft model: {e}")
            print("  Falling back to standard autoregressive sampling")
            draft_model = None

    # V12.8: Wire up decoder for autoregressive REINFORCE sampling
    if TRAIN_CONFIG.get('rl_weight', 0.0) > 0 and TRAIN_CONFIG.get('use_autoregressive_reinforce', True):
        loss_fn.set_decoder(decoder, max_len=TRAIN_CONFIG['max_formula_len'], draft_model=draft_model)
        spec_str = " + speculative" if draft_model is not None else ""
        print(f"REINFORCE: Enabled with rl_weight={TRAIN_CONFIG['rl_weight']}, "
              f"n_samples={TRAIN_CONFIG.get('n_samples_rloo', 2)}, "
              f"autoregressive=True (KV-cached{spec_str})")
    elif TRAIN_CONFIG.get('rl_weight', 0.0) > 0:
        print(f"REINFORCE: Enabled with rl_weight={TRAIN_CONFIG['rl_weight']}, "
              f"n_samples={TRAIN_CONFIG.get('n_samples_rloo', 2)}, "
              f"autoregressive=False (logit sampling)")
    else:
        print("REINFORCE: Disabled (rl_weight=0)")

    # V12.12: Create contrastive loss function
    contrastive_loss_fn = None
    if TRAIN_CONFIG.get('contrastive_mode', False):
        contrastive_loss_fn = SuperconductorContrastiveLoss(
            temperature=TRAIN_CONFIG.get('contrastive_temperature', 0.07),
        ).to(device)
        print(f"Contrastive Loss: Enabled (weight={TRAIN_CONFIG['contrastive_weight']}, "
              f"temp={TRAIN_CONFIG['contrastive_temperature']}, "
              f"warmup={TRAIN_CONFIG['contrastive_warmup_epochs']} epochs)")

    # V12.16: Create consistency and theory loss functions
    consistency_loss_fn = None
    if TRAIN_CONFIG.get('use_consistency_loss', False):
        consistency_config = ConsistencyLossConfig(
            tc_weight=TRAIN_CONFIG.get('consistency_tc_weight', 1.0),
            magpie_weight=TRAIN_CONFIG.get('consistency_magpie_weight', 0.1),
        )
        consistency_loss_fn = CombinedConsistencyLoss(
            config=consistency_config,
            use_bidirectional=False,  # Start without bidirectional (simpler)
        )
        print(f"Consistency Loss: Enabled (weight={TRAIN_CONFIG['consistency_weight']})")

    theory_loss_fn = None
    family_classifier = None
    if TRAIN_CONFIG.get('use_theory_loss', False):
        theory_config = TheoryLossConfig(
            theory_weight=TRAIN_CONFIG.get('theory_weight', 0.05),
            use_soft_constraints=TRAIN_CONFIG.get('theory_use_soft_constraints', True),
            tc_log_transform=TRAIN_CONFIG.get('tc_log_transform', False),  # V12.22
        )
        theory_loss_fn = TheoryRegularizationLoss(config=theory_config).to(device)
        family_classifier = RuleBasedFamilyClassifier()
        n_theory_params = sum(p.numel() for p in theory_loss_fn.parameters())
        print(f"Theory Loss: Enabled (weight={TRAIN_CONFIG['theory_weight']}, "
              f"soft_constraints={TRAIN_CONFIG.get('theory_use_soft_constraints', True)}, "
              f"warmup={TRAIN_CONFIG.get('theory_warmup_epochs', 50)} epochs, "
              f"params={n_theory_params:,})")

    # V12.9: Initialize entropy manager for REINFORCE (prevents entropy collapse)
    entropy_manager = None
    if TRAIN_CONFIG.get('rl_weight', 0.0) > 0:
        entropy_config = EntropyConfig(
            strategy=TRAIN_CONFIG.get('entropy_strategy', 'causal'),
            target_entropy=TRAIN_CONFIG.get('entropy_target', 0.5),
            min_entropy=TRAIN_CONFIG.get('entropy_min', 0.1),
            entropy_weight_base=TRAIN_CONFIG.get('entropy_weight', 0.2),
            entropy_weight_min=TRAIN_CONFIG.get('entropy_weight_min', 0.05),
            entropy_weight_max=TRAIN_CONFIG.get('entropy_weight_max', 1.0),
            plateau_window=TRAIN_CONFIG.get('entropy_plateau_window', 10),
            plateau_threshold=TRAIN_CONFIG.get('entropy_plateau_threshold', 0.01),
            plateau_relative=TRAIN_CONFIG.get('entropy_plateau_relative', True),
            temperature_base=TRAIN_CONFIG.get('rl_temperature', 0.8),
        )
        entropy_manager = EntropyManager(
            strategy=TRAIN_CONFIG.get('entropy_strategy', 'causal'),
            config=entropy_config,
            max_len=TRAIN_CONFIG['max_formula_len'],
        )
        _shutdown_state['entropy_manager'] = entropy_manager  # For graceful shutdown
        strategy_name = TRAIN_CONFIG.get('entropy_strategy', 'causal')
        if strategy_name == 'causal':
            print(f"Entropy Maintenance: {strategy_name} strategy (diagnoses plateau cause before boosting)")
            print(f"  Tiered response: strong=2x, weak=1.3x, poor_history=1.1x")
        else:
            print(f"Entropy Maintenance: {strategy_name} strategy")
        print(f"  (target={entropy_config.target_entropy}, min={entropy_config.min_entropy})")

    # Focal loss kept for backward compatibility (now used inside CombinedLossWithREINFORCE)
    focal_loss_fn = FocalLossWithLabelSmoothing(
        gamma=TRAIN_CONFIG['focal_gamma'],
        smoothing=TRAIN_CONFIG['label_smoothing'],
        ignore_index=PAD_IDX
    )

    # V12.22: Include theory loss learnable params (BCS predictors, cuprate predictors)
    # in encoder optimizer so they receive gradients during training
    import itertools as _itertools
    if theory_loss_fn is not None:
        enc_params = list(_itertools.chain(encoder.parameters(), theory_loss_fn.parameters()))
    else:
        enc_params = list(encoder.parameters())
    enc_opt = torch.optim.AdamW(enc_params, lr=TRAIN_CONFIG['learning_rate'])
    dec_opt = torch.optim.AdamW(decoder.parameters(), lr=TRAIN_CONFIG['learning_rate'])

    # V12.8: Learning rate schedulers (configurable)
    lr_scheduler_type = TRAIN_CONFIG.get('lr_scheduler', 'cosine')
    lr_min = TRAIN_CONFIG['learning_rate'] * TRAIN_CONFIG.get('lr_min_factor', 0.01)

    if lr_scheduler_type == 'cosine_warm_restarts':
        # Warm restarts help break plateaus
        T_0 = TRAIN_CONFIG.get('lr_restart_period', 100)
        T_mult = TRAIN_CONFIG.get('lr_restart_mult', 2)
        enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            enc_opt, T_0=T_0, T_mult=T_mult, eta_min=lr_min
        )
        dec_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            dec_opt, T_0=T_0, T_mult=T_mult, eta_min=lr_min
        )
        print(f"LR Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")
    elif lr_scheduler_type == 'one_cycle':
        # OneCycleLR needs steps_per_epoch
        steps_per_epoch = len(train_loader) // TRAIN_CONFIG.get('accumulation_steps', 1)
        enc_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            enc_opt,
            max_lr=TRAIN_CONFIG['learning_rate'] * 10,
            epochs=TRAIN_CONFIG['num_epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        dec_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            dec_opt,
            max_lr=TRAIN_CONFIG['learning_rate'] * 10,
            epochs=TRAIN_CONFIG['num_epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        print(f"LR Scheduler: OneCycleLR (max_lr={TRAIN_CONFIG['learning_rate'] * 10})")
    else:  # Default: cosine
        enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            enc_opt, T_max=TRAIN_CONFIG['num_epochs'], eta_min=lr_min
        )
        dec_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            dec_opt, T_max=TRAIN_CONFIG['num_epochs'], eta_min=lr_min
        )
        print(f"LR Scheduler: CosineAnnealing (T_max={TRAIN_CONFIG['num_epochs']})")

    # V12.10: Store optimizer/scheduler in shutdown state for graceful interrupt
    _shutdown_state['enc_opt'] = enc_opt
    _shutdown_state['dec_opt'] = dec_opt
    _shutdown_state['enc_scheduler'] = enc_scheduler
    _shutdown_state['dec_scheduler'] = dec_scheduler
    _shutdown_state['theory_loss_fn'] = theory_loss_fn  # V12.22

    # V12.8: AMP dtype configuration with auto-detection
    # bfloat16 requires compute capability 8.0+ (Ampere: RTX 30xx, A100, etc.)
    # Older GPUs (GTX 1080 Ti, RTX 20xx, etc.) only support float16
    amp_dtype_str = TRAIN_CONFIG.get('amp_dtype', 'auto')
    if amp_dtype_str == 'auto':
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere or newer
                amp_dtype_str = 'bfloat16'
                print(f"GPU compute capability {capability[0]}.{capability[1]} - using bfloat16")
            else:
                amp_dtype_str = 'float16'
                print(f"GPU compute capability {capability[0]}.{capability[1]} - using float16 (bfloat16 not supported)")
        else:
            amp_dtype_str = 'float16'
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16

    # AMP scaler (updated API for PyTorch 2.4+)
    # Note: GradScaler may not be needed for bfloat16 (no underflow issues)
    use_scaler = TRAIN_CONFIG['use_amp'] and amp_dtype_str != 'bfloat16'
    # GradScaler - don't pass device arg for compatibility with older PyTorch
    # (Old PyTorch interprets 'cuda' as init_scale, causing errors)
    scaler = GradScaler(enabled=use_scaler)
    print(f"AMP: dtype={amp_dtype_str}, scaler={'enabled' if use_scaler else 'disabled'}")

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    print(f"Epochs: {TRAIN_CONFIG['num_epochs']}")
    actual_batch_size = train_loader.batch_size
    print(f"Batch size: {actual_batch_size} (effective: {actual_batch_size * TRAIN_CONFIG.get('accumulation_steps', 1)})")
    print(f"Accumulation steps: {TRAIN_CONFIG.get('accumulation_steps', 1)}")
    print(f"Workers: {TRAIN_CONFIG['num_workers']}")

    # V12.8: Print optimization status
    print(f"\nV12.8 Optimizations:")
    print(f"  AMP dtype: {amp_dtype_str}")
    print(f"  Matmul precision: {TRAIN_CONFIG.get('matmul_precision', 'highest')}")
    print(f"  Flash SDPA: {TRAIN_CONFIG.get('enable_flash_sdp', True)}")
    print(f"  torch.compile: {TRAIN_CONFIG.get('use_torch_compile', False)}")
    print(f"  Gradient checkpointing: {TRAIN_CONFIG.get('use_gradient_checkpointing', False)}")
    print(f"  Contrastive mode: {TRAIN_CONFIG.get('contrastive_mode', False)}")
    print(f"  Selective backprop: {TRAIN_CONFIG.get('selective_backprop', False)} "
          f"(threshold={TRAIN_CONFIG.get('selective_backprop_threshold', 0.33)})")
    print(f"  Z-caching: {TRAIN_CONFIG.get('cache_z_vectors', False)} "
          f"(interval={TRAIN_CONFIG.get('z_cache_interval', 0)}, "
          f"path={TRAIN_CONFIG.get('z_cache_path', 'outputs/latent_cache.pt')})")
    print(f"\nLoss weights (final):")
    print(f"  Formula: {TRAIN_CONFIG['formula_weight']}")
    print(f"  Tc: {TRAIN_CONFIG['tc_weight']}" +
          (f" (Huber δ={TRAIN_CONFIG['tc_huber_delta']})" if TRAIN_CONFIG.get('tc_huber_delta', 0) > 0 else " (MSE)") +
          (" [log1p]" if TRAIN_CONFIG.get('tc_log_transform', False) else ""))
    print(f"  Magpie: {TRAIN_CONFIG['magpie_weight']}")
    print(f"  Stoich: {TRAIN_CONFIG['stoich_weight']}")
    print(f"  HP: {TRAIN_CONFIG.get('hp_loss_weight', 0)}")
    print(f"  SC: {TRAIN_CONFIG.get('sc_loss_weight', 0)}")
    if TRAIN_CONFIG.get('use_theory_loss', False):
        print(f"  Theory: {TRAIN_CONFIG.get('theory_weight', 0)} "
              f"(warmup={TRAIN_CONFIG.get('theory_warmup_epochs', 50)} epochs)")
    print(f"\nCurriculum:")
    print(f"  Phase 1 (0-{TRAIN_CONFIG['curriculum_phase1_end']}): Tc 5→10, Magpie 1→2")
    print(f"  Phase 2 ({TRAIN_CONFIG['curriculum_phase1_end']}+): Full strength")
    print(f"\nFocal Loss: gamma={TRAIN_CONFIG['focal_gamma']}, smoothing={TRAIN_CONFIG['label_smoothing']}")
    print(f"\nTeacher Forcing: Adaptive (OPTIMIZED)")
    print(f"  TF = 1.0 - exact_match (no floor, can reach 0)")
    print(f"  V12.6: Uses 2-pass parallel approach instead of 60-pass sequential")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_exact = 0
    prev_exact = 0  # Track previous epoch's exact match for adaptive TF
    prev_entropy = 0.5  # Track previous epoch's entropy for entropy manager (V12.9)

    if TRAIN_CONFIG.get('resume_checkpoint'):
        checkpoint_path = PROJECT_ROOT / TRAIN_CONFIG['resume_checkpoint']
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            # V12.10: Load full training state including optimizer/scheduler
            resume_state = load_checkpoint(
                encoder, decoder, checkpoint_path,
                entropy_manager=entropy_manager,
                enc_opt=enc_opt, dec_opt=dec_opt,
                enc_scheduler=enc_scheduler, dec_scheduler=dec_scheduler,
                theory_loss_fn=theory_loss_fn,  # V12.22
            )
            start_epoch = resume_state['start_epoch']
            prev_exact = resume_state['prev_exact']
            best_exact = resume_state['best_exact']
            print(f"  Starting from epoch {start_epoch}")

            # V12.12: Reset drop detector when retraining on new data
            # When the training data changes (e.g., combined dataset), normalization
            # stats shift and the model will see an initial performance drop. Without
            # resetting, the catastrophic drop detector would trigger rollback loops.
            if TRAIN_CONFIG.get('retrain_new_data', False):
                print(f"  [RETRAIN MODE] Resetting drop detector (prev_exact was {prev_exact:.3f})")
                prev_exact = 0.0
                best_exact = 0.0
        else:
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
            print("  Starting from scratch...")

    # V12.10: torch.compile AFTER checkpoint loading (allows loading non-compiled checkpoints)
    # V12.11: Auto-disable on older GPUs (Triton requires compute capability 7.0+)
    use_compile = TRAIN_CONFIG.get('use_torch_compile', False)
    if use_compile and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 7:
            print(f"\ntorch.compile disabled: GPU compute capability {capability[0]}.{capability[1]} < 7.0 (Triton requirement)")
            use_compile = False

    if use_compile:
        compile_mode = TRAIN_CONFIG.get('compile_mode', 'reduce-overhead')
        print(f"\nCompiling models with mode='{compile_mode}'...")
        encoder = torch.compile(encoder, mode=compile_mode)
        # Compile decoder's transformer, not the whole decoder (KV cache needs dynamic shapes)
        decoder.transformer_decoder = torch.compile(
            decoder.transformer_decoder, mode=compile_mode
        )
        # Update references
        _shutdown_state['encoder'] = encoder
        # V12.15: Preserve draft_model when updating decoder reference after compile
        loss_fn.set_decoder(decoder, max_len=TRAIN_CONFIG['max_formula_len'], draft_model=draft_model)
        print("  Encoder and decoder.transformer_decoder compiled")

        # Warmup pass: initialize CUDA graphs with a real batch to prevent NaN on first epoch
        # reduce-overhead mode uses CUDA graphs which need stable tensor shapes/states
        # Without warmup, the first epoch after checkpoint load + compile produces all-NaN
        print("  Running warmup pass to initialize CUDA graphs...", flush=True)
        encoder.eval()
        decoder.eval()
        warmup_batch = next(iter(train_loader))
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
            wb = [b.to(device) for b in warmup_batch]
            elem_idx, elem_frac, elem_mask, tokens, tc, magpie = wb[0], wb[1], wb[2], wb[3], wb[4], wb[5]
            enc_out = encoder(elem_idx, elem_frac, elem_mask, magpie, tc)
            _ = decoder(enc_out['z'], tokens, encoder_skip=enc_out['attended_input'])
        torch.cuda.synchronize()
        encoder.train()
        decoder.train()
        print("  Warmup complete", flush=True)

    # V12.11: Rollback loop detection
    rollback_count = 0
    max_rollbacks = 3  # Stop training if we hit this many rollbacks
    last_rollback_epoch = -100  # Track when last rollback occurred

    for epoch in range(start_epoch, TRAIN_CONFIG['num_epochs']):
        _shutdown_state['epoch'] = epoch

        if _shutdown_state['should_stop']:
            break

        # Get curriculum weights for this epoch
        tc_weight, magpie_weight = get_curriculum_weights(epoch)

        # V12.22: TF locked at 1.0 — feeding wrong predictions as context trains on
        # corrupted input distribution, slowing convergence. Modern LLMs all use TF=1.0.
        # The 83%→55% teacher-forced vs autoregressive gap is better closed via REINFORCE.
        tf_ratio = 1.0

        # V12.9: Update entropy weight and temperature from entropy manager
        if entropy_manager is not None and TRAIN_CONFIG.get('rl_weight', 0.0) > 0:
            # Get dynamic entropy weight and temperature based on previous epoch's entropy
            new_entropy_weight = entropy_manager.get_entropy_weight(epoch, prev_entropy)
            new_temperature = entropy_manager.get_temperature(epoch, prev_entropy)

            # Update loss function parameters
            loss_fn.entropy_weight = new_entropy_weight
            loss_fn.temperature = new_temperature

        # V12.12: Compute contrastive weight with warmup
        effective_contrastive_weight = 0.0
        if contrastive_loss_fn is not None:
            warmup_epochs = TRAIN_CONFIG.get('contrastive_warmup_epochs', 100)
            warmup_progress = min(1.0, epoch / max(warmup_epochs, 1))
            effective_contrastive_weight = TRAIN_CONFIG['contrastive_weight'] * warmup_progress

        # V12.22: Compute theory weight with warmup (like contrastive warmup)
        effective_theory_weight = 0.0
        if theory_loss_fn is not None:
            theory_warmup = TRAIN_CONFIG.get('theory_warmup_epochs', 50)
            theory_warmup_prog = min(1.0, epoch / max(theory_warmup, 1))
            effective_theory_weight = TRAIN_CONFIG['theory_weight'] * theory_warmup_prog

        metrics = train_epoch(
            encoder, decoder, train_loader, loss_fn,
            enc_opt, dec_opt, scaler, device, TRAIN_CONFIG['use_amp'],
            tf_ratio=tf_ratio, tc_weight=tc_weight, magpie_weight=magpie_weight,
            focal_loss_fn=focal_loss_fn,
            amp_dtype=amp_dtype,
            accumulation_steps=TRAIN_CONFIG.get('accumulation_steps', 1),
            # V12.12: Contrastive learning
            contrastive_loss_fn=contrastive_loss_fn,
            contrastive_weight=effective_contrastive_weight,
            non_sc_formula_weight=TRAIN_CONFIG.get('non_sc_formula_weight', 0.5),
            # V12.12: Selective backpropagation
            selective_backprop=TRAIN_CONFIG.get('selective_backprop', False),
            selective_backprop_threshold=TRAIN_CONFIG.get('selective_backprop_threshold', 0.33),
            # V12.15: Timing instrumentation
            enable_timing=TRAIN_CONFIG.get('enable_timing', True),
            # V12.19: High-pressure prediction
            hp_loss_weight=TRAIN_CONFIG.get('hp_loss_weight', 0.0),
            # V12.21: SC/non-SC classification
            sc_loss_weight=TRAIN_CONFIG.get('sc_loss_weight', 0.0),
            # V12.22: Theory-guided consistency
            theory_loss_fn=theory_loss_fn,
            theory_weight=effective_theory_weight,
            norm_stats=norm_stats,
        )

        # V12.11: Catastrophic drop detection - rollback to best checkpoint and halve LR
        # This protects against scheduler bugs, gradient explosions, or other instabilities
        current_exact = metrics['exact_match']
        best_checkpoint_path = OUTPUT_DIR / 'checkpoint_best.pt'
        rolled_back = False

        # Reset rollback counter if training has been stable for 50+ epochs
        if epoch - last_rollback_epoch > 50:
            rollback_count = 0

        if prev_exact > 0.1 and (prev_exact - current_exact) > 0.05:
            drop_pct = (prev_exact - current_exact) * 100
            rollback_count += 1
            last_rollback_epoch = epoch
            print(f"  [SAFETY] Catastrophic drop detected ({drop_pct:.1f}%) - rollback #{rollback_count}", flush=True)

            # Check for rollback loop
            if rollback_count >= max_rollbacks:
                print(f"\n[SAFETY] ERROR: Hit {max_rollbacks} rollbacks - something is fundamentally wrong!", flush=True)
                print(f"[SAFETY] Saving emergency checkpoint and stopping training.", flush=True)
                save_checkpoint(encoder, decoder, epoch, 'emergency',
                               entropy_manager=entropy_manager,
                               enc_opt=enc_opt, dec_opt=dec_opt,
                               enc_scheduler=enc_scheduler, dec_scheduler=dec_scheduler,
                               prev_exact=prev_exact, best_exact=best_exact,
                               theory_loss_fn=theory_loss_fn)
                raise RuntimeError(f"Training stopped: {max_rollbacks} consecutive rollbacks detected. "
                                   f"Check data, model architecture, or hyperparameters.")

            # Halve LR to prevent recurrence
            for opt in [enc_opt, dec_opt]:
                for param_group in opt.param_groups:
                    param_group['lr'] *= 0.5
            new_lr = enc_opt.param_groups[0]['lr']
            print(f"  [SAFETY] LR halved → {new_lr:.2e}", flush=True)

            # Rollback to best checkpoint if available
            if best_checkpoint_path.exists():
                print(f"  [SAFETY] Rolling back to checkpoint_best.pt...", flush=True)
                rollback_state = load_checkpoint(
                    encoder, decoder, best_checkpoint_path,
                    entropy_manager=entropy_manager,
                    # Don't restore optimizers/schedulers - keep current (halved) LR
                )
                prev_exact = rollback_state['prev_exact']
                best_exact = rollback_state['best_exact']
                _shutdown_state['prev_exact'] = prev_exact
                _shutdown_state['best_exact'] = best_exact
                rolled_back = True
                print(f"  [SAFETY] Rolled back to epoch {rollback_state['start_epoch']-1} "
                      f"(exact={prev_exact*100:.1f}%)", flush=True)
            else:
                print(f"  [SAFETY] No checkpoint_best.pt found, continuing with halved LR", flush=True)

        # Update prev_exact for adaptive teacher forcing (skip if we just rolled back)
        # PyTorch 2.4+ should fix the CUDA bug that required keeping TF=1.0
        if not rolled_back:
            prev_exact = metrics['exact_match']
        _shutdown_state['prev_exact'] = prev_exact  # V12.10: Keep shutdown state current

        # V12.9: Update entropy manager with this epoch's metrics
        if entropy_manager is not None and TRAIN_CONFIG.get('rl_weight', 0.0) > 0:
            current_entropy = metrics.get('entropy', 0.5)
            prev_entropy = current_entropy  # Store for next epoch

            entropy_manager.update(
                epoch=epoch,
                entropy=current_entropy,
                reward=metrics.get('mean_reward'),
                exact_match=metrics['exact_match'],
            )

        # Learning rate scheduling
        enc_scheduler.step()
        dec_scheduler.step()

        # Print progress with all metrics including TF ratio and curriculum weights
        base_msg = (f"Epoch {epoch:4d} | TF: {tf_ratio:.2f} | Loss: {metrics['loss']:.4f} | "
                    f"Acc: {metrics['accuracy']*100:.1f}% | Exact: {metrics['exact_match']*100:.1f}% | "
                    f"Tc: {metrics['tc_loss']:.4f} | Magpie: {metrics['magpie_loss']:.4f} | "
                    f"Stoich: {metrics['stoich_loss']:.4f} | "
                    f"zN: {metrics['z_norm']:.1f}")

        # V12.8: Add REINFORCE metrics if enabled (includes entropy in reward)
        # V12.9: Also show adaptive entropy weight
        if TRAIN_CONFIG.get('rl_weight', 0.0) > 0:
            ent_w = loss_fn.entropy_weight if entropy_manager else TRAIN_CONFIG.get('entropy_weight', 0.2)
            base_msg += (f" | RL: {metrics['reinforce_loss']:.4f} | "
                        f"Reward: {metrics['mean_reward']:.3f} | "
                        f"Ent: {metrics.get('entropy', 0):.2f} | "
                        f"EntW: {ent_w:.3f}")

        # V12.12: Add contrastive metrics if enabled
        if contrastive_loss_fn is not None:
            base_msg += f" | Con: {metrics.get('contrastive_loss', 0):.4f} (w={effective_contrastive_weight:.3f})"
            if 'sc_exact_match' in metrics:
                base_msg += f" | SC: {metrics['sc_exact_match']*100:.1f}%"
            if 'non_sc_exact_match' in metrics:
                base_msg += f" | nSC: {metrics['non_sc_exact_match']*100:.1f}%"

        # V12.19: Show HP loss if enabled
        if TRAIN_CONFIG.get('hp_loss_weight', 0) > 0 and metrics.get('hp_loss', 0) > 0:
            base_msg += f" | HP: {metrics['hp_loss']:.4f}"

        # V12.21: Show SC classification loss if enabled
        if TRAIN_CONFIG.get('sc_loss_weight', 0) > 0 and metrics.get('sc_loss', 0) > 0:
            base_msg += f" | SCL: {metrics['sc_loss']:.4f}"

        # V12.22: Show theory loss if enabled
        if theory_loss_fn is not None and metrics.get('theory_loss', 0) > 0:
            base_msg += f" | Thry: {metrics['theory_loss']:.4f} (w={effective_theory_weight:.3f})"

        # V12.12: Show selective backprop stats
        if metrics.get('n_skipped', 0) > 0 and metrics.get('n_total_batches', 0) > 0:
            skip_pct = metrics['n_skipped'] / metrics['n_total_batches'] * 100
            base_msg += f" | Skip: {metrics['n_skipped']}/{metrics['n_total_batches']} ({skip_pct:.0f}%)"
        elif metrics.get('n_total_batches', 0) == 0:
            base_msg += f" | Skip: ALL ({metrics.get('n_skipped', 0)} NaN batches)"

        # V12.9: Show speculative decoding stats if enabled
        if 'spec_acceptance_rate' in metrics:
            base_msg += f" | Spec: {metrics['spec_acceptance_rate']*100:.0f}% ({metrics['spec_tokens_per_step']:.1f}t/s)"

        # V12.15: Show timing breakdown
        if 'timing' in metrics and metrics['timing'] is not None:
            epoch_time = metrics.get('epoch_time', 0)
            timing_summary = metrics['timing'].format_summary(epoch_time)
            base_msg += f" | Time: {epoch_time:.1f}s [{timing_summary}]"

        print(base_msg, flush=True)

        # V12.15: Print detailed timing every 50 epochs
        if epoch % 50 == 0 and 'timing' in metrics and metrics['timing'] is not None:
            print(metrics['timing'].format_detailed())

        # V12.18: Evaluate TRUE autoregressive exact match EVERY epoch
        # with full Z-space diagnostics, Tc/Magpie/stoich reconstruction stats
        # V12.15: Log errors to file for analysis
        true_eval = None
        if epoch % 1 == 0:
            true_eval = evaluate_true_autoregressive(
                encoder, decoder, train_loader, device, max_samples=2000,
                log_errors=True, epoch=epoch
            )
            err_dist = true_eval['error_distribution']
            print(f"  → TRUE Autoregressive: {true_eval['true_exact_match']*100:.1f}% exact "
                  f"({true_eval['exact_count']}/{true_eval['total_samples']}) | "
                  f"Errors: 0={err_dist[0]}, 1={err_dist[1]}, 2={err_dist[2]}, 3={err_dist[3]}, >3={err_dist['more']}", flush=True)

        # V12.21: Add entropy_weight and tf_ratio to metrics for CSV logging
        if entropy_manager is not None:
            metrics['entropy_weight'] = loss_fn.entropy_weight
        metrics['tf_ratio'] = tf_ratio

        # V12.17: Log metrics to CSV for post-training analysis
        training_log_path = OUTPUT_DIR / 'training_log.csv'
        log_training_metrics(epoch, metrics, training_log_path, true_eval=true_eval)

        # Save checkpoints (V12.10: include full training state)
        checkpoint_kwargs = dict(
            entropy_manager=entropy_manager,
            enc_opt=enc_opt, dec_opt=dec_opt,
            enc_scheduler=enc_scheduler, dec_scheduler=dec_scheduler,
            prev_exact=metrics['exact_match'], best_exact=best_exact,
            theory_loss_fn=theory_loss_fn,  # V12.22
        )

        if metrics['exact_match'] > best_exact:
            best_exact = metrics['exact_match']
            _shutdown_state['best_exact'] = best_exact  # V12.10: Keep shutdown state current
            checkpoint_kwargs['best_exact'] = best_exact  # Update with new best
            save_checkpoint(encoder, decoder, epoch, 'best', **checkpoint_kwargs)

            # V12.17: Cache z-vectors on new best checkpoint
            if TRAIN_CONFIG.get('cache_z_vectors', False):
                z_cache_path = PROJECT_ROOT / TRAIN_CONFIG.get('z_cache_path', 'outputs/latent_cache.pt')
                z_cache_mode = TRAIN_CONFIG.get('z_cache_mode', 'z_only')
                cache_z_vectors(encoder, train_loader, device, epoch, z_cache_path,
                               dataset_info=norm_stats, decoder=decoder, mode=z_cache_mode)

        # V12.17: Cache z-vectors at configured interval (or every epoch if configured)
        z_cache_interval = TRAIN_CONFIG.get('z_cache_interval', 0)
        cache_every_epoch = TRAIN_CONFIG.get('z_cache_every_epoch', False)
        should_cache_interval = (z_cache_interval > 0 and (epoch + 1) % z_cache_interval == 0)

        if TRAIN_CONFIG.get('cache_z_vectors', False) and (cache_every_epoch or should_cache_interval):
            z_cache_path = PROJECT_ROOT / TRAIN_CONFIG.get('z_cache_path', 'outputs/latent_cache.pt')
            z_cache_mode = TRAIN_CONFIG.get('z_cache_mode', 'z_only')
            # Save with epoch suffix to track evolution
            epoch_cache_path = z_cache_path.with_stem(f"{z_cache_path.stem}_epoch{epoch:04d}")
            cache_z_vectors(encoder, train_loader, device, epoch, epoch_cache_path,
                           dataset_info=norm_stats, decoder=decoder, mode=z_cache_mode)

        if (epoch + 1) % TRAIN_CONFIG['checkpoint_interval'] == 0:
            save_checkpoint(encoder, decoder, epoch, **checkpoint_kwargs)

    # Final save
    save_checkpoint(encoder, decoder, epoch, 'final', **checkpoint_kwargs)

    # V12.17: Final z-cache
    if TRAIN_CONFIG.get('cache_z_vectors', False):
        z_cache_path = PROJECT_ROOT / TRAIN_CONFIG.get('z_cache_path', 'outputs/latent_cache.pt')
        z_cache_mode = TRAIN_CONFIG.get('z_cache_mode', 'z_only')
        final_cache_path = z_cache_path.with_stem(f"{z_cache_path.stem}_final")
        cache_z_vectors(encoder, train_loader, device, epoch, final_cache_path,
                       dataset_info=norm_stats, decoder=decoder, mode=z_cache_mode)

    print(f"\nTraining complete. Best exact match: {best_exact*100:.1f}%")


if __name__ == '__main__':
    train()
