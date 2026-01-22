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
import signal
import re
import numpy as np
import pandas as pd
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

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, tokenize_formula, tokens_to_indices,
    VOCAB_SIZE, PAD_IDX, START_IDX, END_IDX, IDX_TO_TOKEN
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
    'num_epochs': 2000,
    'learning_rate': 3e-5,      # Reduced for stable fine-tuning (was 1e-4)
    'max_formula_len': 60,
    'checkpoint_interval': 50,

    # =========================================================================
    # BATCH SIZE CONFIGURATION
    # =========================================================================
    # Set to 'auto' to automatically scale based on GPU memory:
    #   - 8GB  (RTX 4060):      batch_size=32
    #   - 11GB (GTX 1080 Ti):   batch_size=48
    #   - 16GB (V100 16GB):     batch_size=64
    #   - 24GB (RTX 3090/4090): batch_size=96
    #   - 40GB+ (A100):         batch_size=128
    #
    # Effective batch size = batch_size * accumulation_steps
    # Larger effective batch = smoother gradients, but may need LR scaling
    # =========================================================================
    'batch_size': 'auto',       # 'auto' scales with GPU memory, or set fixed value (32, 48, etc.)
    'accumulation_steps': 1,    # Gradient accumulation (increase for larger effective batch)

    # V12.8: Data Loading Optimizations
    'num_workers': 4,           # Parallel data loading (set to 0 if WSL2 CUDA issues)
    'pin_memory': True,
    'prefetch_factor': 2,       # Prefetch 2 batches per worker
    'persistent_workers': True, # Keep workers alive between epochs

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
    'use_gradient_checkpointing': False,  # Trade compute for memory (enables larger batch)
    'enable_flash_sdp': True,   # Enable Flash Attention via SDPA

    # Loss weights (final values - curriculum ramps up to these)
    'formula_weight': 1.0,
    'tc_weight': 10.0,
    'magpie_weight': 2.0,
    'stoich_weight': 2.0,  # V12.4: Stoichiometry loss weight
    'kl_weight': 0.0001,

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
    'rl_weight': 1.0,            # REINFORCE weight (0=disabled, 1.0-2.5=typical)
    'ce_weight': 1.0,            # Cross-entropy weight (keep at 1.0)
    'n_samples_rloo': 2,         # Number of samples for RLOO baseline (2-4)
    'rl_temperature': 0.8,       # Sampling temperature for REINFORCE
    'entropy_weight': 0.2,       # Entropy bonus in REINFORCE reward (encourages exploration)
    'use_autoregressive_reinforce': True,  # Use KV-cached autoregressive sampling (recommended)

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
}

DATA_PATH = PROJECT_ROOT / 'data/processed/supercon_fractions.csv'
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
    'prev_exact': 0, 'best_exact': 0
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
            best_exact=_shutdown_state.get('best_exact', 0)
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


def load_holdout_indices(holdout_path: Path) -> set:
    """Load holdout sample indices."""
    with open(holdout_path, 'r') as f:
        data = json.load(f)
    return {s['original_index'] for s in data['holdout_samples']}


def load_and_prepare_data():
    """Load and prepare full materials dataset."""
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples")

    formulas = df['formula'].tolist()
    tc_values = df['Tc'].values

    # Normalize Tc
    tc_mean, tc_std = tc_values.mean(), tc_values.std()
    tc_normalized = (tc_values - tc_mean) / tc_std
    print(f"Tc: mean={tc_mean:.2f}K, std={tc_std:.2f}K")

    # Get Magpie features
    exclude = ['formula', 'Tc', 'composition', 'category', 'compound possible', 'formula_original']
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    magpie_cols = [c for c in numeric_cols if c not in exclude]
    print(f"Found {len(magpie_cols)} Magpie features")

    magpie_data = df[magpie_cols].values.astype(np.float32)

    # Handle NaN
    nan_mask = np.isnan(magpie_data)
    if nan_mask.any():
        col_means = np.nanmean(magpie_data, axis=0)
        for col_idx in range(magpie_data.shape[1]):
            magpie_data[nan_mask[:, col_idx], col_idx] = col_means[col_idx]

    magpie_mean = magpie_data.mean(axis=0)
    magpie_std = magpie_data.std(axis=0) + 1e-8
    magpie_normalized = (magpie_data - magpie_mean) / magpie_std

    # Tokenize formulas
    print("Tokenizing formulas...")
    max_len = TRAIN_CONFIG['max_formula_len']
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

    # Create dataset
    dataset = TensorDataset(
        element_indices, element_fractions, element_mask,
        formula_tokens, tc_tensor, magpie_tensor
    )

    # Get train indices (exclude holdout)
    holdout_indices = load_holdout_indices(HOLDOUT_PATH)
    all_indices = set(range(len(formulas)))
    train_indices = sorted(all_indices - holdout_indices)

    print(f"Training samples: {len(train_indices)}")
    print(f"Holdout samples: {len(holdout_indices)}")

    train_dataset = Subset(dataset, train_indices)

    # V12.11: Auto batch size based on GPU memory
    batch_size = TRAIN_CONFIG['batch_size']
    if batch_size == 'auto':
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Scale batch size with GPU memory (conservative estimates)
            if gpu_mem_gb >= 40:
                batch_size = 128
            elif gpu_mem_gb >= 24:
                batch_size = 96
            elif gpu_mem_gb >= 16:
                batch_size = 64
            elif gpu_mem_gb >= 11:
                batch_size = 48
            else:
                batch_size = 32
            print(f"Auto batch size: {batch_size} (GPU memory: {gpu_mem_gb:.1f}GB)")
        else:
            batch_size = 16  # CPU fallback
            print(f"Auto batch size: {batch_size} (CPU mode)")

    # Create DataLoader with V12.8 optimizations
    use_workers = TRAIN_CONFIG['num_workers']
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': use_workers,
    }
    if use_workers > 0:
        loader_kwargs['pin_memory'] = TRAIN_CONFIG['pin_memory']
        loader_kwargs['prefetch_factor'] = TRAIN_CONFIG['prefetch_factor']
        loader_kwargs['persistent_workers'] = TRAIN_CONFIG.get('persistent_workers', True)
    else:
        loader_kwargs['pin_memory'] = False  # Not useful without workers

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    print(f"DataLoader: workers={use_workers}, pin_memory={loader_kwargs.get('pin_memory', False)}")

    norm_stats = {
        'tc_mean': tc_mean, 'tc_std': tc_std,
        'magpie_mean': magpie_mean.tolist(),
        'magpie_std': magpie_std.tolist(),
    }

    return train_loader, norm_stats, len(magpie_cols)


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
    2. Tc reconstruction (MSE)
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

    def set_decoder(self, decoder, max_len: int = 60):
        """Set decoder for autoregressive REINFORCE sampling (V12.8)."""
        self._decoder = decoder
        self._max_len = max_len

    def compute_rloo_autoregressive(
        self,
        z: torch.Tensor,
        targets: torch.Tensor,
        encoder_skip: torch.Tensor = None,
        stoich_pred: torch.Tensor = None,
    ):
        """
        V12.8: Compute RLOO advantages using autoregressive sampling with KV cache.

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

        # Generate ALL samples in one batched call (2x faster than sequential!)
        sampled_tokens, log_probs, entropy, mask = self._decoder.sample_for_reinforce(
            z=z_expanded,
            encoder_skip=encoder_skip_expanded,
            stoich_pred=stoich_pred_expanded,
            temperature=self.temperature,
            max_len=self._max_len,
        )

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

        # 4. Tc Loss
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

        return {
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


# Backward compatibility alias
CombinedLoss = CombinedLossWithREINFORCE


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(encoder, decoder, epoch, suffix='', entropy_manager=None,
                    enc_opt=None, dec_opt=None, enc_scheduler=None, dec_scheduler=None,
                    prev_exact=None, best_exact=None):
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

    # V12.10: Save training state variables
    if prev_exact is not None:
        checkpoint_data['prev_exact'] = prev_exact
    if best_exact is not None:
        checkpoint_data['best_exact'] = best_exact

    torch.save(checkpoint_data, path)
    print(f"  Saved checkpoint: {path.name}", flush=True)


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
                    enc_opt=None, dec_opt=None, enc_scheduler=None, dec_scheduler=None):
    """Load model checkpoint with full training state for proper resumption.

    Returns:
        dict with keys: 'start_epoch', 'prev_exact', 'best_exact'
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # V12.10: Handle checkpoints saved with torch.compile (have '_orig_mod.' prefix)
    enc_state = checkpoint['encoder_state_dict']
    dec_state = checkpoint['decoder_state_dict']

    # Check if checkpoint was saved with compiled model
    # Check for top-level prefix (encoder) or nested prefix (decoder's transformer_decoder)
    has_compiled_prefix = (
        any(k.startswith('_orig_mod.') for k in enc_state.keys()) or
        any('._orig_mod.' in k for k in dec_state.keys())
    )
    if has_compiled_prefix:
        print("  Detected compiled checkpoint - stripping '_orig_mod.' prefixes", flush=True)
        enc_state = _strip_compiled_prefix(enc_state)
        dec_state = _strip_compiled_prefix(dec_state)

    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}", flush=True)

    # V12.9: Load entropy manager state if available
    if entropy_manager is not None and 'entropy_manager_state' in checkpoint:
        entropy_manager.load_state(checkpoint['entropy_manager_state'])
        print(f"  Restored entropy manager state", flush=True)

    # V12.10: Load optimizer state if available
    if enc_opt is not None and 'enc_optimizer_state_dict' in checkpoint:
        enc_opt.load_state_dict(checkpoint['enc_optimizer_state_dict'])
        print(f"  Restored encoder optimizer state", flush=True)
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
def evaluate_true_autoregressive(encoder, decoder, loader, device, max_samples=1000):
    """
    Evaluate TRUE autoregressive exact match (no teacher forcing).

    This gives the honest metric of how the model performs at inference time,
    where it must use its own predictions for each token.

    Args:
        encoder: The encoder model
        decoder: The decoder model
        loader: DataLoader with evaluation data
        device: torch device
        max_samples: Maximum samples to evaluate (for speed)

    Returns:
        dict with true_exact_match and sample statistics
    """
    encoder.eval()
    decoder.eval()

    total_exact = 0
    total_samples = 0
    n_by_errors = {0: 0, 1: 0, 2: 0, 3: 0, 'more': 0}  # Track error distribution

    for batch in loader:
        if total_samples >= max_samples:
            break

        # Unpack batch (same format as train_epoch)
        elem_idx, elem_frac, elem_mask, formula_tokens, tc, magpie = [b.to(device) for b in batch]

        batch_size = formula_tokens.size(0)

        # Encode (same order as train_epoch)
        encoder_out = encoder(elem_idx, elem_frac, elem_mask, magpie, tc)
        z = encoder_out['z']
        encoder_skip = encoder_out['attended_input']

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

        total_samples += batch_size

    true_exact_match = total_exact / total_samples if total_samples > 0 else 0

    return {
        'true_exact_match': true_exact_match,
        'total_samples': total_samples,
        'exact_count': total_exact,
        'error_distribution': n_by_errors,
    }


def train_epoch(encoder, decoder, loader, loss_fn, enc_opt, dec_opt, scaler, device, use_amp,
                tf_ratio=1.0, tc_weight=10.0, magpie_weight=2.0,
                focal_loss_fn=None, amp_dtype=torch.float16, accumulation_steps=1):
    """Train for one epoch with curriculum weights and teacher forcing.

    V12.8: Added amp_dtype for configurable mixed precision (bfloat16/float16)
           Added accumulation_steps for gradient accumulation (larger effective batch)
           Added REINFORCE support via CombinedLossWithREINFORCE
    """
    encoder.train()
    decoder.train()

    total_loss = 0
    total_acc = 0
    total_exact = 0
    total_tc = 0
    total_magpie = 0
    total_stoich = 0
    total_reinforce = 0
    total_reward = 0
    total_entropy = 0
    n_batches = 0

    # Zero gradients at start of accumulation
    enc_opt.zero_grad()
    dec_opt.zero_grad()

    for batch_idx, batch in enumerate(loader):
        elem_idx, elem_frac, elem_mask, tokens, tc, magpie = [b.to(device) for b in batch]

        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            # Encoder forward
            encoder_out = encoder(elem_idx, elem_frac, elem_mask, magpie, tc)
            z = encoder_out['z']
            tc_pred = encoder_out['tc_pred']
            magpie_pred = encoder_out['magpie_pred']
            attended_input = encoder_out['attended_input']
            kl_loss = encoder_out['kl_loss']

            # Stoichiometry predictions for loss and conditioning
            fraction_pred = encoder_out.get('fraction_pred')
            element_count_pred = encoder_out.get('element_count_pred')
            if fraction_pred is not None and element_count_pred is not None:
                stoich_pred = torch.cat([fraction_pred, element_count_pred.unsqueeze(-1)], dim=-1)
            else:
                stoich_pred = None

            # Decoder forward with teacher forcing ratio
            formula_logits, _ = decoder(
                z, tokens, encoder_skip=attended_input,
                stoich_pred=stoich_pred, teacher_forcing_ratio=tf_ratio
            )
            formula_targets = tokens[:, 1:]

            # V12.8: Use CombinedLossWithREINFORCE
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
                # V12.4: Stoichiometry loss inputs
                fraction_pred=fraction_pred,
                element_fractions=elem_frac,
                element_mask=elem_mask,
                element_count_pred=element_count_pred,
                # V12.8: Autoregressive REINFORCE inputs
                z=z,
                encoder_skip=attended_input,
                stoich_pred_for_reinforce=stoich_pred,
            )

            loss = loss_dict['total']

        # V12.8: Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps

        # Backward with AMP
        scaler.scale(scaled_loss).backward()

        # Step optimizer every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(enc_opt)
            scaler.unscale_(dec_opt)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

            scaler.step(enc_opt)
            scaler.step(dec_opt)
            scaler.update()

            # Zero gradients for next accumulation
            enc_opt.zero_grad()
            dec_opt.zero_grad()

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
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'accuracy': total_acc / n_batches,
        'exact_match': total_exact / n_batches,
        'tc_loss': total_tc / n_batches,
        'magpie_loss': total_magpie / n_batches,
        'stoich_loss': total_stoich / n_batches,
        'reinforce_loss': total_reinforce / n_batches,
        'mean_reward': total_reward / n_batches,
        'entropy': total_entropy / n_batches,
    }


def train():
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    )

    # V12.8: Wire up decoder for autoregressive REINFORCE sampling
    if TRAIN_CONFIG.get('rl_weight', 0.0) > 0 and TRAIN_CONFIG.get('use_autoregressive_reinforce', True):
        loss_fn.set_decoder(decoder, max_len=TRAIN_CONFIG['max_formula_len'])
        print(f"REINFORCE: Enabled with rl_weight={TRAIN_CONFIG['rl_weight']}, "
              f"n_samples={TRAIN_CONFIG.get('n_samples_rloo', 2)}, "
              f"autoregressive=True (KV-cached)")
    elif TRAIN_CONFIG.get('rl_weight', 0.0) > 0:
        print(f"REINFORCE: Enabled with rl_weight={TRAIN_CONFIG['rl_weight']}, "
              f"n_samples={TRAIN_CONFIG.get('n_samples_rloo', 2)}, "
              f"autoregressive=False (logit sampling)")
    else:
        print("REINFORCE: Disabled (rl_weight=0)")

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

    enc_opt = torch.optim.AdamW(encoder.parameters(), lr=TRAIN_CONFIG['learning_rate'])
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
    scaler = GradScaler('cuda', enabled=use_scaler)
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
    print(f"\nLoss weights (final):")
    print(f"  Formula: {TRAIN_CONFIG['formula_weight']}")
    print(f"  Tc: {TRAIN_CONFIG['tc_weight']}")
    print(f"  Magpie: {TRAIN_CONFIG['magpie_weight']}")
    print(f"  Stoich: {TRAIN_CONFIG['stoich_weight']}")
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
                enc_scheduler=enc_scheduler, dec_scheduler=dec_scheduler
            )
            start_epoch = resume_state['start_epoch']
            prev_exact = resume_state['prev_exact']
            best_exact = resume_state['best_exact']
            print(f"  Starting from epoch {start_epoch}")
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
        loss_fn.set_decoder(decoder, max_len=TRAIN_CONFIG['max_formula_len'])
        print("  Encoder and decoder.transformer_decoder compiled")

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

        # Get adaptive teacher forcing ratio based on previous epoch's performance
        tf_ratio = get_teacher_forcing_ratio(prev_exact)

        # V12.9: Update entropy weight and temperature from entropy manager
        if entropy_manager is not None and TRAIN_CONFIG.get('rl_weight', 0.0) > 0:
            # Get dynamic entropy weight and temperature based on previous epoch's entropy
            new_entropy_weight = entropy_manager.get_entropy_weight(epoch, prev_entropy)
            new_temperature = entropy_manager.get_temperature(epoch, prev_entropy)

            # Update loss function parameters
            loss_fn.entropy_weight = new_entropy_weight
            loss_fn.temperature = new_temperature

        metrics = train_epoch(
            encoder, decoder, train_loader, loss_fn,
            enc_opt, dec_opt, scaler, device, TRAIN_CONFIG['use_amp'],
            tf_ratio=tf_ratio, tc_weight=tc_weight, magpie_weight=magpie_weight,
            focal_loss_fn=focal_loss_fn,
            amp_dtype=amp_dtype,  # V12.8: Configurable AMP dtype
            accumulation_steps=TRAIN_CONFIG.get('accumulation_steps', 1),  # V12.8: Grad accumulation
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
                               prev_exact=prev_exact, best_exact=best_exact)
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
                    f"Stoich: {metrics['stoich_loss']:.4f}")

        # V12.8: Add REINFORCE metrics if enabled (includes entropy in reward)
        # V12.9: Also show adaptive entropy weight
        if TRAIN_CONFIG.get('rl_weight', 0.0) > 0:
            ent_w = loss_fn.entropy_weight if entropy_manager else TRAIN_CONFIG.get('entropy_weight', 0.2)
            base_msg += (f" | RL: {metrics['reinforce_loss']:.4f} | "
                        f"Reward: {metrics['mean_reward']:.3f} | "
                        f"Ent: {metrics.get('entropy', 0):.2f} | "
                        f"EntW: {ent_w:.3f}")

        print(base_msg, flush=True)

        # V12.8: Evaluate TRUE autoregressive exact match every 10 epochs
        # This is the honest metric - no teacher forcing, pure model predictions
        if epoch % 10 == 0:
            true_eval = evaluate_true_autoregressive(
                encoder, decoder, train_loader, device, max_samples=2000
            )
            err_dist = true_eval['error_distribution']
            print(f"  → TRUE Autoregressive: {true_eval['true_exact_match']*100:.1f}% exact "
                  f"({true_eval['exact_count']}/{true_eval['total_samples']}) | "
                  f"Errors: 0={err_dist[0]}, 1={err_dist[1]}, 2={err_dist[2]}, 3={err_dist[3]}, >3={err_dist['more']}", flush=True)

        # Save checkpoints (V12.10: include full training state)
        checkpoint_kwargs = dict(
            entropy_manager=entropy_manager,
            enc_opt=enc_opt, dec_opt=dec_opt,
            enc_scheduler=enc_scheduler, dec_scheduler=dec_scheduler,
            prev_exact=metrics['exact_match'], best_exact=best_exact
        )

        if metrics['exact_match'] > best_exact:
            best_exact = metrics['exact_match']
            _shutdown_state['best_exact'] = best_exact  # V12.10: Keep shutdown state current
            checkpoint_kwargs['best_exact'] = best_exact  # Update with new best
            save_checkpoint(encoder, decoder, epoch, 'best', **checkpoint_kwargs)

        if (epoch + 1) % TRAIN_CONFIG['checkpoint_interval'] == 0:
            save_checkpoint(encoder, decoder, epoch, **checkpoint_kwargs)

    # Final save
    save_checkpoint(encoder, decoder, epoch, 'final', **checkpoint_kwargs)
    print(f"\nTraining complete. Best exact match: {best_exact*100:.1f}%")


if __name__ == '__main__':
    train()
