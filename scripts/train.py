#!/usr/bin/env python3
"""
V12 Full Materials VAE Training - COMPLETE EMPIRICAL REPRESENTATION.

This is the foundation for generative superconductor discovery.

Key changes from V11:
1. FULL ENCODER INPUT: composition + Tc + 145 Magpie features
2. MULTI-HEAD DECODER: formula tokens + Tc + Magpie reconstruction
3. NEW ENCODER: FullMaterialsVAE with three-branch fusion

The latent space z (2048 dims) now encodes EVERYTHING knowable about a
superconductor from empirical data - the universal "plug" for theory networks.

Architecture:
  ┌─ Element attention (256 dim) ─┐
  │                                │
  ├─ Magpie MLP (256 dim) ────────┼→ Fusion (768) → VAE Encoder → z (2048)
  │                                │
  └─ Tc embedding (256 dim) ──────┘
                                    ↓
                             z (2048) → Multi-head Decoder
                                    ├→ Tc head (1)
                                    ├→ Magpie head (145)
                                    └→ attended_input → Formula Decoder

December 2025
"""

import os
import sys
import json
import signal
import atexit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.cuda.amp import autocast, GradScaler  # OPTIMIZATION: Mixed precision
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

# ============================================================================
# GRACEFUL SHUTDOWN - Save latest model on interrupt
# ============================================================================

# Global state for graceful shutdown
_shutdown_state = {
    'encoder': None,
    'decoder': None,
    'epoch': 0,
    'best_train_exact': 0.0,
    'history': [],
    'output_dir': None,
    'norm_stats': None,
    'should_stop': False,
}


def _save_latest_model(reason: str = "shutdown"):
    """Save latest model state to latest_model.pt"""
    state = _shutdown_state
    if state['encoder'] is None or state['output_dir'] is None:
        return

    latest_path = state['output_dir'] / 'latest_model.pt'
    try:
        torch.save({
            'epoch': state['epoch'],
            'encoder_state_dict': state['encoder'].state_dict(),
            'decoder_state_dict': state['decoder'].state_dict(),
            'best_train_exact': state['best_train_exact'],
            'history': state['history'],
            'norm_stats': state['norm_stats'],
            'reason': reason,
        }, latest_path)
        print(f"\n  [Latest model saved at epoch {state['epoch']} ({reason})]")
        sys.stdout.flush()
    except Exception as e:
        print(f"\n  [WARNING: Failed to save latest model: {e}]")


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully - save and exit"""
    print(f"\n\n*** Interrupt received (signal {signum}) - saving latest model... ***")
    _shutdown_state['should_stop'] = True
    _save_latest_model("interrupt")
    print("*** Exiting gracefully ***\n")
    sys.exit(0)


def _cleanup_old_checkpoints(output_dir: Path, keep_last_n: int = 3):
    """Delete old periodic checkpoints, keeping only the last N"""
    import re
    checkpoint_pattern = re.compile(r'checkpoint_epoch_(\d+)\.pt')

    checkpoints = []
    for f in output_dir.glob('checkpoint_epoch_*.pt'):
        match = checkpoint_pattern.match(f.name)
        if match:
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, f))

    # Sort by epoch number
    checkpoints.sort(key=lambda x: x[0])

    # Delete all but the last N
    if len(checkpoints) > keep_last_n:
        to_delete = checkpoints[:-keep_last_n]
        for epoch_num, filepath in to_delete:
            try:
                filepath.unlink()
                print(f"  [Deleted old checkpoint: {filepath.name}]")
            except Exception as e:
                print(f"  [WARNING: Failed to delete {filepath.name}: {e}]")


# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, _signal_handler)  # kill command

# Register atexit handler for normal termination
atexit.register(lambda: _save_latest_model("exit") if not _shutdown_state['should_stop'] else None)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent  # scripts/ -> superconductor-vae/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superconductor.data.dataset import SuperconductorDataset
from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder,
    tokenize_formula, tokens_to_indices, indices_to_formula,
    VOCAB_SIZE, PAD_IDX, START_IDX, END_IDX, IDX_TO_TOKEN
)
from superconductor.losses.reward_v10_discriminative import (
    compute_reward_v10, RewardConfigV10,
    get_default_reward_config_v10, TargetCacheV10, tokens_to_string
)
from superconductor.losses.reward_gpu_native import (
    compute_reward_gpu_native, GPURewardConfig, get_default_gpu_reward_config
)
from superconductor.encoders.element_properties import get_atomic_number
from superconductor.encoders.composition_encoder import CompositionEncoder
from superconductor.training.mastery_sampler import (
    MasteryTracker, MasteryAwareSampler, create_mastery_aware_training_components,
    MultiTaskMasteryTracker, create_multitask_mastery_components  # V12.1: Multi-task mastery
)


# ============================================================================
# V12.5 FIX: PROPER FORMULA PARSER
# ============================================================================
# Handles TWO formats in the fractions dataset:
# 1. Fraction notation: 'Ag(1/500)Al(499/500)' -> exact fractions
# 2. Integer notation: 'Ag1B2' -> integer stoichiometry (for simple compounds)

import re

def parse_fraction_formula(formula: str) -> Dict[str, float]:
    """
    Parse formula in either format:
    - Fraction: 'Ag(1/500)Al(499/500)' -> {'Ag': 0.002, 'Al': 0.998}
    - Integer: 'Ag1B2' -> {'Ag': 1.0, 'B': 2.0} (normalized to fractions)

    Returns dict mapping element symbols to their fraction values (normalized to sum to 1).
    """
    result = {}

    # Try fraction notation first: Element(num/den)
    fraction_pattern = r'([A-Z][a-z]?)\((\d+)/(\d+)\)'
    fraction_matches = re.findall(fraction_pattern, formula)

    if fraction_matches:
        # Formula uses fraction notation
        for element, num, den in fraction_matches:
            result[element] = int(num) / int(den)
    else:
        # Formula uses integer notation: Element followed by optional number
        # Pattern: Element (1-2 letters) followed by number (or nothing = 1)
        integer_pattern = r'([A-Z][a-z]?)(\d*)'
        for match in re.finditer(integer_pattern, formula):
            element = match.group(1)
            count_str = match.group(2)
            if element:  # Skip empty matches
                count = int(count_str) if count_str else 1
                result[element] = float(count)

    return result


# ============================================================================
# V12.2: CURRICULUM WEIGHTING - Focus on struggling formula types
# ============================================================================

# Difficult elements (lowest accuracy from analysis)
DIFFICULT_ELEMENTS = {'In', 'Te', 'Zn', 'Ce', 'Pd', 'Mg', 'Ni'}

# V12.3: Additional difficult elements from latest analysis (83-87% accuracy)
DIFFICULT_ELEMENTS_V12_3 = {'In', 'Te', 'Ge', 'F', 'S', 'Ga', 'Zn', 'Al', 'Pt'}

def count_fractions_in_formula(formula: str) -> int:
    """Count fraction expressions like (1/2) in formula."""
    import re
    return len(re.findall(r'\(\d+/\d+\)', formula))

def get_elements_in_formula(formula: str) -> set:
    """Get set of elements in formula."""
    import re
    return set(re.findall(r'[A-Z][a-z]?', formula))

def get_max_digit_count(formula: str) -> int:
    """
    Get maximum digit count in any numerator or denominator.

    V12.3: Analysis showed 86% of errors are digit errors.
    Formulas with 3+ digit numerators (e.g., 881/500) are hardest.

    Examples:
        "Ca(1/2)Si(3/4)" -> 1 (single digits)
        "La(23/25)Ba2" -> 2 (two digits)
        "Fe(181/100)Zn(3/50)" -> 3 (three digits)
    """
    import re
    fractions = re.findall(r'\((\d+)/(\d+)\)', formula)
    if not fractions:
        return 0
    max_digits = 0
    for num, den in fractions:
        max_digits = max(max_digits, len(num), len(den))
    return max_digits

def compute_curriculum_weights(formulas: List[str], config: dict) -> np.ndarray:
    """
    Compute curriculum weights for each sample based on formula characteristics.

    V12.2 Analysis showed:
    - 1-fraction formulas have 69% accuracy (vs 90% for 0-fraction)
    - Elements In, Te, Zn, Ce have lowest accuracy (54-70%)

    V12.3 Analysis showed:
    - 86% of errors are DIGIT errors (wrong numerator/denominator)
    - Formulas with 3+ digit fractions (e.g., 881/500) are hardest

    Returns array of weights (higher = more training focus).
    """
    weights = np.ones(len(formulas), dtype=np.float32)

    one_fraction_boost = config.get('one_fraction_boost', 2.0)  # 2x weight for 1-fraction
    difficult_element_boost = config.get('difficult_element_boost', 1.5)  # 1.5x per difficult element
    digit_difficulty_boost = config.get('digit_difficulty_boost', 1.5)  # V12.3: 1.5x per extra digit

    # Stats for logging
    n_one_fraction = 0
    n_difficult_elem = 0
    n_high_digit = 0

    for i, formula in enumerate(formulas):
        # Check for 1-fraction (hardest case)
        n_fractions = count_fractions_in_formula(formula)
        if n_fractions == 1:
            weights[i] *= one_fraction_boost
            n_one_fraction += 1

        # Check for difficult elements (use expanded V12.3 set)
        elements = get_elements_in_formula(formula)
        difficult_count = len(elements & DIFFICULT_ELEMENTS_V12_3)
        if difficult_count > 0:
            weights[i] *= (difficult_element_boost ** min(difficult_count, 2))  # Cap at 2 elements
            n_difficult_elem += 1

        # V12.3: Boost formulas with multi-digit fractions
        # 1-digit: no boost, 2-digit: 1.5x, 3-digit: 2.25x
        max_digits = get_max_digit_count(formula)
        if max_digits >= 2:
            digit_boost = digit_difficulty_boost ** (max_digits - 1)
            weights[i] *= digit_boost
            if max_digits >= 3:
                n_high_digit += 1

    print(f"    1-fraction samples: {n_one_fraction}")
    print(f"    Difficult element samples: {n_difficult_elem}")
    print(f"    High-digit (3+) samples: {n_high_digit}")

    return weights


# ============================================================================
# V12.3: ERROR-SPECIFIC DIGIT BOOSTING
# ============================================================================

class DigitErrorTracker:
    """
    Track samples that have digit errors and boost their sampling weight.

    V12.3: Analysis showed 86% of errors are digit errors (wrong numerator/denominator).
    This tracker identifies samples where the model gets digits wrong and boosts
    their weight so they're sampled more frequently.

    Unlike mastery tracking (which measures overall correctness), this specifically
    targets DIGIT errors, allowing the model to focus on numeric precision.
    """

    def __init__(
        self,
        n_samples: int,
        digit_error_boost: float = 5.0,
        decay: float = 0.8,
    ):
        """
        Args:
            n_samples: Total number of training samples
            digit_error_boost: Weight multiplier for samples with digit errors
            decay: Per-epoch decay of error boost (0.8 = 80% retained)
        """
        self.n_samples = n_samples
        self.digit_error_boost = digit_error_boost
        self.decay = decay

        # Per-sample error boost (starts at 1.0 = no boost)
        self.error_boost = np.ones(n_samples, dtype=np.float32)

        # Stats tracking
        self.n_digit_errors_this_epoch = 0
        self.n_element_errors_this_epoch = 0
        self.n_total_errors = 0

    def detect_digit_error(self, pred_formula: str, target_formula: str) -> bool:
        """
        Check if error is a digit error (vs element error).

        Digit error: Same elements, different numbers
        Element error: Different elements present
        """
        import re

        # Get elements
        pred_elements = set(re.findall(r'[A-Z][a-z]?', pred_formula))
        target_elements = set(re.findall(r'[A-Z][a-z]?', target_formula))

        # If elements differ, it's an element error
        if pred_elements != target_elements:
            return False

        # Elements same but formulas differ → digit error
        return pred_formula.strip() != target_formula.strip()

    def update_sample(self, sample_idx: int, pred_formula: str, target_formula: str):
        """
        Update error tracking for a sample.

        If prediction is wrong and it's a digit error, boost the sample weight.
        """
        if pred_formula.strip() == target_formula.strip():
            # Correct prediction - no update needed (decay will reduce boost over time)
            return

        self.n_total_errors += 1

        if self.detect_digit_error(pred_formula, target_formula):
            # Digit error - boost this sample
            self.error_boost[sample_idx] = self.digit_error_boost
            self.n_digit_errors_this_epoch += 1
        else:
            # Element error
            self.n_element_errors_this_epoch += 1

    def decay_epoch(self):
        """
        Apply decay to error boosts at end of epoch.

        Samples that keep getting digit errors will maintain high boost.
        Samples that start getting correct will gradually reduce boost.
        """
        # Decay all boosts toward 1.0
        self.error_boost = 1.0 + (self.error_boost - 1.0) * self.decay

        # Reset epoch counters
        stats = {
            'digit_errors': self.n_digit_errors_this_epoch,
            'element_errors': self.n_element_errors_this_epoch,
            'n_boosted': np.sum(self.error_boost > 1.5),
            'max_boost': float(np.max(self.error_boost)),
        }
        self.n_digit_errors_this_epoch = 0
        self.n_element_errors_this_epoch = 0
        return stats

    def get_weights(self) -> np.ndarray:
        """Get current error boost weights for all samples."""
        return self.error_boost.copy()

    def apply_to_curriculum_weights(self, curriculum_weights: np.ndarray) -> np.ndarray:
        """Multiply curriculum weights by error boost weights."""
        return curriculum_weights * self.error_boost


def scan_for_digit_errors(
    encoder, decoder, data_loader, target_formulas, device, config,
    digit_error_tracker, max_samples=5000
):
    """
    Scan training samples to detect digit errors and update the tracker.

    This runs periodically (every N epochs) to identify samples with digit errors
    and boost their sampling weight.

    Args:
        encoder: V12 encoder model
        decoder: Formula decoder model
        data_loader: Training data loader
        target_formulas: List of target formula strings
        device: torch device
        config: training config
        digit_error_tracker: DigitErrorTracker instance
        max_samples: Maximum samples to scan (for efficiency)

    Returns:
        dict with scan statistics
    """
    encoder.eval()
    decoder.eval()

    n_scanned = 0
    n_digit_errors = 0
    n_element_errors = 0
    n_correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if n_scanned >= max_samples:
                break

            elem_idx, elem_frac, elem_mask, tokens, tc, magpie = [b.to(device) for b in batch]
            batch_size = len(tokens)

            # Encode
            encoder_out = encoder(elem_idx, elem_frac, elem_mask, magpie, tc)
            z = encoder_out['z']
            attended_input = encoder_out['attended_input']

            # Decode (greedy generation, no teacher forcing)
            formula_logits, _, *_extra = decoder(
                z, tokens, encoder_skip=attended_input, teacher_forcing_ratio=0.0
            )
            predictions = formula_logits.argmax(dim=-1)

            # Convert to formulas and check for errors
            for i in range(batch_size):
                if n_scanned >= max_samples:
                    break

                sample_idx = batch_idx * config['batch_size'] + i
                if sample_idx >= len(target_formulas):
                    continue

                # Get predicted and target formulas
                pred_indices = predictions[i].cpu().tolist()
                pred_formula = indices_to_formula(pred_indices)
                target_formula = target_formulas[sample_idx]

                # Update tracker
                digit_error_tracker.update_sample(sample_idx, pred_formula, target_formula)

                # Count for stats
                if pred_formula.strip() == target_formula.strip():
                    n_correct += 1
                elif digit_error_tracker.detect_digit_error(pred_formula, target_formula):
                    n_digit_errors += 1
                else:
                    n_element_errors += 1

                n_scanned += 1

    encoder.train()
    decoder.train()

    # Apply decay after scan
    decay_stats = digit_error_tracker.decay_epoch()

    return {
        'n_scanned': n_scanned,
        'n_correct': n_correct,
        'n_digit_errors': n_digit_errors,
        'n_element_errors': n_element_errors,
        'accuracy': n_correct / n_scanned if n_scanned > 0 else 0,
        'digit_error_rate': n_digit_errors / n_scanned if n_scanned > 0 else 0,
        'n_boosted_samples': decay_stats['n_boosted'],
        'max_boost': decay_stats['max_boost'],
    }


# ============================================================================
# ADAPTIVE LOSS WEIGHTING - Cross-learning with attention
# ============================================================================

class AdaptiveLossWeighter:
    """
    Dynamically adjust loss weights so struggling tasks get more attention.

    Key insight: If a loss is HIGH relative to its history, the model is
    struggling with that task → give it MORE weight (attention).

    This creates cross-learning where:
    - Tasks reinforce each other
    - The "weakest" task automatically gets priority
    - Once a task converges, its weight decreases

    Based on: GradNorm (ICML 2018) and Uncertainty Weighting (CVPR 2018)
    """

    def __init__(
        self,
        task_names: List[str],
        base_weights: Dict[str, float],
        ema_decay: float = 0.9,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ):
        self.task_names = task_names
        self.base_weights = base_weights
        self.ema_decay = ema_decay
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Track exponential moving average of each loss
        self.loss_ema = {name: None for name in task_names}
        # Track initial loss for relative scaling
        self.initial_loss = {name: None for name in task_names}

    def update_and_get_weights(self, current_losses: Dict[str, float]) -> Dict[str, float]:
        """
        Update loss history and return adaptive weights.

        Weight formula:
            weight = base_weight * (current_loss / ema_loss)

        If current > ema → weight > base (struggling, need attention)
        If current < ema → weight < base (converging, reduce attention)
        """
        weights = {}

        for name in self.task_names:
            loss = current_losses.get(name, 0.0)

            # Initialize on first call
            if self.loss_ema[name] is None:
                self.loss_ema[name] = loss + 1e-8
                self.initial_loss[name] = loss + 1e-8

            # Update EMA
            self.loss_ema[name] = (
                self.ema_decay * self.loss_ema[name] +
                (1 - self.ema_decay) * loss
            )

            # Compute adaptive weight
            # Higher current_loss relative to EMA → higher weight
            ratio = (loss + 1e-8) / (self.loss_ema[name] + 1e-8)
            adaptive_weight = self.base_weights[name] * ratio

            # Clamp to reasonable range
            adaptive_weight = max(self.min_weight, min(self.max_weight, adaptive_weight))
            weights[name] = adaptive_weight

        return weights

    def get_attention_status(self) -> str:
        """Return which task currently has highest relative attention."""
        if all(v is None for v in self.loss_ema.values()):
            return "initializing"

        # Find task with highest weight relative to base
        max_ratio = 0
        attention_task = None
        for name in self.task_names:
            if self.loss_ema[name] is not None and self.initial_loss[name] is not None:
                # How much has this task improved?
                improvement = self.initial_loss[name] / (self.loss_ema[name] + 1e-8)
                if improvement > max_ratio:
                    max_ratio = improvement
                    attention_task = name

        return f"{attention_task} (improved {max_ratio:.1f}x)" if attention_task else "balanced"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Checkpoint paths - set to None to train from scratch
# These can be set to resume from a previous checkpoint
RESUME_CHECKPOINT = 'outputs/checkpoint_epoch_0699.pt'  # Resume from epoch 699

# Holdout file - these 45 superconductors are NEVER used in training
HOLDOUT_FILE = PROJECT_ROOT / 'data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json'

# V12 Architecture - FULL MATERIALS ENCODER
MODEL_CONFIG = {
    'latent_dim': 2048,
    'fusion_dim': 256,           # Each encoder branch outputs this
    'magpie_dim': 145,           # V12: Number of Magpie features
    'encoder_hidden': [512, 256],
    'decoder_hidden': [256, 512],
    'd_model': 512,              # For formula decoder
    'nhead': 8,
    'num_layers': 12,
    'dim_feedforward': 2048,
    'n_memory_tokens': 16,
    'element_embed_dim': 128,
}

# V12 Training Config - OPTIMIZED FOR GPU UTILIZATION
TRAIN_CONFIG = {
    'num_epochs': 2000,
    'checkpoint_interval': 50,
    'keep_last_n_checkpoints': 3,  # Delete old checkpoints to save disk (keeps last N)

    # OPTIMIZATION 1: Batch size
    # V12.6: Reduced to 16 due to PyTorch 2.4.0 using more VRAM (OOM at 32)
    'batch_size': 16,          # Reduced from 32 to avoid GPU memory pressure
    'accumulation_steps': 1,   # No accumulation = better GPU utilization

    # OPTIMIZATION 2: Multi-worker data loading
    # V12.6: Set to 0 to avoid WSL2 CUDA handle corruption issues with PyTorch 2.4.0
    'num_workers': 0,          # Single-process loading (safer for WSL2)
    'pin_memory': True,        # Faster GPU transfer
    'prefetch_factor': 4,      # Prefetch more batches

    # OPTIMIZATION 3: Mixed precision (FP16)
    'use_amp': True,           # Automatic Mixed Precision

    'encoder_lr': 5e-5,    # Higher LR for new encoder
    'decoder_lr': 1e-5,

    # Loss weights - BALANCED for meaningful gradients
    # Formula loss ~3.0, Tc loss ~0.1, Magpie loss ~0.5
    # Scale weights so each contributes meaningfully
    'formula_weight': 1.0,   # Formula reconstruction (REINFORCE) → ~3.0 contrib
    'tc_weight': 10.0,       # Tc reconstruction (MSE) → ~1.0 contrib (was 0.5!)
    'magpie_weight': 2.0,    # Magpie reconstruction (MSE) → ~1.0 contrib (was 0.1!)
    'kl_weight': 0.0,        # Zero for jagged latent

    # REINFORCE settings
    'ce_weight': 1.0,
    'rl_weight': 2.5,        # Re-enabled: RLOO bug fix (advantages were averaging to 0)
    'n_samples_rloo': 2,     # A100: use 4 samples for better baseline variance
    'temperature': 0.5,      # AGGRESSIVE: 0.8 → 0.5 (sharper distributions)
    'entropy_weight': 0.01,
    # RL method: 'scst' (Self-Critical Sequence Training) or 'rloo'
    # SCST: 1 greedy + 1 sample (2 passes), baseline = greedy reward (Rennie et al. 2017)
    # RLOO: K samples, leave-one-out baseline (Ahmadian et al. 2024)
    'rl_method': 'scst',     # SCST: lower variance, aligns baseline with test-time greedy
    'use_autoregressive_reinforce': True,  # KV-cached autoregressive sampling

    'max_formula_len': 60,  # V12.6: Match checkpoint (pos_encoding.pe shape)

    # Teacher forcing - V12.5: Decay TF to force autoregressive learning
    # Without TF decay, model never learns to handle its own errors
    'tf_start': 1.0,              # Start with full teacher forcing (warm-up)
    'tf_end': 0.0,                # Decay to 0% = full autoregressive
    'tf_decay_epochs': 100,       # Faster decay over 100 epochs
    'tf_adaptive': False,         # Disable adaptive for now

    # Per-sample teacher forcing
    'tf_per_sample': False,       # Disabled - use global TF schedule
    'tf_struggling': 1.0,         # Not used when tf_per_sample=False
    'tf_medium': 0.5,
    'tf_mastered': 0.0,

    # Mastery-aware sampling - DISABLED (was causing issues, PyTorch 2.4 approach)
    'use_mastery_sampling': False,
    'use_multitask_mastery': True,   # V12.1: Track formula + Tc + Magpie separately
    'mastery_window': 5,             # Rolling window for mastery tracking
    'mastery_threshold': 0.8,        # Score to consider "mastered" (for stats)
    'perfection_threshold': 0.95,    # V12.1: Only retire at 95%+ (allows 100% convergence)
    'min_replay_prob': 0.05,         # AGGRESSIVE: 0.1 → 0.05 (less replay of perfected)
    'focus_factor': 3.0,             # AGGRESSIVE: 1.5 → 3.0 (3x focus on struggling)
    # V12.1: Per-task mastery thresholds
    'tc_mastery_threshold': 0.8,     # Tc error threshold for "correct"
    'magpie_mastery_threshold': 0.8, # Magpie error threshold for "correct"
    'tc_error_threshold': 0.1,       # Tc prediction within 10% = correct
    'magpie_error_threshold': 0.2,   # Magpie MSE < 0.2 = correct

    # V12.2: CURRICULUM WEIGHTING (from struggle analysis)
    # Analysis showed: 1-fraction=69% acc, In/Te/Zn/Ce elements=54-70% acc
    'use_curriculum_weights': True,
    'one_fraction_boost': 2.5,       # 2.5x weight for 1-fraction formulas (hardest)
    'difficult_element_boost': 1.8,  # 1.8x weight per difficult element

    # V12.3: DIGIT DIFFICULTY WEIGHTING (86% of errors are digit errors)
    # Analysis showed: 3+ digit numerators (e.g., 881/500) are hardest
    'digit_difficulty_boost': 1.5,   # 1.5x per extra digit (2-digit: 1.5x, 3-digit: 2.25x)

    # V12.3: ERROR-SPECIFIC BOOSTING
    'use_error_specific_boost': True,
    'digit_error_boost': 5.0,        # 5x weight for samples with recent digit errors
    'error_boost_decay': 0.8,        # Decay error boost each epoch (0.8 = 80% retained)

    # V12.6: FOCAL LOSS + LABEL SMOOTHING (to break 98.8% plateau)
    'use_focal_loss': True,          # Use focal loss instead of standard CE
    'focal_gamma': 2.0,              # Focal loss gamma (0=CE, 2=typical focal)
    'label_smoothing': 0.1,          # Label smoothing factor (0=no smoothing)
}

OUTPUT_DIR = PROJECT_ROOT / 'outputs'


# ============================================================================
# FOCAL LOSS WITH LABEL SMOOTHING - Added for V12.6
# ============================================================================

class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss with Label Smoothing for formula token prediction.

    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    - Down-weights easy examples (high p_t), focuses on hard examples (low p_t)
    - gamma=0: standard CE, gamma=2: typical focal loss

    Label Smoothing: Softens one-hot targets to prevent overconfidence
    - Instead of [0,0,1,0], uses [eps/K, eps/K, 1-eps, eps/K]
    - Helps generalization by preventing extreme logits

    Why this helps break 98.8% plateau:
    1. At 98.8%, most tokens are "easy" - CE treats them all equally
    2. Focal loss automatically focuses gradient on the 1.2% hard tokens
    3. Label smoothing prevents the model from being overconfident on easy tokens
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 1.0,
        smoothing: float = 0.1,
        ignore_index: int = -100
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch*seq, vocab_size] or [batch, seq, vocab_size]
            targets: [batch*seq] or [batch, seq]

        Returns:
            Scalar focal loss
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.contiguous().view(-1, vocab_size)
            targets = targets.contiguous().view(-1)
        else:
            vocab_size = logits.shape[-1]

        # Create mask for valid tokens (not padding)
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)

        # Filter to valid tokens only
        logits = logits[valid_mask]
        targets = targets[valid_mask]

        # Apply label smoothing to targets
        # Smooth targets: (1 - smoothing) for correct class, smoothing/(K-1) for others
        n_classes = vocab_size
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma for each class
        # p_t is the probability of the smoothed target
        focal_weight = (1 - probs) ** self.gamma

        # Focal loss with label smoothing
        # Loss = -sum_c(smooth_target_c * focal_weight_c * log_prob_c)
        focal_loss = -self.alpha * (smooth_targets * focal_weight * log_probs).sum(dim=-1)

        return focal_loss.mean()


# ============================================================================
# V12 REINFORCE LOSS - INCLUDES FORMULA + TC + MAGPIE + STOICHIOMETRY (V12.4)
# ============================================================================

class REINFORCELossV12(nn.Module):
    """
    V12 Combined Loss for Full Materials VAE.

    Components:
    1. Formula reconstruction (REINFORCE with RLOO) - **GPU-NATIVE REWARDS**
    2. Tc reconstruction (MSE)
    3. Magpie reconstruction (MSE)
    4. KL divergence (optional, currently 0)
    5. Stoichiometry MSE (V12.4) - Direct fraction value prediction loss

    V12 OPTIMIZATION: Uses GPU-native reward computation (no string parsing)
    for 10-100x faster training.

    V12.4 STOICHIOMETRY-AWARE: Adds a loss that penalizes stoichiometry errors
    proportionally to their magnitude. This helps the model understand that
    1/5 (0.2) and 9/5 (1.8) are very different, not just a single token error.
    """

    def __init__(
        self,
        ce_weight: float = 0.3,
        rl_weight: float = 1.0,
        tc_weight: float = 0.5,
        magpie_weight: float = 0.1,
        kl_weight: float = 0.0,
        entropy_weight: float = 0.01,
        stoichiometry_weight: float = 2.0,  # V12.4: Stoichiometry loss weight
        n_samples_rloo: int = 2,
        temperature: float = 0.8,
        reward_config: Optional[RewardConfigV10] = None,
        target_cache: Optional[TargetCacheV10] = None,
        use_gpu_reward: bool = True,  # V12: GPU-native rewards
        # V12.6: Focal loss parameters
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        use_focal_loss: bool = True,
        # V12.7: True autoregressive REINFORCE with KV caching
        use_autoregressive_reinforce: bool = False,
        # RL method: 'scst' (Self-Critical Sequence Training) or 'rloo'
        rl_method: str = 'scst',
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.rl_weight = rl_weight
        self.tc_weight = tc_weight
        self.magpie_weight = magpie_weight
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        self.stoichiometry_weight = stoichiometry_weight  # V12.4
        self.n_samples_rloo = n_samples_rloo
        self.temperature = temperature

        self.reward_config = reward_config or get_default_reward_config_v10()
        self.gpu_reward_config = get_default_gpu_reward_config()
        self.target_cache = target_cache
        self.use_gpu_reward = use_gpu_reward  # V12: Use GPU-native rewards

        # V12.6: Focal loss with label smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        # RL method and autoregressive sampling
        self.use_autoregressive_reinforce = use_autoregressive_reinforce
        self.rl_method = rl_method  # 'scst' or 'rloo'
        self._decoder = None  # Set via set_decoder()
        self._max_len = 60

        if use_focal_loss:
            self.ce_loss = FocalLossWithLabelSmoothing(
                gamma=focal_gamma,
                smoothing=label_smoothing,
                ignore_index=PAD_IDX
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_IDX)

    def set_target_cache(self, target_cache: TargetCacheV10):
        self.target_cache = target_cache

    def set_decoder(self, decoder, max_len: int = 60):
        """
        Set the decoder for autoregressive REINFORCE sampling (V12.7).

        Args:
            decoder: EnhancedTransformerDecoder instance
            max_len: Maximum sequence length for sampling
        """
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
        V12.7: Compute RLOO advantages using true autoregressive sampling with KV cache.

        This is more accurate than sampling from teacher-forced logits because it
        uses the model's actual autoregressive behavior. The KV caching makes this
        ~60x faster than naive autoregressive sampling.

        Args:
            z: Latent vectors [batch, latent_dim]
            targets: Target tokens [batch, seq_len]
            encoder_skip: Optional skip connection
            stoich_pred: Optional stoichiometry conditioning

        Returns:
            advantages: RLOO advantages [batch]
            mean_log_probs: Mean log probabilities [batch]
            mean_rewards: Mean rewards [batch]
        """
        if self._decoder is None:
            raise RuntimeError("Decoder not set. Call set_decoder() first.")

        batch_size = z.shape[0]
        n_samples = self.n_samples_rloo
        all_rewards = []
        all_log_probs = []

        # Create target mask
        target_mask = (targets != PAD_IDX)

        for _ in range(n_samples):
            # Sample autoregressively using KV-cached generation
            # V12.8: Now returns 4 values (tokens, log_probs, entropy, mask)
            sampled_tokens, log_probs, _entropy, mask = self._decoder.sample_for_reinforce(
                z=z,
                encoder_skip=encoder_skip,
                stoich_pred=stoich_pred,
                temperature=self.temperature,
                max_len=self._max_len,
            )

            # Compute rewards using GPU-native method
            if self.use_gpu_reward:
                # Pad sampled tokens to match target length if needed
                if sampled_tokens.size(1) < targets.size(1):
                    pad_len = targets.size(1) - sampled_tokens.size(1)
                    sampled_tokens = F.pad(sampled_tokens, (0, pad_len), value=PAD_IDX)
                    log_probs = F.pad(log_probs, (0, pad_len), value=0.0)
                    mask = F.pad(mask, (0, pad_len), value=0.0)
                elif sampled_tokens.size(1) > targets.size(1):
                    sampled_tokens = sampled_tokens[:, :targets.size(1)]
                    log_probs = log_probs[:, :targets.size(1)]
                    mask = mask[:, :targets.size(1)]

                rewards = compute_reward_gpu_native(
                    sampled_tokens, targets, mask,
                    config=self.gpu_reward_config,
                    pad_idx=PAD_IDX, end_idx=END_IDX
                )
            else:
                rewards = compute_reward_v10(
                    sampled_tokens, targets, IDX_TO_TOKEN,
                    mask, self.reward_config, self.target_cache
                )

            # Sum log probs over sequence
            seq_log_prob = (log_probs * mask).sum(dim=1)
            all_rewards.append(rewards)
            all_log_probs.append(seq_log_prob)

        # RLOO baseline computation
        rewards_stack = torch.stack(all_rewards, dim=0)
        log_probs_stack = torch.stack(all_log_probs, dim=0)
        total_reward = rewards_stack.sum(dim=0)

        # BUG FIX: Each RLOO sample contributes its own independent gradient.
        # Previously averaged advantages across samples before multiplying with
        # log_probs — since sum of RLOO advantages = 0 for any K, this guaranteed
        # zero gradient. Fix: multiply each sample's advantage with its own
        # log_probs, then sum (not average). More samples = stronger, better-
        # informed gradient. rl_weight controls overall magnitude.
        reinforce_loss = torch.zeros(1, device=rewards_stack.device)
        for i in range(n_samples):
            baseline_i = (total_reward - rewards_stack[i]) / (n_samples - 1)
            advantage_i = rewards_stack[i] - baseline_i
            reinforce_loss = reinforce_loss + -(advantage_i * log_probs_stack[i]).mean()

        mean_rewards = rewards_stack.mean(dim=0)

        return reinforce_loss, mean_rewards

    def compute_scst(
        self,
        z: torch.Tensor,
        targets: torch.Tensor,
        encoder_skip: torch.Tensor = None,
        stoich_pred: torch.Tensor = None,
    ):
        """
        Self-Critical Sequence Training (Rennie et al. 2017).

        Uses greedy decode reward as baseline instead of RLOO leave-one-out.
        Advantage = reward(sampled) - reward(greedy). If the sample beats
        greedy, reinforce it; if greedy beats sample, push away from sample.

        Benefits over RLOO:
        - Baseline aligns with test-time behavior (greedy decode)
        - Only 2 forward passes (1 greedy + 1 sample) vs K for RLOO
        - No advantage-cancellation issues
        - Lower variance baseline (greedy is deterministic)

        Returns:
            reinforce_loss: Scalar SCST loss
            mean_rewards: Mean sampled rewards [batch] (for logging)
        """
        if self._decoder is None:
            raise RuntimeError("Decoder not set. Call set_decoder() first.")

        # Precompute memory once — shared by both greedy and sample passes
        cached_memory = self._decoder.precompute_memory(z, encoder_skip, stoich_pred)

        # 1. Greedy decode (deterministic baseline) — no gradients needed
        with torch.no_grad():
            greedy_tokens, _, _ = self._decoder.generate_with_kv_cache(
                z=z,
                encoder_skip=encoder_skip,
                stoich_pred=stoich_pred,
                temperature=0.0,  # argmax
                max_len=self._max_len,
                return_log_probs=False,
                cached_memory=cached_memory,
            )

            # Pad/truncate greedy tokens to match target length for reward computation
            if greedy_tokens.size(1) < targets.size(1):
                pad_len = targets.size(1) - greedy_tokens.size(1)
                greedy_tokens = F.pad(greedy_tokens, (0, pad_len), value=PAD_IDX)
            elif greedy_tokens.size(1) > targets.size(1):
                greedy_tokens = greedy_tokens[:, :targets.size(1)]

            # Greedy mask (valid until END token)
            greedy_is_end = (greedy_tokens == END_IDX)
            greedy_end_pos = torch.argmax(greedy_is_end.int(), dim=1)
            greedy_has_end = greedy_is_end.any(dim=1)
            greedy_end_pos = torch.where(
                greedy_has_end, greedy_end_pos,
                torch.tensor(greedy_tokens.size(1), device=z.device)
            )
            positions = torch.arange(greedy_tokens.size(1), device=z.device).unsqueeze(0)
            greedy_mask = (positions <= greedy_end_pos.unsqueeze(1)).float()

            # Compute greedy reward (baseline)
            greedy_rewards = compute_reward_gpu_native(
                greedy_tokens, targets, greedy_mask,
                config=self.gpu_reward_config,
                pad_idx=PAD_IDX, end_idx=END_IDX
            )

        # 2. Sample decode — need log_probs for gradient
        sampled_tokens, log_probs, _entropy, sample_mask = self._decoder.sample_for_reinforce(
            z=z,
            encoder_skip=encoder_skip,
            stoich_pred=stoich_pred,
            temperature=self.temperature,
            max_len=self._max_len,
            cached_memory=cached_memory,
        )

        # Pad/truncate sampled tokens to match target length
        if sampled_tokens.size(1) < targets.size(1):
            pad_len = targets.size(1) - sampled_tokens.size(1)
            sampled_tokens = F.pad(sampled_tokens, (0, pad_len), value=PAD_IDX)
            log_probs = F.pad(log_probs, (0, pad_len), value=0.0)
            sample_mask = F.pad(sample_mask, (0, pad_len), value=0.0)
        elif sampled_tokens.size(1) > targets.size(1):
            sampled_tokens = sampled_tokens[:, :targets.size(1)]
            log_probs = log_probs[:, :targets.size(1)]
            sample_mask = sample_mask[:, :targets.size(1)]

        # Compute sample reward
        with torch.no_grad():
            sample_rewards = compute_reward_gpu_native(
                sampled_tokens, targets, sample_mask,
                config=self.gpu_reward_config,
                pad_idx=PAD_IDX, end_idx=END_IDX
            )

        # 3. SCST advantage: how much better is the sample than greedy?
        advantages = sample_rewards - greedy_rewards

        # 4. Sequence log prob (sum over valid positions)
        seq_log_prob = (log_probs * sample_mask).sum(dim=1)

        # 5. REINFORCE loss: push toward samples that beat greedy,
        #    push away from samples worse than greedy
        reinforce_loss = -(advantages * seq_log_prob).mean()

        return reinforce_loss, sample_rewards

    def sample_from_logits(self, logits: torch.Tensor, temperature: float):
        batch_size, seq_len, vocab_size = logits.shape
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        flat_probs = probs.view(-1, vocab_size)
        sampled_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        sampled_tokens = sampled_flat.view(batch_size, seq_len)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        sampled_log_probs = log_probs.gather(2, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        return sampled_tokens, sampled_log_probs

    def compute_rloo_advantages(self, logits, targets, mask):
        batch_size = logits.shape[0]
        n_samples = self.n_samples_rloo
        all_rewards = []
        all_log_probs = []

        for _ in range(n_samples):
            sampled_tokens, sampled_log_probs = self.sample_from_logits(logits, self.temperature)

            # V12: GPU-native reward computation (100x faster)
            if self.use_gpu_reward:
                rewards = compute_reward_gpu_native(
                    sampled_tokens, targets, mask,
                    config=self.gpu_reward_config,
                    pad_idx=PAD_IDX, end_idx=END_IDX
                )
            else:
                # Fallback to CPU-based reward (slower)
                rewards = compute_reward_v10(
                    sampled_tokens, targets, IDX_TO_TOKEN,
                    mask, self.reward_config, self.target_cache
                )

            masked_log_probs = sampled_log_probs * mask.float()
            seq_log_prob = masked_log_probs.sum(dim=1)
            all_rewards.append(rewards)
            all_log_probs.append(seq_log_prob)

        rewards_stack = torch.stack(all_rewards, dim=0)
        log_probs_stack = torch.stack(all_log_probs, dim=0)
        total_reward = rewards_stack.sum(dim=0)

        # BUG FIX: Same fix as compute_rloo_autoregressive — each sample's
        # advantage multiplied with its own log_probs, summed (not averaged).
        reinforce_loss = torch.zeros(1, device=rewards_stack.device)
        for i in range(n_samples):
            baseline_i = (total_reward - rewards_stack[i]) / (n_samples - 1)
            advantage_i = rewards_stack[i] - baseline_i
            reinforce_loss = reinforce_loss + -(advantage_i * log_probs_stack[i]).mean()

        mean_rewards = rewards_stack.mean(dim=0)

        return reinforce_loss, mean_rewards

    def forward(
        self,
        formula_logits: torch.Tensor,
        formula_targets: torch.Tensor,
        tc_pred: torch.Tensor,
        tc_true: torch.Tensor,
        magpie_pred: torch.Tensor,
        magpie_true: torch.Tensor,
        kl_loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        # V12.4: Stoichiometry loss inputs
        fraction_pred: Optional[torch.Tensor] = None,
        element_fractions: Optional[torch.Tensor] = None,
        element_mask: Optional[torch.Tensor] = None,
        element_count_pred: Optional[torch.Tensor] = None,
        # V12.7: Autoregressive REINFORCE inputs (optional)
        z: Optional[torch.Tensor] = None,
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred_for_reinforce: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute V12 combined loss with stoichiometry awareness (V12.4).

        V12.7: When use_autoregressive_reinforce=True and z is provided, uses
        true autoregressive sampling with KV caching for REINFORCE instead of
        sampling from teacher-forced logits.
        """
        import time
        t0 = time.time()

        batch_size, seq_len, vocab_size = formula_logits.shape

        if mask is None:
            mask = formula_targets != PAD_IDX

        # 1. Formula CE Loss (V12.6: handles both focal loss and standard CE)
        t1 = time.time()
        if self.use_focal_loss:
            # FocalLossWithLabelSmoothing returns a scalar directly
            # Pass the full tensors with mask handled internally
            targets_with_pad = formula_targets.clone()
            targets_with_pad[~mask] = PAD_IDX  # Mark non-valid positions
            formula_ce_loss = self.ce_loss(formula_logits, targets_with_pad)
        else:
            # Standard CE returns per-token loss
            logits_flat = formula_logits.contiguous().view(-1, vocab_size)
            targets_flat = formula_targets.contiguous().view(-1)
            ce_loss_per_token = self.ce_loss(logits_flat, targets_flat)
            ce_loss_per_token = ce_loss_per_token.view(batch_size, seq_len)
            formula_ce_loss = (ce_loss_per_token * mask.float()).sum(dim=1).mean()

        t2 = time.time()
        # 2. Formula REINFORCE Loss - SKIP if rl_weight=0
        if self.rl_weight > 0:
            if self.use_autoregressive_reinforce and z is not None and self._decoder is not None:
                # SCST or RLOO with true autoregressive sampling
                if self.rl_method == 'scst':
                    reinforce_loss, mean_rewards = self.compute_scst(
                        z=z,
                        targets=formula_targets,
                        encoder_skip=encoder_skip,
                        stoich_pred=stoich_pred_for_reinforce,
                    )
                else:
                    reinforce_loss, mean_rewards = self.compute_rloo_autoregressive(
                        z=z,
                        targets=formula_targets,
                        encoder_skip=encoder_skip,
                        stoich_pred=stoich_pred_for_reinforce,
                    )
            else:
                # Fallback: sample from teacher-forced logits (faster but less accurate)
                reinforce_loss, mean_rewards = self.compute_rloo_advantages(
                    formula_logits, formula_targets, mask
                )
        else:
            # Skip REINFORCE computation entirely
            reinforce_loss = torch.tensor(0.0, device=formula_logits.device)
            mean_rewards = torch.tensor(0.0, device=formula_logits.device)
        t3 = time.time()
        # Print timing for first 5 calls
        if not hasattr(self, '_timing_count'):
            self._timing_count = 0
        if self._timing_count < 5:
            print(f"  [TIMING] CE: {(t2-t1)*1000:.1f}ms, REINFORCE: {(t3-t2)*1000:.1f}ms", flush=True)
            self._timing_count += 1

        # 3. Entropy Bonus
        probs = F.softmax(formula_logits, dim=-1)
        log_probs = F.log_softmax(formula_logits, dim=-1)
        entropy_per_position = -(probs * log_probs).sum(dim=-1)
        entropy = (entropy_per_position * mask.float()).sum(dim=1).mean()

        # 4. Tc Loss (MSE)
        tc_loss = F.mse_loss(tc_pred, tc_true)

        # 5. Magpie Loss (MSE)
        magpie_loss = F.mse_loss(magpie_pred, magpie_true)

        # 6. Stoichiometry Loss (V12.4) - Direct fraction value prediction
        # This loss penalizes stoichiometry errors proportionally to magnitude:
        #   - Error between 1/5 (0.2) and 2/5 (0.4) = |0.2 - 0.4| = 0.2
        #   - Error between 1/5 (0.2) and 9/5 (1.8) = |0.2 - 1.8| = 1.6 (8x larger!)
        # Unlike token CE where both are "1 token wrong", this captures semantic meaning.
        stoich_loss = torch.tensor(0.0, device=formula_logits.device)
        element_count_loss = torch.tensor(0.0, device=formula_logits.device)

        if fraction_pred is not None and element_fractions is not None and element_mask is not None:
            # Masked MSE on element fractions
            elem_mask_float = element_mask.float()

            # Squared error per element
            squared_error = (fraction_pred - element_fractions) ** 2
            masked_squared_error = squared_error * elem_mask_float

            # Average over valid elements
            n_valid = elem_mask_float.sum(dim=1, keepdim=True).clamp(min=1)
            per_sample_mse = masked_squared_error.sum(dim=1) / n_valid.squeeze(-1)
            stoich_loss = per_sample_mse.mean()

            # Element count loss
            if element_count_pred is not None:
                element_count_target = element_mask.sum(dim=1).float()
                element_count_loss = F.mse_loss(element_count_pred, element_count_target)

        # 7. Combined Loss
        formula_loss = self.ce_weight * formula_ce_loss + self.rl_weight * reinforce_loss
        total_loss = (
            formula_loss +
            self.tc_weight * tc_loss +
            self.magpie_weight * magpie_loss +
            self.kl_weight * kl_loss +
            self.stoichiometry_weight * stoich_loss +
            0.5 * element_count_loss -  # Lower weight for count
            self.entropy_weight * entropy
        )

        # Compute accuracy metrics
        predictions = formula_logits.argmax(dim=-1)
        correct = (predictions == formula_targets) & mask
        token_accuracy = correct.sum().float() / mask.sum().float()
        seq_correct = (correct | ~mask).all(dim=1)
        exact_match = seq_correct.float().mean()

        return {
            'total': total_loss,
            'formula_ce_loss': formula_ce_loss,
            'reinforce_loss': reinforce_loss,
            'tc_loss': tc_loss,
            'magpie_loss': magpie_loss,
            'stoich_loss': stoich_loss,  # V12.4
            'element_count_loss': element_count_loss,  # V12.4
            'kl_loss': kl_loss,
            'entropy': entropy,
            'mean_reward': mean_rewards.mean(),
            'token_accuracy': token_accuracy,
            'exact_match': exact_match,
        }


# ============================================================================
# DATA LOADING - V12: FULL MATERIALS (Formula + Tc + Magpie)
# ============================================================================

def load_holdout_indices(holdout_file: Path, formulas: list = None) -> set:
    """Load holdout sample indices by matching formulas (robust to row reordering).

    Args:
        holdout_file: Path to GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json
        formulas: List of formula strings from the loaded CSV. If provided,
                  matches by formula text. Falls back to original_index if not provided.

    Returns:
        Set of integer indices that are holdout samples.
    """
    with open(holdout_file, 'r') as f:
        holdout_data = json.load(f)

    if formulas is not None:
        # Robust: match by formula string
        holdout_formulas = {s['formula'] for s in holdout_data['holdout_samples']}
        indices = {i for i, f in enumerate(formulas) if f in holdout_formulas}
        if len(indices) != len(holdout_formulas):
            print(f"  WARNING: Found {len(indices)}/{len(holdout_formulas)} holdout samples in data")
        return indices
    else:
        # Legacy fallback: positional index
        return {s['original_index'] for s in holdout_data['holdout_samples']}


def get_magpie_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric Magpie feature columns."""
    # Exclude non-feature columns
    exclude = ['formula', 'Tc', 'composition', 'category', 'is_superconductor', 'compound possible']
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    magpie_cols = [c for c in numeric_cols if c not in exclude]
    return magpie_cols


def load_data():
    """Load and prepare full materials data (formula + Tc + Magpie).

    V12.5 FIX: Now uses ONLY the fractions file for both encoder input and decoder target.
    Previously used raw_df (decimal notation) for encoder, causing input/output mismatch.
    """
    print("=" * 60)
    print("STEP 1: Loading Full Materials Data (V12.5 - FIXED)")
    print("=" * 60)

    fraction_path = PROJECT_ROOT / 'data/processed/supercon_fractions_combined.csv'
    df = pd.read_csv(fraction_path)

    print(f"Loaded {len(df)} samples from {fraction_path}")

    # V12.5: Get Magpie feature columns from fractions file (has same columns)
    magpie_cols = get_magpie_columns(df)
    print(f"Found {len(magpie_cols)} Magpie feature columns")

    holdout_indices = load_holdout_indices(HOLDOUT_FILE, df['formula'].tolist())
    print(f"Loaded {len(holdout_indices)} holdout indices from {HOLDOUT_FILE.name}")

    # V12.5: Return single df instead of fraction_df, raw_df
    return df, holdout_indices, magpie_cols


def prepare_data(df, holdout_indices, magpie_cols, config):
    """Prepare full materials dataset: composition + Tc + Magpie features.

    V12.5 FIX: Now uses single df (fractions file) for both encoder and decoder.
    Uses parse_fraction_formula() to extract exact fractions from notation like 'Ag(1/500)Al(499/500)'.
    """
    formulas = df['formula'].tolist()
    tc_values = df['Tc'].tolist()

    # Normalize Tc
    tc_mean = np.mean(tc_values)
    tc_std = np.std(tc_values)
    tc_normalized = [(tc - tc_mean) / tc_std for tc in tc_values]
    print(f"Tc normalization: mean={tc_mean:.2f}K, std={tc_std:.2f}K")

    # V12: Extract and normalize Magpie features
    print(f"Extracting {len(magpie_cols)} Magpie features...")
    magpie_data = df[magpie_cols].values.astype(np.float32)

    # Handle NaN values
    nan_mask = np.isnan(magpie_data)
    if nan_mask.any():
        print(f"  WARNING: Found {nan_mask.sum()} NaN values, replacing with column means")
        col_means = np.nanmean(magpie_data, axis=0)
        for col_idx in range(magpie_data.shape[1]):
            magpie_data[nan_mask[:, col_idx], col_idx] = col_means[col_idx]

    # Normalize each feature
    magpie_mean = magpie_data.mean(axis=0)
    magpie_std = magpie_data.std(axis=0) + 1e-8
    magpie_normalized = (magpie_data - magpie_mean) / magpie_std
    print(f"  Magpie feature shape: {magpie_normalized.shape}")

    # Tokenize formulas
    max_len = config['max_formula_len']
    print(f"Tokenizing formulas...")

    all_tokens = []
    for formula in formulas:
        tokens = tokenize_formula(formula)
        token_indices = tokens_to_indices(tokens, max_len=max_len)
        all_tokens.append(token_indices)

    formula_tokens = torch.stack(all_tokens)

    # Build target cache for REINFORCE
    print("Building target cache (pre-parsing all formulas)...")
    training_formulas = [f for i, f in enumerate(formulas) if i not in holdout_indices]
    target_cache = TargetCacheV10(training_formulas, IDX_TO_TOKEN)
    print(f"  Cached {len(target_cache.components)} unique formula parses")

    # V12.5 FIX: Prepare encoder inputs using parse_fraction_formula()
    # Now parses fraction notation directly: 'Ag(1/500)Al(499/500)' -> exact fractions
    MAX_ELEMENTS = 12
    print(f"Parsing fraction formulas for encoder input (V12.5 fix)...")

    element_indices = torch.zeros(len(formulas), MAX_ELEMENTS, dtype=torch.long)
    element_fractions = torch.zeros(len(formulas), MAX_ELEMENTS, dtype=torch.float32)
    element_mask = torch.zeros(len(formulas), MAX_ELEMENTS, dtype=torch.bool)

    parse_errors = 0
    for i, formula in enumerate(formulas):
        try:
            # V12.5: Use new fraction parser instead of CompositionEncoder
            parsed = parse_fraction_formula(formula)
            if not parsed:
                parse_errors += 1
                continue

            # Fractions should already sum to 1, but normalize just in case
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
        except Exception as e:
            parse_errors += 1
            continue

    if parse_errors > 0:
        print(f"  WARNING: {parse_errors} formulas failed to parse")
    print(f"  Successfully parsed {len(formulas) - parse_errors} formulas")

    tc_tensor = torch.tensor(tc_normalized, dtype=torch.float32)
    magpie_tensor = torch.tensor(magpie_normalized, dtype=torch.float32)

    # Create training set (exclude 45 holdout)
    n_samples = len(formulas)
    all_indices = set(range(n_samples))
    train_indices = sorted(all_indices - holdout_indices)

    print("=" * 60)
    print("STEP 2: Creating Training Data (V12 - FULL MATERIALS)")
    print("=" * 60)
    print(f"Total samples: {n_samples}")
    print(f"Holdout samples (NEVER TRAIN): {len(holdout_indices)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Features per sample: formula + Tc + {len(magpie_cols)} Magpie")

    full_dataset = TensorDataset(
        element_indices, element_fractions, element_mask,
        formula_tokens, tc_tensor, magpie_tensor
    )

    train_dataset = Subset(full_dataset, train_indices)

    # MASTERY-AWARE SAMPLING: Focus on struggling samples
    mastery_tracker = None
    mastery_sampler = None
    curriculum_weights = None  # V12.3: Initialize for digit error tracking

    if config.get('use_mastery_sampling', False):
        use_multitask = config.get('use_multitask_mastery', False)

        print(f"Enabling mastery-aware sampling:")
        print(f"  Mode: {'MULTI-TASK (formula+Tc+Magpie)' if use_multitask else 'SINGLE-TASK (formula only)'}")
        print(f"  Window size: {config.get('mastery_window', 5)}")
        print(f"  Mastery threshold: {config.get('mastery_threshold', 0.8)}")
        print(f"  Perfection threshold: {config.get('perfection_threshold', 0.95)} (retirement)")
        print(f"  Min replay prob: {config.get('min_replay_prob', 0.1)}")
        print(f"  Focus factor: {config.get('focus_factor', 1.5)}")

        # V12.2/V12.3: Compute curriculum weights (1-fraction, difficult elements, digit difficulty)
        curriculum_weights = None
        if config.get('use_curriculum_weights', False):
            training_formulas_for_curriculum = [formulas[i] for i in train_indices]
            print(f"  V12.3 Curriculum weighting:")
            curriculum_weights = compute_curriculum_weights(training_formulas_for_curriculum, config)
            n_boosted = np.sum(curriculum_weights > 1.0)
            print(f"    Total boosted: {n_boosted}/{len(curriculum_weights)} samples")
            print(f"    1-fraction boost: {config.get('one_fraction_boost', 2.0)}x")
            print(f"    Difficult element boost: {config.get('difficult_element_boost', 1.5)}x")
            print(f"    Digit difficulty boost: {config.get('digit_difficulty_boost', 1.5)}x per extra digit")

        if use_multitask:
            # V12.1: Multi-task mastery - tracks formula + Tc + Magpie separately
            mastery_tracker, mastery_sampler = create_multitask_mastery_components(
                n_samples=len(train_indices),
                window_size=config.get('mastery_window', 5),
                formula_threshold=config.get('mastery_threshold', 0.8),
                tc_threshold=config.get('tc_mastery_threshold', 0.8),
                magpie_threshold=config.get('magpie_mastery_threshold', 0.8),
                min_replay_prob=config.get('min_replay_prob', 0.1),
                focus_factor=config.get('focus_factor', 1.5),
                perfection_threshold=config.get('perfection_threshold', 0.95),
                base_weights=curriculum_weights,  # V12.2: Curriculum-based base weights
            )
        else:
            # Single-task mastery (formula only)
            mastery_tracker, mastery_sampler = create_mastery_aware_training_components(
                n_samples=len(train_indices),
                window_size=config.get('mastery_window', 5),
                mastery_threshold=config.get('mastery_threshold', 0.8),
                min_replay_prob=config.get('min_replay_prob', 0.1),
                focus_factor=config.get('focus_factor', 1.5),
                perfection_threshold=config.get('perfection_threshold', 0.95),
            )

        # V12.6: Build DataLoader kwargs conditionally (prefetch_factor requires num_workers > 0)
        num_workers = config.get('num_workers', 0)
        loader_kwargs = {
            'batch_size': config['batch_size'],
            'sampler': mastery_sampler,
            'num_workers': num_workers,
            'pin_memory': config.get('pin_memory', True),
        }
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = config.get('prefetch_factor', 4)
            loader_kwargs['persistent_workers'] = False

        train_loader = DataLoader(train_dataset, **loader_kwargs)
    else:
        # Standard random sampling
        num_workers = config.get('num_workers', 0)
        loader_kwargs = {
            'batch_size': config['batch_size'],
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': config.get('pin_memory', True),
        }
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = config.get('prefetch_factor', 4)
            loader_kwargs['persistent_workers'] = False

        train_loader = DataLoader(train_dataset, **loader_kwargs)

    # Store normalization stats for later
    norm_stats = {
        'tc_mean': tc_mean, 'tc_std': tc_std,
        'magpie_mean': magpie_mean.tolist(),
        'magpie_std': magpie_std.tolist(),
        'magpie_cols': magpie_cols,
    }

    # V12.3: Return training formulas and curriculum weights for digit error tracking
    training_formulas = [formulas[i] for i in train_indices]

    return (train_loader, target_cache, norm_stats, mastery_tracker, mastery_sampler,
            train_indices, training_formulas, curriculum_weights)


# ============================================================================
# MODEL LOADING - V12: FullMaterialsVAE + Enhanced Formula Decoder
# ============================================================================

def load_models(device, checkpoint_path, magpie_dim, is_v12_checkpoint=False):
    """Load FullMaterialsVAE + EnhancedTransformerDecoder.

    Args:
        device: GPU device
        checkpoint_path: Path to checkpoint file
        magpie_dim: Number of Magpie features
        is_v12_checkpoint: If True, load BOTH encoder and decoder from V12 checkpoint
                          If False, only load decoder (warm-start from V11)

    Returns:
        encoder, formula_decoder, resume_state (dict with epoch, best_train_exact if resuming)
    """
    print("=" * 60)
    print("STEP 3: Creating V12 Full Materials Architecture")
    print("=" * 60)

    # V12: Create FullMaterialsVAE (NEW encoder)
    print("\n  Creating FullMaterialsVAE...")
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

    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"    FullMaterialsVAE parameters: {encoder_params:,}")
    print(f"    Inputs: composition + Tc + {magpie_dim} Magpie features")
    print(f"    Outputs: z (2048) + Tc_pred + Magpie_pred + attended_input")

    # V12: Create formula decoder (warm-start from V11 if available)
    print("\n  Creating EnhancedTransformerDecoder...")
    formula_decoder = EnhancedTransformerDecoder(
        latent_dim=MODEL_CONFIG['latent_dim'],
        d_model=MODEL_CONFIG['d_model'],
        nhead=MODEL_CONFIG['nhead'],
        num_layers=MODEL_CONFIG['num_layers'],
        dim_feedforward=MODEL_CONFIG['dim_feedforward'],
        max_len=TRAIN_CONFIG['max_formula_len'],
        n_memory_tokens=MODEL_CONFIG['n_memory_tokens'],
        encoder_skip_dim=MODEL_CONFIG['fusion_dim'],  # attended_input dim
        use_skip_connection=True,
        # V12.4: Stoichiometry conditioning - decoder gets direct access to predicted fractions
        use_stoich_conditioning=True,
        max_elements=12,  # Maximum elements per formula
        n_stoich_tokens=4,  # 4 memory tokens for stoichiometry
        dropout=0.1
    ).to(device)

    # Resume state for continuing training from checkpoint
    resume_state = {
        'epoch': 0,
        'best_train_exact': 0.0,
        'best_train_acc': 0.0,
        'is_resume': False,
    }

    # Load checkpoint
    if checkpoint_path.exists():
        print(f"\n  Loading checkpoint from {checkpoint_path}...")
        # V12.5 FIX: Load checkpoint on CPU first to avoid WSL2 GPU memory residency issues
        # The dxg driver fails with ENOMEM when loading large checkpoints directly to GPU
        # because it needs to make both old and new tensors resident simultaneously
        torch.cuda.empty_cache()  # Clear any unused GPU memory first
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if is_v12_checkpoint:
            # V12.1: FULL RESUME - load both encoder and decoder
            print(f"    RESUMING from V12 checkpoint (epoch {checkpoint.get('epoch', '?')})")
            print(f"    Previous best exact match: {checkpoint.get('best_train_exact', 0)*100:.2f}%")

            # Load encoder state (V12.4/V12.28: use strict=False + shape-mismatch filtering)
            if 'encoder_state_dict' in checkpoint:
                enc_state = checkpoint['encoder_state_dict']

                # V12.28: Shape-mismatch handling with partial weight preservation
                model_state = encoder.state_dict()
                has_old_tc_head = any(k.startswith('tc_head.') for k in enc_state.keys())
                for key in list(enc_state.keys()):
                    if key in model_state and enc_state[key].shape != model_state[key].shape:
                        old_shape = enc_state[key].shape
                        new_shape = model_state[key].shape
                        preserved = False
                        if len(old_shape) == 2 and len(new_shape) == 2:
                            min_r, min_c = min(old_shape[0], new_shape[0]), min(old_shape[1], new_shape[1])
                            new_w = torch.zeros(new_shape, dtype=enc_state[key].dtype)
                            new_w[:min_r, :min_c] = enc_state[key][:min_r, :min_c]
                            enc_state[key] = new_w
                            preserved = True
                            print(f"    [Checkpoint] Partial preserve {key}: {old_shape}→{new_shape}, kept [{min_r},{min_c}]")
                        elif len(old_shape) == 1 and len(new_shape) == 1:
                            min_len = min(old_shape[0], new_shape[0])
                            new_b = torch.zeros(new_shape, dtype=enc_state[key].dtype)
                            new_b[:min_len] = enc_state[key][:min_len]
                            enc_state[key] = new_b
                            preserved = True
                            print(f"    [Checkpoint] Partial preserve {key}: {old_shape}→{new_shape}, kept [{min_len}]")
                        if not preserved:
                            print(f"    [Checkpoint] Shape mismatch for {key}: "
                                  f"checkpoint {old_shape} vs model {new_shape}, re-initializing")
                            del enc_state[key]

                missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
                if missing:
                    print(f"    Loaded encoder (new layers initialized fresh): {len(missing)} new params")
                    fraction_keys = [k for k in missing if 'fraction_head' in k]
                    if len(fraction_keys) == len(missing):
                        print(f"    ✓ V12.4: FractionValueHead initialized (stoichiometry-aware learning)")
                    else:
                        print(f"    Other missing keys: {[k for k in missing if 'fraction_head' not in k]}")
                else:
                    print(f"    Loaded encoder state dict (all layers)")

                # V12.28: Net2Net weight transfer for old tc_head → new tc_proj/tc_res_block/tc_out
                if has_old_tc_head and hasattr(encoder, 'upgrade_tc_head_from_checkpoint'):
                    encoder.upgrade_tc_head_from_checkpoint(enc_state)
                    print("    [Checkpoint] Applied Net2Net weight transfer for Tc head upgrade")
            else:
                print(f"    WARNING: No encoder state in checkpoint, using fresh weights")

            # Load decoder state (V12.4: use strict=False for new stoich_to_memory)
            if 'decoder_state_dict' in checkpoint:
                missing, unexpected = formula_decoder.load_state_dict(
                    checkpoint['decoder_state_dict'], strict=False
                )
                if missing:
                    print(f"    Loaded decoder (new layers initialized fresh): {len(missing)} new params")
                    stoich_keys = [k for k in missing if 'stoich_to_memory' in k]
                    if len(stoich_keys) == len(missing):
                        print(f"    ✓ V12.4: Stoichiometry memory tokens initialized (decoder integration)")
                    else:
                        print(f"    Other missing keys: {[k for k in missing if 'stoich_to_memory' not in k]}")
                else:
                    print(f"    Loaded decoder state dict (all layers)")
            else:
                print(f"    WARNING: No decoder state in checkpoint, using fresh weights")

            # CRITICAL: Preserve best_train_exact so we don't overwrite with worse models
            resume_state['epoch'] = checkpoint.get('epoch', 0)
            resume_state['best_train_exact'] = checkpoint.get('best_train_exact', 0.0)
            resume_state['best_train_acc'] = checkpoint.get('best_train_acc', 0.0)
            resume_state['is_resume'] = True
            # V12.5: Preserve mastery tracker state for proper sample distribution on resume
            if 'mastery_tracker_state' in checkpoint:
                resume_state['mastery_tracker_state'] = checkpoint['mastery_tracker_state']
                print(f"    ✓ Loaded mastery tracker state (epoch {checkpoint['mastery_tracker_state'].get('current_epoch', '?')})")

        else:
            # Warm-start: only load decoder (for V11/V10 checkpoints)
            # V12.5 FIX: Use direct load_state_dict with strict=False instead of manual tensor copying
            # Manual copying was causing CUDA driver assertion failures
            old_decoder_state = checkpoint.get('decoder_state_dict', {})
            missing, unexpected = formula_decoder.load_state_dict(old_decoder_state, strict=False)
            transferred = len(old_decoder_state) - len(unexpected)
            print(f"    Transferred {transferred} weight tensors from checkpoint (decoder only)")
            if missing:
                print(f"    Note: {len(missing)} keys not in checkpoint (using random init)")
    else:
        print(f"    No checkpoint found, using fresh weights")

    decoder_params = sum(p.numel() for p in formula_decoder.parameters())
    print(f"    Formula decoder parameters: {decoder_params:,}")

    print(f"\nTotal model parameters: {encoder_params + decoder_params:,}")

    return encoder, formula_decoder, resume_state


# ============================================================================
# TRAINING - V12: Full Materials
# ============================================================================

def train_epoch(encoder, formula_decoder, train_loader, loss_fn,
                encoder_optimizer, decoder_optimizer, device, config,
                scaler, tf_ratio=0.5, track_mastery=False, mastery_tracker=None):
    """Train for one epoch with V12: full materials + formula reconstruction.

    OPTIMIZATIONS:
    - Mixed precision (FP16) via autocast + GradScaler
    - Larger batches enabled by reduced memory usage
    - Per-batch adaptive teacher forcing based on mastery

    Returns:
        metrics dict, and if track_mastery=True: (sample_indices, exact_matches)
    """
    encoder.train()
    formula_decoder.train()

    total_loss = 0
    total_reward = 0
    total_acc = 0
    total_exact = 0
    total_tc_loss = 0
    total_magpie_loss = 0
    total_stoich_loss = 0  # V12.4: Stoichiometry loss
    total_tf_used = 0  # Track actual TF used
    n_batches = 0

    use_amp = config.get('use_amp', True)
    use_per_sample_tf = config.get('tf_per_sample', False) and mastery_tracker is not None

    # For mastery tracking (V12.1: multi-task)
    all_sample_indices = []
    all_exact_matches = []      # Formula exact match
    all_tc_correct = []         # Tc prediction correct (within threshold)
    all_magpie_correct = []     # Magpie prediction correct (within threshold)

    # V12.1: Thresholds for Tc/Magpie "correctness"
    tc_error_threshold = config.get('tc_error_threshold', 0.1)       # 10% relative error
    magpie_error_threshold = config.get('magpie_error_threshold', 0.2)  # MSE < 0.2

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    import time
    batch_times = []

    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()
        elem_idx, elem_frac, elem_mask, tokens, tc, magpie = [b.to(device) for b in batch]

        # PER-BATCH ADAPTIVE TEACHER FORCING based on sample mastery
        batch_tf_ratio = tf_ratio  # Default to global TF
        if use_per_sample_tf:
            # Get batch sample indices (from sampler's iteration order)
            batch_start = batch_idx * config['batch_size']
            batch_size_actual = len(tokens)

            # Compute batch-average mastery → adaptive TF
            mastery_scores = mastery_tracker.get_mastery_scores()
            # Note: sampler iterates through weighted indices, we use batch position
            # For simplicity, use average mastery of recently seen samples
            avg_mastery = mastery_scores.mean() if len(mastery_scores) > 0 else 0.5

            # Map average mastery to TF: lower mastery → higher TF
            if avg_mastery < 0.5:
                batch_tf_ratio = config.get('tf_struggling', 0.4)
            elif avg_mastery < 0.8:
                batch_tf_ratio = config.get('tf_medium', 0.15)
            else:
                batch_tf_ratio = config.get('tf_mastered', 0.0)

        # OPTIMIZATION: Mixed precision forward pass
        # V12.6: Always use AMP - the V12.5 workaround was causing epoch boundary crashes
        with autocast(enabled=use_amp):
            # V12: Full materials encoding
            encoder_out = encoder(elem_idx, elem_frac, elem_mask, magpie, tc)
            z = encoder_out['z']
            tc_pred = encoder_out['tc_pred']
            magpie_pred = encoder_out['magpie_pred']
            attended_input = encoder_out['attended_input']
            kl_loss = encoder_out['kl_loss']

            # V12.4: Stoichiometry prediction from encoder
            fraction_pred = encoder_out.get('fraction_pred')
            element_count_pred = encoder_out.get('element_count_pred')

            # V12.4: Create combined stoichiometry conditioning for decoder
            # Concatenate fraction predictions with element count for decoder memory
            if fraction_pred is not None and element_count_pred is not None:
                stoich_pred = torch.cat([
                    fraction_pred,
                    element_count_pred.unsqueeze(-1)
                ], dim=-1)  # [batch, max_elements + 1]
            else:
                stoich_pred = None

            # Formula decoding with skip connection and stoichiometry conditioning
            # V12.5: Use batch_tf_ratio for scheduled sampling
            formula_logits, generated, *_extra = formula_decoder(
                z, tokens, encoder_skip=attended_input, teacher_forcing_ratio=batch_tf_ratio,
                stoich_pred=stoich_pred  # V12.4: Pass stoichiometry to decoder
            )
            formula_targets = tokens[:, 1:]

            # V12 combined loss (with V12.4 stoichiometry loss)
            loss_dict = loss_fn(
                formula_logits=formula_logits,
                formula_targets=formula_targets,
                tc_pred=tc_pred,
                tc_true=tc,
                magpie_pred=magpie_pred,
                magpie_true=magpie,
                kl_loss=kl_loss,
                # V12.4: Stoichiometry loss inputs
                fraction_pred=fraction_pred,
                element_fractions=elem_frac,
                element_mask=elem_mask,
                element_count_pred=element_count_pred,
                # V12.7: Autoregressive REINFORCE inputs
                z=z,
                encoder_skip=attended_input,
                stoich_pred_for_reinforce=stoich_pred,
            )

            loss = loss_dict['total'] / config['accumulation_steps']

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss at batch {batch_idx}, skipping")
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            continue

        # OPTIMIZATION: Scaled backward pass for FP16
        scaler.scale(loss).backward()

        if (batch_idx + 1) % config['accumulation_steps'] == 0:
            # Unscale before clipping
            scaler.unscale_(encoder_optimizer)
            scaler.unscale_(decoder_optimizer)

            encoder_grad = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            decoder_grad = torch.nn.utils.clip_grad_norm_(formula_decoder.parameters(), 1.0)

            if torch.isnan(encoder_grad) or torch.isnan(decoder_grad):
                print(f"  WARNING: NaN gradients at batch {batch_idx}, skipping")
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                scaler.update()
                continue

            # OPTIMIZATION: Scaled optimizer step
            scaler.step(encoder_optimizer)
            scaler.step(decoder_optimizer)
            scaler.update()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

        total_loss += loss_dict['total'].item()
        total_reward += loss_dict['mean_reward'].item()
        total_acc += loss_dict['token_accuracy'].item()
        total_exact += loss_dict['exact_match'].item()
        total_tc_loss += loss_dict['tc_loss'].item()
        total_magpie_loss += loss_dict['magpie_loss'].item()
        # V12.4: Track stoichiometry loss
        total_stoich_loss += loss_dict['stoich_loss'].item()
        # V12.5: Track actual TF used (for logging)
        total_tf_used += batch_tf_ratio
        n_batches += 1

        # TIMING DEBUG: Track batch time
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        if batch_idx < 5 or batch_idx % 100 == 0:
            print(f"  [Batch {batch_idx}: {batch_time:.2f}s]", flush=True)

        # Track per-sample mastery (V12.1: multi-task)
        if track_mastery:
            with torch.no_grad():
                # 1. Formula exact match
                predictions = formula_logits.argmax(dim=-1)
                formula_targets_batch = tokens[:, 1:]
                mask = formula_targets_batch != PAD_IDX
                per_sample_exact = ((predictions == formula_targets_batch) | ~mask).all(dim=1)
                all_exact_matches.append(per_sample_exact.cpu())

                # 2. Tc prediction correctness (relative error < threshold)
                tc_relative_error = torch.abs(tc_pred.squeeze() - tc) / (torch.abs(tc) + 1e-8)
                per_sample_tc_correct = tc_relative_error < tc_error_threshold
                all_tc_correct.append(per_sample_tc_correct.cpu())

                # 3. Magpie prediction correctness (MSE per sample < threshold)
                magpie_mse = ((magpie_pred - magpie) ** 2).mean(dim=1)
                per_sample_magpie_correct = magpie_mse < magpie_error_threshold
                all_magpie_correct.append(per_sample_magpie_correct.cpu())

                # Track sample indices
                all_sample_indices.append(torch.arange(batch_idx * config['batch_size'],
                                                       batch_idx * config['batch_size'] + len(tokens)))

    metrics = {
        'loss': total_loss / n_batches,
        'reward': total_reward / n_batches,
        'accuracy': total_acc / n_batches,
        'exact_match': total_exact / n_batches,
        'tc_loss': total_tc_loss / n_batches,
        'stoich_loss': total_stoich_loss / n_batches,  # V12.4
        'magpie_loss': total_magpie_loss / n_batches,
        'avg_tf': total_tf_used / n_batches,  # V12.5: Actual TF used
    }

    if track_mastery and all_exact_matches:
        # V12.1: Return multi-task mastery data
        mastery_data = {
            'sample_indices': torch.cat(all_sample_indices),
            'formula_correct': torch.cat(all_exact_matches),
            'tc_correct': torch.cat(all_tc_correct),
            'magpie_correct': torch.cat(all_magpie_correct),
        }
        return metrics, mastery_data
    return metrics


def get_curriculum_weights(epoch: int, config: dict) -> Tuple[float, float]:
    """
    Curriculum learning for loss weights - STRONGER SIGNALS.

    Phase 1 (epochs 0-30): Ramp up from moderate to full
        - tc_weight: 5.0 → 10.0 (meaningful from start!)
        - magpie_weight: 1.0 → 2.0

    Phase 2 (epochs 30+): Full strength
        - tc_weight: 10.0
        - magpie_weight: 2.0

    Key insight: Tc/Magpie must contribute ~20-30% of total gradient
    to properly influence encoder learning. Previous weights (0.1, 0.01)
    were contributing <2% - essentially invisible to optimizer.
    """
    phase1_end = 30

    if epoch < phase1_end:
        # Phase 1: Ramp up from moderate to full (not from near-zero!)
        progress = epoch / phase1_end
        tc_weight = 5.0 + 5.0 * progress       # 5.0 → 10.0
        magpie_weight = 1.0 + 1.0 * progress   # 1.0 → 2.0
    else:
        # Phase 2+: Full strength
        tc_weight = config['tc_weight']        # 10.0
        magpie_weight = config['magpie_weight']  # 2.0

    return tc_weight, magpie_weight


def train_v12(encoder, formula_decoder, train_loader, target_cache, device, config, output_dir, norm_stats,
              mastery_tracker=None, mastery_sampler=None, resume_state=None,
              target_formulas=None, curriculum_weights=None):
    """V12 training loop: FULL MATERIALS + FORMULA RECONSTRUCTION.

    OPTIMIZATIONS:
    - Mixed precision (FP16) for ~2x speedup
    - Larger batch sizes (32 vs 16)
    - Multi-worker data loading (4 workers)
    - Mastery-aware sampling (focus on struggling samples)

    CURRICULUM:
    - Phase 1: Focus on formula (low Tc/Magpie weight)
    - Phase 2: Balance all reconstruction targets
    - Phase 3: Full reconstruction strength

    V12.3 ADDITIONS:
    - Digit error tracking: Periodically scan for digit errors and boost their weight
    - target_formulas: List of target formula strings for error classification
    - curriculum_weights: Base curriculum weights to combine with digit error boosts
    """
    use_mastery = mastery_tracker is not None and mastery_sampler is not None

    # V12.3: Initialize digit error tracking
    use_digit_error_boost = config.get('use_error_specific_boost', False) and target_formulas is not None
    digit_error_tracker = None
    if use_digit_error_boost:
        n_train = len(target_formulas)
        digit_error_tracker = DigitErrorTracker(
            n_samples=n_train,
            digit_error_boost=config.get('digit_error_boost', 5.0),
            decay=config.get('error_boost_decay', 0.8)
        )
        print(f"\n*** V12.3: Digit Error Tracking Enabled ***")
        print(f"  - Digit error boost: {config.get('digit_error_boost', 5.0)}x")
        print(f"  - Error boost decay: {config.get('error_boost_decay', 0.8)} per epoch")
        print(f"  - Scan interval: every 25 epochs")

    print("\n" + "=" * 60)
    print("STEP 4: V12 Training Setup - FULL MATERIALS (OPTIMIZED)")
    print("=" * 60)

    print("\n*** V12 KEY CHANGES ***")
    print(f"  - FULL ENCODER: composition + Tc + {len(norm_stats['magpie_cols'])} Magpie features")
    print(f"  - MULTI-HEAD DECODER: formula + Tc + Magpie reconstruction")
    print(f"  - Latent dim: {MODEL_CONFIG['latent_dim']}")

    print("\n*** OPTIMIZATIONS ***")
    print(f"  - Mixed Precision (FP16): {config.get('use_amp', True)}")
    print(f"  - Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['accumulation_steps']})")
    print(f"  - Data workers: {config.get('num_workers', 4)}")

    print("\n*** CURRICULUM ***")
    print(f"  - Phase 1 (0-50): Ramp up Tc/Magpie weights")
    print(f"  - Phase 2 (50-200): Balanced reconstruction")
    print(f"  - Phase 3 (200+): Full strength")

    # Register models for graceful shutdown (saves latest_model.pt on interrupt)
    _shutdown_state['encoder'] = encoder
    _shutdown_state['decoder'] = formula_decoder
    _shutdown_state['output_dir'] = output_dir
    _shutdown_state['norm_stats'] = norm_stats

    # Create V12 loss function (weights will be updated per epoch)
    loss_fn = REINFORCELossV12(
        ce_weight=config['ce_weight'],
        rl_weight=config['rl_weight'],
        tc_weight=config['tc_weight'],
        magpie_weight=config['magpie_weight'],
        kl_weight=config['kl_weight'],
        entropy_weight=config['entropy_weight'],
        n_samples_rloo=config['n_samples_rloo'],
        temperature=config['temperature'],
        target_cache=target_cache,
        # V12.6: Focal loss parameters
        use_focal_loss=config.get('use_focal_loss', True),
        focal_gamma=config.get('focal_gamma', 2.0),
        label_smoothing=config.get('label_smoothing', 0.1),
        # Autoregressive REINFORCE with KV caching
        use_autoregressive_reinforce=config.get('use_autoregressive_reinforce', False),
        rl_method=config.get('rl_method', 'scst'),
    )

    # Set decoder for autoregressive REINFORCE/SCST (if enabled)
    if config.get('use_autoregressive_reinforce', False):
        loss_fn.set_decoder(formula_decoder, max_len=config['max_formula_len'])
        rl_method = config.get('rl_method', 'scst')
        print(f"\n*** AUTOREGRESSIVE RL ENABLED ***")
        print(f"  Method: {rl_method.upper()} ({'Self-Critical Sequence Training' if rl_method == 'scst' else 'RLOO Leave-One-Out'})")
        print(f"  Max sequence length: {config['max_formula_len']}")
        if rl_method == 'scst':
            print(f"  Baseline: greedy decode reward (Rennie et al. 2017)")
            print(f"  Forward passes per batch: 2 (1 greedy + 1 sample)")
        else:
            print(f"  Baseline: leave-one-out (Ahmadian et al. 2024)")
            print(f"  Forward passes per batch: {config['n_samples_rloo']} samples")

    # OPTIMIZATION: Mixed precision scaler
    scaler = GradScaler(enabled=config.get('use_amp', True))

    encoder_optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=config['encoder_lr'], weight_decay=0.01
    )
    decoder_optimizer = torch.optim.AdamW(
        formula_decoder.parameters(), lr=config['decoder_lr'], weight_decay=0.01
    )

    encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        encoder_optimizer, T_max=config['num_epochs'], eta_min=config['encoder_lr'] * 0.1
    )
    decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        decoder_optimizer, T_max=config['num_epochs'], eta_min=config['decoder_lr'] * 0.1
    )

    print(f"\nTraining config:")
    print(f"  Formula weight: {config['formula_weight']}")
    print(f"  Tc weight: {config['tc_weight']}")
    print(f"  Magpie weight: {config['magpie_weight']}")
    print(f"  KL weight: {config['kl_weight']}")
    print(f"  Epochs: {config['num_epochs']}")

    # V12.6: Log focal loss settings
    if config.get('use_focal_loss', True):
        print(f"\n*** V12.6: FOCAL LOSS ENABLED ***")
        print(f"  Gamma: {config.get('focal_gamma', 2.0)} (0=CE, 2=typical focal)")
        print(f"  Label smoothing: {config.get('label_smoothing', 0.1)}")
        print(f"  Purpose: Focus gradient on the ~1% hard tokens to break plateau")
    else:
        print(f"\n  Using standard CrossEntropyLoss (focal loss disabled)")

    history = {
        'train_loss': [], 'train_reward': [], 'train_exact': [],
        'train_acc': [], 'train_tc_loss': [], 'train_magpie_loss': [],
        'tf_ratio': []
    }

    # CRITICAL: When resuming, preserve best_train_exact to avoid overwriting better models
    # AND resume from the saved epoch (not epoch 0)
    start_epoch = 0
    if resume_state and resume_state.get('is_resume', False):
        best_train_exact = resume_state.get('best_train_exact', 0.0)
        best_train_acc = resume_state.get('best_train_acc', 0.0)
        start_epoch = resume_state.get('epoch', 0) + 1  # Resume from NEXT epoch
        print(f"\n*** RESUME: Preserving best_train_exact={best_train_exact*100:.2f}% from checkpoint ***")
        print(f"*** Resuming from epoch {start_epoch} (checkpoint was at epoch {start_epoch-1}) ***")
        print(f"*** Only models BETTER than {best_train_exact*100:.2f}% will be saved as best_model.pt ***\n")
    else:
        best_train_exact = 0
        best_train_acc = 0
    prev_exact = best_train_exact  # Start with saved accuracy for TF adaptive

    print("\n" + "=" * 60)
    print("STEP 5: Training (V12 - Full Materials)")
    print("=" * 60)
    sys.stdout.flush()

    for epoch in range(start_epoch, config['num_epochs']):
        # CURRICULUM: Update loss weights based on epoch
        tc_weight, magpie_weight = get_curriculum_weights(epoch, config)
        loss_fn.tc_weight = tc_weight
        loss_fn.magpie_weight = magpie_weight

        # Teacher forcing decay
        base_tf = max(
            config['tf_end'],
            config['tf_start'] - (config['tf_start'] - config['tf_end']) * epoch / config['tf_decay_epochs']
        )

        if config.get('tf_adaptive', False) and prev_exact > 0:
            adaptive_reduction = prev_exact * (base_tf - config['tf_end'])
            tf_ratio = max(config['tf_end'], base_tf - adaptive_reduction)
        else:
            tf_ratio = base_tf

        # Train epoch with optional mastery tracking
        epoch_result = train_epoch(
            encoder, formula_decoder, train_loader, loss_fn,
            encoder_optimizer, decoder_optimizer, device, config,
            scaler, tf_ratio=tf_ratio, track_mastery=use_mastery
        )

        # Unpack results
        if use_mastery and isinstance(epoch_result, tuple):
            train_metrics, mastery_data = epoch_result

            # V12.1: Update mastery tracker (multi-task or single-task)
            use_multitask = config.get('use_multitask_mastery', False)
            if use_multitask and hasattr(mastery_tracker, 'tc_history'):
                # Multi-task tracker: pass all three correctness signals
                mastery_stats = mastery_tracker.update_epoch(
                    mastery_data['sample_indices'],
                    mastery_data['formula_correct'],
                    mastery_data['tc_correct'],
                    mastery_data['magpie_correct']
                )
            else:
                # Single-task tracker: only formula exact match
                mastery_stats = mastery_tracker.update_epoch(
                    mastery_data['sample_indices'],
                    mastery_data['formula_correct']
                )
            mastery_sampler.update_weights()

            # V12.3: Periodic digit error scanning and weight update
            if use_digit_error_boost and (epoch + 1) % 25 == 0 and epoch > 0:
                print(f"\n  [V12.3: Scanning for digit errors at epoch {epoch}...]")
                scan_stats = scan_for_digit_errors(
                    encoder, formula_decoder, train_loader, target_formulas,
                    device, config, digit_error_tracker, max_samples=5000
                )
                # Update sampler base weights with digit error boosts
                if curriculum_weights is not None:
                    new_base_weights = digit_error_tracker.apply_to_curriculum_weights(curriculum_weights)
                    mastery_sampler.base_weights = new_base_weights
                    mastery_sampler.update_weights()
                print(f"    Scanned: {scan_stats['n_scanned']}, "
                      f"Digit errors: {scan_stats['n_digit_errors']} ({scan_stats['digit_error_rate']*100:.1f}%), "
                      f"Boosted samples: {scan_stats['n_boosted_samples']}")
        else:
            train_metrics = epoch_result
            mastery_stats = None

        encoder_scheduler.step()
        decoder_scheduler.step()

        history['train_loss'].append(train_metrics['loss'])
        history['train_reward'].append(train_metrics['reward'])
        history['train_exact'].append(train_metrics['exact_match'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_tc_loss'].append(train_metrics['tc_loss'])
        history['train_magpie_loss'].append(train_metrics['magpie_loss'])
        history['tf_ratio'].append(tf_ratio)

        # Update shutdown state for graceful interrupt saving
        _shutdown_state['epoch'] = epoch
        _shutdown_state['history'] = history
        _shutdown_state['best_train_exact'] = max(best_train_exact, train_metrics['exact_match'])

        prev_exact = train_metrics['exact_match']

        if epoch % 5 == 0 or epoch < 20:
            # V12.5: Show actual avg TF used (may differ from target due to per-sample adaptation)
            actual_tf = train_metrics.get('avg_tf', tf_ratio)
            log_str = (f"Epoch {epoch:4d} | TF: {actual_tf:.2f} | Loss: {train_metrics['loss']:.4f} | "
                       f"Reward: {train_metrics['reward']:.1f} | "
                       f"Acc: {train_metrics['accuracy']*100:.1f}% | "
                       f"Exact: {train_metrics['exact_match']*100:.1f}% | "
                       f"Tc: {train_metrics['tc_loss']:.4f} | "
                       f"Magpie: {train_metrics['magpie_loss']:.4f} | "
                       f"Stoich: {train_metrics['stoich_loss']:.4f}")
            if mastery_stats is not None:
                log_str += f" | Mastered: {mastery_stats.n_mastered}"
            print(log_str)
            sys.stdout.flush()

        if train_metrics['exact_match'] > best_train_exact:
            best_train_exact = train_metrics['exact_match']
            best_train_acc = train_metrics['accuracy']

            save_dict = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': formula_decoder.state_dict(),
                'best_train_acc': best_train_acc,
                'best_train_exact': best_train_exact,
                'history': history,
                'config': {**MODEL_CONFIG, **config},
                'norm_stats': norm_stats,
            }
            # V12.5: Save mastery tracker state for proper resume
            if mastery_tracker is not None:
                save_dict['mastery_tracker_state'] = mastery_tracker.state_dict()
            torch.save(save_dict, output_dir / 'best_model.pt')
            print(f"  -> New best! Train Exact: {best_train_exact*100:.2f}%, Acc: {best_train_acc*100:.2f}%")
            sys.stdout.flush()

        if (epoch + 1) % config['checkpoint_interval'] == 0:
            ckpt_dict = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': formula_decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'best_train_exact': best_train_exact,
                'history': history,
                'norm_stats': norm_stats,
            }
            # V12.5: Save mastery tracker state for proper resume
            if mastery_tracker is not None:
                ckpt_dict['mastery_tracker_state'] = mastery_tracker.state_dict()
            torch.save(ckpt_dict, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
            print(f"  [Checkpoint saved at epoch {epoch+1}]")
            sys.stdout.flush()

            # Cleanup old checkpoints to save disk space (keep last 3)
            _cleanup_old_checkpoints(output_dir, keep_last_n=config.get('keep_last_n_checkpoints', 3))

            with open(output_dir / 'history.json', 'w') as f:
                json.dump(history, f)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best train exact match: {best_train_exact*100:.2f}%")

    # Save final latest_model.pt on normal completion
    _save_latest_model("training_complete")

    return history


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    print(f"\nV12 Training - FULL MATERIALS (composition + Tc + Magpie)")
    print(f"Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full materials data (V12.5: single df with fraction notation)
    df, holdout_indices, magpie_cols = load_data()
    (train_loader, target_cache, norm_stats, mastery_tracker, mastery_sampler,
     train_indices, training_formulas, curriculum_weights) = prepare_data(
        df, holdout_indices, magpie_cols, TRAIN_CONFIG
    )

    # Check for resume checkpoint
    is_v12_checkpoint = False
    if RESUME_CHECKPOINT is not None and Path(RESUME_CHECKPOINT).exists():
        checkpoint_path = Path(RESUME_CHECKPOINT)
        is_v12_checkpoint = True  # Full resume
        print(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path('/nonexistent')  # Will use fresh weights
        print(f"Training from scratch (no checkpoint)")

    # Create models (with full resume if V12 checkpoint)
    encoder, formula_decoder, resume_state = load_models(device, checkpoint_path, len(magpie_cols), is_v12_checkpoint)

    # V12.5: Restore mastery tracker state if available (for proper sample distribution on resume)
    if mastery_tracker is not None and resume_state.get('mastery_tracker_state'):
        try:
            mastery_tracker.load_state_dict(resume_state['mastery_tracker_state'])
            mastery_sampler.update_weights()  # Recompute sampling weights based on restored mastery
            print(f"✓ Restored mastery tracker state - {sum(mastery_tracker.was_mastered)} mastered samples")
        except Exception as e:
            print(f"WARNING: Could not restore mastery tracker state: {e}")
            print("  Continuing with fresh mastery tracking...")

    # Save normalization stats
    with open(OUTPUT_DIR / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)

    # Train with mastery-aware sampling (V12.3: with digit error tracking)
    history = train_v12(
        encoder, formula_decoder, train_loader, target_cache,
        device, TRAIN_CONFIG, OUTPUT_DIR, norm_stats,
        mastery_tracker=mastery_tracker, mastery_sampler=mastery_sampler,
        resume_state=resume_state,
        target_formulas=training_formulas,
        curriculum_weights=curriculum_weights
    )


if __name__ == "__main__":
    main()
