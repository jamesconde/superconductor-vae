# Training Records

Chronological record of training runs, architecture changes, and optimization decisions for the Multi-Task Superconductor Generator (code class names retain "VAE" for backward compatibility).

---

## V12.21: SC Classification Head + Diagnostic Fixes (2026-02-09)

### SC/non-SC Classification Head (Cross-Head Consistency)

The model previously had no explicit SC/non-SC classifier. The contrastive loss shapes latent geometry but doesn't predict SC status. The competence head existed but was untrained (no loss function).

**New SC head** added to `FullMaterialsVAE`:
- 3-layer MLP: `Linear(2209, 512) → GELU → LayerNorm → Dropout → Linear(512, 128) → GELU → Linear(128, 1)`
- **Cross-head consistency input** (2209 dims): z(2048) + tc_pred(1) + magpie_pred(145) + hp_pred(1) + fraction_pred(12) + element_count_pred(1) + competence(1)
- Trained on ALL samples (SC + non-SC) with BCE loss, weight `sc_loss_weight: 0.5`
- Learns patterns like "high Tc prediction + SC-like Magpie profile → superconductor"

### Cache Tc MAE Fix

Cache `tc_pred_mae` was inflated because it averaged over non-SC samples (where Tc head is untrained, producing ~2.6 normalized error each). Fixed to report:
- **SC-only MAE** (primary metric)
- Non-SC MAE (for monitoring)
- **Kelvin breakdown** by Tc range: 0-10K, 10-50K, 50-100K, 100K+ (requires norm_stats)
- Same SC-only breakdown added to `evaluate_true_autoregressive()` z-diagnostics

### CSV Logging Key Mismatch Fix

The `log_training_metrics()` function was looking for `'rl_loss'` and `'reward'` in the metrics dict, but `train_epoch()` returns `'reinforce_loss'` and `'mean_reward'`. This caused RL loss and reward to always be written as 0 in `training_log.csv`. Also added `entropy_weight` and `tf_ratio` to CSV output.

---

## V12.20: Normalization Audit + Loss Function Fixes (2026-02-09)

### Problem

Data analysis revealed the raw Tc distribution (skewness=2.18, range 0-1103K) caused severe MSE pathologies:

| Issue | Impact |
|-------|--------|
| **Outlier dominance** | Top 1% of samples contribute 93.5% of total Tc MSE |
| **Heteroscedastic errors** | Error magnitude correlates with Tc (r=0.53) |
| **Systematic underprediction** | 99.1% of materials above 100K are underpredicted |
| **Gradient domination** | Single outlier at 1103K has 641K error, dominates gradients |

### Changes

1. **Log-transform of Tc** (`tc_log_transform: True` in TRAIN_CONFIG)
   - Applied `np.log1p(Tc)` before z-score normalization
   - Reduces skewness from 2.18 to -0.17 (near-Gaussian)
   - Compresses dynamic range from 1103 to 7.01 (157x)
   - Stored in `norm_stats.json` so inference can denormalize correctly: `Tc_K = expm1(z * std + mean)`

2. **Huber loss for Tc** (`tc_huber_delta: 1.0` in TRAIN_CONFIG)
   - Replaces MSE with Huber loss (δ=1.0 in normalized space)
   - Quadratic for small errors (|e| < 1 std), linear for large errors
   - At |error|=10σ: 20x gradient reduction vs MSE
   - Prevents outlier samples from hijacking encoder gradients

3. **Magpie: Quantile transform for skewed features** (`magpie_skew_threshold: 3.0`)
   - 18 of 145 Magpie features had extreme skewness (|skew| > 3), mostly zero-inflated (94-100% zeros)
   - Z-score is meaningless for features like `minimum NfUnfilled` (100% zeros, skew=56)
   - Applied rank-based Gaussian transform: `rank → uniform(0,1) → Φ⁻¹ → N(0,1)`
   - Jitter (ε=1e-6, seed=42) breaks ties in zero-inflated features
   - All 18 features now |skew| < 1.0 (most ≈ 0.00)
   - Feature indices stored in `norm_stats.json` for inference reproducibility

4. **Magpie: SC-only normalization** (`magpie_sc_only_norm: True`)
   - Previously used all-sample (SC + non-SC) mean/std — biased by non-SC distribution
   - Top Magpie features show significant SC/non-SC distributional shift (KS up to 0.36)
   - Now uses SC-only stats: SC samples centered at z=0, non-SC offset (as expected)
   - 44/145 features had mean shift > 0.1σ between SC-only and all-sample stats

5. **Cache invalidation**
   - `tc_log_transform`, `magpie_skew_threshold`, `magpie_sc_only_norm` all stored in cache metadata
   - Changing any flag auto-invalidates the tensor cache (forces recomputation)

### Backward Compatibility

- `tc_log_transform: False` + `tc_huber_delta: 0.0` + `magpie_skew_threshold: 0.0` + `magpie_sc_only_norm: False` recovers exact V12.19 behavior
- Existing checkpoints resume normally (normalization changes don't affect model weights — but the Tc/Magpie heads will see shifted inputs, so expect a few epochs of readjustment)
- `norm_stats.json` includes all transform flags for correct inference

### Files Changed

| File | Change |
|------|--------|
| `scripts/train_v12_clean.py` | TRAIN_CONFIG flags, log1p Tc, Huber loss, quantile transform for skewed Magpie, SC-only Magpie stats, cache invalidation |
| `src/superconductor/models/attention_vae.py` | FullMaterialsLoss updated for consistency (not used in training) |

### Expected Impact

- More uniform gradient signal across Tc range (low-Tc and high-Tc contribute equally)
- Better prediction of extreme-Tc superconductors (no longer systematically underpredicted)
- Faster convergence (no wasted gradient budget on irreducible outlier error)
- Magpie reconstruction: skewed features no longer dominated by zero-class; network can learn the non-zero minority
- Tc MAE may initially appear different due to log-space normalization — compare in Kelvin after denormalization

---

## V12.19: High-Pressure Superconductor Labeling & Prediction (2026-02-09)

### Summary

Added high-pressure superconductor (HP-SC) identification, labeling, and prediction to the training pipeline.

### Changes

**New file: `scripts/label_high_pressure.py`**
- Reads NEMAD source CSV pressure column (dropped during original preprocessing)
- Parses free-text pressure values (GPa, kbar, MPa, ambient markers)
- Joins with processed dataset by normalized formula
- Supplements with known HP categories:
  - Hydrogen-rich SC (45, already labeled category)
  - Elemental HP-SC (Ca, Li, S, P etc. with anomalously high Tc)
  - Fullerene HP-SC (Cs3C60 A15 phase, Tc > 33K)
  - Nickelate HP-SC (La/Nd-Ni-O without Cu, Tc > 30K)
- Writes `requires_high_pressure` column (0/1) to CSV
- Generates diagnostic report at `outputs/high_pressure_labeling_report.json`
- **Result: 116 HP labels (0.5% of SC)**: 45 hydrides, 43 NEMAD-matched, 12 elemental, 5 fullerene, 3 nickelate

**Modified: `src/superconductor/models/attention_vae.py`**
- Added `hp_head`: z(2048) -> Linear(2048,256) -> ReLU -> Linear(256,1)
- Returns logits in `hp_pred` key (apply sigmoid for probability)
- Backward compatible: missing keys init fresh when loading old checkpoints

**Modified: `scripts/train_v12_clean.py`**
- `TRAIN_CONFIG['hp_loss_weight']`: 0.05 (BCE with pos_weight for class imbalance)
- `load_and_prepare_data()`: reads `requires_high_pressure`, creates 9th tensor
- `train_epoch()`: unpacks 9 tensors, computes HP BCE loss on SC samples only
- `log_training_metrics()`: logs `hp_loss` column to training_log.csv
- `requires_high_pressure` excluded from Magpie feature columns
- Cache backward compatible (defaults to 0 if hp_tensor missing)

**Modified: `src/superconductor/losses/contrastive.py`**
- Added class 12: `'High-pressure (non-hydride)': 12`
- `category_to_label()` accepts `requires_high_pressure` parameter
- Non-hydride HP-SC → class 12; hydride HP-SC stays class 5
- Total: 13 contrastive classes (was 12)

### HP Loss Design

- Binary cross-entropy with logits on SC samples only
- `pos_weight = n_neg / n_pos` (capped at 50x) for ~100:1 class imbalance
- Weight: 0.05 (intentionally small — auxiliary prediction, not primary objective)
- Non-SC samples excluded (trivially not HP)

---

## Run 6: Training Resumed After NaN Fix (2026-02-03)

**Status**: Running
**Script**: `scripts/train_v12_clean.py` V12.14 (NaN gradient guards)
**Resume from**: checkpoint_final.pt → checkpoint_best.pt (epoch 1354, exact=28.0%)
**Log**: `outputs/training_deterministic_overnight_v4.log`
**GPU**: RTX 4060 Laptop 8GB (local)

### V12.14 NaN Gradient Guards

Run 5 encountered progressive optimizer corruption from sporadic NaN in the encoder latent `z`. The NaN did not always propagate to the total loss (masked by bfloat16 autocast or loss component structure), so the existing loss-level NaN guard (V12.13) did not catch it. NaN gradients accumulated into Adam's `exp_avg` and `exp_avg_sq` momentum buffers, causing permanent corruption.

**Root cause evidence**:
- `z_norm` logged `nan` intermittently during Run 5 (epochs 1353-1373)
- `checkpoint_best.pt` (epoch 1373): 74 encoder weight tensors + 148 optimizer momentum tensors contained NaN
- `checkpoint_final.pt` (epoch 1354): 0 NaN in weights or optimizer — last clean checkpoint

**Fixes applied**:
1. **z-level NaN guard** (line ~1547): After encoder forward, if `z` contains any NaN, batch is skipped before decoder forward and before backward pass. Catches the root cause at the source.
2. **Gradient-level NaN guard** (line ~1712): After `scaler.unscale_()` and `clip_grad_norm_()`, if gradient norm is NaN/Inf for encoder or decoder, that optimizer's step is skipped independently. Prevents NaN from entering Adam momentum buffers.
3. **Checkpoint swap**: Corrupted `checkpoint_best.pt` renamed to `checkpoint_best_corrupted.pt`. Clean `checkpoint_final.pt` copied to `checkpoint_best.pt`.

**Cost**: Lost ~19 epochs of progress (28.9% → 28.0% exact). Model weights and optimizer state fully healthy.

### Early Results (Post-Fix)

| Epoch | Loss | Exact | Acc | zN | Skip |
|-------|------|-------|-----|----|------|
| 1356 | 0.8521 | 28.1% | 80.6% | 18.0 | 2% |
| 1357 | 0.9487 | 27.4% | 78.2% | 18.3 | 1% |
| 1358 | 0.9432 | 27.7% | 78.3% | 18.6 | 2% |
| 1359 | 0.9299 | 27.6% | 78.4% | 18.8 | 2% |

- z_norm consistently real numbers (no NaN) — fix working
- Loss decreasing, metrics stable
- Skip rate normal (1-2%, selective backprop only)

### Progress Through Epoch 1574 (2026-02-04)

Training stopped at epoch 1548 (process killed externally, no error in log). Restarted with explicit `CC` env var for `torch.compile` gcc dependency.

| Epoch | Loss | Exact | Acc | zN |
|-------|------|-------|-----|----|
| 1548 | 0.4681 | 41.6% | 88.4% | 29.0 |
| 1560 | 0.4536 | 42.7% | 88.8% | 29.2 |
| 1574 | 0.4288 | 43.5% | 89.2% | 29.5 |

### Epoch Timing Baseline (2026-02-04)

**Measured**: 27 epochs in 2h 59m 14s = **6.63 minutes/epoch (398 seconds)**

This is the baseline for Run 6 on the local RTX 4060 Laptop GPU with all current optimizations active. Use this to measure the impact of future training changes.

#### Hardware

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, Compute 8.9 |
| **CPU** | Intel Core Ultra 7 155H |
| **RAM** | 7.5 GB (WSL2 allocation) |
| **Platform** | WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2) |

#### Model

| Component | Parameters |
|-----------|-----------|
| Encoder (FullMaterialsVAE) | 4,887,200 |
| Decoder (EnhancedTransformerDecoder) | 102,796,948 |
| **Total** | **107,684,148** |

#### Training Configuration

| Setting | Value | Impact |
|---------|-------|--------|
| Batch size | 32 (effective 64 w/ accumulation) | Memory-limited by 8GB VRAM |
| Accumulation steps | 2 | Smoother convergence |
| AMP dtype | bfloat16 | No GradScaler needed, ~1.5x speedup |
| torch.compile | `reduce-overhead` (CUDA graphs) | ~1.3x speedup (one-time warmup cost) |
| Selective backprop | threshold=0.33 | Skips ~2% of batches (backward only) |
| Gradient checkpointing | Disabled | Not needed at batch_size=32 |
| REINFORCE | Disabled (rl_weight=0.0) | ~2x speedup vs enabled |
| DataLoader workers | Auto-detected by `detect_environment()` | WSL2: 2 workers, pin_memory=False; Colab: up to 4 workers, pin_memory=True |
| Training samples | 46,600 (contrastive SC + non-SC) | 1,457 batches/epoch |
| Flash Attention (SDPA) | Enabled | ~1.2x attention speedup |

#### Historical Timing Comparison

| Run | Config | Epoch Time | Notes |
|-----|--------|-----------|-------|
| Run 3 | REINFORCE enabled, contrastive | ~30 min | Autoregressive sampling bottleneck |
| Run 4 | CE-only, selective backprop | ~4-5 min | ~6-7x faster than Run 3 |
| **Run 6 (baseline)** | **CE-only, torch.compile, selective backprop** | **~6.6 min** | **Current baseline** |

Note: Run 6 is slightly slower than Run 4's measured range due to the larger number of training samples (46,600 vs initial runs) and the torch.compile CUDA graph overhead stabilizing over longer runs. The 4-5 min figure from Run 4 was measured early in training when selective backprop was skipping more batches (higher relative loss variance).

---

## Run 5: Deterministic Latent Space Overnight Training (2026-02-02)

**Status**: Stopped — optimizer corrupted by sporadic NaN (see Run 6 for analysis)
**Script**: `scripts/train_v12_clean.py` (deterministic encoder, L2 reg)
**Resume from**: checkpoint_best.pt (epoch 1351, exact=27.6%)
**Log**: `outputs/training_deterministic_overnight.log`
**GPU**: RTX 4060 Laptop 8GB (local)

### Validation (3-epoch test before overnight run)

| Epoch | Loss | Exact | Tc | Magpie | zN |
|-------|------|-------|----|--------|----|
| 1352 | 0.920 | 28.0% | 0.017 | 0.038 | 16.2 |
| 1353 | 1.001 | 27.2% | 0.017 | 0.034 | 16.8 |
| 1354 | 0.979 | 27.2% | 0.016 | 0.031 | 17.3 |

- Checkpoint loaded successfully (`fc_logvar` weights ignored via `strict=False`)
- Encoder optimizer fresh init (param count changed) — expected
- z_norm stable at ~16-17, not exploding
- Loss finite and reasonable, no accuracy regression
- New best exact match 28.0% on first deterministic epoch

---

## Architectural Change: Probabilistic VAE → Deterministic Encoder (2026-02-02)

**Files modified**:
- `src/superconductor/models/attention_vae.py` — `AttentionVAEEncoder` gains `deterministic=True` mode (skips `fc_logvar`, returns `None` for logvar). `FullMaterialsVAE` uses deterministic mode by default. `reparameterize()` passes through when logvar is None. KL loss replaced with L2 regularization `mean(z²)` under same `kl_loss` key.
- `scripts/train_v12_clean.py` — Checkpoint loading uses `strict=False` to ignore old `fc_logvar` weights. Encoder optimizer restore wrapped in try/except (param count changed). Added `z_norm` metric tracking per epoch (monitors for z explosion/collapse).

**Rationale**: The original probabilistic sampling (reparameterization trick) adds noise to z that hinders contrastive learning and downstream deterministic tasks. Switching to deterministic coordinates means z = fc_mean(h) directly — same z for same input every time. L2 regularization replaces KL to keep z bounded without the distributional constraint. This change effectively transitioned the architecture from a VAE to the deterministic Multi-Task Superconductor Generator it is today.

**Checkpoint compatibility**: `fc_mean` weights load directly (same parameter name). Old `fc_logvar` weights silently ignored via `strict=False`. Encoder optimizer gets fresh init; decoder checkpoint loads unchanged.

**Monitoring**: `zN` field in epoch summary shows mean L2 norm of z vectors. Should be stable (not growing unboundedly).

---

## Run 4b: CE-Phase Contrastive Training (continued, 2026-02-02)

**Status**: Running
**Script**: `scripts/train_v12_clean.py` V12.13
**Resume from**: checkpoint_best.pt (epoch 1318, exact=25.5%)
**Log**: `outputs/training_contrastive_ce_phase_v2.log`

### V12.13 Bugfixes (applied after Run 4a NaN crash at epoch 1319)

1. **NaN guard**: Skip backward pass on batches with NaN/Inf loss (prevents single bad batch from corrupting entire epoch)
2. **Rollback into compiled model fix**: `load_checkpoint` now detects whether target model is compiled and preserves/adds `_orig_mod.` prefixes accordingly, instead of always stripping them

---

## Run 4a: CE-Phase Contrastive Training (2026-02-02)

**Status**: Crashed at epoch 1319 (NaN loss → rollback failed)
**Script**: `scripts/train_v12_clean.py` V12.12
**Data**: `supercon_fractions_contrastive.csv` (46,645 samples: 23,451 SC + 23,194 non-SC)
**Resume from**: checkpoint_best.pt (epoch 1289, exact=25.7%)
**GPU**: RTX 4060 Laptop 8GB (local)
**Log**: `outputs/training_contrastive_ce_phase.log`

### Crash Analysis

- Epochs 1290-1318 trained successfully with steadily improving metrics
- Epoch 1319: Random NaN loss (likely bfloat16 numerical edge case in contrastive/focal loss)
- Catastrophic drop detector triggered rollback, but `load_checkpoint` crashed with `RuntimeError` — it stripped `_orig_mod.` prefixes from the checkpoint but tried to load into compiled `OptimizedModule` that expects those prefixes
- Best checkpoint saved at epoch 1318 (exact=25.5%) was preserved intact

### Run 4a Results (29 epochs: 1290-1318)

| Metric | Start (1290) | End (1318) | Trend |
|---|---|---|---|
| Loss | 1.4426 | 1.1856 | -17.8% |
| Accuracy | 71.8% | 74.6% | +2.8pp |
| SC Exact Match | 22.5% | 25.5% | +3.0pp |
| Magpie MSE | 0.0710 | 0.0460 | -35.2% |
| Contrastive | 4.2427 | 3.9414 | -7.1% |

### Training Paradigm: Two-Phase (CE then RL)

Based on literature review of RLHF/RL training pipelines (OpenAI, Anthropic, DeepMind) and analysis of our own training metrics, we adopted a **two-phase training strategy**:

**Phase 1 (Current): Cross-Entropy Convergence**
- REINFORCE disabled (`rl_weight=0.0`)
- Model learns reconstruction + contrastive objectives without RL noise
- Target: SC exact match > 70-80% before moving to Phase 2

**Phase 2 (Future): REINFORCE Fine-Tuning**
- Re-enable `rl_weight=1.0-2.5` once CE-trained model is competent
- RL rewards become meaningful when model can mostly reconstruct formulas
- KL divergence penalty keeps RL policy close to CE-pretrained model

**Rationale**: During Run 3 (epochs 1289-1291), REINFORCE was contributing zero useful signal:
- Reward: -22.8 (model can't reconstruct formulas → meaningless rewards)
- RL loss: ~0.0000 (no gradient signal)
- Meanwhile, autoregressive sampling (2 RLOO samples × KV cache × 60 tokens per batch) was the most expensive computation, roughly doubling epoch time for zero benefit.

### Optimizations Applied

| Optimization | Setting | Expected Speedup | Rationale |
|---|---|---|---|
| **Disable REINFORCE** | `rl_weight=0.0` | ~2x | No autoregressive sampling overhead |
| **Gradient Accumulation** | `accumulation_steps=2` | Smoother convergence | Effective batch=64 (was 32) |
| **Selective Backpropagation** | `threshold=0.33` | 1.3-1.5x | Skip backward on easy batches (loss < 33% of running avg) |
| Contrastive loss | `weight=0.1, warmup=100 epochs` | — | SC/non-SC latent separation |
| Balanced sampling | `WeightedRandomSampler` | — | ~50/50 SC/non-SC per batch |
| torch.compile | `reduce-overhead` | ~1.3x | CUDA graph capture (one-time warmup cost) |
| bfloat16 AMP | Compute 8.9 GPU | ~1.5x | No GradScaler needed |
| Flash Attention (SDPA) | Enabled | ~1.2x | Memory-efficient attention |

**Measured speedup**: **~6-7x** vs Run 3 (REINFORCE-enabled contrastive).
- Run 3 (with REINFORCE): ~30 min/epoch
- Run 4 (CE-only + selective backprop): **~4-5 min/epoch**
- Overnight projection: ~107 epochs (epoch 1290 → ~1397)

### Loss Architecture

```
SC Samples (full loss):
  CE formula loss (focal, gamma=2.0, smoothing=0.1)
  + 10.0 * Tc MSE
  + 2.0 * Magpie MSE
  + 2.0 * Stoichiometry MSE
  + 0.0001 * KL divergence

Non-SC Samples (formula-only, reduced weight):
  0.5 * CE formula loss

All Samples:
  + 0.1 * SupCon contrastive loss (warmup over 100 epochs)
```

### Key Design Decisions

1. **Tc normalization from SC samples only**: Non-SC materials have Tc=0.0 which would skew the mean. Normalizing from SC samples (mean=32.2K, std=35.4K) preserves the model's learned Tc representation.

2. **No REINFORCE for non-SC samples**: Non-SC formulas have no reward signal (no Tc to predict, no SC-specific quality metric). Formula CE alone trains the decoder.

3. **Selective backprop with EMA threshold**: Uses exponential moving average (momentum=0.95) of batch loss as baseline. Batches with loss < 33% of running average are "easy" — forward pass computed (for metrics) but backward pass skipped. This saves ~30% of gradient computation once the model has learned common patterns.

4. **Retrain mode active**: Catastrophic drop detector reset to prevent rollback loops when transitioning to new data distribution.

### References

- [RLHF Deciphered (ACM Computing Surveys)](https://dl.acm.org/doi/full/10.1145/3743127) — Formal pretrain→SFT→RL workflow
- [CMU RLHF 101 Tutorial (June 2025)](https://blog.ml.cmu.edu/2025/06/01/rlhf-101-a-technical-tutorial-on-reinforcement-learning-from-human-feedback/)
- [VarCon: Variational Supervised Contrastive Learning (Dec 2025)](https://arxiv.org/html/2506.07413) — Faster convergence, lower batch size dependency
- [Contrastive Learning Mitigates Posterior Collapse (OpenReview)](https://openreview.net/forum?id=SrgIkwLjql9) — VAE + contrastive synergy
- [SelectiveBackprop (Angela Jiang et al.)](https://github.com/angelajiang/SelectiveBackprop) — Up to 3.5x speedup skipping easy samples
- [Nathan Lambert RLHF Book (2025)](https://arxiv.org/abs/2504.12501) — Complete RLHF methodology

---

## Run 3: Contrastive Training with REINFORCE (2026-02-02)

**Status**: Stopped — replaced by Run 4
**Data**: `supercon_fractions_contrastive.csv` (46,645 samples)
**Epochs completed**: 1289-1291 (3 epochs)
**Log**: `outputs/training_contrastive.log`

### Results

| Epoch | Exact | Acc | Tc Loss | Contrastive | Reward |
|---|---|---|---|---|---|
| 1289 | 25.7% | 72.1% | 0.0182 | 4.293 | -20.7 |
| 1290 | 25.3% | 70.8% | 0.0181 | 4.229 | -22.6 |
| 1291 | 25.3% | 71.3% | 0.0180 | 4.193 | -22.8 |

- TRUE autoregressive at epoch 1290: **23.7%** (477/2016)
- Non-SC exact match: **0.0-0.1%** (completely new territory for the model)
- Each epoch: ~30 min on RTX 4060 (too slow with REINFORCE overhead)

### Why Stopped
REINFORCE consuming ~50% of compute for zero useful signal. Switched to two-phase paradigm (Run 4).

---

## Run 2: SC-Only Combined Training (2026-02-02)

**Status**: Never completed first epoch — replaced by contrastive approach
**Data**: `supercon_fractions_combined.csv` (23,451 SC samples)
**Log**: `outputs/training_combined_retrain.log`

Started but killed before first epoch completed (torch.compile warmup). User decided to go with full contrastive dataset instead.

---

## Run 1: Original SuperCon Training (2025-2026)

**Status**: Completed
**Data**: `supercon_fractions.csv` (16,521 SC samples)
**Epochs**: ~1284 epochs
**Best Checkpoint**: `outputs/checkpoint_best.pt`

### Final Metrics

| Metric | Value |
|---|---|
| Exact Match (teacher-forced) | 93.1% |
| TRUE Autoregressive Exact | ~92.2% |
| Token Accuracy | 99.0%+ |
| Tc Prediction Error | 0.0001K |
| Magpie MSE | 0.0038 |
| REINFORCE Reward | 73.1 |

### Architecture
- Encoder: FullMaterialsVAE (5.4M params) — element attention + Magpie + Tc
- Decoder: EnhancedTransformerDecoder (102.8M params) — 12 layers, pointer-generator
- Latent: 2048-dim
- Total: 108.2M parameters

### Training Progression
- Epochs 0-100: CE-only, curriculum learning (ramp Tc/Magpie weights)
- Epochs 100-800: CE + REINFORCE introduced, teacher forcing decay
- Epochs 800-1284: Fine-tuning with causal entropy maintenance
- Training on DSMLP and local RTX 4060

---

## Data Evolution

| Version | File | Samples | Date | Notes |
|---|---|---|---|---|
| V1 (Original) | `supercon_fractions.csv` | 16,521 SC | 2025 | SuperCon database |
| V2 (Combined) | `supercon_fractions_combined.csv` | 23,451 SC | Jan 2026 | + 6,930 NEMAD SC |
| V3 (Contrastive) | `supercon_fractions_contrastive.csv` | 46,645 mixed | Jan 2026 | + 23,194 non-SC |

All versions exclude 45 generative holdout samples at training time.

---

## Model Checkpoint History

| Checkpoint | Epoch | Exact Match | Data | Notes |
|---|---|---|---|---|
| `checkpoint_epoch_1299.pt` | 1299 | ~93% | V1 (16.5K SC) | Run 1 late stage |
| `checkpoint_best.pt` | 1574+ | 43.5%+ | V3 (46K contrastive) | Run 6 active (updating) |
| `checkpoint_best_corrupted.pt` | 1373 | 28.9% | V3 (46K contrastive) | Run 5 best — NaN-corrupted optimizer+weights |
| `checkpoint_final.pt` | 1354 | 28.0% | V3 (46K contrastive) | Run 5 last clean checkpoint |

Note: The drop from 93% to 25.7% is expected when transitioning from V1→V3 data due to:
1. New normalization statistics (2x more samples shift Tc/Magpie distributions)
2. 50% of each batch is non-SC formulas the model has never seen
3. Balanced sampling reduces SC samples seen per epoch by ~2x
