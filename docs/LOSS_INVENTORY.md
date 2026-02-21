# Loss Inventory — Superconductor VAE (V14.0)

Complete inventory of every loss component in `scripts/train_v12_clean.py`. Updated 2026-02-20.

---

## Summary

| # | Loss | Status | Weight | Typical Value | Skip Schedule |
|---|------|--------|--------|---------------|---------------|
| 1 | Formula CE (Focal) | **ACTIVE** | `ce_weight: 1.0` | ~0.7 | Never skip |
| 2 | REINFORCE/SCST | **ACTIVE** | `rl_weight: 1.0` | ~18.6 raw | Never skip (V13.2) |
| 3 | Tc MSE | **ACTIVE** | `tc_weight: 10.0` | ~0.43 | Yes (0.5, 0.1) |
| 4 | Magpie MSE | **ACTIVE** | `magpie_weight: 2.0` | ~0.06 | Yes (0.1, 0.1) |
| 5 | Stoichiometry MSE | **ACTIVE** | `stoich_weight: 2.0` | ~0.001 | Yes (0.01, 0.05) |
| 6 | NumDen MSE | **ACTIVE** | `numden_weight: 1.0` | ~2.96 | Yes (3.0, 0.5) |
| 7 | KL / L2 Regularization | **ACTIVE** | `kl_weight: 0.0001` | tiny | Never skip |
| 8 | Element Count MSE | **ACTIVE** | hardcoded 0.5 | tiny | Never skip |
| 9 | Z-Norm Penalty | **ACTIVE** | `z_norm_penalty_weight: 0.001` | tiny | Never skip |
| 10 | Tc Classification | **ACTIVE** | `tc_class_weight: 4.0` | varies | Yes (0.5, 0.2) |
| 11 | Physics Z | **ACTIVE** | warmup → 1.0 | ~0.32 | Yes (0.5, 0.2) |
| 12 | High-Pressure (HP) | **ACTIVE** | `hp_loss_weight: 0.5` | varies | Yes (0.3, 0.1) |
| 13 | SC Classification | **ACTIVE** | `sc_loss_weight: 0.5` | varies | Yes (0.3, 0.1) |
| 14 | Stop Prediction | **ACTIVE** | `stop_loss_weight: 5.0` | varies | Yes (0.1, 0.1) |
| 15 | Family Classifier | **ACTIVE** | `family_classifier_weight: 2.0` | varies | Yes (0.5, 0.2) |
| 16 | Contrastive | **DISABLED** | `contrastive_weight: 0.0` | was ~5.06 | N/A |
| 17 | Theory Consistency | **DISABLED** | `theory_weight: 0.0` | was ~1.43 | N/A |

---

## Detailed Descriptions

### 1. Formula Cross-Entropy (Focal Loss)
- **Version**: V1+ (core), V12.5 focal
- **Config**: `ce_weight: 1.0`, `focal_gamma: 2.0`, `label_smoothing: 0.05`
- **Computed in**: `CombinedLossWithREINFORCE.forward()` (line ~2270)
- **What it does**: Per-token cross-entropy between decoder output logits and target formula tokens. Focal loss variant (gamma=2.0) down-weights easy tokens and focuses gradients on hard tokens. This is the **core training signal** — it teaches the decoder to produce correct formula token sequences.
- **Applied to**: All samples (SC + non-SC)
- **Weight mechanism**: `loss_fn.ce_weight` attribute
- **Skip schedule**: NEVER skip — this is the primary learning signal

### 2. REINFORCE / SCST (Self-Critical Sequence Training)
- **Version**: V12.8, **V14.0 reward redesign**
- **Config**: `rl_weight: 1.0`, `rl_temperature: 0.2`, `n_samples_rloo: 4`, `rl_method: 'scst'`
- **V14.0 Config**: `use_v14_reward: True`, `v14_sharpness: 4.0`, `v14_max_reward: 100.0`
- **Computed in**: `CombinedLossWithREINFORCE.compute_scst()` (line ~2099)
- **What it does**: Sequence-level reinforcement learning. Generates 4 complete formula sequences autoregressively, computes reward (similarity to target), and uses SCST baseline (greedy decode reward) to compute advantages. Provides sequence-level signal that CE alone cannot — rewards whole correct formulas, not just individual tokens.
- **V14.0 Reward Changes**:
  - **Continuous power-law reward**: `max_reward * (n_correct / n_total) ^ sharpness` eliminates the 3→4 error cliff that killed gradient for 72% of the population. The 3→4 transition becomes 41→29 (smooth 30% drop) instead of 10→5 (cliff).
  - **Token-type-aware penalties**: Element errors get -3.0 penalty (wrong compound), integer errors -1.0 (wrong stoichiometry), fractions -0.5, specials -0.5. Overlaid on continuous base reward.
  - **Symmetric too-short detection**: Mirrors too-long handling. Premature EOS with correct prefix gets high reward (50 base minus 5 per missing token, floor 10).
  - **Phased curriculum** (disabled by default): Phase 1 = elements only, Phase 2 = +integers, Phase 3 = full reward with sharpness=6.0.
- **V14.0 Training Controls**:
  - **RL warmup**: Linear ramp from 10% to 100% over 20 epochs (`rl_warmup_epochs`, `rl_warmup_start`)
  - **RL safety guard**: Halves rl_weight if TF exact drops >2% from activation baseline. Pauses RL if weight drops below 0.05 (`rl_safety_exact_drop`, `rl_safety_check_interval`)
  - **PhysZ staggering**: Blocks RL activation until PhysZ warmup is complete (`rl_requires_physz_stable`)
  - **Temperature schedule**: Decays from 0.5 (exploration) to 0.2 (exploitation) over 50 epochs (`rl_temperature_start/end/decay_epochs`). Works WITH entropy manager via `set_base_temperature()`.
- **Applied to**: SC samples only (non-SC get z=None → skipped)
- **Weight mechanism**: `loss_fn.rl_weight` attribute. Guard: `if self.rl_weight > 0` skips all sampling.
- **Compute cost**: **91% of loss computation time** (4x full autoregressive decode per batch)
- **Skip schedule**: Never skip (V13.2) — RL is a policy gradient that fluctuates by design, not a converging loss. Skipping kills the learning signal.

### 3. Tc MSE (Critical Temperature Prediction)
- **Version**: V1+ (core)
- **Config**: `tc_weight: 10.0` (curriculum-controlled)
- **Computed in**: `CombinedLossWithREINFORCE.forward()` (line ~2358)
- **What it does**: MSE between encoder's Tc prediction and true normalized Tc value. Teaches the encoder to predict critical temperature from composition.
- **Applied to**: SC samples only (non-SC get `tc_weight_override=0.0`)
- **Weight mechanism**: Local variable `tc_weight` from `get_curriculum_weights()`, passed as `tc_weight_override`
- **Skip schedule**: Yes — (threshold=0.5, delta=0.1)

### 4. Magpie MSE (Feature Reconstruction)
- **Version**: V12+ (full materials)
- **Config**: `magpie_weight: 2.0` (curriculum-controlled)
- **Computed in**: `CombinedLossWithREINFORCE.forward()` (line ~2370)
- **What it does**: MSE between encoder's Magpie feature predictions (145 dims) and true Magpie compositional features. Forces the latent space to encode materials science knowledge.
- **Applied to**: SC samples only (non-SC get `magpie_weight_override=0.0`)
- **Weight mechanism**: Local variable `magpie_weight` from `get_curriculum_weights()`
- **Skip schedule**: Yes — (threshold=0.1, delta=0.1)

### 5. Stoichiometry MSE
- **Version**: V12.4
- **Config**: `stoich_weight: 2.0`
- **Computed in**: `CombinedLossWithREINFORCE.forward()` (line ~2410)
- **What it does**: MSE between encoder's fraction predictions (12 mole fractions) and true element fractions. Teaches encoder to predict composition ratios that the decoder uses as conditioning.
- **Applied to**: All samples
- **Weight mechanism**: `loss_fn.stoich_weight` attribute
- **Skip schedule**: Yes — (threshold=0.01, delta=0.05)

### 6. Numerator/Denominator MSE
- **Version**: V12.38
- **Config**: `numden_weight: 1.0`
- **Computed in**: `CombinedLossWithREINFORCE.forward()` (line ~2426)
- **What it does**: Masked MSE between encoder's numden predictions (12 log-numerators + 12 log-denominators) and true values in log1p space. Gives the decoder explicit integer fraction information instead of just continuous mole fractions.
- **Applied to**: All samples (masked by element_mask)
- **Weight mechanism**: `loss_fn.numden_weight` attribute
- **Skip schedule**: Yes — (threshold=3.0, delta=0.5)

### 7. KL / L2 Regularization
- **Version**: V1+ (core), repurposed as L2 in V12
- **Config**: `kl_weight: 0.0001`
- **Computed in**: Encoder forward pass (line ~1094 in attention_vae.py)
- **What it does**: Originally KL divergence for VAE. Now the encoder is deterministic (no reparameterization), so this is effectively L2 regularization on z-vectors: `(z ** 2).sum(dim=1).mean()`. Prevents latent space from growing unbounded.
- **Applied to**: All samples
- **Weight mechanism**: `loss_fn.kl_weight` attribute
- **Skip schedule**: Never skip — regularization should always be active

### 8. Element Count MSE
- **Version**: V12.4
- **Config**: Hardcoded weight 0.5 (line ~2456)
- **Computed in**: `CombinedLossWithREINFORCE.forward()` (line ~2411)
- **What it does**: MSE between predicted element count (from fraction_head's last output) and true number of elements. Helps decoder know how many elements to generate.
- **Applied to**: All samples
- **Weight mechanism**: Hardcoded 0.5 in total loss formula
- **Skip schedule**: Never skip — tiny cost, always useful

### 9. Z-Norm Soft Penalty
- **Version**: V12.34
- **Config**: `use_z_norm_penalty: True`, `z_norm_target: 22.0`, `z_norm_penalty_weight: 0.001`
- **Computed in**: `CombinedLossWithREINFORCE.forward()` (line ~2465)
- **What it does**: Penalizes z-vector norms that exceed the target (22.0). One-sided: only penalizes excess, not deficit. Prevents outlier z-norms that decode poorly.
- **Applied to**: All samples with z != None
- **Weight mechanism**: Hardcoded from TRAIN_CONFIG inside loss function
- **Skip schedule**: Never skip — tiny cost, stability measure

### 10. Tc Bucket Classification
- **Version**: V12.28
- **Config**: `tc_class_weight: 4.0`
- **Computed in**: `CombinedLossWithREINFORCE.forward()` (line ~2393)
- **What it does**: Cross-entropy classification of Tc into discrete buckets (0-10K, 10-50K, 50-100K, 100K+). Auxiliary signal that helps encoder learn Tc range even before precise MSE converges.
- **Applied to**: All samples with tc_class_logits != None
- **Weight mechanism**: `loss_fn.tc_class_weight` attribute
- **Skip schedule**: Yes — (threshold=0.5, delta=0.2)

### 11. Physics Z Consistency
- **Version**: V12.31
- **Config**: Warmup to 1.0, sub-weights: `physics_z_comp_weight: 1.0`, `physics_z_magpie_weight: 0.5`, `physics_z_consistency_weight: 0.1`
- **Computed in**: `PhysicsZLoss.forward()` (separate module, line ~4029)
- **What it does**: Multi-component loss ensuring z-vectors encode physically meaningful information. Components: composition prediction from z, Magpie feature encoding, z-space consistency under composition perturbations, thermodynamic consistency (V12.36).
- **Applied to**: All samples
- **Weight mechanism**: Local variable `effective_physics_z_weight` with warmup ramp
- **Skip schedule**: Yes — (threshold=0.5, delta=0.2)

### 12. High-Pressure (HP) Prediction
- **Version**: V12.19, boosted V12.26
- **Config**: `hp_loss_weight: 0.5`
- **Computed in**: `train_epoch()` (line ~3957)
- **What it does**: Binary cross-entropy predicting whether a superconductor requires high pressure. Critical for hydrogen-based superconductors (H3S, LaH10) where Tc is meaningless without pressure context.
- **Applied to**: SC samples only
- **Weight mechanism**: Local variable `epoch_hp_loss_weight`, passed as parameter
- **Skip schedule**: Yes — (threshold=0.3, delta=0.1)

### 13. SC/non-SC Classification
- **Version**: V12.21
- **Config**: `sc_loss_weight: 0.5`
- **Computed in**: `train_epoch()` (line ~3968)
- **What it does**: Binary cross-entropy classifying whether a material is a superconductor. Helps the latent space separate SC from non-SC materials.
- **Applied to**: All samples
- **Weight mechanism**: Local variable `epoch_sc_loss_weight`, passed as parameter
- **Skip schedule**: Yes — (threshold=0.3, delta=0.1)

### 14. Stop Prediction
- **Version**: V12.30
- **Config**: `stop_loss_weight: 5.0`, `stop_end_position_weight: 10.0`
- **Computed in**: `train_epoch()` (line ~3930)
- **What it does**: BCE on stop head predicting END token position. High weight (5.0) due to severe class imbalance (~1 END per 14 non-END tokens). Position-aware weighting (10x at END positions, V12.37) addresses this further.
- **Applied to**: All samples
- **Weight mechanism**: Local variable `epoch_stop_loss_weight`, passed as parameter (V12.40)
- **Skip schedule**: Yes — (threshold=0.1, delta=0.1)

### 15. Hierarchical Family Classifier
- **Version**: V12.33
- **Config**: `family_classifier_weight: 2.0`, `use_family_classifier: True`
- **Computed in**: `train_epoch()` (line ~3974)
- **What it does**: Hierarchical classification of superconductor family: coarse (cuprate/iron/conventional/other), cuprate sub-family (YBCO/LSCO/Bi/Tl/Hg), iron sub-family (1111/122/11). Forces latent space to encode family membership.
- **Applied to**: SC samples only (requires family labels)
- **Weight mechanism**: Local variable `epoch_family_loss_weight`, passed as parameter
- **Skip schedule**: Yes — (threshold=0.5, delta=0.2)

### 16. Contrastive Loss (DISABLED)
- **Version**: V12.12 (disabled V12.26)
- **Config**: `contrastive_weight: 0.0`
- **Reason disabled**: Plateaued at 5.06, consuming 16% of gradient budget with no improvement.
- **What it does**: Pushes SC and non-SC z-vectors apart in latent space using contrastive learning.

### 17. Theory Consistency (DISABLED)
- **Version**: V12.22 (disabled V12.26)
- **Config**: `theory_weight: 0.0`
- **Reason disabled**: Plateaued at 1.43, consuming 22% of gradient budget with no improvement.
- **What it does**: Ensures z-vectors are consistent with theoretical predictions (BCS-like relationships).

---

## Loss Composition

### Inside `CombinedLossWithREINFORCE.forward()` (the `total` tensor)
```
total = ce_weight * formula_CE + rl_weight * reinforce_loss
      + tc_weight * tc_loss
      + magpie_weight * magpie_loss
      + kl_weight * kl_loss
      + stoich_weight * stoich_loss
      + numden_weight * numden_loss
      + 0.5 * element_count_loss
      + tc_class_weight * tc_class_loss
      + z_norm_penalty_weight * z_norm_penalty  (if enabled)
```

### External (added in `train_epoch()` batch loop)
```
loss = sc_frac * sc_loss_dict['total']
     + non_sc_frac * non_sc_formula_weight * non_sc_loss_dict['total']
     + contrastive_weight * contrastive_loss     (DISABLED, weight=0)
     + hp_loss_weight * hp_loss
     + sc_loss_weight * sc_loss
     + theory_weight * theory_loss               (DISABLED, weight=0)
     + stop_loss_weight * stop_loss
     + physics_z_weight * physics_z_loss
     + family_loss_weight * family_loss
```

---

## V12.40 Smart Skip Schedule

Each loss is independently tracked. When converged (below threshold), it's computed every 4th epoch only. If it spikes above baseline + delta, it resumes every-epoch.

```python
'loss_skip_schedule': {
    # metric_key:      (converge_threshold, spike_delta)
    # reinforce_loss: REMOVED in V13.2 — RL fluctuates by design, skipping kills signal
    'tc_loss':         (0.5,   0.1),
    'magpie_loss':     (0.1,   0.1),
    'stoich_loss':     (0.01,  0.05),
    'numden_loss':     (3.0,   0.5),
    'tc_class_loss':   (0.5,   0.2),
    'physics_z_loss':  (0.5,   0.2),
    'hp_loss':         (0.3,   0.1),
    'sc_loss':         (0.3,   0.1),
    'stop_loss':       (0.1,   0.1),
    'family_loss':     (0.5,   0.2),
}
```

Losses NOT in skip schedule (always active):
- Formula CE — core training signal
- KL/L2 regularization — stability
- Element count MSE — tiny, always useful
- Z-norm penalty — tiny, stability

---

## V14.0 RL Curriculum Redesign

### Problem
After 24 epochs of SCST RL (rl_weight=1.0), AR exact match was stuck at ~1.2% despite TF exact being 99.4%. Error analysis revealed:
- 26.7% of AR samples have 1-3 errors (RL-tractable) — RL was moving the distribution but couldn't push across the 0-error threshold
- 72.1% have 4+ errors — reward cliff at 3→4 errors (10.0→5.0) killed gradient for these samples
- 87% of compute was in RL sampling, for near-zero gradient signal on most samples

### Continuous Reward (eliminates the cliff)

Formula: `reward = max_reward * (n_correct / n_total) ^ sharpness`

| Errors (15-token seq) | Old Reward | New Reward (sharpness=4) |
|----------------------|-----------|-------------------------|
| 0 (exact)            | 100.0     | 100.0                   |
| 1                    | 50.0      | 75.8                    |
| 2                    | 25.0      | 56.4                    |
| 3                    | 10.0      | 41.0                    |
| **4 (THE CLIFF)**    | **max 5.0** | **28.9**              |
| 5                    | max 5.0   | 19.8                    |
| 7                    | max 5.0   | 7.7                     |

Config: `GPURewardConfigV14` in `src/superconductor/losses/reward_gpu_native.py`, gated by `use_v14_reward` in TRAIN_CONFIG.

### Token-Type Penalties

| Token Type | Penalty | Rationale |
|-----------|---------|-----------|
| Element   | -3.0    | Wrong element = wrong compound entirely |
| Integer   | -1.0    | Wrong stoichiometry |
| Fraction  | -0.5    | Wrong fraction (value-scaled penalty already exists) |
| Special   | -0.5    | Wrong BOS/EOS/PAD |

### Training Controls

| Control | Config Key | Default | Description |
|---------|-----------|---------|-------------|
| RL Warmup | `rl_warmup_epochs`, `rl_warmup_start` | 20, 0.1 | Linear ramp from 10% to 100% |
| RL Safety Guard | `rl_safety_exact_drop`, `rl_safety_check_interval` | 0.02, 5 | Halve weight on >2% TF regression |
| PhysZ Stagger | `rl_requires_physz_stable` | True | Block RL until PhysZ warmup done |
| Temp Schedule | `rl_temperature_start/end/decay_epochs` | 0.5, 0.2, 50 | Exploration → exploitation |
| Phased Curriculum | `rl_use_phased_curriculum` | False | Staged reward complexity |

### Key Files
- `src/superconductor/losses/reward_gpu_native.py` — `GPURewardConfigV14`, continuous reward, token-type penalties
- `src/superconductor/training/entropy_maintenance.py` — `set_base_temperature()` for temp schedule
- `scripts/train_v12_clean.py` — TRAIN_CONFIG keys, warmup/safety/stagger/schedule logic
