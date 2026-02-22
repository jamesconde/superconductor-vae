# Loss Inventory — Superconductor VAE (V14.3)

Complete inventory of every loss component in the training system. Updated 2026-02-22.

The system has **17 differentiable loss terms** that contribute to the backward pass (14 active, 1 auto-reactivatable, 2 removed), plus **REINFORCE reward signals** (non-differentiable) that shape the policy gradient loss indirectly.

> **Note on weights**: The loss weights listed below are the current values in `TRAIN_CONFIG` as of V14.3. These are **not fixed constants** — they have changed over the course of training based on model performance, gradient budget analysis, and curriculum scheduling. For example, `tc_weight` was 10.0 in V12.26 and increased to 20.0 to focus gradients on Tc accuracy; `family_classifier_weight` was 2.0 in V12.33 and reduced to 0.5 in V13.1b to reduce gradient competition with formula CE; `hp_loss_weight` and `sc_loss_weight` were 0.5 and increased to 1.0 in V13.2. Some weights also change dynamically during a run via curriculum scheduling (Tc, Magpie ramp up over 30 epochs), smart loss skipping (converged losses computed every 4th epoch), and RL auto-scaling (V14.1). Treat the weights here as a snapshot, not as permanent values.

---

## Quick Reference

| # | Loss | Status | Weight | SC-Only? | Computed In | Skip? |
|---|------|--------|--------|----------|-------------|-------|
| 1 | Formula CE (Focal) | **ACTIVE** | `ce_weight: 1.0` | No | CombinedLoss | Never |
| 2 | REINFORCE/SCST | **ACTIVE** | `rl_weight: 1.0` | Yes | CombinedLoss | Never |
| 3 | Tc Regression | **ACTIVE** | `tc_weight: 20.0` | Yes | CombinedLoss | (0.5, 0.1) |
| 4 | Magpie Reconstruction | **ACTIVE** | `magpie_weight: 2.0` | Yes | CombinedLoss | (0.1, 0.1) |
| 5 | KL / L2 Regularization | **ACTIVE** | `kl_weight: 0.0001` | No | Encoder | Never |
| 6 | Stoichiometry MSE | **ACTIVE** | `stoich_weight: 2.0` | No | CombinedLoss | (0.01, 0.05) |
| 7 | Element Count MSE | **ACTIVE** | hardcoded `0.5` | No | CombinedLoss | Never |
| 8 | Tc Bucket Classification | **ACTIVE** | `tc_class_weight: 1.0` | No | CombinedLoss | (0.5, 0.2) |
| 9 | Z-Norm Penalty | **ACTIVE** | `z_norm_penalty_weight: 0.001` | No | CombinedLoss | Never |
| 10 | Constraint Zoo (A3+A5+A6) | **ACTIVE** | `constraint_zoo_weight: 0.5` | No | CombinedLoss | Never |
| 11 | High-Pressure (HP) | **ACTIVE** | `hp_loss_weight: 1.0` | Yes | train_epoch | (0.3, 0.1) |
| 12 | SC Classification | **ACTIVE** | `sc_loss_weight: 1.0` | No | train_epoch | (0.3, 0.1) |
| 13 | Family Classifier | **ACTIVE** | `family_classifier_weight: 0.5` | Yes | train_epoch | (0.5, 0.2) |
| 14 | Stop Prediction | **ACTIVE** | `stop_loss_weight: 5.0` | No | train_epoch | (0.1, 0.1) |
| 15 | Token Type Classification | **ACTIVE** | `token_type_loss_weight: 0.1` | No | train_epoch | Never |
| 16 | Physics Z Supervision | **DORMANT** | `use_physics_z: False` | No | train_epoch | (0.5, 0.2) |
| 17 | Theory Consistency | **REMOVED** | `theory_weight: 0.0` | Yes | train_epoch | N/A |

**REMOVED** (code deleted): Contrastive loss (V12.26), NumDen MSE (V13.0).

---

## Architecture: Two-Layer Loss Assembly

Losses are computed in two places and combined into a single backward pass target.

### Layer 1: Inside `CombinedLossWithREINFORCE.forward()` → `loss_dict['total']`

Contains the core reconstruction losses and REINFORCE. This is the loss function object that knows about the decoder, tokenizer, and RL sampling.

```
total = ce_weight * formula_CE
      + rl_weight * reinforce_loss
      + tc_weight * tc_loss
      + magpie_weight * magpie_loss
      + kl_weight * kl_loss
      + stoich_weight * stoich_loss
      + 0.5 * element_count_loss
      + tc_class_weight * tc_class_loss
      + constraint_zoo_weight * constraint_zoo_loss
      + z_norm_penalty_weight * z_norm_penalty
```

### Layer 2: External losses added in `train_epoch()` batch loop

Auxiliary classification and prediction heads computed outside the loss function.

```
loss = loss_dict['total']                          # Layer 1
     + hp_loss_weight * hp_loss                    # High-pressure prediction
     + sc_loss_weight * sc_loss                    # SC/non-SC classification
     + theory_weight * theory_loss                 # Family-specific physics (weight=0)
     + stop_loss_weight * stop_loss                # Stop prediction head
     + token_type_loss_weight * type_loss          # Token type classification (V14.3)
     + physics_z_loss_val                          # Physics Z (weight already applied)
     + family_loss_weight * family_loss            # Hierarchical family classification
```

### Mixed SC/non-SC Batch Handling

For mixed batches (the common case with contrastive dataset):
```
loss = sc_frac * sc_loss_dict['total']
     + non_sc_frac * non_sc_formula_weight * non_sc_loss_dict['total']
     + <all auxiliary losses above>
```

- SC portion: Full loss (all weights active, REINFORCE enabled)
- Non-SC portion: Formula CE only (`tc_weight=0`, `magpie_weight=0`, `z=None` → no REINFORCE), scaled by `non_sc_formula_weight=0.5`

---

## Detailed Descriptions

### 1. Formula Cross-Entropy (Focal Loss)

The **primary training signal** — token-by-token reconstruction of formula sequences.

- **Type**: Focal Loss with Label Smoothing
- **Config**: `ce_weight: 1.0`, `focal_gamma: 2.0`, `label_smoothing: 0.05`
- **Where**: `CombinedLossWithREINFORCE.forward()` (line ~2719)
- **Inputs**: `formula_logits [batch, seq, vocab]` vs `formula_targets [batch, seq]`
- **Applied to**: All samples (SC + non-SC)
- **Details**:
  - Focal `(1-p)^gamma` weighting focuses gradients on hard-to-predict tokens
  - V13.0: Fraction tokens upweighted by `fraction_token_weight=2.0` (semantic fractions are harder)
  - V12.34: Per-sample weighting by sequence length (`length_weight_base=8`, `length_weight_alpha=1.0`) and element count (`element_count_base=3`, `element_count_beta=0.5`)
- **Why**: Without this, the decoder has no signal to produce correct formula tokens. Everything else is auxiliary.

### 2. REINFORCE / SCST (Self-Critical Sequence Training)

Closes the teacher-forcing → autoregressive gap with sequence-level reward.

- **Type**: Policy gradient (SCST or RLOO)
- **Config**: `rl_weight: 1.0`, `rl_method: 'scst'`, `n_samples_rloo: 4`, `rl_temperature: 0.2`
- **Where**: `CombinedLossWithREINFORCE.compute_scst()` (line ~2522) / `compute_rloo_autoregressive()` (line ~2296)
- **Inputs**: Latent `z`, `formula_targets`, `stoich_pred`, `heads_pred` (V14.3)
- **Applied to**: SC samples only
- **Details**:
  - SCST: Generates sample autoregressively, computes reward vs greedy baseline, `loss = -(advantage * log_prob).mean()`
  - RLOO: K samples, each uses the other K-1 as baseline (lower variance)
  - V14.0 continuous power-law reward: `max_reward * (n_correct / n_total)^sharpness` with `max_reward=100.0`, `sharpness=4.0`
  - Token-type penalties: element -3.0, integer -1.0, fraction -0.5, special -0.5
  - V14.1 auto-scaling: dynamically calibrates rl_weight so `|rl_weight * raw_rl_loss| ~ 10.0`
  - Entropy bonus: `combined_reward = task_reward + entropy_weight * seq_entropy`
  - RL warmup: 10%→100% over 20 epochs. Safety guard halves weight on >2% TF regression.
- **Compute cost**: ~91% of loss computation time (4x full autoregressive decode per batch)
- **Why**: CE teaches token-level prediction. RL teaches whole-sequence generation quality — it rewards producing complete, correct formulas autoregressively.

### 3. Tc Regression

Accurate critical temperature prediction from the latent space.

- **Type**: Huber + asymmetric penalty + Kelvin weighting + relative error blending + binned multipliers
- **Config**: `tc_weight: 20.0` (curriculum: ramps from 5.0 to 20.0 over 30 epochs)
- **Where**: `CombinedLossWithREINFORCE.forward()` (line ~2801)
- **Inputs**: `tc_pred [batch]` vs `tc_true [batch]` (log1p + z-score normalized)
- **Applied to**: SC samples only (weight=0 for non-SC)
- **Details**:
  - Base: `F.huber_loss(delta=1.0)` — robust to outliers
  - Asymmetric: underprediction penalized 1.5x (`tc_underpred_penalty=1.5`) — discovery bias (missing a high-Tc material is worse than overestimating)
  - Kelvin weighting: `weight = 1 + tc_kelvin / 20.0` — high-Tc samples get up to 11x weight at 200K, counteracting log1p gradient compression
  - Relative error blending: 50% Huber + 50% relative error in Kelvin space — treats 80% error uniformly regardless of Tc scale
  - Binned multipliers: `{0K: 1.0, 10K: 1.5, 50K: 2.0, 100K: 2.5, 150K: 3.0}` — progressive upweighting by Tc range
- **Why**: Tc is the single most important property for superconductor discovery. The elaborate weighting scheme ensures the model doesn't ignore rare high-Tc materials in favor of the abundant low-Tc majority.

### 4. Magpie Reconstruction

Forces the latent space to encode materials science knowledge.

- **Type**: MSE
- **Config**: `magpie_weight: 2.0` (curriculum: ramps from 1.0 to 2.0 over 30 epochs)
- **Where**: `CombinedLossWithREINFORCE.forward()` (line ~2869)
- **Inputs**: `magpie_pred [batch, 145]` vs `magpie_true [batch, 145]`
- **Applied to**: SC samples only (weight=0 for non-SC)
- **Why**: The 145 Magpie features encode elemental properties (electronegativity, atomic radius, etc.). Reconstructing them forces z to capture composition-property relationships, not just string patterns.

### 5. KL / L2 Regularization

Keeps latent vectors bounded.

- **Type**: `mean(z^2)` — L2 regularization (NOT KL divergence despite the name)
- **Config**: `kl_weight: 0.0001`
- **Where**: `FullMaterialsVAE.forward()` in `attention_vae.py` (line ~1138)
- **Inputs**: `z [batch, 2048]`
- **Applied to**: All samples
- **Why**: Named `kl_loss` for legacy compatibility but the encoder is deterministic (no reparameterization trick). This prevents z-norms from growing unbounded. Works alongside the Z-Norm Penalty (#9) which is one-sided.

### 6. Stoichiometry MSE

Teaches the encoder to predict composition ratios used as decoder conditioning.

- **Type**: Masked MSE
- **Config**: `stoich_weight: 2.0`
- **Where**: `CombinedLossWithREINFORCE.forward()` (line ~2871)
- **Inputs**: `fraction_pred [batch, max_elements]` vs `element_fractions [batch, max_elements]`, masked by `element_mask`
- **Applied to**: All samples
- **Why**: The decoder receives 4 stoichiometry memory tokens derived from this prediction. Better fraction predictions → better decoder conditioning → more accurate stoichiometry in generated formulas.

### 7. Element Count MSE

Helps the decoder know how many elements to generate.

- **Type**: MSE
- **Config**: Hardcoded weight `0.5`
- **Where**: `CombinedLossWithREINFORCE.forward()` (line ~2883)
- **Inputs**: `element_count_pred [batch]` vs `element_mask.sum(dim=1) [batch]`
- **Applied to**: All samples
- **Why**: A formula with 3 elements should generate exactly 3 element-stoich pairs before EOS. This auxiliary signal helps the model learn formula length.

### 8. Tc Bucket Classification

Auxiliary discrete signal for z-space structure.

- **Type**: Cross-Entropy (5 classes)
- **Config**: `tc_class_weight: 1.0`
- **Where**: `CombinedLossWithREINFORCE.forward()` (line ~2854)
- **Inputs**: `tc_class_logits [batch, 5]` vs `tc_bins [batch]`
- **Buckets**: 0=non-SC(Tc=0), 1=low(0-10K), 2=medium(10-50K), 3=high(50-100K), 4=very-high(100K+)
- **Applied to**: All samples
- **Why**: Classification converges faster than regression. This gives the encoder a coarse Tc signal early in training before the MSE loss (#3) has converged, helping structure the z-space by Tc range.

### 9. Z-Norm Soft Penalty

Prevents outlier z-norms that correlate with poor decoding.

- **Type**: One-sided quadratic penalty
- **Config**: `z_norm_penalty_weight: 0.001`, `z_norm_target: 22.0`
- **Where**: `CombinedLossWithREINFORCE.forward()` (line ~2958)
- **Formula**: `penalty = (clamp(||z|| - 22.0, min=0))^2`
- **Applied to**: All samples with z != None
- **Why**: Analysis showed z-norms > 30 correlate with 36pp lower exact match. This is a softer complement to the L2 reg (#5) — it only penalizes excess, not deficit.

### 10. Constraint Zoo (Composite: A3 + A5 + A6)

Differentiable physics constraints on generated formulas.

- **Config**: `constraint_zoo_weight: 0.5` (overall), individual sub-weights below
- **Where**: `CombinedLossWithREINFORCE.forward()` (line ~2888)

#### 10a. A5 Round-Trip Consistency

- **Type**: MSE on z-space and Tc reconstruction
- **Config**: `a5_round_trip_weight: 1.0`, `a5_z_weight: 1.0`, `a5_tc_weight: 5.0`
- **File**: `src/superconductor/losses/round_trip_loss.py`
- **What**: Decode z→formula→re-encode→z'. Penalizes `||z - z'||`. Ensures z-space is semantically meaningful.
- **Applied to**: 10% random subset per batch (`a5_subset_fraction=0.1`)

#### 10b. A3 Site Occupancy Sum

- **Type**: L1 deviation from target occupancy
- **Config**: `a3_site_occupancy_weight: 1.0`
- **File**: `src/superconductor/losses/constraint_zoo.py`
- **What**: Crystallographic site occupancy sums (e.g., YBCO Y-site → 1.0, LSCO La/Sr-site → 2.0)
- **Applied to**: Only when family classifier confidence > 0.8 and family has defined sites

#### 10c. A6 Charge Balance

- **Type**: tanh penalty on charge imbalance (bounded [0, 1))
- **Config**: `a6_charge_balance_weight: 1.0`, `a6_charge_tolerance: 0.5`
- **File**: `src/superconductor/losses/constraint_zoo.py`
- **What**: Charge neutrality using common oxidation states. Tolerance of 0.5 for mixed-valence materials.
- **Applied to**: All samples with valid element data

**Why constraint zoo exists**: These are chemistry priors that the model can't easily learn from token sequences alone. Site occupancy and charge balance are hard constraints in real crystals.

### 11. High-Pressure (HP) Prediction

Predicts whether a superconductor requires high pressure.

- **Type**: Binary Cross-Entropy with Logits (positive class weighted)
- **Config**: `hp_loss_weight: 1.0`
- **Where**: `train_epoch()` (line ~4661)
- **Inputs**: `encoder_out['hp_pred'] [batch]` vs `hp_labels [batch]`
- **Applied to**: SC samples only. `pos_weight` auto-computed from class imbalance (~100:1 non-HP to HP), capped at 50x.
- **Why**: Hydrogen superconductors (H3S at 203K, LaH10 at 250K) have the highest Tc values but only under extreme pressure (>100 GPa). Without this head, the model might learn to generate hydrides for high-Tc targets without encoding the pressure requirement.

### 12. SC/Non-SC Classification

Binary superconductor vs non-superconductor classification.

- **Type**: Binary Cross-Entropy with Logits
- **Config**: `sc_loss_weight: 1.0`
- **Where**: `train_epoch()` (line ~4674)
- **Inputs**: `encoder_out['sc_pred'] [batch]` vs `is_sc.float() [batch]`
- **Applied to**: All samples
- **Why**: The contrastive dataset has 23K SC + 23K non-SC materials. This head forces the z-space to have a clear SC/non-SC boundary, which is essential for targeted generation (want to generate SCs, not random materials).

### 13. Hierarchical Family Classifier

Three-level classification of superconductor families.

- **Type**: Composite of 3 Cross-Entropy losses
- **Config**: `family_classifier_weight: 0.5`
- **Where**: `train_epoch()` (line ~4684)
- **Internal weights**: coarse 0.6, cuprate-sub 0.3, iron-sub 0.1
- **Levels**:
  - Level 1 (coarse): 7 classes — cuprate, iron, conventional, MgB2, heavy-fermion, organic, other
  - Level 2a (cuprate sub): 6 classes — YBCO, LSCO, Bi-cuprate, Tl-cuprate, Hg-cuprate, other-cuprate
  - Level 2b (iron sub): 2 classes — 1111-type, 122-type
- **Applied to**: SC samples only with valid family labels
- **Why**: Family membership strongly constrains valid formulas. A YBCO must contain Y, Ba, Cu, O. An iron-pnictide must contain Fe and an As/P. This signal structures the z-space by chemistry, not just statistical pattern.

### 14. Stop Prediction

Dedicated binary head for end-of-sequence decision.

- **Type**: Binary Cross-Entropy with Logits (position-weighted)
- **Config**: `stop_loss_weight: 5.0`, `stop_end_position_weight: 10.0`
- **Where**: `train_epoch()` (line ~4636)
- **Inputs**: `stop_logits [batch, seq]` vs `stop_targets = (targets == END_IDX).float()`
- **Applied to**: All samples (valid non-PAD positions only)
- **Details**: END positions get 10x extra weight to address 1:14 class imbalance (only 1 END token per ~14 non-END tokens)
- **Why**: Knowing when to stop is critical for autoregressive generation. Collapsing the stop decision into the 4600+ token softmax dilutes the signal. A dedicated head with high weight ensures the model learns clean sequence termination.

### 15. Token Type Classification (V14.3)

Predicts the TYPE of the next token (element/integer/fraction/special/EOS).

- **Type**: Cross-Entropy (5 classes)
- **Config**: `token_type_loss_weight: 0.1`
- **Where**: `train_epoch()` (line ~4622)
- **Inputs**: `type_logits [batch, seq, 5]` vs `type_targets` (computed from tokenizer LUT)
- **Applied to**: All samples (valid non-PAD positions only)
- **Details**: See `docs/TOKEN_TYPE_CLASSIFIER_V14_3.md` for full design
- **Why**: 50% of autoregressive errors are type confusion (element where fraction should go, etc.). The formula grammar is deterministic: `[Element, Stoich, Element, Stoich, ..., EOS]`. This head learns the pattern during training; at inference, it applies a **hard mask** over vocab logits to block invalid token types.

### 16. Physics Z Supervision (DORMANT)

Enforces physical meaning on named z-coordinate blocks.

- **Type**: Composite of 10 sub-losses (MSE, SmoothL1, hinge)
- **Config**: `use_physics_z: False` (disabled; auto-reactivated by smart scheduler)
- **Where**: `train_epoch()` (line ~4742); implementation in `src/superconductor/losses/z_supervision_loss.py`
- **Sub-components**: Compositional encoding, Magpie projection, GL consistency, BCS consistency, cobordism, dimensionless ratios, thermodynamic, structural, electronic, direct supervision
- **Why**: The 2048-dim z-space has named blocks (1-11) for specific physics quantities (Ginzburg-Landau params, BCS params, crystal structure, etc.). This loss enforces inter-block consistency (e.g., GL kappa = lambda/xi) and alignment with external physics targets. Currently dormant because formula reconstruction takes priority; the scheduler auto-reactivates it when formula CE plateaus.

### 17. Theory Consistency (REMOVED — weight=0, infrastructure retained)

Family-specific physics losses (BCS Allen-Dynes, cuprate Presland dome, etc.).

- **Config**: `theory_weight: 0.0`
- **File**: `src/superconductor/losses/theory_losses.py`
- **Reason disabled**: Plateaued at 1.43, consuming 22% of gradient budget with no formula reconstruction improvement.

---

## Non-Differentiable: REINFORCE Reward Signals

These are NOT losses — they are reward modifiers that shape the REINFORCE policy gradient (#2) by changing the advantage computation. They affect training indirectly through the SCST/RLOO sampling loop.

### Task Reward

- **Formula**: `reward = max_reward * (n_correct / n_total)^sharpness`
- **Config**: `v14_max_reward: 100.0`, `v14_sharpness: 4.0`
- **File**: `src/superconductor/losses/reward_gpu_native.py`
- **Token-type penalties**: element -3.0, integer -1.0, fraction -0.5, special -0.5

### Constraint Reward Penalties (A1, A4, A7, B1-B8)

Non-differentiable physics-grounded penalties applied to REINFORCE-sampled formulas.

- **File**: `src/superconductor/losses/constraint_rewards.py`

| Constraint | Penalty | Description |
|-----------|---------|-------------|
| A1 | -50.0 | Duplicate element in formula |
| A4 | -10.0 | Non-normalized stoichiometry |
| A7 | -30.0 | Impossible element combination |
| B1 | -40.0 | YBCO: wrong oxygen content |
| B2 | -40.0 | LSCO: Sr doping out of range |
| B3 | -40.0 | BSCCO: wrong Ca/Cu content |
| B4 | -30.0 | Hg-cuprate: volatile elements present |
| B5 | -30.0 | Tl-cuprate: poison elements present |
| B6 | -30.0 | Iron-based: wrong oxygen content |
| B7 | -30.0 | MgB2: poison elements present |
| B8 | -30.0 | A15: ratio constraints violated |

B-constraints only apply when family classifier confidence > 0.8.

### Entropy Bonus

- **Formula**: `combined_reward = task_reward + entropy_weight * seq_entropy`
- **Config**: `entropy_weight: 0.2`
- **Managed by**: `EntropyManager` (dynamically adjusts between 0.05 and 1.0)
- **File**: `src/superconductor/training/entropy_maintenance.py`
- **Why**: Prevents entropy collapse (policy becoming too deterministic) during RL training.

---

## Smart Loss Skip Schedule (V12.40)

Each loss is independently tracked. When converged (below threshold), it's computed every `loss_skip_frequency` (default 4) epochs only. If it spikes above baseline + delta, it resumes every-epoch.

```python
'loss_skip_schedule': {
    # metric_key:      (converge_threshold, spike_delta)
    'tc_loss':         (0.5,   0.1),
    'magpie_loss':     (0.1,   0.1),
    'stoich_loss':     (0.01,  0.05),
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
- REINFORCE — policy gradient fluctuates by design (V13.2: removed from skip)
- KL/L2 regularization — stability
- Element count MSE — tiny cost
- Z-norm penalty — tiny cost
- Constraint zoo — tiny cost
- Token type classification — new, always learning

---

## Historical: Removed Losses

| Loss | Version Added | Version Removed | Reason |
|------|--------------|-----------------|--------|
| Contrastive | V12.12 | V12.26 | Plateaued at 5.06, 16% gradient budget, no improvement |
| NumDen MSE | V12.38 | V13.0 | Replaced by semantic fraction tokens (FRAC:p/q) |
| Theory Consistency | V12.22 | V12.26 (weight→0) | Plateaued at 1.43, 22% gradient budget |

---

## Key Files

| File | Contains |
|------|----------|
| `scripts/train_v12_clean.py` | `CombinedLossWithREINFORCE`, `train_epoch()`, all config, loss assembly |
| `src/superconductor/losses/reward_gpu_native.py` | V14.0 continuous reward, token-type penalties |
| `src/superconductor/losses/constraint_rewards.py` | A1, A4, A7, B1-B8 reward penalties |
| `src/superconductor/losses/constraint_zoo.py` | A3 site occupancy, A6 charge balance |
| `src/superconductor/losses/round_trip_loss.py` | A5 round-trip consistency |
| `src/superconductor/losses/z_supervision_loss.py` | Physics Z (10-component) |
| `src/superconductor/losses/theory_losses.py` | Theory losses (BCS, cuprate, iron, etc.) |
| `src/superconductor/losses/consistency_losses.py` | Legacy consistency (unused) |
| `src/superconductor/training/entropy_maintenance.py` | Entropy manager for RL |
| `src/superconductor/models/attention_vae.py` | Encoder KL/L2 loss |
