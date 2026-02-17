# Training Records

Chronological record of training runs, architecture changes, and optimization decisions for the Multi-Task Superconductor Generator (code class names retain "VAE" for backward compatibility).

---

## V12.38: Numerator/Denominator Conditioning for Decoder (2026-02-17)

### Problem

The decoder receives stoichiometry conditioning as 12 mole fractions + 1 element count (13 dims). Mole fractions are continuous floats that sum to ~1.0 (e.g., 0.5385 for 7/13). The decoder must reverse-engineer the exact digit token sequence `('7', '/', '1', '3')` from this float alone -- no explicit signal tells it the denominator is 13 or the numerator is 7.

### Changes

**Approach: Augment `stoich_pred` from 13 to 37 dims**

Instead of threading a new parameter through all function signatures, we widen the existing `stoich_pred` tensor:
- **Before**: `stoich_pred = [fraction_pred(12), element_count_pred(1)] = 13 dims`
- **After**: `stoich_pred = [fraction_pred(12), numden_pred(24), element_count_pred(1)] = 37 dims`

Where `numden_pred` = 12 predicted log1p(numerators) + 12 predicted log1p(denominators).

**Data preprocessing** (`scripts/train_v12_clean.py`):
- Added `parse_numden_from_formula()` to extract raw (num, den) pairs from formulas
- Stores numerators/denominators in log1p space for range compression (values range 1-500)
- Added to tensor cache with invalidation on first run after upgrade

**Encoder** (`src/superconductor/models/attention_vae.py`):
- Added `numden_head`: z -> Linear(2048,128) -> LN -> GELU -> Dropout -> Linear(128,24)
- Predicts 12 log-numerators + 12 log-denominators from latent z
- Old checkpoints: `numden_head.*` keys missing -> `strict=False` -> randomly initialized

**Decoder** (`src/superconductor/models/autoregressive_decoder.py`):
- Changed `stoich_input_dim` from `max_elements + 1` (13) to `max_elements * 3 + 1` (37)
- Checkpoint migration: partial weight preservation for `stoich_to_memory.0.weight` (first 13 cols kept, 24 new cols zero-initialized)

**Loss** (`scripts/train_v12_clean.py`):
- Added `numden_loss`: masked MSE in log1p space between predicted and target numerators/denominators
- Weight: `numden_weight = 1.0` (configurable via TRAIN_CONFIG)
- Added to combined total loss alongside stoich_loss

**Modified Files:**
- `scripts/train_v12_clean.py` -- Parsing, preprocessing, cache, loss, assembly, logging
- `src/superconductor/models/attention_vae.py` -- numden_head in encoder
- `src/superconductor/models/autoregressive_decoder.py` -- stoich_input_dim 13->37

### Checkpoint Compatibility

| Component | Old -> New | Handled By |
|-----------|-----------|------------|
| `encoder.numden_head.*` | Missing keys | `strict=False` -> randomly initialized |
| `decoder.stoich_to_memory.0.weight` | `[512, 13]` -> `[512, 37]` | Shape migration (partial preserve first 13 cols) |
| `decoder.stoich_to_memory.0.bias` | `[512]` -> `[512]` | No change |

---

## V12.37: Plateau-Breaking Interventions (2026-02-16)

### Problem

Model stuck at ~61-65% exact match (epochs 2764-2812). Plateau analysis identified three root causes:
1. **Extra-append failure** (257 samples, 12.75%): Model outputs correct formula then appends extra tokens instead of stopping. Fixing this alone: 63.69% → 76.44%.
2. **Fraction cascade errors** (177 samples): One wrong digit cascades through all subsequent tokens.
3. **RL reward blindness**: Sample with 39/40 correct tokens gets low reward — no gradient signal to "just stop one token earlier."

### Changes

**Intervention 1: Aggressive Stop Token Training**
- `stop_boost`: 4.0 → 10.0 (stronger END logit boost from stop head)
- `hard_stop_threshold`: 0.8 (force END when sigmoid(stop_logit) > 0.8)
- `stop_end_position_weight`: 10.0 (10x weight on END positions in stop BCE loss, addresses 1:14 class imbalance)
- Length-conditional stop boost: after position 10, adds `stop_boost * 0.1 * (position - 10)` to END logit

**Intervention 2: Near-Miss Reward Shaping**
- Length-only error detection in `reward_gpu_native.py`: detects "perfect prefix, just too long" samples
- New `GPURewardConfig` fields: `length_only_base_reward=50.0`, `length_only_per_extra=5.0`, `length_only_floor=10.0`
- These samples get high reward (50 - 5*extra_tokens, min 10) instead of being penalized

**Intervention 3: Hard Sequence Oversampling**
- `oversample_hard_sequences`: True (upweight long/complex formulas in sampler)
- `oversample_length_base`: 15 (sequences > 15 tokens get progressively upweighted)
- Also upweights by element count (4+ elements get 1.5x, 5+ get 2x)

**Intervention 4: Integer-to-Fraction Normalization**
- `normalize_integers_to_fractions`: True (convert `Ba8Cu12O28PrY3` → fraction format)
- Reduces decoder vocabulary ambiguity — only one representation format to learn
- Triggers cache rebuild (new cache invalidation key)

**Modified Files:**
- `scripts/train_v12_clean.py` — Config, position-weighted stop loss, integer normalization, oversampling
- `src/superconductor/models/autoregressive_decoder.py` — Hard stop threshold, length-conditional boost
- `src/superconductor/losses/reward_gpu_native.py` — Length-only error detection + reward shaping

**Deferred to V12.38+:**
- Fraction pre-tokenization (atomic fraction tokens, +5-10% potential but requires vocabulary redesign and full retrain)

---

## V12.36b: Family Prediction Diagnostics in Error Reports (2026-02-16)

### Problem

Family classification loss was being computed during training (coarse family head), but the error reports contained zero family data — no per-sample family labels, no per-family accuracy breakdown, no correlation between family misclassification and formula errors. This made it impossible to diagnose whether certain superconductor families decode worse than others.

### Changes

**Modified Files:**
- `scripts/train_v12_clean.py` — Updated `evaluate_true_autoregressive()` to collect family predictions and add them to error reports
- `scratch/analyze_error_reports.py` — Added `analyze_family_diagnostics()` function

### What's Saved

**Per failed sample** (`error_records[i]`):
```json
"family_true": "CUP_YBCO",           // Fine label (14 classes)
"family_coarse_true": "Cuprate",      // Coarse label (7 classes)
"family_coarse_pred": "Cuprate",      // Model's coarse prediction
"family_correct": true                // Whether coarse prediction matches
```

**Aggregate** (`z_diagnostics.family_diagnostics`):
```json
{
  "coarse_accuracy": 0.8723,
  "corr_family_wrong_vs_formula_errors": 0.142,
  "errors_by_family": {
    "BCS": {"n_samples": 1200, "family_accuracy": 0.95, "formula_exact_pct": 82.1, "avg_formula_errors": 1.2},
    "Cuprate": {"n_samples": 8000, "family_accuracy": 0.88, "formula_exact_pct": 75.3, "avg_formula_errors": 2.1},
    ...
  }
}
```

### Console Output

Added to eval summary line:
```
Family: coarse_acc=87.2% | fam_wrong→err corr=0.142
```

---

## V12.36: Self-Consistency Losses for Unsupervised Z Blocks (2026-02-16)

### Problem

Analysis of the trained latent space revealed that the 6 unsupervised physics Z blocks (Eliashberg, Unconventional, Structural, Electronic, Thermodynamic, Discovery) encode active, material-discriminative information — but with no physics guardrails. For example:
- z[TC] (coord 210, designated Tc coordinate) only correlates r=0.49 with actual Tc, while a random discovery coord hits r=0.87
- Tc ordering (onset >= midpoint >= zero) holds only 42% of the time
- Structural coords (volume, lattice params) have no consistency enforcement

The encoder puts information wherever reconstruction is easiest, not where physics says it should be.

### Changes

**Modified Files:**
- `src/superconductor/losses/z_supervision_loss.py` — Added 3 new loss classes: `ThermodynamicConsistencyLoss`, `StructuralConsistencyLoss`, `ElectronicConsistencyLoss`. Updated `PhysicsZLoss` to include them with `new_consistency_weight` config.
- `scripts/train_v12_clean.py` — Added `physics_z_new_consistency_weight` TRAIN_CONFIG entry (default 0.05). Passes `tc_normalized=tc` to PhysicsZLoss.

### Constraint Details

All constraints are **soft** (penalties, not hard clamps) and use SmoothL1Loss (Huber) for numerical stability. Only mathematical identities and well-established physics are enforced.

#### Thermodynamic Block (Block 7, coords 210-269)
- **z[TC] ≈ tc_normalized**: Direct supervision from the Tc input we already have. Forces the designated TC coordinate to actually hold Tc information.
- **Ordering**: Hinge loss on violations of Tc_onset >= Tc_midpoint >= Tc_zero. No gradient when ordering holds (zero penalty for correct samples).
- **Delta_Tc = Tc_onset - Tc_zero**: Mathematical identity — transition width is defined as the difference.

#### Structural Block (Block 5, coords 110-159)
- **Volume ∝ a * b * c**: Unit cell volume is proportional to lattice parameter product. Exact for orthogonal crystal systems, approximate for others.

#### Electronic Block (Block 6, coords 160-209)
- **Drude weight ∝ plasma_freq²**: Standard Drude model result, valid for all metals.

### What Was NOT Constrained (and why)

| Block | Why Skipped |
|-------|-------------|
| **Unconventional** (70-109) | Gap symmetry, spin fractions, doping — too model-dependent (assumes specific pairing mechanisms). Would penalize unconventional superconductors. |
| **Eliashberg** (50-69) | Coupling relationships overlap with BCS block constraints — risk of double-constraining. Spectral function parameters too esoteric without external data. |
| **Discovery** (512-2047) | Intentionally unsupervised. The whole point is free representation learning. |

### New TRAIN_CONFIG Key

| Key | Default | Description |
|-----|---------|-------------|
| `physics_z_new_consistency_weight` | `0.05` | Weight for V12.36 consistency losses (thermo + structural + electronic). Deliberately lower than existing consistency weight (0.1) to avoid over-constraining. |

### Design Philosophy

These constraints are **safeguards**, not supervision. They ensure the Z coordinates aren't nonsensical without pigeon-holing the network into specific physical models. The weight (0.05) is gentle — strong enough to impose basic consistency, weak enough that the encoder can deviate if reconstruction demands it.

---

## V12.35: Per-Block Physics Z Diagnostics in Error Reports (2026-02-16)

### Problem

Error reports only saved scalar `z_norm` and `z_max_dim` per sample — no visibility into which of the 12 Physics Z coordinate blocks (GL, BCS, Eliashberg, etc.) are healthy vs degraded. This made it impossible to diagnose whether errors correlate with specific physics domains.

### Changes

**Modified Files:**
- `scripts/train_v12_clean.py` — Import `PhysicsZ`, compute per-block L2 norms in eval loop, add `z_block_norms` dict to each `error_record`, add aggregate `z_block_diagnostics` and `z_block_corr_ranked` to `z_diagnostics`
- `scratch/analyze_error_reports.py` — Added `analyze_physics_z_blocks()` function with per-block exact-vs-error comparison table, correlation ranking, worst-error block breakdown, and cross-epoch trend tracking

### What's Saved

**Per failed sample** (`error_records[i]`):
```json
"z_block_norms": {
    "gl": 4.21, "bcs": 5.13, "eliashberg": 3.88,
    "unconventional": 6.72, "structural": 7.44,
    "electronic": 6.91, "thermodynamic": 8.12,
    "compositional": 9.33, "cobordism": 7.01,
    "ratios": 5.89, "magpie": 8.55, "discovery": 38.2
}
```

**Aggregate** (`z_diagnostics.z_block_diagnostics[block_name]`):
- `overall`: mean/std of block norm across all samples
- `exact`: mean/std for exact-match samples
- `error`: mean/std for error samples
- `exact_error_gap`: error_mean - exact_mean (positive = errors have higher norm)
- `corr_vs_errors`: Pearson correlation of block norm with number of errors

**Ranked** (`z_diagnostics.z_block_corr_ranked`):
- Blocks sorted by |correlation| with errors, descending

### Console Output

Top-3 blocks most correlated with errors are printed during eval:
```
Z-blocks (top corr→err): discovery=0.231 | thermodynamic=0.187 | compositional=0.142
```

### Notes

- Block norms squared sum to total z_norm squared (verified mathematically)
- 6 blocks have active PhysicsZLoss supervision: GL (consistency, w=0.1), BCS (consistency, w=0.1), Compositional (direct MSE, w=1.0), Cobordism (derived from GL, w=0.1), Ratios (cross-block checks, w=0.1), Magpie (learnable projection, w=0.5)
- 6 blocks are currently unsupervised (only KL/L2 regularized): Eliashberg, Unconventional, Structural, Electronic, Thermodynamic, Discovery. These are NOT forced to zero — they can learn self-consistent representations via formula reconstruction backprop
- Zero overhead during training; only computed during evaluation

---

## V12.34: Error-Driven Training Refinements A-E + Tc Range Fix (2026-02-16)

### Problem

Error analysis of epochs 2764-2812 reveals the model is in a noisy plateau at 60-65% exact match. The bottleneck is the autoregressive decoder, not the latent representation. Five specific error drivers identified:

1. **Sequence length** (r=0.342) -- 50% of errors in final third of sequences
2. **Z-norm** (r=0.236) -- Q1: 79% exact vs Q4: 43% exact (36pt gap)
3. **Element count** -- 4+ elements: 9-14 avg errors vs 3-4 for 2-3 elements
4. **Fraction representation** -- denominator drift, model biases toward "common" denoms
5. **Error cascade** -- late-position errors compound, causing truncation/overgeneration

Additionally, the Tc range analysis in error reports had a **bug** (comparing normalized values against Kelvin thresholds).

### Changes

**Modified Files:**
- `scripts/train_v12_clean.py` -- TRAIN_CONFIG entries (A-E), `canonicalize_fractions()`, `FocalLossWithLabelSmoothing` per-sample reduction, `CombinedLossWithREINFORCE` A/D weighting + C z-norm penalty, Tc range denormalization fix, `evaluate_true_autoregressive()` gets `norm_stats` param
- `src/superconductor/models/autoregressive_decoder.py` -- `EnhancedTransformerDecoder` position-dependent TF params + use_gt_mask logic
- `scratch/analyze_error_reports.py` -- Added `analyze_tc_ranges()`, updated trends with Kelvin R², updated summary
- `docs/TRAINING_RECORDS.md` -- This entry

### Refinement Details

#### A: Sequence-Length Weighted Loss
- Longer sequences (>15 tokens) get higher formula loss weight
- `w = 1 + alpha * max(0, (len - base) / base)` where base=15, alpha=1.0
- len=30 -> w=2.0, len=45 -> w=3.0
- Targets: 50% of errors concentrated in long sequences
- Config: `use_length_weighting`, `length_weight_base`, `length_weight_alpha`

#### B: Fraction Canonicalization
- GCD-reduces all fractions before tokenization (e.g., 6/10 -> 3/5)
- Eliminates ambiguity when multiple equivalent representations exist
- Some formulas get shorter token sequences
- Triggers automatic cache rebuild via `use_canonical_fractions` in cache metadata
- Config: `use_canonical_fractions`

#### C: Z-Norm Soft Penalty
- Soft barrier on z-norms above target (current mean ~22)
- Penalty: `0.001 * mean(clamp(||z|| - 22, min=0)^2)`
- No effect on z-norms below target; only penalizes Q3-Q4 outliers
- Complements existing L2/kl_loss (which regularizes all dims uniformly)
- Config: `use_z_norm_penalty`, `z_norm_target`, `z_norm_penalty_weight`

#### D: Element-Count Weighted Loss
- Formulas with more elements get higher formula loss weight
- `w = 1 + beta * max(0, n_elem - base)` where base=3, beta=0.5
- n=5 -> w=2.0, n=7 -> w=3.0
- Combined with A (multiplicative) in per-sample formula loss
- Config: `use_element_count_weighting`, `element_count_base`, `element_count_beta`

#### E: Position-Dependent Teacher Forcing (Infrastructure Only)
- When TF < 1.0: `tf(pos) = base_tf * (1 + gamma * (1 - pos/L))`
- Start of sequence gets higher TF, end gets lower TF
- **Currently has no effect**: TF is always 1.0 (full teacher forcing)
- Infrastructure ready for when TF is reduced
- Config: `use_position_dependent_tf`, `tf_position_decay`

#### F: Tc Range Analysis Bug Fix
- **Bug**: `all_tc_true_np` contained normalized values (log1p + z-score), but Tc range buckets compared against Kelvin thresholds (10, 30, 77...). Only "0-10K" ever matched.
- **Fix**: Denormalize to Kelvin before bucketing. Added `norm_stats` parameter to `evaluate_true_autoregressive()`.
- Added per-range R², MAE (Kelvin), and max error to each Tc bucket
- Added overall `tc_r2_kelvin` and `tc_mae_kelvin_overall` to z_diagnostics

### New TRAIN_CONFIG Keys

| Key | Default | Description |
|-----|---------|-------------|
| `use_length_weighting` | `True` | A: Enable seq-length weighted formula loss |
| `length_weight_base` | `15` | A: Seqs <= this get weight 1.0 |
| `length_weight_alpha` | `1.0` | A: Scale factor for length weight |
| `use_canonical_fractions` | `True` | B: GCD-reduce fractions before tokenization |
| `use_z_norm_penalty` | `True` | C: Penalize extreme z-norms |
| `z_norm_target` | `22.0` | C: Z-norm threshold (current mean) |
| `z_norm_penalty_weight` | `0.001` | C: Penalty weight (gentle) |
| `use_element_count_weighting` | `True` | D: Enable element-count weighted loss |
| `element_count_base` | `3` | D: Formulas <= this get weight 1.0 |
| `element_count_beta` | `0.5` | D: Scale factor for element count weight |
| `use_position_dependent_tf` | `True` | E: Position-dependent TF (no effect at TF=1.0) |
| `tf_position_decay` | `0.5` | E: Decay gamma for position TF |

### Disabling Individual Features

All features are independently togglable. To disable any single feature, set its `use_*` key to `False`:
```python
'use_length_weighting': False,     # Disable A
'use_canonical_fractions': False,  # Disable B (triggers cache rebuild)
'use_z_norm_penalty': False,       # Disable C
'use_element_count_weighting': False,  # Disable D
'use_position_dependent_tf': False,    # Disable E
```

### Overhead

Total compute overhead: <0.5% of epoch time. No memory overhead. Fraction canonicalization triggers a one-time cache rebuild (~2 min).

---

## V12.33: Hierarchical Family Classification Head (2026-02-16)

### Problem

V12.32's flat 14-class `family_head` had structural issues:
1. **NOT_SC dominates**: ~23K non-SC samples (50%) compete in the same softmax as 13 SC families
2. **No hierarchy**: YBCO-vs-LSCO (both cuprates) gets the same penalty as YBCO-vs-NOT_SC
3. **No conditioning on sc_head**: Family and SC predictions are independent (can predict "CUPRATE_YBCO" while also predicting "not superconductor")
4. **Rare class suppression**: ORGANIC, HEAVY_FERMION (<100 samples) swamped in 14-way softmax

### Changes

**Modified Files:**
- `src/superconductor/models/attention_vae.py` -- Added `HierarchicalFamilyHead` class, replaced `family_head` with `hierarchical_family_head`, updated `decode()` to expose `backbone_h` and remove `family_logits`, updated `forward()` to call hierarchical head conditioned on `sc_pred.detach()`
- `scripts/train_v12_clean.py` -- Added `build_family_lookup_tensors()` helper, updated config with internal weights, replaced flat CE with 3-level hierarchical loss, added `family_lookup_tables` parameter to `train_epoch()`
- `docs/TRAINING_RECORDS.md` -- This entry

### Architecture

Replaced flat `family_head` with `HierarchicalFamilyHead` — a 3-level tree conditioned on `sc_pred`:
```
P(NOT_SC)   = 1 - P(SC)                                    ← from sc_head
P(BCS)      = P(SC) × P(BCS|SC)                            ← coarse head (7 classes)
P(YBCO)     = P(SC) × P(Cuprate|SC) × P(YBCO|Cuprate)     ← coarse × cuprate sub-head
P(PNICTIDE) = P(SC) × P(Iron|SC) × P(Pnictide|Iron)       ← coarse × iron sub-head
```

Each head receives `h(512) + sc_prob(1)` as input (backbone output concatenated with sigmoid of detached sc_pred logit).

```
Level 1 — Coarse (7 classes, SC samples only):
  0: BCS_CONVENTIONAL, 1: CUPRATE, 2: IRON, 3: MGB2, 4: HEAVY_FERMION, 5: ORGANIC, 6: OTHER_UNKNOWN

Level 2a — Cuprate sub-family (6 classes):
  0: YBCO, 1: LSCO, 2: BSCCO, 3: TBCCO, 4: HBCCO, 5: OTHER

Level 2b — Iron sub-family (2 classes):
  0: PNICTIDE, 1: CHALCOGENIDE
```

### Loss

Separate CE at each level on appropriate subsets:
- Coarse: all SC samples (7-class)
- Cuprate sub: cuprate samples only (6-class)
- Iron sub: iron samples only (2-class)

Internal weights: coarse 0.6, cuprate 0.3, iron 0.1. Master weight 0.5.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SC conditioning | `sc_pred.detach()` | Prevents gradient entanglement with sc_head |
| Gating | Soft (multiply by P(SC)) | Preserves differentiability |
| NOT_SC handling | `1 - P(SC)`, not in softmax | NOT_SC is fundamentally different from "which family" |
| Loss structure | Separate CE per level | No gradient from irrelevant samples |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_family_classifier` | True | Enable hierarchical family head |
| `family_classifier_weight` | 2.0 | Master weight for combined hierarchical loss (boosted from 0.5) |
| `family_coarse_weight` | 0.6 | Internal: 7-class coarse CE weight |
| `family_cuprate_sub_weight` | 0.3 | Internal: 6-class cuprate sub CE weight |
| `family_iron_sub_weight` | 0.1 | Internal: 2-class iron sub CE weight |

### Gradient Rebalancing (Option 3)

RL loss at `rl_weight=2.5` was consuming ~55% of total gradient budget, drowning auxiliary signals (family, tc_class). After verifying the hierarchical head trains (Fam: 0.9292, below random baseline ~1.78), rebalanced weights to let structural heads learn:

| Weight | Before | After | Rationale |
|--------|--------|-------|-----------|
| `rl_weight` | 2.5 | 1.0 | RL has had extensive training; reduce dominance |
| `family_classifier_weight` | 0.5 | 2.0 | 4x boost to strengthen hierarchical family signal |
| `tc_class_weight` | 2.0 | 4.0 | 2x boost to strengthen Tc bucket classification |

Rationale: The model has done extensive RL training already. The RL gradient mainly prevents the network from drifting into "out of bounds" regions. Boosting auxiliary weights lets the family and Tc classification heads catch up without RL overwhelming their gradients.

### Checkpoint Compatibility

Old `family_head` parameters are ignored (absent in new model). New `hierarchical_family_head` initializes randomly when loading old checkpoints (`strict=False`).

### Parameter Count

~272K parameters (was ~135K for flat head) — negligible vs ~30M total model.

---

## V12.32: Family Classification Head (2026-02-16)

*Superseded by V12.33 (hierarchical family head).*

Introduced flat 14-class `family_head` from decoder backbone. Replaced in V12.33 with `HierarchicalFamilyHead` to address NOT_SC dominance, lack of hierarchy, and no sc_head conditioning.

---

## V12.31: Physics-Supervised Z Coordinates (2026-02-14)

### Problem

The 2048-dim Z vector has no enforced physical meaning. While it encodes information needed for reconstruction, there is no guarantee that specific coordinates correspond to physically meaningful quantities. This makes the latent space hard to interpret and prevents physics-guided generation.

### Changes

**New Files:**
- `src/superconductor/models/physics_z.py` -- `PhysicsZ` class defining 12 coordinate blocks (0-511 supervised, 512-2047 discovery)
- `src/superconductor/losses/z_supervision_loss.py` -- 7 loss classes + `PhysicsZLoss` combiner
- `src/superconductor/data/compositional_targets.py` -- Computes 15 Block 8 targets from formula
- `docs/PHYSICS_Z_COORDINATES.md` -- Full documentation of block layout, formulas, and usage

**Modified Files:**
- `scripts/train_v12_clean.py` -- TRAIN_CONFIG keys, dataset augmentation, loss wiring, logging
- `src/superconductor/losses/__init__.py` -- New exports
- `src/superconductor/models/__init__.py` -- PhysicsZ export

### Architecture

No changes to FullMaterialsVAE. The encoder still outputs 2048-dim Z from `fc_mean`. Physics meaning is enforced purely via loss gradient pressure. Existing checkpoints load unchanged.

### Loss Components

| Component | Block | Source | Weight |
|-----------|-------|--------|--------|
| CompositionalSupervisionLoss | 8 | Formula-derived (always available) | 1.0 |
| MagpieEncodingLoss | 11 | Learnable projection 145->62 | 0.5 |
| GLConsistencyLoss | 1 | Self-consistency (kappa=lambda/xi, etc.) | 0.1 |
| BCSConsistencyLoss | 2 | Self-consistency (xi proportional to vF/Delta0) | 0.1 |
| CobordismConsistencyLoss | 9 | Derived from GL coords | 0.1 |
| DimensionlessRatioConsistencyLoss | 10 | Cross-block ratios | 0.1 |
| DirectSupervisionLoss | any | External data (placeholder, weight=0) | 0.0 |

### Parameters Added

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_physics_z` | True | Enable physics Z infrastructure |
| `physics_z_comp_weight` | 1.0 | Block 8 compositional supervision |
| `physics_z_magpie_weight` | 0.5 | Block 11 Magpie encoding |
| `physics_z_consistency_weight` | 0.1 | GL/BCS/cobordism/ratio consistency |
| `physics_z_direct_weight` | 0.0 | Direct supervision (placeholder) |
| `physics_z_warmup_epochs` | 20 | Linear warmup ramp |
| `physics_z_data_path` | None | Path to optional physics data CSV |

### Design Decisions

- **No architecture change**: Checkpoint compatibility preserved. Physics Z is loss-only.
- **Proportional consistency**: GL/BCS formulas enforce ratios, not absolute physical units. The encoder learns its own internal scale.
- **Pre-computed compositional targets**: Computed once during dataset creation, stored as tensor index 10 in TensorDataset. Follows existing pattern.
- **MagpieEncodingLoss learns jointly**: Both encoder and projection trained together (~9K new params in projection).
- **Warmup**: 20-epoch linear ramp prevents physics Z from overwhelming early training.
- **Coexists with theory_losses.py**: Theory losses operate on Tc + Magpie. Physics Z operates on Z coordinates. Complementary, not competing.

### Performance Optimizations (V12.31)

- **CUDA event timing**: Replaced `torch.cuda.synchronize()` in `TimingStats` with async `torch.cuda.Event(enable_timing=True)`. Previously, synchronize was called at every phase start/stop (~16,800 calls per epoch), breaking GPU pipeline overlap. Now events are recorded asynchronously and flushed once at epoch end. Works on all CUDA devices (RTX 4060, A100, Colab).
- **Autoregressive eval frequency**: Reduced `evaluate_true_autoregressive` from every epoch to every 4 epochs. This eval runs 2 full autoregressive decoder passes x 60 sequential steps per sample — expensive. Intermediate evals use 2,000 samples; final epoch evaluates ALL ~50K samples (SC + non-SC) for the complete picture.

---

## V12.30: Stop-Prediction Head (2026-02-13)

### Problem

279 samples (13.8% of all data) at epoch 2542 generate the **perfect formula** then fail to emit `<END>`, appending an extra element token instead. This is the #1 error type (34.4% of all errors). Fixing it would improve true autoregressive exact match from ~60% to ~73%.

**Root cause (two complementary failures)**:

1. **Architectural**: `<END>` (token index 2) competes in softmax with 150+ other tokens via `output_proj`. At the stop position, the model must assign higher probability to END than to any element, but element logits are strongly reinforced by the formula pattern while END is a structurally different decision.
2. **REINFORCE reward bug**: Near-exact cases (1-3 mismatches) get their reward bonus **without length penalty**. "Perfect formula + 1 extra token" scores +50.0 because the length_mismatch_penalty was only applied to partial_credit cases (4+ mismatches). The model is rewarded for this failure.

### Changes

**`src/superconductor/models/autoregressive_decoder.py`**
- Added `stop_head` to `EnhancedTransformerDecoder.__init__()`: `Linear(512->128) + GELU + Linear(128->1)` (~65K params, 0.1% of model)
- `forward()` now returns `(logits, generated, stop_logits)` (was `(logits, generated)`)
- `generate_with_kv_cache()`: new `stop_boost` parameter. When > 0, boosts `END_IDX` logit by `stop_boost * sigmoid(stop_logit)` at each generation step
- `generate()` (legacy): same `stop_boost` parameter added
- `sample_for_reinforce()`: passes `stop_boost` through to `generate_with_kv_cache()`

**`scripts/train_v12_clean.py`**
- New TRAIN_CONFIG keys: `stop_loss_weight` (5.0), `stop_boost` (4.0)
- Decoder forward unpacks `stop_logits` (third return value)
- Computes BCE stop loss: target=1.0 at END positions, 0.0 elsewhere, masked to non-PAD
- Stop loss added to all three loss computation branches (pure SC, pure non-SC, mixed)
- `CombinedLossWithREINFORCE.set_decoder()`: accepts and stores `stop_boost`
- All REINFORCE sampling calls (RLOO and SCST) pass `stop_boost` through
- `evaluate_true_autoregressive()`: new `stop_boost` parameter, passed to generation
- Epoch summary shows stop loss: `Stop: X.XXXX`
- New metric accumulators: `total_stop_loss`, `stop_loss` in results dict

**`src/superconductor/losses/reward_gpu_native.py`**
- Applied `length_penalty` (`length_diff * config.length_mismatch_penalty`) to all three near-exact reward tiers (was only applied to partial_credit). "Perfect + 1 extra token" now gets +50.0 - 2.0 = +48.0 instead of +50.0.

### Design Decisions

- **Soft boost, not hard override**: At inference, stop_head boosts the END logit additively rather than forcing END. Avoids edge cases where the model correctly continues.
- **BCE loss, not cross-entropy**: The stop decision is binary (stop/continue), so BCE is the natural loss.
- **High stop_loss_weight (5.0)**: Only 1 END token per sequence of ~15 tokens. Without upweighting, the stop head sees 14:1 class imbalance.
- **stop_boost=4.0**: A +4.0 additive boost when sigmoid(logit)=1.0 shifts a marginal END probability (~0.3) to dominant (>0.8) in softmax. Conservative enough to avoid premature stopping.
- **Reward fix is complementary**: The stop head teaches "when to stop" during CE training. The reward fix prevents RL from un-teaching it.
- **Old checkpoints**: stop_head params will be randomly initialized when loading pre-V12.30 checkpoints. No crash, but first few epochs will have high stop_loss until the head converges.

### Parameters Added

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stop_loss_weight` | 5.0 | BCE loss weight for stop-prediction head |
| `stop_boost` | 4.0 | Additive END logit boost at inference (0 = disabled) |

### Expected Impact

| Change | Expected Impact |
|--------|----------------|
| Stop head on decoder | +10-13 pp true autoregressive exact match |
| Fix reward length penalty | +2-3 pp (prevents RL from reinforcing the bug) |
| Stop loss in training | Trains the stop head (necessary for stop head to work) |

---

## V12.29: Training Manifest System (2026-02-13)

### Problem

No version tracking embedded in training artifacts (checkpoints, latent caches). When loading a checkpoint saved weeks ago, there was no way to know what code version produced it, what MODEL_CONFIG / TRAIN_CONFIG were active, what dataset was used, or whether the current code/config had drifted. This caused real pain during the V12.28 migration when exact match dropped to 0% due to a magpie_dim mismatch that wasn't detectable from the checkpoint itself.

### Changes

**New file: `src/superconductor/utils/manifest.py`** (~200 lines)
- `get_git_info()` — captures commit, branch, dirty state
- `get_environment_info()` — captures Python, PyTorch, CUDA, GPU
- `get_model_architecture_fingerprint()` — captures all parameter names and shapes
- `compute_config_hash()` — stable SHA-256 hash of config dicts
- `build_manifest()` — assembles complete manifest dict for embedding
- `check_config_drift()` — compares saved vs current manifest with tiered warnings:
  - `[CRITICAL]`: architecture fingerprint mismatch (param shapes differ)
  - `[WARNING]`: model_config_hash or train_config_hash changed
  - `[INFO]`: git commit, environment, or dataset changed

**Modified: `src/superconductor/models/attention_vae.py`**
- Added `get_config()` method to `FullMaterialsVAE` — returns constructor parameters for manifest

**Modified: `scripts/train_v12_clean.py`**
- `save_checkpoint()`: new `manifest=None` parameter, embedded in checkpoint_data
- `cache_z_vectors()`: new `manifest=None` parameter, embedded in cache_data
- `train()`: builds dataset fingerprint and manifest helper after data/model creation
- Config drift detection on checkpoint resume (before weight loading)
- Signal handler and emergency saves include manifest

**Version bump**: `superconductor.__version__` 0.1.0 → 0.2.0

### Design Decisions

- **train_config stored as hash only** — too large and changes between runs; hash detects drift
- **model_config stored in full** — small (8 keys), directly affects architecture compatibility
- **Dataset fingerprint uses row count + magpie_dim** — not a full CSV hash (too expensive for 50K rows)
- **Backward compatible** — old checkpoints without manifests load fine with a single info log line
- **Drift detection runs before weight loading** — warns early, doesn't block loading

---

## V12.26: Disable Contrastive & Theory Losses — Unblock Plateau (2026-02-11)

### Problem

Training plateaued at ~87% exact match (TF) / ~58% true autoregressive for 500+ epochs (epoch 1776-2345). Loss composition analysis revealed:

| Component | Weighted Loss | % of Total |
|-----------|--------------|------------|
| Reconstruction | ~0.104 | 33% |
| Theory (w=0.05) | 0.070 | 22% |
| Magpie | 0.068 | 22% |
| Contrastive (w=0.01) | 0.051 | 16% |
| Tc/Stoich/Other | 0.023 | 7% |

**Contrastive + Theory = 38% of total loss** but both completely flat:
- Contrastive: stuck at ~5.06 for hundreds of epochs
- Theory: stuck at ~1.43 for hundreds of epochs

These provide large, static gradient fields that dilute the reconstruction gradient — the only signal that actually improves formula generation. The entropy system confirmed the plateau: `[Entropy] Boost ended: FAILURE - plateau persists (history: 8 interventions, 1/8 successful)`.

### Changes

**Disable plateaued losses:**
- `contrastive_weight`: 0.01 → **0.0** (disabled — plateaued at 5.06, 16% of gradient budget)
- `theory_weight`: 0.05 → **0.0** (disabled — plateaued at 1.43, 22% of gradient budget)

**Strengthen Tc accuracy for generation quality:**
- `tc_weight`: 10.0 → **20.0** (2x global increase)
- `tc_kelvin_weight_scale`: 50.0 → **20.0** (high-Tc focus: 100K material gets 6x weight vs old 3x)
- Net effect on 150K material: 10×4=40 → 20×8.5=**170** (4.25x more gradient)
- Rationale: 100K+ materials have 6.5K MAE — unacceptable for a generative model. If z doesn't accurately encode what enables high-Tc, generation from that region produces unreliable candidates.

**Strengthen high-pressure prediction:**
- `hp_loss_weight`: 0.05 → **0.5** (10x — HP is critical for high-Tc families like H₃S at 203K, LaH₁₀ at 250K+)

**Disable REINFORCE (zero gradient for 500+ epochs):**
- `rl_weight`: 1.5 → **0.0** (disabled)
- RLOO advantages were ≈0 for 500+ epochs — both samples always nearly identical, so no RL gradient ever flowed
- The reward metric (0→60) tracked exact match improvement but was a *consequence* of CE training, not a cause
- REINFORCE sampling consumed **84% of epoch time** (~100s out of 119s) — pure overhead
- Disabling drops epoch time from ~119s to ~19s (**6x speedup**, ~180 epochs/hour)
- Earlier attempt to revive RL via `rl_temperature` 0.8→1.5 produced diverse but low-quality samples (reward went negative) with advantages still ≈0
- Reset entropy intervention history on resume (stale from prior RL regime)

Config-only changes. All infrastructure (contrastive, theory, REINFORCE) remains intact for future re-enablement.

### V12.27+ Roadmap: Theory as Post-Mastery Regularizer

See **[docs/THEORY_LOSS_ROADMAP.md](THEORY_LOSS_ROADMAP.md)** for the full design document. Key insights:

1. **Phased curriculum:** Theory losses should be applied AFTER data mastery (Phase 2), not during active learning. At weight 0.01-0.02, they refine without disrupting.
2. **Self-consistency as unsupervised learning:** Generated candidates can be checked against physics constraints without labels — a form of self-supervised training that scales with model capability.
3. **Theory validation via model accuracy:** Applying a theory as a loss function tests it against 46K materials simultaneously. Theories that improve accuracy are validated; theories that destroy it are falsified. The model becomes a theory-testing instrument.

### Expected Loss Composition (projected)

| Component | Old Weighted | New Weighted | % of New Total |
|-----------|-------------|-------------|----------------|
| Reconstruction | 0.101 | 0.101 | 42% |
| Tc (w=20, scale=20) | 0.016 | ~0.040 | 17% |
| Magpie | 0.068 | 0.068 | 28% |
| HP (w=0.5) | 0.005 | ~0.052 | 8% |
| Stoich | 0.004 | 0.004 | 2% |
| Contrastive | 0.051 | 0.0 | 0% |
| Theory | 0.070 | 0.0 | 0% |
| **Total** | **0.315** | **~0.240** | |

### Key Design Rationale

The model's purpose is **generation of novel superconductor candidates**. Training error in Tc and HP prediction compounds during generation — a model that's blurry on what enables high-Tc will produce unreliable candidates from the high-Tc region of latent space. The Kelvin weighting change (scale 50→20) specifically concentrates gradient on the 947 samples above 100K where MAE is worst (6.5K), while barely affecting the 16K+ low-Tc samples already at 0.1K MAE.

### Note

The contrastive and theory losses served their purpose during earlier training (structure learning, physics regularization). At this stage the model has internalized those patterns and the losses became dead weight. They can be re-enabled at reduced weights for fine-tuning if needed.

---

## V12.25: Theory Loss Overhaul — Allen-Dynes, Family Priors, VEC Constraints (2026-02-10)

### Problem

The theory loss (`theory_losses.py`) had major gaps:

1. **BCS used McMillan (1968)** — only ~20-30% accurate for strong-coupling materials. Allen-Dynes (1975) adds f1, f2 correction factors that significantly improve accuracy for lambda > 0.5 materials like MgB2.
2. **Heavy Fermion got ZERO loss** — mapped to `UnknownTheoryLoss` which returns 0. These materials have well-known Tc constraints (typically 0.1-2K, max ~18K for PuCoGa5).
3. **Organic got ZERO loss** — same as heavy fermion. BEDT-TTF salts: Tc < 13K; alkali-doped fullerenes: up to ~40K.
4. **Iron-based was trivial** — only a soft cap at 60K, no physics beyond that. No VEC or sub-family constraints.
5. **No universal Matthias VEC prior** for BCS — Tc peaks at VEC~5 and VEC~7 for conventional superconductors.

### Changes

#### 1. Allen-Dynes Replaces McMillan for BCS (`BCSTheoryLoss`)

Replaced `mcmillan_tc()` with `allen_dynes_tc()`:

```
Tc = (omega_log / 1.2) * exp(-1.04(1+lambda)/(lambda - mu*(1+0.62*lambda))) * f1 * f2
```

Where:
- `omega_log = theta_D * 0.827` (approximate for monatomic/simple polyatomic solids)
- `f1 = [1 + (lambda/Lambda1)^(3/2)]^(1/3)`, Lambda1 = 2.46(1 + 3.8*mu*)
- `f2 = 1 + (lambda^2 * (0.5 - mu*)) / (lambda^2 + Lambda2^2)`, Lambda2 = 1.82(1 + 6.3*mu*)
- Reference: Allen & Dynes, Phys. Rev. B 12, 905 (1975)

The f1, f2 factors approach 1.0 for weak coupling (lambda < 0.3), recovering McMillan. For strong coupling (lambda > 0.5), they provide significant corrections — critical for MgB2-type materials.

#### 2. Lindemann Debye Temperature Anchor (`BCSTheoryLoss`)

Physics-informed regularizer on the neural net theta_D predictor:

```
theta_D_est = 41.63 * sqrt(T_m / (M * V^(2/3)))
```

Where T_m = melting temp, M = mean atomic weight, V = mean volume per atom (denormalized from Magpie features).

- Weight: 0.1 (within the already-small 0.05 theory weight)
- Uses Huber loss with delta=100K (very forgiving)
- Requires magpie_mean/magpie_std in TheoryLossConfig (passed from norm_stats)
- Denormalized values are approximate for quantile-transformed features — acceptable for a soft anchor

#### 3. Matthias VEC Prior (`BCSTheoryLoss`)

Tc should be higher near VEC=4.7 and 6.7 (Matthias rule for transition metals):

```
envelope = 40 * (exp(-0.5*((VEC-4.7)/1.0)^2) + exp(-0.5*((VEC-6.7)/1.0)^2))
penalty = softplus(tc - envelope - 5.0, beta=0.5)^2 * 0.01
```

- Very gentle (0.01 weight) — a prior, not a constraint
- Materials at unusual VEC (like MgB2 at VEC=3.5) will survive this push-back

#### 4. HeavyFermionTheoryLoss (New Class)

Replaces `UnknownTheoryLoss` for heavy fermion materials:

- **Log-normal prior**: Huber loss in log-space, centered at 1K (log(1)=0), delta=2.0
  - ~7x tolerance around 1K center before strong penalty
  - PuCoGa5 at 18K gets moderate prior penalty but stays below cap
- **Soft cap at 20K**: `softplus(tc - 20, beta=0.5)^2`
- No learnable parameters (pure physics prior)

#### 5. OrganicTheoryLoss (New Class)

Replaces `UnknownTheoryLoss` for organic superconductors:

- **Soft cap at 15K**: Covers most BEDT-TTF salts (Tc < 13K)
- `softplus(tc - 15, beta=0.5)^2`
- No learnable parameters (pure physics prior)

#### 6. IronBasedTheoryLoss VEC Constraint

Added VEC constraint centered at 6.0 (Fe2+ in tetrahedral coordination):

```
vec_deviation = |VEC - 6.0|
vec_penalty = softplus(vec_deviation - 1.0, beta=1.0)^2 * 0.1
```

- Flat within +/-1 of VEC=6.0 (no penalty for VEC 5.0-7.0)
- Gentle (0.1 weight within theory weight)

#### 7. TheoryRegularizationLoss Wiring

- `HEAVY_FERMION` -> `HeavyFermionTheoryLoss` (was `UnknownTheoryLoss`)
- `ORGANIC` -> `OrganicTheoryLoss` (was `UnknownTheoryLoss`)
- Added `heavy_fermion_mask` and `organic_mask` batch processing
- Return dict includes `heavy_fermion_loss`, `organic_loss`, `heavy_fermion_count`, `organic_count`

#### 8. TheoryLossConfig New Fields

```python
magpie_mean: Optional[List[float]]     # [145] mean per feature (from norm_stats)
magpie_std: Optional[List[float]]      # [145] std per feature
idx_mean_atomic_weight: int = 15       # Magpie feature index
idx_mean_melting_t: int = 21
idx_mean_nvalence: int = 75            # VEC proxy
idx_mean_gsvolume: int = 111
heavy_fermion_tc_center: float = 1.0   # Log-normal center (K)
heavy_fermion_tc_max: float = 20.0     # Soft cap (K)
organic_tc_max: float = 15.0           # BEDT-TTF cap (K)
organic_fullerene_tc_max: float = 45.0 # Fullerene cap (K)
```

#### 9. train_v12_clean.py

- Passes `magpie_mean` and `magpie_std` from `norm_stats` to `TheoryLossConfig`
- Updated print summary to mention V12.25

### Design Decisions

1. **Allen-Dynes over McMillan**: ~10 lines more math, significantly more accurate for strong-coupling. The omega_log ~ 0.827 * theta_D approximation is standard.
2. **Lindemann as anchor, not replacement**: Neural net theta_D predictor stays — Lindemann just pulls it toward physical values.
3. **Matthias VEC very gentle (0.01)**: Known correlation, not a hard rule.
4. **Heavy fermion log-normal**: Huber in log-space with delta=2.0 gives ~7x tolerance.
5. **No new learnable parameters for HF/Organic**: Pure physics priors — checkpoint backward-compatible.
6. **All constraints are soft regularizers**: Theory weight is 0.05 (V12.24). These push back on predictions; if predictions survive, they're discovery candidates.

### Backward Compatibility

- All new TheoryLossConfig fields have defaults reproducing old behavior
- Existing checkpoints resume cleanly (no new learnable params for HF/Organic)
- `magpie_mean=None` / `magpie_std=None` disables Lindemann, Matthias, and iron VEC (graceful fallback)

### Files Changed

| File | Change |
|------|--------|
| `src/superconductor/losses/theory_losses.py` | Allen-Dynes, Lindemann, Matthias VEC, HeavyFermionTheoryLoss, OrganicTheoryLoss, iron VEC, config fields |
| `scripts/train_v12_clean.py` | Pass magpie norm stats to TheoryLossConfig, updated print summary |
| `docs/TRAINING_RECORDS.md` | This entry |

### Expected Behavior

- Theory loss should increase initially (more families now contribute non-zero loss)
- BCS loss slightly different order of magnitude (Allen-Dynes vs McMillan)
- Heavy fermion and organic counts should appear in theory loss breakdown (previously counted as "unknown")
- Tc MAE should not regress (theory weight is only 0.05)
- `n_theory_params` unchanged (no new learnable params for HF/Organic)

---

## V12.24: Relative Tc Error + Loss Budget Rebalance (2026-02-10)

### Problem

After V12.23: Kelvin errors improving (100K+ dropped from 79K to 38.4K), but two issues remain:

1. **Low-Tc relative errors are huge**: 1.6K error at Tc=2K is 80% relative error — terrible for discovery, but the loss function doesn't notice because `|3.6 - 2.0| = 1.6K` is small in absolute terms.

2. **Gradient budget is badly allocated**:
   - Contrastive loss (SupCon): raw=5.0 × w=0.1 = **0.50** (~24% of total gradient) — but SupCon's theoretical minimum is ~log(batch_size) ≈ 5.5, so it's near-converged and just adding noise
   - Theory loss: raw=0.20 × w=0.5 = **0.10** (~5%) — mathematically redundant with Tc loss during training (both penalize wrong Tc predictions, but Tc loss has ground truth; theory loss double-counts the same error with different math)
   - Tc loss: raw=0.08 × w=10 = **0.83** (~40%) — the primary objective gets less than half the gradient

### Changes

#### 1. Relative Error Loss Component (`CombinedLossWithREINFORCE.forward()`)

Blends Huber loss (good for absolute error) with relative error in Kelvin space:

```python
relative_err = |pred_K - true_K| / max(true_K, 1.0)
tc_loss_per_sample = (1 - alpha) * huber + alpha * relative_err
```

- At Tc=2K with 1.6K error: relative = 0.80 (vs Huber ~0.08) — **10x stronger signal**
- At Tc=100K with 38K error: relative = 0.38 — still strong
- `tc_relative_weight=0.5` → 50/50 blend of Huber and relative
- Both the asymmetric penalty (V12.23) and Kelvin weighting apply on top

#### 2. Contrastive Weight: 0.1 → 0.01

SupCon loss is a latent geometry shaping objective ("cherry on top"), not a workhorse. At w=0.1 it consumed ~24% of gradient. At w=0.01 it's ~3% — enough to maintain SC/non-SC separation without dominating gradient updates.

#### 3. Theory Weight: 0.5 → 0.05

Theory loss is mathematically redundant with Tc loss during training: if a BCS material has true Tc=15K and the model predicts 40K, BOTH the Tc Huber loss AND the BCS theory penalty fire. The Tc loss already has the correct answer — the theory penalty just double-counts the error with weaker math (it uses the McMillan formula approximation rather than ground truth).

Theory loss remains valuable for:
- Keeping the learnable BCS/cuprate predictors alive (Debye temp, doping) for later use during generation
- Mild physics-grounding on edge cases where family classification is uncertain

At w=0.05 it provides ~0.01 gradient contribution — maintenance level.

#### 4. Config

```python
'tc_relative_weight': 0.5,         # 50% relative + 50% Huber blend
'contrastive_weight': 0.01,        # Down from 0.1 (geometry shaping, not primary)
'theory_weight': 0.05,             # Down from 0.5 (redundant with Tc loss during training)
```

### New Loss Budget (Estimated)

| Component | Before V12.24 | After V12.24 | Change |
|-----------|--------------|-------------|--------|
| Tc | 0.83 (40%) | ~1.2 (70%) | **Primary objective now dominant** |
| Contrastive | 0.50 (24%) | 0.05 (3%) | Maintenance level |
| Magpie | 0.16 (8%) | 0.16 (9%) | Unchanged |
| Theory | 0.10 (5%) | 0.01 (1%) | Maintenance level |
| Formula | 0.10 (5%) | 0.10 (6%) | Unchanged |
| Other | 0.03 (1%) | 0.03 (2%) | Unchanged |

### Expected Behavior

- Tc loss value will increase initially (relative error component is larger in magnitude than Huber)
- Low-Tc relative errors (0-10K range) should improve — 80% error at 2K now gets strong signal
- High-Tc absolute errors should continue improving (V12.23 Kelvin weighting still active)
- Formula accuracy should not degrade (formula loss unchanged, more gradient budget available)
- Contrastive loss raw value may drift upward slightly (less gradient suppressing it) — this is fine; the geometry is already well-shaped after 2000+ epochs

---

## V12.23: Tc-Weighted Asymmetric Regression Loss (2026-02-10)

### Problem

~79K average Kelvin error for samples with Tc > 100K. Two root causes:

1. **Log1p compression kills high-Tc gradients**: A 10K error at Tc=100K produces only 0.09 loss units in log1p space vs 0.64 at Tc=10K — 7x less gradient signal for the materials we care most about.
2. **Symmetric loss doesn't match discovery objective**: Underpredicting a cuprate at 130K as 30K is worse than overpredicting — false positives are verifiable, missed high-Tc materials are not.

### Changes

#### 1. Kelvin Weighting (`CombinedLossWithREINFORCE.forward()`)

- Per-sample Tc loss computed with `reduction='none'`
- True Tc denormalized back to Kelvin (handles both log1p and linear normalization)
- Weight: `1 + tc_kelvin / scale` where `scale=50` by default
- At 5K: weight=1.1x (minimal impact on low-Tc accuracy)
- At 100K: weight=3x (counteracts log1p compression)
- At 250K: weight=6x (strong signal for highest-Tc materials)

#### 2. Asymmetric Penalty

- Underprediction penalized 1.5x vs overprediction (flat multiplier)
- Combined with Kelvin weighting: cuprate at 100K underpredicted → `3.0 × 1.5 = 4.5x` loss vs correctly-predicted BCS at 5K

#### 3. Config

```python
'tc_kelvin_weighting': True,       # Weight Tc loss by true Tc in Kelvin
'tc_kelvin_weight_scale': 50.0,    # weight = 1 + tc_K / scale
'tc_underpred_penalty': 1.5,       # Asymmetric: underprediction costs 1.5x
```

All defaults are backward-compatible (weighting=False, penalty=1.0).

### Expected Behavior

- Tc loss value may increase initially (weighted mean > unweighted)
- After ~50-100 epochs: Kelvin error for Tc > 100K should drop substantially from 79K
- Low-Tc accuracy should not degrade (only 1.1x weight at 5K)
- Model should develop slight positive bias on Tc predictions (intentional discovery preference)

---

## V12.22: Theory-Guided Consistency Losses (2026-02-10)

### Overview

Enables physics-based theory losses (BCS McMillan formula, cuprate Presland dome, iron-based constraints) that were fully implemented in `theory_losses.py` and `family_classifier.py` but completely disconnected from training (`use_theory_loss: False`, `theory_loss_fn` never passed to `train_epoch()`).

**Philosophy**: No hard Tc caps. If the model predicts something "impossible," we see it. Instead, massively increase the error signal via soft quadratic penalties. Let the network learn Debye temperature and electron-phonon coupling predictors from Magpie features.

### Changes

#### 1. Theory Loss Physics Fixes (`theory_losses.py`)

- **Fixed Tc denormalization for log-transform** (V12.20 broke theory losses silently):
  - Old: `tc = predicted_tc * tc_std + tc_mean` (wrong when `tc_log_transform=True`)
  - New: `tc = expm1(predicted_tc * tc_std + tc_mean)` via shared `_denormalize_tc()` helper
  - Clamp to [0, 500K] as numerical guard (not a physics cap)

- **Replaced hard caps with soft quadratic penalties**:
  - BCS: `F.relu(tc - 40)` → `F.softplus(tc - 40, beta=0.5) ** 2`
  - Iron-based: `F.relu(tc - 60)` → `F.softplus(tc - 60, beta=0.5) ** 2`
  - BCS material at 100K: penalty ~3600 (vs old 60). Signal is massive but model is free to predict it.

- **Added learnable cuprate doping/Tc_max predictors** (~22,850 new params):
  - `doping_predictor`: Magpie(145) → 64 → 32 → 1 → Sigmoid, scaled to [0.05, 0.27]
  - `tc_max_predictor`: Magpie(145) → 64 → 32 → 1 → Softplus, scaled to [30, 165K]
  - Replaces constant stub `extract_doping_level()` (always returned 0.15)
  - Presland dome constrains: Tc = Tc_max * [1 - 82.6(p - 0.16)^2]

- **Added `tc_log_transform` to `TheoryLossConfig`**

#### 2. Pre-computed Family Labels (`train_v12_clean.py`)

- Each formula classified during data loading using `RuleBasedFamilyClassifier.classify_from_elements()`
- SC samples → one of 14 `SuperconductorFamily` values; non-SC → `NOT_SUPERCONDUCTOR (0)`
- Stored as 10th tensor in `TensorDataset` (backward compatible — defaults to zeros if missing in cache)
- Family distribution printed during loading

#### 3. Training Loop Integration (`train_v12_clean.py`)

- `train_epoch()` accepts `theory_loss_fn`, `theory_weight`, `norm_stats` params
- Theory loss computed on SC samples only, added to loss in all three branches (pure SC, pure non-SC, mixed)
- Theory loss parameters (BCS predictors ~22K + cuprate predictors ~22K = ~45K total) added to encoder optimizer
- Gradient clipping includes theory loss parameters
- **Theory warmup**: `theory_warmup_epochs: 50` — ramps weight from 0 → `theory_weight` over 50 epochs (like contrastive warmup)

#### 4. Checkpoint Compatibility

- `theory_loss_fn_state_dict` saved in checkpoint (BCS/cuprate predictor weights)
- Restored on load if key exists; prints message if missing (old checkpoint)
- Encoder optimizer state won't match with old checkpoint → uses fresh optimizer (acceptable, existing handler)

#### 5. Config

```python
'use_theory_loss': True,
'theory_weight': 0.5,              # Strong signal (was 0.05)
'theory_warmup_epochs': 50,        # Ramp up over 50 epochs
'theory_use_soft_constraints': True,
```

#### 6. Metrics

- `theory_loss` added to epoch metrics, CSV logging, and epoch print line
- Format: `Thry: 0.1234 (w=0.250)` showing current effective weight

### Expected Behavior

- Theory loss appears in epoch output after warmup begins
- Per-family breakdown: ~40% BCS, ~30% cuprate, ~10% iron, ~20% unknown
- BCS predictors (Debye temp, lambda) converge to physically plausible ranges after ~100 epochs:
  - Debye temp: 100-500K
  - Lambda (electron-phonon coupling): 0.2-1.0
- Cuprate doping predictor should not collapse to constant 0.16

### Backward Compatibility

- `use_theory_loss: False` recovers exact V12.21 behavior
- Old checkpoints without `theory_loss_fn_state_dict` → theory predictors start fresh
- Old caches without `family_tensor` → defaults to zeros (triggers recomputation on next fresh preprocessing)

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
