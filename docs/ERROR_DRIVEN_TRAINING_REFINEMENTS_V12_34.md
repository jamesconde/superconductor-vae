# Error-Driven Training Refinements (V12.34)

**Date**: 2026-02-26
**Affects**: `scripts/train_v12_clean.py`, `src/superconductor/models/autoregressive_decoder.py`, `scratch/analyze_error_reports.py`

---

## 1. Motivation: The 60-65% Exact Match Plateau

Error analysis of epochs 2764-2812 revealed the model was stuck in a noisy plateau at 60-65% exact match. The autoregressive decoder is the bottleneck, not the latent representation (Magpie MSE is actually *lower* for error samples than exact matches, confirming the encoder is doing its job).

Five specific error drivers were identified through systematic analysis:

| Driver | Evidence | Correlation |
|--------|----------|-------------|
| Sequence length | 50% of errors in final third of sequences | r=0.342 (STRONG) |
| Z-norm extremes | Q1: 79% exact vs Q4: 43% exact (36pt gap) | r=0.236 (MODERATE) |
| Element count | 4+ elements: 9-14 avg errors vs 3-4 for 2-3 elements | implicit in length |
| Fraction representation | Denominator drift, bias toward "common" denoms | qualitative |
| Error cascade | Late-position errors compound | implicit in length |

Additionally, the Tc range analysis in the error reports contained a **bug** that made all Tc range data meaningless.

---

## 2. The Tc Range Analysis Bug (Fix F)

### What Was Wrong

The evaluation function `evaluate_true_autoregressive()` collected Tc values in `all_tc_true_np` and `all_tc_pred_np`, then bucketed them against Kelvin thresholds:

```python
tc_ranges = [(0, 10, '0-10K'), (10, 30, '10-30K'), (30, 77, '30-77K'), ...]
tc_mask = (all_tc_true_np >= lo) & (all_tc_true_np < hi)
```

**The problem**: `all_tc_true_np` contained **normalized** values (log1p + z-score normalization), not Kelvin. Normalized Tc values are typically in the range -2 to +3, so the comparison `>= 10` was almost never true. **Every single sample fell into the "0-10K" bucket**, making the entire Tc range breakdown useless.

This bug was present since the Tc range analysis was first added in V12.18.

### The Fix

1. Added `norm_stats` parameter to `evaluate_true_autoregressive()` so it has access to `tc_mean`, `tc_std`, and `tc_log_transform`.

2. Before bucketing, denormalize both true and predicted Tc values to Kelvin:
```python
tc_denorm = all_tc_true_np * tc_std + tc_mean
if tc_log_transform:
    tc_true_kelvin = np.expm1(tc_denorm)
tc_true_kelvin = np.clip(tc_true_kelvin, 0, None)
```

3. Added per-range R², MAE (Kelvin), and max error to each Tc bucket.

4. Added overall Kelvin-space metrics: `tc_r2_kelvin` and `tc_mae_kelvin_overall`.

### Verification

Test with simulated normalized values confirmed the bug:
- **Old (buggy)**: 7/7 samples in "0-10K", all other buckets empty
- **New (fixed)**: Correct distribution: 1 in 10-30K, 1 in 30-77K, 1 in 77-120K, 1 in 120-200K, 3 in >200K

### Impact on Historical Data

All `errors_by_tc_range` data in error reports prior to V12.34 is **invalid** (everything was in "0-10K"). This does not affect any training behavior — it was purely a diagnostic reporting issue. The overall `tc_r2` (on normalized values) and `tc_mae_overall` (also normalized) were always correct and unaffected.

---

## 3. Refinement A: Sequence-Length Weighted Formula Loss

### Rationale

Error analysis shows a clear correlation between sequence length and errors (r=0.342, the strongest predictor). 50% of token-level errors occur in the final third of sequences. The standard loss treats a 10-token formula and a 45-token formula equally, but the 45-token formula requires far more accurate sequential prediction.

### Implementation

In `CombinedLossWithREINFORCE.forward()`, the formula CE/focal loss is now computed per-sample instead of as a single scalar. Each sample's loss is then weighted:

```
weight = 1 + alpha * max(0, (seq_length - base) / base)
```

Where `base=15` (sequences at or below this length get weight 1.0) and `alpha=1.0`.

**Examples**:
- 10-token formula: weight = 1.0 (no change)
- 20-token formula: weight = 1.33
- 30-token formula: weight = 2.0
- 45-token formula: weight = 3.0

### Technical Detail: FocalLoss Per-Sample Reduction

To enable per-sample weighting, `FocalLossWithLabelSmoothing.forward()` gained a `reduction` parameter:
- `reduction='mean'` (default): Returns scalar loss, identical to pre-V12.34 behavior
- `reduction='per_sample'`: Returns `[batch]` tensor of per-sample losses

The per-sample path scatters token-level focal losses back to `[batch, seq_len]` shape, then averages per sample over valid (non-PAD) positions. This was verified to produce identical results to the scalar path when no weighting is applied.

### Config

```python
'use_length_weighting': True,     # Toggle on/off
'length_weight_base': 15,         # Sequences <= this get weight 1.0
'length_weight_alpha': 1.0,       # Scale factor
```

---

## 4. Refinement B: Fraction Canonicalization

### Rationale

The model sometimes produces correct stoichiometry in a different fraction representation than the target. For example, the dataset might contain `Ba(6/10)` while the model generates `Ba(3/5)` — these are mathematically identical but differ as token sequences. The model is penalized for "errors" that are actually correct predictions.

This also creates training ambiguity: the model sees multiple equivalent representations for the same value (e.g., `6/10`, `3/5`, `60/100`), splitting the probability mass across representations instead of concentrating it.

### Implementation

A new `canonicalize_fractions()` function GCD-reduces all fractions in a formula string before tokenization:

```python
def canonicalize_fractions(formula: str) -> str:
    """Reduce all fractions to lowest terms via GCD.
    'Ba(6/10)Sr(4/10)CuO3' -> 'Ba(3/5)Sr(2/5)CuO3'
    """
    def reduce_match(m):
        num, den = int(m.group(1)), int(m.group(2))
        g = math.gcd(num, den)
        return f"{num // g}/{den // g}"
    return re.sub(r'(\d+)/(\d+)', reduce_match, formula)
```

This is applied in the data loading loop, before tokenization. Fractions already in lowest terms are unchanged (GCD=1).

### Cache Invalidation

The `use_canonical_fractions` setting is tracked in the preprocessed tensor cache metadata. When the setting changes (e.g., from `False` to `True`), the cache is automatically invalidated and rebuilt. The rebuild takes approximately 2 minutes.

### Impact

- **Shorter sequences for some formulas**: e.g., `6/10` (4 tokens) becomes `3/5` (3 tokens), saving 1 token per canonicalized fraction
- **Eliminates representation ambiguity**: The model only sees one canonical form per stoichiometric value
- **Reduces false-negative errors**: Predictions that produce a simplified or expanded equivalent fraction are no longer penalized

Tested with 6 cases:
- `Ba(6/10)Sr(4/10)CuO3` -> `Ba(3/5)Sr(2/5)CuO3` (reduced)
- `La(70/100)Ce(30/100)` -> `La(7/10)Ce(3/10)` (reduced)
- `Ag(1/500)Al(499/500)` -> unchanged (already minimal, GCD=1)
- `YBa2Cu3O(137/20)` -> unchanged (GCD=1)
- `La(193/1000)Sr(807/1000)CuO4` -> unchanged (GCD=1)
- `Ti(4/8)O2` -> `Ti(1/2)O2` (reduced)

### Config

```python
'use_canonical_fractions': True,  # Toggle on/off (triggers cache rebuild on change)
```

---

## 5. Refinement C: Z-Norm Soft Penalty

### Rationale

Z-norm quartile analysis reveals a dramatic gap in decoder performance:
- **Q1 (lowest z_norm)**: 79% exact match
- **Q4 (highest z_norm)**: 43% exact match

This 36-percentage-point gap means the decoder struggles with latent vectors that have large norms. The existing L2/KL regularization (`kl_weight=0.0001`) penalizes all z-dimensions uniformly, which helps but doesn't specifically target the norm outliers that cause the worst decoding failures.

### Implementation

A soft quadratic barrier on z-norms above a target threshold:

```python
z_norms = z.norm(dim=1)                          # [batch]
excess = torch.clamp(z_norms - z_norm_target, min=0.0)
z_norm_penalty = (excess ** 2).mean()
total_loss += z_norm_penalty_weight * z_norm_penalty
```

**Key properties**:
- **No effect on z_norms below 22.0** (the current mean): The `clamp(min=0.0)` ensures only outliers are penalized
- **Gentle weight (0.001)**: Deliberately small to avoid collapsing the latent space. The penalty only fires for ~25-50% of samples (Q3-Q4 range)
- **Complementary to existing KL/L2**: KL regularizes all dimensions uniformly; this targets the aggregate norm specifically
- **Quadratic**: Stronger penalty for more extreme outliers (a z_norm of 30 is penalized 16x more than 26)

### Mathematical Note

With current z_norm distribution (mean ~22, std ~3):
- z_norm = 22: penalty = 0 (below target)
- z_norm = 25: penalty = 0.001 * 9 = 0.009
- z_norm = 30: penalty = 0.001 * 64 = 0.064
- z_norm = 35: penalty = 0.001 * 169 = 0.169

The penalty is logged as `z_norm_penalty` in the loss result dict for monitoring.

### Config

```python
'use_z_norm_penalty': True,
'z_norm_target': 22.0,            # Target z_norm (current population mean)
'z_norm_penalty_weight': 0.001,   # Gentle penalty weight
```

---

## 6. Refinement D: Element-Count Weighted Formula Loss

### Rationale

Formulas with more elements have dramatically higher error rates:
- 2-3 elements: 3-4 average errors per failed sample
- 4+ elements: 9-14 average errors per failed sample

This makes sense: more elements mean longer sequences (more fractions, more element symbols), more diverse vocabulary per sample, and more opportunities for the autoregressive decoder to diverge.

High-element-count formulas include most superconducting cuprates (e.g., `Bi(11/50)Sr(2/5)Ca(1/5)Cu(3/10)O(18/25)` = 5 elements, 37 tokens), which are the most scientifically valuable samples.

### Implementation

Combined multiplicatively with Refinement A in the per-sample formula loss:

```python
sample_weights = torch.ones(batch_size)

# A: Length weighting
sample_weights *= 1 + alpha * max(0, (seq_len - base_len) / base_len)

# D: Element count weighting
sample_weights *= 1 + beta * max(0, n_elements - base_count)

formula_ce_loss = (per_sample_loss * sample_weights).mean()
```

**Examples** (with alpha=1.0, beta=0.5, base_len=15, base_count=3):
- 2-element, 8-token formula: weight = 1.0 * 1.0 = 1.0
- 3-element, 20-token formula: weight = 1.33 * 1.0 = 1.33
- 5-element, 35-token formula: weight = 2.33 * 2.0 = 4.67
- 7-element, 45-token formula: weight = 3.0 * 3.0 = 9.0

The `n_elements` is computed from `elem_mask.bool().sum(dim=1)` and passed to the loss function at all 4 call sites (pure SC, pure non-SC, mixed SC portion, mixed non-SC portion).

### Config

```python
'use_element_count_weighting': True,
'element_count_base': 3,          # Formulas with <= 3 elements get weight 1.0
'element_count_beta': 0.5,        # Scale factor
```

---

## 7. Refinement E: Position-Dependent Teacher Forcing (Infrastructure)

### Rationale

When teacher forcing ratio (TF) is below 1.0, the model sometimes uses its own predictions as input for subsequent positions. Errors in early positions cascade to later positions. The idea is to provide more teacher forcing at the start of sequences (to establish correct element ordering) and less at the end (to force the model to practice self-recovery where 50% of errors occur).

### Implementation

In `EnhancedTransformerDecoder.forward()`, the `use_gt_mask` creation for scheduled sampling was modified:

```python
# Original: uniform TF probability at all positions
use_gt_mask = (torch.rand(B, L) < teacher_forcing_ratio)

# New: position-dependent TF probability
positions = torch.arange(L).float() / max(L - 1, 1)
tf_per_position = base_tf * (1 + gamma * (1 - positions))
use_gt_mask = (torch.rand(B, L) < tf_per_position.unsqueeze(0))
```

With gamma=0.5 and base TF=0.36:
- Position 0 (start): TF = 0.36 * 1.5 = 0.54
- Position L (end): TF = 0.36 * 1.0 = 0.36

### Current Status: NO EFFECT

**Teacher forcing is currently always 1.0** (full teacher forcing). The position-dependent logic only activates when `TF < 1.0`, so this refinement has **zero effect on current training**. It is infrastructure, ready for when TF is reduced in the future.

The constructor parameters (`use_position_dependent_tf`, `tf_position_decay`) are passed from `TRAIN_CONFIG` to `EnhancedTransformerDecoder`, and the logic is tested and verified. When TF eventually drops (e.g., via scheduled sampling or curriculum learning), this will automatically activate.

### Config

```python
'use_position_dependent_tf': True,  # Toggle (no effect at TF=1.0)
'tf_position_decay': 0.5,          # Gamma for position decay
```

---

## 8. Combined A+D Weighting: Mathematical Interaction

Refinements A (length) and D (element count) are applied **multiplicatively** to the same per-sample formula loss. This creates a natural interaction:

| Formula Type | Seq Len | Elements | A Weight | D Weight | Combined |
|-------------|---------|----------|----------|----------|----------|
| Simple binary (NaCl) | 8 | 2 | 1.0 | 1.0 | **1.0x** |
| Simple ternary (BaTiO3) | 12 | 3 | 1.0 | 1.0 | **1.0x** |
| YBCO (YBa2Cu3O7) | 18 | 4 | 1.2 | 1.5 | **1.8x** |
| Doped cuprate (5 elem) | 30 | 5 | 2.0 | 2.0 | **4.0x** |
| Complex cuprate (7 elem) | 45 | 7 | 3.0 | 3.0 | **9.0x** |

This is the desired behavior: the scientifically most valuable (and hardest) superconductor formulas receive the strongest gradient signal.

### REINFORCE Interaction

REINFORCE is currently disabled (`rl_weight` effectively 0 at runtime). The A+D weighting applies only to the CE/focal loss component. If REINFORCE is re-enabled, it would operate on a separate term — the per-sample weighting and REINFORCE address complementary aspects (CE focuses on token-level accuracy, REINFORCE on sequence-level semantic quality).

---

## 9. Error Analysis Script Updates

`scratch/analyze_error_reports.py` was updated to handle the new V12.34 error report format:

1. **Trends table**: Added `Tc_R2_K` column showing Kelvin-space R² (falls back to "N/A" for pre-V12.34 reports)

2. **New `analyze_tc_ranges()` function**: Displays per-Tc-range R², MAE (Kelvin), and max error. Handles both old-format (basic metrics only) and new-format (full Kelvin metrics) reports gracefully. Tracks Kelvin R² across epochs when available.

3. **Updated summary**: Reports both normalized and Kelvin R², plus Kelvin MAE.

The script remains backward-compatible with pre-V12.34 error reports.

---

## 10. Complete File Change Summary

### `scripts/train_v12_clean.py`

| Section | Change |
|---------|--------|
| `TRAIN_CONFIG` | Added 12 new keys for refinements A-E |
| `canonicalize_fractions()` | New function: GCD-reduce fractions in formula strings |
| `_try_load_cache()` | Added `use_canonical_fractions` cache invalidation check |
| `_save_cache()` | Added `use_canonical_fractions` to cache metadata |
| Data loading loop | Integrated canonicalization before tokenization |
| `FocalLossWithLabelSmoothing.forward()` | Added `reduction='per_sample'` mode returning `[batch]` losses |
| `CombinedLossWithREINFORCE.__init__()` | Added 6 params for A+D weighting config |
| `CombinedLossWithREINFORCE.forward()` | Per-sample weighting (A+D), z-norm penalty (C), `n_elements` param |
| Loss function instantiation | Passes A+D config from TRAIN_CONFIG |
| All 4 `loss_fn()` call sites | Added `n_elements=elem_mask.bool().sum(dim=1)` |
| `evaluate_true_autoregressive()` | Added `norm_stats` param, Tc denormalization, per-range R²/MAE |
| Eval call site | Passes `norm_stats` |
| Decoder construction | Passes `use_position_dependent_tf` and `tf_position_decay` |

### `src/superconductor/models/autoregressive_decoder.py`

| Section | Change |
|---------|--------|
| `EnhancedTransformerDecoder.__init__()` | Added `use_position_dependent_tf` and `tf_position_decay` params |
| `EnhancedTransformerDecoder.forward()` | Position-dependent `use_gt_mask` logic in scheduled sampling path |

### `scratch/analyze_error_reports.py`

| Section | Change |
|---------|--------|
| `analyze_trends()` | Added Kelvin R² column to header and row output |
| `analyze_tc_ranges()` | New function: per-range R², MAE, max error in Kelvin space |
| `main()` | Added `analyze_tc_ranges()` call |
| Summary | Reports Kelvin R² and MAE when available |

### `docs/TRAINING_RECORDS.md`

| Section | Change |
|---------|--------|
| Top of file | V12.34 entry with full description of all changes |

---

## 11. Overhead Assessment

| Refinement | Train Overhead | Eval Overhead | Memory |
|------------|---------------|---------------|--------|
| A: Length weighting | <0.1% (per-sample mean) | None | None |
| B: Fraction canon | One-time cache rebuild (~2 min) | None | None |
| C: Z-norm penalty | <0.1% (one norm/batch) | None | None |
| D: Element count weight | <0.1% (per-sample multiply) | None | None |
| E: Position-dependent TF | 0% (TF=1.0 bypass) | None | None |
| F: Tc range fix | None | <0.1% (denorm math) | None |
| **Total** | **<0.5% of epoch time** | **Negligible** | **None** |

---

## 12. How to Disable Individual Features

All features are independently togglable via TRAIN_CONFIG. To disable any single feature without affecting others:

```python
TRAIN_CONFIG = {
    ...
    'use_length_weighting': False,          # Disable A
    'use_canonical_fractions': False,        # Disable B (triggers cache rebuild)
    'use_z_norm_penalty': False,             # Disable C
    'use_element_count_weighting': False,    # Disable D
    'use_position_dependent_tf': False,      # Disable E (already no-op at TF=1.0)
    ...
}
```

The Tc range fix (F) has no toggle — it is always active when `norm_stats` is available (which it always is in normal training). This is a pure bug fix, not a behavioral change.

---

## 13. Verification Summary

All changes were verified through:

1. **Syntax check**: All 3 modified Python files pass `ast.parse()`
2. **Decoder test**: Constructor accepts new params, forward pass works at TF=0.5 and TF=1.0
3. **Fraction canonicalization**: 6 test cases covering reduction, identity, and edge cases
4. **FocalLoss per-sample**: `reduction='per_sample'` produces same mean as `reduction='mean'` when weights are uniform
5. **A+D weighting**: Sample weights computed correctly for various seq_len and n_element combinations
6. **Z-norm penalty**: Correctly zero below target, quadratic above
7. **Tc denormalization**: Verified bug reproduction (all in 0-10K) and fix (correct distribution)
8. **Config keys**: All 12 new TRAIN_CONFIG keys verified present via AST inspection
9. **Call sites**: All 4 `loss_fn()` calls have `n_elements` correctly wired (including SC/non-SC masked variants)
