# V12.41 Backward Compatibility Mode

**Created**: 2026-02-24
**Status**: Active
**Base Model**: V12.41 (epoch 3292, best_exact=85.4%, trained on A100)
**Applies To**: V12.41 checkpoint resumption with V14.3+ codebase

---

## 1. Problem Statement

The V12.41 checkpoint (epoch 3292, 85.4% exact match on A100) is the best-performing model for
autoregressive formula generation. Since V12.41, several incompatible architecture changes were made:

| Version | Change | Impact |
|---------|--------|--------|
| V13.0 | Semantic fraction tokenization | Vocab 148 → 4355, numden_head removed, stoich 37→13 dims |
| V14.0 | Isotope token expansion | Vocab 4355 → 4647, new token type ranges |
| V14.3 | Token type classifier head + heads_to_memory | New heads (compatible — random init) |
| V15.0 | Memory bottleneck layer | Replaced 2-layer MLP with bottleneck (incompatible dims) |

The codebase had hardcoded V13+ assumptions (4647-token vocab, 13-dim stoich, no numden_head),
making it impossible to load or resume training from V12.41.

---

## 2. Solution: Config-Driven Architecture

The codebase was made config-driven so features are conditionally enabled/disabled based on
configuration flags. This allows V12.41 to load and train with worthwhile newer features while
disabling incompatible ones.

### Features Classification

| Feature | Version | Action | Reason |
|---------|---------|--------|--------|
| Token type classifier head | V14.3 | **KEEP** | New head, random init, learns from scratch |
| Enriched decoder memory (heads_to_memory) | V14.3 | **KEEP** | New head, random init, vocab-agnostic |
| Hierarchical family head | V15.0 | **KEEP** | Already in V12.41 checkpoint |
| RL auto-scaling | V14.1 | **KEEP** | Config-only, no architecture change |
| V14 continuous reward | V14.0 | **KEEP** | Needs token range overrides (see Section 7) |
| Semantic fraction tokenization | V13.0 | **DISABLE** | 148 vs 4647 token vocab, incompatible |
| Isotope tokens | V14.0 | **DISABLE** | Requires V13 semantic fractions |
| Memory bottleneck | V15.0 | **DISABLE** | V12.41 has old-style 2-layer MLP |
| Migration LR boost | V14.1 | **DISABLE** | Not needed — no vocab expansion |

---

## 3. Config Defaults

These config keys in `scripts/train_v12_clean.py` control V12.41 mode:

### MODEL_CONFIG

```python
'memory_bottleneck_dim': 0,  # V15.0 bottleneck disabled; V12.41 uses direct MLP
```

### TRAIN_CONFIG

```python
'use_semantic_fractions': False,  # V13.0 semantic fractions disabled
'use_isotope_tokens': False,      # V14.0 isotope tokens disabled
'max_formula_len': 60,            # Digit-by-digit needs longer sequences than single-token fractions
'numden_weight': 1.0,             # MSE loss weight for numden_head predictions
```

### Switching Between Modes

To return to V13+ mode (e.g., for a fresh V13+ checkpoint):

```python
MODEL_CONFIG['memory_bottleneck_dim'] = 1024
TRAIN_CONFIG['use_semantic_fractions'] = True
TRAIN_CONFIG['use_isotope_tokens'] = True
TRAIN_CONFIG['max_formula_len'] = 30
TRAIN_CONFIG['numden_weight'] = 0.0  # or remove
```

---

## 4. Architecture Changes

### 4.1 NumDen Head Restoration (`attention_vae.py`)

**File**: `src/superconductor/models/attention_vae.py`
**Class**: `FullMaterialsVAE`

Added `use_numden_head: bool = False` parameter to `__init__()`. When `True`, creates the
numden_head neural network matching V12.41 checkpoint weight shapes:

```
numden_head:
  Linear(2048, 512) → LayerNorm(512) → GELU → Dropout(0.1)
  → Linear(512, 256) → LayerNorm(256) → GELU → Dropout(0.1)
  → Linear(256, 24)   # 12 numerators + 12 denominators in log1p space
```

In `forward()`, conditionally computes:
```python
numden_pred = self.numden_head(z) if self.use_numden_head else None
```

The encoder construction in `train_v12_clean.py` sets:
```python
use_numden = not TRAIN_CONFIG.get('use_semantic_fractions', False)
encoder = FullMaterialsVAE(..., use_numden_head=use_numden)
```

### 4.2 V12 Token Type Mapping (`autoregressive_decoder.py`)

**File**: `src/superconductor/models/autoregressive_decoder.py`

Two new functions provide token type classification for the old 148-token vocab:

**`get_v12_type_masks(device='cpu')`**
- Builds `[N_TOKEN_TYPES, VOCAB_SIZE]` boolean masks for hard vocab masking during AR generation
- V12 token layout: Special [0-19], Elements [20-137], Digits [138-147]
- `TOKEN_TYPE_FRACTION` stays all-False (no single-token fractions in V12 vocab)

**`compute_v12_token_type_targets(token_ids)`**
- Maps token IDs to type classes for token type loss computation
- Returns tensor of type IDs: SPECIAL, ELEMENT, INTEGER, or EOS

These are used as fallbacks when `v13_tokenizer is None` in:
- Training loop type loss computation
- RLOO generation type masking
- Eval AR generation type masking

### 4.3 Conditional Stoich Dimensions

V12.41 uses 37-dim stoich conditioning; V13+ uses 13-dim:

```
V12.41: fractions(12) + numden(24) + count(1) = 37 dims
V13+:   fractions(12) + count(1) = 13 dims
```

Four locations in `train_v12_clean.py` assemble stoich tensors conditionally:

1. **Training loop predicted stoich** (~line 4845): Concatenates `fraction_pred + numden_pred + count` when numden is available
2. **Training loop GT stoich** (~line 4850): Uses `batch_tensors[10]` (element_num_log) and `batch_tensors[11]` (element_den_log) for ground truth numden
3. **Eval stoich assembly** (~line 4195): Same conditional pattern
4. **RLOO/generation stoich assembly** (~line 3377): Same conditional pattern

---

## 5. Data Loading Changes

### NumDen Extraction (`train_v12_clean.py`, ~line 1820)

When `use_semantic_fractions=False`, the data loader extracts real numerator/denominator pairs
using `parse_numden_from_formula()` (existing function at line ~1266):

```python
if not use_semantic:
    # V12.41 mode: extract real numden for 37-dim stoich conditioning
    for i, formula in enumerate(formulas):
        numden = parse_numden_from_formula(formula)
        for j, (num, den) in enumerate(numden):
            if j < MAX_ELEMENTS and element_mask[i, j]:
                element_numerators[i, j] = num
                element_denominators[i, j] = den
    element_num_log = torch.log1p(element_numerators)
    element_den_log = torch.log1p(element_denominators)
```

When `use_semantic_fractions=True`, zero tensors are created (numden handled by semantic tokens).

**Cache invalidation**: The data cache checks `use_semantic_fractions` in its key (line ~1413-1414).
Switching from `True` to `False` triggers a full cache rebuild automatically.

---

## 6. NumDen Loss

**File**: `scripts/train_v12_clean.py`, loss computation block (~line 4974)

MSE loss in log1p space over valid element positions:

```python
# V12.41: Numden prediction loss
if numden_pred is not None and not _use_semantic:
    gt_num_log = batch_tensors[10]  # [batch, 12]
    gt_den_log = batch_tensors[11]  # [batch, 12]
    gt_numden = torch.cat([gt_num_log, gt_den_log], dim=-1)  # [batch, 24]
    mask_24 = torch.cat([elem_mask, elem_mask], dim=-1)       # [batch, 24]
    n_valid = elem_mask.sum(dim=1, keepdim=True).clamp(min=1)
    numden_loss_val = ((numden_pred - gt_numden)**2 * mask_24).sum(dim=1) / n_valid.squeeze(-1)
    numden_loss_val = numden_loss_val.mean()
```

Added to all 3 loss branches (pure SC, pure non-SC, mixed batch) with configurable weight:
```python
total_loss += TRAIN_CONFIG.get('numden_weight', 1.0) * numden_loss_val
```

Tracked in metrics as `numden_loss` for monitoring.

---

## 7. V14 Reward Token Range Overrides

**File**: `scripts/train_v12_clean.py`, loss function construction (~line 2261)

The V14 continuous reward system uses token range boundaries to identify element, integer, and
fraction tokens for reward computation. V12 and V13 have different vocab layouts:

| Token Type | V12 Range | V13+ Range |
|-----------|-----------|------------|
| Elements | 20–137 | 5–122 |
| Integers | 138–147 | 123–142 |
| Fractions | (none) | 143+ |

When `use_semantic_fractions=False`, overrides are applied:

```python
reward_kwargs.update({
    'v14_element_start': 20,
    'v14_element_end': 137,
    'v14_integer_start': 138,
    'v14_integer_end': 147,
    'v14_fraction_start': 148,  # Beyond vocab → no fraction tokens
})
```

---

## 8. Checkpoint Loading Behavior

When loading V12.41 with the current codebase, expect:

### Successfully Loaded (existing weights)
- All encoder weights (attention layers, element/tc/magpie projections)
- `numden_head` weights (restored via `use_numden_head=True`)
- `latent_to_memory`, `stoich_to_memory`, `token_embedding` (original shapes)
- `family_head` weights (hierarchical family classification)
- All decoder weights (layers, attention, output projection)
- Fraction head, count head, Tc head, Magpie head

### Missing Keys (expected — new heads, random initialized)
- `token_type_head.*` — new V14.3 head, learns from scratch
- `heads_to_memory.*` — new V14.3 decoder memory enrichment

### Unexpected Keys (expected — removed features, safely ignored)
- `skip_to_memory.*` — V15.0 memory bottleneck, not present in model with `memory_bottleneck_dim=0`

These are handled by `strict=False` in `torch.load()` and logged at startup.

### Checkpoint Saving

Saved checkpoints include `stoich_input_dim=37` (was hardcoded to 13) so future loads
correctly reconstruct the decoder with the right stoich conditioning dimension.

---

## 9. Files Modified

| File | Changes |
|------|---------|
| `src/superconductor/models/attention_vae.py` | Added `use_numden_head` param; conditional numden_head creation; `numden_pred` in forward() |
| `src/superconductor/models/autoregressive_decoder.py` | Added `get_v12_type_masks()`, `compute_v12_token_type_targets()`, V12 vocab constants |
| `scripts/train_v12_clean.py` | Config defaults; encoder construction with `use_numden_head`; conditional numden extraction in data loading; 37-dim stoich assembly in 4 locations; numden loss in 3 branches; type mask/target fallbacks in 3 locations; V14 reward vocab overrides; checkpoint `stoich_input_dim` fix |
| `docs/TRAINING_RECORDS.md` | Added V12.41 backward compatibility section |

---

## 10. Migration Safety

All existing migration paths are naturally guarded and won't fire in V12.41 mode:

- **Inline isotope init** (lines 3805-3873): Only when `_vocab_expanded=True`, requires `v13_tokenizer is not None`
- **`migrate_v12_to_v13.py` import** (line 3821): Only inside vocab expansion block
- **V13 phase training** (line 949): `v13_phase: None`
- **Migration LR boost**: Triggered by `vocab_expanded` from `load_checkpoint`, won't fire for V12.41
- **skip_to_memory unexpected keys**: V12.41 has these but model doesn't; `strict=False` ignores them

---

## 11. Verification Checklist

1. **Checkpoint loading**: Load V12.41, confirm numden_head weights load (no "missing keys" for numden), token_type_head + heads_to_memory appear as "missing keys" (expected), skip_to_memory appears as "unexpected keys" (expected)

2. **Forward pass**: Run 1 epoch, verify 37-dim stoich tensors flow through decoder without shape errors

3. **Token type head**: Confirm V12 type masks have correct counts — 118 elements, 10 digits, 0 fractions, 19 special, 1 EOS

4. **NumDen loss**: Confirm numden_loss appears in training metrics and decreases over epochs

5. **AR generation quality**: Run eval with `true_exact` to verify generation quality is preserved (~85.4%)

---

## 12. Usage

### Standard V12.41 Training (default config)

```bash
cd /home/james/superconductor-vae
conda activate recursivemenn-py311
PYTHONPATH=src python scripts/train_v12_clean.py
```

The default config is set for V12.41 mode. The checkpoint auto-detection will find `outputs/checkpoint_best.pt`.

### Explicit Checkpoint Path

In `train_v12_clean.py`, set:
```python
resume_checkpoint = '/path/to/checkpoint_epoch_3292.pt'
```

### Monitoring NumDen Loss

NumDen loss is logged as `numden_loss` in the per-epoch metrics output and in `outputs/training_log.csv`.
A healthy numden_loss should start near the last V12.41 value and decrease or remain stable.
