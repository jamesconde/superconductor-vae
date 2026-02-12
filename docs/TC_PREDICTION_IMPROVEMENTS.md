# Tc Prediction Improvements (V12.28)

## Overview

V12.28 implements 7 improvements to critical temperature (Tc) prediction accuracy,
organized into 4 phases: Data, Architecture, Loss, and Inference.

Previous Tc error: ~19% across all Tc ranges.

---

## Phase A: Data Ingestion

### A1-A2: JARVIS Dataset Integration
**Script**: `scripts/ingest_jarvis.py`

Ingests two JARVIS superconductor datasets:
- `data/raw/jarvis_supercon_chem.csv` (16,414 rows with formula + Tc)
- `data/raw/jarvis_supercon_3d.csv` (1,058 rows with JVASP ID + Tc, formulas recovered from DFT-3D lookup)

Pipeline follows the exact pattern of `scripts/ingest_nemad.py`:
1. Load raw CSV, extract Tc
2. Parse formulas with pymatgen, canonicalize
3. Assign SC family categories (Cuprates, Iron-based, etc.)
4. Compute 145 matminer features (Lp norms + Magpie + valence + ion + TM fraction)
5. Compute 6 new physics features (see A3)
6. Convert to fraction notation
7. Deduplicate against existing data
8. Filter out 45 holdout samples
9. Merge into `supercon_fractions_contrastive.csv`

**Output**: Report to `scratch/jarvis_ingest_report.txt`

### A3: Physics Feature Columns
**Script**: `scripts/add_physics_features.py` (backfill existing CSV)
**Also in**: `scripts/ingest_jarvis.py` (computed during ingestion)

6 new physics-informed features added to all data:

| Column | Type | Description |
|--------|------|-------------|
| `has_cuprate_elements` | int (0/1) | Contains Cu AND O |
| `cu_o_ratio` | float | Cu fraction / O fraction (0 if missing) |
| `has_iron_pnictide` | int (0/1) | Contains Fe AND (As/Se/P) |
| `has_mgb2_elements` | int (0/1) | Contains Mg AND B |
| `hydrogen_fraction` | float | H fraction in composition |
| `transition_metal_count` | int | Number of distinct transition metal elements |

These are automatically picked up as Magpie features by the training script (magpie_dim increases from 145 to 151 when present).

---

## Phase B: Architecture Changes

### B1: Robust Checkpoint Loading
**Files**: `scripts/train_v12_clean.py`, `scripts/train.py`

Added shape-mismatch filtering before `load_state_dict(strict=False)`:

```python
model_state = encoder.state_dict()
for key in list(enc_state.keys()):
    if key in model_state and enc_state[key].shape != model_state[key].shape:
        del enc_state[key]  # Re-initialized from scratch
```

This ensures architecture changes (magpie_dim change, SC head input dim change, etc.) are handled cleanly. Mismatched layers are logged and randomly re-initialized while all compatible weights load normally.

### B2: Deeper Residual Tc Head (Net2Net Transfer)
**File**: `src/superconductor/models/attention_vae.py`

Old architecture (2-layer MLP):
```
tc_head: Linear(512→256) → GELU → Linear(256→1)
```

New architecture (4-layer residual MLP):
```
tc_proj: Linear(512→256)
tc_res_block: Linear(256→256) → LayerNorm → GELU → Dropout → Linear(256→256)  [identity-init]
tc_out: LayerNorm → GELU → Linear(256→128) → GELU → Linear(128→1)
```

**Net2Net Weight Transfer** (`upgrade_tc_head_from_checkpoint()`):
- `tc_head.0` weights → `tc_proj` (direct copy, same shape 512→256)
- `tc_res_block` initialized as identity (preserves function initially)
- `tc_head.2` weights → `tc_out[4]` via wider expansion with identity projection
- Uses `net2net_expansion.py` utilities from existing codebase

The residual connection means the new head starts with approximately the same function as the old head, then has additional capacity to learn.

### B3: Tc Classification Auxiliary Head
**File**: `src/superconductor/models/attention_vae.py`

New head classifies Tc into 5 buckets:
- 0: non-SC (Tc = 0)
- 1: low (0-10 K)
- 2: medium (10-50 K)
- 3: high (50-100 K)
- 4: very-high (100+ K)

```python
tc_class_head: Linear(512→256) → GELU → Dropout → Linear(256→5)
```

Returns `tc_class_logits` in model output dict.

### B4: Dynamic magpie_dim
The training script dynamically detects magpie_dim from CSV columns (all numeric columns except excluded ones). Adding physics features to the CSV automatically increases magpie_dim from 145 to 151 with no code changes needed.

### B5: SC Head Input Dim Update
SC head now receives `tc_class_logits` (5 dims) as additional input:
```
sc_input_dim = latent_dim + 1 + magpie_dim + 1 + 12 + 1 + 1 + 5  # +5 for tc_class
```

Old checkpoint sc_head weights are shape-mismatched → cleanly re-initialized by B1.

---

## Phase C: Loss Function Changes

### C1: Tc Binned Loss Weighting
**File**: `scripts/train_v12_clean.py` (CombinedLossWithREINFORCE)

Per-bin weight multipliers stack on top of existing Kelvin weighting:

```python
'tc_bin_weights': {0: 1.0, 10: 1.5, 50: 2.0, 100: 2.5, 150: 3.0}
```

Samples with Tc >= 150K get 3x weight, Tc >= 100K get 2.5x, etc. This focuses gradient budget on high-Tc bins where absolute error matters most for discovery.

### C2: Tc Classification Loss
**File**: `scripts/train_v12_clean.py` (CombinedLossWithREINFORCE)

Cross-entropy loss on Tc bucket predictions:
```python
'tc_class_weight': 2.0,
'tc_class_bins': [0, 10, 50, 100],  # Creates 5 classes
```

This auxiliary loss provides coarse-grained signal that helps the Tc head learn bucket boundaries, complementing the fine-grained regression loss.

### C3: Forward Signature Updates
All 4 loss_fn call sites (pure SC, pure non-SC, mixed SC, mixed non-SC) now pass `tc_class_logits`. Metrics accumulation and logging updated to track `tc_class_loss`.

---

## Phase D: MC Dropout Inference

### D1: predict_tc_mc Method
**File**: `src/superconductor/models/attention_vae.py`

```python
def predict_tc_mc(self, z, n_samples=10):
    """MC Dropout: run decode N times with dropout active."""
    # Returns (tc_mean, tc_std)
```

At inference time, enables dropout and runs N forward passes:
- Mean prediction is more robust than single-pass
- Standard deviation provides uncertainty estimate
- High uncertainty correlates with high error (can be used to flag unreliable predictions)

Config: `mc_dropout_samples: 10` (adjustable)

---

## New Config Parameters

```python
# TRAIN_CONFIG additions
'tc_class_weight': 2.0,     # Tc bucket classification loss weight
'tc_class_bins': [0, 10, 50, 100],  # Bin edges in Kelvin
'tc_bin_weights': {0: 1.0, 10: 1.5, 50: 2.0, 100: 2.5, 150: 3.0},
'mc_dropout_samples': 10,   # MC Dropout forward passes at eval
```

---

## Files Modified

| File | Changes |
|------|---------|
| `scripts/ingest_jarvis.py` | **NEW** — JARVIS data ingestion |
| `scripts/add_physics_features.py` | **NEW** — Backfill physics features |
| `scripts/migrate_checkpoint_v1228.py` | **NEW** — One-time checkpoint migration (old arch → V12.28) |
| `src/superconductor/models/attention_vae.py` | Tc head upgrade (B2), Tc class head (B3), SC head dim (B5), MC dropout (D1), Net2Net transfer |
| `scripts/train_v12_clean.py` | Checkpoint loading (B1), loss changes (C1-C3), config, forward pass, metrics logging |
| `scripts/train.py` | Checkpoint loading (B1) |
| `docs/TC_PREDICTION_IMPROVEMENTS.md` | **NEW** — This document |

---

## Checkpoint Compatibility

### One-Time Migration Script

**Script**: `scripts/migrate_checkpoint_v1228.py`

After syncing new code to wherever the latest weights live (e.g., Google Drive / Colab), run:
```bash
python scripts/migrate_checkpoint_v1228.py                              # migrates outputs/checkpoint_best.pt
python scripts/migrate_checkpoint_v1228.py --checkpoint path/to/ckpt.pt  # custom path
python scripts/migrate_checkpoint_v1228.py --dry-run                     # preview without saving
```

The script:
1. Backs up the original checkpoint as `checkpoint_best.pt.bak_pre_v1228`
2. Strips `_orig_mod.` prefixes (torch.compile artifacts)
3. Applies shape-mismatch filtering (re-initializes reshaped layers)
4. Applies Net2Net weight transfer (old tc_head → new tc_proj/tc_res_block/tc_out)
5. Saves a clean migrated checkpoint
6. Resets optimizer state (must be rebuilt since param shapes changed)

### What Gets Migrated

When loading an old checkpoint (pre-V12.28) into the new architecture:

1. **Shape-mismatch filtering** removes keys with changed dimensions:
   - `magpie_encoder.0.weight` (145→151 input)
   - `magpie_head.2.weight/bias` (145→151 output)
   - `sc_head.0.weight` (input dim changed by +5 tc_class + +6 physics)

2. **Net2Net transfer** maps old `tc_head.*` weights to new `tc_proj/tc_res_block/tc_out`:
   - Direct copy for compatible layers
   - Identity initialization for new layers
   - Wider expansion for output layer

3. **New heads** (`tc_class_head`, `tc_res_block`, `tc_out`) are randomly initialized (identity where appropriate).

4. **Optimizer state** is reset (parameter shapes changed, old state is invalid).

---

## Verification Checklist

1. Run `python scripts/ingest_jarvis.py` — verify row count increases, no holdout contamination
2. Load old checkpoint with new architecture — verify warning messages, no crashes
3. Run one training epoch — verify tc_loss is non-zero and decreasing
4. Verify `TcCl:` metric appears in training logs and accuracy > 20% quickly (random = 20%)
5. Verify high-Tc samples get larger gradients (check per-bin Tc errors)
6. MC Dropout at eval: verify uncertainty correlates with error
7. After ~50 epochs, Tc loss should be lower than pre-change baseline
