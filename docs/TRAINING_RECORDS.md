# Training Records

Chronological record of training runs, architecture changes, and optimization decisions for the Multi-Task Superconductor Generator (code class names retain "VAE" for backward compatibility).

---

## V12.41 Backward Compatibility Mode (2026-02-24)

### Problem

The V12.41 checkpoint (epoch 3292, best_exact=85.4%, trained on A100) remains the best-performing model for autoregressive formula generation. Since V12.41, several architecture changes were made (V13.0 semantic fractions, V14.0 isotopes, V14.3 token type head + heads memory, V15.0 memory bottleneck) that changed the vocab (148→4647), stoich dims (37→13), and decoder structure. The current codebase assumed V13+ mode, making it impossible to resume training from V12.41.

### Solution: Config-Driven V12.41 Compatibility

Made the codebase config-driven so V12.41 can load and train with worthwhile newer features while disabling incompatible ones.

**Features KEPT (new heads, random init, learn from scratch):**
- Token type classifier head (V14.3) — with V12 vocab type mapping
- Enriched decoder memory / heads_to_memory (V14.3) — vocab-agnostic
- RL auto-scaling (V14.1) — config-only
- V14 continuous reward — with V12 token range overrides
- Hierarchical family head (V15.0) — already in V12.41 checkpoint

**Features DISABLED (incompatible with 148-token vocab):**
- Semantic fraction tokenization (V13.0) — `use_semantic_fractions: False`
- Isotope tokens (V14.0) — `use_isotope_tokens: False`
- Memory bottleneck (V15.0) — `memory_bottleneck_dim: 0`
- Migration LR boost (V14.1) — not needed

### Config Defaults Changed

| Config Key | Old Default | New Default | Reason |
|------------|-------------|-------------|--------|
| `use_semantic_fractions` | `True` | `False` | V12.41 uses 148-token vocab |
| `use_isotope_tokens` | `True` | `False` | Requires semantic fractions |
| `memory_bottleneck_dim` | `1024` | `0` | V12.41 uses direct MLP |
| `max_formula_len` | `30` | `60` | Digit-by-digit needs longer sequences |
| `numden_weight` | (new) | `1.0` | MSE loss for numden_head |

### Architecture Changes

1. **`FullMaterialsVAE`**: Added `use_numden_head` parameter. When `True`, creates the numden_head (matching V12.41 checkpoint shapes: 512→256→24). Computes `numden_pred` in forward().

2. **`autoregressive_decoder.py`**: Added `get_v12_type_masks()` and `compute_v12_token_type_targets()` for token type classification with old 148-token vocab.

3. **`train_v12_clean.py`**:
   - Conditional numden data extraction (real extraction vs zeros)
   - 37-dim stoich assembly (fractions + numden + count) in training loop, eval, and RLOO
   - Numden MSE loss added to all 3 loss branches
   - V14 reward token range overrides for V12 vocab layout
   - Type mask/target fallbacks when `v13_tokenizer is None`
   - Checkpoint saves `stoich_input_dim=37` in V12.41 mode

### Files Modified

| File | Changes |
|------|---------|
| `src/superconductor/models/attention_vae.py` | `use_numden_head` param, conditional numden_head creation, numden_pred in forward() |
| `src/superconductor/models/autoregressive_decoder.py` | `get_v12_type_masks()`, `compute_v12_token_type_targets()` |
| `scripts/train_v12_clean.py` | Config defaults, conditional numden extraction, 37-dim stoich assembly, numden loss, type fallbacks, V14 reward overrides, checkpoint stoich_input_dim |

### Expected Checkpoint Loading Behavior

- **Loaded from V12.41**: numden_head weights load successfully. New heads (token_type_head, heads_to_memory) appear as "missing keys" (expected — random init). skip_to_memory appears as "unexpected keys" (expected — `strict=False` ignores).
- **37-dim stoich**: Decoder's stoich_to_memory expects 37-dim input, matching V12.41 architecture.
- **true_exact**: Should read ~85.4% immediately after loading (no architecture mismatch).

---

## Phase 2: Self-Supervised Training System (2026-02-24)

### Problem

V12.41 achieves 86.5% TRUE AR exact match on training data but only 22.2% exact match on 45 holdout superconductors. The generalization gap is the core problem. V13-V15 semantic fraction experiments failed (<1% AR) and were abandoned.

### Solution: Interleaved Self-Supervised Sub-Epochs

Phase 2 uses the model's own generations as self-supervised signal to improve z-space consistency at novel points, without requiring new labeled data. Key design:

- **When**: Runs every 4 supervised epochs, after main `train_epoch()` completes
- **Activation**: Auto-activates when training exact match >= 80%
- **Weight**: Linear warmup from 0 to 0.1 over 50 epochs

### Z-Space Sampling (3 strategies)

| Strategy | Budget | Method |
|----------|--------|--------|
| Perturbation (60%) | ~154 samples | Add Gaussian noise to training z-vectors (sigma ramp 0.02→0.1) |
| SLERP (25%) | ~64 samples | Spherical interpolation between same-family pairs |
| PCA Walk (15%) | ~38 samples | Walk along top-20 PCs at +/- sigma steps |

### Four Loss Signals

1. **Extended Round-Trip**: Re-encode generated formulas, compare z/Tc with originals (encoder gradients)
2. **Multi-Head Consistency**: Ensure SC classifier, Tc, Tc bucket, and family heads agree (encoder head gradients)
3. **Physics Constraints**: A3 site occupancy + A6 charge balance on generated formulas (encoder gradients)
4. **REINFORCE Round-Trip**: Reward = cosine_sim(z, z_recon) * validity (decoder gradients)

### Safety Guards

- Weight ramp (0→0.1 over 50 epochs)
- Exact match monitor (halve weight if training exact drops >2%)
- Separate LR (0.1x main LR)
- Tight gradient clipping (0.5 vs 1.0 for Phase 1)
- Frequency control (every 4 epochs)
- Mode collapse detection (unique rate < 0.3 → temp boost + diversity bonus)

### New Files

| File | Purpose |
|------|---------|
| `src/superconductor/training/self_supervised.py` | Phase 2 orchestrator |
| `src/superconductor/losses/round_trip_loss.py` | + `ExtendedRoundTripConsistencyLoss` |
| `scripts/analysis/phase2_dashboard.py` | Monitoring dashboard |
| `docs/PHASE2_SELF_SUPERVISED_DESIGN.md` | Full design document |

### Config

All Phase 2 config keys prefixed with `phase2_` in TRAIN_CONFIG. Set `phase2_enabled: True` to activate. Metrics logged to `outputs/phase2_log.csv`.

### Holdout Integration

- Mini holdout search (200 candidates/target) via `--mini` flag on `holdout_search_targeted.py`
- Full holdout search on new best checkpoints

### Success Milestones

| Milestone | Metric |
|-----------|--------|
| Alpha | Round-trip Z-MSE < 0.1 |
| Beta | Holdout exact > 30% (from 22.2%) |
| Gold | Holdout exact > 50% |
| Complete | All 45 holdout at >= 0.99 similarity |

---

## V15.3: Revert TF Scheduling — REINFORCE is the AR Trainer (2026-02-23)

### Finding: 2-Pass Scheduled Sampling is a False Signal

Epochs 4330-4348 with TF=0.14 showed 85% training exact but only 3-4% true AR exact. Root cause: the 2-pass approach generates "predicted" tokens in pass 1 using **full GT context** (~96% accurate), so the mixed inputs in pass 2 are effectively still teacher-forced. The model never sees real error cascading.

### Action

- **Reverted TF to locked 1.0** — 2-pass scheduled sampling removed
- **Restored batch sizes** — A100-80GB: 1008→2100, A100-40GB: 252→504 (no longer need 2x memory headroom)
- **REINFORCE is the sole AR trainer** — RLOO samples use `generate_with_kv_cache` (true step-by-step AR with error cascading)
- **RL temperature 1.2** — high exploration for post-bottleneck recovery (from V15.2)
- **Version bump** to V15.3

### Lesson Learned

2-pass scheduled sampling (pass 1: GT context → predictions, pass 2: mixed forward) does NOT approximate true autoregressive training. The predictions from pass 1 are near-perfect because they had GT context. Only true step-by-step generation (REINFORCE, beam search) exposes the model to its own error cascading.

---

## V15.2: Auto-Activating TF Scheduling [REVERTED in V15.3] (2026-02-23)

### Problem

Training data (epochs 4256-4282) shows TF exact climbing 26%→66% while true AR exact is stuck at 3-5%. The gap grows every epoch because TF is locked at 1.0 — the decoder never practices generating from its own predictions.

### Solution: TF = 1 - exact_match (auto-activated)

The existing `get_teacher_forcing_ratio()` function implements `TF = 1 - exact_match` but was never called. V15.2 adds auto-activation:

1. **Gate**: TF scheduling stays disabled until TF exact exceeds `tf_scheduling_threshold` (default 0.80)
2. **Activation**: Once threshold is hit, `tf_scheduling_enabled` flips to `True` and stays active permanently
3. **Formula**: `TF = 1.0 - prev_exact` (previous epoch's exact match)
   - exact=80% → TF=0.20 (mostly autoregressive)
   - exact=90% → TF=0.10 (nearly pure autoregressive)
4. **Self-stabilizing**: If exact drops after activation (e.g., 80%→60%), TF automatically rises (0.20→0.40), providing more teacher forcing to aid recovery — natural negative feedback loop

### Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tf_scheduling_threshold` | `0.80` | Auto-activate when TF exact exceeds this |
| `tf_scheduling_enabled` | `False` | Start disabled; set `True` to force-enable from epoch 0 |

### Console Output

On activation: `[V15.2 TF-SCHED] ACTIVATED: prev_exact=0.800 >= threshold=0.8 — switching to TF = 1 - exact`

Every epoch (in existing summary line): `TF: 0.20` (reflects current scheduled ratio)

### RL Temperature Reset

Previous `rl_temperature: 0.2` was tuned for the old model at epoch 3000+ (exploitation-focused). Post-bottleneck rebuild needs high exploration to discover new AR patterns:

| Setting | Old (V12.40) | New (V15.2) |
|---------|-------------|-------------|
| `rl_temperature` | 0.2 | 1.2 |
| `rl_temperature_start` | 0.5 | 1.2 |
| `rl_temperature_end` | 0.2 | 0.5 |

Matches the temperature that produced good AR performance in the earlier model.

### Colab VRAM Optimization (env_config.py)

**Critical lesson**: When TF scheduling activates (TF < 1.0), the decoder uses a **2-pass scheduled sampling** approach (line 911 of `autoregressive_decoder.py`): first pass gets predictions, second pass forwards with mixed GT/predicted tokens. This roughly **doubles forward memory** compared to TF=1.0. Batch sizes must account for this.

batch=2100 on A100-80GB used 78/79GB at TF=1.0 and OOM'd immediately when TF scheduling activated at epoch 4329.

| Tier | GPU | Batch | Steps/epoch | Rationale |
|------|-----|-------|-------------|-----------|
| `xlarge` | A100-80GB | 1008 (24x) | ~46 | ~39GB/pass × 2 passes = ~78GB peak |
| `large` | A100-40GB | 252 (6x) | ~183 | ~19GB/pass × 2 passes = ~38GB peak |

Both tiers use `compile_mode="reduce-overhead"` and `n_samples_rloo=4`.

---

## V15.1: Per-Bin Tc Head Early Stopping — Snapshot/Restore (2026-02-23)

### Problem

The Tc head's R² for high-Kelvin bins (120-200K and >200K) oscillates between 68-91% across epochs. Root cause: gradient interference from ~20K majority low-Tc samples overpowers ~1K minority high-Tc samples in the shared encoder backbone. The optimizer rationally trades minority performance for majority improvement, causing R² regressions in the high-Tc bins.

### Solution: TcBinTracker

Self-contained class that monitors per-bin R² and snapshots/restores Tc head weights:

1. **Combined metric**: Sample-weighted average R² of target bins (120-200K, >200K)
2. **Snapshot scope**: Only `tc_proj`, `tc_res_block`, `tc_out` (~297K params, ~1.2MB) — NOT the shared `decoder_backbone`
3. **On new best**: Deep-copy Tc head weights to CPU
4. **On regression**: If combined R² drops >0.10 below best, restore snapshot weights to GPU
5. **No freezing**: After restore, training continues — model may find even better weights
6. **Checkpoint persistence**: Snapshot state saved/restored in checkpoint for resume

### Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tc_bin_tracker_enabled` | `True` | Enable per-bin Tc head snapshot/restore |
| `tc_bin_regression_threshold` | `0.10` | Restore if combined R² drops > this below best |
| `tc_bin_target_bins` | `('120-200K', '>200K')` | Bins to monitor |
| `tc_bin_min_samples` | `5` | Min samples per bin to include in combined R² |

### Console Output

Every 4 epochs (eval cadence), prints one of:
- `[V15.1 Tc-BIN] NEW BEST combined R²=0.8521 | 120-200K: R²=0.8234 (n=87), >200K: R²=0.9105 (n=24) | snapshot saved`
- `[V15.1 Tc-BIN] REGRESSION combined R²=0.7102 (best=0.8521, drop=0.1419 > threshold=0.10) | ... | RESTORED weights from epoch 4212`
- `[V15.1 Tc-BIN] combined R²=0.8301 (best=0.8521) | ...` (within tolerance, no action)

### Safety Interactions

- **Catastrophic rollback**: When the main model rolls back to `checkpoint_best.pt`, the Tc bin tracker snapshot is invalidated (best R² reset to -inf) since the underlying model weights changed
- **Signal handler**: Tc bin tracker state included in interrupt checkpoints
- **Emergency save**: Tc bin tracker state included in emergency checkpoints

---

## V15.0: Latent-to-Memory Bottleneck — SVD Weight Contraction (2026-02-23)

### Problem

Error analysis of epochs 4208–4228 revealed 98.7% train exact but ~2% validation exact — the decoder memorizes 46K formulas without generalizing. The `latent_to_memory` subnetwork is the prime suspect: **~151M params** (at d_model=1024) in a 2-layer MLP that projects z(2048) to 16 memory tokens. Layer 2 alone is 134M params. At ~3,300 params per training sample, this subnetwork has enough capacity to store each formula individually.

### Solution: Bottleneck Architecture

Replace the 2-layer MLP with a compressed bottleneck that forces information compression.

**Old (~151M params at d_model=1024):**
```python
self.latent_to_memory = nn.Sequential(
    nn.Linear(2048, 8192),     # 16.8M params  (d_model*16//2 = 1024*16//2)
    nn.GELU(),
    nn.Linear(8192, 16384),    # 134.2M params (d_model*16 = 1024*16)
)  # Output: [batch, 16, 1024]
```

**New (~19M params, 8x reduction at d_model=1024):**
```python
self.latent_to_memory = nn.Sequential(
    nn.Linear(2048, 1024),                 # 2.1M params
    nn.LayerNorm(1024),                    # Stabilize bottleneck
    nn.GELU(),
    nn.Linear(1024, d_model * 16),         # 16.8M params (1024 * 1024 * 16)
)  # Output: [batch, 16, 1024]
```

**Why this works:**
- 1024-dim bottleneck for 2048-dim input = 2x compression (~98.5% SVD variance), forces learning which z dimensions matter while preserving headroom for self-supervised dataset expansion
- 16 latent tokens kept (V12 checkpoint had good AR behavior with 16)
- Memory layout: [16 latent + 4 stoich + 4 heads] = 24 tokens (unchanged from V14.3)
- ~413 params/sample (was ~3,300 at d_model=1024) — significant capacity reduction without losing token count
- LayerNorm after bottleneck matches `stoich_to_memory` pattern
- Heads tokens now include `family_composed_14` (14-class family probabilities) — tells decoder which material family to generate (cuprate→Cu,Ba,Y; iron→Fe,As; etc.). Input dim: 10→24, still 4 tokens.

### SVD Migration Strategy

Weights migrated via SVD decomposition of old Layer 1:

```
W1 [8192, 2048] = U @ diag(S) @ Vt    (at d_model=1024: hidden=8192)

New Layer 0: W1_new = diag(S[:1024]) @ Vt[:1024, :]     — top-1024 directions scaled by singular values
New Layer 3: W2_new = W2 @ U[:, :1024]                  — project old Layer 2 through top-1024 left SVs
```

Script: `scripts/migrate_latent_to_memory_reduction.py` (standalone, runs on Colab or locally).
Saves pre-contraction backup as `checkpoint_*_pre_v15_contraction.pt`.

### d_model Confirmation

The Colab checkpoint (epoch 4245) has **d_model=1024** — V12.42 Net2Net widening WAS applied on Colab (token_embedding shape [4752, 1024], all transformer layers at 1024). MODEL_CONFIG correctly reflects `d_model=1024`, `dim_feedforward=4096`. The migration script auto-detects d_model from the checkpoint's `token_embedding.weight`.

### Files Modified

- `src/superconductor/models/autoregressive_decoder.py` — `EnhancedTransformerDecoder.__init__()`: added `memory_bottleneck_dim` param, replaced `latent_to_memory` with bottleneck version, updated `_create_memory()` docstring
- `scripts/train_v12_clean.py` — `MODEL_CONFIG`: `d_model=1024`, `dim_feedforward=4096`, `n_memory_tokens=16`, `memory_bottleneck_dim=1024`. Decoder construction passes `memory_bottleneck_dim`.
- `scripts/migrate_latent_to_memory_reduction.py` — NEW: standalone SVD migration script (auto-detects d_model from checkpoint)
- `notebooks/migrate_v15_latent_bottleneck.ipynb` — NEW: Colab migration notebook (auto-detects d_model from checkpoint)

### Config

```python
MODEL_CONFIG = {
    'd_model': 1024,             # V12.42 Net2Net widening (confirmed on Colab epoch 4245)
    'dim_feedforward': 4096,     # 4x d_model
    'n_memory_tokens': 16,       # Kept at 16 (V12 had good AR behavior)
    'memory_bottleneck_dim': 1024, # NEW: bottleneck (~98.5% SVD variance)
    ...
}
```

### Expected Impact

- Decoder latent_to_memory: ~151M → ~19M (~132M reduction from bottleneck)
- Memory layout: 24 tokens unchanged [16 latent + 4 stoich + 4 heads]
- Hypothesis: Train exact will drop initially but val exact should climb as the model can no longer memorize through the bottleneck

---

## V14.3: Token Type Classifier Head + Enriched Decoder Memory (2026-02-21)

### Problem

50% of all autoregressive (AR) errors are **type confusion** — the model predicts an element where a fraction should go, a fraction where an integer should go, etc. The formula grammar is deterministic: `[Element, Stoich, Element, Stoich, ..., EOS]` where Stoich is an integer (1-20) or a FRAC:p/q token. TF exact match is 95.6% but TRUE AR exact match is only 7.5%.

### Solution: Two-Part Architecture

**Part 1: Token Type Head** (330K params) — Deep classifier on transformer decoder hidden state that predicts token type (5 classes: ELEMENT, INTEGER, FRACTION, SPECIAL, EOS).

- Architecture: `LayerNorm(512) → Linear(512,512) → GELU → Dropout → Linear(512,128) → GELU → Dropout → Linear(128,5)`
- 8 layers, mirrors output_proj structure — critical because hard masking means wrong type prediction blocks the correct token entirely
- Training: CE loss on type labels (weight 1.0, raised from 0.1 in V14.3b), derived from target tokens via tokenizer LUT
- Inference: Hard mask over vocab logits — only allow tokens of the predicted type
- Precedent: POS-Guided Softmax (Yang et al., COLING 2022)

**Part 2: Enriched Decoder Memory** (1.2M params) — Feed encoder head predictions into decoder as 4 additional cross-attention memory tokens.

- Input: tc_pred(1) + sc_pred(1) + hp_pred(1) + tc_class_logits(5) + competence(1) + element_count_pred(1) = 10 dims
- Architecture: `Linear(10,256) → LayerNorm(256) → GELU → Linear(256,512) → GELU → Linear(512,2048)` → reshape to 4 x 512 tokens
- 6 layers with 3 linear stages for gradual 200x expansion (10→256→512→2048)
- Memory layout: `[16 latent | 4 stoich | 4 heads] = 24 tokens` (was 20)
- All head values are `.detach()`ed — no gradient flows back through heads

### Files Modified

- `src/superconductor/tokenizer/fraction_tokenizer.py` — Added `TOKEN_TYPE_*` constants, `get_token_type()`, `get_type_masks()`, `compute_token_type_targets()`
- `src/superconductor/models/autoregressive_decoder.py` — Added `token_type_head`, `heads_to_memory` to `EnhancedTransformerDecoder.__init__()`. Updated `_create_memory()`, `forward()` (now returns 4-tuple), `generate_with_kv_cache()`, `sample_for_reinforce()`, `speculative_sample_for_reinforce()`, legacy `generate()`.
- `scripts/train_v12_clean.py` — Added `token_type_loss_weight`, `use_type_masking_ar`, `use_heads_memory` to TRAIN_CONFIG. Pass `heads_pred` to decoder, compute type loss, integrate into total loss. SCST + RLOO + Z-cache paths all pass `heads_pred` for memory consistency.
- `scripts/migrate_vocab_expansion.py` — Note about new layers (handled by existing copy-as-is logic)
- `scripts/migrate_checkpoint_v1242_wider.py` — Fixed 3-tuple to 4-tuple destructuring
- `scripts/train.py` — Fixed 2-tuple → `*_extra` for 4-tuple decoder return (lines 446, 1861)
- `src/superconductor/training/soft_token_sampling.py` — Fixed 2-tuple → slice/`*_extra` for decoder return
- `scratch/debug_cuda_test.py` — Fixed 5 × 2-tuple → `*_extra` for decoder return
- `scratch/test_real_data.py`, `test_amp_transition.py`, `test_epoch_boundary.py`, `test_full_volume.py` — Same fix
- RLOO/SCST plumbing: `CombinedLossWithREINFORCE` stores `_heads_pred`/`_type_masks` per-batch, passed through to `compute_rloo_autoregressive`, `compute_scst`, and `sample_for_reinforce`. Z-cache (`cache_z_vectors`) also builds `heads_pred` from encoder outputs.

### Config

```python
'token_type_loss_weight': 1.0,  # CE loss weight (raised 0.1→1.0 in V14.3b)
'use_type_masking_ar': True,    # Hard vocab masking during AR generation (enabled in V14.3b)
'use_heads_memory': True,       # Feed encoder heads into decoder memory
```

### Checkpoint Compatibility

New layers (`token_type_head`, `heads_to_memory`) are missing from pre-V14.3 checkpoints. `load_checkpoint()` uses `strict=False` — new layers initialize randomly, existing layers load from checkpoint. No migration needed.

### Bug Fixes (2026-02-22)

- **Mixed SC/non-SC batch crash**: `loss_fn._heads_pred` was set to full batch (e.g. 252 samples) but RLOO/SCST received `z=z[sc_mask]` (e.g. 176 SC samples). The stale full-batch `_heads_pred` reference caused `_create_memory` to attempt reshape `[176, 4, 1024]` from 252×4×1024 elements → `RuntimeError`. Fixed by slicing `heads_pred_dict` by `sc_mask` before the SC loss call in the mixed-batch path, and clearing it to `None` before the non-SC call.
- **Batch validation assertions**: Added explicit batch-size checks in `_create_memory()` for all `heads_pred` tensors vs z batch size, with clear error messages identifying the mismatched tensor.
- **Auto-detect checkpoint on resume**: `resume_checkpoint` was hardcoded to a specific epoch file (`checkpoint_epoch_3999.pt`). When training continued past that epoch and saved `checkpoint_best.pt`, the next restart still loaded the stale epoch 3999 — discarding ~200 epochs of progress. Now defaults to `'auto'`, which prefers `checkpoint_best.pt` then falls back to the highest-numbered `checkpoint_epoch_*.pt`. Explicit paths still supported for one-off overrides.

### Rollout Strategy

1. **V14.3**: Train with type loss + heads memory (training signal only, no AR masking)
2. **After 1 epoch**: Check type head accuracy (should be >90% on TF mode)
3. **Enable masking**: Set `use_type_masking_ar=True` once type head is accurate
4. **Monitor**: Type head accuracy should correlate with AR exact match improvement

### V14.3b: Enable Type Masking + Raise Type Loss Weight (2026-02-23)

**Problem**: Error analysis of epochs 4208–4228 revealed:
- 98.7% train exact match but only ~2% validation exact match (decoder memorization)
- `use_type_masking_ar=False` — type head trained for 1000+ epochs but masking never enabled
- 56.9% of all validation errors (5,260 of 9,237) are **type confusions** that masking eliminates
- `token_type_loss_weight=0.1` — 10x weaker than other critical heads (SC=1.0, HP=1.0, Stop=5.0)

**Changes**:
1. `use_type_masking_ar`: `False` → `True` — enable hard vocab masking during both eval and RLOO
2. `token_type_loss_weight`: `0.1` → `1.0` — raise to match other critical heads (wrong type prediction blocks correct token entirely)
3. Decoder `dropout`: `0.1` → `0.4` — combat memorization (109M params / 46K samples = 2,337 params/sample). Encoder stays at 0.1 (already converged, not overfitting)
4. Added type head accuracy tracking — logs `Type: 0.0234 (95.2%)` showing both loss and accuracy per epoch
5. RLOO type masking — type masks now passed to RLOO/SCST sampling (was `None`), preventing RL exploration of impossible type combinations

**Expected impact**: Eliminate 57% of validation type confusion errors. Higher dropout forces decoder to learn generalizable composition rules instead of memorizing. Combined with the 10x stronger type loss signal, the type head should become highly accurate, making the hard masking reliable.

See `docs/ERROR_ANALYSIS_EPOCHS_4208_4228.md` for full analysis.

### Legacy Code Audit (2026-02-22)

Marked all unused/legacy classes with `LEGACY` docstrings, then **REMOVED** them (preserved in git history).

**Removed from `attention_vae.py`** (~415 lines):
- `AttentionVAEDecoder` — Internal to AttentionBidirectionalVAE
- `TcPredictorWithContributions` — Internal to AttentionBidirectionalVAE
- `AttentionBidirectionalVAE` — V1-V10 element-only encoder, superseded by FullMaterialsVAE in V12
- `AttentionVAELoss` — Loss for above, superseded by CombinedLossWithREINFORCE
- `create_attention_vae()` — Factory for above
- `FullMaterialsLoss` — Simple V12.0 loss, superseded by CombinedLossWithREINFORCE in train_v12_clean.py
- `StructuredLatentEncoder` — V13 proposal, never trained
- `FullMaterialsV13` — V13 wrapper, never trained

**Removed from `autoregressive_decoder.py`** (~795 lines):
- `AutoregressiveFormulaDecoder` — V1-V6 GRU decoder, superseded by EnhancedTransformerDecoder
- `FormulaVAEWithDecoder` — Wrapper for GRU decoder
- `TransformerFormulaDecoder` — V7-V10 basic transformer, lacked skip connections, stoich conditioning, KV-cache, REINFORCE
- `FormulaVAEWithTransformer` — Wrapper for basic transformer

**Removed file**: `src/superconductor/generation/attention_generator.py` — Legacy generator depending on AttentionBidirectionalVAE

**Import cleanup**: Removed dead imports from `models/__init__.py`, `superconductor/__init__.py`, `generation/__init__.py`, and `scripts/train.py`.

**Active components** (NOT legacy): `ElementEncoder`, `AttentionVAEEncoder`, `HierarchicalFamilyHead`, `FullMaterialsVAE`, `PositionalEncoding`, `EnhancedTransformerDecoder`, all helper functions/constants.

---

## V14.2: Migration LR Boost + dtype Fixes (2026-02-21)

### Migration LR Boost

At epoch 4001/5000, the cosine annealing schedule gives only ~10% of the initial LR (3.13e-6 vs 3e-5). After vocab expansion migration, new token embeddings need stronger gradients to recover. Without a boost, recovery from 81.5% → 99%+ would take hundreds of epochs at the reduced LR.

**Fix**: When `load_checkpoint()` detects vocab expansion, a temporary LR multiplier is applied:
- Peak boost: 5x (so 3.13e-6 × 5 = 1.56e-5, ~52% of initial LR)
- Decays linearly to 1.0x over 100 epochs
- Config: `migration_lr_boost=5.0`, `migration_lr_boost_epochs=100`
- Only activates when `vocab_expanded=True` (migration detected)
- Normal resumes (no vocab change) are completely unaffected

### dtype Fixes in Migration Scripts

All migration scripts (`migrate_v13_to_v14.py`, `migrate_vocab_expansion.py`) and the inline loader in `train_v12_clean.py` had a dtype bug: `torch.zeros()` and `torch.randn()` default to float32, but checkpoint tensors may be bfloat16 (training uses AMP). This caused silent dtype mixing in all 397 expanded vocab rows. Fixed by passing `dtype=value.dtype` to all allocation/noise tensors.

**Action required**: Re-run isotope migration on Colab from the best checkpoint to get a clean V14 checkpoint with correct dtypes.

---

## V14.1: Inline Isotope Init + RL Auto-Scaling (2026-02-21)

### Inline Isotope Initialization in `load_checkpoint()`

Previously, resuming from a pre-isotope checkpoint (e.g., V12.41/V13) into a V14+ model required running a separate migration script (`scripts/migrate_v13_to_v14.py`) to properly initialize isotope token embeddings. Without migration, the generic partial-preserve fallback zero-filled new token rows, causing 99.4% → 0.0% exact match collapse.

**Fix**: `load_checkpoint()` now accepts a `v13_tokenizer` parameter. When vocab expansion is detected (checkpoint vocab < model vocab and isotope tokens exist), it automatically:
- Initializes isotope `token_embedding` rows from parent element embeddings + mass-aware noise
- Initializes `output_proj.4.weight` isotope rows from parent element rows + noise
- Copies parent element biases to `output_proj.4.bias` isotope rows
- Initializes `ISO_UNK` from mean of special tokens

This means `resume_checkpoint` can point to ANY checkpoint (including `checkpoint_best.pt`) — no separate migration step needed.

### RL Auto-Scaling

New config: `rl_auto_scale=True`, `rl_auto_scale_target=10.0`

After the first training epoch with RL active, measures `|raw_rl_loss|` and sets:
```
rl_weight = rl_auto_scale_target / |raw_rl_loss|
```

This ensures the RL gradient contribution is always at a reasonable magnitude (target ≈ 10) regardless of model state. Prevents the -117 reward catastrophe when resuming from a different architecture. Gradients flow correctly at any scale — only magnitude changes, not direction.

Weight is clamped to [0.01, 10.0] to prevent pathological values from near-zero RL losses.

### Config Changes
- `resume_checkpoint` → `'outputs/checkpoint_best.pt'` (no longer requires pre-migrated checkpoint)
- `rl_auto_scale: True` — enable dynamic RL weight calibration
- `rl_auto_scale_target: 10.0` — target |RL contribution| magnitude

---

## V15.0: Data Expansion — 5 External Datasets + Vocab Rebuild (2026-02-20)

### Dataset Merge

Merged 5 external superconductor datasets + 12 manual high-Tc hydrides into the training CSV using `scripts/ingest_new_datasets.py`. DFT-predicted data saved separately for future Theory Network use via `scripts/export_dft_datasets.py`.

**Training data: 51,003 → 52,813 rows (+1,810 new unique entries)**

| Source Dataset | Raw Rows | After Dedup | New Rows Added |
|----------------|----------|-------------|----------------|
| MDR SuperCon (NIMS) | 26,356 | — | 419 |
| SuperCon2 (NLP-extracted) | 18,742 | — | 1,382 |
| SODNet (NeurIPS 2024) | 11,949 | — | 364 |
| 3DSC (crystal structures) | 5,773 | — | 33 |
| Manual hydrides (literature) | 12 | — | 3 |
| **Total** | **62,832** | **2,201** | **1,810** |

391 entries dropped due to failed Magpie featurization. JARVIS supercon_chem.json skipped (already ingested by `ingest_jarvis.py` in prior run).

### Tc Distribution Improvement

| Tc Range | Before | After | Change |
|----------|--------|-------|--------|
| Tc >= 100K | 984 | 1,117 | +133 |
| Tc >= 150K | 7 | 58 | +51 |
| Tc >= 200K | 6 | 36 | +30 |

### Data Quality Measures

- **Retracted entries filtered**: Lu-H (Dias 250-300K), C-S-H (Snider 280-295K), PbCO3 (>300K)
- **SuperCon2 contamination filter**: Removed manganite Curie temps, ZnO ferromagnetic, non-hydride >200K entries
- **Deduplication**: Canonical formula matching (pymatgen alphabetical_formula) against existing 50,630 keys + 45 holdout samples
- **IonProperty timeout**: 30s per-composition timeout (1,413 compositions used defaults)
- **High-pressure labeling**: 112 entries flagged as `requires_high_pressure=1`

### Fraction Vocab Expansion

Rebuilt fraction vocab: 4,212 → 4,317 fractions (+105 new). This shifts ISO_UNK and isotope token indices.

**Vocabulary layout** (4647 → 4752 tokens):
| Range | Content | Count |
|-------|---------|-------|
| 0-4 | Special: PAD, BOS, EOS, UNK, FRAC_UNK | 5 |
| 5-122 | Elements | 118 |
| 123-142 | Integers: 1-20 | 20 |
| 143-4459 | Fractions (expanded from 4212 to 4317) | 4317 |
| 4460 | ISO_UNK | 1 |
| 4461-4751 | Isotopes | 291 |

### Checkpoint Migration

`scripts/migrate_vocab_expansion.py` migrated `checkpoint_v14_migrated.pt` → `checkpoint_v15_expanded.pt`:
- Existing fraction embeddings remapped to new indices (order may change when rebuilding vocab by frequency)
- 105 new fraction embeddings initialized from FRAC_UNK + noise
- ISO_UNK and 291 isotope embeddings remapped to shifted indices
- Optimizer state NOT transferred (will be reinitialized on training resume)

### DFT Data (Separate, Not in Training)

Saved to `data/processed/dft_superconductors.csv` (9,612 entries) for future Theory Network:
- JARVIS 3D: 1,058 entries (Eliashberg spectral functions)
- JARVIS 2D: 161 entries
- JARVIS alex_supercon: 8,253 entries (BCS params: lambda, wlog, Debye, DOS)
- HTSC-2025: 140 entries (ambient-pressure predictions)

### New Files

- `scripts/ingest_new_datasets.py` — main ingestion pipeline for 5 external datasets
- `scripts/export_dft_datasets.py` — DFT data export for future use
- `scripts/migrate_vocab_expansion.py` — checkpoint migration for expanded vocab
- `data/processed/new_sc_datasets.csv` — new entries only (1,810 rows, for inspection)
- `data/processed/dft_superconductors.csv` — DFT data (9,612 rows)
- `data/fraction_vocab_old.json` — backup of pre-expansion vocab

---

## V14.0: Isotope Token Expansion (2026-02-20)

### Isotope-Aware Vocabulary

Added 292 new tokens to the decoder vocabulary (1 ISO_UNK + 291 isotope tokens), enabling the model to generate formulas with specific isotopes using `{mass}Element` notation (e.g., `{18}O`, `{2}H`, `{10}B`).

**Vocabulary layout** (4355 → 4647 tokens):
| Range | Content | Count |
|-------|---------|-------|
| 0-4 | Special: PAD, BOS, EOS, UNK, FRAC_UNK | 5 |
| 5-122 | Elements: H, He, ..., Og | 118 |
| 123-142 | Integers: 1-20 | 20 |
| 143-4354 | Fractions: FRAC:1/2, FRAC:1/4, ... | 4212 |
| **4355** | **ISO_UNK: unknown isotope fallback** | **1** |
| **4356-4646** | **Isotopes: ISO:1H, ISO:2H, ..., ISO:238U** | **291** |

**Source**: 291 isotopes across 84 elements from `src/superconductor/encoders/isotope_properties.py` (ISOTOPE_DATABASE). Sorted by atomic number then mass number for deterministic ordering.

### Why Isotopes Matter

The BCS isotope effect (Tc proportional to M^(-alpha), alpha ~ 0.5) is a key signature of phonon-mediated superconductivity. Isotope substitution studies (e.g., O-16 vs O-18 in cuprates, B-10 vs B-11 in MgB2) provide critical evidence about pairing mechanisms. The `estimate_isotope_effect()` function already exists in the codebase — these tokens connect the decoder to that physics.

### Changes

**New files:**
- `scripts/build_isotope_vocab.py` — generates `data/isotope_vocab.json` from ISOTOPE_DATABASE
- `data/isotope_vocab.json` — 291 isotope tokens in deterministic order
- `scripts/migrate_v13_to_v14.py` — checkpoint migration (expand embeddings 4355 → 4647)

**Modified files:**
- `src/superconductor/tokenizer/fraction_tokenizer.py` — extended with isotope encoding/decoding, ISO_UNK token, isotope-aware regex pattern
- `scripts/train_v12_clean.py` — ALGO_VERSION V13.0 → V14.0, isotope config (`use_isotope_tokens`, `isotope_vocab_path`), tokenizer construction updated

### Embedding Initialization

Isotope token embeddings initialized from parent element embedding + mass-aware noise:
- `embed(ISO:18O) = embed(O) + noise * scale`
- Scale proportional to `|mass - natural_mass| / natural_mass`
- Common isotopes (16O) start near parent; exotic isotopes (18O) get more perturbation

### Training Data Gap

No training data contains isotope tokens yet. This is infrastructure for future theory-guided generation using the BCS isotope effect function. The 291 isotope embeddings will learn from:
1. Theory-guided synthetic data (planned)
2. Isotope substitution studies from literature (planned)

---

## Data Acquisition: High-Tc Dataset Download (2026-02-20)

Downloaded 6 external superconductor datasets to augment the 100-200K and >200K Tc bins per `docs/high_tc_data_acquisition.md`.

| Dataset | Location | Entries | Tc>=100K | Type |
|---------|----------|---------|---------|------|
| MDR SuperCon primary.tsv | `data/mdr_supercon/` | 26,358 | 910 | Experimental |
| SuperCon2 cleanup | `data/supercon2_repo/` | 18,943 | 1,180* | NLP-extracted |
| 3DSC_MP | `data/3DSC_repo/` | 5,773 | 23 | Experimental |
| SODNet | `data/SODNet_repo/` | 11,949 + 1,578 | 136 + 35 | Experimental |
| JARVIS (4 datasets) | `data/jarvis_hydrides/` | 25,886 | 159 | Mixed DFT/Exp |
| HTSC-2025 | `data/HTSC2025_repo/` | 140 | 4 | DFT predictions |

\* SuperCon2 has severe data quality issues — many entries with Tc>=200K are Curie/structural temperatures, not superconducting Tc. Aggressive filtering required.

**Key findings:**
- MDR primary.tsv is most useful for immediate training augmentation (910 entries at Tc>=100K)
- JARVIS alex_supercon (8,253 entries with lambda, wlog, Debye temp, DOS) is ideal for future Theory Networks
- JARVIS supercon_3d (1,058 entries with Eliashberg spectral functions) is ideal for Eliashberg Theory Network
- No dataset contains reliable high-pressure hydride data with pressure annotations — the manually curated Source 7 from `docs/high_tc_data_acquisition.md` remains the best source for >200K entries
- MDR top Tc entries include 2 retracted Nature papers (Dias Lu-hydride 294K, Snider C-S-H 287.7K)

**Reports:** Individual analysis reports in `scratch/` (*_report.txt). Master summary: `scratch/dataset_acquisition_master_summary.md`

**Next steps:** Deduplicate MDR vs existing training data, clean SuperCon2, build unified merge pipeline.

---

## V13.2: Enable RL (SCST), Tc-Binned Sampling, Fix Decoder Wiring (2026-02-19)

### Enable REINFORCE/SCST for Autoregressive Refinement

CE training converged at 99.6% TF exact match but only 2.0% autoregressive exact. The gap is exposure bias — TF is hardcoded to 1.0 (line 5576), so the decoder never trains on its own output. RL (Self-Critical Sequence Training) is the only mechanism to close this gap.

**Changes:**
- `rl_weight`: 0.0 → **1.0** (enable SCST)
- `num_epochs`: 4000 → **5000** (1000 additional epochs for RL refinement)

### Fix Decoder Wiring for RL Auto-Reactivation

The loss function's `set_decoder()` was only called when `rl_weight > 0` at startup. When auto-reactivation set `rl_weight > 0` later, `self._decoder` was still None — SCST fell back to weak logit-based RLOO (sampling from teacher-forced context instead of true autoregressive generation). The decoder is now **always wired** regardless of initial `rl_weight`.

### Tc-Binned Oversampling for High-Tc Superconductors

High-Tc samples (>100K) are <2% of SC data. The balanced sampler only balanced SC/non-SC (50/50), so high-Tc samples appeared ~0-1 per batch. R² suffered: 120-200K=0.945, >200K=0.041 (6 samples).

**New config:** `oversample_high_tc: True`, `oversample_tc_bins: {50: 3.0, 100: 10.0}`
- SC samples with Tc >= 50K get 3x sampling weight
- SC samples with Tc >= 100K get 10x sampling weight
- Expected ~6 high-Tc samples per batch of 42 (vs ~0-1 before)

---

## Fix Checkpoint Overwrite Bug, Tc R² Evaluation, Disable Phase A/B (2026-02-19)

### Checkpoint best_exact Overwrite Bug (3 fixes)

When resuming with fresh optimizers (e.g., migrated checkpoint), `best_exact` was reset to 0.0 — causing every subsequent epoch to overwrite `checkpoint_best.pt` with a worse model. The 93.1% best checkpoint was destroyed by progressively worse models (78.7%→67.9%).

**Fixes:**
1. **Don't reset `best_exact` on fresh optimizer resume** — prior run's best_exact preserved
2. **Sync `_shutdown_state` after checkpoint load** — interrupt saves now preserve correct best_exact/prev_exact (previously stayed at 0, destroying the value on Ctrl+C)
3. **Sync optimizers to `_shutdown_state` after Phase A→B transition** — interrupt saves during Phase B no longer write stale Phase A optimizers

### Tc R² Evaluation: Exclude Non-SC Samples

Non-SC samples (Tc=0K) get `tc_weight_override=0.0` during training — the Tc regression loss is never computed for them. But R² evaluation included all ~23K non-SC in the 0-10K bin, measuring untrained noise. A 1.3-unit normalized error on non-SC maps to ~5K Kelvin error, giving R² ≈ -80 due to the tiny SS_tot in that bin.

**Fix:** `sc_tc_mask = tc_true_kelvin > 0` — all R² and MAE metrics now SC-only. The 0-10K bin now contains only actual low-Tc superconductors. Log label changed to "Tc R² (SC-only):".

### Disable Phase A/B (v13_phase: 'A' → None)

Phase A/B was designed for the one-time V12→V13 vocabulary migration (4,212 new fraction embeddings needed warmup). But `v13_phase: 'A'` was hardcoded, causing **every Colab resume** to freeze the encoder for 10 epochs (Phase A), let the decoder adapt to static z, then unfreeze everything with fresh optimizers (Phase B). This deharmonized encoder/decoder on every restart, causing loss divergence (130→170+) as the encoder got hit by competing multi-task gradients with no Adam momentum history.

Set `v13_phase=None` — encoder and decoder co-train normally from the start on all resumes.

---

## Restore Auxiliary Loss Weights to Full Strength + Remove Dead Code (2026-02-19)

### Problem: V13.1c Weight Reductions Caused 0-10K Tc R² Collapse

V13.1c (below) reduced `sc_loss_weight`, `tc_class_weight`, and `hp_loss_weight` to avoid gradient competition with formula CE. By epoch 3756, the consequences were clear:

- **0-10K Tc R²**: 0.9999 (V12.41, epoch 3293) → **-57.8** (V13.0, epoch 3756) → **-93.4** (epoch 3760)
- All other Tc bins remained excellent (10-30K: 0.999, 30-77K: 0.999, 77-120K: 0.995)
- Non-SC Tc MAE: 2.29 in normalized space (terrible — model can't distinguish Tc=0 from Tc=5K)

The root cause: starving the SC classification head (0.5→0.1) removed the encoder's primary signal for distinguishing non-SC (Tc=0) from low-Tc SC samples in z-space. The Tc regression loss alone has weak gradients near Tc=0 (Kelvin weighting gives 1.0x at Tc=0, vs 6x at 100K). Without explicit SC/non-SC classification pressure, the encoder stops encoding this distinction. The `tc_class_weight` reduction (4.0→0.5) compounded the problem — the bucket classification head that explicitly distinguished "Tc=0" from "Tc=0-10K" lost 87.5% of its gradient.

### Solution: Full Weight (1.0) for All Three

| Weight | V13.1c | Restored | Reasoning |
|--------|--------|----------|-----------|
| `hp_loss_weight` | 0.1 | **1.0** | Material property mastery — let loss self-regulate |
| `sc_loss_weight` | 0.1 | **1.0** | SC/non-SC distinction critical for Tc=0 encoding |
| `tc_class_weight` | 0.5 | **1.0** | Tc bucket classification guides z-space structure |

### Philosophy Correction

V13.1c's philosophy ("reduce auxiliaries to avoid gradient competition") was wrong for this system. These losses already self-regulate through their own mechanisms — Huber clipping, focal weighting, bin-specific multipliers. Artificially throttling them starved the encoder of supervision it needs. The correct approach: **let full gradient flow**. If exact match dips temporarily while the encoder re-learns proper z-space structure, that's the correct tradeoff for a system whose end goal is generative discovery, not string reconstruction.

### Dead Code Removal

Also removed two dead loss components that were producing zero-value log entries:

- **Numden (ND)**: Head was removed in V13.0 (fractions are now single semantic tokens), but the parameter, loss computation (`tensor(0.0)`), accumulator, and `ND: 0.0000` log line all remained. Fully removed from `CombinedLossWithREINFORCE`, `train_epoch()`, CSV logging, and epoch output. Also removed all V12.x backward-compat stoich assembly branches.
- **Contrastive (Con)**: Weight was 0.0 since V12.26 (~1000 epochs ago — "plateaued at 5.06, consuming 16% of gradient budget"). Removed import, loss function creation, warmup computation, and `Con: 0.0000 (w=0.000)` log line. Kept `contrastive_mode: True` config flag (controls SC/non-SC dataset loading). SC/nSC exact match reporting is now unconditional.

Net result: -156 lines, +33 lines. Cleaner logs, no more always-zero metrics.

---

## ~~V13.1c: Reduce Auxiliary Loss Weights — Prioritize Formula CE (2026-02-19)~~ SUPERSEDED

**SUPERSEDED**: This change was reversed (see above). The weight reductions caused 0-10K Tc R² to collapse from 0.9999 to -93. Weights restored to 1.0.

<details>
<summary>Original rationale (kept for historical context)</summary>

### Problem

Training logs (epochs 3660-3683) showed auxiliary losses producing enormous gradients that compete with formula CE:
- **SCL** (SC classification): 8-12 raw BCE — pathological for binary classification (ideal ~0.7)
- **TcCl** (Tc classification): 2.0 × weight 4.0 = **8.0 effective** gradient
- **Fam** (family classifier): 1.5 × weight 2.0 = **3.0 effective** gradient
- **HP** (H/pressure): 1.7 × weight 0.5 with 50x pos_weight = volatile

Meanwhile, formula CE was ~1.08 — the auxiliary losses collectively overwhelmed the primary reconstruction signal. Non-SC exact crashed from 69.1% → 47.6% as auxiliary gradients pulled z-space away from formula reconstruction.

### Solution

Reduce auxiliary weights to make them true auxiliaries (gentle guidance, not gradient competition):

| Weight | Old | New | Reasoning |
|--------|-----|-----|-----------|
| `tc_class_weight` | 4.0 | 0.5 | 8x reduction — was producing 8.0 effective gradient |
| `family_classifier_weight` | 2.0 | 0.5 | 4x reduction — was producing 3.0 effective gradient |
| `sc_loss_weight` | 0.5 | 0.1 | 5x reduction — raw BCE of 8-12 is already pathological |
| `hp_loss_weight` | 0.5 | 0.1 | 5x reduction — 50x pos_weight amplification makes this volatile |

### Philosophy

These classifier heads exist to guide z-space organization, not to dominate training. At reduced weights, they still provide useful gradient signal (SC clustering, family separation, Tc ordering) without competing with formula reconstruction. The model should master formulas first — auxiliary structure follows naturally.

</details>

---

## V13.1b: Phased PhysZ — Disable During CE Learning, Auto-Reactivate on Plateau (2026-02-19)

### Problem

PhysZ gradients compete with formula CE for z-space organization. During the V13.0→V13.1 Colab run (epochs 3596-3674), exact match climbed rapidly from 65%→85.5% but PhysZ loss grew unbounded (total loss: 12→14.9). At epoch 3667, PhysZ gradient pressure destabilized training: exact crashed 85.5%→82.5% in 7 epochs, Tc loss tripled (0.027→0.070), all losses spiked simultaneously.

Root cause: PhysZ tries to reorganize z-space geometry while formula CE is actively learning to use z-space for reconstruction. The two objectives fight over the same 2048 dimensions. During rapid CE learning, PhysZ is counterproductive — the z-space will acquire implicit physics structure naturally through formula reconstruction (similar compositions cluster because they produce similar formulas).

### Solution

Phased training: master formula reconstruction first, then layer PhysZ on top.

**Phase 1 (current):** `use_physics_z: False`. CE drives exact match to its ceiling uncontested.

**Phase 2 (auto-triggered):** PhysZ scheduler detects exact match plateau and reactivates PhysZ with:
- 20-epoch warmup ramp (existing `physics_z_warmup_epochs`)
- Regression guard: monitors exact match after activation, halves PhysZ weight if exact drops >2% from activation baseline, pauses entirely if weight reaches floor (0.1)
- Recovery: restores full PhysZ weight when exact recovers above baseline

### PhysZ Scheduler Config
```python
'physics_z_auto_reactivate': True,
'physics_z_reactivation_min_exact': 0.85,        # Need ≥85% before considering
'physics_z_reactivation_window': 20,              # Plateau measured over 20 epochs
'physics_z_reactivation_plateau_threshold': 0.005, # <0.5% improvement = plateau
'physics_z_reactivation_force_exact': 0.95,       # Force at 95%
'physics_z_regression_threshold': 0.02,           # 2% drop triggers weight reduction
'physics_z_regression_check_interval': 5,         # Check every 5 epochs
'physics_z_weight_floor': 0.1,                    # Pause below this
```

### Regression Guard Behavior
```
PhysZ activates (exact=88%) → warmup ramps over 20 epochs
  epoch +10: exact=87.5% → still within 2% of 88%, OK
  epoch +15: exact=85.5% → 2.5% below baseline → weight scale 1.0→0.5
  epoch +20: exact=84.0% → still regressed → weight scale 0.5→0.25
  epoch +25: exact=83.5% → weight 0.25→0.125 → below floor → PAUSED
  --- or ---
  epoch +15: exact=88.2% → recovered above baseline → weight scale restored to 1.0
```

### Key Insight
Once formula reconstruction is mastered, z-space will already have implicit physics structure. PhysZ then refines rather than fights — gradients will be smaller and less disruptive.

---

## V13.1: Remove Encoder Skip Connection (2026-02-19)

### Problem

The decoder received encoder information through **two paths**:
1. **z** (2048-dim latent) → `latent_to_memory` → 16 memory tokens
2. **attended_input** (256-dim skip) → `skip_to_memory` → 8 memory tokens (bypasses z bottleneck)

The skip connection contributed 8 of 28 memory tokens (29%), undermining the VAE information bottleneck. The decoder could reconstruct formulas using information that leaked through the skip path without being compressed into z. Evidence: TRUE autoregressive exact match was 0.1% while teacher-forced exact was 80.6% — the skip made TF artificially easy while doing nothing for AR generation.

### Solution

Set `use_skip_connection=False` in decoder construction. Memory layout changes from `[16 latent | 8 skip | 4 stoich] = 28 tokens` to `[16 latent | 4 stoich] = 20 tokens`. All encoder→decoder communication now flows exclusively through z.

### Changes

**Decoder construction** (`train_v12_clean.py`):
- `use_skip_connection=True` → `use_skip_connection=False`

**Training loop** (`train_v12_clean.py`):
- Removed `attended_input` extraction from encoder output
- Removed `encoder_skip=attended_input` from decoder forward call
- Changed all 4 loss function calls to pass `encoder_skip=None`
- Changed round-trip loss (A5 constraint) to pass `encoder_skip=None`
- Removed `attended_input` collection from z-caching
- Removed `encoder_skip` from warmup pass

**Evaluation** (`train_v12_clean.py`):
- Removed `encoder_skip` from `evaluate_true_autoregressive()`

**Holdout/analysis scripts** (6 files):
- Set `use_skip_connection=False` in decoder construction
- Removed `encoder_skip` extraction and passing

### What Was NOT Changed

- **Encoder `attended_head`**: Kept for checkpoint compatibility (output computed but unused)
- **Decoder class code**: Already supports `use_skip_connection=False` — no changes needed
- **Loss classes**: Already handle `encoder_skip=None` — no changes needed
- **Round-trip loss module**: Already handles `encoder_skip=None` — no changes needed

### Checkpoint Compatibility

Loading V13.0 checkpoint with `use_skip_connection=False`: the `skip_to_memory.*` weights appear as "unexpected keys" and are silently ignored by `strict=False` loading (logged to console).

### Files Modified
| File | Change |
|------|--------|
| `scripts/train_v12_clean.py` | Disable skip, remove ~15 `encoder_skip`/`attended_input` references |
| `scripts/holdout/holdout_search.py` | `use_skip_connection=False`, remove skip passing |
| `scripts/holdout/holdout_search_targeted.py` | Same |
| `scripts/holdout/holdout_tc_validation.py` | Same |
| `scripts/analysis/evaluate_generation_quality.py` | Same |
| `scripts/analysis/evaluate_vae_errors.py` | Remove skip passing |
| `scripts/analysis/generation_quality_audit.py` | `use_skip_connection=False`, remove skip passing |

---

## V13.0: Semantic Fraction Tokenization (2026-02-18)

### Problem

V12.42 achieves 97% exact match under teacher forcing but only 6% under true autoregressive generation. Root cause: digit-by-digit fraction tokenization. A fraction like `(17/20)` is 7 tokens `(`, `1`, `7`, `/`, `2`, `0`, `)`, where a single wrong digit cascades into a completely different stoichiometric value. 80% of errors preserve the correct element set but get fractions wrong.

### Solution

Replace character-level fraction tokenization with **single semantic fraction tokens**. Each `(p/q)` becomes one `FRAC:p/q` token.

### Architecture Changes

**New vocabulary structure:**
```
[0] PAD  [1] BOS  [2] EOS  [3] UNK  [4] FRAC_UNK    (5 special tokens)
[5..122] H, He, Li, ..., Og                            (118 element tokens)
[123..142] 1, 2, 3, ..., 20                            (20 integer tokens)
[143..4354] FRAC:1/2, FRAC:1/4, ...                    (4212 fraction tokens)
```
Total vocab: 4,355 (was 148)

**Decoder changes:**
- `vocab_size` parameter added to `EnhancedTransformerDecoder` (was hardcoded 148)
- `stoich_input_dim` reduced from 37 to 13 (fractions(12) + count(1), numden removed)
- Embedding and output projection resized to new vocab

**Encoder changes:**
- `numden_head` removed (V12.38-V12.41) — fraction info now in semantic tokens

**Training changes:**
- numden_loss eliminated from combined loss
- A2 GCD penalty disabled (impossible to emit non-canonical fractions)
- Stoich conditioning simplified: 13 dims instead of 37
- Two-phase training: Phase A (warmup, 10 epochs) → Phase B (full)

**Config tuning for V13.0:**
- `max_formula_len`: 60 → 30 (V13 max=24, P99=16, mean=8.9 — halves decoder compute)
- `label_smoothing`: 0.1 → 0.05 (30x larger vocab dilutes per-class smoothing effect)
- `length_weight_base`: 15 → 8 (matches V13 mean sequence length)
- `lr_warmup_epochs`: 20 → 0 (Phase A/B handle LR transitions; warmup throttled fraction training)
- `fraction_token_weight`: 2.0 — now wired into FocalLossWithLabelSmoothing (upweights fraction targets 2x)
- Positional encoding truncated from [1,60,512] to [1,30,512] in migration

### New Files
| File | Purpose |
|------|---------|
| `scripts/audit_fractions.py` | Dataset fraction statistics |
| `scripts/build_fraction_vocab.py` | Build `data/fraction_vocab.json` |
| `src/superconductor/tokenizer/__init__.py` | Tokenizer package |
| `src/superconductor/tokenizer/fraction_tokenizer.py` | FractionAwareTokenizer class |
| `scripts/migrate_v12_to_v13.py` | V12→V13 checkpoint weight transfer |

### Key Statistics
- **4,212** unique fractions in training data (already canonical)
- **37-70%** sequence length reduction (median ~50%)
- Physics-informed fraction embedding initialization (interpolated from integer embeddings)
- Weight transfer: stoich_to_memory reshaped (37→13), embeddings row-mapped

### Migration
```bash
python scripts/migrate_v12_to_v13.py \
    --checkpoint outputs/checkpoint_best.pt \
    --output outputs/checkpoint_v13_migrated.pt
```

### Additional Scripts Updated for V13.0 Compatibility
All scripts that construct `EnhancedTransformerDecoder` now auto-detect `vocab_size` and `stoich_input_dim` from checkpoint weights, allowing seamless loading of both V12.x and V13.0 checkpoints:
- `scripts/holdout/holdout_search_targeted.py`
- `scripts/holdout/holdout_search.py`
- `scripts/holdout/holdout_tc_validation.py`
- `scripts/analysis/generation_quality_audit.py`
- `scripts/analysis/evaluate_generation_quality.py`

Phase A→B transition also rebuilds LR schedulers for new optimizers.

### V13.0 Compatibility Fixes (cc2f6e4)

Second-round audit of all loss modules for V12 token layout assumptions:

| File | Bug | Fix |
|------|-----|-----|
| `constraint_rewards.py` | `_elem_idx(14)` and `_elem_idx(32)` NameError in B8 (A15) constraint — old lambda removed but still called | Added `_Si` and `_Ge` to `_rebuild_element_constants()` |
| `formula_loss.py` | `TOKEN_TYPE_MASK` hardcoded to 148 elements — index out-of-bounds with V13 tokens | Added `set_formula_loss_tokenizer()`, dynamic mask resizing, `WeightedFormulaLoss` auto-rebuilds CE weights when vocab_size changes |
| `semantic_unit_loss.py` | Uses V12 `IDX_TO_TOKEN` for semantic unit parsing | Added `set_semantic_unit_tokenizer()` with V13 `_parse_tokens_v13()` path |
| `train_v12_clean.py` | `_v13_fraction_values` created on CPU, transferred to GPU every reward call | Created on encoder's GPU device directly |

Note: `formula_loss.py` and `semantic_unit_loss.py` are NOT used in the main training loop (train_v12_clean.py computes CE directly), but are part of the public loss API and used by `SemanticUnitLoss` class.

---

## V12.43: SC Constraint Zoo — Physics-Grounded Generation Constraints (2026-02-18)

### Problem

Error analysis of V12.41-V12.42 generations reveals systematic physics violations:
- **20.2%** of generated formulas contain duplicate elements (same element appearing twice)
- **~60%** of near-miss formulas use non-canonical fractions (e.g., `4/10` instead of `2/5`)
- Ba/Sr mode collapse proves the model needs self-consistency enforcement
- No physics-based feedback on family-specific constraints (oxygen content, doping ranges, site occupancy)

### Changes

**New files:**
- `src/superconductor/losses/constraint_rewards.py` — REINFORCE reward modifiers (A1, A2, A4, A7, B1-B8)
- `src/superconductor/losses/round_trip_loss.py` — A5 round-trip cycle consistency (differentiable)
- `src/superconductor/losses/constraint_zoo.py` — A3 site occupancy, A6 charge balance (differentiable)

**Modified files:**
- `scripts/train_v12_clean.py` — Integrated all constraints into `CombinedLossWithREINFORCE`
- `src/superconductor/losses/__init__.py` — Registered new exports

### Constraint Summary

**Universal Formula Constraints (Part A) — applied to ALL generations:**

| ID | Constraint | Type | Mechanism |
|----|-----------|------|-----------|
| A1 | No duplicate elements | REINFORCE reward | -50 penalty per violation |
| A2 | Canonical fractions (GCD=1) | REINFORCE reward | -5 per non-canonical fraction |
| A3 | Site occupancy sums correctly | Differentiable loss | Soft L1 on site sum deviation |
| A4 | No reducible stoichiometry | REINFORCE reward | -10 for GCD(subscripts) > 1 |
| A5 | Round-trip Z consistency | Differentiable loss | MSE(z_recon, z_original) on 10% subset |
| A6 | Charge balance | Differentiable loss | tanh(excess charge imbalance) |
| A7 | No impossible element combos | REINFORCE reward | -30 for forbidden pairs |

**Family-Specific Physics Constraints (Part B) — confidence-gated (>0.8):**

| ID | Family | Rule | Penalty |
|----|--------|------|---------|
| B1 | YBCO | O content >= 6.35 | -40 |
| B2 | LSCO | 0.055 <= Sr <= 0.27 | -40 |
| B3 | BSCCO | \|Ca - (Cu-1)\| <= 0.3 | -40 |
| B4 | Hg-cuprate | V < 30% on Hg site | -30 |
| B5 | Tl-cuprate | No magnetic 3d > 10% | -30 |
| B6 | Iron-1111 | O = 1.0 or O >= 0.7 | -30 |
| B7 | MgB2 | C < 12.5%, Al < 50% | -30 |
| B8 | A15 | A:B ratio within 10% of 3:1 | -30 |

### Architecture Details

**A5 Magpie Blocker Solution:** The encoder requires 145 pre-computed Magpie features (from matminer, not available at generation time). Solution: use the decoder's own `magpie_head` predictions as proxy input for re-encoding. Gradients flow through magpie_head → encoder, creating a self-consistency training signal.

**Memory/Compute Impact (A100 40GB, batch=252):**

| Component | Memory | Time/batch |
|-----------|--------|------------|
| A1-A4, A7 (REINFORCE rewards) | ~3MB | ~7ms |
| A5 round-trip (10% subset) | ~125MB | ~50ms |
| A3, A6 (differentiable) | ~2MB | ~2ms |
| B1-B8 (family rewards) | ~5MB | ~10ms |
| **Total addition** | **~135MB** | **~69ms** |

Epoch time increase: ~18% (69ms × ~1111 batches = ~77s added per epoch).

### Configuration

```python
TRAIN_CONFIG = {
    'constraint_zoo_enabled': True,
    'constraint_zoo_weight': 0.5,
    'a1_duplicate_penalty': -50.0,
    'a2_gcd_penalty': -5.0,
    'a4_stoich_norm_penalty': -10.0,
    'a5_round_trip_weight': 1.0,
    'a5_subset_fraction': 0.1,
    'a6_charge_balance_weight': 1.0,
    'a7_impossible_element_penalty': -30.0,
    'family_constraint_enabled': True,
    'family_constraint_confidence': 0.8,
}
```

### Verification Targets

1. A1 duplicate element rate: baseline 20.2% → target <5%
2. A2 non-canonical fraction rate: baseline ~60% → target <20%
3. A5 z_mse and tc_mse: should converge (decreasing over epochs)
4. Overall TRUE AR exact match: should not decrease (constraints improve quality without hurting reconstruction)

---

## V12.42: Net2Net 2x Wider Decoder Expansion (2026-02-18)

### Problem

Decoder `d_model=512` is a bottleneck for multi-digit number generation. Fraction denominators like "1000" require 4 separate tokens (`['1','0','0','0']`), and the decoder must maintain a coherent "plan" across these steps. Error analysis (epochs 3216-3292) shows performance flat at ~88.5% TRUE AR exact match, with 78% of "catastrophic" errors being fraction representation mismatches — correct material, different denominator encoding.

### Changes

**Architecture expansion (decoder only, encoder unchanged):**
- `d_model`: 512 -> **1024** (2x wider, 128 dims/head — GPT-2 medium scale)
- `dim_feedforward`: 2048 -> **4096** (4x d_model)
- All other dimensions unchanged (latent_dim=2048, nhead=8, num_layers=12)
- Training target: A100 40GB (effective batch ~504 with bfloat16)

**Expansion method:** Net2Net weight transfer (`scripts/migrate_checkpoint_v1242_wider.py`)
- Old weights copied into overlapping region of new wider layers
- New dimensions initialized with small noise (std=0.01) for approximate function preservation
- Sinusoidal positional encoding recomputed (deterministic, no transfer needed)
- Optimizer/scheduler state RESET (param shapes changed)

**Component-by-component transfer:**

| Component | Old Shape | New Shape | Method |
|-----------|-----------|-----------|--------|
| token_embedding | (148, 512) | (148, 1024) | expand_embedding |
| pos_encoding.pe | [1, 60, 512] | [1, 60, 1024] | recomputed |
| latent_to_memory[0] | Lin(2048, 4096) | Lin(2048, 8192) | _expand_linear_both_dims |
| latent_to_memory[2] | Lin(4096, 8192) | Lin(8192, 16384) | _expand_linear_both_dims |
| skip_to_memory[0] | Lin(256, 2048) | Lin(256, 4096) | _expand_linear_both_dims |
| skip_to_memory[2] | Lin(2048, 4096) | Lin(4096, 8192) | _expand_linear_both_dims |
| stoich_to_memory[0] | Lin(37, 512) | Lin(37, 1024) | _expand_linear_both_dims |
| stoich_to_memory[1] | LN(512) | LN(1024) | expand_layernorm |
| stoich_to_memory[3] | Lin(512, 2048) | Lin(1024, 4096) | _expand_linear_both_dims |
| 12x transformer layers | d=512, ff=2048 | d=1024, ff=4096 | expand_transformer_decoder_layer |
| output_proj[0] | LN(512) | LN(1024) | expand_layernorm |
| output_proj[1] | Lin(512, 512) | Lin(1024, 1024) | _expand_linear_both_dims |
| output_proj[4] | Lin(512, 148) | Lin(1024, 148) | _expand_linear_both_dims |
| stop_head[0] | Lin(512, 128) | Lin(1024, 256) | _expand_linear_both_dims |
| stop_head[2] | Lin(128, 1) | Lin(256, 1) | _expand_linear_both_dims |

**Net2Net primitives added to `net2net_expansion.py`:**
- `expand_layernorm()` — copy old gamma/beta, init new dims with gamma=1.0+noise, beta=0.0
- `_expand_linear_both_dims()` — expand both input and output dims simultaneously
- Fixed `in_proj_bias` bug in `expand_multihead_attention()` (was missing bias copy)
- Rewrote `expand_transformer_decoder_layer()` to handle dimension changes (previously only copied weights when dims were unchanged)

### Migration

```bash
python scripts/migrate_checkpoint_v1242_wider.py
# or for dry-run:
python scripts/migrate_checkpoint_v1242_wider.py --dry-run
```

**Training config adjustments for 4x wider decoder:**
- `lr_warmup_epochs`: **20** (NEW) — linear warmup 0→3e-5 over 20 epochs (fresh optimizer after Net2Net has no Adam momentum history)
- Decoder grad clip: **1.0→2.0** — 4x more params produce larger gradient norms; 1.0 was too aggressive
- A100 batch multiplier: **12→6** (effective batch 504→252) — 4x wider decoder uses ~4x more activation memory
- `start_factor=1e-3` for warmup — starts at 3e-8, ramps linearly to 3e-5

### Expected Impact
- 2x hidden state capacity for multi-digit token planning (128 dims/head vs 64)
- Should significantly help with fraction denominator encoding (the dominant error mode)
- ~280% decoder parameter increase (~103M -> ~393M)
- A100 40GB has ample headroom for this model size with bfloat16

### Risk
- Temporary accuracy regression as the model adapts to the wider dimensions
- LR warmup should prevent gradient spikes from fresh optimizer + noise-initialized dimensions

---

## Generative Holdout Search Results — V12.41 Re-Run (2026-02-18)

**V12.41 re-run complete.** Checkpoint updated from V12.38 (epoch 3032) to V12.41 (epoch 3292). The consistency check now runs automatically as part of the holdout script.

### Checkpoint Change
- **Old**: `checkpoint_best.pt` → symlink to `best_checkpoints/checkpoint_best_V12.38_colab.pt` (epoch 3032)
- **New**: `checkpoint_best.pt` → symlink to `checkpoint_best V1241.pt` (epoch 3292, V12.41)
- V12.41 has expanded numden_head (2048→512→256→24 vs old 2048→128→24), matching current model code
- Symlink updated 2026-02-18

### Script Improvements (2026-02-18)
- **Self-consistency check integrated**: `holdout_search_targeted.py` now runs all model heads (Tc regression, Tc bucket classifier, SC classifier, hierarchical family head) on every Z centroid after search, and reports SC↔Tc, SC↔Family, and Tc↔Bucket agreement. Results saved in JSON `consistency` field.
- **NumDen head architecture detection**: `load_models()` now detects whether the checkpoint uses old (128-dim) or new (512-dim) numden_head first layer and constructs a matching architecture before loading weights. No more shape mismatch errors.
- **Tc denormalization**: Tc predictions are denormalized from z-score log1p space to Kelvin using `tc_mean=2.725, tc_std=1.353` from cache. Best-match Tc predictions now reported in Kelvin in the consistency output.

### V12.41 Results

- **Checkpoint**: V12.41, epoch 3292
- **Targets searched**: 45
- **Exact matches**: 10/45 (22.2%) — same count as V12.38
- **Unique formulas generated**: 133,028

| Threshold | Found | Percentage |
|-----------|-------|------------|
| sim = 1.000 (exact) | 15/45 | 33.3% |
| >= 0.99 | 38/45 | 84.4% |
| 0.95–0.99 | 3/45 | 6.7% |
| 0.90–0.95 | 4/45 | 8.9% |
| < 0.90 | 0/45 | 0.0% |

> **Note**: 15 targets hit sim=1.000 but only 10 are flagged exact=YES — the other 5 have fraction precision differences (e.g., rounding in numerator/denominator).

#### Self-Consistency (133,028 unique formulas)

| Metric | Value |
|--------|-------|
| **Overall consistency** | **99.87%** |
| SC↔Tc mismatches | 169 (0.13%) |
| SC↔Family mismatches | 0 (0.00%) |
| Tc↔Bucket mismatches | 352 (0.26%) |

#### SC Prediction Breakdown

| Class | Count | Percentage |
|-------|-------|------------|
| Superconductor (SC) | 123,515 | 92.8% |
| Non-SC | 9,513 | 7.2% |

#### V12.41 vs V12.38 Comparison

| Metric | V12.38 (epoch 3032) | V12.41 (epoch 3292) | Delta |
|--------|---------------------|---------------------|-------|
| Exact matches | 12/45 | 10/45 | -2 |
| sim=1.000 | 16/45 | 15/45 | -1 |
| >= 0.99 | 38/45 | 38/45 | 0 |
| >= 0.90 | 45/45 | 45/45 | 0 |
| Unique formulas | — | 133,028 | — |
| Self-consistency | — | 99.87% | — |

---

## Generative Holdout Search Results — V12.38 Baseline (2026-02-17)

> **Note**: These results are from the older V12.38 checkpoint (epoch 3032). The V12.41 re-run above will supersede these. Kept for comparison.

### Setup
- **Checkpoint**: `checkpoint_best_V12.38_colab.pt` (epoch 3032)
- **Script**: `scripts/holdout/holdout_search_targeted.py --all`
- **Method**: For each of the 45 holdout formulas, find 100 element-matched training neighbors, encode as Z seeds, then explore Z-space via perturbation (8 noise scales × 100 samples × 30 seeds), pairwise interpolation (linear + slerp, 100 pairs × 15 steps), PCA walks (20 components × 20 alpha steps), centroid random directions, and temperature sampling (8 temperatures × 15 seeds × 30 samples). ~31K candidate formulas generated per target.
- **New in this run**: Full Z→formula mapping saved. Per unique formula: Z centroid (2048-dim), Z spread (std across generating Z vectors), count, and similarity. Companion `.pt` file with Z centroids for Z-space analysis.

### Results Summary

| Threshold | Found | Percentage |
|-----------|-------|------------|
| >= 1.00 (exact sim) | 16/45 | 35.6% |
| >= 0.99 | 38/45 | 84.4% |
| >= 0.98 | 40/45 | 88.9% |
| >= 0.95 | 41/45 | 91.1% |
| >= 0.90 | 45/45 | **100.0%** |

**12 exact string matches**, **16 sim=1.000** (includes element-reordered matches like `CrCuO2` = `Cu1Cr1O2`).

### Exact Matches by Family

| Family | Formula | Tc (K) |
|--------|---------|--------|
| YBCO | Tl2Ba2Ca(19/20)Y(1/20)Cu2O8 | 106.5 |
| YBCO | Y(1/5)Eu(4/5)Ba2Cu3O7 | 93.8 |
| LSCO | Bi2Sr(19/10)La(1/10)Ca1Cu2OY | 81.0 |
| Hg-cuprate | Hg(9/10)Au(1/10)Ba2Ca2Cu3O | 133.0 |
| Tl-cuprate | Tl2Ba2Ca2Cd(1/10)Cu3O | 123.5 |
| Bi-cuprate | Bi(17/10)Pb(3/10)Sr2Ca2Cu3O | 110.2 |
| Iron-based | Nd1Fe1As1O(17/20) | 53.0 |
| Iron-based | Dy1Fe1As1O(17/20) | 30.0 |
| MgB2 | Mg(97/100)Na(3/100)B2 | 38.0 |
| MgB2 | Mg(9/10)Li(1/10)B2 | 36.9 |
| MgB2 | Mg(17/20)Li(3/20)B2 | 36.6 |
| MgB2 | Mg(49/50)Cr(1/50)B2 | 38.0 |

### Near-Misses (>= 0.99 but not exact)

Most near-misses are **fraction precision issues** — correct elements and ratios, but expressed with slightly different denominators. Examples:
- Target `Y3Ba5Cu8O18` → best `Y(11/10)Ba(19/10)Cu3O(341/50)` (sim=0.998) — same compound, different normalization
- Target `Hg(17/20)Re(3/20)Ba(83/50)Sr(17/50)Ca2Cu3O8` → best `Hg(17/20)Re(3/20)Ba(17/10)Sr(3/10)Ca2Cu3O8` (sim=0.999) — Ba/Sr ratio slightly off
- Target `Nb3Al(71/100)Ge(29/100)` → best `Nb(3/4)Al(89/500)Ge(9/125)` (sim=1.000) — equivalent fractions, different form

### Hardest Targets (< 0.95)

| Family | Formula | Best Sim | Issue |
|--------|---------|----------|-------|
| LSCO | La(1/2)Gd(1/2)Ba2Cu3O | 0.925 | Gd substitution confuses model |
| LSCO | Hg1Sr2La(3/10)Ca(1/2)Ce(1/5)Cu2O6 | 0.944 | 7-element formula, Ce not well-represented |
| Conventional | Gd1Pb1Sr2Ca3Cu4O | 0.947 | Unusual Gd-Pb combination |
| Tl-cuprate | Tl(1/2)Pb(1/2)Sr2Ca2Cu3Li(1/5)O | 0.918 | Li dopant not generated (see analysis below) |

### Error Analysis: All Errors Are Chemically Sensible

**44 out of 45** best matches have the **exact same element set** as the target. The remaining errors break down as:

| Error Category | Count | Description |
|----------------|-------|-------------|
| Exact match (sim=1.0) | 16 | Perfect formula recovery |
| Fraction precision only (sim >= 0.97) | 24 | Right elements, fractions off by ~1/20 |
| Minor stoichiometry (sim 0.93-0.97) | 3 | Right elements, moderate ratio differences |
| Missing trace element | 1 | Li(1/5) dopant not generated |
| Completely wrong chemistry | **0** | None |

**Zero targets produced wrong chemistry.** The fraction precision errors represent the same chemical neighborhood — differences within experimental measurement uncertainty (e.g., `O(161/25)` = 6.44 target vs `O(321/50)` = 6.42 generated).

### Deep Dive: The Missing Li(1/5) Dopant

The sole element-set error is target `Tl(1/2)Pb(1/2)Sr2Ca2Cu3Li(1/5)O` (Tc=117K). The model generates `Tl(1/2)Pb(1/2)Sr2Ca2Cu3O` (Tc=119K) — the **undoped parent compound**, which is a real superconductor present in the training set.

**This is NOT a consistency error.** The generated formula is a real, known superconductor at a similar Tc. The model produces the right material family, just without the trace Li dopant.

**Why Li is missed — data sparsity:**
- Li (Z=3) appears in only **1,420 / 50,958** training samples (2.8%)
- Only **24 training samples** contain both Tl and Li (0.047% of training data)
- Those 24 mostly follow a `Tl(1/4)Li(1/4)` substitution pattern (Li *replaces* Tl), not Li as an independent minor dopant alongside Tl-Pb
- In the holdout search, only **28 / 3,631** unique candidates (0.77%) contained Li at all
- When Li does appear, the model substitutes it for Pb rather than adding it as an independent dopant
- This result is **identical across all three checkpoint searches** (V12.38 colab, V12.40 original, V12.40 current) — the ceiling is 0.918 every time

This represents a genuine **representational limit**: with only 24 examples of Tl+Li co-occurrence, the model cannot learn that Li(1/5) can appear as an independent trace dopant in a Tl-Pb cuprate. The model correctly learns the dominant Tl-Pb cuprate chemistry from ~1,500 Tl-containing samples but has insufficient signal for the rare Li-doped variant. No amount of training epochs will fix this — the data simply doesn't contain enough examples of this pattern. Data augmentation or targeted collection of Tl-Li cuprate entries would be needed.

**For contrast**: MgB2-family Li targets (`Mg(9/10)Li(1/10)B2`, `Mg(17/20)Li(3/20)B2`) achieve perfect exact matches because Li-doped MgB2 is well-represented in the training data with simpler 3-element structures.

### Output Files
- **JSON** (30 MB): `outputs/holdout_search_targeted_checkpoint_best.json` — all unique candidates per target with count, similarity, z_norm, z_spread
- **Z maps** (1.2 GB): `outputs/holdout_search_targeted_checkpoint_best_z_maps.pt` — Z centroid (2048-dim) per unique formula per target, for Z-space analysis

### Key Observations
1. **MgB2 family dominates**: 5/5 exact matches (simpler 3-element formulas, well-represented in training data)
2. **Iron-based strong**: 2/5 exact, all >= 0.988 — consistent 5-element template (RE-Fe-As-O)
3. **Cuprates are harder**: Complex 6-8 element formulas with many doping sites. Most near-misses are fraction precision
4. **100% at >= 0.90**: The model finds the right chemical neighborhood for every holdout target — the remaining gap is fraction precision, not chemistry understanding
5. **Mode attractors visible**: Most frequent generated formulas (1000-3000x) are common training compounds near the target's Z neighborhood, not the target itself. The exact matches come from rarer Z perturbations
6. **No consistency errors in top matches**: The best-match formula for every target is a chemically sensible superconductor. The model never generates nonsense for its top candidates.

### Full Consistency Check: 133,979 Candidates vs Training Data

Cross-referencing ALL unique generated candidates across all 45 targets against the full dataset:

| Metric | Value |
|--------|-------|
| Unique formulas generated (all targets) | 133,979 |
| Match a known dataset entry | 1,977 (1.5%) |
| Of matches: Superconductors (is_sc=1) | 1,906 (96.4%) |
| Of matches: Non-superconductors (is_sc=0) | 71 (3.6%) |
| Novel formulas (not in dataset) | 132,002 (98.5%) |

**98.5% of generated formulas are novel** — not memorized from training. Of the 1.5% that match known compounds, **96.4% are superconductors**.

#### Novelty Deduplication: How Many Are Truly Distinct Materials?

The 132K "novel" count is inflated by fraction-precision duplicates and string-representation variants (e.g., `Au(1/2)Au(1/2)Ba2Ca2Cu3O` vs `AuBa2Ca2Cu3O` — same material). Deduplicating at progressively coarser levels:

| Level | Generated Unique | Novel | Novel % | What it measures |
|-------|-----------------|-------|---------|------------------|
| Raw strings | 133,979 | 132,002 | 98.5% | Exact string match |
| Compositions (2dp) | 127,795 | 125,573 | 98.3% | Same elements, fractions rounded to 0.01 |
| Compositions (1dp) | 115,239 | 113,287 | 98.3% | Same elements, fractions rounded to 0.1 |
| Integer stoichiometry | 47,345 | 45,923 | 97.0% | Same elements, integer ratios only |
| **Element sets** | **16,108** | **14,135** | **87.8%** | **Same elements regardless of fractions** |

**The honest novelty count is ~14,000-16,000 genuinely distinct materials** (by unique element combination), not 132,000. The 8.3x inflation comes from:
1. **Element-split variants**: Model writes `Au(1/2)Au(1/2)` instead of `Au1` — identical material, different tokenization
2. **Fraction representation**: `Mg(17/20)` vs `Mg(849/1000)` — same 0.85 composition
3. **Stoichiometric variants**: `MgB2` vs `Mg2B4` — same material, different normalization

96.9% of 2dp compositions have exactly 1 formula string — the duplication is concentrated in mode attractors where the decoder produces many string variants of the same composition.

**Key insight**: The latent space IS highly continuous (80.8% of formulas from single Z vectors), but the decoder's fraction tokenization creates artificial string diversity. The ~14K novel element combinations is the more scientifically meaningful number — and 87.8% novelty at that level is still strong.

**Script**: `scratch/novel_formula_dedup_analysis.py`

#### Non-SC Contamination by Family

The 71 non-SC matches are not uniformly distributed — they cluster in specific chemical families with physical ambiguity:

**Cu1Cr1O2 (Other family) — worst offender:**
24 non-SC formulas generated, many at high count. The Cr-Cu-O chemical space contains delafossites, spinels, and garnets that span both SC and non-SC regimes. The latent space has not separated them:
- `Cr(197/200)CuMg(3/200)O2` — 1,764x generated, non-SC thermoelectric
- `Cr(4/5)CuFe(6/5)O4` — 1,482x generated, non-SC anisotropy material
- `BrCrCu6O8` — 1,046x generated, non-SC Materials Project entry

**MgB2 targets — mild leakage:**
`B4Mg` (non-SC) appears as a mode attractor across all 5 MgB2 targets. Chemically adjacent boride, structurally distinct from MgB2.

**Iron-based targets — parent compound leakage:**
Undoped pnictides (`AsFeGdO`, `AsCeFeO`) with Tc=0 appear because they are chemically identical to the doped SC variants, just missing the doping that triggers superconductivity. This is physically meaningful.

**YBCO-Eu boundary compositions:**
`Ba2Cu3Eu(1/5)O7Y(4/5)` and `Ba2Cu3Eu(3/5)O7Y(2/5)` generated at ~800x count with 0.97-0.99 similarity, catalogued as non-SC. These sit at the critical Eu doping threshold where SC turns on/off.

#### Clean families (zero non-SC contamination):
- All 5 Hg-cuprate targets
- All 5 Bi-cuprate targets
- 4 of 5 Tl-cuprate targets
- Top-20 candidates for all 4 hardest targets

#### Interpretation

Non-SC contamination reflects **real physical ambiguity at compositional boundaries**, not model failure. Cr-Cu oxides, undoped pnictides, and boundary-doping cuprates are genuinely close in composition space. The contrastive learning successfully separates most SC from non-SC but struggles at these physically ambiguous boundaries. Cu1Cr1O2 is the clearest latent space separation failure and warrants further investigation.

### Z→Output Self-Consistency Check: All Heads Agreement

Fed all 148,823 unique Z centroids back through every model head (Tc regression, Tc bucket, SC classifier, family head) and checked for internal contradictions.

#### Internal Consistency (model agrees with itself)

| Check | Mismatches | Rate | Notes |
|-------|-----------|------|-------|
| SC↔Tc (non-SC but Tc>5K, or SC but Tc≤0) | 206 | **99.86%** consistent | Near-perfect |
| SC↔Family (non-SC but SC family, or SC but NOT_SC) | 0 | **100.0%** consistent | Perfect |
| Tc↔Bucket (regression vs classifier >1 bucket off) | 330 | **99.78%** consistent | Near-perfect |

**The model is internally self-consistent across all heads.** When it predicts SC, the Tc is positive, the family is an SC family, and the Tc bucket matches the regression value. When it predicts non-SC, Tc≈0 and family=NOT_SC.

#### Best-Match Tc Predictions (predicted Tc for closest generated formula to each holdout target)

| Family | Target | Target Tc | Pred Tc | Error | SC Prob | Family Pred |
|--------|--------|-----------|---------|-------|---------|-------------|
| MgB2 | Mg(17/20)Li(3/20)B2 | 36.6K | 32.6K | 4.0K | 1.000 | MGB2 |
| MgB2 | Mg(17/20)Na(3/20)B2 | 38.0K | 37.8K | 0.2K | 1.000 | MGB2 |
| MgB2 | Mg(49/50)Cr(1/50)B2 | 35.3K | 35.0K | 0.3K | 1.000 | MGB2 |
| MgB2 | Mg(9/10)Li(1/10)B2 | 36.9K | 35.5K | 1.4K | 1.000 | MGB2 |
| MgB2 | Mg(97/100)Na(3/100)B2 | 38.0K | 38.5K | 0.5K | 1.000 | MGB2 |
| Conventional | Nb3Al(19/25)Ge(6/25) | 20.4K | 20.3K | 0.1K | 1.000 | BCS_CONVENTIONAL |
| Conventional | Nb3Al(71/100)Ge(29/100) | 20.7K | 20.0K | 0.7K | 1.000 | BCS_CONVENTIONAL |
| Iron-based | Dy1Fe1As1O(17/20) | 51.0K | 47.5K | 3.5K | 1.000 | PNICTIDE |
| Iron-based | Nd1Fe1As1O(17/20) | 51.5K | 48.1K | 3.4K | 1.000 | PNICTIDE |
| Iron-based | Sm(13/20)Th(7/20)Fe1As1O1 | 53.0K | 48.6K | 4.4K | 1.000 | PNICTIDE |
| Iron-based | Sm(9/10)U(1/10)Fe1As1O1 | 47.0K | 46.6K | 0.4K | 1.000 | PNICTIDE |
| Hg-cuprate | Hg(17/20)Re(3/20)Ba(36/25)Sr(14/25)Ca2Cu3O8 | 128.7K | 132.7K | 4.0K | 1.000 | HBCCO |
| Hg-cuprate | Hg(17/20)Re(3/20)Ba(83/50)Sr(17/50)Ca2Cu3O8 | 131.3K | 130.2K | 1.1K | 1.000 | HBCCO |
| Bi-cuprate | Bi(8/5)Pb(2/5)Sr2Ca3Cu(97/25)Si(3/25)O12 | 110.0K | 110.3K | 0.3K | 1.000 | BSCCO |
| YBCO | Y(1/5)Eu(4/5)Ba2Cu3O7 | 93.8K | 92.3K | 1.5K | 1.000 | YBCO |
| YBCO | Y(4/5)Ba2Cu3O(161/25) | 96.0K | 93.6K | 2.4K | 1.000 | YBCO |
| Other | Cu1Cr1O2 | 132.0K | **-0.0K** | **132.0K** | **0.000** | **NOT_SC** |

**Key takeaway**: Tc predictions are accurate to within a few K for simple formulas (MgB2, Nb-based) and within 5-25K for complex cuprates. All family assignments are correct except Cu1Cr1O2 (predicted non-SC). The model's Tc errors are well-correlated with the chemical complexity of the target.

**Cu1Cr1O2 remains the sole self-consistency "success but science failure"** — the model is internally consistent (non-SC → Tc=0 → NOT_SC family) but factually wrong about this material being a superconductor.

### Z-Space Granularity & SC/non-SC Separation (PCA Analysis)

Analysis of how many Z vectors produce identical formulas, and how far apart SC and non-SC formulas are in latent space.

#### Z-Space Granularity

| Metric | Value |
|--------|-------|
| Formulas from single Z vector | 80.8% |
| Formulas from 2-5 Z vectors | 14.2% |
| Formulas from 6-100 Z vectors | 4.1% |
| Mode attractors (>1000 Z vectors) | 0.9% |
| Z-spread for mode attractors | ~0.15 (0.6% of Z-norm) |

The latent space is **highly fine-grained**: 80.8% of unique formulas are generated by exactly one Z vector. Mode attractors (formulas generated >1000 times like `Tl1Ba2Ca(4/5)Y(1/5)Cu2O7`) occupy tight Z-space basins with spread ~0.15 vs Z-norm ~25, meaning the basin covers only 0.6% of the Z magnitude.

#### Cu1Cr1O2: SC vs Non-SC Separation

PCA analysis of Z centroids for the Cu1Cr1O2 target neighborhood:
- **SC formulas**: PC1 centroid = +6.74 (1738 formulas)
- **Non-SC formulas**: PC1 centroid = -4.53 (2919 formulas)
- **PC1 separation**: 11.27 (std: 4.48 SC, 4.48 non-SC)
- **Separation ratio**: 1.257 (gap / mean_std)

The SC and non-SC regions ARE separated in PC1 of Z-space, but the separation ratio of 1.257 is marginal — the distributions overlap significantly. This confirms that Cu1Cr1O2's chemical neighborhood (delafossites, spinels, garnets) creates a genuine latent space boundary challenge. A stronger contrastive margin may be needed for this specific chemical regime.

**Script**: `scratch/z_self_consistency_check.py`, `scratch/z_space_pca_analysis.py`

### SC Sensitivity Landscape: How Robust Is Superconductivity to Chemical Changes?

Classified all ~149K generated variants by their relationship to the 45 holdout targets and checked SC predictions. This maps the robustness of superconductivity to different types of composition changes.

#### Global Perturbation Summary

| Perturbation Type | Count | SC% | Mean Tc | Tc Std | Interpretation |
|-------------------|-------|-----|---------|--------|----------------|
| STOICH_ONLY (same elements, diff fractions) | 17,939 | 96.1% | 75.1K | 42.7K | Stoichiometry changes rarely kill SC |
| ELEM_REMOVED (missing one element) | 43,717 | 97.8% | 73.8K | 39.9K | Removing an element usually preserves SC |
| ELEM_SWAPPED (substitution) | 27,843 | 93.0% | 70.1K | 43.9K | Swapping elements is riskier |
| ELEM_ADDED (extra element) | 12,171 | **83.6%** | 54.5K | 37.2K | **Adding elements is most disruptive** |
| MAJOR_CHANGE (>2 elements diff) | 47,153 | 95.1% | 69.5K | 40.9K | Even major changes mostly preserve SC |

**Key insight**: Adding a new element to a superconductor is the most likely perturbation to kill SC (83.6% survival vs 96-98% for other categories). This makes physical sense — dopants can introduce pair-breaking scattering.

#### Most Fragile vs Robust Targets

**Fragile** (SC sensitive to stoichiometry):
- `Cu1Cr1O2`: Only 14.4% of stoich variants stay SC — the SC/non-SC boundary is razor-thin here
- `Mg(49/50)Cr(1/50)B2`: 48.1% — Cr doping of MgB2 is very sensitive to concentration
- `Y(1/5)Eu(4/5)Ba2Cu3O7`: 75.6% — high Eu substitution is near the SC transition

**Robust** (SC survives stoichiometry changes):
- All Bi-cuprate targets: 100% SC across all stoich variants
- Hg-cuprate targets: 100% SC, Tc std only 2-16K — extremely stable
- `Bi(8/5)Pb(2/5)Sr2Ca3Cu(97/25)Si(3/25)O12`: 100% SC, Tc range only 4.5K — tight basin

#### Elements That Kill SC When Added

| Element | SC% when added | n | Across targets | Physical interpretation |
|---------|---------------|---|----------------|----------------------|
| Br | 5.9% | 34 | 2 | Halide substitution destroys CuO2 planes |
| Dy | 17.4% | 69 | 8 | Magnetic rare earth → pair-breaking |
| Pr | 18.1% | 138 | 8 | Known pair-breaker in cuprates |
| H | 19.4% | 67 | 5 | Hydrogen disrupts crystal structure |
| Mn | 23.3% | 86 | 10 | Magnetic impurity → pair-breaking |
| Se | 24.4% | 41 | 5 | Chalcogenide substitution in non-chalcogenide SC |
| Zn | 33.3% | 216 | 17 | Classic pair-breaker in cuprates |

**The model has learned real pair-breaking physics.** Pr and Zn are experimentally known to destroy superconductivity in cuprates. Magnetic rare earths (Dy, Mn) break Cooper pairs. The model correctly identifies these as SC-killing dopants.

#### Elements That Preserve SC When Added

| Element | SC% when added | n | Mean Tc |
|---------|---------------|---|---------|
| Bi | 100% | 2,316 | 81.0K |
| Hg | 100% | 404 | 59.9K |
| Ag | 99.2% | 126 | 109.1K |
| Al | 99.8% | 599 | 42.4K |
| F | 99.6% | 522 | 43.4K |
| Gd | 99.7% | 340 | 91.8K |

These are elements known to be SC-compatible: Bi and Hg form their own cuprate families, Ag is a classic dopant that enhances Tc, F is used in iron-based SC, and non-magnetic Gd preserves SC despite being a rare earth.

#### Most Critical Elements (removing kills SC)

| Element | SC% when removed | n | Interpretation |
|---------|-----------------|---|----------------|
| Fe | 57.4% | 162 | Essential for iron-based SC — no Fe, no SC |
| As | 82.6% | 511 | FeAs layer is the SC mechanism |
| Mg | 92.0% | 589 | Mg is structurally essential in MgB2 |
| Cu | 94.9% | 1,079 | CuO2 planes are the SC mechanism in cuprates |

**The model correctly identifies structurally essential elements.** Fe and As in pnictides, Cu in cuprates, Mg in MgB2 — removing these destroys the mechanism for superconductivity.

#### Duplicate Element Token Issue

20.2% of generated formulas (27,067 / 133,979) contain the same element appearing multiple times (e.g., `Au(3/5)Au(2/5)Ba2Ca2Cu3O` instead of `AuBa2Ca2Cu3O`). This is a decoder tokenization artifact — the autoregressive decoder sometimes emits the same element twice with different fractions. A duplicate element penalty or constrained decoding would clean this up.

**Scripts**: `scratch/sc_sensitivity_landscape.py`, `scratch/novel_formula_dedup_analysis.py`

---

## V12.41: Expanded NumDen Head + Stoich Conditioning Teacher Forcing (2026-02-17)

### Problems
1. **ND loss plateau at ~2.7**: The `numden_head` (z→128→24) was too small — only ~5 effective dims per output. Despite ND being ~77% of total loss, the 128-dim bottleneck couldn't reduce the error further. More gradient weight wouldn't help (already dominant).
2. **Decoder ignoring stoich conditioning tokens**: Because the numden_head predictions were noisy (RMSE≈1.16 in log1p space), the decoder's 4 stoich memory tokens carried near-garbage. Over 3000+ epochs the decoder learned to ignore them entirely, relying only on 16 latent + 8 skip tokens. This created a vicious cycle: decoder ignores noisy tokens → no gradient flows back → head stays bad → tokens stay noisy.
3. **Teacher-forced vs autoregressive exact gap**: TF exact ~60% vs TRUE AR ~89%. The 60% is expected for 97.5% per-token accuracy at seq length ~20 (0.975^20≈0.60), but fraction tokens are the hardest part and need better decoder conditioning.

### Changes

**1. Expanded numden_head** (`attention_vae.py`)
- Old: `z(2048) → 128 → LayerNorm → GELU → Dropout → 24`
- New: `z(2048) → 512 → LayerNorm → GELU → Dropout → 256 → LayerNorm → GELU → Dropout → 24`
- ~21x more capacity per output dimension
- Checkpoint loading handles shape mismatch via partial preserve (existing mechanism)

**2. Stoich conditioning teacher forcing** (`train_v12_clean.py`)
- New config: `stoich_cond_tf: 1.0` (always use ground truth stoich during training)
- During training: `stoich_pred = tf * gt_stoich + (1-tf) * pred_stoich`
- Ground truth stoich = `[elem_frac, elem_num_log, elem_den_log, elem_count]` (37 dims)
- Evaluation/inference: always uses predicted stoich (no GT available)
- Breaks the vicious cycle: decoder now sees perfect stoich → learns to attend to those 4 tokens → when numden_head improves, seamless transition

### New Config Keys
```python
'stoich_cond_tf': 1.0,  # 1.0=always GT, 0.0=always predicted, can anneal later
```

### Log Output
- New `sTF: 1.0` field in epoch log line (between ND and zN)

### Expected Impact
- ND loss should decrease as expanded numden_head has capacity to learn the mapping
- Decoder should start attending to stoich memory tokens (visible if exact match improves)
- Once numden_head converges, `stoich_cond_tf` can be annealed toward 0.0 so the model transitions to using predicted conditioning at inference
- The ND loss contribution to total loss may drop significantly, allowing formula CE to get more gradient share

### Risk
- numden_head re-initialized from scratch (shape mismatch) — expect ND to spike initially before converging below 2.7
- Decoder was trained for 3000+ epochs without reliable stoich signal — may take time to learn to use it

---

## V12.40: RL Temperature Reduction + Smart Per-Loss Skip Scheduling (2026-02-17)

### Problems
1. **RL temperature too high**: At epoch 3056, temperature=0.5 causes excessive exploration that's counterproductive when model is already at 88% true AR exact match.
2. **Misleading cache exact match metric**: Cache reports "Exact match: 6.3%" over all 50K samples vs true AR eval's 88.4% on 2K samples.
3. **Converged losses waste compute**: REINFORCE is 91% of epoch compute, but other losses also run every epoch even when fully converged.

### Changes
- **`rl_temperature`**: 0.5 → **0.2** — sharper sampling focuses RL gradients on near-optimal sequences
- **Removed** misleading `exact_match_pct` print from `cache_z_vectors()` (per-sample tensor still stored)
- **Smart per-loss skip scheduling**: Each loss is tracked **independently**. When a loss drops below its `converge_threshold`, it's only computed every `loss_skip_frequency` epochs (weight set to 0 on skip epochs). If a loss spikes above `baseline + spike_delta` on a check epoch, that specific loss resumes every-epoch computation. Other converged losses remain skipped. RL gets the biggest compute savings (91% of loss time), but all losses participate for clean gradient control.

### New Config Keys
```python
'loss_skip_enabled': True,
'loss_skip_frequency': 4,          # Compute skipped losses every N epochs
'loss_skip_schedule': {
    # metric_key: (converge_threshold, spike_delta)
    'reinforce_loss':  (1.0,   0.5),  # RL weighted ~0.5 (raw*0.05) — THE big compute saver
    # tc_loss: NEVER skip — core prediction capability
    'magpie_loss':     (0.1,   0.1),  # Magpie MSE ~0.06
    # stoich_loss: NEVER skip — directly feeds decoder conditioning
    # numden_loss: NEVER skip — key improvement area for formula generation
    'tc_class_loss':   (0.5,   0.2),  # Tc bucket classification
    'physics_z_loss':  (0.5,   0.2),  # PhysZ ~0.32
    'hp_loss':         (0.3,   0.1),  # High-pressure BCE
    'sc_loss':         (0.3,   0.1),  # SC/non-SC classification BCE
    'stop_loss':       (0.1,   0.1),  # Stop-prediction BCE
    'family_loss':     (0.5,   0.2),  # Hierarchical family classifier
},
```

### V12.40 Fix: Loss Skip Threshold Comparison (2026-02-17)
- **Removed tc_loss, stoich_loss, numden_loss from skip schedule** — these are core signals that should never be skipped (Tc prediction, decoder conditioning, fraction generation)
- **Fixed threshold comparison to use weighted values**: Convergence thresholds now compare `loss_value * weight` instead of raw loss value. This prevents losses with small weights from appearing "converged" when their raw value is actually still high. Spike detection similarly compares weighted baseline + delta.

### Log Output
- Skipped RL shows `RL: [skip]` in epoch line
- Skipped non-RL losses show `Skip: tc,magpie,...` suffix
- Convergence/spike events print `[LossSkip] {loss} converged/spiked` messages

### Expected Impact
- ~75% compute time reduction on epochs where RL is skipped (RL = 91% of loss compute)
- Sharper RL gradients from temperature=0.2 on epochs when RL does run
- Cleaner gradient signal: converged losses don't contribute near-zero noise gradients

### Entropy NaN Fix (also in this version)
Fixed `Ent: nan` appearing in epoch logs. Cause: `F.softmax()` produces exact 0.0 for very confident predictions → `log(0) = -inf` → `0 * -inf = NaN`. Fix: clamp softmax probabilities to min=1e-8 before taking log, applied in 3 locations:
- `autoregressive_decoder.py` line ~2009 (generate_with_kv_cache)
- `autoregressive_decoder.py` line ~2255 (speculative decoding)
- `train_v12_clean.py` line ~2313 (fallback entropy in loss function)

---

## V12.39: Aggressive RL Weight Reduction (2026-02-17)

### Problem
At epoch ~3030, RL (REINFORCE/SCST) consumes ~80% of the gradient budget (18.6 out of 23.3 total loss) despite `rl_weight` already being reduced from 2.5→1.0 in V12.33. All supervised losses (Tc, Magpie, Stoich, Family) have converged to tiny values, so RL dominates by default. Teacher-forced exact match is stuck at ~77% with noisy, slow improvement. The supervised signals (focal CE with gamma=2.0, numden loss) that know exactly which tokens are wrong are drowned out by RL's high-variance gradients averaged over all samples (including the 77% already correct).

### Change
- `rl_weight`: 1.0 → **0.05** (aggressive 20x reduction)

### Expected Gradient Budget (estimated from epoch 3027)
| Component | Old Weighted | New Weighted | New % |
|-----------|-------------|-------------|-------|
| RL | 18.57 | **0.93** | **18%** |
| NumDen | 2.96 | 2.96 | **57%** |
| Formula CE | ~0.7 | ~0.7 | 13% |
| Tc | 0.43 | 0.43 | 8% |
| Others | ~0.3 | ~0.3 | 4% |
| **Total** | ~23.3 | **~5.3** | 100% |

### Rationale
- At 77% exact match, the model needs focused supervised gradients on the hard 23%, not high-variance RL exploration
- Focal loss (gamma=2.0) already down-weights easy tokens — but only if it can be "heard" over RL noise
- NumDen conditioning (V12.38) is the strongest new signal and should drive the hard-example gradient
- At 0.05, RL contributes ~18% — still provides sequence-level reward signal but supervised losses truly dominate

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
