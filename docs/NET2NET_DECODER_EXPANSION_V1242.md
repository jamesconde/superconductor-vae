# V12.42: Net2Net 2x Wider Decoder Expansion

**Date**: 2026-02-18
**Version**: V12.42
**Source checkpoint**: `outputs/checkpoint_best V1241.pt` (epoch 3292, V12.41)
**Migration script**: `scripts/migrate_checkpoint_v1242_wider.py`
**Commit**: `1fb48f6`

---

## 1. Motivation

### The Fraction Encoding Bottleneck

After 3,292 epochs of training, the model plateaued at ~88.5% TRUE autoregressive exact match. Error analysis revealed a highly specific failure mode: **78% of "catastrophic" errors were fraction representation mismatches** -- the model generated the correct material with the correct elements but encoded the stoichiometric fractions with a different denominator.

For example, a target formula like `La(7/10)Sr(3/10)CuO4` requires the decoder to produce the token sequence `['La', '(', '7', '/', '1', '0', ')', 'Sr', '(', '3', '/', '1', '0', ')', 'Cu', 'O', '4']`. The denominator `10` alone requires two separate tokens (`'1'`, `'0'`), and the decoder must maintain a coherent "plan" across these sequential generation steps -- it needs to "remember" that it's in the middle of writing a denominator, what value it committed to, and how many digits remain.

With `d_model=512` and 8 attention heads, each head operated with only 64 dimensions of hidden state. This is a tight budget for the decoder to simultaneously:
1. Track which element it's currently generating stoichiometry for
2. Remember the fraction numerator it already committed to
3. Plan the denominator digit sequence
4. Attend to the 28 memory tokens (16 latent + 8 skip + 4 stoich)
5. Maintain the causal context of all previously generated tokens

### Why Width, Not Depth

The model already has 12 transformer decoder layers, which provides substantial depth for learning hierarchical representations. The bottleneck was not the number of processing steps but the **information bandwidth at each step** -- the hidden state simply didn't have enough dimensions to maintain all the concurrent "plans" needed for multi-digit number generation.

Width expansion directly increases the per-head capacity (64 -> 128 dims/head), giving each attention head more room to represent position-within-fraction, committed-digit-values, and remaining-digit-count simultaneously.

### Why 2x (Not 20% or 50%)

The original plan was a conservative 20% expansion (d_model 512 -> 616). However, given that:
- Training runs on an **A100 40GB** GPU with ample VRAM headroom
- `d_model=1024` is a well-tested, standard transformer dimension (GPT-2 medium, many production models)
- 128 dims/head is a clean power-of-2 that aligns well with GPU tensor core operations
- The fraction encoding problem is fundamentally a capacity issue, not a data or algorithm issue

...a 2x expansion to `d_model=1024` was chosen to give the decoder substantial room to solve the fraction planning problem without needing another expansion later.

---

## 2. What Is Net2Net?

Net2Net ("Net2Net: Accelerating Learning via Knowledge Transfer", Chen et al., 2016) is a technique for expanding neural network architectures while preserving learned knowledge. Instead of training a larger model from scratch, Net2Net initializes the new, wider model so that it computes approximately the same function as the original model, then continues training from that starting point.

### The Core Principle

When expanding a layer from width W to width W', the first W dimensions of the new layer are copied directly from the old layer. The remaining (W' - W) dimensions are initialized with small random noise. This means:
- The model starts at approximately the same loss as before the expansion
- No catastrophic forgetting -- all 3,292 epochs of learned knowledge are preserved
- The new dimensions can learn to specialize for previously impossible representations
- Training converges much faster than training the larger model from scratch

### Why Not Just Train From Scratch?

Training 3,292 epochs took approximately 2 weeks of GPU time. The model has already learned:
- Element embeddings that capture chemical similarity
- Attention patterns for formula structure (element-number-fraction patterns)
- Latent space structure encoding superconductor families, Tc, and composition
- Memory token representations for cross-attention conditioning

All of this knowledge transfers directly through Net2Net. The new dimensions don't destroy this -- they augment it.

---

## 3. Architecture Changes

### Before (V12.41)

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| dim_feedforward | 2048 |
| nhead | 8 |
| dims/head | 64 |
| num_layers | 12 |
| Decoder params | 102,875,029 (~103M) |
| Encoder params | 8,381,972 (~8.4M) |
| **Total params** | **111,257,001 (~111M)** |

### After (V12.42)

| Parameter | Value |
|-----------|-------|
| d_model | **1024** |
| dim_feedforward | **4096** |
| nhead | 8 |
| dims/head | **128** |
| num_layers | 12 |
| Decoder params | **393,051,797 (~393M)** |
| Encoder params | 8,381,972 (~8.4M) |
| **Total params** | **401,433,769 (~401M)** |

**Decoder parameter increase**: +290M (+282%)
**Encoder**: completely unchanged -- expansion is decoder-isolated.

### What Changed and What Didn't

**Changed (decoder only):**
- Token embedding dimension: 512 -> 1024
- Positional encoding: recomputed for d_model=1024 (sinusoidal, deterministic)
- All memory projection layers (latent_to_memory, skip_to_memory, stoich_to_memory)
- All 12 transformer decoder layers (self-attention, cross-attention, FFN, layer norms)
- Output projection layers
- Stop-prediction head

**Unchanged:**
- Encoder architecture and all encoder weights
- Latent dimension (2048)
- Number of attention heads (8)
- Number of decoder layers (12)
- Vocabulary size (148 tokens)
- Number of memory tokens (16 latent + 8 skip + 4 stoich = 28)
- All training infrastructure (loss functions, curriculum, etc.)

---

## 4. Component-by-Component Expansion

Each decoder component was expanded using a specific Net2Net primitive. The table below shows every layer that changed:

| Component | Old Shape | New Shape | Method | Notes |
|-----------|-----------|-----------|--------|-------|
| `token_embedding` | (148, 512) | (148, 1024) | `expand_embedding` | Old embeddings preserved, new dims noise-init |
| `pos_encoding.pe` | [1, 60, 512] | [1, 60, 1024] | Recomputed | Sinusoidal PE is deterministic, no transfer needed |
| `latent_to_memory[0]` | Lin(2048, 4096) | Lin(2048, 8192) | `_expand_linear_both_dims` | Input (latent_dim) unchanged |
| `latent_to_memory[2]` | Lin(4096, 8192) | Lin(8192, 16384) | `_expand_linear_both_dims` | Both dims expand |
| `skip_to_memory[0]` | Lin(256, 2048) | Lin(256, 4096) | `_expand_linear_both_dims` | Input (skip_dim) unchanged |
| `skip_to_memory[2]` | Lin(2048, 4096) | Lin(4096, 8192) | `_expand_linear_both_dims` | Both dims expand |
| `stoich_to_memory[0]` | Lin(37, 512) | Lin(37, 1024) | `_expand_linear_both_dims` | Input (stoich_dim) unchanged |
| `stoich_to_memory[1]` | LN(512) | LN(1024) | `expand_layernorm` | gamma/beta preserved |
| `stoich_to_memory[3]` | Lin(512, 2048) | Lin(1024, 4096) | `_expand_linear_both_dims` | Both dims expand |
| 12x `self_attn` | embed=512, 8 heads | embed=1024, 8 heads | `expand_multihead_attention` | Q/K/V projections + bias |
| 12x `multihead_attn` | embed=512, 8 heads | embed=1024, 8 heads | `expand_multihead_attention` | Cross-attention |
| 12x `linear1` (FFN) | Lin(512, 2048) | Lin(1024, 4096) | `_expand_linear_both_dims` | FFN up-projection |
| 12x `linear2` (FFN) | Lin(2048, 512) | Lin(4096, 1024) | `_expand_linear_both_dims` | FFN down-projection |
| 12x `norm1/norm2/norm3` | LN(512) | LN(1024) | `expand_layernorm` | 3 norms per layer x 12 layers |
| `output_proj[0]` | LN(512) | LN(1024) | `expand_layernorm` | |
| `output_proj[1]` | Lin(512, 512) | Lin(1024, 1024) | `_expand_linear_both_dims` | |
| `output_proj[4]` | Lin(512, 148) | Lin(1024, 148) | `_expand_linear_both_dims` | Output (vocab) unchanged |
| `stop_head[0]` | Lin(512, 128) | Lin(1024, 256) | `_expand_linear_both_dims` | d_model//4 hidden |
| `stop_head[2]` | Lin(128, 1) | Lin(256, 1) | `_expand_linear_both_dims` | Output (logit) unchanged |

---

## 5. Net2Net Primitives Implemented

The expansion required four new/improved utilities in `src/superconductor/models/net2net_expansion.py`:

### 5a. `expand_layernorm(layer, new_features, noise_std=0.01)`

Expands a `nn.LayerNorm` to a wider dimension. Copies old gamma (weight) and beta (bias) for overlapping dimensions. New dimensions initialized with `gamma = 1.0 + noise` (near-identity) and `beta = 0.0`, so the new dimensions pass through approximately unchanged initially.

### 5b. `_expand_linear_both_dims(layer, new_in, new_out, noise_std=0.01)`

Expands a `nn.Linear` layer in **both** input and output dimensions simultaneously. This was the critical missing primitive -- the existing `expand_linear_wider` only handled output expansion with a paired next-layer input expansion. For FFN layers where `d_model` (input) and `dim_feedforward` (output) both change, we need a single operation that:
1. Copies the overlapping region `weight[:old_out, :old_in]`
2. Noise-initializes new output rows `weight[old_out:, :old_in]`
3. Noise-initializes new input columns `weight[:old_out, old_in:]`
4. Noise-initializes the corner `weight[old_out:, old_in:]`
5. Preserves bias for old dimensions, zeros for new

### 5c. `in_proj_bias` bug fix in `expand_multihead_attention()`

The existing `expand_multihead_attention()` copied `in_proj_weight` (the concatenated Q/K/V projection matrix) but **did not copy `in_proj_bias`**. This left the learned bias randomly initialized after expansion, which would have corrupted the attention computation. The fix adds a bias copy block that preserves the old `3*old_embed_dim` bias values and zeros the new dimensions.

### 5d. Rewritten `expand_transformer_decoder_layer()`

The original implementation only copied weights when dimensions were unchanged (`if old_d_model == new_d_model`) -- making it useless for our case. The rewrite uses `expand_multihead_attention` for both self-attention and cross-attention, `_expand_linear_both_dims` for the FFN layers, and `expand_layernorm` for all three layer norms. It also correctly passes `activation='gelu'` and `norm_first=True` to match the decoder's configuration.

### 5e. `expand_enhanced_decoder(old_decoder, new_d_model, new_dim_feedforward)`

High-level function that orchestrates the full expansion. Creates a new `EnhancedTransformerDecoder` with the expanded config, then transfers weights component-by-component using the primitives above. Also recomputes sinusoidal positional encoding (deterministic, so no transfer needed). Prints a detailed log of every component expanded with old/new shapes.

---

## 6. Training Configuration Adjustments

Three training hyperparameters were adjusted to accommodate the 4x larger decoder:

### 6a. Learning Rate Warmup (NEW)

```python
'lr_warmup_epochs': 20  # Linear warmup from ~0 to 3e-5 over 20 epochs
```

**Why**: The optimizer state (Adam's `exp_avg` and `exp_avg_sq` momentum buffers) was reset because parameter shapes changed. On the first step after migration, Adam without history essentially becomes SGD -- and the noise-initialized new dimensions will produce large, uncorrelated gradients. A 20-epoch linear warmup from `3e-8` to `3e-5` gives the optimizer time to build accurate momentum estimates before applying full learning rate.

**Implementation**: The cosine annealing scheduler is wrapped in a `torch.optim.lr_scheduler.SequentialLR` with a `LinearLR` warmup phase. On future checkpoint resumes, the scheduler state records that warmup is already complete, so it only applies once.

### 6b. Decoder Gradient Clipping: 1.0 -> 2.0

```python
dec_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0)
```

**Why**: With ~4x more parameters, the total gradient norm is proportionally larger. Clipping at 1.0 would be overly aggressive and slow down early adaptation. The encoder clip remains at 1.0 since the encoder is unchanged. The NaN gradient guard (V12.14) still protects against catastrophic gradient events.

### 6c. A100 Batch Multiplier: 12 -> 6

```python
batch_size_multiplier = 6.0  # 42 * 6 = 252 effective batch (was 504)
```

**Why**: The 4x wider decoder uses approximately 4x more activation memory per sample. At the old multiplier of 12, the effective batch of 504 would likely OOM on the A100 40GB. Reducing to 6 (effective batch 252) provides safety margin while still maintaining a large batch for smooth gradients. This can be tuned upward after observing actual memory usage during the first training run.

---

## 7. Migration Process

### Running the Migration

```bash
cd /home/james/superconductor-vae
conda activate recursivemenn-py311

# Dry run (verifies everything without saving):
python scripts/migrate_checkpoint_v1242_wider.py --dry-run

# Actual migration:
python scripts/migrate_checkpoint_v1242_wider.py
```

### What the Migration Script Does

1. **Loads checkpoint** from `outputs/checkpoint_best V1241.pt` (1.3GB, epoch 3292)
2. **Strips `_orig_mod.` prefixes** from torch.compile artifacts in state dict keys
3. **Creates old-architecture decoder** (d_model=512) and loads old weights
4. **Applies `expand_enhanced_decoder()`** -- creates new d_model=1024 decoder with transferred weights
5. **Verifies forward pass** -- runs dummy data through the expanded decoder to confirm no shape errors
6. **Builds new checkpoint**: encoder state (unchanged) + expanded decoder state
7. **Carries over** entropy manager, prev_exact, best_exact, manifest, theory/physics loss states
8. **Resets** optimizer and scheduler states (param shapes changed, old states invalid)
9. **Backs up** existing `checkpoint_best.pt` to `checkpoint_best.pt.bak`
10. **Saves** expanded checkpoint as `outputs/checkpoint_best.pt` (1.53GB)

### CLI Options

```
--checkpoint PATH    Source checkpoint (default: outputs/checkpoint_best V1241.pt)
--output PATH        Output checkpoint (default: outputs/checkpoint_best.pt)
--dry-run            Print expansion plan without saving
--noise-std FLOAT    Noise std for new weights (default: 0.01)
```

---

## 8. Memory Estimation (A100 40GB)

| Component | Size |
|-----------|------|
| Decoder params (bfloat16) | 393M x 2B = ~786 MB |
| Encoder params (bfloat16) | 8.4M x 2B = ~17 MB |
| Adam optimizer state (2 moments, fp32) | 393M x 8B = ~3.1 GB |
| Gradients (bfloat16) | 393M x 2B = ~786 MB |
| **Model subtotal** | **~4.7 GB** |
| Activations (batch=252, seq=60, 12 layers) | ~8-12 GB |
| REINFORCE sampling (4 samples, sequential) | ~3-5 GB extra |
| **Peak estimated** | **~16-22 GB** |
| **A100 headroom** | **~18-24 GB** |

The 40GB A100 has comfortable headroom. If memory usage is lower than estimated, the batch multiplier can be increased from 6 back toward 12 in a future version.

---

## 9. Expected Training Behavior

### First ~20 Epochs (Warmup Phase)
- LR ramps from 3e-8 to 3e-5
- Loss will likely spike initially as noise-initialized dimensions produce imperfect outputs
- Adam builds momentum estimates for the new parameter shapes
- Expect accuracy to temporarily drop below the pre-expansion 88.5%

### Epochs 20-100 (Recovery Phase)
- New dimensions begin specializing
- The model should recover to pre-expansion accuracy relatively quickly
- The noise-initialized weights are small (std=0.01), so the function is approximately preserved -- recovery should be faster than training from scratch

### Epochs 100+ (Improvement Phase)
- With 2x hidden state capacity, the decoder has room to solve the fraction planning problem
- Watch for improvement in fraction-specific metrics (denominator exact match rate)
- If the model plateaus again at ~88.5%, the bottleneck is elsewhere (data, loss function, or encoder capacity)

### What to Monitor
- **TRUE AR exact match**: The primary metric. Should recover to 88.5% within ~50-100 epochs, then hopefully surpass it
- **Fraction error rate**: Track what percentage of errors are still fraction mismatches vs. other error types
- **Gradient norms**: Watch `dec_grad_norm` -- if it's consistently hitting the 2.0 clip, consider raising it further
- **Memory usage**: Check GPU memory on first epoch to see if batch multiplier can be increased
- **Loss composition**: The formula CE loss should decrease as the decoder gets better at fractions

---

## 10. Files Modified

| File | Changes |
|------|---------|
| `src/superconductor/models/net2net_expansion.py` | +5 new functions/fixes, `expand_enhanced_decoder()` |
| `scripts/train_v12_clean.py` | d_model=1024, dim_ff=4096, V12.42, warmup, grad clip |
| `src/superconductor/utils/env_config.py` | A100 batch multiplier 12->6 |
| `docs/TRAINING_RECORDS.md` | V12.42 changelog entry |
| `scripts/migrate_checkpoint_v1242_wider.py` | **NEW** -- standalone migration script |

---

## 11. Rollback Plan

If the expansion causes issues:

1. The original V12.41 checkpoint is preserved at `outputs/checkpoint_best V1241.pt` (untouched)
2. The pre-migration `checkpoint_best.pt` is backed up at `outputs/checkpoint_best.pt.bak`
3. To rollback: restore `checkpoint_best.pt.bak` to `checkpoint_best.pt` and revert MODEL_CONFIG in `train_v12_clean.py` to d_model=512, dim_feedforward=2048

---

## 12. References

- Chen, T., Goodfellow, I., & Shlens, J. (2016). *Net2Net: Accelerating Learning via Knowledge Transfer*. ICLR 2016. [arXiv:1511.05641](https://arxiv.org/abs/1511.05641)
- The Net2Net wider expansion preserves the function: `f_new(x) ≈ f_old(x)` at initialization, with new capacity available for learning. This is the key insight that makes architecture expansion practical without retraining from scratch.
