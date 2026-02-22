# V14.3: Token Type Classifier Head + Enriched Decoder Memory

Design document for the token type classification system added in V14.3, which targets the dominant error mode in autoregressive generation.

---

## Problem

The superconductor formula VAE achieves 95.6% teacher-forced (TF) exact match but only 7.5% TRUE autoregressive (AR) exact match. Error analysis reveals **50% of all AR errors are type confusion** — the model predicts an element where a fraction should go, a fraction where an integer should go, etc.

The formula grammar is deterministic:
```
[Element, Stoich, Element, Stoich, ..., EOS]
```
where Stoich is either an integer (1-20) or a FRAC:p/q token. The model knows the pattern during teacher-forced training but loses track during free-running AR generation.

## Literature Precedent

This approach has strong precedent in NLP and molecular generation:

- **POS-Guided Softmax** (Yang et al., COLING 2022) — predicts POS tag then constrains vocab to that POS
- **Class-Factored Softmax** (Grave et al., ICML 2017) — hierarchical type-then-token decomposition
- **Grammar VAE** (Kusner et al., ICML 2017) — grammar-constrained molecular generation
- **F2-Softmax** (Choi et al., EMNLP 2020) — frequency-class factored softmax

The core idea: predict the **class** of the next token first, then constrain the vocabulary to only tokens of that class. Soft guidance during training (auxiliary loss), hard constraints at inference.

## Architecture Gap (Pre-V14.3)

The transformer decoder received via cross-attention memory:
- **16 latent memory tokens** (from z=2048)
- **4 stoich memory tokens** (from stoich_pred=[batch, 13])
- Total: **20 memory tokens**

It did NOT receive: tc_pred, magpie_pred, hp_pred, sc_pred, family classification, tc_class_logits, or competence. These all contain useful information about what kind of material this is, which directly informs what token types to expect (e.g., a high-Tc cuprate will have specific elements and stoichiometry patterns).

## Solution: Two-Part Architecture

### Part 1: Token Type Head (330K parameters)

A deep classifier on the transformer decoder hidden state that predicts the token type at each position.

**Token type classes (5 types):**

| Class | ID | Token ID Range | Count | Description |
|-------|-----|----------------|-------|-------------|
| ELEMENT | 0 | 5-122 | 118 | Chemical elements (H, He, ..., Og) |
| INTEGER | 1 | 123-142 | 20 | Stoichiometry integers 1-20 |
| FRACTION | 2 | 143-4354+ | ~4212 | FRAC:p/q semantic fraction tokens |
| SPECIAL | 3 | 0-4, 4355+ | ~297 | PAD/BOS/UNK/FRAC_UNK + ISO_UNK + isotopes |
| EOS | 4 | 2 | 1 | End of sequence (carved out from SPECIAL for importance) |

EOS is carved out as its own class because knowing when to stop is critical — collapsing it into SPECIAL would dilute the stop signal.

**Architecture (deep, 8 layers):**
```
Transformer Decoder Hidden State [batch, seq_len, d_model]
        |
   LayerNorm(d_model)                    # Normalize transformer output
   Linear(d_model, d_model)              # Full-width projection (no info loss)
   GELU
   Dropout(p)
   Linear(d_model, d_model // 4)         # Compress: 512→128 (or 1024→256)
   GELU
   Dropout(p)
   Linear(d_model // 4, 5)              # Final classification
        |
   type_logits [batch, seq_len, 5]
```

**Why deep?** This head applies **hard masking** at inference — a wrong type prediction completely blocks the correct token (sets its logit to -inf). This is categorically different from the stop_head (which only provides soft additive boosts). The token_type_head must be at least as capable as the output_proj it's constraining, hence the deep architecture with LayerNorm and Dropout matching output_proj's structure.

**Training:** Auxiliary cross-entropy loss on type labels, weighted at 0.1x of the formula CE loss. Type labels are computed from target token IDs using a cached lookup table (LUT) in the tokenizer for efficient vectorized conversion.

**Inference:** Hard mask over vocab logits:
```python
# Predict token type from decoder hidden state
type_logit = self.token_type_head(output)     # [batch, 5]
predicted_type = type_logit.argmax(dim=-1)     # [batch]

# Look up which tokens are valid for this type
valid_mask = type_masks[predicted_type]         # [batch, vocab_size]

# Block all invalid tokens
logits = logits.masked_fill(~valid_mask, float('-inf'))
```

### Part 2: Enriched Decoder Memory (1.2M parameters)

Feed encoder head predictions into the decoder as 4 additional cross-attention memory tokens.

**Input (10 dimensions):**
```python
heads_input = cat([
    tc_pred.unsqueeze(-1),             # 1  — predicted Tc
    sc_pred.unsqueeze(-1),             # 1  — SC/non-SC logit
    hp_pred.unsqueeze(-1),             # 1  — high-pressure logit
    tc_class_logits,                   # 5  — Tc bucket distribution
    competence.unsqueeze(-1),          # 1  — competence score
    element_count_pred.unsqueeze(-1),  # 1  — predicted # elements
], dim=-1)  # Total: 10 dims
```

**Architecture (deep, 6 layers with 3 linear stages):**
```python
self.heads_to_memory = nn.Sequential(
    nn.Linear(10, d_model // 2),           # 10 → 256 (initial expansion)
    nn.LayerNorm(d_model // 2),            # Stabilize after large expansion
    nn.GELU(),
    nn.Linear(d_model // 2, d_model),      # 256 → 512
    nn.GELU(),
    nn.Linear(d_model, d_model * 4),       # 512 → 2048
)
# Reshaped to 4 × d_model memory tokens
```

The 200x expansion (10 → 2048) is done in 3 gradual stages to prevent information loss from a single massive linear layer.

**Updated memory layout:**
```
[16 latent | 4 stoich | 4 heads] = 24 memory tokens (was 20)
```

**Gradient isolation:** All head values are `.detach()`ed before passing to the decoder. The decoder loss does not flow gradients back through the encoder heads — the heads are treated as conditioning signals, not as part of the decoder's optimization path.

**Why this helps:** The decoder now knows "this is a high-Tc cuprate SC" vs "this is a low-Tc BCS conventional" vs "this is a non-superconductor", which strongly constrains what elements and stoichiometry patterns to expect.

## Tokenizer Additions

Added to `FractionAwareTokenizer`:

- **Constants:** `TOKEN_TYPE_ELEMENT=0`, `TOKEN_TYPE_INTEGER=1`, `TOKEN_TYPE_FRACTION=2`, `TOKEN_TYPE_SPECIAL=3`, `TOKEN_TYPE_EOS=4`, `N_TOKEN_TYPES=5`
- **`get_token_type(token_id) -> int`** — Maps a single token ID to its type class. Checks EOS first (carved out), then element/integer/fraction ranges, fallback SPECIAL.
- **`get_type_masks(device) -> Tensor`** — Returns `[5, vocab_size]` boolean mask tensor. Precomputed once, cached. Each row is True for tokens belonging to that type.
- **`compute_token_type_targets(token_ids) -> Tensor`** — Vectorized LUT-based conversion of token IDs to type labels. Cached on first call as `self._token_type_lut`.

## Files Modified

### Core changes:
- **`src/superconductor/models/autoregressive_decoder.py`** — Added `token_type_head`, `heads_to_memory` modules. Updated `_create_memory()`, `forward()` (now returns 4-tuple with `type_logits`), `generate_with_kv_cache()`, `sample_for_reinforce()`, `speculative_sample_for_reinforce()`, legacy `generate()`.
- **`src/superconductor/tokenizer/fraction_tokenizer.py`** — Added type constants and 3 new methods (above).
- **`scripts/train_v12_clean.py`** — Type loss computation, `heads_pred` assembly, RLOO/SCST/Z-cache paths all pass `heads_pred` for memory consistency.

### Backward compatibility fixes (decoder now returns 4-tuple):
- `scripts/train.py` — `*_extra` pattern
- `scripts/migrate_checkpoint_v1242_wider.py` — `*_extra` pattern
- `scripts/migrate_vocab_expansion.py` — Comment noting new layers are handled
- `src/superconductor/training/soft_token_sampling.py` — `result[0], result[1]` and `*_extra`
- `scratch/debug_cuda_test.py`, `test_real_data.py`, `test_amp_transition.py`, `test_epoch_boundary.py`, `test_full_volume.py` — `*_extra` pattern

## Configuration

```python
# In TRAIN_CONFIG (scripts/train_v12_clean.py):
'token_type_loss_weight': 0.1,   # Auxiliary CE loss weight (0.0 to disable)
'use_type_masking_ar': False,    # Hard type masking during AR generation
'use_heads_memory': True,        # Feed encoder heads into decoder memory (24 vs 20 tokens)
```

## Checkpoint Compatibility

New layers (`token_type_head`, `heads_to_memory`) are missing from pre-V14.3 checkpoints. The `load_checkpoint()` function uses `strict=False` — new layers initialize randomly, existing layers load from checkpoint. No migration script needed.

## Rollout Strategy

1. **V14.3**: Train with type loss + heads memory (training signal only, `use_type_masking_ar=False`)
2. **After ~1 epoch**: Check type head accuracy — should be >90% on TF mode (the grammar pattern is highly learnable)
3. **Enable masking**: Set `use_type_masking_ar=True` once type head is accurate
4. **Monitor**: Type head accuracy should correlate with AR exact match improvement

## Expected Impact

- **Type confusion elimination**: 50% of AR errors are type confusion. Hard masking should eliminate these entirely once the type head is accurate.
- **AR exact match jump**: From ~7.5% to potentially 15%+ (removing half the error surface)
- **Richer decoder context**: Heads memory gives the decoder material-type awareness, improving element and stoichiometry selection even beyond type masking

## Verification Checklist

1. Decoder forward pass produces `type_logits` alongside vocab logits (4-tuple return)
2. Type accuracy >90% after 1 epoch of TF training
3. With masking enabled, 0% cross-type errors in AR generation
4. Memory layout: 24 tokens with `heads_pred`, 20 without (backward compatible)
5. All code paths (RLOO, SCST, Z-cache, eval, legacy generate) pass `heads_pred` consistently
