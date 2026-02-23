# Error Analysis: Epochs 4208–4228

Generated 2026-02-23 from error reports in `outputs/error reports/`.

---

## Executive Summary

The model achieves **98.7% exact match on training data** but only **~2% on validation**. This is not a failure to learn — it's a failure to **generalize**. The model has memorized 46K training formulas but cannot reconstruct unseen formulas.

Two root causes are identified:

1. **The decoder is massively overparameterized** — 109M params (93.8% of total) for 46K training samples = 2,337 decoder params per sample. This gives the decoder enough capacity to memorize every training formula without learning generalizable composition rules.

2. **Type masking is disabled** — The V14.3 token type classifier head is trained (`token_type_loss_weight=0.1`) but **`use_type_masking_ar=False`**. Hard vocab masking during AR generation is OFF. As a result, **56.9% of all validation errors are type confusions** (element where fraction should be, fraction where integer should be, etc.) that masking would eliminate.

---

## Key Metrics (Epochs 4208–4228)

| Metric | E4208 | E4212 | E4216 | E4220 | E4228 | Trend |
|--------|-------|-------|-------|-------|-------|-------|
| **Exact Match %** | 1.93 | 2.28 | 2.58 | 2.58 | 2.08 | Flat (noise) |
| **Avg Errors/Failed** | 4.60 | 4.56 | 4.52 | 4.50 | 4.68 | Flat |
| **Tc R² (Kelvin)** | 0.9992 | 0.9995 | 0.9999 | 0.9999 | 0.9997 | Converged |
| **Tc MAE (Kelvin)** | 0.24 | 0.28 | 0.18 | 0.21 | 0.23 | Converged |
| **Magpie MSE** | 0.0700 | 0.0694 | 0.0679 | 0.0664 | 0.0657 | Slowly improving |
| **Stoich MSE** | 0.0084 | 0.0077 | 0.0080 | 0.0078 | 0.0076 | Converged |
| **Family Coarse Acc** | 100% | 100% | 99.9% | 100% | 100% | Converged |

All encoder heads are essentially converged. The bottleneck is entirely the autoregressive formula decoder.

---

## Parameter Analysis: Decoder Overparameterization

```
Encoder (FullMaterialsVAE):         7,184,630 params  (6.2%)
Decoder (EnhancedTransformerDecoder): 109,002,029 params (93.8%)
Total:                              116,186,659 params

Training samples: 46,645
Decoder params per training sample: 2,337
```

### Decoder Breakdown

| Component | Params | % of Decoder |
|-----------|--------|-------------|
| `transformer_decoder` (12 layers) | 50,448,384 | 46.3% |
| `latent_to_memory` (z→16 tokens) | 41,955,328 | 38.5% |
| `skip_to_memory` (skip→8 tokens) | 8,919,040 | 8.2% |
| `output_proj` (→vocab logits) | 2,647,591 | 2.4% |
| `token_embedding` (4647 vocab) | 2,379,264 | 2.2% |
| `heads_to_memory` (V14.3) | 1,185,536 | 1.1% |
| `stoich_to_memory` | 1,071,104 | 1.0% |
| `token_type_head` (V14.3) | 329,989 | 0.3% |
| `stop_head` | 65,793 | 0.1% |

The `latent_to_memory` projection alone is 42M params — nearly 900 params per training sample just to convert z into memory tokens. This is the strongest candidate for capacity reduction.

---

## Type Head Analysis: Trained But Not Used

**Config**: `token_type_loss_weight=0.1` (type head IS trained), `use_type_masking_ar=False` (masking is OFF during generation).

### Type Confusion Breakdown (Epoch 4228)

Of 9,237 total token mismatches:
- **Same-type errors**: 3,977 (43.1%) — correct token type, wrong specific token
- **Type confusion errors**: 5,260 (56.9%) — wrong token type entirely

### Type Confusion Matrix

| Error | Count | % | Description |
|-------|-------|---|-------------|
| elem→elem | 2,009 | 21.7% | Same type (wrong element) |
| frac→frac | 1,461 | 15.8% | Same type (wrong fraction) |
| int→frac | 896 | 9.7% | **TYPE CONFUSION** — especially `2→FRAC:1/2` (34×) |
| elem→EOS | 799 | 8.6% | **TYPE CONFUSION** — early termination |
| elem→frac | 644 | 7.0% | **TYPE CONFUSION** — fraction where element belongs |
| int→elem | 616 | 6.7% | **TYPE CONFUSION** |
| int→EOS | 554 | 6.0% | **TYPE CONFUSION** — early termination |
| int→int | 507 | 5.5% | Same type (wrong integer) |
| elem→int | 417 | 4.5% | **TYPE CONFUSION** |
| frac→int | 318 | 3.4% | **TYPE CONFUSION** |
| frac→EOS | 268 | 2.9% | **TYPE CONFUSION** — early termination |
| frac→elem | 198 | 2.1% | **TYPE CONFUSION** |
| EOS→int | 156 | 1.7% | **TYPE CONFUSION** — late termination |
| EOS→frac | 82 | 0.9% | **TYPE CONFUSION** |
| EOS→elem | 65 | 0.7% | **TYPE CONFUSION** |
| other | 247 | 2.7% | Various special token confusions |

**If type masking were enabled and 100% accurate, it would eliminate 5,260 of 9,237 errors (57%)**, cutting the average errors per failed sample roughly in half.

---

## Error Pattern Details

### Early Termination (85% of EOS errors)

The model terminates sequences too early. Across epochs:
- Early stops: 1,264–1,797 per epoch
- Late stops: 276–361 per epoch
- Exact-match avg length: ~9.5 tokens
- Error avg length: ~7.8 tokens

### Element Confusions (Chemically Related)

The most confused elements occupy similar crystallographic sites in superconductors:
- **Cu ↔ Ba, Ca, La, Sr, O** — perovskite A/B site confusion
- **Ba ↔ Ca, La, Y, Pr** — alkaline earth / rare earth
- **Fe ↔ Co** — iron pnictide substitution

### Fraction Bias Toward Numerator=1

The model defaults to the simplest fraction regardless of true numerator:
- FRAC:3/10 → FRAC:1/10 (13× per epoch)
- FRAC:2/5 → FRAC:1/10 (10× per epoch)
- FRAC:1/5 → FRAC:1/10 (12× per epoch)
- FRAC:1/2 → FRAC:1/10 (10× per epoch)

The denominator is usually correct (the model knows "tenths" vs "twentieths") but defaults to numerator=1.

### Integer 2 Is Catastrophically Confused

The token `2` is misgenerated as a fraction ~250–300 times per epoch:
- `2 → FRAC:1/2`: 31–34 per epoch
- `2 → FRAC:1/10`: 13–30 per epoch
- `2 → FRAC:1/100`: 15–25 per epoch
- `2 → FRAC:1/20`: 13–20 per epoch

### Position-Dependent Degradation

Type confusion worsens deeper in the sequence:
- Position 0: ~3% type confusion
- Positions 1–2: ~45–55% type confusion
- Positions 3–7: ~58–70% type confusion
- Positions 8+: ~60–100% type confusion

### Complexity Correlation

| Correlation | r |
|------------|---|
| Sequence length vs errors | 0.49–0.53 |
| Number of elements vs errors | 0.46–0.48 |
| True Tc vs errors | 0.22–0.27 |

Formulas with 6+ elements are almost never correctly reconstructed.

---

## Conclusions

1. **The model has learned** — 98.7% train exact proves it understands superconductor formulas. The problem is the decoder has enough capacity to memorize training data without learning transferable composition rules.

2. **Type masking should be enabled** — `use_type_masking_ar=True` would eliminate 57% of validation errors if the type head is accurate. This is the single highest-impact change available.

3. **Decoder capacity reduction may force generalization** — reducing `latent_to_memory` (42M params) and/or transformer layers could force the decoder to learn compressed, generalizable representations instead of memorizing.

4. **Magpie MSE is still improving** — the encoder latent space continues to refine even as the decoder plateaus, confirming the bottleneck is decoder-side.
