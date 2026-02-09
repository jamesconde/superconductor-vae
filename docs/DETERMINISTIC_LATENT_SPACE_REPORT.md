# Report: VAE to Deterministic Coordinate Space Conversion

**Date**: 2026-02-02
**Author**: James
**Status**: Implemented, pending training validation

---

## Summary

Converted the encoder from a probabilistic VAE (mu/logvar/reparameterization/KL divergence) to a **deterministic coordinate system** where `z = fc_mean(h)` directly. KL divergence replaced with light L2 regularization (`mean(z^2)`). The decoder, contrastive loss, REINFORCE, and all downstream consumers are unchanged.

---

## Motivation

The VAE's reparameterization trick samples `z = mu + sigma * epsilon` (where epsilon ~ N(0,1)), meaning the **same input produces different z values on every forward pass**. This causes:

1. **Noisy contrastive learning** -- Contrastive loss relies on consistent representations. Stochastic z means anchor/positive pairs get different z each time, weakening the gradient signal.
2. **Unstable REINFORCE baselines** -- Policy gradient methods benefit from deterministic state representations.
3. **KL divergence pressure** -- KL pushes z toward N(0,1), fighting against the encoder's ability to spread materials across the latent space in a structured way. This is counterproductive when the goal is a meaningful coordinate system, not a generative model.

A deterministic encoder gives each material a **fixed coordinate** in latent space -- same input always maps to the same z. This is the correct abstraction for a coordinate system.

---

## Changes Made

### `src/superconductor/models/attention_vae.py`

| Component | Change |
|-----------|--------|
| `AttentionVAEEncoder.__init__()` | Added `deterministic=False` parameter. When `True`, `fc_logvar` is not created. `fc_mean` retained (same name = checkpoint weights load directly). |
| `AttentionVAEEncoder.forward()` | Returns `(fc_mean(h), None)` in deterministic mode. `None` signals no logvar downstream. |
| `FullMaterialsVAE.__init__()` | Passes `deterministic=True` to encoder. |
| `FullMaterialsVAE.reparameterize()` | Returns `mean` directly when `logvar is None` (passthrough). |
| `FullMaterialsVAE.forward()` | KL replaced with `mean(z^2)` L2 regularization. **Returned under same `'kl_loss'` key** (see key reuse note below). |

### `scripts/train_v12_clean.py`

| Component | Change |
|-----------|--------|
| Checkpoint loading | `strict=False` on `encoder.load_state_dict()` to ignore old `fc_logvar` weights. Logs missing/unexpected keys. |
| Optimizer restore | try/except around encoder optimizer load (param count changed, old state incompatible). Falls back to fresh optimizer. |
| Config comment | `kl_weight` annotated as L2 regularization weight. |
| z_norm monitoring | New `total_z_norm` accumulator, `z_norm` in results dict, `zN:` in epoch summary output. |

---

## Key Design Decision: `'kl_loss'` Key Reuse

**The `'kl_loss'` dictionary key is intentionally reused for L2 regularization.** This is documented with comments in both files.

**Why**: The entire downstream pipeline reads the regularization loss via this key:
- `CombinedLossWithREINFORCE` receives it as a parameter
- `train_v12_clean.py` multiplies by `config['kl_weight']`
- Loss logging, checkpoint saving, etc.

Renaming to `'z_reg_loss'` or similar would require touching 10+ callsites for zero functional benefit. The value is `mean(z^2)` (L2 regularization) when `deterministic=True`, NOT KL divergence. Comments in both `attention_vae.py` (forward method) and `train_v12_clean.py` (training loop) document this.

**For future Claude sessions**: If you see `kl_loss` in the encoder output or training loop, check `z_logvar` -- if it's `None`, the encoder is deterministic and `kl_loss` is actually L2 reg.

---

## Checkpoint Compatibility

| Component | Behavior |
|-----------|----------|
| `fc_mean` weights | Load directly (same parameter name and shape) |
| `fc_logvar` weights | Silently ignored via `strict=False` |
| Encoder optimizer state | Fresh init (parameter count changed, old state incompatible) |
| Decoder checkpoint | Loads unchanged |
| Decoder optimizer | Loads unchanged |

---

## What Was NOT Changed

- **Decoder** (`autoregressive_decoder.py`) -- Only consumes `z` tensor
- **Contrastive loss** (`contrastive.py`) -- Takes `z` directly (now deterministic = better signal)
- **REINFORCE** -- Consumes `z`
- **Loss function** (`CombinedLossWithREINFORCE`) -- Receives value by `kl_loss` key
- **All decoder heads** (Tc, Magpie, fraction, competence) -- Consume `z`
- **Old `AttentionVAE` class** -- Retains VAE behavior (uses `deterministic=False` default)

---

## Testing

Unit tests (run with `recursivemenn-py311` conda env):

- Deterministic encoder returns `(mean, None)` -- **PASS**
- Non-deterministic encoder returns `(mean, logvar)` -- **PASS** (backward compat)
- Reparameterize passthrough when `logvar=None` returns mean unchanged -- **PASS**
- Reparameterize with logvar adds noise -- **PASS**
- L2 reg computation produces positive scalar -- **PASS**

---

## Verification Checklist (for first training run)

1. [ ] Load epoch-1318 checkpoint with `strict=False`, confirm no errors
2. [ ] Confirm `encoder_out['z']` shape is `[batch, 2048]` and `z_logvar` is `None`
3. [ ] Verify loss is finite and reasonable (not wildly different from ~1.15)
4. [ ] Train 5-10 epochs, confirm loss decreases and accuracy doesn't regress
5. [ ] Monitor `zN` in epoch summary -- should be stable, not growing unboundedly
6. [ ] If `zN` grows past ~100, consider increasing `kl_weight` (L2 strength)

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| z norms explode without KL pressure | Low (L2 reg prevents this) | Monitor `zN` per epoch; increase `kl_weight` if needed |
| Loss spike on first epoch after conversion | Medium (optimizer state reset for encoder) | Expected -- encoder LR will be fresh. Should converge within a few epochs |
| Contrastive loss changes behavior | None | Contrastive loss already took `z` directly; now it's just deterministic |
| Old checkpoints won't load | None | `strict=False` handles missing `fc_logvar` gracefully |
