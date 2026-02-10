# Multi-Task Superconductor Generator — Architecture Guide

**Last updated**: 2026-02-10 (V12.20)
**Code**: `src/superconductor/models/attention_vae.py`, `src/superconductor/models/autoregressive_decoder.py`
**Training**: `scripts/train_v12_clean.py`

> **Note on naming**: The code uses class names like `FullMaterialsVAE` and `AttentionVAEEncoder`
> from the project's origins as a Variational Autoencoder. The architecture has since evolved into
> a **deterministic multi-task superconductor generator** — there is no stochastic sampling or KL divergence.
> The latent space is shaped by six prediction objectives and contrastive learning, not by a
> variational prior. Code class names are kept for backward compatibility.

---

## Overview

This is a **multi-task superconductor generator** — a neural network that learns a compressed representation of superconducting materials. Given a material's composition, properties, and critical temperature, it encodes everything into a single 2048-dimensional vector (`z`), then uses that vector to reconstruct the original inputs and predict new properties.

The system is **two networks** trained jointly:

1. **Encoder** (`FullMaterialsVAE`) — 5.4M parameters
2. **Formula Decoder** (`EnhancedTransformerDecoder`) — 102.8M parameters

**Total: ~108M parameters**

---

## The Big Picture

```
                        ENCODER (FullMaterialsVAE)
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │  ┌─────────────────┐                                         │
    │  │ Element Encoder  │──┐                                     │
    │  │ (attention-based)│  │                                     │
    │  └─────────────────┘  │  ┌────────┐   ┌─────────────┐       │
    │                       ├──│ Fusion │───│ Latent Enc. │───► z  │
    │  ┌─────────────────┐  │  │ (768→  │   │ (768→512→   │  2048 │
    │  │ Magpie Encoder  │──┘  │  768)  │   │  256→2048)  │  dims │
    │  │ (145→512→256)   │     └────────┘   └─────────────┘       │
    │  └─────────────────┘  │                                      │
    │                       │                                      │
    │  ┌─────────────────┐  │                                      │
    │  │  Tc Encoder     │──┘                                      │
    │  │  (1→128→256)    │                                         │
    │  └─────────────────┘                                         │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
                          z (2048 dims)
                     The learned representation
                                │
                ┌───────────────┼───────────────────────────┐
                │               │                           │
                ▼               ▼                           ▼
    ┌───────────────────┐   Directly from z          ┌──────────────┐
    │  Shared Backbone  │       │                    │   Formula     │
    │  (2048→256→512)   │   ┌───┼───┬───┐            │   Decoder     │
    │                   │   │   │   │   │            │  (Transformer │
    │  ┌─────┐ ┌─────┐ │   │   │   │   │            │   12 layers)  │
    │  │ Tc  │ │Magp.│ │   │   │   │   │            │              │
    │  │Head │ │Head │ │   │   │   │   │            │  102.8M      │
    │  │→1   │ │→145 │ │   │   │   │   │            │  params      │
    │  └──┬──┘ └──┬──┘ │   │   │   │   │            └──────┬───────┘
    │     │       │    │   │   │   │   │                   │
    │  ┌──┴──┐    │    │   │   │   │   │                   │
    │  │Attn.│    │    │   │   │   │   │                   │
    │  │Head │    │    │   │   │   │   │                   ▼
    │  │→256 │    │    │   │   │   │   │            Formula tokens
    │  └──┬──┘    │    │   │   │   │   │            (e.g. "Y1Ba2Cu3O7")
    │     │       │    │   │   │   │   │
    └─────┼───────┼────┘   │   │   │   │
          │       │        │   │   │   │
          ▼       ▼        ▼   ▼   ▼   ▼
        Tc pred  Magpie  Comp. Frac. HP
        (1 val)  (145)   head  head  head
                         (1)  (13)  (1)
```

---

## How It Works: One Network, Many Tasks

The most important concept is **multi-task learning through shared representations**.

### The Latent Bottleneck

All information about a material must pass through `z` — a single vector of 2048 numbers. This forces the encoder to learn what matters. It can't memorize every input; it must find patterns and compress them.

### Why Multiple Heads?

Each prediction head asks a different question about `z`:

| Head | Question | Output | Loss |
|------|----------|--------|------|
| **Tc head** | "What is the critical temperature?" | 1 scalar | Huber (on log1p-normalized Tc) |
| **Magpie head** | "What are the material properties?" | 145 features | MSE |
| **Attended head** | "What conditioning does the formula decoder need?" | 256-dim vector | (indirect) |
| **Competence head** | "How confident is the model?" | 1 probability | BCE |
| **Fraction head** | "What are the element stoichiometries?" | 12 fractions + count | Masked MSE |
| **HP head** | "Does this need high pressure?" | 1 probability | BCE |

**The heads don't just predict. They teach the encoder what to care about.**

Without the HP head, the encoder has no reason to encode pressure information — it would be discarded during compression. With it, the encoder is penalized every time `z` doesn't contain enough information to predict whether a material requires high pressure. Gradients flow backward through the head, through `z`, and into the shared encoder layers.

### Gradient Flow

During one training step, ALL heads produce losses simultaneously. These losses are weighted and summed:

```
total_loss = formula_loss          (weight: 1.0)
           + tc_weight * tc_loss   (weight: 10.0, Huber δ=1.0 on log1p Tc)
           + magpie_loss           (weight: 2.0)
           + stoich_loss           (weight: 2.0)
           + kl_loss               (weight: 0.0001)
           + contrastive_loss      (weight: 0.1)
           + hp_loss               (weight: 0.05)
```

One backward pass sends gradients from ALL losses through the shared encoder. The encoder layers see the combined signal: "encode Tc better AND encode Magpie features better AND encode pressure information AND make the formula decoder's job easier." This produces a richer, more useful latent space than any single task alone.

---

## Detailed Component Reference

### Encoder Input Branches

Three parallel branches process different input types, each producing a 256-dim representation:

**Element Encoder** (133K params)
- Input: atomic numbers [batch, 12], fractions [batch, 12], mask [batch, 12]
- Embeds each element (128-dim), applies multi-head attention (8 heads)
- Output: 256-dim composition representation
- This captures which elements are present and their relative amounts

**Magpie Encoder** (208K params)
- Input: 145 Magpie-derived material property features
- Two-layer MLP: 145 → 512 → 256 (with LayerNorm, GELU, Dropout)
- Output: 256-dim property representation
- Magpie features include things like electronegativity stats, atomic radius stats, etc.

**Tc Encoder** (34K params)
- Input: normalized critical temperature (1 scalar)
- Two-layer MLP: 1 → 128 → 256 (with GELU, LayerNorm)
- Output: 256-dim temperature representation

### Fusion and Compression

**Fusion Layer** (592K params)
- Concatenates all three branch outputs: [256 + 256 + 256] = 768
- Linear(768, 768) + LayerNorm + GELU + Dropout
- Mixes information across branches

**Latent Encoder** (1.05M params) *(code: `AttentionVAEEncoder` with `deterministic=True`)*
- MLP: 768 → 512 → 256 → 2048
- Deterministic: `z = fc_mean(h)` — no sampling, no reparameterization noise
- L2 regularization on z (not KL divergence)
- Produces the final latent vector `z`

### Prediction Heads (from z)

**Shared Decoder Backbone** (658K params)
- MLP: 2048 → 256 → 512 (with LayerNorm, GELU, Dropout)
- The Tc head, Magpie head, and Attended head all branch from this backbone

Three heads use the backbone output (512-dim):

| Head | Architecture | Params | Output |
|------|-------------|--------|--------|
| Tc head | 512 → 256 → 1 | 132K | Critical temperature |
| Magpie head | 512 → 512 → 145 | 337K | Material properties |
| Attended head | 512 → 256 + LayerNorm | 132K | Formula decoder conditioning |

Three heads operate directly on z (2048-dim):

| Head | Architecture | Params | Output |
|------|-------------|--------|--------|
| Competence head | 2048 → 512 → 1 + Sigmoid | 1.05M | Model confidence |
| Fraction head | 2048 → 256 → 128 → 13 | 560K | Stoichiometry (12 fractions + element count) |
| HP head | 2048 → 256 → 1 | 525K | High-pressure prediction (logit) |

### Formula Decoder (Separate Network)

**EnhancedTransformerDecoder** (102.8M params)
- 12-layer Transformer decoder with 8 attention heads
- d_model=512, dim_feedforward=2048
- 16 learned memory tokens for additional context
- Takes `z` and `attended_input` from the encoder
- Autoregressively generates formula tokens (e.g., "Y", "1", "Ba", "2", "Cu", "3", "O", "7")
- Has its own optimizer (separate from encoder)
- Trained with cross-entropy + REINFORCE (reward = formula correctness)

---

## Training Loop Summary

Each training step:

1. **Forward pass**: Inputs → Encoder → z → All heads produce predictions
2. **Formula decoder**: z + attended_input → autoregressive token generation
3. **Loss computation**: Each head computes its loss against ground truth
4. **Contrastive loss**: Pushes same-class z vectors together, different-class apart (13 SC/non-SC categories)
5. **Total loss**: Weighted sum of all losses
6. **Backward pass**: Gradients flow from all losses through z into shared encoder
7. **Optimizer step**: One step updates the entire encoder; a separate step updates the formula decoder

### What Makes This Architecture Distinctive

- **Deterministic encoder**: No sampling noise in z. Uses L2 regularization instead of KL divergence.
- **Multi-task heads**: Seven prediction heads shape the latent space simultaneously (Tc, Magpie, fractions, element count, HP, competence, SC classification).
- **Contrastive learning**: 13-class SupCon loss clusters materials by type in latent space.
- **REINFORCE**: Formula decoder gets reward signal from formula-level correctness, not just token-level cross-entropy.
- **Asymmetric SC/non-SC training**: Superconductors get full loss (all heads). Non-superconductors get formula loss only at 0.5x weight (they have no meaningful Tc or HP to predict).

---

## At Inference Time

Once trained, you can use any head independently on a new material:

```python
# Encode a material
encoder_out = encoder(elem_indices, elem_fractions, elem_mask, magpie_features, tc)
z = encoder_out['z']  # [1, 2048]

# Use any prediction head
tc_predicted = encoder_out['tc_pred']           # Critical temperature
hp_probability = torch.sigmoid(encoder_out['hp_pred'])  # P(high-pressure)
sc_probability = torch.sigmoid(encoder_out['sc_pred'])  # P(superconductor) — cross-head consistency
confidence = encoder_out['competence']          # Model confidence

# Generate formula
formula_tokens = decoder.generate(z, encoder_skip=encoder_out['attended_input'])
```

Or use the latent space directly:
- **Interpolate** between two materials' z vectors to explore intermediate compositions
- **Cluster** z vectors to discover material families
- **Sample** z vectors to generate novel superconductor candidates

---

## File Reference

| File | Contains |
|------|----------|
| `src/superconductor/models/attention_vae.py` | `FullMaterialsVAE` (encoder + all heads) |
| `src/superconductor/models/autoregressive_decoder.py` | `EnhancedTransformerDecoder` (formula decoder) |
| `src/superconductor/losses/contrastive.py` | `SuperconductorContrastiveLoss`, category labels |
| `scripts/train_v12_clean.py` | Training loop, loss weighting, data loading |
| `scripts/label_high_pressure.py` | HP-SC labeling pipeline |
