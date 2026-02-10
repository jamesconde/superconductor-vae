# Contrastive Learning Design for Superconductor Generative Model

**Date**: 2025-01-22
**Status**: Active - Training Overnight
**Author**: Claude + James

---

## Implementation Status (2026-02-09)

**Status**: Active - overnight training run on local RTX 4060

### Implemented
- `src/superconductor/losses/contrastive.py`: SuperconductorContrastiveLoss (SupCon with temperature scaling)
- Modified `scripts/train_v12_clean.py` (V12.12) with:
  - Contrastive mode config flag (`contrastive_mode: True`)
  - Data loading for 46K contrastive CSV (SC + non-SC)
  - Balanced sampling via WeightedRandomSampler (~50/50 SC/non-SC per batch)
  - SC samples: full loss (formula + Tc + Magpie + stoich + KL + REINFORCE)
  - Non-SC samples: formula-only loss at 0.5x weight, no Tc/Magpie/REINFORCE
  - Contrastive loss warmup over 100 epochs (weight 0.1)
  - Per-family extended labels (13 classes: 9 SC families + 4 non-SC categories)
  - SC vs non-SC exact match tracking
  - Retrain mode: resets catastrophic drop detector for data changes

### V12.19: Extended Label Scheme (2026-02-09)

Added class 12: **High-pressure (non-hydride)** for non-hydride HP-SC materials.

| Label | Category | Count |
|-------|----------|-------|
| 0 | Cuprates | 8,179 |
| 1 | Iron-based | 2,353 |
| 2 | Bismuthates | 2,360 |
| 3 | Borocarbides | 416 |
| 4 | Elemental Superconductors | 71 |
| 5 | Hydrogen-rich Superconductors | 45 |
| 6 | Organic Superconductors | 75 |
| 7 | Other | 9,952 |
| 8 | Non-SC: Materials Project | 14,040 |
| 9 | Non-SC: Magnetic | 4,009 |
| 10 | Non-SC: Thermoelectric | 3,287 |
| 11 | Non-SC: Anisotropy | 1,858 |
| 12 | High-pressure (non-hydride) | ~71 |

**Design**: HP-SC that are NOT hydrogen-rich get reassigned from their parent category
(e.g., Cuprates, Other, Elemental) to class 12. This clusters all non-hydride HP-SC
together in latent space. Hydrogen-rich SC stay in class 5 since they already form
a distinct cluster. The `category_to_label()` function now accepts a
`requires_high_pressure` parameter for this override.

### Training Progress (First epochs)
- Epoch 1289: SC Exact=25.7%, Loss=1.45, Contrastive=4.29
- Epoch 1290: SC Exact=25.3%, TRUE Autoregressive=23.7%, Contrastive=4.23
- Expected: Model recovering from normalization shift, should steadily improve
- Each epoch ~30 min on RTX 4060 8GB (46K samples, batch=32)

### Key Design Decisions
- Tc normalization computed from SC samples only (avoids non-SC Tc=0 skewing mean)
- Non-SC formula weight = 0.5x to avoid overwhelming SC training signal
- Balanced sampling ensures both types in every batch (critical for contrastive loss)
- REINFORCE disabled for non-SC samples (no reward signal for non-SC)

---

## 1. Problem Statement

### Current Limitations

The model achieves 82.4% exact match on formula reconstruction, but struggles with:

| Issue | Evidence | Root Cause |
|-------|----------|------------|
| Rare element errors | U: 41%, In: 32%, Te: 31% error rates | Insufficient training examples |
| Fraction digit errors | 32% of errors are 1-2 tokens off | Decoder hasn't seen enough numeric patterns |
| High-Tc compounds | 100-150K range: 23.5% error rate | Only 583 samples in this range |

### The Core Insight

The model has two components with different data needs:

```
┌─────────────────────────────────────────────────────────────┐
│ ENCODER                                                      │
│ Needs: Superconductor-specific features                     │
│ Goal: Learn what makes a material superconduct              │
│ Data: Must stay focused on SC materials                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        ┌─────────┐
                        │ Latent  │
                        │    z    │
                        └─────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ DECODER (NLP)                                                │
│ Needs: General chemistry tokenization                       │
│ Goal: Reconstruct ANY chemical formula accurately           │
│ Data: Benefits from ALL chemical formulas                   │
└─────────────────────────────────────────────────────────────┘
```

**Solution**: Add non-superconductor materials to train the decoder on rare elements, while using contrastive learning to keep the latent space focused on superconductivity.

---

## 2. Contrastive Learning Background

### 2.1 Why Contrastive Learning?

When mixing SC and non-SC data, we need to prevent the latent space from becoming a generic materials encoder. Contrastive learning provides:

1. **Separation**: Push SC and non-SC representations apart
2. **Clustering**: Pull similar SC families together (cuprates, iron-based, etc.)
3. **Structure**: Create a latent space where "superconductivity" is a learnable direction

### 2.2 Industry Standard: NT-Xent / InfoNCE

Based on literature review, the dominant approach is **Normalized Temperature-scaled Cross-Entropy (NT-Xent)**, also known as InfoNCE:

- Used by SimCLR (Google), MoCo (Meta), CLIP (OpenAI)
- Key properties:
  - Cosine similarity (scale-invariant, works in high-D)
  - Temperature scaling (controls sharpness)
  - Pairwise comparisons (no centroid assumptions)
  - Handles multi-modal distributions

**References**:
- [SimCLR Paper](https://arxiv.org/abs/2002.05709) - Chen et al., 2020
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) - Khosla et al., 2020
- [CONSMI for Molecular Generation](https://www.mdpi.com/1420-3049/29/2/495)
- [C-GMVAE for Multi-Label](https://proceedings.mlr.press/v162/bai22c/bai22c.pdf)

---

## 3. Proposed Contrastive Loss

### 3.1 Supervised Contrastive Loss (SupCon)

Since we have labels (SC vs non-SC), we use Supervised Contrastive Loss:

```python
class SuperconductorContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for SC vs non-SC separation.

    Based on Khosla et al. (2020) "Supervised Contrastive Learning"
    with adaptations for materials science.

    Key principles:
    - Cosine similarity (not Euclidean distance)
    - Temperature scaling (τ = 0.07 is standard)
    - Pairwise comparisons (no centroid assumptions)
    - Multiple positives per anchor (all SC samples are mutual positives)
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, z, labels):
        """
        Args:
            z: [batch, latent_dim] - embeddings from encoder
            labels: [batch] - class labels
                    Simple: 0=non-SC, 1=SC
                    Extended: 0-8=SC families, 9+=non-SC categories

        Returns:
            Scalar loss value
        """
        # Project to unit hypersphere (cosine similarity)
        z = F.normalize(z, dim=1)
        batch_size = z.size(0)

        # Pairwise similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [batch, batch]

        # Positive mask: same class (excluding self)
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float()
        pos_mask.fill_diagonal_(0)  # Exclude self-contrast

        # Mask for all valid pairs (exclude self)
        logits_mask = torch.ones_like(pos_mask).fill_diagonal_(0)

        # Numerical stability: subtract max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Compute log-softmax over all pairs
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average log-probability over positive pairs
        # For SC sample: positives = all other SC samples in batch
        # For non-SC sample: positives = all other non-SC samples in batch
        pos_count = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

        # Handle samples with no positives (shouldn't happen with balanced batches)
        mean_log_prob = mean_log_prob * (pos_count > 0).float()

        # Temperature scaling for gradient magnitude
        loss = -(self.temperature / self.base_temperature) * mean_log_prob

        return loss.mean()
```

### 3.2 Temperature Parameter

The temperature τ controls the "hardness" of the contrastive objective:

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| τ = 0.01 | Very sharp, hard negatives | Risk of gradient explosion |
| τ = 0.07 | Standard (SimCLR/MoCo) | **Recommended starting point** |
| τ = 0.1 | Slightly softer | Good for noisy labels |
| τ = 0.5 | Very soft | May lose discriminative power |

**Recommendation**: Start with τ = 0.07, tune if needed.

### 3.3 Extended Labels for SC Families

For better latent structure, use SC family labels instead of binary:

```python
SC_FAMILY_LABELS = {
    # Superconductor families (labels 0-8)
    'YBCO': 0,
    'LSCO': 1,
    'Hg-cuprate': 2,
    'Tl-cuprate': 3,
    'Bi-cuprate': 4,
    'Iron-based': 5,
    'MgB2': 6,
    'Conventional': 7,
    'Other-SC': 8,

    # Non-superconductor categories (labels 9+)
    'Oxide': 9,
    'Metal': 10,
    'Semiconductor': 11,
    'Insulator': 12,
    'Other-nonSC': 13,
}
```

This creates structure within the SC cluster while maintaining SC vs non-SC separation.

---

## 4. Data Strategy

### 4.1 Data Sources

| Source | Size | Coverage | Priority |
|--------|------|----------|----------|
| SuperCon (current) | 16.5K | SC only | Baseline |
| Materials Project | 150K+ | All elements | HIGH - rare elements |
| ICSD | 200K+ | Crystallographic | MEDIUM |
| OQMD | 800K+ | Computed | LOW - theoretical |

### 4.2 Recommended Ratios

Start conservative, increase non-SC data if decoder still struggles:

| Phase | SC Data | Non-SC Data | Ratio | Epochs |
|-------|---------|-------------|-------|--------|
| Phase 0 (current) | 16.5K | 0 | 1:0 | 0-1284 |
| Phase 1 | 16.5K | 8K | 2:1 | 1285-1500 |
| Phase 2 | 16.5K | 16.5K | 1:1 | 1500-1800 |
| Phase 3 | 16.5K | 33K | 1:2 | 1800+ (if needed) |

### 4.3 Non-SC Material Selection

Prioritize materials containing high-error-rate elements:

| Element | SC Error Rate | Target Non-SC Samples | Source |
|---------|---------------|----------------------|--------|
| U | 41.4% | 500+ uranium compounds | Materials Project |
| In | 32.3% | 500+ indium compounds | Materials Project |
| Te | 30.9% | 500+ tellurides | Materials Project |
| Au | 29.6% | 300+ gold compounds | Materials Project |
| Be | 29.5% | 200+ beryllides | Materials Project |
| Sn | 26.9% | 500+ stannides | Materials Project |
| Re | 26.1% | 300+ rhenium compounds | Materials Project |

### 4.4 Curriculum Strategy

Gradually introduce non-SC data:

```python
def get_data_mix_ratio(epoch, phase1_end=200, phase2_end=500):
    """
    Curriculum for mixing SC and non-SC data.

    Returns:
        sc_weight: Sampling weight for SC data
        non_sc_weight: Sampling weight for non-SC data
    """
    if epoch < phase1_end:
        # Phase 1: Mostly SC, introduce non-SC slowly
        progress = epoch / phase1_end
        non_sc_ratio = 0.2 * progress  # 0% → 20%
    elif epoch < phase2_end:
        # Phase 2: Increase to balanced
        progress = (epoch - phase1_end) / (phase2_end - phase1_end)
        non_sc_ratio = 0.2 + 0.3 * progress  # 20% → 50%
    else:
        # Phase 3: Maintain balance
        non_sc_ratio = 0.5

    return 1.0 - non_sc_ratio, non_sc_ratio
```

---

## 5. Loss Function Integration

### 5.1 Combined Loss

```python
def compute_total_loss(
    outputs, targets, z, labels,
    is_superconductor,
    contrastive_weight=0.1,
    epoch=0
):
    """
    Combined loss for mixed SC/non-SC training.

    Args:
        outputs: Decoder outputs (formula logits, Tc pred, Magpie pred)
        targets: Ground truth (tokens, Tc, Magpie features)
        z: Latent representations [batch, latent_dim]
        labels: Class labels for contrastive loss
        is_superconductor: Boolean mask [batch]
        contrastive_weight: Weight for contrastive loss
        epoch: Current epoch (for curriculum)

    Returns:
        total_loss: Combined scalar loss
        loss_components: Dict of individual losses for logging
    """
    # === SC Samples: Full Reconstruction Loss ===
    sc_mask = is_superconductor

    if sc_mask.any():
        # Formula reconstruction (focal loss)
        formula_loss_sc = focal_cross_entropy(
            outputs['logits'][sc_mask],
            targets['tokens'][sc_mask],
            gamma=2.0
        )

        # Tc prediction
        tc_loss = F.mse_loss(
            outputs['tc_pred'][sc_mask],
            targets['tc'][sc_mask]
        )

        # Magpie reconstruction
        magpie_loss = F.mse_loss(
            outputs['magpie_pred'][sc_mask],
            targets['magpie'][sc_mask]
        )

        # Stoichiometry loss
        stoich_loss = F.mse_loss(
            outputs['stoich_pred'][sc_mask],
            targets['stoich'][sc_mask]
        )

        sc_loss = formula_loss_sc + 10*tc_loss + 2*magpie_loss + 2*stoich_loss
    else:
        sc_loss = 0.0

    # === Non-SC Samples: Formula-Only Loss (Lower Weight) ===
    non_sc_mask = ~is_superconductor

    if non_sc_mask.any():
        # Only formula reconstruction, lower weight
        formula_loss_non_sc = focal_cross_entropy(
            outputs['logits'][non_sc_mask],
            targets['tokens'][non_sc_mask],
            gamma=2.0
        )
        non_sc_loss = 0.5 * formula_loss_non_sc  # Half weight
    else:
        non_sc_loss = 0.0

    # === Contrastive Loss: Separate SC from Non-SC ===
    contrastive_loss = supcon_loss(z, labels)

    # === Combine ===
    # Warm up contrastive loss
    contrastive_warmup = min(1.0, epoch / 100)
    effective_contrastive_weight = contrastive_weight * contrastive_warmup

    total_loss = sc_loss + non_sc_loss + effective_contrastive_weight * contrastive_loss

    return total_loss, {
        'sc_loss': sc_loss,
        'non_sc_loss': non_sc_loss,
        'contrastive_loss': contrastive_loss,
        'total_loss': total_loss,
    }
```

### 5.2 Loss Weight Schedule

| Loss Component | Weight | Notes |
|----------------|--------|-------|
| SC Formula | 1.0 | Focal loss, γ=2.0 |
| SC Tc | 10.0 | Critical property |
| SC Magpie | 2.0 | Material features |
| SC Stoichiometry | 2.0 | Fraction accuracy |
| Non-SC Formula | 0.5 | Lower weight, decoder-only |
| Contrastive | 0.1 | Warm up over 100 epochs |

---

## 6. Implementation Plan

### 6.1 Phase 1: Data Preparation

1. **Download Materials Project data** for high-error elements
2. **Parse and tokenize** non-SC formulas
3. **Create unified dataloader** with SC/non-SC mixing
4. **Assign labels** (SC families + non-SC categories)

### 6.2 Phase 2: Loss Function

1. **Implement SuperconductorContrastiveLoss**
2. **Add to training loop** with warmup
3. **Modify loss computation** for mixed batches
4. **Add logging** for contrastive loss component

### 6.3 Phase 3: Training

1. **Resume from checkpoint** (epoch 1284)
2. **Start with 2:1 ratio** (SC:non-SC)
3. **Monitor** SC accuracy, non-SC accuracy, contrastive loss
4. **Adjust ratio** based on results

### 6.4 Phase 4: Evaluation

1. **Re-run full error analysis**
2. **Compare** rare element error rates
3. **Check** latent space structure (t-SNE/UMAP)
4. **Verify** SC vs non-SC separation

---

## 7. Expected Outcomes

### 7.1 Decoder Improvements

| Metric | Current | Expected |
|--------|---------|----------|
| U error rate | 41.4% | <20% |
| In error rate | 32.3% | <15% |
| Te error rate | 30.9% | <15% |
| Overall exact match | 82.4% | >90% |

### 7.2 Latent Space Structure

```
Before (current):
┌─────────────────────────────┐
│ SC samples scattered        │
│ No clear structure          │
│ No non-SC reference         │
└─────────────────────────────┘

After (with contrastive):
┌─────────────────────────────┐
│      ┌───────┐              │
│      │Cuprate│              │
│      │cluster│              │
│      └───────┘              │
│  ┌──────┐    ┌─────────┐    │
│  │Iron- │    │Convent- │    │
│  │based │    │ional    │    │
│  └──────┘    └─────────┘    │
│                             │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │ ← SC/non-SC boundary
│                             │
│    × × × × × × × × ×        │ ← Non-SC materials
│    × × × × × × × × ×        │
└─────────────────────────────┘
```

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Non-SC data overwhelms SC signal | Start with 2:1 ratio, lower non-SC loss weight |
| Contrastive loss destabilizes training | Warm up over 100 epochs, low weight (0.1) |
| SC families collapse together | Use family labels, not just binary |
| Decoder overfits to non-SC | Monitor SC metrics separately |

---

## 9. Open Questions

1. **Optimal temperature**: Start at 0.07, but may need tuning
2. **Family label assignment**: Need heuristics to auto-assign SC families
3. **Non-SC category granularity**: How many categories? Too many may not help
4. **Materials Project API**: Rate limits, data format compatibility

---

## 10. References

1. Chen et al. (2020) "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
2. Khosla et al. (2020) "Supervised Contrastive Learning" (SupCon)
3. Bai et al. (2022) "Gaussian Mixture VAE with Contrastive Learning" (C-GMVAE)
4. CONSMI (2024) "Contrastive Learning in SMILES for Molecular Generation"
5. VECTOR+ (2025) "Valid Property-Enhanced Contrastive Learning for Drug Design"

---

## Appendix A: Full Loss Function Implementation

See `src/superconductor/losses/contrastive.py` (to be created).

## Appendix B: Data Loading Changes

See `src/superconductor/data/mixed_loader.py` (to be created).
