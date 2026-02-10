# Theory-Guided Consistency Feedback (V12.16)

**Date**: February 2026

## Overview

This document describes the theory-guided consistency feedback system for the Multi-Task Superconductor Generator. The system incorporates physics-based constraints during training to improve generative capability.

## Components

### 1. Superconductor Family Classifier

**Location**: `src/superconductor/models/family_classifier.py`

Classifies superconductors into 14 families based on their dominant physical mechanism:

| ID | Family | Description |
|----|--------|-------------|
| 0 | NOT_SUPERCONDUCTOR | Non-superconductor (for mixed training) |
| 1 | BCS_CONVENTIONAL | Phonon-mediated, s-wave |
| 2 | CUPRATE_YBCO | YBa2Cu3O7 family |
| 3 | CUPRATE_LSCO | La2-xSrxCuO4 family |
| 4 | CUPRATE_BSCCO | Bi2Sr2Can-1CunO2n+4 family |
| 5 | CUPRATE_TBCCO | Tl-based cuprates |
| 6 | CUPRATE_HBCCO | Hg-based cuprates |
| 7 | CUPRATE_OTHER | Other cuprates |
| 8 | IRON_PNICTIDE | FeAs-based |
| 9 | IRON_CHALCOGENIDE | FeSe, FeTe-based |
| 10 | MGB2_TYPE | Two-gap BCS |
| 11 | HEAVY_FERMION | f-electron systems |
| 12 | ORGANIC | Organic superconductors |
| 13 | OTHER_UNKNOWN | Unknown mechanism |

**Classifiers Available**:
- `RuleBasedFamilyClassifier`: Deterministic, composition-based
- `LearnedFamilyClassifier`: Neural network, from latent/Magpie
- `HybridFamilyClassifier`: Combines both

### 2. Consistency Losses

**Location**: `src/superconductor/losses/consistency_losses.py`

Ensures encoder-decoder coherence:

```python
# Self-consistency: original vs reconstructed properties match
SelfConsistencyLoss(
    original_tc, reconstructed_tc,
    original_magpie, reconstructed_magpie,
)

# Bidirectional: forward-backward property validation
BidirectionalConsistencyLoss(
    original_tc, reconstructed_formula_tokens,
)
```

### 3. Theory Regularization Losses

**Location**: `src/superconductor/losses/theory_losses.py`

Applies physics-based constraints by family:

| Family | Theory | Constraint |
|--------|--------|------------|
| BCS | McMillan | Tc scales with Debye temperature |
| Cuprate | Doping Dome | Tc follows parabolic dome vs doping |
| Iron-based | Multi-band | Tc < 60K |
| Unknown | None | **No constraints** (intentional) |

**Key Design Decision**: Unknown/Other materials receive NO theory constraints. This is intentional because:
1. We don't know the underlying mechanism
2. Applying wrong constraints would hurt generalization
3. These materials might represent novel physics

### 4. Canonical Ordering

**Location**: `src/superconductor/data/canonical_ordering.py`

Standardizes element ordering in formulas:

```python
orderer = CanonicalOrderer(method=OrderingMethod.ELECTRONEGATIVITY)
# "La(7/10)Sr(3/10)CuO4" -> ordered by electronegativity
canonical = orderer.canonicalize(formula)
```

**Available orderings**:
- `ELECTRONEGATIVITY`: Cations first (most chemically intuitive)
- `ALPHABETICAL`: A-Z order
- `ABUNDANCE`: Most abundant elements first
- `HILL_SYSTEM`: C, H, then alphabetical (organic chemistry)

## Configuration

Add to `TRAIN_CONFIG` in `scripts/train_v12_clean.py`:

```python
TRAIN_CONFIG = {
    # ... existing config ...

    # Consistency feedback
    'use_consistency_loss': True,
    'consistency_weight': 0.1,
    'consistency_tc_weight': 1.0,
    'consistency_magpie_weight': 0.1,

    # Theory regularization
    'use_theory_loss': True,
    'theory_weight': 0.05,
    'theory_use_soft_constraints': True,

    # Family classifier (optional)
    'use_family_classifier': True,
    'family_classifier_weight': 0.01,
}
```

## Usage Examples

### Classify a Formula

```python
from superconductor.models.family_classifier import RuleBasedFamilyClassifier, SuperconductorFamily

classifier = RuleBasedFamilyClassifier()

# From element set
elements = {'Y', 'Ba', 'Cu', 'O'}
family = classifier.classify_from_elements(elements)
print(family)  # SuperconductorFamily.CUPRATE_YBCO

# From formula string
family = classifier.classify_from_formula("La(7/10)Sr(3/10)CuO4")
print(family)  # SuperconductorFamily.CUPRATE_LSCO
```

### Apply Theory Constraints

```python
from superconductor.losses.theory_losses import TheoryRegularizationLoss, TheoryLossConfig

config = TheoryLossConfig(theory_weight=0.05, use_soft_constraints=True)
theory_loss = TheoryRegularizationLoss(config)

loss_dict = theory_loss(
    formula_tokens=generated_tokens,
    predicted_tc=predicted_tc,
    magpie_features=magpie,
)
# loss_dict['total'] contains weighted sum of applicable theory losses
```

### Canonical Ordering

```python
from superconductor.data.canonical_ordering import (
    CanonicalOrderer, OrderingMethod, to_electronegativity_order
)

# Quick function
canonical = to_electronegativity_order("La(7/10)Sr(3/10)CuO4")

# Or with orderer instance
orderer = CanonicalOrderer(OrderingMethod.ELECTRONEGATIVITY)
canonical = orderer.canonicalize("La(7/10)Sr(3/10)CuO4")
```

## Theory Relationships Used

### BCS Theory (McMillan Formula)
**Reference:** McMillan, Phys. Rev. 167, 331 (1968)

```
Tc = (θ_D/1.45) × exp(-1.04(1+λ)/(λ - μ*(1+0.62λ)))
```

Where:
- θ_D = Debye temperature (K), typically 100-500K
- λ = electron-phonon coupling constant, typically 0.3-0.8
- μ* = Coulomb pseudo-potential, typically 0.10-0.15

The implementation learns θ_D and λ from Magpie features.
Hard upper limit: ~40K for conventional BCS materials.

### Cuprate Theory (Presland Formula)
**Reference:** Presland et al., Physica C 176 (1991) 95-105

```
Tc = Tc_max × [1 - 82.6(p - 0.16)²]
```

Where:
- p = hole doping level (typically 0.05-0.27)
- 0.16 = optimal doping (universal for cuprates)
- 82.6 = empirical Presland coefficient
- Tc_max = maximum Tc for the family (e.g., 165K for Hg-cuprates)

This parabolic "dome" is a universal feature of cuprate superconductors.

### Iron-based
**Reference:** Ren et al., Europhys. Lett. 83, 17002 (2008)

- Multi-band superconductivity with s± pairing
- Bulk Tc record: ~55K (SmFeAsO₁₋ₓFₓ)
- Monolayer FeSe/SrTiO₃: ~65K
- Practical limit for bulk: Tc < 60K

## Future Extensions

1. **More Theory Losses**: Add Eliashberg, heavy-fermion constraints
2. **Learned Family Classifier**: Train alongside VAE
3. **Conditional Generation**: Generate materials from target family
4. **BCS Parameter Prediction**: Predict λ, μ*, θ_D from Magpie

## References

- BCS Theory: Bardeen, Cooper, Schrieffer (1957)
- McMillan Formula: McMillan (1968)
- Cuprate Doping Dome: Presland et al. (1991)
