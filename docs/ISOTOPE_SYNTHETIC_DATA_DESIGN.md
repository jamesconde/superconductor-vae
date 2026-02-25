# Isotope-Variant Synthetic Data: Feasibility Analysis & Design

**Date**: 2026-02-20
**Status**: Analysis complete — approach requires significant revision from original assumptions

---

## Executive Summary

The original V14.0 plan assumed we could "fill the isotope data gap later via theory-guided generation using the BCS `estimate_isotope_effect()` function." This document critically examines that assumption.

**Verdict**: The isotope effect formula `Tc_new = Tc_old * (M_old / M_new)^alpha` is **physically valid but practically low-value** for training data generation. The Tc shifts from isotope substitution are sub-Kelvin for most materials — too small to meaningfully augment the model's generative capability. The V14.0 isotope tokens remain valuable infrastructure, but the data pipeline must use different strategies than originally envisioned.

---

## 1. What the BCS Isotope Effect Actually Gives Us

### The Formula

The isotope effect in BCS superconductors follows:

```
Tc ∝ M^(-α)    →    Tc₂ / Tc₁ = (M₁ / M₂)^α
```

where `M` is the isotopic mass and `α` is the isotope effect exponent. BCS theory predicts `α = 0.5` exactly for weak-coupling, single-band, phonon-mediated superconductors.

This is implemented in the codebase at `src/superconductor/encoders/isotope_properties.py:estimate_isotope_effect()`.

### What Alpha Actually Looks Like Experimentally

| Material | Measured α | Type | Citation |
|----------|-----------|------|----------|
| Hg | ~0.50 | Conventional | Maxwell (1950), Phys. Rev. 78, 477 |
| Sn | 0.46–0.47 | Conventional | Serin, Reynolds & Nesbitt (1952), Phys. Rev. 86, 162 |
| Pb | ~0.48 | Conventional | Lock et al. (1951), Nature 168, 245 |
| Tl | 0.50–0.61 | Conventional | Webb et al. (2015), Physica C, arXiv:1502.04724 |
| Cd | ~0.50 | Conventional | Standard reference |
| Zn | 0.30–0.45 | Conventional | Standard reference |
| Re | 0.36–0.38 | Transition metal | Reduced due to d-electron contributions |
| Mo | 0.33–0.37 | Transition metal | Reduced |
| Os | ~0.21 | Transition metal | Nearly absent |
| Ru | ~0.0 | Transition metal | Absent |
| MgB₂ (boron) | 0.28–0.30 | Multi-band | Bud'ko et al. (2001), PRL 86, 1877; Hinks et al. (2001) |
| YBa₂Cu₃O₇ (O, optimal) | 0.02–0.06 | Cuprate | Near zero at optimal doping |
| Cuprates (O, underdoped) | 0.5–1.0+ | Cuprate | Anomalously large; Bussmann-Holder et al. (2007), PNAS |
| SmFeAsO₁₋ₓFₓ (Fe) | >0 (normal) | Iron-based | Normal isotope effect |
| (Ba,K)Fe₂As₂ | Varies, can be <0 | Iron-based | Both normal and inverse; Huang & Lin (2019), Sci. Rep. |

**Key observations**:
- α ≈ 0.5 works well for non-transition-metal elemental superconductors (Hg, Sn, Pb, Tl, Cd)
- Transition metals show systematically reduced α (0.0–0.4) due to d-electron pairing contributions
- Cuprate α is doping-dependent and can exceed 0.5 or be nearly zero — **not predictable from composition**
- Iron-based α can be **negative** (inverse isotope effect)

### The Actual Tc Shifts Are Tiny

For most elements, natural isotope mass variations are small percentages of the total mass. The resulting Tc shifts are sub-Kelvin:

| Material | Tc (K) | Isotope Pair | Mass Ratio | α | ΔTc (K) |
|----------|--------|-------------|-----------|---|---------|
| Sn | 3.72 | ¹¹²Sn → ¹²⁴Sn | 1.107 | 0.47 | -0.17 |
| Hg | 4.15 | ¹⁹⁶Hg → ²⁰⁴Hg | 1.041 | 0.50 | -0.08 |
| Pb | 7.19 | ²⁰⁴Pb → ²⁰⁸Pb | 1.020 | 0.48 | -0.07 |
| Zn | 0.85 | ⁶⁴Zn → ⁷⁰Zn | 1.094 | 0.38 | -0.03 |
| MgB₂ | 39.0 | ¹⁰B → ¹¹B | 1.100 | 0.30 | -1.09 |
| Re | 1.70 | ¹⁸⁵Re → ¹⁸⁷Re | 1.011 | 0.37 | -0.007 |

MgB₂ is the outlier — its high Tc and light boron atoms produce a measurable ~1K shift. But for most materials, ΔTc < 0.2 K. **This is within the noise floor of our model's Tc prediction accuracy** (which has MAE of several Kelvin).

---

## 2. Why "Just Calculate Tc" Doesn't Work

### The Allen-Dynes Formula Requires Inputs We Don't Have

The Allen-Dynes modified McMillan formula (already implemented in `theory_losses.py`) is:

```
Tc = (ω_log / 1.2) × f₁ × f₂ × exp[-1.04(1+λ) / (λ - μ*(1+0.62λ))]
```

This requires:
- **ω_log** (logarithmic average phonon frequency): Needs full phonon spectrum from DFT
- **λ** (electron-phonon coupling constant): Needs Eliashberg spectral function α²F(ω) from DFPT
- **μ*** (Coulomb pseudopotential): Usually treated as adjustable parameter (~0.10–0.16)

**None of these can be determined from chemical composition alone.** As stated by Cerqueira et al. (2023): "it still remains a challenge to calculate Tc from first principles and its dependence on atomic structure, lattice instabilities, and magnetic activity" (npj Computational Materials, doi:10.1038/s41524-022-00933-1).

The codebase works around this by learning λ and θ_D from Magpie features via neural networks — but these are **learned approximations**, not first-principles calculations. They can only predict Tc for materials similar to the training distribution.

### Accuracy When Everything Is Known

Even with DFT-quality inputs, Allen-Dynes achieves MAE ≈ 3.3 K for conventional superconductors (Gao, Sanna et al., 2025, Nature Communications, doi:10.1038/s41467-025-63702-w). It:
- Is only valid for λ ≤ 1.5
- Systematically underpredicts Tc for strong-coupling materials
- Fails entirely for unconventional superconductors (cuprates, iron-based, heavy fermion)

### Composition-Only ML Models

State-of-the-art ML achieves RMSE ≈ 6–9 K from composition + Magpie features:

| Study | Year | MAE (K) | R² | Method |
|-------|------|---------|-----|--------|
| Qu et al. | 2024 | 3.54 | ~0.95 | GBFS (J. Chem. Inf. Model., doi:10.1021/acs.jcim.4c01137) |
| Lee et al. | 2024 | — | 0.952 | CatBoost (Sci. Rep., doi:10.1038/s41598-024-54440-y) |
| Roter & Dordevic | 2024 | — | — | DNN with experimental validation (Eur. Phys. J. Plus) |

These models learn statistical correlations, not physics. They struggle with novel families not in training data.

---

## 3. What IS Viable for Isotope Token Training Data

### Strategy A: Isotope Effect Augmentation (Limited Value)

Apply `Tc_new = Tc_old × (M_old/M_new)^α` to known BCS superconductors.

**Pros**: Physically justified, easy to implement, fully automated
**Cons**: ΔTc < 0.2 K for most materials; augments ~500 conventional SC entries with ~2000 nearly-identical variants

**Realistic yield**: ~2000 synthetic data points with Tc shifts within the model's noise floor. The model would learn "isotope variants have essentially the same Tc" — which is correct but not useful for generative discovery.

**Recommendation**: Implement as a consistency check (does the model produce the same Tc for {16}O and {18}O variants?), not as training augmentation.

### Strategy B: Isostructural Substitution Series (Higher Value)

Generate training data by substituting elements within known superconductor families while preserving crystal structure. This changes Tc by **tens of Kelvin**, not fractions.

Examples:
- REBa₂Cu₃O₇ series: RE = Y, La, Nd, Sm, Eu, Gd, Dy, Ho, Er, Tm, Yb, Lu
- RE₂₋ₓCeₓCuO₄ series (electron-doped cuprates)
- BaFe₂₋ₓCoₓAs₂ doping series

**Pros**: Meaningful Tc variation, physical basis, connects to existing SuperCon data
**Cons**: Need to verify each substitution is physical; not all substitutions form stable phases

### Strategy C: Literature Mining of Isotope Studies (Highest Value)

Extract actual experimental isotope effect data from published papers:
- Measured α exponents per material per element site
- Actual Tc values for isotope-substituted samples

Key sources:
- Franck (1994), review of oxygen isotope effects in cuprates (J. Phys. Soc. Japan)
- Bud'ko et al. (2001), boron isotope effect in MgB₂ (PRL)
- Huang & Lin (2019), iron-based isotope effects (Sci. Rep.)
- Webb et al. (2015), comprehensive elemental superconductor review (Physica C)

**Estimated yield**: ~50–100 data points with measured isotope-specific Tc values. Small but high-quality.

### Strategy D: Pressure-Variant Augmentation (Future, High Value)

High-pressure hydrides are the frontier of conventional superconductivity (H₃S at 203 K, LaH₁₀ at 250 K). DFT calculations of Tc(P) exist for many hydride compositions:
- Drozdov et al. (2015, 2019) for H₃S and LaH₁₀
- Cerqueira et al. (2023) systematic screening
- Choudhary et al. (2024) ML-accelerated electron-phonon calculations

This connects to the isotope tokens because hydrogen isotope effects are the **largest** (mass ratio 1:2:3 for H:D:T), producing measurable ΔTc.

---

## 4. Revised Data Pipeline Design

### Phase 1: Consistency Validation (No New Training Data)

Use isotope tokens + `estimate_isotope_effect()` to **validate** model behavior:
1. Encode known superconductors with natural isotope composition
2. Encode the same formula with isotope substitution (e.g., {18}O)
3. Verify the model's z-space places them near each other
4. Verify Tc head predicts consistent values (within ΔTc tolerance)

This tests whether the isotope embeddings learned meaningful structure.

### Phase 2: Literature-Mined Isotope Data (Small, High Quality)

Curate ~50–100 experimental data points from isotope effect studies:
- Material formula with specific isotope
- Measured Tc
- Measured α exponent
- Source citation

Store as `data/isotope_effect_measurements.json`. Use as validation holdout, not training data.

### Phase 3: BCS-Constrained Augmentation (Moderate Value)

For the ~500 conventional (BCS) superconductors in the dataset where α ≈ 0.5 is justified:
1. Generate all stable isotope variants
2. Calculate Tc using measured or estimated α
3. Add to training set with a **confidence flag** (synthetic, BCS-justified)
4. Weight synthetic samples lower than experimental data (e.g., 0.1x)

### Phase 4: Theory Network Integration (Future)

When Phase 2 of the big-picture vision (Theory Networks) is implemented:
- BCS Theory Network predicts (θ_D, λ, μ*) → Tc via Allen-Dynes
- Isotope substitution modifies θ_D through phonon mass dependence: θ_D ∝ M^(-1/2)
- This gives a **principled** way to generate isotope-variant Tc predictions
- But requires learned or computed (θ_D, λ) for each material — not available today

---

## 5. Honest Assessment

### What We CAN Do Now
- Generate isotope-substituted formulas using V14.0 tokens (done)
- Calculate approximate ΔTc for BCS superconductors using `estimate_isotope_effect()` (done)
- Use isotope variants as **consistency checks** on model behavior
- Mine literature for ~50–100 real isotope effect data points

### What We CANNOT Do
- Calculate Tc from scratch for a novel composition (requires DFT/DFPT pipeline)
- Reliably predict α for materials not in the BCS regime
- Generate meaningful training augmentation from isotope substitution alone (ΔTc too small)
- Apply isotope effect to cuprates or iron-based superconductors without material-specific α data

### What the Isotope Tokens ARE Good For
1. **Representing experimental isotope studies** when that data is available
2. **Encoding the full material space** including isotope variants (the stated V14.0 goal)
3. **Future theory-guided generation** when Theory Networks provide (θ_D, λ) predictions
4. **Validation probes** — checking z-space structure and Tc consistency

---

## References

1. **BCS Original**: Bardeen, Cooper & Schrieffer (1957). "Theory of Superconductivity." Phys. Rev. 108, 1175. doi:10.1103/PhysRev.108.1175

2. **McMillan Formula**: McMillan, W.L. (1968). "Transition Temperature of Strong-Coupled Superconductors." Phys. Rev. 167, 331. doi:10.1103/PhysRev.167.331

3. **Allen-Dynes Formula**: Allen, P.B. & Dynes, R.C. (1975). "Transition temperature of strong-coupled superconductors reanalyzed." Phys. Rev. B 12, 905. doi:10.1103/PhysRevB.12.905

4. **Mercury Isotope Effect**: Maxwell, E. (1950). "Isotope Effect in the Superconductivity of Mercury." Phys. Rev. 78, 477. doi:10.1103/PhysRev.78.477

5. **Tin Isotope Effect**: Serin, B., Reynolds, C.A. & Nesbitt, L.B. (1952). "Mass Dependence of the Superconducting Transition Temperature of Mercury." Phys. Rev. 86, 162. doi:10.1103/PhysRev.86.162

6. **MgB₂ Isotope Effect**: Bud'ko, S.L. et al. (2001). "Boron Isotope Effect in Superconducting MgB₂." Phys. Rev. Lett. 86, 1877. doi:10.1103/PhysRevLett.86.1877

7. **MgB₂ First-Principles**: Choi, H.J. et al. (2002). "The origin of the anomalous superconducting properties of MgB₂." Nature 418, 758. doi:10.1038/nature00898

8. **Cuprate Isotope Effect**: Bussmann-Holder, A., Keller, H. & Müller, K.A. (2007). "Evidences for polaron formation in cuprates." PNAS 104, 12673. doi:10.1073/pnas.0611473104

9. **Iron-Based Isotope Effect**: Huang, P.H. & Lin, J.Y. (2019). "Unconventional isotope effect on superconducting transition temperature in iron-based superconductors." Sci. Rep. 9, 5547. doi:10.1038/s41598-019-42041-z

10. **Elemental Superconductors Review**: Webb, G.W., Marsiglio, F. & Hirsch, J.E. (2015). "Superconductivity in the elements, alloys and simple compounds." Physica C 514, 17. doi:10.1016/j.physc.2015.02.037. arXiv:1502.04724

11. **Allen-Dynes Limitations (2025)**: Gao, P., Sanna, A. et al. (2025). "Pushing the limits of conventional superconductivity: Exploring the McMillan-Allen-Dynes equation." Nature Communications. doi:10.1038/s41467-025-63702-w

12. **First-Principles Screening**: Cerqueira, T.F.T. et al. (2023). "Sampling the materials space for conventional superconducting compounds." npj Computational Materials 9, 13. doi:10.1038/s41524-022-00933-1

13. **ML-Accelerated Electron-Phonon**: Choudhary, K. et al. (2024). "Unified graph neural network force-field for the Materials Project." npj Computational Materials 10, 278. doi:10.1038/s41524-024-01475-4

14. **ML Tc Prediction (Composition-Only)**: Qu, H. et al. (2024). "Predicting Superconducting Transition Temperature through Advanced Machine Learning." J. Chem. Inf. Model. doi:10.1021/acs.jcim.4c01137

15. **ML Tc Prediction (Validated)**: Roter, B. & Dordevic, S. (2024). "Predicting and confirming new superconductors via deep learning." Eur. Phys. J. Plus. doi:10.1140/epjp/s13360-024-05947-w

16. **Generative Superconductor Discovery**: Guided Diffusion for Superconductor Discovery (2025). arXiv:2509.25186

17. **Closed-Loop ML Discovery**: Wines, D. et al. (2023). "Inverse design of next-generation superconductors using data-driven deep generative models." npj Computational Materials 9, 196. doi:10.1038/s41524-023-01131-3
