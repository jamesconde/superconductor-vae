# Tc Prediction Benchmarks: ML State of the Art & First-Principles Theory

A survey of machine learning and theoretical approaches to superconductor critical temperature (Tc) prediction, compiled to contextualize our model's performance and identify competitive targets.

Last updated: 2026-02-11

---

## 1. Composition-Only ML Models

These models predict Tc from chemical formula alone (no crystal structure required). This is the category our superconductor VAE falls into.

### Foundational Work (2018-2021)

| Paper | Year | Model | Dataset | Features | RMSE (K) | R² | DOI |
|-------|------|-------|---------|----------|----------|-----|-----|
| Hamidieh | 2018 | XGBoost | SuperCon (21,263) | 81 elemental properties | 9.5 | 0.92 | [10.1016/j.commatsci.2018.07.052](https://doi.org/10.1016/j.commatsci.2018.07.052) |
| Stanev et al. | 2018 | Random Forest | SuperCon (~12,000) | 145 Magpie + 26 AFLOW | — | 0.885 | [10.1038/s41524-018-0085-8](https://doi.org/10.1038/s41524-018-0085-8) |
| Konno et al. | 2021 | CNN | SuperCon (~13,000) | Periodic table tensor (4x32x7) | — | 0.92 | [10.1103/PhysRevB.103.014509](https://doi.org/10.1103/PhysRevB.103.014509) |

**Hamidieh (2018)** established the XGBoost baseline with 81 hand-crafted features from elemental properties. The SuperCon dataset and feature set became a standard benchmark (available on UCI ML Repository).

**Stanev et al. (2018)** introduced Magpie descriptors (the same 145-feature set our model uses) as the standard composition-based feature representation. Combined SC/non-SC classification with Tc regression, predicting 30+ new candidates from ICSD.

**Konno et al. (2021)** pioneered a "reading the periodic table" approach where a CNN learns chemical composition patterns from a periodic-table-shaped tensor input, avoiding hand-crafted features entirely.

### Current State of the Art (2023-2026)

| Paper | Year | Model | MAE (K) | RMSE (K) | R² | DOI |
|-------|------|-------|---------|----------|-----|-----|
| **GBFS workflow** | 2024 | LightGBM | **3.54** | **6.57** | **0.945** | [10.1021/acs.jcim.4c01137](https://doi.org/10.1021/acs.jcim.4c01137) |
| **CatBoost + Jabir** | 2024 | CatBoost | — | **6.45** | **0.952** | [10.1038/s41598-024-54440-y](https://doi.org/10.1038/s41598-024-54440-y) |
| **MoE + Dopant Recognition** | 2024 | Mixture of Experts | — | — | **0.962** | [10.1021/acsami.4c11997](https://doi.org/10.1021/acsami.4c11997) |
| **Optuna-Stacking Ensemble** | 2023 | Lasso+KNN+XGB+ExtraTrees | — | — | 0.931 | [10.1021/acsomega.2c06324](https://doi.org/10.1021/acsomega.2c06324) |
| **HNN** | 2025 | Hierarchical NN | MAPE <6% | — | 0.956 | [10.15302/frontphys.2025.014205](https://doi.org/10.15302/frontphys.2025.014205) |
| **Attention-based DL** | 2025 | Attention network | 4.6 | — | 0.925 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0927025625004720) |

**Key takeaways:**

- Best composition-only MAE: **~3.5K** (GBFS, 2024)
- Best composition-only R²: **~0.96** (MoE with dopant recognition, 2024)
- Best composition-only RMSE: **~6.5K** (multiple papers, 2024)
- Dopant-aware and family-specific models outperform generic approaches

### Important Caveats

1. **R² inflation**: SuperCon has a heavily skewed Tc distribution (many materials near 0-10K). A model predicting "low Tc" for everything achieves deceptively high R².

2. **Train/test leakage**: SuperCon contains many closely related compositions (e.g., YBa₂Cu₃O₇₋ₓ with varying x). Without careful deduplication, models achieve artificially high scores. The GBFS paper specifically addresses this with ~12,000 unique compositions from ~16,400 records.

3. **High-Tc tail matters most**: For materials discovery, accuracy on high-Tc materials (>30K) matters far more than accuracy on the bulk of low-Tc materials. Several papers report that models perform worst on high-Tc cuprates and hydrides — precisely the materials of greatest interest.

---

## 2. Structure-Aware ML Models

These models require crystal structure information, limiting them to known or DFT-relaxed structures. They achieve better accuracy but cannot screen arbitrary hypothetical compositions.

| Paper | Year | Model | MAE (K) | Notes | DOI |
|-------|------|-------|---------|-------|-----|
| **BEE-NET** | 2026 | Equivariant GNN ensemble | **0.87** | MAE vs DFT-computed Tc, not experiment | [10.1038/s41524-026-01964-8](https://doi.org/10.1038/s41524-026-01964-8) |
| **BETE-NET** | 2024 | Bootstrapped Ensemble GNN | 2.1-2.5 | Predicts Eliashberg spectral function | [10.1038/s41524-024-01475-4](https://doi.org/10.1038/s41524-024-01475-4) |
| **S2SNet** | 2022 | Pre-trained attention | — | AUC 0.92 (classification) | [arXiv:2306.16270](https://arxiv.org/abs/2306.16270) |

**BETE-NET** is notable for its physics-informed approach: it predicts the Eliashberg spectral function alpha²F(omega) from crystal structure, then derives Tc analytically. This bridges ML and theory.

**BEE-NET** achieves 0.87K MAE but this is measured against DFT-computed Tc (Allen-Dynes), not experimental Tc. The distinction is important — DFT Tc itself has ~15-30% error vs experiment for strong-coupling materials.

---

## 3. First-Principles Theoretical Predictions

### BCS / Conventional Superconductors

| Method | Typical Error | Applicability | Key Reference |
|--------|--------------|---------------|---------------|
| **Allen-Dynes formula** | ~3-5K (low-Tc); ~15-30% (high-Tc) | Phonon-mediated only; requires mu* fitting | Allen & Dynes, PRB 12, 905 (1975) |
| **Full Eliashberg** (ab initio) | ~15% (~5-10K for MgB₂-class) | Phonon-mediated; computationally expensive | — |
| **SCDFT** (refined) | "Excellent agreement" | Parameter-free; limited benchmarks | [10.1103/PhysRevLett.125.057001](https://doi.org/10.1103/PhysRevLett.125.057001) |

**Allen-Dynes** is the formula implemented in our BCSTheoryLoss (V12.25). It is semi-empirical: Tc = (omega_log/1.2) * exp(-1.04(1+lambda)/(lambda - mu*(1+0.62*lambda))) * f1 * f2. The Coulomb pseudopotential mu* is typically a fitting parameter (0.10-0.15), introducing uncertainty. For MgB₂ (lambda~0.7), the formula underestimates Tc by ~15%.

**Full Eliashberg theory** solves frequency-dependent gap equations numerically from the electron-phonon spectral function alpha²F(omega). For H₃S at 203K, early predictions varied wildly; including vertex corrections, anharmonicity, and DOS energy dependence brings predictions into agreement with experiment. Error sources include Brillouin zone sampling, exchange-correlation functional choice, and Fermi level sensitivity (a 250 meV shift causes 20-35% error in lambda).

**SCDFT** (Sanna et al., PRL 2020) is parameter-free (no mu* fitting needed) and achieves excellent agreement for Al, Ta, MgB₂, and H₃S with refined functionals. However, benchmarks remain limited.

### Unconventional Superconductors

**There is no reliable first-principles method for predicting Tc of unconventional superconductors.**

Cuprate, iron-based, and heavy fermion superconductors are not electron-phonon mediated. The BCS/Eliashberg framework does not apply to them.

| Method | Status for Unconventional SCs |
|--------|-------------------------------|
| **DMFT** (Dynamical Mean-Field Theory) | Qualitative at best; Tc extremely sensitive to doping, correlation strength, and method details |
| **Cluster DMFT + Eliashberg** | Can show d-wave pairing in the Hubbard model; quantitative Tc prediction unreliable |
| **DFT+U** | Can predict trends but not absolute Tc values |

This is a fundamental advantage of ML approaches: they can capture empirical patterns across all superconductor families that theory cannot yet explain. Our model's ability to predict Tc for cuprates (where no first-principles method works) is scientifically valuable.

---

## 4. Generative Models for Superconductor Discovery

Our superconductor VAE operates in a less crowded but rapidly growing field.

### Crystal Structure Generators

| Paper | Year | Approach | Key Result | DOI |
|-------|------|----------|------------|-----|
| **CDVAE** (Xie et al.) | 2022 | SE(3)-invariant VAE + diffusion | Generates 3D crystal structures | [arXiv:2110.06197](https://arxiv.org/abs/2110.06197) |
| **DiffCSP** (guided) | 2025 | Diffusion, pretrained on 2M structures | 773 candidates with DFT-confirmed Tc>5K | [arXiv:2509.25186](https://arxiv.org/abs/2509.25186) |
| **InvDesFlow-AL** | 2025 | Active learning + diffusion | Claims Li₂AuH₆ at Tc=140K ambient | [10.1038/s41524-025-01830-z](https://doi.org/10.1038/s41524-025-01830-z) |
| **BEE-NET pipeline** | 2026 | Elemental substitution + ML screening | 741 stable compounds with Tc>5K from 1.3M candidates | [10.1038/s41524-026-01964-8](https://doi.org/10.1038/s41524-026-01964-8) |
| **Conditional diffusion** | 2024 | Conditional diffusion for SC families | Generates hypothetical new families | [10.1038/s41598-024-61040-3](https://doi.org/10.1038/s41598-024-61040-3) |

### Our Approach: Composition-Level VAE

Our model generates chemical formulas (not crystal structures) with predicted Tc from a latent space trained on 46,645 samples (23,451 SC + 23,194 non-SC). This is architecturally distinct from the structure-based approaches above:

**Advantages:**
- Can screen any hypothetical formula without requiring crystal structure
- Faster evaluation than structure-based methods (no DFT relaxation needed)
- Latent space enables interpolation between known superconductor families
- Joint formula+Tc training means the latent space encodes what enables superconductivity

**Limitations:**
- Cannot predict structure-dependent properties (Fermi surface topology, phonon spectra)
- Tc prediction is empirical, not physics-derived
- Generation quality depends on training data coverage

### Benchmark Datasets

| Dataset | Year | Size | Description | Reference |
|---------|------|------|-------------|-----------|
| **SuperCon** (NIMS) | ongoing | ~33,000+ | Largest experimental SC database | [supercon.nims.go.jp](https://supercon.nims.go.jp) |
| **HTSC-2025** | 2025 | 155 | Ambient-pressure high-Tc predictions (avg Tc=30.3K) | [arXiv:2506.03837](https://arxiv.org/abs/2506.03837) |
| **Our dataset** | 2026 | 46,645 | SuperCon + non-SC contrastive pairs + Magpie features | Internal |

---

## 5. Competitive Targets for Our Model

### Tc Prediction Accuracy

| Tier | Overall MAE (K) | R² | 100K+ MAE (K) | Assessment |
|------|-----------------|-----|----------------|------------|
| Below standard | >10 | <0.90 | >15 | 2018-era performance |
| Acceptable | 7-10 | 0.90-0.93 | 10-15 | Publishable but not competitive |
| **Competitive** | **3.5-7** | **0.93-0.96** | **5-10** | Matches recent ensemble methods |
| **Strong** | **2-3.5** | **>0.96** | **3-5** | Matches 2024-2025 SOTA |
| State of the art | <2 | >0.97 | <3 | Requires structure-aware methods |

### Our Current Performance (Epoch 2345)

| Tc Range | MAE (K) | Samples | Assessment |
|----------|---------|---------|------------|
| 0-10K | **0.1** | 9,744 | Near-perfect |
| 10-50K | **0.1** | 7,083 | Near-perfect |
| 50-100K | **2.2** | 5,435 | Competitive with structure-aware models |
| 100K+ | **6.5** | 947 | Needs improvement — target <3K |
| **Overall** | **0.8** | 23,209 | Strong (but weighted by many easy low-Tc samples) |

The overall 0.8K MAE is misleadingly good because 72% of samples are below 50K where the model is near-perfect. The 100K+ bucket (6.5K MAE, 947 samples) is the priority for V12.26 loss rebalancing.

### Generation Quality (unique to generative models)

These metrics have no direct comparison in the regression literature since most published models are pure predictors, not generators:

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Formula exact match (TF) | 87% | >95% | Teacher-forced; measures reconstruction quality |
| Formula exact match (autoregressive) | 58% | >80% | True generation quality; limited by TF=1.0 exposure bias |
| Valid formula rate | — | >90% | Charge balance, stoichiometric feasibility |
| High-Tc region sampling | — | — | Does latent space produce plausible high-Tc candidates? |

### Key Competitive Differentiators

1. **Multi-family coverage**: Most published models treat Tc regression as a single task. Our model's theory losses (V12.25) encode family-specific physics — BCS (Allen-Dynes), cuprate (Presland dome), iron-based (VEC constraints), heavy fermion (log-normal prior), organic (soft caps). This family awareness is rare in the literature.

2. **Generative capability**: Pure regression models answer "what is the Tc of this material?" Our model answers "what materials could have Tc > X?" This is the more valuable question for discovery.

3. **Composition-only with Tc**: By avoiding crystal structure requirements, our model can screen hypothetical formulas that have never been synthesized. Structure-based models (BETE-NET, BEE-NET, DiffCSP) are limited to known or computationally predicted structures.

4. **Contrastive training**: The SC vs non-SC contrastive learning (V12.5+) teaches the latent space what distinguishes superconductors from non-superconductors — a capability that pure Tc regression models lack.

---

## 6. References

### Foundational ML for Tc

[Hamidieh2018] S. Hamidieh, "A Data-Driven Statistical Model for Predicting the Critical Temperature of a Superconductor," *Computational Materials Science* **154**, 346-354 (2018). DOI: [10.1016/j.commatsci.2018.07.052](https://doi.org/10.1016/j.commatsci.2018.07.052)

[Stanev2018] V. Stanev, C. Oses, A. G. Kusne, E. Rodriguez, J. Paglione, S. Curtarolo, and I. Takeuchi, "Machine Learning Modeling of Superconducting Critical Temperature," *npj Computational Materials* **4**, 29 (2018). DOI: [10.1038/s41524-018-0085-8](https://doi.org/10.1038/s41524-018-0085-8)

[Konno2021] T. Konno, H. Kurokawa, F. Nabeshima, Y. Sakishita, R. Ogawa, I. Hosako, and A. Maeda, "Deep Learning Model for Finding New Superconductors," *Physical Review B* **103**, 014509 (2021). DOI: [10.1103/PhysRevB.103.014509](https://doi.org/10.1103/PhysRevB.103.014509)

### Current SOTA — Composition-Only

[GBFS2024] "Gradient-Boosted Feature Selection Workflow for Predicting Superconducting Tc," *Journal of Chemical Information and Modeling* (2024). DOI: [10.1021/acs.jcim.4c01137](https://doi.org/10.1021/acs.jcim.4c01137)

[CatBoost2024] "CatBoost with Jabir Atomic Features for Superconductor Tc Prediction," *Scientific Reports* (2024). DOI: [10.1038/s41598-024-54440-y](https://doi.org/10.1038/s41598-024-54440-y)

[MoE2024] "Mixture of Experts with Dopant Recognition for Superconductor Tc," *ACS Applied Materials & Interfaces* (2024). DOI: [10.1021/acsami.4c11997](https://doi.org/10.1021/acsami.4c11997)

[Stacking2023] "Optuna-Stacking Ensemble for Superconductor Tc Prediction," *ACS Omega* (2023). DOI: [10.1021/acsomega.2c06324](https://doi.org/10.1021/acsomega.2c06324)

[HNN2025] "Hierarchical Neural Network with Universal Descriptors for Superconductor Tc," *Frontiers in Physics* (2025). DOI: [10.15302/frontphys.2025.014205](https://doi.org/10.15302/frontphys.2025.014205)

### Structure-Aware Models

[BETENET2024] "Bootstrapped Ensemble GNN for Tc from Eliashberg Spectral Function," *npj Computational Materials* (2024). DOI: [10.1038/s41524-024-01475-4](https://doi.org/10.1038/s41524-024-01475-4)

[BEENET2026] "Equivariant GNN Ensemble for Superconductor Screening," *npj Computational Materials* (2026). DOI: [10.1038/s41524-026-01964-8](https://doi.org/10.1038/s41524-026-01964-8)

### Generative Models

[CDVAE2022] T. Xie, X. Fu, O.-E. Ganea, R. Barzilay, and T. Jaakkola, "Crystal Diffusion Variational Autoencoder for Periodic Material Generation," *ICLR 2022*. [arXiv:2110.06197](https://arxiv.org/abs/2110.06197)

[DiffCSP2025] "Guided Diffusion for Superconductor Discovery," (2025). [arXiv:2509.25186](https://arxiv.org/abs/2509.25186)

[InvDesFlow2025] "InvDesFlow-AL: Active Learning Inverse Design for Superconductors," *npj Computational Materials* (2025). DOI: [10.1038/s41524-025-01830-z](https://doi.org/10.1038/s41524-025-01830-z)

[DiffSC2024] "Diffusion Models for Conditional Generation of Superconductors," *Scientific Reports* (2024). DOI: [10.1038/s41598-024-61040-3](https://doi.org/10.1038/s41598-024-61040-3)

### First-Principles Theory

[AllenDynes1975] P. B. Allen and R. C. Dynes, "Transition temperature of strong-coupled superconductors reanalyzed," *Physical Review B* **12**, 905-922 (1975). DOI: [10.1103/PhysRevB.12.905](https://doi.org/10.1103/PhysRevB.12.905)

[Sanna2020] A. Sanna, C. Pellegrini, and E. K. U. Gross, "Combining Eliashberg Theory with Density Functional Theory for the Accurate Prediction of Superconducting Transition Temperatures and Gap Functions," *Physical Review Letters* **125**, 057001 (2020). DOI: [10.1103/PhysRevLett.125.057001](https://doi.org/10.1103/PhysRevLett.125.057001)

### Benchmark Datasets

[HTSC2025] "HTSC-2025: Benchmark Dataset of Predicted Ambient-Pressure High-Tc Superconductors," (2025). [arXiv:2506.03837](https://arxiv.org/abs/2506.03837)
