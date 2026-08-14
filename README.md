# Superconductor VAE — Hydra MAGE

This repository has been made private while intellectual property and commercialization options are being explored. The reports below are shared publicly; the code and model weights remain private.

## About the Project

**Hydra MAGE** (FullMaterialsVAE) is a multi-task generative model (114.8M parameters) for superconductor discovery. It jointly predicts critical temperature (Tc) and reconstructs chemical formulas in a shared latent space, with auxiliary heads for superconductivity, crystallographic family, and magnetic-ordering properties — enabling targeted generation of superconducting candidates and mechanistic interpretability of what the model has learned.

## Technical Report

The full technical report: **[James_Conde_MAGE_Report_updated_figures.pdf](https://github.com/jamesconde/superconductor-vae/blob/main/James_Conde_MAGE_Report_updated_figures.pdf)** (39 pp) — architecture, mechanistic-interpretability findings, steering/ablation studies, and an honest evaluation of the model's generative behaviour.

### Companion supplements

- **[Supplement A — Holdout Recovery](https://github.com/jamesconde/superconductor-vae/blob/main/Supplement_A_Holdout_Recovery.pdf)**: held-out set construction and per-candidate recovery results, with coverage/interpolation caveats.
- **[Supplement B — Candidate Generation](https://github.com/jamesconde/superconductor-vae/blob/main/Supplement_B_Candidate_Generation.pdf)**: the generation pipeline (strategies, validity gates, novelty and plausibility filters) and the resulting shortlist.
- **[Supplement C — Literature Verification](https://github.com/jamesconde/superconductor-vae/blob/main/Supplement_C_Literature_Verification.pdf)**: manual literature search over the shortlist — rediscoveries vs. related-family variants vs. artifacts, and the single deeply-vetted novel hypothesis.

## Access

If you arrived here from a job application or resume and would like to review the code, please reach out — I'm happy to grant temporary read access or walk through the work in a conversation.

**Contact:** jamesconde07@gmail.com
