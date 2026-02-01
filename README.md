# Superconductor Formula VAE

A Variational Autoencoder for generating and discovering novel superconductor chemical formulas.

## Vision

The long-term goal of this project is to build a **unified latent representation of superconductivity**. By encoding superconductor compositions, critical temperatures, and material properties into a continuous latent space, we aim to:

1. **Discover structure** in how different superconductor families relate to each other
2. **Generate novel candidates** by intelligently navigating the latent space
3. **Eventually integrate theoretical models** (BCS theory, Eliashberg equations) to create a latent space that captures both empirical data and theoretical understanding

This is a stepping stone toward AI-assisted materials discovery where the model doesn't just memorize formulas, but learns the underlying chemistry of superconductivity.

## Overview

This project implements a VAE that learns to encode and decode superconductor chemical formulas along with their critical temperatures (Tc) and material properties:

- **Pointer-Generator Decoder**: Autoregressive decoder with copy mechanism for accurate formula reconstruction
- **Fraction-Aware Tokenization**: Handles complex stoichiometric coefficients (e.g., `Ba2Cu3O6.5`, `Ag(1/500)Al(499/500)`)
- **Magpie Features**: 145 composition-derived features from matminer for rich material representation
- **Memory Tokens**: Learnable memory bank for improved latent-to-sequence generation
- **Full Materials Encoder**: Three-branch architecture fusing element embeddings, Tc, and Magpie features

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+ with CUDA support
- GPU with 8GB+ VRAM recommended

## Project Structure

```
superconductor-vae/
├── src/
│   └── superconductor/
│       ├── models/          # VAE architectures (FullMaterialsVAE, decoders)
│       ├── encoders/        # Composition & element encoders
│       ├── losses/          # REINFORCE, semantic, physics losses
│       ├── training/        # Mastery sampling, KL annealing
│       ├── generation/      # Latent space probing & candidate generation
│       ├── validation/      # Physics-based validation
│       └── data/            # Dataset loaders
├── data/
│   ├── raw/                 # Original superconductor data (16,521 samples)
│   └── processed/           # Fraction-notation formulas
├── scripts/
│   ├── train.py             # Legacy training script
│   └── train_v12_clean.py   # Main training script (V12, recommended)
└── notebooks/
    └── train_colab.ipynb    # Google Colab training notebook
```

## Usage

### Training (Local)

```bash
PYTHONPATH=src python scripts/train_v12_clean.py
```

Training typically requires:
- ~2000 epochs to reach 90%+ reconstruction accuracy
- 8-12 hours on a modern GPU (RTX 3080/4080 or better)

### Training (Google Colab)

A ready-to-use notebook is provided at `notebooks/train_colab.ipynb`. Upload the repo to Google Drive, open the notebook in Colab, and run the cells in order. Checkpoints are saved to Drive and persist across sessions.

**Remote monitoring**: The notebook supports optional live metrics logging via GitHub Gist. Set `GIST_ID` in Cell 2 and store a GitHub token in Colab secrets to push training progress (loss, exact match, accuracy, etc.) to a gist on every checkpoint save. The current training log is viewable at:

https://gist.github.com/jamesconde/acceed7daef4d6893801cc7337531b68

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| FullMaterialsVAE | `models/attention_vae.py` | Main encoder with element attention + Magpie fusion |
| EnhancedTransformerDecoder | `models/autoregressive_decoder.py` | Pointer-generator decoder with memory tokens |
| CompositionEncoder | `encoders/composition_encoder.py` | Parses formulas to element-fraction vectors |

## Data

The dataset includes 16,521 superconductor formulas with:
- Chemical formula (in both decimal and fraction notation)
- Critical temperature (Tc) ranging from 0K to 185K
- 145 Magpie composition-derived features
- Superconductor family classification (YBCO, LSCO, Bi-cuprate, etc.)

### Generative Holdout Set

`data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json` contains **45 superconductors** (5 from each of 9 families) that are **never used in training**. These are the true test of generalization.

## Evaluation Philosophy

**This is NOT a standard train/val/test setup.**

Traditional ML evaluation uses random splits, but for chemical formulas this is problematic:
- Similar formulas (e.g., `YBa2Cu3O6.9` and `YBa2Cu3O7`) can leak between splits
- High test accuracy doesn't prove the model learned superconductor chemistry

### Our Approach: Generative Evaluation

1. **Train** the VAE on all data EXCEPT the 45 holdout superconductors
2. **Achieve high reconstruction** (90%+ exact formula match on training data)
3. **Probe the latent space** by:
   - Interpolating between known superconductors
   - Adding structured perturbations to latent vectors
   - Sampling near high-Tc regions
4. **Success criteria**: Can we generate any of the 45 holdout superconductors the model has never seen?

If the model can produce `HgBa2Ca2Cu3O8` (a 134K superconductor in the holdout set) by navigating its latent space, it demonstrates genuine understanding of superconductor chemistry—not just memorization.

### Why This Matters

A model that scores 95% on random test splits might just be memorizing string patterns. A model that generates unseen superconductors from latent space perturbations has learned something meaningful about what makes a superconductor.

## Future Directions

- **Theory Networks**: Integrate BCS parameters and Eliashberg spectral functions as additional latent space constraints
- **3D Voxel Representations**: Add crystal structure information for materials where it's available
- **Guided Generation**: Use Tc prediction to navigate toward high-temperature regions of latent space
- **Experimental Validation**: Partner with labs to synthesize and test generated candidates

## Contributing

This is an active research project. Contributions welcome in:
- Novel latent space probing strategies
- Additional physics-based losses
- Integration with materials databases (AFLOW, Materials Project)

## Citation

If you use this code, please cite:
```bibtex
@software{superconductor_vae,
  title = {Superconductor Formula VAE},
  author = {James Conde},
  year = {2025},
  url = {https://github.com/jamesconde/superconductor-vae}
}
```

## License

MIT License
