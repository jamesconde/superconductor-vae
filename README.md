# Superconductor Formula VAE

A Variational Autoencoder for generating and discovering novel superconductor chemical formulas.

## Overview

This project implements a VAE that learns to encode and decode superconductor chemical formulas along with their critical temperatures (Tc) and material properties. The model uses:

- **Pointer-Generator Decoder**: Autoregressive decoder with copy mechanism for accurate formula reconstruction
- **Fraction-Aware Tokenization**: Handles complex stoichiometric coefficients (e.g., `Ba2Cu3O6.5`)
- **Magpie Features**: 145 composition-derived features from matminer for rich material representation
- **Memory Tokens**: Learnable memory bank for improved latent-to-sequence generation

## Installation

```bash
pip install -r requirements.txt
```

**GPU Required**: This code is designed to run on CUDA-enabled GPUs.

## Project Structure

```
superconductor-vae/
├── src/
│   └── superconductor/
│       ├── models/          # VAE architectures
│       ├── encoders/        # Composition encoders
│       ├── losses/          # Custom loss functions
│       ├── training/        # Training utilities
│       ├── generation/      # Candidate generation
│       ├── validation/      # Physics validation
│       └── data/            # Data loaders
├── data/                    # Superconductor datasets
├── scripts/
│   └── train.py            # Main training script
└── notebooks/              # Analysis notebooks
```

## Usage

### Training

```bash
PYTHONPATH=src python scripts/train.py
```

### Key Components

- **AttentionBidirectionalVAE**: Main VAE model (`src/superconductor/models/attention_vae.py`)
- **EnhancedAutoRegressiveDecoder**: Pointer-generator decoder (`src/superconductor/models/autoregressive_decoder.py`)
- **CompositionEncoder**: Element-aware encoding (`src/superconductor/encoders/composition_encoder.py`)

## Data

The dataset includes superconductor formulas with:
- Chemical formula (tokenized)
- Critical temperature (Tc)
- 145 Magpie composition features

**Holdout Set**: `data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json` contains 45 superconductors (5 from each family) reserved for generative evaluation.

## Evaluation Philosophy

This VAE is evaluated for **generative capability**, not standard train/val/test accuracy:
1. Train to high reconstruction accuracy (~90%+ exact formula match)
2. Test generalization by probing latent space to generate held-out superconductors
3. Success = generating known superconductors the model never saw during training

## Citation

If you use this code, please cite:
```
@software{superconductor_vae,
  title = {Superconductor Formula VAE},
  year = {2025},
  author = {James Conde}
}
```
