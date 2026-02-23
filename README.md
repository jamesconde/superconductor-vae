# Multi-Task Superconductor Generator

A multi-task encoder-decoder for learning compressed representations of superconducting materials and generating novel superconductor chemical formulas. Currently at **V14.3** (~108M parameters).

> **Note on naming**: The code uses class names like `FullMaterialsVAE` and `AttentionVAEEncoder`
> from the project's origins as a Variational Autoencoder. The architecture has since evolved into
> a deterministic multi-task encoder-decoder — there is no stochastic sampling or KL divergence.
> Code class names are kept for backward compatibility.

## Vision

The long-term goal of this project is to build a **unified latent representation of superconductivity**. By encoding superconductor compositions, critical temperatures, and material properties into a continuous latent space, we aim to:

1. **Discover structure** in how different superconductor families relate to each other
2. **Generate novel candidates** by intelligently navigating the latent space
3. **Eventually integrate theoretical models** (BCS theory, Eliashberg equations) to create a latent space that captures both empirical data and theoretical understanding

This is a stepping stone toward AI-assisted materials discovery where the model doesn't just memorize formulas, but learns the underlying chemistry of superconductivity.

## Architecture (V14.3)

### Encoder: `FullMaterialsVAE`

Three-branch encoder fusing composition, material properties, and critical temperature into a 2048-dim deterministic latent space:

- **Element Branch**: Learnable element embeddings (118 elements + 291 isotopes) with multi-head attention, weighted by stoichiometric fractions
- **Magpie Branch**: MLP over 145 composition-derived material features from matminer
- **Tc Branch**: Critical temperature embedding

The encoder also produces auxiliary predictions used for multi-task training:
- Tc regression, Magpie reconstruction, SC/non-SC classification
- High-pressure flag, Tc bucket classification, element count
- Stoichiometry fractions, hierarchical family classification
- Competence score (model confidence)

### Decoder: `EnhancedTransformerDecoder`

12-layer transformer decoder with cross-attention to encoder latent space:

- **24 cross-attention memory tokens**: 16 latent + 4 stoichiometry + 4 encoder-head conditioning
- **Semantic fraction tokenization**: Single `FRAC:p/q` tokens (e.g., `FRAC:1/3`) instead of digit-by-digit encoding. Vocab: ~4,700 tokens.
- **Token type classifier** (V14.3): 5-class head (element/integer/fraction/special/EOS) that predicts the grammatical type of each position. Enables hard vocab masking at inference to eliminate type confusion errors.
- **Stop head**: Learned EOS prediction for sequence termination
- **KV-cache generation**: Efficient autoregressive inference

### Training: `CombinedLossWithREINFORCE`

17 differentiable loss terms plus REINFORCE reward signals (see `docs/LOSS_INVENTORY.md`):

- Formula cross-entropy, Tc MSE, Magpie MSE, stoichiometry MSE
- SC classification (BCE), high-pressure (BCE), Tc bucket (CE), family (CE)
- Token type auxiliary (CE), stop prediction (BCE), element count (MSE)
- Competence-weighted gating, label smoothing regularization
- REINFORCE with RLOO baseline for autoregressive generation quality
- SCST (Self-Critical Sequence Training) as alternative RL method
- Dynamic loss weights via curriculum scheduling and smart loss skipping

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
├── src/superconductor/
│   ├── models/              # FullMaterialsVAE, EnhancedTransformerDecoder
│   ├── encoders/            # Composition & element encoders
│   ├── tokenizer/           # FractionAwareTokenizer (semantic FRAC:p/q tokens)
│   ├── losses/              # Physics, semantic, REINFORCE losses
│   ├── training/            # Mastery sampling, soft token sampling
│   ├── generation/          # Latent space probing & candidate generation
│   ├── validation/          # Physics-based formula validation
│   ├── postprocessing/      # Output cleanup
│   ├── utils/               # Shared utilities
│   └── data/                # Dataset loaders
├── data/
│   ├── raw/                 # Original SuperCon data
│   ├── processed/           # Contrastive dataset (52K+ samples)
│   └── GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json
├── scripts/
│   ├── train_v12_clean.py   # Main training script (V14.3)
│   ├── holdout/             # Generative holdout evaluation scripts
│   └── migrate_*.py         # Checkpoint migration utilities
├── notebooks/
│   ├── train_colab.ipynb    # Google Colab training notebook
│   └── update_from_github.ipynb  # Sync Colab with latest code
├── docs/                    # Design docs, training records, loss inventory
└── scratch/                 # Debug scripts, analysis, reports
```

## Usage

### Training (Local)

```bash
PYTHONPATH=src python scripts/train_v12_clean.py
```

Training auto-resumes from the best checkpoint by default (`resume_checkpoint='auto'`). This prefers `checkpoint_best.pt`, then falls back to the highest-numbered `checkpoint_epoch_*.pt`. Set to `None` to train from scratch.

### Training (Google Colab)

A ready-to-use notebook is provided at `notebooks/train_colab.ipynb`. Upload the repo to Google Drive, open the notebook in Colab, and run the cells in order. Checkpoints are saved to Drive and persist across sessions.

**Remote monitoring**: The notebook supports optional live metrics logging via GitHub Gist. Set `GIST_ID` in Cell 2 and store a GitHub token in Colab secrets to push training progress (loss, exact match, accuracy, etc.) to a gist on every checkpoint save. The current training log is viewable at:

https://gist.github.com/jamesconde/acceed7daef4d6893801cc7337531b68

### Training (UCSD DSMLP)

See `docs/DSMLP_DEPLOYMENT.md` for instructions on running on UCSD's Data Science/Machine Learning Platform with GPU pods.

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `FullMaterialsVAE` | `src/superconductor/models/attention_vae.py` | Three-branch encoder with element attention + Magpie fusion |
| `EnhancedTransformerDecoder` | `src/superconductor/models/autoregressive_decoder.py` | 12-layer transformer decoder with memory tokens, token type head, stop head |
| `FractionAwareTokenizer` | `src/superconductor/tokenizer/fraction_tokenizer.py` | Semantic `FRAC:p/q` tokenization, token type masks, isotope support |
| `CombinedLossWithREINFORCE` | `scripts/train_v12_clean.py` | 17-term multi-task loss with RLOO/SCST reinforcement learning |
| `CompositionEncoder` | `src/superconductor/encoders/composition_encoder.py` | Parses formulas to element-fraction vectors |

## Data

The contrastive dataset (`data/processed/supercon_fractions_contrastive.csv`) contains **52,800+ samples**:
- **~26,400 superconductors** with Tc ranging from 0K to 185K
- **~26,400 non-superconductors** for contrastive learning (SC/non-SC classification)
- Chemical formula in fraction notation
- 145 Magpie composition-derived features
- Superconductor family classification (YBCO, LSCO, Bi-cuprate, Hg-cuprate, Tl-cuprate, Iron-based, MgB2, Conventional, Other)
- High-pressure flag

Data sources include SuperCon, JARVIS, MDR, HTSC-2025, 3DSC, SODNet, and NEMAD databases.

### Generative Holdout Set

`data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json` contains **45 superconductors** (5 from each of 9 families) that are **never used in training**. These are the true test of generalization.

## Evaluation Philosophy

**This is NOT a standard train/val/test setup.**

Traditional ML evaluation uses random splits, but for chemical formulas this is problematic:
- Similar formulas (e.g., `YBa2Cu3O6.9` and `YBa2Cu3O7`) can leak between splits
- High test accuracy doesn't prove the model learned superconductor chemistry

### Our Approach: Generative Evaluation

1. **Train** the model on all data EXCEPT the 45 holdout superconductors
2. **Achieve high reconstruction** (95%+ exact formula match on training data)
3. **Probe the latent space** by:
   - Interpolating between known superconductors
   - Adding structured perturbations to latent vectors
   - Sampling near high-Tc regions
   - Self-consistency checks across all encoder heads
4. **Success criteria**: Can we generate any of the 45 holdout superconductors the model has never seen?

If the model can produce `HgBa2Ca2Cu3O8` (a 134K superconductor in the holdout set) by navigating its latent space, it demonstrates genuine understanding of superconductor chemistry — not just memorization.

## Version History

| Version | Key Changes |
|---------|-------------|
| V1-V6 | Foundation: GRU decoder, pointer-generator, fraction tokenization |
| V7-V10 | Transformer decoder, REINFORCE (RLOO), discriminative rewards |
| V11 | Skip connections, 16 memory tokens, 2048-dim latent |
| V12 | Full materials encoder (Magpie + Tc input), stoich conditioning, hierarchical family head |
| V13 | Semantic fraction tokenization (`FRAC:p/q`), vocab 148 → 4,355 |
| V14.0 | Isotope token expansion (291 isotopes), vocab → 4,647 |
| V14.1 | Inline isotope init in checkpoint loader, RL auto-scaling |
| V14.3 | Token type classifier head + enriched decoder memory (24 cross-attention tokens) |

See `docs/TRAINING_RECORDS.md` for detailed run history, architecture decisions, and timing.

## Documentation

| Document | Description |
|----------|-------------|
| `docs/TRAINING_RECORDS.md` | Chronological record of all training runs and architecture changes |
| `docs/LOSS_INVENTORY.md` | Complete catalog of all 17 loss terms with weights and purpose |
| `docs/TOKEN_TYPE_CLASSIFIER_V14_3.md` | Design doc for V14.3 token type head + enriched decoder memory |
| `docs/CONTRASTIVE_LEARNING_DESIGN.md` | SC vs non-SC contrastive dataset design |
| `docs/ARCHITECTURE.md` | Model architecture details |
| `docs/SC_CONSTRAINT_ZOO.md` | Physics-based constraint catalog for REINFORCE rewards |
| `docs/DSMLP_DEPLOYMENT.md` | UCSD DSMLP deployment guide |

## Future Directions

- **Theory Networks**: Integrate BCS parameters and Eliashberg spectral functions as additional latent space constraints
- **Latent Space Composition**: Align empirical and theory latent spaces for theory-guided generation
- **Guided Generation**: Use Tc prediction and token type masking to navigate toward high-temperature regions
- **Experimental Validation**: Partner with labs to synthesize and test generated candidates

## Contributing

This is an active research project. Contributions welcome in:
- Novel latent space probing strategies
- Additional physics-based losses
- Integration with materials databases (AFLOW, Materials Project)

## Citation

If you use this code, please cite:
```bibtex
@software{superconductor_generative,
  title = {Multi-Task Superconductor Generator},
  author = {James Conde},
  year = {2025},
  url = {https://github.com/jamesconde/superconductor-vae}
}
```

## License

MIT License
