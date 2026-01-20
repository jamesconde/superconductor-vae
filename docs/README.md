# Superconductor VAE Documentation

This folder contains technical documentation for the Superconductor Formula VAE project.

## Documents

### Training

| Document | Description |
|----------|-------------|
| [COMPUTATIONAL_EFFICIENCY.md](COMPUTATIONAL_EFFICIENCY.md) | Hardware benchmarks, throughput analysis, and optimization guide |
| [ENTROPY_MAINTENANCE.md](ENTROPY_MAINTENANCE.md) | Entropy maintenance strategies for REINFORCE training |

### Evaluation

| Document | Description |
|----------|-------------|
| (Coming soon) | Evaluation methodology and holdout set documentation |

### Theory

| Document | Description |
|----------|-------------|
| [LATENT_SPACE_THEORY.md](LATENT_SPACE_THEORY.md) | Philosophical argument: latent space as effective physical theory |

### Deployment

| Document | Description |
|----------|-------------|
| [DSMLP_DEPLOYMENT.md](DSMLP_DEPLOYMENT.md) | UCSD DSMLP cluster deployment guide |

### Troubleshooting

| Document | Description |
|----------|-------------|
| [CUDA_BUG_INVESTIGATION_20260115.md](troubleshooting/CUDA_BUG_INVESTIGATION_20260115.md) | WSL2/CUDA bug investigation and resolution |

## Quick Links

- **Main README**: [../README.md](../README.md)
- **Training Script**: [../scripts/train_v12_clean.py](../scripts/train_v12_clean.py)
- **Model Code**: [../src/superconductor/models/](../src/superconductor/models/)

## Key Concepts

### Teacher Forcing (TF)

Adaptive teacher forcing adjusts based on model performance:
- `TF = 1.0 - exact_match` (no floor)
- High exact match → more autoregressive practice
- At 87% exact match → TF = 0.13 (87% autoregressive)

### Metrics

| Metric | Mode | Dropout | Purpose |
|--------|------|---------|---------|
| TF-based Exact Match | train() | Active | Training progress (pessimistic) |
| TRUE Autoregressive | eval() | Off | Honest inference performance |

**TRUE Autoregressive is the authoritative metric** - it reflects actual generation capability.

### Entropy Maintenance

The causal entropy scheduler diagnoses plateau causes before intervening:
- Checks if entropy dropped before plateau
- Uses tiered response (strong/weak/none evidence)
- Tracks intervention success for learning

See [ENTROPY_MAINTENANCE.md](ENTROPY_MAINTENANCE.md) for details.

### Checkpoint Resumption (V12.10)

Checkpoints now save full training state for proper resumption:
- **Optimizer state**: AdamW momentum buffers preserved
- **Scheduler state**: LR schedule position preserved
- **Training variables**: `prev_exact`, `best_exact` preserved

This prevents performance regression when resuming training. Without this, resuming could cause:
- LR spike (scheduler resets to initial LR)
- TF ratio jump (prev_exact=0 → TF=1.0)
- Lost optimizer momentum (AdamW restarts from scratch)
