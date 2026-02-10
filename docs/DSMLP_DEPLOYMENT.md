# UCSD DSMLP Deployment Guide

This guide explains how to run Multi-Task Superconductor Generator training on UCSD's Data Science/Machine Learning Platform (DSMLP).

## Quick Start

### 1. SSH to DSMLP

```bash
ssh YOUR_USERNAME@dsmlp-login.ucsd.edu
```

### 2. Launch a GPU Pod

```bash
# Standard GPU pod (1 GPU, 32GB RAM)
launch-scipy-ml.sh -g 1 -m 32

# Or with specific container and more resources
launch.sh -i ucsdets/scipy-ml-notebook:2024.1-cuda -g 1 -m 64 -c 8
```

**Options:**
- `-g N`: Number of GPUs (1-4 typically)
- `-m N`: Memory in GB
- `-c N`: CPU cores

### 3. Clone and Setup

```bash
git clone https://github.com/jamesconde/superconductor-vae.git
cd superconductor-vae
bash scripts/dsmlp_setup.sh
```

### 4. Upload Data (if not in repo)

The training data files are required:
- `data/processed/supercon_fractions.csv`
- `data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json`

Use `scp` to upload:
```bash
# From your local machine
scp data/processed/supercon_fractions.csv YOUR_USERNAME@dsmlp-login.ucsd.edu:~/superconductor-vae/data/processed/
```

### 5. Start Training

```bash
# Interactive (will stop if you disconnect)
python3 scripts/train_v12_clean.py

# Background (survives disconnect) - RECOMMENDED
nohup python3 scripts/train_v12_clean.py > training.log 2>&1 &
tail -f training.log  # Monitor progress
```

## JupyterLab Alternative

You can also use the web interface at https://datahub.ucsd.edu/

1. Log in with your UCSD credentials
2. Open a Terminal from the Launcher
3. Follow steps 3-5 above

**Note:** JupyterLab sessions may have shorter time limits than SSH pods.

## Resuming Training

If training is interrupted, it will save an `checkpoint_interrupt.pt`. To resume:

1. Edit `scripts/train_v12_clean.py`
2. Change `resume_checkpoint` to point to your checkpoint:
   ```python
   'resume_checkpoint': 'outputs/checkpoint_interrupt.pt',
   ```
3. Restart training

## Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch training log
tail -f training.log

# Check training progress (last 20 lines)
tail -20 training.log
```

## Resource Recommendations

| Model Size | GPU Memory | RAM | Expected Epoch Time |
|------------|------------|-----|---------------------|
| Current (108M params) | 8GB+ | 32GB | ~10-12 min |

## Troubleshooting

### "CUDA out of memory"
- Request a larger GPU: `-g 1` with a V100 or A100
- Reduce batch size in TRAIN_CONFIG

### Pod killed unexpectedly
- You may have exceeded time limit
- Use `screen` or `tmux` for long runs:
  ```bash
  screen -S training
  python3 scripts/train_v12_clean.py
  # Ctrl+A, D to detach
  # screen -r training to reattach
  ```

### Missing dependencies
```bash
pip install --user torch pandas numpy
```
