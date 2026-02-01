#!/bin/bash
# DSMLP Setup Script for Superconductor VAE Training
# UCSD Data Science/Machine Learning Platform
#
# Usage:
#   1. SSH: ssh YOUR_USERNAME@dsmlp-login.ucsd.edu
#   2. Launch GPU pod: launch-scipy-ml.sh -g 1 -m 32
#   3. Run this script: bash dsmlp_setup.sh

set -e

echo "=============================================="
echo "DSMLP Setup for Superconductor VAE"
echo "=============================================="

# Check if we're in a GPU pod
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. You may not be in a GPU pod."
    echo "Run: launch-scipy-ml.sh -g 1 -m 32"
    exit 1
fi

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Clone repo if not present
if [ ! -d "superconductor-vae" ]; then
    echo "Cloning repository..."
    git clone https://github.com/jamesconde/superconductor-vae.git
fi

cd superconductor-vae

# Check Python and PyTorch
echo ""
echo "Checking Python environment..."
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install missing dependencies if needed
echo ""
echo "Checking dependencies..."
pip install --quiet --user pandas numpy

# Check if data exists
echo ""
echo "Checking data files..."
if [ -f "data/processed/supercon_fractions_combined.csv" ]; then
    echo "  [OK] Training data found (combined SuperCon + NEMAD)"
elif [ -f "data/processed/supercon_fractions.csv" ]; then
    echo "  [OK] Training data found (original SuperCon only)"
    echo "  NOTE: Run 'python scripts/ingest_nemad.py' to create combined dataset"
else
    echo "  [MISSING] data/processed/supercon_fractions_combined.csv"
    echo "  You need to upload the data file!"
fi

if [ -f "data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json" ]; then
    echo "  [OK] Holdout data found"
else
    echo "  [MISSING] data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json"
fi

# Create outputs directory
mkdir -p outputs

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To start training:"
echo "  cd superconductor-vae"
echo "  python3 scripts/train_v12_clean.py"
echo ""
echo "To start training in background (survives disconnect):"
echo "  nohup python3 scripts/train_v12_clean.py > training.log 2>&1 &"
echo "  tail -f training.log  # Monitor progress"
echo "=============================================="
