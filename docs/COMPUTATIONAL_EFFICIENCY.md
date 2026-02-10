# Computational Efficiency and Training Benchmarks

**Project**: Multi-Task Superconductor Generator
**Last Updated**: January 2026

## Overview

This document details the computational efficiency of training the Multi-Task Superconductor Generator, including hardware requirements, performance benchmarks, and comparisons to published work.

## Hardware Configuration

### Current Training Setup

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| Model Size | 108.2M parameters |
| Encoder | 5.4M parameters |
| Decoder | 102.8M parameters |
| Batch Size | 16 (limited by VRAM) |
| Sequence Length | 60 tokens max |
| Dataset | 16,521 superconductor formulas |

### Memory Usage

- VRAM Used: ~4.2GB / 8GB
- Headroom allows for batch size 16 with REINFORCE sampling
- Larger batch sizes would require gradient accumulation

## Training Performance

### Current Throughput

| Metric | Value |
|--------|-------|
| Time per Epoch | ~14 minutes |
| Training Samples/Second | 19.2 |
| Forward Passes/Second | 57.6 (includes REINFORCE) |
| Tokens/Second | ~3,455 |

### Breakdown of Computation

Each epoch involves:
1. **Training forward pass**: 16,521 samples
2. **REINFORCE sampling**: 2 additional generated sequences per batch for policy gradient
3. **Backward pass**: Gradient computation and updates
4. **TRUE Autoregressive eval**: Every 10 epochs, 2,000 samples (pure autoregressive)

Effective forward passes per epoch: ~50,000 (16,521 × 3)

## Comparison to Published Benchmarks

### Similar Model Sizes (~100M parameters)

| System | GPU | Throughput | Task |
|--------|-----|------------|------|
| GPT-2 Training | A100 (80GB) | 50,000+ tok/s | Standard LM training |
| BERT Fine-tuning | V100 (32GB) | ~20,000 tok/s | Bidirectional encoding |
| Transformer XL | TPU v3 | ~30,000 tok/s | AR with memory |
| **Our VAE** | **RTX 4060 (8GB)** | **~3,500 tok/s** | **AR generation + REINFORCE** |
| LLaMA Inference | RTX 4060 | ~30 tok/s | 7B model inference |

### Expected Ranges by GPU

| GPU | VRAM | Expected Tokens/sec (100M model) |
|-----|------|----------------------------------|
| A100 | 80GB | 30,000 - 100,000 |
| RTX 4090 | 24GB | 15,000 - 50,000 |
| RTX 4060 | 8GB | 3,000 - 10,000 |
| RTX 3060 | 12GB | 4,000 - 12,000 |

**Our throughput of 3,455 tokens/sec is within the expected range for RTX 4060.**

## Why Autoregressive VAE Training is Slower

The ~10x throughput gap compared to standard transformer training is expected due to fundamental architectural constraints:

### 1. Sequential Autoregressive Generation

Standard transformer training can process all tokens in parallel during teacher forcing. Our setup with low teacher forcing (TF=0.13-0.20) requires mostly sequential generation:

```
Standard Training:    [tok1, tok2, tok3, ... tok60] → Parallel forward pass
AR Generation:        tok1 → tok2 → tok3 → ... → tok60  (60 sequential passes)
```

Each token prediction requires a full decoder forward pass, limiting GPU parallelism.

### 2. REINFORCE Policy Gradient Overhead

REINFORCE training requires sampling from the policy to estimate gradients:

```python
# For each batch:
1. Forward pass with teacher forcing (training)
2. Generate sample 1 autoregressively (for REINFORCE)
3. Generate sample 2 autoregressively (for REINFORCE baseline)
```

This triples the forward passes compared to pure supervised training.

### 3. Consumer vs Datacenter Hardware

| Aspect | RTX 4060 Laptop | A100 |
|--------|-----------------|------|
| Memory Bandwidth | 256 GB/s | 2,039 GB/s |
| VRAM | 8 GB | 80 GB |
| Tensor Cores | 3rd Gen | 4th Gen |
| TDP | 115W | 400W |

The memory bandwidth difference (~8x) directly impacts autoregressive generation speed.

## Optimizations Implemented

### Currently Enabled

| Optimization | Status | Impact |
|--------------|--------|--------|
| Mixed Precision (bfloat16) | Enabled | ~1.5-2x speedup |
| Flash Attention (SDPA) | Enabled | ~1.3x speedup, less memory |
| KV-Cache for Generation | Enabled | ~2-3x faster AR generation |
| Multi-worker DataLoader | Environment-aware | Auto-tuned by `detect_environment()` (see below) |
| Pin Memory | Environment-aware | Auto-tuned by `detect_environment()` |

### Environment-Aware DataLoader Auto-Detection

DataLoader settings (num_workers, pin_memory, persistent_workers, prefetch_factor) are
automatically tuned by `detect_environment()` in `src/superconductor/utils/env_config.py`.
This runs once at the start of `train()` and prints a one-line banner.

| Setting | WSL2 | Colab A100 (40GB+) | Colab T4/V100 | Bare Linux (big) |
|---------|------|--------------------|---------------|-------------------|
| num_workers | 2 | min(4, cpus-1) | min(2, cpus-1) | min(8, cpus-1) |
| pin_memory | False | True | True | True |
| persistent_workers | False | True | True | True |
| prefetch_factor | 1 | 2 | 2 | 2 |

**Why**: WSL2 has 7.5GB RAM shared with Windows. Spawning workers (one per CPU thread = 22) caused OOM.
Hardcoding `num_workers=2` everywhere fixed WSL2 but handicapped Colab A100 training.

### V12.9/V12.16: Speculative Decoding

| Optimization | Status | Impact |
|--------------|--------|--------|
| V2 Position-Aware Draft Model | **DISABLED (V12.16.1)** | See explanation below |

Speculative decoding was intended to use a lightweight draft model (n-gram + structural rules) to predict k tokens ahead, then verify all k in a single forward pass.

**V12.16.1 Finding (February 2026): Architecture Mismatch**

Testing revealed that speculative decoding with an n-gram draft model doesn't work well for VAE generation:

| Metric | Expected | Actual |
|--------|----------|--------|
| Acceptance rate | 60-70% | 1-4% |
| Speedup | 2.5-3.4x | **0.2x (5x slower!)** |

**Why it doesn't work:**
1. **N-gram draft model**: Predicts based on corpus statistics (common patterns like "La-Sr-Cu-O")
2. **VAE decoder**: Predicts based on the latent vector z, which encodes a *specific* material
3. The two models have fundamentally different information sources - they don't agree

**This is different from LLM speculative decoding** where both draft and main model see the same context. In VAE generation, only the main decoder knows what z encodes.

**Potential future fix:** A z-conditioned draft model (mini-decoder) that takes the latent as input.

**Files (for reference):**
- `src/superconductor/models/ngram_draft.py` - V2 draft model implementation
- `src/superconductor/models/autoregressive_decoder.py` - Contains `speculative_sample_for_reinforce()`
- `data/processed/draft_model_v2.pkl` - V2 cached draft model (position-aware)

### Not Currently Used

| Optimization | Status | Potential Impact |
|--------------|--------|------------------|
| torch.compile | Disabled | +10-30% (requires testing) |
| Gradient Checkpointing | Disabled | Saves memory, slight slowdown |
| Larger Batch (gradient accum) | Not used | Better GPU utilization |

## GPU Utilization Analysis

Observed GPU utilization: **~23%**

This relatively low utilization is expected due to:

1. **Autoregressive bottleneck**: GPU waits for each token before generating next
2. **Small batch size**: 16 samples don't fully saturate GPU
3. **CPU-bound operations**: Data loading, Python overhead
4. **Memory transfers**: Frequent small tensor operations

### Improving Utilization

To increase GPU utilization:
- Increase batch size (requires more VRAM or gradient accumulation)
- Use torch.compile for kernel fusion
- Batch multiple sequences in generation (speculative decoding)

## Time Estimates

### Training Duration

| Epochs | Time (RTX 4060) | Time (RTX 4090 estimate) |
|--------|-----------------|--------------------------|
| 100 | ~24 hours | ~8 hours |
| 500 | ~5 days | ~1.7 days |
| 1000 | ~10 days | ~3.3 days |
| 2000 | ~20 days | ~6.7 days |

### Scaling Expectations

For faster training, consider:
- **RTX 4090**: ~3x faster (more VRAM, higher bandwidth)
- **A100**: ~10x faster (datacenter GPU)
- **Multi-GPU**: Near-linear scaling for data parallelism

## Metrics and Monitoring

### Key Metrics to Track

| Metric | Frequency | Purpose |
|--------|-----------|---------|
| TF-based Exact Match | Every epoch | Training progress (with dropout noise) |
| TRUE Autoregressive | Every 10 epochs | Honest inference performance |
| Loss | Every epoch | Convergence monitoring |
| Entropy | Every epoch | Exploration health |
| Reward | Every epoch | REINFORCE signal quality |

### TF-based vs TRUE Autoregressive Discrepancy

The TF-based metric is computed during training with dropout **active**, resulting in ~3-4% lower scores than TRUE autoregressive (which uses eval mode with dropout off).

| Mode | Dropout | Use Case |
|------|---------|----------|
| TF-based (train mode) | Active | Training signal, pessimistic |
| TRUE Autoregressive (eval mode) | Off | Honest inference metric |

**TRUE Autoregressive is the authoritative metric** for actual model performance.

## Recommendations

### For Current Hardware (RTX 4060, 8GB)

1. **Batch size 16** is optimal for available VRAM
2. **Monitor TRUE Autoregressive** for real performance
3. **Accept ~14 min/epoch** as baseline
4. Consider **gradient accumulation** for effective larger batches

### For Faster Training

1. **Upgrade to RTX 4090** (~3x speedup, 24GB VRAM)
2. **Enable torch.compile** after validating correctness
3. **Reduce REINFORCE samples** (n_samples=1) for 1.5x speedup at cost of gradient variance
4. **Use cloud GPU** (A100) for intensive experiments

### For Production/Deployment

1. **Quantization** (INT8) for faster inference
2. **ONNX export** for optimized inference runtime
3. **Speculative decoding** for faster AR generation (now implemented!)
4. **Distillation** to smaller model for deployment

## Speculative Decoding Details (V12.9)

### Overview

Speculative decoding speeds up autoregressive generation by using a fast "draft" model to predict multiple tokens ahead, then verifying them in batch with the main model. This exploits the observation that chemical formulas have predictable structure.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Draft Model                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌───────────────────┐                     │
│  │  N-gram Model │    │ Structural Rules  │                     │
│  │  (Trigram)    │    │ (State Machine)   │                     │
│  └───────┬───────┘    └─────────┬─────────┘                     │
│          │     Intersection     │                               │
│          └──────────┬───────────┘                               │
│                     ▼                                           │
│              Draft k tokens                                     │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                Main Transformer Decoder                         │
├─────────────────────────────────────────────────────────────────┤
│  Verify all k tokens in ONE forward pass                        │
│  Accept via rejection sampling: r < p_model(token)              │
│  Sample at rejection point from adjusted distribution           │
└─────────────────────────────────────────────────────────────────┘
```

### Formula Structure Rules

The structural draft model encodes the grammar of fraction-format formulas:

| Current State | Valid Next Tokens |
|---------------|-------------------|
| `<START>` | Element only |
| Element | `(`, Element, digit, `<END>` |
| `(` | Digit only (numerator start) |
| Numerator digit | Digit, `/` |
| `/` | Digit only (denominator start) |
| Denominator digit | Digit, `)` |
| `)` | Element, `<END>` |

### N-gram Statistics

The n-gram model learns from training data:
- **Trigrams**: P(token | prev_2_tokens) - e.g., P(")" | "1", "0") for common /10 denominator
- **Bigrams**: Fallback when trigram unseen
- **Unigrams**: Final fallback

Common patterns learned:
- Top denominators: /10 (3.06%), /5 (2.05%), /100 (1.8%)
- Top elements: O, Cu, Ba, La, Sr (cuprate superconductors)
- Strong element transitions: Cu→O, Ba→Cu, etc.

### Configuration

In `train_v12_clean.py`:

```python
TRAIN_CONFIG = {
    # ...
    'use_speculative_decoding': True,   # Enable/disable
    'speculative_k': 5,                  # Tokens to draft at once
    'draft_model_path': 'data/processed/draft_model_v2.pkl',  # V2 (position-aware)
}
```

### Building the Draft Model

```bash
python scripts/build_ngram_draft.py
```

This builds the draft model from training data and caches to disk. The model is automatically loaded during training if available.

### Performance Metrics

During training, speculative decoding metrics are logged:
- **spec_acceptance_rate**: Fraction of drafted tokens accepted (target: 60-70%)
- **spec_tokens_per_step**: Average tokens generated per speculative step (target: >2.5)

Example output:
```
Epoch  100 | TF: 0.15 | Loss: 0.234 | Exact: 82.1% | ... | Spec: 65% (2.8t/s)
```

### When to Use

Speculative decoding is most beneficial when:
1. **REINFORCE is enabled** (rl_weight > 0) - sampling is the bottleneck
2. **Autoregressive mode** (use_autoregressive_reinforce=True)
3. **Formula structure is predictable** - which it is for superconductors

It provides minimal benefit for:
- Pure teacher forcing (TF=1.0)
- Non-autoregressive sampling from logits

## References

- [Efficient Transformers Survey](https://arxiv.org/abs/2009.06732)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [PyTorch 2.0 torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
