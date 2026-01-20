# CUDA Bug Investigation Report - Jan 15, 2026

## Summary

Investigation into `RuntimeError: handle_0 INTERNAL ASSERT FAILED at "../c10/cuda/driver_api.cpp":15` crash during VAE training with scheduled sampling and exact match computation.

## Environment

```
PyTorch: 2.1.0+cu121
CUDA: 12.1 (runtime) / 12.4 (driver)
cuDNN: 8902
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Platform: WSL2 on Windows (Linux 6.6.87.2-microsoft-standard-WSL2)
```

## Crash Triggers Identified

### Trigger 1: Teacher Forcing < 1.0 (Scheduled Sampling)
- When `teacher_forcing_ratio < 1.0`, the decoder runs in sequential autoregressive mode
- Calls `transformer_decoder` in a loop `seq_len` times (vs 1 call with TF=1.0)
- Crash occurs in the feedforward block's activation function (GELU/ReLU)
- Error at: `torch/nn/modules/transformer.py:874` in `_ff_block`

### Trigger 2: Exact Match Computation
- Boolean OR with negation: `(correct | ~mask)`
- Followed by any reduction: `.all()`, `.min()`, `.sum() == seq_len`
- Even CPU computation doesn't prevent the crash
- Crash occurs on the NEXT CUDA forward pass

## Root Cause Analysis

### 1. Known PyTorch Issue #112957 (Partially Fixed)
- **Original bug**: PyTorch tried to load `libcuda.so` instead of `libcuda.so.1` on WSL2
- **Fix**: PyTorch 2.1.1 fixed this with PR #112996
- **User's version**: 2.1.0+cu121 (BEFORE the fix)
- **Recommendation**: Upgrade to PyTorch 2.1.1+ or newer

### 2. WSL2-Specific CUDA Issues
Multiple reports of similar crashes on WSL2:
- Issue #166234: CUDA Caching Allocator assertion failure on RTX 4090 under WSL2
- Issue #138 (LitGPT): Same error on WSL2 with RTX 3090/3070
- Maintainers labeled these as "3rd party" / WSL2-specific

### 3. AMP Autocast Cache Issue (#112583)
- Autocast caches results based on grad mode
- Nested `no_grad` inside `autocast` can corrupt the cache
- Sequential forward passes in scheduled sampling may trigger this
- Marked "high priority" but no confirmed fix as of late 2024

### 4. Softmax/Activation Crashes (#192286)
- Users report crashes specifically with LogSoftmax/Softmax in transformers
- PyTorch 2.1.0 + CUDA 12.1 on RTX GPUs
- Lowering model dimensionality or batch size sometimes helps
- Driver update to 450.51.06+ resolved some cases

## Recommendations

### Immediate Fixes (Priority Order)

1. **Upgrade PyTorch** (Highest Priority)
   ```bash
   pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   # OR
   pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   - Fixes the driver API .so loading issue
   - Includes FlashAttention-v2 for better performance
   - Better AMP handling

2. **Try Disabling Autocast Cache**
   ```python
   # Before training loop
   torch.cuda.amp.autocast_cache_enabled = False  # If available
   # OR restructure to avoid cache issues
   ```

3. **Disable AMP as Test**
   ```python
   # Temporarily disable to confirm AMP is the issue
   use_amp = False
   ```

4. **Update NVIDIA Driver**
   - Current: 12.4 driver with 12.1 CUDA runtime
   - Try updating Windows NVIDIA driver
   - Run `nvidia-smi` in WSL to verify

### Long-term Solutions

5. **Consider Native Linux**
   - WSL2 has known CUDA issues across multiple PyTorch versions
   - Native Linux installation may resolve the issue entirely

6. **File PyTorch Bug Report**
   Include:
   - Exact environment (provided above)
   - Minimal reproduction script
   - Stack trace
   - Note that TF < 1.0 and boolean reductions trigger it

## Current Workarounds (Working)

1. **TF = 1.0 constant**: Full teacher forcing, no scheduled sampling
2. **Exact match = 0.0**: Skip computation entirely during training
3. **num_workers = 0**: Single-process DataLoader

Training is running successfully with these workarounds:
- Epoch 100+ achieved
- 96.3% token accuracy
- All losses (Formula, Tc, Magpie, Stoich) decreasing

## Test Script for Reproducing

```python
import torch
import torch.nn as nn

# Minimal reproduction
device = torch.device('cuda')
batch_size = 32
seq_len = 50
vocab_size = 100
d_model = 512

decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6).to(device)

with torch.cuda.amp.autocast():
    for step in range(seq_len):
        tgt = torch.randn(step+1, batch_size, d_model, device=device)
        memory = torch.randn(16, batch_size, d_model, device=device)
        mask = torch.triu(torch.ones(step+1, step+1, device=device), diagonal=1).bool()

        # This loop crashes around epoch 1-2 with TF < 1.0 behavior
        output = decoder(tgt, memory, tgt_mask=mask)
```

## References

- [PyTorch Issue #112957](https://github.com/pytorch/pytorch/issues/112957) - Driver API fix
- [PyTorch Issue #112583](https://github.com/pytorch/pytorch/issues/112583) - Autocast cache
- [PyTorch Issue #166234](https://github.com/pytorch/pytorch/issues/166234) - WSL2 caching allocator
- [PyTorch Forums #192286](https://discuss.pytorch.org/t/pytorch-crashes-and-tells-me-to-report-a-bug-when-running-a-transformer-model/192286) - Softmax crash
- [LitGPT Issue #138](https://github.com/Lightning-AI/litgpt/issues/138) - Same error on WSL2
- [PyTorch 2.1.1 Release](https://github.com/pytorch/pytorch/releases/tag/v2.1.1) - Contains driver API fix
