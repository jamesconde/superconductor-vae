# RLOO Bug Fix & SCST Implementation

**Date**: 2026-02-11
**Affects**: `scripts/train.py`, `scripts/train_v12_clean.py`, `src/superconductor/losses/reinforce_loss.py`

---

## 1. Bug Found: RLOO REINFORCE Gradient Was Always Zero

### The Problem

The RLOO (Leave-One-Out) REINFORCE implementation had a mathematical error that guaranteed zero gradient, regardless of model performance or reward values. This bug existed in all three files that implement RLOO.

**Root cause**: The code averaged RLOO advantages across K samples *before* multiplying with log_probs. Since the sum of RLOO advantages is mathematically zero for any K (this is a property of the LOO baseline), this averaging destroyed the gradient signal completely.

### The Math

With K=2 RLOO samples for a given input:
```
advantage_0 = r_0 - r_1
advantage_1 = r_1 - r_0 = -(r_0 - r_1)
mean(advantage_0, advantage_1) = 0   ← always, for every batch element
```

This generalizes to any K. The sum of RLOO advantages for a given prompt is always zero:
```
Σᵢ [rᵢ - (Σⱼ≠ᵢ rⱼ)/(K-1)] = 0
```

### Buggy Code (existed in all 3 files)
```python
# WRONG: averaging advantages cancels them to zero
advantages = torch.stack(advantages_list, dim=0).mean(dim=0)  # ← always 0!
mean_log_probs = log_probs_stack.mean(dim=0)
reinforce_loss = -(advantages * mean_log_probs).mean()  # ← always 0!
```

### Fixed Code
```python
# CORRECT: each sample's advantage pairs with its own log_probs
reinforce_loss = torch.zeros(1, device=rewards_stack.device)
for i in range(n_samples):
    baseline_i = (total_reward - rewards_stack[i]) / (n_samples - 1)
    advantage_i = rewards_stack[i] - baseline_i
    reinforce_loss = reinforce_loss + -(advantage_i * log_probs_stack[i]).mean()
```

Each sample's advantage multiplies with **that same sample's** log_probs. The per-sample losses are summed (not averaged) — more samples means a stronger, better-informed gradient. `rl_weight` controls overall magnitude.

### Impact

- RL was enabled from epoch ~2015 to ~2354, consuming **84% of epoch time** (119s → 19s when disabled) while producing **zero gradient**
- The 27-point gap between teacher-forced exact match (87%) and autoregressive exact match (60%) received no RL training signal the entire time
- `rl_weight` was set to 0.0 at V12.26 because the loss appeared to be zero — this was correct observation, wrong diagnosis (the bug, not the model, caused zero loss)

### Verification
```
K=2: Old (buggy) loss = 0.0,     New (fixed) loss = -3.75
K=4: Old (buggy) loss = -0.00002, New (fixed) loss = -11.10
```

---

## 2. SCST Implementation (Self-Critical Sequence Training)

### What is SCST?

SCST (Rennie et al. 2017, "Self-Critical Sequence Training for Image Captioning") uses the reward of a **greedy decode** as the baseline for REINFORCE, instead of RLOO's leave-one-out baseline.

```
advantage = reward(sampled_sequence) - reward(greedy_sequence)
loss = -(advantage * log_prob(sampled_sequence)).mean()
```

If the stochastic sample beats greedy → reinforce it (make it more likely).
If greedy beats the sample → push away from the sample.

### Why SCST over RLOO?

| Feature | RLOO (K=2) | RLOO (K=4) | SCST |
|---------|-----------|-----------|------|
| Forward passes | 2 sampled | 4 sampled | 1 greedy + 1 sampled |
| Baseline quality | 1 other sample | mean of 3 others | Greedy (deterministic, stable) |
| Variance | Higher | Lower | Low (greedy is stable) |
| Aligns with test-time | No | No | Yes (greedy = inference) |
| Advantage cancellation risk | Yes (if implementation wrong) | Yes | No (single sample) |

### Implementation Details

The `compute_scst()` method:

1. **Precomputes memory once** via `decoder.precompute_memory()` — shared by both greedy and sample passes (no duplicate encoder work)
2. **Greedy decode** with `temperature=0.0` (argmax at each step, no gradients)
3. **Stochastic sample** with normal temperature, returning log_probs for gradient
4. **Advantage** = `sample_reward - greedy_reward` (per batch element)
5. **Loss** = `-(advantage * seq_log_prob).mean()`

### Config

```python
# In TRAIN_CONFIG:
'rl_weight': 2.5,
'rl_method': 'scst',     # 'scst' or 'rloo'
'use_autoregressive_reinforce': True,
'temperature': 0.5,       # For stochastic sampling (not greedy)
```

---

## 3. Files Modified

### `scripts/train.py` (local training)
- **`compute_rloo_autoregressive()`**: Fixed RLOO gradient bug, new return signature `(reinforce_loss, mean_rewards)`
- **`compute_rloo_advantages()`**: Same fix
- **`compute_scst()`**: New method implementing Self-Critical Sequence Training
- **`REINFORCELossV12.__init__()`**: Added `rl_method` parameter
- **`REINFORCELossV12.forward()`**: Routes to SCST or RLOO based on `rl_method`
- **Config**: `rl_weight` re-enabled at 2.5, `rl_method` set to `'scst'`

### `scripts/train_v12_clean.py` (Colab training)
- **`compute_rloo_autoregressive()`**: Fixed RLOO gradient bug, new return signature `(reinforce_loss, mean_rewards, mean_entropy)`
- **`compute_rloo_from_logits()`**: Same fix
- **`compute_scst()`**: New method (same as train.py but includes entropy tracking)
- **`CombinedLossWithREINFORCE.__init__()`**: Added `rl_method` parameter
- **`CombinedLossWithREINFORCE.forward()`**: Routes to SCST or RLOO based on `rl_method`
- **Config**: `rl_weight` re-enabled at 2.5, `rl_method` set to `'scst'`, `n_samples_rloo` set to 4 (A100)

### `src/superconductor/losses/reinforce_loss.py` (standalone module)
- **`compute_rloo_baseline()`**: Fixed RLOO gradient bug
- **`REINFORCELoss.forward()`**: Updated to use new return signature

---

## 4. Literature References

- **RLOO**: Ahmadian et al. 2024, "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (arXiv 2402.14740)
  - Correct RLOO gradient: `(1/K) Σᵢ [(rᵢ - baselineᵢ) * ∇log π(yᵢ|x)]`
  - Each sample's advantage pairs with its own log_probs
  - No advantage normalization

- **SCST**: Rennie et al. 2017, "Self-Critical Sequence Training for Image Captioning" (CVPR 2017, arXiv 1612.00563)
  - Baseline = greedy decode reward
  - Advantage = sample_reward - greedy_reward
  - Single sample, aligns with test-time inference

- **MIXER**: Ranzato et al. 2016, "Sequence Level Training with Recurrent Neural Networks" (ICLR 2016)
  - Curriculum from CE to REINFORCE

---

## 5. Expected Impact

With SCST enabled, the model should now receive actual gradient signal from RL:
- **Samples that beat greedy** get reinforced (more likely)
- **Samples worse than greedy** get pushed away (less likely)
- This directly optimizes for **autoregressive exact match** — the true generative metric
- The 27-point gap between TF exact (87%) and autoregressive exact (60%) should begin closing
- Epoch time will increase somewhat (2 forward passes for SCST vs 0 when RL was disabled) but much less than the old broken RLOO (which did K passes for nothing)
