# V15.3: Curriculum-Based AR Warmup

Design document for the curriculum-based autoregressive warmup system that progressively focuses training on formulas the model can generate, starting with short/simple formulas and advancing to longer ones.

**Created**: 2026-02-27
**Status**: Implemented (disabled by default)
**Base Model**: V12.41 (vocab=148, digit-by-digit fractions)

---

## Problem

The superconductor VAE has ~97% teacher-forced (TF) exact match but only ~15% TRUE autoregressive (AR) exact match. This is the exposure bias gap — the model is trained with ground-truth token inputs (teacher forcing) but must generate from its own predictions at inference time.

Adaptive TF (where `tf_ratio = 1.0 - exact_match`) progressively exposes the model to its own predictions as it improves. However, it treats all formulas equally. In reality, there is a massive difficulty gradient by formula length:

| Length bucket | % of data | Typical AR exact | Difficulty |
|---------------|-----------|------------------|------------|
| 3-6 tokens    | ~10%      | Higher           | Easy — binary compounds like `MgB2` |
| 7-10 tokens   | ~20%      | Moderate         | Ternary with simple stoichiometry |
| 11-15 tokens  | ~20%      | Lower            | Multi-element with fractions |
| 16-23 tokens  | ~30%      | Much lower       | Complex quaternary compounds |
| 24-31 tokens  | ~10%      | Very low         | Long fractional compositions |
| 32+ tokens    | ~10%      | Near zero        | Extended site-specific formulas |

Without curriculum, the model tries to improve AR generation on 30-token cuprate formulas before mastering 5-token binaries. This wastes gradient signal on samples where the model has no hope yet.

### Why Formula Length Matters More Than Usual

In V12.41 mode, fractions are tokenized digit-by-digit: `(1/3)` costs 5 tokens (`(`, `1`, `/`, `3`, `)`). A fraction like `(499/500)` costs 9 tokens. This means:

- A "simple" ternary like `Ba(2/3)La(1/3)CuO3` is actually 20+ tokens
- Each digit is an independent AR prediction — one wrong digit cascades
- Short formulas (integers only) avoid this cascade entirely

The curriculum lets the model master integer-only formulas first, building AR confidence before tackling fraction cascades.

---

## Solution: Length-Bucketed Curriculum with Sampling Weight Boosting

### Core Idea

Divide formulas into length buckets. Boost sampling weight for the "active" bucket so the model sees it ~3x more often. When AR exact match in the active bucket exceeds a threshold for several consecutive evaluations, graduate that bucket and advance to the next.

### Design Principles

1. **Multiplicative weights**: Curriculum weights MULTIPLY existing sampler weights (class balance, Tc-binned oversampling, hard-sequence oversampling). This preserves all prior sampling corrections — a rare high-Tc superconductor in the active bucket gets boosted by both its Tc weight AND the curriculum weight.

2. **Floor weight, not masking**: Future buckets get `0.2x` weight, not zero. The model still sees all formulas at 20% of their normal rate. This prevents catastrophic forgetting and allows the model to occasionally practice hard formulas before they become active.

3. **Graduated buckets keep contributing**: Once a bucket is mastered, its weight drops to `0.5x` (not zero). Mastered formulas still contribute to training, just less than the active focus.

4. **Patience-based advancement**: The active bucket must exceed the AR exact threshold for multiple consecutive evaluations (not just one lucky eval). This prevents noisy metrics from triggering premature advancement.

---

## Architecture

### Bucket Configuration

Default bucket edges `[3, 7, 11, 16, 24, 32, 61]` create 6 buckets aligned with the training data distribution percentiles:

```
Bucket 0 [3-6]:   ~10% of data (P10) — simplest binary compounds
Bucket 1 [7-10]:  ~30% cumulative (P25) — ternary, simple stoich
Bucket 2 [11-15]: ~50% cumulative (P50, median)
Bucket 3 [16-23]: ~80% cumulative (P75) — quaternary with fractions
Bucket 4 [24-31]: ~90% cumulative (P90) — long fractional
Bucket 5 [32+]:   longest 10% — extended site-specific formulas
```

Formula length distribution (52,813 samples, max_len=60):
- Min: 3, Max: 60, Mean: 16.4, Median: 14

### Weight Scheme

At any given phase, each bucket gets a weight multiplier:

| Bucket status | Weight | Description |
|---------------|--------|-------------|
| **Active**    | 3.0x   | Current focus — boosted |
| **Frontier**  | 1.5x   | Next up — mild preview |
| **Graduated** | 0.5x   | Mastered — reduced but present |
| **Future**    | 0.2x   | Not yet active — floor |

Example at Phase 1 (active = bucket 1, [7-10]):
```
Bucket 0 [3-6]:   0.5x  (graduated)
Bucket 1 [7-10]:  3.0x  (ACTIVE)
Bucket 2 [11-15]: 1.5x  (frontier)
Bucket 3 [16-23]: 0.2x  (future)
Bucket 4 [24-31]: 0.2x  (future)
Bucket 5 [32+]:   0.2x  (future)
```

### Advancement Logic

```
Every 4 epochs:
  1. Run evaluate_true_autoregressive()
  2. Compute per-bucket AR exact match
  3. Call curriculum_scheduler.step(ar_exact_per_bucket)
  4. If active_bucket_ar >= advance_threshold (50%):
       advance_counter += 1
     Else:
       advance_counter = 0  (reset on any miss)
  5. If advance_counter >= advance_patience (3):
       ADVANCE to next phase
       Recompute sampler weights in-place
```

With AR eval every 4 epochs and patience=3, advancement requires 12 consecutive epochs of the active bucket exceeding 50% AR exact. This is conservative — prevents advancing on noise.

### Completion

When the last bucket (bucket 5, [32+]) meets the threshold, the scheduler returns `'complete'`. All buckets are then at `graduated_weight` (0.5x) or `active_boost` (3.0x for the last), and the curriculum has no further effect — training continues normally.

---

## Integration Points

### Data Flow

```
load_and_prepare_data()
  ├── Compute sample_weights (class balance, Tc-binned, hard-seq)
  ├── If curriculum_ar_enabled:
  │     ├── Compute seq_lengths from formula_tokens
  │     ├── Create CurriculumScheduler(seq_lengths, ...)
  │     ├── Store _base_sample_weights = sample_weights.copy()
  │     └── sample_weights *= curriculum_scheduler.get_sample_weights()
  └── Create WeightedRandomSampler(sample_weights)

Training loop (every 4 epochs):
  ├── evaluate_true_autoregressive()
  │     └── Compute curriculum_ar_per_bucket (always, even if curriculum disabled)
  ├── curriculum_scheduler.step(ar_per_bucket) → 'hold'/'advance'/'complete'
  └── If 'advance':
        └── train_loader.sampler.weights = _base_sample_weights * scheduler.get_sample_weights()
```

### Checkpoint Persistence

```python
# Save
checkpoint['curriculum_scheduler_state'] = curriculum_scheduler.state_dict()

# Restore
curriculum_scheduler.load_state_dict(checkpoint['curriculum_scheduler_state'])
# Then recompute sampler weights for restored phase
```

State dict contains: `current_phase`, `advance_counter`, `epoch`, `ar_history`, and config for validation.

If `bucket_edges` changed between save and restore, the scheduler resets to phase 0 with a warning.

### Per-Bucket AR Diagnostics

Regardless of whether curriculum is enabled, `evaluate_true_autoregressive()` now always computes and prints per-bucket AR breakdown:

```
AR by length: [3-6]=45.2% (n=201) | [7-10]=22.1% (n=612) | [11-15]=8.3% (n=520) | ...
```

This is stored in `z_diagnostics['curriculum_ar_per_bucket']` for analysis.

---

## Files

| File | Role |
|------|------|
| `src/superconductor/training/curriculum_scheduler.py` | `CurriculumScheduler` class (~180 lines) |
| `scripts/train_v12_clean.py` | Config, init, eval, step, checkpoint integration |
| `docs/CURRICULUM_AR_WARMUP_DESIGN.md` | This document |
| `docs/TRAINING_RECORDS.md` | V15.3 entry |

### CurriculumScheduler API

```python
class CurriculumScheduler:
    def __init__(self, seq_lengths, bucket_edges=None, advance_threshold=0.50,
                 advance_patience=3, active_boost=3.0, frontier_boost=1.5,
                 floor_weight=0.2, graduated_weight=0.5)

    def get_sample_weights(self) -> np.ndarray      # Per-sample weight multipliers
    def step(self, ar_exact_per_bucket) -> str       # 'hold'/'advance'/'complete'
    def get_status_string(self) -> str               # "phase=1/5 ([7-10]), patience=2/3"
    def state_dict(self) -> dict                     # For checkpoint saving
    def load_state_dict(self, state: dict)           # For checkpoint restore
```

---

## Config Keys

```python
TRAIN_CONFIG = {
    'curriculum_ar_enabled': False,                              # Master toggle
    'curriculum_ar_bucket_edges': [3, 7, 11, 16, 24, 32, 61],  # Bucket boundaries
    'curriculum_ar_advance_threshold': 0.50,                     # AR exact to advance
    'curriculum_ar_advance_patience': 3,                         # Consecutive evals above threshold
    'curriculum_ar_active_boost': 3.0,                           # Weight for active bucket
    'curriculum_ar_frontier_boost': 1.5,                         # Weight for next bucket
    'curriculum_ar_floor_weight': 0.2,                           # Min weight for future buckets
    'curriculum_ar_graduated_weight': 0.5,                       # Weight for mastered buckets
}
```

All defaults are conservative. `curriculum_ar_enabled: False` means this feature is completely inert unless explicitly activated.

---

## Checkpoint Compatibility

Old checkpoints (pre-V15.3) simply lack `curriculum_scheduler_state`. On resume:
- If curriculum is enabled: scheduler starts at phase 0 (fresh start)
- If curriculum is disabled: no effect whatsoever

New checkpoints with curriculum state can be loaded by old code — the extra key is ignored by `torch.load`.

---

## Interaction with Other Systems

### Adaptive Teacher Forcing
Curriculum and adaptive TF are complementary. Adaptive TF controls HOW MUCH the model sees its own predictions (globally). Curriculum controls WHICH FORMULAS the model practices AR generation on (by length). Together: the model practices AR generation (low TF) on short formulas first.

### Hard Sequence Oversampling (V12.37)
Hard sequence oversampling boosts long/complex formulas. Curriculum initially DEprioritizes them (future=0.2x). The two systems multiply: a long formula gets `hard_boost * 0.2` during early curriculum phases, then `hard_boost * 3.0` when its bucket becomes active. This is correct — we don't want hard formulas overwhelming the easy-formula phase, but we DO want them heavily sampled when it's their turn.

### Tc-Binned Oversampling (V13.2)
Tc-binned oversampling boosts rare high-Tc samples. This is orthogonal to length curriculum — a rare high-Tc binary (short) gets both its Tc boost AND the active bucket boost. No conflict.

### Phase 2 Self-Supervised Training
Phase 2 generates formulas from z-space and applies self-supervised losses. It does NOT use the sampler, so curriculum has no direct effect on Phase 2. However, better AR capability (from curriculum) means Phase 2 generates higher-quality candidates, improving its loss signals indirectly.

---

## Tuning Guide

### If advancement is too slow
- Lower `advance_threshold` (e.g., 0.30 instead of 0.50)
- Lower `advance_patience` (e.g., 2 instead of 3)
- Increase `active_boost` (e.g., 5.0 for more aggressive focus)

### If advancement is too fast (losing mastery on graduated buckets)
- Raise `graduated_weight` (e.g., 0.8 instead of 0.5)
- Raise `advance_threshold` (e.g., 0.70)
- Raise `advance_patience` (e.g., 5)

### If training is too slow (curriculum overhead)
The curriculum adds negligible computation — it only modifies sampler weights (a numpy multiply) and checks one AR metric per eval. The per-bucket AR breakdown in `evaluate_true_autoregressive` adds ~10 lines of numpy indexing. No measurable performance impact.

### Monitoring
Watch the training log for:
```
[Curriculum AR] phase=0/5 ([3-6]), patience=2/3     ← holding, building patience
[Curriculum AR] ADVANCED → phase=1/5 ([7-10]), ...   ← just advanced
[Curriculum AR] COMPLETE — all buckets graduated!     ← done
AR by length: [3-6]=62% | [7-10]=25% | [11-15]=8% | ...  ← per-bucket breakdown
```

The per-bucket AR breakdown (printed every AR eval) is the most informative metric — it shows exactly where the model's AR capability drops off by length.
