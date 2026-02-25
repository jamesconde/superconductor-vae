# Phase 2: Self-Supervised Training Design Document

**Created**: 2026-02-24
**Status**: Active Implementation
**Base Model**: V12.41 (epoch 3292, d_model=512, 148-token vocab)

## Context

The V12.41 model achieves 86.5% TRUE AR exact match on training data but only 22.2% exact match
on 45 holdout superconductors. The gap between training and holdout performance is the core problem.
The V13-V15 semantic fraction experiment failed catastrophically (<1% AR) and is being abandoned.

**Purpose**: Phase 2 is a MODEL ENHANCEMENT algorithm — it uses the model's own generations as
self-supervised signal to improve generalization. It is NOT a discovery algorithm. Novel
superconductor discovery is a separate effort (holdout search, z-space exploration). However,
if a generated formula passes all validators and is not in the training data, it is flagged and
saved opportunistically (formula + z-vector + predicted Tc → `outputs/phase2_discoveries.jsonl`).

**Decisions**: All 4 loss signals from start (weight ramp handles gradual introduction). Base
model: V12.41 d_model=512 (proven 86.5% AR). Training target: Colab A100 (develop locally,
deploy to Colab).

---

## A. Core Architecture: Interleaved Self-Supervised Sub-Epoch

Phase 2 runs as an additional step **within** the existing training loop, not as a separate phase.
After each supervised epoch completes, a self-supervised sub-epoch runs every K epochs:

```
Phase 2 Sub-Epoch (every 2 supervised epochs):
  1. SAMPLE: Draw N z-vectors from latent space (4 strategies)
  2. GENERATE: Decode z -> token sequences (greedy + temperature)
  3. FILTER: Validate candidates (chemical + physics validators)
  4. SCORE: Compute 4 self-supervised loss signals on ALL valid candidates
  5. UPDATE: Accumulate gradients with ramped weight gate
  6. MONITOR: Track z-space quality metrics + degeneracy diagnostics
```

**No deduplication**: If multiple z-vectors decode to the same formula, losses are computed on
ALL of them — each z is a distinct latent point and needs its own round-trip loss. Degeneracy
(many z → same formula) is tracked as a **diagnostic metric** indicating z-space topology (loops
in latent space). Similarly, small perturbations producing slightly wrong formulas are NOT
consolidated — they are penalized naturally by physics and consistency losses.

N auto-scales continuously with GPU VRAM: `n = clamp(3.2 * vram_gb, 32, 512)` — no hardcoded
tiers.

---

## B. Z-Space Sampling (4 Strategies)

When element data is available (z-cache contains `element_indices` and `element_mask`):

### Source 1: Training Z Neighborhood (40% = ~103 samples)
- Perturb cached training z-vectors: `z_perturbed = z_train + sigma * epsilon`
- Noise schedule: sigma starts at 0.02, increases to 0.1 over 200 epochs
- Reuse: z-cache infrastructure from `train_v12_clean.py:cache_z_vectors()`

### Source 2: Element-Anchored Sampling (20% = ~51 samples)
- Explores z-neighborhoods of chemically similar materials
- Cycles through ALL elements using inverse-visit-count weighting (rare elements get priority)
- For each anchor: find samples sharing >= 2 elements via inverted index
- 30% of budget: SLERP between anchor and random element-neighbor
- 70% of budget: blend toward neighborhood centroid + Gaussian noise (sigma=0.05)
- Fallback chain: min_shared=2 → 1 → plain perturbation
- Element visit counts decay at 0.995/sub-epoch to allow re-exploration
- Rationale: holdout search proves element-anchored exploration finds training samples sharing
  the same elements as a target and explores their z-neighborhood, keeping perturbations in
  chemically meaningful regions. See `docs/ELEMENT_AWARE_SLERP_DESIGN.md` for future enhancements.

### Source 3: Interpolation (25% = ~64 samples)
- SLERP between same-family training z-vector pairs
- t = [0.1, 0.2, ..., 0.9], capped at 100 pairs
- Reuse: `slerp()` from `holdout_search_targeted.py:163`

### Source 4: PCA-Directed Walk (15% = ~38 samples)
- Walk along top-20 PCs of training z-distribution at +/- [0.3, 0.5, 1.0, 1.5, 2.0] sigma
- Reuse: PCA infrastructure from `scratch/z_space_pca_analysis.py`

**Backward compatibility**: If z-cache lacks `element_indices` (old cache), falls back to
original 3-strategy split (60/25/15). First z-cache generated after the code update will
include element data automatically.

---

## C. Generation & Filtering

### Generation
- `decoder.generate_with_kv_cache()` with type masking enabled
- 50% greedy (temperature=0.0), 50% exploratory (temperature=0.1-0.3)
- All generation under `torch.no_grad()` (decoder gradients via REINFORCE only)

### Filtering Pipeline (4 stages, all existing code)
1. **Parse check**: `_formula_to_encoder_input()` from `round_trip_loss.py`
2. **CandidateValidator**: `overall_score >= 0.5` (`validation/candidate_validator.py`)
3. **PhysicsValidator**: `plausibility_score >= 0.4` (`validation/physics_validator.py`)
4. **Constraint rewards A1, A4, A7**: Must pass all three (`losses/constraint_rewards.py`)

Only candidates passing all 4 stages get loss computed.

---

## D. Four Self-Supervised Loss Signals

### Loss 1: Extended Round-Trip Consistency
Extends current A5 loss (`losses/round_trip_loss.py`) to arbitrary z-vectors:
```
z_sampled -> decode(greedy) -> formula -> parse -> re-encode -> z_reconstructed
Loss = z_weight * MSE(z, z') + tc_weight * MSE(Tc(z), Tc(z'))
```
Key difference from A5: operates on perturbed/interpolated z (novel z-space points), not just
training z. Gradients flow through encoder + magpie head. Decoder is under no_grad.

### Loss 2: Multi-Head Self-Consistency
For generated formulas, ensure encoder heads agree:
- SC classifier agrees with Tc (if Tc > 0K, sc_pred should be > 0.5)
- Tc value falls within predicted Tc bucket
- Family prediction matches elements in generated formula
```
L = w1*BCE(sc_pred, should_be_sc) + w2*MSE(tc, tc_from_bucket) + w3*CE(family, inferred_family)
```
Reuse: consistency check logic from `scratch/z_self_consistency_check.py`

### Loss 3: Physics Constraints on Generated Formulas
Apply existing differentiable constraints (`losses/constraint_zoo.py`) to parsed generated
formulas:
- A3: Site occupancy sum (differentiable through element fractions)
- A6: Charge balance (differentiable)

### Loss 4: REINFORCE Round-Trip Reward
Extend existing REINFORCE to reward formulas that round-trip consistently:
```
reward = cosine_sim(z_sampled, z_reconstructed) * physics_validity_score
```
This is the only signal that reaches the decoder.

---

## E. Safety Guards (5 layers)

| Guard | Mechanism | Threshold |
|-------|-----------|-----------|
| Weight ramp | Linear warmup from 0 to max over 50 epochs | `phase2_max_weight = 0.1` |
| Exact match monitor | Halve Phase 2 weight if training exact drops >2% | Same pattern as existing RL guard |
| Separate LR | Phase 2 uses 0.1x main LR | Prevents destabilizing converged heads |
| Gradient clipping | Phase 2 gradients clipped at 0.5 (vs 1.0 for Phase 1) | Tighter clip for novel z |
| Frequency control | Phase 2 runs every 2 epochs (model already at 86.5% — supervised epochs have diminishing returns) | Configurable via `phase2_interval` |

---

## F. Data Augmentation: Loss-Only, NOT New Training Samples

Generated formulas serve as **loss signals only**. They are NOT added to the training set because:
- "Close but not exact" formulas would introduce label noise
- Self-generated data creates amplification feedback loops
- The model already memorizes at 98.7% TF; more data = more memorization
- The 4 loss signals provide sufficient gradient without needing correct targets

---

## G. Z-Space Quality Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Round-Trip Z-MSE | z-space consistency at novel points | Decrease |
| Valid Formula Rate | valid candidates / total generated | Increase |
| Unique Formula Rate | unique formulas / valid (mode collapse detector) | > 0.3 |
| Degeneracy Count | extra copies (same formula from different z), diagnostic only | Diagnostic |
| Cross-Head Agreement | SC/Tc/Family consistency | Increase |
| Holdout Similarity | best match to 45 holdout materials | Increase |
| Generalization Ratio | holdout_exact / training_exact | Increase |

**Mode collapse detection**: If unique/total < 0.3, increase temperature to 0.5 for 2 epochs +
double round-trip weight + add diversity bonus to REINFORCE reward.

---

## H. Holdout Discovery Integration

- **Every 50 epochs**: Mini holdout search (200 candidates per target, ~5 min)
- **New best checkpoint**: Full holdout search (27,550 candidates per target, ~30 min)
- Log: `holdout_exact, holdout_sim_99, holdout_sim_95, holdout_mean_sim` to training CSV

### Success Milestones

| Milestone | Metric |
|-----------|--------|
| Phase 2 alpha | Round-trip Z-MSE < 0.1 |
| Phase 2 beta | Holdout exact > 30% (from 22.2%) |
| Phase 2 gold | Holdout exact > 50% |
| Phase 2 complete | All 45 holdout at >= 0.99 similarity |
| Discovery | Novel formula passing all validators with Tc > 50K |

---

## I. Config Parameters

```python
# Phase 2: Self-Supervised Training (new section in TRAIN_CONFIG)
'phase2_enabled': False,
'phase2_start': 'auto',          # Epoch or 'auto' (activate when exact>=80%, Tc R2>=0.99)
'phase2_interval': 2,            # Run every N supervised epochs (model already at 86.5%)
'phase2_max_weight': 0.1,        # Max total Phase 2 loss weight
'phase2_warmup': 50,             # Epochs to ramp from 0 to max weight
'phase2_n_samples': 'auto',      # 'auto' = continuous VRAM scaling: clamp(3.2*vram_gb, 32, 512)
                                  # Number of z-vectors to sample per sub-epoch. All valid
                                  # candidates get losses (no deduplication). Set int for override.
'phase2_noise_schedule': [0.02, 0.05, 0.08, 0.1],
'phase2_lr_factor': 0.1,         # Phase 2 LR = main LR * this
'phase2_max_grad_norm': 0.5,
'phase2_diversity_bonus': 5.0,   # REINFORCE bonus for unique formulas
'phase2_collapse_threshold': 0.3,
'phase2_coverage_k': 64,        # K-means clusters for z-space coverage tracking
'phase2_coverage_temperature': 1.0,  # Sampling weight exponent
'phase2_coverage_decay': 0.995,      # Visit count decay per sub-epoch
# Strategy 4: Element-Anchored Sampling
'phase2_element_anchored': True,              # Enable element-anchored sampling
'phase2_element_anchored_fraction': 0.20,     # Fraction of sampling budget (reduces perturbation 60%→40%)
'phase2_element_min_shared': 2,               # Min shared elements for neighbor status
'phase2_element_perturb_sigma': 0.05,         # Perturbation sigma for centroid blends
'phase2_element_interpolate_fraction': 0.3,   # Fraction using SLERP (rest = centroid blend)
'holdout_eval_interval': 50,
'holdout_eval_budget': 200,
```

**VRAM Scaling Examples** (continuous formula `clamp(3.2 * GB, 32, 512)`, no hardcoded tiers):
| GPU VRAM | n_samples |
|----------|----------:|
| 8 GB (RTX 4060) | 25 |
| 16 GB | 51 |
| 24 GB (RTX 3090) | 77 |
| 40 GB (A100) | 128 |
| 80 GB (A100 HiRAM) | 256 |

---

## I-bis. Z-Space Coverage Tracking

### Problem
The `ZSpaceSampler` uses uniform random sampling (`torch.randint`) to select base z-vectors.
This means popular cluster regions get sampled repeatedly while sparse regions between clusters
are rarely explored. There's no memory of previously explored regions across sub-epochs.

### Solution: K-Means Coverage Tracking (k=64)

Partition the 2048-dim z-space into 64 clusters using MiniBatchKMeans (already a project dependency),
then track per-cluster visit counts and compute inverse-visit-count sampling weights.

**Implementation**: `src/superconductor/training/coverage_tracker.py` — `CoverageTracker` class.

**Why k=64**: 46K points / 64 = ~720 per cluster. Fine enough for meaningful coverage, coarse enough
that all clusters get visited in reasonable time.

**Sampling weight formula**: `w[cluster] = 1 / (1 + visit_count)^temperature`, mapped to per-training-point
weights via cluster assignment, normalized for `torch.multinomial`.

**Decay**: `visit_counts *= 0.995` each sub-epoch (half-life ~139 sub-epochs). Prevents early visits
from permanently "satisfying" a cluster.

### Per-Cluster Quality Tracking

Each cluster also tracks its valid formula rate (samples passing all filters / total generated):
- **garbage clusters** (<10% valid rate): Regions where the model produces garbage — true bounds on
  the model's latent geometry. These represent the edges of the model's representational capacity.
- **boundary clusters** (10-50% valid rate): The model is close but not reliably correct — these are
  prime targets for additional self-supervised training focus.
- **good clusters** (>=50% valid rate): The model reliably generates valid formulas from this region.

This data is available via `CoverageTracker.get_cluster_quality_report()` and logged as
`coverage_garbage_clusters` / `coverage_boundary_clusters` metrics.

### Coverage Metrics (logged to phase2_log.csv)

| Metric | Description | Target Trend |
|--------|-------------|-------------|
| `coverage_fraction` | Clusters visited / total | Increase toward 1.0 |
| `coverage_clusters_visited` | Count of visited clusters | Increase toward 64 |
| `coverage_visit_gini` | Gini coefficient (0=uniform, 1=concentrated) | Decrease |
| `coverage_visit_entropy_normalized` | Entropy / log(k), 1.0 = uniform | Increase |
| `coverage_total_visits` | Cumulative z-vectors sampled | Increase |
| `coverage_min_visits` / `coverage_max_visits` | Cluster visit range | Min should increase |
| `coverage_garbage_clusters` | Clusters with <10% valid rate (latent bounds) | Diagnostic |
| `coverage_boundary_clusters` | Clusters with 10-50% valid rate (training targets) | Diagnostic |

### Performance
- K-means fit: ~2-3s (once per z-cache load, not per epoch)
- Per-sub-epoch overhead: ~4ms total (assign + record + weights + metrics)
- Checkpoint size increase: ~710 KB (centroids + labels + counts)

---

## I-ter. Phase 2 State Persistence

Phase 2 state (activation epoch, collapse flags, exact history, coverage tracker, discovery tracker
counters) is serialized to checkpoints via `SelfSupervisedEpoch.get_state()/load_state()`.

This ensures Phase 2 resumes correctly after training interruption — no re-exploration of already-visited
regions, no re-detection of already-seen novel formulas, and correct weight ramp continuation.

State saved:
- `activation_epoch`, `activation_exact`, `current_weight`
- `collapse_active`, `collapse_epochs_remaining`
- `exact_history` (for safety guard drop detection)
- `all_unique_formulas` (for REINFORCE diversity bonus)
- `coverage_tracker` state (centroids, cluster labels, visit counts, quality counters, novelty tracking)
- Discovery tracker counters (novel/holdout counts, seen formula sets)
- Sampler state: `element_visit_counts`, `element_seen_formulas` (BUG FIX: previously not persisted)

---

## I-ter-bis. Novelty & Saturation Tracking

### Problem
Coverage tracking tells us WHERE we sample and whether regions produce VALID formulas, but not
whether additional visits produce NEW unique formulas. A cluster visited 1000 times producing the
same 5 formulas is saturated but was previously indistinguishable from a productive cluster.

Similarly, element-anchored sampling (Strategy 2) cycles through elements by inverse visit count,
but doesn't know whether visiting Cu-containing space 100 more times will discover new Cu formulas
or just regenerate the same ones.

### Solution: Per-Region Novelty Rate

#### Cluster-Level Novelty (CoverageTracker)

Each cluster tracks:
- `_cluster_seen_formulas`: cumulative set of all unique formulas ever produced from that cluster
- `_cluster_novel_this_epoch`: count of NEW unique formulas this sub-epoch
- `_cluster_total_this_epoch`: count of ALL valid formulas this sub-epoch

**Novelty rate** = novel / total per cluster per sub-epoch.

Novelty factors into sampling weights:
```
w[cluster] = novelty_factor / (1 + visits)^temperature
novelty_factor = max(0.1, novel / max(1, total))
```

- Saturated clusters (novelty < 5%): get 10% floor weight (not zero — model may improve)
- Unexplored clusters (no data): get 1.0 (max priority)
- Productive clusters (novelty >= 20%): get proportional weight

#### Element-Level Novelty (ZSpaceSampler)

Each element tracks:
- `_element_seen_formulas`: cumulative set of unique formulas containing that element
- Strategy-agnostic: any valid formula with Cu counts as Cu exploration

Element cycling weights incorporate discovery rate:
```
discovery_rate = unique_formulas / max(1, visits)
novelty_factor = clamp(discovery_rate, 0.1, 1.0)
elem_weight = (1 / (1 + visits)) * novelty_factor
```

### New Metrics

| Metric | Description |
|--------|-------------|
| `coverage_avg_novelty` | Mean novelty rate across clusters with valid samples this sub-epoch |
| `coverage_saturated_clusters` | Clusters with novelty rate < 5% (producing same formulas) |
| `coverage_productive_clusters` | Clusters with novelty rate >= 20% (still discovering) |
| `coverage_total_unique_formulas` | Sum of unique formulas across all clusters |
| `phase2_elem_total_unique` | Sum of unique formulas across all elements |

### State Persistence

All novelty data is serialized to checkpoints:
- `cluster_seen_formulas`: list[list[str]] (sets converted for weights_only=True compatibility)
- `cluster_novel_this_epoch`, `cluster_total_this_epoch`: tensors
- `sampler_element_visit_counts`: dict[int, float]
- `sampler_element_seen_formulas`: dict[int, list[str]]

Old checkpoints without novelty fields load with empty defaults (backward compatible).

### Checkpoint Size Impact
- ~64 clusters x ~50 formulas x ~20 chars = ~64KB
- ~80 elements x ~200 formulas x ~20 chars = ~320KB
- Total: ~400KB extra (negligible vs 1.27GB checkpoint)

---

## J. Implementation Plan

### New Files

| File | Purpose |
|------|---------|
| `src/superconductor/training/self_supervised.py` | Phase 2 orchestrator: `SelfSupervisedConfig`, `SelfSupervisedEpoch`, `ZSpaceSampler`, `CandidateFilter`, `Phase2LossComputer` |
| `scripts/analysis/phase2_dashboard.py` | Monitoring: z-space quality plots, holdout trend, mode collapse detection |

### Modified Files

| File | Change |
|------|--------|
| `scripts/train_v12_clean.py` | Add Phase 2 config, call site after `train_epoch()`, Phase 2 metrics in training log |
| `src/superconductor/losses/round_trip_loss.py` | New `ExtendedRoundTripConsistencyLoss` subclass for arbitrary z-vectors |
| `scripts/holdout/holdout_search_targeted.py` | Add `--mini` mode (200 candidates, JSON-only output) |

### Existing Code Reused Directly (no changes)

| Component | File |
|-----------|------|
| Z-vector caching | `train_v12_clean.py:cache_z_vectors()` |
| KV-cached generation | `autoregressive_decoder.py:generate_with_kv_cache()` |
| Formula parsing | `round_trip_loss.py:_formula_to_encoder_input()` |
| Encoder encode/decode | `attention_vae.py:encode()`, `decode()` |
| CandidateValidator | `validation/candidate_validator.py` |
| PhysicsValidator | `validation/physics_validator.py` |
| Constraint rewards A1/A4/A7 | `losses/constraint_rewards.py` |
| Differentiable physics A3/A6 | `losses/constraint_zoo.py` |
| SLERP interpolation | `holdout_search_targeted.py:slerp()` |

---

## K. Deployment Plan (Colab A100)

### Local Development (RTX 4060)
1. Write all Phase 2 code locally
2. Test with `phase2_n_samples=16` (override auto-scaling) for 5 epochs to verify no crashes/NaN
3. Verify safety guards trigger correctly

### Colab Deployment
1. Upload V12.41 checkpoint (`checkpoint_best V1241.pt`, 1.27 GB)
2. Upload updated codebase (src/, scripts/, data/)
3. Set `phase2_enabled=True` (n_samples auto-scales to 256 on A100)
4. Resume from V12.41 epoch 3292 with Phase 2 active
5. Monitor via gist push (existing infrastructure from V15.3 Colab training)

### Key Config for Colab
```python
'd_model': 512,                   # V12.41 original (NOT 1024)
'dim_feedforward': 2048,          # V12.41 original
'memory_bottleneck_dim': 0,       # No bottleneck (V12.41 2-layer MLP)
'use_semantic_fractions': False,   # Digit-by-digit fractions
'vocab_size': 148,                # V12.41 vocab
'max_formula_len': 60,            # V12.41 max length
'stoich_input_dim': 37,           # 12 fractions + 24 numden + 1 count
```

---

## L. Verification Plan

1. **Unit test Phase 2 sub-epoch**: Run 1 sub-epoch on V12.41 checkpoint, verify all 4 losses
   compute without NaN, gradients flow through encoder but not decoder generation
2. **Safety guard test**: Artificially set Phase 2 weight to 10.0, verify exact match monitor
   triggers weight reduction within 2 epochs
3. **Mode collapse test**: Feed 256 identical z-vectors, verify collapse detection triggers and
   diversity bonus activates
4. **Round-trip regression test**: Verify A5 loss on training batch z is unchanged (Phase 2
   doesn't break Phase 1)
5. **Holdout mini-search test**: Run `--mini` mode, verify output format matches full search JSON
6. **End-to-end (local)**: Train 20 epochs with Phase 2 enabled at n_samples=16, verify training
   exact match stays within 2% of baseline AND Phase 2 metrics trend correctly
7. **End-to-end (Colab)**: Full Phase 2 training run at n_samples=256 on A100, monitor for 100+
   epochs
