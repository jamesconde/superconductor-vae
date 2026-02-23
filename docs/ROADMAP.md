# Roadmap

Future improvements and shelved ideas, ordered by priority.

---

## Shelved: Unified Family Consistency Checker (2/23/2026)

**Status**: Shelved until bottleneck migration (V15.0) recovers exact match above 50%.

**Problem**: B1-B8 family constraints in the Constraint Zoo are 8 separate hand-tuned REINFORCE penalties (-30 to -40 each) with hardcoded chemistry rules. They're hard to extend and maintain.

**Proposal**: Single rule-based family consistency checker:

```python
def check_family_consistency(generated_tokens, predicted_family) -> bool:
    """Deterministic chemistry rules — no neural network needed."""
    elements = extract_elements(generated_tokens)
    # Contains {Y, Ba, Cu, O} → YBCO
    # Contains {La, Sr, Cu, O} → LSCO
    # Contains {Fe} + {As or P} → Iron pnictide
    # Contains {Mg, B} → MgB2
    # etc.
```

**Use cases**:
1. **Generation-time rejection filter**: Generate N candidates from same z, score each against family/element count/stoich predictions, keep best. Best-of-N sampling with physics-informed scoring.
2. **REINFORCE training**: Replace B1-B8 with single `check_family_consistency()` call and one unified penalty. Same behavior, cleaner code, easier to extend.
3. **NOT useful during teacher-forced training**: Ground-truth tokens always match their family, so the check would always pass.

**Benefits**:
- Replaces 8 config values with 1
- Single function shared between training (REINFORCE) and inference (rejection filter)
- Easier to extend when adding new families
- No additional parameters or training needed

**Prerequisite**: V15.0 bottleneck recovery. Current exact match is 0% post-migration. Wait until the model relearns cross-attention patterns and exact match stabilizes.

---

## Shelved: Teacher Forcing Scheduling — TF = 1 - Exact (2/23/2026)

**Status**: Shelved as Phase 2 after bottleneck stabilizes.

**Problem**: TF is locked at 1.0 (always ground-truth context). The decoder never practices autoregressive generation during training. The gap between TF accuracy and AR accuracy is the generalization gap.

**Proposal**: Set `TF = 1 - exact_match`. As exact match rises, TF drops, forcing the decoder to practice generation:
- exact=0% → TF=1.0 (full teacher forcing, focus on learning)
- exact=50% → TF=0.5 (half the time uses own predictions)
- exact=90% → TF=0.1 (mostly autoregressive, fine-tuning)

**Prerequisite**: Bottleneck recovery. Introducing TF scheduling while the model is at 0% exact would prevent recovery entirely. Wait until exact match is above 80%.

---

## Active: V15.0/V15.1 Bottleneck Recovery Monitoring (2/23/2026)

**What to watch**:
- Type accuracy should climb above 90% first (decoder relearning cross-attention)
- Exact match should follow within 50-200 epochs
- Encoder metrics (Tc, Magpie, stoich, family) should remain stable (they were untouched)
- If exact match stalls below 50% after 200 epochs, consider restoring from `checkpoint_best_pre_v15_contraction.pt` backup
- **V15.1**: `[V15.1 Tc-BIN]` console output every 4 epochs shows high-Tc bin R² tracking. Verify snapshot/restore actions fire based on R² changes in 120-200K and >200K bins

**Changes applied in V15.0**:
- latent_to_memory bottleneck: 151M → 19M params (1024-dim, ~98.5% SVD variance)
- family_composed_14 added to decoder heads tokens (10 → 24 input dims)
- Fresh decoder optimizer (Adam state reset due to param shape changes)

**Changes applied in V15.1**:
- Per-bin Tc head early stopping (TcBinTracker): snapshot/restore tc_proj, tc_res_block, tc_out weights when high-Tc bin R² regresses >0.10 below best
