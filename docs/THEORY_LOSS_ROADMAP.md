# Theory Loss Roadmap: Phased Curriculum, Self-Consistency Learning, and Theory Validation

A design document for the role of physics-informed losses in the superconductor VAE training pipeline. Documents lessons learned from V12.22-V12.26 and outlines future directions.

Last updated: 2026-02-11

---

## 1. Lessons Learned: Why Theory Losses Failed During Training

### What Happened

Theory losses (V12.22-V12.25) injected physics-informed constraints into training: BCS Allen-Dynes, cuprate Presland dome, iron-based VEC constraints, heavy fermion log-normal prior, and organic soft caps. At `theory_weight=0.05`, they consumed **22% of the gradient budget**.

The result: theory losses **plateaued at ~1.43** for 500+ epochs and became static gradient fields that competed with the reconstruction signal. Exact match stalled at ~87%. When theory losses were disabled (V12.26), the gradient budget was freed for reconstruction and Tc accuracy.

### Root Cause

The theory losses encode *constraints on the output space*, not *instructions for learning*. They say "a heavy fermion SC with Tc > 20K is suspicious" — but this is only useful if the model can already accurately predict heavy fermion Tc. When applied during active learning, theory gradients compete with the reconstruction gradients that drive the model toward data mastery.

The fundamental error was treating theory losses as co-equal training objectives rather than as post-mastery regularizers.

### The Analogy

You don't grade a student's physics intuition before they can read. First master the data, then apply theoretical refinement. The basic CE/Tc/Magpie losses teach the model to "read" (reconstruct formulas, predict properties). Theory losses test whether the model's "reading" is physically consistent. Applying them simultaneously creates conflicting gradients.

---

## 2. Phased Curriculum for Theory Integration

### Phase 1: Data Mastery (Current — V12.26)

**Focus:** Reconstruction + Tc + Magpie + Stoich + HP

**Theory weight:** 0.0

**Goal:** Push exact match toward 95%+, reduce 100K+ Tc MAE to <3K, master high-pressure prediction.

**Rationale:** The model needs to learn the empirical data distribution before physics constraints can refine it. All gradient budget goes to data-driven learning.

### Phase 2: Theory as Regularizer (Future)

**Focus:** Re-enable theory losses at low weight after data mastery

**Theory weight:** 0.01-0.02 (was 0.05 — reduced to prevent gradient competition)

**Goal:** Gently shape the latent space so physically implausible predictions incur soft penalty. The model has already learned the data, so theory can't disrupt it — it can only refine.

**Key difference from V12.22-V12.25:** Theory is applied AFTER the model has converged on data reconstruction, not during active learning. The model's learned representation is stable, and theory losses provide small corrective gradients rather than competing objectives.

**Expected behavior:** Small improvements in Tc accuracy for edge cases (materials near family boundaries, unusual VEC, extreme doping). Theory should NOT change exact match (reconstruction is already mastered).

### Phase 3: Theory as Generation Filter (Future)

**Focus:** Use theory losses at inference/generation time, not training time

**Theory weight:** N/A (not in training loop)

**Goal:** Generate N candidates from the latent space, then rank/filter by physics consistency. Theory becomes a post-hoc scoring function.

**Implementation:** For each generated candidate:
1. Classify into SC family (BCS, cuprate, iron-based, heavy fermion, organic, unknown)
2. Compute family-specific theory score (Allen-Dynes consistency, Presland dome fit, VEC alignment, etc.)
3. Rank candidates by: Tc prediction confidence * theory consistency score
4. Flag candidates that score high on Tc but low on theory consistency as "potential discoveries" — materials that defy current theories

This approach uses theory in its most natural role: diagnostic evaluation of candidates, not gradient-based training.

---

## 3. Self-Consistency as Unsupervised Learning Signal

### The Insight

Once the model generates candidates, we can check whether they satisfy physics constraints WITHOUT needing labeled data. This is a form of self-supervised learning:

```
Model generates candidate → Theory checks consistency → Gradient flows back
```

Even **wrong** candidates provide useful training signal if the consistency check is informative. The model learns that its outputs must satisfy internal constraints, driving it toward physically plausible generations.

### How This Works

Consider a generated formula that the model assigns to the cuprate family with Tc = 95K:

1. **Presland dome check:** Does the predicted doping level place this material near the dome maximum? If Tc = 95K but predicted doping is p = 0.05 (far from optimal 0.16), the dome function says Tc should be ~40K. This inconsistency is a training signal — WITHOUT needing to know the true Tc.

2. **BCS Allen-Dynes check:** If a generated BCS material has predicted lambda = 0.3 but Tc = 50K, Allen-Dynes says this is impossible. The model learns that its lambda and Tc predictions must be mutually consistent.

3. **VEC constraint:** If a generated iron-based SC has VEC = 8.5, this is far from the optimal 6.0. The model learns to associate iron-based SC with appropriate electron counts.

### Why This Is Fertile Ground

- **No labels needed:** The physics constraints are mathematical relationships, not empirical labels. They apply to any generated candidate, real or hypothetical.
- **Scales with generation:** More generated candidates = more self-consistency signal. The model can train on its own outputs.
- **Complementary to CE loss:** CE loss teaches data reconstruction. Self-consistency teaches physical plausibility. These are orthogonal learning objectives that don't compete when applied to generated (not training) data.
- **Naturally curriculum-aware:** Early in training, the model generates garbage — self-consistency checks are meaningless. Late in training, the model generates plausible candidates — self-consistency checks become informative. The signal automatically scales with model capability.

### Implementation Sketch

```python
# Phase 3+: Self-consistency training loop
for epoch in range(n_epochs):
    # Standard supervised training
    supervised_loss = train_epoch_supervised(model, data)

    # Self-consistency on generated candidates
    z_samples = torch.randn(n_generated, latent_dim)  # Sample from latent space
    generated_formulas, predicted_tc, predicted_props = model.generate(z_samples)

    # Theory consistency check (no labels needed)
    family = classify_family(generated_formulas)
    consistency_loss = theory_loss(
        family, predicted_tc, predicted_props,
        magpie_features=compute_magpie(generated_formulas)
    )

    # Combined update
    total_loss = supervised_loss + consistency_weight * consistency_loss
    total_loss.backward()
```

---

## 4. Theory Validation via Model Accuracy

### The Meta-Insight

The trained model has internalized empirical patterns from 46,645 materials. Any physical theory that we apply as a loss function is effectively being **tested against the empirical data distribution**. This creates a novel framework for theory validation:

- **Theory improves model accuracy** → The theory captures real patterns in the data → Evidence FOR the theory
- **Theory destroys model accuracy** → The theory conflicts with empirical data → Evidence AGAINST the theory (or at least against its generality)
- **Theory has no effect** → The theory is either redundant with what the model already learned, or operates at a scale the model can't resolve

### Why This Matters

Traditional theory validation in condensed matter physics requires:
1. Derive prediction from theory
2. Synthesize material
3. Measure Tc experimentally
4. Compare prediction vs measurement

This is expensive, slow, and limited to materials that can actually be synthesized.

The ML approach enables a complementary validation path:
1. Train model on comprehensive empirical dataset
2. Apply theory as loss function
3. Measure impact on model accuracy across all known materials simultaneously
4. Draw conclusions about theory validity

This tests the theory against **thousands of materials at once**, not just one. A theory that improves accuracy across 10,000 BCS superconductors is strongly validated. A theory that improves accuracy for cuprates but destroys it for iron-based SCs reveals a boundary of applicability.

### Examples from Our Training

| Theory Component | Observation | Implication |
|-----------------|-------------|-------------|
| BCS Allen-Dynes (V12.25) | Plateaued at 1.43, no improvement over McMillan | Either the model already learned equivalent patterns from data, or the approximations (omega_log ~ 0.827*theta_D) introduce too much noise |
| Matthias VEC prior | Weight 0.01 had negligible effect | Either VEC correlation is already captured by Magpie features (feature 75 = NValence), or the Gaussian envelope is too loose to add signal |
| Cuprate Presland dome | Contributed to theory loss plateau | The dome function is well-established, but the learnable doping predictor may not extract accurate p from Magpie features |
| Heavy fermion Tc cap at 20K | Part of theory loss | Validates the empirical ceiling — the model's HF predictions should cluster below 20K if the theory is correct |

### Future Experiments

1. **A/B test individual theory components:** Enable ONE theory loss at a time (Phase 2) and measure per-family Tc MAE impact. This isolates each theory's contribution.

2. **Novel theory injection:** Propose a new theoretical relationship (e.g., a hypothesized correlation between atomic radius ratio and Tc in ternary compounds). Apply it as a loss. If it improves accuracy, it may capture real physics.

3. **Theory boundary detection:** Apply cuprate dome loss to ALL families, not just cuprates. Where it improves accuracy in non-cuprate families, those materials may share pairing mechanisms with cuprates — a discovery signal.

4. **Adversarial theory testing:** Inject a deliberately WRONG theory (e.g., "Tc decreases with VEC" — opposite of Matthias). Measure how badly it hurts. The magnitude of accuracy loss quantifies how strongly the data supports the correct theory.

---

## 5. Connection to Broader ML Research

### Self-Supervised Learning in Science

The self-consistency approach (Section 3) connects to several active research areas:

- **Physics-Informed Neural Networks (PINNs):** Enforce PDE constraints on neural network outputs. Our approach is similar but applies discrete physics rules (family classification + theory formulas) rather than continuous PDEs.
- **Consistency Regularization (Semi-Supervised Learning):** Require model outputs to be consistent under perturbation. Our version requires outputs to be consistent with physics.
- **Self-Play (RL):** The model generates candidates and evaluates them against physics — analogous to self-play where the model generates moves and evaluates them against game rules.

### Theory-Guided Machine Learning

The theory validation approach (Section 4) connects to:

- **Scientific Discovery via ML:** Using ML models to test hypotheses, not just make predictions.
- **Interpretable ML:** Theory losses that improve accuracy reveal which physics the model has (or hasn't) learned.
- **Active Learning for Science:** Using model-theory disagreement to identify materials that would be most informative to synthesize and test.

---

## 6. Summary: The Right Role for Theory

| Stage | Theory's Role | Mechanism |
|-------|--------------|-----------|
| **During data learning** | Stay out of the way | Theory weight = 0; let CE/Tc/Magpie drive learning |
| **After data mastery** | Gentle regularizer | Theory weight = 0.01-0.02; refine edges without disrupting learned patterns |
| **At generation time** | Filter/ranker | Score candidates by physics consistency; flag theory violations as discovery candidates |
| **Self-supervised loop** | Training signal | Check generated candidates for self-consistency; no labels needed |
| **Meta-analysis** | Theory validator | Measure how theory losses affect accuracy to test the theories themselves |

The most basic loss functions (CE, Tc regression, Magpie prediction) are the most powerful for learning because they directly encode the training objective. Theory losses are most powerful when used diagnostically — evaluating outputs rather than competing for gradient budget during learning.
