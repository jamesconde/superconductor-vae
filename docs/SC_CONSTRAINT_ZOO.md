# Superconductor Constraint Zoo
## Physics-Grounded Consistency Rules for VAE Self-Supervised Training

**Purpose**: Two-part training signal:
- **Part A** — Formula consistency metrics (self-supervised, no labels needed, computed on model's own generations)
- **Part B** — SC fragility constraints (physics-grounded hard negatives, family-specific doping boundaries that separate SC from non-SC)

**Model context**: V12.38 / V12.41, holdout set covering 9 families × 5 samples each

---

## PART A: Universal Formula Consistency Metrics

These apply to **every generated formula regardless of family**. All can be computed with zero external labels — only the formula string is needed.

---

### A1. Duplicate Element Penalty
**Problem**: 20.2% of generated formulas contain the same element twice (e.g., `Au(1/2)Au(1/2)Ba2Ca2Cu3O`). Chemically nonsensical — gold is gold.

**Loss signal**: Binary flag per formula. If any element symbol appears more than once in the decoded token sequence, penalize.

```python
def duplicate_element_loss(formula_tokens):
    elements_seen = []
    for tok in formula_tokens:
        if tok.is_element_symbol:
            if tok.value in elements_seen:
                return 1.0  # hard penalty
            elements_seen.append(tok.value)
    return 0.0
```

**Training integration**: Back-propagate through decoder logit probabilities. After emitting element `E`, increase loss contribution from any subsequent logit that predicts `E` again.

**Expected impact**: Eliminates ~27K/133K of the string diversity that was being counted as "novel" compositions.

---

### A2. GCD Fraction Canonicality Loss
**Problem**: Model outputs `161/25 → 321/50` (denominator doubled) or `4/10` instead of `2/5`. These are notation artifacts — same mole fraction, different token sequences.

**Loss signal**: For each predicted fraction `p/q`, compute `gcd(p, q)`. If `gcd > 1`, the fraction is non-canonical. The loss is proportional to how far the prediction is from canonical form.

**Physics rationale**: Non-canonical fractions are never the "true" representation in training data — papers write `3/100`, not `6/200`. Making the target distribution unambiguous removes a major source of numden loss plateau.

**Implementation**:
```python
def gcd_canonicality_loss(num_pred, den_pred):
    # num_pred, den_pred are log1p predictions from numden_head
    num = round(expm1(num_pred))
    den = round(expm1(den_pred))
    g = gcd(num, den)
    # Penalize by how much reduction is possible
    return log(g) / log(max(den, 2))
```

**Preprocessing complement**: Also enforce GCD reduction in training data canonicalization — eliminate all non-reduced fractions from targets. This makes the problem unimodal for the decoder.

---

### A3. Site Occupancy Sum Constraint
**Problem**: On a mixed crystal site (e.g., Mg/Na in MgB2, Bi/Pb in BSCCO), the fractional occupancies of all species on that site must sum to 1.0 (or a simple integer). The model sometimes produces `Mg(19/20)Na(1/10)B2` where 19/20 + 1/10 = 1.05 — physically impossible.

**Loss signal**: Identify co-occupying elements by structural template family, then check that their fractions sum to the expected site total. Soft L1 loss on the deviation.

**Family-specific site definitions**:
- **MgB2**: Mg-site = {Mg, Li, Na, Al, Cr, Fe, Co, Ni, ...}. Sum must = 1.0
- **YBCO/RE-BCO**: RE-site = {Y, Eu, Nd, Sm, Gd, Dy, Ho, Er, Yb, Lu}. Sum must = 1.0. Ba-site = {Ba}. Fixed at 2.
- **BSCCO**: Bi-site = {Bi, Pb}. Sum must = 2.0. Sr-site = {Sr}. Fixed at 2.
- **Iron-1111**: RE-site = {La, Nd, Sm, Pr, Gd, Dy, Sm, Th, U}. Sum = 1.0. O-site = {O, F, H}. Sum = 1.0.

**Note**: This requires a family classifier signal or template detection based on element presence. Given that your family classifier is at 91%, this is tractable.

---

### A4. Stoichiometric Normalization Loss
**Problem**: Model generates `Mg2B4` and `MgB2` as distinct formulas when they are the same material. This creates 47K → 133K inflation in the novelty count.

**Loss signal**: Compute the GCD of all integer stoichiometric coefficients (after converting fractions to a common denominator). If `gcd > 1`, the formula is an integer multiple of a simpler form.

```python
def normalization_loss(stoich_vector):
    # stoich_vector = array of mole fractions (after converting to integers)
    integer_coeffs = to_integer_coefficients(stoich_vector)
    g = gcd_all(integer_coeffs)
    return float(g > 1)  # binary: is this an unnecessarily complex representation?
```

---

### A5. Round-Trip Cycle Consistency Loss
**Problem**: The VAE should be self-consistent — a generated formula re-encoded through the encoder should map back to a Z-vector close to the original, and predict the same Tc.

**Loss signal**: 
1. Take generated formula `F_gen` from Z-vector `z`
2. Re-encode `F_gen` → `z_reconstructed`  
3. Compute Tc from `z` (call it `Tc_original`)
4. Compute Tc from `z_reconstructed` (call it `Tc_recon`)
5. Loss = `|Tc_original - Tc_recon|` + `||z - z_reconstructed||²`

**Why this is powerful**: It catches the Ba/Sr mode collapse case. If `Hg(17/20)Re(3/20)Ba(83/50)Sr(17/50)Ca2Cu3O8` and `Hg(17/20)Re(3/20)Ba(36/25)Sr(14/25)Ca2Cu3O8` both decode to `Ba(8/5)Sr(2/5)` — the round-trip will show `Tc_recon ≠ Tc_original` because the two different Z-vectors encode different Tc predictions but the same formula. This is a direct signal to fix.

**Computational cost**: Requires a forward pass through the encoder per sample. Run on a subset (e.g., 10% of batch) to keep cost manageable.

---

### A6. Charge Balance Soft Constraint
**Problem**: Cuprates require a specific carrier density (hole count) to superconduct. Formulas with clearly impossible charge balance (e.g., net charge ≠ 0) are non-physical.

**Loss signal**: Use tabulated common oxidation states for each element. Compute formal charge of the predicted formula. The loss penalizes large deviations from charge neutrality.

**Element oxidation state table** (relevant elements):
```
Y: +3, La: +3, Ba: +2, Sr: +2, Ca: +2
Cu: +2 (nominal, +2.2 for optimally doped cuprates)
O: -2, F: -1
Bi: +3, Pb: +2 or +4, Tl: +1 or +3
Hg: +2, Re: +4-7 (mixed), Au: +1 or +3
Mg: +2, Li: +1, Na: +1, Al: +3
Fe: +2, As: -3, Nd: +3, Sm: +3, Dy: +3
Nb: +3, Al: +3, Ge: +4, Ga: +3
```

**Implementation note**: This is a soft constraint — use tanh gating to make it a smooth penalty rather than a hard cutoff, since cuprate superconductors deliberately maintain slight charge imbalance.

---

### A7. Physically Impossible Element Combinations
**Problem**: Some element pairs are mutually destabilizing at the compositional level and the model should never generate them together.

**Hard rule pairs** (empirically known):
- Fluorine + Thallium in cuprates (TlF₂ is not a superconductor substrate)
- Hydrogen as a main stoichiometric element with Cu-O planes (different family entirely)
- Iron + Copper as co-equal stoichiometric elements (these are competing magnetic orderers)
- Magnetic 3d elements (Mn, Fe, Co, Ni) on Cu sites in cuprates above ~2% — these are pair-breakers

**Loss signal**: Binary flag for element pair violations. These are cheap lookup table operations during decoding.

---

## PART B: Family-Specific SC Fragility Constraints

These are the physics-grounded hard negatives. For each family, we specify:
- The stoichiometric dimensions that control SC vs non-SC
- The specific numerical boundaries
- The known negative examples that should be labeled non-SC
- The error type they catch in the current model

---

### B1. YBCO / RE-BCO Family

**General formula**: `RE₁Ba₂Cu₃O_{7-δ}` where `RE` ∈ {Y, Eu, Nd, Sm, Gd, Dy, Ho, Er, Yb, Lu}

#### B1.1 Oxygen Stoichiometry Cliff (Most Critical)

**Physics**: The orthorhombic-to-tetragonal structural transition at δ ≈ 0.5 (O content = 6.5) destroys the CuO chain ordering that provides holes to the CuO₂ planes.

| O content (7 - δ) | δ value | State | Tc |
|---|---|---|---|
| 7.00 | 0.00 | Optimal SC | ~92 K |
| 6.93 | 0.07 | Optimal SC | ~95 K (peak) |
| 6.50 | 0.50 | Ortho-tetra transition | ~55 K |
| 6.35 | 0.65 | **SC boundary** | ~0 K |
| 6.00 | 1.00 | **Tetragonal insulator** | NON-SC |

**Hard constraints for VAE**:
- `O content < 6.35` (δ > 0.65) → **non-SC label**. This is a cliff, not a gradual decline.
- `O content > 7.05` → physically impossible (fully oxidized), formula invalid
- Optimal: O content ∈ [6.85, 7.00]

**In your formula representation**: `Y1Ba2Cu3O(7-x)` where `x = δ`. Your model generates fractions here. Constraints:
- `O(x)` where `x < 6.35` → non-SC hard negative
- `O(x)` where `x > 7` → invalid, penalize
- `O(x)` as a fraction: ensure denominator consistency — YBCO oxygen content appears as `O(69/10)` = 6.9, `O(161/25)` = 6.44, etc. All valid at different doping levels.

**Structures within YBCO family**:
- **YBa₂Cu₃O₇** (123): Most common. O range 6.35–7.0 for SC.
- **YBa₂Cu₄O₈** (124): Double CuO chains, stoichiometrically fixed at 8.0, Tc = 80 K. No oxygen deficiency possible — rigid stoichiometry.
- **Y₂Ba₄Cu₇O₁₅** (247): Mixed structure, Tc ~ 70 K. Less studied.
- **Y₃Ba₅Cu₈O₁₈** (your holdout case): Non-standard, Tc = 100 K. Very rare variant.

**Model error this catches**: `Y3Ba5Cu8O(179/10)` = 17.9 vs target O18. 17.9/2 sites = 8.95 per formula unit — physically plausible for this structure.

#### B1.2 RE Site Substitution Rules

RE elements are largely interchangeable in the Y site due to similar ionic radii (0.85–1.0 Å). All of these superconduct at similar Tc:
- **Valid RE**: Y, Eu, Nd, Sm, Gd, Dy, Ho, Er, Yb, Lu (all ~90-95 K with Ba₂Cu₃O₇)
- **Invalid RE substitution**: La on the Y site. La is too large (1.36 Å) and occupies the Ba site instead, creating LSCO-type compounds rather than YBCO-type.
- **Pr exception**: PrBa₂Cu₃O₇ is **non-superconducting**. Pr hybridizes with the Cu-O planes and localizes holes. This is a well-known negative.

**Hard negatives for VAE**:
- `Pr₁Ba₂Cu₃O₇` → non-SC. The model's element substitution error (e.g., Gd→Eu) is physically reasonable. But Pr→Y swaps would be catastrophic.
- `La on Y site with Ba on Ba site` → generates YBCO-type formula but wrong physics.

#### B1.3 Cu Site Constraints

Cu₃ in YBCO occupies two sites: Cu(1) in chains, Cu(2) in planes.
- **Replacing Cu with Zn**: Even 2-3% Zn on Cu sites strongly suppresses Tc. ZnBa₂Cu₃O₇ with x > 0.1 is non-SC.
- **Fe on Cu site**: Fe > 5% kills SC. `YBa₂(Cu₁₋ₓFeₓ)₃O₇` with x > 0.05 → non-SC.
- **Cu must be present**: No Cu in a YBCO-type formula → non-SC by definition.

---

### B2. LSCO Family (La-based cuprates)

**General formula**: `La_{2-x}Sr_xCuO_4` and variants with other rare earths or charge reservoir layers

#### B2.1 The Doping Dome — Sharpest Constraint in All Cuprates

**Physics**: LSCO has the most carefully characterized phase diagram of any cuprate. SC exists only in x ∈ [0.055, ~0.27].

| Sr content (x) | State | Tc (approx) |
|---|---|---|
| 0 | AF insulator | NON-SC |
| 0.02 | Spin glass | NON-SC |
| 0.05 | Quantum vortex liquid onset | ~0 K (barely SC) |
| **0.055** | **SC onset** | **~10 K** |
| 0.10 | Underdoped SC | ~30 K |
| 0.15 | Optimal doping | ~38 K (Tc max) |
| 0.19 | Lifshitz transition | ~36 K |
| 0.21 | Overdoped | ~25 K |
| **0.27** | **SC cutoff** | **~0 K** |
| 0.30 | Normal metal | NON-SC |
| 0.35 | Strongly overdoped | NON-SC |

**Hard constraints for VAE** (LSCO-type specifically):
- `Sr < 0.05` → non-SC (antiferromagnet)
- `Sr > 0.27` → non-SC (overdoped metal)
- Optimal SC window: Sr ∈ [0.12, 0.20]
- **Special case x = 1/8** (0.125): This is the "1/8 anomaly" — charge stripe order competes with SC, causing anomalous dip in Tc. Not non-SC, but suppressed.

**In your formula terms**: `La(2-x)Sr(x)CuO4`. If model generates `La(17/10)Sr(3/10)CuO4` (x = 0.3) → non-SC hard negative.

#### B2.2 Electron-Doped Variants

`La_{2-x}Ce_xCuO_{4-δ}` (LCCO): Electron-doped cuprate. Very different dome.
- SC exists only for x ∈ [0.10, 0.18]
- x < 0.10: antiferromagnetic insulator
- x > 0.18: overdoped metal, SC collapses
- Requires oxygen reduction (δ > 0) to superconduct — as-synthesized LCCO is not SC; needs annealing in reducing atmosphere.

**Hard negative**: `La₂CeO₄` with no O deficiency → non-SC even with Ce doping.

#### B2.3 La₂CuO₄ Variants

- `La₂CuO₄` (parent): **Non-SC antiferromagnet**. Tc = 0.
- `La₂CuO_{4+δ}` (excess oxygen): SC at ~35 K, but only in a narrow range 0.02 < δ < 0.12.
- `La₁Ba₂Cu₃O_x` (your holdout #7): Hybrid LSCO/YBCO structure. Valid.

---

### B3. Bi-Cuprate Family (BSCCO)

**General formula**: `Bi₂Sr₂Ca_{n-1}Cu_nO_{2n+4+x}` where n = 1, 2, 3

| Phase | Formula | Tc |
|---|---|---|
| Bi-2201 | Bi₂Sr₂CuO₆₊ₓ | ~20 K |
| Bi-2212 | Bi₂Sr₂CaCu₂O₈₊ₓ | ~95 K |
| Bi-2223 | Bi₂Sr₂Ca₂Cu₃O₁₀₊ₓ | ~110 K |

#### B3.1 Hole Doping via Excess Oxygen

**Physics**: BSCCO requires hole doping via excess oxygen `+x`. The max Tc for Bi-2212 occurs at **0.16 holes per Cu atom**.

- Underdoped (x too low): Tc falls. Fully stoichiometric `Bi₂Sr₂CaCu₂O₈` without excess O → very low Tc or non-SC.
- Optimally doped: x ≈ 0.16 holes/Cu
- Overdoped (x too high): Tc falls. Superconductivity is completely destroyed at extreme overdoping.

**Hard constraints**: BSCCO in your VAE appears as fractions on Bi, Pb sites, not as oxygen excess (since O is usually a fixed integer or near-integer). The model generates things like `Bi(9/5)Pb(1/5)Sr2Ca2Cu2O` where O is implicitly +x. This makes the oxygen doping implicit in the Bi:Pb ratio and the Ca content.

#### B3.2 Pb Substitution Rules

**Pb is uniquely beneficial for Bi-2223 formation** — it kinetically stabilizes the n=3 phase. Optimal Pb content: ~15-20% of the Bi site.

| Pb fraction | Effect |
|---|---|
| 0 | Bi-2223 forms slowly, often impure |
| 0.1–0.4 | Optimal range, stabilizes 2223 |
| > 0.5 | Pb-rich secondary phases form, SC degrades |
| 1.0 (all Pb, no Bi) | Non-SC |

**For the model**: `Bi(1/5)Pb(9/5)Sr2Ca2Cu3O` (90% Pb substitution) → non-SC or severely degraded. The model correctly generates ranges like `Bi(17/10)Pb(3/10)` (15% Pb) which is in the optimal window.

**Hard constraint**: Pb/Bi ratio > 0.5 on the Bi-site → flag as likely non-SC or degraded SC.

#### B3.3 Ca Layer Count vs. Cu Layer Count Must Match

**Physics**: The formula unit requires `n-1` Ca atoms for `n` Cu planes. The n=3 phase requires **2 Ca** for **3 Cu**. Mismatches destroy the crystal structure.

**Hard rule**:
- n=1: No Ca (or Ca ≈ 0), Cu ≈ 1 → Bi-2201, Tc ~ 20K
- n=2: Ca ≈ 1, Cu ≈ 2 → Bi-2212, Tc ~ 95K
- n=3: Ca ≈ 2, Cu ≈ 3 → Bi-2223, Tc ~ 110K

**Error this catches**: `Bi₂Sr₂Ca₂Cu₂O₈` — 2 Ca but only 2 Cu layers — n=2 Ca count with n=2 Cu count, but the 2 Ca implies expecting n=3. This structure is **non-SC** (wrong interlayer coupling).

**For the VAE**: When the model generates Bi-type formulas, Ca and Cu stoichiometries must satisfy `n_Ca ≈ n_Cu - 1`. Floating point: if `abs(Ca - (Cu - 1)) > 0.3` → penalize as structurally inconsistent.

---

### B4. Hg-Cuprate Family (HBCCO)

**General formula**: `HgBa₂Ca_{n-1}Cu_nO_{2n+2+δ}` — highest Tc cuprates under ambient conditions (Tc up to 135 K for n=3)

#### B4.1 Same Ca/Cu Layering Rule as BSCCO

n-1 Ca layers for n Cu planes. Identical constraint structure to BSCCO B3.3.

| n | Formula | Tc |
|---|---|---|
| 1 | HgBa₂CuO₄₊δ | ~97 K |
| 2 | HgBa₂CaCu₂O₆₊δ | ~128 K |
| 3 | HgBa₂Ca₂Cu₃O₈₊δ | ~135 K |

#### B4.2 Hg Site Substitution (Re, Pb, Au, Cr, V)

The Hg site can be partially substituted. This is your most complex holdout family.

**Stabilizing substitutions** (partial replacement, improves Tc or stability):
- Re: `Hg(1-x)Re(x)` for x ≤ 0.25 — known to boost stability, slight Tc effect
- Pb: `Hg(1-x)Pb(x)` for x ≤ 0.3 — similar role to Bi-cuprate
- Au: `Hg(1-x)Au(x)` for x ≤ 0.2 — rare but known

**Destabilizing at high substitution**:
- Re > 0.4: Rhenate phase formation, not superconducting cuprate
- V on Hg site: V is magnetic (V²⁺, V³⁺), pair-breaking. V > 0.3 → non-SC

**Hard constraints**:
- `V(1/2)` on Hg site (50% V substitution, as in your holdout #20): This is at the edge — the holdout model predicts SC with Tc = 110 K. **Physically this is borderline**. V amounts > 30% of the Hg site are not experimentally confirmed as SC. This is your model being optimistic about an edge case.
- Li on Hg site (`Tl(1/2)Pb(1/2)Sr(2)Ca(2)Cu(3)Li(1/5)O`, your hardest case): **Li is not a standard cuprate dopant**. Li is small (0.76 Å vs Hg 2+ 1.02 Å) and would disrupt the Hg-O plane. This explains why Li dopant is "poorly reconstructed" — the model is right to struggle here, because this formula is chemically unusual.

#### B4.3 Ba/Sr Ratio (The Mode Collapse Case)

Your mode collapse (#14, #15) — `Ba(83/50)` vs `Ba(36/25)` — both decode to `Ba(8/5)`.

**Physics**: In Hg-1223, the Ba/Sr ratio controls the effective lattice parameter of the blocking layer. This shifts the Cu-O plane distance and changes hole doping indirectly. The difference between Ba:Sr = 83:17 and Ba:Sr = 72:28 is physically real and affects Tc.

**Constraint for training**: The model needs explicit `||z₁ - z₂||` > threshold signal for these two formulas. The round-trip consistency loss (A5) would directly address this — when the decoder collapses two different Z-vectors to the same formula, the re-encoded Z vectors would be identical, but the original Z vectors are different. The Tc prediction gap between the two should create a gradient.

---

### B5. Tl-Cuprate Family (TBCCO)

**General formula**: `Tl₂Ba₂Ca_{n-1}Cu_nO_{2n+4}` (double Tl) or `TlBa₂Ca_{n-1}Cu_nO_{2n+3}` (single Tl)

#### B5.1 Ca/Cu Layering Rule

Same as BSCCO and HBCCO. n-1 Ca for n Cu.

| Phase | n | Tc |
|---|---|---|
| Tl-2201 | 1 | ~90 K |
| Tl-2212 | 2 | ~108 K |
| Tl-2223 | 3 | ~125 K |

#### B5.2 Tl Toxicity and Substitution

Tl is highly toxic but forms stable superconductors. Partial Pb substitution on Tl site is common and beneficial.

**Valid substitutions on Tl site**:
- Pb: 20-50% Pb → standard, well-characterized SC
- Bi: Small amounts, stabilizing
- Hg: Rare hybrid structures

**Pair-breaking on Tl site**:
- Magnetic elements (Mn, Fe, Co, Ni): Same pair-breaking rule as cuprate Cu-site substitution. Even 1-2% kills SC significantly.
- V: `Tl(3/5)V(1/2)Sr2Ca2Cu3O` (your holdout #20): V occupying 50% of a cation site is extreme. V has variable oxidation states (+2 through +5) and is magnetic at many states. **This material is a borderline hard negative** — the model predicts CUPRATE_OTHER rather than TBCCO, which is correct. SC at Tc=110K for this formula is experimentally unconfirmed.

#### B5.3 Rare Dopants That Kill SC in Tl-Cuprates

**Li**: Li in `Tl(1/2)Pb(1/2)Sr2Ca2Cu3Li(1/5)O` — the worst similarity score in your holdout (0.9177). Li at 20% of a cation site is unusual. Li⁺ (0.76 Å) would go to a site with poor size match in the Tl-cuprate structure. The model correctly struggles here — this formula is chemically borderline.

**U**: `Tl(1/2)Pb(1/2)Sr(4/5)Ba(1/5)Ca2Cu3U(1/25)O` — U as a dopant at 4% of a site. U(IV) (0.89 Å) could substitute on the Ba-type or Ca-type site. Low amounts (2-5%) are known to not kill SC. **U(1/25) = 4% → marginal SC, model correctly predicts SC.**

**Hard constraint for Tl-cuprates**:
- Magnetic 3d transition metals > 10% of any cation site → non-SC flag
- V > 30% → non-SC flag
- Li > 10% on structural cation site → non-SC flag (Li goes to Li-doped structures like Li₂TiO₃, not cuprates)

---

### B6. Iron-Based Pnictide Family (1111-type)

**General formula**: `RE₁Fe₁As₁O_{1-x}F_x` or `RE₁Fe₁As₁O_{1-δ}` where RE ∈ {La, Nd, Sm, Pr, Ce, Gd, Dy}

#### B6.1 Electron Doping Requirement

**Physics**: Unlike cuprates (hole-doped), 1111 pnictides require **electron doping** to become SC. The parent `REFeAsO` is a spin-density-wave antiferromagnetic metal. SC emerges when electrons are added by:
- F substitution for O: `O_{1-x}F_x`, typically x ∈ [0.05, 0.25]
- Oxygen deficiency: `O_{1-δ}`, typically δ ∈ [0.05, 0.20]
- RE site: Th, U, Zr substitution for RE³⁺ (electron doping)

**The SC dome for La-1111** (LaFeAsO₁₋ₓFₓ):
| F content x | State | Tc |
|---|---|---|
| 0 | AFM metal (SDW) | NON-SC |
| 0.03 | SDW suppressed | ~17 K onset |
| 0.05-0.12 | **Optimal SC** | **~26 K** |
| 0.14-0.20 | Overdoped (flat dome) | ~20-26 K |
| 0.25+ | Overdoped | Declining |

**For Sm-1111** (your family): Tc peaks at ~55 K, dome similar shape but higher.

**Hard constraints**:
- `SmFeAsO` with no F, no O-deficiency (x = 0, δ = 0) → **non-SC**. Parent compound.
- `SmFeAsO(1/20)` → δ = 0.95, meaning nearly all O removed → **non-SC** (structure collapses)
- O content below 0.7 of nominal → **non-SC** (too much vacancy, structural instability)

**In your formula representation** (`RE1Fe1As1O(17/20)` = O₀.₈₅ = 15% O-deficiency):
- O fraction ∈ [0.75, 0.95] → SC window
- O fraction < 0.7 → non-SC
- O fraction = 1.0 (undoped) → non-SC (parent compound)

#### B6.2 RE Site Validity

**Valid RE for 1111**: La, Nd, Sm, Pr, Ce, Gd, Dy, Ho (all trivalent, ionic radii ≈ 0.9–1.1 Å)

**Invalid RE substitutions**:
- **Non-trivalent substitutions**: Th⁴⁺ and U⁴⁺ on RE site provide electron doping — this is valid and gives SC (Tc up to 56 K in Gd₁₋ₓThₓFeAsO).
- **Ba or Sr on RE site**: These are 2+ ions and provide hole doping. `Ln_{1-x}SrxFeAsO` gives lower Tc (< 25 K) and is less effective than F-doping.
- **Rare earths with magnetic moments at very low T** (e.g., Ce, Pr): The RE magnetic ordering can compete with SC at low T. Ce-1111 and Pr-1111 superconduct but with complications.

#### B6.3 As and Fe Site Constraints

**Fe site**: Fe₁ is essentially fixed. Substitutions on Fe site:
- Co on Fe: `Fe_{1-x}Co_x` at 10-20% → SC by electron doping (alternative to F). Co-1111 is a valid SC family.
- Ni on Fe: Similar to Co, but lower Tc.
- Cr, Mn: Strongly magnetic, pair-breaking. Cr/Mn > 5% → non-SC.

**As site**: As is nearly fixed. Partial P substitution (As→P) reduces Tc due to smaller pnictogen height:
- `Fe₁As_{1-x}P_x` with x = 1/20 (your holdout #30): Valid — P-substitution known to reduce Tc moderately.
- Full P substitution (LaFePO): Tc = 5K only — much lower than As-based.

---

### B7. MgB₂ Family

**General formula**: `Mg_{1-x}M_xB_2` (Mg-site doping) or `Mg₁(B_{1-y}X_y)₂` (B-site doping)

This is your **best-performing family** (4/5 exact, 0.15K MAE). Constraints here are about not breaking what works and extending it with edge cases.

#### B7.1 B-Site Doping Limits (Carbon)

**Physics**: Carbon substitutes for boron. It's an electron dopant that fills the σ-band, reducing DOS at Fermi level and suppressing Tc.

| C content (x in B_{1-x}C_x) | Tc |
|---|---|
| 0 | 39 K (pure) |
| 0.02 | ~35-37 K |
| 0.05 | ~30 K |
| 0.10 | ~20-25 K |
| **0.125** | **~10 K** |
| **> 0.125** | **Non-SC** (SC completely suppressed) |

**Solubility limit**: C in MgB₂ ≈ 15% maximum (structural stability limit).

**Hard constraint**: `Mg(B_{1-x}C_x)₂` with x > 0.125 → **non-SC**. This is a sharp cliff.

**In your formula**: `Mg1B(17/20)C(3/20)B` type formulas — `C(3/20)` = 15% → borderline non-SC.

#### B7.2 Mg-Site Doping Limits

**Aluminum** (Mg-site, isovalent with extra electron):

| Al content (x in Mg_{1-x}Al_x) | Tc |
|---|---|
| 0 | 39 K |
| 0.1 | ~32 K |
| 0.2 | ~24 K |
| 0.3 | ~15 K |
| 0.5 | ~5-8 K |
| **> 0.5** | **Non-SC** |

**Hard constraint**: `Mg(1-x)Al(x)B₂` with x > 0.5 → **non-SC**.

**Alkali metals (Li, Na, K)** on Mg site:
- Li: Provides hole doping (beneficial up to ~20%). `Mg(9/10)Li(1/10)B2` (your exact match) ✓
- Na: Provides hole doping, similar to Li but less soluble. `Mg(97/100)Na(3/100)B2` (your exact match at 3%) ✓
- `Mg(1/2)Na(1/2)B₂` (50% Na): Na has limited solubility in MgB₂ matrix — this formula is **probably non-SC** (phase separation before reaching this concentration).

**Transition metal dopants** (Mn, Fe, Co, Ni on Mg site):
- Any magnetic element > 2-3% → **strong SC suppression** (Abrikosov-Gorkov pair-breaking via spin-flip scattering)
- `Mg(95/100)Mn(5/100)B₂` → strongly suppressed, likely non-SC
- Mn is the worst: even 0.5% Mn reduces Tc dramatically

**Hard constraint for your model**: If the dopant on the Mg site is a magnetic 3d metal (Cr, Mn, Fe, Co, Ni), amounts > 5% → **non-SC hard negative**.

#### B7.3 B/Mg Ratio Must Be 2:1

**Physics**: The AlB₂-type structure of MgB₂ requires exactly 2 boron per Mg. Deviation destroys the hexagonal boron planes that support the σ and π bands responsible for SC.

**Hard rule**: In `Mg_aB_b`, the ratio `b/a` must be very close to 2.0. Deviations > 10% → structure becomes non-SC or different phase.

- `Mg1B2` → valid (Tc = 39 K)
- `Mg1B(3/2)` → B/Mg = 1.5 → **non-SC** (wrong phase)
- `Mg1B(5/2)` → B/Mg = 2.5 → **non-SC** (excess B forms MgB₄ or other phases)

---

### B8. Conventional / BCS / A15 Family

**General formula for A15**: `A₃B` where A = transition metal (Nb, V, Mo), B = non-magnetic element (Sn, Al, Ge, Ga, Si, Sb, In)

#### B8.1 Stoichiometry Criticality (A:B = 3:1 is Rigid)

**Physics**: A15 compounds have the Cr₃Si-type structure with a **rigid 3:1 composition requirement**. The transition metal (A) forms linear chains along crystal faces that create the high DOS responsible for SC. Off-stoichiometry disrupts chain ordering.

| Composition | State |
|---|---|
| `Nb₃Al` (exact 3:1) | SC, Tc = 18-19 K |
| `Nb₂.₈Al₁.₂` (off-stoich) | Lower Tc due to constitutional disorder |
| `Nb₂Al` (2:1) | **Non-SC** — wrong phase (σ-phase NbAl) |
| `NbAl` (1:1) | **Non-SC** — different compound entirely |

**Hard constraint for your model**: For A15 compounds:
- A:B ratio must be within ±10% of 3:1
- Significant deviation → non-SC (phase separation, wrong crystal structure)

**For your holdout**:
- `Nb3Al(71/100)Ge(29/100)` — Nb₃(AlGe) with Al:Ge = 71:29. This is the A15 structure with mixed B-site occupancy. Total B = 1. Valid.
- `Nb(79/100)Al(41/250)Ga(57/1000)` — your integer form case. Converting: Nb = 0.79, Al = 0.164, Ga = 0.057. Total non-Nb = 0.221. Nb/B_total = 0.79/0.221 ≈ 3.57:1. **Slightly off from 3:1 but within the A15 homogeneity range**.

#### B8.2 A15 Homogeneity Ranges

The A15 phase exists only in a narrow composition window:

| Compound | SC | A15 stability range (at.% B) |
|---|---|---|
| Nb₃Sn | 18 K | 24-26 at.% Sn (narrow!) |
| Nb₃Al | 18-19 K | 19-26 at.% Al (at high T); 19-23 at.% at 1000°C |
| Nb₃Ge | 23 K | 15-19 at.% Ge (off-stoichiometric by nature) |
| Nb₃Ga | 20 K | Narrow, requires special synthesis |
| V₃Si | 17 K | Wider range, easier to make |
| V₃Ga | 15 K | Moderate range |

**Important Nb₃Ge note**: Nb₃Ge does **not exist at stoichiometry in equilibrium**. The SC phase is always slightly Ge-poor. Your model's `Nb(79/100)Al(41/250)Ga(57/1000)` formula (with actual Nb:B ratio ~3.5:1) is consistent with this off-stoichiometric reality for mixed A15 phases.

**Hard negatives for A15**:
- `Nb₃Cr` — Cr is magnetic, no A15 SC exists
- `Nb₃Fe` — Fe is magnetic, non-SC
- `Nb₃Mn` — Mn is magnetic, non-SC
- `Nb₂Al` or `NbAl₃` — wrong stoichiometry phases, non-SC

**Constraint**: If A-site is Nb/V/Mo and B-site is a **magnetic 3d element** (Mn, Fe, Co, Ni, Cr) → **non-SC**, regardless of stoichiometry.

#### B8.3 Binary vs. Ternary A15

`Nb₃Al` is binary. `Nb₃(Al₁₋ₓGeₓ)` = ternary with mixed B-site. This is valid and well-characterized:
- `Nb₃Al(0.8)Ge(0.2)` → Tc ≈ 20.3 K (higher than pure Nb₃Al)
- Mixed B-site occupancy must still sum to ≈ 1 total B.

**For your holdout**: `Nb(79/100)Al(71/100)Ge(29/100)` → Sum of non-Nb = 1.0. This satisfies the B-site constraint perfectly.

---

## Summary: Constraint Priority Matrix

| Constraint | Type | Families | Expected Impact |
|---|---|---|---|
| A1 Duplicate element penalty | Self-supervised | All | Fixes 20.2% of generated formulas |
| A2 GCD canonicality | Self-supervised | All | Resolves denominator doubling (~60% of near-misses) |
| A3 Site occupancy sum | Self-supervised | MgB₂, cuprates | Prevents fraction > 1.0 on shared sites |
| A4 Stoichiometric normalization | Self-supervised | All | Reduces 133K→47K inflation |
| A5 Round-trip consistency | Self-supervised | All | Catches mode collapse (Ba/Sr case) |
| A6 Charge balance | Self-supervised | Cuprates primarily | Catches wild stoichiometry |
| A7 Impossible element pairs | Self-supervised | All | Hard lookup table |
| B1 YBCO O content | Physics hard negative | YBCO | δ > 0.65 → non-SC |
| B2 LSCO doping dome | Physics hard negative | LSCO | x < 0.055 or x > 0.27 → non-SC |
| B3 BSCCO Ca/Cu layering | Physics hard negative | Bi-cuprate | Ca ≠ Cu-1 → non-SC |
| B4 HBCCO Ba/Sr + V limit | Physics hard negative | Hg-cuprate | V > 30% → non-SC |
| B5 Tl-cuprate rare dopants | Physics hard negative | Tl-cuprate | Li, V > 30% → non-SC |
| B6 Pnictide doping dome | Physics hard negative | Iron-based | O = 1.0 undoped → non-SC |
| B7 MgB₂ C substitution | Physics hard negative | MgB₂ | C > 12.5% → non-SC |
| B7 MgB₂ Al substitution | Physics hard negative | MgB₂ | Al > 50% → non-SC |
| B7 MgB₂ magnetic dopant | Physics hard negative | MgB₂ | Mn/Fe/Co > 5% → non-SC |
| B8 A15 stoichiometry | Physics hard negative | Conventional | A:B ≠ 3:1 → non-SC |
| B8 A15 magnetic B-site | Physics hard negative | Conventional | Magnetic 3d on B-site → non-SC |

---

## Implementation Roadmap

### Phase 1 (Immediate, no new data needed)
1. Implement **A1** (duplicate element) + **A2** (GCD canonicality) in decoder training loop
2. Enforce GCD reduction in training data preprocessing — canonicalize all fractions
3. Implement **A5** (round-trip consistency) on 10% of batch

### Phase 2 (Self-supervised, next checkpoint)
4. Add **A3** (site occupancy sum) using family classifier outputs to identify site groups
5. Add **A4** (normalization) to generation postprocessing — collapse integer multiples before novelty counting
6. Add **A6** (charge balance) as soft loss

### Phase 3 (Physics hard negatives)
7. Generate edge-case formulas for each family at the documented boundaries (x = 0.05, 0.27 for LSCO; δ = 0.65 for YBCO; C = 0.125 for MgB₂, etc.)
8. Label these as non-SC with high confidence
9. Train SC classifier with these as negative examples — tightens the SC/non-SC boundary exactly where the model is currently overconfident

### Evaluation Metric
Track the **sensitivity landscape**: for each holdout target, measure the minimum stoichiometric perturbation that flips the SC classifier from SC to non-SC. After Phase 3 training, the flip boundary should align with the physics-documented values above.

---

*Last updated: 2026-02-18. Research basis: experimental literature on YBCO (O stoichiometry), LSCO (doping dome), BSCCO (Pb/Ca constraints), iron-1111 (F/O doping), MgB₂ (C/Al suppression), A15 compounds (Nb₃Al/Ge stoichiometry).*
