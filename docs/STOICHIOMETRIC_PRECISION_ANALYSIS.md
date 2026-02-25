# Stoichiometric Precision Analysis — V12.42 Autoregressive Decoder

**Date**: 2026-02-18
**Model**: V12.42 (epoch 3484, 50% wider decoder via Net2Net expansion)
**Data source**: `outputs/error reports/error_analysis_epoch_3484.json`
**Analysis script**: `scratch/stoich_precision_analysis.py`

---

## Executive Summary

The V12.42 autoregressive decoder achieves 97% exact match under teacher forcing but only 6% under true autoregressive generation. This report investigates whether the autoregressive "errors" represent chemically meaningful differences or fall within experimental measurement precision.

**Finding**: The model's stoichiometric errors are **well beyond experimental precision**. Only 8.6% of error formulas have all elements within EDS-grade tolerance (±0.1), and the median per-element difference of 0.10 (~20% relative) is 10x worse than ICP-OES and 4x worse than EDS. The errors represent genuinely different materials, not acceptable measurement noise.

**Root cause**: The digit-by-digit autoregressive fraction prediction cascades small token errors into large stoichiometric deviations. This motivates a fundamental change to semantic fraction tokenization.

---

## 1. Experimental Measurement Precision (Literature Benchmarks)

Stoichiometric composition of superconductors is determined experimentally using several techniques, each with characteristic precision:

| Technique | Typical Absolute Precision | Typical Relative Precision | Source |
|-----------|---------------------------|---------------------------|--------|
| ICP-OES | — | <1–2% RSD | [Springer: ICP-OES for HTS production control](https://link.springer.com/article/10.1007/s006040070113) |
| ICP-MS | — | ±3% relative | [Spectroscopy Online: ICP-MS matrix tolerance](https://www.spectroscopyonline.com/view/icp-ms-essential-steps-optimize-matrix-tolerance) |
| XRD Rietveld | ±0.01–0.02 in site occupancy | ~5 wt% | [Ebatco: X-Ray Diffraction](https://www.ebatco.com/laboratory-services/x-ray-diffraction/) |
| EDS/WDS | — | ±2–5% relative | Standard reference |
| Nominal (as-synthesized) | Exact by design | Actual may vary ±1–5% | — |

For the SuperCon dataset specifically, most compositions are **nominal** (i.e., the intended synthesis stoichiometry), so they are exact by convention. The question is whether the model's predicted stoichiometry would be distinguishable from the target using standard analytical techniques.

---

## 2. V12.42 Autoregressive Error Overview

From `error_analysis_epoch_3484.json`:

| Metric | Value |
|--------|-------|
| Total samples evaluated | 2,016 |
| Exact matches | 121 (6.0%) |
| Error records | 1,895 (94.0%) |
| Avg token errors per failed sample | 8.5 |

### Error Classification

| Category | Count | Percentage |
|----------|-------|------------|
| Same elements, different fractions only | 1,518 | 80.1% |
| Different element sets | 377 | 19.9% |

**80% of errors preserve the correct element set** — the model identifies the right chemistry but produces different stoichiometric ratios. This is the critical population for precision analysis.

---

## 3. Per-Element Stoichiometric Differences

### 3.1 All Errors (including element mismatches)

Across 7,542 per-element comparisons:

| Statistic | Absolute Difference | Relative Difference |
|-----------|--------------------|--------------------|
| Mean | 1.830 | 91.1% |
| Median | 0.200 | 26.0% |
| P25 | 0.000 | 0.0% |
| P75 | 1.000 | 85.0% |
| P90 | 3.238 | 105.0% |
| P95 | 7.000 | 260.0% |
| P99 | 27.354 | — |
| Max | 190.300 | — |

### 3.2 Same-Element Errors Only (pure stoichiometry differences)

Across 1,518 formulas with matching element sets:

| Statistic | Absolute Difference | Relative Difference |
|-----------|--------------------|--------------------|
| Mean | 1.704 | 95.2% |
| Median | 0.100 | 20.0% |
| P90 | 3.000 | 133.3% |
| P95 | 6.114 | 300.0% |

---

## 4. Comparison Against Experimental Bounds

### 4.1 Per-Element Fraction Within Bounds

What fraction of individual element predictions fall within each experimental precision level?

| Precision Level | Technique | Per-Element Within Bounds |
|----------------|-----------|--------------------------|
| ±0.01 | XRD (best case) | 38.5% |
| ±0.02 | XRD (typical) | 39.5% |
| ±0.05 | ICP-OES | 42.3% |
| ±0.10 | EDS/WDS | 46.5% |
| ±0.20 | Nominal tolerance | 51.0% |

### 4.2 Per-Formula: ALL Elements Within Bounds

A formula is only "within experimental error" if **every** element's stoichiometry falls within the tolerance. This is the practically relevant metric — one wrong element makes a different material.

| Precision Level | Formulas Within Bounds (all elements) |
|----------------|--------------------------------------|
| ±0.02 (XRD) | **1.3%** |
| ±0.05 (ICP-OES) | **3.7%** |
| ±0.10 (EDS/WDS) | **8.6%** |
| ±0.20 (nominal) | **13.4%** |

**Only 8.6% of error formulas would be indistinguishable from the target using EDS.** The remaining 91.4% represent detectably different materials.

### 4.3 Per-Formula Max Absolute Difference Distribution

The worst element difference per formula:

| Max Abs Diff | Count | Percentage | Cumulative |
|-------------|-------|------------|------------|
| <0.001 | 6 | 0.3% | 0.3% |
| <0.005 | 2 | 0.1% | 0.4% |
| <0.01 | 7 | 0.4% | 0.8% |
| <0.02 | 7 | 0.4% | 1.2% |
| <0.05 | 38 | 2.0% | 3.2% |
| <0.1 | 79 | 4.2% | 7.3% |
| <0.2 | 79 | 4.2% | 11.5% |
| <0.5 | 127 | 6.7% | 18.2% |
| <1.0 | 275 | 14.5% | 32.7% |
| ≥1.0 | 1,275 | **67.3%** | 100.0% |

**67% of error formulas have at least one element off by ≥1.0** in absolute stoichiometry. These are unambiguously different materials.

---

## 5. Error Mode Analysis

### 5.1 Fraction Simplification (~40% of errors)

The most common error mode is the decoder producing a simpler fraction that approximates the target:

| Target | Generated | Element | Target Value | Generated Value | Abs Diff | Rel Diff |
|--------|-----------|---------|-------------|----------------|----------|----------|
| `Er(13/25)` | `Er(2/5)` | Er | 0.520 | 0.400 | 0.120 | 23.1% |
| `Lu(7/10)` | `Lu(4/5)` | Lu | 0.700 | 0.800 | 0.100 | 14.3% |
| `Hg(13/20)` | `Hg(7/10)` | Hg | 0.650 | 0.700 | 0.050 | 7.7% |
| `Ag(499/500)` | `Ag(49/50)` | Ag | 0.998 | 0.980 | 0.018 | 1.8% |

These errors arise because the autoregressive decoder must predict each digit of the fraction independently (e.g., `1`, `3`, `/`, `2`, `5` for 13/25). A single wrong digit early in the fraction cascades into a completely different numerical value. The model appears to prefer "rounder" fractions with smaller denominators.

**Assessment**: Some of these are near ICP precision (the Ag example at 1.8%), but most (Er at 23%, Lu at 14%) are far beyond any experimental tolerance. These represent genuinely different doping levels that would produce different physical properties.

### 5.2 Scale/Convention Confusion (~15% of worst errors)

The decoder sometimes confuses percentage-based vs. normalized representations:

| Target | Generated | Issue |
|--------|-----------|-------|
| `In3Sn97` (at%) | `In(3/50)Sn(47/50)` (norm) | 100x scale error: 97 → 0.94 |
| `Bi(2262/25)Tl(238/25)` | `Bi(226/25)Tl(23/25)` | Decimal shift: 90.48 → 9.04 |
| `Al(99961/1000)Ge(39/1000)` | `Al(999/1000)Ge(...)` | 100x: 99.96 → 0.999 |

These are catastrophic errors where the model drops or adds a digit, changing the stoichiometry by orders of magnitude.

### 5.3 Integer Suffix Errors

A subset of errors involve large integer stoichiometries where a trailing digit is added or dropped:

| Target | Generated | Issue |
|--------|-----------|-------|
| `B12` | `B120` | Extra `0` appended |
| `Zr10` | `Zr100` | Extra `0` appended |
| `Pt9` | `Pt95` | Extra `5` appended |

These are purely autoregressive sequencing errors — the model fails to emit a STOP/next-element token at the right position.

---

## 6. Correlation with Sequence Length

The error analysis confirms that sequence length is the strongest predictor of autoregressive failure:

| Sequence Length | N Samples | Exact Match % | Avg Errors |
|----------------|-----------|---------------|------------|
| 1–10 tokens | 918 | 1.3% | 4.0 |
| 11–20 tokens | 498 | 5.0% | 8.9 |
| 21–30 tokens | 407 | 12.5% | 12.0 |
| 31–40 tokens | 167 | 15.6% | 16.3 |
| 41–60 tokens | 26 | 26.9% | 19.2 |

**Counterintuitive finding**: Longer sequences have *higher* exact match rates. This is because short sequences (1–10 tokens) are typically simple formulas like `Nb3Ge` where every token matters and there's no room for recovery, while longer sequences have more redundant structure (repeated element-fraction patterns) where the model can self-correct after a fraction error.

The `seq_len → error` correlation of +0.549 reflects that longer sequences accumulate more total token errors, but the per-token error rate may be lower.

---

## 7. Implications

### 7.1 The Autoregressive Exact Match Metric Is Misleading

The 6% autoregressive exact match does not mean the model produces wrong chemistry 94% of the time. It means:
- 80% of "errors" get the elements right but fractions wrong
- The fraction errors are primarily a tokenization artifact — predicting `1`, `3`, `/`, `2`, `0` sequentially when the semantic unit is `(13/20)`
- However, the numerical magnitude of these fraction errors **is chemically significant** and exceeds experimental precision

### 7.2 The Model Has Not Learned Precise Stoichiometry

Despite 97% teacher-forced exact match, the model cannot reliably reproduce precise stoichiometric ratios during free generation. The median relative error of 20% for same-element formulas means the model typically predicts doping levels that are off by a factor of ~1.2x — enough to shift Tc by 10–30K in cuprates.

### 7.3 Semantic Fraction Tokens Would Address the Root Cause

The current tokenization forces the decoder to solve two problems simultaneously:
1. **What fraction value?** (a chemistry problem)
2. **How to spell it in digits?** (a string formatting problem)

By making each fraction a single token (e.g., `(13/20)` as token #247), the model only needs to solve problem #1. This would:
- Reduce average sequence length by ~40–60% (fewer tokens per formula)
- Eliminate cascading digit errors within fractions
- Make the exact-match metric meaningful (1 wrong fraction = 1 token error, not 5+)
- Allow the model's high teacher-forced accuracy to transfer to autoregressive generation

### 7.4 Recommended Fraction Token Vocabulary

Based on the dataset, the fraction vocabulary should include:
- All fractions observed in training data (likely 200–500 unique fractions)
- Integer stoichiometries 1–12 (already tokens)
- Common reduced fractions: 1/2, 1/3, ..., 1/10, 2/3, 3/4, etc.
- Fine-grained doping fractions: 1/20, 3/20, 7/20, ..., 19/20, etc.
- High-precision fractions from the dataset: 499/500, 99961/1000, etc.

---

## 8. Conclusion

The V12.42 decoder's autoregressive stoichiometric errors are **not within experimental precision**. The model produces detectably different materials 91% of the time when it makes any error at all. The errors are dominated by the digit-level autoregressive fraction prediction, which cascades small token mistakes into large numerical deviations. Transitioning to semantic fraction tokens is the most direct path to resolving this fundamental limitation.

---

## References

1. [Determination of Stoichiometric Composition of HTS by ICP-OES](https://link.springer.com/article/10.1007/s006040070113) — Precision: <1–2% RSD
2. [X-Ray Diffraction — Ebatco](https://www.ebatco.com/laboratory-services/x-ray-diffraction/) — Rietveld refinement: ~5 wt%
3. [ICP-MS Matrix Tolerance — Spectroscopy Online](https://www.spectroscopyonline.com/view/icp-ms-essential-steps-optimize-matrix-tolerance) — ±3% relative
