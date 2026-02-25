# Element-Aware SLERP Design Document

**Created**: 2026-02-25
**Status**: Future Enhancement (not yet implemented)
**Depends On**: Element-Anchored Z-Space Sampling (Strategy 4) — implemented in `self_supervised.py`

---

## Motivation

The current SLERP strategy (Strategy 3) pairs z-vectors by **family label** — e.g., YBCO with YBCO,
iron-based with iron-based. This is chemically coarse: two YBCO superconductors might have very
different dopants, site substitutions, or oxygen stoichiometries. Meanwhile, a YBCO and an LSCO
sample might share 3 out of 4 elements (Cu, O, Ba/La).

**Element-overlap pairing** would replace family-based SLERP with chemistry-based SLERP:
interpolate between samples that share actual elements, not just a broad family classification.
This creates interpolations that traverse chemically plausible regions of z-space.

---

## Current SLERP Implementation (Strategy 3)

```python
# In ZSpaceSampler._sample_slerp():
# 1. Build family_indices: family_id -> list of sample indices
# 2. For each family, pick random pairs within the family
# 3. SLERP between pair members at t in [0.1, 0.2, ..., 0.9]
```

**Limitation**: Family labels are a coarse proxy for chemical similarity. Two materials in the
same family can be chemically distant, while materials in different families can share elements.

---

## Proposed Change: Element-Overlap SLERP

### Algorithm

1. Reuse the inverted index from element-anchored sampling (`_element_to_samples`, `_element_sets`)
2. For each SLERP sample to generate:
   a. Pick an anchor (coverage-weighted, as now)
   b. Find neighbors sharing >= 2 elements (reuse `_find_element_neighbors()`)
   c. SLERP between anchor and a random neighbor at random t in [0.1, 0.9]
3. If no neighbors found: fall back to current family-based pairing

### Code Changes

**File**: `src/superconductor/training/self_supervised.py`

**Method**: `ZSpaceSampler._sample_slerp()`

```python
def _sample_slerp(self, n: int) -> torch.Tensor:
    """Strategy 3: SLERP — element-overlap pairing when available, family fallback."""
    if self._has_element_data:
        return self._sample_slerp_element_aware(n)
    return self._sample_slerp_family(n)  # Current implementation

def _sample_slerp_element_aware(self, n: int) -> torch.Tensor:
    results = []
    t_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for i in range(n):
        anchor_idx = self._weighted_sample_idx(1).item()
        anchor_elems = self._element_sets[anchor_idx]
        neighbors = self._find_element_neighbors(anchor_elems, anchor_idx, min_shared=2)

        if neighbors:
            nb_idx = neighbors[np.random.randint(len(neighbors))]
            t = t_values[i % len(t_values)]
            z_interp = slerp(self._z_cache[anchor_idx], self._z_cache[nb_idx], t)
            results.append(z_interp)
        else:
            # Fallback: random pair SLERP
            pair_idx = self._weighted_sample_idx(2)
            t = t_values[i % len(t_values)]
            z_interp = slerp(self._z_cache[pair_idx[0]], self._z_cache[pair_idx[1]], t)
            results.append(z_interp)

    return torch.stack(results)
```

### Config

No new config keys needed — reuses existing `phase2_element_min_shared`.

---

## Validation Plan

1. **A/B comparison**: Run Phase 2 for 50 sub-epochs with family-SLERP vs element-SLERP
   - Primary metric: `phase2_valid_rate` (expect higher with element-aware)
   - Secondary: `phase2_unique_rate` (element-aware should produce more diverse formulas)
2. **Interpolation quality**: For 100 random SLERP pairs, decode at t=0.0, 0.25, 0.5, 0.75, 1.0
   - Check: do intermediate formulas share elements with endpoints?
   - Family-SLERP: many intermediates may not parse (crossing family boundaries in z-space)
   - Element-SLERP: intermediates should more often share elements with both endpoints
3. **Holdout impact**: Compare holdout exact match after 200 Phase 2 epochs with each strategy

---

## Risks

1. **Reduced diversity**: Element-overlap pairs are more similar than random pairs. SLERP
   interpolations may stay too close to known materials and not explore novel chemistry.
   Mitigation: keep PCA walk (Strategy 4) unchanged for uninformed exploration.

2. **Computational cost**: `_find_element_neighbors()` is called per SLERP sample (vs. per
   element-anchored sample). For n_slerp=64 this is negligible (<1ms total), but worth noting.

3. **Overlap with Strategy 2**: Element-anchored sampling already uses element neighbors for
   30% SLERP. If Strategy 3 also uses element neighbors, the two strategies overlap.
   Mitigation: Strategy 2's SLERP is anchor-to-neighbor (fixed anchor chosen by element cycling);
   Strategy 3's SLERP is peer-to-peer (both endpoints are coverage-weighted). Different anchor
   selection = different z-space regions explored.

---

## Decision

**Deferred**. Implement element-anchored sampling (Strategy 4) first and measure its impact.
If `phase2_valid_rate` improves significantly, element-aware SLERP can amplify the effect.
If not, the issue is elsewhere (decoder capacity, z-space structure) and SLERP pairing won't help.
