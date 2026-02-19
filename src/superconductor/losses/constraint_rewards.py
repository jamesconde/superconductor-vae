"""
SC Constraint Zoo: REINFORCE Reward Modifiers (V12.43)

Physics-grounded generation constraints applied as reward adjustments
during REINFORCE training (SCST/RLOO). All functions operate under
torch.no_grad() since they modify reward signals, not loss gradients.

Constraints implemented:
  A1 — Duplicate Element Penalty
  A2 — GCD Fraction Canonicality Penalty
  A4 — Stoichiometric Normalization Penalty
  A7 — Impossible Element Combination Penalty
  B1-B8 — Family-specific physics constraints

Reference: docs/SC_CONSTRAINT_ZOO.md
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

# Token index constants (from autoregressive_decoder.py VOCAB layout)
# SPECIAL(20) + ELEMENTS[1:](118) + DIGITS(10) = 148 tokens
_ELEMENT_START = 20   # H
_ELEMENT_END = 137    # Og (inclusive)
_DIGIT_START = 138    # '0'
_DIGIT_END = 147      # '9' (inclusive)
_LPAREN_IDX = 4       # '('
_RPAREN_IDX = 5       # ')'
_SLASH_IDX = 16       # '/'
_PAD_IDX = 0
_END_IDX = 2

# Element token indices for specific elements (SPECIAL_offset=20, then 1-based element list)
# Element Z=N maps to token index 20 + N - 1 = 19 + N
_elem_idx = lambda z: 19 + z  # atomic number Z -> token index

# Key element token indices
_Cu = _elem_idx(29)   # 48
_O = _elem_idx(8)     # 27
_Fe = _elem_idx(26)   # 45
_Ba = _elem_idx(56)   # 75
_Sr = _elem_idx(38)   # 57
_Y = _elem_idx(39)    # 58
_La = _elem_idx(57)   # 76
_Bi = _elem_idx(83)   # 102
_Tl = _elem_idx(81)   # 100
_Hg = _elem_idx(80)   # 99
_Mg = _elem_idx(12)   # 31
_B = _elem_idx(5)     # 24
_F = _elem_idx(9)     # 28
_Ca = _elem_idx(20)   # 39
_Pb = _elem_idx(82)   # 101
_As = _elem_idx(33)   # 52
_Se = _elem_idx(34)   # 53
_Te = _elem_idx(52)   # 71
_Nb = _elem_idx(41)   # 60
_Sn = _elem_idx(50)   # 69
_V = _elem_idx(23)    # 42
_Al = _elem_idx(13)   # 32
_C = _elem_idx(6)     # 25
_Li = _elem_idx(3)    # 22
_Na = _elem_idx(11)   # 30

# Magnetic 3d transition metals (Mn, Fe, Co, Ni — Z=25,26,27,28)
_MAGNETIC_3D = frozenset([_elem_idx(25), _elem_idx(26), _elem_idx(27), _elem_idx(28)])


@dataclass
class ConstraintRewardConfig:
    """Configuration for constraint reward penalties."""
    # A1: Duplicate element penalty
    a1_enabled: bool = True
    a1_penalty: float = -50.0

    # A2: GCD fraction canonicality
    a2_enabled: bool = True
    a2_penalty_per_violation: float = -5.0

    # A4: Stoichiometric normalization
    a4_enabled: bool = True
    a4_penalty: float = -10.0

    # A7: Impossible element combinations
    a7_enabled: bool = True
    a7_penalty: float = -30.0


@dataclass
class FamilyConstraintConfig:
    """Configuration for family-specific (B1-B8) constraint rewards."""
    enabled: bool = True
    confidence_threshold: float = 0.8  # Only apply when family classifier > this

    # Per-constraint penalties
    b1_penalty: float = -40.0   # YBCO oxygen content
    b2_penalty: float = -40.0   # LSCO Sr doping range
    b3_penalty: float = -40.0   # BSCCO Ca-Cu balance
    b4_penalty: float = -30.0   # Hg-cuprate volatile substitution
    b5_penalty: float = -30.0   # Tl-cuprate poison elements
    b6_penalty: float = -30.0   # Iron-1111 oxygen content
    b7_penalty: float = -30.0   # MgB2 poison elements
    b8_penalty: float = -30.0   # A15 stoichiometry ratio


# A7: Forbidden element pair lookup table
# (element_idx_a, element_idx_b) pairs that should not co-occur
_FORBIDDEN_PAIRS: List[Tuple[int, int]] = [
    (_F, _Tl),   # F + Tl: fluorine destroys Tl-cuprate structure
]


def _extract_elements_and_fractions(
    tokens: torch.Tensor,   # [seq_len]
    mask: torch.Tensor,     # [seq_len]
) -> Tuple[List[int], Dict[int, float]]:
    """Extract element token indices and their fractions from a single token sequence.

    Returns:
        elements: list of element token indices found
        fractions: dict mapping element token idx -> fraction value
    """
    seq_len = tokens.size(0)
    elements = []
    fractions = {}
    i = 0

    while i < seq_len:
        tok = tokens[i].item()
        if not mask[i].item():
            break
        if tok == _END_IDX:
            break

        # Check if this is an element token
        if _ELEMENT_START <= tok <= _ELEMENT_END:
            elem = tok
            elements.append(elem)
            frac = 1.0  # Default fraction

            # Look ahead for fraction: (num/den) or plain digits
            j = i + 1
            if j < seq_len and tokens[j].item() == _LPAREN_IDX:
                # Parse (numerator/denominator)
                j += 1
                num_digits = []
                den_digits = []
                parsing_num = True
                while j < seq_len:
                    t = tokens[j].item()
                    if t == _SLASH_IDX:
                        parsing_num = False
                    elif t == _RPAREN_IDX:
                        j += 1
                        break
                    elif _DIGIT_START <= t <= _DIGIT_END:
                        digit = t - _DIGIT_START
                        if parsing_num:
                            num_digits.append(digit)
                        else:
                            den_digits.append(digit)
                    else:
                        break  # Malformed fraction
                    j += 1

                if num_digits and den_digits:
                    num = int(''.join(str(d) for d in num_digits))
                    den = int(''.join(str(d) for d in den_digits))
                    if den > 0:
                        frac = num / den
                i = j
            elif j < seq_len and _DIGIT_START <= tokens[j].item() <= _DIGIT_END:
                # Plain integer subscript (e.g., O4, Cu3)
                digits = []
                while j < seq_len and _DIGIT_START <= tokens[j].item() <= _DIGIT_END:
                    digits.append(tokens[j].item() - _DIGIT_START)
                    j += 1
                frac = float(int(''.join(str(d) for d in digits)))
                i = j
            else:
                i += 1

            fractions[elem] = frac
        else:
            i += 1

    return elements, fractions


@torch.no_grad()
def compute_duplicate_element_penalty(
    sampled_tokens: torch.Tensor,  # [batch, seq_len]
    mask: torch.Tensor,            # [batch, seq_len]
    penalty: float = -50.0,
) -> torch.Tensor:  # [batch]
    """A1: Penalize formulas where any element appears more than once.

    Uses scatter_add on a per-batch element accumulator to detect duplicates
    efficiently on GPU.
    """
    batch_size, seq_len = sampled_tokens.shape
    device = sampled_tokens.device

    # Mask to only element tokens
    is_element = (sampled_tokens >= _ELEMENT_START) & (sampled_tokens <= _ELEMENT_END) & mask.bool()

    # Map element tokens to range [0, 117] for scatter
    elem_indices = (sampled_tokens - _ELEMENT_START).clamp(min=0)
    elem_indices = elem_indices * is_element.long()  # Zero out non-element positions

    # Count occurrences per element per batch using scatter_add
    n_elements = _ELEMENT_END - _ELEMENT_START + 1  # 118
    counts = torch.zeros(batch_size, n_elements, device=device, dtype=torch.float32)
    ones = is_element.float()

    # scatter_add: accumulate 1.0 at each element index
    counts.scatter_add_(1, elem_indices * is_element.long(), ones)

    # Any element with count > 1 is a duplicate
    has_duplicate = (counts > 1.0).any(dim=1).float()  # [batch]

    return has_duplicate * penalty


@torch.no_grad()
def compute_gcd_canonicality_penalty(
    sampled_tokens: torch.Tensor,  # [batch, seq_len]
    mask: torch.Tensor,            # [batch, seq_len]
    penalty_per_violation: float = -5.0,
) -> torch.Tensor:  # [batch]
    """A2: Penalize non-canonical fractions where GCD(num, den) > 1.

    Scans for fraction patterns (num/den) in token sequences and checks
    if each fraction is in lowest terms.
    """
    batch_size = sampled_tokens.shape[0]
    device = sampled_tokens.device
    penalties = torch.zeros(batch_size, device=device)

    # Process on CPU for GCD computation (small per-sample cost)
    tokens_cpu = sampled_tokens.cpu()
    mask_cpu = mask.cpu()

    for b in range(batch_size):
        violations = 0
        seq = tokens_cpu[b]
        m = mask_cpu[b]
        i = 0
        seq_len = seq.size(0)

        while i < seq_len:
            if not m[i].item():
                break
            if seq[i].item() == _LPAREN_IDX:
                # Try to parse (num/den)
                j = i + 1
                num_digits = []
                den_digits = []
                parsing_num = True
                valid = False
                while j < seq_len and m[j].item():
                    t = seq[j].item()
                    if t == _SLASH_IDX:
                        parsing_num = False
                    elif t == _RPAREN_IDX:
                        valid = True
                        j += 1
                        break
                    elif _DIGIT_START <= t <= _DIGIT_END:
                        if parsing_num:
                            num_digits.append(t - _DIGIT_START)
                        else:
                            den_digits.append(t - _DIGIT_START)
                    else:
                        break
                    j += 1

                if valid and num_digits and den_digits:
                    num = int(''.join(str(d) for d in num_digits))
                    den = int(''.join(str(d) for d in den_digits))
                    if den > 0 and math.gcd(num, den) > 1:
                        violations += 1
                i = j if j > i + 1 else i + 1
            else:
                i += 1

        if violations > 0:
            penalties[b] = violations * penalty_per_violation

    return penalties.to(device)


@torch.no_grad()
def compute_stoich_normalization_penalty(
    sampled_tokens: torch.Tensor,  # [batch, seq_len]
    mask: torch.Tensor,            # [batch, seq_len]
    penalty: float = -10.0,
) -> torch.Tensor:  # [batch]
    """A4: Penalize reducible stoichiometry (e.g., Mg2B4 instead of MgB2).

    Parses all integer subscripts in a formula and checks if their GCD > 1.
    Fraction-based formulas are not penalized (they handle normalization differently).
    """
    batch_size = sampled_tokens.shape[0]
    device = sampled_tokens.device
    penalties = torch.zeros(batch_size, device=device)

    tokens_cpu = sampled_tokens.cpu()
    mask_cpu = mask.cpu()

    for b in range(batch_size):
        seq = tokens_cpu[b]
        m = mask_cpu[b]
        subscripts = []
        has_fractions = False
        i = 0
        seq_len = seq.size(0)

        while i < seq_len:
            if not m[i].item():
                break
            t = seq[i].item()
            if t == _END_IDX:
                break

            if t == _LPAREN_IDX:
                has_fractions = True
                break  # Fraction formula — skip A4

            if _ELEMENT_START <= t <= _ELEMENT_END:
                # Check for following integer subscript
                j = i + 1
                digits = []
                while j < seq_len and m[j].item() and _DIGIT_START <= seq[j].item() <= _DIGIT_END:
                    digits.append(seq[j].item() - _DIGIT_START)
                    j += 1
                if digits:
                    subscripts.append(int(''.join(str(d) for d in digits)))
                else:
                    subscripts.append(1)
                i = j
            else:
                i += 1

        if has_fractions or len(subscripts) < 2:
            continue

        # Check if GCD of all subscripts > 1
        overall_gcd = subscripts[0]
        for s in subscripts[1:]:
            overall_gcd = math.gcd(overall_gcd, s)
        if overall_gcd > 1:
            penalties[b] = penalty

    return penalties.to(device)


@torch.no_grad()
def compute_impossible_element_penalty(
    sampled_tokens: torch.Tensor,  # [batch, seq_len]
    mask: torch.Tensor,            # [batch, seq_len]
    penalty: float = -30.0,
) -> torch.Tensor:  # [batch]
    """A7: Penalize physically impossible element combinations.

    Checks forbidden element pairs and magnetic 3d metals co-occurring with Cu
    at high fractions on shared sites.
    """
    batch_size = sampled_tokens.shape[0]
    device = sampled_tokens.device
    penalties = torch.zeros(batch_size, device=device)

    tokens_cpu = sampled_tokens.cpu()
    mask_cpu = mask.cpu()

    for b in range(batch_size):
        elements, fractions = _extract_elements_and_fractions(tokens_cpu[b], mask_cpu[b])
        elem_set = set(elements)

        violated = False
        # Check forbidden pairs
        for a, b_elem in _FORBIDDEN_PAIRS:
            if a in elem_set and b_elem in elem_set:
                violated = True
                break

        # Check magnetic 3d metals (>2% fraction) co-occurring with Cu on shared sites
        if not violated and _Cu in elem_set:
            cu_frac = fractions.get(_Cu, 0.0)
            if cu_frac > 0:
                for mag_elem in _MAGNETIC_3D:
                    if mag_elem in elem_set:
                        mag_frac = fractions.get(mag_elem, 0.0)
                        # Only penalize if magnetic element is a significant fraction (>2%)
                        # AND appears to be on Cu sites (fraction comparable to Cu)
                        if mag_frac > 0.02 and mag_frac > 0.5 * cu_frac:
                            violated = True
                            break

        if violated:
            penalties[b] = penalty

    return penalties.to(device)


@torch.no_grad()
def compute_family_constraint_rewards(
    sampled_tokens: torch.Tensor,      # [batch, seq_len]
    mask: torch.Tensor,                # [batch, seq_len]
    family_predictions: torch.Tensor,  # [batch, 14] composed family probabilities
    config: FamilyConstraintConfig,
) -> torch.Tensor:  # [batch]
    """B1-B8: Family-specific physics constraint rewards.

    Only applies constraints when the family classifier is confident (>threshold).
    Family indices (from family_classifier.py SuperconductorFamily):
      0=NOT_SC, 1=BCS, 2=YBCO, 3=LSCO, 4=BSCCO, 5=TBCCO, 6=HBCCO,
      7=CUPRATE_OTHER, 8=IRON_PNICTIDE, 9=IRON_CHALCOGENIDE, 10=MGB2,
      11=HEAVY_FERMION, 12=ORGANIC, 13=OTHER
    """
    batch_size = sampled_tokens.shape[0]
    device = sampled_tokens.device
    penalties = torch.zeros(batch_size, device=device)

    if not config.enabled:
        return penalties

    # Get family predictions and confidence
    family_probs = family_predictions.detach()
    family_confidence, family_ids = family_probs.max(dim=1)  # [batch], [batch]

    tokens_cpu = sampled_tokens.cpu()
    mask_cpu = mask.cpu()
    family_ids_cpu = family_ids.cpu()
    family_conf_cpu = family_confidence.cpu()

    for b in range(batch_size):
        # Skip if classifier not confident
        if family_conf_cpu[b].item() < config.confidence_threshold:
            continue

        fam = family_ids_cpu[b].item()
        elements, fractions = _extract_elements_and_fractions(tokens_cpu[b], mask_cpu[b])
        elem_set = set(elements)

        penalty = 0.0

        # B1: YBCO — O content should be ~6.5-7.0 (delta < 0.65)
        if fam == 2:  # CUPRATE_YBCO
            o_frac = fractions.get(_O, 0.0)
            if o_frac > 0 and o_frac < 6.35:
                penalty += config.b1_penalty

        # B2: LSCO — Sr doping range 0.055-0.27 for superconductivity
        elif fam == 3:  # CUPRATE_LSCO
            sr_frac = fractions.get(_Sr, 0.0)
            if _Sr in elem_set and (sr_frac < 0.055 or sr_frac > 0.27):
                penalty += config.b2_penalty

        # B3: BSCCO — |Ca - (Cu - 1)| should be < 0.3
        elif fam == 4:  # CUPRATE_BSCCO
            ca_frac = fractions.get(_Ca, 0.0)
            cu_frac = fractions.get(_Cu, 0.0)
            if _Ca in elem_set and _Cu in elem_set:
                if abs(ca_frac - (cu_frac - 1)) > 0.3:
                    penalty += config.b3_penalty

        # B4: Hg-cuprate — volatile elements (V) < 30% on Hg site
        elif fam == 6:  # CUPRATE_HBCCO
            v_frac = fractions.get(_V, 0.0)
            if v_frac > 0.30:
                penalty += config.b4_penalty

        # B5: Tl-cuprate — no magnetic 3d > 10%, V < 30%, Li < 10%
        elif fam == 5:  # CUPRATE_TBCCO
            v_frac = fractions.get(_V, 0.0)
            li_frac = fractions.get(_Li, 0.0)
            if v_frac > 0.30:
                penalty += config.b5_penalty
            if li_frac > 0.10:
                penalty += config.b5_penalty
            for mag in _MAGNETIC_3D:
                if mag in elem_set and fractions.get(mag, 0.0) > 0.10:
                    penalty += config.b5_penalty
                    break

        # B6: Iron-1111 — O should be 1.0 (undoped) or > 0.7
        elif fam == 8:  # IRON_PNICTIDE
            o_frac = fractions.get(_O, 0.0)
            if _O in elem_set and o_frac < 0.7 and o_frac != 1.0:
                penalty += config.b6_penalty

        # B7: MgB2 — C < 12.5%, Al < 50%, magnetic 3d < 5%
        elif fam == 10:  # MGB2_TYPE
            c_frac = fractions.get(_C, 0.0)
            al_frac = fractions.get(_Al, 0.0)
            if c_frac > 0.125:
                penalty += config.b7_penalty
            if al_frac > 0.50:
                penalty += config.b7_penalty
            for mag in _MAGNETIC_3D:
                if mag in elem_set and fractions.get(mag, 0.0) > 0.05:
                    penalty += config.b7_penalty
                    break

        # B8: A15 (BCS_CONVENTIONAL with Nb/V + Sn/Al/Ge pattern) — A:B ratio within 10% of 3:1
        elif fam == 1:  # BCS_CONVENTIONAL
            # A15 detection: Nb3Sn, V3Si, Nb3Ge, Nb3Al patterns
            a_site = {_Nb, _V}
            b_site = {_Sn, _Al, _elem_idx(14), _elem_idx(32)}  # Si=14, Ge=32
            a_total = sum(fractions.get(e, 0.0) for e in a_site if e in elem_set)
            b_total = sum(fractions.get(e, 0.0) for e in b_site if e in elem_set)
            if a_total > 0 and b_total > 0:
                ratio = a_total / b_total
                # A15 should be 3:1 ± 10%
                if abs(ratio - 3.0) > 0.3:
                    penalty += config.b8_penalty

        if penalty < 0:
            penalties[b] = penalty

    return penalties.to(device)


@torch.no_grad()
def compute_constraint_rewards(
    sampled_tokens: torch.Tensor,                     # [batch, seq_len]
    mask: torch.Tensor,                               # [batch, seq_len]
    config: ConstraintRewardConfig,
    family_predictions: Optional[torch.Tensor] = None, # [batch, 14]
    family_config: Optional[FamilyConstraintConfig] = None,
) -> torch.Tensor:  # [batch] total constraint reward adjustments
    """Aggregate all constraint reward penalties.

    Returns per-sample reward adjustments (negative = penalizes bad generations).
    """
    device = sampled_tokens.device
    batch_size = sampled_tokens.shape[0]
    total = torch.zeros(batch_size, device=device)

    # A1: Duplicate elements
    if config.a1_enabled:
        total = total + compute_duplicate_element_penalty(
            sampled_tokens, mask, config.a1_penalty
        )

    # A2: GCD canonicality
    if config.a2_enabled:
        total = total + compute_gcd_canonicality_penalty(
            sampled_tokens, mask, config.a2_penalty_per_violation
        )

    # A4: Stoichiometric normalization
    if config.a4_enabled:
        total = total + compute_stoich_normalization_penalty(
            sampled_tokens, mask, config.a4_penalty
        )

    # A7: Impossible element combinations
    if config.a7_enabled:
        total = total + compute_impossible_element_penalty(
            sampled_tokens, mask, config.a7_penalty
        )

    # B1-B8: Family-specific constraints
    if (family_predictions is not None and family_config is not None
            and family_config.enabled):
        total = total + compute_family_constraint_rewards(
            sampled_tokens, mask, family_predictions, family_config
        )

    return total
