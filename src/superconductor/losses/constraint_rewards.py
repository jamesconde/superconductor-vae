"""
SC Constraint Zoo: REINFORCE Reward Modifiers (V12.43 / V13.0)

Physics-grounded generation constraints applied as reward adjustments
during REINFORCE training (SCST/RLOO). All functions operate under
torch.no_grad() since they modify reward signals, not loss gradients.

Constraints implemented:
  A1 — Duplicate Element Penalty
  A2 — GCD Fraction Canonicality Penalty
  A4 — Stoichiometric Normalization Penalty
  A7 — Impossible Element Combination Penalty
  B1-B8 — Family-specific physics constraints

V13.0: Token layout is configurable via VocabConfig to support both
V12 (148 tokens, character-level fractions) and V13 (4355 tokens,
semantic fraction tokens).

Reference: docs/SC_CONSTRAINT_ZOO.md
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch


@dataclass
class VocabConfig:
    """Token vocabulary layout configuration.

    Supports both V12 (character-level) and V13 (semantic fraction) tokenizers.
    """
    element_start: int = 20      # First element token index (H)
    element_end: int = 137       # Last element token index (Og), inclusive
    digit_start: int = 138       # First digit/integer token index
    digit_end: int = 147         # Last digit/integer token index, inclusive
    lparen_idx: int = 4          # '(' token index (V12 only, -1 if absent)
    rparen_idx: int = 5          # ')' token index (V12 only, -1 if absent)
    slash_idx: int = 16          # '/' token index (V12 only, -1 if absent)
    pad_idx: int = 0
    end_idx: int = 2
    use_semantic_fractions: bool = False  # V13.0: fractions are single tokens
    fraction_token_start: int = 0        # V13.0: first FRAC: token index
    fraction_values: Optional[torch.Tensor] = None  # V13.0: [vocab_size] float values

    def elem_idx(self, z: int) -> int:
        """Atomic number Z -> token index."""
        # V12: element_start=20, Z=1(H) -> 20, so offset = element_start - 1 = 19
        # V13: element_start=5, Z=1(H) -> 5, so offset = element_start - 1 = 4
        return self.element_start - 1 + z


# Default V12 config (backward compatibility)
_V12_VOCAB = VocabConfig()

# V13 config is created dynamically via make_v13_vocab_config()
def make_v13_vocab_config(fraction_token_start: int = 143,
                          fraction_values: Optional[torch.Tensor] = None) -> VocabConfig:
    """Create V13.0 vocabulary config."""
    return VocabConfig(
        element_start=5,
        element_end=122,
        digit_start=123,      # Integer tokens (1-20)
        digit_end=142,
        lparen_idx=-1,         # No parens in V13
        rparen_idx=-1,
        slash_idx=-1,
        use_semantic_fractions=True,
        fraction_token_start=fraction_token_start,
        fraction_values=fraction_values,
    )


# Active vocab config (set by set_vocab_config)
_active_vocab: VocabConfig = _V12_VOCAB


def set_vocab_config(config: VocabConfig):
    """Set the active vocabulary config for constraint reward functions."""
    global _active_vocab
    _active_vocab = config
    _rebuild_element_constants()


def _rebuild_element_constants():
    """Rebuild element index constants from active vocab config."""
    global _Cu, _O, _Fe, _Ba, _Sr, _Y, _La, _Bi, _Tl, _Hg, _Mg, _B, _F
    global _Ca, _Pb, _As, _Se, _Te, _Nb, _Sn, _V, _Al, _C, _Li, _Na
    global _Si, _Ge
    global _MAGNETIC_3D, _FORBIDDEN_PAIRS

    v = _active_vocab
    _Cu = v.elem_idx(29)
    _O = v.elem_idx(8)
    _Fe = v.elem_idx(26)
    _Ba = v.elem_idx(56)
    _Sr = v.elem_idx(38)
    _Y = v.elem_idx(39)
    _La = v.elem_idx(57)
    _Bi = v.elem_idx(83)
    _Tl = v.elem_idx(81)
    _Hg = v.elem_idx(80)
    _Mg = v.elem_idx(12)
    _B = v.elem_idx(5)
    _F = v.elem_idx(9)
    _Ca = v.elem_idx(20)
    _Pb = v.elem_idx(82)
    _As = v.elem_idx(33)
    _Se = v.elem_idx(34)
    _Te = v.elem_idx(52)
    _Nb = v.elem_idx(41)
    _Sn = v.elem_idx(50)
    _V = v.elem_idx(23)
    _Al = v.elem_idx(13)
    _C = v.elem_idx(6)
    _Li = v.elem_idx(3)
    _Na = v.elem_idx(11)
    _Si = v.elem_idx(14)
    _Ge = v.elem_idx(32)

    _MAGNETIC_3D = frozenset([v.elem_idx(25), v.elem_idx(26),
                              v.elem_idx(27), v.elem_idx(28)])
    _FORBIDDEN_PAIRS = [(_F, _Tl)]


# Initialize with V12 defaults
_rebuild_element_constants()


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


# A7: Forbidden element pair lookup table — initialized by _rebuild_element_constants()


def _extract_elements_and_fractions(
    tokens: torch.Tensor,   # [seq_len]
    mask: torch.Tensor,     # [seq_len]
) -> Tuple[List[int], Dict[int, float]]:
    """Extract element token indices and their fractions from a single token sequence.

    Supports both V12 (character-level) and V13 (semantic fraction) token layouts,
    based on the active VocabConfig set via set_vocab_config().

    Returns:
        elements: list of element token indices found
        fractions: dict mapping element token idx -> fraction value
    """
    v = _active_vocab
    seq_len = tokens.size(0)
    elements = []
    fractions = {}
    i = 0

    while i < seq_len:
        tok = tokens[i].item()
        if not mask[i].item():
            break
        if tok == v.end_idx:
            break

        # Check if this is an element token
        if v.element_start <= tok <= v.element_end:
            elem = tok
            elements.append(elem)
            frac = 1.0  # Default fraction

            # Look ahead for subscript
            j = i + 1
            if j < seq_len and mask[j].item():
                next_tok = tokens[j].item()

                if v.use_semantic_fractions:
                    # V13.0: Check for integer token (123-142) or fraction token (143+)
                    if v.digit_start <= next_tok <= v.digit_end:
                        # Integer subscript: token value = next_tok - digit_start + 1
                        frac = float(next_tok - v.digit_start + 1)
                        j += 1
                    elif (next_tok >= v.fraction_token_start and
                          v.fraction_values is not None):
                        # Semantic fraction token: look up float value
                        if next_tok < v.fraction_values.shape[0]:
                            frac = v.fraction_values[next_tok].item()
                        j += 1
                else:
                    # V12: Character-level parsing
                    if next_tok == v.lparen_idx:
                        # Parse (numerator/denominator)
                        j += 1
                        num_digits = []
                        den_digits = []
                        parsing_num = True
                        while j < seq_len:
                            t = tokens[j].item()
                            if t == v.slash_idx:
                                parsing_num = False
                            elif t == v.rparen_idx:
                                j += 1
                                break
                            elif v.digit_start <= t <= v.digit_end:
                                digit = t - v.digit_start
                                if parsing_num:
                                    num_digits.append(digit)
                                else:
                                    den_digits.append(digit)
                            else:
                                break
                            j += 1

                        if num_digits and den_digits:
                            num = int(''.join(str(d) for d in num_digits))
                            den = int(''.join(str(d) for d in den_digits))
                            if den > 0:
                                frac = num / den
                    elif v.digit_start <= next_tok <= v.digit_end:
                        # Plain integer subscript (e.g., O4, Cu3)
                        digits = []
                        while j < seq_len and v.digit_start <= tokens[j].item() <= v.digit_end:
                            digits.append(tokens[j].item() - v.digit_start)
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
    v = _active_vocab

    # Mask to only element tokens
    is_element = (sampled_tokens >= v.element_start) & (sampled_tokens <= v.element_end) & mask.bool()

    # Map element tokens to range [0, 117] for scatter
    elem_indices = (sampled_tokens - v.element_start).clamp(min=0)
    elem_indices = elem_indices * is_element.long()  # Zero out non-element positions

    # Count occurrences per element per batch using scatter_add
    n_elements = v.element_end - v.element_start + 1  # 118
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

    V13.0: Fractions are GCD-canonical by construction (tokenizer auto-reduces).
    Returns zero penalty for V13.

    V12: Scans for fraction patterns (num/den) in token sequences and checks
    if each fraction is in lowest terms.
    """
    batch_size = sampled_tokens.shape[0]
    device = sampled_tokens.device
    penalties = torch.zeros(batch_size, device=device)
    v = _active_vocab

    # V13.0: Semantic fractions are always canonical — skip entirely
    if v.use_semantic_fractions:
        return penalties

    # V12: Process on CPU for GCD computation (small per-sample cost)
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
            if seq[i].item() == v.lparen_idx:
                # Try to parse (num/den)
                j = i + 1
                num_digits = []
                den_digits = []
                parsing_num = True
                valid = False
                while j < seq_len and m[j].item():
                    t = seq[j].item()
                    if t == v.slash_idx:
                        parsing_num = False
                    elif t == v.rparen_idx:
                        valid = True
                        j += 1
                        break
                    elif v.digit_start <= t <= v.digit_end:
                        if parsing_num:
                            num_digits.append(t - v.digit_start)
                        else:
                            den_digits.append(t - v.digit_start)
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
    v = _active_vocab

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
            if t == v.end_idx:
                break

            # V13.0: Check for fraction token (single token = has fractions)
            if v.use_semantic_fractions and t >= v.fraction_token_start:
                has_fractions = True
                break
            # V12: Check for '(' = has fractions
            if not v.use_semantic_fractions and t == v.lparen_idx:
                has_fractions = True
                break

            if v.element_start <= t <= v.element_end:
                # Check for following integer subscript
                j = i + 1
                if v.use_semantic_fractions:
                    # V13: Integer tokens at digit_start..digit_end represent values 1-20
                    if j < seq_len and m[j].item() and v.digit_start <= seq[j].item() <= v.digit_end:
                        subscripts.append(seq[j].item() - v.digit_start + 1)
                        j += 1
                    else:
                        subscripts.append(1)
                else:
                    # V12: Multi-digit integer subscripts
                    digits = []
                    while j < seq_len and m[j].item() and v.digit_start <= seq[j].item() <= v.digit_end:
                        digits.append(seq[j].item() - v.digit_start)
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
            b_site = {_Sn, _Al, _Si, _Ge}  # Si=14, Ge=32
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
