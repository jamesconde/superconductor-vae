"""
SC Constraint Zoo: Differentiable Physics Constraints (V12.43)

Differentiable losses for physics constraints that can provide gradients
to guide the model during training:

  A3 — Site Occupancy Sum: Ensures shared crystallographic sites sum correctly
  A6 — Charge Balance: Soft penalty for charge-imbalanced formulas

These losses operate on the element_indices / element_fractions tensors
output by the model, so gradients flow through the stoichiometry predictions.

Reference: docs/SC_CONSTRAINT_ZOO.md (constraints A3, A6)
"""

from typing import Dict, Optional, Set

import torch
import torch.nn as nn


class SiteOccupancySumLoss(nn.Module):
    """A3: Site occupancy sum constraint.

    In many superconductor families, elements share crystallographic sites
    and their fractional occupancies must sum to a specific value (usually 1.0).
    For example, in YBCO: Y and rare-earth dopants share the Y-site and must
    sum to 1.0.

    Uses the family classifier output to determine which site definitions apply,
    then computes a soft L1 loss on deviations from the target sum.
    """

    # Site definitions per family
    # Format: family_id -> list of (site_name, element_Z_set, target_sum)
    # Family IDs from SuperconductorFamily enum:
    #   2=YBCO, 3=LSCO, 4=BSCCO, 5=TBCCO, 6=HBCCO, 8=IRON_PNICTIDE, 10=MGB2
    SITE_DEFINITIONS: Dict[int, list] = {
        # YBCO: RE site (Y, Eu, Nd, Sm, Gd, Dy, Ho, Er, Tm, Yb, Lu, Pr, La) sums to 1.0
        2: [
            ('re_site', {39, 63, 60, 62, 64, 66, 67, 68, 69, 70, 71, 59, 57}, 1.0),
            # Ba site (Ba, Sr, Ca) sums to 2.0
            ('ba_site', {56, 38, 20}, 2.0),
        ],
        # LSCO: La/Sr site sums to 2.0 (La2-xSrxCuO4)
        3: [
            ('la_site', {57, 38, 20, 56}, 2.0),  # La, Sr, Ca, Ba can substitute
        ],
        # BSCCO: Bi/Pb site sums to 2.0 (Bi2-xPbx...)
        4: [
            ('bi_site', {83, 82}, 2.0),  # Bi, Pb
        ],
        # TBCCO: Tl site sums to 2.0 (Tl2Ba2Ca...)
        5: [
            ('tl_site', {81, 82}, 2.0),  # Tl, Pb substitute
        ],
        # HBCCO: Hg site sums to 1.0 (Hg1-xTlx...)
        6: [
            ('hg_site', {80, 81}, 1.0),  # Hg, Tl substitute
        ],
        # Iron-1111: RE site sums to 1.0 (LaFeAsO)
        8: [
            ('re_site', {57, 60, 62, 58, 20, 56}, 1.0),  # La, Nd, Sm, Ce, Ca, Ba
        ],
        # MgB2: Mg site sums to 1.0
        10: [
            ('mg_site', {12, 3, 11, 13, 20}, 1.0),  # Mg, Li, Na, Al, Ca
        ],
    }

    def __init__(self, weight: float = 1.0, confidence_threshold: float = 0.8):
        super().__init__()
        self.weight = weight
        self.confidence_threshold = confidence_threshold

        # Pre-build site membership tensors for vectorized lookup.
        # For each (family_id, site_idx), we store which atomic numbers belong to that site.
        # Max atomic number in any site definition (Lu=71).
        self._max_z = 120
        self._site_data = {}  # (family_id, site_idx) -> (z_membership_set, target_sum)
        for fam_id, sites in self.SITE_DEFINITIONS.items():
            for site_idx, (site_name, site_z_set, target_sum) in enumerate(sites):
                self._site_data[(fam_id, site_idx)] = (site_z_set, target_sum)

    def forward(
        self,
        element_indices: torch.Tensor,         # [batch, max_elements] atomic numbers
        element_fractions: torch.Tensor,       # [batch, max_elements] molar fractions
        element_mask: torch.Tensor,            # [batch, max_elements] valid mask
        family_predictions: Optional[torch.Tensor] = None,  # [batch, 14] composed probs
    ) -> Dict[str, torch.Tensor]:
        """Compute site occupancy sum loss.

        Uses vectorized tensor operations to maintain gradient flow through
        element_fractions. Family selection and confidence gating use detached
        tensors (non-differentiable decisions), but the fraction summation
        and deviation computation are fully differentiable.

        Returns:
            'site_occupancy_loss': scalar loss
            'n_constrained': number of samples with applicable constraints
        """
        device = element_indices.device
        batch_size = element_indices.shape[0]

        if family_predictions is None:
            return {
                'site_occupancy_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'n_constrained': 0,
            }

        # Get family assignments and confidence (detached — routing decisions, not differentiable)
        family_probs = family_predictions.detach()
        family_confidence, family_ids = family_probs.max(dim=1)  # [batch], [batch]

        # Collect per-sample deviations as a list of differentiable tensors
        deviations = []

        for fam_id, sites in self.SITE_DEFINITIONS.items():
            # Which samples belong to this family with sufficient confidence?
            fam_mask = (family_ids == fam_id) & (family_confidence >= self.confidence_threshold)
            if not fam_mask.any():
                continue

            # Extract just the samples for this family
            fam_elem_idx = element_indices[fam_mask]     # [n_fam, max_elements]
            fam_elem_frac = element_fractions[fam_mask]  # [n_fam, max_elements] — HAS GRADIENTS
            fam_elem_mask = element_mask[fam_mask]       # [n_fam, max_elements]

            for site_name, site_z_set, target_sum in sites:
                # Build a boolean mask: which positions have elements in this site?
                # element_indices contains atomic numbers; check membership in site_z_set
                site_member = torch.zeros_like(fam_elem_idx, dtype=torch.bool)
                for z in site_z_set:
                    site_member = site_member | (fam_elem_idx == z)

                # Combined mask: valid element AND in site
                combined_mask = site_member & fam_elem_mask.bool()  # [n_fam, max_elements]

                # Check which samples have at least one site element
                has_site = combined_mask.any(dim=1)  # [n_fam]
                if not has_site.any():
                    continue

                # Sum fractions on this site (differentiable through fam_elem_frac)
                site_sums = (fam_elem_frac * combined_mask.float()).sum(dim=1)  # [n_fam]

                # Deviation from target (only for samples that have site elements)
                dev = torch.abs(site_sums[has_site] - target_sum)  # [n_valid]
                deviations.append(dev)

        if len(deviations) == 0:
            return {
                'site_occupancy_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'n_constrained': 0,
            }

        all_deviations = torch.cat(deviations)  # [total_constrained]
        n_constrained = all_deviations.shape[0]
        mean_loss = all_deviations.mean()

        return {
            'site_occupancy_loss': self.weight * mean_loss,
            'n_constrained': n_constrained,
        }


# Common oxidation states for superconductor-relevant elements
# Uses most common/stable oxidation state for charge balance estimation
_COMMON_OXIDATION: Dict[int, float] = {
    # Alkaline earths & alkali
    3: 1.0,    # Li
    11: 1.0,   # Na
    19: 1.0,   # K
    37: 1.0,   # Rb
    55: 1.0,   # Cs
    4: 2.0,    # Be
    12: 2.0,   # Mg
    20: 2.0,   # Ca
    38: 2.0,   # Sr
    56: 2.0,   # Ba
    # Transition metals (common states in SC compounds)
    39: 3.0,   # Y
    57: 3.0,   # La
    58: 3.0,   # Ce (often +3 in SC)
    59: 3.0,   # Pr
    60: 3.0,   # Nd
    62: 3.0,   # Sm
    63: 2.0,   # Eu (often +2 in cuprates)
    64: 3.0,   # Gd
    66: 3.0,   # Dy
    67: 3.0,   # Ho
    68: 3.0,   # Er
    69: 3.0,   # Tm
    70: 2.0,   # Yb
    71: 3.0,   # Lu
    29: 2.0,   # Cu (+2 average in cuprates, mixed valence)
    26: 2.0,   # Fe (+2 in iron-based SC)
    27: 2.0,   # Co
    28: 2.0,   # Ni
    25: 2.0,   # Mn
    # Post-transition metals
    13: 3.0,   # Al
    31: 3.0,   # Ga
    49: 3.0,   # In
    81: 3.0,   # Tl (+3 in Tl-cuprates, can be +1)
    82: 2.0,   # Pb
    83: 3.0,   # Bi
    50: 4.0,   # Sn
    # Non-metals
    5: 3.0,    # B (in MgB2, effective oxidation ~-1, but simplified)
    6: 4.0,    # C
    7: -3.0,   # N
    8: -2.0,   # O
    9: -1.0,   # F
    15: -3.0,  # P
    16: -2.0,  # S
    17: -1.0,  # Cl
    33: -3.0,  # As
    34: -2.0,  # Se
    52: -2.0,  # Te
    # Pnictides
    80: 2.0,   # Hg
    41: 5.0,   # Nb (in A15 compounds)
    23: 5.0,   # V (in A15 compounds)
    22: 4.0,   # Ti
    40: 4.0,   # Zr
    72: 4.0,   # Hf
    42: 6.0,   # Mo
    74: 6.0,   # W
}


class ChargeBalanceLoss(nn.Module):
    """A6: Charge balance constraint.

    Computes a soft penalty for charge imbalance in generated formulas.
    Uses tanh to create a bounded loss that tolerates slight imbalance
    (common in cuprates with mixed Cu+2/Cu+3 valence).

    loss = tanh(|sum(fraction_i * oxidation_state_i)|)

    The loss is differentiable through element_fractions, providing
    gradients to the stoichiometry prediction heads.
    """

    def __init__(self, weight: float = 1.0, tolerance: float = 0.5):
        """
        Args:
            weight: Loss scaling factor
            tolerance: Charge imbalance below this is considered acceptable
                      (cuprates typically have 0.1-0.5 imbalance due to mixed valence)
        """
        super().__init__()
        self.weight = weight
        self.tolerance = tolerance

        # Register oxidation state lookup as buffer for GPU transfer
        # Size 120 covers all elements up to Oganesson (Z=118) with padding
        max_z = 119
        ox_states = torch.zeros(max_z + 1)
        for z, ox in _COMMON_OXIDATION.items():
            ox_states[z] = ox
        self.register_buffer('oxidation_states', ox_states)

    def forward(
        self,
        element_indices: torch.Tensor,     # [batch, max_elements] atomic numbers
        element_fractions: torch.Tensor,   # [batch, max_elements] molar fractions
        element_mask: torch.Tensor,        # [batch, max_elements] valid mask
    ) -> Dict[str, torch.Tensor]:
        """Compute charge balance loss.

        Returns:
            'charge_balance_loss': scalar loss
            'mean_charge_imbalance': scalar mean absolute charge imbalance (for logging)
        """
        device = element_indices.device

        # Look up oxidation states for each element in batch
        # Clamp indices to valid range for the lookup table
        safe_indices = element_indices.clamp(0, self.oxidation_states.shape[0] - 1)
        ox = self.oxidation_states[safe_indices]  # [batch, max_elements]

        # Compute charge sum: sum(fraction * oxidation_state) for valid elements
        charge_per_element = element_fractions * ox * element_mask.float()  # [batch, max_elements]
        total_charge = charge_per_element.sum(dim=1)  # [batch]

        # Absolute charge imbalance
        abs_charge = torch.abs(total_charge)

        # Soft penalty with tolerance: only penalize above tolerance
        excess = torch.clamp(abs_charge - self.tolerance, min=0.0)
        loss_per_sample = torch.tanh(excess)  # Bounded [0, 1)

        loss = loss_per_sample.mean()

        return {
            'charge_balance_loss': self.weight * loss,
            'mean_charge_imbalance': abs_charge.mean().detach(),
        }
