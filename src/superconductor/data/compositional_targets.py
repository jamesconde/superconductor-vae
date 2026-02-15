"""
Compositional target computer for Physics Z Block 8 supervision (V12.31).

Computes 15 compositional features directly from formula composition
(element indices + molar fractions). No external data needed — these
targets are available for EVERY sample in the dataset.

Uses element_properties.py for all elemental lookups.
"""

import torch
import numpy as np
from typing import Tuple, Optional

from superconductor.encoders.element_properties import (
    ELEMENT_PROPERTIES,
    ELEMENT_SYMBOLS,
)
from superconductor.models.physics_z import PhysicsZ


# Build lookup tables as numpy arrays indexed by atomic number (0-118)
# This avoids per-element dict lookups during batch computation.
_MAX_Z = len(ELEMENT_SYMBOLS)  # 119 (0=placeholder + 118 elements)

_MASS = np.zeros(_MAX_Z, dtype=np.float32)
_ELECTRONEGATIVITY = np.zeros(_MAX_Z, dtype=np.float32)
_ATOMIC_RADIUS = np.zeros(_MAX_Z, dtype=np.float32)
_IONIZATION_ENERGY = np.zeros(_MAX_Z, dtype=np.float32)
_MELTING_POINT = np.zeros(_MAX_Z, dtype=np.float32)
_VALENCE = np.zeros(_MAX_Z, dtype=np.float32)
_D_ELECTRONS = np.zeros(_MAX_Z, dtype=np.float32)
_F_ELECTRONS = np.zeros(_MAX_Z, dtype=np.float32)

for sym, props in ELEMENT_PROPERTIES.items():
    z = props['Z']
    if z < _MAX_Z:
        _MASS[z] = props.get('mass', 0.0)
        _ELECTRONEGATIVITY[z] = props.get('electronegativity', 0.0)
        _ATOMIC_RADIUS[z] = props.get('atomic_radius', 0.0)
        _IONIZATION_ENERGY[z] = props.get('ionization_energy', 0.0)
        _MELTING_POINT[z] = props.get('melting_point', 0.0)
        _VALENCE[z] = props.get('valence', 0.0)
        _D_ELECTRONS[z] = props.get('d_electrons', 0.0)
        _F_ELECTRONS[z] = props.get('f_electrons', 0.0)


class CompositionalTargetComputer:
    """Compute Block 8 (Compositional) supervision targets from formula/composition.

    These are computable for EVERY sample -- no external data needed.
    Pre-computed once during dataset creation and stored as a tensor.
    """

    TARGET_NAMES = [
        'n_elements',      # Count of distinct elements
        'mw',              # Molecular weight (g/mol)
        'x_h',             # Hydrogen fraction
        'z_avg',           # Average atomic number (weighted)
        'z_max',           # Maximum atomic number
        'en_avg',          # Average electronegativity (Pauling)
        'en_diff',         # Electronegativity max-min
        'r_avg',           # Average atomic radius
        'r_ratio',         # Radius ratio max/min
        'vec',             # Valence electron concentration
        'd_orbital_frac',  # Fraction of d-electron elements
        'f_orbital_frac',  # Fraction of f-electron elements
        'ie_avg',          # Average ionization energy
        'tm_avg',          # Average melting temperature
        'delta_size',      # Size mismatch parameter delta
    ]
    N_TARGETS = 15

    # Mapping from target name -> PhysicsZ coordinate index
    TARGET_TO_COORD = {
        'n_elements': PhysicsZ.N_ELEMENTS,
        'mw': PhysicsZ.MW,
        'x_h': PhysicsZ.X_H,
        'z_avg': PhysicsZ.Z_AVG,
        'z_max': PhysicsZ.Z_MAX,
        'en_avg': PhysicsZ.EN_AVG,
        'en_diff': PhysicsZ.EN_DIFF,
        'r_avg': PhysicsZ.R_AVG,
        'r_ratio': PhysicsZ.R_RATIO,
        'vec': PhysicsZ.VEC,
        'd_orbital_frac': PhysicsZ.D_ORBITAL_FRAC,
        'f_orbital_frac': PhysicsZ.F_ORBITAL_FRAC,
        'ie_avg': PhysicsZ.IE_AVG,
        'tm_avg': PhysicsZ.TM_AVG,
        'delta_size': PhysicsZ.DELTA_SIZE,
    }

    def __init__(self):
        # Convert lookup tables to torch tensors for GPU computation
        self.mass_table = torch.from_numpy(_MASS)
        self.en_table = torch.from_numpy(_ELECTRONEGATIVITY)
        self.radius_table = torch.from_numpy(_ATOMIC_RADIUS)
        self.ie_table = torch.from_numpy(_IONIZATION_ENERGY)
        self.tm_table = torch.from_numpy(_MELTING_POINT)
        self.valence_table = torch.from_numpy(_VALENCE)
        self.d_elec_table = torch.from_numpy(_D_ELECTRONS)
        self.f_elec_table = torch.from_numpy(_F_ELECTRONS)

    def compute_from_batch(
        self,
        elem_idx: torch.Tensor,
        elem_frac: torch.Tensor,
        elem_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute targets from batch tensors [batch, max_elements].

        Args:
            elem_idx: [batch, 12] atomic numbers (0=padding)
            elem_frac: [batch, 12] molar fractions
            elem_mask: [batch, 12] valid element mask (bool)

        Returns:
            targets: [batch, 15] raw (un-normalized) compositional targets
        """
        device = elem_idx.device
        batch_size = elem_idx.size(0)
        max_elem = elem_idx.size(1)

        # Move lookup tables to device
        mass = self.mass_table.to(device)
        en = self.en_table.to(device)
        radius = self.radius_table.to(device)
        ie = self.ie_table.to(device)
        tm = self.tm_table.to(device)
        valence = self.valence_table.to(device)
        d_elec = self.d_elec_table.to(device)
        f_elec = self.f_elec_table.to(device)

        # Clamp indices to valid range
        idx = elem_idx.clamp(0, len(mass) - 1).long()  # [B, 12]
        frac = elem_frac * elem_mask.float()  # Zero out padding fractions
        frac_sum = frac.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1]
        frac_norm = frac / frac_sum  # Normalize to sum=1

        # Lookup properties for each element position
        elem_mass = mass[idx]       # [B, 12]
        elem_en = en[idx]           # [B, 12]
        elem_radius = radius[idx]   # [B, 12]
        elem_ie = ie[idx]           # [B, 12]
        elem_tm = tm[idx]           # [B, 12]
        elem_val = valence[idx]     # [B, 12]
        elem_d = d_elec[idx]        # [B, 12]
        elem_f = f_elec[idx]        # [B, 12]
        elem_z = idx.float()        # [B, 12] atomic numbers

        targets = torch.zeros(batch_size, self.N_TARGETS, device=device)

        # 0: n_elements - count of distinct valid elements
        targets[:, 0] = elem_mask.float().sum(dim=1)

        # 1: mw - molecular weight (sum of mass * fraction, but fraction is molar
        #    so MW = sum(fraction_i * mass_i) where fraction is already normalized)
        targets[:, 1] = (frac_norm * elem_mass).sum(dim=1)

        # 2: x_h - hydrogen fraction (atomic number 1)
        h_mask = (idx == 1) & elem_mask
        targets[:, 2] = (frac_norm * h_mask.float()).sum(dim=1)

        # 3: z_avg - weighted average atomic number
        targets[:, 3] = (frac_norm * elem_z).sum(dim=1)

        # 4: z_max - maximum atomic number among valid elements
        # Set invalid positions to 0 so they don't affect max
        z_masked = elem_z * elem_mask.float()
        targets[:, 4] = z_masked.max(dim=1).values

        # 5: en_avg - weighted average electronegativity
        targets[:, 5] = (frac_norm * elem_en).sum(dim=1)

        # 6: en_diff - electronegativity difference (max - min among valid elements)
        en_masked = elem_en.clone()
        en_masked[~elem_mask] = float('inf')
        en_min = en_masked.min(dim=1).values
        en_masked_max = elem_en.clone()
        en_masked_max[~elem_mask] = float('-inf')
        en_max = en_masked_max.max(dim=1).values
        # Guard: if only 1 element, diff = 0
        en_diff = en_max - en_min
        en_diff[en_diff.isinf()] = 0.0
        targets[:, 6] = en_diff

        # 7: r_avg - weighted average atomic radius
        targets[:, 7] = (frac_norm * elem_radius).sum(dim=1)

        # 8: r_ratio - radius ratio max/min
        r_masked = elem_radius.clone()
        r_masked[~elem_mask] = float('inf')
        r_min = r_masked.min(dim=1).values
        r_masked_max = elem_radius.clone()
        r_masked_max[~elem_mask] = float('-inf')
        r_max = r_masked_max.max(dim=1).values
        r_ratio = r_max / r_min.clamp(min=1.0)
        r_ratio[r_ratio.isinf()] = 1.0
        targets[:, 8] = r_ratio

        # 9: vec - valence electron concentration (weighted average)
        targets[:, 9] = (frac_norm * elem_val).sum(dim=1)

        # 10: d_orbital_frac - fraction of composition from d-electron elements
        has_d = (elem_d > 0).float() * elem_mask.float()
        targets[:, 10] = (frac_norm * has_d).sum(dim=1)

        # 11: f_orbital_frac - fraction of composition from f-electron elements
        has_f = (elem_f > 0).float() * elem_mask.float()
        targets[:, 11] = (frac_norm * has_f).sum(dim=1)

        # 12: ie_avg - weighted average ionization energy
        targets[:, 12] = (frac_norm * elem_ie).sum(dim=1)

        # 13: tm_avg - weighted average melting temperature
        targets[:, 13] = (frac_norm * elem_tm).sum(dim=1)

        # 14: delta_size - size mismatch parameter
        # delta = sqrt(sum_i c_i * (1 - r_i / r_avg)^2)
        r_avg = targets[:, 7].unsqueeze(1).clamp(min=1.0)  # [B, 1]
        r_dev = (1.0 - elem_radius / r_avg) ** 2  # [B, 12]
        delta_sq = (frac_norm * r_dev * elem_mask.float()).sum(dim=1)
        targets[:, 14] = delta_sq.sqrt()

        return targets

    def compute_from_dataset(
        self,
        elem_idx_all: torch.Tensor,
        elem_frac_all: torch.Tensor,
        elem_mask_all: torch.Tensor,
        batch_size: int = 1024,
    ) -> Tuple[torch.Tensor, dict]:
        """Pre-compute for entire dataset.

        Args:
            elem_idx_all: [N, 12] atomic numbers
            elem_frac_all: [N, 12] molar fractions
            elem_mask_all: [N, 12] valid element mask

        Returns:
            targets_normalized: [N, 15] z-score normalized compositional targets
            norm_stats: dict with 'mean' and 'std' arrays for denormalization
        """
        N = elem_idx_all.size(0)

        # Compute raw targets in batches to avoid memory issues
        all_targets = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_targets = self.compute_from_batch(
                elem_idx_all[start:end],
                elem_frac_all[start:end],
                elem_mask_all[start:end],
            )
            all_targets.append(batch_targets)
        raw_targets = torch.cat(all_targets, dim=0)  # [N, 15]

        # Normalization strategy per target:
        # - log1p for order-of-magnitude quantities (mw, ie_avg, tm_avg)
        # - linear/z-score for bounded quantities
        LOG_TARGETS = {1, 12, 13}  # mw, ie_avg, tm_avg

        normalized = raw_targets.clone()
        for i in LOG_TARGETS:
            normalized[:, i] = torch.log1p(normalized[:, i].clamp(min=0))

        # Z-score normalization
        mean = normalized.mean(dim=0)  # [15]
        std = normalized.std(dim=0).clamp(min=1e-8)  # [15]
        normalized = (normalized - mean) / std

        norm_stats = {
            'mean': mean.numpy(),
            'std': std.numpy(),
            'log_targets': sorted(LOG_TARGETS),
        }

        return normalized, norm_stats

    @staticmethod
    def get_coord_indices():
        """Get the PhysicsZ coordinate indices for each of the 15 targets.

        Returns list of ints, same order as TARGET_NAMES.
        """
        return [
            CompositionalTargetComputer.TARGET_TO_COORD[name]
            for name in CompositionalTargetComputer.TARGET_NAMES
        ]
