"""
Physics Z supervision losses for the 2048-dim latent vector (V12.31).

Loss classes that enforce physical meaning on named Z coordinate blocks:
- CompositionalSupervisionLoss: Block 8 (formula-derived, always active)
- MagpieEncodingLoss: Block 11 (learnable projection, always active)
- GLConsistencyLoss: Block 1 (Ginzburg-Landau self-consistency)
- BCSConsistencyLoss: Block 2 (BCS self-consistency)
- CobordismConsistencyLoss: Block 9 (derived from GL coords)
- DimensionlessRatioConsistencyLoss: Block 10 (cross-block ratios)
- DirectSupervisionLoss: Placeholder for future external physics data
- PhysicsZLoss: Combined loss aggregating all components

All losses enforce proportional consistency in learned normalized units.
No architectural changes to the encoder -- physics is imposed via gradient pressure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from superconductor.models.physics_z import PhysicsZ
from superconductor.data.compositional_targets import CompositionalTargetComputer


class CompositionalSupervisionLoss(nn.Module):
    """Block 8 supervision: compare Z compositional coords to formula-derived targets.

    Always active (targets available for every sample).
    """

    def __init__(self):
        super().__init__()
        # Cache coordinate indices for the 15 compositional targets
        self.coord_indices = CompositionalTargetComputer.get_coord_indices()

    def forward(self, z: torch.Tensor, comp_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, 2048] latent vector
            comp_targets: [batch, 15] normalized compositional targets

        Returns:
            MSE loss between Z compositional coords and targets
        """
        # Extract Z coords for each compositional target
        z_comp = z[:, self.coord_indices]  # [batch, 15]
        return F.mse_loss(z_comp, comp_targets)


class MagpieEncodingLoss(nn.Module):
    """Block 11 supervision: Z[450:512] should preserve Magpie information.

    Uses a learnable linear projection (145 -> 62) as the target.
    Both the encoder (via Z) and the projection are trained jointly --
    the projection learns the best 62-dim compression of Magpie features,
    and the encoder learns to match it in Z[450:512].
    """

    def __init__(self, magpie_dim: int = 145, z_magpie_dim: int = 62):
        super().__init__()
        self.projection = nn.Linear(magpie_dim, z_magpie_dim)

    def forward(self, z: torch.Tensor, magpie_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, 2048] latent vector
            magpie_features: [batch, 145] normalized Magpie features

        Returns:
            MSE loss between Z[450:512] and projected Magpie
        """
        z_magpie = z[:, PhysicsZ.MAGPIE_START:PhysicsZ.MAGPIE_END]  # [batch, 62]
        target = self.projection(magpie_features)  # [batch, 62]
        return F.mse_loss(z_magpie, target)


class GLConsistencyLoss(nn.Module):
    """Enforce Ginzburg-Landau interrelation constraints on Z.

    These are mathematical identities -- they apply to ALL samples.
    No external data needed.

    Constraints enforced (proportional, in learned normalized units):
    1. kappa = lambda / xi                      [Tinkham Ch. 4]
    2. Hc proportional to 1/(lambda * xi)       [Tinkham Eq. 4.38]
    3. Hc2 proportional to 1/xi^2               [Abrikosov 1957]
    4. E_cond proportional to Hc^2              [Tinkham Eq. 2.16]
    5. Hc1 proportional to (1/lambda^2) * ln(kappa) [Tinkham Eq. 5.22]

    Uses SmoothL1Loss (Huber) for robustness to large initial deviations,
    and clamps derived targets to [-100, 100] to prevent gradient explosions
    during early training when Z coords are randomly distributed.
    """

    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        kappa = z[:, PhysicsZ.KAPPA]
        xi = z[:, PhysicsZ.XI].clamp(min=0.01)
        lam = z[:, PhysicsZ.LAMBDA_L].clamp(min=0.01)
        hc = z[:, PhysicsZ.HC]
        hc1 = z[:, PhysicsZ.HC1]
        hc2 = z[:, PhysicsZ.HC2]
        e_cond = z[:, PhysicsZ.E_COND]

        # kappa = lambda / xi
        kappa_target = (lam / xi).clamp(-100, 100)
        loss_kappa = self.huber(kappa, kappa_target)

        # Hc proportional to 1/(lambda * xi)
        hc_target = (1.0 / (lam * xi)).clamp(-100, 100)
        loss_hc = self.huber(hc, hc_target)

        # Hc2 proportional to 1/xi^2
        hc2_target = (1.0 / xi.pow(2)).clamp(-100, 100)
        loss_hc2 = self.huber(hc2, hc2_target)

        # E_cond proportional to Hc^2
        econd_target = hc.detach().pow(2).clamp(-100, 100)
        loss_econd = self.huber(e_cond, econd_target)

        # Hc1 proportional to (1/lambda^2) * ln(kappa)
        hc1_target = ((1.0 / lam.pow(2)) * torch.log(kappa.clamp(min=1.01))).clamp(-100, 100)
        loss_hc1 = self.huber(hc1, hc1_target)

        return loss_kappa + loss_hc + loss_hc2 + loss_econd + loss_hc1


class BCSConsistencyLoss(nn.Module):
    """Enforce BCS relationships between Z coordinates.

    1. xi_0 proportional to v_F / Delta_0       [BCS 1957]
    2. Gap ratio = Delta_0 / (k_B * Tc) soft-bounded to [1, 5]
    3. lambda_L proportional to sqrt(m*/n_s)     [London 1935]

    Uses SmoothL1Loss and clamped targets for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        vf = z[:, PhysicsZ.V_F].clamp(min=0.01)
        delta0 = z[:, PhysicsZ.DELTA0].clamp(min=0.01)
        xi = z[:, PhysicsZ.XI]
        gap_ratio = z[:, PhysicsZ.GAP_RATIO]

        # xi_0 proportional to v_F / Delta_0
        xi_target = (vf / delta0).clamp(-100, 100)
        loss_xi = self.huber(xi, xi_target)

        # Gap ratio soft bounds [1, 5] -- BCS weak coupling is 3.528
        loss_gap = F.relu(gap_ratio - 5.0).mean() + F.relu(1.0 - gap_ratio).mean()

        return loss_xi + loss_gap


class CobordismConsistencyLoss(nn.Module):
    """Block 9 cobordism quantities derived from GL parameters.

    E_vortex proportional to (1/lambda)^2 * ln(kappa)   [Tinkham Ch. 5]
    E_domain = sigma_ns                                  [same as coord 10]
    Type I/II = kappa - 1/sqrt(2)                        [Abrikosov 1957]
    E_defect_min = min(E_vortex, E_domain, ...)

    Uses SmoothL1Loss and clamped targets for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        kappa = z[:, PhysicsZ.KAPPA]
        lam = z[:, PhysicsZ.LAMBDA_L].clamp(min=0.01)
        e_vortex = z[:, PhysicsZ.E_VORTEX]
        e_domain = z[:, PhysicsZ.E_DOMAIN]
        sigma_ns = z[:, PhysicsZ.SIGMA_NS]
        type_i_ii = z[:, PhysicsZ.TYPE_I_II]
        e_min = z[:, PhysicsZ.E_DEFECT_MIN]

        # E_vortex proportional to (1/lambda)^2 * ln(kappa)
        ev_target = ((1.0 / lam.pow(2)) * torch.log(kappa.clamp(min=1.01))).clamp(-100, 100)
        loss_ev = self.huber(e_vortex, ev_target)

        # E_domain = sigma_ns (same quantity)
        loss_ed = self.huber(e_domain, sigma_ns)

        # Type I/II = kappa - 1/sqrt(2)
        type_target = kappa.detach() - 1.0 / (2.0 ** 0.5)
        loss_type = self.huber(type_i_ii, type_target)

        # E_defect_min = min of defect energies
        e_stack = torch.stack([e_vortex.detach(), e_domain.detach()], dim=-1)
        loss_emin = self.huber(e_min, e_stack.min(dim=-1).values)

        return loss_ev + loss_ed + loss_type + loss_emin


class DimensionlessRatioConsistencyLoss(nn.Module):
    """Block 10 ratios are cross-block consistency checks.

    Tc/Theta_D = z[TC] / z[THETA_D]
    xi/l = z[XI] / z[L_MFP]

    Uses SmoothL1Loss and clamped targets for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Tc / Theta_D
        tc_theta_target = (z[:, PhysicsZ.TC] / z[:, PhysicsZ.THETA_D].clamp(min=0.01)).clamp(-100, 100)
        loss_tc_theta = self.huber(z[:, PhysicsZ.TC_THETA_D], tc_theta_target)

        # xi / l_mfp
        xi_l_target = (z[:, PhysicsZ.XI] / z[:, PhysicsZ.L_MFP].clamp(min=0.01)).clamp(-100, 100)
        loss_xi_l = self.huber(z[:, PhysicsZ.XI_L], xi_l_target)

        return loss_tc_theta + loss_xi_l


class ThermodynamicConsistencyLoss(nn.Module):
    """Block 7 consistency: enforce thermodynamic relationships in Z.

    V12.36: Three constraints, all definitionally true:

    1. z[TC] ≈ tc_normalized  (direct supervision — we have Tc for every sample)
       The encoder already predicts Tc via a separate head, but nothing forces the
       DESIGNATED Tc coordinate (z[210]) to hold that value. Currently z[210] only
       correlates r=0.49 with actual Tc while a random discovery coord hits r=0.87.

    2. z[TC_ONSET] >= z[TC_MIDPOINT] >= z[TC_ZERO]  (ordering)
       By definition, the onset temperature is always >= midpoint >= zero-resistance.
       Uses hinge loss: only penalizes violations, no effect when ordering holds.

    3. z[DELTA_TC] ≈ z[TC_ONSET] - z[TC_ZERO]  (mathematical identity)
       Transition width is defined as onset minus zero-resistance Tc.

    Uses SmoothL1Loss for numerical stability and gentle gradients.
    """

    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    def forward(self, z: torch.Tensor, tc_normalized: Optional[torch.Tensor] = None) -> torch.Tensor:
        tc_coord = z[:, PhysicsZ.TC]
        tc_onset = z[:, PhysicsZ.TC_ONSET]
        tc_mid = z[:, PhysicsZ.TC_MIDPOINT]
        tc_zero = z[:, PhysicsZ.TC_ZERO]
        delta_tc = z[:, PhysicsZ.DELTA_TC]

        loss = torch.tensor(0.0, device=z.device)

        # 1. z[TC] should match the actual normalized Tc input
        if tc_normalized is not None:
            tc_target = tc_normalized.squeeze()
            loss = loss + self.huber(tc_coord, tc_target)

        # 2. Soft ordering: onset >= midpoint >= zero (hinge loss on violations)
        # Only penalize when ordering is violated; no gradient when correct
        loss = loss + F.relu(tc_mid - tc_onset).mean()    # penalize mid > onset
        loss = loss + F.relu(tc_zero - tc_mid).mean()     # penalize zero > mid

        # 3. Delta_Tc = Tc_onset - Tc_zero (mathematical identity)
        delta_target = (tc_onset.detach() - tc_zero.detach())
        loss = loss + self.huber(delta_tc, delta_target)

        return loss


class StructuralConsistencyLoss(nn.Module):
    """Block 5 consistency: enforce structural relationships in Z.

    V12.36: One constraint (mathematical identity):

    z[VOLUME] ∝ z[LATTICE_A] * z[LATTICE_B] * z[LATTICE_C]

    Unit cell volume is proportional to the product of lattice parameters.
    Exact for orthogonal systems (cubic, tetragonal, orthorhombic),
    approximate for others (off by sin(angle) factors).

    Uses SmoothL1Loss with clamped targets to prevent gradient explosions.
    """

    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a = z[:, PhysicsZ.LATTICE_A].clamp(min=0.01)
        b = z[:, PhysicsZ.LATTICE_B].clamp(min=0.01)
        c = z[:, PhysicsZ.LATTICE_C].clamp(min=0.01)
        vol = z[:, PhysicsZ.VOLUME]

        # Volume proportional to a * b * c
        vol_target = (a * b * c).clamp(-100, 100)
        return self.huber(vol, vol_target)


class ElectronicConsistencyLoss(nn.Module):
    """Block 6 consistency: enforce electronic relationships in Z.

    V12.36: One constraint (Drude model, well-established):

    z[DRUDE_WEIGHT] ∝ z[PLASMA_FREQ]^2

    The Drude weight (optical spectral weight) is proportional to the square
    of the plasma frequency. This is a standard result in condensed matter
    physics, valid for all metals.

    Uses SmoothL1Loss with clamped targets to prevent gradient explosions.
    """

    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        plasma = z[:, PhysicsZ.PLASMA_FREQ]
        drude = z[:, PhysicsZ.DRUDE_WEIGHT]

        # Drude weight proportional to plasma_freq^2
        drude_target = plasma.detach().pow(2).clamp(-100, 100)
        return self.huber(drude, drude_target)


class DirectSupervisionLoss(nn.Module):
    """Supervise Z coordinates against ground-truth physics data.

    Only fires where data exists (per-sample masks).
    Currently a placeholder -- activates when external data files are provided.

    Expected data format: dict of {coord_name: (values_tensor, mask_tensor)}
    """

    def forward(
        self,
        z: torch.Tensor,
        physics_targets: Optional[Dict[str, tuple]] = None,
    ) -> torch.Tensor:
        if physics_targets is None:
            return torch.tensor(0.0, device=z.device)

        total = torch.tensor(0.0, device=z.device)
        count = 0
        for coord_name, (values, mask) in physics_targets.items():
            coord_idx = getattr(PhysicsZ, coord_name.upper(), None)
            if coord_idx is None:
                continue
            if mask.any():
                pred = z[mask, coord_idx]
                targ = values[mask].to(z.device)
                total = total + F.mse_loss(pred, targ)
                count += 1
        return total / max(count, 1)


class PhysicsZLoss(nn.Module):
    """Combined physics Z supervision loss.

    Aggregates all Z-coordinate losses with configurable weights.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.comp_loss = CompositionalSupervisionLoss()
        self.magpie_loss = MagpieEncodingLoss(magpie_dim=config.get('magpie_dim', 145))
        self.gl_loss = GLConsistencyLoss()
        self.bcs_loss = BCSConsistencyLoss()
        self.cobordism_loss = CobordismConsistencyLoss()
        self.ratio_loss = DimensionlessRatioConsistencyLoss()
        self.direct_loss = DirectSupervisionLoss()
        # V12.36: New consistency losses for previously unsupervised blocks
        self.thermo_loss = ThermodynamicConsistencyLoss()
        self.structural_loss = StructuralConsistencyLoss()
        self.electronic_loss = ElectronicConsistencyLoss()

    def forward(
        self,
        z: torch.Tensor,
        comp_targets: torch.Tensor,
        magpie_features: torch.Tensor,
        physics_targets: Optional[Dict[str, tuple]] = None,
        tc_normalized: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all physics Z losses.

        Args:
            z: [batch, 2048] latent vector
            comp_targets: [batch, 15] normalized compositional targets
            magpie_features: [batch, 145] normalized Magpie features
            physics_targets: optional dict of external physics data
            tc_normalized: [batch] or [batch, 1] normalized Tc values (V12.36)

        Returns:
            dict with individual loss components and 'total'
        """
        losses = {}
        losses['comp'] = self.comp_loss(z, comp_targets)
        losses['magpie_enc'] = self.magpie_loss(z, magpie_features)
        losses['gl_consistency'] = self.gl_loss(z)
        losses['bcs_consistency'] = self.bcs_loss(z)
        losses['cobordism'] = self.cobordism_loss(z)
        losses['ratios'] = self.ratio_loss(z)
        losses['direct'] = self.direct_loss(z, physics_targets)
        # V12.36: New consistency losses
        losses['thermo_consistency'] = self.thermo_loss(z, tc_normalized)
        losses['structural_consistency'] = self.structural_loss(z)
        losses['electronic_consistency'] = self.electronic_loss(z)

        comp_w = self.config.get('comp_weight', 1.0)
        magpie_w = self.config.get('magpie_enc_weight', 0.5)
        consistency_w = self.config.get('consistency_weight', 0.1)
        direct_w = self.config.get('direct_weight', 0.0)
        # V12.36: Separate weight for new block consistency (default lower than
        # existing consistency to avoid over-constraining during introduction)
        new_consistency_w = self.config.get('new_consistency_weight', 0.05)

        total = (
            comp_w * losses['comp']
            + magpie_w * losses['magpie_enc']
            + consistency_w * (
                losses['gl_consistency']
                + losses['bcs_consistency']
                + losses['cobordism']
                + losses['ratios']
            )
            + direct_w * losses['direct']
            + new_consistency_w * (
                losses['thermo_consistency']
                + losses['structural_consistency']
                + losses['electronic_consistency']
            )
        )
        losses['total'] = total
        return losses
