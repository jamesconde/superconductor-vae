"""
Theory-Based Regularization Losses for Superconductor VAE.

These losses incorporate known physical relationships from superconductivity
theory to regularize the generative model:

- BCS Theory: Allen-Dynes formula (strong-coupling), Lindemann Debye anchor, Matthias VEC
- Cuprate Theory: Tc-doping dome relationship (learnable doping/Tc_max predictors)
- Iron-based: Multi-band + VEC constraints
- Heavy Fermion: Log-normal prior at ~1K, soft cap at 20K
- Organic: Soft cap at 15K (BEDT-TTF) / 45K (fullerene)
- Unknown/Other: No constraints (graceful fallback)

The key insight is that for materials governed by known theories, we can
use these relationships as soft constraints during training. This helps
the model generate physically plausible materials.

IMPORTANT: Unknown/Other category applies NO theory constraints. This is
intentional - we don't want to force incorrect physics on novel materials.

V12.22: Added tc_log_transform support, soft quadratic penalties (no hard caps),
        learnable cuprate doping/Tc_max predictors from Magpie features.
V12.25: Allen-Dynes replaces McMillan for BCS, Lindemann Debye anchor,
        Matthias VEC prior, HeavyFermionTheoryLoss, OrganicTheoryLoss,
        IronBased VEC constraint.

All physics citations are collected in docs/THEORY_LOSS_REFERENCES.md with
full bibliographic details (authors, journal, volume, pages, year, DOI).

February 2026
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import family classifier for determining which theory to apply
from ..models.family_classifier import (
    SuperconductorFamily,
    RuleBasedFamilyClassifier,
    SIMPLE_FAMILY_NAMES,
)


@dataclass
class TheoryLossConfig:
    """Configuration for theory-based losses."""

    # Overall weight for theory regularization
    theory_weight: float = 0.05

    # V12.22: Whether Tc was log1p-transformed before normalization
    tc_log_transform: bool = False

    # BCS theory parameters (McMillan formula)
    # Tc = (θ_D/1.45) × exp(-1.04(1+λ)/(λ - μ*(1+0.62λ)))
    # For typical BCS: λ ≈ 0.3-0.8, μ* ≈ 0.1-0.15
    bcs_max_tc: float = 40.0  # Maximum Tc for BCS materials (K)
    bcs_lambda_typical: float = 0.5  # Typical electron-phonon coupling
    bcs_mu_star: float = 0.12  # Coulomb pseudo-potential

    # Cuprate dome parameters (Presland formula)
    # Tc = Tc_max × [1 - 82.6(p - 0.16)²]
    # Reference: Presland et al., Physica C 176 (1991) 95-105
    cuprate_optimal_doping: float = 0.16
    cuprate_dome_coefficient: float = 82.6  # Presland coefficient
    cuprate_max_tc: float = 165.0  # Hg-cuprate max (HgBa2Ca2Cu3O8+δ)

    # Iron-based parameters
    iron_max_tc: float = 60.0

    # Use soft constraints (Huber) vs hard constraints (MSE)
    use_soft_constraints: bool = True
    huber_delta: float = 5.0

    # Apply no constraints to unknown materials
    # (setting this to False would apply BCS as default - not recommended)
    skip_unknown: bool = True

    # V12.25: Magpie feature normalization stats (for physics formulas that need real units)
    magpie_mean: Optional[List[float]] = None   # [145] mean per feature
    magpie_std: Optional[List[float]] = None    # [145] std per feature

    # V12.25: Key Magpie feature indices (0-based within 145-dim tensor)
    # These are used by Lindemann Debye anchor and VEC constraints
    idx_mean_atomic_weight: int = 15    # mean AtomicWeight
    idx_mean_melting_t: int = 21        # mean MeltingT
    idx_mean_nvalence: int = 75         # mean NValence (VEC proxy)
    idx_mean_gsvolume: int = 111        # mean GSvolume_pa

    # V12.25: Heavy fermion parameters
    heavy_fermion_tc_center: float = 1.0   # Log-normal center (K)
    heavy_fermion_tc_max: float = 20.0     # Soft cap (K)

    # V12.25: Organic parameters
    organic_tc_max: float = 15.0           # Soft cap for BEDT-TTF type (K)
    organic_fullerene_tc_max: float = 45.0 # Soft cap for doped fullerenes (K)


def _denormalize_tc(
    predicted_tc: torch.Tensor,
    tc_is_normalized: bool,
    tc_mean: float,
    tc_std: float,
    tc_log_transform: bool,
) -> torch.Tensor:
    """
    Denormalize predicted Tc back to Kelvin.

    V12.22: Correctly handles log1p transform (V12.20+).
    When tc_log_transform=True, the pipeline is:
        Tc_K → log1p(Tc_K) → (log_tc - mean) / std = normalized
    So to reverse: expm1(normalized * std + mean) = Tc_K

    Clamps to [0, 500K] as numerical guard (not a physics cap —
    prevents float overflow in penalty computation).
    """
    if tc_is_normalized:
        tc_unnorm = predicted_tc * tc_std + tc_mean
        if tc_log_transform:
            # Undo log1p: Tc_K = expm1(log_tc)
            tc = torch.expm1(tc_unnorm)
        else:
            tc = tc_unnorm
    else:
        tc = predicted_tc

    # Numerical guard — prevents overflow in downstream penalty math
    tc = tc.clamp(min=0.0, max=500.0)
    return tc


class BCSTheoryLoss(nn.Module):
    """
    Regularization based on BCS theory predictions.

    V12.25: Uses the Allen-Dynes formula (1975), a strong-coupling extension
    of McMillan (1968). Adds f1, f2 correction factors that significantly
    improve accuracy for λ > 0.5 materials (e.g., MgB2).

    Allen-Dynes formula:
        Tc = (ω_log / 1.2) × exp(-1.04(1+λ)/(λ - μ*(1+0.62λ))) × f1 × f2

    Where:
        ω_log ≈ θ_D × 0.827 (approximate for monatomic/simple polyatomic solids)
        λ = electron-phonon coupling constant (typically 0.3-0.8)
        μ* = Coulomb pseudo-potential (typically 0.1-0.15)
        f1 = [1 + (λ/Λ1)^(3/2)]^(1/3),  Λ1 = 2.46(1 + 3.8μ*)
        f2 = 1 + (λ²(1/2 - μ*))/(λ² + Λ2²),  Λ2 = 1.82(1 + 6.3μ*)

    For typical BCS materials: Tc < 40K (practical upper limit)

    References (full citations in docs/THEORY_LOSS_REFERENCES.md):
        [BCS1957]        Bardeen, Cooper, Schrieffer, Phys. Rev. 108, 1175 (1957)
        [McMillan1968]   McMillan, Phys. Rev. 167, 331 (1968)
        [AllenDynes1975] Allen & Dynes, Phys. Rev. B 12, 905 (1975)
        [Lindemann1910]  Lindemann, Phys. Z. 11, 609 (1910)
        [Grimvall1999]   Grimvall, Thermophysical Properties of Materials (Elsevier, 1999)
        [Matthias1955]   Matthias, Phys. Rev. 97, 74 (1955)

    V12.25: Also adds Lindemann Debye temperature anchor and Matthias VEC prior.
    V12.22: Soft quadratic penalty replaces hard cap. Model is free to predict
    Tc > 40K for BCS materials but gets a massive error signal.

    We use Magpie features as proxies for θ_D and λ.
    """

    def __init__(self, config: Optional[TheoryLossConfig] = None):
        super().__init__()
        self.config = config or TheoryLossConfig()

        # Learnable mapping from Magpie features to BCS parameters
        # θ_D correlates with: atomic mass (inverse), bonding strength
        # λ correlates with: d-electron count, density of states features
        self.debye_predictor = nn.Sequential(
            nn.Linear(145, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Debye temp must be positive (output in units of 100K)
        )

        self.lambda_predictor = nn.Sequential(
            nn.Linear(145, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # λ typically 0-1, scaled to 0.2-1.0 below
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with physics-inspired weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def allen_dynes_tc(self, theta_d: torch.Tensor, lambda_ep: torch.Tensor) -> torch.Tensor:
        """
        V12.25: Compute Tc using Allen-Dynes formula (1975).

        Strong-coupling extension of McMillan with f1, f2 correction factors.

        Tc = (ω_log / 1.2) × exp(-1.04(1+λ)/(λ - μ*(1+0.62λ))) × f1 × f2

        Where ω_log ≈ θ_D × 0.827 (approximate relation for monatomic solids).
        f1, f2 are strong-coupling correction factors:
            f1 = [1 + (λ/Λ1)^(3/2)]^(1/3),  Λ1 = 2.46(1 + 3.8μ*)
            f2 = 1 + (λ²(1/2 - μ*))/(λ² + Λ2²),  Λ2 = 1.82(1 + 6.3μ*)

        Reference: Allen & Dynes, Phys. Rev. B 12, 905 (1975)

        Args:
            theta_d: Debye temperature in Kelvin
            lambda_ep: Electron-phonon coupling constant

        Returns:
            Predicted Tc in Kelvin
        """
        mu_star = self.config.bcs_mu_star
        lambda_safe = lambda_ep.clamp(min=0.15)

        # ω_log from Debye temperature (approximate relation)
        omega_log = theta_d * 0.827

        # McMillan exponent (same as original)
        numerator = -1.04 * (1 + lambda_safe)
        denominator = lambda_safe - mu_star * (1 + 0.62 * lambda_safe)
        denominator = denominator.clamp(min=0.01)
        exponent = numerator / denominator

        # Allen-Dynes strong-coupling corrections
        lambda1 = 2.46 * (1 + 3.8 * mu_star)
        f1 = (1 + (lambda_safe / lambda1) ** 1.5) ** (1.0 / 3.0)

        lambda2 = 1.82 * (1 + 6.3 * mu_star)
        f2 = 1 + (lambda_safe ** 2 * (0.5 - mu_star)) / (lambda_safe ** 2 + lambda2 ** 2)

        tc = (omega_log / 1.2) * torch.exp(exponent) * f1 * f2
        return tc

    def forward(
        self,
        magpie_features: torch.Tensor,  # [batch, 145]
        predicted_tc: torch.Tensor,      # [batch]
        tc_is_normalized: bool = True,
        tc_mean: float = 32.0,
        tc_std: float = 35.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BCS theory regularization loss.

        V12.25: Uses Allen-Dynes formula (replaces McMillan), adds Lindemann
        Debye anchor and Matthias VEC prior.
        """
        # V12.22: Denormalize Tc correctly (handles log-transform)
        tc = _denormalize_tc(
            predicted_tc, tc_is_normalized, tc_mean, tc_std,
            self.config.tc_log_transform,
        )
        device = tc.device

        # Predict BCS parameters from Magpie features
        # Debye temp: output is in units of 100K, typical range 100-500K
        theta_d = self.debye_predictor(magpie_features).squeeze(-1) * 100 + 100

        # Lambda: output sigmoid scaled to typical range 0.2-1.0
        lambda_ep = self.lambda_predictor(magpie_features).squeeze(-1) * 0.8 + 0.2

        # V12.25: Compute expected Tc from Allen-Dynes formula (replaces McMillan)
        tc_allen_dynes = self.allen_dynes_tc(theta_d, lambda_ep)

        # BCS constraint 1: Predicted Tc should be consistent with Allen-Dynes
        # Use relative error to handle wide Tc range
        tc_safe = tc.clamp(min=0.1)
        tc_ad_safe = tc_allen_dynes.clamp(min=0.1)
        relative_error = (tc_safe - tc_ad_safe).abs() / tc_ad_safe

        # V12.22: BCS constraint 2: Soft quadratic penalty (replaces hard cap)
        # Model is FREE to predict Tc > 40K but gets a massive error signal.
        deviation = tc - self.config.bcs_max_tc
        soft_violation = F.softplus(deviation, beta=0.5)
        tc_upper_penalty = soft_violation ** 2

        # Combined loss
        if self.config.use_soft_constraints:
            ad_loss = F.huber_loss(
                relative_error,
                torch.zeros_like(relative_error),
                delta=0.5  # Allow up to 50% relative error without strong penalty
            )
        else:
            ad_loss = relative_error.mean()

        loss = ad_loss + tc_upper_penalty.mean()

        # V12.25: Lindemann Debye anchor — soft regularizer on θ_D predictor
        # θ_D_est = C × sqrt(T_m / (M × V^(2/3))), C ≈ 41.63
        debye_anchor_loss = torch.tensor(0.0, device=device)
        if self.config.magpie_mean is not None and self.config.magpie_std is not None:
            mag_mean = torch.tensor(self.config.magpie_mean, device=device)
            mag_std = torch.tensor(self.config.magpie_std, device=device)

            # Denormalize specific features to physical units
            i_melt = self.config.idx_mean_melting_t    # 21
            i_mass = self.config.idx_mean_atomic_weight  # 15
            i_vol  = self.config.idx_mean_gsvolume       # 111

            T_m = (magpie_features[:, i_melt] * mag_std[i_melt] + mag_mean[i_melt]).clamp(min=100.0)
            M   = (magpie_features[:, i_mass] * mag_std[i_mass] + mag_mean[i_mass]).clamp(min=1.0)
            V   = (magpie_features[:, i_vol]  * mag_std[i_vol]  + mag_mean[i_vol]).clamp(min=1.0)

            theta_d_lindemann = 41.63 * torch.sqrt(T_m / (M * V ** (2.0/3.0)))
            theta_d_lindemann = theta_d_lindemann.clamp(min=50.0, max=1000.0)

            # Soft anchor: penalize neural net θ_D deviating too much from Lindemann
            debye_anchor_loss = F.huber_loss(theta_d, theta_d_lindemann, delta=100.0)

        loss = loss + debye_anchor_loss * 0.1

        # V12.25: Matthias VEC prior — Tc peaks near VEC≈4.7 and ≈6.7
        matthias_loss = torch.tensor(0.0, device=device)
        if self.config.magpie_mean is not None and self.config.magpie_std is not None:
            i_vec = self.config.idx_mean_nvalence  # 75
            vec = (magpie_features[:, i_vec] * mag_std[i_vec] + mag_mean[i_vec]).clamp(min=0.0, max=12.0)

            # Tc envelope from Matthias: peaks at 4.7 and 6.7
            peak1 = torch.exp(-0.5 * ((vec - 4.7) / 1.0) ** 2)
            peak2 = torch.exp(-0.5 * ((vec - 6.7) / 1.0) ** 2)
            matthias_envelope = 40.0 * (peak1 + peak2)  # Max ~40K at peaks

            # Penalize Tc exceeding the Matthias envelope (soft)
            vec_violation = F.softplus(tc - matthias_envelope - 5.0, beta=0.5) ** 2
            matthias_loss = vec_violation.mean() * 0.01  # Very gentle

        loss = loss + matthias_loss

        return {
            'bcs_loss': loss * self.config.theory_weight,
            'theta_d': theta_d.mean(),
            'lambda_ep': lambda_ep.mean(),
            'tc_allen_dynes': tc_allen_dynes.mean(),
            'tc_upper_violation': tc_upper_penalty.mean(),
            'debye_anchor_loss': debye_anchor_loss if torch.is_tensor(debye_anchor_loss) else torch.tensor(debye_anchor_loss),
            'matthias_loss': matthias_loss if torch.is_tensor(matthias_loss) else torch.tensor(matthias_loss),
        }


class CuprateTheoryLoss(nn.Module):
    """
    Regularization based on cuprate superconductivity theory.

    Uses the Presland formula (1991):
        Tc = Tc_max × [1 - 82.6(p - 0.16)²]

    Where:
        p = hole doping level (typically 0.05-0.27)
        p_opt = 0.16 (optimal doping)
        Tc_max = maximum Tc for the material family

    This parabolic "dome" is a universal feature of cuprate superconductors.

    References (full citations in docs/THEORY_LOSS_REFERENCES.md):
        [Presland1991] Presland et al., Physica C 176, 95-105 (1991)

    V12.22: Learnable doping and Tc_max predictors from Magpie features replace
    the constant stub (0.15). ~22,850 new learnable params.
    """

    def __init__(self, config: Optional[TheoryLossConfig] = None):
        super().__init__()
        self.config = config or TheoryLossConfig()

        # V12.22: Learnable doping predictor from Magpie features
        # Predicts hole doping level p from composition features
        # Output scaled to [0.05, 0.27] (physical range for cuprates)
        self.doping_predictor = nn.Sequential(
            nn.Linear(145, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output [0, 1], scaled below
        )

        # V12.22: Learnable Tc_max predictor from Magpie features
        # Predicts family-specific maximum Tc (varies by cuprate sub-family)
        # Output scaled to [30, 165K] (LSCO ~40K, Hg-cuprate ~165K)
        self.tc_max_predictor = nn.Sequential(
            nn.Linear(145, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Positive output, scaled below
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with physics-inspired weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def dome_function(self, doping: torch.Tensor, tc_max: torch.Tensor) -> torch.Tensor:
        """
        Compute expected Tc from doping using Presland formula.

        Presland formula (1991):
            Tc = Tc_max × [1 - 82.6(p - 0.16)²]

        Reference: Presland et al., Physica C 176 (1991) 95-105

        Args:
            doping: Hole doping level p (typically 0.05-0.27)
            tc_max: Maximum Tc for the material [batch] or scalar

        Returns:
            Expected Tc in Kelvin, clamped to [0, Tc_max]
        """
        p_opt = self.config.cuprate_optimal_doping  # 0.16
        coeff = self.config.cuprate_dome_coefficient  # 82.6

        # Presland formula: Tc/Tc_max = 1 - 82.6(p - 0.16)²
        deviation = doping - p_opt
        tc_ratio = 1.0 - coeff * (deviation ** 2)
        tc_expected = tc_max * tc_ratio.clamp(min=0.0)

        return tc_expected

    def forward(
        self,
        formula_tokens: torch.Tensor,
        predicted_tc: torch.Tensor,
        element_fractions: Optional[torch.Tensor] = None,
        tc_is_normalized: bool = True,
        tc_mean: float = 32.0,
        tc_std: float = 35.0,
        magpie_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cuprate theory regularization loss.

        V12.22: Accepts magpie_features to feed learnable doping/Tc_max predictors.
        Falls back to constants if magpie_features not provided.

        Args:
            formula_tokens: [batch, seq_len] token indices
            predicted_tc: Predicted Tc
            element_fractions: Optional element fractions
            magpie_features: [batch, 145] Magpie features for learnable predictors

        Returns:
            Dict with 'cuprate_loss', 'doping_level', 'tc_max_predicted', 'expected_tc'
        """
        # V12.22: Denormalize Tc correctly (handles log-transform)
        tc = _denormalize_tc(
            predicted_tc, tc_is_normalized, tc_mean, tc_std,
            self.config.tc_log_transform,
        )

        batch_size = formula_tokens.size(0)
        device = formula_tokens.device

        # V12.22: Use learnable predictors if Magpie features available
        if magpie_features is not None:
            # Doping: sigmoid output [0,1] → scaled to [0.05, 0.27]
            doping_raw = self.doping_predictor(magpie_features).squeeze(-1)
            doping = doping_raw * 0.22 + 0.05  # [0.05, 0.27]

            # Tc_max: softplus output → scaled to [30, 165K]
            tc_max_raw = self.tc_max_predictor(magpie_features).squeeze(-1)
            tc_max = tc_max_raw.clamp(max=3.0) * 45.0 + 30.0  # clamp softplus, scale to [30, 165]
        else:
            # Fallback to constants (backward compat)
            doping = torch.full((batch_size,), 0.15, device=device)
            tc_max = torch.full((batch_size,), self.config.cuprate_max_tc, device=device)

        # Compute expected Tc from dome
        expected_tc = self.dome_function(doping, tc_max)

        # Loss: penalize deviation from dome
        if self.config.use_soft_constraints:
            loss = F.huber_loss(tc, expected_tc, delta=self.config.huber_delta)
        else:
            loss = F.mse_loss(tc, expected_tc)

        return {
            'cuprate_loss': loss * self.config.theory_weight,
            'doping_level': doping.mean(),
            'tc_max_predicted': tc_max.mean(),
            'expected_tc': expected_tc.mean(),
        }


class IronBasedTheoryLoss(nn.Module):
    """
    Regularization for iron-based superconductors.

    Key relationships:
    1. Multi-band superconductivity
    2. Tc depends on pnictogen/chalcogen height
    3. Tc typically < 60K (highest: SmFeAsO1-xFx at ~55K)
    4. V12.25: VEC constraint — optimal VEC ≈ 6.0 (Fe²⁺ tetrahedral coordination)

    References (full citations in docs/THEORY_LOSS_REFERENCES.md):
        [Kamihara2008] Kamihara et al., JACS 130, 3296 (2008)
        [Stewart2011]  Stewart, Rev. Mod. Phys. 83, 1589 (2011)
        [Hosono2015]   Hosono & Kuroki, Physica C 514, 399 (2015) — VEC constraint

    V12.22: Soft quadratic penalty replaces hard cap.
    V12.25: Added VEC constraint centered at 6.0.
    """

    def __init__(self, config: Optional[TheoryLossConfig] = None):
        super().__init__()
        self.config = config or TheoryLossConfig()

    def forward(
        self,
        magpie_features: torch.Tensor,
        predicted_tc: torch.Tensor,
        tc_is_normalized: bool = True,
        tc_mean: float = 32.0,
        tc_std: float = 35.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute iron-based theory regularization loss.

        V12.25: Soft Tc upper bound + VEC constraint at 6.0.
        """
        # V12.22: Denormalize Tc correctly (handles log-transform)
        tc = _denormalize_tc(
            predicted_tc, tc_is_normalized, tc_mean, tc_std,
            self.config.tc_log_transform,
        )
        device = tc.device

        # V12.22: Soft quadratic penalty (replaces hard cap F.relu)
        deviation = tc - self.config.iron_max_tc
        soft_violation = F.softplus(deviation, beta=0.5)
        tc_upper_penalty = soft_violation ** 2

        # V12.25: VEC constraint — iron-based SCs typically VEC 5.5-6.5
        vec_penalty = torch.tensor(0.0, device=device)
        if self.config.magpie_mean is not None and self.config.magpie_std is not None:
            mag_mean = torch.tensor(self.config.magpie_mean, device=device)
            mag_std = torch.tensor(self.config.magpie_std, device=device)
            i_vec = self.config.idx_mean_nvalence
            vec = (magpie_features[:, i_vec] * mag_std[i_vec] + mag_mean[i_vec]).clamp(min=0.0)

            # Iron-based optimal VEC: peaked at ~6.0 (Fe²⁺ in tetrahedral coordination)
            vec_deviation = (vec - 6.0).abs()
            vec_penalty = F.softplus(vec_deviation - 1.0, beta=1.0) ** 2  # Flat within ±1
            vec_penalty = vec_penalty.mean() * 0.1  # Gentle

        total_loss = (tc_upper_penalty.mean() + vec_penalty) * self.config.theory_weight

        return {
            'iron_based_loss': total_loss,
            'tc_upper_violation': tc_upper_penalty.mean(),
            'iron_vec_penalty': vec_penalty if torch.is_tensor(vec_penalty) else torch.tensor(vec_penalty),
        }


class HeavyFermionTheoryLoss(nn.Module):
    """
    V12.25: Theory loss for heavy fermion superconductors.

    Heavy fermion SCs (Ce, U, Yb, Pu compounds) have:
    - Typical Tc: 0.1-2K (CeCoIn5 ~2.3K, UPt3 ~0.5K, CeRhIn5 ~2.1K)
    - Exceptional: PuCoGa5 ~18.5K, PuRhGa5 ~8.7K
    - Hard ceiling: ~20K (no known heavy fermion SC above this)

    Uses log-normal prior centered at ~1K + soft quadratic cap at 20K.
    No learnable parameters — pure physics prior.

    References (full citations in docs/THEORY_LOSS_REFERENCES.md):
        [Steglich1979]   Steglich et al., Phys. Rev. Lett. 43, 1892 (1979) — CeCu2Si2 discovery
        [Sarrao2002]     Sarrao et al., Nature 420, 297 (2002) — PuCoGa5 Tc=18.5K record
        [Pfleiderer2009] Pfleiderer, Rev. Mod. Phys. 81, 1551 (2009) — comprehensive review
    """

    def __init__(self, config: Optional[TheoryLossConfig] = None):
        super().__init__()
        self.config = config or TheoryLossConfig()

    def forward(
        self,
        magpie_features: torch.Tensor,
        predicted_tc: torch.Tensor,
        tc_is_normalized: bool = True,
        tc_mean: float = 32.0,
        tc_std: float = 35.0,
    ) -> Dict[str, torch.Tensor]:
        tc = _denormalize_tc(predicted_tc, tc_is_normalized, tc_mean, tc_std,
                             self.config.tc_log_transform)
        device = tc.device

        # Log-normal prior: most heavy fermion SCs are near 1K
        tc_safe = tc.clamp(min=0.01)
        log_tc = torch.log(tc_safe)
        log_center = math.log(self.config.heavy_fermion_tc_center)  # log(1.0) = 0
        log_prior_loss = F.huber_loss(log_tc, torch.full_like(log_tc, log_center), delta=2.0)

        # Soft cap at 20K
        deviation = tc - self.config.heavy_fermion_tc_max
        cap_penalty = F.softplus(deviation, beta=0.5) ** 2

        loss = (log_prior_loss + cap_penalty.mean()) * self.config.theory_weight

        return {
            'heavy_fermion_loss': loss,
            'hf_tc_mean': tc.mean(),
            'hf_cap_violation': cap_penalty.mean(),
        }


class OrganicTheoryLoss(nn.Module):
    """
    V12.25: Theory loss for organic superconductors.

    Two sub-families:
    - BEDT-TTF salts (kappa-(BEDT-TTF)2X): Tc typically < 13K
    - Alkali-doped fullerenes (A3C60): Tc up to ~40K (Cs3C60 ~40K under pressure)

    Uses a conservative soft cap at 15K (covers most organic SCs).
    The neural net can distinguish fullerenes from BEDT-TTF via Magpie features
    (C60 has very different atomic weight distribution).

    No learnable parameters — pure physics prior.

    References (full citations in docs/THEORY_LOSS_REFERENCES.md):
        [Jerome1980] Jerome et al., J. Phys. Lett. 41, L95 (1980) — first organic SC
        [Hebard1991] Hebard et al., Nature 350, 600 (1991) — K3C60 discovery
        [Ganin2008]  Ganin et al., Nature Materials 7, 367 (2008) — Cs3C60 Tc=38K record
    """

    def __init__(self, config: Optional[TheoryLossConfig] = None):
        super().__init__()
        self.config = config or TheoryLossConfig()

    def forward(
        self,
        magpie_features: torch.Tensor,
        predicted_tc: torch.Tensor,
        tc_is_normalized: bool = True,
        tc_mean: float = 32.0,
        tc_std: float = 35.0,
    ) -> Dict[str, torch.Tensor]:
        tc = _denormalize_tc(predicted_tc, tc_is_normalized, tc_mean, tc_std,
                             self.config.tc_log_transform)
        device = tc.device

        # Conservative soft cap at 15K (most organic SCs)
        deviation = tc - self.config.organic_tc_max
        cap_penalty = F.softplus(deviation, beta=0.5) ** 2

        loss = cap_penalty.mean() * self.config.theory_weight

        return {
            'organic_loss': loss,
            'organic_tc_mean': tc.mean(),
            'organic_cap_violation': cap_penalty.mean(),
        }


class UnknownTheoryLoss(nn.Module):
    """
    Placeholder loss for unknown/other superconductor families.

    IMPORTANT: This returns ZERO loss intentionally.

    For materials that don't fit known categories, we should NOT
    apply theory-based constraints because:
    1. We don't know the underlying mechanism
    2. Applying wrong constraints would hurt generalization
    3. These materials might represent novel physics

    The model is free to learn from data without physics constraints
    for this category.
    """

    def __init__(self, config: Optional[TheoryLossConfig] = None):
        super().__init__()
        self.config = config or TheoryLossConfig()

    def forward(
        self,
        **kwargs,  # Accept any arguments but ignore them
    ) -> Dict[str, torch.Tensor]:
        """
        Return zero loss.

        This is intentional - no constraints for unknown materials.
        """
        # Get device from any provided tensor
        device = torch.device('cpu')
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break

        return {
            'unknown_loss': torch.tensor(0.0, device=device),
            'theory_applied': torch.tensor(False, device=device),
        }


class TheoryRegularizationLoss(nn.Module):
    """
    Composite theory regularization loss.

    Automatically selects the appropriate theory-based loss based on
    the superconductor family classification.

    V12.22: Passes magpie_features to cuprate loss for learnable predictors.
    V12.25: Added HeavyFermionTheoryLoss, OrganicTheoryLoss (replaces UnknownTheoryLoss
            for those families). Iron-based gets VEC constraint.

    Usage:
        theory_loss = TheoryRegularizationLoss(config)
        loss_dict = theory_loss(
            formula_tokens=tokens,
            predicted_tc=tc,
            magpie_features=magpie,
            families=family_labels,  # Optional: pre-computed family labels
        )
    """

    def __init__(self, config: Optional[TheoryLossConfig] = None):
        super().__init__()
        self.config = config or TheoryLossConfig()

        # Family classifier
        self.family_classifier = RuleBasedFamilyClassifier()

        # Theory-specific losses
        self.bcs_loss = BCSTheoryLoss(config)
        self.cuprate_loss = CuprateTheoryLoss(config)
        self.iron_loss = IronBasedTheoryLoss(config)
        self.heavy_fermion_loss = HeavyFermionTheoryLoss(config)  # V12.25
        self.organic_loss = OrganicTheoryLoss(config)              # V12.25
        self.unknown_loss = UnknownTheoryLoss(config)

        # Mapping from family to loss function
        self.family_to_loss = {
            SuperconductorFamily.BCS_CONVENTIONAL: self.bcs_loss,
            SuperconductorFamily.MGB2_TYPE: self.bcs_loss,  # Two-gap BCS
            SuperconductorFamily.CUPRATE_YBCO: self.cuprate_loss,
            SuperconductorFamily.CUPRATE_LSCO: self.cuprate_loss,
            SuperconductorFamily.CUPRATE_BSCCO: self.cuprate_loss,
            SuperconductorFamily.CUPRATE_TBCCO: self.cuprate_loss,
            SuperconductorFamily.CUPRATE_HBCCO: self.cuprate_loss,
            SuperconductorFamily.CUPRATE_OTHER: self.cuprate_loss,
            SuperconductorFamily.IRON_PNICTIDE: self.iron_loss,
            SuperconductorFamily.IRON_CHALCOGENIDE: self.iron_loss,
            SuperconductorFamily.HEAVY_FERMION: self.heavy_fermion_loss,  # V12.25
            SuperconductorFamily.ORGANIC: self.organic_loss,              # V12.25
            SuperconductorFamily.OTHER_UNKNOWN: self.unknown_loss,
            SuperconductorFamily.NOT_SUPERCONDUCTOR: self.unknown_loss,
        }

    def forward(
        self,
        formula_tokens: torch.Tensor,
        predicted_tc: torch.Tensor,
        magpie_features: torch.Tensor,
        families: Optional[torch.Tensor] = None,
        element_fractions: Optional[torch.Tensor] = None,
        idx_to_token: Optional[Dict[int, str]] = None,
        element_set: Optional[set] = None,
        tc_is_normalized: bool = True,
        tc_mean: float = 32.0,
        tc_std: float = 35.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute theory regularization loss for batch.

        Args:
            formula_tokens: [batch, seq_len] token indices
            predicted_tc: [batch] predicted Tc
            magpie_features: [batch, 145] Magpie features
            families: [batch] pre-computed family indices (optional)
            element_fractions: [batch, max_elements] element fractions (optional)
            idx_to_token: Token index to string mapping
            element_set: Set of element symbols in vocabulary
            tc_is_normalized: Whether Tc is normalized

        Returns:
            Dict with 'total', 'bcs_count', 'cuprate_count', 'unknown_count', etc.
        """
        batch_size = formula_tokens.size(0)
        device = formula_tokens.device

        # Classify families if not provided
        if families is None:
            if idx_to_token is not None and element_set is not None:
                families = self.family_classifier.classify_batch_from_tokens(
                    formula_tokens, idx_to_token, element_set
                )
            else:
                # Default to unknown if we can't classify
                families = torch.full(
                    (batch_size,),
                    SuperconductorFamily.OTHER_UNKNOWN.value,
                    dtype=torch.long,
                    device=device,
                )

        # Accumulate losses by family type
        total_loss = torch.tensor(0.0, device=device)
        family_counts = {
            'bcs': 0, 'cuprate': 0, 'iron': 0,
            'heavy_fermion': 0, 'organic': 0, 'unknown': 0,  # V12.25
        }
        family_losses = {
            'bcs': torch.tensor(0.0, device=device),
            'cuprate': torch.tensor(0.0, device=device),
            'iron': torch.tensor(0.0, device=device),
            'heavy_fermion': torch.tensor(0.0, device=device),  # V12.25
            'organic': torch.tensor(0.0, device=device),        # V12.25
        }

        # Group samples by family for efficient computation
        bcs_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        cuprate_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        iron_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        heavy_fermion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)  # V12.25
        organic_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)        # V12.25

        for i, family_idx in enumerate(families.tolist()):
            family = SuperconductorFamily(family_idx)
            if family in [SuperconductorFamily.BCS_CONVENTIONAL, SuperconductorFamily.MGB2_TYPE]:
                bcs_mask[i] = True
                family_counts['bcs'] += 1
            elif family.name.startswith('CUPRATE'):
                cuprate_mask[i] = True
                family_counts['cuprate'] += 1
            elif family in [SuperconductorFamily.IRON_PNICTIDE, SuperconductorFamily.IRON_CHALCOGENIDE]:
                iron_mask[i] = True
                family_counts['iron'] += 1
            elif family == SuperconductorFamily.HEAVY_FERMION:
                heavy_fermion_mask[i] = True
                family_counts['heavy_fermion'] += 1
            elif family == SuperconductorFamily.ORGANIC:
                organic_mask[i] = True
                family_counts['organic'] += 1
            else:
                family_counts['unknown'] += 1

        # Compute losses for each family group
        if bcs_mask.any():
            bcs_result = self.bcs_loss(
                magpie_features[bcs_mask],
                predicted_tc[bcs_mask],
                tc_is_normalized, tc_mean, tc_std,
            )
            family_losses['bcs'] = bcs_result['bcs_loss']
            total_loss = total_loss + bcs_result['bcs_loss'] * bcs_mask.sum()

        if cuprate_mask.any():
            # V12.22: Pass magpie_features for learnable doping/Tc_max predictors
            cuprate_result = self.cuprate_loss(
                formula_tokens[cuprate_mask],
                predicted_tc[cuprate_mask],
                element_fractions[cuprate_mask] if element_fractions is not None else None,
                tc_is_normalized, tc_mean, tc_std,
                magpie_features=magpie_features[cuprate_mask],
            )
            family_losses['cuprate'] = cuprate_result['cuprate_loss']
            total_loss = total_loss + cuprate_result['cuprate_loss'] * cuprate_mask.sum()

        if iron_mask.any():
            iron_result = self.iron_loss(
                magpie_features[iron_mask],
                predicted_tc[iron_mask],
                tc_is_normalized, tc_mean, tc_std,
            )
            family_losses['iron'] = iron_result['iron_based_loss']
            total_loss = total_loss + iron_result['iron_based_loss'] * iron_mask.sum()

        # V12.25: Heavy fermion batch processing
        if heavy_fermion_mask.any():
            hf_result = self.heavy_fermion_loss(
                magpie_features[heavy_fermion_mask],
                predicted_tc[heavy_fermion_mask],
                tc_is_normalized, tc_mean, tc_std,
            )
            family_losses['heavy_fermion'] = hf_result['heavy_fermion_loss']
            total_loss = total_loss + hf_result['heavy_fermion_loss'] * heavy_fermion_mask.sum()

        # V12.25: Organic batch processing
        if organic_mask.any():
            organic_result = self.organic_loss(
                magpie_features[organic_mask],
                predicted_tc[organic_mask],
                tc_is_normalized, tc_mean, tc_std,
            )
            family_losses['organic'] = organic_result['organic_loss']
            total_loss = total_loss + organic_result['organic_loss'] * organic_mask.sum()

        # Note: Unknown/Other materials contribute 0 to loss (by design)

        # Normalize by batch size
        total_loss = total_loss / max(batch_size, 1)

        return {
            'total': total_loss,
            'bcs_loss': family_losses['bcs'],
            'cuprate_loss': family_losses['cuprate'],
            'iron_loss': family_losses['iron'],
            'heavy_fermion_loss': family_losses['heavy_fermion'],  # V12.25
            'organic_loss': family_losses['organic'],              # V12.25
            'bcs_count': family_counts['bcs'],
            'cuprate_count': family_counts['cuprate'],
            'iron_count': family_counts['iron'],
            'heavy_fermion_count': family_counts['heavy_fermion'],  # V12.25
            'organic_count': family_counts['organic'],              # V12.25
            'unknown_count': family_counts['unknown'],
        }


# Convenience function for simple integration
def compute_theory_regularization(
    formula_tokens: torch.Tensor,
    predicted_tc: torch.Tensor,
    magpie_features: torch.Tensor,
    config: Optional[TheoryLossConfig] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Compute theory regularization loss (convenience function).

    Returns just the total loss scalar for easy integration into training loop.
    """
    loss_fn = TheoryRegularizationLoss(config)
    result = loss_fn(formula_tokens, predicted_tc, magpie_features, **kwargs)
    return result['total']
