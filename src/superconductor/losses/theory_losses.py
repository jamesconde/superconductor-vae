"""
Theory-Based Regularization Losses for Superconductor VAE.

These losses incorporate known physical relationships from superconductivity
theory to regularize the generative model:

- BCS Theory: Tc scaling with Debye temperature, isotope effect
- Cuprate Theory: Tc-doping dome relationship (learnable doping/Tc_max predictors)
- Iron-based: Multi-band considerations
- Unknown/Other: No constraints (graceful fallback)

The key insight is that for materials governed by known theories, we can
use these relationships as soft constraints during training. This helps
the model generate physically plausible materials.

IMPORTANT: Unknown/Other category applies NO theory constraints. This is
intentional - we don't want to force incorrect physics on novel materials.

V12.22: Added tc_log_transform support, soft quadratic penalties (no hard caps),
        learnable cuprate doping/Tc_max predictors from Magpie features.

February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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

    Uses the McMillan formula (1968):
        Tc = (θ_D/1.45) × exp(-1.04(1+λ)/(λ - μ*(1+0.62λ)))

    Where:
        θ_D = Debye temperature (K)
        λ = electron-phonon coupling constant (typically 0.3-0.8)
        μ* = Coulomb pseudo-potential (typically 0.1-0.15)

    For typical BCS materials: Tc < 40K (practical upper limit)
    Reference: McMillan, Phys. Rev. 167, 331 (1968)

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

    def mcmillan_tc(self, theta_d: torch.Tensor, lambda_ep: torch.Tensor) -> torch.Tensor:
        """
        Compute Tc using McMillan formula.

        Tc = (θ_D/1.45) × exp(-1.04(1+λ)/(λ - μ*(1+0.62λ)))

        Args:
            theta_d: Debye temperature in Kelvin
            lambda_ep: Electron-phonon coupling constant

        Returns:
            Predicted Tc in Kelvin
        """
        mu_star = self.config.bcs_mu_star

        # Avoid division by zero: λ must be > μ*(1+0.62λ)
        # Rearranging: λ > μ*/(1 - 0.62μ*) ≈ 0.13 for μ* = 0.12
        lambda_safe = lambda_ep.clamp(min=0.15)

        numerator = -1.04 * (1 + lambda_safe)
        denominator = lambda_safe - mu_star * (1 + 0.62 * lambda_safe)
        denominator = denominator.clamp(min=0.01)  # Avoid division by zero

        exponent = numerator / denominator
        tc = (theta_d / 1.45) * torch.exp(exponent)

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

        Uses McMillan formula to predict expected Tc from Magpie features,
        then penalizes deviations from the predicted Tc.
        """
        # V12.22: Denormalize Tc correctly (handles log-transform)
        tc = _denormalize_tc(
            predicted_tc, tc_is_normalized, tc_mean, tc_std,
            self.config.tc_log_transform,
        )

        # Predict BCS parameters from Magpie features
        # Debye temp: output is in units of 100K, typical range 100-500K
        theta_d = self.debye_predictor(magpie_features).squeeze(-1) * 100 + 100

        # Lambda: output sigmoid scaled to typical range 0.2-1.0
        lambda_ep = self.lambda_predictor(magpie_features).squeeze(-1) * 0.8 + 0.2

        # Compute expected Tc from McMillan formula
        tc_mcmillan = self.mcmillan_tc(theta_d, lambda_ep)

        # BCS constraint 1: Predicted Tc should be consistent with McMillan
        # Use relative error to handle wide Tc range
        tc_safe = tc.clamp(min=0.1)
        tc_mcmillan_safe = tc_mcmillan.clamp(min=0.1)
        relative_error = (tc_safe - tc_mcmillan_safe).abs() / tc_mcmillan_safe

        # V12.22: BCS constraint 2: Soft quadratic penalty (replaces hard cap)
        # Model is FREE to predict Tc > 40K but gets a massive error signal.
        # BCS at 100K: softplus(60, beta=0.5)^2 ≈ 3600 (vs old F.relu penalty of 60)
        deviation = tc - self.config.bcs_max_tc
        soft_violation = F.softplus(deviation, beta=0.5)
        tc_upper_penalty = soft_violation ** 2

        # Combined loss
        if self.config.use_soft_constraints:
            mcmillan_loss = F.huber_loss(
                relative_error,
                torch.zeros_like(relative_error),
                delta=0.5  # Allow up to 50% relative error without strong penalty
            )
        else:
            mcmillan_loss = relative_error.mean()

        loss = mcmillan_loss + tc_upper_penalty.mean()

        return {
            'bcs_loss': loss * self.config.theory_weight,
            'theta_d': theta_d.mean(),
            'lambda_ep': lambda_ep.mean(),
            'tc_mcmillan': tc_mcmillan.mean(),
            'tc_upper_violation': tc_upper_penalty.mean(),
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
    Reference: Presland et al., Physica C 176 (1991) 95-105

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
    3. Tc typically < 60K

    V12.22: Soft quadratic penalty replaces hard cap.
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

        Currently implements soft Tc upper bound.
        TODO: Add pnictogen height relationship.
        """
        # V12.22: Denormalize Tc correctly (handles log-transform)
        tc = _denormalize_tc(
            predicted_tc, tc_is_normalized, tc_mean, tc_std,
            self.config.tc_log_transform,
        )

        # V12.22: Soft quadratic penalty (replaces hard cap F.relu)
        # Iron-based at 100K: softplus(40, beta=0.5)^2 ≈ 1600
        deviation = tc - self.config.iron_max_tc
        soft_violation = F.softplus(deviation, beta=0.5)
        tc_upper_penalty = soft_violation ** 2

        return {
            'iron_based_loss': tc_upper_penalty.mean() * self.config.theory_weight,
            'tc_upper_violation': tc_upper_penalty.mean(),
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
            SuperconductorFamily.HEAVY_FERMION: self.unknown_loss,  # Complex physics
            SuperconductorFamily.ORGANIC: self.unknown_loss,  # Complex physics
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
            'bcs': 0, 'cuprate': 0, 'iron': 0, 'unknown': 0
        }
        family_losses = {
            'bcs': torch.tensor(0.0, device=device),
            'cuprate': torch.tensor(0.0, device=device),
            'iron': torch.tensor(0.0, device=device),
        }

        # Group samples by family for efficient computation
        bcs_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        cuprate_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        iron_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

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

        # Note: Unknown materials contribute 0 to loss (by design)

        # Normalize by batch size
        total_loss = total_loss / max(batch_size, 1)

        return {
            'total': total_loss,
            'bcs_loss': family_losses['bcs'],
            'cuprate_loss': family_losses['cuprate'],
            'iron_loss': family_losses['iron'],
            'bcs_count': family_counts['bcs'],
            'cuprate_count': family_counts['cuprate'],
            'iron_count': family_counts['iron'],
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
