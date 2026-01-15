"""
Candidate generation using AttentionBidirectionalVAE.

Generates new superconductor candidates by:
1. Optimizing in latent space for high Tc
2. Decoding latent vectors to element compositions
3. Using attention weights to guide element selection

The attention weights provide interpretability - we can see which
elements the model thinks are important for high-Tc materials.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..models.attention_vae import AttentionBidirectionalVAE
from ..encoders.isotope_encoder import IsotopeEncoder
from ..encoders.element_properties import ELEMENT_SYMBOLS
from ..validation.candidate_validator import CandidateValidator, ValidationResult
from ..validation.physics_validator import PhysicsValidator, PhysicsValidationResult


@dataclass
class AttentionCandidate:
    """A generated candidate with attention analysis."""
    formula: str
    predicted_tc: float
    confidence: float
    generation_method: str
    latent_vector: np.ndarray
    element_attention: Dict[str, float]  # Element -> attention weight
    element_contributions: Dict[str, float]  # Element -> Tc contribution
    validation: Optional[ValidationResult] = None
    physics_validation: Optional[PhysicsValidationResult] = None


class AttentionCandidateGenerator:
    """
    Generate superconductor candidates using AttentionBidirectionalVAE.

    Key features:
    - Latent space optimization for high Tc
    - Attention-guided element composition
    - Per-element contribution analysis
    """

    def __init__(
        self,
        model: AttentionBidirectionalVAE,
        dataset,  # AttentionSuperconductorDataset
        device: torch.device = None,
        validator: Optional[CandidateValidator] = None,
        physics_validator: Optional[PhysicsValidator] = None
    ):
        self.model = model
        self.dataset = dataset
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validator = validator or CandidateValidator()
        self.physics_validator = physics_validator or PhysicsValidator()
        self.encoder = IsotopeEncoder()

        self.model.to(self.device)
        self.model.eval()

        # Cache high-Tc latent vectors
        self._high_tc_latents = None
        self._high_tc_threshold = None

    def _compute_high_tc_latents(self, tc_threshold: float = 50.0):
        """Cache latent vectors for high-Tc materials."""
        if self._high_tc_latents is not None and self._high_tc_threshold == tc_threshold:
            return

        self._high_tc_threshold = tc_threshold

        # Find high-Tc samples
        high_tc_mask = self.dataset.tc_values_raw >= tc_threshold
        high_tc_indices = np.where(high_tc_mask)[0]

        if len(high_tc_indices) == 0:
            print(f"Warning: No samples above {tc_threshold}K")
            self._high_tc_latents = None
            return

        # Encode high-Tc samples
        latents = []
        with torch.no_grad():
            for idx in high_tc_indices:
                sample = self.dataset[idx]
                z = self.model.encode(
                    element_indices=sample['element_indices'].unsqueeze(0).to(self.device),
                    element_fractions=sample['element_fractions'].unsqueeze(0).to(self.device),
                    element_mask=sample['element_mask'].unsqueeze(0).to(self.device),
                    isotope_features=sample['isotope_features'].unsqueeze(0).to(self.device)
                )
                latents.append(z.cpu())

        self._high_tc_latents = torch.cat(latents, dim=0)
        print(f"Cached {len(self._high_tc_latents)} high-Tc latent vectors (>= {tc_threshold}K)")

    def optimize_latent_for_high_tc(
        self,
        n_candidates: int = 20,
        n_steps: int = 200,
        lr: float = 0.1,
        tc_threshold: float = 50.0,
        diversity_weight: float = 0.1
    ) -> List[AttentionCandidate]:
        """
        Optimize in latent space to find high-Tc compositions.

        Args:
            n_candidates: Number of candidates to generate
            n_steps: Optimization steps per candidate
            lr: Learning rate for optimization
            tc_threshold: Initialize near samples above this Tc
            diversity_weight: Encourage diverse candidates

        Returns:
            List of AttentionCandidate objects
        """
        self._compute_high_tc_latents(tc_threshold)

        if self._high_tc_latents is None:
            return []

        candidates = []
        generated_latents = []

        for i in range(n_candidates):
            # Initialize from random high-Tc sample (with noise)
            init_idx = np.random.randint(len(self._high_tc_latents))
            z = self._high_tc_latents[init_idx].clone().to(self.device)
            z = z + torch.randn_like(z) * 0.5  # Add noise for diversity
            z.requires_grad_(True)

            optimizer = torch.optim.Adam([z], lr=lr)

            for step in range(n_steps):
                optimizer.zero_grad()

                # Predict Tc from latent
                tc_pred = self.model.predict_tc_from_latent(z.unsqueeze(0))

                # Loss: maximize Tc (minimize negative Tc)
                loss = -tc_pred.squeeze()

                # Diversity: push away from previously generated
                if generated_latents and diversity_weight > 0:
                    prev_latents = torch.stack(generated_latents).to(self.device)
                    distances = torch.cdist(z.unsqueeze(0), prev_latents)
                    diversity_loss = -diversity_weight * distances.min()
                    loss = loss + diversity_loss

                loss.backward()
                optimizer.step()

            # Convert optimized latent to candidate
            candidate = self._latent_to_candidate(
                z.detach(),
                generation_method='latent_optimization'
            )
            if candidate is not None:
                candidates.append(candidate)
                generated_latents.append(z.detach().cpu())

        # Sort by predicted Tc
        candidates.sort(key=lambda c: c.predicted_tc, reverse=True)
        return candidates

    def sample_around_known(
        self,
        formula: str,
        n_samples: int = 10,
        noise_scale: float = 0.3
    ) -> List[AttentionCandidate]:
        """
        Generate candidates by sampling around a known superconductor.

        Args:
            formula: Known superconductor formula
            n_samples: Number of samples to generate
            noise_scale: Scale of noise in latent space

        Returns:
            List of AttentionCandidate objects
        """
        # Encode the known formula
        encoded = self.encoder.encode_batch([formula])

        with torch.no_grad():
            z_base = self.model.encode(
                element_indices=encoded['element_indices'].to(self.device),
                element_fractions=encoded['element_fractions'].to(self.device),
                element_mask=encoded['element_mask'].to(self.device),
                isotope_features=encoded['isotope_features'].to(self.device)
            )

        candidates = []
        for _ in range(n_samples):
            # Add noise
            z = z_base + torch.randn_like(z_base) * noise_scale

            candidate = self._latent_to_candidate(
                z.squeeze(0),
                generation_method=f'sample_around_{formula}'
            )
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda c: c.predicted_tc, reverse=True)
        return candidates

    def interpolate_between(
        self,
        formula1: str,
        formula2: str,
        n_steps: int = 10
    ) -> List[AttentionCandidate]:
        """
        Generate candidates by interpolating between two superconductors.

        Args:
            formula1: First superconductor formula
            formula2: Second superconductor formula
            n_steps: Number of interpolation steps

        Returns:
            List of AttentionCandidate objects
        """
        # Encode both formulas
        encoded1 = self.encoder.encode_batch([formula1])
        encoded2 = self.encoder.encode_batch([formula2])

        with torch.no_grad():
            z1 = self.model.encode(
                element_indices=encoded1['element_indices'].to(self.device),
                element_fractions=encoded1['element_fractions'].to(self.device),
                element_mask=encoded1['element_mask'].to(self.device),
                isotope_features=encoded1['isotope_features'].to(self.device)
            )
            z2 = self.model.encode(
                element_indices=encoded2['element_indices'].to(self.device),
                element_fractions=encoded2['element_fractions'].to(self.device),
                element_mask=encoded2['element_mask'].to(self.device),
                isotope_features=encoded2['isotope_features'].to(self.device)
            )

        candidates = []
        for i in range(n_steps):
            t = i / (n_steps - 1)
            z = (1 - t) * z1 + t * z2

            candidate = self._latent_to_candidate(
                z.squeeze(0),
                generation_method=f'interpolate_{formula1}_to_{formula2}'
            )
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def _latent_to_candidate(
        self,
        z: torch.Tensor,
        generation_method: str
    ) -> Optional[AttentionCandidate]:
        """
        Convert latent vector to candidate formula.

        This is the key challenge - we need to decode latent to elements.
        We use a compositional approach based on the learned representations.
        """
        with torch.no_grad():
            # Predict Tc
            tc_pred = self.model.predict_tc_from_latent(z.unsqueeze(0))
            tc_normalized = tc_pred.item()
            tc_kelvin = self.dataset.denormalize_tc(torch.tensor(tc_normalized)).item()

            # Get competence (confidence)
            competence = self.model.competence_head(z.unsqueeze(0)).item()

        # Decode latent to composition
        # This is approximate - we find nearest known compositions
        formula, attention, contributions, stoichiometry = self._decode_to_formula(z)

        if formula is None:
            return None

        # Validate with family-based validator (less important)
        validation = self.validator.validate(formula)

        # Physics-based validation (PRIMARY check for plausibility)
        physics_validation = self.physics_validator.validate(formula, stoichiometry)

        return AttentionCandidate(
            formula=formula,
            predicted_tc=tc_kelvin,
            confidence=competence,
            generation_method=generation_method,
            latent_vector=z.cpu().numpy(),
            element_attention=attention,
            element_contributions=contributions,
            validation=validation,
            physics_validation=physics_validation
        )

    def _decode_to_formula(
        self,
        z: torch.Tensor
    ) -> Tuple[Optional[str], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Decode latent vector to formula using nearest neighbor approach.

        Since we can't directly decode to discrete compositions, we:
        1. Find nearest training samples in latent space
        2. Combine their element distributions weighted by distance
        3. Round to nearest integer stoichiometry

        Returns:
            (formula, attention_weights, contributions, stoichiometry)
        """
        # Encode all training samples (cached)
        if not hasattr(self, '_all_latents'):
            self._cache_all_latents()

        # Find k-nearest neighbors
        k = 5
        distances = torch.cdist(z.unsqueeze(0).to(self.device), self._all_latents.to(self.device))
        _, nearest_idx = distances.topk(k, largest=False)
        nearest_idx = nearest_idx.squeeze().cpu().numpy()

        # Weight by inverse distance
        weights = 1.0 / (distances.squeeze()[nearest_idx].cpu().numpy() + 1e-6)
        weights = weights / weights.sum()

        # Combine element distributions
        element_counts = {}
        for idx, weight in zip(nearest_idx, weights):
            sample = self.dataset[idx]
            indices = sample['element_indices'].numpy()
            fractions = sample['element_fractions'].numpy()
            mask = sample['element_mask'].numpy()

            for i in range(len(indices)):
                if mask[i]:
                    elem_idx = indices[i]
                    if 1 <= elem_idx <= len(ELEMENT_SYMBOLS):
                        symbol = ELEMENT_SYMBOLS[elem_idx]
                        frac = fractions[i]
                        element_counts[symbol] = element_counts.get(symbol, 0) + weight * frac

        # Filter to significant elements and normalize
        significant = {k: v for k, v in element_counts.items() if v > 0.05}
        if not significant:
            return None, {}, {}, {}

        total = sum(significant.values())
        normalized = {k: v / total for k, v in significant.items()}

        # Convert to stoichiometry (round to nearest integer * scale)
        scale = 10  # Scale factor for rounding
        stoich = {}
        for elem, frac in normalized.items():
            count = round(frac * scale)
            if count > 0:
                stoich[elem] = count

        # Simplify (divide by GCD)
        if stoich:
            from math import gcd
            from functools import reduce
            common = reduce(gcd, stoich.values())
            stoich = {k: v // common for k, v in stoich.items()}

        # Build formula string
        formula_parts = []
        for elem in sorted(stoich.keys()):
            count = stoich[elem]
            if count == 1:
                formula_parts.append(elem)
            else:
                formula_parts.append(f"{elem}{count}")
        formula = ''.join(formula_parts)

        # Get attention/contributions from nearest sample
        # Note: ELEMENT_SYMBOLS[0] is empty, index = atomic number, so use i directly
        attention = {ELEMENT_SYMBOLS[i]: float(v) for i, v in
                    zip(sample['element_indices'].numpy(), weights) if 1 <= i < len(ELEMENT_SYMBOLS)}
        contributions = {}  # Would need forward pass to get actual contributions

        # Convert stoich to float for compatibility with physics validator
        stoichiometry = {k: float(v) for k, v in stoich.items()}

        return formula, attention, contributions, stoichiometry

    def _cache_all_latents(self):
        """Cache latent vectors for all training samples."""
        latents = []
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                z = self.model.encode(
                    element_indices=sample['element_indices'].unsqueeze(0).to(self.device),
                    element_fractions=sample['element_fractions'].unsqueeze(0).to(self.device),
                    element_mask=sample['element_mask'].unsqueeze(0).to(self.device),
                    isotope_features=sample['isotope_features'].unsqueeze(0).to(self.device)
                )
                latents.append(z.cpu())
        self._all_latents = torch.cat(latents, dim=0)
        print(f"Cached {len(self._all_latents)} latent vectors for decoding")

    def analyze_element_importance(
        self,
        formulas: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze which elements are important for each formula.

        Args:
            formulas: List of chemical formulas

        Returns:
            Dict mapping formula to element attention weights
        """
        results = {}
        encoded = self.encoder.encode_batch(formulas)

        with torch.no_grad():
            outputs = self.model(
                element_indices=encoded['element_indices'].to(self.device),
                element_fractions=encoded['element_fractions'].to(self.device),
                element_mask=encoded['element_mask'].to(self.device),
                isotope_features=encoded['isotope_features'].to(self.device),
                return_all=True
            )

        for i, formula in enumerate(formulas):
            attention = outputs['attention_weights'][i].cpu().numpy()
            indices = encoded['element_indices'][i].numpy()
            mask = encoded['element_mask'][i].numpy()

            element_weights = {}
            for j in range(len(indices)):
                if mask[j]:
                    elem_idx = indices[j]
                    if 1 <= elem_idx <= len(ELEMENT_SYMBOLS):
                        symbol = ELEMENT_SYMBOLS[elem_idx]
                        element_weights[symbol] = float(attention[j])

            results[formula] = element_weights

        return results
