"""
Candidate generator for superconductor discovery.

Generates new superconductor candidates from latent space using:
1. Latent optimization (gradient ascent on Tc)
2. Cluster sampling (sample near high-Tc regions)
3. Material interpolation (explore between known materials)
4. Genetic/evolutionary sampling
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

from .latent_analyzer import LatentSpaceAnalyzer, ClusterInfo
from ..encoders.composition_encoder import CompositionDecoder
from ..validation.candidate_validator import CandidateValidator, ValidationResult


@dataclass
class GeneratedCandidate:
    """Container for a generated superconductor candidate."""
    latent: np.ndarray
    features: np.ndarray
    formula: str
    predicted_tc: float
    confidence: float

    # Generation metadata
    generation_method: str
    generation_step: int = 0

    # Validation
    validation: Optional[ValidationResult] = None
    is_valid: bool = False

    # Source information (for interpolation/mutation)
    parent_formulas: List[str] = field(default_factory=list)
    alpha: Optional[float] = None  # Interpolation parameter


class CandidateGenerator:
    """
    Generate new superconductor candidates from latent space.

    Example:
        generator = CandidateGenerator(model, analyzer)

        # Optimize for high Tc
        candidates = generator.optimize_latent_for_high_tc(n_candidates=100)

        # Sample from high-Tc clusters
        candidates = generator.sample_from_clusters(n_per_cluster=20)

        # Interpolate between materials
        candidates = generator.interpolate_materials(idx1=0, idx2=10)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        analyzer: LatentSpaceAnalyzer,
        decoder: Optional[CompositionDecoder] = None,
        validator: Optional[CandidateValidator] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize generator.

        Args:
            model: Trained BidirectionalVAE model
            analyzer: LatentSpaceAnalyzer with computed embeddings
            decoder: Feature-to-formula decoder
            validator: Candidate validator
            device: Torch device
        """
        self.model = model
        self.analyzer = analyzer
        self.decoder = decoder or CompositionDecoder()
        self.validator = validator or CandidateValidator()
        self.device = device

        self.model.to(device)
        self.model.eval()

    def optimize_latent_for_high_tc(
        self,
        n_candidates: int = 100,
        n_steps: int = 200,
        lr: float = 0.1,
        regularization: float = 0.01,
        init_near_clusters: bool = True,
        tc_threshold: float = 50.0,
        validate: bool = True
    ) -> List[GeneratedCandidate]:
        """
        Optimize latent vectors to maximize predicted Tc.

        Uses gradient ascent on Tc prediction with regularization
        to stay in valid latent regions.

        Args:
            n_candidates: Number of candidates to generate
            n_steps: Optimization steps
            lr: Learning rate
            regularization: Regularization weight (penalize far from training distribution)
            init_near_clusters: Initialize near high-Tc clusters
            tc_threshold: Tc threshold for cluster initialization
            validate: Whether to validate candidates

        Returns:
            List of GeneratedCandidate objects sorted by predicted Tc
        """
        # Initialize latent vectors
        if init_near_clusters:
            clusters = self.analyzer.cluster_high_tc_regions(tc_threshold=tc_threshold)
            centroids = torch.tensor(
                np.array([c.centroid for c in clusters]),
                dtype=torch.float32,
                device=self.device
            )

            # Sample around centroids
            z_list = []
            per_cluster = n_candidates // len(centroids) + 1
            for centroid in centroids:
                noise = torch.randn(per_cluster, centroid.shape[0], device=self.device) * 0.5
                z_list.append(centroid + noise)

            z = torch.cat(z_list, dim=0)[:n_candidates]
        else:
            # Random initialization
            latent_dim = self.model.latent_dim
            z = torch.randn(n_candidates, latent_dim, device=self.device) * 0.5

        z.requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)

        # Compute mean latent for regularization
        if self.analyzer.latent_cache is not None:
            mean_latent = torch.tensor(
                self.analyzer.latent_cache.mean(axis=0),
                dtype=torch.float32,
                device=self.device
            )
        else:
            mean_latent = torch.zeros(z.shape[1], device=self.device)

        # Optimization loop
        for step in range(n_steps):
            optimizer.zero_grad()

            # Predict Tc
            tc_pred = self.model.predict_from_latent(z)

            # Regularization: stay near training distribution
            latent_dist = ((z - mean_latent) ** 2).mean(dim=1).mean()

            # Loss: minimize negative Tc + regularization
            loss = -tc_pred.mean() + regularization * latent_dist

            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step {step}: Mean Tc = {tc_pred.mean().item():.1f}K, "
                      f"Max Tc = {tc_pred.max().item():.1f}K")

        # Decode candidates
        with torch.no_grad():
            features = self.model.decode(z)
            tc_final = self.model.predict_from_latent(z)
            competence = self.model.competence_from_latent(z)

        # Create candidates
        candidates = []
        for i in range(len(z)):
            feat_np = features[i].cpu().numpy()

            # Decode to formula (only use composition part)
            comp_dim = self.analyzer.dataset.composition_dim
            composition_vector = feat_np[:comp_dim]
            formula = self.decoder.decode(composition_vector)

            candidate = GeneratedCandidate(
                latent=z[i].detach().cpu().numpy(),
                features=feat_np,
                formula=formula,
                predicted_tc=tc_final[i].item(),
                confidence=competence[i].item(),
                generation_method='latent_optimization',
                generation_step=n_steps
            )

            if validate and formula:
                candidate.validation = self.validator.validate(formula)
                candidate.is_valid = candidate.validation.is_valid

            candidates.append(candidate)

        # Sort by predicted Tc
        candidates.sort(key=lambda c: c.predicted_tc, reverse=True)

        return candidates

    def sample_from_clusters(
        self,
        n_per_cluster: int = 20,
        tc_threshold: float = 50.0,
        noise_scale: float = 0.3,
        n_clusters: int = 5,
        validate: bool = True
    ) -> List[GeneratedCandidate]:
        """
        Sample new candidates from high-Tc cluster regions.

        Args:
            n_per_cluster: Samples per cluster
            tc_threshold: Tc threshold for clusters
            noise_scale: Scale of Gaussian noise around centroids
            n_clusters: Number of clusters to sample from
            validate: Whether to validate candidates

        Returns:
            List of GeneratedCandidate objects
        """
        clusters = self.analyzer.cluster_high_tc_regions(
            tc_threshold=tc_threshold,
            n_clusters=n_clusters
        )

        candidates = []

        for cluster in clusters:
            centroid = torch.tensor(
                cluster.centroid,
                dtype=torch.float32,
                device=self.device
            )

            # Sample around centroid
            samples = centroid + torch.randn(
                n_per_cluster,
                len(centroid),
                device=self.device
            ) * noise_scale

            with torch.no_grad():
                features = self.model.decode(samples)
                tc_pred = self.model.predict_from_latent(samples)
                competence = self.model.competence_from_latent(samples)

            for i in range(n_per_cluster):
                feat_np = features[i].cpu().numpy()
                comp_dim = self.analyzer.dataset.composition_dim
                formula = self.decoder.decode(feat_np[:comp_dim])

                candidate = GeneratedCandidate(
                    latent=samples[i].cpu().numpy(),
                    features=feat_np,
                    formula=formula,
                    predicted_tc=tc_pred[i].item(),
                    confidence=competence[i].item(),
                    generation_method='cluster_sampling',
                    parent_formulas=cluster.formulas[:3]  # Top formulas from cluster
                )

                if validate and formula:
                    candidate.validation = self.validator.validate(formula)
                    candidate.is_valid = candidate.validation.is_valid

                candidates.append(candidate)

        # Sort by predicted Tc
        candidates.sort(key=lambda c: c.predicted_tc, reverse=True)

        return candidates

    def interpolate_materials(
        self,
        material1_idx: int,
        material2_idx: int,
        n_points: int = 10,
        validate: bool = True
    ) -> List[GeneratedCandidate]:
        """
        Interpolate between two known materials in latent space.

        Useful for exploring material families.

        Args:
            material1_idx: Index of first material in dataset
            material2_idx: Index of second material in dataset
            n_points: Number of interpolation points
            validate: Whether to validate candidates

        Returns:
            List of GeneratedCandidate objects
        """
        dataset = self.analyzer.dataset

        # Get features
        x1 = dataset.features_tensor[material1_idx:material1_idx+1].to(self.device)
        x2 = dataset.features_tensor[material2_idx:material2_idx+1].to(self.device)

        # Get parent formulas
        formula1 = dataset.formulas[material1_idx]
        formula2 = dataset.formulas[material2_idx]

        with torch.no_grad():
            z1 = self.model.encode(x1, deterministic=True)
            z2 = self.model.encode(x2, deterministic=True)

        # Interpolate
        alphas = torch.linspace(0, 1, n_points, device=self.device)
        candidates = []

        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2

            with torch.no_grad():
                features = self.model.decode(z)
                tc_pred = self.model.predict_from_latent(z)
                competence = self.model.competence_from_latent(z)

            feat_np = features[0].cpu().numpy()
            comp_dim = dataset.composition_dim
            formula = self.decoder.decode(feat_np[:comp_dim])

            candidate = GeneratedCandidate(
                latent=z[0].cpu().numpy(),
                features=feat_np,
                formula=formula,
                predicted_tc=tc_pred[0].item(),
                confidence=competence[0].item(),
                generation_method='interpolation',
                alpha=alpha.item(),
                parent_formulas=[formula1, formula2]
            )

            if validate and formula:
                candidate.validation = self.validator.validate(formula)
                candidate.is_valid = candidate.validation.is_valid

            candidates.append(candidate)

        return candidates

    def mutate_material(
        self,
        material_idx: int,
        n_mutations: int = 20,
        mutation_scale: float = 0.3,
        validate: bool = True
    ) -> List[GeneratedCandidate]:
        """
        Generate variations of a known material.

        Args:
            material_idx: Index of material to mutate
            n_mutations: Number of mutations to generate
            mutation_scale: Scale of random perturbation
            validate: Whether to validate candidates

        Returns:
            List of GeneratedCandidate objects
        """
        dataset = self.analyzer.dataset

        x = dataset.features_tensor[material_idx:material_idx+1].to(self.device)
        parent_formula = dataset.formulas[material_idx]

        with torch.no_grad():
            z_original = self.model.encode(x, deterministic=True)

        candidates = []

        # Generate mutations
        noise = torch.randn(n_mutations, z_original.shape[1], device=self.device) * mutation_scale
        z_mutated = z_original + noise

        with torch.no_grad():
            features = self.model.decode(z_mutated)
            tc_pred = self.model.predict_from_latent(z_mutated)
            competence = self.model.competence_from_latent(z_mutated)

        for i in range(n_mutations):
            feat_np = features[i].cpu().numpy()
            comp_dim = dataset.composition_dim
            formula = self.decoder.decode(feat_np[:comp_dim])

            candidate = GeneratedCandidate(
                latent=z_mutated[i].cpu().numpy(),
                features=feat_np,
                formula=formula,
                predicted_tc=tc_pred[i].item(),
                confidence=competence[i].item(),
                generation_method='mutation',
                parent_formulas=[parent_formula]
            )

            if validate and formula:
                candidate.validation = self.validator.validate(formula)
                candidate.is_valid = candidate.validation.is_valid

            candidates.append(candidate)

        # Sort by predicted Tc
        candidates.sort(key=lambda c: c.predicted_tc, reverse=True)

        return candidates

    def generate_diverse_candidates(
        self,
        n_candidates: int = 100,
        min_tc: float = 50.0,
        min_confidence: float = 0.3,
        validate: bool = True,
        unique_formulas: bool = True
    ) -> List[GeneratedCandidate]:
        """
        Generate diverse candidates using multiple methods.

        Combines optimization, cluster sampling, and mutation
        for a diverse set of candidates.

        Args:
            n_candidates: Target number of candidates
            min_tc: Minimum predicted Tc
            min_confidence: Minimum confidence score
            validate: Whether to validate candidates
            unique_formulas: Whether to deduplicate by formula

        Returns:
            List of GeneratedCandidate objects
        """
        all_candidates = []

        # 1. Latent optimization (40%)
        n_opt = int(n_candidates * 0.4)
        opt_candidates = self.optimize_latent_for_high_tc(
            n_candidates=n_opt,
            validate=validate
        )
        all_candidates.extend(opt_candidates)
        print(f"Generated {len(opt_candidates)} via optimization")

        # 2. Cluster sampling (40%)
        n_cluster = int(n_candidates * 0.4)
        n_per_cluster = n_cluster // 5
        cluster_candidates = self.sample_from_clusters(
            n_per_cluster=n_per_cluster,
            tc_threshold=min_tc,
            validate=validate
        )
        all_candidates.extend(cluster_candidates)
        print(f"Generated {len(cluster_candidates)} via cluster sampling")

        # 3. Mutations of top materials (20%)
        n_mutations = int(n_candidates * 0.2)
        top_tc_indices = np.argsort(self.analyzer.tc_values)[-5:]  # Top 5 Tc materials
        mutations_per_material = n_mutations // len(top_tc_indices)

        for idx in top_tc_indices:
            mutations = self.mutate_material(
                material_idx=idx,
                n_mutations=mutations_per_material,
                validate=validate
            )
            all_candidates.extend(mutations)

        print(f"Total candidates before filtering: {len(all_candidates)}")

        # Filter
        filtered = [
            c for c in all_candidates
            if c.predicted_tc >= min_tc
            and c.confidence >= min_confidence
            and c.formula  # Has valid formula
        ]

        # Deduplicate by formula
        if unique_formulas:
            seen = set()
            unique = []
            for c in filtered:
                if c.formula not in seen:
                    seen.add(c.formula)
                    unique.append(c)
            filtered = unique

        # Sort by predicted Tc
        filtered.sort(key=lambda c: c.predicted_tc, reverse=True)

        print(f"Final candidates after filtering: {len(filtered)}")

        return filtered[:n_candidates]
