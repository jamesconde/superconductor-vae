"""
End-to-end superconductor discovery pipeline.

Orchestrates the full workflow:
1. Train model on known superconductors
2. Analyze latent space
3. Generate candidates
4. Validate candidates
5. Rank and output results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
from datetime import datetime

from ..models.bidirectional_vae import (
    BidirectionalVAE,
    BidirectionalVAELoss,
    create_bidirectional_vae
)
from ..data.dataset import SuperconductorDataset, ContrastiveDataset
from ..encoders.composition_encoder import CompositionDecoder
from ..validation.candidate_validator import CandidateValidator
from .latent_analyzer import LatentSpaceAnalyzer
from .candidate_generator import CandidateGenerator, GeneratedCandidate


@dataclass
class DiscoveryConfig:
    """Configuration for discovery pipeline."""
    # Model architecture
    latent_dim: int = 64
    architecture: str = 'medium'

    # Training
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20

    # Loss weights
    prediction_weight: float = 1.0
    reconstruction_weight: float = 0.1
    kl_weight: float = 0.01
    competence_weight: float = 0.1
    contrastive_weight: float = 0.1

    # Generation
    n_optimization_candidates: int = 100
    n_cluster_candidates: int = 50
    min_tc_threshold: float = 50.0  # Kelvin
    min_confidence: float = 0.5

    # Output
    output_dir: str = 'discovery_results'
    save_model: bool = True
    save_visualizations: bool = True


class SuperconductorDiscoveryPipeline:
    """
    Complete pipeline for superconductor discovery.

    Example:
        # Load data
        dataset = SuperconductorDataset.from_supercon('supercon.csv')

        # Configure and run
        config = DiscoveryConfig(epochs=100, latent_dim=64)
        pipeline = SuperconductorDiscoveryPipeline(dataset, config)

        # Train
        pipeline.train()

        # Discover
        candidates = pipeline.discover()

        # Save results
        pipeline.save_results(candidates)
    """

    def __init__(
        self,
        dataset: SuperconductorDataset,
        config: Optional[DiscoveryConfig] = None,
        negative_dataset: Optional[ContrastiveDataset] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize pipeline.

        Args:
            dataset: SuperconductorDataset with training data
            config: Pipeline configuration
            negative_dataset: Optional contrastive dataset with negatives
            device: Torch device
        """
        self.dataset = dataset
        self.config = config or DiscoveryConfig()
        self.negative_dataset = negative_dataset
        self.device = device

        # Initialize components
        self.model: Optional[BidirectionalVAE] = None
        self.analyzer: Optional[LatentSpaceAnalyzer] = None
        self.generator: Optional[CandidateGenerator] = None
        self.decoder = CompositionDecoder()
        self.validator = CandidateValidator()

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
        }

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_model(self) -> BidirectionalVAE:
        """Create model based on config."""
        return create_bidirectional_vae(
            input_dim=self.dataset.feature_dim,
            latent_dim=self.config.latent_dim,
            architecture=self.config.architecture
        )

    def train(
        self,
        train_dataset: Optional[SuperconductorDataset] = None,
        val_dataset: Optional[SuperconductorDataset] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset (uses split if not provided)
            val_dataset: Validation dataset
            verbose: Whether to print progress

        Returns:
            Training history
        """
        # Split if needed
        if train_dataset is None:
            train_dataset, val_dataset, _ = self.dataset.split(
                train_ratio=0.8, val_ratio=0.1
            )

        # Create model
        self.model = self._create_model()
        self.model.to(self.device)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # Loss and optimizer
        loss_fn = BidirectionalVAELoss(
            prediction_weight=self.config.prediction_weight,
            reconstruction_weight=self.config.reconstruction_weight,
            kl_weight=self.config.kl_weight,
            competence_weight=self.config.competence_weight,
            contrastive_weight=self.config.contrastive_weight,
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_losses = []
            train_maes = []

            for features, tc in train_loader:
                features = features.to(self.device)
                tc = tc.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features, return_all=True)

                # Get negative samples if available
                negative_z = None
                if self.negative_dataset is not None:
                    batch = self.negative_dataset.get_contrastive_batch(len(features))
                    if 'negative_features' in batch:
                        neg_features = batch['negative_features'].to(self.device)
                        negative_z = self.model.encode(neg_features)

                losses = loss_fn(outputs, features, tc, negative_z)
                losses['total'].backward()
                optimizer.step()

                train_losses.append(losses['total'].item())
                train_maes.append(
                    (outputs['tc_pred'] - tc).abs().mean().item()
                )

            # Validate
            self.model.eval()
            val_losses = []
            val_maes = []

            with torch.no_grad():
                for features, tc in val_loader:
                    features = features.to(self.device)
                    tc = tc.to(self.device)

                    outputs = self.model(features, return_all=True)
                    losses = loss_fn(outputs, features, tc)

                    val_losses.append(losses['total'].item())
                    val_maes.append(
                        (outputs['tc_pred'] - tc).abs().mean().item()
                    )

            # Record history
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_mae = np.mean(train_maes)
            val_mae = np.mean(val_maes)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.config.save_model:
                    self._save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}, "
                      f"Val MAE = {val_mae:.2f}")

        # Load best model
        if self.config.save_model and (self.output_dir / 'best_model.pt').exists():
            self._load_checkpoint('best_model.pt')

        if verbose:
            print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

        return self.history

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
        }, self.output_dir / filename)

    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.output_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)

    def setup_discovery(self):
        """Initialize components for discovery phase."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        # Create analyzer
        self.analyzer = LatentSpaceAnalyzer(
            self.model,
            self.dataset,
            device=self.device
        )
        self.analyzer.compute_latent_embeddings()

        # Create generator
        self.generator = CandidateGenerator(
            self.model,
            self.analyzer,
            self.decoder,
            self.validator,
            device=self.device
        )

    def discover(
        self,
        n_candidates: Optional[int] = None,
        min_tc: Optional[float] = None,
        min_confidence: Optional[float] = None,
        methods: List[str] = ['optimization', 'clustering', 'mutation'],
        validate: bool = True
    ) -> List[GeneratedCandidate]:
        """
        Generate and validate superconductor candidates.

        Args:
            n_candidates: Number of candidates to generate
            min_tc: Minimum predicted Tc threshold
            min_confidence: Minimum confidence threshold
            methods: Generation methods to use
            validate: Whether to validate candidates

        Returns:
            List of GeneratedCandidate objects
        """
        if self.generator is None:
            self.setup_discovery()

        n_candidates = n_candidates or self.config.n_optimization_candidates
        min_tc = min_tc or self.config.min_tc_threshold
        min_confidence = min_confidence or self.config.min_confidence

        print("=" * 60)
        print("SUPERCONDUCTOR DISCOVERY PIPELINE")
        print("=" * 60)

        all_candidates = []

        # Generate using different methods
        if 'optimization' in methods:
            print("\n[1/3] Generating via latent optimization...")
            opt_candidates = self.generator.optimize_latent_for_high_tc(
                n_candidates=n_candidates,
                validate=validate
            )
            all_candidates.extend(opt_candidates)
            print(f"  Generated {len(opt_candidates)} candidates")

        if 'clustering' in methods:
            print("\n[2/3] Generating via cluster sampling...")
            cluster_candidates = self.generator.sample_from_clusters(
                n_per_cluster=n_candidates // 5,
                tc_threshold=min_tc,
                validate=validate
            )
            all_candidates.extend(cluster_candidates)
            print(f"  Generated {len(cluster_candidates)} candidates")

        if 'mutation' in methods:
            print("\n[3/3] Generating via material mutation...")
            # Mutate top-Tc materials
            top_indices = np.argsort(self.analyzer.tc_values)[-5:]
            for idx in top_indices:
                mutations = self.generator.mutate_material(
                    material_idx=idx,
                    n_mutations=n_candidates // 10,
                    validate=validate
                )
                all_candidates.extend(mutations)
            print(f"  Generated {n_candidates // 2} mutation candidates")

        # Filter and deduplicate
        print("\nFiltering candidates...")

        # Filter by thresholds
        filtered = [
            c for c in all_candidates
            if c.predicted_tc >= min_tc
            and c.confidence >= min_confidence
            and c.formula
        ]

        # Keep only valid ones if validation was done
        if validate:
            filtered = [c for c in filtered if c.is_valid]

        # Deduplicate by formula
        seen = set()
        unique = []
        for c in filtered:
            if c.formula not in seen:
                seen.add(c.formula)
                unique.append(c)
        filtered = unique

        # Sort by predicted Tc
        filtered.sort(key=lambda c: c.predicted_tc, reverse=True)

        print(f"Valid candidates: {len(filtered)} / {len(all_candidates)}")

        return filtered

    def visualize_results(
        self,
        candidates: List[GeneratedCandidate],
        show: bool = True
    ):
        """Visualize latent space with generated candidates highlighted."""
        if self.analyzer is None:
            self.setup_discovery()

        # Get candidate latents
        candidate_latents = np.array([c.latent for c in candidates[:20]])

        # Find nearest neighbors in training data for comparison
        neighbors, _ = self.analyzer.find_nearest_neighbors(
            candidate_latents[0],
            k=5
        )

        # Visualize
        if self.config.save_visualizations:
            output_path = self.output_dir / 'latent_space.png'
        else:
            output_path = None

        # Add candidate points to visualization
        self.analyzer.visualize_latent_space(
            color_by='tc',
            method='umap',
            output_path=str(output_path) if output_path else None,
            show=show,
            highlight_indices=list(neighbors)
        )

    def save_results(
        self,
        candidates: List[GeneratedCandidate],
        filename: str = 'candidates.csv'
    ) -> pd.DataFrame:
        """
        Save discovery results to CSV.

        Args:
            candidates: List of candidates
            filename: Output filename

        Returns:
            Results DataFrame
        """
        # Create DataFrame
        results = pd.DataFrame([
            {
                'formula': c.formula,
                'predicted_tc': c.predicted_tc,
                'confidence': c.confidence,
                'is_valid': c.is_valid,
                'generation_method': c.generation_method,
                'charge_balance': c.validation.charge_balance if c.validation else None,
                'likely_family': c.validation.likely_family if c.validation else None,
                'overall_score': c.validation.overall_score if c.validation else None,
            }
            for c in candidates
        ])

        # Sort by predicted Tc
        results = results.sort_values('predicted_tc', ascending=False)

        # Save
        output_path = self.output_dir / filename
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        # Also save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_candidates': len(candidates),
            'config': {
                'latent_dim': self.config.latent_dim,
                'architecture': self.config.architecture,
                'min_tc_threshold': self.config.min_tc_threshold,
            },
            'dataset_stats': self.dataset.get_tc_statistics(),
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return results

    def run(
        self,
        skip_training: bool = False,
        model_path: Optional[str] = None
    ) -> Tuple[List[GeneratedCandidate], pd.DataFrame]:
        """
        Run full discovery pipeline.

        Args:
            skip_training: Skip training (load existing model)
            model_path: Path to pre-trained model

        Returns:
            Tuple of (candidates, results_dataframe)
        """
        # Train or load model
        if skip_training and model_path:
            self.model = self._create_model()
            self._load_checkpoint(model_path)
        elif not skip_training:
            self.train()

        # Setup discovery
        self.setup_discovery()

        # Generate candidates
        candidates = self.discover()

        # Visualize
        if self.config.save_visualizations:
            self.visualize_results(candidates, show=False)

        # Save results
        results = self.save_results(candidates)

        # Print top candidates
        print("\n" + "=" * 60)
        print("TOP 10 CANDIDATES")
        print("=" * 60)
        print(results.head(10).to_string(index=False))

        return candidates, results
