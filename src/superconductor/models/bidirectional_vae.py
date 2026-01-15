"""
Bidirectional VAE for superconductor Tc prediction and generation.

Architecture:
    Features → Encoder → Latent Space (z) → Predictor → Tc
                              ↓
                          Decoder → Reconstructed Features

The latent space enables:
1. Tc prediction from known materials
2. Generation of new candidate materials
3. Latent space optimization for high-Tc discovery
4. Contrastive learning with non-superconductors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


class BidirectionalEncoder(nn.Module):
    """
    Encodes features to latent space with uncertainty (VAE-style).

    Outputs mean and log-variance for reparameterization.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize encoder.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # VAE: mean and log-variance projections
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: [batch, input_dim] input features

        Returns:
            Tuple of (mu, logvar), each [batch, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Reparameterization trick for training.

        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            training: Whether in training mode (adds noise)

        Returns:
            Sampled latent vector
        """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Use mean for deterministic inference


class FeatureDecoder(nn.Module):
    """
    Decodes latent space back to feature space.

    Used for reconstruction loss and candidate generation.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 256],
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize decoder.

        Args:
            latent_dim: Latent space dimension
            output_dim: Output feature dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to feature space.

        Args:
            z: [batch, latent_dim] latent vectors

        Returns:
            [batch, output_dim] reconstructed features
        """
        return self.decoder(z)


class TcPredictor(nn.Module):
    """
    Predicts critical temperature (Tc) from latent space.

    Uses Student-t distribution for robust uncertainty estimation.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        use_uncertainty: bool = True
    ):
        """
        Initialize Tc predictor.

        Args:
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            use_uncertainty: Whether to output uncertainty estimates
        """
        super().__init__()

        self.use_uncertainty = use_uncertainty

        # Build predictor
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Output heads
        self.mean_head = nn.Linear(prev_dim, 1)

        if use_uncertainty:
            # Log-scale for positivity
            self.scale_head = nn.Linear(prev_dim, 1)
            # Degrees of freedom (for Student-t)
            self.df_head = nn.Linear(prev_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Predict Tc from latent.

        Args:
            z: [batch, latent_dim] latent vectors
            return_uncertainty: Whether to return uncertainty parameters

        Returns:
            If return_uncertainty:
                Tuple of (mean, scale, df)
            Else:
                [batch] predicted Tc values
        """
        h = self.features(z)
        mean = self.mean_head(h).squeeze(-1)

        if return_uncertainty and self.use_uncertainty:
            scale = F.softplus(self.scale_head(h)).squeeze(-1) + 1e-6
            df = F.softplus(self.df_head(h)).squeeze(-1) + 2.0  # df > 2 for finite variance
            return mean, scale, df

        return mean


class CompetenceHead(nn.Module):
    """
    Estimates model competence/confidence for a sample.

    High competence = model is confident in its prediction.
    Low competence = model is uncertain (should abstain/handoff).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 32
    ):
        """
        Initialize competence head.

        Args:
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Competence in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Estimate competence from latent.

        Args:
            z: [batch, latent_dim] latent vectors

        Returns:
            [batch] competence scores in [0, 1]
        """
        return self.head(z).squeeze(-1)


class BidirectionalVAE(nn.Module):
    """
    Complete Bidirectional VAE for superconductor prediction and generation.

    Supports:
    - Tc prediction with uncertainty
    - Feature reconstruction
    - Latent space encoding/decoding
    - Competence estimation
    - Contrastive learning
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        encoder_hidden: List[int] = [256, 128],
        decoder_hidden: List[int] = [128, 256],
        predictor_hidden: List[int] = [64, 32],
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_uncertainty: bool = True
    ):
        """
        Initialize Bidirectional VAE.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            encoder_hidden: Encoder hidden dimensions
            decoder_hidden: Decoder hidden dimensions
            predictor_hidden: Predictor hidden dimensions
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_uncertainty: Whether to use uncertainty estimation
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_uncertainty = use_uncertainty

        # Components
        self.encoder = BidirectionalEncoder(
            input_dim, latent_dim, encoder_hidden, dropout, use_batch_norm
        )
        self.decoder = FeatureDecoder(
            latent_dim, input_dim, decoder_hidden, dropout, use_batch_norm
        )
        self.predictor = TcPredictor(
            latent_dim, predictor_hidden, dropout, use_uncertainty
        )
        self.competence_head = CompetenceHead(latent_dim)

        # Projection head for contrastive learning
        self.contrastive_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: [batch, input_dim] input features
            return_all: Whether to return all intermediate values

        Returns:
            Dictionary with outputs:
            - tc_pred: Predicted Tc values
            - competence: Competence scores
            If return_all:
            - z: Latent vectors
            - mu: Latent mean
            - logvar: Latent log-variance
            - x_recon: Reconstructed features
            If use_uncertainty:
            - tc_scale: Prediction scale (uncertainty)
            - tc_df: Degrees of freedom
        """
        # Encode
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar, self.training)

        # Predict Tc
        if self.use_uncertainty:
            tc_pred, tc_scale, tc_df = self.predictor(z, return_uncertainty=True)
        else:
            tc_pred = self.predictor(z)

        # Competence
        competence = self.competence_head(z)

        # Build output
        output = {
            'tc_pred': tc_pred,
            'competence': competence,
        }

        if self.use_uncertainty:
            output['tc_scale'] = tc_scale
            output['tc_df'] = tc_df

        if return_all:
            # Reconstruct
            x_recon = self.decoder(z)

            output.update({
                'z': z,
                'mu': mu,
                'logvar': logvar,
                'x_recon': x_recon,
            })

        return output

    def encode(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Encode features to latent space.

        Args:
            x: [batch, input_dim] input features
            deterministic: If True, return mean (no sampling)

        Returns:
            [batch, latent_dim] latent vectors
        """
        mu, logvar = self.encoder(x)
        if deterministic:
            return mu
        return self.encoder.reparameterize(mu, logvar, training=True)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to feature space.

        Args:
            z: [batch, latent_dim] latent vectors

        Returns:
            [batch, input_dim] decoded features
        """
        return self.decoder(z)

    def predict_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict Tc directly from latent vectors.

        Args:
            z: [batch, latent_dim] latent vectors

        Returns:
            [batch] predicted Tc values
        """
        return self.predictor(z, return_uncertainty=False)

    def competence_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get competence from latent vectors.

        Args:
            z: [batch, latent_dim] latent vectors

        Returns:
            [batch] competence scores
        """
        return self.competence_head(z)

    def get_contrastive_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get normalized embedding for contrastive learning.

        Args:
            z: [batch, latent_dim] latent vectors

        Returns:
            [batch, latent_dim] L2-normalized embeddings
        """
        proj = self.contrastive_proj(z)
        return F.normalize(proj, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict Tc from features (convenience method).

        Args:
            x: [batch, input_dim] input features

        Returns:
            [batch] predicted Tc values
        """
        return self(x)['tc_pred']


class BidirectionalVAELoss(nn.Module):
    """
    Multi-objective loss for Bidirectional VAE training.

    Components:
    1. Prediction loss: MSE for Tc prediction
    2. Reconstruction loss: MSE for feature reconstruction
    3. KL divergence: Regularization for latent space
    4. Competence calibration: Train competence to reflect accuracy
    5. Contrastive loss: Separate superconductors from non-superconductors
    """

    def __init__(
        self,
        prediction_weight: float = 1.0,
        reconstruction_weight: float = 0.1,
        kl_weight: float = 0.01,
        competence_weight: float = 0.1,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.07
    ):
        """
        Initialize loss function.

        Args:
            prediction_weight: Weight for Tc prediction loss
            reconstruction_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence
            competence_weight: Weight for competence calibration
            contrastive_weight: Weight for contrastive loss
            contrastive_temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.w_pred = prediction_weight
        self.w_recon = reconstruction_weight
        self.w_kl = kl_weight
        self.w_comp = competence_weight
        self.w_contrast = contrastive_weight
        self.temperature = contrastive_temperature

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        x: torch.Tensor,
        tc_true: torch.Tensor,
        negative_z: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            outputs: Model outputs from forward pass with return_all=True
            x: [batch, input_dim] input features
            tc_true: [batch] true Tc values
            negative_z: Optional [batch_neg, latent_dim] negative sample latents

        Returns:
            Dictionary of losses
        """
        losses = {}

        # 1. Prediction loss
        tc_pred = outputs['tc_pred']
        losses['prediction'] = F.mse_loss(tc_pred, tc_true)

        # 2. Reconstruction loss
        if 'x_recon' in outputs:
            losses['reconstruction'] = F.mse_loss(outputs['x_recon'], x)
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=x.device)

        # 3. KL divergence
        if 'mu' in outputs and 'logvar' in outputs:
            mu = outputs['mu']
            logvar = outputs['logvar']
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            losses['kl'] = kl
        else:
            losses['kl'] = torch.tensor(0.0, device=x.device)

        # 4. Competence calibration
        if 'competence' in outputs:
            competence = outputs['competence']
            # Target competence: high when prediction is accurate
            with torch.no_grad():
                pred_error = (tc_pred - tc_true).abs()
                max_error = pred_error.max() + 1e-8
                target_competence = 1.0 - (pred_error / max_error)

            losses['competence'] = F.mse_loss(competence, target_competence)
        else:
            losses['competence'] = torch.tensor(0.0, device=x.device)

        # 5. Contrastive loss (if negative samples provided)
        if negative_z is not None and 'z' in outputs:
            z = outputs['z']
            losses['contrastive'] = self._contrastive_loss(z, negative_z, tc_true)
        else:
            losses['contrastive'] = torch.tensor(0.0, device=x.device)

        # Total loss
        losses['total'] = (
            self.w_pred * losses['prediction'] +
            self.w_recon * losses['reconstruction'] +
            self.w_kl * losses['kl'] +
            self.w_comp * losses['competence'] +
            self.w_contrast * losses['contrastive']
        )

        return losses

    def _contrastive_loss(
        self,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
        tc_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Superconductors with similar Tc should be close.
        Superconductors should be far from non-superconductors.

        Args:
            z_pos: [batch_pos, latent_dim] superconductor latents
            z_neg: [batch_neg, latent_dim] negative sample latents
            tc_values: [batch_pos] Tc values for weighting

        Returns:
            Contrastive loss value
        """
        # Normalize
        z_pos_norm = F.normalize(z_pos, dim=-1)
        z_neg_norm = F.normalize(z_neg, dim=-1)

        # Positive-positive similarity (weighted by Tc similarity)
        sim_pos_pos = torch.matmul(z_pos_norm, z_pos_norm.T) / self.temperature

        # Positive-negative similarity
        sim_pos_neg = torch.matmul(z_pos_norm, z_neg_norm.T) / self.temperature

        # Tc-based weighting for positives
        tc_diff = torch.abs(tc_values.unsqueeze(1) - tc_values.unsqueeze(0))
        tc_weights = torch.exp(-tc_diff / 50.0)  # 50K scale
        # Mask diagonal
        mask = ~torch.eye(len(tc_values), dtype=torch.bool, device=tc_values.device)
        tc_weights = tc_weights * mask.float()

        # Mask out self-similarity
        sim_pos_pos = sim_pos_pos.masked_fill(
            torch.eye(len(z_pos), dtype=torch.bool, device=z_pos.device),
            float('-inf')
        )

        # InfoNCE-style loss
        # Numerator: similarity to closest similar-Tc superconductor
        weighted_pos_sim = sim_pos_pos + torch.log(tc_weights + 1e-8)
        max_pos = weighted_pos_sim.max(dim=1)[0]

        # Denominator: all negatives
        neg_logsumexp = torch.logsumexp(sim_pos_neg, dim=1)

        # Loss: push positives above negatives
        loss = -max_pos + neg_logsumexp

        return loss.mean()


# Convenience factory
def create_bidirectional_vae(
    input_dim: int,
    latent_dim: int = 64,
    architecture: str = 'medium'
) -> BidirectionalVAE:
    """
    Create Bidirectional VAE with preset architecture.

    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension
        architecture: 'small', 'medium', or 'large'

    Returns:
        BidirectionalVAE instance
    """
    configs = {
        'small': {
            'encoder_hidden': [128, 64],
            'decoder_hidden': [64, 128],
            'predictor_hidden': [32, 16],
        },
        'medium': {
            'encoder_hidden': [256, 128],
            'decoder_hidden': [128, 256],
            'predictor_hidden': [64, 32],
        },
        'large': {
            'encoder_hidden': [512, 256, 128],
            'decoder_hidden': [128, 256, 512],
            'predictor_hidden': [128, 64, 32],
        },
    }

    config = configs.get(architecture, configs['medium'])

    return BidirectionalVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        **config
    )
