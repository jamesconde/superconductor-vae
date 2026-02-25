"""
PCA spectrum analysis for latent space characterization.

Computes eigenvalue spectrum, effective rank, anisotropy, and cumulative
variance explained from the SVD of centered z-vectors.

Uses torch.linalg.svdvals() on GPU with CPU fallback on OOM.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple


def compute_pca_spectrum(
    z: np.ndarray,
    use_gpu: bool = True,
) -> Dict[str, float]:
    """
    Compute PCA spectrum metrics from z-vectors.

    Args:
        z: [N, D] array of latent vectors
        use_gpu: Try GPU first, fall back to CPU on OOM

    Returns:
        Dict with keys:
            pca_effective_rank: exp(Shannon entropy of normalized eigenvalues)
            pca_variance_top10: Cumulative variance explained by top 10 PCs
            pca_variance_top50: Cumulative variance explained by top 50 PCs
            pca_anisotropy: sigma_1 / sigma_min_nonzero
    """
    z_t = torch.from_numpy(z).float()
    mean = z_t.mean(dim=0)
    centered = z_t - mean

    # Try GPU first for speed
    singular_values = _compute_svdvals(centered, use_gpu)

    # Squared singular values = eigenvalues of covariance (up to 1/(N-1) factor)
    eigvals = singular_values ** 2
    total = eigvals.sum()

    if total < 1e-12:
        return {
            'pca_effective_rank': 0.0,
            'pca_variance_top10': 0.0,
            'pca_variance_top50': 0.0,
            'pca_anisotropy': 1.0,
        }

    # Normalized eigenvalue distribution
    p = eigvals / total
    # Effective rank = exp(Shannon entropy)
    # Guard against log(0) with clamp
    log_p = torch.log(p.clamp(min=1e-30))
    entropy = -(p * log_p).sum().item()
    effective_rank = float(np.exp(entropy))

    # Cumulative variance explained
    cumvar = torch.cumsum(p, dim=0)
    var_top10 = cumvar[min(9, len(cumvar) - 1)].item()
    var_top50 = cumvar[min(49, len(cumvar) - 1)].item()

    # Anisotropy: ratio of largest to smallest nonzero singular value
    nonzero_mask = singular_values > 1e-10
    if nonzero_mask.sum() > 1:
        sigma_max = singular_values[0].item()
        sigma_min = singular_values[nonzero_mask][-1].item()
        anisotropy = sigma_max / max(sigma_min, 1e-10)
    else:
        anisotropy = 1.0

    return {
        'pca_effective_rank': effective_rank,
        'pca_variance_top10': var_top10,
        'pca_variance_top50': var_top50,
        'pca_anisotropy': anisotropy,
    }


def compute_pca_full(
    z: np.ndarray,
    use_gpu: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Compute PCA spectrum metrics and return full eigenvalue spectrum.

    Args:
        z: [N, D] array of latent vectors
        use_gpu: Try GPU first

    Returns:
        (metrics_dict, eigenvalue_spectrum as numpy array)
    """
    z_t = torch.from_numpy(z).float()
    mean = z_t.mean(dim=0)
    centered = z_t - mean

    singular_values = _compute_svdvals(centered, use_gpu)
    eigvals = (singular_values ** 2).cpu().numpy()

    metrics = compute_pca_spectrum(z, use_gpu=use_gpu)
    return metrics, eigvals


def _compute_svdvals(centered: torch.Tensor, use_gpu: bool) -> torch.Tensor:
    """Compute singular values with GPU/CPU fallback."""
    if use_gpu and torch.cuda.is_available():
        try:
            centered_gpu = centered.cuda()
            sv = torch.linalg.svdvals(centered_gpu)
            return sv.cpu()
        except RuntimeError:
            # OOM fallback to CPU
            pass

    return torch.linalg.svdvals(centered)
