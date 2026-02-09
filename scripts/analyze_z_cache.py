#!/usr/bin/env python3
"""
Analyze cached latent z vectors from training.

Usage:
    python scripts/analyze_z_cache.py [--cache-path outputs/latent_cache.pt]

Outputs:
    - Z-space statistics (mean, std, norms)
    - Correlation analysis with Tc
    - Dimension importance ranking
    - t-SNE/UMAP visualization (optional)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import torch
import numpy as np
import pandas as pd
from typing import Optional


def load_z_cache(cache_path: Path) -> dict:
    """Load z-cache from disk."""
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    print(f"Loaded z-cache from {cache_path}")
    print(f"  Epoch: {cache['epoch']}")
    print(f"  Samples: {cache['n_samples']}")
    print(f"  Latent dim: {cache['latent_dim']}")
    print(f"  Timestamp: {cache['timestamp']}")
    return cache


def analyze_z_statistics(cache: dict) -> dict:
    """Compute z-space statistics."""
    z = cache['z_vectors'].numpy()
    tc = cache['tc_values'].numpy()
    is_sc = cache['is_sc'].numpy()

    stats = {
        'n_samples': len(z),
        'latent_dim': z.shape[1],
        'z_norm_mean': np.linalg.norm(z, axis=1).mean(),
        'z_norm_std': np.linalg.norm(z, axis=1).std(),
    }

    # Per-dimension statistics
    stats['dim_means'] = z.mean(axis=0)
    stats['dim_stds'] = z.std(axis=0)
    stats['dim_mins'] = z.min(axis=0)
    stats['dim_maxs'] = z.max(axis=0)

    # Find most variable dimensions
    dim_variance = z.var(axis=0)
    top_dims = np.argsort(dim_variance)[::-1][:20]
    stats['top_variance_dims'] = top_dims
    stats['top_variance_values'] = dim_variance[top_dims]

    print("\n=== Z-Space Statistics ===")
    print(f"Z-norm: mean={stats['z_norm_mean']:.2f}, std={stats['z_norm_std']:.2f}")
    print(f"\nTop 10 most variable dimensions:")
    for i, dim in enumerate(top_dims[:10]):
        print(f"  Dim {dim}: variance={dim_variance[dim]:.4f}, "
              f"mean={stats['dim_means'][dim]:.4f}, std={stats['dim_stds'][dim]:.4f}")

    return stats


def analyze_tc_correlations(cache: dict) -> dict:
    """Analyze correlations between z dimensions and Tc."""
    z = cache['z_vectors'].numpy()
    tc = cache['tc_values'].numpy()
    is_sc = cache['is_sc'].numpy().astype(bool)

    # Only analyze superconductors for Tc correlation
    z_sc = z[is_sc]
    tc_sc = tc[is_sc]

    if len(tc_sc) == 0:
        print("\nNo superconductors in cache - skipping Tc correlation analysis")
        return {}

    # Compute correlations
    correlations = np.array([np.corrcoef(z_sc[:, i], tc_sc)[0, 1] for i in range(z.shape[1])])

    # Handle NaN correlations
    correlations = np.nan_to_num(correlations, nan=0.0)

    # Find most correlated dimensions
    top_pos = np.argsort(correlations)[::-1][:10]
    top_neg = np.argsort(correlations)[:10]

    print("\n=== Tc Correlations (Superconductors Only) ===")
    print(f"Analyzed {len(tc_sc)} superconductors")
    print(f"\nTop 10 dimensions POSITIVELY correlated with Tc:")
    for dim in top_pos:
        print(f"  Dim {dim}: r={correlations[dim]:.4f}")

    print(f"\nTop 10 dimensions NEGATIVELY correlated with Tc:")
    for dim in top_neg:
        print(f"  Dim {dim}: r={correlations[dim]:.4f}")

    return {
        'correlations': correlations,
        'top_positive_dims': top_pos,
        'top_negative_dims': top_neg,
    }


def analyze_sc_vs_non_sc(cache: dict) -> dict:
    """Analyze differences between SC and non-SC z vectors."""
    z = cache['z_vectors'].numpy()
    is_sc = cache['is_sc'].numpy().astype(bool)

    z_sc = z[is_sc]
    z_non_sc = z[~is_sc]

    if len(z_non_sc) == 0:
        print("\nNo non-superconductors in cache - skipping SC vs non-SC analysis")
        return {}

    print(f"\n=== SC vs Non-SC Analysis ===")
    print(f"Superconductors: {len(z_sc)}")
    print(f"Non-superconductors: {len(z_non_sc)}")

    # Compare norms
    sc_norms = np.linalg.norm(z_sc, axis=1)
    non_sc_norms = np.linalg.norm(z_non_sc, axis=1)
    print(f"\nZ-norm comparison:")
    print(f"  SC: mean={sc_norms.mean():.2f}, std={sc_norms.std():.2f}")
    print(f"  Non-SC: mean={non_sc_norms.mean():.2f}, std={non_sc_norms.std():.2f}")

    # Find dimensions with largest mean difference
    mean_diff = z_sc.mean(axis=0) - z_non_sc.mean(axis=0)
    top_diff_dims = np.argsort(np.abs(mean_diff))[::-1][:10]

    print(f"\nTop 10 dimensions with largest SC vs non-SC difference:")
    for dim in top_diff_dims:
        print(f"  Dim {dim}: SC_mean={z_sc[:, dim].mean():.4f}, "
              f"nonSC_mean={z_non_sc[:, dim].mean():.4f}, diff={mean_diff[dim]:.4f}")

    return {
        'mean_diff': mean_diff,
        'top_diff_dims': top_diff_dims,
    }


def visualize_z_space(cache: dict, method: str = 'tsne', output_path: Optional[Path] = None):
    """Create 2D visualization of z-space."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("\nSkipping visualization - matplotlib/sklearn not available")
        return

    z = cache['z_vectors'].numpy()
    tc = cache['tc_values'].numpy()
    is_sc = cache['is_sc'].numpy().astype(bool)

    print(f"\n=== Z-Space Visualization ({method.upper()}) ===")

    # Subsample if too large
    max_samples = 5000
    if len(z) > max_samples:
        indices = np.random.choice(len(z), max_samples, replace=False)
        z = z[indices]
        tc = tc[indices]
        is_sc = is_sc[indices]
        print(f"Subsampled to {max_samples} points")

    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    print("Computing 2D embedding...")
    z_2d = reducer.fit_transform(z)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Color by Tc
    ax1 = axes[0]
    scatter1 = ax1.scatter(z_2d[is_sc, 0], z_2d[is_sc, 1],
                           c=tc[is_sc], cmap='plasma', s=10, alpha=0.6)
    ax1.scatter(z_2d[~is_sc, 0], z_2d[~is_sc, 1],
                c='gray', s=5, alpha=0.3, label='Non-SC')
    plt.colorbar(scatter1, ax=ax1, label='Tc (K)')
    ax1.set_title(f'Z-Space ({method.upper()}) - Colored by Tc')
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    ax1.legend()

    # Plot 2: Color by SC/non-SC
    ax2 = axes[1]
    ax2.scatter(z_2d[~is_sc, 0], z_2d[~is_sc, 1],
                c='gray', s=5, alpha=0.3, label='Non-SC')
    ax2.scatter(z_2d[is_sc, 0], z_2d[is_sc, 1],
                c='blue', s=10, alpha=0.6, label='SC')
    ax2.set_title(f'Z-Space ({method.upper()}) - SC vs Non-SC')
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax2.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze cached z vectors')
    parser.add_argument('--cache-path', type=str, default='outputs/latent_cache.pt',
                        help='Path to z-cache file')
    parser.add_argument('--visualize', action='store_true',
                        help='Create t-SNE visualization')
    parser.add_argument('--vis-method', type=str, default='tsne',
                        choices=['tsne', 'pca'], help='Visualization method')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Save visualization to file')
    args = parser.parse_args()

    cache_path = PROJECT_ROOT / args.cache_path
    if not cache_path.exists():
        print(f"Error: Cache file not found at {cache_path}")
        print("Run training with 'cache_z_vectors: True' to generate the cache.")
        sys.exit(1)

    cache = load_z_cache(cache_path)

    # Run analyses
    z_stats = analyze_z_statistics(cache)
    tc_corr = analyze_tc_correlations(cache)
    sc_diff = analyze_sc_vs_non_sc(cache)

    # Visualize if requested
    if args.visualize:
        output_path = Path(args.output_plot) if args.output_plot else None
        visualize_z_space(cache, method=args.vis_method, output_path=output_path)

    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()
