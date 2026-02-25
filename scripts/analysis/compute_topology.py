#!/usr/bin/env python3
"""
Compute latent space topology metrics from z-cache files.

Standalone CLI for analyzing the topology of the VAE latent space.
Produces both human-readable output and machine-readable JSONL.

Usage:
    # Compact analysis of latest cache (~10s)
    python scripts/analysis/compute_topology.py

    # Compact with HDBSCAN (~45s)
    python scripts/analysis/compute_topology.py --hdbscan

    # Full analysis with per-sample metadata (~70s)
    python scripts/analysis/compute_topology.py --full

    # Specific cache file
    python scripts/analysis/compute_topology.py --cache-path outputs/latent_cache_epoch1792.pt

    # Longitudinal: process all epoch caches
    python scripts/analysis/compute_topology.py --longitudinal

    # Custom k for k-NN
    python scripts/analysis/compute_topology.py --k 30

    # Force CPU
    python scripts/analysis/compute_topology.py --no-gpu
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import glob
import re
import numpy as np
from typing import Optional

from superconductor.analysis.topology_analyzer import TopologyAnalyzer


def parse_epoch_from_path(path: Path) -> int:
    """Extract epoch number from cache filename."""
    name = path.stem
    # Match patterns like latent_cache_epoch1792
    match = re.search(r'epoch(\d+)', name)
    if match:
        return int(match.group(1))
    # Best checkpoint / final
    if 'final' in name:
        return -2
    return -1


def find_cache_files(output_dir: Path) -> list:
    """Find all latent cache files sorted by epoch."""
    pattern = str(output_dir / 'latent_cache*.pt')
    files = [Path(f) for f in glob.glob(pattern)]

    # Sort by epoch number
    def sort_key(p):
        epoch = parse_epoch_from_path(p)
        return (epoch if epoch >= 0 else 999999, p.name)

    return sorted(files, key=sort_key)


def process_single_cache(
    cache_path: Path,
    analyzer: TopologyAnalyzer,
    full: bool = False,
    include_hdbscan: bool = False,
    output_dir: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
) -> None:
    """Process a single z-cache file."""
    print(f"\nLoading: {cache_path}")
    cache = analyzer.load_z_cache(cache_path)

    z = cache['z_vectors'].numpy()
    is_sc = cache['is_sc'].numpy()
    epoch = cache.get('epoch', parse_epoch_from_path(cache_path))

    print(f"  Epoch: {epoch}, Samples: {len(z)}, Latent dim: {z.shape[1]}")

    tc_values = cache.get('tc_values', None)
    if tc_values is not None:
        tc_values = tc_values.numpy()
        # Flatten [N,1] to [N] if needed
        if tc_values.ndim > 1:
            tc_values = tc_values.flatten()

    if full:
        print("  Running FULL analysis (includes HDBSCAN)...")
        snapshot, metadata = analyzer.analyze_full(z, is_sc, tc_values=tc_values, epoch=epoch)

        # Save per-sample metadata
        if output_dir:
            meta_path = output_dir / f'topology_metadata_epoch{epoch:04d}.pt'
            analyzer.save_full_metadata(metadata, meta_path)
            print(f"  Saved full metadata to: {meta_path}")
    else:
        hdb_str = " + HDBSCAN" if include_hdbscan else ""
        print(f"  Running compact analysis{hdb_str}...")
        snapshot = analyzer.analyze_compact(
            z, is_sc, epoch=epoch, include_hdbscan=include_hdbscan
        )

    # Print summary
    print(snapshot.summary_str())

    # Save to JSONL
    if jsonl_path:
        analyzer.save_snapshot(snapshot, jsonl_path)
        print(f"  Appended to: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute latent space topology metrics from z-cache files'
    )
    parser.add_argument(
        '--cache-path', type=str, default=None,
        help='Path to specific z-cache .pt file (default: outputs/latent_cache.pt)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: outputs/)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run full analysis with per-sample metadata + HDBSCAN (~70s)'
    )
    parser.add_argument(
        '--compact', action='store_true',
        help='Run compact analysis only (~10s, default)'
    )
    parser.add_argument(
        '--hdbscan', action='store_true',
        help='Include HDBSCAN in compact mode (~35s extra). Always included in --full.'
    )
    parser.add_argument(
        '--longitudinal', action='store_true',
        help='Process all epoch cache files for longitudinal tracking'
    )
    parser.add_argument(
        '--k', type=int, default=20,
        help='Number of neighbors for k-NN (default: 20)'
    )
    parser.add_argument(
        '--n-clusters', type=int, default=9,
        help='Number of SC clusters for KMeans (default: 9)'
    )
    parser.add_argument(
        '--hdbscan-min-cluster-size', type=int, default=100,
        help='HDBSCAN min_cluster_size (default: 100)'
    )
    parser.add_argument(
        '--hdbscan-pca-dims', type=int, default=20,
        help='PCA dimensions for HDBSCAN pre-reduction (default: 20)'
    )
    parser.add_argument(
        '--no-gpu', action='store_true',
        help='Force CPU-only computation'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'outputs'
    jsonl_path = output_dir / 'topology_summary.jsonl'
    use_gpu = not args.no_gpu
    full = args.full and not args.compact  # --compact overrides --full
    include_hdbscan = args.hdbscan or full

    analyzer = TopologyAnalyzer(
        k=args.k,
        n_clusters=args.n_clusters,
        use_gpu=use_gpu,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_pca_dims=args.hdbscan_pca_dims,
    )

    if args.longitudinal:
        # Process all epoch caches
        cache_files = find_cache_files(output_dir)
        if not cache_files:
            print(f"No cache files found in {output_dir}")
            sys.exit(1)

        print(f"Found {len(cache_files)} cache files for longitudinal analysis")
        for cache_path in cache_files:
            try:
                process_single_cache(
                    cache_path, analyzer,
                    full=full, include_hdbscan=include_hdbscan,
                    output_dir=output_dir, jsonl_path=jsonl_path,
                )
            except Exception as e:
                print(f"  ERROR processing {cache_path}: {e}")
                continue

        print(f"\nLongitudinal analysis complete. Results in: {jsonl_path}")
    else:
        # Single cache
        if args.cache_path:
            cache_path = Path(args.cache_path)
            if not cache_path.is_absolute():
                cache_path = PROJECT_ROOT / cache_path
        else:
            cache_path = output_dir / 'latent_cache.pt'

        if not cache_path.exists():
            print(f"Error: Cache file not found at {cache_path}")
            print("Run training with 'cache_z_vectors: True' to generate the cache.")
            sys.exit(1)

        process_single_cache(
            cache_path, analyzer,
            full=full, include_hdbscan=include_hdbscan,
            output_dir=output_dir, jsonl_path=jsonl_path,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
