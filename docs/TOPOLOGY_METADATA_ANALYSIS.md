# Latent Space Topology Metadata Analysis

## Overview

The topology analysis module characterizes the structure of the VAE's 2048-dimensional latent space. It provides intrinsic dimensionality estimates, local density analysis, SC/non-SC boundary detection, cluster topology, and pairwise distance statistics.

**Module location**: `src/superconductor/analysis/`
**CLI script**: `scripts/analysis/compute_topology.py`
**Output**: `outputs/topology_summary.jsonl` (compact) + `outputs/topology_metadata_epochNNNN.pt` (full)

## Quick Start

```bash
# Compact analysis of latest z-cache (~10-15s)
cd /home/james/superconductor-vae
PYTHONPATH=src python scripts/analysis/compute_topology.py

# Compact with HDBSCAN (~45s)
PYTHONPATH=src python scripts/analysis/compute_topology.py --hdbscan

# Full analysis with per-sample metadata + HDBSCAN (~70s)
PYTHONPATH=src python scripts/analysis/compute_topology.py --full

# Specific cache file
PYTHONPATH=src python scripts/analysis/compute_topology.py --cache-path outputs/latent_cache_epoch1792.pt

# Longitudinal: process all epoch caches
PYTHONPATH=src python scripts/analysis/compute_topology.py --longitudinal

# Custom k-NN
PYTHONPATH=src python scripts/analysis/compute_topology.py --k 30

# Force CPU only
PYTHONPATH=src python scripts/analysis/compute_topology.py --no-gpu
```

## Training Loop Integration

Enable in `TRAIN_CONFIG`:
```python
'topology_tracking': True,          # Enable topology analysis after z-cache saves
'topology_full_on_best': True,      # Full analysis on best checkpoint only
'topology_k': 20,                   # k-NN neighbors
'topology_n_clusters': 9,           # SC family clusters
```

When enabled:
- **Every z-cache save** (interval or every-epoch): runs compact analysis (~15s), appends to `topology_summary.jsonl`
- **Best checkpoint**: runs full analysis (~30s), saves per-sample metadata to `topology_metadata_epochNNNN.pt`

## Two-Tier Output

### Compact Snapshot (JSONL)

`outputs/topology_summary.jsonl` — one JSON dict per line, ~25 scalar metrics per epoch.

```python
import json
snapshots = [json.loads(line) for line in open('outputs/topology_summary.jsonl')]
```

### Full Metadata (.pt)

`outputs/topology_metadata_epochNNNN.pt` — per-sample tensors via `torch.load()`.

Contents:
- `per_point_intrinsic_dim`: [N] local intrinsic dimensionality
- `per_sample_density`: [N] normalized local density (0-1)
- `heterogeneity`: [N] fraction of k-NN with different SC label
- `boundary_flag`: [N] int8 (1=boundary sample, 0=interior)
- `cluster_labels_kmeans`: [N] int32 (-1=non-SC, 0..8=SC cluster from KMeans)
- `cluster_labels_hdbscan`: [N] int32 (-2=non-SC, -1=noise, 0+=cluster from HDBSCAN)
- `cluster_tc_stats_kmeans`: dict with per-cluster Tc stats (KMeans)
- `cluster_tc_stats_hdbscan`: dict with per-cluster Tc stats + noise stats (HDBSCAN)
- `snapshot`: dict of scalar metrics

## Metrics Reference

### Intrinsic Dimensionality

| Metric | Method | Description |
|--------|--------|-------------|
| `intrinsic_dim_mle` | Levina-Bickel MLE | Global effective degrees of freedom. Much less than 2048 means the model learned a low-dimensional manifold. |
| `intrinsic_dim_mle_sc` | MLE, SC subset | Intrinsic dim for superconductors only |
| `intrinsic_dim_mle_nonsc` | MLE, non-SC subset | Intrinsic dim for non-superconductors only |
| `intrinsic_dim_correlation` | Grassberger-Procaccia | Independent validation via correlation dimension (5K subsample) |

**MLE formula**: For each point, `d_hat_i = [(1/(k-1)) * sum_{j=1}^{k-1} log(r_k/r_j)]^{-1}`. Global = harmonic mean.

**Expected range**: 10-100 for well-trained model (vs 2048 ambient dimensions).

### PCA Spectrum

| Metric | Description |
|--------|-------------|
| `pca_effective_rank` | exp(Shannon entropy of normalized eigenvalues). Number of "effective" orthogonal directions. |
| `pca_variance_top10` | Fraction of variance in top 10 PCs |
| `pca_variance_top50` | Fraction of variance in top 50 PCs |
| `pca_anisotropy` | sigma_max / sigma_min_nonzero. Higher = more elongated latent space. |

### k-NN Density

| Metric | Description |
|--------|-------------|
| `knn_mean_distance` | Mean distance to k-th nearest neighbor |
| `knn_median_distance` | Median distance to k-th neighbor |
| `knn_distance_std` | Std of k-NN distances |
| `density_contrast_sc_nonsc` | mean_r_nonsc / mean_r_sc. >1 means SC region is denser than non-SC. |

### SC/non-SC Boundary

| Metric | Description |
|--------|-------------|
| `boundary_thickness` | Mean distance from boundary samples to nearest opposite-label neighbor. Thin = sharp phase boundary. |
| `boundary_n_samples` | Number of samples in the boundary zone (heterogeneity > 0.3) |
| `sc_nonsc_centroid_distance` | Euclidean distance between SC and non-SC centroids |
| `sc_nonsc_separation_ratio` | Fisher ratio: centroid_dist / sqrt(0.5*(var_SC + var_nonSC)). >1 = separated, <1 = overlapping. |

### KMeans Cluster Topology (SC subset, fixed k=9)

| Metric | Description |
|--------|-------------|
| `n_clusters_sc` | Number of clusters (default 9 = SC families) |
| `silhouette_score_sc` | Silhouette score [-1, 1]. Higher = better cluster separation. |
| `inter_cluster_distance_mean` | Mean pairwise centroid distance |
| `intra_cluster_distance_mean` | Mean distance to own centroid |

### HDBSCAN Density-Based Clustering (SC subset, auto k)

HDBSCAN discovers natural clusters without requiring k. Uses PCA-20 pre-reduction for speed.
See `docs/HDBSCAN_CLUSTERING_GUIDE.md` for detailed usage guide.

| Metric | Description |
|--------|-------------|
| `hdbscan_n_clusters` | Natural cluster count discovered (0 if not run) |
| `hdbscan_noise_fraction` | Fraction of samples not assigned to any cluster |
| `hdbscan_silhouette` | Silhouette on non-noise samples. Typically 2-4x higher than KMeans. |
| `hdbscan_largest_cluster_fraction` | Fraction of data in the dominant cluster |
| `hdbscan_tc_range_largest` | Tc range within the largest cluster |

### Pairwise Distance Distribution

| Metric | Description |
|--------|-------------|
| `pairwise_dist_mean` | Mean pairwise distance (5K subsample) |
| `pairwise_dist_std` | Std of pairwise distances |
| `pairwise_dist_skewness` | Positive skew = more close pairs than expected |
| `pairwise_dist_kurtosis` | Excess kurtosis of distance distribution |

## Physical Interpretation

| Observation | Interpretation |
|-------------|---------------|
| `intrinsic_dim_mle` << 2048 | Model learned a low-dimensional manifold of materials |
| `intrinsic_dim_mle_sc` < `intrinsic_dim_mle_nonsc` | Superconductors live on a simpler submanifold |
| `sc_nonsc_separation_ratio` > 1 | Clear "saturation boundary" in latent space |
| `boundary_thickness` decreasing over epochs | SC/non-SC boundary is sharpening |
| `silhouette_score_sc` > 0.3 | SC families form distinct clusters |
| `intrinsic_dim` increasing during Phase 2 | Self-supervised training expands exploration |
| `density_contrast_sc_nonsc` > 1 | SC samples are more tightly clustered than non-SC |

## Longitudinal Tracking

Use `--longitudinal` to process all epoch caches and track how topology evolves during training:

```bash
PYTHONPATH=src python scripts/analysis/compute_topology.py --longitudinal
```

This produces a JSONL file where each line is a snapshot at a different epoch. Plot metrics over time to track convergence:

```python
import json
import matplotlib.pyplot as plt

snapshots = [json.loads(line) for line in open('outputs/topology_summary.jsonl')]
epochs = [s['epoch'] for s in snapshots]

plt.plot(epochs, [s['intrinsic_dim_mle'] for s in snapshots], label='Intrinsic Dim')
plt.plot(epochs, [s['sc_nonsc_separation_ratio'] for s in snapshots], label='Separation Ratio')
plt.legend()
plt.xlabel('Epoch')
plt.show()
```

## Performance

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| SVD (PCA) | 2-5s | ~700MB VRAM | GPU with CPU fallback |
| k-NN (k=20) | 2-3s GPU / 65s CPU | ~400MB VRAM | GPU chunked cdist + topk |
| MiniBatchKMeans (k=9) | ~2s | ~200MB RAM | SC subset only |
| HDBSCAN (mcs=100) | ~35s | ~500MB RAM | PCA-20 pre-reduction |
| Pairwise dist | 2-3s | ~200MB RAM | 5K stratified subsample |
| Correlation dim | ~25s | ~200MB RAM | 5K subsample, pdist + fit |
| **Compact total** | **~10-15s** | | KMeans only, no HDBSCAN |
| **Compact + HDBSCAN** | **~45s** | | Add --hdbscan flag |
| **Full total** | **~70s** | | All analyses including HDBSCAN |

## Module Structure

```
src/superconductor/analysis/
    __init__.py                 # Exports TopologyAnalyzer, TopologySnapshot
    topology_analyzer.py        # Main orchestrator (GPU k-NN, two-tier output)
    pca_spectrum.py             # SVD eigenvalue spectrum
    intrinsic_dimension.py      # MLE + correlation dimension
    density_estimator.py        # k-NN local density
    boundary_detector.py        # SC/non-SC boundary detection
    cluster_topology.py         # KMeans clustering + silhouette
    hdbscan_topology.py         # HDBSCAN density-based clustering (PCA-20)
    distance_distribution.py    # Pairwise distance statistics
```

## Programmatic Usage

```python
from superconductor.analysis import TopologyAnalyzer

analyzer = TopologyAnalyzer(k=20, use_gpu=True)

# Load z-cache
cache = analyzer.load_z_cache('outputs/latent_cache.pt')
z = cache['z_vectors'].numpy()
is_sc = cache['is_sc'].numpy()
tc = cache['tc_values'].numpy().flatten()  # Flatten [N,1] to [N]

# Compact analysis
snapshot = analyzer.analyze_compact(z, is_sc, epoch=3292)
print(snapshot.summary_str())

# Full analysis with per-sample metadata
snapshot, metadata = analyzer.analyze_full(z, is_sc, tc_values=tc, epoch=3292)
print(metadata['cluster_labels_kmeans'].shape)  # [N]
print(metadata['per_sample_density'].shape)  # [N]
```

## Dependencies

Only uses already-installed libraries:
- `numpy`, `torch` — core computation
- `scikit-learn` — NearestNeighbors, KMeans, silhouette_score
- `scipy` — pdist, skew, kurtosis
- `matplotlib` — optional visualization
