# HDBSCAN Clustering Guide for Latent Space Analysis

## What Is HDBSCAN?

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm. Unlike KMeans, it:

- **Discovers the natural cluster count** — no need to specify k upfront
- **Identifies noise/outlier points** — samples that don't belong to any cluster get label -1
- **Handles non-convex cluster shapes** — finds clusters of arbitrary shape, not just spherical
- **Finds clusters of varying density** — a small dense cluster and a large sparse cluster can coexist

## When to Use HDBSCAN vs KMeans

| Scenario | Use KMeans | Use HDBSCAN |
|----------|-----------|-------------|
| Tracking known SC family clusters over training | Yes (k=9 fixed) | No |
| Discovering natural groupings in latent space | No | Yes |
| Identifying outlier/unusual materials | No (forces assignment) | Yes (labels noise) |
| Per-epoch training loop monitoring | Yes (fast, ~2s) | No (slow, ~35s) |
| Best checkpoint deep analysis | Both | Both |
| Understanding latent space structure | Supplementary | Primary |
| Comparing across model versions | Yes (consistent k) | Yes (track cluster emergence) |

**Rule of thumb**: Use KMeans for fast, consistent longitudinal tracking during training. Use HDBSCAN for deeper structural analysis at key checkpoints.

## How HDBSCAN Works in This Codebase

### PCA Pre-Reduction (Critical)

HDBSCAN is O(N^2) and struggles with the curse of dimensionality in high-D spaces. Running it on raw 2048D latent vectors would take hours and produce poor results.

We apply **PCA to 20 dimensions** before HDBSCAN. This:
- Reduces computation from hours to ~35 seconds
- Captures 61% of variance (top 10) / 91% (top 50) — 20 PCs is sufficient
- Removes noise dimensions that confuse density-based clustering
- Matches the intrinsic dimensionality (~4.5D MLE) well

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 100 | Minimum samples to form a cluster. Lower = more granular clusters + more noise. Higher = fewer, larger clusters. |
| `pca_dims` | 20 | PCA components for pre-reduction. 20 is the sweet spot for this data. |

### Tuning `min_cluster_size`

Benchmarked on V12.41 epoch 1792 (50,958 SC samples, PCA-20):

| min_cluster_size | Clusters | Noise % | Silhouette | Time |
|-----------------|----------|---------|------------|------|
| 50 | 13 | 47.7% | 0.120 | 36s |
| **100** | **6** | **42.6%** | **0.287** | **36s** |
| 200 | 3 | 41.6% | 0.350 | 40s |
| 500 | 3 | 48.0% | 0.368 | 37s |

**Default 100 is recommended** — finds a moderate number of meaningful clusters with good silhouette score.

- Use **50** when looking for fine-grained substructure (e.g., cuprate sub-families)
- Use **200-500** when looking for only the major population divisions

## Interpreting HDBSCAN Results

### Metrics

| Metric | Description | Good Sign |
|--------|-------------|-----------|
| `hdbscan_n_clusters` | Natural cluster count discovered | Matches expected SC family structure |
| `hdbscan_noise_fraction` | Fraction of points not assigned to any cluster | 20-50% is normal; >70% means data is diffuse |
| `hdbscan_silhouette` | Cluster separation quality [-1, 1] | > 0.2 (good), > 0.4 (excellent) |
| `hdbscan_largest_cluster_fraction` | Fraction of data in the largest cluster | < 0.5 (diverse structure), > 0.8 (one dominant group) |
| `hdbscan_tc_range_largest` | Tc range within the largest cluster | Small = tight Tc grouping |

### Cluster Labels

In full metadata (`topology_metadata_epochNNNN.pt`):
- `cluster_labels_hdbscan[i] = -2`: non-SC sample (not analyzed)
- `cluster_labels_hdbscan[i] = -1`: noise/outlier (no cluster assignment)
- `cluster_labels_hdbscan[i] >= 0`: cluster ID

### What Noise Means

HDBSCAN noise points are **not errors** — they represent:
- **Transitional materials** between SC families
- **Rare/unusual compositions** that don't fit any dense group
- **Boundary materials** between distinct regions of latent space

These are often the most interesting samples for discovery. Cross-reference with `heterogeneity` and `boundary_flag` from the topology metadata.

## Physical Interpretation

### V12.41 Epoch 1792 Results

HDBSCAN found 6 natural clusters in the SC latent space:

| Cluster | Count | Tc Mean (z-score) | Interpretation |
|---------|-------|-------------------|----------------|
| 3 | 21,815 | -2.014 | Low-Tc conventional (BCS) — massive group |
| 5 | 4,568 | 0.991 | High-Tc cuprates |
| 2 | 2,063 | -0.590 | Moderate-Tc (mixed families) |
| 4 | 444 | 0.124 | Moderate-Tc distinct group |
| 1 | 218 | 0.648 | Small high-Tc cluster |
| 0 | 131 | -0.870 | Small low-Tc outlier group |
| Noise | 21,719 | -0.604 | Unassigned (transitional) |

Key observations:
- The **dominant cluster (3)** captures the large BCS/conventional SC population at very low Tc
- **Cluster 5** isolates high-Tc cuprates — the model separates these from the main population
- **42.6% noise** indicates significant transitional material between the main populations
- The silhouette score (0.287) is **3.3x better than KMeans' (0.087)** — HDBSCAN finds more natural structure

### Longitudinal Tracking

Over training, watch for:
- `hdbscan_n_clusters` increasing: model is differentiating more SC families
- `hdbscan_noise_fraction` decreasing: latent space is becoming more structured
- `hdbscan_silhouette` increasing: clusters are becoming more distinct
- Cluster Tc distributions narrowing: model is learning to separate by Tc

## CLI Usage

```bash
cd /home/james/superconductor-vae

# Compact analysis with HDBSCAN (~45s)
PYTHONPATH=src python scripts/analysis/compute_topology.py --hdbscan

# Full analysis always includes HDBSCAN (~70s)
PYTHONPATH=src python scripts/analysis/compute_topology.py --full

# Custom HDBSCAN parameters
PYTHONPATH=src python scripts/analysis/compute_topology.py --hdbscan \
    --hdbscan-min-cluster-size 200 \
    --hdbscan-pca-dims 30

# Longitudinal with HDBSCAN
PYTHONPATH=src python scripts/analysis/compute_topology.py --longitudinal --hdbscan
```

## Programmatic Usage

```python
from superconductor.analysis import TopologyAnalyzer

analyzer = TopologyAnalyzer(
    hdbscan_min_cluster_size=100,
    hdbscan_pca_dims=20,
)

cache = analyzer.load_z_cache('outputs/latent_cache.pt')
z = cache['z_vectors'].numpy()
is_sc = cache['is_sc'].numpy()
tc = cache['tc_values'].numpy().flatten()

# Compact with HDBSCAN
snapshot = analyzer.analyze_compact(z, is_sc, epoch=1792, include_hdbscan=True)
print(f"HDBSCAN found {snapshot.hdbscan_n_clusters} clusters")
print(f"  Noise: {snapshot.hdbscan_noise_fraction*100:.1f}%")
print(f"  Silhouette: {snapshot.hdbscan_silhouette:.4f}")

# Full analysis (always includes HDBSCAN)
snapshot, metadata = analyzer.analyze_full(z, is_sc, tc_values=tc, epoch=1792)

# Get per-sample HDBSCAN labels
hdb_labels = metadata['cluster_labels_hdbscan'].numpy()
noise_mask = hdb_labels == -1
print(f"Noise samples: {noise_mask.sum()}")

# Per-cluster Tc statistics
for k, v in metadata['cluster_tc_stats_hdbscan'].items():
    print(f"  {k}: n={v['count']}, tc_mean={v['tc_mean']:.3f}")
```

## HDBSCAN vs KMeans: Side-by-Side Comparison

On V12.41 epoch 1792 (50,958 SC samples):

| | KMeans (k=9) | HDBSCAN (mcs=100) |
|---|---|---|
| Clusters | 9 (forced) | 6 (discovered) |
| Noise/unassigned | 0% | 42.6% |
| Silhouette | 0.087 | 0.287 |
| Compute time | ~2s | ~35s |
| Assumes shape | Spherical | Arbitrary |
| Consistent across epochs | Yes (same k) | Natural variation |

**HDBSCAN's silhouette is 3.3x higher** because it only assigns confident points to clusters, while KMeans forces every point into a cluster (degrading its score with borderline assignments).

## Module Location

- **Implementation**: `src/superconductor/analysis/hdbscan_topology.py`
- **Orchestrator**: `src/superconductor/analysis/topology_analyzer.py`
- **CLI**: `scripts/analysis/compute_topology.py`
- **This guide**: `docs/HDBSCAN_CLUSTERING_GUIDE.md`

## Dependencies

Uses `sklearn.cluster.HDBSCAN` (built into scikit-learn >= 1.3). No additional packages required.
