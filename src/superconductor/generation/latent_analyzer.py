"""
Latent space analysis for superconductor discovery.

Provides tools to:
- Compute and cache latent embeddings
- Cluster high-Tc regions
- Visualize latent space
- Identify interesting regions for sampling
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClusterInfo:
    """Information about a cluster in latent space."""
    centroid: np.ndarray
    indices: np.ndarray
    mean_tc: float
    std_tc: float
    max_tc: float
    n_samples: int
    formulas: List[str]


class LatentSpaceAnalyzer:
    """
    Analyze and visualize the learned latent space.

    Example:
        analyzer = LatentSpaceAnalyzer(model, dataset)
        analyzer.compute_latent_embeddings()

        # Find high-Tc clusters
        clusters = analyzer.cluster_high_tc_regions(tc_threshold=77.0)

        # Visualize
        analyzer.visualize_latent_space(color_by='tc')
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Any,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize analyzer.

        Args:
            model: Trained BidirectionalVAE model
            dataset: SuperconductorDataset instance
            device: Torch device
        """
        self.model = model
        self.dataset = dataset
        self.device = device

        # Cache
        self.latent_cache: Optional[np.ndarray] = None
        self.tc_values: Optional[np.ndarray] = None
        self.formulas: Optional[List[str]] = None

        self.model.to(device)
        self.model.eval()

    def compute_latent_embeddings(
        self,
        batch_size: int = 256
    ) -> np.ndarray:
        """
        Compute latent embeddings for all samples in dataset.

        Args:
            batch_size: Batch size for encoding

        Returns:
            [n_samples, latent_dim] numpy array of latent vectors
        """
        latents = []
        tc_values = []

        with torch.no_grad():
            for i in range(0, len(self.dataset), batch_size):
                end_idx = min(i + batch_size, len(self.dataset))
                batch_features = self.dataset.features_tensor[i:end_idx].to(self.device)
                batch_tc = self.dataset.tc_values_raw[i:end_idx]

                z = self.model.encode(batch_features, deterministic=True)

                latents.append(z.cpu().numpy())
                tc_values.append(batch_tc)

        self.latent_cache = np.concatenate(latents, axis=0)
        self.tc_values = np.concatenate(tc_values, axis=0)
        self.formulas = self.dataset.formulas

        return self.latent_cache

    def cluster_high_tc_regions(
        self,
        tc_threshold: float = 50.0,
        n_clusters: int = 5,
        method: str = 'kmeans'
    ) -> List[ClusterInfo]:
        """
        Find clusters of high-Tc materials in latent space.

        Args:
            tc_threshold: Minimum Tc to consider as "high"
            n_clusters: Number of clusters to find
            method: Clustering method ('kmeans', 'hdbscan', 'gaussian_mixture')

        Returns:
            List of ClusterInfo objects describing each cluster
        """
        if self.latent_cache is None:
            self.compute_latent_embeddings()

        # Filter high-Tc samples
        high_tc_mask = self.tc_values >= tc_threshold
        high_tc_latents = self.latent_cache[high_tc_mask]
        high_tc_values = self.tc_values[high_tc_mask]
        high_tc_formulas = [f for f, m in zip(self.formulas, high_tc_mask) if m]

        if len(high_tc_latents) < n_clusters:
            print(f"Warning: Only {len(high_tc_latents)} samples above threshold")
            n_clusters = max(1, len(high_tc_latents) // 2)

        # Cluster
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(high_tc_latents)
            centroids = clusterer.cluster_centers_

        elif method == 'gaussian_mixture':
            from sklearn.mixture import GaussianMixture
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = clusterer.fit_predict(high_tc_latents)
            centroids = clusterer.means_

        elif method == 'hdbscan':
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, len(high_tc_latents) // 20))
                labels = clusterer.fit_predict(high_tc_latents)
                # Compute centroids for each cluster
                unique_labels = set(labels) - {-1}  # Exclude noise
                centroids = np.array([
                    high_tc_latents[labels == l].mean(axis=0)
                    for l in sorted(unique_labels)
                ])
                n_clusters = len(unique_labels)
            except ImportError:
                print("HDBSCAN not installed, falling back to kmeans")
                return self.cluster_high_tc_regions(tc_threshold, n_clusters, 'kmeans')
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Build cluster info
        clusters = []
        for i in range(n_clusters):
            if method == 'hdbscan':
                cluster_mask = labels == sorted(set(labels) - {-1})[i]
            else:
                cluster_mask = labels == i

            cluster_indices = np.where(high_tc_mask)[0][cluster_mask]
            cluster_tc = high_tc_values[cluster_mask]
            cluster_formulas = [high_tc_formulas[j] for j in range(len(cluster_mask)) if cluster_mask[j]]

            clusters.append(ClusterInfo(
                centroid=centroids[i],
                indices=cluster_indices,
                mean_tc=float(cluster_tc.mean()),
                std_tc=float(cluster_tc.std()),
                max_tc=float(cluster_tc.max()),
                n_samples=int(cluster_mask.sum()),
                formulas=cluster_formulas
            ))

        # Sort by mean Tc descending
        clusters.sort(key=lambda c: c.mean_tc, reverse=True)

        return clusters

    def visualize_latent_space(
        self,
        color_by: str = 'tc',
        method: str = 'umap',
        output_path: Optional[str] = None,
        show: bool = True,
        highlight_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, Any]:
        """
        Visualize latent space in 2D.

        Args:
            color_by: What to color points by ('tc', 'family', 'cluster')
            method: Dimensionality reduction method ('umap', 'tsne', 'pca')
            output_path: Path to save figure
            show: Whether to display figure
            highlight_indices: Indices of points to highlight

        Returns:
            Tuple of (2D embeddings, matplotlib figure)
        """
        if self.latent_cache is None:
            self.compute_latent_embeddings()

        # Reduce to 2D
        if method == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
            except ImportError:
                print("UMAP not installed, falling back to PCA")
                method = 'pca'

        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)

        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)

        embedding_2d = reducer.fit_transform(self.latent_cache)

        # Import matplotlib
        import matplotlib.pyplot as plt

        # Set up colors
        if color_by == 'tc':
            colors = self.tc_values
            cmap = 'viridis'
            label = 'Tc (K)'
        elif color_by == 'log_tc':
            colors = np.log1p(self.tc_values)
            cmap = 'viridis'
            label = 'log(1 + Tc)'
        else:
            colors = self.tc_values
            cmap = 'viridis'
            label = 'Tc (K)'

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=colors,
            cmap=cmap,
            alpha=0.6,
            s=20
        )

        # Highlight specific points
        if highlight_indices is not None:
            ax.scatter(
                embedding_2d[highlight_indices, 0],
                embedding_2d[highlight_indices, 1],
                c='red',
                marker='*',
                s=100,
                label='Highlighted',
                zorder=5
            )

        plt.colorbar(scatter, ax=ax, label=label)
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_title('Superconductor Latent Space')

        if highlight_indices is not None:
            ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return embedding_2d, fig

    def find_nearest_neighbors(
        self,
        query_z: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors in latent space.

        Args:
            query_z: [latent_dim] or [n_queries, latent_dim] query vector(s)
            k: Number of neighbors

        Returns:
            Tuple of (indices, distances)
        """
        if self.latent_cache is None:
            self.compute_latent_embeddings()

        query_z = np.atleast_2d(query_z)

        # Compute distances
        distances = np.linalg.norm(
            self.latent_cache[np.newaxis, :, :] - query_z[:, np.newaxis, :],
            axis=2
        )

        # Get top-k
        indices = np.argsort(distances, axis=1)[:, :k]
        sorted_distances = np.take_along_axis(distances, indices, axis=1)

        return indices.squeeze(), sorted_distances.squeeze()

    def get_region_statistics(
        self,
        z_center: np.ndarray,
        radius: float = 1.0
    ) -> Dict[str, Any]:
        """
        Get statistics for materials within a latent space region.

        Args:
            z_center: [latent_dim] center of region
            radius: Radius of region

        Returns:
            Dictionary with statistics
        """
        if self.latent_cache is None:
            self.compute_latent_embeddings()

        distances = np.linalg.norm(self.latent_cache - z_center, axis=1)
        mask = distances <= radius

        if not mask.any():
            return {'n_samples': 0, 'warning': 'No samples in region'}

        tc_in_region = self.tc_values[mask]
        formulas_in_region = [f for f, m in zip(self.formulas, mask) if m]

        return {
            'n_samples': int(mask.sum()),
            'mean_tc': float(tc_in_region.mean()),
            'std_tc': float(tc_in_region.std()),
            'min_tc': float(tc_in_region.min()),
            'max_tc': float(tc_in_region.max()),
            'formulas': formulas_in_region[:10],  # First 10
            'tc_values': tc_in_region.tolist(),
        }

    def identify_promising_regions(
        self,
        n_regions: int = 10,
        min_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify promising regions for candidate generation.

        Finds regions with high mean Tc and sufficient samples.

        Args:
            n_regions: Number of regions to return
            min_samples: Minimum samples per region

        Returns:
            List of region dictionaries with statistics
        """
        clusters = self.cluster_high_tc_regions(
            tc_threshold=0,  # Consider all samples
            n_clusters=n_regions * 2  # Get more, filter later
        )

        # Filter by minimum samples
        clusters = [c for c in clusters if c.n_samples >= min_samples]

        # Sort by mean Tc
        clusters.sort(key=lambda c: c.mean_tc, reverse=True)

        # Convert to region dictionaries
        regions = []
        for c in clusters[:n_regions]:
            regions.append({
                'centroid': c.centroid,
                'mean_tc': c.mean_tc,
                'max_tc': c.max_tc,
                'n_samples': c.n_samples,
                'top_formulas': c.formulas[:5]
            })

        return regions
