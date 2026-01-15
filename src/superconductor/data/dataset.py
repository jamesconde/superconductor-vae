"""
PyTorch Dataset implementations for superconductor data.

Provides:
- SuperconductorDataset: Main dataset for Tc prediction
- ContrastiveDataset: Dataset for contrastive learning with negative samples
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

from ..encoders.composition_encoder import CompositionEncoder, EncodedComposition
from ..encoders.feature_groups import FeatureGroups


@dataclass
class SuperconductorSample:
    """Container for a single superconductor sample."""
    formula: str
    tc: float  # Critical temperature in Kelvin
    features: torch.Tensor  # Encoded features
    composition: torch.Tensor  # Composition vector
    element_stats: torch.Tensor  # Element statistics

    # Optional additional data
    structure_type: Optional[str] = None
    space_group: Optional[int] = None
    family: Optional[str] = None  # cuprate, iron-based, etc.
    reference: Optional[str] = None
    year_discovered: Optional[int] = None

    # Metadata
    idx: int = 0


class SuperconductorDataset(Dataset):
    """
    PyTorch Dataset for superconductor Tc prediction.

    Loads superconductor data, encodes chemical formulas,
    and provides batched access for training.

    Example:
        dataset = SuperconductorDataset.from_csv('supercon.csv')
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for features, tc in loader:
            # features: [batch, feature_dim]
            # tc: [batch]
            pass
    """

    def __init__(
        self,
        formulas: List[str],
        tc_values: np.ndarray,
        encoder: Optional[CompositionEncoder] = None,
        additional_features: Optional[Dict[str, np.ndarray]] = None,
        normalize_tc: bool = True,
        tc_log_transform: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize superconductor dataset.

        Args:
            formulas: List of chemical formula strings
            tc_values: Array of critical temperatures (Kelvin)
            encoder: Optional pre-configured encoder
            additional_features: Dict of additional feature arrays
            normalize_tc: Whether to normalize Tc values
            tc_log_transform: Whether to log-transform Tc
            device: Torch device
        """
        self.formulas = formulas
        self.tc_values_raw = np.array(tc_values, dtype=np.float32)
        self.additional_features = additional_features or {}
        self.normalize_tc = normalize_tc
        self.tc_log_transform = tc_log_transform
        self.device = device

        # Initialize encoder
        self.encoder = encoder or CompositionEncoder(device='cpu')

        # Encode all formulas
        self._encode_all()

        # Normalize Tc if requested
        self._process_tc()

    def _encode_all(self):
        """Encode all chemical formulas."""
        self.encoded: List[EncodedComposition] = []
        self.composition_vectors: List[torch.Tensor] = []
        self.element_stats_list: List[torch.Tensor] = []

        for formula in self.formulas:
            encoded = self.encoder.encode(formula)
            self.encoded.append(encoded)
            self.composition_vectors.append(encoded.composition_vector.cpu())
            self.element_stats_list.append(encoded.element_stats.cpu())

        # Stack into tensors
        self.composition_tensor = torch.stack(self.composition_vectors)
        self.element_stats_tensor = torch.stack(self.element_stats_list)

        # Concatenate to full feature tensor
        self.features_tensor = torch.cat([
            self.composition_tensor,
            self.element_stats_tensor
        ], dim=-1)

    def _process_tc(self):
        """Process Tc values (normalize, log-transform)."""
        tc = self.tc_values_raw.copy()

        # Log transform if requested (helps with wide Tc range)
        if self.tc_log_transform:
            tc = np.log1p(tc)  # log(1 + tc) to handle tc=0

        # Compute normalization statistics
        self.tc_mean = tc.mean()
        self.tc_std = tc.std() + 1e-8

        # Normalize
        if self.normalize_tc:
            tc = (tc - self.tc_mean) / self.tc_std

        self.tc_values = torch.tensor(tc, dtype=torch.float32)

    def denormalize_tc(self, tc_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized Tc back to original scale."""
        tc = tc_normalized
        if self.normalize_tc:
            tc = tc * self.tc_std + self.tc_mean
        if self.tc_log_transform:
            tc = torch.expm1(tc)  # exp(tc) - 1
        return tc

    def __len__(self) -> int:
        return len(self.formulas)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.

        Returns:
            Tuple of (features, tc)
        """
        return self.features_tensor[idx], self.tc_values[idx]

    def get_sample(self, idx: int) -> SuperconductorSample:
        """Get full sample with metadata."""
        encoded = self.encoded[idx]
        return SuperconductorSample(
            formula=encoded.formula,
            tc=self.tc_values_raw[idx],
            features=self.features_tensor[idx],
            composition=self.composition_tensor[idx],
            element_stats=self.element_stats_tensor[idx],
            idx=idx
        )

    def get_feature_groups(self, idx: int) -> FeatureGroups:
        """Get features as FeatureGroups object."""
        return FeatureGroups(
            composition=self.composition_tensor[idx:idx+1],
            element_stats=self.element_stats_tensor[idx:idx+1]
        )

    def get_batch_feature_groups(
        self,
        indices: List[int]
    ) -> FeatureGroups:
        """Get batch of features as FeatureGroups."""
        return FeatureGroups(
            composition=self.composition_tensor[indices],
            element_stats=self.element_stats_tensor[indices]
        )

    @property
    def feature_dim(self) -> int:
        """Total feature dimension."""
        return self.features_tensor.shape[-1]

    @property
    def composition_dim(self) -> int:
        """Composition vector dimension."""
        return self.composition_tensor.shape[-1]

    @property
    def stats_dim(self) -> int:
        """Element statistics dimension."""
        return self.element_stats_tensor.shape[-1]

    def get_tc_statistics(self) -> Dict[str, float]:
        """Get Tc value statistics."""
        tc = self.tc_values_raw
        return {
            'min': float(tc.min()),
            'max': float(tc.max()),
            'mean': float(tc.mean()),
            'std': float(tc.std()),
            'median': float(np.median(tc)),
            'count': len(tc),
            'count_above_77K': int((tc > 77).sum()),  # Liquid N2
            'count_above_90K': int((tc > 90).sum()),  # YBCO regime
        }

    def filter_by_tc(
        self,
        min_tc: Optional[float] = None,
        max_tc: Optional[float] = None
    ) -> 'SuperconductorDataset':
        """Create filtered dataset by Tc range."""
        mask = np.ones(len(self), dtype=bool)
        if min_tc is not None:
            mask &= self.tc_values_raw >= min_tc
        if max_tc is not None:
            mask &= self.tc_values_raw <= max_tc

        indices = np.where(mask)[0]
        formulas = [self.formulas[i] for i in indices]
        tc_values = self.tc_values_raw[indices]

        return SuperconductorDataset(
            formulas=formulas,
            tc_values=tc_values,
            encoder=self.encoder,
            normalize_tc=self.normalize_tc,
            tc_log_transform=self.tc_log_transform,
            device=self.device
        )

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_state: int = 42
    ) -> Tuple['SuperconductorDataset', 'SuperconductorDataset', 'SuperconductorDataset']:
        """
        Split dataset into train/val/test.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            random_state: Random seed

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        np.random.seed(random_state)
        n = len(self)
        indices = np.random.permutation(n)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        def make_subset(idx):
            formulas = [self.formulas[i] for i in idx]
            tc_values = self.tc_values_raw[idx]
            return SuperconductorDataset(
                formulas=formulas,
                tc_values=tc_values,
                encoder=self.encoder,
                normalize_tc=self.normalize_tc,
                tc_log_transform=self.tc_log_transform,
                device=self.device
            )

        return make_subset(train_idx), make_subset(val_idx), make_subset(test_idx)

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        formula_column: str = 'formula',
        tc_column: str = 'tc',
        **kwargs
    ) -> 'SuperconductorDataset':
        """
        Load dataset from CSV file.

        Args:
            path: Path to CSV file
            formula_column: Column name for chemical formulas
            tc_column: Column name for Tc values
            **kwargs: Additional arguments for __init__

        Returns:
            SuperconductorDataset instance
        """
        df = pd.read_csv(path)

        # Handle common column name variations
        if formula_column not in df.columns:
            for alt in ['Formula', 'FORMULA', 'composition', 'Composition']:
                if alt in df.columns:
                    formula_column = alt
                    break

        if tc_column not in df.columns:
            for alt in ['Tc', 'TC', 'critical_temperature', 'Tc_K', 'tc_k']:
                if alt in df.columns:
                    tc_column = alt
                    break

        formulas = df[formula_column].tolist()
        tc_values = df[tc_column].values

        # Filter out invalid entries
        valid_mask = ~np.isnan(tc_values) & (tc_values >= 0)
        formulas = [f for f, v in zip(formulas, valid_mask) if v]
        tc_values = tc_values[valid_mask]

        return cls(formulas=formulas, tc_values=tc_values, **kwargs)

    @classmethod
    def from_supercon(
        cls,
        path: Union[str, Path],
        **kwargs
    ) -> 'SuperconductorDataset':
        """
        Load from SuperCon database format.

        The SuperCon database typically has columns like:
        - formula or name
        - Tc or critical_temperature

        Args:
            path: Path to SuperCon data file
            **kwargs: Additional arguments

        Returns:
            SuperconductorDataset instance
        """
        # SuperCon format detection
        path = Path(path)

        if path.suffix == '.csv':
            return cls.from_csv(path, **kwargs)
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
            # Find formula and Tc columns
            formula_col = None
            tc_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'formula' in col_lower or 'composition' in col_lower:
                    formula_col = col
                if 'tc' in col_lower or 'critical' in col_lower:
                    tc_col = col

            if formula_col is None or tc_col is None:
                raise ValueError(f"Could not find formula/Tc columns in {df.columns.tolist()}")

            formulas = df[formula_col].tolist()
            tc_values = df[tc_col].values
            valid_mask = ~np.isnan(tc_values) & (tc_values >= 0)
            formulas = [f for f, v in zip(formulas, valid_mask) if v]
            tc_values = tc_values[valid_mask]

            return cls(formulas=formulas, tc_values=tc_values, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @classmethod
    def from_featurized_supercon(
        cls,
        path: Union[str, Path],
        use_magpie_features: bool = True,
        use_composition_encoding: bool = True,
        formula_column: str = 'formula',
        tc_column: str = 'Tc',
        category_column: str = 'category',
        **kwargs
    ) -> 'FeaturizedSuperconductorDataset':
        """
        Load pre-featurized SuperCon data with Magpie features.

        This handles the supercon.csv format which already contains
        ~150 Magpie elemental property features.

        Args:
            path: Path to featurized CSV
            use_magpie_features: Whether to use pre-computed Magpie features
            use_composition_encoding: Whether to add our composition encoding
            formula_column: Column name for formulas
            tc_column: Column name for Tc values
            category_column: Column name for superconductor category
            **kwargs: Additional arguments

        Returns:
            FeaturizedSuperconductorDataset instance
        """
        return FeaturizedSuperconductorDataset.from_csv(
            path,
            use_magpie_features=use_magpie_features,
            use_composition_encoding=use_composition_encoding,
            formula_column=formula_column,
            tc_column=tc_column,
            category_column=category_column,
            **kwargs
        )


class FeaturizedSuperconductorDataset(Dataset):
    """
    Dataset for pre-featurized SuperCon data with Magpie features.

    Handles the supercon.csv format which contains ~150 pre-computed
    Magpie elemental property features.

    Example:
        dataset = FeaturizedSuperconductorDataset.from_csv(
            'data/superconductor/raw/supercon.csv'
        )
        print(f"Features: {dataset.feature_dim}")  # ~150+ features
    """

    # Columns to exclude from features
    NON_FEATURE_COLUMNS = {
        'formula', 'Tc', 'composition', 'category',
        'compound possible', 'transition metal fraction'
    }

    def __init__(
        self,
        formulas: List[str],
        tc_values: np.ndarray,
        magpie_features: Optional[np.ndarray] = None,
        categories: Optional[List[str]] = None,
        encoder: Optional[CompositionEncoder] = None,
        use_composition_encoding: bool = True,
        normalize_tc: bool = True,
        tc_log_transform: bool = False,
        normalize_features: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize featurized dataset.

        Args:
            formulas: List of chemical formula strings
            tc_values: Array of critical temperatures (Kelvin)
            magpie_features: Pre-computed Magpie feature matrix [n_samples, n_features]
            categories: Superconductor category labels
            encoder: Optional composition encoder
            use_composition_encoding: Whether to add composition encoding
            normalize_tc: Whether to normalize Tc values
            tc_log_transform: Whether to log-transform Tc
            normalize_features: Whether to normalize Magpie features
            device: Torch device
        """
        self.formulas = formulas
        self.tc_values_raw = np.array(tc_values, dtype=np.float32)
        self.categories = categories
        self.use_composition_encoding = use_composition_encoding
        self.normalize_tc = normalize_tc
        self.tc_log_transform = tc_log_transform
        self.normalize_features = normalize_features
        self.device = device

        # Store Magpie features
        self.magpie_features_raw = magpie_features
        if magpie_features is not None:
            self.n_magpie_features = magpie_features.shape[1]
        else:
            self.n_magpie_features = 0

        # Initialize encoder if using composition encoding
        if use_composition_encoding:
            self.encoder = encoder or CompositionEncoder(device='cpu')
            self._encode_compositions()
        else:
            self.encoder = None
            self.composition_tensor = None
            self.element_stats_tensor = None

        # Process features
        self._process_features()

        # Process Tc
        self._process_tc()

    def _encode_compositions(self):
        """Encode all chemical formulas."""
        composition_vectors = []
        element_stats_list = []

        for formula in self.formulas:
            encoded = self.encoder.encode(formula)
            composition_vectors.append(encoded.composition_vector.cpu())
            element_stats_list.append(encoded.element_stats.cpu())

        self.composition_tensor = torch.stack(composition_vectors)
        self.element_stats_tensor = torch.stack(element_stats_list)

    def _process_features(self):
        """Combine and normalize features."""
        feature_parts = []

        # Add Magpie features
        if self.magpie_features_raw is not None:
            magpie = self.magpie_features_raw.copy()

            # Handle NaN values
            magpie = np.nan_to_num(magpie, nan=0.0, posinf=0.0, neginf=0.0)

            if self.normalize_features:
                # Z-score normalization
                self.magpie_mean = magpie.mean(axis=0)
                self.magpie_std = magpie.std(axis=0) + 1e-8
                magpie = (magpie - self.magpie_mean) / self.magpie_std

            feature_parts.append(torch.tensor(magpie, dtype=torch.float32))

        # Add composition encoding
        if self.use_composition_encoding and self.composition_tensor is not None:
            feature_parts.append(self.composition_tensor)
            feature_parts.append(self.element_stats_tensor)

        # Concatenate all features
        if feature_parts:
            self.features_tensor = torch.cat(feature_parts, dim=-1)
        else:
            raise ValueError("No features available. Enable Magpie or composition features.")

    def _process_tc(self):
        """Process Tc values."""
        tc = self.tc_values_raw.copy()

        if self.tc_log_transform:
            tc = np.log1p(tc)

        self.tc_mean = tc.mean()
        self.tc_std = tc.std() + 1e-8

        if self.normalize_tc:
            tc = (tc - self.tc_mean) / self.tc_std

        self.tc_values = torch.tensor(tc, dtype=torch.float32)

    def denormalize_tc(self, tc_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized Tc back to original scale."""
        tc = tc_normalized
        if self.normalize_tc:
            tc = tc * self.tc_std + self.tc_mean
        if self.tc_log_transform:
            tc = torch.expm1(tc)
        return tc

    def __len__(self) -> int:
        return len(self.formulas)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features_tensor[idx], self.tc_values[idx]

    @property
    def feature_dim(self) -> int:
        return self.features_tensor.shape[-1]

    @property
    def composition_dim(self) -> int:
        if self.composition_tensor is not None:
            return self.composition_tensor.shape[-1]
        return 0

    def get_tc_statistics(self) -> Dict[str, float]:
        """Get Tc value statistics."""
        tc = self.tc_values_raw
        return {
            'min': float(tc.min()),
            'max': float(tc.max()),
            'mean': float(tc.mean()),
            'std': float(tc.std()),
            'median': float(np.median(tc)),
            'count': len(tc),
            'count_above_77K': int((tc > 77).sum()),
            'count_above_90K': int((tc > 90).sum()),
            'n_magpie_features': self.n_magpie_features,
            'total_features': self.feature_dim,
        }

    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of superconductor categories."""
        if self.categories is None:
            return {}
        from collections import Counter
        return dict(Counter(self.categories))

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_state: int = 42,
        stratify_by_category: bool = False
    ) -> Tuple['FeaturizedSuperconductorDataset', 'FeaturizedSuperconductorDataset', 'FeaturizedSuperconductorDataset']:
        """Split dataset into train/val/test."""
        np.random.seed(random_state)
        n = len(self)

        if stratify_by_category and self.categories is not None:
            # Stratified split by category
            from sklearn.model_selection import train_test_split
            indices = np.arange(n)
            train_idx, temp_idx = train_test_split(
                indices, train_size=train_ratio,
                stratify=self.categories, random_state=random_state
            )
            val_size = val_ratio / (1 - train_ratio)
            temp_categories = [self.categories[i] for i in temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx, train_size=val_size,
                stratify=temp_categories, random_state=random_state
            )
        else:
            indices = np.random.permutation(n)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]

        def make_subset(idx):
            formulas = [self.formulas[i] for i in idx]
            tc_values = self.tc_values_raw[idx]
            magpie = self.magpie_features_raw[idx] if self.magpie_features_raw is not None else None
            categories = [self.categories[i] for i in idx] if self.categories else None

            return FeaturizedSuperconductorDataset(
                formulas=formulas,
                tc_values=tc_values,
                magpie_features=magpie,
                categories=categories,
                encoder=self.encoder,
                use_composition_encoding=self.use_composition_encoding,
                normalize_tc=self.normalize_tc,
                tc_log_transform=self.tc_log_transform,
                normalize_features=self.normalize_features,
                device=self.device
            )

        return make_subset(train_idx), make_subset(val_idx), make_subset(test_idx)

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        use_magpie_features: bool = True,
        use_composition_encoding: bool = True,
        formula_column: str = 'formula',
        tc_column: str = 'Tc',
        category_column: str = 'category',
        **kwargs
    ) -> 'FeaturizedSuperconductorDataset':
        """
        Load from pre-featurized CSV.

        Args:
            path: Path to CSV file
            use_magpie_features: Whether to use Magpie features
            use_composition_encoding: Whether to add composition encoding
            formula_column: Column for formulas
            tc_column: Column for Tc values
            category_column: Column for categories
            **kwargs: Additional arguments

        Returns:
            FeaturizedSuperconductorDataset instance
        """
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")

        # Extract formulas and Tc
        formulas = df[formula_column].tolist()
        tc_values = df[tc_column].values

        # Extract categories if available
        categories = None
        if category_column in df.columns:
            categories = df[category_column].tolist()

        # Extract Magpie features
        magpie_features = None
        if use_magpie_features:
            # Get all numeric columns that aren't metadata
            exclude_cols = {formula_column, tc_column, category_column, 'composition'}
            exclude_cols.update(cls.NON_FEATURE_COLUMNS)

            feature_cols = [
                col for col in df.columns
                if col not in exclude_cols
                and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            ]

            if feature_cols:
                magpie_features = df[feature_cols].values.astype(np.float32)
                print(f"  Using {len(feature_cols)} Magpie features")

        # Filter invalid entries
        valid_mask = ~np.isnan(tc_values) & (tc_values >= 0)
        formulas = [f for f, v in zip(formulas, valid_mask) if v]
        tc_values = tc_values[valid_mask]
        if magpie_features is not None:
            magpie_features = magpie_features[valid_mask]
        if categories is not None:
            categories = [c for c, v in zip(categories, valid_mask) if v]

        print(f"  Valid samples: {len(formulas)}")

        return cls(
            formulas=formulas,
            tc_values=tc_values,
            magpie_features=magpie_features,
            categories=categories,
            use_composition_encoding=use_composition_encoding,
            **kwargs
        )


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning with negative samples.

    Provides batches of:
    - Superconductors (positive samples)
    - Non-superconductors (negative samples)
    - Magnetic materials (hard negatives)

    This enables learning representations that capture
    what makes materials superconduct.
    """

    def __init__(
        self,
        superconductor_dataset: SuperconductorDataset,
        negative_formulas: Optional[List[str]] = None,
        magnetic_formulas: Optional[List[str]] = None,
        encoder: Optional[CompositionEncoder] = None,
        negative_ratio: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize contrastive dataset.

        Args:
            superconductor_dataset: Dataset of superconductors
            negative_formulas: List of non-superconductor formulas
            magnetic_formulas: List of magnetic material formulas (hard negatives)
            encoder: Chemical formula encoder
            negative_ratio: Ratio of negatives to positives per batch
            device: Torch device
        """
        self.sc_dataset = superconductor_dataset
        self.encoder = encoder or superconductor_dataset.encoder
        self.negative_ratio = negative_ratio
        self.device = device

        # Encode negative samples
        self.negative_features: Optional[torch.Tensor] = None
        self.magnetic_features: Optional[torch.Tensor] = None

        if negative_formulas:
            self._encode_negatives(negative_formulas, 'negative')

        if magnetic_formulas:
            self._encode_negatives(magnetic_formulas, 'magnetic')

    def _encode_negatives(self, formulas: List[str], neg_type: str):
        """Encode negative sample formulas."""
        features = []
        for formula in formulas:
            encoded = self.encoder.encode(formula)
            feature = torch.cat([
                encoded.composition_vector,
                encoded.element_stats
            ]).cpu()
            features.append(feature)

        tensor = torch.stack(features)
        if neg_type == 'negative':
            self.negative_features = tensor
            self.negative_formulas = formulas
        else:
            self.magnetic_features = tensor
            self.magnetic_formulas = formulas

    def __len__(self) -> int:
        return len(self.sc_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item with positive and negative samples.

        Returns:
            Dict with:
            - 'positive': superconductor features
            - 'tc': critical temperature
            - 'negative': negative sample features (if available)
            - 'magnetic': magnetic material features (if available)
        """
        features, tc = self.sc_dataset[idx]

        result = {
            'positive': features,
            'tc': tc,
        }

        # Sample random negatives
        if self.negative_features is not None:
            neg_idx = np.random.randint(0, len(self.negative_features))
            result['negative'] = self.negative_features[neg_idx]

        if self.magnetic_features is not None:
            mag_idx = np.random.randint(0, len(self.magnetic_features))
            result['magnetic'] = self.magnetic_features[mag_idx]

        return result

    def get_contrastive_batch(
        self,
        batch_size: int,
        n_negatives: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get batch for contrastive learning.

        Args:
            batch_size: Number of superconductors
            n_negatives: Number of negative samples (default: batch_size * negative_ratio)

        Returns:
            Dict with batched tensors
        """
        if n_negatives is None:
            n_negatives = int(batch_size * self.negative_ratio)

        # Sample superconductors
        sc_indices = np.random.choice(len(self.sc_dataset), batch_size, replace=False)
        sc_features = self.sc_dataset.features_tensor[sc_indices]
        sc_tc = self.sc_dataset.tc_values[sc_indices]

        result = {
            'positive_features': sc_features,
            'positive_tc': sc_tc,
        }

        # Sample negatives
        if self.negative_features is not None and n_negatives > 0:
            neg_indices = np.random.choice(
                len(self.negative_features),
                min(n_negatives, len(self.negative_features)),
                replace=False
            )
            result['negative_features'] = self.negative_features[neg_indices]

        if self.magnetic_features is not None:
            n_magnetic = n_negatives // 2  # Half magnetic, half regular negative
            mag_indices = np.random.choice(
                len(self.magnetic_features),
                min(n_magnetic, len(self.magnetic_features)),
                replace=False
            )
            result['magnetic_features'] = self.magnetic_features[mag_indices]

        return result

    @classmethod
    def from_materials_project(
        cls,
        sc_dataset: SuperconductorDataset,
        api_key: Optional[str] = None,
        n_negatives: int = 1000,
        n_magnetic: int = 500,
        **kwargs
    ) -> 'ContrastiveDataset':
        """
        Create contrastive dataset with Materials Project data.

        Args:
            sc_dataset: Superconductor dataset
            api_key: Materials Project API key
            n_negatives: Number of non-superconductor samples
            n_magnetic: Number of magnetic material samples
            **kwargs: Additional arguments

        Returns:
            ContrastiveDataset instance
        """
        # Consult with James: Materials Project API integration
        # This requires the mp-api package and a valid API key
        # For now, return dataset without API-fetched negatives

        print("Note: Materials Project API integration not yet implemented.")
        print("Returning dataset without external negative samples.")
        print("To add negatives, provide negative_formulas parameter.")

        return cls(
            superconductor_dataset=sc_dataset,
            **kwargs
        )
