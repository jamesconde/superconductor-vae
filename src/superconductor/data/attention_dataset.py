"""
Attention-compatible dataset for superconductor data.

Returns element-level features suitable for FullMaterialsVAE:
- element_indices: Atomic numbers for each element in composition
- element_fractions: Molar fractions
- element_mask: Valid element mask
- isotope_features: Aggregated isotope features
- tc: Critical temperature (normalized)

Example:
    dataset = AttentionSuperconductorDataset.from_csv('supercon.csv')
    batch = dataset[0]
    # batch contains: element_indices, element_fractions, element_mask, isotope_features, tc
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from ..encoders.isotope_encoder import IsotopeEncoder


class AttentionSuperconductorDataset(Dataset):
    """
    Dataset that returns element-level features for attention models.

    Unlike FeaturizedSuperconductorDataset which returns Magpie features,
    this returns raw element information that the attention model can learn from.
    """

    def __init__(
        self,
        formulas: List[str],
        tc_values: np.ndarray,
        max_elements: int = 20,
        normalize_tc: bool = True,
        tc_mean: Optional[float] = None,
        tc_std: Optional[float] = None
    ):
        """
        Args:
            formulas: List of chemical formula strings
            tc_values: Array of Tc values in Kelvin
            max_elements: Maximum elements per formula (for padding)
            normalize_tc: Whether to normalize Tc values
            tc_mean: Mean for normalization (computed if None)
            tc_std: Std for normalization (computed if None)
        """
        self.formulas = formulas
        self.tc_values_raw = tc_values.copy()
        self.max_elements = max_elements

        # Normalize Tc
        if normalize_tc:
            self.tc_mean = tc_mean if tc_mean is not None else np.mean(tc_values)
            self.tc_std = tc_std if tc_std is not None else np.std(tc_values)
            self.tc_values = (tc_values - self.tc_mean) / (self.tc_std + 1e-8)
        else:
            self.tc_mean = 0.0
            self.tc_std = 1.0
            self.tc_values = tc_values

        # Create isotope encoder
        self.encoder = IsotopeEncoder()

        # Pre-encode all formulas
        self._precompute_encodings()

    def _precompute_encodings(self):
        """Pre-encode all formulas for faster training."""
        batch = self.encoder.encode_batch(self.formulas, self.max_elements)

        self.element_indices = batch['element_indices']
        self.element_fractions = batch['element_fractions']
        self.element_masses = batch['element_masses']
        self.element_spins = batch['element_spins']
        self.element_mask = batch['element_mask']
        self.composition_vectors = batch['composition_vector']
        self.isotope_features = batch['isotope_features']

    def __len__(self) -> int:
        return len(self.formulas)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dict with keys: element_indices, element_fractions, element_mask,
                           isotope_features, tc, tc_raw, formula
        """
        return {
            'element_indices': self.element_indices[idx],
            'element_fractions': self.element_fractions[idx],
            'element_mask': self.element_mask[idx],
            'isotope_features': self.isotope_features[idx],
            'composition_vector': self.composition_vectors[idx],
            'tc': torch.tensor(self.tc_values[idx], dtype=torch.float32),
            'tc_raw': torch.tensor(self.tc_values_raw[idx], dtype=torch.float32),
        }

    def get_collate_fn(self):
        """Get collate function for DataLoader."""
        def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            return {
                'element_indices': torch.stack([b['element_indices'] for b in batch]),
                'element_fractions': torch.stack([b['element_fractions'] for b in batch]),
                'element_mask': torch.stack([b['element_mask'] for b in batch]),
                'isotope_features': torch.stack([b['isotope_features'] for b in batch]),
                'composition_vector': torch.stack([b['composition_vector'] for b in batch]),
                'tc': torch.stack([b['tc'] for b in batch]),
                'tc_raw': torch.stack([b['tc_raw'] for b in batch]),
            }
        return collate

    def denormalize_tc(self, tc_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized Tc back to Kelvin."""
        return tc_normalized * self.tc_std + self.tc_mean

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        formula_column: str = 'formula',
        tc_column: str = 'Tc',
        max_elements: int = 20,
        normalize_tc: bool = True
    ) -> 'AttentionSuperconductorDataset':
        """
        Load dataset from CSV file.

        Args:
            csv_path: Path to CSV file
            formula_column: Column containing chemical formulas
            tc_column: Column containing Tc values
            max_elements: Maximum elements per formula
            normalize_tc: Whether to normalize Tc

        Returns:
            AttentionSuperconductorDataset instance
        """
        df = pd.read_csv(csv_path)

        # Get formulas and Tc values
        formulas = df[formula_column].tolist()
        tc_values = df[tc_column].values.astype(np.float32)

        # Filter out invalid entries
        valid_mask = ~np.isnan(tc_values)
        formulas = [f for f, v in zip(formulas, valid_mask) if v]
        tc_values = tc_values[valid_mask]

        print(f"Loaded {len(formulas)} samples from {csv_path}")

        return cls(
            formulas=formulas,
            tc_values=tc_values,
            max_elements=max_elements,
            normalize_tc=normalize_tc
        )

    @classmethod
    def from_supercon(
        cls,
        data_dir: str = 'data/superconductor/raw',
        filename: str = 'supercon.csv',
        **kwargs
    ) -> 'AttentionSuperconductorDataset':
        """
        Load from SuperCon dataset.

        Args:
            data_dir: Directory containing the data
            filename: CSV filename
            **kwargs: Additional arguments to from_csv

        Returns:
            AttentionSuperconductorDataset instance
        """
        csv_path = Path(data_dir) / filename
        return cls.from_csv(str(csv_path), **kwargs)


def create_attention_dataloaders(
    dataset: AttentionSuperconductorDataset,
    batch_size: int = 64,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        dataset: AttentionSuperconductorDataset
        batch_size: Batch size
        train_split: Fraction for training
        val_split: Fraction for validation
        seed: Random seed

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, Subset

    n_samples = len(dataset)
    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    collate_fn = dataset.get_collate_fn()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
