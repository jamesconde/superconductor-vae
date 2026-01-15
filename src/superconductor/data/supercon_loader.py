"""
SuperCon database loader.

The SuperCon database (https://supercon.nims.go.jp/) is a comprehensive
database of superconducting materials maintained by NIMS (Japan).

This loader handles various SuperCon export formats and data cleaning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import re
import warnings

from .dataset import SuperconductorDataset


class SuperConLoader:
    """
    Loader for SuperCon database files.

    Handles data cleaning, formula standardization, and filtering.

    Example:
        loader = SuperConLoader('path/to/supercon_data.csv')
        dataset = loader.load()
        print(dataset.get_tc_statistics())
    """

    # Common column name mappings for SuperCon exports
    COLUMN_MAPPINGS = {
        'formula': ['formula', 'Formula', 'FORMULA', 'composition', 'Composition',
                    'material', 'Material', 'name', 'Name', 'compound', 'Compound'],
        'tc': ['tc', 'Tc', 'TC', 'critical_temperature', 'Critical_Temperature',
               'Tc_K', 'tc_k', 'T_c', 't_c', 'Tc(K)', 'tc(K)'],
        'family': ['family', 'Family', 'type', 'Type', 'category', 'Category',
                   'class', 'Class', 'superconductor_type'],
        'structure': ['structure', 'Structure', 'crystal_structure', 'structure_type',
                      'space_group', 'Space_Group', 'spacegroup'],
        'year': ['year', 'Year', 'discovery_year', 'Discovery_Year', 'date', 'Date'],
        'reference': ['reference', 'Reference', 'ref', 'Ref', 'doi', 'DOI', 'source'],
    }

    # Superconductor family classifications
    SUPERCONDUCTOR_FAMILIES = {
        'cuprate': ['YBCO', 'BSCCO', 'LSCO', 'TBCCO', 'HBCCO', 'cuprate', 'Cuprate'],
        'iron_based': ['FeAs', 'FeSe', 'iron', 'Iron', 'pnictide', 'Pnictide'],
        'conventional': ['BCS', 'conventional', 'Conventional', 'elemental', 'Elemental'],
        'heavy_fermion': ['heavy', 'Heavy', 'Ce', 'U', 'fermion', 'Fermion'],
        'organic': ['organic', 'Organic', 'BEDT', 'TTF'],
        'MgB2': ['MgB', 'Mg-B', 'magnesium diboride'],
        'hydride': ['H', 'hydride', 'Hydride', 'hydrogen'],
        'A15': ['A15', 'Nb3Sn', 'Nb3Ge', 'V3Si'],
        'chevrel': ['Chevrel', 'Mo6', 'PbMo'],
        'other': []
    }

    def __init__(
        self,
        path: Union[str, Path],
        verbose: bool = True
    ):
        """
        Initialize SuperCon loader.

        Args:
            path: Path to SuperCon data file
            verbose: Whether to print loading information
        """
        self.path = Path(path)
        self.verbose = verbose
        self.df: Optional[pd.DataFrame] = None
        self.column_map: Dict[str, str] = {}

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def _load_file(self) -> pd.DataFrame:
        """Load data file based on extension."""
        suffix = self.path.suffix.lower()

        if suffix == '.csv':
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return pd.read_csv(self.path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode CSV file with any common encoding")

        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(self.path)

        elif suffix == '.json':
            return pd.read_json(self.path)

        elif suffix == '.tsv':
            return pd.read_csv(self.path, sep='\t')

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _identify_columns(self) -> Dict[str, str]:
        """Identify columns in the dataframe."""
        column_map = {}

        for target, candidates in self.COLUMN_MAPPINGS.items():
            for candidate in candidates:
                if candidate in self.df.columns:
                    column_map[target] = candidate
                    break

        return column_map

    def _clean_formula(self, formula: str) -> str:
        """Clean and standardize chemical formula."""
        if pd.isna(formula):
            return ""

        formula = str(formula).strip()

        # Remove common annotations
        formula = re.sub(r'\s*\(.*?\)\s*$', '', formula)  # Remove trailing parenthetical
        formula = re.sub(r'\s+', '', formula)  # Remove whitespace

        # Standardize doping notation
        formula = re.sub(r'[-−–]\s*[δxyzn]\s*$', '', formula, flags=re.IGNORECASE)

        return formula

    def _clean_tc(self, tc_value: Any) -> Optional[float]:
        """Clean and validate Tc value."""
        if pd.isna(tc_value):
            return None

        # Handle string values
        if isinstance(tc_value, str):
            # Remove units
            tc_str = re.sub(r'\s*[Kk]\s*$', '', tc_value.strip())
            # Handle ranges (take midpoint)
            if '-' in tc_str or '–' in tc_str:
                parts = re.split(r'[-–]', tc_str)
                try:
                    values = [float(p.strip()) for p in parts if p.strip()]
                    if values:
                        return sum(values) / len(values)
                except ValueError:
                    return None
            # Handle "~" or "approximately"
            tc_str = re.sub(r'^[~≈]', '', tc_str)
            try:
                return float(tc_str)
            except ValueError:
                return None

        # Handle numeric
        try:
            tc = float(tc_value)
            if tc < 0 or tc > 500:  # Sanity check (no known superconductor above ~250K)
                return None
            return tc
        except (ValueError, TypeError):
            return None

    def _classify_family(self, formula: str, row: pd.Series) -> str:
        """Classify superconductor into family based on formula and metadata."""
        formula_upper = formula.upper()

        # Check explicit family column first
        if 'family' in self.column_map:
            family_val = row.get(self.column_map['family'], '')
            if pd.notna(family_val):
                family_str = str(family_val).lower()
                for family, keywords in self.SUPERCONDUCTOR_FAMILIES.items():
                    for keyword in keywords:
                        if keyword.lower() in family_str:
                            return family

        # Classify by formula patterns
        # Cuprates: contain Cu-O and alkaline earth/rare earth
        if 'CU' in formula_upper and 'O' in formula_upper:
            if any(x in formula_upper for x in ['BA', 'SR', 'CA', 'LA', 'Y', 'BI', 'TL', 'HG']):
                return 'cuprate'

        # Iron-based: contain Fe and pnictogen/chalcogen
        if 'FE' in formula_upper:
            if any(x in formula_upper for x in ['AS', 'P', 'SE', 'TE', 'S']):
                return 'iron_based'

        # MgB2 type
        if 'MG' in formula_upper and 'B' in formula_upper:
            return 'MgB2'

        # A15 compounds
        if any(x in formula_upper for x in ['NB3', 'V3', 'TA3']):
            return 'A15'

        # Hydrides
        if 'H' in formula_upper and formula_upper.count('H') > 2:
            return 'hydride'

        # Heavy fermion (Ce or U compounds)
        if any(x in formula_upper for x in ['CE', 'U', 'YB', 'PR']):
            if any(x in formula_upper for x in ['CU', 'SI', 'GE', 'IN', 'SN']):
                return 'heavy_fermion'

        # Elemental superconductors
        if re.match(r'^[A-Z][a-z]?$', formula):
            return 'conventional'

        return 'other'

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries, keeping highest Tc for each formula."""
        formula_col = self.column_map['formula']
        tc_col = self.column_map['tc']

        # Sort by Tc descending and keep first (highest) for each formula
        df_sorted = df.sort_values(tc_col, ascending=False)
        df_dedup = df_sorted.drop_duplicates(subset=[formula_col], keep='first')

        n_removed = len(df) - len(df_dedup)
        if n_removed > 0:
            self._log(f"  Removed {n_removed} duplicate entries")

        return df_dedup

    def load(
        self,
        min_tc: float = 0.0,
        max_tc: float = 500.0,
        remove_duplicates: bool = True,
        classify_families: bool = True,
        **kwargs
    ) -> SuperconductorDataset:
        """
        Load and process SuperCon data.

        Args:
            min_tc: Minimum Tc to include
            max_tc: Maximum Tc to include
            remove_duplicates: Whether to remove duplicate formulas
            classify_families: Whether to classify into superconductor families
            **kwargs: Additional arguments for SuperconductorDataset

        Returns:
            SuperconductorDataset instance
        """
        self._log(f"Loading SuperCon data from {self.path}")

        # Load file
        self.df = self._load_file()
        self._log(f"  Loaded {len(self.df)} rows")

        # Identify columns
        self.column_map = self._identify_columns()
        if 'formula' not in self.column_map:
            raise ValueError(f"Could not find formula column. Available: {self.df.columns.tolist()}")
        if 'tc' not in self.column_map:
            raise ValueError(f"Could not find Tc column. Available: {self.df.columns.tolist()}")

        self._log(f"  Formula column: {self.column_map['formula']}")
        self._log(f"  Tc column: {self.column_map['tc']}")

        # Clean data
        formula_col = self.column_map['formula']
        tc_col = self.column_map['tc']

        self.df['_clean_formula'] = self.df[formula_col].apply(self._clean_formula)
        self.df['_clean_tc'] = self.df[tc_col].apply(self._clean_tc)

        # Filter valid entries
        valid_mask = (
            (self.df['_clean_formula'] != '') &
            (self.df['_clean_tc'].notna()) &
            (self.df['_clean_tc'] >= min_tc) &
            (self.df['_clean_tc'] <= max_tc)
        )
        df_valid = self.df[valid_mask].copy()
        self._log(f"  Valid entries: {len(df_valid)} ({len(self.df) - len(df_valid)} removed)")

        # Remove duplicates
        if remove_duplicates:
            # Update column map for cleaned columns
            orig_formula = self.column_map['formula']
            orig_tc = self.column_map['tc']
            self.column_map['formula'] = '_clean_formula'
            self.column_map['tc'] = '_clean_tc'
            df_valid = self._remove_duplicates(df_valid)
            self.column_map['formula'] = orig_formula
            self.column_map['tc'] = orig_tc

        # Classify families
        if classify_families:
            df_valid['_family'] = df_valid.apply(
                lambda row: self._classify_family(row['_clean_formula'], row),
                axis=1
            )
            family_counts = df_valid['_family'].value_counts()
            self._log(f"  Family distribution:")
            for family, count in family_counts.items():
                self._log(f"    {family}: {count}")

        # Extract final data
        formulas = df_valid['_clean_formula'].tolist()
        tc_values = df_valid['_clean_tc'].values

        self._log(f"  Final dataset: {len(formulas)} materials")
        self._log(f"  Tc range: {tc_values.min():.2f}K - {tc_values.max():.2f}K")

        # Create dataset
        dataset = SuperconductorDataset(
            formulas=formulas,
            tc_values=tc_values,
            **kwargs
        )

        return dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded data."""
        if self.df is None:
            return {}

        stats = {
            'total_rows': len(self.df),
            'columns': self.df.columns.tolist(),
            'identified_columns': self.column_map,
        }

        if '_clean_tc' in self.df.columns:
            valid_tc = self.df['_clean_tc'].dropna()
            stats['tc_stats'] = {
                'count': len(valid_tc),
                'min': valid_tc.min(),
                'max': valid_tc.max(),
                'mean': valid_tc.mean(),
                'median': valid_tc.median(),
            }

        if '_family' in self.df.columns:
            stats['family_distribution'] = self.df['_family'].value_counts().to_dict()

        return stats


def load_supercon(
    path: Union[str, Path],
    **kwargs
) -> SuperconductorDataset:
    """
    Convenience function to load SuperCon data.

    Args:
        path: Path to SuperCon data file
        **kwargs: Additional arguments for loader

    Returns:
        SuperconductorDataset instance
    """
    loader = SuperConLoader(path)
    return loader.load(**kwargs)
