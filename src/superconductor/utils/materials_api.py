"""
Materials Project API integration for superconductor discovery.

Provides access to:
- Non-superconductor materials (negative samples for contrastive learning)
- Magnetic materials (hard negatives)
- Formation energies and stability data
- Crystal structures

API Key Setup:
    Option 1: Environment variable
        export MP_API_KEY="your_key_here"

    Option 2: Config file at ~/.mprc
        api_key = your_key_here

    Option 3: Pass directly to functions
"""

import os
import warnings
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

import numpy as np


def get_api_key(api_key: Optional[str] = None) -> str:
    """
    Get Materials Project API key from various sources.

    Priority:
    1. Passed argument
    2. Environment variable MP_API_KEY
    3. Config file ~/.mprc
    4. Project config file

    Args:
        api_key: Optional API key to use directly

    Returns:
        API key string

    Raises:
        ValueError: If no API key found
    """
    # 1. Direct argument
    if api_key:
        return api_key

    # 2. Environment variable
    env_key = os.environ.get('MP_API_KEY')
    if env_key:
        return env_key

    # 3. Home config file
    home_config = Path.home() / '.mprc'
    if home_config.exists():
        with open(home_config) as f:
            for line in f:
                if line.strip().startswith('api_key'):
                    return line.split('=')[1].strip()

    # 4. Project config
    project_config = Path(__file__).parent.parent.parent.parent.parent / 'config' / 'materials_api.json'
    if project_config.exists():
        with open(project_config) as f:
            config = json.load(f)
            if 'mp_api_key' in config:
                return config['mp_api_key']

    raise ValueError(
        "No Materials Project API key found. Set MP_API_KEY environment variable "
        "or create ~/.mprc with 'api_key = YOUR_KEY'"
    )


class MaterialsProjectClient:
    """
    Client for Materials Project API.

    Example:
        client = MaterialsProjectClient(api_key="your_key")
        insulators = client.get_non_superconductors(n_samples=1000)
        magnetic = client.get_magnetic_materials(n_samples=500)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize client.

        Args:
            api_key: Materials Project API key
        """
        self.api_key = get_api_key(api_key)
        self._client = None
        self._check_mp_api()

    def _check_mp_api(self):
        """Check if mp-api is installed and initialize client."""
        try:
            from mp_api.client import MPRester
            self._client = MPRester(self.api_key)
            print("Materials Project API initialized successfully")
        except ImportError:
            warnings.warn(
                "mp-api package not installed. Install with: pip install mp-api"
            )
            self._client = None

    @property
    def is_available(self) -> bool:
        """Check if API is available."""
        return self._client is not None

    def get_non_superconductors(
        self,
        n_samples: int = 1000,
        exclude_magnetic: bool = True,
        band_gap_min: float = 0.5,
        random_state: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Get non-superconducting materials (insulators/semiconductors).

        Args:
            n_samples: Number of samples to retrieve
            exclude_magnetic: Exclude magnetic materials
            band_gap_min: Minimum band gap (eV) to ensure insulating
            random_state: Random seed for sampling

        Returns:
            List of material dictionaries with formula and properties
        """
        if not self.is_available:
            print("Warning: Materials Project API not available")
            return []

        try:
            # Query for insulators/semiconductors
            results = self._client.materials.summary.search(
                band_gap=(band_gap_min, None),  # Band gap > min
                is_metal=False,
                fields=['formula_pretty', 'band_gap', 'formation_energy_per_atom',
                        'energy_above_hull', 'is_magnetic']
            )

            # Filter magnetic if requested
            if exclude_magnetic:
                results = [r for r in results if not r.is_magnetic]

            # Sample randomly
            np.random.seed(random_state)
            if len(results) > n_samples:
                indices = np.random.choice(len(results), n_samples, replace=False)
                results = [results[i] for i in indices]

            # Convert to dictionaries
            materials = []
            for r in results:
                materials.append({
                    'formula': r.formula_pretty,
                    'band_gap': r.band_gap,
                    'formation_energy': r.formation_energy_per_atom,
                    'energy_above_hull': r.energy_above_hull,
                    'is_superconductor': False,
                    'source': 'materials_project'
                })

            print(f"Retrieved {len(materials)} non-superconductor materials")
            return materials

        except Exception as e:
            print(f"Error querying Materials Project: {e}")
            return []

    def get_magnetic_materials(
        self,
        n_samples: int = 500,
        min_magnetization: float = 0.5,
        random_state: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Get magnetic materials (hard negatives for contrastive learning).

        Magnetic ordering typically competes with superconductivity.

        Args:
            n_samples: Number of samples
            min_magnetization: Minimum magnetization (μB/atom)
            random_state: Random seed

        Returns:
            List of magnetic material dictionaries
        """
        if not self.is_available:
            print("Warning: Materials Project API not available")
            return []

        try:
            # Query for materials with significant magnetization
            # Note: Newer mp-api doesn't have is_magnetic filter, so we query
            # for materials and filter by total_magnetization
            results = self._client.materials.summary.search(
                total_magnetization=(min_magnetization, None),
                fields=['formula_pretty', 'total_magnetization',
                        'formation_energy_per_atom', 'energy_above_hull']
            )

            # Filter by magnetization (in case API filter didn't work)
            results = [
                r for r in results
                if r.total_magnetization and r.total_magnetization > min_magnetization
            ]

            # Sample
            np.random.seed(random_state)
            if len(results) > n_samples:
                indices = np.random.choice(len(results), n_samples, replace=False)
                results = [results[i] for i in indices]

            materials = []
            for r in results:
                materials.append({
                    'formula': r.formula_pretty,
                    'magnetization': r.total_magnetization,
                    'formation_energy': r.formation_energy_per_atom,
                    'energy_above_hull': r.energy_above_hull,
                    'is_magnetic': True,
                    'is_superconductor': False,
                    'source': 'materials_project'
                })

            print(f"Retrieved {len(materials)} magnetic materials")
            return materials

        except Exception as e:
            print(f"Error querying Materials Project: {e}")
            return []

    def get_stability_data(
        self,
        formulas: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get thermodynamic stability data for formulas.

        Args:
            formulas: List of chemical formulas

        Returns:
            Dictionary mapping formula to stability data
        """
        if not self.is_available:
            return {}

        stability = {}
        for formula in formulas:
            try:
                results = self._client.materials.summary.search(
                    formula=formula,
                    fields=['formula_pretty', 'formation_energy_per_atom',
                            'energy_above_hull']
                )

                if results:
                    r = results[0]
                    stability[formula] = {
                        'formation_energy': r.formation_energy_per_atom,
                        'energy_above_hull': r.energy_above_hull,
                        'is_stable': r.energy_above_hull < 0.05  # Within 50 meV/atom
                    }
            except Exception:
                continue

        return stability

    def validate_candidates(
        self,
        formulas: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Validate candidate formulas against Materials Project.

        Checks if similar compositions exist and their stability.

        Args:
            formulas: List of candidate formulas

        Returns:
            List of validation results
        """
        if not self.is_available:
            return [{'formula': f, 'mp_validated': False, 'reason': 'API unavailable'}
                    for f in formulas]

        results = []
        for formula in formulas:
            try:
                # Search for exact or similar compositions
                mp_results = self._client.materials.summary.search(
                    formula=formula,
                    fields=['formula_pretty', 'formation_energy_per_atom',
                            'energy_above_hull', 'structure']
                )

                if mp_results:
                    r = mp_results[0]
                    results.append({
                        'formula': formula,
                        'mp_validated': True,
                        'exists_in_mp': True,
                        'formation_energy': r.formation_energy_per_atom,
                        'energy_above_hull': r.energy_above_hull,
                        'is_stable': r.energy_above_hull < 0.05,
                    })
                else:
                    results.append({
                        'formula': formula,
                        'mp_validated': True,
                        'exists_in_mp': False,
                        'reason': 'No matching composition in Materials Project'
                    })

            except Exception as e:
                results.append({
                    'formula': formula,
                    'mp_validated': False,
                    'reason': str(e)
                })

        return results


def fetch_negative_samples(
    api_key: Optional[str] = None,
    n_insulators: int = 1000,
    n_magnetic: int = 500,
    output_path: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Convenience function to fetch negative samples for contrastive learning.

    Args:
        api_key: Materials Project API key
        n_insulators: Number of insulator samples
        n_magnetic: Number of magnetic samples
        output_path: Optional path to save results

    Returns:
        Tuple of (insulator_formulas, magnetic_formulas)
    """
    client = MaterialsProjectClient(api_key)

    if not client.is_available:
        print("Materials Project API not available.")
        print("Install mp-api: pip install mp-api")
        return [], []

    insulators = client.get_non_superconductors(n_samples=n_insulators)
    magnetic = client.get_magnetic_materials(n_samples=n_magnetic)

    insulator_formulas = [m['formula'] for m in insulators]
    magnetic_formulas = [m['formula'] for m in magnetic]

    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump({
                'insulators': insulators,
                'magnetic': magnetic
            }, f, indent=2)
        print(f"Saved to {output_path}")

    return insulator_formulas, magnetic_formulas


# Pre-compiled lists of common non-superconductors and magnetic materials
# Use these if API is not available
DEFAULT_NON_SUPERCONDUCTORS = [
    "SiO2", "Al2O3", "TiO2", "Fe2O3", "ZnO", "MgO", "CaO", "NaCl",
    "KCl", "CaCO3", "BaSO4", "NiO", "CoO", "MnO", "Cr2O3", "V2O5",
    "WO3", "MoO3", "ZrO2", "HfO2", "CeO2", "SnO2", "In2O3", "Ga2O3",
    "GeO2", "PbO", "PbO2", "Bi2O3", "Sb2O3", "As2O3", "P2O5",
    "Si3N4", "AlN", "GaN", "InN", "BN", "TiN", "ZrN", "HfN",
    "GaAs", "InAs", "GaP", "InP", "GaSb", "InSb", "ZnS", "ZnSe",
    "CdS", "CdSe", "CdTe", "PbS", "PbSe", "PbTe", "SnS", "SnSe",
]

DEFAULT_MAGNETIC_MATERIALS = [
    "Fe3O4", "Fe2O3", "CoFe2O4", "NiFe2O4", "MnFe2O4", "ZnFe2O4",
    "BaFe12O19", "SrFe12O19", "Y3Fe5O12", "Gd3Fe5O12", "SmCo5",
    "Nd2Fe14B", "AlNiCo", "FePt", "CoPt", "FePd", "MnBi",
    "CrO2", "EuO", "EuS", "GdN", "MnAs", "MnSb", "MnBi",
    "La0.7Sr0.3MnO3", "La0.7Ca0.3MnO3", "Pr0.7Ca0.3MnO3",
    "Fe", "Co", "Ni", "Gd", "Dy", "Tb", "Ho", "Er",
]
