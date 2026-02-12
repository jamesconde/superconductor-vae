#!/usr/bin/env python3
"""
Add Physics Feature Columns to Existing CSV

Backfills 6 physics-informed feature columns to supercon_fractions_contrastive.csv
without re-running the full Magpie pipeline. These features encode domain knowledge
about superconductor families.

New columns:
  - has_cuprate_elements: Cu AND O present (0/1)
  - cu_o_ratio: Cu fraction / O fraction (float, 0 if no Cu or O)
  - has_iron_pnictide: Fe AND (As/Se/P) present (0/1)
  - has_mgb2_elements: Mg AND B present (0/1)
  - hydrogen_fraction: H fraction in composition (float)
  - transition_metal_count: number of distinct transition metal elements (int)

Usage:
  python scripts/add_physics_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_contrastive.csv"

# Transition metals (groups 3-12, periods 4-6)
TRANSITION_METALS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
}

PHYSICS_COLS = [
    "has_cuprate_elements",
    "cu_o_ratio",
    "has_iron_pnictide",
    "has_mgb2_elements",
    "hydrogen_fraction",
    "transition_metal_count",
]


def compute_physics_features(comp) -> dict:
    """Compute 6 physics features from a pymatgen Composition."""
    elements = {str(e) for e in comp.elements}
    el_fracs = comp.fractional_composition.as_dict()

    has_cu = "Cu" in elements
    has_o = "O" in elements
    cu_frac = el_fracs.get("Cu", 0.0)
    o_frac = el_fracs.get("O", 0.0)

    return {
        "has_cuprate_elements": int(has_cu and has_o),
        "cu_o_ratio": cu_frac / o_frac if (has_cu and o_frac > 0) else 0.0,
        "has_iron_pnictide": int("Fe" in elements and bool(elements & {"As", "Se", "P"})),
        "has_mgb2_elements": int("Mg" in elements and "B" in elements),
        "hydrogen_fraction": el_fracs.get("H", 0.0),
        "transition_metal_count": len(elements & TRANSITION_METALS),
    }


def main():
    from pymatgen.core import Composition

    print(f"Loading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # Check if physics columns already exist
    existing_physics = [col for col in PHYSICS_COLS if col in df.columns]
    if existing_physics:
        print(f"  Physics columns already present: {existing_physics}")
        response = input("  Overwrite existing physics columns? [y/N]: ").strip().lower()
        if response != "y":
            print("  Aborting.")
            sys.exit(0)
        # Drop existing physics columns
        df = df.drop(columns=existing_physics)

    # Compute physics features
    print(f"  Computing physics features for {len(df)} rows...")
    physics_data = []
    n_failed = 0
    for comp_str in tqdm(df["composition"], desc="Physics features"):
        try:
            comp = Composition(str(comp_str))
            physics_data.append(compute_physics_features(comp))
        except Exception:
            n_failed += 1
            physics_data.append({col: 0 for col in PHYSICS_COLS})

    if n_failed > 0:
        print(f"  WARNING: {n_failed} rows failed composition parsing, defaulting to 0")

    physics_df = pd.DataFrame(physics_data)

    # Insert physics columns before formula_original
    # Find insertion point
    if "formula_original" in df.columns:
        insert_idx = df.columns.get_loc("formula_original")
    elif "requires_high_pressure" in df.columns:
        insert_idx = df.columns.get_loc("requires_high_pressure")
    else:
        insert_idx = len(df.columns)

    # Insert each column
    for i, col in enumerate(PHYSICS_COLS):
        df.insert(insert_idx + i, col, physics_data_col(physics_df, col))

    print(f"  Added {len(PHYSICS_COLS)} physics columns")
    print(f"  New column count: {len(df.columns)}")

    # Verify
    for col in PHYSICS_COLS:
        assert col in df.columns, f"Missing column: {col}"
        vals = df[col]
        print(f"    {col}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

    # Save
    df.to_csv(CSV_PATH, index=False)
    print(f"\n  Saved updated CSV: {CSV_PATH}")
    print(f"  Total rows: {len(df)}, Total columns: {len(df.columns)}")


def physics_data_col(physics_df, col):
    """Extract a column from physics DataFrame."""
    return physics_df[col].values


if __name__ == "__main__":
    main()
