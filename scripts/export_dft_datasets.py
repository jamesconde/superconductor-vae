#!/usr/bin/env python3
"""
Export DFT Superconductor Datasets for Future Theory Network Use

Reads JARVIS 3D, 2D, alex_supercon, and HTSC-2025 datasets, converts
formulas to fraction notation, and saves as a single CSV for future use.

These datasets contain DFT-computed (not experimental) Tc values and are
NOT merged into the training CSV. They are preserved for Phase 2 Theory
Network work (BCS parameters, Eliashberg spectral functions, etc.).

Datasets:
  - JARVIS supercon_3d (1,058 entries) — Eliashberg spectral functions
  - JARVIS supercon_2d (161 entries)
  - JARVIS alex_supercon (8,253 entries) — BCS parameters: lambda, wlog, Debye, DOS
  - HTSC-2025 (140 entries) — ambient-pressure predictions

Output:
  data/processed/dft_superconductors.csv

Usage:
  python scripts/export_dft_datasets.py
"""

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Dataset paths
JARVIS_3D = PROJECT_ROOT / "data" / "jarvis_hydrides" / "supercon_3d.json"
JARVIS_2D = PROJECT_ROOT / "data" / "jarvis_hydrides" / "supercon_2d.json"
JARVIS_ALEX = PROJECT_ROOT / "data" / "jarvis_hydrides" / "alex_supercon.json"
HTSC_2025 = PROJECT_ROOT / "data" / "HTSC2025_repo" / "HTSC-2025.json"

OUTPUT = PROJECT_ROOT / "data" / "processed" / "dft_superconductors.csv"


def extract_formula_from_atoms(atoms_dict: dict) -> str:
    """Extract formula from JARVIS atoms dictionary."""
    if not atoms_dict:
        return None
    # JARVIS atoms dicts have 'elements' and 'coords'
    if "elements" in atoms_dict:
        from collections import Counter
        elem_counts = Counter(atoms_dict["elements"])
        parts = []
        for el in sorted(elem_counts.keys()):
            count = elem_counts[el]
            if count == 1:
                parts.append(el)
            else:
                parts.append(f"{el}{count}")
        return "".join(parts)
    return None


def composition_to_fraction_formula(comp) -> str:
    """Convert pymatgen Composition to fraction notation."""
    parts = []
    for el, amt in sorted(comp.as_dict().items()):
        frac = Fraction(amt).limit_denominator(1000)
        if frac.denominator == 1:
            if frac.numerator == 1:
                parts.append(str(el))
            else:
                parts.append(f"{el}{frac.numerator}")
        else:
            parts.append(f"{el}({frac})")
    return "".join(parts)


def read_jarvis_3d(path: Path) -> pd.DataFrame:
    """Read JARVIS supercon_3d.json — 1058 entries with Eliashberg data."""
    print(f"Reading JARVIS 3D from {path}")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        formula = extract_formula_from_atoms(entry.get("atoms"))
        if formula is None:
            continue
        rows.append({
            "formula_raw": formula,
            "Tc": entry.get("Tc", 0),
            "tc_source": "dft",
            "dataset": "jarvis_3d",
            "lambda": entry.get("lamb"),
            "wlog": entry.get("wlog"),
            "debye_temp": None,
            "dos_ef": None,
            "pressure_gpa": None,
            "jid": entry.get("jid"),
        })

    print(f"  Parsed {len(rows)} entries")
    return pd.DataFrame(rows)


def read_jarvis_2d(path: Path) -> pd.DataFrame:
    """Read JARVIS supercon_2d.json — 161 entries."""
    print(f"Reading JARVIS 2D from {path}")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        formula = extract_formula_from_atoms(entry.get("atoms"))
        if formula is None:
            continue
        rows.append({
            "formula_raw": formula,
            "Tc": entry.get("Tc", 0),
            "tc_source": "dft",
            "dataset": "jarvis_2d",
            "lambda": entry.get("lamb"),
            "wlog": entry.get("wlog"),
            "debye_temp": None,
            "dos_ef": None,
            "pressure_gpa": entry.get("press"),
            "jid": entry.get("jid"),
        })

    print(f"  Parsed {len(rows)} entries")
    return pd.DataFrame(rows)


def read_jarvis_alex(path: Path) -> pd.DataFrame:
    """Read JARVIS alex_supercon.json — 8253 entries with BCS parameters."""
    print(f"Reading JARVIS alex_supercon from {path}")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        formula = extract_formula_from_atoms(entry.get("atoms"))
        if formula is None:
            continue
        rows.append({
            "formula_raw": formula,
            "Tc": entry.get("Tc", 0),
            "tc_source": "dft",
            "dataset": "jarvis_alex",
            "lambda": entry.get("la"),
            "wlog": entry.get("wlog"),
            "debye_temp": entry.get("debye"),
            "dos_ef": entry.get("dosef"),
            "pressure_gpa": None,
            "jid": None,
        })

    print(f"  Parsed {len(rows)} entries")
    return pd.DataFrame(rows)


def read_htsc_2025(path: Path) -> pd.DataFrame:
    """Read HTSC-2025.json — 140 ambient-pressure predictions."""
    print(f"Reading HTSC-2025 from {path}")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for key, entry in data.items():
        # key format: "M3XH8-NbAl3H8" — extract formula from after the dash
        parts = key.split("-", 1)
        formula = parts[1] if len(parts) > 1 else parts[0]

        rows.append({
            "formula_raw": formula,
            "Tc": entry.get("tc", 0),
            "tc_source": "dft_predicted",
            "dataset": "htsc_2025",
            "lambda": None,
            "wlog": None,
            "debye_temp": None,
            "dos_ef": None,
            "pressure_gpa": 0,  # ambient-pressure predictions
            "jid": None,
        })

    print(f"  Parsed {len(rows)} entries")
    return pd.DataFrame(rows)


def main():
    from pymatgen.core import Composition

    dfs = []

    if JARVIS_3D.exists():
        dfs.append(read_jarvis_3d(JARVIS_3D))
    if JARVIS_2D.exists():
        dfs.append(read_jarvis_2d(JARVIS_2D))
    if JARVIS_ALEX.exists():
        dfs.append(read_jarvis_alex(JARVIS_ALEX))
    if HTSC_2025.exists():
        dfs.append(read_htsc_2025(HTSC_2025))

    if not dfs:
        print("No DFT datasets found. Exiting.")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined DFT entries: {len(df)}")

    # Parse formulas and convert to fraction notation
    print("Parsing formulas with pymatgen...")
    formulas_frac = []
    valid_mask = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing"):
        try:
            comp = Composition(row["formula_raw"])
            formulas_frac.append(composition_to_fraction_formula(comp))
            valid_mask.append(True)
        except Exception:
            formulas_frac.append(None)
            valid_mask.append(False)

    df["formula"] = formulas_frac
    n_valid = sum(valid_mask)
    print(f"  Successfully parsed: {n_valid} / {len(df)}")

    # Drop unparseable
    df = df[df["formula"].notna()].copy()

    # Select output columns
    output_cols = [
        "formula", "Tc", "tc_source", "dataset",
        "pressure_gpa", "lambda", "wlog", "debye_temp", "dos_ef"
    ]
    df_out = df[output_cols].copy()

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df_out)} DFT entries to {OUTPUT}")

    # Summary
    print(f"\nPer-dataset counts:")
    for ds, count in df_out["dataset"].value_counts().items():
        print(f"  {ds}: {count}")

    print(f"\nTc distribution:")
    print(f"  Mean: {df_out['Tc'].mean():.1f} K")
    print(f"  Max: {df_out['Tc'].max():.1f} K")
    print(f"  Tc>0: {(df_out['Tc'] > 0).sum()}")
    print(f"  Tc>=100K: {(df_out['Tc'] >= 100).sum()}")


if __name__ == "__main__":
    main()
