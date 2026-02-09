#!/usr/bin/env python3
"""
High-Pressure Superconductor Labeling Script

Identifies high-pressure superconductors (HP-SC) in the processed dataset by:
1. Joining with NEMAD source data (has pressure column)
2. Supplementing with known HP-SC categories (hydrides already labeled, elemental HP, etc.)
3. Writing updated CSV with `requires_high_pressure` column (0/1)
4. Generating diagnostic report

Usage:
    cd /home/james/superconductor-vae
    PYTHONPATH=src python scripts/label_high_pressure.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEMAD_PATH = PROJECT_ROOT / 'DSC288 Data' / 'DSC288 Data' / 'NEMAD' / 'NEMAD_superconductor_materials_unique.csv'
DATASET_PATH = PROJECT_ROOT / 'data' / 'processed' / 'supercon_fractions_contrastive.csv'
REPORT_PATH = PROJECT_ROOT / 'outputs' / 'high_pressure_labeling_report.json'

# Pressure threshold in GPa — materials above this are labeled HP
HP_THRESHOLD_GPA = 1.0


def parse_pressure_to_gpa(val: str) -> float | None:
    """Parse a free-text pressure string to GPa.

    Handles: '100 GPa', '50.5GPa', '15 kbar', '1 atm', 'ambient', etc.

    Returns:
        Pressure in GPa, or None if unparseable/ambient.
    """
    if pd.isna(val):
        return None
    val = str(val).strip()
    if not val:
        return None

    val_lower = val.lower()

    # Ambient / atmospheric / normal pressure markers → 0 GPa (not HP)
    ambient_markers = [
        'ambient', 'atmospheric', 'normal pressure', '1 atm',
        'room pressure', 'atm', '0 gpa', '0 kbar',
    ]
    for marker in ambient_markers:
        if marker in val_lower:
            return 0.0

    # "High Pressure" without numeric value — treat as HP (conservative: 10 GPa)
    if 'high pressure' in val_lower:
        return 10.0

    # GPa
    m = re.match(r'([\d.]+)\s*GPa', val, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # kbar → GPa (1 kbar = 0.1 GPa)
    m = re.match(r'([\d.]+)\s*kbar', val, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 10.0

    # MPa → GPa (1 MPa = 0.001 GPa)
    m = re.match(r'([\d.]+)\s*MPa', val, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 1000.0

    # bar → GPa (1 bar ≈ 0.0001 GPa)
    m = re.match(r'([\d.]+)\s*bar\b', val, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 10000.0

    # mmHg → GPa (760 mmHg ≈ 0.000101325 GPa)
    m = re.match(r'([\d.]+)\s*mmHg', val, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 0.000101325 / 760.0

    # Bare number (assume GPa if large, skip if unclear)
    m = re.match(r'^([\d.]+)$', val)
    if m:
        v = float(m.group(1))
        if v > 0:
            return v  # Assume GPa for bare numbers
        return 0.0

    return None


def normalize_formula_for_matching(formula: str) -> str:
    """Normalize a chemical formula for matching between datasets.

    Strips whitespace, parentheses annotations, and sorts elements alphabetically.
    This handles cases like 'H3S' vs 'H3S1' and 'LaH10' vs 'La1H10'.
    """
    if pd.isna(formula):
        return ''
    formula = str(formula).strip()

    # Remove parenthetical annotations like (La,Al) → LaAl
    formula = re.sub(r'[()]', '', formula)

    # Remove spaces and hyphens used as separators
    formula = formula.replace(' ', '').replace('-', '')

    return formula.lower()


def extract_elements(formula: str) -> set:
    """Extract element symbols from a formula string."""
    if pd.isna(formula):
        return set()
    # Match element symbols (uppercase letter optionally followed by lowercase)
    elements = re.findall(r'[A-Z][a-z]?', str(formula))
    return set(elements)


def is_hydrogen_rich(formula: str) -> bool:
    """Check if formula is hydrogen-rich (contains H with significant fraction)."""
    elements = extract_elements(formula)
    return 'H' in elements


def is_fullerene_based(formula: str, tc: float) -> bool:
    """Check if formula is a fullerene/C60-based compound requiring high pressure.

    Most alkali-doped fullerides (K3C60, Rb3C60) are ambient-pressure SC.
    Only specific cases require pressure:
    - Cs3C60 in A15 structure (needs ~7 kbar)
    - Fullerides with Tc > 33K (generally need pressure to stabilize)
    """
    f = str(formula)
    if not re.search(r'C60|C70|C36|C28', f):
        return False

    # Cs3C60 with high Tc (A15 phase needs pressure)
    if 'Cs' in f and tc > 33:
        return True

    # Very high Tc fullerides likely need pressure stabilization
    if tc > 38:
        return True

    return False


def is_nickelate_hp(formula: str, tc: float) -> bool:
    """Check if formula is a high-pressure nickelate superconductor.

    Actual HP nickelates are La3Ni2O7-type (no Cu). Must distinguish from
    Ni-doped cuprates like La2-xSrxCu1-yNiyO4 which are ambient pressure.
    Key difference: HP nickelates have Ni but NO Cu.
    """
    elements = extract_elements(formula)
    # Must have Ni and O, must have La or Nd, must NOT have Cu
    # (Cu presence means it's a Ni-doped cuprate, not a nickelate HP-SC)
    if 'Ni' in elements and 'O' in elements and 'Cu' not in elements:
        if 'La' in elements or 'Nd' in elements:
            if tc > 30:
                return True
    return False


def is_elemental_hp(formula: str, tc: float) -> bool:
    """Check if formula is an elemental superconductor at anomalously high Tc.

    Pure elements under high pressure can have Tc far exceeding their ambient values.
    Examples: Ca (25K at 161 GPa), Li (20K at 48 GPa), S (17K at 160 GPa).
    """
    elements = extract_elements(formula)
    if len(elements) != 1:
        return False

    # Known elemental HP-SC with Tc thresholds (ambient Tc → HP Tc)
    elemental_hp_thresholds = {
        'Ca': 10.0,   # Ambient ~0K, HP up to 25K at 161 GPa
        'Li': 10.0,   # Ambient <1mK, HP up to 20K at 48 GPa
        'S': 10.0,    # Ambient ~0K, HP up to 17K at 160 GPa
        'P': 5.0,     # Ambient ~0K, HP up to 13K at 30 GPa
        'B': 8.0,     # Ambient ~0K, HP up to 11K at 250 GPa
        'Fe': 1.0,    # Ambient ~0K, HP up to 2K at 15 GPa
        'Si': 5.0,    # Ambient ~0K, HP up to 8.2K at 15 GPa
        'Ge': 4.0,    # Ambient ~0K, HP up to 5.4K at 11 GPa
        'Bi': 6.0,    # Ambient ~0K, HP up to 8.5K at 9 GPa
        'Se': 5.0,    # Ambient ~0K, HP up to 6.9K at 13 GPa
        'Te': 5.0,    # Ambient ~0K, HP up to 7.4K at 35 GPa
        'Sn': 5.0,    # Ambient 3.7K, HP up to 9.3K at 11 GPa
        'Y': 10.0,    # Ambient ~0K, HP up to 19.5K at 115 GPa
        'Sc': 10.0,   # Ambient ~0K, HP up to 19.1K at 107 GPa
    }

    elem = list(elements)[0]
    threshold = elemental_hp_thresholds.get(elem)
    if threshold is not None and tc > threshold:
        return True
    return False


def main():
    print("=" * 70)
    print("High-Pressure Superconductor Labeling")
    print("=" * 70)

    # =========================================================================
    # Step 1a: Extract pressure from NEMAD source
    # =========================================================================
    print("\n--- Step 1a: NEMAD Pressure Extraction ---")

    if not NEMAD_PATH.exists():
        print(f"ERROR: NEMAD source not found at {NEMAD_PATH}")
        sys.exit(1)

    nemad = pd.read_csv(NEMAD_PATH)
    print(f"NEMAD source: {len(nemad)} rows")

    # Parse pressure values
    nemad['pressure_gpa'] = nemad['Pressure'].apply(parse_pressure_to_gpa)

    n_with_pressure = nemad['pressure_gpa'].notna().sum()
    n_hp = (nemad['pressure_gpa'] > HP_THRESHOLD_GPA).sum()
    print(f"  Rows with parseable pressure: {n_with_pressure}")
    print(f"  Rows with pressure > {HP_THRESHOLD_GPA} GPa: {n_hp}")

    # Create NEMAD lookup: formula → max pressure (take highest pressure for each formula)
    nemad['formula_key'] = nemad['Chemical_Composition'].apply(normalize_formula_for_matching)
    nemad_pressure = nemad.groupby('formula_key')['pressure_gpa'].max().to_dict()

    print(f"  Unique NEMAD formulas with pressure data: {len(nemad_pressure)}")

    # =========================================================================
    # Step 1b: Load processed dataset and match
    # =========================================================================
    print("\n--- Step 1b: Dataset Matching ---")

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"Processed dataset: {len(df)} rows")

    # Initialize HP label column
    df['requires_high_pressure'] = 0

    # Create normalized formula keys for matching
    df['formula_key'] = df['formula'].apply(normalize_formula_for_matching)

    # Also try matching on formula_original if available
    if 'formula_original' in df.columns:
        df['formula_original_key'] = df['formula_original'].apply(normalize_formula_for_matching)

    # Track labeling sources for diagnostics
    label_source = ['none'] * len(df)

    # Match against NEMAD pressure data
    nemad_matched = 0
    nemad_hp = 0
    for idx, row in df.iterrows():
        # Try matching on both formula and formula_original
        keys_to_try = [row['formula_key']]
        if 'formula_original_key' in df.columns and row['formula_original_key']:
            keys_to_try.append(row['formula_original_key'])

        for key in keys_to_try:
            if key in nemad_pressure:
                pressure = nemad_pressure[key]
                if pressure is not None:
                    nemad_matched += 1
                    if pressure > HP_THRESHOLD_GPA:
                        df.at[idx, 'requires_high_pressure'] = 1
                        label_source[idx] = f'nemad_pressure_{pressure:.1f}GPa'
                        nemad_hp += 1
                    break

    print(f"  NEMAD matches: {nemad_matched} formulas matched")
    print(f"  NEMAD HP labels: {nemad_hp} formulas marked as HP")

    # =========================================================================
    # Step 1b continued: Supplement with known HP categories
    # =========================================================================
    print("\n--- Step 1b: Known HP Category Supplementation ---")

    # 1. All "Hydrogen-rich Superconductors" → HP
    hydride_mask = df['category'] == 'Hydrogen-rich Superconductors'
    hydride_already_labeled = (df.loc[hydride_mask, 'requires_high_pressure'] == 1).sum()
    df.loc[hydride_mask, 'requires_high_pressure'] = 1
    for idx in df[hydride_mask & (pd.Series(label_source) == 'none')].index:
        label_source[idx] = 'category_hydrogen_rich'
    hydride_new = hydride_mask.sum() - hydride_already_labeled
    print(f"  Hydrogen-rich category: {hydride_mask.sum()} total, {hydride_new} newly labeled")

    # 2. Elemental SC with anomalously high Tc → likely HP
    sc_mask = df['is_superconductor'] == 1
    elemental_hp_count = 0
    for idx, row in df[sc_mask].iterrows():
        if label_source[idx] != 'none':
            continue
        if is_elemental_hp(row['formula'], row['Tc']):
            df.at[idx, 'requires_high_pressure'] = 1
            label_source[idx] = 'elemental_hp'
            elemental_hp_count += 1
    print(f"  Elemental HP-SC: {elemental_hp_count} newly labeled")

    # 3. C60/fullerene-based → HP (15-35 GPa needed for SC)
    fullerene_count = 0
    for idx, row in df[sc_mask].iterrows():
        if label_source[idx] != 'none':
            continue
        if is_fullerene_based(row['formula'], row['Tc']):
            df.at[idx, 'requires_high_pressure'] = 1
            label_source[idx] = 'fullerene'
            fullerene_count += 1
    print(f"  Fullerene-based HP-SC: {fullerene_count} newly labeled")

    # 4. Nickelates (La/Nd-Ni-O with Tc > 30K) → recently discovered HP-SC
    nickelate_count = 0
    for idx, row in df[sc_mask].iterrows():
        if label_source[idx] != 'none':
            continue
        if is_nickelate_hp(row['formula'], row['Tc']):
            df.at[idx, 'requires_high_pressure'] = 1
            label_source[idx] = 'nickelate_hp'
            nickelate_count += 1
    print(f"  Nickelate HP-SC: {nickelate_count} newly labeled")

    # 5. Non-superconductors → always 0 (trivially not HP)
    non_sc_mask = df['is_superconductor'] == 0
    df.loc[non_sc_mask, 'requires_high_pressure'] = 0
    for idx in df[non_sc_mask].index:
        if label_source[idx] != 'none':
            label_source[idx] = 'non_sc_override'

    # =========================================================================
    # Step 1c: Summary and output
    # =========================================================================
    print("\n--- Summary ---")

    total_hp = df['requires_high_pressure'].sum()
    sc_count = sc_mask.sum()
    non_sc_count = non_sc_mask.sum()
    hp_among_sc = df.loc[sc_mask, 'requires_high_pressure'].sum()

    print(f"  Total samples: {len(df)}")
    print(f"  Superconductors: {sc_count}")
    print(f"  Non-superconductors: {non_sc_count}")
    print(f"  HP labels (total): {total_hp}")
    print(f"  HP among SC: {hp_among_sc} ({hp_among_sc/sc_count*100:.1f}%)")
    print(f"  HP among non-SC: {df.loc[non_sc_mask, 'requires_high_pressure'].sum()} (should be 0)")

    # Source breakdown
    source_counts = Counter(label_source)
    print(f"\n  Label source breakdown:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {source}: {count}")

    # Category breakdown of HP labels
    print(f"\n  HP labels by category:")
    hp_by_cat = df[df['requires_high_pressure'] == 1].groupby('category').size()
    for cat, count in hp_by_cat.sort_values(ascending=False).items():
        print(f"    {cat}: {count}")

    # Spot checks
    print("\n--- Spot Checks ---")
    spot_checks = {
        'LaH10': True,       # Famous hydride HP-SC
        'H3S1': True,        # H3S, 200 GPa
        'YBa2Cu3O7': False,  # YBCO cuprate, ambient pressure
        'Nb3Sn': False,      # Classic A15, ambient
        'MgB2': False,       # MgB2, ambient pressure
    }

    for formula, expected_hp in spot_checks.items():
        # Try exact match first, then normalized
        match = df[df['formula'] == formula]
        if match.empty:
            norm = normalize_formula_for_matching(formula)
            match = df[df['formula_key'] == norm]
        if match.empty:
            print(f"  {formula}: NOT IN DATASET")
        else:
            actual_hp = bool(match.iloc[0]['requires_high_pressure'])
            status = 'OK' if actual_hp == expected_hp else 'MISMATCH'
            print(f"  {formula}: HP={actual_hp} (expected={expected_hp}) [{status}]")

    # =========================================================================
    # Write updated CSV
    # =========================================================================
    print(f"\n--- Writing Updated Dataset ---")

    # Drop temporary columns before saving
    cols_to_drop = ['formula_key']
    if 'formula_original_key' in df.columns:
        cols_to_drop.append('formula_original_key')
    df_out = df.drop(columns=cols_to_drop)

    df_out.to_csv(DATASET_PATH, index=False)
    print(f"  Saved to {DATASET_PATH}")
    print(f"  Columns: {len(df_out.columns)} (added requires_high_pressure)")

    # =========================================================================
    # Write diagnostic report
    # =========================================================================
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        'summary': {
            'total_samples': int(len(df)),
            'total_sc': int(sc_count),
            'total_non_sc': int(non_sc_count),
            'total_hp': int(total_hp),
            'hp_among_sc': int(hp_among_sc),
            'hp_percent_of_sc': float(round(hp_among_sc / sc_count * 100, 2)),
            'hp_threshold_gpa': HP_THRESHOLD_GPA,
        },
        'label_sources': {k: int(v) for k, v in sorted(source_counts.items(), key=lambda x: -x[1])},
        'hp_by_category': {str(k): int(v) for k, v in hp_by_cat.sort_values(ascending=False).items()},
        'nemad_stats': {
            'total_nemad_rows': int(len(nemad)),
            'parseable_pressure': int(n_with_pressure),
            'nemad_hp': int(n_hp),
            'matched_to_dataset': nemad_matched,
            'matched_hp': nemad_hp,
        },
        'supplemental_labels': {
            'hydrogen_rich': int(hydride_mask.sum()),
            'elemental_hp': elemental_hp_count,
            'fullerene': fullerene_count,
            'nickelate_hp': nickelate_count,
        },
        'spot_checks': {},
    }

    # Add spot check results to report
    for formula, expected_hp in spot_checks.items():
        match = df[df['formula'] == formula]
        if match.empty:
            norm = normalize_formula_for_matching(formula)
            match = df[df['formula_key'] == norm]
        if not match.empty:
            actual_hp = bool(match.iloc[0]['requires_high_pressure'])
            report['spot_checks'][formula] = {
                'expected_hp': expected_hp,
                'actual_hp': actual_hp,
                'correct': actual_hp == expected_hp,
            }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report to {REPORT_PATH}")

    print(f"\nDone. {total_hp} high-pressure labels applied.")


if __name__ == '__main__':
    main()
