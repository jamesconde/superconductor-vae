#!/usr/bin/env python3
"""
New External Superconductor Dataset Ingestion Pipeline

Ingests 5 external experimental superconductor datasets + 12 manual hydrides,
cleans formulas, converts to fraction notation, computes Magpie/physics features,
deduplicates against existing training CSV + holdout set, and merges.

Datasets ingested (experimental only):
  1. MDR SuperCon (NIMS primary.tsv)        — ~26K entries
  2. SuperCon2 (NLP-extracted, cleanup CSV)  — ~19K entries
  3. 3DSC (3D crystal structures, MP.csv)    — ~41K entries
  4. SODNet (curated, NeurIPS 2024)          — ~12K entries
  5. Manual high-Tc hydrides (12 entries)    — from literature

NOT ingested (DFT, saved separately by export_dft_datasets.py):
  - JARVIS 3D, 2D, alex_supercon, HTSC-2025

JARVIS supercon_chem.json is SKIPPED — already ingested by ingest_jarvis.py.

Outputs:
  data/processed/supercon_fractions_contrastive.csv — updated with new data
  data/processed/new_sc_datasets.csv               — new entries only (for inspection)
  scratch/ingest_new_report.txt                    — detailed run report

Usage:
  python scripts/ingest_new_datasets.py [--dry-run] [--checkpoint scratch/ingest_checkpoint.pkl]
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import Counter, OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXISTING_CSV = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_contrastive.csv"
HOLDOUT_JSON = PROJECT_ROOT / "data" / "GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json"

# Dataset paths
MDR_TSV = PROJECT_ROOT / "data" / "mdr_supercon" / "primary.tsv"
SUPERCON2_CSV = PROJECT_ROOT / "data" / "supercon2_repo" / "data" / "supercon2_v23.04.28_cleanup.csv"
THREESC_CSV = PROJECT_ROOT / "data" / "3DSC_repo" / "superconductors_3D" / "data" / "final" / "MP" / "3DSC_MP.csv"
SODNET_CSV = PROJECT_ROOT / "data" / "SODNet_repo" / "datasets" / "SuperCon" / "SuperCon_11949.csv"

# Outputs
OUT_COMBINED = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_contrastive.csv"
OUT_NEW_ONLY = PROJECT_ROOT / "data" / "processed" / "new_sc_datasets.csv"
OUT_REPORT = PROJECT_ROOT / "scratch" / "ingest_new_report.txt"


# ── Report logger ──────────────────────────────────────────────────────────
class ReportLogger:
    """Accumulates log lines for the final report file."""

    def __init__(self):
        self.lines = []
        self.skip_reasons = Counter()

    def log(self, msg: str):
        self.lines.append(msg)
        print(msg)

    def skip(self, reason: str, formula: str = ""):
        self.skip_reasons[reason] += 1
        self.lines.append(f"  SKIP [{reason}]: {formula[:80]}")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(self.lines))
            f.write("\n\n=== SKIP REASON SUMMARY ===\n")
            for reason, count in self.skip_reasons.most_common():
                f.write(f"  {reason}: {count}\n")
        print(f"\nReport saved to {path}")


report = ReportLogger()


# ══════════════════════════════════════════════════════════════════════════
# Dataset Readers
# ══════════════════════════════════════════════════════════════════════════

def read_mdr(path: Path) -> pd.DataFrame:
    """Read MDR SuperCon NIMS primary.tsv.

    Has 3 header rows. Column 2 (index 2) is 'chemical formula' (element),
    column 5 (index 5) is Tc. Strips variable oxygen suffixes.
    """
    report.log(f"\n{'='*60}")
    report.log(f"Reading MDR SuperCon from {path}")
    report.log(f"{'='*60}")

    df = pd.read_csv(path, sep='\t', skiprows=3, header=None,
                     on_bad_lines='skip', dtype=str, encoding='utf-8')
    report.log(f"  Raw rows: {len(df)}")

    # Column 2 = chemical formula, column 5 = Tc
    df = df.rename(columns={2: "formula_raw", 5: "Tc_raw"})

    # Parse Tc
    df["Tc"] = pd.to_numeric(df["Tc_raw"], errors="coerce")
    df = df[df["Tc"].notna() & np.isfinite(df["Tc"])].copy()
    report.log(f"  After Tc parse: {len(df)}")

    # Filter Tc range: 0 <= Tc <= 400K
    df = df[(df["Tc"] >= 0) & (df["Tc"] <= 400)].copy()
    report.log(f"  After Tc range filter (0-400K): {len(df)}")

    # Clean formulas: strip variable oxygen suffixes
    def clean_mdr_formula(raw):
        if not isinstance(raw, str) or not raw.strip():
            return None
        s = raw.strip()
        # Strip variable oxygen suffixes: -Y, -Z, -X, -d, +X, Oz, +y, -z etc.
        s = re.sub(r'[-+][YZXdyzx]$', '', s)
        # Strip trailing Oz notation
        s = re.sub(r'O[zZxXyYdD]$', lambda m: 'O', s)
        return s if s else None

    df["formula"] = df["formula_raw"].apply(clean_mdr_formula)
    df = df[df["formula"].notna()].copy()
    report.log(f"  After formula cleaning: {len(df)}")

    # Filter retracted/unreliable entries
    n_before = len(df)
    retracted_mask = pd.Series(False, index=df.index)

    for idx, row in df.iterrows():
        f = row["formula"]
        tc = row["Tc"]
        # Lu+H with Tc 250-300K (Dias, retracted)
        if re.search(r'Lu.*H|H.*Lu', f) and 250 <= tc <= 300:
            retracted_mask.at[idx] = True
        # C+S+H with Tc 280-295K (Snider, retracted)
        if all(x in f for x in ['C', 'S', 'H']) and 280 <= tc <= 295:
            # Check it's actually a C-S-H system, not a compound with these elements
            from pymatgen.core import Composition
            try:
                comp = Composition(f)
                elems = {str(e) for e in comp.elements}
                if elems == {'C', 'S', 'H'} or (len(elems) <= 4 and {'C', 'S', 'H'}.issubset(elems)):
                    retracted_mask.at[idx] = True
            except Exception:
                pass
        # PbCO3-based with Tc > 300K (unconfirmed)
        if 'Pb' in f and 'C' in f and 'O' in f and tc > 300:
            retracted_mask.at[idx] = True

    df = df[~retracted_mask].copy()
    report.log(f"  Removed {n_before - len(df)} retracted/unreliable entries")

    # Determine is_superconductor
    df["is_superconductor"] = (df["Tc"] > 0).astype(int)
    df["tc_source"] = "experimental"
    df["dataset"] = "mdr_supercon"

    report.log(f"  Final MDR rows: {len(df)}")
    report.log(f"    SC: {(df.is_superconductor == 1).sum()}, Non-SC: {(df.is_superconductor == 0).sum()}")
    report.log(f"    Tc>=100K: {(df.Tc >= 100).sum()}, Tc>=200K: {(df.Tc >= 200).sum()}")

    return df[["formula", "Tc", "is_superconductor", "tc_source", "dataset"]].copy()


def read_supercon2(path: Path) -> pd.DataFrame:
    """Read SuperCon2 cleanup CSV with contamination filtering.

    Filters out manganites, ZnO, and other non-SC Tc values.
    """
    report.log(f"\n{'='*60}")
    report.log(f"Reading SuperCon2 from {path}")
    report.log(f"{'='*60}")

    df = pd.read_csv(path, on_bad_lines='skip')
    report.log(f"  Raw rows: {len(df)}")

    # Use 'formula' column and 'criticalTemperature' as Tc
    df = df[df["formula"].notna() & (df["formula"].str.strip() != "")].copy()
    df["Tc"] = pd.to_numeric(df["criticalTemperature"], errors="coerce")
    df = df[df["Tc"].notna() & np.isfinite(df["Tc"]) & (df["Tc"] >= 0)].copy()
    report.log(f"  After formula+Tc filter: {len(df)}")

    # Contamination filter: remove likely non-SC Tc values
    n_before = len(df)
    contamination_mask = pd.Series(False, index=df.index)

    for idx, row in df.iterrows():
        f = str(row["formula"])
        tc = row["Tc"]
        mat_class = str(row.get("materialClass", "")).lower()

        # La/Ca/Sr + Mn + O compounds with Tc > 50K (Curie temperatures of manganites)
        if any(x in f for x in ['La', 'Ca', 'Sr']) and 'Mn' in f and 'O' in f and tc > 50:
            contamination_mask.at[idx] = True
            continue

        # ZnO-based with Tc > 50K (ferromagnetic)
        if 'Zn' in f and 'O' in f and tc > 50 and len(re.findall(r'[A-Z][a-z]?', f)) <= 3:
            contamination_mask.at[idx] = True
            continue

        # Non-hydride entry with Tc > 200K (likely structural/magnetic transitions)
        if tc > 200 and 'H' not in f:
            contamination_mask.at[idx] = True
            continue

        # Oxides with Tc > 200K unless also hydrides
        if 'oxide' in mat_class and tc > 200:
            if 'H' not in f:
                contamination_mask.at[idx] = True
                continue

    df = df[~contamination_mask].copy()
    report.log(f"  Removed {n_before - len(df)} contaminated entries")

    # Filter Tc range: 0 <= Tc <= 400K
    df = df[(df["Tc"] >= 0) & (df["Tc"] <= 400)].copy()

    # Extract pressure
    df["pressure_gpa"] = pd.to_numeric(df.get("appliedPressure", pd.Series(dtype=float)), errors="coerce")
    pressure_unit = df.get("appliedPressureUnit", pd.Series(dtype=str))
    # Convert MPa to GPa
    mpa_mask = pressure_unit.str.lower().str.contains("mpa", na=False)
    df.loc[mpa_mask, "pressure_gpa"] = df.loc[mpa_mask, "pressure_gpa"] / 1000.0

    df["is_superconductor"] = (df["Tc"] > 0).astype(int)
    df["tc_source"] = "experimental"
    df["dataset"] = "supercon2"

    # Rename formula column to use 'formula'
    report.log(f"  Final SuperCon2 rows: {len(df)}")
    report.log(f"    SC: {(df.is_superconductor == 1).sum()}, Non-SC: {(df.is_superconductor == 0).sum()}")
    report.log(f"    Tc>=100K: {(df.Tc >= 100).sum()}, Tc>=200K: {(df.Tc >= 200).sum()}")
    report.log(f"    With pressure data: {df.pressure_gpa.notna().sum()}")

    return df[["formula", "Tc", "is_superconductor", "tc_source", "dataset", "pressure_gpa"]].copy()


def read_3dsc(path: Path) -> pd.DataFrame:
    """Read 3DSC_MP.csv (has comment line at start).

    Uses formula_sc column, tc column.
    """
    report.log(f"\n{'='*60}")
    report.log(f"Reading 3DSC from {path}")
    report.log(f"{'='*60}")

    df = pd.read_csv(path, comment='#', on_bad_lines='skip')
    report.log(f"  Raw rows: {len(df)}")

    # Use formula_sc and tc columns
    df = df[df["formula_sc"].notna() & (df["formula_sc"].str.strip() != "")].copy()
    df["formula"] = df["formula_sc"]
    df["Tc"] = pd.to_numeric(df["tc"], errors="coerce")
    df = df[df["Tc"].notna() & np.isfinite(df["Tc"]) & (df["Tc"] >= 0) & (df["Tc"] <= 400)].copy()
    report.log(f"  After formula+Tc filter: {len(df)}")

    df["is_superconductor"] = (df["Tc"] > 0).astype(int)
    df["tc_source"] = "experimental"
    df["dataset"] = "3dsc"

    # Keep sc_class for category hint
    df["_sc_class"] = df.get("sc_class", pd.Series(dtype=str))

    report.log(f"  Final 3DSC rows: {len(df)}")
    report.log(f"    SC: {(df.is_superconductor == 1).sum()}, Non-SC: {(df.is_superconductor == 0).sum()}")
    report.log(f"    Tc>=100K: {(df.Tc >= 100).sum()}, Tc>=200K: {(df.Tc >= 200).sum()}")

    return df[["formula", "Tc", "is_superconductor", "tc_source", "dataset", "_sc_class"]].copy()


def read_sodnet(path: Path) -> pd.DataFrame:
    """Read SODNet SuperCon_11949.csv.

    Uses Formula and Tc columns, Materials_family for category hint.
    """
    report.log(f"\n{'='*60}")
    report.log(f"Reading SODNet from {path}")
    report.log(f"{'='*60}")

    df = pd.read_csv(path, on_bad_lines='skip')
    report.log(f"  Raw rows: {len(df)}")

    df = df[df["Formula"].notna() & (df["Formula"].str.strip() != "")].copy()
    df["formula"] = df["Formula"]
    df["Tc"] = pd.to_numeric(df["Tc"], errors="coerce")
    df = df[df["Tc"].notna() & np.isfinite(df["Tc"]) & (df["Tc"] >= 0) & (df["Tc"] <= 400)].copy()
    report.log(f"  After formula+Tc filter: {len(df)}")

    df["is_superconductor"] = (df["Tc"] > 0).astype(int)
    df["tc_source"] = "experimental"
    df["dataset"] = "sodnet"
    df["_materials_family"] = df.get("Materials_family", pd.Series(dtype=str))

    report.log(f"  Final SODNet rows: {len(df)}")
    report.log(f"    SC: {(df.is_superconductor == 1).sum()}, Non-SC: {(df.is_superconductor == 0).sum()}")
    report.log(f"    Tc>=100K: {(df.Tc >= 100).sum()}, Tc>=200K: {(df.Tc >= 200).sum()}")

    return df[["formula", "Tc", "is_superconductor", "tc_source", "dataset", "_materials_family"]].copy()


def read_manual_hydrides() -> pd.DataFrame:
    """Hardcoded 12 experimental hydrides from literature.

    Source: docs/high_tc_data_acquisition.md, Source 7.
    """
    report.log(f"\n{'='*60}")
    report.log(f"Reading manual experimental hydrides (12 entries)")
    report.log(f"{'='*60}")

    entries = [
        # >200K bin
        {"formula": "H3S",          "Tc": 203, "pressure_gpa": 155},
        {"formula": "LaH10",        "Tc": 250, "pressure_gpa": 170},
        {"formula": "LaH10",        "Tc": 260, "pressure_gpa": 190},
        {"formula": "YH9",          "Tc": 243, "pressure_gpa": 201},
        {"formula": "YH6",          "Tc": 224, "pressure_gpa": 166},
        {"formula": "CaH6",         "Tc": 215, "pressure_gpa": 172},
        {"formula": "La0.5Y0.5H10", "Tc": 253, "pressure_gpa": 183},
        # 100-200K bin
        {"formula": "La0.5Ce0.5H9", "Tc": 178, "pressure_gpa": 97},
        {"formula": "ThH10",        "Tc": 161, "pressure_gpa": 175},
        {"formula": "ThH9",         "Tc": 146, "pressure_gpa": 170},
        {"formula": "CeH9",         "Tc": 117, "pressure_gpa": 95},
        {"formula": "LaBeH8",       "Tc": 110, "pressure_gpa": 80},
    ]

    df = pd.DataFrame(entries)
    df["is_superconductor"] = 1
    df["tc_source"] = "experimental"
    df["dataset"] = "manual_hydrides"

    report.log(f"  Manual hydrides: {len(df)}")
    report.log(f"    Tc>=200K: {(df.Tc >= 200).sum()}")
    report.log(f"    100K<=Tc<200K: {((df.Tc >= 100) & (df.Tc < 200)).sum()}")

    return df


# ══════════════════════════════════════════════════════════════════════════
# Formula Cleaning & Parsing (reused from ingest_jarvis.py)
# ══════════════════════════════════════════════════════════════════════════

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def clean_formula(raw: str) -> Optional[str]:
    """Clean a raw formula string for pymatgen parsing."""
    if not isinstance(raw, str) or not raw.strip():
        return None

    s = raw.strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_SUBSCRIPT_MAP)
    s = s.translate(_SUPERSCRIPT_MAP)
    s = s.replace("·", "").replace("•", "")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("\u200b", "").replace("\u00a0", " ")

    # Remove delta/variable notation
    s = re.sub(r"[±+\-]?\s*[δΔ]", "", s)
    s = re.sub(r"[+\-]\s*[xyzn]\b", "", s)

    # Remove variable oxygen suffixes (MDR-style)
    s = re.sub(r'[-+][YZXdyzx]$', '', s)

    # Remove percentage notation
    if re.search(r"\d+\s*%", s):
        return None

    s = re.sub(r"[{}]", "", s)
    s = s.strip()

    if not s or len(s) < 2:
        return None
    if not re.search(r"[A-Z]", s):
        return None

    return s


def parse_compositions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse cleaned formulas with pymatgen, deduplicate within dataset."""
    from pymatgen.core import Composition

    report.log("\nFormula cleaning and pymatgen parsing")

    compositions = []
    canonical_keys = []
    cleaned_formulas = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing formulas"):
        raw = row["formula"]
        cleaned = clean_formula(raw)

        if cleaned is None:
            compositions.append(None)
            canonical_keys.append(None)
            cleaned_formulas.append(None)
            report.skip("formula_clean_failed", str(raw)[:60])
            continue

        try:
            comp = Composition(cleaned)
            if len(comp.elements) == 0:
                raise ValueError("No elements")
            canonical = comp.alphabetical_formula
            compositions.append(comp)
            canonical_keys.append(canonical)
            cleaned_formulas.append(cleaned)
        except Exception as e:
            compositions.append(None)
            canonical_keys.append(None)
            cleaned_formulas.append(None)
            report.skip("pymatgen_parse_error", f"{str(raw)[:40]}: {str(e)[:30]}")

    df = df.copy()
    df["_composition"] = compositions
    df["_canonical"] = canonical_keys
    df["_cleaned"] = cleaned_formulas

    n_parsed = sum(c is not None for c in compositions)
    report.log(f"  Successfully parsed: {n_parsed} / {len(df)}")

    # Drop unparseable
    df = df[df["_composition"].notna()].copy()
    report.log(f"  After dropping unparseable: {len(df)}")

    return df


# ══════════════════════════════════════════════════════════════════════════
# Category Assignment (reused from ingest_jarvis.py)
# ══════════════════════════════════════════════════════════════════════════

def assign_category(comp, hint: str = "") -> str:
    """Map composition to SC category using element heuristics.

    Args:
        comp: pymatgen Composition object
        hint: Optional category hint from source dataset (e.g., 3DSC sc_class,
              SODNet Materials_family)
    """
    elements = {str(e) for e in comp.elements}
    el_fracs = comp.fractional_composition.as_dict()

    # Use hints from source datasets when available
    hint_lower = str(hint).lower() if hint else ""
    if "cuprate" in hint_lower:
        return "Cuprates"
    if "iron" in hint_lower or "pnictide" in hint_lower:
        return "Iron-based"

    # Single element -> Elemental
    if len(elements) == 1:
        return "Elemental Superconductors"

    # Hydrogen-rich: H fraction > 50%
    h_frac = el_fracs.get("H", 0.0)
    if h_frac > 0.5:
        return "Hydrogen-rich Superconductors"

    # Iron-based
    if "Fe" in elements and any(e in elements for e in ["As", "Se", "P", "Te"]):
        return "Iron-based"

    # Cuprate detection
    has_cu = "Cu" in elements
    has_o = "O" in elements

    if has_cu and has_o:
        if "Bi" in elements:
            return "Bismuthates"
        return "Cuprates"

    # Bismuthates (Ba-K-Bi-O type)
    if "Bi" in elements and "O" in elements and not has_cu:
        return "Bismuthates"

    # Borocarbides
    if "B" in elements and "C" in elements and any(
        e in elements for e in ["Y", "Lu", "Er", "Ho", "Dy", "Tm", "Ni"]
    ):
        return "Borocarbides"

    # MgB2-family
    if "Mg" in elements and "B" in elements:
        return "Other"

    return "Other"


# ══════════════════════════════════════════════════════════════════════════
# Feature Computation (reused from ingest_jarvis.py)
# ══════════════════════════════════════════════════════════════════════════

def compute_physics_features(comp) -> dict:
    """Compute 6 physics-informed feature columns from a pymatgen Composition."""
    elements = {str(e) for e in comp.elements}
    el_fracs = comp.fractional_composition.as_dict()

    transition_metals = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    }

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
        "transition_metal_count": len(elements & transition_metals),
    }


def compute_lp_norms(compositions) -> pd.DataFrame:
    """Compute L-p norm features from composition fractions."""
    norms_data = []
    for comp in compositions:
        fracs = np.array(list(comp.fractional_composition.as_dict().values()))
        norms_data.append({
            "0-norm": float(len(fracs)),
            "2-norm": float(np.linalg.norm(fracs, 2)),
            "3-norm": float(np.linalg.norm(fracs, 3)),
            "5-norm": float(np.linalg.norm(fracs, 5)),
            "7-norm": float(np.linalg.norm(fracs, 7)),
            "10-norm": float(np.linalg.norm(fracs, 10)),
        })
    return pd.DataFrame(norms_data)


def compute_matminer_features(df: pd.DataFrame, checkpoint_path: Optional[Path] = None) -> pd.DataFrame:
    """Compute Magpie + valence + ion + TM features with matminer.

    Uses n_jobs=1 to avoid multiprocessing deadlocks, and per-composition
    timeout for IonProperty (some compositions cause combinatorial explosion
    in oxidation state enumeration).

    Args:
        df: DataFrame with '_composition' column containing pymatgen Composition objects
        checkpoint_path: Optional path to save/load intermediate checkpoint

    Returns:
        DataFrame with feature columns
    """
    import signal
    import warnings
    from matminer.featurizers.composition import (
        ElementProperty, IonProperty, TMetalFraction, ValenceOrbital
    )

    # Check for checkpoint
    if checkpoint_path and checkpoint_path.exists():
        report.log(f"  Loading feature checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)

    compositions = df["_composition"].tolist()

    ep = ElementProperty.from_preset("magpie")
    vo = ValenceOrbital()
    ip = IonProperty()
    tm = TMetalFraction()

    # Use n_jobs=1 to avoid multiprocessing pipe deadlocks from warning flood
    ep.set_n_jobs(1)
    vo.set_n_jobs(1)
    ip.set_n_jobs(1)
    tm.set_n_jobs(1)

    feat_df = pd.DataFrame({"composition": compositions})

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No Pauling electronegativity")

        # Magpie
        report.log("  Computing MagpieData features (132 columns)...")
        feat_df = ep.featurize_dataframe(feat_df, "composition", ignore_errors=True)

        # Valence
        report.log("  Computing ValenceOrbital features (4 columns)...")
        feat_df = vo.featurize_dataframe(feat_df, "composition", ignore_errors=True)

        # IonProperty with per-composition timeout (some compositions cause
        # combinatorial explosion in oxidation state enumeration)
        report.log("  Computing IonProperty features (3 columns) with 30s/composition timeout...")
        ION_TIMEOUT_SEC = 30

        class IonPropertyTimeout(Exception):
            pass

        def _ion_timeout_handler(signum, frame):
            raise IonPropertyTimeout()

        ip_labels = ip.feature_labels()
        ion_results = []
        n_timeouts = 0
        for comp in tqdm(compositions, desc="IonProperty"):
            old_handler = signal.signal(signal.SIGALRM, _ion_timeout_handler)
            signal.alarm(ION_TIMEOUT_SEC)
            try:
                result = ip.featurize(comp)
                ion_results.append(result)
            except (IonPropertyTimeout, Exception):
                ion_results.append([False, 0.0, 0.0])
                n_timeouts += 1
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        for i, label in enumerate(ip_labels):
            feat_df[label] = [r[i] for r in ion_results]
        if n_timeouts > 0:
            report.log(f"  IonProperty: {n_timeouts} compositions timed out (>{ION_TIMEOUT_SEC}s), used defaults")

        # TM fraction
        report.log("  Computing TMetalFraction feature (1 column)...")
        feat_df = tm.featurize_dataframe(feat_df, "composition", ignore_errors=True)

    # Drop extra columns
    frac_valence_labels = [
        "frac s valence electrons", "frac p valence electrons",
        "frac d valence electrons", "frac f valence electrons"
    ]
    drop_cols = ["composition"] + frac_valence_labels
    for col in drop_cols:
        if col in feat_df.columns:
            feat_df = feat_df.drop(columns=[col])

    # Save checkpoint
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(feat_df, f)
        report.log(f"  Saved feature checkpoint to {checkpoint_path}")

    return feat_df


# ══════════════════════════════════════════════════════════════════════════
# Formula Notation (reused from ingest_jarvis.py)
# ══════════════════════════════════════════════════════════════════════════

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


def composition_to_decimal_formula(comp) -> str:
    """Convert pymatgen Composition to decimal notation."""
    parts = []
    for el, amt in sorted(comp.as_dict().items()):
        if amt == int(amt):
            if int(amt) == 1:
                parts.append(str(el))
            else:
                parts.append(f"{el}{int(amt)}")
        else:
            parts.append(f"{el}{amt}")
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# Deduplication & Holdout Filter (reused from ingest_jarvis.py)
# ══════════════════════════════════════════════════════════════════════════

def load_existing_canonical_keys(csv_path: Path) -> set:
    """Load existing compositions as canonical keys."""
    from pymatgen.core import Composition

    report.log(f"\nLoading existing data for dedup from {csv_path}")
    existing = pd.read_csv(csv_path, usecols=["composition"])
    keys = set()
    for comp_str in tqdm(existing["composition"], desc="Canonicalizing existing"):
        try:
            comp = Composition(str(comp_str))
            keys.add(comp.alphabetical_formula)
        except Exception:
            pass
    report.log(f"  Existing canonical keys: {len(keys)}")
    return keys


def load_holdout_keys(json_path: Path) -> set:
    """Load holdout sample composition keys."""
    from pymatgen.core import Composition

    report.log(f"Loading holdout samples from {json_path}")
    with open(json_path) as f:
        holdout = json.load(f)
    keys = set()
    for sample in holdout["holdout_samples"]:
        try:
            comp = Composition(sample["formula"])
            keys.add(comp.alphabetical_formula)
        except Exception:
            pass
    report.log(f"  Holdout canonical keys: {len(keys)}")
    return keys


# ══════════════════════════════════════════════════════════════════════════
# High-Pressure Labeling
# ══════════════════════════════════════════════════════════════════════════

def label_high_pressure(df: pd.DataFrame) -> np.ndarray:
    """Label entries as requiring high pressure.

    Rules:
    - Explicit pressure_gpa > 1.0 → requires_high_pressure = 1
    - H-containing with Tc > 100K and no pressure data → flag conservatively
    """
    requires_hp = np.zeros(len(df), dtype=int)

    for i, (_, row) in enumerate(df.iterrows()):
        # Explicit pressure data
        pressure = row.get("pressure_gpa", None)
        if pd.notna(pressure) and float(pressure) > 1.0:
            requires_hp[i] = 1
            continue

        # Conservative flag for high-Tc hydrides without pressure data
        comp = row.get("_composition")
        if comp is not None:
            el_fracs = comp.fractional_composition.as_dict()
            h_frac = el_fracs.get("H", 0.0)
            if h_frac > 0.3 and row["Tc"] > 100:
                requires_hp[i] = 1

    return requires_hp


# ══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ingest new external superconductor datasets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and clean only, don't compute features or save")
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "scratch" / "ingest_checkpoint.pkl"),
                        help="Path for feature computation checkpoint")
    args = parser.parse_args()

    t0 = time.time()
    report.log("=" * 70)
    report.log("New External Superconductor Dataset Ingestion Pipeline")
    report.log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.log("=" * 70)

    # ── Step 1: Read all datasets ──
    dfs = []

    if MDR_TSV.exists():
        dfs.append(read_mdr(MDR_TSV))
    else:
        report.log(f"\nWARNING: MDR file not found: {MDR_TSV}")

    if SUPERCON2_CSV.exists():
        dfs.append(read_supercon2(SUPERCON2_CSV))
    else:
        report.log(f"\nWARNING: SuperCon2 file not found: {SUPERCON2_CSV}")

    if THREESC_CSV.exists():
        dfs.append(read_3dsc(THREESC_CSV))
    else:
        report.log(f"\nWARNING: 3DSC file not found: {THREESC_CSV}")

    if SODNET_CSV.exists():
        dfs.append(read_sodnet(SODNET_CSV))
    else:
        report.log(f"\nWARNING: SODNet file not found: {SODNET_CSV}")

    dfs.append(read_manual_hydrides())

    # JARVIS chem is SKIPPED — already ingested by ingest_jarvis.py
    report.log(f"\n  NOTE: JARVIS supercon_chem.json SKIPPED (already ingested by ingest_jarvis.py)")

    # ── Combine all datasets ──
    # Normalize columns before concat
    common_cols = ["formula", "Tc", "is_superconductor", "tc_source", "dataset"]
    combined_parts = []
    for part_df in dfs:
        part = part_df.copy()
        for col in common_cols:
            if col not in part.columns:
                part[col] = None
        # Carry forward pressure_gpa if present
        if "pressure_gpa" not in part.columns:
            part["pressure_gpa"] = np.nan
        # Carry forward category hints
        if "_sc_class" not in part.columns:
            part["_sc_class"] = ""
        if "_materials_family" not in part.columns:
            part["_materials_family"] = ""
        combined_parts.append(part)

    df = pd.concat(combined_parts, ignore_index=True)
    report.log(f"\n{'='*60}")
    report.log(f"Combined raw rows from all sources: {len(df)}")
    report.log(f"{'='*60}")

    # Per-dataset counts
    for ds, count in df["dataset"].value_counts().items():
        report.log(f"  {ds}: {count}")

    # ── Step 2: Parse formulas with pymatgen ──
    df = parse_compositions(df)

    # ── Step 3: Internal dedup (keep first occurrence, prefer experimental) ──
    before_internal = len(df)
    df = df.drop_duplicates(subset="_canonical", keep="first")
    report.log(f"\nInternal dedup: {before_internal} -> {len(df)} (removed {before_internal - len(df)} internal dupes)")

    # ── Step 4: Dedup against existing CSV + holdout ──
    existing_keys = load_existing_canonical_keys(EXISTING_CSV)
    holdout_keys = load_holdout_keys(HOLDOUT_JSON)

    mask_existing = df["_canonical"].isin(existing_keys)
    n_existing_dupes = mask_existing.sum()
    df = df[~mask_existing].copy()
    report.log(f"  Removed {n_existing_dupes} entries duplicating existing training data")

    mask_holdout = df["_canonical"].isin(holdout_keys)
    n_holdout = mask_holdout.sum()
    df = df[~mask_holdout].copy()
    report.log(f"  Removed {n_holdout} holdout samples")
    report.log(f"  After all dedup: {len(df)} unique new entries")

    if len(df) == 0:
        report.log("\nNo new entries to add. Exiting.")
        report.save(OUT_REPORT)
        return

    # Per-dataset counts after dedup
    report.log(f"\nPer-dataset counts after dedup:")
    for ds, count in df["dataset"].value_counts().items():
        report.log(f"  {ds}: {count}")

    if args.dry_run:
        report.log(f"\n  DRY RUN — skipping feature computation and save")
        report.log(f"  Would add {len(df)} new rows to training CSV")
        report.save(OUT_REPORT)
        return

    # ── Step 5: Category assignment ──
    report.log("\nAssigning categories")
    categories = []
    for _, row in df.iterrows():
        hint = str(row.get("_sc_class", "")) + " " + str(row.get("_materials_family", ""))
        categories.append(assign_category(row["_composition"], hint=hint))
    df["category"] = categories

    report.log(f"  Category distribution:")
    for cat, count in pd.Series(categories).value_counts().items():
        report.log(f"    {cat}: {count}")

    # ── Step 6: Compute features ──
    report.log("\nComputing Lp norms")
    norms_df = compute_lp_norms(df["_composition"].tolist())

    report.log("\nComputing matminer features (this may take several hours)...")
    checkpoint_path = Path(args.checkpoint)
    feat_df = compute_matminer_features(df, checkpoint_path=checkpoint_path)

    report.log("\nComputing physics features (6 columns)")
    physics_data = []
    for comp in tqdm(df["_composition"], desc="Physics features"):
        physics_data.append(compute_physics_features(comp))
    physics_df = pd.DataFrame(physics_data)

    # ── Step 7: Generate formula notations ──
    report.log("\nGenerating formula notations")
    formulas_frac = []
    formulas_dec = []
    compositions_str = []
    for comp in tqdm(df["_composition"], desc="Formula notation"):
        formulas_frac.append(composition_to_fraction_formula(comp))
        formulas_dec.append(composition_to_decimal_formula(comp))
        compositions_str.append(comp.formula)

    # ── Step 8: High-pressure labeling ──
    report.log("\nLabeling high-pressure entries")
    requires_hp = label_high_pressure(df)
    n_hp = requires_hp.sum()
    report.log(f"  High-pressure entries: {n_hp}")

    # ── Step 9: Assemble output DataFrame ──
    report.log("\nAssembling output DataFrame")

    # Load target column order from existing CSV
    existing_df_cols = pd.read_csv(EXISTING_CSV, nrows=0)
    target_columns = list(existing_df_cols.columns)
    report.log(f"  Target columns: {len(target_columns)}")

    from matminer.featurizers.composition import ElementProperty, IonProperty, TMetalFraction
    ep = ElementProperty.from_preset("magpie")
    magpie_labels = ep.feature_labels()
    valence_labels = [
        "avg s valence electrons", "avg p valence electrons",
        "avg d valence electrons", "avg f valence electrons"
    ]
    ion_labels = ["compound possible", "max ionic char", "avg ionic char"]
    tm_labels = ["transition metal fraction"]
    physics_labels = [
        "has_cuprate_elements", "cu_o_ratio", "has_iron_pnictide",
        "has_mgb2_elements", "hydrogen_fraction", "transition_metal_count"
    ]

    col_data = OrderedDict()
    col_data["formula"] = formulas_frac
    col_data["Tc"] = df["Tc"].values
    col_data["composition"] = compositions_str
    col_data["category"] = df["category"].values
    col_data["is_superconductor"] = df["is_superconductor"].values

    # Lp norms
    for col in norms_df.columns:
        col_data[col] = norms_df[col].values

    # Matminer features
    for col in magpie_labels + valence_labels + ion_labels + tm_labels:
        if col in feat_df.columns:
            col_data[col] = feat_df[col].values
        else:
            report.log(f"  WARNING: Missing feature column '{col}', filling with 0")
            col_data[col] = 0.0

    col_data["formula_original"] = formulas_dec
    col_data["requires_high_pressure"] = requires_hp

    # Physics features
    for col in physics_labels:
        col_data[col] = physics_df[col].values

    new_df = pd.DataFrame(col_data)

    # Drop rows where Magpie featurization failed
    before = len(new_df)
    new_df = new_df.dropna(subset=magpie_labels[:5])
    if len(new_df) < before:
        report.log(f"  Dropped {before - len(new_df)} rows with failed Magpie featurization")

    # Fill NaN in IonProperty columns
    n_ion_nan = new_df["compound possible"].isna().sum()
    if n_ion_nan > 0:
        report.log(f"  Filling {n_ion_nan} NaN IonProperty rows")
        pd.set_option('future.no_silent_downcasting', True)
        new_df["compound possible"] = new_df["compound possible"].fillna(False).infer_objects(copy=False)
        new_df["max ionic char"] = new_df["max ionic char"].fillna(0.0).infer_objects(copy=False)
        new_df["avg ionic char"] = new_df["avg ionic char"].fillna(0.0).infer_objects(copy=False)

    # Verify column alignment
    if list(new_df.columns) != target_columns:
        report.log(f"  Reordering columns to match target schema")
        # Ensure all target columns exist
        for col in target_columns:
            if col not in new_df.columns:
                report.log(f"    Adding missing column '{col}' with default 0")
                new_df[col] = 0
        new_df = new_df[target_columns]

    report.log(f"  New entries assembled: {len(new_df)} rows x {len(new_df.columns)} columns")

    # ── Step 10: Merge and save ──
    report.log(f"\n{'='*60}")
    report.log("Merging and saving")
    report.log(f"{'='*60}")

    existing_full = pd.read_csv(EXISTING_CSV)
    report.log(f"  Existing rows: {len(existing_full)}")
    report.log(f"  New rows to add: {len(new_df)}")

    # Ensure column match
    new_cols = set(new_df.columns)
    existing_cols = set(existing_full.columns)
    if new_cols != existing_cols:
        for col in existing_cols - new_cols:
            new_df[col] = 0
        for col in new_cols - existing_cols:
            existing_full[col] = 0
        new_df = new_df[existing_full.columns]

    combined = pd.concat([existing_full, new_df], ignore_index=True)
    report.log(f"  Combined rows: {len(combined)}")

    # ── Verification ──
    report.log(f"\n{'='*60}")
    report.log("VERIFICATION")
    report.log(f"{'='*60}")

    n_cols = len(combined.columns)
    report.log(f"  Columns: {n_cols}")
    assert n_cols == 159, f"Expected 159 columns, got {n_cols}"
    report.log(f"  [PASS] Column count: {n_cols}")

    assert combined["formula"].notna().all(), "NaN found in formula column!"
    assert combined["Tc"].notna().all(), "NaN found in Tc column!"
    assert combined["is_superconductor"].notna().all(), "NaN found in is_superconductor column!"
    report.log(f"  [PASS] No NaN in critical columns (formula, Tc, is_superconductor)")

    # Verify no holdout contamination
    from pymatgen.core import Composition as Comp
    new_comps = set()
    for comp_str in new_df["composition"]:
        try:
            new_comps.add(Comp(comp_str).alphabetical_formula)
        except Exception:
            pass
    holdout_overlap = holdout_keys.intersection(new_comps)
    assert len(holdout_overlap) == 0, f"Holdout contamination: {holdout_overlap}"
    report.log(f"  [PASS] No holdout samples in new data")

    # Tc distribution
    report.log(f"\n  Tc Distribution:")
    report.log(f"    {'':20s} {'Existing':>10s} {'New':>10s} {'Combined':>10s}")
    for label, mask_fn in [
        ("Count", lambda tc: pd.Series(True, index=tc.index)),
        ("SC (Tc>0)", lambda tc: tc > 0),
        ("Tc>=77K", lambda tc: tc >= 77),
        ("Tc>=100K", lambda tc: tc >= 100),
        ("Tc>=150K", lambda tc: tc >= 150),
        ("Tc>=200K", lambda tc: tc >= 200),
    ]:
        e_count = mask_fn(existing_full["Tc"]).sum()
        n_count = mask_fn(new_df["Tc"]).sum()
        c_count = mask_fn(combined["Tc"]).sum()
        report.log(f"    {label:20s} {e_count:10d} {n_count:10d} {c_count:10d}")

    # Category distribution
    report.log(f"\n  Combined category distribution:")
    for cat, count in combined["category"].value_counts().items():
        report.log(f"    {cat}: {count}")

    # Per-dataset is_superconductor counts
    report.log(f"\n  is_superconductor distribution:")
    report.log(f"    SC: {(combined.is_superconductor == 1).sum()}")
    report.log(f"    Non-SC: {(combined.is_superconductor == 0).sum()}")

    # Spot-check new entries
    if len(new_df) >= 5:
        report.log(f"\n  Spot-check 5 new entries:")
        sample = new_df.sample(min(5, len(new_df)), random_state=42)
        for _, row in sample.iterrows():
            report.log(f"    Formula: {row['formula']}")
            report.log(f"    Tc: {row['Tc']} K")
            report.log(f"    Category: {row['category']}")
            report.log(f"    formula_original: {row['formula_original']}")
            report.log(f"    requires_high_pressure: {row['requires_high_pressure']}")
            report.log(f"    ---")

    # ── Save outputs ──
    OUT_COMBINED.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_COMBINED, index=False)
    report.log(f"\n  Saved combined CSV: {OUT_COMBINED} ({len(combined)} rows)")

    OUT_NEW_ONLY.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(OUT_NEW_ONLY, index=False)
    report.log(f"  Saved new-only CSV: {OUT_NEW_ONLY} ({len(new_df)} rows)")

    elapsed = time.time() - t0
    report.log(f"\n  Total pipeline time: {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    report.log("=" * 70)
    report.log("Pipeline complete.")

    report.save(OUT_REPORT)


if __name__ == "__main__":
    main()
