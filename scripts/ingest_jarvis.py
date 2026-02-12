#!/usr/bin/env python3
"""
JARVIS Superconductor Data Ingestion Pipeline

Ingests JARVIS superconductor datasets (chemical and 3D) and merges them with
the existing SuperCon VAE training data, producing an updated contrastive CSV.

Data Sources:
  1. jarvis_supercon_chem.csv  — 16,414 rows with formula + Tc
  2. jarvis_supercon_3d.csv    — 1,058 rows with JVASP ID + Tc (needs formula lookup)

Outputs:
  data/processed/supercon_fractions_contrastive.csv — updated with JARVIS data
  data/processed/jarvis_fractions.csv               — JARVIS-only for inspection
  scratch/jarvis_ingest_report.txt                  — detailed run report

Usage:
  python scripts/ingest_jarvis.py
"""

import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
JARVIS_CHEM_CSV = PROJECT_ROOT / "data" / "raw" / "jarvis_supercon_chem.csv"
JARVIS_3D_CSV = PROJECT_ROOT / "data" / "raw" / "jarvis_supercon_3d.csv"
EXISTING_CSV = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_contrastive.csv"
HOLDOUT_JSON = PROJECT_ROOT / "data" / "GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json"

OUT_COMBINED = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_contrastive.csv"
OUT_JARVIS = PROJECT_ROOT / "data" / "processed" / "jarvis_fractions.csv"
OUT_REPORT = PROJECT_ROOT / "scratch" / "jarvis_ingest_report.txt"


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
# Step 1: Load JARVIS Chemical Data
# ══════════════════════════════════════════════════════════════════════════

def load_jarvis_chem(path: Path) -> pd.DataFrame:
    """Load JARVIS supercon chem CSV and extract formula + Tc."""
    report.log(f"Step 1a: Loading JARVIS chem from {path}")
    df = pd.read_csv(path)
    report.log(f"  Raw rows: {len(df)}")
    report.log(f"  Columns: {list(df.columns)}")

    # Filter rows with valid formula and Tc
    df = df[df["formula"].notna() & (df["formula"].str.strip() != "")].copy()
    report.log(f"  After formula filter: {len(df)}")

    # Extract Tc
    df["Tc"] = pd.to_numeric(df["tc"], errors="coerce")
    df = df[df["Tc"].notna() & np.isfinite(df["Tc"]) & (df["Tc"] >= 0)].copy()
    report.log(f"  After Tc filter (>= 0, finite): {len(df)}")

    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 1b: Load JARVIS 3D Data (formula lookup by JVASP ID)
# ══════════════════════════════════════════════════════════════════════════

def load_jarvis_3d(path: Path) -> pd.DataFrame:
    """Load JARVIS supercon 3D CSV and try to recover formulas from JVASP IDs."""
    report.log(f"\nStep 1b: Loading JARVIS 3D from {path}")
    df = pd.read_csv(path)
    report.log(f"  Raw rows: {len(df)}")
    report.log(f"  Columns: {list(df.columns)}")

    # These rows have jid + tc but no formula
    df = df[df["jid"].notna() & (df["jid"].str.strip() != "")].copy()
    df["Tc"] = pd.to_numeric(df["tc"], errors="coerce")
    df = df[df["Tc"].notna() & np.isfinite(df["Tc"]) & (df["Tc"] >= 0)].copy()
    report.log(f"  After jid+Tc filter: {len(df)}")

    # Try to look up formulas using jarvis_dft_3d.csv as a local lookup table
    dft3d_path = path.parent / "jarvis_dft_3d.csv"
    if dft3d_path.exists():
        report.log(f"  Looking up formulas from {dft3d_path}")
        dft3d = pd.read_csv(dft3d_path)
        # Build jid → formula mapping
        jid_to_formula = {}
        if "jid" in dft3d.columns and "formula" in dft3d.columns:
            for _, row in dft3d.iterrows():
                if pd.notna(row["jid"]) and pd.notna(row["formula"]):
                    jid_to_formula[str(row["jid"]).strip()] = str(row["formula"]).strip()
            report.log(f"  Built jid→formula lookup: {len(jid_to_formula)} entries")
        else:
            report.log(f"  WARNING: jarvis_dft_3d.csv missing jid/formula columns")

        # Look up formulas
        formulas = []
        for jid in df["jid"]:
            formulas.append(jid_to_formula.get(str(jid).strip(), None))
        df["formula"] = formulas
        n_found = sum(f is not None for f in formulas)
        report.log(f"  Formulas recovered: {n_found} / {len(df)}")

        # Drop rows without formula
        df = df[df["formula"].notna()].copy()
        report.log(f"  After formula recovery filter: {len(df)}")
    else:
        report.log(f"  WARNING: {dft3d_path} not found, trying jarvis-tools package")
        try:
            from jarvis.db.figshare import data as jarvis_data
            dft3d = jarvis_data("dft_3d")
            jid_to_formula = {d["jid"]: d.get("formula", d.get("atoms", {}).get("formula", "")) for d in dft3d}
            report.log(f"  Built jid→formula lookup from jarvis-tools: {len(jid_to_formula)} entries")

            formulas = [jid_to_formula.get(str(jid).strip(), None) for jid in df["jid"]]
            df["formula"] = formulas
            n_found = sum(f is not None for f in formulas)
            report.log(f"  Formulas recovered: {n_found} / {len(df)}")
            df = df[df["formula"].notna()].copy()
        except ImportError:
            report.log("  jarvis-tools not installed. To install: pip install jarvis-tools")
            report.log("  Skipping 3D data — no formula recovery possible")
            return pd.DataFrame(columns=["formula", "Tc"])

    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 2: Formula Parsing with pymatgen
# ══════════════════════════════════════════════════════════════════════════

# Unicode subscript/superscript digit mapping
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def clean_formula(raw: str) -> Optional[str]:
    """Clean a JARVIS formula string for pymatgen parsing."""
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
    """Parse cleaned formulas with pymatgen, deduplicate within JARVIS."""
    from pymatgen.core import Composition

    report.log("\nStep 2: Formula cleaning and pymatgen parsing")

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

    # Deduplicate within JARVIS by canonical formula (keep first)
    before_dedup = len(df)
    df = df.drop_duplicates(subset="_canonical", keep="first")
    report.log(f"  After JARVIS internal dedup: {len(df)} (removed {before_dedup - len(df)})")

    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 3: Category Assignment
# ══════════════════════════════════════════════════════════════════════════

def assign_category(comp) -> str:
    """
    Map composition to SC category using element heuristics.

    Categories: Other, Cuprates, Iron-based, Bismuthates,
    Borocarbides, Organic Superconductors, Elemental Superconductors,
    Hydrogen-rich Superconductors
    """
    elements = {str(e) for e in comp.elements}
    el_fracs = comp.fractional_composition.as_dict()

    # Single element → Elemental
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
        # Check for Bi → Bismuthates category
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
# Step 4: Compute Features (145 standard + 6 physics)
# ══════════════════════════════════════════════════════════════════════════

def compute_physics_features(comp) -> dict:
    """
    Compute 6 physics-informed feature columns from a pymatgen Composition.

    Returns dict with:
      - has_cuprate_elements: Cu AND O present (0/1)
      - cu_o_ratio: Cu fraction / O fraction (0 if missing)
      - has_iron_pnictide: Fe AND (As/Se/P) present (0/1)
      - has_mgb2_elements: Mg AND B present (0/1)
      - hydrogen_fraction: H fraction in composition (float)
      - transition_metal_count: number of distinct TM elements (int)
    """
    elements = {str(e) for e in comp.elements}
    el_fracs = comp.fractional_composition.as_dict()

    # Transition metals (groups 3-12, periods 4-6)
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


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all numeric feature columns:
      - 6 Lp norms
      - 132 Magpie features
      - 4 valence features
      - 3 ion property features
      - 1 transition metal fraction
      - 6 physics features (new V12.28)
    """
    from matminer.featurizers.composition import (
        ElementProperty, IonProperty, TMetalFraction, ValenceOrbital
    )

    report.log("\nStep 4: Computing features with matminer")

    # Set up featurizers
    ep = ElementProperty.from_preset("magpie")
    vo = ValenceOrbital()
    ip = IonProperty()
    tm = TMetalFraction()

    magpie_labels = ep.feature_labels()
    valence_labels = ["avg s valence electrons", "avg p valence electrons",
                      "avg d valence electrons", "avg f valence electrons"]
    ion_labels = ip.feature_labels()
    tm_labels = tm.feature_labels()

    # Build dataframe for matminer
    feat_df = pd.DataFrame({"composition": df["_composition"].tolist()})

    # Compute features
    report.log("  Computing MagpieData features (132 columns)...")
    ep_results = ep.featurize_dataframe(feat_df, "composition", ignore_errors=True)

    report.log("  Computing ValenceOrbital features (4 columns)...")
    vo_results = vo.featurize_dataframe(ep_results, "composition", ignore_errors=True)

    report.log("  Computing IonProperty features (3 columns)...")
    ip_results = ip.featurize_dataframe(vo_results, "composition", ignore_errors=True)

    report.log("  Computing TMetalFraction feature (1 column)...")
    all_results = tm.featurize_dataframe(ip_results, "composition", ignore_errors=True)

    # Drop extra columns
    frac_valence_labels = ["frac s valence electrons", "frac p valence electrons",
                           "frac d valence electrons", "frac f valence electrons"]
    drop_cols = ["composition"] + frac_valence_labels
    for col in drop_cols:
        if col in all_results.columns:
            all_results = all_results.drop(columns=[col])

    # Check for featurization failures
    magpie_mask = all_results[magpie_labels].isna().all(axis=1)
    n_feat_fail = magpie_mask.sum()
    if n_feat_fail > 0:
        report.log(f"  WARNING: {n_feat_fail} rows failed featurization")

    return all_results


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


# ══════════════════════════════════════════════════════════════════════════
# Step 5: Formula Notation
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
# Step 6: Deduplication & Holdout Filter
# ══════════════════════════════════════════════════════════════════════════

def load_existing_canonical_keys(csv_path: Path) -> set:
    """Load existing compositions as canonical keys."""
    from pymatgen.core import Composition

    report.log(f"\nStep 6a: Loading existing data for dedup from {csv_path}")
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

    report.log(f"\nStep 6b: Loading holdout samples from {json_path}")
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
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    report.log("=" * 70)
    report.log("JARVIS Superconductor Data Ingestion Pipeline")
    report.log("=" * 70)

    # ── Step 1: Load JARVIS data ──
    df_chem = load_jarvis_chem(JARVIS_CHEM_CSV)
    df_3d = load_jarvis_3d(JARVIS_3D_CSV)

    # Combine chem + 3d
    report.log(f"\nCombining: {len(df_chem)} chem + {len(df_3d)} 3D")
    # Keep only formula and Tc columns from both
    df_chem_slim = df_chem[["formula", "Tc"]].copy()
    df_3d_slim = df_3d[["formula", "Tc"]].copy() if len(df_3d) > 0 else pd.DataFrame(columns=["formula", "Tc"])
    df = pd.concat([df_chem_slim, df_3d_slim], ignore_index=True)
    report.log(f"  Combined raw JARVIS rows: {len(df)}")

    # ── Step 2: Formula parsing ──
    df = parse_compositions(df)

    # ── Step 3: Category assignment ──
    report.log("\nStep 3: Assigning categories")
    df["category"] = df["_composition"].apply(assign_category)
    report.log(f"  Category distribution:")
    for cat, count in df["category"].value_counts().items():
        report.log(f"    {cat}: {count}")

    # ── Step 4: Compute features ──
    report.log("\nStep 4a: Computing Lp norms")
    norms_df = compute_lp_norms(df["_composition"].tolist())

    feat_df = compute_features(df)

    # Compute physics features
    report.log("\nStep 4b: Computing physics features (6 columns)")
    physics_data = []
    for comp in tqdm(df["_composition"], desc="Physics features"):
        physics_data.append(compute_physics_features(comp))
    physics_df = pd.DataFrame(physics_data)
    report.log(f"  Physics features computed: {len(physics_df)} rows × {len(physics_df.columns)} columns")

    # ── Step 5: Formula notation ──
    report.log("\nStep 5: Generating formula notations")
    formulas_frac = []
    formulas_dec = []
    compositions_str = []
    for comp in tqdm(df["_composition"], desc="Formula notation"):
        formulas_frac.append(composition_to_fraction_formula(comp))
        formulas_dec.append(composition_to_decimal_formula(comp))
        compositions_str.append(comp.formula)

    # ── Determine is_superconductor and requires_high_pressure ──
    # JARVIS: Tc > 0 → superconductor, Tc == 0 → non-superconductor
    is_sc = (df["Tc"].values > 0).astype(int)
    # Default: requires_high_pressure = 0 (JARVIS doesn't have this info)
    requires_hp = np.zeros(len(df), dtype=int)
    # Hydrogen-rich with high Tc likely require high pressure
    for i, comp in enumerate(df["_composition"]):
        el_fracs = comp.fractional_composition.as_dict()
        h_frac = el_fracs.get("H", 0.0)
        if h_frac > 0.3 and df["Tc"].iloc[i] > 100:
            requires_hp[i] = 1

    # ── Load existing column order ──
    existing_df = pd.read_csv(EXISTING_CSV, nrows=0)
    target_columns = list(existing_df.columns)
    report.log(f"\nTarget columns: {len(target_columns)} (from existing CSV)")

    # ── Build JARVIS output dataframe ──
    report.log("\nAssembling JARVIS dataframe")

    from matminer.featurizers.composition import ElementProperty, IonProperty, TMetalFraction
    ep = ElementProperty.from_preset("magpie")
    magpie_labels = ep.feature_labels()
    valence_labels = ["avg s valence electrons", "avg p valence electrons",
                      "avg d valence electrons", "avg f valence electrons"]
    ion_labels = ["compound possible", "max ionic char", "avg ionic char"]
    tm_labels = ["transition metal fraction"]

    # Physics feature column names (V12.28)
    physics_labels = ["has_cuprate_elements", "cu_o_ratio", "has_iron_pnictide",
                      "has_mgb2_elements", "hydrogen_fraction", "transition_metal_count"]

    # Assemble in target column order
    col_data = OrderedDict()
    col_data["formula"] = formulas_frac
    col_data["Tc"] = df["Tc"].values
    col_data["composition"] = compositions_str
    col_data["category"] = df["category"].values
    col_data["is_superconductor"] = is_sc

    # Add norms
    for col in norms_df.columns:
        col_data[col] = norms_df[col].values

    # Add matminer features
    for col in magpie_labels + valence_labels + ion_labels + tm_labels:
        if col in feat_df.columns:
            col_data[col] = feat_df[col].values
        else:
            report.log(f"  WARNING: Missing feature column {col}, filling with 0")
            col_data[col] = 0.0

    col_data["formula_original"] = formulas_dec
    col_data["requires_high_pressure"] = requires_hp

    jarvis_out = pd.DataFrame(col_data)

    # Check if existing CSV has physics feature columns — if not, they'll need to be added
    has_physics_cols = all(col in target_columns for col in physics_labels)
    if has_physics_cols:
        # Add physics features to JARVIS output
        for col in physics_labels:
            jarvis_out[col] = physics_df[col].values
    else:
        report.log(f"  NOTE: Existing CSV lacks physics feature columns — will add them during merge")

    # Verify column alignment (if physics cols already exist)
    if has_physics_cols:
        if list(jarvis_out.columns) != target_columns:
            report.log(f"  WARNING: Column mismatch, reordering to match target")
            jarvis_out = jarvis_out[target_columns]

    # Drop rows where Magpie featurization failed
    before = len(jarvis_out)
    jarvis_out = jarvis_out.dropna(subset=magpie_labels[:5])
    if len(jarvis_out) < before:
        report.log(f"  Dropped {before - len(jarvis_out)} rows with failed Magpie featurization")

    # Fill NaN in IonProperty columns
    n_ion_nan = jarvis_out["compound possible"].isna().sum()
    if n_ion_nan > 0:
        report.log(f"  Filling {n_ion_nan} NaN IonProperty rows")
        pd.set_option('future.no_silent_downcasting', True)
        jarvis_out["compound possible"] = jarvis_out["compound possible"].fillna(False).infer_objects(copy=False)
        jarvis_out["max ionic char"] = jarvis_out["max ionic char"].fillna(0.0).infer_objects(copy=False)
        jarvis_out["avg ionic char"] = jarvis_out["avg ionic char"].fillna(0.0).infer_objects(copy=False)

    # ── Step 6: Dedup against existing + holdout filter ──
    existing_keys = load_existing_canonical_keys(EXISTING_CSV)
    holdout_keys = load_holdout_keys(HOLDOUT_JSON)

    from pymatgen.core import Composition as Comp
    jarvis_canonical = []
    for comp_str in jarvis_out["composition"]:
        try:
            jarvis_canonical.append(Comp(comp_str).alphabetical_formula)
        except Exception:
            jarvis_canonical.append(None)
    jarvis_out["_canonical"] = jarvis_canonical

    # Remove duplicates against existing
    before_dedup = len(jarvis_out)
    mask_existing = jarvis_out["_canonical"].isin(existing_keys)
    jarvis_out = jarvis_out[~mask_existing].copy()
    report.log(f"  Removed {mask_existing.sum()} entries duplicating existing data")
    report.log(f"  After existing dedup: {len(jarvis_out)}")

    # Remove holdout samples
    mask_holdout = jarvis_out["_canonical"].isin(holdout_keys)
    n_holdout_removed = mask_holdout.sum()
    jarvis_out = jarvis_out[~mask_holdout].copy()
    report.log(f"  Removed {n_holdout_removed} holdout samples")
    report.log(f"  After holdout filter: {len(jarvis_out)}")

    # Drop temporary column
    jarvis_out = jarvis_out.drop(columns=["_canonical"])

    # ── Step 7: Merge & Save ──
    report.log("\nStep 7: Merging and saving")

    existing_full = pd.read_csv(EXISTING_CSV)
    report.log(f"  Existing rows: {len(existing_full)}")
    report.log(f"  New JARVIS rows: {len(jarvis_out)}")

    # Add physics features to existing CSV if missing
    if not has_physics_cols:
        report.log("  Adding physics feature columns to existing data...")
        existing_physics = _compute_physics_for_existing(existing_full)
        for col in physics_labels:
            existing_full[col] = existing_physics[col].values
            jarvis_out[col] = physics_df.loc[jarvis_out.index, col].values if col in physics_df.columns else 0.0

    # Ensure column match
    jarvis_cols = set(jarvis_out.columns)
    existing_cols = set(existing_full.columns)
    if jarvis_cols != existing_cols:
        # Add missing columns with defaults
        for col in existing_cols - jarvis_cols:
            jarvis_out[col] = 0
        for col in jarvis_cols - existing_cols:
            existing_full[col] = 0
        # Reorder to match
        jarvis_out = jarvis_out[existing_full.columns]

    # Concatenate
    combined = pd.concat([existing_full, jarvis_out], ignore_index=True)
    report.log(f"  Combined rows: {len(combined)}")

    # ── Verification ──
    report.log("\n=== VERIFICATION ===")

    n_cols = len(combined.columns)
    report.log(f"  Columns: {n_cols}")
    assert combined["formula"].notna().all(), "NaN found in formula column!"
    assert combined["Tc"].notna().all(), "NaN found in Tc column!"
    report.log(f"  [PASS] No NaN in formula or Tc columns")

    # Verify no holdout contamination from JARVIS additions
    jarvis_comps = set()
    for comp_str in jarvis_out["composition"]:
        try:
            jarvis_comps.add(Comp(comp_str).alphabetical_formula)
        except Exception:
            pass
    assert len(holdout_keys.intersection(jarvis_comps)) == 0, "Holdout samples in JARVIS output!"
    report.log(f"  [PASS] No holdout samples in JARVIS additions")

    # Tc distribution
    existing_tc = existing_full["Tc"]
    jarvis_tc = jarvis_out["Tc"]
    combined_tc = combined["Tc"]
    report.log(f"\n  Tc Distribution:")
    report.log(f"    {'':20s} {'Existing':>10s} {'JARVIS':>10s} {'Combined':>10s}")
    report.log(f"    {'Count':20s} {len(existing_tc):10d} {len(jarvis_tc):10d} {len(combined_tc):10d}")
    report.log(f"    {'Mean':20s} {existing_tc.mean():10.2f} {jarvis_tc.mean():10.2f} {combined_tc.mean():10.2f}")
    report.log(f"    {'Std':20s} {existing_tc.std():10.2f} {jarvis_tc.std():10.2f} {combined_tc.std():10.2f}")
    report.log(f"    {'Min':20s} {existing_tc.min():10.2f} {jarvis_tc.min():10.2f} {combined_tc.min():10.2f}")
    report.log(f"    {'Max':20s} {existing_tc.max():10.2f} {jarvis_tc.max():10.2f} {combined_tc.max():10.2f}")
    report.log(f"    {'Median':20s} {existing_tc.median():10.2f} {jarvis_tc.median():10.2f} {combined_tc.median():10.2f}")
    report.log(f"    {'SC (Tc>0)':20s} {(existing_tc > 0).sum():10d} {(jarvis_tc > 0).sum():10d} {(combined_tc > 0).sum():10d}")
    report.log(f"    {'Above 77K':20s} {(existing_tc > 77).sum():10d} {(jarvis_tc > 77).sum():10d} {(combined_tc > 77).sum():10d}")

    # Category distribution
    report.log(f"\n  Combined category distribution:")
    for cat, count in combined["category"].value_counts().items():
        report.log(f"    {cat}: {count}")

    # Spot-check
    if len(jarvis_out) >= 5:
        report.log(f"\n  Spot-check 5 random JARVIS entries:")
        sample_idx = jarvis_out.sample(min(5, len(jarvis_out)), random_state=42).index
        for idx in sample_idx:
            row = jarvis_out.loc[idx]
            report.log(f"    Formula: {row['formula']}")
            report.log(f"    Tc: {row['Tc']} K")
            report.log(f"    Category: {row['category']}")
            report.log(f"    formula_original: {row['formula_original']}")
            report.log(f"    ---")

    # ── Save outputs ──
    OUT_COMBINED.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_COMBINED, index=False)
    report.log(f"\n  Saved combined CSV: {OUT_COMBINED} ({len(combined)} rows)")

    jarvis_out.to_csv(OUT_JARVIS, index=False)
    report.log(f"  Saved JARVIS-only CSV: {OUT_JARVIS} ({len(jarvis_out)} rows)")

    elapsed = time.time() - t0
    report.log(f"\n  Total pipeline time: {elapsed:.1f}s")
    report.log("=" * 70)
    report.log("Pipeline complete.")

    report.save(OUT_REPORT)


def _compute_physics_for_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physics features for existing CSV rows."""
    from pymatgen.core import Composition

    report.log("  Computing physics features for existing data...")
    physics_data = []
    for comp_str in tqdm(df["composition"], desc="Physics features (existing)"):
        try:
            comp = Composition(str(comp_str))
            physics_data.append(compute_physics_features(comp))
        except Exception:
            physics_data.append({
                "has_cuprate_elements": 0,
                "cu_o_ratio": 0.0,
                "has_iron_pnictide": 0,
                "has_mgb2_elements": 0,
                "hydrogen_fraction": 0.0,
                "transition_metal_count": 0,
            })
    return pd.DataFrame(physics_data)


if __name__ == "__main__":
    main()
