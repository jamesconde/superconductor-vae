#!/usr/bin/env python3
"""
NEMAD Superconductor Data Ingestion Pipeline

Ingests the NEMAD superconductor dataset and merges it with the existing
SuperCon VAE training data, producing a combined CSV compatible with the
current training pipeline.

Outputs:
  data/processed/supercon_fractions_combined.csv — merged SuperCon + NEMAD
  data/processed/nemad_fractions.csv             — NEMAD-only for inspection
  scratch/nemad_ingest_report.txt                — detailed run report

Usage:
  python scripts/ingest_nemad.py
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
NEMAD_CSV = PROJECT_ROOT / "DSC288 Data" / "DSC288 Data" / "NEMAD" / "NEMAD_superconductor_materials_unique.csv"
EXISTING_CSV = PROJECT_ROOT / "data" / "processed" / "supercon_fractions.csv"
HOLDOUT_JSON = PROJECT_ROOT / "data" / "GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json"

OUT_COMBINED = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_combined.csv"
OUT_NEMAD = PROJECT_ROOT / "data" / "processed" / "nemad_fractions.csv"
OUT_REPORT = PROJECT_ROOT / "scratch" / "nemad_ingest_report.txt"


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
# Step 1: Load & Filter
# ══════════════════════════════════════════════════════════════════════════

def load_and_filter_nemad(path: Path) -> pd.DataFrame:
    """Load NEMAD CSV and filter to experimental-only rows."""
    report.log(f"Step 1: Loading NEMAD from {path}")
    df = pd.read_csv(path)
    report.log(f"  Raw rows: {len(df)}")

    # Filter experimental
    df = df[df["Experimental"].isin(["TRUE", "True"])].copy()
    report.log(f"  After experimental filter: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 2: Extract Numeric Tc
# ══════════════════════════════════════════════════════════════════════════

def parse_tc_text(text: str) -> Optional[float]:
    """
    Parse a Tc text string into a numeric value in Kelvin.
    Returns None if the value cannot be reliably extracted.
    """
    if not isinstance(text, str):
        return None

    s = text.strip()
    if not s or s.lower() in ("nsc", "n/a", "na", "-", "n.s.", "n.s"):
        return None

    # Skip descriptive/qualitative text
    skip_prefixes = (
        "nsc", "no supercond", "not supercond", "not observed",
        "no transition", "variable", "varies", "suppressed",
        "depression", "decreases", "enhanced", "lower than",
        "anomalously", "strongly", "nearly", "rapidly", "broad",
        "filamentary", "abrupt", "behaves", "rises", "plateau",
        "peak", "maximum", "minimum", "pair-breaking", "sc under",
        "above", "below", "between", "change less", "detected",
        "increased", "local", "lower tc", "onset at",
        "pressure-induced", "proximity-induced", "room temp",
        "shows normal", "superconducting above", "traces of",
    )
    s_lower = s.lower()
    if any(s_lower.startswith(p) for p in skip_prefixes):
        return None

    # Skip mK/meV/conditional values
    if "mk" in s_lower or "mev" in s_lower or "µev" in s_lower:
        return None

    # Skip multi-condition values (onset/midpoint splits, pressure-dependent, etc.)
    if any(kw in s_lower for kw in [
        "onset", "midpoint", "zero resistance", "as-synthesized",
        "annealed", "as-prepared", "fast cooling", "slow cooling",
        "sample #", "kbar", "gpa", "pressure", "calculated",
        "over-reduced", "specific heat", "magnetic suscept",
        "resistivity", "pristine", "irrad", "figure",
        "o16", "o18", "tc+", "tc-", "tc1", "tc2", "tc,on",
        "tc,0", "tc^", "tcl", "tcu", "tconset", "tczero",
        "tc,max", "tc(onset", "tc(r=",
        "bicrystal", "single crystal", "thickness",
        "lpa", "hpa", "shunt", "antidot",
    ]):
        return None

    # Skip multi-value entries (commas with K, semicolons)
    if ";" in s or (", " in s and "K" in s):
        return None

    # Skip entries with "/" that indicate dual values
    if "/" in s and "K" in s:
        return None

    # Pattern: "NUMBER K"
    m = re.match(r"^([\d.]+)\s*K$", s)
    if m:
        return float(m.group(1))

    # Pattern: "~NUMBER K" or "∼NUMBER K" or "≈NUMBER K"
    m = re.match(r"^[~∼≈]([\d.]+)\s*K$", s)
    if m:
        return float(m.group(1))

    # Pattern: "NUMBER-NUMBER K" (range → midpoint)
    m = re.match(r"^([\d.]+)\s*[-–—]\s*([\d.]+)\s*K$", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo <= hi:
            return (lo + hi) / 2.0
        return None

    # Pattern: "NUMBER ± NUMBER K"
    m = re.match(r"^([\d.]+)\s*[±]\s*[\d.]+\s*K?$", s)
    if m:
        return float(m.group(1))

    # Pattern: "NUMBER ± NUMBER" (no K)
    m = re.match(r"^([\d.]+)\s*[±]\s*[\d.]+$", s)
    if m:
        return float(m.group(1))

    # Pattern: plain number (no K)
    m = re.match(r"^([\d.]+)$", s)
    if m:
        return float(m.group(1))

    # Pattern: "NUMBER-NUMBER" range without K
    m = re.match(r"^([\d.]+)\s*[-–—]\s*([\d.]+)$", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo <= hi:
            return (lo + hi) / 2.0
        return None

    # Pattern: "≃NUMBER K" or "≤NUMBER K" etc.
    m = re.match(r"^[≃≤≥<>]([\d.]+)\s*K?$", s)
    if m:
        return None  # Inequality - unreliable

    return None


def extract_tc(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric Tc from NEMAD data, using Median_Tc primary, text fallback."""
    report.log("\nStep 2: Extracting numeric Tc")

    tc_values = []
    tc_sources = []

    for _, row in df.iterrows():
        median_tc = row.get("Median_Tc_By_Composition_K")
        tc_text = row.get("Superconducting_Transition_Temperature")

        # Primary: Median_Tc_By_Composition_K
        if pd.notna(median_tc):
            try:
                val = float(median_tc)
                if np.isfinite(val) and val >= 0:
                    tc_values.append(val)
                    tc_sources.append("median")
                    continue
            except (ValueError, TypeError):
                pass

        # Fallback: parse text
        parsed = parse_tc_text(str(tc_text) if pd.notna(tc_text) else "")
        if parsed is not None and np.isfinite(parsed) and parsed >= 0:
            tc_values.append(parsed)
            tc_sources.append("text")
        else:
            tc_values.append(np.nan)
            tc_sources.append("none")
            formula = row.get("Chemical_Composition", "?")
            reason = str(tc_text)[:40] if pd.notna(tc_text) else "no_tc_data"
            report.skip(f"no_valid_tc({reason})", formula)

    df = df.copy()
    df["Tc"] = tc_values
    df["_tc_source"] = tc_sources

    n_median = tc_sources.count("median")
    n_text = tc_sources.count("text")
    n_none = tc_sources.count("none")
    report.log(f"  Tc from Median_Tc: {n_median}")
    report.log(f"  Tc from text parse: {n_text}")
    report.log(f"  No valid Tc (dropped): {n_none}")

    # Drop rows with no Tc
    df = df.dropna(subset=["Tc"])
    report.log(f"  After Tc filter: {len(df)}")

    # Validate: all Tc >= 0, finite
    assert (df["Tc"] >= 0).all(), "Negative Tc values found!"
    assert df["Tc"].apply(np.isfinite).all(), "Non-finite Tc values found!"

    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 3: Formula Normalization
# ══════════════════════════════════════════════════════════════════════════

# Unicode subscript digit mapping
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# Unicode superscript digit mapping
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def clean_nemad_formula(raw: str) -> Optional[str]:
    """
    Clean a NEMAD formula string for pymatgen parsing.

    Returns cleaned formula or None if unparseable.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None

    s = raw.strip()

    # 1. Unicode normalization
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_SUBSCRIPT_MAP)
    s = s.translate(_SUPERSCRIPT_MAP)
    # Middle dot → nothing (sometimes used as separator)
    s = s.replace("·", "").replace("•", "")
    # En-dash, em-dash → hyphen
    s = s.replace("–", "-").replace("—", "-")
    # Thin/non-breaking spaces
    s = s.replace("\u200b", "").replace("\u00a0", " ")

    # 2. Remove delta/variable notation: ±δ, +δ, -δ, +x, -y, etc.
    s = re.sub(r"[±+\-]?\s*[δΔ]", "", s)
    s = re.sub(r"[+\-]\s*[xyzn]\b", "", s)
    s = re.sub(r"[±]\s*\d*\.?\d*\s*[xyzn]?\b", "", s)

    # 3. Strip dopant additions: "+ N wt%", "+ N at%", "with N%", etc.
    s = re.sub(r"\+\s*[\d.]+\s*(wt|at|mol|vol)\s*%.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"with\s+[\d.]+\s*%.*", "", s, flags=re.IGNORECASE)

    # 4. Handle slash composites
    if "/" in s:
        parts = s.split("/")
        # If left side looks like a standalone element/number wrapper e.g. (Ag)0.5/...,
        # take the right side
        left = parts[0].strip()
        right = "/".join(parts[1:]).strip()

        # Check if left is just a dopant prefix like (Ag)0.5 or (Co3O4)0.25
        # These are "additive/base" patterns - take the base (right)
        if re.match(r"^\([A-Za-z0-9]+\)\s*[\d.]+$", left):
            s = right
        else:
            # Take left side by default
            s = left

    # 5. Remove percentage notation entries
    if re.search(r"\d+\s*%", s):
        return None

    # 6. Remove dash-only alloys (e.g., "Nb-Ti" without stoichiometry)
    # But keep formulas like "Nb3-xTixSn" → after variable removal becomes "Nb3Ti0Sn" etc.
    # Only skip if it's ONLY element-dash-element with no numbers
    if re.match(r"^[A-Z][a-z]?\s*-\s*[A-Z][a-z]?$", s):
        return None

    # Remove any remaining hyphens between elements that aren't part of numbers
    # e.g., "La-Ba-Cu-O" → skip these, they have no stoichiometry
    if re.match(r"^([A-Z][a-z]?\s*-\s*){2,}[A-Z][a-z]?$", s):
        return None

    # 7. Remove content in angle brackets or curly braces (rare notation)
    s = re.sub(r"[{}]", "", s)

    # 8. Remove trailing notation like "(CO3)0.9" carbonates that are part of formula - keep
    # Remove trailing comments in parentheses that aren't chemical
    # Only remove if it contains lowercase words
    s = re.sub(r"\s*\([a-z ]+\)\s*$", "", s, flags=re.IGNORECASE)

    # 9. Final whitespace cleanup
    s = s.strip()

    if not s or len(s) < 2:
        return None

    # Check there's at least one element-like capital letter
    if not re.search(r"[A-Z]", s):
        return None

    return s


# ══════════════════════════════════════════════════════════════════════════
# Step 4: pymatgen Composition Parsing
# ══════════════════════════════════════════════════════════════════════════

def parse_compositions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse cleaned formulas with pymatgen, deduplicate within NEMAD."""
    from pymatgen.core import Composition

    report.log("\nStep 3-4: Formula cleaning and pymatgen parsing")

    compositions = []
    canonical_keys = []
    cleaned_formulas = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing formulas"):
        raw = row["Chemical_Composition"]
        cleaned = clean_nemad_formula(raw)

        if cleaned is None:
            compositions.append(None)
            canonical_keys.append(None)
            cleaned_formulas.append(None)
            report.skip("formula_clean_failed", str(raw)[:60])
            continue

        try:
            comp = Composition(cleaned)
            # Validate: at least one element
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
            report.skip(f"pymatgen_parse_error", f"{raw[:40]} → {cleaned[:40]}: {str(e)[:30]}")

    df = df.copy()
    df["_composition"] = compositions
    df["_canonical"] = canonical_keys
    df["_cleaned"] = cleaned_formulas

    n_parsed = sum(c is not None for c in compositions)
    report.log(f"  Successfully parsed: {n_parsed} / {len(df)}")

    # Drop unparseable
    df = df[df["_composition"].notna()].copy()
    report.log(f"  After dropping unparseable: {len(df)}")

    # Deduplicate within NEMAD by canonical formula (keep first)
    before_dedup = len(df)
    df = df.drop_duplicates(subset="_canonical", keep="first")
    report.log(f"  After NEMAD internal dedup: {len(df)} (removed {before_dedup - len(df)})")

    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 5: Category Assignment
# ══════════════════════════════════════════════════════════════════════════

def assign_category(row) -> str:
    """
    Map NEMAD Superconductor_Type + formula composition to existing categories.

    Existing categories: Other, Cuprates, Iron-based, Bismuthates,
    Borocarbides, Organic Superconductors, Elemental Superconductors,
    Hydrogen-rich Superconductors
    """
    sc_type = str(row.get("Superconductor_Type", "")).lower()
    comp = row["_composition"]
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
    if "iron" in sc_type and "Fe" in elements:
        return "Iron-based"

    # Cuprate / High-Tc cuprate detection
    is_cuprate_type = any(kw in sc_type for kw in ["cuprate", "high-tc", "high tc"])
    has_cu = "Cu" in elements
    has_o = "O" in elements

    if (is_cuprate_type or ("cuprate" in sc_type)) and has_cu and has_o:
        # Check for Bi → Bismuthates category
        if "Bi" in elements:
            return "Bismuthates"
        return "Cuprates"

    # Bismuthates (Ba-K-Bi-O type, or Bi-based with O)
    if "Bi" in elements and "O" in elements and not has_cu:
        return "Bismuthates"

    # Borocarbides
    if "B" in elements and "C" in elements and any(
        e in elements for e in ["Y", "Lu", "Er", "Ho", "Dy", "Tm", "Ni"]
    ):
        return "Borocarbides"

    # A15 compounds (often classified as Other in existing data)
    if "a15" in sc_type:
        return "Other"

    # Heavy fermion, Chevrel, conventional → Other
    return "Other"


# ══════════════════════════════════════════════════════════════════════════
# Step 6: Compute 145 Feature Columns
# ══════════════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 145 numeric feature columns (6 norms + 132 Magpie + 4 valence + 3 ion).
    Plus transition metal fraction.
    """
    from matminer.featurizers.composition import (
        ElementProperty, IonProperty, TMetalFraction, ValenceOrbital
    )
    from pymatgen.core import Composition

    report.log("\nStep 6: Computing features with matminer")

    # Set up featurizers
    ep = ElementProperty.from_preset("magpie")
    vo = ValenceOrbital()
    ip = IonProperty()
    tm = TMetalFraction()

    # We need these exact column names from the existing CSV
    magpie_labels = ep.feature_labels()  # 132 columns
    valence_labels = ["avg s valence electrons", "avg p valence electrons",
                      "avg d valence electrons", "avg f valence electrons"]
    ion_labels = ip.feature_labels()  # compound possible, max ionic char, avg ionic char
    tm_labels = tm.feature_labels()   # transition metal fraction

    # Build a dataframe for matminer (needs "composition" column of Composition objects)
    feat_df = pd.DataFrame({"composition": df["_composition"].tolist()})

    # Compute Magpie features
    report.log("  Computing MagpieData features (132 columns)...")
    ep_results = ep.featurize_dataframe(feat_df, "composition", ignore_errors=True)

    # Compute ValenceOrbital (8 features, we only keep first 4)
    report.log("  Computing ValenceOrbital features (4 columns)...")
    vo_results = vo.featurize_dataframe(ep_results, "composition", ignore_errors=True)

    # Compute IonProperty
    report.log("  Computing IonProperty features (3 columns)...")
    ip_results = ip.featurize_dataframe(vo_results, "composition", ignore_errors=True)

    # Compute TMetalFraction
    report.log("  Computing TMetalFraction feature (1 column)...")
    all_results = tm.featurize_dataframe(ip_results, "composition", ignore_errors=True)

    # Drop the matminer composition column and extra valence columns
    # ValenceOrbital produces 8 cols; we only want the first 4 (avg, not frac)
    frac_valence_labels = ["frac s valence electrons", "frac p valence electrons",
                           "frac d valence electrons", "frac f valence electrons"]
    drop_cols = ["composition"] + frac_valence_labels
    for col in drop_cols:
        if col in all_results.columns:
            all_results = all_results.drop(columns=[col])

    # Check for rows where featurization failed (all NaN in Magpie cols)
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
            "0-norm": float(len(fracs)),  # Number of elements
            "2-norm": float(np.linalg.norm(fracs, 2)),
            "3-norm": float(np.linalg.norm(fracs, 3)),
            "5-norm": float(np.linalg.norm(fracs, 5)),
            "7-norm": float(np.linalg.norm(fracs, 7)),
            "10-norm": float(np.linalg.norm(fracs, 10)),
        })
    return pd.DataFrame(norms_data)


# ══════════════════════════════════════════════════════════════════════════
# Step 7: Formula Notation
# ══════════════════════════════════════════════════════════════════════════

def composition_to_fraction_formula(comp) -> str:
    """
    Convert pymatgen Composition to fraction notation.
    e.g., Ag0.002 Al0.998 → Ag(1/500)Al(499/500)
    """
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
    """
    Convert pymatgen Composition to decimal notation.
    e.g., Composition("Ag0.002 Al0.998") → "Ag0.002Al0.998"
    """
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
# Step 8-9: Deduplication Against SuperCon & Holdout Filter
# ══════════════════════════════════════════════════════════════════════════

def load_existing_canonical_keys(csv_path: Path) -> set:
    """Load existing SuperCon compositions as canonical keys."""
    from pymatgen.core import Composition

    report.log(f"\nStep 8: Loading existing SuperCon for dedup from {csv_path}")
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

    report.log(f"\nStep 9: Loading holdout samples from {json_path}")
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
    report.log("NEMAD Superconductor Data Ingestion Pipeline")
    report.log("=" * 70)

    # ── Step 1: Load & Filter ──
    df = load_and_filter_nemad(NEMAD_CSV)

    # ── Step 2: Extract Tc ──
    df = extract_tc(df)

    # ── Steps 3-4: Formula cleaning & pymatgen parsing ──
    df = parse_compositions(df)

    # ── Step 5: Category assignment ──
    report.log("\nStep 5: Assigning categories")
    df["category"] = df.apply(assign_category, axis=1)
    report.log(f"  Category distribution:")
    for cat, count in df["category"].value_counts().items():
        report.log(f"    {cat}: {count}")

    # ── Step 6: Compute features ──
    # First compute Lp norms
    report.log("\nStep 6a: Computing Lp norms")
    norms_df = compute_lp_norms(df["_composition"].tolist())

    # Compute matminer features
    feat_df = compute_features(df)

    # ── Step 7: Formula notation ──
    report.log("\nStep 7: Generating formula notations")
    formulas_frac = []
    formulas_dec = []
    compositions_str = []
    for comp in tqdm(df["_composition"], desc="Formula notation"):
        formulas_frac.append(composition_to_fraction_formula(comp))
        formulas_dec.append(composition_to_decimal_formula(comp))
        compositions_str.append(comp.formula)

    # ── Load existing column order ──
    existing_df = pd.read_csv(EXISTING_CSV, nrows=0)
    target_columns = list(existing_df.columns)
    report.log(f"\nTarget columns: {len(target_columns)} (from existing CSV)")

    # ── Build NEMAD output dataframe ──
    report.log("\nAssembling NEMAD dataframe")

    # Get feature column names from existing
    from matminer.featurizers.composition import (
        ElementProperty, IonProperty, TMetalFraction,
    )
    ep = ElementProperty.from_preset("magpie")
    magpie_labels = ep.feature_labels()
    valence_labels = ["avg s valence electrons", "avg p valence electrons",
                      "avg d valence electrons", "avg f valence electrons"]
    ion_labels = ["compound possible", "max ionic char", "avg ionic char"]
    tm_labels = ["transition metal fraction"]

    # All feature columns in order
    feature_cols = (
        list(norms_df.columns) +
        magpie_labels +
        valence_labels +
        ion_labels +
        tm_labels
    )

    # Assemble all columns into a dict first, then create DataFrame at once
    # (avoids fragmentation warnings from repeated column insertion)
    col_data = OrderedDict()
    col_data["formula"] = formulas_frac
    col_data["Tc"] = df["Tc"].values
    col_data["composition"] = compositions_str
    col_data["category"] = df["category"].values

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

    nemad_out = pd.DataFrame(col_data)

    # Verify column alignment
    assert list(nemad_out.columns) == target_columns, (
        f"Column mismatch!\n"
        f"Expected: {target_columns}\n"
        f"Got:      {list(nemad_out.columns)}"
    )
    report.log(f"  Column alignment verified: {len(nemad_out.columns)} columns match")

    # Drop rows where Magpie featurization fully failed (NaN in core Magpie cols)
    before = len(nemad_out)
    nemad_out = nemad_out.dropna(subset=magpie_labels[:5])
    if len(nemad_out) < before:
        report.log(f"  Dropped {before - len(nemad_out)} rows with failed Magpie featurization")

    # Fill NaN in IonProperty columns (matminer returns NaN when no valid ionic model exists)
    # Default: compound_possible=False, ionic char values=0.0
    n_ion_nan = nemad_out["compound possible"].isna().sum()
    if n_ion_nan > 0:
        report.log(f"  Filling {n_ion_nan} NaN IonProperty rows (compound possible=False, ionic char=0.0)")
        pd.set_option('future.no_silent_downcasting', True)
        nemad_out["compound possible"] = nemad_out["compound possible"].fillna(False).infer_objects(copy=False)
        nemad_out["max ionic char"] = nemad_out["max ionic char"].fillna(0.0).infer_objects(copy=False)
        nemad_out["avg ionic char"] = nemad_out["avg ionic char"].fillna(0.0).infer_objects(copy=False)

    # ── Steps 8-9: Dedup against existing + holdout filter ──
    existing_keys = load_existing_canonical_keys(EXISTING_CSV)
    holdout_keys = load_holdout_keys(HOLDOUT_JSON)

    # Build canonical keys for NEMAD output
    from pymatgen.core import Composition as Comp
    nemad_canonical = []
    for comp_str in nemad_out["composition"]:
        try:
            nemad_canonical.append(Comp(comp_str).alphabetical_formula)
        except Exception:
            nemad_canonical.append(None)
    nemad_out["_canonical"] = nemad_canonical

    # Remove duplicates against existing SuperCon
    before_dedup = len(nemad_out)
    mask_existing = nemad_out["_canonical"].isin(existing_keys)
    nemad_out = nemad_out[~mask_existing].copy()
    report.log(f"  Removed {mask_existing.sum()} entries duplicating existing SuperCon")
    report.log(f"  After existing dedup: {len(nemad_out)}")

    # Remove holdout samples
    mask_holdout = nemad_out["_canonical"].isin(holdout_keys)
    n_holdout_removed = mask_holdout.sum()
    nemad_out = nemad_out[~mask_holdout].copy()
    report.log(f"  Removed {n_holdout_removed} holdout samples")
    report.log(f"  After holdout filter: {len(nemad_out)}")

    # Drop the temporary _canonical column
    nemad_out = nemad_out.drop(columns=["_canonical"])

    # ── Step 10: Merge & Save ──
    report.log("\nStep 10: Merging and saving")

    # Load full existing CSV
    existing_full = pd.read_csv(EXISTING_CSV)
    report.log(f"  Existing SuperCon rows: {len(existing_full)}")
    report.log(f"  New NEMAD rows: {len(nemad_out)}")

    # Verify column match
    assert list(nemad_out.columns) == list(existing_full.columns), "Column mismatch at merge!"

    # Concatenate
    combined = pd.concat([existing_full, nemad_out], ignore_index=True)
    report.log(f"  Combined rows: {len(combined)}")

    # ── Verification ──
    report.log("\n=== VERIFICATION ===")

    # 1. 151 columns with matching names
    assert len(combined.columns) == 151, f"Expected 151 columns, got {len(combined.columns)}"
    assert list(combined.columns) == target_columns
    report.log(f"  [PASS] 151 columns with matching names and order")

    # 2. No NaN in formula or Tc
    assert combined["formula"].notna().all(), "NaN found in formula column!"
    assert combined["Tc"].notna().all(), "NaN found in Tc column!"
    report.log(f"  [PASS] No NaN in formula or Tc columns")

    # 3. No holdout samples
    combined_comps = set()
    for comp_str in combined["composition"]:
        try:
            combined_comps.add(Comp(comp_str).alphabetical_formula)
        except Exception:
            pass
    holdout_in_combined = holdout_keys.intersection(combined_comps)
    # Note: holdout might be in original SuperCon (that's expected), just not in NEMAD additions
    nemad_comps = set()
    for comp_str in nemad_out["composition"]:
        try:
            nemad_comps.add(Comp(comp_str).alphabetical_formula)
        except Exception:
            pass
    assert len(holdout_keys.intersection(nemad_comps)) == 0, "Holdout samples in NEMAD output!"
    report.log(f"  [PASS] No holdout samples in NEMAD additions")

    # 4. Tc distribution stats
    existing_tc = existing_full["Tc"]
    nemad_tc = nemad_out["Tc"]
    combined_tc = combined["Tc"]
    report.log(f"\n  Tc Distribution:")
    report.log(f"    {'':20s} {'Existing':>10s} {'NEMAD':>10s} {'Combined':>10s}")
    report.log(f"    {'Count':20s} {len(existing_tc):10d} {len(nemad_tc):10d} {len(combined_tc):10d}")
    report.log(f"    {'Mean':20s} {existing_tc.mean():10.2f} {nemad_tc.mean():10.2f} {combined_tc.mean():10.2f}")
    report.log(f"    {'Std':20s} {existing_tc.std():10.2f} {nemad_tc.std():10.2f} {combined_tc.std():10.2f}")
    report.log(f"    {'Min':20s} {existing_tc.min():10.2f} {nemad_tc.min():10.2f} {combined_tc.min():10.2f}")
    report.log(f"    {'Max':20s} {existing_tc.max():10.2f} {nemad_tc.max():10.2f} {combined_tc.max():10.2f}")
    report.log(f"    {'Median':20s} {existing_tc.median():10.2f} {nemad_tc.median():10.2f} {combined_tc.median():10.2f}")
    report.log(f"    {'Above 77K':20s} {(existing_tc > 77).sum():10d} {(nemad_tc > 77).sum():10d} {(combined_tc > 77).sum():10d}")

    # 5. Spot-check 5 random NEMAD entries
    report.log(f"\n  Spot-check 5 random NEMAD entries:")
    sample_idx = nemad_out.sample(min(5, len(nemad_out)), random_state=42).index
    for idx in sample_idx:
        row = nemad_out.loc[idx]
        report.log(f"    Formula: {row['formula']}")
        report.log(f"    Tc: {row['Tc']} K")
        report.log(f"    Category: {row['category']}")
        report.log(f"    0-norm: {row['0-norm']}, 2-norm: {row['2-norm']:.6f}")
        report.log(f"    MagpieData mean Number: {row['MagpieData mean Number']:.4f}")
        report.log(f"    transition metal fraction: {row['transition metal fraction']:.4f}")
        report.log(f"    formula_original: {row['formula_original']}")
        report.log(f"    ---")

    # ── Save outputs ──
    OUT_COMBINED.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_COMBINED, index=False)
    report.log(f"\n  Saved combined CSV: {OUT_COMBINED} ({len(combined)} rows)")

    nemad_out.to_csv(OUT_NEMAD, index=False)
    report.log(f"  Saved NEMAD-only CSV: {OUT_NEMAD} ({len(nemad_out)} rows)")

    # ── Category distribution in combined ──
    report.log(f"\n  Combined category distribution:")
    for cat, count in combined["category"].value_counts().items():
        report.log(f"    {cat}: {count}")

    elapsed = time.time() - t0
    report.log(f"\n  Total pipeline time: {elapsed:.1f}s")
    report.log("=" * 70)
    report.log("Pipeline complete.")

    # Save report
    report.save(OUT_REPORT)


if __name__ == "__main__":
    main()
