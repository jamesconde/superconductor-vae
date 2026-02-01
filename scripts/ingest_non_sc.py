#!/usr/bin/env python3
"""
Non-Superconductor Data Ingestion for Contrastive Learning

Ingests non-superconductor materials from NEMAD (magnetic, thermoelectric,
anisotropy) and Materials Project into the training format, adds an
`is_superconductor` column, and produces a unified contrastive-ready dataset.

Target: ~23K non-SC samples to match the ~23K SC samples (1:1 ratio).

Outputs:
  data/processed/supercon_fractions_contrastive.csv — SC + non-SC with is_superconductor column (152 cols)
  data/processed/non_sc_fractions.csv              — non-SC only for inspection
  scratch/non_sc_ingest_report.txt                 — detailed run report

Usage:
  python scripts/ingest_non_sc.py
"""

import json
import os
import sys
import time
from collections import Counter, OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Non-SC data sources
NEMAD_DIR = PROJECT_ROOT / "DSC288 Data" / "DSC288 Data" / "NEMAD"
MAGNETIC_CSV = NEMAD_DIR / "magnetic_materials.csv"
THERMOELECTRIC_CSV = NEMAD_DIR / "thermoelectric_materials.csv"
ANISOTROPY_CSV = NEMAD_DIR / "magnetic_anisotropy_materials.csv"
MP_CSV = PROJECT_ROOT / "DSC288 Data" / "DSC288 Data" / "materials_project.csv"

# Existing SC data
SC_COMBINED_CSV = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_combined.csv"
HOLDOUT_JSON = PROJECT_ROOT / "data" / "GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json"

# Output paths
OUT_CONTRASTIVE = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_contrastive.csv"
OUT_NON_SC = PROJECT_ROOT / "data" / "processed" / "non_sc_fractions.csv"
OUT_REPORT = PROJECT_ROOT / "scratch" / "non_sc_ingest_report.txt"

# Sampling targets
TARGET_NON_SC = 23400  # ~1:1 with SC data (23,451)
MP_SHARE = 0.60        # 60% from Materials Project
NEMAD_SHARE = 0.40     # 40% from NEMAD
NEMAD_FE_MN_CO_CAP = 0.30  # Cap Fe/Mn/Co-only compounds at 30% of NEMAD share

RANDOM_SEED = 42


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
# Reuse formula cleaning from ingest_nemad.py
# ══════════════════════════════════════════════════════════════════════════

# Import clean_nemad_formula from existing script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from ingest_nemad import clean_nemad_formula


# ══════════════════════════════════════════════════════════════════════════
# Step 1: Load All Non-SC Sources
# ══════════════════════════════════════════════════════════════════════════

def load_all_sources() -> Dict[str, pd.DataFrame]:
    """Load all non-SC data sources, extract formulas."""
    sources = {}

    report.log("=" * 70)
    report.log("Step 1: Loading all non-SC data sources")
    report.log("=" * 70)

    # NEMAD Magnetic
    report.log(f"\n  Loading NEMAD Magnetic from {MAGNETIC_CSV}")
    df_mag = pd.read_csv(MAGNETIC_CSV)
    report.log(f"    Raw rows: {len(df_mag)}")
    df_mag = df_mag[["Material_Name"]].copy()
    df_mag.columns = ["raw_formula"]
    df_mag["source"] = "magnetic"
    sources["magnetic"] = df_mag

    # NEMAD Thermoelectric
    report.log(f"\n  Loading NEMAD Thermoelectric from {THERMOELECTRIC_CSV}")
    df_thermo = pd.read_csv(THERMOELECTRIC_CSV)
    report.log(f"    Raw rows: {len(df_thermo)}")
    df_thermo = df_thermo[["Material_Name"]].copy()
    df_thermo.columns = ["raw_formula"]
    df_thermo["source"] = "thermoelectric"
    sources["thermoelectric"] = df_thermo

    # NEMAD Magnetic Anisotropy
    report.log(f"\n  Loading NEMAD Anisotropy from {ANISOTROPY_CSV}")
    df_aniso = pd.read_csv(ANISOTROPY_CSV)
    report.log(f"    Raw rows: {len(df_aniso)}")
    df_aniso = df_aniso[["Material_Name"]].copy()
    df_aniso.columns = ["raw_formula"]
    df_aniso["source"] = "anisotropy"
    sources["anisotropy"] = df_aniso

    # Materials Project
    report.log(f"\n  Loading Materials Project from {MP_CSV}")
    df_mp = pd.read_csv(MP_CSV)
    report.log(f"    Raw rows: {len(df_mp)}")
    # Keep is_stable and band_gap for sampling strategy
    mp_data = df_mp[["formula"]].copy()
    mp_data.columns = ["raw_formula"]
    mp_data["source"] = "materials_project"
    mp_data["is_stable"] = df_mp["is_stable"].values if "is_stable" in df_mp.columns else True
    mp_data["band_gap"] = df_mp["band_gap"].values if "band_gap" in df_mp.columns else 0.0
    sources["materials_project"] = mp_data

    total = sum(len(v) for v in sources.values())
    report.log(f"\n  Total raw entries across all sources: {total}")

    return sources


# ══════════════════════════════════════════════════════════════════════════
# Step 2: Deduplicate & Clean Formulas
# ══════════════════════════════════════════════════════════════════════════

def clean_and_parse_all(sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Clean formulas, parse with pymatgen, deduplicate across all sources."""
    from pymatgen.core import Composition

    report.log("\n" + "=" * 70)
    report.log("Step 2: Cleaning, parsing, and deduplicating formulas")
    report.log("=" * 70)

    # Concatenate all sources (priority order: magnetic > thermoelectric > anisotropy > MP)
    priority_order = ["magnetic", "thermoelectric", "anisotropy", "materials_project"]
    all_entries = []
    for src in priority_order:
        if src in sources:
            all_entries.append(sources[src])

    combined = pd.concat(all_entries, ignore_index=True)
    report.log(f"\n  Combined raw entries: {len(combined)}")

    # Clean formulas
    report.log("  Cleaning formulas...")
    cleaned = []
    for _, row in tqdm(combined.iterrows(), total=len(combined), desc="Cleaning formulas"):
        raw = row["raw_formula"]
        if row["source"] == "materials_project":
            # MP formulas are already clean pymatgen-style strings
            cleaned.append(str(raw).strip() if pd.notna(raw) else None)
        else:
            cleaned.append(clean_nemad_formula(str(raw)) if pd.notna(raw) else None)
    combined["cleaned"] = cleaned

    # Drop failed cleaning
    before = len(combined)
    combined = combined[combined["cleaned"].notna()].copy()
    report.log(f"  After cleaning: {len(combined)} (dropped {before - len(combined)})")

    # Parse with pymatgen
    report.log("  Parsing with pymatgen...")
    compositions = []
    canonical_keys = []
    for _, row in tqdm(combined.iterrows(), total=len(combined), desc="Pymatgen parsing"):
        try:
            comp = Composition(row["cleaned"])
            if len(comp.elements) == 0:
                raise ValueError("No elements")
            compositions.append(comp)
            canonical_keys.append(comp.alphabetical_formula)
        except Exception as e:
            compositions.append(None)
            canonical_keys.append(None)
            report.skip("pymatgen_parse_error", f"{row['cleaned'][:40]}: {str(e)[:30]}")

    combined["_composition"] = compositions
    combined["_canonical"] = canonical_keys

    before = len(combined)
    combined = combined[combined["_composition"].notna()].copy()
    report.log(f"  After pymatgen parsing: {len(combined)} (dropped {before - len(combined)})")

    # Deduplicate by canonical formula (keep first occurrence, priority order already set)
    before = len(combined)
    combined = combined.drop_duplicates(subset="_canonical", keep="first")
    report.log(f"  After cross-source dedup: {len(combined)} (removed {before - len(combined)} duplicates)")

    # Report per-source counts after dedup
    for src in priority_order:
        count = (combined["source"] == src).sum()
        report.log(f"    {src}: {count} unique formulas")

    return combined


def remove_sc_and_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any formulas that match existing SC compositions or holdout samples."""
    from pymatgen.core import Composition

    report.log("\n" + "=" * 70)
    report.log("Step 2b: Removing SC overlaps and holdout samples")
    report.log("=" * 70)

    # Load existing SC canonical keys
    report.log(f"  Loading SC compositions from {SC_COMBINED_CSV}")
    sc_df = pd.read_csv(SC_COMBINED_CSV, usecols=["composition"])
    sc_keys = set()
    for comp_str in tqdm(sc_df["composition"], desc="Canonicalizing SC"):
        try:
            sc_keys.add(Composition(str(comp_str)).alphabetical_formula)
        except Exception:
            pass
    report.log(f"  SC canonical keys: {len(sc_keys)}")

    # Load holdout keys
    report.log(f"  Loading holdout samples from {HOLDOUT_JSON}")
    with open(HOLDOUT_JSON) as f:
        holdout = json.load(f)
    holdout_keys = set()
    for sample in holdout["holdout_samples"]:
        try:
            holdout_keys.add(Composition(sample["formula"]).alphabetical_formula)
        except Exception:
            pass
    report.log(f"  Holdout canonical keys: {len(holdout_keys)}")

    # Remove overlaps
    before = len(df)
    mask_sc = df["_canonical"].isin(sc_keys)
    n_sc_overlap = mask_sc.sum()
    df = df[~mask_sc].copy()
    report.log(f"  Removed {n_sc_overlap} SC overlaps")

    mask_holdout = df["_canonical"].isin(holdout_keys)
    n_holdout = mask_holdout.sum()
    df = df[~mask_holdout].copy()
    report.log(f"  Removed {n_holdout} holdout overlaps")

    report.log(f"  After SC/holdout removal: {len(df)} (was {before})")

    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 3: Representative Sampling to Target Count
# ══════════════════════════════════════════════════════════════════════════

def get_element_set(comp) -> set:
    """Get set of element symbols from a pymatgen Composition."""
    return {str(e) for e in comp.elements}


def representative_sample(df: pd.DataFrame, target: int) -> pd.DataFrame:
    """
    Sample ~target entries with representative chemical diversity.

    Strategy:
    - 60% from Materials Project (broad chemical diversity)
    - 40% from NEMAD (experimental grounding)
    - Cap Fe/Mn/Co-dominated NEMAD compounds
    """
    rng = np.random.RandomState(RANDOM_SEED)

    report.log("\n" + "=" * 70)
    report.log(f"Step 3: Representative sampling to {target} entries")
    report.log("=" * 70)

    mp_target = int(target * MP_SHARE)
    nemad_target = target - mp_target

    report.log(f"  MP target: {mp_target}")
    report.log(f"  NEMAD target: {nemad_target}")

    # ── Materials Project sampling ──
    mp_df = df[df["source"] == "materials_project"].copy()
    report.log(f"\n  MP available: {len(mp_df)}")

    if len(mp_df) <= mp_target:
        mp_sampled = mp_df
        report.log(f"  MP: Using all {len(mp_sampled)} (fewer than target)")
    else:
        # Prefer stable compounds
        if "is_stable" in mp_df.columns:
            stable = mp_df[mp_df["is_stable"] == True]
            unstable = mp_df[mp_df["is_stable"] != True]
            report.log(f"  MP stable: {len(stable)}, unstable: {len(unstable)}")

            # Take as many stable as possible, fill remainder with unstable
            if len(stable) >= mp_target:
                mp_sampled = stable.sample(n=mp_target, random_state=RANDOM_SEED)
            else:
                remainder = mp_target - len(stable)
                unstable_sample = unstable.sample(n=min(remainder, len(unstable)), random_state=RANDOM_SEED)
                mp_sampled = pd.concat([stable, unstable_sample])
        else:
            mp_sampled = mp_df.sample(n=mp_target, random_state=RANDOM_SEED)

    report.log(f"  MP sampled: {len(mp_sampled)}")

    # ── NEMAD sampling ──
    nemad_sources = ["magnetic", "thermoelectric", "anisotropy"]
    nemad_df = df[df["source"].isin(nemad_sources)].copy()
    report.log(f"\n  NEMAD available: {len(nemad_df)}")

    # Count per NEMAD source
    for src in nemad_sources:
        report.log(f"    {src}: {(nemad_df['source'] == src).sum()}")

    # Cap Fe/Mn/Co-only compounds in NEMAD
    fe_mn_co_only = {"Fe", "Mn", "Co"}

    def is_fe_mn_co_dominated(comp):
        """Check if compound only contains Fe, Mn, Co (plus O)."""
        elements = get_element_set(comp)
        non_o = elements - {"O"}
        return non_o.issubset(fe_mn_co_only) and len(non_o) > 0

    nemad_df["_fe_mn_co"] = nemad_df["_composition"].apply(is_fe_mn_co_dominated)
    n_dominated = nemad_df["_fe_mn_co"].sum()
    report.log(f"  Fe/Mn/Co-dominated NEMAD compounds: {n_dominated}")

    # Split NEMAD into dominated and diverse
    nemad_dominated = nemad_df[nemad_df["_fe_mn_co"]].copy()
    nemad_diverse = nemad_df[~nemad_df["_fe_mn_co"]].copy()

    # Cap dominated at 30% of NEMAD share
    max_dominated = int(nemad_target * NEMAD_FE_MN_CO_CAP)

    # Sample NEMAD sources with approximate targets
    # Magnetic: ~4000, Thermoelectric: ~3400, Anisotropy: ~2000
    mag_target = int(nemad_target * 0.43)
    thermo_target = int(nemad_target * 0.36)
    aniso_target = nemad_target - mag_target - thermo_target

    sampled_parts = []

    for src, src_target in [("magnetic", mag_target), ("thermoelectric", thermo_target), ("anisotropy", aniso_target)]:
        src_diverse = nemad_diverse[nemad_diverse["source"] == src]
        src_dominated = nemad_dominated[nemad_dominated["source"] == src]

        # Allocate dominated cap proportionally
        src_dom_cap = int(max_dominated * src_target / nemad_target)

        # Take diverse first, then fill with capped dominated
        if len(src_diverse) >= src_target:
            sampled = src_diverse.sample(n=src_target, random_state=RANDOM_SEED)
        else:
            # Take all diverse, fill with dominated up to cap
            remainder = src_target - len(src_diverse)
            dom_take = min(remainder, src_dom_cap, len(src_dominated))
            dom_sample = src_dominated.sample(n=dom_take, random_state=RANDOM_SEED) if dom_take > 0 else pd.DataFrame()
            sampled = pd.concat([src_diverse, dom_sample])

        report.log(f"    {src}: sampled {len(sampled)} (diverse: {min(len(src_diverse), src_target)}, dominated: {max(0, len(sampled) - min(len(src_diverse), src_target))})")
        sampled_parts.append(sampled)

    nemad_sampled = pd.concat(sampled_parts, ignore_index=True)
    report.log(f"  NEMAD sampled: {len(nemad_sampled)}")

    # ── Combine ──
    result = pd.concat([mp_sampled, nemad_sampled], ignore_index=True)
    report.log(f"\n  Total sampled: {len(result)}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# Step 4: Compute Features (identical to SC pipeline)
# ══════════════════════════════════════════════════════════════════════════

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


def compute_features(compositions) -> pd.DataFrame:
    """
    Compute all matminer feature columns: 132 Magpie + 4 valence + 3 ion + 1 TM fraction.
    """
    import warnings
    from matminer.featurizers.composition import (
        ElementProperty, IonProperty, TMetalFraction, ValenceOrbital
    )

    report.log("\nStep 4: Computing features with matminer")

    ep = ElementProperty.from_preset("magpie")
    vo = ValenceOrbital()
    ip = IonProperty()
    tm = TMetalFraction()

    # Use n_jobs=1 to avoid multiprocessing pipe deadlocks from warning flood
    ep.set_n_jobs(1)
    vo.set_n_jobs(1)
    ip.set_n_jobs(1)
    tm.set_n_jobs(1)

    # Build a dataframe for matminer
    feat_df = pd.DataFrame({"composition": compositions})

    # Suppress Pauling electronegativity warnings (harmless, floods output)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No Pauling electronegativity")

        # Compute Magpie features
        report.log("  Computing MagpieData features (132 columns)...")
        feat_df = ep.featurize_dataframe(feat_df, "composition", ignore_errors=True)

        # Compute ValenceOrbital (8 features, we only keep first 4)
        report.log("  Computing ValenceOrbital features (4 columns)...")
        feat_df = vo.featurize_dataframe(feat_df, "composition", ignore_errors=True)

        # Compute IonProperty with per-composition timeout (some MP compositions
        # have many elements causing combinatorial explosion in oxidation state enumeration)
        report.log("  Computing IonProperty features (3 columns) with 30s/composition timeout...")
        import signal

        ION_TIMEOUT_SEC = 30  # Kill compositions that take longer than this

        class IonPropertyTimeout(Exception):
            pass

        def _ion_timeout_handler(signum, frame):
            raise IonPropertyTimeout()

        ip_labels = ip.feature_labels()
        ion_results = []
        n_timeouts = 0
        from tqdm import tqdm as _tqdm
        for comp in _tqdm(compositions, desc="IonProperty"):
            old_handler = signal.signal(signal.SIGALRM, _ion_timeout_handler)
            signal.alarm(ION_TIMEOUT_SEC)
            try:
                result = ip.featurize(comp)
                ion_results.append(result)
            except (IonPropertyTimeout, Exception):
                # Timeout or error — use defaults (False, 0.0, 0.0)
                ion_results.append([False, 0.0, 0.0])
                n_timeouts += 1
            finally:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)

        for i, label in enumerate(ip_labels):
            feat_df[label] = [r[i] for r in ion_results]
        if n_timeouts > 0:
            report.log(f"  IonProperty: {n_timeouts} compositions timed out (>{ION_TIMEOUT_SEC}s), used defaults")

        # Compute TMetalFraction
        report.log("  Computing TMetalFraction feature (1 column)...")
        feat_df = tm.featurize_dataframe(feat_df, "composition", ignore_errors=True)

    # Drop extra columns
    frac_valence_labels = ["frac s valence electrons", "frac p valence electrons",
                           "frac d valence electrons", "frac f valence electrons"]
    drop_cols = ["composition"] + frac_valence_labels
    for col in drop_cols:
        if col in feat_df.columns:
            feat_df = feat_df.drop(columns=[col])

    return feat_df


# ══════════════════════════════════════════════════════════════════════════
# Step 5: Formula Notation (reuse from ingest_nemad.py)
# ══════════════════════════════════════════════════════════════════════════

from ingest_nemad import composition_to_fraction_formula, composition_to_decimal_formula


# ══════════════════════════════════════════════════════════════════════════
# Step 6-7: Build Non-SC DataFrame & Update SC Data
# ══════════════════════════════════════════════════════════════════════════

def build_non_sc_dataframe(sampled_df: pd.DataFrame, target_columns_152: List[str]) -> pd.DataFrame:
    """Build the non-SC dataframe with 152 columns matching the contrastive schema."""
    from matminer.featurizers.composition import ElementProperty

    report.log("\n" + "=" * 70)
    report.log("Step 5-6: Building non-SC dataframe")
    report.log("=" * 70)

    compositions = sampled_df["_composition"].tolist()

    # Compute Lp norms
    report.log("  Computing Lp norms...")
    norms_df = compute_lp_norms(compositions)

    # Compute matminer features
    feat_df = compute_features(compositions)

    # Formula notation
    report.log("  Generating formula notations...")
    formulas_frac = []
    formulas_dec = []
    compositions_str = []
    for comp in tqdm(compositions, desc="Formula notation"):
        formulas_frac.append(composition_to_fraction_formula(comp))
        formulas_dec.append(composition_to_decimal_formula(comp))
        compositions_str.append(comp.formula)

    # Assign categories based on source
    source_to_category = {
        "magnetic": "Non-SC: Magnetic",
        "thermoelectric": "Non-SC: Thermoelectric",
        "anisotropy": "Non-SC: Anisotropy",
        "materials_project": "Non-SC: Materials Project",
    }
    categories = [source_to_category[src] for src in sampled_df["source"].values]

    # Get feature column names
    ep = ElementProperty.from_preset("magpie")
    magpie_labels = ep.feature_labels()
    valence_labels = ["avg s valence electrons", "avg p valence electrons",
                      "avg d valence electrons", "avg f valence electrons"]
    ion_labels = ["compound possible", "max ionic char", "avg ionic char"]
    tm_labels = ["transition metal fraction"]

    # Assemble all columns
    col_data = OrderedDict()
    col_data["formula"] = formulas_frac
    col_data["Tc"] = 0.0  # Non-SC → Tc = 0
    col_data["composition"] = compositions_str
    col_data["category"] = categories
    col_data["is_superconductor"] = 0  # Non-SC flag

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

    non_sc_df = pd.DataFrame(col_data)

    # Verify column alignment with target schema
    assert list(non_sc_df.columns) == target_columns_152, (
        f"Column mismatch!\n"
        f"Expected ({len(target_columns_152)}): {target_columns_152[:10]}...\n"
        f"Got ({len(non_sc_df.columns)}): {list(non_sc_df.columns)[:10]}..."
    )
    report.log(f"  Column alignment verified: {len(non_sc_df.columns)} columns match target schema")

    # Drop rows where Magpie featurization fully failed
    before = len(non_sc_df)
    non_sc_df = non_sc_df.dropna(subset=magpie_labels[:5])
    if len(non_sc_df) < before:
        report.log(f"  Dropped {before - len(non_sc_df)} rows with failed Magpie featurization")

    # Fill NaN in IonProperty columns
    n_ion_nan = non_sc_df["compound possible"].isna().sum()
    if n_ion_nan > 0:
        report.log(f"  Filling {n_ion_nan} NaN IonProperty rows")
        pd.set_option('future.no_silent_downcasting', True)
        non_sc_df["compound possible"] = non_sc_df["compound possible"].fillna(False).infer_objects(copy=False)
        non_sc_df["max ionic char"] = non_sc_df["max ionic char"].fillna(0.0).infer_objects(copy=False)
        non_sc_df["avg ionic char"] = non_sc_df["avg ionic char"].fillna(0.0).infer_objects(copy=False)

    report.log(f"  Final non-SC dataframe: {len(non_sc_df)} rows × {len(non_sc_df.columns)} cols")

    return non_sc_df


# ══════════════════════════════════════════════════════════════════════════
# Element Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════

def compute_element_distribution(compositions) -> Counter:
    """Count element frequency across a list of pymatgen Compositions."""
    counter = Counter()
    for comp in compositions:
        for el in comp.elements:
            counter[str(el)] += 1
    return counter


def element_distribution_report(
    sc_comps, non_sc_comps, mp_comps,
    sc_label="SC", non_sc_label="Non-SC", mp_label="MP (reference)"
):
    """Print top-30 element distributions and flag over/under-representation."""
    report.log("\n" + "=" * 70)
    report.log("Element Distribution Analysis")
    report.log("=" * 70)

    sc_dist = compute_element_distribution(sc_comps)
    non_sc_dist = compute_element_distribution(non_sc_comps)
    mp_dist = compute_element_distribution(mp_comps)

    sc_total = sum(sc_dist.values())
    non_sc_total = sum(non_sc_dist.values())
    mp_total = sum(mp_dist.values())

    # Get all elements, sorted by MP frequency
    all_elements = set(sc_dist.keys()) | set(non_sc_dist.keys()) | set(mp_dist.keys())

    # Top 30 by MP frequency
    mp_sorted = sorted(all_elements, key=lambda e: mp_dist.get(e, 0), reverse=True)[:30]

    report.log(f"\n  {'Element':>8s}  {sc_label+' %':>10s}  {non_sc_label+' %':>10s}  {mp_label+' %':>12s}  {'Ratio':>8s}")
    report.log(f"  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 12}  {'─' * 8}")

    flagged = []
    for el in mp_sorted:
        sc_pct = 100.0 * sc_dist.get(el, 0) / sc_total if sc_total > 0 else 0
        non_sc_pct = 100.0 * non_sc_dist.get(el, 0) / non_sc_total if non_sc_total > 0 else 0
        mp_pct = 100.0 * mp_dist.get(el, 0) / mp_total if mp_total > 0 else 0

        ratio_str = ""
        if mp_pct > 0.1:  # Only flag if MP has meaningful frequency
            ratio = non_sc_pct / mp_pct if mp_pct > 0 else float("inf")
            ratio_str = f"{ratio:.2f}x"
            if ratio > 3.0 or ratio < 1.0 / 3.0:
                flagged.append((el, ratio, non_sc_pct, mp_pct))
                ratio_str += " ⚠"

        report.log(f"  {el:>8s}  {sc_pct:>9.1f}%  {non_sc_pct:>9.1f}%  {mp_pct:>11.1f}%  {ratio_str:>8s}")

    if flagged:
        report.log(f"\n  ⚠ Flagged elements (>3x over/under-represented vs MP):")
        for el, ratio, non_sc_pct, mp_pct in flagged:
            direction = "OVER" if ratio > 3.0 else "UNDER"
            report.log(f"    {el}: {direction}-represented ({ratio:.2f}x vs MP, non-SC={non_sc_pct:.1f}% vs MP={mp_pct:.1f}%)")
    else:
        report.log(f"\n  No elements flagged (all within 3x of MP baseline)")


# ══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    report.log("=" * 70)
    report.log("Non-Superconductor Data Ingestion for Contrastive Learning")
    report.log("=" * 70)
    report.log(f"  Target non-SC samples: {TARGET_NON_SC}")
    report.log(f"  Random seed: {RANDOM_SEED}")

    # ── Step 1: Load all sources ──
    sources = load_all_sources()

    # ── Step 2: Clean, parse, deduplicate ──
    combined = clean_and_parse_all(sources)

    # Remove SC overlaps and holdout
    combined = remove_sc_and_holdout(combined)

    # ── Step 3: Representative sampling ──
    sampled = representative_sample(combined, TARGET_NON_SC)

    # ── Define 152-column target schema ──
    # Load existing SC CSV to get column order, then insert is_superconductor
    existing_sc = pd.read_csv(SC_COMBINED_CSV, nrows=0)
    old_columns = list(existing_sc.columns)  # 151 columns

    # Insert is_superconductor after category (position 4)
    target_columns_152 = old_columns[:4] + ["is_superconductor"] + old_columns[4:]
    report.log(f"\n  Target schema: {len(target_columns_152)} columns")
    report.log(f"  New column 'is_superconductor' at position 5 (after 'category')")

    # ── Steps 4-6: Compute features & build non-SC dataframe ──
    non_sc_df = build_non_sc_dataframe(sampled, target_columns_152)

    # ── Step 6: Update SC data with is_superconductor column ──
    report.log("\n" + "=" * 70)
    report.log("Step 7: Updating SC data with is_superconductor column")
    report.log("=" * 70)

    sc_df = pd.read_csv(SC_COMBINED_CSV)
    report.log(f"  Loaded SC data: {len(sc_df)} rows × {len(sc_df.columns)} cols")

    # Insert is_superconductor = 1 after category
    cat_pos = list(sc_df.columns).index("category")
    sc_df.insert(cat_pos + 1, "is_superconductor", 1)
    report.log(f"  Inserted is_superconductor=1 at position {cat_pos + 1}")
    report.log(f"  SC schema now: {len(sc_df.columns)} columns")

    # Verify columns match
    assert list(sc_df.columns) == target_columns_152, (
        f"SC column mismatch after adding is_superconductor!\n"
        f"Expected: {target_columns_152[:6]}...\n"
        f"Got: {list(sc_df.columns)[:6]}..."
    )

    # ── Step 7: Merge & Save ──
    report.log("\n" + "=" * 70)
    report.log("Step 8: Merging and saving")
    report.log("=" * 70)

    contrastive = pd.concat([sc_df, non_sc_df], ignore_index=True)
    report.log(f"  SC rows: {len(sc_df)}")
    report.log(f"  Non-SC rows: {len(non_sc_df)}")
    report.log(f"  Combined contrastive rows: {len(contrastive)}")

    # ══════════════════════════════════════════════════════════════════════
    # Verification
    # ══════════════════════════════════════════════════════════════════════
    report.log("\n" + "=" * 70)
    report.log("VERIFICATION")
    report.log("=" * 70)

    # 1. Assert 152 columns
    assert len(contrastive.columns) == 152, f"Expected 152 columns, got {len(contrastive.columns)}"
    report.log(f"  [PASS] 152 columns")

    # 2. No NaN in critical columns
    assert contrastive["formula"].notna().all(), "NaN in formula!"
    assert contrastive["Tc"].notna().all(), "NaN in Tc!"
    assert contrastive["is_superconductor"].notna().all(), "NaN in is_superconductor!"
    report.log(f"  [PASS] No NaN in formula, Tc, or is_superconductor")

    # 3. is_superconductor is exactly {0, 1}
    assert set(contrastive["is_superconductor"].unique()) == {0, 1}, \
        f"is_superconductor values: {contrastive['is_superconductor'].unique()}"
    sc_count = (contrastive["is_superconductor"] == 1).sum()
    non_sc_count = (contrastive["is_superconductor"] == 0).sum()
    report.log(f"  [PASS] is_superconductor: SC={sc_count}, non-SC={non_sc_count}")

    # 4. SC rows match original (minus the new column)
    sc_part = contrastive[contrastive["is_superconductor"] == 1].copy()
    sc_orig = pd.read_csv(SC_COMBINED_CSV)
    assert len(sc_part) == len(sc_orig), f"SC row count mismatch: {len(sc_part)} vs {len(sc_orig)}"
    # Check a few values match
    for col in ["formula", "Tc", "composition", "category"]:
        assert (sc_part[col].values == sc_orig[col].values).all(), f"SC data mismatch in column {col}"
    report.log(f"  [PASS] SC rows match original combined CSV")

    # 5. No SC compositions in non-SC rows
    from pymatgen.core import Composition
    sc_keys = set()
    for comp_str in tqdm(sc_orig["composition"], desc="Verifying no SC in non-SC"):
        try:
            sc_keys.add(Composition(str(comp_str)).alphabetical_formula)
        except Exception:
            pass

    non_sc_part = contrastive[contrastive["is_superconductor"] == 0]
    non_sc_canonical = set()
    for comp_str in non_sc_part["composition"]:
        try:
            non_sc_canonical.add(Composition(str(comp_str)).alphabetical_formula)
        except Exception:
            pass
    overlap = sc_keys.intersection(non_sc_canonical)
    assert len(overlap) == 0, f"SC/non-SC overlap: {len(overlap)} compositions!"
    report.log(f"  [PASS] No SC compositions in non-SC rows")

    # 6. No holdout samples in non-SC rows
    with open(HOLDOUT_JSON) as f:
        holdout = json.load(f)
    holdout_keys = set()
    for sample in holdout["holdout_samples"]:
        try:
            holdout_keys.add(Composition(sample["formula"]).alphabetical_formula)
        except Exception:
            pass
    holdout_in_non_sc = holdout_keys.intersection(non_sc_canonical)
    assert len(holdout_in_non_sc) == 0, f"Holdout in non-SC: {holdout_in_non_sc}"
    report.log(f"  [PASS] No holdout samples in non-SC rows")

    # 7. Distribution stats
    report.log(f"\n  Category breakdown:")
    for cat, count in contrastive["category"].value_counts().items():
        report.log(f"    {cat}: {count}")

    report.log(f"\n  Tc stats (SC only):")
    sc_tc = contrastive[contrastive["is_superconductor"] == 1]["Tc"]
    report.log(f"    Mean: {sc_tc.mean():.2f} K, Std: {sc_tc.std():.2f} K")
    report.log(f"    Min: {sc_tc.min():.2f} K, Max: {sc_tc.max():.2f} K")

    report.log(f"\n  Tc stats (non-SC):")
    non_sc_tc = contrastive[contrastive["is_superconductor"] == 0]["Tc"]
    report.log(f"    All Tc=0: {(non_sc_tc == 0.0).all()}")

    # 8. Spot-check 5 random non-SC entries
    report.log(f"\n  Spot-check 5 random non-SC entries:")
    spot = non_sc_df.sample(min(5, len(non_sc_df)), random_state=RANDOM_SEED)
    for _, row in spot.iterrows():
        report.log(f"    Formula: {row['formula']}")
        report.log(f"    Tc: {row['Tc']}")
        report.log(f"    Category: {row['category']}")
        report.log(f"    is_superconductor: {row['is_superconductor']}")
        report.log(f"    0-norm: {row['0-norm']}, 2-norm: {row['2-norm']:.6f}")
        if "MagpieData mean Number" in row.index:
            report.log(f"    MagpieData mean Number: {row['MagpieData mean Number']:.4f}")
        report.log(f"    transition metal fraction: {row['transition metal fraction']:.4f}")
        report.log(f"    formula_original: {row['formula_original']}")
        report.log(f"    ---")

    # 9. Element distribution report
    from pymatgen.core import Composition as Comp

    sc_comps = []
    for comp_str in sc_orig["composition"]:
        try:
            sc_comps.append(Comp(str(comp_str)))
        except Exception:
            pass

    non_sc_comps = sampled["_composition"].tolist()

    # Load full MP for reference distribution
    mp_full = pd.read_csv(MP_CSV, usecols=["formula"])
    mp_comps = []
    for f in mp_full["formula"]:
        try:
            mp_comps.append(Comp(str(f)))
        except Exception:
            pass

    element_distribution_report(sc_comps, non_sc_comps, mp_comps)

    # ── Save outputs ──
    OUT_CONTRASTIVE.parent.mkdir(parents=True, exist_ok=True)
    contrastive.to_csv(OUT_CONTRASTIVE, index=False)
    report.log(f"\n  Saved contrastive CSV: {OUT_CONTRASTIVE} ({len(contrastive)} rows × {len(contrastive.columns)} cols)")

    non_sc_df.to_csv(OUT_NON_SC, index=False)
    report.log(f"  Saved non-SC CSV: {OUT_NON_SC} ({len(non_sc_df)} rows)")

    elapsed = time.time() - t0
    report.log(f"\n  Total pipeline time: {elapsed:.1f}s")
    report.log("=" * 70)
    report.log("Pipeline complete.")

    # Save report
    report.save(OUT_REPORT)


if __name__ == "__main__":
    main()
