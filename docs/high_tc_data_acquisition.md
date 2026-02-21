# High-Tc Superconductor Data Acquisition — Claude Code Instructions

## Objective

Download, clean, and merge publicly available superconductor datasets to augment the 100–200K and >200K Tc bins for the fraction-aware VAE. The existing model already has a high-pressure (HP) classifier, so hydride data under extreme pressure is fine to include — just ensure a `pressure_gpa` column is populated.

## Priority Order

Work through these sources in order. After each, report:
- Total rows acquired
- Rows with Tc ≥ 100K
- Rows with Tc ≥ 200K
- Column schema

---

## Source 1: MDR SuperCon Datasheet (NIMS) — PRIMARY

The official re-release of the SuperCon database. Contains the full oxide/metallic superconductor catalog with recommended Tc values.

```bash
# The datasheet lives at NIMS MDR. Download the primary.tsv file which has
# recommended Tc for all entries (designed for ML use).
# Landing page: https://mdr.nims.go.jp/collections/5712mb227
# Readme DOI: https://doi.org/10.48505/nims.3740
# Data DOI: https://doi.org/10.48505/nims.3837

# Try direct download first. If the DOI redirects to a landing page,
# you may need to navigate the NIMS MDR download links.
# The key file is: primary.tsv (tab-separated, has recommended Tc per entry)
# The full datasheets are in: OxideMetallic/ and Organic/ subdirectories

# If direct download fails, fall back to the Stanev et al. cleaned subset
# which is widely mirrored:
pip install kaggle --break-system-packages
# Download from: https://www.kaggle.com/datasets/chaozhuang/supercon-dataset
# OR use the UCI ML Repository version (Hamidieh, 21263 samples):
# https://archive.ics.uci.edu/dataset/464/superconductivty+data
```

### Processing
```python
import pandas as pd

# Load the TSV
df = pd.read_csv('primary.tsv', sep='\t')

# Key columns to extract/rename:
# - Chemical formula (exact string, preserve fractions like Ba2Cu3O6.97)
# - Tc (K) — use the "recommended" Tc column
# - Pressure (GPa) — if available, else fill 0 (ambient)
# - Material class — cuprate, ferrite, heavy_fermion, oxide, etc.

# Filter for high-Tc entries
high_tc = df[df['Tc'] >= 100]
print(f"Entries with Tc >= 100K: {len(high_tc)}")
print(f"Entries with Tc >= 200K: {len(df[df['Tc'] >= 200])}")

# IMPORTANT: The full SuperCon has ~33,000 entries total.
# The commonly used Hamidieh/Stanev subsets only have ~16,000-21,000.
# The difference contains many cuprate doping variants in the 100-150K range
# that were dropped during deduplication. We WANT those variants.
```

---

## Source 2: SuperCon2 (NLP-extracted, ~40K records)

Automatically extracted from 37,700 papers. Contains entries not in the original SuperCon, especially recent discoveries.

```bash
# Clone the SuperCon2 repository
git clone https://github.com/lfoppiano/supercon.git supercon2_repo

# The main data file is:
# supercon2_repo/data/supercon2_v22.12.03.csv
# Also check: supercon2_repo/data/supercon2_1203_papers.csv (paper metadata)

# Alternative: Download from NIMS MDR directly
# https://mdr.nims.go.jp/concern/datasets/4q77fv540
```

### Processing
```python
df_sc2 = pd.read_csv('supercon2_repo/data/supercon2_v22.12.03.csv')

# Actual column names (from repo README):
# id, rawMaterial, materialId, name, formula, doping, shape,
# materialClass, fabrication, substrate, variables,
# criticalTemperature, criticalTemperatureMeasurementMethod,
# appliedPressure, section, subsection, hash,
# title, doi, authors, publisher, journal, year
#
# Key columns to use:
# - formula: chemical formula
# - criticalTemperature: Tc (K) — NOTE: column name, not "tc"
# - appliedPressure: pressure (GPa) — CRITICAL for HP classifier
# - criticalTemperatureMeasurementMethod: resistivity, susceptibility, etc.
# - materialClass: cuprate, iron-based, etc.

# WARNING: This dataset was NLP-extracted and may contain:
# 1. Duplicate entries from different papers reporting the same material
# 2. Misattributed Tc values (e.g., onset vs. midpoint vs. zero-resistance)
# 3. Entries where pressure is missing but material is a high-pressure hydride

# Deduplicate by formula + Tc (within 2K tolerance):
# Group by normalized formula, keep median Tc per unique material
# For formulas with large Tc variance (>10K), flag for manual review

high_tc_sc2 = df_sc2[df_sc2['criticalTemperature'] >= 100]
print(f"SuperCon2 entries with Tc >= 100K: {len(high_tc_sc2)}")
```

---

## Source 3: 3DSC Dataset (with crystal structures)

Superconductors matched to 3D crystal structures. Useful because high-Tc cuprates matched to more structures per entry, naturally oversampling the target range.

```bash
# Available on figshare:
# https://doi.org/10.6084/m9.figshare.c.6914407.v1
# OR from GitHub:
git clone https://github.com/aimat-lab/3DSC.git

# The key file is: 3DSC_MP.csv (Materials Project matched)
# Extract if compressed:
# tar -xzvf 3DSC_MP.tar.gz
```

### Processing
```python
df_3dsc = pd.read_csv('3DSC/superconductors_3D/data/final/MP/3DSC_MP.csv')

# Key columns (verify exact names by inspecting the CSV header):
# - formula_sc: chemical formula (original SuperCon entry) — may also be named 'formula' or 'name'
# - tc: critical temperature (K) — confirmed from repo README
# - sc_class: Cuprate, Ferrite, Heavy_fermion, Chevrel, Oxide, Carbon, Other
# - weight: inverse of number of structures per SuperCon entry
#           (use this for deduplication — entries with weight < 1 have
#            multiple crystal structure matches)

# For your VAE (formula-based, not structure-based), deduplicate:
# Use weight=1 entries OR take unique formula_sc + tc combinations
df_3dsc_unique = df_3dsc.drop_duplicates(subset=['formula_sc', 'tc'])

# Cuprate-specific subset (where most 100K+ entries live):
cuprates = df_3dsc_unique[df_3dsc_unique['sc_class'] == 'Cuprate']
print(f"Unique cuprate entries: {len(cuprates)}")
print(f"Cuprates with Tc >= 100K: {len(cuprates[cuprates['tc'] >= 100])}")
```

---

## Source 4: SODNet Dataset (NeurIPS 2024)

Curated dataset with verified Tc values and crystal structures.

```bash
git clone https://github.com/pincher-chen/SODNet.git

# Data lives in: SODNet/datasets/SuperCon/
# Key files:
# - reference.csv (Tc with literature references)
# - The dataset includes hydrogen-enriched superconductors from literature
```

### Processing
```python
# Load and check for high-Tc entries, especially hydrides
# This dataset explicitly includes H-enriched superconductors
# collected from literature beyond the standard SuperCon entries
# NOTE: Verify exact directory structure and column names after cloning.
# The path below is approximate — check the repo README.
df_sod = pd.read_csv('SODNet/datasets/SuperCon/reference.csv')  # adjust path as needed

# Tag any hydride entries (formulas containing H with high Tc)
# These likely need pressure annotation — SODNet may not include pressure
hydrides = df_sod[df_sod['formula'].str.contains('H') & (df_sod['tc'] >= 100)]
print(f"High-Tc hydrides in SODNet: {len(hydrides)}")
```

---

## Source 5: NIST High-Pressure Hydride DFT Dataset

Over 900 hydride materials with DFT-computed Tc at 0–500 GPa. This is the main source for >200K augmentation.

```bash
# From the paper: "Data-driven Design of High Pressure Hydride Superconductors
# using DFT and Deep Learning" (Materials Futures, 2024)
# Authors: Choudhary, Garrity et al. (NIST)
#
# Data available through JARVIS:
pip install jarvis-tools --break-system-packages

# OR check supplementary materials / JARVIS database directly:
# https://jarvis.nist.gov/
# Look for the hydride superconductor dataset

# The paper's Table 1 and Table S1 contain all dynamically stable
# structures with Tc > 39K (MgB2 threshold), including:
# - LaH10: 160-187K at 200-250 GPa (DFT; experimental is ~250K)
# - H3S: 154K at 200 GPa (DFT; experimental is 203K)
# - CaH6, YH9, ScH3, MgH2, etc.
```

### Processing
```python
# IMPORTANT: These are DFT-PREDICTED Tc values, not experimental.
# DFT systematically underestimates Tc for the highest-Tc hydrides
# (by up to ~40-100K for strongly coupled systems like LaH10).
# 
# Options:
# 1. Use DFT values as-is (conservative, internally consistent)
# 2. Use experimental values where available, DFT for rest
# 3. Add a column `tc_source` = "experimental" | "dft_mcmillan_allen_dynes"
#
# Recommended: Option 3. Your VAE should know the data provenance.

# For each entry, ensure you capture:
# - formula
# - tc (K)
# - pressure_gpa (CRITICAL — these are all high-pressure)
# - tc_source: "dft" or "experimental"
# - is_dynamically_stable: bool
```

---

## Source 6: HTSC-2025 Benchmark (June 2025)

Brand new benchmark of ambient-pressure high-Tc BCS superconductors predicted 2023-2025.

```bash
git clone https://github.com/xqh19970407/HTSC-2025.git

# Contains CIF files and predicted Tc for multiple structural families:
# - X2YH6 system
# - Perovskite MXH3 system  
# - M3XH8 system
# - BCN-doped cage structures (LaH10-derived)
# - 2D superconductors
```

### Processing
```python
# These are AMBIENT PRESSURE predictions — extremely valuable
# because they don't require the HP classifier flag.
# But they are ALL theoretical predictions, not experimental.
# Tag accordingly: tc_source = "dft_2025_benchmark"

# Extract formula + Tc from the dataset structure
# (check the repo README for exact file format)
```

---

## Source 7: Experimentally Confirmed High-Tc Hydrides (Manual Curation)

Manually curated from literature. These are the highest-confidence data points
for augmenting both the >200K and 100–200K bins. **All require extreme pressure.**

```python
import pandas as pd

# Manually curated from literature — ALL experimental
# NOTE on alloy formulas: (La,Ce)H9 and (La,Y)H10 are substitutional
# alloys. Exact stoichiometries vary by sample; formulas below are
# representative. Your tokenizer must handle fractional subscripts.

experimental_hydrides_high_tc = [
    # --- >200K bin ---
    {"formula": "H3S",            "tc": 203, "pressure_gpa": 155, "tc_source": "experimental", "ref": "Drozdov2015_Nature"},
    {"formula": "LaH10",          "tc": 250, "pressure_gpa": 170, "tc_source": "experimental", "ref": "Drozdov2019_Nature"},
    {"formula": "LaH10",          "tc": 260, "pressure_gpa": 190, "tc_source": "experimental", "ref": "Somayazulu2019_PRL"},
    {"formula": "YH9",            "tc": 243, "pressure_gpa": 201, "tc_source": "experimental", "ref": "Kong2021_NatComm"},
    {"formula": "YH6",            "tc": 224, "pressure_gpa": 166, "tc_source": "experimental", "ref": "Troyan2021_AdvMater"},
    {"formula": "CaH6",           "tc": 215, "pressure_gpa": 172, "tc_source": "experimental", "ref": "Ma2022_PRL"},
    {"formula": "La0.5Y0.5H10",   "tc": 253, "pressure_gpa": 183, "tc_source": "experimental", "ref": "Semenok2021_MaterToday"},

    # --- 100–200K bin (high-pressure hydrides) ---
    {"formula": "La0.5Ce0.5H9",   "tc": 178, "pressure_gpa": 97,  "tc_source": "experimental", "ref": "Bi2022_NatComm"},
    # ^ Bi et al. 2022 NatComm 13:5952. Equal-atomic (La,Ce)H9 alloy.
    #   Tc ranges 148-178K over 97-172 GPa. Using max Tc at lowest P here.
    #   NOTE: La0.75Ce0.25H10 seen in some papers is a COMPUTATIONAL study,
    #   not experimental. Do not confuse with this entry.
    {"formula": "ThH10",          "tc": 161, "pressure_gpa": 175, "tc_source": "experimental", "ref": "Semenok2020_MaterToday"},
    {"formula": "ThH9",           "tc": 146, "pressure_gpa": 170, "tc_source": "experimental", "ref": "Semenok2020_MaterToday"},
    {"formula": "CeH9",           "tc": 117, "pressure_gpa": 95,  "tc_source": "experimental", "ref": "Chen2021_PRL"},
    {"formula": "LaBeH8",         "tc": 110, "pressure_gpa": 80,  "tc_source": "experimental", "ref": "Song2023_PRL"},
]

df_exp_hydrides = pd.DataFrame(experimental_hydrides_high_tc)
```

---

## Merging Pipeline

After downloading all sources, merge into a single unified dataset:

```python
import pandas as pd
import re

def normalize_formula(formula):
    """
    Normalize chemical formula for deduplication.
    Strip whitespace and remove internal spaces.
    Keep fractional stoichiometries (critical for your VAE tokenizer).
    
    NOTE: This does NOT sort elements alphabetically — that would break
    formulas where element order encodes structure (e.g., YBa2Cu3O7).
    If your existing dataset uses a canonical ordering, apply that here.
    """
    if pd.isna(formula):
        return None
    formula = str(formula).strip()
    # Remove any spaces within formula
    formula = re.sub(r'\s+', '', formula)
    return formula

def merge_datasets():
    """
    Merge all downloaded datasets into unified format.
    """
    frames = []
    
    # ---- Load each source (adjust paths to where you downloaded) ----
    
    # Source 1: MDR SuperCon
    # df1 = pd.read_csv('primary.tsv', sep='\t')
    # df1 = df1.rename(columns={...})  # map to unified schema
    # df1['source'] = 'mdr_supercon'
    # df1['tc_source'] = 'experimental'
    # df1['pressure_gpa'] = df1.get('pressure_gpa', 0).fillna(0)
    # frames.append(df1[unified_cols])
    
    # Source 2: SuperCon2
    # df2 = pd.read_csv('supercon2_v22.12.03.csv')
    # ... similar processing
    
    # ... (repeat for each source)
    
    # ---- Unified schema ----
    unified_cols = [
        'formula',          # str: chemical formula with fractions preserved
        'tc',               # float: critical temperature (K)
        'pressure_gpa',     # float: applied pressure (0 = ambient)
        'tc_source',        # str: experimental | dft | dft_2025_benchmark
        'material_class',   # str: cuprate | iron_based | hydride | heavy_fermion | other
        'source_db',        # str: which database this came from
        'is_high_pressure',  # bool: pressure_gpa > 1.0
    ]
    
    # ---- Concatenate ----
    merged = pd.concat(frames, ignore_index=True)
    
    # ---- Normalize formulas ----
    merged['formula_norm'] = merged['formula'].apply(normalize_formula)
    
    # ---- Deduplicate ----
    # Strategy: for duplicate (formula_norm, is_high_pressure) pairs,
    # prefer experimental over DFT, then take median Tc
    merged = merged.sort_values('tc_source', ascending=True)  # 'dft' before 'experimental'
    
    # Group and aggregate
    def agg_group(group):
        # Prefer experimental Tc if available
        exp = group[group['tc_source'] == 'experimental']
        if len(exp) > 0:
            best = exp.iloc[0].copy()
            best['tc'] = exp['tc'].median()
        else:
            best = group.iloc[-1].copy()
            best['tc'] = group['tc'].median()
        best['n_sources'] = len(group['source_db'].unique())
        return best
    
    deduped = merged.groupby(['formula_norm', 'is_high_pressure']).apply(agg_group)
    deduped = deduped.reset_index(drop=True)
    
    # ---- Report ----
    print(f"\n{'='*60}")
    print(f"MERGED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total unique entries: {len(deduped)}")
    print(f"\nTc distribution:")
    bins = [(0, 10), (10, 50), (50, 100), (100, 150), (150, 200), (200, float('inf'))]
    for lo, hi in bins:
        label = f">{lo}K" if hi == float('inf') else f"{lo}-{hi}K"
        count = len(deduped[(deduped['tc'] >= lo) & (deduped['tc'] < hi)])
        pct = count / len(deduped) * 100
        print(f"  {label:>10s}: {count:>6d} ({pct:.1f}%)")
    
    print(f"\nBy tc_source:")
    print(deduped['tc_source'].value_counts().to_string())
    
    print(f"\nHigh-pressure entries: {deduped['is_high_pressure'].sum()}")
    
    # ---- Save ----
    deduped.to_csv('merged_superconductor_dataset.csv', index=False)
    print(f"\nSaved to merged_superconductor_dataset.csv")
    
    return deduped

# Run the merge
df = merge_datasets()
```

---

## Validation Checks

After merging, run these sanity checks:

```python
import re

# 1. No Tc < 0
assert (df['tc'] >= 0).all(), "Negative Tc values found"

# 2. All high-pressure hydrides flagged
# NOTE: Simple str.contains('H') catches Hg-based cuprates (HgBa2Ca2Cu3O8)
# and many non-hydride compounds. Use a smarter heuristic:
def is_likely_hydride(formula):
    """Detect hydrides: formula contains H as an element (not Hg, Hf, Ho, etc.)
    H as an element is followed by a digit, uppercase, or end-of-string —
    NOT a lowercase letter (which would make it Hg, Hf, Ho, Hs, He, Ha)."""
    if pd.isna(formula):
        return False
    return bool(re.search(r'H(?![a-z])', str(formula)))

hydride_mask = df['formula'].apply(is_likely_hydride) & (df['tc'] >= 100)
hp_mask = df['is_high_pressure']
unflagged = df[hydride_mask & ~hp_mask & (df['tc'] >= 200)]
if len(unflagged) > 0:
    print(f"WARNING: {len(unflagged)} high-Tc hydrides without pressure flag:")
    print(unflagged[['formula', 'tc', 'pressure_gpa']].to_string())

# 3. Cuprate Tc sanity: should cluster 20-135K at ambient pressure
cuprates = df[(df['material_class'] == 'cuprate') & (~df['is_high_pressure'])]
assert cuprates['tc'].max() < 140, f"Suspiciously high cuprate Tc: {cuprates['tc'].max()}"
assert cuprates['tc'].min() > 0, "Cuprate with Tc = 0 found"

# 4. Check formula parsability with your tokenizer
# Run a quick test through your fraction-aware tokenizer
# to ensure all formulas can be tokenized without errors
# (especially the hydride formulas like La0.5Y0.5H10)

# 5. Distribution comparison vs. original dataset
# Load your original training data and compare histograms
# to quantify the augmentation effect per bin
```

---

## Expected Yield

| Tc Bin    | Original Count | Expected After Merge | Main New Sources |
|-----------|---------------|---------------------|-----------------|
| 100-150K  | 977           | ~1,200-1,500        | SuperCon2 cuprate variants, 3DSC, CeH9(117K), ThH9(146K), LaBeH8(110K) |
| 150-200K  | ~247          | ~350-500            | SuperCon2, MDR full (Hg/Tl cuprate doping variants), (La,Ce)H9(178K), ThH10(161K), HTSC-2025 |
| >200K     | 6             | ~20-30 (experimental) + ~100-200 (DFT) | NIST hydrides, HTSC-2025, manual curation |

The >200K bin will remain small for experimental data (it IS a frontier of physics), but adding DFT-predicted hydrides gives the model exposure to that compositional space. Your HP classifier will handle the pressure confound.

---

## Notes

- **Tokenizer compatibility**: Verify that hydride formulas (e.g., `Li2NaH17`, `LaH10`, `H3S`) tokenize correctly in your fraction-aware pointer-generator decoder. These have very different element distributions than cuprates.
- **Loss weighting**: After augmentation, recompute your Tc-bin weights. The 100-150K bin will be less sparse, so you may be able to reduce the bin multiplier from 3.0x to ~2.0x while increasing the >200K multiplier.
- **Tc-binned sampler**: With more 100K+ data, the sampler target of 10% for 100K+ becomes more feasible without extreme oversampling.
