# Processed Superconductor Data

## Files

### `supercon_fractions.csv` (Original)
- **Source**: SuperCon database
- **Rows**: 16,521 superconductors
- **Columns**: 151
- **Description**: Original processed SuperCon dataset with fraction-notation formulas and pre-computed Magpie features.

### `supercon_fractions_combined.csv` (Combined — SC-only training file)
- **Source**: SuperCon + NEMAD datasets merged
- **Rows**: 23,451 superconductors (16,521 existing + 6,930 NEMAD)
- **Columns**: 151 (identical schema to original)
- **Created by**: `scripts/ingest_nemad.py`
- **Description**: Merged dataset combining the original SuperCon data with new entries from the NEMAD superconductor materials database. NEMAD entries are deduplicated against existing SuperCon compositions and filtered to exclude generative holdout samples.

### `supercon_fractions_contrastive.csv` (Contrastive — SC + non-SC)
- **Source**: SC combined + non-SC materials (NEMAD magnetic/thermoelectric/anisotropy + Materials Project)
- **Rows**: 46,645 (23,451 SC + 23,194 non-SC)
- **Columns**: 152 (151 original + `is_superconductor`)
- **Created by**: `scripts/ingest_non_sc.py`
- **Description**: Contrastive-ready dataset with both superconductors and non-superconductors. The `is_superconductor` column (0 or 1) distinguishes them. SC rows are identical to `supercon_fractions_combined.csv` with the added column. Non-SC materials have Tc=0.0 and categories prefixed with "Non-SC:".

### `non_sc_fractions.csv` (Non-SC only)
- **Source**: NEMAD non-SC materials + Materials Project
- **Rows**: 23,194
- **Columns**: 152 (same schema as contrastive)
- **Description**: Non-superconductor entries only, for inspection. (206 compositions dropped due to failed Magpie featurization)

### `nemad_fractions.csv` (NEMAD SC-only)
- **Source**: NEMAD dataset (experimental entries only)
- **Rows**: 6,930 unique superconductors (not in original SuperCon)
- **Columns**: 151 (identical schema)
- **Description**: NEMAD-only entries for inspection. Subset of NEMAD that is new relative to the existing SuperCon data.

## Column Schema

### Original Schema (151 columns) — `supercon_fractions.csv`, `supercon_fractions_combined.csv`

| Column(s) | Count | Description |
|-----------|-------|-------------|
| `formula` | 1 | Fraction notation (e.g., `Ag(1/500)Al(499/500)`) |
| `Tc` | 1 | Critical temperature in Kelvin |
| `composition` | 1 | Pymatgen formula string |
| `category` | 1 | Superconductor family classification |
| `0-norm` through `10-norm` | 6 | Lp-norm features of composition fractions |
| `MagpieData *` | 132 | Magpie elemental property statistics |
| `avg [s/p/d/f] valence electrons` | 4 | Valence orbital averages |
| `compound possible` | 1 | Whether a valid ionic model exists (bool) |
| `max ionic char`, `avg ionic char` | 2 | Ionic character features |
| `transition metal fraction` | 1 | Fraction of transition metal elements |
| `formula_original` | 1 | Decimal notation formula |

### Contrastive Schema (152 columns) — `supercon_fractions_contrastive.csv`, `non_sc_fractions.csv`

Same as original with `is_superconductor` inserted after `category`:

| Column(s) | Count | Description |
|-----------|-------|-------------|
| `formula` | 1 | Fraction notation |
| `Tc` | 1 | Critical temperature (K). 0.0 for non-SC |
| `composition` | 1 | Pymatgen formula string |
| `category` | 1 | Classification (SC families or "Non-SC: *") |
| **`is_superconductor`** | **1** | **1 = superconductor, 0 = non-superconductor** |
| `0-norm` through `10-norm` | 6 | Lp-norm features |
| `MagpieData *` | 132 | Magpie elemental property statistics |
| `avg [s/p/d/f] valence electrons` | 4 | Valence orbital averages |
| `compound possible` | 1 | Ionic model exists (bool) |
| `max ionic char`, `avg ionic char` | 2 | Ionic character features |
| `transition metal fraction` | 1 | Transition metal fraction |
| `formula_original` | 1 | Decimal notation formula |

## Categories

### SC Categories (from combined dataset)

| Category | Count |
|----------|-------|
| Other | 9,952 |
| Cuprates | 8,179 |
| Bismuthates | 2,360 |
| Iron-based | 2,353 |
| Borocarbides | 416 |
| Organic Superconductors | 75 |
| Elemental Superconductors | 71 |
| Hydrogen-rich Superconductors | 45 |

### Non-SC Categories (from contrastive dataset)

| Category | Count | Source |
|----------|-------|--------|
| Non-SC: Materials Project | 14,040 | Materials Project (DFT, prefer stable) |
| Non-SC: Magnetic | 4,009 | NEMAD magnetic_materials.csv |
| Non-SC: Thermoelectric | 3,287 | NEMAD thermoelectric_materials.csv |
| Non-SC: Anisotropy | 1,858 | NEMAD magnetic_anisotropy_materials.csv |

## Non-SC Ingestion Pipeline

The non-SC data was ingested using `scripts/ingest_non_sc.py` with the following pipeline:

1. **Load Sources**: NEMAD magnetic, thermoelectric, anisotropy + Materials Project
2. **Clean & Parse**: Reuse `clean_nemad_formula()` for NEMAD; MP formulas are already clean
3. **Deduplication**: By canonical formula (priority: NEMAD magnetic > thermo > aniso > MP)
4. **SC/Holdout Removal**: Remove any compositions matching existing SC data or holdout samples
5. **Representative Sampling**: 23,400 sampled (60% MP for chemical diversity, 40% NEMAD for experimental grounding)
6. **Feature Computation**: Identical to SC pipeline (Lp-norms, Magpie, ValenceOrbital, IonProperty with 30s/composition timeout, TMetalFraction). 206 rows dropped due to failed Magpie featurization → 23,194 final non-SC rows
7. **Schema Creation**: Add `is_superconductor` column to both SC and non-SC data
8. **Merge & Verify**: Concatenate, verify 152 columns, validate no SC/holdout overlap

Detailed run report: `scratch/non_sc_ingest_report.txt`

## NEMAD SC Ingestion Pipeline

The NEMAD SC data was ingested using `scripts/ingest_nemad.py` with the following pipeline:

1. **Load & Filter**: 19,058 raw NEMAD rows filtered to 15,339 experimental-only
2. **Extract Tc**: Primary from `Median_Tc_By_Composition_K`, fallback from text parsing
3. **Formula Normalization**: Unicode cleanup, delta/variable removal, dopant stripping
4. **Pymatgen Parsing**: Composition validation and canonicalization
5. **Category Assignment**: Mapped from NEMAD `Superconductor_Type` to existing categories
6. **Feature Computation**: 145 features via matminer (Magpie, ValenceOrbital, IonProperty, TMetalFraction) + 6 Lp-norms
7. **Deduplication**: Removed entries matching existing SuperCon compositions
8. **Holdout Filter**: Excluded 45 generative holdout samples
9. **Merge**: Concatenated with existing SuperCon data preserving row order

Detailed run report: `scratch/nemad_ingest_report.txt`

## Google Colab Training

A ready-to-use Colab notebook is available at `notebooks/train_colab.ipynb`. Upload the entire repo to Google Drive, open the notebook in Colab, and run the cells in order. Checkpoints are saved to `outputs/` on Drive and persist across sessions. See the notebook for configuration options.

## Holdout Filtering

The 45 generative holdout samples (`data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json`) are present in both CSV files but are **excluded at training time** by the training scripts (`scripts/train.py`, `scripts/train_v12_clean.py`). Filtering is done by formula matching, not positional index, so it is robust to row reordering.

- **Combined CSV rows**: 23,451
- **After holdout exclusion**: 23,406 training samples

## Provenance

- **Original SuperCon data**: Pre-existing processed dataset (seed=42 split conventions)
- **NEMAD SC source**: `DSC288 Data/DSC288 Data/NEMAD/NEMAD_superconductor_materials_unique.csv`
- **NEMAD non-SC sources**: `DSC288 Data/DSC288 Data/NEMAD/magnetic_materials.csv`, `thermoelectric_materials.csv`, `magnetic_anisotropy_materials.csv`
- **Materials Project source**: `DSC288 Data/DSC288 Data/materials_project.csv`
- **Holdout set**: `data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json` (45 samples, NEVER train on these)
- **SC pipeline date**: January 2026
- **Non-SC pipeline date**: January 2026
