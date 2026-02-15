# Physics-Supervised Z Coordinates (V12.31)

## Overview

The 2048-dim deterministic Z vector is partitioned into **named coordinate blocks** encoding specific physical quantities. Physics meaning is enforced purely via loss functions -- no architectural changes to FullMaterialsVAE. Existing checkpoints load unchanged.

## Block Layout

| Block | Name | Coords | Size | Status | Description |
|-------|------|--------|------|--------|-------------|
| 1 | Ginzburg-Landau | 0-19 | 20 | Self-consistency | GL parameters (kappa, xi, lambda, Hc, Hc1, Hc2, ...) |
| 2 | BCS/Microscopic | 20-49 | 30 | Self-consistency | BCS quantities (vF, kF, EF, lambda_ep, gap ratio, ...) |
| 3 | Eliashberg | 50-69 | 20 | Placeholder | Strong-coupling spectral function params |
| 4 | Unconventional | 70-109 | 40 | Placeholder | Gap symmetry, doping, cuprate/pnictide-specific |
| 5 | Structural | 110-159 | 50 | Placeholder | Crystal structure (space group, lattice params) |
| 6 | Electronic | 160-209 | 50 | Placeholder | Band structure, DOS, transport |
| 7 | Thermodynamic | 210-269 | 60 | Placeholder | Tc variants, critical fields, specific heat |
| 8 | Compositional | 270-339 | 70 | **Active** | Formula-derived features (MW, Z_avg, VEC, ...) |
| 9 | Cobordism | 340-399 | 60 | Self-consistency | Topological defect energies derived from GL |
| 10 | Dimensionless Ratios | 400-449 | 50 | Self-consistency | Cross-block ratio consistency checks |
| 11 | Magpie Encoding | 450-511 | 62 | **Active** | Learnable projection of 145 Magpie features |
| 12 | Discovery | 512-2047 | 1536 | Unsupervised | Free for model to find novel structure |

**Total**: 512 supervised/constrained + 1536 discovery = 2048

## What Works NOW (Zero External Data)

1. **Block 8 (Compositional)**: 15 targets computable from formula via `element_properties.py`
   - n_elements, MW, hydrogen fraction, Z_avg, Z_max
   - electronegativity (avg, diff), atomic radius (avg, ratio)
   - VEC, d-orbital fraction, f-orbital fraction
   - ionization energy avg, melting temp avg, size mismatch delta

2. **Block 11 (Magpie Encoding)**: Learnable linear projection (145 -> 62) trains jointly with encoder

3. **GL Self-Consistency** (Block 1): kappa=lambda/xi, Hc proportional to 1/(lambda*xi), etc.

4. **BCS Self-Consistency** (Block 2): xi proportional to vF/Delta0, gap ratio bounds

5. **Cobordism** (Block 9): E_vortex, E_domain, Type I/II derived from GL coords

6. **Dimensionless Ratios** (Block 10): Tc/Theta_D, xi/l_mfp cross-checks

## Loss Weights (TRAIN_CONFIG)

```python
'use_physics_z': True,                    # Enable/disable
'physics_z_comp_weight': 1.0,             # Compositional supervision
'physics_z_magpie_weight': 0.5,           # Magpie encoding
'physics_z_consistency_weight': 0.1,      # GL/BCS/cobordism/ratios
'physics_z_direct_weight': 0.0,           # External data (placeholder)
'physics_z_warmup_epochs': 20,            # Warmup ramp
'physics_z_data_path': None,              # Future: CSV with physics data
```

## How to Add External Physics Data

When experimental data becomes available (e.g., measured GL parameters, BCS quantities):

1. Create a CSV with columns matching `PhysicsZ` coordinate names (e.g., `KAPPA`, `XI`, `LAMBDA_L`)
2. Set `'physics_z_data_path': 'path/to/physics_data.csv'` in TRAIN_CONFIG
3. Set `'physics_z_direct_weight': 1.0'` (or desired weight)
4. The `DirectSupervisionLoss` will automatically activate for any column present in the CSV
5. Missing values (NaN) are handled via per-sample masks -- only samples with data contribute

No code changes needed. The coordinate names in the CSV header map directly to `PhysicsZ` class attributes.

## Self-Consistency Formulas

### Ginzburg-Landau (Block 1)
- kappa = lambda_L / xi [Tinkham Ch. 4]
- Hc proportional to 1/(lambda_L * xi) [Tinkham Eq. 4.38]
- Hc2 proportional to 1/xi^2 [Abrikosov 1957]
- E_cond proportional to Hc^2 [Tinkham Eq. 2.16]
- Hc1 proportional to (1/lambda^2) * ln(kappa) [Tinkham Eq. 5.22]

### BCS (Block 2)
- xi_0 proportional to v_F / Delta_0 [BCS 1957]
- Gap ratio 2*Delta_0/(k_B*Tc) soft-bounded to [1, 5] (BCS weak coupling: 3.528)

### Cobordism (Block 9)
- E_vortex proportional to (1/lambda)^2 * ln(kappa) [Tinkham Ch. 5]
- E_domain = sigma_ns (N-S surface energy)
- Type I/II = kappa - 1/sqrt(2) [Abrikosov 1957]
- E_defect_min = min(E_vortex, E_domain)

### Dimensionless Ratios (Block 10)
- Tc/Theta_D = z[TC] / z[THETA_D]
- xi/l = z[XI] / z[L_MFP]

## Key Files

| File | Description |
|------|-------------|
| `src/superconductor/models/physics_z.py` | PhysicsZ coordinate map + physical constants |
| `src/superconductor/losses/z_supervision_loss.py` | All Z supervision losses |
| `src/superconductor/data/compositional_targets.py` | Block 8 target computation from formula |
| `scripts/train_v12_clean.py` | Training integration |

## Checkpoint Compatibility

No model architecture changes. The encoder still outputs 2048-dim Z from `fc_mean`. The `PhysicsZLoss` module has its own parameters (MagpieEncodingLoss projection: ~9K params) which are included in the encoder optimizer. These parameters are randomly initialized on first use and do not affect existing checkpoint loading.
