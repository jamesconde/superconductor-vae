#!/usr/bin/env python3
"""
Analyze PhysicsZ block structure in the latent space of the trained FullMaterialsVAE.

Loads the best checkpoint, runs 100 training samples through the encoder,
and examines each of the 12 PhysicsZ blocks for structure vs noise.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/home/james/superconductor-vae")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.physics_z import PhysicsZ

# ============================================================================
# Configuration
# ============================================================================
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "cache"
CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoint_best.pt"
N_SAMPLES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Load metadata
# ============================================================================
with open(CACHE_DIR / "cache_meta.json", "r") as f:
    meta = json.load(f)

n_magpie_cols = meta["n_magpie_cols"]
tc_mean = meta["tc_mean"]
tc_std = meta["tc_std"]
tc_log_transform = meta.get("tc_log_transform", False)

print("=" * 80)
print("PhysicsZ Latent Space Analysis")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Magpie dims: {n_magpie_cols}")
print(f"Tc mean: {tc_mean:.4f}, std: {tc_std:.4f}, log_transform: {tc_log_transform}")
print(f"N samples: {N_SAMPLES}")
print()

# ============================================================================
# Load data from cache
# ============================================================================
print("Loading cached tensors...")
element_indices = torch.load(CACHE_DIR / "element_indices.pt", weights_only=True)
element_fractions = torch.load(CACHE_DIR / "element_fractions.pt", weights_only=True)
element_mask = torch.load(CACHE_DIR / "element_mask.pt", weights_only=True)
magpie_tensor = torch.load(CACHE_DIR / "magpie_tensor.pt", weights_only=True)
tc_tensor = torch.load(CACHE_DIR / "tc_tensor.pt", weights_only=True)

# Get train indices and take first N_SAMPLES
train_indices = meta["train_indices"][:N_SAMPLES]

# Subset
elem_idx = element_indices[train_indices].to(DEVICE)
elem_frac = element_fractions[train_indices].to(DEVICE)
elem_mask = element_mask[train_indices].to(DEVICE)
magpie = magpie_tensor[train_indices].to(DEVICE)
tc = tc_tensor[train_indices].to(DEVICE)

print(f"Loaded {N_SAMPLES} samples")
print(f"  element_indices: {elem_idx.shape}")
print(f"  magpie: {magpie.shape}")
print(f"  tc: {tc.shape}, range [{tc.min():.4f}, {tc.max():.4f}]")

# Denormalize Tc for correlation analysis
if tc_log_transform:
    # tc_normalized = (log1p(tc_raw) - mean) / std
    # tc_raw = expm1(tc_normalized * std + mean)
    tc_raw = torch.expm1(tc.cpu() * tc_std + tc_mean)
else:
    tc_raw = tc.cpu() * tc_std + tc_mean
print(f"  tc_raw (denormalized): range [{tc_raw.min():.2f}, {tc_raw.max():.2f}] K")
print()

# ============================================================================
# Create and load model
# ============================================================================
print("Creating model...")
encoder = FullMaterialsVAE(
    n_elements=118,
    element_embed_dim=128,
    n_attention_heads=8,
    magpie_dim=n_magpie_cols,
    fusion_dim=256,
    encoder_hidden=[512, 256],
    latent_dim=2048,
    decoder_hidden=[256, 512],
    dropout=0.1,
).to(DEVICE)

print("Loading checkpoint...")
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
enc_state = checkpoint["encoder_state_dict"]

# Strip _orig_mod. prefix if present (compiled checkpoint)
if any(k.startswith("_orig_mod.") for k in enc_state.keys()):
    print("  Stripping '_orig_mod.' prefixes from compiled checkpoint")
    enc_state = {k.replace("_orig_mod.", ""): v for k, v in enc_state.items()}

# Handle shape mismatches (partial preserve)
model_state = encoder.state_dict()
for key in list(enc_state.keys()):
    if key in model_state and enc_state[key].shape != model_state[key].shape:
        old_shape = enc_state[key].shape
        new_shape = model_state[key].shape
        if len(old_shape) == 2 and len(new_shape) == 2:
            min_r, min_c = min(old_shape[0], new_shape[0]), min(old_shape[1], new_shape[1])
            new_w = torch.zeros(new_shape, dtype=enc_state[key].dtype)
            new_w[:min_r, :min_c] = enc_state[key][:min_r, :min_c]
            enc_state[key] = new_w
            print(f"  Partial preserve {key}: {old_shape} -> {new_shape}")
        elif len(old_shape) == 1 and len(new_shape) == 1:
            min_len = min(old_shape[0], new_shape[0])
            new_b = torch.zeros(new_shape, dtype=enc_state[key].dtype)
            new_b[:min_len] = enc_state[key][:min_len]
            enc_state[key] = new_b
            print(f"  Partial preserve {key}: {old_shape} -> {new_shape}")

# Remove unexpected keys
unexpected = set(enc_state.keys()) - set(model_state.keys())
for k in unexpected:
    del enc_state[k]
    
missing = set(model_state.keys()) - set(enc_state.keys())
if missing:
    print(f"  Missing keys (using defaults): {missing}")

encoder.load_state_dict(enc_state, strict=False)
encoder.eval()

epoch = checkpoint.get("epoch", "unknown")
print(f"  Loaded checkpoint from epoch {epoch}")
print()

# ============================================================================
# Run encoder
# ============================================================================
print("Running encoder forward pass...")
with torch.no_grad():
    enc_out = encoder.encode(elem_idx, elem_frac, elem_mask, magpie, tc)
    z = enc_out["z"]  # [N_SAMPLES, 2048]

print(f"Z shape: {z.shape}")
print(f"Z global: mean={z.mean():.6f}, std={z.std():.6f}, min={z.min():.4f}, max={z.max():.4f}")
print()

z_cpu = z.cpu().numpy()
tc_raw_np = tc_raw.numpy().flatten()

# ============================================================================
# SECTION 1: All 12 PhysicsZ blocks overview
# ============================================================================
print("=" * 80)
print("SECTION 1: All 12 PhysicsZ Blocks Overview")
print("=" * 80)

blocks = PhysicsZ.get_block_ranges()
block_order = [
    "gl", "bcs", "eliashberg", "unconventional", "structural",
    "electronic", "thermodynamic", "compositional", "cobordism",
    "ratios", "magpie", "discovery"
]

for i, name in enumerate(block_order, 1):
    start, end = blocks[name]
    block_data = z_cpu[:, start:end]  # [N_SAMPLES, block_size]
    
    # Per-coordinate stats (averaged across samples)
    coord_means = block_data.mean(axis=0)  # mean of each coordinate across samples
    coord_stds = block_data.std(axis=0)    # std of each coordinate across samples
    
    # Cross-sample variance (do different materials get different values?)
    sample_variance = block_data.var(axis=0).mean()  # avg variance across coords
    
    # Within-sample variance (are coords within a single sample diverse?)
    within_sample_var = block_data.var(axis=1).mean()  # avg coord variance within each sample
    
    # Overall stats
    overall_mean = block_data.mean()
    overall_std = block_data.std()
    overall_min = block_data.min()
    overall_max = block_data.max()
    
    # Near-constant coords (std < 0.01 across samples)
    n_near_const = (coord_stds < 0.01).sum()
    n_coords = end - start
    
    # Structured vs noise heuristic:
    # If variance across coordinates within a sample is much higher than
    # variance of each coordinate across samples, it's more "structured"
    # (different coords encode different things)
    structure_ratio = within_sample_var / (sample_variance + 1e-10)
    
    print(f"\nBlock {i:2d}: {name.upper():16s} [coords {start:4d}:{end:4d}] ({n_coords} dims)")
    print(f"  Overall:  mean={overall_mean:+.6f}  std={overall_std:.6f}  min={overall_min:+.4f}  max={overall_max:+.4f}")
    print(f"  Coord means range: [{coord_means.min():+.6f}, {coord_means.max():+.6f}]")
    print(f"  Coord stds range:  [{coord_stds.min():.6f}, {coord_stds.max():.6f}]")
    print(f"  Cross-sample variance (avg): {sample_variance:.6f}")
    print(f"  Within-sample variance (avg): {within_sample_var:.6f}")
    print(f"  Structure ratio (within/cross): {structure_ratio:.2f}")
    print(f"  Near-constant coords (std<0.01): {n_near_const}/{n_coords}")

# ============================================================================
# SECTION 2: Unsupervised blocks deep dive
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Unsupervised Blocks Deep Dive")
print("=" * 80)

unsupervised_blocks = {
    "eliashberg": (PhysicsZ.ELIASHBERG_START, PhysicsZ.ELIASHBERG_END),
    "unconventional": (PhysicsZ.UNCONVENTIONAL_START, PhysicsZ.UNCONVENTIONAL_END),
    "structural": (PhysicsZ.STRUCTURAL_START, PhysicsZ.STRUCTURAL_END),
    "electronic": (PhysicsZ.ELECTRONIC_START, PhysicsZ.ELECTRONIC_END),
    "thermodynamic": (PhysicsZ.THERMO_START, PhysicsZ.THERMO_END),
    "discovery": (PhysicsZ.DISCOVERY_START, PhysicsZ.DISCOVERY_END),
}

for name, (start, end) in unsupervised_blocks.items():
    block_data = z_cpu[:, start:end]
    n_coords = end - start
    
    print(f"\n{'='*60}")
    print(f"Block: {name.upper()} [coords {start}:{end}] ({n_coords} dims)")
    print(f"{'='*60}")
    
    # 2a: Per-coordinate mean/std for sample coordinates
    print(f"\n  --- Per-coordinate stats (first 10 coords) ---")
    sample_coords = min(10, n_coords)
    for ci in range(sample_coords):
        coord_vals = block_data[:, ci]
        print(f"    coord {start+ci:4d}: mean={coord_vals.mean():+.6f}  std={coord_vals.std():.6f}  "
              f"min={coord_vals.min():+.6f}  max={coord_vals.max():+.6f}")
    if n_coords > 10:
        # Also show last 3
        print(f"    ...")
        for ci in range(n_coords - 3, n_coords):
            coord_vals = block_data[:, ci]
            print(f"    coord {start+ci:4d}: mean={coord_vals.mean():+.6f}  std={coord_vals.std():.6f}  "
                  f"min={coord_vals.min():+.6f}  max={coord_vals.max():+.6f}")
    
    # 2b: Cross-sample variance check
    coord_stds = block_data.std(axis=0)
    n_near_const = (coord_stds < 0.01).sum()
    n_low_var = (coord_stds < 0.05).sum()
    n_med_var = ((coord_stds >= 0.05) & (coord_stds < 0.1)).sum()
    n_high_var = (coord_stds >= 0.1).sum()
    
    print(f"\n  --- Cross-sample variance distribution ---")
    print(f"    Near-constant (std < 0.01): {n_near_const}/{n_coords} ({100*n_near_const/n_coords:.1f}%)")
    print(f"    Low variance  (std < 0.05): {n_low_var}/{n_coords} ({100*n_low_var/n_coords:.1f}%)")
    print(f"    Medium var    (0.05-0.10):  {n_med_var}/{n_coords} ({100*n_med_var/n_coords:.1f}%)")
    print(f"    High variance (std >= 0.10): {n_high_var}/{n_coords} ({100*n_high_var/n_coords:.1f}%)")
    
    # 2c: Correlation with Tc
    print(f"\n  --- Correlation with Tc ---")
    correlations = np.array([
        np.corrcoef(block_data[:, ci], tc_raw_np)[0, 1]
        for ci in range(n_coords)
    ])
    # Handle NaN correlations (constant coords)
    valid_corr = correlations[~np.isnan(correlations)]
    if len(valid_corr) > 0:
        abs_corr = np.abs(valid_corr)
        top_k = min(5, len(valid_corr))
        top_indices = np.argsort(abs_corr)[-top_k:][::-1]
        
        # Map back to original indices
        valid_idx = np.where(~np.isnan(correlations))[0]
        
        print(f"    Valid correlations: {len(valid_corr)}/{n_coords}")
        print(f"    Max |corr|: {abs_corr.max():.4f}")
        print(f"    Mean |corr|: {abs_corr.mean():.4f}")
        print(f"    Top-{top_k} correlated coords with Tc:")
        for rank, idx in enumerate(top_indices):
            orig_ci = valid_idx[idx]
            print(f"      #{rank+1}: coord {start+orig_ci:4d}  r={correlations[orig_ci]:+.4f}")
    else:
        print(f"    All correlations NaN (all coords constant?)")
    
    # For discovery block, subsample to avoid printing 1536 coords
    if name == "discovery":
        # Show distribution summary instead of all coords
        print(f"\n  --- Discovery block distribution summary ---")
        # Split into sub-regions
        sub_regions = [
            ("first 100 (512-611)", 0, 100),
            ("mid 100 (780-879)", 268, 368),
            ("last 100 (1948-2047)", n_coords-100, n_coords),
        ]
        for label, s, e in sub_regions:
            sub = block_data[:, s:e]
            sub_stds = sub.std(axis=0)
            print(f"    {label}: mean={sub.mean():+.6f}, std={sub.std():.6f}, "
                  f"near_const={int((sub_stds<0.01).sum())}/100")

# ============================================================================
# SECTION 3: Thermodynamic block Tc analysis
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Thermodynamic Block - Tc Coordinate Analysis")
print("=" * 80)

thermo_start = PhysicsZ.THERMO_START
tc_coord = z_cpu[:, PhysicsZ.TC]         # coord 210
tc_onset = z_cpu[:, PhysicsZ.TC_ONSET]    # coord 211
tc_mid = z_cpu[:, PhysicsZ.TC_MIDPOINT]   # coord 212
tc_zero = z_cpu[:, PhysicsZ.TC_ZERO]      # coord 213
delta_tc = z_cpu[:, PhysicsZ.DELTA_TC]    # coord 214

print(f"\nCoord {PhysicsZ.TC} (TC):         mean={tc_coord.mean():+.6f}, std={tc_coord.std():.6f}, "
      f"min={tc_coord.min():+.6f}, max={tc_coord.max():+.6f}")
print(f"Coord {PhysicsZ.TC_ONSET} (TC_ONSET):   mean={tc_onset.mean():+.6f}, std={tc_onset.std():.6f}, "
      f"min={tc_onset.min():+.6f}, max={tc_onset.max():+.6f}")
print(f"Coord {PhysicsZ.TC_MIDPOINT} (TC_MIDPOINT): mean={tc_mid.mean():+.6f}, std={tc_mid.std():.6f}, "
      f"min={tc_mid.min():+.6f}, max={tc_mid.max():+.6f}")
print(f"Coord {PhysicsZ.TC_ZERO} (TC_ZERO):   mean={tc_zero.mean():+.6f}, std={tc_zero.std():.6f}, "
      f"min={tc_zero.min():+.6f}, max={tc_zero.max():+.6f}")
print(f"Coord {PhysicsZ.DELTA_TC} (DELTA_TC):  mean={delta_tc.mean():+.6f}, std={delta_tc.std():.6f}, "
      f"min={delta_tc.min():+.6f}, max={delta_tc.max():+.6f}")

# Correlation of z[TC] with actual Tc input
corr_tc = np.corrcoef(tc_coord, tc_raw_np)[0, 1]
print(f"\nCorrelation of z[TC] (coord 210) with actual Tc: r = {corr_tc:+.4f}")

# Also check correlation for other Tc coords
corr_onset = np.corrcoef(tc_onset, tc_raw_np)[0, 1]
corr_mid = np.corrcoef(tc_mid, tc_raw_np)[0, 1]
corr_zero = np.corrcoef(tc_zero, tc_raw_np)[0, 1]
print(f"Correlation of z[TC_ONSET] (coord 211) with actual Tc: r = {corr_onset:+.4f}")
print(f"Correlation of z[TC_MIDPOINT] (coord 212) with actual Tc: r = {corr_mid:+.4f}")
print(f"Correlation of z[TC_ZERO] (coord 213) with actual Tc: r = {corr_zero:+.4f}")

# Check ordering: Tc_onset >= Tc_midpoint >= Tc_zero
n_onset_ge_mid = (tc_onset >= tc_mid).sum()
n_mid_ge_zero = (tc_mid >= tc_zero).sum()
n_both_ordered = ((tc_onset >= tc_mid) & (tc_mid >= tc_zero)).sum()

print(f"\nTc ordering check (Tc_onset >= Tc_midpoint >= Tc_zero):")
print(f"  Tc_onset >= Tc_midpoint: {n_onset_ge_mid}/{N_SAMPLES} ({100*n_onset_ge_mid/N_SAMPLES:.1f}%)")
print(f"  Tc_midpoint >= Tc_zero:  {n_mid_ge_zero}/{N_SAMPLES} ({100*n_mid_ge_zero/N_SAMPLES:.1f}%)")
print(f"  Both (fully ordered):    {n_both_ordered}/{N_SAMPLES} ({100*n_both_ordered/N_SAMPLES:.1f}%)")

# Show some example triples
print(f"\n  First 10 sample Tc triples [onset, midpoint, zero] vs actual Tc:")
for i in range(min(10, N_SAMPLES)):
    ordered = "OK" if tc_onset[i] >= tc_mid[i] >= tc_zero[i] else "VIOLATED"
    print(f"    Sample {i:3d}: [{tc_onset[i]:+.4f}, {tc_mid[i]:+.4f}, {tc_zero[i]:+.4f}]  "
          f"Tc_actual={tc_raw_np[i]:.2f}K  {ordered}")

# ============================================================================
# SECTION 4: Summary statistics
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Summary")
print("=" * 80)

print("\nBlock-level overview table:")
print(f"{'Block':<18s} {'Coords':>10s} {'Mean':>10s} {'Std':>10s} {'NearConst':>10s} {'|Tc_corr|':>10s}")
print("-" * 70)

for name in block_order:
    start, end = blocks[name]
    block_data = z_cpu[:, start:end]
    coord_stds = block_data.std(axis=0)
    n_near_const = int((coord_stds < 0.01).sum())
    n_coords = end - start
    
    # Tc correlation (mean absolute)
    correlations = np.array([
        np.corrcoef(block_data[:, ci], tc_raw_np)[0, 1]
        for ci in range(n_coords)
    ])
    valid = correlations[~np.isnan(correlations)]
    mean_abs_corr = np.abs(valid).mean() if len(valid) > 0 else float('nan')
    
    print(f"{name:<18s} {f'{start}:{end}':>10s} {block_data.mean():>+10.4f} {block_data.std():>10.4f} "
          f"{f'{n_near_const}/{n_coords}':>10s} {mean_abs_corr:>10.4f}")

print("\nDone.")
