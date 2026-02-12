#!/usr/bin/env python3
"""
One-Time Checkpoint Migration: V12.27 → V12.28

Migrates an old checkpoint (with 2-layer tc_head, 145-dim magpie) to the new
V12.28 architecture (4-layer residual Tc head, Tc classification head, 151-dim
magpie with physics features, updated SC head).

This script should be run ONCE after syncing the new code from GitHub to wherever
the latest checkpoint lives (e.g., Google Drive / Colab).

What it does:
  1. Loads the old checkpoint
  2. Strips _orig_mod. prefixes (torch.compile artifacts)
  3. Creates the new model architecture
  4. Applies shape-mismatch filtering (re-initializes reshaped layers)
  5. Applies Net2Net weight transfer (old tc_head → new tc_proj/tc_res_block/tc_out)
  6. Loads all compatible weights
  7. Saves a migrated checkpoint (backs up the original first)
  8. Resets optimizer state (must be rebuilt since param shapes changed)

Usage:
  python scripts/migrate_checkpoint_v1228.py [--checkpoint PATH] [--magpie-dim DIM]

  Defaults:
    --checkpoint  outputs/checkpoint_best.pt
    --magpie-dim  auto (detected from CSV, falls back to 151)
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _strip_compiled_prefix(state_dict):
    """Strip '_orig_mod.' prefix from compiled model state dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('_orig_mod.'):
            new_key = new_key[len('_orig_mod.'):]
        new_key = new_key.replace('._orig_mod.', '.')
        new_state_dict[new_key] = value
    return new_state_dict


def detect_magpie_dim():
    """Detect magpie_dim from the CSV (same logic as train_v12_clean.py)."""
    import pandas as pd

    csv_path = PROJECT_ROOT / "data" / "processed" / "supercon_fractions_contrastive.csv"
    if not csv_path.exists():
        print(f"  CSV not found at {csv_path}, using magpie_dim=151")
        return 151

    df = pd.read_csv(csv_path, nrows=5)

    # Same exclusion logic as train_v12_clean.py load_and_prepare_data()
    exclude_cols = {
        'composition', 'formula_original', 'tc_original', 'tc', 'tc_log',
        'is_superconductor', 'requires_high_pressure', 'sc_family',
        'data_source', 'source', 'formula_fraction', 'unnamed: 0',
    }
    magpie_cols = [
        c for c in df.columns
        if c.lower() not in exclude_cols
        and df[c].dtype in ('float64', 'float32', 'int64', 'int32')
    ]
    dim = len(magpie_cols)
    print(f"  Detected magpie_dim={dim} from CSV ({len(df.columns)} total columns)")
    return dim


def main():
    parser = argparse.ArgumentParser(description="Migrate checkpoint to V12.28 architecture")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint_best.pt",
                        help="Path to checkpoint (relative to project root or absolute)")
    parser.add_argument("--magpie-dim", type=str, default="auto",
                        help="Magpie feature dimension (auto=detect from CSV, or integer)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without saving")
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print("V12.28 Checkpoint Migration")
    print("=" * 70)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # ── Step 1: Load checkpoint ──────────────────────────────────────────
    print("Step 1: Loading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Epoch: {epoch}")
    print(f"  Top-level keys: {list(checkpoint.keys())}")

    enc_state = checkpoint['encoder_state_dict']
    dec_state = checkpoint['decoder_state_dict']

    # ── Step 2: Strip _orig_mod. prefixes ────────────────────────────────
    print("\nStep 2: Handling _orig_mod. prefixes...")
    checkpoint_is_compiled = any(k.startswith('_orig_mod.') for k in enc_state.keys())
    if checkpoint_is_compiled:
        print(f"  Compiled checkpoint detected ({sum(1 for k in enc_state if k.startswith('_orig_mod.'))} keys with prefix)")
        enc_state = _strip_compiled_prefix(enc_state)
        dec_state = _strip_compiled_prefix(dec_state)
        print("  Stripped prefixes")
    else:
        print("  No _orig_mod. prefixes found")

    # Check for old tc_head
    old_tc_keys = [k for k in enc_state if k.startswith('tc_head.')]
    has_old_tc_head = len(old_tc_keys) > 0
    print(f"  Old tc_head keys: {old_tc_keys}")

    # Check old magpie dim
    if 'magpie_encoder.0.weight' in enc_state:
        old_magpie_dim = enc_state['magpie_encoder.0.weight'].shape[1]
        print(f"  Old magpie_dim: {old_magpie_dim}")
    else:
        old_magpie_dim = 145

    # ── Step 3: Determine new architecture params ────────────────────────
    print("\nStep 3: Determining new architecture parameters...")
    if args.magpie_dim == "auto":
        magpie_dim = detect_magpie_dim()
    else:
        magpie_dim = int(args.magpie_dim)
        print(f"  Using magpie_dim={magpie_dim} (from command line)")

    # ── Step 4: Create new model ─────────────────────────────────────────
    print("\nStep 4: Creating new V12.28 model architecture...")
    from superconductor.models.attention_vae import FullMaterialsVAE

    model = FullMaterialsVAE(
        magpie_dim=magpie_dim,
        latent_dim=2048,
    )

    # Verify new architecture
    assert hasattr(model, 'tc_proj'), "Missing tc_proj"
    assert hasattr(model, 'tc_res_block'), "Missing tc_res_block"
    assert hasattr(model, 'tc_out'), "Missing tc_out"
    assert hasattr(model, 'tc_class_head'), "Missing tc_class_head"
    assert hasattr(model, 'upgrade_tc_head_from_checkpoint'), "Missing upgrade method"
    print(f"  tc_proj: {model.tc_proj}")
    print(f"  tc_res_block: {model.tc_res_block}")
    print(f"  tc_out: {model.tc_out}")
    print(f"  tc_class_head: {model.tc_class_head}")

    # ── Step 5: Shape-mismatch filtering ─────────────────────────────────
    print("\nStep 5: Shape-mismatch filtering...")
    model_state = model.state_dict()
    dropped_keys = []
    for key in list(enc_state.keys()):
        if key in model_state and enc_state[key].shape != model_state[key].shape:
            print(f"  MISMATCH: {key}: checkpoint {enc_state[key].shape} vs model {model_state[key].shape}")
            dropped_keys.append(key)
            del enc_state[key]

    if dropped_keys:
        print(f"  Dropped {len(dropped_keys)} mismatched keys (will be re-initialized)")
    else:
        print("  No shape mismatches found")

    # ── Step 6: Load compatible weights ──────────────────────────────────
    print("\nStep 6: Loading compatible weights...")
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    print(f"  Missing keys (new/reshaped — randomly initialized): {len(missing)}")
    for k in missing:
        print(f"    {k}")
    print(f"  Unexpected keys (old — ignored): {len(unexpected)}")
    for k in unexpected:
        print(f"    {k}")

    # ── Step 7: Net2Net weight transfer ──────────────────────────────────
    print("\nStep 7: Net2Net weight transfer...")
    if has_old_tc_head:
        # Reload the original (pre-filtered) state for tc_head keys
        raw_enc_state = checkpoint['encoder_state_dict']
        if checkpoint_is_compiled:
            raw_enc_state = _strip_compiled_prefix(raw_enc_state)

        model.upgrade_tc_head_from_checkpoint(raw_enc_state)
        print("  Applied Net2Net transfer: tc_head.0 → tc_proj, tc_head.2 → tc_out")

        # Verify transfer
        with torch.no_grad():
            old_w = raw_enc_state['tc_head.0.weight']
            new_w = model.tc_proj.weight
            diff = (old_w - new_w).abs().max().item()
            print(f"  Verification: tc_proj weight max diff from old tc_head.0 = {diff:.6e}")
    else:
        print("  No old tc_head found — skipping Net2Net transfer")
        print("  (Checkpoint may already be V12.28 format)")

    # ── Step 8: Save migrated checkpoint ─────────────────────────────────
    print("\nStep 8: Saving migrated checkpoint...")

    if args.dry_run:
        print("  DRY RUN — not saving anything")
        print("\n  Migration would succeed. Run without --dry-run to apply.")
        return

    # Back up original
    backup_path = ckpt_path.with_suffix('.pt.bak_pre_v1228')
    if not backup_path.exists():
        shutil.copy2(ckpt_path, backup_path)
        print(f"  Backed up original to: {backup_path}")
    else:
        print(f"  Backup already exists: {backup_path}")

    # Build new checkpoint
    new_checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': model.state_dict(),
        'decoder_state_dict': dec_state,
        'prev_exact': checkpoint.get('prev_exact', 0.0),
        'best_exact': checkpoint.get('best_exact', 0.0),
    }

    # Preserve entropy manager state if present
    if 'entropy_manager_state' in checkpoint:
        new_checkpoint['entropy_manager_state'] = checkpoint['entropy_manager_state']

    # Preserve theory loss state if present
    if 'theory_loss_fn_state_dict' in checkpoint:
        new_checkpoint['theory_loss_fn_state_dict'] = checkpoint['theory_loss_fn_state_dict']

    # NOTE: Optimizer and scheduler states are NOT carried over.
    # Parameter shapes changed (magpie, sc_head, tc_head → tc_proj/tc_res_block/tc_out),
    # so the old optimizer state is invalid. The training script will create fresh optimizers.
    print("  Optimizer/scheduler state reset (param shapes changed)")

    torch.save(new_checkpoint, ckpt_path)
    print(f"  Saved migrated checkpoint: {ckpt_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Migration complete!")
    print("=" * 70)
    print(f"  Epoch preserved: {epoch}")
    print(f"  magpie_dim: {old_magpie_dim} → {magpie_dim}")
    print(f"  Tc head: 2-layer MLP → 4-layer residual (Net2Net transferred)")
    print(f"  New heads: tc_class_head (5 buckets), updated sc_head")
    print(f"  Optimizer: Reset (will be rebuilt on training resume)")
    print(f"  Backup: {backup_path}")
    print(f"\n  Next: Resume training normally with train_v12_clean.py")


if __name__ == '__main__':
    main()
