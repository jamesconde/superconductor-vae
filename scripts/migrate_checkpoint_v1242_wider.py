#!/usr/bin/env python3
"""
V12.42 Migration Script: Net2Net 2x Wider Decoder Expansion

Expands the decoder from d_model=512 to d_model=1024 and dim_feedforward
from 2048 to 4096 using Net2Net weight transfer.

The encoder is left UNCHANGED — this is a decoder-only expansion.

Source: "outputs/checkpoint_best V1241.pt" (V12.41, d_model=512, dim_ff=2048)
Target: "outputs/checkpoint_best.pt" (V12.42, d_model=1024, dim_ff=4096)

Usage:
    python scripts/migrate_checkpoint_v1242_wider.py
    python scripts/migrate_checkpoint_v1242_wider.py --dry-run
    python scripts/migrate_checkpoint_v1242_wider.py --checkpoint "path/to/checkpoint.pt"
    python scripts/migrate_checkpoint_v1242_wider.py --output "path/to/output.pt"
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, VOCAB_SIZE
)
from superconductor.models.net2net_expansion import expand_enhanced_decoder


# Old config (V12.41)
OLD_CONFIG = {
    'd_model': 512,
    'dim_feedforward': 2048,
    'latent_dim': 2048,
    'nhead': 8,
    'num_layers': 12,
    'n_memory_tokens': 16,
    'max_len': 60,
    'encoder_skip_dim': 256,
    'max_elements': 12,
    'n_stoich_tokens': 4,
}

# New config (V12.42)
NEW_CONFIG = {
    'd_model': 1024,
    'dim_feedforward': 4096,
}

DEFAULT_CHECKPOINT = PROJECT_ROOT / 'outputs' / 'checkpoint_best V1241.pt'
DEFAULT_OUTPUT = PROJECT_ROOT / 'outputs' / 'checkpoint_best.pt'


def strip_compiled_prefix(state_dict):
    """Strip '_orig_mod.' prefix from compiled model state dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('_orig_mod.'):
            new_key = new_key[len('_orig_mod.'):]
        new_key = new_key.replace('._orig_mod.', '.')
        new_state_dict[new_key] = value
    return new_state_dict


def count_params(state_dict, prefix=''):
    """Count total parameters in a state dict, optionally filtered by prefix."""
    total = 0
    for key, tensor in state_dict.items():
        if prefix and not key.startswith(prefix):
            continue
        total += tensor.numel()
    return total


def main():
    parser = argparse.ArgumentParser(
        description='V12.42: Net2Net 20% wider decoder expansion'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT),
        help=f'Source checkpoint path (default: {DEFAULT_CHECKPOINT})'
    )
    parser.add_argument(
        '--output', type=str, default=str(DEFAULT_OUTPUT),
        help=f'Output checkpoint path (default: {DEFAULT_OUTPUT})'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print expansion plan without saving'
    )
    parser.add_argument(
        '--noise-std', type=float, default=0.01,
        help='Noise std for new weight initialization (default: 0.01)'
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    print("=" * 70)
    print("V12.42 Net2Net Decoder Expansion: d_model 512 -> 1024")
    print("=" * 70)
    print(f"  Source: {checkpoint_path}")
    print(f"  Output: {output_path}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # 1. Load checkpoint
    print("[1/6] Loading checkpoint...", flush=True)
    if not checkpoint_path.exists():
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded epoch {epoch}", flush=True)
    print(f"  Keys: {list(checkpoint.keys())}", flush=True)

    # 2. Strip _orig_mod. prefixes
    print("\n[2/6] Stripping torch.compile prefixes...", flush=True)
    enc_state = checkpoint['encoder_state_dict']
    dec_state = checkpoint['decoder_state_dict']

    enc_compiled = any(k.startswith('_orig_mod.') or '._orig_mod.' in k for k in enc_state.keys())
    dec_compiled = any(k.startswith('_orig_mod.') or '._orig_mod.' in k for k in dec_state.keys())

    if enc_compiled:
        enc_state = strip_compiled_prefix(enc_state)
        print("  Stripped encoder prefixes", flush=True)
    else:
        print("  Encoder: no compiled prefixes found", flush=True)

    if dec_compiled:
        dec_state = strip_compiled_prefix(dec_state)
        print("  Stripped decoder prefixes", flush=True)
    else:
        print("  Decoder: no compiled prefixes found", flush=True)

    # 3. Create old-architecture decoder and load weights
    print("\n[3/6] Creating old-architecture decoder (d_model=512)...", flush=True)
    old_decoder = EnhancedTransformerDecoder(
        latent_dim=OLD_CONFIG['latent_dim'],
        d_model=OLD_CONFIG['d_model'],
        nhead=OLD_CONFIG['nhead'],
        num_layers=OLD_CONFIG['num_layers'],
        dim_feedforward=OLD_CONFIG['dim_feedforward'],
        dropout=0.1,
        max_len=OLD_CONFIG['max_len'],
        n_memory_tokens=OLD_CONFIG['n_memory_tokens'],
        encoder_skip_dim=OLD_CONFIG['encoder_skip_dim'],
        use_skip_connection=True,
        use_stoich_conditioning=True,
        max_elements=OLD_CONFIG['max_elements'],
        n_stoich_tokens=OLD_CONFIG['n_stoich_tokens'],
        use_gradient_checkpointing=False,
    )

    # Load old weights into old decoder
    missing, unexpected = old_decoder.load_state_dict(dec_state, strict=False)
    if missing:
        print(f"  WARNING: Missing keys: {missing}", flush=True)
    if unexpected:
        print(f"  WARNING: Unexpected keys: {unexpected}", flush=True)
    if not missing and not unexpected:
        print("  All decoder weights loaded successfully", flush=True)

    old_dec_params = sum(p.numel() for p in old_decoder.parameters())
    print(f"  Old decoder parameters: {old_dec_params:,}", flush=True)

    # 4. Apply Net2Net expansion
    print(f"\n[4/6] Expanding decoder: d_model {OLD_CONFIG['d_model']} -> {NEW_CONFIG['d_model']}, "
          f"dim_ff {OLD_CONFIG['dim_feedforward']} -> {NEW_CONFIG['dim_feedforward']}...", flush=True)

    new_decoder = expand_enhanced_decoder(
        old_decoder,
        new_d_model=NEW_CONFIG['d_model'],
        new_dim_feedforward=NEW_CONFIG['dim_feedforward'],
        noise_std=args.noise_std,
        verbose=True
    )

    new_dec_params = sum(p.numel() for p in new_decoder.parameters())
    print(f"\n  New decoder parameters: {new_dec_params:,}", flush=True)
    print(f"  Increase: +{new_dec_params - old_dec_params:,} "
          f"({(new_dec_params / old_dec_params - 1) * 100:.1f}%)", flush=True)

    # 5. Verify shapes by forward pass
    print("\n[5/6] Verifying forward pass...", flush=True)
    new_decoder.eval()
    with torch.no_grad():
        batch_size = 2
        z = torch.randn(batch_size, OLD_CONFIG['latent_dim'])
        # Target tokens: [batch, seq_len] of token indices
        target = torch.randint(0, VOCAB_SIZE, (batch_size, 20))
        encoder_skip = torch.randn(batch_size, OLD_CONFIG['encoder_skip_dim'])
        stoich_pred = torch.randn(batch_size, 37)  # max_elements*3 + 1

        try:
            logits, generated, stop_logits, *_extra = new_decoder(
                z, target, encoder_skip=encoder_skip,
                teacher_forcing_ratio=1.0, stoich_pred=stoich_pred
            )
            print(f"  Forward pass OK!", flush=True)
            print(f"    logits shape: {logits.shape}", flush=True)
            print(f"    generated shape: {generated.shape}", flush=True)
            print(f"    stop_logits shape: {stop_logits.shape}", flush=True)
        except Exception as e:
            print(f"  ERROR in forward pass: {e}", flush=True)
            import traceback
            traceback.print_exc()
            if not args.dry_run:
                print("\n  Aborting migration due to forward pass failure.", flush=True)
                sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Would save to:", output_path)
        print("  Exiting without saving.")
        return

    # 6. Build new checkpoint and save
    print("\n[6/6] Building and saving new checkpoint...", flush=True)
    new_dec_state = new_decoder.state_dict()

    new_checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': enc_state,  # Unchanged
        'decoder_state_dict': new_dec_state,
    }

    # Carry over non-model state (entropy manager, etc.) but NOT optimizer/scheduler
    # (param shapes changed, so optimizer state is invalid)
    carry_over_keys = ['entropy_manager_state', 'prev_exact', 'best_exact',
                       'manifest', 'theory_loss_fn_state_dict', 'physics_z_loss_fn_state_dict']
    for key in carry_over_keys:
        if key in checkpoint:
            new_checkpoint[key] = checkpoint[key]
            print(f"  Carried over: {key}", flush=True)

    # Explicitly do NOT carry over optimizer/scheduler state
    skipped = []
    for key in ['enc_optimizer_state_dict', 'dec_optimizer_state_dict',
                'enc_scheduler_state_dict', 'dec_scheduler_state_dict', 'scheduler_type']:
        if key in checkpoint:
            skipped.append(key)
    if skipped:
        print(f"  Skipped (shapes changed): {skipped}", flush=True)

    # Backup original if output would overwrite
    if output_path.exists():
        backup_path = output_path.with_suffix('.pt.bak')
        print(f"  Backing up existing: {output_path} -> {backup_path}", flush=True)
        shutil.copy2(output_path, backup_path)

    torch.save(new_checkpoint, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)", flush=True)

    # Summary
    enc_params = count_params(enc_state)
    print("\n" + "=" * 70)
    print("Migration Complete!")
    print("=" * 70)
    print(f"  Encoder params: {enc_params:,} (unchanged)")
    print(f"  Decoder params: {old_dec_params:,} -> {new_dec_params:,} "
          f"(+{(new_dec_params / old_dec_params - 1) * 100:.1f}%)")
    print(f"  Total params:   {enc_params + old_dec_params:,} -> {enc_params + new_dec_params:,}")
    print(f"  d_model:        {OLD_CONFIG['d_model']} -> {NEW_CONFIG['d_model']}")
    print(f"  dim_feedforward: {OLD_CONFIG['dim_feedforward']} -> {NEW_CONFIG['dim_feedforward']}")
    print(f"  Optimizer state: RESET (param shapes changed)")
    print(f"\n  Next: Update MODEL_CONFIG in train_v12_clean.py and resume training")


if __name__ == '__main__':
    main()
