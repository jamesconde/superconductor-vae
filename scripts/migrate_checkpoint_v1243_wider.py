#!/usr/bin/env python3
"""
V12.43 Migration Script: Net2Net 12.5% Wider Encoder + Decoder Expansion

Expands BOTH the encoder and decoder:
  - Encoder: fusion_dim 256→288, encoder_hidden [512,256]→[576,288],
             decoder_hidden [256,512]→[288,576]
  - Decoder: d_model 512→576, dim_feedforward 2048→2304

Uses Net2Net weight transfer to preserve learned representations while
adding ~14-16% more capacity across the full model.

Source: "outputs/checkpoint_best V1241.pt" (V12.41, d_model=512)
Target: "outputs/checkpoint_best.pt" (V12.43, d_model=576)

Usage:
    python scripts/migrate_checkpoint_v1243_wider.py
    python scripts/migrate_checkpoint_v1243_wider.py --dry-run
    python scripts/migrate_checkpoint_v1243_wider.py --checkpoint "path/to/checkpoint.pt"
    python scripts/migrate_checkpoint_v1243_wider.py --output "path/to/output.pt"
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

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, VOCAB_SIZE
)
from superconductor.models.net2net_expansion import (
    expand_enhanced_decoder,
    expand_full_materials_vae,
)


# Old config (V12.41)
OLD_CONFIG = {
    'latent_dim': 2048,
    'fusion_dim': 256,
    'magpie_dim': 145,
    'encoder_hidden': [512, 256],
    'decoder_hidden': [256, 512],
    'd_model': 512,
    'dim_feedforward': 2048,
    'nhead': 8,
    'num_layers': 12,
    'n_memory_tokens': 16,
    'max_len': 60,
    'encoder_skip_dim': 256,
    'max_elements': 12,
    'n_stoich_tokens': 4,
    'element_embed_dim': 128,
}

# New config (V12.43)
NEW_CONFIG = {
    'fusion_dim': 288,
    'encoder_hidden': [576, 288],
    'decoder_hidden': [288, 576],
    'd_model': 576,
    'dim_feedforward': 2304,
}

DEFAULT_CHECKPOINT = PROJECT_ROOT / 'outputs' / 'checkpoint_best V1241.pt'
DEFAULT_OUTPUT = PROJECT_ROOT / 'outputs' / 'checkpoint_best.pt'
DEFAULT_BACKUP = PROJECT_ROOT / 'outputs' / 'checkpoint_best_V1241_pre_v1243.pt'


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
        description='V12.43: Net2Net 12.5% wider encoder + decoder expansion'
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
        '--backup', type=str, default=str(DEFAULT_BACKUP),
        help=f'Backup path for source checkpoint (default: {DEFAULT_BACKUP})'
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
    backup_path = Path(args.backup)

    print("=" * 70)
    print("V12.43 Net2Net Encoder+Decoder Expansion")
    print("  fusion_dim:      256 -> 288")
    print("  encoder_hidden:  [512, 256] -> [576, 288]")
    print("  decoder_hidden:  [256, 512] -> [288, 576]")
    print("  d_model:         512 -> 576")
    print("  dim_feedforward: 2048 -> 2304")
    print("  latent_dim:      2048 (unchanged)")
    print("=" * 70)
    print(f"  Source: {checkpoint_path}")
    print(f"  Output: {output_path}")
    print(f"  Backup: {backup_path}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # =========================================================================
    # 1. Load checkpoint
    # =========================================================================
    print("[1/7] Loading checkpoint...", flush=True)
    if not checkpoint_path.exists():
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    epoch = checkpoint.get('epoch', 'unknown')
    best_exact = checkpoint.get('best_exact', 'unknown')
    print(f"  Loaded epoch {epoch}, best_exact={best_exact}", flush=True)
    print(f"  Keys: {list(checkpoint.keys())}", flush=True)

    # =========================================================================
    # 1b. Auto-detect magpie_dim from checkpoint
    # =========================================================================
    enc_state_raw = checkpoint['encoder_state_dict']
    # Look for magpie_encoder.0.weight to detect actual magpie_dim
    for k, v in enc_state_raw.items():
        clean_k = k.replace('_orig_mod.', '')
        if clean_k == 'magpie_encoder.0.weight':
            detected_magpie = v.shape[1]
            if detected_magpie != OLD_CONFIG['magpie_dim']:
                print(f"  Auto-detected magpie_dim={detected_magpie} from checkpoint "
                      f"(config had {OLD_CONFIG['magpie_dim']})", flush=True)
                OLD_CONFIG['magpie_dim'] = detected_magpie
            break

    # =========================================================================
    # 2. Strip _orig_mod. prefixes
    # =========================================================================
    print("\n[2/7] Stripping torch.compile prefixes...", flush=True)
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

    # =========================================================================
    # 3. Create old-architecture models and load weights
    # =========================================================================
    print("\n[3/7] Creating old-architecture models...", flush=True)

    # Encoder (V12.41)
    old_encoder = FullMaterialsVAE(
        n_elements=118,
        element_embed_dim=OLD_CONFIG['element_embed_dim'],
        n_attention_heads=8,
        magpie_dim=OLD_CONFIG['magpie_dim'],
        fusion_dim=OLD_CONFIG['fusion_dim'],
        encoder_hidden=OLD_CONFIG['encoder_hidden'],
        latent_dim=OLD_CONFIG['latent_dim'],
        decoder_hidden=OLD_CONFIG['decoder_hidden'],
        dropout=0.1,
        use_numden_head=True,  # V12.41 has numden_head
    )

    missing_enc, unexpected_enc = old_encoder.load_state_dict(enc_state, strict=False)
    if missing_enc:
        print(f"  Encoder missing keys: {missing_enc}", flush=True)
    if unexpected_enc:
        print(f"  Encoder unexpected keys: {unexpected_enc}", flush=True)
    if not missing_enc and not unexpected_enc:
        print("  All encoder weights loaded successfully", flush=True)

    old_enc_params = sum(p.numel() for p in old_encoder.parameters())
    print(f"  Old encoder parameters: {old_enc_params:,}", flush=True)

    # Decoder (V12.41)
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
        memory_bottleneck_dim=0,  # V12.41 uses direct MLP (no bottleneck)
    )

    missing_dec, unexpected_dec = old_decoder.load_state_dict(dec_state, strict=False)
    if missing_dec:
        # V14.3 heads not in V12.41 checkpoint — expected
        v143_missing = [k for k in missing_dec if 'heads_to_memory' in k or 'token_type_head' in k]
        other_missing = [k for k in missing_dec if k not in v143_missing]
        if v143_missing:
            print(f"  Decoder: {len(v143_missing)} V14.3 keys missing (expected, will be freshly initialized)", flush=True)
        if other_missing:
            print(f"  WARNING: Decoder missing non-V14.3 keys: {other_missing}", flush=True)
    if unexpected_dec:
        print(f"  Decoder unexpected keys: {unexpected_dec}", flush=True)
    if not missing_dec and not unexpected_dec:
        print("  All decoder weights loaded successfully", flush=True)

    old_dec_params = sum(p.numel() for p in old_decoder.parameters())
    print(f"  Old decoder parameters: {old_dec_params:,}", flush=True)

    # =========================================================================
    # 4. Apply Net2Net expansion to ENCODER
    # =========================================================================
    print(f"\n[4/7] Expanding encoder: fusion_dim {OLD_CONFIG['fusion_dim']} -> {NEW_CONFIG['fusion_dim']}...",
          flush=True)

    new_encoder = expand_full_materials_vae(
        old_encoder,
        new_fusion_dim=NEW_CONFIG['fusion_dim'],
        new_encoder_hidden=NEW_CONFIG['encoder_hidden'],
        new_decoder_hidden=NEW_CONFIG['decoder_hidden'],
        noise_std=args.noise_std,
        verbose=True
    )

    new_enc_params = sum(p.numel() for p in new_encoder.parameters())

    # =========================================================================
    # 5. Apply Net2Net expansion to DECODER
    # =========================================================================
    print(f"\n[5/7] Expanding decoder: d_model {OLD_CONFIG['d_model']} -> {NEW_CONFIG['d_model']}, "
          f"dim_ff {OLD_CONFIG['dim_feedforward']} -> {NEW_CONFIG['dim_feedforward']}...", flush=True)

    new_decoder = expand_enhanced_decoder(
        old_decoder,
        new_d_model=NEW_CONFIG['d_model'],
        new_dim_feedforward=NEW_CONFIG['dim_feedforward'],
        noise_std=args.noise_std,
        verbose=True
    )

    new_dec_params = sum(p.numel() for p in new_decoder.parameters())

    # =========================================================================
    # 6. Verify forward pass through BOTH models
    # =========================================================================
    print("\n[6/7] Verifying forward pass...", flush=True)
    new_encoder.eval()
    new_decoder.eval()

    with torch.no_grad():
        batch_size = 2

        # Encoder forward pass
        element_indices = torch.randint(1, 118, (batch_size, 12))
        element_fractions = torch.rand(batch_size, 12)
        element_mask = torch.ones(batch_size, 12, dtype=torch.bool)
        magpie_features = torch.randn(batch_size, OLD_CONFIG['magpie_dim'])
        tc = torch.randn(batch_size, 1)

        try:
            enc_out = new_encoder(
                element_indices, element_fractions, element_mask,
                magpie_features, tc
            )
            z = enc_out['z']
            print(f"  Encoder forward pass OK!", flush=True)
            print(f"    z shape: {z.shape}", flush=True)
            print(f"    tc_pred shape: {enc_out['tc_pred'].shape}", flush=True)
            print(f"    attended_input shape: {enc_out['attended_input'].shape}", flush=True)
        except Exception as e:
            print(f"  ERROR in encoder forward pass: {e}", flush=True)
            import traceback
            traceback.print_exc()
            if not args.dry_run:
                print("\n  Aborting migration due to encoder forward pass failure.", flush=True)
                sys.exit(1)

        # Decoder forward pass
        target = torch.randint(0, VOCAB_SIZE, (batch_size, 20))
        encoder_skip = torch.randn(batch_size, OLD_CONFIG['encoder_skip_dim'])
        stoich_pred = torch.randn(batch_size, 37)  # V12.41: max_elements*3 + 1

        try:
            logits, generated, stop_logits, *_extra = new_decoder(
                z, target, encoder_skip=encoder_skip,
                teacher_forcing_ratio=1.0, stoich_pred=stoich_pred
            )
            print(f"  Decoder forward pass OK!", flush=True)
            print(f"    logits shape: {logits.shape}", flush=True)
            print(f"    generated shape: {generated.shape}", flush=True)
            print(f"    stop_logits shape: {stop_logits.shape}", flush=True)
        except Exception as e:
            print(f"  ERROR in decoder forward pass: {e}", flush=True)
            import traceback
            traceback.print_exc()
            if not args.dry_run:
                print("\n  Aborting migration due to decoder forward pass failure.", flush=True)
                sys.exit(1)

    # Verify gradient flow (backward pass) — outside no_grad context
    new_encoder.train()
    new_decoder.train()
    enc_out2 = new_encoder(element_indices, element_fractions, element_mask, magpie_features, tc)
    z2 = enc_out2['z']
    logits2, _, stop_logits2, *_ = new_decoder(
        z2, target, encoder_skip=encoder_skip,
        teacher_forcing_ratio=1.0, stoich_pred=stoich_pred
    )
    loss = logits2.sum() + enc_out2['tc_pred'].sum()
    loss.backward()

    # Check for non-zero gradients
    enc_grad_ok = sum(1 for p in new_encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    dec_grad_ok = sum(1 for p in new_decoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    enc_total = sum(1 for p in new_encoder.parameters() if p.requires_grad)
    dec_total = sum(1 for p in new_decoder.parameters() if p.requires_grad)
    print(f"  Gradient flow: encoder {enc_grad_ok}/{enc_total}, decoder {dec_grad_ok}/{dec_total}", flush=True)

    # Check no NaN
    enc_nan = any(torch.isnan(p.grad).any() for p in new_encoder.parameters() if p.grad is not None)
    dec_nan = any(torch.isnan(p.grad).any() for p in new_decoder.parameters() if p.grad is not None)
    if enc_nan or dec_nan:
        print(f"  WARNING: NaN gradients detected! enc={enc_nan}, dec={dec_nan}", flush=True)
    else:
        print(f"  No NaN gradients - OK", flush=True)

    # Zero out grads
    new_encoder.zero_grad()
    new_decoder.zero_grad()

    if args.dry_run:
        print("\n[DRY RUN] Would save to:", output_path)
        print("  Exiting without saving.")
        _print_summary(old_enc_params, new_enc_params, old_dec_params, new_dec_params, epoch)
        return

    # =========================================================================
    # 7. Build new checkpoint and save
    # =========================================================================
    print("\n[7/7] Building and saving new checkpoint...", flush=True)
    new_enc_state = new_encoder.state_dict()
    new_dec_state = new_decoder.state_dict()

    new_checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': new_enc_state,
        'decoder_state_dict': new_dec_state,
    }

    # Carry over non-model state (entropy manager, etc.) but NOT optimizer/scheduler
    carry_over_keys = ['entropy_manager_state', 'prev_exact', 'best_exact',
                       'manifest', 'theory_loss_fn_state_dict', 'physics_z_loss_fn_state_dict']
    for key in carry_over_keys:
        if key in checkpoint:
            new_checkpoint[key] = checkpoint[key]
            print(f"  Carried over: {key}", flush=True)

    # Explicitly do NOT carry over optimizer/scheduler state (shapes changed)
    skipped = []
    for key in ['enc_optimizer_state_dict', 'dec_optimizer_state_dict',
                'enc_scheduler_state_dict', 'dec_scheduler_state_dict', 'scheduler_type']:
        if key in checkpoint:
            skipped.append(key)
    if skipped:
        print(f"  Skipped (shapes changed): {skipped}", flush=True)

    # Save backup of source checkpoint
    if not backup_path.exists():
        print(f"  Backing up source: {checkpoint_path} -> {backup_path}", flush=True)
        shutil.copy2(checkpoint_path, backup_path)
    else:
        print(f"  Backup already exists: {backup_path}", flush=True)

    # Save expanded checkpoint
    torch.save(new_checkpoint, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)", flush=True)

    _print_summary(old_enc_params, new_enc_params, old_dec_params, new_dec_params, epoch)


def _print_summary(old_enc, new_enc, old_dec, new_dec, epoch):
    """Print migration summary."""
    old_total = old_enc + old_dec
    new_total = new_enc + new_dec
    print("\n" + "=" * 70)
    print("Migration Complete!")
    print("=" * 70)
    print(f"  Epoch:          {epoch}")
    print(f"  Encoder params: {old_enc:,} -> {new_enc:,} "
          f"(+{(new_enc / old_enc - 1) * 100:.1f}%)")
    print(f"  Decoder params: {old_dec:,} -> {new_dec:,} "
          f"(+{(new_dec / old_dec - 1) * 100:.1f}%)")
    print(f"  Total params:   {old_total:,} -> {new_total:,} "
          f"(+{(new_total / old_total - 1) * 100:.1f}%)")
    print()
    print(f"  Encoder dimensions:")
    print(f"    fusion_dim:      256 -> 288")
    print(f"    encoder_hidden:  [512, 256] -> [576, 288]")
    print(f"    decoder_hidden:  [256, 512] -> [288, 576]")
    print(f"  Decoder dimensions:")
    print(f"    d_model:         512 -> 576")
    print(f"    dim_feedforward: 2048 -> 2304")
    print(f"    nhead:           8 (head_dim 64 -> 72)")
    print(f"  Unchanged:")
    print(f"    latent_dim:      2048")
    print(f"    num_layers:      12")
    print(f"    n_memory_tokens: 16")
    print(f"  Optimizer state:   RESET (param shapes changed)")
    print(f"\n  Next: Update MODEL_CONFIG in train_v12_clean.py and resume training")


if __name__ == '__main__':
    main()
