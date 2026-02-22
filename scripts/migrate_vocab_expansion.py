#!/usr/bin/env python3
"""
Vocab Expansion Checkpoint Migration Script

After merging new datasets, the fraction vocab grows (new stoichiometries from
decimal->fraction conversion). This script expands the token_embedding and
output_proj layers in the checkpoint to accommodate the new vocab size.

The vocab layout is:
  [0-4] special tokens | [5-122] elements | [123-142] integers |
  [143..143+N_frac-1] fractions | [ISO_UNK] | [isotopes...]

When new fractions are added, ISO_UNK and isotope indices shift automatically
(FractionAwareTokenizer computes them from _frac_offset + n_fractions).
This script handles the reindexing by:
  1. Copying all pre-fraction weights as-is (unchanged)
  2. Building a mapping from old fraction/iso indices to new indices
  3. Copying old fraction weights to their new positions
  4. Initializing NEW fraction weights from FRAC_UNK embedding + noise
  5. Copying ISO_UNK and isotope weights to their new positions

Usage:
    python scripts/migrate_vocab_expansion.py \\
        --checkpoint outputs/checkpoint_v14_migrated.pt \\
        --output outputs/checkpoint_v15_expanded.pt \\
        --fraction-vocab data/fraction_vocab.json \\
        --isotope-vocab data/isotope_vocab.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superconductor.tokenizer.fraction_tokenizer import FractionAwareTokenizer
from superconductor.models.autoregressive_decoder import EnhancedTransformerDecoder


def migrate_checkpoint(
    checkpoint_path: str,
    output_path: str,
    fraction_vocab_path: str,
    isotope_vocab_path: str,
    max_formula_len: int = 30,
):
    """Migrate checkpoint for expanded fraction vocab.

    Args:
        checkpoint_path: Path to current checkpoint (V14.0)
        output_path: Path to save expanded checkpoint
        fraction_vocab_path: Path to NEW fraction_vocab.json (rebuilt after merge)
        isotope_vocab_path: Path to isotope_vocab.json (unchanged)
        max_formula_len: Maximum formula length
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dicts
    if 'encoder_state_dict' in checkpoint:
        enc_state = checkpoint['encoder_state_dict']
        dec_state = checkpoint['decoder_state_dict']
    elif 'model_state_dict' in checkpoint:
        full_state = checkpoint['model_state_dict']
        enc_state = {k.replace('encoder.', ''): v for k, v in full_state.items() if k.startswith('encoder.')}
        dec_state = {k.replace('decoder.', ''): v for k, v in full_state.items() if k.startswith('decoder.')}
    else:
        raise ValueError(f"Checkpoint format not recognized. Keys: {list(checkpoint.keys())}")

    # Strip torch.compile prefix
    def strip_orig_mod(state_dict):
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    if any('_orig_mod.' in k for k in enc_state):
        enc_state = strip_orig_mod(enc_state)
    if any('_orig_mod.' in k for k in dec_state):
        dec_state = strip_orig_mod(dec_state)

    # Load OLD fraction vocab to build index mapping.
    # fraction_vocab_old.json must exist (checked into repo) for correct index remapping.
    # Without it, old and new fraction lists are identical and migration silently fails.
    old_frac_vocab_path = PROJECT_ROOT / "data" / "fraction_vocab_old.json"
    if not old_frac_vocab_path.exists():
        raise FileNotFoundError(
            f"fraction_vocab_old.json not found at {old_frac_vocab_path}. "
            f"This file is required for fraction index remapping during migration. "
            f"It should be checked into the repo (git pull to restore)."
        )

    # Load OLD tokenizer (with pre-expansion vocab)
    old_tokenizer = FractionAwareTokenizer(
        str(old_frac_vocab_path),
        max_len=max_formula_len,
        isotope_vocab_path=isotope_vocab_path
    )

    # Load NEW tokenizer (with expanded vocab)
    new_tokenizer = FractionAwareTokenizer(
        fraction_vocab_path, max_len=max_formula_len,
        isotope_vocab_path=isotope_vocab_path
    )

    old_vocab_size = old_tokenizer.vocab_size
    new_vocab_size = new_tokenizer.vocab_size
    d_model = dec_state['token_embedding.weight'].shape[1]

    print(f"  Old tokenizer: vocab={old_vocab_size}, fractions={old_tokenizer.n_fraction_tokens}")
    print(f"  New tokenizer: vocab={new_vocab_size}, fractions={new_tokenizer.n_fraction_tokens}")
    print(f"  d_model: {d_model}")

    if old_vocab_size == new_vocab_size:
        print("  Vocab size unchanged — no migration needed.")
        return

    n_new_fractions = new_tokenizer.n_fraction_tokens - old_tokenizer.n_fraction_tokens
    print(f"  New fractions added: {n_new_fractions}")

    # Build old->new index mapping for fractions
    # The frac_offset is the same (143), but the fraction list order may differ
    frac_offset = old_tokenizer.fraction_token_start  # 143

    # Load old and new fraction lists
    with open(str(old_frac_vocab_path) if old_frac_vocab_path.exists() else fraction_vocab_path) as f:
        old_vocab = json.load(f)
    with open(fraction_vocab_path) as f:
        new_vocab = json.load(f)

    old_frac_list = old_vocab["fractions"]
    new_frac_list = new_vocab["fractions"]

    # Map: old fraction string -> old index
    old_frac_to_idx = {frac: frac_offset + i for i, frac in enumerate(old_frac_list)}
    # Map: new fraction string -> new index
    new_frac_to_idx = {frac: frac_offset + i for i, frac in enumerate(new_frac_list)}

    # Build old_idx -> new_idx mapping for all tokens
    # Indices before fractions (0..142) are unchanged
    idx_map = {i: i for i in range(frac_offset)}

    # Map old fractions to new positions
    for frac_str, old_idx in old_frac_to_idx.items():
        if frac_str in new_frac_to_idx:
            idx_map[old_idx] = new_frac_to_idx[frac_str]

    # Map old ISO_UNK and isotope indices to new positions
    old_iso_unk = old_tokenizer.iso_unk_idx
    new_iso_unk = new_tokenizer.iso_unk_idx
    if old_iso_unk is not None and new_iso_unk is not None:
        idx_map[old_iso_unk] = new_iso_unk

    old_iso_start = old_tokenizer.isotope_token_start
    new_iso_start = new_tokenizer.isotope_token_start
    if old_iso_start is not None and new_iso_start is not None:
        n_isotopes = old_tokenizer.n_isotope_tokens
        for i in range(n_isotopes):
            idx_map[old_iso_start + i] = new_iso_start + i

    print(f"  Index mapping covers {len(idx_map)} tokens")

    # =========================================================================
    # ENCODER: Copy as-is
    # =========================================================================
    print("\n--- Encoder ---")
    new_enc_state = dict(enc_state)
    print(f"  Copied: {len(new_enc_state)} parameters (unchanged)")

    # =========================================================================
    # DECODER: Expand vocab-dependent layers
    # =========================================================================
    print("\n--- Decoder ---")
    new_dec_state = {}

    FRAC_UNK_IDX = 4  # FRAC_UNK is always at index 4

    for key, value in dec_state.items():
        # --- Token embedding ---
        if key == 'token_embedding.weight':
            old_embed = value  # [old_vocab, d_model]
            new_embed = torch.zeros(new_vocab_size, d_model, dtype=value.dtype)

            # Copy mapped weights
            for old_idx, new_idx in idx_map.items():
                if old_idx < old_embed.shape[0]:
                    new_embed[new_idx] = old_embed[old_idx]

            # Initialize NEW fraction tokens from FRAC_UNK + noise
            frac_unk_embed = old_embed[FRAC_UNK_IDX]
            new_frac_strs = set(new_frac_list) - set(old_frac_list)
            n_initialized = 0
            for frac_str in new_frac_strs:
                new_idx = new_frac_to_idx[frac_str]
                new_embed[new_idx] = frac_unk_embed + torch.randn(d_model, dtype=value.dtype) * 0.02
                n_initialized += 1
            print(f"  {key}: [{old_vocab_size}, {d_model}] -> [{new_vocab_size}, {d_model}]")
            print(f"    Initialized {n_initialized} new fraction embeddings from FRAC_UNK")

            new_dec_state[key] = new_embed
            continue

        # --- Output projection weight ---
        if key == 'output_proj.4.weight':
            old_weight = value  # [old_vocab, d_model]
            new_weight = torch.zeros(new_vocab_size, d_model, dtype=value.dtype)

            for old_idx, new_idx in idx_map.items():
                if old_idx < old_weight.shape[0]:
                    new_weight[new_idx] = old_weight[old_idx]

            # New fractions: small Xavier init
            new_frac_strs = set(new_frac_list) - set(old_frac_list)
            for frac_str in new_frac_strs:
                new_idx = new_frac_to_idx[frac_str]
                nn.init.xavier_uniform_(new_weight[new_idx:new_idx + 1])

            new_dec_state[key] = new_weight
            print(f"  {key}: [{old_weight.shape[0]}, {d_model}] -> [{new_vocab_size}, {d_model}]")
            continue

        # --- Output projection bias ---
        if key == 'output_proj.4.bias':
            old_bias = value  # [old_vocab]
            new_bias = torch.zeros(new_vocab_size, dtype=value.dtype)

            for old_idx, new_idx in idx_map.items():
                if old_idx < old_bias.shape[0]:
                    new_bias[new_idx] = old_bias[old_idx]

            new_dec_state[key] = new_bias
            print(f"  {key}: [{old_bias.shape[0]}] -> [{new_vocab_size}]")
            continue

        # Everything else: copy as-is
        # This includes V14.3 token_type_head.* and heads_to_memory.* parameters
        # which are vocab-independent and transfer directly.
        new_dec_state[key] = value

    # =========================================================================
    # Build output checkpoint
    # =========================================================================
    new_checkpoint = {
        'encoder_state_dict': new_enc_state,
        'decoder_state_dict': new_dec_state,
        'epoch': checkpoint.get('epoch', 0),
        'algo_version': 'V15.0',
        'migration_source': str(checkpoint_path),
        'migration_source_version': checkpoint.get('algo_version', 'V14.x'),
        'migration_source_epoch': checkpoint.get('epoch', 0),
        'tokenizer_vocab_size': new_tokenizer.vocab_size,
        'tokenizer_n_fractions': new_tokenizer.n_fraction_tokens,
        'tokenizer_n_isotopes': new_tokenizer.n_isotope_tokens,
        'stoich_input_dim': checkpoint.get('stoich_input_dim', 13),
        'd_model': d_model,
        'nhead': checkpoint.get('nhead', 8),
        'dim_feedforward': dec_state.get(
            'transformer_decoder.layers.0.linear1.weight', torch.zeros(2048, d_model)
        ).shape[0],
        'num_layers': max(
            int(k.split('.')[2]) for k in dec_state
            if k.startswith('transformer_decoder.layers.')
        ) + 1 if any(k.startswith('transformer_decoder.layers.') for k in dec_state) else 12,
    }

    if 'optimizer_state_dict' in checkpoint:
        print("\n  NOTE: Optimizer state NOT transferred (will be reinitialized)")

    # V14.1: Carry over training metadata so checkpoint_best isn't overwritten on resume
    for key in ['prev_exact', 'best_exact', 'manifest', 'scheduler_type']:
        if key in checkpoint:
            new_checkpoint[key] = checkpoint[key]
            if key in ('prev_exact', 'best_exact'):
                print(f"  Preserved {key}={checkpoint[key]}")

    print(f"\nSaving expanded checkpoint to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)

    # =========================================================================
    # Verification
    # =========================================================================
    print("\n--- Verification ---")
    try:
        dim_feedforward = new_dec_state.get(
            'transformer_decoder.layers.0.linear1.weight', torch.zeros(2048, d_model)
        ).shape[0]
        num_layers = max(
            int(k.split('.')[2]) for k in new_dec_state
            if k.startswith('transformer_decoder.layers.')
        ) + 1
        nhead = checkpoint.get('nhead', 8)

        test_decoder = EnhancedTransformerDecoder(
            latent_dim=2048,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_formula_len,
            n_memory_tokens=16,
            encoder_skip_dim=256,
            use_skip_connection=False,
            use_stoich_conditioning=True,
            max_elements=12,
            n_stoich_tokens=4,
            vocab_size=new_tokenizer.vocab_size,
            stoich_input_dim=13,
        )
        missing, unexpected = test_decoder.load_state_dict(new_dec_state, strict=False)
        if missing:
            print(f"  WARNING: Missing keys: {missing}")
        if unexpected:
            print(f"  WARNING: Unexpected keys: {unexpected}")
        if not missing and not unexpected:
            print(f"  PASS: Decoder loads cleanly (vocab={new_tokenizer.vocab_size})")

        print(f"  Encoder: {len(new_enc_state)} parameters (unchanged)")
        print(f"\n  Migration complete!")
    except Exception as e:
        print(f"  FAIL: Verification error: {e}")
        raise

    return new_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Expand checkpoint for new fraction vocab')
    parser.add_argument('--checkpoint', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'checkpoint_v14_migrated.pt'),
                        help='Path to current checkpoint')
    parser.add_argument('--output', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'checkpoint_v15_expanded.pt'),
                        help='Path to save expanded checkpoint')
    parser.add_argument('--fraction-vocab', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'fraction_vocab.json'),
                        help='Path to NEW fraction_vocab.json')
    parser.add_argument('--isotope-vocab', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'isotope_vocab.json'),
                        help='Path to isotope_vocab.json')
    parser.add_argument('--max-len', type=int, default=30,
                        help='Maximum formula length (default: 30)')
    args = parser.parse_args()

    migrate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        fraction_vocab_path=args.fraction_vocab,
        isotope_vocab_path=args.isotope_vocab,
        max_formula_len=args.max_len,
    )


if __name__ == '__main__':
    main()
