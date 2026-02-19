#!/usr/bin/env python3
"""
V12.x → V13.0 Checkpoint Weight Transfer Script

Transfers weights from a V12.41/V12.42 checkpoint to a V13.0 model with
semantic fraction tokenization.

Weight transfer strategy:
- Full copy: All encoder weights, encoder heads, decoder internals
- Partial transfer: Token embeddings and output projections (row-mapped by token)
- Reshaped: stoich_to_memory (37→13 input dim)
- Dropped: encoder.numden_head (removed in V13.0)
- New (random/physics-informed init): Fraction token embedding rows

Usage:
    python scripts/migrate_v12_to_v13.py \\
        --checkpoint outputs/checkpoint_best.pt \\
        --output outputs/checkpoint_v13_migrated.pt \\
        --fraction-vocab data/fraction_vocab.json
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superconductor.tokenizer.fraction_tokenizer import FractionAwareTokenizer
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, TOKEN_TO_IDX as OLD_TOKEN_TO_IDX,
    IDX_TO_TOKEN as OLD_IDX_TO_TOKEN, VOCAB_SIZE as OLD_VOCAB_SIZE,
    PAD_IDX as OLD_PAD_IDX
)
from superconductor.models.attention_vae import FullMaterialsVAE


def build_token_mapping(tokenizer: FractionAwareTokenizer) -> dict:
    """Build old→new token index mapping for embedding/output transfer.

    Returns:
        dict mapping old_token_idx → new_token_idx for tokens in both vocabs
    """
    mapping = {}

    # Special tokens: old <PAD>=0, <START>=1, <END>=2 → new <PAD>=0, <BOS>=1, <EOS>=2
    old_specials = {'<PAD>': 0, '<START>': 1, '<END>': 2}
    for old_tok, new_idx in old_specials.items():
        if old_tok in OLD_TOKEN_TO_IDX:
            mapping[OLD_TOKEN_TO_IDX[old_tok]] = new_idx

    # Element tokens
    from superconductor.tokenizer.fraction_tokenizer import ELEMENTS
    for elem in ELEMENTS[1:]:
        if elem in OLD_TOKEN_TO_IDX and elem in tokenizer._token_to_id:
            mapping[OLD_TOKEN_TO_IDX[elem]] = tokenizer._token_to_id[elem]

    # Digit tokens: old has '0'-'9', new has '1'-'20'
    # Map digits '1'-'9' directly (they exist in both)
    for d in range(1, 10):
        old_tok = str(d)
        if old_tok in OLD_TOKEN_TO_IDX and old_tok in tokenizer._token_to_id:
            mapping[OLD_TOKEN_TO_IDX[old_tok]] = tokenizer._token_to_id[old_tok]

    return mapping


def initialize_fraction_embeddings(
    tokenizer: FractionAwareTokenizer,
    old_embedding: torch.Tensor,
    d_model: int,
) -> torch.Tensor:
    """Physics-informed initialization for fraction token embeddings.

    Strategy: interpolate between nearest integer token embeddings weighted
    by the fraction's float value. E.g., FRAC:3/4 initialized as
    0.75 * embed(1) + 0.25 * embed(0) plus small noise.

    Args:
        tokenizer: The V13 FractionAwareTokenizer
        old_embedding: Old embedding weight matrix [old_vocab_size, d_model]
        d_model: Embedding dimension

    Returns:
        Tensor of shape [n_fraction_tokens, d_model] with initialized embeddings
    """
    n_fractions = tokenizer.n_fraction_tokens
    frac_embeddings = torch.zeros(n_fractions, d_model)

    # Get old integer embeddings (digits '0'-'9' from old vocab)
    old_digit_embeds = {}
    for d in range(10):
        tok = str(d)
        if tok in OLD_TOKEN_TO_IDX:
            old_digit_embeds[d] = old_embedding[OLD_TOKEN_TO_IDX[tok]].clone()

    # Default: use mean of digit embeddings as base
    if old_digit_embeds:
        mean_embed = torch.stack(list(old_digit_embeds.values())).mean(dim=0)
    else:
        mean_embed = torch.zeros(d_model)

    for i in range(n_fractions):
        token_id = tokenizer.fraction_token_start + i
        frac_value = tokenizer.fraction_token_to_value(token_id)

        # Interpolate between floor and ceil integer embeddings
        floor_val = int(math.floor(frac_value))
        ceil_val = int(math.ceil(frac_value))
        alpha = frac_value - floor_val  # Interpolation weight

        if floor_val in old_digit_embeds and ceil_val in old_digit_embeds:
            base = (1 - alpha) * old_digit_embeds[floor_val] + alpha * old_digit_embeds[ceil_val]
        elif floor_val in old_digit_embeds:
            base = old_digit_embeds[floor_val]
        elif ceil_val in old_digit_embeds:
            base = old_digit_embeds[ceil_val]
        else:
            base = mean_embed

        # Add small noise for diversity
        noise = torch.randn(d_model) * 0.01
        frac_embeddings[i] = base + noise

    return frac_embeddings


def migrate_checkpoint(
    checkpoint_path: str,
    output_path: str,
    fraction_vocab_path: str,
    max_formula_len: int = 30,
):
    """Migrate V12.x checkpoint to V13.0 format.

    Args:
        checkpoint_path: Path to V12.x checkpoint
        output_path: Path to save V13.0 checkpoint
        fraction_vocab_path: Path to fraction_vocab.json
        max_formula_len: Maximum formula length (V13.0 default: 30)
    """
    print(f"Loading V12.x checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dicts
    if 'encoder_state_dict' in checkpoint:
        enc_state = checkpoint['encoder_state_dict']
        dec_state = checkpoint['decoder_state_dict']
    elif 'model_state_dict' in checkpoint:
        # Handle unified state dict
        full_state = checkpoint['model_state_dict']
        enc_state = {k.replace('encoder.', ''): v for k, v in full_state.items() if k.startswith('encoder.')}
        dec_state = {k.replace('decoder.', ''): v for k, v in full_state.items() if k.startswith('decoder.')}
    else:
        raise ValueError(f"Checkpoint format not recognized. Keys: {list(checkpoint.keys())}")

    # Handle torch.compile _orig_mod prefix — strip it for clean loading
    def strip_orig_mod(state_dict):
        new_state = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state[new_key] = value
        return new_state

    n_compiled_enc = sum(1 for k in enc_state if '_orig_mod.' in k)
    n_compiled_dec = sum(1 for k in dec_state if '_orig_mod.' in k)
    if n_compiled_enc > 0 or n_compiled_dec > 0:
        print(f"  Stripping torch.compile _orig_mod prefix ({n_compiled_enc} enc, {n_compiled_dec} dec keys)")
        enc_state = strip_orig_mod(enc_state)
        dec_state = strip_orig_mod(dec_state)

    # Load tokenizer
    tokenizer = FractionAwareTokenizer(fraction_vocab_path, max_len=max_formula_len)
    print(f"V13.0 tokenizer: vocab_size={tokenizer.vocab_size}, n_fractions={tokenizer.n_fraction_tokens}")

    # Build token mapping
    token_map = build_token_mapping(tokenizer)
    print(f"Token mapping: {len(token_map)} tokens mapped from old→new")

    # =========================================================================
    # ENCODER: Remove numden_head, keep everything else
    # =========================================================================
    print("\n--- Encoder Weight Transfer ---")
    new_enc_state = {}
    dropped_keys = []
    for key, value in enc_state.items():
        if key.startswith('numden_head.'):
            dropped_keys.append(key)
            continue
        new_enc_state[key] = value
    print(f"  Copied: {len(new_enc_state)} encoder parameters")
    print(f"  Dropped: {len(dropped_keys)} numden_head parameters: {dropped_keys}")

    # =========================================================================
    # DECODER: Transfer with token remapping
    # =========================================================================
    print("\n--- Decoder Weight Transfer ---")
    new_dec_state = {}
    d_model = dec_state['token_embedding.weight'].shape[1]
    old_vocab_size = dec_state['token_embedding.weight'].shape[0]
    new_vocab_size = tokenizer.vocab_size
    print(f"  Old vocab: {old_vocab_size}, New vocab: {new_vocab_size}, d_model: {d_model}")

    for key, value in dec_state.items():
        # --- Token embedding: remap rows ---
        if key == 'token_embedding.weight':
            new_embed = torch.zeros(new_vocab_size, d_model)
            # Copy mapped tokens
            for old_idx, new_idx in token_map.items():
                if old_idx < old_vocab_size:
                    new_embed[new_idx] = value[old_idx]
            # Initialize integer tokens 10-20 from digit embeddings
            # '10' = mean('1', '0'), etc.
            for val in range(10, 21):
                tok_str = str(val)
                if tok_str in tokenizer._token_to_id:
                    new_idx = tokenizer._token_to_id[tok_str]
                    # Initialize as mean of digit embeddings
                    digits = [int(d) for d in str(val)]
                    digit_embeds = []
                    for d in digits:
                        old_tok = str(d)
                        if old_tok in OLD_TOKEN_TO_IDX:
                            digit_embeds.append(value[OLD_TOKEN_TO_IDX[old_tok]])
                    if digit_embeds:
                        new_embed[new_idx] = torch.stack(digit_embeds).mean(dim=0) + torch.randn(d_model) * 0.01

            # Initialize fraction tokens with physics-informed interpolation
            frac_embeds = initialize_fraction_embeddings(tokenizer, value, d_model)
            frac_start = tokenizer.fraction_token_start
            new_embed[frac_start:frac_start + tokenizer.n_fraction_tokens] = frac_embeds

            new_dec_state[key] = new_embed
            print(f"  {key}: [{old_vocab_size}, {d_model}] → [{new_vocab_size}, {d_model}]")
            continue

        # --- Output projection final layer: remap rows ---
        if key == 'output_proj.4.weight':  # nn.Linear(d_model, vocab_size) weight
            new_weight = torch.zeros(new_vocab_size, d_model)
            for old_idx, new_idx in token_map.items():
                if old_idx < value.shape[0]:
                    new_weight[new_idx] = value[old_idx]
            # Fraction rows: small random init (will be trained in Phase A)
            frac_start = tokenizer.fraction_token_start
            nn.init.xavier_uniform_(new_weight[frac_start:frac_start + tokenizer.n_fraction_tokens].unsqueeze(0))
            new_dec_state[key] = new_weight.squeeze(0) if new_weight.dim() > 2 else new_weight
            print(f"  {key}: [{value.shape[0]}, {d_model}] → [{new_vocab_size}, {d_model}]")
            continue

        if key == 'output_proj.4.bias':  # nn.Linear(d_model, vocab_size) bias
            new_bias = torch.zeros(new_vocab_size)
            for old_idx, new_idx in token_map.items():
                if old_idx < value.shape[0]:
                    new_bias[new_idx] = value[old_idx]
            new_dec_state[key] = new_bias
            print(f"  {key}: [{value.shape[0]}] → [{new_vocab_size}]")
            continue

        # --- stoich_to_memory: reshape from 37→13 input dim ---
        if key == 'stoich_to_memory.0.weight':  # First linear: [d_model, stoich_input_dim]
            old_in_dim = value.shape[1]  # Should be 37
            new_in_dim = 13  # fractions(12) + count(1)
            if old_in_dim == 37 and new_in_dim == 13:
                # Copy columns: first 12 (fractions) + last 1 (count)
                new_weight = torch.zeros(value.shape[0], new_in_dim)
                new_weight[:, :12] = value[:, :12]  # Fraction columns
                new_weight[:, 12] = value[:, 36]    # Count column (last of 37)
                new_dec_state[key] = new_weight
                print(f"  {key}: [{value.shape[0]}, {old_in_dim}] → [{value.shape[0]}, {new_in_dim}] (fractions + count)")
            else:
                print(f"  WARNING: {key} unexpected shape [{value.shape}], copying as-is")
                new_dec_state[key] = value
            continue

        # --- Positional encoding: truncate to new max_len if needed ---
        if key == 'pos_encoding.pe':
            old_len = value.shape[1]  # [1, old_max_len, d_model]
            if old_len > max_formula_len:
                new_dec_state[key] = value[:, :max_formula_len, :]
                print(f"  {key}: [{1}, {old_len}, {d_model}] → [{1}, {max_formula_len}, {d_model}] (truncated)")
                continue
            # else: copy as-is (same or smaller)

        # --- Everything else: copy as-is ---
        new_dec_state[key] = value

    # Count copied vs modified
    n_copied = sum(1 for k in new_dec_state if k not in {
        'token_embedding.weight', 'output_proj.4.weight', 'output_proj.4.bias',
        'stoich_to_memory.0.weight', 'pos_encoding.pe'
    })
    print(f"  Copied unchanged: {n_copied} decoder parameters")

    # =========================================================================
    # Build output checkpoint
    # =========================================================================
    new_checkpoint = {
        'encoder_state_dict': new_enc_state,
        'decoder_state_dict': new_dec_state,
        'epoch': checkpoint.get('epoch', 0),
        'algo_version': 'V13.0',
        'migration_source': str(checkpoint_path),
        'migration_source_version': checkpoint.get('algo_version', 'V12.x'),
        'migration_source_epoch': checkpoint.get('epoch', 0),
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'tokenizer_n_fractions': tokenizer.n_fraction_tokens,
        'stoich_input_dim': 13,
    }

    # Copy optimizer state if present (will need resetting for Phase A)
    if 'optimizer_state_dict' in checkpoint:
        print("\n  NOTE: Optimizer state NOT transferred (will be reinitialized for V13.0 Phase A)")

    # V13.0: Do NOT carry over best_exact — V12 performance is not comparable to V13
    # (different tokenization, different vocab, different sequence lengths)

    # Copy other metadata (manifest only)
    for key in ['manifest']:
        if key in checkpoint:
            new_checkpoint[key] = checkpoint[key]

    print(f"\nSaving V13.0 checkpoint to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)

    # Verification: try loading into new model
    print("\n--- Verification ---")
    try:
        # Build new decoder with V13.0 config
        test_decoder = EnhancedTransformerDecoder(
            latent_dim=2048,
            d_model=d_model,
            nhead=8,
            num_layers=12,
            dim_feedforward=2048,
            max_len=max_formula_len,
            n_memory_tokens=16,
            encoder_skip_dim=256,
            use_skip_connection=True,
            use_stoich_conditioning=True,
            max_elements=12,
            n_stoich_tokens=4,
            vocab_size=tokenizer.vocab_size,
            stoich_input_dim=13,
        )
        missing, unexpected = test_decoder.load_state_dict(new_dec_state, strict=False)
        if missing:
            print(f"  WARNING: Missing keys in decoder: {missing}")
        if unexpected:
            print(f"  WARNING: Unexpected keys in decoder: {unexpected}")
        if not missing and not unexpected:
            print(f"  PASS: Decoder state loads cleanly (vocab_size={tokenizer.vocab_size})")

        # Test encoder loading with strict=False (numden_head keys dropped)
        print(f"  Encoder: {len(new_enc_state)} parameters transferred (numden_head dropped)")
        print(f"\n  Migration complete!")
    except Exception as e:
        print(f"  FAIL: Verification error: {e}")
        raise

    return new_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Migrate V12.x checkpoint to V13.0')
    parser.add_argument('--checkpoint', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'checkpoint_best.pt'),
                        help='Path to V12.x checkpoint')
    parser.add_argument('--output', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'checkpoint_v13_migrated.pt'),
                        help='Path to save V13.0 checkpoint')
    parser.add_argument('--fraction-vocab', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'fraction_vocab.json'),
                        help='Path to fraction_vocab.json')
    parser.add_argument('--max-len', type=int, default=30,
                        help='Maximum formula length (V13.0 default: 30)')
    args = parser.parse_args()

    migrate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        fraction_vocab_path=args.fraction_vocab,
        max_formula_len=args.max_len,
    )


if __name__ == '__main__':
    main()
