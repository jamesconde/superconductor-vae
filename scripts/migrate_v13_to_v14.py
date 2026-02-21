#!/usr/bin/env python3
"""
V13.0 → V14.0 Checkpoint Weight Transfer Script

Expands decoder embeddings and output projection from V13 vocab (4355 tokens)
to V14 vocab (4647 tokens = 4355 + 1 ISO_UNK + 291 isotopes).

Weight transfer strategy:
- Full copy: All encoder weights (unchanged)
- Full copy: All decoder weights except token_embedding and output_proj final layer
- Expanded: Token embedding rows (291 isotope rows initialized from parent element + noise)
- Expanded: Output projection final layer (same initialization strategy)
- New: ISO_UNK token row initialized to mean of special tokens

Isotope embedding initialization:
    Each isotope token is initialized as:
        embed(ISO:massX) = embed(X) + noise * scale
    where scale is proportional to |mass - natural_mass| / natural_mass,
    encoding that isotope variants should be close to their parent element
    but slightly perturbed by mass deviation.

Usage:
    python scripts/migrate_v13_to_v14.py \\
        --checkpoint outputs/checkpoint_v13_migrated.pt \\
        --output outputs/checkpoint_v14_migrated.pt \\
        --fraction-vocab data/fraction_vocab.json \\
        --isotope-vocab data/isotope_vocab.json
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superconductor.tokenizer.fraction_tokenizer import FractionAwareTokenizer
from superconductor.models.autoregressive_decoder import EnhancedTransformerDecoder


def initialize_isotope_embeddings(
    tokenizer: FractionAwareTokenizer,
    base_embedding: torch.Tensor,
    d_model: int,
    noise_scale: float = 0.02,
) -> torch.Tensor:
    """Physics-informed initialization for isotope token embeddings.

    Strategy: each isotope embedding = parent element embedding + scaled noise.
    The noise scale is proportional to mass deviation from the most abundant isotope,
    so common isotopes (e.g., 16O) start very close to the element embedding while
    rare/exotic ones (e.g., 18O) have more perturbation.

    Args:
        tokenizer: The V14 FractionAwareTokenizer (with isotope vocab loaded)
        base_embedding: Current embedding weight matrix [current_vocab_size, d_model]
        d_model: Embedding dimension
        noise_scale: Base noise magnitude

    Returns:
        Tensor of shape [n_isotope_tokens, d_model] with initialized embeddings
    """
    n_isotopes = tokenizer.n_isotope_tokens
    iso_embeddings = torch.zeros(n_isotopes, d_model, dtype=base_embedding.dtype)

    # Try to load natural abundances for mass-aware scaling
    try:
        from superconductor.encoders.isotope_properties import (
            ISOTOPE_DATABASE, get_most_abundant_isotope
        )
        has_isotope_db = True
    except ImportError:
        has_isotope_db = False

    iso_start = tokenizer.isotope_token_start
    for i in range(n_isotopes):
        token_id = iso_start + i
        elem_idx = tokenizer.element_idx_for_isotope(token_id)
        parent_embed = base_embedding[elem_idx].clone()

        # Compute mass-aware noise scale
        scale = noise_scale
        if has_isotope_db:
            elem = tokenizer.isotope_token_to_element(token_id)
            mass = tokenizer.isotope_token_to_mass(token_id)
            most_abundant = get_most_abundant_isotope(elem)
            if most_abundant is not None and most_abundant.atomic_mass > 0:
                # Scale noise by relative mass deviation
                mass_deviation = abs(mass - most_abundant.mass_number) / most_abundant.mass_number
                scale = noise_scale * (1.0 + mass_deviation)

        noise = torch.randn(d_model, dtype=base_embedding.dtype) * scale
        iso_embeddings[i] = parent_embed + noise

    return iso_embeddings


def migrate_checkpoint(
    checkpoint_path: str,
    output_path: str,
    fraction_vocab_path: str,
    isotope_vocab_path: str,
    max_formula_len: int = 30,
):
    """Migrate V13.0 checkpoint to V14.0 format.

    Args:
        checkpoint_path: Path to V13.0 checkpoint
        output_path: Path to save V14.0 checkpoint
        fraction_vocab_path: Path to fraction_vocab.json
        isotope_vocab_path: Path to isotope_vocab.json
        max_formula_len: Maximum formula length
    """
    print(f"Loading V13.0 checkpoint from {checkpoint_path}...")
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

    # Handle torch.compile _orig_mod prefix
    def strip_orig_mod(state_dict):
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    n_compiled_enc = sum(1 for k in enc_state if '_orig_mod.' in k)
    n_compiled_dec = sum(1 for k in dec_state if '_orig_mod.' in k)
    if n_compiled_enc > 0 or n_compiled_dec > 0:
        print(f"  Stripping torch.compile _orig_mod prefix ({n_compiled_enc} enc, {n_compiled_dec} dec keys)")
        enc_state = strip_orig_mod(enc_state)
        dec_state = strip_orig_mod(dec_state)

    # Load V14 tokenizer
    tokenizer = FractionAwareTokenizer(
        fraction_vocab_path, max_len=max_formula_len,
        isotope_vocab_path=isotope_vocab_path
    )
    print(f"V14.0 tokenizer: vocab_size={tokenizer.vocab_size}, "
          f"n_fractions={tokenizer.n_fraction_tokens}, n_isotopes={tokenizer.n_isotope_tokens}")

    old_vocab_size = dec_state['token_embedding.weight'].shape[0]
    new_vocab_size = tokenizer.vocab_size
    d_model = dec_state['token_embedding.weight'].shape[1]
    print(f"  Old vocab: {old_vocab_size}, New vocab: {new_vocab_size}, d_model: {d_model}")

    if old_vocab_size >= new_vocab_size:
        print(f"  WARNING: Old vocab ({old_vocab_size}) >= new vocab ({new_vocab_size}). "
              f"Checkpoint may already be V14.0.")

    # =========================================================================
    # ENCODER: Copy everything as-is (no changes for V14)
    # =========================================================================
    print("\n--- Encoder Weight Transfer ---")
    new_enc_state = dict(enc_state)
    print(f"  Copied: {len(new_enc_state)} encoder parameters (unchanged)")

    # =========================================================================
    # DECODER: Expand token_embedding and output_proj for isotope tokens
    # =========================================================================
    print("\n--- Decoder Weight Transfer ---")
    new_dec_state = {}

    for key, value in dec_state.items():
        # --- Token embedding: expand with isotope rows ---
        if key == 'token_embedding.weight':
            new_embed = torch.zeros(new_vocab_size, d_model, dtype=value.dtype)
            # Copy all existing V13 rows as-is
            new_embed[:old_vocab_size] = value

            # Initialize ISO_UNK token as mean of special tokens (PAD, BOS, EOS, UNK, FRAC_UNK)
            iso_unk_idx = tokenizer.iso_unk_idx
            if iso_unk_idx is not None:
                special_mean = value[:5].mean(dim=0)
                new_embed[iso_unk_idx] = special_mean + torch.randn(d_model, dtype=value.dtype) * 0.01

            # Initialize isotope tokens from parent element embeddings
            iso_embeds = initialize_isotope_embeddings(tokenizer, value, d_model)
            iso_start = tokenizer.isotope_token_start
            new_embed[iso_start:iso_start + tokenizer.n_isotope_tokens] = iso_embeds

            new_dec_state[key] = new_embed
            print(f"  {key}: [{old_vocab_size}, {d_model}] -> [{new_vocab_size}, {d_model}]")
            continue

        # --- Output projection final layer weight: expand rows ---
        if key == 'output_proj.4.weight':
            new_weight = torch.zeros(new_vocab_size, d_model, dtype=value.dtype)
            new_weight[:old_vocab_size] = value

            # ISO_UNK: small random init
            iso_unk_idx = tokenizer.iso_unk_idx
            if iso_unk_idx is not None:
                nn.init.xavier_uniform_(new_weight[iso_unk_idx:iso_unk_idx + 1])

            # Isotope rows: copy parent element row + small noise
            iso_start = tokenizer.isotope_token_start
            for i in range(tokenizer.n_isotope_tokens):
                token_id = iso_start + i
                elem_idx = tokenizer.element_idx_for_isotope(token_id)
                if elem_idx < old_vocab_size:
                    new_weight[token_id] = value[elem_idx] + torch.randn(d_model, dtype=value.dtype) * 0.01
                else:
                    nn.init.xavier_uniform_(new_weight[token_id:token_id + 1])

            new_dec_state[key] = new_weight
            print(f"  {key}: [{value.shape[0]}, {d_model}] -> [{new_vocab_size}, {d_model}]")
            continue

        # --- Output projection final layer bias: expand ---
        if key == 'output_proj.4.bias':
            new_bias = torch.zeros(new_vocab_size, dtype=value.dtype)
            new_bias[:old_vocab_size] = value

            # Isotope biases: copy parent element bias
            iso_start = tokenizer.isotope_token_start
            for i in range(tokenizer.n_isotope_tokens):
                token_id = iso_start + i
                elem_idx = tokenizer.element_idx_for_isotope(token_id)
                if elem_idx < old_vocab_size:
                    new_bias[token_id] = value[elem_idx]

            new_dec_state[key] = new_bias
            print(f"  {key}: [{value.shape[0]}] -> [{new_vocab_size}]")
            continue

        # --- Everything else: copy as-is ---
        new_dec_state[key] = value

    n_copied = sum(1 for k in new_dec_state if k not in {
        'token_embedding.weight', 'output_proj.4.weight', 'output_proj.4.bias'
    })
    print(f"  Copied unchanged: {n_copied} decoder parameters")

    # =========================================================================
    # Build output checkpoint
    # =========================================================================
    new_checkpoint = {
        'encoder_state_dict': new_enc_state,
        'decoder_state_dict': new_dec_state,
        'epoch': checkpoint.get('epoch', 0),
        'algo_version': 'V14.0',
        'migration_source': str(checkpoint_path),
        'migration_source_version': checkpoint.get('algo_version', 'V13.x'),
        'migration_source_epoch': checkpoint.get('epoch', 0),
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'tokenizer_n_fractions': tokenizer.n_fraction_tokens,
        'tokenizer_n_isotopes': tokenizer.n_isotope_tokens,
        'stoich_input_dim': checkpoint.get('stoich_input_dim', 13),
        'd_model': d_model,
        'nhead': checkpoint.get('nhead', 8),
        'dim_feedforward': dec_state.get('transformer_decoder.layers.0.linear1.weight', torch.zeros(2048, d_model)).shape[0],
        'num_layers': max(
            int(k.split('.')[2]) for k in dec_state
            if k.startswith('transformer_decoder.layers.')
        ) + 1 if any(k.startswith('transformer_decoder.layers.') for k in dec_state) else 12,
    }

    if 'optimizer_state_dict' in checkpoint:
        print("\n  NOTE: Optimizer state NOT transferred (will be reinitialized for V14.0)")

    # V14.1: Carry over training metadata so checkpoint_best isn't overwritten on resume
    for key in ['prev_exact', 'best_exact', 'manifest', 'scheduler_type']:
        if key in checkpoint:
            new_checkpoint[key] = checkpoint[key]
            if key in ('prev_exact', 'best_exact'):
                print(f"  Preserved {key}={checkpoint[key]}")

    print(f"\nSaving V14.0 checkpoint to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)

    # Verification
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
        print(f"  Auto-detected: d_model={d_model}, dim_feedforward={dim_feedforward}, "
              f"nhead={nhead}, num_layers={num_layers}")

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

        print(f"  Encoder: {len(new_enc_state)} parameters transferred (unchanged)")
        print(f"\n  Migration complete!")
    except Exception as e:
        print(f"  FAIL: Verification error: {e}")
        raise

    return new_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Migrate V13.0 checkpoint to V14.0')
    parser.add_argument('--checkpoint', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'checkpoint_v13_migrated.pt'),
                        help='Path to V13.0 checkpoint')
    parser.add_argument('--output', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'checkpoint_v14_migrated.pt'),
                        help='Path to save V14.0 checkpoint')
    parser.add_argument('--fraction-vocab', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'fraction_vocab.json'),
                        help='Path to fraction_vocab.json')
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
