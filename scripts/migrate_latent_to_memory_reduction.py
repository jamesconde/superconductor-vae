#!/usr/bin/env python3
"""V15.0: Migrate latent_to_memory from 42M-param 2-layer MLP to 1.6M-param bottleneck.

Standalone script — run on Colab (where the latest checkpoint lives) or locally.
No external dependencies beyond torch.

The migration uses SVD to find the most important directions in the old Layer 1
weights, preserving maximum information through a 512-dim bottleneck.

Usage:
    # Phase 1 only — SVD spectrum analysis (no changes written)
    python migrate_latent_to_memory_reduction.py checkpoint_best.pt --analyze-only

    # Full migration (saves backup + migrated checkpoint)
    python migrate_latent_to_memory_reduction.py checkpoint_best.pt

    # Custom bottleneck size
    python migrate_latent_to_memory_reduction.py checkpoint_best.pt --bottleneck-dim 256

    # Custom output path
    python migrate_latent_to_memory_reduction.py checkpoint_best.pt -o checkpoint_v15_migrated.pt
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch


def analyze_svd_spectrum(W1: torch.Tensor) -> torch.Tensor:
    """Analyze the SVD spectrum of Layer 1 weights.

    Prints cumulative variance retained at various bottleneck sizes.
    Returns singular values for further analysis.
    """
    print(f"\n{'='*60}")
    print("Phase 1: SVD Spectrum Analysis")
    print(f"{'='*60}")
    print(f"Layer 1 (latent_to_memory.0) shape: {list(W1.shape)}")
    print(f"  Parameters: {W1.numel():,}")

    # SVD: W1 = U @ diag(S) @ Vt
    U, S, Vt = torch.linalg.svd(W1.float(), full_matrices=False)
    total_variance = (S ** 2).sum().item()

    print(f"\nSingular value spectrum:")
    print(f"  Total variance (sum of S^2): {total_variance:.2f}")
    print(f"  Max singular value: {S[0]:.4f}")
    print(f"  Min singular value: {S[-1]:.6f}")
    print(f"  Condition number: {S[0] / S[-1]:.1f}")

    # Cumulative variance at different bottleneck sizes
    cumvar = torch.cumsum(S ** 2, dim=0) / total_variance
    print(f"\nCumulative variance retained:")
    for k in [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]:
        if k <= len(S):
            pct = cumvar[k - 1].item() * 100
            print(f"  Top-{k:>4d} singular vectors: {pct:6.2f}%")

    # Where does 95% / 99% / 99.9% lie?
    for threshold in [0.95, 0.99, 0.999]:
        idx = (cumvar >= threshold).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            k = idx[0].item() + 1
            print(f"  {threshold*100:.1f}% variance at k={k}")

    return S


def apply_migration(
    checkpoint: dict,
    bottleneck_dim: int = 1024,
    n_new_tokens: int = 16,
    d_model_override: int = None,
) -> dict:
    """Apply SVD-based migration to latent_to_memory weights.

    Old architecture:
        Layer 0: Linear(latent_dim → hidden)      [latent_to_memory.0.weight, .0.bias]
        Layer 1: GELU                              (no params)
        Layer 2: Linear(hidden → d_model*n_tokens) [latent_to_memory.2.weight, .2.bias]

    New architecture (bottleneck):
        Layer 0: Linear(latent_dim → bottleneck_dim)              [latent_to_memory.0.weight, .0.bias]
        Layer 1: LayerNorm(bottleneck_dim)                         [latent_to_memory.1.weight, .1.bias]
        Layer 2: GELU                                              (no params)
        Layer 3: Linear(bottleneck_dim → d_model * n_new_tokens)   [latent_to_memory.3.weight, .3.bias]

    d_model is auto-detected from token_embedding.weight in the checkpoint.

    Returns:
        Modified checkpoint dict (in-place mutation of decoder_state_dict).
    """
    dec_state = checkpoint['decoder_state_dict']

    # Strip _orig_mod prefix if present (compiled checkpoints)
    prefix = ''
    if any(k.startswith('_orig_mod.') for k in dec_state.keys()):
        prefix = '_orig_mod.'

    # Auto-detect d_model from token_embedding (shape: [vocab_size, d_model])
    embed_key = f'{prefix}token_embedding.weight'
    if embed_key in dec_state:
        d_model = dec_state[embed_key].shape[1]
        print(f"Auto-detected d_model={d_model} from token_embedding.weight {list(dec_state[embed_key].shape)}")
    elif d_model_override is not None:
        d_model = d_model_override
        print(f"Using d_model={d_model} from --d-model flag")
    else:
        print("ERROR: Cannot auto-detect d_model (no token_embedding.weight found) "
              "and no --d-model flag provided.")
        sys.exit(1)

    if d_model_override is not None and d_model_override != d_model:
        print(f"WARNING: --d-model {d_model_override} overrides auto-detected {d_model}")
        d_model = d_model_override

    key_w1 = f'{prefix}latent_to_memory.0.weight'
    key_b1 = f'{prefix}latent_to_memory.0.bias'
    key_w2 = f'{prefix}latent_to_memory.2.weight'
    key_b2 = f'{prefix}latent_to_memory.2.bias'

    # Validate old keys exist
    for k in [key_w1, key_b1, key_w2, key_b2]:
        if k not in dec_state:
            print(f"ERROR: Expected key '{k}' not found in decoder state dict.")
            print(f"  Available latent_to_memory keys:")
            for dk in sorted(dec_state.keys()):
                if 'latent_to_memory' in dk:
                    print(f"    {dk}: {dec_state[dk].shape}")
            sys.exit(1)

    W1 = dec_state[key_w1].float()
    b1 = dec_state[key_b1].float()
    W2 = dec_state[key_w2].float()
    b2 = dec_state[key_b2].float()

    old_n_tokens = W2.shape[0] // d_model
    new_output_dim = d_model * n_new_tokens

    # Sanity check: old token count should match n_new_tokens (unless intentionally reducing)
    if old_n_tokens != n_new_tokens:
        print(f"NOTE: Token count changing from {old_n_tokens} to {n_new_tokens}")

    print(f"\n{'='*60}")
    print("Phase 2: Applying SVD Migration")
    print(f"{'='*60}")
    print(f"Old architecture:")
    print(f"  Layer 0: Linear(2048 → {W1.shape[0]})  [{W1.numel():,} params]")
    print(f"  Layer 2: Linear({W2.shape[1]} → {W2.shape[0]})  [{W2.numel():,} params]")
    print(f"  Total: {W1.numel() + b1.numel() + W2.numel() + b2.numel():,} params")
    print(f"  Memory tokens: {old_n_tokens}, d_model: {d_model}")
    print(f"\nNew architecture:")
    print(f"  Layer 0: Linear(2048 → {bottleneck_dim})  [{2048 * bottleneck_dim:,} params]")
    print(f"  Layer 1: LayerNorm({bottleneck_dim})")
    print(f"  Layer 3: Linear({bottleneck_dim} → {new_output_dim})  [{bottleneck_dim * new_output_dim:,} params]")
    new_total = 2048 * bottleneck_dim + bottleneck_dim + bottleneck_dim * new_output_dim + new_output_dim + 2 * bottleneck_dim
    print(f"  Total: ~{new_total:,} params")
    print(f"  Memory tokens: {n_new_tokens}, d_model: {d_model}")

    # SVD of Layer 1: W1 [4096, 2048] = U [4096, 2048] @ diag(S) [2048] @ Vt [2048, 2048]
    U, S, Vt = torch.linalg.svd(W1, full_matrices=False)
    cumvar = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    retained = cumvar[bottleneck_dim - 1].item() * 100
    print(f"\nSVD: retaining top-{bottleneck_dim} directions ({retained:.1f}% variance)")

    # New Layer 0 weights: top-k right singular vectors, scaled by singular values
    # W1_new[i, j] = S[i] * Vt[i, j]  for i in [0, bottleneck_dim)
    # This maps 2048-dim input to bottleneck_dim-dim output along the most important directions
    S_top = S[:bottleneck_dim]  # [bottleneck_dim]
    Vt_top = Vt[:bottleneck_dim, :]  # [bottleneck_dim, 2048]
    W1_new = torch.diag(S_top) @ Vt_top  # [bottleneck_dim, 2048]

    # New Layer 0 bias: project old bias through top-k left singular vectors
    # The old bias b1 lives in the 4096-dim intermediate space.
    # To get the bottleneck bias, project: b1_new = U[:, :k]^T @ b1
    U_top = U[:, :bottleneck_dim]  # [4096, bottleneck_dim]
    b1_new = U_top.T @ b1  # [bottleneck_dim]

    # New Layer 3 weights: project old Layer 2 through top-k left singular vectors
    # Old: output = W2 @ GELU(W1 @ x + b1) + b2
    # The GELU(W1 @ x + b1) lives in 4096-dim space. After SVD reduction,
    # the bottleneck output lives in span(U_top). So:
    # W2_new = W2[:new_output_dim, :] @ U_top
    # When n_new_tokens == old_n_tokens, this uses the full W2 matrix.
    # When n_new_tokens < old_n_tokens, only first new_output_dim rows are kept.
    W2_new = W2[:new_output_dim, :] @ U_top  # [new_output_dim, bottleneck_dim]

    # New Layer 3 bias: first new_output_dim elements of old bias
    b2_new = b2[:new_output_dim]  # [new_output_dim]

    # LayerNorm init: gamma=1, beta=0 (identity transform initially)
    ln_weight = torch.ones(bottleneck_dim)
    ln_bias = torch.zeros(bottleneck_dim)

    print(f"\nNew weight shapes:")
    print(f"  latent_to_memory.0.weight: {list(W1_new.shape)}")
    print(f"  latent_to_memory.0.bias:   {list(b1_new.shape)}")
    print(f"  latent_to_memory.1.weight: {list(ln_weight.shape)}  (LayerNorm gamma=1)")
    print(f"  latent_to_memory.1.bias:   {list(ln_bias.shape)}  (LayerNorm beta=0)")
    print(f"  latent_to_memory.3.weight: {list(W2_new.shape)}")
    print(f"  latent_to_memory.3.bias:   {list(b2_new.shape)}")

    # Sanity: verify reconstruction quality on a random input
    with torch.no_grad():
        test_z = torch.randn(32, 2048)

        # Old path: W2 @ GELU(W1 @ z + b1) + b2, then take first new_output_dim dims
        old_hidden = torch.nn.functional.gelu(test_z @ W1.T + b1)
        old_full = old_hidden @ W2.T + b2
        old_output = old_full[:, :new_output_dim]

        # New path: W2_new @ GELU(W1_new @ z + b1_new) + b2_new
        # (no LayerNorm in this test — it's identity-initialized)
        new_hidden = torch.nn.functional.gelu(test_z @ W1_new.T + b1_new)
        new_output = new_hidden @ W2_new.T + b2_new

        cosine = torch.nn.functional.cosine_similarity(
            old_output.flatten(1), new_output.flatten(1), dim=1
        ).mean()
        mse = (old_output - new_output).pow(2).mean()
        rel_error = mse / old_output.pow(2).mean()

        print(f"\nReconstruction quality (32 random z vectors, first {n_new_tokens} tokens):")
        print(f"  Cosine similarity: {cosine:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  Relative MSE: {rel_error:.6f}")

    # Cast to original dtype
    orig_dtype = dec_state[key_w1].dtype
    W1_new = W1_new.to(orig_dtype)
    b1_new = b1_new.to(orig_dtype)
    W2_new = W2_new.to(orig_dtype)
    b2_new = b2_new.to(orig_dtype)
    ln_weight = ln_weight.to(orig_dtype)
    ln_bias = ln_bias.to(orig_dtype)

    # Remove old keys and insert new keys
    # Old: .0.weight, .0.bias, .2.weight, .2.bias
    # New: .0.weight, .0.bias, .1.weight, .1.bias (LayerNorm), .3.weight, .3.bias
    del dec_state[key_w1]
    del dec_state[key_b1]
    del dec_state[key_w2]
    del dec_state[key_b2]

    dec_state[f'{prefix}latent_to_memory.0.weight'] = W1_new
    dec_state[f'{prefix}latent_to_memory.0.bias'] = b1_new
    dec_state[f'{prefix}latent_to_memory.1.weight'] = ln_weight
    dec_state[f'{prefix}latent_to_memory.1.bias'] = ln_bias
    dec_state[f'{prefix}latent_to_memory.3.weight'] = W2_new
    dec_state[f'{prefix}latent_to_memory.3.bias'] = b2_new

    # Update n_memory_tokens in config if stored
    if 'config' in checkpoint:
        checkpoint['config']['n_memory_tokens'] = n_new_tokens
        checkpoint['config']['memory_bottleneck_dim'] = bottleneck_dim
        print(f"\nUpdated checkpoint config: n_memory_tokens={n_new_tokens}, "
              f"memory_bottleneck_dim={bottleneck_dim}")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='V15.0: Migrate latent_to_memory to bottleneck architecture via SVD'
    )
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to checkpoint file (e.g., checkpoint_best.pt)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only run SVD spectrum analysis, do not modify checkpoint')
    parser.add_argument('--bottleneck-dim', type=int, default=1024,
                        help='Bottleneck dimension (default: 1024)')
    parser.add_argument('--n-tokens', type=int, default=16,
                        help='Number of latent memory tokens (default: 16, same as original)')
    parser.add_argument('--d-model', type=int, default=None,
                        help='Transformer d_model (default: auto-detect from checkpoint)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path for migrated checkpoint (default: <dir>/checkpoint_v15_migrated.pt)')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)

    epoch = checkpoint.get('epoch', '?')
    best_exact = checkpoint.get('best_exact', '?')
    print(f"  Epoch: {epoch}, Best exact: {best_exact}")

    # Find Layer 1 weights (handle compiled prefix)
    dec_state = checkpoint['decoder_state_dict']
    prefix = ''
    if any(k.startswith('_orig_mod.') for k in dec_state.keys()):
        prefix = '_orig_mod.'

    key_w1 = f'{prefix}latent_to_memory.0.weight'
    if key_w1 not in dec_state:
        print(f"ERROR: Key '{key_w1}' not found in decoder state dict.")
        print("  This checkpoint may already be migrated or has unexpected structure.")
        sys.exit(1)

    W1 = dec_state[key_w1]
    print(f"  latent_to_memory.0.weight shape: {list(W1.shape)}")

    # Phase 1: SVD analysis
    analyze_svd_spectrum(W1)

    if args.analyze_only:
        print("\n--analyze-only: Stopping before migration.")
        return

    # Save backup BEFORE any modifications
    backup_path = checkpoint_path.parent / f"{checkpoint_path.stem}_pre_v15_contraction{checkpoint_path.suffix}"
    print(f"\nSaving pre-contraction backup: {backup_path}")
    shutil.copy2(str(checkpoint_path), str(backup_path))
    print(f"  Backup saved ({backup_path.stat().st_size / 1e9:.2f} GB)")

    # Phase 2: Apply migration
    checkpoint = apply_migration(
        checkpoint,
        bottleneck_dim=args.bottleneck_dim,
        n_new_tokens=args.n_tokens,
        d_model_override=args.d_model,
    )

    # Save migrated checkpoint
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.parent / f"checkpoint_v15_migrated.pt"

    print(f"\nSaving migrated checkpoint: {output_path}")
    torch.save(checkpoint, str(output_path))
    output_size = output_path.stat().st_size / 1e9
    print(f"  Saved ({output_size:.2f} GB)")

    # Summary
    old_params = W1.numel() + dec_state.get(f'{prefix}latent_to_memory.0.bias', torch.zeros(1)).numel()
    print(f"\n{'='*60}")
    print("Migration Complete")
    print(f"{'='*60}")
    print(f"  Backup: {backup_path}")
    print(f"  Migrated: {output_path}")
    print(f"  Bottleneck: {args.bottleneck_dim}")
    print(f"  Memory tokens: {args.n_tokens}")
    print(f"  Total memory layout: [{args.n_tokens} latent | 4 stoich | 4 heads] = {args.n_tokens + 8} tokens")
    print(f"\nNext steps:")
    print(f"  1. Resume training from {output_path}")
    print(f"  2. The model code must use the V15 bottleneck architecture")
    print(f"     (memory_bottleneck_dim={args.bottleneck_dim}, n_memory_tokens={args.n_tokens})")


if __name__ == '__main__':
    main()
