#!/usr/bin/env python3
"""
Full Enchilada Quality Audit
==============================

For the same Z vectors, check formula quality + Tc accuracy + Magpie accuracy
simultaneously. Answers: "When the formula is wrong, is the Tc/Magpie also wrong?"

This directly tests the concern that a single Z space is supposed to drive
accurate reconstruction of ALL properties simultaneously.

Usage:
    cd /home/james/superconductor-vae
    PYTHONPATH=src python -u scripts/evaluate_generation_quality.py
    PYTHONPATH=src python -u scripts/evaluate_generation_quality.py --checkpoint outputs/best_checkpoints/checkpoint_best_V12.38_colab.pt
    PYTHONPATH=src python -u scripts/evaluate_generation_quality.py --checkpoint outputs/checkpoint_best.pt --n-samples 5000
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, IDX_TO_TOKEN,
    PAD_IDX, START_IDX, END_IDX,
)
from superconductor.data.canonical_ordering import CanonicalOrderer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/analysis/ -> scripts/ -> project root
CACHE_DIR = PROJECT_ROOT / 'data' / 'processed' / 'cache'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tc denormalization constants (from cache_meta.json)
TC_MEAN = 2.725219433789196
TC_STD = 1.3527019896187407

_CANONICALIZER = CanonicalOrderer()

VALID_ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am',
}
METALS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
    'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
    'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U',
}


def classify_formula(formula):
    """Classify formula as valid/malformed/dubious/empty."""
    if not formula or len(formula.strip()) == 0:
        return 'empty'
    f = formula.strip()
    if len(f) < 2:
        return 'malformed'
    if '?' in f:
        return 'malformed'
    if f.endswith('(') or f.endswith('/') or f.endswith('(0') or f.endswith('(1'):
        return 'malformed'
    if f.count('(') != f.count(')'):
        return 'malformed'
    if '((' in f or '))' in f:
        return 'malformed'
    try:
        elements = _CANONICALIZER.parse_formula(f)
    except Exception:
        return 'malformed'
    if not elements:
        return 'malformed'
    elem_symbols = [e.element for e in elements]
    for sym in elem_symbols:
        if sym not in VALID_ELEMENTS:
            return 'malformed'
    for e in elements:
        try:
            fv = e.fraction_value
        except Exception:
            return 'malformed'
        if fv is None or fv <= 0:
            return 'dubious'
        if fv > 100:
            return 'dubious'
    n_unique = len(set(elem_symbols))
    if n_unique < 2:
        return 'dubious'
    if n_unique > 12:
        return 'dubious'
    if not any(sym in METALS for sym in elem_symbols):
        return 'dubious'
    if len(elem_symbols) != len(set(elem_symbols)):
        return 'dubious'
    return 'valid'


def tokens_to_formula(token_ids):
    tokens = []
    for tid in token_ids:
        tid = int(tid)
        if tid == PAD_IDX or tid == START_IDX:
            continue
        if tid == END_IDX:
            break
        token = IDX_TO_TOKEN.get(tid, '?')
        tokens.append(token)
    return ''.join(tokens)


def denormalize_tc(tc_norm):
    """Convert normalized Tc back to Kelvin."""
    tc_log = tc_norm * TC_STD + TC_MEAN
    return max(0.0, np.expm1(tc_log))


def load_models(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    enc_state_raw = checkpoint.get('encoder_state_dict', {})
    magpie_dim = 145
    for k, v in enc_state_raw.items():
        if 'magpie_encoder' in k and k.endswith('.weight') and v.dim() == 2:
            magpie_dim = v.shape[1]
            break

    enc_state = {k.replace('_orig_mod.', ''): v for k, v in enc_state_raw.items()}

    # Detect V12.38+ numden_head
    has_numden_head = any('numden_head.' in k for k in enc_state)

    encoder = FullMaterialsVAE(
        n_elements=118, element_embed_dim=128, n_attention_heads=8,
        magpie_dim=magpie_dim, fusion_dim=256, encoder_hidden=[512, 256],
        latent_dim=2048, decoder_hidden=[256, 512], dropout=0.1
    ).to(DEVICE)

    has_old_tc_head = any('tc_head.' in k for k in enc_state)
    has_new_tc_proj = any('tc_proj.' in k for k in enc_state)
    if has_old_tc_head and not has_new_tc_proj:
        encoder.upgrade_tc_head_from_checkpoint(enc_state)
        print("  Applied Net2Net weight transfer for Tc head")

    encoder.load_state_dict(enc_state, strict=False)

    dec_state_raw = checkpoint.get('decoder_state_dict', {})
    dec_state = {k.replace('_orig_mod.', ''): v for k, v in dec_state_raw.items()}

    # Detect stoich_input_dim from checkpoint weights
    stoich_weight_key = 'stoich_to_memory.0.weight'
    if stoich_weight_key in dec_state:
        stoich_dim = dec_state[stoich_weight_key].shape[1]
    else:
        stoich_dim = None
    max_elements = 12

    # V13.0: Auto-detect vocab_size from checkpoint
    dec_vocab_size = checkpoint.get('tokenizer_vocab_size', None)
    if dec_vocab_size is None and 'token_embedding.weight' in dec_state:
        dec_vocab_size = dec_state['token_embedding.weight'].shape[0]

    # Auto-detect decoder architecture from checkpoint (V12.42 wider model support)
    _d_model = checkpoint.get('d_model', None)
    if _d_model is None and 'token_embedding.weight' in dec_state:
        _d_model = dec_state['token_embedding.weight'].shape[1]
    _d_model = _d_model or 512
    _dim_ff = checkpoint.get('dim_feedforward', None)
    if _dim_ff is None and 'transformer_decoder.layers.0.linear1.weight' in dec_state:
        _dim_ff = dec_state['transformer_decoder.layers.0.linear1.weight'].shape[0]
    _dim_ff = _dim_ff or 2048
    _nhead = checkpoint.get('nhead', 8)
    _num_layers = checkpoint.get('num_layers', 12)
    _max_len = checkpoint.get('max_formula_len', 60)

    decoder = EnhancedTransformerDecoder(
        latent_dim=2048, d_model=_d_model, nhead=_nhead, num_layers=_num_layers,
        dim_feedforward=_dim_ff, dropout=0.1, max_len=_max_len,
        n_memory_tokens=16, encoder_skip_dim=256,
        use_skip_connection=False, use_stoich_conditioning=True,  # V13.1: skip removed
        max_elements=max_elements, n_stoich_tokens=4,
        vocab_size=dec_vocab_size,
        stoich_input_dim=stoich_dim,
    ).to(DEVICE)
    decoder.load_state_dict(dec_state, strict=False)

    encoder.eval()
    decoder.eval()

    epoch = checkpoint.get('epoch', '?')
    print(f"  Loaded epoch {epoch}, magpie_dim={magpie_dim}, numden_head={has_numden_head}")
    print(f"  Decoder stoich_input_dim={decoder.stoich_to_memory[0].in_features}")
    return encoder, decoder, magpie_dim, epoch, has_numden_head


def load_data(magpie_dim):
    data = {
        'elem_idx': torch.load(CACHE_DIR / 'element_indices.pt', map_location='cpu', weights_only=True),
        'elem_frac': torch.load(CACHE_DIR / 'element_fractions.pt', map_location='cpu', weights_only=True),
        'elem_mask': torch.load(CACHE_DIR / 'element_mask.pt', map_location='cpu', weights_only=True),
        'tc': torch.load(CACHE_DIR / 'tc_tensor.pt', map_location='cpu', weights_only=True),
        'magpie': torch.load(CACHE_DIR / 'magpie_tensor.pt', map_location='cpu', weights_only=True),
        'tokens': torch.load(CACHE_DIR / 'formula_tokens.pt', map_location='cpu', weights_only=True),
    }
    if data['magpie'].shape[1] > magpie_dim:
        data['magpie'] = data['magpie'][:, :magpie_dim]
    meta = json.load(open(CACHE_DIR / 'cache_meta.json'))
    data['train_indices'] = meta.get('train_indices', list(range(len(data['elem_idx']))))
    print(f"  {len(data['elem_idx'])} total samples, {len(data['train_indices'])} train")
    return data


@torch.no_grad()
def full_enchilada_batch(encoder, decoder, data, indices, has_numden_head, temperature=0.01):
    """Encode -> Z -> decode ALL properties for a batch of dataset indices.

    Returns list of dicts with formula, tc, magpie for each sample.
    """
    batch_size = 64
    all_results = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        idx_t = torch.tensor(batch_idx, dtype=torch.long)

        # Encode
        enc_result = encoder.encode(
            data['elem_idx'][idx_t].to(DEVICE),
            data['elem_frac'][idx_t].to(DEVICE),
            data['elem_mask'][idx_t].to(DEVICE),
            data['magpie'][idx_t].to(DEVICE),
            data['tc'][idx_t].to(DEVICE),
        )
        z = enc_result['z']

        # Decode all properties from Z
        dec_out = encoder.decode(z)
        tc_pred = dec_out['tc_pred'].detach().cpu()           # [batch, 1]
        magpie_pred = dec_out['magpie_pred'].detach().cpu()   # [batch, 145]
        # V12.38: Assemble stoich_pred with numden conditioning
        # fraction_head outputs [batch, max_elements+1] where last dim is element count
        fraction_output = encoder.fraction_head(z)
        fraction_pred = fraction_output[:, :encoder.max_elements]  # [batch, 12]
        element_count_pred = fraction_output[:, -1]                # [batch]

        if has_numden_head and hasattr(encoder, 'numden_head'):
            numden_pred = encoder.numden_head(z)  # [batch, 24]
            stoich_pred = torch.cat([fraction_pred, numden_pred, element_count_pred.unsqueeze(-1)], dim=-1)
        else:
            stoich_pred = torch.cat([fraction_pred, element_count_pred.unsqueeze(-1)], dim=-1)

        # Generate formula (V13.1: no skip connection)
        generated, _, _ = decoder.generate_with_kv_cache(
            z=z, stoich_pred=stoich_pred,
            temperature=temperature,
        )

        # Ground truth
        tc_true = data['tc'][idx_t]           # [batch, 1] normalized
        magpie_true = data['magpie'][idx_t]   # [batch, 145]

        for i in range(len(batch_idx)):
            formula_gen = tokens_to_formula(generated[i])
            formula_orig = tokens_to_formula(data['tokens'][batch_idx[i]])

            tc_pred_norm = tc_pred[i].item()
            tc_true_norm = tc_true[i].item()
            tc_pred_K = denormalize_tc(tc_pred_norm)
            tc_true_K = denormalize_tc(tc_true_norm)

            magpie_mse = F.mse_loss(magpie_pred[i], magpie_true[i]).item()
            magpie_cos = F.cosine_similarity(
                magpie_pred[i].unsqueeze(0), magpie_true[i].unsqueeze(0)
            ).item()

            all_results.append({
                'idx': batch_idx[i],
                'formula_gen': formula_gen,
                'formula_orig': formula_orig,
                'formula_exact': formula_gen.strip() == formula_orig.strip(),
                'formula_quality': classify_formula(formula_gen),
                'tc_pred_K': tc_pred_K,
                'tc_true_K': tc_true_K,
                'tc_error_K': abs(tc_pred_K - tc_true_K),
                'tc_pred_norm': tc_pred_norm,
                'tc_true_norm': tc_true_norm,
                'tc_norm_error': abs(tc_pred_norm - tc_true_norm),
                'magpie_mse': magpie_mse,
                'magpie_cos': magpie_cos,
            })

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Full Enchilada Quality Audit')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoint_best.pt',
                        help='Path to checkpoint (relative to project root or absolute)')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of training samples to evaluate')
    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    t0 = time.time()
    print("=" * 70)
    print("FULL ENCHILADA QUALITY AUDIT")
    print("Same Z -> Formula + Tc + Magpie -- are they correlated?")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"N samples: {args.n_samples}")

    encoder, decoder, magpie_dim, epoch, has_numden_head = load_models(checkpoint_path)
    data = load_data(magpie_dim)

    train_indices = data['train_indices']
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(train_indices, size=min(args.n_samples, len(train_indices)), replace=False).tolist()

    print(f"\nProcessing {len(sample_idx)} training samples (full enchilada)...")
    t1 = time.time()
    results = full_enchilada_batch(encoder, decoder, data, sample_idx, has_numden_head)
    print(f"  Done in {time.time()-t1:.1f}s")

    # =========================================================================
    # OVERALL STATS
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"OVERALL STATS ({len(results)} samples)")
    print(f"{'='*70}")

    # Formula quality
    quality_counts = Counter(r['formula_quality'] for r in results)
    exact_count = sum(1 for r in results if r['formula_exact'])
    print(f"\n  Formula quality:")
    for cat in ['valid', 'dubious', 'malformed', 'empty']:
        count = quality_counts.get(cat, 0)
        print(f"    {cat:<12s}: {count:5d} ({count/len(results)*100:.1f}%)")
    print(f"    Exact match: {exact_count}/{len(results)} ({exact_count/len(results)*100:.1f}%)")

    # Tc stats
    tc_errors = [r['tc_error_K'] for r in results]
    print(f"\n  Tc prediction (Kelvin):")
    print(f"    MAE:    {np.mean(tc_errors):.2f} K")
    print(f"    Median: {np.median(tc_errors):.2f} K")
    print(f"    P95:    {np.percentile(tc_errors, 95):.2f} K")
    print(f"    Max:    {np.max(tc_errors):.2f} K")

    # Magpie stats
    magpie_mses = [r['magpie_mse'] for r in results]
    magpie_coss = [r['magpie_cos'] for r in results]
    print(f"\n  Magpie reconstruction:")
    print(f"    MSE mean:  {np.mean(magpie_mses):.4f}")
    print(f"    MSE median: {np.median(magpie_mses):.4f}")
    print(f"    Cosine sim mean: {np.mean(magpie_coss):.4f}")
    print(f"    Cosine sim median: {np.median(magpie_coss):.4f}")

    # =========================================================================
    # BREAKDOWN BY FORMULA QUALITY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"BREAKDOWN BY FORMULA QUALITY")
    print(f"Does bad formula correlate with bad Tc/Magpie?")
    print(f"{'='*70}")

    print(f"\n  {'Category':<12s} {'Count':>6s}  {'Tc MAE(K)':>10s}  {'Tc Med(K)':>10s}  "
          f"{'Magp MSE':>10s}  {'Magp Cos':>10s}  {'Exact%':>8s}")
    print(f"  {'-'*76}")

    for cat in ['valid', 'dubious', 'malformed', 'empty']:
        subset = [r for r in results if r['formula_quality'] == cat]
        if not subset:
            continue
        tc_errs = [r['tc_error_K'] for r in subset]
        mg_mses = [r['magpie_mse'] for r in subset]
        mg_coss = [r['magpie_cos'] for r in subset]
        n_exact = sum(1 for r in subset if r['formula_exact'])
        print(f"  {cat:<12s} {len(subset):6d}  {np.mean(tc_errs):10.2f}  {np.median(tc_errs):10.2f}  "
              f"{np.mean(mg_mses):10.4f}  {np.mean(mg_coss):10.4f}  "
              f"{n_exact/len(subset)*100:7.1f}%")

    # Also break down exact vs non-exact
    print(f"\n  {'Match':<12s} {'Count':>6s}  {'Tc MAE(K)':>10s}  {'Tc Med(K)':>10s}  "
          f"{'Magp MSE':>10s}  {'Magp Cos':>10s}")
    print(f"  {'-'*68}")
    for label, subset in [
        ('Exact', [r for r in results if r['formula_exact']]),
        ('Non-exact', [r for r in results if not r['formula_exact']]),
    ]:
        if not subset:
            continue
        tc_errs = [r['tc_error_K'] for r in subset]
        mg_mses = [r['magpie_mse'] for r in subset]
        mg_coss = [r['magpie_cos'] for r in subset]
        print(f"  {label:<12s} {len(subset):6d}  {np.mean(tc_errs):10.2f}  {np.median(tc_errs):10.2f}  "
              f"{np.mean(mg_mses):10.4f}  {np.mean(mg_coss):10.4f}")

    # =========================================================================
    # Tc ERROR DISTRIBUTION BY FORMULA QUALITY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Tc ERROR DISTRIBUTION")
    print(f"{'='*70}")

    # Bucket Tc errors
    buckets = [(0, 1, '<1K'), (1, 5, '1-5K'), (5, 20, '5-20K'),
               (20, 50, '20-50K'), (50, 100, '50-100K'), (100, 9999, '>100K')]

    print(f"\n  {'Category':<12s}", end='')
    for _, _, label in buckets:
        print(f"  {label:>8s}", end='')
    print()
    print(f"  {'-'*70}")

    for cat in ['valid', 'dubious', 'malformed']:
        subset = [r for r in results if r['formula_quality'] == cat]
        if not subset:
            continue
        print(f"  {cat:<12s}", end='')
        for lo, hi, _ in buckets:
            count = sum(1 for r in subset if lo <= r['tc_error_K'] < hi)
            pct = count / len(subset) * 100
            print(f"  {pct:7.1f}%", end='')
        print()

    # =========================================================================
    # WORST Tc PREDICTIONS
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"WORST 20 Tc PREDICTIONS (by absolute error)")
    print(f"{'='*70}")

    sorted_by_tc = sorted(results, key=lambda r: -r['tc_error_K'])
    print(f"  {'Idx':>6s}  {'Quality':<10s}  {'Tc_true':>8s}  {'Tc_pred':>8s}  {'Error':>8s}  "
          f"{'Magp_MSE':>8s}  Formula")
    print(f"  {'-'*90}")
    for r in sorted_by_tc[:20]:
        print(f"  {r['idx']:6d}  {r['formula_quality']:<10s}  {r['tc_true_K']:8.1f}  "
              f"{r['tc_pred_K']:8.1f}  {r['tc_error_K']:8.1f}  {r['magpie_mse']:8.4f}  "
              f"{r['formula_gen'][:40]}")

    # =========================================================================
    # WORST MAGPIE PREDICTIONS
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"WORST 20 MAGPIE PREDICTIONS (by MSE)")
    print(f"{'='*70}")

    sorted_by_magpie = sorted(results, key=lambda r: -r['magpie_mse'])
    print(f"  {'Idx':>6s}  {'Quality':<10s}  {'Magp_MSE':>8s}  {'Magp_Cos':>8s}  "
          f"{'Tc_err':>8s}  Formula")
    print(f"  {'-'*80}")
    for r in sorted_by_magpie[:20]:
        print(f"  {r['idx']:6d}  {r['formula_quality']:<10s}  {r['magpie_mse']:8.4f}  "
              f"{r['magpie_cos']:8.4f}  {r['tc_error_K']:8.1f}  "
              f"{r['formula_gen'][:40]}")

    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"CORRELATION: Are formula errors, Tc errors, and Magpie errors correlated?")
    print(f"{'='*70}")

    # Convert formula quality to numeric for correlation
    quality_to_num = {'valid': 0, 'dubious': 1, 'malformed': 2, 'empty': 3}
    quality_nums = np.array([quality_to_num[r['formula_quality']] for r in results])
    tc_errors_arr = np.array([r['tc_error_K'] for r in results])
    magpie_mses_arr = np.array([r['magpie_mse'] for r in results])
    exact_arr = np.array([1.0 if r['formula_exact'] else 0.0 for r in results])

    # Spearman rank correlation (more robust for non-normal distributions)
    from scipy import stats
    rho_tc_magpie, p_tc_magpie = stats.spearmanr(tc_errors_arr, magpie_mses_arr)
    rho_quality_tc, p_quality_tc = stats.spearmanr(quality_nums, tc_errors_arr)
    rho_quality_magpie, p_quality_magpie = stats.spearmanr(quality_nums, magpie_mses_arr)
    rho_exact_tc, p_exact_tc = stats.spearmanr(exact_arr, tc_errors_arr)

    print(f"\n  Spearman rank correlations:")
    print(f"    Tc error <-> Magpie MSE:       rho={rho_tc_magpie:.3f}  (p={p_tc_magpie:.2e})")
    print(f"    Formula quality <-> Tc error:   rho={rho_quality_tc:.3f}  (p={p_quality_tc:.2e})")
    print(f"    Formula quality <-> Magpie MSE: rho={rho_quality_magpie:.3f}  (p={p_quality_magpie:.2e})")
    print(f"    Exact match <-> Tc error:       rho={rho_exact_tc:.3f}  (p={p_exact_tc:.2e})")

    print(f"\n  Interpretation:")
    if abs(rho_quality_tc) < 0.1:
        print(f"    Formula quality and Tc error are WEAKLY correlated -- errors are mostly independent")
    elif abs(rho_quality_tc) < 0.3:
        print(f"    Formula quality and Tc error are MODERATELY correlated -- some shared failure modes")
    else:
        print(f"    Formula quality and Tc error are STRONGLY correlated -- same Z regions fail across the board")

    if abs(rho_tc_magpie) < 0.1:
        print(f"    Tc and Magpie errors are WEAKLY correlated -- different Z regions matter for each")
    elif abs(rho_tc_magpie) < 0.3:
        print(f"    Tc and Magpie errors are MODERATELY correlated")
    else:
        print(f"    Tc and Magpie errors are STRONGLY correlated -- when one fails, both fail")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Checkpoint: {checkpoint_path.name} (epoch {epoch})")
    print(f"  Samples: {len(results)}")
    print(f"  Formula valid: {quality_counts.get('valid', 0)/len(results)*100:.1f}%")
    print(f"  Formula exact: {exact_count/len(results)*100:.1f}%")
    print(f"  Tc MAE: {np.mean(tc_errors):.2f} K")
    print(f"  Magpie cosine sim: {np.mean(magpie_coss):.4f}")
    print(f"  Time: {elapsed:.0f}s")

    # Save
    # Name output after checkpoint for easy comparison
    ckpt_stem = checkpoint_path.stem
    output_path = PROJECT_ROOT / 'outputs' / f'quality_audit_{ckpt_stem}.json'
    summary = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_name': checkpoint_path.name,
        'epoch': epoch,
        'n_samples': len(results),
        'formula_valid_pct': quality_counts.get('valid', 0) / len(results) * 100,
        'formula_exact_pct': exact_count / len(results) * 100,
        'tc_mae_K': float(np.mean(tc_errors)),
        'tc_median_K': float(np.median(tc_errors)),
        'magpie_mse_mean': float(np.mean(magpie_mses)),
        'magpie_cos_mean': float(np.mean(magpie_coss)),
        'correlation_tc_magpie': float(rho_tc_magpie),
        'correlation_quality_tc': float(rho_quality_tc),
        'correlation_quality_magpie': float(rho_quality_magpie),
        'by_quality': {},
    }
    for cat in ['valid', 'dubious', 'malformed']:
        subset = [r for r in results if r['formula_quality'] == cat]
        if subset:
            summary['by_quality'][cat] = {
                'count': len(subset),
                'tc_mae_K': float(np.mean([r['tc_error_K'] for r in subset])),
                'magpie_mse': float(np.mean([r['magpie_mse'] for r in subset])),
                'magpie_cos': float(np.mean([r['magpie_cos'] for r in subset])),
            }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
