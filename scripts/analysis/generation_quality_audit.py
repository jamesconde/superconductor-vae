#!/usr/bin/env python3
"""
Generation Quality Audit — Comprehensive
==========================================

Answers: "How many clearly wrong samples are being generated?"

Strategy: Generate a large representative sample from the model by:
1. Encoding random TRAINING samples → Z → decode (roundtrip quality)
2. Perturbing training Z with noise → decode (perturbation quality)
3. Random Z vectors from prior → decode (unconditional quality)
4. Interpolations between training Z pairs → decode (interpolation quality)

Classifies EVERY generated formula as:
- VALID: parseable, all real elements, reasonable stoichiometry, has metal
- MALFORMED: unparseable, truncated, unbalanced parens, unknown tokens
- DUBIOUS: parseable but chemically weird (single element, no metal, duplicate, >12 elements)
- EMPTY: blank or trivially short

Usage:
    cd /home/james/superconductor-vae
    PYTHONPATH=src python -u scratch/generation_quality_audit.py
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, IDX_TO_TOKEN, TOKEN_TO_IDX,
    PAD_IDX, START_IDX, END_IDX,
)
from superconductor.data.canonical_ordering import CanonicalOrderer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / 'outputs' / 'checkpoint_best.pt'
CACHE_DIR = PROJECT_ROOT / 'data' / 'processed' / 'cache'
OUTPUT_PATH = PROJECT_ROOT / 'scratch' / 'generation_quality_audit_results.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Audit config — how many samples per method
N_ROUNDTRIP = 2000       # Encode training → decode
N_PERTURBED = 2000       # Perturb training Z → decode (500 seeds × 4 noise levels)
N_RANDOM_Z = 1000        # Random Z from N(0,1) → decode
N_INTERPOLATIONS = 500   # Interpolation between training Z pairs

_CANONICALIZER = CanonicalOrderer()

# Known elements
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

# Metals
METALS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
    'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
    'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U',
}


def classify_formula(formula):
    """Classify a generated formula.

    Returns: (category, reason)
    Categories: 'valid', 'malformed', 'dubious', 'empty'
    """
    if not formula or len(formula.strip()) == 0:
        return 'empty', 'empty string'

    f = formula.strip()

    if len(f) < 2:
        return 'malformed', 'too short'

    if '?' in f:
        return 'malformed', 'unknown token'

    # Truncation
    if f.endswith('(') or f.endswith('/') or f.endswith('(0') or f.endswith('(1'):
        return 'malformed', 'truncated'

    if f.count('(') != f.count(')'):
        return 'malformed', 'unbalanced parens'

    if '((' in f or '))' in f:
        return 'malformed', 'nested/double parens'

    # Try to parse
    try:
        elements = _CANONICALIZER.parse_formula(f)
    except Exception as e:
        return 'malformed', f'parse error: {str(e)[:60]}'

    if not elements:
        return 'malformed', 'parsed to empty'

    # Check element validity
    elem_symbols = [e.element for e in elements]
    for sym in elem_symbols:
        if sym not in VALID_ELEMENTS:
            return 'malformed', f'unknown element: {sym}'

    # Check fractions
    for e in elements:
        try:
            fv = e.fraction_value
        except (ZeroDivisionError, Exception):
            return 'malformed', f'bad fraction for {e.element}'
        if fv is None:
            return 'malformed', f'no fraction for {e.element}'
        if fv <= 0:
            return 'dubious', f'non-positive fraction: {e.element}={fv}'
        if fv > 100:
            return 'dubious', f'unreasonable fraction: {e.element}={fv}'

    # Chemical plausibility
    n_unique = len(set(elem_symbols))

    if n_unique < 2:
        return 'dubious', 'single element'

    if n_unique > 12:
        return 'dubious', f'too many elements ({n_unique})'

    has_metal = any(sym in METALS for sym in elem_symbols)
    if not has_metal:
        return 'dubious', 'no metal'

    # Duplicate element entries (e.g., Cu appearing twice)
    elem_counts = Counter(elem_symbols)
    duplicates = {sym: cnt for sym, cnt in elem_counts.items() if cnt > 1}
    if duplicates:
        return 'dubious', f'duplicate elements: {duplicates}'

    return 'valid', 'ok'


def tokens_to_formula(token_ids):
    """Convert token IDs to formula string."""
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


def load_models():
    """Load encoder and decoder from checkpoint."""
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    enc_state_raw = checkpoint.get('encoder_state_dict', {})
    magpie_dim = 145
    for k, v in enc_state_raw.items():
        if 'magpie_encoder' in k and k.endswith('.weight') and v.dim() == 2:
            magpie_dim = v.shape[1]
            break
    print(f"  magpie_dim={magpie_dim}")

    enc_state = {k.replace('_orig_mod.', ''): v for k, v in enc_state_raw.items()}

    encoder = FullMaterialsVAE(
        n_elements=118, element_embed_dim=128, n_attention_heads=8,
        magpie_dim=magpie_dim, fusion_dim=256, encoder_hidden=[512, 256],
        latent_dim=2048, decoder_hidden=[256, 512], dropout=0.1
    ).to(DEVICE)

    # V12.28: Net2Net weight transfer for old tc_head → new architecture
    has_old_tc_head = any('tc_head.' in k for k in enc_state)
    has_new_tc_proj = any('tc_proj.' in k for k in enc_state)
    if has_old_tc_head and not has_new_tc_proj:
        encoder.upgrade_tc_head_from_checkpoint(enc_state)
        print("  Applied Net2Net weight transfer for Tc head")

    encoder.load_state_dict(enc_state, strict=False)

    dec_state_raw = checkpoint.get('decoder_state_dict', {})
    dec_state = {k.replace('_orig_mod.', ''): v for k, v in dec_state_raw.items()}

    # V13.0: Auto-detect vocab_size and stoich_input_dim from checkpoint
    dec_vocab_size = checkpoint.get('tokenizer_vocab_size', None)
    if dec_vocab_size is None and 'token_embedding.weight' in dec_state:
        dec_vocab_size = dec_state['token_embedding.weight'].shape[0]
    stoich_dim = None
    if 'stoich_to_memory.0.weight' in dec_state:
        stoich_dim = dec_state['stoich_to_memory.0.weight'].shape[1]

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
        max_elements=12, n_stoich_tokens=4,
        vocab_size=dec_vocab_size,
        stoich_input_dim=stoich_dim,
    ).to(DEVICE)
    decoder.load_state_dict(dec_state, strict=False)

    encoder.eval()
    decoder.eval()

    epoch = checkpoint.get('epoch', '?')
    print(f"  Loaded epoch {epoch}")
    return encoder, decoder, magpie_dim, epoch


def load_data(magpie_dim):
    """Load cached tensors."""
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
def encode_batch(encoder, data, indices):
    """Encode specific dataset indices → Z vectors."""
    batch_size = 128
    all_z = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        idx_t = torch.tensor(batch_idx, dtype=torch.long)
        result = encoder.encode(
            data['elem_idx'][idx_t].to(DEVICE),
            data['elem_frac'][idx_t].to(DEVICE),
            data['elem_mask'][idx_t].to(DEVICE),
            data['magpie'][idx_t].to(DEVICE),
            data['tc'][idx_t].to(DEVICE),
        )
        all_z.append(result['z'].cpu())
    return torch.cat(all_z, dim=0)


@torch.no_grad()
def decode_z_batch(encoder, decoder, z_batch, temperature=0.01):
    """Decode Z vectors → formula strings."""
    batch_size = 64
    all_formulas = []
    for start in range(0, len(z_batch), batch_size):
        z = z_batch[start:start + batch_size].to(DEVICE)
        # Build stoich_pred: V12 (37-dim with numden) or V13 (13-dim without)
        fraction_output = encoder.fraction_head(z)
        if hasattr(encoder, 'numden_head'):
            fraction_pred = fraction_output[:, :12]
            element_count_pred = fraction_output[:, 12]
            numden_pred = encoder.numden_head(z)
            stoich_pred = torch.cat([fraction_pred, numden_pred, element_count_pred.unsqueeze(-1)], dim=-1)
        else:
            stoich_pred = fraction_output  # [batch, 13]
        generated, _, _ = decoder.generate_with_kv_cache(
            z=z, stoich_pred=stoich_pred,  # V12: 37-dim, V13+: 13-dim
            temperature=temperature,
        )
        for i in range(len(z)):
            all_formulas.append(tokens_to_formula(generated[i]))
    return all_formulas


def get_original_formula(data, idx):
    """Get original formula string from token IDs."""
    tokens = data['tokens'][idx]
    return tokens_to_formula(tokens)


def report_quality(formulas, label, original_formulas=None):
    """Classify and report quality of a list of formulas.

    If original_formulas is provided, also compute roundtrip exact match rate.
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Total formulas: {len(formulas):,}")
    print(f"{'='*70}")

    categories = defaultdict(list)
    reasons = Counter()

    for i, f in enumerate(formulas):
        cat, reason = classify_formula(f)
        categories[cat].append((i, f))
        reasons[reason] += 1

    total = len(formulas)
    print(f"\n  Quality Distribution:")
    for cat in ['valid', 'dubious', 'malformed', 'empty']:
        count = len(categories[cat])
        pct = count / total * 100 if total > 0 else 0
        print(f"    {cat:<12s}: {count:5d} / {total:5d} ({pct:5.1f}%)")

    # Malformed breakdown
    if categories['malformed']:
        print(f"\n  Malformed reasons:")
        malformed_reasons = Counter()
        for _, f in categories['malformed']:
            _, reason = classify_formula(f)
            malformed_reasons[reason] += 1
        for reason, count in malformed_reasons.most_common(10):
            print(f"    {count:4d}x  {reason}")
        print(f"  Examples:")
        for idx, f in categories['malformed'][:5]:
            _, reason = classify_formula(f)
            print(f"    [{idx:4d}] '{f[:60]}' — {reason}")

    # Dubious breakdown
    if categories['dubious']:
        print(f"\n  Dubious reasons:")
        dubious_reasons = Counter()
        for _, f in categories['dubious']:
            _, reason = classify_formula(f)
            dubious_reasons[reason] += 1
        for reason, count in dubious_reasons.most_common(10):
            print(f"    {count:4d}x  {reason}")
        print(f"  Examples:")
        for idx, f in categories['dubious'][:5]:
            _, reason = classify_formula(f)
            print(f"    [{idx:4d}] '{f[:60]}' — {reason}")

    # Valid examples
    if categories['valid']:
        print(f"\n  Valid examples (first 10):")
        for idx, f in categories['valid'][:10]:
            print(f"    [{idx:4d}] {f}")

    # Roundtrip exact match
    if original_formulas is not None:
        exact = 0
        for i, (gen, orig) in enumerate(zip(formulas, original_formulas)):
            if gen.strip() == orig.strip():
                exact += 1
        print(f"\n  Roundtrip exact match: {exact}/{total} ({exact/total*100:.1f}%)")

    # Uniqueness
    unique = set(f.strip() for f in formulas if f.strip())
    print(f"  Unique formulas: {len(unique):,} / {total:,} ({len(unique)/max(total,1)*100:.1f}%)")

    # Element distribution in valid formulas
    if categories['valid']:
        all_elements = Counter()
        n_elements_dist = Counter()
        for _, f in categories['valid']:
            try:
                elems = _CANONICALIZER.parse_formula(f)
                syms = set(e.element for e in elems)
                for s in syms:
                    all_elements[s] += 1
                n_elements_dist[len(syms)] += 1
            except Exception:
                pass

        print(f"\n  Element count distribution (valid formulas):")
        for n in sorted(n_elements_dist.keys()):
            count = n_elements_dist[n]
            pct = count / len(categories['valid']) * 100
            bar = '#' * int(pct / 2)
            print(f"    {n:2d} elements: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n  Top 20 elements (valid formulas):")
        for elem, count in all_elements.most_common(20):
            print(f"    {elem:3s}: {count:4d}")

    return {
        'total': total,
        'valid': len(categories['valid']),
        'dubious': len(categories['dubious']),
        'malformed': len(categories['malformed']),
        'empty': len(categories['empty']),
        'valid_pct': len(categories['valid']) / max(total, 1) * 100,
        'unique': len(unique),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("COMPREHENSIVE GENERATION QUALITY AUDIT")
    print("How many clearly wrong samples are being generated?")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    encoder, decoder, magpie_dim, epoch = load_models()
    data = load_data(magpie_dim)

    train_indices = data['train_indices']
    rng = np.random.RandomState(42)

    all_results = {}

    # =========================================================================
    # TEST 1: ROUNDTRIP — Encode training samples, decode back
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# TEST 1: ROUNDTRIP (encode training → decode)")
    print(f"# {N_ROUNDTRIP} random training samples")
    print(f"{'#'*70}")

    sample_idx = rng.choice(train_indices, size=min(N_ROUNDTRIP, len(train_indices)), replace=False).tolist()
    print(f"  Encoding {len(sample_idx)} training samples...")
    z_train = encode_batch(encoder, data, sample_idx)
    print(f"  Z shape: {z_train.shape}, norm: {z_train.norm(dim=-1).mean():.2f}")

    print(f"  Decoding (greedy, temp=0.01)...")
    t1 = time.time()
    roundtrip_formulas = decode_z_batch(encoder, decoder, z_train, temperature=0.01)
    print(f"  Decoded in {time.time()-t1:.1f}s")

    # Get original formulas for comparison
    original_formulas = [get_original_formula(data, idx) for idx in sample_idx]

    all_results['roundtrip'] = report_quality(
        roundtrip_formulas, "TEST 1: ROUNDTRIP (training → encode → decode)", original_formulas
    )

    # =========================================================================
    # TEST 2: PERTURBATION — Add noise to training Z
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# TEST 2: PERTURBATION (training Z + noise → decode)")
    print(f"# 500 seeds × 4 noise levels = {N_PERTURBED}")
    print(f"{'#'*70}")

    n_seeds = 500
    noise_scales = [0.05, 0.1, 0.2, 0.5]
    seed_idx = rng.choice(train_indices, size=min(n_seeds, len(train_indices)), replace=False).tolist()
    print(f"  Encoding {len(seed_idx)} seeds...")
    z_seeds = encode_batch(encoder, data, seed_idx)

    all_perturbed_z = []
    all_perturbed_info = []
    for scale in noise_scales:
        noise = torch.randn_like(z_seeds) * scale
        z_noisy = z_seeds + noise
        all_perturbed_z.append(z_noisy)
        for i in range(len(z_seeds)):
            all_perturbed_info.append((seed_idx[i], scale))

    z_perturbed = torch.cat(all_perturbed_z, dim=0)
    print(f"  Total perturbed Z: {z_perturbed.shape[0]}, decoding...")
    t1 = time.time()
    perturbed_formulas = decode_z_batch(encoder, decoder, z_perturbed, temperature=0.01)
    print(f"  Decoded in {time.time()-t1:.1f}s")

    all_results['perturbed'] = report_quality(
        perturbed_formulas, "TEST 2: PERTURBATION (training Z + noise → decode)"
    )

    # Breakdown by noise scale
    print(f"\n  Breakdown by noise scale:")
    for si, scale in enumerate(noise_scales):
        start = si * n_seeds
        end = start + n_seeds
        subset = perturbed_formulas[start:end]
        n_valid = sum(1 for f in subset if classify_formula(f)[0] == 'valid')
        n_malf = sum(1 for f in subset if classify_formula(f)[0] == 'malformed')
        n_dub = sum(1 for f in subset if classify_formula(f)[0] == 'dubious')
        print(f"    noise={scale:.2f}: valid={n_valid}/{n_seeds} ({n_valid/n_seeds*100:.1f}%), "
              f"malformed={n_malf} ({n_malf/n_seeds*100:.1f}%), "
              f"dubious={n_dub} ({n_dub/n_seeds*100:.1f}%)")

    # =========================================================================
    # TEST 3: RANDOM Z — Sample from standard normal prior
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# TEST 3: RANDOM Z (N(0,1) prior → decode)")
    print(f"# {N_RANDOM_Z} random Z vectors")
    print(f"{'#'*70}")

    # Get typical Z norm from training data to scale random Z
    z_norm_mean = z_train.norm(dim=-1).mean().item()
    z_norm_std = z_train.norm(dim=-1).std().item()
    print(f"  Training Z norm: {z_norm_mean:.2f} ± {z_norm_std:.2f}")

    # Method A: Raw N(0,1) — likely wrong scale
    z_random_raw = torch.randn(N_RANDOM_Z // 2, 2048)
    # Method B: Scaled to match training Z norm
    z_random_scaled = torch.randn(N_RANDOM_Z // 2, 2048)
    z_random_scaled = F.normalize(z_random_scaled, dim=-1) * z_norm_mean

    z_random = torch.cat([z_random_raw, z_random_scaled], dim=0)
    print(f"  Random Z: {z_random.shape[0]} ({N_RANDOM_Z//2} raw + {N_RANDOM_Z//2} scaled)")
    print(f"  Decoding...")
    t1 = time.time()
    random_formulas = decode_z_batch(encoder, decoder, z_random, temperature=0.01)
    print(f"  Decoded in {time.time()-t1:.1f}s")

    all_results['random'] = report_quality(
        random_formulas, "TEST 3: RANDOM Z (unconditional generation)"
    )

    # Breakdown: raw vs scaled
    raw_formulas = random_formulas[:N_RANDOM_Z // 2]
    scaled_formulas = random_formulas[N_RANDOM_Z // 2:]
    n_valid_raw = sum(1 for f in raw_formulas if classify_formula(f)[0] == 'valid')
    n_valid_scaled = sum(1 for f in scaled_formulas if classify_formula(f)[0] == 'valid')
    half = N_RANDOM_Z // 2
    print(f"\n  Raw N(0,1): valid={n_valid_raw}/{half} ({n_valid_raw/half*100:.1f}%)")
    print(f"  Scaled to training norm: valid={n_valid_scaled}/{half} ({n_valid_scaled/half*100:.1f}%)")

    # =========================================================================
    # TEST 4: INTERPOLATION — Slerp between training Z pairs
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# TEST 4: INTERPOLATION (slerp between training Z pairs)")
    print(f"# {N_INTERPOLATIONS} interpolated points")
    print(f"{'#'*70}")

    n_pairs = 100
    steps = [0.2, 0.4, 0.6, 0.8, 1.0]  # 5 points per pair = 500 total
    pair_idx1 = rng.choice(train_indices, size=n_pairs, replace=False).tolist()
    pair_idx2 = rng.choice(train_indices, size=n_pairs, replace=False).tolist()

    print(f"  Encoding {n_pairs} × 2 endpoints...")
    z1 = encode_batch(encoder, data, pair_idx1)
    z2 = encode_batch(encoder, data, pair_idx2)

    all_interp_z = []
    for t in steps:
        # Slerp
        z1_n = F.normalize(z1, dim=-1)
        z2_n = F.normalize(z2, dim=-1)
        cos_sim = (z1_n * z2_n).sum(dim=-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(cos_sim).clamp(min=1e-6)
        sin_omega = torch.sin(omega)
        # Fallback to linear if omega too small
        mask = sin_omega.abs() > 1e-6
        s1 = torch.where(mask, torch.sin((1 - t) * omega) / sin_omega, torch.full_like(sin_omega, 1 - t))
        s2 = torch.where(mask, torch.sin(t * omega) / sin_omega, torch.full_like(sin_omega, t))
        mag1 = z1.norm(dim=-1, keepdim=True)
        mag2 = z2.norm(dim=-1, keepdim=True)
        mag = (1 - t) * mag1 + t * mag2
        z_interp = (s1 * z1_n + s2 * z2_n) * mag
        all_interp_z.append(z_interp)

    z_interp_all = torch.cat(all_interp_z, dim=0)
    print(f"  Interpolated Z: {z_interp_all.shape[0]}, decoding...")
    t1 = time.time()
    interp_formulas = decode_z_batch(encoder, decoder, z_interp_all, temperature=0.01)
    print(f"  Decoded in {time.time()-t1:.1f}s")

    all_results['interpolation'] = report_quality(
        interp_formulas, "TEST 4: INTERPOLATION (slerp between training pairs)"
    )

    # =========================================================================
    # TEST 5: TEMPERATURE SAMPLING — Same Z, different temperatures
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# TEST 5: TEMPERATURE SAMPLING (same Z, varying temperature)")
    print(f"# 200 seeds × 5 temperatures = 1000")
    print(f"{'#'*70}")

    n_temp_seeds = 200
    temps = [0.01, 0.1, 0.3, 0.7, 1.0]
    temp_idx = rng.choice(train_indices, size=n_temp_seeds, replace=False).tolist()
    z_temp_seeds = encode_batch(encoder, data, temp_idx)

    all_temp_formulas = {}
    for temp in temps:
        print(f"  Decoding at temp={temp}...")
        t1 = time.time()
        temp_formulas = decode_z_batch(encoder, decoder, z_temp_seeds, temperature=temp)
        print(f"    Decoded in {time.time()-t1:.1f}s")
        all_temp_formulas[temp] = temp_formulas

    print(f"\n  Quality by temperature:")
    print(f"  {'Temp':>6s}  {'Valid':>8s}  {'Malformed':>10s}  {'Dubious':>8s}  {'Empty':>6s}  {'Unique':>8s}")
    print(f"  {'-'*55}")
    for temp in temps:
        formulas = all_temp_formulas[temp]
        n_valid = sum(1 for f in formulas if classify_formula(f)[0] == 'valid')
        n_malf = sum(1 for f in formulas if classify_formula(f)[0] == 'malformed')
        n_dub = sum(1 for f in formulas if classify_formula(f)[0] == 'dubious')
        n_empty = sum(1 for f in formulas if classify_formula(f)[0] == 'empty')
        n_unique = len(set(f.strip() for f in formulas if f.strip()))
        print(f"  {temp:6.2f}  {n_valid:5d}/{n_temp_seeds:3d}  {n_malf:7d}/{n_temp_seeds:3d}  "
              f"{n_dub:5d}/{n_temp_seeds:3d}  {n_empty:3d}/{n_temp_seeds:3d}  {n_unique:5d}/{n_temp_seeds:3d}")

    all_results['temperature'] = {}
    for temp in temps:
        formulas = all_temp_formulas[temp]
        n_valid = sum(1 for f in formulas if classify_formula(f)[0] == 'valid')
        all_results['temperature'][str(temp)] = {
            'total': len(formulas),
            'valid': n_valid,
            'valid_pct': n_valid / len(formulas) * 100,
        }

    # =========================================================================
    # GRAND SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"GRAND SUMMARY")
    print(f"{'='*70}")

    total_gen = 0
    total_valid = 0
    total_malf = 0
    total_dub = 0

    for test_name in ['roundtrip', 'perturbed', 'random', 'interpolation']:
        r = all_results[test_name]
        total_gen += r['total']
        total_valid += r['valid']
        total_malf += r['malformed']
        total_dub += r['dubious']

    pct_valid = total_valid / max(total_gen, 1) * 100
    pct_malf = total_malf / max(total_gen, 1) * 100
    pct_dub = total_dub / max(total_gen, 1) * 100
    pct_wrong = (total_malf + total_dub) / max(total_gen, 1) * 100

    print(f"\n  Across all tests ({total_gen:,} total generations):")
    print(f"    Valid:     {total_valid:5d} ({pct_valid:.1f}%)")
    print(f"    Malformed: {total_malf:5d} ({pct_malf:.1f}%)")
    print(f"    Dubious:   {total_dub:5d} ({pct_dub:.1f}%)")
    print(f"    CLEARLY WRONG (malformed + dubious): {total_malf + total_dub:5d} ({pct_wrong:.1f}%)")
    print(f"    VALID RATE: {pct_valid:.1f}%")

    print(f"\n  Per-test summary:")
    print(f"  {'Test':<25s} {'Total':>6s} {'Valid%':>8s} {'Malf%':>8s} {'Dub%':>8s}")
    print(f"  {'-'*55}")
    for test_name in ['roundtrip', 'perturbed', 'random', 'interpolation']:
        r = all_results[test_name]
        print(f"  {test_name:<25s} {r['total']:6d} {r['valid_pct']:7.1f}% "
              f"{r['malformed']/r['total']*100:7.1f}% {r['dubious']/r['total']*100:7.1f}%")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Checkpoint: epoch {epoch}")

    # Save results
    all_results['grand_total'] = {
        'total': total_gen,
        'valid': total_valid,
        'malformed': total_malf,
        'dubious': total_dub,
        'valid_pct': pct_valid,
        'clearly_wrong_pct': pct_wrong,
        'epoch': epoch,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
