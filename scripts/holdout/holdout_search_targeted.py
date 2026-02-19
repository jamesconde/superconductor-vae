#!/usr/bin/env python3
"""
Targeted Holdout Search — Element-Anchored Z-Space Exploration
===============================================================

For each holdout formula, find training samples that share the same
elements, encode those as seeds, and explore that specific Z neighborhood.

Uses PCA walks along principal components of neighbor Z-vectors to
generate candidates, plus perturbation and interpolation strategies.

Outputs:
    - JSON file: holdout_search_targeted_{ckpt}.json — per-target results with
      all unique candidate formulas, counts, similarities, Z norms and spreads
    - PT file: holdout_search_targeted_{ckpt}_z_maps.pt — Z centroid (2048-dim)
      for each unique generated formula, keyed by target formula

Usage:
    cd /home/james/superconductor-vae
    PYTHONPATH=src python -u scripts/holdout/holdout_search_targeted.py --all --targets 1
    PYTHONPATH=src python -u scripts/holdout/holdout_search_targeted.py --checkpoint outputs/best_checkpoints/checkpoint_best_V12.38_colab.pt
    PYTHONPATH=src python -u scripts/holdout/holdout_search_targeted.py --checkpoint outputs/best_checkpoints/checkpoint_best_V12.38_colab.pt --all
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from collections import defaultdict

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, IDX_TO_TOKEN, TOKEN_TO_IDX,
    PAD_IDX, START_IDX, END_IDX,
)
from superconductor.data.canonical_ordering import CanonicalOrderer, ElementWithFraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/analysis/ -> scripts/ -> project root
HOLDOUT_PATH = PROJECT_ROOT / 'data' / 'GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json'
CACHE_DIR = PROJECT_ROOT / 'data' / 'processed' / 'cache'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Increased generation budget for targeted search
N_PERTURBATIONS = 100
NOISE_SCALES = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
N_INTERPOLATION_STEPS = 15
N_TEMPERATURE_SAMPLES = 30
TEMPERATURES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Only search for holdouts below this threshold from first run
REDO_THRESHOLD = 0.95

# Tc normalization stats (from data/processed/cache/cache_meta.json)
# Model predicts Tc in z-score normalized log1p space.
# Denormalize: tc_kelvin = expm1(tc_pred * TC_STD + TC_MEAN)
TC_MEAN = 2.725219433789196
TC_STD = 1.3527019896187407

# Family class names (14 classes from HierarchicalFamilyHead composed_14)
FAMILY_NAMES = [
    'NOT_SC', 'BCS_CONVENTIONAL', 'YBCO', 'LSCO', 'BSCCO', 'TBCCO',
    'HBCCO', 'OTHER_CUPRATE', 'PNICTIDE', 'CHALCOGENIDE', 'MGB2',
    'HEAVY_FERMION', 'ORGANIC', 'OTHER_UNKNOWN',
]

# Tc bucket names (5 classes from tc_class_head)
TC_BUCKET_NAMES = ['non-SC (0K)', 'low (0-10K)', 'medium (10-50K)',
                   'high (50-100K)', 'very-high (100K+)']

_CANONICALIZER = CanonicalOrderer()


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


def parse_formula_elements(formula):
    """Extract {element: fraction_value} from formula string."""
    try:
        elements = _CANONICALIZER.parse_formula(formula)
        if not elements:
            return {}
        result = {}
        for ef in elements:
            val = ef.fraction_value
            result[ef.element] = result.get(ef.element, 0) + val
        return result
    except Exception:
        return {}


def element_similarity(formula_a, formula_b):
    """Compositional similarity: Jaccard on elements + fraction overlap."""
    parsed_a = parse_formula_elements(formula_a)
    parsed_b = parse_formula_elements(formula_b)
    if not parsed_a or not parsed_b:
        return 0.0

    all_elements = set(parsed_a.keys()) | set(parsed_b.keys())
    shared = set(parsed_a.keys()) & set(parsed_b.keys())
    if not all_elements:
        return 0.0

    jaccard = len(shared) / len(all_elements)

    if shared:
        total_a = sum(parsed_a.values())
        total_b = sum(parsed_b.values())
        frac_overlap = 0.0
        for elem in shared:
            fa = parsed_a[elem] / max(total_a, 1e-8)
            fb = parsed_b[elem] / max(total_b, 1e-8)
            frac_overlap += min(fa, fb)
        frac_sim = frac_overlap
    else:
        frac_sim = 0.0

    return 0.5 * jaccard + 0.5 * frac_sim


def element_overlap_score(target_elements, candidate_formula_tokens, cache_elem_idx):
    """Score a training sample by how many target elements it contains.

    Args:
        target_elements: set of atomic numbers for the holdout formula
        candidate_formula_tokens: not used (we use elem_idx directly)
        cache_elem_idx: [max_elements] atomic numbers for this training sample

    Returns:
        (n_shared, jaccard) tuple for sorting
    """
    candidate_elements = set(int(z) for z in cache_elem_idx if z > 0)
    shared = target_elements & candidate_elements
    all_elem = target_elements | candidate_elements
    if not all_elem:
        return (0, 0.0)
    return (len(shared), len(shared) / len(all_elem))


def slerp(z1, z2, t):
    """Spherical linear interpolation."""
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    omega = torch.acos(torch.clamp(
        (z1_norm * z2_norm).sum(dim=-1, keepdim=True), -1.0, 1.0
    ))
    omega = omega.clamp(min=1e-6)
    sin_omega = torch.sin(omega)
    if sin_omega.abs().min() < 1e-6:
        return (1 - t) * z1 + t * z2
    s1 = torch.sin((1 - t) * omega) / sin_omega
    s2 = torch.sin(t * omega) / sin_omega
    mag1 = z1.norm(dim=-1, keepdim=True)
    mag2 = z2.norm(dim=-1, keepdim=True)
    mag = (1 - t) * mag1 + t * mag2
    return (s1 * z1_norm + s2 * z2_norm) * mag


# Element symbol → atomic number mapping
ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
    'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
}


def load_models(checkpoint_path):
    """Load encoder and decoder from checkpoint."""
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

    # Detect numden_head architecture from checkpoint weights
    # Current code has V12.41 expanded numden_head (z→512→256→24) but older
    # checkpoints may have the original (z→128→24). Detect and adapt.
    numden_first_key = 'numden_head.0.weight'
    old_numden_arch = False
    if numden_first_key in enc_state:
        numden_first_dim = enc_state[numden_first_key].shape[0]
        if numden_first_dim == 128:
            old_numden_arch = True
            print(f"  Detected OLD numden_head architecture (128-dim first layer)")

    encoder = FullMaterialsVAE(
        n_elements=118, element_embed_dim=128, n_attention_heads=8,
        magpie_dim=magpie_dim, fusion_dim=256, encoder_hidden=[512, 256],
        latent_dim=2048, decoder_hidden=[256, 512], dropout=0.1
    ).to(DEVICE)

    # Replace numden_head with old architecture if checkpoint uses it
    if old_numden_arch:
        import torch.nn as nn
        encoder.numden_head = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, encoder.max_elements * 2),
        ).to(DEVICE)
        print(f"  Replaced numden_head with old architecture (2048→128→24)")

    encoder.load_state_dict(enc_state, strict=False)

    dec_state_raw = checkpoint.get('decoder_state_dict', {})
    dec_state = {k.replace('_orig_mod.', ''): v for k, v in dec_state_raw.items()}

    # Detect stoich_input_dim from checkpoint weights
    stoich_weight_key = 'stoich_to_memory.0.weight'
    if stoich_weight_key in dec_state:
        stoich_dim = dec_state[stoich_weight_key].shape[1]
    else:
        stoich_dim = 37  # Default: V12.38 format
    max_elements = 12

    # Detect vocab_size from checkpoint (V13.0 stores it as metadata)
    dec_vocab_size = checkpoint.get('tokenizer_vocab_size', None)
    if dec_vocab_size is None:
        # Fall back to detecting from embedding weight shape
        embed_key = 'token_embedding.weight'
        if embed_key in dec_state:
            dec_vocab_size = dec_state[embed_key].shape[0]

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
        use_skip_connection=True, use_stoich_conditioning=True,
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
    return encoder, decoder, magpie_dim, has_numden_head


def load_data(magpie_dim):
    """Load cached tensors."""
    data = {
        'elem_idx': torch.load(CACHE_DIR / 'element_indices.pt', map_location='cpu', weights_only=True),
        'elem_frac': torch.load(CACHE_DIR / 'element_fractions.pt', map_location='cpu', weights_only=True),
        'elem_mask': torch.load(CACHE_DIR / 'element_mask.pt', map_location='cpu', weights_only=True),
        'tc': torch.load(CACHE_DIR / 'tc_tensor.pt', map_location='cpu', weights_only=True),
        'magpie': torch.load(CACHE_DIR / 'magpie_tensor.pt', map_location='cpu', weights_only=True),
        'is_sc': torch.load(CACHE_DIR / 'is_sc_tensor.pt', map_location='cpu', weights_only=True),
        'tokens': torch.load(CACHE_DIR / 'formula_tokens.pt', map_location='cpu', weights_only=True),
    }
    if data['magpie'].shape[1] > magpie_dim:
        data['magpie'] = data['magpie'][:, :magpie_dim]
    meta = json.load(open(CACHE_DIR / 'cache_meta.json'))
    data['train_indices'] = meta.get('train_indices', list(range(len(data['elem_idx']))))
    print(f"  {len(data['elem_idx'])} total samples, {len(data['train_indices'])} train")
    return data


@torch.no_grad()
def encode_indices(encoder, data, indices):
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
def decode_z_batch(encoder, decoder, z_batch, has_numden_head=False, temperature=0.01):
    """Decode Z vectors → formula strings with encoder conditioning."""
    batch_size = 64
    all_formulas = []
    for start in range(0, len(z_batch), batch_size):
        z = z_batch[start:start + batch_size].to(DEVICE)
        dec_out = encoder.decode(z)
        encoder_skip = dec_out['attended_input']

        # V12.38: Assemble stoich_pred with numden conditioning
        fraction_output = encoder.fraction_head(z)
        fraction_pred = fraction_output[:, :encoder.max_elements]
        element_count_pred = fraction_output[:, -1]

        if has_numden_head and hasattr(encoder, 'numden_head'):
            numden_pred = encoder.numden_head(z)
            stoich_pred = torch.cat([fraction_pred, numden_pred, element_count_pred.unsqueeze(-1)], dim=-1)
        else:
            stoich_pred = torch.cat([fraction_pred, element_count_pred.unsqueeze(-1)], dim=-1)

        generated, _, _ = decoder.generate_with_kv_cache(
            z=z, encoder_skip=encoder_skip, stoich_pred=stoich_pred,
            temperature=temperature,
        )
        for i in range(len(z)):
            all_formulas.append(tokens_to_formula(generated[i]))
    return all_formulas


def find_element_neighbors(target_formula, data, top_k=50):
    """Find training samples sharing the most elements with the target.

    Scores by: (n_shared_elements, jaccard_similarity) so samples with
    ALL target elements rank highest.

    Returns list of dataset indices sorted by element overlap.
    """
    # Parse target elements
    parsed = parse_formula_elements(target_formula)
    if not parsed:
        print(f"    WARNING: Could not parse target formula: {target_formula}")
        return []

    target_atomic_nums = set()
    for elem in parsed.keys():
        z = ELEMENT_TO_Z.get(elem)
        if z:
            target_atomic_nums.add(z)

    print(f"    Target elements: {set(parsed.keys())} → atomic nums: {target_atomic_nums}")

    # Score all training samples
    scores = []
    for i in data['train_indices']:
        n_shared, jaccard = element_overlap_score(
            target_atomic_nums, None, data['elem_idx'][i]
        )
        if n_shared > 0:
            scores.append((i, n_shared, jaccard))

    # Sort by (n_shared desc, jaccard desc)
    scores.sort(key=lambda x: (-x[1], -x[2]))

    if scores:
        best = scores[0]
        best_formula = tokens_to_formula(data['tokens'][best[0]])
        print(f"    Best neighbor: idx={best[0]}, shared={best[1]}, jaccard={best[2]:.3f}, formula={best_formula}")
        print(f"    Total neighbors with overlap: {len(scores)}")

    return [s[0] for s in scores[:top_k]]


def search_single_target(encoder, decoder, data, target_formula, target_tc, target_family, has_numden_head=False):
    """Targeted search for a single holdout formula.

    Strategy:
    1. Find training samples sharing elements with this target
    2. Encode those as seeds
    3. Apply perturbation, interpolation, temperature sampling
    4. Compare generated formulas against this specific target
    """
    print(f"\n  TARGET: {target_formula} (Tc={target_tc}K, {target_family})")

    # Step 1: Find element-matched neighbors
    neighbor_indices = find_element_neighbors(target_formula, data, top_k=100)

    if len(neighbor_indices) < 3:
        print(f"    ERROR: Only {len(neighbor_indices)} element neighbors found")
        return {
            'target': target_formula, 'target_tc': target_tc, 'target_family': target_family,
            'best_sim': 0.0, 'best_gen': '', 'n_unique': 0, 'n_total': 0,
            'n_neighbors': len(neighbor_indices), 'exact': False,
            'top_matches': [], 'top_frequent': [],
            'all_candidates': [], 'formula_to_z': {},
        }

    # Step 2: Encode neighbors
    print(f"    Encoding {len(neighbor_indices)} element neighbors...")
    z_neighbors = encode_indices(encoder, data, neighbor_indices)
    print(f"    Z shape: {z_neighbors.shape}, norm: {z_neighbors.norm(dim=-1).mean():.2f}")

    # Use top 30 as seeds (best element overlap)
    z_seeds = z_neighbors[:min(30, len(z_neighbors))]

    # Step 3: Generate candidates
    all_candidates_z = []

    # Strategy 1: Fine-grained perturbation (more scales, more samples)
    for z in z_seeds:
        z_exp = z.unsqueeze(0)
        for scale in NOISE_SCALES:
            noise = torch.randn(N_PERTURBATIONS, z.shape[0]) * scale
            all_candidates_z.append(z_exp + noise)

    # Strategy 2: Pairwise interpolation (linear + slerp)
    n_seeds = len(z_seeds)
    max_pairs = min(n_seeds * (n_seeds - 1) // 2, 100)
    pair_count = 0
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            if pair_count >= max_pairs:
                break
            z1 = z_seeds[i].unsqueeze(0)
            z2 = z_seeds[j].unsqueeze(0)
            for t in np.linspace(0.05, 0.95, N_INTERPOLATION_STEPS):
                all_candidates_z.append((1 - t) * z1 + t * z2)
                all_candidates_z.append(slerp(z1, z2, float(t)))
            pair_count += 1
        if pair_count >= max_pairs:
            break

    # Strategy 3: Centroid + PCA walks
    centroid = z_seeds.mean(dim=0, keepdim=True)
    std = z_seeds.std(dim=0, keepdim=True).clamp(min=1e-6)
    for scale in [0.3, 0.5, 1.0, 1.5, 2.0]:
        directions = torch.randn(30, z_seeds.shape[1])
        directions = F.normalize(directions, dim=-1)
        all_candidates_z.append(centroid + scale * std * directions)

    # PCA of neighbor distribution
    if len(z_neighbors) >= 10:
        z_np = z_neighbors.numpy()
        mean = z_np.mean(axis=0)
        centered = z_np - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        n_comp = min(20, len(S))
        for c in range(n_comp):
            direction = torch.from_numpy(Vt[c]).float()
            std_along = S[c] / np.sqrt(len(z_neighbors) - 1)
            for alpha in np.linspace(-3.0, 3.0, 20):
                z_walked = torch.from_numpy(mean).float().unsqueeze(0) + alpha * std_along * direction.unsqueeze(0)
                all_candidates_z.append(z_walked)

    all_z = torch.cat(all_candidates_z, dim=0)
    print(f"    Total Z candidates: {len(all_z)}")

    # Step 4: Decode — greedy
    t0 = time.time()
    greedy_formulas = decode_z_batch(encoder, decoder, all_z, has_numden_head=has_numden_head, temperature=0.01)
    print(f"    Greedy decoded in {time.time()-t0:.1f}s")

    # Step 5: Temperature sampling from seeds (track Z vectors too)
    temp_z_list = []
    temp_formulas = []
    for temp in TEMPERATURES:
        for z_idx in range(min(len(z_seeds), 15)):
            z_repeated = z_seeds[z_idx].unsqueeze(0).repeat(N_TEMPERATURE_SAMPLES, 1)
            temp_z_list.append(z_repeated)
            temp_formulas.extend(decode_z_batch(encoder, decoder, z_repeated, has_numden_head=has_numden_head, temperature=temp))
    temp_z = torch.cat(temp_z_list, dim=0)

    # Combined Z vectors and formulas (greedy + temperature)
    total_z = torch.cat([all_z, temp_z], dim=0)
    all_generated = greedy_formulas + temp_formulas
    unique_formulas = set(f for f in all_generated if f and len(f) > 1)
    print(f"    Total: {len(all_generated)}, unique: {len(unique_formulas)}")

    # Step 6: Build per-formula Z mapping (running sum to save memory)
    formula_z_data = defaultdict(lambda: {'count': 0, 'z_sum': None, 'z_sq_sum': None})
    for i, formula in enumerate(all_generated):
        if not formula or len(formula) <= 1:
            continue
        entry = formula_z_data[formula]
        entry['count'] += 1
        z_i = total_z[i]
        if entry['z_sum'] is None:
            entry['z_sum'] = z_i.clone()
            entry['z_sq_sum'] = (z_i ** 2).clone()
        else:
            entry['z_sum'] += z_i
            entry['z_sq_sum'] += z_i ** 2

    # Compute centroids, spread, and similarity for each unique formula
    formula_to_z = {}
    all_candidates = []
    best_sim = 0.0
    best_gen = ''
    top_matches = []

    for formula, info in formula_z_data.items():
        z_centroid = info['z_sum'] / info['count']
        z_var = info['z_sq_sum'] / info['count'] - z_centroid ** 2
        z_std_mean = z_var.clamp(min=0).sqrt().mean().item()
        sim = element_similarity(formula, target_formula)

        formula_to_z[formula] = z_centroid  # 2048-dim tensor

        all_candidates.append({
            'formula': formula,
            'count': info['count'],
            'similarity': float(sim),
            'z_norm': float(z_centroid.norm()),
            'z_spread': float(z_std_mean),
        })

        if sim > best_sim:
            best_sim = sim
            best_gen = formula
        if sim >= 0.8:
            top_matches.append((formula, sim))

    all_candidates.sort(key=lambda x: -x['similarity'])
    top_matches.sort(key=lambda x: -x[1])

    # Check exact
    target_norm = target_formula.strip()
    exact = any(f.strip() == target_norm for f in unique_formulas)

    print(f"    RESULT: best_sim={best_sim:.4f}, exact={'YES' if exact else 'no'}")
    if top_matches:
        print(f"    Top matches (>0.8):")
        for f, s in top_matches[:5]:
            print(f"      {s:.4f}: {f}")

    # Top generated formulas by frequency
    top_freq = sorted([(c['formula'], c['count']) for c in all_candidates], key=lambda x: -x[1])[:10]
    print(f"    Most frequent:")
    for f, c in top_freq:
        sim = element_similarity(f, target_formula)
        print(f"      {c:4d}x (sim={sim:.3f}): {f}")

    return {
        'target': target_formula,
        'target_tc': target_tc,
        'target_family': target_family,
        'exact': exact,
        'best_sim': float(best_sim),
        'best_gen': best_gen,
        'n_unique': len(unique_formulas),
        'n_total': len(all_generated),
        'n_neighbors': len(neighbor_indices),
        'top_matches': [(f, float(s)) for f, s in top_matches[:10]],
        'top_frequent': [(f, c) for f, c in top_freq],
        'all_candidates': all_candidates,
        'formula_to_z': formula_to_z,
    }


def run_consistency_check(encoder, z_maps, results):
    """
    Run Z centroids through all model heads and check self-consistency.

    For each target's Z centroids, predict: Tc, SC probability, family, Tc bucket.
    Flag inconsistencies:
      - SC prob > 0.5 but Tc <= 0, or SC prob < 0.5 but Tc > 5K
      - SC prob > 0.8 but family = NOT_SC, or SC prob < 0.5 but family != NOT_SC
      - Tc and Tc bucket disagree by more than 1 bucket
    """
    print("\n" + "=" * 70)
    print("SELF-CONSISTENCY CHECK (all heads)")
    print("=" * 70)

    # Build target metadata lookup
    target_meta = {}
    for r in results:
        target_meta[r['target']] = r

    global_stats = {
        'total': 0, 'sc_tc_mismatch': 0, 'sc_family_mismatch': 0,
        'tc_bucket_mismatch': 0, 'pred_sc': 0, 'pred_nonsc': 0,
    }
    consistency_results = {}

    for target_formula, formula_z_map in sorted(z_maps.items()):
        if not formula_z_map:
            continue

        meta = target_meta.get(target_formula, {})
        target_tc = meta.get('target_tc', 0)
        target_family = meta.get('target_family', 'Other')

        formulas = list(formula_z_map.keys())
        z_centroids = torch.stack([formula_z_map[f] for f in formulas]).to(DEVICE)

        # Run all heads in batches
        batch_size = 512
        all_tc, all_sc_prob, all_family, all_tc_bucket = [], [], [], []

        for i in range(0, len(formulas), batch_size):
            z_batch = z_centroids[i:i+batch_size]
            with torch.no_grad():
                # Decode pathway
                dec_out = encoder.decode(z_batch)
                tc_norm = dec_out['tc_pred']
                tc_kelvin = torch.expm1(tc_norm * TC_STD + TC_MEAN)

                # Tc bucket
                tc_bucket = dec_out['tc_class_logits'].argmax(-1)

                # Auxiliary heads
                competence = encoder.competence_head(z_batch).squeeze(-1)
                fraction_output = encoder.fraction_head(z_batch)
                fraction_pred = fraction_output[:, :encoder.max_elements]
                element_count_pred = fraction_output[:, -1]
                numden_pred = encoder.numden_head(z_batch)
                hp_pred = encoder.hp_head(z_batch).squeeze(-1)
                tc_class_logits = dec_out['tc_class_logits']

                # SC head (cross-head consistency input)
                sc_input = torch.cat([
                    z_batch,
                    dec_out['tc_pred'].unsqueeze(-1),
                    dec_out['magpie_pred'],
                    hp_pred.unsqueeze(-1),
                    fraction_pred,
                    element_count_pred.unsqueeze(-1),
                    competence.unsqueeze(-1),
                    tc_class_logits,
                ], dim=-1)
                sc_logit = encoder.sc_head(sc_input).squeeze(-1)
                sc_prob = torch.sigmoid(sc_logit)

                # Family
                family_out = encoder.hierarchical_family_head(
                    dec_out['backbone_h'], sc_logit.detach()
                )
                family_pred = family_out['composed_14'].argmax(-1)

            all_tc.append(tc_kelvin.cpu())
            all_sc_prob.append(sc_prob.cpu())
            all_family.append(family_pred.cpu())
            all_tc_bucket.append(tc_bucket.cpu())

        tc_vals = torch.cat(all_tc).numpy()
        sc_probs = torch.cat(all_sc_prob).numpy()
        family_vals = torch.cat(all_family).numpy()
        tc_buckets = torch.cat(all_tc_bucket).numpy()

        # Check consistency rules
        n = len(formulas)
        issues = {'sc_tc': 0, 'sc_family': 0, 'tc_bucket': 0}

        for j in range(n):
            tc = tc_vals[j]
            sc_p = sc_probs[j]
            fam = family_vals[j]
            bucket = tc_buckets[j]

            # SC↔Tc
            if (sc_p < 0.5 and tc > 5.0) or (sc_p > 0.8 and tc <= 0.0):
                issues['sc_tc'] += 1

            # SC↔Family
            if (sc_p < 0.5 and fam != 0) or (sc_p > 0.8 and fam == 0):
                issues['sc_family'] += 1

            # Tc↔Bucket (expected bucket from tc value)
            if tc <= 0:
                exp_bucket = 0
            elif tc <= 10:
                exp_bucket = 1
            elif tc <= 50:
                exp_bucket = 2
            elif tc <= 100:
                exp_bucket = 3
            else:
                exp_bucket = 4
            if abs(exp_bucket - bucket) > 1:
                issues['tc_bucket'] += 1

        n_sc = int((sc_probs > 0.5).sum())
        n_nonsc = n - n_sc
        global_stats['total'] += n
        global_stats['sc_tc_mismatch'] += issues['sc_tc']
        global_stats['sc_family_mismatch'] += issues['sc_family']
        global_stats['tc_bucket_mismatch'] += issues['tc_bucket']
        global_stats['pred_sc'] += n_sc
        global_stats['pred_nonsc'] += n_nonsc

        total_issues = issues['sc_tc'] + issues['sc_family'] + issues['tc_bucket']
        flag = " !!" if total_issues > 0 else " ok"

        # Get best match predictions
        best_gen = meta.get('best_gen', '')
        best_info = {}
        if best_gen in formulas:
            idx = formulas.index(best_gen)
            best_info = {
                'tc_pred_K': round(float(tc_vals[idx]), 1),
                'sc_prob': round(float(sc_probs[idx]), 3),
                'family': FAMILY_NAMES[family_vals[idx]],
                'tc_bucket': TC_BUCKET_NAMES[tc_buckets[idx]],
            }

        consistency_results[target_formula] = {
            'n_formulas': n,
            'n_sc': n_sc,
            'n_nonsc': n_nonsc,
            'mean_tc_K': round(float(tc_vals.mean()), 1),
            'sc_tc_mismatch': issues['sc_tc'],
            'sc_family_mismatch': issues['sc_family'],
            'tc_bucket_mismatch': issues['tc_bucket'],
            'best_match_preds': best_info,
        }

        print(f"  [{target_family:15s}] {target_formula[:50]:50s} | "
              f"SC:{n_sc:5d} nonSC:{n_nonsc:4d} | "
              f"mean_tc={tc_vals.mean():6.1f}K | "
              f"mismatches: sc_tc={issues['sc_tc']}, sc_fam={issues['sc_family']}, "
              f"tc_bkt={issues['tc_bucket']}{flag}")

    # Summary
    total = global_stats['total']
    print(f"\n{'='*70}")
    print(f"CONSISTENCY SUMMARY ({total:,} unique formulas)")
    print(f"{'='*70}")
    print(f"  SC predictions:   {global_stats['pred_sc']:,} ({global_stats['pred_sc']/total*100:.1f}%)")
    print(f"  Non-SC predictions: {global_stats['pred_nonsc']:,} ({global_stats['pred_nonsc']/total*100:.1f}%)")
    total_mismatches = (global_stats['sc_tc_mismatch'] +
                        global_stats['sc_family_mismatch'] +
                        global_stats['tc_bucket_mismatch'])
    print(f"  SC↔Tc mismatches:    {global_stats['sc_tc_mismatch']:,} "
          f"({global_stats['sc_tc_mismatch']/total*100:.2f}%)")
    print(f"  SC↔Family mismatches: {global_stats['sc_family_mismatch']:,} "
          f"({global_stats['sc_family_mismatch']/total*100:.2f}%)")
    print(f"  Tc↔Bucket mismatches: {global_stats['tc_bucket_mismatch']:,} "
          f"({global_stats['tc_bucket_mismatch']/total*100:.2f}%)")
    consistency_rate = (1 - total_mismatches / (total * 3)) * 100  # 3 checks per formula
    print(f"  Overall consistency: {consistency_rate:.2f}%")

    return consistency_results


def main():
    parser = argparse.ArgumentParser(description='Targeted Holdout Search (Element-Anchored)')
    parser.add_argument('--checkpoint', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'checkpoint_best.pt'),
                        help='Path to checkpoint file')
    parser.add_argument('--all', action='store_true',
                        help='Search all 45 holdouts (not just previous failures)')
    parser.add_argument('--targets', type=int, default=0,
                        help='Only search first N targets (0=all)')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    ckpt_stem = checkpoint_path.stem
    output_path = PROJECT_ROOT / 'outputs' / f'holdout_search_targeted_{ckpt_stem}.json'

    print("=" * 70)
    print("TARGETED HOLDOUT SEARCH (Element-Anchored)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_path}")

    encoder, decoder, magpie_dim, has_numden_head = load_models(checkpoint_path)
    data = load_data(magpie_dim)

    # Load holdout
    with open(HOLDOUT_PATH) as f:
        holdout = json.load(f)

    # Build target list
    if args.all:
        # Search all 45 holdouts
        targets_to_search = []
        for sample in holdout['holdout_samples']:
            targets_to_search.append({
                'formula': sample['formula'],
                'Tc': sample['Tc'],
                'family': sample['family'],
                'prev_best': 0.0,
            })
        print(f"\nSearching ALL {len(targets_to_search)} holdout targets")
    else:
        # Load previous results to know which targets to redo
        prev_path = PROJECT_ROOT / 'scratch' / 'holdout_search_results.json'
        prev_best = {}
        if prev_path.exists():
            with open(prev_path) as f:
                prev_data = json.load(f)
            for fam, res in prev_data['results'].items():
                for gen, target in res['exact_matches']:
                    prev_best[target] = 1.0
                for target, matches in res['fuzzy_matches'].items():
                    if matches:
                        prev_best[target] = max(prev_best.get(target, 0), matches[0][1])

        targets_to_search = []
        for sample in holdout['holdout_samples']:
            formula = sample['formula']
            prev_sim = prev_best.get(formula, 0.0)
            if prev_sim < REDO_THRESHOLD:
                targets_to_search.append({
                    'formula': formula,
                    'Tc': sample['Tc'],
                    'family': sample['family'],
                    'prev_best': prev_sim,
                })

        print(f"\nTargets below {REDO_THRESHOLD} threshold: {len(targets_to_search)}/45")

    # Limit number of targets if --targets N specified
    if args.targets > 0:
        targets_to_search = targets_to_search[:args.targets]
        print(f"\n  --targets {args.targets}: limiting to first {len(targets_to_search)} targets")

    for t in targets_to_search:
        print(f"  [{t['family']:12s}] prev={t['prev_best']:.3f} | {t['formula']}")

    # Search each target
    results = []
    for target in targets_to_search:
        result = search_single_target(
            encoder, decoder, data,
            target['formula'], target['Tc'], target['family'],
            has_numden_head=has_numden_head,
        )
        result['prev_best'] = target['prev_best']
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("TARGETED SEARCH SUMMARY")
    print("=" * 70)

    found_count = 0
    for r in results:
        found_flag = r['best_sim'] >= 0.95
        if found_flag:
            found_count += 1
        marker = '***' if r.get('exact') else ' + ' if found_flag else '   '
        print(f"{marker} [{r['target_family']:12s}] "
              f"best={r['best_sim']:.3f} | {r['target']}")
        if r['best_gen']:
            print(f"     Best: {r['best_gen']}")

    print(f"\nFound (>=0.95): {found_count}/{len(results)}")
    print(f"Exact matches: {sum(1 for r in results if r.get('exact'))}/{len(results)}")

    thresholds = [1.0, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80]
    print(f"\nRESULTS BY THRESHOLD:")
    for thresh in thresholds:
        found = sum(1 for r in results if r['best_sim'] >= thresh)
        print(f"  >= {thresh:.2f}: {found}/{len(results)} ({found/len(results)*100:.1f}%)")

    # Extract Z maps from results before JSON serialization
    z_maps = {}
    for result in results:
        z_maps[result['target']] = result.pop('formula_to_z')

    # Run self-consistency check across all model heads
    consistency_results = run_consistency_check(encoder, z_maps, results)

    # Save JSON (without Z tensors — they're too large for JSON)
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoint': str(checkpoint_path),
        'search_mode': 'all' if args.all else 'failures_only',
        'n_targets_searched': len(targets_to_search),
        'n_found_095': found_count,
        'n_exact': sum(1 for r in results if r.get('exact')),
        'consistency': consistency_results,
        'results': results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Save Z centroid maps as companion .pt file
    z_map_path = output_path.with_name(output_path.stem + '_z_maps.pt')
    torch.save(z_maps, z_map_path)
    print(f"Z maps saved to: {z_map_path}")


if __name__ == '__main__':
    main()
