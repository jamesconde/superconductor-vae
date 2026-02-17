#!/usr/bin/env python3
"""
Targeted Holdout Search — Element-Anchored Z-Space Exploration
===============================================================

For each missed holdout formula, find training samples that share the same
elements, encode those as seeds, and explore that specific Z neighborhood.

This complements holdout_search.py which used family-level (highest Tc) seeds.

Usage:
    cd /home/james/superconductor-vae
    PYTHONPATH=src python -u scratch/holdout_search_targeted.py
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, IDX_TO_TOKEN, TOKEN_TO_IDX,
    PAD_IDX, START_IDX, END_IDX,
)
from superconductor.data.canonical_ordering import CanonicalOrderer, ElementWithFraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / 'outputs' / 'checkpoint_best.pt'
HOLDOUT_PATH = PROJECT_ROOT / 'data' / 'GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json'
CACHE_DIR = PROJECT_ROOT / 'data' / 'processed' / 'cache'
OUTPUT_PATH = PROJECT_ROOT / 'scratch' / 'holdout_search_targeted_results.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Increased generation budget for targeted search
N_PERTURBATIONS = 100
NOISE_SCALES = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
N_INTERPOLATION_STEPS = 15
N_TEMPERATURE_SAMPLES = 30
TEMPERATURES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Only search for holdouts below this threshold from first run
REDO_THRESHOLD = 0.95

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
    encoder.load_state_dict(enc_state, strict=False)

    dec_state_raw = checkpoint.get('decoder_state_dict', {})
    dec_state = {k.replace('_orig_mod.', ''): v for k, v in dec_state_raw.items()}

    decoder = EnhancedTransformerDecoder(
        latent_dim=2048, d_model=512, nhead=8, num_layers=12,
        dim_feedforward=2048, dropout=0.1, max_len=60,
        n_memory_tokens=16, encoder_skip_dim=256,
        use_skip_connection=True, use_stoich_conditioning=True,
        max_elements=12, n_stoich_tokens=4,
    ).to(DEVICE)
    decoder.load_state_dict(dec_state, strict=False)

    encoder.eval()
    decoder.eval()

    epoch = checkpoint.get('epoch', '?')
    print(f"  Loaded epoch {epoch}")
    return encoder, decoder, magpie_dim


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
def decode_z_batch(encoder, decoder, z_batch, temperature=0.01):
    """Decode Z vectors → formula strings with encoder conditioning."""
    batch_size = 64
    all_formulas = []
    for start in range(0, len(z_batch), batch_size):
        z = z_batch[start:start + batch_size].to(DEVICE)
        dec_out = encoder.decode(z)
        encoder_skip = dec_out['attended_input']
        fraction_output = encoder.fraction_head(z)
        generated, _, _ = decoder.generate_with_kv_cache(
            z=z, encoder_skip=encoder_skip, stoich_pred=fraction_output,
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


def search_single_target(encoder, decoder, data, target_formula, target_tc, target_family):
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
        return {'target': target_formula, 'best_sim': 0.0, 'best_gen': '', 'n_unique': 0}

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
    greedy_formulas = decode_z_batch(encoder, decoder, all_z, temperature=0.01)
    print(f"    Greedy decoded in {time.time()-t0:.1f}s")

    # Step 5: Temperature sampling from seeds
    temp_formulas = []
    for temp in TEMPERATURES:
        for z_idx in range(min(len(z_seeds), 15)):
            z = z_seeds[z_idx].unsqueeze(0).repeat(N_TEMPERATURE_SAMPLES, 1)
            temp_formulas.extend(decode_z_batch(encoder, decoder, z, temperature=temp))

    all_generated = greedy_formulas + temp_formulas
    unique_formulas = set(f for f in all_generated if f and len(f) > 1)
    print(f"    Total: {len(all_generated)}, unique: {len(unique_formulas)}")

    # Step 6: Find best match
    best_sim = 0.0
    best_gen = ''
    top_matches = []

    for formula in unique_formulas:
        sim = element_similarity(formula, target_formula)
        if sim > best_sim:
            best_sim = sim
            best_gen = formula
        if sim >= 0.8:
            top_matches.append((formula, sim))

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
    formula_counts = defaultdict(int)
    for f in all_generated:
        if f and len(f) > 1:
            formula_counts[f] += 1
    top_freq = sorted(formula_counts.items(), key=lambda x: -x[1])[:10]
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
    }


def main():
    print("=" * 70)
    print("TARGETED HOLDOUT SEARCH (Element-Anchored)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    encoder, decoder, magpie_dim = load_models()
    data = load_data(magpie_dim)

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

    # Load holdout
    with open(HOLDOUT_PATH) as f:
        holdout = json.load(f)

    # Filter to targets below threshold
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
    for t in targets_to_search:
        print(f"  [{t['family']:12s}] prev={t['prev_best']:.3f} | {t['formula']}")

    # Search each target
    results = []
    for target in targets_to_search:
        result = search_single_target(
            encoder, decoder, data,
            target['formula'], target['Tc'], target['family'],
        )
        result['prev_best'] = target['prev_best']
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("TARGETED SEARCH SUMMARY")
    print("=" * 70)

    improved = 0
    now_found = 0
    for r in results:
        improved_flag = r['best_sim'] > r['prev_best']
        found_flag = r['best_sim'] >= 0.95
        if improved_flag:
            improved += 1
        if found_flag:
            now_found += 1
        marker = '***' if found_flag else ' + ' if improved_flag else '   '
        print(f"{marker} [{r['target_family']:12s}] "
              f"prev={r['prev_best']:.3f} → new={r['best_sim']:.3f} | {r['target']}")
        if r['best_gen']:
            print(f"     Best: {r['best_gen']}")

    print(f"\nImproved: {improved}/{len(results)}")
    print(f"Now found (>={REDO_THRESHOLD}): {now_found}/{len(results)}")

    # Combine with previous results for final count
    combined_best = dict(prev_best)
    for r in results:
        if r['best_sim'] > combined_best.get(r['target'], 0):
            combined_best[r['target']] = r['best_sim']

    thresholds = [1.0, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80]
    print(f"\nCOMBINED RESULTS (family search + targeted search):")
    for thresh in thresholds:
        found = sum(1 for v in combined_best.values() if v >= thresh)
        print(f"  >= {thresh:.2f}: {found}/45 ({found/45*100:.1f}%)")

    # Save
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoint': str(CHECKPOINT_PATH),
        'redo_threshold': REDO_THRESHOLD,
        'n_targets_searched': len(targets_to_search),
        'n_improved': improved,
        'n_now_found': now_found,
        'results': results,
        'combined_best': {k: float(v) for k, v in combined_best.items()},
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
