#!/usr/bin/env python3
"""
Holdout Material Search via Latent Space Exploration
=====================================================

Uses the trained checkpoint (epoch 2523/1775) to search for the 45 holdout
superconductors using family-specific Z-space exploration techniques.

Approach:
1. Load model + cached latent representations
2. For each holdout family, find training samples from that family
3. Apply multiple generation strategies centered on family Z-neighborhoods:
   - Nearest-neighbor perturbation
   - Family centroid walks
   - Pairwise interpolation within family
   - Temperature sampling from family members
   - SLERP interpolation
4. Compare ALL generated formulas against holdout set (exact + fuzzy match)

Usage:
    cd /home/james/superconductor-vae
    conda activate recursivemenn-py311
    PYTHONPATH=src python scratch/holdout_search.py
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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, IDX_TO_TOKEN, TOKEN_TO_IDX,
    PAD_IDX, START_IDX, END_IDX, tokenize_formula,
)
from superconductor.models.family_classifier import SuperconductorFamily
from superconductor.data.canonical_ordering import CanonicalOrderer, ElementWithFraction

# ─── Config ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / 'outputs' / 'checkpoint_best.pt'
HOLDOUT_PATH = PROJECT_ROOT / 'data' / 'GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json'
CACHE_DIR = PROJECT_ROOT / 'data' / 'processed' / 'cache'
OUTPUT_PATH = PROJECT_ROOT / 'scratch' / 'holdout_search_results.json'

# Generation hyperparameters
N_PERTURBATIONS = 50       # Per seed sample
N_INTERPOLATION_STEPS = 10  # Steps between pairs
N_TEMPERATURE_SAMPLES = 20  # Per seed at each temperature
N_SEEDS_PER_FAMILY = 20    # Top K nearest neighbors to use as seeds
NOISE_SCALES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
TEMPERATURES = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
STOP_BOOST = 0.0  # V12.30 stop boost (0 = disabled)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Holdout family → SuperconductorFamily enum mapping
HOLDOUT_TO_FAMILY_ENUM = {
    'YBCO': [SuperconductorFamily.CUPRATE_YBCO],
    'LSCO': [SuperconductorFamily.CUPRATE_LSCO],
    'Hg-cuprate': [SuperconductorFamily.CUPRATE_HBCCO],
    'Tl-cuprate': [SuperconductorFamily.CUPRATE_TBCCO],
    'Bi-cuprate': [SuperconductorFamily.CUPRATE_BSCCO],
    'Iron-based': [SuperconductorFamily.IRON_PNICTIDE, SuperconductorFamily.IRON_CHALCOGENIDE],
    'MgB2': [SuperconductorFamily.MGB2_TYPE],
    'Conventional': [SuperconductorFamily.BCS_CONVENTIONAL],
    'Other': [SuperconductorFamily.OTHER_UNKNOWN, SuperconductorFamily.HEAVY_FERMION,
              SuperconductorFamily.CUPRATE_OTHER],
}


# ─── Utilities ────────────────────────────────────────────────────────────────

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


def normalize_formula(formula):
    """Normalize formula for comparison (strip whitespace, sort elements)."""
    return formula.strip()


_CANONICALIZER = CanonicalOrderer()

def parse_formula_elements(formula):
    """Extract element set and approximate fractions from formula string.

    Returns dict of {element: approximate_molar_fraction}.
    """
    try:
        elements = _CANONICALIZER.parse_formula(formula)
        if not elements:
            return {}
        # Convert to dict of element: fraction_value
        result = {}
        for ef in elements:
            val = ef.fraction_value
            result[ef.element] = result.get(ef.element, 0) + val
        return result
    except Exception:
        return {}


def slerp(z1, z2, t):
    """Spherical linear interpolation between z1 and z2."""
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    omega = torch.acos(torch.clamp(
        (z1_norm * z2_norm).sum(dim=-1, keepdim=True), -1.0, 1.0
    ))
    omega = omega.clamp(min=1e-6)
    sin_omega = torch.sin(omega)
    # Handle near-parallel vectors
    if sin_omega.abs().min() < 1e-6:
        return (1 - t) * z1 + t * z2
    s1 = torch.sin((1 - t) * omega) / sin_omega
    s2 = torch.sin(t * omega) / sin_omega
    # Scale to original magnitudes
    mag1 = z1.norm(dim=-1, keepdim=True)
    mag2 = z2.norm(dim=-1, keepdim=True)
    mag = (1 - t) * mag1 + t * mag2
    return (s1 * z1_norm + s2 * z2_norm) * mag


def element_similarity(formula_a, formula_b):
    """Compute compositional similarity between two formulas.

    Returns Jaccard-like similarity based on shared elements and fraction overlap.
    """
    parsed_a = parse_formula_elements(formula_a)
    parsed_b = parse_formula_elements(formula_b)

    if not parsed_a or not parsed_b:
        return 0.0

    all_elements = set(parsed_a.keys()) | set(parsed_b.keys())
    shared = set(parsed_a.keys()) & set(parsed_b.keys())

    if not all_elements:
        return 0.0

    # Jaccard on element sets
    jaccard = len(shared) / len(all_elements)

    # Fraction overlap for shared elements
    if shared:
        total_a = sum(parsed_a.values())
        total_b = sum(parsed_b.values())
        frac_overlap = 0.0
        for elem in shared:
            fa = parsed_a[elem] / max(total_a, 1e-8)
            fb = parsed_b[elem] / max(total_b, 1e-8)
            frac_overlap += min(fa, fb)
        frac_sim = frac_overlap / max(1.0, 1.0)
    else:
        frac_sim = 0.0

    return 0.5 * jaccard + 0.5 * frac_sim


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_models():
    """Load encoder and decoder from checkpoint."""
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    # Detect magpie_dim from checkpoint weights
    enc_state_raw = checkpoint.get('encoder_state_dict', checkpoint.get('model_state_dict', {}))
    magpie_dim = 145  # default
    for k, v in enc_state_raw.items():
        if 'magpie_encoder' in k and k.endswith('.weight') and v.dim() == 2:
            magpie_dim = v.shape[1]  # Input dim of first magpie layer
            break
    print(f"  Detected magpie_dim={magpie_dim} from checkpoint")

    # Clean _orig_mod. prefix from torch.compile
    enc_state = {k.replace('_orig_mod.', ''): v for k, v in enc_state_raw.items()}

    # Encoder
    encoder = FullMaterialsVAE(
        n_elements=118, element_embed_dim=128, n_attention_heads=8,
        magpie_dim=magpie_dim, fusion_dim=256, encoder_hidden=[512, 256],
        latent_dim=2048, decoder_hidden=[256, 512], dropout=0.1
    ).to(DEVICE)

    enc_missing, enc_unexpected = encoder.load_state_dict(enc_state, strict=False)
    if enc_missing:
        print(f"  Encoder missing keys: {len(enc_missing)}")
        for k in enc_missing[:5]:
            print(f"    {k}")
    if enc_unexpected:
        print(f"  Encoder unexpected keys: {len(enc_unexpected)}")

    # Decoder
    dec_state_raw = checkpoint.get('decoder_state_dict', {})
    dec_state = {k.replace('_orig_mod.', ''): v for k, v in dec_state_raw.items()}

    # V13.0: Auto-detect vocab_size and stoich_input_dim from checkpoint
    dec_vocab_size = checkpoint.get('tokenizer_vocab_size', None)
    if dec_vocab_size is None and 'token_embedding.weight' in dec_state:
        dec_vocab_size = dec_state['token_embedding.weight'].shape[0]
    stoich_dim = None
    if 'stoich_to_memory.0.weight' in dec_state:
        stoich_dim = dec_state['stoich_to_memory.0.weight'].shape[1]

    decoder = EnhancedTransformerDecoder(
        latent_dim=2048, d_model=512, nhead=8, num_layers=12,
        dim_feedforward=2048, dropout=0.1, max_len=60,
        n_memory_tokens=16, encoder_skip_dim=256,
        use_skip_connection=True, use_stoich_conditioning=True,
        max_elements=12, n_stoich_tokens=4,
        vocab_size=dec_vocab_size,
        stoich_input_dim=stoich_dim,
    ).to(DEVICE)

    dec_missing, dec_unexpected = decoder.load_state_dict(dec_state, strict=False)
    if dec_missing:
        print(f"  Decoder missing keys: {len(dec_missing)}")
        for k in dec_missing[:5]:
            print(f"    {k}")
    if dec_unexpected:
        print(f"  Decoder unexpected keys: {len(dec_unexpected)}")

    encoder.eval()
    decoder.eval()

    epoch = checkpoint.get('epoch', '?')
    best_exact = checkpoint.get('best_exact', '?')
    print(f"  Loaded epoch {epoch}, best exact: {best_exact}%")

    return encoder, decoder, magpie_dim


def load_data():
    """Load cached dataset tensors and holdout set."""
    print("\nLoading cached data...")

    # Load cached tensors
    elem_idx = torch.load(CACHE_DIR / 'element_indices.pt', map_location='cpu', weights_only=True)
    elem_frac = torch.load(CACHE_DIR / 'element_fractions.pt', map_location='cpu', weights_only=True)
    elem_mask = torch.load(CACHE_DIR / 'element_mask.pt', map_location='cpu', weights_only=True)
    tc_tensor = torch.load(CACHE_DIR / 'tc_tensor.pt', map_location='cpu', weights_only=True)
    magpie_tensor = torch.load(CACHE_DIR / 'magpie_tensor.pt', map_location='cpu', weights_only=True)
    family_tensor = torch.load(CACHE_DIR / 'family_tensor.pt', map_location='cpu', weights_only=True)
    formula_tokens = torch.load(CACHE_DIR / 'formula_tokens.pt', map_location='cpu', weights_only=True)
    is_sc = torch.load(CACHE_DIR / 'is_sc_tensor.pt', map_location='cpu', weights_only=True)

    # Get train indices from cache metadata
    meta = json.load(open(CACHE_DIR / 'cache_meta.json'))
    train_indices = meta.get('train_indices', list(range(len(elem_idx))))

    print(f"  Total samples: {len(elem_idx)}")
    print(f"  Train indices: {len(train_indices)}")
    print(f"  Family distribution:")
    for fam in sorted(torch.unique(family_tensor).tolist()):
        count = (family_tensor == fam).sum().item()
        name = SuperconductorFamily(fam).name if fam < 14 else f"UNKNOWN_{fam}"
        print(f"    {fam} ({name}): {count}")

    # Load holdout set
    with open(HOLDOUT_PATH) as f:
        holdout_data = json.load(f)

    holdout_samples = holdout_data['holdout_samples']
    print(f"\n  Holdout set: {len(holdout_samples)} formulas across {len(set(s['family'] for s in holdout_samples))} families")

    # Group holdout by family
    holdout_by_family = defaultdict(list)
    for s in holdout_samples:
        holdout_by_family[s['family']].append(s)

    for fam, samples in holdout_by_family.items():
        formulas = [s['formula'] for s in samples]
        tcs = [s['Tc'] for s in samples]
        print(f"    {fam}: {len(samples)} formulas, Tc range: {min(tcs):.1f}-{max(tcs):.1f}K")

    data = {
        'elem_idx': elem_idx,
        'elem_frac': elem_frac,
        'elem_mask': elem_mask,
        'tc': tc_tensor,
        'magpie': magpie_tensor,
        'family': family_tensor,
        'tokens': formula_tokens,
        'is_sc': is_sc,
        'train_indices': train_indices,
    }

    return data, holdout_samples, holdout_by_family


# ─── Encoding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_samples(encoder, data, indices):
    """Encode a set of samples into Z-space.

    Returns z [N, 2048] on CPU.
    """
    batch_size = 128
    all_z = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        idx_t = torch.tensor(batch_idx, dtype=torch.long)

        ei = data['elem_idx'][idx_t].to(DEVICE)
        ef = data['elem_frac'][idx_t].to(DEVICE)
        em = data['elem_mask'][idx_t].to(DEVICE)
        tc = data['tc'][idx_t].to(DEVICE)
        mag = data['magpie'][idx_t].to(DEVICE)

        result = encoder.encode(ei, ef, em, mag, tc)
        z = result['z']
        all_z.append(z.cpu())

    return torch.cat(all_z, dim=0)


@torch.no_grad()
def get_decoder_conditioning(encoder, z_batch):
    """Get encoder_skip (attended_input) and stoich_pred from Z vectors.

    The decoder needs these for skip connections and stoichiometry conditioning.

    Args:
        encoder: FullMaterialsVAE (has decode() method: z -> attended_input, fraction_pred, etc.)
        z_batch: [N, 2048] latent vectors (CPU or GPU)

    Returns:
        attended_input: [N, 256] encoder skip connection
        fraction_pred: [N, 13] stoichiometry prediction
    """
    batch_size = 128
    all_skip = []
    all_stoich = []

    for start in range(0, len(z_batch), batch_size):
        z = z_batch[start:start + batch_size].to(DEVICE)

        # encoder.decode() gives us tc_pred, magpie_pred, attended_input
        dec_out = encoder.decode(z)
        all_skip.append(dec_out['attended_input'].cpu())

        # Get fraction_pred from fraction_head
        fraction_output = encoder.fraction_head(z)
        fraction_pred = fraction_output[:, :12]  # max_elements=12
        all_stoich.append(fraction_pred.cpu())

    return torch.cat(all_skip, dim=0), torch.cat(all_stoich, dim=0)


@torch.no_grad()
def decode_z_batch(encoder, decoder, z_batch, temperature=0.01, stop_boost=0.0):
    """Decode a batch of Z vectors into formula strings.

    Uses encoder.decode() to get skip connections and stoichiometry conditioning,
    then passes them to the decoder for generation.

    Args:
        encoder: FullMaterialsVAE (for computing attended_input + fraction_pred)
        decoder: EnhancedTransformerDecoder
        z_batch: [N, 2048] latent vectors (on CPU or GPU)
        temperature: sampling temperature
        stop_boost: V12.30 stop head boost

    Returns:
        list of formula strings
    """
    batch_size = min(64, len(z_batch))
    all_formulas = []

    for start in range(0, len(z_batch), batch_size):
        z = z_batch[start:start + batch_size].to(DEVICE)

        # Get decoder conditioning from encoder heads
        dec_out = encoder.decode(z)
        encoder_skip = dec_out['attended_input']

        # Get stoichiometry prediction
        fraction_output = encoder.fraction_head(z)
        stoich_pred = fraction_output  # [batch, 13] (12 fractions + count)

        generated, _, _ = decoder.generate_with_kv_cache(
            z=z,
            encoder_skip=encoder_skip,
            stoich_pred=stoich_pred,
            temperature=temperature,
            stop_boost=stop_boost,
        )
        for i in range(len(z)):
            formula = tokens_to_formula(generated[i])
            all_formulas.append(formula)

    return all_formulas


# ─── Generation Strategies ────────────────────────────────────────────────────

def strategy_perturbation(z_seeds, noise_scales, n_per_scale):
    """Random perturbation around seed Z vectors.

    For each seed and each noise scale, generate n_per_scale candidates.
    """
    candidates = []
    for z in z_seeds:
        z = z.unsqueeze(0)  # [1, 2048]
        for scale in noise_scales:
            noise = torch.randn(n_per_scale, z.shape[1]) * scale
            z_perturbed = z + noise
            candidates.append(z_perturbed)

    return torch.cat(candidates, dim=0)


def strategy_interpolation(z_seeds, n_steps=10, method='linear'):
    """Pairwise interpolation between seed Z vectors."""
    candidates = []
    n = len(z_seeds)
    if n < 2:
        return torch.empty(0, z_seeds.shape[1])

    # For tractability, limit to top pairs
    max_pairs = min(n * (n - 1) // 2, 50)
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if pair_count >= max_pairs:
                break
            z1 = z_seeds[i].unsqueeze(0)
            z2 = z_seeds[j].unsqueeze(0)
            for t_val in np.linspace(0.1, 0.9, n_steps):
                t = float(t_val)
                if method == 'slerp':
                    z_interp = slerp(z1, z2, t)
                else:
                    z_interp = (1 - t) * z1 + t * z2
                candidates.append(z_interp)
            pair_count += 1
        if pair_count >= max_pairs:
            break

    if not candidates:
        return torch.empty(0, z_seeds.shape[1])
    return torch.cat(candidates, dim=0)


def strategy_centroid_walk(z_seeds, n_steps=20, walk_scales=[0.5, 1.0, 1.5, 2.0]):
    """Walk outward from the family centroid in random directions."""
    centroid = z_seeds.mean(dim=0, keepdim=True)  # [1, 2048]
    candidates = []

    for scale in walk_scales:
        # Random directions
        directions = torch.randn(n_steps, z_seeds.shape[1])
        directions = F.normalize(directions, dim=-1)
        z_walked = centroid + scale * z_seeds.std(dim=0, keepdim=True) * directions
        candidates.append(z_walked)

    return torch.cat(candidates, dim=0)


def strategy_pca_walk(z_seeds, n_per_component=10, n_components=10, walk_range=3.0):
    """Walk along PCA directions of the family's Z distribution."""
    if len(z_seeds) < 5:
        return torch.empty(0, z_seeds.shape[1])

    z_np = z_seeds.numpy()
    mean = z_np.mean(axis=0)
    centered = z_np - mean

    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n_comp = min(n_components, len(S))

    candidates = []
    for c in range(n_comp):
        direction = torch.from_numpy(Vt[c]).float()
        std_along = S[c] / np.sqrt(len(z_seeds) - 1)
        for alpha in np.linspace(-walk_range, walk_range, n_per_component):
            z_walked = torch.from_numpy(mean).float().unsqueeze(0) + alpha * std_along * direction.unsqueeze(0)
            candidates.append(z_walked)

    return torch.cat(candidates, dim=0)


def strategy_temperature_sampling(encoder, decoder, z_seeds, temperatures, n_per_temp):
    """Sample from seed Z vectors at different temperatures.

    Unlike other strategies, this generates formulas directly (different
    randomness at each temperature).
    """
    all_formulas = []
    for temp in temperatures:
        for z_idx in range(min(len(z_seeds), 10)):  # Top 10 seeds
            z = z_seeds[z_idx].unsqueeze(0).repeat(n_per_temp, 1)
            formulas = decode_z_batch(encoder, decoder, z, temperature=temp)
            all_formulas.extend(formulas)
    return all_formulas


# ─── Matching ─────────────────────────────────────────────────────────────────

def check_exact_match(generated_formula, holdout_formulas):
    """Check if generated formula exactly matches any holdout formula."""
    gen_norm = normalize_formula(generated_formula)
    for hf in holdout_formulas:
        if gen_norm == normalize_formula(hf):
            return hf
    return None


def check_fuzzy_match(generated_formula, holdout_formulas, threshold=0.7):
    """Check if generated formula is compositionally similar to any holdout formula."""
    matches = []
    for hf in holdout_formulas:
        sim = element_similarity(generated_formula, hf)
        if sim >= threshold:
            matches.append((hf, sim))
    return sorted(matches, key=lambda x: -x[1])


# ─── Main Search ──────────────────────────────────────────────────────────────

def search_family(encoder, decoder, data, family_name, holdout_formulas,
                  family_enum_values, all_formulas_set):
    """Search for holdout materials within a specific family's Z-space.

    Args:
        encoder: FullMaterialsVAE
        decoder: EnhancedTransformerDecoder
        data: dict of cached tensors
        family_name: holdout family name (e.g., 'YBCO')
        holdout_formulas: list of formula strings to find
        family_enum_values: list of SuperconductorFamily enum values
        all_formulas_set: set of all holdout formulas (for matching)

    Returns:
        dict with results for this family
    """
    print(f"\n{'='*60}")
    print(f"SEARCHING: {family_name}")
    print(f"  Target formulas: {holdout_formulas}")
    print(f"  Family enum values: {[v.name for v in family_enum_values]}")

    # Find training samples from this family
    family_mask = torch.zeros(len(data['family']), dtype=torch.bool)
    for fev in family_enum_values:
        family_mask |= (data['family'] == fev.value)

    # Only use SC samples from this family
    sc_mask = data['is_sc'] == 1
    combined_mask = family_mask & sc_mask
    family_indices = torch.where(combined_mask)[0].tolist()

    print(f"  Family samples (SC only): {len(family_indices)}")

    if len(family_indices) < 3:
        print(f"  WARNING: Too few family samples ({len(family_indices)}), expanding to similar families")
        # Fallback: include all cuprate families for cuprate holdouts
        if 'cuprate' in family_name.lower() or family_name in ['YBCO', 'LSCO']:
            for fam in [2, 3, 4, 5, 6, 7]:  # All cuprate families
                family_mask |= (data['family'] == fam)
            combined_mask = family_mask & sc_mask
            family_indices = torch.where(combined_mask)[0].tolist()
            print(f"  Expanded to {len(family_indices)} samples")

    if len(family_indices) == 0:
        print(f"  ERROR: No samples found for {family_name}")
        return {'family': family_name, 'exact_matches': [], 'fuzzy_matches': [],
                'n_candidates': 0, 'n_unique': 0}

    # Get formulas for these indices (for logging)
    family_formulas = []
    for idx in family_indices[:10]:  # First 10 for display
        formula = tokens_to_formula(data['tokens'][idx])
        family_formulas.append(formula)
    print(f"  Sample formulas: {family_formulas[:5]}")

    # Encode family samples
    print(f"  Encoding {len(family_indices)} family samples...")
    # Limit encoding to manageable size
    encode_indices = family_indices[:2000]
    z_family = encode_samples(encoder, data, encode_indices)
    print(f"  Z shape: {z_family.shape}, norm: {z_family.norm(dim=-1).mean():.2f}±{z_family.norm(dim=-1).std():.2f}")

    # Select seed Z vectors (highest Tc samples)
    tc_values = data['tc'][torch.tensor(encode_indices, dtype=torch.long)].squeeze()
    top_tc_indices = tc_values.argsort(descending=True)[:N_SEEDS_PER_FAMILY]
    z_seeds = z_family[top_tc_indices]

    print(f"  Using {len(z_seeds)} seed Z vectors")

    # ─── Apply all generation strategies ─────────────────────────────────

    all_candidates_z = []
    strategy_names = []

    # Strategy 1: Random perturbation
    print("  Strategy 1: Random perturbation...")
    z_perturbed = strategy_perturbation(z_seeds, NOISE_SCALES, N_PERTURBATIONS)
    all_candidates_z.append(z_perturbed)
    strategy_names.append(f'perturbation ({len(z_perturbed)})')
    print(f"    Generated {len(z_perturbed)} Z vectors")

    # Strategy 2: Linear interpolation
    print("  Strategy 2: Linear interpolation...")
    z_interp = strategy_interpolation(z_seeds, n_steps=N_INTERPOLATION_STEPS, method='linear')
    if len(z_interp) > 0:
        all_candidates_z.append(z_interp)
        strategy_names.append(f'linear_interp ({len(z_interp)})')
        print(f"    Generated {len(z_interp)} Z vectors")

    # Strategy 3: SLERP interpolation
    print("  Strategy 3: SLERP interpolation...")
    z_slerp = strategy_interpolation(z_seeds, n_steps=N_INTERPOLATION_STEPS, method='slerp')
    if len(z_slerp) > 0:
        all_candidates_z.append(z_slerp)
        strategy_names.append(f'slerp ({len(z_slerp)})')
        print(f"    Generated {len(z_slerp)} Z vectors")

    # Strategy 4: Centroid walk
    print("  Strategy 4: Centroid walk...")
    z_centroid = strategy_centroid_walk(z_seeds)
    all_candidates_z.append(z_centroid)
    strategy_names.append(f'centroid_walk ({len(z_centroid)})')
    print(f"    Generated {len(z_centroid)} Z vectors")

    # Strategy 5: PCA-directed walk
    print("  Strategy 5: PCA-directed walk...")
    z_pca = strategy_pca_walk(z_family, n_per_component=15, n_components=15)
    if len(z_pca) > 0:
        all_candidates_z.append(z_pca)
        strategy_names.append(f'pca_walk ({len(z_pca)})')
        print(f"    Generated {len(z_pca)} Z vectors")

    # Combine all Z candidates
    all_z = torch.cat(all_candidates_z, dim=0)
    print(f"\n  Total Z candidates: {len(all_z)}")

    # Decode all Z candidates with greedy decoding (temperature=0.01)
    print("  Decoding Z candidates (greedy)...")
    t0 = time.time()
    greedy_formulas = decode_z_batch(encoder, decoder, all_z, temperature=0.01)
    print(f"    Decoded in {time.time()-t0:.1f}s")

    # Strategy 6: Temperature sampling (generates formulas directly)
    print("  Strategy 6: Temperature sampling...")
    temp_formulas = strategy_temperature_sampling(
        encoder, decoder, z_seeds, TEMPERATURES, N_TEMPERATURE_SAMPLES
    )
    print(f"    Generated {len(temp_formulas)} formulas")

    # Combine all formulas
    all_generated = greedy_formulas + temp_formulas
    unique_formulas = set(f for f in all_generated if f and len(f) > 1)
    print(f"\n  Total formulas: {len(all_generated)}, unique: {len(unique_formulas)}")

    # ─── Check matches ───────────────────────────────────────────────────

    exact_matches = []
    fuzzy_matches = defaultdict(list)

    for formula in unique_formulas:
        # Check against THIS family's holdout formulas
        em = check_exact_match(formula, holdout_formulas)
        if em:
            exact_matches.append((formula, em))

        # Check against ALL holdout formulas
        em_all = check_exact_match(formula, list(all_formulas_set))
        if em_all:
            if em_all not in [e[1] for e in exact_matches]:
                exact_matches.append((formula, em_all))

        # Fuzzy match
        fm = check_fuzzy_match(formula, holdout_formulas, threshold=0.6)
        for target, sim in fm:
            fuzzy_matches[target].append((formula, sim))

    # Sort fuzzy matches by similarity
    for target in fuzzy_matches:
        fuzzy_matches[target] = sorted(fuzzy_matches[target], key=lambda x: -x[1])[:10]

    # Report
    print(f"\n  RESULTS for {family_name}:")
    print(f"    Exact matches: {len(exact_matches)}")
    for gen, target in exact_matches:
        print(f"      FOUND: {gen} == {target}")

    print(f"    Fuzzy matches (>0.6 similarity):")
    for target, matches in fuzzy_matches.items():
        print(f"      Target: {target}")
        for gen, sim in matches[:3]:
            print(f"        {gen} (sim={sim:.3f})")

    # Top generated formulas for inspection
    formula_counts = defaultdict(int)
    for f in all_generated:
        if f and len(f) > 1:
            formula_counts[f] += 1
    top_formulas = sorted(formula_counts.items(), key=lambda x: -x[1])[:20]
    print(f"    Top generated formulas:")
    for f, count in top_formulas:
        print(f"      {f} (count={count})")

    return {
        'family': family_name,
        'n_family_samples': len(family_indices),
        'n_seeds': len(z_seeds),
        'n_candidates_z': len(all_z),
        'n_total_formulas': len(all_generated),
        'n_unique': len(unique_formulas),
        'exact_matches': [(g, t) for g, t in exact_matches],
        'fuzzy_matches': {t: [(g, s) for g, s in ms[:5]] for t, ms in fuzzy_matches.items()},
        'top_generated': top_formulas[:20],
        'strategies': strategy_names,
    }


def main():
    print("=" * 60)
    print("HOLDOUT MATERIAL SEARCH")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()

    # Load models
    encoder, decoder, magpie_dim = load_models()

    # Load data
    data, holdout_samples, holdout_by_family = load_data()

    # Slice magpie features to match model's expected dimension
    if data['magpie'].shape[1] > magpie_dim:
        print(f"\nSlicing Magpie features from {data['magpie'].shape[1]} to {magpie_dim}")
        data['magpie'] = data['magpie'][:, :magpie_dim]

    # All holdout formulas (for cross-family matching)
    all_holdout_formulas = set(s['formula'] for s in holdout_samples)
    print(f"\nTotal holdout formulas to find: {len(all_holdout_formulas)}")

    # Search each family
    results = {}
    total_exact = 0
    total_fuzzy = 0

    for family_name, samples in holdout_by_family.items():
        holdout_formulas = [s['formula'] for s in samples]
        family_enums = HOLDOUT_TO_FAMILY_ENUM.get(family_name, [SuperconductorFamily.OTHER_UNKNOWN])

        result = search_family(
            encoder, decoder, data,
            family_name, holdout_formulas,
            family_enums, all_holdout_formulas,
        )
        results[family_name] = result
        total_exact += len(result['exact_matches'])
        total_fuzzy += sum(len(v) for v in result['fuzzy_matches'].values())

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SEARCH SUMMARY")
    print("=" * 60)
    print(f"Families searched: {len(results)}")
    print(f"Total exact matches: {total_exact} / {len(all_holdout_formulas)}")
    print(f"Total fuzzy matches (>0.6): {total_fuzzy}")
    print()

    for fam, res in results.items():
        n_exact = len(res['exact_matches'])
        n_fuzzy = sum(len(v) for v in res['fuzzy_matches'].values())
        print(f"  {fam}: {n_exact} exact, {n_fuzzy} fuzzy | "
              f"{res['n_unique']} unique from {res['n_total_formulas']} total | "
              f"{res['n_family_samples']} family samples")

    # Save results
    # Convert for JSON serialization
    json_results = {}
    for fam, res in results.items():
        json_results[fam] = {
            'family': res['family'],
            'n_family_samples': res['n_family_samples'],
            'n_seeds': res['n_seeds'],
            'n_candidates_z': res['n_candidates_z'],
            'n_total_formulas': res['n_total_formulas'],
            'n_unique': res['n_unique'],
            'exact_matches': res['exact_matches'],
            'fuzzy_matches': {t: [(g, float(s)) for g, s in ms] for t, ms in res['fuzzy_matches'].items()},
            'top_generated': [(f, c) for f, c in res['top_generated']],
            'strategies': res['strategies'],
        }

    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoint': str(CHECKPOINT_PATH),
        'device': str(DEVICE),
        'n_holdout': len(all_holdout_formulas),
        'total_exact_matches': total_exact,
        'total_fuzzy_matches': total_fuzzy,
        'config': {
            'n_perturbations': N_PERTURBATIONS,
            'noise_scales': NOISE_SCALES,
            'n_interpolation_steps': N_INTERPOLATION_STEPS,
            'n_temperature_samples': N_TEMPERATURE_SAMPLES,
            'temperatures': TEMPERATURES,
            'n_seeds_per_family': N_SEEDS_PER_FAMILY,
        },
        'results': json_results,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
