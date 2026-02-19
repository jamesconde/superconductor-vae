#!/usr/bin/env python3
"""
Holdout Full Enchilada Validation
==================================

For each of the 45 holdout materials, evaluates EVERY model capability:
1. Formula generation (encode → Z → decode → token sequence)
2. Tc prediction (regression, continuous Kelvin)
3. Tc classification (5-bucket: non-SC, 0-10K, 10-50K, 50-100K, 100K+)
4. Magpie feature reconstruction (145-dim compositional descriptors)
5. SC classification (binary: is this a superconductor?)
6. Family classification (14-class hierarchical: coarse family → sub-family)
7. High-pressure prediction (binary: requires high pressure?)

V12.40: Uses encoder.forward() to get all heads in one pass.

Usage:
    cd /home/james/superconductor-vae
    PYTHONPATH=src python -u scripts/holdout/holdout_tc_validation.py
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, IDX_TO_TOKEN,
    PAD_IDX, START_IDX, END_IDX,
)
from superconductor.data.canonical_ordering import CanonicalOrderer
from superconductor.models.family_classifier import SuperconductorFamily

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser(description='Holdout validation for superconductor VAE')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to checkpoint file (default: outputs/checkpoint_best.pt)')
args, _ = parser.parse_known_args()

if args.checkpoint:
    CHECKPOINT_PATH = Path(args.checkpoint)
else:
    CHECKPOINT_PATH = PROJECT_ROOT / 'outputs' / 'checkpoint_best.pt'

# Build output filename from checkpoint name
ckpt_stem = CHECKPOINT_PATH.stem  # e.g. 'checkpoint_best_V12.38_colab'
OUTPUT_PATH = PROJECT_ROOT / 'outputs' / f'holdout_tc_validation_{ckpt_stem}.json'

HOLDOUT_PATH = PROJECT_ROOT / 'data' / 'GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json'
CACHE_DIR = PROJECT_ROOT / 'data' / 'processed' / 'cache'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CANONICALIZER = CanonicalOrderer()

# Tc normalization parameters (from cache_meta.json)
# Tc pipeline: raw_K → log1p → z-score(mean=TC_MEAN, std=TC_STD) → normalized
# Denorm: normalized → * TC_STD + TC_MEAN → expm1 → raw_K
TC_MEAN = 2.725219433789196
TC_STD = 1.3527019896187407
TC_LOG_TRANSFORM = True


def denormalize_tc(tc_norm):
    """Convert normalized Tc prediction back to Kelvin."""
    tc_log = tc_norm * TC_STD + TC_MEAN
    if TC_LOG_TRANSFORM:
        return max(0.0, np.expm1(tc_log))
    return max(0.0, tc_log)


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


def parse_formula_elements(formula):
    try:
        elements = _CANONICALIZER.parse_formula(formula)
        if not elements:
            return {}
        return {e.element: e.fraction_value for e in elements}
    except Exception:
        return {}


def element_similarity(gen_formula, target_formula):
    gen_comp = parse_formula_elements(gen_formula)
    target_comp = parse_formula_elements(target_formula)
    if not gen_comp or not target_comp:
        return 0.0
    all_elements = set(gen_comp.keys()) | set(target_comp.keys())
    if not all_elements:
        return 0.0
    shared = set(gen_comp.keys()) & set(target_comp.keys())
    jaccard = len(shared) / len(all_elements)
    if not shared:
        return 0.0
    frac_overlap = 0.0
    total_weight = 0.0
    for elem in all_elements:
        g = gen_comp.get(elem, 0.0)
        t = target_comp.get(elem, 0.0)
        max_val = max(abs(g), abs(t))
        if max_val > 0:
            frac_overlap += 1.0 - abs(g - t) / max_val
            total_weight += 1.0
    frac_sim = frac_overlap / total_weight if total_weight > 0 else 0.0
    return 0.4 * jaccard + 0.6 * frac_sim


def load_models():
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

    # V12.28: Net2Net weight transfer for old tc_head → new tc_proj/tc_res_block/tc_out
    has_old_tc_head = any('tc_head.' in k for k in enc_state)
    has_new_tc_proj = any('tc_proj.' in k for k in enc_state)
    if has_old_tc_head and not has_new_tc_proj:
        encoder.upgrade_tc_head_from_checkpoint(enc_state)
        print("  Applied Net2Net weight transfer for Tc head upgrade")

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
        use_skip_connection=True, use_stoich_conditioning=True,
        max_elements=12, n_stoich_tokens=4,
        vocab_size=dec_vocab_size,
        stoich_input_dim=stoich_dim,
    ).to(DEVICE)
    decoder.load_state_dict(dec_state, strict=False)

    encoder.eval()
    decoder.eval()
    epoch = checkpoint.get('epoch', '?')
    print(f"  Loaded epoch {epoch}")
    return encoder, decoder, magpie_dim


def load_data(magpie_dim):
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
    return data


@torch.no_grad()
def full_forward(encoder, decoder, elem_idx, elem_frac, elem_mask, magpie, tc, temperature=0.01):
    """Run encoder.forward() to get ALL head predictions, then formula decoder.

    V12.40: Uses forward() instead of separate encode/decode to capture
    sc_pred, family classification, and hp_pred.

    Returns dict with every prediction the model makes.
    """
    # Full forward pass through encoder — runs ALL heads
    enc_out = encoder(
        elem_idx.to(DEVICE),
        elem_frac.to(DEVICE),
        elem_mask.to(DEVICE),
        magpie.to(DEVICE),
        tc.to(DEVICE),
    )

    z = enc_out['z']
    tc_pred_norm = enc_out['tc_pred'][0].item()
    magpie_pred = enc_out['magpie_pred'][0].cpu()

    # Stoichiometry for decoder conditioning
    fraction_pred = enc_out['fraction_pred']
    element_count_pred = enc_out['element_count_pred']
    numden_pred = enc_out.get('numden_pred')

    # Assemble stoich_pred matching train_v12_clean.py logic
    if numden_pred is not None:
        stoich_pred = torch.cat([fraction_pred, numden_pred, element_count_pred.unsqueeze(-1)], dim=-1)
    else:
        stoich_pred = torch.cat([fraction_pred, element_count_pred.unsqueeze(-1)], dim=-1)

    # Skip connection for formula decoder
    encoder_skip = enc_out['attended_input']

    # Formula generation
    generated, log_probs, entropy = decoder.generate_with_kv_cache(
        z=z, encoder_skip=encoder_skip, stoich_pred=stoich_pred,
        temperature=temperature,
    )
    formula = tokens_to_formula(generated[0])

    # Denormalize Tc
    tc_kelvin = denormalize_tc(tc_pred_norm)

    result = {
        'formula': formula,
        'tc_pred_norm': tc_pred_norm,
        'tc_pred_kelvin': tc_kelvin,
        'magpie_pred': magpie_pred,
        'stoich_pred': stoich_pred[0].cpu(),
        'log_prob': log_probs[0].sum().item() if log_probs is not None else None,
    }

    # Tc classification (5 buckets)
    tc_class_logits = enc_out.get('tc_class_logits')
    if tc_class_logits is not None:
        probs = torch.softmax(tc_class_logits[0].cpu(), dim=-1)
        result['tc_class_probs'] = probs
        result['tc_class'] = probs.argmax().item()

    # SC classification (binary, sigmoid of logit)
    sc_pred = enc_out.get('sc_pred')
    if sc_pred is not None:
        sc_logit = sc_pred[0].item()
        sc_prob = torch.sigmoid(sc_pred[0]).item()
        result['sc_logit'] = sc_logit
        result['sc_prob'] = sc_prob
        result['sc_pred'] = sc_prob > 0.5  # True = superconductor

    # Family classification (14-class hierarchical)
    family_composed = enc_out.get('family_composed_14')
    if family_composed is not None:
        family_probs = family_composed[0].cpu()
        result['family_probs_14'] = family_probs
        result['family_pred_14'] = family_probs.argmax().item()
        # Also extract coarse family
        coarse_logits = enc_out.get('family_coarse_logits')
        if coarse_logits is not None:
            result['family_coarse_probs'] = torch.softmax(coarse_logits[0].cpu(), dim=-1)

    # High-pressure prediction (binary, sigmoid of logit)
    hp_pred = enc_out.get('hp_pred')
    if hp_pred is not None:
        hp_prob = torch.sigmoid(hp_pred[0]).item()
        result['hp_prob'] = hp_prob
        result['hp_pred'] = hp_prob > 0.5  # True = high-pressure

    return result


def main():
    print("=" * 70)
    print("HOLDOUT FULL ENCHILADA VALIDATION")
    print("(Formula + Tc + Magpie from Z)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    encoder, decoder, magpie_dim = load_models()
    data = load_data(magpie_dim)

    with open(HOLDOUT_PATH) as f:
        holdout = json.load(f)

    holdout_samples = holdout['holdout_samples']
    print(f"\n{len(holdout_samples)} holdout materials")

    # ===================================================================
    # PART 1: Encode actual holdout samples → decode everything
    # ===================================================================
    # NOTE: These samples ARE in the dataset (they have original_index),
    # but the model was trained with them excluded from training loss.
    # The encoder HAS their correct (elem_idx, magpie, tc) in the cache
    # because they were part of the original data loading.
    # ===================================================================

    print("\n" + "=" * 70)
    print("PART 1: ROUNDTRIP — Encode holdout → Z → Decode everything")
    print("=" * 70)
    print("Each holdout: encode(formula,Tc,Magpie) → Z → decode(Z) → formula+Tc+Magpie")
    print()

    # V12.40: Correct Tc classification bins matching training config
    # tc_class_bins: [0, 10, 50, 100] → 5 classes
    tc_class_names = ['non-SC (0K)', '0-10K', '10-50K', '50-100K', '100K+']
    tc_class_bins = [0, 10, 50, 100]  # Must match TRAIN_CONFIG['tc_class_bins']
    def true_tc_class(tc):
        """Assign Tc bucket matching training script logic."""
        bucket = 0
        for i, edge in enumerate(tc_class_bins):
            if tc > edge:
                bucket = i + 1
        return bucket

    # V12.40: Family name mapping (14-class hierarchical)
    FAMILY_14_NAMES = [
        'NOT_SC',           # 0
        'BCS_CONVENTIONAL', # 1
        'CUPRATE_YBCO',     # 2
        'CUPRATE_LSCO',     # 3
        'CUPRATE_BSCCO',    # 4
        'CUPRATE_TBCCO',    # 5
        'CUPRATE_HBCCO',    # 6
        'CUPRATE_OTHER',    # 7
        'IRON_PNICTIDE',    # 8
        'IRON_CHALCOGENIDE',# 9
        'MGB2_TYPE',        # 10
        'HEAVY_FERMION',    # 11
        'ORGANIC',          # 12
        'OTHER_UNKNOWN',    # 13
    ]
    # Map holdout family strings to expected 14-class indices
    HOLDOUT_FAMILY_TO_14 = {
        'YBCO': 2,
        'LSCO': 3,
        'Hg-cuprate': 6,
        'Tl-cuprate': 5,
        'Bi-cuprate': 4,
        'Iron-based': 8,
        'MgB2': 10,
        'Conventional': 1,
        'Other': 13,
    }

    results = []
    tc_errors = []
    magpie_mses = []
    formula_sims = []

    header = f"{'Family':<14s} {'True Tc':>8s} {'Pred Tc':>9s} {'Err':>7s} {'%Err':>6s} {'MagMSE':>8s} {'Sim':>5s} {'SC?':>4s} {'FamPred':<16s} | Formula"
    print(header)
    print("-" * len(header) + "-" * 40)

    sc_correct = 0
    sc_total = 0
    family_correct = 0
    family_total = 0

    for sample in holdout_samples:
        formula = sample['formula']
        true_tc = sample['Tc']
        family = sample['family']
        orig_idx = sample.get('original_index', None)

        if orig_idx is None:
            print(f"  [{family:<12s}] SKIP — no original_index for {formula}")
            continue

        # V12.40: Full forward pass — runs ALL heads (sc, family, hp, tc_class, etc.)
        idx_t = torch.tensor([orig_idx], dtype=torch.long)
        decoded = full_forward(
            encoder, decoder,
            data['elem_idx'][idx_t],
            data['elem_frac'][idx_t],
            data['elem_mask'][idx_t],
            data['magpie'][idx_t],
            data['tc'][idx_t],
            temperature=0.01,
        )

        # Compare
        pred_tc = decoded['tc_pred_kelvin']
        pred_tc_norm = decoded['tc_pred_norm']
        gen_formula = decoded['formula']
        magpie_pred = decoded['magpie_pred']

        # Tc error (in Kelvin)
        tc_err = pred_tc - true_tc
        pct_err = abs(tc_err) / max(abs(true_tc), 0.1) * 100
        tc_errors.append(abs(tc_err))

        # Magpie MSE
        true_magpie = data['magpie'][orig_idx]
        mag_mse = F.mse_loss(magpie_pred, true_magpie).item()
        magpie_mses.append(mag_mse)

        # Formula similarity
        sim = element_similarity(gen_formula, formula)
        formula_sims.append(sim)

        exact = gen_formula.strip() == formula.strip()

        # SC classification
        sc_str = ''
        if 'sc_pred' in decoded:
            sc_pred_bool = decoded['sc_pred']
            sc_prob = decoded['sc_prob']
            sc_str = f"{sc_prob:.2f}"
            sc_total += 1
            if sc_pred_bool:  # All holdout samples ARE superconductors
                sc_correct += 1

        # Family classification
        family_pred_str = ''
        if 'family_pred_14' in decoded:
            pred_family_idx = decoded['family_pred_14']
            pred_family_name = FAMILY_14_NAMES[pred_family_idx] if pred_family_idx < len(FAMILY_14_NAMES) else f'?{pred_family_idx}'
            true_family_idx = HOLDOUT_FAMILY_TO_14.get(family, -1)
            family_match = pred_family_idx == true_family_idx
            family_pred_str = pred_family_name
            if family_match:
                family_pred_str += ' [OK]'
                family_correct += 1
            else:
                family_pred_str += ' [X]'
            family_total += 1

        # HP prediction
        hp_str = ''
        if 'hp_pred' in decoded:
            hp_str = f"HP={decoded['hp_prob']:.2f}" if decoded['hp_prob'] > 0.1 else ''

        print(f"  [{family:<12s}] {true_tc:8.1f} {pred_tc:9.1f} {tc_err:+7.1f} {pct_err:5.1f}% {mag_mse:8.4f} {sim:5.3f} {sc_str:>4s} {family_pred_str:<16s} | {formula}")
        if not exact:
            print(f"     → Generated: {gen_formula}")
        else:
            print(f"     → Generated: {gen_formula} [EXACT MATCH]")
        if hp_str:
            print(f"     → {hp_str}")

        result_entry = {
            'formula': formula,
            'generated_formula': gen_formula,
            'exact_match': exact,
            'comp_similarity': float(sim),
            'true_tc': true_tc,
            'pred_tc_kelvin': pred_tc,
            'pred_tc_norm': pred_tc_norm,
            'tc_error': tc_err,
            'tc_pct_error': pct_err,
            'magpie_mse': mag_mse,
            'family': family,
            'stoich_pred': decoded['stoich_pred'].tolist(),
        }
        # Tc classification
        if 'tc_class' in decoded:
            result_entry['tc_class_pred'] = decoded['tc_class']
            result_entry['tc_class_true'] = true_tc_class(true_tc)
            result_entry['tc_class_pred_from_tc'] = true_tc_class(pred_tc)
            result_entry['tc_class_probs'] = decoded['tc_class_probs'].tolist()
        # SC classification
        if 'sc_pred' in decoded:
            result_entry['sc_pred'] = bool(decoded['sc_pred'])
            result_entry['sc_prob'] = decoded['sc_prob']
            result_entry['sc_logit'] = decoded['sc_logit']
        # Family classification
        if 'family_pred_14' in decoded:
            result_entry['family_pred_14'] = decoded['family_pred_14']
            result_entry['family_pred_name'] = FAMILY_14_NAMES[decoded['family_pred_14']]
            result_entry['family_true_14'] = HOLDOUT_FAMILY_TO_14.get(family, -1)
            result_entry['family_probs_14'] = decoded['family_probs_14'].tolist()
        # HP prediction
        if 'hp_pred' in decoded:
            result_entry['hp_pred'] = bool(decoded['hp_pred'])
            result_entry['hp_prob'] = decoded['hp_prob']
        results.append(result_entry)

    # ===================================================================
    # PART 2: Perturbation — Small Z noise, check Tc stability
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 2: Z PERTURBATION — Does small noise preserve Tc?")
    print("=" * 70)
    print("For 5 holdouts, perturb Z with noise scale [0.02, 0.05, 0.1]")
    print("and check Tc prediction stability.\n")

    # Pick 5 samples spanning different Tc ranges
    sample_indices = [0, 9, 20, 30, 40]  # Spread across families
    noise_scales = [0.02, 0.05, 0.1, 0.2, 0.5]

    perturbation_results = []
    for si in sample_indices:
        if si >= len(holdout_samples):
            continue
        sample = holdout_samples[si]
        formula = sample['formula']
        true_tc = sample['Tc']
        family = sample['family']
        orig_idx = sample.get('original_index')
        if orig_idx is None:
            continue

        idx_t = torch.tensor([orig_idx], dtype=torch.long)
        enc_out = encoder.encode(
            data['elem_idx'][idx_t].to(DEVICE),
            data['elem_frac'][idx_t].to(DEVICE),
            data['elem_mask'][idx_t].to(DEVICE),
            data['magpie'][idx_t].to(DEVICE),
            data['tc'][idx_t].to(DEVICE),
        )
        z_orig = enc_out['z']  # [1, 2048]

        print(f"  [{family:<12s}] {formula} (Tc={true_tc:.1f}K)")

        for scale in noise_scales:
            # Generate 20 perturbed Z vectors
            noise = torch.randn(20, z_orig.shape[1]).to(DEVICE) * scale
            z_perturbed = z_orig + noise  # [20, 2048]

            # Get Tc predictions for all 20
            dec_out = encoder.decode(z_perturbed)
            tc_preds_norm = dec_out['tc_pred'].detach().cpu().numpy()
            tc_preds_K = np.array([denormalize_tc(t) for t in tc_preds_norm])

            tc_mean = tc_preds_K.mean()
            tc_std_val = tc_preds_K.std()

            print(f"    noise={scale:.2f}: Tc={tc_mean:.1f}±{tc_std_val:.1f}K "
                  f"(err={abs(tc_mean-true_tc):.1f}K)")

            perturbation_results.append({
                'formula': formula, 'true_tc': true_tc, 'noise_scale': scale,
                'tc_mean_K': float(tc_mean), 'tc_std_K': float(tc_std_val),
            })

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    tc_arr = np.array(tc_errors)
    mag_arr = np.array(magpie_mses)
    sim_arr = np.array(formula_sims)

    print(f"\nTc Prediction (from Z):")
    print(f"  Mean absolute error: {tc_arr.mean():.2f} K")
    print(f"  Median absolute error: {np.median(tc_arr):.2f} K")
    print(f"  Max absolute error: {tc_arr.max():.2f} K")
    print(f"  Within 1K: {(tc_arr < 1).sum()}/{len(tc_arr)}")
    print(f"  Within 5K: {(tc_arr < 5).sum()}/{len(tc_arr)}")
    print(f"  Within 10K: {(tc_arr < 10).sum()}/{len(tc_arr)}")

    print(f"\nMagpie Reconstruction:")
    print(f"  Mean MSE: {mag_arr.mean():.6f}")
    print(f"  Median MSE: {np.median(mag_arr):.6f}")
    print(f"  Max MSE: {mag_arr.max():.6f}")

    print(f"\nFormula Reconstruction (roundtrip):")
    print(f"  Mean similarity: {sim_arr.mean():.3f}")
    print(f"  Exact matches: {(sim_arr >= 0.999).sum()}/{len(sim_arr)}")
    print(f"  > 0.95 similarity: {(sim_arr > 0.95).sum()}/{len(sim_arr)}")

    # Tc class accuracy
    class_correct = sum(1 for r in results if r.get('tc_class_pred') == r.get('tc_class_true'))
    class_total = sum(1 for r in results if 'tc_class_pred' in r)
    if class_total > 0:
        print(f"\nTc Classification (5 buckets: {', '.join(tc_class_names)}):")
        print(f"  Accuracy: {class_correct}/{class_total} ({class_correct/class_total*100:.1f}%)")
        # Show per-bucket breakdown
        for bucket in range(5):
            in_bucket = [r for r in results if r.get('tc_class_true') == bucket]
            if in_bucket:
                correct = sum(1 for r in in_bucket if r.get('tc_class_pred') == bucket)
                print(f"  {tc_class_names[bucket]:>12s}: {correct}/{len(in_bucket)} correct")

    # SC classification accuracy
    if sc_total > 0:
        print(f"\nSC Classification (all holdout samples are superconductors):")
        print(f"  Correctly classified as SC: {sc_correct}/{sc_total} ({sc_correct/sc_total*100:.1f}%)")
        sc_probs = [r['sc_prob'] for r in results if 'sc_prob' in r]
        if sc_probs:
            print(f"  SC probability: mean={np.mean(sc_probs):.3f}, min={np.min(sc_probs):.3f}, max={np.max(sc_probs):.3f}")

    # Family classification accuracy
    if family_total > 0:
        print(f"\nFamily Classification (14-class hierarchical):")
        print(f"  Accuracy: {family_correct}/{family_total} ({family_correct/family_total*100:.1f}%)")
        # Per-family breakdown
        for fam_name in ['YBCO', 'LSCO', 'Hg-cuprate', 'Tl-cuprate', 'Bi-cuprate',
                         'Iron-based', 'MgB2', 'Conventional', 'Other']:
            fam_results = [r for r in results if r.get('family') == fam_name and 'family_pred_14' in r]
            if fam_results:
                true_idx = HOLDOUT_FAMILY_TO_14.get(fam_name, -1)
                correct = sum(1 for r in fam_results if r['family_pred_14'] == true_idx)
                preds = [FAMILY_14_NAMES[r['family_pred_14']] for r in fam_results]
                print(f"  {fam_name:>14s}: {correct}/{len(fam_results)} | predictions: {', '.join(preds)}")

    # Save
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoint': str(CHECKPOINT_PATH),
        'roundtrip_results': results,
        'perturbation_results': perturbation_results,
        'summary': {
            'tc_mae': float(tc_arr.mean()),
            'tc_median_ae': float(np.median(tc_arr)),
            'magpie_mean_mse': float(mag_arr.mean()),
            'formula_mean_sim': float(sim_arr.mean()),
            'n_exact_roundtrip': int((sim_arr >= 0.999).sum()),
            'tc_class_accuracy': class_correct / class_total if class_total > 0 else None,
            'sc_accuracy': sc_correct / sc_total if sc_total > 0 else None,
            'family_accuracy': family_correct / family_total if family_total > 0 else None,
        }
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
