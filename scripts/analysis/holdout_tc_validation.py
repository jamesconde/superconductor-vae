#!/usr/bin/env python3
"""
Holdout Tc/Magpie Validation — "The Full Enchilada"
=====================================================

For each of the 45 holdout materials:
1. Encode the actual holdout sample through the encoder → Z
2. encoder.decode(Z) → tc_pred, magpie_pred, attended_input
3. decoder.generate(Z, skip, stoich) → formula string
4. Compare: tc_pred vs true Tc, magpie_pred vs true Magpie, formula vs original

This proves that Z encodes the complete material information, and
manipulating Z gives us formula + Tc + Magpie simultaneously.

Usage:
    cd /home/james/superconductor-vae
    PYTHONPATH=src python -u scripts/analysis/holdout_tc_validation.py
    PYTHONPATH=src python -u scripts/analysis/holdout_tc_validation.py --checkpoint outputs/best_checkpoints/checkpoint_best_V12.38_colab.pt
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

from superconductor.models.attention_vae import FullMaterialsVAE
from superconductor.models.autoregressive_decoder import (
    EnhancedTransformerDecoder, IDX_TO_TOKEN,
    PAD_IDX, START_IDX, END_IDX,
)
from superconductor.data.canonical_ordering import CanonicalOrderer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/analysis/ -> scripts/ -> project root
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

    # V12.28: Net2Net weight transfer for old tc_head → new tc_proj/tc_res_block/tc_out
    has_old_tc_head = any('tc_head.' in k for k in enc_state)
    has_new_tc_proj = any('tc_proj.' in k for k in enc_state)
    if has_old_tc_head and not has_new_tc_proj:
        encoder.upgrade_tc_head_from_checkpoint(enc_state)
        print("  Applied Net2Net weight transfer for Tc head upgrade")

    encoder.load_state_dict(enc_state, strict=False)

    dec_state_raw = checkpoint.get('decoder_state_dict', {})
    dec_state = {k.replace('_orig_mod.', ''): v for k, v in dec_state_raw.items()}

    # Detect stoich_input_dim from checkpoint weights
    stoich_weight_key = 'stoich_to_memory.0.weight'
    if stoich_weight_key in dec_state:
        stoich_dim = dec_state[stoich_weight_key].shape[1]
        if stoich_dim == 37:
            max_elements = 12  # V12.38: 12*3+1=37
        else:
            max_elements = stoich_dim - 1  # Pre-V12.38: 12+1=13, so max_elements=12
    else:
        max_elements = 12

    decoder = EnhancedTransformerDecoder(
        latent_dim=2048, d_model=512, nhead=8, num_layers=12,
        dim_feedforward=2048, dropout=0.1, max_len=60,
        n_memory_tokens=16, encoder_skip_dim=256,
        use_skip_connection=True, use_stoich_conditioning=True,
        max_elements=max_elements, n_stoich_tokens=4,
    ).to(DEVICE)
    decoder.load_state_dict(dec_state, strict=False)

    encoder.eval()
    decoder.eval()
    epoch = checkpoint.get('epoch', '?')
    print(f"  Loaded epoch {epoch}, magpie_dim={magpie_dim}, numden_head={has_numden_head}")
    print(f"  Decoder stoich_input_dim={decoder.stoich_to_memory[0].in_features}")
    return encoder, decoder, magpie_dim, has_numden_head


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
def full_decode(encoder, decoder, z, has_numden_head=False, temperature=0.01):
    """Decode Z → formula + Tc + Magpie + stoichiometry (the full enchilada).

    Returns dict with all predictions the model makes from a Z vector.
    """
    z_dev = z.to(DEVICE)
    if z_dev.dim() == 1:
        z_dev = z_dev.unsqueeze(0)

    # encoder.decode gives us Tc and Magpie predictions from Z
    dec_out = encoder.decode(z_dev)
    tc_pred = dec_out['tc_pred'][0].item()
    magpie_pred = dec_out['magpie_pred'][0].cpu()
    tc_class_logits = dec_out.get('tc_class_logits', None)

    # V12.38: Assemble stoich_pred with numden conditioning
    fraction_output = encoder.fraction_head(z_dev)
    fraction_pred = fraction_output[:, :encoder.max_elements]  # [batch, 12]
    element_count_pred = fraction_output[:, -1]                # [batch]

    if has_numden_head and hasattr(encoder, 'numden_head'):
        numden_pred = encoder.numden_head(z_dev)  # [batch, 24]
        stoich_pred = torch.cat([fraction_pred, numden_pred, element_count_pred.unsqueeze(-1)], dim=-1)
    else:
        stoich_pred = torch.cat([fraction_pred, element_count_pred.unsqueeze(-1)], dim=-1)

    stoich_pred_cpu = stoich_pred[0].cpu()

    # Skip connection for formula decoder
    encoder_skip = dec_out['attended_input']

    # Formula generation
    generated, log_probs, entropy = decoder.generate_with_kv_cache(
        z=z_dev, encoder_skip=encoder_skip, stoich_pred=stoich_pred,
        temperature=temperature,
    )
    formula = tokens_to_formula(generated[0])

    # Denormalize Tc from z-score log1p space to Kelvin
    tc_kelvin = denormalize_tc(tc_pred)

    result = {
        'formula': formula,
        'tc_pred_norm': tc_pred,
        'tc_pred_kelvin': tc_kelvin,
        'magpie_pred': magpie_pred,
        'stoich_pred': stoich_pred_cpu,
        'log_prob': log_probs[0].sum().item() if log_probs is not None else None,
    }
    if tc_class_logits is not None:
        probs = torch.softmax(tc_class_logits[0].cpu(), dim=-1)
        result['tc_class_probs'] = probs
        result['tc_class'] = probs.argmax().item()
    return result


def main():
    parser = argparse.ArgumentParser(description='Holdout Tc/Magpie Validation')
    parser.add_argument('--checkpoint', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'checkpoint_best.pt'),
                        help='Path to checkpoint file')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Output path named after checkpoint
    ckpt_stem = checkpoint_path.stem
    output_path = PROJECT_ROOT / 'outputs' / f'holdout_tc_validation_{ckpt_stem}.json'

    print("=" * 70)
    print("HOLDOUT FULL ENCHILADA VALIDATION")
    print("(Formula + Tc + Magpie from Z)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_path}")

    encoder, decoder, magpie_dim, has_numden_head = load_models(checkpoint_path)
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

    tc_class_names = ['0-20K', '20-40K', '40-77K', '77K+']
    def true_tc_class(tc):
        if tc < 20: return 0
        elif tc < 40: return 1
        elif tc < 77: return 2
        else: return 3

    results = []
    tc_errors = []
    magpie_mses = []
    formula_sims = []

    header = f"{'Family':<14s} {'True Tc':>8s} {'Pred Tc':>9s} {'Err':>7s} {'%Err':>6s} {'MagMSE':>8s} {'Sim':>5s} | Formula"
    print(header)
    print("-" * len(header) + "-" * 40)

    for sample in holdout_samples:
        formula = sample['formula']
        true_tc = sample['Tc']
        family = sample['family']
        orig_idx = sample.get('original_index', None)

        if orig_idx is None:
            print(f"  [{family:<12s}] SKIP — no original_index for {formula}")
            continue

        # Encode the holdout sample
        idx_t = torch.tensor([orig_idx], dtype=torch.long)
        enc_out = encoder.encode(
            data['elem_idx'][idx_t].to(DEVICE),
            data['elem_frac'][idx_t].to(DEVICE),
            data['elem_mask'][idx_t].to(DEVICE),
            data['magpie'][idx_t].to(DEVICE),
            data['tc'][idx_t].to(DEVICE),
        )
        z = enc_out['z']  # [1, 2048]

        # Decode EVERYTHING from Z
        decoded = full_decode(encoder, decoder, z, has_numden_head=has_numden_head, temperature=0.01)

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

        print(f"  [{family:<12s}] {true_tc:8.1f} {pred_tc:9.1f} {tc_err:+7.1f} {pct_err:5.1f}% {mag_mse:8.4f} {sim:5.3f} | {formula}")
        if not exact:
            print(f"     → Generated: {gen_formula}")
        else:
            print(f"     → Generated: {gen_formula} [EXACT MATCH]")

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
        if 'tc_class' in decoded:
            result_entry['tc_class_pred'] = decoded['tc_class']
            result_entry['tc_class_true'] = true_tc_class(true_tc)
            result_entry['tc_class_pred_from_tc'] = true_tc_class(pred_tc)
            result_entry['tc_class_probs'] = decoded['tc_class_probs'].tolist()
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
            # Denormalize to Kelvin
            tc_preds_K = np.array([denormalize_tc(t) for t in tc_preds_norm])

            tc_mean = tc_preds_K.mean()
            tc_std_val = tc_preds_K.std()

            # Decode a couple to see formulas
            sample_formulas = []
            for zi in range(min(3, len(z_perturbed))):
                decoded = full_decode(encoder, decoder, z_perturbed[zi:zi+1], has_numden_head=has_numden_head)
                sample_formulas.append(decoded['formula'])

            print(f"    noise={scale:.2f}: Tc={tc_mean:.1f}±{tc_std_val:.1f}K "
                  f"(err={abs(tc_mean-true_tc):.1f}K) | {sample_formulas[0]}")

            perturbation_results.append({
                'formula': formula, 'true_tc': true_tc, 'noise_scale': scale,
                'tc_mean_K': float(tc_mean), 'tc_std_K': float(tc_std_val),
                'sample_formulas': sample_formulas,
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
        print(f"\nTc Class Accuracy: {class_correct}/{class_total} ({class_correct/class_total*100:.1f}%)")

    # Save
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoint': str(checkpoint_path),
        'roundtrip_results': results,
        'perturbation_results': perturbation_results,
        'summary': {
            'tc_mae': float(tc_arr.mean()),
            'tc_median_ae': float(np.median(tc_arr)),
            'magpie_mean_mse': float(mag_arr.mean()),
            'formula_mean_sim': float(sim_arr.mean()),
            'n_exact_roundtrip': int((sim_arr >= 0.999).sum()),
            'tc_class_accuracy': class_correct / class_total if class_total > 0 else None,
        }
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
