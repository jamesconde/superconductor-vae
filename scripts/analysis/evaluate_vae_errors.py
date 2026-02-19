"""
VAE Error Analysis Script
Analyzes what the model is getting wrong to inform training improvements.
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json
import re
from typing import Dict, Optional

from superconductor.models.autoregressive_decoder import (
    tokenize_formula, tokens_to_indices, indices_to_formula,
    VOCAB_SIZE, PAD_IDX, START_IDX, END_IDX, IDX_TO_TOKEN
)
from superconductor.encoders.element_properties import get_atomic_number

# Import config directly from training script (single source of truth)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_v12_clean import MODEL_CONFIG, TRAIN_CONFIG, DATA_PATH, create_models

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
CHECKPOINT_PATH = PROJECT_ROOT / 'outputs/checkpoint_best.pt'
HOLDOUT_PATH = PROJECT_ROOT / 'data/GENERATIVE_HOLDOUT_DO_NOT_TRAIN.json'


def parse_fraction_formula(formula: str) -> Optional[Dict[str, float]]:
    """Parse formula like 'Ag(1/500)Al(499/500)' to element fractions."""
    pattern = r'([A-Z][a-z]?)(?:\((\d+)/(\d+)\)|(\d*\.?\d+))?'
    matches = re.findall(pattern, formula)

    result = {}
    for match in matches:
        element = match[0]
        if not element:
            continue
        if match[1] and match[2]:
            result[element] = float(match[1]) / float(match[2])
        elif match[3]:
            result[element] = float(match[3])
        else:
            result[element] = 1.0
    return result if result else None


def load_data():
    """Load and prepare data matching training script."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples")

    formulas = df['formula'].tolist()
    tc_values = df['Tc'].values

    # Normalize Tc
    tc_mean, tc_std = tc_values.mean(), tc_values.std()
    tc_normalized = (tc_values - tc_mean) / tc_std
    print(f"Tc: mean={tc_mean:.2f}K, std={tc_std:.2f}K")

    # Get Magpie features
    exclude = ['formula', 'Tc', 'composition', 'category', 'compound possible', 'formula_original']
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    magpie_cols = [c for c in numeric_cols if c not in exclude]
    print(f"Found {len(magpie_cols)} Magpie features")

    magpie_features = df[magpie_cols].values
    magpie_mean = magpie_features.mean(axis=0)
    magpie_std = magpie_features.std(axis=0) + 1e-8
    magpie_normalized = (magpie_features - magpie_mean) / magpie_std

    # Tokenize formulas
    print("Tokenizing formulas...")
    max_len = 60
    all_tokens = []
    for formula in formulas:
        tokens = tokenize_formula(formula)
        indices = tokens_to_indices(tokens, max_len=max_len)
        all_tokens.append(indices)
    formula_tokens = torch.stack(all_tokens)

    # Parse element compositions
    print("Parsing element compositions...")
    MAX_ELEMENTS = 12
    element_indices = torch.zeros(len(formulas), MAX_ELEMENTS, dtype=torch.long)
    element_fractions = torch.zeros(len(formulas), MAX_ELEMENTS, dtype=torch.float32)
    element_mask = torch.zeros(len(formulas), MAX_ELEMENTS, dtype=torch.bool)

    for i, formula in enumerate(formulas):
        parsed = parse_fraction_formula(formula)
        if not parsed:
            continue
        total = sum(parsed.values())
        for j, (element, frac) in enumerate(parsed.items()):
            if j >= MAX_ELEMENTS:
                break
            try:
                atomic_num = get_atomic_number(element)
                element_indices[i, j] = atomic_num
                element_fractions[i, j] = frac / total if total > 0 else frac
                element_mask[i, j] = True
            except:
                continue

    # Create tensors
    tc_tensor = torch.tensor(tc_normalized, dtype=torch.float32).unsqueeze(1)
    magpie_tensor = torch.tensor(magpie_normalized, dtype=torch.float32)

    return {
        'formulas': formulas,
        'tokens': formula_tokens,
        'tc': tc_tensor,
        'tc_raw': tc_values,
        'magpie': magpie_tensor,
        'element_indices': element_indices,
        'element_fractions': element_fractions,
        'element_mask': element_mask,
        'tc_mean': tc_mean,
        'tc_std': tc_std,
        'num_magpie': len(magpie_cols),
    }


def load_model(data):
    """Load trained model from checkpoint using training script's create_models."""
    print(f"Loading model from {CHECKPOINT_PATH}...")

    # Use the same create_models function as training
    encoder, decoder = create_models(data['num_magpie'], device)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

    # Handle compiled model state dict
    enc_state = checkpoint['encoder_state_dict']
    dec_state = checkpoint['decoder_state_dict']

    # Strip _orig_mod. prefixes if present
    def strip_prefix(state_dict):
        new_state = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith('_orig_mod.'):
                new_key = new_key[len('_orig_mod.'):]
            new_key = new_key.replace('._orig_mod.', '.')
            new_state[new_key] = v
        return new_state

    if any(k.startswith('_orig_mod.') for k in enc_state.keys()) or \
       any('._orig_mod.' in k for k in dec_state.keys()):
        print("  Stripping compiled prefixes...")
        enc_state = strip_prefix(enc_state)
        dec_state = strip_prefix(dec_state)

    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)

    encoder = encoder.eval()
    decoder = decoder.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best exact match: {checkpoint.get('best_exact', 0)*100:.1f}%")

    return encoder, decoder


def evaluate_autoregressive(encoder, decoder, data, max_samples=None, batch_size=64):
    """
    Evaluate model in TRUE autoregressive mode (no teacher forcing).
    Uses batched KV-cached generation for speed.
    """
    print("\nRunning autoregressive evaluation (batched, KV-cached)...")

    formulas = data['formulas']
    tokens = data['tokens']
    tc = data['tc']
    magpie = data['magpie']
    element_indices = data['element_indices']
    element_fractions = data['element_fractions']
    element_mask = data['element_mask']
    tc_raw = data['tc_raw']

    if max_samples:
        np.random.seed(42)  # Reproducible
        sample_indices = list(np.random.choice(len(formulas), min(max_samples, len(formulas)), replace=False))
    else:
        sample_indices = list(range(len(formulas)))

    results = []
    exact_matches = 0
    total = 0

    encoder.eval()
    decoder.eval()

    # Process in batches for GPU efficiency
    num_batches = (len(sample_indices) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(sample_indices))
            batch_indices = sample_indices[start:end]
            current_batch_size = len(batch_indices)

            # Prepare batch inputs
            elem_idx = element_indices[batch_indices].to(device)
            elem_frac = element_fractions[batch_indices].to(device)
            elem_msk = element_mask[batch_indices].to(device)
            tc_val = tc[batch_indices].to(device)
            magpie_val = magpie[batch_indices].to(device)

            # Encode batch
            enc_outputs = encoder.encode(
                elem_idx, elem_frac, elem_msk,
                magpie_val, tc_val
            )
            z = enc_outputs['z']

            # Decode autoregressively with KV cache (V13.1: no skip connection)
            generated_tokens, _, _ = decoder.generate_with_kv_cache(
                z,
                max_len=60,
                temperature=1.0,
            )

            # Process each sample in batch
            for i, idx in enumerate(batch_indices):
                target_formula = formulas[idx]

                # Convert token indices to formula string
                pred_indices = generated_tokens[i].cpu().tolist()
                pred_formula = indices_to_formula(torch.tensor(pred_indices))

                # Check match
                is_exact = (pred_formula == target_formula)
                if is_exact:
                    exact_matches += 1

                # Tokenize both formulas for comparison
                target_tokens = tokenize_formula(target_formula)
                pred_tokens = tokenize_formula(pred_formula)

                # Simple token error count
                errors = 0
                max_len_cmp = max(len(target_tokens), len(pred_tokens))
                for j in range(max_len_cmp):
                    t = target_tokens[j] if j < len(target_tokens) else None
                    p = pred_tokens[j] if j < len(pred_tokens) else None
                    if t != p:
                        errors += 1

                results.append({
                    'idx': int(idx),
                    'target': target_formula,
                    'predicted': pred_formula,
                    'exact_match': is_exact,
                    'num_errors': errors,
                    'tc': float(tc_raw[idx]),
                    'target_len': len(target_tokens),
                    'pred_len': len(pred_tokens),
                })

                total += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"  Processed {total}/{len(sample_indices)} samples... "
                      f"(current exact: {exact_matches/total*100:.1f}%)")

    exact_rate = exact_matches / total if total > 0 else 0
    print(f"\nExact match rate: {exact_rate*100:.2f}% ({exact_matches}/{total})")

    return results, exact_rate


def extract_elements(formula):
    """Extract element symbols from a formula string."""
    elements = re.findall(r'[A-Z][a-z]?', formula)
    return elements


def analyze_errors(results, data):
    """Analyze error patterns."""
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    errors = [r for r in results if not r['exact_match']]
    correct = [r for r in results if r['exact_match']]

    print(f"\nTotal samples: {len(results)}")
    print(f"Correct: {len(correct)} ({len(correct)/len(results)*100:.1f}%)")
    print(f"Errors: {len(errors)} ({len(errors)/len(results)*100:.1f}%)")

    if len(errors) == 0:
        print("No errors to analyze!")
        return {}

    # Error distribution by number of token errors
    print("\n--- Error Distribution by Token Errors ---")
    error_counts = Counter(r['num_errors'] for r in errors)
    for n_err in sorted(error_counts.keys())[:10]:
        count = error_counts[n_err]
        print(f"  {n_err} token errors: {count} samples ({count/len(errors)*100:.1f}%)")

    # Tc distribution for errors vs correct
    print("\n--- Tc Distribution ---")
    error_tc = [r['tc'] for r in errors]
    correct_tc = [r['tc'] for r in correct]
    print(f"  Correct samples: Tc mean={np.mean(correct_tc):.1f}K, std={np.std(correct_tc):.1f}K")
    print(f"  Error samples:   Tc mean={np.mean(error_tc):.1f}K, std={np.std(error_tc):.1f}K")

    # Tc ranges
    print("\n--- Error Rate by Tc Range ---")
    tc_ranges = [(0, 10), (10, 30), (30, 50), (50, 77), (77, 100), (100, 150), (150, 300)]
    for low, high in tc_ranges:
        in_range = [r for r in results if low <= r['tc'] < high]
        if in_range:
            err_in_range = [r for r in in_range if not r['exact_match']]
            err_rate = len(err_in_range) / len(in_range) * 100
            print(f"  Tc {low:3d}-{high:3d}K: {len(in_range):4d} samples, {err_rate:5.1f}% error rate")

    # Formula length analysis
    print("\n--- Error Rate by Formula Length ---")
    len_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60)]
    for low, high in len_ranges:
        in_range = [r for r in results if low <= r['target_len'] < high]
        if in_range:
            err_in_range = [r for r in in_range if not r['exact_match']]
            err_rate = len(err_in_range) / len(in_range) * 100
            print(f"  Length {low:2d}-{high:2d}: {len(in_range):4d} samples, {err_rate:5.1f}% error rate")

    # Analyze element patterns in errors
    print("\n--- Elements in Error Formulas ---")
    error_elements = Counter()
    correct_elements = Counter()

    for r in errors:
        formula = r['target']
        elements = extract_elements(formula)
        for el in elements:
            error_elements[el] += 1

    for r in correct:
        formula = r['target']
        elements = extract_elements(formula)
        for el in elements:
            correct_elements[el] += 1

    # Find elements with high error rates
    print("\n  Elements with highest error rates:")
    element_error_rates = {}
    for el in set(error_elements.keys()) | set(correct_elements.keys()):
        err_count = error_elements.get(el, 0)
        corr_count = correct_elements.get(el, 0)
        total = err_count + corr_count
        if total >= 10:  # Only consider elements with enough samples
            element_error_rates[el] = (err_count / total, total)

    sorted_elements = sorted(element_error_rates.items(), key=lambda x: -x[1][0])
    for el, (rate, total) in sorted_elements[:15]:
        print(f"    {el:3s}: {rate*100:5.1f}% error rate ({total:4d} samples)")

    # Show some example errors
    print("\n--- Example Errors ---")
    sorted_errors = sorted(errors, key=lambda x: x['num_errors'])

    print("\n  Near misses (1-2 token errors):")
    near_misses = [r for r in sorted_errors if r['num_errors'] <= 2][:10]
    for r in near_misses:
        print(f"    Target: {r['target']}")
        print(f"    Pred:   {r['predicted']}")
        print(f"    Tc={r['tc']:.1f}K, errors={r['num_errors']}")
        print()

    print("\n  Worst errors (many token errors):")
    for r in sorted_errors[-5:]:
        print(f"    Target: {r['target']}")
        print(f"    Pred:   {r['predicted']}")
        print(f"    Tc={r['tc']:.1f}K, errors={r['num_errors']}")
        print()

    return {
        'error_count': len(errors),
        'correct_count': len(correct),
        'error_rate': len(errors) / len(results),
        'element_error_rates': element_error_rates,
        'error_tc_mean': np.mean(error_tc),
        'correct_tc_mean': np.mean(correct_tc),
    }


def main(max_samples=5000):
    print("="*60)
    print("VAE ERROR ANALYSIS")
    print("="*60)

    # Load data
    data = load_data()
    print(f"Total samples: {len(data['formulas'])}")

    # Load model
    encoder, decoder = load_model(data)

    # Run evaluation
    print(f"\nEvaluating on {max_samples} samples...")
    results, exact_rate = evaluate_autoregressive(
        encoder, decoder, data,
        max_samples=max_samples
    )

    # Analyze errors
    analysis = analyze_errors(results, data)

    # Save detailed results
    output_path = PROJECT_ROOT / 'scratch/error_analysis_results.json'
    save_results = {
        'exact_rate': exact_rate,
        'num_samples': len(results),
        'errors': [r for r in results if not r['exact_match']],
        'analysis': {
            'error_rate': analysis.get('error_rate', 0),
            'error_tc_mean': analysis.get('error_tc_mean', 0),
            'correct_tc_mean': analysis.get('correct_tc_mean', 0),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples to evaluate')
    args = parser.parse_args()
    main(max_samples=args.samples)
