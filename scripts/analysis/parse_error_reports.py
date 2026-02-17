#!/usr/bin/env python3
"""Parse error report JSONs for epochs 2764-2812 and extract key metrics."""

import json
import os
from collections import defaultdict, Counter

ERROR_DIR = "/home/james/superconductor-vae/outputs/error reports"
EPOCHS = list(range(2764, 2816, 4))  # 2764, 2768, ..., 2812

results = {}

for epoch in EPOCHS:
    fpath = os.path.join(ERROR_DIR, f"error_analysis_epoch_{epoch}.json")
    if not os.path.exists(fpath):
        print(f"MISSING: {fpath}")
        continue

    with open(fpath, 'r') as f:
        data = json.load(f)

    # --- Summary metrics ---
    summary = data.get('summary', {})

    # --- Error distribution ---
    error_dist = data.get('error_distribution', {})

    # --- SC vs non-SC ---
    sc_stats = data.get('sc_vs_nonsc', data.get('by_category', {}))

    # --- Error types ---
    error_types = data.get('error_types', data.get('error_type_counts', {}))

    # --- Element errors ---
    element_errors = data.get('element_errors', data.get('most_common_error_elements', {}))

    # --- Position analysis ---
    position_analysis = data.get('position_analysis', {})

    # --- Cascade errors ---
    cascade = data.get('cascade_errors', data.get('cascade_analysis', {}))

    results[epoch] = {
        'summary': summary,
        'error_dist': error_dist,
        'sc_stats': sc_stats,
        'error_types': error_types,
        'element_errors': element_errors,
        'position_analysis': position_analysis,
        'cascade': cascade,
        'all_keys': list(data.keys()),
    }

    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}")
    print(f"{'='*60}")
    print(f"Top-level keys: {list(data.keys())}")
    print(f"Summary: {json.dumps(summary, indent=2)[:2000]}")
    print(f"Error distribution: {json.dumps(error_dist, indent=2)[:1000]}")
    print(f"SC vs non-SC: {json.dumps(sc_stats, indent=2)[:1000]}")
    print(f"Error types: {json.dumps(error_types, indent=2)[:1000]}")
    print(f"Element errors (truncated): {json.dumps(element_errors, indent=2)[:500]}")
    print(f"Position analysis (truncated): {json.dumps(position_analysis, indent=2)[:500]}")
    print(f"Cascade: {json.dumps(cascade, indent=2)[:500]}")

# Now check the first file structure in detail
print("\n\n" + "="*80)
print("FIRST FILE FULL STRUCTURE EXPLORATION")
print("="*80)
first_epoch = EPOCHS[0]
fpath = os.path.join(ERROR_DIR, f"error_analysis_epoch_{first_epoch}.json")
with open(fpath, 'r') as f:
    data = json.load(f)

for key in data.keys():
    val = data[key]
    if isinstance(val, dict):
        print(f"\n--- {key} (dict, {len(val)} keys) ---")
        for k2, v2 in list(val.items())[:10]:
            if isinstance(v2, (dict, list)):
                print(f"  {k2}: {type(v2).__name__}, len={len(v2)}, preview={json.dumps(v2, indent=2)[:300]}")
            else:
                print(f"  {k2}: {v2}")
    elif isinstance(val, list):
        print(f"\n--- {key} (list, {len(val)} items) ---")
        if len(val) > 0:
            print(f"  First item: {json.dumps(val[0], indent=2)[:500]}")
    else:
        print(f"\n--- {key}: {val}")
