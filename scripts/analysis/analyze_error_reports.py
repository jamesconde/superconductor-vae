"""Analyze error reports from epochs 2764-2812 for training refinement insights."""
import json
import os
from collections import Counter, defaultdict
import re

ERROR_DIR = "/home/james/superconductor-vae/outputs/error reports"

# Latest batch: epochs 2764-2812 (every 4 epochs)
LATEST_EPOCHS = list(range(2764, 2816, 4))  # 2764, 2768, ..., 2812

def load_report(epoch):
    path = os.path.join(ERROR_DIR, f"error_analysis_epoch_{epoch}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def analyze_trends(reports):
    """Track key metrics across epochs."""
    print("=" * 80)
    print("TREND ANALYSIS: Epochs 2764-2812")
    print("=" * 80)

    print(f"\n{'Epoch':>6} | {'Exact%':>7} | {'Errors':>6} | {'SC_Err':>6} | {'NonSC':>6} | {'AvgErr/Fail':>11} | {'z_norm':>7} | {'Tc_R2':>8} | {'Tc_R2_K':>8}")
    print("-" * 95)

    for r in reports:
        e = r['epoch']
        exact = r['exact_match_pct']
        total_err = r['total_errors']
        sc_err = r['errors_by_type']['sc_errors']
        nsc_err = r['errors_by_type']['non_sc_errors']
        avg_err = r['avg_errors_per_failed']
        z_norm = r['z_diagnostics']['z_norm_overall']['mean']
        tc_r2 = r['z_diagnostics']['tc_r2']
        # V12.34: Kelvin-space R² (only available in newer reports)
        tc_r2_k = r['z_diagnostics'].get('tc_r2_kelvin')
        tc_r2_k_str = f"{tc_r2_k:>8.6f}" if tc_r2_k is not None else f"{'N/A':>8}"
        print(f"{e:>6} | {exact:>7.2f} | {total_err:>6} | {sc_err:>6} | {nsc_err:>6} | {avg_err:>11.2f} | {z_norm:>7.2f} | {tc_r2:>8.6f} | {tc_r2_k_str}")

    first, last = reports[0], reports[-1]
    print(f"\nDelta (2764->2812):")
    print(f"  Exact match: {first['exact_match_pct']:.2f}% -> {last['exact_match_pct']:.2f}% (Δ {last['exact_match_pct'] - first['exact_match_pct']:+.2f}%)")
    print(f"  Total errors: {first['total_errors']} -> {last['total_errors']} (Δ {last['total_errors'] - first['total_errors']:+d})")
    print(f"  SC errors: {first['errors_by_type']['sc_errors']} -> {last['errors_by_type']['sc_errors']} (Δ {last['errors_by_type']['sc_errors'] - first['errors_by_type']['sc_errors']:+d})")
    print(f"  NonSC errors: {first['errors_by_type']['non_sc_errors']} -> {last['errors_by_type']['non_sc_errors']} (Δ {last['errors_by_type']['non_sc_errors'] - first['errors_by_type']['non_sc_errors']:+d})")
    print(f"  Avg errors/failed: {first['avg_errors_per_failed']:.2f} -> {last['avg_errors_per_failed']:.2f}")

def analyze_error_distribution(reports):
    """Analyze how errors distribute (0, 1, 2, 3, more)."""
    print("\n" + "=" * 80)
    print("ERROR DISTRIBUTION ACROSS EPOCHS")
    print("=" * 80)

    print(f"\n{'Epoch':>6} | {'0 err':>6} | {'1 err':>6} | {'2 err':>6} | {'3 err':>6} | {'>3 err':>6} | {'% 0-err':>7} | {'% >3':>6}")
    print("-" * 80)

    for r in reports:
        e = r['epoch']
        d = r['error_distribution']
        total = r['total_samples']
        pct_0 = d['0'] / total * 100
        pct_more = d['more'] / total * 100
        print(f"{e:>6} | {d['0']:>6} | {d['1']:>6} | {d['2']:>6} | {d['3']:>6} | {d['more']:>6} | {pct_0:>7.1f} | {pct_more:>6.1f}")

def analyze_z_norm_quartiles(reports):
    """Analyze how z_norm quartiles relate to errors."""
    print("\n" + "=" * 80)
    print("Z-NORM QUARTILE ANALYSIS (Latest: Epoch 2812)")
    print("=" * 80)

    r = reports[-1]
    quartiles = r['z_diagnostics']['errors_by_z_norm_quartile']

    print(f"\n{'Quartile':>12} | {'Samples':>7} | {'Exact%':>7} | {'AvgErr':>7} | {'z_norm range':>25}")
    print("-" * 70)
    for q_name, q_data in quartiles.items():
        z_range = f"[{q_data['z_norm_range'][0]:.1f}, {q_data['z_norm_range'][1]:.1f}]"
        print(f"{q_name:>12} | {q_data['n_samples']:>7} | {q_data['exact_pct']:>7.1f} | {q_data['avg_errors']:>7.2f} | {z_range:>25}")

    # Track Q4 (highest z_norm) across epochs
    print(f"\nQ4 (highest z_norm) trend across epochs:")
    print(f"{'Epoch':>6} | {'Q4 Exact%':>9} | {'Q4 AvgErr':>9} | {'Q1 Exact%':>9}")
    print("-" * 45)
    for r in reports:
        q4 = r['z_diagnostics']['errors_by_z_norm_quartile']['Q4_highest']
        q1 = r['z_diagnostics']['errors_by_z_norm_quartile']['Q1_lowest']
        print(f"{r['epoch']:>6} | {q4['exact_pct']:>9.1f} | {q4['avg_errors']:>9.2f} | {q1['exact_pct']:>9.1f}")

def analyze_seq_len_buckets(reports):
    """Analyze errors by sequence length."""
    print("\n" + "=" * 80)
    print("SEQUENCE LENGTH ANALYSIS (Latest: Epoch 2812)")
    print("=" * 80)

    r = reports[-1]
    buckets = r['z_diagnostics']['errors_by_seq_len_bucket']

    print(f"\n{'SeqLen':>10} | {'Samples':>7} | {'Exact%':>7} | {'AvgErr':>7} | {'Avg z_norm':>10}")
    print("-" * 55)
    for b_name, b_data in buckets.items():
        print(f"{b_name:>10} | {b_data['n_samples']:>7} | {b_data['exact_pct']:>7.1f} | {b_data['avg_errors']:>7.2f} | {b_data['avg_z_norm']:>10.2f}")

    # Track short vs long across epochs
    print(f"\nSeq length bucket trends:")
    print(f"{'Epoch':>6} | {'1-10 Exact%':>11} | {'11-20':>7} | {'21-30':>7} | {'31-40':>7} | {'41-60':>7}")
    print("-" * 60)
    for r in reports:
        b = r['z_diagnostics']['errors_by_seq_len_bucket']
        vals = []
        for key in ['1-10', '11-20', '21-30', '31-40', '41-60']:
            if key in b:
                vals.append(f"{b[key]['exact_pct']:>7.1f}")
            else:
                vals.append(f"{'N/A':>7}")
        print(f"{r['epoch']:>6} | {vals[0]:>11} | {' | '.join(vals[1:])}")

def analyze_correlations(reports):
    """Analyze what correlates with errors."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS (Latest: Epoch 2812)")
    print("=" * 80)

    r = reports[-1]
    z = r['z_diagnostics']

    correlations = [
        ("z_norm vs errors", z['corr_z_norm_vs_errors']),
        ("seq_len vs errors", z['corr_seq_len_vs_errors']),
        ("tc_true vs errors", z['corr_tc_true_vs_errors']),
        ("n_elements vs errors", z['corr_n_elements_vs_errors']),
        ("stoich_mse vs errors", z['corr_stoich_mse_vs_errors']),
        ("tc_error vs formula_errors", z['corr_tc_error_vs_formula_errors']),
        ("magpie_mse vs errors", z['corr_magpie_mse_vs_errors']),
    ]

    print(f"\n{'Correlation':>30} | {'Value':>8} | Strength")
    print("-" * 60)
    for name, val in sorted(correlations, key=lambda x: abs(x[1]), reverse=True):
        strength = "STRONG" if abs(val) > 0.3 else "MODERATE" if abs(val) > 0.15 else "WEAK"
        print(f"{name:>30} | {val:>8.4f} | {strength}")

    # Track key correlations across epochs
    print(f"\nCorrelation trends:")
    print(f"{'Epoch':>6} | {'z_norm':>8} | {'seq_len':>8} | {'tc_true':>8} | {'n_elem':>8}")
    print("-" * 48)
    for r in reports:
        z = r['z_diagnostics']
        print(f"{r['epoch']:>6} | {z['corr_z_norm_vs_errors']:>8.4f} | {z['corr_seq_len_vs_errors']:>8.4f} | {z['corr_tc_true_vs_errors']:>8.4f} | {z['corr_n_elements_vs_errors']:>8.4f}")

def analyze_sc_vs_nonsc(reports):
    """Analyze SC vs non-SC error patterns."""
    print("\n" + "=" * 80)
    print("SC vs NON-SC ERROR ANALYSIS")
    print("=" * 80)

    r = reports[-1]
    z = r['z_diagnostics']

    print(f"\nTc MAE (superconductors only): {z['tc_mae_sc_only']:.6f}")
    print(f"Tc MAE (non-superconductors):  {z['tc_mae_non_sc']:.6f}")
    print(f"Tc R²: {z['tc_r2']:.6f}")
    print(f"Magpie MSE (exact matches):    {z['magpie_mse_exact']:.6f}")
    print(f"Magpie MSE (errors):           {z['magpie_mse_errors']:.6f}")
    print(f"Stoich MSE (exact matches):    {z['stoich_mse_exact']:.6f}")
    print(f"Stoich MSE (errors):           {z['stoich_mse_errors']:.6f}")

    total_samples = r['total_samples']
    sc_err = r['errors_by_type']['sc_errors']
    nsc_err = r['errors_by_type']['non_sc_errors']
    total_err = r['total_errors']

    print(f"\nSC errors: {sc_err}/{total_err} ({sc_err/total_err*100:.1f}% of all errors)")
    print(f"NonSC errors: {nsc_err}/{total_err} ({nsc_err/total_err*100:.1f}% of all errors)")

def analyze_tc_ranges(reports):
    """V12.34: Analyze Tc prediction quality by temperature range in Kelvin."""
    print("\n" + "=" * 80)
    print("Tc RANGE ANALYSIS (Kelvin-Space R² and MAE)")
    print("=" * 80)

    r = reports[-1]
    z = r['z_diagnostics']
    tc_ranges = z.get('errors_by_tc_range', {})

    if not tc_ranges:
        print("  No Tc range data available (pre-V12.34 report)")
        return

    # Check if new metrics are available
    has_kelvin_metrics = any('tc_r2' in v for v in tc_ranges.values())

    if has_kelvin_metrics:
        print(f"\n{'Range':>10} | {'Samples':>7} | {'Exact%':>7} | {'AvgErr':>7} | {'Tc R²':>8} | {'MAE (K)':>8} | {'MaxErr(K)':>9}")
        print("-" * 75)
        for label, data in tc_ranges.items():
            r2 = data.get('tc_r2')
            mae = data.get('tc_mae_kelvin')
            max_err = data.get('tc_max_error_kelvin')
            r2_str = f"{r2:>8.4f}" if r2 is not None else f"{'N/A':>8}"
            mae_str = f"{mae:>8.2f}" if mae is not None else f"{'N/A':>8}"
            max_str = f"{max_err:>9.2f}" if max_err is not None else f"{'N/A':>9}"
            print(f"{label:>10} | {data['n_samples']:>7} | {data['exact_pct']:>7.1f} | {data['avg_errors']:>7.2f} | {r2_str} | {mae_str} | {max_str}")
    else:
        print(f"\n{'Range':>10} | {'Samples':>7} | {'Exact%':>7} | {'AvgErr':>7} | {'z_norm':>7}")
        print("-" * 50)
        for label, data in tc_ranges.items():
            print(f"{label:>10} | {data['n_samples']:>7} | {data['exact_pct']:>7.1f} | {data['avg_errors']:>7.2f} | {data['avg_z_norm']:>7.2f}")

    # Overall Kelvin metrics
    r2_k = z.get('tc_r2_kelvin')
    mae_k = z.get('tc_mae_kelvin_overall')
    if r2_k is not None:
        print(f"\n  Overall Tc R² (Kelvin): {r2_k:.6f}")
        print(f"  Overall Tc MAE (Kelvin): {mae_k:.2f}K")
    else:
        print(f"\n  (Kelvin-space R²/MAE not available — use V12.34+ reports)")

    # Track across epochs if available
    has_kelvin_trend = any('tc_r2_kelvin' in r['z_diagnostics'] for r in reports)
    if has_kelvin_trend:
        print(f"\nTc Kelvin R² trend across epochs:")
        print(f"{'Epoch':>6} | {'Tc R² (norm)':>12} | {'Tc R² (K)':>10} | {'Tc MAE (K)':>10}")
        print("-" * 45)
        for r in reports:
            z = r['z_diagnostics']
            r2_norm = z['tc_r2']
            r2_k = z.get('tc_r2_kelvin')
            mae_k = z.get('tc_mae_kelvin_overall')
            r2_k_str = f"{r2_k:>10.6f}" if r2_k is not None else f"{'N/A':>10}"
            mae_k_str = f"{mae_k:>10.2f}" if mae_k is not None else f"{'N/A':>10}"
            print(f"{r['epoch']:>6} | {r2_norm:>12.6f} | {r2_k_str} | {mae_k_str}")


def analyze_error_patterns(reports):
    """Deep analysis of the worst error cases - what patterns emerge."""
    print("\n" + "=" * 80)
    print("ERROR PATTERN ANALYSIS (Epoch 2812 - Worst Cases)")
    print("=" * 80)

    r = reports[-1]
    records = r['error_records']

    # Categorize error types
    fraction_simplification = 0  # e.g., 54/25 -> 2, getting the right value but wrong representation
    element_order_swap = 0       # Elements appear but in wrong order
    truncation = 0               # Generated sequence ends too early
    wrong_fraction_digits = 0    # Right structure but wrong digits in fractions
    element_substitution = 0     # Wrong element entirely

    # Analyze n_elements distribution of errors
    n_elem_counts = Counter()
    n_elem_error_sum = defaultdict(int)

    # SC vs non-SC error severity
    sc_errors_list = []
    nsc_errors_list = []

    # Track specific fraction error patterns
    fraction_pattern = re.compile(r'(\d+)/(\d+)')

    early_end_count = 0  # How many errors involve premature <END>
    late_end_count = 0   # Generated goes past target

    for rec in records:
        n_elem = rec.get('n_elements', 0)
        n_elem_counts[n_elem] += 1
        n_elem_error_sum[n_elem] += rec['n_errors']

        if rec['is_sc']:
            sc_errors_list.append(rec['n_errors'])
        else:
            nsc_errors_list.append(rec['n_errors'])

        # Check for truncation/early end patterns
        for m in rec['mismatches']:
            if '-><END>' in m and '<END>->' not in m:
                early_end_count += 1
            if '<END>->' in m:
                late_end_count += 1

        # Analyze target vs generated
        target = rec['target']
        generated = rec['generated']

        # Check fraction precision: are fractions close but different denominators?
        target_fracs = fraction_pattern.findall(target)
        gen_fracs = fraction_pattern.findall(generated)

        for tf in target_fracs:
            t_val = int(tf[0]) / int(tf[1])
            for gf in gen_fracs:
                g_val = int(gf[0]) / int(gf[1])
                if abs(t_val - g_val) < 0.05 and tf != gf:
                    wrong_fraction_digits += 1
                    break

    print(f"\nErrors by number of elements in formula:")
    print(f"{'n_elements':>10} | {'Count':>6} | {'Avg Errors':>10}")
    print("-" * 35)
    for n in sorted(n_elem_counts.keys()):
        avg = n_elem_error_sum[n] / n_elem_counts[n]
        print(f"{n:>10} | {n_elem_counts[n]:>6} | {avg:>10.1f}")

    print(f"\nSC error severity: avg={sum(sc_errors_list)/len(sc_errors_list):.1f} errors/sample ({len(sc_errors_list)} failed SC samples)")
    if nsc_errors_list:
        print(f"NonSC error severity: avg={sum(nsc_errors_list)/len(nsc_errors_list):.1f} errors/sample ({len(nsc_errors_list)} failed nonSC samples)")

    print(f"\nMismatch patterns across all error records:")
    print(f"  Early <END> tokens (truncation): {early_end_count}")
    print(f"  Late continuation past <END>:    {late_end_count}")
    print(f"  Fraction near-misses (close values, different representation): {wrong_fraction_digits}")

    # Analyze the TYPES of position errors
    # Are errors clustered at beginning, middle, or end of sequences?
    pos_errors = defaultdict(int)
    total_mismatches = 0
    for rec in records:
        seq_len = rec['seq_len']
        for m in rec['mismatches']:
            pos_match = re.match(r'pos(\d+):', m)
            if pos_match:
                pos = int(pos_match.group(1))
                # Normalize position to 0-1 range
                if seq_len > 0:
                    normalized = pos / seq_len
                    if normalized < 0.33:
                        pos_errors['early (0-33%)'] += 1
                    elif normalized < 0.66:
                        pos_errors['middle (33-66%)'] += 1
                    else:
                        pos_errors['late (66-100%)'] += 1
                total_mismatches += 1

    print(f"\nError position distribution (normalized within each sequence):")
    for region in ['early (0-33%)', 'middle (33-66%)', 'late (66-100%)']:
        count = pos_errors[region]
        pct = count / total_mismatches * 100 if total_mismatches > 0 else 0
        print(f"  {region}: {count} ({pct:.1f}%)")

def analyze_fraction_precision_deep(reports):
    """Deep dive into fraction representation errors."""
    print("\n" + "=" * 80)
    print("FRACTION PRECISION DEEP DIVE (Epoch 2812)")
    print("=" * 80)

    r = reports[-1]
    records = r['error_records']

    fraction_pattern = re.compile(r'\((\d+)/(\d+)\)')

    # Compare denominators between target and generated
    target_denoms = Counter()
    gen_denoms = Counter()

    # Track cases where model simplifies fractions
    simplification_cases = []

    for rec in records:
        target = rec['target']
        generated = rec['generated']

        t_fracs = fraction_pattern.findall(target)
        g_fracs = fraction_pattern.findall(generated)

        for num, den in t_fracs:
            target_denoms[int(den)] += 1
        for num, den in g_fracs:
            gen_denoms[int(den)] += 1

        # Check if generated simplifies target fractions
        for t_num, t_den in t_fracs:
            t_val = int(t_num) / int(t_den)
            for g_num, g_den in g_fracs:
                g_val = int(g_num) / int(g_den)
                if abs(t_val - g_val) < 0.02 and int(t_den) != int(g_den):
                    simplification_cases.append({
                        'target_frac': f"{t_num}/{t_den}",
                        'gen_frac': f"{g_num}/{g_den}",
                        'target_val': t_val,
                        'gen_val': g_val,
                        'target_den': int(t_den),
                        'gen_den': int(g_den),
                    })

    print(f"\nTop 15 target denominators:")
    for den, count in target_denoms.most_common(15):
        print(f"  /{den}: {count} occurrences")

    print(f"\nTop 15 generated denominators:")
    for den, count in gen_denoms.most_common(15):
        print(f"  /{den}: {count} occurrences")

    print(f"\nFraction simplification/rewriting cases: {len(simplification_cases)}")
    if simplification_cases:
        # Group by pattern
        den_changes = Counter()
        for case in simplification_cases:
            key = f"/{case['target_den']} -> /{case['gen_den']}"
            den_changes[key] += 1

        print(f"Top denominator changes:")
        for change, count in den_changes.most_common(20):
            print(f"  {change}: {count}")

def analyze_element_level(reports):
    """Which elements are most commonly involved in errors?"""
    print("\n" + "=" * 80)
    print("ELEMENT-LEVEL ERROR ANALYSIS (Epoch 2812)")
    print("=" * 80)

    r = reports[-1]
    records = r['error_records']

    # Count which elements appear in targets of failed samples
    element_pattern = re.compile(r'([A-Z][a-z]?)')

    elem_in_failed = Counter()
    elem_in_all_targets = Counter()

    for rec in records:
        target_elems = set(element_pattern.findall(rec['target']))
        for elem in target_elems:
            # Filter out things that look like elements but are tokens
            if elem in ['O', 'H', 'N', 'C', 'S', 'P', 'F', 'I', 'B', 'K', 'U', 'V', 'W', 'Y',
                        'Ba', 'Bi', 'Ca', 'Cu', 'Sr', 'La', 'Nd', 'Gd', 'Pb', 'Tl', 'Hg', 'Fe',
                        'Mg', 'Al', 'Si', 'Sn', 'Sb', 'Te', 'In', 'Zn', 'Ni', 'Co', 'Mn', 'Ti',
                        'Zr', 'Nb', 'Mo', 'Ta', 'Hf', 'Ce', 'Pr', 'Sm', 'Eu', 'Tb', 'Dy', 'Ho',
                        'Er', 'Yb', 'Lu', 'Sc', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Ag', 'Cd',
                        'Cs', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Li', 'Na', 'Be', 'Cl', 'Cr', 'Ru',
                        'Rh', 'Pd', 'Th']:
                elem_in_failed[elem] += 1

    print(f"\nMost common elements in failed formulas:")
    for elem, count in elem_in_failed.most_common(20):
        print(f"  {elem}: {count} failed samples")

def analyze_physics_z_blocks(reports):
    """V12.35: Analyze per-block Physics Z norms — which blocks predict errors?"""
    print("\n" + "=" * 80)
    print("PHYSICS Z BLOCK DIAGNOSTICS")
    print("=" * 80)

    # Use latest report that has z_block_diagnostics
    for r in reversed(reports):
        zd = r.get('z_diagnostics', {})
        if 'z_block_diagnostics' in zd:
            break
    else:
        print("  No z_block_diagnostics found in any report (need V12.35+ error reports)")
        return

    block_diag = zd['z_block_diagnostics']
    ranked = zd.get('z_block_corr_ranked', [])

    print(f"\nEpoch {r['epoch']}: Per-block Z norm comparison (exact vs error samples)")
    print(f"{'Block':>16} | {'Overall':>10} | {'Exact':>10} | {'Error':>10} | {'Gap':>8} | {'Corr→Err':>9}")
    print("-" * 75)

    for block_name in ['gl', 'bcs', 'eliashberg', 'unconventional', 'structural',
                       'electronic', 'thermodynamic', 'compositional', 'cobordism',
                       'ratios', 'magpie', 'discovery']:
        stats = block_diag.get(block_name)
        if stats is None:
            continue
        overall = stats['overall']['mean']
        exact_mean = stats.get('exact', {}).get('mean', float('nan'))
        error_mean = stats.get('error', {}).get('mean', float('nan'))
        gap = stats.get('exact_error_gap', float('nan'))
        corr = stats.get('corr_vs_errors', 0.0)
        print(f"{block_name:>16} | {overall:>10.3f} | {exact_mean:>10.3f} | {error_mean:>10.3f} | {gap:>+8.3f} | {corr:>+9.4f}")

    # Highlight blocks most correlated with errors
    if ranked:
        print(f"\nBlocks ranked by |correlation| with errors:")
        for i, entry in enumerate(ranked):
            marker = " ← STRONGEST" if i < 3 else ""
            print(f"  {i+1}. {entry['block']:>16}: r={entry['corr']:+.4f}{marker}")

    # Per-block norms in error_records (if available)
    records = r.get('error_records', [])
    has_block_norms = any('z_block_norms' in rec for rec in records[:5])
    if has_block_norms and records:
        print(f"\nTop 5 worst errors — per-block Z norms:")
        for rec in records[:5]:
            bn = rec.get('z_block_norms', {})
            top_blocks = sorted(bn.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ', '.join(f"{n}={v:.2f}" for n, v in top_blocks)
            print(f"  [{rec['n_errors']} err] {rec['target'][:40]:40s} | z={rec['z_norm']:.1f} | top: {top_str}")

    # Trend analysis across multiple reports
    epochs_with_blocks = [(r2['epoch'], r2['z_diagnostics']['z_block_diagnostics'])
                          for r2 in reports
                          if 'z_block_diagnostics' in r2.get('z_diagnostics', {})]
    if len(epochs_with_blocks) >= 2:
        print(f"\nBlock norm trends ({len(epochs_with_blocks)} epochs):")
        first_ep, first_bd = epochs_with_blocks[0]
        last_ep, last_bd = epochs_with_blocks[-1]
        for block_name in ['gl', 'bcs', 'eliashberg', 'unconventional', 'structural',
                           'electronic', 'thermodynamic', 'compositional', 'cobordism',
                           'ratios', 'magpie', 'discovery']:
            if block_name in first_bd and block_name in last_bd:
                first_mean = first_bd[block_name]['overall']['mean']
                last_mean = last_bd[block_name]['overall']['mean']
                delta = last_mean - first_mean
                print(f"  {block_name:>16}: {first_mean:.3f} → {last_mean:.3f} (Δ{delta:+.3f})")


def analyze_family_diagnostics(reports):
    """V12.36b: Analyze family classification accuracy and its correlation with formula errors."""
    print("\n" + "=" * 80)
    print("FAMILY CLASSIFICATION DIAGNOSTICS")
    print("=" * 80)

    # Find latest report with family data
    for r in reversed(reports):
        zd = r.get('z_diagnostics', {})
        if 'family_diagnostics' in zd:
            break
    else:
        print("  No family_diagnostics found in any report (need V12.36b+ error reports)")
        return

    fam = zd['family_diagnostics']
    epoch = r['epoch']

    print(f"\nEpoch {epoch}: Coarse family accuracy = {fam['coarse_accuracy']:.1%}")
    print(f"Correlation (family wrong → formula errors): {fam['corr_family_wrong_vs_formula_errors']:.4f}")

    errors_by_fam = fam.get('errors_by_family', {})
    if errors_by_fam:
        print(f"\n{'Family':>12} | {'Samples':>7} | {'Fam Acc':>8} | {'Exact%':>7} | {'AvgErr':>7}")
        print("-" * 55)
        for fam_name, fdata in sorted(errors_by_fam.items(), key=lambda x: -x[1]['n_samples']):
            print(f"{fam_name:>12} | {fdata['n_samples']:>7} | "
                  f"{fdata['family_accuracy']:>8.1%} | "
                  f"{fdata['formula_exact_pct']:>7.1f} | "
                  f"{fdata['avg_formula_errors']:>7.2f}")

    # Per-sample family data in error records
    records = r.get('error_records', [])
    has_family = any(rec.get('family_true') is not None for rec in records[:10])
    if has_family and records:
        # Which families have the worst formula errors?
        from collections import defaultdict
        fam_errors = defaultdict(list)
        for rec in records:
            ft = rec.get('family_coarse_true')
            if ft:
                fam_errors[ft].append(rec['n_errors'])
        if fam_errors:
            print(f"\nError records by family (failed samples only):")
            print(f"{'Family':>12} | {'Failed':>6} | {'AvgErr':>7} | {'MaxErr':>6}")
            print("-" * 45)
            for fam_name, errs in sorted(fam_errors.items(), key=lambda x: -len(x[1])):
                print(f"{fam_name:>12} | {len(errs):>6} | {sum(errs)/len(errs):>7.1f} | {max(errs):>6}")

    # Trend across epochs (if multiple reports have family data)
    fam_epochs = [(r2['epoch'], r2['z_diagnostics']['family_diagnostics'])
                  for r2 in reports
                  if 'family_diagnostics' in r2.get('z_diagnostics', {})]
    if len(fam_epochs) >= 2:
        print(f"\nFamily accuracy trend ({len(fam_epochs)} epochs):")
        print(f"{'Epoch':>6} | {'Coarse Acc':>10} | {'Corr→Err':>9}")
        print("-" * 32)
        for ep, fd in fam_epochs:
            print(f"{ep:>6} | {fd['coarse_accuracy']:>10.1%} | {fd['corr_family_wrong_vs_formula_errors']:>+9.4f}")


def main():
    reports = []
    for epoch in LATEST_EPOCHS:
        r = load_report(epoch)
        if r:
            reports.append(r)
            print(f"Loaded epoch {epoch}: {r['exact_match_pct']:.2f}% exact match")
        else:
            print(f"WARNING: Missing epoch {epoch}")

    if not reports:
        print("No reports found!")
        return

    print(f"\nLoaded {len(reports)} reports: epochs {reports[0]['epoch']}-{reports[-1]['epoch']}")

    analyze_trends(reports)
    analyze_error_distribution(reports)
    analyze_z_norm_quartiles(reports)
    analyze_seq_len_buckets(reports)
    analyze_correlations(reports)
    analyze_sc_vs_nonsc(reports)
    analyze_tc_ranges(reports)
    analyze_physics_z_blocks(reports)
    analyze_family_diagnostics(reports)
    analyze_error_patterns(reports)
    analyze_fraction_precision_deep(reports)
    analyze_element_level(reports)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    last = reports[-1]
    first = reports[0]

    improvement = last['exact_match_pct'] - first['exact_match_pct']
    z = last['z_diagnostics']

    print(f"\nOverall: {first['exact_match_pct']:.2f}% -> {last['exact_match_pct']:.2f}% exact match over {last['epoch'] - first['epoch']} epochs ({improvement:+.2f}%)")
    print(f"Total samples evaluated: {last['total_samples']}")
    print(f"Tc R² (normalized): {z['tc_r2']:.6f}")
    if z.get('tc_r2_kelvin') is not None:
        print(f"Tc R² (Kelvin):     {z['tc_r2_kelvin']:.6f}")
        print(f"Tc MAE (Kelvin):    {z['tc_mae_kelvin_overall']:.2f}K")
    print(f"Strongest error predictors: seq_len (r={z['corr_seq_len_vs_errors']:.3f}), tc_true (r={z['corr_tc_true_vs_errors']:.3f}), z_norm (r={z['corr_z_norm_vs_errors']:.3f})")

if __name__ == "__main__":
    main()
